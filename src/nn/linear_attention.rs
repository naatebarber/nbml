use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, Axis, concatenate, s, stack,
};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToIntermediates, ToParams},
};

const NORM_EPS: f32 = 1e-6;

// Feature map φ applied to queries and keys before the state update / readout.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureMap {
    // φ(x) = x. Scores can be negative, so this cannot be combined with normalization.
    #[default]
    Identity,
    // φ(x) = elu(x) + 1 (Katharopoulos et al. 2020). Strictly positive.
    EluPlusOne,
}

impl FeatureMap {
    #[inline]
    pub fn phi(self, x: f32) -> f32 {
        match self {
            Self::Identity => x,
            Self::EluPlusOne => {
                if x > 0. {
                    x + 1.
                } else {
                    x.exp()
                }
            }
        }
    }

    #[inline]
    pub fn d_phi(self, x: f32) -> f32 {
        match self {
            Self::Identity => 1.,
            Self::EluPlusOne => {
                if x > 0. {
                    1.
                } else {
                    x.exp()
                }
            }
        }
    }

    /// Whether φ(x) > 0 for all x, which is required for the φ(q)ᵀz denominator to be safe.
    pub fn is_positive(self) -> bool {
        match self {
            Self::Identity => false,
            Self::EluPlusOne => true,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct LinearAttentionCache {
    pub x_2d: Array2<f32>,
    /// query / key projections *before* the feature map
    pub q: Array3<f32>,
    pub k: Array3<f32>,
    pub v: Array3<f32>,
    /// (seq_len + 1, B, d_k, d_head + normalize). When normalizing, the last column holds z.
    pub states: Array4<f32>,
    pub attn_2d: Array2<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct LinearAttentionGrads {
    pub d_w_qkv: Array2<f32>,
    pub d_b_qkv: Array1<f32>,
    pub d_w_o: Array2<f32>,
    pub d_b_o: Array1<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinearAttention {
    pub d_in: usize,
    pub d_head: usize,

    #[serde(default)]
    pub feature_map: FeatureMap,
    /// Katharopoulos normalization: o = φ(q)ᵀS / φ(q)ᵀz. Tracked as an augmented
    /// state S' = Σ φ(k)[v; 1]ᵀ, whose last column is z.
    #[serde(default)]
    pub normalize: bool,

    pub w_qkv: Array2<f32>,
    pub b_qkv: Array1<f32>,
    pub w_o: Array2<f32>,
    pub b_o: Array1<f32>,

    #[serde(skip)]
    pub cache: LinearAttentionCache,
    #[serde(skip)]
    pub grads: LinearAttentionGrads,
}

impl LinearAttention {
    /// Unnormalized linear attention with the identity feature map.
    pub fn new(d_in: usize, d_head: usize) -> Self {
        Self::new_with_feature_map(d_in, d_head, FeatureMap::Identity, false)
    }

    /// Vanilla Katharopoulos linear attention: φ = elu + 1, normalized by φ(q)ᵀz.
    pub fn katharopoulos(d_in: usize, d_head: usize) -> Self {
        Self::new_with_feature_map(d_in, d_head, FeatureMap::EluPlusOne, true)
    }

    pub fn new_with_feature_map(
        d_in: usize,
        d_head: usize,
        feature_map: FeatureMap,
        normalize: bool,
    ) -> Self {
        assert!(
            !normalize || feature_map.is_positive(),
            "normalization requires a strictly positive feature map, got {feature_map:?}"
        );

        Self {
            d_in,
            d_head,
            feature_map,
            normalize,

            w_qkv: f::xavier_normal((d_in, 3 * d_head)),
            b_qkv: Array1::zeros(3 * d_head),
            w_o: f::xavier_normal((d_head, d_in)),
            b_o: Array1::zeros(d_in),

            cache: LinearAttentionCache::default(),
            grads: LinearAttentionGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array3<f32>, grad: bool) -> Array3<f32> {
        let (batch_size, seq_len, features) = x.dim();

        let x_2d = x
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();

        let qkv_2d = x_2d.dot(&self.w_qkv) + &self.b_qkv;
        let qkv = qkv_2d
            .into_shape_clone((batch_size, seq_len, 3, self.d_head))
            .unwrap()
            .permuted_axes([2, 0, 1, 3])
            .to_owned();

        let q_pre = qkv.slice(s![0, .., .., ..]);
        let k_pre = qkv.slice(s![1, .., .., ..]);
        let v = qkv.slice(s![2, .., .., ..]);

        let q = self.apply_phi(q_pre);
        let k = self.apply_phi(k_pre);
        let v_s = self.augment_v(v);
        let d_s = self.state_width();

        if grad {
            self.cache.x_2d = x_2d;
            self.cache.q = q_pre.to_owned();
            self.cache.k = k_pre.to_owned();
            self.cache.v = v.to_owned();
            self.cache.states = Array4::zeros((seq_len + 1, batch_size, self.d_head, d_s));
        }

        let mut state = self.new_state(batch_size);
        let mut attn = Array3::zeros((batch_size, seq_len, self.d_head));

        for t in 0..seq_len {
            let q_t = q.slice(s![.., t, ..]);
            let k_t = k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = v_s.slice(s![.., t, ..]); // (B, d_s)

            Self::write(&mut state, k_t, v_t);
            let attn_t = self.read(q_t, &state);

            attn.slice_mut(s![.., t, ..]).assign(&attn_t);

            if grad {
                self.cache
                    .states
                    .slice_mut(s![t + 1, .., .., ..])
                    .assign(&state);
            }
        }

        let attn_2d = attn
            .into_shape_clone((batch_size * seq_len, self.d_head))
            .unwrap();
        let output_2d = attn_2d.dot(&self.w_o) + &self.b_o;
        let output = output_2d
            .into_shape_clone((batch_size, seq_len, self.d_in))
            .unwrap();

        if grad {
            self.cache.attn_2d = attn_2d;
        }

        output
    }

    pub fn backward(&mut self, d_loss: Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, features) = d_loss.dim();
        let (_, _, d_k, d_s) = self.cache.states.dim();

        if self.grads.d_w_o.dim() == (0, 0) {
            self.grads.d_w_o = Array2::zeros(self.w_o.dim())
        }

        if self.grads.d_b_o.dim() == 0 {
            self.grads.d_b_o = Array1::zeros(self.b_o.dim())
        }

        if self.grads.d_w_qkv.dim() == (0, 0) {
            self.grads.d_w_qkv = Array2::zeros(self.w_qkv.dim());
        }

        if self.grads.d_b_qkv.dim() == 0 {
            self.grads.d_b_qkv = Array1::zeros(self.b_qkv.dim());
        }

        let d_loss_2d = d_loss
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();
        self.grads.d_w_o += &(self.cache.attn_2d.t().dot(&d_loss_2d));
        self.grads.d_b_o += &d_loss_2d.sum_axis(Axis(0));

        let d_loss_2d = d_loss_2d.dot(&self.w_o.t());
        let d_loss = d_loss_2d
            .into_shape_clone((batch_size, seq_len, self.d_head))
            .unwrap();

        let q = self.apply_phi(self.cache.q.view());
        let k = self.apply_phi(self.cache.k.view());
        let v_s = self.augment_v(self.cache.v.view());

        let mut d_loss_q = Array3::zeros((batch_size, seq_len, d_k));
        let mut d_loss_k = Array3::zeros((batch_size, seq_len, d_k));
        let mut d_loss_v = Array3::zeros((batch_size, seq_len, self.d_head));

        // gradient w.r.t. the (possibly augmented) state, (B, d_k, d_s)
        let mut d_loss_state = Array3::zeros((batch_size, d_k, d_s));

        for t in (0..seq_len).rev() {
            let d_loss_t = d_loss.slice(s![.., t, ..]); // (B, d_v)
            let state_t = self.cache.states.slice(s![t + 1, .., .., ..]); // (B, d_k, d_s)
            let q_t = q.slice(s![.., t, ..]); // (B, d_k)
            let k_t = k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = v_s.slice(s![.., t, ..]); // (B, d_s)

            // gradient w.r.t. the raw readout r = φ(q)ᵀS', (B, d_s)
            let d_loss_r_t = if self.normalize {
                // r = [num | den], o = num / (den + eps)
                let r_t = (&q_t.insert_axis(Axis(2)) * &state_t).sum_axis(Axis(1));
                let num = r_t.slice(s![.., ..self.d_head]);
                let den = r_t.slice(s![.., self.d_head..]).mapv(|d| d + NORM_EPS); // (B, 1)
                let o = &num / &den;

                let d_num = &d_loss_t / &den;
                let d_den = -(&d_loss_t * &o).sum_axis(Axis(1)).insert_axis(Axis(1)) / &den;

                concatenate(Axis(1), &[d_num.view(), d_den.view()]).unwrap()
            } else {
                d_loss_t.to_owned()
            };

            // (B, d_s) -> (B, 1, d_s)
            let d_loss_r_t = d_loss_r_t.insert_axis(Axis(1));

            // (B, d_k, d_s) * (B, 1, d_s) -> sum over d_s -> (B, d_k)
            let d_loss_q_t = (&state_t * &d_loss_r_t).sum_axis(Axis(2));
            d_loss_q.slice_mut(s![.., t, ..]).assign(&d_loss_q_t);

            // (B, d_k, 1) * (B, 1, d_s) -> (B, d_k, d_s)
            d_loss_state += &(&q_t.insert_axis(Axis(2)) * &d_loss_r_t);

            // (B, 1, d_s) * (B, d_k, d_s) -> sum over d_s -> (B, d_k)
            let d_loss_k_t = (&v_t.insert_axis(Axis(1)) * &d_loss_state).sum_axis(Axis(2));
            d_loss_k.slice_mut(s![.., t, ..]).assign(&d_loss_k_t);

            // (B, d_k, 1) * (B, d_k, d_s) -> sum over d_k -> (B, d_s)
            // the augmented ones column is a constant, so drop its gradient
            let d_loss_v_t = (&k_t.insert_axis(Axis(2)) * &d_loss_state).sum_axis(Axis(1));
            d_loss_v
                .slice_mut(s![.., t, ..])
                .assign(&d_loss_v_t.slice(s![.., ..self.d_head]));
        }

        // back through the feature map
        let fm = self.feature_map;
        if fm != FeatureMap::Identity {
            d_loss_q *= &self.cache.q.mapv(|x| fm.d_phi(x));
            d_loss_k *= &self.cache.k.mapv(|x| fm.d_phi(x));
        }

        // (3, B, S, D) -> (B, S, 3, D)
        let d_loss_qkv = stack![Axis(0), d_loss_q.view(), d_loss_k.view(), d_loss_v.view()]
            .permuted_axes([1, 2, 0, 3])
            .to_owned()
            .into_shape_clone((batch_size * seq_len, 3 * self.d_head))
            .unwrap();

        self.grads.d_w_qkv += &(self.cache.x_2d.t().dot(&d_loss_qkv));
        self.grads.d_b_qkv += &d_loss_qkv.sum_axis(Axis(0));

        let d_x_2d = d_loss_qkv.dot(&self.w_qkv.t());

        let d_x = d_x_2d
            .into_shape_clone((batch_size, seq_len, self.d_in))
            .unwrap();

        d_x
    }

    /// Width of the state's value axis: d_head, plus one column for z when normalizing.
    pub fn state_width(&self) -> usize {
        self.d_head + self.normalize as usize
    }

    /// Zero state for `step`, shape (B, d_head, state_width()).
    pub fn new_state(&self, batch_size: usize) -> Array3<f32> {
        Array3::zeros((batch_size, self.d_head, self.state_width()))
    }

    fn apply_phi(&self, x: ArrayView3<f32>) -> Array3<f32> {
        let fm = self.feature_map;
        match fm {
            FeatureMap::Identity => x.to_owned(),
            _ => x.mapv(|v| fm.phi(v)),
        }
    }

    /// [v; 1] along the feature axis when normalizing, so the state also accumulates z.
    fn augment_v(&self, v: ArrayView3<f32>) -> Array3<f32> {
        if self.normalize {
            let (b, t, _) = v.dim();
            let ones = Array3::ones((b, t, 1));
            concatenate(Axis(2), &[v, ones.view()]).unwrap()
        } else {
            v.to_owned()
        }
    }

    /// S += φ(k) vᵀ
    fn write(state: &mut Array3<f32>, k: ArrayView2<f32>, v: ArrayView2<f32>) {
        // (B, d_k, 1) * (B, 1, d_s) -> (B, d_k, d_s)
        let k_inner = k.insert_axis(Axis(2));
        let v_outer = v.insert_axis(Axis(1));
        *state = &(*state) + &k_inner * &v_outer;
    }

    /// φ(q)ᵀS, divided by φ(q)ᵀz when normalizing. Returns (B, d_head).
    fn read(&self, q: ArrayView2<f32>, state: &Array3<f32>) -> Array2<f32> {
        // (B, d_k, 1) * (B, d_k, d_s) -> sum over d_k -> (B, d_s)
        let q_inner = q.insert_axis(Axis(2));
        let r: Array2<f32> = (&q_inner * state).sum_axis(Axis(1));

        if self.normalize {
            let num = r.slice(s![.., ..self.d_head]);
            let den = r.slice(s![.., self.d_head..]).mapv(|d| d + NORM_EPS);
            &num / &den
        } else {
            r
        }
    }

    /// Single recurrent step. `state` has shape (B, d_head, state_width()); see `new_state`.
    pub fn step(&self, x: &Array2<f32>, state: &mut Array3<f32>) -> Array2<f32> {
        let (batch_size, _) = x.dim();

        let qkv = x.dot(&self.w_qkv) + &self.b_qkv;
        let qkv = qkv
            .into_shape_clone((batch_size, 1, 3, self.d_head))
            .unwrap()
            .permuted_axes([2, 0, 1, 3])
            .to_owned();

        let q = self.apply_phi(qkv.slice(s![0, .., .., ..]));
        let k = self.apply_phi(qkv.slice(s![1, .., .., ..]));
        let v = self.augment_v(qkv.slice(s![2, .., .., ..]));

        Self::write(state, k.slice(s![.., 0, ..]), v.slice(s![.., 0, ..]));
        let attn = self.read(q.slice(s![.., 0, ..]), state);

        attn.dot(&self.w_o) + &self.b_o
    }
}

impl ToParams for LinearAttention {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_qkv).with_grad(&mut self.grads.d_w_qkv),
            Param::new(&mut self.b_qkv).with_grad(&mut self.grads.d_b_qkv),
            Param::new(&mut self.w_o).with_grad(&mut self.grads.d_w_o),
            Param::new(&mut self.b_o).with_grad(&mut self.grads.d_b_o),
        ]
    }
}

impl ToIntermediates for LinearAttention {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![
            &mut self.cache.x_2d,
            &mut self.cache.q,
            &mut self.cache.k,
            &mut self.cache.v,
            &mut self.cache.states,
            &mut self.cache.attn_2d,
        ]
    }
}

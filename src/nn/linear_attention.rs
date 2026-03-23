use ndarray::{Array1, Array2, Array3, Array4, Axis, s, stack};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct LinearAttentionCache {
    pub x_2d: Array2<f32>,
    pub q: Array3<f32>,
    pub k: Array3<f32>,
    pub v: Array3<f32>,
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
    pub fn new(d_in: usize, d_head: usize) -> Self {
        Self {
            d_in,
            d_head,

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

        let q = qkv.slice(s![0, .., .., ..]);
        let k = qkv.slice(s![1, .., .., ..]);
        let v = qkv.slice(s![2, .., .., ..]);

        if grad {
            self.cache.x_2d = x_2d;
            self.cache.q = q.to_owned();
            self.cache.k = k.to_owned();
            self.cache.v = v.to_owned();
            self.cache.states = Array4::zeros((seq_len + 1, batch_size, self.d_head, self.d_head));
        }

        let mut state = Array3::zeros((batch_size, self.d_head, self.d_head));
        let mut attn = Array3::zeros((batch_size, seq_len, self.d_head));

        for t in 0..seq_len {
            let q_t = q.slice(s![.., t, ..]);
            let k_t = k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = v.slice(s![.., t, ..]); // (B, d_v)

            // leverage broadcasting trick
            // (B, d_k, 1) * (B, 1, d_v) -> (B, d_k, d_v)
            let k_t_inner = k_t.insert_axis(Axis(2));
            let v_t_outer = v_t.insert_axis(Axis(1));
            state = &state + &k_t_inner * &v_t_outer;

            // (B, d_q) -> (B, d_q, 1)
            let q_t_inner = q_t.insert_axis(Axis(2));
            let attn_t = (&q_t_inner * &state).sum_axis(Axis(1));

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
        let (_, _, d_k, d_v) = self.cache.states.dim();

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

        let mut d_loss_q = Array3::zeros((batch_size, seq_len, self.d_head));
        let mut d_loss_k = Array3::zeros((batch_size, seq_len, d_k));
        let mut d_loss_v = Array3::zeros((batch_size, seq_len, d_v));

        let mut d_loss_state = Array3::zeros((batch_size, d_k, d_v));

        for t in (0..seq_len).rev() {
            let d_loss_t = d_loss.slice(s![.., t, ..]); // (B, d_v)
            let state_t = self.cache.states.slice(s![t + 1, .., .., ..]); // (B, d_k, d_v)
            let q_t = self.cache.q.slice(s![.., t, ..]); // (B, d_q)
            let k_t = self.cache.k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = self.cache.v.slice(s![.., t, ..]); // (B, d_v)

            // (B, d_v) -> (B, 1, d_v)
            let d_loss_t = d_loss_t.to_owned().insert_axis(Axis(1));

            // state_t * d_loss_t
            // (B, d_k, d_v) * (B, 1, d_v) -> (B, d_k, d_v)
            let d_loss_q_t = (&state_t * &d_loss_t).sum_axis(Axis(2));
            d_loss_q.slice_mut(s![.., t, ..]).assign(&d_loss_q_t);

            // q_t * d_loss_t
            // (B, d_k, 1) * (B, 1, d_v) -> (B, d_k, d_v)
            d_loss_state += &(&q_t.insert_axis(Axis(2)) * &d_loss_t);

            // (B, d_v) -> (B, 1, d_v)
            // (B, 1, d_v) * (B, d_k, d_v) -> (B, d_k, sum(d_v)) -> (B, d_k)
            let d_loss_k_t =
                (&v_t.to_owned().insert_axis(Axis(1)) * &d_loss_state).sum_axis(Axis(2));
            d_loss_k.slice_mut(s![.., t, ..]).assign(&d_loss_k_t);

            // (B, d_k) -> (B, d_k, 1)
            // (B, d_k, 1) * (B, d_k, d_v) -> (B, sum(d_k), d_v) -> (B, d_v)
            let d_loss_v_t =
                (&k_t.to_owned().insert_axis(Axis(2)) * &d_loss_state).sum_axis(Axis(1));
            d_loss_v.slice_mut(s![.., t, ..]).assign(&d_loss_v_t);
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

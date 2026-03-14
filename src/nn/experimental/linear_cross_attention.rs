use ndarray::{Array2, Array3, Array4, Axis, s};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct LinearCrossAttentionCache {
    pub x_q_2d: Array2<f64>,
    pub x_kv_2d: Array2<f64>,
    pub q: Array3<f64>,
    pub k: Array3<f64>,
    pub v: Array3<f64>,
    pub states: Array4<f64>,
}

#[derive(Default, Debug, Clone)]
pub struct LinearCrossAttentionGrads {
    pub d_w_q: Array2<f64>,
    pub d_w_k: Array2<f64>,
    pub d_w_v: Array2<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinearCrossAttention {
    pub d_in: usize,
    pub d_head: usize,

    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,

    #[serde(skip)]
    pub cache: LinearCrossAttentionCache,
    #[serde(skip)]
    pub grads: LinearCrossAttentionGrads,
}

impl LinearCrossAttention {
    pub fn new(d_in: usize, d_head: usize) -> Self {
        Self {
            d_in,
            d_head,

            w_q: f::xavier_normal((d_in, d_head)),
            w_k: f::xavier_normal((d_in, d_head)),
            w_v: f::xavier_normal((d_in, d_head)),

            cache: LinearCrossAttentionCache::default(),
            grads: LinearCrossAttentionGrads::default(),
        }
    }

    pub fn forward(&mut self, x_q: Array3<f64>, x_kv: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len_q, features) = x_q.dim();
        let (_, seq_len_kv, _) = x_kv.dim();

        let x_q_2d = x_q
            .into_shape_clone((batch_size * seq_len_q, features))
            .unwrap();
        let x_kv_2d = x_kv
            .into_shape_clone((batch_size * seq_len_kv, features))
            .unwrap();

        let q_2d = x_q_2d.dot(&self.w_q);
        let k_2d = x_kv_2d.dot(&self.w_k);
        let v_2d = x_kv_2d.dot(&self.w_v);

        let q = q_2d
            .into_shape_clone((batch_size, seq_len_q, self.d_head))
            .unwrap();
        let k = k_2d
            .into_shape_clone((batch_size, seq_len_kv, self.d_head))
            .unwrap();
        let v = v_2d
            .into_shape_clone((batch_size, seq_len_kv, self.d_head))
            .unwrap();

        if grad {
            self.cache.x_q_2d = x_q_2d;
            self.cache.x_kv_2d = x_kv_2d;
            self.cache.q = q.clone();
            self.cache.k = k.clone();
            self.cache.v = v.clone();
            self.cache.states =
                Array4::zeros((seq_len_kv + 1, batch_size, self.d_head, self.d_head));
        }

        let mut state = Array3::zeros((batch_size, self.d_head, self.d_head));
        let mut outputs = Array3::zeros((batch_size, seq_len_q, self.d_head));

        for t in 0..seq_len_kv {
            let k_t = k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = v.slice(s![.., t, ..]); // (B, d_v)

            // leverage broadcasting trick
            // (B, d_k, 1) * (B, 1, d_v) -> (B, d_k, d_v)
            let k_t_inner = k_t.insert_axis(Axis(2));
            let v_t_outer = v_t.insert_axis(Axis(1));
            state = &state + &k_t_inner * &v_t_outer;

            if grad {
                self.cache
                    .states
                    .slice_mut(s![t + 1, .., .., ..])
                    .assign(&state);
            }
        }

        for t in 0..seq_len_q {
            let q_t = q.slice(s![.., t, ..]); // (B, d_q)

            // leverage broadcasting trick again
            // (B, d_k, 1) * (B, d_k, d_v) -> (B, d_k, d_v)
            // sum axis 1
            // (B, d_v)
            let q_t_inner = q_t.insert_axis(Axis(2));
            let output = (&q_t_inner * &state).sum_axis(Axis(1));
            outputs.slice_mut(s![.., t, ..]).assign(&output);
        }

        outputs
    }

    pub fn backward(&mut self, d_loss: Array3<f64>) -> (Array3<f64>, Array3<f64>) {
        let (batch_size, seq_len_q, features) = d_loss.dim();
        let (seq_len_kv_plus1, _, d_k, d_v) = self.cache.states.dim();
        let seq_len_kv = seq_len_kv_plus1 - 1;

        if self.grads.d_w_q.dim() == (0, 0) {
            self.grads.d_w_q = Array2::zeros(self.w_q.dim());
        }

        if self.grads.d_w_k.dim() == (0, 0) {
            self.grads.d_w_k = Array2::zeros(self.w_k.dim());
        }

        if self.grads.d_w_v.dim() == (0, 0) {
            self.grads.d_w_v = Array2::zeros(self.w_v.dim());
        }

        let mut d_loss_q = Array3::zeros((batch_size, seq_len_q, features));
        let mut d_loss_state = Array3::zeros((batch_size, d_k, d_v));

        let state = self.cache.states.slice(s![-1, .., .., ..]);

        for t in (0..seq_len_q).rev() {
            let d_loss_t = d_loss.slice(s![.., t, ..]); // (B, d_v)
            let q_t = self.cache.q.slice(s![.., t, ..]); // (B, d_k)

            // (B, d_v) -> (B, 1, d_v)
            let d_loss_t = d_loss_t.to_owned().insert_axis(Axis(1));

            // q_t * d_loss_t
            // (B, d_k, 1) * (B, 1, d_v) -> (B, d_k, d_v)
            d_loss_state += &(&q_t.insert_axis(Axis(2)) * &d_loss_t);

            // state_t * d_loss_t
            // (B, d_k, d_v) * (B, 1, d_v) -> (B, d_k, d_v)
            let d_loss_q_t = (&state * &d_loss_t).sum_axis(Axis(2));
            d_loss_q.slice_mut(s![.., t, ..]).assign(&d_loss_q_t);
        }

        let mut d_loss_k = Array3::zeros((batch_size, seq_len_kv, d_k));
        let mut d_loss_v = Array3::zeros((batch_size, seq_len_kv, d_v));

        for t in (0..seq_len_kv).rev() {
            let k_t = self.cache.k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = self.cache.v.slice(s![.., t, ..]); // (B, d_v)

            // (B, d_v) -> (B, 1, d_v)
            // (B, 1, d_v) * (B, d_k, d_v) -> (B, d_k, sum(d_v)) -> (B, d_k)
            let d_loss_k_t =
                (&v_t.to_owned().insert_axis(Axis(1)) * &d_loss_state).sum_axis(Axis(2));

            // (B, d_k) -> (B, d_k, 1)
            // (B, d_k, 1) * (B, d_k, d_v) -> (B, sum(d_k), d_v) -> (B, d_v)
            let d_loss_v_t =
                (&k_t.to_owned().insert_axis(Axis(2)) * &d_loss_state).sum_axis(Axis(1));

            d_loss_k.slice_mut(s![.., t, ..]).assign(&d_loss_k_t);
            d_loss_v.slice_mut(s![.., t, ..]).assign(&d_loss_v_t);
        }

        let d_loss_q_2d = d_loss_q
            .into_shape_clone((batch_size * seq_len_q, features))
            .unwrap();
        let d_loss_k_2d = d_loss_k
            .into_shape_clone((batch_size * seq_len_kv, features))
            .unwrap();
        let d_loss_v_2d = d_loss_v
            .into_shape_clone((batch_size * seq_len_kv, features))
            .unwrap();

        self.grads.d_w_q += &(self.cache.x_q_2d.t().dot(&d_loss_q_2d));
        self.grads.d_w_k += &(self.cache.x_kv_2d.t().dot(&d_loss_k_2d));
        self.grads.d_w_v += &(self.cache.x_kv_2d.t().dot(&d_loss_v_2d));

        let d_x_q_2d = d_loss_q_2d.dot(&self.w_q.t());
        let d_x_k_2d = d_loss_k_2d.dot(&self.w_k.t());
        let d_x_v_2d = d_loss_v_2d.dot(&self.w_v.t());

        let d_x_kv_2d = &d_x_k_2d + &d_x_v_2d;

        let d_x_q = d_x_q_2d
            .into_shape_clone((batch_size, seq_len_q, features))
            .unwrap();
        let d_x_kv = d_x_kv_2d
            .into_shape_clone((batch_size, seq_len_kv, features))
            .unwrap();

        (d_x_q, d_x_kv)
    }
}

impl ToParams for LinearCrossAttention {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_q).with_grad(&mut self.grads.d_w_q),
            Param::new(&mut self.w_k).with_grad(&mut self.grads.d_w_k),
            Param::new(&mut self.w_v).with_grad(&mut self.grads.d_w_v),
        ]
    }
}

impl ToIntermediates for LinearCrossAttention {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![
            &mut self.cache.x_q_2d,
            &mut self.cache.x_kv_2d,
            &mut self.cache.q,
            &mut self.cache.k,
            &mut self.cache.v,
            &mut self.cache.states,
        ]
    }
}

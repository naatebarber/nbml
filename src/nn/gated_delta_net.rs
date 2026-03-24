use ndarray::{Array1, Array2, Array3, Array4, Axis, concatenate, s, stack};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct GatedDeltaNetCache {
    pub x_2d: Array2<f32>,
    pub q_preactivations: Array2<f32>,
    pub k_preactivations: Array2<f32>,
    pub q: Array3<f32>,
    pub k: Array3<f32>,
    pub v: Array3<f32>,
    pub ab_preactivations: Array2<f32>,
    pub ab: Array3<f32>,
    pub rs: Array3<f32>,
    pub updates: Array4<f32>,
    pub states: Array4<f32>,
    pub attn_2d: Array2<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct GatedDeltaNetGrads {
    pub d_w_qkv: Array2<f32>,
    pub d_b_qkv: Array1<f32>,
    pub d_w_ab: Array2<f32>,
    pub d_b_ab: Array1<f32>,
    pub d_w_o: Array2<f32>,
    pub d_b_o: Array1<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GatedDeltaNet {
    pub d_in: usize,
    pub d_head: usize,
    pub d_out: usize,

    pub w_qkv: Array2<f32>,
    pub b_qkv: Array1<f32>,
    pub w_ab: Array2<f32>,
    pub b_ab: Array1<f32>,
    pub w_o: Array2<f32>,
    pub b_o: Array1<f32>,

    #[serde(skip)]
    pub cache: GatedDeltaNetCache,
    #[serde(skip)]
    pub grads: GatedDeltaNetGrads,
}

impl GatedDeltaNet {
    pub fn new(d_in: usize, d_head: usize, d_out: usize) -> Self {
        Self {
            d_in,
            d_head,
            d_out,

            w_qkv: f::xavier_normal((d_in, 3 * d_head)),
            b_qkv: Array1::zeros(3 * d_head),
            w_ab: f::xavier_normal((d_in, 2)),
            b_ab: Array1::zeros(2),
            w_o: f::xavier_normal((d_head, d_out)),
            b_o: Array1::zeros(d_out),

            cache: GatedDeltaNetCache::default(),
            grads: GatedDeltaNetGrads::default(),
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

        let q_preactivations = qkv.slice(s![0, .., .., ..]).to_owned();
        let k_preactivations = qkv.slice(s![1, .., .., ..]).to_owned();
        let v = qkv.slice(s![2, .., .., ..]);

        let q_preactivations_2d = q_preactivations
            .into_shape_clone((batch_size * seq_len, self.d_head))
            .unwrap();
        let k_preactivations_2d = k_preactivations
            .into_shape_clone((batch_size * seq_len, self.d_head))
            .unwrap();

        let q = f::l2_norm(&q_preactivations_2d)
            .into_shape_clone((batch_size, seq_len, self.d_head))
            .unwrap();
        let k = f::l2_norm(&k_preactivations_2d)
            .into_shape_clone((batch_size, seq_len, self.d_head))
            .unwrap();

        let ab_preactivations = x_2d.dot(&self.w_ab) + &self.b_ab;
        let ab_2d = f::sigmoid(&ab_preactivations);
        let ab = ab_2d.into_shape_clone((batch_size, seq_len, 2)).unwrap();

        if grad {
            self.cache.x_2d = x_2d;
            self.cache.q_preactivations = q_preactivations_2d;
            self.cache.k_preactivations = k_preactivations_2d;
            self.cache.q = q.to_owned();
            self.cache.k = k.to_owned();
            self.cache.v = v.to_owned();
            self.cache.ab_preactivations = ab_preactivations;
            self.cache.ab = ab.clone();
            self.cache.rs = Array3::zeros((seq_len, batch_size, self.d_head));
            self.cache.updates = Array4::zeros((seq_len, batch_size, self.d_head, self.d_head));
            self.cache.states = Array4::zeros((seq_len + 1, batch_size, self.d_head, self.d_head));
        }

        let mut state = Array3::zeros((batch_size, self.d_head, self.d_head));
        let mut attn = Array3::zeros((batch_size, seq_len, self.d_head));

        for t in 0..seq_len {
            let q_t = q.slice(s![.., t, ..]);
            let k_t = k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = v.slice(s![.., t, ..]); // (B, d_v)
            let alpha_t = ab.slice(s![.., t, 0..1]).insert_axis(Axis(2));
            let beta_t = ab.slice(s![.., t, 1..2]).insert_axis(Axis(2));

            let alpha_t = alpha_t.broadcast(state.dim()).unwrap();
            let beta_t = beta_t.broadcast(state.dim()).unwrap();

            let r = (&state * &k_t.insert_axis(Axis(2))).sum_axis(Axis(1));
            let error = &v_t - &r;
            let update = &k_t.insert_axis(Axis(2)) * &error.insert_axis(Axis(1));

            state = &alpha_t * &state + &beta_t * &update;

            // (B, d_q) -> (B, d_q, 1)
            let q_t_inner = q_t.insert_axis(Axis(2));
            let attn_t = (&q_t_inner * &state).sum_axis(Axis(1));

            attn.slice_mut(s![.., t, ..]).assign(&attn_t);

            if grad {
                self.cache.rs.slice_mut(s![t, .., ..]).assign(&r);
                self.cache
                    .updates
                    .slice_mut(s![t, .., .., ..])
                    .assign(&update);
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
            .into_shape_clone((batch_size, seq_len, self.d_out))
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

        if self.grads.d_w_ab.dim() == (0, 0) {
            self.grads.d_w_ab = Array2::zeros(self.w_ab.dim());
        }

        if self.grads.d_b_ab.dim() == 0 {
            self.grads.d_b_ab = Array1::zeros(self.b_ab.dim());
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
        let mut d_loss_beta = Array3::zeros((batch_size, seq_len, 1));
        let mut d_loss_alpha = Array3::zeros((batch_size, seq_len, 1));

        let mut d_resid = Array3::zeros((batch_size, d_k, d_v));

        for t in (0..seq_len).rev() {
            let d_loss_t = d_loss.slice(s![.., t, ..]); // (B, d_v)
            let state_prev = self.cache.states.slice(s![t, .., .., ..]);
            let state_t = self.cache.states.slice(s![t + 1, .., .., ..]); // (B, d_k, d_v)
            let q_t = self.cache.q.slice(s![.., t, ..]); // (B, d_q)
            let k_t = self.cache.k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = self.cache.v.slice(s![.., t, ..]); // (B, d_v)
            let alpha_t = self.cache.ab.slice(s![.., t, 0..1]).insert_axis(Axis(2)); // (B, 1, 1)
            let beta_t = self.cache.ab.slice(s![.., t, 1..2]).insert_axis(Axis(2)); // (B, 1, 1)

            let r_t = self.cache.rs.slice(s![t, .., ..]);
            let update_t = self.cache.updates.slice(s![t, .., .., ..]);

            let alpha_t = alpha_t.broadcast(state_prev.dim()).unwrap();
            let beta_t = beta_t.broadcast(state_prev.dim()).unwrap();

            let d_q = (&state_t * &d_loss_t.insert_axis(Axis(1))).sum_axis(Axis(2));
            d_loss_q.slice_mut(s![.., t, ..]).assign(&d_q);

            let d_state = &q_t.insert_axis(Axis(2)) * &d_loss_t.insert_axis(Axis(1)) + &d_resid;

            // r = sum(S * k)
            // error = v_t - r
            // update = k.T * error
            // S = S_prev + beta * (update)

            // get d_beta
            let d_beta = &update_t * &d_state;
            let d_beta = d_beta
                .sum_axis(Axis(2))
                .sum_axis(Axis(1))
                .insert_axis(Axis(1));
            d_loss_beta.slice_mut(s![.., t, ..]).assign(&d_beta);

            // get d_update
            let d_update = &beta_t * &d_state;

            // get d_k for update
            let error = &v_t - &r_t;
            let d_k_update = (&d_update * &error.insert_axis(Axis(1))).sum_axis(Axis(2));

            // get d_error
            let d_error = (&k_t.insert_axis(Axis(2)) * &d_update).sum_axis(Axis(1));

            // get d_r, d_v_t
            let d_r = -1. * &d_error;
            let d_v = &d_error;
            d_loss_v.slice_mut(s![.., t, ..]).assign(&d_v);

            // get d_state_prev, d_k
            let d_k_r = (&state_prev * &d_r.clone().insert_axis(Axis(1))).sum_axis(Axis(2));
            let d_state_prev_r = &k_t.insert_axis(Axis(2)) * &d_r.insert_axis(Axis(1));

            // combine gradients for k
            let d_k = &d_k_r + &d_k_update;
            d_loss_k.slice_mut(s![.., t, ..]).assign(&d_k);

            // get d_alpha
            let d_alpha = &d_state * &state_prev;
            let d_alpha = d_alpha
                .sum_axis(Axis(2))
                .sum_axis(Axis(1))
                .insert_axis(Axis(1));
            d_loss_alpha.slice_mut(s![.., t, ..]).assign(&d_alpha);

            // combine gradients for state
            let d_state_prev = &alpha_t * &d_state + &d_state_prev_r;

            d_resid = d_state_prev;
        }

        // update beta weights

        let d_loss_ab = concatenate![Axis(2), d_loss_alpha.view(), d_loss_beta.view()];

        let d_loss_ab_2d = d_loss_ab
            .into_shape_clone((batch_size * seq_len, 2))
            .unwrap();
        let d_ab_dz = &d_loss_ab_2d * f::d_sigmoid(&self.cache.ab_preactivations);

        self.grads.d_w_ab += &(self.cache.x_2d.t().dot(&d_ab_dz));
        self.grads.d_b_ab += &(d_ab_dz.sum_axis(Axis(0)));

        let d_x_beta = d_ab_dz.dot(&self.w_ab.t());

        // grad of norm on q and k

        let d_q_2d = d_loss_q
            .into_shape_clone((batch_size * seq_len, self.d_head))
            .unwrap();
        let d_loss_q = f::d_l2_norm(&self.cache.q_preactivations, &d_q_2d)
            .into_shape_clone((batch_size, seq_len, self.d_head))
            .unwrap();

        let d_k_2d = d_loss_k
            .into_shape_clone((batch_size * seq_len, self.d_head))
            .unwrap();
        let d_loss_k = f::d_l2_norm(&self.cache.k_preactivations, &d_k_2d)
            .into_shape_clone((batch_size, seq_len, self.d_head))
            .unwrap();

        // (3, B, S, D) -> (B, S, 3, D)
        let d_loss_qkv = stack![Axis(0), d_loss_q.view(), d_loss_k.view(), d_loss_v.view()]
            .permuted_axes([1, 2, 0, 3])
            .to_owned()
            .into_shape_clone((batch_size * seq_len, 3 * self.d_head))
            .unwrap();

        self.grads.d_w_qkv += &(self.cache.x_2d.t().dot(&d_loss_qkv));
        self.grads.d_b_qkv += &d_loss_qkv.sum_axis(Axis(0));

        let d_x_2d = d_loss_qkv.dot(&self.w_qkv.t());

        let d_x_2d = d_x_2d + d_x_beta;

        let d_x = d_x_2d
            .into_shape_clone((batch_size, seq_len, self.d_in))
            .unwrap();

        d_x
    }

    pub fn step(&self, x: &Array2<f32>, state: &mut Array3<f32>) -> Array2<f32> {
        let (batch_size, _) = x.dim();

        let qkv = x.dot(&self.w_qkv) + &self.b_qkv;
        let qkv = qkv
            .into_shape_clone((batch_size, 3, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2])
            .to_owned();

        let q_preactivations = qkv.slice(s![0, .., ..]).to_owned();
        let k_preactivations = qkv.slice(s![1, .., ..]).to_owned();
        let v = qkv.slice(s![2, .., ..]);

        let q = f::l2_norm(&q_preactivations);
        let k = f::l2_norm(&k_preactivations);

        let ab_preactivations = x.dot(&self.w_ab) + &self.b_ab;
        let ab = f::sigmoid(&ab_preactivations);

        let alpha = ab.slice(s![.., 0..1]).insert_axis(Axis(2));
        let beta = ab.slice(s![.., 1..2]).insert_axis(Axis(2));

        let alpha = alpha.broadcast(state.dim()).unwrap();
        let beta = beta.broadcast(state.dim()).unwrap();

        let k = k.insert_axis(Axis(2));

        let r = (&(*state) * &k).sum_axis(Axis(1));
        let error = &v - &r;
        let update = &k * &error.insert_axis(Axis(1));

        *state = &alpha * &(*state) + &beta * &update;

        // (B, d_q) -> (B, d_q, 1)
        let q_t_inner = q.insert_axis(Axis(2));
        let attn = (&q_t_inner * &(*state)).sum_axis(Axis(1));

        let output = attn.dot(&self.w_o) + &self.b_o;

        output
    }
}

impl ToParams for GatedDeltaNet {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_qkv).with_grad(&mut self.grads.d_w_qkv),
            Param::new(&mut self.b_qkv).with_grad(&mut self.grads.d_b_qkv),
            Param::new(&mut self.w_ab).with_grad(&mut self.grads.d_w_ab),
            Param::new(&mut self.b_ab).with_grad(&mut self.grads.d_b_ab),
            Param::new(&mut self.w_o).with_grad(&mut self.grads.d_w_o),
            Param::new(&mut self.b_o).with_grad(&mut self.grads.d_b_o),
        ]
    }
}

impl ToIntermediates for GatedDeltaNet {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![
            &mut self.cache.x_2d,
            &mut self.cache.q_preactivations,
            &mut self.cache.k_preactivations,
            &mut self.cache.q,
            &mut self.cache.k,
            &mut self.cache.v,
            &mut self.cache.ab_preactivations,
            &mut self.cache.ab,
            &mut self.cache.rs,
            &mut self.cache.updates,
            &mut self.cache.states,
            &mut self.cache.attn_2d,
        ]
    }
}

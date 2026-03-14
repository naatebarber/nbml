use ndarray::{Array1, Array2, Array3, Array4, Axis, concatenate, s, stack};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToIntermediates, ToParams},
};

// TODO
// cross attention variant

#[derive(Default, Debug, Clone)]
pub struct GLACache {
    pub x_2d: Array2<f64>,
    pub q: Array3<f64>,
    pub k: Array3<f64>,
    pub v: Array3<f64>,
    pub forget_2d: Array2<f64>,
    pub forget_gates: Array3<f64>,
    pub states: Array4<f64>,
    pub attn_2d: Array2<f64>,
}

#[derive(Default, Debug, Clone)]
pub struct GLAGrads {
    pub d_w_qkv: Array2<f64>,
    pub d_b_qkv: Array1<f64>,
    pub d_w_forget: Array2<f64>,
    pub d_b_forget: Array1<f64>,
    pub d_w_o: Array2<f64>,
    pub d_b_o: Array1<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GatedLinearAttention {
    pub d_in: usize,
    pub d_head: usize,
    pub n_head: usize,

    pub w_qkv: Array2<f64>,
    pub b_qkv: Array1<f64>,
    pub w_forget: Array2<f64>,
    pub b_forget: Array1<f64>,
    pub w_o: Array2<f64>,
    pub b_o: Array1<f64>,

    #[serde(skip)]
    pub cache: GLACache,
    #[serde(skip)]
    pub grads: GLAGrads,
}

impl GatedLinearAttention {
    pub fn new(d_in: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            d_in,
            d_head,
            n_head,

            w_qkv: f::xavier_normal((d_in, 3 * n_head * d_head)),
            b_qkv: Array1::zeros(3 * n_head * d_head),
            w_forget: f::xavier_normal((d_in, n_head * 2 * d_head)),
            b_forget: Array1::from_elem(n_head * 2 * d_head, 1.),
            w_o: f::xavier_normal((n_head * d_head, d_in)),
            b_o: Array1::zeros(d_in),

            cache: GLACache::default(),
            grads: GLAGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        let x_2d = x
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();

        let qkv_2d = x_2d.dot(&self.w_qkv) + &self.b_qkv;
        let qkv = qkv_2d
            .into_shape_clone((batch_size, seq_len, 3, self.n_head, self.d_head))
            .unwrap()
            .permuted_axes([2, 0, 3, 1, 4]) // (3, B, nH, S, dH)
            .to_owned()
            .into_shape_clone((3, batch_size * self.n_head, seq_len, self.d_head))
            .unwrap();

        let q = qkv.slice(s![0, .., .., ..]);
        let k = qkv.slice(s![1, .., .., ..]);
        let v = qkv.slice(s![2, .., .., ..]);

        let forget_2d = x_2d.dot(&self.w_forget) + &self.b_forget;
        let forget_gates_2d = f::sigmoid(&forget_2d);

        let forget_gates = forget_gates_2d
            .into_shape_clone((batch_size, seq_len, self.n_head, 2 * self.d_head))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .into_shape_clone((batch_size * self.n_head, seq_len, 2 * self.d_head))
            .unwrap();

        if grad {
            self.cache.x_2d = x_2d;
            self.cache.q = q.to_owned();
            self.cache.k = k.to_owned();
            self.cache.v = v.to_owned();
            self.cache.forget_2d = forget_2d;
            self.cache.forget_gates = forget_gates.clone();
            self.cache.states = Array4::zeros((
                seq_len + 1,
                batch_size * self.n_head,
                self.d_head,
                self.d_head,
            ));
        }

        let mut state = Array3::zeros((batch_size * self.n_head, self.d_head, self.d_head));
        let mut attn = Array3::zeros((batch_size * self.n_head, seq_len, self.d_head));

        for t in 0..seq_len {
            let q_t = q.slice(s![.., t, ..]);
            let k_t = k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = v.slice(s![.., t, ..]); // (B, d_v)

            let forget_gate_t = forget_gates.slice(s![.., t, ..]);
            let forget_alpha = forget_gate_t.slice(s![.., 0..self.d_head]);
            let forget_beta = forget_gate_t.slice(s![.., self.d_head..]);
            let forget_expanded =
                &forget_alpha.insert_axis(Axis(2)) * &forget_beta.insert_axis(Axis(1));

            // (B, d_k, 1) * (B, 1, d_v) -> (B, d_k, d_v)
            let k_t_inner = k_t.insert_axis(Axis(2));
            let v_t_outer = v_t.insert_axis(Axis(1));
            let kv = &k_t_inner * &v_t_outer;

            state = &state * &forget_expanded + &kv;

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
            .into_shape_clone((batch_size, self.n_head, seq_len, self.d_head))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .into_shape_clone((batch_size * seq_len, self.n_head * self.d_head))
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

    pub fn backward(&mut self, d_loss: Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, features) = d_loss.dim();
        let (_, _, d_k, d_v) = self.cache.states.dim();

        if self.grads.d_w_o.dim() == (0, 0) {
            self.grads.d_w_o = Array2::zeros(self.w_o.dim())
        }

        if self.grads.d_b_o.dim() == 0 {
            self.grads.d_b_o = Array1::zeros(self.b_o.dim())
        }

        if self.grads.d_w_forget.dim() == (0, 0) {
            self.grads.d_w_forget = Array2::zeros(self.w_forget.dim());
        }

        if self.grads.d_b_forget.dim() == 0 {
            self.grads.d_b_forget = Array1::zeros(self.b_forget.dim());
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

        let d_attn = d_loss_2d
            .dot(&self.w_o.t())
            .into_shape_clone((batch_size, seq_len, self.n_head, self.d_head))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .into_shape_clone((batch_size * self.n_head, seq_len, self.d_head))
            .unwrap();

        let mut d_loss_q = Array3::zeros((batch_size * self.n_head, seq_len, self.d_head));
        let mut d_loss_k = Array3::zeros((batch_size * self.n_head, seq_len, d_k));
        let mut d_loss_v = Array3::zeros((batch_size * self.n_head, seq_len, d_v));

        let mut d_loss_state = Array3::zeros((batch_size * self.n_head, d_k, d_v));
        let mut d_loss_forget = Array3::zeros((batch_size * self.n_head, seq_len, 2 * self.d_head));

        for t in (0..seq_len).rev() {
            let d_loss_t = d_attn.slice(s![.., t, ..]); // (B, d_v)
            let state_prev = self.cache.states.slice(s![t, .., .., ..]);
            let state_next = self.cache.states.slice(s![t + 1, .., .., ..]); // (B, d_k, d_v)
            let q_t = self.cache.q.slice(s![.., t, ..]); // (B, d_q)
            let k_t = self.cache.k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = self.cache.v.slice(s![.., t, ..]); // (B, d_v)

            let forget_gate_t = self.cache.forget_gates.slice(s![.., t, ..]); // (B, dv)
            let forget_alpha = forget_gate_t.slice(s![.., 0..self.d_head]);
            let forget_beta = forget_gate_t.slice(s![.., self.d_head..]);
            let forget_expanded_t =
                &forget_alpha.insert_axis(Axis(2)) * &forget_beta.insert_axis(Axis(1));

            // (B, d_v) -> (B, 1, d_v)
            let d_loss_t = d_loss_t.to_owned().insert_axis(Axis(1));

            // state_t * d_loss_t
            // (B, d_k, d_v) * (B, 1, d_v) -> (B, d_k, d_v)
            let d_loss_q_t = (&state_next * &d_loss_t).sum_axis(Axis(2));
            d_loss_q.slice_mut(s![.., t, ..]).assign(&d_loss_q_t);

            // q_t * d_loss_t
            // (B, d_k, 1) * (B, 1, d_v) -> (B, d_k, d_v)
            // this is next state, as in:
            // S' = f * S + k.dot(v.T)
            d_loss_state += &(&q_t.insert_axis(Axis(2)) * &d_loss_t);

            // (B, d_k, d_v)
            let d_forget_expanded = &d_loss_state * &state_prev;
            let d_forget_alpha =
                (&d_forget_expanded * &forget_beta.insert_axis(Axis(1))).sum_axis(Axis(2));
            let d_forget_beta =
                (&d_forget_expanded * &forget_alpha.insert_axis(Axis(2))).sum_axis(Axis(1));
            let d_forget_t = concatenate![Axis(1), d_forget_alpha.view(), d_forget_beta.view()];
            d_loss_forget.slice_mut(s![.., t, ..]).assign(&d_forget_t);

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

            // pass state backward modified based on forget gate
            d_loss_state = &d_loss_state * &forget_expanded_t;
        }

        let d_z_forget = f::d_sigmoid(&self.cache.forget_2d);
        let d_loss_forget_2d = d_loss_forget
            .into_shape_clone((batch_size, self.n_head, seq_len, 2 * self.d_head))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .into_shape_clone((batch_size * seq_len, self.n_head * 2 * self.d_head))
            .unwrap()
            * &d_z_forget;

        self.grads.d_w_forget += &(self.cache.x_2d.t().dot(&d_loss_forget_2d));
        self.grads.d_b_forget += &d_loss_forget_2d.sum_axis(Axis(0));
        let d_x_forget_2d = d_loss_forget_2d.dot(&self.w_forget.t());
        let d_x_forget = d_x_forget_2d
            .into_shape_clone((batch_size, seq_len, self.d_in))
            .unwrap();

        // (3, B * nH, S, D) -> (B * S, 3 * nH * D)
        let d_loss_qkv = stack![Axis(0), d_loss_q.view(), d_loss_k.view(), d_loss_v.view()]
            .into_shape_clone((3, batch_size, self.n_head, seq_len, self.d_head))
            .unwrap()
            .permuted_axes([1, 3, 0, 2, 4])
            .into_shape_clone((batch_size * seq_len, 3 * self.n_head * self.d_head))
            .unwrap();

        self.grads.d_w_qkv += &(self.cache.x_2d.t().dot(&d_loss_qkv));
        self.grads.d_b_qkv += &d_loss_qkv.sum_axis(Axis(0));

        let d_x_qkv_2d = d_loss_qkv.dot(&self.w_qkv.t());

        let d_x_qkv = d_x_qkv_2d
            .into_shape_clone((batch_size, seq_len, self.d_in))
            .unwrap();

        d_x_qkv + d_x_forget
    }
}

impl ToParams for GatedLinearAttention {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_qkv).with_grad(&mut self.grads.d_w_qkv),
            Param::new(&mut self.b_qkv).with_grad(&mut self.grads.d_b_qkv),
            Param::new(&mut self.w_forget).with_grad(&mut self.grads.d_w_forget),
            Param::new(&mut self.b_forget).with_grad(&mut self.grads.d_b_forget),
            Param::new(&mut self.w_o).with_grad(&mut self.grads.d_w_o),
            Param::new(&mut self.b_o).with_grad(&mut self.grads.d_b_o),
        ]
    }
}

impl ToIntermediates for GatedLinearAttention {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![
            &mut self.cache.x_2d,
            &mut self.cache.q,
            &mut self.cache.k,
            &mut self.cache.v,
            &mut self.cache.forget_2d,
            &mut self.cache.forget_gates,
            &mut self.cache.states,
            &mut self.cache.attn_2d,
        ]
    }
}

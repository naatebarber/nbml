use serde::{Deserialize, Serialize};

use crate::{
    f2 as f,
    optim2::{Param, ToParams},
    s,
    tensor::{Tensor, Tensor1, Tensor2, Tensor3, Tensor4},
    util::Cache,
};

// TODO
// make multiheaded. easy since attention only loops over sequence
// multihead is just a projection problem, heads added to batch dim
//
// TODO
// cross attention variant

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GatedLinearAttention {
    d_in: usize,
    d_head: usize,
    n_head: usize,

    w_qkv: Tensor2,
    b_qkv: Tensor1,
    w_forget: Tensor2,
    b_forget: Tensor1,
    w_o: Tensor2,
    b_o: Tensor1,

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
}

impl GatedLinearAttention {
    pub fn new(d_in: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            d_in,
            d_head,
            n_head,

            w_qkv: f::xavier_normal((d_in, 3 * n_head * d_head)),
            b_qkv: Tensor1::zeros(3 * n_head * d_head),
            w_forget: f::xavier_normal((d_in, n_head * 2 * d_head)),
            b_forget: Tensor::from_elem(n_head * 2 * d_head, 1.),
            w_o: f::xavier_normal((n_head * d_head, d_in)),
            b_o: Tensor1::zeros(d_in),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor3, grad: bool) -> Tensor3 {
        let (batch_size, seq_len, features) = x.dim3();

        let x_2d = x.reshape((batch_size * seq_len, features));

        let qkv_2d = x_2d.dot(&self.w_qkv) + &self.b_qkv;
        let qkv = qkv_2d
            .reshape((batch_size, seq_len, 3, self.n_head, self.d_head))
            .permute(&[2, 0, 3, 1, 4]) // (3, B, nH, S, dH)
            .reshape((3, batch_size * self.n_head, seq_len, self.d_head));

        let q = qkv.slice(s![0, .., .., ..]);
        let k = qkv.slice(s![1, .., .., ..]);
        let v = qkv.slice(s![2, .., .., ..]);

        let forget_2d = x_2d.dot(&self.w_forget) + &self.b_forget;
        let forget_gates_2d = f::sigmoid(&forget_2d);

        let forget_gates = forget_gates_2d
            .reshape((batch_size, seq_len, self.n_head, 2 * self.d_head))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * self.n_head, seq_len, 2 * self.d_head));

        if grad {
            self.cache.set("x_2d", x_2d);
            self.cache.set("q", q.clone());
            self.cache.set("k", k.clone());
            self.cache.set("v", v.clone());
            self.cache.set("forget_2d", forget_2d);
            self.cache.set("forget_gates", forget_gates.clone());
            self.cache.set(
                "states",
                Tensor4::zeros((
                    seq_len + 1,
                    batch_size * self.n_head,
                    self.d_head,
                    self.d_head,
                )),
            );
        }

        let mut state = Tensor3::zeros((batch_size * self.n_head, self.d_head, self.d_head));
        let mut attn = Tensor3::zeros((batch_size * self.n_head, seq_len, self.d_head));

        for t in 0..seq_len {
            let q_t = q.slice(s![.., t, ..]);
            let k_t = k.slice(s![.., t, ..]); // (B, d_k)
            let v_t = v.slice(s![.., t, ..]); // (B, d_v)

            let forget_gate_t = forget_gates.slice(s![.., t, ..]);
            let forget_alpha = forget_gate_t.slice(s![.., 0..self.d_head as isize]);
            let forget_beta = forget_gate_t.slice(s![.., self.d_head as isize..]);
            let forget_expanded = &forget_alpha.insert_axis(2) * &forget_beta.insert_axis(1);

            // (B, d_k, 1) * (B, 1, d_v) -> (B, d_k, d_v)
            let k_t_inner = k_t.insert_axis(2);
            let v_t_outer = v_t.insert_axis(1);
            let kv = k_t_inner * v_t_outer;

            state = &state * forget_expanded + kv;

            // (B, d_q) -> (B, d_q, 1)
            let q_t_inner = q_t.insert_axis(2);
            let attn_t = (q_t_inner * &state).sum_axis(1);

            attn.slice_assign(s![.., t, ..], &attn_t);

            if grad {
                self.cache["states"].slice_assign(s![t + 1, .., .., ..], &state);
            }
        }

        let attn_2d = attn
            .reshape((batch_size, self.n_head, seq_len, self.d_head))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * seq_len, self.n_head * self.d_head));

        let output_2d = attn_2d.dot(&self.w_o) + &self.b_o;
        let output = output_2d.reshape((batch_size, seq_len, self.d_in));

        if grad {
            self.cache.set("attn_2d", attn_2d);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Tensor3) -> Tensor3 {
        let (batch_size, seq_len, features) = d_loss.dim3();
        let (_, _, d_k, d_v) = self.cache["states"].dim4();

        let d_loss_2d = d_loss.reshape((batch_size * seq_len, features));

        self.grads
            .accumulate("d_w_o", self.cache["attn_2d"].t().dot(&d_loss_2d));
        self.grads.accumulate("d_b_o", d_loss_2d.sum_axis(0));

        let d_attn = d_loss_2d
            .dot(&self.w_o.t())
            .reshape((batch_size, seq_len, self.n_head, self.d_head))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * self.n_head, seq_len, self.d_head));

        let mut d_loss_q = Tensor3::zeros((batch_size * self.n_head, seq_len, self.d_head));
        let mut d_loss_k = Tensor3::zeros((batch_size * self.n_head, seq_len, d_k));
        let mut d_loss_v = Tensor3::zeros((batch_size * self.n_head, seq_len, d_v));

        let mut d_loss_state = Tensor3::zeros((batch_size * self.n_head, d_k, d_v));
        let mut d_loss_forget =
            Tensor3::zeros((batch_size * self.n_head, seq_len, 2 * self.d_head));

        for t in (0..seq_len).rev() {
            let d_loss_t = d_attn.slice(s![.., t, ..]); // (B, d_v)
            let state_prev = self.cache["states"].slice(s![t, .., .., ..]);
            let state_next = self.cache["states"].slice(s![t + 1, .., .., ..]); // (B, d_k, d_v)
            let q_t = self.cache["q"].slice(s![.., t, ..]); // (B, d_q)
            let k_t = self.cache["k"].slice(s![.., t, ..]); // (B, d_k)
            let v_t = self.cache["v"].slice(s![.., t, ..]); // (B, d_v)

            let forget_gate_t = self.cache["forget_gates"].slice(s![.., t, ..]); // (B, 2*dH)
            let forget_alpha = forget_gate_t.slice(s![.., 0..self.d_head as isize]);
            let forget_beta = forget_gate_t.slice(s![.., self.d_head as isize..]);
            let forget_expanded_t = forget_alpha.insert_axis(2) * forget_beta.insert_axis(1);

            // (B, d_v) -> (B, 1, d_v)
            let d_loss_t = d_loss_t.insert_axis(1);

            // state_t * d_loss_t
            // (B, d_k, d_v) * (B, 1, d_v) -> (B, d_k, d_v)
            let d_loss_q_t = (state_next * &d_loss_t).sum_axis(2);
            d_loss_q.slice_assign(s![.., t, ..], &d_loss_q_t);

            // q_t * d_loss_t
            // (B, d_k, 1) * (B, 1, d_v) -> (B, d_k, d_v)
            // this is next state, as in:
            // S' = f * S + k.dot(v.T)
            d_loss_state += q_t.insert_axis(2) * &d_loss_t;

            // forget gate grads
            // (B, d_k, d_v)
            let d_forget_expanded = &d_loss_state * state_prev;
            let d_forget_alpha = (&d_forget_expanded * forget_beta.insert_axis(1)).sum_axis(2);
            let d_forget_beta = (d_forget_expanded * forget_alpha.insert_axis(2)).sum_axis(1);
            let d_forget_t = Tensor::concatenate(1, &[&d_forget_alpha, &d_forget_beta]);
            d_loss_forget.slice_assign(s![.., t, ..], &d_forget_t);

            // (B, d_v) -> (B, 1, d_v)
            // (B, 1, d_v) * (B, d_k, d_v) -> (B, d_k, sum(d_v)) -> (B, d_k)
            let d_loss_k_t = (v_t.insert_axis(1) * &d_loss_state).sum_axis(2);
            d_loss_k.slice_assign(s![.., t, ..], &d_loss_k_t);

            // (B, d_k) -> (B, d_k, 1)
            // (B, d_k, 1) * (B, d_k, d_v) -> (B, sum(d_k), d_v) -> (B, d_v)
            let d_loss_v_t = (k_t.insert_axis(2) * &d_loss_state).sum_axis(1);
            d_loss_v.slice_assign(s![.., t, ..], &d_loss_v_t);

            // pass state backward modified based on forget gate
            d_loss_state = d_loss_state * forget_expanded_t;
        }

        let d_z_forget = f::d_sigmoid(&self.cache["forget_2d"]);
        let d_loss_forget_2d = d_loss_forget
            .reshape((batch_size, self.n_head, seq_len, 2 * self.d_head))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * seq_len, self.n_head * 2 * self.d_head))
            * &d_z_forget;

        self.grads
            .accumulate("d_w_forget", self.cache["x_2d"].t().dot(&d_loss_forget_2d));
        self.grads
            .accumulate("d_b_forget", d_loss_forget_2d.sum_axis(0));
        let d_x_forget = d_loss_forget_2d
            .dot(&self.w_forget.t())
            .reshape((batch_size, seq_len, self.d_in));

        // (3, B * nH, S, D) -> (B * S, 3 * nH * D)
        let d_loss_qkv = Tensor::stack(0, &[&d_loss_q, &d_loss_k, &d_loss_v])
            .reshape((3, batch_size, self.n_head, seq_len, self.d_head))
            .permute(&[1, 3, 0, 2, 4])
            .reshape((batch_size * seq_len, 3 * self.n_head * self.d_head));

        self.grads
            .accumulate("d_w_qkv", self.cache["x_2d"].t().dot(&d_loss_qkv));
        self.grads.accumulate("d_b_qkv", d_loss_qkv.sum_axis(0));

        let d_x_qkv = d_loss_qkv
            .dot(&self.w_qkv.t())
            .reshape((batch_size, seq_len, self.d_in));

        d_x_qkv + d_x_forget
    }
}

impl ToParams for GatedLinearAttention {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_qkv).with_grad(&mut self.grads["d_w_qkv"]),
            Param::new(&mut self.b_qkv).with_grad(&mut self.grads["d_b_qkv"]),
            Param::new(&mut self.w_forget).with_grad(&mut self.grads["d_w_forget"]),
            Param::new(&mut self.b_forget).with_grad(&mut self.grads["d_b_forget"]),
            Param::new(&mut self.w_o).with_grad(&mut self.grads["d_w_o"]),
            Param::new(&mut self.b_o).with_grad(&mut self.grads["d_b_o"]),
        ]
    }
}

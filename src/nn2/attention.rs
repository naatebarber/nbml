use serde::{Deserialize, Serialize};

use crate::{
    Tensor, f2 as f,
    optim2::{Param, ToParams},
    s,
    tensor::{Float, Tensor1, Tensor2, Tensor3},
    util::Cache,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Attention {
    #[serde(skip)]
    pub cache: Cache,
}

impl Attention {
    pub fn new() -> Self {
        Self {
            cache: Cache::new(),
        }
    }

    pub fn forward(
        &mut self,
        q: Tensor3,
        k: Tensor3,
        v: Tensor3,
        mask: Tensor3,
        grad: bool,
    ) -> Tensor3 {
        let (batch_size, seq_len_q, _) = q.dim3();
        let (_, seq_len_k, features_k) = k.dim3();
        let (_, _, features_v) = v.dim3();

        assert!(
            q.dim3().0 == k.dim3().0 && k.dim3().0 == v.dim3().0,
            "batch size mismatch"
        );
        assert!(k.dim3().1 == v.dim3().1, "k/v seq len mismatch");
        assert!(q.dim3().2 == k.dim3().2, "q/k feature size mismatch");
        assert!(
            mask.dim3() == (batch_size, seq_len_q, seq_len_k),
            "mask dim != (batch_size, seq_len_q, seq_len_k) mask={:?} desired={:?}",
            mask.dim3(),
            (batch_size, seq_len_q, seq_len_k)
        );

        if grad {
            self.cache.set("q", q.clone());
            self.cache.set("k", k.clone());
            self.cache.set("v", v.clone());
            self.cache.set(
                "weights",
                Tensor3::zeros((batch_size, seq_len_q, seq_len_k)),
            );
        }

        let mut output = Tensor3::zeros((batch_size, seq_len_q, features_v));

        let mask = mask.mapv(|x| if x == 0. { -Float::INFINITY } else { 0. });

        for i in 0..batch_size {
            let q_i = q.slice(s![i, .., ..]);
            let k_i = k.slice(s![i, .., ..]);
            let v_i = v.slice(s![i, .., ..]);

            let scores = q_i.dot(&k_i.t());
            let scores = (scores / (features_k as Float).sqrt()) + mask.slice(s![i, .., ..]);
            let weights = f::softmax(&scores);

            let out = weights.dot(&v_i);
            output.slice_assign(s![i, .., ..], &out);

            if grad {
                self.cache["weights"].slice_assign(s![i, .., ..], &weights);
            }
        }

        output
    }

    pub fn backward(&mut self, d_loss: Tensor3) -> (Tensor3, Tensor3, Tensor3) {
        let (batch_size, _, _) = self.cache["q"].dim3();
        let (_, _, features_k) = self.cache["k"].dim3();

        let mut d_q = Tensor3::zeros_like(&self.cache["q"]);
        let mut d_k = Tensor3::zeros_like(&self.cache["k"]);
        let mut d_v = Tensor3::zeros_like(&self.cache["v"]);

        for i in 0..batch_size {
            let d_loss_i = d_loss.slice(s![i, .., ..]);

            let weights_i = self.cache["weights"].slice(s![i, .., ..]);
            let q_i = self.cache["q"].slice(s![i, .., ..]);
            let k_i = self.cache["k"].slice(s![i, .., ..]);
            let v_i = self.cache["v"].slice(s![i, .., ..]);

            let d_v_i = weights_i.t().dot(&d_loss_i);
            d_v.slice_assign(s![i, .., ..], &d_v_i);

            let d_weights = d_loss_i.dot(&v_i.t());
            let d_scores = f::d_softmax(&weights_i.to_owned(), &d_weights);
            let d_scores = d_scores / (features_k as Float).sqrt();

            let d_k_i_t = q_i.t().dot(&d_scores);
            let d_k_i = d_k_i_t.t();
            d_k.slice_assign(s![i, .., ..], &d_k_i);

            let d_q_i = d_scores.dot(&k_i);
            d_q.slice_assign(s![i, .., ..], &d_q_i);
        }

        (d_q, d_k, d_v)
    }

    pub fn weights(&self) -> &Tensor3 {
        &self.cache["weights"]
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SelfAttention {
    pub d_in: usize,
    pub d_head: usize,
    pub n_head: usize,

    pub w_qkv: Tensor2,
    pub b_qkv: Tensor1,
    pub w_o: Tensor2,
    pub b_o: Tensor1,

    pub attention: Attention,

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
}

impl SelfAttention {
    pub fn new(d_in: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            d_in,
            d_head,
            n_head,

            w_qkv: f::xavier_normal((d_in, 3 * n_head * d_head)),
            b_qkv: Tensor1::zeros(3 * n_head * d_head),
            w_o: f::xavier_normal((n_head * d_head, d_in)),
            b_o: Tensor1::zeros(d_in),

            attention: Attention::new(),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor3, mask: Tensor3, grad: bool) -> Tensor3 {
        let (batch_size, seq_len, features) = x.dim3();
        assert!(features == self.d_in, "feature dimension mismatch");

        // x -> (B * S, F)
        let x_2d = x.reshape((batch_size * seq_len, features));

        if grad {
            self.cache.set("x_2d", x_2d.clone());
        }

        // qkv = (B * S, 3 * nH * dH)
        let qkv = x_2d.dot(&self.w_qkv) + &self.b_qkv;
        // qkv -> (B, S, 3, nH, dH)
        let qkv = qkv.reshape((batch_size, seq_len, 3, self.n_head, self.d_head));
        // qkv -> (3, nH, B, S, dH)
        let qkv_permuted = qkv.permute(&[2, 0, 3, 1, 4]);

        let q = qkv_permuted.slice(s![0, .., .., .., ..]).reshape((
            batch_size * self.n_head,
            seq_len,
            self.d_head,
        ));
        let k = qkv_permuted.slice(s![1, .., .., .., ..]).reshape((
            batch_size * self.n_head,
            seq_len,
            self.d_head,
        ));
        let v = qkv_permuted.slice(s![2, .., .., .., ..]).reshape((
            batch_size * self.n_head,
            seq_len,
            self.d_head,
        ));

        // mask = (B, S, S) -> (B, nH, S, S) -> (B * nH, S, S)
        let mask = mask
            .insert_axis(1)
            .broadcast((batch_size, self.n_head, seq_len, seq_len))
            .reshape((batch_size * self.n_head, seq_len, seq_len));

        // attn = (B * nH, S, dH)
        let attn = self.attention.forward(q, k, v, mask, grad);

        // attn -> (B * S, nH * dH)
        let attn_2d = attn
            .reshape((batch_size, self.n_head, seq_len, self.d_head))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * seq_len, self.n_head * self.d_head));

        if grad {
            self.cache.set("attn_2d", attn_2d.clone());
        }

        // o = (B * S, F)
        let o_2d = attn_2d.dot(&self.w_o) + &self.b_o;
        // o -> (B, S, F)
        o_2d.reshape((batch_size, seq_len, features))
    }

    pub fn backward(&mut self, d_loss: Tensor3) -> Tensor3 {
        let (batch_size, seq_len, features) = d_loss.dim3();

        let d_loss_2d = d_loss.reshape((batch_size * seq_len, features));

        let d_b_o = d_loss_2d.sum_axis(0);
        let d_w_o = self.cache["attn_2d"].t().dot(&d_loss_2d);
        self.grads.accumulate("d_b_o", d_b_o);
        self.grads.accumulate("d_w_o", d_w_o);

        // (B * S, nH * dH)
        let d_o_2d = d_loss_2d.dot(&self.w_o.t());

        // (B * S, nH * dH) -> (B, S, nH, dH) -> (B, nH, S, dH) -> (B * nH, S, dH)
        let d_o = d_o_2d
            .reshape((batch_size, seq_len, self.n_head, self.d_head))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * self.n_head, seq_len, self.d_head));

        // d_q,d_k,d_v = (B * nH, S, dH)
        let (d_q, d_k, d_v) = self.attention.backward(d_o);

        // d_q,d_k,d_v -> (B, nH, S, dH)
        let d_q = d_q.reshape((batch_size, self.n_head, seq_len, self.d_head));
        let d_k = d_k.reshape((batch_size, self.n_head, seq_len, self.d_head));
        let d_v = d_v.reshape((batch_size, self.n_head, seq_len, self.d_head));

        // d_qkv = (3, B, nH, S, dH)
        let d_qkv = Tensor::stack(0, &[&d_q, &d_k, &d_v]);

        // d_qkv -> (B, S, 3, nH, dH)
        let d_qkv = d_qkv.permute(&[1, 3, 0, 2, 4]);
        // d_qkv -> (B * S, 3 * nH * dH)
        let d_qkv = d_qkv.reshape((batch_size * seq_len, 3 * self.n_head * self.d_head));

        let d_b_qkv = d_qkv.sum_axis(0);
        let d_w_qkv = self.cache["x_2d"].t().dot(&d_qkv);
        self.grads.accumulate("d_b_qkv", d_b_qkv);
        self.grads.accumulate("d_w_qkv", d_w_qkv);

        let d_proj_2d = d_qkv.dot(&self.w_qkv.t());

        d_proj_2d.reshape((batch_size, seq_len, features))
    }
}

impl ToParams for SelfAttention {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_qkv).with_grad(&mut self.grads["d_w_qkv"]),
            Param::new(&mut self.b_qkv).with_grad(&mut self.grads["d_b_qkv"]),
            Param::new(&mut self.w_o).with_grad(&mut self.grads["d_w_o"]),
            Param::new(&mut self.b_o).with_grad(&mut self.grads["d_b_o"]),
        ]
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CrossAttention {
    pub d_in: usize,
    pub d_head: usize,
    pub n_head: usize,

    pub w_q: Tensor2,
    pub b_q: Tensor1,
    pub w_kv: Tensor2,
    pub b_kv: Tensor1,
    pub w_o: Tensor2,
    pub b_o: Tensor1,

    pub attention: Attention,

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
}

impl CrossAttention {
    pub fn new(d_in: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            d_in,
            d_head,
            n_head,

            w_q: f::xavier_normal((d_in, n_head * d_head)),
            b_q: Tensor1::zeros(n_head * d_head),
            w_kv: f::xavier_normal((d_in, 2 * n_head * d_head)),
            b_kv: Tensor1::zeros(2 * n_head * d_head),
            w_o: f::xavier_normal((n_head * d_head, d_in)),
            b_o: Tensor1::zeros(d_in),

            attention: Attention::new(),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x_q: Tensor3, x_kv: Tensor3, mask: Tensor3, grad: bool) -> Tensor3 {
        let (batch_size, seq_len_q, features_q) = x_q.dim3();
        let (_, seq_len_kv, features_kv) = x_kv.dim3();

        assert!(x_q.dim3().0 == x_kv.dim3().0, "q/kv batch size mismatch");
        assert!(features_q == features_kv, "q/kv feature dim mismatch");

        let x_q_2d = x_q.reshape((batch_size * seq_len_q, features_q));
        let x_kv_2d = x_kv.reshape((batch_size * seq_len_kv, features_kv));

        if grad {
            self.cache.set("x_q_2d", x_q_2d.clone());
            self.cache.set("x_kv_2d", x_kv_2d.clone());
        }

        let q = x_q_2d.dot(&self.w_q) + &self.b_q;
        let kv = x_kv_2d.dot(&self.w_kv) + &self.b_kv;

        let q = q
            .reshape((batch_size, seq_len_q, self.n_head, self.d_head))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * self.n_head, seq_len_q, self.d_head));

        let kv = kv
            .reshape((batch_size, seq_len_kv, 2, self.n_head, self.d_head))
            .permute(&[2, 0, 3, 1, 4])
            .reshape((2, batch_size * self.n_head, seq_len_kv, self.d_head));

        let k = kv.slice(s![0, .., .., ..]);
        let v = kv.slice(s![1, .., .., ..]);

        // mask = (B, S, S) -> (B, nH, S, S) -> (B * nH, S, S)
        let mask = mask
            .insert_axis(1)
            .broadcast((batch_size, self.n_head, seq_len_q, seq_len_kv))
            .reshape((batch_size * self.n_head, seq_len_q, seq_len_kv));

        // attn = (B * nH, S, dH)
        let attn = self.attention.forward(q, k, v, mask, grad);

        // attn -> (B * S, nH * dH)
        let attn_2d = attn
            .reshape((batch_size, self.n_head, seq_len_q, self.d_head))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * seq_len_q, self.n_head * self.d_head));

        if grad {
            self.cache.set("attn_2d", attn_2d.clone());
        }

        let output_2d = attn_2d.dot(&self.w_o) + &self.b_o;

        output_2d.reshape((batch_size, seq_len_q, self.d_in))
    }

    pub fn backward(&mut self, d_loss: Tensor3) -> (Tensor3, Tensor3) {
        let (batch_size, seq_len, features) = d_loss.dim3();

        let d_loss_2d = d_loss.reshape((batch_size * seq_len, features));

        let d_b_o = d_loss_2d.sum_axis(0);
        let d_w_o = self.cache["attn_2d"].t().dot(&d_loss_2d);

        self.grads.accumulate("d_b_o", d_b_o);
        self.grads.accumulate("d_w_o", d_w_o);

        let d_o_2d = d_loss_2d.dot(&self.w_o.t());

        // (B * S, nH * dH) -> (B, S, nH, dH) -> (B, nH, S, dH) -> (B * nH, S, dH)
        let d_o = d_o_2d
            .reshape((batch_size, seq_len, self.n_head, self.d_head))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * self.n_head, seq_len, self.d_head));

        // (B * nH, S, Fq/Fk/Fv)
        let (d_q, d_k, d_v) = self.attention.backward(d_o);
        let (_, seq_len_q, features_q) = d_q.dim3();
        let (_, seq_len_k, features_k) = d_k.dim3();
        let (_, seq_len_v, features_v) = d_v.dim3();

        let d_q = d_q
            .reshape((batch_size, self.n_head, seq_len_q, features_q))
            .permute(&[0, 2, 1, 3])
            .reshape((batch_size * seq_len, self.n_head * features_q));

        // k, v = (B * nH, seq_len, dH) -> (B, seq_len, nH, dH)
        let d_k = d_k
            .reshape((batch_size, self.n_head, seq_len_k, features_k))
            .permute(&[0, 2, 1, 3]);
        let d_v = d_v
            .reshape((batch_size, self.n_head, seq_len_v, features_v))
            .permute(&[0, 2, 1, 3]);

        // (2, B, seq_len, nH, dH) -> (B, seq_len, 2, nH, dH)

        let d_kv = Tensor::stack(0, &[&d_k, &d_v])
            .permute(&[1, 2, 0, 3, 4])
            .reshape((batch_size * seq_len_k, 2 * self.n_head * self.d_head));

        let d_b_q = d_q.sum_axis(0);
        let d_w_q = self.cache["x_q_2d"].t().dot(&d_q);
        let d_b_kv = d_kv.sum_axis(0);
        let d_w_kv = self.cache["x_kv_2d"].t().dot(&d_kv);

        self.grads.accumulate("d_b_q", d_b_q);
        self.grads.accumulate("d_w_q", d_w_q);
        self.grads.accumulate("d_b_kv", d_b_kv);
        self.grads.accumulate("d_w_kv", d_w_kv);

        (
            d_q.dot(&self.w_q.t())
                .reshape((batch_size, seq_len_q, self.d_in)),
            d_kv.dot(&self.w_kv.t())
                .reshape((batch_size, seq_len_k, self.d_in)),
        )
    }
}

impl ToParams for CrossAttention {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_q).with_grad(&mut self.grads["d_w_q"]),
            Param::new(&mut self.b_q).with_grad(&mut self.grads["d_b_q"]),
            Param::new(&mut self.w_kv).with_grad(&mut self.grads["d_w_kv"]),
            Param::new(&mut self.b_kv).with_grad(&mut self.grads["d_b_kv"]),
            Param::new(&mut self.w_o).with_grad(&mut self.grads["d_w_o"]),
            Param::new(&mut self.b_o).with_grad(&mut self.grads["d_b_o"]),
        ]
    }
}

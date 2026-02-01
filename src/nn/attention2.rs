use std::f64;

use ndarray::{Array1, Array2, Array3, Axis, s, stack};

use crate::{f::{self, d_softmax}};

pub struct Attention {
    q: Array3<f64>,
    k: Array3<f64>,
    v: Array3<f64>,
    weights: Array3<f64>,
}

impl Attention {
    pub fn new() -> Self {
        Self {
            q: Array3::zeros((0, 0, 0)),
            k: Array3::zeros((0, 0, 0)),
            v: Array3::zeros((0, 0, 0)),
            weights: Array3::zeros((0, 0, 0)),
        }
    }

    pub fn forward(
        &mut self, 
        q: Array3<f64>, 
        k: Array3<f64>, 
        v: Array3<f64>,
        mask: Array3<f64>,
        grad: bool
    ) -> Array3<f64> {
        let (batch_size, seq_len_q, _) = q.dim();
        let (_, seq_len_k, features_k) = k.dim();
        let (_, _, features_v) = v.dim();

        assert!(q.dim().0 == k.dim().0 && k.dim().0 == v.dim().0, "batch size mismatch");
        assert!(k.dim().1 == v.dim().1, "k/v seq len mismatch");
        assert!(q.dim().2 == k.dim().2, "q/k feature size mismatch");
        assert!(mask.dim() == (batch_size, seq_len_q, seq_len_k), "mask dim != (batch_size, seq_len_q, seq_len_k)");

        self.q = if grad {
            q.clone()
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.k = if grad {
            k.clone()
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.v = if grad {
            v.clone()
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.weights = if grad {
            Array3::zeros((batch_size, seq_len_q, seq_len_k))
        } else {
            Array3::zeros((0, 0, 0))
        };

        let mut output = Array3::zeros((batch_size, seq_len_q, features_v));

        let mask = mask.mapv(|x| {
            if x == 0. {
                -f64::INFINITY
            } else {
                0.
            }
        });

        for i in 0..batch_size {
            let q_i = q.slice(s![i, .., ..]);
            let k_i = k.slice(s![i, .., ..]);
            let v_i = v.slice(s![i, .., ..]);

            let scores = q_i.dot(&k_i.t());
            let scores = (scores / (features_k as f64).sqrt()) + &mask.slice(s![i, .., ..]);
            let weights = f::softmax(&scores);

            let out = weights.dot(&v_i);
            output.slice_mut(s![i, .., ..]).assign(&out);

            if grad {
                self.weights.slice_mut(s![i, .., ..]).assign(&weights);
            }
        }

        output
    }

    pub fn backward(
        &mut self,
        d_loss: Array3<f64>,
    ) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let (batch_size, _, _) = self.q.dim();
        let (_, _, features_k) = self.k.dim();
        
        let mut d_q = Array3::zeros(self.q.dim());
        let mut d_k = Array3::zeros(self.k.dim());
        let mut d_v = Array3::zeros(self.v.dim());
        
        for i in 0..batch_size {
            let d_loss_i = d_loss.slice(s![i, .., ..]);
            let weights_i = self.weights.slice(s![i, .., ..]);
            let q_i = self.q.slice(s![i, .., ..]);
            let k_i = self.k.slice(s![i, .., ..]);
            let v_i = self.v.slice(s![i, .., ..]);

            let d_v_i = weights_i.t().dot(&d_loss_i);
            d_v.slice_mut(s![i, .., ..]).assign(&d_v_i);

            let d_weights = d_loss_i.dot(&v_i.t());
            let d_scores = d_softmax(&weights_i.to_owned(), &d_weights);
            let d_scores = d_scores / (features_k as f64).sqrt();
            
            let d_k_i_t = q_i.t().dot(&d_scores);
            let d_k_i = d_k_i_t.t();
            d_k.slice_mut(s![i, .., ..]).assign(&d_k_i);

            let d_q_i = d_scores.dot(&k_i);
            d_q.slice_mut(s![i, .., ..]).assign(&d_q_i);
        }

        (d_q, d_k, d_v)
    }
}

pub struct SelfAttention {
    d_in: usize,
    d_head: usize,
    n_head: usize,

    w_qkv: Array2<f64>,
    b_qkv: Array1<f64>,
    w_o: Array2<f64>,
    b_o: Array1<f64>,

    attention: Attention,

    x_2d: Array2<f64>,
    attn_2d: Array2<f64>,

    d_w_qkv: Array2<f64>,
    d_b_qkv: Array1<f64>,
    d_w_o: Array2<f64>,
    d_b_o: Array1<f64>,
}

impl SelfAttention {
    pub fn new(d_in: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            d_in,
            d_head,
            n_head,

            w_qkv: f::xavier_normal((d_in, 3 * n_head * d_head)),
            b_qkv: Array1::zeros(3 * n_head * d_head),
            w_o: f::xavier_normal((n_head * d_head, d_in)),
            b_o: Array1::zeros(d_in),

            attention: Attention::new(),

            x_2d: Array2::zeros((0, 0)),
            attn_2d: Array2::zeros((0, 0)),

            d_w_qkv: Array2::zeros((0, 0)),
            d_b_qkv: Array1::zeros(0),
            d_w_o: Array2::zeros((0, 0)),
            d_b_o: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, mask: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();
        assert!(features == self.d_in, "feature dimension mismatch");

        // x -> (B * S, F)
        let x_2d = x.into_shape_clone((batch_size * seq_len, features)).unwrap(); 

        self.x_2d = if grad {
            x_2d.clone()
        } else {
            Array2::zeros((0, 0))
        };

        // qkv = (B * S, 3 * nH * dH)
        let qkv = x_2d.dot(&self.w_qkv) + &self.b_qkv; 
        // qkv -> (B, S, 3, nH, dH)
        let qkv = qkv.into_shape_clone((batch_size, seq_len, 3, self.n_head, self.d_head)).unwrap();
        // qkv -> (3, nH, B, S, dH)
        let qkv_permuted = qkv.permuted_axes([2, 0, 3, 1, 4]);

        let q = qkv_permuted.slice(s![0, .., .., .., ..])
            .to_owned()
            .into_shape_clone((batch_size * self.n_head, seq_len, self.d_head))
            .unwrap();
        let k = qkv_permuted.slice(s![1, .., .., .., ..])
            .to_owned()
            .into_shape_clone((batch_size * self.n_head, seq_len, self.d_head))
            .unwrap();
        let v = qkv_permuted.slice(s![2, .., .., .., ..])
            .to_owned()
            .into_shape_clone((batch_size * self.n_head, seq_len, self.d_head))
            .unwrap();

        // attn = (B * nH, S, dH)
        let attn = self.attention.forward(q, k, v, mask, grad);

        // attn -> (B * S, nH * dH)
        let attn_2d = attn.into_shape_clone((batch_size, self.n_head, seq_len, self.d_head))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned()
            .into_shape_clone((batch_size * seq_len, self.n_head * self.d_head))
            .unwrap();

        self.attn_2d = if grad {
            attn_2d.clone()
        } else {
            Array2::zeros((0, 0))
        };

        // o = (B * S, F)
        let o_2d = attn_2d.dot(&self.w_o) + &self.b_o;
        // o -> (B, S, F)
        o_2d.into_shape_clone((batch_size, seq_len, features)).unwrap()
    }

    pub fn backward(&mut self, d_loss: Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, features) = d_loss.dim();

        let d_loss_2d = d_loss.into_shape_clone((batch_size * seq_len, features)).unwrap();
        self.d_b_o += &d_loss_2d.sum_axis(Axis(0));
        self.d_w_o += &(self.attn_2d.t().dot(&d_loss_2d));

        let d_o_2d = d_loss_2d.dot(&self.w_o.t());
        let d_o = d_o_2d.into_shape_clone((batch_size, seq_len, self.d_head)).unwrap();

        // d_q,d_k,d_v = (B * nH, S, dH)
        let (d_q, d_k, d_v) = self.attention.backward(d_o);

        // d_q,d_k,d_v -> (B, nH, S, dH)
        let d_q = d_q.into_shape_clone((batch_size, self.n_head, seq_len, self.d_head)).unwrap();
        let d_k = d_k.into_shape_clone((batch_size, self.n_head, seq_len, self.d_head)).unwrap();
        let d_v = d_v.into_shape_clone((batch_size, self.n_head, seq_len, self.d_head)).unwrap();

        // d_qkv = (3, B, nH, S, dH)
        let d_qkv = stack![Axis(0), d_q.view(), d_k.view(), d_v.view()];
        // d_qkv -> (B, S, 3, nH, dH)
        let d_qkv = d_qkv.permuted_axes([1, 3, 0, 2, 4]);
        // d_qkv -> (B * S, 3 * nH * dH)
        let d_qkv = d_qkv.to_owned().into_shape_clone((batch_size * seq_len, 3 * self.n_head * self.d_head)).unwrap();

        self.d_b_qkv += &d_qkv.sum_axis(Axis(0));
        self.d_w_qkv += &(self.x_2d.t().dot(&d_qkv));

        let d_proj_2d = d_qkv.dot(&self.w_qkv.t());
        
        d_proj_2d.into_shape_clone((batch_size, seq_len, features)).unwrap()
    }
}

pub struct CrossAttention {
    d_in: usize,
    d_head: usize,
    n_head: usize,

    w_q: Array2<f64>,
    b_q: Array1<f64>,
    w_k: Array2<f64>,
    b_k: Array1<f64>,
    w_v: Array2<f64>,
    b_v: Array1<f64>,
    w_o: Array2<f64>,
    b_o: Array1<f64>,

    attention: Attention,
    
    x_q_2d: Array2<f64>,
    x_k_2d: Array2<f64>,
    x_v_2d: Array2<f64>,
    attn_2d: Array2<f64>,

    d_w_q: Array2<f64>,
    d_b_q: Array1<f64>,
    d_w_k: Array2<f64>,
    d_b_k: Array1<f64>,
    d_w_v: Array2<f64>,
    d_b_v: Array1<f64>,
    d_w_o: Array2<f64>,
    d_b_o: Array1<f64>,
}

impl CrossAttention {
    pub fn new(d_in: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            d_in,
            d_head,
            n_head,

            w_q: f::xavier_normal((d_in, n_head * d_head)),
            b_q: Array1::zeros(n_head * d_head),
            w_k: f::xavier_normal((d_in, n_head * d_head)),
            b_k: Array1::zeros(n_head * d_head),
            w_v: f::xavier_normal((d_in, n_head * d_head)),
            b_v: Array1::zeros(n_head * d_head),
            w_o: f::xavier_normal((n_head * d_head, d_in)),
            b_o: Array1::zeros(d_in),
            
            attention: Attention::new(),

            x_q_2d: Array2::zeros((0, 0)),
            x_k_2d: Array2::zeros((0, 0)),
            x_v_2d: Array2::zeros((0, 0)),
            attn_2d: Array2::zeros((0, 0)),

            d_w_q: Array2::zeros((0, 0)),
            d_b_q: Array1::zeros(0),
            d_w_k: Array2::zeros((0, 0)),
            d_b_k: Array1::zeros(0),
            d_w_v: Array2::zeros((0, 0)),
            d_b_v: Array1::zeros(0),
            d_w_o: Array2::zeros((0, 0)),
            d_b_o: Array1::zeros(0),
        }
    }

    pub fn forward(
        &mut self, 
        x_q: Array3<f64>, 
        x_k: Array3<f64>, 
        x_v: Array3<f64>, 
        mask: Array3<f64>, 
        grad: bool
    ) -> Array3<f64> {
        let (batch_size, seq_len_q, features_q) = x_q.dim();
        let (_, seq_len_k, features_k) = x_k.dim();
        let (_, seq_len_v, features_v) = x_v.dim();
        
        assert!(x_q.dim().0 == x_k.dim().0 && x_k.dim().0 == x_v.dim().0, "q/k/v batch size mismatch");
        assert!(features_q == features_k, "q/k feature dim mismatch");
        assert!(seq_len_k == seq_len_v, "k/v seq len mismatch"); 

        let x_q_2d = x_q.into_shape_clone((batch_size * seq_len_q, features_q)).unwrap();
        let x_k_2d = x_k.into_shape_clone((batch_size * seq_len_k, features_k)).unwrap();
        let x_v_2d = x_v.into_shape_clone((batch_size * seq_len_v, features_v)).unwrap();

        if grad {
            self.x_q_2d = x_q_2d.clone();
            self.x_k_2d = x_k_2d.clone();
            self.x_v_2d = x_v_2d.clone();
        }

        let q = x_q_2d.dot(&self.w_q) + &self.b_q;
        let k = x_k_2d.dot(&self.w_k) + &self.b_k;
        let v = x_v_2d.dot(&self.w_v) + &self.b_v;

        let q = q.into_shape_clone((batch_size, seq_len_q, self.n_head, self.d_head))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned()
            .into_shape_clone((batch_size * self.n_head, seq_len_q, self.d_head))
            .unwrap();

        let k = k.into_shape_clone((batch_size, seq_len_k, self.n_head, self.d_head))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned()
            .into_shape_clone((batch_size * self.n_head, seq_len_k, self.d_head))
            .unwrap();

        let v = v.into_shape_clone((batch_size, seq_len_v, self.n_head, self.d_head))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned()
            .into_shape_clone((batch_size * self.n_head, seq_len_v, self.d_head))
            .unwrap();

        // attn = (B * nH, S, dH)
        let attn = self.attention.forward(q, k, v, mask, grad);

        // attn -> (B * S, nH * dH)
        let attn_2d = attn.into_shape_clone((batch_size, self.n_head, seq_len_q, features_v))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned()
            .into_shape_clone((batch_size * seq_len_q, self.n_head * features_v))
            .unwrap();

        if grad {
            self.attn_2d = attn_2d.clone();
        }
        
        let output_2d = attn_2d.dot(&self.w_o) + &self.b_o;

        output_2d.into_shape_clone((batch_size, seq_len_q, self.d_in)).unwrap()
    }

    pub fn backward(&mut self, d_loss: Array3<f64>) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let (batch_size, seq_len, features) = d_loss.dim();

        let d_loss_2d = d_loss.into_shape_clone((batch_size * seq_len, features)).unwrap();

        self.d_b_o += &d_loss_2d.sum_axis(Axis(0));
        self.d_w_o += &(self.attn_2d.t().dot(&d_loss_2d));

        let d_o_2d = d_loss_2d.dot(&self.w_o.t());
        let (_, features_v) = d_o_2d.dim();

        let d_o = d_o_2d.into_shape_clone((batch_size, seq_len, features_v)).unwrap();

        // (B * nH, S, Fq/Fk/Fv)
        let (d_q, d_k, d_v) = self.attention.backward(d_o);
        let (_, seq_len_q, features_q) = d_q.dim();
        let (_, seq_len_k, features_k) = d_k.dim();
        let (_, seq_len_v, features_v) = d_v.dim();
        
        let d_q = d_q.into_shape_clone((batch_size, self.n_head, seq_len_q, features_q))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned()
            .into_shape_clone((batch_size * seq_len, self.n_head * features_q))
            .unwrap();

        let d_k = d_k.into_shape_clone((batch_size, self.n_head, seq_len_k, features_k))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned()
            .into_shape_clone((batch_size * seq_len_k, self.n_head * features_k))
            .unwrap();

        let d_v = d_v.into_shape_clone((batch_size, self.n_head, seq_len_v, features_v))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned()
            .into_shape_clone((batch_size * seq_len_v, self.n_head * features_v))
            .unwrap();

        self.d_b_q += &d_q.sum_axis(Axis(0));
        self.d_w_q += &(self.x_q_2d.t().dot(&d_q));
        self.d_b_k += &d_k.sum_axis(Axis(0));
        self.d_w_k += &(self.x_k_2d.t().dot(&d_k));
        self.d_b_v += &d_v.sum_axis(Axis(0));
        self.d_w_v += &(self.x_v_2d.t().dot(&d_v));

        (
            d_q.dot(&self.w_q.t()).into_shape_clone((batch_size, seq_len_q, features_q)).unwrap(),
            d_k.dot(&self.w_k.t()).into_shape_clone((batch_size, seq_len_k, features_k)).unwrap(),
            d_v.dot(&self.w_v.t()).into_shape_clone((batch_size, seq_len_v, features_v)).unwrap()
        )
    }
}

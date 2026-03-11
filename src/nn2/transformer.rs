use serde::{Deserialize, Serialize};

use crate::f2::{d_relu, he, relu, xavier_normal};
use crate::layers::{LayerNorm, Linear};
use crate::nn2::SelfAttention;
use crate::optim2::{Param, ToParams};
use crate::tensor::{Tensor2, Tensor3};
use crate::util::Cache;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FeedForward {
    pub d_in: usize,
    pub d_hidden: usize,
    pub d_out: usize,

    pub layer_1: Linear,
    pub layer_2: Linear,

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
}

impl FeedForward {
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize) -> Self {
        Self {
            d_in,
            d_hidden,
            d_out,

            layer_1: Linear::new(d_in, d_hidden, he),
            layer_2: Linear::new(d_hidden, d_out, xavier_normal),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor2, grad: bool) -> Tensor2 {
        let z = self.layer_1.forward(&x, grad);

        if grad {
            self.cache.set("z", z.clone());
        }

        let y_1 = relu(&z);
        self.layer_2.forward(&y_1, grad)
    }

    pub fn backward(&mut self, d_loss: Tensor2) -> Tensor2 {
        let d_loss = self.layer_2.backward(&d_loss);
        let d_z = &d_loss * d_relu(&self.cache["z"]);
        self.layer_1.backward(&d_z)
    }
}

impl ToParams for FeedForward {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];

        params.append(&mut self.layer_1.params());
        params.append(&mut self.layer_2.params());

        params
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Transformer {
    pub decoder: bool,
    pub d_in: usize,
    pub attn: SelfAttention,
    pub norm_attn: LayerNorm,
    pub feed_forward: FeedForward,
    pub norm_feed_forward: LayerNorm,
}

impl Transformer {
    fn new(d_in: usize, d_head: usize, n_head: usize, decoder: bool) -> Self {
        Self {
            decoder,
            d_in,
            attn: SelfAttention::new(d_in, d_head, n_head),
            norm_attn: LayerNorm::new(d_in),
            feed_forward: FeedForward::new(d_in, 4 * d_in, d_in),
            norm_feed_forward: LayerNorm::new(d_in),
        }
    }

    pub fn new_encoder(d_in: usize, d_head: usize, n_head: usize) -> Self {
        Transformer::new(d_in, d_head, n_head, false)
    }

    pub fn new_decoder(d_in: usize, d_head: usize, n_head: usize) -> Self {
        Transformer::new(d_in, d_head, n_head, true)
    }

    pub fn pad_mask(&self, pad_mask: Tensor2) -> Tensor3 {
        let (batch_size, seq_len) = pad_mask.dim2();

        let attn_mask = pad_mask
            .insert_axis(1)
            .broadcast((batch_size, seq_len, seq_len));

        attn_mask
    }

    pub fn causal_mask(&self, batch_size: usize, seq_len: usize) -> Tensor3 {
        let mut attn_mask = Tensor2::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    attn_mask[[i, j]] = 1.
                }
            }
        }

        attn_mask
            .insert_axis(0)
            .broadcast((batch_size, seq_len, seq_len))
    }

    pub fn forward(&mut self, x: Tensor3, pad_mask: Tensor2, grad: bool) -> Tensor3 {
        let (batch_size, seq_len, features) = x.dim3();

        let norm_x = self.norm_attn.forward(x.clone(), grad);

        let mut attn_mask = self.pad_mask(pad_mask);

        if self.decoder {
            attn_mask *= &self.causal_mask(batch_size, seq_len)
        }

        let attn_x = self.attn.forward(norm_x, attn_mask, grad);
        let x = x + &attn_x;

        let norm_x = self.norm_feed_forward.forward(x.clone(), grad);
        let norm_x_2d = norm_x.reshape((batch_size * seq_len, features));
        let ff_x_2d = self.feed_forward.forward(norm_x_2d, grad);
        let ff_x = ff_x_2d.reshape((batch_size, seq_len, features));
        let x = x + ff_x;

        x
    }

    pub fn backward(&mut self, d_loss: Tensor3) -> Tensor3 {
        let (batch_size, seq_len, features) = d_loss.dim3();

        let d_loss_2d = d_loss.clone().reshape((batch_size * seq_len, features));
        let d_ffn_2d = self.feed_forward.backward(d_loss_2d);
        let d_ffn = d_ffn_2d.reshape((batch_size, seq_len, features));
        let d_norm_1 = d_loss + self.norm_feed_forward.backward(d_ffn);

        let d_attn = self.attn.backward(d_norm_1.clone());
        let d_norm = self.norm_attn.backward(d_attn);

        d_norm + d_norm_1
    }
}

impl ToParams for Transformer {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];

        params.append(&mut self.attn.params());
        params.append(&mut self.norm_attn.params());
        params.append(&mut self.feed_forward.params());
        params.append(&mut self.norm_feed_forward.params());

        params
    }
}

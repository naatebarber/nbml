use ndarray::{Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

use crate::nn::SelfAttention;
use crate::optim::param::ToParams;

use super::ffn::{FFN, LayerDef};
use super::layernorm::LayerNorm;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Transformer {
    pub decoder: bool,
    pub d_in: usize,
    pub attn: SelfAttention,
    pub norm_attn: LayerNorm,
    pub feed_forward: FFN,
    pub norm_feed_forward: LayerNorm,
}

impl Transformer {
    fn new(
        d_in: usize,
        d_head: usize,
        n_head: usize,
        ff_layers: Vec<LayerDef>,
        decoder: bool,
    ) -> Self {
        Self {
            decoder,
            d_in,
            attn: SelfAttention::new(d_in, d_head, n_head),
            norm_attn: LayerNorm::new(d_in),
            feed_forward: FFN::new(ff_layers),
            norm_feed_forward: LayerNorm::new(d_in),
        }
    }

    pub fn new_encoder(
        d_in: usize,
        d_head: usize,
        n_head: usize,
        ff_layers: Vec<LayerDef>,
    ) -> Self {
        Transformer::new(d_in, d_head, n_head, ff_layers, false)
    }

    pub fn new_decoder(
        d_in: usize,
        d_head: usize,
        n_head: usize,
        ff_layers: Vec<LayerDef>,
    ) -> Self {
        Transformer::new(d_in, d_head, n_head, ff_layers, true)
    }

    pub fn pad_mask(&self, pad_mask: Array2<f64>) -> Array3<f64> {
        let (batch_size, seq_len) = pad_mask.dim();

        let attn_mask = pad_mask
            .insert_axis(Axis(1))
            .broadcast((batch_size, seq_len, seq_len))
            .unwrap()
            .to_owned();

        attn_mask
    }

    pub fn causal_mask(&self, batch_size: usize, seq_len: usize) -> Array3<f64> {
        let mut attn_mask = Array2::zeros((seq_len, seq_len));

        attn_mask.indexed_iter_mut().for_each(|((y, x), v)| {
            if x <= y {
                *v = 1.
            }
        });

        attn_mask
            .insert_axis(Axis(0))
            .broadcast((batch_size, seq_len, seq_len))
            .unwrap()
            .to_owned()
    }

    pub fn forward(&mut self, x: Array3<f64>, pad_mask: Array2<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        let norm_x = self.norm_attn.forward(x.clone(), grad);

        let mut attn_mask = self.pad_mask(pad_mask);

        if self.decoder {
            attn_mask *= &self.causal_mask(batch_size, seq_len)
        }

        let attn_x = self.attn.forward(norm_x, attn_mask, grad);
        let x = x + &attn_x;

        let norm_x = self.norm_feed_forward.forward(x.clone(), grad);
        let norm_x_2d = norm_x
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();
        let ff_x_2d = self.feed_forward.forward(norm_x_2d, grad);
        let ff_x = ff_x_2d
            .into_shape_clone((batch_size, seq_len, features))
            .unwrap();
        let x = x + ff_x;

        x
    }

    pub fn backward(&mut self, d_loss: Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, features) = d_loss.dim();

        let d_loss_2d = d_loss
            .clone()
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();
        let d_ffn_2d = self.feed_forward.backward(d_loss_2d);
        let d_ffn = d_ffn_2d
            .into_shape_clone((batch_size, seq_len, features))
            .unwrap();
        let d_norm_1 = d_loss + self.norm_feed_forward.backward(d_ffn);

        let d_attn = self.attn.backward(d_norm_1.clone());
        let d_norm = self.norm_attn.backward(d_attn);

        d_norm + d_norm_1
    }
}

impl ToParams for Transformer {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];

        params.append(&mut self.attn.params());
        params.append(&mut self.norm_attn.params());
        params.append(&mut self.feed_forward.params());
        params.append(&mut self.norm_feed_forward.params());

        params
    }
}

use ndarray::Array3;
use serde::{Deserialize, Serialize};

use crate::optim::param::ToParams;

use super::attention::AttentionHead;
use super::ffn::{FFN, LayerDef};
use super::layernorm::LayerNorm;
use super::pad_mask::PadMask;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Transformer {
    pub d_in: usize,
    pub head: AttentionHead,
    pub norm_head: LayerNorm,
    pub feed_forward: FFN,
    pub norm_feed_forward: LayerNorm,
}

impl Transformer {
    pub fn new(d_in: usize, d_head: usize, n_head: usize, ff_layers: Vec<LayerDef>) -> Transformer {
        Transformer {
            d_in,

            head: AttentionHead::new(d_in, d_head, n_head),
            norm_head: LayerNorm::new(d_in),
            feed_forward: FFN::new(ff_layers),
            norm_feed_forward: LayerNorm::new(d_in),
        }
    }

    pub fn forward(&mut self, x: &Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, sequence_len, features) = x.dim();

        let mask = PadMask::zero_mask_batch(x);

        let x_attention = self.head.forward(&x, &mask, grad); // [B, S, F]

        let x_norm_head = self.norm_head.forward(x + x_attention); // [B, S, F]

        let x_norm_head_2 = x_norm_head
            .clone()
            .into_shape_clone((batch_size * sequence_len, features))
            .unwrap();

        let x_ff_2 = self.feed_forward.forward(x_norm_head_2, grad);

        let (.., d_out) = x_ff_2.dim();

        let x_ff = x_ff_2
            .into_shape_clone((batch_size, sequence_len, d_out))
            .unwrap();

        let x_norm_ff = self.norm_feed_forward.forward(x_norm_head + x_ff);

        x_norm_ff
    }

    pub fn backward(&mut self, d_a: Array3<f64>) -> Array3<f64> {
        let (batch_size, sequence_len, d_out) = d_a.dim();

        let d_norm_ff = self.norm_feed_forward.backward(d_a);

        let d_norm_ff_2 = d_norm_ff
            .clone()
            .into_shape_clone((batch_size * sequence_len, d_out))
            .unwrap();

        let d_feedforward_2 = self.feed_forward.backward(d_norm_ff_2);

        let d_feedforward = d_feedforward_2
            .into_shape_clone((batch_size, sequence_len, self.d_in))
            .unwrap();

        let d_resid = self.norm_head.backward(d_feedforward + d_norm_ff);

        let d_attention = self.head.backward(d_resid.clone());

        &d_resid + &d_attention
    }
}

impl ToParams for Transformer {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];

        params.append(&mut self.head.params());
        params.append(&mut self.norm_head.params());
        params.append(&mut self.feed_forward.params());
        params.append(&mut self.norm_feed_forward.params());

        params
    }
}

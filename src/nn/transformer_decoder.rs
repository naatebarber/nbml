use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::optim::param::ToParams;

use super::attention::AttentionHead;
use super::ffn::{FFN, LayerDef};
use super::layernorm::LayerNorm;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransformerDecoder {
    pub d_in: usize,
    pub head: AttentionHead,
    pub norm_head: LayerNorm,
    pub feed_forward: FFN,
    pub norm_feed_forward: LayerNorm,
}

impl TransformerDecoder {
    pub fn new(d_in: usize, d_head: usize, n_head: usize, ff_layers: Vec<LayerDef>) -> Self {
        Self {
            d_in,

            head: AttentionHead::new(d_in, d_head, n_head),
            norm_head: LayerNorm::new(d_in),
            feed_forward: FFN::new(ff_layers),
            norm_feed_forward: LayerNorm::new(d_in),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, attn_mask: Array2<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        let norm_x = self.norm_head.forward(x.clone(), grad);
        let attn_x = self.head.forward(&norm_x, &attn_mask, true, grad);
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

        let d_attn = self.head.backward(d_norm_1.clone());
        let d_norm = self.norm_head.backward(d_attn);

        d_norm + d_norm_1
    }
}

impl ToParams for TransformerDecoder {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];

        params.append(&mut self.head.params());
        params.append(&mut self.norm_head.params());
        params.append(&mut self.feed_forward.params());
        params.append(&mut self.norm_feed_forward.params());

        params
    }
}

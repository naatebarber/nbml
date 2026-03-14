use ndarray::Array3;
use serde::{Deserialize, Serialize};

use crate::f::Activation;
use crate::layers::LayerNorm;
use crate::nn::{FFN, GatedLinearAttention};
use crate::optim::{Param, ToParams};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GlaTransformer {
    pub d_in: usize,
    pub attn: GatedLinearAttention,
    pub norm_attn: LayerNorm,
    pub feed_forward: FFN,
    pub norm_feed_forward: LayerNorm,
}

impl GlaTransformer {
    pub fn new(d_in: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            d_in,
            attn: GatedLinearAttention::new(d_in, d_head, n_head),
            norm_attn: LayerNorm::new(d_in),
            feed_forward: FFN::new(vec![
                (d_in, 4 * d_in, Activation::Relu),
                (4 * d_in, d_in, Activation::Identity),
            ]),
            norm_feed_forward: LayerNorm::new(d_in),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        let norm_x = self.norm_attn.forward(x.clone(), grad);

        let attn_x = self.attn.forward(norm_x, grad);
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

impl ToParams for GlaTransformer {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];

        params.append(&mut self.attn.params());
        params.append(&mut self.norm_attn.params());
        params.append(&mut self.feed_forward.params());
        params.append(&mut self.norm_feed_forward.params());

        params
    }
}

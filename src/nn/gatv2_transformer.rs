use crate::layers::LayerNorm;
use crate::nn::{GATv2, PositionwiseFFN};
use crate::optim::ToParams;
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GATv2Transformer {
    pub norm_a: LayerNorm,
    pub attn: GATv2,
    pub norm_b: LayerNorm,
    pub ffn: PositionwiseFFN,
}

impl GATv2Transformer {
    pub fn new(d_model: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            norm_a: LayerNorm::new(d_model),
            attn: GATv2::new(d_model, d_head, n_head, d_model),
            norm_b: LayerNorm::new(d_model),
            ffn: PositionwiseFFN::new(d_model, 4 * d_model, d_model),
        }
    }

    pub fn forward(&mut self, x: Array2<f32>, index: &Array2<usize>, grad: bool) -> Array2<f32> {
        let x = x.clone()
            + self.attn.forward(
                self.norm_a
                    .forward(x.insert_axis(Axis(0)), grad)
                    .remove_axis(Axis(0)),
                index,
                grad,
            );
        let x = x.clone()
            + self.ffn.forward(
                self.norm_b
                    .forward(x.insert_axis(Axis(0)), grad)
                    .remove_axis(Axis(0)),
                grad,
            );
        x
    }

    pub fn backward(&mut self, d_loss: Array2<f32>, index: &Array2<usize>) -> Array2<f32> {
        let d_loss = d_loss.clone()
            + self
                .norm_b
                .backward(self.ffn.backward(d_loss).insert_axis(Axis(0)))
                .remove_axis(Axis(0));
        let d_loss = d_loss.clone()
            + self
                .norm_a
                .backward(self.attn.backward(d_loss, &index).insert_axis(Axis(0)))
                .remove_axis(Axis(0));
        d_loss
    }
}

impl ToParams for GATv2Transformer {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        let mut params = vec![];

        params.append(&mut self.norm_a.params());
        params.append(&mut self.attn.params());
        params.append(&mut self.norm_b.params());
        params.append(&mut self.ffn.params());

        params
    }
}

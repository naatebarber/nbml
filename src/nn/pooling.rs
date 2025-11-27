use ndarray::{Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SequencePooling {
    pub d_last: usize,
}

impl SequencePooling {
    pub fn new() -> Self {
        Self { d_last: 0 }
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array2<f64> {
        if grad {
            self.d_last = x.dim().1;
        }

        x.mean_axis(Axis(1)).unwrap()
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) -> Array3<f64> {
        let (batch, features) = d_loss.dim();

        let d_loss = d_loss
            .insert_axis(Axis(1))
            .broadcast((batch, self.d_last, features))
            .unwrap()
            .to_owned();

        &d_loss * (1. / self.d_last as f64)
    }
}

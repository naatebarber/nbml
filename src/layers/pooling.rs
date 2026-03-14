use ndarray::{Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone)]
pub struct SequencePoolingCache {
    pub mask: Array2<f64>,
    pub seq_lens: Array2<f64>,
}

impl SequencePoolingCache {
    pub fn clear(&mut self) {
        *self = SequencePoolingCache::default()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SequencePooling {
    #[serde(skip)]
    pub cache: SequencePoolingCache,
}

impl SequencePooling {
    pub fn new() -> Self {
        Self {
            cache: SequencePoolingCache::default(),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, mask: Array2<f64>, grad: bool) -> Array2<f64> {
        let seq_lens = mask.sum_axis(Axis(1)).insert_axis(Axis(1));

        if grad {
            self.cache.mask = mask.clone();
            self.cache.seq_lens = seq_lens.clone();
        }

        let x_sums = (&x * &mask.insert_axis(Axis(2))).sum_axis(Axis(1));

        &x_sums / &seq_lens
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) -> Array3<f64> {
        let (batch_size, features) = d_loss.dim();
        let (.., seq_len) = self.cache.mask.dim();

        let d_loss_mean = &d_loss / &self.cache.seq_lens;
        let d_loss_3d = d_loss_mean
            .insert_axis(Axis(1))
            .broadcast((batch_size, seq_len, features))
            .unwrap()
            .to_owned();
        let d_loss_masked = &d_loss_3d * &self.cache.mask.clone().insert_axis(Axis(2));

        d_loss_masked
    }
}

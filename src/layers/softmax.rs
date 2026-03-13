use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::f;

#[derive(Default, Debug, Clone)]
pub struct SoftmaxCache {
    pub softmax: Array2<f64>,
}

impl SoftmaxCache {
    pub fn clear(&mut self) {
        *self = SoftmaxCache::default()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Softmax {
    #[serde(skip)]
    pub cache: SoftmaxCache,
}

impl Softmax {
    pub fn new() -> Self {
        Self {
            cache: SoftmaxCache::default(),
        }
    }

    pub fn forward(&mut self, x: Array2<f64>, grad: bool) -> Array2<f64> {
        let softmax = f::softmax(&x);

        if grad {
            self.cache.softmax = softmax.clone()
        }

        softmax
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) -> Array2<f64> {
        f::d_softmax(&self.cache.softmax, &d_loss)
    }
}

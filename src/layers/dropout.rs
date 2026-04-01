use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone)]
pub struct DropoutCache {
    pub mask: Array2<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dropout {
    pub p: f32,

    #[serde(skip)]
    pub cache: DropoutCache,
}

impl Default for Dropout {
    fn default() -> Self {
        Self {
            p: 0.,
            cache: DropoutCache::default(),
        }
    }
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self {
            p,
            cache: DropoutCache::default(),
        }
    }

    pub fn forward(&mut self, x: Array2<f32>, grad: bool) -> Array2<f32> {
        if !grad {
            return x;
        }

        let scale = 1.0 / (1.0 - self.p);
        self.cache.mask = x.mapv(|_| {
            if rand::random::<f32>() < self.p {
                0.0
            } else {
                scale
            }
        });

        &x * &self.cache.mask
    }

    pub fn backward(&self, d_loss: Array2<f32>) -> Array2<f32> {
        &d_loss * &self.cache.mask
    }
}

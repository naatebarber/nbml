use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct SoftmaxCache {
    pub softmax: Array2<f32>,
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

    pub fn forward(&mut self, x: Array2<f32>, grad: bool) -> Array2<f32> {
        let softmax = f::softmax(&x);

        if grad {
            self.cache.softmax = softmax.clone()
        }

        softmax
    }

    pub fn backward(&mut self, d_loss: Array2<f32>) -> Array2<f32> {
        f::d_softmax(&self.cache.softmax, &d_loss)
    }
}

impl ToParams for Softmax {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        vec![]
    }
}

impl ToIntermediates for Softmax {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![&mut self.cache.softmax]
    }
}

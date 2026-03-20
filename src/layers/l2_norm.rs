use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct L2NormCache {
    pub x: Array2<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct L2Norm {
    #[serde(skip)]
    pub cache: L2NormCache,
}

impl L2Norm {
    pub fn new() -> Self {
        Self {
            cache: L2NormCache::default(),
        }
    }

    pub fn forward(&mut self, x: Array2<f64>, grad: bool) -> Array2<f64> {
        let l2_norm = f::l2_norm(&x);

        if grad {
            self.cache.x = x.clone()
        }

        l2_norm
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) -> Array2<f64> {
        f::d_l2_norm(&self.cache.x, &d_loss)
    }
}

impl ToParams for L2Norm {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        vec![]
    }
}

impl ToIntermediates for L2Norm {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![&mut self.cache.x]
    }
}

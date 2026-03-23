use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    f::{self, xavier_normal},
    layers::Linear,
    optim::{ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct PositionwiseFFNCache {
    pub x1: Array2<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PositionwiseFFN {
    pub linear_1: Linear,
    pub linear_2: Linear,

    #[serde(skip)]
    pub cache: PositionwiseFFNCache,
}

impl PositionwiseFFN {
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize) -> Self {
        Self {
            linear_1: Linear::new_with_init(d_in, d_hidden, xavier_normal),
            linear_2: Linear::new_with_init(d_hidden, d_out, xavier_normal),

            cache: PositionwiseFFNCache::default(),
        }
    }

    pub fn forward(&mut self, x: Array2<f32>, grad: bool) -> Array2<f32> {
        let x = self.linear_1.forward(x, grad);

        if grad {
            self.cache.x1 = x.clone()
        }

        let z = f::relu(&x);
        self.linear_2.forward(z, grad)
    }

    pub fn backward(&mut self, d_loss: Array2<f32>) -> Array2<f32> {
        let d_loss = self.linear_2.backward(d_loss);
        let d_loss_dz = d_loss * f::d_relu(&self.cache.x1);
        self.linear_1.backward(d_loss_dz)
    }
}

impl ToParams for PositionwiseFFN {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        let mut params = vec![];
        params.append(&mut self.linear_1.params());
        params.append(&mut self.linear_2.params());
        params
    }
}

impl ToIntermediates for PositionwiseFFN {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        let mut intermediates = vec![];
        intermediates.append(&mut self.linear_1.intermediates());
        intermediates.append(&mut self.linear_2.intermediates());
        intermediates.push(&mut self.cache.x1);
        intermediates
    }
}

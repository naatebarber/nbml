use serde::{Deserialize, Serialize};

use crate::{
    optim2::{Optimizer, ToParams},
    tensor::Float,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SGD {
    pub learning_rate: Float,
}

impl Default for SGD {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
        }
    }
}

impl Optimizer for SGD {
    fn with(self, _: &mut impl ToParams) -> Self {
        self
    }

    fn step(&mut self, optimizable: &mut impl ToParams) {
        unsafe {
            for param in optimizable.params().into_iter() {
                match (&param.target, &param.grad) {
                    (Some(t), Some(g)) => {
                        **t -= self.learning_rate * &(**g);
                    }
                    _ => (),
                }
            }
        }
    }
}

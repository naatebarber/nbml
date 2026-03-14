use serde::{Deserialize, Serialize};

use crate::optim::{Optimizer, ToParams};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SGD {
    pub learning_rate: f64,
}

impl Default for SGD {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
        }
    }
}

impl Optimizer for SGD {
    fn with(self, _optimizable: &mut impl ToParams) -> Self {
        self
    }

    fn step(&mut self, optimizable: &mut impl ToParams) {
        unsafe {
            for param in optimizable.params().into_iter() {
                match (param.target, param.grad) {
                    (Some(target), Some(grad)) => {
                        assert!(
                            target.dim() == grad.dim(),
                            "attempted to update target of dim {:?} with grad of dim {:?}",
                            target.dim(),
                            grad.dim()
                        );

                        let mut target_view = target.deref_into_view_mut();
                        let grad_view = grad.deref_into_view();

                        target_view -= &(&grad_view * self.learning_rate);
                    }
                    _ => (),
                }
            }
        }
    }
}

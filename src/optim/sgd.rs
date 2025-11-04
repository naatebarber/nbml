use super::{
    optimizer::Optimizer,
    param::{Param, ToParams},
};

#[derive(Debug, Clone)]
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
                match param {
                    Param::Scalar { target, grad } => {
                        *target -= *grad * self.learning_rate;
                    }
                    Param::Vector { target, grad } => {
                        *target -= &(&(*grad) * self.learning_rate);
                    }
                    Param::Matrix { target, grad } => {
                        *target -= &(&(*grad) * self.learning_rate);
                    }
                }
            }
        }
    }
}

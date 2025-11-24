use super::{
    optimizer::Optimizer,
    param::{ParamValue, ToParams},
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
                match (param.target, param.grad) {
                    (ParamValue::Scalar(target), ParamValue::Scalar(grad)) => {
                        *target -= *grad * self.learning_rate;
                    }
                    (ParamValue::Vector(target), ParamValue::Vector(grad)) => {
                        *target -= &(&(*grad) * self.learning_rate);
                    }
                    (ParamValue::Matrix(target), ParamValue::Matrix(grad)) => {
                        *target -= &(&(*grad) * self.learning_rate);
                    }
                    _ => (),
                }
            }
        }
    }
}

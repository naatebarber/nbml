use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::Tensor;

pub struct Param {
    pub target: Option<*mut Tensor>,
    pub grad: Option<*mut Tensor>,
    pub enable_weight_decay: bool,
}

impl Default for Param {
    fn default() -> Self {
        Self {
            target: None,
            grad: None,
            enable_weight_decay: false,
        }
    }
}

impl Param {
    pub fn new(target: &mut Tensor) -> Param {
        let mut param = Param::default();
        param.target = Some(target as *mut Tensor);
        param
    }

    pub fn with_grad(mut self, grad: &mut Tensor) -> Param {
        self.grad = Some(grad as *mut Tensor);
        self
    }
}

pub trait ToParams {
    fn params(&mut self) -> Vec<Param>;

    fn zero_grads(&mut self) {
        unsafe {
            for param in self.params() {
                if let Some(grad) = &param.grad {
                    **grad *= 0.;
                }
            }
        }
    }
}

pub trait Optimizer: Serialize + Deserialize<'static> + Debug + Clone {
    fn with(self, optimizable: &mut impl ToParams) -> Self;
    fn step(&mut self, optimizable: &mut impl ToParams);
}

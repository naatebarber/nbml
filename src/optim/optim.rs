use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub enum ParamValue {
    None,
    Scalar(*mut f64),
    Vector(*mut Array1<f64>),
    Matrix(*mut Array2<f64>),
}

pub struct Param {
    pub target: ParamValue,
    pub grad: ParamValue,
    pub enable_weight_decay: bool,
}

impl Default for Param {
    fn default() -> Self {
        Self {
            target: ParamValue::None,
            grad: ParamValue::None,
            enable_weight_decay: false,
        }
    }
}

impl Param {
    pub fn scalar(target: &mut f64) -> Param {
        let mut param = Param::default();
        param.target = ParamValue::Scalar(target as *mut f64);
        param
    }

    pub fn with_scalar_grad(mut self, grad: &mut f64) -> Param {
        self.grad = ParamValue::Scalar(grad as *mut f64);
        self
    }

    pub fn vector(target: &mut Array1<f64>) -> Param {
        let mut param = Param::default();
        param.target = ParamValue::Vector(target as *mut Array1<f64>);
        param
    }

    pub fn with_vector_grad(mut self, grad: &mut Array1<f64>) -> Param {
        self.grad = ParamValue::Vector(grad as *mut Array1<f64>);
        self
    }

    pub fn matrix(target: &mut Array2<f64>) -> Param {
        let mut param = Param::default();
        param.target = ParamValue::Matrix(target as *mut Array2<f64>);
        param
    }

    pub fn with_matrix_grad(mut self, grad: &mut Array2<f64>) -> Param {
        self.grad = ParamValue::Matrix(grad as *mut Array2<f64>);
        self
    }
}

pub trait ToParams {
    fn params(&mut self) -> Vec<Param>;

    fn zero_grads(&mut self) {
        unsafe {
            for param in self.params() {
                match param.grad {
                    ParamValue::Scalar(grad) => *grad = 0.,
                    ParamValue::Vector(grad) => *grad = Array1::zeros(0),
                    ParamValue::Matrix(grad) => *grad = Array2::zeros((0, 0)),
                    _ => (),
                }
            }
        }
    }
}

pub trait Optimizer: Serialize + Deserialize<'static> + Debug + Clone {
    fn with(self, optimizable: &mut impl ToParams) -> Self;
    fn step(&mut self, optimizable: &mut impl ToParams);
}

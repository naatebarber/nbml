use std::{error::Error, fs, path::PathBuf};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_binary::binary_stream::Endian;

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

#[derive(Serialize, Deserialize)]
pub enum Weight {
    None,
    Scalar(f64),
    Vector(Array1<f64>),
    Matrix(Array2<f64>),
}

impl Weight {
    pub fn from_param(param: Param) -> Weight {
        unsafe {
            match param.target {
                ParamValue::None => Weight::None,
                ParamValue::Scalar(target) => Weight::Scalar((*target).clone()),
                ParamValue::Vector(target) => Weight::Vector((*target).clone()),
                ParamValue::Matrix(target) => Weight::Matrix((*target).clone()),
            }
        }
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

    fn save_weights(&mut self, path: &PathBuf) -> Result<(), Box<dyn Error>> {
        let weights = self
            .params()
            .into_iter()
            .map(|p| Weight::from_param(p))
            .collect::<Vec<Weight>>();
        let blob = serde_binary::to_vec(&weights, Endian::Big)?;
        fs::write(path, &blob)?;
        Ok(())
    }

    fn from_weights(&mut self, path: &PathBuf) -> Result<(), Box<dyn Error>> {
        let blob = fs::read(path)?;
        let weights: Vec<Weight> = serde_binary::from_vec(blob, Endian::Big)?;

        unsafe {
            for (param, weight) in self.params().into_iter().zip(weights.into_iter()) {
                match (param.target, weight) {
                    (ParamValue::Scalar(target), Weight::Scalar(weight)) => {
                        *target = weight;
                    }
                    (ParamValue::Vector(target), Weight::Vector(weight)) => {
                        *target = weight;
                    }
                    (ParamValue::Matrix(target), Weight::Matrix(weight)) => {
                        *target = weight;
                    }
                    _ => (),
                }
            }
        }

        Ok(())
    }
}

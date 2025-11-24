use std::{error::Error, fs, path::PathBuf};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_binary::binary_stream::Endian;

pub enum Param {
    Scalar {
        target: *mut f64,
        grad: *mut f64,
    },
    Vector {
        target: *mut Array1<f64>,
        grad: *mut Array1<f64>,
    },
    Matrix {
        target: *mut Array2<f64>,
        grad: *mut Array2<f64>,
    },
}

impl Param {
    pub fn from_scalars(target: &mut f64, grad: &mut f64) -> Param {
        Param::Scalar {
            target: target as *mut f64,
            grad: grad as *mut f64,
        }
    }

    pub fn from_array1(target: &mut Array1<f64>, grad: &mut Array1<f64>) -> Param {
        Param::Vector {
            target: target as *mut Array1<f64>,
            grad: grad as *mut Array1<f64>,
        }
    }

    pub fn from_array2(target: &mut Array2<f64>, grad: &mut Array2<f64>) -> Param {
        Param::Matrix {
            target: target as *mut Array2<f64>,
            grad: grad as *mut Array2<f64>,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum Weight {
    Scalar(f64),
    Vector(Array1<f64>),
    Matrix(Array2<f64>),
}

impl Weight {
    pub fn from_param(param: Param) -> Weight {
        unsafe {
            match param {
                Param::Scalar { target, .. } => Weight::Scalar((*target).clone()),
                Param::Vector { target, .. } => Weight::Vector((*target).clone()),
                Param::Matrix { target, .. } => Weight::Matrix((*target).clone()),
            }
        }
    }
}

pub trait ToParams {
    fn params(&mut self) -> Vec<Param>;

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
                match (param, weight) {
                    (Param::Scalar { target, .. }, Weight::Scalar(weight)) => {
                        *target = weight;
                    }
                    (Param::Vector { target, .. }, Weight::Vector(weight)) => {
                        *target = weight;
                    }
                    (Param::Matrix { target, .. }, Weight::Matrix(weight)) => {
                        *target = weight;
                    }
                    _ => (),
                }
            }
        }

        Ok(())
    }
}

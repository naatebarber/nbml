use ndarray::{ArrayBase, DataMut, Dimension, IxDyn, RawArrayViewMut};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub struct Param {
    pub target: Option<RawArrayViewMut<f64, IxDyn>>,
    pub grad: Option<RawArrayViewMut<f64, IxDyn>>,
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
    pub fn new<S, D>(target: &mut ArrayBase<S, D>) -> Self
    where
        S: DataMut<Elem = f64>,
        D: Dimension,
    {
        let mut param = Param::default();
        param.target = Some(target.raw_view_mut().into_dyn());
        param
    }

    pub fn with_grad<S, D>(mut self, grad: &mut ArrayBase<S, D>) -> Self
    where
        S: DataMut<Elem = f64>,
        D: Dimension,
    {
        self.grad = Some(grad.raw_view_mut().into_dyn());
        self
    }
}

pub trait ToParams {
    fn params(&mut self) -> Vec<Param>;

    fn zero_grads(&mut self) {
        unsafe {
            for param in self.params() {
                match param.grad {
                    Some(grad) => {
                        let mut grad_view = grad.deref_into_view_mut();
                        grad_view *= 0.;
                    }
                    _ => (),
                }
            }
        }
    }
}

pub trait Intermediate {
    fn stash(&self) -> Vec<u8>;
    fn apply(&mut self, bytes: &Vec<u8>);
    fn clear(&mut self);
}

impl<T: Serialize + Deserialize<'static> + Default> Intermediate for T {
    fn stash(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }
    fn apply(&mut self, bytes: &Vec<u8>) {
        unsafe {
            *self = bincode::deserialize((bytes as *const Vec<u8>).as_ref().unwrap()).unwrap();
        }
    }
    fn clear(&mut self) {
        *self = T::default();
    }
}

pub trait ToIntermediates {
    fn intermediates(&mut self) -> Vec<&mut dyn Intermediate> {
        vec![]
    }

    fn stash_intermediates(&mut self) -> Vec<Vec<u8>> {
        self.intermediates()
            .into_iter()
            .map(|i| i.stash())
            .collect()
    }

    fn apply_intermediates(&mut self, intermediates_b: Vec<Vec<u8>>) {
        self.intermediates()
            .into_iter()
            .zip(intermediates_b.into_iter())
            .for_each(|(i, b)| {
                i.apply(&b);
            });
    }
}

pub trait Optimizer: Serialize + Deserialize<'static> + Debug + Clone {
    fn with(self, optimizable: &mut impl ToParams) -> Self;
    fn step(&mut self, optimizable: &mut impl ToParams);
}

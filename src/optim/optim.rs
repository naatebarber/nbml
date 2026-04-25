use ndarray::{Array, ArrayBase, ArrayD, DataMut, Dimension, IxDyn, RawArrayViewMut};
use serde::{Deserialize, Serialize};
use std::{error::Error, fmt::Debug, fs};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub struct Param {
    pub target: Option<RawArrayViewMut<f32, IxDyn>>,
    pub grad: Option<RawArrayViewMut<f32, IxDyn>>,
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
        S: DataMut<Elem = f32>,
        D: Dimension,
    {
        let mut param = Param::default();
        param.target = Some(target.raw_view_mut().into_dyn());
        param
    }

    pub fn with_grad<S, D>(mut self, grad: &mut ArrayBase<S, D>) -> Self
    where
        S: DataMut<Elem = f32>,
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

    fn dump(&mut self) -> Vec<TensorData> {
        let mut params = vec![];

        unsafe {
            self.params()
                .iter_mut()
                .filter_map(|param| param.target.take())
                .for_each(|target| {
                    let target_view = target.deref_into_view();
                    let shape = target_view.shape().to_owned();
                    let flat = target_view.flatten().to_vec();

                    let data = TensorData { data: flat, shape };
                    params.push(data);
                });
        }

        params
    }

    fn load(&mut self, params: Vec<TensorData>) {
        unsafe {
            self.params()
                .iter_mut()
                .filter_map(|param| param.target.take())
                .zip(params.into_iter())
                .for_each(|(target, TensorData { data, shape })| {
                    let mut target_view = target.deref_into_view_mut();
                    let incoming = ArrayD::from_shape_vec(shape, data).unwrap();
                    assert!(incoming.shape() == target_view.shape());

                    target_view *= 0.;
                    target_view += &incoming;
                });
        }
    }

    fn persist(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let params = self.dump();
        fs::write(path, bincode::serialize(&params)?)?;

        Ok(())
    }

    fn restore(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        let params = bincode::deserialize(&fs::read(path)?)?;
        self.load(params);

        Ok(())
    }
}

pub trait Intermediate {
    fn stash(&self) -> ArrayD<f32>;
    fn apply(&mut self, stashed: ArrayD<f32>);
    fn clear(&mut self);
}

pub type IntermediateCache = Vec<ArrayD<f32>>;

impl<D: Dimension> Intermediate for Array<f32, D> {
    fn stash(&self) -> ArrayD<f32> {
        self.clone().into_dyn()
    }

    fn apply(&mut self, stashed: ArrayD<f32>) {
        *self = stashed.into_dimensionality().unwrap();
    }

    fn clear(&mut self) {
        self.fill(0.0);
    }
}

pub trait ToIntermediates {
    fn intermediates(&mut self) -> Vec<&mut dyn Intermediate> {
        vec![]
    }

    fn stash_intermediates(&mut self) -> Vec<ArrayD<f32>> {
        self.intermediates()
            .into_iter()
            .map(|i| i.stash())
            .collect()
    }

    fn apply_intermediates(&mut self, intermediates_b: Vec<ArrayD<f32>>) {
        self.intermediates()
            .into_iter()
            .zip(intermediates_b.into_iter())
            .for_each(|(i, b)| {
                i.apply(b);
            });
    }
}

pub trait Optimizer: Serialize + Deserialize<'static> + Debug + Clone {
    fn with(self, optimizable: &mut impl ToParams) -> Self;
    fn step(&mut self, optimizable: &mut impl ToParams);
}

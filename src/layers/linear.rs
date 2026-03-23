use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use serde::{Deserialize, Serialize};

use crate::{
    f::InitializationFn,
    optim::{Param, ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct LinearCache {
    pub x: Array2<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct LinearGrads {
    pub d_w: Array2<f32>,
    pub d_b: Array1<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Linear {
    pub w: Array2<f32>,
    pub b: Array1<f32>,

    #[serde(skip)]
    pub cache: LinearCache,
    #[serde(skip)]
    pub grads: LinearGrads,
}

impl Linear {
    pub fn new(d_in: usize, d_out: usize) -> Self {
        Self {
            w: Array2::random((d_in, d_out), Normal::new(0., 1.).unwrap()),
            b: Array1::zeros(d_out),

            cache: LinearCache::default(),
            grads: LinearGrads::default(),
        }
    }

    pub fn new_with_init(d_in: usize, d_out: usize, init: InitializationFn) -> Self {
        Self {
            w: init((d_in, d_out)),
            b: Array1::zeros(d_out),

            cache: LinearCache::default(),
            grads: LinearGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array2<f32>, grad: bool) -> Array2<f32> {
        if grad {
            self.cache.x = x.clone();
        }

        x.dot(&self.w) + &self.b
    }

    pub fn backward(&mut self, d_loss: Array2<f32>) -> Array2<f32> {
        if self.grads.d_w.dim() == (0, 0) {
            self.grads.d_w = Array2::zeros(self.w.dim());
        }

        if self.grads.d_b.dim() == 0 {
            self.grads.d_b = Array1::zeros(self.b.dim());
        }

        self.grads.d_b += &d_loss.sum_axis(Axis(0));
        self.grads.d_w += &(self.cache.x.t().dot(&d_loss));
        d_loss.dot(&self.w.t())
    }
}

impl ToParams for Linear {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        vec![
            Param::new(&mut self.w).with_grad(&mut self.grads.d_w),
            Param::new(&mut self.b).with_grad(&mut self.grads.d_b),
        ]
    }
}

impl ToIntermediates for Linear {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![&mut self.cache.x]
    }
}

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{cache::Cache, param::{Param, ToParams}},
};

pub type LayerDef = (usize, usize, f::Activation);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Layer {
    pub w: Array2<f64>,
    pub b: Array1<f64>,

    pub activation: f::Activation,

    #[serde(skip)]
    pub cache: Cache,

    pub d_w: Array2<f64>,
    pub d_b: Array1<f64>,
}

impl Layer {
    pub fn new(d_in: usize, d_out: usize, activation: f::Activation) -> Layer {
        Layer {
            w: activation.wake().2((d_in, d_out)),
            b: Array1::zeros(d_out),
            activation,

            cache: Cache::new(),

            d_w: Array2::zeros((0, 0)),
            d_b: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, x: Array2<f64>, grad: bool) -> Array2<f64> {
        let z = x.dot(&self.w) + &self.b;

        if grad {
            self.cache.set("x", x);
            self.cache.set("z", z.clone());
        }

        (self.activation.wake().0)(&z)
    }

    pub fn backward(&mut self, d_a: Array2<f64>) -> Array2<f64> {
        let d_z = d_a * &(self.activation.wake().1)(&self.cache.get::<Array2<f64>>("z"));
        let d_w = self.cache.get::<Array2<f64>>("x").t().dot(&d_z);
        let d_b = d_z.sum_axis(Axis(0));

        self.d_w = if self.d_w.dim() == (0, 0) {
            d_w
        } else {
            &self.d_w + &d_w
        };

        self.d_b = if self.d_b.dim() == 0 {
            d_b
        } else {
            &self.d_b + &d_b
        };

        d_z.dot(&self.w.t())
    }
}

impl ToParams for Layer {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::matrix(&mut self.w).with_matrix_grad(&mut self.d_w),
            Param::vector(&mut self.b).with_vector_grad(&mut self.d_b),
        ]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FFN {
    pub layers: Vec<Layer>,
}

impl FFN {
    pub fn new(layers: Vec<LayerDef>) -> FFN {
        FFN {
            layers: layers
                .into_iter()
                .map(|(d_in, d_out, a)| Layer::new(d_in, d_out, a))
                .collect(),
        }
    }

    pub fn forward(&mut self, mut x: Array2<f64>, grad: bool) -> Array2<f64> {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x, grad)
        }

        x
    }

    pub fn backward(&mut self, mut d_a: Array2<f64>) -> Array2<f64> {
        for layer in self.layers.iter_mut().rev() {
            d_a = layer.backward(d_a);
        }

        d_a
    }
}

impl ToParams for FFN {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];
        self.layers
            .iter_mut()
            .for_each(|l| params.append(&mut l.params()));
        params
    }
}

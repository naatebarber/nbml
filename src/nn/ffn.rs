use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToIntermediates, ToParams},
};

pub type LayerDef = (usize, usize, f::Activation);

#[derive(Default, Debug, Clone)]
pub struct LayerCache {
    pub x: Array2<f32>,
    pub z: Array2<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct LayerGrads {
    pub d_w: Array2<f32>,
    pub d_b: Array1<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Layer {
    pub w: Array2<f32>,
    pub b: Array1<f32>,

    pub activation: f::Activation,

    #[serde(skip)]
    pub cache: LayerCache,
    #[serde(skip)]
    pub grads: LayerGrads,
}

impl Layer {
    pub fn new(d_in: usize, d_out: usize, activation: f::Activation) -> Layer {
        Layer {
            w: activation.wake().2((d_in, d_out)),
            b: Array1::zeros(d_out),
            activation,

            cache: LayerCache::default(),
            grads: LayerGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array2<f32>, grad: bool) -> Array2<f32> {
        let z = x.dot(&self.w) + &self.b;

        if grad {
            self.cache.x = x.clone();
            self.cache.z = z.clone();
        }

        (self.activation.wake().0)(&z)
    }

    pub fn backward(&mut self, d_a: Array2<f32>) -> Array2<f32> {
        let d_z = d_a * &(self.activation.wake().1)(&self.cache.z);
        let d_w = self.cache.x.t().dot(&d_z);
        let d_b = d_z.sum_axis(Axis(0));

        self.grads.d_w = if self.grads.d_w.dim() == (0, 0) {
            d_w
        } else {
            &self.grads.d_w + &d_w
        };

        self.grads.d_b = if self.grads.d_b.dim() == 0 {
            d_b
        } else {
            &self.grads.d_b + &d_b
        };

        d_z.dot(&self.w.t())
    }
}

impl ToParams for Layer {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w).with_grad(&mut self.grads.d_w),
            Param::new(&mut self.b).with_grad(&mut self.grads.d_b),
        ]
    }
}

impl ToIntermediates for Layer {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![&mut self.cache.x, &mut self.cache.z]
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

    pub fn forward(&mut self, mut x: Array2<f32>, grad: bool) -> Array2<f32> {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x, grad)
        }

        x
    }

    pub fn backward(&mut self, mut d_a: Array2<f32>) -> Array2<f32> {
        for layer in self.layers.iter_mut().rev() {
            d_a = layer.backward(d_a);
        }

        d_a
    }
}

impl ToParams for FFN {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];
        self.layers
            .iter_mut()
            .for_each(|l| params.append(&mut l.params()));
        params
    }
}

impl ToIntermediates for FFN {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        let mut intermediates = vec![];
        self.layers
            .iter_mut()
            .for_each(|l| intermediates.append(&mut l.intermediates()));
        intermediates
    }
}

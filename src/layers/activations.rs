use std::fmt::Debug;

use crate::Tensor;
use crate::f2;
use crate::f2::softmax;
use crate::optim2::{Param, ToParams};
use crate::util::Cache;
use serde::{Deserialize, Serialize};

pub trait Activation {
    fn forward(&mut self, x: &Tensor, grad: bool) -> Tensor;
    fn backward(&mut self, d_loss: &Tensor) -> Tensor;
}

// Stateless activations

macro_rules! stateless_activation_layer {
    ($activation_name:ident, $forward:path, $backward:path) => {
        #[derive(Serialize, Deserialize, Debug, Clone)]
        pub struct $activation_name {}

        impl Activation for $activation_name {
            fn forward(&mut self, x: &Tensor, _grad: bool) -> Tensor {
                $forward(x)
            }

            fn backward(&mut self, d_loss: &Tensor) -> Tensor {
                $backward(d_loss)
            }
        }

        impl ToParams for $activation_name {
            fn params(&mut self) -> Vec<Param> {
                vec![]
            }
        }
    };
}

stateless_activation_layer!(Relu, f2::relu, f2::d_relu);
stateless_activation_layer!(Tanh, f2::tanh, f2::d_tanh);
stateless_activation_layer!(Sigmoid, f2::sigmoid, f2::d_sigmoid);
stateless_activation_layer!(Exp, f2::exp, f2::d_exp);
stateless_activation_layer!(LeakyRelu, f2::leaky_relu, f2::d_leaky_relu);
stateless_activation_layer!(Elu, f2::elu, f2::d_elu);
stateless_activation_layer!(Softplus, f2::softplus, f2::d_softplus);
stateless_activation_layer!(Ident, f2::ident, f2::d_ident);

// Stateful activations

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Softmax {
    #[serde(skip)]
    pub cache: Cache,
}

impl Activation for Softmax {
    fn forward(&mut self, x: &Tensor, grad: bool) -> Tensor {
        let softmax = softmax(x);

        if grad {
            self.cache.set("softmax", softmax.clone());
        }

        softmax
    }

    fn backward(&mut self, d_loss: &Tensor) -> Tensor {
        f2::softmax_vector_jacobian_product(d_loss, &self.cache["softmax"])
    }
}

impl ToParams for Softmax {
    fn params(&mut self) -> Vec<Param> {
        vec![]
    }
}

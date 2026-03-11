use serde::{Deserialize, Serialize};

use crate::{
    Tensor,
    f2::InitializationFn,
    optim2::{Param, ToParams},
    tensor::Tensor1,
    util::Cache,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
}

impl Linear {
    pub fn new(d_in: usize, d_out: usize, init: InitializationFn) -> Self {
        Self {
            w: init((d_in, d_out)),
            b: Tensor1::zeros(d_out),
            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: &Tensor, grad: bool) -> Tensor {
        if grad {
            self.cache.set("x", x.clone())
        }

        x.dot(&self.w) + &self.b
    }

    pub fn backward(&mut self, d_loss: &Tensor) -> Tensor {
        let d_w = self.cache["x"].t().dot(&d_loss);
        let d_b = d_loss.sum_axis(0);

        self.grads.accumulate("d_w", d_w);
        self.grads.accumulate("d_b", d_b);

        let d_z = d_loss.dot(&self.w.t());
        d_z
    }
}

impl ToParams for Linear {
    fn params(&mut self) -> Vec<crate::optim2::Param> {
        vec![
            Param::new(&mut self.w).with_grad(&mut self.grads["d_w"]),
            Param::new(&mut self.b).with_grad(&mut self.grads["d_b"]),
        ]
    }
}

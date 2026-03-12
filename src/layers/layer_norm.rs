use serde::{Deserialize, Serialize};

use crate::{
    optim2::{Param, ToParams},
    tensor::{Float, Tensor1, Tensor3},
    util::Cache,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Tensor1,
    pub beta: Tensor1,

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
}

impl LayerNorm {
    pub fn new(d_in: usize) -> LayerNorm {
        LayerNorm {
            gamma: Tensor1::ones(d_in),
            beta: Tensor1::zeros(d_in),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor3, grad: bool) -> Tensor3 {
        let (batch_size, seq_len, features) = x.dim3();

        let x = x.reshape((batch_size * seq_len, features));

        let m = (1. / features as Float) * x.sum_axis(1).insert_axis(1);
        let u = x - m;
        let v = (1. / features as Float) * u.clone().powi(2).sum_axis(1).insert_axis(1);

        let o = (v + 1e-5).sqrt();
        let x_h = u / &o;
        let y_2 = (&x_h * &self.gamma) + &self.beta;

        if grad {
            self.cache.set("o", o.clone());
            self.cache.set("x_h", x_h.clone());
        }

        y_2.reshape((batch_size, seq_len, features))
    }

    pub fn backward(&mut self, d_loss: Tensor3) -> Tensor3 {
        let (batch_size, seq_len, features) = d_loss.dim3();
        let d_loss = d_loss.reshape((batch_size * seq_len, features));

        let d_gamma = (&d_loss * &self.cache["x_h"]).sum_axis(0);
        let d_beta = d_loss.sum_axis(0);

        self.grads.accumulate("d_gamma", d_gamma);
        self.grads.accumulate("d_beta", d_beta);

        let dx_hat = d_loss * &self.gamma;
        let dx = (1. / (features as Float * &self.cache["o"]))
            * (features as Float * &dx_hat
                - dx_hat.sum_axis(1).insert_axis(1)
                - &self.cache["x_h"] * (dx_hat * &self.cache["x_h"]).sum_axis(1).insert_axis(1));

        dx.reshape((batch_size, seq_len, features))
    }
}

impl ToParams for LayerNorm {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.gamma).with_grad(&mut self.grads["d_gamma"]),
            Param::new(&mut self.beta).with_grad(&mut self.grads["d_beta"]),
        ]
    }
}

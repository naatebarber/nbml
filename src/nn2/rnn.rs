use serde::{Deserialize, Serialize};

use crate::{
    f2 as f,
    optim2::{Param, ToParams},
    s,
    tensor::{Tensor1, Tensor2, Tensor3},
    util::Cache,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RNN {
    d_model: usize,

    w_i: Tensor2,
    w_r: Tensor2,
    b: Tensor1,

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
}

impl RNN {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,

            w_i: f::xavier((d_model, d_model)),
            w_r: f::xavier((d_model, d_model)),
            b: Tensor1::zeros(d_model),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor3, grad: bool) -> Tensor3 {
        let (batch_size, seq_len, features) = x.dim3();

        assert!(features == self.d_model, "feature dimension != d_model");

        let mut state = Tensor2::zeros((batch_size, features));
        let mut output = Tensor3::zeros_like(&x);

        if grad {
            self.cache
                .set("x", Tensor3::zeros((seq_len, batch_size, features)));
            self.cache
                .set("states", Tensor3::zeros((seq_len, batch_size, features)));
            self.cache.set(
                "preactivations",
                Tensor3::zeros((seq_len, batch_size, features)),
            );
        }

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            let x_i = x_t.dot(&self.w_i);
            let r = state.dot(&self.w_r);

            let preactivations = &x_i + &r + &self.b;
            let activations = f::tanh(&preactivations);

            if grad {
                self.cache["x"].slice_assign(s![t, .., ..], &x_t);
                self.cache["states"].slice_assign(s![t, .., ..], &state);
                self.cache["preactivations"].slice_assign(s![t, .., ..], &preactivations);
            }

            state = activations;
            output.slice_assign(s![.., t, ..], &state);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Tensor3) -> Tensor3 {
        let (batch_size, seq_len, features) = d_loss.dim3();

        let mut d_x = Tensor3::zeros_like(&d_loss);
        let mut resid = Tensor2::zeros((batch_size, features));

        for t in (0..seq_len).rev() {
            let d_loss_t = &d_loss.slice(s![.., t, ..]) + &resid;

            let preactivations = self.cache["preactivations"].slice(s![t, .., ..]);
            let state = self.cache["states"].slice(s![t, .., ..]);
            let x = self.cache["x"].slice(s![t, .., ..]);

            let d_z = &d_loss_t * f::d_tanh(&preactivations);

            self.grads.accumulate("d_wi", x.t().dot(&d_z));
            self.grads.accumulate("d_wr", state.t().dot(&d_z));
            self.grads.accumulate("d_b", d_z.sum_axis(0));

            let d_x_t = d_z.dot(&self.w_i.t());
            d_x.slice_assign(s![.., t, ..], &d_x_t);

            resid = d_z.dot(&self.w_r.t());
        }

        d_x
    }
}

impl ToParams for RNN {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_i).with_grad(&mut self.grads["d_wi"]),
            Param::new(&mut self.w_r).with_grad(&mut self.grads["d_wr"]),
            Param::new(&mut self.b).with_grad(&mut self.grads["d_b"]),
        ]
    }
}

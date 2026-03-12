use serde::{Deserialize, Serialize};

use crate::{
    f2::InitializationFn,
    optim2::{Param, ToParams},
    s,
    tensor::{Tensor2, Tensor3},
    util::Cache,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinearSSM {
    pub d_model: usize,
    pub d_in: usize,
    pub d_out: usize,

    pub a: Tensor2,
    pub b: Tensor2,
    pub c: Tensor2,

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
}

impl LinearSSM {
    pub fn new(
        d_model: usize,
        d_in: usize,
        d_out: usize,
        init_a: InitializationFn,
        init_b: InitializationFn,
        init_c: InitializationFn,
    ) -> Self {
        Self {
            d_model,
            d_in,
            d_out,

            a: init_a((d_model, d_model)),
            b: init_b((d_in, d_model)),
            c: init_c((d_model, d_out)),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor3, grad: bool) -> Tensor3 {
        let (batch_size, seq_len, features) = x.dim3();

        assert!(features == self.d_in, "feature dimension != d_in");

        let mut state = Tensor2::zeros((batch_size, self.d_model));

        if grad {
            self.cache
                .set("x", Tensor3::zeros((seq_len, batch_size, features)));
            self.cache.set(
                "states",
                Tensor3::zeros((seq_len + 1, batch_size, self.d_model)),
            );
        }

        let mut output = Tensor3::zeros((batch_size, seq_len, self.d_out));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            let x_b = x_t.dot(&self.b);
            let r = state.dot(&self.a);

            if grad {
                self.cache["x"].slice_assign(s![t, .., ..], &x_t);
                self.cache["states"].slice_assign(s![t, .., ..], &state);
            }

            state = r + x_b;
            let y = state.dot(&self.c);
            output.slice_assign(s![.., t, ..], &y);
        }

        if grad {
            self.cache["states"].slice_assign(s![seq_len, .., ..], &state);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Tensor3) -> Tensor3 {
        let (batch_size, seq_len, _) = d_loss.dim3();

        println!("d_loss: {:?}", d_loss.shape());

        let mut d_x = Tensor3::zeros((batch_size, seq_len, self.d_in));
        let mut resid = Tensor2::zeros((batch_size, self.d_model));

        for t in (0..seq_len).rev() {
            let d_loss_t = d_loss.slice(s![.., t, ..]);
            let state_t = self.cache["states"].slice(s![t, .., ..]);
            let state_next = self.cache["states"].slice(s![t + 1, .., ..]);
            let x_t = self.cache["x"].slice(s![t, .., ..]);

            self.grads.accumulate("d_c", state_next.t().dot(&d_loss_t));

            let d_state_next = d_loss_t.dot(&self.c.t()) + &resid;
            self.grads.accumulate("d_a", state_t.t().dot(&d_state_next));
            self.grads.accumulate("d_b", x_t.t().dot(&d_state_next));

            let d_x_t = d_state_next.dot(&self.b.t());
            resid = d_state_next.dot(&self.a.t());

            d_x.slice_assign(s![.., t, ..], &d_x_t);
        }

        d_x
    }
}

impl ToParams for LinearSSM {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.a).with_grad(&mut self.grads["d_a"]),
            Param::new(&mut self.b).with_grad(&mut self.grads["d_b"]),
            Param::new(&mut self.c).with_grad(&mut self.grads["d_c"]),
        ]
    }
}

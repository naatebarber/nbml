use ndarray::{Array2, Array3, s};
use serde::{Deserialize, Serialize};

use crate::{
    f::InitializationFn,
    optim::{Param, ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct LinearSSMCache {
    pub x: Array3<f32>,
    pub states: Array3<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct LinearSSMGrads {
    pub d_a: Array2<f32>,
    pub d_b: Array2<f32>,
    pub d_c: Array2<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinearSSM {
    pub d_model: usize,
    pub d_in: usize,
    pub d_out: usize,

    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub c: Array2<f32>,

    #[serde(skip)]
    pub cache: LinearSSMCache,
    #[serde(skip)]
    pub grads: LinearSSMGrads,
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

            cache: LinearSSMCache::default(),
            grads: LinearSSMGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array3<f32>, grad: bool) -> Array3<f32> {
        let (batch_size, seq_len, features) = x.dim();

        assert!(features == self.d_in, "feature dimension != d_in");

        let mut state = Array2::zeros((batch_size, self.d_model));

        if grad {
            self.cache.x = Array3::zeros((seq_len, batch_size, features));
            self.cache.states = Array3::zeros((seq_len + 1, batch_size, self.d_model));
        }

        let mut output = Array3::zeros((batch_size, seq_len, self.d_out));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            let x_b = x_t.dot(&self.b);
            let r = state.dot(&self.a);

            if grad {
                self.cache.x.slice_mut(s![t, .., ..]).assign(&x_t);
                self.cache.states.slice_mut(s![t, .., ..]).assign(&state);
            }

            state = &r + &x_b;
            let y = state.dot(&self.c);
            output.slice_mut(s![.., t, ..]).assign(&y);
        }

        if grad {
            self.cache
                .states
                .slice_mut(s![seq_len, .., ..])
                .assign(&state);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, _) = d_loss.dim();

        println!("d_loss: {:?}", d_loss.dim());

        let mut d_x = Array3::zeros((batch_size, seq_len, self.d_in));

        if self.grads.d_a.dim() == (0, 0) {
            self.grads.d_a = Array2::zeros(self.a.dim());
        }

        if self.grads.d_b.dim() == (0, 0) {
            self.grads.d_b = Array2::zeros(self.b.dim());
        }

        if self.grads.d_c.dim() == (0, 0) {
            self.grads.d_c = Array2::zeros(self.c.dim());
        }

        let mut resid = Array2::zeros((batch_size, self.d_model));

        for t in (0..seq_len).rev() {
            let d_loss_t = d_loss.slice(s![.., t, ..]);
            let state_t = self.cache.states.slice(s![t, .., ..]);
            let state_next = self.cache.states.slice(s![t + 1, .., ..]);
            let x_t = self.cache.x.slice(s![t, .., ..]);

            self.grads.d_c += &state_next.t().dot(&d_loss_t);

            let d_state_next = d_loss_t.dot(&self.c.t()) + &resid;
            self.grads.d_a += &state_t.t().dot(&d_state_next);
            self.grads.d_b += &x_t.t().dot(&d_state_next);

            let d_x_t = d_state_next.dot(&self.b.t());
            resid = d_state_next.dot(&self.a.t());

            d_x.slice_mut(s![.., t, ..]).assign(&d_x_t);
        }

        d_x
    }
}

impl ToParams for LinearSSM {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.a).with_grad(&mut self.grads.d_a),
            Param::new(&mut self.b).with_grad(&mut self.grads.d_b),
            Param::new(&mut self.c).with_grad(&mut self.grads.d_c),
        ]
    }
}

impl ToIntermediates for LinearSSM {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![&mut self.cache.x, &mut self.cache.states]
    }
}

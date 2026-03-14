use ndarray::{Array1, Array2, Array3, Axis, s};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct RNNCache {
    pub x: Array3<f64>,
    pub preactivations: Array3<f64>,
    pub states: Array3<f64>,
}

impl RNNCache {
    pub fn clear(&mut self) {
        *self = RNNCache::default()
    }
}

#[derive(Default, Debug, Clone)]
pub struct RNNGrads {
    pub d_wi: Array2<f64>,
    pub d_wr: Array2<f64>,
    pub d_b: Array1<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RNN {
    pub d_model: usize,

    pub w_i: Array2<f64>,
    pub w_r: Array2<f64>,
    pub b: Array1<f64>,

    #[serde(skip)]
    pub cache: RNNCache,
    #[serde(skip)]
    pub grads: RNNGrads,
}

impl RNN {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,

            w_i: f::xavier((d_model, d_model)),
            w_r: f::xavier((d_model, d_model)),
            b: Array1::zeros(d_model),

            cache: RNNCache::default(),
            grads: RNNGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        assert!(features == self.d_model, "feature dimension != d_model");

        let mut state = Array2::zeros((batch_size, features));
        let mut output = Array3::zeros(x.dim());

        if grad {
            self.cache.x = Array3::zeros((seq_len, batch_size, features));
            self.cache.states = Array3::zeros((seq_len, batch_size, features));
            self.cache.preactivations = Array3::zeros((seq_len, batch_size, features));
        }

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            let x_i = x_t.dot(&self.w_i);
            let r = state.dot(&self.w_r);

            let preactivations = &x_i + &r + &self.b;
            let activations = f::tanh(&preactivations);

            if grad {
                self.cache.x.slice_mut(s![t, .., ..]).assign(&x_t);
                self.cache.states.slice_mut(s![t, .., ..]).assign(&state);
                self.cache
                    .preactivations
                    .slice_mut(s![t, .., ..])
                    .assign(&preactivations);
            }

            state = activations;
            output.slice_mut(s![.., t, ..]).assign(&state);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, features) = d_loss.dim();

        if self.grads.d_wi.dim() == (0, 0) {
            self.grads.d_wi = Array2::zeros(self.w_i.dim());
        }

        if self.grads.d_wr.dim() == (0, 0) {
            self.grads.d_wr = Array2::zeros(self.w_r.dim());
        }

        if self.grads.d_b.dim() == 0 {
            self.grads.d_b = Array1::zeros(self.d_model);
        }

        let mut d_x = Array3::zeros(d_loss.dim());
        let mut resid = Array2::zeros((batch_size, features));

        for t in (0..seq_len).rev() {
            let d_loss_t = &d_loss.slice(s![.., t, ..]) + &resid;

            let preactivations = self.cache.preactivations.slice(s![t, .., ..]).to_owned();
            let state = self.cache.states.slice(s![t, .., ..]);
            let x = self.cache.x.slice(s![t, .., ..]);

            let d_z = &d_loss_t * f::d_tanh(&preactivations);
            self.grads.d_wi += &x.t().dot(&d_z);
            self.grads.d_wr += &state.t().dot(&d_z);
            self.grads.d_b += &d_z.sum_axis(Axis(0));

            let d_x_t = d_z.dot(&self.w_i.t());
            d_x.slice_mut(s![.., t, ..]).assign(&d_x_t);

            resid = d_z.dot(&self.w_r.t());
        }

        d_x
    }
}

impl ToParams for RNN {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::matrix(&mut self.w_i).with_matrix_grad(&mut self.grads.d_wi),
            Param::matrix(&mut self.w_r).with_matrix_grad(&mut self.grads.d_wr),
            Param::vector(&mut self.b).with_vector_grad(&mut self.grads.d_b),
        ]
    }
}

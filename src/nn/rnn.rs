use ndarray::{Array1, Array2, Array3, Axis, s};

use crate::{
    f,
    optim::param::{Param, ToParams},
};

pub struct RNN {
    d_model: usize,

    w_i: Array2<f64>,
    w_r: Array2<f64>,
    b: Array1<f64>,

    x: Array3<f64>,
    preactivations: Array3<f64>,
    states: Array3<f64>,

    d_wi: Array2<f64>,
    d_wr: Array2<f64>,
    d_b: Array1<f64>,
}

impl RNN {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,

            w_i: f::xavier((d_model, d_model)),
            w_r: f::xavier((d_model, d_model)),
            b: Array1::zeros(d_model),

            x: Array3::zeros((0, 0, 0)),
            states: Array3::zeros((0, 0, 0)),
            preactivations: Array3::zeros((0, 0, 0)),

            d_wi: Array2::zeros((0, 0)),
            d_wr: Array2::zeros((0, 0)),
            d_b: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        let mut state = Array2::zeros((batch_size, features));
        let mut output = Array3::zeros(x.dim());

        self.x = if grad {
            Array3::zeros((seq_len, batch_size, features))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.states = if grad {
            Array3::zeros((seq_len, batch_size, features))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.preactivations = if grad {
            Array3::zeros((seq_len, batch_size, features))
        } else {
            Array3::zeros((0, 0, 0))
        };

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            let x_i = x_t.dot(&self.w_i);
            let r = state.dot(&self.w_r);

            let preactivations = &x_i + &r + &self.b;
            let activations = f::tanh(&preactivations);

            if grad {
                self.x.slice_mut(s![t, .., ..]).assign(&x_t);
                self.states.slice_mut(s![t, .., ..]).assign(&state);
                self.preactivations
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

        if self.d_wi.dim() == (0, 0) {
            self.d_wi = Array2::zeros(self.w_i.dim());
        }

        if self.d_wr.dim() == (0, 0) {
            self.d_wr = Array2::zeros(self.w_r.dim());
        }

        if self.d_b.dim() == 0 {
            self.d_b = Array1::zeros(self.d_model);
        }

        let mut d_output = Array3::zeros(d_loss.dim());
        let mut resid = Array2::zeros((batch_size, features));

        for t in (0..seq_len).rev() {
            let d_loss_t = &d_loss.slice(s![.., t, ..]) + &resid;

            let preactivations = self.preactivations.slice(s![t, .., ..]).to_owned();
            let state = self.states.slice(s![t, .., ..]);
            let x = self.x.slice(s![t, .., ..]);

            let d_z = &d_loss_t * f::d_tanh(&preactivations);
            self.d_wi += &x.t().dot(&d_z);
            self.d_wr += &state.t().dot(&d_z);
            self.d_b += &d_z.sum_axis(Axis(0));

            let d_x = d_z.dot(&self.w_i.t());
            d_output.slice_mut(s![.., t, ..]).assign(&d_x);

            resid = d_z.dot(&self.w_r.t());
        }

        d_output
    }
}

impl ToParams for RNN {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::matrix(&mut self.w_i).with_matrix_grad(&mut self.d_wi),
            Param::matrix(&mut self.w_r).with_matrix_grad(&mut self.d_wr),
            Param::vector(&mut self.b).with_vector_grad(&mut self.d_b),
        ]
    }
}

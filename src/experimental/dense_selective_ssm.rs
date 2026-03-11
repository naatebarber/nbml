use ndarray::{Array1, Array2, Array3, s};
use ndarray_rand::{RandomExt, rand_distr::Normal};

use crate::{
    f::{self, xavier_normal},
    optim::param::{Param, ToParams},
};

// discretization rule means dynamic taus per input in the standard eq A' = A.exp(t) where in
// standard LTI SSMs, T is not input dependent. Here T is a learnable function of input u.

// ∆t = softplus(u • t)
// A' = A * exp(∆t)
// B' = B * exp(∆t)
//
// h_t' = h_t • A' + u • B'
// o = h_t' • C

pub struct DenseSelectiveSSM {
    pub d_model: usize,
    pub d_in: usize,
    pub d_out: usize,

    pub t_w: Array2<f64>,
    pub t_b: Array1<f64>,
    pub a: Array1<f64>,
    pub b: Array2<f64>,
    pub c: Array2<f64>,

    pub x: Array3<f64>,
    pub deltas: Array3<f64>,
    pub a_bar: Array3<f64>,
    pub b_bar: Array3<f64>,
    pub states: Array3<f64>,

    pub d_tw: Array2<f64>,
    pub d_tb: Array1<f64>,
    pub d_b: Array2<f64>,
    pub d_c: Array2<f64>,
}

impl DenseSelectiveSSM {
    pub fn new(d_model: usize, d_in: usize, d_out: usize) -> Self {
        let a = (0..d_model)
            .map(|x| -1. * ((x + 1) as f64))
            .collect::<Array1<f64>>();

        Self {
            d_model,
            d_in,
            d_out,

            t_w: xavier_normal((d_in, d_model)),
            t_b: Array1::zeros(d_model),
            a,
            b: Array2::random((d_in, d_model), Normal::new(0., 1e-2).unwrap()),
            c: Array2::random(
                (d_model, d_out),
                Normal::new(0., 1. / (d_model as f64).sqrt()).unwrap(),
            ),

            x: Array3::zeros((0, 0, 0)),
            deltas: Array3::zeros((0, 0, 0)),
            a_bar: Array3::zeros((0, 0, 0)),
            b_bar: Array3::zeros((0, 0, 0)),
            states: Array3::zeros((0, 0, 0)),

            d_tw: Array2::zeros((0, 0)),
            d_tb: Array1::zeros(0),
            d_b: Array2::zeros((0, 0)),
            d_c: Array2::zeros((0, 0)),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        assert!(
            features == self.d_in,
            "dimension mismatch, x_features={features} d_in={}",
            self.d_in
        );

        let mut state = Array2::zeros((batch_size, self.d_model));

        self.x = if grad {
            Array3::zeros((seq_len, batch_size, self.d_in))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.deltas = if grad {
            Array3::zeros((seq_len, batch_size, self.d_model))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.a_bar = if grad {
            Array3::zeros((seq_len, batch_size, self.d_model))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.b_bar = if grad {
            Array3::zeros((seq_len, batch_size, self.d_model))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.states = if grad {
            Array3::zeros((seq_len + 1, batch_size, self.d_model))
        } else {
            Array3::zeros((0, 0, 0))
        };

        let mut output = Array3::zeros((batch_size, seq_len, self.d_out));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);

            let delta_preactivation = x_t.dot(&self.t_w) + &self.t_b;
            let delta = f::softplus(&delta_preactivation); // (B, D)

            let a_bar = (&delta * &self.a).exp();

            let b_disc = (&a_bar - 1.) / &self.a;
            let b = &b_disc * x_t.dot(&self.b);

            if grad {
                self.x.slice_mut(s![t, .., ..]).assign(&x_t);
                self.deltas.slice_mut(s![t, .., ..]).assign(&delta);
                self.a_bar.slice_mut(s![t, .., ..]).assign(&a_bar);
                // self.b_bar.slice_mut(s![t, .., ..]).assign(&b_bar);
                self.states.slice_mut(s![t, .., ..]).assign(&state);
            }

            state = a_bar * &state + &b;

            let o = state.dot(&self.c);
            output.slice_mut(s![.., t, ..]).assign(&o);
        }

        if grad {
            self.states.slice_mut(s![seq_len, .., ..]).assign(&state);
        }

        output
    }
}

impl ToParams for DenseSelectiveSSM {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::matrix(&mut self.t_w).with_matrix_grad(&mut self.d_tw),
            Param::vector(&mut self.t_b).with_vector_grad(&mut self.d_tb),
            Param::matrix(&mut self.b).with_matrix_grad(&mut self.d_b),
            Param::matrix(&mut self.c).with_matrix_grad(&mut self.d_c),
        ]
    }
}

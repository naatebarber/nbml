use ndarray::{Array1, Array2, Array3, Axis, s};
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

pub struct SelectiveSSM {
    pub d_model: usize,
    pub d_in: usize,

    pub t_w: Array2<f64>,
    pub t_b: Array1<f64>,
    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub c: Array2<f64>,

    pub x: Array3<f64>,

    pub d_tw: Array2<f64>,
    pub d_tb: Array1<f64>,
    pub d_b: Array2<f64>,
    pub d_c: Array2<f64>,
}

impl SelectiveSSM {
    pub fn new(d_model: usize, d_in: usize) -> Self {
        let a = (0..d_model)
            .map(|x| -1. * ((x + 1) as f64))
            .collect::<Array1<f64>>()
            .insert_axis(Axis(0))
            .broadcast((d_in, d_model))
            .unwrap()
            .to_owned();

        Self {
            d_model,
            d_in,

            t_w: xavier_normal((d_in, d_in)),
            t_b: Array1::zeros(d_model),
            a,
            b: Array2::random((d_in, d_model), Normal::new(0., 1e-2).unwrap()),
            c: Array2::random(
                (d_in, d_model),
                Normal::new(0., 1. / (d_model as f64).sqrt()).unwrap(),
            ),

            x: Array3::zeros((0, 0, 0)),

            d_tw: Array2::zeros((0, 0)),
            d_tb: Array1::zeros(0),
            d_b: Array2::zeros((0, 0)),
            d_c: Array2::zeros((0, 0)),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, _grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        assert!(
            features == self.d_in,
            "dimension mismatch, x_features={features} d_in={}",
            self.d_in
        );

        let mut state = Array3::zeros((batch_size, self.d_in, self.d_model));

        let mut output = Array3::zeros((batch_size, seq_len, self.d_in));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);

            let b = x_t.dot(&self.b); // (B, N)
            let c = x_t.dot(&self.c); // (B, N)
            let delta_preactivation = x_t.dot(&self.t_w) + &self.t_b; // (B, D)
            let delta = f::softplus(&delta_preactivation); // (B, D)

            // i have deltas of (B, D) and i have A of (D, N). I want shape (B, D, N)
            // for A bar

            // add 0 axis to A, making (1, D, N). add axis deltas of (B, D) to make (B, D, 1)
            // multiply should broadcast

            let a_bar = (&delta.insert_axis(Axis(2)) * &self.a.clone().insert_axis(Axis(0))).exp();

            // i have A bar of (B, D, N) and B of (B, N). I want shape (B, D, N) for B bar.

            // create the b discretization coefficient from A.
            // A bar of shape (B, D, N) - 1. / A of shape (D, N)

            // insert an axis on A to make it (1, D, N)
            // the division should broadcast

            let b_disc = (&a_bar - 1.) / &self.a.clone().insert_axis(Axis(0));

            // i have b discretization coefficient of shape (B, D, N) and B of shape (B, N). I want
            // B bar of shape (B, D, N)

            // insert axis 1 on B. multipication should broadcast

            let b_bar = &b_disc * &b.insert_axis(Axis(1));

            // perform the recurrence elementwise, with no channel mixing.
            // A bar, B bar, and state are of shape (B, D, N)
            // input x is of shape (B, D)

            // insert axis 2 on x, mutliplication with B bar should broadcast

            state = &a_bar * &state + &b_bar * &x_t.insert_axis(Axis(2));

            // i have C of shape (B, N) and state of shape (B, D, N)
            // i want output of shape (B, D)

            // I use C to collapse each channels state dimension to a scalar, resulting in (B, D)
            // Add an axis to C turning (B, N) -> (B, 1, N)
            // Multiply with state (B, D, N) * (B, 1, N) = (B, D, N)
            // sum over axis(2) for dot product (B, D, N) -> (B, D)

            let output_t = (&c.insert_axis(Axis(1)) * &state).sum_axis(Axis(2));
            output.slice_mut(s![.., t, ..]).assign(&output_t);
        }

        output
    }
}

impl ToParams for SelectiveSSM {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::matrix(&mut self.t_w).with_matrix_grad(&mut self.d_tw),
            Param::vector(&mut self.t_b).with_vector_grad(&mut self.d_tb),
            Param::matrix(&mut self.b).with_matrix_grad(&mut self.d_b),
            Param::matrix(&mut self.c).with_matrix_grad(&mut self.d_c),
        ]
    }
}

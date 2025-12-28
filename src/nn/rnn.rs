use core::f64;
use ndarray::{Array1, Array2, Axis};
use std::collections::VecDeque;

use crate::{
    f,
    nn::ffn::FFN,
    optim::param::{Param, ToParams},
};

#[derive(Debug, Clone)]
pub struct RecurrentTrajectory {
    pub input: Array2<f64>,
    pub state: Array2<f64>,
    pub preactivations: Array2<f64>,
    pub activations: Array2<f64>,
    pub d_loss: Array2<f64>,
}

impl RecurrentTrajectory {
    pub fn new() -> Self {
        Self {
            input: Array2::zeros((0, 0)),
            state: Array2::zeros((0, 0)),
            preactivations: Array2::zeros((0, 0)),
            activations: Array2::zeros((0, 0)),
            d_loss: Array2::zeros((0, 0)),
        }
    }
}

pub struct Recurrent {
    pub size: usize,

    pub states: Array2<f64>,
    pub w_i: Array2<f64>,
    pub w_r: Array2<f64>,
    pub b: Array1<f64>,

    pub trajectory: RecurrentTrajectory,
    pub trajectories: VecDeque<RecurrentTrajectory>,

    pub d_wi: Array2<f64>,
    pub d_wr: Array2<f64>,
    pub d_b: Array1<f64>,
}

impl Recurrent {
    pub fn new(size: usize) -> Self {
        Self {
            size,

            states: Array2::zeros((1, size)),
            w_i: f::xavier_normal((size, size)),
            w_r: f::xavier_normal((size, size)),
            b: Array1::zeros(size),

            trajectory: RecurrentTrajectory::new(),
            trajectories: VecDeque::new(),

            d_wi: Array2::zeros((0, 0)),
            d_wr: Array2::zeros((0, 0)),
            d_b: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, x: Array2<f64>, grad: bool) -> Array2<f64> {
        let x_w = x.dot(&self.w_i);
        let r_w = self.states.dot(&self.w_r);
        let preactivations = &x_w + &r_w + &self.b.clone().insert_axis(Axis(0));
        let activations = f::tanh(&preactivations);

        if grad {
            self.trajectory = RecurrentTrajectory {
                input: x.clone(),
                state: self.states.clone(),
                preactivations,
                activations: activations.clone(),
                d_loss: Array2::zeros((1, self.size)),
            };
        }

        self.states = activations;
        self.states.clone()
    }

    pub fn cache(&mut self, d_loss: Array2<f64>, retain: usize) {
        let mut t = self.trajectory.clone();
        t.d_loss = d_loss;
        self.trajectories.push_back(t);

        while self.trajectories.len() > retain {
            self.trajectories.pop_front();
        }
    }

    pub fn backward(&mut self) -> Array2<f64> {
        let steps = self.trajectories.len();

        let mut d_state_next = Array2::zeros((1, self.size));

        for i in (0..steps).rev() {
            let t = &self.trajectories[i];

            d_state_next += &t.d_loss;
            let d_z = &d_state_next * &f::d_tanh(&t.preactivations);

            let d_wi = t.input.t().dot(&d_z);
            let d_wr = t.state.t().dot(&d_z);
            let d_b = d_z.sum_axis(Axis(0));

            self.d_wi = if self.d_wi.dim() == (0, 0) {
                d_wi
            } else {
                &self.d_wi + &d_wi
            };

            self.d_wr = if self.d_wr.dim() == (0, 0) {
                d_wr
            } else {
                &self.d_wr + &d_wr
            };

            self.d_b = if self.d_b.dim() == 0 {
                d_b
            } else {
                &self.d_b + &d_b
            };

            let diag = Array2::from_diag(&f::d_tanh(&t.preactivations).remove_axis(Axis(0)));
            let jac = diag.dot(&self.w_r.t());

            d_state_next = d_state_next.dot(&jac);
        }

        let d_input = d_state_next.dot(&self.w_i.t());
        d_input
    }
}

impl ToParams for Recurrent {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::matrix(&mut self.w_i).with_matrix_grad(&mut self.d_wi),
            Param::matrix(&mut self.w_r).with_matrix_grad(&mut self.d_wr),
            Param::vector(&mut self.b).with_vector_grad(&mut self.d_b),
        ]
    }
}

pub struct RNN {
    pub size: usize,
    pub d_in: usize,
    pub d_out: usize,

    pub projection: FFN,
    pub recurrent: Recurrent,
    pub readout: FFN,
}

impl RNN {
    pub fn new(size: usize, d_in: usize, d_out: usize) -> Self {
        Self {
            size,
            d_in,
            d_out,

            projection: FFN::new(vec![(d_in, size, f::Activation::Identity)]),
            recurrent: Recurrent::new(size),
            readout: FFN::new(vec![(size, d_out, f::Activation::Identity)]),
        }
    }

    pub fn forward(&mut self, x: Array2<f64>, grad: bool) -> Array2<f64> {
        let x = self.projection.forward(x, grad);
        let x = self.recurrent.forward(x, grad);
        self.readout.forward(x, grad)
    }

    pub fn backward_esn(&mut self, d_loss: Array2<f64>) {
        self.readout.backward(d_loss);
    }

    pub fn backward_bptt(&mut self, d_loss: Array2<f64>, retain: usize) -> Array2<f64> {
        let d_readout = self.readout.backward(d_loss);
        self.recurrent.cache(d_readout.clone(), retain);
        let d_recurrent = self.recurrent.backward();
        self.projection.backward(d_recurrent)
    }

    pub fn reset(&mut self) {
        self.recurrent.states = Array2::zeros((1, self.recurrent.size));
    }
}

impl ToParams for RNN {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];
        params.append(&mut self.readout.params());
        params.append(&mut self.recurrent.params());
        params.append(&mut self.projection.params());
        params
    }
}

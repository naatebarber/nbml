use core::f64;
use ndarray::{Array1, Array2, Axis};

use crate::{
    f::{self, Activation},
    nn::ffn::FFN,
    optim::param::{Param, ToParams},
};

// The network is split into three parts: Closure, Readout and Recurrent
// The closure network maps recurrent state to some nonlinear mapping of recurrent state
// This is then:
//  - Fed to the readout for task prediction
//  - Stored as a target for the recurrent network
//
// The readout makes a prediction and receives task error. This error is backpropagated
// through the readout / closure network chain, aligning them with the task.
//
// The recurrent network is then updated such that its recurrent weights (diag of W_R)
// satisfy local hebbian loss (recurrent state - closure map of recurrent state)
// This stretches squeezes the temporal gain / timescales of neurons by how much they contribute to
// or detract from the self-mapping being predictable. This essentially allows the closure network
// to ask for timescales that would help it in its prediction task.
//
// I intended this as an extension of reservoir computing, where the readout and the reservoir
// co-adapt, such that the reservoir learns retention schemes that assist with the readout, while
// the readout learns mappings that leverage the reservoir.

pub struct Closure {
    pub size: usize,
    pub d_out: usize,

    c: FFN,
    r: FFN,

    target: Array2<f64>,
}

impl Closure {
    pub fn new(size: usize, d_out: usize, c_a: Activation, r_a: Activation) -> Self {
        Self {
            size,
            d_out,

            c: FFN::new(vec![(size, size, c_a)]),
            r: FFN::new(vec![(size, d_out, r_a)]),

            target: Array2::zeros((1, size)),
        }
    }

    pub fn forward(&mut self, state: Array2<f64>) -> Array2<f64> {
        self.target = self.c.forward(state, true);
        self.r.forward(self.target.clone(), true)
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) -> Array2<f64> {
        let d_r = self.r.backward(d_loss);
        self.c.backward(d_r)
    }
}

impl ToParams for Closure {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];

        params.append(&mut self.c.params());
        params.append(&mut self.r.params());

        params
    }
}

pub struct Recurrent {
    pub size: usize,

    pub states: Array2<f64>,
    pub w_i: Array2<f64>,
    pub w_r: Array2<f64>,
    pub b: Array1<f64>,

    pub _input: Array2<f64>,
    pub _states: Array2<f64>,
    pub _x_w: Array2<f64>,
    pub _r_w: Array2<f64>,
    pub _preactivations: Array2<f64>,
    pub _activations: Array2<f64>,

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

            _input: Array2::zeros((1, size)),
            _states: Array2::zeros((1, size)),
            _x_w: Array2::zeros((size, size)),
            _r_w: Array2::zeros((size, size)),
            _preactivations: Array2::zeros((size, size)),
            _activations: Array2::zeros((size, size)),

            d_wi: Array2::zeros((size, size)),
            d_wr: Array2::zeros((size, size)),
            d_b: Array1::zeros(size),
        }
    }

    pub fn forward(&mut self, x: Array2<f64>, grad: bool) -> Array2<f64> {
        let x_w = x.dot(&self.w_i);
        let r_w = self.states.dot(&self.w_r);
        let preactivations = &x_w + &r_w + &self.b.clone().insert_axis(Axis(0));
        let activations = f::tanh(&preactivations);

        if grad {
            self._input = x.clone();
            self._states = self.states.clone();
            self._x_w = x_w;
            self._r_w = r_w;
            self._preactivations = preactivations;
            self._activations = activations.clone();
        }

        self.states = activations;
        self.states.clone()
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) -> Array2<f64> {
        let d_z = &d_loss * &f::d_tanh(&self._preactivations);
        let d_wi = Array2::from_diag(&(&d_z * &self._input).remove_axis(Axis(0))).to_owned();
        let d_wr = Array2::from_diag(&(&d_z * &self._states).remove_axis(Axis(0))).to_owned();
        let d_b = d_z.sum_axis(Axis(0));

        self.d_wi = d_wi;
        self.d_wr = d_wr;
        self.d_b = d_b;

        let d_input = d_z.dot(&self.w_i.t());
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

pub struct ClosureNet {
    pub projection: FFN,
    pub recurrent: Recurrent,
    pub closure: Closure,
}

impl ClosureNet {
    pub fn new(
        d_in: usize,
        d_hidden: usize,
        d_out: usize,
        c_a: Activation,
        r_a: Activation,
    ) -> Self {
        Self {
            projection: FFN::new(vec![(d_in, d_hidden, f::Activation::Identity)]),
            recurrent: Recurrent::new(d_hidden),
            closure: Closure::new(d_hidden, d_out, c_a, r_a),
        }
    }

    pub fn forward(&mut self, x: Array2<f64>, grad: bool) -> Array2<f64> {
        let p = self.projection.forward(x, grad);
        let r = self.recurrent.forward(p, grad);
        let c = self.closure.forward(r);

        c
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) {
        self.closure.backward(d_loss);

        let d_closure = 2. * (&self.recurrent.states - &self.closure.target);
        self.recurrent.backward(d_closure);

        // updating the projection
        // only adds noise to the system and degrades performance.
        // let d_projection = self.projection.backward(d_target_mse);
        // d_projection
    }

    pub fn reset(&mut self) {
        self.recurrent.states = Array2::zeros((1, self.recurrent.size))
    }
}

impl ToParams for ClosureNet {
    fn params(&mut self) -> Vec<Param> {
        let mut params = vec![];
        params.append(&mut self.projection.params());
        params.append(&mut self.recurrent.params());
        params.append(&mut self.closure.params());
        params
    }
}

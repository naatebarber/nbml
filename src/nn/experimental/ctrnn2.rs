use std::collections::VecDeque;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

use crate::{
    f,
    optim::param::{Param, ToParams},
};

pub struct CTRNN {
    pub size: usize,
    pub d_in: usize,
    pub d_out: usize,

    pub input_layer: Array2<f64>,
    pub states: Array1<f64>,
    pub w: Array2<f64>,
    pub b: Array1<f64>,
    pub taus: Array1<f64>,
    pub output_layer: Array2<f64>,

    pub _dts: VecDeque<f64>,
    pub _states: VecDeque<Array1<f64>>,
    pub _preactivations: VecDeque<Array1<f64>>,
    pub _activations: VecDeque<Array1<f64>>,
    pub _drifts: VecDeque<Array1<f64>>,
    pub _d_loss: VecDeque<Option<Array1<f64>>>,

    pub d_w: Array2<f64>,
    pub d_b: Array1<f64>,
    pub d_taus: Array1<f64>,
}

impl CTRNN {
    pub fn new(size: usize, d_in: usize, d_out: usize) -> CTRNN {
        Self {
            size,
            d_in,
            d_out,

            input_layer: f::xavier_normal((d_in, size)),
            states: Array1::zeros(size),
            w: f::xavier_normal((size, size)),
            b: Array1::zeros(size),
            taus: Array1::random(size, Uniform::new(0.01, 3.)),
            output_layer: f::xavier_normal((size, d_out)),

            _dts: VecDeque::new(),
            _states: VecDeque::new(),
            _preactivations: VecDeque::new(),
            _activations: VecDeque::new(),
            _drifts: VecDeque::new(),
            _d_loss: VecDeque::new(),

            d_w: Array2::zeros((size, size)),
            d_b: Array1::zeros(size),
            d_taus: Array1::zeros(size),
        }
    }

    pub fn euler_step(&mut self, x: Array1<f64>, dt: f64, grad: bool) {
        // r = dt/T
        // s+1 = r * W • a(s + b) + x

        let r = dt / &self.taus;
        let preactivations = &self.states + &self.b;
        let activations =
            f::tanh(&preactivations.clone().insert_axis(Axis(0))).remove_axis(Axis(0));
        let recurrent = self
            .w
            .dot(&activations.clone().insert_axis(Axis(1)))
            .remove_axis(Axis(1));
        let drift = recurrent + &x;

        if grad {
            self._dts.push_back(dt);
            self._states.push_back(self.states.clone());
            self._preactivations.push_back(preactivations);
            self._activations.push_back(activations);
            self._drifts.push_back(drift.clone());
            self._d_loss.push_back(None);
        }

        self.states = &self.states + &r * &drift;
    }

    pub fn forward(&mut self, x: Array2<f64>, dt: f64, step_size: f64, grad: bool) -> Array2<f64> {
        let mut s = step_size;

        let hidden = x.dot(&self.input_layer).remove_axis(Axis(0));

        self.euler_step(hidden, step_size, grad);
        while s < dt {
            self.euler_step(Array1::zeros(self.size), step_size, grad);
            s += step_size;
        }

        self.states
            .clone()
            .insert_axis(Axis(0))
            .dot(&self.output_layer)
    }

    pub fn backward_step(&mut self, d_loss: &Array1<f64>, t: usize, steps: usize) {
        let step_weight = 1. / steps as f64;

        // r = dt/T
        // s+1 = r * W • a(s + b) + x

        let dt = &self._dts[t];
        let preactivations = &self._preactivations[t];
        let activations = &self._activations[t];
        let drift = &self._drifts[t];

        let r = *dt / &self.taus;

        let dr_dt = -dt / self.taus.powi(2);
        let dy_dr = drift;
        let dl_dt = d_loss * dy_dr * dr_dt;
        self.d_taus += &(&dl_dt * step_weight);

        // How w shaped the rate of change of activations at timestep T
        // Im taking the derivative with RESPECT TO W, being derived, W goes from rank 1 to rank 0.
        // im left with the outer product between the two things it multiplies with: r and a
        //
        // To do this i need the outer product between activations (input to W) and (rate * loss)
        // (multiple to the output of W)
        let dl_dw = activations
            .clone()
            .insert_axis(Axis(1))
            .dot(&(&r * d_loss).insert_axis(Axis(0)));
        self.d_w += &(&dl_dw * step_weight);

        let d_activation =
            f::d_tanh(&preactivations.clone().insert_axis(Axis(0))).remove_axis(Axis(0));

        // input is d_loss, first multipled by the outside term Rate, then multipled against the
        // transpose of W to undo its change, then against the derivative of the activation
        // function.
        let dl_db = (d_loss * r).dot(&self.w.t()) * d_activation;
        self.d_b += &(&dl_db * step_weight);
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) {
        let d_loss_pre = d_loss.dot(&self.output_layer.t()).remove_axis(Axis(0));

        let Some(last) = self._d_loss.back_mut() else {
            panic!("No gradient tracking");
        };

        *last = Some(d_loss_pre);

        let steps = self._dts.len();
        let mut d_state_next = Array1::zeros(self.size);
        for t in (0..steps).rev() {
            let d_loss = &self._d_loss[t];
            if let Some(d_loss) = d_loss {
                d_state_next += d_loss;
            }

            self.backward_step(&d_state_next, t, steps);

            let dt = self._dts[t];
            let preactivations = &self._preactivations[t];
            let r = dt / &self.taus;
            let d_activation =
                f::d_tanh(&(preactivations.clone()).insert_axis(Axis(0))).remove_axis(Axis(0));

            let diag_d_activation = Array2::from_diag(&d_activation);
            let diag_r = Array2::from_diag(&r);
            let jacobian =
                &Array2::<f64>::eye(self.size) + diag_r.dot(&self.w.dot(&diag_d_activation));

            d_state_next = jacobian
                .dot(&d_state_next.insert_axis(Axis(1)))
                .remove_axis(Axis(1));
        }
    }

    pub fn retain_steps(&mut self, n: usize) {
        let steps = self._dts.len();

        if n > steps {
            return;
        }

        let d = steps - n;
        self._dts.drain(0..d);
        self._states.drain(0..d);
        self._preactivations.drain(0..d);
        self._activations.drain(0..d);
        self._drifts.drain(0..d);
    }

    pub fn zero_grad(&mut self) {
        self.d_b = Array1::zeros(self.size);
        self.d_w = Array2::zeros((self.size, self.size));
    }
}

impl ToParams for CTRNN {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        let mut params = vec![];
        params.push(Param::from_array2(&mut self.w, &mut self.d_w));
        params.push(Param::from_array1(&mut self.b, &mut self.d_b));
        params.push(Param::from_array1(&mut self.taus, &mut self.d_taus));
        params
    }
}

use core::f64;
use ndarray::{Array1, Array2, Axis, Slice, concatenate};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

use crate::{
    f::{self, Activation},
    optim::param::{Param, ToParams},
};

pub struct CTRNN {
    size: usize,
    a: Activation,
    d: Activation,
    tau_min: f64,
    tau_max: f64,

    pub states: Array1<f64>,
    weights: Array2<f64>,
    biases: Array1<f64>,
    taus: Array1<f64>,

    d_loss: Array2<f64>,
    step_dt: Vec<f64>,
    x_prev: Array2<f64>,
    preactivations: Array2<f64>,
    activations: Array2<f64>,
    dx_dt: Array2<f64>,

    d_taus: Array1<f64>,
    d_weights: Array2<f64>,
    d_biases: Array1<f64>,
}

impl CTRNN {
    pub fn new(size: usize, a: Activation, d: Activation) -> Self {
        let tau_min = 0.01;
        let tau_max = 3.;

        Self {
            size,
            a,
            d,
            tau_min,
            tau_max,

            states: Array1::zeros(size),
            weights: Array2::random((size, size), Uniform::new(-0.1, 0.1)),
            biases: Array1::zeros(size),
            taus: Array1::random(size, Uniform::new(tau_min, tau_max)),

            d_loss: Array2::zeros((0, size)),
            step_dt: Vec::new(),
            x_prev: Array2::zeros((0, size)),
            preactivations: Array2::zeros((0, size)),
            activations: Array2::zeros((0, size)),
            dx_dt: Array2::zeros((0, size)),

            d_taus: Array1::zeros(size),
            d_weights: Array2::zeros((size, size)),
            d_biases: Array1::zeros(size),
        }
    }

    pub fn log_taus(&mut self, low: f64, high: f64) {
        self.tau_min = low;
        self.tau_max = high;

        self.taus = Array1::random(
            self.size,
            Uniform::new(self.tau_min.ln(), self.tau_max.ln()),
        );
    }

    pub fn uniform_taus(&mut self, low: f64, high: f64) {
        self.tau_min = low;
        self.tau_max = high;

        self.taus = Array1::random(self.size, Uniform::new(self.tau_min, self.tau_max));
    }

    pub fn euler_step(&mut self, input: &Array1<f64>, dt: f64) {
        self.step_dt.push(dt);
        self.x_prev.push(Axis(0), self.states.view()).unwrap();
        self.d_loss
            .push(Axis(0), Array1::zeros(self.size).view())
            .unwrap();

        let preactivations = &self.states + &self.biases;
        self.preactivations
            .push(Axis(0), preactivations.view())
            .unwrap();

        let activations = (self.a)(&preactivations.insert_axis(Axis(0))).remove_axis(Axis(0));
        self.activations.push(Axis(0), activations.view()).unwrap();

        let recurrent = self.weights.dot(&activations);
        let update = &recurrent + input;
        self.dx_dt.push(Axis(0), update.view()).unwrap();

        let smoothed_taus = self.tau_min + self.taus.exp();
        let rate = dt / smoothed_taus;

        self.states = (&self.states + update * &rate) / (1. + &rate);

        if self.states.is_any_nan() {
            panic!("exploded");
        }
    }

    pub fn fit(&self, input: &Array1<f64>) -> Array1<f64> {
        let frame = Array1::zeros(self.size);
        frame + input
    }

    pub fn forward(&mut self, mut input: Array1<f64>, dt: f64, step_size: f64) {
        input = self.fit(&input);

        assert!(input.len() == self.size);

        let mut t = 0.;

        while t < dt {
            self.euler_step(&input, step_size);
            t += step_size;
        }
    }

    pub fn backward_step(&mut self, d_loss: &Array1<f64>, t: usize) {
        // Differentiate loss w.r.t. taus.
        //
        // st = t_min + T.exp()
        //
        // y = (S + U * (dt/st)) / (1 + dt/st)
        // r = dt/st
        // y = (S + U * r) / (1 + r)
        //
        // ∂y∂r = (U - S) / (1 + r)^2
        // ∂r∂T = -dt/st^2
        // ∂st∂T = T.exp()
        // ∂y∂T = ∂y∂r * ∂r∂st * ∂st∂T
        // ∂L∂T = ∂L * ∂y∂T

        let dt = self.step_dt[t];
        let x_prev = self.x_prev.row(t);
        let preactivation = self.preactivations.row(t);
        let dx_dt = self.dx_dt.row(t);

        let exp_taus = self.taus.exp();
        let smoothed_taus = self.tau_min + &exp_taus;
        let r = dt / &smoothed_taus;

        let dy_dr = (&dx_dt - &x_prev) / (1. + &r).powi(2);
        let dr_dst = -dt / &smoothed_taus.powi(2);
        let dst_dt = exp_taus;

        let dy_dt = dy_dr * dr_dst * dst_dt;
        let dl_dt = d_loss * dy_dt;
        self.d_taus = dl_dt;

        // Differentiate loss w.r.t. W

        // y = (S + U * r) / (1 + r)
        // y = (S / 1 + r) + (r / (1 + r)) * U
        // U = W • O + I
        //
        // ∂y∂U = (r / (1 + r))
        // ∂U∂W = O.t
        // ∂y∂W = ∂y∂U * ∂

        let r_factor = &r / (1. + &r);
        let grad_scale = d_loss * &r_factor; // (S)

        let grad_w = grad_scale
            .clone()
            .insert_axis(Axis(1))
            .dot(&dx_dt.clone().insert_axis(Axis(0)));

        if self.d_weights.len() == 0 {
            self.d_weights = grad_w;
        } else {
            self.d_weights += &grad_w;
        }

        // Differentiate loss w.r.t. B

        let d_biased =
            (self.d)(&preactivation.to_owned().insert_axis(Axis(0))).remove_axis(Axis(0));

        let grad_b = &self.weights.t().dot(&grad_scale) * d_biased;

        if self.d_biases.len() == 0 {
            self.d_biases = grad_b;
        } else {
            self.d_biases += &grad_b;
        }
    }

    pub fn backward(&mut self, d_loss: Array1<f64>) {
        let steps = self.step_dt.len();
        self.d_loss.row_mut(steps - 1).assign(&d_loss);

        let mut d_state_next: Array1<f64> = Array1::zeros(self.size);

        for t in (0..self.step_dt.len()).rev() {
            let d_t = &self.d_loss.row(t) + &d_state_next;
            self.backward_step(&d_t, t);

            // find ∂x+1/∂x
            // by taking partial derivative of update rule w.r.t. x.

            let dt = self.step_dt[t];
            let exp_taus = self.taus.exp();
            let smoothed_taus = self.tau_min + exp_taus;
            let r = dt / smoothed_taus;

            let a = 1. / (1. + &r);
            let b = &r / (1. + &r);

            // ∂x_{t+1}/∂x_t ≈ 1 - (dt/τ) + (dt/τ)*f'(x_t)
            // for small dt, often approximated as 1/(1 + r)
            // but we will be deriving the whole thing
            //
            // update rule is
            // s+1 = a * s + b * ((W • sig(s + b)) + i)
            // ∂W = W.T.dot()
            // ∂s+1/∂s = a + b * (W.T • d_sig(s + b))

            let d_sig =
                f::d_sigmoid(&(&self.x_prev.row(t).view() + &self.biases).insert_axis(Axis(0)))
                    .remove_axis(Axis(0)); // (S,)
            let a_term = &a * &d_t; // (S,)
            let b_term = &b * &d_t; // (S,)
            let recurrent = b_term.dot(&self.weights); // (S,)
            d_state_next = a_term + recurrent * &d_sig; // (S,)
        }
    }

    pub fn retain_grads(&mut self, n: usize) {
        if self.step_dt.len() < n {
            return;
        }

        let slice_at = self.step_dt.len() - n;
        self.step_dt = self.step_dt.split_off(slice_at);
        self.x_prev = self
            .x_prev
            .slice_axis(Axis(0), Slice::from(slice_at..))
            .to_owned();
        self.preactivations = self
            .preactivations
            .slice_axis(Axis(0), Slice::from(slice_at..))
            .to_owned();
        self.activations = self
            .activations
            .slice_axis(Axis(0), Slice::from(slice_at..))
            .to_owned();
        self.dx_dt = self
            .dx_dt
            .slice_axis(Axis(0), Slice::from(slice_at..))
            .to_owned();
    }

    pub fn zero_grads(&mut self) {
        self.step_dt = Vec::new();
        self.x_prev = Array2::zeros((0, self.size));
        self.preactivations = Array2::zeros((0, self.size));
        self.activations = Array2::zeros((0, self.size));
        self.dx_dt = Array2::zeros((0, self.size));
    }

    pub fn state(&self) -> Array1<f64> {
        self.states.clone()
    }

    pub fn slice_front(&self, n: usize) -> Array1<f64> {
        if n > self.size {
            panic!("front neural slice greater than network size");
        }

        self.states
            .slice_axis(Axis(0), Slice::from(0..n))
            .to_owned()
    }

    pub fn slice_back(&self, n: usize) -> Array1<f64> {
        if n > self.size {
            panic!("back neural slice greater than network size");
        }

        self.states
            .slice_axis(Axis(0), Slice::from((self.size - n)..self.size))
            .to_owned()
    }

    pub fn concat_front(&self, x: Array1<f64>) -> Array1<f64> {
        if x.len() > self.size {
            panic!("front neural concat greater than network size");
        }

        let diff = self.size - x.len();
        let zeros = Array1::zeros(diff);
        concatenate(Axis(0), &[x.view(), zeros.view()]).unwrap()
    }

    pub fn concat_back(&self, x: Array1<f64>) -> Array1<f64> {
        if x.len() > self.size {
            panic!("back neural concat greater than network size");
        }

        let diff = self.size - x.len();
        let zeros = Array1::zeros(diff);
        concatenate(Axis(0), &[zeros.view(), x.view()]).unwrap()
    }

    pub fn reset(&mut self) {
        self.zero_grads();
        self.states = Array1::zeros(self.size);
    }
}

impl ToParams for CTRNN {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::from_array1(&mut self.taus, &mut self.d_taus),
            Param::from_array2(&mut self.weights, &mut self.d_weights),
            Param::from_array1(&mut self.biases, &mut self.d_biases),
        ]
    }
}

use crate::f::{self, Activation};
use crate::ndarray::{Array1, Array2};
use crate::ndarray_rand::{RandomExt, rand_distr::Uniform};
use crate::nn::FFN;
use crate::optim::param::ToParams;
use core::f64;
use ndarray::Axis;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SNNReservoir {
    pub d_in: usize,
    pub d_hidden: usize,

    pub w_p: Array2<f64>,
    pub taus: Array1<f64>,
    pub thresholds: Array1<f64>,
    pub w_r: Array2<f64>,
}

impl SNNReservoir {
    pub fn new(d_in: usize, d_hidden: usize) -> Self {
        Self {
            d_in,
            d_hidden,

            w_p: f::xavier((d_in, d_hidden)),
            taus: Array1::random(d_hidden, Uniform::new(0., 1.)),
            thresholds: Array1::random(d_hidden, Uniform::new(0., 1.)),
            w_r: f::xavier((d_hidden, d_hidden)),
        }
    }

    pub fn set_threshold_range(&mut self, low: f64, high: f64) {
        self.thresholds = Array1::random(self.d_hidden, Uniform::new(low, high));
    }

    pub fn set_tau_range(&mut self, low: f64, high: f64) {
        self.taus = Array1::random(self.d_hidden, Uniform::new(low, high));
    }

    pub fn set_spectral_radius(&mut self, desired: f64, n: usize) -> f64 {
        let lambda = f::calculate_spectral_radius(&self.w_r, n);
        self.w_r *= desired / lambda;
        f::calculate_spectral_radius(&self.w_r, n)
    }

    pub fn step(&self, x: &Array2<f64>, state: &mut Array2<f64>, delta: f64) {
        let decay = self.taus.map(|t| (-delta / t).exp()).insert_axis(Axis(0));
        *state *= &decay;
        *state += &x.dot(&self.w_p);

        let spikes = (&*state - &self.thresholds).mapv(|x| if x > 0. { 1. } else { 0. });

        let p = &*state * &spikes;

        *state -= &p;
        *state += &p.dot(&self.w_r);
    }
}

pub struct LSM {
    pub reservoir: SNNReservoir,
    readout: FFN,
}

impl LSM {
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize) -> Self {
        Self {
            reservoir: SNNReservoir::new(d_in, d_hidden),
            readout: FFN::new(vec![(d_hidden, d_out, Activation::Identity)]),
        }
    }

    pub fn set_readout(&mut self, readout: FFN) {
        self.readout = readout;
    }

    pub fn step(
        &mut self,
        x: &Array2<f64>,
        delta: f64,
        state: &mut Array2<f64>,
        grad: bool,
    ) -> Array2<f64> {
        self.reservoir.step(&x, state, delta);
        self.readout.forward(state.clone(), grad)
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) {
        self.readout.backward(d_loss);
    }
}

impl ToParams for LSM {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        self.readout.params()
    }
}

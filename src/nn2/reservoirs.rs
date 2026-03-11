use serde::{Deserialize, Serialize};

use crate::{
    f2 as f, s,
    tensor::{Tensor1, Tensor2, Tensor3},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RNNReservoir {
    pub d_in: usize,
    pub d_hidden: usize,

    pub w_p: Tensor2,
    pub w_r: Tensor2,
    pub b: Tensor1,
}

impl RNNReservoir {
    pub fn new(d_in: usize, d_hidden: usize) -> Self {
        Self {
            d_in,
            d_hidden,

            w_p: f::xavier((d_in, d_hidden)),
            w_r: f::xavier((d_hidden, d_hidden)),
            b: Tensor1::zeros(d_hidden),
        }
    }

    pub fn set_spectral_radius(&mut self, desired: f64, n: usize) -> f64 {
        let lambda = f::calculate_spectral_radius(&self.w_r, n);
        self.w_r *= desired / lambda;
        f::calculate_spectral_radius(&self.w_r, n)
    }

    pub fn forward(&self, x: Tensor3) -> Tensor3 {
        let (batch_size, seq_len, features) = x.dim3();

        assert!(features == self.d_in, "feature dimension != d_in");

        let mut state = Tensor2::zeros((batch_size, self.d_hidden));
        let mut encoded = Tensor3::zeros((batch_size, seq_len, self.d_hidden));

        let x_p = x
            .reshape((batch_size * seq_len, features))
            .dot(&self.w_p)
            .reshape((batch_size, seq_len, self.d_hidden));

        for t in 0..seq_len {
            let x_p_t = x_p.slice(s![.., t, ..]);
            let r = state.dot(&self.w_r);

            let preactivations = &x_p_t + &r + &self.b;
            let activations = f::tanh(&preactivations);

            state = activations;
            encoded.slice_assign(s![.., t, ..], &state);
        }

        encoded
    }

    pub fn step(&self, x: &Tensor2, h: &mut Tensor2) {
        let x_p = x.dot(&self.w_p);
        let r = h.dot(&self.w_r);

        let preactivations = &x_p + &r + &self.b;
        let activations = f::tanh(&preactivations);
        *h = activations;
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SNNReservoir {
    pub d_in: usize,
    pub d_hidden: usize,

    pub w_p: Tensor2,
    pub taus: Tensor1,
    pub thresholds: Tensor1,
    pub w_r: Tensor2,
}

impl SNNReservoir {
    pub fn new(d_in: usize, d_hidden: usize) -> Self {
        Self {
            d_in,
            d_hidden,

            w_p: f::xavier((d_in, d_hidden)),
            taus: Tensor1::random_uniform(d_hidden),
            thresholds: Tensor1::random_uniform(d_hidden),
            w_r: f::xavier((d_hidden, d_hidden)),
        }
    }

    pub fn set_threshold_range(&mut self, low: f64, high: f64) {
        self.thresholds = Tensor1::random_uniform(self.d_hidden) * (high - low) + low;
    }

    pub fn set_tau_range(&mut self, low: f64, high: f64) {
        self.taus = Tensor1::random_uniform(self.d_hidden) * (high - low) + low;
    }

    pub fn set_spectral_radius(&mut self, desired: f64, n: usize) -> f64 {
        let lambda = f::calculate_spectral_radius(&self.w_r, n);
        self.w_r *= desired / lambda;
        f::calculate_spectral_radius(&self.w_r, n)
    }

    pub fn step(&self, x: &Tensor2, state: &mut Tensor2, delta: f64) {
        let decay = self.taus.mapv(|t| (-delta / t).exp()).insert_axis(0);
        *state *= &decay;
        *state += &x.dot(&self.w_p);

        let spikes = (state as &Tensor2 - &self.thresholds).mapv(|x| if x > 0. { 1. } else { 0. });

        let p = (state as &Tensor2) * &spikes;

        *state -= &p;
        *state += &p.dot(&self.w_r);
    }
}

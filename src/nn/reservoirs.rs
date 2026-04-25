use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToParams},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RNNReservoir {
    pub d_in: usize,
    pub d_hidden: usize,

    pub w_p: Array2<f32>,
    pub w_r: Array2<f32>,
    pub b: Array1<f32>,
}

impl RNNReservoir {
    pub fn new(d_in: usize, d_hidden: usize) -> Self {
        Self {
            d_in,
            d_hidden,

            w_p: f::xavier((d_in, d_hidden)),
            w_r: f::xavier((d_hidden, d_hidden)),
            b: Array1::zeros(d_hidden),
        }
    }

    pub fn set_spectral_radius(&mut self, desired: f32, n: usize) -> f32 {
        let lambda = f::calculate_spectral_radius(&self.w_r, n);
        self.w_r *= desired / lambda;
        f::calculate_spectral_radius(&self.w_r, n)
    }

    pub fn forward(&self, x: Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, features) = x.dim();

        assert!(features == self.d_in, "feature dimension != d_in");

        let mut state = Array2::zeros((batch_size, self.d_hidden));
        let mut encoded = Array3::zeros((batch_size, seq_len, self.d_hidden));

        let x_2d = x
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();
        let x_p_2d = x_2d.dot(&self.w_p);
        let x_p = x_p_2d
            .into_shape_clone((batch_size, seq_len, self.d_hidden))
            .unwrap();

        for t in 0..seq_len {
            let x_p_t = x_p.slice(s![.., t, ..]);
            let r = state.dot(&self.w_r);

            let preactivations = &x_p_t + &r + &self.b;
            let activations = f::tanh(&preactivations);

            state = activations;
            encoded.slice_mut(s![.., t, ..]).assign(&state);
        }

        encoded
    }

    pub fn step(&self, x: &Array2<f32>, h: &mut Array2<f32>) {
        let x_p = x.dot(&self.w_p);
        let r = h.dot(&self.w_r);

        let preactivations = &x_p + &r + &self.b;
        let activations = f::tanh(&preactivations);
        *h = activations;
    }
}

impl ToParams for RNNReservoir {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_p),
            Param::new(&mut self.w_r),
            Param::new(&mut self.b),
        ]
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SNNReservoir {
    pub d_in: usize,
    pub d_hidden: usize,

    pub w_p: Array2<f32>,
    pub taus: Array1<f32>,
    pub thresholds: Array1<f32>,
    pub w_r: Array2<f32>,
}

impl SNNReservoir {
    pub fn new(d_in: usize, d_hidden: usize) -> Self {
        Self {
            d_in,
            d_hidden,

            w_p: f::xavier((d_in, d_hidden)),
            taus: Array1::random(d_hidden, Uniform::new(0., 1.).unwrap()),
            thresholds: Array1::random(d_hidden, Uniform::new(0., 1.).unwrap()),
            w_r: f::xavier((d_hidden, d_hidden)),
        }
    }

    pub fn set_threshold_range(&mut self, low: f32, high: f32) {
        self.thresholds = Array1::random(self.d_hidden, Uniform::new(low, high).unwrap());
    }

    pub fn set_tau_range(&mut self, low: f32, high: f32) {
        self.taus = Array1::random(self.d_hidden, Uniform::new(low, high).unwrap());
    }

    pub fn set_spectral_radius(&mut self, desired: f32, n: usize) -> f32 {
        let lambda = f::calculate_spectral_radius(&self.w_r, n);
        self.w_r *= desired / lambda;
        f::calculate_spectral_radius(&self.w_r, n)
    }

    pub fn step(&self, x: &Array2<f32>, state: &mut Array2<f32>, delta: f32) {
        let decay = self.taus.map(|t| (-delta / t).exp()).insert_axis(Axis(0));
        *state *= &decay;
        *state += &x.dot(&self.w_p);

        let spikes = (&*state - &self.thresholds).mapv(|x| if x > 0. { 1. } else { 0. });

        let p = &*state * &spikes;

        *state -= &p;
        *state += &p.dot(&self.w_r);
    }
}

impl ToParams for SNNReservoir {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        vec![
            Param::new(&mut self.w_p),
            Param::new(&mut self.taus),
            Param::new(&mut self.thresholds),
            Param::new(&mut self.w_r),
        ]
    }
}

use ndarray::{Array1, Array2, Array3, s};

use crate::{
    f::{self, Activation},
    nn::FFN,
    optim::param::ToParams,
};

pub struct RNNReservoir {
    pub d_in: usize,
    pub d_hidden: usize,

    pub w_p: Array2<f64>,
    pub w_r: Array2<f64>,
    pub b: Array1<f64>,
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

    pub fn set_spectral_radius(&mut self, desired: f64, n: usize) -> f64 {
        let lambda = f::calculate_spectral_radius(&self.w_r, n);
        self.w_r *= desired / lambda;
        f::calculate_spectral_radius(&self.w_r, n)
    }

    pub fn forward(&self, x: Array3<f64>) -> Array3<f64> {
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

    pub fn step(&self, x: &Array2<f64>, h: &mut Array2<f64>) {
        let x_p = x.dot(&self.w_p);
        let r = h.dot(&self.w_r);

        let preactivations = &x_p + &r + &self.b;
        let activations = f::tanh(&preactivations);
        *h = activations;
    }
}

pub struct ESN {
    pub d_in: usize,
    pub d_hidden: usize,
    pub d_out: usize,

    pub reservoir: RNNReservoir,
    pub readout: FFN,
}

impl ESN {
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize) -> Self {
        Self {
            d_in,
            d_hidden,
            d_out,

            reservoir: RNNReservoir::new(d_in, d_hidden),
            readout: FFN::new(vec![(d_hidden, d_out, Activation::Identity)]),
        }
    }

    pub fn set_readout(&mut self, readout: FFN) {
        self.d_out = readout.layers.last().unwrap().b.dim();
        self.readout = readout;
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, _) = x.dim();
        let encoded = self.reservoir.forward(x);

        let encoded_2d = encoded
            .into_shape_clone((batch_size * seq_len, self.d_hidden))
            .unwrap();
        let output_2d = self.readout.forward(encoded_2d, grad);
        let output = output_2d
            .into_shape_clone((batch_size, seq_len, self.d_out))
            .unwrap();

        output
    }

    pub fn step(&mut self, x: &Array2<f64>, h: &mut Array2<f64>, grad: bool) -> Array2<f64> {
        self.reservoir.step(x, h);
        self.readout.forward(h.clone(), grad)
    }

    pub fn backward(&mut self, d_loss: Array3<f64>) {
        let (batch_size, seq_len, features) = d_loss.dim();

        let d_loss_2d = d_loss
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();
        self.readout.backward(d_loss_2d);
    }
}

impl ToParams for ESN {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        self.readout.params()
    }
}

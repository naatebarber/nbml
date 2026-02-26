use ndarray::{Array1, Array2, Array3, s};

use crate::{
    f::{self, Activation},
    nn::FFN,
    optim::param::ToParams,
};

pub struct ESN {
    d_in: usize,
    d_hidden: usize,
    d_out: usize,

    w_p: Array2<f64>,
    w_r: Array2<f64>,
    b: Array1<f64>,
    readout: FFN,
}

impl ESN {
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize) -> Self {
        Self {
            d_in,
            d_hidden,
            d_out,

            w_p: f::xavier((d_in, d_hidden)),
            w_r: f::xavier((d_hidden, d_hidden)),
            b: Array1::zeros(d_hidden),
            readout: FFN::new(vec![
                (d_hidden, 2 * d_hidden, Activation::Relu),
                (2 * d_hidden, d_out, Activation::Identity),
            ]),
        }
    }

    pub fn set_spectral_radius(&mut self, desired: f64, n: usize) -> f64 {
        let lambda = f::calculate_spectral_radius(&self.w_r, n);
        self.w_r *= desired / lambda;
        f::calculate_spectral_radius(&self.w_r, n)
    }

    pub fn set_output_activation(&mut self, activation: Activation) {
        self.readout.layers[1].activation = activation;
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
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
        let x_p = x.dot(&self.w_p);
        let r = h.dot(&self.w_r);

        let preactivations = &x_p + &r + &self.b;
        let activations = f::tanh(&preactivations);
        *h = activations;

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

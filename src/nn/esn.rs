use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::{
    f::Activation,
    nn::{FFN, RNNReservoir},
    optim::{Param, ToIntermediates, ToParams},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
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
    fn params(&mut self) -> Vec<Param> {
        self.readout.params()
    }
}

impl ToIntermediates for ESN {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        self.readout.intermediates()
    }
}

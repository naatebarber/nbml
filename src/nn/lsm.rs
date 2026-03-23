use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    f::Activation,
    nn::{FFN, SNNReservoir},
    optim::{ToIntermediates, ToParams},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LSM {
    pub reservoir: SNNReservoir,
    pub readout: FFN,
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
        x: &Array2<f32>,
        delta: f32,
        state: &mut Array2<f32>,
        grad: bool,
    ) -> Array2<f32> {
        self.reservoir.step(&x, state, delta);
        self.readout.forward(state.clone(), grad)
    }

    pub fn backward(&mut self, d_loss: Array2<f32>) {
        self.readout.backward(d_loss);
    }
}

impl ToParams for LSM {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        self.readout.params()
    }
}

impl ToIntermediates for LSM {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        self.readout.intermediates()
    }
}

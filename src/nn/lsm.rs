use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    f::xavier_normal,
    layers::Linear,
    nn::SNNReservoir,
    optim::{ToIntermediates, ToParams},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LSM {
    pub reservoir: SNNReservoir,
    pub readout: Linear,
}

impl LSM {
    pub fn new(d_in: usize, d_hidden: usize, d_out: usize) -> Self {
        Self {
            reservoir: SNNReservoir::new(d_in, d_hidden),
            readout: Linear::new_with_init(d_hidden, d_out, xavier_normal),
        }
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
        let mut params = vec![];
        params.append(&mut self.reservoir.params());
        params.append(&mut self.readout.params());
        params
    }
}

impl ToIntermediates for LSM {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        self.readout.intermediates()
    }
}

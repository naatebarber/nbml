use serde::{Deserialize, Serialize};

use crate::{
    optim2::ToParams,
    tensor::{Tensor2, Tensor3},
    util::Cache,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SequencePooling {
    #[serde(skip)]
    pub cache: Cache,
}

impl SequencePooling {
    pub fn new() -> Self {
        Self {
            cache: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor3, mask: Tensor3, grad: bool) -> Tensor2 {
        let seq_lens = mask.sum_axis(1).insert_axis(1);

        if grad {
            self.cache.set("mask", mask.clone());
            self.cache.set("seq_lens", seq_lens.clone());
        }

        let x_sums = (&x * &mask.insert_axis(2)).sum_axis(1);

        &x_sums / &seq_lens
    }

    pub fn backward(&mut self, d_loss: Tensor2) -> Tensor3 {
        let (batch_size, features) = d_loss.dim2();
        let (.., seq_len) = self.cache["mask"].dim2();

        let d_loss_mean = &d_loss / &self.cache["seq_lens"];

        let d_loss_3d = d_loss_mean
            .insert_axis(1)
            .broadcast((batch_size, seq_len, features));

        let d_loss_masked = &d_loss_3d * self.cache["mask"].clone().insert_axis(2);

        d_loss_masked
    }
}

impl ToParams for SequencePooling {
    fn params(&mut self) -> Vec<crate::optim2::Param> {
        vec![]
    }
}

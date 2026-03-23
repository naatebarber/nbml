use ndarray::{Array2, Array3, s};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use serde::{Deserialize, Serialize};

use crate::optim::{Param, ToParams};

#[derive(Default, Debug, Clone)]
pub struct EmbeddingCache {
    pub x: Array2<usize>,
}

#[derive(Default, Debug, Clone)]
pub struct EmbeddingGrads {
    pub d_weights: Array2<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Embedding {
    pub vocab_size: usize,
    pub d_model: usize,

    pub weights: Array2<f32>,

    #[serde(skip)]
    pub cache: EmbeddingCache,
    #[serde(skip)]
    pub grads: EmbeddingGrads,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self {
            vocab_size,
            d_model,

            weights: Array2::random((vocab_size, d_model), Normal::new(0., 0.02).unwrap()),

            cache: EmbeddingCache::default(),
            grads: EmbeddingGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array2<usize>, grad: bool) -> Array3<f32> {
        let mut embedding = Array3::zeros((x.dim().0, x.dim().1, self.d_model));

        for b in 0..x.dim().0 {
            for (t, token) in x.row(b).iter().enumerate() {
                embedding
                    .slice_mut(s![b, t, ..])
                    .assign(&self.weights.row(*token));
            }
        }

        if grad {
            self.cache.x = x;
        }

        embedding
    }

    pub fn backward(&mut self, d_loss: Array3<f32>) {
        let mut d_weights = Array2::zeros(self.weights.dim());

        if self.grads.d_weights.dim() == (0, 0) {
            self.grads.d_weights = Array2::zeros(self.weights.dim())
        }

        for b in 0..self.cache.x.dim().0 {
            for (t, token_id) in self.cache.x.row(b).iter().enumerate() {
                let d_loss_token = d_loss.slice(s![b, t, ..]);
                let mut d_weight_t = d_weights.slice_mut(s![*token_id, ..]);
                d_weight_t += &d_loss_token;
            }
        }

        self.grads.d_weights += &d_weights;
    }
}

impl ToParams for Embedding {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        vec![Param::new(&mut self.weights).with_grad(&mut self.grads.d_weights)]
    }
}

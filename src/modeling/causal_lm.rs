use ndarray::{Array2, Array3};

use crate::{
    f,
    layers::{Embedding, Linear},
    optim::ToParams,
};

pub trait CausalSequenceModel: ToParams {
    fn new(d_model: usize) -> Self;
    fn forward(&mut self, x: Array3<f32>, mask: Array2<f32>, grad: bool) -> Array3<f32>;
    fn backward(&mut self, d_loss: Array3<f32>) -> Array3<f32>;
}

pub struct CausalLM<M> {
    pub vocab_size: usize,
    pub d_model: usize,

    pub embedding: Embedding,
    pub model: M,
    pub lm_head: Linear,
}

impl<M: CausalSequenceModel> CausalLM<M> {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self {
            vocab_size,
            d_model,

            embedding: Embedding::new(vocab_size, d_model),
            model: M::new(d_model),
            lm_head: Linear::new(d_model, vocab_size),
        }
    }

    pub fn forward(&mut self, x: Array2<usize>, mask: Array2<f32>, grad: bool) -> Array3<f32> {
        let x_embedding = self.embedding.forward(x, grad);

        let (batch_size, seq_len, features) = x_embedding.dim();

        let x_model = self.model.forward(x_embedding, mask, grad);
        let x_model_2d = x_model
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();
        let x_lm_head = self.lm_head.forward(x_model_2d, grad);
        x_lm_head
            .into_shape_clone((batch_size, seq_len, self.vocab_size))
            .unwrap()
    }

    pub fn backward(&mut self, d_loss: Array3<f32>) {
        let (batch_size, seq_len, _) = d_loss.dim();

        let d_loss_2d = d_loss
            .into_shape_clone((batch_size * seq_len, self.vocab_size))
            .unwrap();
        let d_lm_head_2d = self.lm_head.backward(d_loss_2d);
        let d_lm_head = d_lm_head_2d
            .into_shape_clone((batch_size, seq_len, self.d_model))
            .unwrap();
        let d_model = self.model.backward(d_lm_head);
        self.embedding.backward(d_model);
    }

    pub fn train_step(&mut self, x: Array2<usize>, y: Array2<usize>, mask: Array2<f32>) -> f32 {
        let x_logits = self.forward(x, mask, true);
        let (loss, d_loss) = f::cross_entropy_loss(x_logits, &y);
        self.backward(d_loss);
        loss
    }
}

impl<M: CausalSequenceModel> ToParams for CausalLM<M> {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        let mut params = vec![];
        params.append(&mut self.embedding.params());
        params.append(&mut self.model.params());
        params.append(&mut self.lm_head.params());
        params
    }
}

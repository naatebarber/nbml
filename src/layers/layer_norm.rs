use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

use crate::optim::{Param, ToIntermediates, ToParams};

#[derive(Default, Debug, Clone)]
pub struct LayerNormCache {
    pub o: Array2<f32>,
    pub x_h: Array2<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct LayerNormGrads {
    pub d_gamma: Array1<f32>,
    pub d_beta: Array1<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,

    #[serde(skip)]
    pub cache: LayerNormCache,
    #[serde(skip)]
    pub grads: LayerNormGrads,
}

impl LayerNorm {
    pub fn new(d_in: usize) -> LayerNorm {
        LayerNorm {
            gamma: Array1::ones(d_in),
            beta: Array1::zeros(d_in),

            cache: LayerNormCache::default(),
            grads: LayerNormGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array3<f32>, grad: bool) -> Array3<f32> {
        let (batch_size, seq_len, feature_size) = x.dim();

        let x = x
            .into_shape_clone((batch_size * seq_len, feature_size))
            .unwrap();

        let m = (1. / feature_size as f32) * x.sum_axis(Axis(1)).insert_axis(Axis(1));
        let u = &x - &m;
        let v = (1. / feature_size as f32)
            * &(u.clone().powi(2)).sum_axis(Axis(1)).insert_axis(Axis(1));
        let o = (&v + 1e-5).sqrt(); // (B * S, 1)

        let x_h = &u / &o;
        let y_2 = (&x_h * &self.gamma) + &self.beta;

        if grad {
            self.cache.o = o;
            self.cache.x_h = x_h;
        }

        y_2.into_shape_clone((batch_size, seq_len, feature_size))
            .unwrap()
    }

    pub fn backward(&mut self, d_loss: Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, features) = d_loss.dim();
        let d_loss = d_loss
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();

        let d_gamma = (&d_loss * &self.cache.x_h).sum_axis(Axis(0));
        let d_beta = d_loss.sum_axis(Axis(0));

        self.grads.d_gamma = if self.grads.d_gamma.dim() == 0 {
            d_gamma
        } else {
            &self.grads.d_gamma + &d_gamma
        };

        self.grads.d_beta = if self.grads.d_beta.dim() == 0 {
            d_beta
        } else {
            &self.grads.d_beta + &d_beta
        };

        let dx_hat = &d_loss * &self.gamma;
        let dx = (1. / (features as f32 * &self.cache.o))
            * (features as f32 * &dx_hat
                - &dx_hat.sum_axis(Axis(1)).insert_axis(Axis(1))
                - &self.cache.x_h
                    * (&dx_hat * &self.cache.x_h)
                        .sum_axis(Axis(1))
                        .insert_axis(Axis(1)));

        dx.into_shape_clone((batch_size, seq_len, features))
            .unwrap()
    }
}

impl ToParams for LayerNorm {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.gamma).with_grad(&mut self.grads.d_gamma),
            Param::new(&mut self.beta).with_grad(&mut self.grads.d_beta),
        ]
    }
}

impl ToIntermediates for LayerNorm {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![&mut self.cache.o, &mut self.cache.x_h]
    }
}

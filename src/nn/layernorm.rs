use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

use crate::optim::{cache::Cache, param::{Param, ToParams}};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Array1<f64>,
    pub beta: Array1<f64>,

    #[serde(skip)]
    pub cache: Cache,

    pub d_gamma: Array1<f64>,
    pub d_beta: Array1<f64>,
}

impl LayerNorm {
    pub fn new(d_in: usize) -> LayerNorm {
        LayerNorm {
            gamma: Array1::ones(d_in),
            beta: Array1::zeros(d_in),

            cache: Cache::new(),

            d_gamma: Array1::zeros(0),
            d_beta: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, feature_size) = x.dim();

        let x = x
            .into_shape_clone((batch_size * seq_len, feature_size))
            .unwrap();

        let m = (1. / feature_size as f64) * x.sum_axis(Axis(1)).insert_axis(Axis(1));
        let u = &x - &m;
        let v = (1. / feature_size as f64)
            * &(u.clone().powi(2)).sum_axis(Axis(1)).insert_axis(Axis(1));
        let o = (&v + 1e-5).sqrt(); // (B * S, 1)

        let x_h = &u / &o;
        let y_2 = (&x_h * &self.gamma) + &self.beta;

        if grad {
            self.cache.set("o", o);
            self.cache.set("x_h", x_h);
        }

        y_2.into_shape_clone((batch_size, seq_len, feature_size))
            .unwrap()
    }

    pub fn backward(&mut self, d_loss: Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, features) = d_loss.dim();
        let d_loss = d_loss
            .into_shape_clone((batch_size * seq_len, features))
            .unwrap();

        let x_h = self.cache.get::<Array2<f64>>("x_h");
        let o = self.cache.get::<Array2<f64>>("o");

        let d_gamma = (&d_loss * x_h).sum_axis(Axis(0));
        let d_beta = d_loss.sum_axis(Axis(0));

        self.d_gamma = if self.d_gamma.dim() == 0 {
            d_gamma
        } else {
            &self.d_gamma + &d_gamma
        };

        self.d_beta = if self.d_beta.dim() == 0 {
            d_beta
        } else {
            &self.d_beta + &d_beta
        };

        let dx_hat = &d_loss * &self.gamma;
        let dx = (1. / (features as f64 * o))
            * (features as f64 * &dx_hat
                - &dx_hat.sum_axis(Axis(1)).insert_axis(Axis(1))
                - x_h * (&dx_hat * x_h).sum_axis(Axis(1)).insert_axis(Axis(1)));

        dx.into_shape_clone((batch_size, seq_len, features))
            .unwrap()
    }
}

impl ToParams for LayerNorm {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::vector(&mut self.gamma).with_vector_grad(&mut self.d_gamma),
            Param::vector(&mut self.beta).with_vector_grad(&mut self.d_beta),
        ]
    }
}

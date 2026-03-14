use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use crate::optim::{Optimizer, ToParams};

pub type AdamParam = Option<(ArrayD<f64>, ArrayD<f64>)>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdamW {
    pub learning_rate: f64,
    pub clip_grad: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub t: usize,

    pub params: Vec<AdamParam>,
}

impl Default for AdamW {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            learning_rate: 3e-4,
            clip_grad: 1.0,
            weight_decay: 0.001,
            epsilon: 1e-8,
            t: 1,
            params: vec![],
        }
    }
}

impl Optimizer for AdamW {
    fn with(mut self, optimizable: &mut impl ToParams) -> Self {
        let params = optimizable.params();

        for param in params {
            self.params.push(match param.target {
                Some(target) => Some((ArrayD::zeros(target.dim()), ArrayD::zeros(target.dim()))),
                None => None,
            });
        }

        self
    }

    fn step(&mut self, optimizable: &mut impl ToParams) {
        self.t += 1;
        let bc1 = self.beta1.powi(self.t as i32);
        let bc2 = self.beta2.powi(self.t as i32);

        unsafe {
            for (param, adam_param) in optimizable.params().into_iter().zip(self.params.iter_mut())
            {
                match (adam_param, param.target, param.grad) {
                    (Some((m, v)), Some(target), Some(grad)) => {
                        assert!(
                            target.dim() == grad.dim(),
                            "attempted to update target of dim {:?} with grad of dim {:?}",
                            target.dim(),
                            grad.dim()
                        );

                        let mut grad_view = grad.deref_into_view_mut();
                        let mut target_view = target.deref_into_view_mut();

                        // Clip grad
                        let norm_sq = grad_view.mapv(|x| x * x).sum();
                        let norm = norm_sq.sqrt();

                        if norm > self.clip_grad {
                            grad_view.mapv_inplace(|x| x * (self.clip_grad / (norm + 1e-6)));
                        }

                        *m = self.beta1 * &(*m) + (1. - self.beta1) * &grad_view;
                        *v = self.beta2 * &(*v) + (1. - self.beta2) * grad_view.powi(2);

                        let m_hat = &(*m) / (1. - bc1);
                        let v_hat = &(*v) / (1. - bc2);

                        let delta = m_hat / (v_hat.sqrt() + self.epsilon);

                        if param.enable_weight_decay {
                            target_view -= &(self.learning_rate
                                * &(&delta + self.weight_decay * &target_view));
                        } else {
                            target_view -= &(&delta * self.learning_rate);
                        }
                    }
                    _ => (),
                }
            }
        }
    }
}

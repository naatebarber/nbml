use serde::{Deserialize, Serialize};

use crate::tensor::Float;
use crate::{Tensor, f2 as f};

use crate::optim2::{Optimizer, ToParams};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdamParam(Tensor, Tensor);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdamW {
    pub learning_rate: Float,
    pub clip_grad: Float,
    pub beta1: Float,
    pub beta2: Float,
    pub epsilon: Float,
    pub weight_decay: Float,
    pub t: usize,

    pub params: Vec<Option<AdamParam>>,
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

        unsafe {
            for param in params {
                let adam_param = match (&param.target, &param.grad) {
                    (Some(target), Some(_)) => Some(AdamParam(
                        Tensor::zeros_like(&(**target)),
                        Tensor::zeros_like(&(**target)),
                    )),
                    _ => None,
                };

                self.params.push(adam_param);
            }
        }

        self
    }

    fn step(&mut self, optimizable: &mut impl ToParams) {
        self.t += 1;
        let bc1 = self.beta1.powi(self.t as i32);
        let bc2 = self.beta2.powi(self.t as i32);

        unsafe {
            for (param, mut adam_param) in
                optimizable.params().into_iter().zip(self.params.iter_mut())
            {
                match (&param.target, &param.grad, &mut adam_param) {
                    (Some(target), Some(grad), Some(AdamParam(m, v))) => {
                        let g = (*grad).to_owned();
                        let t = (*target).to_owned();
                        let g = f::clip_grad((*g).clone(), self.clip_grad);

                        *m = self.beta1 * &(*m) + (1. - self.beta1) * &g;
                        *v = self.beta2 * &(*v) + (1. - self.beta2) * g.powi(2);

                        let m_hat = &(*m) / (1. - bc1);
                        let v_hat = &(*v) / (1. - bc2);

                        let delta = m_hat / (v_hat.sqrt() + self.epsilon);

                        if param.enable_weight_decay {
                            *t -= &(self.learning_rate * &(&delta + self.weight_decay * &(*t)));
                        } else {
                            *t -= &(&delta * self.learning_rate);
                        }
                    }
                    _ => (),
                }
            }
        }
    }
}

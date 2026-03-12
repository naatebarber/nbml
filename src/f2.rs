use crate::s;
use crate::tensor::{Float, Tensor};
use rand::RngExt;
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};

pub type ActivationFn = fn(&Tensor) -> Tensor;
pub type InitializationFn = fn((usize, usize)) -> Tensor;

// ACTIVATIONS

pub fn relu(x: &Tensor) -> Tensor {
    x.mapv(|v| v.max(0.))
}

pub fn d_relu(x: &Tensor) -> Tensor {
    x.mapv(|v| v.signum().max(0.))
}

pub fn tanh(x: &Tensor) -> Tensor {
    x.mapv(|v| v.tanh())
}

pub fn d_tanh(x: &Tensor) -> Tensor {
    x.mapv(|v| 1. - v.tanh().powi(2))
}

pub fn leaky_relu(x: &Tensor) -> Tensor {
    x.mapv(|x| if x >= 0. { x } else { 0.01 * x })
}

pub fn d_leaky_relu(x: &Tensor) -> Tensor {
    x.mapv(|x| if x >= 0. { 1. } else { 0.01 })
}

pub fn exp(x: &Tensor) -> Tensor {
    x.exp()
}

pub fn d_exp(x: &Tensor) -> Tensor {
    x.exp()
}

pub fn elu(x: &Tensor) -> Tensor {
    let x_pos = x.mapv(|v| v.max(0.));
    let x_neg = x.mapv(|v| v.min(0.));
    &x_pos + (x_neg.exp() - 1.)
}

pub fn d_elu(x: &Tensor) -> Tensor {
    let x_pos = x.mapv(|v| if v >= 0. { 1. } else { 0. });
    let x_neg = x.mapv(|v| v.min(0.));
    &x_pos + x_neg.exp()
}

pub fn softplus(x: &Tensor) -> Tensor {
    x.mapv(|v| if v > 20.0 { v } else { (1.0 + v.exp()).ln() })
}

pub fn d_softplus(x: &Tensor) -> Tensor {
    sigmoid(x)
}

pub fn ident(x: &Tensor) -> Tensor {
    x.clone()
}

pub fn d_ident(x: &Tensor) -> Tensor {
    Tensor::ones(x.shape())
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn d_sigmoid(x: &Tensor) -> Tensor {
    let s = sigmoid(x);
    let one_minus_s = s.mapv(|v| 1. - v);
    &s * &one_minus_s
}

pub fn softmax(x: &Tensor) -> Tensor {
    let maxes = x.max_axis(1).insert_axis(1);
    let d = (x - &maxes).exp();
    let sums = d.sum_axis(1).insert_axis(1);
    &d / &sums
}

pub fn d_softmax(s: &Tensor, g: &Tensor) -> Tensor {
    let dot = (g * s).sum_axis(1).insert_axis(1);
    s * &(g - &dot)
}

pub fn d_softmax_cross_entropy(x: &Tensor) -> Tensor {
    Tensor::ones(x.shape())
}

// INITS

pub fn he(shape: (usize, usize)) -> Tensor {
    let bound = Float::sqrt(6.) / Float::sqrt(shape.0 as Float);
    Tensor::random_uniform(shape) * (2. * bound) - bound
}

pub fn xavier_normal(shape: (usize, usize)) -> Tensor {
    let std = (2. / ((shape.0 + shape.1) as Float)).sqrt();
    Tensor::random_normal(shape) * std
}

pub fn xavier(shape: (usize, usize)) -> Tensor {
    let bound = (6. / (shape.0 as Float + shape.1 as Float)).sqrt();
    Tensor::random_uniform(shape) * (2. * bound) - bound
}

// LOSSES

pub fn cross_entropy_loss(probs: &Tensor, targets: &Tensor) -> Float {
    let log_probs = probs.mapv(|p| p.max(1e-10).ln());
    let (batch, seq, _) = probs.dim3();
    -(targets * &log_probs).sum() / (batch * seq) as Float
}

// MISC

pub fn linear_norm(x: &Tensor) -> Tensor {
    let sum = x.sum_axis(1).insert_axis(1);
    x / &sum
}

pub fn softmax_vector_jacobian_product(upstream: &Tensor, softmax_out: &Tensor) -> Tensor {
    let dot = (upstream * softmax_out).sum_axis(1).insert_axis(1);
    softmax_out * &(upstream - &dot)
}

pub fn positional_encoding_seq(seq_len: usize, features: usize) -> Tensor {
    let mut pe = Tensor::zeros((seq_len, features));
    let n: Float = 10_000.;

    for k in 0..seq_len {
        for i in 0..(features / 2) {
            let exp = n.powf((2.0 * i as Float) / features as Float);
            pe[[k, 2 * i]] = Float::sin((k as Float) / exp);
            if (2 * i + 1) < features {
                pe[[k, 2 * i + 1]] = Float::cos((k as Float) / exp);
            }
        }
    }

    pe
}

pub fn log_softmax(x: &Tensor) -> Tensor {
    let maxes = x.max_axis(1).insert_axis(1);
    let d = x - &maxes;
    let sums = d.exp().sum_axis(1).insert_axis(1).ln();
    &d - &sums
}

pub fn l2(v: &Tensor) -> Float {
    v.powi(2).sum().sqrt()
}

pub fn l2_norm(v: &Tensor) -> Tensor {
    v / l2(v)
}

pub fn clip_grad(grad: Tensor, clip: Float) -> Tensor {
    let norm = grad.mapv(|x| x * x).sum().sqrt();
    if norm > clip {
        grad.mapv(|x| x * (clip / (norm + 1e-6)))
    } else {
        grad
    }
}

pub fn causal_mask(n: usize) -> Tensor {
    let mut mask = Tensor::zeros((n, n));

    for i in 0..n {
        mask.slice_assign(s![i as isize, 0..(i + 1) as isize], &Tensor::ones(i + 1));
    }

    mask
}

pub fn sample_categorical(probs: &Tensor, rng: &mut ThreadRng) -> usize {
    let data = probs.to_vec();
    let mut u: Float = rng.random();

    for (i, &p) in data.iter().enumerate() {
        if u < p {
            return i;
        }
        u -= p;
    }

    0
}

pub fn sample_gumbel_categorical(log_probs: &Tensor, rng: &mut ThreadRng) -> usize {
    let gumbel: Vec<Float> = (0..log_probs.len())
        .map(|_| -Float::ln(-Float::ln(rng.random())))
        .collect();
    let u = Tensor::from_vec(log_probs.shape().to_vec(), gumbel);
    (log_probs + &u).argmax()[0]
}

pub fn gaussian_log_prob(x: &Tensor, u: &Tensor, o: &Tensor) -> Tensor {
    let eps = 1e-8;
    let quadratic = (x - u).powi(2) / o.powi(2);
    let norm = o.mapv(|v| v.max(eps).ln()) * 2.;
    let constant = (core::f64::consts::PI as Float * 2.).ln();
    let dims = (quadratic + norm + constant) * -0.5;
    dims.sum_axis(1)
}

pub fn tanh_gaussian_correction(a_raw: &Tensor) -> Tensor {
    let d = d_tanh(a_raw);
    d.ln().sum_axis(1)
}

pub fn tanh_gaussian_correction_eps(a_raw: &Tensor) -> Tensor {
    let deriv = a_raw.mapv(|u| 1.0 - u.tanh().powi(2));
    deriv.mapv(|v| (v + 1e-8).ln()).sum_axis(1)
}

pub fn calculate_spectral_radius(w_r: &Tensor, n: usize) -> Float {
    let dim = w_r.dim2().0;
    let mut h = Tensor::ones((1, dim));
    let mut radius = 0.;
    for _ in 0..n {
        h = h.dot(w_r);
        radius = h.powi(2).sum().sqrt();
        h /= radius;
    }
    radius
}

// ENUMS

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Activation {
    Relu,
    LeakyRelu,
    SoftmaxCrossEntropy,
    Identity,
    Exp,
    Sigmoid,
    Tanh,
    Softplus,
}

impl Activation {
    pub fn wake(&self) -> (ActivationFn, ActivationFn, InitializationFn) {
        match self {
            Self::Relu => (relu, d_relu, he),
            Self::LeakyRelu => (leaky_relu, d_leaky_relu, he),
            Self::SoftmaxCrossEntropy => (softmax, d_softmax_cross_entropy, xavier),
            Self::Identity => (ident, d_ident, he),
            Self::Exp => (exp, d_exp, he),
            Self::Sigmoid => (sigmoid, d_sigmoid, xavier),
            Self::Tanh => (tanh, d_tanh, xavier_normal),
            Self::Softplus => (softplus, d_softplus, xavier),
        }
    }
}

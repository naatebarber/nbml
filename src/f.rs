use core::f64;

use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray_stats::QuantileExt;
use rand::{Rng, rngs::ThreadRng};
use serde::{Deserialize, Serialize};

pub type ActivationFn = fn(&Array2<f64>) -> Array2<f64>;
pub type InitializationFn = fn((usize, usize)) -> Array2<f64>;

// ACTIVATIONS

pub fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.))
}

pub fn d_relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.signum().max(0.))
}

pub fn tanh(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.tanh())
}

pub fn d_tanh(x: &Array2<f64>) -> Array2<f64> {
    1. - (x.mapv(|v| v.tanh())).powi(2)
}

pub fn leaky_relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|x| if x >= 0. { x } else { 0.01 * x })
}

pub fn d_leaky_relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|x| if x >= 0. { 1. } else { 0.01 })
}

pub fn exp(x: &Array2<f64>) -> Array2<f64> {
    x.exp()
}

pub fn d_exp(x: &Array2<f64>) -> Array2<f64> {
    x.exp()
}

pub fn softplus(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 20.0 { v } else { (1.0 + v.exp()).ln() })
}

pub fn d_softplus(x: &Array2<f64>) -> Array2<f64> {
    sigmoid(x)
}

pub fn ident(x: &Array2<f64>) -> Array2<f64> {
    x.to_owned()
}

pub fn d_ident(x: &Array2<f64>) -> Array2<f64> {
    Array2::ones(x.dim())
}

pub fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn d_sigmoid(x: &Array2<f64>) -> Array2<f64> {
    let s = sigmoid(x);
    &s * &(1.0 - &s)
}

pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let maxes = x
        .map_axis(Axis(1), |row| row.max().cloned().unwrap_or(1e-4))
        .insert_axis(Axis(1));

    let mut d = x - &maxes;

    d.mapv_inplace(|x| x.exp());

    let sums = d.map_axis(Axis(1), |row| row.sum()).insert_axis(Axis(1));

    let last = &d / &sums;
    return last;
}

pub fn d_softmax_cross_entropy(x: &Array2<f64>) -> Array2<f64> {
    x.to_owned()
}

// ACTIVATIONS

pub fn he(shape: (usize, usize)) -> Array2<f64> {
    let bound = f64::sqrt(6.) / f64::sqrt(shape.0 as f64);
    return Array2::random(shape, Uniform::new(-bound, bound));
}

pub fn xavier_normal(shape: (usize, usize)) -> Array2<f64> {
    let std = (2. / ((shape.0 + shape.1) as f64)).sqrt();
    Array2::random(
        shape,
        ndarray_rand::rand_distr::Normal::new(0., std).unwrap(),
    )
}

pub fn xavier(shape: (usize, usize)) -> Array2<f64> {
    let bound = (6. / (shape.0 as f64 + shape.1 as f64)).sqrt();
    return Array2::random(shape, Uniform::new(-bound, bound));
}

// MISC

pub fn linear_norm(x: &Array2<f64>) -> Array2<f64> {
    let sum = x.sum_axis(Axis(1)).insert_axis(Axis(1));
    x / &sum
}

pub fn softmax_vector_jacobian_product(
    upstream: &Array2<f64>,
    softmax_out: &Array2<f64>,
) -> Array2<f64> {
    let mut grad = upstream.clone();

    for ((mut g_row, s_row), u_row) in grad
        .axis_iter_mut(Axis(0))
        .zip(softmax_out.axis_iter(Axis(0)))
        .zip(upstream.axis_iter(Axis(0)))
    {
        let dot = u_row.dot(&s_row);

        for ((g, &s), &u) in g_row.iter_mut().zip(s_row.iter()).zip(u_row.iter()) {
            *g = s * (u - dot);
        }
    }

    grad
}

pub fn positional_encoding_seq(seq_len: usize, features: usize) -> Array2<f64> {
    assert!(
        features % 2 == 0,
        "positional encoding: features must be even, got {}",
        features
    );
    let mut pe = Array2::zeros((seq_len, features));
    let n: f64 = 10_000.;

    for k in 0..seq_len {
        for i in 0..(features / 2) {
            let exp = n.powi(((2 * i) / features) as i32);
            pe[[k, 2 * i]] = f64::sin((k as f64) / exp);
            pe[[k, 2 * i + 1]] = f64::cos((k as f64) / exp);
        }
    }

    pe
}

// TODO: Use with Gumbel max sampling, avoid standard softmax entirely.
pub fn log_softmax(x: Array2<f64>) -> Array2<f64> {
    let maxes = x
        .map_axis(Axis(1), |row| row.max().cloned().unwrap_or(1e-4))
        .insert_axis(Axis(1));

    let d = &x - &maxes;
    let sums = d
        .map_axis(Axis(1), |row| row.exp().sum())
        .insert_axis(Axis(1));

    &d - sums.ln()
}

pub fn l2(v: &Array1<f64>) -> f64 {
    v.pow2().sum().sqrt()
}

pub fn l2_norm(v: &Array1<f64>) -> Array1<f64> {
    v / l2(v)
}

pub fn clip_grad(mut grad: Array2<f64>, clip: f64) -> Array2<f64> {
    let norm_sq = grad.mapv(|x| x * x).sum();
    let norm = norm_sq.sqrt();

    if norm > clip {
        grad.mapv_inplace(|x| x * (clip / (norm + 1e-6)));
    }

    grad
}

pub fn causal_mask(n: usize) -> Array2<f64> {
    let mut mask = Array2::zeros((n, n));

    for i in 0..n {
        mask.slice_mut(s![i, 0..(i + 1)])
            .assign(&Array1::ones(i + 1));
    }

    mask
}

pub fn sample_categorical(probs: &Array1<f64>, rng: &mut ThreadRng) -> usize {
    let mut u: f64 = rng.random();

    for (i, &p) in probs.iter().enumerate() {
        if u < p {
            return i;
        }

        u -= p;
    }

    return 0;
}

pub fn sample_gumbel_categorical(log_probs: &Array1<f64>, rng: &mut ThreadRng) -> usize {
    let u = (0..log_probs.len())
        .map(|_| -f64::ln(-f64::ln(rng.random())))
        .collect::<Array1<f64>>();

    (log_probs + u).argmax().unwrap()
}

pub fn gaussian_log_prob(x: &Array2<f64>, u: &Array2<f64>, o: &Array2<f64>) -> Array1<f64> {
    let eps = 1e-8;
    let quadratic = (x - u).powi(2) / o.powi(2);
    let norm = 2. * o.map(|v| v.max(eps).ln());
    let constant = (f64::consts::PI * 2.).ln();

    let dims = -0.5 * (quadratic + norm + constant);

    dims.sum_axis(Axis(1))
}

pub fn tanh_gaussian_correction(a_raw: &Array2<f64>) -> Array1<f64> {
    let d_tanh = d_tanh(a_raw);
    d_tanh.ln().sum_axis(Axis(1))
}

pub fn tanh_gaussian_correction_eps(a_raw: &Array2<f64>) -> Array1<f64> {
    let deriv = 1.0 - a_raw.mapv(|u| u.tanh().powi(2)); // d_tanh(u)
    deriv.mapv(|v| (v + 1e-8).ln()).sum_axis(Axis(1))
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

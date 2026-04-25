use core::f32;

use ndarray::{Array1, Array2, Array3, ArrayRef2, Axis, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use ndarray_stats::QuantileExt;
use rand::{Rng, rngs::ThreadRng};
use serde::{Deserialize, Serialize};

pub type ActivationFn = fn(&ArrayRef2<f32>) -> Array2<f32>;
pub type InitializationFn = fn((usize, usize)) -> Array2<f32>;

// ACTIVATIONS

pub fn relu(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.mapv(|v| v.max(0.))
}

pub fn d_relu(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.mapv(|v| v.signum().max(0.))
}

pub fn tanh(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.mapv(|v| v.tanh())
}

pub fn d_tanh(x: &ArrayRef2<f32>) -> Array2<f32> {
    1. - (x.mapv(|v| v.tanh())).powi(2)
}

pub fn leaky_relu(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.mapv(|x| if x >= 0. { x } else { 0.01 * x })
}

pub fn d_leaky_relu(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.mapv(|x| if x >= 0. { 1. } else { 0.01 })
}

pub fn exp(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.exp()
}

pub fn d_exp(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.exp()
}

pub fn elu(x: &ArrayRef2<f32>) -> Array2<f32> {
    let x_pos = x.clamp(0., f32::MAX);
    let x_neg = x.clamp(f32::MIN, 0.);

    &x_pos + (x_neg.exp() - 1.)
}

pub fn d_elu(x: &ArrayRef2<f32>) -> Array2<f32> {
    let x_pos = x.clamp(0., f32::MAX);
    let x_neg = x.clamp(f32::MIN, 0.);

    &x_pos.signum() + x_neg.exp()
}

pub fn softplus(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.mapv(|v| if v > 20.0 { v } else { (1.0 + v.exp()).ln() })
}

pub fn d_softplus(x: &ArrayRef2<f32>) -> Array2<f32> {
    sigmoid(x)
}

pub fn ident(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.to_owned()
}

pub fn d_ident(x: &ArrayRef2<f32>) -> Array2<f32> {
    Array2::ones(x.dim())
}

pub fn sigmoid(x: &ArrayRef2<f32>) -> Array2<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn d_sigmoid(x: &ArrayRef2<f32>) -> Array2<f32> {
    let s = sigmoid(x);
    &s * &(1.0 - &s)
}

pub fn softmax(x: &ArrayRef2<f32>) -> Array2<f32> {
    let maxes = x
        .map_axis(Axis(1), |row| row.max().cloned().unwrap_or(1e-4))
        .insert_axis(Axis(1));

    let mut d = x - maxes;

    d.mapv_inplace(|x| x.exp());

    let sums = d.map_axis(Axis(1), |row| row.sum()).insert_axis(Axis(1));

    let last = &d / &sums;
    return last;
}

pub fn d_softmax(s: &ArrayRef2<f32>, g: &ArrayRef2<f32>) -> Array2<f32> {
    let dot = (g * s).sum_axis(Axis(1)).insert_axis(Axis(1));
    s * (g - dot)
}

/**
 * The output of this method gets multiplied against the incoming gradient. Since the gradient of
 * cross entropy loss is y_pred - y, we don't want to mutate it at all, and return 1
 */
pub fn d_softmax_cross_entropy(x: &ArrayRef2<f32>) -> Array2<f32> {
    Array2::ones(x.dim())
}

// INITS

pub fn he(shape: (usize, usize)) -> Array2<f32> {
    let bound = f32::sqrt(6.) / f32::sqrt(shape.0 as f32);
    return Array2::random(shape, Uniform::new(-bound, bound).unwrap());
}

pub fn xavier_normal(shape: (usize, usize)) -> Array2<f32> {
    let std = (2. / ((shape.0 + shape.1) as f32)).sqrt();
    Array2::random(
        shape,
        ndarray_rand::rand_distr::Normal::new(0., std).unwrap(),
    )
}

pub fn xavier(shape: (usize, usize)) -> Array2<f32> {
    let bound = (6. / (shape.0 as f32 + shape.1 as f32)).sqrt();
    return Array2::random(shape, Uniform::new(-bound, bound).unwrap());
}

// LOSSES

pub fn soft_cross_entropy_loss(
    probs: &ArrayRef2<f32>,
    targets: &ArrayRef2<f32>,
) -> (f32, Array2<f32>) {
    let log_probs = probs.mapv(|p| p.max(1e-10).ln());
    let loss = -(targets * log_probs).sum() / probs.dim().0 as f32;

    let d_loss = probs - targets;

    (loss, d_loss)
}

pub fn batch_soft_cross_entropy_loss(
    probs: Array3<f32>,
    targets: Array3<f32>,
) -> (f32, Array3<f32>) {
    let (b, s, d) = probs.dim();
    let probs_2d = probs.into_shape_clone((b * s, d)).unwrap();
    let targets_2d = targets.into_shape_clone((b * s, d)).unwrap();
    let (loss, d_loss_2d) = soft_cross_entropy_loss(&probs_2d, &targets_2d);
    let d_loss = d_loss_2d.into_shape_clone((b, s, d)).unwrap();
    (loss, d_loss)
}

pub fn cross_entropy_loss(logits: Array3<f32>, labels: &Array2<usize>) -> (f32, Array3<f32>) {
    let (batch_size, seq_len, _vocab_size) = logits.dim();
    let n = (batch_size * seq_len) as f32;

    // numerical stability, subtract max per position
    let max_logits = logits
        .map_axis(Axis(2), |row| {
            *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        })
        .insert_axis(Axis(2));
    let shifted = &logits - &max_logits;

    // softmax
    let exp = shifted.mapv(f32::exp);
    let sum_exp = exp.sum_axis(Axis(2)).insert_axis(Axis(2));
    let softmax = &exp / &sum_exp;

    // gather -log(prob) at target indices
    let mut total_loss = 0.0f32;
    for b in 0..batch_size {
        for t in 0..seq_len {
            total_loss -= softmax[[b, t, labels[[b, t]]]].ln();
        }
    }

    // grad softmax - one_hot(target), averaged
    let mut d_logits = softmax;
    for b in 0..batch_size {
        for t in 0..seq_len {
            d_logits[[b, t, labels[[b, t]]]] -= 1.0;
        }
    }
    d_logits /= n;

    (total_loss / n, d_logits)
}

pub fn bce_loss(sigmoid_probs: &ArrayRef2<f32>, targets: &ArrayRef2<f32>) -> (f32, Array2<f32>) {
    let eps = 1e-10;
    let n = sigmoid_probs.len() as f32;

    let loss = -(targets * sigmoid_probs.mapv(|p| p.max(eps).ln())
        + (1.0 - targets) * &sigmoid_probs.mapv(|p| (1.0 - p).max(eps).ln()))
        .sum()
        / n;

    let d_loss = (sigmoid_probs - targets)
        / &(sigmoid_probs * (1.0 - sigmoid_probs)).mapv(|v| v.max(eps))
        / n;

    (loss, d_loss)
}

pub fn batch_bce_loss(sigmoid_probs: Array3<f32>, targets: Array3<f32>) -> (f32, Array3<f32>) {
    let (b, s, d) = sigmoid_probs.dim();
    let sigmoid_probs_2d = sigmoid_probs.into_shape_clone((b * s, d)).unwrap();
    let targets_2d = targets.into_shape_clone((b * s, d)).unwrap();
    let (loss, d_loss_2d) = bce_loss(&sigmoid_probs_2d, &targets_2d);
    let d_loss = d_loss_2d.into_shape_clone((b, s, d)).unwrap();
    (loss, d_loss)
}

pub fn mse_loss(logits: &ArrayRef2<f32>, targets: &ArrayRef2<f32>) -> (f32, Array2<f32>) {
    let diff = logits - targets;
    let n = logits.len() as f32;

    let loss = diff.pow2().mean().unwrap();

    let d_loss = 2. * &diff / n;

    (loss, d_loss)
}

pub fn batch_mse_loss(logits: Array3<f32>, targets: Array3<f32>) -> (f32, Array3<f32>) {
    let (b, s, d) = logits.dim();
    let logits_2d = logits.into_shape_clone((b * s, d)).unwrap();
    let targets_2d = targets.into_shape_clone((b * s, d)).unwrap();
    let (loss, d_loss_2d) = mse_loss(&logits_2d, &targets_2d);
    let d_loss = d_loss_2d.into_shape_clone((b, s, d)).unwrap();
    (loss, d_loss)
}

// NORM

pub fn l2(x: &ArrayRef2<f32>) -> Array1<f32> {
    let eps = 1e-6;
    x.pow2().sum_axis(Axis(1)).sqrt().mapv(|x| x.max(eps))
}

pub fn l2_norm(x: &ArrayRef2<f32>) -> Array2<f32> {
    x / l2(&x).insert_axis(Axis(1))
}

pub fn d_l2_norm(x: &ArrayRef2<f32>, grad: &ArrayRef2<f32>) -> Array2<f32> {
    let norm = l2(x).insert_axis(Axis(1));
    let x_hat = x / norm.to_owned();
    let dot = (x_hat.to_owned() * grad)
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));
    (grad - &x_hat * dot) / &norm
}

// MISC

pub fn linear_norm(x: &ArrayRef2<f32>) -> Array2<f32> {
    let sum = x.sum_axis(Axis(1)).insert_axis(Axis(1));
    x / sum
}

pub fn softmax_vector_jacobian_product(
    upstream: &ArrayRef2<f32>,
    softmax_out: &ArrayRef2<f32>,
) -> Array2<f32> {
    let mut grad = upstream.to_owned();

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

pub fn positional_encoding_seq(seq_len: usize, features: usize) -> Array2<f32> {
    let mut pe = Array2::zeros((seq_len, features));
    let n: f32 = 10_000.;

    for k in 0..seq_len {
        for i in 0..(features / 2) {
            let exp = n.powf((2.0 * i as f32) / features as f32);
            pe[[k, 2 * i]] = f32::sin((k as f32) / exp);
            if (2 * i + 1) < features {
                pe[[k, 2 * i + 1]] = f32::cos((k as f32) / exp);
            }
        }
    }

    pe
}

// TODO: Use with Gumbel max sampling, avoid standard softmax entirely.
pub fn log_softmax(x: Array2<f32>) -> Array2<f32> {
    let maxes = x
        .map_axis(Axis(1), |row| row.max().cloned().unwrap_or(1e-4))
        .insert_axis(Axis(1));

    let d = &x - &maxes;
    let sums = d
        .map_axis(Axis(1), |row| row.exp().sum())
        .insert_axis(Axis(1));

    &d - sums.ln()
}

pub fn clip_grad(mut grad: Array2<f32>, clip: f32) -> Array2<f32> {
    let norm_sq = grad.mapv(|x| x * x).sum();
    let norm = norm_sq.sqrt();

    if norm > clip {
        grad.mapv_inplace(|x| x * (clip / (norm + 1e-6)));
    }

    grad
}

pub fn causal_mask(n: usize) -> Array2<f32> {
    let mut mask = Array2::zeros((n, n));

    for i in 0..n {
        mask.slice_mut(s![i, 0..(i + 1)])
            .assign(&Array1::ones(i + 1));
    }

    mask
}

pub fn sample_categorical(probs: &Array1<f32>, rng: &mut ThreadRng) -> usize {
    let mut u: f32 = rng.random();

    for (i, &p) in probs.iter().enumerate() {
        if u < p {
            return i;
        }

        u -= p;
    }

    return 0;
}

pub fn sample_gumbel_categorical(log_probs: &Array1<f32>, rng: &mut ThreadRng) -> usize {
    let u = (0..log_probs.len())
        .map(|_| -f32::ln(-f32::ln(rng.random())))
        .collect::<Array1<f32>>();

    (log_probs + u).argmax().unwrap()
}

pub fn gaussian_log_prob(
    x: &ArrayRef2<f32>,
    u: &ArrayRef2<f32>,
    o: &ArrayRef2<f32>,
) -> Array1<f32> {
    let eps = 1e-8;
    let quadratic = (x - u).powi(2) / o.powi(2);
    let norm = 2. * o.map(|v| v.max(eps).ln());
    let constant = (f32::consts::PI * 2.).ln();

    let dims = -0.5 * (quadratic + norm + constant);

    dims.sum_axis(Axis(1))
}

pub fn tanh_gaussian_correction(a_raw: &ArrayRef2<f32>) -> Array1<f32> {
    let d_tanh = d_tanh(a_raw);
    d_tanh.ln().sum_axis(Axis(1))
}

pub fn tanh_gaussian_correction_eps(a_raw: &ArrayRef2<f32>) -> Array1<f32> {
    let deriv = 1.0 - a_raw.mapv(|u| u.tanh().powi(2)); // d_tanh(u)
    deriv.mapv(|v| (v + 1e-8).ln()).sum_axis(Axis(1))
}

pub fn calculate_spectral_radius(w_r: &ArrayRef2<f32>, n: usize) -> f32 {
    let mut h = Array2::ones((1, w_r.dim().0));
    let mut radius = 0.;
    for _ in 0..n {
        h = h.dot(w_r);
        radius = l2(&h)[[0]];
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

use std::time::Instant;

use nbml::{
    ndarray::{Array2, Array3, Axis, s},
    ndarray_rand::{RandomExt, rand_distr::Uniform},
    nn::GatedLinearAttention,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use rand::Rng;

pub const D_MODEL: usize = 32;
pub const N_PAIRS: usize = 500;

fn associative_recall_dataset(num_pairs: usize, d_model: usize) -> (Array2<f64>, Array2<f64>) {
    let keys = Array2::random((num_pairs, d_model), Uniform::new(0., 10.));
    let values = Array2::random((num_pairs, d_model), Uniform::new(0., 10.));

    let mut sequence = Array2::zeros((num_pairs * 2 + 2, d_model));

    for i in 0..num_pairs {
        sequence.slice_mut(s![2 * i, ..]).assign(&keys.row(i));
        sequence.slice_mut(s![2 * i + 1, ..]).assign(&values.row(i));
    }

    let recall_i = rand::rng().random_range(0..num_pairs);
    let key_i = keys.row(recall_i);
    sequence.slice_mut(s![-2, ..]).assign(&key_i);

    let x = sequence.clone();

    let mut y = Array2::zeros(x.dim());
    let values_i = values.row(recall_i);
    y.slice_mut(s![-1, ..]).assign(&values_i);

    (x, y)
}

pub fn associative_recall() {
    let d_model = D_MODEL;
    let num_pairs = N_PAIRS;
    println!("d_model {d_model} seq_len {}", num_pairs * 2);
    let start = Instant::now();

    let mut model = GatedLinearAttention::new(d_model, 2 * d_model);
    let mut optim = AdamW::default().with(&mut model);

    for epoch in 0..500 {
        let (x, y) = associative_recall_dataset(num_pairs, d_model);
        let x = x.insert_axis(Axis(0));
        let y = y.insert_axis(Axis(0));

        let y_pred = model.forward(x, true);

        let diff = &y_pred.slice(s![.., -1, ..]) - &y.slice(s![.., -1, ..]);
        let loss = diff.pow2().mean().unwrap();

        let mut d_loss = Array3::zeros(y_pred.dim());
        let d_loss_diff = 2. * &diff;
        d_loss.slice_mut(s![.., -1, ..]).assign(&d_loss_diff);

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if epoch % 100 == 0 {
            println!("epoch {epoch} loss={loss}");
        }
    }

    let mut test_losses = vec![];

    for _ in 0..100 {
        let (x, y) = associative_recall_dataset(num_pairs, d_model);
        let x = x.insert_axis(Axis(0));
        let y = y.insert_axis(Axis(0));

        let y_pred = model.forward(x, false);

        let diff = &y_pred.slice(s![.., -1, ..]) - &y.slice(s![.., -1, ..]);
        let loss = diff.pow2().mean().unwrap();

        test_losses.push(loss);
    }

    let avg_loss = test_losses.iter().sum::<f64>() / test_losses.len() as f64;

    let end = Instant::now();
    let time = end.duration_since(start).as_secs_f32();
    println!("avg loss: {avg_loss} time {time}s");
}

pub fn main() {
    println!("Running Associative Recall on Linear Self Attention");

    associative_recall();
}

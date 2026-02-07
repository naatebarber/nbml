use std::time::Instant;

use nbml::{
    ndarray::{Array1, concatenate, s},
    nn::GatedLinearAttention,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use rand::{
    rng,
    seq::{IndexedRandom, SliceRandom},
};

use nbml::{
    f::cross_entropy_loss,
    ndarray::{Array2, Array3, Axis},
    ndarray_stats::QuantileExt,
};

pub const D_MODEL: usize = 16;
pub const PATTERN_LEN: usize = 10;
pub const DELAY: usize = 20;
pub const BATCH_SIZE: usize = 10;

pub const LEARNING_RATE: f64 = 1e-3;
pub const EPOCHS: usize = 10000;

fn make_selective_copy_dataset(
    pattern_len: usize,
    delay: usize,
    features: usize,
) -> (Array2<f64>, Array2<f64>) {
    let vocab = (0..features)
        .map(|t| {
            let mut token = Array1::zeros(features);
            token[t] = 1.;
            token
        })
        .collect::<Vec<Array1<f64>>>();

    let mut pattern = Array2::zeros((pattern_len, features));

    for i in 0..pattern_len {
        let tok = vocab.choose(&mut rng()).unwrap();
        pattern.slice_mut(s![i, ..]).assign(&tok);
    }

    let mut buffer = Array2::zeros((pattern_len + delay, features));

    let mut indices = (0..buffer.dim().0).collect::<Vec<usize>>();
    indices.shuffle(&mut rng());
    let mut indices = indices[0..pattern_len].to_owned();
    indices.sort();

    for (i, index) in indices.iter().enumerate() {
        buffer.slice_mut(s![*index, ..]).assign(&pattern.row(i));
    }

    let x = concatenate![Axis(0), buffer.view(), Array2::zeros(pattern.dim())];
    let y = concatenate![Axis(0), Array2::zeros(buffer.dim()), pattern.view()];

    (x, y)
}

fn batch_selective_copy_dataset(
    pattern_len: usize,
    delay: usize,
    features: usize,
    batch_size: usize,
) -> (Array3<f64>, Array3<f64>) {
    let mut xs = vec![];
    let mut ys = vec![];

    for _ in 0..batch_size {
        let (x, y) = make_selective_copy_dataset(pattern_len, delay, features);
        xs.push(x);
        ys.push(y);
    }

    let mut x = xs.pop().unwrap().insert_axis(Axis(0));
    let mut y = ys.pop().unwrap().insert_axis(Axis(0));

    xs.iter().zip(ys.iter()).for_each(|(xt, yt)| {
        x = concatenate![Axis(0), x.view(), xt.to_owned().insert_axis(Axis(0)).view()];
        y = concatenate![Axis(0), y.view(), yt.to_owned().insert_axis(Axis(0)).view()];
    });

    (x, y)
}

pub fn selective_copy() {
    let d_model = D_MODEL;
    let pattern_len = PATTERN_LEN;
    let delay = DELAY;
    let batch_size = BATCH_SIZE;
    println!("d_model {d_model} seq_len {}", 2 * pattern_len + delay);
    let start = Instant::now();

    let mut model = GatedLinearAttention::new(d_model, 2 * d_model, 4);
    let mut optim = AdamW::default().with(&mut model);

    let epochs = EPOCHS;
    let mut final_loss = f64::MAX;

    for epoch in 0..epochs {
        let (x, y) = batch_selective_copy_dataset(pattern_len, delay, d_model, batch_size);

        let output = model.forward(x, true);

        let tok_output = output.slice(s![.., -(pattern_len as isize).., ..]);
        let tok_y = y.slice(s![.., -(pattern_len as isize).., ..]);

        let diff = (&tok_output - &tok_y) / tok_output.shape()[1] as f64;
        let loss = cross_entropy_loss(&tok_output.to_owned(), &tok_y.to_owned());

        let mut d_loss = Array3::zeros(output.dim());
        d_loss
            .slice_mut(s![.., -(pattern_len as isize).., ..])
            .assign(&diff);

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if epoch % (EPOCHS / 10) == 0 {
            println!("epoch {}: loss = {:.6}", epoch, loss);
        }

        final_loss = loss;
    }

    println!("final loss: {:.6}", final_loss);

    let mut losses = vec![];
    let mut accuracies = vec![];

    for _ in 0..100 {
        let (x, y) = make_selective_copy_dataset(pattern_len, delay, d_model);
        let x = x.insert_axis(Axis(0));
        let y = y.insert_axis(Axis(0));

        let output = model.forward(x, false);
        let tok_output = output.slice(s![.., -(pattern_len as isize).., ..]);
        let tok_y = y.slice(s![.., -(pattern_len as isize).., ..]);

        let loss = cross_entropy_loss(&tok_output.to_owned(), &tok_y.to_owned());
        losses.push(loss);

        let mut y = 0;
        let mut n = 0;

        for i in 0..pattern_len {
            let output = tok_output.slice(s![0, i, ..]);
            let y_true = tok_y.slice(s![0, i, ..]);

            if output.argmax().unwrap() == y_true.argmax().unwrap() {
                y += 1;
            } else {
                n += 1;
            }
        }

        let acc = ((y as f64) / (y + n) as f64) * 100.;
        accuracies.push(acc);
    }

    let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
    let avg_acc = accuracies.iter().sum::<f64>() / losses.len() as f64;

    let end = Instant::now();
    let time = end.duration_since(start).as_secs_f32();
    println!(
        "avg loss: {}, avg accuracy: {}% time {}s",
        avg_loss, avg_acc, time
    );
}

fn main() {
    selective_copy();
}

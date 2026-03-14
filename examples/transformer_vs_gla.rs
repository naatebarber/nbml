use std::time::Instant;

use nbml::{
    f::{cross_entropy_loss, softmax},
    ndarray::{Array1, Array2, Array3, Axis, concatenate, s},
    ndarray_stats::QuantileExt,
    nn::{GlaTransformer, Transformer},
    optim::{AdamW, Optimizer, ToParams},
};
use rand::{rng, seq::IndexedRandom};

// Task: Associative Recall with variable-length distractor padding
//
// The model sees key-value pairs interleaved with distractor (noise) tokens,
// followed by a query key, and must recall the associated value.
//
// Example (n_pairs=3, distractors=2 per gap):
//   [k1, v1, noise, noise, k2, v2, noise, noise, k3, v3, noise, noise, QUERY, k2]
//                                                                              ^ predict v2
//
// By increasing distractors we stretch the sequence length without making the
// recall task harder, letting us see where GLA's O(n) scaling beats O(n²) attention.

const D_MODEL: usize = 16;
const D_HEAD: usize = 8;
const N_HEAD: usize = 4;
const N_PAIRS: usize = 4;
const BATCH_SIZE: usize = 16;
const EPOCHS: usize = 3000;
const LEARNING_RATE: f64 = 1e-3;
const EVAL_SAMPLES: usize = 100;

fn one_hot(index: usize, size: usize) -> Array1<f64> {
    let mut v = Array1::zeros(size);
    v[index] = 1.0;
    v
}

fn random_noise(vocab_size: usize) -> Array1<f64> {
    let tokens: Vec<usize> = (0..vocab_size).collect();
    let idx = *tokens.choose(&mut rng()).unwrap();
    // Use small random values instead of one-hot to distinguish from real tokens
    let mut v = Array1::zeros(vocab_size);
    v[idx] = 0.1;
    v
}

/// Build one associative recall sample with `distractors` noise tokens after each pair.
fn make_sample(n_pairs: usize, vocab_size: usize, distractors: usize) -> (Array2<f64>, usize) {
    let tokens: Vec<usize> = (0..vocab_size).collect();

    let mut keys = vec![];
    while keys.len() < n_pairs {
        let k = *tokens.choose(&mut rng()).unwrap();
        if !keys.contains(&k) {
            keys.push(k);
        }
    }

    let values: Vec<usize> = (0..n_pairs)
        .map(|_| *tokens.choose(&mut rng()).unwrap())
        .collect();

    let indices: Vec<usize> = (0..n_pairs).collect();
    let query_idx = *indices.choose(&mut rng()).unwrap();
    let query_key = keys[query_idx];
    let target_value = values[query_idx];

    // seq_len = n_pairs * (2 + distractors) + 2  (query marker + query key)
    let seq_len = n_pairs * (2 + distractors) + 2;
    let mut input = Array2::zeros((seq_len, vocab_size));

    let mut pos = 0;
    for i in 0..n_pairs {
        input.row_mut(pos).assign(&one_hot(keys[i], vocab_size));
        pos += 1;
        input.row_mut(pos).assign(&one_hot(values[i], vocab_size));
        pos += 1;
        for _ in 0..distractors {
            input.row_mut(pos).assign(&random_noise(vocab_size));
            pos += 1;
        }
    }
    // Query marker (all zeros) — already set
    pos += 1;
    // Query key
    input.row_mut(pos).assign(&one_hot(query_key, vocab_size));

    (input, target_value)
}

fn make_batch(
    n_pairs: usize,
    vocab_size: usize,
    batch_size: usize,
    distractors: usize,
) -> (Array3<f64>, Array3<f64>, Vec<usize>) {
    let mut xs = vec![];
    let mut targets = vec![];

    for _ in 0..batch_size {
        let (x, t) = make_sample(n_pairs, vocab_size, distractors);
        xs.push(x);
        targets.push(t);
    }

    let x_views: Vec<_> = xs.iter().map(|x| x.view().insert_axis(Axis(0))).collect();
    let x_batch = concatenate(Axis(0), &x_views).unwrap();

    let mut y_batch = Array3::zeros((batch_size, 1, vocab_size));
    for (i, &t) in targets.iter().enumerate() {
        y_batch[[i, 0, t]] = 1.0;
    }

    (x_batch, y_batch, targets)
}

fn train_transformer(distractors: usize) -> (f64, f64) {
    let vocab_size = D_MODEL;
    let seq_len = N_PAIRS * (2 + distractors) + 2;
    let mut model = Transformer::new_decoder(vocab_size, D_HEAD, N_HEAD);
    let mut optim = AdamW {
        learning_rate: LEARNING_RATE,
        ..AdamW::default()
    }
    .with(&mut model);

    let pad_mask = Array2::ones((BATCH_SIZE, seq_len));

    let start = Instant::now();

    for epoch in 0..EPOCHS {
        let (x, y, _) = make_batch(N_PAIRS, vocab_size, BATCH_SIZE, distractors);
        let output = model.forward(x, pad_mask.clone(), true);

        // Softmax the last position logits -> probabilities
        let logits = output.slice(s![.., -1, ..]).to_owned(); // (batch, vocab)
        let probs = softmax(&logits); // (batch, vocab)
        let y_2d = y.slice(s![.., 0, ..]).to_owned(); // (batch, vocab)

        // Softmax + cross-entropy gradient: (probs - target) / batch
        let diff = (&probs - &y_2d) / BATCH_SIZE as f64;
        let mut d_loss = Array3::zeros(output.dim());
        d_loss.slice_mut(s![.., -1, ..]).assign(&diff);

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if epoch % (EPOCHS / 5) == 0 {
            let probs_3d = probs.clone().insert_axis(Axis(1));
            let loss = cross_entropy_loss(&probs_3d, &y);
            println!("  [TF  seq={seq_len}] epoch {epoch}: loss = {loss:.4}");
        }
    }

    let train_time = start.elapsed().as_secs_f64();

    // Eval
    let mut correct = 0;
    let pad_eval = Array2::ones((1, seq_len));
    for _ in 0..EVAL_SAMPLES {
        let (x, _, targets) = make_batch(N_PAIRS, vocab_size, 1, distractors);
        let output = model.forward(x, pad_eval.clone(), false);
        let predicted = output.slice(s![0, -1, ..]).argmax().unwrap();
        if predicted == targets[0] {
            correct += 1;
        }
    }
    let accuracy = (correct as f64 / EVAL_SAMPLES as f64) * 100.0;

    (accuracy, train_time)
}

fn train_gla(distractors: usize) -> (f64, f64) {
    let vocab_size = D_MODEL;
    let seq_len = N_PAIRS * (2 + distractors) + 2;
    let mut model = GlaTransformer::new(vocab_size, D_HEAD, N_HEAD);
    let mut optim = AdamW {
        learning_rate: LEARNING_RATE,
        ..AdamW::default()
    }
    .with(&mut model);

    let start = Instant::now();

    for epoch in 0..EPOCHS {
        let (x, y, _) = make_batch(N_PAIRS, vocab_size, BATCH_SIZE, distractors);
        let output = model.forward(x, true);

        let logits = output.slice(s![.., -1, ..]).to_owned();
        let probs = softmax(&logits);
        let y_2d = y.slice(s![.., 0, ..]).to_owned();

        let diff = (&probs - &y_2d) / BATCH_SIZE as f64;
        let mut d_loss = Array3::zeros(output.dim());
        d_loss.slice_mut(s![.., -1, ..]).assign(&diff);

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if epoch % (EPOCHS / 5) == 0 {
            let probs_3d = probs.clone().insert_axis(Axis(1));
            let loss = cross_entropy_loss(&probs_3d, &y);
            println!("  [GLA seq={seq_len}] epoch {epoch}: loss = {loss:.4}");
        }
    }

    let train_time = start.elapsed().as_secs_f64();

    // Eval
    let mut correct = 0;
    for _ in 0..EVAL_SAMPLES {
        let (x, _, targets) = make_batch(N_PAIRS, vocab_size, 1, distractors);
        let output = model.forward(x, false);
        let predicted = output.slice(s![0, -1, ..]).argmax().unwrap();
        if predicted == targets[0] {
            correct += 1;
        }
    }
    let accuracy = (correct as f64 / EVAL_SAMPLES as f64) * 100.0;

    (accuracy, train_time)
}

fn main() {
    println!("=== Transformer vs GLA: Scaling with Sequence Length ===");
    println!("d_model={D_MODEL}, n_pairs={N_PAIRS}, batch={BATCH_SIZE}, epochs={EPOCHS}");
    println!("Distractor tokens are inserted between pairs to stretch sequence length.\n");

    let distractor_counts = vec![0, 10, 30]; // vec![0, 10, 30, 80];

    let mut results: Vec<(usize, f64, f64, f64, f64)> = vec![];

    for &d in &distractor_counts {
        let seq_len = N_PAIRS * (2 + d) + 2;
        println!("--- distractors={d}, seq_len={seq_len} ---");

        let (tf_acc, tf_time) = train_transformer(d);
        let (gla_acc, gla_time) = train_gla(d);

        results.push((seq_len, tf_acc, tf_time, gla_acc, gla_time));
        println!();
    }

    println!("=== Results ===");
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Seq Len", "TF Acc%", "TF Time", "GLA Acc%", "GLA Time", "Speedup"
    );
    println!("{:-<62}", "");
    for (seq_len, tf_acc, tf_time, gla_acc, gla_time) in &results {
        println!(
            "{:<10} {:>9.1}% {:>9.2}s {:>9.1}% {:>9.2}s {:>9.2}x",
            seq_len,
            tf_acc,
            tf_time,
            gla_acc,
            gla_time,
            tf_time / gla_time
        );
    }
}

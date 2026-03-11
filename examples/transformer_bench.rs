use std::time::Instant;

use nbml::{
    Tensor, f::Activation, nn::Transformer as TransformerV1, nn2::Transformer as TransformerV2,
    optim::adam::AdamW as AdamWV1, optim::optimizer::Optimizer as OptimizerV1,
    optim::param::ToParams as ToParamsV1, optim2::Optimizer as OptimizerV2,
    optim2::ToParams as ToParamsV2, optim2::adam::AdamW as AdamWV2,
};
use ndarray::{Array2, Array3};

fn bench_v1(
    d_in: usize,
    d_head: usize,
    n_head: usize,
    batch_size: usize,
    seq_len: usize,
    epochs: usize,
) -> f64 {
    let ff_layers = vec![
        (d_in, 4 * d_in, Activation::Relu),
        (4 * d_in, d_in, Activation::Identity),
    ];

    let mut model = TransformerV1::new_encoder(d_in, d_head, n_head, ff_layers);
    let mut optim = AdamWV1::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let x = Array3::ones((batch_size, seq_len, d_in));
    let y: Array3<f64> = Array3::ones((batch_size, seq_len, d_in)) * 0.5;
    let pad_mask = Array2::ones((batch_size, seq_len));

    let start = Instant::now();

    for _ in 0..epochs {
        let y_pred = model.forward(x.clone(), pad_mask.clone(), true);
        let d_loss = 2. * (&y_pred - &y);
        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }

    start.elapsed().as_secs_f64()
}

fn bench_v2(
    d_in: usize,
    d_head: usize,
    n_head: usize,
    batch_size: usize,
    seq_len: usize,
    epochs: usize,
) -> f64 {
    let mut model = TransformerV2::new_encoder(d_in, d_head, n_head);
    let mut optim = AdamWV2::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let x = Tensor::ones((batch_size, seq_len, d_in));
    let y = Tensor::from_elem((batch_size, seq_len, d_in), 0.5);
    let pad_mask = Tensor::ones((batch_size, seq_len));

    let start = Instant::now();

    for _ in 0..epochs {
        let y_pred = model.forward(x.clone(), pad_mask.clone(), true);
        let d_loss = (&y_pred - &y) * 2.0;
        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }

    start.elapsed().as_secs_f64()
}

fn main() {
    let d_in = 32;
    let d_head = 8;
    let n_head = 4;
    let batch_size = 16;
    let seq_len = 16;
    let epochs = 100;

    println!(
        "Transformer benchmark: d_in={d_in} d_head={d_head} n_head={n_head} batch={batch_size} seq={seq_len} epochs={epochs}"
    );
    println!();

    // warmup
    bench_v1(d_in, d_head, n_head, batch_size, seq_len, 5);
    bench_v2(d_in, d_head, n_head, batch_size, seq_len, 5);

    let t1 = bench_v1(d_in, d_head, n_head, batch_size, seq_len, epochs);
    let t2 = bench_v2(d_in, d_head, n_head, batch_size, seq_len, epochs);

    println!(
        "nn  (ndarray):  {t1:.3}s  ({:.1} ms/epoch)",
        t1 / epochs as f64 * 1000.0
    );
    println!(
        "nn2 (Tensor):   {t2:.3}s  ({:.1} ms/epoch)",
        t2 / epochs as f64 * 1000.0
    );
    println!();

    let ratio = t1 / t2;
    if ratio > 1.0 {
        println!("nn2 is {ratio:.2}x faster");
    } else {
        println!("nn is {:.2}x faster", 1.0 / ratio);
    }
}

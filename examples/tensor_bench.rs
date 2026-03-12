use std::time::Instant;

use nbml::{
    Tensor, f::Activation, f::InitializationFn as InitV1, f2::InitializationFn as InitV2,
    nn::Conv2D as Conv2DV1, nn::GatedLinearAttention as GLAV1, nn::LSTM as LSTMV1,
    nn::Transformer as TransformerV1, nn2::GatedLinearAttention as GLAV2, nn2::LSTM as LSTMV2,
    nn2::Transformer as TransformerV2, nn2::conv2d::Conv2D as Conv2DV2,
    optim::adam::AdamW as AdamWV1, optim::optimizer::Optimizer as OptimizerV1,
    optim::param::ToParams as ToParamsV1, optim2::Optimizer as OptimizerV2,
    optim2::ToParams as ToParamsV2, optim2::adam::AdamW as AdamWV2,
};
use ndarray::{Array2, Array3, Array4};

fn print_result(name: &str, t1: f64, t2: f64, epochs: usize) {
    println!("{name}");
    println!(
        "  nn  (ndarray):  {t1:.3}s  ({:.1} ms/epoch)",
        t1 / epochs as f64 * 1000.0
    );
    println!(
        "  nn2 (Tensor):   {t2:.3}s  ({:.1} ms/epoch)",
        t2 / epochs as f64 * 1000.0
    );

    let ratio = t1 / t2;
    if ratio > 1.0 {
        println!("  => nn2 is {ratio:.2}x faster");
    } else {
        println!("  => nn is {:.2}x faster", 1.0 / ratio);
    }
    println!();
}

// --- Transformer ---

fn transformer_v1(
    d_in: usize,
    d_head: usize,
    n_head: usize,
    batch: usize,
    seq: usize,
    epochs: usize,
) -> f64 {
    let ff_layers = vec![
        (d_in, 4 * d_in, Activation::Relu),
        (4 * d_in, d_in, Activation::Identity),
    ];
    let mut model = TransformerV1::new_encoder(d_in, d_head, n_head, ff_layers);
    let mut optim = AdamWV1::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let x = Array3::ones((batch, seq, d_in));
    let y: Array3<f64> = Array3::ones((batch, seq, d_in)) * 0.5;
    let pad_mask = Array2::ones((batch, seq));

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

fn transformer_v2(
    d_in: usize,
    d_head: usize,
    n_head: usize,
    batch: usize,
    seq: usize,
    epochs: usize,
) -> f64 {
    let mut model = TransformerV2::new_encoder(d_in, d_head, n_head);
    let mut optim = AdamWV2::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let x = Tensor::ones((batch, seq, d_in));
    let y = Tensor::from_elem((batch, seq, d_in), 0.5);
    let pad_mask = Tensor::ones((batch, seq));

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

// --- LSTM ---

fn lstm_v1(d_model: usize, batch: usize, seq: usize, epochs: usize) -> f64 {
    let mut model = LSTMV1::new(d_model);
    let mut optim = AdamWV1::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let x = Array3::ones((batch, seq, d_model));
    let y: Array3<f64> = Array3::ones((batch, seq, d_model)) * 0.5;

    let start = Instant::now();
    for _ in 0..epochs {
        let y_pred = model.forward(x.clone(), true);
        let d_loss = 2. * (&y_pred - &y);
        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }
    start.elapsed().as_secs_f64()
}

fn lstm_v2(d_model: usize, batch: usize, seq: usize, epochs: usize) -> f64 {
    let mut model = LSTMV2::new(d_model);
    let mut optim = AdamWV2::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let x = Tensor::ones((batch, seq, d_model));
    let y = Tensor::from_elem((batch, seq, d_model), 0.5);

    let start = Instant::now();
    for _ in 0..epochs {
        let y_pred = model.forward(x.clone(), true);
        let d_loss = (&y_pred - &y) * 2.0;
        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }
    start.elapsed().as_secs_f64()
}

// --- Gated Linear Attention ---

fn gla_v1(
    d_in: usize,
    d_head: usize,
    n_head: usize,
    batch: usize,
    seq: usize,
    epochs: usize,
) -> f64 {
    let mut model = GLAV1::new(d_in, d_head, n_head);
    let mut optim = AdamWV1::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let x = Array3::ones((batch, seq, d_in));
    let y: Array3<f64> = Array3::ones((batch, seq, d_in)) * 0.5;

    let start = Instant::now();
    for _ in 0..epochs {
        let y_pred = model.forward(x.clone(), true);
        let d_loss = 2. * (&y_pred - &y);
        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }
    start.elapsed().as_secs_f64()
}

fn gla_v2(
    d_in: usize,
    d_head: usize,
    n_head: usize,
    batch: usize,
    seq: usize,
    epochs: usize,
) -> f64 {
    let mut model = GLAV2::new(d_in, d_head, n_head);
    let mut optim = AdamWV2::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let x = Tensor::ones((batch, seq, d_in));
    let y = Tensor::from_elem((batch, seq, d_in), 0.5);

    let start = Instant::now();
    for _ in 0..epochs {
        let y_pred = model.forward(x.clone(), true);
        let d_loss = (&y_pred - &y) * 2.0;
        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }
    start.elapsed().as_secs_f64()
}

// --- Conv2D ---

fn conv2d_v1(
    c_in: usize,
    c_out: usize,
    k: usize,
    batch: usize,
    h: usize,
    w: usize,
    epochs: usize,
) -> f64 {
    let mut model = Conv2DV1::new(c_in, c_out, k, k, nbml::f::he);
    let mut optim = AdamWV1::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let out_h = h - k + 1;
    let out_w = w - k + 1;
    let x = Array4::ones((batch, c_in, h, w));
    let y: Array4<f64> = Array4::ones((batch, c_out, out_h, out_w)) * 0.5;

    let start = Instant::now();
    for _ in 0..epochs {
        let y_pred = model.forward(x.clone(), true);
        let d_loss = 2. * (&y_pred - &y);
        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }
    start.elapsed().as_secs_f64()
}

fn conv2d_v2(
    c_in: usize,
    c_out: usize,
    k: usize,
    batch: usize,
    h: usize,
    w: usize,
    epochs: usize,
) -> f64 {
    let mut model = Conv2DV2::new(c_in, c_out, k, k, nbml::f2::he);
    let mut optim = AdamWV2::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let out_h = h - k + 1;
    let out_w = w - k + 1;
    let x = Tensor::ones((batch, c_in, h, w));
    let y = Tensor::from_elem((batch, c_out, out_h, out_w), 0.5);

    let start = Instant::now();
    for _ in 0..epochs {
        let y_pred = model.forward(x.clone(), true);
        let d_loss = (&y_pred - &y) * 2.0;
        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }
    start.elapsed().as_secs_f64()
}

fn main() {
    let batch = 16;
    let seq = 16;
    let epochs = 100;

    println!("=== nn vs nn2 benchmark ({epochs} epochs, batch={batch}, seq={seq}) ===\n");

    // Transformer
    {
        let d_in = 64;
        let d_head = 8;
        let n_head = 4;
        println!("Config: d_in={d_in} d_head={d_head} n_head={n_head}");

        transformer_v1(d_in, d_head, n_head, batch, seq, 5);
        transformer_v2(d_in, d_head, n_head, batch, seq, 5);

        let t1 = transformer_v1(d_in, d_head, n_head, batch, seq, epochs);
        let t2 = transformer_v2(d_in, d_head, n_head, batch, seq, epochs);
        print_result("Transformer", t1, t2, epochs);
    }

    // LSTM
    {
        let d_model = 64;
        println!("Config: d_model={d_model}");

        lstm_v1(d_model, batch, seq, 5);
        lstm_v2(d_model, batch, seq, 5);

        let t1 = lstm_v1(d_model, batch, seq, epochs);
        let t2 = lstm_v2(d_model, batch, seq, epochs);
        print_result("LSTM", t1, t2, epochs);
    }

    // Gated Linear Attention
    {
        let d_in = 64;
        let d_head = 16;
        let n_head = 4;
        println!("Config: d_in={d_in} d_head={d_head} n_head={n_head}");

        gla_v1(d_in, d_head, n_head, batch, seq, 5);
        gla_v2(d_in, d_head, n_head, batch, seq, 5);

        let t1 = gla_v1(d_in, d_head, n_head, batch, seq, epochs);
        let t2 = gla_v2(d_in, d_head, n_head, batch, seq, epochs);
        print_result("Gated Linear Attention", t1, t2, epochs);
    }

    // Conv2D (im2col)
    {
        let c_in = 3;
        let c_out = 16;
        let k = 3;
        let h = 32;
        let w = 32;
        println!("Config: c_in={c_in} c_out={c_out} kernel={k}x{k} input={h}x{w}");

        conv2d_v1(c_in, c_out, k, batch, h, w, 5);
        conv2d_v2(c_in, c_out, k, batch, h, w, 5);

        let t1 = conv2d_v1(c_in, c_out, k, batch, h, w, epochs);
        let t2 = conv2d_v2(c_in, c_out, k, batch, h, w, epochs);
        print_result("Conv2D (im2col)", t1, t2, epochs);
    }
}

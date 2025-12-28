use nbml::{
    f::Activation,
    nn::TransformerEncoder,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array2, Array3, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

const EMBED_DIM: usize = 16;
const NUM_HEADS: usize = 2;
const SEQ_LEN: usize = 4;
const BATCH_SIZE: usize = 4;

#[test]
fn identity() {
    println!("=== Testing Transformer End-to-End ===\n");

    // Small transformer for testing
    let mut transformer = TransformerEncoder::new(
        EMBED_DIM,
        NUM_HEADS,
        2, // num_layers
        vec![(EMBED_DIM, EMBED_DIM, Activation::Relu)],
    );

    let mut optim = AdamW::default().with(&mut transformer);
    optim.learning_rate = 1e-3;

    println!("Test 1: Identity Task (learns to output = input)");
    println!("This tests if the transformer can learn with residual connections\n");

    let x = Array3::random((BATCH_SIZE, SEQ_LEN, EMBED_DIM), Uniform::new(-1., 1.));
    let y_target = x.clone();

    let mut losses = Vec::new();

    for epoch in 0..2000 {
        let mask = Array2::ones((x.dim().0, x.dim().1));
        let y_pred = transformer.forward(x.clone(), mask, true);
        let loss = (&y_pred - &y_target).mapv(|v| v.powi(2)).mean().unwrap();
        let d_loss = 2. * (&y_pred - &y_target) / (y_pred.len() as f64);

        transformer.backward(d_loss);
        optim.step(&mut transformer);
        transformer.zero_grads();

        if epoch % 200 == 0 {
            println!("  Epoch {}: loss = {:.6}", epoch, loss);
        }

        losses.push(loss);
    }

    let final_loss = losses.last().unwrap();
    let initial_loss = losses[0];

    println!("\nIdentity task results:");
    println!("  Initial loss: {:.6}", initial_loss);
    println!("  Final loss: {:.6}", final_loss);
    println!(
        "  Improvement: {:.2}%",
        (1.0 - final_loss / initial_loss) * 100.0
    );

    if *final_loss < 0.1 {
        println!("  ✓ PASS: Can learn identity mapping");
    } else {
        println!("  ✗ FAIL: Struggled with identity mapping");
        assert!(false);
    }
}

#[test]
pub fn mean_pooling() {
    // Test 2: Mean pooling (requires attention)
    println!("Test 2: Mean Pooling Task");
    println!("Output should be the mean of all sequence positions");
    println!("This tests if attention mechanism works\n");

    let mut transformer2 = TransformerEncoder::new(
        EMBED_DIM,
        NUM_HEADS,
        2,
        vec![(EMBED_DIM, EMBED_DIM, Activation::Relu)],
    );

    let mut optim2 = AdamW::default().with(&mut transformer2);
    optim2.learning_rate = 1e-3;

    let x2 = Array3::random((BATCH_SIZE, SEQ_LEN, EMBED_DIM), Uniform::new(-1., 1.));
    let mean = x2.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let y_target2 = mean
        .broadcast((BATCH_SIZE, SEQ_LEN, EMBED_DIM))
        .unwrap()
        .to_owned();

    let mut losses2 = Vec::new();

    for epoch in 0..3000 {
        let mask = Array2::ones((x2.dim().0, x2.dim().1));
        let y_pred = transformer2.forward(x2.clone(), mask, true);
        let loss = (&y_pred - &y_target2).mapv(|v| v.powi(2)).mean().unwrap();
        let d_loss = 2. * (&y_pred - &y_target2) / (y_pred.len() as f64);

        transformer2.backward(d_loss);
        optim2.step(&mut transformer2);
        transformer2.zero_grads();

        if epoch % 300 == 0 {
            println!("  Epoch {}: loss = {:.6}", epoch, loss);
        }

        losses2.push(loss);
    }

    let final_loss2 = losses2.last().unwrap();
    let initial_loss2 = losses2[0];

    println!("\nMean pooling task results:");
    println!("  Initial loss: {:.6}", initial_loss2);
    println!("  Final loss: {:.6}", final_loss2);
    println!(
        "  Improvement: {:.2}%",
        (1.0 - final_loss2 / initial_loss2) * 100.0
    );

    if *final_loss2 < 0.05 {
        println!("  ✓ PASS: Attention mechanism works");
    } else if *final_loss2 < 0.2 {
        println!("  ⚠ PARTIAL: Attention works but could be better");
    } else {
        println!("  ✗ FAIL: Attention mechanism may have issues");
        assert!(false);
    }
}

#[test]
fn gradient_flow() {
    let mut transformer = TransformerEncoder::new(
        EMBED_DIM,
        NUM_HEADS,
        3, // Deeper network to test gradient flow
        vec![(EMBED_DIM, EMBED_DIM, Activation::Relu)],
    );

    let x = Array3::random((BATCH_SIZE, SEQ_LEN, EMBED_DIM), Uniform::new(-1., 1.));
    let mask = Array2::ones((x.dim().0, x.dim().1));
    let y = transformer.forward(x.clone(), mask, true);

    // Create a gradient signal
    let d_loss = Array3::ones(y.dim());
    let dx = transformer.backward(d_loss);

    // Check gradient statistics
    let grad_mean = dx.mean().unwrap();
    let grad_std = dx.std(0.);
    let grad_max = dx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let grad_min = dx.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("  Gradient statistics:");
    println!("    Mean: {:.6}", grad_mean);
    println!("    Std:  {:.6}", grad_std);
    println!("    Min:  {:.6}", grad_min);
    println!("    Max:  {:.6}", grad_max);

    let has_nan = dx.iter().any(|&x| x.is_nan());
    let has_inf = dx.iter().any(|&x| x.is_infinite());
    let is_vanishing = grad_std < 1e-7;
    let is_exploding = grad_std > 1e3;

    if has_nan {
        println!("  ✗ FAIL: Gradients contain NaN");
    } else if has_inf {
        println!("  ✗ FAIL: Gradients contain Inf");
    } else if is_vanishing {
        println!("  ✗ FAIL: Gradients are vanishing (std too small)");
    } else if is_exploding {
        println!("  ✗ FAIL: Gradients are exploding (std too large)");
    } else {
        println!("  ✓ PASS: Gradients are healthy");
        return;
    }

    assert!(false);
}

#[test]
fn overfitting() {
    let mut transformer = TransformerEncoder::new(
        EMBED_DIM,
        NUM_HEADS,
        2,
        vec![(EMBED_DIM, EMBED_DIM, Activation::Relu)],
    );

    let mut optim = AdamW::default().with(&mut transformer);
    optim.learning_rate = 6e-3; // Higher learning rate for faster overfitting

    // Fixed small dataset - should memorize perfectly
    let x = Array3::random((BATCH_SIZE, SEQ_LEN, EMBED_DIM), Uniform::new(-1., 1.));
    let y_target = Array3::random((BATCH_SIZE, SEQ_LEN, EMBED_DIM), Uniform::new(-1., 1.));

    let mut final_loss = 1.0;

    for epoch in 0..5000 {
        let mask = Array2::ones((x.dim().0, x.dim().1));
        let y_pred = transformer.forward(x.clone(), mask, true);
        let loss = (&y_pred - &y_target).mapv(|v| v.powi(2)).mean().unwrap();
        let d_loss = 2. * (&y_pred - &y_target) / (y_pred.len() as f64);

        transformer.backward(d_loss);
        optim.step(&mut transformer);
        transformer.zero_grads();

        final_loss = loss;

        if epoch % 500 == 0 {
            println!("  Epoch {}: loss = {:.6}", epoch, loss);
        }

        // Early stopping if converged
        if loss < 1e-4 {
            println!("  Converged at epoch {}", epoch);
            break;
        }
    }

    println!("\n  Final loss: {:.6}", final_loss);

    if final_loss < 0.05 {
        println!("  ✓ PASS: Can memorize small dataset");
    } else if final_loss < 0.1 {
        println!("  ⚠ PARTIAL: Can learn but not perfectly");
        debug_assert!(false)
    } else {
        println!("  ✗ FAIL: Cannot even overfit small dataset");
        println!("  This suggests a fundamental issue with the model or optimizer");
        assert!(false)
    }
}

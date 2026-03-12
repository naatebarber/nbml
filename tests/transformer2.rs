use nbml::{
    Tensor,
    f2::positional_encoding_seq,
    nn2::Transformer,
    optim2::{Optimizer, ToParams, adam::AdamW},
    s,
    tensor::{Tensor2, Tensor3},
};
use rand::{RngExt, rng};

const EMBED_DIM: usize = 16;
const D_HEAD: usize = 8;
const NUM_HEADS: usize = 2;
const SEQ_LEN: usize = 4;
const BATCH_SIZE: usize = 4;

#[test]
fn identity() {
    println!("=== Testing Transformer End-to-End ===\n");

    // Small transformer for testing
    let mut transformer = Transformer::new_encoder(EMBED_DIM, D_HEAD, NUM_HEADS);

    let mut optim = AdamW::default().with(&mut transformer);
    optim.learning_rate = 1e-3;

    println!("Test 1: Identity Task (learns to output = input)");
    println!("This tests if the transformer can learn with residual connections\n");

    let x = Tensor::random_uniform((BATCH_SIZE, SEQ_LEN, EMBED_DIM)) * 2.0 - 1.0;
    let y_target = x.clone();

    let mut losses = Vec::new();

    for epoch in 0..2000 {
        let mask = Tensor2::ones((BATCH_SIZE, SEQ_LEN));
        let y_pred = transformer.forward(x.clone(), mask, true);
        let loss = (&y_pred - &y_target).mapv(|v| v.powi(2)).mean();
        let d_loss =
            (&y_pred - &y_target) * 2.0 / (y_pred.shape().iter().product::<usize>() as f64);

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
        println!("  PASS: Can learn identity mapping");
    } else {
        println!("  FAIL: Struggled with identity mapping");
        assert!(false);
    }
}

#[test]
pub fn mean_pooling() {
    // Test 2: Mean pooling (requires attention)
    println!("Test 2: Mean Pooling Task");
    println!("Output should be the mean of all sequence positions");
    println!("This tests if attention mechanism works\n");

    let mut transformer2 = Transformer::new_encoder(EMBED_DIM, NUM_HEADS, 2);

    let mut optim2 = AdamW::default().with(&mut transformer2);
    optim2.learning_rate = 1e-3;

    let x2 = Tensor::random_uniform((BATCH_SIZE, SEQ_LEN, EMBED_DIM)) * 2.0 - 1.0;
    let mean = x2.mean_axis(1).insert_axis(1);
    let y_target2 = mean.broadcast((BATCH_SIZE, SEQ_LEN, EMBED_DIM));

    let mut losses2 = Vec::new();

    for epoch in 0..3000 {
        let mask = Tensor2::ones((BATCH_SIZE, SEQ_LEN));
        let y_pred = transformer2.forward(x2.clone(), mask, true);
        let loss = (&y_pred - &y_target2).mapv(|v| v.powi(2)).mean();
        let d_loss =
            (&y_pred - &y_target2) * 2.0 / (y_pred.shape().iter().product::<usize>() as f64);

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
        println!("  PASS: Attention mechanism works");
    } else if *final_loss2 < 0.2 {
        println!("  PARTIAL: Attention works but could be better");
    } else {
        println!("  FAIL: Attention mechanism may have issues");
        assert!(false);
    }
}

#[test]
fn gradient_flow() {
    let mut transformer = Transformer::new_encoder(EMBED_DIM, D_HEAD, NUM_HEADS);

    let x = Tensor::random_uniform((BATCH_SIZE, SEQ_LEN, EMBED_DIM)) * 2.0 - 1.0;
    let mask = Tensor2::ones((BATCH_SIZE, SEQ_LEN));
    let y = transformer.forward(x.clone(), mask, true);

    // Create a gradient signal
    let d_loss = Tensor3::ones(y.shape());
    let dx = transformer.backward(d_loss);

    // Check gradient statistics
    let grad_mean = dx.mean();
    let grad_std = dx.std(0.);
    let grad_max = dx.max();
    let grad_min = dx.min();

    println!("  Gradient statistics:");
    println!("    Mean: {:.6}", grad_mean);
    println!("    Std:  {:.6}", grad_std);
    println!("    Min:  {:.6}", grad_min);
    println!("    Max:  {:.6}", grad_max);

    let has_nan = grad_mean.is_nan();
    let has_inf = grad_mean.is_infinite();
    let is_vanishing = grad_std < 1e-7;

    if has_nan {
        println!("  FAIL: Gradients contain NaN");
    } else if has_inf {
        println!("  FAIL: Gradients contain Inf");
    } else if is_vanishing {
        println!("  FAIL: Gradients are vanishing (std too small)");
    } else {
        println!("  PASS: Gradients are healthy");
        return;
    }

    assert!(false);
}

#[test]
fn overfitting() {
    let mut transformer = Transformer::new_encoder(EMBED_DIM, NUM_HEADS, 2);

    let mut optim = AdamW::default().with(&mut transformer);
    optim.learning_rate = 6e-3; // Higher learning rate for faster overfitting

    // Fixed small dataset - should memorize perfectly
    let x = Tensor::random_uniform((BATCH_SIZE, SEQ_LEN, EMBED_DIM)) * 2.0 - 1.0;
    let y_target = Tensor::random_uniform((BATCH_SIZE, SEQ_LEN, EMBED_DIM)) * 2.0 - 1.0;

    let mut final_loss = 1.0;

    for epoch in 0..5000 {
        let mask = Tensor2::ones((BATCH_SIZE, SEQ_LEN));
        let y_pred = transformer.forward(x.clone(), mask, true);
        let loss = (&y_pred - &y_target).mapv(|v| v.powi(2)).mean();
        let d_loss =
            (&y_pred - &y_target) * 2.0 / (y_pred.shape().iter().product::<usize>() as f64);

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
        println!("  PASS: Can memorize small dataset");
    } else if final_loss < 0.1 {
        println!("  PARTIAL: Can learn but not perfectly");
        debug_assert!(false)
    } else {
        println!("  FAIL: Cannot even overfit small dataset");
        println!("  This suggests a fundamental issue with the model or optimizer");
        assert!(false)
    }
}

#[test]
fn retrieval_by_marker() {
    // Input: [v1, v2, v3, MARKER, zeros]
    // Output: [zeros, zeros, zeros, zeros, v_marked]
    //
    // The marker position (one-hot in last dim) tells which vector to output.
    // This is pure content-based attention - what transformers are MADE for.

    let mut model = Transformer::new_decoder(16, 8, 2);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let mut final_loss = 0.;
    for e in 0..4000 {
        // 3 random vectors
        let v1 = Tensor::random_uniform(16);
        let v2 = Tensor::random_uniform(16);
        let v3 = Tensor::random_uniform(16);

        // Marker: which one to retrieve (0, 1, or 2)
        let target_idx = rng().random_range(0..3);
        let mut marker = Tensor::zeros(16);
        marker[[target_idx]] = 1.0;

        let target = match target_idx {
            0 => v1.clone(),
            1 => v2.clone(),
            _ => v3.clone(),
        };

        // Build sequence: [v1, v2, v3, marker, output_position]
        let x = Tensor::stack(0, &[&v1, &v2, &v3, &marker, &Tensor::zeros(16)]).insert_axis(0); // (1, 5, 16)

        let mut y = Tensor3::zeros((1, 5, 16));
        y.slice_assign(s![0, 4, ..], &target);

        let mask = Tensor2::ones((1, 5));
        let y_pred = model.forward(x, mask, true);

        // Only grade last position
        let pred_last = y_pred.slice(s![.., 4..5, ..]);
        let y_last = y.slice(s![.., 4..5, ..]);

        let loss = (&pred_last - &y_last).mapv(|v| v * v).mean();

        let mut d_loss = Tensor3::zeros((1, 5, 16));
        let grad_last = (&pred_last - &y_last) * 2.0;
        d_loss.slice_assign(s![.., 4..5, ..], &grad_last);

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if e % 200 == 0 {
            println!("epoch {e} loss {loss:.6}");
        }

        final_loss = loss;
    }

    assert!(
        final_loss < 0.1,
        "retrieval by marker failed: loss = {}",
        final_loss
    );
}

#[test]
fn fixed_position_retrieval() {
    // Input: [v0, v1, v2, v3, v4, zeros]
    // Output: [zeros, zeros, zeros, zeros, zeros, v2]
    //
    // Always retrieve position 2. No marker, just "learn that position 5 copies position 2"

    let mut model = Transformer::new_decoder(16, 8, 2);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let pe = positional_encoding_seq(6, 16).insert_axis(0);

    let mut final_loss = 0.;
    for e in 0..2000 {
        let vectors: Vec<Tensor> = (0..5).map(|_| Tensor::random_uniform(16)).collect();

        let mut x = Tensor3::zeros((1, 6, 16));
        for (i, v) in vectors.iter().enumerate() {
            x.slice_assign(s![0, i, ..], v);
        }
        // Position 5 is zeros (output position)

        let x_pe = &x + &pe;

        // Target: copy position 2 to position 5
        let mut y = Tensor3::zeros((1, 6, 16));
        y.slice_assign(s![0, 5, ..], &vectors[2]);

        let mask = Tensor2::ones((1, 6));
        let y_pred = model.forward(x_pe.clone(), mask, true);

        let pred_last = y_pred.slice(s![.., 5..6, ..]);
        let y_last = y.slice(s![.., 5..6, ..]);

        let loss = (&pred_last - &y_last).mapv(|v| v * v).mean();

        let mut d_loss = Tensor3::zeros((1, 6, 16));
        let grad_last = (&pred_last - &y_last) * 2.0;
        d_loss.slice_assign(s![.., 5..6, ..], &grad_last);

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if e % 200 == 0 {
            println!("epoch {e} loss {loss:.6}");
        }

        final_loss = loss;
    }

    assert!(
        final_loss < 0.1,
        "fixed position retrieval failed: loss = {}",
        final_loss
    );
}

#[test]
fn delayed_copy_single_token() {
    // Input: [v0, v1, v2, v3, v4, zeros, zeros, zeros, zeros, zeros]
    // Output: [zeros, zeros, zeros, zeros, zeros, v0, v1, v2, v3, v4]
    //
    // Position 5 copies position 0
    // Position 6 copies position 1
    // etc.

    let mut model = Transformer::new_decoder(16, 8, 2);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let pe = positional_encoding_seq(10, 16).insert_axis(0);

    let mut final_loss = 0.;
    for e in 0..3000 {
        let vectors: Vec<Tensor> = (0..5).map(|_| Tensor::random_uniform(16)).collect();

        let mut x = Tensor3::zeros((1, 10, 16));
        for (i, v) in vectors.iter().enumerate() {
            x.slice_assign(s![0, i, ..], v);
        }
        // Positions 5-9 are zeros

        let x_pe = &x + &pe;

        // Target: positions 5-9 copy positions 0-4
        let mut y = Tensor3::zeros((1, 10, 16));
        for (i, v) in vectors.iter().enumerate() {
            y.slice_assign(s![0, i + 5, ..], v);
        }

        let mask = Tensor2::ones((1, 10));
        let y_pred = model.forward(x_pe.clone(), mask, true);

        // Grade only output positions
        let pred_out = y_pred.slice(s![.., 5.., ..]);
        let y_out = y.slice(s![.., 5.., ..]);

        let loss = (&pred_out - &y_out).mapv(|v| v * v).mean();

        let mut d_loss = Tensor3::zeros((1, 10, 16));
        let grad_out = (&pred_out - &y_out) * 2.0;
        d_loss.slice_assign(s![.., 5.., ..], &grad_out);

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if e % 300 == 0 {
            println!("epoch {e} loss {loss:.6}");
        }

        final_loss = loss;
    }

    assert!(
        final_loss < 0.1,
        "delayed copy failed: loss = {}",
        final_loss
    );
}

#[test]
fn memorize_one_delayed_copy() {
    let mut model = Transformer::new_decoder(16, 8, 2);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let pe = positional_encoding_seq(10, 16).insert_axis(0);

    // Fixed vectors - same every epoch
    let vectors: Vec<Tensor> = (0..5).map(|_| Tensor::random_uniform(16)).collect();

    let mut x = Tensor3::zeros((1, 10, 16));
    for (i, v) in vectors.iter().enumerate() {
        x.slice_assign(s![0, i, ..], v);
    }
    let x_pe = &x + &pe;

    let mut y = Tensor3::zeros((1, 10, 16));
    for (i, v) in vectors.iter().enumerate() {
        y.slice_assign(s![0, i + 5, ..], v);
    }

    let mut final_loss = f64::MAX;

    for e in 0..5000 {
        let mask = Tensor2::ones((1, 10));
        let y_pred = model.forward(x_pe.clone(), mask, true);

        let pred_out = y_pred.slice(s![.., 5.., ..]);
        let y_out = y.slice(s![.., 5.., ..]);

        let loss = (&pred_out - &y_out).mapv(|v| v * v).mean();

        let mut d_loss = Tensor3::zeros((1, 10, 16));
        let grad_out = (&pred_out - &y_out) * 2.0;
        d_loss.slice_assign(s![.., 5.., ..], &grad_out);

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if e % 500 == 0 {
            println!("epoch {e} loss {loss:.6}");
        }

        final_loss = loss;
    }

    assert!(
        final_loss < 0.001,
        "failed to memorize one delayed copy: loss = {}",
        final_loss
    );
}

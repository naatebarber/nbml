#![allow(deprecated)]

use nbml::{
    nn::{AttentionHead, LayerNorm},
    optim::param::ToParams,
};
use ndarray::{Array2, Array3};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn attention_gradients() {
    let d_in = 8;
    let d_head = 4;
    let n_head = 2;
    let batch_size = 2;
    let seq_len = 3;

    let mut attn = AttentionHead::new(d_in, d_head, n_head);

    // Small random input
    let x = Array3::random((batch_size, seq_len, d_in), Uniform::new(-0.5, 0.5));

    // No masking for simplicity (all ones = attend to everything)
    let mask = Array2::ones((batch_size, seq_len));

    println!("Testing Attention Gradients...\n");

    // Analytical gradient via backward pass
    let y = attn.forward(&x, &mask, false, true);
    let d_loss = Array3::ones(y.dim()); // Upstream gradient
    let dx_analytical = attn.backward(d_loss.clone());

    // Numerical gradient via finite differences
    let epsilon = 1e-5;
    let mut dx_numerical = Array3::zeros(x.dim());

    println!("Computing numerical gradients (this may take a moment)...");

    for b in 0..batch_size {
        for s in 0..seq_len {
            for f in 0..d_in {
                // Perturb input
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[b, s, f]] += epsilon;
                x_minus[[b, s, f]] -= epsilon;

                // Forward passes
                let mut attn_plus = AttentionHead::new(d_in, d_head, n_head);
                attn_plus.qkv_w = attn.qkv_w.clone();
                attn_plus.qkv_b = attn.qkv_b.clone();
                attn_plus.o_w = attn.o_w.clone();
                attn_plus.o_b = attn.o_b.clone();
                let y_plus = attn_plus.forward(&x_plus, &mask, false, false);

                let mut attn_minus = AttentionHead::new(d_in, d_head, n_head);
                attn_minus.qkv_w = attn.qkv_w.clone();
                attn_minus.qkv_b = attn.qkv_b.clone();
                attn_minus.o_w = attn.o_w.clone();
                attn_minus.o_b = attn.o_b.clone();
                let y_minus = attn_minus.forward(&x_minus, &mask, false, false);

                // Compute numerical gradient
                let loss_plus = (&d_loss * &y_plus).sum();
                let loss_minus = (&d_loss * &y_minus).sum();
                dx_numerical[[b, s, f]] = (loss_plus - loss_minus) / (2. * epsilon);
            }
        }
        println!("  Batch {}/{} complete", b + 1, batch_size);
    }

    // Compare gradients
    let diff = (&dx_analytical - &dx_numerical).mapv(|v| v.abs());
    let max_diff = diff.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let rel_error = &diff / &dx_numerical.mapv(|v| v.abs() + 1e-8);
    let max_rel_error = rel_error.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n=== Gradient Comparison ===");
    println!("Analytical gradient sample (first 3 elements):");
    for i in 0..3.min(dx_analytical.len()) {
        println!("  [{i}] = {:.6}", dx_analytical.as_slice().unwrap()[i]);
    }

    println!("\nNumerical gradient sample (first 3 elements):");
    for i in 0..3.min(dx_numerical.len()) {
        println!("  [{i}] = {:.6}", dx_numerical.as_slice().unwrap()[i]);
    }

    println!("\nMax absolute difference: {:.2e}", max_diff);
    println!("Max relative error: {:.2e}", max_rel_error);

    // Test parameter gradients too
    println!("\n=== Testing Parameter Gradients ===");
    attn.zero_grads();
    test_param_gradient(&mut attn, &x, &mask, "qkv_w", epsilon);
    attn.zero_grads();
    test_param_gradient(&mut attn, &x, &mask, "o_w", epsilon);

    // Success criteria
    if max_diff < 1e-4 {
        println!("\n✓ PASS: Attention input gradients are correct!");
    } else {
        println!("\n✗ FAIL: Attention input gradients are incorrect!");
        println!("Expected max diff < 1e-4, got {:.2e}", max_diff);
    }
}

fn test_param_gradient(
    attn: &mut AttentionHead,
    x: &Array3<f64>,
    mask: &Array2<f64>,
    param_name: &str,
    epsilon: f64,
) {
    println!("\nTesting {} gradients...", param_name);

    let y = attn.forward(x, mask, false, true);
    let d_loss = Array3::ones(y.dim());
    let _ = attn.backward(d_loss.clone());

    // Get analytical gradient
    let analytical_grad = match param_name {
        "qkv_w" => attn.d_qkv_w.clone(),
        "o_w" => attn.d_o_w.clone(),
        _ => panic!("Unknown parameter"),
    };

    // Sample a few elements for numerical check (checking all would be slow)
    let samples = 5;
    let mut total_error = 0.0;

    for _ in 0..samples {
        let i = rand::random_range(0..1e10 as usize) % analytical_grad.shape()[0];
        let j = rand::random_range(0..1e10 as usize) % analytical_grad.shape()[1];

        // Numerical gradient for this element
        let original_val = match param_name {
            "qkv_w" => attn.qkv_w[[i, j]],
            "o_w" => attn.o_w[[i, j]],
            _ => panic!("Unknown parameter"),
        };

        // Plus perturbation
        match param_name {
            "qkv_w" => attn.qkv_w[[i, j]] = original_val + epsilon,
            "o_w" => attn.o_w[[i, j]] = original_val + epsilon,
            _ => {}
        }
        let y_plus = attn.forward(x, mask, false, false);
        let loss_plus = (&d_loss * &y_plus).sum();

        // Minus perturbation
        match param_name {
            "qkv_w" => attn.qkv_w[[i, j]] = original_val - epsilon,
            "o_w" => attn.o_w[[i, j]] = original_val - epsilon,
            _ => {}
        }
        let y_minus = attn.forward(x, mask, false, false);
        let loss_minus = (&d_loss * &y_minus).sum();

        // Restore
        match param_name {
            "qkv_w" => attn.qkv_w[[i, j]] = original_val,
            "o_w" => attn.o_w[[i, j]] = original_val,
            _ => {}
        }

        let numerical = (loss_plus - loss_minus) / (2. * epsilon);
        let analytical = analytical_grad[[i, j]];
        let error = (analytical - numerical).abs();

        total_error += error;

        if error > 1e-4 {
            println!(
                "  Sample [{},{}]: analytical={:.6}, numerical={:.6}, diff={:.2e}",
                i, j, analytical, numerical, error
            );
        }
    }

    let avg_error = total_error / samples as f64;
    if avg_error < 1e-4 {
        println!(
            "  ✓ {} gradients correct (avg error: {:.2e})",
            param_name, avg_error
        );
    } else {
        println!(
            "  ✗ {} gradients incorrect (avg error: {:.2e})",
            param_name, avg_error
        );
        assert!(false);
    }
}

#[test]
fn test_layernorm_gradients() {
    let features = 5;
    let mut ln = LayerNorm::new(features);

    // Small random input
    let x = Array3::random((2, 3, features), Uniform::new(-1., 1.));

    // Analytical gradient via backward pass
    let y = ln.forward(x.clone(), true);
    let d_loss = Array3::random(y.dim(), Uniform::new(0., 1.)); // Gradient from "loss"
    let dx_analytical = ln.backward(d_loss.clone());

    // Numerical gradient via finite differences
    let epsilon = 1e-5;
    let mut dx_numerical = Array3::zeros(x.dim());

    for b in 0..2 {
        for s in 0..3 {
            for f in 0..features {
                // Perturb input slightly
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[b, s, f]] += epsilon;
                x_minus[[b, s, f]] -= epsilon;

                // Forward pass with perturbed inputs
                let mut ln_plus = LayerNorm::new(features);
                ln_plus.gamma = ln.gamma.clone();
                ln_plus.beta = ln.beta.clone();
                let y_plus = ln_plus.forward(x_plus, false);

                let mut ln_minus = LayerNorm::new(features);
                ln_minus.gamma = ln.gamma.clone();
                ln_minus.beta = ln.beta.clone();
                let y_minus = ln_minus.forward(x_minus, false);

                // Compute numerical gradient: dL/dx ≈ (L(x+ε) - L(x-ε)) / 2ε
                // Where L = sum of (d_loss * y)
                let loss_plus = (&d_loss * &y_plus).sum();
                let loss_minus = (&d_loss * &y_minus).sum();
                dx_numerical[[b, s, f]] = (loss_plus - loss_minus) / (2. * epsilon);
            }
        }
    }

    // Compare analytical vs numerical
    let diff = (&dx_analytical - &dx_numerical).mapv(|v| v.abs());
    let max_diff = diff.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let rel_error = &diff / &dx_numerical.mapv(|v| v.abs() + 1e-8);
    let max_rel_error = rel_error.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Analytical gradient:\n{:.6}", dx_analytical);
    println!("\nNumerical gradient:\n{:.6}", dx_numerical);
    println!("\nAbsolute difference:\n{:.6}", diff);
    println!("\nMax absolute difference: {:.2e}", max_diff);
    println!("Max relative error: {:.2e}", max_rel_error);

    // Success criteria
    if max_diff < 1e-5 {
        println!("\n✓ PASS: LayerNorm gradients are correct!");
    } else {
        println!("\n✗ FAIL: LayerNorm gradients are incorrect!");
        println!("Expected max diff < 1e-5, got {:.2e}", max_diff);
        assert!(false);
    }
}

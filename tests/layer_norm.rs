#![allow(deprecated)]

use nbml::layers::LayerNorm;
use ndarray::Array3;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

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

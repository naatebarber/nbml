use nbml::{
    layers::Linear,
    optim::{AdamW, Optimizer, ToParams},
};
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn test_linear_gradients() {
    let batch = 5;
    let features = 5;

    let mut lin = Linear::new(features, features);

    // Small random input
    let x = Array2::random((batch, features), Uniform::new(-1., 1.));

    // Analytical gradient via backward pass
    let y = lin.forward(x.clone(), true);
    let d_loss = Array2::random(y.dim(), Uniform::new(0., 1.));
    let dx_analytical = lin.backward(d_loss.clone());

    // Numerical gradient via finite differences
    let epsilon = 1e-5;
    let mut dx_numerical = Array2::zeros(x.dim());

    for b in 0..batch {
        for f in 0..features {
            // Perturb input slightly
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[[b, f]] += epsilon;
            x_minus[[b, f]] -= epsilon;

            // Forward pass with perturbed inputs
            let mut lin_plus = Linear::new(features, features);
            lin_plus.w = lin.w.clone();
            lin_plus.b = lin.b.clone();
            let y_plus = lin_plus.forward(x_plus, false);

            let mut lin_minus = Linear::new(features, features);
            lin_minus.w = lin.w.clone();
            lin_minus.b = lin.b.clone();
            let y_minus = lin_minus.forward(x_minus, false);

            // Compute numerical gradient: dL/dx ≈ (L(x+ε) - L(x-ε)) / 2ε
            // Where L = sum of (d_loss * y)
            let loss_plus = (&d_loss * &y_plus).sum();
            let loss_minus = (&d_loss * &y_minus).sum();
            dx_numerical[[b, f]] = (loss_plus - loss_minus) / (2. * epsilon);
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
        println!("\n✓ PASS: Linear gradients are correct!");
    } else {
        println!("\n✗ FAIL: Linear gradients are incorrect!");
        println!("Expected max diff < 1e-5, got {:.2e}", max_diff);
        assert!(false);
    }
}

#[test]
fn fixed_transform() {
    let mut lin = Linear::new(5, 5);
    let mut optim = AdamW::default().with(&mut lin);
    optim.learning_rate = 1e-2;

    let start = Array2::random((1, 5), Uniform::new(0., 1.));
    let end = Array2::random((1, 5), Uniform::new(0., 1.));

    let mut loss = 0.;
    for e in 0..100 {
        let prediction = lin.forward(start.clone(), true);
        loss = (&prediction - &end).pow2().mean().unwrap();
        println!("epoch={e} loss={loss}");
        let d_loss = 2. * (&prediction - &end);

        lin.backward(d_loss);
        optim.step(&mut lin);
        lin.zero_grads();
    }

    let max_loss = 0.1;
    assert!(loss < max_loss, "linear failed to learn transformation");
}

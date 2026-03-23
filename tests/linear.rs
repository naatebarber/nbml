use nbml::{
    f, layers::Linear, optim::{AdamW, Optimizer, ToParams}
};
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn test_linear_gradients() {
    let d_in = 4;
    let batch_size = 3;
    let eps = 1e-3;

    let mut ln = Linear::new(d_in, d_in);
    let x = f::xavier_normal((batch_size, d_in));

    // forward + backward
    let out = ln.forward(x.clone(), true);
    let d_loss = Array2::ones(out.dim());
    let d_x = ln.backward(d_loss.clone());

    // numerical gradient for each element of x
    for b in 0..batch_size {
        for f in 0..d_in {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[[b, f]] += eps;
            x_minus[[b, f]] -= eps;

            let mut ln_plus = ln.clone();
            let mut ln_minus = ln.clone();
            let out_plus = ln_plus.forward(x_plus, false);
            let out_minus = ln_minus.forward(x_minus, false);

            let numerical = (&out_plus - &out_minus).sum() / (2.0 * eps);
            let analytical = d_x[[b, f]];

            let diff = (numerical - analytical).abs();
            assert!(
                diff < 1e-3,
                "gradient mismatch at [{},{}]: numerical={}, analytical={}, diff={}",
                b,
                f,
                numerical,
                analytical,
                diff
            );
        }
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

#![allow(deprecated)]

use nbml::{
    f, layers::LayerNorm, optim::{ToIntermediates, ToParams}
};
use ndarray::Array3;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn intermediate_caching() {
    let mut model = LayerNorm::new(4);
    let x = Array3::random((2, 3, 4), Uniform::new(0., 1.));
    let x2 = Array3::random((2, 3, 4), Uniform::new(0., 1.));
    let d = Array3::ones((2, 3, 4));

    model.forward(x.clone(), true);
    let cache_a = model.stash_intermediates();
    model.backward(d.clone());
    let grads_a = model.grads.clone();
    model.zero_grads();

    model.forward(x2.clone(), true);
    let cache_b = model.stash_intermediates();
    model.backward(d.clone());
    let grads_b = model.grads.clone();
    model.zero_grads();

    model.apply_intermediates(cache_a.clone());
    model.backward(d.clone());
    let grads_c = model.grads.clone();

    assert!(x != x2);
    assert!(cache_a != cache_b);
    assert!(grads_a.d_gamma != grads_b.d_gamma);
    assert!(
        grads_a.d_gamma == grads_c.d_gamma,
        "intermediate caching process fucks with gradients"
    );
}

#[test]
fn test_layernorm_gradients() {
    let d_in = 4;
    let batch_size = 3;
    let seq_len = 3;
    let eps = 1e-3;

    let mut ln = LayerNorm::new(d_in);
    let x = f::xavier_normal((batch_size * seq_len, d_in))
        .into_shape_clone((batch_size, seq_len, d_in))
        .unwrap();

    // forward + backward
    let out = ln.forward(x.clone(), true);
    let d_loss = Array3::ones(out.dim());
    let d_x = ln.backward(d_loss.clone());

    // numerical gradient for each element of x
    for b in 0..batch_size {
        for s in 0..seq_len {
            for f in 0..d_in {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[b, s, f]] += eps;
                x_minus[[b, s, f]] -= eps;

                let mut ln_plus = ln.clone();
                let mut ln_minus = ln.clone();
                let out_plus = ln_plus.forward(x_plus, false);
                let out_minus = ln_minus.forward(x_minus, false);

                let numerical = (&out_plus - &out_minus).sum() / (2.0 * eps);
                let analytical = d_x[[b, s, f]];

                let diff = (numerical - analytical).abs();
                assert!(
                    diff < 1e-3,
                    "gradient mismatch at [{},{},{}]: numerical={}, analytical={}, diff={}",
                    b,
                    s,
                    f,
                    numerical,
                    analytical,
                    diff
                );
            }
        }
    }
}

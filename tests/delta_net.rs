use nbml::{
    f,
    nn::DeltaNet,
    optim::{ToIntermediates, ToParams},
};
use ndarray::Array3;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn delta_net_intermediate_caching() {
    let mut model = DeltaNet::new(4, 4);
    let x = Array3::random((2, 3, 4), Uniform::new(0., 1.).unwrap());
    let x2 = Array3::random((2, 3, 4), Uniform::new(0., 1.).unwrap());
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
    assert!(grads_a.d_w_o != grads_b.d_w_o);
    assert!(
        grads_a.d_w_o == grads_c.d_w_o,
        "intermediate caching process fucks with gradients"
    );
}

#[test]
fn delta_net_gradient_check() {
    let d_in = 4;
    let d_head = 4;
    let batch_size = 3;
    let seq_len = 3;
    let eps = 1e-3;

    let mut attn = DeltaNet::new(d_in, d_head);
    let x = f::xavier_normal((batch_size * seq_len, d_in))
        .into_shape_clone((batch_size, seq_len, d_in))
        .unwrap();

    let out = attn.forward(x.clone(), true);
    let d_loss = Array3::ones(out.dim());
    let d_x = attn.backward(d_loss.clone());

    for b in 0..batch_size {
        for s in 0..seq_len {
            for f in 0..d_in {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[b, s, f]] += eps;
                x_minus[[b, s, f]] -= eps;

                let mut attn_plus = attn.clone();
                let mut attn_minus = attn.clone();
                let out_plus = attn_plus.forward(x_plus, false);
                let out_minus = attn_minus.forward(x_minus, false);

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

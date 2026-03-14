use nbml::{
    f,
    nn::LinearSelfAttention,
    nn::experimental::linear_cross_attention::LinearCrossAttention,
    optim::{AdamW, Optimizer, ToIntermediates, ToParams},
};
use ndarray::{Array1, Array3, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::{rng, seq::IteratorRandom};

#[test]
fn linear_self_attention_intermediate_caching() {
    let mut model = LinearSelfAttention::new(4, 4);
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
    assert!(grads_a.d_w_o != grads_b.d_w_o);
    assert!(
        grads_a.d_w_o == grads_c.d_w_o,
        "intermediate caching process fucks with gradients"
    );
}

#[test]
fn linear_cross_attention_intermediate_caching() {
    let mut model = LinearCrossAttention::new(4, 4);
    let x_q = Array3::random((2, 3, 4), Uniform::new(0., 1.));
    let x_kv = Array3::random((2, 2, 4), Uniform::new(0., 1.));
    let x_q2 = Array3::random((2, 3, 4), Uniform::new(0., 1.));
    let x_kv2 = Array3::random((2, 2, 4), Uniform::new(0., 1.));
    let d = Array3::ones((2, 3, 4));

    model.forward(x_q.clone(), x_kv.clone(), true);
    let cache_a = model.stash_intermediates();
    model.backward(d.clone());
    let grads_a = model.grads.clone();
    model.zero_grads();

    model.forward(x_q2.clone(), x_kv2.clone(), true);
    let cache_b = model.stash_intermediates();
    model.backward(d.clone());
    let grads_b = model.grads.clone();
    model.zero_grads();

    model.apply_intermediates(cache_a.clone());
    model.backward(d.clone());
    let grads_c = model.grads.clone();

    assert!(x_q != x_q2);
    assert!(cache_a != cache_b);
    assert!(grads_a.d_w_q != grads_b.d_w_q);
    assert!(
        grads_a.d_w_q == grads_c.d_w_q,
        "intermediate caching process fucks with gradients"
    );
}

#[test]
fn linear_self_attention_gradient_check() {
    let d_in = 4;
    let d_head = 4;
    let batch_size = 3;
    let seq_len = 3;
    let eps = 1e-5;

    let mut attn = LinearSelfAttention::new(d_in, d_head);
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
                    diff < 1e-4,
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

#[test]
fn linear_cross_attention_gradient_check() {
    let d_in = 4;
    let d_head = 4;
    let batch_size = 3;
    let seq_len_q = 3;
    let seq_len_kv = 2;
    let eps = 1e-5;

    let mut attn = LinearCrossAttention::new(d_in, d_head);

    let x_q = f::xavier_normal((batch_size * seq_len_q, d_in))
        .into_shape_clone((batch_size, seq_len_q, d_in))
        .unwrap();
    let x_kv = f::xavier_normal((batch_size * seq_len_kv, d_in))
        .into_shape_clone((batch_size, seq_len_kv, d_in))
        .unwrap();

    let out = attn.forward(x_q.clone(), x_kv.clone(), true);
    let d_loss = Array3::ones(out.dim());
    let (dx_q, dx_kv) = attn.backward(d_loss.clone());

    for b in 0..batch_size {
        for s in 0..seq_len_q {
            for f in 0..d_head {
                let mut x_plus_q = x_q.clone();
                let mut x_minus_q = x_q.clone();
                x_plus_q[[b, s, f]] += eps;
                x_minus_q[[b, s, f]] -= eps;

                let mut attn_plus = attn.clone();
                let mut attn_minus = attn.clone();
                let out_plus = attn_plus.forward(x_plus_q, x_kv.clone(), false);
                let out_minus = attn_minus.forward(x_minus_q, x_kv.clone(), false);

                let numerical = (&out_plus - &out_minus).sum() / (2.0 * eps);
                let analytical = dx_q[[b, s, f]];

                let diff = (numerical - analytical).abs();
                assert!(
                    diff < 1e-4,
                    "x_q gradient mismatch at [{},{},{}]: numerical={}, analytical={}, diff={}",
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

    for b in 0..batch_size {
        for s in 0..seq_len_kv {
            for f in 0..d_in {
                let mut x_plus_kv = x_kv.clone();
                let mut x_minus_kv = x_kv.clone();
                x_plus_kv[[b, s, f]] += eps;
                x_minus_kv[[b, s, f]] -= eps;

                let mut attn_plus = attn.clone();
                let mut attn_minus = attn.clone();
                let out_plus = attn_plus.forward(x_q.clone(), x_plus_kv, false);
                let out_minus = attn_minus.forward(x_q.clone(), x_minus_kv, false);

                let numerical = (&out_plus - &out_minus).sum() / (2.0 * eps);
                let analytical = dx_kv[[b, s, f]];

                let diff = (numerical - analytical).abs();
                assert!(
                    diff < 1e-4,
                    "x_kv gradient mismatch at [{},{},{}]: numerical={}, analytical={}, diff={}",
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

fn make_associative_recall_dataset(
    batch_size: usize,
    d_model: usize,
) -> (Array3<f64>, Array3<f64>) {
    let num_pairs = 3;
    let seq_len = num_pairs * 2 + 2;

    let mut x = Array3::zeros((batch_size, seq_len, d_model));
    let mut y = Array3::zeros((batch_size, seq_len, d_model));

    for b in 0..batch_size {
        let keys: Vec<Array1<f64>> = (0..num_pairs)
            .map(|_| Array1::random(d_model, Uniform::new(0., 10.)))
            .collect();
        let values: Vec<Array1<f64>> = (0..num_pairs)
            .map(|_| Array1::random(d_model, Uniform::new(0., 10.)))
            .collect();

        for i in 0..num_pairs {
            x.slice_mut(s![b, i * 2, ..]).assign(&keys[i]);
            x.slice_mut(s![b, i * 2 + 1, ..]).assign(&values[i]);
        }

        x.slice_mut(s![b, num_pairs * 2, ..])
            .assign(&Array1::from_elem(d_model, 1.0));

        let query_idx = (0..num_pairs).choose(&mut rng()).unwrap();
        x.slice_mut(s![b, num_pairs * 2 + 1, ..])
            .assign(&keys[query_idx]);

        y.slice_mut(s![b, num_pairs * 2 + 1, ..])
            .assign(&values[query_idx]);
    }

    (x, y)
}

#[test]
fn linear_self_attention_associative_recall() {
    let d_in = 8;
    let d_head = 8;
    let batch_size = 10;

    let mut attn = LinearSelfAttention::new(d_in, d_head);
    let mut optim = AdamW::default().with(&mut attn);
    optim.learning_rate = 1e-3;

    let mut final_loss = f64::MAX;

    let (x, y) = make_associative_recall_dataset(batch_size, d_in);

    for epoch in 0..3000 {
        let y_pred = attn.forward(x.clone(), true);

        let mut y_pred_mask = Array3::zeros(y.dim());
        y_pred_mask
            .slice_mut(s![.., -1, ..])
            .assign(&y_pred.slice(s![.., -1, ..]));

        let d_loss = 2. * (&y_pred_mask - &y);
        let loss = (&y_pred_mask - &y).pow2().mean().unwrap();

        attn.backward(d_loss);
        optim.step(&mut attn);
        attn.zero_grads();

        if epoch % 100 == 0 {
            println!("epoch {} loss {}", epoch, loss);
        }

        final_loss = loss;
    }

    let max_loss = 0.1;
    assert!(
        final_loss < max_loss,
        "linear self attention failed associative recall with loss {final_loss} > {max_loss}"
    );
}

fn make_cross_attention_associative_recall_dataset(
    batch_size: usize,
    d_model: usize,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let num_pairs = 3;
    let num_queries = 2;

    let mut x_kv = Array3::zeros((batch_size, num_pairs * 2, d_model));
    let mut x_q = Array3::zeros((batch_size, num_queries, d_model));
    let mut y = Array3::zeros((batch_size, num_queries, d_model));

    for b in 0..batch_size {
        let keys: Vec<Array1<f64>> = (0..num_pairs)
            .map(|_| Array1::random(d_model, Uniform::new(0., 10.)))
            .collect();
        let values: Vec<Array1<f64>> = (0..num_pairs)
            .map(|_| Array1::random(d_model, Uniform::new(0., 10.)))
            .collect();

        for i in 0..num_pairs {
            x_kv.slice_mut(s![b, i * 2, ..]).assign(&keys[i]);
            x_kv.slice_mut(s![b, i * 2 + 1, ..]).assign(&values[i]);
        }

        for q in 0..num_queries {
            let query_idx = (0..num_pairs).choose(&mut rng()).unwrap();
            x_q.slice_mut(s![b, q, ..]).assign(&keys[query_idx]);
            y.slice_mut(s![b, q, ..]).assign(&values[query_idx]);
        }
    }

    (x_q, x_kv, y)
}

#[test]
#[ignore]
fn linear_cross_attention_associative_recall() {
    let d_in = 16;
    let d_head = 16;
    let batch_size = 10;

    let mut attn = LinearCrossAttention::new(d_in, d_head);
    let mut optim = AdamW::default().with(&mut attn);
    optim.learning_rate = 1e-3;

    let mut final_loss = f64::MAX;
    let (x_q, x_kv, y) = make_cross_attention_associative_recall_dataset(batch_size, d_in);

    for epoch in 0..5000 {
        let y_pred = attn.forward(x_q.clone(), x_kv.clone(), true);
        let d_loss = 2. * (&y_pred - &y);
        let loss = (&y_pred - &y).pow2().mean().unwrap();
        attn.backward(d_loss);
        optim.step(&mut attn);
        attn.zero_grads();
        if epoch % 100 == 0 {
            println!("epoch {} loss {}", epoch, loss);
        }
        final_loss = loss;
    }

    let max_loss = 0.1;
    assert!(
        final_loss < max_loss,
        "linear cross attention failed associative recall with loss {final_loss} > {max_loss}"
    );
}

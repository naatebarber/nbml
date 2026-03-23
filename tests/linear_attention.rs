use nbml::{
    f,
    nn::{GatedLinearAttention, LinearAttention},
    optim::{AdamW, Optimizer, ToIntermediates, ToParams},
};
use ndarray::{Array1, Array3, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::{rng, seq::IteratorRandom};

#[test]
fn linear_attention_intermediate_caching() {
    let mut model = LinearAttention::new(4, 4);
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
fn linear_attention_gradient_check() {
    let d_in = 4;
    let d_head = 4;
    let batch_size = 3;
    let seq_len = 3;
    let eps = 1e-3;

    let mut attn = LinearAttention::new(d_in, d_head);
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

#[test]
fn gated_linear_attention_gradient_check() {
    let d_in = 4;
    let d_head = 4;
    let batch_size = 3;
    let seq_len = 5;
    let eps = 1e-3;

    let mut attn = GatedLinearAttention::new(d_in, d_head);
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

fn make_associative_recall_dataset(
    batch_size: usize,
    d_model: usize,
) -> (Array3<f32>, Array3<f32>) {
    let num_pairs = 3;
    let seq_len = num_pairs * 2 + 2;

    let mut x = Array3::zeros((batch_size, seq_len, d_model));
    let mut y = Array3::zeros((batch_size, seq_len, d_model));

    for b in 0..batch_size {
        let keys: Vec<Array1<f32>> = (0..num_pairs)
            .map(|_| Array1::random(d_model, Uniform::new(0., 10.)))
            .collect();
        let values: Vec<Array1<f32>> = (0..num_pairs)
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
fn linear_attention_associative_recall() {
    let d_in = 8;
    let d_head = 8;
    let batch_size = 10;

    let mut attn = LinearAttention::new(d_in, d_head);
    let mut optim = AdamW::default().with(&mut attn);
    optim.learning_rate = 1e-3;

    let mut final_loss = f32::MAX;

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

#[test]
fn gated_linear_attention_associative_recall() {
    let d_in = 8;
    let d_head = 8;
    let batch_size = 10;

    let mut attn = GatedLinearAttention::new(d_in, d_head);
    let mut optim = AdamW::default().with(&mut attn);
    optim.learning_rate = 1e-3;

    let mut final_loss = f32::MAX;

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

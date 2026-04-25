use nbml::{
    f::{self, causal_mask},
    nn::{Attention, CrossAttention, SelfAttention},
    optim::{AdamW, Optimizer, ToIntermediates, ToParams},
};
use ndarray::{Array1, Array2, Array3, Axis, concatenate, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::{rng, seq::IteratorRandom};

#[test]
fn intermediate_caching() {
    let mut attn = SelfAttention::new(10, 5, 2);
    let x = Array3::random((1, 1, 10), Uniform::new(0., 1.).unwrap());
    let x2 = Array3::random((1, 1, 10), Uniform::new(0., 1.).unwrap());

    let d = Array3::ones((1, 1, 10));

    attn.forward(x.clone(), Array3::ones((1, 1, 1)), true);
    let cache_a = attn.stash_intermediates();
    attn.backward(d.clone());
    let grads_a = attn.grads.clone();
    attn.zero_grads();

    attn.forward(x2.clone(), Array3::ones((1, 1, 1)), true);
    let cache_b = attn.stash_intermediates();
    attn.backward(d.clone());
    let grads_b = attn.grads.clone();
    attn.zero_grads();

    attn.apply_intermediates(cache_a.clone());
    attn.backward(d.clone());
    let grads_c = attn.grads.clone();

    assert!(x != x2);
    assert!(cache_a != cache_b);
    assert!(grads_a.d_w_o != grads_b.d_w_o);
    assert!(
        grads_a.d_w_o == grads_c.d_w_o,
        "intermediate caching process fucks with gradients"
    )
}

#[test]
fn attention_scores() {
    let mut attn = Attention::new();

    let q = Array3::from_shape_vec((1, 1, 2), vec![0., 1.]).unwrap();
    let k = Array3::from_shape_vec((1, 2, 2), vec![0., 1., 1., 0.]).unwrap();
    let v = Array3::from_shape_vec((1, 2, 2), vec![10., 0., 0., 10.]).unwrap();

    let mask = Array3::ones((1, 1, 2));
    let out = attn.forward(q, k, v, mask, true);

    let weights = attn
        .weights()
        .to_owned()
        .remove_axis(Axis(0))
        .remove_axis(Axis(0));
    let output = out.remove_axis(Axis(0)).remove_axis(Axis(0));

    assert!(
        weights[0] > weights[1],
        "parallel feature vector not selected by attention"
    );
    assert!(output[0] > output[1], "attention weights not respected");
    assert!(
        output[0] > 6.,
        "failed to retrieve proper V with respect to attention weights"
    );
}

#[test]
fn attention_mask() {
    let mut attn = Attention::new();

    let q = Array3::from_shape_vec((1, 1, 2), vec![0., 1.]).unwrap();
    let k = Array3::from_shape_vec((1, 2, 2), vec![0., 1., 1., 0.]).unwrap();
    let v = Array3::from_shape_vec((1, 2, 2), vec![10., 0., 0., 10.]).unwrap();

    let mut mask = Array3::ones((1, 1, 2));
    mask.slice_mut(s![.., .., 0]).assign(&Array2::zeros((1, 1)));
    let out = attn.forward(q, k, v, mask, true);

    let weights = attn
        .weights()
        .to_owned()
        .remove_axis(Axis(0))
        .remove_axis(Axis(0));
    let output = out.remove_axis(Axis(0)).remove_axis(Axis(0));

    println!("weights {:?}", weights);
    println!("output {:?}", output);

    assert!(
        weights[0] < weights[1],
        "mask failed to send all attention weight to orthogonal feature vector"
    );
    assert!(output[0] < output[1], "attention weights not respected");
    assert!(
        output[1] == 10.,
        "failed to retrieve proper V with respect to attention weights"
    );
}

#[test]
fn self_attention_gradient_check() {
    let d_in = 4;
    let d_head = 2;
    let n_head = 2;
    let batch_size = 3;
    let seq_len = 3;
    let eps = 1e-3;

    let mut attn = SelfAttention::new(d_in, d_head, n_head);
    let x = f::xavier_normal((batch_size * seq_len, d_in))
        .into_shape_clone((batch_size, seq_len, d_in))
        .unwrap();
    let mask = Array3::ones((batch_size, seq_len, seq_len));

    // forward + backward
    let out = attn.forward(x.clone(), mask.clone(), true);
    let d_loss = Array3::ones(out.dim());
    let d_x = attn.backward(d_loss.clone());

    // numerical gradient for each element of x
    for b in 0..batch_size {
        for s in 0..seq_len {
            for f in 0..d_in {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[b, s, f]] += eps;
                x_minus[[b, s, f]] -= eps;

                let mut attn_plus = attn.clone();
                let mut attn_minus = attn.clone();
                let out_plus = attn_plus.forward(x_plus, mask.clone(), false);
                let out_minus = attn_minus.forward(x_minus, mask.clone(), false);

                let numerical = (&out_plus - &out_minus).sum() / (2.0 * eps);
                let analytical = d_x[[b, s, f]];

                let diff = (numerical - analytical).abs();
                assert!(
                    diff < 1e-2,
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
fn cross_attention_gradient_check() {
    let d_in = 4;
    let d_head = 2;
    let n_head = 2;
    let batch_size = 3;
    let seq_len_xq = 4;
    let seq_len_xkv = 2;
    let eps = 1e-3;

    let mut attn = CrossAttention::new(d_in, d_head, n_head);

    let x_q = f::xavier_normal((batch_size * seq_len_xq, d_in))
        .into_shape_clone((batch_size, seq_len_xq, d_in))
        .unwrap();
    let x_kv = f::xavier_normal((batch_size * seq_len_xkv, d_in))
        .into_shape_clone((batch_size, seq_len_xkv, d_in))
        .unwrap();

    let mask = Array3::ones((batch_size, seq_len_xq, seq_len_xkv));

    // forward + backward
    let out = attn.forward(x_q.clone(), x_kv.clone(), mask.clone(), true);
    let d_loss = Array3::ones(out.dim());
    let (dx_q, dx_kv) = attn.backward(d_loss.clone());

    // numerical gradient for each element of x
    for b in 0..batch_size {
        for s in 0..seq_len_xq {
            for f in 0..d_in {
                let mut x_plus_q = x_q.clone();
                let mut x_minus_q = x_q.clone();
                x_plus_q[[b, s, f]] += eps;
                x_minus_q[[b, s, f]] -= eps;

                let mut attn_plus = attn.clone();
                let mut attn_minus = attn.clone();
                let out_plus = attn_plus.forward(x_plus_q, x_kv.clone(), mask.clone(), false);
                let out_minus = attn_minus.forward(x_minus_q, x_kv.clone(), mask.clone(), false);

                let numerical = (&out_plus - &out_minus).sum() / (2.0 * eps);
                let analytical = dx_q[[b, s, f]];

                let diff = (numerical - analytical).abs();
                assert!(
                    diff < 1e-3,
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
        for s in 0..seq_len_xkv {
            for f in 0..d_in {
                let mut x_plus_kv = x_kv.clone();
                let mut x_minus_kv = x_kv.clone();
                x_plus_kv[[b, s, f]] += eps;
                x_minus_kv[[b, s, f]] -= eps;

                let mut attn_plus = attn.clone();
                let mut attn_minus = attn.clone();
                let out_plus = attn_plus.forward(x_q.clone(), x_plus_kv, mask.clone(), false);
                let out_minus = attn_minus.forward(x_q.clone(), x_minus_kv, mask.clone(), false);

                let numerical = (&out_plus - &out_minus).sum() / (2.0 * eps);
                let analytical = dx_kv[[b, s, f]];

                let diff = (numerical - analytical).abs();
                assert!(
                    diff < 1e-3,
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

#[test]
fn kv_caching() {
    let batch_size = 5;
    let seq_len = 10;
    let d_model = 5;
    let d_head = 2;
    let n_head = 2;

    let x = Array3::random(
        (batch_size, seq_len, d_model),
        Uniform::new(0., 1.).unwrap(),
    );

    let mask = causal_mask(10)
        .insert_axis(Axis(0))
        .broadcast((5, 10, 10))
        .unwrap()
        .to_owned();

    let mut model = SelfAttention::new(d_model, d_head, n_head);

    let y_immediate = model.forward(x.clone(), mask.clone(), false);

    let mut k = Array3::zeros((batch_size * n_head, 0, d_head));
    let mut v = Array3::zeros((batch_size * n_head, 0, d_head));
    let mut y_stepped = Array3::zeros((batch_size, 0, d_model));

    for t in 0..x.dim().1 {
        let x_t = x.slice(s![.., t, ..]).to_owned().insert_axis(Axis(1));
        let mask = Array3::ones((batch_size, 1, k.dim().1 + 1));

        println!("mask dim {:?}", mask.dim());

        let y_t = model.forward_cached(x_t, mask, &mut k, &mut v);
        y_stepped = concatenate![Axis(1), y_stepped.view(), y_t.view()];
    }

    let soft_eq = (&y_stepped - &y_immediate)
        .mapv(|x| x.abs() < 1e-5)
        .iter()
        .all(|&b| b);

    assert!(
        soft_eq,
        "full forward and kv cached forward diverge in computation\n{y_immediate:?}\n{y_stepped:?}"
    )
}

#[test]
fn kv_caching_2() {
    let batch_size = 5;
    let seq_len = 10;
    let d_model = 4;
    let d_head = 2;
    let n_head = 4;

    let mut model = SelfAttention::new(d_model, d_head, n_head);

    let x = Array3::random(
        (batch_size, seq_len, d_model),
        Uniform::new(0., 1.).unwrap(),
    );
    let mask = causal_mask(10)
        .insert_axis(Axis(0))
        .broadcast((5, 10, 10))
        .unwrap()
        .to_owned();

    let y_immediate = model.forward(x.clone(), mask, false);

    let mut k_cache = Array3::zeros((batch_size * n_head, 0, d_head));
    let mut v_cache = Array3::zeros((batch_size * n_head, 0, d_head));

    let x_start = x.slice(s![.., 0..(seq_len / 2), ..]).to_owned();
    let x_end = x.slice(s![.., (seq_len / 2).., ..]).to_owned();

    let seq_x_start = x_start.dim().1;
    let mask_causal = causal_mask(seq_x_start)
        .insert_axis(Axis(0))
        .broadcast((5, seq_x_start, seq_x_start))
        .unwrap()
        .to_owned();
    let mask_padding = Array3::ones((5, x_start.dim().1, k_cache.dim().1));
    let mask = concatenate![Axis(2), mask_padding.view(), mask_causal.view()];
    let y_stepped_1 = model.forward_cached(x_start, mask, &mut k_cache, &mut v_cache);

    let seq_x_end = x_end.dim().1;
    let mask_causal = causal_mask(seq_x_end)
        .insert_axis(Axis(0))
        .broadcast((5, seq_x_end, seq_x_end))
        .unwrap()
        .to_owned();
    let mask_padding = Array3::ones((5, x_end.dim().1, k_cache.dim().1));
    let mask = concatenate![Axis(2), mask_padding.view(), mask_causal.view()];
    let y_stepped_2 = model.forward_cached(x_end, mask, &mut k_cache, &mut v_cache);

    let y_stepped = concatenate![Axis(1), y_stepped_1.view(), y_stepped_2.view()];

    let soft_eq = (&y_stepped - &y_immediate)
        .mapv(|x| x.abs() < 1e-5)
        .iter()
        .all(|&b| b);

    assert!(
        soft_eq,
        "full forward and kv cached forward diverge in computation\n{y_immediate:?}\n{y_stepped:?}"
    );
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
            .map(|_| Array1::random(d_model, Uniform::new(0., 10.).unwrap()))
            .collect();
        let values: Vec<Array1<f32>> = (0..num_pairs)
            .map(|_| Array1::random(d_model, Uniform::new(0., 10.).unwrap()))
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
fn self_attention_associative_recall() {
    let d_in = 5;
    let d_head = 5;
    let n_head = 4;
    let batch_size = 10;

    let mut attn = SelfAttention::new(d_in, d_head, n_head);
    let mut optim = AdamW::default().with(&mut attn);
    optim.learning_rate = 1e-3;

    let mut final_loss = f32::MAX;

    let (x, y) = make_associative_recall_dataset(batch_size, d_in);
    let mask = Array3::ones((x.dim().0, x.dim().1, x.dim().1));

    for epoch in 0..1000 {
        let y_pred = attn.forward(x.clone(), mask.clone(), true);

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
        "self attention failed associative recall with loss {final_loss} > {max_loss}"
    );
}

fn make_cross_attention_associative_recall_dataset(
    batch_size: usize,
    d_model: usize,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let num_pairs = 3;
    let num_queries = 2;

    let mut x_kv = Array3::zeros((batch_size, num_pairs * 2, d_model));
    let mut x_q = Array3::zeros((batch_size, num_queries, d_model));
    let mut y = Array3::zeros((batch_size, num_queries, d_model));

    for b in 0..batch_size {
        let keys: Vec<Array1<f32>> = (0..num_pairs)
            .map(|_| Array1::random(d_model, Uniform::new(0., 10.).unwrap()))
            .collect();
        let values: Vec<Array1<f32>> = (0..num_pairs)
            .map(|_| Array1::random(d_model, Uniform::new(0., 10.).unwrap()))
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
fn cross_attention_associative_recall() {
    let d_in = 5;
    let d_head = 5;
    let n_head = 4;
    let batch_size = 10;

    let mut attn = CrossAttention::new(d_in, d_head, n_head);
    let mut optim = AdamW::default().with(&mut attn);
    optim.learning_rate = 1e-3;

    let mut final_loss = f32::MAX;
    let (x_q, x_kv, y) = make_cross_attention_associative_recall_dataset(batch_size, d_in);
    let mask = Array3::ones((batch_size, x_q.dim().1, x_kv.dim().1));

    for epoch in 0..1000 {
        let y_pred = attn.forward(x_q.clone(), x_kv.clone(), mask.clone(), true);
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
        "cross attention failed associative recall with loss {final_loss} > {max_loss}"
    );
}

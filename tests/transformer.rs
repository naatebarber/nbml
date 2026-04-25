use nbml::{
    f::positional_encoding_seq,
    nn::Transformer,
    optim::{AdamW, Optimizer, ToIntermediates, ToParams},
};
use ndarray::{Array1, Array2, Array3, Axis, concatenate, s, stack};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::{Rng, rng};

#[test]
fn intermediate_caching() {
    let mut model = Transformer::new_encoder(8, 4, 2);
    let x = Array3::random((2, 3, 8), Uniform::new(0., 1.).unwrap());
    let x2 = Array3::random((2, 3, 8), Uniform::new(0., 1.).unwrap());
    let mask = Array2::ones((2, 3));
    let d = Array3::ones((2, 3, 8));

    model.forward(x.clone(), mask.clone(), true);
    let cache_a = model.stash_intermediates();
    model.backward(d.clone());
    let grads_a = model.attn.grads.clone();
    model.zero_grads();

    model.forward(x2.clone(), mask.clone(), true);
    let cache_b = model.stash_intermediates();
    model.backward(d.clone());
    let grads_b = model.attn.grads.clone();
    model.zero_grads();

    model.apply_intermediates(cache_a.clone());
    model.backward(d.clone());
    let grads_c = model.attn.grads.clone();

    assert!(x != x2);
    assert!(cache_a != cache_b);
    assert!(grads_a.d_w_o != grads_b.d_w_o);
    assert!(
        grads_a.d_w_o == grads_c.d_w_o,
        "intermediate caching process fucks with gradients"
    );
}

#[test]
fn kv_caching() {
    let batch_size = 5;
    let seq_len = 10;
    let d_model = 4;
    let d_head = 2;
    let n_head = 4;

    let mut model = Transformer::new_decoder(d_model, d_head, n_head);

    let x = Array3::random(
        (batch_size, seq_len, d_model),
        Uniform::new(0., 1.).unwrap(),
    );
    let pad_mask = Array2::ones((batch_size, seq_len));

    let y_immediate = model.forward(x.clone(), pad_mask, false);

    let mut k_cache = Array3::zeros((batch_size * n_head, 0, d_head));
    let mut v_cache = Array3::zeros((batch_size * n_head, 0, d_head));
    let mut y_stepped = Array3::zeros((batch_size, 0, d_model));
    for t in 0..x.dim().1 {
        let x_t = x.slice(s![.., t, ..]).to_owned();
        let pad_mask = Array2::ones((batch_size, 1));

        let y_t = model.forward_cached(
            x_t.insert_axis(Axis(1)),
            pad_mask,
            &mut k_cache,
            &mut v_cache,
        );
        y_stepped = concatenate![Axis(1), y_stepped.view(), y_t.view()]
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

    let mut model = Transformer::new_decoder(d_model, d_head, n_head);

    let x = Array3::random(
        (batch_size, seq_len, d_model),
        Uniform::new(0., 1.).unwrap(),
    );
    let pad_mask = Array2::ones((batch_size, seq_len));

    let y_immediate = model.forward(x.clone(), pad_mask, false);

    let mut k_cache = Array3::zeros((batch_size * n_head, 0, d_head));
    let mut v_cache = Array3::zeros((batch_size * n_head, 0, d_head));

    let x_start = x.slice(s![.., 0..(seq_len / 2), ..]).to_owned();
    let x_end = x.slice(s![.., (seq_len / 2).., ..]).to_owned();

    let pad_mask = Array2::ones((x_start.dim().0, x_start.dim().1));
    let y_stepped_1 = model.forward_cached(x_start, pad_mask, &mut k_cache, &mut v_cache);

    let pad_mask = Array2::ones((x_end.dim().0, x_end.dim().1));
    let y_stepped_2 = model.forward_cached(x_end, pad_mask, &mut k_cache, &mut v_cache);

    let y_stepped = concatenate![Axis(1), y_stepped_1.view(), y_stepped_2.view()];

    let soft_eq = (&y_stepped - &y_immediate)
        .mapv(|x| x.abs() < 1e-5)
        .iter()
        .all(|&b| b);

    assert!(
        soft_eq,
        "full forward and kv cached forward diverge in computation\n{y_immediate:?}\n{y_stepped:?}"
    )
}

const EMBED_DIM: usize = 16;
const D_HEAD: usize = 8;
const NUM_HEADS: usize = 2;
const SEQ_LEN: usize = 4;
const BATCH_SIZE: usize = 4;

#[test]
fn identity() {
    println!("=== Testing Transformer End-to-End ===\n");

    // Small transformer for testing
    let mut transformer = Transformer::new_encoder(
        EMBED_DIM, D_HEAD, NUM_HEADS, // num_layers
    );

    let mut optim = AdamW::default().with(&mut transformer);
    optim.learning_rate = 1e-3;

    println!("Test 1: Identity Task (learns to output = input)");
    println!("This tests if the transformer can learn with residual connections\n");

    let x = Array3::random(
        (BATCH_SIZE, SEQ_LEN, EMBED_DIM),
        Uniform::new(-1., 1.).unwrap(),
    );
    let y_target = x.clone();

    let mut losses = Vec::new();

    for epoch in 0..2000 {
        let mask = Array2::ones((x.dim().0, x.dim().1));
        let y_pred = transformer.forward(x.clone(), mask, true);
        let loss = (&y_pred - &y_target).mapv(|v| v.powi(2)).mean().unwrap();
        let d_loss = 2. * (&y_pred - &y_target) / (y_pred.len() as f32);

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
        println!("  ✓ PASS: Can learn identity mapping");
    } else {
        println!("  ✗ FAIL: Struggled with identity mapping");
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

    let x2 = Array3::random(
        (BATCH_SIZE, SEQ_LEN, EMBED_DIM),
        Uniform::new(-1., 1.).unwrap(),
    );
    let mean = x2.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let y_target2 = mean
        .broadcast((BATCH_SIZE, SEQ_LEN, EMBED_DIM))
        .unwrap()
        .to_owned();

    let mut losses2 = Vec::new();

    for epoch in 0..3000 {
        let mask = Array2::ones((x2.dim().0, x2.dim().1));
        let y_pred = transformer2.forward(x2.clone(), mask, true);
        let loss = (&y_pred - &y_target2).mapv(|v| v.powi(2)).mean().unwrap();
        let d_loss = 2. * (&y_pred - &y_target2) / (y_pred.len() as f32);

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
        println!("  ✓ PASS: Attention mechanism works");
    } else if *final_loss2 < 0.2 {
        println!("  ⚠ PARTIAL: Attention works but could be better");
    } else {
        println!("  ✗ FAIL: Attention mechanism may have issues");
        assert!(false);
    }
}

#[test]
fn overfitting() {
    let mut transformer = Transformer::new_encoder(EMBED_DIM, NUM_HEADS, 2);

    let mut optim = AdamW::default().with(&mut transformer);
    optim.learning_rate = 6e-3; // Higher learning rate for faster overfitting

    // Fixed small dataset - should memorize perfectly
    let x = Array3::random(
        (BATCH_SIZE, SEQ_LEN, EMBED_DIM),
        Uniform::new(-1., 1.).unwrap(),
    );
    let y_target = Array3::random(
        (BATCH_SIZE, SEQ_LEN, EMBED_DIM),
        Uniform::new(-1., 1.).unwrap(),
    );

    let mut final_loss = 1.0;

    for epoch in 0..5000 {
        let mask = Array2::ones((x.dim().0, x.dim().1));
        let y_pred = transformer.forward(x.clone(), mask, true);
        let loss = (&y_pred - &y_target).mapv(|v| v.powi(2)).mean().unwrap();
        let d_loss = 2. * (&y_pred - &y_target) / (y_pred.len() as f32);

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
        println!("  ✓ PASS: Can memorize small dataset");
    } else if final_loss < 0.1 {
        println!("  ⚠ PARTIAL: Can learn but not perfectly");
        debug_assert!(false)
    } else {
        println!("  ✗ FAIL: Cannot even overfit small dataset");
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

    let mut model = Transformer::new_decoder(16, 6, 4);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let mut final_loss = 0.;
    for e in 0..6000 {
        // 3 random vectors
        let v1 = Array1::random(16, Uniform::new(0., 1.).unwrap());
        let v2 = Array1::random(16, Uniform::new(0., 1.).unwrap());
        let v3 = Array1::random(16, Uniform::new(0., 1.).unwrap());

        // Marker: which one to retrieve (0, 1, or 2)
        let target_idx = rng().random_range(0..3);
        let mut marker = Array1::zeros(16);
        marker[target_idx] = 1.0;

        let target = match target_idx {
            0 => v1.clone(),
            1 => v2.clone(),
            _ => v3.clone(),
        };

        // Build sequence: [v1, v2, v3, marker, output_position]
        let x = stack![
            Axis(0),
            v1.view(),
            v2.view(),
            v3.view(),
            marker.view(),
            Array1::zeros(16).view()
        ]
        .insert_axis(Axis(0)); // (1, 5, 16)

        let mut y = Array3::zeros((1, 5, 16));
        y.slice_mut(s![0, 4, ..]).assign(&target);

        let mask = Array2::ones((1, 5));
        let y_pred = model.forward(x, mask, true);

        // Only grade last position
        let pred_last = y_pred.slice(s![.., 4..5, ..]);
        let y_last = y.slice(s![.., 4..5, ..]);

        let loss = (&pred_last - &y_last).mapv(|v| v * v).mean().unwrap();

        let mut d_loss = Array3::zeros((1, 5, 16));
        d_loss
            .slice_mut(s![.., 4..5, ..])
            .assign(&(2.0 * (&pred_last - &y_last)));

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if e % 200 == 0 {
            println!("epoch {e} loss {loss:.6}");
        }

        final_loss = 0.9 * final_loss + 0.1 * loss;
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

    let pe = positional_encoding_seq(6, 16).insert_axis(Axis(0));

    let mut final_loss = 0.;
    for e in 0..2000 {
        let vectors: Vec<Array1<f32>> = (0..5)
            .map(|_| Array1::random(16, Uniform::new(0., 1.).unwrap()))
            .collect();

        let mut x = Array3::zeros((1, 6, 16));
        for (i, v) in vectors.iter().enumerate() {
            x.slice_mut(s![0, i, ..]).assign(v);
        }
        // Position 5 is zeros (output position)

        let x_pe = &x + &pe;

        // Target: copy position 2 to position 5
        let mut y = Array3::zeros((1, 6, 16));
        y.slice_mut(s![0, 5, ..]).assign(&vectors[2]);

        let mask = Array2::ones((1, 6));
        let y_pred = model.forward(x_pe.clone(), mask, true);

        let pred_last = y_pred.slice(s![.., 5..6, ..]);
        let y_last = y.slice(s![.., 5..6, ..]);

        let loss = (&pred_last - &y_last).mapv(|v| v * v).mean().unwrap();

        let mut d_loss = Array3::zeros((1, 6, 16));
        d_loss
            .slice_mut(s![.., 5..6, ..])
            .assign(&(2.0 * (&pred_last - &y_last)));

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        if e % 200 == 0 {
            println!("epoch {e} loss {loss:.6}");
        }

        final_loss = 0.9 * final_loss + 0.1 * loss;
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

    let pe = positional_encoding_seq(10, 16).insert_axis(Axis(0));

    let mut final_loss = 0.;
    for e in 0..3000 {
        let vectors: Vec<Array1<f32>> = (0..5)
            .map(|_| Array1::random(16, Uniform::new(0., 1.).unwrap()))
            .collect();

        let mut x = Array3::zeros((1, 10, 16));
        for (i, v) in vectors.iter().enumerate() {
            x.slice_mut(s![0, i, ..]).assign(v);
        }
        // Positions 5-9 are zeros

        let x_pe = &x + &pe;

        // Target: positions 5-9 copy positions 0-4
        let mut y = Array3::zeros((1, 10, 16));
        for (i, v) in vectors.iter().enumerate() {
            y.slice_mut(s![0, i + 5, ..]).assign(v);
        }

        let mask = Array2::ones((1, 10));
        let y_pred = model.forward(x_pe.clone(), mask, true);

        // Grade only output positions
        let pred_out = y_pred.slice(s![.., 5.., ..]);
        let y_out = y.slice(s![.., 5.., ..]);

        let loss = (&pred_out - &y_out).mapv(|v| v * v).mean().unwrap();

        let mut d_loss = Array3::zeros((1, 10, 16));
        d_loss
            .slice_mut(s![.., 5.., ..])
            .assign(&(2.0 * (&pred_out - &y_out)));

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

    let pe = positional_encoding_seq(10, 16).insert_axis(Axis(0));

    // Fixed vectors - same every epoch
    let vectors: Vec<Array1<f32>> = (0..5)
        .map(|_| Array1::random(16, Uniform::new(0., 1.).unwrap()))
        .collect();

    let mut x = Array3::zeros((1, 10, 16));
    for (i, v) in vectors.iter().enumerate() {
        x.slice_mut(s![0, i, ..]).assign(v);
    }
    let x_pe = &x + &pe;

    let mut y = Array3::zeros((1, 10, 16));
    for (i, v) in vectors.iter().enumerate() {
        y.slice_mut(s![0, i + 5, ..]).assign(v);
    }

    let mut final_loss = f32::MAX;

    for e in 0..5000 {
        let mask = Array2::ones((1, 10));
        let y_pred = model.forward(x_pe.clone(), mask, true);

        let pred_out = y_pred.slice(s![.., 5.., ..]);
        let y_out = y.slice(s![.., 5.., ..]);

        let loss = (&pred_out - &y_out).mapv(|v| v * v).mean().unwrap();

        let mut d_loss = Array3::zeros((1, 10, 16));
        d_loss
            .slice_mut(s![.., 5.., ..])
            .assign(&(2.0 * (&pred_out - &y_out)));

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

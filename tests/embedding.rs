use nbml::{
    layers::Embedding,
    optim::{AdamW, Optimizer, ToParams},
};
use ndarray::{Array2, Array3, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn learn_target_embeddings() {
    let vocab_size = 4;
    let d_model = 8;
    let mut emb = Embedding::new(vocab_size, d_model);
    let mut optim = AdamW::default().with(&mut emb);
    optim.learning_rate = 1e-2;

    // Target embeddings to learn
    let targets = Array2::random((vocab_size, d_model), Uniform::new(-1., 1.));

    // Sequence with duplicate token: token 1 appears twice, gets 2x gradient signal
    let tokens = Array2::from_shape_vec((1, 4), vec![0, 1, 2, 1]).unwrap();

    // Build target output from the token sequence
    let mut target_out = Array3::zeros((1, 4, d_model));
    for (t, &tok) in [0usize, 1, 2, 1].iter().enumerate() {
        target_out.slice_mut(s![0, t, ..]).assign(&targets.row(tok));
    }

    let mut loss = f32::MAX;
    for e in 0..200 {
        let out = emb.forward(tokens.clone(), true);
        let diff = &out - &target_out;
        loss = diff.mapv(|v| v * v).mean().unwrap();
        if e % 50 == 0 {
            println!("epoch={e} loss={loss:.6}");
        }

        let d_loss = 2.0 * &diff / (4 * d_model) as f32;
        emb.backward(d_loss.to_owned());
        optim.step(&mut emb);
        emb.zero_grads();
    }

    println!("final loss={loss:.6}");
    assert!(
        loss < 1e-4,
        "embedding failed to learn targets, loss={loss}"
    );
}

#[test]
fn round_trip_lookup() {
    let vocab_size = 8;
    let d_model = 4;
    let mut emb = Embedding::new(vocab_size, d_model);

    // batch=2, seq_len=3
    let tokens = Array2::from_shape_vec((2, 3), vec![0, 3, 7, 1, 5, 2]).unwrap();
    let out = emb.forward(tokens.clone(), false);

    assert_eq!(out.dim(), (2, 3, d_model));

    for b in 0..2 {
        for t in 0..3 {
            let token_id = tokens[[b, t]];
            let expected = emb.weights.row(token_id);
            let got = out.slice(s![b, t, ..]);
            assert_eq!(
                expected, got,
                "mismatch at batch={b} pos={t} token={token_id}"
            );
        }
    }
}

#[test]
fn duplicate_tokens_accumulate_gradients() {
    let vocab_size = 4;
    let d_model = 3;
    let mut emb = Embedding::new(vocab_size, d_model);

    // Token 1 appears three times in a single sequence
    let tokens = Array2::from_shape_vec((1, 4), vec![1, 2, 1, 1]).unwrap();
    let out = emb.forward(tokens, true);

    // Backward with all-ones gradient
    let d_loss = Array3::ones(out.dim());
    emb.backward(d_loss);

    // Token 1 appeared 3 times, so its gradient row should be 3x the per-position gradient (1.0)
    let grad_token_1 = emb.grads.d_weights.row(1);
    for &v in grad_token_1.iter() {
        assert!(
            (v - 3.0).abs() < 1e-12,
            "expected gradient 3.0 for token 1, got {v}"
        );
    }

    // Token 2 appeared once
    let grad_token_2 = emb.grads.d_weights.row(2);
    for &v in grad_token_2.iter() {
        assert!(
            (v - 1.0).abs() < 1e-12,
            "expected gradient 1.0 for token 2, got {v}"
        );
    }

    // Token 0 and 3 never appeared — gradient should be zero
    for id in [0, 3] {
        let grad = emb.grads.d_weights.row(id);
        for &v in grad.iter() {
            assert!(
                v.abs() < 1e-12,
                "expected gradient 0.0 for unused token {id}, got {v}"
            );
        }
    }
}

use nbml::layers::{LastTokenPooling, SequencePooling};
use ndarray::{Array2, Array3, array};

// ── MeanPooling ──

#[test]
fn mean_pooling_forward_uniform_mask() {
    let mut pool = SequencePooling::new();

    // (batch=2, seq=3, features=2)
    let x = Array3::from_shape_vec(
        (2, 3, 2),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, // batch 1
        ],
    )
    .unwrap();

    // all tokens valid
    let mask = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];

    let out = pool.forward(x, mask, false);

    // batch 0: mean over seq => (3,4)
    // batch 1: mean over seq => (30,40)
    assert_eq!(out.dim(), (2, 2));
    assert!((out[[0, 0]] - 3.0).abs() < 1e-6);
    assert!((out[[0, 1]] - 4.0).abs() < 1e-6);
    assert!((out[[1, 0]] - 30.0).abs() < 1e-6);
    assert!((out[[1, 1]] - 40.0).abs() < 1e-6);
}

#[test]
fn mean_pooling_forward_partial_mask() {
    let mut pool = SequencePooling::new();

    let x = Array3::from_shape_vec(
        (1, 4, 2),
        vec![1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 100.0, 200.0],
    )
    .unwrap();

    // only first 2 tokens valid
    let mask = array![[1.0, 1.0, 0.0, 0.0]];

    let out = pool.forward(x, mask, false);

    // mean of first 2 tokens: (2, 3)
    assert!((out[[0, 0]] - 2.0).abs() < 1e-6);
    assert!((out[[0, 1]] - 3.0).abs() < 1e-6);
}

#[test]
fn mean_pooling_backward_uniform_mask() {
    let mut pool = SequencePooling::new();

    let x = Array3::from_shape_vec((1, 3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mask = array![[1.0, 1.0, 1.0]];

    let _out = pool.forward(x, mask, true);

    let d_loss = array![[3.0, 6.0]];
    let grad = pool.backward(d_loss);

    // d_loss / seq_len broadcast across all seq positions
    // each position gets 3/3=1.0 and 6/3=2.0
    assert_eq!(grad.dim(), (1, 3, 2));
    for t in 0..3 {
        assert!((grad[[0, t, 0]] - 1.0).abs() < 1e-6);
        assert!((grad[[0, t, 1]] - 2.0).abs() < 1e-6);
    }
}

#[test]
fn mean_pooling_backward_partial_mask() {
    let mut pool = SequencePooling::new();

    let x = Array3::from_shape_vec((1, 3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mask = array![[1.0, 1.0, 0.0]];

    let _out = pool.forward(x, mask, true);

    let d_loss = array![[4.0, 6.0]];
    let grad = pool.backward(d_loss);

    // d_loss / 2 for valid positions, 0 for masked
    assert!((grad[[0, 0, 0]] - 2.0).abs() < 1e-6);
    assert!((grad[[0, 0, 1]] - 3.0).abs() < 1e-6);
    assert!((grad[[0, 1, 0]] - 2.0).abs() < 1e-6);
    assert!((grad[[0, 1, 1]] - 3.0).abs() < 1e-6);
    assert!((grad[[0, 2, 0]]).abs() < 1e-6);
    assert!((grad[[0, 2, 1]]).abs() < 1e-6);
}

#[test]
fn mean_pooling_numerical_gradient_check() {
    use ndarray_rand::{RandomExt, rand_distr::Uniform};

    let eps = 1e-3;
    let (batch, seq, features) = (2, 4, 3);

    let x = Array3::random((batch, seq, features), Uniform::new(-2.0, 2.0).unwrap());
    let mask = array![[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]];
    let g = Array2::random((batch, features), Uniform::new(-1.0, 1.0).unwrap());

    let mut pool = SequencePooling::new();
    let _out = pool.forward(x.clone(), mask.clone(), true);
    let analytic = pool.backward(g.clone());

    for b in 0..batch {
        for t in 0..seq {
            for f in 0..features {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[b, t, f]] += eps;
                x_minus[[b, t, f]] -= eps;

                let mut p = SequencePooling::new();
                let out_plus = p.forward(x_plus, mask.clone(), false);
                let mut p = SequencePooling::new();
                let out_minus = p.forward(x_minus, mask.clone(), false);

                let numerical = ((&out_plus - &out_minus) * &g).sum() / (2.0 * eps);
                let analytical = analytic[[b, t, f]];

                let diff = (numerical - analytical).abs();
                assert!(
                    diff < 1e-4,
                    "mean pooling grad mismatch at [{},{},{}]: num={}, ana={}, diff={}",
                    b,
                    t,
                    f,
                    numerical,
                    analytical,
                    diff
                );
            }
        }
    }
}

// ── LastTokenPooling ──

#[test]
fn last_token_pooling_forward_full_mask() {
    let mut pool = LastTokenPooling::new();

    // (batch=2, seq=3, features=2)
    let x = Array3::from_shape_vec(
        (2, 3, 2),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, // batch 1
        ],
    )
    .unwrap();

    let mask = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];

    let out = pool.forward(x, mask, false);

    // last token (index 2) for each batch
    assert!((out[[0, 0]] - 5.0).abs() < 1e-6);
    assert!((out[[0, 1]] - 6.0).abs() < 1e-6);
    assert!((out[[1, 0]] - 50.0).abs() < 1e-6);
    assert!((out[[1, 1]] - 60.0).abs() < 1e-6);
}

#[test]
fn last_token_pooling_forward_partial_mask() {
    let mut pool = LastTokenPooling::new();

    let x = Array3::from_shape_vec((1, 4, 2), vec![1.0, 2.0, 3.0, 4.0, 99.0, 99.0, 99.0, 99.0])
        .unwrap();

    // only first 2 tokens valid => last valid is index 1
    let mask = array![[1.0, 1.0, 0.0, 0.0]];

    let out = pool.forward(x, mask, false);

    assert!((out[[0, 0]] - 3.0).abs() < 1e-6);
    assert!((out[[0, 1]] - 4.0).abs() < 1e-6);
}

#[test]
fn last_token_pooling_backward() {
    let mut pool = LastTokenPooling::new();

    let x = Array3::from_shape_vec((1, 3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mask = array![[1.0, 1.0, 1.0]];

    let _out = pool.forward(x, mask, true);

    let d_loss = array![[7.0, 8.0]];
    let grad = pool.backward(d_loss);

    // gradient should only flow to the last valid token (index 2)
    assert_eq!(grad.dim(), (1, 3, 2));
    assert!((grad[[0, 0, 0]]).abs() < 1e-6);
    assert!((grad[[0, 0, 1]]).abs() < 1e-6);
    assert!((grad[[0, 1, 0]]).abs() < 1e-6);
    assert!((grad[[0, 1, 1]]).abs() < 1e-6);
    assert!((grad[[0, 2, 0]] - 7.0).abs() < 1e-6);
    assert!((grad[[0, 2, 1]] - 8.0).abs() < 1e-6);
}

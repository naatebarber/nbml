use nbml::{f, nn::PositionwiseFFN, optim::{AdamW, Optimizer, ToParams}};
use ndarray::Array2;

#[test]
fn test_positionwise_ffn_gradients() {
    let d_in = 4;
    let batch_size = 3;
    let eps = 1e-3;

    let mut ln = PositionwiseFFN::new(d_in, d_in, d_in);
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
fn positionwise_ffn_learns_xor() {
    let mut model = PositionwiseFFN::new(2, 16, 1);

    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();
    let y = Array2::from_shape_vec((4, 1), vec![0., 1., 1., 0.]).unwrap();

    for _ in 0..1000 {
        let y_pred = model.forward(x.clone(), true);
        let y_pred_sigmoid = f::sigmoid(&y_pred);

        let d_loss = 2. * (&y_pred_sigmoid - &y);
        let d_loss_d_sigmoid = &d_loss * f::d_sigmoid(&y_pred);

        model.backward(d_loss_d_sigmoid);
        optim.step(&mut model);
        model.zero_grads();
    }

    let y_pred = f::sigmoid(&model.forward(x, false));

    for i in 0..4 {
        let pred = y_pred[[i, 0]];

        let target = y[[i, 0]];
        let rounded = if pred > 0.5 { 1.0 } else { 0.0 };
        assert!(
            rounded == target,
            "XOR failed at sample {i}: pred={pred:.4} expected={target}"
        );
    }
}



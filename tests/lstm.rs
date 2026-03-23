use nbml::{
    f, nn::LSTM, optim::{AdamW, Optimizer, ToIntermediates, ToParams}
};
use ndarray::{Array2, Array3, Axis, concatenate, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn intermediate_caching() {
    let mut model = LSTM::new(4);
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
    assert!(grads_a.d_wi != grads_b.d_wi);
    assert!(
        grads_a.d_wi == grads_c.d_wi,
        "intermediate caching process fucks with gradients"
    );
}


#[test]
fn delta_net_gradient_check() {
    let d_in = 4;
    let batch_size = 3;
    let seq_len = 3;
    let eps = 1e-3;

    let mut lstm = LSTM::new(d_in);
    let x = f::xavier_normal((batch_size * seq_len, d_in))
        .into_shape_clone((batch_size, seq_len, d_in))
        .unwrap();

    let out = lstm.forward(x.clone(), true);
    let d_loss = Array3::ones(out.dim());
    let d_x = lstm.backward(d_loss.clone());

    for b in 0..batch_size {
        for s in 0..seq_len {
            for f in 0..d_in {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[b, s, f]] += eps;
                x_minus[[b, s, f]] -= eps;

                let mut lstm_plus = lstm.clone();
                let mut lstm_minus = lstm.clone();
                let out_plus = lstm_plus.forward(x_plus, false);
                let out_minus = lstm_minus.forward(x_minus, false);

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
fn lstm_forward_and_step_compute_same_value() {
    let mut lstm = LSTM::new(12);
    let x = Array3::random((1, 12, 12), Uniform::new(0., 1.));

    let pred_forward = lstm.forward(x.clone(), true);

    let mut h = Array2::zeros((1, 12));
    let mut cell = Array2::zeros((1, 12));

    for i in 0..12 {
        let x_t = x.slice(s![.., i, ..]).to_owned();
        lstm.step(&x_t, &mut h, &mut cell);
    }

    assert!(
        pred_forward.slice(s![.., -1, ..]).to_owned() == h,
        "forward scan evolution and step evolution produce different results"
    );
}

#[test]
fn lstm_sequence_pred() {
    let batch_size = 20;
    let seq_len = 10;
    let features = 10;

    let mut model = LSTM::new(features);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let seed = Array3::random((batch_size, 2, features), Uniform::new(-1., 1.));

    let batch = Array3::zeros((batch_size, seq_len - 2, features));
    let mut batch = concatenate![Axis(1), seed.view(), batch.view()];

    for t in 2..batch.dim().1 {
        let a = 0.52;
        let b = 0.48;

        let next = a * &batch.slice(s![.., t - 1, ..]) + b * &batch.slice(s![.., t - 2, ..]);
        batch.slice_mut(s![.., t, ..]).assign(&next);
    }

    let x = batch.slice(s![.., 1..(seq_len - 1), ..]).to_owned();
    let y = batch.slice(s![.., 2.., ..]).to_owned();

    println!("x {:?} y {:?}", x.dim(), y.dim());

    for e in 0..1000 {
        let y_pred = model.forward(x.clone(), true);
        let d_loss = 2. * (&y_pred - &y);
        let loss = (&y_pred - &y).powi(2).mean().unwrap();
        println!("{e} loss={loss}");

        model.backward(d_loss);
        optim.step(&mut model);

        model.zero_grads();
    }

    let y_pred = model.forward(x.clone(), true);
    let loss = (&y_pred - &y).powi(2).mean().unwrap();

    let max_viable_loss = 0.01;
    assert!(
        loss < max_viable_loss,
        "LSTM doesnt effectively train with test loss {loss} > {max_viable_loss}"
    );
}

#[test]
fn lstm_sequence_pred_step_forward() {
    let batch_size = 10;
    let seq_len = 10;
    let features = 10;

    let mut model = LSTM::new(features);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let seed = Array3::random((batch_size, 2, features), Uniform::new(-1., 1.));

    let batch = Array3::zeros((batch_size, seq_len - 2, features));
    let mut batch = concatenate![Axis(1), seed.view(), batch.view()];

    for t in 2..batch.dim().1 {
        let a = 0.52;
        let b = 0.48;

        let next = a * &batch.slice(s![.., t - 1, ..]) + b * &batch.slice(s![.., t - 2, ..]);
        batch.slice_mut(s![.., t, ..]).assign(&next);
    }

    let x = batch.slice(s![.., 1..(seq_len - 1), ..]).to_owned();
    let y = batch.slice(s![.., 2.., ..]).to_owned();

    println!("x {:?} y {:?}", x.dim(), y.dim());

    for e in 0..1000 {
        let (batch_size, seq_len, features) = x.dim();

        let mut h = Array2::zeros((batch_size, features));
        let mut cell = Array2::zeros((batch_size, features));
        let mut y_pred = Array3::zeros((batch_size, seq_len, features));

        for i in 0..seq_len {
            let x_t = x.slice(s![.., i, ..]).to_owned();
            model.step_forward(&x_t, &mut h, &mut cell);

            y_pred.slice_mut(s![.., i, ..]).assign(&h);
        }

        let d_loss = 2. * (&y_pred - &y);
        let loss = (&y_pred - &y).powi(2).mean().unwrap();
        println!("{e} loss={loss}");

        model.backward(d_loss);
        model.forward(Array3::zeros((0, 0, features)), false);
        optim.step(&mut model);

        model.zero_grads();
        model.clear_intermediates();
    }

    let y_pred = model.forward(x.clone(), true);
    let loss = (&y_pred - &y).powi(2).mean().unwrap();

    let max_viable_loss = 0.01;
    assert!(
        loss < max_viable_loss,
        "LSTM doesnt effectively train with step forward, test loss {loss} > {max_viable_loss}"
    );
}

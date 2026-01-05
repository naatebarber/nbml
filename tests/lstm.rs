use nbml::{
    nn::LSTM,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array2, Array3, Axis, concatenate, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

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
fn lstm_forward_and_scan_compute_same_value() {
    let mut lstm = LSTM::new(12);
    let x = Array3::random((1, 12, 12), Uniform::new(0., 1.));

    let pred_forward = lstm.forward(x.clone(), false);
    let (h, _cell) = lstm.scan(&x);

    assert!(
        pred_forward == h,
        "forward scan evolution and scan evolution produce different results"
    );
}

#[test]
fn lstm_sequence_pred() {
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

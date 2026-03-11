use nbml::{
    Tensor,
    nn2::LSTM,
    optim2::{Optimizer, ToParams, adam::AdamW},
    s,
    tensor::{Tensor2, Tensor3},
};

#[test]
fn lstm_forward_and_step_compute_same_value() {
    let mut lstm = LSTM::new(12);
    let x = Tensor::random_uniform((1, 12, 12));

    let pred_forward = lstm.forward(x.clone(), true);

    let mut h = Tensor2::zeros((1, 12));
    let mut cell = Tensor2::zeros((1, 12));

    for i in 0..12 {
        let x_t = x.slice(s![.., i, ..]);
        lstm.step(&x_t, &mut h, &mut cell);
    }

    let last = pred_forward.slice(s![.., -1, ..]);
    let diff = (&last - &h).powi(2).mean();
    assert!(
        diff < 1e-10,
        "forward scan evolution and step evolution produce different results (diff={diff})"
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

    let seed = Tensor::random_uniform((batch_size, 2, features)) * 2.0 - 1.0;

    let zeros = Tensor3::zeros((batch_size, seq_len - 2, features));
    let mut batch = Tensor::concatenate(1, &[&seed, &zeros]);

    for t in 2..seq_len {
        let a = 0.52;
        let b = 0.48;

        let prev1 = batch.slice(s![.., (t - 1), ..]);
        let prev2 = batch.slice(s![.., (t - 2), ..]);
        let next = prev1 * a + prev2 * b;
        batch.slice_assign(s![.., t, ..], &next);
    }

    let x = batch.slice(s![.., 1..(seq_len - 1), ..]);
    let y = batch.slice(s![.., 2.., ..]);

    println!("x {:?} y {:?}", x.shape(), y.shape());

    for e in 0..1000 {
        let y_pred = model.forward(x.clone(), true);
        let d_loss = (&y_pred - &y) * 2.0;
        let loss = (&y_pred - &y).powi(2).mean();
        println!("{e} loss={loss}");

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }

    let y_pred = model.forward(x.clone(), true);
    let loss = (&y_pred - &y).powi(2).mean();

    let max_viable_loss = 0.01;
    assert!(
        loss < max_viable_loss,
        "LSTM doesnt effectively train with test loss {loss} > {max_viable_loss}"
    );
}

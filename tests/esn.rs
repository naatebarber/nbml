use nbml::{
    nn::ESN,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array2, Array3, Axis, concatenate, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn esn_forward_and_step_compute_same_value() {
    let mut esn = ESN::new(12, 24, 8);
    let x = Array3::random((1, 12, 12), Uniform::new(0., 1.));

    let pred_forward = esn.forward(x.clone(), false);

    let mut h = Array2::zeros((1, 24));
    let mut output = Array2::zeros((1, 8));

    for i in 0..x.dim().1 {
        let x_t = x.slice(s![.., i, ..]).to_owned();
        output = esn.step(&x_t, &mut h, false);
    }

    let forward_result = pred_forward.slice(s![.., -1, ..]);

    assert!(
        forward_result == output,
        "forward scan evolution and step evolution produce different results forward={:?} step={:?}",
        forward_result,
        output
    );
}

#[test]
fn esn_sequence_pred() {
    let batch_size = 10;
    let seq_len = 10;
    let features = 10;

    let mut model = ESN::new(features, 2 * features, features);
    model.set_spectral_radius(0.95, 2000);

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
        "ESN doesnt effectively train with test loss {loss} > {max_viable_loss}"
    );
}

#[test]
fn esn_sequence_pred_step() {
    let batch_size = 10;
    let seq_len = 10;
    let features = 10;
    let hidden = 20;

    let mut model = ESN::new(features, hidden, features);
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

        let mut h = Array2::zeros((batch_size, hidden));
        let mut y_pred = Array3::zeros((batch_size, seq_len, features));

        for i in 0..seq_len {
            let x_t = x.slice(s![.., i, ..]).to_owned();
            let y_pred_t = model.step(&x_t, &mut h, true);
            y_pred.slice_mut(s![.., i, ..]).assign(&y_pred_t);

            let y_t = y.slice(s![.., i, ..]);
            let d_loss = 2. * (&y_pred_t - &y_t);
            model.backward(d_loss.insert_axis(Axis(0)));
        }

        let loss = (&y_pred - &y).powi(2).mean().unwrap();
        println!("{e} loss={loss}");

        optim.step(&mut model);

        model.zero_grads();
    }

    let y_pred = model.forward(x.clone(), true);
    let loss = (&y_pred - &y).powi(2).mean().unwrap();

    let max_viable_loss = 0.01;
    assert!(
        loss < max_viable_loss,
        "LSTM doesnt effectively train with step forward, test loss {loss} > {max_viable_loss}"
    );
}

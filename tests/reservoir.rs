use nbml::{
    f::Activation,
    nn::{FFN, RNNReservoir, SNNReservoir},
    optim::{AdamW, Optimizer, ToParams},
};
use ndarray::{Array1, Array2, Array3, Axis, concatenate, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

// --- RNNReservoir (ESN-style) tests ---

#[test]
fn rnn_reservoir_forward_and_step_same_value() {
    let d_in = 12;
    let d_hidden = 24;
    let d_out = 8;

    let reservoir = RNNReservoir::new(d_in, d_hidden);
    let mut readout = FFN::new(vec![(d_hidden, d_out, Activation::Identity)]);

    let x = Array3::random((1, 12, d_in), Uniform::new(0., 1.));

    // forward path
    let encoded = reservoir.forward(x.clone());
    let encoded_2d = encoded.into_shape_clone((1 * 12, d_hidden)).unwrap();
    let pred_forward = readout
        .forward(encoded_2d, false)
        .into_shape_clone((1, 12, d_out))
        .unwrap();

    // step path
    let mut h = Array2::zeros((1, d_hidden));
    let mut output = Array2::zeros((1, d_out));

    for i in 0..x.dim().1 {
        let x_t = x.slice(s![.., i, ..]).to_owned();
        reservoir.step(&x_t, &mut h);
        output = readout.forward(h.clone(), false);
    }

    let forward_result = pred_forward.slice(s![.., -1, ..]);

    assert!(
        forward_result == output,
        "forward scan and step produce different results forward={:?} step={:?}",
        forward_result,
        output
    );
}

#[test]
fn rnn_reservoir_sequence_pred() {
    let batch_size = 10;
    let seq_len = 10;
    let features = 10;
    let hidden = 2 * features;

    let reservoir = RNNReservoir::new(features, hidden);
    let mut readout = FFN::new(vec![
        (hidden, hidden, Activation::Relu),
        (hidden, features, Activation::Identity),
    ]);

    let mut optim = AdamW::default().with(&mut readout);
    optim.learning_rate = 1e-2;

    let seed = Array3::random((batch_size, 2, features), Uniform::new(-1., 1.));
    let batch = Array3::zeros((batch_size, seq_len - 2, features));
    let mut batch = concatenate![Axis(1), seed.view(), batch.view()];

    for t in 2..batch.dim().1 {
        let next = 0.52 * &batch.slice(s![.., t - 1, ..]) + 0.48 * &batch.slice(s![.., t - 2, ..]);
        batch.slice_mut(s![.., t, ..]).assign(&next);
    }

    let x = batch.slice(s![.., 1..(seq_len - 1), ..]).to_owned();
    let y = batch.slice(s![.., 2.., ..]).to_owned();

    for e in 0..1000 {
        let encoded = reservoir.forward(x.clone());
        let encoded_2d = encoded
            .into_shape_clone((batch_size * (seq_len - 2), hidden))
            .unwrap();
        let y_pred_2d = readout.forward(encoded_2d, true);
        let y_pred = y_pred_2d
            .into_shape_clone((batch_size, seq_len - 2, features))
            .unwrap();

        let d_loss = 2. * (&y_pred - &y);
        let loss = (&y_pred - &y).powi(2).mean().unwrap();
        if e % 200 == 0 {
            println!("{e} loss={loss}");
        }

        let d_loss_2d = d_loss
            .into_shape_clone((batch_size * (seq_len - 2), features))
            .unwrap();
        readout.backward(d_loss_2d);
        optim.step(&mut readout);
        readout.zero_grads();
    }

    let encoded = reservoir.forward(x.clone());
    let encoded_2d = encoded
        .into_shape_clone((batch_size * (seq_len - 2), hidden))
        .unwrap();
    let y_pred_2d = readout.forward(encoded_2d, false);
    let y_pred = y_pred_2d
        .into_shape_clone((batch_size, seq_len - 2, features))
        .unwrap();
    let loss = (&y_pred - &y).powi(2).mean().unwrap();

    let max_viable_loss = 0.05;
    assert!(
        loss < max_viable_loss,
        "RNNReservoir sequence pred loss {loss} > {max_viable_loss}"
    );
}

#[test]
fn rnn_reservoir_sequence_pred_step() {
    let batch_size = 10;
    let seq_len = 10;
    let features = 10;
    let hidden = 20;

    let reservoir = RNNReservoir::new(features, hidden);
    let mut readout = FFN::new(vec![
        (hidden, hidden, Activation::Relu),
        (hidden, features, Activation::Identity),
    ]);

    let mut optim = AdamW::default().with(&mut readout);
    optim.learning_rate = 1e-2;

    let seed = Array3::random((batch_size, 2, features), Uniform::new(-1., 1.));
    let batch = Array3::zeros((batch_size, seq_len - 2, features));
    let mut batch = concatenate![Axis(1), seed.view(), batch.view()];

    for t in 2..batch.dim().1 {
        let next = 0.52 * &batch.slice(s![.., t - 1, ..]) + 0.48 * &batch.slice(s![.., t - 2, ..]);
        batch.slice_mut(s![.., t, ..]).assign(&next);
    }

    let x = batch.slice(s![.., 1..(seq_len - 1), ..]).to_owned();
    let y = batch.slice(s![.., 2.., ..]).to_owned();

    for e in 0..1000 {
        let (batch_size, seq_len, features) = x.dim();

        let mut h = Array2::zeros((batch_size, hidden));
        let mut y_pred = Array3::zeros((batch_size, seq_len, features));

        for i in 0..seq_len {
            let x_t = x.slice(s![.., i, ..]).to_owned();
            reservoir.step(&x_t, &mut h);
            let y_pred_t = readout.forward(h.clone(), true);
            y_pred.slice_mut(s![.., i, ..]).assign(&y_pred_t);

            let y_t = y.slice(s![.., i, ..]);
            let d_loss = 2. * (&y_pred_t - &y_t);
            readout.backward(d_loss);
        }

        let loss = (&y_pred - &y).powi(2).mean().unwrap();
        if e % 200 == 0 {
            println!("{e} loss={loss}");
        }

        optim.step(&mut readout);
        readout.zero_grads();
    }

    // evaluate with forward scan
    let encoded = reservoir.forward(x.clone());
    let (batch_size, seq_len, _) = x.dim();
    let encoded_2d = encoded
        .into_shape_clone((batch_size * seq_len, hidden))
        .unwrap();
    let y_pred_2d = readout.forward(encoded_2d, false);
    let y_pred = y_pred_2d
        .into_shape_clone((batch_size, seq_len, features))
        .unwrap();
    let loss = (&y_pred - &y).powi(2).mean().unwrap();

    let max_viable_loss = 0.05;
    assert!(
        loss < max_viable_loss,
        "RNNReservoir step pred loss {loss} > {max_viable_loss}"
    );
}

// --- SNNReservoir (LSM-style) tests ---

#[test]
fn snn_reservoir_temporal_classification() {
    let samples_per_class = 32;
    let batch_size = samples_per_class * 2;
    let seq_len = 20;
    let features = 1;
    let hidden = 40;
    let delta = 0.1;

    let mut reservoir = SNNReservoir::new(features, hidden);
    reservoir.set_spectral_radius(0.9, 2000);
    reservoir.set_tau_range(0.05, 0.5);
    reservoir.set_threshold_range(0.3, 0.8);

    let mut readout = FFN::new(vec![(hidden, 1, Activation::Sigmoid)]);

    let mut optim = AdamW::default().with(&mut readout);
    optim.learning_rate = 5e-3;

    let mut x = Array3::zeros((batch_size, seq_len, features));
    let mut y = Array2::zeros((batch_size, 1));

    for b in 0..batch_size {
        let noise = Array1::random(seq_len, Uniform::new(-0.05, 0.05));
        for t in 0..seq_len {
            let frac = t as f32 / seq_len as f32;
            let val = if b < samples_per_class {
                frac + noise[t]
            } else {
                (-3.0 * frac).exp() + noise[t]
            };
            x[[b, t, 0]] = val;
        }
        if b >= samples_per_class {
            y[[b, 0]] = 1.0;
        }
    }

    for e in 0..300 {
        let mut state = Array2::zeros((batch_size, hidden));
        let mut output = Array2::zeros((batch_size, 1));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]).to_owned();
            reservoir.step(&x_t, &mut state, delta);
            output = readout.forward(state.clone(), true);
        }

        let d_loss = &output - &y;
        readout.backward(d_loss.to_owned());

        if e % 50 == 0 {
            let bce = -(&y * &output.mapv(|v| (v + 1e-8).ln())
                + &(1.0 - &y) * &output.mapv(|v| (1.0 - v + 1e-8).ln()))
                .mean()
                .unwrap();
            println!("{e} bce_loss={bce:.4}");
        }

        optim.step(&mut readout);
        readout.zero_grads();
    }

    let mut state = Array2::zeros((batch_size, hidden));
    let mut output = Array2::zeros((batch_size, 1));

    for t in 0..seq_len {
        let x_t = x.slice(s![.., t, ..]).to_owned();
        reservoir.step(&x_t, &mut state, delta);
        output = readout.forward(state.clone(), false);
    }

    let mut correct = 0;
    for b in 0..batch_size {
        let pred = if output[[b, 0]] > 0.5 { 1 } else { 0 };
        let label = if y[[b, 0]] > 0.5 { 1 } else { 0 };
        if pred == label {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / batch_size as f32;
    println!("classification accuracy: {accuracy:.2}");

    assert!(
        accuracy > 0.9,
        "SNNReservoir temporal classification accuracy {accuracy:.2} <= 0.9"
    );
}

#[test]
fn snn_reservoir_heterogeneous_delta() {
    let samples_per_class = 32;
    let batch_size = samples_per_class * 2;
    let seq_len = 20;
    let features = 1;
    let hidden = 40;

    let mut reservoir = SNNReservoir::new(features, hidden);
    reservoir.set_spectral_radius(0.9, 2000);
    reservoir.set_tau_range(0.05, 0.5);
    reservoir.set_threshold_range(0.3, 0.8);

    let mut readout = FFN::new(vec![(hidden, 1, Activation::Sigmoid)]);

    let mut optim = AdamW::default().with(&mut readout);
    optim.learning_rate = 5e-3;

    let deltas = Array1::random(seq_len, Uniform::new(0.02, 0.3));

    let mut cum_time = vec![0.0f32; seq_len + 1];
    for t in 0..seq_len {
        cum_time[t + 1] = cum_time[t] + deltas[t];
    }
    let total_time = cum_time[seq_len];

    let mut x = Array3::zeros((batch_size, seq_len, features));
    let mut y = Array2::zeros((batch_size, 1));

    for b in 0..batch_size {
        let noise = Array1::random(seq_len, Uniform::new(-0.05, 0.05));
        for t in 0..seq_len {
            let frac = cum_time[t + 1] / total_time;
            let val = if b < samples_per_class {
                (2.0 * std::f32::consts::PI * frac).sin() + noise[t]
            } else {
                (1.0 - frac) + noise[t]
            };
            x[[b, t, 0]] = val;
        }
        if b >= samples_per_class {
            y[[b, 0]] = 1.0;
        }
    }

    for e in 0..300 {
        let mut state = Array2::zeros((batch_size, hidden));
        let mut output = Array2::zeros((batch_size, 1));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]).to_owned();
            reservoir.step(&x_t, &mut state, deltas[t]);
            output = readout.forward(state.clone(), true);
        }

        let d_loss = &output - &y;
        readout.backward(d_loss.to_owned());

        if e % 50 == 0 {
            let bce = -(&y * &output.mapv(|v| (v + 1e-8).ln())
                + &(1.0 - &y) * &output.mapv(|v| (1.0 - v + 1e-8).ln()))
                .mean()
                .unwrap();
            println!("{e} bce_loss={bce:.4}");
        }

        optim.step(&mut readout);
        readout.zero_grads();
    }

    let mut state = Array2::zeros((batch_size, hidden));
    let mut output = Array2::zeros((batch_size, 1));

    for t in 0..seq_len {
        let x_t = x.slice(s![.., t, ..]).to_owned();
        reservoir.step(&x_t, &mut state, deltas[t]);
        output = readout.forward(state.clone(), false);
    }

    let mut correct = 0;
    for b in 0..batch_size {
        let pred = if output[[b, 0]] > 0.5 { 1 } else { 0 };
        let label = if y[[b, 0]] > 0.5 { 1 } else { 0 };
        if pred == label {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / batch_size as f32;
    println!("heterogeneous delta accuracy: {accuracy:.2}");

    assert!(
        accuracy > 0.9,
        "SNNReservoir heterogeneous delta accuracy {accuracy:.2} <= 0.9"
    );
}

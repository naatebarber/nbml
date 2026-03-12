use nbml::{
    Tensor,
    f2::xavier_normal,
    layers::Linear,
    nn2::reservoirs::{RNNReservoir, SNNReservoir},
    optim2::{Optimizer, ToParams, adam::AdamW},
    s,
    tensor::{Float, Tensor2, Tensor3},
};

// ── ESN-style tests (RNNReservoir + Linear) ─────────────────────────

#[test]
fn esn_forward_and_step_compute_same_value() {
    let d_in = 12;
    let d_hidden = 24;
    let d_out = 8;

    let mut reservoir = RNNReservoir::new(d_in, d_hidden);
    reservoir.set_spectral_radius(0.95, 1000);

    let mut readout = Linear::new(d_hidden, d_out, xavier_normal);

    let x = Tensor::random_uniform((1, 12, d_in));

    // full forward
    let encoded = reservoir.forward(x.clone());
    let (batch_size, seq_len, _) = encoded.dim3();
    let encoded_2d = encoded.reshape((batch_size * seq_len, d_hidden));
    let pred_forward = readout
        .forward(&encoded_2d, false)
        .reshape((batch_size, seq_len, d_out));

    // step-by-step
    let mut h = Tensor2::zeros((1, d_hidden));
    let mut output = Tensor2::zeros((1, d_out));

    for i in 0..seq_len {
        let x_t = x.slice(s![.., i, ..]);
        reservoir.step(&x_t, &mut h);
        output = readout.forward(&h, false);
    }

    let forward_last = pred_forward.slice(s![.., -1, ..]);

    let diff = (&forward_last - &output).powi(2).sum();
    assert!(
        diff < 1e-10,
        "forward scan and step produce different results, diff={diff}"
    );
}

fn generate_linear_recurrence(batch_size: usize, seq_len: usize, features: usize) -> Tensor3 {
    let seed = Tensor::random_uniform((batch_size, 2, features)) * 2.0 - 1.0;
    let zeros = Tensor3::zeros((batch_size, seq_len - 2, features));
    let mut batch = Tensor::concatenate(1, &[&seed, &zeros]);

    for t in 2..seq_len {
        let prev1 = batch.slice(s![.., (t - 1), ..]);
        let prev2 = batch.slice(s![.., (t - 2), ..]);
        let next = prev1 * 0.52 + prev2 * 0.48;
        batch.slice_assign(s![.., t, ..], &next);
    }
    batch
}

#[test]
fn esn_sequence_pred() {
    let batch_size = 10;
    let seq_len = 10;
    let features = 10;
    let hidden = 2 * features;

    let mut reservoir = RNNReservoir::new(features, hidden);
    reservoir.set_spectral_radius(0.9, 1000);

    let mut readout = Linear::new(hidden, features, xavier_normal);

    let mut optim = AdamW::default().with(&mut readout);
    optim.learning_rate = 1e-2;

    let batch = generate_linear_recurrence(batch_size, seq_len, features);
    let x = batch.slice(s![.., 1..(seq_len as isize - 1), ..]);
    let y = batch.slice(s![.., 2.., ..]);

    for e in 0..1000 {
        let encoded = reservoir.forward(x.clone());
        let (b, s, _) = encoded.dim3();
        let encoded_2d = encoded.reshape((b * s, hidden));
        let y_pred_2d = readout.forward(&encoded_2d, true);
        let y_pred = y_pred_2d.reshape((b, s, features));

        let d_loss = (&y_pred - &y) * 2.0;
        let loss = (&y_pred - &y).powi(2).mean();
        println!("{e} loss={loss}");

        let d_loss_2d = d_loss.reshape((b * s, features));
        readout.backward(&d_loss_2d);
        optim.step(&mut readout);
        readout.zero_grads();
    }

    let encoded = reservoir.forward(x.clone());
    let (b, s, _) = encoded.dim3();
    let y_pred = readout
        .forward(&encoded.reshape((b * s, hidden)), false)
        .reshape((b, s, features));
    let loss = (&y_pred - &y).powi(2).mean();

    let max_viable_loss = 0.05;
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

    let mut reservoir = RNNReservoir::new(features, hidden);
    reservoir.set_spectral_radius(0.95, 1000);

    let mut readout = Linear::new(hidden, features, xavier_normal);

    let mut optim = AdamW::default().with(&mut readout);
    optim.learning_rate = 1e-2;

    let batch = generate_linear_recurrence(batch_size, seq_len, features);
    let x = batch.slice(s![.., 1..(seq_len as isize - 1), ..]);
    let y = batch.slice(s![.., 2.., ..]);

    let (_, actual_seq_len, _) = x.dim3();

    for e in 0..1000 {
        let mut h = Tensor2::zeros((batch_size, hidden));
        let mut y_pred = Tensor3::zeros((batch_size, actual_seq_len, features));

        for i in 0..actual_seq_len {
            let x_t = x.slice(s![.., i, ..]);
            reservoir.step(&x_t, &mut h);
            let y_pred_t = readout.forward(&h, true);
            y_pred.slice_assign(s![.., i, ..], &y_pred_t);

            let y_t = y.slice(s![.., i, ..]);
            let d_loss = (&y_pred_t - &y_t) * 2.0;
            readout.backward(&d_loss);
        }

        let loss = (&y_pred - &y).powi(2).mean();
        println!("{e} loss={loss}");

        optim.step(&mut readout);
        readout.zero_grads();
    }

    // evaluate with full forward
    let encoded = reservoir.forward(x.clone());
    let (b, s, _) = encoded.dim3();
    let y_pred = readout
        .forward(&encoded.reshape((b * s, hidden)), false)
        .reshape((b, s, features));
    let loss = (&y_pred - &y).powi(2).mean();

    let max_viable_loss = 0.05;
    assert!(
        loss < max_viable_loss,
        "ESN doesnt effectively train with step forward, test loss {loss} > {max_viable_loss}"
    );
}

// ── LSM-style tests (SNNReservoir + Linear) ─────────────────────────

#[test]
fn lsm_temporal_classification() {
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

    let mut readout = Linear::new(hidden, 1, xavier_normal);

    let mut optim = AdamW::default().with(&mut readout);
    optim.learning_rate = 5e-3;

    // build dataset: first half ramps (label 0), second half decays (label 1)
    let mut x = Tensor3::zeros((batch_size, seq_len, features));
    let mut y = Tensor2::zeros((batch_size, 1));

    for b in 0..batch_size {
        let noise = Tensor::random_uniform(seq_len) * 0.1 - 0.05;

        for t in 0..seq_len {
            let frac = t as Float / seq_len as Float;
            let val = if b < samples_per_class {
                frac + noise[[t]]
            } else {
                (-3.0 * frac).exp() + noise[[t]]
            };
            x[[b, t, 0]] = val;
        }

        if b >= samples_per_class {
            y[[b, 0]] = 1.0;
        }
    }

    // train
    for e in 0..300 {
        let mut state = Tensor2::zeros((batch_size, hidden));
        let mut output = Tensor2::zeros((batch_size, 1));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            reservoir.step(&x_t, &mut state, delta);
            output = readout
                .forward(&state, true)
                .mapv(|v| 1.0 / (1.0 + (-v).exp()));
        }

        // BCE gradient for sigmoid output: pred - target
        let d_loss = &output - &y;
        readout.backward(&d_loss);

        if e % 50 == 0 {
            let bce = -(&y * &output.mapv(|v| (v + 1e-8).ln())
                + &(1.0 - &y) * &output.mapv(|v| (1.0 - v + 1e-8).ln()))
                .mean();
            println!("{e} bce_loss={bce:.4}");
        }

        optim.step(&mut readout);
        readout.zero_grads();
    }

    // evaluate
    let mut state = Tensor2::zeros((batch_size, hidden));
    let mut output = Tensor2::zeros((batch_size, 1));

    for t in 0..seq_len {
        let x_t = x.slice(s![.., t, ..]);
        reservoir.step(&x_t, &mut state, delta);
        output = readout
            .forward(&state, false)
            .mapv(|v| 1.0 / (1.0 + (-v).exp()));
    }

    let mut correct = 0;
    for b in 0..batch_size {
        let pred = if output[[b, 0]] > 0.5 { 1 } else { 0 };
        let label = if y[[b, 0]] > 0.5 { 1 } else { 0 };
        if pred == label {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / batch_size as f64;
    println!("classification accuracy: {accuracy:.2}");

    assert!(
        accuracy > 0.9,
        "LSM temporal classification accuracy {accuracy:.2} <= 0.9"
    );
}

#[test]
fn lsm_heterogeneous_delta() {
    let samples_per_class = 32;
    let batch_size = samples_per_class * 2;
    let seq_len = 20;
    let features = 1;
    let hidden = 40;

    let mut reservoir = SNNReservoir::new(features, hidden);
    reservoir.set_spectral_radius(0.9, 2000);
    reservoir.set_tau_range(0.05, 0.5);
    reservoir.set_threshold_range(0.3, 0.8);

    let mut readout = Linear::new(hidden, 1, xavier_normal);

    let mut optim = AdamW::default().with(&mut readout);
    optim.learning_rate = 5e-3;

    // random per-step deltas in [0.02, 0.3]
    let deltas_tensor = Tensor::random_uniform(seq_len) * 0.28 + 0.02;
    let deltas: Vec<Float> = deltas_tensor.to_vec();

    // cumulative time for signal generation
    let mut cum_time = vec![0.0; seq_len + 1];
    for t in 0..seq_len {
        cum_time[t + 1] = cum_time[t] + deltas[t];
    }
    let total_time = cum_time[seq_len];

    // build dataset
    let mut x = Tensor3::zeros((batch_size, seq_len, features));
    let mut y = Tensor2::zeros((batch_size, 1));

    for b in 0..batch_size {
        let noise = Tensor::random_uniform(seq_len) * 0.1 - 0.05;

        for t in 0..seq_len {
            let frac = cum_time[t + 1] / total_time;
            let val = if b < samples_per_class {
                (2.0 * std::f64::consts::PI as Float * frac).sin() + noise[[t]]
            } else {
                (1.0 - frac) + noise[[t]]
            };
            x[[b, t, 0]] = val;
        }

        if b >= samples_per_class {
            y[[b, 0]] = 1.0;
        }
    }

    // train
    for e in 0..300 {
        let mut state = Tensor2::zeros((batch_size, hidden));
        let mut output = Tensor2::zeros((batch_size, 1));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            reservoir.step(&x_t, &mut state, deltas[t]);
            output = readout
                .forward(&state, true)
                .mapv(|v| 1.0 / (1.0 + (-v).exp()));
        }

        let d_loss = &output - &y;
        readout.backward(&d_loss);

        if e % 50 == 0 {
            let bce = -(&y * &output.mapv(|v| (v + 1e-8).ln())
                + &(1.0 - &y) * &output.mapv(|v| (1.0 - v + 1e-8).ln()))
                .mean();
            println!("{e} bce_loss={bce:.4}");
        }

        optim.step(&mut readout);
        readout.zero_grads();
    }

    // evaluate
    let mut state = Tensor2::zeros((batch_size, hidden));
    let mut output = Tensor2::zeros((batch_size, 1));

    for t in 0..seq_len {
        let x_t = x.slice(s![.., t, ..]);
        reservoir.step(&x_t, &mut state, deltas[t]);
        output = readout
            .forward(&state, false)
            .mapv(|v| 1.0 / (1.0 + (-v).exp()));
    }

    let mut correct = 0;
    for b in 0..batch_size {
        let pred = if output[[b, 0]] > 0.5 { 1 } else { 0 };
        let label = if y[[b, 0]] > 0.5 { 1 } else { 0 };
        if pred == label {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / batch_size as f64;
    println!("heterogeneous delta accuracy: {accuracy:.2}");

    assert!(
        accuracy > 0.9,
        "LSM heterogeneous delta accuracy {accuracy:.2} <= 0.9"
    );
}

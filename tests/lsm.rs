use nbml::{
    f::Activation,
    nn::{FFN, LSM},
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array1, Array2, Array3, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

/// Test LSM temporal pattern classification.
/// Two classes of 1-d time series are fed through the reservoir:
///   class 0: rising ramp   x(t) = t / seq_len
///   class 1: decaying pulse x(t) = exp(-3t / seq_len)
/// The readout must learn to classify based on final reservoir state,
/// exercising the reservoir's nonlinear spike dynamics and fading memory.
#[test]
fn lsm_temporal_classification() {
    let samples_per_class = 32;
    let batch_size = samples_per_class * 2;
    let seq_len = 20;
    let features = 1;
    let hidden = 40;
    let delta = 0.1;

    let mut model = LSM::new(features, hidden, 1);

    model.set_readout(FFN::new(vec![(hidden, 1, Activation::Sigmoid)]));
    model.reservoir.set_spectral_radius(0.9, 2000);
    model.reservoir.set_tau_range(0.05, 0.5);
    model.reservoir.set_threshold_range(0.3, 0.8);

    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 5e-3;

    // build dataset: first half ramps (label 0), second half decays (label 1)
    let mut x = Array3::zeros((batch_size, seq_len, features));
    let mut y = Array2::zeros((batch_size, 1)); // 0 = ramp, 1 = decay

    for b in 0..batch_size {
        let noise_scale = 0.05;
        let noise = Array1::random(seq_len, Uniform::new(-noise_scale, noise_scale));

        for t in 0..seq_len {
            let frac = t as f64 / seq_len as f64;
            let val = if b < samples_per_class {
                frac + noise[t]
            } else {
                (-3.0 * frac).exp() + noise[t]
            };
            x[[b, t, 0]] = val;
        }

        if b >= samples_per_class {
            y[[b, 0]] = 1.0; // class 1 = decay
        }
    }

    // train
    for e in 0..300 {
        let mut state = Array2::zeros((batch_size, hidden));

        let mut output = Array2::zeros((batch_size, 1));

        // run full sequence, only use final timestep for classification
        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]).to_owned();
            output = model.step(&x_t, delta, &mut state, true);
        }

        // BCE gradient for sigmoid output: pred - target
        let d_loss = &output - &y;
        model.backward(d_loss.to_owned());

        if e % 50 == 0 {
            let bce = -(&y * &output.mapv(|v| (v + 1e-8).ln())
                + &(1.0 - &y) * &output.mapv(|v| (1.0 - v + 1e-8).ln()))
                .mean()
                .unwrap();
            println!("{e} bce_loss={bce:.4}");
        }

        optim.step(&mut model);
        model.zero_grads();
    }

    // evaluate on the same data (no grad)
    let mut state = Array2::zeros((batch_size, hidden));
    let mut output = Array2::zeros((batch_size, 1));

    for t in 0..seq_len {
        let x_t = x.slice(s![.., t, ..]).to_owned();
        output = model.step(&x_t, delta, &mut state, false);
    }

    // compute accuracy
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

/// Test LSM with heterogeneous timesteps (varying delta per step).
/// The reservoir's tau-based decay exp(-delta/tau) and refractory tracking
/// must handle irregular time spacing. Input signals are sampled at random
/// intervals, so the readout must learn despite non-uniform dynamics.
#[test]
fn lsm_heterogeneous_delta() {
    let samples_per_class = 32;
    let batch_size = samples_per_class * 2;
    let seq_len = 20;
    let features = 1;
    let hidden = 40;

    let mut model = LSM::new(features, hidden, 1);

    model.set_readout(FFN::new(vec![(hidden, 1, Activation::Sigmoid)]));
    model.reservoir.set_spectral_radius(0.9, 2000);
    model.reservoir.set_tau_range(0.05, 0.5);
    model.reservoir.set_threshold_range(0.3, 0.8);

    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 5e-3;

    // random per-step deltas in [0.02, 0.3] — shared across epochs for consistency
    let deltas = Array1::random(seq_len, Uniform::new(0.02, 0.3));

    // cumulative time for signal generation
    let mut cum_time = vec![0.0f64; seq_len + 1];
    for t in 0..seq_len {
        cum_time[t + 1] = cum_time[t] + deltas[t];
    }
    let total_time = cum_time[seq_len];

    // build dataset using continuous-time signals sampled at irregular times
    //   class 0: sin(2π t / T)        (oscillating)
    //   class 1: 1 - t/T              (linear decay)
    let mut x = Array3::zeros((batch_size, seq_len, features));
    let mut y = Array2::zeros((batch_size, 1));

    for b in 0..batch_size {
        let noise = Array1::random(seq_len, Uniform::new(-0.05, 0.05));

        for t in 0..seq_len {
            let frac = cum_time[t + 1] / total_time;
            let val = if b < samples_per_class {
                (2.0 * std::f64::consts::PI * frac).sin() + noise[t]
            } else {
                (1.0 - frac) + noise[t]
            };
            x[[b, t, 0]] = val;
        }

        if b >= samples_per_class {
            y[[b, 0]] = 1.0;
        }
    }

    // train
    for e in 0..300 {
        let mut state = Array2::zeros((batch_size, hidden));
        let mut output = Array2::zeros((batch_size, 1));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]).to_owned();
            output = model.step(&x_t, deltas[t], &mut state, true);
        }

        let d_loss = &output - &y;
        model.backward(d_loss.to_owned());

        if e % 50 == 0 {
            let bce = -(&y * &output.mapv(|v| (v + 1e-8).ln())
                + &(1.0 - &y) * &output.mapv(|v| (1.0 - v + 1e-8).ln()))
                .mean()
                .unwrap();
            println!("{e} bce_loss={bce:.4}");
        }

        optim.step(&mut model);
        model.zero_grads();
    }

    // evaluate
    let mut state = Array2::zeros((batch_size, hidden));
    let mut output = Array2::zeros((batch_size, 1));

    for t in 0..seq_len {
        let x_t = x.slice(s![.., t, ..]).to_owned();
        output = model.step(&x_t, deltas[t], &mut state, false);
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

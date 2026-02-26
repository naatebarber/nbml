use nbml::{
    nn::LSM,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array2, Array3, Axis, concatenate, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn lsm_sequence_pred_step() {
    let batch_size = 10;
    let seq_len = 10;
    let features = 10;
    let hidden = 20;

    let mut model = LSM::new(features, hidden, features);
    model.reservoir.set_spectral_radius(0.95, 2000);

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

        let mut state = Array2::zeros((batch_size, hidden));
        let mut since_spike = Array2::zeros((batch_size, hidden));

        let mut y_pred = Array3::zeros((batch_size, seq_len, features));

        for i in 0..seq_len {
            let x_t = x.slice(s![.., i, ..]).to_owned();
            let y_pred_t = model.step(&x_t, 0.1, &mut state, &mut since_spike, true);
            y_pred.slice_mut(s![.., i, ..]).assign(&y_pred_t);

            let y_t = y.slice(s![.., i, ..]);
            let d_loss = 2. * (&y_pred_t - &y_t);
            model.backward(d_loss);
        }

        let loss = (&y_pred - &y).powi(2).mean().unwrap();
        println!("{e} loss={loss}");

        optim.step(&mut model);

        model.zero_grads();
    }

    let mut state = Array2::zeros((x.dim().0, hidden));
    let mut since_spike = Array2::zeros(state.dim());

    let mut y_pred = Array3::zeros(x.dim());
    for t in 0..x.dim().1 {
        let x_t = x.slice(s![.., t, ..]).to_owned();
        y_pred.slice_mut(s![.., t, ..]).assign(&model.step(&x_t, 0.1, &mut state, &mut since_spike, false));
    }

    let loss = (&y_pred - &y).powi(2).mean().unwrap();

    let max_viable_loss = 0.01;
    assert!(
        loss < max_viable_loss,
        "LSTM doesnt effectively train with step forward, test loss {loss} > {max_viable_loss}"
    );
}


use nbml::{
    nn::RNN,
    optim::{AdamW, Optimizer, ToIntermediates, ToParams},
};
use ndarray::{Array2, Array3, Axis, concatenate, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn intermediate_caching() {
    let mut model = RNN::new(4);
    let x = Array3::random((2, 3, 4), Uniform::new(0., 1.).unwrap());
    let x2 = Array3::random((2, 3, 4), Uniform::new(0., 1.).unwrap());
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
fn rnn_sequence_pred() {
    let batch_size = 10;
    let seq_len = 10;
    let features = 10;

    let mut model = RNN::new(features);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let seed = Array3::random((batch_size, 2, features), Uniform::new(-1., 1.).unwrap());

    let batch = Array3::zeros((batch_size, seq_len - 2, features));
    let mut batch = concatenate![Axis(1), seed.view(), batch.view()];

    for t in 2..batch.dim().1 {
        let a = 0.52;
        let b = 0.48;

        let prev_1 = a * batch.slice(s![.., t - 1, ..]).to_owned();
        let prev_2 = b * batch.slice(s![.., t - 2, ..]).to_owned();
        let c: Array2<f32> = &prev_1 + &prev_2;

        batch.slice_mut(s![.., t, ..]).assign(&c)
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
        "RNN doesnt effectively train with test loss {loss} > {max_viable_loss}"
    );
}

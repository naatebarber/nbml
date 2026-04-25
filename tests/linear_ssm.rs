use nbml::{
    f::he,
    nn::LinearSSM,
    optim::{AdamW, Optimizer, ToIntermediates, ToParams},
};
use ndarray::{Array2, Array3, Axis, concatenate, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn intermediate_caching() {
    let mut model = LinearSSM::new(4, 3, 3, he, he, he);
    let x = Array3::random((2, 4, 3), Uniform::new(0., 1.).unwrap());
    let x2 = Array3::random((2, 4, 3), Uniform::new(0., 1.).unwrap());
    let d = Array3::ones((2, 4, 3));

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
    assert!(grads_a.d_a != grads_b.d_a);
    assert!(
        grads_a.d_a == grads_c.d_a,
        "intermediate caching process fucks with gradients"
    );
}

fn generate_linear_recurrence(batch: usize, len: usize, feat: usize) -> Array3<f32> {
    let seed = Array3::random((batch, 2, feat), Uniform::new(-1., 1.).unwrap());
    let rest = Array3::zeros((batch, len - 2, feat));
    let mut seq = concatenate![Axis(1), seed.view(), rest.view()];

    for t in 2..len {
        let next: Array2<f32> =
            0.52 * &seq.slice(s![.., t - 1, ..]) + 0.48 * &seq.slice(s![.., t - 2, ..]);
        seq.slice_mut(s![.., t, ..]).assign(&next);
    }
    seq
}

#[test]
fn linear_ssm_sequence_pred() {
    let seq_len = 20;
    let features = 10;

    let d_model = 20;

    let mut model = LinearSSM::new(d_model, features, features, he, he, he);

    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let train_batch = generate_linear_recurrence(100, seq_len, features);
    let x = train_batch.slice(s![.., 1..(seq_len - 1), ..]).to_owned();
    let y = train_batch.slice(s![.., 2.., ..]).to_owned();

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

    let test_batch = generate_linear_recurrence(10, seq_len, features);
    let x = test_batch.slice(s![.., 1..(seq_len - 1), ..]).to_owned();
    let y = test_batch.slice(s![.., 2.., ..]).to_owned();

    let y_pred = model.forward(x.clone(), false);
    let loss = (&y_pred - &y).powi(2).mean().unwrap();

    let max_viable_loss = 0.01;
    assert!(
        loss < max_viable_loss,
        "LinearSSM doesnt effectively train with test loss {loss} > {max_viable_loss}"
    );
}

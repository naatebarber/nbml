use nbml::{
    f::he,
    nn::LinearSSM,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array3, Axis, concatenate, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

fn generate_linear_recurrence(batch: usize, len: usize, feat: usize) -> Array3<f64> {
    let seed = Array3::random((batch, 2, feat), Uniform::new(-1., 1.));
    let rest = Array3::zeros((batch, len - 2, feat));
    let mut seq = concatenate![Axis(1), seed.view(), rest.view()];

    for t in 2..len {
        let next = 0.52 * &seq.slice(s![.., t - 1, ..]) + 0.48 * &seq.slice(s![.., t - 2, ..]);
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

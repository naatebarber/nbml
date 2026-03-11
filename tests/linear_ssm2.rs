use nbml::{
    Tensor,
    f2::he,
    nn2::LinearSSM,
    optim2::{Optimizer, ToParams, adam::AdamW},
    s,
    tensor::Tensor3,
};

fn generate_linear_recurrence(batch: usize, len: usize, feat: usize) -> Tensor3 {
    let seed = Tensor::random_uniform((batch, 2, feat)) * 2.0 - 1.0;
    let zeros = Tensor3::zeros((batch, len - 2, feat));
    let mut seq = Tensor::concatenate(1, &[&seed, &zeros]);

    for t in 2..len {
        let prev1 = seq.slice(s![.., (t - 1), ..]);
        let prev2 = seq.slice(s![.., (t - 2), ..]);
        let next = prev1 * 0.52 + prev2 * 0.48;
        seq.slice_assign(s![.., t, ..], &next);
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
    let x = train_batch.slice(s![.., 1..(seq_len as isize - 1), ..]);
    let y = train_batch.slice(s![.., 2.., ..]);

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

    let test_batch = generate_linear_recurrence(10, seq_len, features);
    let x = test_batch.slice(s![.., 1..(seq_len as isize - 1), ..]);
    let y = test_batch.slice(s![.., 2.., ..]);

    let y_pred = model.forward(x.clone(), false);
    let loss = (&y_pred - &y).powi(2).mean();

    let max_viable_loss = 0.01;
    assert!(
        loss < max_viable_loss,
        "LinearSSM doesnt effectively train with test loss {loss} > {max_viable_loss}"
    );
}

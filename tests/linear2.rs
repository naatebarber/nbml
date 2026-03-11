use nbml::{
    Tensor,
    f2::xavier_normal,
    layers::Linear,
    optim2::{Optimizer, ToParams, adam::AdamW},
};

#[test]
fn linear_learns_identity() {
    let d = 8;
    let mut model = Linear::new(d, d, xavier_normal);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let x = Tensor::random_uniform((16, d)) * 2.0 - 1.0;
    let y = x.clone();

    let mut final_loss = f64::MAX;
    for epoch in 0..1000 {
        let y_pred = model.forward(&x, true);
        let loss = (&y_pred - &y).powi(2).mean();
        let n = y_pred.shape().iter().product::<usize>() as f64;
        let d_loss = (&y_pred - &y) * (2.0 / n);

        model.backward(&d_loss);
        optim.step(&mut model);
        model.zero_grads();

        final_loss = loss;

        if epoch % 200 == 0 {
            println!("epoch {epoch} loss={loss:.6}");
        }
    }

    println!("final loss={final_loss:.6}");
    assert!(
        final_loss < 0.01,
        "Linear failed to learn identity: loss={final_loss:.6}"
    );
}

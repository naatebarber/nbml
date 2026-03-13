use nbml::{
    f::Activation,
    nn::FFN,
    optim::{AdamW, Optimizer, ToParams},
};
use ndarray::Array2;

#[test]
fn ffn_learns_xor() {
    let mut model = FFN::new(vec![
        (2, 16, Activation::Relu),
        (16, 1, Activation::Sigmoid),
    ]);

    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let x = Array2::from_shape_vec((4, 2), vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();

    let y = Array2::from_shape_vec((4, 1), vec![0., 1., 1., 0.]).unwrap();

    for _ in 0..1000 {
        let y_pred = model.forward(x.clone(), true);
        let d_loss = 2. * (&y_pred - &y);
        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();
    }

    let y_pred = model.forward(x, false);

    for i in 0..4 {
        let pred = y_pred[[i, 0]];
        let target = y[[i, 0]];
        let rounded = if pred > 0.5 { 1.0 } else { 0.0 };
        assert!(
            rounded == target,
            "XOR failed at sample {i}: pred={pred:.4} expected={target}"
        );
    }
}

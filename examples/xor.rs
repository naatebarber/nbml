use ndarray::array;

use nbml::{
    f,
    nn::ffn::FFN,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};

pub fn main() {
    // XOR dataset
    let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],];
    let y = array![[0.0], [1.0], [1.0], [0.0]];

    // Define network: 2 -> 4 -> 1 with sigmoid output
    let mut net = FFN::new(vec![
        (2, 4, f::Activation::Relu),
        (4, 1, f::Activation::Sigmoid),
    ]);

    // Optimizer
    let mut opt = AdamW::default().with(&mut net);
    opt.learning_rate = 0.01;

    for epoch in 0..10000 {
        // Forward pass
        let y_pred = net.forward(x.clone(), true);

        // Loss = MSE
        let loss = (&y_pred - &y).mapv(|v| v.powi(2)).mean().unwrap();

        // Backward pass: dL/dy = 2*(y_pred - y)/N
        let d_loss = 2.0 * (&y_pred - &y) / (y.shape()[0] as f64);
        net.backward(d_loss);

        // Update
        opt.step(&mut net);
        net.zero_grads();

        if epoch % 1000 == 0 {
            println!("epoch {epoch}, loss {loss}");
        }
    }

    // Final predictions
    let y_final = net.forward(x.clone(), false);
    println!("XOR predictions:\n{:?}", y_final);
}

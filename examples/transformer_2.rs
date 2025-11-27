use nbml::{
    f::Activation,
    nn::{FFN, SequencePooling, TransformerEncoder},
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array1, Array2, Array3, Axis, stack};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

struct Classifier {
    transformer: TransformerEncoder,
    pooling: SequencePooling,
    feed_forward: FFN,
}

impl Classifier {
    pub fn new(d_model: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            transformer: TransformerEncoder::new(
                d_model,
                d_head,
                n_head,
                vec![(d_model, d_model, Activation::Relu)],
            ),
            pooling: SequencePooling::new(),
            feed_forward: FFN::new(vec![(d_model, 1, Activation::Sigmoid)]),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>) -> Array2<f64> {
        let x = self.transformer.forward(x, true);
        let x = self.pooling.forward(x, true);
        self.feed_forward.forward(x, true)
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) -> Array3<f64> {
        let d = self.feed_forward.backward(d_loss);
        let d = self.pooling.backward(d);
        self.transformer.backward(d)
    }
}

impl ToParams for Classifier {
    fn params(&mut self) -> Vec<nbml::optim::param::Param> {
        let mut params = vec![];
        params.append(&mut self.transformer.params());
        params.append(&mut self.feed_forward.params());
        params
    }
}

// Generate synthetic data with clear patterns
fn generate_data(n_samples: usize, seq_len: usize, d_model: usize) -> (Array3<f64>, Array2<f64>) {
    let mut x_samples = Vec::new();
    let mut y_samples = Vec::new();

    for i in 0..n_samples {
        let label = if i < n_samples / 2 { 1.0 } else { 0.0 };

        let mut sequence = Array2::zeros((seq_len, d_model));

        if label == 1.0 {
            // Class 1: Positive values, increasing pattern
            for t in 0..seq_len {
                for d in 0..d_model {
                    sequence[[t, d]] =
                        0.5 + 0.5 * (t as f64 / seq_len as f64) + 0.1 * (d as f64 / d_model as f64);
                }
            }
        } else {
            // Class 0: Negative values, decreasing pattern
            for t in 0..seq_len {
                for d in 0..d_model {
                    sequence[[t, d]] = -0.5
                        - 0.5 * (t as f64 / seq_len as f64)
                        - 0.1 * (d as f64 / d_model as f64);
                }
            }
        }

        // Add some noise
        sequence = sequence + Array2::random((seq_len, d_model), Uniform::new(-0.1, 0.1));

        x_samples.push(sequence);
        y_samples.push(Array1::from_vec(vec![label]));
    }

    // Stack into batches
    let x_views: Vec<_> = x_samples.iter().map(|a| a.view()).collect();
    let y_views: Vec<_> = y_samples.iter().map(|a| a.view()).collect();

    let x = stack(Axis(0), &x_views).unwrap();
    let y = stack(Axis(0), &y_views).unwrap();

    (x, y)
}

fn main() {
    println!("=== Transformer Sequence Classification Test ===\n");

    // Model hyperparameters
    let d_model = 32;
    let d_head = 8;
    let n_head = 4;
    let seq_len = 10;
    let n_train = 100;
    let n_test = 20;
    let epochs = 500;
    let learning_rate = 0.001;

    println!("Configuration:");
    println!("  d_model: {}", d_model);
    println!("  d_head: {}", d_head);
    println!("  n_head: {}", n_head);
    println!("  seq_len: {}", seq_len);
    println!("  train samples: {}", n_train);
    println!("  test samples: {}", n_test);
    println!("  learning_rate: {}\n", learning_rate);

    // Generate training and test data
    let (x_train, y_train) = generate_data(n_train, seq_len, d_model);
    let (x_test, y_test) = generate_data(n_test, seq_len, d_model);

    println!("Data shapes:");
    println!("  x_train: {:?}", x_train.shape());
    println!("  y_train: {:?}", y_train.shape());
    println!("  x_test: {:?}", x_test.shape());
    println!("  y_test: {:?}\n", y_test.shape());

    // Initialize model and optimizer
    let mut model = Classifier::new(d_model, d_head, n_head);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = learning_rate;
    optim.beta1 = 0.9;
    optim.beta2 = 0.999;
    optim.epsilon = 1e-8;
    optim.weight_decay = 0.;

    println!("Starting training...\n");

    let mut best_loss = f64::INFINITY;
    let mut losses = Vec::new();

    for epoch in 0..epochs {
        // Forward pass
        let y_pred = model.forward(x_train.clone());

        // Calculate loss with numerical stability
        let eps = 1e-7;
        let y_pred_clipped = y_pred.mapv(|v| v.max(eps).min(1. - eps));
        let loss = -(&y_train * &y_pred_clipped.mapv(|v| v.ln())
            + (1. - &y_train) * (1. - &y_pred_clipped).mapv(|v| v.ln()))
        .mean()
        .unwrap();

        losses.push(loss);

        // Backward pass
        let d_loss = &y_pred - &y_train;
        model.backward(d_loss);

        // Update weights
        optim.step(&mut model);

        // Track best loss
        if loss < best_loss {
            best_loss = loss;
        }

        // Print progress
        if epoch % 50 == 0 || epoch == epochs - 1 {
            // Calculate training accuracy
            let train_acc = calculate_accuracy(&y_pred, &y_train);

            // Evaluate on test set
            let y_test_pred = model.forward(x_test.clone());
            let test_loss_val = -(&y_test * &y_test_pred.mapv(|v| v.max(eps).min(1. - eps).ln())
                + (1. - &y_test)
                    * (1. - &y_test_pred).mapv(|v| (1. - v.max(eps).min(1. - eps)).ln()))
            .mean()
            .unwrap();
            let test_acc = calculate_accuracy(&y_test_pred, &y_test);

            println!(
                "Epoch {:3} | Train Loss: {:.6} | Train Acc: {:.2}% | Test Loss: {:.6} | Test Acc: {:.2}%",
                epoch,
                loss,
                train_acc * 100.0,
                test_loss_val,
                test_acc * 100.0
            );
        }
    }

    println!("\n=== Final Evaluation ===");

    // Final test evaluation
    let y_test_pred = model.forward(x_test.clone());
    let test_acc = calculate_accuracy(&y_test_pred, &y_test);

    println!("Best training loss: {:.6}", best_loss);
    println!("Final test accuracy: {:.2}%\n", test_acc * 100.0);

    // Show some predictions
    println!("Sample predictions (first 10 test samples):");
    println!("{:<10} {:<10} {:<10}", "Predicted", "Actual", "Correct?");
    println!("{}", "-".repeat(35));

    for i in 0..10.min(n_test) {
        let pred = y_test_pred[[i, 0]];
        let actual = y_test[[i, 0]];
        let pred_class = if pred > 0.5 { 1.0 } else { 0.0 };
        let correct = if pred_class == actual { "✓" } else { "✗" };

        println!("{:<10.4} {:<10.0} {:<10}", pred, actual, correct);
    }

    // Analyze loss trend
    println!("\n=== Loss Analysis ===");
    let initial_loss = losses[0];
    let final_loss = losses[losses.len() - 1];
    let loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100.0;

    println!("Initial loss: {:.6}", initial_loss);
    println!("Final loss: {:.6}", final_loss);
    println!("Loss reduction: {:.2}%", loss_reduction);

    // Check if learning occurred
    if loss_reduction < 5.0 {
        println!("\n⚠️  WARNING: Loss did not decrease significantly!");
        println!("Possible issues:");
        println!("  - Gradients not flowing properly");
        println!("  - Learning rate too small");
        println!("  - Model capacity insufficient");
        println!("  - Bug in backward pass");
    } else if test_acc < 0.6 {
        println!("\n⚠️  WARNING: Poor test accuracy despite loss reduction!");
        println!("Possible issues:");
        println!("  - Overfitting to training data");
        println!("  - Data not separable enough");
        println!("  - Model architecture issue");
    } else {
        println!("\n✓ Model appears to be learning correctly!");
    }
}

fn calculate_accuracy(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let pred_classes = predictions.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
    let correct = pred_classes
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| (**p - **t).abs() < 0.1)
        .count();

    correct as f64 / predictions.len() as f64
}

use nbml::{
    Tensor,
    f2::{d_relu, he, relu, xavier_normal},
    layers::{LayerNorm, Linear},
    optim2::{Optimizer, Param, ToParams, adam::AdamW},
    tensor::{Float, Tensor3},
    util::Cache,
};

#[test]
fn test_layernorm_gradients() {
    let features = 5;
    let mut ln = LayerNorm::new(features);

    // Small random input
    let x = Tensor::random_uniform((2, 3, features)) * 2.0 - 1.0;

    // Analytical gradient via backward pass
    let y = ln.forward(x.clone(), true);
    let d_loss = Tensor::random_uniform(y.shape());
    let dx_analytical = ln.backward(d_loss.clone());

    // Numerical gradient via finite differences
    let epsilon = 1e-5;
    let mut dx_numerical = Tensor3::zeros(x.shape());

    for b in 0..2 {
        for s in 0..3 {
            for f in 0..features {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[b, s, f]] += epsilon;
                x_minus[[b, s, f]] -= epsilon;

                let mut ln_plus = LayerNorm::new(features);
                ln_plus.gamma = ln.gamma.clone();
                ln_plus.beta = ln.beta.clone();
                let y_plus = ln_plus.forward(x_plus, false);

                let mut ln_minus = LayerNorm::new(features);
                ln_minus.gamma = ln.gamma.clone();
                ln_minus.beta = ln.beta.clone();
                let y_minus = ln_minus.forward(x_minus, false);

                let loss_plus = (&d_loss * &y_plus).sum();
                let loss_minus = (&d_loss * &y_minus).sum();
                dx_numerical[[b, s, f]] = (loss_plus - loss_minus) / (2. * epsilon);
            }
        }
    }

    let diff = (&dx_analytical - &dx_numerical).mapv(|v| v.abs());
    let max_diff = diff.max();

    println!("Max absolute difference: {:.2e}", max_diff);

    assert!(
        max_diff < 1e-5,
        "LayerNorm gradients incorrect: max diff {:.2e}",
        max_diff
    );
}

#[test]
fn layernorm_output_is_normalized() {
    let d = 16;
    let mut ln = LayerNorm::new(d);

    let x = Tensor::random_uniform((2, 4, d)) * 10.0 - 5.0;
    let y = ln.forward(x, false);

    for b in 0..2 {
        for s in 0..4 {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for f in 0..d {
                let v = y[[b, s, f]];
                sum += v;
                sum_sq += v * v;
            }
            let mean = sum / d as Float;
            let var = sum_sq / d as Float - mean * mean;

            assert!(
                mean.abs() < 1e-6,
                "mean should be ~0, got {mean} at [{b},{s}]"
            );
            assert!(
                (var - 1.0).abs() < 0.1,
                "variance should be ~1, got {var} at [{b},{s}]"
            );
        }
    }
}

struct LNModel {
    linear1: Linear,
    ln: LayerNorm,
    linear2: Linear,
    cache: Cache,
}

impl LNModel {
    fn new(d: usize) -> Self {
        Self {
            linear1: Linear::new(d, d, he),
            ln: LayerNorm::new(d),
            linear2: Linear::new(d, d, xavier_normal),
            cache: Cache::new(),
        }
    }

    fn forward(&mut self, x: Tensor3, grad: bool) -> Tensor3 {
        let (b, s, f) = x.dim3();
        let x2 = x.reshape((b * s, f));
        let z = self.linear1.forward(&x2, grad);

        if grad {
            self.cache.set("z", z.clone());
        }

        let a = relu(&z).reshape((b, s, f));
        let n = self.ln.forward(a, grad);
        let n2 = n.reshape((b * s, f));
        let out = self.linear2.forward(&n2, grad);
        out.reshape((b, s, f))
    }

    fn backward(&mut self, d_loss: Tensor3) -> Tensor3 {
        let (b, s, f) = d_loss.dim3();
        let d = d_loss.reshape((b * s, f));
        let d = self.linear2.backward(&d);
        let d = d.reshape((b, s, f));
        let d = self.ln.backward(d);
        let d = d.reshape((b * s, f));
        let d = &d * d_relu(&self.cache["z"]);
        let d = self.linear1.backward(&d);
        d.reshape((b, s, f))
    }
}

impl ToParams for LNModel {
    fn params(&mut self) -> Vec<Param> {
        let mut p = vec![];
        p.append(&mut self.linear1.params());
        p.append(&mut self.ln.params());
        p.append(&mut self.linear2.params());
        p
    }

    fn zero_grads(&mut self) {
        self.linear1.zero_grads();
        self.ln.zero_grads();
        self.linear2.zero_grads();
    }
}

#[test]
fn layernorm_trains_in_model() {
    let d = 8;
    let mut model = LNModel::new(d);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-3;

    let x = Tensor::random_uniform((4, 3, d)) * 2.0 - 1.0;
    let y = Tensor::random_uniform((4, 3, d)) * 2.0 - 1.0;

    let initial_loss;
    {
        let y_pred = model.forward(x.clone(), false);
        initial_loss = (&y_pred - &y).powi(2).mean();
    }

    let mut final_loss = initial_loss;
    for epoch in 0..2000 {
        let y_pred = model.forward(x.clone(), true);
        let loss = (&y_pred - &y).powi(2).mean();
        let n = y_pred.shape().iter().product::<usize>() as Float;
        let d_loss = (&y_pred - &y) * (2.0 / n);

        model.backward(d_loss);
        optim.step(&mut model);
        model.zero_grads();

        final_loss = loss;

        if epoch % 500 == 0 {
            println!("epoch {epoch} loss={loss:.6}");
        }
    }

    println!("initial={initial_loss:.6} final={final_loss:.6}");
    assert!(
        final_loss < initial_loss * 0.05,
        "model with LayerNorm failed to train: initial={initial_loss:.6} final={final_loss:.6}"
    );
    assert!(
        final_loss < 0.05,
        "model with LayerNorm did not converge: final={final_loss:.6}"
    );
}

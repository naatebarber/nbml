#![allow(deprecated)]

use nbml::{
    nn::attention_head::AttentionHead,
    optim::{adam::AdamW, optimizer::Optimizer},
};
use ndarray::{Array1, Array2, Array3, Axis};
use rand::{Rng, rngs::ThreadRng};

fn model_optimizer(
    d_model: usize,
    d_head: usize,
    n_head: usize,
) -> (AttentionHead, impl Optimizer) {
    let mut ah = AttentionHead::new(d_model, d_head, n_head);
    let mut optim = AdamW::default().with(&mut ah);
    optim.learning_rate = 1e-4;

    (ah, optim)
}

fn corruption_test() {
    let (mut ah, mut optim) = model_optimizer(1, 12, 1);

    let src = Array1::from_vec(vec![1., 2., 3., 4., 5.]);
    let x = src.clone().insert_axis(Axis(1)).insert_axis(Axis(0));
    let y_train = x.clone();

    let epochs = 10000;

    for e in 0..epochs {
        let y_pred = ah.forward(&x, &Array2::ones((1, 5)), false, true);
        let mse = (&y_pred - &y_train).mapv(|x| x * x).mean().unwrap();

        let d_loss = (&y_pred - &y_train) * 2.;

        ah.backward(d_loss);
        optim.step(&mut ah);

        if e > 100 && e % 100 == 0 {
            println!("(epoch {e}) loss={mse}");
        }
    }

    let y_pred = ah.forward(&x, &Array2::ones((1, 5)), false, false);
    println!("CONTROL: {:?}", y_pred);

    let mut y_test = src.clone();
    y_test[2] = 0.;
    let x_test = y_test.insert_axis(Axis(1)).insert_axis(Axis(0));

    let y_pred = ah.forward(&x_test, &Array2::ones((1, 5)), false, false);
    println!("TEST: {:?}", y_pred);
}

/**
 * FINDINGS
 *
 * Positional encoding makes or breaks relativistic inference. It needs to be correct.
 */
fn inference_test() {
    let (mut ah, mut optim) = model_optimizer(1, 12, 1);

    fn make_pair(rng: &mut ThreadRng) -> (Array3<f64>, Array3<f64>, Array2<f64>, usize) {
        let ix = rng.random_range(0..5);
        let mut x = vec![];
        let mut y = vec![];
        let mut lm = vec![];

        for i in 0..5 {
            if i == ix {
                y.push((i + 1) as f64);
                x.push(0.);
                lm.push(1.);
            } else {
                y.push(0.);
                x.push((i + 1) as f64);
                lm.push(0.);
            }

            x[i] += i as f64 * 0.1;
        }

        (
            Array1::from_vec(x)
                .insert_axis(Axis(1))
                .insert_axis(Axis(0)),
            Array1::from_vec(y)
                .insert_axis(Axis(1))
                .insert_axis(Axis(0)),
            Array1::from_vec(lm).insert_axis(Axis(0)),
            ix,
        )
    }

    let epochs = 100000;
    let mut rng = rand::rng();

    for e in 0..epochs {
        let (x, y, lm, ..) = make_pair(&mut rng);

        let (b, s, f) = y.dim();
        let d_loss_mask = lm
            .insert_axis(Axis(2))
            .broadcast((b, s, f))
            .unwrap()
            .to_owned();

        let y_pred = ah.forward(&x, &Array2::ones((1, 5)), false, true);
        let mse = ((&y_pred - &y) * &d_loss_mask)
            .mapv(|x| x * x)
            .mean()
            .unwrap();

        let mut d_loss = (&y_pred - &y) * 2.;
        d_loss = &d_loss_mask * &d_loss;

        ah.backward(d_loss);
        optim.step(&mut ah);

        if e > 100 && e % 100 == 0 {
            println!("(epoch {e}) loss={mse}");
        }
    }

    let (test_x, test_y, lm, ix) = make_pair(&mut rng);
    let (b, s, f) = test_y.dim();
    let _d_loss_mask = lm
        .insert_axis(Axis(2))
        .broadcast((b, s, f))
        .unwrap()
        .to_owned();

    let y_pred = ah.forward(&test_x, &Array2::ones((1, 5)), false, false);
    println!("ix: {}", ix);
    println!("QUESTION: {:?}", test_x);
    println!("TEST: {:?}", y_pred);
    println!("ANSWER: {:?}", test_y);
}

fn main() {
    corruption_test();
    inference_test();
}

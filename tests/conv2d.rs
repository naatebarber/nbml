use nbml::{
    f::he,
    nn::conv2d::{Conv2D, PatchwiseConv2D},
    optim::{AdamW, Optimizer, ToIntermediates, ToParams},
};
use ndarray::{Array4, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn intermediate_caching() {
    let mut model = Conv2D::new(3, 6, 3, 3, he);
    let x = Array4::random((2, 3, 8, 8), Uniform::new(0., 1.));
    let x2 = Array4::random((2, 3, 8, 8), Uniform::new(0., 1.));
    let d = Array4::ones((2, 6, 6, 6));

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
    assert!(grads_a.d_w != grads_b.d_w);
    assert!(
        grads_a.d_w == grads_c.d_w,
        "intermediate caching process fucks with gradients"
    );
}

#[test]
fn im2col_conv2d_produces_correct_shapes() {
    let mut conv = Conv2D::new(3, 6, 5, 5, he);
    let x = Array4::ones((5, 3, 38, 38));

    let x_conv = conv.forward(x.clone(), true);
    assert!(
        x_conv.dim() == (5, 6, 34, 34),
        "im2col conv2d produces incorrect shapes on forward, expected={:?} actual={:?}",
        (5, 6, 34, 34),
        x_conv.dim()
    );

    let d_conv = conv.backward(x_conv);
    assert!(
        x.dim() == d_conv.dim(),
        "im2col conv2d produces incorrect shapes on backward, expected={:?} actual={:?}",
        x.dim(),
        d_conv.dim()
    );
}

#[test]
fn patchwise_conv2d_produces_correct_shapes() {
    let mut conv = PatchwiseConv2D::new(3, 6, 5, 5, he);
    let x = Array4::ones((5, 3, 38, 38));

    let x_conv = conv.forward(x.clone(), true);
    assert!(
        x_conv.dim() == (5, 6, 34, 34),
        "patchwise conv2d produces incorrect shapes on forward, expected={:?} actual={:?}",
        (5, 6, 34, 34),
        x_conv.dim()
    );

    let d_conv = conv.backward(x_conv);
    assert!(
        x.dim() == d_conv.dim(),
        "patchwise conv2d produces incorrect shapes on backward, expected={:?} actual={:?}",
        x.dim(),
        d_conv.dim()
    );
}

fn inversion_dataset() -> (Array4<f64>, Array4<f64>) {
    let x = Array4::random((10, 1, 10, 10), Uniform::new(0., 1.)).round();
    let y = (&x - 1.).abs();

    (x, y)
}

#[test]
fn im2col_conv2d_trains_inversion() {
    let mut conv = Conv2D::new(1, 1, 1, 1, he);
    let mut optim = AdamW::default().with(&mut conv);
    optim.learning_rate = 0.01;

    let (x, y) = inversion_dataset();

    for e in 0..500 {
        let y_pred = conv.forward(x.clone(), true);
        let loss = (&y_pred - &y).pow2().mean().unwrap();

        println!("e={e} loss={loss}");

        let d_loss = 2. * (&y_pred - &y);
        conv.backward(d_loss);
        optim.step(&mut conv);
        conv.zero_grads();
    }

    let y_pred = conv.forward(x.clone(), false);
    let loss = (&y_pred - &y).pow2().mean().unwrap();

    let max_loss = 0.1;

    assert!(
        loss < max_loss,
        "im2col conv2d failed to train inversion, loss too high actual={loss} max={max_loss}"
    );
}

#[test]
fn patchwise_conv2d_trains_inversion() {
    let mut conv = PatchwiseConv2D::new(1, 1, 1, 1, he);
    let mut optim = AdamW::default().with(&mut conv);
    optim.learning_rate = 0.01;

    let (x, y) = inversion_dataset();

    for e in 0..500 {
        let y_pred = conv.forward(x.clone(), true);
        let loss = (&y_pred - &y).pow2().mean().unwrap();

        println!("e={e} loss={loss}");

        let d_loss = 2. * (&y_pred - &y);
        conv.backward(d_loss);
        optim.step(&mut conv);
        conv.zero_grads();
    }

    let mut losses = vec![];

    for e in 0..100 {
        let y_pred = conv.forward(x.clone(), false);
        let loss = (&y_pred - &y).pow2().mean().unwrap();

        println!("e={e} loss={loss}");
        losses.push(loss);
    }

    let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
    let max_loss = 0.1;

    assert!(
        avg_loss < max_loss,
        "patchwise conv2d failed to train inversion, loss too high actual={avg_loss} max={max_loss}"
    );
}

fn shift_dataset() -> (Array4<f64>, Array4<f64>) {
    let mut batch = Array4::zeros((10, 1, 11, 11));
    for i in 0..10 {
        let mut vertical_slice = batch.slice_mut(s![i, .., .., i]);
        vertical_slice += 1.;
    }

    let x = batch.slice(s![0..9, .., .., ..]).to_owned();
    let y = batch.slice(s![1.., .., 0..10, 0..10]).to_owned();

    (x, y)
}

#[test]
fn im2col_conv2d_trains_shift() {
    let mut conv = Conv2D::new(1, 1, 2, 2, he);
    let mut optim = AdamW::default().with(&mut conv);
    optim.learning_rate = 0.01;

    let (x, y) = shift_dataset();

    for e in 0..100 {
        let y_pred = conv.forward(x.clone(), true);

        println!("y_pred {:?} y {:?}", y_pred.dim(), y.dim());
        let loss = (&y_pred - &y).pow2().mean().unwrap();

        println!("e={e} loss={loss}");

        let d_loss = 2. * (&y_pred - &y);
        conv.backward(d_loss);
        optim.step(&mut conv);
        conv.zero_grads();
    }

    let y_pred = conv.forward(x.clone(), false);
    let loss = (&y_pred - &y).pow2().mean().unwrap();

    let max_loss = 0.1;

    assert!(
        loss < max_loss,
        "im2col conv2d failed to train shift, loss too high actual={loss} max={max_loss}"
    );
}

#[test]
fn patchwise_conv2d_trains_shift() {
    let mut conv = PatchwiseConv2D::new(1, 1, 2, 2, he);
    let mut optim = AdamW::default().with(&mut conv);
    optim.learning_rate = 0.01;

    let (x, y) = shift_dataset();

    for e in 0..100 {
        let y_pred = conv.forward(x.clone(), true);

        println!("y_pred {:?} y {:?}", y_pred.dim(), y.dim());
        let loss = (&y_pred - &y).pow2().mean().unwrap();

        println!("e={e} loss={loss}");

        let d_loss = 2. * (&y_pred - &y);
        conv.backward(d_loss);
        optim.step(&mut conv);
        conv.zero_grads();
    }

    let y_pred = conv.forward(x.clone(), false);
    let loss = (&y_pred - &y).pow2().mean().unwrap();

    let max_loss = 0.1;

    assert!(
        loss < max_loss,
        "patchwise conv2d failed to train shift, loss too high actual={loss} max={max_loss}"
    );
}

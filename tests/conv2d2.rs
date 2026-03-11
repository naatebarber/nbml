use nbml::{
    Tensor,
    f2::he,
    nn2::conv2d::{Conv2D, PatchwiseConv2D},
    optim2::{Optimizer, ToParams, adam::AdamW},
    s,
};

#[test]
fn im2col_conv2d_produces_correct_shapes() {
    let mut conv = Conv2D::new(3, 6, 5, 5, he);
    let x = Tensor::ones((5, 3, 38, 38));

    let x_conv = conv.forward(x.clone(), true);
    assert!(
        x_conv.shape() == [5, 6, 34, 34],
        "im2col conv2d produces incorrect shapes on forward, expected={:?} actual={:?}",
        [5, 6, 34, 34],
        x_conv.shape()
    );

    let d_conv = conv.backward(x_conv);
    assert!(
        x.shape() == d_conv.shape(),
        "im2col conv2d produces incorrect shapes on backward, expected={:?} actual={:?}",
        x.shape(),
        d_conv.shape()
    );
}

#[test]
fn patchwise_conv2d_produces_correct_shapes() {
    let mut conv = PatchwiseConv2D::new(3, 6, 5, 5, he);
    let x = Tensor::ones((5, 3, 38, 38));

    let x_conv = conv.forward(x.clone(), true);
    assert!(
        x_conv.shape() == [5, 6, 34, 34],
        "patchwise conv2d produces incorrect shapes on forward, expected={:?} actual={:?}",
        [5, 6, 34, 34],
        x_conv.shape()
    );

    let d_conv = conv.backward(x_conv);
    assert!(
        x.shape() == d_conv.shape(),
        "patchwise conv2d produces incorrect shapes on backward, expected={:?} actual={:?}",
        x.shape(),
        d_conv.shape()
    );
}

fn inversion_dataset() -> (Tensor, Tensor) {
    let x = (Tensor::random_uniform((10, 1, 10, 10))).mapv(|v| v.round());
    let y = x.mapv(|v| (v - 1.).abs());

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
        let loss = (&y_pred - &y).powi(2).mean();

        println!("e={e} loss={loss}");

        let d_loss = (&y_pred - &y) * 2.0;
        conv.backward(d_loss);
        optim.step(&mut conv);
        conv.zero_grads();
    }

    let y_pred = conv.forward(x.clone(), false);
    let loss = (&y_pred - &y).powi(2).mean();

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
        let loss = (&y_pred - &y).powi(2).mean();

        println!("e={e} loss={loss}");

        let d_loss = (&y_pred - &y) * 2.0;
        conv.backward(d_loss);
        optim.step(&mut conv);
        conv.zero_grads();
    }

    let mut losses = vec![];

    for e in 0..100 {
        let y_pred = conv.forward(x.clone(), false);
        let loss = (&y_pred - &y).powi(2).mean();

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

fn shift_dataset() -> (Tensor, Tensor) {
    let mut batch = Tensor::zeros((10, 1, 11, 11));
    for i in 0..10usize {
        let col = Tensor::ones((1, 1, 11, 1));
        batch.slice_assign(
            s![i as isize, .., .., i as isize..(i + 1) as isize],
            &col.slice(s![0, .., .., ..]),
        );
    }

    let x = batch.slice(s![0..9, .., .., ..]);
    let y = batch.slice(s![1.., .., 0..10, 0..10]);

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

        println!("y_pred {:?} y {:?}", y_pred.shape(), y.shape());
        let loss = (&y_pred - &y).powi(2).mean();

        println!("e={e} loss={loss}");

        let d_loss = (&y_pred - &y) * 2.0;
        conv.backward(d_loss);
        optim.step(&mut conv);
        conv.zero_grads();
    }

    let y_pred = conv.forward(x.clone(), false);
    let loss = (&y_pred - &y).powi(2).mean();

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

        println!("y_pred {:?} y {:?}", y_pred.shape(), y.shape());
        let loss = (&y_pred - &y).powi(2).mean();

        println!("e={e} loss={loss}");

        let d_loss = (&y_pred - &y) * 2.0;
        conv.backward(d_loss);
        optim.step(&mut conv);
        conv.zero_grads();
    }

    let y_pred = conv.forward(x.clone(), false);
    let loss = (&y_pred - &y).powi(2).mean();

    let max_loss = 0.1;

    assert!(
        loss < max_loss,
        "patchwise conv2d failed to train shift, loss too high actual={loss} max={max_loss}"
    );
}

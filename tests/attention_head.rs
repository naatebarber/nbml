use nbml::{
    nn::AttentionHead,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::random_range;

fn model_optimizer(
    d_model: usize,
    d_head: usize,
    n_head: usize,
) -> (AttentionHead, impl Optimizer) {
    let mut ah = AttentionHead::new(d_model, d_head, n_head);
    let mut optim = AdamW::default().with(&mut ah);
    optim.learning_rate = 1e-3;

    (ah, optim)
}

#[test]
fn encoder_vs_decoder() {
    let (mut attn, _optim) = model_optimizer(6, 2, 3);
    let input = Array3::random((1, 6, 6), Uniform::new(0., 1.));
    let mask = Array2::ones((1, 6));
    let y_encoder = attn.forward(&input, &mask, false, false);
    let y_decoder = attn.forward(&input, &mask, true, false);

    assert_ne!(
        y_encoder.slice(s![0, 0..5, ..]),
        y_decoder.slice(s![0, 0..5, ..])
    );
    assert_eq!(y_encoder.slice(s![0, 5, ..]), y_decoder.slice(s![0, 5, ..]));
}

#[test]
fn decoder_with_padding() {
    let seq_len = 6;
    let d_model = 6;
    let d_head = 2;
    let n_head = 3;

    let (mut attn, _optim) = model_optimizer(d_model, d_head, n_head);

    let input = Array3::random((1, 6, 6), Uniform::new(0., 1.));
    let mut mask = Array2::ones((1, 6));
    mask.slice_mut(s![0, 4..]).assign(&Array1::zeros(2));

    attn.forward(&input, &mask, true, true);

    println!("attn scores: {:?}", attn.scores);

    for h in 0..n_head {
        let head_scores = attn.scores.slice(s![h, .., ..]);

        for i in 0..seq_len {
            let score_slice = head_scores.slice(s![i, (i + 1)..]).to_owned();
            assert_eq!(score_slice.sum(), 0.);
        }
    }
}

#[test]
fn encoder_selector() {
    let (mut attn, mut optim) = model_optimizer(10, 2, 5);

    let make_xy = || {
        let mut x = Array2::random((4, 10), Uniform::new(0., 1.));
        let selector = random_range(1..4);
        x.slice_mut(s![0, ..])
            .assign(&Array1::from_elem(10, selector as f64));

        let y = x
            .row(selector as usize)
            .broadcast((4, 10))
            .unwrap()
            .to_owned();

        (x.insert_axis(Axis(0)), y.insert_axis(Axis(0)))
    };

    for e in 0..1000 {
        let (x, y) = make_xy();

        let mask = Array2::ones((1, 4));

        let y_pred = attn.forward(&x, &mask, false, true);
        let loss = (&y_pred - &y).mean().unwrap().powi(2);
        let d_loss = 2. * &y_pred - &y;
        attn.backward(d_loss);
        optim.step(&mut attn);
        attn.zero_grads();

        println!("epoch {e} loss {loss}");
    }

    let mut test_losses = vec![];

    for _ in 0..100 {
        let (x, y) = make_xy();
        let mask = Array2::ones((1, 4));
        let y_pred = attn.forward(&x, &mask, false, true);

        let mse = (&y_pred - &y).mean().unwrap().powi(2);

        test_losses.push(mse);
    }

    let avg_test_mse = test_losses.iter().sum::<f64>() / test_losses.len() as f64;
    println!("avg test mse: {avg_test_mse}");

    if avg_test_mse < 0.1 {
        assert!(true);
    } else {
        assert!(false);
    }
}

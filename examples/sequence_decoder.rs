use nbml::{
    f::Activation,
    nn::TransformerDecoder,
    optim::{adam::AdamW, optimizer::Optimizer},
};
use ndarray::{Array1, Array2, Axis};
use rand::random_range;

fn main() {
    let mut t = TransformerDecoder::new(
        1,
        1,
        1,
        vec![(1, 24, Activation::Relu), (24, 1, Activation::Identity)],
    );

    let mut optim = AdamW::default().with(&mut t);

    let make_xy = || {
        let start = random_range(0..10);
        let x = (start..(start + 6))
            .map(|x| x as f64)
            .collect::<Array1<_>>();
        let y = &x + 1.;

        (
            x.insert_axis(Axis(1)).insert_axis(Axis(0)),
            y.insert_axis(Axis(1)).insert_axis(Axis(0)),
        )
    };

    for e in 0..1000 {
        let (x, y) = make_xy();

        let mask = Array2::ones((x.dim().0, x.dim().1));
        let y_pred = t.forward(x, mask, true);
        let loss = (&y_pred - &y).mean().unwrap().powi(2);
        let d_loss = 2. * (&y_pred - &y);

        t.backward(d_loss);
        optim.step(&mut t);

        println!("epoch {} loss {}", e, loss);
    }

    let mut s = vec![1.];
    for _ in 0..100 {
        let x = Array1::from_vec(s.clone())
            .insert_axis(Axis(1))
            .insert_axis(Axis(0));
        let mask = Array2::ones((x.dim().0, x.dim().1));
        let y_pred = t.forward(x, mask, false);

        s.push(y_pred[[0, s.len() - 1, 0]]);
    }

    println!("{:?}", s)
}

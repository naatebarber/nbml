use nbml::{
    nn::experimental::linear_self_attention::LinearSelfAttention,
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array1, Array3, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn linear_self_attention_learns_to_attend_to_highest() {
    fn make_dataset() -> (Array3<f64>, Array3<f64>) {
        let mut x = Array3::random((1, 5, 5), Uniform::new(0., 1.));
        let high = 50.;
        x.slice_mut(s![0, 0, ..])
            .assign(&Array1::from_elem(5, high));

        let y = Array3::from_elem((1, 5, 5), high);

        (x, y)
    }

    let d_in = 5;
    let d_head = 5;
    let mut attn = LinearSelfAttention::new(d_in, d_head);
    let mut optim = AdamW::default().with(&mut attn);

    let mut final_loss = f64::MAX;

    for epoch in 0..1000 {
        let (x, y) = make_dataset();

        let y_pred = attn.forward(x, true);
        let d_loss = 2. * (&y_pred - &y);
        let loss = (&y_pred - &y).pow2().mean().unwrap();

        attn.backward(d_loss);
        optim.step(&mut attn);
        attn.zero_grads();

        if epoch % 100 == 0 {
            println!("epoch {} loss {}", epoch, loss);
        }

        final_loss = loss;
    }

    assert!(
        final_loss < 0.1,
        "self attention did not learn to attend to highest value: desired=0.1, final_loss={final_loss}"
    );
}

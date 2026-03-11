use nbml::{
    Tensor, f2,
    nn2::GatedLinearAttention,
    optim2::{Optimizer, ToParams, adam::AdamW},
    s,
    tensor::Tensor3,
};
use rand::{rng, seq::IteratorRandom};

#[test]
fn gla_self_attention_gradient_check() {
    let d_in = 4;
    let d_head = 2;
    let n_head = 2;
    let batch_size = 3;
    let seq_len = 3;
    let eps = 1e-5;

    let mut attn = GatedLinearAttention::new(d_in, d_head, n_head);
    let x = f2::xavier_normal((batch_size * seq_len, d_in)).reshape((batch_size, seq_len, d_in));

    // forward + backward
    let out = attn.forward(x.clone(), true);
    let d_loss = Tensor::ones(out.shape());
    let d_x = attn.backward(d_loss.clone());

    // numerical gradient for each element of x
    for b in 0..batch_size {
        for s in 0..seq_len {
            for f in 0..d_in {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[[b, s, f]] += eps;
                x_minus[[b, s, f]] -= eps;

                let mut attn_plus = attn.clone();
                let mut attn_minus = attn.clone();
                let out_plus = attn_plus.forward(x_plus, false);
                let out_minus = attn_minus.forward(x_minus, false);

                let numerical = (&out_plus - &out_minus).sum() / (2.0 * eps);
                let analytical = d_x[[b, s, f]];

                let diff = (numerical - analytical).abs();
                assert!(
                    diff < 1e-4,
                    "gradient mismatch at [{},{},{}]: numerical={}, analytical={}, diff={}",
                    b,
                    s,
                    f,
                    numerical,
                    analytical,
                    diff
                );
            }
        }
    }
}

fn make_associative_recall_dataset(batch_size: usize, d_model: usize) -> (Tensor3, Tensor3) {
    let num_pairs = 3;
    let seq_len = num_pairs * 2 + 2;

    let mut x = Tensor3::zeros((batch_size, seq_len, d_model));
    let mut y = Tensor3::zeros((batch_size, seq_len, d_model));

    for b in 0..batch_size {
        let keys: Vec<Tensor> = (0..num_pairs)
            .map(|_| Tensor::random_uniform(d_model) * 10.0)
            .collect();
        let values: Vec<Tensor> = (0..num_pairs)
            .map(|_| Tensor::random_uniform(d_model) * 10.0)
            .collect();

        for i in 0..num_pairs {
            x.slice_assign(s![b as isize, (i * 2) as isize, ..], &keys[i]);
            x.slice_assign(s![b as isize, (i * 2 + 1) as isize, ..], &values[i]);
        }

        x.slice_assign(
            s![b as isize, (num_pairs * 2) as isize, ..],
            &Tensor::from_elem(d_model, 1.0),
        );

        let query_idx = (0..num_pairs).choose(&mut rng()).unwrap();
        x.slice_assign(
            s![b as isize, (num_pairs * 2 + 1) as isize, ..],
            &keys[query_idx],
        );

        y.slice_assign(
            s![b as isize, (num_pairs * 2 + 1) as isize, ..],
            &values[query_idx],
        );
    }

    (x, y)
}

#[test]
fn gla_associative_recall() {
    let d_in = 5;
    let d_head = 5;
    let n_head = 4;
    let batch_size = 10;

    let mut attn = GatedLinearAttention::new(d_in, d_head, n_head);
    let mut optim = AdamW::default().with(&mut attn);
    optim.learning_rate = 1e-3;

    let mut final_loss = f64::MAX;

    let (x, y) = make_associative_recall_dataset(batch_size, d_in);

    for epoch in 0..1000 {
        let y_pred = attn.forward(x.clone(), true);

        let mut y_pred_mask = Tensor3::zeros(y.shape());
        y_pred_mask.slice_assign(s![.., -1, ..], &y_pred.slice(s![.., -1, ..]));

        let d_loss = (&y_pred_mask - &y) * 2.0;
        let loss = (&y_pred_mask - &y).powi(2).mean();

        attn.backward(d_loss);
        optim.step(&mut attn);
        attn.zero_grads();

        if epoch % 100 == 0 {
            println!("epoch {} loss {}", epoch, loss);
        }

        final_loss = loss;
    }

    let max_loss = 0.1;
    assert!(
        final_loss < max_loss,
        "gated linear attention failed associative recall with loss {final_loss} > {max_loss}"
    );
}

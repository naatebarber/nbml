use nbml::nn::{
    DeltaNetTransformer, GdnTransformer, GlaTransformer, LinearAttention, LinearAttentionLike,
    LinearTransformer, LinearTransformerBase,
};
use ndarray::{Array3, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

fn assert_forward_matches_step<A: LinearAttentionLike>(
    name: &str,
    mut make: impl FnMut(usize, usize) -> LinearTransformerBase<A>,
) {
    for (batch_size, d_in, d_head) in [(4, 16, 8), (8, 16, 8), (1, 16, 32)] {
        let seq_len = 12;
        let x = Array3::random((batch_size, seq_len, d_in), Uniform::new(-1., 1.).unwrap());

        let mut model = make(d_in, d_head);
        let y_forward = model.forward(x.clone(), false);

        let mut state = model.new_state(batch_size);
        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]).to_owned();
            let y_step = model.step(&x_t, &mut state);

            let y_f = y_forward.slice(s![.., t, ..]);
            let scale = y_f.fold(1f32, |a, &v| a.max(v.abs()));
            let diff = (&y_f - &y_step).abs().fold(0f32, |a, &v| a.max(v));
            assert!(
                diff < 1e-4 * scale,
                "{name}: forward and step diverge at t={t} (B={batch_size}, d_in={d_in}, d_head={d_head}): max diff {diff}"
            );
        }
    }
}

#[test]
fn linear_transformer_forward_and_step_match() {
    assert_forward_matches_step("LinearTransformer", LinearTransformer::new);
}

#[test]
fn katharopoulos_transformer_forward_and_step_match() {
    assert_forward_matches_step("Katharopoulos", |d_in, d_head| {
        LinearTransformer::from_attn(d_in, LinearAttention::katharopoulos(d_in, d_head))
    });
}

#[test]
fn gla_transformer_forward_and_step_match() {
    assert_forward_matches_step("GlaTransformer", GlaTransformer::new);
}

#[test]
fn delta_net_transformer_forward_and_step_match() {
    assert_forward_matches_step("DeltaNetTransformer", DeltaNetTransformer::new);
}

#[test]
fn gdn_transformer_forward_and_step_match() {
    assert_forward_matches_step("GdnTransformer", GdnTransformer::new);
}

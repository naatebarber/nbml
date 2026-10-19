use ndarray::Array2;

/// Creates a 2d "blanket" positional encoding of size (S, D).
pub fn sinusoidal_positional_encoding(seq_len: usize, features: usize) -> Array2<f32> {
    let mut pe = Array2::zeros((seq_len, features));
    let n: f32 = 10_000.;

    for k in 0..seq_len {
        for i in 0..(features / 2) {
            let exp = n.powf((2.0 * i as f32) / features as f32);
            pe[[k, 2 * i]] = f32::sin((k as f32) / exp);
            if (2 * i + 1) < features {
                pe[[k, 2 * i + 1]] = f32::cos((k as f32) / exp);
            }
        }
    }

    pe
}

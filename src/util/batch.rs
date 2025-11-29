use ndarray::{Array2, Array3};

pub fn batch_with_mask(x: Vec<Array2<f64>>) -> (Array3<f64>, Array2<f64>) {
    let batch_size = x.len();
    let max_seq_len = x.iter().map(|x| x.dim().0).max().unwrap_or(0);
    let features = x.first().map(|x| x.dim().1).unwrap_or(0);

    let mut x3 = Array3::zeros((batch_size, max_seq_len, features));
    let mut mask = Array2::zeros((batch_size, max_seq_len));

    for (i, row) in x.iter().enumerate() {
        x3.slice_mut(ndarray::s![i, 0..row.dim().0, ..]).assign(row);
        mask.slice_mut(ndarray::s![i, 0..row.dim().0]).fill(1.0);
    }

    (x3, mask)
}

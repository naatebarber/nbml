use ndarray::{Array2, Array3, Axis};

pub struct PadMask {}

impl PadMask {
    pub fn zero_mask_batch(x: &Array3<f64>) -> Array2<f64> {
        x.sum_axis(Axis(2)).mapv(|x| if x == 0. { 0. } else { 1. }) // (B, S)
    }
}

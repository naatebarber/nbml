use nbml::f;
use ndarray::{Array3, s};

fn main() {
    let x = Array3::zeros((5, 5, 5));
    let x_t = x.slice(s![.., 0, ..]);
    f::relu(&x_t);
}

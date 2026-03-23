use ndarray::{Array1, Array2};

fn main() {
    let x = Array2::<f32>::zeros((5, 5));
    let y = Array1::<f32>::ones(5);

    println!("{:?}", &x + &y);
}

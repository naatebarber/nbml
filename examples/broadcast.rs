use ndarray::{Array1, Array2};

fn main() {
    let x = Array2::<f64>::zeros((5, 5));
    let y = Array1::<f64>::ones(5);

    println!("{:?}", &x + &y);
}

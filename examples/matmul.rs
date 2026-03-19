use std::time::Instant;

use nbml::ndarray::Array2;

fn main() {
    let start = Instant::now();

    let a: Array2<f64> = Array2::ones((4000, 4000));
    let b = Array2::ones((4000, 4000));

    a.dot(&b);

    let end = Instant::now();

    println!("matmul took {}s", end.duration_since(start).as_secs_f32());
}

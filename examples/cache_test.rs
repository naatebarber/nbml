use nbml::optim::cache::Cache;
use ndarray::{Array1, Array2};

fn main() {
    let mut cache = Cache::new();

    cache.set::<Array1<f64>>("d_b", Array1::zeros(2));
    println!("{:?}", cache.take::<Array1<f64>>("d_b"));

    let d_w = cache.init::<Array2<f64>>("d_w", Array2::zeros((1, 1)));

    *d_w += 5.;

    println!("{:?}", cache.get_mut::<Array2<f64>>("d_w"));
}

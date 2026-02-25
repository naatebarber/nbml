use nbml::s;
use nbml::tensor::Tensor;

fn main() {
    let a = Tensor::random_normal(5);
    let mut b = Tensor::zeros((5, 5));

    b.slice_assign(s![0, ..], &a);

    println!("{b:?}");

    println!("{:?}", b[[0, 0]])
}

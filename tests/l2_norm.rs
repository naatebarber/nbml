use nbml::layers::L2Norm;
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn l2_norm_backward_numerical_gradient_check() {
    let eps = 1e-3;
    let (batch, features) = (3, 5);

    let mut l2_norm = L2Norm::new();
    let x = Array2::random((batch, features), Uniform::new(-2., 2.));

    let _out = l2_norm.forward(x.clone(), true);

    // upstream gradient
    let g = Array2::random((batch, features), Uniform::new(-1., 1.));
    let analytic = l2_norm.backward(g.clone());

    for i in 0..batch {
        for j in 0..features {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[[i, j]] += eps;
            x_minus[[i, j]] -= eps;

            let mut l2 = L2Norm::new();
            let out_plus = l2.forward(x_plus, false);
            let mut l2 = L2Norm::new();
            let out_minus = l2.forward(x_minus, false);

            let numerical = ((&out_plus - &out_minus) * &g).sum() / (2.0 * eps);
            let analytical = analytic[[i, j]];

            let diff = (numerical - analytical).abs();
            assert!(
                diff < 1e-4,
                "gradient mismatch at [{},{}]: numerical={}, analytical={}, diff={}",
                i,
                j,
                numerical,
                analytical,
                diff
            );
        }
    }
}

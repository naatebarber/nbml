use nbml::layers::Softmax;
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn softmax_rows_sum_to_one() {
    let mut softmax = Softmax::new();
    let x = Array2::random((4, 8), Uniform::new(-3., 3.));
    let out = softmax.forward(x, false);

    for row in out.rows() {
        let sum: f32 = row.sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax row sum {sum} != 1.0");
    }
}

#[test]
fn softmax_outputs_positive() {
    let mut softmax = Softmax::new();
    let x = Array2::random((4, 8), Uniform::new(-10., 10.));
    let out = softmax.forward(x, false);

    assert!(
        out.iter().all(|&v| v > 0.0),
        "softmax produced non-positive values"
    );
}

#[test]
fn softmax_numerically_stable_with_large_values() {
    let mut softmax = Softmax::new();
    let mut x = Array2::random((2, 5), Uniform::new(0., 1.));
    x[[0, 0]] = 1000.0;
    x[[1, 2]] = -1000.0;

    let out = softmax.forward(x, false);

    assert!(
        out.iter().all(|v| v.is_finite()),
        "softmax produced NaN/Inf with extreme values"
    );

    for row in out.rows() {
        let sum: f32 = row.sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax row sum {sum} != 1.0 with extreme values"
        );
    }
}

#[test]
fn softmax_backward_numerical_gradient_check() {
    let eps = 1e-3;
    let (batch, features) = (3, 5);

    let mut softmax = Softmax::new();
    let x = Array2::random((batch, features), Uniform::new(-2., 2.));

    let _out = softmax.forward(x.clone(), true);

    // upstream gradient
    let g = Array2::random((batch, features), Uniform::new(-1., 1.));
    let analytic = softmax.backward(g.clone());

    // numerical: d/dx_ij of sum(g * softmax(x))
    let mut numerical = Array2::zeros((batch, features));
    for i in 0..batch {
        for j in 0..features {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[[i, j]] += eps;
            x_minus[[i, j]] -= eps;

            let mut sm = Softmax::new();
            let out_plus = sm.forward(x_plus, false);
            let mut sm = Softmax::new();
            let out_minus = sm.forward(x_minus, false);

            // f = sum(g * softmax(x)), so df/dx_ij via finite diff
            let f_plus: f32 = (&g * &out_plus).sum();
            let f_minus: f32 = (&g * &out_minus).sum();
            numerical[[i, j]] = (f_plus - f_minus) / (2.0 * eps);
        }
    }

    let max_err = (&analytic - &numerical)
        .mapv(f32::abs)
        .into_iter()
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "softmax backward gradient check failed, max error {max_err}"
    );
}

#[test]
fn softmax_uniform_input_gives_uniform_output() {
    let mut softmax = Softmax::new();
    let x = Array2::from_elem((2, 6), 5.0);
    let out = softmax.forward(x, false);

    let expected = 1.0 / 6.0;
    for &v in out.iter() {
        assert!(
            (v - expected).abs() < 1e-10,
            "uniform input should give uniform output, got {v} expected {expected}"
        );
    }
}

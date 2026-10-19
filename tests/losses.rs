use nbml::f;
use ndarray::array;

#[test]
fn focal_reduces_to_half_bce() {
    let x = array![[-3.0f32, -0.5, 0.0, 2.0, 25.0]];
    let y = array![[0.0f32, 1.0, 1.0, 0.0, 1.0]];
    let (lb, gb) = f::bce_with_logits(&x, &y);
    let (lf, gf) = f::focal_with_logits(&x, &y, 0.5, 0.0);
    assert!((lf - 0.5 * lb).abs() < 1e-6);
    for (a, b) in gf.iter().zip(gb.iter()) {
        assert!((a - 0.5 * b).abs() < 1e-6);
    }
}

#[test]
fn focal_grad_check() {
    let x = array![[-3.0f32, -0.5, 0.3, 2.0, 4.0]];
    let y = array![[0.0f32, 1.0, 1.0, 0.0, 1.0]];
    let (_, g) = f::focal_with_logits(&x, &y, 0.25, 2.0);
    let eps = 1e-3;
    for j in 0..x.ncols() {
        let mut xp = x.clone();
        xp[[0, j]] += eps;
        let mut xm = x.clone();
        xm[[0, j]] -= eps;
        let num = (f::focal_with_logits(&xp, &y, 0.25, 2.0).0
            - f::focal_with_logits(&xm, &y, 0.25, 2.0).0)
            / (2.0 * eps);
        assert!(
            (num - g[[0, j]]).abs() < 1e-3 + 1e-2 * num.abs(),
            "x[{j}]: {num} vs {}",
            g[[0, j]]
        );
    }
}

use nbml::nn::GATv2;
use ndarray::{Array2, array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::Normal;

const EPS: f32 = 5e-3;

/// 4 nodes, self-loops on every node, plus a few cross edges. Rows are (src, dst).
fn setup() -> (GATv2, Array2<f32>, Array2<usize>, Array2<f32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let (n, d_in, d_head, n_head, d_out) = (4, 3, 2, 2, 3);
    let layer = GATv2::new(d_in, d_head, n_head, d_out);
    let x = Array2::random_using((n, d_in), Normal::new(0f32, 1f32).unwrap(), &mut rng);
    let index = array![
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [1, 0],
        [2, 0],
        [3, 0],
        [0, 1],
        [2, 1],
        [0, 3]
    ];
    // L = sum(out * w), so dL/d_out = w
    let w = Array2::random_using((n, d_out), Normal::new(0f32, 1f32).unwrap(), &mut rng);
    (layer, x, index, w)
}

fn loss(layer: &mut GATv2, x: &Array2<f32>, index: &Array2<usize>, w: &Array2<f32>) -> f32 {
    let out = layer.forward(x.clone(), index, false);
    (&out * w).sum()
}

fn assert_close(name: &str, analytic: f32, numeric: f32) {
    let err = (analytic - numeric).abs();
    let tol = 5e-3 + 5e-2 * analytic.abs().max(numeric.abs());
    assert!(
        err <= tol,
        "{name}: analytic {analytic:.6} vs numeric {numeric:.6} (err {err:.2e})"
    );
}

#[test]
fn alpha_rows_sum_to_one() {
    let (mut layer, x, index, _) = setup();
    layer.forward(x, &index, true);
    let recv = index.column(1).to_vec();
    let sums = GATv2::scatter_sum(&layer.cache.alpha, &recv, 4);
    for v in sums.iter() {
        assert!((v - 1.0).abs() < 1e-5, "row sum {v}");
    }
}

#[test]
fn grad_check_input() {
    let (mut layer, x, index, w) = setup();
    layer.forward(x.clone(), &index, true);
    let d_x = layer.backward(w.clone(), &index);

    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let mut xp = x.clone();
            xp[[i, j]] += EPS;
            let mut xm = x.clone();
            xm[[i, j]] -= EPS;
            let numeric = (loss(&mut layer, &xp, &index, &w) - loss(&mut layer, &xm, &index, &w))
                / (2.0 * EPS);
            assert_close(&format!("d_x[{i},{j}]"), d_x[[i, j]], numeric);
        }
    }
}

#[test]
fn grad_check_a() {
    let (mut layer, x, index, w) = setup();
    layer.forward(x.clone(), &index, true);
    layer.backward(w.clone(), &index);
    let d_a = layer.grads.d_a.clone();

    for h in 0..layer.n_head {
        for k in 0..layer.d_head {
            let orig = layer.a[[h, k]];
            layer.a[[h, k]] = orig + EPS;
            let lp = loss(&mut layer, &x, &index, &w);
            layer.a[[h, k]] = orig - EPS;
            let lm = loss(&mut layer, &x, &index, &w);
            layer.a[[h, k]] = orig;
            assert_close(
                &format!("d_a[{h},{k}]"),
                d_a[[h, k]],
                (lp - lm) / (2.0 * EPS),
            );
        }
    }
}

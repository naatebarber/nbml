use nbml::nn::CfC;
use nbml::optim::{AdamW, Optimizer, ToParams};
use ndarray::{Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::{Normal, Uniform};

const EPS: f32 = 5e-3;

/// batch 2, seq 4, d_in 3, backbone 5, d_model 4. dt is strictly positive and varied
/// so the time-constant path (g, tau_exp) actually contributes to the gradient.
fn setup() -> (CfC, Array3<f32>, Array3<f32>, Array3<f32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let (b, s, d_in, d_backbone, d_model) = (2, 4, 3, 5, 4);
    let mut layer = CfC::new(d_in, d_backbone, d_model);
    // non-zero biases so their gradients aren't trivially checked at the origin
    layer.zx_b =
        ndarray::Array1::random_using(d_backbone, Normal::new(0f32, 0.5).unwrap(), &mut rng);
    layer.fgc_b =
        ndarray::Array1::random_using(3 * d_model, Normal::new(0f32, 0.5).unwrap(), &mut rng);

    let x = Array3::random_using((b, s, d_in), Normal::new(0f32, 1f32).unwrap(), &mut rng);
    let dt = Array3::random_using((b, s, 1), Uniform::new(0.1f32, 1.5).unwrap(), &mut rng);
    // L = sum(out * w), so dL/d_out = w
    let w = Array3::random_using((b, s, d_model), Normal::new(0f32, 1f32).unwrap(), &mut rng);
    (layer, x, dt, w)
}

fn loss(layer: &mut CfC, x: &Array3<f32>, dt: &Array3<f32>, w: &Array3<f32>) -> f32 {
    let out = layer.forward(x.clone(), dt.clone(), false);
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

/// Finite-difference check for one parameter tensor. `param` returns the parameter as a
/// flat mutable slice, `grad` returns the matching analytic gradient.
fn check_param(
    name: &str,
    param: impl Fn(&mut CfC) -> &mut [f32],
    grad: impl Fn(&CfC) -> Vec<f32>,
) {
    let (mut layer, x, dt, w) = setup();
    layer.forward(x.clone(), dt.clone(), true);
    layer.backward(w.clone());
    let analytic = grad(&layer);
    assert_eq!(
        analytic.len(),
        param(&mut layer).len(),
        "{name}: grad shape mismatch"
    );

    for i in 0..analytic.len() {
        let orig = param(&mut layer)[i];
        param(&mut layer)[i] = orig + EPS;
        let lp = loss(&mut layer, &x, &dt, &w);
        param(&mut layer)[i] = orig - EPS;
        let lm = loss(&mut layer, &x, &dt, &w);
        param(&mut layer)[i] = orig;
        assert_close(
            &format!("{name}[{i}]"),
            analytic[i],
            (lp - lm) / (2.0 * EPS),
        );
    }
}

#[test]
fn grad_check_input() {
    let (mut layer, x, dt, w) = setup();
    layer.forward(x.clone(), dt.clone(), true);
    let d_x = layer.backward(w.clone());
    assert_eq!(d_x.dim(), x.dim());

    for idx in ndarray::indices(x.dim()) {
        let mut xp = x.clone();
        xp[idx] += EPS;
        let mut xm = x.clone();
        xm[idx] -= EPS;
        let numeric =
            (loss(&mut layer, &xp, &dt, &w) - loss(&mut layer, &xm, &dt, &w)) / (2.0 * EPS);
        assert_close(&format!("d_x{idx:?}"), d_x[idx], numeric);
    }
}

#[test]
fn grad_check_zx_w() {
    check_param(
        "d_zx_w",
        |l| l.zx_w.as_slice_mut().unwrap(),
        |l| l.grads.d_zx_w.iter().copied().collect(),
    );
}

#[test]
fn grad_check_zx_b() {
    check_param(
        "d_zx_b",
        |l| l.zx_b.as_slice_mut().unwrap(),
        |l| l.grads.d_zx_b.iter().copied().collect(),
    );
}

#[test]
fn grad_check_zh_w() {
    check_param(
        "d_zh_w",
        |l| l.zh_w.as_slice_mut().unwrap(),
        |l| l.grads.d_zh_w.iter().copied().collect(),
    );
}

#[test]
fn grad_check_fgc_w() {
    check_param(
        "d_fgc_w",
        |l| l.fgc_w.as_slice_mut().unwrap(),
        |l| l.grads.d_fgc_w.iter().copied().collect(),
    );
}

#[test]
fn grad_check_fgc_b() {
    check_param(
        "d_fgc_b",
        |l| l.fgc_b.as_slice_mut().unwrap(),
        |l| l.grads.d_fgc_b.iter().copied().collect(),
    );
}

#[test]
fn output_shape() {
    let (mut layer, x, dt, _) = setup();
    let out = layer.forward(x.clone(), dt, false);
    assert_eq!(out.dim(), (x.dim().0, x.dim().1, layer.d_model));
}

/// With dt = 0, exp(-g * dt) = 1, so h_t collapses to the f head exactly.
#[test]
fn zero_dt_selects_f_head() {
    let (mut layer, x, mut dt, _) = setup();
    dt.fill(0.);
    let out = layer.forward(x, dt, true);
    let diff = (&out - &layer.cache.f).abs().fold(0f32, |a, &b| a.max(b));
    assert!(diff < 1e-6, "dt=0 output deviates from f head by {diff}");
}

/// With very large dt, exp(-g * dt) -> 0, so h_t converges to the c head.
#[test]
fn large_dt_selects_c_head() {
    let (mut layer, x, mut dt, _) = setup();
    dt.fill(1e4);
    let out = layer.forward(x, dt, true);
    let diff = (&out - &layer.cache.c).abs().fold(0f32, |a, &b| a.max(b));
    assert!(
        diff < 1e-4,
        "large dt output deviates from c head by {diff}"
    );
}

/// Outputs at step t must not depend on inputs after t.
#[test]
fn causal() {
    let (mut layer, x, dt, _) = setup();
    let out = layer.forward(x.clone(), dt.clone(), false);

    let mut x2 = x.clone();
    x2.slice_mut(s![.., 2.., ..]).fill(10.);
    let out2 = layer.forward(x2, dt, false);

    assert_eq!(
        out.slice(s![.., ..2, ..]),
        out2.slice(s![.., ..2, ..]),
        "future inputs leaked into earlier outputs"
    );
    assert_ne!(out.slice(s![.., 2.., ..]), out2.slice(s![.., 2.., ..]));
}

/// Stepping one timestep at a time should reproduce the full-sequence forward scan.
#[test]
fn forward_and_step_compute_same_value() {
    let (mut layer, x, dt, _) = setup();
    let out = layer.forward(x.clone(), dt.clone(), false);

    let (b, s, _) = x.dim();
    let mut h = Array2::zeros((b, layer.d_model));
    for t in 0..s {
        let x_t = x.slice(s![.., t, ..]).to_owned();
        let dt_t = dt.slice(s![.., t, ..]).to_owned();
        layer.step(&x_t, &dt_t, &mut h);

        let diff = (&out.slice(s![.., t, ..]) - &h)
            .abs()
            .fold(0f32, |a, &v| a.max(v));
        assert!(
            diff < 1e-5,
            "step and forward diverge at t={t} (max diff {diff})"
        );
    }
}

/// backward should accumulate into grads (consistent with other modules + optim zero_grads).
#[test]
fn grads_accumulate() {
    let (mut layer, x, dt, w) = setup();
    layer.forward(x.clone(), dt.clone(), true);
    let d_x1 = layer.backward(w.clone());
    let g1 = layer.grads.clone();
    let d_x2 = layer.backward(w.clone());

    assert_eq!(
        d_x1, d_x2,
        "input grad changed between identical backward calls"
    );
    let check = |name: &str, a: f32, b: f32| {
        assert!(
            (2. * a - b).abs() < 1e-4,
            "{name}: expected {} got {b}",
            2. * a
        )
    };
    for (a, b) in g1.d_zx_w.iter().zip(layer.grads.d_zx_w.iter()) {
        check("d_zx_w", *a, *b);
    }
    for (a, b) in g1.d_zh_w.iter().zip(layer.grads.d_zh_w.iter()) {
        check("d_zh_w", *a, *b);
    }
    for (a, b) in g1.d_fgc_w.iter().zip(layer.grads.d_fgc_w.iter()) {
        check("d_fgc_w", *a, *b);
    }
}

/// Learn the running mean of the inputs over an irregularly-sampled sequence. The output
/// is the hidden state itself, so the state has to carry the mean forward.
#[test]
fn overfitting() {
    let mut rng = StdRng::seed_from_u64(1);
    let (b, s, d) = (16, 8, 2);

    let x = Array3::random_using((b, s, d), Uniform::new(-1f32, 1.).unwrap(), &mut rng);
    let dt = Array3::random_using((b, s, 1), Uniform::new(0.5f32, 1.5).unwrap(), &mut rng);
    let mut y = x.clone();
    y.accumulate_axis_inplace(Axis(1), |&prev, cur| *cur += prev);
    for t in 0..s {
        y.slice_mut(s![.., t, ..])
            .mapv_inplace(|v| v / (t + 1) as f32);
    }

    let mut model = CfC::new(d, 16, d);
    let mut optim = AdamW::default().with(&mut model);
    optim.learning_rate = 1e-2;

    let mse = |model: &mut CfC| {
        let y_pred = model.forward(x.clone(), dt.clone(), false);
        (&y_pred - &y).powi(2).mean().unwrap()
    };

    let initial = mse(&mut model);
    for _ in 0..1000 {
        let y_pred = model.forward(x.clone(), dt.clone(), true);
        model.backward(2. * (&y_pred - &y));
        optim.step(&mut model);
        model.zero_grads();
    }
    let final_loss = mse(&mut model);

    println!("initial={initial} final={final_loss}");
    assert!(
        final_loss < initial * 0.25,
        "CfC didn't train: initial {initial} -> final {final_loss}"
    );
}

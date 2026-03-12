use std::time::Instant;

use nbml::{Tensor, s};
use ndarray::{ArcArrayD, ArrayD, Axis, Ix2};

const WARMUP: usize = 10;
const ITERS: usize = 500;

fn bench(name: &str, f_tensor: impl Fn(), f_ndarray: impl Fn()) {
    for _ in 0..WARMUP {
        f_tensor();
        f_ndarray();
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        f_tensor();
    }
    let t_tensor = start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..ITERS {
        f_ndarray();
    }
    let t_ndarray = start.elapsed().as_secs_f64();

    let ratio = t_tensor / t_ndarray;
    let marker = if ratio > 1.05 {
        "SLOWER"
    } else if ratio < 0.95 {
        "FASTER"
    } else {
        "  ~   "
    };

    println!(
        "{:<25} tensor={:.4}s  ndarray={:.4}s  ratio={:.2}x  [{}]",
        name, t_tensor, t_ndarray, ratio, marker
    );
}

fn main() {
    let b = 16;
    let s = 16;
    let d = 32;
    let h = 4;
    let dh = d / h;

    println!("Tensor vs ndarray microbenchmarks ({ITERS} iters each)\n");
    println!("Shape: batch={b} seq={s} d_model={d} n_head={h} d_head={dh}\n");

    // --- slice (rank-3, extract one timestep) ---
    {
        let t = Tensor::random_uniform((b, s, d));
        let a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s, d]);

        bench(
            "slice [.., t, ..]",
            || {
                let _ = t.slice(s![.., 5, ..]);
            },
            || {
                let _ = a.clone().slice_move(ndarray::s![.., 5, ..]);
            },
        );
    }

    // --- dot (2D x 2D matmul) ---
    {
        let t_a = Tensor::random_uniform((b * s, d));
        let t_b = Tensor::random_uniform((d, d));
        let a_a: ArcArrayD<f64> = ArcArrayD::ones(vec![b * s, d]);
        let a_b: ArcArrayD<f64> = ArcArrayD::ones(vec![d, d]);

        bench(
            "dot (2D x 2D)",
            || {
                let _ = t_a.dot(&t_b);
            },
            || {
                let l = a_a.view().into_dimensionality::<Ix2>().unwrap();
                let r = a_b.view().into_dimensionality::<Ix2>().unwrap();
                let _: ArrayD<f64> = l.dot(&r).into_dyn();
            },
        );
    }

    // --- reshape ---
    {
        let t = Tensor::random_uniform((b, s, d));
        let a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s, d]);

        bench(
            "reshape",
            || {
                let _ = t.clone().reshape((b * s, d));
            },
            || {
                let _ = a.clone().into_shape_clone(vec![b * s, d]).unwrap();
            },
        );
    }

    // --- transpose ---
    {
        let t = Tensor::random_uniform((b * s, d));
        let a: ArcArrayD<f64> = ArcArrayD::ones(vec![b * s, d]);

        bench(
            "t() transpose",
            || {
                let _ = t.t();
            },
            || {
                let _ = a.clone().reversed_axes();
            },
        );
    }

    // --- permute (4D, attention-style) ---
    {
        let t = Tensor::random_uniform((b, s, h, dh));
        let a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s, h, dh]);

        bench(
            "permute [0,2,1,3]",
            || {
                let _ = t.permute(&[0, 2, 1, 3]);
            },
            || {
                let _ = a.clone().permuted_axes(vec![0, 2, 1, 3]);
            },
        );
    }

    // --- insert_axis ---
    {
        let t = Tensor::random_uniform((b, s));
        let a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s]);

        bench(
            "insert_axis(1)",
            || {
                let _ = t.insert_axis(1);
            },
            || {
                let _ = a.clone().insert_axis(Axis(1));
            },
        );
    }

    // --- broadcast + to_owned ---
    {
        let t = Tensor::random_uniform((b, 1, s));
        let a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, 1, s]);

        bench(
            "broadcast",
            || {
                let _ = t.broadcast((b, s, s));
            },
            || {
                let _ = a.broadcast(vec![b, s, s]).unwrap().to_owned();
            },
        );
    }

    // --- sum_axis ---
    {
        let t = Tensor::random_uniform((b, s, d));
        let a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s, d]);

        bench(
            "sum_axis(2)",
            || {
                let _ = t.sum_axis(2);
            },
            || {
                let _ = a.sum_axis(Axis(2));
            },
        );
    }

    // --- tensor + tensor (arithmetic) ---
    {
        let t_a = Tensor::random_uniform((b, s, d));
        let t_b = Tensor::random_uniform((b, s, d));
        let a_a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s, d]);
        let a_b: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s, d]);

        bench(
            "&tensor + &tensor",
            || {
                let _ = &t_a + &t_b;
            },
            || {
                let _: ArcArrayD<f64> = (&a_a + &a_b).into();
            },
        );
    }

    // --- tensor * scalar ---
    {
        let t = Tensor::random_uniform((b, s, d));
        let a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s, d]);

        bench(
            "&tensor * scalar",
            || {
                let _ = &t * 2.0;
            },
            || {
                let _: ArcArrayD<f64> = (a.clone() * 2.0).into();
            },
        );
    }

    // --- mapv (tanh) ---
    {
        let t = Tensor::random_uniform((b, s, d));
        let a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s, d]);

        bench(
            "mapv(tanh)",
            || {
                let _ = t.mapv(|x| x.tanh());
            },
            || {
                let _ = a.mapv(|x: f64| x.tanh());
            },
        );
    }

    // --- slice_assign ---
    {
        let src = Tensor::random_uniform((b, d));

        bench(
            "slice_assign",
            || {
                let mut t = Tensor::zeros((b, s, d));
                t.slice_assign(s![.., 5, ..], &src);
            },
            || {
                let mut a: ArcArrayD<f64> = ArcArrayD::zeros(vec![b, s, d]);
                let src_a: ArcArrayD<f64> = ArcArrayD::ones(vec![b, d]);
                a.slice_mut(ndarray::s![.., 5, ..]).assign(&src_a);
            },
        );
    }

    // --- combined: slice -> dot -> reshape (common forward pass pattern) ---
    {
        let t_x = Tensor::random_uniform((b, s, d));
        let t_w = Tensor::random_uniform((d, d));
        let a_x: ArcArrayD<f64> = ArcArrayD::ones(vec![b, s, d]);
        let a_w: ArcArrayD<f64> = ArcArrayD::ones(vec![d, d]);

        bench(
            "slice->dot->reshape",
            || {
                let x_t = t_x.slice(s![.., 5, ..]);
                let y = x_t.dot(&t_w);
                let _ = y.reshape((b, 1, d));
            },
            || {
                let x_t = a_x.clone().slice_move(ndarray::s![.., 5, ..]);
                let l = x_t.view().into_dimensionality::<Ix2>().unwrap();
                let r = a_w.view().into_dimensionality::<Ix2>().unwrap();
                let y: ArcArrayD<f64> = l.dot(&r).into_dyn().into();
                let _ = y.into_shape_clone(vec![b, 1, d]).unwrap();
            },
        );
    }
}

use ndarray::{ArcArrayD, Array3, ArrayD, Ix3};
use std::time::Instant;

const WARMUP: usize = 10;
const ITERS: usize = 500;

fn bench(name: &str, f1: impl Fn(), f2: impl Fn()) {
    for _ in 0..WARMUP {
        f1();
        f2();
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        f1();
    }
    let t1 = start.elapsed().as_secs_f64();

    let start = Instant::now();
    for _ in 0..ITERS {
        f2();
    }
    let t2 = start.elapsed().as_secs_f64();

    let ratio = t2 / t1;
    println!(
        "{:<40} static={:.4}s  dynamic={:.4}s  ratio={:.2}x",
        name, t1, t2, ratio
    );
}

fn main() {
    let b = 16;
    let m = 200;
    let n = 200;

    println!("Static (Array3) vs Dynamic (ArrayD/ArcArrayD) elementwise ops");
    println!("Shape: ({b}, {m}, {n}), {ITERS} iterations\n");

    // Array3 vs ArrayD: &ref * &ref
    {
        let a3 = Array3::<f64>::ones((b, m, n));
        let b3 = Array3::<f64>::ones((b, m, n));
        let ad = ArrayD::<f64>::ones(vec![b, m, n]);
        let bd = ArrayD::<f64>::ones(vec![b, m, n]);

        bench(
            "&Array3 * &Array3 vs &ArrayD * &ArrayD",
            || {
                let _ = &a3 * &b3;
            },
            || {
                let _ = &ad * &bd;
            },
        );
    }

    // Array3 vs ArcArrayD: &ref * &ref
    {
        let a3 = Array3::<f64>::ones((b, m, n));
        let b3 = Array3::<f64>::ones((b, m, n));
        let ad = ArcArrayD::<f64>::ones(vec![b, m, n]);
        let bd = ArcArrayD::<f64>::ones(vec![b, m, n]);

        bench(
            "&Array3 * &Array3 vs &ArcArrayD * &ArcArrayD",
            || {
                let _ = &a3 * &b3;
            },
            || {
                let _ = &ad * &bd;
            },
        );
    }

    // ArrayD via views
    {
        let a3 = Array3::<f64>::ones((b, m, n));
        let b3 = Array3::<f64>::ones((b, m, n));
        let ad = ArrayD::<f64>::ones(vec![b, m, n]);
        let bd = ArrayD::<f64>::ones(vec![b, m, n]);

        bench(
            "&Array3 * &Array3 vs &ViewD * &ViewD",
            || {
                let _ = &a3 * &b3;
            },
            || {
                let _ = &ad.view() * &bd.view();
            },
        );
    }

    // ArrayD cast to Ix3 then multiply
    {
        let a3 = Array3::<f64>::ones((b, m, n));
        let b3 = Array3::<f64>::ones((b, m, n));
        let ad = ArrayD::<f64>::ones(vec![b, m, n]);
        let bd = ArrayD::<f64>::ones(vec![b, m, n]);

        bench(
            "&Array3 * &Array3 vs into_dim::<Ix3> * Ix3",
            || {
                let _ = &a3 * &b3;
            },
            || {
                let av = ad.view().into_dimensionality::<Ix3>().unwrap();
                let bv = bd.view().into_dimensionality::<Ix3>().unwrap();
                let _ = (&av * &bv).into_dyn();
            },
        );
    }

    // ArcArrayD cast to Ix3 then multiply
    {
        let a3 = Array3::<f64>::ones((b, m, n));
        let b3 = Array3::<f64>::ones((b, m, n));
        let ad = ArcArrayD::<f64>::ones(vec![b, m, n]);
        let bd = ArcArrayD::<f64>::ones(vec![b, m, n]);

        bench(
            "&Array3 * &Array3 vs ArcArrayD->Ix3 * Ix3",
            || {
                let _ = &a3 * &b3;
            },
            || {
                let av = ad.view().into_dimensionality::<Ix3>().unwrap();
                let bv = bd.view().into_dimensionality::<Ix3>().unwrap();
                let _ = (&av * &bv).into_dyn();
            },
        );
    }

    println!("\n--- With broadcasting (outer product pattern) ---");
    println!("Shape: ({b}, {m}, 1) * ({b}, 1, {n}) -> ({b}, {m}, {n})\n");

    // Broadcast mul: static vs dynamic
    {
        let a3 = Array3::<f64>::ones((b, m, 1));
        let b3 = Array3::<f64>::ones((b, 1, n));
        let ad = ArrayD::<f64>::ones(vec![b, m, 1]);
        let bd = ArrayD::<f64>::ones(vec![b, 1, n]);

        bench(
            "broadcast: &Array3 * &Array3 vs &ArrayD",
            || {
                let _ = &a3 * &b3;
            },
            || {
                let _ = &ad * &bd;
            },
        );
    }

    // Broadcast: ArrayD cast to Ix3
    {
        let a3 = Array3::<f64>::ones((b, m, 1));
        let b3 = Array3::<f64>::ones((b, 1, n));
        let ad = ArrayD::<f64>::ones(vec![b, m, 1]);
        let bd = ArrayD::<f64>::ones(vec![b, 1, n]);

        bench(
            "broadcast: &Array3 * &Array3 vs ->Ix3 * Ix3",
            || {
                let _ = &a3 * &b3;
            },
            || {
                let av = ad.view().into_dimensionality::<Ix3>().unwrap();
                let bv = bd.view().into_dimensionality::<Ix3>().unwrap();
                let _ = (&av * &bv).into_dyn();
            },
        );
    }

    // + and - for completeness
    println!("\n--- Other ops ---\n");

    {
        let a3 = Array3::<f64>::ones((b, m, n));
        let b3 = Array3::<f64>::ones((b, m, n));
        let ad = ArrayD::<f64>::ones(vec![b, m, n]);
        let bd = ArrayD::<f64>::ones(vec![b, m, n]);

        bench(
            "&Array3 + &Array3 vs &ArrayD + &ArrayD",
            || {
                let _ = &a3 + &b3;
            },
            || {
                let _ = &ad + &bd;
            },
        );

        bench(
            "&Array3 + &Array3 vs ->Ix3 + Ix3",
            || {
                let _ = &a3 + &b3;
            },
            || {
                let av = ad.view().into_dimensionality::<Ix3>().unwrap();
                let bv = bd.view().into_dimensionality::<Ix3>().unwrap();
                let _ = (&av + &bv).into_dyn();
            },
        );
    }
}

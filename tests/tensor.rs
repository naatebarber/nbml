use nbml::s;
use nbml::tensor::{SliceAxis, Tensor};

// ── Initialization ──────────────────────────────────────────────────

#[test]
fn zeros() {
    let t = Tensor::zeros((2, 3));
    assert_eq!(t.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(t[[i, j]], 0.0);
        }
    }
}

#[test]
fn ones() {
    let t = Tensor::ones((3, 2));
    assert_eq!(t.shape(), &[3, 2]);
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(t[[i, j]], 1.0);
        }
    }
}

#[test]
fn from_elem() {
    let t = Tensor::from_elem((2, 2), 5.0);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t[[0, 0]], 5.0);
    assert_eq!(t[[1, 1]], 5.0);
}

#[test]
fn random_normal() {
    let t = Tensor::random_normal((4, 5));
    assert_eq!(t.shape(), &[4, 5]);
}

#[test]
fn random_uniform() {
    let t = Tensor::random_uniform(100);
    assert_eq!(t.shape(), &[100]);
    for i in 0..100 {
        assert!(t[[i]] >= 0.0 && t[[i]] < 1.0);
    }
}

#[test]
fn from_vec() {
    let t = Tensor::from_vec((2, 2), vec![1., 2., 3., 4.]);
    assert_eq!(t[[0, 0]], 1.);
    assert_eq!(t[[0, 1]], 2.);
    assert_eq!(t[[1, 0]], 3.);
    assert_eq!(t[[1, 1]], 4.);
}

// ── Shape descriptions ──────────────────────────────────────────────

#[test]
fn rank() {
    assert_eq!(Tensor::zeros(2).rank(), 1);
    assert_eq!(Tensor::zeros((2, 3)).rank(), 2);
    assert_eq!(Tensor::zeros((2, 3, 4)).rank(), 3);
}

#[test]
fn shape() {
    assert_eq!(Tensor::zeros((2, 3, 4)).shape(), &[2, 3, 4]);
    assert_eq!(Tensor::zeros(1).shape(), &[1]);
}

#[test]
fn dim1() {
    assert_eq!(Tensor::zeros(10).dim1(), 10)
}

#[test]
#[should_panic(expected = "attempted to call dim1")]
fn dim1_wrong_rank() {
    Tensor::zeros((2, 3)).dim1();
}

#[test]
fn dim2() {
    assert_eq!(Tensor::zeros((3, 5)).dim2(), (3, 5));
    assert_eq!(Tensor::zeros((1, 1)).dim2(), (1, 1));
}

#[test]
#[should_panic(expected = "attempted to call dim2")]
fn dim2_wrong_rank() {
    Tensor::zeros((2, 3, 4)).dim2();
}

#[test]
fn dim3() {
    assert_eq!(Tensor::zeros((2, 3, 4)).dim3(), (2, 3, 4));
}

#[test]
#[should_panic(expected = "attempted to call dim3")]
fn dim3_wrong_rank() {
    Tensor::zeros((2, 3)).dim3();
}

#[test]
fn dim4() {
    assert_eq!(Tensor::zeros((2, 3, 4, 5)).dim4(), (2, 3, 4, 5));
}

#[test]
#[should_panic(expected = "attempted to call dim4")]
fn dim4_wrong_rank() {
    Tensor::zeros((2, 3)).dim4();
}

// ── Elementwise ops ─────────────────────────────────────────────────

#[test]
fn exp() {
    let e = std::f64::consts::E;
    let r = Tensor::from_elem(2, 1.0).exp();
    assert!((r[[0]] - e).abs() < 1e-10);
    assert!((r[[1]] - e).abs() < 1e-10);

    // exp(0) = 1
    let r = Tensor::zeros(1).exp();
    assert!((r[[0]] - 1.0).abs() < 1e-10);
}

#[test]
fn ln() {
    let e = std::f64::consts::E;

    // ln(e) = 1
    let r = Tensor::from_elem(2, e).ln();
    assert!((r[[0]] - 1.0).abs() < 1e-10);

    // ln(1) = 0
    let r = Tensor::ones(1).ln();
    assert!((r[[0]]).abs() < 1e-10);

    // exp/ln roundtrip
    let t = Tensor::from_elem(3, 2.5);
    let r = t.exp().ln();
    for i in 0..3 {
        assert!((r[[i]] - 2.5).abs() < 1e-10);
    }
}

#[test]
fn mapv() {
    let r = Tensor::from_elem(3, 4.0).mapv(|x| x * x + 1.0);
    for i in 0..3 {
        assert_eq!(r[[i]], 17.0);
    }

    // identity
    let t = Tensor::from_elem(2, 7.0);
    let r = t.mapv(|x| x);
    assert_eq!(r[[0]], 7.0);
}

#[test]
fn powi() {
    let r = Tensor::from_elem(2, 3.0).powi(3);
    assert_eq!(r[[0]], 27.0);
    assert_eq!(r[[1]], 27.0);

    // x^0 = 1
    let r = Tensor::from_elem(2, 5.0).powi(0);
    assert_eq!(r[[0]], 1.0);

    // x^1 = x
    let r = Tensor::from_elem(2, 5.0).powi(1);
    assert_eq!(r[[0]], 5.0);
}

#[test]
fn powf() {
    // sqrt via powf
    let r = Tensor::from_elem(2, 4.0).powf(0.5);
    assert!((r[[0]] - 2.0).abs() < 1e-10);

    // cube
    let r = Tensor::from_elem(1, 2.0).powf(3.0);
    assert!((r[[0]] - 8.0).abs() < 1e-10);
}

#[test]
fn sqrt() {
    assert!((Tensor::from_elem(1, 9.0).sqrt()[[0]] - 3.0).abs() < 1e-10);
    assert!((Tensor::from_elem(1, 0.0).sqrt()[[0]]).abs() < 1e-10);
    assert!((Tensor::from_elem(1, 1.0).sqrt()[[0]] - 1.0).abs() < 1e-10);
}

#[test]
fn log() {
    // log10(100) = 2
    assert!((Tensor::from_elem(1, 100.0).log(10.0)[[0]] - 2.0).abs() < 1e-10);
    // log2(8) = 3
    assert!((Tensor::from_elem(1, 8.0).log(2.0)[[0]] - 3.0).abs() < 1e-10);
}

// ── Global reductions ────────────────────────────────────────────────

#[test]
fn sum() {
    // all ones
    assert_eq!(Tensor::ones((2, 3)).sum(), 6.0);

    // mixed values
    let t = Tensor::from_vec(3, vec![1.0, 2.0, 3.0]);
    assert_eq!(t.sum(), 6.0);

    // zeros
    assert_eq!(Tensor::zeros((4, 4)).sum(), 0.0);
}

#[test]
fn mean() {
    let t = Tensor::from_vec(4, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.mean(), 2.5);

    assert_eq!(Tensor::ones((3, 3)).mean(), 1.0);
}

#[test]
fn std_dev() {
    // known values: [2, 4, 4, 4, 5, 5, 7, 9], population std = 2.0
    let t = Tensor::from_vec(8, vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert!((t.std(0.0) - 2.0).abs() < 1e-10);

    // sample std (ddof=1)
    let sample_std = t.std(1.0);
    assert!((sample_std - (32.0_f64 / 7.0).sqrt()).abs() < 1e-10);

    // constant tensor has zero std
    assert_eq!(Tensor::from_elem((2, 3), 5.0).std(0.0), 0.0);
}

#[test]
fn var() {
    let t = Tensor::from_vec(8, vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    assert!((t.var(0.0) - 4.0).abs() < 1e-10);

    // sample variance (ddof=1)
    assert!((t.var(1.0) - 32.0 / 7.0).abs() < 1e-10);

    // constant tensor
    assert_eq!(Tensor::from_elem(10, 3.0).var(0.0), 0.0);
}

// ── Indexing ─────────────────────────────────────────────────────────

#[test]
fn index() {
    // 1D
    assert_eq!(Tensor::ones(5)[[3]], 1.0);

    // 2D read + write
    let mut t = Tensor::zeros((2, 3));
    t[[1, 2]] = 42.0;
    assert_eq!(t[[1, 2]], 42.0);
    assert_eq!(t[[0, 0]], 0.0);

    // 3D
    let mut t = Tensor::zeros((2, 3, 4));
    t[[1, 2, 3]] = 7.0;
    assert_eq!(t[[1, 2, 3]], 7.0);

    // 4D
    let mut t = Tensor::zeros((2, 2, 2, 2));
    t[[1, 1, 1, 1]] = 99.0;
    assert_eq!(t[[1, 1, 1, 1]], 99.0);
}

// ── Arithmetic ops (Tensor <op> Tensor) ─────────────────────────────

#[test]
fn add_tensor() {
    // ref + ref
    let a = Tensor::from_elem((2, 2), 1.0);
    let b = Tensor::from_elem((2, 2), 2.0);
    assert_eq!((&a + &b)[[0, 0]], 3.0);

    // owned + owned
    assert_eq!((Tensor::ones(2) + Tensor::ones(2))[[0]], 2.0);

    // owned + ref
    let r = Tensor::ones(2);
    assert_eq!((Tensor::ones(2) + &r)[[0]], 2.0);

    // ref + owned
    let l = Tensor::ones(2);
    assert_eq!((&l + Tensor::ones(2))[[0]], 2.0);
}

#[test]
fn sub_tensor() {
    assert_eq!(
        (&Tensor::from_elem(2, 5.0) - &Tensor::from_elem(2, 3.0))[[0]],
        2.0
    );
}

#[test]
fn mul_tensor() {
    assert_eq!(
        (&Tensor::from_elem(2, 3.0) * &Tensor::from_elem(2, 4.0))[[0]],
        12.0
    );
}

#[test]
fn div_tensor() {
    assert_eq!(
        (&Tensor::from_elem(2, 10.0) / &Tensor::from_elem(2, 2.0))[[0]],
        5.0
    );
}

#[test]
fn negate_tensor() {
    assert_eq!(-Tensor::from_elem(1, 1.)[[0]], -1.)
}

// ── Assign ops (Tensor <op>= Tensor) ────────────────────────────────

#[test]
fn add_assign_tensor() {
    // ref rhs
    let mut a = Tensor::from_elem(2, 1.0);
    a += &Tensor::from_elem(2, 2.0);
    assert_eq!(a[[0]], 3.0);

    // owned rhs
    let mut a = Tensor::from_elem(2, 1.0);
    a += Tensor::from_elem(2, 9.0);
    assert_eq!(a[[0]], 10.0);
}

#[test]
fn sub_assign_tensor() {
    let mut a = Tensor::from_elem(2, 5.0);
    a -= &Tensor::from_elem(2, 2.0);
    assert_eq!(a[[0]], 3.0);
}

#[test]
fn mul_assign_tensor() {
    let mut a = Tensor::from_elem(2, 3.0);
    a *= &Tensor::from_elem(2, 4.0);
    assert_eq!(a[[0]], 12.0);
}

#[test]
fn div_assign_tensor() {
    let mut a = Tensor::from_elem(2, 10.0);
    a /= &Tensor::from_elem(2, 2.0);
    assert_eq!(a[[0]], 5.0);
}

// ── Scalar ops (Tensor <op> Float) ──────────────────────────────────

#[test]
fn add_scalar() {
    // owned
    assert_eq!((Tensor::from_elem(2, 1.0) + 3.0)[[0]], 4.0);
    // ref
    assert_eq!((&Tensor::from_elem(2, 1.0) + 3.0)[[0]], 4.0);
}

#[test]
fn sub_scalar() {
    assert_eq!((Tensor::from_elem(2, 5.0) - 2.0)[[0]], 3.0);
}

#[test]
fn mul_scalar() {
    assert_eq!((Tensor::from_elem(2, 3.0) * 4.0)[[0]], 12.0);
}

#[test]
fn div_scalar() {
    assert_eq!((Tensor::from_elem(2, 10.0) / 2.0)[[0]], 5.0);
}

// ── Scalar assign ops ───────────────────────────────────────────────

#[test]
fn scalar_assign_ops() {
    let mut t = Tensor::from_elem(2, 1.0);
    t += 2.0;
    assert_eq!(t[[0]], 3.0);

    t -= 1.0;
    assert_eq!(t[[0]], 2.0);

    t *= 5.0;
    assert_eq!(t[[0]], 10.0);

    t /= 2.0;
    assert_eq!(t[[0]], 5.0);
}

// ── Dot product ─────────────────────────────────────────────────────

#[test]
fn dot() {
    // 1d * 1d (elementwise via Mul path)
    let c = Tensor::from_elem(3, 2.0).dot(&Tensor::from_elem(3, 3.0));
    assert_eq!(c[[0]], 6.0);
    assert_eq!(c[[2]], 6.0);

    // 2d @ 2d identity
    let mut a = Tensor::zeros((2, 2));
    a[[0, 0]] = 1.0;
    a[[0, 1]] = 2.0;
    a[[1, 0]] = 3.0;
    a[[1, 1]] = 4.0;
    let mut id = Tensor::zeros((2, 2));
    id[[0, 0]] = 1.0;
    id[[1, 1]] = 1.0;
    let c = a.dot(&id);
    assert_eq!(c[[0, 0]], 1.0);
    assert_eq!(c[[0, 1]], 2.0);
    assert_eq!(c[[1, 0]], 3.0);
    assert_eq!(c[[1, 1]], 4.0);

    // 2d @ 2d matmul: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    let mut b = Tensor::zeros((2, 2));
    b[[0, 0]] = 5.0;
    b[[0, 1]] = 6.0;
    b[[1, 0]] = 7.0;
    b[[1, 1]] = 8.0;
    let c = a.dot(&b);
    assert_eq!(c[[0, 0]], 19.0);
    assert_eq!(c[[0, 1]], 22.0);
    assert_eq!(c[[1, 0]], 43.0);
    assert_eq!(c[[1, 1]], 50.0);

    // 2d @ 1d: [[1,2],[3,4]] @ [1,0] = [1,3]
    let mut v = Tensor::zeros(2);
    v[[0]] = 1.0;
    let c = a.dot(&v);
    assert_eq!(c.shape(), &[2]);
    assert_eq!(c[[0]], 1.0);
    assert_eq!(c[[1]], 3.0);

    // 1d @ 2d: [1,0] @ [[1,2],[3,4]] = [1,2]
    let c = v.dot(&a);
    assert_eq!(c.shape(), &[2]);
    assert_eq!(c[[0]], 1.0);
    assert_eq!(c[[1]], 2.0);
}

#[test]
#[should_panic(expected = "attempted to dot tensor of rank")]
fn dot_unsupported_ranks() {
    Tensor::zeros((2, 3, 4)).dot(&Tensor::zeros((4, 5, 6)));
}

// ── Transpose ───────────────────────────────────────────────────────

#[test]
fn transpose() {
    // 2D transpose: shape flips
    let mut t = Tensor::zeros((2, 3));
    t[[0, 1]] = 5.0;
    t[[1, 2]] = 9.0;
    let tr = t.t();
    assert_eq!(tr.shape(), &[3, 2]);
    assert_eq!(tr[[1, 0]], 5.0);
    assert_eq!(tr[[2, 1]], 9.0);

    // double transpose is identity
    let mut a = Tensor::zeros((2, 2));
    a[[0, 0]] = 1.0;
    a[[0, 1]] = 2.0;
    a[[1, 0]] = 3.0;
    a[[1, 1]] = 4.0;
    let roundtrip = a.t().t();
    assert_eq!(roundtrip[[0, 0]], 1.0);
    assert_eq!(roundtrip[[0, 1]], 2.0);
    assert_eq!(roundtrip[[1, 0]], 3.0);
    assert_eq!(roundtrip[[1, 1]], 4.0);

    // 3D transpose reverses all axes: [2,3,4] -> [4,3,2]
    let mut t = Tensor::zeros((2, 3, 4));
    t[[0, 1, 2]] = 7.0;
    let tr = t.t();
    assert_eq!(tr.shape(), &[4, 3, 2]);
    assert_eq!(tr[[2, 1, 0]], 7.0);

    // 1D transpose is a no-op
    let t = Tensor::from_elem(5, 3.0);
    let tr = t.t();
    assert_eq!(tr.shape(), &[5]);
    assert_eq!(tr[[0]], 3.0);

    // transpose works with dot: A @ A^T
    let mut m = Tensor::zeros((2, 3));
    m[[0, 0]] = 1.0;
    m[[0, 1]] = 2.0;
    m[[0, 2]] = 3.0;
    m[[1, 0]] = 4.0;
    m[[1, 1]] = 5.0;
    m[[1, 2]] = 6.0;
    let mt = m.t();
    assert_eq!(mt.shape(), &[3, 2]);
    let aat = m.dot(&mt);
    assert_eq!(aat.shape(), &[2, 2]);
    // [1,2,3]·[1,2,3] = 14, [1,2,3]·[4,5,6] = 32
    assert_eq!(aat[[0, 0]], 14.0);
    assert_eq!(aat[[0, 1]], 32.0);
    assert_eq!(aat[[1, 0]], 32.0);
    assert_eq!(aat[[1, 1]], 77.0);
}

// ── Broadcast ───────────────────────────────────────────────────────

#[test]
fn broadcast() {
    // scalar-like broadcast: [1] -> [4]
    let t = Tensor::from_elem(1, 3.0);
    let b = t.broadcast(4);
    assert_eq!(b.shape(), &[4]);
    for i in 0..4 {
        assert_eq!(b[[i]], 3.0);
    }

    // row broadcast: [1, 3] -> [4, 3]
    let mut t = Tensor::zeros((1, 3));
    t[[0, 0]] = 1.0;
    t[[0, 1]] = 2.0;
    t[[0, 2]] = 3.0;
    let b = t.broadcast((4, 3));
    assert_eq!(b.shape(), &[4, 3]);
    for i in 0..4 {
        assert_eq!(b[[i, 0]], 1.0);
        assert_eq!(b[[i, 1]], 2.0);
        assert_eq!(b[[i, 2]], 3.0);
    }

    // col broadcast: [3, 1] -> [3, 4]
    let mut t = Tensor::zeros((3, 1));
    t[[0, 0]] = 10.0;
    t[[1, 0]] = 20.0;
    t[[2, 0]] = 30.0;
    let b = t.broadcast((3, 4));
    assert_eq!(b.shape(), &[3, 4]);
    for j in 0..4 {
        assert_eq!(b[[0, j]], 10.0);
        assert_eq!(b[[1, j]], 20.0);
        assert_eq!(b[[2, j]], 30.0);
    }

    // higher-rank broadcast: [1, 3] -> [2, 4, 3]
    let mut t = Tensor::zeros((1, 3));
    t[[0, 0]] = 1.0;
    t[[0, 1]] = 2.0;
    t[[0, 2]] = 3.0;
    let b = t.broadcast((2, 4, 3));
    assert_eq!(b.shape(), &[2, 4, 3]);
    for i in 0..2 {
        for j in 0..4 {
            assert_eq!(b[[i, j, 0]], 1.0);
            assert_eq!(b[[i, j, 2]], 3.0);
        }
    }

    // broadcast + arithmetic: mask-style usage
    let data = Tensor::ones((3, 4));
    let mask = Tensor::from_elem((1, 4), 2.0).broadcast((3, 4));
    let result = &data * &mask;
    assert_eq!(result.shape(), &[3, 4]);
    assert_eq!(result[[2, 3]], 2.0);
}

#[test]
#[should_panic]
fn broadcast_incompatible() {
    Tensor::zeros((2, 3)).broadcast((4, 4));
}

// ── Shape manipulation ──────────────────────────────────────────────

#[test]
fn permute() {
    // 2D
    let mut t = Tensor::zeros((2, 3));
    t[[0, 1]] = 5.0;
    let p = t.permute(&[1, 0]);
    assert_eq!(p.shape(), &[3, 2]);
    assert_eq!(p[[1, 0]], 5.0);

    // 3D
    let p = Tensor::zeros((2, 3, 4)).permute(&[2, 0, 1]);
    assert_eq!(p.shape(), &[4, 2, 3]);
}

#[test]
fn reshape() {
    // flatten
    let t = Tensor::ones((2, 3));
    let r = t.reshape(6);
    assert_eq!(r.shape(), &[6]);
    assert_eq!(r[[0]], 1.0);
    assert_eq!(r[[5]], 1.0);

    // data preserved
    let mut t = Tensor::zeros((2, 3));
    t[[1, 2]] = 42.0;
    assert_eq!(t.reshape(6)[[5]], 42.0);

    // expand dims
    let r = Tensor::ones(6).reshape((2, 3));
    assert_eq!(r.shape(), &[2, 3]);
}

#[test]
fn insert_axis() {
    let t = Tensor::zeros((3, 4));

    // front
    assert_eq!(t.insert_axis(0).shape(), &[1, 3, 4]);
    // middle
    assert_eq!(t.insert_axis(1).shape(), &[3, 1, 4]);
    // end
    assert_eq!(t.insert_axis(2).shape(), &[3, 4, 1]);
}

#[test]
#[should_panic(expected = "attempted to insert axis")]
fn insert_axis_out_of_bounds() {
    Tensor::zeros((3, 4)).insert_axis(3);
}

// ── Slicing ─────────────────────────────────────────────────────────

#[test]
fn slice() {
    // range slice
    let mut t = Tensor::zeros((4, 4));
    t[[1, 2]] = 7.0;
    let s = t.slice(s![1..3, 2..4]);
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s[[0, 0]], 7.0);

    // index + range-from: pick single row
    let mut t = Tensor::zeros((3, 4));
    t[[1, 0]] = 10.0;
    t[[1, 3]] = 20.0;
    let s = t.slice(s![1, 0..]);
    assert_eq!(s.shape(), &[4]);
    assert_eq!(s[[0]], 10.0);
    assert_eq!(s[[3]], 20.0);

    // full range
    let t = Tensor::ones((3, 4));
    assert_eq!(t.slice(s![.., ..]).shape(), &[3, 4]);

    // range-from
    let mut t = Tensor::zeros(5);
    t[[3]] = 1.0;
    t[[4]] = 2.0;
    let s = t.slice(s![3..]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(s[[0]], 1.0);
    assert_eq!(s[[1]], 2.0);

    // range-to
    let mut t = Tensor::zeros(5);
    t[[0]] = 1.0;
    t[[1]] = 2.0;
    let s = t.slice(s![..2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(s[[0]], 1.0);

    // negative end
    let mut t = Tensor::zeros(5);
    t[[0]] = 1.0;
    t[[2]] = 3.0;
    let s = t.slice(s![0..-2]);
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s[[0]], 1.0);
    assert_eq!(s[[2]], 3.0);
}

#[test]
#[should_panic(expected = "attempted to slice")]
fn slice_wrong_rank() {
    Tensor::zeros((3, 4)).slice(s![0..]);
}

#[test]
fn slice_assign() {
    let mut t = Tensor::zeros((3, 3));
    let patch = Tensor::ones((2, 2));
    t.slice_assign(s![0..2, 0..2], &patch);
    assert_eq!(t[[0, 0]], 1.0);
    assert_eq!(t[[1, 1]], 1.0);
    assert_eq!(t[[2, 2]], 0.0);
}

#[test]
#[should_panic(expected = "cannot assign a tensor of shape")]
fn slice_assign_shape_mismatch() {
    let mut t = Tensor::zeros((3, 3));
    t.slice_assign(s![0..2, 0..2], &Tensor::ones((3, 2)));
}

// ── Concatenate ─────────────────────────────────────────────────────

#[test]
fn concatenate() {
    // axis 0
    let a = Tensor::zeros((2, 3));
    let b = Tensor::ones((1, 3));
    let c = Tensor::concatenate(0, &[&a, &b]);
    assert_eq!(c.shape(), &[3, 3]);
    assert_eq!(c[[2, 0]], 1.0);

    // axis 1
    let a = Tensor::zeros((2, 2));
    let b = Tensor::ones((2, 3));
    let c = Tensor::concatenate(1, &[&a, &b]);
    assert_eq!(c.shape(), &[2, 5]);
    assert_eq!(c[[0, 2]], 1.0);
}

#[test]
#[should_panic(expected = "cannot concatenate a list of 0")]
fn concatenate_empty() {
    Tensor::concatenate(0, &[]);
}

#[test]
#[should_panic(expected = "cannot concatenate tensor of dim")]
fn concatenate_shape_mismatch() {
    Tensor::concatenate(0, &[&Tensor::zeros((2, 3)), &Tensor::zeros((2, 4))]);
}

// ── Stack ───────────────────────────────────────────────────────────

#[test]
fn stack() {
    // axis 0
    let a = Tensor::zeros((2, 3));
    let b = Tensor::ones((2, 3));
    let c = Tensor::stack(0, &[&a, &b]);
    assert_eq!(c.shape(), &[2, 2, 3]);
    assert_eq!(c[[0, 0, 0]], 0.0);
    assert_eq!(c[[1, 0, 0]], 1.0);

    // axis 1
    let c = Tensor::stack(1, &[&a, &b]);
    assert_eq!(c.shape(), &[2, 2, 3]);
}

#[test]
#[should_panic(expected = "cannot stack a list of 0")]
fn stack_empty() {
    Tensor::stack(0, &[]);
}

#[test]
#[should_panic(expected = "cannot stack tensor of dim")]
fn stack_shape_mismatch() {
    Tensor::stack(0, &[&Tensor::zeros((2, 3)), &Tensor::zeros((2, 4))]);
}

// ── Reduction ops ───────────────────────────────────────────────────

#[test]
fn sum_axis() {
    let t = Tensor::ones((3, 4));

    // axis 0: [3,4] -> [4], each element = 3
    let s = t.sum_axis(0);
    assert_eq!(s.shape(), &[4]);
    assert_eq!(s[[0]], 3.0);

    // axis 1: [3,4] -> [3], each element = 4
    let s = t.sum_axis(1);
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s[[0]], 4.0);
}

#[test]
#[should_panic(expected = "attempted to sum axis")]
fn sum_axis_out_of_bounds() {
    Tensor::zeros((3, 4)).sum_axis(2);
}

#[test]
fn mean_axis() {
    let mut t = Tensor::zeros((2, 2));
    t[[0, 0]] = 2.0;
    t[[0, 1]] = 4.0;
    t[[1, 0]] = 6.0;
    t[[1, 1]] = 8.0;

    // axis 0: col means
    let m = t.mean_axis(0);
    assert_eq!(m.shape(), &[2]);
    assert_eq!(m[[0]], 4.0); // (2+6)/2
    assert_eq!(m[[1]], 6.0); // (4+8)/2

    // axis 1: row means
    let m = t.mean_axis(1);
    assert_eq!(m.shape(), &[2]);
    assert_eq!(m[[0]], 3.0); // (2+4)/2
    assert_eq!(m[[1]], 7.0); // (6+8)/2
}

#[test]
#[should_panic(expected = "attempted to mean axis")]
fn mean_axis_out_of_bounds() {
    Tensor::zeros((3, 4)).mean_axis(2);
}

// ── Min / Max / Argmin / Argmax ─────────────────────────────────────

#[test]
fn min_max() {
    let mut t = Tensor::zeros((2, 3));
    t[[0, 0]] = -5.0;
    t[[1, 2]] = 10.0;
    assert_eq!(t.min(), -5.0);
    assert_eq!(t.max(), 10.0);
}

#[test]
fn argmin_argmax() {
    let mut t = Tensor::zeros((2, 3));
    t[[0, 2]] = -1.0;
    t[[1, 1]] = 5.0;
    assert_eq!(t.argmin(), vec![0, 2]);
    assert_eq!(t.argmax(), vec![1, 1]);
}

// ── Clone / Debug / Serialize ───────────────────────────────────────

#[test]
fn clone_tensor() {
    let t = Tensor::from_elem(2, 7.0);
    let c = t.clone();
    assert_eq!(c[[0]], 7.0);
    assert_eq!(c[[1]], 7.0);
}

#[test]
fn debug_format() {
    let s = format!("{:?}", Tensor::zeros(2));
    assert!(s.contains("Tensor"));
}

#[test]
fn serde_roundtrip() {
    let mut t = Tensor::zeros((2, 3));
    t[[0, 1]] = 42.0;
    let json = serde_json::to_string(&t).unwrap();
    let d: Tensor = serde_json::from_str(&json).unwrap();
    assert_eq!(d.shape(), &[2, 3]);
    assert_eq!(d[[0, 1]], 42.0);
}

// ── SliceAxis conversions ───────────────────────────────────────────

#[test]
fn slice_axis_conversions() {
    // from isize
    match SliceAxis::from(3isize) {
        SliceAxis::Index(i) => assert_eq!(i, 3),
        _ => panic!("expected Index"),
    }

    // from Range<isize>
    match SliceAxis::from(1..5isize) {
        SliceAxis::Range { start, end } => {
            assert_eq!(start, Some(1));
            assert_eq!(end, Some(5));
        }
        _ => panic!("expected Range"),
    }

    // from RangeFrom<isize>
    match SliceAxis::from(2isize..) {
        SliceAxis::Range { start, end } => {
            assert_eq!(start, Some(2));
            assert_eq!(end, None);
        }
        _ => panic!("expected Range"),
    }

    // from RangeTo<isize>
    match SliceAxis::from(..4isize) {
        SliceAxis::Range { start, end } => {
            assert_eq!(start, None);
            assert_eq!(end, Some(4));
        }
        _ => panic!("expected Range"),
    }

    // from RangeFull
    match SliceAxis::from(..) {
        SliceAxis::Range { start, end } => {
            assert_eq!(start, Some(0));
            assert_eq!(end, None);
        }
        _ => panic!("expected Range"),
    }

    // from_range helper
    match SliceAxis::from_range(1..5isize) {
        SliceAxis::Range { start, end } => {
            assert_eq!(start, Some(1));
            assert_eq!(end, Some(5));
        }
        _ => panic!("expected Range"),
    }
}

// ── s! macro ────────────────────────────────────────────────────────

#[test]
fn s_macro() {
    assert_eq!(s![0..3].len(), 1);
    assert_eq!(s![0..3, 1..4].len(), 2);
    assert_eq!(s![0, 1..4].len(), 2);
}

// ── Broadcasting arithmetic ─────────────────────────────────────────

#[test]
fn broadcast_arithmetic() {
    let a = Tensor::from_elem(3, 2.0);
    let b = Tensor::from_elem(1, 1.0);
    let c = &a + &b;
    assert_eq!(c[[0]], 3.0);
    assert_eq!(c[[2]], 3.0);
}

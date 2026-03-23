use std::fs;

use nbml::{
    nn::{LSM, LSTM},
    optim::ToParams,
};
use ndarray::{Array2, Array3};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn serde_model() {
    let x = Array3::random((5, 5, 32), Uniform::new(0., 1.));

    let mut lstm = LSTM::new(32);
    lstm.persist("./lstm1.bin").unwrap();
    let y = lstm.forward(x.clone(), false);

    let mut lstm2 = LSTM::new(32);
    lstm2.restore("./lstm1.bin").unwrap();
    let y2 = lstm2.forward(x, false);

    fs::remove_file("./lstm1.bin").unwrap();

    assert!(y == y2, "loaded model is not the same as dumped model");
}

#[test]
fn serde_partial_grad_model() {
    let x = Array2::random((5, 32), Uniform::new(0., 1.));

    let mut state = Array2::zeros((5, 64));
    let mut lsm = LSM::new(32, 64, 32);
    lsm.persist("./lsm1.bin").unwrap();
    let y = lsm.step(&x, 1., &mut state, false);

    let mut state = Array2::zeros((5, 64));
    let mut lsm2 = LSM::new(32, 64, 32);
    lsm2.restore("./lsm1.bin").unwrap();
    let y2 = lsm2.step(&x, 1., &mut state, false);

    fs::remove_file("./lsm1.bin").unwrap();

    assert!(
        y == y2,
        "loaded partial grad model is not the same as dumped model"
    );
}

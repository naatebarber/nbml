use std::fs;

use nbml::{nn::LSTM, optim::ToParams};
use ndarray::Array3;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[test]
fn serde_model() {
    let x = Array3::random((5, 5, 32), Uniform::new(0., 1.));

    let mut lstm = LSTM::new(32);
    lstm.dump_model("./lstm1.bin").unwrap();
    let y = lstm.forward(x.clone(), false);

    let mut lstm2 = LSTM::new(32);
    lstm2.load_model("./lstm1.bin").unwrap();
    let y2 = lstm2.forward(x, false);

    fs::remove_file("./lstm1.bin").unwrap();

    assert!(y == y2, "loaded model is not the same as dumped model");
}

#![allow(deprecated)]

use std::time::Instant;

use nbml::nn::{AttentionHead, SelfAttention};
use ndarray::{Array2, Array3};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

fn main() {
    println!("comparing speed of AttentionHead to newer SelfAttention");
    println!("each forward a (10, 300, 300) matrix 5 times.");

    let mut attnhead = AttentionHead::new(300, 300, 20);
    let mut selfattn = SelfAttention::new(300, 300, 20);

    let x = Array3::random((10, 300, 300), Uniform::new(0., 1.));

    let mask_3d = Array3::ones((10, 300, 300));
    let mask_2d = Array2::ones((10, 300));

    let attnhead_start = Instant::now();
    for i in 0..5 {
        attnhead.forward(&x, &mask_2d, false, true);
        println!("(attnhead) {i}");
    }
    let attnhead_end = Instant::now().duration_since(attnhead_start).as_secs_f32();

    let selfattn_start = Instant::now();
    for i in 0..5 {
        selfattn.forward(x.clone(), mask_3d.clone(), true);
        println!("(selfattn) {i}");
    }
    let selfattn_end = Instant::now().duration_since(selfattn_start).as_secs_f32();

    println!("attnhead: {} sec", attnhead_end);
    println!("selfattn: {} sec", selfattn_end);
}

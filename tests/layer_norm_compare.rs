use nbml::{Tensor, layers::LayerNorm as LayerNorm2, nn::LayerNorm as LayerNorm1};
use ndarray::{Array3, Axis};

#[test]
fn find_backward_divergence() {
    let features = 5;
    let mut ln1 = LayerNorm1::new(features);
    let mut ln2 = LayerNorm2::new(features);

    let data: Vec<f64> = (0..30).map(|i| (i as f64 - 15.0) / 10.0).collect();
    let x_nd = Array3::from_shape_vec((2, 3, 5), data.clone()).unwrap();
    let x_t = Tensor::from_vec((2, 3, 5), data);

    let _y1 = ln1.forward(x_nd, true);
    let _y2 = ln2.forward(x_t, true);

    let dl_data: Vec<f64> = (0..30).map(|i| (i as f64) / 30.0).collect();
    let dl_nd = Array3::from_shape_vec((2, 3, 5), dl_data.clone()).unwrap();
    let dl_t = Tensor::from_vec((2, 3, 5), dl_data);

    // ---- OLD backward, manually ----
    let d_loss_2d = dl_nd.into_shape_clone((6, 5)).unwrap();
    let dx_hat_old = &d_loss_2d * &ln1.gamma;

    let term_a = 5.0_f64 * &dx_hat_old;
    let term_b = dx_hat_old.sum_axis(Axis(1)).insert_axis(Axis(1));
    let term_c = &ln1.x_h
        * (&dx_hat_old * &ln1.x_h)
            .sum_axis(Axis(1))
            .insert_axis(Axis(1));
    let bracket = &term_a - &term_b - &term_c;
    let scale = 1.0 / (5.0_f64 * &ln1.o);
    let dx_old = &scale * &bracket;

    // ---- NEW backward, manually ----
    // Replicate what layers/layer_norm.rs does
    let d_loss_2d_new = dl_t.reshape((6, 5));
    let dx_hat_new = &d_loss_2d_new * &ln2.gamma;

    // The new code line 63-66:
    // (1. / (features as Float * &self.cache["o"]))
    //     * (features as Float * &dx_hat
    //         - &dx_hat.sum_axis(1).insert_axis(1)
    //         - &self.cache["x_h"] * (&dx_hat * &self.cache["x_h"]).sum_axis(1).insert_axis(1))

    let term_a_new = dx_hat_new.clone() * 5.0;
    let term_b_new = dx_hat_new.sum_axis(1).insert_axis(1);
    let term_c_new =
        &ln2.cache["x_h"] * (&dx_hat_new * &ln2.cache["x_h"]).sum_axis(1).insert_axis(1);

    // Compare intermediate terms
    for i in 0..5 {
        let a1 = term_a.as_slice().unwrap()[i];
        let a2 = term_a_new[[0, i]];
        let b1 = term_b[[0, 0]];
        let b2 = term_b_new[[0, 0]];
        let c1 = term_c.as_slice().unwrap()[i];
        let c2 = term_c_new[[0, i]];

        println!("elem {i}: term_a old={a1:.8} new={a2:.8} | term_c old={c1:.8} new={c2:.8}");
        if i == 0 {
            println!("         term_b old={b1:.8} new={b2:.8}");
        }
    }

    let scale_old_val = scale.as_slice().unwrap()[0];
    // For new: 1. / (5.0 * o[0])
    let o_new_val = ln2.cache["o"][[0, 0]];
    let scale_new_val = 1.0 / (5.0 * o_new_val);
    println!("\nscale old={scale_old_val:.8} new={scale_new_val:.8}");
    println!(
        "o old={:.8} new={o_new_val:.8}",
        ln1.o.as_slice().unwrap()[0]
    );

    // Now check: what does `1. / (features as Float * &self.cache["o"])` actually produce?
    // Is it 1.0 / (5.0 * o)  or  (1.0 / 5.0) * (1.0 / o)?
    // Both should be the same, but let's verify the Tensor arithmetic
    let o_cache = &ln2.cache["o"];
    let scale_tensor = 1.0 / (5.0 as f64 * o_cache);
    let scale_tensor_val = scale_tensor[[0, 0]];
    println!("scale via tensor: {scale_tensor_val:.8}");

    // Final dx comparison
    let bracket_new = &term_a_new - &term_b_new - &term_c_new;
    let dx_new_manual = &scale_tensor * &bracket_new;

    println!(
        "\ndx_old[0]={:.10} dx_new_manual[0]={:.10} ratio={:.6}",
        dx_old.as_slice().unwrap()[0],
        dx_new_manual[[0, 0]],
        dx_old.as_slice().unwrap()[0] / dx_new_manual[[0, 0]]
    );

    // Also check what the actual backward function returns
    let mut ln2b = LayerNorm2::new(features);
    let data2: Vec<f64> = (0..30).map(|i| (i as f64 - 15.0) / 10.0).collect();
    let x_t2 = Tensor::from_vec((2, 3, 5), data2);
    let dl_data2: Vec<f64> = (0..30).map(|i| (i as f64) / 30.0).collect();
    let dl_t2 = Tensor::from_vec((2, 3, 5), dl_data2);
    let _ = ln2b.forward(x_t2, true);
    let dx_actual = ln2b.backward(dl_t2);

    println!(
        "dx_actual[0,0,0]={:.10} dx_new_manual[0,0]={:.10}",
        dx_actual[[0, 0, 0]],
        dx_new_manual[[0, 0]]
    );
    println!(
        "dx_old[0]={:.10} dx_actual[0,0,0]={:.10} ratio={:.6}",
        dx_old.as_slice().unwrap()[0],
        dx_actual[[0, 0, 0]],
        dx_old.as_slice().unwrap()[0] / dx_actual[[0, 0, 0]]
    );
}

use ndarray::{Array1, Axis};
use prims::{
    nn::experimental::ctrnn2::CTRNN,
    optim::{optimizer::Optimizer, sgd::SGD},
    util::plot,
};

fn main() {
    let mut c = CTRNN::new(11, 1, 1);
    let mut o = SGD::default().with(&mut c);
    o.learning_rate = 1e-3;

    let dt = 0.01;
    let s = 0.001;

    let train_epochs = 10_000;
    let mut ys = vec![];
    let mut ys_pred = vec![];

    for i in 0..train_epochs {
        let x = (i as f64 / 10.).sin();
        let y = x.cos();

        let y_pred = c.forward(Array1::from_vec(vec![x]).insert_axis(Axis(0)), dt, s, true);

        let loss = (y - &y_pred).powi(2);
        let d_loss = 2. * (y - &y_pred);

        c.retain_steps(100);
        c.zero_grad();
        c.backward(-d_loss);
        o.step(&mut c);

        if i % 10 == 0 {
            println!("epoch={i} loss={loss}")
        }

        ys.push(y);
        ys_pred.push(y_pred[[0, 0]]);
    }

    for i in train_epochs..train_epochs + 1000 {
        let x = (i as f64 / 10.).sin();
        let y = x.cos();

        let y_pred = c.forward(Array1::from_vec(vec![x]).insert_axis(Axis(0)), dt, s, true);

        let loss = (y - &y_pred).powi(2);
        if i % 10 == 0 {
            println!("test epoch={i} loss={loss}")
        }

        ys.push(y);
        ys_pred.push(y_pred[[0, 0]]);
    }

    plot(vec![ys_pred, ys]);
}

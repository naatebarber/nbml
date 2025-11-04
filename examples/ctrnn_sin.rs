use ndarray::Array1;
use prims::{
    f,
    nn::experimental::ctrnn::CTRNN,
    optim::{adam::AdamW, optimizer::Optimizer},
    util::plot,
};

fn main() {
    let mut c = CTRNN::new(11, f::tanh, f::d_tanh);
    let mut o = AdamW::default().with(&mut c);
    o.learning_rate = 1e-3;
    let dt = 1.;
    let s = 0.01;

    let train_epochs = 10_000;
    let mut ys = vec![];
    let mut ys_pred = vec![];

    for i in 0..train_epochs {
        let y = (i as f64 / 10.).cos();
        c.forward(c.concat_front(Array1::from_vec(vec![i as f64])), dt, s);

        let y_pred = c.slice_back(1)[0];

        let loss = (y - y_pred).powi(2);
        let d_loss = c.concat_back(Array1::from_vec(vec![2. * (y - y_pred)]));

        c.backward(-d_loss);
        o.step(&mut c);

        if i % 10 == 0 {
            println!("epoch={i} loss={loss}")
        }

        ys.push(y);
        ys_pred.push(y_pred);

        c.retain_grads(100);
    }

    for i in train_epochs..train_epochs + 1000 {
        let y = (i as f64 / 10.).cos();
        c.forward(c.concat_front(Array1::from_vec(vec![i as f64])), dt, s);

        let y_pred = c.slice_back(1)[0];

        let loss = (y - y_pred).powi(2);
        if i % 10 == 0 {
            println!("test epoch={i} loss={loss}")
        }

        ys.push(y);
        ys_pred.push(y_pred);

        c.retain_grads(100);
    }

    plot(vec![ys_pred, ys]);
}

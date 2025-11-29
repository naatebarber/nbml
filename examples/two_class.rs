use nbml::{
    f::Activation,
    nn::{FFN, SequencePooling, TransformerEncoder},
    optim::{adam::AdamW, optimizer::Optimizer, param::ToParams},
};
use ndarray::{Array1, Array2, Array3, Axis, stack};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

struct Classifier {
    transformer: TransformerEncoder,
    pooling: SequencePooling,
    feed_forward: FFN,
}

impl Classifier {
    pub fn new(d_model: usize, d_head: usize, n_head: usize) -> Self {
        Self {
            transformer: TransformerEncoder::new(
                d_model,
                d_head,
                n_head,
                vec![(d_model, d_model, Activation::Relu)],
            ),
            pooling: SequencePooling::new(),
            feed_forward: FFN::new(vec![(d_model, 1, Activation::Sigmoid)]),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>) -> Array2<f64> {
        let mask = Array2::ones((x.dim().0, x.dim().1));
        let x = self.transformer.forward(x, mask.clone(), true);
        let x = self.pooling.forward(x, mask, true);
        self.feed_forward.forward(x, true)
    }

    pub fn backward(&mut self, d_loss: Array2<f64>) -> Array3<f64> {
        let d = self.feed_forward.backward(d_loss);
        let d = self.pooling.backward(d);
        self.transformer.backward(d)
    }
}

impl ToParams for Classifier {
    fn params(&mut self) -> Vec<nbml::optim::param::Param> {
        let mut params = vec![];
        params.append(&mut self.transformer.params());
        params.append(&mut self.feed_forward.params());
        params
    }
}

fn main() {
    let mut c = Classifier::new(50, 10, 5);
    let mut optim = AdamW::default().with(&mut c);
    optim.weight_decay = 0.001;
    optim.learning_rate = 0.0001;

    let x1 = Array2::random((5, 50), Uniform::new(0., 1.));
    let y1 = Array1::from_vec(vec![1.]).insert_axis(Axis(0));

    let x2 = Array2::random((5, 50), Uniform::new(0., 1.));
    let y2 = Array1::from_vec(vec![0.]).insert_axis(Axis(0));

    let x = stack(Axis(0), &[x1.view(), x2.view()]).unwrap();
    let y = stack(Axis(0), &[y1.view(), y2.view()]).unwrap();

    for e in 0..1000 {
        let y_pred = c.forward(x.clone());
        let y = y.clone().remove_axis(Axis(1));

        println!("x: {:?}", x.shape());
        println!("y_pred: {:?}", y_pred);

        let loss = -(&y * &y_pred.ln() + (1. - &y) * (1. - &y_pred).ln())
            .mean()
            .unwrap();
        let d_loss = (&y_pred - &y) / x.dim().0 as f64;

        c.backward(d_loss);
        optim.step(&mut c);

        println!("epoch={} loss={}", e, loss);
    }
}

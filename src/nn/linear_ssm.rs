use ndarray::{Array2, Array3, s};

use crate::{
    f::InitializationFn,
    optim::param::{Param, ToParams},
};

pub struct LinearSSM {
    pub d_model: usize,
    pub d_in: usize,
    pub d_out: usize,

    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub c: Array2<f64>,

    pub x: Array3<f64>,
    pub states: Array3<f64>,

    pub d_a: Array2<f64>,
    pub d_b: Array2<f64>,
    pub d_c: Array2<f64>,
}

impl LinearSSM {
    pub fn new(
        d_model: usize,
        d_in: usize,
        d_out: usize,
        init_a: InitializationFn,
        init_b: InitializationFn,
        init_c: InitializationFn,
    ) -> Self {
        Self {
            d_model,
            d_in,
            d_out,

            a: init_a((d_model, d_model)),
            b: init_b((d_in, d_model)),
            c: init_c((d_model, d_out)),

            x: Array3::zeros((0, 0, 0)),
            states: Array3::zeros((0, 0, 0)),

            d_a: Array2::zeros((0, 0)),
            d_b: Array2::zeros((0, 0)),
            d_c: Array2::zeros((0, 0)),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        assert!(features == self.d_in, "feature dimension != d_in");

        let mut state = Array2::zeros((batch_size, self.d_model));

        self.x = if grad {
            Array3::zeros((seq_len, batch_size, features))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.states = if grad {
            Array3::zeros((seq_len + 1, batch_size, self.d_model))
        } else {
            Array3::zeros((0, 0, 0))
        };

        let mut output = Array3::zeros((batch_size, seq_len, self.d_out));

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            let x_b = x_t.dot(&self.b);
            let r = state.dot(&self.a);

            if grad {
                self.x.slice_mut(s![t, .., ..]).assign(&x_t);
                self.states.slice_mut(s![t, .., ..]).assign(&state);
            }

            state = &r + &x_b;
            let y = state.dot(&self.c);
            output.slice_mut(s![.., t, ..]).assign(&y);
        }

        if grad {
            self.states.slice_mut(s![seq_len, .., ..]).assign(&state);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, _) = d_loss.dim();

        println!("d_loss: {:?}", d_loss.dim());

        let mut d_x = Array3::zeros((batch_size, seq_len, self.d_in));

        if self.d_a.dim() == (0, 0) {
            self.d_a = Array2::zeros(self.a.dim());
        }

        if self.d_b.dim() == (0, 0) {
            self.d_b = Array2::zeros(self.b.dim());
        }

        if self.d_c.dim() == (0, 0) {
            self.d_c = Array2::zeros(self.c.dim());
        }

        let mut resid = Array2::zeros((batch_size, self.d_model));

        for t in (0..seq_len).rev() {
            let d_loss_t = d_loss.slice(s![.., t, ..]);
            let state_t = self.states.slice(s![t, .., ..]);
            let state_next = self.states.slice(s![t + 1, .., ..]);
            let x_t = self.x.slice(s![t, .., ..]);

            self.d_c += &state_next.t().dot(&d_loss_t);

            let d_state_next = d_loss_t.dot(&self.c.t()) + &resid;
            self.d_a += &state_t.t().dot(&d_state_next);
            self.d_b += &x_t.t().dot(&d_state_next);

            let d_x_t = d_state_next.dot(&self.b.t());
            resid = d_state_next.dot(&self.a.t());

            d_x.slice_mut(s![.., t, ..]).assign(&d_x_t);
        }

        d_x
    }
}

impl ToParams for LinearSSM {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::matrix(&mut self.a).with_matrix_grad(&mut self.d_a),
            Param::matrix(&mut self.b).with_matrix_grad(&mut self.d_b),
            Param::matrix(&mut self.c).with_matrix_grad(&mut self.d_c),
        ]
    }
}

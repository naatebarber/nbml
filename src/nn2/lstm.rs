use serde::{Deserialize, Serialize};

use crate::{
    Tensor, f2 as f,
    optim2::{Param, ToParams},
    s,
    tensor::{Tensor1, Tensor2, Tensor3},
    util::Cache,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LSTM {
    d_model: usize,

    w_i: Tensor2,
    w_r: Tensor2,
    b: Tensor1,

    #[serde(skip)]
    cache: Cache,
    #[serde(skip)]
    grads: Cache,
}

impl LSTM {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,

            w_i: f::xavier((d_model, 4 * d_model)),
            w_r: f::xavier((d_model, 4 * d_model)),
            b: Tensor::concatenate(0, &[&Tensor1::ones(d_model), &Tensor::zeros(3 * d_model)]),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor3, grad: bool) -> Tensor3 {
        let (batch_size, seq_len, features) = x.dim3();

        assert!(features == self.d_model, "feature dimension != d_model");

        let mut state = Tensor2::zeros((batch_size, features));
        let mut cell = Tensor2::zeros((batch_size, features));
        let mut output = Tensor3::zeros_like(&x);

        if grad {
            self.cache
                .set("x", Tensor3::zeros((seq_len, batch_size, features)));
            self.cache
                .set("states", Tensor3::zeros((seq_len, batch_size, features)));
            self.cache.set(
                "preactivations",
                Tensor3::zeros((seq_len, batch_size, 4 * features)),
            );
            self.cache
                .set("gates", Tensor3::zeros((seq_len, batch_size, 4 * features)));
            self.cache
                .set("cells", Tensor3::zeros((seq_len, batch_size, features)));
        }

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            let x_i = x_t.dot(&self.w_i);
            let r = state.dot(&self.w_r);

            let preactivations = &x_i + &r + &self.b;

            let forget_gate = f::sigmoid(&preactivations.slice(s![.., 0..self.d_model]));
            let input_gate = f::sigmoid(
                &preactivations
                    .slice(s![.., self.d_model..(2 * self.d_model)])
                    .to_owned(),
            );
            let cell_gate = f::tanh(
                &preactivations
                    .slice(s![.., (2 * self.d_model)..(3 * self.d_model)])
                    .to_owned(),
            );
            let output_gate = f::sigmoid(
                &preactivations
                    .slice(s![.., (3 * self.d_model)..])
                    .to_owned(),
            );

            if grad {
                self.cache["x"].slice_assign(s![t, .., ..], &x_t);
                self.cache["states"].slice_assign(s![t, .., ..], &state);
                self.cache["preactivations"].slice_assign(s![t, .., ..], &preactivations);

                let gates =
                    Tensor::concatenate(1, &[&forget_gate, &input_gate, &cell_gate, &output_gate]);

                self.cache["gates"].slice_assign(s![t, .., ..], &gates);
                self.cache["cells"].slice_assign(s![t, .., ..], &cell);
            }

            cell = &cell * &forget_gate + (&input_gate * &cell_gate);
            state = &output_gate * f::tanh(&cell);

            output.slice_assign(s![.., t, ..], &state);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Tensor3) -> Tensor3 {
        let (batch_size, seq_len, features) = d_loss.dim3();

        let mut d_x = Tensor3::zeros_like(&d_loss);
        let mut resid = Tensor2::zeros((batch_size, features));
        let mut cell_resid = Tensor2::zeros((batch_size, features));

        for t in (0..seq_len).rev() {
            let d_loss_t = &d_loss.slice(s![.., t, ..]) + &resid;

            let x = self.cache["x"].slice(s![t, .., ..]);
            let state = self.cache["states"].slice(s![t, .., ..]);
            let preactivations = self.cache["preactivations"].slice(s![t, .., ..]);
            let preactivations_forget = preactivations.slice(s![.., 0..self.d_model]);
            let preactivations_input =
                preactivations.slice(s![.., self.d_model..(2 * self.d_model)]);
            let preactivations_cell =
                preactivations.slice(s![.., (2 * self.d_model)..(3 * self.d_model)]);
            let preactivations_output = preactivations.slice(s![.., (3 * self.d_model)..]);

            let gates = self.cache["gates"].slice(s![t, .., ..]);
            let forget_gate = gates.slice(s![.., 0..self.d_model]);
            let input_gate = gates.slice(s![.., self.d_model..(2 * self.d_model)]);
            let cell_gate = gates.slice(s![.., (2 * self.d_model)..(3 * self.d_model)]);
            let output_gate = gates.slice(s![.., (3 * self.d_model)..]);

            let cell = self.cache["cells"].slice(s![t, .., ..]);

            // recompute cell next, used in creating output state
            let cell_next = &cell * &forget_gate + (&input_gate * &cell_gate);

            // calculate dz for output gate, will use later
            let d_output_gate = &d_loss_t * &f::tanh(&cell_next);
            let d_output_dz = &d_output_gate * &f::d_sigmoid(&preactivations_output.to_owned());

            // differentiate through output update to cell update
            let d_cell = &d_loss_t * &f::d_tanh(&cell_next) * &output_gate + &cell_resid;
            cell_resid = &d_cell * &forget_gate;

            // calculate dz for forget gate
            let d_forget_gate = &d_cell * &cell;
            let d_forget_dz = &d_forget_gate * &f::d_sigmoid(&preactivations_forget.to_owned());

            // calculate dz for input gate
            let d_input_gate = &d_cell * &cell_gate;
            let d_input_dz = &d_input_gate * &f::d_sigmoid(&preactivations_input.to_owned());

            // calculate dz for cell gate
            let d_cell_gate = &d_cell * &input_gate;
            let d_cell_dz = &d_cell_gate * &f::d_tanh(&preactivations_cell.to_owned());

            // stack all gate dz

            let d_gates_dz =
                Tensor::concatenate(1, &[&d_forget_dz, &d_input_dz, &d_cell_dz, &d_output_dz]);

            // calculate gradients for w_i, w_r, b

            self.grads.accumulate("d_wi", x.t().dot(&d_gates_dz));
            self.grads.accumulate("d_wr", state.t().dot(&d_gates_dz));
            self.grads.accumulate("d_b", d_gates_dz.sum_axis(0));

            let d_x_t = &d_gates_dz.dot(&self.w_i.t());
            d_x.slice_assign(s![.., t, ..], &d_x_t);

            resid = d_gates_dz.dot(&self.w_r.t());
        }

        d_x
    }

    pub fn step(&self, x: &Tensor2, h: &mut Tensor2, cell: &mut Tensor2) {
        assert!(
            self.d_model == x.dim2().1 && x.dim2() == h.dim2() && h.dim2() == cell.dim2(),
            "dimension mismatch, d_model={} d_x={:?} d_h={:?} d_cell={:?}",
            self.d_model,
            x.dim2(),
            h.dim2(),
            cell.dim2()
        );

        let x_i = x.dot(&self.w_i);
        let r = h.dot(&self.w_r);

        let preactivatons = &x_i + &r + &self.b;

        let forget_gate = f::sigmoid(&preactivatons.slice(s![.., 0..self.d_model]).to_owned());
        let input_gate = f::sigmoid(
            &preactivatons
                .slice(s![.., self.d_model..(2 * self.d_model)])
                .to_owned(),
        );
        let cell_gate = f::tanh(
            &preactivatons
                .slice(s![.., (2 * self.d_model)..(3 * self.d_model)])
                .to_owned(),
        );
        let output_gate = f::sigmoid(&preactivatons.slice(s![.., (3 * self.d_model)..]).to_owned());

        *cell = &(*cell) * &forget_gate + (&input_gate * &cell_gate);
        *h = &output_gate * f::tanh(&cell);
    }
}

impl ToParams for LSTM {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_i).with_grad(&mut self.grads["d_wi"]),
            Param::new(&mut self.w_r).with_grad(&mut self.grads["d_wr"]),
            Param::new(&mut self.b).with_grad(&mut self.grads["d_b"]),
        ]
    }
}

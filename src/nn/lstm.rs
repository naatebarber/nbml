use ndarray::{Array1, Array2, Array3, Axis, concatenate, s};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct LSTMCache {
    pub x: Array3<f32>,
    pub states: Array3<f32>,
    pub preactivations: Array3<f32>,
    pub gates: Array3<f32>,
    pub cells: Array3<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct LSTMGrads {
    pub d_wi: Array2<f32>,
    pub d_wr: Array2<f32>,
    pub d_b: Array1<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LSTM {
    pub d_model: usize,

    pub w_i: Array2<f32>,
    pub w_r: Array2<f32>,
    pub b: Array1<f32>,

    #[serde(skip)]
    pub cache: LSTMCache,
    #[serde(skip)]
    pub grads: LSTMGrads,
}

impl LSTM {
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,

            w_i: f::xavier((d_model, 4 * d_model)),
            w_r: f::xavier((d_model, 4 * d_model)),
            b: concatenate![
                Axis(0),
                Array1::ones(d_model).view(),
                Array1::zeros(d_model * 3).view()
            ],

            cache: LSTMCache::default(),
            grads: LSTMGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array3<f32>, grad: bool) -> Array3<f32> {
        let (batch_size, seq_len, features) = x.dim();

        assert!(features == self.d_model, "feature dimension != d_model");

        let mut state = Array2::zeros((batch_size, features));
        let mut cell = Array2::zeros((batch_size, features));
        let mut output = Array3::zeros(x.dim());

        if grad {
            self.cache.x = Array3::zeros((seq_len, batch_size, features));
            self.cache.states = Array3::zeros((seq_len, batch_size, features));
            self.cache.preactivations = Array3::zeros((seq_len, batch_size, 4 * features));
            self.cache.gates = Array3::zeros((seq_len, batch_size, 4 * features));
            self.cache.cells = Array3::zeros((seq_len, batch_size, features));
        }

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);
            let x_i = x_t.dot(&self.w_i);
            let r = state.dot(&self.w_r);

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
            let output_gate =
                f::sigmoid(&preactivatons.slice(s![.., (3 * self.d_model)..]).to_owned());

            if grad {
                self.cache.x.slice_mut(s![t, .., ..]).assign(&x_t);
                self.cache.states.slice_mut(s![t, .., ..]).assign(&state);
                self.cache
                    .preactivations
                    .slice_mut(s![t, .., ..])
                    .assign(&preactivatons);
                let gates = concatenate![
                    Axis(1),
                    forget_gate.view(),
                    input_gate.view(),
                    cell_gate.view(),
                    output_gate.view()
                ];
                self.cache.gates.slice_mut(s![t, .., ..]).assign(&gates);
                self.cache.cells.slice_mut(s![t, .., ..]).assign(&cell);
            }

            cell = &cell * &forget_gate + (&input_gate * &cell_gate);
            state = &output_gate * f::tanh(&cell);

            output.slice_mut(s![.., t, ..]).assign(&state);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, features) = d_loss.dim();

        if self.grads.d_wi.dim() == (0, 0) {
            self.grads.d_wi = Array2::zeros(self.w_i.dim());
        }

        if self.grads.d_wr.dim() == (0, 0) {
            self.grads.d_wr = Array2::zeros(self.w_r.dim());
        }

        if self.grads.d_b.dim() == 0 {
            self.grads.d_b = Array1::zeros(self.b.dim());
        }

        let mut d_x = Array3::zeros(d_loss.dim());
        let mut resid = Array2::zeros((batch_size, features));
        let mut cell_resid = Array2::zeros((batch_size, features));

        for t in (0..seq_len).rev() {
            let d_loss_t = &d_loss.slice(s![.., t, ..]) + &resid;

            let x = self.cache.x.slice(s![t, .., ..]);
            let state = self.cache.states.slice(s![t, .., ..]);
            let preactivations = self.cache.preactivations.slice(s![t, .., ..]);
            let preactivations_forget = preactivations.slice(s![.., 0..self.d_model]);
            let preactivations_input =
                preactivations.slice(s![.., self.d_model..(2 * self.d_model)]);
            let preactivations_cell =
                preactivations.slice(s![.., (2 * self.d_model)..(3 * self.d_model)]);
            let preactivations_output = preactivations.slice(s![.., (3 * self.d_model)..]);

            let gates = self.cache.gates.slice(s![t, .., ..]);
            let forget_gate = gates.slice(s![.., 0..self.d_model]);
            let input_gate = gates.slice(s![.., self.d_model..(2 * self.d_model)]);
            let cell_gate = gates.slice(s![.., (2 * self.d_model)..(3 * self.d_model)]);
            let output_gate = gates.slice(s![.., (3 * self.d_model)..]);

            let cell = self.cache.cells.slice(s![t, .., ..]);

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
            let d_gates_dz = concatenate![
                Axis(1),
                d_forget_dz.view(),
                d_input_dz.view(),
                d_cell_dz.view(),
                d_output_dz.view()
            ];

            // calculate gradients for w_i, w_r, b
            self.grads.d_wi += &x.t().dot(&d_gates_dz);
            self.grads.d_wr += &state.t().dot(&d_gates_dz);
            self.grads.d_b += &d_gates_dz.sum_axis(Axis(0));

            let d_x_t = &d_gates_dz.dot(&self.w_i.t());
            d_x.slice_mut(s![.., t, ..]).assign(&d_x_t);

            resid = d_gates_dz.dot(&self.w_r.t());
        }

        d_x
    }

    pub fn step(&self, x: &Array2<f32>, h: &mut Array2<f32>, cell: &mut Array2<f32>) {
        assert!(
            self.d_model == x.dim().1 && x.dim() == h.dim() && h.dim() == cell.dim(),
            "dimension mismatch, d_model={} d_x={:?} d_h={:?} d_cell={:?}",
            self.d_model,
            x.dim(),
            h.dim(),
            cell.dim()
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

    pub fn step_forward(&mut self, x: &Array2<f32>, h: &mut Array2<f32>, cell: &mut Array2<f32>) {
        let (batch_size, features) = x.dim();

        assert!(
            self.d_model == x.dim().1 && x.dim() == h.dim() && h.dim() == cell.dim(),
            "dimension mismatch, d_model={} d_x={:?} d_h={:?} d_cell={:?}",
            self.d_model,
            x.dim(),
            h.dim(),
            cell.dim()
        );

        if self.cache.x.dim() == (0, 0, 0) {
            self.cache.x = Array3::zeros((0, batch_size, features))
        }

        if self.cache.states.dim() == (0, 0, 0) {
            self.cache.states = Array3::zeros((0, batch_size, features))
        }

        if self.cache.preactivations.dim() == (0, 0, 0) {
            self.cache.preactivations = Array3::zeros((0, batch_size, 4 * features))
        }

        if self.cache.gates.dim() == (0, 0, 0) {
            self.cache.gates = Array3::zeros((0, batch_size, 4 * features))
        }

        if self.cache.cells.dim() == (0, 0, 0) {
            self.cache.cells = Array3::zeros((0, batch_size, features))
        }

        self.cache.x.push(Axis(0), x.view()).unwrap();
        self.cache.states.push(Axis(0), h.view()).unwrap();
        self.cache.cells.push(Axis(0), cell.view()).unwrap();

        let x_i = x.dot(&self.w_i);
        let r = h.dot(&self.w_r);
        let preactivatons = &x_i + &r + &self.b;

        self.cache
            .preactivations
            .push(Axis(0), preactivatons.view())
            .unwrap();

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

        let gates = concatenate![
            Axis(1),
            forget_gate.view(),
            input_gate.view(),
            cell_gate.view(),
            output_gate.view()
        ];
        self.cache.gates.push(Axis(0), gates.view()).unwrap();

        *cell = &(*cell) * &forget_gate + (&input_gate * &cell_gate);
        *h = &output_gate * f::tanh(&cell);
    }

    pub fn clear_intermediates(&mut self) {
        self.cache = LSTMCache::default()
    }
}

impl ToParams for LSTM {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w_i).with_grad(&mut self.grads.d_wi),
            Param::new(&mut self.w_r).with_grad(&mut self.grads.d_wr),
            Param::new(&mut self.b).with_grad(&mut self.grads.d_b),
        ]
    }
}

impl ToIntermediates for LSTM {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![
            &mut self.cache.x,
            &mut self.cache.states,
            &mut self.cache.preactivations,
            &mut self.cache.gates,
            &mut self.cache.cells,
        ]
    }
}

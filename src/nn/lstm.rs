use ndarray::{Array1, Array2, Array3, Axis, concatenate, s};

use crate::{
    f,
    optim::param::{Param, ToParams},
};

pub struct LSTM {
    d_model: usize,

    w_i: Array2<f64>,
    w_r: Array2<f64>,
    b: Array1<f64>,

    x: Array3<f64>,
    states: Array3<f64>,
    preactivations: Array3<f64>,
    gates: Array3<f64>,
    cells: Array3<f64>,

    d_wi: Array2<f64>,
    d_wr: Array2<f64>,
    d_b: Array1<f64>,
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

            x: Array3::zeros((0, 0, 0)),
            states: Array3::zeros((0, 0, 0)),
            preactivations: Array3::zeros((0, 0, 0)),
            gates: Array3::zeros((0, 0, 0)),
            cells: Array3::zeros((0, 0, 0)),

            d_wi: Array2::zeros((0, 0)),
            d_wr: Array2::zeros((0, 0)),
            d_b: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, x: Array3<f64>, grad: bool) -> Array3<f64> {
        let (batch_size, seq_len, features) = x.dim();

        assert!(features == self.d_model, "feature dimension != d_model");

        let mut state = Array2::zeros((batch_size, features));
        let mut cell = Array2::zeros((batch_size, features));
        let mut output = Array3::zeros(x.dim());

        self.x = if grad {
            Array3::zeros((seq_len, batch_size, features))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.states = if grad {
            Array3::zeros((seq_len, batch_size, features))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.preactivations = if grad {
            Array3::zeros((seq_len, batch_size, 4 * features))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.gates = if grad {
            Array3::zeros((seq_len, batch_size, 4 * features))
        } else {
            Array3::zeros((0, 0, 0))
        };

        self.cells = if grad {
            Array3::zeros((seq_len, batch_size, features))
        } else {
            Array3::zeros((0, 0, 0))
        };

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
                self.x.slice_mut(s![t, .., ..]).assign(&x_t);
                self.states.slice_mut(s![t, .., ..]).assign(&state);
                self.preactivations
                    .slice_mut(s![t, .., ..])
                    .assign(&preactivatons);
                let gates = concatenate![
                    Axis(1),
                    forget_gate.view(),
                    input_gate.view(),
                    cell_gate.view(),
                    output_gate.view()
                ];
                self.gates.slice_mut(s![t, .., ..]).assign(&gates);
                self.cells.slice_mut(s![t, .., ..]).assign(&cell);
            }

            cell = &cell * &forget_gate + (&input_gate * &cell_gate);
            state = &output_gate * f::tanh(&cell);

            output.slice_mut(s![.., t, ..]).assign(&state);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, features) = d_loss.dim();

        if self.d_wi.dim() == (0, 0) {
            self.d_wi = Array2::zeros(self.w_i.dim());
        }

        if self.d_wr.dim() == (0, 0) {
            self.d_wr = Array2::zeros(self.w_r.dim());
        }

        if self.d_b.dim() == 0 {
            self.d_b = Array1::zeros(self.b.dim());
        }

        let mut d_x = Array3::zeros(d_loss.dim());
        let mut resid = Array2::zeros((batch_size, features));
        let mut cell_resid = Array2::zeros((batch_size, features));

        for t in (0..seq_len).rev() {
            let d_loss_t = &d_loss.slice(s![.., t, ..]) + &resid;

            let x = self.x.slice(s![t, .., ..]);
            let state = self.states.slice(s![t, .., ..]);
            let preactivations = self.preactivations.slice(s![t, .., ..]);
            let preactivations_forget = preactivations.slice(s![.., 0..self.d_model]);
            let preactivations_input =
                preactivations.slice(s![.., self.d_model..(2 * self.d_model)]);
            let preactivations_cell =
                preactivations.slice(s![.., (2 * self.d_model)..(3 * self.d_model)]);
            let preactivations_output = preactivations.slice(s![.., (3 * self.d_model)..]);

            let gates = self.gates.slice(s![t, .., ..]);
            let forget_gate = gates.slice(s![.., 0..self.d_model]);
            let input_gate = gates.slice(s![.., self.d_model..(2 * self.d_model)]);
            let cell_gate = gates.slice(s![.., (2 * self.d_model)..(3 * self.d_model)]);
            let output_gate = gates.slice(s![.., (3 * self.d_model)..]);

            let cell = self.cells.slice(s![t, .., ..]);

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
            self.d_wi += &x.t().dot(&d_gates_dz);
            self.d_wr += &state.t().dot(&d_gates_dz);
            self.d_b += &d_gates_dz.sum_axis(Axis(0));

            let d_x_t = &d_gates_dz.dot(&self.w_i.t());
            d_x.slice_mut(s![.., t, ..]).assign(&d_x_t);

            resid = d_gates_dz.dot(&self.w_r.t());
        }

        d_x
    }

    pub fn scan(&self, x: &Array3<f64>) -> (Array3<f64>, Array3<f64>) {
        let (batch_size, seq_len, features) = x.dim();

        assert!(
            features == self.d_model,
            "dimension mismatch, d_model={} d_x={:?}",
            self.d_model,
            x.dim()
        );

        let mut state = Array2::zeros((batch_size, features));
        let mut cell = Array2::zeros((batch_size, features));
        let mut output = Array3::zeros(x.dim());
        let mut cell_output = Array3::zeros(x.dim());

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

            cell = &cell * &forget_gate + (&input_gate * &cell_gate);
            state = &output_gate * f::tanh(&cell);

            output.slice_mut(s![.., t, ..]).assign(&state);
            cell_output.slice_mut(s![.., t, ..]).assign(&cell);
        }

        (output, cell_output)
    }

    pub fn step(&self, x: &Array2<f64>, h: &mut Array2<f64>, cell: &mut Array2<f64>) {
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

    pub fn step_forward(&mut self, x: &Array2<f64>, h: &mut Array2<f64>, cell: &mut Array2<f64>) {
        let (batch_size, features) = x.dim();

        assert!(
            self.d_model == x.dim().1 && x.dim() == h.dim() && h.dim() == cell.dim(),
            "dimension mismatch, d_model={} d_x={:?} d_h={:?} d_cell={:?}",
            self.d_model,
            x.dim(),
            h.dim(),
            cell.dim()
        );

        if self.x.dim() == (0, 0, 0) {
            self.x = Array3::zeros((0, batch_size, features))
        }

        if self.states.dim() == (0, 0, 0) {
            self.states = Array3::zeros((0, batch_size, features))
        }

        if self.preactivations.dim() == (0, 0, 0) {
            self.preactivations = Array3::zeros((0, batch_size, 4 * features))
        }

        if self.gates.dim() == (0, 0, 0) {
            self.gates = Array3::zeros((0, batch_size, 4 * features))
        }

        if self.cells.dim() == (0, 0, 0) {
            self.cells = Array3::zeros((0, batch_size, features))
        }

        self.x.push(Axis(0), x.view()).unwrap();
        self.states.push(Axis(0), h.view()).unwrap();
        self.cells.push(Axis(0), cell.view()).unwrap();

        let x_i = x.dot(&self.w_i);
        let r = h.dot(&self.w_r);
        let preactivatons = &x_i + &r + &self.b;

        self.preactivations
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
        self.gates.push(Axis(0), gates.view()).unwrap();

        *cell = &(*cell) * &forget_gate + (&input_gate * &cell_gate);
        *h = &output_gate * f::tanh(&cell);
    }
}

impl ToParams for LSTM {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::matrix(&mut self.w_i).with_matrix_grad(&mut self.d_wi),
            Param::matrix(&mut self.w_r).with_matrix_grad(&mut self.d_wr),
            Param::vector(&mut self.b).with_vector_grad(&mut self.d_b),
        ]
    }
}

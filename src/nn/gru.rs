use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, concatenate, s};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToIntermediates, ToParams},
};

#[derive(Default, Debug, Clone)]
pub struct GRUCache {
    pub x: Array3<f32>,
    pub ru_preactivations: Array3<f32>,
    pub reset_gates: Array3<f32>,
    pub update_gates: Array3<f32>,
    pub reset_states: Array3<f32>,
    pub c_preactivations: Array3<f32>,
    pub c: Array3<f32>,
    pub states: Array3<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct GRUGrads {
    pub d_w_ru: Array2<f32>,
    pub d_b_ru: Array1<f32>,
    pub d_w_c: Array2<f32>,
    pub d_b_c: Array1<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GRU {
    pub d_in: usize,
    pub d_model: usize,

    pub w_ru: Array2<f32>,
    pub b_ru: Array1<f32>,
    pub w_c: Array2<f32>,
    pub b_c: Array1<f32>,

    #[serde(skip)]
    pub cache: GRUCache,
    #[serde(skip)]
    pub grads: GRUGrads,
}

impl GRU {
    pub fn new(d_in: usize, d_model: usize) -> Self {
        Self {
            d_in,
            d_model,

            w_ru: f::xavier_normal((d_in + d_model, 2 * d_model)),
            b_ru: Array1::zeros(2 * d_model),
            w_c: f::xavier_normal((d_in + d_model, d_model)),
            b_c: Array1::zeros(d_model),

            cache: GRUCache::default(),
            grads: GRUGrads::default(),
        }
    }

    pub fn forward(&mut self, x: Array3<f32>, grad: bool) -> Array3<f32> {
        let (batch_size, seq_len, _) = x.dim();

        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));
        let mut state = Array2::zeros((batch_size, self.d_model));

        if grad {
            self.cache.x = x.clone();
            self.cache.ru_preactivations = Array3::zeros((batch_size, seq_len, 2 * self.d_model));
            self.cache.reset_gates = Array3::zeros((batch_size, seq_len, self.d_model));
            self.cache.update_gates = Array3::zeros((batch_size, seq_len, self.d_model));
            self.cache.reset_states = Array3::zeros((batch_size, seq_len, self.d_model));
            self.cache.c_preactivations = Array3::zeros((batch_size, seq_len, self.d_model));
            self.cache.c = Array3::zeros((batch_size, seq_len, self.d_model));
            self.cache.states = Array3::zeros((batch_size, 1 + seq_len, self.d_model));
        }

        for t in 0..seq_len {
            let x_t = x.slice(s![.., t, ..]);

            let x_state = concatenate![Axis(1), x_t.view(), state.view()];
            let ru_preactivations = x_state.dot(&self.w_ru) + &self.b_ru;

            let reset_gate = f::sigmoid(&ru_preactivations.slice(s![.., 0..self.d_model]));
            let update_gate = f::sigmoid(&ru_preactivations.slice(s![.., self.d_model..]));

            let reset_state = &state * &reset_gate;
            let x_reset_state = concatenate![Axis(1), x_t.view(), reset_state.view()];
            let c_preactivations = x_reset_state.dot(&self.w_c) + &self.b_c;
            let c = f::tanh(&c_preactivations);

            state = &update_gate * &c + &(1. - &update_gate) * &state;

            if grad {
                self.cache
                    .ru_preactivations
                    .slice_mut(s![.., t, ..])
                    .assign(&ru_preactivations);
                self.cache
                    .reset_gates
                    .slice_mut(s![.., t, ..])
                    .assign(&reset_gate);
                self.cache
                    .update_gates
                    .slice_mut(s![.., t, ..])
                    .assign(&update_gate);
                self.cache
                    .reset_states
                    .slice_mut(s![.., t, ..])
                    .assign(&reset_state);
                self.cache
                    .c_preactivations
                    .slice_mut(s![.., t, ..])
                    .assign(&c_preactivations);
                self.cache.c.slice_mut(s![.., t, ..]).assign(&c);
                self.cache
                    .states
                    .slice_mut(s![.., t + 1, ..])
                    .assign(&state);
            }

            output.slice_mut(s![.., t, ..]).assign(&state);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, _) = d_loss.dim();

        if self.grads.d_w_ru.dim() == (0, 0) {
            self.grads.d_w_ru = Array2::zeros(self.w_ru.dim());
        }

        if self.grads.d_b_ru.dim() == 0 {
            self.grads.d_b_ru = Array1::zeros(self.b_ru.dim());
        }

        if self.grads.d_w_c.dim() == (0, 0) {
            self.grads.d_w_c = Array2::zeros(self.w_c.dim());
        }

        if self.grads.d_b_c.dim() == 0 {
            self.grads.d_b_c = Array1::zeros(self.b_c.dim());
        }

        let mut d_x = Array3::zeros((batch_size, seq_len, self.d_in));
        let mut d_resid = Array2::zeros((batch_size, self.d_model));

        for t in (0..seq_len).rev() {
            let d_loss_t: Array2<f32> = &d_loss.slice(s![.., t, ..]) + &d_resid;
            let update_gate_t: ArrayView2<f32> = self.cache.update_gates.slice(s![.., t, ..]);
            let reset_gate_t: ArrayView2<f32> = self.cache.reset_gates.slice(s![.., t, ..]);
            let c_t: ArrayView2<f32> = self.cache.c.slice(s![.., t, ..]);
            let reset_state_t: ArrayView2<f32> = self.cache.reset_states.slice(s![.., t, ..]);
            let c_preactivations_t: ArrayView2<f32> =
                self.cache.c_preactivations.slice(s![.., t, ..]);
            let ru_preactivations_t: ArrayView2<f32> =
                self.cache.ru_preactivations.slice(s![.., t, ..]);
            let x_t: ArrayView2<f32> = self.cache.x.slice(s![.., t, ..]);
            let state_prev: ArrayView2<f32> = self.cache.states.slice(s![.., t, ..]);

            let reset_preactivations: ArrayView2<f32> =
                ru_preactivations_t.slice(s![.., 0..self.d_model]);
            let update_preactivations: ArrayView2<f32> =
                ru_preactivations_t.slice(s![.., self.d_model..]);

            // get d_update_gate from side of c
            let d_update_gate_c = &c_t * &d_loss_t;

            // get d_c
            let d_c = &update_gate_t * &d_loss_t;

            // get grads of w_c and b_c
            let d_c_dz = &d_c * f::d_tanh(&c_preactivations_t);

            let x_reset_state = concatenate![Axis(1), x_t.view(), reset_state_t.view()];
            self.grads.d_w_c += &(x_reset_state.t().dot(&d_c_dz));
            self.grads.d_b_c += &(d_c_dz.sum_axis(Axis(0)));

            // get d_x_c_gate and d_reset_state_c_gate, for x and reset gate which contributed to C
            let d_c_gate = d_c_dz.dot(&self.w_c.t());
            let d_x_c_gate = d_c_gate.slice(s![.., 0..self.d_in]);
            let d_reset_state_c_gate = d_c_gate.slice(s![.., self.d_in..]);

            // differentiate through reset gate
            let d_state_c_reset = &d_reset_state_c_gate * &reset_gate_t;
            let d_reset_gate = &d_reset_state_c_gate * &state_prev;
            let d_reset_gate_dz = &d_reset_gate * f::d_sigmoid(&reset_preactivations);

            // pause here since reset gate is ready for differentiation, differentiate update gate
            // w.r.t. prev state

            let d_update_gate_state = -1. * &state_prev * &d_loss_t;
            let d_state_update_gate = &(1. - &update_gate_t) * &d_loss_t;

            // collect and both update gate gradients

            let d_update_gate = &d_update_gate_c + &d_update_gate_state;
            let d_update_gate_dz = &d_update_gate * f::d_sigmoid(&update_preactivations);

            // concatenate both d_update_gate and d_reset_gate, get gradients for w_ru and b_ru

            let d_ru = concatenate![Axis(1), d_reset_gate_dz.view(), d_update_gate_dz.view()];
            let x_state = concatenate![Axis(1), x_t.view(), state_prev.view()];

            self.grads.d_w_ru += &(x_state.t().dot(&d_ru));
            self.grads.d_b_ru += &(d_ru.sum_axis(Axis(0)));

            // backpropagate through w_ru

            let d_ru_gate = d_ru.dot(&self.w_ru.t());
            let d_x_ru = d_ru_gate.slice(s![.., 0..self.d_in]);
            let d_state_ru = d_ru_gate.slice(s![.., self.d_in..]);

            // collect x gradients and state_prev gradients

            let d_x_t = &d_x_ru + &d_x_c_gate;
            let d_state = &d_state_c_reset + &d_state_update_gate + &d_state_ru;

            d_x.slice_mut(s![.., t, ..]).assign(&d_x_t);
            d_resid = d_state;
        }

        d_x
    }

    pub fn step(&self, x: &Array2<f32>, state: &mut Array2<f32>) {
        let x_state = concatenate![Axis(1), x.view(), state.view()];
        let ru_preactivations = x_state.dot(&self.w_ru) + &self.b_ru;

        let reset_gate = f::sigmoid(&ru_preactivations.slice(s![.., 0..self.d_model]));
        let update_gate = f::sigmoid(&ru_preactivations.slice(s![.., self.d_model..]));

        let reset_state = &(*state) * &reset_gate;
        let x_reset_state = concatenate![Axis(1), x.view(), reset_state.view()];
        let c_preactivations = x_reset_state.dot(&self.w_c) + &self.b_c;
        let c = f::tanh(&c_preactivations);

        *state = &update_gate * &c + &(1. - &update_gate) * &(*state);
    }
}

impl ToParams for GRU {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        vec![
            Param::new(&mut self.w_ru).with_grad(&mut self.grads.d_w_ru),
            Param::new(&mut self.b_ru).with_grad(&mut self.grads.d_b_ru),
            Param::new(&mut self.w_c).with_grad(&mut self.grads.d_w_c),
            Param::new(&mut self.b_c).with_grad(&mut self.grads.d_b_c),
        ]
    }
}

impl ToIntermediates for GRU {
    fn intermediates(&mut self) -> Vec<&mut dyn crate::optim::Intermediate> {
        vec![
            &mut self.cache.x,
            &mut self.cache.ru_preactivations,
            &mut self.cache.reset_gates,
            &mut self.cache.update_gates,
            &mut self.cache.reset_states,
            &mut self.cache.c_preactivations,
            &mut self.cache.c,
            &mut self.cache.states,
        ]
    }
}

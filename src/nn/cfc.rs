use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

use crate::{
    f,
    optim::{Param, ToParams},
};

// Closed-form Continuous-time neural network

#[derive(Clone, Debug, Default)]
pub struct CfCCache {
    pub x_2d: Array2<f32>,
    pub z_pre: Array3<f32>,
    pub z: Array3<f32>,
    pub fgc_pre: Array3<f32>,
    pub dt: Array3<f32>,
    pub f: Array3<f32>,
    pub c: Array3<f32>,
    pub tau_exp: Array3<f32>,
    pub h: Array3<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct CfCGrads {
    pub d_zx_w: Array2<f32>,
    pub d_zx_b: Array1<f32>,
    pub d_zh_w: Array2<f32>,
    pub d_fgc_w: Array2<f32>,
    pub d_fgc_b: Array1<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CfC {
    pub d_in: usize,
    pub d_backbone: usize,
    pub d_model: usize,

    pub zx_w: Array2<f32>,
    pub zx_b: Array1<f32>,
    pub zh_w: Array2<f32>,
    pub fgc_w: Array2<f32>,
    pub fgc_b: Array1<f32>,

    #[serde(skip)]
    pub cache: CfCCache,
    #[serde(skip)]
    pub grads: CfCGrads,
}

impl CfC {
    pub fn new(d_in: usize, d_backbone: usize, d_model: usize) -> Self {
        Self {
            d_in,
            d_backbone,
            d_model,

            zx_w: f::xavier_normal((d_in, d_backbone)),
            zx_b: Array1::zeros(d_backbone),
            zh_w: f::xavier_normal((d_model, d_backbone)),
            fgc_w: f::xavier_normal((d_backbone, 3 * d_model)),
            fgc_b: Array1::zeros(3 * d_model),

            cache: CfCCache::default(),
            grads: CfCGrads::default(),
        }
    }

    /// ∆t must be >= 0
    /// ∆t normalization will help smoothify the gradient
    pub fn forward(&mut self, x: Array3<f32>, dt: Array3<f32>, grad: bool) -> Array3<f32> {
        debug_assert!(
            x.dim().0 == dt.dim().0 && x.dim().1 == dt.dim().1 && dt.dim().2 == 1,
            "dt should be (x.0, x.1, 1)"
        );

        let (b, s, _) = x.dim();

        let mut h = Array3::zeros((b, s + 1, self.d_model));
        let x_2d = x.into_shape_clone((b * s, self.d_in)).unwrap();

        if grad {
            self.cache.x_2d = x_2d.clone();
            self.cache.dt = dt.clone();
        }

        let zx_pre_2d = x_2d.dot(&self.zx_w) + &self.zx_b;
        let zx_pre = zx_pre_2d.into_shape_clone((b, s, self.d_backbone)).unwrap();

        if grad {
            self.cache.z_pre = Array3::zeros((b, s, self.d_backbone));
            self.cache.z = Array3::zeros((b, s, self.d_backbone));
            self.cache.fgc_pre = Array3::zeros((b, s, 3 * self.d_model));
            self.cache.f = Array3::zeros((b, s, self.d_model));
            self.cache.c = Array3::zeros((b, s, self.d_model));
            self.cache.tau_exp = Array3::zeros((b, s, self.d_model));
        }

        for t in 0..s {
            let dt_t: ArrayView2<f32> = dt.slice(ndarray::s![.., t, ..]);

            let h_prev: ArrayView2<f32> = h.slice(ndarray::s![.., t, ..]);
            let zh_pre_t = h_prev.dot(&self.zh_w);
            let zx_pre_t: ArrayView2<f32> = zx_pre.slice(ndarray::s![.., t, ..]);

            let z_pre_t = &zx_pre_t + &zh_pre_t;
            let z_t = f::tanh(&z_pre_t);

            if grad {
                self.cache
                    .z_pre
                    .slice_mut(ndarray::s![.., t, ..])
                    .assign(&z_pre_t);
                self.cache.z.slice_mut(ndarray::s![.., t, ..]).assign(&z_t);
            }

            let fgc_pre_t = z_t.dot(&self.fgc_w) + &self.fgc_b;
            let f_pre: ArrayView2<f32> = fgc_pre_t.slice(ndarray::s![.., 0..self.d_model]);
            let g_pre: ArrayView2<f32> =
                fgc_pre_t.slice(ndarray::s![.., self.d_model..(2 * self.d_model)]);
            let c_pre: ArrayView2<f32> = fgc_pre_t.slice(ndarray::s![.., (2 * self.d_model)..]);

            let f_t = f::tanh(&f_pre);
            let g_t = f::softplus(&g_pre);
            let c_t = f::tanh(&c_pre);

            let tau = -1. * &g_t * &dt_t;
            let tau_exp = tau.exp();

            let h_next = &f_t * &tau_exp + &c_t * (1. - &tau_exp);
            h.slice_mut(ndarray::s![.., t + 1, ..]).assign(&h_next);

            if grad {
                self.cache
                    .fgc_pre
                    .slice_mut(ndarray::s![.., t, ..])
                    .assign(&fgc_pre_t);
                self.cache.f.slice_mut(ndarray::s![.., t, ..]).assign(&f_t);
                self.cache.c.slice_mut(ndarray::s![.., t, ..]).assign(&c_t);
                self.cache
                    .tau_exp
                    .slice_mut(ndarray::s![.., t, ..])
                    .assign(&tau_exp);
            }
        }

        if grad {
            self.cache.h = h.clone();
        }

        h.slice(ndarray::s![.., 1.., ..]).to_owned()
    }

    pub fn backward(&mut self, d_loss: Array3<f32>) -> Array3<f32> {
        let (b, s, _) = d_loss.dim();

        if self.grads.d_zx_w.dim() == (0, 0) {
            self.grads.d_zx_w = Array2::zeros(self.zx_w.dim())
        }

        if self.grads.d_zx_b.dim() == 0 {
            self.grads.d_zx_b = Array1::zeros(self.zx_b.dim())
        }

        if self.grads.d_zh_w.dim() == (0, 0) {
            self.grads.d_zh_w = Array2::zeros(self.zh_w.dim())
        }

        if self.grads.d_fgc_w.dim() == (0, 0) {
            self.grads.d_fgc_w = Array2::zeros(self.fgc_w.dim())
        }

        if self.grads.d_fgc_b.dim() == 0 {
            self.grads.d_fgc_b = Array1::zeros(self.fgc_b.dim())
        }

        let mut d_loss_resid = Array2::zeros((b, self.d_model));
        let mut d_zx_pre = Array3::zeros((b, s, self.d_backbone));

        for t in (0..s).rev() {
            let d_loss_t = &d_loss.slice(ndarray::s![.., t, ..]) + &d_loss_resid;

            let c_t: ArrayView2<f32> = self.cache.c.slice(ndarray::s![.., t, ..]);
            let f_t: ArrayView2<f32> = self.cache.f.slice(ndarray::s![.., t, ..]);
            let tau_exp: ArrayView2<f32> = self.cache.tau_exp.slice(ndarray::s![.., t, ..]);
            let dt_t: ArrayView2<f32> = self.cache.dt.slice(ndarray::s![.., t, ..]);

            let d_tau_exp_c = -1. * &c_t * &d_loss_t;
            let d_tau_exp_f = &f_t * &d_loss_t;
            let d_tau_exp = &d_tau_exp_f + &d_tau_exp_c;
            let d_g_t = &d_tau_exp * &tau_exp * -1. * &dt_t;

            let d_f_t = &d_loss_t * &tau_exp;
            let d_c_t = &d_loss_t - &d_loss_t * &tau_exp;

            let f_pre: ArrayView2<f32> =
                self.cache
                    .fgc_pre
                    .slice(ndarray::s![.., t, 0..self.d_model]);
            let g_pre: ArrayView2<f32> =
                self.cache
                    .fgc_pre
                    .slice(ndarray::s![.., t, self.d_model..(2 * self.d_model)]);
            let c_pre: ArrayView2<f32> =
                self.cache
                    .fgc_pre
                    .slice(ndarray::s![.., t, (2 * self.d_model)..]);

            let d_f_pre = &d_f_t * f::d_tanh(&f_pre);
            let d_g_pre = &d_g_t * f::d_softplus(&g_pre);
            let d_c_pre = &d_c_t * f::d_tanh(&c_pre);

            let d_fgc_pre: Array2<f32> =
                ndarray::concatenate(Axis(1), &[d_f_pre.view(), d_g_pre.view(), d_c_pre.view()])
                    .unwrap();
            let z_t: ArrayView2<f32> = self.cache.z.slice(ndarray::s![.., t, ..]);
            self.grads.d_fgc_w += &(z_t.t().dot(&d_fgc_pre));
            self.grads.d_fgc_b += &(d_fgc_pre.sum_axis(Axis(0)));

            let d_z_t = d_fgc_pre.dot(&self.fgc_w.t());
            let z_pre_t: ArrayView2<f32> = self.cache.z_pre.slice(ndarray::s![.., t, ..]);
            let d_z_pre = &d_z_t * f::d_tanh(&z_pre_t);

            d_zx_pre.slice_mut(ndarray::s![.., t, ..]).assign(&d_z_pre);

            let h_prev: ArrayView2<f32> = self.cache.h.slice(ndarray::s![.., t, ..]);
            self.grads.d_zh_w += &(h_prev.t().dot(&d_z_pre));
            d_loss_resid = d_z_pre.dot(&self.zh_w.t());
        }

        let d_zx_pre_2d = d_zx_pre.into_shape_clone((b * s, self.d_backbone)).unwrap();
        self.grads.d_zx_w += &(self.cache.x_2d.t().dot(&d_zx_pre_2d));
        self.grads.d_zx_b += &(d_zx_pre_2d.sum_axis(Axis(0)));

        d_zx_pre_2d
            .dot(&self.zx_w.t())
            .into_shape_clone((b, s, self.d_in))
            .unwrap()
    }

    pub fn step(&self, x: &Array2<f32>, dt: &Array2<f32>, h: &mut Array2<f32>) {
        let zx = x.dot(&self.zx_w) + &self.zx_b;
        let zh = h.dot(&self.zh_w);
        let z_pre = &zx + &zh;
        let z = f::tanh(&z_pre);

        let fgc_pre = z.dot(&self.fgc_w) + &self.fgc_b;
        let _f = f::tanh(&fgc_pre.slice(ndarray::s![.., 0..self.d_model]));
        let _g = f::softplus(&fgc_pre.slice(ndarray::s![.., self.d_model..(2 * self.d_model)]));
        let _c = f::tanh(&fgc_pre.slice(ndarray::s![.., (2 * self.d_model)..]));

        let tau = -1. * &_g * dt;
        let tau_exp = tau.exp();

        *h = &_f * &tau_exp + &_c * (1. - tau_exp);
    }
}

impl ToParams for CfC {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        vec![
            Param::new(&mut self.zx_w).with_grad(&mut self.grads.d_zx_w),
            Param::new(&mut self.zx_b).with_grad(&mut self.grads.d_zx_b),
            Param::new(&mut self.zh_w).with_grad(&mut self.grads.d_zh_w),
            Param::new(&mut self.fgc_w).with_grad(&mut self.grads.d_fgc_w),
            Param::new(&mut self.fgc_b).with_grad(&mut self.grads.d_fgc_b),
        ]
    }
}

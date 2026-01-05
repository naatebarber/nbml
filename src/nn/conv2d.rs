use ndarray::{Array1, Array2, Array4, Axis, s};

use crate::{
    f::InitializationFn,
    optim::param::{Param, ToParams},
};

pub struct Conv2D {
    pub channels_in: usize,
    pub channels_out: usize,
    pub k_h: usize,
    pub k_w: usize,

    pub w: Array2<f64>, // (C_in * kH * kW, C_out)
    pub b: Array1<f64>, // (C_out)

    pub stack: Array2<f64>,

    pub d_w: Array2<f64>,
    pub d_b: Array1<f64>,
}

impl Conv2D {
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        k_h: usize,
        k_w: usize,
        init: InitializationFn,
    ) -> Self {
        Self {
            channels_in,
            channels_out,
            k_h,
            k_w,

            w: init((channels_in * k_h * k_w, channels_out)),
            b: Array1::zeros(channels_out),

            stack: Array2::zeros((0, 0)),

            d_w: Array2::zeros((0, 0)),
            d_b: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, x: Array4<f64>, grad: bool) -> Array4<f64> {
        let (batch_size, channels_in, h, w) = x.dim();

        assert!(
            channels_in == self.channels_in,
            "channel dimension mismatch, model_channels={} x_channels={}",
            self.channels_in,
            channels_in
        );

        let strides_h = h - self.k_h + 1;
        let strides_w = w - self.k_w + 1;

        let mut stack = Array2::zeros((
            batch_size * strides_h * strides_w,
            channels_in * self.k_h * self.k_w,
        ));

        for i_h in 0..strides_h {
            for i_w in 0..strides_w {
                let patch = x
                    .slice(s![.., .., i_h..(i_h + self.k_h), i_w..(i_w + self.k_w)])
                    .to_owned();
                let patch_2d = patch
                    .into_shape_clone((batch_size, channels_in * self.k_h * self.k_w))
                    .unwrap();

                let stride_abs = batch_size * ((strides_w * i_h) + i_w);
                stack
                    .slice_mut(s![stride_abs..(stride_abs + batch_size), ..])
                    .assign(&patch_2d);
            }
        }

        let conv_out_2d = stack.dot(&self.w) + &self.b;
        let conv_out = conv_out_2d
            .into_shape_clone((strides_h, strides_w, batch_size, self.channels_out))
            .unwrap();
        let conv_out = conv_out.permuted_axes([2, 3, 0, 1]);

        self.stack = if grad { stack } else { Array2::zeros((0, 0)) };

        conv_out
    }

    pub fn backward(&mut self, d_loss: Array4<f64>) -> Array4<f64> {
        let (batch_size, channels_out, strides_h, strides_w) = d_loss.dim();

        let d_z = d_loss
            .permuted_axes([2, 3, 0, 1])
            .into_shape_clone((strides_h * strides_w * batch_size, channels_out))
            .unwrap();

        if self.d_w.dim() == (0, 0) {
            self.d_w = Array2::zeros(self.w.dim());
        }

        if self.d_b.dim() == 0 {
            self.d_b = Array1::zeros(self.b.dim());
        }

        self.d_w += &self.stack.t().dot(&d_z);
        self.d_b += &d_z.sum_axis(Axis(0));

        let d_stack = d_z.dot(&self.w.t()); // (batch_size * strides_h * strides_w, channels_in * k_h * k_w)

        let h = strides_h + self.k_h - 1;
        let w = strides_w + self.k_w - 1;
        let mut d_x = Array4::zeros((batch_size, self.channels_in, h, w)); // (batch_size, channels_in, h, w)

        // start with d_stack = (batch_size * strides_h * strides_w, channels * k_h * k_w)
        // split channels out
        // d_stack_1 = (batch_size * strides_h * strides_w, channels, k_h * k_w)
        // turn kernel back into matrix
        // d_stack_2 = (batch_size * strides_h * strides_w, channels, k_h, k_w)
        // split batch out
        // d_stack_3 = (batch_size, strides_h * strides_w, channels, k_h, k_w)
        //
        // now the stride index represents where the matrix needs to be added to d_x by
        // stride_tot = i in 0..d_stack_3.len()
        // offset_h = Floor(stride_tot / stride_w)
        // offset_x = stride_w % stride_tot

        let d_stack = d_stack
            .into_shape_clone((
                batch_size,
                strides_h * strides_w,
                self.channels_in,
                self.k_h,
                self.k_w,
            ))
            .unwrap();

        let strides_tot = strides_h * strides_w;

        for i in 0..strides_tot {
            let offset_h = i / strides_w;
            let offset_w = i % strides_w;

            let mut recv = d_x.slice_mut(s![
                ..,
                ..,
                offset_h..(offset_h + self.k_h),
                offset_w..(offset_w + self.k_w)
            ]);
            recv += &d_stack.slice(s![.., i, .., .., ..]);
        }

        d_x
    }
}

impl ToParams for Conv2D {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::matrix(&mut self.w).with_matrix_grad(&mut self.d_w),
            Param::vector(&mut self.b).with_vector_grad(&mut self.d_b),
        ]
    }
}

pub struct PatchwiseConv2D {
    pub channels_in: usize,
    pub channels_out: usize,
    pub k_h: usize,
    pub k_w: usize,

    pub w: Array2<f64>,
    pub b: Array1<f64>,

    pub x: Array4<f64>,

    pub d_w: Array2<f64>,
    pub d_b: Array1<f64>,
}

impl PatchwiseConv2D {
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        k_h: usize,
        k_w: usize,
        init: InitializationFn,
    ) -> Self {
        Self {
            channels_in,
            channels_out,
            k_h,
            k_w,

            w: init((channels_in * k_h * k_w, channels_out)),
            b: Array1::zeros(channels_out),

            x: Array4::zeros((0, 0, 0, 0)),

            d_w: Array2::zeros((0, 0)),
            d_b: Array1::zeros(0),
        }
    }

    pub fn forward(&mut self, x: Array4<f64>, grad: bool) -> Array4<f64> {
        let (batch_size, channels, h, w) = x.dim();
        assert!(
            channels == self.channels_in,
            "dimension mismatch, model_channels_in={} x_channels_in={}",
            self.channels_in,
            channels
        );

        let strides_h = h - self.k_h + 1;
        let strides_w = w - self.k_w + 1;

        let mut output = Array4::zeros((batch_size, self.channels_out, strides_h, strides_w));

        for i_h in 0..strides_h {
            for i_w in 0..strides_w {
                let patch = x
                    .slice(s![.., .., i_h..(i_h + self.k_h), i_w..(i_w + self.k_w)])
                    .to_owned(); // (B, C, kH, kW)
                let patch = patch
                    .into_shape_clone((batch_size, channels * self.k_h * self.k_w))
                    .unwrap();

                let k_out = patch.dot(&self.w) + &self.b;

                output.slice_mut(s![.., .., i_h, i_w]).assign(&k_out);
            }
        }

        self.x = if grad { x } else { Array4::zeros((0, 0, 0, 0)) };

        output
    }

    pub fn backward(&mut self, d_loss: Array4<f64>) -> Array4<f64> {
        let (batch_size, channels_out, strides_h, strides_w) = d_loss.dim();

        if self.d_w.dim() == (0, 0) {
            self.d_w = Array2::zeros(self.w.dim());
        }

        if self.d_b.dim() == 0 {
            self.d_b = Array1::zeros(self.b.dim());
        }

        let mut d_x = Array4::zeros(self.x.dim());

        for i_h in 0..strides_h {
            for i_w in 0..strides_w {
                let d_patch = d_loss
                    .slice(s![.., .., i_h, i_w])
                    .to_owned()
                    .into_shape_clone((batch_size, channels_out))
                    .unwrap();

                let patch = self
                    .x
                    .slice(s![.., .., i_h..(i_h + self.k_h), i_w..(i_w + self.k_w)])
                    .to_owned()
                    .into_shape_clone((batch_size, self.channels_in * self.k_h * self.k_w))
                    .unwrap();

                self.d_w += &(patch.t().dot(&d_patch));
                self.d_b += &d_patch.sum_axis(Axis(0));

                let dx_dpatch_2d = d_patch.dot(&self.w.t()); // (B, C * kh * kw)
                let dx_dpatch = dx_dpatch_2d
                    .into_shape_clone((batch_size, self.channels_in, self.k_h, self.k_w))
                    .unwrap();
                let mut dx_accum =
                    d_x.slice_mut(s![.., .., i_h..(i_h + self.k_h), i_w..(i_w + self.k_w)]);
                dx_accum += &dx_dpatch;
            }
        }

        d_x
    }
}

impl ToParams for PatchwiseConv2D {
    fn params(&mut self) -> Vec<crate::optim::param::Param> {
        vec![
            Param::matrix(&mut self.w).with_matrix_grad(&mut self.d_w),
            Param::vector(&mut self.b).with_vector_grad(&mut self.d_b),
        ]
    }
}

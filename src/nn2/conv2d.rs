use serde::{Deserialize, Serialize};

use crate::{
    f2::InitializationFn,
    optim2::{Param, ToParams},
    s,
    tensor::{Tensor1, Tensor2, Tensor4},
    util::Cache,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Conv2D {
    pub channels_in: usize,
    pub channels_out: usize,
    pub k_h: usize,
    pub k_w: usize,

    pub w: Tensor2, // (C_in * kH * kW, C_out)
    pub b: Tensor1, // (C_out)

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
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
            b: Tensor1::zeros(channels_out),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor4, grad: bool) -> Tensor4 {
        let (batch_size, channels_in, h, w) = x.dim4();

        assert!(
            channels_in == self.channels_in,
            "channel dimension mismatch, model_channels={} x_channels={}",
            self.channels_in,
            channels_in
        );

        let strides_h = h - self.k_h + 1;
        let strides_w = w - self.k_w + 1;

        let mut stack = Tensor2::zeros((
            batch_size * strides_h * strides_w,
            channels_in * self.k_h * self.k_w,
        ));

        for i_h in 0..strides_h {
            for i_w in 0..strides_w {
                let patch = x.slice(s![
                    ..,
                    ..,
                    i_h as isize..(i_h + self.k_h) as isize,
                    i_w as isize..(i_w + self.k_w) as isize
                ]);
                let patch_2d = patch.reshape((batch_size, channels_in * self.k_h * self.k_w));

                let stride_abs = batch_size * ((strides_w * i_h) + i_w);
                stack.slice_assign(
                    s![stride_abs as isize..(stride_abs + batch_size) as isize, ..],
                    &patch_2d,
                );
            }
        }

        let conv_out_2d = stack.dot(&self.w) + &self.b;
        let conv_out = conv_out_2d
            .reshape((strides_h, strides_w, batch_size, self.channels_out))
            .permute(&[2, 3, 0, 1]);

        if grad {
            self.cache.set("stack", stack);
        }

        conv_out
    }

    pub fn backward(&mut self, d_loss: Tensor4) -> Tensor4 {
        let (batch_size, channels_out, strides_h, strides_w) = d_loss.dim4();

        let d_z = d_loss
            .permute(&[2, 3, 0, 1])
            .reshape((strides_h * strides_w * batch_size, channels_out));

        let stack = &self.cache["stack"];

        self.grads.accumulate("d_w", stack.t().dot(&d_z));
        self.grads.accumulate("d_b", d_z.sum_axis(0));

        let d_stack = d_z.dot(&self.w.t());

        let h = strides_h + self.k_h - 1;
        let w = strides_w + self.k_w - 1;
        let mut d_x = Tensor4::zeros((batch_size, self.channels_in, h, w));

        // d_stack = (batch_size * strides_h * strides_w, channels_in * k_h * k_w)
        // reshape to (batch_size, strides_h * strides_w, channels_in, k_h, k_w)
        let d_stack = d_stack.reshape((
            batch_size,
            strides_h * strides_w,
            self.channels_in,
            self.k_h,
            self.k_w,
        ));

        let strides_tot = strides_h * strides_w;

        for i in 0..strides_tot {
            let offset_h = i / strides_w;
            let offset_w = i % strides_w;

            let patch = d_stack.slice(s![.., i as isize, .., .., ..]);
            d_x.slice_accumulate(
                s![
                    ..,
                    ..,
                    offset_h as isize..(offset_h + self.k_h) as isize,
                    offset_w as isize..(offset_w + self.k_w) as isize
                ],
                &patch,
            );
        }

        d_x
    }
}

impl ToParams for Conv2D {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w).with_grad(&mut self.grads["d_w"]),
            Param::new(&mut self.b).with_grad(&mut self.grads["d_b"]),
        ]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PatchwiseConv2D {
    pub channels_in: usize,
    pub channels_out: usize,
    pub k_h: usize,
    pub k_w: usize,

    pub w: Tensor2,
    pub b: Tensor1,

    #[serde(skip)]
    pub cache: Cache,
    #[serde(skip)]
    pub grads: Cache,
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
            b: Tensor1::zeros(channels_out),

            cache: Cache::new(),
            grads: Cache::new(),
        }
    }

    pub fn forward(&mut self, x: Tensor4, grad: bool) -> Tensor4 {
        let (batch_size, channels, h, w) = x.dim4();
        assert!(
            channels == self.channels_in,
            "dimension mismatch, model_channels_in={} x_channels_in={}",
            self.channels_in,
            channels
        );

        let strides_h = h - self.k_h + 1;
        let strides_w = w - self.k_w + 1;

        let mut output = Tensor4::zeros((batch_size, self.channels_out, strides_h, strides_w));

        for i_h in 0..strides_h {
            for i_w in 0..strides_w {
                let patch = x.slice(s![
                    ..,
                    ..,
                    i_h as isize..(i_h + self.k_h) as isize,
                    i_w as isize..(i_w + self.k_w) as isize
                ]);
                let patch = patch.reshape((batch_size, channels * self.k_h * self.k_w));

                let k_out = patch.dot(&self.w) + &self.b;

                output.slice_assign(s![.., .., i_h as isize, i_w as isize], &k_out);
            }
        }

        if grad {
            self.cache.set("x", x);
        }

        output
    }

    pub fn backward(&mut self, d_loss: Tensor4) -> Tensor4 {
        let (batch_size, channels_out, strides_h, strides_w) = d_loss.dim4();

        let x = &self.cache["x"];
        let x_shape = x.shape().to_vec();
        let mut d_x = Tensor4::zeros(x_shape);

        for i_h in 0..strides_h {
            for i_w in 0..strides_w {
                let d_patch = d_loss
                    .slice(s![.., .., i_h as isize, i_w as isize])
                    .reshape((batch_size, channels_out));

                let patch = x
                    .slice(s![
                        ..,
                        ..,
                        i_h as isize..(i_h + self.k_h) as isize,
                        i_w as isize..(i_w + self.k_w) as isize
                    ])
                    .reshape((batch_size, self.channels_in * self.k_h * self.k_w));

                self.grads.accumulate("d_w", patch.t().dot(&d_patch));
                self.grads.accumulate("d_b", d_patch.sum_axis(0));

                let dx_dpatch = d_patch.dot(&self.w.t()).reshape((
                    batch_size,
                    self.channels_in,
                    self.k_h,
                    self.k_w,
                ));

                d_x.slice_accumulate(
                    s![
                        ..,
                        ..,
                        i_h as isize..(i_h + self.k_h) as isize,
                        i_w as isize..(i_w + self.k_w) as isize
                    ],
                    &dx_dpatch,
                );
            }
        }

        d_x
    }
}

impl ToParams for PatchwiseConv2D {
    fn params(&mut self) -> Vec<Param> {
        vec![
            Param::new(&mut self.w).with_grad(&mut self.grads["d_w"]),
            Param::new(&mut self.b).with_grad(&mut self.grads["d_b"]),
        ]
    }
}

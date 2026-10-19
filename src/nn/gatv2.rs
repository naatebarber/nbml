use crate::{
    f,
    layers::Linear,
    optim::{Param, ToParams},
};
use ndarray::{Array2, Array3, ArrayViewMut1, Axis, Zip};
use ndarray_rand::{RandomExt, rand_distr::Normal};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone)]
pub struct GATv2Cache {
    pub s: Array2<f32>,
    pub r: Array2<f32>,
    pub alpha: Array2<f32>,
    pub denom: Array2<f32>,
    pub exp: Array2<f32>,
    pub z: Array3<f32>,
}

#[derive(Debug, Default, Clone)]
pub struct GATv2Grads {
    pub d_a: Array2<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GATv2 {
    pub d_in: usize,
    pub d_head: usize,
    pub n_head: usize,
    pub d_out: usize,

    pub s_proj: Linear,
    pub r_proj: Linear,
    pub o_proj: Linear,
    pub a: Array2<f32>,

    #[serde(skip)]
    pub cache: GATv2Cache,
    #[serde(skip)]
    pub grads: GATv2Grads,
}

impl GATv2 {
    pub fn new(d_in: usize, d_head: usize, n_head: usize, d_out: usize) -> Self {
        Self {
            d_in,
            d_head,
            n_head,
            d_out,

            s_proj: Linear::new(d_in, n_head * d_head),
            r_proj: Linear::new(d_in, n_head * d_head),
            o_proj: Linear::new(n_head * d_head, d_out),
            a: Array2::random((n_head, d_head), Normal::new(0., 1.).unwrap()),

            cache: GATv2Cache::default(),
            grads: GATv2Grads::default(),
        }
    }

    /// Scatter sum, where indices.len() == x.dim().0
    /// and indices contains 0..n
    pub fn scatter_sum(x: &Array2<f32>, indices: &[usize], n: usize) -> Array2<f32> {
        let e = x.dim().0;
        let mut n_sum: Array2<f32> = Array2::zeros((n, x.ncols()));

        for i in 0..e {
            let mut n_dst: ArrayViewMut1<f32> = n_sum.row_mut(indices[i]);
            n_dst += &(x.row(i));
        }

        n_sum
    }

    pub fn scatter_max(x: &Array2<f32>, indices: &[usize], n: usize) -> Array2<f32> {
        let e = x.dim().0;
        let mut n_max: Array2<f32> = Array2::from_elem((n, x.ncols()), f32::NEG_INFINITY);

        for i in 0..e {
            let dst = indices[i];
            let mut n_dst: ArrayViewMut1<f32> = n_max.row_mut(dst);

            Zip::from(&mut n_dst).and(&x.row(i)).for_each(|a, b| {
                *a = (*a).max(*b);
            });
        }

        n_max
    }

    /// COO format.
    /// X is a 2d tensor of node states (N, D)
    /// edge_index is a 2d tensor representing src,dst pairings (E, 2)
    pub fn forward(&mut self, x: Array2<f32>, index: &Array2<usize>, grad: bool) -> Array2<f32> {
        let n = x.nrows();
        let e = index.dim().0;

        let x_s = self.s_proj.forward(x.clone(), grad); // (n, n_head * d_head)
        let x_r = self.r_proj.forward(x, grad); // (n, n_head * d_head)

        let send = index.column(0).to_vec();
        let recv = index.column(1).to_vec();
        let s = x_s.select(Axis(0), &send); // (e, n_head * d_head)
        let r = x_r.select(Axis(0), &recv); // (e, n_head * d_head)

        if grad {
            self.cache.s = s.clone();
            self.cache.r = r.clone();
        }

        // another way of doing scoring. not similarity based.

        let z = f::leaky_relu(&(&s + &r))
            .into_shape_clone((e, self.n_head, self.d_head))
            .unwrap(); // (e, n_head, d_head)

        // (e, n_head, d_head) * (1, n_head, d_head) = (e, n_head, d_head) -> sum(axis(2)) -> (e, n_head)
        // doing elementwise multiply then sum is the same as a vector dot product. thats why we do
        // that

        let scores = (&z * &self.a.view().insert_axis(Axis(0))).sum_axis(Axis(2)); // (e, n_head)

        // ragged softmax

        let m = GATv2::scatter_max(&scores, &recv, n).select(Axis(0), &recv); // (e, n_head)
        let exp = (&scores - &m).exp(); // (e, n_head)
        let denom = GATv2::scatter_sum(&exp, &recv, n).select(Axis(0), &recv); // (e, n_head)
        let alpha = &exp / &denom;

        if grad {
            self.cache.z = z.clone();
            self.cache.alpha = alpha.clone();
        }

        // apply attention

        let v = s
            .view()
            .into_shape_with_order((e, self.n_head, self.d_head))
            .unwrap();
        let msg = &v * &alpha.view().insert_axis(Axis(2));
        let msg = msg
            .into_shape_with_order((e, self.n_head * self.d_head))
            .unwrap(); // (e, d_head * n_head)

        // sum into receivers

        let agg = GATv2::scatter_sum(&msg, &recv, n); // (e, ..) -> (n, d_head * n_head)

        // output proj

        self.o_proj.forward(agg, grad) // (n, d_out)
    }

    pub fn backward(&mut self, d_loss: Array2<f32>, index: &Array2<usize>) -> Array2<f32> {
        if self.grads.d_a.dim() == (0, 0) {
            self.grads.d_a = Array2::zeros(self.a.dim())
        }

        let send = index.column(0).to_vec();
        let recv = index.column(1).to_vec();
        let e = send.len();
        let n = d_loss.nrows();

        // through the output projection and agg scatter

        let d_agg = self.o_proj.backward(d_loss); // (n, n_head * d_head)
        let d_msg = d_agg.select(Axis(0), &recv); // (e, n_head * d_head)

        // through the application of attention

        let alpha = &self.cache.alpha;
        let d_msg = d_msg
            .into_shape_with_order((e, self.n_head, self.d_head))
            .unwrap(); // (e, n_head, d_head)
        let v = self
            .cache
            .s
            .view()
            .into_shape_with_order((e, self.n_head, self.d_head))
            .unwrap(); // (e, n_head, d_head)
        let d_v = &d_msg * &alpha.view().insert_axis(Axis(2));
        let d_alpha = (&d_msg * &v).sum_axis(Axis(2));

        // ragged softmax

        let ag = alpha * &d_alpha; // (e, n_head)
        let row = GATv2::scatter_sum(&ag, &recv, n).select(Axis(0), &recv); // (e, n_head)
        let d_scores = alpha * &(&d_alpha - &row); // (e, n_head)

        // scores

        let z = &self.cache.z;
        let d_scores_3 = d_scores.view().insert_axis(Axis(2));
        let d_a = (z * &d_scores_3).sum_axis(Axis(0));
        let d_z = &self.a * &d_scores_3;

        // z = leaky_relu(u), u = s + r
        let d_z = d_z
            .into_shape_with_order((e, self.n_head * self.d_head))
            .unwrap();
        let z = z
            .view()
            .into_shape_with_order((e, self.n_head * self.d_head))
            .unwrap();
        let d_u = d_z * f::d_leaky_relu(&z);

        // u feeds both s and r; s also fed the messages as v
        let d_v = d_v
            .into_shape_with_order((e, self.n_head * self.d_head))
            .unwrap();
        let d_s = &d_u + &d_v;
        let d_r = d_u
            .into_shape_with_order((e, self.n_head * self.d_head))
            .unwrap();

        // backward of the gathers (= scatter-sum by the same index)
        let d_x_s = GATv2::scatter_sum(&d_s, &send, n); // (n, n_head * d_head)
        let d_x_r = GATv2::scatter_sum(&d_r, &recv, n); // ..

        self.grads.d_a += &d_a;
        let d_x1 = self.s_proj.backward(d_x_s);
        let d_x2 = self.r_proj.backward(d_x_r);
        d_x1 + d_x2
    }
}

impl ToParams for GATv2 {
    fn params(&mut self) -> Vec<crate::optim::Param> {
        let mut params = vec![];

        params.append(&mut self.s_proj.params());
        params.append(&mut self.r_proj.params());
        params.append(&mut self.o_proj.params());
        params.push(Param::new(&mut self.a).with_grad(&mut self.grads.d_a));

        params
    }
}

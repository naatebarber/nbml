use std::{
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    usize,
};

use ndarray::{ArrayD, Axis, Dimension, Ix1, Ix2, SliceInfoElem, concatenate, stack};
use ndarray_rand::{
    RandomExt,
    rand_distr::{Normal, Uniform},
};
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};

pub type Float = f64;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Tensor {
    data: ArrayD<Float>,
}

impl Tensor {
    fn from(data: ArrayD<Float>) -> Tensor {
        Self { data }
    }

    // Initialization

    pub fn zeros(shape: impl IntoShape) -> Tensor {
        Self {
            data: ArrayD::zeros(shape.into_shape()),
        }
    }

    pub fn ones(shape: impl IntoShape) -> Tensor {
        Self {
            data: ArrayD::ones(shape.into_shape()),
        }
    }

    pub fn from_elem(shape: impl IntoShape, elem: Float) -> Tensor {
        Self {
            data: ArrayD::from_elem(shape.into_shape(), elem),
        }
    }

    pub fn random_normal(shape: impl IntoShape) -> Tensor {
        Self {
            data: ArrayD::random(shape.into_shape(), Normal::new(0., 1.).unwrap()),
        }
    }

    pub fn random_uniform(shape: impl IntoShape) -> Tensor {
        Self {
            data: ArrayD::random(shape.into_shape(), Uniform::new(0., 1.)),
        }
    }

    pub fn from_vec(shape: impl IntoShape, data: Vec<Float>) -> Tensor {
        Self {
            data: ArrayD::from_shape_vec(shape.into_shape(), data).unwrap(),
        }
    }

    // Elementwise ops

    pub fn exp(&self) -> Tensor {
        Tensor::from(self.data.exp())
    }

    pub fn ln(&self) -> Tensor {
        Tensor::from(self.data.ln())
    }

    pub fn mapv(&self, lambda: impl Fn(Float) -> Float) -> Tensor {
        Tensor::from(self.data.mapv(lambda))
    }

    pub fn powi(&self, pow: isize) -> Tensor {
        Tensor::from(self.data.powi(pow as i32))
    }

    pub fn powf(&self, pow: Float) -> Tensor {
        Tensor::from(self.data.powf(pow))
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor::from(self.data.sqrt())
    }

    pub fn log(&self, base: Float) -> Tensor {
        Tensor::from(self.data.log(base))
    }

    // Global reductions

    pub fn sum(&self) -> Float {
        self.data.sum()
    }

    pub fn mean(&self) -> Float {
        self.data.mean().unwrap()
    }

    pub fn std(&self, ddof: Float) -> Float {
        self.data.std(ddof)
    }

    pub fn var(&self, ddof: Float) -> Float {
        self.data.var(ddof)
    }

    // Shape descriptions

    pub fn rank(&self) -> usize {
        self.data.shape().len()
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn dim1(&self) -> usize {
        assert!(
            self.rank() == 1,
            "attempted to call dim1 on tensor of rank {}",
            self.rank()
        );
        self.shape()[0]
    }

    pub fn dim2(&self) -> (usize, usize) {
        assert!(
            self.rank() == 2,
            "attempted to call dim2 on tensor of rank {}",
            self.rank()
        );
        (self.shape()[0], self.shape()[1])
    }

    pub fn dim3(&self) -> (usize, usize, usize) {
        assert!(
            self.rank() == 3,
            "attempted to call dim3 on tensor of rank {}",
            self.rank()
        );
        (self.shape()[0], self.shape()[1], self.shape()[2])
    }

    pub fn dim4(&self) -> (usize, usize, usize, usize) {
        assert!(
            self.rank() == 4,
            "attempted to call dim4 on tensor of rank {}",
            self.rank()
        );
        (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
        )
    }

    // Min max

    pub fn min(&self) -> Float {
        self.data.min().unwrap().to_owned()
    }

    pub fn max(&self) -> Float {
        self.data.max().unwrap().to_owned()
    }

    pub fn argmin(&self) -> Vec<usize> {
        let index = self
            .data
            .argmin()
            .unwrap()
            .to_owned()
            .as_array_view()
            .to_vec();
        index
    }

    pub fn argmax(&self) -> Vec<usize> {
        let index = self
            .data
            .argmax()
            .unwrap()
            .to_owned()
            .as_array_view()
            .to_vec();
        index
    }

    // Matrix ops

    pub fn dot(&self, rhs: &Self) -> Tensor {
        match (self.rank(), rhs.rank()) {
            (0, 0) => self * rhs,
            (1, 0) => self * rhs,
            (0, 1) => self * rhs,
            (1, 1) => self * rhs,
            (1, 2) => {
                let left = self.data.view().into_dimensionality::<Ix1>().unwrap();
                let right = rhs.data.view().into_dimensionality::<Ix2>().unwrap();
                Tensor::from(left.dot(&right).into_dyn())
            }
            (2, 1) => {
                let left = self.data.view().into_dimensionality::<Ix2>().unwrap();
                let right = rhs.data.view().into_dimensionality::<Ix1>().unwrap();
                Tensor::from(left.dot(&right).into_dyn())
            }
            (2, 2) => {
                let left = self.data.view().into_dimensionality::<Ix2>().unwrap();
                let right = rhs.data.view().into_dimensionality::<Ix2>().unwrap();
                Tensor::from(left.dot(&right).into_dyn())
            }
            _ => panic!(
                "attempted to dot tensor of rank {} with tensor of rank {}",
                self.rank(),
                rhs.rank()
            ),
        }
    }

    // Shape manipulation

    pub fn permute(&self, axes: &[usize]) -> Tensor {
        Tensor::from(self.data.view().permuted_axes(axes).to_owned())
    }

    pub fn reshape(&self, shape: impl IntoShape) -> Tensor {
        let shape = shape.into_shape();
        Tensor::from(
            self.data
                .view()
                .to_shape(shape.as_slice())
                .unwrap()
                .to_owned(),
        )
    }

    pub fn broadcast(&self, shape: impl IntoShape) -> Tensor {
        let shape = shape.into_shape();
        Tensor::from(self.data.broadcast(shape.as_slice()).unwrap().to_owned())
    }

    pub fn t(&self) -> Tensor {
        Tensor::from(self.data.view().reversed_axes().to_owned())
    }

    pub fn insert_axis(&self, axis: usize) -> Tensor {
        assert!(
            axis <= self.rank(),
            "attempted to insert axis {} on rank {} tensor",
            axis,
            self.rank()
        );
        Tensor::from(self.data.view().insert_axis(Axis(axis)).to_owned())
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(
            axis < self.rank(),
            "attempted to sum axis {} on rank {} tensor",
            axis,
            self.rank()
        );
        Tensor::from(self.data.view().sum_axis(Axis(axis)).to_owned())
    }

    pub fn mean_axis(&self, axis: usize) -> Tensor {
        assert!(
            axis < self.rank(),
            "attempted to mean axis {} on rank {} tensor",
            axis,
            self.rank()
        );
        let shape = self.shape();
        let mut tensor = self.sum_axis(axis);
        tensor /= shape[axis] as f64;
        tensor
    }

    // Slice, stack, concatenate

    /**
     * Converts tensor::SliceAxis to ndarray::SliceInfoElem
     */
    fn slice_axis_to_ndarray(slice: &[SliceAxis]) -> Vec<SliceInfoElem> {
        slice
            .into_iter()
            .map(|slice| match slice {
                SliceAxis::Index(i) => SliceInfoElem::Index(*i),
                SliceAxis::Range { start, end } => SliceInfoElem::Slice {
                    start: start.unwrap_or(0),
                    end: *end,
                    step: 1,
                },
            })
            .collect::<Vec<SliceInfoElem>>()
    }

    /**
     * Outputs the dimensions of the resulting tensor if sliced
     */
    fn sliced_shape(&self, slice: &[SliceAxis]) -> Vec<usize> {
        let mut sliced_shape = vec![];
        for (axis, dim) in self.shape().iter().enumerate() {
            let slice = &slice[axis];
            match slice {
                SliceAxis::Index(_) => continue,
                SliceAxis::Range { start, end } => {
                    let dim = *dim as isize;
                    let mut s = start.unwrap_or(0);
                    let mut e = end.unwrap_or(dim as isize);

                    if s < 0 {
                        s = dim + s;
                    }

                    if e < 0 {
                        e = dim + e;
                    }

                    let size = e - s;

                    if size < 0 {
                        panic!("slice results in a tensor with negative size along axis {axis}")
                    }

                    sliced_shape.push(size as usize);
                }
            }
        }

        sliced_shape
    }

    pub fn slice(&self, slice: &[SliceAxis]) -> Tensor {
        assert!(
            slice.len() == self.rank(),
            "attempted to slice a rank {} tensor with a rank {} slice",
            self.rank(),
            slice.len()
        );
        let ndarray_slice = Tensor::slice_axis_to_ndarray(slice);
        Tensor::from(self.data.view().slice(ndarray_slice.as_slice()).to_owned())
    }

    pub fn slice_assign(&mut self, slice: &[SliceAxis], assign: &Tensor) {
        let sliced_shape = self.sliced_shape(slice);
        assert!(
            sliced_shape == assign.shape(),
            "cannot assign a tensor of shape {:?} to a slice of shape {:?}",
            assign.shape(),
            sliced_shape
        );
        let ndarray_slice = Tensor::slice_axis_to_ndarray(slice);
        self.data
            .slice_mut(ndarray_slice.as_slice())
            .assign(&assign.data);
    }

    pub fn concatenate(axis: usize, tensors: &[&Tensor]) -> Tensor {
        assert!(tensors.len() > 0, "cannot concatenate a list of 0 tensors");

        let mut fixed_shape = tensors[0].shape().to_vec();
        fixed_shape[axis] = 0;
        for i in 1..tensors.len() {
            let mut next_shape = tensors[i].shape().to_vec();
            next_shape[axis] = 0;
            assert!(
                next_shape == fixed_shape,
                "cannot concatenate tensor of dim {:?} with tensor of dim {:?}",
                tensors[i - 1].shape(),
                tensors[i].shape()
            );
            fixed_shape = next_shape.to_vec();
        }

        Tensor::from(
            concatenate(
                Axis(axis),
                tensors
                    .iter()
                    .map(|t| t.data.view())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap(),
        )
    }

    pub fn stack(axis: usize, tensors: &[&Tensor]) -> Tensor {
        assert!(tensors.len() > 0, "cannot stack a list of 0 tensors");

        let mut fixed_shape = tensors[0].shape().to_vec();
        for i in 1..tensors.len() {
            let next_shape = tensors[i].shape().to_vec();
            assert!(
                next_shape == fixed_shape,
                "cannot stack tensor of dim {:?} with tensor of dim {:?}",
                tensors[i - 1].shape(),
                tensors[i].shape()
            );
            fixed_shape = next_shape.to_vec();
        }

        Tensor::from(
            stack(
                Axis(axis),
                tensors
                    .iter()
                    .map(|t| t.data.view())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
            .unwrap(),
        )
    }
}

// Convenience trait for defining shapes

pub trait IntoShape {
    fn into_shape(self) -> Vec<usize>;
}

impl IntoShape for usize {
    fn into_shape(self) -> Vec<usize> {
        vec![self]
    }
}

impl IntoShape for (usize,) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0]
    }
}

impl IntoShape for (usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl IntoShape for (usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl IntoShape for (usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

impl IntoShape for (usize, usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

// Slice definition enum

pub enum SliceAxis {
    Index(isize),
    Range {
        start: Option<isize>,
        end: Option<isize>,
    },
}

impl SliceAxis {
    pub fn from_range<R: Into<SliceAxis>>(r: R) -> SliceAxis {
        r.into()
    }
}

impl From<isize> for SliceAxis {
    fn from(value: isize) -> Self {
        SliceAxis::Index(value)
    }
}

impl From<std::ops::Range<isize>> for SliceAxis {
    fn from(r: std::ops::Range<isize>) -> SliceAxis {
        SliceAxis::Range {
            start: Some(r.start),
            end: Some(r.end),
        }
    }
}

impl From<std::ops::RangeFrom<isize>> for SliceAxis {
    fn from(r: std::ops::RangeFrom<isize>) -> SliceAxis {
        SliceAxis::Range {
            start: Some(r.start),
            end: None,
        }
    }
}

impl From<std::ops::RangeTo<isize>> for SliceAxis {
    fn from(r: std::ops::RangeTo<isize>) -> SliceAxis {
        SliceAxis::Range {
            start: None,
            end: Some(r.end),
        }
    }
}

impl From<std::ops::RangeFull> for SliceAxis {
    fn from(_: std::ops::RangeFull) -> Self {
        SliceAxis::Range {
            start: Some(0),
            end: None,
        }
    }
}

// Slice macro

#[macro_export]
macro_rules! s {
    (@convert $r:expr) => {
        $crate::tensor::SliceAxis::from($r)
    };
    // last item
    (@parse [$($acc:expr),*] $r:expr) => {
        &[$($acc,)* s!(@convert $r)]
    };
    // more items
    (@parse [$($acc:expr),*] $r:expr, $($rest:tt)*) => {
        s!(@parse [$($acc,)* s!(@convert $r)] $($rest)*)
    };
    // entry
    ($($t:tt)*) => {
        s!(@parse [] $($t)*)
    };
}

// Allow basic arithmetic

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        self.mapv(|x| -x)
    }
}

macro_rules! impl_index {
    ($n:literal) => {
        impl Index<[usize; $n]> for Tensor {
            type Output = Float;
            fn index(&self, idx: [usize; $n]) -> &Float {
                &self.data[&idx[..]]
            }
        }
        impl IndexMut<[usize; $n]> for Tensor {
            fn index_mut(&mut self, idx: [usize; $n]) -> &mut Float {
                &mut self.data[&idx[..]]
            }
        }
    };
}

impl_index!(1);
impl_index!(2);
impl_index!(3);
impl_index!(4);

macro_rules! impl_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::from(&self.data $op &rhs.data)
            }
        }

        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::from(&self.data $op &rhs.data)
            }
        }

        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::from(&self.data $op &rhs.data)
            }
        }

        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::from(&self.data $op &rhs.data)
            }
        }
    }
}

impl_op!(Add, add, +);
impl_op!(Sub, sub, -);
impl_op!(Mul, mul, *);
impl_op!(Div, div, /);

macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<Tensor> for Tensor {
            fn $method(&mut self, rhs: Tensor) {
                self.data $op &rhs.data
            }
        }

        impl $trait<&Tensor> for Tensor {
            fn $method(&mut self, rhs: &Tensor) {
                self.data $op &rhs.data
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, +=);
impl_assign_op!(SubAssign, sub_assign, -=);
impl_assign_op!(MulAssign, mul_assign, *=);
impl_assign_op!(DivAssign, div_assign, /=);

macro_rules! impl_primitive_op {
    ($primitive:ident, $trait:ident, $method:ident, $op:tt) => {
        impl $trait<$primitive> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: $primitive) -> Tensor {
                Tensor::from(self.data $op rhs)
            }
        }

        impl $trait<$primitive> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: $primitive) -> Tensor {
                Tensor::from(self.data.to_owned() $op rhs)
            }
        }
    }
}

impl_primitive_op!(Float, Add, add, +);
impl_primitive_op!(Float, Sub, sub, -);
impl_primitive_op!(Float, Mul, mul, *);
impl_primitive_op!(Float, Div, div, /);

macro_rules! impl_primitive_assign_op {
    ($primitive:ident, $trait:ident, $method:ident, $op:tt) => {
        impl $trait<$primitive> for Tensor {
            fn $method(&mut self, rhs: $primitive) {
                self.data $op rhs;
            }
        }

        impl $trait<$primitive> for &mut Tensor {
            fn $method(&mut self, rhs: $primitive) {
                self.data $op rhs;
            }
        }
    }
}

impl_primitive_assign_op!(Float, AddAssign, add_assign, +=);
impl_primitive_assign_op!(Float, SubAssign, sub_assign, -=);
impl_primitive_assign_op!(Float, MulAssign, mul_assign, *=);
impl_primitive_assign_op!(Float, DivAssign, div_assign, /=);

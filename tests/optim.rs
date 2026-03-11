use nbml::{
    Tensor,
    optim2::{Param, ToParams},
    tensor::Tensor2,
    util::Cache,
};

struct DummyModel {
    w: Tensor,
    grads: Cache,
}

impl DummyModel {
    fn new() -> Self {
        Self {
            w: Tensor2::ones((3, 3)),
            grads: Cache::new(),
        }
    }

    fn backward(&mut self) {
        self.grads.accumulate("d_w", Tensor2::ones((3, 3)) * 5.0);
    }
}

impl ToParams for DummyModel {
    fn params(&mut self) -> Vec<Param> {
        vec![Param::new(&mut self.w).with_grad(&mut self.grads["d_w"])]
    }
}

#[test]
fn params_returns_zero_shape_grad_before_backward() {
    let mut model = DummyModel::new();
    let params = model.params();

    unsafe {
        let grad = &**params[0].grad.as_ref().unwrap();
        assert_eq!(
            grad.shape(),
            &[0],
            "grad shape should be [0] before any backward, got {:?}",
            grad.shape()
        );
    }
}

#[test]
fn zero_grads_zeros_after_backward() {
    let mut model = DummyModel::new();

    model.backward();

    // verify grads are non-zero
    let params = model.params();
    unsafe {
        let grad = &**params[0].grad.as_ref().unwrap();
        assert_eq!(grad.shape(), &[3, 3]);
        assert!(grad.sum() > 0.0, "grads should be non-zero after backward");
    }

    model.zero_grads();

    // verify grads are zeroed but retain shape
    let params = model.params();
    unsafe {
        let grad = &**params[0].grad.as_ref().unwrap();
        assert_eq!(
            grad.shape(),
            &[3, 3],
            "grad shape should be preserved after zero_grads"
        );
        assert_eq!(grad.sum(), 0.0, "grads should be zero after zero_grads");
    }
}

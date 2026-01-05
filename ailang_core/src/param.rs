use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct Param {
    pub value: Tensor,
    pub grad: Tensor,
}

impl Param {
    pub fn new(value: Tensor) -> Self {
        let shape = value.shape().to_vec();
        Self {
            grad: Tensor::zeros(&shape),
            value,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad.data.fill(0.0);
    }

    pub fn sgd_step(&mut self, lr: f32) {
        // value -= lr * grad
        self.value.data = &self.value.data - &(self.grad.data.mapv(|g| lr * g));
    }
}

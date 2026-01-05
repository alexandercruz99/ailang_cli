use ndarray::{ArrayD, IxDyn};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: ArrayD<f32>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::<f32>::zeros(IxDyn(shape)),
        }
    }

    pub fn from_vec(shape: &[usize], values: Vec<f32>) -> Self {
        Self {
            data: ArrayD::from_shape_vec(IxDyn(shape), values)
                .expect("shape does not match number of values"),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    pub fn scalar(&self) -> f32 {
        *self
            .data
            .iter()
            .next()
            .expect("Tensor::scalar: tensor must have at least one element")
    }
}

#[derive(Clone, Debug)]
pub struct TokenIds {
    pub data: Vec<usize>,
    pub shape: Vec<usize>,
}

impl TokenIds {
    pub fn new(shape: &[usize], data: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "TokenIds: shape does not match data length"
        );
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn num_elements(&self) -> usize {
        self.data.len()
    }
}

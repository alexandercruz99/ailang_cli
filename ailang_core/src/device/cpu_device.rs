//! CPU device implementation using ndarray

use super::{Device, DeviceKind, TensorStorage};
use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::fmt;

/// CPU device implementation
#[derive(Clone)]
pub struct CpuDevice {
    name: String,
}

impl fmt::Debug for CpuDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuDevice")
            .field("name", &self.name)
            .finish()
    }
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            name: "cpu".to_string(),
        }
    }
}

impl Device for CpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU tensor storage backed by ndarray
#[derive(Debug, Clone)]
pub struct CpuTensor {
    data: ArrayD<f32>,
}

impl CpuTensor {
    /// Create from existing ndarray
    pub fn from_array(data: ArrayD<f32>) -> Self {
        Self { data }
    }

    /// Get reference to underlying array
    pub fn as_array(&self) -> &ArrayD<f32> {
        &self.data
    }

    /// Convert to Tensor (for backward compatibility)
    pub fn to_tensor(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
        }
    }

    /// Create from Tensor (for backward compatibility)
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            data: tensor.data.clone(),
        }
    }
}

impl TensorStorage for CpuTensor {
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::<f32>::zeros(shape),
        }
    }

    fn from_vec_f32(shape: &[usize], data: Vec<f32>) -> Self {
        Self {
            data: ArrayD::from_shape_vec(shape, data).expect("Shape mismatch in from_vec_f32"),
        }
    }

    fn to_vec_f32(&self) -> Vec<f32> {
        self.data.iter().copied().collect()
    }

    fn scalar(&self) -> f32 {
        assert_eq!(self.data.len(), 1, "scalar() called on non-scalar tensor");
        self.data.iter().next().copied().unwrap_or(0.0)
    }
}

//! Device backend abstraction for AILang
//!
//! This module provides a device abstraction layer that allows AILang
//! to run on different backends (CPU, GPU) while maintaining determinism
//! and backward compatibility.

mod cpu_device;

#[cfg(feature = "gpu-wgpu")]
mod wgpu_device;

#[cfg(feature = "gpu-wgpu")]
mod gpu_kernels;

pub use cpu_device::{CpuDevice, CpuTensor};

#[cfg(feature = "gpu-wgpu")]
pub use wgpu_device::{GpuDevice, GpuTensor};

#[cfg(feature = "gpu-wgpu")]
pub use gpu_kernels::GpuKernels;

// Re-export for convenience
pub use cpu_device::CpuDevice as DefaultDevice;

// Tensor import removed - not needed in device module

/// Device kind enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

/// Device trait for backend abstraction
pub trait Device: Send + Sync {
    /// Get the device kind
    fn kind(&self) -> DeviceKind;

    /// Get a human-readable device name
    fn name(&self) -> &str;
}

/// Tensor storage trait for device-specific tensor buffers
pub trait TensorStorage: Send + Sync {
    /// Get the shape of the tensor
    fn shape(&self) -> &[usize];

    /// Get the number of elements
    fn len(&self) -> usize;

    /// Check if the tensor is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the device kind this storage is on
    fn device_kind(&self) -> DeviceKind;

    /// Create a zero-filled tensor with the given shape
    fn zeros(shape: &[usize]) -> Self
    where
        Self: Sized;

    /// Create a tensor from a vector of f32 values
    fn from_vec_f32(shape: &[usize], data: Vec<f32>) -> Self
    where
        Self: Sized;

    /// Convert tensor data to a vector of f32 (for CPU fallback/export)
    fn to_vec_f32(&self) -> Vec<f32>;

    /// Get a scalar value (for 0D tensors)
    fn scalar(&self) -> f32;
}

/// Device reference type for passing devices around
pub type DeviceRef = Box<dyn Device>;

/// Default CPU device (singleton pattern)
pub fn default_cpu_device() -> CpuDevice {
    CpuDevice::new()
}

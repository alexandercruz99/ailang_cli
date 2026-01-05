//! GPU device implementation using wgpu (Metal backend on Apple Silicon)
//!
//! This module provides GPU acceleration via wgpu, which uses Metal
//! on Apple Silicon devices. All operations maintain determinism.

#[cfg(not(feature = "gpu-wgpu"))]
compile_error!("wgpu_device module requires gpu-wgpu feature");

use super::{Device as DeviceTrait, DeviceKind, TensorStorage};
use crate::diagnostic::Diagnostic;
use crate::error::RuntimeError;
use pollster::FutureExt;
use std::sync::Arc;
use wgpu::{
    Adapter, Backend, Backends, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    Device as WgpuDevice, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    Maintain, MapMode, PowerPreference, Queue, RequestAdapterOptions,
};

/// GPU device implementation using wgpu
pub struct GpuDevice {
    instance: Instance,
    adapter: Adapter,
    device: Arc<WgpuDevice>,
    queue: Arc<Queue>,
    name: String,
}

unsafe impl Send for GpuDevice {}
unsafe impl Sync for GpuDevice {}

impl GpuDevice {
    /// Create a new GPU device
    pub fn new() -> Result<Self, Diagnostic> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::METAL,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()
            .ok_or_else(|| {
                Diagnostic::new("E_GPU_UNAVAILABLE", "GPU device not available".to_string())
                    .with_field(
                        "details".to_string(),
                        "No compatible GPU adapter found".to_string(),
                    )
            })?;

        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("AILang GPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                },
                None,
            )
            .block_on()
            .map_err(|e| {
                Diagnostic::new(
                    "E_GPU_UNAVAILABLE",
                    "Failed to create GPU device".to_string(),
                )
                .with_field("details".to_string(), format!("{:?}", e))
            })?;

        // Store device and queue as Arc for sharing
        let device_arc = Arc::new(device);
        let queue_arc = Arc::new(queue);

        let adapter_info = adapter.get_info();
        let backend_name = match adapter_info.backend {
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
            Backend::Dx12 => "dx12",
            Backend::Gl => "gl",
            _ => "unknown",
        };

        Ok(Self {
            instance,
            adapter,
            device: device_arc,
            queue: queue_arc,
            name: format!("gpu (wgpu/{})", backend_name),
        })
    }

    /// Get reference to wgpu device
    pub fn wgpu_device(&self) -> Arc<WgpuDevice> {
        self.device.clone()
    }

    /// Get reference to wgpu queue
    pub fn wgpu_queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }
}

impl DeviceTrait for GpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Gpu
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// GPU tensor storage backed by wgpu buffers
pub struct GpuTensor {
    pub(crate) buffer: Buffer,
    pub(crate) shape: Vec<usize>,
    pub(crate) len: usize,
    pub(crate) device: Arc<WgpuDevice>,
    pub(crate) queue: Arc<Queue>,
}

unsafe impl Send for GpuTensor {}
unsafe impl Sync for GpuTensor {}

impl GpuTensor {
    /// Create from CPU data (upload to GPU)
    pub fn from_cpu_data(
        device: Arc<WgpuDevice>,
        queue: Arc<Queue>,
        shape: &[usize],
        data: &[f32],
    ) -> Result<Self, RuntimeError> {
        let len = data.len();
        let size = (len * std::mem::size_of::<f32>()) as u64;

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("GpuTensor buffer"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));

        Ok(Self {
            buffer,
            shape: shape.to_vec(),
            len,
            device: device.clone(),
            queue: queue.clone(),
        })
    }

    /// Download data to CPU
    pub fn to_cpu_data(&self) -> Result<Vec<f32>, RuntimeError> {
        use std::sync::Arc;

        let size = (self.len * std::mem::size_of::<f32>()) as u64;
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("GpuTensor staging buffer"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("GpuTensor download encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Get reference to wgpu buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl TensorStorage for GpuTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn len(&self) -> usize {
        self.len
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Gpu
    }

    fn zeros(_shape: &[usize]) -> Self {
        // This requires a device/queue, so we'll need to refactor
        // For now, this is a placeholder
        panic!("GpuTensor::zeros requires device context - use from_cpu_data instead");
    }

    fn from_vec_f32(_shape: &[usize], _data: Vec<f32>) -> Self {
        // This requires a device/queue, so we'll need to refactor
        panic!("GpuTensor::from_vec_f32 requires device context - use from_cpu_data instead");
    }

    fn to_vec_f32(&self) -> Vec<f32> {
        self.to_cpu_data().unwrap_or_else(|e| {
            panic!("Failed to download GPU tensor: {:?}", e);
        })
    }

    fn scalar(&self) -> f32 {
        assert_eq!(self.len, 1, "scalar() called on non-scalar tensor");
        self.to_vec_f32()[0]
    }
}

// f32 is already Pod and Zeroable via bytemuck

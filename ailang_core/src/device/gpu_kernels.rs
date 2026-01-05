//! GPU kernel implementations using WGSL compute shaders
//!
//! This module provides GPU-accelerated implementations of core operations
//! using wgpu compute shaders. All operations maintain determinism.

#[cfg(not(feature = "gpu-wgpu"))]
compile_error!("gpu_kernels module requires gpu-wgpu feature");

use super::wgpu_device::GpuTensor;
use crate::diagnostic::Diagnostic;
use crate::error::RuntimeError;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device as WgpuDevice, PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderSource,
    ShaderStages,
};

/// GPU kernel executor
pub struct GpuKernels {
    device: Arc<WgpuDevice>,
    queue: Arc<Queue>,
    add_pipeline: ComputePipeline,
    relu_pipeline: ComputePipeline,
    matmul_pipeline: ComputePipeline,
}

impl GpuKernels {
    /// Create a new GPU kernel executor
    pub fn new(device: Arc<WgpuDevice>, queue: Arc<Queue>) -> Result<Self, Diagnostic> {
        // Add kernel shader
        let add_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("add_shader"),
            source: ShaderSource::Wgsl(ADD_SHADER.into()),
        });

        // Relu kernel shader
        let relu_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("relu_shader"),
            source: ShaderSource::Wgsl(RELU_SHADER.into()),
        });

        // MatMul kernel shader
        let matmul_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: ShaderSource::Wgsl(MATMUL_SHADER.into()),
        });

        // Create bind group layouts
        let add_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("add_bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let relu_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("relu_bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let matmul_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("matmul_bind_group_layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create compute pipelines
        let add_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("add_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("add_pipeline_layout"),
                bind_group_layouts: &[&add_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &add_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let relu_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("relu_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("relu_pipeline_layout"),
                bind_group_layouts: &[&relu_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &relu_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let matmul_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("matmul_pipeline_layout"),
                bind_group_layouts: &[&matmul_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &matmul_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            device,
            queue,
            add_pipeline,
            relu_pipeline,
            matmul_pipeline,
        })
    }

    /// Execute elementwise addition: out = a + b
    pub fn add(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        out_shape: &[usize],
    ) -> Result<GpuTensor, RuntimeError> {
        let len = out_shape.iter().product();
        let size = (len * std::mem::size_of::<f32>()) as u64;

        // Create output buffer
        let out_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("add_out_buffer"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("add_bind_group"),
            layout: &self.add_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: out_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: a.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: b.buffer().as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("add_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("add_compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.add_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((len as u32 + 255) / 256, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: out_buffer,
            shape: out_shape.to_vec(),
            len,
            device: self.device.clone(),
            queue: self.queue.clone(),
        })
    }

    /// Execute ReLU: out = max(0, x)
    pub fn relu(&self, x: &GpuTensor, out_shape: &[usize]) -> Result<GpuTensor, RuntimeError> {
        let len = out_shape.iter().product();
        let size = (len * std::mem::size_of::<f32>()) as u64;

        let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relu_out_buffer"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("relu_bind_group"),
            layout: &self.relu_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: out_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: x.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("relu_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("relu_compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.relu_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((len as u32 + 255) / 256, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: out_buffer,
            shape: out_shape.to_vec(),
            len,
            device: self.device.clone(),
            queue: self.queue.clone(),
        })
    }

    /// Execute matrix multiplication: C = A @ B
    /// A: [M, K], B: [K, N], C: [M, N]
    pub fn matmul(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuTensor, RuntimeError> {
        let out_shape = &[m, n];
        let len = m * n;
        let size = (len * std::mem::size_of::<f32>()) as u64;

        let out_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("matmul_out_buffer"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &self.matmul_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: out_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: a.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: b.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("matmul_compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch one workgroup per output element (naive, but correct)
            compute_pass.dispatch_workgroups((m as u32 + 15) / 16, (n as u32 + 15) / 16, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: out_buffer,
            shape: out_shape.to_vec(),
            len,
            device: self.device.clone(),
            queue: self.queue.clone(),
        })
    }
}

// WGSL compute shaders

const ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&out)) {
        return;
    }
    out[index] = a[index] + b[index];
}
"#;

const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&out)) {
        return;
    }
    out[index] = max(0.0, x[index]);
}
"#;

const MATMUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> c: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;

// MatMul: C = A @ B where A is [M, K] and B is [K, N], result is [M, N]
// We use a 2D workgroup dispatch: one thread per output element
// For MVP, we infer dimensions from array lengths (simplified approach)
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    let c_len = arrayLength(&c);
    let a_len = arrayLength(&a);
    let b_len = arrayLength(&b);
    
    // Infer dimensions from array lengths
    // C = [M, N] -> c_len = M * N
    // A = [M, K] -> a_len = M * K
    // B = [K, N] -> b_len = K * N
    // We can compute: K = a_len / M, but we need M
    // Since c_len = M * N and we dispatch a 2D grid, we can infer:
    // For MVP, use simplified inference: assume square-ish matrices
    // M ≈ sqrt(c_len), but this is approximate
    
    // Better approach: use the fact that K = a_len / M = b_len / N
    // So: a_len * N = b_len * M, and M * N = c_len
    // This gives us: M = sqrt(c_len * a_len / b_len), N = c_len / M, K = a_len / M
    
    // For MVP, use a simpler approach: compute from known relationships
    // Since we know the dispatch grid size, we can infer M and N
    // But for now, let's use array length relationships
    
    // Infer dimensions more accurately:
    // We know: c_len = M * N, a_len = M * K, b_len = K * N
    // From: a_len * b_len = (M * K) * (K * N) = M * K^2 * N = K^2 * (M * N) = K^2 * c_len
    // So: K = sqrt(a_len * b_len / c_len)
    // Then: M = a_len / K, N = b_len / K
    
    let k_squared = (a_len * b_len) / c_len;
    if (k_squared == 0u) {
        return;
    }
    
    // Approximate K (WGSL doesn't have sqrt, so we use integer division approximation)
    // For MVP, we'll use a simpler approach: try common K values
    var k = 1u;
    var m = a_len;
    var n = b_len;
    
    // Find K by testing: if a_len % k == 0 and b_len % k == 0 and (a_len/k) * (b_len/k) == c_len
    // For MVP, use a simple heuristic: K is likely a factor of both a_len and b_len
    // Try K values from 1 to min(sqrt(a_len), sqrt(b_len))
    let max_k = min(a_len, b_len);
    for (var test_k = 1u; test_k <= max_k; test_k++) {
        if ((a_len % test_k == 0u) && (b_len % test_k == 0u)) {
            let test_m = a_len / test_k;
            let test_n = b_len / test_k;
            if (test_m * test_n == c_len) {
                k = test_k;
                m = test_m;
                n = test_n;
                break;
            }
        }
    }
    
    // Bounds check
    if (row >= m || col >= n) {
        return;
    }
    
    // Compute dot product: C[row, col] = sum_k A[row, k] * B[k, col]
    var sum = 0.0;
    for (var i = 0u; i < k; i++) {
        let a_idx = row * k + i;
        let b_idx = i * n + col;
        if (a_idx < a_len && b_idx < b_len) {
            sum = sum + a[a_idx] * b[b_idx];
        }
    }
    
    let c_idx = row * n + col;
    if (c_idx < c_len) {
        c[c_idx] = sum;
    }
}
"#;

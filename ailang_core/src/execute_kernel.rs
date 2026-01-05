// Unified op kernel dispatch
// This module contains the single source of truth for forward op execution.
// Both full-graph and selective executors call this module.

use crate::capability::Capabilities;
use crate::error::{FileError, RuntimeError};
use crate::ir::{CompareOp, LogicalOp, Op};
use crate::tensor::{Tensor, TokenIds};
use ndarray::IxDyn;
use std::collections::HashMap;
use std::fs;

/// Execute a single op forward pass.
/// This is the ONLY place where op forward execution is implemented.
///
/// # Arguments
/// * `op` - The operation to execute
/// * `input_values` - Values of input nodes (by node_id index)
/// * `inputs` - Graph inputs (tensor inputs)
/// * `token_ids` - Token IDs side-channel
/// * `paths` - File paths side-channel
/// * `capabilities` - Allowed capabilities
/// * `dim_bindings` - Optional dimension bindings for symbolic operations
/// * `device` - Optional device for GPU execution (None = CPU)
#[cfg(feature = "gpu-wgpu")]
pub fn execute_op_forward(
    op: &Op,
    input_values: &[Tensor],
    inputs: &[Tensor],
    token_ids: &[TokenIds],
    paths: &[String],
    capabilities: &Capabilities,
    dim_bindings: Option<&HashMap<String, usize>>,
    device: Option<&crate::device::GpuDevice>,
) -> Result<Tensor, RuntimeError> {
    use crate::device::{GpuDevice, GpuKernels, GpuTensor};
    use std::sync::Arc;

    // Check if we should use GPU for supported ops
    let use_gpu = device.is_some() && matches!(op, Op::Add | Op::ReLU | Op::MatMul2D);

    if use_gpu {
        let gpu_device = device.unwrap();
        let wgpu_device = gpu_device.wgpu_device();
        let wgpu_queue = gpu_device.wgpu_queue();
        let kernels = GpuKernels::new(wgpu_device.clone(), wgpu_queue.clone()).map_err(|diag| {
            RuntimeError::Other(format!("Failed to create GPU kernels: {}", diag.title))
        })?;

        match op {
            Op::Add => {
                assert_eq!(input_values.len(), 2);
                // Convert CPU tensors to GPU
                let a_data = input_values[0].data.as_slice().unwrap();
                let b_data = input_values[1].data.as_slice().unwrap();
                let a_shape = input_values[0].shape();
                let b_shape = input_values[1].shape();

                let a_gpu = GpuTensor::from_cpu_data(
                    wgpu_device.clone(),
                    wgpu_queue.clone(),
                    a_shape,
                    a_data,
                )?;
                let b_gpu = GpuTensor::from_cpu_data(
                    wgpu_device.clone(),
                    wgpu_queue.clone(),
                    b_shape,
                    b_data,
                )?;

                // Compute output shape (broadcast rules)
                let out_shape = if a_shape == b_shape {
                    a_shape.to_vec()
                } else {
                    // For MVP, require same shape
                    return Err(RuntimeError::Other(
                        "Add: GPU path requires same shape tensors".to_string(),
                    ));
                };

                let out_gpu = kernels.add(&a_gpu, &b_gpu, &out_shape)?;
                let out_data = out_gpu.to_cpu_data()?;
                Ok(Tensor::from_vec(&out_shape, out_data))
            }
            Op::ReLU => {
                assert_eq!(input_values.len(), 1);
                let x_data = input_values[0].data.as_slice().unwrap();
                let x_shape = input_values[0].shape();

                let x_gpu = GpuTensor::from_cpu_data(
                    wgpu_device.clone(),
                    wgpu_queue.clone(),
                    x_shape,
                    x_data,
                )?;
                let out_gpu = kernels.relu(&x_gpu, x_shape)?;
                let out_data = out_gpu.to_cpu_data()?;
                Ok(Tensor::from_vec(x_shape, out_data))
            }
            Op::MatMul2D => {
                assert_eq!(input_values.len(), 2);
                let a = &input_values[0];
                let b = &input_values[1];

                let a_2d = a
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        RuntimeError::Other("MatMul2D: first input must be 2D".to_string())
                    })?;
                let b_2d = b
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        RuntimeError::Other("MatMul2D: second input must be 2D".to_string())
                    })?;

                let m = a_2d.shape()[0];
                let k = a_2d.shape()[1];
                let n = b_2d.shape()[1];

                if b_2d.shape()[0] != k {
                    return Err(RuntimeError::Other(format!(
                        "MatMul2D: inner dimension mismatch: {} != {}",
                        k,
                        b_2d.shape()[0]
                    )));
                }

                let a_data = a.data.as_slice().unwrap();
                let b_data = b.data.as_slice().unwrap();

                let a_gpu = GpuTensor::from_cpu_data(
                    wgpu_device.clone(),
                    wgpu_queue.clone(),
                    &[m, k],
                    a_data,
                )?;
                let b_gpu = GpuTensor::from_cpu_data(
                    wgpu_device.clone(),
                    wgpu_queue.clone(),
                    &[k, n],
                    b_data,
                )?;

                let out_gpu = kernels.matmul(&a_gpu, &b_gpu, m, k, n)?;
                let out_data = out_gpu.to_cpu_data()?;
                Ok(Tensor::from_vec(&[m, n], out_data))
            }
            _ => {
                // Fall through to CPU path
                execute_op_forward_cpu(
                    op,
                    input_values,
                    inputs,
                    token_ids,
                    paths,
                    capabilities,
                    dim_bindings,
                )
            }
        }
    } else {
        // CPU path
        execute_op_forward_cpu(
            op,
            input_values,
            inputs,
            token_ids,
            paths,
            capabilities,
            dim_bindings,
        )
    }
}

#[cfg(not(feature = "gpu-wgpu"))]
pub fn execute_op_forward(
    op: &Op,
    input_values: &[Tensor],
    inputs: &[Tensor],
    token_ids: &[TokenIds],
    paths: &[String],
    capabilities: &Capabilities,
    dim_bindings: Option<&HashMap<String, usize>>,
    _device: Option<()>, // Placeholder for API compatibility
) -> Result<Tensor, RuntimeError> {
    execute_op_forward_cpu(
        op,
        input_values,
        inputs,
        token_ids,
        paths,
        capabilities,
        dim_bindings,
    )
}

/// CPU-only execution path (internal helper)
fn execute_op_forward_cpu(
    op: &Op,
    input_values: &[Tensor],
    inputs: &[Tensor],
    token_ids: &[TokenIds],
    paths: &[String],
    capabilities: &Capabilities,
    dim_bindings: Option<&HashMap<String, usize>>,
) -> Result<Tensor, RuntimeError> {
    match op {
        Op::Input(idx) => Ok(inputs[*idx].clone()),
        Op::Add => {
            assert_eq!(input_values.len(), 2);
            Ok(Tensor {
                data: &input_values[0].data + &input_values[1].data,
            })
        }
        Op::Sub => {
            assert_eq!(input_values.len(), 2);
            Ok(Tensor {
                data: &input_values[0].data - &input_values[1].data,
            })
        }
        Op::Mul => {
            assert_eq!(input_values.len(), 2);
            Ok(Tensor {
                data: &input_values[0].data * &input_values[1].data,
            })
        }
        Op::MatMul2D => {
            assert_eq!(input_values.len(), 2);
            let a = &input_values[0];
            let b = &input_values[1];
            let a_2d = a
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| RuntimeError::Other("MatMul2D: first input must be 2D".to_string()))?;
            let b_2d = b
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    RuntimeError::Other("MatMul2D: second input must be 2D".to_string())
                })?;
            Ok(Tensor {
                data: a_2d.dot(&b_2d).into_dyn(),
            })
        }
        Op::ReLU => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];
            Ok(Tensor {
                data: x.data.mapv(|v| v.max(0.0)),
            })
        }
        Op::Sum => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];
            let sum_val: f32 = x.data.iter().sum();
            Ok(Tensor {
                data: ndarray::ArrayD::from_elem(IxDyn(&[]), sum_val),
            })
        }
        Op::Mean => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];
            let sum_val: f32 = x.data.iter().sum();
            let count = x.data.len() as f32;
            Ok(Tensor {
                data: ndarray::ArrayD::from_elem(IxDyn(&[]), sum_val / count),
            })
        }
        Op::Embedding(token_ids_idx) => {
            assert_eq!(input_values.len(), 1);
            let weight = &input_values[0];
            let ids = &token_ids[*token_ids_idx];

            // weight: [vocab, dim], ids: [batch, time] -> output: [batch, time, dim]
            let weight_2d = weight
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| RuntimeError::Other("Embedding: weight must be 2D".to_string()))?;
            let vocab_size = weight_2d.shape()[0];
            let dim = weight_2d.shape()[1];

            let batch = ids.shape()[0];
            let time = ids.shape().get(1).copied().unwrap_or(1);

            let mut output_data = Vec::with_capacity(batch * time * dim);
            for &id in &ids.data {
                assert!(
                    id < vocab_size,
                    "Embedding: token id {} >= vocab_size {}",
                    id,
                    vocab_size
                );
                let row = weight_2d.row(id);
                output_data.extend(row.iter().copied());
            }

            let output_shape = if ids.shape().len() == 2 {
                vec![batch, time, dim]
            } else {
                vec![batch, dim]
            };

            Ok(Tensor {
                data: ndarray::ArrayD::from_shape_vec(IxDyn(&output_shape), output_data).map_err(
                    |_| RuntimeError::Other("Embedding: failed to create output".to_string()),
                )?,
            })
        }
        Op::Softmax(axis) => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];

            // Handle 2D and 3D cases (matching forward.rs implementation)
            if x.shape().len() == 2 {
                // 2D case: [N, M]
                let x_2d = x
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| {
                        RuntimeError::Other("Softmax: input must be 2D or 3D".to_string())
                    })?;
                let mut result = x_2d.to_owned();

                if *axis == 1 {
                    // Softmax along columns (last dim)
                    for mut row in result.rows_mut() {
                        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        row.mapv_inplace(|v| (v - max_val).exp());
                        let sum: f32 = row.sum();
                        row.mapv_inplace(|v| v / sum);
                    }
                } else if *axis == 0 {
                    // Softmax along rows (first dim)
                    for mut col in result.columns_mut() {
                        let max_val = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        col.mapv_inplace(|v| (v - max_val).exp());
                        let sum: f32 = col.sum();
                        col.mapv_inplace(|v| v / sum);
                    }
                } else {
                    return Err(RuntimeError::Other(format!(
                        "Softmax: unsupported axis {} for 2D",
                        axis
                    )));
                }

                Ok(Tensor {
                    data: result.into_dyn(),
                })
            } else if x.shape().len() == 3 {
                // 3D case: [B, T, T] - softmax over last axis (axis 2)
                let x_3d = x
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix3>()
                    .map_err(|_| {
                        RuntimeError::Other("Softmax: input must be 2D or 3D".to_string())
                    })?;
                let mut result = x_3d.to_owned();

                if *axis == 2 {
                    // Softmax along last dimension
                    for mut slice in result.axis_iter_mut(ndarray::Axis(0)) {
                        for mut row in slice.rows_mut() {
                            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            row.mapv_inplace(|v| (v - max_val).exp());
                            let sum: f32 = row.sum();
                            row.mapv_inplace(|v| v / sum);
                        }
                    }
                } else {
                    return Err(RuntimeError::Other(format!(
                        "Softmax: unsupported axis {} for 3D",
                        axis
                    )));
                }

                Ok(Tensor {
                    data: result.into_dyn(),
                })
            } else {
                Err(RuntimeError::Other(format!(
                    "Softmax: input must be 2D or 3D, got shape {:?}",
                    x.shape()
                )))
            }
        }
        Op::CrossEntropy(target_ids_idx) => {
            assert_eq!(input_values.len(), 1);
            let logits = &input_values[0];
            let target_ids = &token_ids[*target_ids_idx];

            let logits_2d = logits
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| RuntimeError::Other("CrossEntropy: logits must be 2D".to_string()))?;
            let batch = logits_2d.shape()[0];
            let num_classes = logits_2d.shape()[1];

            let mut loss_sum = 0.0;
            for (i, &target_id) in target_ids.data.iter().enumerate() {
                assert!(
                    target_id < num_classes,
                    "CrossEntropy: target_id {} >= num_classes {}",
                    target_id,
                    num_classes
                );

                let row = logits_2d.row(i);
                let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = row.iter().map(|&v| (v - max_logit).exp()).sum();
                let log_sum = max_logit + exp_sum.ln();

                let log_prob = row[target_id] - log_sum;
                loss_sum -= log_prob;
            }

            let loss = loss_sum / batch as f32;
            Ok(Tensor {
                data: ndarray::ArrayD::from_elem(IxDyn(&[]), loss),
            })
        }
        Op::LayerNorm(gamma_idx, beta_idx, eps) => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];
            let gamma = inputs[*gamma_idx].clone();
            let beta = inputs[*beta_idx].clone();

            let x_shape = x.data.shape();
            let last_dim = x_shape[x_shape.len() - 1];
            let num_elements = x.data.len();
            let num_samples = num_elements / last_dim;

            let mut x_norm = x.data.clone();
            let gamma_1d = gamma
                .data
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| RuntimeError::Other("LayerNorm: gamma must be 1D".to_string()))?;
            let beta_1d = beta
                .data
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| RuntimeError::Other("LayerNorm: beta must be 1D".to_string()))?;

            for i in 0..num_samples {
                let start = i * last_dim;
                let end = start + last_dim;
                let slice = &x.data.as_slice().unwrap()[start..end];
                let mean: f32 = slice.iter().sum::<f32>() / last_dim as f32;
                let variance: f32 =
                    slice.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / last_dim as f32;
                let std = (variance + eps).sqrt();

                for j in 0..last_dim {
                    let idx = start + j;
                    let normalized = (x.data.as_slice().unwrap()[idx] - mean) / std;
                    x_norm.as_slice_mut().unwrap()[idx] = normalized * gamma_1d[j] + beta_1d[j];
                }
            }

            Ok(Tensor { data: x_norm })
        }
        Op::MeanPoolTime => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];
            // Input: [B, T, D], Output: [B, D]
            let x_3d = x
                .data
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| {
                    RuntimeError::Other("MeanPoolTime: input must be 3D [B, T, D]".to_string())
                })?;
            let (b, t, d) = (x_3d.shape()[0], x_3d.shape()[1], x_3d.shape()[2]);

            // Mean over time axis (axis 1)
            let mut result = ndarray::Array2::<f32>::zeros((b, d));
            for i in 0..b {
                for j in 0..d {
                    let mut sum = 0.0;
                    for k in 0..t {
                        sum += x_3d[(i, k, j)];
                    }
                    result[(i, j)] = sum / t as f32;
                }
            }

            Ok(Tensor {
                data: result.into_dyn(),
            })
        }
        Op::Transpose3D => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];
            // Input: [B, T, D], Output: [B, D, T] (swap last two axes)
            let x_3d = x
                .data
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| RuntimeError::Other("Transpose3D: input must be 3D".to_string()))?;
            let (b, t, d) = (x_3d.shape()[0], x_3d.shape()[1], x_3d.shape()[2]);

            let mut result = ndarray::Array3::<f32>::zeros((b, d, t));
            for i in 0..b {
                for j in 0..t {
                    for k in 0..d {
                        result[(i, k, j)] = x_3d[(i, j, k)];
                    }
                }
            }

            Ok(Tensor {
                data: result.into_dyn(),
            })
        }
        Op::Reshape(spec) => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];
            let input_shape = x.shape();

            // Resolve symbolic shape specification at runtime
            let dim_bindings_ref = dim_bindings.map(|db| {
                let mut map = HashMap::new();
                for (k, v) in db {
                    map.insert(k.clone(), *v);
                }
                map
            });
            let resolved_shape = spec
                .resolve(input_shape, dim_bindings_ref.as_ref())
                .map_err(RuntimeError::from)?;

            let data_vec: Vec<f32> = x.data.iter().copied().collect();
            Ok(Tensor {
                data: ndarray::ArrayD::from_shape_vec(IxDyn(&resolved_shape), data_vec).map_err(
                    |e| RuntimeError::Other(format!("Reshape: failed to reshape: {:?}", e)),
                )?,
            })
        }
        Op::BatchMatMul => {
            assert_eq!(input_values.len(), 2);
            let a = &input_values[0];
            let b = &input_values[1];

            // Input A: [B, M, K], Input B: [B, K, N] -> Output: [B, M, N]
            let a_3d = a
                .data
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| {
                    RuntimeError::Other("BatchMatMul: first input must be 3D [B, M, K]".to_string())
                })?;
            let b_3d = b
                .data
                .view()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| {
                    RuntimeError::Other(
                        "BatchMatMul: second input must be 3D [B, K, N]".to_string(),
                    )
                })?;

            let (b_batch, a_m, a_k) = (a_3d.shape()[0], a_3d.shape()[1], a_3d.shape()[2]);
            let (b_batch2, b_k, b_n) = (b_3d.shape()[0], b_3d.shape()[1], b_3d.shape()[2]);

            if b_batch != b_batch2 {
                return Err(RuntimeError::Other(format!(
                    "BatchMatMul: batch sizes must match (got {} and {})",
                    b_batch, b_batch2
                )));
            }
            if a_k != b_k {
                return Err(RuntimeError::Other(format!(
                    "BatchMatMul: inner dimension K must match (got {} and {})",
                    a_k, b_k
                )));
            }

            let mut result = ndarray::Array3::<f32>::zeros((b_batch, a_m, b_n));

            // Compute per-batch: for each b, compute A[b] @ B[b]
            for batch_idx in 0..b_batch {
                let a_slice = a_3d.slice(ndarray::s![batch_idx, .., ..]);
                let b_slice = b_3d.slice(ndarray::s![batch_idx, .., ..]);

                let a_2d = a_slice.into_dimensionality::<ndarray::Ix2>().unwrap();
                let b_2d = b_slice.into_dimensionality::<ndarray::Ix2>().unwrap();

                let matmul_result = a_2d.dot(&b_2d);
                for i in 0..a_m {
                    for j in 0..b_n {
                        result[(batch_idx, i, j)] = matmul_result[(i, j)];
                    }
                }
            }

            Ok(Tensor {
                data: result.into_dyn(),
            })
        }
        Op::Now => {
            // Capability check happens in caller or early precheck
            use std::time::{SystemTime, UNIX_EPOCH};
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| RuntimeError::Other("System time error".to_string()))?;
            let seconds = now.as_secs_f32();
            Ok(Tensor {
                data: ndarray::ArrayD::from_elem(IxDyn(&[]), seconds),
            })
        }
        Op::ReadFileText(path_idx) => {
            // Capability check happens in caller or early precheck
            let path = paths.get(*path_idx).ok_or_else(|| {
                RuntimeError::Other(format!(
                    "ReadFileText: path index {} out of bounds (paths.len() = {})",
                    path_idx,
                    paths.len()
                ))
            })?;

            let file_bytes = match fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    return Err(match e.kind() {
                        std::io::ErrorKind::NotFound => FileError::NotFound { path: path.clone() },
                        _ => FileError::IOError {
                            path: path.clone(),
                            io_error_kind: format!("{:?}", e.kind()),
                        },
                    }
                    .into());
                }
            };

            let contents = match String::from_utf8(file_bytes) {
                Ok(s) => s,
                Err(_) => {
                    return Err(FileError::InvalidUTF8 { path: path.clone() }.into());
                }
            };

            let hash = crate::execute_kernel::hash_file_contents(&contents);

            let mut hash_vector = Vec::new();
            for i in 0..8 {
                let idx = i * 4;
                let chunk =
                    u32::from_le_bytes([hash[idx], hash[idx + 1], hash[idx + 2], hash[idx + 3]]);
                hash_vector.push(f32::from_bits(chunk));
            }

            Ok(Tensor {
                data: ndarray::ArrayD::from_shape_vec(IxDyn(&[8]), hash_vector)
                    .expect("ReadFileText: failed to create hash vector"),
            })
        }
        Op::Concat { axis } => {
            assert_eq!(input_values.len(), 2);
            let a = &input_values[0];
            let b = &input_values[1];

            // Only support 2D tensors and axis=1
            if *axis != 1 {
                return Err(RuntimeError::Other(format!(
                    "Concat: only axis=1 is supported (got {})",
                    axis
                )));
            }

            let a_2d = a
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| RuntimeError::Other("Concat: first input must be 2D".to_string()))?;
            let b_2d = b
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| RuntimeError::Other("Concat: second input must be 2D".to_string()))?;

            let (a_rows, a_cols) = (a_2d.shape()[0], a_2d.shape()[1]);
            let (b_rows, b_cols) = (b_2d.shape()[0], b_2d.shape()[1]);

            if a_rows != b_rows {
                return Err(RuntimeError::Other(format!(
                    "Concat: row counts must match (got {} and {})",
                    a_rows, b_rows
                )));
            }

            // Concatenate along axis=1 (columns)
            let mut result = ndarray::Array2::<f32>::zeros((a_rows, a_cols + b_cols));
            result.slice_mut(ndarray::s![.., ..a_cols]).assign(&a_2d);
            result.slice_mut(ndarray::s![.., a_cols..]).assign(&b_2d);

            Ok(Tensor {
                data: result.into_dyn(),
            })
        }
        Op::SliceRows { start, len } => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];

            let x_2d = x
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| RuntimeError::Other("SliceRows: input must be 2D".to_string()))?;

            let rows = x_2d.shape()[0];
            if *start + *len > rows {
                return Err(RuntimeError::Other(format!(
                    "SliceRows: start {} + len {} exceeds rows {}",
                    start, len, rows
                )));
            }

            let sliced = x_2d
                .slice(ndarray::s![*start..*start + *len, ..])
                .to_owned();

            Ok(Tensor {
                data: sliced.into_dyn(),
            })
        }
        Op::GatherRows => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];
            // indices come from token_ids (first token_ids input)
            // For MVP, we use token_ids[0] for the indices
            if token_ids.is_empty() {
                return Err(RuntimeError::Other(
                    "GatherRows: requires token_ids indices".to_string(),
                ));
            }
            let indices = &token_ids[0];

            let x_2d = x
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| RuntimeError::Other("GatherRows: input must be 2D".to_string()))?;

            let (rows, cols) = (x_2d.shape()[0], x_2d.shape()[1]);
            let k = indices.data.len();

            // Validate indices
            for &idx in &indices.data {
                if idx >= rows {
                    return Err(RuntimeError::Other(format!(
                        "GatherRows: index {} out of range [0, {})",
                        idx, rows
                    )));
                }
            }

            let mut result = ndarray::Array2::<f32>::zeros((k, cols));
            for (i, &idx) in indices.data.iter().enumerate() {
                result.row_mut(i).assign(&x_2d.row(idx));
            }

            Ok(Tensor {
                data: result.into_dyn(),
            })
        }
        Op::Dropout { p } => {
            assert_eq!(input_values.len(), 1);
            let x = &input_values[0];

            // Dropout is training-only, but we need to support inference mode
            // For now, we'll always apply dropout (training mode)
            // In a real system, we'd have a training flag, but for MVP we always train

            // Validate p
            if *p < 0.0 || *p >= 1.0 {
                return Err(RuntimeError::Other(format!(
                    "Dropout: p must be in [0.0, 1.0) (got {})",
                    p
                )));
            }

            // For deterministic dropout, we need a seed
            // We'll use a hash of the tensor data as a seed for the mask
            // This ensures determinism but is not ideal - in a real system we'd pass seed
            // For MVP, we'll use a simple deterministic approach
            let scale = 1.0 / (1.0 - p);
            let mut result = x.data.clone();

            // Generate deterministic mask based on tensor position
            // This is a simplified approach - in practice we'd use a proper RNG with seed
            let mut mask_seed = 0u64;
            for (i, &val) in x.data.iter().enumerate() {
                mask_seed = mask_seed
                    .wrapping_add(i as u64)
                    .wrapping_add(val.to_bits() as u64);
            }

            // Simple deterministic "random" based on seed
            let mut rng_state = mask_seed;
            for val in result.iter_mut() {
                // Simple LCG for deterministic "random"
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let rand_val = (rng_state >> 16) as f32 / 65536.0;
                if rand_val < *p {
                    *val = 0.0;
                } else {
                    *val *= scale;
                }
            }

            Ok(Tensor { data: result })
        }
        Op::If { .. } => {
            assert_eq!(input_values.len(), 3);
            let cond = &input_values[0];
            let then_val = &input_values[1];
            let else_val = &input_values[2];

            // Condition must be scalar-compatible
            let cond_scalar: f32 = if cond.data.len() == 1 {
                cond.data.iter().next().copied().unwrap_or(0.0)
            } else {
                // For non-scalar, check if any element is non-zero
                cond.data.iter().any(|&v| v != 0.0) as u8 as f32
            };

            // Both branches must have same shape
            if then_val.data.shape() != else_val.data.shape() {
                return Err(RuntimeError::Other(format!(
                    "If: branch shape mismatch: then {:?} vs else {:?}",
                    then_val.data.shape(),
                    else_val.data.shape()
                )));
            }

            // Select branch based on condition (non-zero = true)
            if cond_scalar != 0.0 {
                Ok(then_val.clone())
            } else {
                Ok(else_val.clone())
            }
        }
        Op::Compare { op } => {
            assert_eq!(input_values.len(), 2);
            let left = &input_values[0];
            let right = &input_values[1];

            // For scalar comparison
            if left.data.len() == 1 && right.data.len() == 1 {
                let left_val = left.data.iter().next().copied().unwrap_or(0.0);
                let right_val = right.data.iter().next().copied().unwrap_or(0.0);
                let result = match op {
                    CompareOp::Eq => (left_val == right_val) as u8 as f32,
                    CompareOp::Ne => (left_val != right_val) as u8 as f32,
                    CompareOp::Lt => (left_val < right_val) as u8 as f32,
                    CompareOp::Le => (left_val <= right_val) as u8 as f32,
                    CompareOp::Gt => (left_val > right_val) as u8 as f32,
                    CompareOp::Ge => (left_val >= right_val) as u8 as f32,
                };
                Ok(Tensor {
                    data: ndarray::ArrayD::from_elem(IxDyn(&[]), result),
                })
            } else {
                // Element-wise comparison (broadcast if needed)
                // For MVP, require same shape
                if left.data.shape() != right.data.shape() {
                    return Err(RuntimeError::Other(
                        "Compare: shape mismatch (element-wise comparison requires same shape)"
                            .to_string(),
                    ));
                }
                let result = left.data.iter().zip(right.data.iter()).map(|(l, r)| {
                    let res = match op {
                        CompareOp::Eq => (*l == *r) as u8 as f32,
                        CompareOp::Ne => (*l != *r) as u8 as f32,
                        CompareOp::Lt => (*l < *r) as u8 as f32,
                        CompareOp::Le => (*l <= *r) as u8 as f32,
                        CompareOp::Gt => (*l > *r) as u8 as f32,
                        CompareOp::Ge => (*l >= *r) as u8 as f32,
                    };
                    res
                });
                Ok(Tensor {
                    data: ndarray::ArrayD::from_shape_vec(left.data.shape(), result.collect())
                        .map_err(|e| RuntimeError::Other(format!("Compare: shape error: {}", e)))?,
                })
            }
        }
        Op::Logical { op } => {
            match op {
                LogicalOp::Not => {
                    assert_eq!(input_values.len(), 1);
                    let x = &input_values[0];
                    // Logical not: 0 -> 1, non-zero -> 0
                    let result = if x.data.len() == 1 {
                        let val = x.data.iter().next().copied().unwrap_or(0.0);
                        (val == 0.0) as u8 as f32
                    } else {
                        // For non-scalar, check if all elements are zero
                        x.data.iter().all(|&v| v == 0.0) as u8 as f32
                    };
                    Ok(Tensor {
                        data: ndarray::ArrayD::from_elem(IxDyn(&[]), result),
                    })
                }
                LogicalOp::And => {
                    assert_eq!(input_values.len(), 2);
                    let left = &input_values[0];
                    let right = &input_values[1];
                    // Logical and: both non-zero -> 1, else -> 0
                    let left_val = if left.data.len() == 1 {
                        left.data.iter().next().copied().unwrap_or(0.0)
                    } else {
                        left.data.iter().any(|&v| v != 0.0) as u8 as f32
                    };
                    let right_val = if right.data.len() == 1 {
                        right.data.iter().next().copied().unwrap_or(0.0)
                    } else {
                        right.data.iter().any(|&v| v != 0.0) as u8 as f32
                    };
                    let result = ((left_val != 0.0) && (right_val != 0.0)) as u8 as f32;
                    Ok(Tensor {
                        data: ndarray::ArrayD::from_elem(IxDyn(&[]), result),
                    })
                }
                LogicalOp::Or => {
                    assert_eq!(input_values.len(), 2);
                    let left = &input_values[0];
                    let right = &input_values[1];
                    // Logical or: either non-zero -> 1, else -> 0
                    let left_val = if left.data.len() == 1 {
                        left.data.iter().next().copied().unwrap_or(0.0)
                    } else {
                        left.data.iter().any(|&v| v != 0.0) as u8 as f32
                    };
                    let right_val = if right.data.len() == 1 {
                        right.data.iter().next().copied().unwrap_or(0.0)
                    } else {
                        right.data.iter().any(|&v| v != 0.0) as u8 as f32
                    };
                    let result = ((left_val != 0.0) || (right_val != 0.0)) as u8 as f32;
                    Ok(Tensor {
                        data: ndarray::ArrayD::from_elem(IxDyn(&[]), result),
                    })
                }
            }
        }
        Op::ConstScalar { value } => {
            // Create a scalar tensor
            Ok(Tensor {
                data: ndarray::ArrayD::from_elem(IxDyn(&[]), *value),
            })
        }
        Op::Stack { axis } => {
            if input_values.is_empty() {
                return Err(RuntimeError::Other(
                    "Stack: requires at least one input".to_string(),
                ));
            }

            if *axis != 0 {
                return Err(RuntimeError::Other(format!(
                    "Stack: only axis=0 is supported (got {})",
                    axis
                )));
            }

            // Check all inputs have same shape
            let first_shape = input_values[0].data.shape();
            for (i, val) in input_values.iter().enumerate().skip(1) {
                if val.data.shape() != first_shape {
                    return Err(RuntimeError::Other(format!(
                        "Stack: shape mismatch at input {}: {:?} vs {:?}",
                        i,
                        val.data.shape(),
                        first_shape
                    )));
                }
            }

            // Stack along axis 0
            let mut stacked_shape = vec![input_values.len()];
            stacked_shape.extend_from_slice(first_shape);

            let mut stacked_data = Vec::new();
            for val in input_values {
                stacked_data.extend_from_slice(val.data.as_slice().unwrap());
            }

            Ok(Tensor {
                data: ndarray::ArrayD::from_shape_vec(IxDyn(&stacked_shape), stacked_data)
                    .map_err(|e| RuntimeError::Other(format!("Stack: shape error: {}", e)))?,
            })
        }
        Op::Max2 => {
            assert_eq!(input_values.len(), 2);
            let left = &input_values[0];
            let right = &input_values[1];

            if left.data.shape() != right.data.shape() {
                return Err(RuntimeError::Other(format!(
                    "Max2: shape mismatch: {:?} vs {:?}",
                    left.data.shape(),
                    right.data.shape()
                )));
            }

            let result = left
                .data
                .iter()
                .zip(right.data.iter())
                .map(|(l, r)| l.max(*r))
                .collect::<Vec<_>>();

            Ok(Tensor {
                data: ndarray::ArrayD::from_shape_vec(left.data.shape(), result)
                    .map_err(|e| RuntimeError::Other(format!("Max2: shape error: {}", e)))?,
            })
        }
        Op::Min2 => {
            assert_eq!(input_values.len(), 2);
            let left = &input_values[0];
            let right = &input_values[1];

            if left.data.shape() != right.data.shape() {
                return Err(RuntimeError::Other(format!(
                    "Min2: shape mismatch: {:?} vs {:?}",
                    left.data.shape(),
                    right.data.shape()
                )));
            }

            let result = left
                .data
                .iter()
                .zip(right.data.iter())
                .map(|(l, r)| l.min(*r))
                .collect::<Vec<_>>();

            Ok(Tensor {
                data: ndarray::ArrayD::from_shape_vec(left.data.shape(), result)
                    .map_err(|e| RuntimeError::Other(format!("Min2: shape error: {}", e)))?,
            })
        }
    }
}

// Deterministic hash function for file contents
// Uses FNV-1a hash expanded to 256 bits (32 bytes)
pub(crate) fn hash_file_contents(contents: &str) -> [u8; 32] {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in contents.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    let mut result = [0u8; 32];
    let bytes = hash.to_le_bytes();
    result[0..8].copy_from_slice(&bytes);

    for i in 1..4 {
        let seed = (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
        let mut h = hash.wrapping_add(seed);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        let bytes = h.to_le_bytes();
        result[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }

    result
}

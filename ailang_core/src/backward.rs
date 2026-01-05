use crate::ir::{Graph, NodeId, Op};
use crate::tensor::{Tensor, TokenIds};
use ndarray::{Axis, IxDyn};

pub fn execute_backward(
    graph: &Graph,
    activations: &[Tensor],
    loss_id: NodeId,
    token_ids: &[TokenIds],
) -> Vec<Option<Tensor>> {
    let mut grads: Vec<Option<Tensor>> = vec![None; graph.nodes.len()];

    // Initialize loss gradient to 1.0
    grads[loss_id] = Some(Tensor {
        data: ndarray::ArrayD::from_elem(IxDyn(&[]), 1.0),
    });

    // Traverse graph in reverse order
    for node_id in (0..graph.nodes.len()).rev() {
        if grads[node_id].is_none() {
            continue;
        }
        let grad_out = grads[node_id].as_ref().unwrap().clone();
        let node = &graph.nodes[node_id];

        match &node.op {
            Op::Input(_) => {
                // Inputs don't propagate gradients (they're leaves)
            }
            Op::Add => {
                assert_eq!(node.inputs.len(), 2);
                for &input_id in &node.inputs {
                    if grads[input_id].is_none() {
                        grads[input_id] = Some(grad_out.clone());
                    } else {
                        // Accumulate gradients
                        let existing = grads[input_id].take().unwrap();
                        grads[input_id] = Some(Tensor {
                            data: &existing.data + &grad_out.data,
                        });
                    }
                }
            }
            Op::Sub => {
                assert_eq!(node.inputs.len(), 2);
                // d/dx (x - y) = grad_out
                // d/dy (x - y) = -grad_out
                if grads[node.inputs[0]].is_none() {
                    grads[node.inputs[0]] = Some(grad_out.clone());
                } else {
                    let existing = grads[node.inputs[0]].take().unwrap();
                    grads[node.inputs[0]] = Some(Tensor {
                        data: &existing.data + &grad_out.data,
                    });
                }

                let neg_grad = Tensor {
                    data: -&grad_out.data,
                };
                if grads[node.inputs[1]].is_none() {
                    grads[node.inputs[1]] = Some(neg_grad);
                } else {
                    let existing = grads[node.inputs[1]].take().unwrap();
                    grads[node.inputs[1]] = Some(Tensor {
                        data: &existing.data - &grad_out.data,
                    });
                }
            }
            Op::Mul => {
                assert_eq!(node.inputs.len(), 2);
                let a = &activations[node.inputs[0]];
                let b = &activations[node.inputs[1]];

                // d/dx (x * y) = y * grad_out
                // d/dy (x * y) = x * grad_out
                let grad_a = Tensor {
                    data: &b.data * &grad_out.data,
                };
                let grad_b = Tensor {
                    data: &a.data * &grad_out.data,
                };

                for (i, &input_id) in node.inputs.iter().enumerate() {
                    let grad = if i == 0 { &grad_a } else { &grad_b };
                    if grads[input_id].is_none() {
                        grads[input_id] = Some(grad.clone());
                    } else {
                        let existing = grads[input_id].take().unwrap();
                        grads[input_id] = Some(Tensor {
                            data: &existing.data + &grad.data,
                        });
                    }
                }
            }
            Op::MatMul2D => {
                assert_eq!(node.inputs.len(), 2);
                let a = &activations[node.inputs[0]];
                let b = &activations[node.inputs[1]];

                let grad_2d = grad_out
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("MatMul2D grad: gradient must be 2D");
                let a_2d = a
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("MatMul2D grad: first input must be 2D");
                let b_2d = b
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("MatMul2D grad: second input must be 2D");

                // d/dA (A @ B) = grad_out @ B^T
                // d/dB (A @ B) = A^T @ grad_out
                let grad_a = Tensor {
                    data: grad_2d.dot(&b_2d.t()).into_dyn(),
                };
                let a_t = a_2d.t();
                let grad_b = Tensor {
                    data: a_t.dot(&grad_2d).into_dyn(),
                };

                for (i, &input_id) in node.inputs.iter().enumerate() {
                    let grad = if i == 0 { &grad_a } else { &grad_b };
                    if grads[input_id].is_none() {
                        grads[input_id] = Some(grad.clone());
                    } else {
                        let existing = grads[input_id].take().unwrap();
                        grads[input_id] = Some(Tensor {
                            data: &existing.data + &grad.data,
                        });
                    }
                }
            }
            Op::ReLU => {
                assert_eq!(node.inputs.len(), 1);
                let x = &activations[node.inputs[0]];
                // d/dx ReLU(x) = 1 if x > 0, else 0
                let mask = x.data.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
                let grad_in = Tensor {
                    data: &mask * &grad_out.data,
                };

                let input_id = node.inputs[0];
                if grads[input_id].is_none() {
                    grads[input_id] = Some(grad_in);
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: &existing.data + &grad_in.data,
                    });
                }
            }
            Op::Sum => {
                assert_eq!(node.inputs.len(), 1);
                let x = &activations[node.inputs[0]];
                // d/dx sum(x) = ones_like(x) * grad_out (scalar)
                let grad_val = grad_out.scalar();
                let grad_in = Tensor {
                    data: ndarray::ArrayD::from_elem(x.shape(), grad_val),
                };

                let input_id = node.inputs[0];
                if grads[input_id].is_none() {
                    grads[input_id] = Some(grad_in);
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: &existing.data + &grad_in.data,
                    });
                }
            }
            Op::Mean => {
                assert_eq!(node.inputs.len(), 1);
                let x = &activations[node.inputs[0]];
                // d/dx mean(x) = ones_like(x) * grad_out / num_elements
                let grad_val = grad_out.scalar();
                let num_elements = x.num_elements() as f32;
                let grad_in = Tensor {
                    data: ndarray::ArrayD::from_elem(x.shape(), grad_val / num_elements),
                };

                let input_id = node.inputs[0];
                if grads[input_id].is_none() {
                    grads[input_id] = Some(grad_in);
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: &existing.data + &grad_in.data,
                    });
                }
            }
            Op::Embedding(token_ids_idx) => {
                assert_eq!(node.inputs.len(), 1);
                let weight = &activations[node.inputs[0]];
                let ids = &token_ids[*token_ids_idx];

                // grad_out: [batch, time, dim] or [batch, dim], accumulate into weight: [vocab, dim]
                let weight_shape = weight.shape();
                let vocab_size = weight_shape[0];
                let dim = weight_shape[1];

                let batch = ids.shape()[0];
                let time = if ids.shape().len() == 2 {
                    ids.shape()[1]
                } else {
                    1
                };

                // Initialize gradient accumulator
                let mut grad_weight = ndarray::Array2::<f32>::zeros((vocab_size, dim));

                // Accumulate gradients for each (batch, time) position
                let grad_flat = grad_out.data.as_slice().unwrap();
                for b in 0..batch {
                    for t in 0..time {
                        let id_idx = if ids.shape().len() == 2 {
                            b * time + t
                        } else {
                            b
                        };
                        let id = ids.data[id_idx];
                        let grad_offset = if ids.shape().len() == 2 {
                            (b * time + t) * dim
                        } else {
                            b * dim
                        };
                        for d in 0..dim {
                            grad_weight[(id, d)] += grad_flat[grad_offset + d];
                        }
                    }
                }

                let input_id = node.inputs[0];
                if grads[input_id].is_none() {
                    grads[input_id] = Some(Tensor {
                        data: grad_weight.into_dyn(),
                    });
                } else {
                    let existing = grads[input_id].take().unwrap();
                    let existing_2d = existing
                        .data
                        .view()
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    grads[input_id] = Some(Tensor {
                        data: (&existing_2d + &grad_weight).into_dyn(),
                    });
                }
            }
            Op::Softmax(axis) => {
                assert_eq!(node.inputs.len(), 1);
                let softmax_output = &activations[node_id]; // The output of this softmax node

                // Gradient of softmax: grad_out * softmax - softmax * sum(grad_out * softmax)
                let axis_idx = Axis(*axis);
                let softmax_grad_out = &softmax_output.data * &grad_out.data;
                let sum_per_sample: ndarray::ArrayD<f32> = softmax_grad_out
                    .map_axis(axis_idx, |row| row.sum())
                    .into_dyn();

                let mut grad_in = softmax_grad_out.clone();
                let mut sum_iter = sum_per_sample.iter();
                let mut softmax_iter = softmax_output.data.axis_iter(axis_idx);
                for mut slice in grad_in.axis_iter_mut(axis_idx) {
                    let sum_val = sum_iter.next().unwrap();
                    let softmax_slice = softmax_iter.next().unwrap();
                    slice.zip_mut_with(&softmax_slice, |g, &s| *g = *g - s * sum_val);
                }

                let input_id = node.inputs[0];
                if grads[input_id].is_none() {
                    grads[input_id] = Some(Tensor { data: grad_in });
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: &existing.data + &grad_in,
                    });
                }
            }
            Op::CrossEntropy(target_ids_idx) => {
                assert_eq!(node.inputs.len(), 1);
                let logits = &activations[node.inputs[0]];
                let target_ids = &token_ids[*target_ids_idx];

                // Compute softmax
                let logits_2d = logits
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("CrossEntropy backward: logits must be 2D");
                let batch = logits_2d.shape()[0];

                // Compute softmax probabilities
                let mut softmax = logits_2d.to_owned();
                for mut row in softmax.rows_mut() {
                    let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    row.mapv_inplace(|v| (v - max_logit).exp());
                    let sum: f32 = row.sum();
                    row.mapv_inplace(|v| v / sum);
                }

                // Gradient: (softmax - one_hot) / batch
                let mut grad_logits = softmax.clone();
                for (i, &target_id) in target_ids.data.iter().enumerate() {
                    grad_logits[(i, target_id)] -= 1.0;
                }
                grad_logits.mapv_inplace(|v| v / batch as f32);

                let input_id = node.inputs[0];
                let grad_scalar = grad_out.scalar();
                if grads[input_id].is_none() {
                    grads[input_id] = Some(Tensor {
                        data: (&grad_logits * grad_scalar).into_dyn(),
                    });
                } else {
                    let existing = grads[input_id].take().unwrap();
                    let existing_2d = existing
                        .data
                        .view()
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    grads[input_id] = Some(Tensor {
                        data: (&existing_2d + &(&grad_logits * grad_scalar)).into_dyn(),
                    });
                }
            }
            Op::LayerNorm(_gamma_idx, _beta_idx, _eps) => {
                // Simplified: for now, just pass gradient through (no learnable params)
                // Full implementation would compute gradients w.r.t. x, gamma, beta
                assert_eq!(node.inputs.len(), 1);
                let input_id = node.inputs[0];
                if grads[input_id].is_none() {
                    grads[input_id] = Some(grad_out.clone());
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: &existing.data + &grad_out.data,
                    });
                }
            }
            Op::MeanPoolTime => {
                assert_eq!(node.inputs.len(), 1);
                let x = &activations[node.inputs[0]];
                // grad_out: [B, D], need to distribute equally across T dimension
                // grad_in: [B, T, D] where each [B, T, :] gets grad_out[B, :] / T
                let grad_out_2d = grad_out
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("MeanPoolTime backward: grad_out must be 2D [B, D]");
                let (b, d) = (grad_out_2d.shape()[0], grad_out_2d.shape()[1]);
                let x_3d = x
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix3>()
                    .expect("MeanPoolTime backward: input must be 3D [B, T, D]");
                let t = x_3d.shape()[1];

                let mut grad_in = ndarray::Array3::<f32>::zeros((b, t, d));
                for i in 0..b {
                    for j in 0..t {
                        for k in 0..d {
                            grad_in[(i, j, k)] = grad_out_2d[(i, k)] / t as f32;
                        }
                    }
                }

                let input_id = node.inputs[0];
                if grads[input_id].is_none() {
                    grads[input_id] = Some(Tensor {
                        data: grad_in.into_dyn(),
                    });
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: &existing.data + &grad_in.into_dyn(),
                    });
                }
            }
            Op::Transpose3D => {
                assert_eq!(node.inputs.len(), 1);
                // Transpose backward: just transpose again
                let grad_out_3d = grad_out
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix3>()
                    .expect("Transpose3D backward: grad_out must be 3D [B, D, T]");
                let (b, d, t) = (
                    grad_out_3d.shape()[0],
                    grad_out_3d.shape()[1],
                    grad_out_3d.shape()[2],
                );

                let mut grad_in = ndarray::Array3::<f32>::zeros((b, t, d));
                for i in 0..b {
                    for j in 0..t {
                        for k in 0..d {
                            grad_in[(i, j, k)] = grad_out_3d[(i, k, j)];
                        }
                    }
                }

                let input_id = node.inputs[0];
                if grads[input_id].is_none() {
                    grads[input_id] = Some(Tensor {
                        data: grad_in.into_dyn(),
                    });
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: &existing.data + &grad_in.into_dyn(),
                    });
                }
            }
            Op::Reshape(_spec) => {
                assert_eq!(node.inputs.len(), 1);
                let x = &activations[node.inputs[0]];
                // Reshape backward: reshape gradient back to input shape
                let input_shape = x.shape();
                let grad_data: Vec<f32> = grad_out.data.iter().copied().collect();
                let grad_reshaped = ndarray::ArrayD::from_shape_vec(IxDyn(input_shape), grad_data)
                    .expect("Reshape backward: failed to reshape gradient");

                let input_id = node.inputs[0];
                if grads[input_id].is_none() {
                    grads[input_id] = Some(Tensor {
                        data: grad_reshaped,
                    });
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: &existing.data + &grad_reshaped,
                    });
                }
            }
            Op::BatchMatMul => {
                assert_eq!(node.inputs.len(), 2);
                let a = &activations[node.inputs[0]];
                let b = &activations[node.inputs[1]];

                let grad_3d = grad_out
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix3>()
                    .expect("BatchMatMul backward: grad_out must be 3D [B, M, N]");
                let a_3d = a
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix3>()
                    .expect("BatchMatMul backward: first input must be 3D [B, M, K]");
                let b_3d = b
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix3>()
                    .expect("BatchMatMul backward: second input must be 3D [B, K, N]");

                let batch = grad_3d.shape()[0];
                let (a_m, a_k) = (a_3d.shape()[1], a_3d.shape()[2]);
                let (b_k, b_n) = (b_3d.shape()[1], b_3d.shape()[2]);

                // Gradients: per-batch matmul gradients
                let mut grad_a = ndarray::Array3::<f32>::zeros((batch, a_m, a_k));
                let mut grad_b = ndarray::Array3::<f32>::zeros((batch, b_k, b_n));

                for batch_idx in 0..batch {
                    let grad_slice = grad_3d.slice(ndarray::s![batch_idx, .., ..]);
                    let a_slice = a_3d.slice(ndarray::s![batch_idx, .., ..]);
                    let b_slice = b_3d.slice(ndarray::s![batch_idx, .., ..]);

                    let grad_2d = grad_slice.into_dimensionality::<ndarray::Ix2>().unwrap();
                    let a_2d = a_slice.into_dimensionality::<ndarray::Ix2>().unwrap();
                    let b_2d = b_slice.into_dimensionality::<ndarray::Ix2>().unwrap();

                    // d/dA (A @ B) = grad_out @ B^T
                    // d/dB (A @ B) = A^T @ grad_out
                    let grad_a_2d = grad_2d.dot(&b_2d.t());
                    let grad_b_2d = a_2d.t().dot(&grad_2d);

                    for i in 0..a_m {
                        for j in 0..a_k {
                            grad_a[(batch_idx, i, j)] = grad_a_2d[(i, j)];
                        }
                    }
                    for i in 0..b_k {
                        for j in 0..b_n {
                            grad_b[(batch_idx, i, j)] = grad_b_2d[(i, j)];
                        }
                    }
                }

                for (i, &input_id) in node.inputs.iter().enumerate() {
                    let grad = if i == 0 { &grad_a } else { &grad_b };
                    if grads[input_id].is_none() {
                        grads[input_id] = Some(Tensor {
                            data: grad.clone().into_dyn(),
                        });
                    } else {
                        let existing = grads[input_id].take().unwrap();
                        let existing_3d = existing
                            .data
                            .view()
                            .into_dimensionality::<ndarray::Ix3>()
                            .unwrap();
                        grads[input_id] = Some(Tensor {
                            data: (&existing_3d + grad).into_dyn(),
                        });
                    }
                }
            }
            Op::Now => {
                // Now has no inputs and no backward pass
                // This should never be reached in backward pass, but handle gracefully
            }
            Op::ReadFileText(_) => {
                // ReadFileText has no backward pass
                // This should never be reached in backward pass, but handle gracefully
            }
            Op::Concat { axis } => {
                assert_eq!(node.inputs.len(), 2);
                if *axis != 1 {
                    // Should not happen if forward validation worked
                    return grads;
                }

                let a = &activations[node.inputs[0]];
                let a_2d = a
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("Concat backward: first input must be 2D");
                let a_cols = a_2d.shape()[1];

                let grad_out_2d = grad_out
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("Concat backward: grad_out must be 2D");

                // dA = dOut[:, :D1]
                let grad_a = grad_out_2d.slice(ndarray::s![.., ..a_cols]).to_owned();
                // dB = dOut[:, D1:]
                let grad_b = grad_out_2d.slice(ndarray::s![.., a_cols..]).to_owned();

                for (i, &input_id) in node.inputs.iter().enumerate() {
                    let grad = if i == 0 {
                        grad_a.clone()
                    } else {
                        grad_b.clone()
                    };
                    if grads[input_id].is_none() {
                        grads[input_id] = Some(Tensor {
                            data: grad.into_dyn(),
                        });
                    } else {
                        let existing = grads[input_id].take().unwrap();
                        let existing_2d = existing
                            .data
                            .view()
                            .into_dimensionality::<ndarray::Ix2>()
                            .expect("Concat backward: existing grad must be 2D");
                        grads[input_id] = Some(Tensor {
                            data: (&existing_2d + &grad).into_dyn(),
                        });
                    }
                }
            }
            Op::SliceRows { start, len } => {
                assert_eq!(node.inputs.len(), 1);
                let input_id = node.inputs[0];
                let x = &activations[input_id];

                let x_2d = x
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("SliceRows backward: input must be 2D");
                let (rows, cols) = (x_2d.shape()[0], x_2d.shape()[1]);

                let grad_out_2d = grad_out
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("SliceRows backward: grad_out must be 2D");

                // Scatter gradients back into original tensor
                let mut grad_in = ndarray::Array2::<f32>::zeros((rows, cols));
                grad_in
                    .slice_mut(ndarray::s![*start..*start + *len, ..])
                    .assign(&grad_out_2d);

                if grads[input_id].is_none() {
                    grads[input_id] = Some(Tensor {
                        data: grad_in.into_dyn(),
                    });
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: (&existing.data + &grad_in).into_dyn(),
                    });
                }
            }
            Op::GatherRows => {
                assert_eq!(node.inputs.len(), 1);
                let input_id = node.inputs[0];
                let x = &activations[input_id];

                // Get indices from token_ids[0]
                if token_ids.is_empty() {
                    return grads;
                }
                let indices = &token_ids[0];

                let x_2d = x
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("GatherRows backward: input must be 2D");
                let (rows, cols) = (x_2d.shape()[0], x_2d.shape()[1]);

                let grad_out_2d = grad_out
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .expect("GatherRows backward: grad_out must be 2D");

                // Accumulate gradients into original tensor
                // For repeated indices, we accumulate
                let mut grad_in = ndarray::Array2::<f32>::zeros((rows, cols));
                for (i, &idx) in indices.data.iter().enumerate() {
                    if idx < rows {
                        let grad_row = grad_out_2d.row(i);
                        let mut target_row = grad_in.row_mut(idx);
                        for j in 0..cols {
                            target_row[j] += grad_row[j];
                        }
                    }
                }

                if grads[input_id].is_none() {
                    grads[input_id] = Some(Tensor {
                        data: grad_in.into_dyn(),
                    });
                } else {
                    let existing = grads[input_id].take().unwrap();
                    let existing_2d = existing
                        .data
                        .view()
                        .into_dimensionality::<ndarray::Ix2>()
                        .expect("GatherRows backward: existing grad must be 2D");
                    grads[input_id] = Some(Tensor {
                        data: (&existing_2d + &grad_in).into_dyn(),
                    });
                }
            }
            Op::If { .. } => {
                // If: gradient flows to the executed branch only
                // For simplicity, we propagate to both branches (this is correct for differentiable conditionals)
                assert_eq!(node.inputs.len(), 3);
                let _cond_id = node.inputs[0];
                let then_id = node.inputs[1];
                let else_id = node.inputs[2];
                // Propagate gradient to both branches (the condition doesn't get gradients)
                if grads[then_id].is_none() {
                    grads[then_id] = Some(grad_out.clone());
                } else {
                    let existing = grads[then_id].take().unwrap();
                    grads[then_id] = Some(Tensor {
                        data: &existing.data + &grad_out.data,
                    });
                }
                if grads[else_id].is_none() {
                    grads[else_id] = Some(grad_out.clone());
                } else {
                    let existing = grads[else_id].take().unwrap();
                    grads[else_id] = Some(Tensor {
                        data: &existing.data + &grad_out.data,
                    });
                }
            }
            Op::Compare { .. } => {
                // Comparisons don't propagate gradients (they're boolean operations)
                // No gradient flow
            }
            Op::Logical { .. } => {
                // Logical operations don't propagate gradients (they're boolean operations)
                // No gradient flow
            }
            Op::ConstScalar { .. } => {
                // Constants don't propagate gradients
            }
            Op::Stack { axis } => {
                // Stack backward: split gradient along axis and accumulate into each child
                assert_eq!(*axis, 0); // Only axis 0 supported

                // Get output shape
                let out_shape = grad_out.data.shape();
                if !out_shape.is_empty() {
                    // Split along axis 0
                    let inner_shape = &out_shape[1..];
                    let inner_size: usize = inner_shape.iter().product();

                    for (i, &input_id) in node.inputs.iter().enumerate() {
                        // Extract slice for this input
                        let start = i * inner_size;
                        let end = (i + 1) * inner_size;
                        let slice_data: Vec<f32> =
                            grad_out.data.as_slice().unwrap()[start..end].to_vec();

                        let grad_slice = Tensor {
                            data: ndarray::ArrayD::from_shape_vec(inner_shape, slice_data)
                                .unwrap_or_else(|_| {
                                    // Fallback: create zero tensor
                                    ndarray::ArrayD::from_elem(IxDyn(inner_shape), 0.0)
                                }),
                        };

                        if grads[input_id].is_none() {
                            grads[input_id] = Some(grad_slice);
                        } else {
                            let existing = grads[input_id].take().unwrap();
                            grads[input_id] = Some(Tensor {
                                data: &existing.data + &grad_slice.data,
                            });
                        }
                    }
                }
            }
            Op::Max2 => {
                // Max2 backward: gradient goes to the maximum element
                assert_eq!(node.inputs.len(), 2);
                let left_id = node.inputs[0];
                let right_id = node.inputs[1];

                let left_val = &activations[left_id];
                let right_val = &activations[right_id];

                // Create mask: 1.0 where left >= right, 0.0 otherwise
                let left_mask: Vec<f32> = left_val
                    .data
                    .iter()
                    .zip(right_val.data.iter())
                    .map(|(l, r)| if l >= r { 1.0 } else { 0.0 })
                    .collect();

                let right_mask: Vec<f32> = left_val
                    .data
                    .iter()
                    .zip(right_val.data.iter())
                    .map(|(l, r)| if r > l { 1.0 } else { 0.0 })
                    .collect();

                let left_mask_arr =
                    match ndarray::ArrayD::from_shape_vec(left_val.data.shape(), left_mask) {
                        Ok(arr) => arr,
                        Err(_) => continue, // Skip if shape mismatch
                    };

                let right_mask_arr =
                    match ndarray::ArrayD::from_shape_vec(right_val.data.shape(), right_mask) {
                        Ok(arr) => arr,
                        Err(_) => continue, // Skip if shape mismatch
                    };

                let left_grad = Tensor {
                    data: (&grad_out.data * &left_mask_arr).into_dyn(),
                };

                let right_grad = Tensor {
                    data: (&grad_out.data * &right_mask_arr).into_dyn(),
                };

                if grads[left_id].is_none() {
                    grads[left_id] = Some(left_grad);
                } else {
                    let existing = grads[left_id].take().unwrap();
                    grads[left_id] = Some(Tensor {
                        data: &existing.data + &left_grad.data,
                    });
                }

                if grads[right_id].is_none() {
                    grads[right_id] = Some(right_grad);
                } else {
                    let existing = grads[right_id].take().unwrap();
                    grads[right_id] = Some(Tensor {
                        data: &existing.data + &right_grad.data,
                    });
                }
            }
            Op::Min2 => {
                // Min2 backward: gradient goes to the minimum element
                assert_eq!(node.inputs.len(), 2);
                let left_id = node.inputs[0];
                let right_id = node.inputs[1];

                let left_val = &activations[left_id];
                let right_val = &activations[right_id];

                // Create mask: 1.0 where left <= right, 0.0 otherwise
                let left_mask: Vec<f32> = left_val
                    .data
                    .iter()
                    .zip(right_val.data.iter())
                    .map(|(l, r)| if l <= r { 1.0 } else { 0.0 })
                    .collect();

                let right_mask: Vec<f32> = left_val
                    .data
                    .iter()
                    .zip(right_val.data.iter())
                    .map(|(l, r)| if r < l { 1.0 } else { 0.0 })
                    .collect();

                let left_mask_arr =
                    match ndarray::ArrayD::from_shape_vec(left_val.data.shape(), left_mask) {
                        Ok(arr) => arr,
                        Err(_) => continue, // Skip if shape mismatch
                    };

                let right_mask_arr =
                    match ndarray::ArrayD::from_shape_vec(right_val.data.shape(), right_mask) {
                        Ok(arr) => arr,
                        Err(_) => continue, // Skip if shape mismatch
                    };

                let left_grad = Tensor {
                    data: (&grad_out.data * &left_mask_arr).into_dyn(),
                };

                let right_grad = Tensor {
                    data: (&grad_out.data * &right_mask_arr).into_dyn(),
                };

                if grads[left_id].is_none() {
                    grads[left_id] = Some(left_grad);
                } else {
                    let existing = grads[left_id].take().unwrap();
                    grads[left_id] = Some(Tensor {
                        data: &existing.data + &left_grad.data,
                    });
                }

                if grads[right_id].is_none() {
                    grads[right_id] = Some(right_grad);
                } else {
                    let existing = grads[right_id].take().unwrap();
                    grads[right_id] = Some(Tensor {
                        data: &existing.data + &right_grad.data,
                    });
                }
            }
            Op::Dropout { p } => {
                assert_eq!(node.inputs.len(), 1);
                let input_id = node.inputs[0];
                let x_activations = &activations[input_id];

                // Reconstruct mask from activations
                // If activation == 0, it was dropped (mask = 0)
                // If activation != 0, it was kept (mask = 1), and original = activation / scale
                let scale = 1.0 / (1.0 - p);
                let mut grad_in = grad_out.data.clone();

                // Apply same mask: zero out dropped elements
                // Note: We reconstruct the mask deterministically using the same logic as forward
                // This ensures backward matches forward
                let mut hash_seed = 0u64;
                for (i, &val) in x_activations.data.iter().enumerate() {
                    hash_seed = hash_seed
                        .wrapping_mul(31)
                        .wrapping_add(i as u64)
                        .wrapping_add(val.to_bits() as u64);
                }

                let mut rng_state = hash_seed;
                for (i, grad_val) in grad_in.iter_mut().enumerate() {
                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    let rand_val = ((rng_state >> 16) & 0x7FFF) as f32 / 32768.0;
                    if rand_val < *p {
                        // Was dropped, no gradient
                        *grad_val = 0.0;
                    } else {
                        // Was kept, gradient flows through with same scale
                        *grad_val *= scale;
                    }
                }

                if grads[input_id].is_none() {
                    grads[input_id] = Some(Tensor { data: grad_in });
                } else {
                    let existing = grads[input_id].take().unwrap();
                    grads[input_id] = Some(Tensor {
                        data: &existing.data + &grad_in,
                    });
                }
            }
        }
    }

    grads
}

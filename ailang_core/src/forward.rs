use crate::capability::Capabilities;
use crate::error::{CapabilityError, FileError, RuntimeError};
use crate::execute_kernel;
use crate::input_spec::DimBindings;
use crate::ir::{Graph, Op};
use crate::tensor::{Tensor, TokenIds};
use std::collections::HashMap;

pub fn execute_forward(
    graph: &Graph,
    inputs: &[Tensor],
    token_ids: &[TokenIds],
) -> Result<Vec<Tensor>, RuntimeError> {
    execute_forward_with_capabilities(graph, inputs, token_ids, &[], &Capabilities::empty())
}

pub fn execute_forward_with_capabilities(
    graph: &Graph,
    inputs: &[Tensor],
    token_ids: &[TokenIds],
    paths: &[String],
    capabilities: &Capabilities,
) -> Result<Vec<Tensor>, RuntimeError> {
    // Validate inputs against input signatures if present
    let mut dim_bindings = DimBindings::new();
    if !graph.input_specs.is_empty() {
        // Validate token_ids inputs (they come first in the signature)
        for (i, spec) in graph.input_specs.iter().enumerate() {
            if i < token_ids.len() {
                spec.validate(token_ids[i].shape(), &mut dim_bindings)?;
                // Also bind T from the validated input shape if it's the sequence dimension
                if token_ids[i].shape().len() >= 2 {
                    dim_bindings.insert("T".to_string(), token_ids[i].shape()[1]);
                }
            }
        }
    }

    let mut activations = vec![Tensor::zeros(&[]); graph.nodes.len()];

    for (node_id, node) in graph.nodes.iter().enumerate() {
        // Check capability for effectful ops
        if let Some(required_cap) = node.op.required_capability() {
            if !capabilities.has(&required_cap) {
                let (cap_name, op_name, action) = match &node.op {
                    Op::Now => ("Clock", "Now", "read system clock"),
                    Op::ReadFileText(_) => {
                        ("FileRead", "ReadFileText", "read file from filesystem")
                    }
                    _ => unreachable!(),
                };
                return Err(CapabilityError::Denied {
                    capability: cap_name.to_string(),
                    op: op_name.to_string(),
                    attempted_action: action.to_string(),
                }
                .into());
            }
        }

        // Collect input values for this node
        let mut input_vals = Vec::new();
        for &input_id in &node.inputs {
            input_vals.push(activations[input_id].clone());
        }

        // Convert dim_bindings to HashMap format if needed
        let dim_bindings_map: Option<HashMap<String, usize>> = if dim_bindings.is_empty() {
            None
        } else {
            Some(dim_bindings.iter().map(|(k, v)| (k.clone(), *v)).collect())
        };

        let output = execute_kernel::execute_op_forward(
            &node.op,
            &input_vals,
            inputs,
            token_ids,
            paths,
            capabilities,
            Some(&dim_bindings),
            #[cfg(feature = "gpu-wgpu")]
            None::<&crate::device::GpuDevice>, // CPU for now
            #[cfg(not(feature = "gpu-wgpu"))]
            None, // CPU only
        )?;

        activations[node_id] = output;
    }

    Ok(activations)
}

// Re-export hash_file_contents for backward compatibility
pub(crate) fn hash_file_contents(contents: &str) -> [u8; 32] {
    execute_kernel::hash_file_contents(contents)
}

// Selective evaluation implementation
// This file contains the selective evaluation logic that only evaluates nodes needed for requested outputs.

use crate::capability::Capabilities;
use crate::error::RuntimeError;
use crate::ir::{Graph, NodeId, Op};
use crate::tensor::{Tensor, TokenIds};
use std::collections::{HashSet, VecDeque};

/// Execute forward pass for selected nodes only (selective evaluation).
/// Only computes nodes needed to produce the `requested` outputs.
/// If a denied effect op is not needed for the requested outputs, it is skipped and does not error.
///
/// This is the public, stable API. For execution tracing (tests only), use
/// `execute_forward_selected_with_capabilities_traced`.
pub fn execute_forward_selected_with_capabilities(
    graph: &Graph,
    inputs: &[Tensor],
    token_ids: &[TokenIds],
    paths: &[String],
    capabilities: &Capabilities,
    requested: &[NodeId],
) -> Result<Vec<Tensor>, RuntimeError> {
    execute_forward_selected_with_capabilities_traced(
        graph,
        inputs,
        token_ids,
        paths,
        capabilities,
        requested,
        None,
    )
}

/// Internal version with execution tracing support.
/// Used by tests to verify zero execution on capability denial.
///
/// # Arguments
/// * `executed` - Optional trace vector. If provided, NodeIds are pushed immediately BEFORE executing their op.
///                If capability precheck fails, this remains empty, proving zero execution occurred.
pub fn execute_forward_selected_with_capabilities_traced(
    graph: &Graph,
    inputs: &[Tensor],
    token_ids: &[TokenIds],
    paths: &[String],
    capabilities: &Capabilities,
    requested: &[NodeId],
    mut executed: Option<&mut Vec<NodeId>>,
) -> Result<Vec<Tensor>, RuntimeError> {
    // Build dependency closure: all nodes needed to compute requested outputs
    let mut needed = HashSet::new();
    let mut queue = VecDeque::new();

    // Start with requested nodes
    for &node_id in requested {
        if node_id >= graph.nodes.len() {
            return Err(RuntimeError::Other(format!(
                "Requested node {} out of bounds (graph has {} nodes)",
                node_id,
                graph.nodes.len()
            )));
        }
        needed.insert(node_id);
        queue.push_back(node_id);
    }

    // Reverse walk: add all dependencies
    while let Some(node_id) = queue.pop_front() {
        let node = &graph.nodes[node_id];
        for &input_id in &node.inputs {
            if needed.insert(input_id) {
                queue.push_back(input_id);
            }
        }
    }

    // Check if any needed node requires a denied capability
    // We do this BEFORE evaluation to fail fast
    // This is the key: we only check capabilities for nodes that are actually needed
    // IMPORTANT: If this check fails, we return immediately without executing ANY ops.
    // The executed trace (if provided) remains empty, proving zero execution occurred.
    for &node_id in &needed {
        let node = &graph.nodes[node_id];
        if let Some(required_cap) = node.op.required_capability() {
            if !capabilities.has(&required_cap) {
                let (cap_name, op_name, action) = match &node.op {
                    Op::Now => ("Clock", "Now", "read system clock"),
                    Op::ReadFileText(_) => {
                        ("FileRead", "ReadFileText", "read file from filesystem")
                    }
                    _ => unreachable!(),
                };
                return Err(crate::error::CapabilityError::Denied {
                    capability: cap_name.to_string(),
                    op: op_name.to_string(),
                    attempted_action: action.to_string(),
                }
                .into());
            }
        }
    }

    // Topologically sort needed nodes for evaluation
    // Build a map of node -> list of nodes that depend on it
    let mut dependents: Vec<Vec<NodeId>> = vec![Vec::new(); graph.nodes.len()];
    let mut in_degree = vec![0; graph.nodes.len()];

    for &node_id in &needed {
        let node = &graph.nodes[node_id];
        for &input_id in &node.inputs {
            if needed.contains(&input_id) {
                dependents[input_id].push(node_id);
                in_degree[node_id] += 1;
            }
        }
    }

    // Find nodes with no dependencies (within needed set)
    let mut ready = VecDeque::new();
    for &node_id in &needed {
        if in_degree[node_id] == 0 {
            ready.push_back(node_id);
        }
    }

    // Topological sort
    let mut sorted = Vec::new();
    while let Some(node_id) = ready.pop_front() {
        sorted.push(node_id);
        for &dependent in &dependents[node_id] {
            in_degree[dependent] -= 1;
            if in_degree[dependent] == 0 {
                ready.push_back(dependent);
            }
        }
    }

    // Initialize activations (allocate all for simplicity, but only compute needed ones)
    let mut activations = vec![Tensor::zeros(&[]); graph.nodes.len()];

    // Evaluate only needed nodes in topological order
    // Record execution trace BEFORE executing each op
    for &node_id in &sorted {
        // Record this node as executed BEFORE calling the kernel
        // This proves that if capability check failed above, no ops executed
        if let Some(ref mut executed_trace) = executed {
            executed_trace.push(node_id);
        }

        let node = &graph.nodes[node_id];

        // Collect input values for this node
        let mut input_vals = Vec::new();
        for &input_id in &node.inputs {
            input_vals.push(activations[input_id].clone());
        }

        // Use unified kernel for execution
        use crate::execute_kernel;
        use std::collections::HashMap;
        activations[node_id] = execute_kernel::execute_op_forward(
            &node.op,
            &input_vals,
            inputs,
            token_ids,
            paths,
            capabilities,
            None, // dim_bindings not needed for selective eval
            #[cfg(feature = "gpu-wgpu")]
            None::<&crate::device::GpuDevice>, // CPU for now
            #[cfg(not(feature = "gpu-wgpu"))]
            None, // CPU only
        )?;
    }

    // Return only requested outputs
    let mut result = Vec::new();
    for &node_id in requested {
        result.push(activations[node_id].clone());
    }
    Ok(result)
}

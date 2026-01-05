// Model artifact export/import
use crate::{Graph, InputSpec, Param, ReshapeSpec, Tensor};
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ModelArtifact {
    pub version: String,
    pub dtype: String,
    pub seed: u64,
    pub graph: GraphData,
    pub weights: Vec<WeightData>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GraphData {
    pub nodes: Vec<NodeData>,
    pub input_count: usize,
    pub token_ids_count: usize,
    #[serde(default)]
    pub input_specs: Vec<InputSpec>, // Legacy: kept for backwards compatibility
    #[serde(default)]
    pub infer_input_specs: Vec<InputSpec>, // Inputs needed for inference (forward_output)
    #[serde(default)]
    pub train_input_specs: Vec<InputSpec>, // Inputs needed for training (loss_output)
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct NodeData {
    pub op: String,
    pub inputs: Vec<usize>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct WeightData {
    pub node_id: usize,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub size: usize,
}

/// Compute input dependencies for a given output node
fn compute_input_dependencies(
    graph: &Graph,
    output_node: usize,
    all_input_specs: &[InputSpec],
) -> Vec<InputSpec> {
    use std::collections::HashSet;

    // Build dependency closure by traversing backwards from output
    let mut needed = HashSet::new();
    let mut to_visit = vec![output_node];

    while let Some(node_id) = to_visit.pop() {
        if needed.contains(&node_id) {
            continue;
        }
        needed.insert(node_id);

        if node_id >= graph.nodes.len() {
            continue;
        }

        let node = &graph.nodes[node_id];

        // Add input nodes to visit
        for &input_id in &node.inputs {
            to_visit.push(input_id);
        }
    }

    // Collect input specs for nodes that are Input ops
    let mut result = Vec::new();
    for node_id in &needed {
        if *node_id < graph.nodes.len() {
            if let crate::Op::Input(input_idx) = &graph.nodes[*node_id].op {
                if *input_idx < all_input_specs.len() {
                    let spec = &all_input_specs[*input_idx];
                    // Only include if not already in result
                    if !result.iter().any(|s: &InputSpec| s.name == spec.name) {
                        result.push(spec.clone());
                    }
                }
            }
        }
    }

    // Also include side-channel inputs (token_ids, labels) that are referenced
    // Check which side-channel indices are used
    let mut used_token_indices = HashSet::new();
    for node_id in &needed {
        if *node_id < graph.nodes.len() {
            match &graph.nodes[*node_id].op {
                crate::Op::Embedding(idx) => {
                    used_token_indices.insert(*idx);
                }
                crate::Op::CrossEntropy(idx) => {
                    used_token_indices.insert(*idx);
                }
                _ => {}
            }
        }
    }

    // Map side-channel indices to input specs
    let mut side_channel_idx = 0;
    for spec in all_input_specs {
        if spec.dtype == "token_ids" {
            if used_token_indices.contains(&side_channel_idx) {
                if !result.iter().any(|s: &InputSpec| s.name == spec.name) {
                    result.push(spec.clone());
                }
            }
            side_channel_idx += 1;
        } else if spec.name == "labels" && spec.dtype == "labels" {
            // Labels are in the side-channel after token_ids
            // The index for labels in the side-channel is token_ids_count
            if used_token_indices.contains(&side_channel_idx) {
                if !result.iter().any(|s: &InputSpec| s.name == spec.name) {
                    result.push(spec.clone());
                }
            }
            side_channel_idx += 1;
        }
    }

    result
}

pub fn export_model(
    graph: &Graph,
    params: &[(usize, &Param)], // (node_id, param)
    seed: u64,
    path: &Path,
    forward_output: Option<usize>,
    loss_output: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create directory if needed
    fs::create_dir_all(path)?;

    // Serialize graph structure
    let mut nodes_data = Vec::new();
    for node in &graph.nodes {
        let op_str = match &node.op {
            crate::Op::Add => format!("Add"),
            crate::Op::Sub => format!("Sub"),
            crate::Op::Mul => format!("Mul"),
            crate::Op::MatMul2D => format!("MatMul2D"),
            crate::Op::BatchMatMul => format!("BatchMatMul"),
            crate::Op::ReLU => format!("ReLU"),
            crate::Op::Sum => format!("Sum"),
            crate::Op::Mean => format!("Mean"),
            crate::Op::Input(idx) => format!("Input({})", idx),
            crate::Op::Embedding(idx) => format!("Embedding({})", idx),
            crate::Op::Softmax(axis) => format!("Softmax({})", axis),
            crate::Op::CrossEntropy(idx) => format!("CrossEntropy({})", idx),
            crate::Op::LayerNorm(gamma_idx, beta_idx, eps) => {
                format!("LayerNorm({},{},{})", gamma_idx, beta_idx, eps)
            }
            crate::Op::MeanPoolTime => format!("MeanPoolTime"),
            crate::Op::Transpose3D => format!("Transpose3D"),
            crate::Op::Now => format!("Now"),
            crate::Op::ReadFileText(path_idx) => {
                // Serialize as JSON to preserve path index
                serde_json::json!({
                    "op": "ReadFileText",
                    "path_idx": path_idx
                })
                .to_string()
            }
            crate::Op::Reshape(spec) => {
                // Serialize ReshapeSpec as JSON
                serde_json::to_string(spec).unwrap_or_else(|_| "Reshape(invalid)".to_string())
            }
            crate::Op::Concat { axis } => format!("Concat({})", axis),
            crate::Op::SliceRows { start, len } => format!("SliceRows({},{})", start, len),
            crate::Op::GatherRows => format!("GatherRows"),
            crate::Op::Dropout { p } => format!("Dropout({})", p),
            crate::Op::If { .. } => format!("If"),
            crate::Op::Compare { op } => {
                let op_str = match op {
                    crate::ir::CompareOp::Eq => "==",
                    crate::ir::CompareOp::Ne => "!=",
                    crate::ir::CompareOp::Lt => "<",
                    crate::ir::CompareOp::Le => "<=",
                    crate::ir::CompareOp::Gt => ">",
                    crate::ir::CompareOp::Ge => ">=",
                };
                format!("Compare({})", op_str)
            }
            crate::Op::Logical { op } => {
                let op_str = match op {
                    crate::ir::LogicalOp::And => "and",
                    crate::ir::LogicalOp::Or => "or",
                    crate::ir::LogicalOp::Not => "not",
                };
                format!("Logical({})", op_str)
            }
            crate::Op::ConstScalar { value } => format!("ConstScalar({})", value),
            crate::Op::Stack { axis } => format!("Stack({})", axis),
            crate::Op::Max2 => format!("Max2"),
            crate::Op::Min2 => format!("Min2"),
        };
        nodes_data.push(NodeData {
            op: op_str,
            inputs: node.inputs.clone(),
        });
    }

    // Compute input dependencies
    let infer_input_specs = if let Some(forward_out) = forward_output {
        compute_input_dependencies(graph, forward_out, &graph.input_specs)
    } else {
        // Fallback: use all input_specs if forward_output not provided, but filter labels
        graph
            .input_specs
            .iter()
            .filter(|spec| spec.name != "labels" || spec.dtype != "labels")
            .cloned()
            .collect()
    };

    // Filter out labels from infer_input_specs
    let infer_input_specs: Vec<InputSpec> = infer_input_specs
        .into_iter()
        .filter(|spec| spec.name != "labels" || spec.dtype != "labels")
        .collect();

    let train_input_specs = if let Some(loss_out) = loss_output {
        compute_input_dependencies(graph, loss_out, &graph.input_specs)
    } else {
        // Fallback: use all input_specs if loss_output not provided
        graph.input_specs.clone()
    };

    let graph_data = GraphData {
        nodes: nodes_data,
        input_count: graph.input_count,
        token_ids_count: graph.token_ids_count,
        input_specs: graph.input_specs.clone(), // Legacy: kept for backwards compatibility
        infer_input_specs,
        train_input_specs,
    };

    // Collect weights and compute offsets
    let mut weights_data = Vec::new();
    let mut offset = 0;
    for &(node_id, param) in params {
        let shape = param.value.shape();
        let size: usize = shape.iter().product();
        weights_data.push(WeightData {
            node_id,
            shape: shape.to_vec(),
            offset,
            size,
        });
        offset += size;
    }

    let artifact = ModelArtifact {
        version: "1.0".to_string(),
        dtype: "f32".to_string(),
        seed,
        graph: graph_data,
        weights: weights_data,
    };

    // Write model.json
    let model_json = serde_json::to_string_pretty(&artifact)?;
    let model_path = path.join("model.json");
    fs::write(&model_path, model_json)?;

    // Write weights.bin
    let mut weights_bin = Vec::new();
    for &(_, param) in params {
        let data: Vec<f32> = param.value.data.iter().copied().collect();
        for val in data {
            weights_bin.write_all(&val.to_le_bytes())?;
        }
    }
    let weights_path = path.join("weights.bin");
    fs::write(&weights_path, weights_bin)?;

    // Write meta.json
    let meta = serde_json::json!({
        "version": "1.0",
        "dtype": "f32",
        "seed": seed,
        "checksum": "simple" // Could add real checksum later
    });
    let meta_path = path.join("meta.json");
    fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

    Ok(())
}

/// Load model and return graph, weights, seed, and input specs
pub fn load_model(
    path: &Path,
) -> Result<
    (
        Graph,
        Vec<(usize, Tensor)>,
        u64,
        Vec<InputSpec>,
        Vec<InputSpec>,
    ),
    Box<dyn std::error::Error>,
> {
    // Read model.json
    let model_path = path.join("model.json");
    let model_json = fs::read_to_string(&model_path)?;
    let artifact: ModelArtifact = serde_json::from_str(&model_json)?;

    // Reconstruct graph
    let mut nodes = Vec::new();
    for node_data in &artifact.graph.nodes {
        let op = if node_data.op == "Add" {
            crate::Op::Add
        } else if node_data.op == "Sub" {
            crate::Op::Sub
        } else if node_data.op == "Mul" {
            crate::Op::Mul
        } else if node_data.op == "MatMul2D" {
            crate::Op::MatMul2D
        } else if node_data.op == "BatchMatMul" {
            crate::Op::BatchMatMul
        } else if node_data.op == "ReLU" {
            crate::Op::ReLU
        } else if node_data.op == "Sum" {
            crate::Op::Sum
        } else if node_data.op == "Mean" {
            crate::Op::Mean
        } else if node_data.op == "MeanPoolTime" {
            crate::Op::MeanPoolTime
        } else if node_data.op == "Transpose3D" {
            crate::Op::Transpose3D
        } else if node_data.op == "Now" {
            crate::Op::Now
        } else if node_data.op.starts_with("{") && node_data.op.contains("ReadFileText") {
            // Try to deserialize ReadFileText JSON
            match serde_json::from_str::<serde_json::Value>(&node_data.op) {
                Ok(json) => {
                    if let Some(path_idx) = json.get("path_idx").and_then(|p| p.as_u64()) {
                        crate::Op::ReadFileText(path_idx as usize)
                    } else {
                        return Err("ReadFileText: missing path_idx field".into());
                    }
                }
                Err(_) => {
                    return Err(format!("Failed to parse ReadFileText op: {}", node_data.op).into());
                }
            }
        } else if node_data.op.starts_with("Input(") {
            let idx_str = &node_data.op[6..node_data.op.len() - 1];
            let idx = idx_str.parse::<usize>()?;
            crate::Op::Input(idx)
        } else if node_data.op.starts_with("Embedding(") {
            let idx_str = &node_data.op[10..node_data.op.len() - 1];
            let idx = idx_str.parse::<usize>()?;
            crate::Op::Embedding(idx)
        } else if node_data.op.starts_with("Softmax(") {
            let axis_str = &node_data.op[8..node_data.op.len() - 1];
            let axis = axis_str.parse::<usize>()?;
            crate::Op::Softmax(axis)
        } else if node_data.op.starts_with("CrossEntropy(") {
            let idx_str = &node_data.op[13..node_data.op.len() - 1];
            let idx = idx_str.parse::<usize>()?;
            crate::Op::CrossEntropy(idx)
        } else if node_data.op.starts_with("Concat(") {
            let axis_str = &node_data.op[7..node_data.op.len() - 1];
            let axis = axis_str.parse::<usize>()?;
            crate::Op::Concat { axis }
        } else if node_data.op.starts_with("SliceRows(") {
            let params_str = &node_data.op[10..node_data.op.len() - 1];
            let parts: Vec<&str> = params_str.split(',').collect();
            if parts.len() != 2 {
                return Err(
                    format!("SliceRows: expected 2 parameters, got {}", parts.len()).into(),
                );
            }
            let start = parts[0].parse::<usize>()?;
            let len = parts[1].parse::<usize>()?;
            crate::Op::SliceRows { start, len }
        } else if node_data.op == "GatherRows" {
            crate::Op::GatherRows
        } else if node_data.op.starts_with("Dropout(") {
            let p_str = &node_data.op[8..node_data.op.len() - 1];
            let p = p_str.parse::<f32>()?;
            crate::Op::Dropout { p }
        } else if node_data.op.starts_with("{") {
            // Try to deserialize as ReshapeSpec JSON
            match serde_json::from_str::<ReshapeSpec>(&node_data.op) {
                Ok(spec) => crate::Op::Reshape(spec),
                Err(_) => {
                    // Fallback: try old format for backward compatibility
                    if node_data.op.starts_with("Reshape([") {
                        let shape_str = &node_data.op[9..node_data.op.len() - 2];
                        let shape: Vec<usize> = if shape_str.is_empty() {
                            Vec::new()
                        } else {
                            shape_str
                                .split(',')
                                .map(|s: &str| s.parse::<usize>().unwrap())
                                .collect()
                        };
                        crate::Op::Reshape(ReshapeSpec::from_literals(shape))
                    } else {
                        return Err(format!("Unknown op: {}", node_data.op).into());
                    }
                }
            }
        } else {
            return Err(format!("Unknown op: {}", node_data.op).into());
        };
        nodes.push(crate::ir::Node {
            op,
            inputs: node_data.inputs.clone(),
        });
    }

    // Handle backwards compatibility: if infer_input_specs/train_input_specs are empty,
    // use the legacy input_specs for both
    let infer_input_specs = if artifact.graph.infer_input_specs.is_empty() {
        // Filter out labels from legacy input_specs for inference
        artifact
            .graph
            .input_specs
            .iter()
            .filter(|spec| spec.name != "labels" || spec.dtype != "labels")
            .cloned()
            .collect()
    } else {
        artifact.graph.infer_input_specs
    };

    let train_input_specs = if artifact.graph.train_input_specs.is_empty() {
        artifact.graph.input_specs
    } else {
        artifact.graph.train_input_specs
    };

    let graph = Graph {
        nodes,
        input_count: artifact.graph.input_count,
        token_ids_count: artifact.graph.token_ids_count,
        input_specs: train_input_specs.clone(), // Use train_input_specs as the default
        input_node_ids: Vec::new(),             // Will be populated if needed
    };

    // Read weights.bin
    let weights_path = path.join("weights.bin");
    let weights_bytes = fs::read(&weights_path)?;
    let mut weights = Vec::new();
    for weight_data in &artifact.weights {
        let mut data = Vec::new();
        for i in 0..weight_data.size {
            let offset = (weight_data.offset + i) * 4;
            let bytes = [
                weights_bytes[offset],
                weights_bytes[offset + 1],
                weights_bytes[offset + 2],
                weights_bytes[offset + 3],
            ];
            let val = f32::from_le_bytes(bytes);
            data.push(val);
        }
        let tensor = Tensor::from_vec(&weight_data.shape, data);
        weights.push((weight_data.node_id, tensor));
    }

    Ok((
        graph,
        weights,
        artifact.seed,
        infer_input_specs,
        train_input_specs,
    ))
}

/// Backwards-compatible load_model that returns only graph, weights, and seed
pub fn load_model_legacy(
    path: &Path,
) -> Result<(Graph, Vec<(usize, Tensor)>, u64), Box<dyn std::error::Error>> {
    let (graph, weights, seed, _, _) = load_model(path)?;
    Ok((graph, weights, seed))
}

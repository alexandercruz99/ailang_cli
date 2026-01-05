use ailang_core::{
    execute_forward_selected_with_capabilities,
    ir::{Graph, Op},
    tensor::Tensor,
    Capabilities, Capability,
};
use std::fs;

#[test]
fn test_selective_eval_skips_denied_effect_op() {
    // Build a mixed graph: safe branch (Linear + ReLU) and effect branch (ReadFileText)
    let mut graph = Graph::new(2); // 2 tensor inputs: weight for linear, and a dummy

    // Safe branch: input -> MatMul2D -> ReLU
    let weight_id = graph.input_node(0);
    let input_id = graph.input_node(1);
    let matmul_id = graph.add_node(Op::MatMul2D, vec![input_id, weight_id]);
    let safe_output_id = graph.add_node(Op::ReLU, vec![matmul_id]);

    // Effect branch: ReadFileText (not connected to safe branch)
    let temp_file =
        std::env::temp_dir().join(format!("ailang_test_selective_{}.txt", std::process::id()));
    fs::write(&temp_file, b"test content").unwrap();
    let paths = vec![temp_file.to_str().unwrap().to_string()];
    let _effect_output_id = graph.add_node(Op::ReadFileText(0), vec![]);

    // Execute requesting ONLY safe output with empty capabilities
    let capabilities = Capabilities::empty();
    let inputs = vec![
        Tensor::from_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), // weight
        Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), // input
    ];
    let token_ids = vec![];

    // This should succeed because ReadFileText is not needed for safe_output_id
    let result = execute_forward_selected_with_capabilities(
        &graph,
        &inputs,
        &token_ids,
        &paths,
        &capabilities,
        &[safe_output_id],
    );

    assert!(
        result.is_ok(),
        "Selective eval should succeed when effect op is not needed"
    );
    let outputs = result.unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape().len(), 2); // ReLU output should be 2D

    // Cleanup
    fs::remove_file(&temp_file).ok();
}

#[test]
fn test_selective_eval_denies_when_effect_is_required() {
    // Build same mixed graph
    let mut graph = Graph::new(2);
    let weight_id = graph.input_node(0);
    let input_id = graph.input_node(1);
    let matmul_id = graph.add_node(Op::MatMul2D, vec![input_id, weight_id]);
    let _safe_output_id = graph.add_node(Op::ReLU, vec![matmul_id]);

    let temp_file =
        std::env::temp_dir().join(format!("ailang_test_selective2_{}.txt", std::process::id()));
    fs::write(&temp_file, b"test content").unwrap();
    let paths = vec![temp_file.to_str().unwrap().to_string()];
    let effect_output_id = graph.add_node(Op::ReadFileText(0), vec![]);

    // Execute requesting effect output with empty capabilities
    let capabilities = Capabilities::empty();
    let inputs = vec![
        Tensor::from_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ];
    let token_ids = vec![];

    let result = execute_forward_selected_with_capabilities(
        &graph,
        &inputs,
        &token_ids,
        &paths,
        &capabilities,
        &[effect_output_id],
    );

    assert!(
        result.is_err(),
        "Selective eval should fail when effect op is needed but denied"
    );

    let err = result.unwrap_err();
    let diag = err.diagnostic();

    // Check error code
    assert_eq!(diag.code, "E_CAPABILITY_DENIED");
    assert_eq!(diag.title, "Capability denied");

    // Check fields
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("capability"), Some(&"FileRead".to_string()));
    assert_eq!(fields_map.get("op"), Some(&"ReadFileText".to_string()));
    assert_eq!(
        fields_map.get("attempted_action"),
        Some(&"read file from filesystem".to_string())
    );

    // Cleanup
    fs::remove_file(&temp_file).ok();
}

#[test]
fn test_selective_eval_executes_effect_when_allowed() {
    // Build graph with ReadFileText
    let mut graph = Graph::new(0);
    let temp_file =
        std::env::temp_dir().join(format!("ailang_test_selective3_{}.txt", std::process::id()));
    fs::write(&temp_file, b"hello world").unwrap();
    let paths = vec![temp_file.to_str().unwrap().to_string()];
    let effect_output_id = graph.add_node(Op::ReadFileText(0), vec![]);

    // Execute with FileRead capability
    let capabilities = Capabilities::new().with(Capability::FileRead);
    let inputs = vec![];
    let token_ids = vec![];

    let result = execute_forward_selected_with_capabilities(
        &graph,
        &inputs,
        &token_ids,
        &paths,
        &capabilities,
        &[effect_output_id],
    );

    assert!(
        result.is_ok(),
        "Selective eval should succeed when effect op is allowed"
    );
    let outputs = result.unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), &[8]); // Hash vector should be [8]

    // Cleanup
    fs::remove_file(&temp_file).ok();
}

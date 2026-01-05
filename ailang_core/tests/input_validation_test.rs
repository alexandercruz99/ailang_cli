use ailang_core::{
    forward::execute_forward,
    infer_dtype_from_name,
    ir::{Graph, Op},
    model::{export_model, load_model},
    param::Param,
    tensor::{Tensor, TokenIds},
    DimSpec, InputSpec, ReshapeDim, ReshapeError, ReshapeSpec, RuntimeError, ValidationError,
};
use std::fs;

#[test]
fn test_input_validation_valid() {
    // Build a simple graph with input signature
    const T: usize = 3;
    const D: usize = 4;
    const VOCAB: usize = 5;
    const NUM_CLASSES: usize = 2;

    let embed_weight = Param::new(Tensor::from_vec(
        &[VOCAB, D],
        (0..VOCAB * D).map(|i| i as f32 * 0.1).collect(),
    ));
    let out_weight = Param::new(Tensor::from_vec(
        &[D, NUM_CLASSES],
        (0..D * NUM_CLASSES).map(|i| i as f32 * 0.1).collect(),
    ));

    let mut graph = Graph::new_with_token_ids(2, 1);

    // Declare input signature: tokens: [B, T] where B is variable, T is fixed
    let input_specs = vec![InputSpec::new(
        "tokens".to_string(),
        infer_dtype_from_name("tokens").to_string(),
        vec![DimSpec::Named("B".to_string()), DimSpec::Literal(T)],
    )];
    graph = graph.with_input_specs(input_specs);

    let embed_w_id = graph.input_node(0);
    let out_w_id = graph.input_node(1);

    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);
    let pool_id = graph.add_node(Op::MeanPoolTime, vec![embed_id]);
    let logits_id = graph.add_node(Op::MatMul2D, vec![pool_id, out_w_id]);

    // Test with valid input: B=2, T=3
    let token_ids_input = TokenIds::new(&[2, T], vec![0, 1, 2, 1, 2, 0]);
    let inputs = vec![embed_weight.value.clone(), out_weight.value.clone()];
    let token_ids = vec![token_ids_input];

    // Should succeed
    let activations = execute_forward(&graph, &inputs, &token_ids).unwrap();
    assert_eq!(activations[logits_id].shape(), &[2, NUM_CLASSES]);
}

#[test]
fn test_input_validation_invalid_rank() {
    // Build a simple graph with input signature
    const T: usize = 3;
    const D: usize = 4;
    const VOCAB: usize = 5;
    const NUM_CLASSES: usize = 2;

    let embed_weight = Param::new(Tensor::from_vec(
        &[VOCAB, D],
        (0..VOCAB * D).map(|i| i as f32 * 0.1).collect(),
    ));
    let out_weight = Param::new(Tensor::from_vec(
        &[D, NUM_CLASSES],
        (0..D * NUM_CLASSES).map(|i| i as f32 * 0.1).collect(),
    ));

    let mut graph = Graph::new_with_token_ids(2, 1);

    let input_specs = vec![InputSpec::new(
        "tokens".to_string(),
        infer_dtype_from_name("tokens").to_string(),
        vec![DimSpec::Named("B".to_string()), DimSpec::Literal(T)],
    )];
    graph = graph.with_input_specs(input_specs);

    let embed_w_id = graph.input_node(0);
    let out_w_id = graph.input_node(1);

    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);
    let pool_id = graph.add_node(Op::MeanPoolTime, vec![embed_id]);
    let _logits_id = graph.add_node(Op::MatMul2D, vec![pool_id, out_w_id]);

    // Test with invalid rank: 1D instead of 2D
    let token_ids_input = TokenIds::new(&[2], vec![0, 1]);
    let inputs = vec![embed_weight.value.clone(), out_weight.value.clone()];
    let token_ids = vec![token_ids_input];

    // Should return validation error
    let result = execute_forward(&graph, &inputs, &token_ids);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let diag = err.diagnostic();
    assert_eq!(diag.code, "E_INPUT_RANK_MISMATCH");
    assert_eq!(diag.title, "Input rank mismatch");
    // Check fields
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("input"), Some(&"tokens".to_string()));
    assert_eq!(fields_map.get("expected_rank"), Some(&"2".to_string()));
    assert_eq!(fields_map.get("received_rank"), Some(&"1".to_string()));
}

#[test]
fn test_input_validation_invalid_fixed_dim() {
    // Build a simple graph with input signature
    const T: usize = 3;
    const T_WRONG: usize = 5;
    const D: usize = 4;
    const VOCAB: usize = 5;
    const NUM_CLASSES: usize = 2;

    let embed_weight = Param::new(Tensor::from_vec(
        &[VOCAB, D],
        (0..VOCAB * D).map(|i| i as f32 * 0.1).collect(),
    ));
    let out_weight = Param::new(Tensor::from_vec(
        &[D, NUM_CLASSES],
        (0..D * NUM_CLASSES).map(|i| i as f32 * 0.1).collect(),
    ));

    let mut graph = Graph::new_with_token_ids(2, 1);

    let input_specs = vec![InputSpec::new(
        "tokens".to_string(),
        "token_ids".to_string(),
        vec![
            DimSpec::Named("B".to_string()),
            DimSpec::Literal(T), // Expect T=3
        ],
    )];
    graph = graph.with_input_specs(input_specs);

    let embed_w_id = graph.input_node(0);
    let out_w_id = graph.input_node(1);

    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);
    let pool_id = graph.add_node(Op::MeanPoolTime, vec![embed_id]);
    let _logits_id = graph.add_node(Op::MatMul2D, vec![pool_id, out_w_id]);

    // Test with wrong T: T=5 instead of T=3
    let token_ids_input = TokenIds::new(&[2, T_WRONG], vec![0; 2 * T_WRONG]);
    let inputs = vec![embed_weight.value.clone(), out_weight.value.clone()];
    let token_ids = vec![token_ids_input];

    // Should return validation error
    let result = execute_forward(&graph, &inputs, &token_ids);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let diag = err.diagnostic();
    assert_eq!(diag.code, "E_INPUT_DIM_MISMATCH");
    assert_eq!(diag.title, "Input dimension mismatch");
    // Check fields
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("input"), Some(&"tokens".to_string()));
    assert_eq!(fields_map.get("dimension"), Some(&"1".to_string()));
    assert_eq!(fields_map.get("expected"), Some(&T.to_string()));
    assert_eq!(fields_map.get("received"), Some(&T_WRONG.to_string()));
}

#[test]
fn test_named_dim_consistency() {
    // Test that named dimensions are consistent across inputs
    const T: usize = 3;
    const D: usize = 4;
    const VOCAB: usize = 5;

    let embed_weight = Param::new(Tensor::from_vec(
        &[VOCAB, D],
        (0..VOCAB * D).map(|i| i as f32 * 0.1).collect(),
    ));

    let mut graph = Graph::new_with_token_ids(1, 2);

    // Declare two input specs that both use B
    // Note: "tokens1" and "tokens2" are not "tokens", so they map to "tensor"
    // But for this test we need token_ids, so we use "tokens" as the name
    let input_specs = vec![
        InputSpec::new(
            "tokens".to_string(),
            infer_dtype_from_name("tokens").to_string(),
            vec![DimSpec::Named("B".to_string()), DimSpec::Literal(T)],
        ),
        InputSpec::new(
            "tokens".to_string(),
            infer_dtype_from_name("tokens").to_string(),
            vec![
                DimSpec::Named("B".to_string()), // Same B
                DimSpec::Literal(T),
            ],
        ),
    ];
    graph = graph.with_input_specs(input_specs);

    let embed_w_id = graph.input_node(0);
    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);

    // Test with consistent B: both inputs have B=2
    let token_ids_input1 = TokenIds::new(&[2, T], vec![0, 1, 2, 1, 2, 0]);
    let token_ids_input2 = TokenIds::new(&[2, T], vec![0, 1, 2, 1, 2, 0]);
    let inputs = vec![embed_weight.value.clone()];
    let token_ids = vec![token_ids_input1, token_ids_input2];

    // Should succeed
    let _activations = execute_forward(&graph, &inputs, &token_ids).unwrap();
}

#[test]
fn test_named_dim_inconsistency() {
    // Test that inconsistent named dimensions cause an error
    const T: usize = 3;
    const D: usize = 4;
    const VOCAB: usize = 5;

    let embed_weight = Param::new(Tensor::from_vec(
        &[VOCAB, D],
        (0..VOCAB * D).map(|i| i as f32 * 0.1).collect(),
    ));

    let mut graph = Graph::new_with_token_ids(1, 2);

    // Use "tokens" as name to get token_ids dtype via canonical function
    let input_specs = vec![
        InputSpec::new(
            "tokens".to_string(),
            infer_dtype_from_name("tokens").to_string(),
            vec![DimSpec::Named("B".to_string()), DimSpec::Literal(T)],
        ),
        InputSpec::new(
            "tokens".to_string(),
            infer_dtype_from_name("tokens").to_string(),
            vec![
                DimSpec::Named("B".to_string()), // Same B name
                DimSpec::Literal(T),
            ],
        ),
    ];
    graph = graph.with_input_specs(input_specs);

    let embed_w_id = graph.input_node(0);
    let _embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);

    // Test with inconsistent B: B=2 vs B=3
    let token_ids_input1 = TokenIds::new(&[2, T], vec![0, 1, 2, 1, 2, 0]);
    let token_ids_input2 = TokenIds::new(&[3, T], vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
    let inputs = vec![embed_weight.value.clone()];
    let token_ids = vec![token_ids_input1, token_ids_input2];

    // Should return validation error about inconsistent B
    let result = execute_forward(&graph, &inputs, &token_ids);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let diag = err.diagnostic();
    assert_eq!(diag.code, "E_NAMED_DIM_CONFLICT");
    assert_eq!(diag.title, "Named dimension conflict");
    // Check fields
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("named_dim"), Some(&"B".to_string()));
    assert_eq!(fields_map.get("previous_value"), Some(&"2".to_string()));
    assert_eq!(fields_map.get("new_value"), Some(&"3".to_string()));
    assert_eq!(fields_map.get("input"), Some(&"tokens2".to_string()));
}

#[test]
fn test_export_load_with_input_specs() {
    // Test that input specs are preserved through export/load
    const T: usize = 3;
    const D: usize = 4;
    const VOCAB: usize = 5;
    const NUM_CLASSES: usize = 2;

    let embed_weight = Param::new(Tensor::from_vec(
        &[VOCAB, D],
        (0..VOCAB * D).map(|i| i as f32 * 0.1).collect(),
    ));
    let out_weight = Param::new(Tensor::from_vec(
        &[D, NUM_CLASSES],
        (0..D * NUM_CLASSES).map(|i| i as f32 * 0.1).collect(),
    ));

    let mut graph = Graph::new_with_token_ids(2, 1);

    let input_specs = vec![InputSpec::new(
        "tokens".to_string(),
        infer_dtype_from_name("tokens").to_string(),
        vec![DimSpec::Named("B".to_string()), DimSpec::Literal(T)],
    )];
    graph = graph.with_input_specs(input_specs);

    let embed_w_id = graph.input_node(0);
    let out_w_id = graph.input_node(1);

    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);
    let pool_id = graph.add_node(Op::MeanPoolTime, vec![embed_id]);
    let logits_id = graph.add_node(Op::MatMul2D, vec![pool_id, out_w_id]);

    // Export
    let temp_dir = std::env::temp_dir().join("ailang_input_spec_test");
    fs::create_dir_all(&temp_dir).unwrap();

    let params = vec![(embed_w_id, &embed_weight), (out_w_id, &out_weight)];
    export_model(&graph, &params, 42, &temp_dir, Some(logits_id), None).unwrap();

    // Load
    let (loaded_graph, loaded_weights, _loaded_seed, _, _) = load_model(&temp_dir).unwrap();

    // Verify input specs are preserved
    assert_eq!(loaded_graph.input_specs.len(), 1);
    assert_eq!(loaded_graph.input_specs[0].name, "tokens");
    assert_eq!(loaded_graph.input_specs[0].dims.len(), 2);

    // Test inference with different batch sizes
    for batch_size in [1, 2, 4] {
        let token_ids_input = TokenIds::new(&[batch_size, T], vec![0; batch_size * T]);
        let inputs = loaded_weights
            .iter()
            .map(|(_, t)| t.clone())
            .collect::<Vec<_>>();
        let token_ids = vec![token_ids_input];

        let activations = execute_forward(&loaded_graph, &inputs, &token_ids).unwrap();
        assert_eq!(activations[logits_id].shape(), &[batch_size, NUM_CLASSES]);
    }

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}

#[test]
fn test_validation_diagnostic() {
    // Test that ValidationError produces correct diagnostic
    let err = ValidationError::RankMismatch {
        input_name: "test_input".to_string(),
        expected: 3,
        got: 2,
    };
    let diag = err.diagnostic();
    assert_eq!(diag.code, "E_INPUT_RANK_MISMATCH");
    assert_eq!(diag.title, "Input rank mismatch");
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("input"), Some(&"test_input".to_string()));
    assert_eq!(fields_map.get("expected_rank"), Some(&"3".to_string()));
    assert_eq!(fields_map.get("received_rank"), Some(&"2".to_string()));
    assert!(diag.hint.is_some());
}

#[test]
fn test_reshape_diagnostic() {
    // Test that ReshapeError produces correct diagnostic
    let err = ReshapeError::NamedDimensionNotFound {
        name: "D".to_string(),
    };
    let diag = err.diagnostic();
    assert_eq!(diag.code, "E_RESHAPE_NAMED_DIM_NOT_FOUND");
    assert_eq!(diag.title, "Named dimension not found");
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("named_dim"), Some(&"D".to_string()));
    assert!(diag.hint.is_some());
}

#[test]
fn test_runtime_diagnostic() {
    // Test that RuntimeError produces correct diagnostic
    let validation_err = ValidationError::DimensionMismatch {
        input_name: "tokens".to_string(),
        dim_index: 1,
        expected: 6,
        got: 5,
    };
    let runtime_err = RuntimeError::Validation(validation_err);
    let diag = runtime_err.diagnostic();
    assert_eq!(diag.code, "E_INPUT_DIM_MISMATCH");
    assert_eq!(diag.title, "Input dimension mismatch");
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("input"), Some(&"tokens".to_string()));
    assert_eq!(fields_map.get("dimension"), Some(&"1".to_string()));
    assert_eq!(fields_map.get("expected"), Some(&"6".to_string()));
    assert_eq!(fields_map.get("received"), Some(&"5".to_string()));

    // Test RuntimeError::Other
    let other_err = RuntimeError::Other("test message".to_string());
    let other_diag = other_err.diagnostic();
    assert_eq!(other_diag.code, "E_EXECUTION_FAILURE");
    assert_eq!(other_diag.title, "Execution failure");
    let other_fields: std::collections::HashMap<_, _> = other_diag.fields.iter().cloned().collect();
    assert_eq!(
        other_fields.get("message"),
        Some(&"test message".to_string())
    );
}

#[test]
fn test_reshape_resolution_failure() {
    use ailang_core::ReshapeError;

    // Test reshape with missing named dimension
    const T: usize = 3;
    const D: usize = 4;
    const VOCAB: usize = 5;

    let embed_weight = Param::new(Tensor::from_vec(
        &[VOCAB, D],
        (0..VOCAB * D).map(|i| i as f32 * 0.1).collect(),
    ));

    let mut graph = Graph::new_with_token_ids(1, 1);

    // No input spec, so no dimension bindings
    let embed_w_id = graph.input_node(0);
    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);

    // Try to reshape using named dimension "B" which isn't bound
    let reshape_id = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Named("B".to_string()),
            ReshapeDim::Named("T".to_string()),
            ReshapeDim::Named("D".to_string()),
        ])),
        vec![embed_id],
    );

    let token_ids_input = TokenIds::new(&[2, T], vec![0, 1, 2, 1, 2, 0]);
    let inputs = vec![embed_weight.value.clone()];
    let token_ids = vec![token_ids_input];

    // Should return reshape error
    let result = execute_forward(&graph, &inputs, &token_ids);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let diag = err.diagnostic();
    assert_eq!(diag.code, "E_RESHAPE_NAMED_DIM_NOT_FOUND");
    assert_eq!(diag.title, "Named dimension not found");
    // Check fields
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("named_dim"), Some(&"B".to_string()));
}

#[test]
fn test_dtype_inference_from_name() {
    // Test centralized dtype inference function directly
    // Rule: "tokens" => "token_ids", "labels" => "labels", else => "tensor"

    // Test "tokens" => "token_ids"
    assert_eq!(infer_dtype_from_name("tokens"), "token_ids");

    // Test "labels" => "labels"
    assert_eq!(infer_dtype_from_name("labels"), "labels");

    // Test other names => "tensor"
    assert_eq!(infer_dtype_from_name("x"), "tensor");
    assert_eq!(infer_dtype_from_name("a"), "tensor");
    assert_eq!(infer_dtype_from_name("b"), "tensor");
    assert_eq!(infer_dtype_from_name("input"), "tensor");
    assert_eq!(infer_dtype_from_name("data"), "tensor");

    // Test that rank/shape doesn't matter - only name
    // x [B, T] should be tensor (not token_ids) unless name is "tokens"
    assert_eq!(infer_dtype_from_name("x"), "tensor");

    // tokens [B] should be token_ids (name-based, not rank-based)
    assert_eq!(infer_dtype_from_name("tokens"), "token_ids");
}

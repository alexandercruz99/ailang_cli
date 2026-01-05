// Tests for execution trace hooks in selective evaluation
// These tests prove that:
// 1) No ops execute when capabilities are denied (zero execution guarantee)
// 2) DSL programs work correctly with selective execution + kernel

use ailang_core::forward_selected::execute_forward_selected_with_capabilities_traced;
use ailang_core::{
    frontend::{lexer::Lexer, lower::lower, parser::Parser},
    ir::{Graph, Op},
    tensor::Tensor,
    Capabilities,
};

#[test]
fn test_no_execution_when_capability_denied() {
    // Build a graph where requested output depends on Op::Now
    let mut graph = Graph::new(1);

    // Input node
    let input_node = graph.input_node(0);

    // Now node (requires Clock capability)
    let now_node = graph.add_node(Op::Now, vec![]);

    // Output depends on Now
    let output_node = graph.add_node(Op::Add, vec![input_node, now_node]);

    // Create dummy input
    let inputs = vec![Tensor::from_vec(&[1], vec![1.0])];
    let token_ids = vec![];
    let paths = vec![];
    let capabilities = Capabilities::empty(); // No Clock capability

    // Trace vector to record executed nodes
    let mut executed = Vec::new();

    // Call selective execution requesting output_node (using traced version for test)
    let result = execute_forward_selected_with_capabilities_traced(
        &graph,
        &inputs,
        &token_ids,
        &paths,
        &capabilities,
        &[output_node],
        Some(&mut executed),
    );

    // Assert: result is Err with E_CAPABILITY_DENIED
    assert!(result.is_err(), "Should fail with capability denied");

    let err = result.unwrap_err();
    let diag = err.diagnostic();
    assert_eq!(
        diag.code, "E_CAPABILITY_DENIED",
        "Error code must be E_CAPABILITY_DENIED"
    );

    // Check diagnostic fields
    let fields: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields.get("capability"), Some(&"Clock".to_string()));
    assert_eq!(fields.get("op"), Some(&"Now".to_string()));
    assert_eq!(
        fields.get("attempted_action"),
        Some(&"read system clock".to_string())
    );

    // CRITICAL: executed trace must be empty, proving zero execution occurred
    assert_eq!(
        executed.len(),
        0,
        "No ops should execute when capability is denied"
    );
}

#[test]
fn test_no_execution_when_file_read_denied() {
    // Build a graph where requested output depends on Op::ReadFileText
    let mut graph = Graph::new(1);

    // Input node
    let input_node = graph.input_node(0);

    // ReadFileText node (requires FileRead capability)
    let read_node = graph.add_node(Op::ReadFileText(0), vec![]);

    // Output depends on ReadFileText
    let output_node = graph.add_node(Op::Add, vec![input_node, read_node]);

    // Create dummy input
    let inputs = vec![Tensor::from_vec(&[8], vec![1.0; 8])];
    let token_ids = vec![];
    let paths = vec!["/nonexistent/file.txt".to_string()];
    let capabilities = Capabilities::empty(); // No FileRead capability

    // Trace vector to record executed nodes
    let mut executed = Vec::new();

    // Call selective execution requesting output_node (using traced version for test)
    let result = execute_forward_selected_with_capabilities_traced(
        &graph,
        &inputs,
        &token_ids,
        &paths,
        &capabilities,
        &[output_node],
        Some(&mut executed),
    );

    // Assert: result is Err with E_CAPABILITY_DENIED
    assert!(result.is_err(), "Should fail with capability denied");

    let err = result.unwrap_err();
    let diag = err.diagnostic();
    assert_eq!(
        diag.code, "E_CAPABILITY_DENIED",
        "Error code must be E_CAPABILITY_DENIED"
    );

    // Check diagnostic fields
    let fields: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields.get("capability"), Some(&"FileRead".to_string()));
    assert_eq!(fields.get("op"), Some(&"ReadFileText".to_string()));
    assert_eq!(
        fields.get("attempted_action"),
        Some(&"read file from filesystem".to_string())
    );

    // CRITICAL: executed trace must be empty, proving zero execution occurred
    assert_eq!(
        executed.len(),
        0,
        "No ops should execute when capability is denied"
    );
}

#[test]
fn test_dsl_minimal_program_runs_under_selective_exec() {
    // Parse and lower a minimal DSL program
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]
param E: [V, D]
param W: [D, C]

forward {
  let x = embedding(tokens, E);
  let y = mean_pool_time(x);
  let logits = matmul(y, W);
  return logits;
}
"#;

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let lowered = lower(&program, 42).unwrap();

    // Create synthetic inputs for inference
    let mut token_ids = vec![];
    let mut inputs = Vec::new();

    // Add params as tensor inputs
    for (_, param) in &lowered.params {
        inputs.push(param.value.clone());
    }

    // Create token_ids input
    let batch_size = 2;
    let seq_len = 6;
    let vocab_size = 20;
    let mut rng = ailang_core::SeededRng::new(42 + 1000);
    let mut tokens = Vec::new();
    for _ in 0..batch_size * seq_len {
        tokens.push((rng.gen() * vocab_size as f32) as usize);
    }
    token_ids.push(ailang_core::TokenIds::new(&[batch_size, seq_len], tokens));

    // Prepend empty tensors for token_ids input slots
    let num_token_inputs = lowered
        .input_specs
        .iter()
        .filter(|s| s.dtype == "token_ids")
        .count();
    for _ in 0..num_token_inputs {
        inputs.insert(0, Tensor::zeros(&[]));
    }

    // Trace vector to record executed nodes
    let mut executed = Vec::new();

    // Run selective execution requesting only the return node
    let result = execute_forward_selected_with_capabilities_traced(
        &lowered.graph,
        &inputs,
        &token_ids,
        &[],
        &Capabilities::empty(),
        &[lowered.forward_output],
        Some(&mut executed),
    );

    // Assert: result is Ok
    assert!(result.is_ok(), "DSL program should execute successfully");

    let outputs = result.unwrap();
    assert_eq!(outputs.len(), 1, "Should return one output");

    // Check output shape: [batch_size, C] = [2, 2]
    let output = &outputs[0];
    assert_eq!(output.shape(), &[2, 2], "Output shape should be [2, 2]");

    // CRITICAL: executed trace must be non-empty, proving execution occurred
    assert!(
        executed.len() > 0,
        "At least some ops should execute for successful run"
    );

    // Verify that the output node is in the executed trace
    assert!(
        executed.contains(&lowered.forward_output),
        "Output node should be in executed trace"
    );
}

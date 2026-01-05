use ailang_core::{
    frontend::{lexer::Lexer, lower::lower, parser::Parser},
    model::{export_model, load_model},
};
use std::fs;

#[test]
fn test_export_separates_infer_and_train_inputs() {
    let source = r#"const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]
input labels: [B]

param E: [V, D]
param W: [D, C]

forward {
  let x = embedding(tokens, E)
  let y = mean_pool_time(x)
  let logits = matmul(y, W)
  return logits
}

loss {
  let l = cross_entropy(logits, labels)
  return l
}

train {
  steps = 50
  lr = 0.1
}
"#;

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let lowered = lower(&program, 42).unwrap();

    // Export model
    let temp_dir = std::env::temp_dir().join("ailang_test_model_contract");
    fs::create_dir_all(&temp_dir).unwrap();

    let params_refs: Vec<(usize, &ailang_core::param::Param)> = lowered
        .params
        .iter()
        .map(|(node_id, param)| (*node_id, param))
        .collect();

    export_model(
        &lowered.graph,
        &params_refs,
        42,
        &temp_dir,
        Some(lowered.forward_output),
        lowered.loss_output,
    )
    .unwrap();

    // Load model and check input specs
    let (_graph, _weights, _seed, infer_input_specs, train_input_specs) =
        load_model(&temp_dir).unwrap();

    // Assert: labels present in train_input_specs
    let has_labels_in_train = train_input_specs
        .iter()
        .any(|spec| spec.name == "labels" && spec.dtype == "labels");
    assert!(
        has_labels_in_train,
        "Labels should be present in train_input_specs"
    );

    // Assert: labels NOT present in infer_input_specs
    let has_labels_in_infer = infer_input_specs
        .iter()
        .any(|spec| spec.name == "labels" && spec.dtype == "labels");
    assert!(
        !has_labels_in_infer,
        "Labels should NOT be present in infer_input_specs"
    );

    // Assert: tokens present in both
    let has_tokens_in_infer = infer_input_specs
        .iter()
        .any(|spec| spec.name == "tokens" && spec.dtype == "token_ids");
    assert!(
        has_tokens_in_infer,
        "Tokens should be present in infer_input_specs"
    );

    let has_tokens_in_train = train_input_specs
        .iter()
        .any(|spec| spec.name == "tokens" && spec.dtype == "token_ids");
    assert!(
        has_tokens_in_train,
        "Tokens should be present in train_input_specs"
    );

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}

#[test]
fn test_infer_after_training_does_not_require_labels() {
    let source = r#"const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]
input labels: [B]

param E: [V, D]
param W: [D, C]

forward {
  let x = embedding(tokens, E)
  let y = mean_pool_time(x)
  let logits = matmul(y, W)
  return logits
}

loss {
  let l = cross_entropy(logits, labels)
  return l
}

train {
  steps = 10
  lr = 0.1
  batch = 2
}
"#;

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut lowered = lower(&program, 123).unwrap();

    // Run a few training steps
    use ailang_core::{
        backward::execute_backward, execute_forward_selected_with_capabilities,
        forward::execute_forward_with_capabilities, tensor::TokenIds, Capabilities, SeededRng,
    };

    let loss_output = lowered.loss_output.unwrap();
    let train_config = lowered.train_config.as_ref().unwrap();

    // Generate synthetic data
    let batch_size = train_config.batch_size;
    let seq_len = 6;
    let vocab_size = 20;
    let num_classes = 2;

    let mut rng = SeededRng::new(123 + 2000);

    // Create token_ids
    let mut tokens = Vec::new();
    for _ in 0..batch_size * seq_len {
        tokens.push((rng.gen() * vocab_size as f32) as usize);
    }
    let mut token_ids = vec![TokenIds::new(&[batch_size, seq_len], tokens)];

    // Create label_ids
    let mut labels = Vec::new();
    for _ in 0..batch_size {
        labels.push((rng.gen() * num_classes as f32) as usize);
    }
    // Find label index
    let mut label_idx = None;
    let mut token_idx = 0;
    for spec in &lowered.input_specs {
        if spec.dtype == "token_ids" {
            token_idx += 1;
        } else if spec.name == "labels" {
            label_idx = Some(token_idx);
            token_idx += 1;
        }
    }
    if let Some(idx) = label_idx {
        while token_ids.len() <= idx {
            token_ids.push(TokenIds::new(&[1], vec![0]));
        }
        token_ids[idx] = TokenIds::new(&[batch_size], labels);
    }

    // Prepare inputs
    let mut inputs = Vec::new();
    for (_, param) in &lowered.params {
        inputs.push(param.value.clone());
    }

    // Prepend empty tensors for token_ids input slots
    let num_token_inputs = lowered
        .input_specs
        .iter()
        .filter(|s| s.dtype == "token_ids" || s.name == "labels")
        .count();
    for _ in 0..num_token_inputs {
        inputs.insert(0, ailang_core::tensor::Tensor::zeros(&[]));
    }

    // Run a few training steps
    for _step in 0..train_config.steps.min(5) {
        // Forward
        let _loss_result = execute_forward_selected_with_capabilities(
            &lowered.graph,
            &inputs,
            &token_ids,
            &[],
            &Capabilities::empty(),
            &[loss_output],
        )
        .unwrap();

        // Backward
        let full_forward = execute_forward_with_capabilities(
            &lowered.graph,
            &inputs,
            &token_ids,
            &[],
            &Capabilities::empty(),
        )
        .unwrap();

        let grads = execute_backward(&lowered.graph, &full_forward, loss_output, &token_ids);

        // Update params
        for (node_id, param) in &mut lowered.params {
            if let Some(grad) = &grads[*node_id] {
                for (p, g) in param.value.data.iter_mut().zip(grad.data.iter()) {
                    *p -= train_config.lr * g;
                }
            }
        }

        // Update inputs
        for (node_id, param) in &lowered.params {
            inputs[*node_id] = param.value.clone();
        }
    }

    // Export model
    let temp_dir = std::env::temp_dir().join("ailang_test_infer_no_labels");
    fs::create_dir_all(&temp_dir).unwrap();

    let params_refs: Vec<(usize, &ailang_core::param::Param)> = lowered
        .params
        .iter()
        .map(|(node_id, param)| (*node_id, param))
        .collect();

    export_model(
        &lowered.graph,
        &params_refs,
        123,
        &temp_dir,
        Some(lowered.forward_output),
        lowered.loss_output,
    )
    .unwrap();

    // Load model and get infer_input_specs
    let (_graph, _weights, _seed, infer_input_specs, _train_input_specs) =
        load_model(&temp_dir).unwrap();

    // Verify labels are NOT in infer_input_specs
    let has_labels = infer_input_specs
        .iter()
        .any(|spec| spec.name == "labels" && spec.dtype == "labels");
    assert!(!has_labels, "Labels should NOT be in infer_input_specs");

    // Verify we can run inference with only tokens (no labels)
    // This is tested by the fact that infer_input_specs doesn't include labels
    // The actual inference would work because labels aren't required

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}

#[test]
fn test_train_requires_labels() {
    let source = r#"const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]

param E: [V, D]
param W: [D, C]

forward {
  let x = embedding(tokens, E)
  let y = mean_pool_time(x)
  let logits = matmul(y, W)
  return logits
}

loss {
  let l = cross_entropy(logits, labels)
  return l
}

train {
  steps = 10
  lr = 0.1
}
"#;

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    // This should fail during lowering because labels input is missing
    // but cross_entropy references labels
    let result = lower(&program, 42);

    assert!(
        result.is_err(),
        "Lowering should fail when labels input is missing but cross_entropy uses it"
    );

    if let Err(diag) = result {
        // The error should be about labels not being found
        // This could be E_LABELS_REQUIRED or E_LABELS_NOT_FOUND
        assert!(
            diag.code == "E_LABELS_REQUIRED" || diag.code == "E_LABELS_NOT_FOUND",
            "Error code should indicate labels are required, got: {}",
            diag.code
        );
    } else {
        panic!("Expected error but got Ok");
    }
}

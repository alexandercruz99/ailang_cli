use ailang_core::{
    backward::execute_backward,
    execute_forward_selected_with_capabilities,
    forward::execute_forward_with_capabilities,
    frontend::{lexer::Lexer, lower::lower, parser::Parser},
    tensor::TokenIds,
    Capabilities,
};

#[test]
fn test_training_lowering_produces_loss_node() {
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
  steps = 200
  lr = 0.1
}
"#;

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let lowered = lower(&program, 42).unwrap();

    // Assert loss_output is set
    assert!(
        lowered.loss_output.is_some(),
        "Loss output node should be created"
    );

    // Assert forward_output is set
    assert!(
        lowered.forward_output < lowered.graph.nodes.len(),
        "Forward output node should be valid"
    );

    // Assert train_config is set
    assert!(lowered.train_config.is_some(), "Train config should be set");
}

#[test]
fn test_training_loop_decreases_loss_deterministically() {
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
  batch = 4
}
"#;

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();

    let mut lowered = lower(&program, 12345).unwrap();

    let loss_output = lowered.loss_output.expect("Loss output must exist");
    let train_config = lowered
        .train_config
        .as_ref()
        .expect("Train config must exist");

    // Generate synthetic data
    let batch_size = train_config.batch_size;
    let seq_len = 6;
    let vocab_size = 20;
    let num_classes = 2;

    use ailang_core::SeededRng;
    let mut rng = SeededRng::new(12345 + 2000);

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

    // Training loop
    let steps = train_config.steps;
    let lr = train_config.lr;
    let mut initial_loss = None;
    let mut final_loss = None;

    for step in 0..steps {
        // Forward: compute loss
        let loss_result = execute_forward_selected_with_capabilities(
            &lowered.graph,
            &inputs,
            &token_ids,
            &[],
            &Capabilities::empty(),
            &[loss_output],
        );

        let loss_tensor = loss_result.expect("Loss computation should succeed");
        let loss_value = loss_tensor[0].scalar();

        if step == 0 {
            initial_loss = Some(loss_value);
        }
        if step == steps - 1 {
            final_loss = Some(loss_value);
        }

        // Backward: compute gradients
        let full_forward = execute_forward_with_capabilities(
            &lowered.graph,
            &inputs,
            &token_ids,
            &[],
            &Capabilities::empty(),
        )
        .expect("Full forward should succeed");

        let grads = execute_backward(&lowered.graph, &full_forward, loss_output, &token_ids);

        // Update params using gradients
        for (node_id, param) in &mut lowered.params {
            if let Some(grad) = &grads[*node_id] {
                // SGD step: param = param - lr * grad
                for (p, g) in param.value.data.iter_mut().zip(grad.data.iter()) {
                    *p -= lr * g;
                }
            }
        }

        // Update inputs array with updated params
        for (node_id, param) in &lowered.params {
            inputs[*node_id] = param.value.clone();
        }
    }

    // Assertions
    let initial = initial_loss.expect("Initial loss should be recorded");
    let final_val = final_loss.expect("Final loss should be recorded");

    assert!(
        final_val < initial,
        "Loss should decrease: initial={:.6}, final={:.6}",
        initial,
        final_val
    );
}

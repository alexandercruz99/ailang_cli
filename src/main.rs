mod diagnostic_format;

use ailang_core::{
    backward::execute_backward,
    dataset::{load_jsonl_dataset, load_tsv_dataset, shuffle_dataset, split_dataset, Dataset},
    execute_forward_selected_with_capabilities, export_model,
    forward::{execute_forward, execute_forward_with_capabilities},
    frontend::{lexer::Lexer, lower::lower, parser::Parser, EvalConfig},
    gradcheck_matmul_relu_sum,
    ir::{Graph, Op},
    load_model,
    param::Param,
    tensor::{Tensor, TokenIds},
    Capabilities, Capability, Diagnostic, DimBindings, DimSpec, InputSpec, ReshapeDim, ReshapeSpec,
    RuntimeError, SeededRng,
};
use std::env;
use std::f32::consts::SQRT_2;
use std::path::Path;

fn gradcheck_demo() {
    println!("Running gradient check...");
    let passed = gradcheck_matmul_relu_sum();
    if passed {
        println!("✓ GRADCHECK PASSED");
    } else {
        println!("✗ GRADCHECK FAILED");
        std::process::exit(1);
    }
}

fn training_demo(seed: u64) {
    println!("Training MLP: y = ReLU(X @ W) @ V");
    println!("Loss = mean((y - target)^2)");
    println!("Seed: {}\n", seed);

    // Hyperparameters
    const BATCH_SIZE: usize = 4;
    const INPUT_DIM: usize = 3;
    const HIDDEN_DIM: usize = 5;
    const OUTPUT_DIM: usize = 2;
    const LEARNING_RATE: f32 = 0.01;
    const NUM_STEPS: usize = 200;

    // Initialize RNG with seed
    let mut rng = SeededRng::new(seed);

    // Initialize parameters with seeded random values (Xavier-like init)
    let scale_w = SQRT_2 / (INPUT_DIM as f32).sqrt();
    let w_data: Vec<f32> = (0..INPUT_DIM * HIDDEN_DIM)
        .map(|_| rng.normal(0.0, scale_w * 0.1))
        .collect();
    let mut w = Param::new(Tensor::from_vec(&[INPUT_DIM, HIDDEN_DIM], w_data));

    let scale_v = SQRT_2 / (HIDDEN_DIM as f32).sqrt();
    let v_data: Vec<f32> = (0..HIDDEN_DIM * OUTPUT_DIM)
        .map(|_| rng.normal(0.0, scale_v * 0.1))
        .collect();
    let mut v = Param::new(Tensor::from_vec(&[HIDDEN_DIM, OUTPUT_DIM], v_data));

    // Create fixed input and target
    let x = Tensor::from_vec(
        &[BATCH_SIZE, INPUT_DIM],
        vec![0.5, 1.0, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.6, 0.4, 0.5, 0.8],
    );
    let target = Tensor::from_vec(
        &[BATCH_SIZE, OUTPUT_DIM],
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    );

    // Build computation graph: y = ReLU(x @ w) @ v, loss = mean((y - target)^2)
    // Inputs: [x, w.value, v.value, target]
    let mut graph = Graph::new(4);
    let x_id = graph.input_node(0);
    let w_id = graph.input_node(1);
    let v_id = graph.input_node(2);
    let target_id = graph.input_node(3);

    // Forward: ReLU(x @ w) @ v
    let matmul1_id = graph.add_node(Op::MatMul2D, vec![x_id, w_id]);
    let relu_id = graph.add_node(Op::ReLU, vec![matmul1_id]);
    let matmul2_id = graph.add_node(Op::MatMul2D, vec![relu_id, v_id]);
    let y_id = matmul2_id;

    // Loss: mean((y - target)^2)
    let diff_id = graph.add_node(Op::Sub, vec![y_id, target_id]);
    let sq_id = graph.add_node(Op::Mul, vec![diff_id, diff_id]);
    let loss_id = graph.add_node(Op::Mean, vec![sq_id]);

    // Training loop
    for step in 0..NUM_STEPS {
        // Zero gradients
        w.zero_grad();
        v.zero_grad();

        // Forward pass
        let inputs = vec![x.clone(), w.value.clone(), v.value.clone(), target.clone()];
        let activations = execute_forward(&graph, &inputs, &[]).unwrap();
        let loss_val = activations[loss_id].scalar();

        // Backward pass
        let grads = execute_backward(&graph, &activations, loss_id, &[]);

        // Update parameter gradients (copy from graph gradients)
        if let Some(grad_w) = &grads[w_id] {
            w.grad.data = grad_w.data.clone();
        }
        if let Some(grad_v) = &grads[v_id] {
            v.grad.data = grad_v.data.clone();
        }

        // SGD step
        w.sgd_step(LEARNING_RATE);
        v.sgd_step(LEARNING_RATE);

        if step < 10 || step % 20 == 0 || step == NUM_STEPS - 1 {
            println!("Step {}: loss = {:.6}", step, loss_val);
        }
    }

    println!("\nTraining complete!");
}

fn toy_classifier_demo(seed: u64) {
    println!("Toy Token Classifier: Embedding -> Linear -> CrossEntropy");
    println!("Task: Classify tokens into 2 classes (even/odd)");
    println!("Seed: {}\n", seed);

    // Hyperparameters
    const VOCAB_SIZE: usize = 20;
    const EMBED_DIM: usize = 8;
    const BATCH_SIZE: usize = 8;
    const NUM_CLASSES: usize = 2;
    const LEARNING_RATE: f32 = 0.1;
    const NUM_STEPS: usize = 200;

    // Initialize RNG with seed
    let mut rng = SeededRng::new(seed);

    // Initialize embedding weight
    let scale = SQRT_2 / (VOCAB_SIZE as f32).sqrt();
    let embed_data: Vec<f32> = (0..VOCAB_SIZE * EMBED_DIM)
        .map(|_| rng.normal(0.0, scale * 0.1))
        .collect();
    let mut embed_weight = Param::new(Tensor::from_vec(&[VOCAB_SIZE, EMBED_DIM], embed_data));

    // Initialize linear layer
    let linear_scale = SQRT_2 / (EMBED_DIM as f32).sqrt();
    let linear_data: Vec<f32> = (0..EMBED_DIM * NUM_CLASSES)
        .map(|_| rng.normal(0.0, linear_scale * 0.1))
        .collect();
    let mut linear_weight = Param::new(Tensor::from_vec(&[EMBED_DIM, NUM_CLASSES], linear_data));

    // Create synthetic dataset using seeded RNG
    let mut rng_data = SeededRng::new(seed + 1000); // Different seed for data
    let token_ids_input_data: Vec<usize> = (0..BATCH_SIZE)
        .map(|_| (rng_data.gen() * VOCAB_SIZE as f32) as usize)
        .collect();
    let target_labels: Vec<usize> = token_ids_input_data
        .iter()
        .map(|&t| t % NUM_CLASSES)
        .collect();
    let token_ids_input = TokenIds::new(&[BATCH_SIZE], token_ids_input_data);
    let token_ids_target = TokenIds::new(&[BATCH_SIZE], target_labels.clone());

    // Build graph: Embedding -> MatMul -> CrossEntropy
    let mut graph = Graph::new_with_token_ids(2, 2);
    let embed_w_id = graph.input_node(0);
    let linear_w_id = graph.input_node(1);

    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);
    let logits_id = graph.add_node(Op::MatMul2D, vec![embed_id, linear_w_id]);
    let loss_id = graph.add_node(Op::CrossEntropy(1), vec![logits_id]);

    // Training loop
    for step in 0..NUM_STEPS {
        // Zero gradients
        embed_weight.zero_grad();
        linear_weight.zero_grad();

        // Forward pass
        let inputs = vec![embed_weight.value.clone(), linear_weight.value.clone()];
        let token_ids = vec![token_ids_input.clone(), token_ids_target.clone()];
        let activations = execute_forward(&graph, &inputs, &token_ids).unwrap();
        let loss_val = activations[loss_id].scalar();

        // Compute accuracy
        let logits = &activations[logits_id];
        let logits_slice = logits.data.as_slice().unwrap();
        let mut correct = 0;
        for i in 0..BATCH_SIZE {
            let offset = i * NUM_CLASSES;
            let pred = if logits_slice[offset] > logits_slice[offset + 1] {
                0
            } else {
                1
            };
            if pred == target_labels[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / BATCH_SIZE as f32;

        // Backward pass
        let grads = execute_backward(&graph, &activations, loss_id, &token_ids);

        // Update gradients
        if let Some(grad_embed) = &grads[embed_w_id] {
            embed_weight.grad.data = grad_embed.data.clone();
        }
        if let Some(grad_linear) = &grads[linear_w_id] {
            linear_weight.grad.data = grad_linear.data.clone();
        }

        // SGD step
        embed_weight.sgd_step(LEARNING_RATE);
        linear_weight.sgd_step(LEARNING_RATE);

        if step < 10 || step % 40 == 0 || step == NUM_STEPS - 1 {
            println!(
                "Step {}: loss = {:.6}, accuracy = {:.2}%",
                step,
                loss_val,
                accuracy * 100.0
            );
        }
    }

    println!("\nTraining complete!");
}

fn attention_demo(seed: u64, export_path: Option<&Path>) {
    println!("Single-Head Self-Attention Classifier");
    println!("Model: Embedding -> Attention -> MeanPool -> Linear -> CrossEntropy");
    println!("Seed: {}\n", seed);

    // Hyperparameters
    const VOCAB_SIZE: usize = 20;
    const EMBED_DIM: usize = 8;
    const SEQ_LEN: usize = 6;
    const BATCH_SIZE: usize = 32;
    const NUM_CLASSES: usize = 2;
    const LEARNING_RATE: f32 = 0.05;
    const NUM_STEPS: usize = 400;

    // Initialize RNG with seed
    let mut rng = SeededRng::new(seed);

    // Initialize embedding weight
    let scale_embed = SQRT_2 / (VOCAB_SIZE as f32).sqrt();
    let embed_data: Vec<f32> = (0..VOCAB_SIZE * EMBED_DIM)
        .map(|_| rng.normal(0.0, scale_embed * 0.1))
        .collect();
    let mut embed_weight = Param::new(Tensor::from_vec(&[VOCAB_SIZE, EMBED_DIM], embed_data));

    // Initialize attention projection weights: Q, K, V
    let scale_attn = SQRT_2 / (EMBED_DIM as f32).sqrt();
    let wq_data: Vec<f32> = (0..EMBED_DIM * EMBED_DIM)
        .map(|_| rng.normal(0.0, scale_attn * 0.1))
        .collect();
    let mut wq = Param::new(Tensor::from_vec(&[EMBED_DIM, EMBED_DIM], wq_data));

    let wk_data: Vec<f32> = (0..EMBED_DIM * EMBED_DIM)
        .map(|_| rng.normal(0.0, scale_attn * 0.1))
        .collect();
    let mut wk = Param::new(Tensor::from_vec(&[EMBED_DIM, EMBED_DIM], wk_data));

    let wv_data: Vec<f32> = (0..EMBED_DIM * EMBED_DIM)
        .map(|_| rng.normal(0.0, scale_attn * 0.1))
        .collect();
    let mut wv = Param::new(Tensor::from_vec(&[EMBED_DIM, EMBED_DIM], wv_data));

    // Initialize output projection
    let scale_out = SQRT_2 / (EMBED_DIM as f32).sqrt();
    let out_data: Vec<f32> = (0..EMBED_DIM * NUM_CLASSES)
        .map(|_| rng.normal(0.0, scale_out * 0.1))
        .collect();
    let mut out_weight = Param::new(Tensor::from_vec(&[EMBED_DIM, NUM_CLASSES], out_data));

    // Create synthetic dataset using seeded RNG
    let mut rng_data = SeededRng::new(seed + 2000); // Different seed for data
    let mut token_seqs = Vec::new();
    let mut target_labels = Vec::new();
    for _b in 0..BATCH_SIZE {
        let mut seq = Vec::new();
        let mut has_token_zero = false;
        for _t in 0..SEQ_LEN {
            let token = (rng_data.gen() * VOCAB_SIZE as f32) as usize;
            seq.push(token);
            if token == 0 {
                has_token_zero = true;
            }
        }
        token_seqs.push(seq);
        target_labels.push(if has_token_zero { 1 } else { 0 });
    }

    // Flatten token sequences for TokenIds
    let token_ids_seq: Vec<usize> = token_seqs.iter().flat_map(|s| s.iter().copied()).collect();
    let token_ids_input = TokenIds::new(&[BATCH_SIZE, SEQ_LEN], token_ids_seq);
    let token_ids_target = TokenIds::new(&[BATCH_SIZE], target_labels.clone());

    // Build graph: Embedding -> Attention -> MeanPool -> Linear -> CrossEntropy
    // Inputs: [embed_weight, wq, wk, wv, out_weight]
    let mut graph = Graph::new_with_token_ids(5, 2);

    // Declare input signature: tokens: [B, T] where B is variable, T is fixed
    // Also need to bind D (embedding dimension) - it will be bound from embedding output
    let input_specs = vec![InputSpec::new(
        "tokens".to_string(),
        "token_ids".to_string(),
        vec![
            DimSpec::Named("B".to_string()), // Batch size (variable)
            DimSpec::Literal(SEQ_LEN),       // Sequence length (fixed) - also bound as "T"
        ],
    )];
    // Also bind T and D as named dimensions for use in reshapes
    // T will be bound from input validation, D will be bound from embedding output
    graph = graph.with_input_specs(input_specs);

    let embed_w_id = graph.input_node(0);
    let wq_id = graph.input_node(1);
    let wk_id = graph.input_node(2);
    let wv_id = graph.input_node(3);
    let out_w_id = graph.input_node(4);

    // Embedding: [B, T] -> [B, T, D]
    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);

    // Attention computation
    // Use named dimensions: [B, T, D] -> [B*T, D] using mul("B", "T")
    let embed_reshaped_q = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Mul(
                Box::new(ReshapeDim::Named("B".to_string())),
                Box::new(ReshapeDim::Named("T".to_string())),
            ),
            ReshapeDim::Named("D".to_string()),
        ])),
        vec![embed_id],
    );
    let q_reshaped = graph.add_node(Op::MatMul2D, vec![embed_reshaped_q, wq_id]);
    // Reshape back: [B*T, D] -> [B, T, D] using named dims
    let q_id = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Named("B".to_string()),
            ReshapeDim::Named("T".to_string()),
            ReshapeDim::Named("D".to_string()),
        ])),
        vec![q_reshaped],
    );

    let embed_reshaped_k = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Mul(
                Box::new(ReshapeDim::Named("B".to_string())),
                Box::new(ReshapeDim::Named("T".to_string())),
            ),
            ReshapeDim::Named("D".to_string()),
        ])),
        vec![embed_id],
    );
    let k_reshaped = graph.add_node(Op::MatMul2D, vec![embed_reshaped_k, wk_id]);
    let k_id = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Named("B".to_string()),
            ReshapeDim::Named("T".to_string()),
            ReshapeDim::Named("D".to_string()),
        ])),
        vec![k_reshaped],
    );

    let embed_reshaped_v = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Mul(
                Box::new(ReshapeDim::Named("B".to_string())),
                Box::new(ReshapeDim::Named("T".to_string())),
            ),
            ReshapeDim::Named("D".to_string()),
        ])),
        vec![embed_id],
    );
    let v_reshaped = graph.add_node(Op::MatMul2D, vec![embed_reshaped_v, wv_id]);
    let v_id = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Named("B".to_string()),
            ReshapeDim::Named("T".to_string()),
            ReshapeDim::Named("D".to_string()),
        ])),
        vec![v_reshaped],
    );

    // K^T: [B, T, D] -> [B, D, T]
    let k_transpose_id = graph.add_node(Op::Transpose3D, vec![k_id]);

    // scores = Q @ K^T: [B, T, D] @ [B, D, T] -> [B, T, T]
    let scores_id = graph.add_node(Op::BatchMatMul, vec![q_id, k_transpose_id]);

    // weights = Softmax(scores, axis=2): [B, T, T]
    let weights_id = graph.add_node(Op::Softmax(2), vec![scores_id]);

    // context = weights @ V: [B, T, T] @ [B, T, D] -> [B, T, D]
    let context_id = graph.add_node(Op::BatchMatMul, vec![weights_id, v_id]);

    // MeanPoolTime: [B, T, D] -> [B, D]
    let pool_id = graph.add_node(Op::MeanPoolTime, vec![context_id]);

    // Linear: [B, D] @ [D, NUM_CLASSES] -> [B, NUM_CLASSES]
    let logits_id = graph.add_node(Op::MatMul2D, vec![pool_id, out_w_id]);
    let loss_id = graph.add_node(Op::CrossEntropy(1), vec![logits_id]);

    // Training loop
    for step in 0..NUM_STEPS {
        // Zero gradients
        embed_weight.zero_grad();
        wq.zero_grad();
        wk.zero_grad();
        wv.zero_grad();
        out_weight.zero_grad();

        // Forward pass
        let inputs = vec![
            embed_weight.value.clone(),
            wq.value.clone(),
            wk.value.clone(),
            wv.value.clone(),
            out_weight.value.clone(),
        ];
        let token_ids = vec![token_ids_input.clone(), token_ids_target.clone()];
        let activations = execute_forward(&graph, &inputs, &token_ids).unwrap();
        let loss_val = activations[loss_id].scalar();

        // Compute accuracy
        let logits = &activations[logits_id];
        let logits_slice = logits.data.as_slice().unwrap();
        let mut correct = 0;
        for i in 0..BATCH_SIZE {
            let offset = i * NUM_CLASSES;
            let pred = if logits_slice[offset] > logits_slice[offset + 1] {
                0
            } else {
                1
            };
            if pred == target_labels[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / BATCH_SIZE as f32;

        // Backward pass
        let grads = execute_backward(&graph, &activations, loss_id, &token_ids);

        // Update gradients
        if let Some(grad_embed) = &grads[embed_w_id] {
            embed_weight.grad.data = grad_embed.data.clone();
        }
        if let Some(grad_wq) = &grads[wq_id] {
            wq.grad.data = grad_wq.data.clone();
        }
        if let Some(grad_wk) = &grads[wk_id] {
            wk.grad.data = grad_wk.data.clone();
        }
        if let Some(grad_wv) = &grads[wv_id] {
            wv.grad.data = grad_wv.data.clone();
        }
        if let Some(grad_out) = &grads[out_w_id] {
            out_weight.grad.data = grad_out.data.clone();
        }

        // SGD step
        embed_weight.sgd_step(LEARNING_RATE);
        wq.sgd_step(LEARNING_RATE);
        wk.sgd_step(LEARNING_RATE);
        wv.sgd_step(LEARNING_RATE);
        out_weight.sgd_step(LEARNING_RATE);

        if step < 10 || step % 50 == 0 || step == NUM_STEPS - 1 {
            println!(
                "Step {}: loss = {:.6}, accuracy = {:.2}%",
                step,
                loss_val,
                accuracy * 100.0
            );
        }
    }

    // Export model if path provided
    if let Some(path) = export_path {
        let params = vec![
            (embed_w_id, &embed_weight),
            (wq_id, &wq),
            (wk_id, &wk),
            (wv_id, &wv),
            (out_w_id, &out_weight),
        ];
        if let Err(e) = export_model(&graph, &params, seed, path, None, None) {
            eprintln!("Failed to export model: {}", e);
        } else {
            println!("\nModel exported to: {:?}", path);
        }
    }

    println!("\nTraining complete!");
}

fn inference_demo(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running inference on model: {:?}\n", model_path);

    // Load model
    let (graph, weights, seed, infer_input_specs, _train_input_specs) = match load_model(model_path)
    {
        Ok(x) => x,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    println!("Loaded model (seed: {})", seed);
    println!(
        "Graph has {} nodes, {} inputs, {} token_ids",
        graph.nodes.len(),
        graph.input_count,
        graph.token_ids_count
    );
    println!("Loaded {} weight tensors", weights.len());
    println!("Inference inputs: {}", infer_input_specs.len());
    for spec in &infer_input_specs {
        println!("  - {}: {} {:?}", spec.name, spec.dtype, spec.dims);
    }
    println!();

    // Use infer_input_specs to determine what inputs are needed
    // Labels should NOT be in infer_input_specs
    let mut token_ids_needed = Vec::new();
    for spec in &infer_input_specs {
        if spec.dtype == "token_ids" {
            token_ids_needed.push(spec.clone());
        }
    }

    // Create a small fixed input for inference
    let mut rng = SeededRng::new(seed + 3000);
    const VOCAB_SIZE: usize = 20;
    const SEQ_LEN: usize = 6;

    // Test with batch size 1
    let batch_size_1 = 1;
    let mut token_ids_1 = Vec::new();
    for spec in &token_ids_needed {
        let token_ids_seq: Vec<usize> = (0..batch_size_1 * SEQ_LEN)
            .map(|_| (rng.gen() * VOCAB_SIZE as f32) as usize)
            .collect();
        token_ids_1.push(TokenIds::new(&[batch_size_1, SEQ_LEN], token_ids_seq));
    }

    // Find the logits node (input to CrossEntropy, or last MatMul2D before CrossEntropy)
    let mut output_node_id = None;
    for (_i, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::CrossEntropy(_)) {
            // CrossEntropy takes logits as input
            if let Some(&logits_id) = node.inputs.first() {
                output_node_id = Some(logits_id);
                break;
            }
        }
    }
    // Fallback: find last non-Input, non-CrossEntropy node
    if output_node_id.is_none() {
        for (i, node) in graph.nodes.iter().enumerate().rev() {
            if !matches!(node.op, Op::Input(_) | Op::CrossEntropy(_)) {
                output_node_id = Some(i);
                break;
            }
        }
    }

    if let Some(output_id) = output_node_id {
        // Forward pass only (inference) - NO labels needed
        let inputs: Vec<Tensor> = weights.iter().map(|(_, t)| t.clone()).collect();
        let activations_1 =
            execute_forward(&graph, &inputs, &token_ids_1).map_err(|e: RuntimeError| {
                let diag = e.diagnostic();
                diagnostic_format::print_diagnostic(&diag);
                format!("Inference failed with error code: {}", diag.code)
            })?;
        let output_1 = &activations_1[output_id];

        println!("Test 1: Batch size 1");
        if !token_ids_1.is_empty() {
            println!("Input token sequence: {:?}", token_ids_1[0].data);
        }
        println!("Output shape: {:?}", output_1.shape());

        // Print logits if it's a classification output
        if output_1.shape().len() == 2 && output_1.shape()[1] <= 10 {
            let logits_slice = output_1.data.as_slice().unwrap();
            print!("Logits: [");
            for (i, &val) in logits_slice.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.4}", val);
            }
            println!("]");

            // Predict class
            let mut max_idx = 0;
            let mut max_val = logits_slice[0];
            for (i, &val) in logits_slice.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            println!("Predicted class: {}", max_idx);
        } else {
            println!("Output (first 10 values):");
            let slice = output_1.data.as_slice().unwrap();
            for (i, &val) in slice.iter().take(10).enumerate() {
                println!("  [{}] = {:.6}", i, val);
            }
        }

        // Test 2: Different batch size (B=3, same T, D)
        println!("\nTest 2: Batch size 3 (different from training)");
        let batch_size_2 = 3;
        let token_ids_seq_2: Vec<usize> = (0..batch_size_2 * SEQ_LEN)
            .map(|_| (rng.gen() * VOCAB_SIZE as f32) as usize)
            .collect();
        let token_ids_input_2 = TokenIds::new(&[batch_size_2, SEQ_LEN], token_ids_seq_2.clone());
        let token_ids_target_2 = TokenIds::new(&[batch_size_2], vec![0; batch_size_2]);
        let token_ids_2 = if graph.token_ids_count == 2 {
            vec![token_ids_input_2, token_ids_target_2]
        } else {
            vec![token_ids_input_2]
        };

        let activations_2 =
            execute_forward(&graph, &inputs, &token_ids_2).map_err(|e: RuntimeError| {
                let diag = e.diagnostic();
                diagnostic_format::print_diagnostic(&diag);
                format!("Inference failed with error code: {}", diag.code)
            })?;
        let output_2 = &activations_2[output_id];
        println!("Input token sequence: {:?}", token_ids_2[0].data);
        println!("Output shape: {:?} (expected [3, 2])", output_2.shape());
        assert_eq!(
            output_2.shape(),
            &[batch_size_2, 2],
            "Output shape should be [3, 2]"
        );
        println!("✓ Inference with different batch size works!");
    } else {
        eprintln!("Could not find output node");
        return Err("Could not find output node".into());
    }

    Ok(())
}

fn attention_gradcheck() -> bool {
    println!("Running attention path gradient check (finite difference)...");

    // Tiny sizes for gradcheck
    const B: usize = 1;
    const T: usize = 2;
    const D: usize = 3;
    const VOCAB: usize = 5;
    const NUM_CLASSES: usize = 2;
    const EPS: f32 = 1e-4;

    // Initialize with fixed seed for reproducibility
    let seed = 42;
    let mut rng = SeededRng::new(seed);

    // Create tiny model
    let embed_data: Vec<f32> = (0..VOCAB * D).map(|_| rng.normal(0.0, 0.1)).collect();
    let embed_weight = Param::new(Tensor::from_vec(&[VOCAB, D], embed_data));

    let wq_data: Vec<f32> = (0..D * D).map(|_| rng.normal(0.0, 0.1)).collect();
    let wq = Param::new(Tensor::from_vec(&[D, D], wq_data));

    let wk_data: Vec<f32> = (0..D * D).map(|_| rng.normal(0.0, 0.1)).collect();
    let wk = Param::new(Tensor::from_vec(&[D, D], wk_data));

    let wv_data: Vec<f32> = (0..D * D).map(|_| rng.normal(0.0, 0.1)).collect();
    let wv = Param::new(Tensor::from_vec(&[D, D], wv_data));

    let out_data: Vec<f32> = (0..D * NUM_CLASSES).map(|_| rng.normal(0.0, 0.1)).collect();
    let out_weight = Param::new(Tensor::from_vec(&[D, NUM_CLASSES], out_data));

    // Fixed input
    let token_ids_seq = vec![0, 1];
    let token_ids_input = TokenIds::new(&[B, T], token_ids_seq);
    let token_ids_target = TokenIds::new(&[B], vec![0]);

    // Build graph
    let mut graph = Graph::new_with_token_ids(5, 2);
    let embed_w_id = graph.input_node(0);
    let wq_id = graph.input_node(1);
    let wk_id = graph.input_node(2);
    let wv_id = graph.input_node(3);
    let out_w_id = graph.input_node(4);

    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);
    let embed_reshaped_q = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Mul(Box::new(ReshapeDim::Ref(0)), Box::new(ReshapeDim::Ref(1))),
            ReshapeDim::Ref(2),
        ])),
        vec![embed_id],
    );
    let q_reshaped = graph.add_node(Op::MatMul2D, vec![embed_reshaped_q, wq_id]);
    let q_id = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Inferred,
            ReshapeDim::Literal(T),
            ReshapeDim::Literal(D),
        ])),
        vec![q_reshaped],
    );

    let embed_reshaped_k = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Mul(Box::new(ReshapeDim::Ref(0)), Box::new(ReshapeDim::Ref(1))),
            ReshapeDim::Ref(2),
        ])),
        vec![embed_id],
    );
    let k_reshaped = graph.add_node(Op::MatMul2D, vec![embed_reshaped_k, wk_id]);
    let k_id = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Inferred,
            ReshapeDim::Literal(T),
            ReshapeDim::Literal(D),
        ])),
        vec![k_reshaped],
    );

    let embed_reshaped_v = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Mul(Box::new(ReshapeDim::Ref(0)), Box::new(ReshapeDim::Ref(1))),
            ReshapeDim::Ref(2),
        ])),
        vec![embed_id],
    );
    let v_reshaped = graph.add_node(Op::MatMul2D, vec![embed_reshaped_v, wv_id]);
    let v_id = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Inferred,
            ReshapeDim::Literal(T),
            ReshapeDim::Literal(D),
        ])),
        vec![v_reshaped],
    );

    let k_transpose_id = graph.add_node(Op::Transpose3D, vec![k_id]);
    let scores_id = graph.add_node(Op::BatchMatMul, vec![q_id, k_transpose_id]);
    let weights_id = graph.add_node(Op::Softmax(2), vec![scores_id]);
    let context_id = graph.add_node(Op::BatchMatMul, vec![weights_id, v_id]);
    let pool_id = graph.add_node(Op::MeanPoolTime, vec![context_id]);
    let logits_id = graph.add_node(Op::MatMul2D, vec![pool_id, out_w_id]);
    let loss_id = graph.add_node(Op::CrossEntropy(1), vec![logits_id]);

    // Forward pass
    let inputs = vec![
        embed_weight.value.clone(),
        wq.value.clone(),
        wk.value.clone(),
        wv.value.clone(),
        out_weight.value.clone(),
    ];
    let token_ids = vec![token_ids_input.clone(), token_ids_target.clone()];
    let activations = execute_forward(&graph, &inputs, &token_ids).unwrap();
    let loss_base = activations[loss_id].scalar();

    // Backward pass to get analytical gradient
    let grads = execute_backward(&graph, &activations, loss_id, &token_ids);

    // Check gradient for embed_weight[0, 0]
    let embed_data_vec: Vec<f32> = embed_weight.value.data.iter().copied().collect();
    let mut embed_data_perturbed = embed_data_vec.clone();
    embed_data_perturbed[0] += EPS;
    let embed_weight_perturbed = Tensor::from_vec(embed_weight.value.shape(), embed_data_perturbed);

    let inputs_perturbed = vec![
        embed_weight_perturbed,
        wq.value.clone(),
        wk.value.clone(),
        wv.value.clone(),
        out_weight.value.clone(),
    ];
    let activations_perturbed = execute_forward(&graph, &inputs_perturbed, &token_ids).unwrap();
    let loss_perturbed = activations_perturbed[loss_id].scalar();

    let grad_numerical = (loss_perturbed - loss_base) / EPS;
    let grad_analytical = if let Some(grad) = &grads[embed_w_id] {
        let grad_vec: Vec<f32> = grad.data.iter().copied().collect();
        grad_vec[0]
    } else {
        0.0
    };

    println!("Checking gradient for embed_weight[0, 0]:");
    println!("  Numerical:  {:.8}", grad_numerical);
    println!("  Analytical: {:.8}", grad_analytical);
    println!(
        "  Difference: {:.8}",
        (grad_numerical - grad_analytical).abs()
    );

    let diff = (grad_numerical - grad_analytical).abs();
    let passed = diff < 1e-3;

    // Check gradient for out_weight[0, 0]
    let out_data_vec: Vec<f32> = out_weight.value.data.iter().copied().collect();
    let mut out_data_perturbed = out_data_vec.clone();
    out_data_perturbed[0] += EPS;
    let out_weight_perturbed = Tensor::from_vec(out_weight.value.shape(), out_data_perturbed);

    let inputs_perturbed2 = vec![
        embed_weight.value.clone(),
        wq.value.clone(),
        wk.value.clone(),
        wv.value.clone(),
        out_weight_perturbed,
    ];
    let activations_perturbed2 = execute_forward(&graph, &inputs_perturbed2, &token_ids).unwrap();
    let loss_perturbed2 = activations_perturbed2[loss_id].scalar();

    let grad_numerical2 = (loss_perturbed2 - loss_base) / EPS;
    let grad_analytical2 = if let Some(grad) = &grads[out_w_id] {
        let grad_vec: Vec<f32> = grad.data.iter().copied().collect();
        grad_vec[0]
    } else {
        0.0
    };

    println!("\nChecking gradient for out_weight[0, 0]:");
    println!("  Numerical:  {:.8}", grad_numerical2);
    println!("  Analytical: {:.8}", grad_analytical2);
    println!(
        "  Difference: {:.8}",
        (grad_numerical2 - grad_analytical2).abs()
    );

    let diff2 = (grad_numerical2 - grad_analytical2).abs();
    let passed2 = diff2 < 1e-3;

    if passed && passed2 {
        println!("\n✓ ATTENTION GRADCHECK PASSED");
        true
    } else {
        println!("\n✗ ATTENTION GRADCHECK FAILED");
        false
    }
}

fn now_demo(capabilities: &Capabilities) {
    println!("Now Demo: Getting current time");
    println!("Capabilities: {:?}\n", capabilities.allowed);

    // Build a tiny graph with Op::Now
    let mut graph = Graph::new(0);
    let now_id = graph.add_node(Op::Now, vec![]);

    // Execute
    let inputs = vec![];
    let token_ids = vec![];
    let paths = vec![];
    match execute_forward_with_capabilities(&graph, &inputs, &token_ids, &paths, capabilities) {
        Ok(activations) => {
            let time = activations[now_id].scalar();
            println!("Current time: {:.6} seconds since Unix epoch", time);
            println!("✓ Successfully read system clock");
        }
        Err(e) => {
            let diag = e.diagnostic();
            diagnostic_format::print_diagnostic(&diag);
            std::process::exit(1);
        }
    }
}

fn read_file_demo(file_path: &str, capabilities: &Capabilities) {
    println!("Read File Demo: Reading and hashing file");
    println!("File path: {}", file_path);
    println!("Capabilities: {:?}\n", capabilities.allowed);

    // Build a tiny graph with Op::ReadFileText
    let mut graph = Graph::new(0);
    let paths = vec![file_path.to_string()];
    let read_id = graph.add_node(Op::ReadFileText(0), vec![]);

    // Execute
    let inputs = vec![];
    let token_ids = vec![];
    match execute_forward_with_capabilities(&graph, &inputs, &token_ids, &paths, capabilities) {
        Ok(activations) => {
            let hash_output = &activations[read_id];
            println!("File hash vector (8 f32 values):");
            let hash_slice = hash_output.data.as_slice().unwrap();
            for (i, &val) in hash_slice.iter().enumerate() {
                println!("  [{}] = {:.6}", i, val);
            }
            println!("✓ Successfully read and hashed file");
        }
        Err(e) => {
            let diag = e.diagnostic();
            diagnostic_format::print_diagnostic(&diag);
            std::process::exit(1);
        }
    }
}

fn selective_demo(capabilities: &Capabilities) {
    println!("Selective Evaluation Demo: Mixed Safe + Effect Graph");
    println!("Capabilities: {:?}\n", capabilities.allowed);

    // Build a graph with TWO branches:
    // 1. Safe branch: Linear (MatMul2D) + ReLU
    // 2. Effect branch: ReadFileText

    let mut graph = Graph::new(2); // 2 tensor inputs

    // Safe branch: input -> MatMul2D -> ReLU
    let weight_id = graph.input_node(0);
    let input_id = graph.input_node(1);
    let matmul_id = graph.add_node(Op::MatMul2D, vec![input_id, weight_id]);
    let safe_output_id = graph.add_node(Op::ReLU, vec![matmul_id]);

    // Effect branch: ReadFileText (not connected to safe branch)
    let temp_file =
        std::env::temp_dir().join(format!("ailang_selective_demo_{}.txt", std::process::id()));
    std::fs::write(&temp_file, b"demo content").unwrap();
    let paths = vec![temp_file.to_str().unwrap().to_string()];
    let effect_output_id = graph.add_node(Op::ReadFileText(0), vec![]);

    let inputs = vec![
        Tensor::from_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), // weight [3, 2]
        Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), // input [2, 3]
    ];
    let token_ids = vec![];

    println!("Test 1: Request safe output with empty capabilities");
    println!("  (ReadFileText should be skipped, not checked)");
    match execute_forward_selected_with_capabilities(
        &graph,
        &inputs,
        &token_ids,
        &paths,
        capabilities,
        &[safe_output_id],
    ) {
        Ok(outputs) => {
            println!("  ✓ Success! Safe output computed without FileRead capability");
            println!("  Output shape: {:?}", outputs[0].shape());
        }
        Err(e) => {
            let diag = e.diagnostic();
            diagnostic_format::print_diagnostic(&diag);
            println!("  ✗ Failed (unexpected)");
        }
    }

    println!("\nTest 2: Request effect output with empty capabilities");
    println!("  (ReadFileText is required, should be denied)");
    match execute_forward_selected_with_capabilities(
        &graph,
        &inputs,
        &token_ids,
        &paths,
        capabilities,
        &[effect_output_id],
    ) {
        Ok(_) => {
            println!("  ✗ Unexpected success (should have been denied)");
        }
        Err(e) => {
            let diag = e.diagnostic();
            println!("  ✓ Correctly denied:");
            diagnostic_format::print_diagnostic(&diag);
        }
    }

    println!("\nTest 3: Request effect output WITH FileRead capability");
    let capabilities_with_file = capabilities.clone().with(Capability::FileRead);
    match execute_forward_selected_with_capabilities(
        &graph,
        &inputs,
        &token_ids,
        &paths,
        &capabilities_with_file,
        &[effect_output_id],
    ) {
        Ok(outputs) => {
            println!("  ✓ Success! Effect output computed with FileRead capability");
            println!("  Output shape: {:?}", outputs[0].shape());
            let hash_slice = outputs[0].data.as_slice().unwrap();
            println!(
                "  Hash vector (first 3 values): [{:.2}, {:.2}, {:.2}, ...]",
                hash_slice[0], hash_slice[1], hash_slice[2]
            );
        }
        Err(e) => {
            let diag = e.diagnostic();
            diagnostic_format::print_diagnostic(&diag);
            println!("  ✗ Failed (unexpected)");
        }
    }

    // Cleanup
    std::fs::remove_file(&temp_file).ok();
}

fn run_ail_file(file_path: &str, seed: u64, export_dir: Option<&str>, capabilities: &Capabilities) {
    use std::fs;
    use std::path::Path;

    println!("Running AIL file: {}", file_path);
    println!("Seed: {}\n", seed);

    // Read file
    let source = fs::read_to_string(file_path).unwrap_or_else(|e| {
        eprintln!("Failed to read file {}: {}", file_path, e);
        std::process::exit(1);
    });

    // Lex
    let mut lexer = Lexer::new(&source);
    let tokens = lexer
        .tokenize()
        .map_err(|e| {
            eprintln!("Lexer error: {}", e);
            std::process::exit(1);
        })
        .unwrap();

    // Parse
    let mut parser = Parser::new(tokens);
    let program = parser
        .parse()
        .map_err(|diag| {
            diagnostic_format::print_diagnostic(&diag);
            std::process::exit(1);
        })
        .unwrap();

    // Lower
    let lowered = lower(&program, seed)
        .map_err(|diag| {
            diagnostic_format::print_diagnostic(&diag);
            std::process::exit(1);
        })
        .unwrap();

    println!("✓ Parsed and lowered successfully");
    println!("  Graph nodes: {}", lowered.graph.nodes.len());
    println!("  Params: {}", lowered.params.len());
    println!("  Input specs: {}", lowered.input_specs.len());

    // Print stable Inputs section (deterministic order, name + dtype only)
    if !lowered.input_specs.is_empty() {
        println!("\nInputs:");
        for spec in &lowered.input_specs {
            println!("  {}: {}", spec.name, spec.dtype);
        }
    }
    println!();

    // Create synthetic inputs for inference
    // The graph expects inputs in order: [input_tensors..., param_tensors...]
    // token_ids inputs are passed separately, not as tensor inputs

    // Build dimension bindings from consts
    let mut dim_bindings = ailang_core::DimBindings::new();
    for const_decl in &program.consts {
        dim_bindings.insert(const_decl.name.clone(), const_decl.value as usize);
    }
    // Also bind common dimensions
    dim_bindings.insert("B".to_string(), 2); // Default batch size

    let mut token_ids = Vec::new();
    let mut inputs = Vec::new();

    // Process inputs and params in the order they appear in the graph
    // Graph structure: inputs first, then params
    // But token_ids inputs should NOT be in the tensor inputs array

    // Find token_ids inputs (2D [B, T]) - these go in token_ids array, not inputs
    // For --run, we only need forward_output, so filter out labels
    // Note: This is for forward-only DSL files, so we use all input_specs but filter labels
    for spec in &lowered.input_specs {
        if spec.dtype == "token_ids" {
            // Extract dimensions from spec
            let batch_size = 2; // Default
            let seq_len = if spec.dims.len() >= 2 {
                // Try to resolve T dimension
                match &spec.dims[1] {
                    ailang_core::DimSpec::Named(name) => {
                        dim_bindings.get(name).copied().unwrap_or(6)
                    }
                    ailang_core::DimSpec::Literal(n) => *n,
                    ailang_core::DimSpec::Free => 6,
                }
            } else {
                6
            };
            let vocab_size = 20;
            let mut rng = ailang_core::SeededRng::new(seed + 1000);
            let mut tokens = Vec::new();
            for _ in 0..batch_size * seq_len {
                tokens.push((rng.gen() * vocab_size as f32) as usize);
            }
            token_ids.push(ailang_core::TokenIds::new(&[batch_size, seq_len], tokens));
        }
        // Skip labels for inference
    }

    // Also need to handle regular tensor inputs (non-token_ids, non-labels)
    // These should be added to the inputs array
    let mut tensor_inputs = Vec::new();
    for spec in &lowered.input_specs {
        // Regular tensor inputs: dtype is "tensor" (not "token_ids" or "labels")
        // Note: 2D inputs are classified as token_ids by lowering, but for --run we treat
        // non-embedding 2D inputs as regular tensors if they're not actually used as token_ids
        if spec.dtype == "tensor" || (spec.dtype == "token_ids" && spec.name != "tokens") {
            // Create synthetic tensor input based on shape
            let shape: Vec<usize> = spec
                .dims
                .iter()
                .map(|d| match d {
                    ailang_core::DimSpec::Named(name) => {
                        dim_bindings.get(name).copied().unwrap_or(2)
                    }
                    ailang_core::DimSpec::Literal(n) => *n,
                    ailang_core::DimSpec::Free => 2,
                })
                .collect();
            let num_elements: usize = shape.iter().product();
            let mut rng = ailang_core::SeededRng::new(seed + 2000);
            let mut data = Vec::new();
            for _ in 0..num_elements {
                data.push(rng.gen() * 2.0 - 1.0); // [-1, 1]
            }
            tensor_inputs.push(ailang_core::Tensor::from_vec(&shape, data));
        }
    }

    // Graph structure: inputs are ordered as:
    // 1. Regular tensor inputs (non-token_ids, non-labels) - indices 0..num_regular
    // 2. Token_ids inputs (empty tensors) - indices num_regular..num_regular+num_token_ids
    // 3. Params - indices num_regular+num_token_ids..

    // Count token_ids inputs that need empty tensor slots
    let num_token_inputs = lowered
        .input_specs
        .iter()
        .filter(|s| s.dtype == "token_ids" && s.name == "tokens")
        .count();

    // Build inputs array in graph order
    // 1. Regular tensor inputs
    inputs.extend(tensor_inputs);

    // 2. Empty tensors for token_ids inputs (they're passed via token_ids side-channel)
    for _ in 0..num_token_inputs {
        inputs.push(ailang_core::Tensor::zeros(&[]));
    }

    // 3. Params
    for (_, param) in &lowered.params {
        inputs.push(param.value.clone());
    }

    // Debug: verify input count matches graph expectation
    // The graph has input_count = tensor_inputs.len() + params.len()
    // (token_ids inputs are not in the tensor inputs array)
    // But token_ids inputs are not in the tensor inputs array
    // So we need to pad with empty tensors for token_ids inputs
    // Actually, looking at the graph structure, token_ids inputs still have Op::Input nodes
    // but they're not used as tensor inputs - they're passed via token_ids side-channel
    // So we need to provide empty tensors for those input slots

    // For now, let's just pass the params - the graph should handle token_ids separately
    // But wait, the graph.input_count includes both inputs and params
    // So we need to provide inputs for all input_count slots
    // Let me check the actual graph structure...

    // Actually, the issue is that Op::Input nodes for token_ids still expect tensor inputs
    // But we're not providing them. Let me check how the runtime handles this.

    // For MVP, let's just provide empty tensors for token_ids input slots
    // Exclude labels from count for inference
    let num_token_inputs = lowered
        .input_specs
        .iter()
        .filter(|s| s.dtype == "token_ids")
        .count();

    // Note: token_ids inputs are handled separately via token_ids array
    // The graph's input_count includes both tensor inputs and params
    // We've already added params and tensor inputs, so we're good

    // Run inference using selective evaluation (only requested return node)
    println!("Running inference...");
    let result = execute_forward_selected_with_capabilities(
        &lowered.graph,
        &inputs,
        &token_ids,
        &[],
        capabilities,
        &[lowered.forward_output],
    );

    match result {
        Ok(outputs) => {
            let output = &outputs[0]; // Selective eval returns only requested outputs
            println!("✓ Inference successful");
            println!("  Output shape: {:?}", output.shape());

            // Print first few values
            let num_print = output.num_elements().min(10);
            if num_print > 0 {
                let slice = output.data.as_slice().unwrap();
                print!("  First {} values: [", num_print);
                for (i, &val) in slice.iter().take(num_print).enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{:.4}", val);
                }
                println!("]");
            }

            // Export if requested
            if let Some(export_dir) = export_dir {
                let export_path = Path::new(export_dir);
                fs::create_dir_all(export_path).unwrap_or_else(|e| {
                    eprintln!("Failed to create export directory: {}", e);
                    std::process::exit(1);
                });

                println!("\nExporting model to: {}", export_dir);
                let params_refs: Vec<(usize, &Param)> = lowered
                    .params
                    .iter()
                    .map(|(id, param)| (*id, param))
                    .collect();
                export_model(
                    &lowered.graph,
                    &params_refs,
                    seed,
                    export_path,
                    Some(lowered.forward_output),
                    lowered.loss_output,
                )
                .map_err(|e| {
                    eprintln!("Export failed: {}", e);
                    std::process::exit(1);
                })
                .unwrap();
                println!("✓ Model exported successfully");
            }
        }
        Err(e) => {
            let diag = e.diagnostic();
            diagnostic_format::print_diagnostic(&diag);
            std::process::exit(1);
        }
    }
}

fn train_ail_file(
    file_path: &str,
    seed: u64,
    export_dir: Option<&str>,
    capabilities: &Capabilities,
) {
    use ailang_core::param::Param;
    use std::fs;
    use std::path::Path;

    println!("Training from: {}", file_path);
    let source = fs::read_to_string(file_path).unwrap_or_else(|e| {
        eprintln!("Failed to read file: {}", e);
        std::process::exit(1);
    });

    // Parse
    let mut lexer = Lexer::new(&source);
    let tokens = match lexer.tokenize() {
        Ok(tokens) => tokens,
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            std::process::exit(1);
        }
    };

    let mut parser = Parser::new(tokens);
    let program = match parser.parse() {
        Ok(program) => program,
        Err(diag) => {
            diagnostic_format::print_diagnostic(&diag);
            std::process::exit(1);
        }
    };

    // Validate: must have loss and train blocks
    if program.loss.is_none() {
        eprintln!("Error: Training requires a loss block");
        std::process::exit(1);
    }
    if program.train.is_none() {
        eprintln!("Error: Training requires a train block");
        std::process::exit(1);
    }

    // Lower
    let mut lowered = match lower(&program, seed) {
        Ok(lowered) => lowered,
        Err(diag) => {
            diagnostic_format::print_diagnostic(&diag);
            std::process::exit(1);
        }
    };

    let loss_output = lowered.loss_output.expect("Loss node must exist");
    let forward_output = lowered.forward_output;
    let train_block = program.train.as_ref().expect("Train block must exist");

    // Get training hyperparameters
    let steps = train_block.steps.unwrap_or(200) as usize;
    let lr = train_block.lr.unwrap_or(0.1);
    let batch_size = train_block.batch.unwrap_or(4) as usize;

    // Print stable Inputs section (deterministic order, name + dtype only)
    if !lowered.input_specs.is_empty() {
        println!("\nInputs:");
        for spec in &lowered.input_specs {
            println!("  {}: {}", spec.name, spec.dtype);
        }
    }

    // Find vocab size and num classes from consts
    let vocab_size = program
        .consts
        .iter()
        .find(|c| c.name == "V")
        .map(|c| c.value as usize)
        .unwrap_or(20);
    let num_classes = program
        .consts
        .iter()
        .find(|c| c.name == "C")
        .map(|c| c.value as usize)
        .unwrap_or(2);
    let seq_len = program
        .consts
        .iter()
        .find(|c| c.name == "T")
        .map(|c| c.value as usize)
        .unwrap_or(6);

    // Create inputs: token_ids and label_ids
    let mut token_ids = Vec::new();
    let mut inputs = Vec::new();

    // Add params as tensor inputs
    for (_, param) in &lowered.params {
        inputs.push(param.value.clone());
    }

    // Find label_ids index
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

    // Load dataset if present, otherwise generate synthetic data
    // Returns: (train_samples, validation_split)
    let (dataset_samples, validation_split): (
        Option<(Vec<Vec<usize>>, Vec<usize>)>,
        Option<(Vec<Vec<usize>>, Vec<usize>)>,
    ) = if let Some(ref dataset_config) = lowered.dataset {
        // Check capability BEFORE loading
        if !capabilities.has(&Capability::FileRead) {
            let diag = Diagnostic::new(
                "E_DATASET_CAPABILITY_DENIED",
                "Dataset loading requires FileRead capability".to_string(),
            )
            .with_field("capability".to_string(), "FileRead".to_string())
            .with_field(
                "attempted_action".to_string(),
                "read dataset file".to_string(),
            )
            .with_field("path".to_string(), dataset_config.path.clone());
            diagnostic_format::print_diagnostic(&diag);
            std::process::exit(1);
        }

        // Load dataset based on format
        let mut dataset = match dataset_config.format.as_str() {
            "tsv" => {
                match load_tsv_dataset(&dataset_config.path, seq_len, num_classes, capabilities) {
                    Ok(d) => d,
                    Err(diag) => {
                        diagnostic_format::print_diagnostic(&diag);
                        std::process::exit(1);
                    }
                }
            }
            "jsonl" => {
                match load_jsonl_dataset(&dataset_config.path, seq_len, num_classes, capabilities) {
                    Ok(d) => d,
                    Err(diag) => {
                        diagnostic_format::print_diagnostic(&diag);
                        std::process::exit(1);
                    }
                }
            }
            _ => {
                eprintln!("Unknown dataset format: {}", dataset_config.format);
                std::process::exit(1);
            }
        };

        // Apply shuffle if requested
        if dataset_config.shuffle.unwrap_or(false) {
            shuffle_dataset(&mut dataset, seed);
        }

        // Apply split if requested
        let (dataset_to_use, validation_split): (Dataset, Option<(Vec<Vec<usize>>, Vec<usize>)>) =
            if let Some(split_ratio) = dataset_config.split {
                match split_dataset(&dataset, split_ratio) {
                    Ok(split) => {
                        // For training, use train split; store validation for eval
                        (
                            split.train,
                            Some((
                                split.validation.samples_tokens,
                                split.validation.samples_labels,
                            )),
                        )
                    }
                    Err(diag) => {
                        diagnostic_format::print_diagnostic(&diag);
                        std::process::exit(1);
                    }
                }
            } else {
                (dataset, None::<(Vec<Vec<usize>>, Vec<usize>)>)
            };

        // Print stable Dataset section (deterministic format)
        println!("\nDataset:");
        println!("  format: {}", dataset_config.format);
        println!("  path: {}", dataset_config.path);
        println!("  train_count: {}", dataset_to_use.samples_tokens.len());
        if let Some((ref val_tokens, _)) = validation_split {
            println!("  val_count: {}", val_tokens.len());
        }

        (
            Some((dataset_to_use.samples_tokens, dataset_to_use.samples_labels)),
            validation_split,
        )
    } else {
        (
            None::<(Vec<Vec<usize>>, Vec<usize>)>,
            None::<(Vec<Vec<usize>>, Vec<usize>)>,
        )
    };

    // Function to get a batch (from dataset or synthetic)
    let mut rng = SeededRng::new(seed + 2000);

    // Print stable Train section (deterministic format)
    println!("\nTrain:");
    println!("  steps: {}", steps);
    println!("  lr: {}", lr);
    println!("  batch_size: {}", batch_size);

    // Print stable Eval section if present (deterministic format)
    if let Some(ref eval_config) = lowered.eval_config {
        println!("\nEval:");
        println!("  split: {}", eval_config.split);
        println!("  every: {}", eval_config.every);
        println!("  metrics: {:?}", eval_config.metrics);
    }

    // Prepend empty tensors for token_ids input slots
    let num_token_inputs = lowered
        .input_specs
        .iter()
        .filter(|s| s.dtype == "token_ids" || s.name == "labels")
        .count();
    for _ in 0..num_token_inputs {
        inputs.insert(0, Tensor::zeros(&[]));
    }

    // Training loop
    println!("\nStarting training...");

    // Store validation split for eval if available
    let eval_validation_split = validation_split.clone();

    for step in 0..steps {
        // Get batch for this step (from dataset or synthetic)
        let (batch_tokens, batch_labels) = if let Some((ref samples_tokens, ref samples_labels)) =
            dataset_samples
        {
            // Deterministic sampling from dataset
            let mut batch_tokens = Vec::new();
            let mut batch_labels = Vec::new();
            for _ in 0..batch_size {
                let idx = (rng.gen() * samples_tokens.len() as f32) as usize % samples_tokens.len();
                batch_tokens.extend_from_slice(&samples_tokens[idx]);
                batch_labels.push(samples_labels[idx]);
            }
            (batch_tokens, batch_labels)
        } else {
            // Synthetic data
            let mut tokens = Vec::new();
            for _ in 0..batch_size * seq_len {
                tokens.push((rng.gen() * vocab_size as f32) as usize);
            }
            let mut labels = Vec::new();
            for _ in 0..batch_size {
                labels.push((rng.gen() * num_classes as f32) as usize);
            }
            (tokens, labels)
        };

        // Create token_ids for this batch
        token_ids.clear();
        token_ids.push(TokenIds::new(&[batch_size, seq_len], batch_tokens));

        // Add labels if needed
        if let Some(idx) = label_idx {
            while token_ids.len() <= idx {
                token_ids.push(TokenIds::new(&[1], vec![0]));
            }
            token_ids[idx] = TokenIds::new(&[batch_size], batch_labels);
        }

        // Forward: compute loss
        let loss_result = execute_forward_selected_with_capabilities(
            &lowered.graph,
            &inputs,
            &token_ids,
            &[],
            capabilities,
            &[loss_output],
        );

        let loss_tensor = match loss_result {
            Ok(outputs) => outputs[0].clone(),
            Err(e) => {
                eprintln!("Loss computation error at step {}: {}", step, e);
                std::process::exit(1);
            }
        };

        let loss_value = loss_tensor.scalar();

        // Backward: compute gradients
        // First, run full forward to get activations
        let full_forward = execute_forward_with_capabilities(
            &lowered.graph,
            &inputs,
            &token_ids,
            &[],
            capabilities,
        )
        .unwrap();

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
        // The node_id corresponds to the input node index in the graph
        for (node_id, param) in &lowered.params {
            inputs[*node_id] = param.value.clone();
        }

        // Print loss every 10 steps
        if step % 10 == 0 || step == steps - 1 {
            println!("Step {}: loss = {:.6}", step, loss_value);
        }

        // Run evaluation if configured
        if let Some(ref eval_config) = lowered.eval_config {
            if step % eval_config.every == 0 || step == steps - 1 {
                // Determine which split to use for eval
                let eval_samples = match eval_config.split.as_str() {
                    "val" => eval_validation_split.as_ref(),
                    "train" => dataset_samples.as_ref(),
                    _ => None,
                };

                if let Some((ref eval_tokens, ref eval_labels)) = eval_samples {
                    // Compute metrics on eval split
                    let mut eval_loss = None;
                    let mut eval_accuracy = None;

                    // Get eval batch (use same batch size as training)
                    let mut eval_rng = SeededRng::new(seed + 3000 + step as u64);
                    let mut eval_batch_tokens = Vec::new();
                    let mut eval_batch_labels = Vec::new();
                    for _ in 0..batch_size {
                        let idx = (eval_rng.gen() * eval_tokens.len() as f32) as usize
                            % eval_tokens.len();
                        eval_batch_tokens.extend_from_slice(&eval_tokens[idx]);
                        eval_batch_labels.push(eval_labels[idx]);
                    }

                    // Create token_ids for eval batch
                    let mut eval_token_ids = Vec::new();
                    eval_token_ids.push(TokenIds::new(&[batch_size, seq_len], eval_batch_tokens));
                    if let Some(idx) = label_idx {
                        while eval_token_ids.len() <= idx {
                            eval_token_ids.push(TokenIds::new(&[1], vec![0]));
                        }
                        eval_token_ids[idx] =
                            TokenIds::new(&[batch_size], eval_batch_labels.clone());
                    }

                    // Determine which nodes to compute
                    let mut requested_nodes = Vec::new();
                    if eval_config.metrics.contains(&"loss".to_string()) {
                        requested_nodes.push(loss_output);
                    }
                    if eval_config.metrics.contains(&"accuracy".to_string()) {
                        requested_nodes.push(forward_output);
                    }

                    if !requested_nodes.is_empty() {
                        // Forward-only execution for eval
                        let eval_result = execute_forward_selected_with_capabilities(
                            &lowered.graph,
                            &inputs,
                            &eval_token_ids,
                            &[],
                            capabilities,
                            &requested_nodes,
                        );

                        match eval_result {
                            Ok(eval_outputs) => {
                                // Compute loss metric
                                if eval_config.metrics.contains(&"loss".to_string()) {
                                    let loss_idx = requested_nodes
                                        .iter()
                                        .position(|&n| n == loss_output)
                                        .unwrap();
                                    let loss_val =
                                        eval_outputs[loss_idx].data.as_slice().unwrap()[0];
                                    eval_loss = Some(loss_val);
                                }

                                // Compute accuracy metric
                                if eval_config.metrics.contains(&"accuracy".to_string()) {
                                    let logits_idx = requested_nodes
                                        .iter()
                                        .position(|&n| n == forward_output)
                                        .unwrap();
                                    let logits = &eval_outputs[logits_idx];

                                    // logits is [B, C], labels is [B]
                                    // argmax(logits, axis=1) == labels
                                    let logits_shape = logits.shape();
                                    let batch_size_eval = logits_shape[0];
                                    let num_classes = logits_shape[1];
                                    let logits_data = logits.data.as_slice().unwrap();

                                    let mut correct = 0;
                                    for i in 0..batch_size_eval {
                                        let mut max_idx = 0;
                                        let mut max_val = logits_data[i * num_classes + 0];
                                        for j in 1..num_classes {
                                            let val = logits_data[i * num_classes + j];
                                            if val > max_val {
                                                max_val = val;
                                                max_idx = j;
                                            }
                                        }
                                        if max_idx == eval_batch_labels[i] {
                                            correct += 1;
                                        }
                                    }
                                    eval_accuracy = Some(correct as f32 / batch_size_eval as f32);
                                }

                                // Print eval metrics (stable format: eval/ prefix)
                                if let Some(loss) = eval_loss {
                                    println!("eval/loss = {:.6}", loss);
                                }
                                if let Some(acc) = eval_accuracy {
                                    println!("eval/accuracy = {:.4}", acc);
                                }
                            }
                            Err(e) => {
                                eprintln!("Eval error at step {}: {}", step, e);
                            }
                        }
                    }
                }
            }
        }
    }

    println!("\n✓ Training completed");

    // Export if requested
    if let Some(export_dir) = export_dir {
        let export_path = Path::new(export_dir);
        fs::create_dir_all(export_path).unwrap_or_else(|e| {
            eprintln!("Failed to create export directory: {}", e);
            std::process::exit(1);
        });

        println!("\nExporting trained model to: {}", export_dir);
        let params_refs: Vec<(usize, &Param)> = lowered
            .params
            .iter()
            .map(|(node_id, param)| (*node_id, param))
            .collect();
        match export_model(
            &lowered.graph,
            &params_refs,
            seed,
            export_path,
            Some(lowered.forward_output),
            lowered.loss_output,
        ) {
            Ok(_) => println!("✓ Model exported successfully"),
            Err(e) => {
                eprintln!("Export error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse all flags first, then execute command
    let mut seed = 42u64; // Default seed
    let mut export_path: Option<String> = None;
    let mut infer_path: Option<String> = None;
    let mut read_file_path: Option<String> = None;
    let mut run_file_path: Option<String> = None;
    let mut allowed_capabilities: Vec<String> = Vec::new();
    let mut command: Option<&str> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                if i + 1 < args.len() {
                    seed = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid seed value: {}", args[i + 1]);
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("--seed requires a value");
                    std::process::exit(1);
                }
            }
            "--export" => {
                if i + 1 < args.len() {
                    export_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--export requires a path");
                    std::process::exit(1);
                }
            }
            "--infer" => {
                if i + 1 < args.len() {
                    infer_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--infer requires a path");
                    std::process::exit(1);
                }
            }
            "--allow" => {
                if i + 1 < args.len() {
                    allowed_capabilities.push(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("--allow requires a capability name");
                    std::process::exit(1);
                }
            }
            "--read-file-demo" => {
                if i + 1 < args.len() {
                    read_file_path = Some(args[i + 1].clone());
                    command = Some("--read-file-demo");
                    i += 2;
                } else {
                    eprintln!("--read-file-demo requires a file path");
                    std::process::exit(1);
                }
            }
            "--run" => {
                if i + 1 < args.len() {
                    run_file_path = Some(args[i + 1].clone());
                    command = Some("--run");
                    i += 2;
                } else {
                    eprintln!("--run requires a file path");
                    std::process::exit(1);
                }
            }
            "--train" => {
                if i + 1 < args.len() {
                    run_file_path = Some(args[i + 1].clone());
                    command = Some("--train");
                    i += 2;
                } else {
                    eprintln!("--train requires a file path");
                    std::process::exit(1);
                }
            }
            "--selective-demo" => {
                command = Some("--selective-demo");
                i += 1;
            }
            "--gradcheck"
            | "--attention-gradcheck"
            | "--mlp"
            | "--toy-classifier"
            | "--attention-demo"
            | "--now-demo" => {
                command = Some(args[i].as_str());
                i += 1;
            }
            _ => {
                eprintln!("Unknown flag: {}", args[i]);
                eprintln!("Usage: {} [--seed <u64>] [--export <path>] [--infer <path>] [--allow <capability>] [--gradcheck|--attention-gradcheck|--mlp|--toy-classifier|--attention-demo|--now-demo|--selective-demo]", args[0]);
                std::process::exit(1);
            }
        }
    }

    // Build capabilities from --allow flags
    let mut capabilities = Capabilities::new();
    for cap_name in &allowed_capabilities {
        match cap_name.as_str() {
            "clock" => {
                capabilities = capabilities.with(Capability::Clock);
            }
            "fileread" => {
                capabilities = capabilities.with(Capability::FileRead);
            }
            "filewrite" => {
                capabilities = capabilities.with(Capability::FileWrite);
            }
            "network" => {
                capabilities = capabilities.with(Capability::Network);
            }
            "env" => {
                capabilities = capabilities.with(Capability::Env);
            }
            "process" => {
                capabilities = capabilities.with(Capability::Process);
            }
            _ => {
                eprintln!("Unknown capability: {}", cap_name);
                eprintln!("Valid capabilities: clock, fileread, filewrite, network, env, process");
                std::process::exit(1);
            }
        }
    }

    // Handle inference mode
    if let Some(path_str) = infer_path {
        if let Err(_e) = inference_demo(Path::new(&path_str)) {
            // Error already printed as diagnostic
            std::process::exit(1);
        }
        return;
    }

    // Execute command
    match command {
        Some("--gradcheck") => gradcheck_demo(),
        Some("--attention-gradcheck") => {
            let passed = attention_gradcheck();
            if !passed {
                std::process::exit(1);
            }
        }
        Some("--mlp") => training_demo(seed),
        Some("--toy-classifier") => toy_classifier_demo(seed),
        Some("--attention-demo") => {
            attention_demo(seed, export_path.as_ref().map(|s| Path::new(s)));
        }
        Some("--now-demo") => {
            now_demo(&capabilities);
        }
        Some("--read-file-demo") => {
            if let Some(path) = read_file_path {
                read_file_demo(&path, &capabilities);
            } else {
                eprintln!("--read-file-demo requires a file path");
                std::process::exit(1);
            }
        }
        Some("--selective-demo") => {
            selective_demo(&capabilities);
        }
        Some("--run") => {
            if let Some(path) = run_file_path {
                run_ail_file(&path, seed, export_path.as_deref(), &capabilities);
            } else {
                eprintln!("--run requires a file path");
                std::process::exit(1);
            }
        }
        Some("--train") => {
            if let Some(path) = run_file_path {
                train_ail_file(&path, seed, export_path.as_deref(), &capabilities);
            } else {
                eprintln!("--train requires a file path");
                std::process::exit(1);
            }
        }
        None => {
            // Default to MLP demo if no command specified
            training_demo(seed);
        }
        _ => unreachable!(),
    }
}

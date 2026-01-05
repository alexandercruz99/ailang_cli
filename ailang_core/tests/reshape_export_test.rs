use ailang_core::{
    forward::execute_forward,
    ir::{Graph, Op},
    model::{export_model, load_model},
    param::Param,
    tensor::{Tensor, TokenIds},
    ReshapeDim, ReshapeSpec,
};
use std::fs;

#[test]
fn test_attention_export_load_inference() {
    // Build a tiny attention graph
    const B: usize = 2;
    const T: usize = 3;
    const D: usize = 4;
    const VOCAB: usize = 5;
    const NUM_CLASSES: usize = 2;

    // Create parameters
    let embed_weight = Param::new(Tensor::from_vec(
        &[VOCAB, D],
        (0..VOCAB * D).map(|i| i as f32 * 0.1).collect(),
    ));
    let wq = Param::new(Tensor::from_vec(
        &[D, D],
        (0..D * D).map(|i| i as f32 * 0.1).collect(),
    ));
    let wk = Param::new(Tensor::from_vec(
        &[D, D],
        (0..D * D).map(|i| i as f32 * 0.1).collect(),
    ));
    let wv = Param::new(Tensor::from_vec(
        &[D, D],
        (0..D * D).map(|i| i as f32 * 0.1).collect(),
    ));
    let out_weight = Param::new(Tensor::from_vec(
        &[D, NUM_CLASSES],
        (0..D * NUM_CLASSES).map(|i| i as f32 * 0.1).collect(),
    ));

    // Build graph
    let mut graph = Graph::new_with_token_ids(5, 2);
    let embed_w_id = graph.input_node(0);
    let wq_id = graph.input_node(1);
    let wk_id = graph.input_node(2);
    let wv_id = graph.input_node(3);
    let out_w_id = graph.input_node(4);

    let embed_id = graph.add_node(Op::Embedding(0), vec![embed_w_id]);

    // Reshape: [B, T, D] -> [B*T, D]
    let embed_reshaped_q = graph.add_node(
        Op::Reshape(ReshapeSpec::new(vec![
            ReshapeDim::Mul(Box::new(ReshapeDim::Ref(0)), Box::new(ReshapeDim::Ref(1))),
            ReshapeDim::Ref(2),
        ])),
        vec![embed_id],
    );
    let q_reshaped = graph.add_node(Op::MatMul2D, vec![embed_reshaped_q, wq_id]);
    // Reshape back: [B*T, D] -> [B, T, D] using -1, T, D
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

    // Test 1: Run inference on original model
    let token_ids_input = TokenIds::new(&[B, T], vec![0, 1, 2, 1, 2, 0]);
    let token_ids_target = TokenIds::new(&[B], vec![0, 1]);

    let inputs = vec![
        embed_weight.value.clone(),
        wq.value.clone(),
        wk.value.clone(),
        wv.value.clone(),
        out_weight.value.clone(),
    ];
    let token_ids = vec![token_ids_input.clone(), token_ids_target.clone()];
    let activations_orig = execute_forward(&graph, &inputs, &token_ids).unwrap();
    let output_orig = &activations_orig[logits_id];
    let loss_orig = activations_orig[loss_id].scalar();

    // Export model
    let temp_dir = std::env::temp_dir().join("ailang_reshape_test");
    fs::create_dir_all(&temp_dir).unwrap();

    let params = vec![
        (embed_w_id, &embed_weight),
        (wq_id, &wq),
        (wk_id, &wk),
        (wv_id, &wv),
        (out_w_id, &out_weight),
    ];
    export_model(&graph, &params, 42, &temp_dir, Some(logits_id), None).unwrap();

    // Load model
    let (loaded_graph, loaded_weights, _loaded_seed, _, _) = load_model(&temp_dir).unwrap();

    // Test 2: Run inference on loaded model with same input
    let inputs_loaded = loaded_weights
        .iter()
        .map(|(_, t)| t.clone())
        .collect::<Vec<_>>();
    let activations_loaded = execute_forward(&loaded_graph, &inputs_loaded, &token_ids).unwrap();
    let output_loaded = &activations_loaded[logits_id];
    let loss_loaded = activations_loaded[loss_id].scalar();

    // Verify outputs match
    assert_eq!(output_orig.shape(), output_loaded.shape());
    let orig_vec: Vec<f32> = output_orig.data.iter().copied().collect();
    let loaded_vec: Vec<f32> = output_loaded.data.iter().copied().collect();
    assert_eq!(orig_vec.len(), loaded_vec.len());
    for (i, (&o, &l)) in orig_vec.iter().zip(loaded_vec.iter()).enumerate() {
        assert!(
            (o - l).abs() < 1e-5,
            "Output mismatch at index {}: {} vs {}",
            i,
            o,
            l
        );
    }
    assert!(
        (loss_orig - loss_loaded).abs() < 1e-5,
        "Loss mismatch: {} vs {}",
        loss_orig,
        loss_loaded
    );

    // Test 3: Run inference with different batch size B=3 (same T, D)
    const B2: usize = 3;
    let token_ids_input2 = TokenIds::new(&[B2, T], vec![0, 1, 2, 1, 2, 0, 2, 0, 1]);
    let token_ids_target2 = TokenIds::new(&[B2], vec![0, 1, 0]);
    let token_ids2 = vec![token_ids_input2, token_ids_target2];

    // This should work without panicking
    let activations_loaded2 = execute_forward(&loaded_graph, &inputs_loaded, &token_ids2).unwrap();
    let output_loaded2 = &activations_loaded2[logits_id];

    // Verify output shape is correct: [B2, NUM_CLASSES]
    assert_eq!(output_loaded2.shape(), &[B2, NUM_CLASSES]);
    assert_eq!(output_loaded2.num_elements(), B2 * NUM_CLASSES);

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}

use ailang_core::{
    forward::execute_forward,
    ir::{Graph, Op},
    model::{export_model, load_model},
    param::Param,
    tensor::Tensor,
};
use std::fs;

#[test]
fn test_model_export_load() {
    // Create a simple graph: MatMul2D
    let mut graph = Graph::new(2);
    let a_id = graph.input_node(0);
    let b_id = graph.input_node(1);
    let matmul_id = graph.add_node(Op::MatMul2D, vec![a_id, b_id]);

    // Create parameters
    let a = Param::new(Tensor::from_vec(
        &[2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = Param::new(Tensor::from_vec(
        &[3, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    // Export model
    let temp_dir = std::env::temp_dir().join("ailang_test_model");
    fs::create_dir_all(&temp_dir).unwrap();

    let params = vec![(a_id, &a), (b_id, &b)];
    export_model(&graph, &params, 42, &temp_dir, Some(matmul_id), None).unwrap();

    // Load model
    let (loaded_graph, loaded_weights, loaded_seed, _, _) = load_model(&temp_dir).unwrap();

    assert_eq!(loaded_seed, 42);
    assert_eq!(loaded_graph.nodes.len(), graph.nodes.len());
    assert_eq!(loaded_weights.len(), 2);

    // Verify weights match
    let loaded_a = &loaded_weights[0].1;
    let loaded_b = &loaded_weights[1].1;

    assert_eq!(loaded_a.shape(), a.value.shape());
    assert_eq!(loaded_b.shape(), b.value.shape());

    // Run inference with original and loaded model
    let inputs_orig = vec![a.value.clone(), b.value.clone()];
    let activations_orig = execute_forward(&graph, &inputs_orig, &[]).unwrap();
    let output_orig = &activations_orig[matmul_id];

    let inputs_loaded = vec![loaded_a.clone(), loaded_b.clone()];
    let activations_loaded = execute_forward(&loaded_graph, &inputs_loaded, &[]).unwrap();
    // Find the matmul node in loaded graph (should be at same index)
    let output_loaded = &activations_loaded[matmul_id];

    // Compare outputs
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

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}

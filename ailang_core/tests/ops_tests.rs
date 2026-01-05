use ailang_core::{
    forward::execute_forward,
    ir::{Graph, Op},
    tensor::{Tensor, TokenIds},
};

#[test]
fn test_softmax_sums_to_one() {
    let mut graph = Graph::new(1);
    let x_id = graph.input_node(0);
    let softmax_id = graph.add_node(Op::Softmax(1), vec![x_id]); // axis=1 (last dim)

    let x = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let inputs = vec![x];
    let activations = execute_forward(&graph, &inputs, &[]).unwrap();
    let softmax_output = &activations[softmax_id];

    // Check that each row sums to ~1.0
    let softmax_2d = softmax_output
        .data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for row in softmax_2d.rows() {
        let sum: f32 = row.sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Softmax row sum should be ~1.0, got {}",
            sum
        );
    }
}

#[test]
fn test_crossentropy_simple() {
    // Simple test: logits [2, 3], target_ids [2] = [1, 0]
    let mut graph = Graph::new_with_token_ids(1, 1);
    let logits_id = graph.input_node(0);
    let target_ids = TokenIds::new(&[2], vec![1, 0]);
    let loss_id = graph.add_node(Op::CrossEntropy(0), vec![logits_id]);

    // Logits: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    // Targets: [1, 0] -> should pick logits[0][1]=2.0 and logits[1][0]=4.0
    let logits = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let inputs = vec![logits];
    let activations = execute_forward(&graph, &inputs, &[target_ids]).unwrap();
    let loss = activations[loss_id].scalar();

    // Loss should be positive
    assert!(loss > 0.0, "CrossEntropy loss should be positive");
    assert!(loss < 10.0, "CrossEntropy loss should be reasonable");
}

#[test]
fn test_embedding_simple() {
    let mut graph = Graph::new_with_token_ids(1, 1);
    let weight_id = graph.input_node(0);
    let token_ids = TokenIds::new(&[2], vec![0, 1]);
    let embed_id = graph.add_node(Op::Embedding(0), vec![weight_id]);

    // Weight: vocab=3, dim=2
    let weight = Tensor::from_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let inputs = vec![weight];
    let activations = execute_forward(&graph, &inputs, &[token_ids]).unwrap();
    let embed_output = &activations[embed_id];

    // Output should be [2, 2]: [[1.0, 2.0], [3.0, 4.0]]
    assert_eq!(embed_output.shape(), &[2, 2]);
    let embed_2d = embed_output
        .data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    assert_eq!(embed_2d[(0, 0)], 1.0);
    assert_eq!(embed_2d[(0, 1)], 2.0);
    assert_eq!(embed_2d[(1, 0)], 3.0);
    assert_eq!(embed_2d[(1, 1)], 4.0);
}

#[test]
fn test_mean_pool_time() {
    let mut graph = Graph::new(1);
    let x_id = graph.input_node(0);
    let pool_id = graph.add_node(Op::MeanPoolTime, vec![x_id]);

    // Input: [2, 3, 4] -> Output: [2, 4]
    let x = Tensor::from_vec(
        &[2, 3, 4],
        vec![
            // Batch 0, Time 0
            1.0, 2.0, 3.0, 4.0, // Batch 0, Time 1
            5.0, 6.0, 7.0, 8.0, // Batch 0, Time 2
            9.0, 10.0, 11.0, 12.0, // Batch 1, Time 0
            13.0, 14.0, 15.0, 16.0, // Batch 1, Time 1
            17.0, 18.0, 19.0, 20.0, // Batch 1, Time 2
            21.0, 22.0, 23.0, 24.0,
        ],
    );
    let inputs = vec![x];
    let activations = execute_forward(&graph, &inputs, &[]).unwrap();
    let pool_output = &activations[pool_id];

    assert_eq!(pool_output.shape(), &[2, 4]);
    let pool_2d = pool_output
        .data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    // Batch 0, dim 0: (1.0 + 5.0 + 9.0) / 3 = 5.0
    assert!((pool_2d[(0, 0)] - 5.0).abs() < 1e-5);
    // Batch 0, dim 1: (2.0 + 6.0 + 10.0) / 3 = 6.0
    assert!((pool_2d[(0, 1)] - 6.0).abs() < 1e-5);
}

#[test]
fn test_transpose3d() {
    let mut graph = Graph::new(1);
    let x_id = graph.input_node(0);
    let transpose_id = graph.add_node(Op::Transpose3D, vec![x_id]);

    // Input: [2, 3, 4] -> Output: [2, 4, 3]
    let x = Tensor::from_vec(
        &[2, 3, 4],
        vec![
            // Batch 0, Time 0
            1.0, 2.0, 3.0, 4.0, // Batch 0, Time 1
            5.0, 6.0, 7.0, 8.0, // Batch 0, Time 2
            9.0, 10.0, 11.0, 12.0, // Batch 1, Time 0
            13.0, 14.0, 15.0, 16.0, // Batch 1, Time 1
            17.0, 18.0, 19.0, 20.0, // Batch 1, Time 2
            21.0, 22.0, 23.0, 24.0,
        ],
    );
    let inputs = vec![x];
    let activations = execute_forward(&graph, &inputs, &[]).unwrap();
    let transpose_output = &activations[transpose_id];

    assert_eq!(transpose_output.shape(), &[2, 4, 3]);
    let trans_3d = transpose_output
        .data
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .unwrap();
    // Original [0,0,0]=1.0 should be at [0,0,0] after transpose
    assert_eq!(trans_3d[(0, 0, 0)], 1.0);
    // Original [0,0,1]=2.0 should be at [0,1,0] after transpose
    assert_eq!(trans_3d[(0, 1, 0)], 2.0);
    // Original [0,1,0]=5.0 should be at [0,0,1] after transpose
    assert_eq!(trans_3d[(0, 0, 1)], 5.0);
}

#[test]
fn test_batch_matmul() {
    let mut graph = Graph::new(2);
    let a_id = graph.input_node(0);
    let b_id = graph.input_node(1);
    let bmm_id = graph.add_node(Op::BatchMatMul, vec![a_id, b_id]);

    // Input A: [B=2, M=3, K=4]
    // Input B: [B=2, K=4, N=2]
    // Output: [B=2, M=3, N=2]
    let a = Tensor::from_vec(
        &[2, 3, 4],
        vec![
            // Batch 0: [3, 4]
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            // Batch 1: [3, 4]
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
    );
    let b = Tensor::from_vec(
        &[2, 4, 2],
        vec![
            // Batch 0: [4, 2]
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Batch 1: [4, 2]
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );

    let inputs = vec![a, b];
    let activations = execute_forward(&graph, &inputs, &[]).unwrap();
    let output = &activations[bmm_id];

    // Expected output shape: [2, 3, 2]
    assert_eq!(output.shape(), &[2, 3, 2]);

    // Batch 0: [3, 4] @ [4, 2] = [3, 2]
    // Row 0: [1,2,3,4] @ [[1,2],[3,4],[5,6],[7,8]] = [50, 60]
    // Row 1: [5,6,7,8] @ [[1,2],[3,4],[5,6],[7,8]] = [114, 140]
    // Row 2: [9,10,11,12] @ [[1,2],[3,4],[5,6],[7,8]] = [178, 220]
    let output_3d = output
        .data
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .unwrap();
    assert!((output_3d[(0, 0, 0)] - 50.0).abs() < 1e-5);
    assert!((output_3d[(0, 0, 1)] - 60.0).abs() < 1e-5);
    assert!((output_3d[(0, 1, 0)] - 114.0).abs() < 1e-5);
    assert!((output_3d[(0, 1, 1)] - 140.0).abs() < 1e-5);
}

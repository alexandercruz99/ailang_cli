use ailang_core::{
    backward::execute_backward,
    execute_kernel::execute_op_forward,
    ir::{Graph, Op},
    tensor::{Tensor, TokenIds},
    Capabilities,
};

#[test]
fn test_concat_forward_backward() {
    // Create a simple graph: concat(a, b) where a=[2,3], b=[2,4]
    let mut graph = Graph::new(2);
    let a_id = graph.input_node(0);
    let b_id = graph.input_node(1);
    let concat_id = graph.add_node(Op::Concat { axis: 1 }, vec![a_id, b_id]);

    // Create inputs
    let a = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_vec(&[2, 4], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]);
    let inputs = vec![a.clone(), b.clone()];

    // Forward
    let result = execute_op_forward(
        &Op::Concat { axis: 1 },
        &[a.clone(), b.clone()],
        &inputs,
        &[],
        &[],
        &Capabilities::empty(),
        None,
        #[cfg(feature = "gpu-wgpu")]
        None::<&crate::device::GpuDevice>,
        #[cfg(not(feature = "gpu-wgpu"))]
        None,
    )
    .unwrap();

    // Check shape: [2, 7]
    assert_eq!(result.shape(), &[2, 7]);
    // Check values: first row should be [1,2,3,7,8,9,10]
    assert_eq!(result.data.as_slice().unwrap()[0], 1.0);
    assert_eq!(result.data.as_slice().unwrap()[3], 7.0);
    assert_eq!(result.data.as_slice().unwrap()[6], 10.0);

    // Backward: add a Sum node to reduce to scalar
    let sum_id = graph.add_node(Op::Sum, vec![concat_id]);
    let sum_result = execute_op_forward(
        &Op::Sum,
        &[result.clone()],
        &inputs,
        &[],
        &[],
        &Capabilities::empty(),
        None,
        #[cfg(feature = "gpu-wgpu")]
        None::<&crate::device::GpuDevice>,
        #[cfg(not(feature = "gpu-wgpu"))]
        None,
    )
    .unwrap();

    let activations = vec![a.clone(), b.clone(), result.clone(), sum_result.clone()];
    let grads = execute_backward(&graph, &activations, sum_id, &[]);

    // Check gradients
    assert!(grads[a_id].is_some());
    assert!(grads[b_id].is_some());

    let grad_a = grads[a_id].as_ref().unwrap();
    let grad_b = grads[b_id].as_ref().unwrap();

    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[2, 4]);
}

#[test]
fn test_slice_rows_forward_backward() {
    // Create graph: slice_rows(x, 1, 2)
    let mut graph = Graph::new(1);
    let x_id = graph.input_node(0);
    let slice_id = graph.add_node(Op::SliceRows { start: 1, len: 2 }, vec![x_id]);

    // Create input: [4, 3]
    let x = Tensor::from_vec(
        &[4, 3],
        vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ],
    );
    let inputs = vec![x.clone()];

    // Forward
    let result = execute_op_forward(
        &Op::SliceRows { start: 1, len: 2 },
        &[x.clone()],
        &inputs,
        &[],
        &[],
        &Capabilities::empty(),
        None,
        #[cfg(feature = "gpu-wgpu")]
        None::<&crate::device::GpuDevice>,
        #[cfg(not(feature = "gpu-wgpu"))]
        None,
    )
    .unwrap();

    // Check shape: [2, 3]
    assert_eq!(result.shape(), &[2, 3]);
    // Check values: should be rows 1 and 2
    assert_eq!(result.data.as_slice().unwrap()[0], 4.0);
    assert_eq!(result.data.as_slice().unwrap()[3], 7.0);

    // Backward: add Sum to reduce to scalar
    let sum_id = graph.add_node(Op::Sum, vec![slice_id]);
    let sum_result = execute_op_forward(
        &Op::Sum,
        &[result.clone()],
        &inputs,
        &[],
        &[],
        &Capabilities::empty(),
        None,
        #[cfg(feature = "gpu-wgpu")]
        None::<&crate::device::GpuDevice>,
        #[cfg(not(feature = "gpu-wgpu"))]
        None,
    )
    .unwrap();

    let activations = vec![x.clone(), result.clone(), sum_result.clone()];
    let grads = execute_backward(&graph, &activations, sum_id, &[]);

    assert!(grads[x_id].is_some());
    let grad_x = grads[x_id].as_ref().unwrap();
    assert_eq!(grad_x.shape(), &[4, 3]);
    // Gradients should be zero for rows 0 and 3, non-zero for rows 1 and 2
    assert_eq!(grad_x.data.as_slice().unwrap()[0], 0.0); // row 0
    assert_ne!(grad_x.data.as_slice().unwrap()[3], 0.0); // row 1
}

#[test]
fn test_gather_rows_repeated_indices() {
    // Create graph: gather_rows(x, indices)
    let mut graph = Graph::new(1);
    let x_id = graph.input_node(0);
    let gather_id = graph.add_node(Op::GatherRows, vec![x_id]);

    // Create input: [4, 3]
    let x = Tensor::from_vec(
        &[4, 3],
        vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
            10.0, 11.0, 12.0, // row 3
        ],
    );
    let inputs = vec![x.clone()];

    // Create indices: [1, 0, 1, 2] (with repetition)
    let indices = TokenIds::new(&[4], vec![1, 0, 1, 2]);
    let token_ids = vec![indices.clone()];

    // Forward
    let result = execute_op_forward(
        &Op::GatherRows,
        &[x.clone()],
        &inputs,
        &token_ids,
        &[],
        &Capabilities::empty(),
        None,
        #[cfg(feature = "gpu-wgpu")]
        None::<&crate::device::GpuDevice>,
        #[cfg(not(feature = "gpu-wgpu"))]
        None,
    )
    .unwrap();

    // Check shape: [4, 3]
    assert_eq!(result.shape(), &[4, 3]);
    // Check values: row 0 should be x[1], row 1 should be x[0], row 2 should be x[1], row 3 should be x[2]
    assert_eq!(result.data.as_slice().unwrap()[0], 4.0); // x[1][0]
    assert_eq!(result.data.as_slice().unwrap()[3], 1.0); // x[0][0]
    assert_eq!(result.data.as_slice().unwrap()[6], 4.0); // x[1][0] (repeated)
    assert_eq!(result.data.as_slice().unwrap()[9], 7.0); // x[2][0]

    // Backward: add Sum to reduce to scalar
    let sum_id = graph.add_node(Op::Sum, vec![gather_id]);
    let sum_result = execute_op_forward(
        &Op::Sum,
        &[result.clone()],
        &inputs,
        &token_ids,
        &[],
        &Capabilities::empty(),
        None,
        #[cfg(feature = "gpu-wgpu")]
        None::<&crate::device::GpuDevice>,
        #[cfg(not(feature = "gpu-wgpu"))]
        None,
    )
    .unwrap();

    let activations = vec![x.clone(), result.clone(), sum_result.clone()];
    let grads = execute_backward(&graph, &activations, sum_id, &token_ids);

    assert!(grads[x_id].is_some());
    let grad_x = grads[x_id].as_ref().unwrap();
    assert_eq!(grad_x.shape(), &[4, 3]);
    // Row 1 should have accumulated gradients from both gather operations (indices 0 and 2 both point to row 1)
    // Row 0 should have gradient from one gather (index 1)
    // Row 2 should have gradient from one gather (index 3)
    // Row 3 should have zero gradient
    assert_ne!(grad_x.data.as_slice().unwrap()[3], 0.0); // row 1 (accumulated from indices 0 and 2)
    assert_ne!(grad_x.data.as_slice().unwrap()[0], 0.0); // row 0 (from index 1)
    assert_ne!(grad_x.data.as_slice().unwrap()[6], 0.0); // row 2 (from index 3)
    assert_eq!(grad_x.data.as_slice().unwrap()[9], 0.0); // row 3 (not gathered)
}

#[test]
fn test_dropout_determinism() {
    // Create input
    let x = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let inputs = vec![x.clone()];

    // Run dropout twice with same input
    let result1 = execute_op_forward(
        &Op::Dropout { p: 0.5 },
        &[x.clone()],
        &inputs,
        &[],
        &[],
        &Capabilities::empty(),
        None,
        #[cfg(feature = "gpu-wgpu")]
        None::<&crate::device::GpuDevice>,
        #[cfg(not(feature = "gpu-wgpu"))]
        None,
    )
    .unwrap();

    let result2 = execute_op_forward(
        &Op::Dropout { p: 0.5 },
        &[x.clone()],
        &inputs,
        &[],
        &[],
        &Capabilities::empty(),
        None,
        #[cfg(feature = "gpu-wgpu")]
        None::<&crate::device::GpuDevice>,
        #[cfg(not(feature = "gpu-wgpu"))]
        None,
    )
    .unwrap();

    // Results should be identical (deterministic)
    assert_eq!(result1.data, result2.data);
}

#[test]
fn test_dropout_inference_identity() {
    // Note: For MVP, dropout always applies (no inference mode flag)
    // This test verifies that dropout scales correctly
    let x = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let inputs = vec![x.clone()];

    let result = execute_op_forward(
        &Op::Dropout { p: 0.0 }, // p=0 means no dropout
        &[x.clone()],
        &inputs,
        &[],
        &[],
        &Capabilities::empty(),
        None,
        #[cfg(feature = "gpu-wgpu")]
        None::<&crate::device::GpuDevice>,
        #[cfg(not(feature = "gpu-wgpu"))]
        None,
    )
    .unwrap();

    // With p=0, scale = 1/(1-0) = 1, so output should equal input
    assert_eq!(result.data, x.data);
}

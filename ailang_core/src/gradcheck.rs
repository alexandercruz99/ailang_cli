use crate::backward::execute_backward;
use crate::forward::execute_forward;
use crate::ir::{Graph, Op};
use crate::tensor::Tensor;

pub fn gradcheck_matmul_relu_sum() -> bool {
    const EPS: f32 = 1e-4;
    const TOL: f32 = 5e-3; // Slightly more lenient for finite difference errors

    // Build graph: X @ W -> ReLU -> Sum
    // X: (2, 3), W: (3, 4)
    let mut graph = Graph::new(2);
    let x_id = graph.input_node(0);
    let w_id = graph.input_node(1);

    let matmul_id = graph.add_node(Op::MatMul2D, vec![x_id, w_id]);
    let relu_id = graph.add_node(Op::ReLU, vec![matmul_id]);
    let sum_id = graph.add_node(Op::Sum, vec![relu_id]);

    // Create small test inputs
    let x = Tensor::from_vec(&[2, 3], vec![0.5, 1.0, 0.3, 0.8, 0.2, 0.9]);
    let w = Tensor::from_vec(
        &[3, 4],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2],
    );

    // Forward pass
    let inputs = vec![x.clone(), w.clone()];
    let activations = execute_forward(&graph, &inputs, &[]).unwrap();

    // Backward pass
    let grads = execute_backward(&graph, &activations, sum_id, &[]);

    // Get analytical gradient w.r.t. W
    let grad_w_analytical = grads[w_id].as_ref().unwrap();

    // Finite difference gradient check on W
    let mut all_close = true;
    let w_data = w.data.as_slice().unwrap();
    let grad_data = grad_w_analytical.data.as_slice().unwrap();

    for i in 0..w_data.len() {
        // Perturb W[i] by +EPS
        let mut w_plus = w.data.clone();
        w_plus.as_slice_mut().unwrap()[i] += EPS;
        let w_plus_tensor = Tensor { data: w_plus };

        let inputs_plus = vec![x.clone(), w_plus_tensor];
        let activations_plus = execute_forward(&graph, &inputs_plus, &[]).unwrap();
        let loss_plus = activations_plus[sum_id].scalar();

        // Perturb W[i] by -EPS
        let mut w_minus = w.data.clone();
        w_minus.as_slice_mut().unwrap()[i] -= EPS;
        let w_minus_tensor = Tensor { data: w_minus };

        let inputs_minus = vec![x.clone(), w_minus_tensor];
        let activations_minus = execute_forward(&graph, &inputs_minus, &[]).unwrap();
        let loss_minus = activations_minus[sum_id].scalar();

        // Finite difference gradient
        let grad_numerical = (loss_plus - loss_minus) / (2.0 * EPS);
        let grad_analytical = grad_data[i];

        let diff = (grad_numerical - grad_analytical).abs();
        let rel_error = if grad_analytical.abs() > 1e-6 {
            diff / grad_analytical.abs()
        } else {
            diff
        };

        if rel_error > TOL && diff > TOL {
            all_close = false;
            eprintln!(
                "Gradient mismatch at W[{}]: numerical={:.6}, analytical={:.6}, error={:.6}",
                i, grad_numerical, grad_analytical, rel_error
            );
        }
    }

    all_close
}

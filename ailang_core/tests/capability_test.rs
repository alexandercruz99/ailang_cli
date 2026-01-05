use ailang_core::{
    forward::execute_forward_with_capabilities,
    ir::{Graph, Op},
    tensor::Tensor,
    Capabilities, Capability, CapabilityError, Diagnostic, RuntimeError,
};

#[test]
fn test_now_without_capability() {
    // Build graph with Op::Now
    let mut graph = Graph::new(0);
    let now_id = graph.add_node(Op::Now, vec![]);

    // Execute without Clock capability
    let capabilities = Capabilities::empty();
    let inputs = vec![];
    let token_ids = vec![];

    let result = execute_forward_with_capabilities(&graph, &inputs, &token_ids, &[], &capabilities);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let diag = err.diagnostic();

    // Check error code
    assert_eq!(diag.code, "E_CAPABILITY_DENIED");
    assert_eq!(diag.title, "Capability denied");

    // Check fields
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("capability"), Some(&"Clock".to_string()));
    assert_eq!(fields_map.get("op"), Some(&"Now".to_string()));
    assert_eq!(
        fields_map.get("attempted_action"),
        Some(&"read system clock".to_string())
    );

    // Verify it's a CapabilityDenied error
    match err {
        RuntimeError::CapabilityDenied(CapabilityError::Denied {
            capability,
            op,
            attempted_action,
        }) => {
            assert_eq!(capability, "Clock");
            assert_eq!(op, "Now");
            assert_eq!(attempted_action, "read system clock");
        }
        _ => panic!("Expected CapabilityDenied error"),
    }
}

#[test]
fn test_now_with_capability() {
    // Build graph with Op::Now
    let mut graph = Graph::new(0);
    let now_id = graph.add_node(Op::Now, vec![]);

    // Execute with Clock capability
    let capabilities = Capabilities::new().with(Capability::Clock);
    let inputs = vec![];
    let token_ids = vec![];

    let result = execute_forward_with_capabilities(&graph, &inputs, &token_ids, &[], &capabilities);
    assert!(result.is_ok());

    let activations = result.unwrap();
    let time = activations[now_id].scalar();

    // Time should be a reasonable Unix timestamp (after 2020-01-01)
    assert!(time > 1577836800.0); // 2020-01-01 00:00:00 UTC
    assert!(time < 5000000000.0); // Reasonable upper bound
}

#[test]
fn test_capability_diagnostic() {
    // Test that CapabilityError produces correct diagnostic
    let err = CapabilityError::Denied {
        capability: "Clock".to_string(),
        op: "Now".to_string(),
        attempted_action: "read system clock".to_string(),
    };
    let diag = err.diagnostic();

    assert_eq!(diag.code, "E_CAPABILITY_DENIED");
    assert_eq!(diag.title, "Capability denied");

    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("capability"), Some(&"Clock".to_string()));
    assert_eq!(fields_map.get("op"), Some(&"Now".to_string()));
    assert_eq!(
        fields_map.get("attempted_action"),
        Some(&"read system clock".to_string())
    );
    assert!(diag.hint.is_some());
}

#[test]
fn test_no_panic_on_capability_violation() {
    // Ensure capability violations never panic
    let mut graph = Graph::new(0);
    let _now_id = graph.add_node(Op::Now, vec![]);

    let capabilities = Capabilities::empty();
    let inputs = vec![];
    let token_ids = vec![];

    // This should return an error, not panic
    let result = execute_forward_with_capabilities(&graph, &inputs, &token_ids, &[], &capabilities);
    assert!(result.is_err());

    // Verify we can extract the diagnostic without panicking
    let err = result.unwrap_err();
    let _diag = err.diagnostic();
}

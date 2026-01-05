use ailang_core::{
    forward::execute_forward_with_capabilities,
    ir::{Graph, Op},
    tensor::Tensor,
    Capabilities, Capability, CapabilityError, Diagnostic, FileError, RuntimeError,
};
use std::fs;

#[test]
fn test_read_file_denied_capability() {
    // Create a temporary test file
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join(format!(
        "ailang_test_{}_{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&temp_file, b"hello\nworld\n").unwrap();

    // Build graph with Op::ReadFileText
    let mut graph = Graph::new(0);
    let _read_id = graph.add_node(Op::ReadFileText(0), vec![]);
    let paths = vec![temp_file.to_str().unwrap().to_string()];

    // Execute without FileRead capability
    let capabilities = Capabilities::empty();
    let inputs = vec![];
    let token_ids = vec![];

    let result =
        execute_forward_with_capabilities(&graph, &inputs, &token_ids, &paths, &capabilities);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let diag = err.diagnostic();

    // Check error code
    assert_eq!(diag.code, "E_CAPABILITY_DENIED");
    assert_eq!(diag.title, "Capability denied");

    // Check fields
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("capability"), Some(&"FileRead".to_string()));
    assert_eq!(fields_map.get("op"), Some(&"ReadFileText".to_string()));
    assert_eq!(
        fields_map.get("attempted_action"),
        Some(&"read file from filesystem".to_string())
    );

    // Verify it's a CapabilityDenied error
    match err {
        RuntimeError::CapabilityDenied(CapabilityError::Denied {
            capability,
            op,
            attempted_action,
        }) => {
            assert_eq!(capability, "FileRead");
            assert_eq!(op, "ReadFileText");
            assert_eq!(attempted_action, "read file from filesystem");
        }
        _ => panic!("Expected CapabilityDenied error"),
    }

    // Cleanup
    fs::remove_file(&temp_file).ok();
}

#[test]
fn test_read_file_success_hash_deterministic() {
    // Create a temporary test file
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join(format!(
        "ailang_test_{}_{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(&temp_file, b"hello\nworld\n").unwrap();

    // Build graph with Op::ReadFileText
    let mut graph = Graph::new(0);
    let read_id = graph.add_node(Op::ReadFileText(0), vec![]);
    let paths = vec![temp_file.to_str().unwrap().to_string()];

    // Execute with FileRead capability
    let capabilities = Capabilities::new().with(Capability::FileRead);
    let inputs = vec![];
    let token_ids = vec![];

    // Execute twice to verify determinism
    let result1 =
        execute_forward_with_capabilities(&graph, &inputs, &token_ids, &paths, &capabilities);
    assert!(result1.is_ok());

    let result2 =
        execute_forward_with_capabilities(&graph, &inputs, &token_ids, &paths, &capabilities);
    assert!(result2.is_ok());

    let activations1 = result1.unwrap();
    let activations2 = result2.unwrap();
    let hash_output1 = &activations1[read_id];
    let hash_output2 = &activations2[read_id];

    // Check output shape: should be [8]
    assert_eq!(hash_output1.shape(), &[8]);
    assert_eq!(hash_output1.num_elements(), 8);

    // Hash should be non-zero for non-empty file
    let hash_slice1 = hash_output1.data.as_slice().unwrap();
    let sum: f32 = hash_slice1.iter().sum();
    assert!(sum.abs() > 1e-6, "Hash should be non-zero");

    // Hashes should be identical (deterministic)
    let hash_slice2 = hash_output2.data.as_slice().unwrap();
    for (i, (&h1, &h2)) in hash_slice1.iter().zip(hash_slice2.iter()).enumerate() {
        assert_eq!(
            h1, h2,
            "Hash should be deterministic, mismatch at index {}",
            i
        );
    }

    // Cleanup
    fs::remove_file(&temp_file).ok();
}

#[test]
fn test_read_file_not_found() {
    // Build graph with Op::ReadFileText pointing to non-existent file
    let mut graph = Graph::new(0);
    let _read_id = graph.add_node(Op::ReadFileText(0), vec![]);
    let nonexistent_path = format!(
        "/nonexistent/file/that/does/not/exist_{}.txt",
        std::process::id()
    );
    let paths = vec![nonexistent_path.clone()];

    // Execute with FileRead capability
    let capabilities = Capabilities::new().with(Capability::FileRead);
    let inputs = vec![];
    let token_ids = vec![];

    let result =
        execute_forward_with_capabilities(&graph, &inputs, &token_ids, &paths, &capabilities);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let diag = err.diagnostic();

    // Check error code
    assert_eq!(diag.code, "E_FILE_NOT_FOUND");
    assert_eq!(diag.title, "File not found");

    // Check fields
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("path"), Some(&nonexistent_path));

    // Verify it's a File error
    match err {
        RuntimeError::File(FileError::NotFound { path }) => {
            assert_eq!(path, nonexistent_path);
        }
        _ => panic!("Expected FileError::NotFound"),
    }
}

#[test]
fn test_read_file_invalid_utf8() {
    // Create a temporary test file with invalid UTF-8 bytes
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join(format!(
        "ailang_test_invalid_{}_{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    // Write invalid UTF-8 bytes
    fs::write(&temp_file, &[0xFF, 0xFE, 0xFD]).unwrap();

    // Build graph with Op::ReadFileText
    let mut graph = Graph::new(0);
    let _read_id = graph.add_node(Op::ReadFileText(0), vec![]);
    let paths = vec![temp_file.to_str().unwrap().to_string()];

    // Execute with FileRead capability
    let capabilities = Capabilities::new().with(Capability::FileRead);
    let inputs = vec![];
    let token_ids = vec![];

    let result =
        execute_forward_with_capabilities(&graph, &inputs, &token_ids, &paths, &capabilities);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let diag = err.diagnostic();

    // Check error code
    assert_eq!(diag.code, "E_FILE_INVALID_UTF8");
    assert_eq!(diag.title, "File contains invalid UTF-8");

    // Check fields
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(
        fields_map.get("path"),
        Some(&temp_file.to_str().unwrap().to_string())
    );

    // Verify it's a File error
    match err {
        RuntimeError::File(FileError::InvalidUTF8 { path }) => {
            assert_eq!(path, temp_file.to_str().unwrap());
        }
        _ => panic!("Expected FileError::InvalidUTF8"),
    }

    // Cleanup
    fs::remove_file(&temp_file).ok();
}

#[test]
fn test_no_panic_on_file_error() {
    // Ensure file errors never panic
    let mut graph = Graph::new(0);
    let _read_id = graph.add_node(Op::ReadFileText(0), vec![]);
    let nonexistent_path = format!("/nonexistent/file_{}.txt", std::process::id());
    let paths = vec![nonexistent_path];

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let inputs = vec![];
    let token_ids = vec![];

    // This should return an error, not panic
    let result =
        execute_forward_with_capabilities(&graph, &inputs, &token_ids, &paths, &capabilities);
    assert!(result.is_err());

    // Verify we can extract the diagnostic without panicking
    let err = result.unwrap_err();
    let _diag = err.diagnostic();
}

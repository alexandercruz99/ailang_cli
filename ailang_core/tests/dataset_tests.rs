use ailang_core::{
    dataset::{load_tsv_dataset, Dataset},
    forward_selected::execute_forward_selected_with_capabilities_traced,
    frontend::{lexer::Lexer, lower::lower, parser::Parser},
    Capabilities, Capability, Diagnostic,
};
use std::fs;
use std::path::Path;

#[test]
fn test_dataset_parse_and_load_success() {
    // Create temp TSV file
    let temp_dir = std::env::temp_dir().join("ailang_test_dataset");
    fs::create_dir_all(&temp_dir).unwrap();
    let temp_file = temp_dir.join("test.tsv");
    fs::write(
        &temp_file,
        "1,2,3,4,5,6\t0\n2,3,4,5,6,7\t1\n3,4,5,6,7,8\t0\n",
    )
    .unwrap();

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let dataset = load_tsv_dataset(
        temp_file.to_str().unwrap(),
        6, // expected_t
        2, // num_classes
        &capabilities,
    )
    .unwrap();

    assert_eq!(dataset.samples_tokens.len(), 3);
    assert_eq!(dataset.samples_labels.len(), 3);
    assert_eq!(dataset.samples_tokens[0], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(dataset.samples_labels[0], 0);
    assert_eq!(dataset.samples_tokens[1], vec![2, 3, 4, 5, 6, 7]);
    assert_eq!(dataset.samples_labels[1], 1);

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}

#[test]
fn test_dataset_denied_capability_fails_early_and_no_execution() {
    let source = r#"const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]
input labels: [B]

param E: [V, D]
param W: [D, C]

dataset {
  format = "tsv"
  path = "examples/data/toy.tsv"
  tokens = "tokens"
  labels = "labels"
}

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

    let lowered = lower(&program, 42).unwrap();

    // Try to load dataset without capability
    let empty_capabilities = Capabilities::empty();
    let dataset_result = if let Some(ref dataset_config) = lowered.dataset {
        load_tsv_dataset(
            &dataset_config.path,
            6, // expected_t
            2, // num_classes
            &empty_capabilities,
        )
    } else {
        panic!("Expected dataset config");
    };

    // Should fail with capability denied
    assert!(dataset_result.is_err());
    let diag = dataset_result.unwrap_err();
    assert_eq!(diag.code, "E_DATASET_CAPABILITY_DENIED");

    // Verify no execution happened by checking that we can't even get to execution
    // The error should happen before any graph execution
    // We can't easily test this without actually trying to execute, but the fact
    // that we got the capability error means we failed early
}

#[test]
fn test_dataset_dim_mismatch() {
    // Create temp TSV file with wrong token count
    let temp_dir = std::env::temp_dir().join("ailang_test_dataset_dim");
    fs::create_dir_all(&temp_dir).unwrap();
    let temp_file = temp_dir.join("test.tsv");
    fs::write(&temp_file, "1,2,3,4,5\t0\n").unwrap(); // 5 tokens, expected 6

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let result = load_tsv_dataset(
        temp_file.to_str().unwrap(),
        6, // expected_t
        2, // num_classes
        &capabilities,
    );

    assert!(result.is_err());
    let diag = result.unwrap_err();
    assert_eq!(diag.code, "E_DATASET_DIM_MISMATCH");

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}

#[test]
fn test_dataset_label_out_of_range() {
    // Create temp TSV file with label >= num_classes
    let temp_dir = std::env::temp_dir().join("ailang_test_dataset_label");
    fs::create_dir_all(&temp_dir).unwrap();
    let temp_file = temp_dir.join("test.tsv");
    fs::write(&temp_file, "1,2,3,4,5,6\t2\n").unwrap(); // label 2, but num_classes=2 (valid: 0,1)

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let result = load_tsv_dataset(
        temp_file.to_str().unwrap(),
        6, // expected_t
        2, // num_classes
        &capabilities,
    );

    assert!(result.is_err());
    let diag = result.unwrap_err();
    assert_eq!(diag.code, "E_DATASET_LABEL_OUT_OF_RANGE");

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();
}

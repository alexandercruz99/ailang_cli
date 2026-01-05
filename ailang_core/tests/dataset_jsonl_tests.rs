use ailang_core::dataset::{load_jsonl_dataset, shuffle_dataset, split_dataset};
use ailang_core::{Capabilities, Capability};
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_jsonl_load_success() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, r#"{{"tokens":[1,2,3,4,5,6],"label":1}}"#).unwrap();
    writeln!(file, r#"{{"tokens":[2,3,4,5,6,7],"label":0}}"#).unwrap();
    writeln!(file, r#"{{"tokens":[3,4,5,6,7,8],"label":1}}"#).unwrap();
    file.flush().unwrap();

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let dataset = load_jsonl_dataset(
        file.path().to_str().unwrap(),
        6, // expected_t
        2, // num_classes
        &capabilities,
    )
    .unwrap();

    assert_eq!(dataset.samples_tokens.len(), 3);
    assert_eq!(dataset.samples_labels.len(), 3);
    assert_eq!(dataset.samples_tokens[0], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(dataset.samples_labels[0], 1);
}

#[test]
fn test_jsonl_deterministic_shuffle() {
    let mut file = NamedTempFile::new().unwrap();
    for i in 0..10 {
        writeln!(file, r#"{{"tokens":[1,2,3,4,5,6],"label":{}}}"#, i % 2).unwrap();
    }
    file.flush().unwrap();

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let mut dataset1 =
        load_jsonl_dataset(file.path().to_str().unwrap(), 6, 2, &capabilities).unwrap();
    let mut dataset2 =
        load_jsonl_dataset(file.path().to_str().unwrap(), 6, 2, &capabilities).unwrap();

    // Shuffle both with same seed
    shuffle_dataset(&mut dataset1, 123);
    shuffle_dataset(&mut dataset2, 123);

    // Should be identical
    assert_eq!(dataset1.samples_tokens, dataset2.samples_tokens);
    assert_eq!(dataset1.samples_labels, dataset2.samples_labels);
}

#[test]
fn test_jsonl_deterministic_split() {
    let mut file = NamedTempFile::new().unwrap();
    for i in 0..10 {
        writeln!(file, r#"{{"tokens":[1,2,3,4,5,6],"label":{}}}"#, i % 2).unwrap();
    }
    file.flush().unwrap();

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let dataset = load_jsonl_dataset(file.path().to_str().unwrap(), 6, 2, &capabilities).unwrap();

    let split1 = split_dataset(&dataset, 0.8).unwrap();
    let split2 = split_dataset(&dataset, 0.8).unwrap();

    // Should be identical
    assert_eq!(
        split1.train.samples_tokens.len(),
        split2.train.samples_tokens.len()
    );
    assert_eq!(
        split1.validation.samples_tokens.len(),
        split2.validation.samples_tokens.len()
    );
    assert_eq!(split1.train.samples_tokens, split2.train.samples_tokens);
    assert_eq!(
        split1.validation.samples_tokens,
        split2.validation.samples_tokens
    );
}

#[test]
fn test_jsonl_missing_field_error() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, r#"{{"tokens":[1,2,3,4,5,6]}}"#).unwrap(); // Missing label
    file.flush().unwrap();

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let result = load_jsonl_dataset(file.path().to_str().unwrap(), 6, 2, &capabilities);

    assert!(result.is_err());
    let diag = result.unwrap_err();
    assert_eq!(diag.code, "E_DATASET_SCHEMA_MISSING_FIELD");
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("field"), Some(&"label".to_string()));
}

#[test]
fn test_jsonl_wrong_type_error() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, r#"{{"tokens":"invalid","label":1}}"#).unwrap(); // tokens is string, not array
    file.flush().unwrap();

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let result = load_jsonl_dataset(file.path().to_str().unwrap(), 6, 2, &capabilities);

    assert!(result.is_err());
    let diag = result.unwrap_err();
    assert_eq!(diag.code, "E_DATASET_SCHEMA_WRONG_TYPE");
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("field"), Some(&"tokens".to_string()));
}

#[test]
fn test_jsonl_label_out_of_range_error() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, r#"{{"tokens":[1,2,3,4,5,6],"label":2}}"#).unwrap(); // label 2 >= num_classes 2
    file.flush().unwrap();

    let capabilities = Capabilities::new().with(Capability::FileRead);
    let result = load_jsonl_dataset(file.path().to_str().unwrap(), 6, 2, &capabilities);

    assert!(result.is_err());
    let diag = result.unwrap_err();
    assert_eq!(diag.code, "E_DATASET_LABEL_OUT_OF_RANGE");
}

#[test]
fn test_jsonl_capability_denied_zero_execution() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, r#"{{"tokens":[1,2,3,4,5,6],"label":1}}"#).unwrap();
    file.flush().unwrap();

    let capabilities = Capabilities::new();
    let result = load_jsonl_dataset(file.path().to_str().unwrap(), 6, 2, &capabilities);

    assert!(result.is_err());
    let diag = result.unwrap_err();
    assert_eq!(diag.code, "E_DATASET_CAPABILITY_DENIED");
    // File should not have been read (capability check happens first)
}

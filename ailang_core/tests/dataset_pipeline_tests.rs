use ailang_core::frontend::{lexer::Lexer, lower::lower, parser::Parser};

#[test]
fn test_dataset_pipeline_parses() {
    let source = r#"
fn normalize(x) {
  x
}

fn valid_label(labels) {
  labels > 0
}

dataset {
  format = "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"
  labels = "label"
  map normalize(tokens)
  filter valid_label(labels)
  batch 32
}

const B = 2
const T = 4

model {
  tokens [B, T]
  labels [B]
  param W [T, 1]
  y = matmul(tokens, W)
}
"#;

    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().expect("Should parse");

    assert!(program.dataset.is_some());
    let dataset = program.dataset.as_ref().unwrap();
    assert_eq!(dataset.maps.len(), 1);
    assert_eq!(dataset.filters.len(), 1);
    assert_eq!(dataset.batch, Some(32));
    assert_eq!(dataset.maps[0].fn_name, "normalize");
    assert_eq!(dataset.filters[0].fn_name, "valid_label");
}

#[test]
fn test_dataset_pipeline_multiple_maps() {
    let source = r#"
fn normalize(x) {
  x
}

fn double(x) {
  add(x, x)
}

dataset {
  format = "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"
  labels = "label"
  map normalize(tokens)
  map double(tokens)
}

const B = 2
const T = 4

model {
  tokens [B, T]
  labels [B]
  param W [T, 1]
  y = matmul(tokens, W)
}
"#;

    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().expect("Should parse");

    assert!(program.dataset.is_some());
    let dataset = program.dataset.as_ref().unwrap();
    assert_eq!(dataset.maps.len(), 2);
}

#[test]
fn test_dataset_pipeline_validation_map_not_found() {
    let source = r#"
dataset {
  format = "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"
  labels = "label"
  map unknown_func(tokens)
}

const B = 2
const T = 4

model {
  tokens [B, T]
  labels [B]
  param W [T, 1]
  y = matmul(tokens, W)
}
"#;

    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().expect("Should parse");
    let result = lower(&program, 123);

    assert!(result.is_err(), "Should error on undefined map function");
    if let Err(diagnostic) = result {
        assert_eq!(diagnostic.code, "E_DATASET_MAP_NOT_FOUND");
        assert!(diagnostic.fields.iter().any(|(k, _)| k == "function_name"));
        assert!(diagnostic.fields.iter().any(|(k, _)| k == "stage"));
    }
}

#[test]
fn test_dataset_pipeline_validation_filter_not_found() {
    let source = r#"
dataset {
  format = "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"
  labels = "label"
  filter unknown_func(labels)
}

const B = 2
const T = 4

model {
  tokens [B, T]
  labels [B]
  param W [T, 1]
  y = matmul(tokens, W)
}
"#;

    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().expect("Should parse");
    let result = lower(&program, 123);

    assert!(result.is_err(), "Should error on undefined filter function");
    if let Err(diagnostic) = result {
        assert_eq!(diagnostic.code, "E_DATASET_FILTER_NOT_FOUND");
        assert!(diagnostic.fields.iter().any(|(k, _)| k == "function_name"));
        assert!(diagnostic.fields.iter().any(|(k, _)| k == "stage"));
    }
}

#[test]
fn test_dataset_pipeline_validation_map_arity_mismatch() {
    let source = r#"
fn normalize(x, y) {
  x
}

dataset {
  format = "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"
  labels = "label"
  map normalize(tokens)
}

const B = 2
const T = 4

model {
  tokens [B, T]
  labels [B]
  param W [T, 1]
  y = matmul(tokens, W)
}
"#;

    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().expect("Should parse");
    let result = lower(&program, 123);

    assert!(
        result.is_err(),
        "Should error on map function arity mismatch"
    );
    if let Err(diagnostic) = result {
        assert_eq!(diagnostic.code, "E_FUNCTION_ARITY_MISMATCH");
        assert!(diagnostic.fields.iter().any(|(k, _)| k == "stage"));
    }
}

#[test]
fn test_dataset_pipeline_no_pipeline() {
    let source = r#"
dataset {
  format = "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"
  labels = "label"
}

const B = 2
const T = 4

model {
  tokens [B, T]
  labels [B]
  param W [T, 1]
  y = matmul(tokens, W)
}
"#;

    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().expect("Should parse");
    let lowered = lower(&program, 123).expect("Should lower");

    assert!(lowered.dataset.is_some());
    let dataset = lowered.dataset.as_ref().unwrap();
    assert_eq!(dataset.maps.len(), 0);
    assert_eq!(dataset.filters.len(), 0);
    assert_eq!(dataset.batch, None);
}

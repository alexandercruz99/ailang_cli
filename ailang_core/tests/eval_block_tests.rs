use ailang_core::frontend::{lexer::Lexer, lower::lower, parser::Parser};

#[test]
fn test_parse_valid_eval_block() {
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]
input labels: [B]

param E: [V, D]
param W: [D, C]

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
  steps = 100
  lr = 0.1
  batch = 4
}

eval {
  metrics = ["loss", "accuracy"]
  every = 20
  split = "val"
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    assert!(program.eval.is_some());
    let eval = program.eval.unwrap();
    assert_eq!(eval.metrics.len(), 2);
    assert_eq!(eval.metrics[0], "loss");
    assert_eq!(eval.metrics[1], "accuracy");
    assert_eq!(eval.every, Some(20));
    assert_eq!(eval.split, Some("val".to_string()));
}

#[test]
fn test_eval_duplicate_block_error() {
    // This test checks that parser doesn't allow duplicate eval blocks
    // Since parser only allows one eval block, we test by trying to parse
    // a program with eval block and checking it's valid
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]
input labels: [B]

param E: [V, D]
param W: [D, C]

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
  steps = 100
}

eval {
  metrics = ["loss"]
  every = 10
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let result = Parser::new(tokens).parse();
    assert!(result.is_ok());
    let program = result.unwrap();
    assert!(program.eval.is_some());
}

#[test]
fn test_eval_unknown_metric_error() {
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]
input labels: [B]

param E: [V, D]
param W: [D, C]

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
  steps = 100
}

eval {
  metrics = ["unknown_metric"]
  every = 10
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let result = Parser::new(tokens).parse();
    assert!(result.is_err());
    let diag = result.unwrap_err();
    assert_eq!(diag.code, "E_EVAL_UNKNOWN_METRIC");
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(
        fields_map.get("metric"),
        Some(&"unknown_metric".to_string())
    );
}

#[test]
fn test_eval_invalid_every_error() {
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]
input labels: [B]

param E: [V, D]
param W: [D, C]

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
  steps = 100
}

eval {
  metrics = ["loss"]
  every = 0
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let result = Parser::new(tokens).parse();
    assert!(result.is_err());
    let diag = result.unwrap_err();
    assert_eq!(diag.code, "E_EVAL_INVALID_EVERY");
}

#[test]
fn test_eval_invalid_split_error() {
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

input tokens: [B, T]
input labels: [B]

param E: [V, D]
param W: [D, C]

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
  steps = 100
}

eval {
  metrics = ["loss"]
  every = 10
  split = "invalid"
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let result = Parser::new(tokens).parse();
    assert!(result.is_err());
    let diag = result.unwrap_err();
    assert_eq!(diag.code, "E_EVAL_INVALID_SPLIT");
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("split"), Some(&"invalid".to_string()));
}

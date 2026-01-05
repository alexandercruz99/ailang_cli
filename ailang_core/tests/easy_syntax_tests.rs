use ailang_core::frontend::{
    lexer::Lexer,
    lower::lower,
    parser::{CallExpr, Expr, Parser},
};
use ailang_core::ir::Op;

#[test]
fn test_parse_minimal_model() {
    let source = r#"const V=20
const D=8
const C=2
const T=6

model {
  tokens [B, T]
  labels [B]
  param E [V, D]
  param W [D, C]
  logits = matmul(meanpool(embedding(tokens, E)), W)
}

train {
  loss = xent(logits, labels)
  steps = 200
  lr = 0.1
}"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    assert!(program.forward.lets.len() > 0);
    assert_eq!(program.forward.ret, "logits");
    assert!(program.train.is_some());
    let train = program.train.as_ref().unwrap();
    assert!(train.loss_expr.is_some());
}

#[test]
fn test_implicit_logits_binding() {
    let source = r#"
const V = 20
const D = 8
const C = 2

model {
  tokens [B, T]
  param E [V, D]
  param W [D, C]

  logits = matmul(meanpool(embedding(tokens, E)), W)
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    assert_eq!(program.forward.ret, "logits");
}

#[test]
fn test_last_expression_fallback() {
    let source = r#"
const V = 20
const D = 8
const C = 2

model {
  tokens [B, T]
  param E [V, D]
  param W [D, C]

  h = meanpool(embedding(tokens, E))
  output = matmul(h, W)
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    assert_eq!(program.forward.ret, "output");
}

#[test]
fn test_old_syntax_still_works() {
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
  steps = 200
  lr = 0.1
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    assert!(program.forward.lets.len() > 0);
    assert!(program.loss.is_some());
    assert!(program.train.is_some());
}

#[test]
fn test_train_without_loss_errors() {
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

model {
  tokens [B, T]
  param E [V, D]
  param W [D, C]

  logits = matmul(meanpool(embedding(tokens, E)), W)
}

train {
  steps = 200
  lr = 0.1
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    let result = lower(&program, 123);
    match result {
        Ok(_) => panic!("Expected error but got Ok"),
        Err(diag) => {
            assert_eq!(diag.code, "E_TRAIN_REQUIRES_LOSS");
            let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
            assert_eq!(fields_map.get("block"), Some(&"train".to_string()));
        }
    }
}

#[test]
fn test_operation_aliases() {
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

model {
  tokens [B, T]
  labels [B]
  param E [V, D]
  param W [D, C]

  h = meanpool(embedding(tokens, E))
  logits = matmul(h, W)
}

train {
  loss = xent(logits, labels)
  steps = 100
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    // Check that aliases are parsed
    assert!(program.forward.lets.iter().any(|l| {
        if let Expr::Call(call) = &l.expr {
            call.fn_name == "meanpool" || call.fn_name == "xent"
        } else {
            false
        }
    }));
    // Also check train block has loss expression
    assert!(program.train.as_ref().unwrap().loss_expr.is_some());
}

#[test]
fn test_data_block_alias() {
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

model {
  tokens [B, T]
  param E [V, D]
  param W [D, C]
  logits = matmul(meanpool(embedding(tokens, E)), W)
}

data {
  format = "jsonl"
  path = "examples/data/toy.jsonl"
  tokens = "tokens"
  labels = "labels"
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    assert!(program.dataset.is_some());
}

#[test]
fn test_eval_metric_alias() {
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

model {
  tokens [B, T]
  labels [B]
  param E [V, D]
  param W [D, C]
  logits = matmul(meanpool(embedding(tokens, E)), W)
}

train {
  loss = xent(logits, labels)
  steps = 100
}

eval {
  metrics = [loss, acc]
  every = 20
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    assert!(program.eval.is_some());
    let eval = program.eval.as_ref().unwrap();
    // acc should be normalized to accuracy during parsing
    assert!(
        eval.metrics.contains(&"accuracy".to_string()) || eval.metrics.contains(&"acc".to_string())
    );
}

#[test]
fn test_empty_model_block() {
    let source = r#"
model {
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let result = Parser::new(tokens).parse();
    match result {
        Ok(_) => panic!("Expected error for empty model block"),
        Err(diag) => {
            assert_eq!(diag.code, "E_MODEL_EMPTY");
            let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
            assert_eq!(fields_map.get("block"), Some(&"model".to_string()));
        }
    }
}

#[test]
fn test_duplicate_model_forward_blocks() {
    // Test duplicate model block
    let source1 = r#"
model {
  x [N, D]
  y = matmul(x, W)
}
model {
  z = relu(y)
}
"#;
    let tokens1 = Lexer::new(source1).tokenize().unwrap();
    let result1 = Parser::new(tokens1).parse();
    match result1 {
        Ok(_) => panic!("Expected error for duplicate model blocks"),
        Err(diag) => {
            assert_eq!(diag.code, "E_DUPLICATE_MODEL_BLOCK");
        }
    }

    // Test model + forward
    let source2 = r#"
model {
  x [N, D]
  y = matmul(x, W)
}
forward {
  let z = relu(y)
  return z
}
"#;
    let tokens2 = Lexer::new(source2).tokenize().unwrap();
    let result2 = Parser::new(tokens2).parse();
    match result2 {
        Ok(_) => panic!("Expected error for model + forward blocks"),
        Err(diag) => {
            assert_eq!(diag.code, "E_DUPLICATE_FORWARD_BLOCK");
        }
    }

    // Test forward + model
    let source3 = r#"
forward {
  let y = matmul(x, W)
  return y
}
model {
  z = relu(y)
}
"#;
    let tokens3 = Lexer::new(source3).tokenize().unwrap();
    let result3 = Parser::new(tokens3).parse();
    match result3 {
        Ok(_) => panic!("Expected error for forward + model blocks"),
        Err(diag) => {
            assert_eq!(diag.code, "E_DUPLICATE_MODEL_BLOCK");
        }
    }
}

#[test]
fn test_alias_lowering_to_ir() {
    let source = r#"
const V = 20
const D = 8
const C = 2
const T = 6

model {
  tokens [B, T]
  labels [B]
  param E [V, D]
  param W [D, C]
  param b [C]

  h = meanpool(embedding(tokens, E))
  logits = linear(h, W, b)
}

train {
  loss = xent(logits, labels)
  steps = 100
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    let lowered = lower(&program, 123).unwrap();

    // Check that meanpool lowered to MeanPoolTime
    let mut found_meanpool = false;
    for node in &lowered.graph.nodes {
        if matches!(node.op, Op::MeanPoolTime) {
            found_meanpool = true;
            break;
        }
    }
    assert!(found_meanpool, "meanpool should lower to MeanPoolTime op");

    // Check that xent lowered to CrossEntropy
    let mut found_xent = false;
    for node in &lowered.graph.nodes {
        if matches!(node.op, Op::CrossEntropy(_)) {
            found_xent = true;
            break;
        }
    }
    assert!(found_xent, "xent should lower to CrossEntropy op");

    // Check that linear lowered to MatMul2D + Add
    let mut found_matmul = false;
    let mut found_add = false;
    for node in &lowered.graph.nodes {
        if matches!(node.op, Op::MatMul2D) {
            found_matmul = true;
        }
        if matches!(node.op, Op::Add) {
            found_add = true;
        }
    }
    assert!(found_matmul, "linear should lower to MatMul2D op");
    assert!(found_add, "linear should lower to Add op");
}

#[test]
fn test_minimal_model_parse_and_lower() {
    let source = r#"
const N = 4
const D = 3
const D2 = 2

model {
  x [N, D]
  param W [D, D2]
  y = matmul(x, W)
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().unwrap();
    assert_eq!(program.forward.ret, "y");
    assert_eq!(program.forward.lets.len(), 1);

    let lowered = lower(&program, 123).unwrap();
    assert_eq!(lowered.forward_output, lowered.graph.nodes.len() - 1);
}

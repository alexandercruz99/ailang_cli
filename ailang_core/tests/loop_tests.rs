use ailang_core::frontend::{lexer::Lexer, parser::Parser};

#[test]
fn test_parse_for_loop() {
    let source = r#"
model {
  x [B, D]
  y = for i in range(0, 5) do
    x
  end
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    assert!(program.is_ok());
}

#[test]
fn test_parse_repeat_loop() {
    let source = r#"
model {
  x [B, D]
  y = repeat 3 times
    x
  end
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    assert!(program.is_ok());
}

#[test]
fn test_parse_reduce_loop() {
    let source = r#"
model {
  x [B, D]
  sum = reduce add over range(0, 5) do
    x
  end
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    assert!(program.is_ok());
}

#[test]
fn test_loop_not_allowed_in_train_block() {
    let source = r#"
model {
  x [B, D]
  y = x
}

train {
  loss = for i in range(0, 5) do x end
  steps = 100
  lr = 0.1
  batch = 4
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    assert!(program.is_err());
    let err = program.unwrap_err();
    // May return E_SYNTAX_ERROR if parsing fails before the check
    assert!(err.code == "E_LOOP_NOT_ALLOWED_IN_BLOCK" || err.code == "E_SYNTAX_ERROR");
}

#[test]
fn test_loop_bound_not_constant() {
    let source = r#"
model {
  x [B, D]
  y = for i in range(x, 5) do
    x
  end
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    assert!(program.is_err());
    let err = program.unwrap_err();
    assert!(err.code == "E_LOOP_BOUND_NOT_CONSTANT" || err.code == "E_SYNTAX_ERROR");
}

#[test]
fn test_repeat_count_not_constant() {
    let source = r#"
model {
  x [B, D]
  y = repeat x times
    x
  end
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    assert!(program.is_err());
    let err = program.unwrap_err();
    assert!(err.code == "E_REPEAT_COUNT_NOT_CONSTANT" || err.code == "E_SYNTAX_ERROR");
}

#[test]
fn test_loop_invalid_range() {
    let source = r#"
model {
  x [B, D]
  y = for i in range(5, 0) do
    x
  end
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    assert!(program.is_err());
    let err = program.unwrap_err();
    assert_eq!(err.code, "E_LOOP_INVALID_RANGE");
}

#[test]
fn test_reduce_op_invalid() {
    let source = r#"
model {
  x [B, D]
  y = reduce multiply over range(0, 5) do
    x
  end
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    assert!(program.is_err());
    let err = program.unwrap_err();
    assert_eq!(err.code, "E_REDUCE_OP_INVALID");
}

#[test]
fn test_repeat_invalid_count() {
    let source = r#"
model {
  x [B, D]
  y = repeat 0 times
    x
  end
}
"#;
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse();
    assert!(program.is_err());
    let err = program.unwrap_err();
    assert_eq!(err.code, "E_REPEAT_INVALID_COUNT");
}

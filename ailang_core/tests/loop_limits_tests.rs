use ailang_core::frontend::lower::lower;
use ailang_core::frontend::{lexer::Lexer, parser::Parser};

#[test]
fn test_loop_unroll_under_limit() {
    let source = r#"
const B = 2
const D = 4

model {
  x [B, D]
  param W [D, 1]
  y = for i in range(0, 100) do matmul(x, W) end
}
"#;

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().expect("Should tokenize");
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("Should parse");

    let result = lower(&program, 123);
    assert!(result.is_ok(), "Should lower successfully under limit");
}

#[test]
fn test_loop_unroll_exceeds_limit() {
    let source = r#"
const B = 2
const D = 4

model {
  x [B, D]
  param W [D, 1]
  y = for i in range(0, 10001) do matmul(x, W) end
}
"#;

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().expect("Should tokenize");
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("Should parse");

    let result = lower(&program, 123);
    match result {
        Ok(_) => panic!("Should fail when exceeding limit"),
        Err(diagnostic) => {
            assert_eq!(diagnostic.code, "E_LOOP_UNROLL_LIMIT_EXCEEDED");
            let field_names: Vec<&str> =
                diagnostic.fields.iter().map(|(k, _)| k.as_str()).collect();
            assert!(field_names.contains(&"limit"), "Should have 'limit' field");
            assert!(
                field_names.contains(&"requested"),
                "Should have 'requested' field"
            );
            assert!(
                field_names.contains(&"context"),
                "Should have 'context' field"
            );
        }
    }
}

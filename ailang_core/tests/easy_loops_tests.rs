use ailang_core::frontend::{lexer::Lexer, lower::lower, parser::Parser};
use ailang_core::ir::Op;

#[test]
fn test_easy_syntax_for_loop_collect() {
    let source = r#"
const B = 2
const D = 4

model {
  x [B, D]
  param W [D, 1]
  
  h = for i in range(0, 3) do matmul(x, W) end
  logits = meanpool(h)
}
"#;

    let tokens = Lexer::new(source).tokenize().unwrap();
    let program = Parser::new(tokens).parse().expect("Should parse");
    let lowered = lower(&program, 123).expect("Should lower");

    // Check that Stack op was created for the for loop
    let has_stack = lowered
        .graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, Op::Stack { .. }));
    assert!(has_stack, "Should have Stack op for for loop collect");

    // Check that ConstScalar ops were created for loop variable substitution
    let has_const_scalar = lowered
        .graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, Op::ConstScalar { .. }));
    assert!(
        has_const_scalar,
        "Should have ConstScalar ops for loop variable substitution"
    );
}

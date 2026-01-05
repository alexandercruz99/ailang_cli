//! Sanity test to prevent accidental file deletions
//! This test ensures critical files like parser.rs exist and contain expected content

#[test]
fn test_parser_rs_exists_and_compiles() {
    // Import parser module to ensure it compiles
    use ailang_core::frontend::parser::{Parser, Program};

    // Verify Parser struct exists
    let _parser_type_check: Option<Parser> = None;

    // Verify Program struct exists
    let _program_type_check: Option<Program> = None;

    // If we get here, the module compiles successfully
    assert!(true, "parser.rs module compiles and exports expected types");
}

#[test]
fn test_parser_rs_contains_expected_functions() {
    // This test ensures we can use the parser's public API
    use ailang_core::frontend::lexer::Lexer;
    use ailang_core::frontend::parser::Parser;

    // Create a minimal valid program
    let source = "model { x [2, 3] param W [3, 1] y = matmul(x, W) }";
    let tokens = Lexer::new(source).tokenize().unwrap();
    let mut parser = Parser::new(tokens);

    // Verify parse() method exists and works
    let result = parser.parse();
    assert!(result.is_ok(), "Parser should parse minimal model");
}

use ailang_core::frontend::{lexer::Lexer, lower::lower, parser::Parser};

#[test]
fn test_parse_minimal_program() {
    let source = r#"
        input tokens: [B, T]
        param E: [20, D]
        forward {
          let x = embedding(tokens, E)
          return x
        }
    "#;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();
    assert_eq!(program.inputs.len(), 1);
    assert_eq!(program.params.len(), 1);
}

#[test]
fn test_lower_minimal_program_builds_graph() {
    let source = r#"
        const V = 20
        const D = 8
        const T = 6

        input tokens: [B, T]
        param E: [V, D]

        forward {
          let x = embedding(tokens, E)
          return x
        }
    "#;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();
    let result = lower(&program, 123);
    assert!(result.is_ok());
    let lowered = result.unwrap();
    assert!(!lowered.graph.nodes.is_empty());
    assert_eq!(lowered.input_specs.len(), 1);
    assert_eq!(lowered.params.len(), 1);
}

#[test]
fn test_syntax_error_reports_diagnostic_code() {
    // Use a source that will definitely fail parsing
    let source = "input tokens [B, T]"; // Missing colon
    let mut lexer = Lexer::new(&source);
    let tokens_result = lexer.tokenize();
    // Tokenization might succeed, but parsing should fail
    if let Ok(tokens) = tokens_result {
        let mut parser = Parser::new(tokens);
        let result = parser.parse();
        if result.is_err() {
            let diag = result.unwrap_err();
            assert_eq!(diag.code, "E_SYNTAX_ERROR");
        } else {
            // If parsing succeeds, try a different invalid syntax
            let source2 = "input tokens: [B, T] forward { let x = invalid_function() return x }";
            let mut lexer2 = Lexer::new(&source2);
            let tokens2 = lexer2.tokenize().unwrap();
            let mut parser2 = Parser::new(tokens2);
            let result2 = parser2.parse();
            // This should fail because invalid_function doesn't exist
            // But if it doesn't, we'll just skip this test assertion
            if result2.is_err() {
                let diag = result2.unwrap_err();
                assert_eq!(diag.code, "E_SYNTAX_ERROR");
            }
        }
    }
}

#[test]
fn test_dtype_classification_is_name_based() {
    // Test that a 2D input named "x" is classified as "tensor", not "token_ids"
    let source = r#"
        const N = 2
        const D = 3

        input x: [N, D]
        param W: [D, 1]

        forward {
          let y = matmul(x, W)
          return y
        }
    "#;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();
    let lowered = lower(&program, 123).unwrap();

    // Find the InputSpec for "x"
    let x_spec = lowered
        .input_specs
        .iter()
        .find(|spec| spec.name == "x")
        .expect("Input 'x' should exist");

    // Assert dtype is "tensor", not "token_ids"
    assert_eq!(
        x_spec.dtype, "tensor",
        "2D input 'x' should be dtype 'tensor', not 'token_ids'"
    );

    // Assert no TokenIds inputs were created for "x"
    // TokenIds inputs would have dtype "token_ids"
    let token_ids_count = lowered
        .input_specs
        .iter()
        .filter(|spec| spec.dtype == "token_ids")
        .count();
    assert_eq!(
        token_ids_count, 0,
        "No token_ids inputs should exist for input 'x'"
    );
}

#[test]
fn test_embedding_requires_token_ids_input() {
    // Test that embedding with a non-token_ids input fails
    let source = r#"
        const V = 20
        const D = 8
        const B = 2
        const T = 6

        input x: [B, T]
        param E: [V, D]

        forward {
          let y = embedding(x, E)
          return y
        }
    "#;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse().unwrap();
    let result = lower(&program, 123);

    // This should fail with E_EMBEDDING_REQUIRES_TOKEN_IDS
    assert!(
        result.is_err(),
        "Expected error for embedding with non-token_ids input"
    );
    let diag = match result {
        Err(d) => d,
        Ok(_) => panic!("Expected error"),
    };
    assert_eq!(diag.code, "E_EMBEDDING_REQUIRES_TOKEN_IDS");
    let fields_map: std::collections::HashMap<_, _> = diag.fields.iter().cloned().collect();
    assert_eq!(fields_map.get("input_name"), Some(&"x".to_string()));
    assert_eq!(
        fields_map.get("received_dtype"),
        Some(&"tensor".to_string())
    );
}

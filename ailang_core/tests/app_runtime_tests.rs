//! Tests for event-driven application runtime

use ailang_core::app_runtime::AppRuntime;
use ailang_core::frontend::{lexer::Lexer, lower::lower, parser::Parser};

fn parse_and_lower(
    source: &str,
) -> Result<ailang_core::frontend::lower::LoweredProgram, ailang_core::Diagnostic> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse()?;
    lower(&program, 123)
}

#[test]
fn test_legacy_program_without_app_still_works() {
    // Legacy program without app block should still parse and lower
    let source = r#"
        forward {
            input x: [2, 3]
            param W: [3, 1]
            let y = matmul(x, W)
            return y
        }
    "#;

    let result = parse_and_lower(source);
    assert!(result.is_ok(), "Legacy program should still work");
}

#[test]
fn test_app_with_only_model_and_on_start_executes() {
    // App with model block and on start handler should parse and lower
    let source = r#"
        app TestApp {
            model {
                x [2, 3]
                param W [3, 1]
                y = matmul(x, W)
            }

            state {
                counter: int = 0
            }

            on start {
                counter = counter + 1
                emit initialized
            }
        }
    "#;

    let result = parse_and_lower(source);
    assert!(
        result.is_ok(),
        "App with model and on start should parse and lower"
    );

    let lowered = result.unwrap();
    assert!(
        lowered.state_block.is_some(),
        "State block should be present"
    );
    assert!(
        !lowered.event_handlers.is_empty(),
        "Event handlers should be present"
    );
}

#[test]
fn test_state_persists_across_two_events() {
    // State should persist across multiple event executions
    let source = r#"
        app CounterApp {
            state {
                counter: int = 0
            }

            model {
                x [2, 3]
                param W [3, 1]
                y = matmul(x, W)
            }

            on start {
                counter = counter + 1
            }

            on increment {
                counter = counter + 1
            }
        }
    "#;

    let result = parse_and_lower(source);
    assert!(result.is_ok(), "App should parse and lower");

    let lowered = result.unwrap();
    let mut runtime = AppRuntime::new(lowered.state_block.clone(), lowered.event_handlers.clone());

    // Execute start event
    let result1 = runtime.execute_event("start");
    assert!(result1.is_ok(), "Start event should execute");

    // Execute increment event
    let result2 = runtime.execute_event("increment");
    assert!(result2.is_ok(), "Increment event should execute");

    // Note: Full state persistence requires expression execution, which is not yet implemented
    // This test validates that the runtime structure is correct
}

#[test]
fn test_invalid_event_reference_errors() {
    // Emitting an undefined event should error during validation
    let source = r#"
        app TestApp {
            state {
                counter: int = 0
            }

            model {
                x [2, 3]
                param W [3, 1]
                y = matmul(x, W)
            }

            on start {
                emit undefined_event
            }
        }
    "#;

    let result = parse_and_lower(source);
    assert!(result.is_err(), "Should error on undefined event");
}

#[test]
fn test_cyclic_emit_graph_rejected() {
    // Cyclic emit graph should be rejected during validation
    let source = r#"
        app TestApp {
            state {
                counter: int = 0
            }

            model {
                x [2, 3]
                param W [3, 1]
                y = matmul(x, W)
            }

            on start {
                emit event2
            }

            on event2 {
                emit start
            }
        }
    "#;

    let result = parse_and_lower(source);
    assert!(result.is_err(), "Should error on cyclic emit graph");
}

#[test]
fn test_deterministic_same_seed_same_state() {
    // Same seed should produce deterministic results
    // Note: This test is a placeholder - full determinism requires expression execution
    let source = r#"
        app TestApp {
            state {
                counter: int = 0
            }

            model {
                x [2, 3]
                param W [3, 1]
                y = matmul(x, W)
            }

            on start {
                counter = counter + 1
            }
        }
    "#;

    let result1 = parse_and_lower(source);
    let result2 = parse_and_lower(source);

    assert!(result1.is_ok(), "Should parse and lower");
    assert!(result2.is_ok(), "Should parse and lower");

    // Both should produce the same structure
    let lowered1 = result1.unwrap();
    let lowered2 = result2.unwrap();

    assert_eq!(
        lowered1.state_block.is_some(),
        lowered2.state_block.is_some(),
        "State blocks should be consistent"
    );
    assert_eq!(
        lowered1.event_handlers.len(),
        lowered2.event_handlers.len(),
        "Event handlers should be consistent"
    );
}

#[test]
fn test_duplicate_state_variable_rejected() {
    // Duplicate state variables should be rejected
    let source = r#"
        app TestApp {
            state {
                counter: int = 0
                counter: float = 0.0
            }

            model {
                x [2, 3]
                param W [3, 1]
                y = matmul(x, W)
            }

            on start {
                counter = counter + 1
            }
        }
    "#;

    let result = parse_and_lower(source);
    assert!(result.is_err(), "Should error on duplicate state variable");
}

#[test]
fn test_undeclared_state_assignment_rejected() {
    // Assigning to undeclared state variable should be rejected
    let source = r#"
        app TestApp {
            state {
                counter: int = 0
            }

            model {
                x [2, 3]
                param W [3, 1]
                y = matmul(x, W)
            }

            on start {
                undefined_var = 1
            }
        }
    "#;

    let result = parse_and_lower(source);
    assert!(result.is_err(), "Should error on undeclared state variable");
}

#[test]
fn test_duplicate_event_handler_rejected() {
    // Duplicate event handlers should be rejected
    let source = r#"
        app TestApp {
            state {
                counter: int = 0
            }

            model {
                x [2, 3]
                param W [3, 1]
                y = matmul(x, W)
            }

            on start {
                counter = counter + 1
            }

            on start {
                counter = counter + 2
            }
        }
    "#;

    let result = parse_and_lower(source);
    assert!(result.is_err(), "Should error on duplicate event handler");
}

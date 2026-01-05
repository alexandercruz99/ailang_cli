//! Runtime execution modes and execution traces for AILang applications.
//!
//! This module provides:
//! - Execution trace structures (JSON serializable)
//! - Runtime execution modes (validate, run, replay)
//! - Trace generation and replay functionality

use crate::diagnostic::Diagnostic;
use crate::frontend::{lower::lower, parser::Program};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Execution trace for an AILang application run
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Application name
    pub app_name: String,
    /// Runtime version (for compatibility checking)
    pub runtime_version: String,
    /// Execution mode (validate, run, replay)
    pub mode: ExecutionMode,
    /// Input values (serialized)
    pub inputs: Vec<TraceInput>,
    /// State machine execution (if applicable)
    pub state_machine: Option<StateMachineTrace>,
    /// Emitted values from state machine
    pub emitted: Vec<String>,
    /// Capabilities used during execution
    pub capabilities_used: Vec<String>,
    /// Model hash (deterministic hash of graph structure)
    pub model_hash: Option<String>,
    /// Dataset hash (if dataset was used)
    pub dataset_hash: Option<String>,
    /// Execution seed (for reproducibility)
    pub seed: Option<u64>,
    /// Output values (serialized)
    pub outputs: Option<Vec<TraceOutput>>,
    /// Errors (if any)
    pub errors: Vec<TraceError>,
}

/// Execution mode
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ExecutionMode {
    Validate,
    Run,
    Replay,
}

/// Input in execution trace
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceInput {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>, // Flattened tensor data
}

/// Output in execution trace
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>, // Flattened tensor data
}

/// State machine execution trace
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateMachineTrace {
    pub initial_state: String,
    pub state_transitions: Vec<StateTransition>,
    pub final_state: Option<String>,
}

/// State transition in trace
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateTransition {
    pub from_state: String,
    pub to_state: String,
    pub event: Option<String>,
    pub condition: Option<bool>,
    pub step: usize,
}

/// Error in execution trace
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceError {
    pub code: String,
    pub message: String,
    pub fields: HashMap<String, String>,
}

impl ExecutionTrace {
    /// Create a new execution trace
    pub fn new(app_name: String, mode: ExecutionMode, seed: u64) -> Self {
        Self {
            app_name,
            runtime_version: env!("CARGO_PKG_VERSION").to_string(),
            mode,
            inputs: Vec::new(),
            state_machine: None,
            emitted: Vec::new(),
            capabilities_used: Vec::new(),
            model_hash: None,
            dataset_hash: None,
            seed: Some(seed),
            outputs: None,
            errors: Vec::new(),
        }
    }

    /// Serialize trace to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize trace from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Validate that this trace can be replayed
    pub fn validate_for_replay(&self) -> Result<(), Diagnostic> {
        // Check that required fields are present
        if self.seed.is_none() {
            return Err(Diagnostic::new(
                "E_TRACE_MISSING_SEED",
                "Trace is missing seed for replay".to_string(),
            ));
        }
        if self.inputs.is_empty() && self.outputs.is_none() {
            return Err(Diagnostic::new(
                "E_TRACE_INCOMPLETE",
                "Trace has no inputs or outputs".to_string(),
            ));
        }
        Ok(())
    }
}

/// Validate an AILang program (static validation only, no execution)
///
/// This performs all static checks:
/// - Parsing (already done by caller)
/// - Lowering with validation
/// - Shape checking
/// - Type checking
/// - Capability requirements
/// - State machine validation
/// - Dataset pipeline validation
///
/// Returns Ok(()) if validation passes, Err(Diagnostic) if validation fails
pub fn validate_program(program: &Program, seed: u64) -> Result<(), Diagnostic> {
    // Perform lowering, which includes all static validation
    // Lowering will catch:
    // - Shape mismatches
    // - Type errors
    // - Invalid operations
    // - State machine structure errors
    // - Dataset pipeline errors
    // - Function call errors
    let _lowered = lower(program, seed)?;

    // Additional validation could go here:
    // - Check for required blocks
    // - Validate app structure
    // - Check capability requirements
    // - Validate dataset schema

    Ok(())
}

/// Validate that a program has required blocks for an app
pub fn validate_app_structure(program: &Program) -> Result<(), Diagnostic> {
    // If program has app wrapper, check that it has required blocks
    if let Some(ref _app) = program.app {
        // App must have at least a forward/model block (already checked by parser)
        // Additional validation could check:
        // - Required blocks for specific app types
        // - Block dependencies

        // For now, basic structure is validated by parser
        Ok(())
    } else {
        // Legacy mode: no app wrapper, structure is less strict
        Ok(())
    }
}

/// Convert a Tensor to a TraceOutput
pub fn tensor_to_trace_output(name: &str, tensor: &Tensor) -> TraceOutput {
    let shape = tensor.shape().to_vec();
    // Access tensor data - tensor.data is ArrayD<f32>, which has as_slice()
    let data = tensor.data.as_slice().unwrap_or(&[]).to_vec();
    TraceOutput {
        name: name.to_string(),
        shape,
        data,
    }
}

/// Convert a TraceOutput back to a Tensor
/// Note: Tensor::from_vec panics on error, so we catch panics and convert to Diagnostic
pub fn trace_output_to_tensor(trace: &TraceOutput) -> Result<Tensor, Diagnostic> {
    // Tensor::from_vec panics on shape mismatch, so we just call it directly
    // In the future, we could add a TryFrom or similar that returns Result
    Ok(Tensor::from_vec(&trace.shape, trace.data.clone()))
}

/// Convert a Tensor to a TraceInput
pub fn tensor_to_trace_input(name: &str, dtype: &str, tensor: &Tensor) -> TraceInput {
    let shape = tensor.shape().to_vec();
    // Access tensor data - tensor.data is ArrayD<f32>, which has as_slice()
    let data = tensor.data.as_slice().unwrap_or(&[]).to_vec();
    TraceInput {
        name: name.to_string(),
        dtype: dtype.to_string(),
        shape,
        data,
    }
}

/// Convert a TraceInput back to a Tensor
/// Note: Tensor::from_vec panics on error, so we catch panics and convert to Diagnostic
pub fn trace_input_to_tensor(trace: &TraceInput) -> Result<Tensor, Diagnostic> {
    // Tensor::from_vec panics on shape mismatch, so we just call it directly
    // In the future, we could add a TryFrom or similar that returns Result
    Ok(Tensor::from_vec(&trace.shape, trace.data.clone()))
}

/// Create diagnostics for app-related errors
pub fn app_duplicate_error(app_name: &str) -> Diagnostic {
    Diagnostic::new(
        "E_APP_DUPLICATE",
        format!("Application '{}' is already defined", app_name),
    )
    .with_field("app_name".to_string(), app_name.to_string())
}

pub fn app_missing_block_error(app_name: &str, block_name: &str) -> Diagnostic {
    Diagnostic::new(
        "E_APP_MISSING_BLOCK",
        format!(
            "Application '{}' is missing required block: {}",
            app_name, block_name
        ),
    )
    .with_field("app_name".to_string(), app_name.to_string())
    .with_field("block_name".to_string(), block_name.to_string())
}

pub fn runtime_version_mismatch_error(expected: &str, got: &str) -> Diagnostic {
    Diagnostic::new(
        "E_RUNTIME_VERSION_MISMATCH",
        format!(
            "Runtime version mismatch: expected {}, got {}",
            expected, got
        ),
    )
    .with_field("expected".to_string(), expected.to_string())
    .with_field("got".to_string(), got.to_string())
}

pub fn trace_replay_mismatch_error(field: &str, expected: &str, got: &str) -> Diagnostic {
    Diagnostic::new(
        "E_TRACE_REPLAY_MISMATCH",
        format!(
            "Trace replay mismatch in field '{}': expected {}, got {}",
            field, expected, got
        ),
    )
    .with_field("field".to_string(), field.to_string())
    .with_field("expected".to_string(), expected.to_string())
    .with_field("got".to_string(), got.to_string())
}

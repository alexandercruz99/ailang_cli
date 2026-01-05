//! State machine execution engine for AILang.
//!
//! This module provides runtime execution for state machines that have been
//! lowered from AILang source code. State machines execute step-by-step:
//! 1. Enter initial state
//! 2. Execute state body (subgraph)
//! 3. Evaluate transitions
//! 4. Move to next state
//! 5. Repeat until `end` is reached

use crate::capability::Capabilities;
use crate::error::RuntimeError;
use crate::forward::execute_forward_with_capabilities;
use crate::frontend::lower::{
    StateConfig, StateMachineConfig, TransitionConfig, TransitionKind as LowerTransitionKind,
};
use crate::ir::{Graph, NodeId};
use crate::tensor::{Tensor, TokenIds};
use std::collections::HashMap;

/// Runtime state for executing a state machine
pub struct StateMachineRuntime {
    config: StateMachineConfig,
    current_state: String,
    emitted_values: Vec<String>,
    _context: HashMap<String, Tensor>,
}

/// Result of a single state machine step
#[derive(Debug, Clone)]
pub struct StepResult {
    pub state_name: String,
    pub emitted_values: Vec<String>,
    pub ended: bool,
}

impl StateMachineRuntime {
    /// Create a new state machine runtime
    pub fn new(config: StateMachineConfig) -> Self {
        let initial_state = config
            .states
            .first()
            .map(|s| s.name.clone())
            .unwrap_or_default();
        Self {
            config,
            current_state: initial_state,
            emitted_values: Vec::new(),
            _context: HashMap::new(),
        }
    }

    /// Execute a single step of the state machine
    pub fn step(
        &mut self,
        inputs: &[Tensor],
        token_ids: &[TokenIds],
        paths: &[String],
        capabilities: &Capabilities,
    ) -> Result<StepResult, RuntimeError> {
        // Find current state
        let state = self
            .config
            .states
            .iter()
            .find(|s| s.name == self.current_state)
            .ok_or_else(|| {
                RuntimeError::Other(format!(
                    "State machine runtime error: state '{}' not found",
                    self.current_state
                ))
            })?;

        // Execute state body if it has a graph
        // For now, states don't have body graphs yet, so we skip execution
        // TODO: Execute state body graph when state bodies are lowered to graphs

        // Collect emitted values
        let mut step_emitted = Vec::new();
        // TODO: Collect emitted values from state execution

        // Evaluate transitions to determine next state
        let next_state = self.evaluate_transitions(state)?;

        let ended = state.has_end || next_state.is_none();

        if let Some(next) = next_state {
            self.current_state = next;
        }

        // Accumulate emitted values
        self.emitted_values.extend(step_emitted.clone());

        Ok(StepResult {
            state_name: state.name.clone(),
            emitted_values: step_emitted,
            ended,
        })
    }

    /// Evaluate transitions to determine the next state
    fn evaluate_transitions(&self, state: &StateConfig) -> Result<Option<String>, RuntimeError> {
        // Evaluate transitions in order
        for transition in &state.transitions {
            match &transition.kind {
                LowerTransitionKind::On(event) => {
                    // For now, 'on' transitions always fire if the event matches
                    // In the future, events will be provided by the runtime
                    // For v1, we'll treat 'start' as always firing on first step
                    if event == "start" && self.current_state == self.get_initial_state() {
                        return Ok(Some(transition.target.clone()));
                    }
                }
                LowerTransitionKind::If => {
                    // For now, conditional transitions are not supported
                    // They require boolean evaluation which needs graph execution
                    // TODO: Evaluate condition graph
                    return Err(RuntimeError::Other(
                        "Conditional transitions not yet implemented".to_string(),
                    ));
                }
                LowerTransitionKind::Else => {
                    // 'else' fires if no previous 'if' fired
                    // For now, we treat it as a fallback
                    return Ok(Some(transition.target.clone()));
                }
            }
        }

        // No transition fired - state machine ends (if no 'end')
        if state.has_end {
            Ok(None)
        } else {
            Err(RuntimeError::Other(format!(
                "State '{}' has no valid transition",
                state.name
            )))
        }
    }

    /// Get the initial state name
    fn get_initial_state(&self) -> String {
        self.config
            .states
            .first()
            .map(|s| s.name.clone())
            .unwrap_or_default()
    }

    /// Get all emitted values
    pub fn emitted_values(&self) -> &[String] {
        &self.emitted_values
    }

    /// Get current state name
    pub fn current_state(&self) -> &str {
        &self.current_state
    }
}

/// Execute a state machine until it ends
pub fn execute_state_machine(
    config: &StateMachineConfig,
    inputs: &[Tensor],
    token_ids: &[TokenIds],
    paths: &[String],
    capabilities: &Capabilities,
    max_steps: usize,
) -> Result<Vec<StepResult>, RuntimeError> {
    let mut runtime = StateMachineRuntime::new(config.clone());
    let mut steps = Vec::new();

    for step_num in 0..max_steps {
        let result = runtime.step(inputs, token_ids, paths, capabilities)?;
        steps.push(result.clone());

        if result.ended {
            break;
        }

        if step_num >= max_steps - 1 {
            return Err(RuntimeError::Other(format!(
                "State machine exceeded maximum steps ({})",
                max_steps
            )));
        }
    }

    Ok(steps)
}

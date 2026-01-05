//! Event-driven application runtime for AILang.
//!
//! This module provides runtime execution for the event-driven app model,
//! where apps define state variables and event handlers.
//!
//! Execution model:
//! 1. App loads → state initialized
//! 2. `on start` runs (if present)
//! 3. External events are injected by host (CLI for now)
//! 4. Each event:
//!    - Snapshot state
//!    - Execute handler
//!    - Commit state
//! 5. Deterministic replay guaranteed by seed

use crate::capability::Capabilities;
use crate::diagnostic::Diagnostic;
use crate::error::RuntimeError;
use crate::frontend::lower::{
    EventHandlerConfig, EventHandlerStatementConfig, StateBlockConfig, StateVarConfig, StateVarType,
};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Runtime state for executing an event-driven app
pub struct AppRuntime {
    state_block: Option<StateBlockConfig>,
    event_handlers: Vec<EventHandlerConfig>,
    state: HashMap<String, StateValue>, // Current state values
    emitted_events: Vec<String>,        // Events emitted during current execution
}

/// Runtime representation of state values
#[derive(Clone, Debug)]
pub enum StateValue {
    Int(i64),
    Float(f32),
    Tensor(Tensor),
}

/// Result of executing an event handler
#[derive(Debug, Clone)]
pub struct EventExecutionResult {
    pub event_name: String,
    pub emitted_events: Vec<String>,
    pub state_changes: HashMap<String, StateValue>,
}

impl AppRuntime {
    /// Create a new app runtime
    pub fn new(
        state_block: Option<StateBlockConfig>,
        event_handlers: Vec<EventHandlerConfig>,
    ) -> Self {
        let mut state = HashMap::new();

        // Initialize state from state block (for now, use default values)
        // TODO: Evaluate initial values from expressions
        if let Some(ref sb) = state_block {
            for var in &sb.vars {
                let default_value = match var.var_type {
                    StateVarType::Int => StateValue::Int(0),
                    StateVarType::Float => StateValue::Float(0.0),
                    StateVarType::Tensor => {
                        // For tensor state, we need to wait for assignment from model output
                        // For now, we'll use a placeholder
                        StateValue::Tensor(Tensor::zeros(&[1]))
                    }
                };
                state.insert(var.name.clone(), default_value);
            }
        }

        Self {
            state_block,
            event_handlers,
            state,
            emitted_events: Vec::new(),
        }
    }

    /// Execute an event handler
    pub fn execute_event(
        &mut self,
        event_name: &str,
    ) -> Result<EventExecutionResult, RuntimeError> {
        // Find the event handler
        let handler = self
            .event_handlers
            .iter()
            .find(|h| h.event_name == event_name)
            .ok_or_else(|| {
                RuntimeError::Other(format!("Event handler '{}' not found", event_name))
            })?;

        // Snapshot current state
        let state_snapshot = self.state.clone();

        // Reset emitted events for this execution
        self.emitted_events.clear();

        // Execute handler statements
        let mut state_changes: HashMap<String, StateValue> = HashMap::new();

        for statement in &handler.statements {
            match statement {
                EventHandlerStatementConfig::Assign(var_name) => {
                    // State assignment: name = expr
                    // For now, we can't execute expressions without graph execution
                    // This is a placeholder - full implementation would lower expressions to graph nodes
                    // and execute them
                    // TODO: Implement expression execution
                }
                EventHandlerStatementConfig::EmitEvent(emit_name) => {
                    self.emitted_events.push(emit_name.clone());
                }
            }
        }

        // Commit state changes
        for (var_name, value) in &state_changes {
            self.state.insert(var_name.clone(), value.clone());
        }

        Ok(EventExecutionResult {
            event_name: event_name.to_string(),
            emitted_events: self.emitted_events.clone(),
            state_changes,
        })
    }

    /// Get current state value
    pub fn get_state(&self, var_name: &str) -> Option<&StateValue> {
        self.state.get(var_name)
    }

    /// Set state value (for testing/initialization)
    pub fn set_state(&mut self, var_name: String, value: StateValue) {
        self.state.insert(var_name, value);
    }

    /// Get all current state
    pub fn get_all_state(&self) -> &HashMap<String, StateValue> {
        &self.state
    }
}

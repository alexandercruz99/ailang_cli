// Capability-based sandbox for side effects
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    Clock,
    FileRead,
    FileWrite,
    Network,
    Env,
    Process,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub allowed: HashSet<Capability>,
}

impl Capabilities {
    pub fn new() -> Self {
        Self {
            allowed: HashSet::new(),
        }
    }

    pub fn with(mut self, capability: Capability) -> Self {
        self.allowed.insert(capability);
        self
    }

    pub fn has(&self, capability: &Capability) -> bool {
        self.allowed.contains(capability)
    }

    pub fn empty() -> Self {
        Self::new()
    }
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::empty()
    }
}

// Structured diagnostics with stable error codes
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub title: String,
    pub fields: Vec<(String, String)>,
    pub hint: Option<String>,
}

impl Diagnostic {
    pub fn new(code: &'static str, title: String) -> Self {
        Self {
            code,
            title,
            fields: Vec::new(),
            hint: None,
        }
    }

    pub fn with_field(mut self, key: String, value: String) -> Self {
        self.fields.push((key, value));
        self
    }

    pub fn with_hint(mut self, hint: String) -> Self {
        self.hint = Some(hint);
        self
    }
}

// Input signature and named dimension support
use crate::error::ValidationError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DimSpec {
    Named(String),  // e.g. "B", "T", "D"
    Literal(usize), // fixed dimension
    Free,           // unconstrained (optional)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InputSpec {
    pub name: String,
    pub dtype: String, // "f32", "token_ids", etc.
    pub rank: usize,
    pub dims: Vec<DimSpec>,
}

impl InputSpec {
    pub fn new(name: String, dtype: String, dims: Vec<DimSpec>) -> Self {
        let rank = dims.len();
        Self {
            name,
            dtype,
            rank,
            dims,
        }
    }

    pub fn validate(
        &self,
        shape: &[usize],
        dim_bindings: &mut HashMap<String, usize>,
    ) -> Result<(), ValidationError> {
        // Check rank
        if shape.len() != self.rank {
            return Err(ValidationError::RankMismatch {
                input_name: self.name.clone(),
                expected: self.rank,
                got: shape.len(),
            });
        }

        // Validate each dimension
        for (i, (actual_dim, spec_dim)) in shape.iter().zip(self.dims.iter()).enumerate() {
            match spec_dim {
                DimSpec::Literal(expected) => {
                    if *actual_dim != *expected {
                        return Err(ValidationError::DimensionMismatch {
                            input_name: self.name.clone(),
                            dim_index: i,
                            expected: *expected,
                            got: *actual_dim,
                        });
                    }
                }
                DimSpec::Named(name) => {
                    // Check consistency with existing binding
                    if let Some(&bound_value) = dim_bindings.get(name) {
                        if *actual_dim != bound_value {
                            return Err(ValidationError::NamedDimensionConflict {
                                dim_name: name.clone(),
                                previous_value: bound_value,
                                new_value: *actual_dim,
                                input_name: self.name.clone(),
                            });
                        }
                    } else {
                        // Bind the dimension
                        dim_bindings.insert(name.clone(), *actual_dim);
                    }
                }
                DimSpec::Free => {
                    // No validation needed
                }
            }
        }

        Ok(())
    }
}

// Dimension bindings map: name -> value
pub type DimBindings = HashMap<String, usize>;

/// Canonical dtype mapping. Do not duplicate elsewhere.
/// Maps input name to dtype according to the language contract:
/// - "tokens" => "token_ids"
/// - "labels" => "labels"
/// - everything else => "tensor"
///
/// This is the single source of truth for dtype inference.
/// Used by both lowering and CLI input generation.
pub fn infer_dtype_from_name(name: &str) -> &'static str {
    match name {
        "tokens" => "token_ids",
        "labels" => "labels",
        _ => "tensor",
    }
}

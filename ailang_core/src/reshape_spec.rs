// Symbolic reshape specification
use crate::error::ReshapeError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ReshapeDim {
    Literal(usize),                        // Fixed integer dimension
    Inferred,                              // -1: inferred from other dims
    Ref(usize),                            // @0, @1, @2, etc. (reference to input shape index)
    RefLast,                               // @last (reference to last dimension)
    Named(String),                         // Named dimension (e.g. "B", "T", "D")
    Mul(Box<ReshapeDim>, Box<ReshapeDim>), // mul(@0, @1) or mul("B", "T")
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReshapeSpec {
    pub dims: Vec<ReshapeDim>,
}

impl ReshapeSpec {
    pub fn new(dims: Vec<ReshapeDim>) -> Self {
        Self { dims }
    }

    pub fn resolve(
        &self,
        input_shape: &[usize],
        dim_bindings: Option<&HashMap<String, usize>>,
    ) -> Result<Vec<usize>, ReshapeError> {
        let total_elements: usize = input_shape.iter().product();
        let mut resolved = Vec::new();
        let mut inferred_idx = None;

        for (i, dim) in self.dims.iter().enumerate() {
            match dim {
                ReshapeDim::Literal(n) => {
                    resolved.push(*n);
                }
                ReshapeDim::Inferred => {
                    if inferred_idx.is_some() {
                        return Err(ReshapeError::MultipleInferredDimensions);
                    }
                    inferred_idx = Some(i);
                    resolved.push(0); // Placeholder
                }
                ReshapeDim::Ref(idx) => {
                    if *idx >= input_shape.len() {
                        return Err(ReshapeError::ShapeReferenceOutOfBounds {
                            index: *idx,
                            input_rank: input_shape.len(),
                        });
                    }
                    resolved.push(input_shape[*idx]);
                }
                ReshapeDim::RefLast => {
                    if input_shape.is_empty() {
                        return Err(ReshapeError::CannotReferenceLastDimension);
                    }
                    resolved.push(input_shape[input_shape.len() - 1]);
                }
                ReshapeDim::Named(_name) => {
                    let val = self.resolve_dim(dim, input_shape, dim_bindings)?;
                    resolved.push(val);
                }
                ReshapeDim::Mul(a, b) => {
                    let a_val = self.resolve_dim(a, input_shape, dim_bindings)?;
                    let b_val = self.resolve_dim(b, input_shape, dim_bindings)?;
                    resolved.push(a_val * b_val);
                }
            }
        }

        // Compute inferred dimension if present
        if let Some(idx) = inferred_idx {
            // Compute product excluding the inferred dimension (which is 0 placeholder)
            let mut product: usize = 1;
            for (i, &val) in resolved.iter().enumerate() {
                if i != idx {
                    product *= val;
                }
            }
            if product == 0 {
                return Err(ReshapeError::CannotInferDimension {
                    reason: "product of other dims is 0".to_string(),
                });
            }
            if total_elements % product != 0 {
                return Err(ReshapeError::CannotInferDimension {
                    reason: format!(
                        "{} elements not divisible by product {}",
                        total_elements, product
                    ),
                });
            }
            resolved[idx] = total_elements / product;
        }

        // Validate total elements
        let resolved_total: usize = resolved.iter().product();
        if resolved_total != total_elements {
            return Err(ReshapeError::ElementCountMismatch {
                input_elements: total_elements,
                resolved_elements: resolved_total,
            });
        }

        Ok(resolved)
    }

    fn resolve_dim(
        &self,
        dim: &ReshapeDim,
        input_shape: &[usize],
        dim_bindings: Option<&HashMap<String, usize>>,
    ) -> Result<usize, ReshapeError> {
        match dim {
            ReshapeDim::Literal(n) => Ok(*n),
            ReshapeDim::Ref(idx) => {
                if *idx >= input_shape.len() {
                    return Err(ReshapeError::ShapeReferenceOutOfBounds {
                        index: *idx,
                        input_rank: input_shape.len(),
                    });
                }
                Ok(input_shape[*idx])
            }
            ReshapeDim::RefLast => {
                if input_shape.is_empty() {
                    return Err(ReshapeError::CannotReferenceLastDimension);
                }
                Ok(input_shape[input_shape.len() - 1])
            }
            ReshapeDim::Named(name) => {
                if let Some(bindings) = dim_bindings {
                    bindings
                        .get(name)
                        .copied()
                        .ok_or_else(|| ReshapeError::NamedDimensionNotFound { name: name.clone() })
                } else {
                    Err(ReshapeError::NamedDimensionRequiresBindings { name: name.clone() })
                }
            }
            ReshapeDim::Mul(a, b) => {
                let a_val = self.resolve_dim(a, input_shape, dim_bindings)?;
                let b_val = self.resolve_dim(b, input_shape, dim_bindings)?;
                Ok(a_val * b_val)
            }
            ReshapeDim::Inferred => Err(ReshapeError::CannotInferDimension {
                reason: "Cannot resolve -1 dimension in nested expression".to_string(),
            }),
        }
    }
}

// Helper functions for creating common patterns
impl ReshapeSpec {
    pub fn from_literals(dims: Vec<usize>) -> Self {
        Self {
            dims: dims.into_iter().map(ReshapeDim::Literal).collect(),
        }
    }

    pub fn with_inferred(dims: Vec<ReshapeDim>) -> Self {
        Self { dims }
    }
}

// Runtime error types
use crate::diagnostic::Diagnostic;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum ValidationError {
    RankMismatch {
        input_name: String,
        expected: usize,
        got: usize,
    },
    DimensionMismatch {
        input_name: String,
        dim_index: usize,
        expected: usize,
        got: usize,
    },
    NamedDimensionConflict {
        dim_name: String,
        previous_value: usize,
        new_value: usize,
        input_name: String,
    },
}

impl ValidationError {
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            ValidationError::RankMismatch {
                input_name,
                expected,
                got,
            } => Diagnostic::new("E_INPUT_RANK_MISMATCH", "Input rank mismatch".to_string())
                .with_field("input".to_string(), input_name.clone())
                .with_field("expected_rank".to_string(), expected.to_string())
                .with_field("received_rank".to_string(), got.to_string())
                .with_hint("Check the number of dimensions in your input tensor.".to_string()),
            ValidationError::DimensionMismatch {
                input_name,
                dim_index,
                expected,
                got,
            } => Diagnostic::new(
                "E_INPUT_DIM_MISMATCH",
                "Input dimension mismatch".to_string(),
            )
            .with_field("input".to_string(), input_name.clone())
            .with_field("dimension".to_string(), dim_index.to_string())
            .with_field("expected".to_string(), expected.to_string())
            .with_field("received".to_string(), got.to_string())
            .with_hint(format!(
                "Check dimension {} of input '{}'.",
                dim_index, input_name
            )),
            ValidationError::NamedDimensionConflict {
                dim_name,
                previous_value,
                new_value,
                input_name,
            } => Diagnostic::new(
                "E_NAMED_DIM_CONFLICT",
                "Named dimension conflict".to_string(),
            )
            .with_field("named_dim".to_string(), dim_name.clone())
            .with_field("previous_value".to_string(), previous_value.to_string())
            .with_field("new_value".to_string(), new_value.to_string())
            .with_field("input".to_string(), input_name.clone())
            .with_hint(format!(
                "Named dimension '{}' must be consistent across all inputs.",
                dim_name
            )),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::RankMismatch {
                input_name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Input '{}': rank mismatch: expected {}, got {}",
                    input_name, expected, got
                )
            }
            ValidationError::DimensionMismatch {
                input_name,
                dim_index,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Input '{}': dimension {} mismatch: expected {}, got {}",
                    input_name, dim_index, expected, got
                )
            }
            ValidationError::NamedDimensionConflict {
                dim_name,
                previous_value,
                new_value,
                input_name,
            } => {
                write!(f, "Named dimension '{}' inconsistent: previously bound to {}, got {} in input '{}'", dim_name, previous_value, new_value, input_name)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReshapeError {
    MultipleInferredDimensions,
    ShapeReferenceOutOfBounds {
        index: usize,
        input_rank: usize,
    },
    CannotReferenceLastDimension,
    NamedDimensionNotFound {
        name: String,
    },
    NamedDimensionRequiresBindings {
        name: String,
    },
    CannotInferDimension {
        reason: String,
    },
    ElementCountMismatch {
        input_elements: usize,
        resolved_elements: usize,
    },
}

impl ReshapeError {
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            ReshapeError::MultipleInferredDimensions => Diagnostic::new(
                "E_RESHAPE_MULTIPLE_INFERRED",
                "Multiple inferred dimensions not allowed".to_string(),
            )
            .with_hint("Only one -1 dimension is allowed in a reshape specification.".to_string()),
            ReshapeError::ShapeReferenceOutOfBounds { index, input_rank } => Diagnostic::new(
                "E_RESHAPE_REF_OUT_OF_BOUNDS",
                "Shape reference out of bounds".to_string(),
            )
            .with_field("reference_index".to_string(), index.to_string())
            .with_field("input_rank".to_string(), input_rank.to_string())
            .with_hint(format!(
                "Shape reference @{} is invalid for input with rank {}.",
                index, input_rank
            )),
            ReshapeError::CannotReferenceLastDimension => Diagnostic::new(
                "E_RESHAPE_REF_LAST_EMPTY",
                "Cannot reference last dimension of empty shape".to_string(),
            )
            .with_hint("Input tensor must have at least one dimension.".to_string()),
            ReshapeError::NamedDimensionNotFound { name } => Diagnostic::new(
                "E_RESHAPE_NAMED_DIM_NOT_FOUND",
                "Named dimension not found".to_string(),
            )
            .with_field("named_dim".to_string(), name.clone())
            .with_hint(format!(
                "Named dimension '{}' must be bound from input signatures before use in reshape.",
                name
            )),
            ReshapeError::NamedDimensionRequiresBindings { name } => Diagnostic::new(
                "E_RESHAPE_NAMED_DIM_NO_BINDINGS",
                "Named dimension requires bindings".to_string(),
            )
            .with_field("named_dim".to_string(), name.clone())
            .with_hint(format!(
                "Named dimension '{}' cannot be resolved without dimension bindings.",
                name
            )),
            ReshapeError::CannotInferDimension { reason } => Diagnostic::new(
                "E_RESHAPE_CANNOT_INFER",
                "Cannot infer dimension".to_string(),
            )
            .with_field("reason".to_string(), reason.clone())
            .with_hint(
                "Check that the reshape specification is valid and element counts match."
                    .to_string(),
            ),
            ReshapeError::ElementCountMismatch {
                input_elements,
                resolved_elements,
            } => Diagnostic::new(
                "E_RESHAPE_ELEMENT_MISMATCH",
                "Reshape element count mismatch".to_string(),
            )
            .with_field("input_elements".to_string(), input_elements.to_string())
            .with_field(
                "resolved_elements".to_string(),
                resolved_elements.to_string(),
            )
            .with_hint(
                "The reshape specification must preserve the total number of elements.".to_string(),
            ),
        }
    }
}

impl fmt::Display for ReshapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReshapeError::MultipleInferredDimensions => {
                write!(f, "Multiple -1 dimensions not allowed")
            }
            ReshapeError::ShapeReferenceOutOfBounds { index, input_rank } => {
                write!(
                    f,
                    "Shape reference @{} out of bounds (input has {} dims)",
                    index, input_rank
                )
            }
            ReshapeError::CannotReferenceLastDimension => {
                write!(f, "Cannot reference last dimension of empty shape")
            }
            ReshapeError::NamedDimensionNotFound { name } => {
                write!(f, "Named dimension '{}' not found in bindings", name)
            }
            ReshapeError::NamedDimensionRequiresBindings { name } => {
                write!(f, "Named dimension '{}' requires dimension bindings", name)
            }
            ReshapeError::CannotInferDimension { reason } => {
                write!(f, "Cannot infer dimension: {}", reason)
            }
            ReshapeError::ElementCountMismatch {
                input_elements,
                resolved_elements,
            } => {
                write!(
                    f,
                    "Reshape element count mismatch: input has {}, resolved shape has {}",
                    input_elements, resolved_elements
                )
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CapabilityError {
    Denied {
        capability: String,
        op: String,
        attempted_action: String,
    },
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CapabilityError::Denied {
                capability,
                op,
                attempted_action,
            } => {
                write!(
                    f,
                    "Capability '{}' denied for op '{}': {}",
                    capability, op, attempted_action
                )
            }
        }
    }
}

impl CapabilityError {
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            CapabilityError::Denied {
                capability,
                op,
                attempted_action,
            } => Diagnostic::new("E_CAPABILITY_DENIED", "Capability denied".to_string())
                .with_field("capability".to_string(), capability.clone())
                .with_field("op".to_string(), op.clone())
                .with_field("attempted_action".to_string(), attempted_action.clone())
                .with_hint(format!(
                    "Grant the '{}' capability to allow this operation.",
                    capability
                )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FileError {
    NotFound { path: String },
    InvalidUTF8 { path: String },
    IOError { path: String, io_error_kind: String },
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileError::NotFound { path } => {
                write!(f, "File not found: {}", path)
            }
            FileError::InvalidUTF8 { path } => {
                write!(f, "File contains invalid UTF-8: {}", path)
            }
            FileError::IOError {
                path,
                io_error_kind,
            } => {
                write!(f, "I/O error reading file '{}': {}", path, io_error_kind)
            }
        }
    }
}

impl FileError {
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            FileError::NotFound { path } => {
                Diagnostic::new("E_FILE_NOT_FOUND", "File not found".to_string())
                    .with_field("path".to_string(), path.clone())
                    .with_hint(
                        "Check that the file path is correct and the file exists.".to_string(),
                    )
            }
            FileError::InvalidUTF8 { path } => Diagnostic::new(
                "E_FILE_INVALID_UTF8",
                "File contains invalid UTF-8".to_string(),
            )
            .with_field("path".to_string(), path.clone())
            .with_hint("The file must contain valid UTF-8 text.".to_string()),
            FileError::IOError {
                path,
                io_error_kind,
            } => Diagnostic::new("E_FILE_IO_ERROR", "File I/O error".to_string())
                .with_field("path".to_string(), path.clone())
                .with_field("io_error_kind".to_string(), io_error_kind.clone())
                .with_hint("Check file permissions and disk space.".to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeError {
    Validation(ValidationError),
    Reshape(ReshapeError),
    CapabilityDenied(CapabilityError),
    File(FileError),
    Other(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::Validation(e) => write!(f, "Validation error: {}", e),
            RuntimeError::Reshape(e) => write!(f, "Reshape error: {}", e),
            RuntimeError::CapabilityDenied(e) => write!(f, "Capability error: {}", e),
            RuntimeError::File(e) => write!(f, "File error: {}", e),
            RuntimeError::Other(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl RuntimeError {
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            RuntimeError::Validation(e) => e.diagnostic(),
            RuntimeError::Reshape(e) => e.diagnostic(),
            RuntimeError::CapabilityDenied(e) => e.diagnostic(),
            RuntimeError::File(e) => e.diagnostic(),
            RuntimeError::Other(msg) => {
                Diagnostic::new("E_EXECUTION_FAILURE", "Execution failure".to_string())
                    .with_field("message".to_string(), msg.clone())
                    .with_hint("Check the model graph and input shapes.".to_string())
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<ValidationError> for RuntimeError {
    fn from(e: ValidationError) -> Self {
        RuntimeError::Validation(e)
    }
}

impl From<ReshapeError> for RuntimeError {
    fn from(e: ReshapeError) -> Self {
        RuntimeError::Reshape(e)
    }
}

impl From<CapabilityError> for RuntimeError {
    fn from(e: CapabilityError) -> Self {
        RuntimeError::CapabilityDenied(e)
    }
}

impl From<FileError> for RuntimeError {
    fn from(e: FileError) -> Self {
        RuntimeError::File(e)
    }
}

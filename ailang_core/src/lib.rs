pub mod app_runtime;
pub mod artifact;
pub mod backward;
mod capability;
pub mod dataset;
pub mod device;
mod diagnostic;
mod error;
pub mod execute_kernel;
pub mod forward;
pub mod forward_selected;
pub mod frontend;
pub mod gradcheck;
mod input_spec;
pub mod ir;
pub mod model;
pub mod optim;
pub mod param;
mod reshape_spec;
pub mod rng;
pub mod runtime;
pub mod state_machine;
pub mod tensor;

pub use app_runtime::{AppRuntime, EventExecutionResult, StateValue};
pub use backward::execute_backward;
pub use capability::{Capabilities, Capability};
pub use dataset::{load_tsv_dataset, Dataset};
pub use device::{
    default_cpu_device, CpuDevice, CpuTensor, Device, DeviceKind, DeviceRef, TensorStorage,
};
#[cfg(feature = "gpu-wgpu")]
pub use device::{GpuDevice, GpuTensor};
pub use diagnostic::Diagnostic;
pub use error::{CapabilityError, FileError, ReshapeError, RuntimeError, ValidationError};
pub use forward::{execute_forward, execute_forward_with_capabilities};
pub use forward_selected::execute_forward_selected_with_capabilities;
pub use gradcheck::gradcheck_matmul_relu_sum;
pub use input_spec::{infer_dtype_from_name, DimBindings, DimSpec, InputSpec};
pub use ir::{Graph, NodeId, Op};
pub use model::{export_model, load_model, load_model_legacy, ModelArtifact};
pub use optim::sgd_step;
pub use param::Param;
pub use reshape_spec::{ReshapeDim, ReshapeSpec};
pub use rng::SeededRng;
pub use runtime::{
    app_duplicate_error, app_missing_block_error, runtime_version_mismatch_error,
    tensor_to_trace_input, tensor_to_trace_output, trace_input_to_tensor, trace_output_to_tensor,
    trace_replay_mismatch_error, validate_app_structure, validate_program, ExecutionMode,
    ExecutionTrace, StateMachineTrace, StateTransition, TraceError, TraceInput, TraceOutput,
};
pub use state_machine::{execute_state_machine, StateMachineRuntime, StepResult};
pub use tensor::{Tensor, TokenIds};

# AILang Runtime

This document describes the AILang execution model, runtime semantics, and implementation details.

## Execution Model

AILang programs follow a three-phase execution model:

1. **Parse**: Source text is parsed into an Abstract Syntax Tree (AST)
2. **Lower**: AST is lowered into a Graph IR (nodes, edges, inputs, parameters)
3. **Execute**: Graph is executed deterministically with provided inputs

All phases are deterministic: same source + same seed + same inputs = identical results.

## Graph Intermediate Representation

AILang programs are compiled to a graph IR where:

* **Nodes** represent operations (ops)
* **Edges** represent data flow (tensor values)
* **Inputs** are provided at execution time
* **Parameters** are learned weights (initialized, then updated during training)

The graph is statically analyzable: all operations, shapes, and control flow are known at compile time.

## Forward Pass Execution

Forward pass execution evaluates the graph from inputs to outputs:

1. **Input binding**: Provided inputs are bound to input nodes
2. **Topological traversal**: Nodes are executed in topological order
3. **Operation execution**: Each node executes its operation, producing output tensors
4. **Output collection**: Final node outputs are collected as results

### Kernel Execution

Each operation (op) in the graph is executed by a kernel:

* **CPU kernels**: Implemented in Rust using `ndarray`
* **GPU kernels**: Implemented in WGSL (WebGPU Shading Language) for supported operations
* **Device dispatch**: Operations are routed to CPU or GPU based on device selection

### Selective Execution

AILang supports selective execution where only parts of the graph are evaluated:

* Conditional expressions (`if/then/else`) only execute the selected branch
* Capability checks occur before execution
* If a capability is denied, the entire operation (and dependent operations) are skipped

**Zero-execution guarantee**: Denied capabilities prevent any execution, not just the denied operation.

## Training Execution

Training execution adds gradient computation and parameter updates:

1. **Forward pass**: Evaluate the graph with current parameters
2. **Loss computation**: Compute loss from model outputs and labels
3. **Backward pass**: Compute gradients via automatic differentiation
4. **Parameter update**: Update parameters using gradients and learning rate
5. **Repeat**: Iterate for specified number of steps

### Gradient Computation

AILang uses automatic differentiation (autograd) to compute gradients:

* Gradients are computed via reverse-mode automatic differentiation
* Each operation has a corresponding backward implementation
* Gradients flow from loss back to parameters
* Gradient computation is deterministic

### Deterministic Batching

Training uses deterministic batching:

* Dataset is shuffled deterministically using the seed
* Batches are generated in a fixed order
* Same seed + same dataset = same batch sequence
* Batch boundaries are deterministic

## Device Execution

### CPU Execution

CPU execution uses Rust code with `ndarray` for tensor operations:

* All operations are supported on CPU
* Execution is sequential (single-threaded)
* Deterministic floating-point operations
* No external dependencies at runtime

### GPU Execution

GPU execution uses Metal (via `wgpu`) on Apple Silicon:

* Supported operations: `add`, `relu`, `matmul`
* Unsupported operations fall back to CPU
* Tensors are uploaded to GPU before execution
* Results are downloaded back to CPU
* GPU kernels are implemented in WGSL

**GPU limitations:**

* Not all operations are GPU-accelerated
* Device mismatch between inputs is not allowed
* GPU support requires building with `--features gpu-wgpu`
* GPU operations are deterministic for supported kernels

### Device Selection

Device selection occurs at execution time:

* Default device is CPU
* GPU device is selected via `--device gpu` flag
* Device is passed through the execution pipeline
* Operations route to appropriate kernels based on device

## Capability Enforcement

Capability checks occur before execution:

1. **Capability analysis**: Required capabilities are determined from the graph
2. **Capability check**: Required capabilities are checked against granted capabilities
3. **Execution decision**: If any required capability is missing, execution fails immediately

**Zero-execution guarantee**: Missing capabilities prevent execution before any computation occurs.

This prevents:
* Information leakage through timing
* Partial execution that reveals program structure
* Side effects from denied operations

## Determinism Guarantees

AILang guarantees deterministic execution:

### Seeded Execution

All randomness is controlled by a seed:

* Random number generation uses the seed
* Same seed produces same random values
* Dataset shuffling uses the seed
* Dropout uses seeded randomness

### Deterministic Operations

All operations are deterministic:

* Floating-point operations follow IEEE 754 (no non-deterministic reductions)
* GPU operations are deterministic for supported kernels
* No race conditions (single-threaded CPU, deterministic GPU)
* No operation order variations

### Deterministic Batching

Batching is deterministic:

* Dataset shuffling is seeded
* Batch boundaries are fixed
* Batch order is deterministic

## Execution Trace

AILang execution produces a deterministic trace:

* Input values (shapes and data)
* Intermediate computations (for debugging)
* Output values
* Capabilities used
* Device used

This trace enables:
* Debugging and reproducibility
* Audit trails
* Result verification

## Memory Management

AILang uses Rust's ownership model for memory safety:

* Tensors are owned by the execution context
* No manual memory management
* No memory leaks (guaranteed by Rust)
* Efficient tensor reuse where possible

## Performance Characteristics

### CPU Performance

* Single-threaded execution
* Optimized Rust code
* Efficient tensor operations via `ndarray`
* Memory-efficient graph evaluation

### GPU Performance

* Parallel execution on GPU
* Efficient kernel implementations
* Minimal CPU-GPU transfer overhead
* Experimental performance (not yet optimized)

## Future Improvements

Planned runtime improvements:

* Multi-threaded CPU execution
* Expanded GPU operation support
* Performance optimizations
* Better memory management
* Execution profiling


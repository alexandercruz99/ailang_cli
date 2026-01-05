# AILang

**A deterministic, capability-safe AI programming language and runtime.**

AILang is a domain-specific language for machine learning and application logic that prioritizes determinism, security, and auditability. Programs are defined in `.ail` files and compiled to a graph intermediate representation for execution.

## What is AILang

AILang combines:

* **Deterministic execution**: Same program + same seed + same data = identical results
* **Capability-gated security**: No ambient authority; explicit grants required for system resources
* **Single-binary deployment**: Rust-based runtime with no external dependencies
* **GPU-ready architecture**: CPU execution now, GPU acceleration available on Apple Silicon via Metal

AILang programs compile to a graph IR that executes deterministically. The language is designed for auditable ML pipelines, secure environments, and scenarios where reproducibility matters.

## Why AILang Exists

Python ML frameworks (PyTorch, TensorFlow) are powerful but come with trade-offs:

**Determinism problems**: Same code can produce different results across runs due to:
* Non-deterministic GPU operations
* Race conditions in multi-threaded execution
* Floating-point operation order variations
* Random initialization that isn't seeded

**Security concerns**: ML code often needs to run in sandboxed environments but Python frameworks:
* Have large attack surfaces
* Allow arbitrary file access
* Enable network operations without explicit permission
* Execute dynamic code that's hard to audit

**Boilerplate and implicit behavior**: Traditional ML frameworks require:
* Verbose model definitions
* Hidden execution paths (autograd, optimizers)
* Implicit data loading and preprocessing
* Configuration scattered across code and files

**Auditability gaps**: It's difficult to:
* Verify what a model actually does
* Reproduce exact training runs
* Audit security properties
* Export models with full execution semantics

AILang addresses these by providing:
* Guaranteed determinism through seeded execution
* Explicit capability model for security
* Minimal syntax for clarity
* Exportable graph representations for auditability

## Key Guarantees

### Determinism Contract

AILang guarantees that identical inputs produce identical outputs:

* Same `.ail` program file
* Same seed (`--seed N`)
* Same dataset (if used)
* Same device (CPU or GPU)

Produces **identical results** across runs, machines, and executions.

This guarantee applies to:
* Forward passes (inference)
* Training loops and gradient updates
* Evaluation metrics
* All intermediate computations

Violations of determinism are considered bugs.

### Capability Safety

AILang uses a capability-based security model:

* **No ambient authority**: Programs cannot access system resources by default
* **Explicit grants**: File access, clock access, and network access require CLI flags
* **Zero-execution guarantee**: If a required capability is denied, the program does not execute

Capabilities are granted via `--allow <capability>` flags.

### No Hidden IO

All I/O operations are explicit and capability-gated:

* Dataset loading requires `--allow fileread`
* `now()` requires `--allow clock`
* Network operations require `--allow network` (if supported)

If a capability is missing, execution fails before any computation occurs.

### No Dynamic Execution During Training

Training loops are statically analyzable:

* All control flow is deterministic
* Loops are compile-time unrolled
* No runtime code generation
* No dynamic graph modifications

This ensures training is reproducible and auditable.

## Quick Start

### Build

```bash
cargo build
```

For GPU support on Apple Silicon:

```bash
cargo build --features gpu-wgpu
```

### Run Inference

```bash
cargo run -p ailang_cli -- --run examples/easy_minimal.ail --seed 123
```

With GPU (if built with GPU features):

```bash
cargo run -p ailang_cli --features gpu-wgpu -- --run examples/easy_minimal.ail --seed 123 --device gpu
```

### Train a Model

```bash
cargo run -p ailang_cli -- --train examples/easy_train.ail --seed 123 --allow fileread
```

The `--allow fileread` flag is required because training with datasets requires reading files from disk. This is a security feature: by default, AILang blocks all file access.

## Easy Syntax Overview

AILang uses a simple, assignment-based syntax that removes boilerplate from traditional ML frameworks.

### Minimal Inference Example

```ail
const N = 4
const D = 3

model {
  x [N, D]
  param W [D, 1]
  y = matmul(x, W)
}
```

**What this does:**

* `x [N, D]`: Declares an input named `x` with shape `[N, D]`. No `input` keyword needed.
* `param W [D, 1]`: Declares a parameter (weight matrix) with shape `[D, 1]`.
* `y = matmul(x, W)`: Assigns the result of matrix multiplication to `y`.
* **Implicit output**: Since `y` is the last assignment, it becomes the model output.

**Key points:**

* Inputs are declared by name and shape: `name [dim1, dim2, ...]`
* Parameters use the `param` keyword: `param name [shape]`
* Assignments use `=`, no `let` keyword needed
* The last assignment (or a variable named `logits`) becomes the output
* No semicolons required (but allowed)

### Alias Functions

AILang provides syntactic aliases for common operations:

* `linear(x, W, b?)` → `matmul(x, W) + b` (bias optional)
* `meanpool(x)` → `mean_pool_time(x)`
* `xent(logits, labels)` → `cross_entropy(logits, labels)`

These are compile-time rewrites and add no runtime overhead.

### Training Example

```ail
const V = 20
const D = 8
const C = 2
const T = 6

model {
  tokens [B, T]
  labels [B]

  param E [V, D]
  param W [D, C]
  param b [C]

  logits = linear(meanpool(embedding(tokens, E)), W, b)
}

data {
  format = "jsonl"
  path = "examples/data/toy.jsonl"
  tokens = "tokens"
  labels = "labels"
  shuffle = true
  split = 0.8
}

train {
  loss = xent(logits, labels)
  steps = 200
  lr = 0.1
  batch = 4
}

eval {
  every = 20
  metrics = [loss, acc]
  split = "val"
}
```

**What this does:**

* `tokens [B, T]`: Input for token sequences (batch size `B`, sequence length `T`)
* `labels [B]`: Input for class labels (one per batch item)
* `embedding(tokens, E)`: Embeds tokens using embedding matrix `E`
* `meanpool(...)`: Pools over the time dimension
* `linear(..., W, b)`: Applies linear transformation with bias
* `logits`: Since this variable is named `logits`, it becomes the output (even if not last)
* `loss = xent(logits, labels)`: Defines loss using cross-entropy
* `eval` block: Runs evaluation every 20 steps on the validation split

## Control Flow

AILang supports conditional expressions for deterministic control flow.

### If/Then/Else Expressions

```ail
model {
  x [N, D]
  param threshold [1]
  
  y = if x > threshold then relu(x) else x
}
```

**Rules:**

* Expression-level: `if condition then true_expr else false_expr`
* Both branches must have the same shape and dtype
* Condition must be scalar (boolean)
* Allowed in `model` and `forward` blocks
* **Not allowed** in `dataset`, `train`, or `eval` blocks

### Boolean Expressions

AILang supports comparisons and logical operations:

**Comparisons:** `==`, `!=`, `<`, `<=`, `>`, `>=`

**Logical ops:** `and`, `or`, `not`

**Example:**

```ail
model {
  x [N]
  param threshold [1]
  
  is_positive = x > threshold
  is_valid = is_positive and x < 100.0
  y = if is_valid then x else 0.0
}
```

**Restrictions:**

* Boolean values cannot be stored as parameters
* Booleans cannot flow into tensor operations directly
* Conditionals are deterministic (no side effects)

## Iteration & Loops

AILang supports statically-bounded loops that are fully unrolled at compile time.

### For Loops (Collect Semantics)

```ail
model {
  x [B, D]
  param W [D, 1]
  
  h = for i in range(0, 4) do matmul(x, W) end
  logits = meanpool(h)
}
```

**What this does:**

* `for i in range(0, 4) do ... end`: Iterates 4 times with loop variable `i` (values 0, 1, 2, 3)
* **Collect semantics**: The `for` loop collects all iteration outputs and stacks them
* **Result shape**: If the body returns shape `[...S]`, the result is `[4, ...S]` (stacked along axis 0)
* **Loop variable**: `i` is available inside the loop body as a constant scalar

### Repeat Loops (Iterative Accumulation)

```ail
model {
  x [D]
  param W [D, D]
  
  y = repeat 3 times init x do relu(matmul(y, W)) end
}
```

**What this does:**

* `repeat N times init <expr> do <body> end`: Iterates N times with accumulator
* `init x`: Initial value for the accumulator
* Body expression must return same shape/dtype as accumulator
* Result is the final accumulator value after N iterations

### Reduce Loops

```ail
model {
  x [10, D]
  
  sum = reduce add over range(0, 10) do x[i] end
  max_val = reduce max over range(0, 10) do x[i] end
  min_val = reduce min over range(0, 10) do x[i] end
}
```

**Reduction operations:** `add`, `max`, `min`

**Rules:**

* Body returns tensor `[...S]`, result is `[...S]`
* Reduction operation is explicit
* `max` and `min` are supported for forward pass only (no gradient in training)

### Loop Restrictions

**Static bounds only:**

* Loop bounds must be compile-time constants (integer literals or `const` values)
* No data-dependent iteration counts
* No `while` loops

**Unroll limit:**

* Maximum 10,000 total iterations per program (sum across all loops)
* Enforced during compilation
* Prevents excessive code generation

**Where loops are allowed:**

* Allowed in `model` and `forward` blocks
* **Not allowed** in `dataset`, `train`, or `eval` blocks

**Why loops are restricted:**

Loops are compile-time unrolled for determinism and static analysis. Runtime loops would require:
* Dynamic graph construction (breaks determinism)
* Variable iteration counts (hard to audit)
* Hidden control flow (security risk)

Static unrolling ensures:
* Deterministic execution
* Full graph visibility
* Auditable computation

## Datasets

AILang supports loading training data from files in TSV or JSONL format.

### TSV Format

Tab-separated values with one example per line:

```
1	2	3	4	5	0
2	3	4	5	6	1
```

First columns are tokens, last column is the label.

### JSONL Format

JSON Lines format, one JSON object per line:

```json
{"tokens":[1,2,3,4,5],"label":1}
{"tokens":[2,3,4,5,6],"label":0}
```

### Dataset Block Options

```ail
data {
  format = "jsonl"        // "tsv" or "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"       // Field name (JSONL) or column indices (TSV)
  labels = "label"        // Field name (JSONL) or column index (TSV)
  shuffle = true          // Shuffle before splitting
  split = 0.8             // Train/validation split ratio (0.8 = 80% train, 20% val)
}
```

### Dataset Pipelines (Map/Filter/Batch)

AILang supports map/filter/batch pipelines for dataset transformations:

```ail
fn normalize(tokens) {
  tokens  // transformation logic
}

fn valid_label(labels) {
  labels > 0  // filter predicate
}

data {
  format = "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"
  labels = "label"
  map normalize(tokens)
  filter valid_label(labels)
  batch 32
}
```

**Pipeline semantics:**

* `map f(x)`: Applies function `f` to every sample (must return same rank)
* `filter f(x)`: Drops samples where `f(x)` returns `false` (must return boolean scalar)
* `batch N`: Groups samples into batches of size N
* Pipeline is applied **before** training/evaluation
* Functions must be user-defined and pure (no side effects)

**Capability requirements:**

* Map/filter functions cannot use `now()`, read files, or access environment
* Violation → `E_DATASET_PIPELINE_CAPABILITY_DENIED`
* Checks occur before any sample is processed

## Training

AILang supports training models with gradient descent.

### Loss Block

```ail
model {
  logits [B, C]
  labels [B]
  // ... model definition ...
}

train {
  loss = xent(logits, labels)
}
```

**Rules:**

* Loss must be a single scalar value
* Loss is computed per batch
* Loss is minimized via gradient descent

### Train Block

```ail
train {
  loss = xent(logits, labels)
  steps = 200              // Number of training steps
  lr = 0.1                 // Learning rate
  batch = 4                // Batch size
}
```

**Training algorithm:**

* Stochastic Gradient Descent (SGD)
* Deterministic batching (seeded shuffle)
* Gradient computation via automatic differentiation
* Parameter updates: `param = param - lr * grad`

**Labels handling:**

* Labels are provided via `labels` input (type: `labels`)
* Labels are matched to outputs by name
* Cross-entropy loss requires labels dtype

**Deterministic batching:**

* Batches are generated deterministically using the seed
* Same seed + same dataset = same batch order
* Shuffling (if enabled) is deterministic

### Training Output

When training runs, you'll see:

```
Inputs:
  tokens: token_ids
  labels: labels

Dataset:
  format: jsonl
  path: examples/data/toy.jsonl
  train_count: 8
  val_count: 2

Train:
  steps: 200
  lr: 0.1
  batch_size: 4

Eval:
  split: val
  every: 20
  metrics: ["loss", "accuracy"]

Starting training...
Step 0: loss = 0.693147
eval/loss = 0.693147
eval/accuracy = 0.5000
Step 20: loss = 0.542123
eval/loss = 0.501234
eval/accuracy = 0.7500
...
```

## Evaluation

AILang supports evaluation metrics during and after training.

### Eval Block

```ail
eval {
  every = 20              // Evaluate every N steps
  metrics = [loss, acc]   // Metrics to compute
  split = "val"           // "train" or "val"
}
```

**Supported metrics:**

* `loss`: Loss value on the selected split
* `acc`: Accuracy (for classification tasks)

**Evaluation semantics:**

* Evaluation runs during training at specified intervals
* Metrics are computed on the selected data split
* Results are printed to stdout
* Evaluation is deterministic (uses the same seed)

**Output format:**

```
eval/loss = 0.501234
eval/accuracy = 0.7500
```

## User-Defined Functions

AILang supports user-defined functions for code reuse.

### Function Definition

```ail
fn encoder(x, W, b) {
  relu(add(matmul(x, W), b))
}

model {
  x [B, D]
  param W [D, H]
  param b [H]
  
  h = encoder(x, W, b)
}
```

**Rules:**

* Functions are defined at the top level with `fn name(params) { body }`
* Functions are **compile-time expanded** (inlined) - zero runtime overhead
* No recursion allowed
* No side effects
* Functions cannot contain `train`, `eval`, `dataset`, or loops
* Functions are erased from exported models (only IR ops remain)
* Each function call produces a fresh subgraph

**Function arity:**

* Functions must have at least one parameter
* No default arguments
* Parameters are passed by value

## Capabilities Model

AILang uses a capability-based security model to control access to system resources.

### Available Capabilities

* `fileread`: Read files from disk (required for dataset loading)
* `filewrite`: Write files to disk (if supported)
* `clock`: Access system clock via `now()`
* `network`: Network operations (if supported)
* `env`: Environment variable access (if supported)
* `process`: Process execution (if supported)

### Granting Capabilities

Capabilities are granted via CLI flags:

```bash
--allow fileread    # Enable file reading
--allow clock       # Enable now() operation
--allow network     # Enable network operations
```

Multiple capabilities can be granted:

```bash
--allow fileread --allow clock
```

### What Happens When a Capability is Denied

If your program tries to use a capability that wasn't granted, execution stops immediately with a diagnostic:

```
E_DATASET_CAPABILITY_DENIED
Dataset loading requires FileRead capability
```

**Zero-execution guarantee**: If a required capability is missing, the program does not execute. This prevents:
* Information leakage through timing
* Partial execution that reveals program structure
* Side effects from denied operations

### Why Capabilities Exist

Capabilities prevent:
* Accidental data exposure (files, network)
* Malicious code from accessing system resources
* Non-deterministic behavior from system calls
* Security vulnerabilities in sandboxed environments

The capability model makes security properties explicit and auditable.

## Devices

AILang supports execution on different hardware devices.

### CPU (Default)

CPU execution uses optimized Rust code with `ndarray` for tensor operations.

```bash
cargo run -p ailang_cli -- --run example.ail --seed 123 --device cpu
```

CPU is the default device and requires no special flags.

### GPU (Apple Silicon)

GPU execution is available on Apple Silicon via Metal (through `wgpu`).

```bash
cargo run -p ailang_cli --features gpu-wgpu -- --run example.ail --seed 123 --device gpu
```

**GPU-supported operations:**

* `add` (element-wise addition)
* `relu` (ReLU activation)
* `matmul` (matrix multiplication)

**Limitations:**

* GPU support is experimental
* Not all operations are GPU-accelerated (others fall back to CPU)
* GPU support requires building with `--features gpu-wgpu`
* Determinism is preserved on GPU for supported operations

**GPU architecture:**

* GPU tensors are uploaded from CPU before execution
* Results are downloaded back to CPU
* GPU kernels are implemented in WGSL (WebGPU Shading Language)
* Device mismatch between inputs is not allowed (all inputs must be on the same device)

### Determinism Per Device

AILang guarantees determinism on both CPU and GPU:

* Same program + same seed + same device = identical results
* CPU and GPU results may differ due to floating-point precision, but each is deterministic
* GPU operations are deterministic for supported kernels

## What AILang Is NOT

**Not a replacement for PyTorch/TensorFlow:**

* AILang is designed for small to medium models
* Limited operation set compared to full frameworks
* GPU support is experimental and limited
* No distributed training

**Not a general-purpose language:**

* AILang is domain-specific for ML and application logic
* Loops are compile-time unrolled (not runtime)
* No recursion, no dynamic dispatch
* Focused on declarative model definitions

**Not a scripting language:**

* Programs are compiled, not interpreted
* No REPL or interactive execution
* No dynamic code loading
* Execution is deterministic and reproducible

**Not production-ready for all use cases:**

* Early-stage project
* Limited operation set
* GPU support is experimental
* Primarily for experimentation, education, and auditable ML

## Who Should Use AILang

**Audited ML pipelines:**

* Scenarios where model behavior must be verifiable
* Regulatory compliance requirements
* Security-sensitive deployments

**Education:**

* Teaching ML concepts without framework complexity
* Understanding how models actually work
* Learning about determinism and reproducibility

**Secure environments:**

* Sandboxed execution
* Capability-gated resource access
* Zero-trust ML deployments

**Research requiring determinism:**

* Reproducible experiments
* Debugging model behavior
* Verifying results across runs

**Embedded/edge ML:**

* Single-binary deployment
* No external dependencies
* Deterministic execution
* Small memory footprint

## Roadmap

**Current status:**

* Easy Syntax v1 (model blocks, assignments, aliases)
* Control flow (if/then/else, boolean expressions)
* Bounded iteration (for, repeat, reduce loops)
* User-defined functions
* Dataset loading (TSV, JSONL) with splits and shuffling
* Dataset pipelines (map, filter, batch)
* Training loops with evaluation
* Capability-gated security model
* Deterministic execution with seeds
* GPU support (Apple Silicon, experimental)

**Planned features:**

* Expanded GPU operation support
* More operations (attention, convolution, etc.)
* Performance optimizations
* Better error messages and diagnostics
* Application runtime improvements
* Documentation and tutorials

**Long-term vision:**

* Full GPU support across platforms
* Production-ready performance
* Extended operation set
* Advanced optimization passes
* Community and ecosystem growth

## Examples Index

AILang includes example programs demonstrating various features:

### Basic Examples

* `examples/easy_minimal.ail` - Minimal inference example using model block syntax
* `examples/easy_train.ail` - Training example with dataset, loss, and eval
* `examples/minimal.ail` - Minimal forward-only example
* `examples/op_sanity.ail` - Operation testing example

### Training Examples

* `examples/train_dataset.ail` - Training with TSV dataset
* `examples/train_dataset_jsonl.ail` - Training with JSONL dataset
* `examples/train_eval.ail` - Training with evaluation metrics
* `examples/train_minimal.ail` - Minimal training example

### Loop Examples

* `examples/easy_loops.ail` - Simple loop examples
* `examples/loops_collect.ail` - For loops with collect semantics
* `examples/loops_train_safe.ail` - Training with loops in model block
* `examples/loop_example.ail` - Various loop patterns
* `examples/reduce_example.ail` - Reduction loop examples

### Standard Library Examples

* `examples/stdlib_infer.ail` - Inference using standard library patterns
* `examples/stdlib_train.ail` - Training using standard library patterns

### Application Runtime Examples

* `examples/app_demo.ail` - State machine and application runtime example

### Error Examples

* `examples/errors/loop_bound_not_constant.ail` - Invalid loop bound
* `examples/errors/loop_not_allowed_in_train.ail` - Loop in train block
* `examples/errors/loop_var_out_of_scope.ail` - Loop variable scope error
* `examples/errors/repeat_requires_init.ail` - Missing init in repeat
* `examples/errors/stack_shape_mismatch.ail` - Shape mismatch in stack

### Test Examples

* `examples/test_selective.ail` - Selective execution testing
* `examples/test_train.ail` - Training test example

See individual example files for detailed usage patterns.

## License / Contribution

AILang is a Rust project. Contributions are welcome!

**Requirements for contributions:**

* All code must pass `cargo fmt` and `cargo test`
* New features require tests
* Follow existing code style and patterns
* Documentation updates are appreciated

**Getting started:**

1. Fork the repository
2. Make your changes
3. Add tests
4. Run `cargo test` to verify
5. Submit a pull request

For documentation-only changes (like this README), submit directly without code changes.

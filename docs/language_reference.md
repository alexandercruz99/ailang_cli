# AILang Language Reference v1.0

**Status:** Normative Specification  
**Version:** 1.0  
**Date:** 2025-01-03

This document is a formal specification of the AILang programming language. It defines the syntax, semantics, and behavior of AILang v1.0 programs.

---

## 1. Language Overview

### 1.1 What is AILang

AILang is a deterministic, capability-gated programming language for machine learning and application logic. Programs are defined in text files (`.ail` extension) and compiled to a graph intermediate representation (IR) for execution.

### 1.2 Execution Model

AILang programs follow a three-phase execution model:

1. **Parse**: Source text is parsed into an Abstract Syntax Tree (AST)
2. **Lower**: AST is lowered into a Graph IR (nodes, edges, inputs, parameters)
3. **Execute**: Graph is executed deterministically with provided inputs

All phases are deterministic: same source + same seed + same inputs = identical results.

### 1.3 Safety Model

AILang uses a capability-gated security model:

* **No ambient authority**: Programs cannot access system resources by default
* **Explicit capabilities**: File access, clock access, and network access require explicit grants via CLI flags
* **Zero-execution guarantee**: If a required capability is denied, the program does not execute

Capabilities are granted via CLI flags:
* `--allow fileread`: Enables file reading (required for dataset loading)
* `--allow clock`: Enables `now()` operation
* `--allow network`: Enables network operations (if supported)

### 1.4 Determinism Guarantees

AILang guarantees deterministic execution:

* Same program (`.ail` file)
* Same seed (`--seed N`)
* Same dataset (if used)
* **= Identical results**

This guarantee applies to:
* Forward passes (inference)
* Training loops
* Evaluation metrics
* All intermediate computations

**Violations of determinism are considered bugs.**

---

## 2. Program Structure

### 2.1 Top-Level Blocks

An AILang program consists of:

```
program ::= const_decl* block+
```

Where `block` is one of:
* `model { }` or `forward { }` (exactly one required)
* `train { }` (optional)
* `eval { }` (optional)
* `dataset { }` or `data { }` (optional)

### 2.2 Constants

**Syntax:**
```
const_decl ::= const IDENT = (integer | float | string | boolean)
```

**Rules:**
* Constants are compile-time values
* Can be referenced in shapes and expressions
* Must be declared before use
* No type annotations required

**Example:**
```
const V = 1000
const D = 64
const LR = 0.001
```

### 2.3 Model Block

**Syntax:**
```
model_block ::= model { statement* }
```

**Statements:**
* Input declarations: `IDENT [shape]` or `input IDENT: [shape]`
* Parameter declarations: `param IDENT: [shape]`
* Assignments: `IDENT = expr`

**Rules:**
* Exactly one `model` or `forward` block required per program
* Cannot have both `model` and `forward` blocks
* Must contain at least one assignment
* Implicit output: Last assignment (or variable named `logits`) becomes `forward_output`

**Diagnostics:**
* `E_MODEL_EMPTY`: Model block has no statements (fields: `block="model"`)
* `E_DUPLICATE_MODEL_BLOCK`: Both `model` and `forward` blocks present
* `E_DUPLICATE_FORWARD_BLOCK`: Both `model` and `forward` blocks present

**Example:**
```
model {
  tokens [B, T]
  labels [B]
  param E [V, D]
  param W [D, C]
  logits = linear(meanpool(embedding(tokens, E)), W)
}
```

### 2.4 Forward Block (Legacy)

**Syntax:**
```
forward_block ::= forward { let_stmt* return_stmt }
```

**Rules:**
* Alternative to `model` block (legacy syntax)
* Must use `let` for bindings
* Must use `return` for output
* Semicolons required

**Status:** Deprecated in favor of `model` block. Still supported for backward compatibility.

### 2.5 Train Block

**Syntax:**
```
train_block ::= train { field* }
```

**Fields:**
* `loss = expr` (required)
* `steps = integer`
* `lr = float`
* `batch = integer`

**Rules:**
* Optional block
* Must appear after `model`/`forward` block
* `loss` assignment is required if block is present
* Loss expression must be a scalar

**Diagnostics:**
* `E_TRAIN_REQUIRES_LOSS`: Train block exists but no loss assignment (fields: `block="train"`)

**Example:**
```
train {
  loss = xent(logits, labels)
  steps = 200
  lr = 0.1
  batch = 4
}
```

### 2.6 Eval Block

**Syntax:**
```
eval_block ::= eval { field* }
```

**Fields:**
* `every = integer` (evaluation frequency in steps)
* `metrics = [IDENT, ...]` (list of metric names)
* `split = ("train" | "val")` (default: `"val"`)

**Rules:**
* Optional block
* Must appear after `train` block
* `metrics` field is required
* Supported metrics: `loss`, `acc` (alias for `accuracy`)

**Example:**
```
eval {
  every = 20
  metrics = [loss, acc]
  split = "val"
}
```

### 2.7 Dataset Block

The `dataset` block (or `data` block) declares a dataset for training and evaluation. It supports loading from TSV or JSONL files, optional shuffling and splitting, and **pipeline transformations** (map, filter, batch).

**Basic Syntax:**
```ail
dataset {
  format = "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"
  labels = "label"
  shuffle = true
  split = 0.8
}
```

**With Pipeline:**
```ail
dataset {
  format = "jsonl"
  path = "data/train.jsonl"
  tokens = "tokens"
  labels = "label"
  map normalize(tokens)
  filter valid_label(labels)
  batch 32
}
```

**Fields:**
* `format`: Dataset format (`"tsv"` or `"jsonl"`)
* `path`: Path to dataset file
* `tokens`: Field name for token sequences (JSONL) or column index (TSV)
* `labels`: Field name for labels (JSONL) or column index (TSV)
* `shuffle`: Optional boolean (default: `false`)
* `split`: Optional float (0.0-1.0, train/val split ratio)

**Pipeline Stages:**
* `map <function>(<arg>)`: Apply transformation function to each sample
* `filter <function>(<arg>)`: Filter samples using predicate function
* `batch <N>`: Group samples into batches of size N

**Pipeline Rules:**
* Map/filter functions must be user-defined (not built-ins)
* Functions must take exactly 1 argument (the sample)
* Filter functions must return a boolean scalar
* Pipeline stages are applied in order: map → filter → batch
* Pipeline is applied **before** training/evaluation

**Capability Requirements:**
* Dataset loading requires `--allow fileread`
* Pipeline functions cannot use `now()`, file I/O, or network operations

**Diagnostics:**
* `E_DATASET_MAP_NOT_FOUND`: Map function not defined
* `E_DATASET_FILTER_NOT_FOUND`: Filter function not defined
* `E_DATASET_FILTER_NOT_BOOLEAN`: Filter function doesn't return boolean
* `E_DATASET_MAP_SHAPE_MISMATCH`: Map function changes tensor rank
* `E_DATASET_PIPELINE_CAPABILITY_DENIED`: Pipeline function uses denied capability

### 2.8 Dataset Pipelines (v1)

**Syntax:**
```
dataset_block ::= (dataset | data) { field* }
```

**Fields:**
* `format = ("tsv" | "jsonl")` (required)
* `path = string` (required)
* `tokens = string` (required: field name or column spec)
* `labels = string` (required: field name or column spec)
* `shuffle = boolean` (optional, default: `false`)
* `split = float` (optional, default: `1.0`, train/val split ratio)

**Rules:**
* Optional block
* Can appear after `model`/`forward` or after `train`
* Requires `--allow fileread` capability
* Deterministic: Same seed → same data order

**Diagnostics:**
* `E_DATASET_CAPABILITY_DENIED`: Dataset loading requires FileRead capability
* `E_FILE_NOT_FOUND`: File not found (fields: `path`)
* `E_FILE_INVALID_UTF8`: File contains invalid UTF-8 (fields: `path`)
* `E_FILE_IO_ERROR`: File I/O error (fields: `path`, `io_error_kind`)

**Example:**
```
data {
  format = "jsonl"
  path = "examples/data/toy.jsonl"
  tokens = "tokens"
  labels = "labels"
  shuffle = true
  split = 0.8
}
```

### 2.9 State Machines (v1)

**Syntax:**
```
state_machine_block ::= state_machine IDENT { state_decl+ }
state_decl ::= state IDENT { state_body* transition* (end)? }
state_body ::= (let_binding | emit_statement)+
transition ::= on IDENT -> IDENT | if expr -> IDENT | else -> IDENT
emit_statement ::= emit expr
```

**Rules:**
* Optional top-level block
* States are named, immutable blocks
* Each state can have:
  * Local bindings (`let name = expr`)
  * Emit statements (`emit "value"`)
  * Transitions (`on event -> State`, `if condition -> State`, `else -> State`)
  * `end` keyword (terminates state machine)
* First state declared is the initial state
* Transitions evaluated in order; first matching transition fires
* `on` transitions match events (e.g., `start`, `input`, `tick`)
* `if` transitions evaluate boolean conditions
* `else` transition fires if no preceding `if` fired
* Exactly one transition must fire, or state must have `end`
* Deterministic execution: same inputs → same state sequence

**Diagnostics:**
* `E_STATE_DUPLICATE`: Duplicate state name (fields: `state_name`)
* `E_STATE_UNREACHABLE`: State cannot be reached from initial state (fields: `state_name`)
* `E_STATE_NO_TRANSITION`: State has no transitions and no `end` (fields: `state_name`)
* `E_TRANSITION_CONFLICT`: Invalid transition ordering (fields: `state_name`, `transition_index`)
* `E_INVALID_EVENT`: Unknown event name (fields: `event_name`, `state_name`)
* `E_STATE_MACHINE_EMPTY`: State machine has no states

**Example:**
```
state_machine App {
  state Idle {
    on start -> Running
  }

  state Running {
    emit "running"
    on input -> Processing
    if score > 0.8 -> Approved
    else -> Rejected
  }

  state Approved {
    emit "approved"
    end
  }

  state Rejected {
    emit "rejected"
    end
  }
}
```

**Semantics:**
* State machine execution is step-based and deterministic
* Each step:
  1. Enter current state
  2. Execute state body (local bindings, emit statements)
  3. Evaluate transitions in order
  4. Move to next state (or end)
* Emitted values are collected by the runtime
* State machines do not support loops, recursion, or side effects in state bodies

### 2.10 Block Ordering Rules

**Valid order:**
1. `const*` (anywhere, typically at top)
2. `model` or `forward` (exactly one, required)
3. `train` (optional, must appear after model/forward)
4. `dataset` or `data` (optional, can appear after model/forward or after train)
5. `eval` (optional, must appear after train)

**Invalid:**
* Both `model` and `forward`: Error `E_DUPLICATE_MODEL_BLOCK` or `E_DUPLICATE_FORWARD_BLOCK`
* `train` before `model`/`forward`: Syntax error
* `eval` before `train`: Syntax error

---

## 3. Types and DTypes

### 3.1 Data Types

AILang has three data types:

1. **`tensor`**: General-purpose multi-dimensional array
2. **`token_ids`**: Special type for token sequences (used by `embedding`)
3. **`labels`**: Special type for classification labels (used by `cross_entropy`)

### 3.2 Dtype Inference

Dtype is inferred from input name using the canonical mapping:

```
"tokens" => "token_ids"
"labels" => "labels"
_ => "tensor"
```

This mapping is **the single source of truth**. It is implemented in `infer_dtype_from_name()` and used by:
* Lowering (when creating `InputSpec`)
* CLI (when generating synthetic inputs)

**No rank-based inference.** Dtype is determined solely by name.

### 3.3 Type Errors

#### Embedding Requires Token IDs

**Operation:** `embedding(tokens, weight)`

**Requirement:** First argument must be an input with dtype `token_ids`.

**Error:** `E_EMBEDDING_REQUIRES_TOKEN_IDS`

**Fields:**
* `input_name`: Name of the input argument
* `received_dtype`: Dtype that was received

**Example:**
```
# Invalid: x has dtype "tensor"
model {
  x [B, T]
  param E [V, D]
  h = embedding(x, E)  # Error: E_EMBEDDING_REQUIRES_TOKEN_IDS
}

# Valid: tokens has dtype "token_ids"
model {
  tokens [B, T]
  param E [V, D]
  h = embedding(tokens, E)  # OK
}
```

#### Cross-Entropy Requires Labels

**Operation:** `cross_entropy(logits, labels)` or `xent(logits, labels)`

**Requirement:** Second argument must be an input with dtype `labels`.

**Error:** `E_LABELS_REQUIRED`

**Example:**
```
# Invalid: y has dtype "tensor"
model {
  logits [B, C]
  y [B]
  loss = xent(logits, y)  # Error: E_LABELS_REQUIRES_LABELS
}

# Valid: labels has dtype "labels"
model {
  logits [B, C]
  labels [B]
  loss = xent(logits, labels)  # OK
}
```

---

## 4. Expressions

### 4.1 Expression Grammar

```
expr ::= IDENT                    # Variable reference
       | call                     # Function call
       | expr + expr              # Addition
       | expr - expr              # Subtraction
       | expr * expr              # Multiplication
       | (expr)                   # Parenthesized

call ::= IDENT ( arg_list? )
arg_list ::= expr (, expr)*
```

### 4.2 Assignments

**Syntax:**
```
assignment ::= IDENT = expr
```

**Rules:**
* No `let` keyword required in `model` blocks
* Semicolons optional
* Variables are immutable after assignment
* Last assignment (or variable named `logits`) becomes output

### 4.3 Function Calls

**Syntax:**
```
call ::= IDENT ( arg_list? )
```

**Rules:**
* No user-defined functions
* Only built-in operations
* Arguments are expressions
* No keyword arguments (except for some operations like `softmax(axis=...)`)

### 4.4 Operation Aliases

The following aliases are expanded during lowering:

* `meanpool(x)` → `mean_pool_time(x)`
* `xent(logits, labels)` → `cross_entropy(logits, labels)`
* `linear(x, W, b)` → `matmul(x, W) + b`

Aliases are **frontend sugar only**. They produce the same IR as the expanded form.

### 4.5 Restrictions

**Not supported:**
* User-defined functions
* Control flow (if/else, loops, recursion)
* Mutation (all values are immutable)
* Dynamic typing
* Runtime reflection

---

## 5. Operations Reference

### 5.1 Math Operations

#### `matmul(a, b)`

**Signature:** `matmul(tensor [M, K], tensor [K, N]) -> tensor [M, N]`

**Inputs:**
* `a`: 2D tensor `[M, K]`
* `b`: 2D tensor `[K, N]`

**Output:** 2D tensor `[M, N]`

**Deterministic:** Yes

**Gradient:** Supported

**Errors:**
* `E_INVALID_ARGUMENTS`: Wrong number of arguments (fields: `function`, `expected`, `got`)

#### `add(a, b)`

**Signature:** `add(tensor, tensor) -> tensor`

**Inputs:**
* `a`: Any tensor (broadcast-compatible)
* `b`: Any tensor (broadcast-compatible)

**Output:** Element-wise sum (same shape as inputs after broadcasting)

**Deterministic:** Yes

**Gradient:** Supported

#### `sub(a, b)`

**Signature:** `sub(tensor, tensor) -> tensor`

**Inputs:**
* `a`: Any tensor (broadcast-compatible)
* `b`: Any tensor (broadcast-compatible)

**Output:** Element-wise difference

**Deterministic:** Yes

**Gradient:** Supported

#### `mul(a, b)`

**Signature:** `mul(tensor, tensor) -> tensor`

**Inputs:**
* `a`: Any tensor (broadcast-compatible)
* `b`: Any tensor (broadcast-compatible)

**Output:** Element-wise product

**Deterministic:** Yes

**Gradient:** Supported

### 5.2 Activation Functions

#### `relu(x)`

**Signature:** `relu(tensor) -> tensor`

**Input:** Any tensor

**Output:** Same shape, `max(0, x)` element-wise

**Deterministic:** Yes

**Gradient:** Supported

#### `softmax(x, axis?)`

**Signature:** `softmax(tensor, axis?: int) -> tensor`

**Input:**
* `x`: Any tensor
* `axis`: Optional integer (default: last dimension)

**Output:** Same shape, softmax along axis

**Deterministic:** Yes

**Gradient:** Supported

### 5.3 Neural Network Layers

#### `embedding(tokens, weight)`

**Signature:** `embedding(token_ids [B, T], param [V, D]) -> tensor [B, T, D]`

**Inputs:**
* `tokens`: Input with dtype `token_ids` (required)
* `weight`: Parameter tensor `[V, D]` (embedding matrix)

**Output:** `[B, T, D]` where `B` is batch, `T` is sequence length, `D` is embedding dimension

**Deterministic:** Yes

**Gradient:** Supported

**Errors:**
* `E_EMBEDDING_REQUIRES_TOKEN_IDS`: First argument is not `token_ids` (fields: `input_name`, `received_dtype`)
* `E_INVALID_ARGUMENTS`: Wrong number of arguments

#### `linear(x, W, b)`

**Signature:** `linear(tensor [B, D_in], param [D_in, D_out], param [D_out]) -> tensor [B, D_out]`

**Inputs:**
* `x`: `[B, D_in]`
* `W`: `[D_in, D_out]`
* `b`: `[D_out]`

**Output:** `[B, D_out]`

**Note:** Alias for `matmul(x, W) + b`

**Deterministic:** Yes

**Gradient:** Supported

**Errors:**
* `E_INVALID_ARGUMENTS`: Wrong number of arguments (expected 3)

#### `dropout(x, p)`

**Signature:** `dropout(tensor, float) -> tensor`

**Inputs:**
* `x`: Any tensor
* `p`: Dropout probability (float literal or const)

**Output:** Same shape (training-only, identity in inference)

**Deterministic:** No (uses RNG, but seeded)

**Gradient:** Supported

### 5.4 Pooling

#### `meanpool(x)` or `mean_pool_time(x)`

**Signature:** `meanpool(tensor [B, T, D]) -> tensor [B, D]`

**Input:** `[B, T, D]`

**Output:** `[B, D]` (mean over time dimension)

**Deterministic:** Yes

**Gradient:** Supported

**Note:** `meanpool` is an alias for `mean_pool_time`

### 5.5 Loss Functions

#### `cross_entropy(logits, labels)` or `xent(logits, labels)`

**Signature:** `cross_entropy(tensor [B, C], labels [B]) -> scalar`

**Inputs:**
* `logits`: `[B, C]` (batch, classes)
* `labels`: Input with dtype `labels` (required)

**Output:** Scalar loss

**Deterministic:** Yes

**Gradient:** Supported

**Errors:**
* `E_LABELS_REQUIRED`: Labels input not found
* `E_INVALID_ARGUMENTS`: Wrong number of arguments

**Note:** `xent` is an alias for `cross_entropy`

### 5.6 Tensor Manipulation

#### `concat(axis, a, b)`

**Signature:** `concat(int, tensor, tensor) -> tensor`

**Inputs:**
* `axis`: Integer literal (currently only `1` supported)
* `a`, `b`: Tensors to concatenate

**Output:** Concatenated tensor along axis

**Deterministic:** Yes

**Gradient:** Supported

**Errors:**
* `E_INVALID_ARGUMENTS`: Wrong number of arguments or axis != 1

#### `slice_rows(x, start, len)`

**Signature:** `slice_rows(tensor [N, D], int, int) -> tensor [len, D]`

**Inputs:**
* `x`: 2D tensor `[N, D]`
* `start`: Integer literal (start index)
* `len`: Integer literal (length)

**Output:** `[len, D]`

**Deterministic:** Yes

**Gradient:** Supported

#### `gather_rows(x, indices)`

**Signature:** `gather_rows(tensor [N, D], token_ids) -> tensor [B, D]`

**Inputs:**
* `x`: 2D tensor `[N, D]`
* `indices`: TokenIds (indices into first dimension)

**Output:** `[B, D]` where `B` is batch size from indices

**Deterministic:** Yes

**Gradient:** Supported

#### `reshape(x, shape)`

**Signature:** `reshape(tensor, ReshapeSpec) -> tensor`

**Inputs:**
* `x`: Any tensor
* `shape`: ReshapeSpec (see Section 6.1)

**Output:** Reshaped tensor (same element count)

**Deterministic:** Yes

**Gradient:** Supported

**Errors:**
* `E_RESHAPE_MULTIPLE_INFERRED`: Multiple `-1` dimensions
* `E_RESHAPE_REF_OUT_OF_BOUNDS`: Shape reference index out of bounds (fields: `reference_index`, `input_rank`)
* `E_RESHAPE_NAMED_DIM_NOT_FOUND`: Named dimension not found (fields: `named_dim`)
* `E_RESHAPE_ELEMENT_MISMATCH`: Element count mismatch (fields: `input_elements`, `resolved_elements`)

### 5.7 Special Operations

#### `now()`

**Signature:** `now() -> scalar`

**Inputs:** None

**Output:** Scalar float (current time)

**Deterministic:** No (depends on system clock)

**Gradient:** Not supported

**Capability:** Requires `--allow clock`

**Errors:**
* `E_CAPABILITY_DENIED`: Clock capability not granted (fields: `capability`, `op`, `attempted_action`)

#### `read_file_text(path)`

**Signature:** `read_file_text(string) -> tensor [N]`

**Inputs:**
* `path`: String literal or const

**Output:** `[N]` (hash of file contents as float vector)

**Deterministic:** Yes (for same file contents)

**Gradient:** Not supported

**Capability:** Requires `--allow fileread`

**Errors:**
* `E_CAPABILITY_DENIED`: FileRead capability not granted
* `E_FILE_NOT_FOUND`: File not found (fields: `path`)
* `E_FILE_INVALID_UTF8`: File contains invalid UTF-8 (fields: `path`)
* `E_FILE_IO_ERROR`: File I/O error (fields: `path`, `io_error_kind`)

---

## 6. Shape System

### 6.1 Shape Syntax

Shapes are lists of dimensions enclosed in brackets:

```
shape ::= [dim (, dim)*]
dim ::= IDENT | integer | -1
```

**Examples:**
* `[B, T]` - Named dimensions
* `[32, 128]` - Literal dimensions
* `[B, T, -1]` - Inferred dimension (must be exactly one `-1`)

### 6.2 Named Dimensions

Named dimensions (e.g., `B`, `T`, `D`) are bound from input signatures:

* When an input declares `tokens [B, T]`, `B` and `T` are bound to the actual dimensions
* These bindings are used to resolve shapes in parameters and operations
* **Conflict rule**: If the same named dimension appears in multiple inputs with different values, error `E_NAMED_DIM_CONFLICT`

**Error:** `E_NAMED_DIM_CONFLICT` (fields: `named_dim`, `previous_value`, `new_value`, `input`)

### 6.3 Shape References (Reshape)

In reshape operations, shapes can reference input dimensions:

* `@0`, `@1`, `@2`, ... - Reference dimension at index
* `@last` - Reference last dimension
* Named dimensions: `"B"`, `"T"`, etc. (must be bound from inputs)
* `-1` - Inferred dimension (exactly one allowed)
* `mul(@0, @1)` - Multiply two dimensions

**Errors:**
* `E_RESHAPE_MULTIPLE_INFERRED`: Multiple `-1` dimensions
* `E_RESHAPE_REF_OUT_OF_BOUNDS`: Reference index out of range
* `E_RESHAPE_NAMED_DIM_NOT_FOUND`: Named dimension not bound
* `E_RESHAPE_ELEMENT_MISMATCH`: Element count mismatch

### 6.4 Shape Validation

At runtime, input shapes are validated:

* **Rank mismatch**: Error `E_INPUT_RANK_MISMATCH` (fields: `input`, `expected_rank`, `received_rank`)
* **Dimension mismatch**: Error `E_INPUT_DIM_MISMATCH` (fields: `input`, `dimension`, `expected`, `received`)
* **Named dimension conflict**: Error `E_NAMED_DIM_CONFLICT` (fields: `named_dim`, `previous_value`, `new_value`, `input`)

---

## 7. Dataset Block Specification

### 7.1 Supported Formats

**TSV (Tab-Separated Values):**
* One example per line
* Tab-separated columns
* Last column is label, preceding columns are tokens

**JSONL (JSON Lines):**
* One JSON object per line
* Fields: `tokens` (array), `label` (integer)

### 7.2 Required Fields

* `format`: `"tsv"` or `"jsonl"`
* `path`: Path to dataset file (string)
* `tokens`: Field name for tokens (JSONL) or column indices (TSV)
* `labels`: Field name for labels (JSONL) or column index (TSV)

### 7.3 Optional Fields

* `shuffle`: `true` or `false` (default: `false`)
* `split`: Float between 0 and 1 (default: `1.0`, train/val split ratio)

### 7.4 Determinism Rules

* Same seed → same data order
* Shuffling uses seed (deterministic)
* Train/val split uses seed (deterministic)

### 7.5 Capability Requirements

* Requires `--allow fileread` capability
* If denied: Error `E_DATASET_CAPABILITY_DENIED`
* Zero-execution guarantee: Program does not execute if capability denied

### 7.6 Diagnostics

* `E_DATASET_CAPABILITY_DENIED`: Dataset loading requires FileRead capability
* `E_FILE_NOT_FOUND`: File not found (fields: `path`)
* `E_FILE_INVALID_UTF8`: File contains invalid UTF-8 (fields: `path`)
* `E_FILE_IO_ERROR`: File I/O error (fields: `path`, `io_error_kind`)

---

## 8. Training Semantics

### 8.1 Training Loop

Training executes the following loop:

1. For each step (1 to `steps`):
   a. Sample batch from dataset (deterministic, seeded)
   b. Forward pass: compute loss
   c. Backward pass: compute gradients
   d. Update parameters: `param = param - lr * gradient`
   e. If `every` steps (from `eval` block): run evaluation

### 8.2 Loss Requirements

* Must be assigned in `train` block: `loss = expr`
* Loss must be a scalar (0D tensor)
* Loss is minimized via gradient descent

**Error:** `E_TRAIN_REQUIRES_LOSS` if `train` block exists but no `loss` assignment (fields: `block="train"`)

### 8.3 Gradient Flow

* Gradients are computed via automatic differentiation
* All operations that support gradients have gradients computed
* Gradient flow follows the computation graph

### 8.4 Parameter Updates

Parameters are updated via **stochastic gradient descent (SGD)**:

* Update rule: `param = param - lr * gradient`
* Learning rate: `lr` from `train` block
* Batch size: `batch` from `train` block

### 8.5 Deterministic Batching

* Batches are sampled deterministically using the seed
* Same seed → same batches → same training trajectory
* Shuffling (if enabled) uses seed (deterministic)

---

## 9. Evaluation Semantics

### 9.1 Eval Scheduling

Evaluation runs at regular intervals:

* `every = N`: Evaluate every N training steps
* Evaluation does not update parameters
* Evaluation uses the current parameter state

### 9.2 Metrics

Supported metrics:

* `loss`: Loss value (scalar)
* `acc` or `accuracy`: Classification accuracy (0-1)

Metrics are computed on the specified split (`train` or `val`).

### 9.3 Split Selection

Evaluation can run on:

* `split = "train"`: Training split
* `split = "val"`: Validation split (default)

### 9.4 Output Format Guarantees

Evaluation output is printed with `eval/` prefix:

```
eval/loss = 0.693147
eval/accuracy = 0.5000
```

Output format is stable and deterministic.

---

## 10. Error Model

### 10.1 Diagnostics

All errors in AILang are reported as **diagnostics**:

* **Code**: Stable string identifier (e.g., `E_EMBEDDING_REQUIRES_TOKEN_IDS`)
* **Title**: Human-readable title
* **Fields**: Key-value pairs (structured data)
* **Hint**: Optional suggestion (not guaranteed)

### 10.2 Stable Error Codes

Error codes are **stable** across versions:

* Not changed without a major version bump
* Documented in this reference
* Used in tests (not string matching)

### 10.3 Required Fields

Each diagnostic has required fields that are guaranteed:

* Fields are documented in this reference
* Fields are stable (same fields for same error)
* Tests should assert fields, not prose

### 10.4 Execution Guarantees

**No panics**: AILang never panics. All errors are reported as diagnostics.

**No partial execution**: If a capability is denied or a fatal error occurs, the program does not execute. This prevents information leakage through timing or partial results.

**Zero-execution guarantee**: Denied capabilities prevent any computation.

---

## 10. Application Runtime (Event-Driven Model)

AILang supports an event-driven application runtime model that allows programs to define persistent state and event handlers, enabling deterministic application logic beyond pure ML inference.

### 10.1 Syntax

**State Block:**
```ail
state {
  counter: int = 0
  score: float = 0.0
  logits: tensor
}
```

State variables must have explicit types (`int`, `float`, `tensor`) and may have optional initial values.

**Event Handlers:**
```ail
on start {
  counter = counter + 1
  emit initialized
}

on predict {
  last_score = logits
  counter = counter + 1
  emit result
}
```

Event handlers execute code in response to events. They can:
- Assign to state variables
- Emit events (synchronous)
- Use `let` bindings for local computations

**Event Emission:**
```ail
emit <event_name>
```

Events are emitted synchronously and must reference defined event handlers.

### 10.2 Semantics

**State Persistence:**
- State variables persist across event executions
- State is initialized before the first event
- State changes are committed after event handler execution

**Event Execution:**
1. App loads → state initialized
2. `on start` runs (if present)
3. External events are injected by host (CLI for now)
4. Each event:
   - Snapshot state
   - Execute handler
   - Commit state
5. Deterministic replay guaranteed by seed

**Event Emission:**
- Events are emitted synchronously
- Emitted events must be defined
- Cycle detection: no circular event emissions allowed
- Event graph must be acyclic

### 10.3 Validation

The compiler validates:
- **State variables**: No duplicate declarations (`E_STATE_REDECLARED`)
- **State assignments**: All assignments reference declared variables (`E_STATE_NOT_DECLARED`)
- **Event handlers**: No duplicate handlers (`E_EVENT_DUPLICATE`)
- **Event emission**: All emitted events must be defined (`E_EVENT_NOT_DEFINED`)
- **Cycle detection**: Event emission graph must be acyclic (`E_EVENT_CYCLE_DETECTED`)

### 10.4 Restrictions

- Event handlers are pure and deterministic
- No loops with unbounded iteration in event handlers
- No dataset access in event handlers
- State variables are immutable unless explicitly reassigned
- No side effects outside state

### 10.5 CLI Integration

Run apps with event handlers:
```bash
cargo run -p ailang_cli -- --run-app examples/app_demo.ail --event start --seed 123
```

The CLI executes the specified event and prints:
- Execution result
- Emitted events
- State changes
- Current state

## 11. Stability Guarantees

### 11.1 What is Stable in v1.0

The following are **stable** and will not change without a major version bump:

* Syntax of all blocks (`model`, `forward`, `train`, `eval`, `dataset`)
* Operation signatures and semantics
* Diagnostic error codes and required fields
* Dtype inference rules (name-based only)
* Determinism guarantees
* Capability model

### 11.2 What May Change

The following may change in future versions:

* New operations may be added
* New block types may be added
* Syntax extensions (e.g., `repeat` blocks, function definitions)
* New diagnostics may be added (but existing ones remain stable)

### 11.3 Backward Compatibility

AILang v1.0 programs will continue to work in future versions:

* Existing syntax remains supported
* Existing operations remain supported
* Existing diagnostics remain stable

**Breaking changes** will result in a new major version.

### 11.4 Non-Goals (Explicitly Not Supported)

The following are **explicitly not supported** in v1.0:

* Runtime loops (`while`, `for` with dynamic bounds)
* Conditionals (`if/else`, branching based on runtime values)
* User-defined functions
* Module system / imports
* Dynamic typing
* Runtime reflection
* General-purpose IO (except via capabilities)
* Side effects (except RNG, which is seeded)
* Early returns (`break`, `continue`)
* Recursion

**Rationale**: These features would break determinism, safety, or auditability.

---

## Appendix A: Complete Diagnostic Catalog

### Parsing Errors

* `E_MODEL_EMPTY`: Model block has no statements (fields: `block="model"`)
* `E_DUPLICATE_MODEL_BLOCK`: Both `model` and `forward` blocks present
* `E_DUPLICATE_FORWARD_BLOCK`: Both `model` and `forward` blocks present
* `E_TRAIN_REQUIRES_LOSS`: Train block exists but no loss assignment (fields: `block="train"`)

### Type Errors

* `E_EMBEDDING_REQUIRES_TOKEN_IDS`: Embedding called with non-token_ids input (fields: `input_name`, `received_dtype`)
* `E_LABELS_REQUIRED`: Cross-entropy requires labels input

### Shape Errors

* `E_INPUT_RANK_MISMATCH`: Input rank doesn't match expected (fields: `input`, `expected_rank`, `received_rank`)
* `E_INPUT_DIM_MISMATCH`: Input dimension doesn't match expected (fields: `input`, `dimension`, `expected`, `received`)
* `E_NAMED_DIM_CONFLICT`: Named dimension has conflicting values (fields: `named_dim`, `previous_value`, `new_value`, `input`)

### Reshape Errors

* `E_RESHAPE_MULTIPLE_INFERRED`: Multiple `-1` dimensions in reshape
* `E_RESHAPE_REF_OUT_OF_BOUNDS`: Shape reference index out of bounds (fields: `reference_index`, `input_rank`)
* `E_RESHAPE_REF_LAST_EMPTY`: Cannot reference last dimension of empty shape
* `E_RESHAPE_NAMED_DIM_NOT_FOUND`: Named dimension not found in bindings (fields: `named_dim`)
* `E_RESHAPE_NAMED_DIM_NO_BINDINGS`: Named dimension requires bindings (fields: `named_dim`)
* `E_RESHAPE_CANNOT_INFER`: Cannot infer dimension (fields: `reason`)
* `E_RESHAPE_ELEMENT_MISMATCH`: Reshape element count mismatch (fields: `input_elements`, `resolved_elements`)

### Argument Errors

* `E_INVALID_ARGUMENTS`: Wrong number of arguments (fields: `function`, `expected`, `got`)

### Capability Errors

* `E_CAPABILITY_DENIED`: Required capability not granted (fields: `capability`, `op`, `attempted_action`)
* `E_DATASET_CAPABILITY_DENIED`: Dataset loading requires FileRead capability

### File Errors

* `E_FILE_NOT_FOUND`: File not found (fields: `path`)
* `E_FILE_INVALID_UTF8`: File contains invalid UTF-8 (fields: `path`)
* `E_FILE_IO_ERROR`: File I/O error (fields: `path`, `io_error_kind`)

### Execution Errors

* `E_EXECUTION_FAILURE`: General execution failure (fields: `message`)

---

## Appendix B: Grammar Summary

```
program ::= const_decl* block+

const_decl ::= const IDENT = (integer | float | string | boolean)

block ::= model_block | forward_block | train_block | eval_block | dataset_block

model_block ::= model { statement* }
statement ::= input_decl | param_decl | assignment
input_decl ::= IDENT [shape] | input IDENT: [shape]
param_decl ::= param IDENT: [shape]
assignment ::= IDENT = expr

forward_block ::= forward { let_stmt* return_stmt }
let_stmt ::= let IDENT = expr;
return_stmt ::= return IDENT;

train_block ::= train { field* }
eval_block ::= eval { field* }
dataset_block ::= (dataset | data) { field* }

expr ::= IDENT | call | if_expr | compare_expr | logical_expr | expr + expr | expr - expr | expr * expr | (expr)
if_expr ::= if expr then expr else expr
compare_expr ::= expr (== | != | < | <= | > | >=) expr
logical_expr ::= expr (and | or) expr | not expr
call ::= IDENT ( arg_list? )
arg_list ::= expr (, expr)*

shape ::= [dim (, dim)*]
dim ::= IDENT | integer | -1
```

---

---

## 12. Control Flow (v1)

### 12.1 Conditional Expressions

AILang v1 supports **expression-level conditionals** using the `if-then-else` syntax:

**Syntax:**
```
if_expr ::= if expr then expr else expr
```

**Example:**
```
y = if x > 0 then relu(x) else x
```

**Rules:**
* `if` is an expression, not a statement
* Both branches must return the same shape and dtype
* Condition must be scalar-compatible (evaluates to a boolean-like value)
* No side effects inside conditions
* Conditionals are **not allowed** in `train`, `eval`, or `dataset` blocks

**Diagnostics:**
* `E_IF_NOT_ALLOWED_IN_BLOCK`: Conditional used in disallowed block
* `E_IF_BRANCH_SHAPE_MISMATCH`: Branches have different shapes (runtime error)
* `E_IF_BRANCH_DTYPE_MISMATCH`: Branches have different dtypes (runtime error)
* `E_IF_CONDITION_NOT_SCALAR`: Condition is not scalar-compatible (runtime error)

### 12.2 Boolean Expressions

AILang v1 supports boolean operations for conditions:

**Comparison Operators:**
* `==` (equal)
* `!=` (not equal)
* `<` (less than)
* `<=` (less than or equal)
* `>` (greater than)
* `>=` (greater than or equal)

**Logical Operators:**
* `and` (logical AND)
* `or` (logical OR)
* `not` (logical NOT)

**Rules:**
* Booleans are runtime values (represented as 0.0 or 1.0)
* Cannot be stored as parameters
* Cannot flow into tensor operations directly (must be used in conditionals)
* Comparisons return scalar values (0.0 = false, 1.0 = true)

**Diagnostics:**
* `E_BOOLEAN_USED_AS_TENSOR`: Boolean value used where tensor expected
* `E_TENSOR_USED_AS_BOOLEAN`: Tensor used where boolean expected

### 12.3 Selective Execution Semantics

Control flow interacts with selective execution:

* **Only the executed branch runs**: If condition is true, only `then` branch executes; if false, only `else` branch executes
* **Capability checks apply only to executed paths**: If a branch requires a capability but is not executed, the capability check is skipped
* **Unexecuted branches must not trigger side effects**: Operations in unexecuted branches (file reads, clock access) are not executed
* **Zero-execution guarantee still holds**: If a required capability is denied in an executed branch, execution stops immediately

### 12.4 Lowering Strategy

Conditionals lower into the IR as `Op::If`:

* **Condition node**: The condition expression is lowered to a node
* **True branch node**: The `then` expression is lowered to a node
* **False branch node**: The `else` expression is lowered to a node
* **If node**: Takes three inputs: `[condition, then_branch, else_branch]`

**Deterministic lowering**: Same source → same graph structure (deterministic).

### 12.5 Training and Evaluation Interaction

**Allowed:**
* Conditionals in `model` blocks
* Conditionals in `forward` blocks

**Not allowed:**
* Conditionals in `train` blocks (loss must be single-valued)
* Conditionals in `eval` blocks (metrics must be deterministic)
* Conditionals in `dataset` blocks (dataset configuration is static)

**Diagnostics:**
* `E_IF_NOT_ALLOWED_IN_BLOCK`: Conditional used in disallowed block (fields: `block`)

### 12.6 Gradient Flow

For training, gradients flow through conditionals:

* **If operation**: Gradients propagate to both branches (this is correct for differentiable conditionals)
* **Compare operations**: Do not propagate gradients (boolean operations)
* **Logical operations**: Do not propagate gradients (boolean operations)

**Note**: This means conditionals are differentiable, but comparisons and logical operations are not.

---

## 13. Iteration & Loops (v1)

### 13.1 Overview

AILang v1 supports **bounded iteration** through three loop constructs:

* `for` loops over static ranges
* `repeat` loops with constant counts
* `reduce` loops for explicit reductions

All loops are **statically bounded** and **fully unrolled** at lowering time. This ensures:

* Determinism: Same program → same unrolled graph
* Safety: No unbounded loops or data-dependent iteration
* Analyzability: Graph remains finite and statically known

### 13.2 For Loops

**Syntax:**
```
for var in range(start, end) do
    expr
end
```

**Example:**
```
y = for i in range(0, 10) do
    f(i, x)
end
```

**Rules:**
* `start` and `end` must be compile-time constants (integer literals or const declarations)
* Loop variable `var` is immutable within the loop body and can be used in expressions
* Loop variable is substituted with constant scalar values during lowering
* Loop body `expr` is evaluated for each iteration
* Range is `[start, end)` (start inclusive, end exclusive)
* **COLLECT semantics**: If body returns tensor of shape `[...S]`, result is tensor of shape `[N, ...S]` (stacked along new leading axis)
* Loop variable is only in scope within the loop body

**Diagnostics:**
* `E_LOOP_BOUND_NOT_CONSTANT`: Bound is not a compile-time constant
* `E_LOOP_INVALID_RANGE`: Range end ≤ start
* `E_LOOP_BODY_INVALID_RETURN`: Loop body does not produce a value
* `E_LOOP_VAR_OUT_OF_SCOPE`: Loop variable used outside loop body
* `E_STACK_SHAPE_MISMATCH`: Collected outputs have mismatched shapes
* `E_STACK_EMPTY`: Stack operation has no inputs

### 13.3 Repeat Loops

**Syntax:**
```
repeat N times init <init_expr> do
    <body_expr>
end
```

**Example:**
```
y = repeat 5 times init x do
    relu(y)
end
```

**Rules:**
* `N` must be a compile-time constant (integer literal or const)
* `N` must be greater than 0
* `init` expression is required and provides the initial accumulator value
* Loop body `body_expr` is evaluated `N` times, using the accumulator variable
* Accumulator variable (typically `y`) is in scope within the body
* Body expression must return same shape and dtype as accumulator
* Result is the final accumulator value after `N` iterations
* Equivalent to unrolled iteration: `y1 = f(y0)`, `y2 = f(y1)`, ..., `yN = f(y(N-1))`

**Diagnostics:**
* `E_REPEAT_COUNT_NOT_CONSTANT`: Count is not a compile-time constant
* `E_REPEAT_INVALID_COUNT`: Count ≤ 0
* `E_REPEAT_REQUIRES_INIT`: Missing `init` expression
* `E_REPEAT_BODY_SHAPE_MISMATCH`: Body output shape doesn't match accumulator
* `E_REPEAT_BODY_DTYPE_MISMATCH`: Body output dtype doesn't match accumulator

### 13.4 Reduce Loops

**Syntax:**
```
reduce op over range(start, end) do
    expr
end
```

**Example:**
```
sum = reduce add over range(0, 10) do
    f(i)
end
```

**Rules:**
* `op` must be one of: `add`, `max`, `min`
* `start` and `end` must be compile-time constants
* Loop variable is implicit (currently `i`) and can be used in the body expression
* Loop variable is substituted with constant scalar values during lowering
* Body returns tensor of shape `[...S]`
* Result is tensor of shape `[...S]` (same as body output)
* Reduction operator is applied across all iterations:
  * `add`: Sum all body outputs
  * `max`: Maximum of all body outputs (element-wise)
  * `min`: Minimum of all body outputs (element-wise)

**Diagnostics:**
* `E_REDUCE_OP_INVALID`: Unknown reduction operator
* `E_LOOP_BOUND_NOT_CONSTANT`: Bound is not a compile-time constant
* `E_LOOP_INVALID_RANGE`: Range end ≤ start

### 13.5 Execution Semantics

**Unrolling:**
* All loops are **fully unrolled** at lowering time
* No runtime loop counters or iteration logic
* Kernel sees a static, finite graph

**Determinism:**
* Identical seed + program = identical unrolled graph
* No iteration-order nondeterminism
* Loop bounds are statically known

**Capability Safety:**
* Capabilities are checked **before loop execution**
* If any iteration would require a denied capability → entire loop fails
* No partial execution

### 13.6 Training & Eval Rules

**Allowed:**
* Loops inside `model` blocks
* Loops inside `forward` blocks

**Disallowed:**
* Loops in `train` blocks (loss must be single-valued)
* Loops in `eval` blocks (metrics must be deterministic)
* Loops in `dataset` blocks (dataset configuration is static)

**Diagnostics:**
* `E_LOOP_NOT_ALLOWED_IN_BLOCK`: Loop used in disallowed block (fields: `block`)

### 13.7 Lowering Strategy

Loops lower into the IR by:

* **Unrolling**: Each iteration becomes a separate subgraph
* **No loop ops**: No runtime loop constructs in IR
* **Node reuse**: Common subexpressions are shared where possible
* **Deterministic**: Same source → same graph structure

### 13.8 Stack Operation

The `Stack` operation is used internally by `for` loops to collect outputs:

**Semantics:**
* Takes multiple tensors of identical shape `[...S]`
* Stacks them along axis 0 to produce shape `[N, ...S]`
* Only axis 0 is supported

**Backward Pass:**
* Splits gradient along axis 0
* Distributes gradient slices to each input

**Diagnostics:**
* `E_STACK_SHAPE_MISMATCH`: Input tensors have different shapes
* `E_STACK_EMPTY`: No inputs provided
* `E_STACK_AXIS_INVALID`: Unsupported axis (only 0 is supported)

### 13.9 Max2 and Min2 Operations

Binary maximum and minimum operations used by reduce loops:

**Semantics:**
* `Max2`: Element-wise maximum of two tensors (same shape required)
* `Min2`: Element-wise minimum of two tensors (same shape required)

**Backward Pass:**
* Gradient flows only to the maximum/minimum element (argmax/argmin mask)
* Correctly handles ties (gradient split equally)

### 13.10 Loop Variable Scoping

**Rules:**
* Loop variables are only in scope within their loop body
* Loop variables cannot be used outside the loop
* Nested loops can use different variable names
* Using the same variable name in nested loops is allowed (inner shadows outer)

**Diagnostics:**
* `E_LOOP_VAR_OUT_OF_SCOPE`: Loop variable used outside loop body
* `E_LOOP_VAR_REDECLARED`: Nested loop uses same variable name (warning, not error)

### 13.11 Limitations & Non-Goals

**Current Limitations:**
* Repeat loop accumulator variable substitution is simplified (assumes variable name "y")
* Full expression rewriting for variable substitution is not yet implemented
* **Unroll limit**: Maximum 10,000 total iterations per program (sum across all loops)
  * Diagnostic: `E_LOOP_UNROLL_LIMIT_EXCEEDED` (fields: `limit`, `requested`, `context`)
  * Enforced at lowering time (before graph construction)
  * Prevents excessive graph expansion and compilation time

**Explicit Non-Goals:**
* `while` loops
* Data-dependent iteration counts
* Early exits (`break`, `continue`)
* User-defined iterators
* Recursion
* Dynamic loop bounds

### 13.9 Rationale

Bounded iteration enables:

* Batch processing
* Repeated computation
* Simple algorithms
* Model-style iteration

…while preserving:

* Determinism
* Safety
* Analyzability
* Static graph structure

---

## 14. Functions (v1)

AILang v1 supports **user-defined functions** for abstraction and code reuse. Functions are **compile-time expanded** (inlined) during lowering, ensuring they have zero runtime overhead and maintain determinism.

### 14.1 Syntax

Function definitions appear at the top level, before `model` or `forward` blocks:

```ail
fn encoder(x, W, b) {
  relu(add(matmul(x, W), b))
}
```

Function calls use standard call syntax:

```ail
y = encoder(tokens, W1, b1)
```

### 14.2 Semantics

* **Pure functions**: Same inputs → same outputs (no side effects)
* **Compile-time expansion**: Functions are fully inlined during lowering
* **Parameter substitution**: Arguments are substituted for parameters by value
* **Fresh subgraphs**: Each call produces a separate subgraph (no node collisions)
* **Static validation**: Shapes and dtypes must resolve at compile time

### 14.3 Restrictions

Functions may only contain expressions valid in `model` / `forward` blocks:
* No `train`, `eval`, or `dataset` blocks inside function bodies
* No loops inside function bodies (v1)
* No recursion (direct or indirect)
* No default arguments (v1)
* No side effects

### 14.4 Function Calls

Function calls are resolved in this order:
1. Check if function is user-defined (in `fn_env`)
2. If found, expand by inlining
3. Otherwise, check built-in functions
4. If not found, error `E_FUNCTION_NOT_FOUND`

### 14.5 Context Restrictions

User-defined functions **cannot** be called in:
* `train` blocks → `E_FUNCTION_INVALID_CONTEXT`
* `eval` blocks → `E_FUNCTION_INVALID_CONTEXT`
* `dataset` blocks → `E_FUNCTION_INVALID_CONTEXT`
* `loss` blocks → `E_FUNCTION_INVALID_CONTEXT`

Built-in functions (e.g., `matmul`, `relu`, `add`) are allowed in all contexts.

### 14.6 Diagnostics

* `E_FUNCTION_NOT_FOUND`: Function name not defined
* `E_FUNCTION_ARITY_MISMATCH`: Wrong number of arguments (fields: `expected`, `received`)
* `E_FUNCTION_RECURSION`: Recursive call detected (fields: `function_name`, `call_site`)
* `E_FUNCTION_DUPLICATE_NAME`: Function name already defined
* `E_FUNCTION_INVALID_CONTEXT`: Function called in disallowed block (fields: `function_name`, `context`)

### 14.7 Lowering Strategy

Functions are expanded during lowering:
1. Lookup function definition
2. Lower all arguments first
3. Create new symbol table with parameter → argument substitutions
4. Lower function body with substituted parameters
5. Return result node

No function metadata remains in the lowered IR. Exported models contain only IR operations.

### 14.8 Limitations

* No recursion (prevents infinite expansion)
* No higher-order functions
* No closures
* No default arguments
* Functions must be defined before use

### 14.9 Examples

**Simple function:**
```ail
fn double(x) {
  add(x, x)
}

forward {
  input x: [B, D]
  let y = double(x)
  return y
}
```

**Nested functions:**
```ail
fn linear(x, W, b) {
  add(matmul(x, W), b)
}

fn encoder(x, W, b) {
  relu(linear(x, W, b))
}

forward {
  input x: [B, D]
  param W: [D, 1]
  param b: [1]
  let y = encoder(x, W, b)
  return y
}
```

---

**End of Language Reference v1.0**


# AILang Language Reference

**Version:** 1.0 (Easy Syntax v1)  
**Status:** Normative Specification

This document is a formal specification of the AILang programming language. It defines the syntax, semantics, and behavior of AILang programs.

---

## 1. Overview

### 1.1 What is an AILang Program?

An AILang program is a text file (`.ail` extension) that defines:

* Constants
* A computation graph (model or forward block)
* Optional dataset configuration
* Optional training configuration
* Optional evaluation configuration

Programs are **declarative**: they describe what to compute, not how to execute it step-by-step.

### 1.2 Determinism Contract

AILang guarantees **deterministic execution**:

* Same program (`.ail` file)
* Same seed (`--seed N`)
* Same dataset (if used)
* **= Identical results**

This contract holds for:
* Forward passes (inference)
* Training loops
* Evaluation metrics
* All intermediate computations

**Violations of determinism are considered bugs.**

### 1.3 Capability Contract

AILang uses a **capability-gated security model**:

* Programs declare what system resources they need (files, clock, network)
* Execution is **blocked** if required capabilities are not granted
* **Zero-execution guarantee**: If a capability is denied, the program never executes

Capabilities are granted via CLI flags:
* `--allow fileread`: Enable file reading (required for datasets)
* `--allow clock`: Enable `now()` operation
* `--allow network`: Enable network operations (if supported)

**Default state**: All capabilities are denied.

---

## 2. Lexical Grammar

### 2.1 Identifiers

Identifiers start with a letter or underscore, followed by letters, digits, or underscores:

```
identifier ::= [a-zA-Z_][a-zA-Z0-9_]*
```

Examples: `x`, `W1`, `hidden_dim`, `_temp`

### 2.2 Keywords

Reserved keywords (case-sensitive):

```
const, input, param, forward, model, data, dataset, let, return, loss, train, eval, true, false
```

Keywords cannot be used as identifiers.

### 2.3 Literals

**Integer literals:**
```
integer ::= [0-9]+
```
Examples: `0`, `42`, `100`

**Float literals:**
```
float ::= [0-9]+\.[0-9]+
```
Examples: `0.0`, `3.14`, `0.001`

**String literals:**
```
string ::= "([^"\\]|\\.)*"
```
Examples: `"hello"`, `"path/to/file.jsonl"`, `"tokens"`

**Boolean literals:**
```
boolean ::= true | false
```

### 2.4 Comments

Single-line comments start with `#`:

```
comment ::= #.*
```

Comments extend to the end of the line. No multi-line comments.

### 2.5 Whitespace and Semicolons

* Whitespace (spaces, tabs, newlines) is ignored except as separators
* Semicolons (`;`) are **optional** in most contexts
* Semicolons are allowed but not required after statements in `model` blocks
* Semicolons are required in `forward` blocks (legacy syntax)

---

## 3. Program Structure

### 3.1 Program Syntax

```
program ::= const_decl* block+
```

A program consists of:
1. Zero or more constant declarations
2. One or more blocks (in order: model/forward, optional data, optional train, optional eval)

### 3.2 Constant Declarations

```
const_decl ::= const IDENT = (integer | float | string | boolean)
```

Constants are compile-time values. They can be referenced in shapes and expressions.

Examples:
```ail
const B = 32
const D = 128
const LR = 0.001
const PATH = "data.jsonl"
```

### 3.3 Model Block

**Syntax:**
```
model_block ::= model { statement* }
```

**Statements:**
* Input declarations (shorthand or explicit)
* Parameter declarations
* Assignments

**Example:**
```ail
model {
  tokens [B, T]
  labels [B]
  
  param E [V, D]
  param W [D, C]
  
  h = embedding(tokens, E)
  h = meanpool(h)
  logits = matmul(h, W)
}
```

**Rules:**
* Input declarations: `IDENT [shape]` or `input IDENT: [shape]`
* Parameter declarations: `param IDENT: [shape]`
* Assignments: `IDENT = expr`
* Implicit output: Last assignment (or variable named `logits`) becomes `forward_output`
* Empty model block: Error `E_MODEL_EMPTY`

### 3.4 Forward Block (Legacy)

**Syntax:**
```
forward_block ::= forward { let_stmt* return_stmt }
```

**Example:**
```ail
forward {
  let h = embedding(tokens, E);
  let logits = matmul(h, W);
  return logits;
}
```

**Rules:**
* Must use `let` for bindings
* Must use `return` for output
* Semicolons required

**Backward compatibility:** Forward blocks are still supported but deprecated in favor of model blocks.

### 3.5 Data Block

**Syntax:**
```
data_block ::= (data | dataset) { field* }
```

**Fields:**
* `format = ("tsv" | "jsonl")`
* `path = string`
* `tokens = string` (field name or column spec)
* `labels = string` (field name or column spec)
* `shuffle = boolean` (default: `false`)
* `split = float` (train/val split ratio, default: `1.0`)

**Example:**
```ail
data {
  format = "jsonl"
  path = "examples/data/toy.jsonl"
  tokens = "tokens"
  labels = "labels"
  shuffle = true
  split = 0.8
}
```

### 3.6 Train Block

**Syntax:**
```
train_block ::= train { field* }
```

**Fields:**
* `loss = expr` (required)
* `steps = integer`
* `lr = float`
* `batch = integer`

**Example:**
```ail
train {
  loss = xent(logits, labels)
  steps = 200
  lr = 0.1
  batch = 4
}
```

**Rules:**
* `loss` assignment is required. If missing: Error `E_TRAIN_REQUIRES_LOSS` (fields: `block="train"`)

### 3.7 Eval Block

**Syntax:**
```
eval_block ::= eval { field* }
```

**Fields:**
* `every = integer` (evaluation frequency in steps)
* `metrics = [IDENT, ...]` (list of metric names)
* `split = ("train" | "val")` (default: `"val"`)

**Example:**
```ail
eval {
  every = 20
  metrics = [loss, acc]
  split = "val"
}
```

**Metric aliases:**
* `acc` → `accuracy`

### 3.8 Block Ordering Rules

**Valid orderings:**
1. `const*` (anywhere, but typically at top)
2. `model` or `forward` (exactly one, required)
3. `data` or `dataset` (optional, can appear after model/forward or after train)
4. `train` (optional, must appear after model/forward)
5. `eval` (optional, must appear after train)

**Invalid:**
* Both `model` and `forward`: Error `E_DUPLICATE_MODEL_BLOCK` or `E_DUPLICATE_FORWARD_BLOCK`
* `train` before `model`/`forward`: Syntax error
* `eval` before `train`: Syntax error

---

## 4. Type System

### 4.1 Data Types

AILang has **three data types**:

1. **`tensor`**: General-purpose multi-dimensional array
2. **`token_ids`**: Special type for token sequences (used by `embedding`)
3. **`labels`**: Special type for classification labels (used by `cross_entropy`)

### 4.2 Canonical Dtype Inference

Dtype is inferred from input name using the canonical mapping:

```rust
"tokens" => "token_ids"
"labels" => "labels"
_ => "tensor"
```

This mapping is **the single source of truth**. It is implemented in `infer_dtype_from_name()` and used by:
* Lowering (when creating `InputSpec`)
* CLI (when generating synthetic inputs)

**No rank-based inference.** Dtype is determined solely by name.

### 4.3 Type Errors

**Embedding requires token_ids:**
* Operation: `embedding(tokens, weight)`
* Requirement: First argument must be an input with dtype `token_ids`
* Error: `E_EMBEDDING_REQUIRES_TOKEN_IDS`
* Fields: `input_name`, `received_dtype`

**Cross-entropy requires labels:**
* Operation: `cross_entropy(logits, labels)` or `xent(logits, labels)`
* Requirement: Second argument must be an input with dtype `labels`
* Error: `E_LABELS_REQUIRED` (if labels input not found)

---

## 5. Shape System

### 5.1 Shape Syntax

Shapes are lists of dimensions enclosed in brackets:

```
shape ::= [dim (, dim)*]
dim ::= IDENT | integer | -1
```

Examples:
* `[B, T]` - Named dimensions
* `[32, 128]` - Literal dimensions
* `[B, T, -1]` - Inferred dimension (must be exactly one `-1`)

### 5.2 Named Dimensions

Named dimensions (e.g., `B`, `T`, `D`) are bound from input signatures:

* When an input declares `tokens [B, T]`, `B` and `T` are bound to the actual dimensions
* These bindings are used to resolve shapes in parameters and operations
* **Conflict rule**: If the same named dimension appears in multiple inputs with different values, error `E_NAMED_DIM_CONFLICT`

### 5.3 Shape References (Reshape)

In reshape operations, shapes can reference input dimensions:

* `@0`, `@1`, `@2`, ... - Reference dimension at index
* `@last` - Reference last dimension
* Named dimensions: `"B"`, `"T"`, etc. (must be bound from inputs)
* `-1` - Inferred dimension (exactly one allowed)
* `mul(@0, @1)` - Multiply two dimensions

**Example:**
```ail
x = reshape(x, [@0, mul(@1, @2)])
```

**Errors:**
* `E_RESHAPE_MULTIPLE_INFERRED`: Multiple `-1` dimensions
* `E_RESHAPE_REF_OUT_OF_BOUNDS`: Reference index out of range
* `E_RESHAPE_NAMED_DIM_NOT_FOUND`: Named dimension not bound
* `E_RESHAPE_ELEMENT_MISMATCH`: Element count mismatch

### 5.4 Shape Validation

At runtime, input shapes are validated:

* **Rank mismatch**: Error `E_INPUT_RANK_MISMATCH` (fields: `input`, `expected_rank`, `received_rank`)
* **Dimension mismatch**: Error `E_INPUT_DIM_MISMATCH` (fields: `input`, `dimension`, `expected`, `received`)
* **Named dimension conflict**: Error `E_NAMED_DIM_CONFLICT` (fields: `named_dim`, `previous_value`, `new_value`, `input`)

---

## 6. Expressions & Operations

### 6.1 Expression Syntax

```
expr ::= IDENT                    # Variable reference
       | call                     # Function call
       | expr + expr              # Addition
       | expr - expr              # Subtraction
       | expr * expr              # Multiplication
       | (expr)                   # Parenthesized
```

### 6.2 Operation Catalog

#### 6.2.1 Math Operations

**`matmul(a, b)`**
* Inputs: Two 2D tensors
* Output: 2D tensor `[M, N]` where `a: [M, K]`, `b: [K, N]`
* Deterministic: Yes
* Gradient: Supported

**`add(a, b)`**
* Inputs: Two tensors (broadcast-compatible)
* Output: Element-wise sum
* Deterministic: Yes
* Gradient: Supported

**`sub(a, b)`**
* Inputs: Two tensors (broadcast-compatible)
* Output: Element-wise difference
* Deterministic: Yes
* Gradient: Supported

**`mul(a, b)`**
* Inputs: Two tensors (broadcast-compatible)
* Output: Element-wise product
* Deterministic: Yes
* Gradient: Supported

#### 6.2.2 Activation Functions

**`relu(x)`**
* Input: Any tensor
* Output: Same shape, `max(0, x)` element-wise
* Deterministic: Yes
* Gradient: Supported

**`softmax(x, axis?)`**
* Input: Tensor
* Output: Same shape, softmax along axis (default: last dimension)
* Deterministic: Yes
* Gradient: Supported

#### 6.2.3 Neural Network Layers

**`embedding(tokens, weight)`**
* Inputs:
  * `tokens`: Input with dtype `token_ids` (required)
  * `weight`: Parameter tensor `[V, D]`
* Output: `[B, T, D]` where `B` is batch, `T` is sequence length
* Deterministic: Yes
* Gradient: Supported
* Error: `E_EMBEDDING_REQUIRES_TOKEN_IDS` if first arg is not `token_ids`

**`linear(x, W, b)`** (alias)
* Expands to: `matmul(x, W) + b`
* Inputs:
  * `x`: `[B, D_in]`
  * `W`: `[D_in, D_out]`
  * `b`: `[D_out]` (optional, but currently required in syntax)
* Output: `[B, D_out]`
* Deterministic: Yes
* Gradient: Supported

**`dropout(x, p)`**
* Inputs:
  * `x`: Any tensor
  * `p`: Dropout probability (float literal or const)
* Output: Same shape (training-only, identity in inference)
* Deterministic: No (uses RNG, but seeded)
* Gradient: Supported

#### 6.2.4 Pooling

**`meanpool(x)`** or **`mean_pool_time(x)`**
* Input: `[B, T, D]`
* Output: `[B, D]` (mean over time dimension)
* Deterministic: Yes
* Gradient: Supported

#### 6.2.5 Loss Functions

**`cross_entropy(logits, labels)`** or **`xent(logits, labels)`**
* Inputs:
  * `logits`: `[B, C]` (batch, classes)
  * `labels`: Input with dtype `labels` (required)
* Output: Scalar loss
* Deterministic: Yes
* Gradient: Supported
* Error: `E_LABELS_REQUIRED` if labels input not found

#### 6.2.6 Tensor Manipulation

**`concat(axis, a, b)`**
* Inputs:
  * `axis`: Integer literal (currently only `1` supported)
  * `a`, `b`: Tensors to concatenate
* Output: Concatenated tensor along axis
* Deterministic: Yes
* Gradient: Supported
* Error: `E_INVALID_ARGUMENTS` if axis != 1

**`slice_rows(x, start, len)`**
* Inputs:
  * `x`: 2D tensor `[N, D]`
  * `start`: Integer literal (start index)
  * `len`: Integer literal (length)
* Output: `[len, D]`
* Deterministic: Yes
* Gradient: Supported

**`gather_rows(x, indices)`**
* Inputs:
  * `x`: 2D tensor `[N, D]`
  * `indices`: TokenIds (indices into first dimension)
* Output: `[B, D]` where `B` is batch size from indices
* Deterministic: Yes
* Gradient: Supported

**`reshape(x, shape)`**
* Inputs:
  * `x`: Any tensor
  * `shape`: ReshapeSpec (see Section 5.3)
* Output: Reshaped tensor (same element count)
* Deterministic: Yes
* Gradient: Supported
* Errors: See Section 5.3

#### 6.2.7 Special Operations

**`now()`**
* Inputs: None
* Output: Scalar float (current time)
* Deterministic: No (depends on system clock)
* Gradient: Not supported
* Capability: Requires `--allow clock`
* Error: `E_CAPABILITY_DENIED` if clock capability not granted

**`read_file_text(path)`**
* Inputs:
  * `path`: String literal or const
* Output: `[N]` (hash of file contents as float vector)
* Deterministic: Yes (for same file contents)
* Gradient: Not supported
* Capability: Requires `--allow fileread`
* Error: `E_CAPABILITY_DENIED` if fileread capability not granted
* Errors: `E_FILE_NOT_FOUND`, `E_FILE_INVALID_UTF8`, `E_FILE_IO_ERROR`

### 6.3 Operation Aliases

The following aliases are expanded during lowering:

* `meanpool(x)` → `mean_pool_time(x)`
* `xent(logits, labels)` → `cross_entropy(logits, labels)`
* `linear(x, W, b)` → `matmul(x, W) + b`

Aliases are **frontend sugar only**. They produce the same IR as the expanded form.

### 6.4 Operation Errors

All operations validate arguments:

* **Wrong argument count**: Error `E_INVALID_ARGUMENTS` (fields: `function`, `expected`, `got`)
* **Type errors**: See Section 4.3
* **Capability errors**: See Section 1.3

---

## 7. Execution Semantics

### 7.1 Lowering

AILang programs are **lowered** into a Graph IR:

1. **Parse**: Source text → AST
2. **Lower**: AST → Graph IR (nodes, edges, inputs, parameters)
3. **Validate**: Check shapes, types, capabilities
4. **Execute**: Run the graph

Lowering is **deterministic**: same program → same graph.

### 7.2 Selective Execution

AILang supports **selective execution**:

* Only compute nodes needed for a specific output
* Unused nodes are not executed
* Used for evaluation metrics that don't require full forward pass

### 7.3 Capability Precheck

Before execution, AILang performs a **capability precheck**:

1. Scan the graph for operations requiring capabilities
2. Check if required capabilities are granted
3. If any capability is denied: **stop immediately, do not execute**

**Zero-execution guarantee**: Denied capabilities prevent any computation.

### 7.4 Execution Order

Graph execution follows **topological order**:

* Nodes are executed in dependency order
* All inputs to a node must be computed before the node executes
* Parallel execution is allowed for independent nodes (implementation detail)

---

## 8. Training Semantics

### 8.1 Loss Handling

Training requires a **loss expression**:

* Must be assigned in `train` block: `loss = expr`
* Loss must be a scalar (0D tensor)
* Loss is minimized via gradient descent

**Error**: `E_TRAIN_REQUIRES_LOSS` if `train` block exists but no `loss` assignment (fields: `block="train"`)

### 8.2 Parameter Updates

Parameters are updated via **gradient descent**:

* Compute gradients: `∇loss` w.r.t. all parameters
* Update rule: `param = param - lr * gradient`
* Learning rate: `lr` from `train` block

### 8.3 Dataset Sampling

Datasets are sampled deterministically:

* **Seed**: From `--seed` CLI flag
* **Shuffle**: If `shuffle = true`, shuffle using seed
* **Split**: Train/val split using seed (deterministic)
* **Batching**: Sequential batching (deterministic order)

**Determinism**: Same seed → same batches → same training trajectory.

### 8.4 Training Loop

Training loop semantics:

1. For each step (1 to `steps`):
   a. Sample batch from dataset (deterministic)
   b. Forward pass: compute loss
   c. Backward pass: compute gradients
   d. Update parameters
   e. If `every` steps (from `eval` block): run evaluation

2. Evaluation:
   a. Switch to evaluation split (`train` or `val`)
   b. Compute metrics (loss, accuracy, etc.)
   c. Print results with `eval/` prefix

---

## 9. Evaluation Semantics

### 9.1 Metrics

Supported metrics:

* `loss`: Loss value (scalar)
* `acc` or `accuracy`: Classification accuracy (0-1)

Metrics are computed on the specified split (`train` or `val`).

### 9.2 Scheduling

Evaluation runs at regular intervals:

* `every = N`: Evaluate every N training steps
* Evaluation does not update parameters
* Evaluation uses the current parameter state

### 9.3 Split Selection

Evaluation can run on:

* `split = "train"`: Training split
* `split = "val"`: Validation split (default)

---

## 10. Diagnostics Contract

### 10.1 Diagnostic Structure

All diagnostics have:

* **Code**: Stable string identifier (e.g., `E_EMBEDDING_REQUIRES_TOKEN_IDS`)
* **Title**: Human-readable title
* **Fields**: Key-value pairs (structured data)
* **Hint**: Optional suggestion (not guaranteed)

### 10.2 Stable Error Codes

Error codes are **stable** across versions. They are:

* Not changed without a major version bump
* Documented in this reference
* Used in tests (not string matching)

### 10.3 Diagnostic Catalog

#### Parsing Errors

* `E_MODEL_EMPTY`: Model block has no statements (fields: `block="model"`)
* `E_DUPLICATE_MODEL_BLOCK`: Both `model` and `forward` blocks present
* `E_DUPLICATE_FORWARD_BLOCK`: Both `model` and `forward` blocks present
* `E_TRAIN_REQUIRES_LOSS`: Train block exists but no loss assignment (fields: `block="train"`)

#### Type Errors

* `E_EMBEDDING_REQUIRES_TOKEN_IDS`: Embedding called with non-token_ids input (fields: `input_name`, `received_dtype`)
* `E_LABELS_REQUIRED`: Cross-entropy requires labels input

#### Shape Errors

* `E_INPUT_RANK_MISMATCH`: Input rank doesn't match expected (fields: `input`, `expected_rank`, `received_rank`)
* `E_INPUT_DIM_MISMATCH`: Input dimension doesn't match expected (fields: `input`, `dimension`, `expected`, `received`)
* `E_NAMED_DIM_CONFLICT`: Named dimension has conflicting values (fields: `named_dim`, `previous_value`, `new_value`, `input`)

#### Reshape Errors

* `E_RESHAPE_MULTIPLE_INFERRED`: Multiple `-1` dimensions in reshape
* `E_RESHAPE_REF_OUT_OF_BOUNDS`: Shape reference index out of bounds (fields: `reference_index`, `input_rank`)
* `E_RESHAPE_REF_LAST_EMPTY`: Cannot reference last dimension of empty shape
* `E_RESHAPE_NAMED_DIM_NOT_FOUND`: Named dimension not found in bindings (fields: `named_dim`)
* `E_RESHAPE_NAMED_DIM_NO_BINDINGS`: Named dimension requires bindings (fields: `named_dim`)
* `E_RESHAPE_CANNOT_INFER`: Cannot infer dimension (fields: `reason`)
* `E_RESHAPE_ELEMENT_MISMATCH`: Reshape element count mismatch (fields: `input_elements`, `resolved_elements`)

#### Argument Errors

* `E_INVALID_ARGUMENTS`: Wrong number of arguments (fields: `function`, `expected`, `got`)

#### Capability Errors

* `E_CAPABILITY_DENIED`: Required capability not granted (fields: `capability`, `op`, `attempted_action`)
* `E_DATASET_CAPABILITY_DENIED`: Dataset loading requires FileRead capability

#### File Errors

* `E_FILE_NOT_FOUND`: File not found (fields: `path`)
* `E_FILE_INVALID_UTF8`: File contains invalid UTF-8 (fields: `path`)
* `E_FILE_IO_ERROR`: File I/O error (fields: `path`, `io_error_kind`)

#### Execution Errors

* `E_EXECUTION_FAILURE`: General execution failure (fields: `message`)

### 10.4 Testing Diagnostics

Tests should:

* Assert error **codes**, not prose
* Assert required **fields**
* Not rely on hints (they may change)

**Example:**
```rust
assert_eq!(diag.code, "E_EMBEDDING_REQUIRES_TOKEN_IDS");
let fields: HashMap<_, _> = diag.fields.iter().cloned().collect();
assert_eq!(fields.get("input_name"), Some(&"x".to_string()));
assert_eq!(fields.get("received_dtype"), Some(&"tensor".to_string()));
```

---

## 11. Out of Scope (Explicitly Forbidden)

The following features are **not supported** and are explicitly forbidden:

* **Runtime loops**: No `while`, `for` with dynamic bounds, or data-dependent iteration
* **Conditionals**: No `if/else` or branching based on runtime values
* **Dynamic typing**: All types are known at compile time
* **Runtime reflection**: No introspection of types or shapes at runtime
* **General IO**: No arbitrary file I/O, network I/O, or system calls (except via capabilities)
* **Side effects**: Operations are pure (except RNG, which is seeded)
* **Early returns**: No `break`, `continue`, or early exit from blocks
* **Recursion**: No recursive function calls
* **Closures**: No anonymous functions or closures
* **Generics**: No type parameters or generic functions

**Rationale**: These features would break determinism, safety, or auditability.

---

## 12. Versioning

This specification is for **AILang v1.0 (Easy Syntax v1)**.

Future versions may:
* Add new operations
* Extend syntax (e.g., `repeat` blocks)
* Add new diagnostics
* Relax restrictions (with careful design)

**Breaking changes** will result in a new major version.

---

## Appendix A: Grammar Summary

```
program ::= const_decl* block+

const_decl ::= const IDENT = (integer | float | string | boolean)

block ::= model_block | forward_block | data_block | train_block | eval_block

model_block ::= model { statement* }
statement ::= input_decl | param_decl | assignment
input_decl ::= IDENT [shape] | input IDENT: [shape]
param_decl ::= param IDENT: [shape]
assignment ::= IDENT = expr

forward_block ::= forward { let_stmt* return_stmt }
let_stmt ::= let IDENT = expr;
return_stmt ::= return IDENT;

data_block ::= (data | dataset) { field* }
train_block ::= train { field* }
eval_block ::= eval { field* }

expr ::= IDENT | call | expr + expr | expr - expr | expr * expr | (expr)
call ::= IDENT ( arg_list? )
arg_list ::= expr (, expr)*

shape ::= [dim (, dim)*]
dim ::= IDENT | integer | -1
```

---

## Appendix B: Operation Reference Table

| Operation | Inputs | Output Shape | Deterministic | Gradient | Capability |
|-----------|--------|--------------|---------------|----------|------------|
| `matmul` | 2 | `[M, N]` | Yes | Yes | None |
| `add` | 2 | Same as inputs | Yes | Yes | None |
| `relu` | 1 | Same as input | Yes | Yes | None |
| `embedding` | 2 | `[B, T, D]` | Yes | Yes | None |
| `linear` | 3 | `[B, D_out]` | Yes | Yes | None |
| `dropout` | 2 | Same as input | Seeded | Yes | None |
| `meanpool` | 1 | `[B, D]` | Yes | Yes | None |
| `xent` | 2 | Scalar | Yes | Yes | None |
| `concat` | 3 | Concatenated | Yes | Yes | None |
| `slice_rows` | 3 | `[len, D]` | Yes | Yes | None |
| `gather_rows` | 2 | `[B, D]` | Yes | Yes | None |
| `reshape` | 2 | Reshaped | Yes | Yes | None |
| `now` | 0 | Scalar | No | No | Clock |
| `read_file_text` | 1 | `[N]` | Yes | No | FileRead |

---

**End of Language Reference**


# AILang Standard Library v0

## What This Is

The AILang Standard Library v0 is a collection of **reusable patterns and examples** for common machine learning operations. It provides:

* **Documented patterns** for composing operations into common architectures
* **Example code snippets** you can copy into your `.ail` files
* **Best practices** for shape management and dtype requirements

## What This Is NOT

* **Not a module system**: There are no `import` statements. You copy/paste patterns.
* **Not runtime functions**: These are not callable functions. They are example code.
* **Not a package manager**: No versioning, no dependencies, just documentation.
* **Not compiler extensions**: All patterns use existing operations only.

## How to Use Today (Without Imports)

Since AILang v0 does not support modules or imports, you use the standard library by:

1. **Browse** the patterns in `std/*.ail` files
2. **Copy** the code you need into your `.ail` file
3. **Adapt** shapes and variable names to your use case

**Example**: To use the classifier pattern:

```ail
# Copy from std/nn.ail:
model {
  tokens [B, T]
  labels [B]
  
  param E [V, D]
  param W [D, C]
  param b [C]
  
  # Classifier pattern: embed → meanpool → linear
  logits = linear(meanpool(embedding(tokens, E)), W, b)
}
```

**Module/import support is planned for a future version.**

## Standard Library Functions

### Core Operations (`std/core.ail`)

| Function | Signature | Input Dtypes | Output Shape | Notes |
|----------|-----------|--------------|--------------|-------|
| `relu(x)` | `relu(x)` | `tensor` | Same as input | Built-in alias |
| `softmax(x)` | `softmax(x)` | `tensor` | Same as input | Built-in |
| `dropout(x, p)` | `dropout(x, p)` | `tensor`, `float` | Same as input | Built-in, training-only |
| `concat(axis, a, b)` | `concat(1, a, b)` | `tensor`, `tensor` | Concatenated | Only axis=1 supported |
| `slice_rows(x, start, len)` | `slice_rows(x, 0, 10)` | `tensor`, `int`, `int` | `[len, D]` | Built-in |
| `gather_rows(x, indices)` | `gather_rows(x, indices)` | `tensor`, `token_ids` | `[B, D]` | Built-in |

### Neural Network Patterns (`std/nn.ail`)

| Pattern | Signature | Input Dtypes | Output Shape | Failure Modes |
|---------|-----------|--------------|--------------|----------------|
| `embedding(tokens, E)` | `embedding(tokens, E)` | `token_ids`, `tensor` | `[B, T, D]` | `E_EMBEDDING_REQUIRES_TOKEN_IDS` if tokens not `token_ids` |
| `classifier(tokens, E, W, b)` | `linear(meanpool(embedding(tokens, E)), W, b)` | `token_ids`, `tensor`, `tensor`, `tensor` | `[B, C]` | Same as embedding |
| `mlp2(x, W1, b1, W2, b2)` | `linear(relu(linear(x, W1, b1)), W2, b2)` | `tensor`, `tensor`, `tensor`, `tensor`, `tensor` | `[B, D_out]` | Shape mismatches |

### Loss Functions (`std/loss.ail`)

| Function | Signature | Input Dtypes | Output Shape | Failure Modes |
|----------|-----------|--------------|--------------|---------------|
| `xent(logits, labels)` | `xent(logits, labels)` | `tensor`, `labels` | Scalar | `E_LABELS_REQUIRED` if labels not found |

### Metrics (`std/metrics.ail`)

Metrics are **not graph operations**. They are computed by the CLI during evaluation.

| Metric | Usage | Description |
|--------|-------|-------------|
| `loss` | `metrics = [loss]` | Training loss value |
| `acc` / `accuracy` | `metrics = [loss, acc]` | Classification accuracy (0-1) |

## Complete Examples

### Example 1: Inference with Classifier

See `examples/stdlib_infer.ail`:

```ail
const V = 1000
const D = 64
const C = 10
const T = 128

model {
  tokens [B, T]
  
  param E [V, D]
  param W [D, C]
  param b [C]
  
  # Classifier pattern from std/nn.ail
  logits = linear(meanpool(embedding(tokens, E)), W, b)
}
```

**Run**: `cargo run -p ailang_cli -- --run examples/stdlib_infer.ail --seed 123`

### Example 2: Training with Dataset

See `examples/stdlib_train.ail`:

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
  
  # Classifier pattern
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
  # Loss pattern from std/loss.ail
  loss = xent(logits, labels)
  steps = 200
  lr = 0.1
  batch = 4
}

eval {
  # Metrics pattern from std/metrics.ail
  every = 20
  metrics = [loss, acc]
  split = "val"
}
```

**Run**: `cargo run -p ailang_cli -- --train examples/stdlib_train.ail --seed 123 --allow fileread`

## Failure Modes

### Embedding Requires Token IDs

**Error**: `E_EMBEDDING_REQUIRES_TOKEN_IDS`

**Cause**: `embedding()` first argument must have dtype `token_ids`.

**Fix**: Use an input named `tokens` (automatically gets `token_ids` dtype), or declare explicitly:
```ail
input tokens: [B, T]  # Must be named "tokens" for automatic dtype
```

### Labels Required for Cross-Entropy

**Error**: `E_LABELS_REQUIRED`

**Cause**: `xent()` or `cross_entropy()` requires a labels input.

**Fix**: Use an input named `labels` (automatically gets `labels` dtype), or declare explicitly:
```ail
input labels: [B]  # Must be named "labels" for automatic dtype
```

### Shape Mismatches

**Error**: `E_INPUT_RANK_MISMATCH` or `E_INPUT_DIM_MISMATCH`

**Cause**: Operation inputs have incompatible shapes.

**Fix**: Check shape annotations and ensure dimensions match operation requirements.

## Future: Module System

A future version of AILang will support:

* `import std.nn`
* `import std.loss`
* Function definitions in `.ail` files
* Package versioning

Until then, copy/paste patterns as needed.

---

**Standard Library v0** — Patterns and examples only. No runtime API.


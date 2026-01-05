# AILang Security Model

This document describes AILang's security model, threat model, and security guarantees.

## Capability System

AILang uses a capability-based security model to control access to system resources.

### Design Principles

* **No ambient authority**: Programs cannot access system resources by default
* **Explicit grants**: Capabilities must be explicitly granted via CLI flags
* **Zero-execution guarantee**: Missing capabilities prevent execution before any computation occurs
* **Auditable**: All capability requirements are explicit and verifiable

### Available Capabilities

* **`fileread`**: Read files from disk (required for dataset loading)
* **`filewrite`**: Write files to disk (if supported)
* **`clock`**: Access system clock via `now()` operation
* **`network`**: Network operations (if supported)
* **`env`**: Environment variable access (if supported)
* **`process`**: Process execution (if supported)

### Granting Capabilities

Capabilities are granted via CLI flags:

```bash
--allow fileread    # Enable file reading
--allow clock       # Enable now() operation
```

Multiple capabilities can be granted:

```bash
--allow fileread --allow clock
```

### Capability Enforcement

Capability checks occur before execution:

1. **Static analysis**: Required capabilities are determined from the program
2. **Capability check**: Required capabilities are checked against granted capabilities
3. **Execution decision**: If any required capability is missing, execution fails with a diagnostic

**Zero-execution guarantee**: Missing capabilities prevent execution before any computation occurs.

This prevents:
* Information leakage through timing attacks
* Partial execution that reveals program structure
* Side effects from denied operations
* Time-of-check to time-of-use (TOCTOU) vulnerabilities

## Threat Model

AILang's security model addresses the following threats:

### Untrusted Code Execution

**Threat**: Executing untrusted AILang programs that attempt to access system resources.

**Mitigation**: Capability system prevents access unless explicitly granted.

**Example**: A program that tries to read files fails immediately if `--allow fileread` is not granted.

### Data Exfiltration

**Threat**: Programs attempting to exfiltrate data via file access, network access, or environment variables.

**Mitigation**: All I/O operations require explicit capabilities. By default, no I/O is allowed.

**Example**: A program cannot read files, send network requests, or access environment variables without explicit grants.

### Information Leakage

**Threat**: Programs leaking information through timing, error messages, or partial execution.

**Mitigation**: Zero-execution guarantee prevents information leakage. Failed capability checks prevent any execution.

**Example**: A program that requires a denied capability fails before any computation, preventing timing-based information leakage.

### Non-Deterministic Behavior

**Threat**: System calls introducing non-deterministic behavior that breaks reproducibility.

**Mitigation**: System calls require explicit capabilities. Deterministic execution is guaranteed when capabilities are not used.

**Example**: `now()` requires `--allow clock`. Programs without clock access are fully deterministic.

## Why Capabilities Exist

### Security in Sandboxed Environments

AILang is designed for sandboxed execution:

* **Containerized deployments**: Programs run in containers with limited permissions
* **Edge devices**: Programs run on devices with restricted access
* **Cloud services**: Programs run in serverless environments with security constraints

Capabilities make security properties explicit and auditable.

### Auditable ML Pipelines

Capabilities enable auditability:

* **Explicit permissions**: All resource access is explicit and documented
* **Verifiable security**: Security properties can be verified statically
* **Compliance**: Capability model supports regulatory compliance requirements

### Prevention of Accidental Exposure

Capabilities prevent accidental data exposure:

* **Fail-safe defaults**: Programs cannot access resources by default
* **Explicit grants**: Resource access requires conscious decisions
* **No hidden I/O**: All I/O operations are explicit and capability-gated

## Security Guarantees

### Zero-Execution Guarantee

If a required capability is missing, the program does not execute:

* No computation occurs
* No side effects
* No information leakage
* Immediate failure with diagnostic

This guarantee prevents:
* Timing attacks
* Partial execution information leakage
* TOCTOU vulnerabilities
* Covert channels

### Determinism Guarantee

Programs without system calls are fully deterministic:

* Same program + same seed + same inputs = identical results
* No hidden state
* No external dependencies
* Reproducible execution

This guarantee enables:
* Reproducible experiments
* Verifiable results
* Debugging and auditing
* Compliance requirements

### Static Analysis

AILang programs are statically analyzable:

* All capability requirements are known at compile time
* No dynamic capability checks
* No runtime code generation
* Full program visibility

This enables:
* Security auditing
* Capability requirement analysis
* Compliance verification
* Static security analysis

## Restricted Operations

Some operations are restricted for security reasons:

### File Access

File access requires `--allow fileread`:

* Dataset loading reads files
* File reading operations require capability
* No file access by default

### Clock Access

Clock access requires `--allow clock`:

* `now()` operation requires capability
* Clock access breaks determinism
* No clock access by default

### Network Access

Network access requires `--allow network` (if supported):

* Network operations require capability
* Network access introduces non-determinism
* No network access by default

### Process Execution

Process execution requires `--allow process` (if supported):

* Process execution is highly dangerous
* No process execution by default
* Not recommended for security-sensitive deployments

## Best Practices

### Minimal Capabilities

Grant only the capabilities that are strictly necessary:

```bash
# Good: Only grant required capability
--allow fileread

# Bad: Grant unnecessary capabilities
--allow fileread --allow clock --allow network
```

### Capability Auditing

Audit capability requirements:

* Review programs for required capabilities
* Document capability requirements
* Verify capability grants match requirements

### Security Testing

Test security properties:

* Verify denied capabilities prevent execution
* Test capability enforcement
* Verify zero-execution guarantee

## Future Security Improvements

Planned security improvements:

* Capability scope (limit file paths, network endpoints)
* Capability timeouts
* Resource limits (memory, CPU, execution time)
* Enhanced auditing and logging
* Security documentation and guidelines

## Summary

AILang's security model provides:

* **Explicit capabilities**: All resource access is explicit and auditable
* **Zero-execution guarantee**: Missing capabilities prevent execution
* **Determinism guarantee**: Programs without system calls are fully deterministic
* **Static analysis**: Security properties can be verified statically

This model enables secure, auditable ML pipelines suitable for security-sensitive deployments.


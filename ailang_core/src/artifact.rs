//! Artifact export and import for AILang applications.
//!
//! This module provides functionality to export AILang programs as immutable
//! artifacts (`.aic` directories) and import them for execution.
//!
//! Artifact format:
//! - app.json: AST + config (serialized)
//! - graph.bin: Lowered graph (binary)
//! - model.bin: Model weights (binary)
//! - dataset.meta: Dataset schema (if applicable)
//! - runtime.lock: Runtime version pin

use crate::diagnostic::Diagnostic;
use crate::frontend::{lower::LoweredProgram, parser::Program};
use crate::param::Param;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Artifact metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    /// Application name
    pub app_name: String,
    /// Runtime version (pinned)
    pub runtime_version: String,
    /// Export timestamp
    pub timestamp: u64,
    /// Export seed (for reproducibility)
    pub seed: u64,
    /// Capabilities required
    pub required_capabilities: Vec<String>,
}

/// Artifact configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactConfig {
    /// Application AST (simplified representation)
    pub app_ast: ArtifactAST,
    /// Input specifications
    pub input_specs: Vec<ArtifactInputSpec>,
    /// Parameter specifications
    pub param_specs: Vec<ArtifactParamSpec>,
    /// Forward output node ID
    pub forward_output: Option<usize>,
    /// Loss output node ID (if applicable)
    pub loss_output: Option<usize>,
}

/// Simplified AST representation for artifacts
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactAST {
    pub app_name: Option<String>,
    pub has_state_machine: bool,
    pub has_dataset: bool,
    pub has_train: bool,
    pub has_eval: bool,
}

/// Input specification in artifact
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactInputSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<String>, // Dimension specs as strings
}

/// Parameter specification in artifact
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactParamSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub node_id: usize,
}

/// Export an AILang program as an artifact
pub fn export_artifact(
    program: &Program,
    lowered: &LoweredProgram,
    seed: u64,
    output_path: &Path,
    capabilities: &[String],
) -> Result<(), Diagnostic> {
    // Create artifact directory
    fs::create_dir_all(output_path).map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_EXPORT_ERROR",
            format!("Failed to create artifact directory: {}", e),
        )
    })?;

    // Create metadata
    let app_name = if let Some(ref app) = program.app {
        app.name.clone()
    } else {
        "unnamed".to_string()
    };

    let metadata = ArtifactMetadata {
        app_name: app_name.clone(),
        runtime_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        seed,
        required_capabilities: capabilities.to_vec(),
    };

    // Write metadata (as app.json)
    let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_EXPORT_ERROR",
            format!("Failed to serialize metadata: {}", e),
        )
    })?;
    fs::write(output_path.join("app.json"), metadata_json).map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_EXPORT_ERROR",
            format!("Failed to write app.json: {}", e),
        )
    })?;

    // Create artifact config
    let config = ArtifactConfig {
        app_ast: ArtifactAST {
            app_name: Some(app_name),
            has_state_machine: program.state_machine.is_some(),
            has_dataset: program.dataset.is_some(),
            has_train: program.train.is_some(),
            has_eval: program.eval.is_some(),
        },
        input_specs: lowered
            .input_specs
            .iter()
            .map(|spec| ArtifactInputSpec {
                name: spec.name.clone(),
                dtype: spec.dtype.clone(),
                shape: spec.dims.iter().map(|d| format!("{:?}", d)).collect(),
            })
            .collect(),
        param_specs: lowered
            .params
            .iter()
            .enumerate()
            .map(|(idx, (node_id, param))| {
                // Get parameter name from program.params (by index)
                let name = program
                    .params
                    .get(idx)
                    .map(|p| p.name.clone())
                    .unwrap_or_else(|| format!("param_{}", node_id));
                ArtifactParamSpec {
                    name,
                    shape: param.value.shape().to_vec(),
                    node_id: *node_id,
                }
            })
            .collect(),
        forward_output: Some(lowered.forward_output),
        loss_output: lowered.loss_output,
    };

    // Write config (append to app.json or separate file)
    // For now, we'll extend app.json to include config
    let config_json = serde_json::to_string_pretty(&config).map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_EXPORT_ERROR",
            format!("Failed to serialize config: {}", e),
        )
    })?;

    // Create combined app.json with metadata and config
    #[derive(Serialize)]
    struct AppJson {
        metadata: ArtifactMetadata,
        config: ArtifactConfig,
    }

    let app_json = AppJson {
        metadata: metadata.clone(),
        config,
    };

    let app_json_str = serde_json::to_string_pretty(&app_json).map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_EXPORT_ERROR",
            format!("Failed to serialize app.json: {}", e),
        )
    })?;
    fs::write(output_path.join("app.json"), app_json_str).map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_EXPORT_ERROR",
            format!("Failed to write app.json: {}", e),
        )
    })?;

    // Export graph and model using existing export_model function
    // We'll reuse the existing export_model which writes graph.bin and model.bin
    use crate::model::export_model;
    let params_refs: Vec<(usize, &Param)> = lowered
        .params
        .iter()
        .map(|(id, param)| (*id, param))
        .collect();

    export_model(
        &lowered.graph,
        &params_refs,
        seed,
        output_path,
        Some(lowered.forward_output),
        lowered.loss_output,
    )
    .map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_EXPORT_ERROR",
            format!("Failed to export model: {}", e),
        )
    })?;

    // Write runtime.lock (version pin)
    let runtime_lock = format!("runtime_version = \"{}\"\n", metadata.runtime_version);
    fs::write(output_path.join("runtime.lock"), runtime_lock).map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_EXPORT_ERROR",
            format!("Failed to write runtime.lock: {}", e),
        )
    })?;

    Ok(())
}

/// Import an artifact and validate it
pub fn import_artifact(artifact_path: &Path) -> Result<ArtifactMetadata, Diagnostic> {
    // Check if artifact directory exists
    if !artifact_path.exists() {
        return Err(Diagnostic::new(
            "E_ARTIFACT_NOT_FOUND",
            format!("Artifact directory not found: {}", artifact_path.display()),
        )
        .with_field("path".to_string(), artifact_path.display().to_string()));
    }

    if !artifact_path.is_dir() {
        return Err(Diagnostic::new(
            "E_ARTIFACT_INVALID",
            format!(
                "Artifact path is not a directory: {}",
                artifact_path.display()
            ),
        )
        .with_field("path".to_string(), artifact_path.display().to_string()));
    }

    // Read app.json
    let app_json_path = artifact_path.join("app.json");
    let app_json_str = fs::read_to_string(&app_json_path).map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_READ_ERROR",
            format!("Failed to read app.json: {}", e),
        )
        .with_field("path".to_string(), app_json_path.display().to_string())
    })?;

    // Parse app.json
    #[derive(Deserialize)]
    struct AppJson {
        metadata: ArtifactMetadata,
        config: ArtifactConfig,
    }

    let app_json: AppJson = serde_json::from_str(&app_json_str).map_err(|e| {
        Diagnostic::new(
            "E_ARTIFACT_PARSE_ERROR",
            format!("Failed to parse app.json: {}", e),
        )
    })?;

    // Check runtime version compatibility
    let current_version = env!("CARGO_PKG_VERSION");
    if app_json.metadata.runtime_version != current_version {
        return Err(Diagnostic::new(
            "E_RUNTIME_VERSION_MISMATCH",
            format!(
                "Runtime version mismatch: artifact requires {}, current runtime is {}",
                app_json.metadata.runtime_version, current_version
            ),
        )
        .with_field(
            "expected".to_string(),
            app_json.metadata.runtime_version.clone(),
        )
        .with_field("got".to_string(), current_version.to_string()));
    }

    // Check required files exist
    let required_files = ["model.json", "weights.bin", "runtime.lock"];
    for file in &required_files {
        let file_path = artifact_path.join(file);
        if !file_path.exists() {
            return Err(Diagnostic::new(
                "E_ARTIFACT_INCOMPLETE",
                format!("Required file missing: {}", file),
            )
            .with_field("file".to_string(), file.to_string())
            .with_field("path".to_string(), artifact_path.display().to_string()));
        }
    }

    Ok(app_json.metadata)
}

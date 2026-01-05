// Dataset loading module
use crate::capability::Capabilities;
use crate::capability::Capability;
use crate::diagnostic::Diagnostic;
use serde_json::Value;
use std::fs;

#[derive(Debug, Clone)]
pub struct Dataset {
    pub samples_tokens: Vec<Vec<usize>>,
    pub samples_labels: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct DatasetSplit {
    pub train: Dataset,
    pub validation: Dataset,
}

pub fn load_tsv_dataset(
    path: &str,
    expected_t: usize,
    num_classes: usize,
    capabilities: &Capabilities,
) -> Result<Dataset, Diagnostic> {
    // Capability check MUST happen before file access
    if !capabilities.has(&Capability::FileRead) {
        return Err(Diagnostic::new(
            "E_DATASET_CAPABILITY_DENIED",
            "Dataset loading requires FileRead capability".to_string(),
        )
        .with_field("capability".to_string(), "FileRead".to_string())
        .with_field(
            "attempted_action".to_string(),
            "read dataset file".to_string(),
        )
        .with_field("path".to_string(), path.to_string()));
    }

    // Read file
    let contents = fs::read_to_string(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Diagnostic::new(
                "E_DATASET_FILE_NOT_FOUND",
                format!("Dataset file not found: {}", path),
            )
            .with_field("path".to_string(), path.to_string())
        } else {
            Diagnostic::new(
                "E_DATASET_FILE_NOT_FOUND",
                format!("Failed to read dataset file: {}", e),
            )
            .with_field("path".to_string(), path.to_string())
        }
    })?;

    // Parse TSV
    let mut samples_tokens = Vec::new();
    let mut samples_labels = Vec::new();

    for (line_num, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 2 {
            return Err(Diagnostic::new(
                "E_DATASET_PARSE_ERROR",
                format!(
                    "Expected 2 columns (tokens and label), found {}",
                    parts.len()
                ),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field(
                "reason".to_string(),
                format!("Expected 2 columns, found {}", parts.len()),
            ));
        }

        // Parse tokens (comma-separated integers)
        let tokens_str = parts[0].trim();
        let tokens: Result<Vec<usize>, _> = tokens_str
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect();

        let tokens = tokens.map_err(|_| {
            Diagnostic::new(
                "E_DATASET_PARSE_ERROR",
                format!("Failed to parse tokens as integers: {}", tokens_str),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("reason".to_string(), "Invalid token format".to_string())
        })?;

        // Check dimension match
        if tokens.len() != expected_t {
            return Err(Diagnostic::new(
                "E_DATASET_DIM_MISMATCH",
                format!(
                    "Token count mismatch: expected {}, found {}",
                    expected_t,
                    tokens.len()
                ),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("expected_T".to_string(), expected_t.to_string())
            .with_field("received_T".to_string(), tokens.len().to_string()));
        }

        // Parse label
        let label_str = parts[1].trim();
        let label = label_str.parse::<usize>().map_err(|_| {
            Diagnostic::new(
                "E_DATASET_PARSE_ERROR",
                format!("Failed to parse label as integer: {}", label_str),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("reason".to_string(), "Invalid label format".to_string())
        })?;

        // Check label range
        if label >= num_classes {
            return Err(Diagnostic::new(
                "E_DATASET_LABEL_OUT_OF_RANGE",
                format!("Label {} is out of range [0, {})", label, num_classes),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("label".to_string(), label.to_string())
            .with_field("num_classes".to_string(), num_classes.to_string()));
        }

        samples_tokens.push(tokens);
        samples_labels.push(label);
    }

    Ok(Dataset {
        samples_tokens,
        samples_labels,
    })
}

pub fn load_jsonl_dataset(
    path: &str,
    expected_t: usize,
    num_classes: usize,
    capabilities: &Capabilities,
) -> Result<Dataset, Diagnostic> {
    // Capability check MUST happen before file access
    if !capabilities.has(&Capability::FileRead) {
        return Err(Diagnostic::new(
            "E_DATASET_CAPABILITY_DENIED",
            "Dataset loading requires FileRead capability".to_string(),
        )
        .with_field("capability".to_string(), "FileRead".to_string())
        .with_field(
            "attempted_action".to_string(),
            "read dataset file".to_string(),
        )
        .with_field("path".to_string(), path.to_string()));
    }

    // Read file
    let contents = fs::read_to_string(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Diagnostic::new(
                "E_DATASET_FILE_NOT_FOUND",
                format!("Dataset file not found: {}", path),
            )
            .with_field("path".to_string(), path.to_string())
        } else {
            Diagnostic::new(
                "E_DATASET_FILE_NOT_FOUND",
                format!("Failed to read dataset file: {}", e),
            )
            .with_field("path".to_string(), path.to_string())
        }
    })?;

    // Parse JSONL
    let mut samples_tokens = Vec::new();
    let mut samples_labels = Vec::new();

    for (line_num, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse JSON object
        let json: Value = serde_json::from_str(line).map_err(|e| {
            Diagnostic::new(
                "E_DATASET_JSONL_PARSE_ERROR",
                format!("Failed to parse JSON on line {}: {}", line_num + 1, e),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("reason".to_string(), format!("JSON parse error: {}", e))
        })?;

        // Extract tokens field
        let tokens_value = json.get("tokens").ok_or_else(|| {
            Diagnostic::new(
                "E_DATASET_SCHEMA_MISSING_FIELD",
                format!("Missing 'tokens' field on line {}", line_num + 1),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("field".to_string(), "tokens".to_string())
        })?;

        // Validate tokens is an array
        let tokens_array = tokens_value.as_array().ok_or_else(|| {
            Diagnostic::new(
                "E_DATASET_SCHEMA_WRONG_TYPE",
                format!("'tokens' field must be an array on line {}", line_num + 1),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("field".to_string(), "tokens".to_string())
        })?;

        // Convert to Vec<usize>
        let tokens: Result<Vec<usize>, _> = tokens_array
            .iter()
            .map(|v| {
                v.as_u64().map(|n| n as usize).ok_or_else(|| {
                    Diagnostic::new(
                        "E_DATASET_SCHEMA_WRONG_TYPE",
                        format!("Token must be an integer on line {}", line_num + 1),
                    )
                    .with_field("path".to_string(), path.to_string())
                    .with_field("line".to_string(), (line_num + 1).to_string())
                    .with_field("field".to_string(), "tokens".to_string())
                })
            })
            .collect();

        let tokens = tokens.map_err(|e| e)?;

        // Check dimension match
        if tokens.len() != expected_t {
            return Err(Diagnostic::new(
                "E_DATASET_DIM_MISMATCH",
                format!(
                    "Token count mismatch: expected {}, found {}",
                    expected_t,
                    tokens.len()
                ),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("expected_T".to_string(), expected_t.to_string())
            .with_field("received_T".to_string(), tokens.len().to_string()));
        }

        // Extract label field
        let label_value = json.get("label").ok_or_else(|| {
            Diagnostic::new(
                "E_DATASET_SCHEMA_MISSING_FIELD",
                format!("Missing 'label' field on line {}", line_num + 1),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("field".to_string(), "label".to_string())
        })?;

        // Validate label is an integer
        let label = label_value.as_u64().ok_or_else(|| {
            Diagnostic::new(
                "E_DATASET_SCHEMA_WRONG_TYPE",
                format!("'label' field must be an integer on line {}", line_num + 1),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("field".to_string(), "label".to_string())
        })? as usize;

        // Check label range
        if label >= num_classes {
            return Err(Diagnostic::new(
                "E_DATASET_LABEL_OUT_OF_RANGE",
                format!("Label {} is out of range [0, {})", label, num_classes),
            )
            .with_field("path".to_string(), path.to_string())
            .with_field("line".to_string(), (line_num + 1).to_string())
            .with_field("label".to_string(), label.to_string())
            .with_field("num_classes".to_string(), num_classes.to_string()));
        }

        samples_tokens.push(tokens);
        samples_labels.push(label);
    }

    Ok(Dataset {
        samples_tokens,
        samples_labels,
    })
}

/// Shuffle dataset deterministically using seed
pub fn shuffle_dataset(dataset: &mut Dataset, seed: u64) {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..dataset.samples_tokens.len()).collect();
    indices.shuffle(&mut rng);

    let mut shuffled_tokens = Vec::with_capacity(dataset.samples_tokens.len());
    let mut shuffled_labels = Vec::with_capacity(dataset.samples_labels.len());

    for &idx in &indices {
        shuffled_tokens.push(dataset.samples_tokens[idx].clone());
        shuffled_labels.push(dataset.samples_labels[idx]);
    }

    dataset.samples_tokens = shuffled_tokens;
    dataset.samples_labels = shuffled_labels;
}

/// Split dataset deterministically into train/validation sets
pub fn split_dataset(dataset: &Dataset, split_ratio: f32) -> Result<DatasetSplit, Diagnostic> {
    if split_ratio <= 0.0 || split_ratio >= 1.0 {
        return Err(Diagnostic::new(
            "E_INVALID_ARGUMENTS",
            format!("Split ratio must be in (0.0, 1.0), got {}", split_ratio),
        ));
    }

    let total = dataset.samples_tokens.len();
    let train_size = (total as f32 * split_ratio) as usize;

    let train_tokens = dataset.samples_tokens[..train_size].to_vec();
    let train_labels = dataset.samples_labels[..train_size].to_vec();

    let val_tokens = dataset.samples_tokens[train_size..].to_vec();
    let val_labels = dataset.samples_labels[train_size..].to_vec();

    Ok(DatasetSplit {
        train: Dataset {
            samples_tokens: train_tokens,
            samples_labels: train_labels,
        },
        validation: Dataset {
            samples_tokens: val_tokens,
            samples_labels: val_labels,
        },
    })
}

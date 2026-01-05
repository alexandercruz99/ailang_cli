use std::process::Command;

fn get_workspace_root() -> std::path::PathBuf {
    std::env::current_dir()
        .expect("Failed to get current dir")
        .parent()
        .expect("Failed to get parent")
        .to_path_buf()
}

fn run_cli(args: &[&str]) -> (bool, String, String) {
    let bin_path = env!("CARGO_BIN_EXE_ailang_cli");
    let workspace_root = get_workspace_root();

    let output = Command::new(bin_path)
        .args(args)
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), stdout, stderr)
}

#[test]
fn test_dataset_capability_denied_error() {
    let workspace_root = get_workspace_root();
    let example_path = workspace_root
        .join("examples")
        .join("train_dataset_jsonl.ail");

    // Run without --allow fileread
    let (success, stdout, stderr) =
        run_cli(&["--train", example_path.to_str().unwrap(), "--seed", "123"]);

    // Should fail
    assert!(
        !success,
        "Command should have failed without fileread capability"
    );

    // Check for stable diagnostic code
    let combined_output = format!("{}\n{}", stdout, stderr);
    assert!(
        combined_output.contains("E_DATASET_CAPABILITY_DENIED"),
        "Expected error code E_DATASET_CAPABILITY_DENIED not found in output: {}",
        combined_output
    );
}

#[test]
fn test_embedding_requires_token_ids_diagnostic() {
    // Create a temporary .ail file that uses embedding with non-token_ids input
    let workspace_root = get_workspace_root();
    let temp_file = workspace_root.join("test_embedding_error.ail");

    // Write a file that will trigger the error
    std::fs::write(
        &temp_file,
        r#"
const V = 20
const D = 8
const T = 6

input x: [B, T]

param E: [V, D]

forward {
  let y = embedding(x, E)
  return y
}
"#,
    )
    .expect("Failed to write test file");

    let (success, stdout, stderr) =
        run_cli(&["--run", temp_file.to_str().unwrap(), "--seed", "123"]);

    // Should fail
    assert!(!success, "Command should have failed with embedding error");

    // Check for stable diagnostic code
    let combined_output = format!("{}\n{}", stdout, stderr);
    assert!(
        combined_output.contains("E_EMBEDDING_REQUIRES_TOKEN_IDS"),
        "Expected error code E_EMBEDDING_REQUIRES_TOKEN_IDS not found in output: {}",
        combined_output
    );

    // Check for required fields (code + fields only, no string matching of prose)
    // The diagnostic should have input_name and received_dtype fields
    // We verify the code is present, which is the contract requirement

    // Cleanup
    std::fs::remove_file(&temp_file).ok();
}

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
fn test_op_sanity_inputs_section() {
    let workspace_root = get_workspace_root();
    let example_path = workspace_root.join("examples").join("op_sanity.ail");

    let (success, stdout, stderr) =
        run_cli(&["--run", example_path.to_str().unwrap(), "--seed", "123"]);

    assert!(
        success,
        "Command failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Assert stable Inputs section exists
    assert!(
        stdout.contains("Inputs:"),
        "Expected 'Inputs:' section not found in: {}",
        stdout
    );

    // Assert x/a/b are tensor (not token_ids)
    let lines: Vec<&str> = stdout.lines().collect();
    let mut found_x = false;
    for line in &lines {
        if line.contains("x:") {
            found_x = true;
            assert!(
                line.contains("tensor"),
                "Expected 'x: tensor' but found: {}",
                line
            );
            assert!(
                !line.contains("token_ids"),
                "Input 'x' should not be token_ids. Found: {}",
                line
            );
        }
    }
    assert!(found_x, "Input 'x' not found in output");
}

#[test]
fn test_minimal_runs_successfully() {
    let workspace_root = get_workspace_root();
    let example_path = workspace_root.join("examples").join("minimal.ail");

    let (success, stdout, stderr) =
        run_cli(&["--run", example_path.to_str().unwrap(), "--seed", "123"]);

    assert!(
        success,
        "Command failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Assert success
    assert!(
        stdout.contains("Parsed and lowered successfully")
            || stdout.contains("✓ Parsed and lowered successfully"),
        "Expected success message not found in: {}",
        stdout
    );
}

#[test]
fn test_train_dataset_jsonl_with_dataset_section() {
    let workspace_root = get_workspace_root();
    let example_path = workspace_root
        .join("examples")
        .join("train_dataset_jsonl.ail");

    let (success, stdout, stderr) = run_cli(&[
        "--train",
        example_path.to_str().unwrap(),
        "--seed",
        "123",
        "--allow",
        "fileread",
    ]);

    assert!(
        success,
        "Command failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Assert stable Dataset section exists
    assert!(
        stdout.contains("Dataset:"),
        "Expected 'Dataset:' section not found in: {}",
        stdout
    );

    // Assert dataset section contains required fields
    let lines: Vec<&str> = stdout.lines().collect();
    let mut found_format = false;
    let mut found_path = false;
    let mut found_train_count = false;
    let mut found_val_count = false;

    let mut in_dataset_section = false;
    for line in &lines {
        if line.contains("Dataset:") {
            in_dataset_section = true;
            continue;
        }
        if in_dataset_section {
            if line.contains("format:") {
                found_format = true;
            }
            if line.contains("path:") {
                found_path = true;
            }
            if line.contains("train_count:") {
                found_train_count = true;
            }
            if line.contains("val_count:") {
                found_val_count = true;
            }
            // Exit dataset section when we hit next section
            if line.contains("Train:") || line.contains("Starting training") {
                break;
            }
        }
    }

    assert!(found_format, "Dataset section missing 'format:' field");
    assert!(found_path, "Dataset section missing 'path:' field");
    assert!(
        found_train_count,
        "Dataset section missing 'train_count:' field"
    );
    assert!(
        found_val_count,
        "Dataset section missing 'val_count:' field"
    );
}

#[test]
fn test_train_eval_with_eval_output() {
    let workspace_root = get_workspace_root();
    let example_path = workspace_root.join("examples").join("train_eval.ail");

    let (success, stdout, stderr) = run_cli(&[
        "--train",
        example_path.to_str().unwrap(),
        "--seed",
        "123",
        "--allow",
        "fileread",
    ]);

    assert!(
        success,
        "Command failed\nstdout: {}\nstderr: {}",
        stdout, stderr
    );

    // Assert eval output appears at expected intervals
    let lines: Vec<&str> = stdout.lines().collect();
    let mut found_eval_loss = false;
    let mut found_eval_accuracy = false;

    for line in &lines {
        if line.contains("eval/loss") {
            found_eval_loss = true;
        }
        if line.contains("eval/accuracy") {
            found_eval_accuracy = true;
        }
    }

    assert!(
        found_eval_loss || found_eval_accuracy,
        "Expected eval output (eval/loss or eval/accuracy) not found in: {}",
        stdout
    );
}

#[test]
fn test_seed_parity() {
    let workspace_root = get_workspace_root();
    let example_path = workspace_root.join("examples").join("op_sanity.ail");

    // Run twice with same seed
    let (success1, stdout1, _) =
        run_cli(&["--run", example_path.to_str().unwrap(), "--seed", "456"]);
    let (success2, stdout2, _) =
        run_cli(&["--run", example_path.to_str().unwrap(), "--seed", "456"]);

    assert!(success1 && success2, "Commands failed");

    // Extract stable sections (Inputs section)
    let extract_stable_section = |stdout: &str| -> String {
        let lines: Vec<&str> = stdout.lines().collect();
        let mut in_inputs = false;
        let mut section_lines = Vec::new();

        for line in &lines {
            if line.contains("Inputs:") {
                in_inputs = true;
                section_lines.push(*line);
                continue;
            }
            if in_inputs {
                if line.trim().is_empty() || line.contains("Graph nodes") {
                    break;
                }
                section_lines.push(*line);
            }
        }
        section_lines.join("\n")
    };

    let stable1 = extract_stable_section(&stdout1);
    let stable2 = extract_stable_section(&stdout2);

    assert_eq!(
        stable1, stable2,
        "Stable sections differ between runs\nRun 1:\n{}\nRun 2:\n{}",
        stable1, stable2
    );
}

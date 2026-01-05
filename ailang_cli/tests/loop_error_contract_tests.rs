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
fn test_loop_bound_not_constant_diagnostic() {
    let workspace_root = get_workspace_root();
    let example_path = workspace_root
        .join("examples")
        .join("errors")
        .join("loop_bound_not_constant.ail");

    let (success, stdout, stderr) =
        run_cli(&["--run", example_path.to_str().unwrap(), "--seed", "123"]);

    assert!(!success, "Command should have failed");

    let combined_output = format!("{}\n{}", stdout, stderr);
    assert!(
        combined_output.contains("E_LOOP_BOUND_NOT_CONSTANT"),
        "Expected error code E_LOOP_BOUND_NOT_CONSTANT not found in output: {}",
        combined_output
    );
}

#[test]
fn test_loop_not_allowed_in_train_diagnostic() {
    let workspace_root = get_workspace_root();
    let example_path = workspace_root
        .join("examples")
        .join("errors")
        .join("loop_not_allowed_in_train.ail");

    let (success, stdout, stderr) =
        run_cli(&["--train", example_path.to_str().unwrap(), "--seed", "123"]);

    assert!(!success, "Command should have failed");

    let combined_output = format!("{}\n{}", stdout, stderr);
    assert!(
        combined_output.contains("E_LOOP_NOT_ALLOWED_IN_BLOCK"),
        "Expected error code E_LOOP_NOT_ALLOWED_IN_BLOCK not found in output: {}",
        combined_output
    );
}

#[test]
fn test_repeat_requires_init_diagnostic() {
    let workspace_root = get_workspace_root();
    let example_path = workspace_root
        .join("examples")
        .join("errors")
        .join("repeat_requires_init.ail");

    let (success, stdout, stderr) =
        run_cli(&["--run", example_path.to_str().unwrap(), "--seed", "123"]);

    assert!(!success, "Command should have failed");

    let combined_output = format!("{}\n{}", stdout, stderr);
    assert!(
        combined_output.contains("E_REPEAT_REQUIRES_INIT"),
        "Expected error code E_REPEAT_REQUIRES_INIT not found in output: {}",
        combined_output
    );
}

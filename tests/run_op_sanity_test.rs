use std::process::Command;

#[test]
fn test_op_sanity_runs_successfully() {
    let bin_path = env!("CARGO_BIN_EXE_ailang_cli");

    // Get the workspace root (parent of target/)
    let workspace_root = std::env::current_dir()
        .expect("Failed to get current dir")
        .parent()
        .expect("Failed to get parent")
        .to_path_buf();

    let example_path = workspace_root.join("examples").join("op_sanity.ail");

    let output = Command::new(bin_path)
        .args(&["--run", example_path.to_str().unwrap(), "--seed", "123"])
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to execute command");

    assert!(
        output.status.success(),
        "Command failed with status: {:?}\nstdout: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Assert success messages
    assert!(
        stdout.contains("Parsed and lowered successfully")
            || stdout.contains("✓ Parsed and lowered successfully"),
        "Expected success message not found in: {}",
        stdout
    );

    assert!(
        stdout.contains("Inference successful") || stdout.contains("✓ Inference successful"),
        "Expected inference success message not found in: {}",
        stdout
    );

    // Assert input summary shows "x: tensor" (not token_ids)
    assert!(
        stdout.contains("x: tensor") || stdout.contains("Inputs:") && stdout.contains("tensor"),
        "Expected input dtype 'tensor' for 'x' not found. Output: {}",
        stdout
    );

    // Ensure it did NOT report x as token_ids
    let lines: Vec<&str> = stdout.lines().collect();
    let x_line = lines.iter().find(|line| line.contains("x:"));
    if let Some(line) = x_line {
        assert!(
            !line.contains("token_ids"),
            "Input 'x' should not be reported as token_ids. Found: {}",
            line
        );
    }
}

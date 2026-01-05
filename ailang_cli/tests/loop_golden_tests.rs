use std::process::Command;

fn get_workspace_root() -> std::path::PathBuf {
    std::env::current_dir()
        .expect("Failed to get current dir")
        .parent()
        .expect("Failed to get parent")
        .to_path_buf()
}

#[test]
fn test_loops_collect_output_shape() {
    let workspace_root = get_workspace_root();
    let bin_path = env!("CARGO_BIN_EXE_ailang_cli");

    let output = Command::new(bin_path)
        .args(&["--run", "examples/loops_collect.ail", "--seed", "123"])
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Command failed. stdout: {}\nstderr: {}",
        stdout,
        stderr
    );

    // Assert Inputs section exists
    assert!(
        stdout.contains("Inputs:"),
        "Output should contain 'Inputs:' section. stdout: {}",
        stdout
    );

    // Assert output shape starts with 4 (stack axis 0 from 4 iterations)
    assert!(
        stdout.contains("[4,") || stdout.contains("Output shape: [4,"),
        "Output shape should start with 4. stdout: {}",
        stdout
    );
}

#[test]
fn test_loops_collect_seed_parity() {
    let workspace_root = get_workspace_root();
    let bin_path = env!("CARGO_BIN_EXE_ailang_cli");

    // Run twice with same seed
    let output1 = Command::new(bin_path)
        .args(&["--run", "examples/loops_collect.ail", "--seed", "123"])
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to execute command");

    let output2 = Command::new(bin_path)
        .args(&["--run", "examples/loops_collect.ail", "--seed", "123"])
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to execute command");

    assert!(output1.status.success() && output2.status.success());

    let stdout1 = String::from_utf8_lossy(&output1.stdout);
    let stdout2 = String::from_utf8_lossy(&output2.stdout);

    // Extract stable sections (Inputs and Output shape)
    let stable1: Vec<&str> = stdout1
        .lines()
        .filter(|l| l.contains("Inputs:") || l.contains("Output shape:"))
        .collect();
    let stable2: Vec<&str> = stdout2
        .lines()
        .filter(|l| l.contains("Inputs:") || l.contains("Output shape:"))
        .collect();

    assert_eq!(
        stable1, stable2,
        "Stable sections should match for same seed"
    );
}

// Diagnostic formatting for CLI
use ailang_core::Diagnostic;

pub fn format_diagnostic(diagnostic: &Diagnostic) -> String {
    let mut output = String::new();

    // Header
    output.push_str(&format!("Error [{}]\n", diagnostic.code));
    output.push_str(&format!("{}\n", diagnostic.title));
    output.push_str("\n");

    // Fields
    for (key, value) in &diagnostic.fields {
        output.push_str(&format!("{}: {}\n", key, value));
    }

    // Hint
    if let Some(hint) = &diagnostic.hint {
        output.push_str("\n");
        output.push_str(&format!("hint: {}\n", hint));
    }

    output
}

pub fn print_diagnostic(diagnostic: &Diagnostic) {
    eprintln!("{}", format_diagnostic(diagnostic));
}

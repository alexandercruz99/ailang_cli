mod context_map;
mod output_convert;

use ailang_core::frontend::{lexer::Lexer, lower::lower, parser::Parser};
use ailang_core::forward_selected::execute_forward_selected_with_capabilities;
use ailang_core::Capabilities;
use context_map::{build_inputs, StrategyContextTick};
use output_convert::tensors_to_desired_targets;
use serde::{Deserialize, Serialize};
use serde_json;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::process;

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum DaemonMessage {
    #[serde(rename = "init")]
    Init {
        seed: u64,
        program_path: String,
        products: Vec<String>,
        #[serde(default)]
        caps: Option<CapsConfig>,
        #[serde(default)]
        candles_1m_len: Option<usize>,
        #[serde(default)]
        candles_5m_len: Option<usize>,
    },
    #[serde(rename = "tick")]
    Tick {
        seq: u64,
        ts: Option<u64>,
        context: serde_json::Value,
    },
    #[serde(rename = "reload")]
    Reload {
        program_path: String,
    },
    #[serde(rename = "shutdown")]
    Shutdown,
}

#[derive(Debug, Deserialize)]
struct CapsConfig {
    max_spread_bps: Option<u32>,
    min_book_depth_usd: Option<f64>,
}

#[derive(Debug, Serialize)]
struct DecisionOutput {
    seq: u64,
    decision_hash: String,
    desired_targets: Vec<serde_json::Value>,
    emits: Vec<String>,
}

// Store compiled program for execution
struct CompiledProgram {
    source: String,
    lowered: ailang_core::frontend::lower::LoweredProgram,
    program_hash: String,
}

struct DaemonState {
    program: Option<CompiledProgram>,
    seed: Option<u64>,
    products: Vec<String>,
    candles_1m_len: usize,
    candles_5m_len: usize,
    initialized: bool,
}

impl DaemonState {
    fn new() -> Self {
        DaemonState {
            program: None,
            seed: None,
            products: Vec::new(),
            candles_1m_len: 120, // Default
            candles_5m_len: 120, // Default
            initialized: false,
        }
    }

    fn handle_init(
        &mut self,
        program_path: String,
        seed: u64,
        products: Vec<String>,
        candles_1m_len: usize,
        candles_5m_len: usize,
    ) -> Result<(), String> {
        // Read and compile program
        let source = fs::read_to_string(&program_path)
            .map_err(|e| format!("Failed to read program file: {}", e))?;

        // Lex
        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize().map_err(|e| format!("Lexer error: {}", e))?;

        // Parse
        let mut parser = Parser::new(tokens);
        let program = parser.parse().map_err(|diag| {
            format!("Parse error: {:?}", diag)
        })?;

        // Lower
        let lowered = lower(&program, seed).map_err(|diag| {
            format!("Lower error: {:?}", diag)
        })?;

        // Compute program hash (deterministic hash of source + seed)
        let program_hash = compute_hash(&format!("{}{}", source, seed));

        self.program = Some(CompiledProgram {
            source: source.clone(),
            lowered,
            program_hash,
        });
        self.seed = Some(seed);
        self.products = products;
        self.candles_1m_len = candles_1m_len;
        self.candles_5m_len = candles_5m_len;
        self.initialized = true;
        Ok(())
    }

    fn handle_tick(&mut self, seq: u64, ts: Option<u64>, context: serde_json::Value) -> Result<DecisionOutput, String> {
        if !self.initialized {
            return Err("Not initialized".to_string());
        }

        let program = self.program.as_ref().ok_or("Program not loaded")?;
        
        // Parse context JSON
        let tick_context: StrategyContextTick = serde_json::from_value(context.clone())
            .map_err(|e| format!("Failed to parse context JSON: {}", e))?;

        // Build tensor inputs from context
        let ts_value = ts.unwrap_or(0);
        let input_map = build_inputs(
            &self.products,
            self.candles_1m_len,
            self.candles_5m_len,
            tick_context,
            seq,
            ts_value,
        );

        // Convert HashMap to ordered Vec<Tensor> matching program's input_specs
        // For MVP, we skip detailed shape validation and let the runtime handle it
        let mut tensor_inputs = Vec::new();
        
        // Count token_ids inputs (they need empty tensor slots but are passed via token_ids array)
        let num_token_inputs = program.lowered.input_specs
            .iter()
            .filter(|s| s.dtype == "token_ids")
            .count();
        
        // Process tensor inputs (skip token_ids and labels)
        for spec in &program.lowered.input_specs {
            if spec.dtype == "tensor" {
                let input_name = &spec.name;
                match input_map.get(input_name) {
                    Some(tensor) => {
                        // Basic shape validation: check rank matches
                        let actual_shape = tensor.shape();
                        if actual_shape.len() != spec.rank {
                            return Err(format!(
                                "E_SHAPE_MISMATCH: Input '{}' rank mismatch: expected {}, got {}",
                                input_name, spec.rank, actual_shape.len()
                            ));
                        }
                        tensor_inputs.push(tensor.clone());
                    }
                    None => {
                        return Err(format!(
                            "E_INPUT_MISSING: Required input '{}' not found in canonical inputs",
                            input_name
                        ));
                    }
                }
            }
        }
        
        // Build inputs array in graph order: tensor inputs, empty tensors for token_ids, then params
        let mut inputs = tensor_inputs;
        
        // Add empty tensors for token_ids input slots
        for _ in 0..num_token_inputs {
            inputs.push(ailang_core::tensor::Tensor::zeros(&[]));
        }
        
        // Add params
        for (_, param) in &program.lowered.params {
            inputs.push(param.value.clone());
        }

        // Execute program using forward pass (runtime will validate shapes)
        let outputs = ailang_core::forward::execute_forward(
            &program.lowered.graph,
            &inputs,
            &[], // No token_ids
        ).map_err(|e| format!("Execution error: {:?}", e))?;

        // Convert outputs to DesiredTargets
        let desired_targets = tensors_to_desired_targets(&outputs, &program.lowered);
        let targets_json = serde_json::to_string(&desired_targets)
            .map_err(|e| format!("Failed to serialize targets: {}", e))?;
        
        let program_hash = &program.program_hash;
        let seed = self.seed.unwrap();
        let context_hash = compute_hash(&serde_json::to_string(&context).unwrap());
        let decision_hash_input = format!("{}{}{}{}{}", program_hash, seed, seq, context_hash, targets_json);
        let decision_hash = format!("{:x}", Sha256::digest(decision_hash_input.as_bytes()));

        Ok(DecisionOutput {
            seq,
            decision_hash,
            desired_targets,
            emits: vec![],
        })
    }

    fn handle_reload(&mut self, program_path: String) -> Result<(), String> {
        let seed = self.seed.unwrap_or(42);
        let products = self.products.clone();
        let c1 = self.candles_1m_len;
        let c5 = self.candles_5m_len;
        self.handle_init(program_path, seed, products, c1, c5)
    }
}

fn compute_hash(input: &str) -> String {
    format!("{:x}", Sha256::digest(input.as_bytes()))
}

fn main() {
    let stdin = io::stdin();
    let reader = BufReader::new(stdin.lock());
    let mut state = DaemonState::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("ERROR {{\"seq\":0,\"code\":\"E_IO\",\"message\":\"{}\"}}", e);
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let msg: DaemonMessage = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("ERROR {{\"seq\":0,\"code\":\"E_PARSE\",\"message\":\"{}\"}}", e);
                continue;
            }
        };

        match msg {
            DaemonMessage::Init {
                seed,
                program_path,
                products,
                caps: _,
                candles_1m_len,
                candles_5m_len,
            } => {
                let c1 = candles_1m_len.unwrap_or(120);
                let c5 = candles_5m_len.unwrap_or(120);
                match state.handle_init(program_path, seed, products, c1, c5) {
                    Ok(_) => {
                        // INIT successful, daemon ready
                    }
                    Err(e) => {
                        eprintln!("ERROR {{\"seq\":0,\"code\":\"E_INIT\",\"message\":\"{}\"}}", e);
                    }
                }
            }
            DaemonMessage::Tick { seq, ts, context } => {
                match state.handle_tick(seq, ts, context) {
                    Ok(decision) => {
                        println!("DECISION {}", serde_json::to_string(&decision).unwrap());
                    }
                    Err(e) => {
                        // Try to extract error code from error message
                        let code = if e.contains("E_INPUT_MISSING") {
                            "E_INPUT_MISSING"
                        } else if e.contains("E_SHAPE_MISMATCH") {
                            "E_SHAPE_MISMATCH"
                        } else {
                            "E_TICK"
                        };
                        eprintln!("ERROR {{\"seq\":{},\"code\":\"{}\",\"message\":\"{}\"}}", seq, code, e);
                    }
                }
            }
            DaemonMessage::Reload { program_path } => {
                match state.handle_reload(program_path) {
                    Ok(_) => {
                        // RELOAD successful
                    }
                    Err(e) => {
                        eprintln!("ERROR {{\"seq\":0,\"code\":\"E_RELOAD\",\"message\":\"{}\"}}", e);
                    }
                }
            }
            DaemonMessage::Shutdown => {
                process::exit(0);
            }
        }
    }
}

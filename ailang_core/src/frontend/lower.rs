use super::parser::{
    CallExpr, CompareOp, DatasetDecl, DimSpec, EventHandlerStatement, Expr, FnDecl, ForwardBlock,
    LogicalOp, OnEventHandler, Program, StateBlock, StateDecl, StateMachineDecl, StateVarDecl,
    StateVarType as ASTStateVarType, TransitionDecl, TransitionKind as ASTTransitionKind,
};
use crate::diagnostic::Diagnostic;
use crate::input_spec::{DimSpec as InputDimSpec, InputSpec};
use crate::ir::{Graph, NodeId, Op};
use crate::param::Param;
use crate::rng::SeededRng;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub steps: usize,
    pub lr: f32,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub metrics: Vec<String>,
    pub every: usize,
    pub split: String, // "train" or "val"
}

pub struct LoweredProgram {
    pub graph: Graph,
    pub params: Vec<(usize, Param)>,
    pub input_specs: Vec<InputSpec>,
    pub forward_output: NodeId,
    pub loss_output: Option<NodeId>,
    pub trainable_param_node_ids: Vec<NodeId>,
    pub train_config: Option<TrainConfig>,
    pub dataset: Option<DatasetConfig>,
    pub eval_config: Option<EvalConfig>,
    pub state_machine: Option<StateMachineConfig>,
    // New event-driven runtime model
    pub state_block: Option<StateBlockConfig>,
    pub event_handlers: Vec<EventHandlerConfig>,
}

#[derive(Debug, Clone)]
pub struct StateMachineConfig {
    pub name: String,
    pub states: Vec<StateConfig>,
}

// New event-driven runtime model config structures
#[derive(Debug, Clone)]
pub struct StateBlockConfig {
    pub vars: Vec<StateVarConfig>,
}

#[derive(Debug, Clone)]
pub struct StateVarConfig {
    pub name: String,
    pub var_type: StateVarType,
    // Initial value will be stored as a constant or evaluated expression
    pub has_init: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StateVarType {
    Int,
    Float,
    Tensor,
}

#[derive(Debug, Clone)]
pub struct EventHandlerConfig {
    pub event_name: String,
    pub statements: Vec<EventHandlerStatementConfig>,
}

#[derive(Debug, Clone)]
pub enum EventHandlerStatementConfig {
    Assign(String), // State assignment: name = expr (lowered to graph node)
    EmitEvent(String), // emit <event_name>
                    // Let bindings are lowered to graph nodes, not stored separately
}

#[derive(Debug, Clone)]
pub struct StateConfig {
    pub name: String,
    pub transitions: Vec<TransitionConfig>,
    pub has_end: bool,
}

#[derive(Debug, Clone)]
pub struct TransitionConfig {
    pub kind: TransitionKind,
    pub target: String,
    pub condition_node: Option<NodeId>, // For if transitions, the condition node ID
}

#[derive(Debug, Clone)]
pub enum TransitionKind {
    On(String), // on event -> State
    If,         // if condition -> State
    Else,       // else -> State
}

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub format: String,
    pub path: String,
    pub tokens_input: String,
    pub labels_input: String,
    pub shuffle: Option<bool>,
    pub split: Option<f32>,
    pub maps: Vec<CallExpr>,    // map transformations
    pub filters: Vec<CallExpr>, // filter predicates
    pub batch: Option<usize>,   // batch size for pipeline
}

#[derive(Debug)]
struct DimEnv {
    consts: HashMap<String, usize>,
    resolved_dims: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct FnEnv {
    functions: HashMap<String, FnDecl>,
}

impl FnEnv {
    fn new(functions: &[FnDecl]) -> Self {
        let mut map = HashMap::new();
        for func in functions {
            map.insert(func.name.clone(), func.clone());
        }
        Self { functions: map }
    }

    fn get(&self, name: &str) -> Option<&FnDecl> {
        self.functions.get(name)
    }
}

impl DimEnv {
    fn new() -> Self {
        Self {
            consts: HashMap::new(),
            resolved_dims: HashMap::new(),
        }
    }

    fn add_const(&mut self, name: String, value: usize) -> Result<(), Diagnostic> {
        if self.consts.contains_key(&name) {
            return Err(
                Diagnostic::new("E_CONST_REDECLARED", "Const redeclared".to_string())
                    .with_field("name".to_string(), name),
            );
        }
        self.consts.insert(name.clone(), value);
        // Also add to resolved_dims if not present
        self.resolved_dims.entry(name).or_insert(value);
        Ok(())
    }

    fn resolve_dim(&self, dim: &DimSpec, context: &str) -> Result<usize, Diagnostic> {
        match dim {
            DimSpec::Literal(n) => Ok(*n),
            DimSpec::Free => {
                return Err(Diagnostic::new(
                    "E_INVALID_DIM_SPEC",
                    "Free dimension spec not allowed in this context".to_string(),
                ));
            }
            DimSpec::Named(name) => {
                // Check if it's a const
                if let Some(&value) = self.consts.get(name) {
                    return Ok(value);
                }
                // Check if it's already resolved
                if let Some(&value) = self.resolved_dims.get(name) {
                    return Ok(value);
                }
                // Special case: B (batch) can remain unresolved for inputs
                if name == "B" {
                    // For MVP, we'll allow B to be unresolved in input specs
                    // but it must be resolved for params
                    return Err(Diagnostic::new("E_DIM_UNBOUND", "Dimension unbound".to_string())
                        .with_field("name".to_string(), name.clone())
                        .with_field("context".to_string(), context.to_string())
                        .with_hint(format!("Dimension '{}' must be bound to a const or resolved from another dimension", name)));
                }
                Err(Diagnostic::new("E_DIM_UNBOUND", "Dimension unbound".to_string())
                    .with_field("name".to_string(), name.clone())
                    .with_field("context".to_string(), context.to_string())
                    .with_hint(format!("Dimension '{}' must be bound to a const or resolved from another dimension", name)))
            }
        }
    }

    fn bind_dim(&mut self, name: String, value: usize, context: &str) -> Result<(), Diagnostic> {
        if let Some(&existing) = self.resolved_dims.get(&name) {
            if existing != value {
                return Err(
                    Diagnostic::new("E_DIM_CONFLICT", "Dimension conflict".to_string())
                        .with_field("name".to_string(), name.clone())
                        .with_field("first_value".to_string(), existing.to_string())
                        .with_field("second_value".to_string(), value.to_string())
                        .with_field("context".to_string(), context.to_string())
                        .with_hint(format!(
                            "Dimension '{}' was bound to {} but now being bound to {}",
                            name, existing, value
                        )),
                );
            }
        }
        self.resolved_dims.insert(name, value);
        Ok(())
    }
}

pub fn lower(program: &Program, seed: u64) -> Result<LoweredProgram, Diagnostic> {
    // Unroll limit: maximum total iterations across all loops
    const UNROLL_LIMIT: usize = 10_000;
    let mut unroll_budget_used: usize = 0;

    // Build dimension environment
    let mut dim_env = DimEnv::new();

    // First pass: collect all consts
    for const_decl in &program.consts {
        let value = const_decl.value as usize;
        dim_env.add_const(const_decl.name.clone(), value)?;
    }

    // Count token_ids inputs (name-based: "tokens" => token_ids)
    // Count label_ids inputs (name-based: "labels" => labels)
    let token_ids_count = program
        .inputs
        .iter()
        .filter(|input| input.name == "tokens")
        .count();
    let label_ids_count = program
        .inputs
        .iter()
        .filter(|input| input.name == "labels")
        .count();

    // Total side-channel count: token_ids + label_ids
    let total_side_channels = token_ids_count + label_ids_count;

    // Find the index for labels in the token_ids array (after all token_ids)
    let label_ids_index = if label_ids_count > 0 {
        Some(token_ids_count) // Labels come after token_ids
    } else {
        None
    };

    let mut graph = Graph::new_with_token_ids(
        program.inputs.len() + program.params.len(),
        total_side_channels,
    );
    let mut params = Vec::new();
    let mut input_specs = Vec::new();
    let mut symbol_table: HashMap<String, NodeId> = HashMap::new();
    let mut rng = SeededRng::new(seed);

    // Create input nodes and input specs
    for (idx, input_decl) in program.inputs.iter().enumerate() {
        let node_id = graph.input_node(idx);
        symbol_table.insert(input_decl.name.clone(), node_id);

        // Convert shape to InputSpec
        // Dtype classification is NAME-BASED only (never infer from rank/shape):
        // Use centralized dtype inference function
        let dtype = crate::infer_dtype_from_name(&input_decl.name).to_string();

        // Resolve input dims: consts become Literal, named dims stay Named (B can be variable)
        let dims: Vec<InputDimSpec> = input_decl
            .shape
            .iter()
            .map(|dim| match dim {
                DimSpec::Named(name) => {
                    // Try to resolve from consts, but allow B to remain symbolic
                    if let Ok(value) =
                        dim_env.resolve_dim(dim, &format!("input {}", input_decl.name))
                    {
                        InputDimSpec::Literal(value)
                    } else if name == "B" {
                        // B can remain symbolic for inputs
                        InputDimSpec::Named(name.clone())
                    } else {
                        // For other named dims, try to resolve or error
                        // In MVP, we'll try to resolve but allow B
                        InputDimSpec::Named(name.clone())
                    }
                }
                DimSpec::Literal(n) => InputDimSpec::Literal(*n),
                DimSpec::Free => {
                    // Free dimensions not allowed for inputs
                    InputDimSpec::Named("?".to_string()) // Placeholder, should error earlier
                }
            })
            .collect();

        input_specs.push(InputSpec::new(input_decl.name.clone(), dtype, dims));
    }

    // Create param nodes - resolve all dimensions
    for (idx, param_decl) in program.params.iter().enumerate() {
        let param_idx = program.inputs.len() + idx;
        let node_id = graph.input_node(param_idx);
        symbol_table.insert(param_decl.name.clone(), node_id);

        // Resolve param shape dimensions
        let mut shape = Vec::new();
        for (dim_idx, dim) in param_decl.shape.iter().enumerate() {
            let context = format!("param {} dim {}", param_decl.name, dim_idx);
            let resolved = dim_env.resolve_dim(dim, &context)?;
            shape.push(resolved);

            // If this was a named dim, bind it for future use
            if let DimSpec::Named(name) = dim {
                if name != "B" {
                    // Bind the dimension (will error if conflict)
                    dim_env.bind_dim(name.clone(), resolved, &context)?;
                }
            }
        }

        // Initialize param with random values
        let num_elements: usize = shape.iter().product();
        let mut data = Vec::with_capacity(num_elements);
        for _ in 0..num_elements {
            // Xavier initialization
            data.push(
                (rng.gen() * 2.0 - 1.0) * (6.0 / (shape.iter().sum::<usize>() as f32)).sqrt(),
            );
        }

        let param = Param::new(Tensor::from_vec(&shape, data));
        params.push((node_id, param));
    }

    // Create function environment
    let fn_env = FnEnv::new(&program.functions);

    // Lower forward block
    let forward_output = lower_forward_block(
        &mut graph,
        &program.forward,
        &mut symbol_table,
        label_ids_index,
        &dim_env,
        &input_specs,
        &fn_env,
        &mut unroll_budget_used,
    )?;

    // Lower loss block if present
    let loss_output = if let Some(ref loss_block) = program.loss {
        Some(lower_loss_block(
            &mut graph,
            loss_block,
            &mut symbol_table,
            label_ids_index,
            &dim_env,
            &input_specs,
            &fn_env,
            &mut unroll_budget_used,
        )?)
    } else {
        if program.train.is_some() {
            return Err(Diagnostic::new(
                "E_TRAIN_REQUIRES_LOSS",
                "Train block requires loss block".to_string(),
            )
            .with_field("block".to_string(), "train".to_string()));
        }
        None
    };

    // Collect trainable param node IDs (all params are trainable for MVP)
    let trainable_param_node_ids: Vec<NodeId> =
        params.iter().map(|(node_id, _)| *node_id).collect();

    // Process train block if present
    let train_config = if let Some(ref train_block) = program.train {
        // Validate: train block requires loss (either loss block or loss = ... in train block)
        if train_block.loss_expr.is_none() && loss_output.is_none() {
            return Err(Diagnostic::new(
                "E_TRAIN_REQUIRES_LOSS",
                "Train block requires loss (either loss block or loss = ... in train block)"
                    .to_string(),
            )
            .with_field("block".to_string(), "train".to_string()));
        }
        Some(TrainConfig {
            steps: train_block.steps.unwrap_or(200) as usize,
            lr: train_block.lr.unwrap_or(0.1),
            batch_size: train_block.batch.unwrap_or(4) as usize,
        })
    } else {
        None
    };

    // Update loss_output if it came from train block (before eval config check)
    let final_loss_output = if let Some(ref train_block) = program.train {
        if let Some(ref loss_expr) = train_block.loss_expr {
            // Lower loss expression from train block
            let loss_symbol_table = symbol_table.clone();
            let mut call_stack = Vec::new();
            match lower_expr_with_specs_internal(
                &mut graph,
                loss_expr,
                &loss_symbol_table,
                label_ids_index,
                &dim_env,
                &input_specs,
                &fn_env,
                &mut call_stack,
                false, // Functions not allowed in train block
                &mut unroll_budget_used,
            ) {
                Ok(loss_id) => Some(loss_id),
                Err(e) => return Err(e),
            }
        } else {
            loss_output
        }
    } else {
        loss_output
    };

    // Process eval block if present
    let eval_config = if let Some(ref eval_block) = program.eval {
        // Validate: eval block requires train block
        if train_config.is_none() {
            return Err(Diagnostic::new(
                "E_EVAL_REQUIRES_TRAIN",
                "Eval block requires train block".to_string(),
            ));
        }
        // Validate: eval block requires loss (for loss metric)
        if eval_block.metrics.contains(&"loss".to_string()) && final_loss_output.is_none() {
            return Err(Diagnostic::new(
                "E_EVAL_REQUIRES_LOSS",
                "Eval block with 'loss' metric requires loss".to_string(),
            ));
        }
        Some(EvalConfig {
            metrics: eval_block.metrics.clone(),
            every: eval_block.every.unwrap_or(20) as usize,
            split: eval_block
                .split
                .clone()
                .unwrap_or_else(|| "val".to_string()),
        })
    } else {
        None
    };

    // Set input specs on graph
    graph = graph.with_input_specs(input_specs.clone());

    // Store dataset config if present
    let dataset = program
        .dataset
        .as_ref()
        .map(|d| {
            // Validate pipeline functions exist and have correct signatures
            validate_dataset_pipeline(d, &fn_env)?;

            Ok(DatasetConfig {
                format: d.format.clone(),
                path: d.path.clone(),
                tokens_input: d.tokens_input.clone(),
                labels_input: d.labels_input.clone(),
                shuffle: d.shuffle,
                split: d.split,
                maps: d.maps.clone(),
                filters: d.filters.clone(),
                batch: d.batch,
            })
        })
        .transpose()?;

    // Process state machine if present
    let state_machine = if let Some(ref sm_decl) = program.state_machine {
        // Validate state machine
        validate_state_machine(sm_decl)?;

        // Lower state machine (create subgraphs for each state)
        Some(lower_state_machine(
            &mut graph,
            sm_decl,
            &mut symbol_table,
            label_ids_index,
            &dim_env,
            &input_specs,
            &fn_env,
            &mut unroll_budget_used,
        )?)
    } else {
        None
    };

    // Validate and lower state block and event handlers (new event-driven runtime model)
    let state_block = if let Some(ref sb_decl) = program.state_block {
        validate_state_block(sb_decl, &program.event_handlers)?;
        Some(lower_state_block(sb_decl)?)
    } else {
        None
    };

    let event_handlers = if !program.event_handlers.is_empty() {
        validate_event_handlers(&program.event_handlers, &program.state_block)?;
        lower_event_handlers(&program.event_handlers)?
    } else {
        Vec::new()
    };

    Ok(LoweredProgram {
        graph,
        params,
        input_specs,
        forward_output,
        loss_output: final_loss_output,
        trainable_param_node_ids,
        train_config,
        dataset,
        eval_config,
        state_machine,
        state_block,
        event_handlers,
    })
}

fn lower_forward_block(
    graph: &mut Graph,
    forward: &ForwardBlock,
    symbol_table: &mut HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[crate::input_spec::InputSpec],
    fn_env: &FnEnv,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    let mut call_stack = Vec::new();
    // Lower each let binding
    for let_binding in &forward.lets {
        let node_id = lower_expr_with_specs(
            graph,
            &let_binding.expr,
            symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            &mut call_stack,
            unroll_budget_used,
        )?;
        symbol_table.insert(let_binding.name.clone(), node_id);
    }

    // Get return node
    let output_node_id = symbol_table.get(&forward.ret).ok_or_else(|| {
        Diagnostic::new(
            "E_RETURN_NOT_FOUND",
            "Return identifier not found".to_string(),
        )
        .with_field("name".to_string(), forward.ret.clone())
    })?;

    Ok(*output_node_id)
}

fn lower_loss_block(
    graph: &mut Graph,
    loss: &super::parser::LossBlock,
    symbol_table: &mut HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[crate::input_spec::InputSpec],
    fn_env: &FnEnv,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    let mut call_stack = Vec::new();
    // Functions not allowed in loss blocks (same as train/eval)
    let allow_functions = false;
    // Lower each let binding
    for let_binding in &loss.lets {
        let node_id = lower_expr_with_specs_internal(
            graph,
            &let_binding.expr,
            symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            &mut call_stack,
            allow_functions,
            unroll_budget_used,
        )?;
        symbol_table.insert(let_binding.name.clone(), node_id);
    }

    // Get return node
    let loss_node_id = symbol_table.get(&loss.ret).ok_or_else(|| {
        Diagnostic::new(
            "E_RETURN_NOT_FOUND",
            "Return identifier not found in loss block".to_string(),
        )
        .with_field("name".to_string(), loss.ret.clone())
    })?;

    Ok(*loss_node_id)
}

fn lower_expr(
    graph: &mut Graph,
    expr: &Expr,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    fn_env: &FnEnv,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    let mut call_stack = Vec::new();
    lower_expr_with_specs_internal(
        graph,
        expr,
        symbol_table,
        label_ids_index,
        dim_env,
        &[],
        fn_env,
        &mut call_stack,
        true, // allow_functions = true
        unroll_budget_used,
    )
}

// Public wrapper that allows functions (for model/forward blocks)
fn lower_expr_with_specs(
    graph: &mut Graph,
    expr: &Expr,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[crate::input_spec::InputSpec],
    fn_env: &FnEnv,
    call_stack: &mut Vec<String>,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    lower_expr_with_specs_internal(
        graph,
        expr,
        symbol_table,
        label_ids_index,
        dim_env,
        input_specs,
        fn_env,
        call_stack,
        true, // allow_functions = true for model/forward blocks
        unroll_budget_used,
    )
}

// Internal version with allow_functions flag
fn lower_expr_with_specs_internal(
    graph: &mut Graph,
    expr: &Expr,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[crate::input_spec::InputSpec],
    fn_env: &FnEnv,
    call_stack: &mut Vec<String>,
    allow_functions: bool,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    match expr {
        Expr::String(_) => {
            return Err(Diagnostic::new(
                "E_STRING_NOT_ALLOWED_IN_EXPRESSION",
                "String literals are only allowed in emit statements, not in expressions"
                    .to_string(),
            ));
        }
        Expr::Ident(name) => {
            // Check if this is a loop variable in the current scope
            // For now, we'll handle this in the loop lowering functions
            symbol_table.get(name).copied().ok_or_else(|| {
                Diagnostic::new("E_IDENTIFIER_NOT_FOUND", "Identifier not found".to_string())
                    .with_field("name".to_string(), name.clone())
            })
        }
        Expr::Call(call) => lower_call(
            graph,
            call,
            symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            call_stack,
            allow_functions,
            unroll_budget_used,
        ),
        Expr::If {
            cond,
            then_branch,
            else_branch,
        } => lower_if_expr(
            graph,
            cond,
            then_branch,
            else_branch,
            symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            call_stack,
            unroll_budget_used,
        ),
        Expr::Compare { op, left, right } => lower_compare(
            graph,
            op,
            left,
            right,
            symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            call_stack,
            unroll_budget_used,
        ),
        Expr::Logical { op, left, right } => lower_logical(
            graph,
            op,
            left,
            right,
            symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            call_stack,
            unroll_budget_used,
        ),
        Expr::ForLoop {
            var,
            start,
            end,
            body,
        } => {
            let start_val = extract_int_literal(start, dim_env)?;
            let end_val = extract_int_literal(end, dim_env)?;
            lower_for_loop(
                graph,
                &var,
                start_val as i64,
                end_val as i64,
                body,
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )
        }
        Expr::RepeatLoop {
            count,
            var: _,
            init,
            body,
        } => {
            let count_val = extract_int_literal(count, dim_env)?;
            lower_repeat_loop(
                graph,
                count_val as i64,
                Some(init.as_ref()),
                body,
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )
        }
        Expr::ReduceLoop {
            op,
            var,
            start,
            end,
            body,
        } => {
            let start_val = extract_int_literal(start, dim_env)?;
            let end_val = extract_int_literal(end, dim_env)?;
            lower_reduce_loop(
                graph,
                &op,
                &var,
                start_val as i64,
                end_val as i64,
                body,
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )
        }
        Expr::Int(_) | Expr::Float(_) => Err(Diagnostic::new(
            "E_INVALID_EXPR",
            "Numeric literals cannot be used as expressions".to_string(),
        )),
    }
}

fn lower_call(
    graph: &mut Graph,
    call: &CallExpr,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[crate::input_spec::InputSpec],
    fn_env: &FnEnv,
    call_stack: &mut Vec<String>,
    allow_functions: bool,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    // Check if this is a user-defined function first
    if let Some(func_decl) = fn_env.get(&call.fn_name) {
        // Check if functions are allowed in this context
        if !allow_functions {
            return Err(Diagnostic::new(
                "E_FUNCTION_INVALID_CONTEXT",
                format!("Function '{}' cannot be called in this block", call.fn_name),
            )
            .with_field("function_name".to_string(), call.fn_name.clone())
            .with_field("context".to_string(), "train/eval/dataset/loss".to_string()));
        }

        // Check for undefined function (shouldn't happen if fn_env is correct, but check anyway)
        // This is already handled by the Option check above
        // Check for recursion
        if call_stack.contains(&call.fn_name) {
            return Err(Diagnostic::new(
                "E_FUNCTION_RECURSION",
                format!("Recursive call to function '{}'", call.fn_name),
            )
            .with_field("function_name".to_string(), call.fn_name.clone())
            .with_field(
                "call_site".to_string(),
                format!("{}", call_stack.join(" -> ")),
            ));
        }

        // Check arity
        if call.args.len() != func_decl.params.len() {
            return Err(Diagnostic::new(
                "E_FUNCTION_ARITY_MISMATCH",
                format!(
                    "Function '{}' expects {} arguments, got {}",
                    call.fn_name,
                    func_decl.params.len(),
                    call.args.len()
                ),
            )
            .with_field("function_name".to_string(), call.fn_name.clone())
            .with_field("expected".to_string(), func_decl.params.len().to_string())
            .with_field("received".to_string(), call.args.len().to_string())
            .with_field(
                "call_site".to_string(),
                format!("{}", call_stack.join(" -> ")),
            ));
        }

        // Lower all arguments first
        let mut arg_nodes = Vec::new();
        for arg_expr in &call.args {
            let arg_node = lower_expr_with_specs_internal(
                graph,
                arg_expr,
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                allow_functions, // Functions allowed in function arguments
                unroll_budget_used,
            )?;
            arg_nodes.push(arg_node);
        }

        // Create a new symbol table for function body with parameter substitutions
        let mut fn_symbol_table: HashMap<String, NodeId> = HashMap::new();
        for (param_name, arg_node) in func_decl.params.iter().zip(arg_nodes.iter()) {
            fn_symbol_table.insert(param_name.clone(), *arg_node);
        }

        // Add function to call stack
        call_stack.push(call.fn_name.clone());

        // Lower function body (let bindings) with substituted parameters
        // Functions are allowed in function bodies (they can call other functions)
        let mut last_node_id = None;
        for let_binding in &func_decl.body {
            let node_id = lower_expr_with_specs_internal(
                graph,
                &let_binding.expr,
                &fn_symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                true, // Functions allowed in function bodies
                unroll_budget_used,
            )?;
            fn_symbol_table.insert(let_binding.name.clone(), node_id);
            last_node_id = Some(node_id);
        }

        // Remove function from call stack
        call_stack.pop();

        // Return the last node ID from the function body
        return last_node_id.ok_or_else(|| {
            Diagnostic::new(
                "E_FUNCTION_EMPTY_BODY",
                format!("Function '{}' has an empty body", call.fn_name),
            )
            .with_field("function_name".to_string(), call.fn_name.clone())
        });
    }

    // Check if function is undefined (not in fn_env and not a built-in)
    // We'll check built-ins below, but if it's not a built-in and not in fn_env, it's undefined
    // Actually, we can't know if it's undefined until we check all built-ins, so we'll do that check after the match

    // Built-in functions
    match call.fn_name.as_str() {
        "embedding" => {
            if call.args.len() != 2 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "embedding".to_string())
                .with_field("expected".to_string(), "2".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            // First arg is tokens (should be an input identifier)
            // Second arg is embedding weight (param)
            let tokens_expr = &call.args[0];
            let embedding_expr = &call.args[1];

            // Validate that the first argument is a token_ids input
            let tokens_name = match tokens_expr {
                Expr::Ident(name) => name,
                _ => {
                    return Err(Diagnostic::new(
                        "E_INVALID_ARGUMENTS",
                        "embedding first argument must be an input identifier".to_string(),
                    ));
                }
            };

            // Check if this identifier refers to an input with dtype token_ids
            // Look up the InputSpec by name to get its dtype
            let input_spec = input_specs.iter().find(|spec| spec.name == *tokens_name);

            let received_dtype = if let Some(spec) = input_spec {
                &spec.dtype
            } else {
                // If not found in input_specs, it might be a derived value
                // For now, we'll assume it's not an input and error
                // In a more complete system, we'd track types through the symbol table
                return Err(Diagnostic::new(
                    "E_EMBEDDING_REQUIRES_TOKEN_IDS",
                    "Embedding requires token_ids input".to_string(),
                )
                .with_field("input_name".to_string(), tokens_name.clone())
                .with_field("received_dtype".to_string(), "unknown".to_string())
                .with_hint("Use an input named 'tokens' (dtype token_ids) or declare the input as token_ids.".to_string()));
            };

            if received_dtype != "token_ids" {
                return Err(Diagnostic::new(
                    "E_EMBEDDING_REQUIRES_TOKEN_IDS",
                    "Embedding requires token_ids input".to_string(),
                )
                .with_field("input_name".to_string(), tokens_name.clone())
                .with_field("received_dtype".to_string(), received_dtype.clone())
                .with_hint("Use an input named 'tokens' (dtype token_ids) or declare the input as token_ids.".to_string()));
            }

            let embedding_node_id = lower_expr_with_specs_internal(
                graph,
                embedding_expr,
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                allow_functions, // Built-ins allow functions in args
                unroll_budget_used,
            )?;

            // Create embedding node - it takes the embedding weight as input
            // Token IDs will be passed at runtime via the token_ids side-channel
            // For MVP, we use index 0 for token_ids (assuming first token_ids input)
            let node_id = graph.add_node(Op::Embedding(0), vec![embedding_node_id]);
            Ok(node_id)
        }
        "add" => {
            if call.args.len() != 2 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "add".to_string())
                .with_field("expected".to_string(), "2".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let a_id = lower_expr_with_specs_internal(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                allow_functions,
                unroll_budget_used,
            )?;
            let b_id = lower_expr_with_specs_internal(
                graph,
                &call.args[1],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                allow_functions,
                unroll_budget_used,
            )?;
            let node_id = graph.add_node(Op::Add, vec![a_id, b_id]);
            Ok(node_id)
        }
        "mean_pool_time" | "meanpool" => {
            if call.args.len() != 1 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), call.fn_name.clone())
                .with_field("expected".to_string(), "1".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let x_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let node_id = graph.add_node(Op::MeanPoolTime, vec![x_id]);
            Ok(node_id)
        }
        "matmul" => {
            if call.args.len() != 2 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "matmul".to_string())
                .with_field("expected".to_string(), "2".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let a_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let b_id = lower_expr_with_specs(
                graph,
                &call.args[1],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let node_id = graph.add_node(Op::MatMul2D, vec![a_id, b_id]);
            Ok(node_id)
        }
        "relu" => {
            if call.args.len() != 1 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "relu".to_string())
                .with_field("expected".to_string(), "1".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let x_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let node_id = graph.add_node(Op::ReLU, vec![x_id]);
            Ok(node_id)
        }
        "softmax" => {
            if call.args.len() != 1 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "softmax".to_string())
                .with_field("expected".to_string(), "1".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let x_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;

            // Get axis from kwargs, default to -1 (last dimension)
            // For MVP, try to extract integer literal from expr
            let axis: i32 = call
                .kwargs
                .iter()
                .find(|(name, _)| name == "axis")
                .and_then(|(_, _expr)| {
                    // For MVP, we'd need to evaluate the expr to get the integer
                    // For now, default to last dimension
                    None
                })
                .unwrap_or(-1);

            // For MVP, use axis 0 if axis is negative or not found
            let axis_idx = if axis < 0 {
                // Would need to resolve at runtime, for MVP use 0
                0
            } else {
                axis as usize
            };

            let node_id = graph.add_node(Op::Softmax(axis_idx), vec![x_id]);
            Ok(node_id)
        }
        "now" => {
            if !call.args.is_empty() {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "now".to_string())
                .with_field("expected".to_string(), "0".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let node_id = graph.add_node(Op::Now, vec![]);
            Ok(node_id)
        }
        "cross_entropy" | "xent" => {
            if call.args.len() != 2 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), call.fn_name.clone())
                .with_field("expected".to_string(), "2".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let x_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let logits_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            // Second arg is labels (should be an identifier, but we don't use it as a node)
            // Labels come via token_ids side-channel
            let label_idx = label_ids_index.ok_or_else(|| {
                Diagnostic::new(
                    "E_LABELS_REQUIRED",
                    "Labels input required for cross_entropy".to_string(),
                )
            })?;
            let node_id = graph.add_node(Op::CrossEntropy(label_idx), vec![logits_id]);
            Ok(node_id)
        }
        "linear" => {
            // linear(x, W, b) -> matmul(x, W) + b
            if call.args.len() != 3 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "linear".to_string())
                .with_field("expected".to_string(), "3 (x, W, b)".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let x_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let w_id = lower_expr_with_specs(
                graph,
                &call.args[1],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let b_id = lower_expr_with_specs(
                graph,
                &call.args[2],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            // matmul(x, W)
            let matmul_id = graph.add_node(Op::MatMul2D, vec![x_id, w_id]);
            // matmul(x, W) + b
            let add_id = graph.add_node(Op::Add, vec![matmul_id, b_id]);
            Ok(add_id)
        }
        "concat" => {
            if call.args.len() != 3 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "concat".to_string())
                .with_field("expected".to_string(), "3 (axis, a, b)".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            // Extract axis from first arg - must be integer literal or const
            let axis = extract_int_literal(&call.args[0], &dim_env)?;
            if axis != 1 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "concat only supports axis=1".to_string(),
                ));
            }
            let a_id = lower_expr_with_specs(
                graph,
                &call.args[1],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let b_id = lower_expr_with_specs(
                graph,
                &call.args[2],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let node_id = graph.add_node(Op::Concat { axis }, vec![a_id, b_id]);
            Ok(node_id)
        }
        "slice_rows" => {
            if call.args.len() != 3 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "slice_rows".to_string())
                .with_field("expected".to_string(), "3 (x, start, len)".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let x_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let start = extract_int_literal(&call.args[1], &dim_env)?;
            let len = extract_int_literal(&call.args[2], &dim_env)?;
            let node_id = graph.add_node(Op::SliceRows { start, len }, vec![x_id]);
            Ok(node_id)
        }
        "gather_rows" => {
            if call.args.len() != 2 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "gather_rows".to_string())
                .with_field("expected".to_string(), "2 (x, indices)".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let x_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            // indices come from second arg which should be a TokenIds input identifier
            // The actual indices are provided via token_ids side-channel at runtime
            let node_id = graph.add_node(Op::GatherRows, vec![x_id]);
            // Note: We assume token_ids[0] contains the indices at runtime
            Ok(node_id)
        }
        "dropout" => {
            if call.args.len() != 2 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Invalid arguments".to_string(),
                )
                .with_field("function".to_string(), "dropout".to_string())
                .with_field("expected".to_string(), "2 (x, p)".to_string())
                .with_field("got".to_string(), call.args.len().to_string()));
            }
            let x_id = lower_expr_with_specs(
                graph,
                &call.args[0],
                symbol_table,
                label_ids_index,
                dim_env,
                input_specs,
                fn_env,
                call_stack,
                unroll_budget_used,
            )?;
            let p = extract_float_literal(&call.args[1], &dim_env)?;
            if p < 0.0 || p >= 1.0 {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "dropout p must be in [0.0, 1.0)".to_string(),
                ));
            }
            let node_id = graph.add_node(Op::Dropout { p }, vec![x_id]);
            Ok(node_id)
        }
        _ => {
            // Check if it's an undefined user-defined function
            if fn_env.get(&call.fn_name).is_none() {
                Err(Diagnostic::new(
                    "E_FUNCTION_NOT_FOUND",
                    format!("Function '{}' is not defined", call.fn_name),
                )
                .with_field("function_name".to_string(), call.fn_name.clone())
                .with_field("call_site".to_string(), "unknown".to_string()))
            } else {
                Err(
                    Diagnostic::new("E_UNKNOWN_FUNCTION", "Unknown function".to_string())
                        .with_field("name".to_string(), call.fn_name.clone()),
                )
            }
        }
    }
}

// Helper to extract integer literal from Expr
fn extract_int_literal(expr: &Expr, dim_env: &DimEnv) -> Result<usize, Diagnostic> {
    match expr {
        Expr::Int(n) => {
            if *n < 0 {
                Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Integer literal must be non-negative".to_string(),
                ))
            } else {
                Ok(*n as usize)
            }
        }
        Expr::Ident(name) => {
            // Try to resolve as const
            dim_env.consts.get(name).copied().ok_or_else(|| {
                Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    format!(
                        "Expected integer literal or const, got identifier: {}",
                        name
                    ),
                )
            })
        }
        Expr::Float(_) => Err(Diagnostic::new(
            "E_INVALID_ARGUMENTS",
            "Expected integer literal, got float".to_string(),
        )),
        Expr::Call(_) => Err(Diagnostic::new(
            "E_INVALID_ARGUMENTS",
            "Expected integer literal, got function call".to_string(),
        )),
        Expr::If { .. }
        | Expr::Compare { .. }
        | Expr::Logical { .. }
        | Expr::ForLoop { .. }
        | Expr::RepeatLoop { .. }
        | Expr::ReduceLoop { .. } => Err(Diagnostic::new(
            "E_INVALID_ARGUMENTS",
            "Expected integer literal, got conditional or loop expression".to_string(),
        )),
        Expr::String(_) => Err(Diagnostic::new(
            "E_INVALID_ARGUMENTS",
            "Expected integer literal, got string".to_string(),
        )),
    }
}

// Helper to extract float literal from Expr
fn extract_float_literal(expr: &Expr, dim_env: &DimEnv) -> Result<f32, Diagnostic> {
    match expr {
        Expr::Float(f) => Ok(*f),
        Expr::Int(n) => Ok(*n as f32),
        Expr::Ident(name) => {
            // Try to resolve as const and convert to float
            dim_env
                .consts
                .get(name)
                .copied()
                .map(|v| v as f32)
                .ok_or_else(|| {
                    Diagnostic::new(
                        "E_INVALID_ARGUMENTS",
                        format!("Expected float literal or const, got identifier: {}", name),
                    )
                })
        }
        Expr::Call(_) => Err(Diagnostic::new(
            "E_INVALID_ARGUMENTS",
            "Expected float literal, got function call".to_string(),
        )),
        Expr::If { .. }
        | Expr::Compare { .. }
        | Expr::Logical { .. }
        | Expr::ForLoop { .. }
        | Expr::RepeatLoop { .. }
        | Expr::ReduceLoop { .. } => Err(Diagnostic::new(
            "E_INVALID_ARGUMENTS",
            "Expected float literal, got conditional or loop expression".to_string(),
        )),
        Expr::String(_) => Err(Diagnostic::new(
            "E_INVALID_ARGUMENTS",
            "Expected float literal, got string".to_string(),
        )),
    }
}

fn lower_if_expr(
    graph: &mut Graph,
    cond: &Expr,
    then_branch: &Expr,
    else_branch: &Expr,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[InputSpec],
    fn_env: &FnEnv,
    call_stack: &mut Vec<String>,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    // Lower condition - must be scalar-compatible
    let cond_id = lower_expr_with_specs(
        graph,
        cond,
        symbol_table,
        label_ids_index,
        dim_env,
        input_specs,
        fn_env,
        call_stack,
        unroll_budget_used,
    )?;

    // Lower both branches
    let then_id = lower_expr_with_specs(
        graph,
        then_branch,
        symbol_table,
        label_ids_index,
        dim_env,
        input_specs,
        fn_env,
        call_stack,
        unroll_budget_used,
    )?;

    let else_id = lower_expr_with_specs(
        graph,
        else_branch,
        symbol_table,
        label_ids_index,
        dim_env,
        input_specs,
        fn_env,
        call_stack,
        unroll_budget_used,
    )?;

    // Validate branches have same shape (runtime check will also verify, but we can check statically if possible)
    // For now, we'll rely on runtime validation in the kernel

    // Create If node: inputs are [condition, then_branch, else_branch]
    let node_id = graph.add_node(Op::If {}, vec![cond_id, then_id, else_id]);
    Ok(node_id)
}

fn lower_compare(
    graph: &mut Graph,
    op: &CompareOp,
    left: &Expr,
    right: &Expr,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[InputSpec],
    fn_env: &FnEnv,
    call_stack: &mut Vec<String>,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    let left_id = lower_expr_with_specs(
        graph,
        left,
        symbol_table,
        label_ids_index,
        dim_env,
        input_specs,
        fn_env,
        call_stack,
        unroll_budget_used,
    )?;

    let right_id = lower_expr_with_specs(
        graph,
        right,
        symbol_table,
        label_ids_index,
        dim_env,
        input_specs,
        fn_env,
        call_stack,
        unroll_budget_used,
    )?;

    let op_kind = match op {
        CompareOp::Eq => crate::ir::CompareOp::Eq,
        CompareOp::Ne => crate::ir::CompareOp::Ne,
        CompareOp::Lt => crate::ir::CompareOp::Lt,
        CompareOp::Le => crate::ir::CompareOp::Le,
        CompareOp::Gt => crate::ir::CompareOp::Gt,
        CompareOp::Ge => crate::ir::CompareOp::Ge,
    };

    let node_id = graph.add_node(Op::Compare { op: op_kind }, vec![left_id, right_id]);
    Ok(node_id)
}

fn lower_logical(
    graph: &mut Graph,
    op: &LogicalOp,
    left: &Expr,
    right: &Option<Box<Expr>>,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[InputSpec],
    fn_env: &FnEnv,
    call_stack: &mut Vec<String>,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    let left_id = lower_expr_with_specs(
        graph,
        left,
        symbol_table,
        label_ids_index,
        dim_env,
        input_specs,
        fn_env,
        call_stack,
        unroll_budget_used,
    )?;

    let op_kind = match op {
        LogicalOp::Not => {
            if right.is_some() {
                return Err(Diagnostic::new(
                    "E_INVALID_ARGUMENTS",
                    "Logical 'not' takes only one argument".to_string(),
                ));
            }
            let node_id = graph.add_node(
                Op::Logical {
                    op: crate::ir::LogicalOp::Not,
                },
                vec![left_id],
            );
            return Ok(node_id);
        }
        LogicalOp::And => crate::ir::LogicalOp::And,
        LogicalOp::Or => crate::ir::LogicalOp::Or,
    };

    let right_expr = right.as_ref().ok_or_else(|| {
        Diagnostic::new(
            "E_INVALID_ARGUMENTS",
            "Logical 'and' and 'or' require two arguments".to_string(),
        )
    })?;
    let right_id = lower_expr_with_specs(
        graph,
        right_expr,
        symbol_table,
        label_ids_index,
        dim_env,
        input_specs,
        fn_env,
        call_stack,
        unroll_budget_used,
    )?;

    let node_id = graph.add_node(Op::Logical { op: op_kind }, vec![left_id, right_id]);
    Ok(node_id)
}

fn lower_for_loop(
    graph: &mut Graph,
    var: &str,
    start: i64,
    end: i64,
    body: &Expr,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[InputSpec],
    fn_env: &FnEnv,
    call_stack: &mut Vec<String>,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    // For loops are unrolled: create a node for each iteration
    let iterations = (end - start) as usize;
    if iterations == 0 {
        return Err(Diagnostic::new(
            "E_LOOP_INVALID_RANGE",
            "Loop range is empty".to_string(),
        ));
    }

    // Check unroll limit
    *unroll_budget_used += iterations;
    if *unroll_budget_used > 10_000 {
        return Err(Diagnostic::new(
            "E_LOOP_UNROLL_LIMIT_EXCEEDED",
            "Loop unroll limit exceeded".to_string(),
        )
        .with_field("limit".to_string(), "10000".to_string())
        .with_field("requested".to_string(), unroll_budget_used.to_string())
        .with_field(
            "context".to_string(),
            format!("for loop: {} iterations", iterations),
        ));
    }

    // Collect all iteration results
    let mut body_nodes = Vec::new();

    for i in 0..iterations {
        let loop_value = (start + i as i64) as f32;

        // Create a constant scalar node for the loop variable
        let loop_var_node = graph.add_node(Op::ConstScalar { value: loop_value }, vec![]);

        // Add loop variable to symbol table for this iteration
        let mut loop_symbol_table = symbol_table.clone();
        loop_symbol_table.insert(var.to_string(), loop_var_node);

        // Lower body with loop variable in scope
        let body_node_id = lower_expr_with_specs(
            graph,
            body,
            &loop_symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            call_stack,
            unroll_budget_used,
        )?;

        body_nodes.push(body_node_id);
    }

    if body_nodes.is_empty() {
        return Err(Diagnostic::new(
            "E_LOOP_BODY_INVALID_RETURN",
            "Loop body must produce a value".to_string(),
        ));
    }

    // Stack all results along axis 0
    let stack_node = graph.add_node(Op::Stack { axis: 0 }, body_nodes);

    Ok(stack_node)
}

fn lower_repeat_loop(
    graph: &mut Graph,
    count: i64,
    init: Option<&Expr>,
    body: &Expr,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[InputSpec],
    fn_env: &FnEnv,
    call_stack: &mut Vec<String>,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    // Repeat loops are unrolled: create the body N times
    let iterations = count as usize;
    if iterations == 0 {
        return Err(Diagnostic::new(
            "E_REPEAT_INVALID_COUNT",
            "Repeat count must be greater than 0".to_string(),
        ));
    }

    // Check unroll limit
    *unroll_budget_used += iterations;
    if *unroll_budget_used > 10_000 {
        return Err(Diagnostic::new(
            "E_LOOP_UNROLL_LIMIT_EXCEEDED",
            "Loop unroll limit exceeded".to_string(),
        )
        .with_field("limit".to_string(), "10000".to_string())
        .with_field("requested".to_string(), unroll_budget_used.to_string())
        .with_field(
            "context".to_string(),
            format!("repeat loop: {} iterations", iterations),
        ));
    }

    // Lower init expression
    let mut current_value = if let Some(init_expr) = init {
        lower_expr_with_specs(
            graph,
            init_expr,
            symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            call_stack,
            unroll_budget_used,
        )?
    } else {
        return Err(Diagnostic::new(
            "E_REPEAT_REQUIRES_INIT",
            "Repeat loop requires an 'init' expression".to_string(),
        ));
    };

    // Unroll: y1 = f(y0), y2 = f(y1), ...
    // For repeat loops, we need to identify the accumulator variable in the body
    // For MVP, we'll assume the body expression references variables from symbol_table
    // and we'll need to substitute the accumulator variable with current_value
    // This is simplified - in full implementation, we'd parse the body to find variable references

    // For now, we'll use a simple approach: assume the body can reference a variable
    // that we'll add to the symbol table as the accumulator
    // The body expression should use the same variable name that was used in init
    // For MVP, let's assume it's a simple expression that we can lower

    for _i in 0..iterations {
        // Create a symbol table with current_value as the accumulator
        // We need to identify which variable in the body refers to the accumulator
        // For MVP, we'll use a heuristic: if body is a simple call or expression,
        // we'll try to substitute common accumulator variable names

        // Simplified: just lower the body and use it as the next value
        // In full implementation, we'd need expression rewriting to substitute variables
        let mut loop_symbol_table = symbol_table.clone();
        // Add current_value to symbol table - we'll use a placeholder name
        // In full implementation, we'd parse the body to find which variable to substitute
        loop_symbol_table.insert("y".to_string(), current_value);

        let body_node_id = lower_expr_with_specs(
            graph,
            body,
            &loop_symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            call_stack,
            unroll_budget_used,
        )?;

        current_value = body_node_id;
    }

    Ok(current_value)
}

fn lower_reduce_loop(
    graph: &mut Graph,
    op: &str,
    var: &str,
    start: i64,
    end: i64,
    body: &Expr,
    symbol_table: &HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[InputSpec],
    fn_env: &FnEnv,
    call_stack: &mut Vec<String>,
    unroll_budget_used: &mut usize,
) -> Result<NodeId, Diagnostic> {
    // Reduce loops: unroll and apply reduction operator
    let iterations = (end - start) as usize;
    if iterations == 0 {
        return Err(Diagnostic::new(
            "E_LOOP_INVALID_RANGE",
            "Loop range is empty".to_string(),
        ));
    }

    // Check unroll limit
    *unroll_budget_used += iterations;
    if *unroll_budget_used > 10_000 {
        return Err(Diagnostic::new(
            "E_LOOP_UNROLL_LIMIT_EXCEEDED",
            "Loop unroll limit exceeded".to_string(),
        )
        .with_field("limit".to_string(), "10000".to_string())
        .with_field("requested".to_string(), unroll_budget_used.to_string())
        .with_field(
            "context".to_string(),
            format!("reduce loop: {} iterations", iterations),
        ));
    }

    // Unroll body for each iteration with loop variable substitution
    let mut body_nodes = Vec::new();

    for i in 0..iterations {
        let loop_value = (start + i as i64) as f32;

        // Create a constant scalar node for the loop variable
        let loop_var_node = graph.add_node(Op::ConstScalar { value: loop_value }, vec![]);

        // Add loop variable to symbol table for this iteration
        let mut loop_symbol_table = symbol_table.clone();
        loop_symbol_table.insert(var.to_string(), loop_var_node);

        // Lower body with loop variable in scope
        let body_node_id = lower_expr_with_specs(
            graph,
            body,
            &loop_symbol_table,
            label_ids_index,
            dim_env,
            input_specs,
            fn_env,
            call_stack,
            unroll_budget_used,
        )?;

        body_nodes.push(body_node_id);
    }

    if body_nodes.is_empty() {
        return Err(Diagnostic::new(
            "E_LOOP_BODY_INVALID_RETURN",
            "Loop body must produce a value".to_string(),
        ));
    }

    // Apply reduction operator
    match op {
        "add" => {
            // For add reduction, accumulate all body nodes
            let mut result_id = body_nodes[0];
            for &node_id in &body_nodes[1..] {
                result_id = graph.add_node(Op::Add, vec![result_id, node_id]);
            }
            Ok(result_id)
        }
        "max" => {
            // Max reduction: fold with Max2
            let mut result_id = body_nodes[0];
            for &node_id in &body_nodes[1..] {
                result_id = graph.add_node(Op::Max2, vec![result_id, node_id]);
            }
            Ok(result_id)
        }
        "min" => {
            // Min reduction: fold with Min2
            let mut result_id = body_nodes[0];
            for &node_id in &body_nodes[1..] {
                result_id = graph.add_node(Op::Min2, vec![result_id, node_id]);
            }
            Ok(result_id)
        }
        _ => Err(Diagnostic::new(
            "E_REDUCE_OP_INVALID",
            format!("Unknown reduction operator: {}", op),
        )),
    }
}

// Check for undefined function (after all built-ins checked)
// This should be called from the default case of the built-in match
fn check_undefined_function(fn_name: &str, fn_env: &FnEnv) -> Result<(), Diagnostic> {
    // If it's not in fn_env, it's undefined
    if fn_env.get(fn_name).is_none() {
        return Err(Diagnostic::new(
            "E_FUNCTION_NOT_FOUND",
            format!("Function '{}' is not defined", fn_name),
        )
        .with_field("function_name".to_string(), fn_name.to_string())
        .with_field("call_site".to_string(), "unknown".to_string()));
    }
    Ok(())
}

// Validation and lowering for new event-driven runtime model

/// Validate state block: check for duplicate state variables
fn validate_state_block(
    state_block: &StateBlock,
    event_handlers: &[OnEventHandler],
) -> Result<(), Diagnostic> {
    let mut var_names = std::collections::HashSet::new();

    for var in &state_block.vars {
        if var_names.contains(&var.name) {
            return Err(Diagnostic::new(
                "E_STATE_REDECLARED",
                format!("State variable '{}' is declared multiple times", var.name),
            )
            .with_field("var_name".to_string(), var.name.clone()));
        }
        var_names.insert(var.name.clone());
    }

    // Validate that event handlers only assign to declared state variables
    for handler in event_handlers {
        for statement in &handler.body {
            if let EventHandlerStatement::Assign(var_name, _) = statement {
                if !var_names.contains(var_name) {
                    return Err(Diagnostic::new(
                        "E_STATE_NOT_DECLARED",
                        format!(
                            "Event handler '{}' assigns to undeclared state variable '{}'",
                            handler.event_name, var_name
                        ),
                    )
                    .with_field("event_name".to_string(), handler.event_name.clone())
                    .with_field("var_name".to_string(), var_name.clone()));
                }
            }
        }
    }

    Ok(())
}

/// Validate event handlers: check for duplicate events, undefined events in emit, cycle detection
fn validate_event_handlers(
    event_handlers: &[OnEventHandler],
    state_block: &Option<StateBlock>,
) -> Result<(), Diagnostic> {
    let mut event_names = std::collections::HashSet::new();

    // Check for duplicate event handlers
    for handler in event_handlers {
        if event_names.contains(&handler.event_name) {
            return Err(Diagnostic::new(
                "E_EVENT_DUPLICATE",
                format!(
                    "Event handler '{}' is defined multiple times",
                    handler.event_name
                ),
            )
            .with_field("event_name".to_string(), handler.event_name.clone()));
        }
        event_names.insert(handler.event_name.clone());
    }

    // Build emit graph and detect cycles
    let mut emit_graph: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for handler in event_handlers {
        let mut emitted_events = Vec::new();
        for statement in &handler.body {
            if let EventHandlerStatement::EmitEvent(event_name) = statement {
                emitted_events.push(event_name.clone());
                // Check if emitted event exists
                if !event_names.contains(event_name) {
                    return Err(Diagnostic::new(
                        "E_EVENT_NOT_DEFINED",
                        format!(
                            "Event handler '{}' emits undefined event '{}'",
                            handler.event_name, event_name
                        ),
                    )
                    .with_field("event_name".to_string(), handler.event_name.clone())
                    .with_field("emitted_event".to_string(), event_name.clone()));
                }
            }
        }
        if !emitted_events.is_empty() {
            emit_graph.insert(handler.event_name.clone(), emitted_events);
        }
    }

    // Detect cycles in emit graph (simple DFS)
    for event in event_names.iter() {
        let mut visited = std::collections::HashSet::new();
        let mut path = Vec::new();
        if detect_emit_cycle(&emit_graph, event, &mut visited, &mut path) {
            return Err(Diagnostic::new(
                "E_EVENT_CYCLE_DETECTED",
                format!("Event emission cycle detected: {}", path.join(" -> ")),
            )
            .with_field("cycle".to_string(), path.join(" -> ")));
        }
    }

    Ok(())
}

/// Detect cycles in emit graph using DFS
fn detect_emit_cycle(
    graph: &std::collections::HashMap<String, Vec<String>>,
    current: &str,
    visited: &mut std::collections::HashSet<String>,
    path: &mut Vec<String>,
) -> bool {
    if path.contains(&current.to_string()) {
        // Cycle detected
        path.push(current.to_string());
        return true;
    }

    if visited.contains(current) {
        return false;
    }

    visited.insert(current.to_string());
    path.push(current.to_string());

    if let Some(emitted) = graph.get(current) {
        for next_event in emitted {
            if detect_emit_cycle(graph, next_event, visited, path) {
                return true;
            }
        }
    }

    path.pop();
    false
}

/// Lower state block to config
fn lower_state_block(state_block: &StateBlock) -> Result<StateBlockConfig, Diagnostic> {
    let vars = state_block
        .vars
        .iter()
        .map(|var| {
            let var_type = match var.var_type {
                ASTStateVarType::Int => StateVarType::Int,
                ASTStateVarType::Float => StateVarType::Float,
                ASTStateVarType::Tensor => StateVarType::Tensor,
            };
            StateVarConfig {
                name: var.name.clone(),
                var_type,
                has_init: var.init_value.is_some(),
            }
        })
        .collect();

    Ok(StateBlockConfig { vars })
}

/// Lower event handlers to config
fn lower_event_handlers(
    event_handlers: &[OnEventHandler],
) -> Result<Vec<EventHandlerConfig>, Diagnostic> {
    let mut configs = Vec::new();

    for handler in event_handlers {
        let mut statements = Vec::new();

        for stmt in &handler.body {
            match stmt {
                EventHandlerStatement::Assign(var_name, _) => {
                    statements.push(EventHandlerStatementConfig::Assign(var_name.clone()));
                }
                EventHandlerStatement::EmitEvent(event_name) => {
                    statements.push(EventHandlerStatementConfig::EmitEvent(event_name.clone()));
                }
                EventHandlerStatement::Let(_) => {
                    // Let bindings are lowered to graph nodes elsewhere
                    // For config, we skip them (they don't need separate config entries)
                    continue;
                }
                EventHandlerStatement::Expr(_) => {
                    // Expr statements are not stored in config (unused for now)
                    continue;
                }
            }
        }

        configs.push(EventHandlerConfig {
            event_name: handler.event_name.clone(),
            statements,
        });
    }

    Ok(configs)
}

// Validate state machine structure
fn validate_state_machine(sm: &StateMachineDecl) -> Result<(), Diagnostic> {
    let mut state_names = std::collections::HashSet::new();

    // Check for duplicate states
    for state in &sm.states {
        if state_names.contains(&state.name) {
            return Err(Diagnostic::new(
                "E_STATE_DUPLICATE",
                format!("Duplicate state name: '{}'", state.name),
            )
            .with_field("state_name".to_string(), state.name.clone()));
        }
        state_names.insert(state.name.clone());
    }

    // Check that all transition targets are valid states
    for state in &sm.states {
        for (idx, transition) in state.transitions.iter().enumerate() {
            if !state_names.contains(&transition.target) {
                return Err(Diagnostic::new(
                    "E_STATE_NOT_FOUND",
                    format!(
                        "Transition target '{}' is not a valid state",
                        transition.target
                    ),
                )
                .with_field("state_name".to_string(), state.name.clone())
                .with_field("target".to_string(), transition.target.clone())
                .with_field("transition_index".to_string(), idx.to_string()));
            }
        }

        // Check that state has at least one transition or end
        if state.transitions.is_empty() && !state.has_end {
            return Err(Diagnostic::new(
                "E_STATE_NO_TRANSITION",
                format!("State '{}' has no transitions and no 'end'", state.name),
            )
            .with_field("state_name".to_string(), state.name.clone()));
        }

        // Check for transition conflicts (multiple if/else without proper ordering)
        let mut has_if = false;
        let mut has_else = false;
        for transition in &state.transitions {
            match &transition.kind {
                ASTTransitionKind::If => {
                    if has_else {
                        return Err(Diagnostic::new(
                            "E_TRANSITION_CONFLICT",
                            format!(
                                "State '{}' has 'else' transition before 'if' transition",
                                state.name
                            ),
                        )
                        .with_field("state_name".to_string(), state.name.clone()));
                    }
                    has_if = true;
                }
                ASTTransitionKind::Else => {
                    if !has_if {
                        return Err(Diagnostic::new(
                            "E_TRANSITION_CONFLICT",
                            format!(
                                "State '{}' has 'else' transition without preceding 'if'",
                                state.name
                            ),
                        )
                        .with_field("state_name".to_string(), state.name.clone()));
                    }
                    has_else = true;
                }
                _ => {}
            }
        }
    }

    Ok(())
}

// Lower state machine into subgraphs
fn lower_state_machine(
    graph: &mut Graph,
    sm: &StateMachineDecl,
    symbol_table: &mut HashMap<String, NodeId>,
    label_ids_index: Option<usize>,
    dim_env: &DimEnv,
    input_specs: &[InputSpec],
    fn_env: &FnEnv,
    unroll_budget_used: &mut usize,
) -> Result<StateMachineConfig, Diagnostic> {
    let mut states = Vec::new();

    for state_decl in &sm.states {
        let mut transitions = Vec::new();

        // Lower transitions
        for transition_decl in &state_decl.transitions {
            let condition_node = match &transition_decl.kind {
                ASTTransitionKind::If => {
                    // Lower the condition expression
                    if let Some(ref condition_expr) = transition_decl.condition {
                        let mut call_stack = Vec::new();
                        Some(lower_expr_with_specs_internal(
                            graph,
                            condition_expr,
                            symbol_table,
                            label_ids_index,
                            dim_env,
                            input_specs,
                            fn_env,
                            &mut call_stack,
                            true, // Allow functions in state machine conditions
                            unroll_budget_used,
                        )?)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            transitions.push(TransitionConfig {
                kind: match &transition_decl.kind {
                    ASTTransitionKind::On(event) => TransitionKind::On(event.clone()),
                    ASTTransitionKind::If => TransitionKind::If,
                    ASTTransitionKind::Else => TransitionKind::Else,
                },
                target: transition_decl.target.clone(),
                condition_node,
            });
        }

        states.push(StateConfig {
            name: state_decl.name.clone(),
            transitions,
            has_end: state_decl.has_end,
        });
    }

    Ok(StateMachineConfig {
        name: sm.name.clone(),
        states,
    })
}

// Validate dataset pipeline functions
fn validate_dataset_pipeline(dataset: &DatasetDecl, fn_env: &FnEnv) -> Result<(), Diagnostic> {
    // Validate map functions
    for (idx, map_call) in dataset.maps.iter().enumerate() {
        // Check function exists
        let func_decl = fn_env.get(&map_call.fn_name).ok_or_else(|| {
            Diagnostic::new(
                "E_DATASET_MAP_NOT_FOUND",
                format!("Map function '{}' is not defined", map_call.fn_name),
            )
            .with_field("function_name".to_string(), map_call.fn_name.clone())
            .with_field("stage".to_string(), "map".to_string())
            .with_field("dataset_path".to_string(), dataset.path.clone())
        })?;

        // Check arity (map functions should take 1 argument: the sample)
        if map_call.args.len() != 1 {
            return Err(Diagnostic::new(
                "E_FUNCTION_ARITY_MISMATCH",
                format!(
                    "Map function '{}' expects 1 argument (sample), got {}",
                    map_call.fn_name,
                    map_call.args.len()
                ),
            )
            .with_field("function_name".to_string(), map_call.fn_name.clone())
            .with_field("expected".to_string(), "1".to_string())
            .with_field("received".to_string(), map_call.args.len().to_string())
            .with_field("stage".to_string(), "map".to_string())
            .with_field("dataset_path".to_string(), dataset.path.clone()));
        }

        // Check function arity matches
        if func_decl.params.len() != 1 {
            return Err(Diagnostic::new(
                "E_FUNCTION_ARITY_MISMATCH",
                format!(
                    "Map function '{}' must take exactly 1 parameter, got {}",
                    map_call.fn_name,
                    func_decl.params.len()
                ),
            )
            .with_field("function_name".to_string(), map_call.fn_name.clone())
            .with_field("expected".to_string(), "1".to_string())
            .with_field("received".to_string(), func_decl.params.len().to_string())
            .with_field("stage".to_string(), "map".to_string())
            .with_field("dataset_path".to_string(), dataset.path.clone()));
        }
    }

    // Validate filter functions
    for (idx, filter_call) in dataset.filters.iter().enumerate() {
        // Check function exists
        let func_decl = fn_env.get(&filter_call.fn_name).ok_or_else(|| {
            Diagnostic::new(
                "E_DATASET_FILTER_NOT_FOUND",
                format!("Filter function '{}' is not defined", filter_call.fn_name),
            )
            .with_field("function_name".to_string(), filter_call.fn_name.clone())
            .with_field("stage".to_string(), "filter".to_string())
            .with_field("dataset_path".to_string(), dataset.path.clone())
        })?;

        // Check arity (filter functions should take 1 argument: the sample)
        if filter_call.args.len() != 1 {
            return Err(Diagnostic::new(
                "E_FUNCTION_ARITY_MISMATCH",
                format!(
                    "Filter function '{}' expects 1 argument (sample), got {}",
                    filter_call.fn_name,
                    filter_call.args.len()
                ),
            )
            .with_field("function_name".to_string(), filter_call.fn_name.clone())
            .with_field("expected".to_string(), "1".to_string())
            .with_field("received".to_string(), filter_call.args.len().to_string())
            .with_field("stage".to_string(), "filter".to_string())
            .with_field("dataset_path".to_string(), dataset.path.clone()));
        }

        // Check function arity matches
        if func_decl.params.len() != 1 {
            return Err(Diagnostic::new(
                "E_FUNCTION_ARITY_MISMATCH",
                format!(
                    "Filter function '{}' must take exactly 1 parameter, got {}",
                    filter_call.fn_name,
                    func_decl.params.len()
                ),
            )
            .with_field("function_name".to_string(), filter_call.fn_name.clone())
            .with_field("expected".to_string(), "1".to_string())
            .with_field("received".to_string(), func_decl.params.len().to_string())
            .with_field("stage".to_string(), "filter".to_string())
            .with_field("dataset_path".to_string(), dataset.path.clone()));
        }

        // Note: We can't statically verify that filter returns boolean,
        // but we'll check at runtime. For now, we just validate the function exists.
    }

    Ok(())
}

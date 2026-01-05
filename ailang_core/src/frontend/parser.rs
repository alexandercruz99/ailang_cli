use super::lexer::{Token, TokenKind};
use crate::diagnostic::Diagnostic;

#[derive(Clone, Debug)]
pub struct Program {
    pub app: Option<AppDecl>, // Optional app wrapper (for backward compatibility)
    pub functions: Vec<FnDecl>,
    pub consts: Vec<ConstDecl>,
    pub inputs: Vec<InputDecl>,
    pub params: Vec<ParamDecl>,
    pub forward: ForwardBlock,
    pub loss: Option<LossBlock>,
    pub train: Option<TrainBlock>,
    pub dataset: Option<DatasetDecl>,
    pub eval: Option<EvalBlock>,
    pub state_machine: Option<StateMachineDecl>,
    // New event-driven runtime model (for backward compatibility, only in app block)
    pub state_block: Option<StateBlock>,
    pub event_handlers: Vec<OnEventHandler>,
}

#[derive(Clone, Debug)]
pub struct AppDecl {
    pub name: String,
    pub functions: Vec<FnDecl>,
    pub consts: Vec<ConstDecl>,
    pub inputs: Vec<InputDecl>,
    pub params: Vec<ParamDecl>,
    pub forward: ForwardBlock,
    pub loss: Option<LossBlock>,
    pub train: Option<TrainBlock>,
    pub dataset: Option<DatasetDecl>,
    pub eval: Option<EvalBlock>,
    pub state_machine: Option<StateMachineDecl>,
    // New event-driven runtime model
    pub state_block: Option<StateBlock>,
    pub event_handlers: Vec<OnEventHandler>,
}

#[derive(Clone, Debug)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<String>,   // Parameter names
    pub body: Vec<LetBinding>, // Function body (let bindings)
}

#[derive(Clone, Debug)]
pub struct ConstDecl {
    pub name: String,
    pub value: i64,
}

#[derive(Clone, Debug)]
pub struct InputDecl {
    pub name: String,
    pub shape: Vec<DimSpec>,
}

#[derive(Clone, Debug)]
pub struct ParamDecl {
    pub name: String,
    pub shape: Vec<DimSpec>,
}

#[derive(Clone, Debug)]
pub enum DimSpec {
    Named(String),
    Literal(usize),
    Free,
}

#[derive(Clone, Debug)]
pub struct ForwardBlock {
    pub lets: Vec<LetBinding>,
    pub ret: String,
}

#[derive(Clone, Debug)]
pub struct LossBlock {
    pub lets: Vec<LetBinding>,
    pub ret: String,
}

#[derive(Clone, Debug)]
pub struct TrainBlock {
    pub steps: Option<i64>,
    pub lr: Option<f32>,
    pub batch: Option<i64>,
    pub loss_expr: Option<Expr>,
}

#[derive(Clone, Debug)]
pub struct DatasetDecl {
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

#[derive(Clone, Debug)]
pub struct EvalBlock {
    pub metrics: Vec<String>,
    pub every: Option<i64>,
    pub split: Option<String>,
}

#[derive(Clone, Debug)]
pub struct StateMachineDecl {
    pub name: String,
    pub states: Vec<StateDecl>,
}

#[derive(Clone, Debug)]
pub struct StateDecl {
    pub name: String,
    pub body: Vec<StateStatement>,
    pub transitions: Vec<TransitionDecl>,
    pub has_end: bool,
}

#[derive(Clone, Debug)]
pub enum StateStatement {
    Let(LetBinding),
    Emit(EmitDecl),
}

#[derive(Clone, Debug)]
pub struct EmitDecl {
    pub value: Expr,
}

#[derive(Clone, Debug)]
pub struct TransitionDecl {
    pub kind: TransitionKind,
    pub target: String,
    pub condition: Option<Expr>, // For if transitions
}

#[derive(Clone, Debug, PartialEq)]
pub enum TransitionKind {
    On(String), // on event -> State
    If,         // if condition -> State
    Else,       // else -> State
}

#[derive(Clone, Debug)]
pub struct LetBinding {
    pub name: String,
    pub expr: Expr,
}

// New event-driven runtime model structures
#[derive(Clone, Debug)]
pub struct StateBlock {
    pub vars: Vec<StateVarDecl>,
}

#[derive(Clone, Debug)]
pub struct StateVarDecl {
    pub name: String,
    pub var_type: StateVarType,
    pub init_value: Option<Expr>, // Optional initial value
}

#[derive(Clone, Debug)]
pub enum StateVarType {
    Int,
    Float,
    Tensor, // Tensor type (from model outputs)
}

#[derive(Clone, Debug)]
pub struct OnEventHandler {
    pub event_name: String,
    pub body: Vec<EventHandlerStatement>,
}

#[derive(Clone, Debug)]
pub enum EventHandlerStatement {
    Let(LetBinding),
    Assign(String, Expr), // State assignment: name = expr
    EmitEvent(String),    // emit <event_name>
    Expr(Expr),           // Expression statement (for side effects, currently unused)
}

#[derive(Clone, Debug)]
pub enum Expr {
    Ident(String),
    Call(CallExpr),
    Int(i64),
    Float(f32),
    String(String),
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
    Compare {
        op: CompareOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Logical {
        op: LogicalOp,
        left: Box<Expr>,
        right: Option<Box<Expr>>,
    },
    ForLoop {
        var: String,
        start: Box<Expr>,
        end: Box<Expr>,
        body: Box<Expr>,
    },
    RepeatLoop {
        count: Box<Expr>,
        var: String,
        init: Box<Expr>,
        body: Box<Expr>,
    },
    ReduceLoop {
        op: String, // "add", "max", "min"
        var: String,
        start: Box<Expr>,
        end: Box<Expr>,
        body: Box<Expr>,
    },
}

#[derive(Clone, Debug)]
pub struct CallExpr {
    pub fn_name: String,
    pub args: Vec<Expr>,
    pub kwargs: Vec<(String, Expr)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CompareOp {
    Eq, // ==
    Ne, // !=
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=
}

#[derive(Clone, Debug, PartialEq)]
pub enum LogicalOp {
    And,
    Or,
    Not,
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    program: Option<Program>, // For const lookup during parsing
    allow_conditionals: bool, // Whether conditionals are allowed in current context
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            program: None,
            allow_conditionals: true,
        }
    }

    pub fn parse(&mut self) -> Result<Program, Diagnostic> {
        // Check if this is an app block
        if self.peek_kind() == TokenKind::App {
            return self.parse_app_block();
        }

        // Legacy mode: parse without app wrapper
        let mut functions = Vec::new();
        let mut consts = Vec::new();
        let mut inputs = Vec::new();
        let mut params = Vec::new();

        // Parse functions, const, input, param, and dataset/data declarations (any order)
        // Functions must appear before model/forward blocks
        let mut dataset = None;
        while !self.is_eof() {
            match self.peek_kind() {
                TokenKind::Fn => {
                    let fn_decl = self.parse_fn_decl()?;
                    // Check for duplicate function names
                    if functions.iter().any(|f: &FnDecl| f.name == fn_decl.name) {
                        return Err(Diagnostic::new(
                            "E_FUNCTION_DUPLICATE_NAME",
                            format!("Function '{}' is already defined", fn_decl.name),
                        )
                        .with_field("function_name".to_string(), fn_decl.name.clone()));
                    }
                    functions.push(fn_decl);
                }
                TokenKind::Const => {
                    consts.push(self.parse_const_decl()?);
                }
                TokenKind::Input => {
                    inputs.push(self.parse_input_decl()?);
                }
                TokenKind::Param => {
                    params.push(self.parse_param_decl()?);
                }
                TokenKind::Dataset | TokenKind::Data => {
                    // Dataset/Data can appear before forward/model
                    if dataset.is_some() {
                        return Err(Diagnostic::new(
                            "E_DUPLICATE_DATASET_BLOCK",
                            "Only one dataset/data block is allowed".to_string(),
                        ));
                    }
                    dataset = Some(self.parse_dataset_block()?);
                }
                TokenKind::Forward | TokenKind::Model => {
                    break;
                }
                _ => {
                    return Err(
                        self.error("Expected 'fn', 'const', 'input', 'param', 'data', 'dataset', 'model', 'forward', or 'state_machine'")
                    );
                }
            }
        }

        // Parse forward or model block (only one allowed)
        let forward = if self.peek_kind() == TokenKind::Model {
            let model_block = self.parse_model_block(&mut inputs, &mut params, &consts)?;
            // Check for duplicate forward block
            if self.peek_kind() == TokenKind::Forward {
                return Err(Diagnostic::new(
                    "E_DUPLICATE_FORWARD_BLOCK",
                    "Cannot have both model and forward blocks".to_string(),
                ));
            }
            model_block
        } else if self.peek_kind() == TokenKind::Forward {
            let forward_block = self.parse_forward_block()?;
            // Check for duplicate model block
            if self.peek_kind() == TokenKind::Model {
                return Err(Diagnostic::new(
                    "E_DUPLICATE_MODEL_BLOCK",
                    "Cannot have both forward and model blocks".to_string(),
                ));
            }
            forward_block
        } else {
            return Err(self.error("Expected 'model' or 'forward' block"));
        };

        // Parse optional loss block
        let loss = if self.peek_kind() == TokenKind::Loss {
            Some(self.parse_loss_block()?)
        } else {
            None
        };

        // Parse optional train block
        let train = if self.peek_kind() == TokenKind::Train {
            Some(self.parse_train_block()?)
        } else {
            None
        };

        // Parse optional dataset/data block (can appear after model/forward if not already parsed)
        if dataset.is_none()
            && (self.peek_kind() == TokenKind::Dataset || self.peek_kind() == TokenKind::Data)
        {
            dataset = Some(self.parse_dataset_block()?);
        }

        // Parse optional eval block (must appear after train)
        let eval = if self.peek_kind() == TokenKind::Eval {
            Some(self.parse_eval_block()?)
        } else {
            None
        };

        // Parse optional dataset/data block (can appear after eval if not already parsed)
        if dataset.is_none()
            && (self.peek_kind() == TokenKind::Dataset || self.peek_kind() == TokenKind::Data)
        {
            dataset = Some(self.parse_dataset_block()?);
        }

        // Parse optional state machine block (can appear after eval)
        let state_machine = if self.peek_kind() == TokenKind::StateMachine {
            Some(self.parse_state_machine()?)
        } else {
            None
        };

        // Expect EOF
        if !self.is_eof() {
            return Err(self.error("Unexpected tokens after blocks"));
        }

        Ok(Program {
            app: None, // Legacy mode: no app wrapper
            functions,
            consts,
            inputs,
            params,
            forward,
            loss,
            train,
            dataset,
            eval,
            state_machine,
            state_block: None,
            event_handlers: Vec::new(),
        })
    }

    fn parse_app_block(&mut self) -> Result<Program, Diagnostic> {
        self.expect(TokenKind::App)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;

        // Parse app body (same structure as legacy Program)
        let mut functions = Vec::new();
        let mut consts = Vec::new();
        let mut inputs = Vec::new();
        let mut params = Vec::new();
        let mut dataset = None;
        let mut state_block = None;

        // Parse functions, const, input, param first
        while !self.is_eof() {
            match self.peek_kind() {
                TokenKind::Fn => {
                    let fn_decl = self.parse_fn_decl()?;
                    if functions.iter().any(|f: &FnDecl| f.name == fn_decl.name) {
                        return Err(Diagnostic::new(
                            "E_FUNCTION_DUPLICATE_NAME",
                            format!("Function '{}' is already defined", fn_decl.name),
                        )
                        .with_field("function_name".to_string(), fn_decl.name.clone()));
                    }
                    functions.push(fn_decl);
                }
                TokenKind::Const => {
                    consts.push(self.parse_const_decl()?);
                }
                TokenKind::Input => {
                    inputs.push(self.parse_input_decl()?);
                }
                TokenKind::Param => {
                    params.push(self.parse_param_decl()?);
                }
                TokenKind::Dataset | TokenKind::Data => {
                    if dataset.is_some() {
                        return Err(Diagnostic::new(
                            "E_DUPLICATE_DATASET_BLOCK",
                            "Only one dataset/data block is allowed".to_string(),
                        ));
                    }
                    dataset = Some(self.parse_dataset_block()?);
                }
                TokenKind::Forward | TokenKind::Model => {
                    break;
                }
                TokenKind::State => {
                    // state block can appear before model/forward
                    if state_block.is_some() {
                        return Err(Diagnostic::new(
                            "E_DUPLICATE_STATE_BLOCK",
                            "Only one state block allowed in app".to_string(),
                        ));
                    }
                    state_block = Some(self.parse_state_block()?);
                }
                TokenKind::On => {
                    // Event handlers can appear anywhere, we'll collect them later
                    // For now, skip them here and parse after model/forward
                    break;
                }
                TokenKind::RBrace => {
                    return Err(self.error("Expected model or forward block in app"));
                }
                _ => {
                    return Err(self.error("Expected declaration or block in app"));
                }
            }
        }

        // Parse forward or model block
        let forward = if self.peek_kind() == TokenKind::Model {
            let model_block = self.parse_model_block(&mut inputs, &mut params, &consts)?;
            if self.peek_kind() == TokenKind::Forward {
                return Err(Diagnostic::new(
                    "E_DUPLICATE_FORWARD_BLOCK",
                    "Cannot have both model and forward blocks".to_string(),
                ));
            }
            model_block
        } else if self.peek_kind() == TokenKind::Forward {
            let forward_block = self.parse_forward_block()?;
            if self.peek_kind() == TokenKind::Model {
                return Err(Diagnostic::new(
                    "E_DUPLICATE_MODEL_BLOCK",
                    "Cannot have both forward and model blocks".to_string(),
                ));
            }
            forward_block
        } else {
            return Err(self.error("Expected 'model' or 'forward' block in app"));
        };

        // Parse optional blocks
        let loss = if self.peek_kind() == TokenKind::Loss {
            Some(self.parse_loss_block()?)
        } else {
            None
        };

        let train = if self.peek_kind() == TokenKind::Train {
            Some(self.parse_train_block()?)
        } else {
            None
        };

        if dataset.is_none()
            && (self.peek_kind() == TokenKind::Dataset || self.peek_kind() == TokenKind::Data)
        {
            dataset = Some(self.parse_dataset_block()?);
        }

        let eval = if self.peek_kind() == TokenKind::Eval {
            Some(self.parse_eval_block()?)
        } else {
            None
        };

        if dataset.is_none()
            && (self.peek_kind() == TokenKind::Dataset || self.peek_kind() == TokenKind::Data)
        {
            dataset = Some(self.parse_dataset_block()?);
        }

        let state_machine = if self.peek_kind() == TokenKind::StateMachine {
            Some(self.parse_state_machine()?)
        } else {
            None
        };

        // Parse optional state block (new event-driven runtime model)
        let state_block = if self.peek_kind() == TokenKind::State {
            Some(self.parse_state_block()?)
        } else {
            None
        };

        // Parse optional event handlers (on <event>)
        let mut event_handlers = Vec::new();
        while self.peek_kind() == TokenKind::On {
            event_handlers.push(self.parse_on_event_handler()?);
        }

        self.expect(TokenKind::RBrace)?;

        if !self.is_eof() {
            return Err(self.error("Unexpected tokens after app block"));
        }

        let app_decl = AppDecl {
            name,
            functions,
            consts,
            inputs,
            params,
            forward,
            loss,
            train,
            dataset,
            eval,
            state_machine,
            state_block,
            event_handlers,
        };

        // Return Program with app wrapper
        Ok(Program {
            app: Some(app_decl.clone()),
            functions: app_decl.functions,
            consts: app_decl.consts,
            inputs: app_decl.inputs,
            params: app_decl.params,
            forward: app_decl.forward,
            loss: app_decl.loss,
            train: app_decl.train,
            dataset: app_decl.dataset,
            eval: app_decl.eval,
            state_machine: app_decl.state_machine,
            state_block: app_decl.state_block,
            event_handlers: app_decl.event_handlers,
        })
    }

    fn parse_const_decl(&mut self) -> Result<ConstDecl, Diagnostic> {
        self.expect(TokenKind::Const)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Eq)?;
        let value = match self.peek_kind() {
            TokenKind::Int(n) => {
                self.advance();
                n
            }
            _ => {
                return Err(self.error("Expected integer literal for const value"));
            }
        };
        Ok(ConstDecl { name, value })
    }

    fn parse_input_decl(&mut self) -> Result<InputDecl, Diagnostic> {
        self.expect(TokenKind::Input)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Colon)?;
        let shape = self.parse_shape()?;
        Ok(InputDecl { name, shape })
    }

    fn parse_param_decl(&mut self) -> Result<ParamDecl, Diagnostic> {
        self.expect(TokenKind::Param)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Colon)?;
        let shape = self.parse_shape()?;
        Ok(ParamDecl { name, shape })
    }

    fn parse_fn_decl(&mut self) -> Result<FnDecl, Diagnostic> {
        self.expect(TokenKind::Fn)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LParen)?;

        let mut params = Vec::new();
        if self.peek_kind() != TokenKind::RParen {
            loop {
                params.push(self.expect_ident()?);
                if self.peek_kind() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::LBrace)?;

        // Parse function body (let bindings)
        // Functions can only contain expressions valid in model/forward blocks
        let old_allow = self.allow_conditionals;
        self.allow_conditionals = true; // Allow conditionals in function bodies

        let mut body = Vec::new();
        while self.peek_kind() != TokenKind::RBrace {
            // Function body can contain let bindings (assignments)
            let name = self.expect_ident()?;
            self.expect(TokenKind::Eq)?;
            let expr = self.parse_expr()?;
            body.push(LetBinding { name, expr });
            // Semicolon is optional
            if self.peek_kind() == TokenKind::Semicolon {
                self.advance();
            }
        }
        self.allow_conditionals = old_allow;

        self.expect(TokenKind::RBrace)?;

        Ok(FnDecl { name, params, body })
    }

    fn parse_shape(&mut self) -> Result<Vec<DimSpec>, Diagnostic> {
        self.expect(TokenKind::LBracket)?;
        let mut dims = Vec::new();

        if self.peek_kind() != TokenKind::RBracket {
            loop {
                dims.push(self.parse_dim_spec()?);
                if self.peek_kind() == TokenKind::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(TokenKind::RBracket)?;
        Ok(dims)
    }

    fn parse_dim_spec(&mut self) -> Result<DimSpec, Diagnostic> {
        match self.peek_kind() {
            TokenKind::Int(n) => {
                self.advance();
                Ok(DimSpec::Literal(n as usize))
            }
            TokenKind::Ident(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(DimSpec::Named(name))
            }
            _ => Err(self.error("Expected dimension (identifier or integer)")),
        }
    }

    fn parse_forward_block(&mut self) -> Result<ForwardBlock, Diagnostic> {
        self.expect(TokenKind::Forward)?;
        self.expect(TokenKind::LBrace)?;

        let old_allow = self.allow_conditionals;
        self.allow_conditionals = true;

        let mut lets = Vec::new();
        let mut ret = None;

        while self.peek_kind() != TokenKind::RBrace {
            if self.peek_kind() == TokenKind::Let {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(TokenKind::Eq)?;
                let expr = self.parse_expr()?;
                lets.push(LetBinding { name, expr });
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
            } else if self.peek_kind() == TokenKind::Return {
                self.advance();
                ret = Some(self.expect_ident()?);
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
                break;
            } else {
                return Err(self.error("Expected 'let' or 'return' in forward block"));
            }
        }

        self.allow_conditionals = old_allow;
        self.expect(TokenKind::RBrace)?;

        let ret = ret.ok_or_else(|| self.error("Expected 'return' statement in forward block"))?;

        Ok(ForwardBlock { lets, ret })
    }

    fn parse_model_block(
        &mut self,
        inputs: &mut Vec<InputDecl>,
        params: &mut Vec<ParamDecl>,
        consts: &[ConstDecl],
    ) -> Result<ForwardBlock, Diagnostic> {
        self.expect(TokenKind::Model)?;
        self.expect(TokenKind::LBrace)?;

        let old_allow = self.allow_conditionals;
        self.allow_conditionals = true;

        let mut lets = Vec::new();
        let mut forward_output = None;

        // Build const lookup map
        let const_map: std::collections::HashMap<String, i64> =
            consts.iter().map(|c| (c.name.clone(), c.value)).collect();

        while self.peek_kind() != TokenKind::RBrace {
            // Check for input declaration (shorthand: name [shape])
            if matches!(self.peek_kind(), TokenKind::Ident(_)) {
                let lookahead = self.tokens.get(self.pos + 1);
                if let Some(Token {
                    kind: TokenKind::LBracket,
                    ..
                }) = lookahead
                {
                    // Input declaration shorthand
                    let name = self.expect_ident()?;
                    let shape = self.parse_shape()?;
                    inputs.push(InputDecl { name, shape });
                    continue;
                }
            }

            // Check for param declaration
            if self.peek_kind() == TokenKind::Param {
                self.advance(); // consume "param"
                let name = self.expect_ident()?;
                // Optional colon
                if self.peek_kind() == TokenKind::Colon {
                    self.advance();
                }
                let shape = self.parse_shape()?;
                params.push(ParamDecl { name, shape });
                continue;
            }

            // Assignment (without let)
            let name = self.expect_ident()?;
            self.expect(TokenKind::Eq)?;
            let expr = self.parse_expr()?;
            lets.push(LetBinding {
                name: name.clone(),
                expr,
            });

            // Track forward output: if name is "logits", use it; otherwise use last assignment
            if name == "logits" {
                forward_output = Some("logits".to_string());
            } else if forward_output.is_none() {
                forward_output = Some(name);
            }

            // Optional semicolon
            if self.peek_kind() == TokenKind::Semicolon {
                self.advance();
            }
        }

        self.allow_conditionals = old_allow;
        self.expect(TokenKind::RBrace)?;

        let ret = forward_output.ok_or_else(|| {
            Diagnostic::new(
                "E_MODEL_EMPTY",
                "Model block must have at least one assignment".to_string(),
            )
            .with_field("block".to_string(), "model".to_string())
        })?;

        Ok(ForwardBlock { lets, ret })
    }

    fn parse_loss_block(&mut self) -> Result<LossBlock, Diagnostic> {
        self.expect(TokenKind::Loss)?;
        self.expect(TokenKind::LBrace)?;

        let old_allow = self.allow_conditionals;
        self.allow_conditionals = false; // Disallow conditionals in loss blocks

        let mut lets = Vec::new();
        let mut ret = None;

        while self.peek_kind() != TokenKind::RBrace {
            if self.peek_kind() == TokenKind::Let {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(TokenKind::Eq)?;
                let expr = self.parse_expr()?;
                lets.push(LetBinding { name, expr });
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
            } else if self.peek_kind() == TokenKind::Return {
                self.advance();
                ret = Some(self.expect_ident()?);
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
                break;
            } else {
                return Err(self.error("Expected 'let' or 'return' in loss block"));
            }
        }

        self.allow_conditionals = old_allow;
        self.expect(TokenKind::RBrace)?;

        let ret = ret.ok_or_else(|| self.error("Expected 'return' statement in loss block"))?;

        Ok(LossBlock { lets, ret })
    }

    fn parse_train_block(&mut self) -> Result<TrainBlock, Diagnostic> {
        self.expect(TokenKind::Train)?;
        self.expect(TokenKind::LBrace)?;

        let old_allow = self.allow_conditionals;
        self.allow_conditionals = false; // Disallow conditionals in train blocks

        let mut steps = None;
        let mut lr = None;
        let mut batch = None;
        let mut loss_expr = None;

        while self.peek_kind() != TokenKind::RBrace {
            if self.peek_kind() == TokenKind::Loss {
                self.advance();
                self.expect(TokenKind::Eq)?;
                loss_expr = Some(self.parse_expr()?);
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
            } else if let TokenKind::Ident(ref id) = self.peek_kind() {
                if id == "steps" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    steps = match self.peek_kind() {
                        TokenKind::Int(n) => {
                            self.advance();
                            Some(n)
                        }
                        _ => return Err(self.error("Expected integer for steps")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else if id == "lr" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    lr = match self.peek_kind() {
                        TokenKind::Float(f) => {
                            self.advance();
                            Some(f)
                        }
                        TokenKind::Int(n) => {
                            self.advance();
                            Some(n as f32)
                        }
                        _ => return Err(self.error("Expected float for lr")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else if id == "batch" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    batch = match self.peek_kind() {
                        TokenKind::Int(n) => {
                            self.advance();
                            Some(n)
                        }
                        _ => return Err(self.error("Expected integer for batch")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else {
                    return Err(
                        self.error("Expected 'loss', 'steps', 'lr', or 'batch' in train block")
                    );
                }
            } else {
                return Err(self.error("Expected 'loss', 'steps', 'lr', or 'batch' in train block"));
            }
        }

        self.allow_conditionals = old_allow;
        self.expect(TokenKind::RBrace)?;

        Ok(TrainBlock {
            steps,
            lr,
            batch,
            loss_expr,
        })
    }

    fn parse_dataset_block(&mut self) -> Result<DatasetDecl, Diagnostic> {
        if self.peek_kind() == TokenKind::Dataset {
            self.advance();
        } else if self.peek_kind() == TokenKind::Data {
            self.advance();
        } else {
            return Err(self.error("Expected 'dataset' or 'data'"));
        }

        self.expect(TokenKind::LBrace)?;

        let mut format = None;
        let mut path = None;
        let mut tokens_input = None;
        let mut labels_input = None;
        let mut shuffle = None;
        let mut split = None;
        let mut maps = Vec::new();
        let mut filters = Vec::new();
        let mut batch = None;

        while self.peek_kind() != TokenKind::RBrace {
            if let TokenKind::Ident(ref id) = self.peek_kind() {
                if id == "format" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    format = match self.peek_kind() {
                        TokenKind::String(s) => {
                            self.advance();
                            Some(s)
                        }
                        _ => return Err(self.error("Expected string for format")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else if id == "path" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    path = match self.peek_kind() {
                        TokenKind::String(s) => {
                            self.advance();
                            Some(s)
                        }
                        _ => return Err(self.error("Expected string for path")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else if id == "tokens" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    tokens_input = match self.peek_kind() {
                        TokenKind::String(s) => {
                            self.advance();
                            Some(s)
                        }
                        _ => return Err(self.error("Expected string for tokens")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else if id == "labels" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    labels_input = match self.peek_kind() {
                        TokenKind::String(s) => {
                            self.advance();
                            Some(s)
                        }
                        _ => return Err(self.error("Expected string for labels")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else if id == "shuffle" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    shuffle = match self.peek_kind() {
                        TokenKind::True => {
                            self.advance();
                            Some(true)
                        }
                        TokenKind::False => {
                            self.advance();
                            Some(false)
                        }
                        _ => return Err(self.error("Expected boolean for shuffle")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else if id == "split" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    split = match self.peek_kind() {
                        TokenKind::Float(f) => {
                            self.advance();
                            Some(f)
                        }
                        TokenKind::Int(n) => {
                            self.advance();
                            Some(n as f32)
                        }
                        _ => return Err(self.error("Expected float for split")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else {
                    return Err(self.error("Unexpected identifier in dataset block"));
                }
            } else if self.peek_kind() == TokenKind::Map {
                self.advance();
                // Parse function call: map normalize(tokens)
                let fn_name = self.expect_ident()?;
                self.expect(TokenKind::LParen)?;
                let mut args = Vec::new();
                if self.peek_kind() != TokenKind::RParen {
                    loop {
                        args.push(self.parse_expr()?);
                        if self.peek_kind() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::RParen)?;
                maps.push(CallExpr {
                    fn_name,
                    args,
                    kwargs: Vec::new(),
                });
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
            } else if self.peek_kind() == TokenKind::Filter {
                self.advance();
                // Parse function call: filter valid_label(labels)
                let fn_name = self.expect_ident()?;
                self.expect(TokenKind::LParen)?;
                let mut args = Vec::new();
                if self.peek_kind() != TokenKind::RParen {
                    loop {
                        args.push(self.parse_expr()?);
                        if self.peek_kind() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(TokenKind::RParen)?;
                filters.push(CallExpr {
                    fn_name,
                    args,
                    kwargs: Vec::new(),
                });
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
            } else if self.peek_kind() == TokenKind::Batch {
                self.advance();
                // Parse integer: batch 32
                batch = match self.peek_kind() {
                    TokenKind::Int(n) => {
                        self.advance();
                        Some(n as usize)
                    }
                    _ => return Err(self.error("Expected integer for batch")),
                };
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
            } else {
                return Err(self.error("Unexpected token in dataset block"));
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(DatasetDecl {
            format: format.ok_or_else(|| self.error("Missing 'format' field in dataset block"))?,
            path: path.ok_or_else(|| self.error("Missing 'path' field in dataset block"))?,
            tokens_input: tokens_input
                .ok_or_else(|| self.error("Missing 'tokens' field in dataset block"))?,
            labels_input: labels_input
                .ok_or_else(|| self.error("Missing 'labels' field in dataset block"))?,
            shuffle,
            split,
            maps,
            filters,
            batch,
        })
    }

    fn parse_eval_block(&mut self) -> Result<EvalBlock, Diagnostic> {
        self.expect(TokenKind::Eval)?;
        self.expect(TokenKind::LBrace)?;

        let mut metrics = Vec::new();
        let mut every = None;
        let mut split = None;

        while self.peek_kind() != TokenKind::RBrace {
            if let TokenKind::Ident(ref id) = self.peek_kind() {
                if id == "metrics" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    self.expect(TokenKind::LBracket)?;
                    while self.peek_kind() != TokenKind::RBracket {
                        if let TokenKind::Ident(ref name) = self.peek_kind() {
                            metrics.push(name.clone());
                            self.advance();
                        } else if let TokenKind::String(ref s) = self.peek_kind() {
                            metrics.push(s.clone());
                            self.advance();
                        } else {
                            return Err(self.error("Expected identifier or string in metrics list"));
                        }
                        if self.peek_kind() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(TokenKind::RBracket)?;
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else if id == "every" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    every = match self.peek_kind() {
                        TokenKind::Int(n) => {
                            self.advance();
                            Some(n)
                        }
                        _ => return Err(self.error("Expected integer for every")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else if id == "split" {
                    self.advance();
                    self.expect(TokenKind::Eq)?;
                    split = match self.peek_kind() {
                        TokenKind::String(s) => {
                            self.advance();
                            Some(s)
                        }
                        _ => return Err(self.error("Expected string for split")),
                    };
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else {
                    return Err(self.error("Unexpected identifier in eval block"));
                }
            } else {
                return Err(self.error("Unexpected token in eval block"));
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(EvalBlock {
            metrics,
            every,
            split,
        })
    }

    fn parse_state_machine(&mut self) -> Result<StateMachineDecl, Diagnostic> {
        self.expect(TokenKind::StateMachine)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut states = Vec::new();
        while self.peek_kind() != TokenKind::RBrace {
            states.push(self.parse_state_decl()?);
        }

        self.expect(TokenKind::RBrace)?;

        // Check for duplicate state names
        let mut seen = std::collections::HashSet::new();
        for state in &states {
            if !seen.insert(state.name.clone()) {
                return Err(Diagnostic::new(
                    "E_STATE_DUPLICATE",
                    format!("Duplicate state name: '{}'", state.name),
                )
                .with_field("state_name".to_string(), state.name.clone()));
            }
        }

        Ok(StateMachineDecl { name, states })
    }

    fn parse_state_decl(&mut self) -> Result<StateDecl, Diagnostic> {
        self.expect(TokenKind::State)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut body = Vec::new();
        let mut transitions = Vec::new();
        let mut has_end = false;

        while self.peek_kind() != TokenKind::RBrace {
            if self.peek_kind() == TokenKind::On {
                // on event -> State
                self.advance();
                let event = if matches!(self.peek_kind(), TokenKind::Ident(_)) {
                    self.expect_ident()?
                } else if matches!(self.peek_kind(), TokenKind::Input) {
                    self.advance();
                    "input".to_string()
                } else {
                    return Err(self.error("Expected event name (identifier or keyword)"));
                };
                self.expect(TokenKind::Arrow)?;
                let target = self.expect_ident()?;
                transitions.push(TransitionDecl {
                    kind: TransitionKind::On(event),
                    target,
                    condition: None,
                });
            } else if self.peek_kind() == TokenKind::If {
                // if condition -> State
                self.advance();
                let condition = self.parse_expr()?;
                self.expect(TokenKind::Arrow)?;
                let target = self.expect_ident()?;
                transitions.push(TransitionDecl {
                    kind: TransitionKind::If,
                    target,
                    condition: Some(condition),
                });
            } else if self.peek_kind() == TokenKind::Else {
                // else -> State
                self.advance();
                self.expect(TokenKind::Arrow)?;
                let target = self.expect_ident()?;
                transitions.push(TransitionDecl {
                    kind: TransitionKind::Else,
                    target,
                    condition: None,
                });
            } else if self.peek_kind() == TokenKind::Emit {
                // emit expression
                self.advance();
                let value = self.parse_expr()?;
                body.push(StateStatement::Emit(EmitDecl { value }));
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
            } else if self.peek_kind() == TokenKind::End {
                // end (terminates state machine)
                self.advance();
                has_end = true;
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
            } else if self.peek_kind() == TokenKind::Let {
                // let binding
                self.advance();
                let name = self.expect_ident()?;
                self.expect(TokenKind::Eq)?;
                let expr = self.parse_expr()?;
                body.push(StateStatement::Let(LetBinding { name, expr }));
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
            } else if matches!(self.peek_kind(), TokenKind::Ident(_)) {
                // Assignment without let (for compatibility)
                let name = self.expect_ident()?;
                if self.peek_kind() == TokenKind::Eq {
                    self.advance();
                    let expr = self.parse_expr()?;
                    body.push(StateStatement::Let(LetBinding { name, expr }));
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                } else {
                    return Err(self.error("Expected '=' after identifier"));
                }
            } else {
                return Err(self.error("Expected state statement, transition, or 'end'"));
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(StateDecl {
            name,
            body,
            transitions,
            has_end,
        })
    }

    fn parse_expr(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<Expr, Diagnostic> {
        let mut left = self.parse_logical_and()?;

        while self.peek_kind() == TokenKind::Or {
            self.advance();
            let right = self.parse_logical_and()?;
            left = Expr::Logical {
                op: LogicalOp::Or,
                left: Box::new(left),
                right: Some(Box::new(right)),
            };
        }

        Ok(left)
    }

    fn parse_logical_and(&mut self) -> Result<Expr, Diagnostic> {
        let mut left = self.parse_logical_not()?;

        while self.peek_kind() == TokenKind::And {
            self.advance();
            let right = self.parse_logical_not()?;
            left = Expr::Logical {
                op: LogicalOp::And,
                left: Box::new(left),
                right: Some(Box::new(right)),
            };
        }

        Ok(left)
    }

    fn parse_logical_not(&mut self) -> Result<Expr, Diagnostic> {
        if self.peek_kind() == TokenKind::Not {
            self.advance();
            let expr = self.parse_logical_not()?;
            Ok(Expr::Logical {
                op: LogicalOp::Not,
                left: Box::new(expr),
                right: None,
            })
        } else {
            self.parse_compare()
        }
    }

    fn parse_compare(&mut self) -> Result<Expr, Diagnostic> {
        let mut left = self.parse_additive()?;

        while matches!(
            self.peek_kind(),
            TokenKind::EqEq
                | TokenKind::Ne
                | TokenKind::Lt
                | TokenKind::Le
                | TokenKind::Gt
                | TokenKind::Ge
        ) {
            let op = match self.peek_kind() {
                TokenKind::EqEq => {
                    self.advance();
                    CompareOp::Eq
                }
                TokenKind::Ne => {
                    self.advance();
                    CompareOp::Ne
                }
                TokenKind::Lt => {
                    self.advance();
                    CompareOp::Lt
                }
                TokenKind::Le => {
                    self.advance();
                    CompareOp::Le
                }
                TokenKind::Gt => {
                    self.advance();
                    CompareOp::Gt
                }
                TokenKind::Ge => {
                    self.advance();
                    CompareOp::Ge
                }
                _ => unreachable!(),
            };

            let right = self.parse_additive()?;
            left = Expr::Compare {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_multiplicative()
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_unary()
    }

    fn parse_unary(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_if_expr()
    }

    fn parse_if_expr(&mut self) -> Result<Expr, Diagnostic> {
        if !self.allow_conditionals {
            return self.parse_loop_expr();
        }

        if self.peek_kind() == TokenKind::If {
            self.advance();
            let cond = self.parse_expr()?;
            self.expect(TokenKind::Then)?;
            let then_branch = self.parse_expr()?;
            self.expect(TokenKind::Else)?;
            let else_branch = self.parse_expr()?;
            Ok(Expr::If {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            })
        } else {
            self.parse_loop_expr()
        }
    }

    fn parse_loop_expr(&mut self) -> Result<Expr, Diagnostic> {
        // Check for for loop
        if self.peek_kind() == TokenKind::For {
            self.advance(); // consume "for"
            let var = self.expect_ident()?;
            self.expect(TokenKind::In)?;
            self.expect(TokenKind::Range)?;
            self.expect(TokenKind::LParen)?;
            let start = self.parse_expr()?;
            self.expect(TokenKind::Comma)?;
            let end = self.parse_expr()?;
            self.expect(TokenKind::RParen)?;
            self.expect(TokenKind::Do)?;
            let body = self.parse_expr()?;
            self.expect(TokenKind::End)?;
            return Ok(Expr::ForLoop {
                var,
                start: Box::new(start),
                end: Box::new(end),
                body: Box::new(body),
            });
        }

        // Check for repeat loop
        if self.peek_kind() == TokenKind::Repeat {
            self.advance(); // consume "repeat"
            let count = self.parse_expr()?;
            self.expect(TokenKind::Times)?;
            // Check for "init" keyword
            if let TokenKind::Ident(ref init_kw) = self.peek_kind() {
                if init_kw == "init" {
                    self.advance(); // consume "init"
                    let init = self.parse_expr()?;
                    self.expect(TokenKind::Do)?;
                    let var = self.expect_ident()?; // accumulator variable name
                                                    // Note: body uses this var, but for now we'll parse the body
                    let body = self.parse_expr()?;
                    self.expect(TokenKind::End)?;
                    return Ok(Expr::RepeatLoop {
                        count: Box::new(count),
                        var,
                        init: Box::new(init),
                        body: Box::new(body),
                    });
                } else {
                    return Err(self.error("repeat loop requires 'init' keyword"));
                }
            } else {
                return Err(self.error("repeat loop requires 'init' keyword"));
            }
        }

        // Check for reduce loop
        if self.peek_kind() == TokenKind::Reduce {
            self.advance(); // consume "reduce"
            let op_str = self.expect_ident()?; // "add", "max", or "min"
            self.expect(TokenKind::Over)?;
            self.expect(TokenKind::Range)?;
            self.expect(TokenKind::LParen)?;
            let start = self.parse_expr()?;
            self.expect(TokenKind::Comma)?;
            let end = self.parse_expr()?;
            self.expect(TokenKind::RParen)?;
            self.expect(TokenKind::Do)?;
            let body = self.parse_expr()?;
            self.expect(TokenKind::End)?;
            return Ok(Expr::ReduceLoop {
                op: op_str,
                var: "i".to_string(), // Default loop variable name
                start: Box::new(start),
                end: Box::new(end),
                body: Box::new(body),
            });
        }

        self.parse_call_expr()
    }

    fn parse_call_expr(&mut self) -> Result<Expr, Diagnostic> {
        let mut expr = self.parse_primary()?;

        loop {
            if let TokenKind::LParen = self.peek_kind() {
                // Function call
                self.advance(); // consume LParen
                let mut args = Vec::new();
                let mut kwargs = Vec::new();

                if self.peek_kind() != TokenKind::RParen {
                    loop {
                        // Check if this is a keyword argument
                        if matches!(self.peek_kind(), TokenKind::Ident(_)) {
                            let lookahead = self.tokens.get(self.pos + 1);
                            if let Some(Token {
                                kind: TokenKind::Eq,
                                ..
                            }) = lookahead
                            {
                                // Keyword argument
                                let kw_name = self.expect_ident()?;
                                self.expect(TokenKind::Eq)?;
                                let kw_value = self.parse_expr()?;
                                kwargs.push((kw_name, kw_value));
                            } else {
                                // Positional argument
                                args.push(self.parse_expr()?);
                            }
                        } else {
                            // Positional argument
                            args.push(self.parse_expr()?);
                        }

                        if self.peek_kind() == TokenKind::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }

                self.expect(TokenKind::RParen)?;

                // Extract function name from expr
                let fn_name = match expr {
                    Expr::Ident(name) => name,
                    _ => return Err(self.error("Expected function name")),
                };

                expr = Expr::Call(CallExpr {
                    fn_name,
                    args,
                    kwargs,
                });
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, Diagnostic> {
        match self.peek_kind() {
            TokenKind::Int(n) => {
                self.advance();
                Ok(Expr::Int(n))
            }
            TokenKind::Float(f) => {
                self.advance();
                Ok(Expr::Float(f))
            }
            TokenKind::True => {
                self.advance();
                Ok(Expr::Int(1)) // Represent true as 1
            }
            TokenKind::False => {
                self.advance();
                Ok(Expr::Int(0)) // Represent false as 0
            }
            TokenKind::String(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Expr::String(s))
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::Ident(_) => {
                let name = self.expect_ident()?;
                Ok(Expr::Ident(name))
            }
            _ => Err(self.error("Expected expression")),
        }
    }

    fn peek_kind(&self) -> TokenKind {
        if self.pos >= self.tokens.len() {
            return TokenKind::Eof;
        }
        self.tokens[self.pos].kind.clone()
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len() || matches!(self.peek_kind(), TokenKind::Eof)
    }

    fn expect(&mut self, kind: TokenKind) -> Result<(), Diagnostic> {
        if self.peek_kind() == kind {
            self.advance();
            Ok(())
        } else {
            Err(self.error(&format!("Expected {:?}", kind)))
        }
    }

    fn expect_ident(&mut self) -> Result<String, Diagnostic> {
        match self.peek_kind() {
            TokenKind::Ident(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(self.error("Expected identifier")),
        }
    }

    fn error(&self, msg: &str) -> Diagnostic {
        let line = if self.pos < self.tokens.len() {
            self.tokens[self.pos].span.line
        } else if !self.tokens.is_empty() {
            self.tokens.last().unwrap().span.line + 1
        } else {
            1
        };
        let col = if self.pos < self.tokens.len() {
            self.tokens[self.pos].span.col
        } else if !self.tokens.is_empty() {
            self.tokens.last().unwrap().span.col + 1
        } else {
            1
        };

        Diagnostic::new("E_SYNTAX_ERROR", "Syntax error".to_string())
            .with_field("line".to_string(), line.to_string())
            .with_field("col".to_string(), col.to_string())
            .with_field("message".to_string(), msg.to_string())
            .with_hint(format!("At line {} column {}", line, col))
    }

    // New event-driven runtime parsing functions

    /// Parse a state block: `state { counter: int = 0 }`
    fn parse_state_block(&mut self) -> Result<StateBlock, Diagnostic> {
        self.expect(TokenKind::State)?;
        self.expect(TokenKind::LBrace)?;

        let mut vars = Vec::new();

        while self.peek_kind() != TokenKind::RBrace {
            // Parse state variable: name: type [= initial_value]
            let name = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;

            // Parse type (int, float, tensor)
            let var_type = match self.peek_kind() {
                TokenKind::Ident(type_name) => {
                    self.advance();
                    match type_name.as_str() {
                        "int" => StateVarType::Int,
                        "float" => StateVarType::Float,
                        "tensor" => StateVarType::Tensor,
                        _ => {
                            return Err(Diagnostic::new(
                                "E_STATE_TYPE_INVALID",
                                format!("Invalid state variable type: {}. Expected 'int', 'float', or 'tensor'", type_name),
                            )
                            .with_field("type".to_string(), type_name));
                        }
                    }
                }
                _ => {
                    return Err(
                        self.error("Expected type name (int, float, or tensor) for state variable")
                    );
                }
            };

            // Parse optional initial value: = expr
            let init_value = if self.peek_kind() == TokenKind::Eq {
                self.advance();
                Some(self.parse_expr()?)
            } else {
                None
            };

            vars.push(StateVarDecl {
                name,
                var_type,
                init_value,
            });

            // Optional comma or semicolon separator (for now, just check for RBrace or next variable)
            if self.peek_kind() == TokenKind::Comma || self.peek_kind() == TokenKind::Semicolon {
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(StateBlock { vars })
    }

    /// Parse an `on <event>` handler: `on start { ... }`
    fn parse_on_event_handler(&mut self) -> Result<OnEventHandler, Diagnostic> {
        self.expect(TokenKind::On)?;
        let event_name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut body = Vec::new();

        while self.peek_kind() != TokenKind::RBrace {
            body.push(self.parse_event_handler_statement()?);
        }

        self.expect(TokenKind::RBrace)?;

        Ok(OnEventHandler { event_name, body })
    }

    /// Parse a statement inside an event handler
    fn parse_event_handler_statement(&mut self) -> Result<EventHandlerStatement, Diagnostic> {
        match self.peek_kind() {
            TokenKind::Let => {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(TokenKind::Eq)?;
                let expr = self.parse_expr()?;
                // Optional semicolon
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
                Ok(EventHandlerStatement::Let(LetBinding { name, expr }))
            }
            TokenKind::Emit => {
                self.advance();
                let event_name = self.expect_ident()?;
                // Optional semicolon
                if self.peek_kind() == TokenKind::Semicolon {
                    self.advance();
                }
                Ok(EventHandlerStatement::EmitEvent(event_name))
            }
            TokenKind::Ident(_) => {
                // State assignment: name = expr
                let name = self.expect_ident()?;
                if self.peek_kind() == TokenKind::Eq {
                    self.advance();
                    let expr = self.parse_expr()?;
                    // Optional semicolon
                    if self.peek_kind() == TokenKind::Semicolon {
                        self.advance();
                    }
                    Ok(EventHandlerStatement::Assign(name, expr))
                } else {
                    return Err(Diagnostic::new(
                        "E_EVENT_HANDLER_STATEMENT_INVALID",
                        format!("Invalid statement in event handler. Expected assignment, 'let', or 'emit', but found: {}", name),
                    )
                    .with_field("statement".to_string(), name));
                }
            }
            _ => {
                return Err(
                    self.error("Expected statement in event handler (let, emit, or assignment)")
                );
            }
        }
    }
}

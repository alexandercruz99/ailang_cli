pub type NodeId = usize;

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<NodeId>,
}

#[derive(Clone, Debug)]
pub enum Op {
    Add,
    Sub,
    Mul,
    MatMul2D,
    ReLU,
    Sum,
    Mean,
    Input(usize),                              // index into input tensor list
    Embedding(usize),                          // index into token_ids list
    Softmax(usize),                            // axis
    CrossEntropy(usize),                       // index into token_ids list for target_ids
    LayerNorm(usize, usize, f32),              // gamma input idx, beta input idx, eps
    MeanPoolTime,                              // mean over time axis: [B, T, D] -> [B, D]
    Transpose3D,                               // transpose last two axes: [B, T, D] -> [B, D, T]
    Reshape(crate::reshape_spec::ReshapeSpec), // reshape with symbolic specification
    BatchMatMul,                               // batched matmul: [B, M, K] @ [B, K, N] -> [B, M, N]
    Now,                                       // get current time (requires Capability::Clock)
    ReadFileText(usize), // read text file, hash to f32 vector (requires Capability::FileRead); usize is index into paths array
    Concat {
        axis: usize,
    }, // concat along axis (2D only, axis=1 only)
    SliceRows {
        start: usize,
        len: usize,
    }, // slice rows from 2D tensor
    GatherRows,          // gather rows using TokenIds indices
    Dropout {
        p: f32,
    }, // dropout with probability p (training-only)
    If {
        // Conditional: if condition then true_branch else false_branch
        // Inputs: [condition_node, true_branch_node, false_branch_node]
        // condition_node must be scalar-compatible
        // true_branch_node and false_branch_node must have same shape and dtype
    },
    Compare {
        op: CompareOp,
    },
    Logical {
        op: LogicalOp,
    },
    ConstScalar {
        value: f32,
    },
    Stack {
        axis: usize,
    },
    Max2,
    Min2,
}

#[derive(Clone, Debug)]
pub enum CompareOp {
    Eq, // ==
    Ne, // !=
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=
}

#[derive(Clone, Debug)]
pub enum LogicalOp {
    And,
    Or,
    Not,
}

impl Op {
    /// Returns the capability required by this op, if any.
    pub fn required_capability(&self) -> Option<crate::capability::Capability> {
        match self {
            Op::Now => Some(crate::capability::Capability::Clock),
            Op::ReadFileText(_) => Some(crate::capability::Capability::FileRead),
            Op::If { .. } => {
                // If op itself doesn't require capabilities, but branches might
                // Capability checking happens during selective execution
                None
            }
            Op::ConstScalar { .. } | Op::Stack { .. } | Op::Max2 | Op::Min2 => None,
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub input_count: usize,
    pub token_ids_count: usize,
    pub input_specs: Vec<crate::input_spec::InputSpec>, // Input signatures
    pub input_node_ids: Vec<NodeId>,                    // Node IDs corresponding to graph inputs
}

impl Graph {
    pub fn new(input_count: usize) -> Self {
        Self {
            nodes: Vec::new(),
            input_count,
            token_ids_count: 0,
            input_specs: Vec::new(),
            input_node_ids: Vec::new(),
        }
    }

    pub fn new_with_token_ids(input_count: usize, token_ids_count: usize) -> Self {
        Self {
            nodes: Vec::new(),
            input_count,
            token_ids_count,
            input_specs: Vec::new(),
            input_node_ids: Vec::new(),
        }
    }

    pub fn with_input_specs(mut self, specs: Vec<crate::input_spec::InputSpec>) -> Self {
        self.input_specs = specs;
        self
    }

    pub fn add_node(&mut self, op: Op, inputs: Vec<NodeId>) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node { op, inputs });
        id
    }

    pub fn input_node(&mut self, input_idx: usize) -> NodeId {
        assert!(input_idx < self.input_count);
        self.add_node(Op::Input(input_idx), vec![])
    }
}

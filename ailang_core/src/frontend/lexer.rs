#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub col: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // Keywords
    Const,
    Input,
    Param,
    Forward,
    Model,
    Fn,
    Data,
    Let,
    Return,
    Loss,
    Train,
    Dataset,
    Eval,
    True,
    False,
    Repeat,
    In,
    If,
    Then,
    Else,
    And,
    Or,
    Not,
    For,
    Do,
    End,
    App,
    Times,
    Reduce,
    Over,
    Range,
    Map,
    Filter,
    Batch,
    StateMachine,
    State,
    On,
    Emit,

    // Identifiers
    Ident(String),

    // Literals
    Int(i64),
    Float(f32),
    String(String),

    // Punctuation
    Colon,     // :
    LBracket,  // [
    RBracket,  // ]
    LBrace,    // {
    RBrace,    // }
    LParen,    // (
    RParen,    // )
    Comma,     // ,
    Eq,        // =
    EqEq,      // ==
    Ne,        // !=
    Lt,        // <
    Le,        // <=
    Gt,        // >
    Ge,        // >=
    Semicolon, // ;
    Arrow,     // ->

    // Special
    Eof,
}

pub struct Lexer {
    input: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, crate::error::RuntimeError> {
        let mut tokens = Vec::new();

        while !self.is_eof() {
            self.skip_whitespace();
            if self.is_eof() {
                break;
            }

            let start = self.pos;
            let line = self.line;
            let col = self.col;

            let kind = self.next_token()?;
            let end = self.pos;

            tokens.push(Token {
                kind,
                span: Span {
                    start,
                    end,
                    line,
                    col,
                },
            });
        }

        // Add EOF token
        tokens.push(Token {
            kind: TokenKind::Eof,
            span: Span {
                start: self.pos,
                end: self.pos,
                line: self.line,
                col: self.col,
            },
        });

        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<TokenKind, crate::error::RuntimeError> {
        let ch = self.peek();

        match ch {
            ':' => {
                self.advance();
                Ok(TokenKind::Colon)
            }
            '[' => {
                self.advance();
                Ok(TokenKind::LBracket)
            }
            ']' => {
                self.advance();
                Ok(TokenKind::RBracket)
            }
            '{' => {
                self.advance();
                Ok(TokenKind::LBrace)
            }
            '}' => {
                self.advance();
                Ok(TokenKind::RBrace)
            }
            '(' => {
                self.advance();
                Ok(TokenKind::LParen)
            }
            ')' => {
                self.advance();
                Ok(TokenKind::RParen)
            }
            ',' => {
                self.advance();
                Ok(TokenKind::Comma)
            }
            '=' => {
                self.advance();
                if self.peek() == '=' {
                    self.advance();
                    Ok(TokenKind::EqEq)
                } else {
                    Ok(TokenKind::Eq)
                }
            }
            '!' => {
                self.advance();
                if self.peek() == '=' {
                    self.advance();
                    Ok(TokenKind::Ne)
                } else {
                    Err(crate::error::RuntimeError::Other(
                        "Unexpected '!' character".to_string(),
                    ))
                }
            }
            '<' => {
                self.advance();
                if self.peek() == '=' {
                    self.advance();
                    Ok(TokenKind::Le)
                } else {
                    Ok(TokenKind::Lt)
                }
            }
            '>' => {
                self.advance();
                if self.peek() == '=' {
                    self.advance();
                    Ok(TokenKind::Ge)
                } else {
                    Ok(TokenKind::Gt)
                }
            }
            ';' => {
                self.advance();
                Ok(TokenKind::Semicolon)
            }
            '-' => {
                self.advance();
                if self.peek() == '>' {
                    self.advance();
                    Ok(TokenKind::Arrow)
                } else {
                    Err(crate::error::RuntimeError::Other(
                        "Unexpected '-' character (expected '->')".to_string(),
                    ))
                }
            }
            '#' => {
                // Comment: skip to end of line
                while !self.is_eof() && self.peek() != '\n' {
                    self.advance();
                }
                // Don't call next_token recursively - just skip whitespace and continue
                self.skip_whitespace();
                if self.is_eof() {
                    Ok(TokenKind::Eof)
                } else {
                    self.next_token()
                }
            }
            '"' => self.read_string(),
            c if c.is_ascii_alphabetic() || c == '_' => self.read_identifier_or_keyword(),
            c if c.is_ascii_digit() => self.read_number(),
            _ => Err(crate::error::RuntimeError::Other(format!(
                "Unexpected character: '{}' at line {} col {}",
                ch, self.line, self.col
            ))),
        }
    }

    fn read_identifier_or_keyword(&mut self) -> Result<TokenKind, crate::error::RuntimeError> {
        let start = self.pos;
        while !self.is_eof() && (self.peek().is_ascii_alphanumeric() || self.peek() == '_') {
            self.advance();
        }
        let text: String = self.input[start..self.pos].iter().collect();

        Ok(match text.as_str() {
            "const" => TokenKind::Const,
            "input" => TokenKind::Input,
            "param" => TokenKind::Param,
            "forward" => TokenKind::Forward,
            "model" => TokenKind::Model,
            "fn" => TokenKind::Fn,
            "data" => TokenKind::Data,
            "let" => TokenKind::Let,
            "return" => TokenKind::Return,
            "loss" => TokenKind::Loss,
            "train" => TokenKind::Train,
            "dataset" => TokenKind::Dataset,
            "eval" => TokenKind::Eval,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "repeat" => TokenKind::Repeat,
            "in" => TokenKind::In,
            "if" => TokenKind::If,
            "then" => TokenKind::Then,
            "else" => TokenKind::Else,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "not" => TokenKind::Not,
            "for" => TokenKind::For,
            "do" => TokenKind::Do,
            "end" => TokenKind::End,
            "times" => TokenKind::Times,
            "reduce" => TokenKind::Reduce,
            "over" => TokenKind::Over,
            "range" => TokenKind::Range,
            "map" => TokenKind::Map,
            "filter" => TokenKind::Filter,
            "batch" => TokenKind::Batch,
            "state_machine" => TokenKind::StateMachine,
            "state" => TokenKind::State,
            "on" => TokenKind::On,
            "emit" => TokenKind::Emit,
            "end" => TokenKind::End,
            "app" => TokenKind::App,
            _ => TokenKind::Ident(text),
        })
    }

    fn read_string(&mut self) -> Result<TokenKind, crate::error::RuntimeError> {
        self.advance(); // Skip opening quote
        let mut value = String::new();

        while !self.is_eof() {
            let ch = self.peek();
            if ch == '"' {
                self.advance(); // Skip closing quote
                return Ok(TokenKind::String(value));
            } else if ch == '\\' {
                self.advance();
                if self.is_eof() {
                    break;
                }
                let escaped = self.peek();
                match escaped {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    _ => value.push(escaped),
                }
                self.advance();
            } else {
                value.push(ch);
                self.advance();
            }
        }

        Err(crate::error::RuntimeError::Other(format!(
            "Unterminated string literal at line {} col {}",
            self.line, self.col
        )))
    }

    fn read_number(&mut self) -> Result<TokenKind, crate::error::RuntimeError> {
        let start = self.pos;
        while !self.is_eof() && self.peek().is_ascii_digit() {
            self.advance();
        }

        // Check if there's a decimal point followed by digits
        if !self.is_eof() && self.peek() == '.' {
            self.advance(); // consume '.'
            if !self.is_eof() && self.peek().is_ascii_digit() {
                // It's a float
                while !self.is_eof() && self.peek().is_ascii_digit() {
                    self.advance();
                }
                let text: String = self.input[start..self.pos].iter().collect();
                let value = text.parse::<f32>().map_err(|_| {
                    crate::error::RuntimeError::Other(format!(
                        "Invalid float: {} at line {} col {}",
                        text, self.line, self.col
                    ))
                })?;
                return Ok(TokenKind::Float(value));
            } else {
                // Just a '.' without digits, treat as integer and put back the '.'
                self.pos -= 1;
            }
        }

        // It's an integer
        let text: String = self.input[start..self.pos].iter().collect();
        let value = text.parse::<i64>().map_err(|_| {
            crate::error::RuntimeError::Other(format!(
                "Invalid integer: {} at line {} col {}",
                text, self.line, self.col
            ))
        })?;
        Ok(TokenKind::Int(value))
    }

    fn skip_whitespace(&mut self) {
        while !self.is_eof() && self.peek().is_whitespace() {
            if self.peek() == '\n' {
                self.line += 1;
                self.col = 1;
            }
            self.advance();
        }
    }

    fn peek(&self) -> char {
        if self.pos >= self.input.len() {
            '\0'
        } else {
            self.input[self.pos]
        }
    }

    fn advance(&mut self) {
        if !self.is_eof() {
            self.pos += 1;
            self.col += 1;
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }
}

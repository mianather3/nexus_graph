//! Lexical retrieval: tokenization and BM25 over an inverted index.

pub mod bm25;
pub mod tokenizer;

pub use bm25::{Bm25Index, Bm25Params, LexicalHit};
pub use tokenizer::{is_stop_word, Tokenizer};

//! In-memory inverted index with Okapi BM25 ranking.
//!
//! ```text
//! score(D, Q) = sum over distinct t in Q of
//!     IDF(t) * f(t, D) * (k1 + 1) / (f(t, D) + k1 * (1 - b + b * |D| / avgdl))
//! IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
//! ```
//!
//! `f(t, D)` is the frequency of `t` in `D`, `|D|` the token count of `D`,
//! `avgdl` the mean token count, `N` the number of documents and `n(t)` the
//! number of documents containing `t`. A term repeated in the query counts
//! once.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::NexusError;
use crate::lexical::tokenizer::Tokenizer;
use crate::models::EntityId;
use crate::topk::TopK;

/// BM25 free parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bm25Params {
    /// Term-frequency saturation. Must be finite and non-negative.
    pub k1: f64,
    /// Document-length normalization strength, in `[0.0, 1.0]`.
    pub b: f64,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// One ranked document.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LexicalHit {
    pub id: EntityId,
    pub score: f64,
}

/// Entry of a posting list: a document (by internal slot) and how often the
/// term occurs in it.
#[derive(Debug, Clone, Copy)]
struct Posting {
    slot: u32,
    term_frequency: u32,
}

/// Inverted index over tokenized documents.
#[derive(Debug, Clone, Default)]
pub struct Bm25Index {
    tokenizer: Tokenizer,
    params: Bm25Params,
    /// Term -> postings, ordered by ascending slot (insertion order).
    postings: HashMap<String, Vec<Posting>>,
    /// Slot -> external id.
    doc_ids: Vec<EntityId>,
    /// Slot -> document length in tokens.
    doc_lens: Vec<u32>,
    slots: HashMap<EntityId, u32>,
    total_len: u64,
}

impl Bm25Index {
    /// Creates an empty index, validating the parameters.
    pub fn new(tokenizer: Tokenizer, params: Bm25Params) -> Result<Self, NexusError> {
        if !params.k1.is_finite() || params.k1 < 0.0 {
            return Err(NexusError::InvalidConfig(
                "BM25 k1 must be finite and non-negative",
            ));
        }
        if !(0.0..=1.0).contains(&params.b) {
            return Err(NexusError::InvalidConfig("BM25 b must be within [0, 1]"));
        }
        Ok(Self {
            tokenizer,
            params,
            ..Self::default()
        })
    }

    /// Number of indexed documents (`N`).
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Number of distinct terms.
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    pub fn contains(&self, id: EntityId) -> bool {
        self.slots.contains_key(&id)
    }

    /// Token count of a document (`|D|`).
    pub fn document_length(&self, id: EntityId) -> Option<u32> {
        let slot = *self.slots.get(&id)?;
        self.doc_len(slot)
    }

    /// Mean document length (`avgdl`); `0.0` for an empty index.
    #[allow(clippy::cast_precision_loss)] // corpus sizes are far below 2^52
    pub fn average_document_length(&self) -> f64 {
        if self.doc_ids.is_empty() {
            0.0
        } else {
            self.total_len as f64 / self.doc_ids.len() as f64
        }
    }

    /// Number of documents containing `term` (`n(t)`). `term` must already be
    /// in tokenized form.
    pub fn document_frequency(&self, term: &str) -> usize {
        self.postings.get(term).map_or(0, Vec::len)
    }

    /// `IDF(t)` of an already-tokenized term. Always positive.
    pub fn idf(&self, term: &str) -> f64 {
        self.idf_for(self.document_frequency(term))
    }

    /// Tokenizes and indexes one document. A document that yields no tokens
    /// is still counted in `N` with length zero.
    pub fn add_document(&mut self, id: EntityId, text: &str) -> Result<(), NexusError> {
        if self.slots.contains_key(&id) {
            return Err(NexusError::DuplicateDocument(id));
        }
        let slot = u32::try_from(self.doc_ids.len()).map_err(|_| NexusError::CapacityExceeded)?;

        let mut frequencies: HashMap<String, u32> = HashMap::new();
        let mut length: u32 = 0;
        self.tokenizer.for_each_token(text, |token| {
            length = length.saturating_add(1);
            // Only the first occurrence of a term allocates.
            if let Some(count) = frequencies.get_mut(token) {
                *count = count.saturating_add(1);
            } else {
                frequencies.insert(token.to_owned(), 1);
            }
        });

        for (term, term_frequency) in frequencies {
            self.postings.entry(term).or_default().push(Posting {
                slot,
                term_frequency,
            });
        }
        self.doc_ids.push(id);
        self.doc_lens.push(length);
        self.slots.insert(id, slot);
        self.total_len += u64::from(length);
        Ok(())
    }

    /// Ranks documents against `query` and returns the `top_k` best, sorted by
    /// descending score, ties by ascending id. Only documents sharing at least
    /// one term with the query are returned; an empty query, a query made of
    /// stop words or one of unseen terms yields no hits.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<LexicalHit> {
        if top_k == 0 || self.is_empty() {
            return Vec::new();
        }
        // Sorted and deduplicated: each term counts once, and every document's
        // partial scores are summed in the same order on every run.
        let mut terms = self.tokenizer.tokenize(query);
        terms.sort_unstable();
        terms.dedup();

        let average = self.average_document_length();
        let Bm25Params { k1, b } = self.params;
        let mut scores: HashMap<u32, f64> = HashMap::new();
        for term in &terms {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let idf = self.idf_for(postings.len());
            for posting in postings {
                let length = f64::from(self.doc_len(posting.slot).unwrap_or(0));
                let length_ratio = if average > 0.0 { length / average } else { 1.0 };
                let frequency = f64::from(posting.term_frequency);
                let denominator = frequency + k1 * (1.0 - b + b * length_ratio);
                *scores.entry(posting.slot).or_insert(0.0) +=
                    idf * frequency * (k1 + 1.0) / denominator;
            }
        }

        let mut best = TopK::new(top_k);
        for (slot, score) in scores {
            if let Some(&id) = usize::try_from(slot).ok().and_then(|s| self.doc_ids.get(s)) {
                best.push(id, score);
            }
        }
        best.into_sorted()
            .into_iter()
            .map(|(id, score)| LexicalHit { id, score })
            .collect()
    }

    fn doc_len(&self, slot: u32) -> Option<u32> {
        self.doc_lens.get(usize::try_from(slot).ok()?).copied()
    }

    #[allow(clippy::cast_precision_loss)] // document counts are far below 2^52
    fn idf_for(&self, document_frequency: usize) -> f64 {
        let total = self.doc_ids.len() as f64;
        let containing = document_frequency as f64;
        (1.0 + (total - containing + 0.5) / (containing + 0.5)).ln()
    }
}

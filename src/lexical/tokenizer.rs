//! Text tokenizer: Unicode normalization, lowercasing, punctuation stripping
//! and stop-word filtering.
//!
//! Pipeline, per character:
//!
//! 1. NFKD-decompose (skipped entirely for pure-ASCII input), so that
//!    compatibility forms fold together (`ﬁ` -> `fi`, full-width `Ａ` -> `A`)
//!    and accents become separate combining marks,
//! 2. drop combining marks (`é` -> `e`) and apostrophes (`don't` -> `dont`),
//! 3. lowercase,
//! 4. treat every remaining non-alphanumeric character as a separator.
//!
//! Tokens are then filtered by length and against the stop-word list.

use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

/// English stop words. Must stay sorted: lookup is a binary search.
const STOP_WORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "me", "more", "most", "my", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over",
    "own", "same", "she", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "you", "your", "yours",
];

/// Whether `token` (already lowercased) is on the built-in stop-word list.
pub fn is_stop_word(token: &str) -> bool {
    STOP_WORDS.binary_search(&token).is_ok()
}

/// Configurable tokenizer. Documents and queries must be tokenized with the
/// same configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tokenizer {
    /// Drop tokens on the built-in English stop-word list.
    pub filter_stop_words: bool,
    /// Tokens shorter than this many bytes are dropped.
    pub min_token_len: usize,
    /// Tokens longer than this many bytes are dropped (hashes, base64 blobs).
    pub max_token_len: usize,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self {
            filter_stop_words: true,
            min_token_len: 1,
            max_token_len: 64,
        }
    }
}

impl Tokenizer {
    /// Calls `emit` once per token, in order of appearance.
    ///
    /// Tokens are assembled in a single reused buffer, so tokenizing a text
    /// performs one allocation regardless of its length; callers that only
    /// look tokens up (queries, counting) never need an owned `String`.
    pub fn for_each_token(&self, text: &str, mut emit: impl FnMut(&str)) {
        let mut buffer = String::new();
        if text.is_ascii() {
            for c in text.chars() {
                self.feed(c, &mut buffer, &mut emit);
            }
        } else {
            for c in text.nfkd() {
                self.feed(c, &mut buffer, &mut emit);
            }
        }
        self.flush(&mut buffer, &mut emit);
    }

    /// Collects the tokens of `text` into owned strings.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        self.for_each_token(text, |token| tokens.push(token.to_owned()));
        tokens
    }

    fn feed(&self, c: char, buffer: &mut String, emit: &mut impl FnMut(&str)) {
        if c.is_alphanumeric() {
            // Lowercasing can itself produce combining marks (`İ` -> `i̇`).
            buffer.extend(c.to_lowercase().filter(|&lower| !is_combining_mark(lower)));
        } else if !(is_combining_mark(c) || matches!(c, '\'' | '\u{2019}')) {
            self.flush(buffer, emit);
        }
    }

    fn flush(&self, buffer: &mut String, emit: &mut impl FnMut(&str)) {
        let keep = (self.min_token_len..=self.max_token_len).contains(&buffer.len())
            && !buffer.is_empty()
            && !(self.filter_stop_words && is_stop_word(buffer));
        if keep {
            emit(buffer);
        }
        buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_words_are_sorted_and_unique() {
        assert!(STOP_WORDS.windows(2).all(|pair| pair[0] < pair[1]));
    }
}

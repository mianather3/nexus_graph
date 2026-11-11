//! Bounded top-K selection shared by the vector and lexical rankers.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::models::EntityId;

/// A float score with a total order (`f32::total_cmp` / `f64::total_cmp`).
pub(crate) trait Score: Copy {
    fn total_order(&self, other: &Self) -> Ordering;
}

impl Score for f32 {
    fn total_order(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl Score for f64 {
    fn total_order(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

/// Orders by score, then by *descending* id, so that among equal scores the
/// smaller id is the greater (better) entry.
struct Entry<S> {
    score: S,
    id: EntityId,
}

impl<S: Score> Ord for Entry<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_order(&other.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl<S: Score> PartialOrd for Entry<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: Score> PartialEq for Entry<S> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<S: Score> Eq for Entry<S> {}

/// Keeps the `limit` best `(id, score)` pairs seen so far in a min-heap whose
/// root is the current worst survivor: `O(n log limit)` time, `O(limit)` space.
pub(crate) struct TopK<S> {
    limit: usize,
    heap: BinaryHeap<Reverse<Entry<S>>>,
}

impl<S: Score> TopK<S> {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit,
            // Cap the up-front reservation: `limit` may be far larger than the corpus.
            heap: BinaryHeap::with_capacity(limit.min(1024)),
        }
    }

    pub(crate) fn push(&mut self, id: EntityId, score: S) {
        let entry = Entry { score, id };
        if self.heap.len() < self.limit {
            self.heap.push(Reverse(entry));
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if entry > worst.0 {
                *worst = Reverse(entry);
            }
        }
    }

    /// Best first: descending score, ties by ascending id.
    pub(crate) fn into_sorted(self) -> Vec<(EntityId, S)> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(entry)| (entry.id, entry.score))
            .collect()
    }
}

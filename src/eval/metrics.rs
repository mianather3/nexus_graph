//! Retrieval quality metrics. All functions are pure and operate on entity
//! ids, so they can be unit-tested without a graph.

// The metric functions take plain `HashSet`s: being generic over the hasher
// would add noise to every signature for no caller that needs it.
#![allow(clippy::implicit_hasher)]

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::models::EntityId;

/// `1 / rank` of the first relevant item in `ranked` (1-based), or `0.0` if
/// none of them is relevant.
pub fn reciprocal_rank(ranked: &[EntityId], relevant: &HashSet<EntityId>) -> f64 {
    ranked
        .iter()
        .zip(1_u32..)
        .find(|(id, _)| relevant.contains(id))
        .map_or(0.0, |(_, rank)| 1.0 / f64::from(rank))
}

/// Fraction of the relevant items found in the first `k` positions of
/// `ranked`. `0.0` if there are no relevant items.
pub fn recall_at_k(ranked: &[EntityId], relevant: &HashSet<EntityId>, k: usize) -> f64 {
    let found: HashSet<EntityId> = ranked
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .copied()
        .collect();
    ratio(found.len(), relevant.len())
}

/// Fraction of the distinct `targets` present in `context`. `0.0` if there
/// are no targets.
pub fn coverage(context: &HashSet<EntityId>, targets: &[EntityId]) -> f64 {
    let distinct: HashSet<EntityId> = targets.iter().copied().collect();
    let present = distinct.iter().filter(|id| context.contains(id)).count();
    ratio(present, distinct.len())
}

/// An undirected edge key: the two endpoint ids in ascending order.
pub fn edge_key(a: EntityId, b: EntityId) -> (EntityId, EntityId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Whether every hop of `path` is present in `context_edges` (undirected
/// keys from [`edge_key`]), i.e. whether a reader of the context could walk
/// the whole evidence chain. Paths with fewer than two entities have no hops
/// and are never "recovered".
pub fn path_recovered(context_edges: &HashSet<(EntityId, EntityId)>, path: &[EntityId]) -> bool {
    path.len() >= 2
        && path
            .iter()
            .zip(path.iter().skip(1))
            .all(|(&a, &b)| context_edges.contains(&edge_key(a, b)))
}

#[allow(clippy::cast_precision_loss)] // set sizes are tiny
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Aggregated metrics of one retrieval system over a set of queries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Queries averaged over.
    pub queries: usize,
    /// Mean reciprocal rank of the first relevant entity among the hits.
    pub mrr: f64,
    /// Mean recall of relevant entities within the top 5 hits.
    pub recall_at_5: f64,
    /// Mean recall of relevant entities within the top 10 hits.
    pub recall_at_10: f64,
    /// Mean fraction of relevant entities present anywhere in the returned
    /// context (hits plus graph expansion). Equals recall at `top_k` for a
    /// system without expansion.
    pub answer_in_context: f64,
    /// Context Coverage Score: mean fraction of ground-truth evidence
    /// entities (the evidence path, or the relevant set if the query has no
    /// path) present in the returned context.
    pub evidence_coverage: f64,
    /// Queries that carry an evidence path of at least one hop.
    pub path_queries: usize,
    /// Path reachability: fraction of `path_queries` whose every evidence
    /// hop is an edge of the returned context.
    pub path_recovery: f64,
    /// Mean number of entities in the returned context: what the coverage
    /// above costs in prompt budget.
    pub mean_context_entities: f64,
}

/// Running sums behind [`Metrics`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MetricsAccumulator {
    queries: usize,
    reciprocal_rank: f64,
    recall_at_5: f64,
    recall_at_10: f64,
    answer_in_context: f64,
    evidence_coverage: f64,
    path_queries: usize,
    paths_recovered: usize,
    context_entities: usize,
}

/// Per-query measurements fed to a [`MetricsAccumulator`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueryOutcome {
    pub reciprocal_rank: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub answer_in_context: f64,
    pub evidence_coverage: f64,
    /// `None` if the query has no evidence path.
    pub path_recovered: Option<bool>,
    pub context_entities: usize,
}

impl MetricsAccumulator {
    pub(crate) fn add(&mut self, outcome: &QueryOutcome) {
        self.queries += 1;
        self.reciprocal_rank += outcome.reciprocal_rank;
        self.recall_at_5 += outcome.recall_at_5;
        self.recall_at_10 += outcome.recall_at_10;
        self.answer_in_context += outcome.answer_in_context;
        self.evidence_coverage += outcome.evidence_coverage;
        if let Some(recovered) = outcome.path_recovered {
            self.path_queries += 1;
            self.paths_recovered += usize::from(recovered);
        }
        self.context_entities += outcome.context_entities;
    }

    #[allow(clippy::cast_precision_loss)] // query counts are tiny
    pub(crate) fn finish(&self) -> Metrics {
        let mean = |sum: f64| {
            if self.queries == 0 {
                0.0
            } else {
                sum / self.queries as f64
            }
        };
        Metrics {
            queries: self.queries,
            mrr: mean(self.reciprocal_rank),
            recall_at_5: mean(self.recall_at_5),
            recall_at_10: mean(self.recall_at_10),
            answer_in_context: mean(self.answer_in_context),
            evidence_coverage: mean(self.evidence_coverage),
            path_queries: self.path_queries,
            path_recovery: ratio(self.paths_recovered, self.path_queries),
            mean_context_entities: mean(self.context_entities as f64),
        }
    }
}

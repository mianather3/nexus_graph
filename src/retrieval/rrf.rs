//! Reciprocal Rank Fusion.
//!
//! ```text
//! RRF(d) = sum over rankings m of  w_m / (k + rank_m(d))
//! ```
//!
//! with 1-based ranks. A document absent from a ranking contributes nothing
//! for it. Because only ranks enter the formula, rankings whose raw scores
//! live on unrelated scales (cosine similarity vs. BM25) can be merged without
//! any score calibration.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::NexusError;
use crate::models::EntityId;

/// Default smoothing constant.
pub const DEFAULT_RRF_K: f64 = 60.0;

/// One input ranking: ids best-first, and the weight of the channel.
#[derive(Debug, Clone, Copy)]
pub struct RankedList<'a> {
    pub weight: f64,
    pub ids: &'a [EntityId],
}

/// One fused result.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FusedHit {
    pub id: EntityId,
    pub score: f64,
}

/// Fuses rankings into one list sorted by descending RRF score, ties by
/// ascending id.
///
/// `k` must be finite and non-negative; weights must be finite and
/// non-negative. If an id occurs more than once in a ranking, only its first
/// (best) position counts.
pub fn reciprocal_rank_fusion(
    lists: &[RankedList<'_>],
    k: f64,
) -> Result<Vec<FusedHit>, NexusError> {
    if !k.is_finite() || k < 0.0 {
        return Err(NexusError::InvalidConfig(
            "RRF k must be finite and non-negative",
        ));
    }
    if lists
        .iter()
        .any(|l| !l.weight.is_finite() || l.weight < 0.0)
    {
        return Err(NexusError::InvalidConfig(
            "RRF weights must be finite and non-negative",
        ));
    }

    // Contributions are added list by list, so each id's sum is evaluated in a
    // fixed order and the result is bit-for-bit reproducible.
    let mut scores: HashMap<EntityId, f64> = HashMap::new();
    for list in lists {
        let mut seen: HashSet<EntityId> = HashSet::with_capacity(list.ids.len());
        let mut rank: u32 = 0;
        for &id in list.ids {
            if !seen.insert(id) {
                continue;
            }
            rank = rank.checked_add(1).ok_or(NexusError::CapacityExceeded)?;
            *scores.entry(id).or_insert(0.0) += list.weight / (k + f64::from(rank));
        }
    }

    let mut fused: Vec<FusedHit> = scores
        .into_iter()
        .map(|(id, score)| FusedHit { id, score })
        .collect();
    fused.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    Ok(fused)
}

//! Deterministic verification of LLM-extracted claims against the graph.
//!
//! A claim `S -R-> T` is **verified** when, in order of preference:
//!
//! 1. the direct edge `S -R-> T` exists, or
//! 2. `R` is transitive ([`RelationType::is_transitive`]) and a walk of at most
//!    `max_hops` edges, all of type `R`, leads from `S` to `T`.
//!
//! Anything else is **rejected** with a reason and a broken-path trace. The
//! whole batch is checked under one read lock, and no step depends on hash
//! iteration order, so the same graph and claims always produce the same
//! report.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fmt;

use petgraph::graph::{EdgeIndex, NodeIndex};
use serde::{Deserialize, Serialize};

use crate::graph::{GraphState, KnowledgeGraph, Path};
use crate::models::{GraphClaim, RelationType};

/// Tunables of the verification contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierConfig {
    /// Longest chain of same-typed edges accepted as proof of a transitive
    /// relation. `1` accepts direct edges only.
    pub max_hops: usize,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self { max_hops: 3 }
    }
}

/// How a verified claim is supported by the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    /// A single edge states the claim verbatim.
    Direct,
    /// A chain of edges of the claimed (transitive) relation entails it.
    Transitive,
}

/// A claim the graph supports, with the evidence path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedClaim {
    /// Position of the claim in the input slice.
    pub claim_index: usize,
    pub claim: GraphClaim,
    pub evidence: EvidenceKind,
    pub path: Path,
    /// Product of the edge weights along `path`.
    pub confidence: f32,
}

/// Why a claim was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectionReason {
    /// One or both entities are not in the graph (names as given in the claim).
    UnknownEntity { missing: Vec<String> },
    /// Source and target are directly connected, but only by other relations.
    RelationMismatch { found: Vec<RelationType> },
    /// The claimed relation exists in the opposite direction only.
    DirectionReversed,
    /// A chain of the claimed relation exists but is longer than `max_hops`.
    DepthExceeded {
        required_hops: usize,
        max_hops: usize,
    },
    /// No chain of the claimed relation leads from source to target.
    NoSupportingPath,
}

/// The single hop that would have been needed but is absent from the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingHop {
    pub from: String,
    pub relation: RelationType,
    pub to: String,
}

/// Where the evidence for a rejected claim breaks down.
///
/// `reached` is the part of the graph that does support the claim: a chain of
/// edges of the claimed relation starting at the claim's source. If the target
/// is reachable only beyond `max_hops`, it is the first `max_hops` edges of the
/// shortest such chain. Otherwise it is the longest shortest-chain within
/// `max_hops` (ties broken by smallest entity id), or empty when the source has
/// no usable outgoing edge (always the case for non-transitive relations).
///
/// `missing_hop` is the edge that would have completed the proof: from the end
/// of `reached` straight to the claim's target. It is absent from the graph,
/// except under [`RejectionReason::DepthExceeded`], where the target lies
/// further along the chain than the hop budget allows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrokenPathTrace {
    pub reached: Path,
    pub missing_hop: MissingHop,
}

/// Renders as `a -[DependsOn]-> b -[DependsOn]-x-> c`, the `x` marking the
/// missing hop.
impl fmt::Display for BrokenPathTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.reached.is_empty() {
            f.write_str(&self.missing_hop.from)?;
        } else {
            write!(f, "{}", self.reached)?;
        }
        write!(
            f,
            " -[{}]-x-> {}",
            self.missing_hop.relation, self.missing_hop.to
        )
    }
}

/// A claim the graph does not support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrokenClaim {
    /// Position of the claim in the input slice.
    pub claim_index: usize,
    pub claim: GraphClaim,
    pub reason: RejectionReason,
    pub trace: BrokenPathTrace,
}

/// Outcome of verifying a batch of claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub total_claims: usize,
    /// Supported claims, in input order.
    pub verified: Vec<VerifiedClaim>,
    /// Hallucinated claims, in input order.
    pub hallucinated: Vec<BrokenClaim>,
    /// Mean per-claim confidence in `[0.0, 1.0]`, where a rejected claim counts
    /// as `0.0`. An empty batch scores `1.0` (nothing was contradicted).
    pub confidence_score: f32,
}

impl VerificationReport {
    /// `true` when every claim was verified.
    pub fn is_fully_verified(&self) -> bool {
        self.hallucinated.is_empty()
    }

    /// Fraction of claims that were verified, ignoring edge weights.
    #[allow(clippy::cast_precision_loss)]
    pub fn verified_ratio(&self) -> f32 {
        if self.total_claims == 0 {
            1.0
        } else {
            self.verified.len() as f32 / self.total_claims as f32
        }
    }
}

/// Verifies `claims` against `graph` under a single consistent read view.
#[allow(clippy::cast_precision_loss)]
pub fn verify_claims(
    graph: &KnowledgeGraph,
    claims: &[GraphClaim],
    config: VerifierConfig,
) -> VerificationReport {
    let view = graph.read();
    let mut verified = Vec::new();
    let mut hallucinated = Vec::new();

    for (claim_index, claim) in claims.iter().enumerate() {
        match verify_one(&view, claim, config) {
            Ok((evidence, path)) => verified.push(VerifiedClaim {
                claim_index,
                claim: claim.clone(),
                evidence,
                confidence: path.confidence().clamp(0.0, 1.0),
                path,
            }),
            Err(rejection) => {
                let (reason, trace) = *rejection;
                hallucinated.push(BrokenClaim {
                    claim_index,
                    claim: claim.clone(),
                    reason,
                    trace,
                });
            }
        }
    }

    let confidence_score = if claims.is_empty() {
        1.0
    } else {
        let sum: f32 = verified.iter().map(|v| v.confidence).sum();
        (sum / claims.len() as f32).clamp(0.0, 1.0)
    };

    VerificationReport {
        total_claims: claims.len(),
        verified,
        hallucinated,
        confidence_score,
    }
}

type Rejection = (RejectionReason, BrokenPathTrace);

fn verify_one(
    view: &GraphState,
    claim: &GraphClaim,
    config: VerifierConfig,
) -> Result<(EvidenceKind, Path), Box<Rejection>> {
    let source = view.node_index_by_name(&claim.source_entity);
    let target = view.node_index_by_name(&claim.target_entity);
    let (Some(source), Some(target)) = (source, target) else {
        let mut missing = Vec::new();
        if source.is_none() {
            missing.push(claim.source_entity.clone());
        }
        if target.is_none() {
            missing.push(claim.target_entity.clone());
        }
        let trace = BrokenPathTrace {
            reached: Path::default(),
            missing_hop: MissingHop {
                from: claim.source_entity.clone(),
                relation: claim.relation.clone(),
                to: claim.target_entity.clone(),
            },
        };
        return Err(Box::new((
            RejectionReason::UnknownEntity { missing },
            trace,
        )));
    };

    if config.max_hops >= 1 {
        let direct =
            view.find_connecting_paths_filtered(source, target, 1, Some(&claim.relation), 1);
        if let Some(path) = direct.into_iter().next() {
            return Ok((EvidenceKind::Direct, path));
        }
    }
    let chain_limit = if claim.relation.is_transitive() {
        config.max_hops
    } else {
        config.max_hops.min(1)
    };
    if chain_limit >= 2 {
        if let Some(path) = view.best_relation_path(source, target, &claim.relation, chain_limit) {
            return Ok((EvidenceKind::Transitive, path));
        }
    }

    Err(Box::new(diagnose(
        view,
        claim,
        source,
        target,
        chain_limit,
        config,
    )))
}

/// Explains a rejection for a claim whose entities both exist.
fn diagnose(
    view: &GraphState,
    claim: &GraphClaim,
    source: NodeIndex,
    target: NodeIndex,
    chain_limit: usize,
    config: VerifierConfig,
) -> Rejection {
    let name_of = |idx: NodeIndex, fallback: &str| {
        view.node(idx)
            .map_or_else(|| fallback.to_owned(), |n| n.canonical_name.clone())
    };

    // Shortest chains of the claimed relation from the source. Unbounded for
    // transitive relations so that an over-long chain can be reported as such.
    // Non-transitive relations admit no chain at all, so nothing is explored.
    let search_limit = if claim.relation.is_transitive() {
        usize::MAX
    } else {
        0
    };
    let reach = RelationReach::explore(view, source, &claim.relation, search_limit);
    let required_hops = reach.depth_of(target).filter(|&d| d > chain_limit);

    let other_relations: Vec<RelationType> = view
        .relations_between(source, target)
        .into_iter()
        .map(|e| e.relation.clone())
        .filter(|r| r != &claim.relation)
        .collect();
    let reason = if !other_relations.is_empty() {
        RejectionReason::RelationMismatch {
            found: other_relations,
        }
    } else if view
        .find_relation(target, source, &claim.relation)
        .is_some()
    {
        RejectionReason::DirectionReversed
    } else if let Some(required_hops) = required_hops {
        RejectionReason::DepthExceeded {
            required_hops,
            max_hops: config.max_hops,
        }
    } else {
        RejectionReason::NoSupportingPath
    };

    let reached_edges = if required_hops.is_some() {
        let mut edges = reach.edges_to(target);
        edges.truncate(chain_limit);
        edges
    } else {
        reach
            .deepest_within(view, chain_limit)
            .map(|node| reach.edges_to(node))
            .unwrap_or_default()
    };
    let reached = view.path_from_edges(&reached_edges);
    let from = reached.hops.last().map_or_else(
        || name_of(source, &claim.source_entity),
        |hop| hop.target_name.clone(),
    );
    let trace = BrokenPathTrace {
        reached,
        missing_hop: MissingHop {
            from,
            relation: claim.relation.clone(),
            to: name_of(target, &claim.target_entity),
        },
    };
    (reason, trace)
}

/// Breadth-first shortest-chain tree over edges of a single relation type.
struct RelationReach {
    source: NodeIndex,
    /// node -> (depth, edge and predecessor it was first reached through).
    /// The source itself is only recorded if a cycle leads back to it.
    reached: HashMap<NodeIndex, (usize, EdgeIndex, NodeIndex)>,
}

impl RelationReach {
    fn explore(
        view: &GraphState,
        source: NodeIndex,
        relation: &RelationType,
        limit: usize,
    ) -> Self {
        let mut reached: HashMap<NodeIndex, (usize, EdgeIndex, NodeIndex)> = HashMap::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::from([(source, 0)]);
        while let Some((node, depth)) = queue.pop_front() {
            if depth >= limit {
                continue;
            }
            for (next, edge) in view.outgoing_edges(node, Some(relation)) {
                if let Entry::Vacant(slot) = reached.entry(next) {
                    slot.insert((depth + 1, edge, node));
                    if next != source {
                        queue.push_back((next, depth + 1));
                    }
                }
            }
        }
        Self { source, reached }
    }

    fn depth_of(&self, node: NodeIndex) -> Option<usize> {
        self.reached.get(&node).map(|&(depth, _, _)| depth)
    }

    /// Edges of the shortest chain from the source to `node`, in order.
    fn edges_to(&self, node: NodeIndex) -> Vec<EdgeIndex> {
        let mut edges = Vec::new();
        let mut cursor = node;
        while let Some(&(_, edge, prev)) = self.reached.get(&cursor) {
            edges.push(edge);
            if prev == self.source {
                break;
            }
            cursor = prev;
        }
        edges.reverse();
        edges
    }

    /// The node farthest from the source within `limit` hops; ties are broken
    /// by smallest entity id. The source is never returned.
    fn deepest_within(&self, view: &GraphState, limit: usize) -> Option<NodeIndex> {
        self.reached
            .iter()
            .filter(|&(&node, &(depth, _, _))| depth <= limit && node != self.source)
            .filter_map(|(&node, &(depth, _, _))| Some((depth, view.node(node)?.id, node)))
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
            .map(|(_, _, node)| node)
    }
}

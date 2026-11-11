//! `NexusGraph`: a hybrid vector-knowledge graph retrieval engine with
//! deterministic path verification.
//!
//! * [`models`] - entity, relation and claim schema contracts.
//! * [`graph`] - thread-safe in-memory graph, indices and traversal.
//! * [`verifier`] - deterministic verification of LLM claims against the graph.
//! * [`storage`] - atomic binary snapshots.

pub mod graph;
pub mod models;
pub mod storage;
pub mod verifier;

pub use graph::{
    GraphError, GraphSnapshot, GraphState, KnowledgeGraph, Path, PathHop, Subgraph, SubgraphNode,
};
pub use models::{
    normalize_entity_name, Attributes, CitationRef, EntityId, EntityNode, EntityType, GraphClaim,
    RelationEdge, RelationType,
};
pub use petgraph::graph::{EdgeIndex, NodeIndex};
pub use storage::{load_snapshot, save_snapshot, StorageError};
pub use verifier::{
    verify_claims, BrokenClaim, BrokenPathTrace, EvidenceKind, MissingHop, RejectionReason,
    VerificationReport, VerifiedClaim, VerifierConfig,
};

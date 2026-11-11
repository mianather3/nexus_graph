//! `NexusGraph`: a hybrid vector-knowledge graph retrieval engine with
//! deterministic path verification.
//!
//! * [`models`] - entity, relation and claim schema contracts.
//! * [`graph`] - thread-safe in-memory graph, indices and traversal.
//! * [`verifier`] - deterministic verification of LLM claims against the graph.
//! * [`storage`] - atomic binary snapshots.
//! * [`vector`] - SIMD dot/cosine kernels and a contiguous dense vector store.
//! * [`lexical`] - tokenizer and BM25 inverted index.
//! * [`retrieval`] - hybrid search: k-NN + BM25, rank fusion, graph expansion.
//! * [`server`] - multi-tenant HTTP/SSE service over all of the above.
//! * [`error`] - [`NexusError`], the error type of the retrieval layers.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

pub mod error;
pub mod graph;
pub mod lexical;
pub mod models;
pub mod retrieval;
pub mod server;
pub mod storage;
mod topk;
pub mod vector;
pub mod verifier;

pub use error::NexusError;
pub use graph::{
    GraphError, GraphSnapshot, GraphState, KnowledgeGraph, Path, PathHop, Subgraph, SubgraphNode,
};
pub use lexical::{Bm25Index, Bm25Params, LexicalHit, Tokenizer};
pub use models::{
    normalize_entity_name, Attributes, CitationRef, EntityId, EntityNode, EntityType, GraphClaim,
    RelationEdge, RelationType,
};
pub use petgraph::graph::{EdgeIndex, NodeIndex};
pub use retrieval::{
    reciprocal_rank_fusion, ChannelHit, FusedHit, HybridQuery, HybridRetriever, IndexStats,
    RankedList, RetrievalBundle, RetrievalConfig, RetrievedEntity, DEFAULT_RRF_K,
};
pub use storage::{load_snapshot, save_snapshot, StorageError};
pub use vector::{
    cosine_distance, cosine_similarity, cosine_similarity_with, dot_product, dot_product_with,
    Kernel, VectorHit, VectorStore,
};
pub use verifier::{
    verify_claims, BrokenClaim, BrokenPathTrace, EvidenceKind, MissingHop, RejectionReason,
    VerificationReport, VerifiedClaim, VerifierConfig,
};

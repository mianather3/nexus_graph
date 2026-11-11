//! Request and response bodies of the HTTP API.

use serde::{Deserialize, Serialize};

use crate::models::{Attributes, EntityType, GraphClaim, RelationType};

/// An entity to ingest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityInput {
    pub name: String,
    #[serde(rename = "type")]
    pub entity_type: EntityType,
    #[serde(default)]
    pub attributes: Attributes,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

/// A relation to ingest; endpoints are referenced by canonical entity name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationInput {
    pub source: String,
    pub target: String,
    pub relation: RelationType,
    /// Confidence in `[0, 1]`; defaults to `1.0`.
    #[serde(default = "full_confidence")]
    pub weight: f32,
    #[serde(default)]
    pub metadata: Attributes,
}

fn full_confidence() -> f32 {
    1.0
}

fn yes() -> bool {
    true
}

/// Body of `POST .../entities/batch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityBatch {
    pub entities: Vec<EntityInput>,
}

/// Body of `POST .../relations/batch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationBatch {
    pub relations: Vec<RelationInput>,
}

/// Query parameters of the batch ingestion endpoints.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestOptions {
    /// `true` (default): respond once the batch has been applied, with an
    /// [`IngestReport`]. `false`: respond `202 Accepted` as soon as the batch
    /// is queued.
    #[serde(default = "yes")]
    pub wait: bool,
}

/// One batch item that could not be applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedItem {
    /// Position of the item in the submitted batch.
    pub index: usize,
    pub detail: String,
}

/// Outcome of an applied batch. Items are applied independently and in order:
/// a rejected item does not undo the ones before it or stop the ones after.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestReport {
    pub ingested: usize,
    pub rejected: Vec<RejectedItem>,
}

/// Response of a batch submitted with `wait=false`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedResponse {
    pub status: String,
    /// Jobs accepted for this namespace and not yet applied, including this one.
    pub pending_jobs: u64,
}

/// Body of `POST .../search/hybrid`. At least one of `text` and `embedding`
/// is required; every other member overrides a server default.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridSearchRequest {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub candidate_pool: Option<usize>,
    #[serde(default)]
    pub rrf_k: Option<f64>,
    #[serde(default)]
    pub vector_weight: Option<f64>,
    #[serde(default)]
    pub lexical_weight: Option<f64>,
    #[serde(default)]
    pub expansion_hops: Option<usize>,
    /// Return the embeddings of context entities. Off by default: they are
    /// large and rarely useful to a caller that already ran the search.
    #[serde(default)]
    pub include_embeddings: bool,
}

/// Body of `POST .../verify` and `POST .../verify/stream`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifyRequest {
    pub claims: Vec<GraphClaim>,
    #[serde(default)]
    pub max_hops: Option<usize>,
}

/// Query parameters of `GET .../verify/stream`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifyStreamParams {
    /// URL-encoded JSON array of claims.
    pub claims: String,
    #[serde(default)]
    pub max_hops: Option<usize>,
}

/// Payload of the final `complete` event of a verification stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamSummary {
    pub total_claims: usize,
    pub verified: usize,
    pub rejected: usize,
    /// Same definition as `VerificationReport::confidence_score`.
    pub confidence_score: f32,
}

/// Query parameters of `POST .../snapshot`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotOptions {
    /// `false` (default): respond `202 Accepted` and persist in the
    /// background. `true`: respond once the snapshot is on disk.
    #[serde(default)]
    pub wait: bool,
}

/// A completed snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotInfo {
    pub completed_at_unix_ms: u64,
    pub duration_ms: u64,
    pub bytes: u64,
}

/// Persistence state of a namespace.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotStatus {
    /// Snapshots requested and not yet finished.
    pub pending: u64,
    pub last_success: Option<SnapshotInfo>,
    pub last_error: Option<String>,
}

/// Ingestion queue state of a namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestStats {
    pub queue_capacity: usize,
    /// Jobs waiting in the queue; at `queue_capacity` new jobs get `429`.
    pub queued_jobs: usize,
    /// Jobs accepted and not yet applied (queued or in flight).
    pub pending_jobs: u64,
    pub accepted_jobs: u64,
    pub completed_jobs: u64,
    /// Jobs refused with `429` because the queue was full.
    pub rejected_jobs: u64,
}

/// Response of `GET .../stats`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespaceStats {
    pub namespace: String,
    pub entities: usize,
    pub relations: usize,
    pub vectors: usize,
    pub vector_dim: Option<usize>,
    pub terms: usize,
    pub documents: usize,
    pub ingest: IngestStats,
    pub snapshot: SnapshotStatus,
}

//! Core data primitives and schema contracts.
//!
//! Every type here is serialized both as JSON (API contracts) and as bincode
//! (disk snapshots). Bincode is not self-describing, so these types must not
//! use `#[serde(skip_serializing_if)]`, `#[serde(untagged)]` or `#[serde(flatten)]`.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Free-form string attributes. A `BTreeMap` keeps serialization order stable,
/// which makes snapshots byte-for-byte reproducible.
pub type Attributes = BTreeMap<String, String>;

/// Stable, graph-unique identifier of an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EntityId(pub u64);

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Category of an entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EntityType {
    Organization,
    Concept,
    Service,
    Artifact,
    /// Domain-specific type not covered by the built-in variants.
    Custom(String),
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Organization => f.write_str("Organization"),
            Self::Concept => f.write_str("Concept"),
            Self::Service => f.write_str("Service"),
            Self::Artifact => f.write_str("Artifact"),
            Self::Custom(name) => f.write_str(name),
        }
    }
}

/// Semantic type of a directed relation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RelationType {
    DependsOn,
    ProducedBy,
    Supersedes,
    Contains,
    /// Domain-specific relation not covered by the built-in variants.
    Custom(String),
}

impl RelationType {
    /// Whether a chain `A -R-> B -R-> C` entails `A -R-> C`.
    ///
    /// This is the relationship contract the verifier applies to multi-hop
    /// evidence: only transitive relations may be proven by a path longer than
    /// one edge. Custom relations are conservatively treated as non-transitive.
    pub fn is_transitive(&self) -> bool {
        match self {
            Self::DependsOn | Self::Supersedes | Self::Contains => true,
            Self::ProducedBy | Self::Custom(_) => false,
        }
    }
}

impl fmt::Display for RelationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DependsOn => f.write_str("DependsOn"),
            Self::ProducedBy => f.write_str("ProducedBy"),
            Self::Supersedes => f.write_str("Supersedes"),
            Self::Contains => f.write_str("Contains"),
            Self::Custom(name) => f.write_str(name),
        }
    }
}

/// A vertex of the knowledge graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityNode {
    pub id: EntityId,
    /// Human-readable canonical name. Unique per graph after normalization
    /// (see [`normalize_entity_name`]).
    pub canonical_name: String,
    pub entity_type: EntityType,
    pub attributes: Attributes,
    /// Optional dense embedding. All embeddings in one graph share a dimension.
    pub embedding: Option<Vec<f32>>,
}

/// A directed, typed, weighted edge of the knowledge graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationEdge {
    pub source: EntityId,
    pub target: EntityId,
    pub relation: RelationType,
    /// Confidence in `[0.0, 1.0]`.
    pub weight: f32,
    pub metadata: Attributes,
}

/// Pointer to the retrieved chunk an LLM cited as evidence for a claim.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CitationRef {
    pub document_id: String,
    pub chunk_id: String,
}

/// A structured assertion extracted from an LLM response:
/// "`source_entity` -`relation`-> `target_entity`, according to `citation`".
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphClaim {
    pub source_entity: String,
    pub target_entity: String,
    pub relation: RelationType,
    pub citation: CitationRef,
}

/// Key under which an entity name is indexed: trimmed and lowercased, so that
/// `" Auth-Service "` and `"auth-service"` resolve to the same entity.
pub fn normalize_entity_name(name: &str) -> String {
    name.trim().to_lowercase()
}

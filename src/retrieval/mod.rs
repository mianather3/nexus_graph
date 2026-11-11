//! Hybrid retrieval orchestrator.
//!
//! A query runs through up to two channels: exact cosine k-NN over entity
//! embeddings and BM25 over entity text. The two rankings are merged with
//! Reciprocal Rank Fusion, and the fused top-K entities seed a k-hop
//! expansion of the knowledge graph, so the returned bundle carries both the
//! ranked hits and the topology around them.

pub mod rrf;

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use rrf::{reciprocal_rank_fusion, FusedHit, RankedList, DEFAULT_RRF_K};

use crate::error::NexusError;
use crate::graph::{GraphError, KnowledgeGraph, Subgraph};
use crate::lexical::{Bm25Index, Bm25Params, Tokenizer};
use crate::models::{EntityId, EntityNode, EntityType};
use crate::vector::VectorStore;

/// Tunables of a hybrid search.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Number of fused entities returned, and used as expansion seeds.
    pub top_k: usize,
    /// Candidates taken from each channel before fusion. Raised to `top_k`
    /// if smaller.
    pub candidate_pool: usize,
    /// RRF smoothing constant `k`.
    pub rrf_k: f64,
    /// RRF weight of the dense vector channel. `0.0` disables the channel.
    pub vector_weight: f64,
    /// RRF weight of the BM25 channel. `0.0` disables the channel.
    pub lexical_weight: f64,
    /// Hops of graph context gathered around the fused hits. `0` returns just
    /// the hits and the edges among them.
    pub expansion_hops: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            candidate_pool: 50,
            rrf_k: DEFAULT_RRF_K,
            vector_weight: 1.0,
            lexical_weight: 1.0,
            expansion_hops: 1,
        }
    }
}

impl RetrievalConfig {
    pub fn validate(&self) -> Result<(), NexusError> {
        if self.top_k == 0 {
            return Err(NexusError::InvalidConfig("top_k must be at least 1"));
        }
        if !self.rrf_k.is_finite() || self.rrf_k < 0.0 {
            return Err(NexusError::InvalidConfig(
                "RRF k must be finite and non-negative",
            ));
        }
        let valid_weight = |w: f64| w.is_finite() && w >= 0.0;
        if !valid_weight(self.vector_weight) || !valid_weight(self.lexical_weight) {
            return Err(NexusError::InvalidConfig(
                "RRF weights must be finite and non-negative",
            ));
        }
        if self.vector_weight == 0.0 && self.lexical_weight == 0.0 {
            return Err(NexusError::InvalidConfig(
                "at least one channel needs a positive weight",
            ));
        }
        Ok(())
    }
}

/// A hybrid query. Either part may be omitted, but not both.
#[derive(Debug, Clone, Copy, Default)]
pub struct HybridQuery<'a> {
    pub text: Option<&'a str>,
    pub embedding: Option<&'a [f32]>,
}

/// Position and raw score of a hit within one channel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChannelHit {
    /// 1-based rank within the channel.
    pub rank: usize,
    /// Cosine similarity (vector channel) or BM25 score (lexical channel).
    pub score: f64,
}

/// One fused result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedEntity {
    pub id: EntityId,
    pub name: String,
    pub entity_type: EntityType,
    /// RRF score.
    pub fused_score: f64,
    /// Present if the vector channel ranked this entity.
    pub vector: Option<ChannelHit>,
    /// Present if the lexical channel ranked this entity.
    pub lexical: Option<ChannelHit>,
}

/// Result of a hybrid search: ranked hits plus their graph neighborhood.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalBundle {
    /// Fused hits, best first.
    pub hits: Vec<RetrievedEntity>,
    /// `expansion_hops`-hop subgraph around the hits. Hits appear at depth 0.
    pub context: Subgraph,
}

/// Size of the retrieval indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexStats {
    pub documents: usize,
    pub terms: usize,
    pub vectors: usize,
    pub vector_dim: Option<usize>,
}

/// Hybrid vector + lexical retriever over the entities of a knowledge graph.
///
/// Indexing takes `&mut self` and searching `&self`; wrap the retriever in a
/// lock to index while serving queries. The graph itself is shared and may
/// keep changing: entities added after construction become searchable once
/// passed to [`Self::index_entity`], and always show up in graph context.
#[derive(Debug)]
pub struct HybridRetriever {
    graph: Arc<KnowledgeGraph>,
    /// Created by the first indexed embedding, which fixes the dimension.
    vectors: Option<VectorStore>,
    lexical: Bm25Index,
    config: RetrievalConfig,
}

impl HybridRetriever {
    /// Creates a retriever with empty indices.
    pub fn new(graph: Arc<KnowledgeGraph>, config: RetrievalConfig) -> Result<Self, NexusError> {
        config.validate()?;
        Ok(Self {
            graph,
            vectors: None,
            lexical: Bm25Index::new(Tokenizer::default(), Bm25Params::default())?,
            config,
        })
    }

    /// Creates a retriever and indexes every entity currently in the graph.
    pub fn from_graph(
        graph: Arc<KnowledgeGraph>,
        config: RetrievalConfig,
    ) -> Result<Self, NexusError> {
        let mut retriever = Self::new(graph, config)?;
        let graph = Arc::clone(&retriever.graph);
        let view = graph.read();
        for node in view.nodes() {
            retriever.index_node(node)?;
        }
        Ok(retriever)
    }

    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    pub fn graph(&self) -> &Arc<KnowledgeGraph> {
        &self.graph
    }

    pub fn stats(&self) -> IndexStats {
        IndexStats {
            documents: self.lexical.len(),
            terms: self.lexical.term_count(),
            vectors: self.vectors.as_ref().map_or(0, VectorStore::len),
            vector_dim: self.vectors.as_ref().map(VectorStore::dim),
        }
    }

    /// Indexes one entity of the graph: its text (name, type and attribute
    /// values) for BM25 and, if it has one, its embedding for k-NN.
    pub fn index_entity(&mut self, id: EntityId) -> Result<(), NexusError> {
        let graph = Arc::clone(&self.graph);
        let view = graph.read();
        let node = view
            .node_index_by_id(id)
            .and_then(|index| view.node(index))
            .ok_or(GraphError::EntityIdNotFound(id))?;
        self.index_node(node)
    }

    fn index_node(&mut self, node: &EntityNode) -> Result<(), NexusError> {
        if self.lexical.contains(node.id) {
            return Err(NexusError::DuplicateDocument(node.id));
        }
        // The vector goes first: it is the only step that can fail from here.
        if let Some(embedding) = node.embedding.as_deref() {
            if let Some(store) = &mut self.vectors {
                store.insert(node.id, embedding)?;
            } else {
                let mut store = VectorStore::new(embedding.len())?;
                store.insert(node.id, embedding)?;
                self.vectors = Some(store);
            }
        }
        self.lexical.add_document(node.id, &entity_text(node))
    }

    /// Runs a hybrid search with the retriever's configuration.
    pub fn search(&self, query: &HybridQuery<'_>) -> Result<RetrievalBundle, NexusError> {
        self.search_with(query, &self.config)
    }

    /// Runs a hybrid search with a per-query configuration.
    pub fn search_with(
        &self,
        query: &HybridQuery<'_>,
        config: &RetrievalConfig,
    ) -> Result<RetrievalBundle, NexusError> {
        config.validate()?;
        if query.text.is_none() && query.embedding.is_none() {
            return Err(NexusError::EmptyQuery);
        }
        let pool = config.candidate_pool.max(config.top_k);

        let vector_hits = match (query.embedding, &self.vectors) {
            (Some(embedding), Some(store)) if config.vector_weight > 0.0 => {
                store.search(embedding, pool)?
            }
            _ => Vec::new(),
        };
        let lexical_hits = match query.text {
            Some(text) if config.lexical_weight > 0.0 => self.lexical.search(text, pool),
            _ => Vec::new(),
        };

        let vector_ids: Vec<EntityId> = vector_hits.iter().map(|hit| hit.id).collect();
        let lexical_ids: Vec<EntityId> = lexical_hits.iter().map(|hit| hit.id).collect();
        let mut fused = reciprocal_rank_fusion(
            &[
                RankedList {
                    weight: config.vector_weight,
                    ids: &vector_ids,
                },
                RankedList {
                    weight: config.lexical_weight,
                    ids: &lexical_ids,
                },
            ],
            config.rrf_k,
        )?;
        fused.truncate(config.top_k);

        let vector_by_id: HashMap<EntityId, ChannelHit> = vector_hits
            .iter()
            .enumerate()
            .map(|(position, hit)| {
                let channel = ChannelHit {
                    rank: position + 1,
                    score: f64::from(hit.similarity),
                };
                (hit.id, channel)
            })
            .collect();
        let lexical_by_id: HashMap<EntityId, ChannelHit> = lexical_hits
            .iter()
            .enumerate()
            .map(|(position, hit)| {
                let channel = ChannelHit {
                    rank: position + 1,
                    score: hit.score,
                };
                (hit.id, channel)
            })
            .collect();

        // One read view for hit metadata and expansion, so that the bundle is
        // a consistent cut of the graph.
        let view = self.graph.read();
        let mut hits = Vec::with_capacity(fused.len());
        let mut seeds = Vec::with_capacity(fused.len());
        for FusedHit { id, score } in fused {
            let index = view
                .node_index_by_id(id)
                .ok_or(GraphError::EntityIdNotFound(id))?;
            let node = view.node(index).ok_or(GraphError::EntityIdNotFound(id))?;
            seeds.push(index);
            hits.push(RetrievedEntity {
                id,
                name: node.canonical_name.clone(),
                entity_type: node.entity_type.clone(),
                fused_score: score,
                vector: vector_by_id.get(&id).copied(),
                lexical: lexical_by_id.get(&id).copied(),
            });
        }
        let context = view.get_k_hop_subgraph(&seeds, config.expansion_hops);

        Ok(RetrievalBundle { hits, context })
    }
}

/// The text an entity is indexed under: name, type, then attribute values in
/// key order.
fn entity_text(node: &EntityNode) -> String {
    let mut text = format!("{} {}", node.canonical_name, node.entity_type);
    for value in node.attributes.values() {
        text.push(' ');
        text.push_str(value);
    }
    text
}

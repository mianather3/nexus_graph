//! Differential retrieval evaluation.
//!
//! Runs the same queries through four retrieval configurations of one
//! [`HybridRetriever`] and scores each against ground truth:
//!
//! | System                | Channels          | Returned context                    |
//! |-----------------------|-------------------|-------------------------------------|
//! | [`System::VectorOnly`]  | dense k-NN        | the top-K hits                      |
//! | [`System::LexicalOnly`] | BM25              | the top-K hits                      |
//! | [`System::HybridRrf`]   | both, RRF-fused   | the top-K hits                      |
//! | [`System::HybridGraph`] | both, RRF-fused   | the top-K hits + k-hop expansion    |
//!
//! Graph expansion adds context around the hits; it does not rerank them. The
//! rank metrics (MRR, Recall@K) of the two hybrid rows are therefore
//! identical by construction, and what expansion buys shows up in the context
//! metrics, next to what it costs in context size.

pub mod metrics;

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use metrics::{coverage, edge_key, path_recovered, recall_at_k, reciprocal_rank, Metrics};

use crate::error::NexusError;
use crate::graph::KnowledgeGraph;
use crate::models::EntityId;
use crate::retrieval::{HybridQuery, HybridRetriever, RetrievalBundle, RetrievalConfig};
use crate::server::dto::{EntityInput, RelationInput};
use metrics::{MetricsAccumulator, QueryOutcome};

/// The graph a dataset is evaluated against.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalGraph {
    pub entities: Vec<EntityInput>,
    #[serde(default)]
    pub relations: Vec<RelationInput>,
}

/// One evaluation query with its ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalQuery {
    pub id: String,
    /// Text for the lexical channel.
    #[serde(default)]
    pub text: Option<String>,
    /// Embedding for the vector channel.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    /// Free-form label used to break results down (e.g. `multi_hop`).
    #[serde(default)]
    pub category: String,
    /// Names of the entities that answer the query. Must not be empty.
    pub relevant: Vec<String>,
    /// Names of the entities along the reasoning chain that justifies the
    /// answer, in order; consecutive entities must be connected in the graph.
    /// Empty for single-entity lookups.
    #[serde(default)]
    pub evidence_path: Vec<String>,
}

/// A self-contained evaluation dataset: graph, queries, ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalDataset {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub graph: EvalGraph,
    pub queries: Vec<EvalQuery>,
}

/// Evaluation settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Hits retrieved per query by every system.
    pub top_k: usize,
    /// Expansion depth of [`System::HybridGraph`].
    pub expansion_hops: usize,
    /// RRF smoothing constant of the hybrid systems.
    pub rrf_k: f64,
    /// RRF weight of the vector channel in the hybrid systems.
    pub vector_weight: f64,
    /// RRF weight of the lexical channel in the hybrid systems.
    pub lexical_weight: f64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        let retrieval = RetrievalConfig::default();
        Self {
            top_k: 10,
            expansion_hops: 1,
            rrf_k: retrieval.rrf_k,
            vector_weight: retrieval.vector_weight,
            lexical_weight: retrieval.lexical_weight,
        }
    }
}

/// A retrieval configuration under evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum System {
    VectorOnly,
    LexicalOnly,
    HybridRrf,
    HybridGraph,
}

impl System {
    pub const ALL: [Self; 4] = [
        Self::VectorOnly,
        Self::LexicalOnly,
        Self::HybridRrf,
        Self::HybridGraph,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::VectorOnly => "Vector only",
            Self::LexicalOnly => "BM25 only",
            Self::HybridRrf => "Hybrid RRF (no expansion)",
            Self::HybridGraph => "NexusGraph hybrid (RRF + graph)",
        }
    }

    fn retrieval_config(self, config: &EvalConfig) -> RetrievalConfig {
        let (vector_weight, lexical_weight, expansion_hops) = match self {
            Self::VectorOnly => (1.0, 0.0, 0),
            Self::LexicalOnly => (0.0, 1.0, 0),
            Self::HybridRrf => (config.vector_weight, config.lexical_weight, 0),
            Self::HybridGraph => (
                config.vector_weight,
                config.lexical_weight,
                config.expansion_hops,
            ),
        };
        RetrievalConfig {
            top_k: config.top_k,
            candidate_pool: config.top_k.max(RetrievalConfig::default().candidate_pool),
            rrf_k: config.rrf_k,
            vector_weight,
            lexical_weight,
            expansion_hops,
        }
    }
}

/// Scores of one system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemReport {
    pub system: System,
    pub label: String,
    pub overall: Metrics,
    pub by_category: BTreeMap<String, Metrics>,
}

/// Result of an evaluation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub dataset: String,
    pub entities: usize,
    pub relations: usize,
    pub queries: usize,
    pub config: EvalConfig,
    /// One entry per [`System`], in [`System::ALL`] order.
    pub systems: Vec<SystemReport>,
}

impl EvalReport {
    pub fn system(&self, system: System) -> Option<&SystemReport> {
        self.systems.iter().find(|report| report.system == system)
    }
}

/// A query with its ground truth resolved to entity ids.
struct ResolvedQuery<'a> {
    query: &'a EvalQuery,
    relevant: HashSet<EntityId>,
    /// Evidence path, or the relevant entities if the query has no path.
    evidence: Vec<EntityId>,
    /// The evidence path if it has at least one hop.
    path: Option<Vec<EntityId>>,
}

fn invalid(message: String) -> NexusError {
    NexusError::InvalidDataset(message)
}

fn build_graph(dataset: &EvalDataset) -> Result<KnowledgeGraph, NexusError> {
    let graph = KnowledgeGraph::new();
    for entity in &dataset.graph.entities {
        graph
            .add_entity(
                entity.name.clone(),
                entity.entity_type.clone(),
                entity.attributes.clone(),
                entity.embedding.clone(),
            )
            .map_err(|error| invalid(format!("entity `{}`: {error}", entity.name)))?;
    }
    for relation in &dataset.graph.relations {
        graph
            .add_relation_by_name(
                &relation.source,
                &relation.target,
                relation.relation.clone(),
                relation.weight,
                relation.metadata.clone(),
            )
            .map_err(|error| {
                invalid(format!(
                    "relation `{}` -[{}]-> `{}`: {error}",
                    relation.source, relation.relation, relation.target
                ))
            })?;
    }
    Ok(graph)
}

fn resolve<'a>(
    graph: &KnowledgeGraph,
    query: &'a EvalQuery,
) -> Result<ResolvedQuery<'a>, NexusError> {
    let view = graph.read();
    let lookup = |name: &str| {
        view.node_index_by_name(name)
            .and_then(|index| view.node(index).map(|node| (index, node.id)))
            .ok_or_else(|| {
                invalid(format!(
                    "query `{}` references unknown entity `{name}`",
                    query.id
                ))
            })
    };

    if query.text.is_none() && query.embedding.is_none() {
        return Err(invalid(format!(
            "query `{}` has neither text nor an embedding",
            query.id
        )));
    }
    if query.relevant.is_empty() {
        return Err(invalid(format!(
            "query `{}` has no relevant entities",
            query.id
        )));
    }
    let mut relevant = HashSet::new();
    for name in &query.relevant {
        relevant.insert(lookup(name)?.1);
    }

    let mut path = Vec::with_capacity(query.evidence_path.len());
    let mut previous = None;
    for name in &query.evidence_path {
        let (index, id) = lookup(name)?;
        if let Some((previous_index, previous_name)) = previous {
            let connected = !view.relations_between(previous_index, index).is_empty()
                || !view.relations_between(index, previous_index).is_empty();
            if !connected {
                return Err(invalid(format!(
                    "query `{}`: evidence path hop `{previous_name}` - `{name}` is not an edge of the graph",
                    query.id
                )));
            }
        }
        previous = Some((index, name.as_str()));
        path.push(id);
    }

    let evidence = if path.is_empty() {
        relevant.iter().copied().collect()
    } else {
        path.clone()
    };
    Ok(ResolvedQuery {
        query,
        relevant,
        evidence,
        path: (path.len() >= 2).then_some(path),
    })
}

fn score(bundle: &RetrievalBundle, resolved: &ResolvedQuery<'_>) -> QueryOutcome {
    let ranked: Vec<EntityId> = bundle.hits.iter().map(|hit| hit.id).collect();
    let context: HashSet<EntityId> = bundle
        .context
        .nodes
        .iter()
        .map(|node| node.entity.id)
        .collect();
    let context_edges: HashSet<(EntityId, EntityId)> = bundle
        .context
        .edges
        .iter()
        .map(|edge| edge_key(edge.source, edge.target))
        .collect();
    let answers: Vec<EntityId> = resolved.relevant.iter().copied().collect();

    QueryOutcome {
        reciprocal_rank: reciprocal_rank(&ranked, &resolved.relevant),
        recall_at_5: recall_at_k(&ranked, &resolved.relevant, 5),
        recall_at_10: recall_at_k(&ranked, &resolved.relevant, 10),
        answer_in_context: coverage(&context, &answers),
        evidence_coverage: coverage(&context, &resolved.evidence),
        path_recovered: resolved
            .path
            .as_deref()
            .map(|path| path_recovered(&context_edges, path)),
        context_entities: context.len(),
    }
}

/// Runs every [`System`] over every query of `dataset`.
///
/// Fails with [`NexusError::InvalidDataset`] if the graph cannot be built or
/// a query's ground truth does not match it (unknown entity names, evidence
/// hops that are not edges, queries without input or without an answer).
pub fn evaluate(dataset: &EvalDataset, config: &EvalConfig) -> Result<EvalReport, NexusError> {
    let graph = Arc::new(build_graph(dataset)?);
    let queries = dataset
        .queries
        .iter()
        .map(|query| resolve(&graph, query))
        .collect::<Result<Vec<_>, _>>()?;
    let retriever = HybridRetriever::from_graph(Arc::clone(&graph), RetrievalConfig::default())?;

    let mut systems = Vec::with_capacity(System::ALL.len());
    for system in System::ALL {
        let retrieval = system.retrieval_config(config);
        retrieval.validate()?;
        let mut overall = MetricsAccumulator::default();
        let mut by_category: BTreeMap<String, MetricsAccumulator> = BTreeMap::new();

        for resolved in &queries {
            // A single-channel system only sees its own half of the query.
            let query = HybridQuery {
                text: resolved
                    .query
                    .text
                    .as_deref()
                    .filter(|_| system != System::VectorOnly),
                embedding: resolved
                    .query
                    .embedding
                    .as_deref()
                    .filter(|_| system != System::LexicalOnly),
            };
            let bundle = if query.text.is_none() && query.embedding.is_none() {
                // The query has no input for this channel: a miss, not an error.
                RetrievalBundle::default()
            } else {
                retriever
                    .search_with(&query, &retrieval)
                    .map_err(|error| invalid(format!("query `{}`: {error}", resolved.query.id)))?
            };
            let outcome = score(&bundle, resolved);
            overall.add(&outcome);
            by_category
                .entry(resolved.query.category.clone())
                .or_default()
                .add(&outcome);
        }

        systems.push(SystemReport {
            system,
            label: system.label().to_owned(),
            overall: overall.finish(),
            by_category: by_category
                .into_iter()
                .map(|(category, accumulator)| (category, accumulator.finish()))
                .collect(),
        });
    }

    Ok(EvalReport {
        dataset: dataset.name.clone(),
        entities: graph.node_count(),
        relations: graph.edge_count(),
        queries: queries.len(),
        config: *config,
        systems,
    })
}

impl fmt::Display for EvalReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "dataset: {} ({} entities, {} relations, {} queries)",
            self.dataset, self.entities, self.relations, self.queries
        )?;
        writeln!(
            f,
            "config:  top_k={} expansion_hops={} rrf_k={} vector_weight={} lexical_weight={}",
            self.config.top_k,
            self.config.expansion_hops,
            self.config.rrf_k,
            self.config.vector_weight,
            self.config.lexical_weight
        )?;

        let mut categories: Vec<&str> = self
            .systems
            .iter()
            .flat_map(|system| system.by_category.keys().map(String::as_str))
            .collect();
        categories.sort_unstable();
        categories.dedup();

        self.write_table(f, "all queries", |system| Some(&system.overall))?;
        if categories.len() > 1 {
            for category in categories {
                let title = if category.is_empty() {
                    "(uncategorized)".to_owned()
                } else {
                    format!("category: {category}")
                };
                self.write_table(f, &title, |system| system.by_category.get(category))?;
            }
        }
        writeln!(f)?;
        writeln!(
            f,
            "MRR, R@5, R@10: rank metrics over the hits. ans@ctx: relevant entities anywhere in the"
        )?;
        writeln!(
            f,
            "returned context. coverage: evidence entities in the context. path: evidence chains whose"
        )?;
        write!(
            f,
            "every hop is a context edge (n/a without path queries). ctx: mean context entities."
        )
    }
}

impl EvalReport {
    fn write_table<'a>(
        &'a self,
        f: &mut fmt::Formatter<'_>,
        title: &str,
        select: impl Fn(&'a SystemReport) -> Option<&'a Metrics>,
    ) -> fmt::Result {
        let queries = self
            .systems
            .iter()
            .find_map(&select)
            .map_or(0, |metrics| metrics.queries);
        writeln!(f)?;
        writeln!(f, "{title} (n={queries})")?;
        writeln!(
            f,
            "{:<32} {:>6} {:>6} {:>6} {:>8} {:>9} {:>6} {:>6}",
            "system", "MRR", "R@5", "R@10", "ans@ctx", "coverage", "path", "ctx"
        )?;
        for system in &self.systems {
            let Some(metrics) = select(system) else {
                continue;
            };
            let path = if metrics.path_queries == 0 {
                "n/a".to_owned()
            } else {
                format!("{:.3}", metrics.path_recovery)
            };
            writeln!(
                f,
                "{:<32} {:>6.3} {:>6.3} {:>6.3} {:>8.3} {:>9.3} {:>6} {:>6.1}",
                system.label,
                metrics.mrr,
                metrics.recall_at_5,
                metrics.recall_at_10,
                metrics.answer_in_context,
                metrics.evidence_coverage,
                path,
                metrics.mean_context_entities
            )?;
        }
        Ok(())
    }
}

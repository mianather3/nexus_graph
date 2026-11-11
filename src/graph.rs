//! In-memory, thread-safe knowledge graph with secondary indices and
//! deterministic traversal.
//!
//! The graph is append-only: entities and relations can be added or updated
//! but not removed, so a [`NodeIndex`] stays valid for the lifetime of the
//! graph and across snapshot roundtrips.

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

use parking_lot::{RwLock, RwLockReadGuard};
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::models::{
    normalize_entity_name, Attributes, EntityId, EntityNode, EntityType, RelationEdge, RelationType,
};

/// Errors raised when a mutation would violate a graph invariant.
#[derive(Debug, Error, PartialEq)]
pub enum GraphError {
    #[error("entity name must not be empty")]
    EmptyName,
    #[error("an entity named `{0}` already exists")]
    DuplicateName(String),
    #[error("entity id {0} already exists")]
    DuplicateId(EntityId),
    #[error("entity id space exhausted")]
    IdSpaceExhausted,
    #[error("no node at index {0}")]
    NodeNotFound(usize),
    #[error("no entity with id {0}")]
    EntityIdNotFound(EntityId),
    #[error("no entity named `{0}`")]
    EntityNameNotFound(String),
    #[error("relation weight must be a finite value in [0.0, 1.0], got {0}")]
    InvalidWeight(f32),
    #[error("embedding must be non-empty and contain only finite values")]
    InvalidEmbedding,
    #[error("embedding dimension mismatch: graph uses {expected}, got {actual}")]
    EmbeddingDimensionMismatch { expected: usize, actual: usize },
}

/// An entity inside a [`Subgraph`], with its BFS distance from the seed set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubgraphNode {
    pub entity: EntityNode,
    pub depth: usize,
}

/// Result of a k-hop expansion.
///
/// `nodes` is sorted by `(depth, id)` and `edges` by `(source, target,
/// relation)`, so two expansions of the same graph compare equal.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Subgraph {
    pub nodes: Vec<SubgraphNode>,
    pub edges: Vec<RelationEdge>,
}

impl Subgraph {
    pub fn entity_ids(&self) -> BTreeSet<EntityId> {
        self.nodes.iter().map(|n| n.entity.id).collect()
    }

    pub fn contains(&self, id: EntityId) -> bool {
        self.nodes.iter().any(|n| n.entity.id == id)
    }
}

/// One traversed edge of a [`Path`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathHop {
    pub source: EntityId,
    pub source_name: String,
    pub target: EntityId,
    pub target_name: String,
    pub relation: RelationType,
    pub weight: f32,
}

/// A directed walk through the graph, as a sequence of contiguous hops.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Path {
    pub hops: Vec<PathHop>,
}

impl Path {
    /// Number of edges in the path.
    pub fn len(&self) -> usize {
        self.hops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hops.is_empty()
    }

    /// Product of the edge weights; `1.0` for an empty path.
    pub fn confidence(&self) -> f32 {
        self.hops.iter().map(|h| h.weight).product()
    }

    /// Canonical names of the visited entities, in order.
    pub fn entity_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .hops
            .first()
            .map(|h| h.source_name.as_str())
            .into_iter()
            .collect();
        names.extend(self.hops.iter().map(|h| h.target_name.as_str()));
        names
    }
}

/// Renders as `a -[DependsOn]-> b -[DependsOn]-> c`.
impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(first) = self.hops.first() {
            f.write_str(&first.source_name)?;
        }
        for hop in &self.hops {
            write!(f, " -[{}]-> {}", hop.relation, hop.target_name)?;
        }
        Ok(())
    }
}

/// Plain-data image of a graph: nodes in node-index order and edges in
/// edge-index order. Rebuilding from a snapshot reproduces identical indices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub nodes: Vec<EntityNode>,
    pub edges: Vec<RelationEdge>,
    pub next_id: u64,
}

/// The lock-protected state of a [`KnowledgeGraph`].
///
/// Obtained through [`KnowledgeGraph::read`]; exposes every read-only query so
/// that callers can run several queries against one consistent view.
#[derive(Debug, Default)]
pub struct GraphState {
    graph: DiGraph<EntityNode, RelationEdge>,
    /// Normalized canonical name -> node.
    name_index: HashMap<String, NodeIndex>,
    id_index: HashMap<EntityId, NodeIndex>,
    type_index: HashMap<EntityType, BTreeSet<NodeIndex>>,
    embedding_dim: Option<usize>,
    next_id: u64,
}

impl GraphState {
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Dimension shared by all embeddings, if any entity has one.
    pub fn embedding_dim(&self) -> Option<usize> {
        self.embedding_dim
    }

    pub fn node(&self, index: NodeIndex) -> Option<&EntityNode> {
        self.graph.node_weight(index)
    }

    /// Looks an entity up by canonical name (case- and whitespace-insensitive).
    pub fn node_index_by_name(&self, name: &str) -> Option<NodeIndex> {
        self.name_index.get(&normalize_entity_name(name)).copied()
    }

    pub fn node_index_by_id(&self, id: EntityId) -> Option<NodeIndex> {
        self.id_index.get(&id).copied()
    }

    /// All nodes of the given type, in ascending index order.
    pub fn nodes_by_type(&self, entity_type: &EntityType) -> Vec<NodeIndex> {
        self.type_index
            .get(entity_type)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// The edge `source -relation-> target`, if present.
    pub fn find_relation(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        relation: &RelationType,
    ) -> Option<&RelationEdge> {
        self.find_edge_index(source, target, relation)
            .and_then(|e| self.graph.edge_weight(e))
    }

    /// Every edge `source -> target`, sorted by relation type.
    pub fn relations_between(&self, source: NodeIndex, target: NodeIndex) -> Vec<&RelationEdge> {
        let mut edges: Vec<&RelationEdge> = self
            .graph
            .edges_connecting(source, target)
            .map(|e| e.weight())
            .collect();
        edges.sort_by(|a, b| a.relation.cmp(&b.relation));
        edges
    }

    /// Outgoing edges of `node`, optionally restricted to one relation type,
    /// in a deterministic order: by target id, then relation, then edge index.
    pub fn outgoing_edges(
        &self,
        node: NodeIndex,
        relation: Option<&RelationType>,
    ) -> Vec<(NodeIndex, EdgeIndex)> {
        let mut out: Vec<(&RelationEdge, NodeIndex, EdgeIndex)> = self
            .graph
            .edges_directed(node, Direction::Outgoing)
            .filter(|e| relation.map_or(true, |r| &e.weight().relation == r))
            .map(|e| (e.weight(), e.target(), e.id()))
            .collect();
        out.sort_by(|a, b| {
            a.0.target
                .cmp(&b.0.target)
                .then_with(|| a.0.relation.cmp(&b.0.relation))
                .then(a.2.cmp(&b.2))
        });
        out.into_iter().map(|(_, t, e)| (t, e)).collect()
    }

    /// Materializes a sequence of edge indices into a [`Path`].
    /// Indices that do not exist are skipped.
    pub fn path_from_edges(&self, edges: &[EdgeIndex]) -> Path {
        let name_of = |idx: NodeIndex| {
            self.graph
                .node_weight(idx)
                .map(|n| n.canonical_name.clone())
                .unwrap_or_default()
        };
        let hops = edges
            .iter()
            .filter_map(|&e| {
                let (s, t) = self.graph.edge_endpoints(e)?;
                let edge = self.graph.edge_weight(e)?;
                Some(PathHop {
                    source: edge.source,
                    source_name: name_of(s),
                    target: edge.target,
                    target_name: name_of(t),
                    relation: edge.relation.clone(),
                    weight: edge.weight,
                })
            })
            .collect();
        Path { hops }
    }

    /// Breadth-first expansion of up to `k` hops around `start_nodes`.
    ///
    /// Neighbors are followed in both edge directions (retrieval context cares
    /// about what points at an entity as much as what it points to). The result
    /// is the induced subgraph: every edge whose two endpoints were both
    /// reached is included. Start indices that do not exist are ignored.
    pub fn get_k_hop_subgraph(&self, start_nodes: &[NodeIndex], k: usize) -> Subgraph {
        let mut depth: HashMap<NodeIndex, usize> = HashMap::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        for &start in start_nodes {
            if self.graph.node_weight(start).is_some() && depth.insert(start, 0).is_none() {
                queue.push_back((start, 0));
            }
        }
        while let Some((node, d)) = queue.pop_front() {
            if d >= k {
                continue;
            }
            for neighbor in self.graph.neighbors_undirected(node) {
                if let Entry::Vacant(slot) = depth.entry(neighbor) {
                    slot.insert(d + 1);
                    queue.push_back((neighbor, d + 1));
                }
            }
        }

        let mut nodes: Vec<SubgraphNode> = depth
            .iter()
            .filter_map(|(&idx, &d)| {
                self.graph.node_weight(idx).map(|entity| SubgraphNode {
                    entity: entity.clone(),
                    depth: d,
                })
            })
            .collect();
        nodes.sort_by_key(|n| (n.depth, n.entity.id));

        let mut edges: Vec<RelationEdge> = depth
            .keys()
            .flat_map(|&idx| self.graph.edges_directed(idx, Direction::Outgoing))
            .filter(|e| depth.contains_key(&e.target()))
            .map(|e| e.weight().clone())
            .collect();
        edges.sort_by(|a, b| {
            a.source
                .cmp(&b.source)
                .then(a.target.cmp(&b.target))
                .then_with(|| a.relation.cmp(&b.relation))
        });

        Subgraph { nodes, edges }
    }

    /// Enumerates every directed simple path (no entity visited twice) from
    /// `source` to `target` with at most `max_depth` edges, across all relation
    /// types. Shortest paths come first; ties keep a deterministic order.
    ///
    /// If `source == target` the result is the set of simple cycles through it.
    /// The number of simple paths can grow exponentially with `max_depth`; use
    /// [`Self::find_connecting_paths_filtered`] to bound the work.
    pub fn find_connecting_paths(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        max_depth: usize,
    ) -> Vec<Path> {
        self.find_connecting_paths_filtered(source, target, max_depth, None, usize::MAX)
    }

    /// Like [`Self::find_connecting_paths`], optionally following only edges of
    /// one relation type, and stopping after `max_paths` paths have been found
    /// (in depth-first discovery order).
    pub fn find_connecting_paths_filtered(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        max_depth: usize,
        relation: Option<&RelationType>,
        max_paths: usize,
    ) -> Vec<Path> {
        let mut found: Vec<Path> = Vec::new();
        let endpoints_exist =
            self.graph.node_weight(source).is_some() && self.graph.node_weight(target).is_some();
        if max_depth == 0 || max_paths == 0 || !endpoints_exist {
            return found;
        }

        // Iterative DFS: `frames[i]` holds the unexplored out-edges of the i-th
        // node on the current path; `trail[i]` is the edge taken out of it.
        let mut on_path: HashSet<NodeIndex> = HashSet::from([source]);
        let mut trail: Vec<(NodeIndex, EdgeIndex)> = Vec::new();
        let mut frames = vec![self.outgoing_edges(source, relation).into_iter()];

        while let Some(frame) = frames.last_mut() {
            let Some((next, edge)) = frame.next() else {
                frames.pop();
                if let Some((node, _)) = trail.pop() {
                    on_path.remove(&node);
                }
                continue;
            };
            if next == target {
                let mut edges: Vec<EdgeIndex> = trail.iter().map(|&(_, e)| e).collect();
                edges.push(edge);
                found.push(self.path_from_edges(&edges));
                if found.len() >= max_paths {
                    break;
                }
            } else if trail.len() + 1 < max_depth && on_path.insert(next) {
                trail.push((next, edge));
                frames.push(self.outgoing_edges(next, relation).into_iter());
            }
        }

        found.sort_by_key(Path::len);
        found
    }

    /// The highest-confidence walk from `source` to `target` that uses only
    /// `relation` edges and at most `max_depth` of them, where confidence is
    /// the product of edge weights. Ties prefer fewer hops.
    ///
    /// Runs in `O(max_depth * E)` rather than enumerating paths.
    pub fn best_relation_path(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        relation: &RelationType,
        max_depth: usize,
    ) -> Option<Path> {
        // layers[h][v] = best (score, last edge, predecessor) over walks of
        // exactly h + 1 edges from `source` to v. Weights are <= 1, so no walk
        // with more edges than there are nodes can beat a shorter one.
        let rounds = max_depth.min(self.graph.node_count());
        let mut layers: Vec<HashMap<NodeIndex, (f32, EdgeIndex, NodeIndex)>> = Vec::new();
        let mut frontier: Vec<(NodeIndex, f32)> = vec![(source, 1.0)];
        let mut best: Option<(f32, usize)> = None;

        for round in 0..rounds {
            let mut layer: HashMap<NodeIndex, (f32, EdgeIndex, NodeIndex)> = HashMap::new();
            for &(node, score) in &frontier {
                for (next, edge) in self.outgoing_edges(node, Some(relation)) {
                    let Some(weight) = self.graph.edge_weight(edge).map(|e| e.weight) else {
                        continue;
                    };
                    let candidate = score * weight;
                    let improves = layer
                        .get(&next)
                        .map_or(true, |&(existing, _, _)| candidate > existing);
                    if improves {
                        layer.insert(next, (candidate, edge, node));
                    }
                }
            }
            if layer.is_empty() {
                break;
            }
            if let Some(&(score, _, _)) = layer.get(&target) {
                if best.map_or(true, |(b, _)| score > b) {
                    best = Some((score, round));
                }
            }
            frontier = layer.iter().map(|(&n, &(s, _, _))| (n, s)).collect();
            frontier.sort_by_key(|&(n, _)| n);
            layers.push(layer);
        }

        let (_, last_round) = best?;
        let mut edges: Vec<EdgeIndex> = Vec::with_capacity(last_round + 1);
        let mut cursor = target;
        for round in (0..=last_round).rev() {
            let &(_, edge, prev) = layers.get(round)?.get(&cursor)?;
            edges.push(edge);
            cursor = prev;
        }
        edges.reverse();
        Some(self.path_from_edges(&edges))
    }

    /// Captures the full topology as plain data.
    pub fn snapshot(&self) -> GraphSnapshot {
        GraphSnapshot {
            nodes: self.graph.node_weights().cloned().collect(),
            edges: self.graph.edge_weights().cloned().collect(),
            next_id: self.next_id,
        }
    }

    fn find_edge_index(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        relation: &RelationType,
    ) -> Option<EdgeIndex> {
        self.graph
            .edges_connecting(source, target)
            .find(|e| &e.weight().relation == relation)
            .map(|e| e.id())
    }

    fn insert_node(&mut self, node: EntityNode) -> Result<NodeIndex, GraphError> {
        let key = normalize_entity_name(&node.canonical_name);
        if key.is_empty() {
            return Err(GraphError::EmptyName);
        }
        if self.id_index.contains_key(&node.id) {
            return Err(GraphError::DuplicateId(node.id));
        }
        if self.name_index.contains_key(&key) {
            return Err(GraphError::DuplicateName(node.canonical_name));
        }
        let dim = node.embedding.as_ref().map(Vec::len);
        if let Some(embedding) = node.embedding.as_deref() {
            if embedding.is_empty() || embedding.iter().any(|v| !v.is_finite()) {
                return Err(GraphError::InvalidEmbedding);
            }
            if let Some(expected) = self.embedding_dim {
                if expected != embedding.len() {
                    return Err(GraphError::EmbeddingDimensionMismatch {
                        expected,
                        actual: embedding.len(),
                    });
                }
            }
        }
        let following_id = node
            .id
            .0
            .checked_add(1)
            .ok_or(GraphError::IdSpaceExhausted)?;

        let id = node.id;
        let entity_type = node.entity_type.clone();
        let index = self.graph.add_node(node);
        self.name_index.insert(key, index);
        self.id_index.insert(id, index);
        self.type_index
            .entry(entity_type)
            .or_default()
            .insert(index);
        self.embedding_dim = self.embedding_dim.or(dim);
        self.next_id = self.next_id.max(following_id);
        Ok(index)
    }

    fn add_relation(
        &mut self,
        source: NodeIndex,
        target: NodeIndex,
        relation: RelationType,
        weight: f32,
        metadata: Attributes,
    ) -> Result<EdgeIndex, GraphError> {
        let source_id = self
            .graph
            .node_weight(source)
            .map(|n| n.id)
            .ok_or(GraphError::NodeNotFound(source.index()))?;
        let target_id = self
            .graph
            .node_weight(target)
            .map(|n| n.id)
            .ok_or(GraphError::NodeNotFound(target.index()))?;
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err(GraphError::InvalidWeight(weight));
        }

        if let Some(existing) = self.find_edge_index(source, target, &relation) {
            if let Some(edge) = self.graph.edge_weight_mut(existing) {
                edge.weight = weight;
                edge.metadata = metadata;
            }
            return Ok(existing);
        }
        let edge = RelationEdge {
            source: source_id,
            target: target_id,
            relation,
            weight,
            metadata,
        };
        Ok(self.graph.add_edge(source, target, edge))
    }

    fn add_edge(&mut self, edge: RelationEdge) -> Result<EdgeIndex, GraphError> {
        let source = self
            .node_index_by_id(edge.source)
            .ok_or(GraphError::EntityIdNotFound(edge.source))?;
        let target = self
            .node_index_by_id(edge.target)
            .ok_or(GraphError::EntityIdNotFound(edge.target))?;
        self.add_relation(source, target, edge.relation, edge.weight, edge.metadata)
    }
}

/// Thread-safe hybrid knowledge graph.
///
/// A single [`parking_lot::RwLock`] guards the topology together with its
/// indices, so readers always observe them in sync. Every method takes the
/// lock for the duration of one call; use [`KnowledgeGraph::read`] to run
/// several queries against one consistent view.
#[derive(Debug, Default)]
pub struct KnowledgeGraph {
    state: RwLock<GraphState>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquires a shared read view.
    ///
    /// Do not call a mutating method of the same graph from the thread holding
    /// the guard: the lock is not reentrant and the call would deadlock.
    pub fn read(&self) -> RwLockReadGuard<'_, GraphState> {
        self.state.read()
    }

    /// Adds an entity under a freshly allocated id.
    pub fn add_entity(
        &self,
        canonical_name: impl Into<String>,
        entity_type: EntityType,
        attributes: Attributes,
        embedding: Option<Vec<f32>>,
    ) -> Result<NodeIndex, GraphError> {
        let mut state = self.state.write();
        let node = EntityNode {
            id: EntityId(state.next_id),
            canonical_name: canonical_name.into(),
            entity_type,
            attributes,
            embedding,
        };
        state.insert_node(node)
    }

    /// Adds an entity that already carries its id.
    pub fn insert_node(&self, node: EntityNode) -> Result<NodeIndex, GraphError> {
        self.state.write().insert_node(node)
    }

    /// Adds the edge `source -relation-> target`.
    ///
    /// At most one edge exists per `(source, target, relation)`: adding it
    /// again updates the weight and metadata of the existing edge and returns
    /// its index.
    pub fn add_relation(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        relation: RelationType,
        weight: f32,
        metadata: Attributes,
    ) -> Result<EdgeIndex, GraphError> {
        self.state
            .write()
            .add_relation(source, target, relation, weight, metadata)
    }

    /// Like [`Self::add_relation`], resolving the endpoints by canonical name.
    pub fn add_relation_by_name(
        &self,
        source: &str,
        target: &str,
        relation: RelationType,
        weight: f32,
        metadata: Attributes,
    ) -> Result<EdgeIndex, GraphError> {
        let mut state = self.state.write();
        let source_index = state
            .node_index_by_name(source)
            .ok_or_else(|| GraphError::EntityNameNotFound(source.to_owned()))?;
        let target_index = state
            .node_index_by_name(target)
            .ok_or_else(|| GraphError::EntityNameNotFound(target.to_owned()))?;
        state.add_relation(source_index, target_index, relation, weight, metadata)
    }

    /// Like [`Self::add_relation`], resolving the endpoints by entity id.
    pub fn add_edge(&self, edge: RelationEdge) -> Result<EdgeIndex, GraphError> {
        self.state.write().add_edge(edge)
    }

    pub fn node_count(&self) -> usize {
        self.read().node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.read().edge_count()
    }

    pub fn node(&self, index: NodeIndex) -> Option<EntityNode> {
        self.read().node(index).cloned()
    }

    /// See [`GraphState::node_index_by_name`].
    pub fn node_index_by_name(&self, name: &str) -> Option<NodeIndex> {
        self.read().node_index_by_name(name)
    }

    pub fn node_index_by_id(&self, id: EntityId) -> Option<NodeIndex> {
        self.read().node_index_by_id(id)
    }

    /// See [`GraphState::nodes_by_type`].
    pub fn nodes_by_type(&self, entity_type: &EntityType) -> Vec<NodeIndex> {
        self.read().nodes_by_type(entity_type)
    }

    /// See [`GraphState::find_relation`].
    pub fn find_relation(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        relation: &RelationType,
    ) -> Option<RelationEdge> {
        self.read().find_relation(source, target, relation).cloned()
    }

    /// See [`GraphState::get_k_hop_subgraph`].
    pub fn get_k_hop_subgraph(&self, start_nodes: &[NodeIndex], k: usize) -> Subgraph {
        self.read().get_k_hop_subgraph(start_nodes, k)
    }

    /// See [`GraphState::find_connecting_paths`].
    pub fn find_connecting_paths(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        max_depth: usize,
    ) -> Vec<Path> {
        self.read().find_connecting_paths(source, target, max_depth)
    }

    /// See [`GraphState::find_connecting_paths_filtered`].
    pub fn find_connecting_paths_filtered(
        &self,
        source: NodeIndex,
        target: NodeIndex,
        max_depth: usize,
        relation: Option<&RelationType>,
        max_paths: usize,
    ) -> Vec<Path> {
        self.read()
            .find_connecting_paths_filtered(source, target, max_depth, relation, max_paths)
    }

    /// Captures a consistent image of the graph under one read lock.
    pub fn snapshot(&self) -> GraphSnapshot {
        self.read().snapshot()
    }

    /// Rebuilds a graph, including all secondary indices, from a snapshot.
    /// Fails if the snapshot violates a graph invariant (duplicate names or
    /// ids, edges pointing at unknown entities, invalid weights).
    pub fn from_snapshot(snapshot: GraphSnapshot) -> Result<Self, GraphError> {
        let mut state = GraphState::default();
        for node in snapshot.nodes {
            state.insert_node(node)?;
        }
        for edge in snapshot.edges {
            state.add_edge(edge)?;
        }
        state.next_id = state.next_id.max(snapshot.next_id);
        Ok(Self {
            state: RwLock::new(state),
        })
    }
}

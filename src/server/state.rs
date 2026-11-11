//! Namespace registry and per-namespace engine state.
//!
//! Each namespace owns an isolated engine (knowledge graph, vector store and
//! BM25 index, bundled in a [`HybridRetriever`]) behind an async `RwLock`,
//! plus a bounded ingestion queue drained by a dedicated worker task.
//!
//! Concurrency model:
//!
//! * Queries take the engine lock shared; the ingestion worker takes it
//!   exclusive. Locks are awaited on the async side and the owned guard is
//!   moved into `spawn_blocking`, so no Tokio worker thread ever blocks on a
//!   lock, on disk, or on CPU-heavy engine work.
//! * All writes of a namespace flow through its queue and are applied by one
//!   worker in arrival order. A full queue is reported to the caller instead
//!   of buffering without bound.

use std::collections::HashMap;
use std::fmt;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tokio::task::spawn_blocking;

use crate::error::NexusError;
use crate::graph::{GraphError, KnowledgeGraph};
use crate::retrieval::HybridRetriever;
use crate::server::dto::{
    EntityInput, IngestReport, IngestStats, NamespaceStats, RejectedItem, RelationInput,
    SnapshotInfo, SnapshotStatus,
};
use crate::server::error::ApiError;
use crate::server::ServerConfig;
use crate::storage::{load_snapshot, save_snapshot, StorageError};
use crate::vector::checked_norm;

/// File extension of namespace snapshots inside the snapshot directory.
pub const SNAPSHOT_EXTENSION: &str = "ngsnap";

/// Seconds a client is told to wait after a `429` from a full queue.
const RETRY_AFTER_SECS: u64 = 1;

/// A validated namespace identifier: 1 to 64 characters from `[A-Za-z0-9_-]`,
/// not starting with `-`.
///
/// The id doubles as the snapshot file stem, so the restricted alphabet is
/// what rules out path traversal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NamespaceId(String);

impl NamespaceId {
    pub const MAX_LEN: usize = 64;

    pub fn parse(raw: &str) -> Result<Self, ApiError> {
        let valid_chars = raw
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
        if raw.is_empty() || raw.len() > Self::MAX_LEN || !valid_chars || raw.starts_with('-') {
            // Echo at most a bounded prefix of the offending value.
            return Err(ApiError::InvalidNamespace(raw.chars().take(80).collect()));
        }
        Ok(Self(raw.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NamespaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A unit of work for a namespace's ingestion worker.
#[derive(Debug)]
pub enum IngestJob {
    Entities(Vec<EntityInput>),
    Relations(Vec<RelationInput>),
}

impl IngestJob {
    fn len(&self) -> usize {
        match self {
            Self::Entities(items) => items.len(),
            Self::Relations(items) => items.len(),
        }
    }
}

struct Envelope {
    job: IngestJob,
    /// Present when the submitter waits for the outcome.
    reply: Option<oneshot::Sender<IngestReport>>,
}

/// Result of handing a job to the ingestion queue.
#[derive(Debug)]
pub enum Submission {
    /// The job was applied (the submitter waited).
    Completed(IngestReport),
    /// The job is queued; `pending_jobs` counts it.
    Queued { pending_jobs: u64 },
}

#[derive(Debug, Default)]
struct IngestCounters {
    accepted: AtomicU64,
    completed: AtomicU64,
    rejected: AtomicU64,
}

/// One tenant: an isolated engine, its ingestion queue and snapshot state.
pub struct Namespace {
    id: NamespaceId,
    engine: Arc<RwLock<HybridRetriever>>,
    graph: Arc<KnowledgeGraph>,
    ingest_tx: mpsc::Sender<Envelope>,
    queue_capacity: usize,
    counters: Arc<IngestCounters>,
    /// Serializes snapshots, so an older one can never overwrite a newer one.
    snapshot_gate: Mutex<()>,
    snapshot_status: parking_lot::Mutex<SnapshotStatus>,
}

impl fmt::Debug for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Namespace")
            .field("id", &self.id)
            .field("queue_capacity", &self.queue_capacity)
            .finish_non_exhaustive()
    }
}

impl Namespace {
    /// Wraps an engine and starts its ingestion worker. Must be called from
    /// within a Tokio runtime.
    fn start(id: NamespaceId, retriever: HybridRetriever, queue_capacity: usize) -> Arc<Self> {
        let queue_capacity = queue_capacity.max(1);
        let graph = Arc::clone(retriever.graph());
        let engine = Arc::new(RwLock::new(retriever));
        let counters = Arc::new(IngestCounters::default());
        let (ingest_tx, ingest_rx) = mpsc::channel(queue_capacity);
        // The worker holds no reference to the `Namespace`, so it stops as
        // soon as the namespace (and with it the only sender) is dropped.
        tokio::spawn(ingest_worker(
            Arc::clone(&engine),
            ingest_rx,
            Arc::clone(&counters),
        ));
        Arc::new(Self {
            id,
            engine,
            graph,
            ingest_tx,
            queue_capacity,
            counters,
            snapshot_gate: Mutex::new(()),
            snapshot_status: parking_lot::Mutex::new(SnapshotStatus::default()),
        })
    }

    pub fn id(&self) -> &NamespaceId {
        &self.id
    }

    /// The engine lock. Queries take it shared, the ingestion worker
    /// exclusive; holding a shared guard therefore pauses ingestion.
    pub fn engine(&self) -> &Arc<RwLock<HybridRetriever>> {
        &self.engine
    }

    /// The namespace's graph. It is internally synchronized, so it can be
    /// read without the engine lock when batch-level consistency with the
    /// retrieval indices is not needed.
    pub fn graph(&self) -> &Arc<KnowledgeGraph> {
        &self.graph
    }

    /// Hands a job to the bounded ingestion queue without ever waiting for
    /// queue space: a full queue yields [`ApiError::Backpressure`].
    ///
    /// With `wait`, resolves once the worker has applied the job.
    pub async fn submit(&self, job: IngestJob, wait: bool) -> Result<Submission, ApiError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let envelope = Envelope {
            job,
            reply: wait.then_some(reply_tx),
        };
        match self.ingest_tx.try_send(envelope) {
            Ok(()) => {
                self.counters.accepted.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Full(_)) => {
                self.counters.rejected.fetch_add(1, Ordering::Relaxed);
                return Err(ApiError::Backpressure {
                    queue_capacity: self.queue_capacity,
                    retry_after_secs: RETRY_AFTER_SECS,
                });
            }
            Err(TrySendError::Closed(_)) => {
                return Err(ApiError::Unavailable(
                    "the ingestion worker of this namespace has stopped".into(),
                ));
            }
        }
        if wait {
            let report = reply_rx
                .await
                .map_err(|_| ApiError::Internal("the ingestion worker dropped the job".into()))?;
            Ok(Submission::Completed(report))
        } else {
            Ok(Submission::Queued {
                pending_jobs: self.ingest_stats().pending_jobs,
            })
        }
    }

    pub fn ingest_stats(&self) -> IngestStats {
        let accepted = self.counters.accepted.load(Ordering::Relaxed);
        let completed = self.counters.completed.load(Ordering::Relaxed);
        IngestStats {
            queue_capacity: self.queue_capacity,
            queued_jobs: self
                .queue_capacity
                .saturating_sub(self.ingest_tx.capacity()),
            pending_jobs: accepted.saturating_sub(completed),
            accepted_jobs: accepted,
            completed_jobs: completed,
            rejected_jobs: self.counters.rejected.load(Ordering::Relaxed),
        }
    }

    pub fn snapshot_status(&self) -> SnapshotStatus {
        self.snapshot_status.lock().clone()
    }

    /// Marks a snapshot as requested; pair with [`Self::persist`].
    pub(crate) fn snapshot_requested(&self) {
        self.snapshot_status.lock().pending += 1;
    }

    /// Writes an atomic snapshot of the graph to `path` on the blocking pool.
    ///
    /// The graph is captured under its own read lock, so the snapshot is a
    /// consistent cut even while ingestion continues. Call
    /// [`Self::snapshot_requested`] first.
    pub(crate) async fn persist(&self, path: PathBuf) -> Result<SnapshotInfo, ApiError> {
        let _gate = self.snapshot_gate.lock().await;
        let graph = Arc::clone(&self.graph);
        let started = Instant::now();
        let written = spawn_blocking(move || -> Result<u64, StorageError> {
            let io_error = |source| StorageError::Io {
                path: path.clone(),
                source,
            };
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).map_err(io_error)?;
            }
            save_snapshot(&graph, &path)?;
            Ok(std::fs::metadata(&path).map_err(io_error)?.len())
        })
        .await;

        let outcome = match written {
            Ok(Ok(bytes)) => Ok(SnapshotInfo {
                completed_at_unix_ms: unix_millis(),
                duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                bytes,
            }),
            Ok(Err(error)) => Err(ApiError::Internal(format!("snapshot failed: {error}"))),
            Err(join_error) => Err(ApiError::from(join_error)),
        };
        let mut status = self.snapshot_status.lock();
        status.pending = status.pending.saturating_sub(1);
        match &outcome {
            Ok(info) => {
                status.last_success = Some(info.clone());
                status.last_error = None;
            }
            Err(error) => status.last_error = Some(error.to_string()),
        }
        outcome
    }

    /// Current sizes of the namespace's graph and indices.
    pub async fn stats(&self) -> NamespaceStats {
        let engine = self.engine.read().await;
        let index = engine.stats();
        NamespaceStats {
            namespace: self.id.to_string(),
            entities: self.graph.node_count(),
            relations: self.graph.edge_count(),
            vectors: index.vectors,
            vector_dim: index.vector_dim,
            terms: index.terms,
            documents: index.documents,
            ingest: self.ingest_stats(),
            snapshot: self.snapshot_status(),
        }
    }
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|elapsed| u64::try_from(elapsed.as_millis()).ok())
        .unwrap_or(0)
}

/// Drains a namespace's queue, applying one job at a time under the exclusive
/// engine lock, on the blocking pool.
async fn ingest_worker(
    engine: Arc<RwLock<HybridRetriever>>,
    mut jobs: mpsc::Receiver<Envelope>,
    counters: Arc<IngestCounters>,
) {
    while let Some(Envelope { job, reply }) = jobs.recv().await {
        let items = job.len();
        let mut guard = Arc::clone(&engine).write_owned().await;
        let applied = spawn_blocking(move || apply_job(&mut guard, job)).await;
        let report = applied.unwrap_or_else(|join_error| {
            tracing::error!(error = %join_error, "ingestion job crashed");
            IngestReport {
                ingested: 0,
                rejected: (0..items)
                    .map(|index| RejectedItem {
                        index,
                        detail: "the ingestion job crashed; this item may not have been applied"
                            .into(),
                    })
                    .collect(),
            }
        });
        counters.completed.fetch_add(1, Ordering::Relaxed);
        if let Some(reply) = reply {
            // The submitter may have gone away (client disconnect, timeout).
            let _ = reply.send(report);
        }
    }
}

fn apply_job(retriever: &mut HybridRetriever, job: IngestJob) -> IngestReport {
    let graph = Arc::clone(retriever.graph());
    let mut report = IngestReport::default();
    let mut record = |index: usize, outcome: Result<(), NexusError>| match outcome {
        Ok(()) => report.ingested += 1,
        Err(error) => report.rejected.push(RejectedItem {
            index,
            detail: error.to_string(),
        }),
    };
    match job {
        IngestJob::Entities(items) => {
            for (index, item) in items.into_iter().enumerate() {
                record(index, ingest_entity(retriever, &graph, item));
            }
        }
        IngestJob::Relations(items) => {
            for (index, item) in items.into_iter().enumerate() {
                let added = graph
                    .add_relation_by_name(
                        &item.source,
                        &item.target,
                        item.relation,
                        item.weight,
                        item.metadata,
                    )
                    .map(|_| ())
                    .map_err(NexusError::from);
                record(index, added);
            }
        }
    }
    report
}

fn ingest_entity(
    retriever: &mut HybridRetriever,
    graph: &KnowledgeGraph,
    item: EntityInput,
) -> Result<(), NexusError> {
    // The graph accepts any finite embedding but the vector index needs a
    // direction; refuse zero vectors before the entity exists anywhere.
    if let Some(embedding) = item.embedding.as_deref() {
        if !embedding.is_empty() {
            checked_norm(embedding)?;
        }
    }
    let index = graph.add_entity(item.name, item.entity_type, item.attributes, item.embedding)?;
    let id = graph
        .read()
        .node(index)
        .map(|node| node.id)
        .ok_or(GraphError::NodeNotFound(index.index()))?;
    retriever.index_entity(id)
}

/// Registry of namespaces, provisioning them on demand.
#[derive(Debug)]
pub struct EngineManager {
    config: ServerConfig,
    namespaces: RwLock<HashMap<NamespaceId, Arc<Namespace>>>,
    /// Serializes the slow path of provisioning, so two concurrent first
    /// requests for a namespace cannot both load or create it.
    provisioning: Mutex<()>,
}

impl EngineManager {
    pub fn new(config: ServerConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            namespaces: RwLock::new(HashMap::new()),
            provisioning: Mutex::new(()),
        })
    }

    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Where the snapshot of `id` lives.
    pub fn snapshot_path(&self, id: &NamespaceId) -> PathBuf {
        self.config
            .snapshot_dir
            .join(format!("{id}.{SNAPSHOT_EXTENSION}"))
    }

    /// The namespace `id`, loading it from its snapshot if it is not in
    /// memory. Fails with `404` if it exists neither in memory nor on disk.
    pub async fn get(&self, id: &NamespaceId) -> Result<Arc<Namespace>, ApiError> {
        self.provision(id, false)
            .await?
            .ok_or_else(|| ApiError::NamespaceNotFound(id.to_string()))
    }

    /// Like [`Self::get`], but creates an empty namespace if there is none.
    pub async fn get_or_create(&self, id: &NamespaceId) -> Result<Arc<Namespace>, ApiError> {
        self.provision(id, true)
            .await?
            .ok_or_else(|| ApiError::Internal(format!("namespace `{id}` could not be provisioned")))
    }

    /// Namespaces currently in memory, ordered by id.
    pub async fn loaded(&self) -> Vec<Arc<Namespace>> {
        let mut namespaces: Vec<Arc<Namespace>> =
            self.namespaces.read().await.values().cloned().collect();
        namespaces.sort_by(|a, b| a.id.cmp(&b.id));
        namespaces
    }

    async fn provision(
        &self,
        id: &NamespaceId,
        create: bool,
    ) -> Result<Option<Arc<Namespace>>, ApiError> {
        if let Some(namespace) = self.namespaces.read().await.get(id) {
            return Ok(Some(Arc::clone(namespace)));
        }

        let _provisioning = self.provisioning.lock().await;
        if let Some(namespace) = self.namespaces.read().await.get(id) {
            return Ok(Some(Arc::clone(namespace)));
        }

        let path = self.snapshot_path(id);
        let retrieval = self.config.retrieval;
        let restored = spawn_blocking(move || -> Result<Option<HybridRetriever>, NexusError> {
            match load_snapshot(&path) {
                Ok(graph) => Ok(Some(HybridRetriever::from_graph(
                    Arc::new(graph),
                    retrieval,
                )?)),
                Err(StorageError::Io { ref source, .. })
                    if source.kind() == ErrorKind::NotFound =>
                {
                    Ok(None)
                }
                Err(error) => Err(error.into()),
            }
        })
        .await?
        .map_err(|error| {
            ApiError::Internal(format!(
                "loading the snapshot of namespace `{id}` failed: {error}"
            ))
        })?;

        let retriever = match restored {
            Some(retriever) => retriever,
            None if create => HybridRetriever::new(Arc::new(KnowledgeGraph::new()), retrieval)?,
            None => return Ok(None),
        };
        let mut namespaces = self.namespaces.write().await;
        if namespaces.len() >= self.config.max_namespaces {
            return Err(ApiError::NamespaceLimit(self.config.max_namespaces));
        }
        let namespace = Namespace::start(id.clone(), retriever, self.config.ingest_queue_capacity);
        namespaces.insert(id.clone(), Arc::clone(&namespace));
        tracing::info!(namespace = %id, "namespace provisioned");
        Ok(Some(namespace))
    }
}

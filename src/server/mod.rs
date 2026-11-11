//! Multi-tenant HTTP service over the engine.
//!
//! | Method | Path                                           | Purpose                              |
//! |--------|------------------------------------------------|--------------------------------------|
//! | POST   | `/v1/namespaces/{namespace_id}/entities/batch` | Queue a batch of entities            |
//! | POST   | `/v1/namespaces/{namespace_id}/relations/batch`| Queue a batch of relations           |
//! | POST   | `/v1/namespaces/{namespace_id}/search/hybrid`  | Hybrid search with graph context     |
//! | POST   | `/v1/namespaces/{namespace_id}/verify`         | Verify claims, full report           |
//! | GET    | `/v1/namespaces/{namespace_id}/verify/stream`  | Verify claims, SSE (claims in query) |
//! | POST   | `/v1/namespaces/{namespace_id}/verify/stream`  | Verify claims, SSE (claims in body)  |
//! | POST   | `/v1/namespaces/{namespace_id}/snapshot`       | Persist the namespace                |
//! | GET    | `/v1/namespaces/{namespace_id}/stats`          | Namespace statistics                 |
//! | GET    | `/healthz`                                     | Liveness                             |
//! | GET    | `/metrics`                                     | Prometheus metrics                   |
//!
//! Namespaces are created by their first write and loaded from the snapshot
//! directory on first access. Every error is an RFC 7807 problem document.

pub mod dto;
pub mod error;
pub mod extract;
pub mod handlers;
pub mod middleware;
pub mod state;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::http::header::{ACCEPT, CONTENT_TYPE};
use axum::http::{HeaderValue, Method, Request, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::{DefaultOnResponse, TraceLayer};
use tracing::Level;

pub use error::{ApiError, ProblemDetails, PROBLEM_JSON};
pub use state::{EngineManager, IngestJob, Namespace, NamespaceId, Submission};

use crate::retrieval::RetrievalConfig;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Directory holding one `<namespace_id>.ngsnap` file per namespace.
    pub snapshot_dir: PathBuf,
    /// Jobs each namespace's ingestion queue holds before answering `429`.
    pub ingest_queue_capacity: usize,
    /// Largest accepted entity or relation batch.
    pub max_batch_items: usize,
    /// Largest accepted claim list.
    pub max_claims: usize,
    /// Largest accepted request body, in bytes.
    pub max_body_bytes: usize,
    /// Time allowed to produce a response (for SSE: to start the stream).
    pub request_timeout: Duration,
    /// Namespaces held in memory at once.
    pub max_namespaces: usize,
    /// Upper bound for `top_k` and `candidate_pool` in search requests.
    pub max_top_k: usize,
    /// Upper bound for `expansion_hops` in search requests.
    pub max_expansion_hops: usize,
    /// Upper bound for `max_hops` in verification requests.
    pub max_verify_hops: usize,
    /// Events buffered per SSE stream before verification pauses.
    pub sse_buffer: usize,
    /// Origins allowed to call the API from a browser. Empty: none.
    pub cors_allowed_origins: Vec<String>,
    /// Search defaults; requests may override individual fields.
    pub retrieval: RetrievalConfig,
}

impl ServerConfig {
    pub fn new(snapshot_dir: impl Into<PathBuf>) -> Self {
        Self {
            snapshot_dir: snapshot_dir.into(),
            ingest_queue_capacity: 64,
            max_batch_items: 10_000,
            max_claims: 10_000,
            max_body_bytes: 32 * 1024 * 1024,
            request_timeout: Duration::from_secs(30),
            max_namespaces: 1024,
            max_top_k: 1000,
            max_expansion_hops: 4,
            max_verify_hops: 8,
            sse_buffer: 32,
            cors_allowed_origins: Vec::new(),
            retrieval: RetrievalConfig::default(),
        }
    }
}

/// Builds the application router around a namespace registry.
pub fn router(manager: Arc<EngineManager>) -> Router {
    let config = manager.config().clone();

    let origins: Vec<HeaderValue> = config
        .cors_allowed_origins
        .iter()
        .filter_map(|origin| HeaderValue::from_str(origin).ok())
        .collect();
    let cors = CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([CONTENT_TYPE, ACCEPT, middleware::REQUEST_ID_HEADER]);

    let trace = TraceLayer::new_for_http()
        .make_span_with(|request: &Request<_>| {
            let request_id = request
                .headers()
                .get(&middleware::REQUEST_ID_HEADER)
                .and_then(|value| value.to_str().ok())
                .unwrap_or("-");
            tracing::info_span!(
                "http",
                method = %request.method(),
                path = %request.uri().path(),
                request_id = %request_id,
            )
        })
        .on_response(DefaultOnResponse::new().level(Level::INFO));

    // Outermost first. The problem normalizer sits outside the timeout and
    // body limit so that the statuses they produce become problem documents.
    let layers = ServiceBuilder::new()
        .layer(axum::middleware::from_fn(middleware::request_id))
        .layer(trace)
        .layer(cors)
        .layer(axum::middleware::from_fn(middleware::problem_details))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            config.request_timeout,
        ))
        .layer(DefaultBodyLimit::max(config.max_body_bytes));

    let namespace_routes = Router::new()
        .route("/entities/batch", post(handlers::ingest::entities))
        .route("/relations/batch", post(handlers::ingest::relations))
        .route("/search/hybrid", post(handlers::search::hybrid))
        .route("/verify", post(handlers::verify::verify))
        .route(
            "/verify/stream",
            get(handlers::verify::stream_get).post(handlers::verify::stream_post),
        )
        .route("/snapshot", post(handlers::snapshot::trigger))
        .route("/stats", get(handlers::ops::stats));

    Router::new()
        .route("/healthz", get(handlers::ops::healthz))
        .route("/metrics", get(handlers::ops::metrics))
        .nest("/v1/namespaces/{namespace_id}", namespace_routes)
        .fallback(handlers::ops::not_found)
        .method_not_allowed_fallback(handlers::ops::method_not_allowed)
        .layer(layers)
        .with_state(manager)
}

/// Binds `addr` and serves until Ctrl-C, then stops accepting connections and
/// lets in-flight requests finish.
pub async fn serve(addr: SocketAddr, config: ServerConfig) -> std::io::Result<()> {
    tokio::fs::create_dir_all(&config.snapshot_dir).await?;
    let manager = EngineManager::new(config);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        address = %listener.local_addr()?,
        snapshot_dir = %manager.config().snapshot_dir.display(),
        "nexus_graph listening"
    );
    axum::serve(listener, router(manager))
        .with_graceful_shutdown(async {
            // If the signal handler cannot be installed, keep serving.
            if tokio::signal::ctrl_c().await.is_err() {
                std::future::pending::<()>().await;
            }
            tracing::info!("shutdown signal received");
        })
        .await
}

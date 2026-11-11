//! Operational endpoints: liveness, statistics, metrics, fallbacks.

use std::fmt::Write as _;

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::Uri;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::server::dto::NamespaceStats;
use crate::server::error::ApiError;
use crate::server::handlers::AppState;
use crate::server::state::NamespaceId;
use crate::vector::Kernel;

/// `GET /healthz`: the process is up and serving requests.
#[allow(clippy::unused_async)] // axum handlers must be async
pub async fn healthz() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// `GET /v1/namespaces/{namespace_id}/stats`
pub async fn stats(
    State(manager): State<AppState>,
    namespace: NamespaceId,
) -> Result<Json<NamespaceStats>, ApiError> {
    let namespace = manager.get(&namespace).await?;
    Ok(Json(namespace.stats().await))
}

/// Metric name suffix, Prometheus type and accessor.
type Metric = (&'static str, &'static str, fn(&NamespaceStats) -> u64);

/// `GET /metrics`: Prometheus text exposition for the namespaces currently
/// in memory.
pub async fn metrics(State(manager): State<AppState>) -> Response {
    let namespaces = manager.loaded().await;
    let mut stats = Vec::with_capacity(namespaces.len());
    for namespace in &namespaces {
        stats.push(namespace.stats().await);
    }

    let mut body = String::new();
    let _ = writeln!(
        body,
        "# TYPE nexus_graph_namespaces gauge\nnexus_graph_namespaces {}",
        stats.len()
    );
    let _ = writeln!(
        body,
        "# TYPE nexus_graph_simd_kernel_info gauge\nnexus_graph_simd_kernel_info{{kernel=\"{}\"}} 1",
        Kernel::active()
    );
    let metrics: [Metric; 9] = [
        ("entities", "gauge", |s| s.entities as u64),
        ("relations", "gauge", |s| s.relations as u64),
        ("vectors", "gauge", |s| s.vectors as u64),
        ("terms", "gauge", |s| s.terms as u64),
        ("ingest_queue_capacity", "gauge", |s| {
            s.ingest.queue_capacity as u64
        }),
        ("ingest_pending_jobs", "gauge", |s| s.ingest.pending_jobs),
        ("ingest_accepted_jobs_total", "counter", |s| {
            s.ingest.accepted_jobs
        }),
        ("ingest_completed_jobs_total", "counter", |s| {
            s.ingest.completed_jobs
        }),
        ("ingest_rejected_jobs_total", "counter", |s| {
            s.ingest.rejected_jobs
        }),
    ];
    for (name, kind, value) in metrics {
        let _ = writeln!(body, "# TYPE nexus_graph_{name} {kind}");
        for namespace in &stats {
            // Namespace ids are restricted to [A-Za-z0-9_-]: no escaping needed.
            let _ = writeln!(
                body,
                "nexus_graph_{name}{{namespace=\"{}\"}} {}",
                namespace.namespace,
                value(namespace)
            );
        }
    }
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
        .into_response()
}

/// Any path that matches no route.
#[allow(clippy::unused_async)]
pub async fn not_found(uri: Uri) -> ApiError {
    ApiError::NotFound(format!("no route for `{}`", uri.path()))
}

/// A known path called with an unsupported method.
#[allow(clippy::unused_async)]
pub async fn method_not_allowed() -> ApiError {
    ApiError::Status(axum::http::StatusCode::METHOD_NOT_ALLOWED)
}

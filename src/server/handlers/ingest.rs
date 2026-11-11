//! Batch ingestion. Batches are not applied by the request handler: they are
//! handed to the namespace's bounded queue, and a full queue is answered with
//! `429 Too Many Requests`.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::server::dto::{EntityBatch, IngestOptions, IngestReport, QueuedResponse, RelationBatch};
use crate::server::error::ApiError;
use crate::server::extract::{ApiJson, ApiQuery};
use crate::server::handlers::AppState;
use crate::server::state::{IngestJob, NamespaceId, Submission};

/// `POST /v1/namespaces/{namespace_id}/entities/batch`
pub async fn entities(
    State(manager): State<AppState>,
    namespace: NamespaceId,
    ApiQuery(options): ApiQuery<IngestOptions>,
    ApiJson(batch): ApiJson<EntityBatch>,
) -> Result<Response, ApiError> {
    let count = batch.entities.len();
    submit(
        &manager,
        &namespace,
        IngestJob::Entities(batch.entities),
        count,
        options,
    )
    .await
}

/// `POST /v1/namespaces/{namespace_id}/relations/batch`
pub async fn relations(
    State(manager): State<AppState>,
    namespace: NamespaceId,
    ApiQuery(options): ApiQuery<IngestOptions>,
    ApiJson(batch): ApiJson<RelationBatch>,
) -> Result<Response, ApiError> {
    let count = batch.relations.len();
    submit(
        &manager,
        &namespace,
        IngestJob::Relations(batch.relations),
        count,
        options,
    )
    .await
}

async fn submit(
    manager: &AppState,
    namespace: &NamespaceId,
    job: IngestJob,
    items: usize,
    options: IngestOptions,
) -> Result<Response, ApiError> {
    let limit = manager.config().max_batch_items;
    if items > limit {
        return Err(ApiError::PayloadTooLarge(format!(
            "batch has {items} items, the limit is {limit}"
        )));
    }
    if items == 0 {
        // Nothing to write: do not provision a namespace for it.
        return Ok(Json(IngestReport::default()).into_response());
    }

    // A write provisions the namespace if it does not exist yet.
    let namespace = manager.get_or_create(namespace).await?;
    match namespace.submit(job, options.wait).await? {
        Submission::Completed(report) => Ok(Json(report).into_response()),
        Submission::Queued { pending_jobs } => Ok((
            StatusCode::ACCEPTED,
            Json(QueuedResponse {
                status: "queued".into(),
                pending_jobs,
            }),
        )
            .into_response()),
    }
}

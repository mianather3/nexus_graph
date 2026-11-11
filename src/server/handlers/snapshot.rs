//! Snapshot persistence.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::server::dto::SnapshotOptions;
use crate::server::error::ApiError;
use crate::server::extract::ApiQuery;
use crate::server::handlers::AppState;
use crate::server::state::NamespaceId;

/// `POST /v1/namespaces/{namespace_id}/snapshot`
///
/// Persists the namespace's graph to `<snapshot-dir>/<namespace_id>.ngsnap`
/// with an atomic replace. By default the snapshot runs in the background and
/// the response is `202 Accepted`; its outcome is visible in `.../stats`.
/// With `?wait=true` the response is sent once the file is on disk.
pub async fn trigger(
    State(manager): State<AppState>,
    namespace_id: NamespaceId,
    ApiQuery(options): ApiQuery<SnapshotOptions>,
) -> Result<Response, ApiError> {
    let namespace = manager.get(&namespace_id).await?;
    let path = manager.snapshot_path(&namespace_id);
    namespace.snapshot_requested();

    if options.wait {
        let info = namespace.persist(path).await?;
        return Ok(Json(info).into_response());
    }
    tokio::spawn(async move {
        if let Err(error) = namespace.persist(path).await {
            tracing::error!(namespace = %namespace.id(), %error, "background snapshot failed");
        }
    });
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({ "status": "scheduled", "namespace": namespace_id.as_str() })),
    )
        .into_response())
}

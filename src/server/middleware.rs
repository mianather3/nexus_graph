//! Cross-cutting HTTP middleware: request ids and problem normalization.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::Request;
use axum::http::header::{HeaderName, HeaderValue, CONTENT_TYPE};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::server::error::{ApiError, PROBLEM_JSON};

/// Header carrying the request id, on requests and responses.
pub const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);
static PROCESS_TAG: OnceLock<u64> = OnceLock::new();

/// A client-supplied id is reused if it is short and plain; anything else is
/// replaced, so that ids are safe to put in logs.
fn acceptable_request_id(value: &HeaderValue) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .as_bytes()
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

fn generate_request_id() -> HeaderValue {
    // Distinguishes ids across restarts; uniqueness within a process comes
    // from the counter.
    let tag = *PROCESS_TAG.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs())
    });
    let sequence = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    HeaderValue::from_str(&format!("{tag:x}-{sequence:x}"))
        .unwrap_or_else(|_| HeaderValue::from_static("unknown"))
}

/// Ensures every request has an `x-request-id` (reusing the client's when
/// acceptable) and echoes it on the response, so that a client-visible
/// failure can be matched to its server-side trace.
pub async fn request_id(mut request: Request, next: Next) -> Response {
    let id = request
        .headers()
        .get(&REQUEST_ID_HEADER)
        .filter(|value| acceptable_request_id(value))
        .cloned()
        .unwrap_or_else(generate_request_id);
    request.headers_mut().insert(REQUEST_ID_HEADER, id.clone());

    let mut response = next.run(request).await;
    response.headers_mut().insert(REQUEST_ID_HEADER, id);
    response
}

/// Rewrites any error response that is not already a problem document
/// (timeouts, body-size limits and other statuses produced by lower layers)
/// into one, so that clients can rely on `application/problem+json` for every
/// 4xx and 5xx.
pub async fn problem_details(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    let status = response.status();
    let is_problem = response
        .headers()
        .get(CONTENT_TYPE)
        .is_some_and(|value| value.as_bytes().starts_with(PROBLEM_JSON.as_bytes()));
    if (status.is_client_error() || status.is_server_error()) && !is_problem {
        return ApiError::Status(status).into_response();
    }
    response
}

//! RFC 7807 "Problem Details" error mapping.
//!
//! Every failure a client can observe is an [`ApiError`], rendered as
//! `application/problem+json` with the standard `type`, `title`, `status` and
//! `detail` members plus problem-specific extension members.

use axum::extract::rejection::{JsonRejection, PathRejection, QueryRejection};
use axum::http::header::{HeaderValue, CONTENT_TYPE, RETRY_AFTER};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::error::NexusError;
use crate::graph::GraphError;

/// Media type of every error body.
pub const PROBLEM_JSON: &str = "application/problem+json";

/// Body sent if a problem document itself cannot be serialized.
const FALLBACK_BODY: &str = r#"{"type":"about:blank","title":"Internal Server Error","status":500,"detail":"failed to render the error response"}"#;

/// An RFC 7807 problem document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProblemDetails {
    /// URI reference identifying the problem type; `about:blank` when the
    /// HTTP status code says it all.
    #[serde(rename = "type")]
    pub problem_type: String,
    pub title: String,
    pub status: u16,
    pub detail: String,
    /// Problem-specific extension members.
    #[serde(flatten)]
    pub extensions: Map<String, Value>,
}

/// Every failure the HTTP layer can report.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),
    #[error(
        "invalid namespace id `{0}`: use 1 to 64 characters from [A-Za-z0-9_-], not starting with `-`"
    )]
    InvalidNamespace(String),
    #[error("{0}")]
    NotFound(String),
    #[error("namespace `{0}` does not exist")]
    NamespaceNotFound(String),
    #[error("{0}")]
    PayloadTooLarge(String),
    #[error("{0}")]
    UnsupportedMediaType(String),
    #[error("{0}")]
    Unprocessable(String),
    /// The namespace's bounded ingestion queue is full.
    #[error("the ingestion queue of this namespace is full ({queue_capacity} jobs); retry later")]
    Backpressure {
        queue_capacity: usize,
        retry_after_secs: u64,
    },
    #[error("the server holds its maximum of {0} namespaces")]
    NamespaceLimit(usize),
    #[error("{0}")]
    Unavailable(String),
    #[error("{0}")]
    Internal(String),
    /// A bare status produced by a lower layer (timeout, body limit, ...).
    #[error("{}", .0.canonical_reason().unwrap_or("request failed"))]
    Status(StatusCode),
}

impl ApiError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) | Self::InvalidNamespace(_) => StatusCode::BAD_REQUEST,
            Self::NotFound(_) | Self::NamespaceNotFound(_) => StatusCode::NOT_FOUND,
            Self::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UnsupportedMediaType(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::Unprocessable(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Backpressure { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::NamespaceLimit(_) => StatusCode::INSUFFICIENT_STORAGE,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Status(status) => *status,
        }
    }

    /// Problem type URI, for the problems that have their own semantics
    /// beyond the status code.
    fn problem_type(&self) -> &'static str {
        match self {
            Self::InvalidNamespace(_) => "urn:nexus-graph:problem:invalid-namespace",
            Self::NamespaceNotFound(_) => "urn:nexus-graph:problem:namespace-not-found",
            Self::Backpressure { .. } => "urn:nexus-graph:problem:ingest-backpressure",
            Self::NamespaceLimit(_) => "urn:nexus-graph:problem:namespace-limit",
            _ => "about:blank",
        }
    }

    fn title(&self) -> &'static str {
        match self {
            Self::InvalidNamespace(_) => "Invalid Namespace Id",
            Self::NamespaceNotFound(_) => "Namespace Not Found",
            Self::Backpressure { .. } => "Ingestion Queue Full",
            Self::NamespaceLimit(_) => "Namespace Limit Reached",
            _ => self.status().canonical_reason().unwrap_or("Error"),
        }
    }

    /// The problem document for this error.
    pub fn problem(&self) -> ProblemDetails {
        let mut extensions = Map::new();
        if let Self::Backpressure {
            queue_capacity,
            retry_after_secs,
        } = self
        {
            extensions.insert("queue_capacity".into(), Value::from(*queue_capacity));
            extensions.insert("retry_after_seconds".into(), Value::from(*retry_after_secs));
        }
        ProblemDetails {
            problem_type: self.problem_type().to_owned(),
            title: self.title().to_owned(),
            status: self.status().as_u16(),
            detail: self.to_string(),
            extensions,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        if status.is_server_error() {
            tracing::error!(%status, error = %self, "request failed");
        }
        let (status, body) = match serde_json::to_vec(&self.problem()) {
            Ok(body) => (status, body),
            Err(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                FALLBACK_BODY.as_bytes().to_vec(),
            ),
        };
        let mut response = (status, body).into_response();
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static(PROBLEM_JSON));
        if let Self::Backpressure {
            retry_after_secs, ..
        } = self
        {
            response
                .headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from(retry_after_secs));
        }
        response
    }
}

impl From<NexusError> for ApiError {
    fn from(error: NexusError) -> Self {
        match error {
            // Caused by the request's content: the client can fix these.
            NexusError::ZeroDimension
            | NexusError::DimensionMismatch { .. }
            | NexusError::NonFiniteVector
            | NexusError::ZeroMagnitude
            | NexusError::InvalidConfig(_)
            | NexusError::EmptyQuery
            | NexusError::DuplicateDocument(_)
            | NexusError::Graph(_) => Self::Unprocessable(error.to_string()),
            NexusError::Storage(_)
            | NexusError::NumericOverflow
            | NexusError::KernelUnavailable(_)
            | NexusError::BufferLengthMismatch { .. }
            | NexusError::CapacityExceeded => Self::Internal(error.to_string()),
        }
    }
}

impl From<GraphError> for ApiError {
    fn from(error: GraphError) -> Self {
        Self::Unprocessable(error.to_string())
    }
}

impl From<tokio::task::JoinError> for ApiError {
    fn from(error: tokio::task::JoinError) -> Self {
        Self::Internal(format!("background task failed: {error}"))
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        from_rejection(rejection.status(), rejection.body_text())
    }
}

impl From<QueryRejection> for ApiError {
    fn from(rejection: QueryRejection) -> Self {
        from_rejection(rejection.status(), rejection.body_text())
    }
}

impl From<PathRejection> for ApiError {
    fn from(rejection: PathRejection) -> Self {
        from_rejection(rejection.status(), rejection.body_text())
    }
}

fn from_rejection(status: StatusCode, detail: String) -> ApiError {
    match status {
        StatusCode::PAYLOAD_TOO_LARGE => ApiError::PayloadTooLarge(detail),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => ApiError::UnsupportedMediaType(detail),
        StatusCode::UNPROCESSABLE_ENTITY => ApiError::Unprocessable(detail),
        status if status.is_server_error() => ApiError::Internal(detail),
        // Syntax errors, wrong content type, malformed query strings.
        _ => ApiError::BadRequest(detail),
    }
}

//! Claim verification: one-shot and as a Server-Sent Events stream.
//!
//! The stream emits one event per claim, in submission order, as soon as that
//! claim has been checked:
//!
//! ```text
//! event: claim_verified          event: claim_rejected
//! data: {VerifiedClaim JSON}     data: {BrokenClaim JSON}
//! ```
//!
//! followed by exactly one terminal event:
//!
//! ```text
//! event: complete
//! data: {"total_claims":3,"verified":2,"rejected":1,"confidence_score":0.57}
//! ```
//!
//! If verification itself fails mid-stream, the terminal event is `error`
//! with a problem document as data instead.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::task::spawn_blocking;
use tokio_stream::wrappers::ReceiverStream;

use crate::graph::KnowledgeGraph;
use crate::models::GraphClaim;
use crate::server::dto::{StreamSummary, VerifyRequest, VerifyStreamParams};
use crate::server::error::ApiError;
use crate::server::extract::{ApiJson, ApiQuery};
use crate::server::handlers::AppState;
use crate::server::state::NamespaceId;
use crate::server::ServerConfig;
use crate::verifier::{verify_claims, VerificationReport, VerifierConfig};

pub const EVENT_VERIFIED: &str = "claim_verified";
pub const EVENT_REJECTED: &str = "claim_rejected";
pub const EVENT_COMPLETE: &str = "complete";
pub const EVENT_ERROR: &str = "error";

fn verifier_config(
    server: &ServerConfig,
    claims: usize,
    max_hops: Option<usize>,
) -> Result<VerifierConfig, ApiError> {
    if claims > server.max_claims {
        return Err(ApiError::PayloadTooLarge(format!(
            "request has {claims} claims, the limit is {}",
            server.max_claims
        )));
    }
    let max_hops = max_hops.unwrap_or(VerifierConfig::default().max_hops);
    if max_hops > server.max_verify_hops {
        return Err(ApiError::Unprocessable(format!(
            "max_hops must not exceed {}",
            server.max_verify_hops
        )));
    }
    Ok(VerifierConfig { max_hops })
}

/// `POST /v1/namespaces/{namespace_id}/verify`
pub async fn verify(
    State(manager): State<AppState>,
    namespace: NamespaceId,
    ApiJson(request): ApiJson<VerifyRequest>,
) -> Result<Json<VerificationReport>, ApiError> {
    let config = verifier_config(manager.config(), request.claims.len(), request.max_hops)?;
    let namespace = manager.get(&namespace).await?;
    // The shared engine lock keeps ingestion out for the duration of the
    // batch, so the whole report reflects one state of the graph.
    let engine = Arc::clone(namespace.engine()).read_owned().await;
    let report =
        spawn_blocking(move || verify_claims(engine.graph(), &request.claims, config)).await?;
    Ok(Json(report))
}

/// `GET /v1/namespaces/{namespace_id}/verify/stream?claims=<json>&max_hops=<n>`
pub async fn stream_get(
    State(manager): State<AppState>,
    namespace: NamespaceId,
    ApiQuery(params): ApiQuery<VerifyStreamParams>,
) -> Result<Response, ApiError> {
    let claims: Vec<GraphClaim> = serde_json::from_str(&params.claims).map_err(|error| {
        ApiError::BadRequest(format!(
            "query parameter `claims` must be a JSON array of claims: {error}"
        ))
    })?;
    stream(&manager, &namespace, claims, params.max_hops).await
}

/// `POST /v1/namespaces/{namespace_id}/verify/stream`, for claim lists too
/// large for a query string.
pub async fn stream_post(
    State(manager): State<AppState>,
    namespace: NamespaceId,
    ApiJson(request): ApiJson<VerifyRequest>,
) -> Result<Response, ApiError> {
    stream(&manager, &namespace, request.claims, request.max_hops).await
}

async fn stream(
    manager: &AppState,
    namespace: &NamespaceId,
    claims: Vec<GraphClaim>,
    max_hops: Option<usize>,
) -> Result<Response, ApiError> {
    let config = verifier_config(manager.config(), claims.len(), max_hops)?;
    let namespace = manager.get(namespace).await?;
    let graph = Arc::clone(namespace.graph());

    // Bounded: a slow consumer pauses verification instead of piling events
    // up in memory. No engine lock is held across the stream, so a slow
    // consumer cannot stall ingestion either; each claim is checked against
    // the graph as it is at that moment.
    let (events, receiver) = mpsc::channel(manager.config().sse_buffer.max(1));
    tokio::spawn(produce_events(graph, claims, config, events));

    Ok(Sse::new(ReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn produce_events(
    graph: Arc<KnowledgeGraph>,
    claims: Vec<GraphClaim>,
    config: VerifierConfig,
    events: mpsc::Sender<Result<Event, Infallible>>,
) {
    let total_claims = claims.len();
    let mut verified = 0_usize;
    let mut rejected = 0_usize;
    let mut confidence_sum = 0.0_f32;

    for (claim_index, claim) in claims.into_iter().enumerate() {
        let graph = Arc::clone(&graph);
        let checked =
            spawn_blocking(move || verify_claims(&graph, std::slice::from_ref(&claim), config))
                .await;
        let event = match checked {
            Ok(report) => {
                if let Some(mut hit) = report.verified.into_iter().next() {
                    hit.claim_index = claim_index;
                    verified += 1;
                    confidence_sum += hit.confidence;
                    sse_event(EVENT_VERIFIED, &hit)
                } else if let Some(mut miss) = report.hallucinated.into_iter().next() {
                    miss.claim_index = claim_index;
                    rejected += 1;
                    sse_event(EVENT_REJECTED, &miss)
                } else {
                    continue;
                }
            }
            Err(join_error) => {
                let problem = ApiError::from(join_error).problem();
                let _ = events.send(Ok(sse_event(EVENT_ERROR, &problem))).await;
                return;
            }
        };
        if events.send(Ok(event)).await.is_err() {
            // The client disconnected; stop verifying.
            return;
        }
    }

    let summary = StreamSummary {
        total_claims,
        verified,
        rejected,
        confidence_score: mean_confidence(confidence_sum, total_claims),
    };
    let _ = events.send(Ok(sse_event(EVENT_COMPLETE, &summary))).await;
}

/// Same definition as `VerificationReport::confidence_score`: rejected claims
/// count as zero, an empty batch scores `1.0`.
#[allow(clippy::cast_precision_loss)]
fn mean_confidence(sum: f32, total: usize) -> f32 {
    if total == 0 {
        1.0
    } else {
        (sum / total as f32).clamp(0.0, 1.0)
    }
}

fn sse_event(name: &str, payload: &impl Serialize) -> Event {
    match serde_json::to_string(payload) {
        // Compact JSON contains no raw newlines, so it is a single data line.
        Ok(json) => Event::default().event(name).data(json),
        Err(_) => Event::default()
            .event(EVENT_ERROR)
            .data(r#"{"detail":"failed to encode the event payload"}"#),
    }
}

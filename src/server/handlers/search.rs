//! Hybrid search.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use tokio::task::spawn_blocking;

use crate::retrieval::{HybridQuery, RetrievalBundle, RetrievalConfig};
use crate::server::dto::HybridSearchRequest;
use crate::server::error::ApiError;
use crate::server::extract::ApiJson;
use crate::server::handlers::AppState;
use crate::server::state::NamespaceId;

/// `POST /v1/namespaces/{namespace_id}/search/hybrid`
pub async fn hybrid(
    State(manager): State<AppState>,
    namespace: NamespaceId,
    ApiJson(request): ApiJson<HybridSearchRequest>,
) -> Result<Json<RetrievalBundle>, ApiError> {
    let server = manager.config();
    let defaults = server.retrieval;
    let config = RetrievalConfig {
        top_k: request.top_k.unwrap_or(defaults.top_k),
        candidate_pool: request.candidate_pool.unwrap_or(defaults.candidate_pool),
        rrf_k: request.rrf_k.unwrap_or(defaults.rrf_k),
        vector_weight: request.vector_weight.unwrap_or(defaults.vector_weight),
        lexical_weight: request.lexical_weight.unwrap_or(defaults.lexical_weight),
        expansion_hops: request.expansion_hops.unwrap_or(defaults.expansion_hops),
    };
    // Validate before touching the namespace, so that a bad request is a 422
    // whether or not the namespace exists.
    config.validate()?;
    if config.top_k > server.max_top_k || config.candidate_pool > server.max_top_k {
        return Err(ApiError::Unprocessable(format!(
            "top_k and candidate_pool must not exceed {}",
            server.max_top_k
        )));
    }
    if config.expansion_hops > server.max_expansion_hops {
        return Err(ApiError::Unprocessable(format!(
            "expansion_hops must not exceed {}",
            server.max_expansion_hops
        )));
    }
    if request.text.is_none() && request.embedding.is_none() {
        return Err(crate::error::NexusError::EmptyQuery.into());
    }

    let namespace = manager.get(&namespace).await?;
    // Wait for the lock on the async side; scan and rank on the blocking pool.
    let engine = Arc::clone(namespace.engine()).read_owned().await;
    let include_embeddings = request.include_embeddings;
    let mut bundle = spawn_blocking(move || {
        let query = HybridQuery {
            text: request.text.as_deref(),
            embedding: request.embedding.as_deref(),
        };
        engine.search_with(&query, &config)
    })
    .await??;

    if !include_embeddings {
        for node in &mut bundle.context.nodes {
            node.entity.embedding = None;
        }
    }
    Ok(Json(bundle))
}

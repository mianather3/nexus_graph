//! Route handlers.

pub mod ingest;
pub mod ops;
pub mod search;
pub mod snapshot;
pub mod verify;

use std::sync::Arc;

use crate::server::state::EngineManager;

/// Shared state of every handler.
pub type AppState = Arc<EngineManager>;

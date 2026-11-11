//! Crate-wide error type for the retrieval layers.

use thiserror::Error;

use crate::graph::GraphError;
use crate::models::EntityId;
use crate::storage::StorageError;
use crate::vector::Kernel;

/// Every failure of the vector, lexical and retrieval layers.
#[derive(Debug, Error)]
pub enum NexusError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Storage(#[from] StorageError),

    #[error("vector dimension must be greater than zero")]
    ZeroDimension,
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("vector contains NaN or infinite components")]
    NonFiniteVector,
    #[error("vector has zero magnitude, its direction is undefined")]
    ZeroMagnitude,
    #[error("numeric overflow while computing a vector kernel")]
    NumericOverflow,
    #[error("the {0} kernel is not available on this CPU")]
    KernelUnavailable(Kernel),
    #[error("bulk load of {ids} ids with dimension {dim} needs {expected} values, got {actual}")]
    BufferLengthMismatch {
        ids: usize,
        dim: usize,
        expected: usize,
        actual: usize,
    },

    #[error("entity {0} is already indexed")]
    DuplicateDocument(EntityId),
    #[error("index capacity exceeded")]
    CapacityExceeded,
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("query has neither text nor an embedding")]
    EmptyQuery,
}

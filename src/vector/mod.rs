//! Dense vector kernels and storage.
//!
//! The functions here are the checked entry points to the SIMD kernels: they
//! reject dimension mismatches, empty vectors and NaN/Inf components before
//! any kernel runs, and reject results that overflowed.

mod simd;
pub mod store;

pub use simd::Kernel;
pub use store::{VectorHit, VectorStore};

use crate::error::NexusError;

/// Dot product using the fastest kernel available on this CPU.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    dot_product_with(Kernel::active(), a, b)
}

/// Dot product using a specific kernel.
pub fn dot_product_with(kernel: Kernel, a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    validate_pair(a, b)?;
    let value = simd::dot_with(kernel, a, b).ok_or(NexusError::KernelUnavailable(kernel))?;
    finite(value)
}

/// Cosine similarity in `[-1.0, 1.0]`, using the fastest available kernel.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    cosine_similarity_with(Kernel::active(), a, b)
}

/// Cosine similarity in `[-1.0, 1.0]`, using a specific kernel.
pub fn cosine_similarity_with(kernel: Kernel, a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    validate_pair(a, b)?;
    let (dot, norm_a, norm_b) =
        simd::dot_and_norms_with(kernel, a, b).ok_or(NexusError::KernelUnavailable(kernel))?;
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(NexusError::ZeroMagnitude);
    }
    let similarity = finite(dot)? / (finite(norm_a)?.sqrt() * finite(norm_b)?.sqrt());
    Ok(finite(similarity)?.clamp(-1.0, 1.0))
}

/// Cosine distance `1 - cosine_similarity`, in `[0.0, 2.0]`.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    cosine_similarity(a, b).map(|similarity| 1.0 - similarity)
}

/// Euclidean norm of a non-empty, finite, non-zero vector.
pub(crate) fn checked_norm(vector: &[f32]) -> Result<f32, NexusError> {
    if vector.iter().any(|v| !v.is_finite()) {
        return Err(NexusError::NonFiniteVector);
    }
    let norm = finite(simd::dot(vector, vector))?.sqrt();
    if norm == 0.0 {
        return Err(NexusError::ZeroMagnitude);
    }
    Ok(norm)
}

/// Unchecked dot product for callers that have already validated both sides.
#[inline]
pub(crate) fn dot_unchecked(a: &[f32], b: &[f32]) -> f32 {
    simd::dot(a, b)
}

fn validate_pair(a: &[f32], b: &[f32]) -> Result<(), NexusError> {
    if a.len() != b.len() {
        return Err(NexusError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    if a.is_empty() {
        return Err(NexusError::ZeroDimension);
    }
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(NexusError::NonFiniteVector);
    }
    Ok(())
}

fn finite(value: f32) -> Result<f32, NexusError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NexusError::NumericOverflow)
    }
}

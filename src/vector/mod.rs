//! Dense vector kernels and storage.
//!
//! The functions here are the checked entry points to the SIMD kernels: they
//! reject dimension mismatches and empty vectors before any kernel runs, and
//! NaN/Inf components and overflowed results right after it (see
//! `non_finite_cause` for why that order is sound).

mod simd;
pub mod store;

pub use simd::Kernel;
pub use store::{VectorHit, VectorStore};

use crate::error::NexusError;

impl Kernel {
    /// Raw dot product with this kernel, skipping all input validation:
    /// mismatched lengths are truncated to the shorter slice and NaN/Inf
    /// propagate per IEEE-754. Returns `None` if the CPU lacks the kernel.
    ///
    /// This is the hot-loop primitive (and what the benchmarks measure); use
    /// [`dot_product_with`] when the inputs are not already validated.
    pub fn dot_raw(self, a: &[f32], b: &[f32]) -> Option<f32> {
        simd::dot_with(self, a, b)
    }

    /// Raw cosine similarity with this kernel, skipping all input validation.
    /// Not clamped; NaN for zero-magnitude inputs. Returns `None` if the CPU
    /// lacks the kernel.
    pub fn cosine_raw(self, a: &[f32], b: &[f32]) -> Option<f32> {
        let (dot, norm_a, norm_b) = simd::dot_and_norms_with(self, a, b)?;
        Some(dot / (norm_a.sqrt() * norm_b.sqrt()))
    }
}

/// Dot product using the fastest kernel available on this CPU.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    dot_product_with(Kernel::active(), a, b)
}

/// Dot product using a specific kernel.
pub fn dot_product_with(kernel: Kernel, a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    check_dimensions(a, b)?;
    let value = simd::dot_with(kernel, a, b).ok_or(NexusError::KernelUnavailable(kernel))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(non_finite_cause(&[a, b]))
    }
}

/// Cosine similarity in `[-1.0, 1.0]`, using the fastest available kernel.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    cosine_similarity_with(Kernel::active(), a, b)
}

/// Cosine similarity in `[-1.0, 1.0]`, using a specific kernel.
pub fn cosine_similarity_with(kernel: Kernel, a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    check_dimensions(a, b)?;
    let (dot, norm_a, norm_b) =
        simd::dot_and_norms_with(kernel, a, b).ok_or(NexusError::KernelUnavailable(kernel))?;
    if !(dot.is_finite() && norm_a.is_finite() && norm_b.is_finite()) {
        return Err(non_finite_cause(&[a, b]));
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(NexusError::ZeroMagnitude);
    }
    let similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
    Ok(finite(similarity)?.clamp(-1.0, 1.0))
}

/// Cosine distance `1 - cosine_similarity`, in `[0.0, 2.0]`.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32, NexusError> {
    cosine_similarity(a, b).map(|similarity| 1.0 - similarity)
}

/// Euclidean norm of a non-empty, finite, non-zero vector.
pub(crate) fn checked_norm(vector: &[f32]) -> Result<f32, NexusError> {
    let squared = simd::dot(vector, vector);
    if !squared.is_finite() {
        return Err(non_finite_cause(&[vector]));
    }
    let norm = squared.sqrt();
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

fn check_dimensions(a: &[f32], b: &[f32]) -> Result<(), NexusError> {
    if a.len() != b.len() {
        return Err(NexusError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    if a.is_empty() {
        return Err(NexusError::ZeroDimension);
    }
    Ok(())
}

/// Explains a non-finite kernel result: a NaN/Inf input, or an overflow of
/// finite inputs.
///
/// NaN/Inf inputs are detected through the kernel's own result instead of a
/// separate scan before it, which keeps the valid path a single pass. This is
/// sound because IEEE-754 arithmetic cannot turn a non-finite term back into
/// a finite sum: `NaN` is absorbing, `Inf * 0` is `NaN`, `Inf * x` is `+-Inf`,
/// and `Inf + (-Inf)` is `NaN`. Every kernel visits every element, so a
/// finite result proves that all inputs were finite; only on the failure path
/// are the inputs scanned, to tell the two causes apart.
fn non_finite_cause(inputs: &[&[f32]]) -> NexusError {
    if inputs.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
        NexusError::NonFiniteVector
    } else {
        NexusError::NumericOverflow
    }
}

fn finite(value: f32) -> Result<f32, NexusError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(NexusError::NumericOverflow)
    }
}

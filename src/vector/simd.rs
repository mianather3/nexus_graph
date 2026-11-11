//! Dot-product kernels: NEON, AVX2+FMA and a portable scalar fallback.
//!
//! This is the only module in the crate allowed to contain `unsafe`.
//!
//! # Safety audit
//!
//! There are exactly two sources of unsafety, and each is discharged locally:
//!
//! 1. **CPU features.** The `#[target_feature]` kernels may only run on a CPU
//!    that has the feature. They are private to this module and are called
//!    from one place each ([`dot_with`] / [`dot_and_norms_with`]), directly
//!    behind the matching runtime `is_*_feature_detected!` check.
//! 2. **Raw-pointer loads.** Every kernel first computes
//!    `len = a.len().min(b.len())` and only loads `LANES` elements at offset
//!    `i` after checking `i + LANES <= len`. No kernel trusts its caller for
//!    bounds, so a dimension mismatch can never read out of bounds; it only
//!    truncates. All loads and stores are the unaligned variants, so `Vec<f32>`
//!    alignment (4 bytes) is sufficient.
//!
//! Everything exported from here is a safe function. Rejecting mismatched
//! dimensions and NaN/Inf values is the job of the checked wrappers in the
//! parent module; the kernels themselves simply propagate IEEE-754 semantics.
#![allow(unsafe_code)]

use std::fmt;

/// A dot-product implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kernel {
    /// Portable code written so that the compiler can auto-vectorize it.
    Scalar,
    /// 128-bit NEON with fused multiply-add (aarch64, e.g. Apple Silicon).
    Neon,
    /// 256-bit AVX2 with FMA (`x86_64`).
    Avx2Fma,
}

impl Kernel {
    pub const ALL: [Self; 3] = [Self::Scalar, Self::Neon, Self::Avx2Fma];

    /// Whether this kernel can run on the current CPU.
    pub fn is_available(self) -> bool {
        match self {
            Self::Scalar => true,
            Self::Neon => neon_available(),
            Self::Avx2Fma => avx2_fma_available(),
        }
    }

    /// The fastest kernel available on the current CPU.
    pub fn active() -> Self {
        if Self::Neon.is_available() {
            Self::Neon
        } else if Self::Avx2Fma.is_available() {
            Self::Avx2Fma
        } else {
            Self::Scalar
        }
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Avx2Fma => "avx2+fma",
        })
    }
}

fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn avx2_fma_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Dot product over the common prefix of `a` and `b`, using the fastest
/// available kernel.
#[inline]
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_with(Kernel::active(), a, b).unwrap_or_else(|| scalar::dot(a, b))
}

/// Dot product with an explicit kernel; `None` if the CPU lacks it.
pub(super) fn dot_with(kernel: Kernel, a: &[f32], b: &[f32]) -> Option<f32> {
    match kernel {
        Kernel::Scalar => Some(scalar::dot(a, b)),
        Kernel::Neon => {
            #[cfg(target_arch = "aarch64")]
            if neon_available() {
                // SAFETY: NEON support was verified on the line above.
                return Some(unsafe { neon::dot(a, b) });
            }
            None
        }
        Kernel::Avx2Fma => {
            #[cfg(target_arch = "x86_64")]
            if avx2_fma_available() {
                // SAFETY: AVX2 and FMA support were verified on the line above.
                return Some(unsafe { avx2::dot(a, b) });
            }
            None
        }
    }
}

/// `(a.b, a.a, b.b)` over the common prefix of `a` and `b`, in one pass, with
/// an explicit kernel; `None` if the CPU lacks it.
pub(super) fn dot_and_norms_with(kernel: Kernel, a: &[f32], b: &[f32]) -> Option<(f32, f32, f32)> {
    match kernel {
        Kernel::Scalar => Some(scalar::dot_and_norms(a, b)),
        Kernel::Neon => {
            #[cfg(target_arch = "aarch64")]
            if neon_available() {
                // SAFETY: NEON support was verified on the line above.
                return Some(unsafe { neon::dot_and_norms(a, b) });
            }
            None
        }
        Kernel::Avx2Fma => {
            #[cfg(target_arch = "x86_64")]
            if avx2_fma_available() {
                // SAFETY: AVX2 and FMA support were verified on the line above.
                return Some(unsafe { avx2::dot_and_norms(a, b) });
            }
            None
        }
    }
}

/// Portable kernels. Eight independent accumulators break the dependency chain
/// of a naive sum, which lets LLVM vectorize the loop without `-ffast-math`.
mod scalar {
    const LANES: usize = 8;

    pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
        let a_chunks = a.chunks_exact(LANES);
        let b_chunks = b.chunks_exact(LANES);
        let consumed = a_chunks.len().min(b_chunks.len()) * LANES;

        let mut acc = [0.0_f32; LANES];
        for (chunk_a, chunk_b) in a_chunks.zip(b_chunks) {
            for ((slot, x), y) in acc.iter_mut().zip(chunk_a).zip(chunk_b) {
                *slot += x * y;
            }
        }
        acc.iter().sum::<f32>() + tail_dot(a, b, consumed)
    }

    pub(super) fn dot_and_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let a_chunks = a.chunks_exact(LANES);
        let b_chunks = b.chunks_exact(LANES);
        let consumed = a_chunks.len().min(b_chunks.len()) * LANES;

        let mut dot = [0.0_f32; LANES];
        let mut norm_a = [0.0_f32; LANES];
        let mut norm_b = [0.0_f32; LANES];
        for (chunk_a, chunk_b) in a_chunks.zip(b_chunks) {
            let lanes = dot.iter_mut().zip(&mut norm_a).zip(&mut norm_b);
            for (((d, na), nb), (x, y)) in lanes.zip(chunk_a.iter().zip(chunk_b)) {
                *d += x * y;
                *na += x * x;
                *nb += y * y;
            }
        }
        let (tail_d, tail_a, tail_b) = tail_dot_and_norms(a, b, consumed);
        (
            dot.iter().sum::<f32>() + tail_d,
            norm_a.iter().sum::<f32>() + tail_a,
            norm_b.iter().sum::<f32>() + tail_b,
        )
    }

    /// Dot product of the elements from `start` to the end of the shorter
    /// slice. Shared by every kernel for the remainder that does not fill a
    /// full SIMD register.
    pub(super) fn tail_dot(a: &[f32], b: &[f32], start: usize) -> f32 {
        a.iter()
            .zip(b)
            .skip(start)
            .fold(0.0, |sum, (x, y)| sum + x * y)
    }

    pub(super) fn tail_dot_and_norms(a: &[f32], b: &[f32], start: usize) -> (f32, f32, f32) {
        a.iter()
            .zip(b)
            .skip(start)
            .fold((0.0, 0.0, 0.0), |(d, na, nb), (x, y)| {
                (d + x * y, na + x * x, nb + y * y)
            })
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    use super::scalar;

    const LANES: usize = 4;

    /// # Safety
    ///
    /// The CPU must support NEON.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let (ptr_a, ptr_b) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;

        // SAFETY: NEON is available (function contract). Each `vld1q_f32`
        // reads `LANES` f32s starting at offset `i + j * LANES`; the loop
        // conditions guarantee the last element read is below `len`, which is
        // within both slices. `vld1q_f32` has no alignment requirement.
        let vector_sum = unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            while i + 4 * LANES <= len {
                acc0 = vfmaq_f32(acc0, vld1q_f32(ptr_a.add(i)), vld1q_f32(ptr_b.add(i)));
                acc1 = vfmaq_f32(
                    acc1,
                    vld1q_f32(ptr_a.add(i + LANES)),
                    vld1q_f32(ptr_b.add(i + LANES)),
                );
                acc2 = vfmaq_f32(
                    acc2,
                    vld1q_f32(ptr_a.add(i + 2 * LANES)),
                    vld1q_f32(ptr_b.add(i + 2 * LANES)),
                );
                acc3 = vfmaq_f32(
                    acc3,
                    vld1q_f32(ptr_a.add(i + 3 * LANES)),
                    vld1q_f32(ptr_b.add(i + 3 * LANES)),
                );
                i += 4 * LANES;
            }
            while i + LANES <= len {
                acc0 = vfmaq_f32(acc0, vld1q_f32(ptr_a.add(i)), vld1q_f32(ptr_b.add(i)));
                i += LANES;
            }
            vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
        };
        vector_sum + scalar::tail_dot(a, b, i)
    }

    /// # Safety
    ///
    /// The CPU must support NEON.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot_and_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let len = a.len().min(b.len());
        let (ptr_a, ptr_b) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;

        // SAFETY: NEON is available (function contract). Each `vld1q_f32`
        // reads `LANES` f32s at offset `i` only while `i + LANES <= len`, and
        // `len` is within both slices. No alignment requirement.
        let (dot, norm_a, norm_b) = unsafe {
            let mut dot = vdupq_n_f32(0.0);
            let mut norm_a = vdupq_n_f32(0.0);
            let mut norm_b = vdupq_n_f32(0.0);
            while i + LANES <= len {
                let va = vld1q_f32(ptr_a.add(i));
                let vb = vld1q_f32(ptr_b.add(i));
                dot = vfmaq_f32(dot, va, vb);
                norm_a = vfmaq_f32(norm_a, va, va);
                norm_b = vfmaq_f32(norm_b, vb, vb);
                i += LANES;
            }
            (vaddvq_f32(dot), vaddvq_f32(norm_a), vaddvq_f32(norm_b))
        };
        let (tail_d, tail_a, tail_b) = scalar::tail_dot_and_norms(a, b, i);
        (dot + tail_d, norm_a + tail_a, norm_b + tail_b)
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    use super::scalar;

    const LANES: usize = 8;

    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let (ptr_a, ptr_b) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;
        let mut lanes = [0.0_f32; LANES];

        // SAFETY: AVX2 and FMA are available (function contract). Each
        // `_mm256_loadu_ps` reads `LANES` f32s starting at offset
        // `i + j * LANES`; the loop conditions guarantee the last element read
        // is below `len`, which is within both slices. `_mm256_storeu_ps`
        // writes exactly `LANES` f32s into `lanes`, which holds `LANES`
        // elements. Both are unaligned operations.
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            while i + 4 * LANES <= len {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(ptr_a.add(i)),
                    _mm256_loadu_ps(ptr_b.add(i)),
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(ptr_a.add(i + LANES)),
                    _mm256_loadu_ps(ptr_b.add(i + LANES)),
                    acc1,
                );
                acc2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(ptr_a.add(i + 2 * LANES)),
                    _mm256_loadu_ps(ptr_b.add(i + 2 * LANES)),
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(ptr_a.add(i + 3 * LANES)),
                    _mm256_loadu_ps(ptr_b.add(i + 3 * LANES)),
                    acc3,
                );
                i += 4 * LANES;
            }
            while i + LANES <= len {
                acc0 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(ptr_a.add(i)),
                    _mm256_loadu_ps(ptr_b.add(i)),
                    acc0,
                );
                i += LANES;
            }
            let total = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            _mm256_storeu_ps(lanes.as_mut_ptr(), total);
        }
        lanes.iter().sum::<f32>() + scalar::tail_dot(a, b, i)
    }

    /// # Safety
    ///
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::similar_names)] // the a/b pairs are deliberately parallel
    pub(super) unsafe fn dot_and_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let len = a.len().min(b.len());
        let (ptr_a, ptr_b) = (a.as_ptr(), b.as_ptr());
        let mut i = 0;
        let mut dot_lanes = [0.0_f32; LANES];
        let mut norm_a_lanes = [0.0_f32; LANES];
        let mut norm_b_lanes = [0.0_f32; LANES];

        // SAFETY: AVX2 and FMA are available (function contract). Each
        // `_mm256_loadu_ps` reads `LANES` f32s at offset `i` only while
        // `i + LANES <= len`, and `len` is within both slices. Each
        // `_mm256_storeu_ps` writes exactly `LANES` f32s into an array of
        // `LANES` elements. All are unaligned operations.
        unsafe {
            let mut dot = _mm256_setzero_ps();
            let mut norm_a = _mm256_setzero_ps();
            let mut norm_b = _mm256_setzero_ps();
            while i + LANES <= len {
                let va = _mm256_loadu_ps(ptr_a.add(i));
                let vb = _mm256_loadu_ps(ptr_b.add(i));
                dot = _mm256_fmadd_ps(va, vb, dot);
                norm_a = _mm256_fmadd_ps(va, va, norm_a);
                norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
                i += LANES;
            }
            _mm256_storeu_ps(dot_lanes.as_mut_ptr(), dot);
            _mm256_storeu_ps(norm_a_lanes.as_mut_ptr(), norm_a);
            _mm256_storeu_ps(norm_b_lanes.as_mut_ptr(), norm_b);
        }
        let (tail_d, tail_a, tail_b) = scalar::tail_dot_and_norms(a, b, i);
        (
            dot_lanes.iter().sum::<f32>() + tail_d,
            norm_a_lanes.iter().sum::<f32>() + tail_a,
            norm_b_lanes.iter().sum::<f32>() + tail_b,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kernels must tolerate mismatched lengths by truncating, never by
    /// reading past the shorter slice. (The checked API rejects mismatches;
    /// this guards the memory-safety argument itself.)
    #[test]
    fn kernels_truncate_to_the_shorter_slice() {
        let long: Vec<f32> = (0..40_u8).map(f32::from).collect();
        for short_len in 0..40 {
            let short = vec![1.0_f32; short_len];
            let expected: f32 = long.iter().take(short_len).sum();
            for kernel in Kernel::ALL {
                if let Some(value) = dot_with(kernel, &long, &short) {
                    assert_eq!(value, expected, "{kernel} dot, len {short_len}");
                }
                if let Some((value, _, norm_short)) = dot_and_norms_with(kernel, &long, &short) {
                    assert_eq!(value, expected, "{kernel} dot_and_norms, len {short_len}");
                    assert_eq!(norm_short, f32::from(u8::try_from(short_len).unwrap()));
                }
            }
        }
    }

    #[test]
    fn kernels_propagate_non_finite_values() {
        let ones = [1.0_f32; 19];
        for position in 0..ones.len() {
            for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut poisoned = ones;
                poisoned[position] = poison;
                for kernel in Kernel::ALL {
                    if let Some(value) = dot_with(kernel, &poisoned, &ones) {
                        assert!(!value.is_finite(), "{kernel} hid {poison} at {position}");
                    }
                }
            }
        }
    }

    #[test]
    fn active_kernel_is_available() {
        assert!(Kernel::active().is_available());
        assert!(Kernel::Scalar.is_available());
    }
}

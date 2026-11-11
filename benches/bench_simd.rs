//! Micro-benchmarks of the dot-product and cosine kernels.
//!
//! Compares every kernel the CPU supports (scalar, plus NEON on aarch64 or
//! AVX2+FMA on `x86_64`) across embedding dimensions, at cache-line-aligned
//! and deliberately misaligned addresses, and measures what input validation
//! adds on top of the raw kernel.
//!
//! ```text
//! cargo bench --bench bench_simd
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    missing_docs
)]

mod common;

use std::hint::black_box;

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use nexus_graph::{cosine_similarity, dot_product, Kernel};

use common::Lcg;

const DIMENSIONS: [usize; 3] = [384, 768, 1536];
const CACHE_LINE: usize = 64;

fn available_kernels() -> impl Iterator<Item = Kernel> {
    Kernel::ALL
        .into_iter()
        .filter(|kernel| kernel.is_available())
}

fn dot_kernels(c: &mut Criterion) {
    let mut rng = Lcg(1);
    let mut group = c.benchmark_group("dot");
    for dim in DIMENSIONS {
        let a = rng.vector(dim);
        let b = rng.vector(dim);
        group.throughput(Throughput::Elements(dim as u64));
        for kernel in available_kernels() {
            group.bench_with_input(
                BenchmarkId::new(kernel.to_string(), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| kernel.dot_raw(black_box(&a), black_box(&b)));
                },
            );
        }
    }
    group.finish();
}

fn cosine_kernels(c: &mut Criterion) {
    let mut rng = Lcg(2);
    let mut group = c.benchmark_group("cosine");
    for dim in DIMENSIONS {
        let a = rng.vector(dim);
        let b = rng.vector(dim);
        group.throughput(Throughput::Elements(dim as u64));
        for kernel in available_kernels() {
            group.bench_with_input(
                BenchmarkId::new(kernel.to_string(), dim),
                &dim,
                |bench, _| {
                    bench.iter(|| kernel.cosine_raw(black_box(&a), black_box(&b)));
                },
            );
        }
    }
    group.finish();
}

/// A `dim`-element window of `buffer` whose address is `offset_bytes` past a
/// cache-line boundary.
fn window(buffer: &[f32], dim: usize, offset_bytes: usize) -> &[f32] {
    let start = (0..CACHE_LINE)
        .find(|&i| (buffer[i..].as_ptr() as usize) % CACHE_LINE == offset_bytes)
        .expect("a 64-element prefix contains every 4-byte phase of a cache line");
    &buffer[start..start + dim]
}

fn alignment(c: &mut Criterion) {
    const DIM: usize = 768;
    let mut rng = Lcg(3);
    let buffer_a = rng.vector(DIM + 2 * CACHE_LINE);
    let buffer_b = rng.vector(DIM + 2 * CACHE_LINE);

    let mut group = c.benchmark_group("dot_alignment_768");
    group.throughput(Throughput::Elements(DIM as u64));
    // 0: both operands start on a cache line. 4: every 16- and 32-byte load
    // is misaligned and one load in four (NEON) or two (AVX2) splits a line.
    for (label, offset) in [("cache_aligned", 0), ("offset_4_bytes", 4)] {
        let a = window(&buffer_a, DIM, offset);
        let b = window(&buffer_b, DIM, offset);
        for kernel in available_kernels() {
            group.bench_function(BenchmarkId::new(kernel.to_string(), label), |bench| {
                bench.iter(|| kernel.dot_raw(black_box(a), black_box(b)));
            });
        }
    }
    group.finish();
}

/// The public, validating entry points versus the raw kernel underneath:
/// the price of the dimension and NaN/Inf checks.
fn validation_overhead(c: &mut Criterion) {
    const DIM: usize = 768;
    let mut rng = Lcg(4);
    let a = rng.vector(DIM);
    let b = rng.vector(DIM);
    let kernel = Kernel::active();

    let mut group = c.benchmark_group("validation_overhead_768");
    group.throughput(Throughput::Elements(DIM as u64));
    group.bench_function("dot/raw", |bench| {
        bench.iter(|| kernel.dot_raw(black_box(&a), black_box(&b)));
    });
    group.bench_function("dot/checked", |bench| {
        bench.iter(|| dot_product(black_box(&a), black_box(&b)));
    });
    group.bench_function("cosine/raw", |bench| {
        bench.iter(|| kernel.cosine_raw(black_box(&a), black_box(&b)));
    });
    group.bench_function("cosine/checked", |bench| {
        bench.iter(|| cosine_similarity(black_box(&a), black_box(&b)));
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = dot_kernels, cosine_kernels, alignment, validation_overhead
}

fn main() {
    println!("active kernel: {}", Kernel::active());
    benches();
    Criterion::default().configure_from_args().final_summary();
}

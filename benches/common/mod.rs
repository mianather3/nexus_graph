//! Shared benchmark support: deterministic data and tail-latency reports.
//!
//! Criterion estimates a central tendency (with confidence intervals and
//! outlier classification); it does not report tail percentiles. The
//! `report_latency` helpers complement it by timing every single call of an
//! operation and printing p50 / p95 / p99.

#![allow(dead_code)]

use std::time::{Duration, Instant};

use criterion::Criterion;

/// Criterion settings shared by all benchmark binaries: explicit warm-up,
/// fixed measurement window, and a noise threshold below which a change is
/// not reported as a regression or improvement. Command-line arguments
/// (filters, `--save-baseline`, ...) still apply.
pub fn criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(60)
        .noise_threshold(0.03)
        .configure_from_args()
}

/// Deterministic pseudo-random numbers (64-bit LCG), so that every run
/// benchmarks identical data.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    /// Uniform in `[0, bound)`.
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound.max(1) as u64) as usize
    }

    /// Uniform in `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u64() >> 29) as f32 / (1_u64 << 24) as f32;
        unit * 2.0 - 1.0
    }

    pub fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

/// Whether tail-latency reports should run: not when Cargo merely smoke-tests
/// the binary (`--test`) or lists benchmarks (`--list`).
pub fn latency_reports_enabled() -> bool {
    !std::env::args().any(|arg| arg == "--test" || arg == "--list")
}

/// Whether `name` passes the benchmark filter given on the command line (the
/// first positional argument, matched as a substring like Criterion does).
pub fn selected(name: &str) -> bool {
    std::env::args()
        .skip(1)
        .find(|arg| !arg.starts_with('-'))
        .map_or(true, |filter| name.contains(&filter))
}

fn format_duration(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos < 10_000 {
        format!("{nanos} ns")
    } else if nanos < 10_000_000 {
        format!("{:.1} us", nanos as f64 / 1e3)
    } else {
        format!("{:.2} ms", nanos as f64 / 1e6)
    }
}

/// Nearest-rank percentile of an ascending-sorted sample.
pub fn percentile(sorted: &[Duration], quantile: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = (quantile * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Prints the p50 / p95 / p99 / max of individually timed calls.
pub fn print_latency(name: &str, samples: &mut [Duration]) {
    samples.sort_unstable();
    println!(
        "latency {name:<46} n={:<6} p50={:<10} p95={:<10} p99={:<10} max={}",
        samples.len(),
        format_duration(percentile(samples, 0.50)),
        format_duration(percentile(samples, 0.95)),
        format_duration(percentile(samples, 0.99)),
        format_duration(samples.last().copied().unwrap_or_default()),
    );
}

/// Times `iterations` individual calls of `operation` (after a 10% warm-up)
/// and prints their latency percentiles.
pub fn report_latency(name: &str, iterations: usize, mut operation: impl FnMut()) {
    if !latency_reports_enabled() || !selected(name) {
        return;
    }
    for _ in 0..(iterations / 10).max(1) {
        operation();
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed());
    }
    print_latency(name, &mut samples);
}

//! Builds a vector store of a given shape and reports its heap accounting,
//! so that the process's peak RSS can be measured from outside:
//!
//! ```text
//! cargo build --release --example memory_footprint
//! /usr/bin/time -l target/release/examples/memory_footprint 100000 384   # macOS
//! /usr/bin/time -v target/release/examples/memory_footprint 100000 384   # Linux
//! ```

use anyhow::{Context, Result};
use nexus_graph::{EntityId, VectorStore};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let rows: usize = args
        .next()
        .as_deref()
        .unwrap_or("100000")
        .parse()
        .context("rows")?;
    let dim: usize = args
        .next()
        .as_deref()
        .unwrap_or("384")
        .parse()
        .context("dim")?;

    // Deterministic filler; the values only need to be finite and non-zero.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        f32::from(u16::try_from(state >> 48).unwrap_or(1)) / 65_536.0 + 0.001
    };
    let flat: Vec<f32> = (0..rows * dim).map(|_| next()).collect();
    let ids: Vec<EntityId> = (0..rows as u64).map(EntityId).collect();

    let mut store = VectorStore::new(dim)?;
    // An empty store adopts the buffer: the vectors exist once in memory.
    store.bulk_load(ids, flat)?;

    let query: Vec<f32> = (0..dim).map(|_| next()).collect();
    let hits = store.search(&query, 10)?;

    let raw = rows * dim * std::mem::size_of::<f32>();
    println!("rows={rows} dim={dim}");
    println!("raw vector payload:  {raw} bytes ({:.1} MiB)", mib(raw));
    println!(
        "store heap estimate: {} bytes ({:.1} MiB)",
        store.memory_bytes(),
        mib(store.memory_bytes())
    );
    println!(
        "overhead per vector: {:.1} bytes",
        bytes_per_row(store.memory_bytes() - raw, rows)
    );
    println!("top hit: {:?}", hits.first().map(|hit| hit.id));
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[allow(clippy::cast_precision_loss)]
fn bytes_per_row(bytes: usize, rows: usize) -> f64 {
    bytes as f64 / rows.max(1) as f64
}

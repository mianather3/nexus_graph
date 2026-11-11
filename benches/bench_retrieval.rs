//! Retrieval benchmarks: exact vector scan, BM25 lookup, rank fusion and the
//! end-to-end hybrid query.
//!
//! ```text
//! cargo bench --bench bench_retrieval
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

use std::fmt::Write as _;
use std::hint::black_box;
use std::sync::Arc;

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use nexus_graph::{
    reciprocal_rank_fusion, Attributes, Bm25Index, EntityId, EntityType, HybridQuery,
    HybridRetriever, KnowledgeGraph, RankedList, RelationType, RetrievalConfig, VectorStore,
};

use common::{report_latency, Lcg};

const TOP_K: usize = 10;

fn vector_store(rows: usize, dim: usize, seed: u64) -> VectorStore {
    let mut rng = Lcg(seed);
    let ids = (0..rows as u64).map(EntityId).collect();
    let flat = (0..rows * dim).map(|_| rng.next_f32()).collect();
    let mut store = VectorStore::new(dim).unwrap();
    store.bulk_load(ids, flat).unwrap();
    store
}

/// Zipf-distributed synthetic vocabulary: word `i` has weight `1 / (i + 1)`.
struct Vocabulary {
    cumulative: Vec<f64>,
}

impl Vocabulary {
    fn new(size: usize) -> Self {
        let mut total = 0.0;
        let cumulative = (0..size)
            .map(|rank| {
                total += 1.0 / (rank + 1) as f64;
                total
            })
            .collect();
        Self { cumulative }
    }

    fn sample(&self, rng: &mut Lcg) -> usize {
        let total = self.cumulative.last().copied().unwrap_or(1.0);
        let target = f64::from(rng.next_f32().abs()) * total;
        self.cumulative.partition_point(|&c| c < target)
    }

    fn document(&self, rng: &mut Lcg, tokens: usize) -> String {
        let mut text = String::with_capacity(tokens * 7);
        for _ in 0..tokens {
            let _ = write!(text, "w{} ", self.sample(rng));
        }
        text
    }
}

fn bm25_index(documents: usize, seed: u64) -> Bm25Index {
    let vocabulary = Vocabulary::new(20_000);
    let mut rng = Lcg(seed);
    let mut index = Bm25Index::default();
    for id in 0..documents as u64 {
        index
            .add_document(EntityId(id), &vocabulary.document(&mut rng, 40))
            .unwrap();
    }
    index
}

/// One common, one mid-frequency and one rare term.
const BM25_QUERY: &str = "w3 w150 w9000";

fn vector_scan(c: &mut Criterion) {
    let mut rng = Lcg(10);

    // Scale: rows grow, dimension fixed.
    let mut group = c.benchmark_group("vector_scan_dim384");
    for rows in [1_000_usize, 10_000, 100_000] {
        let store = vector_store(rows, 384, rows as u64);
        let query = rng.vector(384);
        group.throughput(Throughput::Elements(rows as u64));
        group.sample_size(if rows >= 100_000 { 30 } else { 60 });
        group.bench_with_input(BenchmarkId::from_parameter(rows), &rows, |bench, _| {
            bench.iter(|| store.search(black_box(&query), TOP_K).unwrap());
        });
    }
    group.finish();

    // Dimension: rows fixed.
    let mut group = c.benchmark_group("vector_scan_10k_rows");
    for dim in [384_usize, 768, 1536] {
        let store = vector_store(10_000, dim, dim as u64);
        let query = rng.vector(dim);
        group.throughput(Throughput::Elements(10_000));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| store.search(black_box(&query), TOP_K).unwrap());
        });
    }
    group.finish();
}

fn bm25_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_query");
    for documents in [1_000_usize, 10_000, 100_000] {
        let index = bm25_index(documents, 20);
        group.bench_with_input(
            BenchmarkId::from_parameter(documents),
            &documents,
            |bench, _| bench.iter(|| index.search(black_box(BM25_QUERY), TOP_K)),
        );
    }
    group.finish();
}

fn rank_fusion(c: &mut Criterion) {
    let mut rng = Lcg(30);
    let mut group = c.benchmark_group("rrf_fusion");
    for channels in [2_usize, 4] {
        for length in [10_usize, 100, 1_000] {
            // Channels draw from a shared pool twice their length, so roughly
            // half of each list overlaps with the others.
            let lists: Vec<Vec<EntityId>> = (0..channels)
                .map(|_| {
                    let mut pool: Vec<u64> = (0..2 * length as u64).collect();
                    for i in (1..pool.len()).rev() {
                        pool.swap(i, rng.below(i + 1));
                    }
                    pool.into_iter().take(length).map(EntityId).collect()
                })
                .collect();
            let ranked: Vec<RankedList<'_>> = lists
                .iter()
                .map(|ids| RankedList { weight: 1.0, ids })
                .collect();
            group.throughput(Throughput::Elements((channels * length) as u64));
            group.bench_function(
                BenchmarkId::new(format!("{channels}_channels"), length),
                |bench| bench.iter(|| reciprocal_rank_fusion(black_box(&ranked), 60.0).unwrap()),
            );
        }
    }
    group.finish();
}

/// A graph of `entities` nodes with 384-dimensional embeddings, a short text
/// description each, and about four relations per node.
fn hybrid_retriever(entities: usize) -> (HybridRetriever, Vec<f32>) {
    const DIM: usize = 384;
    let vocabulary = Vocabulary::new(5_000);
    let mut rng = Lcg(40);
    let graph = KnowledgeGraph::new();
    let mut nodes = Vec::with_capacity(entities);
    for i in 0..entities {
        let mut attributes = Attributes::new();
        attributes.insert("description".into(), vocabulary.document(&mut rng, 24));
        let node = graph
            .add_entity(
                format!("entity-{i}"),
                EntityType::Service,
                attributes,
                Some(rng.vector(DIM)),
            )
            .unwrap();
        nodes.push(node);
    }
    for &source in &nodes {
        for _ in 0..2 {
            let target = nodes[rng.below(entities)];
            graph
                .add_relation(
                    source,
                    target,
                    RelationType::DependsOn,
                    0.9,
                    Attributes::new(),
                )
                .unwrap();
        }
    }
    let retriever =
        HybridRetriever::from_graph(Arc::new(graph), RetrievalConfig::default()).unwrap();
    (retriever, rng.vector(DIM))
}

fn hybrid_search(c: &mut Criterion) {
    let (retriever, embedding) = hybrid_retriever(10_000);
    let query = HybridQuery {
        text: Some("w3 w150 w900"),
        embedding: Some(&embedding),
    };
    let mut group = c.benchmark_group("hybrid_search_10k_entities");
    for hops in [0_usize, 1, 2] {
        let config = RetrievalConfig {
            expansion_hops: hops,
            ..RetrievalConfig::default()
        };
        group.bench_with_input(
            BenchmarkId::new("expansion_hops", hops),
            &hops,
            |bench, _| {
                bench.iter(|| retriever.search_with(black_box(&query), &config).unwrap());
            },
        );
    }
    group.finish();
}

fn latency_reports() {
    if !common::latency_reports_enabled() {
        return;
    }
    let mut rng = Lcg(50);

    for rows in [10_000_usize, 100_000] {
        let name = format!("vector_scan dim=384 rows={rows} top_k={TOP_K}");
        if !common::selected(&name) {
            continue;
        }
        let store = vector_store(rows, 384, rows as u64);
        let query = rng.vector(384);
        report_latency(&name, 1_000, || {
            black_box(store.search(black_box(&query), TOP_K).unwrap());
        });
    }

    let name = "bm25_query docs=100000 terms=3";
    if common::selected(name) {
        let index = bm25_index(100_000, 20);
        report_latency(name, 2_000, || {
            black_box(index.search(black_box(BM25_QUERY), TOP_K));
        });
    }

    let name = "hybrid_search entities=10000 dim=384 hops=1";
    if common::selected(name) {
        let (retriever, embedding) = hybrid_retriever(10_000);
        let query = HybridQuery {
            text: Some("w3 w150 w900"),
            embedding: Some(&embedding),
        };
        report_latency(name, 2_000, || {
            black_box(retriever.search(black_box(&query)).unwrap());
        });
    }
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = vector_scan, bm25_query, rank_fusion, hybrid_search
}

fn main() {
    benches();
    latency_reports();
    Criterion::default().configure_from_args().final_summary();
}

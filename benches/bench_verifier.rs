//! Graph benchmarks: k-hop BFS expansion and deterministic claim verification.
//!
//! ```text
//! cargo bench --bench bench_verifier
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
use nexus_graph::{
    verify_claims, Attributes, CitationRef, EntityType, GraphClaim, KnowledgeGraph, NodeIndex,
    RelationType, VerifierConfig,
};

use common::{report_latency, Lcg};

fn claim(source: &str, target: &str) -> GraphClaim {
    GraphClaim {
        source_entity: source.to_owned(),
        target_entity: target.to_owned(),
        relation: RelationType::DependsOn,
        citation: CitationRef {
            document_id: "bench".into(),
            chunk_id: "0".into(),
        },
    }
}

/// `nodes` entities, each with `out_degree` random `DependsOn` edges.
fn random_graph(nodes: usize, out_degree: usize, seed: u64) -> (KnowledgeGraph, Vec<NodeIndex>) {
    let mut rng = Lcg(seed);
    let graph = KnowledgeGraph::new();
    let indices: Vec<NodeIndex> = (0..nodes)
        .map(|i| {
            graph
                .add_entity(
                    format!("n{i}"),
                    EntityType::Service,
                    Attributes::new(),
                    None,
                )
                .unwrap()
        })
        .collect();
    for &source in &indices {
        for _ in 0..out_degree {
            let target = indices[rng.below(nodes)];
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
    (graph, indices)
}

/// A complete `branching`-ary dependency tree of the given depth (root at
/// level 0), plus one isolated entity. Returns the graph and the leaf names.
fn dependency_tree(branching: usize, depth: usize) -> (KnowledgeGraph, Vec<String>) {
    let graph = KnowledgeGraph::new();
    let add = |name: &str| {
        graph
            .add_entity(name, EntityType::Service, Attributes::new(), None)
            .unwrap()
    };
    add("isolated");
    let mut level = vec![("root".to_owned(), add("root"))];
    for _ in 0..depth {
        let mut next = Vec::with_capacity(level.len() * branching);
        for (parent_name, parent) in &level {
            for child in 0..branching {
                let name = format!("{parent_name}.{child}");
                let node = add(&name);
                graph
                    .add_relation(
                        *parent,
                        node,
                        RelationType::DependsOn,
                        0.95,
                        Attributes::new(),
                    )
                    .unwrap();
                next.push((name, node));
            }
        }
        level = next;
    }
    (graph, level.into_iter().map(|(name, _)| name).collect())
}

fn bfs_depth_scaling(c: &mut Criterion) {
    let (graph, nodes) = random_graph(50_000, 4, 1);
    let mut rng = Lcg(2);
    let starts: Vec<NodeIndex> = (0..64).map(|_| nodes[rng.below(nodes.len())]).collect();

    let mut group = c.benchmark_group("bfs_k_hop_50k_nodes_degree8");
    for k in [1_usize, 2, 3] {
        let reached: usize = starts
            .iter()
            .map(|&start| graph.get_k_hop_subgraph(&[start], k).nodes.len())
            .sum();
        println!("k={k}: mean subgraph size {} nodes", reached / starts.len());
        let mut next = 0;
        group.bench_with_input(BenchmarkId::new("k", k), &k, |bench, &k| {
            bench.iter(|| {
                next = (next + 1) % starts.len();
                graph.get_k_hop_subgraph(black_box(&[starts[next]]), k)
            });
        });
    }
    group.finish();
}

fn verification_branch_factor(c: &mut Criterion) {
    const CLAIMS: usize = 64;
    let config = VerifierConfig { max_hops: 3 };
    let mut rng = Lcg(3);

    let mut group = c.benchmark_group("verify_64_claims_depth3_tree");
    group.throughput(Throughput::Elements(CLAIMS as u64));
    for branching in [2_usize, 4, 8, 16] {
        let (graph, leaves) = dependency_tree(branching, 3);
        // Verified through a 3-hop chain: root -> ... -> leaf.
        let reachable: Vec<GraphClaim> = (0..CLAIMS)
            .map(|_| claim("root", &leaves[rng.below(leaves.len())]))
            .collect();
        // Rejected: the whole tree is explored before giving up.
        let unreachable: Vec<GraphClaim> = (0..CLAIMS).map(|_| claim("root", "isolated")).collect();
        assert!(verify_claims(&graph, &reachable, config).is_fully_verified());
        assert_eq!(
            verify_claims(&graph, &unreachable, config).verified.len(),
            0
        );

        group.bench_with_input(
            BenchmarkId::new("verified_3_hop", branching),
            &branching,
            |bench, _| bench.iter(|| verify_claims(&graph, black_box(&reachable), config)),
        );
        group.bench_with_input(
            BenchmarkId::new("rejected_unreachable", branching),
            &branching,
            |bench, _| bench.iter(|| verify_claims(&graph, black_box(&unreachable), config)),
        );
    }
    group.finish();
}

fn verification_direct(c: &mut Criterion) {
    const CLAIMS: usize = 1_000;
    let (graph, nodes) = random_graph(50_000, 4, 4);
    let view = graph.read();
    let claims: Vec<GraphClaim> = nodes
        .iter()
        .take(CLAIMS)
        .map(|&source| {
            let (target, _) = view.outgoing_edges(source, None)[0];
            claim(
                &view.node(source).unwrap().canonical_name,
                &view.node(target).unwrap().canonical_name,
            )
        })
        .collect();
    drop(view);
    let config = VerifierConfig::default();
    assert!(verify_claims(&graph, &claims, config).is_fully_verified());

    let mut group = c.benchmark_group("verify_direct_claims_50k_nodes");
    group.throughput(Throughput::Elements(CLAIMS as u64));
    group.bench_function("1000_claims", |bench| {
        bench.iter(|| verify_claims(&graph, black_box(&claims), config));
    });
    group.finish();
}

fn latency_reports() {
    if !common::latency_reports_enabled() {
        return;
    }
    let (graph, nodes) = random_graph(50_000, 4, 1);
    let mut rng = Lcg(5);
    for k in [1_usize, 2, 3] {
        report_latency(&format!("bfs k={k} nodes=50000 degree=8"), 5_000, || {
            let start = nodes[rng.below(nodes.len())];
            black_box(graph.get_k_hop_subgraph(&[start], k));
        });
    }

    let config = VerifierConfig { max_hops: 3 };
    for branching in [4_usize, 16] {
        let (tree, leaves) = dependency_tree(branching, 3);
        report_latency(
            &format!("verify 1 claim, 3-hop chain, branching={branching}"),
            5_000,
            || {
                let single = [claim("root", &leaves[rng.below(leaves.len())])];
                black_box(verify_claims(&tree, &single, config));
            },
        );
    }
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bfs_depth_scaling, verification_branch_factor, verification_direct
}

fn main() {
    benches();
    latency_reports();
    Criterion::default().configure_from_args().final_summary();
}

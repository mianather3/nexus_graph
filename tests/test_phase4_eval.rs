//! Phase 4 tests: retrieval metrics and the differential evaluation runner.

// `assert_is_empty` only exists in recent clippy releases.
#![allow(unknown_lints)]
#![allow(
    clippy::assert_is_empty,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use std::collections::HashSet;

use nexus_graph::eval::{
    coverage, edge_key, evaluate, path_recovered, recall_at_k, reciprocal_rank, EvalConfig,
    EvalDataset, Metrics, System,
};
use nexus_graph::{EntityId, Kernel, NexusError};
use serde_json::{json, Value};

fn ids(raw: &[u64]) -> Vec<EntityId> {
    raw.iter().copied().map(EntityId).collect()
}

fn set(raw: &[u64]) -> HashSet<EntityId> {
    raw.iter().copied().map(EntityId).collect()
}

// ---------------------------------------------------------------------------
// Metric definitions
// ---------------------------------------------------------------------------

#[test]
fn reciprocal_rank_uses_the_first_relevant_hit() {
    let ranked = ids(&[7, 3, 9, 4]);
    assert_eq!(reciprocal_rank(&ranked, &set(&[7])), 1.0);
    assert_eq!(reciprocal_rank(&ranked, &set(&[3])), 0.5);
    assert_eq!(reciprocal_rank(&ranked, &set(&[4])), 0.25);
    // Several relevant items: only the best-ranked one counts.
    assert_eq!(reciprocal_rank(&ranked, &set(&[4, 9])), 1.0 / 3.0);
    assert_eq!(reciprocal_rank(&ranked, &set(&[100])), 0.0);
    assert_eq!(reciprocal_rank(&[], &set(&[7])), 0.0);
    assert_eq!(reciprocal_rank(&ranked, &set(&[])), 0.0);
}

#[test]
fn recall_at_k_counts_relevant_items_within_the_cutoff() {
    let ranked = ids(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    let relevant = set(&[2, 6, 11, 99]);
    assert_eq!(recall_at_k(&ranked, &relevant, 1), 0.0);
    assert_eq!(recall_at_k(&ranked, &relevant, 5), 0.25);
    assert_eq!(recall_at_k(&ranked, &relevant, 10), 0.5);
    assert_eq!(recall_at_k(&ranked, &relevant, 100), 0.75);
    assert_eq!(recall_at_k(&ranked, &relevant, 0), 0.0);
    assert_eq!(recall_at_k(&ranked, &set(&[]), 5), 0.0);
    // A duplicated hit is not counted twice.
    assert_eq!(recall_at_k(&ids(&[2, 2, 2]), &set(&[2, 6]), 3), 0.5);
}

#[test]
fn coverage_is_the_fraction_of_distinct_targets_in_context() {
    let context = set(&[1, 2, 3]);
    assert_eq!(coverage(&context, &ids(&[1, 2, 3])), 1.0);
    assert_eq!(coverage(&context, &ids(&[1, 9])), 0.5);
    assert_eq!(coverage(&context, &ids(&[8, 9])), 0.0);
    assert_eq!(coverage(&context, &ids(&[1, 1, 9, 9])), 0.5);
    assert_eq!(coverage(&context, &[]), 0.0);
    assert_eq!(coverage(&set(&[]), &ids(&[1])), 0.0);
}

#[test]
fn path_recovery_needs_every_hop_as_a_context_edge() {
    let edge = |a: u64, b: u64| edge_key(EntityId(a), EntityId(b));
    assert_eq!(edge(5, 2), edge(2, 5), "edge keys are undirected");

    let edges: HashSet<_> = [edge(1, 2), edge(3, 2)].into();
    assert!(path_recovered(&edges, &ids(&[1, 2])));
    assert!(
        path_recovered(&edges, &ids(&[1, 2, 3])),
        "direction is ignored"
    );
    assert!(path_recovered(&edges, &ids(&[3, 2, 1])));
    assert!(
        !path_recovered(&edges, &ids(&[1, 3])),
        "entities present, hop absent"
    );
    assert!(!path_recovered(&edges, &ids(&[1, 2, 4])));
    // No hops, nothing to recover.
    assert!(!path_recovered(&edges, &ids(&[1])));
    assert!(!path_recovered(&edges, &[]));
}

// ---------------------------------------------------------------------------
// Runner on a hand-checkable dataset
// ---------------------------------------------------------------------------

/// Chain a -> b -> c -> d plus an unrelated pair x -> y.
/// Embedding axes: [a, b, c, d, x/y].
fn tiny_dataset() -> Value {
    let entity = |name: &str, text: &str, embedding: [f32; 5]| json!({ "name": name, "type": "Service", "attributes": { "description": text }, "embedding": embedding });
    json!({
        "name": "tiny",
        "graph": {
            "entities": [
                entity("a", "alpha gateway", [1.0, 0.0, 0.0, 0.0, 0.0]),
                entity("b", "bravo queue", [0.0, 1.0, 0.0, 0.0, 0.0]),
                entity("c", "charlie cache", [0.0, 0.0, 1.0, 0.0, 0.0]),
                entity("d", "delta disk", [0.0, 0.0, 0.0, 1.0, 0.0]),
                entity("x", "xray scanner", [0.0, 0.0, 0.0, 0.0, 1.0]),
                entity("y", "yankee printer", [0.1, 0.0, 0.0, 0.0, 0.9]),
            ],
            "relations": [
                { "source": "a", "target": "b", "relation": "DependsOn" },
                { "source": "b", "target": "c", "relation": "DependsOn" },
                { "source": "c", "target": "d", "relation": "DependsOn" },
                { "source": "x", "target": "y", "relation": "Contains" },
            ],
        },
        "queries": [
            // Both channels point at `a`; the answer `b` is one hop away.
            { "id": "q-hop", "category": "multi_hop", "text": "alpha gateway", "embedding": [1.0, 0.0, 0.0, 0.0, 0.0],
              "relevant": ["b"], "evidence_path": ["a", "b"] },
            // Text names `x` exactly; the embedding points elsewhere (at `d`).
            { "id": "q-lex", "category": "lookup", "text": "xray scanner", "embedding": [0.0, 0.0, 0.0, 1.0, 0.0],
              "relevant": ["x"] },
            // Text matches nothing; the embedding points at `c`.
            { "id": "q-sem", "category": "lookup", "text": "zulu", "embedding": [0.0, 0.0, 1.0, 0.0, 0.0],
              "relevant": ["c"] },
        ],
    })
}

fn parse(value: Value) -> EvalDataset {
    serde_json::from_value(value).unwrap()
}

fn config(top_k: usize, expansion_hops: usize) -> EvalConfig {
    EvalConfig {
        top_k,
        expansion_hops,
        ..EvalConfig::default()
    }
}

#[test]
fn runner_scores_each_system_as_computed_by_hand() {
    let report = evaluate(&parse(tiny_dataset()), &config(1, 1)).unwrap();
    assert_eq!(report.dataset, "tiny");
    assert_eq!(
        (report.entities, report.relations, report.queries),
        (6, 4, 3)
    );
    assert_eq!(report.systems.len(), 4);
    let overall = |system| report.system(system).unwrap().overall;

    // With top_k = 1 every system returns one hit per query.
    //
    //            q-hop (answer b)   q-lex (answer x)   q-sem (answer c)
    // vector     a -> miss          d -> miss          c -> hit
    // bm25       a -> miss          x -> hit           (none) -> miss
    let vector = overall(System::VectorOnly);
    assert_eq!(vector.queries, 3);
    assert!((vector.mrr - 1.0 / 3.0).abs() < 1e-12);
    assert!((vector.recall_at_5 - 1.0 / 3.0).abs() < 1e-12);
    assert!((vector.answer_in_context - 1.0 / 3.0).abs() < 1e-12);
    // Evidence: q-hop has {a, b}, a retrieved -> 0.5; q-lex 0; q-sem 1.
    assert!((vector.evidence_coverage - 0.5).abs() < 1e-12);
    assert_eq!((vector.path_queries, vector.path_recovery), (1, 0.0));
    assert_eq!(vector.mean_context_entities, 1.0);

    let lexical = overall(System::LexicalOnly);
    assert!((lexical.mrr - 1.0 / 3.0).abs() < 1e-12);
    assert!((lexical.evidence_coverage - 0.5).abs() < 1e-12);
    // q-sem has no lexical match at all: contexts hold 1, 1 and 0 entities.
    assert!((lexical.mean_context_entities - 2.0 / 3.0).abs() < 1e-12);

    // Hybrid: q-hop -> a (both agree, miss). q-lex -> x: it is lexical #1 and
    // also appears, far down, in the vector list, which beats d's lone vector
    // #1. q-sem -> c (vector #1, no lexical hits).
    let hybrid = overall(System::HybridRrf);
    assert!((hybrid.mrr - 2.0 / 3.0).abs() < 1e-12);
    assert!((hybrid.evidence_coverage - 2.5 / 3.0).abs() < 1e-12);
    assert_eq!(hybrid.path_recovery, 0.0);
    assert_eq!(hybrid.mean_context_entities, 1.0);

    // Expansion adds the 1-hop neighbourhood of the hit; the ranking is unchanged.
    let graph = overall(System::HybridGraph);
    assert_eq!(graph.mrr, hybrid.mrr);
    assert_eq!(graph.recall_at_5, hybrid.recall_at_5);
    assert_eq!(graph.recall_at_10, hybrid.recall_at_10);
    // Contexts: q-hop {a, b}, q-lex {x, y}, q-sem {c, b, d}. Every answer is in.
    assert_eq!(graph.answer_in_context, 1.0);
    assert_eq!(graph.evidence_coverage, 1.0);
    assert_eq!(graph.path_recovery, 1.0);
    assert!((graph.mean_context_entities - 7.0 / 3.0).abs() < 1e-12);

    // Category breakdown partitions the queries.
    let by_category = &report.system(System::HybridGraph).unwrap().by_category;
    assert_eq!(
        by_category.keys().collect::<Vec<_>>(),
        ["lookup", "multi_hop"]
    );
    assert_eq!(by_category["multi_hop"].queries, 1);
    assert_eq!(by_category["multi_hop"].path_recovery, 1.0);
    assert_eq!(by_category["lookup"].queries, 2);
    assert_eq!(by_category["lookup"].path_queries, 0);
    assert_eq!(by_category["lookup"].answer_in_context, 1.0);
}

#[test]
fn runner_configuration_changes_only_what_it_should() {
    let dataset = parse(tiny_dataset());
    let narrow = evaluate(&dataset, &config(1, 0)).unwrap();
    let wide = evaluate(&dataset, &config(6, 0)).unwrap();

    // Without expansion the two hybrid rows are the same system.
    assert_eq!(
        narrow.system(System::HybridGraph).unwrap().overall,
        narrow.system(System::HybridRrf).unwrap().overall
    );
    // With top_k covering the whole graph, vector retrieval returns every
    // entity, so nothing can be missing from its context.
    let vector = wide.system(System::VectorOnly).unwrap().overall;
    assert_eq!(vector.recall_at_10, 1.0);
    assert_eq!(vector.evidence_coverage, 1.0);
    assert_eq!(vector.path_recovery, 1.0);
    assert_eq!(vector.mean_context_entities, 6.0);

    // A deeper expansion can only add context.
    let hop1 = evaluate(&dataset, &config(1, 1)).unwrap();
    let hop3 = evaluate(&dataset, &config(1, 3)).unwrap();
    let (near, far) = (
        hop1.system(System::HybridGraph).unwrap().overall,
        hop3.system(System::HybridGraph).unwrap().overall,
    );
    assert!(far.mean_context_entities > near.mean_context_entities);
    assert!(far.evidence_coverage >= near.evidence_coverage);
    assert_eq!(far.mrr, near.mrr);

    // Deterministic, and serializable.
    assert_eq!(evaluate(&dataset, &config(1, 1)).unwrap(), hop1);
    let json = serde_json::to_string(&hop1).unwrap();
    assert_eq!(
        serde_json::from_str::<nexus_graph::eval::EvalReport>(&json).unwrap(),
        hop1
    );
    let rendered = hop1.to_string();
    for system in System::ALL {
        assert!(rendered.contains(system.label()), "{rendered}");
    }
    assert!(rendered.contains("category: multi_hop (n=1)"), "{rendered}");
}

#[test]
fn queries_missing_one_modality_are_misses_for_that_channel_only() {
    let mut dataset = tiny_dataset();
    dataset["queries"] = json!([
        { "id": "text-only", "text": "delta disk", "relevant": ["d"] },
        { "id": "vector-only", "embedding": [0.0, 1.0, 0.0, 0.0, 0.0], "relevant": ["b"] },
    ]);
    let report = evaluate(&parse(dataset), &config(1, 0)).unwrap();
    let mrr = |system| report.system(system).unwrap().overall.mrr;
    assert_eq!(mrr(System::VectorOnly), 0.5);
    assert_eq!(mrr(System::LexicalOnly), 0.5);
    assert_eq!(mrr(System::HybridRrf), 1.0);
}

#[test]
fn invalid_datasets_are_rejected_with_a_precise_message() {
    let broken = |mutate: fn(&mut Value)| {
        let mut dataset = tiny_dataset();
        mutate(&mut dataset);
        match evaluate(&parse(dataset), &EvalConfig::default()) {
            Err(NexusError::InvalidDataset(message)) => message,
            other => panic!("expected InvalidDataset, got {other:?}"),
        }
    };

    let message = broken(|d| d["queries"][0]["relevant"] = json!(["nobody"]));
    assert!(
        message.contains("q-hop") && message.contains("nobody"),
        "{message}"
    );

    let message = broken(|d| d["queries"][0]["relevant"] = json!([]));
    assert!(message.contains("no relevant entities"), "{message}");

    // a and c both exist but are not adjacent.
    let message = broken(|d| d["queries"][0]["evidence_path"] = json!(["a", "c"]));
    assert!(
        message.contains("`a` - `c`") && message.contains("not an edge"),
        "{message}"
    );

    let message = broken(|d| {
        d["queries"][0]["text"] = Value::Null;
        d["queries"][0]["embedding"] = Value::Null;
    });
    assert!(
        message.contains("neither text nor an embedding"),
        "{message}"
    );

    let message = broken(|d| d["graph"]["relations"][0]["target"] = json!("ghost"));
    assert!(message.contains("ghost"), "{message}");

    let message = broken(|d| d["graph"]["entities"][1]["name"] = json!("a"));
    assert!(message.contains("already exists"), "{message}");

    let message = broken(|d| d["queries"][0]["embedding"] = json!([1.0, 0.0]));
    assert!(
        message.contains("q-hop") && message.contains("dimension"),
        "{message}"
    );

    // Unknown fields are a parse error, so typos in ground truth cannot pass silently.
    let mut dataset = tiny_dataset();
    dataset["queries"][0]["relevent"] = json!(["b"]);
    assert!(serde_json::from_value::<EvalDataset>(dataset).is_err());

    // Invalid settings are reported as such.
    assert!(matches!(
        evaluate(&parse(tiny_dataset()), &config(0, 1)),
        Err(NexusError::InvalidConfig(_))
    ));
}

// ---------------------------------------------------------------------------
// The shipped example dataset
// ---------------------------------------------------------------------------

fn example_dataset() -> EvalDataset {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/eval_dataset.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn in_unit_range(metrics: &Metrics) {
    for value in [
        metrics.mrr,
        metrics.recall_at_5,
        metrics.recall_at_10,
        metrics.answer_in_context,
        metrics.evidence_coverage,
        metrics.path_recovery,
    ] {
        assert!((0.0..=1.0).contains(&value), "{metrics:?}");
    }
    assert!(metrics.recall_at_5 <= metrics.recall_at_10);
    assert!(metrics.recall_at_10 <= metrics.answer_in_context + 1e-12);
}

#[test]
fn example_dataset_is_valid_and_shows_the_documented_behaviour() {
    let dataset = example_dataset();
    assert_eq!(dataset.graph.entities.len(), 168);
    assert_eq!(dataset.queries.len(), 70);

    let report = evaluate(&dataset, &EvalConfig::default()).unwrap();
    println!("kernel: {}\n{report}", Kernel::active());
    for system in &report.systems {
        in_unit_range(&system.overall);
        system.by_category.values().for_each(in_unit_range);
        assert_eq!(system.overall.queries, 70);
        assert_eq!(system.overall.path_queries, 30);
    }
    let overall = |system| report.system(system).unwrap().overall;
    let category = |system, name: &str| report.system(system).unwrap().by_category[name];

    // The structural properties the README describes. These are properties of
    // how the dataset was constructed, asserted loosely so that they do not
    // depend on float summation order across SIMD kernels.
    //
    // 1. Each single channel fails on the class built against it.
    assert!(category(System::LexicalOnly, "lexical").mrr > 0.95);
    assert!(category(System::VectorOnly, "lexical").mrr < 0.5);
    assert!(category(System::VectorOnly, "semantic").mrr > 0.95);
    assert!(category(System::LexicalOnly, "semantic").mrr < 0.05);
    // 2. Fusion is more robust overall than either channel alone...
    assert!(overall(System::HybridRrf).recall_at_10 > overall(System::VectorOnly).recall_at_10);
    assert!(overall(System::HybridRrf).recall_at_10 > overall(System::LexicalOnly).recall_at_10);
    // ...but is not the best system on every class.
    assert!(
        category(System::HybridRrf, "lexical").mrr < category(System::LexicalOnly, "lexical").mrr
    );
    // 3. Expansion never changes the ranking.
    assert_eq!(
        overall(System::HybridGraph).mrr,
        overall(System::HybridRrf).mrr
    );
    assert_eq!(
        overall(System::HybridGraph).recall_at_10,
        overall(System::HybridRrf).recall_at_10
    );
    // 4. Expansion recovers evidence chains the hits alone miss, for a larger context.
    for name in ["multi_hop_1", "multi_hop_2"] {
        let flat = category(System::HybridRrf, name);
        let expanded = category(System::HybridGraph, name);
        assert!(expanded.path_recovery > flat.path_recovery + 0.3, "{name}");
        assert!(
            expanded.evidence_coverage > flat.evidence_coverage,
            "{name}"
        );
        assert!(
            expanded.mean_context_entities > flat.mean_context_entities,
            "{name}"
        );
    }
    // 2-hop answers are out of reach of every flat top-10.
    assert_eq!(
        category(System::VectorOnly, "multi_hop_2").recall_at_10,
        0.0
    );
    assert_eq!(category(System::HybridRrf, "multi_hop_2").recall_at_10, 0.0);
}

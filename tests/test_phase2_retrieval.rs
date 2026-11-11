//! Phase 2 tests: SIMD kernels, vector store, tokenizer, BM25, rank fusion
//! and end-to-end hybrid retrieval with graph expansion.

// `assert_is_empty` only exists in recent clippy releases.
#![allow(unknown_lints)]
#![allow(
    clippy::assert_is_empty,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use std::collections::BTreeSet;
use std::sync::Arc;

use nexus_graph::lexical::is_stop_word;
use nexus_graph::{
    cosine_distance, cosine_similarity, cosine_similarity_with, dot_product, dot_product_with,
    load_snapshot, reciprocal_rank_fusion, save_snapshot, Attributes, Bm25Index, Bm25Params,
    EntityId, EntityType, HybridQuery, HybridRetriever, Kernel, KnowledgeGraph, NexusError,
    RankedList, RelationType, RetrievalConfig, Tokenizer, VectorStore, DEFAULT_RRF_K,
};

/// Deterministic pseudo-random floats in `[-1, 1)` (64-bit LCG).
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        unit * 2.0 - 1.0
    }

    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

fn reference_dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

fn reference_cosine(a: &[f32], b: &[f32]) -> f64 {
    reference_dot(a, b) / (reference_dot(a, a).sqrt() * reference_dot(b, b).sqrt())
}

fn ids(raw: &[u64]) -> Vec<EntityId> {
    raw.iter().copied().map(EntityId).collect()
}

// ---------------------------------------------------------------------------
// SIMD kernels
// ---------------------------------------------------------------------------

#[test]
fn expected_kernel_is_selected_for_this_cpu() {
    let active = Kernel::active();
    assert!(active.is_available());
    assert!(Kernel::Scalar.is_available());
    #[cfg(target_arch = "aarch64")]
    assert_eq!(active, Kernel::Neon);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        assert_eq!(active, Kernel::Avx2Fma);
    }
    #[cfg(not(target_arch = "aarch64"))]
    assert!(!Kernel::Neon.is_available());
    #[cfg(not(target_arch = "x86_64"))]
    assert!(!Kernel::Avx2Fma.is_available());
}

#[test]
fn simd_and_scalar_kernels_agree_at_every_length() {
    let mut rng = Lcg(0x5eed);
    // Every remainder class of the 4-, 8-, 16- and 32-wide loops, plus
    // realistic embedding sizes.
    let lengths = (1..=67).chain([128, 384, 768, 1536, 1537]);
    for len in lengths {
        let a = rng.vector(len);
        let b = rng.vector(len);
        let expected_dot = reference_dot(&a, &b);
        let expected_cos = reference_cosine(&a, &b);
        // Rounding error of a float sum is bounded relative to the sum of
        // magnitudes, not to the (possibly cancelling) result.
        let magnitude: f64 = a.iter().zip(&b).map(|(x, y)| f64::from(x * y).abs()).sum();
        let tolerance = 1e-6 * magnitude.max(1.0);

        for kernel in Kernel::ALL {
            if !kernel.is_available() {
                assert!(matches!(
                    dot_product_with(kernel, &a, &b),
                    Err(NexusError::KernelUnavailable(k)) if k == kernel
                ));
                assert!(matches!(
                    cosine_similarity_with(kernel, &a, &b),
                    Err(NexusError::KernelUnavailable(_))
                ));
                continue;
            }
            let dot = f64::from(dot_product_with(kernel, &a, &b).unwrap());
            assert!(
                (dot - expected_dot).abs() <= tolerance,
                "{kernel} dot, len {len}: {dot} vs {expected_dot}"
            );
            let cos = f64::from(cosine_similarity_with(kernel, &a, &b).unwrap());
            assert!(
                (cos - expected_cos).abs() <= 1e-5,
                "{kernel} cosine, len {len}: {cos} vs {expected_cos}"
            );
        }

        // The dispatching entry points use the active kernel.
        assert_eq!(
            dot_product(&a, &b).unwrap(),
            dot_product_with(Kernel::active(), &a, &b).unwrap()
        );
    }
}

#[test]
fn kernels_are_exact_on_integer_valued_inputs() {
    // Small integers are exactly representable, so every kernel, whatever its
    // summation order or use of FMA, must return the exact answer.
    for len in 1..=100_usize {
        let a: Vec<f32> = (0..len).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..len).map(|i| (i % 5) as f32 - 2.0).collect();
        let expected = reference_dot(&a, &b) as f32;
        for kernel in Kernel::ALL.into_iter().filter(|k| k.is_available()) {
            assert_eq!(
                dot_product_with(kernel, &a, &b).unwrap(),
                expected,
                "{kernel}, len {len}"
            );
        }
    }
}

#[test]
fn cosine_similarity_is_mathematically_correct() {
    let close = |actual: f32, expected: f32| (actual - expected).abs() < 1e-6;

    // Known value: (1,2,3).(4,5,6) = 32, norms sqrt(14) and sqrt(77).
    let value = cosine_similarity(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
    assert!(close(value, 0.974_631_85));
    assert_eq!(
        dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap(),
        32.0
    );

    let v = [0.3, -1.2, 4.5, 0.0, 2.2, -0.7, 9.1, 3.3, -5.5];
    let negated: Vec<f32> = v.iter().map(|x| -x).collect();
    let scaled: Vec<f32> = v.iter().map(|x| x * 1234.5).collect();
    assert!(close(cosine_similarity(&v, &v).unwrap(), 1.0));
    assert!(close(cosine_similarity(&v, &negated).unwrap(), -1.0));
    assert!(close(cosine_similarity(&v, &scaled).unwrap(), 1.0));
    assert!(close(
        cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap(),
        0.0
    ));
    assert!(close(
        cosine_similarity(&[1.0, 0.0], &[1.0, 1.0]).unwrap(),
        std::f32::consts::FRAC_1_SQRT_2
    ));

    // Symmetry, range and the distance identity on random data.
    let mut rng = Lcg(7);
    for _ in 0..200 {
        let a = rng.vector(37);
        let b = rng.vector(37);
        let ab = cosine_similarity(&a, &b).unwrap();
        assert_eq!(ab, cosine_similarity(&b, &a).unwrap());
        assert!((-1.0..=1.0).contains(&ab));
        assert_eq!(cosine_distance(&a, &b).unwrap(), 1.0 - ab);
    }
    assert!(close(cosine_distance(&v, &v).unwrap(), 0.0));
    assert!(close(cosine_distance(&v, &negated).unwrap(), 2.0));
}

#[test]
fn kernels_reject_invalid_input() {
    let ok = [1.0_f32, 2.0, 3.0];
    for kernel in Kernel::ALL.into_iter().filter(|k| k.is_available()) {
        assert!(matches!(
            dot_product_with(kernel, &ok, &[1.0, 2.0]),
            Err(NexusError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
        assert!(matches!(
            cosine_similarity_with(kernel, &[1.0], &ok),
            Err(NexusError::DimensionMismatch {
                expected: 1,
                actual: 3
            })
        ));
        assert!(matches!(
            dot_product_with(kernel, &[], &[]),
            Err(NexusError::ZeroDimension)
        ));
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // Poison every position, so both SIMD bodies and scalar tails are hit.
            for position in 0..21 {
                let mut bad = [1.0_f32; 21];
                bad[position] = poison;
                assert!(matches!(
                    dot_product_with(kernel, &bad, &[1.0; 21]),
                    Err(NexusError::NonFiniteVector)
                ));
                assert!(matches!(
                    cosine_similarity_with(kernel, &[1.0; 21], &bad),
                    Err(NexusError::NonFiniteVector)
                ));
            }
        }
        assert!(matches!(
            cosine_similarity_with(kernel, &ok, &[0.0, 0.0, 0.0]),
            Err(NexusError::ZeroMagnitude)
        ));
        // Finite inputs whose product overflows f32 are reported, not returned as inf.
        assert!(matches!(
            dot_product_with(kernel, &[f32::MAX, f32::MAX], &[f32::MAX, f32::MAX]),
            Err(NexusError::NumericOverflow)
        ));
    }
}

// ---------------------------------------------------------------------------
// Vector store
// ---------------------------------------------------------------------------

#[test]
fn vector_store_ranks_by_cosine_similarity() {
    let mut store = VectorStore::new(3).unwrap();
    store.insert(EntityId(10), &[1.0, 0.0, 0.0]).unwrap();
    store.insert(EntityId(20), &[0.0, 5.0, 0.0]).unwrap();
    store.insert(EntityId(30), &[3.0, 3.0, 0.0]).unwrap();
    store.insert(EntityId(40), &[-2.0, 0.0, 0.0]).unwrap();
    assert_eq!(store.len(), 4);
    assert_eq!(store.dim(), 3);

    let hits = store.search(&[10.0, 1.0, 0.0], 4).unwrap();
    let order: Vec<u64> = hits.iter().map(|h| h.id.0).collect();
    assert_eq!(order, [10, 30, 20, 40]);
    for hit in &hits {
        assert!((hit.distance - (1.0 - hit.similarity)).abs() < 1e-7);
    }
    assert!((hits[0].similarity - 0.995_037_2).abs() < 1e-6);
    assert!((hits[3].similarity + 0.995_037_2).abs() < 1e-6);

    // Magnitude does not affect ranking; it is kept alongside the unit vector.
    assert_eq!(store.norm(EntityId(20)), Some(5.0));
    assert_eq!(store.get(EntityId(20)), Some(&[0.0, 1.0, 0.0][..]));
    assert_eq!(store.get(EntityId(99)), None);

    // Bounded selection.
    assert_eq!(store.search(&[10.0, 1.0, 0.0], 2).unwrap(), hits[..2]);
    assert_eq!(store.search(&[10.0, 1.0, 0.0], 100).unwrap(), hits);
    assert!(store.search(&[10.0, 1.0, 0.0], 0).unwrap().is_empty());
}

#[test]
fn vector_store_breaks_ties_by_id() {
    let mut store = VectorStore::new(2).unwrap();
    for id in [5, 3, 9, 1, 7] {
        store.insert(EntityId(id), &[2.0, 2.0]).unwrap();
    }
    let hits = store.search(&[1.0, 1.0], 3).unwrap();
    let order: Vec<u64> = hits.iter().map(|h| h.id.0).collect();
    assert_eq!(order, [1, 3, 5]);
}

#[test]
fn vector_store_matches_brute_force_reference() {
    const DIM: usize = 33;
    const ROWS: usize = 500;
    let mut rng = Lcg(42);
    let flat: Vec<f32> = (0..DIM * ROWS).map(|_| rng.next_f32()).collect();
    let row_ids: Vec<EntityId> = (0..ROWS as u64).map(EntityId).collect();

    let mut bulk = VectorStore::new(DIM).unwrap();
    bulk.bulk_load(row_ids.clone(), flat.clone()).unwrap();
    let mut incremental = VectorStore::new(DIM).unwrap();
    for (id, row) in row_ids.iter().zip(flat.chunks_exact(DIM)) {
        incremental.insert(*id, row).unwrap();
    }

    for _ in 0..20 {
        let query = rng.vector(DIM);
        let mut expected: Vec<(u64, f64)> = flat
            .chunks_exact(DIM)
            .enumerate()
            .map(|(i, row)| (i as u64, reference_cosine(&query, row)))
            .collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1));

        let hits = bulk.search(&query, 10).unwrap();
        assert_eq!(hits.len(), 10);
        for (hit, (id, similarity)) in hits.iter().zip(&expected) {
            assert_eq!(hit.id.0, *id);
            assert!((f64::from(hit.similarity) - similarity).abs() < 1e-5);
        }
        // Bulk loading and row-by-row insertion build the same index.
        assert_eq!(incremental.search(&query, 10).unwrap(), hits);
    }
}

#[test]
fn vector_store_rejects_invalid_input_without_side_effects() {
    assert!(matches!(
        VectorStore::new(0),
        Err(NexusError::ZeroDimension)
    ));

    let mut store = VectorStore::new(2).unwrap();
    store.insert(EntityId(1), &[1.0, 0.0]).unwrap();

    assert!(matches!(
        store.insert(EntityId(2), &[1.0, 0.0, 0.0]),
        Err(NexusError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    ));
    assert!(matches!(
        store.insert(EntityId(2), &[f32::NAN, 0.0]),
        Err(NexusError::NonFiniteVector)
    ));
    assert!(matches!(
        store.insert(EntityId(2), &[0.0, 0.0]),
        Err(NexusError::ZeroMagnitude)
    ));
    assert!(matches!(
        store.insert(EntityId(1), &[0.0, 1.0]),
        Err(NexusError::DuplicateDocument(EntityId(1)))
    ));
    assert!(matches!(
        store.search(&[1.0], 5),
        Err(NexusError::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        store.search(&[f32::INFINITY, 1.0], 5),
        Err(NexusError::NonFiniteVector)
    ));
    assert!(matches!(
        store.search(&[0.0, 0.0], 5),
        Err(NexusError::ZeroMagnitude)
    ));

    // A bulk load is all-or-nothing.
    assert!(matches!(
        store.bulk_load(ids(&[2, 3]), vec![1.0, 1.0, 1.0]),
        Err(NexusError::BufferLengthMismatch {
            expected: 4,
            actual: 3,
            ..
        })
    ));
    assert!(matches!(
        store.bulk_load(ids(&[2, 3]), vec![1.0, 1.0, 0.0, 0.0]),
        Err(NexusError::ZeroMagnitude)
    ));
    assert!(matches!(
        store.bulk_load(ids(&[2, 2]), vec![1.0, 1.0, 1.0, 1.0]),
        Err(NexusError::DuplicateDocument(EntityId(2)))
    ));
    assert!(matches!(
        store.bulk_load(ids(&[2, 1]), vec![1.0, 1.0, 1.0, 1.0]),
        Err(NexusError::DuplicateDocument(EntityId(1)))
    ));
    assert_eq!(store.len(), 1);
    assert!(!store.contains(EntityId(2)));

    // Appending to a non-empty store works too.
    store
        .bulk_load(ids(&[2, 3]), vec![0.0, 3.0, 1.0, 1.0])
        .unwrap();
    assert_eq!(store.len(), 3);
    assert_eq!(store.search(&[0.0, 1.0], 1).unwrap()[0].id, EntityId(2));
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[test]
fn tokenizer_normalizes_lowercases_and_strips() {
    let tokenizer = Tokenizer::default();
    let tokens = |text: &str| tokenizer.tokenize(text);

    assert_eq!(tokens("The Quick, brown fox!"), ["quick", "brown", "fox"]);
    assert_eq!(
        tokens("auth-service_v2.1 (prod)"),
        ["auth", "service", "v2", "1", "prod"]
    );
    // Accents, ligatures, full-width forms and non-Latin scripts.
    assert_eq!(
        tokens("Café NAÏVE ﬁle ＡＢＣ１２３"),
        ["cafe", "naive", "file", "abc123"]
    );
    assert_eq!(tokens("İstanbul"), ["istanbul"]);
    assert_eq!(tokens("Привет, МИР!"), ["привет", "мир"]);
    // Precomposed and decomposed spellings produce the same token.
    assert_eq!(tokens("caf\u{e9}"), tokens("cafe\u{301}"));
    // Apostrophes join rather than split.
    assert_eq!(
        tokens("don't stop, it\u{2019}s fine"),
        ["dont", "stop", "fine"]
    );

    assert!(tokens("").is_empty());
    assert!(tokens("  \t\n ... --- !!! ").is_empty());
    assert!(tokens("the and of to is").is_empty());

    // Over-long tokens are dropped.
    let blob = "a".repeat(65);
    assert_eq!(tokens(&format!("keep {blob} this-also")), ["keep", "also"]);
}

#[test]
fn tokenizer_configuration_is_respected() {
    assert!(is_stop_word("the"));
    assert!(!is_stop_word("graph"));

    let keep_all = Tokenizer {
        filter_stop_words: false,
        ..Tokenizer::default()
    };
    assert_eq!(keep_all.tokenize("The graph"), ["the", "graph"]);

    let long_only = Tokenizer {
        min_token_len: 4,
        ..Tokenizer::default()
    };
    assert_eq!(long_only.tokenize("a db for graphs"), ["graphs"]);

    // The streaming form yields the same tokens without collecting.
    let mut streamed = Vec::new();
    Tokenizer::default().for_each_token("Hybrid   vector search", |t| streamed.push(t.len()));
    assert_eq!(streamed, [6, 6, 6]);
}

// ---------------------------------------------------------------------------
// BM25
// ---------------------------------------------------------------------------

/// d1 has 3 tokens, d2 has 3, d3 has 6 ("for" is a stop word): avgdl = 4.
fn small_corpus() -> Bm25Index {
    let mut index = Bm25Index::default();
    index
        .add_document(EntityId(1), "Rust graph engine")
        .unwrap();
    index
        .add_document(EntityId(2), "Graph, graph database!")
        .unwrap();
    index
        .add_document(EntityId(3), "vector search engine for Rust rust RUST")
        .unwrap();
    index
}

/// Independent transcription of the BM25 formula.
fn bm25_term(n_docs: f64, n_t: f64, tf: f64, doc_len: f64, avgdl: f64) -> f64 {
    let (k1, b) = (1.2, 0.75);
    let idf = (1.0 + (n_docs - n_t + 0.5) / (n_t + 0.5)).ln();
    idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * doc_len / avgdl))
}

#[test]
fn bm25_scores_match_the_formula() {
    let index = small_corpus();
    assert_eq!(index.len(), 3);
    assert_eq!(index.document_length(EntityId(1)), Some(3));
    assert_eq!(index.document_length(EntityId(3)), Some(6));
    assert_eq!(index.average_document_length(), 4.0);
    assert_eq!(index.document_frequency("graph"), 2);
    assert_eq!(index.document_frequency("rust"), 2);
    assert_eq!(index.document_frequency("database"), 1);
    assert!((index.idf("graph") - 1.6_f64.ln()).abs() < 1e-12);
    assert!((index.idf("database") - 2.666_666_666_666_666_5_f64.ln()).abs() < 1e-12);

    // "graph": d2 has tf 2, d1 has tf 1, both of length 3.
    let hits = index.search("graph", 10);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, EntityId(2));
    assert_eq!(hits[1].id, EntityId(1));
    assert!((hits[0].score - bm25_term(3.0, 2.0, 2.0, 3.0, 4.0)).abs() < 1e-12);
    assert!((hits[1].score - bm25_term(3.0, 2.0, 1.0, 3.0, 4.0)).abs() < 1e-12);
    // Hand-computed literals.
    assert!((hits[0].score - 0.695_131_418).abs() < 1e-8);
    assert!((hits[1].score - 0.523_548_347).abs() < 1e-8);

    // Multi-term query: scores add up per document.
    let hits = index.search("rust engine", 10);
    let d1 = bm25_term(3.0, 2.0, 1.0, 3.0, 4.0) * 2.0;
    let d3 = bm25_term(3.0, 2.0, 3.0, 6.0, 4.0) + bm25_term(3.0, 2.0, 1.0, 6.0, 4.0);
    assert_eq!(hits.len(), 2);
    let by_id = |id: u64| hits.iter().find(|h| h.id == EntityId(id)).unwrap().score;
    assert!((by_id(1) - d1).abs() < 1e-12);
    assert!((by_id(3) - d3).abs() < 1e-12);
    assert!(hits[0].score >= hits[1].score);

    // Query-side normalization matches document-side normalization.
    assert_eq!(index.search("GRAPH!!", 10), index.search("graph", 10));
    // A repeated query term counts once.
    assert_eq!(
        index.search("graph graph graph", 10),
        index.search("graph", 10)
    );
}

#[test]
fn bm25_handles_edge_cases() {
    let index = small_corpus();

    assert!(index.search("", 10).is_empty());
    assert!(index.search("   ?!  ", 10).is_empty());
    assert!(index.search("the of and", 10).is_empty(), "stop words only");
    assert!(
        index.search("kubernetes zookeeper", 10).is_empty(),
        "unseen tokens"
    );
    assert!(index.search("graph", 0).is_empty());
    assert_eq!(index.document_frequency("kubernetes"), 0);
    assert!(index.idf("kubernetes") > index.idf("graph"));

    // Unseen tokens neither help nor hurt the terms that do match.
    assert_eq!(
        index.search("kubernetes graph", 10),
        index.search("graph", 10)
    );
    // top_k bounds the result.
    assert_eq!(index.search("graph", 1).len(), 1);
    assert_eq!(index.search("graph", 1)[0].id, EntityId(2));

    // Empty index.
    let empty = Bm25Index::default();
    assert!(empty.is_empty());
    assert!(empty.search("graph", 10).is_empty());
    assert_eq!(empty.average_document_length(), 0.0);

    // Duplicate ids are rejected; documents without tokens still count in N.
    let mut index = small_corpus();
    assert!(matches!(
        index.add_document(EntityId(1), "again"),
        Err(NexusError::DuplicateDocument(EntityId(1)))
    ));
    let idf_before = index.idf("database");
    index.add_document(EntityId(4), "... the ...").unwrap();
    assert_eq!(index.len(), 4);
    assert_eq!(index.document_length(EntityId(4)), Some(0));
    assert!(index.idf("database") > idf_before);
    assert!(index
        .search("database", 10)
        .iter()
        .all(|h| h.id != EntityId(4)));

    // Invalid parameters.
    for params in [
        Bm25Params { k1: -1.0, b: 0.75 },
        Bm25Params {
            k1: f64::NAN,
            b: 0.75,
        },
        Bm25Params { k1: 1.2, b: 1.5 },
        Bm25Params {
            k1: 1.2,
            b: f64::NAN,
        },
    ] {
        assert!(matches!(
            Bm25Index::new(Tokenizer::default(), params),
            Err(NexusError::InvalidConfig(_))
        ));
    }
}

#[test]
fn bm25_normalizes_by_document_length_and_saturates_term_frequency() {
    let mut index = Bm25Index::default();
    index.add_document(EntityId(1), "cache").unwrap();
    index
        .add_document(EntityId(2), "cache layer sits front database tier")
        .unwrap();
    index
        .add_document(EntityId(3), "cache cache cache cache cache cache")
        .unwrap();
    index
        .add_document(EntityId(4), "unrelated words entirely")
        .unwrap();

    let hits = index.search("cache", 10);
    let score = |id: u64| hits.iter().find(|h| h.id == EntityId(id)).unwrap().score;
    // Same term frequency: the shorter document wins.
    assert!(score(1) > score(2));
    // Same length: more occurrences win, but with diminishing returns
    // (bounded by k1 + 1 times the single-occurrence ceiling).
    assert!(score(3) > score(2));
    assert!(score(3) < index.idf("cache") * 2.2);

    // Equal documents tie and are ordered by id.
    let mut twins = Bm25Index::default();
    for id in [9, 4, 6] {
        twins.add_document(EntityId(id), "twin text").unwrap();
    }
    let order: Vec<u64> = twins.search("twin", 10).iter().map(|h| h.id.0).collect();
    assert_eq!(order, [4, 6, 9]);
}

// ---------------------------------------------------------------------------
// Reciprocal Rank Fusion
// ---------------------------------------------------------------------------

#[test]
fn rrf_scores_match_the_formula() {
    let a = ids(&[1, 2, 3]);
    let b = ids(&[3, 1, 4]);
    let lists = [
        RankedList {
            weight: 1.0,
            ids: &a,
        },
        RankedList {
            weight: 1.0,
            ids: &b,
        },
    ];
    let fused = reciprocal_rank_fusion(&lists, DEFAULT_RRF_K).unwrap();

    let order: Vec<u64> = fused.iter().map(|h| h.id.0).collect();
    assert_eq!(order, [1, 3, 2, 4]);
    let expected = [
        1.0 / 61.0 + 1.0 / 62.0,
        1.0 / 63.0 + 1.0 / 61.0,
        1.0 / 62.0,
        1.0 / 63.0,
    ];
    for (hit, expected) in fused.iter().zip(expected) {
        assert!((hit.score - expected).abs() < 1e-15);
    }

    // Weights scale each channel's contribution.
    let weighted = [
        RankedList {
            weight: 0.25,
            ids: &a,
        },
        RankedList {
            weight: 2.0,
            ids: &b,
        },
    ];
    let fused = reciprocal_rank_fusion(&weighted, 10.0).unwrap();
    let order: Vec<u64> = fused.iter().map(|h| h.id.0).collect();
    assert_eq!(order, [3, 1, 4, 2]);
    assert!((fused[0].score - (0.25 / 13.0 + 2.0 / 11.0)).abs() < 1e-15);
    assert!((fused[3].score - 0.25 / 12.0).abs() < 1e-15);
}

#[test]
fn rrf_ranking_is_stable() {
    // Perfectly symmetric input: equal scores, order decided by id.
    let a = ids(&[7, 2]);
    let b = ids(&[2, 7]);
    let symmetric = [
        RankedList {
            weight: 1.0,
            ids: &a,
        },
        RankedList {
            weight: 1.0,
            ids: &b,
        },
    ];
    let fused = reciprocal_rank_fusion(&symmetric, 60.0).unwrap();
    assert_eq!(fused[0].score, fused[1].score);
    assert_eq!([fused[0].id, fused[1].id], [EntityId(2), EntityId(7)]);

    // Repeated runs are bit-for-bit identical (no dependence on hash order).
    let mut rng = Lcg(99);
    let big_a: Vec<EntityId> = (0..300)
        .map(|_| EntityId(u64::from(rng.next_f32().to_bits() % 400)))
        .collect();
    let big_b: Vec<EntityId> = (0..300)
        .map(|_| EntityId(u64::from(rng.next_f32().to_bits() % 400)))
        .collect();
    let lists = [
        RankedList {
            weight: 1.0,
            ids: &big_a,
        },
        RankedList {
            weight: 0.7,
            ids: &big_b,
        },
    ];
    let first = reciprocal_rank_fusion(&lists, 60.0).unwrap();
    for _ in 0..20 {
        assert_eq!(reciprocal_rank_fusion(&lists, 60.0).unwrap(), first);
    }
    // Output is sorted, and holds each id once.
    assert!(first
        .windows(2)
        .all(|w| w[0].score > w[1].score || (w[0].score == w[1].score && w[0].id < w[1].id)));
    let unique: BTreeSet<EntityId> = first.iter().map(|h| h.id).collect();
    assert_eq!(unique.len(), first.len());

    // A single list is returned in its own order: fusion never reorders it.
    let single = ids(&[5, 1, 9, 3]);
    let fused = reciprocal_rank_fusion(
        &[RankedList {
            weight: 1.0,
            ids: &single,
        }],
        60.0,
    )
    .unwrap();
    assert_eq!(fused.iter().map(|h| h.id).collect::<Vec<_>>(), single);

    // Agreement beats a single first place: an item ranked 2nd by both lists
    // outranks items ranked 1st by only one.
    let x = ids(&[1, 3]);
    let y = ids(&[2, 3]);
    let fused = reciprocal_rank_fusion(
        &[
            RankedList {
                weight: 1.0,
                ids: &x,
            },
            RankedList {
                weight: 1.0,
                ids: &y,
            },
        ],
        60.0,
    )
    .unwrap();
    assert_eq!(fused[0].id, EntityId(3));

    // A duplicated id only counts at its best position.
    let duplicated = ids(&[4, 4, 8]);
    let fused = reciprocal_rank_fusion(
        &[RankedList {
            weight: 1.0,
            ids: &duplicated,
        }],
        60.0,
    )
    .unwrap();
    assert_eq!(fused.len(), 2);
    assert!((fused[1].score - 1.0 / 62.0).abs() < 1e-15);
}

#[test]
fn rrf_rejects_invalid_parameters_and_handles_empty_input() {
    let a = ids(&[1]);
    let list = |weight| [RankedList { weight, ids: &a }];
    for k in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            reciprocal_rank_fusion(&list(1.0), k),
            Err(NexusError::InvalidConfig(_))
        ));
    }
    for weight in [-0.5, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            reciprocal_rank_fusion(&list(weight), 60.0),
            Err(NexusError::InvalidConfig(_))
        ));
    }
    assert!(reciprocal_rank_fusion(&[], 60.0).unwrap().is_empty());
    assert!(reciprocal_rank_fusion(
        &[RankedList {
            weight: 1.0,
            ids: &[]
        }],
        60.0
    )
    .unwrap()
    .is_empty());
    // k = 0 is legal: ranks are 1-based, so the denominator stays positive.
    assert_eq!(
        reciprocal_rank_fusion(&list(1.0), 0.0).unwrap()[0].score,
        1.0
    );
}

// ---------------------------------------------------------------------------
// End-to-end hybrid retrieval
// ---------------------------------------------------------------------------

const ACME: &str = "Acme Corp";
const BILLING: &str = "billing-service";
const AUTH: &str = "auth-service";
const POSTGRES: &str = "postgres";
const STORAGE: &str = "object-storage";
const REDIS: &str = "redis";
const ON_CALL: &str = "on-call rotation";

/// Embedding axes: [payments, identity, storage, organization].
fn knowledge_graph() -> Arc<KnowledgeGraph> {
    let graph = KnowledgeGraph::new();
    let entities: [(&str, EntityType, &str, Option<[f32; 4]>); 7] = [
        (
            ACME,
            EntityType::Organization,
            "Parent company operating the payments platform",
            Some([0.1, 0.1, 0.1, 0.9]),
        ),
        (
            BILLING,
            EntityType::Service,
            "Generates invoices and charges customer credit cards",
            Some([0.9, 0.1, 0.2, 0.1]),
        ),
        (
            AUTH,
            EntityType::Service,
            "Issues login tokens and validates user sessions",
            Some([0.1, 0.9, 0.1, 0.1]),
        ),
        (
            POSTGRES,
            EntityType::Service,
            "Relational database storing user accounts, sessions and invoices",
            Some([0.3, 0.4, 0.8, 0.1]),
        ),
        (
            STORAGE,
            EntityType::Service,
            "Durable blob storage for database backups",
            Some([0.2, 0.1, 0.9, 0.2]),
        ),
        (
            REDIS,
            EntityType::Service,
            "In-memory cache for session tokens",
            Some([0.1, 0.7, 0.6, 0.1]),
        ),
        (
            ON_CALL,
            EntityType::Custom("Process".into()),
            "Weekly schedule of engineers paged for incidents",
            None,
        ),
    ];
    for (name, entity_type, description, embedding) in entities {
        let mut attributes = Attributes::new();
        attributes.insert("description".into(), description.into());
        graph
            .add_entity(name, entity_type, attributes, embedding.map(Vec::from))
            .unwrap();
    }
    let relations = [
        (BILLING, AUTH, RelationType::DependsOn, 0.9),
        (BILLING, POSTGRES, RelationType::DependsOn, 0.95),
        (AUTH, POSTGRES, RelationType::DependsOn, 0.8),
        (AUTH, REDIS, RelationType::DependsOn, 0.9),
        (POSTGRES, STORAGE, RelationType::DependsOn, 0.5),
        (ACME, BILLING, RelationType::Contains, 1.0),
        (ACME, AUTH, RelationType::Contains, 1.0),
    ];
    for (source, target, relation, weight) in relations {
        graph
            .add_relation_by_name(source, target, relation, weight, Attributes::new())
            .unwrap();
    }
    Arc::new(graph)
}

fn retriever(config: RetrievalConfig) -> HybridRetriever {
    HybridRetriever::from_graph(knowledge_graph(), config).unwrap()
}

fn hit_names(bundle: &nexus_graph::RetrievalBundle) -> Vec<&str> {
    bundle.hits.iter().map(|h| h.name.as_str()).collect()
}

fn context_names(bundle: &nexus_graph::RetrievalBundle) -> BTreeSet<(&str, usize)> {
    bundle
        .context
        .nodes
        .iter()
        .map(|n| (n.entity.canonical_name.as_str(), n.depth))
        .collect()
}

const IDENTITY_QUERY: [f32; 4] = [0.0, 1.0, 0.0, 0.0];

#[test]
fn hybrid_search_fuses_channels_and_expands_graph_context() {
    let config = RetrievalConfig {
        top_k: 2,
        expansion_hops: 1,
        ..RetrievalConfig::default()
    };
    let retriever = retriever(config);
    let stats = retriever.stats();
    assert_eq!(
        (stats.documents, stats.vectors, stats.vector_dim),
        (7, 6, Some(4))
    );

    let bundle = retriever
        .search(&HybridQuery {
            text: Some("Login tokens?"),
            embedding: Some(&IDENTITY_QUERY),
        })
        .unwrap();

    // Both channels rank auth first and redis second.
    assert_eq!(hit_names(&bundle), [AUTH, REDIS]);
    let (auth, redis) = (&bundle.hits[0], &bundle.hits[1]);
    assert_eq!(auth.entity_type, EntityType::Service);
    assert!((auth.fused_score - 2.0 / 61.0).abs() < 1e-15);
    assert!((redis.fused_score - 2.0 / 62.0).abs() < 1e-15);
    assert_eq!(auth.vector.unwrap().rank, 1);
    assert_eq!(auth.lexical.unwrap().rank, 1);
    assert_eq!(redis.vector.unwrap().rank, 2);
    assert_eq!(redis.lexical.unwrap().rank, 2);
    // Raw channel scores are carried through: cos = 0.9 / sqrt(0.84).
    assert!((auth.vector.unwrap().score - 0.9 / 0.84_f64.sqrt()).abs() < 1e-6);
    assert!(auth.lexical.unwrap().score > redis.lexical.unwrap().score);

    // 1-hop context around {auth, redis}: exact node and edge coverage.
    let expected_nodes: BTreeSet<(&str, usize)> = [
        (AUTH, 0),
        (REDIS, 0),
        (ACME, 1),
        (BILLING, 1),
        (POSTGRES, 1),
    ]
    .into();
    assert_eq!(context_names(&bundle), expected_nodes);
    assert_eq!(bundle.context.edges.len(), 6);
    let direct = retriever.graph().get_k_hop_subgraph(
        &[
            retriever.graph().node_index_by_name(AUTH).unwrap(),
            retriever.graph().node_index_by_name(REDIS).unwrap(),
        ],
        1,
    );
    assert_eq!(bundle.context, direct);

    // Expansion depth is configurable per query.
    let query = HybridQuery {
        text: Some("login tokens"),
        embedding: Some(&IDENTITY_QUERY),
    };
    let flat = retriever
        .search_with(
            &query,
            &RetrievalConfig {
                expansion_hops: 0,
                ..config
            },
        )
        .unwrap();
    assert_eq!(context_names(&flat), [(AUTH, 0), (REDIS, 0)].into());
    assert_eq!(flat.context.edges.len(), 1);
    let wide = retriever
        .search_with(
            &query,
            &RetrievalConfig {
                expansion_hops: 2,
                ..config
            },
        )
        .unwrap();
    assert_eq!(wide.context.nodes.len(), 6);
    assert_eq!(wide.context.edges.len(), 7);
    assert!(context_names(&wide).contains(&(STORAGE, 2)));

    // Deterministic.
    assert_eq!(
        retriever.search(&query).unwrap(),
        retriever.search(&query).unwrap()
    );
}

#[test]
fn hybrid_search_works_with_a_single_channel() {
    let retriever = retriever(RetrievalConfig::default());

    // Text only: entities are found by description, type or name.
    let bundle = retriever
        .search(&HybridQuery {
            text: Some("invoices"),
            embedding: None,
        })
        .unwrap();
    let found: BTreeSet<&str> = hit_names(&bundle).into_iter().collect();
    assert_eq!(found, [BILLING, POSTGRES].into());
    assert!(bundle
        .hits
        .iter()
        .all(|h| h.vector.is_none() && h.lexical.is_some()));
    let by_type = retriever
        .search(&HybridQuery {
            text: Some("organization"),
            embedding: None,
        })
        .unwrap();
    assert_eq!(hit_names(&by_type), [ACME]);

    // Vector only: every embedded entity is ranked by cosine similarity.
    let bundle = retriever
        .search(&HybridQuery {
            text: None,
            embedding: Some(&IDENTITY_QUERY),
        })
        .unwrap();
    assert_eq!(
        hit_names(&bundle),
        [AUTH, REDIS, POSTGRES, ACME, BILLING, STORAGE]
    );
    assert!(bundle.hits.iter().all(|h| h.lexical.is_none()));
    let similarities: Vec<f64> = bundle
        .hits
        .iter()
        .map(|h| h.vector.unwrap().score)
        .collect();
    assert!(similarities.windows(2).all(|w| w[0] >= w[1]));

    // An entity without an embedding is still reachable through text, and
    // reported as missing from the vector channel.
    let bundle = retriever
        .search(&HybridQuery {
            text: Some("who is paged for incidents"),
            embedding: Some(&IDENTITY_QUERY),
        })
        .unwrap();
    let on_call = bundle.hits.iter().find(|h| h.name == ON_CALL).unwrap();
    assert!(on_call.vector.is_none());
    assert_eq!(on_call.lexical.unwrap().rank, 1);

    // A text query that matches nothing falls back to the vector ranking.
    let bundle = retriever
        .search(&HybridQuery {
            text: Some("kubernetes"),
            embedding: Some(&IDENTITY_QUERY),
        })
        .unwrap();
    assert_eq!(bundle.hits[0].name, AUTH);

    // Nothing matches at all: empty bundle, not an error.
    let bundle = retriever
        .search(&HybridQuery {
            text: Some("kubernetes"),
            embedding: None,
        })
        .unwrap();
    assert!(bundle.hits.is_empty());
    assert!(bundle.context.nodes.is_empty());
}

#[test]
fn channel_weights_steer_the_fusion() {
    // Text points at the databases, the embedding at identity services.
    let query = HybridQuery {
        text: Some("database"),
        embedding: Some(&IDENTITY_QUERY),
    };
    let base = RetrievalConfig::default();
    let retriever = retriever(base);

    let vector_only = retriever
        .search_with(
            &query,
            &RetrievalConfig {
                lexical_weight: 0.0,
                ..base
            },
        )
        .unwrap();
    assert_eq!(vector_only.hits[0].name, AUTH);
    assert!(vector_only.hits.iter().all(|h| h.lexical.is_none()));

    let lexical_only = retriever
        .search_with(
            &query,
            &RetrievalConfig {
                vector_weight: 0.0,
                ..base
            },
        )
        .unwrap();
    let names: BTreeSet<&str> = hit_names(&lexical_only).into_iter().collect();
    assert_eq!(names, [POSTGRES, STORAGE].into());

    let lexical_heavy = retriever
        .search_with(
            &query,
            &RetrievalConfig {
                lexical_weight: 5.0,
                ..base
            },
        )
        .unwrap();
    let top_two: BTreeSet<&str> = hit_names(&lexical_heavy).into_iter().take(2).collect();
    assert_eq!(top_two, [POSTGRES, STORAGE].into());

    // A dominant vector weight restores the vector channel's favourite, but
    // the entity both channels agree on still beats the vector runner-up.
    let vector_heavy = retriever
        .search_with(
            &query,
            &RetrievalConfig {
                vector_weight: 50.0,
                ..base
            },
        )
        .unwrap();
    assert_eq!(hit_names(&vector_heavy)[..3], [AUTH, POSTGRES, REDIS]);

    // Balanced: postgres is the only entity both channels rank highly
    // (vector #3, lexical #1 or #2), so agreement puts it first.
    let balanced = retriever.search(&query).unwrap();
    assert_eq!(balanced.hits[0].name, POSTGRES);
    assert!(balanced.hits[0].vector.is_some() && balanced.hits[0].lexical.is_some());
}

#[test]
fn hybrid_search_reports_invalid_requests() {
    let retriever = retriever(RetrievalConfig::default());

    assert!(matches!(
        retriever.search(&HybridQuery::default()),
        Err(NexusError::EmptyQuery)
    ));
    assert!(matches!(
        retriever.search(&HybridQuery {
            text: None,
            embedding: Some(&[1.0, 0.0])
        }),
        Err(NexusError::DimensionMismatch {
            expected: 4,
            actual: 2
        })
    ));
    assert!(matches!(
        retriever.search(&HybridQuery {
            text: None,
            embedding: Some(&[f32::NAN, 0.0, 0.0, 0.0])
        }),
        Err(NexusError::NonFiniteVector)
    ));

    let invalid = [
        RetrievalConfig {
            top_k: 0,
            ..RetrievalConfig::default()
        },
        RetrievalConfig {
            rrf_k: -1.0,
            ..RetrievalConfig::default()
        },
        RetrievalConfig {
            vector_weight: f64::NAN,
            ..RetrievalConfig::default()
        },
        RetrievalConfig {
            vector_weight: 0.0,
            lexical_weight: 0.0,
            ..RetrievalConfig::default()
        },
    ];
    for config in invalid {
        assert!(matches!(
            config.validate(),
            Err(NexusError::InvalidConfig(_))
        ));
        assert!(matches!(
            HybridRetriever::new(knowledge_graph(), config),
            Err(NexusError::InvalidConfig(_))
        ));
        assert!(matches!(
            retriever.search_with(
                &HybridQuery {
                    text: Some("x"),
                    embedding: None
                },
                &config
            ),
            Err(NexusError::InvalidConfig(_))
        ));
    }
}

#[test]
fn entities_added_later_are_indexed_incrementally() {
    let graph = knowledge_graph();
    let mut retriever =
        HybridRetriever::from_graph(Arc::clone(&graph), RetrievalConfig::default()).unwrap();
    let query = HybridQuery {
        text: Some("kafka event stream"),
        embedding: None,
    };
    assert!(retriever.search(&query).unwrap().hits.is_empty());

    let mut attributes = Attributes::new();
    attributes.insert("description".into(), "Event stream broker".into());
    let node = graph
        .add_entity(
            "kafka",
            EntityType::Service,
            attributes,
            Some(vec![0.5, 0.1, 0.5, 0.1]),
        )
        .unwrap();
    graph
        .add_relation_by_name(
            BILLING,
            "kafka",
            RelationType::DependsOn,
            0.7,
            Attributes::new(),
        )
        .unwrap();
    let id = graph.node(node).unwrap().id;

    retriever.index_entity(id).unwrap();
    let bundle = retriever.search(&query).unwrap();
    assert_eq!(hit_names(&bundle), ["kafka"]);
    assert_eq!(context_names(&bundle), [("kafka", 0), (BILLING, 1)].into());
    assert_eq!(retriever.stats().vectors, 7);

    assert!(matches!(
        retriever.index_entity(id),
        Err(NexusError::DuplicateDocument(_))
    ));
    assert!(matches!(
        retriever.index_entity(EntityId(9999)),
        Err(NexusError::Graph(_))
    ));
}

#[test]
fn retrieval_is_identical_after_a_snapshot_roundtrip() {
    let config = RetrievalConfig {
        top_k: 3,
        expansion_hops: 2,
        ..RetrievalConfig::default()
    };
    let original = retriever(config);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.ngsnap");
    save_snapshot(original.graph(), &path).unwrap();
    let restored =
        HybridRetriever::from_graph(Arc::new(load_snapshot(&path).unwrap()), config).unwrap();
    assert_eq!(restored.stats(), original.stats());

    let queries = [
        HybridQuery {
            text: Some("user sessions database"),
            embedding: Some(&[0.2, 0.5, 0.7, 0.0]),
        },
        HybridQuery {
            text: Some("invoices"),
            embedding: None,
        },
        HybridQuery {
            text: None,
            embedding: Some(&[0.9, 0.0, 0.1, 0.3]),
        },
    ];
    for query in &queries {
        let bundle = original.search(query).unwrap();
        assert!(!bundle.hits.is_empty());
        assert_eq!(restored.search(query).unwrap(), bundle);
        // The bundle is a plain serializable value.
        let json = serde_json::to_string(&bundle).unwrap();
        assert_eq!(
            serde_json::from_str::<nexus_graph::RetrievalBundle>(&json).unwrap(),
            bundle
        );
    }
}

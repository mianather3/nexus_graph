//! Integration and invariant tests for the graph, verifier and storage layers.

// `assert_is_empty` only exists in recent clippy releases.
#![allow(unknown_lints)]
#![allow(
    clippy::assert_is_empty,
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use std::collections::BTreeSet;

use nexus_graph::storage::{decode_snapshot, encode_snapshot};
use nexus_graph::{
    load_snapshot, save_snapshot, verify_claims, Attributes, CitationRef, EntityId, EntityNode,
    EntityType, EvidenceKind, GraphClaim, GraphError, KnowledgeGraph, NodeIndex, RejectionReason,
    RelationType, StorageError, Subgraph, VerifierConfig,
};

use RelationType::{Contains, DependsOn, ProducedBy, Supersedes};

const ACME: &str = "Acme Corp";
const BILLING: &str = "billing-service";
const AUTH: &str = "auth-service";
const POSTGRES: &str = "postgres";
const REDIS: &str = "redis";
const STORAGE: &str = "object-storage";
const PIPELINE: &str = "build-pipeline";
const IMAGE_V1: &str = "auth-image-v1";
const IMAGE_V2: &str = "auth-image-v2";
const ZERO_TRUST: &str = "zero-trust";

/// Fixture topology:
///
/// ```text
/// billing -D(0.9)-> auth -D(0.8)-> postgres -D(0.5)-> object-storage
///                   auth -D(1.0)-> redis
/// acme -Contains-> billing, auth
/// image-v2 -Supersedes-> image-v1
/// image-v2 -ProducedBy-> build-pipeline -ProducedBy-> acme
/// zero-trust (isolated)
/// ```
fn fixture() -> KnowledgeGraph {
    let graph = KnowledgeGraph::new();
    let entities = [
        (ACME, EntityType::Organization),
        (BILLING, EntityType::Service),
        (AUTH, EntityType::Service),
        (POSTGRES, EntityType::Service),
        (REDIS, EntityType::Service),
        (STORAGE, EntityType::Service),
        (PIPELINE, EntityType::Service),
        (IMAGE_V1, EntityType::Artifact),
        (IMAGE_V2, EntityType::Artifact),
        (ZERO_TRUST, EntityType::Concept),
    ];
    for (i, (name, entity_type)) in entities.into_iter().enumerate() {
        let mut attributes = Attributes::new();
        attributes.insert("origin".into(), "fixture".into());
        let embedding = Some(vec![i as f32, 1.0, -0.5]);
        graph
            .add_entity(name, entity_type, attributes, embedding)
            .unwrap();
    }
    let relations = [
        (BILLING, AUTH, DependsOn, 0.9),
        (AUTH, POSTGRES, DependsOn, 0.8),
        (AUTH, REDIS, DependsOn, 1.0),
        (POSTGRES, STORAGE, DependsOn, 0.5),
        (ACME, BILLING, Contains, 1.0),
        (ACME, AUTH, Contains, 1.0),
        (IMAGE_V2, IMAGE_V1, Supersedes, 1.0),
        (IMAGE_V2, PIPELINE, ProducedBy, 1.0),
        (PIPELINE, ACME, ProducedBy, 1.0),
    ];
    for (source, target, relation, weight) in relations {
        graph
            .add_relation_by_name(source, target, relation, weight, Attributes::new())
            .unwrap();
    }
    graph
}

fn idx(graph: &KnowledgeGraph, name: &str) -> NodeIndex {
    graph
        .node_index_by_name(name)
        .unwrap_or_else(|| panic!("fixture entity `{name}` missing"))
}

fn claim(source: &str, relation: RelationType, target: &str) -> GraphClaim {
    GraphClaim {
        source_entity: source.into(),
        target_entity: target.into(),
        relation,
        citation: CitationRef {
            document_id: "doc-1".into(),
            chunk_id: format!("{source}->{target}"),
        },
    }
}

fn node_names(subgraph: &Subgraph) -> BTreeSet<(String, usize)> {
    subgraph
        .nodes
        .iter()
        .map(|n| (n.entity.canonical_name.clone(), n.depth))
        .collect()
}

fn edge_triples(graph: &KnowledgeGraph, subgraph: &Subgraph) -> BTreeSet<(String, String, String)> {
    let name = |id: EntityId| {
        let index = graph.node_index_by_id(id).unwrap();
        graph.node(index).unwrap().canonical_name
    };
    subgraph
        .edges
        .iter()
        .map(|e| (name(e.source), e.relation.to_string(), name(e.target)))
        .collect()
}

fn expect_nodes(items: &[(&str, usize)]) -> BTreeSet<(String, usize)> {
    items.iter().map(|&(n, d)| (n.to_owned(), d)).collect()
}

fn expect_edges(items: &[(&str, &str, &str)]) -> BTreeSet<(String, String, String)> {
    items
        .iter()
        .map(|&(s, r, t)| (s.to_owned(), r.to_owned(), t.to_owned()))
        .collect()
}

// ---------------------------------------------------------------------------
// Graph construction and indices
// ---------------------------------------------------------------------------

#[test]
fn builds_multi_entity_graph_with_consistent_indices() {
    let graph = fixture();
    assert_eq!(graph.node_count(), 10);
    assert_eq!(graph.edge_count(), 9);

    // Name index is case- and whitespace-insensitive.
    let auth = idx(&graph, AUTH);
    assert_eq!(graph.node_index_by_name("  AUTH-Service "), Some(auth));
    assert_eq!(graph.node_index_by_name("no-such-entity"), None);

    // Id index agrees with the name index.
    let node = graph.node(auth).unwrap();
    assert_eq!(node.canonical_name, AUTH);
    assert_eq!(graph.node_index_by_id(node.id), Some(auth));

    // Type index partitions the node set.
    let services = graph.nodes_by_type(&EntityType::Service);
    assert_eq!(services.len(), 6);
    assert!(services.contains(&auth));
    assert_eq!(graph.nodes_by_type(&EntityType::Organization).len(), 1);
    assert_eq!(graph.nodes_by_type(&EntityType::Artifact).len(), 2);
    assert_eq!(graph.nodes_by_type(&EntityType::Concept).len(), 1);
    assert!(graph
        .nodes_by_type(&EntityType::Custom("Team".into()))
        .is_empty());
}

#[test]
fn rejects_mutations_that_break_invariants() {
    let graph = fixture();
    let auth = idx(&graph, AUTH);
    let redis = idx(&graph, REDIS);

    assert_eq!(
        graph.add_entity("AUTH-SERVICE", EntityType::Service, Attributes::new(), None),
        Err(GraphError::DuplicateName("AUTH-SERVICE".into()))
    );
    assert_eq!(
        graph.add_entity("   ", EntityType::Concept, Attributes::new(), None),
        Err(GraphError::EmptyName)
    );
    assert_eq!(
        graph.add_entity(
            "gpu",
            EntityType::Concept,
            Attributes::new(),
            Some(vec![1.0])
        ),
        Err(GraphError::EmbeddingDimensionMismatch {
            expected: 3,
            actual: 1
        })
    );
    assert_eq!(
        graph.add_entity(
            "nan",
            EntityType::Concept,
            Attributes::new(),
            Some(vec![f32::NAN, 0.0, 0.0])
        ),
        Err(GraphError::InvalidEmbedding)
    );
    let duplicate_id = EntityNode {
        id: graph.node(auth).unwrap().id,
        canonical_name: "fresh-name".into(),
        entity_type: EntityType::Service,
        attributes: Attributes::new(),
        embedding: None,
    };
    assert!(matches!(
        graph.insert_node(duplicate_id),
        Err(GraphError::DuplicateId(_))
    ));

    for bad_weight in [-0.1, 1.5, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            graph.add_relation(auth, redis, DependsOn, bad_weight, Attributes::new()),
            Err(GraphError::InvalidWeight(_))
        ));
    }
    assert_eq!(
        graph.add_relation(auth, NodeIndex::new(999), DependsOn, 1.0, Attributes::new()),
        Err(GraphError::NodeNotFound(999))
    );
    assert_eq!(
        graph.add_relation_by_name(AUTH, "kafka", DependsOn, 1.0, Attributes::new()),
        Err(GraphError::EntityNameNotFound("kafka".into()))
    );

    // Nothing above may have leaked into the graph.
    assert_eq!(graph.node_count(), 10);
    assert_eq!(graph.edge_count(), 9);
}

#[test]
fn re_adding_a_relation_updates_it_in_place() {
    let graph = fixture();
    let auth = idx(&graph, AUTH);
    let redis = idx(&graph, REDIS);

    let first = graph
        .add_relation(auth, redis, DependsOn, 0.25, Attributes::new())
        .unwrap();
    assert_eq!(
        graph.edge_count(),
        9,
        "same (source, target, relation) must not duplicate"
    );
    assert_eq!(
        graph.find_relation(auth, redis, &DependsOn).unwrap().weight,
        0.25
    );

    // A different relation between the same endpoints is a distinct edge.
    let second = graph
        .add_relation(auth, redis, Contains, 1.0, Attributes::new())
        .unwrap();
    assert_ne!(first, second);
    assert_eq!(graph.edge_count(), 10);
}

#[test]
fn handles_cyclic_relationships() {
    let graph = KnowledgeGraph::new();
    let a = graph
        .add_entity("a", EntityType::Service, Attributes::new(), None)
        .unwrap();
    let b = graph
        .add_entity("b", EntityType::Service, Attributes::new(), None)
        .unwrap();
    let c = graph
        .add_entity("c", EntityType::Service, Attributes::new(), None)
        .unwrap();
    for (s, t) in [(a, b), (b, c), (c, a)] {
        graph
            .add_relation(s, t, DependsOn, 1.0, Attributes::new())
            .unwrap();
    }

    // BFS terminates on a cycle and covers it exactly once, whatever k is.
    let subgraph = graph.get_k_hop_subgraph(&[a], usize::MAX);
    assert_eq!(subgraph.nodes.len(), 3);
    assert_eq!(subgraph.edges.len(), 3);

    // Path enumeration never revisits an entity.
    let paths = graph.find_connecting_paths(a, c, 50);
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].to_string(), "a -[DependsOn]-> b -[DependsOn]-> c");

    // Source == target yields the simple cycles through it.
    let cycles = graph.find_connecting_paths(a, a, 50);
    assert_eq!(cycles.len(), 1);
    assert_eq!(cycles[0].entity_names(), ["a", "b", "c", "a"]);
    assert!(graph.find_connecting_paths(a, a, 2).is_empty());

    // The verifier follows the cycle without looping.
    let report = verify_claims(
        &graph,
        &[claim("c", DependsOn, "b"), claim("a", DependsOn, "a")],
        VerifierConfig { max_hops: 3 },
    );
    assert!(report.is_fully_verified());
    assert_eq!(
        report.verified[0].path.to_string(),
        "c -[DependsOn]-> a -[DependsOn]-> b"
    );
    assert_eq!(report.verified[1].path.len(), 3);

    // A self-loop is a legal edge and a direct proof.
    graph
        .add_relation(a, a, DependsOn, 1.0, Attributes::new())
        .unwrap();
    let report = verify_claims(
        &graph,
        &[claim("a", DependsOn, "a")],
        VerifierConfig::default(),
    );
    assert_eq!(report.verified[0].evidence, EvidenceKind::Direct);
}

// ---------------------------------------------------------------------------
// Traversal
// ---------------------------------------------------------------------------

#[test]
fn k_hop_expansion_has_exact_node_and_edge_coverage() {
    let graph = fixture();
    let auth = idx(&graph, AUTH);

    let k0 = graph.get_k_hop_subgraph(&[auth], 0);
    assert_eq!(node_names(&k0), expect_nodes(&[(AUTH, 0)]));
    assert!(k0.edges.is_empty());

    let k1 = graph.get_k_hop_subgraph(&[auth], 1);
    assert_eq!(
        node_names(&k1),
        expect_nodes(&[
            (AUTH, 0),
            (BILLING, 1),
            (POSTGRES, 1),
            (REDIS, 1),
            (ACME, 1)
        ])
    );
    let k1_edges = [
        (BILLING, "DependsOn", AUTH),
        (AUTH, "DependsOn", POSTGRES),
        (AUTH, "DependsOn", REDIS),
        (ACME, "Contains", AUTH),
        // Both endpoints are within one hop, so the interconnecting edge is included.
        (ACME, "Contains", BILLING),
    ];
    assert_eq!(edge_triples(&graph, &k1), expect_edges(&k1_edges));
    assert_eq!(k1.edges.len(), 5);

    let k2 = graph.get_k_hop_subgraph(&[auth], 2);
    assert_eq!(
        node_names(&k2),
        expect_nodes(&[
            (AUTH, 0),
            (BILLING, 1),
            (POSTGRES, 1),
            (REDIS, 1),
            (ACME, 1),
            (STORAGE, 2),
            (PIPELINE, 2),
        ])
    );
    let mut k2_edges = k1_edges.to_vec();
    k2_edges.push((POSTGRES, "DependsOn", STORAGE));
    k2_edges.push((PIPELINE, "ProducedBy", ACME));
    assert_eq!(edge_triples(&graph, &k2), expect_edges(&k2_edges));
    assert_eq!(k2.edges.len(), 7);

    // The isolated concept is never reached; the whole component is 9 nodes.
    let all = graph.get_k_hop_subgraph(&[auth], 100);
    assert_eq!(all.nodes.len(), 9);
    assert_eq!(all.edges.len(), 9);
    assert!(!all.contains(graph.node(idx(&graph, ZERO_TRUST)).unwrap().id));
}

#[test]
fn k_hop_expansion_merges_seeds_and_ignores_unknown_indices() {
    let graph = fixture();
    let storage = idx(&graph, STORAGE);
    let v1 = idx(&graph, IMAGE_V1);

    let subgraph = graph.get_k_hop_subgraph(&[storage, v1, storage, NodeIndex::new(999)], 1);
    assert_eq!(
        node_names(&subgraph),
        expect_nodes(&[(STORAGE, 0), (IMAGE_V1, 0), (POSTGRES, 1), (IMAGE_V2, 1)])
    );
    assert_eq!(
        edge_triples(&graph, &subgraph),
        expect_edges(&[
            (POSTGRES, "DependsOn", STORAGE),
            (IMAGE_V2, "Supersedes", IMAGE_V1)
        ])
    );
    // Ordering contract: by depth, then id.
    let depths: Vec<usize> = subgraph.nodes.iter().map(|n| n.depth).collect();
    assert_eq!(depths, [0, 0, 1, 1]);

    assert_eq!(graph.get_k_hop_subgraph(&[], 3), Subgraph::default());
    assert_eq!(
        graph.get_k_hop_subgraph(&[NodeIndex::new(999)], 3),
        Subgraph::default()
    );
}

#[test]
fn connecting_paths_are_exact_and_depth_bounded() {
    let graph = fixture();
    let (acme, billing, auth, storage) = (
        idx(&graph, ACME),
        idx(&graph, BILLING),
        idx(&graph, AUTH),
        idx(&graph, STORAGE),
    );

    let paths = graph.find_connecting_paths(billing, storage, 3);
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].entity_names(), [BILLING, AUTH, POSTGRES, STORAGE]);
    assert!((paths[0].confidence() - 0.36).abs() < 1e-6);
    assert!(graph.find_connecting_paths(billing, storage, 2).is_empty());
    assert!(graph.find_connecting_paths(billing, storage, 0).is_empty());

    // Edges are directed: nothing leads back up the dependency chain.
    assert!(graph.find_connecting_paths(storage, billing, 10).is_empty());

    // Mixed-relation walks are enumerated, shortest first.
    let rendered: Vec<String> = graph
        .find_connecting_paths(acme, auth, 5)
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(
        rendered,
        [
            "Acme Corp -[Contains]-> auth-service",
            "Acme Corp -[Contains]-> billing-service -[DependsOn]-> auth-service",
        ]
    );

    // Relation filter and path cap.
    let only_contains =
        graph.find_connecting_paths_filtered(acme, auth, 5, Some(&Contains), usize::MAX);
    assert_eq!(only_contains.len(), 1);
    assert_eq!(
        graph
            .find_connecting_paths_filtered(acme, auth, 5, None, 1)
            .len(),
        1
    );
}

// ---------------------------------------------------------------------------
// Claim verification
// ---------------------------------------------------------------------------

fn valid_claims() -> Vec<GraphClaim> {
    vec![
        claim(BILLING, DependsOn, AUTH),                // direct, 0.9
        claim(BILLING, DependsOn, POSTGRES),            // 2 hops, 0.72
        claim(" Billing-Service ", DependsOn, STORAGE), // 3 hops, 0.36
        claim(ACME, Contains, AUTH),                    // direct, 1.0
        claim(IMAGE_V2, Supersedes, IMAGE_V1),          // direct, 1.0
        claim(IMAGE_V2, ProducedBy, PIPELINE),          // direct, 1.0
    ]
}

#[test]
fn valid_multi_hop_claims_are_fully_verified() {
    let graph = fixture();
    let claims = valid_claims();
    let report = verify_claims(&graph, &claims, VerifierConfig { max_hops: 3 });

    assert_eq!(report.total_claims, 6);
    assert_eq!(report.verified.len(), 6);
    assert!(report.hallucinated.is_empty());
    assert!(report.is_fully_verified());
    assert_eq!(report.verified_ratio(), 1.0);

    let evidence: Vec<EvidenceKind> = report.verified.iter().map(|v| v.evidence).collect();
    assert_eq!(
        evidence,
        [
            EvidenceKind::Direct,
            EvidenceKind::Transitive,
            EvidenceKind::Transitive,
            EvidenceKind::Direct,
            EvidenceKind::Direct,
            EvidenceKind::Direct,
        ]
    );
    assert_eq!(
        report.verified[1].path.to_string(),
        "billing-service -[DependsOn]-> auth-service -[DependsOn]-> postgres"
    );
    assert_eq!(
        report.verified[2].path.entity_names(),
        [BILLING, AUTH, POSTGRES, STORAGE]
    );
    for (i, verified) in report.verified.iter().enumerate() {
        assert_eq!(verified.claim_index, i);
        assert_eq!(verified.claim, claims[i]);
    }

    // Score is the mean of the path confidences.
    let expected = (0.9 + 0.72 + 0.36 + 1.0 + 1.0 + 1.0) / 6.0;
    assert!((report.confidence_score - expected).abs() < 1e-6);
    assert!((0.0..=1.0).contains(&report.confidence_score));
}

#[test]
fn transitive_evidence_prefers_the_highest_confidence_chain() {
    let graph = fixture();
    // A weak shortcut billing -> postgres does not exist yet; add a stronger
    // alternative route billing -> redis -> postgres next to the 0.72 one.
    graph
        .add_relation_by_name(BILLING, REDIS, DependsOn, 1.0, Attributes::new())
        .unwrap();
    graph
        .add_relation_by_name(REDIS, POSTGRES, DependsOn, 0.95, Attributes::new())
        .unwrap();

    let report = verify_claims(
        &graph,
        &[claim(BILLING, DependsOn, POSTGRES)],
        VerifierConfig::default(),
    );
    assert_eq!(
        report.verified[0].path.entity_names(),
        [BILLING, REDIS, POSTGRES]
    );
    assert!((report.confidence_score - 0.95).abs() < 1e-6);
}

#[test]
fn hallucinated_relations_are_rejected_with_exact_traces() {
    let graph = fixture();
    let claims = vec![
        claim(BILLING, DependsOn, AUTH),     // 0: valid, 0.9
        claim(AUTH, DependsOn, BILLING),     // 1: reversed direction
        claim(ACME, DependsOn, BILLING),     // 2: wrong relation on a real edge
        claim(IMAGE_V2, ProducedBy, ACME),   // 3: ProducedBy is not transitive
        claim(BILLING, DependsOn, "kafka"),  // 4: unknown target
        claim("ghost", Contains, "phantom"), // 5: both unknown
        claim(REDIS, DependsOn, POSTGRES),   // 6: no edge at all
        claim(ACME, Contains, POSTGRES),     // 7: Contains chain stops at auth
        claim(BILLING, DependsOn, POSTGRES), // 8: valid, 0.72
    ];
    let report = verify_claims(&graph, &claims, VerifierConfig { max_hops: 3 });

    assert_eq!(report.total_claims, 9);
    let verified: Vec<usize> = report.verified.iter().map(|v| v.claim_index).collect();
    let rejected: Vec<usize> = report.hallucinated.iter().map(|b| b.claim_index).collect();
    assert_eq!(verified, [0, 8]);
    assert_eq!(rejected, [1, 2, 3, 4, 5, 6, 7]);
    assert!(!report.is_fully_verified());

    let reasons: Vec<&RejectionReason> = report.hallucinated.iter().map(|b| &b.reason).collect();
    assert_eq!(
        reasons,
        [
            &RejectionReason::DirectionReversed,
            &RejectionReason::RelationMismatch {
                found: vec![Contains]
            },
            &RejectionReason::NoSupportingPath,
            &RejectionReason::UnknownEntity {
                missing: vec!["kafka".into()]
            },
            &RejectionReason::UnknownEntity {
                missing: vec!["ghost".into(), "phantom".into()]
            },
            &RejectionReason::NoSupportingPath,
            &RejectionReason::NoSupportingPath,
        ]
    );

    let traces: Vec<String> = report
        .hallucinated
        .iter()
        .map(|b| b.trace.to_string())
        .collect();
    assert_eq!(
        traces,
        [
            // Longest DependsOn chain out of auth, then the hop that does not exist.
            "auth-service -[DependsOn]-> postgres -[DependsOn]-> object-storage -[DependsOn]-x-> billing-service",
            "Acme Corp -[DependsOn]-x-> billing-service",
            "auth-image-v2 -[ProducedBy]-x-> Acme Corp",
            "billing-service -[DependsOn]-x-> kafka",
            "ghost -[Contains]-x-> phantom",
            "redis -[DependsOn]-x-> postgres",
            // Depth-1 tie between billing and auth is broken by the smaller id.
            "Acme Corp -[Contains]-> billing-service -[Contains]-x-> postgres",
        ]
    );

    // Structured form of one trace.
    let reversed = &report.hallucinated[0];
    assert_eq!(reversed.claim, claims[1]);
    assert_eq!(
        reversed.trace.reached.entity_names(),
        [AUTH, POSTGRES, STORAGE]
    );
    assert_eq!(reversed.trace.missing_hop.from, STORAGE);
    assert_eq!(reversed.trace.missing_hop.relation, DependsOn);
    assert_eq!(reversed.trace.missing_hop.to, BILLING);

    // Rejected claims contribute zero to the score.
    let expected = (0.9 + 0.72) / 9.0;
    assert!((report.confidence_score - expected).abs() < 1e-6);
    assert!((report.verified_ratio() - 2.0 / 9.0).abs() < 1e-6);
}

#[test]
fn hop_budget_is_enforced_with_a_depth_exceeded_trace() {
    let graph = fixture();
    let claims = [claim(BILLING, DependsOn, STORAGE)];

    assert!(verify_claims(&graph, &claims, VerifierConfig { max_hops: 3 }).is_fully_verified());

    let report = verify_claims(&graph, &claims, VerifierConfig { max_hops: 2 });
    assert_eq!(report.verified.len(), 0);
    assert_eq!(report.confidence_score, 0.0);
    let broken = &report.hallucinated[0];
    assert_eq!(
        broken.reason,
        RejectionReason::DepthExceeded {
            required_hops: 3,
            max_hops: 2
        }
    );
    assert_eq!(
        broken.trace.to_string(),
        "billing-service -[DependsOn]-> auth-service -[DependsOn]-> postgres -[DependsOn]-x-> object-storage"
    );

    // max_hops = 1 accepts direct edges only.
    let report = verify_claims(
        &graph,
        &[
            claim(BILLING, DependsOn, AUTH),
            claim(BILLING, DependsOn, POSTGRES),
        ],
        VerifierConfig { max_hops: 1 },
    );
    assert_eq!(report.verified.len(), 1);
    assert_eq!(
        report.hallucinated[0].reason,
        RejectionReason::DepthExceeded {
            required_hops: 2,
            max_hops: 1
        }
    );
}

#[test]
fn injecting_a_hallucination_flips_exactly_that_claim() {
    let graph = fixture();
    let config = VerifierConfig::default();
    let baseline = verify_claims(&graph, &valid_claims(), config);
    assert!(baseline.is_fully_verified());

    for position in 0..valid_claims().len() {
        let mut mutated = valid_claims();
        mutated[position].target_entity = ZERO_TRUST.into();
        let report = verify_claims(&graph, &mutated, config);

        assert_eq!(report.hallucinated.len(), 1, "claim {position}");
        assert_eq!(report.hallucinated[0].claim_index, position);
        assert_eq!(
            report.hallucinated[0].reason,
            RejectionReason::NoSupportingPath
        );
        assert_eq!(report.hallucinated[0].trace.missing_hop.to, ZERO_TRUST);
        assert_eq!(report.verified.len(), baseline.verified.len() - 1);
        assert!(report.confidence_score < baseline.confidence_score);

        // The untouched claims keep their exact evidence.
        for verified in &report.verified {
            assert_eq!(verified, &baseline.verified[verified.claim_index]);
        }
    }
}

#[test]
fn verification_is_deterministic() {
    let claims: Vec<GraphClaim> = valid_claims()
        .into_iter()
        .chain([
            claim(AUTH, DependsOn, BILLING),
            claim(ACME, Contains, POSTGRES),
            claim(REDIS, Supersedes, "kafka"),
        ])
        .collect();
    let config = VerifierConfig::default();

    let graph = fixture();
    let first = verify_claims(&graph, &claims, config);
    for _ in 0..25 {
        assert_eq!(verify_claims(&graph, &claims, config), first);
    }
    // An independently built identical graph (fresh hash seeds) agrees too.
    assert_eq!(verify_claims(&fixture(), &claims, config), first);

    let empty = verify_claims(&graph, &[], config);
    assert_eq!(empty.total_claims, 0);
    assert_eq!(empty.confidence_score, 1.0);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

#[test]
fn snapshot_roundtrip_preserves_topology_indices_and_behavior() {
    let graph = fixture();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.ngsnap");

    save_snapshot(&graph, &path).unwrap();
    let restored = load_snapshot(&path).unwrap();

    // Topology and payloads (attributes, embeddings, weights) are identical.
    assert_eq!(restored.snapshot(), graph.snapshot());
    assert_eq!(restored.node_count(), 10);
    assert_eq!(restored.edge_count(), 9);

    // Indices are rebuilt and node indices are stable across the roundtrip.
    for name in [
        ACME, BILLING, AUTH, POSTGRES, REDIS, STORAGE, PIPELINE, IMAGE_V1, IMAGE_V2,
    ] {
        assert_eq!(
            restored.node_index_by_name(name),
            graph.node_index_by_name(name)
        );
    }
    let auth = idx(&restored, AUTH);
    let node = restored.node(auth).unwrap();
    assert_eq!(restored.node_index_by_id(node.id), Some(auth));
    assert_eq!(node.embedding, Some(vec![2.0, 1.0, -0.5]));
    assert_eq!(
        node.attributes.get("origin").map(String::as_str),
        Some("fixture")
    );
    assert_eq!(
        restored.nodes_by_type(&EntityType::Service),
        graph.nodes_by_type(&EntityType::Service)
    );

    // Queries and verification behave identically.
    assert_eq!(
        restored.get_k_hop_subgraph(&[auth], 2),
        graph.get_k_hop_subgraph(&[auth], 2)
    );
    let mut claims = valid_claims();
    claims.push(claim(AUTH, DependsOn, BILLING));
    let config = VerifierConfig::default();
    assert_eq!(
        verify_claims(&restored, &claims, config),
        verify_claims(&graph, &claims, config)
    );

    // Id allocation continues after the highest restored id.
    let fresh = restored
        .add_entity("kafka", EntityType::Service, Attributes::new(), None)
        .unwrap();
    assert_eq!(restored.node(fresh).unwrap().id, EntityId(10));

    // Encoding is reproducible byte for byte.
    assert_eq!(
        encode_snapshot(&graph.snapshot()).unwrap(),
        encode_snapshot(&fixture().snapshot()).unwrap()
    );
}

#[test]
fn snapshot_save_atomically_replaces_the_previous_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.ngsnap");

    let graph = fixture();
    save_snapshot(&graph, &path).unwrap();
    graph
        .add_entity(
            "kafka",
            EntityType::Service,
            Attributes::new(),
            Some(vec![0.0, 0.0, 1.0]),
        )
        .unwrap();
    graph
        .add_relation_by_name(BILLING, "kafka", DependsOn, 0.7, Attributes::new())
        .unwrap();
    save_snapshot(&graph, &path).unwrap();

    let restored = load_snapshot(&path).unwrap();
    assert_eq!(restored.node_count(), 11);
    assert_eq!(restored.edge_count(), 10);

    // No temporary files are left behind.
    let entries: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(entries, ["graph.ngsnap"]);

    // A failed save (missing directory) reports an error and creates nothing.
    let missing = dir.path().join("no-such-dir").join("graph.ngsnap");
    assert!(matches!(
        save_snapshot(&graph, &missing),
        Err(StorageError::Io { .. })
    ));
    assert!(matches!(
        load_snapshot(&missing),
        Err(StorageError::Io { .. })
    ));
}

#[test]
fn corrupted_snapshots_are_rejected() {
    let graph = fixture();
    let bytes = encode_snapshot(&graph.snapshot()).unwrap();
    assert_eq!(decode_snapshot(&bytes).unwrap(), graph.snapshot());

    let mut flipped = bytes.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 0xff;
    assert!(matches!(
        decode_snapshot(&flipped),
        Err(StorageError::ChecksumMismatch)
    ));

    assert!(matches!(
        decode_snapshot(&bytes[..bytes.len() - 7]),
        Err(StorageError::LengthMismatch)
    ));
    assert!(matches!(
        decode_snapshot(&bytes[..10]),
        Err(StorageError::LengthMismatch)
    ));
    assert!(matches!(decode_snapshot(b""), Err(StorageError::BadMagic)));
    assert!(matches!(
        decode_snapshot(b"this is definitely not a snapshot"),
        Err(StorageError::BadMagic)
    ));

    let mut future = bytes.clone();
    future[4..8].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(
        decode_snapshot(&future),
        Err(StorageError::UnsupportedVersion { found: 99 })
    ));

    // A well-formed file whose content breaks graph invariants is refused on load.
    let mut dangling = graph.snapshot();
    dangling.edges[0].target = EntityId(4242);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dangling.ngsnap");
    std::fs::write(&path, encode_snapshot(&dangling).unwrap()).unwrap();
    assert!(matches!(
        load_snapshot(&path),
        Err(StorageError::InvalidGraph(GraphError::EntityIdNotFound(
            EntityId(4242)
        )))
    ));
}

// ---------------------------------------------------------------------------
// Concurrency and schema contracts
// ---------------------------------------------------------------------------

#[test]
fn concurrent_readers_and_writers_keep_indices_consistent() {
    const WRITERS: usize = 4;
    const PER_WRITER: usize = 200;

    let graph = fixture();
    let hub = idx(&graph, ACME);
    let claims = valid_claims();
    let baseline = verify_claims(&graph, &claims, VerifierConfig::default());

    std::thread::scope(|scope| {
        for writer in 0..WRITERS {
            let graph = &graph;
            scope.spawn(move || {
                for i in 0..PER_WRITER {
                    let node = graph
                        .add_entity(
                            format!("team-{writer}-{i}"),
                            EntityType::Custom("Team".into()),
                            Attributes::new(),
                            None,
                        )
                        .unwrap();
                    graph
                        .add_relation(hub, node, Contains, 1.0, Attributes::new())
                        .unwrap();
                }
            });
        }
        for _ in 0..4 {
            let (graph, claims, baseline) = (&graph, &claims, &baseline);
            scope.spawn(move || {
                for _ in 0..200 {
                    // Writers only add unrelated entities, so verdicts never change.
                    assert_eq!(
                        &verify_claims(graph, claims, VerifierConfig::default()),
                        baseline
                    );
                    // Every view is internally consistent: each edge's endpoints are present.
                    let subgraph = graph.get_k_hop_subgraph(&[hub], 1);
                    let ids = subgraph.entity_ids();
                    assert!(subgraph
                        .edges
                        .iter()
                        .all(|e| ids.contains(&e.source) && ids.contains(&e.target)));
                }
            });
        }
    });

    let added = WRITERS * PER_WRITER;
    assert_eq!(graph.node_count(), 10 + added);
    assert_eq!(graph.edge_count(), 9 + added);
    assert_eq!(
        graph
            .nodes_by_type(&EntityType::Custom("Team".into()))
            .len(),
        added
    );

    // Ids handed out under contention are unique.
    let ids: BTreeSet<EntityId> = graph.snapshot().nodes.iter().map(|n| n.id).collect();
    assert_eq!(ids.len(), 10 + added);
}

#[test]
fn schema_contracts_roundtrip_through_json() {
    let original = claim(BILLING, RelationType::Custom("OwnedBy".into()), ACME);
    let json = serde_json::to_string(&original).unwrap();
    assert_eq!(serde_json::from_str::<GraphClaim>(&json).unwrap(), original);

    let parsed: GraphClaim = serde_json::from_str(
        r#"{
            "source_entity": "billing-service",
            "target_entity": "auth-service",
            "relation": "DependsOn",
            "citation": { "document_id": "runbook.md", "chunk_id": "c-17" }
        }"#,
    )
    .unwrap();
    assert_eq!(parsed.relation, DependsOn);
    assert_eq!(parsed.citation.chunk_id, "c-17");

    let graph = fixture();
    let report = verify_claims(&graph, &[parsed, original], VerifierConfig::default());
    let json = serde_json::to_string(&report).unwrap();
    assert_eq!(
        serde_json::from_str::<nexus_graph::VerificationReport>(&json).unwrap(),
        report
    );

    let snapshot = graph.snapshot();
    let json = serde_json::to_string(&snapshot).unwrap();
    assert_eq!(
        serde_json::from_str::<nexus_graph::GraphSnapshot>(&json).unwrap(),
        snapshot
    );
}

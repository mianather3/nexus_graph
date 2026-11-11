//! Phase 3 tests: the HTTP API, driven in-process through `tower::ServiceExt`
//! (no sockets are bound).

// `assert_is_empty` only exists in recent clippy releases.
#![allow(unknown_lints)]
#![allow(
    clippy::assert_is_empty,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::header::{CONTENT_TYPE, RETRY_AFTER};
use axum::http::{HeaderMap, Method, Request, StatusCode};
use axum::Router;
use nexus_graph::server::{router, EngineManager, NamespaceId, ServerConfig, PROBLEM_JSON};
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct TestServer {
    app: Router,
    manager: Arc<EngineManager>,
    dir: TempDir,
}

impl TestServer {
    fn new() -> Self {
        Self::with(|_| {})
    }

    fn with(configure: impl FnOnce(&mut ServerConfig)) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let (app, manager) = open(dir.path(), configure);
        Self { app, manager, dir }
    }

    async fn namespace(&self, id: &str) -> Arc<nexus_graph::server::Namespace> {
        self.manager
            .get(&NamespaceId::parse(id).unwrap())
            .await
            .unwrap()
    }
}

/// A fresh server process over an existing snapshot directory.
fn open(dir: &Path, configure: impl FnOnce(&mut ServerConfig)) -> (Router, Arc<EngineManager>) {
    let mut config = ServerConfig::new(dir);
    configure(&mut config);
    let manager = EngineManager::new(config);
    (router(Arc::clone(&manager)), manager)
}

struct Reply {
    status: StatusCode,
    headers: HeaderMap,
    body: Vec<u8>,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("body is not JSON ({e}): {}", self.text()))
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    fn content_type(&self) -> &str {
        self.headers
            .get(CONTENT_TYPE)
            .map_or("", |v| v.to_str().unwrap())
    }

    /// Asserts this is a well-formed RFC 7807 response with `status`.
    fn problem(&self, status: StatusCode) -> Value {
        assert_eq!(self.status, status, "body: {}", self.text());
        assert_eq!(self.content_type(), PROBLEM_JSON, "body: {}", self.text());
        let problem = self.json();
        assert_eq!(problem["status"], u64::from(status.as_u16()));
        assert!(problem["type"].is_string());
        assert!(!problem["title"].as_str().unwrap().is_empty());
        assert!(!problem["detail"].as_str().unwrap().is_empty());
        problem
    }
}

async fn send(app: &Router, request: Request<Body>) -> Reply {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap()
        .to_vec();
    Reply {
        status,
        headers,
        body,
    }
}

async fn get(app: &Router, uri: &str) -> Reply {
    let request = Request::builder().uri(uri).body(Body::empty()).unwrap();
    send(app, request).await
}

async fn post(app: &Router, uri: &str, body: &Value) -> Reply {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    send(app, request).await
}

async fn post_empty(app: &Router, uri: &str) -> Reply {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    send(app, request).await
}

fn percent_encode(raw: &str) -> String {
    raw.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                char::from(b).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

/// Polls `condition` until it holds; panics after two seconds.
async fn eventually<F, Fut>(what: &str, mut condition: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    while !condition().await {
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for: {what}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

// ---------------------------------------------------------------------------
// Fixture data. Embedding axes: [payments, identity, storage, organization].
// ---------------------------------------------------------------------------

fn entity(name: &str, kind: &str, description: &str, embedding: Option<[f32; 4]>) -> Value {
    let mut value = json!({
        "name": name,
        "type": kind,
        "attributes": { "description": description },
    });
    if let Some(embedding) = embedding {
        value["embedding"] = json!(embedding);
    }
    value
}

fn platform_entities() -> Value {
    json!({ "entities": [
        entity("Acme Corp", "Organization", "Parent company operating the payments platform", Some([0.1, 0.1, 0.1, 0.9])),
        entity("billing-service", "Service", "Generates invoices and charges customer credit cards", Some([0.9, 0.1, 0.2, 0.1])),
        entity("auth-service", "Service", "Issues login tokens and validates user sessions", Some([0.1, 0.9, 0.1, 0.1])),
        entity("postgres", "Service", "Relational database storing user accounts and invoices", Some([0.3, 0.4, 0.8, 0.1])),
        entity("redis", "Service", "In-memory cache for session tokens", Some([0.1, 0.7, 0.6, 0.1])),
        entity("on-call rotation", "Concept", "Weekly schedule of engineers paged for incidents", None),
    ]})
}

fn platform_relations() -> Value {
    json!({ "relations": [
        { "source": "billing-service", "target": "auth-service", "relation": "DependsOn", "weight": 0.9 },
        { "source": "auth-service", "target": "postgres", "relation": "DependsOn", "weight": 0.8 },
        { "source": "auth-service", "target": "redis", "relation": "DependsOn", "weight": 0.9 },
        { "source": "Acme Corp", "target": "billing-service", "relation": "Contains" },
        { "source": "Acme Corp", "target": "auth-service", "relation": "Contains" },
    ]})
}

async fn seed_platform(app: &Router, namespace: &str) {
    let reply = post(
        app,
        &format!("/v1/namespaces/{namespace}/entities/batch"),
        &platform_entities(),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    assert_eq!(reply.json(), json!({ "ingested": 6, "rejected": [] }));
    let reply = post(
        app,
        &format!("/v1/namespaces/{namespace}/relations/batch"),
        &platform_relations(),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    assert_eq!(reply.json(), json!({ "ingested": 5, "rejected": [] }));
}

fn claim(source: &str, relation: &str, target: &str) -> Value {
    json!({
        "source_entity": source,
        "target_entity": target,
        "relation": relation,
        "citation": { "document_id": "doc-1", "chunk_id": format!("{source}->{target}") },
    })
}

fn names(hits: &Value) -> Vec<&str> {
    hits.as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["name"].as_str().unwrap())
        .collect()
}

// ---------------------------------------------------------------------------
// End-to-end happy path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_ingest_search_verify_and_stats() {
    let server = TestServer::new();
    let app = &server.app;

    let reply = get(app, "/healthz").await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.json(), json!({ "status": "ok" }));
    assert!(reply.headers.contains_key("x-request-id"));

    seed_platform(app, "prod").await;

    // Stats reflect the graph and both indices.
    let stats = get(app, "/v1/namespaces/prod/stats").await.json();
    assert_eq!(stats["namespace"], "prod");
    assert_eq!(stats["entities"], 6);
    assert_eq!(stats["relations"], 5);
    assert_eq!(stats["vectors"], 5);
    assert_eq!(stats["vector_dim"], 4);
    assert_eq!(stats["documents"], 6);
    assert!(stats["terms"].as_u64().unwrap() > 20);
    assert_eq!(stats["ingest"]["accepted_jobs"], 2);
    assert_eq!(stats["ingest"]["completed_jobs"], 2);
    assert_eq!(stats["ingest"]["pending_jobs"], 0);

    // Hybrid search: fused hits plus 1-hop graph context.
    let reply = post(
        app,
        "/v1/namespaces/prod/search/hybrid",
        &json!({ "text": "login tokens", "embedding": [0.0, 1.0, 0.0, 0.0], "top_k": 2 }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let bundle = reply.json();
    assert_eq!(names(&bundle["hits"]), ["auth-service", "redis"]);
    assert_eq!(bundle["hits"][0]["vector"]["rank"], 1);
    assert_eq!(bundle["hits"][0]["lexical"]["rank"], 1);
    assert!((bundle["hits"][0]["fused_score"].as_f64().unwrap() - 2.0 / 61.0).abs() < 1e-12);
    let mut context: Vec<(&str, u64)> = bundle["context"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| {
            (
                n["entity"]["canonical_name"].as_str().unwrap(),
                n["depth"].as_u64().unwrap(),
            )
        })
        .collect();
    context.sort_unstable();
    assert_eq!(
        context,
        [
            ("Acme Corp", 1),
            ("auth-service", 0),
            ("billing-service", 1),
            ("postgres", 1),
            ("redis", 0)
        ]
    );
    assert_eq!(bundle["context"]["edges"].as_array().unwrap().len(), 5);
    // Embeddings are omitted from context unless asked for.
    assert!(bundle["context"]["nodes"][0]["entity"]["embedding"].is_null());
    let with_embeddings = post(
        app,
        "/v1/namespaces/prod/search/hybrid",
        &json!({ "text": "login tokens", "top_k": 1, "expansion_hops": 0, "include_embeddings": true }),
    )
    .await
    .json();
    assert_eq!(
        with_embeddings["context"]["nodes"][0]["entity"]["embedding"]
            .as_array()
            .unwrap()
            .len(),
        4
    );

    // Synchronous verification: a direct edge, a 2-hop chain, a hallucination.
    let reply = post(
        app,
        "/v1/namespaces/prod/verify",
        &json!({ "claims": [
            claim("billing-service", "DependsOn", "auth-service"),
            claim("billing-service", "DependsOn", "postgres"),
            claim("postgres", "DependsOn", "billing-service"),
        ]}),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let report = reply.json();
    assert_eq!(report["total_claims"], 3);
    assert_eq!(report["verified"].as_array().unwrap().len(), 2);
    assert_eq!(report["verified"][1]["evidence"], "Transitive");
    assert_eq!(report["hallucinated"][0]["claim_index"], 2);
    let expected = (0.9 + 0.9 * 0.8) / 3.0;
    assert!((report["confidence_score"].as_f64().unwrap() - expected).abs() < 1e-6);

    // max_hops is honoured.
    let strict = post(
        app,
        "/v1/namespaces/prod/verify",
        &json!({ "claims": [claim("billing-service", "DependsOn", "postgres")], "max_hops": 1 }),
    )
    .await
    .json();
    assert_eq!(
        strict["hallucinated"][0]["reason"]["DepthExceeded"]["required_hops"],
        2
    );

    // Metrics expose the same numbers in Prometheus text format.
    let metrics = get(app, "/metrics").await;
    assert_eq!(metrics.status, StatusCode::OK);
    assert!(metrics.content_type().starts_with("text/plain"));
    let text = metrics.text();
    assert!(text.contains("nexus_graph_namespaces 1\n"), "{text}");
    assert!(
        text.contains("nexus_graph_entities{namespace=\"prod\"} 6\n"),
        "{text}"
    );
    assert!(
        text.contains("nexus_graph_relations{namespace=\"prod\"} 5\n"),
        "{text}"
    );
    assert!(
        text.contains("nexus_graph_vectors{namespace=\"prod\"} 5\n"),
        "{text}"
    );
    assert!(
        text.contains("# TYPE nexus_graph_ingest_accepted_jobs_total counter\n"),
        "{text}"
    );
}

#[tokio::test]
async fn batch_items_are_applied_independently() {
    let server = TestServer::new();
    let app = &server.app;

    let reply = post(
        app,
        "/v1/namespaces/n/entities/batch",
        &json!({ "entities": [
            { "name": "alpha", "type": "Service", "embedding": [1.0, 0.0] },
            { "name": "ALPHA ", "type": "Service" },
            { "name": "zero", "type": "Service", "embedding": [0.0, 0.0] },
            { "name": "wide", "type": "Service", "embedding": [1.0, 2.0, 3.0] },
            { "name": "beta", "type": { "Custom": "Team" } },
        ]}),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let report = reply.json();
    assert_eq!(report["ingested"], 2);
    let rejected: Vec<u64> = report["rejected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["index"].as_u64().unwrap())
        .collect();
    assert_eq!(rejected, [1, 2, 3]);
    assert!(report["rejected"][0]["detail"]
        .as_str()
        .unwrap()
        .contains("already exists"));
    assert!(report["rejected"][1]["detail"]
        .as_str()
        .unwrap()
        .contains("zero magnitude"));
    assert!(report["rejected"][2]["detail"]
        .as_str()
        .unwrap()
        .contains("dimension mismatch"));

    let reply = post(
        app,
        "/v1/namespaces/n/relations/batch",
        &json!({ "relations": [
            { "source": "alpha", "target": "beta", "relation": "Contains" },
            { "source": "alpha", "target": "ghost", "relation": "Contains" },
            { "source": "alpha", "target": "beta", "relation": "DependsOn", "weight": 1.5 },
        ]}),
    )
    .await;
    let report = reply.json();
    assert_eq!(report["ingested"], 1);
    assert_eq!(report["rejected"][0]["index"], 1);
    assert_eq!(report["rejected"][1]["index"], 2);

    // Rejected items left no trace: the graph and both indices agree.
    let stats = get(app, "/v1/namespaces/n/stats").await.json();
    assert_eq!(
        (stats["entities"].as_u64(), stats["documents"].as_u64()),
        (Some(2), Some(2))
    );
    assert_eq!(stats["vectors"], 1);
    assert_eq!(stats["relations"], 1);

    // wait=false acknowledges immediately; the batch lands shortly after.
    let reply = post(
        app,
        "/v1/namespaces/n/entities/batch?wait=false",
        &json!({ "entities": [{ "name": "gamma", "type": "Concept" }] }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::ACCEPTED);
    assert_eq!(reply.json()["status"], "queued");
    eventually("queued batch is applied", || async {
        get(app, "/v1/namespaces/n/stats").await.json()["entities"] == 3
    })
    .await;

    // An empty batch is a no-op and does not provision a namespace.
    let reply = post(
        app,
        "/v1/namespaces/untouched/entities/batch",
        &json!({ "entities": [] }),
    )
    .await;
    assert_eq!(reply.json(), json!({ "ingested": 0, "rejected": [] }));
    get(app, "/v1/namespaces/untouched/stats")
        .await
        .problem(StatusCode::NOT_FOUND);
}

// ---------------------------------------------------------------------------
// Namespace isolation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn namespaces_are_strictly_isolated() {
    let server = TestServer::new();
    let app = &server.app;

    seed_platform(app, "tenant-a").await;
    // Tenant B: different data, and a different embedding dimension.
    let reply = post(
        app,
        "/v1/namespaces/tenant-b/entities/batch",
        &json!({ "entities": [
            { "name": "warehouse", "type": "Service", "attributes": { "description": "Stores pallets and login badges" }, "embedding": [1.0, 0.0] },
            { "name": "forklift", "type": "Artifact", "attributes": { "description": "Moves pallets" }, "embedding": [0.0, 1.0] },
            // Same name as an entity of tenant A: no conflict across tenants.
            { "name": "auth-service", "type": "Concept", "attributes": { "description": "Badge reader" }, "embedding": [0.7, 0.7] },
        ]}),
    )
    .await;
    assert_eq!(reply.json()["ingested"], 3, "{}", reply.text());
    let reply = post(
        app,
        "/v1/namespaces/tenant-b/relations/batch",
        &json!({ "relations": [{ "source": "warehouse", "target": "forklift", "relation": "Contains" }] }),
    )
    .await;
    assert_eq!(reply.json()["ingested"], 1);

    // Statistics are per tenant.
    let a = get(app, "/v1/namespaces/tenant-a/stats").await.json();
    let b = get(app, "/v1/namespaces/tenant-b/stats").await.json();
    assert_eq!(
        (
            a["entities"].as_u64(),
            a["relations"].as_u64(),
            a["vector_dim"].as_u64()
        ),
        (Some(6), Some(5), Some(4))
    );
    assert_eq!(
        (
            b["entities"].as_u64(),
            b["relations"].as_u64(),
            b["vector_dim"].as_u64()
        ),
        (Some(3), Some(1), Some(2))
    );

    // Lexical search: A's vocabulary finds nothing in B and vice versa.
    let search = |namespace: &'static str, body: Value| async move {
        post(
            app,
            &format!("/v1/namespaces/{namespace}/search/hybrid"),
            &body,
        )
        .await
    };
    let a_hits = search("tenant-a", json!({ "text": "invoices credit cards" }))
        .await
        .json();
    assert_eq!(names(&a_hits["hits"])[0], "billing-service");
    let b_hits = search(
        "tenant-b",
        json!({ "text": "invoices credit cards postgres redis billing" }),
    )
    .await
    .json();
    assert!(b_hits["hits"].as_array().unwrap().is_empty());
    assert!(b_hits["context"]["nodes"].as_array().unwrap().is_empty());
    let a_hits = search("tenant-a", json!({ "text": "pallets warehouse forklift" }))
        .await
        .json();
    assert!(a_hits["hits"].as_array().unwrap().is_empty());

    // A shared term only returns the tenant's own documents.
    let b_login = search("tenant-b", json!({ "text": "login" })).await.json();
    assert_eq!(names(&b_login["hits"]), ["warehouse"]);
    let a_login = search("tenant-a", json!({ "text": "login" })).await.json();
    assert_eq!(names(&a_login["hits"]), ["auth-service"]);

    // Vector search: every hit and every context entity belongs to the tenant.
    let b_vectors = search(
        "tenant-b",
        json!({ "embedding": [1.0, 0.1], "expansion_hops": 3 }),
    )
    .await
    .json();
    let mut b_names = names(&b_vectors["hits"]);
    b_names.sort_unstable();
    assert_eq!(b_names, ["auth-service", "forklift", "warehouse"]);
    for node in b_vectors["context"]["nodes"].as_array().unwrap() {
        let name = node["entity"]["canonical_name"].as_str().unwrap();
        assert!(
            ["auth-service", "forklift", "warehouse"].contains(&name),
            "leaked `{name}`"
        );
    }
    // A's 4-dimensional query is a dimension error in B, not a cross-tenant hit.
    search("tenant-b", json!({ "embedding": [0.0, 1.0, 0.0, 0.0] }))
        .await
        .problem(StatusCode::UNPROCESSABLE_ENTITY);

    // Same-named entity resolves to each tenant's own record.
    let a_auth = search(
        "tenant-a",
        json!({ "text": "auth service", "top_k": 1, "expansion_hops": 0 }),
    )
    .await
    .json();
    let b_auth = search(
        "tenant-b",
        json!({ "text": "auth service", "top_k": 1, "expansion_hops": 0 }),
    )
    .await
    .json();
    assert_eq!(a_auth["hits"][0]["entity_type"], "Service");
    assert_eq!(b_auth["hits"][0]["entity_type"], "Concept");

    // Verification: facts of A are hallucinations in B.
    let claims = json!({ "claims": [
        claim("billing-service", "DependsOn", "auth-service"),
        claim("warehouse", "Contains", "forklift"),
    ]});
    let a_report = post(app, "/v1/namespaces/tenant-a/verify", &claims)
        .await
        .json();
    let b_report = post(app, "/v1/namespaces/tenant-b/verify", &claims)
        .await
        .json();
    assert_eq!(a_report["verified"][0]["claim_index"], 0);
    assert_eq!(a_report["hallucinated"][0]["claim_index"], 1);
    assert_eq!(b_report["verified"][0]["claim_index"], 1);
    assert_eq!(b_report["hallucinated"][0]["claim_index"], 0);
    assert_eq!(
        b_report["hallucinated"][0]["reason"]["UnknownEntity"]["missing"],
        json!(["billing-service"])
    );

    // Writing to B never changed A.
    assert_eq!(
        get(app, "/v1/namespaces/tenant-a/stats").await.json()["entities"],
        6
    );

    // Reads never create a namespace; unknown tenants are 404 everywhere.
    for (method, path, body) in [
        (Method::GET, "stats", None),
        (
            Method::POST,
            "search/hybrid",
            Some(json!({ "text": "login" })),
        ),
        (Method::POST, "verify", Some(claims.clone())),
        (Method::POST, "snapshot", None),
    ] {
        let uri = format!("/v1/namespaces/tenant-c/{path}");
        let reply = match (method, body) {
            (Method::GET, _) => get(app, &uri).await,
            (_, Some(body)) => post(app, &uri, &body).await,
            (_, None) => post_empty(app, &uri).await,
        };
        let problem = reply.problem(StatusCode::NOT_FOUND);
        assert_eq!(
            problem["type"],
            "urn:nexus-graph:problem:namespace-not-found"
        );
    }
    assert_eq!(server.manager.loaded().await.len(), 2);

    // Snapshots go to separate files.
    for namespace in ["tenant-a", "tenant-b"] {
        let reply = post_empty(
            app,
            &format!("/v1/namespaces/{namespace}/snapshot?wait=true"),
        )
        .await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    }
    let mut files: Vec<String> = std::fs::read_dir(server.dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    assert_eq!(files, ["tenant-a.ngsnap", "tenant-b.ngsnap"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_first_writes_provision_a_namespace_once() {
    let server = TestServer::new();
    let mut tasks = Vec::new();
    for i in 0..24 {
        let app = server.app.clone();
        tasks.push(tokio::spawn(async move {
            let namespace = if i % 2 == 0 { "even" } else { "odd" };
            let body =
                json!({ "entities": [{ "name": format!("entity-{i}"), "type": "Concept" }] });
            post(
                &app,
                &format!("/v1/namespaces/{namespace}/entities/batch"),
                &body,
            )
            .await
        }));
    }
    for task in tasks {
        let reply = task.await.unwrap();
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
        assert_eq!(reply.json()["ingested"], 1);
    }
    // Had provisioning raced, some writes would have landed in a discarded copy.
    assert_eq!(server.manager.loaded().await.len(), 2);
    for namespace in ["even", "odd"] {
        let stats = get(&server.app, &format!("/v1/namespaces/{namespace}/stats"))
            .await
            .json();
        assert_eq!(stats["entities"], 12);
        assert_eq!(stats["ingest"]["completed_jobs"], 12);
    }
}

// ---------------------------------------------------------------------------
// Backpressure
// ---------------------------------------------------------------------------

#[tokio::test]
async fn saturated_ingestion_queue_signals_backpressure() {
    const CAPACITY: usize = 2;
    let server = TestServer::with(|config| config.ingest_queue_capacity = CAPACITY);
    let app = &server.app;
    let uri = "/v1/namespaces/busy/entities/batch";
    let batch = |name: &str| json!({ "entities": [{ "name": name, "type": "Concept" }] });

    assert_eq!(post(app, uri, &batch("seed")).await.status, StatusCode::OK);
    let namespace = server.namespace("busy").await;

    // Stall the ingestion worker: it needs the engine lock exclusively.
    let stall = namespace.engine().clone().read_owned().await;

    // The first job is taken off the queue by the worker, which then blocks.
    let queued = format!("{uri}?wait=false");
    assert_eq!(
        post(app, &queued, &batch("in-flight")).await.status,
        StatusCode::ACCEPTED
    );
    eventually("worker picks up the first job", || async {
        namespace.ingest_stats().queued_jobs == 0
    })
    .await;

    // The queue now takes exactly CAPACITY more jobs...
    for i in 0..CAPACITY {
        let reply = post(app, &queued, &batch(&format!("queued-{i}"))).await;
        assert_eq!(reply.status, StatusCode::ACCEPTED, "{}", reply.text());
        assert_eq!(reply.json()["pending_jobs"], (i + 2) as u64);
    }
    assert_eq!(namespace.ingest_stats().queued_jobs, CAPACITY);

    // ...and everything beyond that is refused immediately, in both modes.
    for (attempt, target) in [&queued, &queued, &uri.to_owned()].into_iter().enumerate() {
        let reply = post(app, target, &batch(&format!("overflow-{attempt}"))).await;
        let problem = reply.problem(StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            problem["type"],
            "urn:nexus-graph:problem:ingest-backpressure"
        );
        assert_eq!(problem["title"], "Ingestion Queue Full");
        assert_eq!(problem["queue_capacity"], CAPACITY as u64);
        assert_eq!(problem["retry_after_seconds"], 1);
        assert_eq!(reply.headers.get(RETRY_AFTER).unwrap(), "1");
    }
    // Relations share the namespace's queue.
    let reply = post(
        app,
        "/v1/namespaces/busy/relations/batch",
        &json!({ "relations": [{ "source": "seed", "target": "seed", "relation": "Contains" }] }),
    )
    .await;
    reply.problem(StatusCode::TOO_MANY_REQUESTS);

    let stats = namespace.ingest_stats();
    assert_eq!(stats.queue_capacity, CAPACITY);
    assert_eq!(stats.accepted_jobs, 4);
    assert_eq!(stats.completed_jobs, 1);
    assert_eq!(stats.pending_jobs, 3);
    assert_eq!(stats.rejected_jobs, 4);

    // Other namespaces have their own queue and are unaffected.
    let reply = post(app, "/v1/namespaces/idle/entities/batch", &batch("fine")).await;
    assert_eq!(reply.status, StatusCode::OK);

    // Once the worker can run again the queue drains, in order, and accepts work.
    drop(stall);
    eventually("queue drains", || async {
        namespace.ingest_stats().pending_jobs == 0
    })
    .await;
    let stats = get(app, "/v1/namespaces/busy/stats").await.json();
    assert_eq!(
        stats["entities"], 4,
        "seed + in-flight + {CAPACITY} queued; refused jobs were dropped"
    );
    assert_eq!(stats["ingest"]["rejected_jobs"], 4);
    let reply = post(app, uri, &batch("after-recovery")).await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.json()["ingested"], 1);

    // None of the refused batches was applied.
    let found = post(
        app,
        "/v1/namespaces/busy/search/hybrid",
        &json!({ "text": "overflow", "top_k": 10 }),
    )
    .await
    .json();
    assert!(found["hits"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn slow_requests_time_out_with_a_problem_document() {
    let server = TestServer::with(|config| config.request_timeout = Duration::from_millis(100));
    let app = &server.app;
    let uri = "/v1/namespaces/slow/entities/batch";
    let batch = |name: &str| json!({ "entities": [{ "name": name, "type": "Concept" }] });
    assert_eq!(post(app, uri, &batch("seed")).await.status, StatusCode::OK);

    let namespace = server.namespace("slow").await;
    let stall = namespace.engine().clone().read_owned().await;
    // A waiting ingest cannot complete while the worker is stalled.
    let reply = post(app, uri, &batch("late")).await;
    reply.problem(StatusCode::GATEWAY_TIMEOUT);

    // The job was accepted before the timeout and is still applied.
    drop(stall);
    eventually("timed-out job is applied", || async {
        namespace.ingest_stats().pending_jobs == 0
    })
    .await;
    assert_eq!(
        get(app, "/v1/namespaces/slow/stats").await.json()["entities"],
        2
    );
}

// ---------------------------------------------------------------------------
// Server-Sent Events
// ---------------------------------------------------------------------------

/// Parses an SSE body into `(event, data)` pairs, skipping keep-alive comments.
fn parse_sse(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter_map(|block| {
            let mut event = None;
            let mut data = Vec::new();
            for line in block.lines() {
                if let Some(name) = line.strip_prefix("event: ") {
                    event = Some(name.to_owned());
                } else if let Some(payload) = line.strip_prefix("data: ") {
                    data.push(payload);
                } else {
                    assert!(
                        line.is_empty() || line.starts_with(':'),
                        "unexpected SSE line `{line}`"
                    );
                }
            }
            let event = event?;
            let data = serde_json::from_str(&data.join("\n"))
                .unwrap_or_else(|e| panic!("event `{event}` has non-JSON data: {e}"));
            Some((event, data))
        })
        .collect()
}

#[tokio::test]
async fn sse_stream_emits_one_event_per_claim_in_order_then_a_summary() {
    let server = TestServer::new();
    let app = &server.app;
    seed_platform(app, "prod").await;

    let claims = json!([
        claim("billing-service", "DependsOn", "auth-service"), // direct
        claim("billing-service", "DependsOn", "kafka"),        // unknown entity
        claim("billing-service", "DependsOn", "redis"),        // 2-hop chain
        claim("redis", "DependsOn", "auth-service"),           // reversed
        claim("Acme Corp", "Contains", "auth-service"),        // direct
    ]);
    let uri = format!(
        "/v1/namespaces/prod/verify/stream?claims={}",
        percent_encode(&claims.to_string())
    );
    let reply = get(app, &uri).await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    assert!(reply.content_type().starts_with("text/event-stream"));
    assert_eq!(reply.headers.get("cache-control").unwrap(), "no-cache");

    // Wire format: `event:` line, single `data:` line, blank line.
    let text = reply.text();
    assert!(text.starts_with("event: claim_verified\ndata: {"), "{text}");
    assert!(text.ends_with("}\n\n"), "{text}");
    assert_eq!(
        text.matches("\n\n").count(),
        6,
        "five claims and one summary"
    );

    let events = parse_sse(&text);
    let order: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        order,
        [
            "claim_verified",
            "claim_rejected",
            "claim_verified",
            "claim_rejected",
            "claim_verified",
            "complete"
        ]
    );
    // Claim events arrive in submission order and carry their own index and claim.
    for (index, (_, data)) in events.iter().take(5).enumerate() {
        assert_eq!(data["claim_index"], index as u64);
        assert_eq!(data["claim"], claims[index]);
    }
    assert_eq!(events[0].1["evidence"], "Direct");
    assert_eq!(
        events[1].1["reason"]["UnknownEntity"]["missing"],
        json!(["kafka"])
    );
    assert_eq!(events[2].1["evidence"], "Transitive");
    assert_eq!(events[2].1["path"]["hops"].as_array().unwrap().len(), 2);
    assert_eq!(events[3].1["reason"], "DirectionReversed");
    assert_eq!(events[3].1["trace"]["missing_hop"]["to"], "auth-service");

    // The stream agrees exactly with the synchronous endpoint.
    let report = post(
        app,
        "/v1/namespaces/prod/verify",
        &json!({ "claims": claims }),
    )
    .await
    .json();
    let streamed_verified: Vec<&Value> = events
        .iter()
        .filter(|(n, _)| n == "claim_verified")
        .map(|(_, d)| d)
        .collect();
    let streamed_rejected: Vec<&Value> = events
        .iter()
        .filter(|(n, _)| n == "claim_rejected")
        .map(|(_, d)| d)
        .collect();
    assert_eq!(
        streamed_verified,
        report["verified"]
            .as_array()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    );
    assert_eq!(
        streamed_rejected,
        report["hallucinated"]
            .as_array()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    );

    let summary = &events[5].1;
    assert_eq!(summary["total_claims"], 5);
    assert_eq!(summary["verified"], 3);
    assert_eq!(summary["rejected"], 2);
    assert_eq!(summary["confidence_score"], report["confidence_score"]);
    let expected = (0.9 + 0.9 * 0.9 + 1.0) / 5.0;
    assert!((summary["confidence_score"].as_f64().unwrap() - expected).abs() < 1e-6);

    // The POST variant streams the same events; max_hops applies per stream.
    let reply = post(
        app,
        "/v1/namespaces/prod/verify/stream",
        &json!({ "claims": claims }),
    )
    .await;
    assert!(reply.content_type().starts_with("text/event-stream"));
    assert_eq!(parse_sse(&reply.text()), events);
    let strict = get(app, &format!("{uri}&max_hops=1")).await;
    let strict = parse_sse(&strict.text());
    assert_eq!(strict[2].0, "claim_rejected");
    assert_eq!(strict[5].1["verified"], 2);

    // No claims: just the terminal event.
    let empty = get(app, "/v1/namespaces/prod/verify/stream?claims=%5B%5D").await;
    assert_eq!(
        parse_sse(&empty.text()),
        [(
            "complete".to_owned(),
            json!({ "total_claims": 0, "verified": 0, "rejected": 0, "confidence_score": 1.0 })
        )]
    );

    // Request errors are problem documents, not streams.
    get(app, "/v1/namespaces/prod/verify/stream")
        .await
        .problem(StatusCode::BAD_REQUEST);
    get(app, "/v1/namespaces/prod/verify/stream?claims=not-json")
        .await
        .problem(StatusCode::BAD_REQUEST);
    get(
        app,
        "/v1/namespaces/prod/verify/stream?claims=%5B%5D&max_hops=99",
    )
    .await
    .problem(StatusCode::UNPROCESSABLE_ENTITY);
    get(app, "/v1/namespaces/nobody/verify/stream?claims=%5B%5D")
        .await
        .problem(StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn sse_stream_applies_backpressure_to_slow_consumers() {
    use tokio_stream::StreamExt;

    // A one-event buffer: the producer can only run one event ahead of the client.
    let server = TestServer::with(|config| config.sse_buffer = 1);
    let app = &server.app;
    seed_platform(app, "prod").await;
    let claims: Vec<Value> = (0..50)
        .map(|_| claim("billing-service", "DependsOn", "auth-service"))
        .collect();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/namespaces/prod/verify/stream")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(json!({ "claims": claims }).to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Events can be consumed one at a time, as they are produced.
    let mut frames = response.into_body().into_data_stream();
    let first = frames.next().await.unwrap().unwrap();
    let first = String::from_utf8_lossy(&first).into_owned();
    assert!(first.starts_with("event: claim_verified\n"), "{first}");
    assert_eq!(parse_sse(&first)[0].1["claim_index"], 0);

    // Dropping the body mid-stream (client disconnect) must not wedge the server.
    drop(frames);
    let reply = get(app, "/v1/namespaces/prod/stats").await;
    assert_eq!(reply.status, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Snapshot persistence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn snapshots_persist_in_the_background_and_reload_on_demand() {
    let server = TestServer::new();
    let app = &server.app;
    seed_platform(app, "prod").await;
    let file = server.dir.path().join("prod.ngsnap");
    assert!(!file.exists());

    // Default mode: acknowledged immediately, persisted in the background.
    let reply = post_empty(app, "/v1/namespaces/prod/snapshot").await;
    assert_eq!(reply.status, StatusCode::ACCEPTED, "{}", reply.text());
    assert_eq!(
        reply.json(),
        json!({ "status": "scheduled", "namespace": "prod" })
    );
    eventually("background snapshot completes", || async {
        let snapshot = &get(app, "/v1/namespaces/prod/stats").await.json()["snapshot"];
        snapshot["pending"] == 0 && snapshot["last_success"].is_object()
    })
    .await;
    assert!(file.exists());
    let persisted = get(app, "/v1/namespaces/prod/stats").await.json()["snapshot"].clone();
    assert!(persisted["last_error"].is_null());
    assert_eq!(
        persisted["last_success"]["bytes"],
        std::fs::metadata(&file).unwrap().len()
    );

    let search_body =
        json!({ "text": "login tokens", "embedding": [0.0, 1.0, 0.0, 0.0], "top_k": 3 });
    let verify_body = json!({ "claims": [
        claim("billing-service", "DependsOn", "postgres"),
        claim("redis", "DependsOn", "billing-service"),
    ]});
    let search_before = post(app, "/v1/namespaces/prod/search/hybrid", &search_body)
        .await
        .json();
    let verify_before = post(app, "/v1/namespaces/prod/verify", &verify_body)
        .await
        .json();

    // A new process over the same directory starts empty and loads on demand.
    let (restarted, manager) = open(server.dir.path(), |_| {});
    assert!(manager.loaded().await.is_empty());
    let stats = get(&restarted, "/v1/namespaces/prod/stats").await;
    assert_eq!(stats.status, StatusCode::OK, "{}", stats.text());
    let stats = stats.json();
    assert_eq!(
        (stats["entities"].as_u64(), stats["relations"].as_u64()),
        (Some(6), Some(5))
    );
    assert_eq!(
        (stats["vectors"].as_u64(), stats["documents"].as_u64()),
        (Some(5), Some(6))
    );
    assert_eq!(manager.loaded().await.len(), 1);
    // Graph, vector index and lexical index all behave exactly as before.
    assert_eq!(
        post(
            &restarted,
            "/v1/namespaces/prod/search/hybrid",
            &search_body
        )
        .await
        .json(),
        search_before
    );
    assert_eq!(
        post(&restarted, "/v1/namespaces/prod/verify", &verify_body)
            .await
            .json(),
        verify_before
    );
    // Namespaces without a snapshot are still unknown.
    get(&restarted, "/v1/namespaces/other/stats")
        .await
        .problem(StatusCode::NOT_FOUND);

    // A write to a snapshotted namespace loads it first rather than starting empty.
    let (third, _) = open(server.dir.path(), |_| {});
    let reply = post(
        &third,
        "/v1/namespaces/prod/entities/batch",
        &json!({ "entities": [entity("kafka", "Service", "Event stream broker", Some([0.5, 0.1, 0.5, 0.1]))] }),
    )
    .await;
    assert_eq!(reply.json()["ingested"], 1, "{}", reply.text());
    assert_eq!(
        get(&third, "/v1/namespaces/prod/stats").await.json()["entities"],
        7
    );

    // Synchronous mode replaces the file atomically and reports its size.
    let reply = post_empty(&third, "/v1/namespaces/prod/snapshot?wait=true").await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.text());
    let info = reply.json();
    assert_eq!(info["bytes"], std::fs::metadata(&file).unwrap().len());
    assert!(info["completed_at_unix_ms"].as_u64().unwrap() > 1_600_000_000_000);
    let files: Vec<String> = std::fs::read_dir(server.dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(files, ["prod.ngsnap"], "no temporary files left behind");

    let (fourth, _) = open(server.dir.path(), |_| {});
    assert_eq!(
        get(&fourth, "/v1/namespaces/prod/stats").await.json()["entities"],
        7
    );
    let hits = post(
        &fourth,
        "/v1/namespaces/prod/search/hybrid",
        &json!({ "text": "event stream broker" }),
    )
    .await
    .json();
    assert_eq!(names(&hits["hits"]), ["kafka"]);

    // A corrupted snapshot is an error, never a silently empty namespace.
    std::fs::write(&file, b"NXGS garbage").unwrap();
    let (broken, manager) = open(server.dir.path(), |_| {});
    get(&broken, "/v1/namespaces/prod/stats")
        .await
        .problem(StatusCode::INTERNAL_SERVER_ERROR);
    let reply = post(
        &broken,
        "/v1/namespaces/prod/entities/batch",
        &json!({ "entities": [{ "name": "x", "type": "Concept" }] }),
    )
    .await;
    reply.problem(StatusCode::INTERNAL_SERVER_ERROR);
    assert!(manager.loaded().await.is_empty());
}

// ---------------------------------------------------------------------------
// Error contract
// ---------------------------------------------------------------------------

#[tokio::test]
async fn every_error_is_an_rfc7807_problem_document() {
    let server = TestServer::with(|config| {
        config.max_body_bytes = 2048;
        config.max_batch_items = 6;
        config.max_claims = 2;
        config.max_namespaces = 1;
    });
    let app = &server.app;
    seed_platform(app, "prod").await;

    let raw = |method: Method, uri: &str, content_type: Option<&str>, body: &str| {
        let mut request = Request::builder().method(method).uri(uri);
        if let Some(content_type) = content_type {
            request = request.header(CONTENT_TYPE, content_type);
        }
        request.body(Body::from(body.to_owned())).unwrap()
    };
    let json_type = Some("application/json");
    let entities = "/v1/namespaces/prod/entities/batch";
    let search = "/v1/namespaces/prod/search/hybrid";

    // Routing.
    let problem = get(app, "/v1/nope").await.problem(StatusCode::NOT_FOUND);
    assert_eq!(problem["type"], "about:blank");
    assert_eq!(problem["title"], "Not Found");
    get(app, entities)
        .await
        .problem(StatusCode::METHOD_NOT_ALLOWED);
    send(app, raw(Method::DELETE, "/healthz", None, ""))
        .await
        .problem(StatusCode::METHOD_NOT_ALLOWED);

    // Namespace ids.
    let too_long = "x".repeat(65);
    for bad in [
        "a.b",
        "..",
        "-lead",
        "sp%20ace",
        "%2E%2E%2Fetc",
        too_long.as_str(),
    ] {
        let problem = get(app, &format!("/v1/namespaces/{bad}/stats"))
            .await
            .problem(StatusCode::BAD_REQUEST);
        assert_eq!(
            problem["type"], "urn:nexus-graph:problem:invalid-namespace",
            "{bad}"
        );
    }

    // Malformed bodies.
    send(app, raw(Method::POST, entities, json_type, "{ not json"))
        .await
        .problem(StatusCode::BAD_REQUEST);
    send(app, raw(Method::POST, entities, Some("text/plain"), "{}"))
        .await
        .problem(StatusCode::UNSUPPORTED_MEDIA_TYPE);
    send(
        app,
        raw(Method::POST, entities, json_type, r#"{"entities": 5}"#),
    )
    .await
    .problem(StatusCode::UNPROCESSABLE_ENTITY);
    send(
        app,
        raw(
            Method::POST,
            entities,
            json_type,
            r#"{"entities": [], "extra": 1}"#,
        ),
    )
    .await
    .problem(StatusCode::UNPROCESSABLE_ENTITY);
    send(
        app,
        raw(
            Method::POST,
            entities,
            json_type,
            r#"{"entities": [{"name": "x", "type": "NoSuchType"}]}"#,
        ),
    )
    .await
    .problem(StatusCode::UNPROCESSABLE_ENTITY);
    post(
        app,
        &format!("{entities}?wait=maybe"),
        &json!({ "entities": [] }),
    )
    .await
    .problem(StatusCode::BAD_REQUEST);

    // Size limits.
    let huge = json!({ "entities": [{ "name": "x".repeat(4096), "type": "Concept" }] });
    post(app, entities, &huge)
        .await
        .problem(StatusCode::PAYLOAD_TOO_LARGE);
    let many: Vec<Value> = (0..7)
        .map(|i| json!({ "name": format!("e{i}"), "type": "Concept" }))
        .collect();
    post(app, entities, &json!({ "entities": many }))
        .await
        .problem(StatusCode::PAYLOAD_TOO_LARGE);
    let claims = vec![claim("a", "DependsOn", "b"); 3];
    post(
        app,
        "/v1/namespaces/prod/verify",
        &json!({ "claims": claims }),
    )
    .await
    .problem(StatusCode::PAYLOAD_TOO_LARGE);

    // Semantically invalid searches.
    for body in [
        json!({}),
        json!({ "text": "x", "top_k": 0 }),
        json!({ "text": "x", "top_k": 100_000 }),
        json!({ "text": "x", "expansion_hops": 50 }),
        json!({ "text": "x", "rrf_k": -1.0 }),
        json!({ "text": "x", "vector_weight": 0.0, "lexical_weight": 0.0 }),
        json!({ "embedding": [1.0, 2.0] }),
        json!({ "embedding": [0.0, 0.0, 0.0, 0.0] }),
        json!({ "embedding": [] }),
    ] {
        let reply = post(app, search, &body).await;
        reply.problem(StatusCode::UNPROCESSABLE_ENTITY);
    }
    post(
        app,
        "/v1/namespaces/prod/verify",
        &json!({ "claims": [], "max_hops": 99 }),
    )
    .await
    .problem(StatusCode::UNPROCESSABLE_ENTITY);

    // Namespace quota.
    let reply = post(
        app,
        "/v1/namespaces/second/entities/batch",
        &json!({ "entities": [{ "name": "x", "type": "Concept" }] }),
    )
    .await;
    let problem = reply.problem(StatusCode::INSUFFICIENT_STORAGE);
    assert_eq!(problem["type"], "urn:nexus-graph:problem:namespace-limit");

    // Request ids: generated when absent, echoed when supplied and well-formed.
    let generated = get(app, "/v1/nope").await;
    assert!(!generated.headers.get("x-request-id").unwrap().is_empty());
    let mut request = raw(Method::GET, "/v1/nope", None, "");
    request
        .headers_mut()
        .insert("x-request-id", "trace-1234".parse().unwrap());
    assert_eq!(
        send(app, request)
            .await
            .headers
            .get("x-request-id")
            .unwrap(),
        "trace-1234"
    );
    let mut request = raw(Method::GET, "/healthz", None, "");
    request
        .headers_mut()
        .insert("x-request-id", "bad id with spaces".parse().unwrap());
    assert_ne!(
        send(app, request)
            .await
            .headers
            .get("x-request-id")
            .unwrap(),
        "bad id with spaces"
    );

    // None of the rejected requests changed anything.
    assert_eq!(
        get(app, "/v1/namespaces/prod/stats").await.json()["entities"],
        6
    );
}

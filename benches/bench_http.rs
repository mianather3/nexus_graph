//! Concurrent HTTP benchmarks.
//!
//! Requests are dispatched to the axum router in-process (no sockets), on a
//! multi-threaded Tokio runtime, so the numbers cover routing, middleware,
//! JSON (de)serialization, lock acquisition and the `spawn_blocking` hop, but
//! not the network stack.
//!
//! ```text
//! cargo bench --bench bench_http
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

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::header::CONTENT_TYPE;
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use nexus_graph::server::{router, EngineManager, ServerConfig};
use serde_json::{json, Value};
use tokio::runtime::Runtime;
use tower::ServiceExt;

use common::Lcg;

const ENTITIES: usize = 5_000;
const DIM: usize = 128;
const CONCURRENCY: [usize; 3] = [1, 8, 32];

async fn call(app: &Router, method: Method, uri: &str, body: Option<Vec<u8>>) -> StatusCode {
    let mut request = Request::builder().method(method).uri(uri);
    if body.is_some() {
        request = request.header(CONTENT_TYPE, "application/json");
    }
    let request = request
        .body(body.map_or_else(Body::empty, Body::from))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    // Drain the body: serialization is part of the cost being measured.
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    status
}

struct Fixture {
    runtime: Runtime,
    app: Router,
    search_body: Vec<u8>,
    verify_body: Vec<u8>,
    _snapshots: tempfile::TempDir,
}

fn fixture() -> Fixture {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let snapshots = tempfile::tempdir().unwrap();
    let app = router(EngineManager::new(ServerConfig::new(snapshots.path())));

    let mut rng = Lcg(1);
    runtime.block_on(async {
        for batch in 0..ENTITIES / 1_000 {
            let entities: Vec<Value> = (0..1_000)
                .map(|i| {
                    let id = batch * 1_000 + i;
                    json!({
                        "name": format!("entity-{id}"),
                        "type": "Service",
                        "attributes": { "description": format!("service number {id} of shard w{} tier w{}", id % 97, id % 13) },
                        "embedding": rng.vector(DIM),
                    })
                })
                .collect();
            let body = serde_json::to_vec(&json!({ "entities": entities })).unwrap();
            let status = call(&app, Method::POST, "/v1/namespaces/bench/entities/batch", Some(body)).await;
            assert_eq!(status, StatusCode::OK);
        }
        let relations: Vec<Value> = (0..ENTITIES)
            .flat_map(|id| {
                let near = (id + 1) % ENTITIES;
                let far = (id * 7 + 13) % ENTITIES;
                [near, far].map(|target| {
                    json!({ "source": format!("entity-{id}"), "target": format!("entity-{target}"), "relation": "DependsOn", "weight": 0.9 })
                })
            })
            .collect();
        for chunk in relations.chunks(5_000) {
            let body = serde_json::to_vec(&json!({ "relations": chunk })).unwrap();
            let status = call(&app, Method::POST, "/v1/namespaces/bench/relations/batch", Some(body)).await;
            assert_eq!(status, StatusCode::OK);
        }
    });

    let search_body = serde_json::to_vec(&json!({
        "text": "service shard w42 tier w7",
        "embedding": rng.vector(DIM),
        "top_k": 10,
        "expansion_hops": 1,
    }))
    .unwrap();
    let claims: Vec<Value> = (0..16)
        .map(|i| {
            json!({
                "source_entity": format!("entity-{i}"),
                "target_entity": format!("entity-{}", i + 2),
                "relation": "DependsOn",
                "citation": { "document_id": "bench", "chunk_id": "0" },
            })
        })
        .collect();
    let verify_body = serde_json::to_vec(&json!({ "claims": claims })).unwrap();

    Fixture {
        runtime,
        app,
        search_body,
        verify_body,
        _snapshots: snapshots,
    }
}

/// Fires `concurrency` identical requests at once and waits for all of them.
fn burst(
    fixture: &Fixture,
    concurrency: usize,
    method: &Method,
    uri: &'static str,
    body: Option<&Vec<u8>>,
) {
    fixture.runtime.block_on(async {
        let tasks: Vec<_> = (0..concurrency)
            .map(|_| {
                let app = fixture.app.clone();
                let method = method.clone();
                let body = body.cloned();
                tokio::spawn(async move { call(&app, method, uri, body).await })
            })
            .collect();
        for task in tasks {
            assert_eq!(task.await.unwrap(), StatusCode::OK);
        }
    });
}

fn http_benchmarks(c: &mut Criterion) {
    let fixture = fixture();

    let mut group = c.benchmark_group("http_healthz");
    for concurrency in CONCURRENCY {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_function(BenchmarkId::new("concurrency", concurrency), |bench| {
            bench.iter(|| burst(&fixture, concurrency, &Method::GET, "/healthz", None));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("http_hybrid_search_5k_entities_dim128");
    for concurrency in CONCURRENCY {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_function(BenchmarkId::new("concurrency", concurrency), |bench| {
            bench.iter(|| {
                burst(
                    &fixture,
                    concurrency,
                    &Method::POST,
                    "/v1/namespaces/bench/search/hybrid",
                    Some(&fixture.search_body),
                );
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("http_verify_16_claims");
    for concurrency in CONCURRENCY {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_function(BenchmarkId::new("concurrency", concurrency), |bench| {
            bench.iter(|| {
                burst(
                    &fixture,
                    concurrency,
                    &Method::POST,
                    "/v1/namespaces/bench/verify",
                    Some(&fixture.verify_body),
                );
            });
        });
    }
    group.finish();

    latency_reports(&fixture);
}

/// Closed-loop load: `concurrency` clients each issue requests back to back;
/// every request is timed individually.
fn latency_reports(fixture: &Fixture) {
    const REQUESTS: usize = 4_000;
    if !common::latency_reports_enabled() {
        return;
    }
    let endpoints: [(&str, &'static str, &Vec<u8>); 2] = [
        (
            "hybrid search",
            "/v1/namespaces/bench/search/hybrid",
            &fixture.search_body,
        ),
        (
            "verify x16",
            "/v1/namespaces/bench/verify",
            &fixture.verify_body,
        ),
    ];
    for (label, uri, body) in endpoints {
        for concurrency in CONCURRENCY {
            let name = format!("http {label} concurrency={concurrency}");
            if !common::selected(&name) {
                continue;
            }
            let per_client = REQUESTS / concurrency;
            let body = Arc::new(body.clone());
            let started = Instant::now();
            let mut samples: Vec<Duration> = fixture.runtime.block_on(async {
                let clients: Vec<_> = (0..concurrency)
                    .map(|_| {
                        let app = fixture.app.clone();
                        let body = Arc::clone(&body);
                        tokio::spawn(async move {
                            let mut latencies = Vec::with_capacity(per_client);
                            for _ in 0..per_client {
                                let start = Instant::now();
                                let status =
                                    call(&app, Method::POST, uri, Some(body.to_vec())).await;
                                latencies.push(start.elapsed());
                                assert_eq!(status, StatusCode::OK);
                            }
                            latencies
                        })
                    })
                    .collect();
                let mut all = Vec::with_capacity(REQUESTS);
                for client in clients {
                    all.extend(client.await.unwrap());
                }
                all
            });
            let throughput = samples.len() as f64 / started.elapsed().as_secs_f64();
            common::print_latency(&name, &mut samples);
            println!("        {name:<46} throughput={throughput:.0} req/s");
        }
    }
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = http_benchmarks
}

fn main() {
    benches();
    Criterion::default().configure_from_args().final_summary();
}

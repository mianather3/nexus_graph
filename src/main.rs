//! `nexus_graph` command-line interface.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::net::{TcpStream, ToSocketAddrs};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;

use nexus_graph::eval::{evaluate, EvalConfig, EvalDataset};
use nexus_graph::server::dto::{EntityInput, RelationInput};
use nexus_graph::server::{self, ServerConfig};
use nexus_graph::{
    load_snapshot, save_snapshot, verify_claims, EntityId, GraphClaim, HybridQuery,
    HybridRetriever, Kernel, KnowledgeGraph, RetrievalBundle, RetrievalConfig, VerifierConfig,
};

const USAGE: &str = "\
NexusGraph - hybrid knowledge graph retrieval with deterministic claim verification

USAGE:
    nexus_graph ingest <graph.json> <snapshot>
        Build a binary snapshot from a JSON graph description.
    nexus_graph stats <snapshot>
        Print entity and relation counts of a snapshot.
    nexus_graph verify <snapshot> <claims.json> [max_hops]
        Verify a JSON array of claims and print the report as JSON.
        Exits with status 2 if any claim is rejected.
    nexus_graph index <snapshot>
        Build the vector and lexical indices and print their statistics.
    nexus_graph search <snapshot> <query> [top_k]
        BM25 text search over entity names, types and attributes.
    nexus_graph hybrid <snapshot> [OPTIONS]
        Hybrid vector + text search with rank fusion and graph expansion.
    nexus_graph serve --bind <addr> --snapshot-dir <path> [OPTIONS]
        Run the multi-tenant HTTP API (see SERVE OPTIONS).
    nexus_graph eval --dataset <path> [--top-k <n>] [--hops <n>] [--json]
        Score vector-only, BM25-only and hybrid retrieval against a
        ground-truth dataset (MRR, Recall@K, context coverage, path recovery).
    nexus_graph healthcheck [addr]
        Probe GET /healthz of a running server [default: 127.0.0.1:8080].
        Exits 0 if it answers 200, 1 otherwise.

HYBRID OPTIONS (at least one of --text, --vector, --like is required):
    --text <query>          Text query for the BM25 channel
    --vector <f,f,...>      Query embedding, comma-separated
    --like <entity>         Use the embedding of this entity as the query
    --top-k <n>             Fused results to return            [default: 10]
    --hops <n>              Graph expansion depth              [default: 1]
    --rrf-k <x>             RRF smoothing constant             [default: 60]
    --vector-weight <w>     RRF weight of the vector channel   [default: 1]
    --lexical-weight <w>    RRF weight of the lexical channel  [default: 1]
    --json                  Print the full result bundle as JSON

SERVE OPTIONS:
    --bind <addr>           Socket address to listen on, e.g. 127.0.0.1:8080
    --snapshot-dir <path>   Directory of per-namespace snapshots (created if missing)
    --queue-capacity <n>    Ingestion jobs queued per namespace before 429  [default: 64]
    --timeout-secs <n>      Time allowed to produce a response              [default: 30]
    --cors-origin <origin>  Browser origin allowed to call the API (repeatable)
";

/// JSON input of `ingest`. Relations refer to entities by canonical name.
#[derive(Debug, Deserialize)]
struct GraphDocument {
    entities: Vec<EntityInput>,
    #[serde(default)]
    relations: Vec<RelationInput>,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode> {
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["ingest", input, snapshot] => ingest(input, snapshot),
        ["stats", snapshot] => stats(snapshot),
        ["verify", snapshot, claims] => verify(snapshot, claims, VerifierConfig::default()),
        ["verify", snapshot, claims, max_hops] => {
            let max_hops = parse_value("max_hops", max_hops)?;
            verify(snapshot, claims, VerifierConfig { max_hops })
        }
        ["index", snapshot] => index(snapshot),
        ["search", snapshot, query] => search(snapshot, query, RetrievalConfig::default().top_k),
        ["search", snapshot, query, top_k] => search(snapshot, query, parse_value("top_k", top_k)?),
        ["hybrid", snapshot, options @ ..] => hybrid(snapshot, options),
        ["serve", options @ ..] => serve(options),
        ["eval", options @ ..] => eval(options),
        ["healthcheck"] => healthcheck("127.0.0.1:8080"),
        ["healthcheck", addr] => healthcheck(addr),
        [] | ["help" | "--help" | "-h"] => {
            print!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        _ => bail!("unrecognized arguments\n\n{USAGE}"),
    }
}

fn parse_value<T>(name: &str, raw: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    raw.parse()
        .with_context(|| format!("invalid value `{raw}` for {name}"))
}

fn ingest(input: &str, snapshot: &str) -> Result<ExitCode> {
    let text = fs::read_to_string(input).with_context(|| format!("reading `{input}`"))?;
    let document: GraphDocument =
        serde_json::from_str(&text).with_context(|| format!("parsing `{input}`"))?;

    let graph = KnowledgeGraph::new();
    for entity in document.entities {
        let name = entity.name.clone();
        graph
            .add_entity(
                entity.name,
                entity.entity_type,
                entity.attributes,
                entity.embedding,
            )
            .with_context(|| format!("adding entity `{name}`"))?;
    }
    for relation in document.relations {
        graph
            .add_relation_by_name(
                &relation.source,
                &relation.target,
                relation.relation.clone(),
                relation.weight,
                relation.metadata,
            )
            .with_context(|| {
                format!(
                    "adding relation `{}` -[{}]-> `{}`",
                    relation.source, relation.relation, relation.target
                )
            })?;
    }

    save_snapshot(&graph, snapshot)?;
    println!(
        "wrote {snapshot}: {} entities, {} relations",
        graph.node_count(),
        graph.edge_count()
    );
    Ok(ExitCode::SUCCESS)
}

fn stats(snapshot: &str) -> Result<ExitCode> {
    let graph = load_snapshot(snapshot)?;
    println!("entities:  {}", graph.node_count());
    println!("relations: {}", graph.edge_count());
    Ok(ExitCode::SUCCESS)
}

fn verify(snapshot: &str, claims_path: &str, config: VerifierConfig) -> Result<ExitCode> {
    let graph = load_snapshot(snapshot)?;
    let text =
        fs::read_to_string(claims_path).with_context(|| format!("reading `{claims_path}`"))?;
    let claims: Vec<GraphClaim> =
        serde_json::from_str(&text).with_context(|| format!("parsing `{claims_path}`"))?;

    let report = verify_claims(&graph, &claims, config);
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(if report.is_fully_verified() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

fn load_retriever(snapshot: &str, config: RetrievalConfig) -> Result<HybridRetriever> {
    let graph = Arc::new(load_snapshot(snapshot)?);
    HybridRetriever::from_graph(graph, config).context("building retrieval indices")
}

fn index(snapshot: &str) -> Result<ExitCode> {
    let retriever = load_retriever(snapshot, RetrievalConfig::default())?;
    let stats = retriever.stats();
    println!("documents:      {}", stats.documents);
    println!("distinct terms: {}", stats.terms);
    println!("vectors:        {}", stats.vectors);
    match stats.vector_dim {
        Some(dim) => println!("vector dim:     {dim}"),
        None => println!("vector dim:     - (no embeddings)"),
    }
    println!("simd kernel:    {}", Kernel::active());
    Ok(ExitCode::SUCCESS)
}

fn search(snapshot: &str, query: &str, top_k: usize) -> Result<ExitCode> {
    let config = RetrievalConfig {
        top_k,
        vector_weight: 0.0,
        expansion_hops: 0,
        ..RetrievalConfig::default()
    };
    let retriever = load_retriever(snapshot, config)?;
    let bundle = retriever.search(&HybridQuery {
        text: Some(query),
        embedding: None,
    })?;
    print_hits(&bundle);
    Ok(ExitCode::SUCCESS)
}

fn hybrid(snapshot: &str, options: &[&str]) -> Result<ExitCode> {
    let mut config = RetrievalConfig::default();
    let mut text: Option<&str> = None;
    let mut vector: Option<Vec<f32>> = None;
    let mut like: Option<&str> = None;
    let mut json = false;

    let mut remaining = options.iter().copied();
    while let Some(flag) = remaining.next() {
        if flag == "--json" {
            json = true;
            continue;
        }
        let value = remaining
            .next()
            .ok_or_else(|| anyhow!("missing value for `{flag}`"))?;
        match flag {
            "--text" => text = Some(value),
            "--vector" => {
                let parsed = value
                    .split(',')
                    .map(|component| parse_value("--vector", component.trim()))
                    .collect::<Result<Vec<f32>>>()?;
                vector = Some(parsed);
            }
            "--like" => like = Some(value),
            "--top-k" => config.top_k = parse_value(flag, value)?,
            "--hops" => config.expansion_hops = parse_value(flag, value)?,
            "--rrf-k" => config.rrf_k = parse_value(flag, value)?,
            "--vector-weight" => config.vector_weight = parse_value(flag, value)?,
            "--lexical-weight" => config.lexical_weight = parse_value(flag, value)?,
            _ => bail!("unknown option `{flag}`\n\n{USAGE}"),
        }
    }
    if vector.is_some() && like.is_some() {
        bail!("--vector and --like are mutually exclusive");
    }

    let retriever = load_retriever(snapshot, config)?;
    if let Some(name) = like {
        let graph = retriever.graph();
        let embedding = graph
            .node_index_by_name(name)
            .and_then(|node| graph.node(node))
            .ok_or_else(|| anyhow!("no entity named `{name}`"))?
            .embedding
            .ok_or_else(|| anyhow!("entity `{name}` has no embedding"))?;
        vector = Some(embedding);
    }

    let bundle = retriever.search(&HybridQuery {
        text,
        embedding: vector.as_deref(),
    })?;
    if json {
        println!("{}", serde_json::to_string_pretty(&bundle)?);
    } else {
        print_hits(&bundle);
        print_context(&bundle, config.expansion_hops);
    }
    Ok(ExitCode::SUCCESS)
}

fn print_hits(bundle: &RetrievalBundle) {
    if bundle.hits.is_empty() {
        println!("no results");
        return;
    }
    for (position, hit) in bundle.hits.iter().enumerate() {
        let vector = hit.vector.map_or_else(String::new, |channel| {
            format!("  vector #{} cos={:.4}", channel.rank, channel.score)
        });
        let lexical = hit.lexical.map_or_else(String::new, |channel| {
            format!("  lexical #{} bm25={:.4}", channel.rank, channel.score)
        });
        let line = format!(
            "{:>2}. {} ({})  rrf={:.5}{vector}{lexical}",
            position + 1,
            hit.name,
            hit.entity_type,
            hit.fused_score
        );
        println!("{line}");
    }
}

fn print_context(bundle: &RetrievalBundle, hops: usize) {
    let context = &bundle.context;
    println!(
        "\ncontext ({hops}-hop): {} entities, {} relations",
        context.nodes.len(),
        context.edges.len()
    );
    let names: HashMap<EntityId, &str> = context
        .nodes
        .iter()
        .map(|node| (node.entity.id, node.entity.canonical_name.as_str()))
        .collect();
    let name_of = |id: EntityId| names.get(&id).copied().unwrap_or("?");
    for edge in &context.edges {
        println!(
            "    {} -[{}]-> {}  ({:.2})",
            name_of(edge.source),
            edge.relation,
            name_of(edge.target),
            edge.weight
        );
    }
}

fn serve(options: &[&str]) -> Result<ExitCode> {
    let mut bind: Option<SocketAddr> = None;
    let mut snapshot_dir: Option<&str> = None;
    let mut queue_capacity: Option<usize> = None;
    let mut timeout_secs: Option<u64> = None;
    let mut cors_origins: Vec<String> = Vec::new();

    let mut remaining = options.iter().copied();
    while let Some(flag) = remaining.next() {
        let value = remaining
            .next()
            .ok_or_else(|| anyhow!("missing value for `{flag}`"))?;
        match flag {
            "--bind" => bind = Some(parse_value(flag, value)?),
            "--snapshot-dir" => snapshot_dir = Some(value),
            "--queue-capacity" => queue_capacity = Some(parse_value(flag, value)?),
            "--timeout-secs" => timeout_secs = Some(parse_value(flag, value)?),
            "--cors-origin" => cors_origins.push(value.to_owned()),
            _ => bail!("unknown option `{flag}`\n\n{USAGE}"),
        }
    }
    let bind = bind.ok_or_else(|| anyhow!("`serve` requires --bind <addr>"))?;
    let snapshot_dir =
        snapshot_dir.ok_or_else(|| anyhow!("`serve` requires --snapshot-dir <path>"))?;

    let mut config = ServerConfig::new(snapshot_dir);
    if let Some(capacity) = queue_capacity {
        if capacity == 0 {
            bail!("--queue-capacity must be at least 1");
        }
        config.ingest_queue_capacity = capacity;
    }
    if let Some(secs) = timeout_secs {
        config.request_timeout = Duration::from_secs(secs);
    }
    config.cors_allowed_origins = cors_origins;

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .init();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?
        .block_on(server::serve(bind, config))
        .with_context(|| format!("serving on {bind}"))?;
    Ok(ExitCode::SUCCESS)
}

fn eval(options: &[&str]) -> Result<ExitCode> {
    let mut dataset_path: Option<&str> = None;
    let mut config = EvalConfig::default();
    let mut json = false;

    let mut remaining = options.iter().copied();
    while let Some(flag) = remaining.next() {
        if flag == "--json" {
            json = true;
            continue;
        }
        let value = remaining
            .next()
            .ok_or_else(|| anyhow!("missing value for `{flag}`"))?;
        match flag {
            "--dataset" => dataset_path = Some(value),
            "--top-k" => config.top_k = parse_value(flag, value)?,
            "--hops" => config.expansion_hops = parse_value(flag, value)?,
            "--rrf-k" => config.rrf_k = parse_value(flag, value)?,
            "--vector-weight" => config.vector_weight = parse_value(flag, value)?,
            "--lexical-weight" => config.lexical_weight = parse_value(flag, value)?,
            _ => bail!("unknown option `{flag}`\n\n{USAGE}"),
        }
    }
    let path = dataset_path.ok_or_else(|| anyhow!("`eval` requires --dataset <path>"))?;
    let text = fs::read_to_string(path).with_context(|| format!("reading `{path}`"))?;
    let dataset: EvalDataset =
        serde_json::from_str(&text).with_context(|| format!("parsing `{path}`"))?;

    let report = evaluate(&dataset, &config)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{report}");
    }
    Ok(ExitCode::SUCCESS)
}

/// Minimal HTTP/1.1 probe, so that a container image without a shell or curl
/// can still health-check itself.
fn healthcheck(addr: &str) -> Result<ExitCode> {
    let timeout = Duration::from_secs(2);
    let socket = addr
        .to_socket_addrs()
        .with_context(|| format!("resolving `{addr}`"))?
        .next()
        .ok_or_else(|| anyhow!("`{addr}` resolves to no address"))?;
    let mut stream = TcpStream::connect_timeout(&socket, timeout)
        .with_context(|| format!("connecting to {socket}"))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    write!(
        stream,
        "GET /healthz HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream
        .take(4096)
        .read_to_string(&mut response)
        .context("reading the response")?;
    let status_line = response.lines().next().unwrap_or_default();
    if status_line.split_whitespace().nth(1) == Some("200") {
        println!("healthy");
        Ok(ExitCode::SUCCESS)
    } else {
        bail!("unhealthy: `{status_line}`")
    }
}

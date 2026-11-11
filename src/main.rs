//! `nexus_graph` command-line interface.

use std::fs;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use nexus_graph::{
    load_snapshot, save_snapshot, verify_claims, Attributes, EntityType, GraphClaim,
    KnowledgeGraph, RelationType, VerifierConfig,
};

const USAGE: &str = "\
NexusGraph - hybrid knowledge graph with deterministic claim verification

USAGE:
    nexus_graph ingest <graph.json> <snapshot>
        Build a binary snapshot from a JSON graph description.
    nexus_graph stats <snapshot>
        Print entity and relation counts of a snapshot.
    nexus_graph verify <snapshot> <claims.json> [max_hops]
        Verify a JSON array of claims and print the report as JSON.
        Exits with status 2 if any claim is rejected.
";

/// JSON input of `ingest`. Relations refer to entities by canonical name.
#[derive(Debug, Deserialize)]
struct GraphDocument {
    entities: Vec<EntityInput>,
    #[serde(default)]
    relations: Vec<RelationInput>,
}

#[derive(Debug, Deserialize)]
struct EntityInput {
    name: String,
    #[serde(rename = "type")]
    entity_type: EntityType,
    #[serde(default)]
    attributes: Attributes,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct RelationInput {
    source: String,
    target: String,
    relation: RelationType,
    #[serde(default = "full_confidence")]
    weight: f32,
    #[serde(default)]
    metadata: Attributes,
}

fn full_confidence() -> f32 {
    1.0
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
            let max_hops = max_hops
                .parse()
                .with_context(|| format!("invalid max_hops `{max_hops}`"))?;
            verify(snapshot, claims, VerifierConfig { max_hops })
        }
        [] | ["help" | "--help" | "-h"] => {
            print!("{USAGE}");
            Ok(ExitCode::SUCCESS)
        }
        _ => bail!("unrecognized arguments\n\n{USAGE}"),
    }
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

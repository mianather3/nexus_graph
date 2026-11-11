//! Atomic binary snapshots of a [`KnowledgeGraph`].
//!
//! File layout (all integers little-endian):
//!
//! | bytes | content                               |
//! |-------|---------------------------------------|
//! | 4     | magic `NXGS`                          |
//! | 4     | format version                        |
//! | 8     | payload length                        |
//! | 8     | FNV-1a 64 checksum of the payload     |
//! | n     | bincode-encoded [`GraphSnapshot`]     |
//!
//! The payload holds the topology only. Secondary indices are rebuilt and
//! re-validated on load, so a snapshot can never restore an index that
//! disagrees with the topology.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use crate::graph::{GraphError, GraphSnapshot, KnowledgeGraph};

const MAGIC: [u8; 4] = *b"NXGS";

/// Version of the on-disk format written by this build.
pub const FORMAT_VERSION: u32 = 1;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Errors raised while saving or loading a snapshot.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("I/O error on `{}`: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("snapshot path `{}` has no file name", .0.display())]
    InvalidPath(PathBuf),
    #[error("failed to encode snapshot: {0}")]
    Encode(#[source] bincode::Error),
    #[error("failed to decode snapshot payload: {0}")]
    Decode(#[source] bincode::Error),
    #[error("not a NexusGraph snapshot (bad magic bytes)")]
    BadMagic,
    #[error(
        "unsupported snapshot format version {found} (this build reads version {FORMAT_VERSION})"
    )]
    UnsupportedVersion { found: u32 },
    #[error("snapshot is truncated or has trailing data")]
    LengthMismatch,
    #[error("snapshot checksum mismatch: the file is corrupted")]
    ChecksumMismatch,
    #[error("snapshot violates a graph invariant: {0}")]
    InvalidGraph(#[from] GraphError),
}

/// Encodes a snapshot into the framed binary format.
pub fn encode_snapshot(snapshot: &GraphSnapshot) -> Result<Vec<u8>, StorageError> {
    let payload = bincode::serialize(snapshot).map_err(StorageError::Encode)?;
    let mut bytes = Vec::with_capacity(payload.len() + 24);
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&fnv1a64(&payload).to_le_bytes());
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

/// Decodes the framed binary format, verifying magic, version, length and
/// checksum before touching the payload.
pub fn decode_snapshot(bytes: &[u8]) -> Result<GraphSnapshot, StorageError> {
    let (magic, rest) = bytes
        .split_first_chunk::<4>()
        .ok_or(StorageError::BadMagic)?;
    if *magic != MAGIC {
        return Err(StorageError::BadMagic);
    }
    let (version, rest) = rest
        .split_first_chunk::<4>()
        .ok_or(StorageError::LengthMismatch)?;
    let version = u32::from_le_bytes(*version);
    if version != FORMAT_VERSION {
        return Err(StorageError::UnsupportedVersion { found: version });
    }
    let (length, rest) = rest
        .split_first_chunk::<8>()
        .ok_or(StorageError::LengthMismatch)?;
    let (checksum, payload) = rest
        .split_first_chunk::<8>()
        .ok_or(StorageError::LengthMismatch)?;
    if u64::from_le_bytes(*length) != payload.len() as u64 {
        return Err(StorageError::LengthMismatch);
    }
    if u64::from_le_bytes(*checksum) != fnv1a64(payload) {
        return Err(StorageError::ChecksumMismatch);
    }
    bincode::deserialize(payload).map_err(StorageError::Decode)
}

/// Writes a consistent snapshot of `graph` to `path`, atomically.
///
/// The data is written to a temporary file in the same directory, flushed to
/// disk, and then renamed over `path`. A crash at any point leaves either the
/// previous snapshot or the new one, never a partial file.
pub fn save_snapshot(graph: &KnowledgeGraph, path: impl AsRef<Path>) -> Result<(), StorageError> {
    let path = path.as_ref();
    let bytes = encode_snapshot(&graph.snapshot())?;

    let file_name = path
        .file_name()
        .ok_or_else(|| StorageError::InvalidPath(path.to_path_buf()))?;
    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let mut temp_name = OsString::from(".");
    temp_name.push(file_name);
    temp_name.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let temp_path = dir.join(temp_name);

    let written = write_durably(&temp_path, &bytes).and_then(|()| fs::rename(&temp_path, path));
    if let Err(source) = written {
        // Best effort: the temp file may not exist if creation itself failed.
        let _ = fs::remove_file(&temp_path);
        return Err(StorageError::Io {
            path: path.to_path_buf(),
            source,
        });
    }
    sync_directory(dir);
    Ok(())
}

/// Loads a snapshot from `path` and rebuilds the graph and its indices.
pub fn load_snapshot(path: impl AsRef<Path>) -> Result<KnowledgeGraph, StorageError> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|source| StorageError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let snapshot = decode_snapshot(&bytes)?;
    Ok(KnowledgeGraph::from_snapshot(snapshot)?)
}

fn write_durably(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Persists the rename itself. Best effort: not every platform or filesystem
/// supports syncing a directory, and the snapshot is already complete.
fn sync_directory(dir: &Path) {
    #[cfg(unix)]
    if let Ok(handle) = File::open(dir) {
        let _ = handle.sync_all();
    }
    #[cfg(not(unix))]
    let _ = dir;
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    })
}

//! Flat, contiguous dense vector store with exact cosine k-NN search.
//!
//! All vectors live back to back in one `Vec<f32>` (row `i` occupies
//! `data[i * dim..(i + 1) * dim]`), so a scan walks memory linearly with no
//! pointer chasing. Vectors are L2-normalized on the way in, which turns
//! cosine similarity into a single dot product per row at query time; the
//! original magnitudes are kept in a parallel array.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::NexusError;
use crate::models::EntityId;
use crate::topk::TopK;
use crate::vector::{checked_norm, dot_unchecked};

/// One k-NN result.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VectorHit {
    pub id: EntityId,
    /// Cosine similarity in `[-1.0, 1.0]`.
    pub similarity: f32,
    /// Cosine distance `1 - similarity`, in `[0.0, 2.0]`.
    pub distance: f32,
}

/// Dense vector store of a fixed dimension.
#[derive(Debug, Clone)]
pub struct VectorStore {
    dim: usize,
    /// Row-major unit vectors, `len == ids.len() * dim`.
    data: Vec<f32>,
    /// Original magnitude of each row.
    norms: Vec<f32>,
    ids: Vec<EntityId>,
    rows: HashMap<EntityId, usize>,
}

impl VectorStore {
    /// Creates an empty store for vectors of `dim` components.
    pub fn new(dim: usize) -> Result<Self, NexusError> {
        if dim == 0 {
            return Err(NexusError::ZeroDimension);
        }
        Ok(Self {
            dim,
            data: Vec::new(),
            norms: Vec::new(),
            ids: Vec::new(),
            rows: HashMap::new(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Approximate heap bytes held by the store: the flat vector buffer, the
    /// norms, the ids and the id-to-row map (estimated at its load factor).
    pub fn memory_bytes(&self) -> usize {
        use std::mem::size_of;
        // hashbrown: one control byte per bucket plus the (key, value) slot.
        let map_slot = size_of::<(EntityId, usize)>() + 1;
        self.data.capacity() * size_of::<f32>()
            + self.norms.capacity() * size_of::<f32>()
            + self.ids.capacity() * size_of::<EntityId>()
            + self.rows.capacity() * map_slot
    }

    pub fn contains(&self, id: EntityId) -> bool {
        self.rows.contains_key(&id)
    }

    /// The stored unit vector of `id`, borrowed straight from the flat buffer.
    pub fn get(&self, id: EntityId) -> Option<&[f32]> {
        let row = *self.rows.get(&id)?;
        self.data.chunks_exact(self.dim).nth(row)
    }

    /// The magnitude `id`'s vector had before normalization.
    pub fn norm(&self, id: EntityId) -> Option<f32> {
        let row = *self.rows.get(&id)?;
        self.norms.get(row).copied()
    }

    /// Appends one vector. Fails, leaving the store untouched, if the id is
    /// already present or the vector has the wrong dimension, a NaN/Inf
    /// component, or zero magnitude.
    pub fn insert(&mut self, id: EntityId, vector: &[f32]) -> Result<(), NexusError> {
        self.check_dim(vector.len())?;
        if self.rows.contains_key(&id) {
            return Err(NexusError::DuplicateDocument(id));
        }
        let norm = checked_norm(vector)?;

        self.data.extend(vector.iter().map(|v| v / norm));
        self.norms.push(norm);
        self.rows.insert(id, self.ids.len());
        self.ids.push(id);
        Ok(())
    }

    /// Appends `ids.len()` vectors given as one row-major buffer.
    ///
    /// The whole batch is validated before anything is stored, so a failure
    /// leaves the store untouched. If the store is empty the buffer is adopted
    /// as the backing storage and normalized in place, without copying.
    pub fn bulk_load(
        &mut self,
        ids: Vec<EntityId>,
        mut vectors: Vec<f32>,
    ) -> Result<(), NexusError> {
        let expected = ids
            .len()
            .checked_mul(self.dim)
            .ok_or(NexusError::CapacityExceeded)?;
        if vectors.len() != expected {
            return Err(NexusError::BufferLengthMismatch {
                ids: ids.len(),
                dim: self.dim,
                expected,
                actual: vectors.len(),
            });
        }
        let mut batch_ids: HashSet<EntityId> = HashSet::with_capacity(ids.len());
        for &id in &ids {
            if self.rows.contains_key(&id) || !batch_ids.insert(id) {
                return Err(NexusError::DuplicateDocument(id));
            }
        }
        let norms = vectors
            .chunks_exact(self.dim)
            .map(checked_norm)
            .collect::<Result<Vec<f32>, NexusError>>()?;

        for (row, norm) in vectors.chunks_exact_mut(self.dim).zip(&norms) {
            for value in row {
                *value /= norm;
            }
        }
        let first_row = self.ids.len();
        if self.data.is_empty() {
            self.data = vectors;
        } else {
            self.data.extend_from_slice(&vectors);
        }
        self.norms.extend(norms);
        for (offset, &id) in ids.iter().enumerate() {
            self.rows.insert(id, first_row + offset);
        }
        self.ids.extend(ids);
        Ok(())
    }

    /// Exact cosine k-NN: scans every row and keeps the `top_k` most similar
    /// in a bounded min-heap. Results are sorted by descending similarity,
    /// ties by ascending id.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>, NexusError> {
        self.check_dim(query.len())?;
        let norm = checked_norm(query)?;
        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let unit_query: Vec<f32> = query.iter().map(|v| v / norm).collect();

        let mut best = TopK::new(top_k);
        for (row, &id) in self.data.chunks_exact(self.dim).zip(&self.ids) {
            best.push(id, dot_unchecked(&unit_query, row));
        }
        Ok(best
            .into_sorted()
            .into_iter()
            .map(|(id, score)| {
                let similarity = score.clamp(-1.0, 1.0);
                VectorHit {
                    id,
                    similarity,
                    distance: 1.0 - similarity,
                }
            })
            .collect())
    }

    fn check_dim(&self, actual: usize) -> Result<(), NexusError> {
        if actual == self.dim {
            Ok(())
        } else {
            Err(NexusError::DimensionMismatch {
                expected: self.dim,
                actual,
            })
        }
    }
}

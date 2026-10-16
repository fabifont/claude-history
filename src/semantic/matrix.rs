//! Passage embeddings stored as one memory-mapped `f32` matrix, keyed by
//! the hash of the passage text.
//!
//! Layout of the file:
//!
//! ```text
//! Header | EmbeddingKey[count] (ascending) | last_used: u64[count] |
//! protected: u8[count] (padded to 64) | f32[count * dims]
//! ```
//!
//! A file is written whole into a temporary file and renamed into place, and
//! never modified afterwards, so a reader's map stays valid while newer
//! files replace it.

use bytemuck::{Pod, Zeroable};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::ops::Range;
use std::path::Path;

const MAGIC: [u8; 8] = *b"CHEMBMX\0";
const VERSION: u32 = 1;
/// Alignment of the vector section, so rows can be read as `f32` slices.
const VECTOR_ALIGN: usize = 64;

/// Hash of a passage's text: the first 16 bytes of its blake3 hash.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Pod, Zeroable)]
pub struct EmbeddingKey(pub [u8; 16]);

impl EmbeddingKey {
    pub fn of(text: &str) -> Self {
        let hash = blake3::hash(text.as_bytes());
        Self(hash.as_bytes()[..16].try_into().expect("16-byte slice"))
    }

    /// Key of a hex-encoded blake3 hash, as `embeddings-v1.bin` stored them.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hash = blake3::Hash::from_hex(hex).ok()?;
        Some(Self(hash.as_bytes()[..16].try_into().ok()?))
    }
}

/// What the stored embeddings were computed with; a file whose settings
/// differ from the current ones is not used.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct MatrixSettings {
    /// Hash of the model name and the cache schema version.
    pub model_hash: u64,
    pub chunk_target_chars: u32,
    pub chunk_overlap_chars: u32,
    pub chunk_context_turns: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Header {
    magic: [u8; 8],
    version: u32,
    dims: u32,
    count: u64,
    access_counter: u64,
    settings: MatrixSettings,
}

/// One row to write: key, last use, whether it is protected from pruning,
/// and its vector.
pub struct MatrixRow<'a> {
    pub key: EmbeddingKey,
    pub last_used: u64,
    pub protected: bool,
    pub vector: &'a [f32],
}

pub struct EmbeddingMatrix {
    map: memmap2::Mmap,
    header: Header,
    keys: Range<usize>,
    last_used: Range<usize>,
    protected: Range<usize>,
    vectors: Range<usize>,
}

struct Layout {
    keys: Range<usize>,
    last_used: Range<usize>,
    protected: Range<usize>,
    vectors: Range<usize>,
}

fn layout(count: usize, dims: usize) -> Option<Layout> {
    let keys_start = std::mem::size_of::<Header>();
    let keys = keys_start..keys_start.checked_add(count.checked_mul(16)?)?;
    let last_used = keys.end..keys.end.checked_add(count.checked_mul(8)?)?;
    let protected = last_used.end..last_used.end.checked_add(count)?;
    let vectors_start = protected.end.div_ceil(VECTOR_ALIGN) * VECTOR_ALIGN;
    let vectors =
        vectors_start..vectors_start.checked_add(count.checked_mul(dims)?.checked_mul(4)?)?;
    Some(Layout {
        keys,
        last_used,
        protected,
        vectors,
    })
}

impl EmbeddingMatrix {
    /// Maps the matrix at `path`; `None` when it is missing or invalid.
    pub fn open(path: &Path) -> Option<Self> {
        let file = File::open(path).ok()?;
        // SAFETY: matrix files are written whole, renamed into place and
        // never modified, so the mapped bytes cannot change underneath.
        let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
        let header_len = std::mem::size_of::<Header>();
        let header: Header = bytemuck::try_pod_read_unaligned(map.get(..header_len)?).ok()?;
        if header.magic != MAGIC || header.version != VERSION {
            return None;
        }
        let count = usize::try_from(header.count).ok()?;
        let dims = header.dims as usize;
        let layout = layout(count, dims)?;
        if layout.vectors.end != map.len() || (count > 0 && dims == 0) {
            return None;
        }
        let matrix = Self {
            header,
            keys: layout.keys,
            last_used: layout.last_used,
            protected: layout.protected,
            vectors: layout.vectors,
            map,
        };
        let keys = matrix.keys();
        let sorted = keys.windows(2).all(|pair| pair[0] < pair[1]);
        let aligned = bytemuck::try_cast_slice::<u8, f32>(&matrix.map[matrix.vectors.clone()])
            .is_ok()
            && bytemuck::try_cast_slice::<u8, u64>(&matrix.map[matrix.last_used.clone()]).is_ok();
        (sorted && aligned).then_some(matrix)
    }

    pub fn settings(&self) -> MatrixSettings {
        self.header.settings
    }

    pub fn access_counter(&self) -> u64 {
        self.header.access_counter
    }

    pub fn dims(&self) -> usize {
        self.header.dims as usize
    }

    pub fn keys(&self) -> &[EmbeddingKey] {
        bytemuck::cast_slice(&self.map[self.keys.clone()])
    }

    pub fn last_used(&self, row: u32) -> u64 {
        bytemuck::cast_slice::<u8, u64>(&self.map[self.last_used.clone()])[row as usize]
    }

    pub fn protected(&self, row: u32) -> bool {
        self.map[self.protected.start + row as usize] != 0
    }

    pub fn vector(&self, row: u32) -> &[f32] {
        let dims = self.dims();
        let start = row as usize * dims;
        &bytemuck::cast_slice::<u8, f32>(&self.map[self.vectors.clone()])[start..start + dims]
    }

    /// Writes `rows` (any order, unique keys, all `dims` long) to `path`
    /// through a temporary file in the same directory renamed into place.
    pub fn write(
        path: &Path,
        settings: MatrixSettings,
        access_counter: u64,
        dims: usize,
        mut rows: Vec<MatrixRow<'_>>,
    ) -> io::Result<()> {
        rows.sort_unstable_by_key(|row| row.key);
        rows.dedup_by_key(|row| row.key);
        if rows.iter().any(|row| row.vector.len() != dims) {
            return Err(io::Error::other("embedding rows differ in dimensions"));
        }
        let parent = path
            .parent()
            .ok_or_else(|| io::Error::other("matrix path has no parent"))?;
        std::fs::create_dir_all(parent)?;
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            dims: u32::try_from(dims).map_err(|_| io::Error::other("too many dimensions"))?,
            count: rows.len() as u64,
            access_counter,
            settings,
        };
        let layout =
            layout(rows.len(), dims).ok_or_else(|| io::Error::other("matrix too large"))?;
        let tmp = tempfile::NamedTempFile::new_in(parent)?;
        {
            let mut out = BufWriter::with_capacity(1 << 20, tmp.as_file());
            out.write_all(bytemuck::bytes_of(&header))?;
            for row in &rows {
                out.write_all(&row.key.0)?;
            }
            for row in &rows {
                out.write_all(&row.last_used.to_le_bytes())?;
            }
            for row in &rows {
                out.write_all(&[u8::from(row.protected)])?;
            }
            out.write_all(&vec![0; layout.vectors.start - layout.protected.end])?;
            for row in &rows {
                out.write_all(bytemuck::cast_slice(row.vector))?;
            }
            out.flush()?;
        }
        tmp.as_file().sync_all()?;
        tmp.persist(path).map_err(|error| error.error)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl EmbeddingMatrix {
        fn find(&self, key: &EmbeddingKey) -> Option<u32> {
            self.keys().binary_search(key).ok().map(|row| row as u32)
        }
    }

    fn settings() -> MatrixSettings {
        MatrixSettings {
            model_hash: 7,
            chunk_target_chars: 1,
            chunk_overlap_chars: 2,
            chunk_context_turns: 3,
            _pad: 0,
        }
    }

    #[test]
    fn round_trips_rows_sorted_by_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("matrix.bin");
        let texts = ["gamma", "alpha", "beta"];
        let vectors: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32, 0.5, -1.0]).collect();
        let rows = texts
            .iter()
            .zip(&vectors)
            .enumerate()
            .map(|(i, (text, vector))| MatrixRow {
                key: EmbeddingKey::of(text),
                last_used: i as u64 * 10,
                protected: i == 1,
                vector,
            })
            .collect();
        EmbeddingMatrix::write(&path, settings(), 42, 3, rows).unwrap();

        let matrix = EmbeddingMatrix::open(&path).unwrap();
        assert_eq!(matrix.keys().len(), 3);
        assert_eq!(matrix.settings(), settings());
        assert_eq!(matrix.access_counter(), 42);
        for (i, text) in texts.iter().enumerate() {
            let row = matrix.find(&EmbeddingKey::of(text)).unwrap();
            assert_eq!(matrix.vector(row), vectors[i].as_slice());
            assert_eq!(matrix.last_used(row), i as u64 * 10);
            assert_eq!(matrix.protected(row), i == 1);
        }
        assert_eq!(matrix.find(&EmbeddingKey::of("delta")), None);
    }

    #[test]
    fn rejects_truncated_or_foreign_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("matrix.bin");
        let vector = [1.0, 2.0];
        EmbeddingMatrix::write(
            &path,
            settings(),
            0,
            2,
            vec![MatrixRow {
                key: EmbeddingKey::of("a"),
                last_used: 0,
                protected: false,
                vector: &vector,
            }],
        )
        .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(EmbeddingMatrix::open(&path).is_some());
        std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
        assert!(EmbeddingMatrix::open(&path).is_none());
        std::fs::write(&path, b"not a matrix").unwrap();
        assert!(EmbeddingMatrix::open(&path).is_none());
        assert!(
            EmbeddingMatrix::write(
                &path,
                settings(),
                0,
                3,
                vec![MatrixRow {
                    key: EmbeddingKey::of("a"),
                    last_used: 0,
                    protected: false,
                    vector: &vector,
                }],
            )
            .is_err()
        );
    }

    #[test]
    fn hex_keys_match_text_keys() {
        let hex = blake3::hash(b"passage").to_hex().to_string();
        assert_eq!(
            EmbeddingKey::from_hex(&hex),
            Some(EmbeddingKey::of("passage"))
        );
        assert_eq!(EmbeddingKey::from_hex("zz"), None);
    }
}

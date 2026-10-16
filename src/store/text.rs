//! Conversation text, stored once as one zstd frame per conversation.
//!
//! A generation's text lives in segment files `text-NNNN.zst` inside its
//! directory, and each card points at its frame with a `BlobRef`. Segments
//! are written whole and never modified, so a later generation can link an
//! unchanged segment instead of rewriting it and add new segments beside it;
//! collection still only unlinks whole directories.

use super::card::{BlobRef, CardRow};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

/// zstd level for text frames: fast to write, about 4x smaller than the text.
const LEVEL: i32 = 3;
/// Conversations compressed per parallel batch while writing a segment.
const BATCH: usize = 256;
/// Largest frame content accepted when reading, so a corrupt header cannot
/// trigger a huge allocation.
const MAX_TEXT: u64 = 1 << 32;

pub fn segment_path(dir: &Path, seg: u32) -> PathBuf {
    dir.join(format!("text-{seg:04}.zst"))
}

/// Writes segment `seg` to `path` with one frame per entry of `texts`, in
/// order, and returns their refs. Empty texts get an empty ref and no frame.
pub fn write_segment<S: AsRef<str> + Sync>(
    path: &Path,
    seg: u32,
    texts: &[S],
) -> io::Result<Vec<BlobRef>> {
    let file = File::create(path)?;
    let mut out = BufWriter::with_capacity(1 << 20, &file);
    let mut refs = Vec::with_capacity(texts.len());
    let mut off = 0u64;
    for batch in texts.chunks(BATCH) {
        let frames: Vec<Vec<u8>> = batch
            .par_iter()
            .map(|text| compress(text.as_ref().as_bytes()))
            .collect::<io::Result<_>>()?;
        for frame in frames {
            let len = u32::try_from(frame.len())
                .map_err(|_| io::Error::other("text frame exceeds 4 GiB"))?;
            refs.push(if len == 0 {
                BlobRef::default()
            } else {
                BlobRef { seg, len, off }
            });
            out.write_all(&frame)?;
            off += u64::from(len);
        }
    }
    out.flush()?;
    drop(out);
    file.sync_all()?;
    Ok(refs)
}

fn compress(text: &[u8]) -> io::Result<Vec<u8>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    thread_local! {
        static COMPRESSOR: RefCell<Option<zstd::bulk::Compressor<'static>>> =
            const { RefCell::new(None) };
    }
    COMPRESSOR.with(|cell| {
        let mut cell = cell.borrow_mut();
        let compressor = match &mut *cell {
            Some(compressor) => compressor,
            empty => {
                let mut compressor = zstd::bulk::Compressor::new(LEVEL)?;
                compressor.include_checksum(true)?;
                empty.insert(compressor)
            }
        };
        compressor.compress(text)
    })
}

/// The memory-mapped text segments of one generation.
pub struct TextSegments {
    segments: Vec<memmap2::Mmap>,
}

impl TextSegments {
    /// Maps the segments in `dir` that `rows` refer to. Returns `None` when
    /// one is missing or a ref points outside its segment.
    pub fn open(dir: &Path, rows: &[CardRow]) -> Option<Self> {
        let count = rows
            .iter()
            .filter(|row| row.text.len > 0)
            .map(|row| row.text.seg + 1)
            .max()
            .unwrap_or(0);
        let mut segments = Vec::with_capacity(count as usize);
        for seg in 0..count {
            let file = File::open(segment_path(dir, seg)).ok()?;
            // SAFETY: published segments are never written again and are only
            // removed by unlinking, so the mapped bytes cannot change underneath.
            segments.push(unsafe { memmap2::Mmap::map(&file) }.ok()?);
        }
        let in_bounds = |r: BlobRef| {
            r.len == 0
                || segments.get(r.seg as usize).is_some_and(|segment| {
                    r.off
                        .checked_add(u64::from(r.len))
                        .is_some_and(|end| end <= segment.len() as u64)
                })
        };
        rows.iter()
            .all(|row| in_bounds(row.text))
            .then_some(Self { segments })
    }

    /// Decompresses the text behind `r`. A frame that fails to decode or
    /// whose checksum does not match reads as `None`.
    pub fn read(&self, r: BlobRef) -> Option<String> {
        if r.len == 0 {
            return Some(String::new());
        }
        let segment = self.segments.get(r.seg as usize)?;
        let start = usize::try_from(r.off).ok()?;
        let frame = segment.get(start..start.checked_add(r.len as usize)?)?;
        decompress(frame)
    }

    /// Total bytes of all segments.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn disk_bytes(&self) -> u64 {
        self.segments
            .iter()
            .map(|segment| segment.len() as u64)
            .sum()
    }
}

fn decompress(frame: &[u8]) -> Option<String> {
    let size = zstd::zstd_safe::get_frame_content_size(frame).ok()??;
    if size > MAX_TEXT {
        return None;
    }
    thread_local! {
        static DECOMPRESSOR: RefCell<Option<zstd::bulk::Decompressor<'static>>> =
            const { RefCell::new(None) };
    }
    let bytes = DECOMPRESSOR.with(|cell| {
        let mut cell = cell.borrow_mut();
        let decompressor = match &mut *cell {
            Some(decompressor) => decompressor,
            empty => empty.insert(zstd::bulk::Decompressor::new().ok()?),
        };
        let bytes = decompressor.decompress(frame, size as usize).ok()?;
        (bytes.len() as u64 == size).then_some(bytes)
    })?;
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    fn rows(refs: &[BlobRef]) -> Vec<CardRow> {
        refs.iter()
            .map(|&text| CardRow {
                text,
                ..CardRow::zeroed()
            })
            .collect()
    }

    #[test]
    fn round_trips_frames_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let texts: Vec<String> = (0..600)
            .map(|i| match i % 3 {
                0 => String::new(),
                1 => format!("conversation {i} é 测试 {}", "word ".repeat(i)),
                _ => "x".repeat(i * 37),
            })
            .collect();
        let refs = write_segment(&segment_path(dir.path(), 0), 0, &texts).unwrap();
        assert_eq!(refs.len(), texts.len());
        assert_eq!(refs[0], BlobRef::default());
        let segments = TextSegments::open(dir.path(), &rows(&refs)).unwrap();
        for (text, r) in texts.iter().zip(&refs) {
            assert_eq!(segments.read(*r).as_deref(), Some(text.as_str()));
        }
        let raw: usize = texts.iter().map(String::len).sum();
        assert!(segments.disk_bytes() < raw as u64 / 4);
    }

    #[test]
    fn open_rejects_missing_segments_and_refs_out_of_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let refs = write_segment(&segment_path(dir.path(), 0), 0, &["alpha", "beta"]).unwrap();
        assert!(TextSegments::open(dir.path(), &rows(&refs)).is_some());

        let mut past_end = refs.clone();
        past_end[1].off += 1;
        assert!(TextSegments::open(dir.path(), &rows(&past_end)).is_none());
        let mut other_segment = refs.clone();
        other_segment[0].seg = 1;
        assert!(TextSegments::open(dir.path(), &rows(&other_segment)).is_none());
    }

    #[test]
    fn corrupt_frames_read_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(dir.path(), 0);
        let refs = write_segment(&path, 0, &["alpha beta gamma"]).unwrap();
        let good = std::fs::read(&path).unwrap();
        for i in 0..good.len() {
            let mut bytes = good.clone();
            bytes[i] ^= 0x55;
            std::fs::remove_file(&path).unwrap();
            std::fs::write(&path, &bytes).unwrap();
            let segments = TextSegments::open(dir.path(), &rows(&refs)).unwrap();
            assert_eq!(segments.read(refs[0]), None, "byte {i}");
        }
    }
}

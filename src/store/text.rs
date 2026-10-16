//! Conversation text, stored once as one zstd frame per conversation.
//!
//! A generation's text lives in segment files `text-NNNN.zst` inside its
//! directory, and each card points at its frame with a `BlobRef`. Segments
//! are written whole and never modified, so a later generation can link an
//! unchanged segment instead of rewriting it and add new segments beside it;
//! collection still only unlinks whole directories. Semantic data (see
//! `semantic`) is stored the same way in `sem-NNNN.zst`.

use super::card::BlobRef;
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

/// File name prefix of conversation text segments.
pub const TEXT: &str = "text";
/// File name prefix of semantic data segments.
pub const SEM: &str = "sem";

pub fn segment_file(seg: u32) -> String {
    kind_file(TEXT, seg)
}

pub fn segment_path(dir: &Path, seg: u32) -> PathBuf {
    dir.join(segment_file(seg))
}

/// Segment `seg` of the segments named `kind` (`TEXT` or `SEM`).
pub fn kind_file(kind: &str, seg: u32) -> String {
    format!("{kind}-{seg:04}.zst")
}

/// Writes segment `seg` to `path` with one frame per entry of `texts`, in
/// order, and returns their refs. Empty texts get an empty ref and no frame.
pub fn write_segment<S: AsRef<[u8]> + Sync>(
    path: &Path,
    seg: u32,
    texts: &[S],
) -> io::Result<Vec<BlobRef>> {
    let mut out = SegmentWriter::create(path, seg)?;
    let mut refs = Vec::with_capacity(texts.len());
    for batch in texts.chunks(BATCH) {
        let frames: Vec<Vec<u8>> = batch
            .par_iter()
            .map(|text| compress(text.as_ref()))
            .collect::<io::Result<_>>()?;
        for frame in frames {
            refs.push(out.push(&frame)?);
        }
    }
    out.finish()?;
    Ok(refs)
}

/// Writes segment `seg` to `path` from already compressed `frames`, in
/// order, and returns their refs. Empty frames get an empty ref.
pub fn write_frames(path: &Path, seg: u32, frames: &[&[u8]]) -> io::Result<Vec<BlobRef>> {
    let mut out = SegmentWriter::create(path, seg)?;
    let refs = frames
        .iter()
        .map(|frame| out.push(frame))
        .collect::<io::Result<_>>()?;
    out.finish()?;
    Ok(refs)
}

struct SegmentWriter {
    file: File,
    out: BufWriter<File>,
    seg: u32,
    off: u64,
}

impl SegmentWriter {
    fn create(path: &Path, seg: u32) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            out: BufWriter::with_capacity(1 << 20, file.try_clone()?),
            file,
            seg,
            off: 0,
        })
    }

    fn push(&mut self, frame: &[u8]) -> io::Result<BlobRef> {
        let len =
            u32::try_from(frame.len()).map_err(|_| io::Error::other("text frame exceeds 4 GiB"))?;
        if len == 0 {
            return Ok(BlobRef::default());
        }
        self.out.write_all(frame)?;
        let r = BlobRef {
            seg: self.seg,
            len,
            off: self.off,
        };
        self.off += u64::from(len);
        Ok(r)
    }

    fn finish(mut self) -> io::Result<()> {
        self.out.flush()?;
        drop(self.out);
        self.file.sync_all()
    }
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
    /// Indexed by segment number; `None` for segments no card refers to.
    segments: Vec<Option<memmap2::Mmap>>,
}

impl TextSegments {
    /// Maps the `kind` segments in `dir` that `refs` point into. Returns
    /// `None` when one is missing or a ref points outside its segment.
    pub fn open(
        dir: &Path,
        kind: &str,
        refs: impl Iterator<Item = BlobRef> + Clone,
    ) -> Option<Self> {
        let mut segments: Vec<Option<memmap2::Mmap>> = Vec::new();
        for r in refs.clone().filter(|r| r.len > 0) {
            let seg = r.seg as usize;
            if segments.len() <= seg {
                segments.resize_with(seg + 1, || None);
            }
            if segments[seg].is_none() {
                let file = File::open(dir.join(kind_file(kind, r.seg))).ok()?;
                // SAFETY: published segments are never written again and are
                // only removed by unlinking, so the mapped bytes cannot change
                // underneath.
                segments[seg] = Some(unsafe { memmap2::Mmap::map(&file) }.ok()?);
            }
        }
        let text = Self { segments };
        refs.into_iter()
            .all(|r| r.len == 0 || text.frame(r).is_some())
            .then_some(text)
    }

    /// The compressed frame behind `r`; `None` when out of bounds.
    pub fn frame(&self, r: BlobRef) -> Option<&[u8]> {
        let segment = self.segments.get(r.seg as usize)?.as_ref()?;
        let start = usize::try_from(r.off).ok()?;
        segment.get(start..start.checked_add(r.len as usize)?)
    }

    /// Decompresses the text behind `r`. A frame that fails to decode or
    /// whose checksum does not match reads as `None`.
    pub fn read(&self, r: BlobRef) -> Option<String> {
        String::from_utf8(self.read_bytes(r)?).ok()
    }

    /// Decompresses the bytes behind `r`, like `read`.
    pub fn read_bytes(&self, r: BlobRef) -> Option<Vec<u8>> {
        if r.len == 0 {
            return Some(Vec::new());
        }
        decompress(self.frame(r)?)
    }

    /// Total bytes of all mapped segments.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn disk_bytes(&self) -> u64 {
        self.segments
            .iter()
            .flatten()
            .map(|segment| segment.len() as u64)
            .sum()
    }
}

fn decompress(frame: &[u8]) -> Option<Vec<u8>> {
    let size = zstd::zstd_safe::get_frame_content_size(frame).ok()??;
    if size > MAX_TEXT {
        return None;
    }
    thread_local! {
        static DECOMPRESSOR: RefCell<Option<zstd::bulk::Decompressor<'static>>> =
            const { RefCell::new(None) };
    }
    DECOMPRESSOR.with(|cell| {
        let mut cell = cell.borrow_mut();
        let decompressor = match &mut *cell {
            Some(decompressor) => decompressor,
            empty => empty.insert(zstd::bulk::Decompressor::new().ok()?),
        };
        let bytes = decompressor.decompress(frame, size as usize).ok()?;
        (bytes.len() as u64 == size).then_some(bytes)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(dir: &Path, refs: &[BlobRef]) -> Option<TextSegments> {
        TextSegments::open(dir, TEXT, refs.iter().copied())
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
        let segments = open(dir.path(), &refs).unwrap();
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
        assert!(open(dir.path(), &refs).is_some());

        let mut past_end = refs.clone();
        past_end[1].off += 1;
        assert!(open(dir.path(), &past_end).is_none());
        let mut other_segment = refs.clone();
        other_segment[0].seg = 1;
        assert!(open(dir.path(), &other_segment).is_none());
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
            let segments = open(dir.path(), &refs).unwrap();
            assert_eq!(segments.read(refs[0]), None, "byte {i}");
        }
    }
}

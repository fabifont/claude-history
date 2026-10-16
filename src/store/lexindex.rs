//! Word-start suffix index over normalized conversation text.
//!
//! Files in a generation directory:
//!
//! ```text
//! base.fst    header | PartRow[] | one fst map per term range
//! base.post   header | postings
//! base.short  header | ShortRow[] | columns
//! ```
//!
//! Each segment indexes every `Field` of its documents in its own files,
//! `own-base.*` for `Field::Own`, under the same doc ids.
//!
//! A document has one term per word start of its normalized text: the rest
//! of that space-delimited chunk, cut to `TERM_CAP` bytes on a char
//! boundary. When the chunk rest fits uncut and the chunk is followed by a
//! single space and another chunk, the term continues with that space and
//! the next chunk's `pair_prefix`: its first `PAIR_PREFIX` bytes, or its
//! first char when that char is longer.
//!
//! Normalized query words contain no spaces and every prefix match starts at
//! a word start inside one chunk, so the summed term frequencies of all terms
//! starting with a word count its word-start occurrences. Likewise the terms
//! starting with an adjacent pair (`search::lexical::adjacent_pairs`) count
//! the pair's occurrences. These equal what `count_prefix_matches` and
//! `count_adjacent_pairs` count unless matches can overlap or a word is
//! longer than `TERM_CAP` (see `search::is_exact_word`).
//!
//! Ideographs (U+4E00..=U+9FFF) that follow an alphanumeric char in their
//! chunk also get an inner term: a space, then the rest of the chunk cut to
//! `TERM_CAP` bytes. No ordinary term or query word starts with a space, so
//! inner terms leave word-start counts untouched while letting `inner_tf`
//! find ideographic words inside runs of CJK text.
//!
//! Term frequencies are capped at `TF_CAP` per (term, document). Doc ids are
//! row indices of the generation's cards, newest first.
//!
//! An fst value is either a single inline posting (`INLINE` set, tf in bits
//! 32..40, doc id in the low 32 bits) or an offset, relative to the
//! partition's start in `base.post`, of `varint count` followed by
//! `count` × `varint (doc id delta << 4 | tf)` in doc order. Words of at most
//! `SHORT_LEN` bytes would visit a large share of all terms, so their
//! per-document tf is materialized in `base.short` instead, as a dense
//! `[u8; n_docs]` column or a sparse posting list, whichever is smaller.

use super::generation::write_synced;
use crate::search::lexical::pair_prefix;
use crate::text_match::truncate_on_char_boundary;
use bytemuck::{Pod, Zeroable};
use fst::{IntoStreamer, Streamer};
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io;
use std::ops::Range;
use std::path::Path;

#[cfg(test)]
pub const FST_FILE: &str = "base.fst";
#[cfg(test)]
pub const POST_FILE: &str = "base.post";
#[cfg(test)]
pub const SHORT_FILE: &str = "base.short";

/// The text of a conversation an index covers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Field {
    /// `Conversation::full_text`.
    Body,
    /// `Conversation::own_text`.
    Own,
}

impl Field {
    pub const ALL: [Field; 2] = [Field::Body, Field::Own];
}

/// File stem of index segment `id`: `base` for segment 0, `delta-NNNN` for
/// the others, prefixed with `own-` for `Field::Own`.
pub fn segment_stem(id: u32, field: Field) -> String {
    let stem = if id == 0 {
        "base".to_owned()
    } else {
        format!("delta-{id:04}")
    };
    match field {
        Field::Body => stem,
        Field::Own => format!("own-{stem}"),
    }
}

/// The three file names of `field` in index segment `id`.
pub fn segment_files(id: u32, field: Field) -> [String; 3] {
    let stem = segment_stem(id, field);
    [
        format!("{stem}.fst"),
        format!("{stem}.post"),
        format!("{stem}.short"),
    ]
}

/// Every file of index segment `id`, all fields.
pub fn all_segment_files(id: u32) -> impl Iterator<Item = String> {
    Field::ALL
        .into_iter()
        .flat_map(move |field| segment_files(id, field))
}

/// `remap` value of a document that no longer exists.
pub const DEAD: u32 = u32::MAX;

/// Longest stored term, in bytes.
pub const TERM_CAP: usize = 24;
/// Highest stored term frequency, matching the scorer's cap.
pub const TF_CAP: u8 = 10;
/// Words up to this many bytes are answered from `base.short`.
pub const SHORT_LEN: usize = 2;

const FST_MAGIC: [u8; 8] = *b"CHLXFST\0";
const POST_MAGIC: [u8; 8] = *b"CHLXPST\0";
const SHORT_MAGIC: [u8; 8] = *b"CHLXSHT\0";
const VERSION: u32 = 2;
const _: () = assert!(TF_CAP < 16);
const INLINE: u64 = 1 << 63;
/// Upper bound on fst partitions, which are built in parallel.
const MAX_PARTS: usize = 64;
/// One in this many terms of each document is sampled to pick partition
/// boundaries.
const SAMPLE_EVERY: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FileHeader {
    magic: [u8; 8],
    version: u32,
    n_docs: u32,
    count: u32,
    _pad: u32,
}

/// One fst map, covering a contiguous term range.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PartRow {
    fst_off: u64,
    fst_len: u64,
    /// Start of this partition's postings in `base.post`.
    post_off: u64,
}

/// One materialized short prefix, sorted by key bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ShortRow {
    key: [u8; 2],
    key_len: u8,
    /// 1 for a dense column, 0 for a posting list.
    dense: u8,
    _pad: [u8; 4],
    off: u64,
    len: u64,
}

impl ShortRow {
    fn key(&self) -> &[u8] {
        &self.key[..self.key_len as usize]
    }
}

/// Calls `emit` with the term of every word start in `lowered` (normalized
/// text): the rest of its chunk cut to `TERM_CAP` bytes, and when the chunk
/// fits uncut and is followed by one space and another chunk, that space and
/// the next chunk's `pair_prefix`.
pub fn for_each_term<'t>(lowered: &'t str, mut emit: impl FnMut(&'t str)) {
    let mut chunk_start = 0;
    while chunk_start < lowered.len() {
        let chunk_end = lowered[chunk_start..]
            .find(' ')
            .map_or(lowered.len(), |i| chunk_start + i);
        let next = &lowered[(chunk_end + 1).min(lowered.len())..];
        let next = pair_prefix(&next[..next.find(' ').unwrap_or(next.len())]);
        let mut prev_alnum = false;
        for (i, ch) in lowered[chunk_start..chunk_end].char_indices() {
            if !prev_alnum {
                let start = chunk_start + i;
                let end = if chunk_end - start > TERM_CAP {
                    start + truncate_on_char_boundary(&lowered[start..chunk_end], TERM_CAP).len()
                } else if next.is_empty() {
                    chunk_end
                } else {
                    chunk_end + 1 + next.len()
                };
                emit(&lowered[start..end]);
            }
            prev_alnum = ch.is_alphanumeric();
        }
        chunk_start = chunk_end + 1;
    }
}

/// Whether `c` is an ideograph that gets inner terms.
pub fn is_ideograph(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
}

/// Calls `emit` with the chunk rest, cut to `TERM_CAP` bytes, at every
/// ideograph of `lowered` (normalized text) that follows an alphanumeric char
/// in its chunk. The stored inner term is a space followed by that rest.
pub fn for_each_inner_term<'t>(lowered: &'t str, mut emit: impl FnMut(&'t str)) {
    // Ideographs are three-byte sequences with these lead bytes.
    if !lowered.bytes().any(|b| (0xE4..=0xE9).contains(&b)) {
        return;
    }
    for chunk in lowered.split(' ') {
        let mut prev_alnum = false;
        for (i, ch) in chunk.char_indices() {
            if prev_alnum && is_ideograph(ch) {
                emit(truncate_on_char_boundary(&chunk[i..], TERM_CAP));
            }
            prev_alnum = ch.is_alphanumeric();
        }
    }
}

/// Multiply-rotate hash for short keys, much cheaper than SipHash when
/// counting the hundreds of millions of term occurrences of a full build.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.add(u64::from_le_bytes(*chunk));
        }
        let mut tail = [0u8; 8];
        tail[..rest.len()].copy_from_slice(rest);
        self.add(u64::from_le_bytes(tail) ^ rest.len() as u64);
    }

    fn write_u8(&mut self, byte: u8) {
        self.add(u64::from(byte));
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl FxHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

/// Distinct terms of one document, sorted, with capped term frequencies.
struct DocTerms {
    bytes: Vec<u8>,
    ends: Vec<u32>,
    tfs: Vec<u8>,
    short: Vec<(ShortKey, u8)>,
}

type ShortKey = ([u8; 2], u8);

impl DocTerms {
    fn new(lowered: &str) -> Self {
        let mut counts: HashMap<&str, u32, BuildHasherDefault<FxHasher>> = HashMap::default();
        for_each_term(lowered, |term| *counts.entry(term).or_default() += 1);
        let mut inner: HashMap<&str, u32, BuildHasherDefault<FxHasher>> = HashMap::default();
        for_each_inner_term(lowered, |rest| *inner.entry(rest).or_default() += 1);
        let mut terms: Vec<(Cow<str>, u32)> = counts
            .into_iter()
            .map(|(term, n)| (Cow::Borrowed(term), n))
            .chain(
                inner
                    .into_iter()
                    .map(|(rest, n)| (Cow::Owned(format!(" {rest}")), n)),
            )
            .collect();
        terms.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

        let mut doc = DocTerms {
            bytes: Vec::new(),
            ends: Vec::new(),
            tfs: Vec::new(),
            short: Vec::new(),
        };
        let mut short: [Option<(ShortKey, u8)>; SHORT_LEN] = [None; SHORT_LEN];
        for (term, n) in &terms {
            let (term, n) = (term.as_ref(), *n);
            let tf = n.min(u32::from(TF_CAP)) as u8;
            doc.bytes.extend_from_slice(term.as_bytes());
            doc.ends.push(doc.bytes.len() as u32);
            doc.tfs.push(tf);

            // Terms sharing a short prefix are adjacent in sorted order.
            for (len, slot) in (1..=SHORT_LEN).zip(short.iter_mut()) {
                if term.len() < len || !term.is_char_boundary(len) || term[..len].contains(' ') {
                    continue;
                }
                let mut key = [0; 2];
                key[..len].copy_from_slice(&term.as_bytes()[..len]);
                let key = (key, len as u8);
                match slot {
                    Some((current, sum)) if *current == key => *sum = (*sum + tf).min(TF_CAP),
                    _ => {
                        if let Some(done) = slot.replace((key, tf)) {
                            doc.short.push(done);
                        }
                    }
                }
            }
        }
        doc.short.extend(short.into_iter().flatten());
        doc
    }

    fn len(&self) -> usize {
        self.ends.len()
    }

    fn term(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.bytes[start..self.ends[i] as usize]
    }

    /// Index of the first term not less than `bound`.
    fn lower_bound(&self, bound: &[u8]) -> usize {
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.term(mid) < bound {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Sizes and counts of a built index.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Default, Clone)]
pub struct BuildStats {
    pub docs: usize,
    pub terms: u64,
    pub postings: u64,
    pub inline_postings: u64,
    pub short_prefixes: usize,
    pub parts: usize,
    pub fst_bytes: usize,
    pub post_bytes: usize,
    pub short_bytes: usize,
}

/// Index file contents, ready to be written into a generation directory.
pub struct Built {
    pub fst: Vec<u8>,
    pub post: Vec<u8>,
    pub short: Vec<u8>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub stats: BuildStats,
}

impl Built {
    /// Writes the `field` files of index segment `id` into `dir`.
    pub fn write(&self, dir: &Path, id: u32, field: Field) -> io::Result<()> {
        let [fst, post, short] = segment_files(id, field);
        write_synced(&dir.join(fst), &self.fst)?;
        write_synced(&dir.join(post), &self.post)?;
        write_synced(&dir.join(short), &self.short)
    }

    pub fn disk_bytes(&self) -> u64 {
        (self.fst.len() + self.post.len() + self.short.len()) as u64
    }
}

struct Part {
    fst: Vec<u8>,
    post: Vec<u8>,
    terms: u64,
    postings: u64,
    inline_postings: u64,
}

/// Builds the index for `n_docs` documents. `lowered(doc)` returns the
/// normalized body text (`normalize_for_search(full_text)`) of doc `doc`.
pub fn build<'a, F>(n_docs: usize, lowered: F) -> Built
where
    F: Fn(u32) -> Cow<'a, str> + Sync,
{
    assert!(n_docs <= u32::MAX as usize);
    let docs: Vec<DocTerms> = (0..n_docs as u32)
        .into_par_iter()
        .map(|doc| DocTerms::new(&lowered(doc)))
        .collect();
    let bounds = partition_bounds(&docs);
    let parts: Vec<Part> = (0..bounds.len() + 1)
        .into_par_iter()
        .map(|part| {
            let lo = part.checked_sub(1).map(|i| bounds[i].as_slice());
            let hi = bounds.get(part).map(Vec::as_slice);
            build_part(&docs, lo, hi)
        })
        .collect();
    let short = build_short(&docs, n_docs);
    drop(docs);
    assemble(parts, short, n_docs)
}

/// Lays out built partitions and the short file as index files.
fn assemble(parts: Vec<Part>, short: (Vec<u8>, usize), n_docs: usize) -> Built {
    let header = |magic, count: usize| FileHeader {
        magic,
        version: VERSION,
        n_docs: n_docs as u32,
        count: count as u32,
        _pad: 0,
    };
    let mut post = bytemuck::bytes_of(&header(POST_MAGIC, 0)).to_vec();
    let mut rows = Vec::with_capacity(parts.len());
    let mut fst_off =
        std::mem::size_of::<FileHeader>() + parts.len() * std::mem::size_of::<PartRow>();
    for part in &parts {
        rows.push(PartRow {
            fst_off: fst_off as u64,
            fst_len: part.fst.len() as u64,
            post_off: post.len() as u64,
        });
        fst_off += part.fst.len();
        post.extend_from_slice(&part.post);
    }
    let mut fst = bytemuck::bytes_of(&header(FST_MAGIC, parts.len())).to_vec();
    fst.extend_from_slice(bytemuck::cast_slice(&rows));
    for part in &parts {
        fst.extend_from_slice(&part.fst);
    }

    let stats = BuildStats {
        docs: n_docs,
        terms: parts.iter().map(|part| part.terms).sum(),
        postings: parts.iter().map(|part| part.postings).sum(),
        inline_postings: parts.iter().map(|part| part.inline_postings).sum(),
        short_prefixes: short.1,
        parts: parts.len(),
        fst_bytes: fst.len(),
        post_bytes: post.len(),
        short_bytes: short.0.len(),
    };
    Built {
        fst,
        post,
        short: short.0,
        stats,
    }
}

/// Sorted, distinct terms splitting the term space into partitions of
/// similar posting counts.
fn partition_bounds(docs: &[DocTerms]) -> Vec<Vec<u8>> {
    let mut sample: Vec<&[u8]> = docs
        .iter()
        .flat_map(|doc| (0..doc.len()).step_by(SAMPLE_EVERY).map(|i| doc.term(i)))
        .collect();
    sample.par_sort_unstable();
    let parts = MAX_PARTS.min(sample.len() / 16).max(1);
    let mut bounds: Vec<Vec<u8>> = (1..parts)
        .map(|i| sample[i * sample.len() / parts].to_vec())
        .collect();
    bounds.dedup();
    bounds
}

/// Builds the fst and postings for terms in `[lo, hi)`.
fn build_part(docs: &[DocTerms], lo: Option<&[u8]>, hi: Option<&[u8]>) -> Part {
    // (first 8 term bytes, doc, term index): most comparisons are decided by
    // the inline key without touching the term bytes.
    let mut entries: Vec<(u64, u32, u32)> = Vec::new();
    for (doc_id, doc) in docs.iter().enumerate() {
        let start = lo.map_or(0, |lo| doc.lower_bound(lo));
        let end = hi.map_or(doc.len(), |hi| doc.lower_bound(hi));
        entries.extend((start..end).map(|i| {
            let mut head = [0u8; 8];
            let term = doc.term(i);
            let n = term.len().min(8);
            head[..n].copy_from_slice(&term[..n]);
            (u64::from_be_bytes(head), doc_id as u32, i as u32)
        }));
    }
    let term = |&(_, doc, i): &(u64, u32, u32)| docs[doc as usize].term(i as usize);
    // Postings of each term end up in doc order.
    entries.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| term(a).cmp(term(b)))
            .then(a.1.cmp(&b.1))
    });

    let mut builder = fst::MapBuilder::memory();
    let mut post = Vec::new();
    let mut part = Part {
        fst: Vec::new(),
        post: Vec::new(),
        terms: 0,
        postings: entries.len() as u64,
        inline_postings: 0,
    };
    let mut i = 0;
    while i < entries.len() {
        let key = term(&entries[i]);
        let mut j = i + 1;
        while j < entries.len() && term(&entries[j]) == key {
            j += 1;
        }
        let tf = |&(_, doc, t): &(u64, u32, u32)| docs[doc as usize].tfs[t as usize];
        let value = if j - i == 1 {
            part.inline_postings += 1;
            INLINE | (u64::from(tf(&entries[i])) << 32) | u64::from(entries[i].1)
        } else {
            let off = post.len() as u64;
            write_varint(&mut post, (j - i) as u64);
            let mut prev = 0;
            for entry in &entries[i..j] {
                write_posting(&mut post, entry.1 - prev, tf(entry));
                prev = entry.1;
            }
            off
        };
        builder
            .insert(key, value)
            .expect("terms are inserted in sorted order");
        part.terms += 1;
        i = j;
    }
    part.fst = builder.into_inner().expect("in-memory fst write");
    part.post = post;
    part
}

/// Serializes `base.short`; returns its bytes and prefix count.
fn build_short(docs: &[DocTerms], n_docs: usize) -> (Vec<u8>, usize) {
    let mut columns: BTreeMap<&[u8], Vec<(u32, u8)>> = BTreeMap::new();
    for (doc_id, doc) in docs.iter().enumerate() {
        for (key, tf) in &doc.short {
            columns
                .entry(&key.0[..key.1 as usize])
                .or_default()
                .push((doc_id as u32, *tf));
        }
    }
    write_short(&columns, n_docs)
}

/// Serializes short columns whose postings are in doc order.
fn write_short<K: AsRef<[u8]>>(
    columns: &BTreeMap<K, Vec<(u32, u8)>>,
    n_docs: usize,
) -> (Vec<u8>, usize) {
    let header = FileHeader {
        magic: SHORT_MAGIC,
        version: VERSION,
        n_docs: n_docs as u32,
        count: columns.len() as u32,
        _pad: 0,
    };
    let data_off =
        std::mem::size_of::<FileHeader>() + columns.len() * std::mem::size_of::<ShortRow>();
    let mut rows = Vec::with_capacity(columns.len());
    let mut data = Vec::new();
    for (key, postings) in columns {
        let key = key.as_ref();
        let mut sparse = Vec::new();
        write_varint(&mut sparse, postings.len() as u64);
        let mut prev = 0;
        for &(doc, tf) in postings {
            write_posting(&mut sparse, doc - prev, tf);
            prev = doc;
        }
        let dense = sparse.len() > n_docs;
        let off = data_off + data.len();
        if dense {
            let mut column = vec![0; n_docs];
            for &(doc, tf) in postings {
                column[doc as usize] = tf;
            }
            data.extend_from_slice(&column);
        } else {
            data.extend_from_slice(&sparse);
        }
        let mut row_key = [0; 2];
        row_key[..key.len()].copy_from_slice(key);
        rows.push(ShortRow {
            key: row_key,
            key_len: key.len() as u8,
            dense: u8::from(dense),
            _pad: [0; 4],
            off: off as u64,
            len: (data_off + data.len() - off) as u64,
        });
    }
    let mut out = bytemuck::bytes_of(&header).to_vec();
    out.extend_from_slice(bytemuck::cast_slice(&rows));
    out.extend_from_slice(&data);
    (out, columns.len())
}

/// One posting: `delta << 4 | tf` as a varint (`TF_CAP` fits in 4 bits).
fn write_posting(out: &mut Vec<u8>, delta: u32, tf: u8) {
    write_varint(out, (u64::from(delta) << 4) | u64::from(tf));
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_varint(data: &[u8], pos: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *data.get(*pos)?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

/// Calls `f` with every `(doc, tf)` of the posting list at the start of
/// `data`. Stops at the first malformed entry or when `f` returns `None`.
fn for_each_posting(data: &[u8], mut f: impl FnMut(u64, u8) -> Option<()>) -> Option<()> {
    let mut pos = 0;
    let count = read_varint(data, &mut pos)?;
    let mut doc = 0u64;
    for _ in 0..count {
        let posting = read_varint(data, &mut pos)?;
        doc = doc.checked_add(posting >> 4)?;
        f(doc, (posting & 0xf) as u8)?;
    }
    Some(())
}

/// Adds `tf` to the slot of local doc `doc` in `acc`, capping at `TF_CAP`.
/// `remap` maps local docs to slots, `DEAD` docs are skipped. `None` when
/// `doc` is out of range.
#[inline]
fn add_tf(acc: &mut [u8], remap: Option<&[u32]>, doc: u64, tf: u8) -> Option<()> {
    let doc = usize::try_from(doc).ok()?;
    let slot = match remap {
        Some(remap) => match *remap.get(doc)? {
            DEAD => return Some(()),
            slot => slot as usize,
        },
        None => doc,
    };
    let slot = acc.get_mut(slot)?;
    *slot = slot.saturating_add(tf).min(TF_CAP);
    Some(())
}

/// `(doc, tf)` pairs in doc order.
type Postings = Vec<(u32, u8)>;

/// A partition's fst map and the postings it points into.
type PartMap<'a> = (fst::Map<&'a [u8]>, &'a [u8]);

/// A validated, memory-mapped index.
pub struct LexIndex {
    n_docs: u32,
    fst: memmap2::Mmap,
    parts: Vec<(Range<usize>, usize)>,
    post: memmap2::Mmap,
    short: memmap2::Mmap,
    short_rows: Range<usize>,
}

fn map_file(path: &Path) -> Option<memmap2::Mmap> {
    let file = File::open(path).ok()?;
    // SAFETY: published index files are never written again and are only
    // removed by unlinking, so the mapped bytes cannot change underneath.
    unsafe { memmap2::Mmap::map(&file) }.ok()
}

fn read_header(map: &[u8], magic: [u8; 8], n_docs: u32) -> Option<FileHeader> {
    let header: FileHeader =
        bytemuck::try_pod_read_unaligned(map.get(..std::mem::size_of::<FileHeader>())?).ok()?;
    (header.magic == magic && header.version == VERSION && header.n_docs == n_docs)
        .then_some(header)
}

/// Range of `count` records of `T` right after the header.
fn table<T>(len: usize, count: u32) -> Option<Range<usize>> {
    let start = std::mem::size_of::<FileHeader>();
    let end = start.checked_add((count as usize).checked_mul(std::mem::size_of::<T>())?)?;
    (end <= len).then_some(start..end)
}

impl LexIndex {
    /// Opens the `field` files of index segment `id` in `dir`, built for
    /// `n_docs` documents. Returns `None` when any file is missing or fails
    /// validation.
    pub fn open(dir: &Path, id: u32, field: Field, n_docs: u32) -> Option<Self> {
        let [fst, post, short] = segment_files(id, field);
        let fst = map_file(&dir.join(fst))?;
        let post = map_file(&dir.join(post))?;
        let short = map_file(&dir.join(short))?;
        read_header(&post, POST_MAGIC, n_docs)?;

        let header = read_header(&fst, FST_MAGIC, n_docs)?;
        let part_table = table::<PartRow>(fst.len(), header.count)?;
        let part_rows: &[PartRow] = bytemuck::try_cast_slice(&fst[part_table]).ok()?;
        let mut parts = Vec::with_capacity(part_rows.len());
        for row in part_rows {
            let start = usize::try_from(row.fst_off).ok()?;
            let end = start.checked_add(usize::try_from(row.fst_len).ok()?)?;
            let post_off = usize::try_from(row.post_off).ok()?;
            if end > fst.len() || post_off > post.len() {
                return None;
            }
            fst::Map::new(&fst[start..end]).ok()?;
            parts.push((start..end, post_off));
        }

        let header = read_header(&short, SHORT_MAGIC, n_docs)?;
        let short_rows = table::<ShortRow>(short.len(), header.count)?;
        let rows: &[ShortRow] = bytemuck::try_cast_slice(&short[short_rows.clone()]).ok()?;
        let rows_valid = rows.iter().all(|row| {
            let end = row.off.checked_add(row.len);
            (1..=SHORT_LEN as u8).contains(&row.key_len)
                && row.dense <= 1
                && (row.dense == 0 || row.len == u64::from(n_docs))
                && end.is_some_and(|end| end <= short.len() as u64)
        }) && rows.windows(2).all(|pair| pair[0].key() < pair[1].key());
        if !rows_valid {
            return None;
        }

        Some(Self {
            n_docs,
            fst,
            parts,
            post,
            short,
            short_rows,
        })
    }

    pub fn n_docs(&self) -> u32 {
        self.n_docs
    }

    fn short_rows(&self) -> &[ShortRow] {
        bytemuck::cast_slice(&self.short[self.short_rows.clone()])
    }

    /// Per-document count of `word` (normalized, starting with an ideograph)
    /// occurring right after an alphanumeric char, cut to `TERM_CAP` bytes
    /// like `prefix_tf`. With `prefix_tf`, it bounds where `word` occurs at
    /// all.
    #[cfg(test)]
    pub fn inner_tf(&self, word: &str) -> Vec<u8> {
        let mut acc = vec![0; self.n_docs as usize];
        self.inner_tf_into(word, &mut acc, None);
        acc
    }

    /// Capped count of word-start occurrences of `word` (a normalized word,
    /// or an adjacent pair) in each document. Words longer than `TERM_CAP`
    /// are counted by their `TERM_CAP`-byte prefix, which can only
    /// overcount.
    #[cfg(test)]
    pub fn prefix_tf(&self, word: &str) -> Vec<u8> {
        let mut acc = vec![0; self.n_docs as usize];
        self.prefix_tf_into(word, &mut acc, None);
        acc
    }

    /// `inner_tf` added into `acc`; see `prefix_tf_into`.
    pub fn inner_tf_into(&self, word: &str, acc: &mut [u8], remap: Option<&[u32]>) {
        if word.starts_with(is_ideograph) {
            let term = format!(" {}", truncate_on_char_boundary(word, TERM_CAP));
            self.prefix_tf_into(&term, acc, remap);
        }
    }

    /// `prefix_tf` added into `acc`, capping at `TF_CAP`. With `remap`, the
    /// tf of doc `d` goes to `acc[remap[d]]` and `DEAD` docs are skipped;
    /// otherwise `acc` is indexed by doc.
    pub fn prefix_tf_into(&self, word: &str, acc: &mut [u8], remap: Option<&[u32]>) {
        if word.is_empty() {
            return;
        }
        if word.len() <= SHORT_LEN {
            let rows = self.short_rows();
            if let Ok(i) = rows.binary_search_by(|row| row.key().cmp(word.as_bytes())) {
                let row = rows[i];
                let data = &self.short[row.off as usize..(row.off + row.len) as usize];
                if row.dense == 1 {
                    if remap.is_none() && acc.len() == data.len() {
                        for (slot, &tf) in acc.iter_mut().zip(data) {
                            *slot = slot.saturating_add(tf).min(TF_CAP);
                        }
                    } else {
                        for (doc, &tf) in data.iter().enumerate() {
                            if tf > 0 && add_tf(acc, remap, doc as u64, tf).is_none() {
                                break;
                            }
                        }
                    }
                } else {
                    for_each_posting(data, |doc, tf| add_tf(acc, remap, doc, tf));
                }
            }
            return;
        }

        let key = if word.contains(' ') {
            word
        } else {
            truncate_on_char_boundary(word, TERM_CAP)
        }
        .as_bytes();
        for (range, post_off) in &self.parts {
            let Ok(map) = fst::Map::new(&self.fst[range.clone()]) else {
                continue;
            };
            let mut stream = map.range().ge(key).into_stream();
            while let Some((term, value)) = stream.next() {
                if !term.starts_with(key) {
                    break;
                }
                if value & INLINE != 0 {
                    add_tf(acc, remap, value & 0xffff_ffff, (value >> 32) as u8);
                } else if let Some(data) = usize::try_from(value)
                    .ok()
                    .and_then(|off| off.checked_add(*post_off))
                    .and_then(|start| self.post.get(start..))
                {
                    for_each_posting(data, |doc, tf| add_tf(acc, remap, doc, tf));
                }
            }
        }
    }

    /// Fst maps of the partitions, in term order, with their postings.
    fn part_maps(&self) -> Vec<PartMap<'_>> {
        self.parts
            .iter()
            .filter_map(|(range, post_off)| {
                let map = fst::Map::new(&self.fst[range.clone()]).ok()?;
                Some((map, self.post.get(*post_off..)?))
            })
            .collect()
    }

    /// Calls `f` with every `(doc, tf)` stored under fst value `value` of a
    /// partition whose postings start at `post`.
    fn for_each_value_posting(value: u64, post: &[u8], mut f: impl FnMut(u64, u8)) {
        if value & INLINE != 0 {
            f(value & 0xffff_ffff, (value >> 32) as u8);
        } else if let Some(data) = usize::try_from(value).ok().and_then(|off| post.get(off..)) {
            for_each_posting(data, |doc, tf| {
                f(doc, tf);
                Some(())
            });
        }
    }

    /// Every short column: key and `(doc, tf)` with nonzero tf, doc order.
    fn short_columns(&self) -> Vec<(&[u8], Postings)> {
        self.short_rows()
            .iter()
            .map(|row| {
                let data = &self.short[row.off as usize..(row.off + row.len) as usize];
                let mut postings = Vec::new();
                if row.dense == 1 {
                    postings.extend(
                        data.iter()
                            .enumerate()
                            .filter(|(_, tf)| **tf > 0)
                            .map(|(doc, &tf)| (doc as u32, tf)),
                    );
                } else {
                    for_each_posting(data, |doc, tf| {
                        postings.push((u32::try_from(doc).ok()?, tf));
                        Some(())
                    });
                }
                (row.key(), postings)
            })
            .collect()
    }
}

/// Merges index segments into one index over `n_docs` documents. Each
/// source comes with its `remap`: the new doc id of each of its docs, or
/// `DEAD` to drop it. Every new doc must come from exactly one source. The
/// result equals `build` over the same documents in the new order, without
/// reading their text.
pub fn merge(sources: &[(&LexIndex, &[u32])], n_docs: usize) -> Built {
    assert!(n_docs <= u32::MAX as usize);
    let maps: Vec<Vec<PartMap>> = sources.iter().map(|(index, _)| index.part_maps()).collect();
    // Partition like the largest source, whose parts already split its terms
    // into similar posting counts.
    let bounds: Vec<Vec<u8>> = maps
        .iter()
        .max_by_key(|parts| parts.len())
        .map(|parts| {
            parts
                .iter()
                .skip(1)
                .filter_map(|(map, _)| map.stream().next().map(|(term, _)| term.to_vec()))
                .collect()
        })
        .unwrap_or_default();
    let parts: Vec<Part> = (0..bounds.len() + 1)
        .into_par_iter()
        .map(|part| {
            let lo = part.checked_sub(1).map(|i| bounds[i].as_slice());
            let hi = bounds.get(part).map(Vec::as_slice);
            merge_part(sources, &maps, lo, hi)
        })
        .collect();

    let mut columns: BTreeMap<Vec<u8>, Vec<(u32, u8)>> = BTreeMap::new();
    for (index, remap) in sources {
        for (key, postings) in index.short_columns() {
            let column = columns.entry(key.to_vec()).or_default();
            column.extend(postings.into_iter().filter_map(|(doc, tf)| {
                let doc = *remap.get(doc as usize)?;
                (doc != DEAD).then_some((doc, tf))
            }));
        }
    }
    columns.retain(|_, postings| !postings.is_empty());
    for postings in columns.values_mut() {
        postings.sort_unstable_by_key(|&(doc, _)| doc);
    }
    let short = write_short(&columns, n_docs);
    assemble(parts, short, n_docs)
}

/// Merges the terms in `[lo, hi)` of every source into one partition.
fn merge_part(
    sources: &[(&LexIndex, &[u32])],
    maps: &[Vec<PartMap>],
    lo: Option<&[u8]>,
    hi: Option<&[u8]>,
) -> Part {
    let mut union = fst::map::OpBuilder::new();
    // Union index -> (source, postings of the partition).
    let mut origins = Vec::new();
    for (source, parts) in maps.iter().enumerate() {
        for (map, post) in parts {
            let range = map.range();
            let range = match lo {
                Some(lo) => range.ge(lo),
                None => range,
            };
            let range = match hi {
                Some(hi) => range.lt(hi),
                None => range,
            };
            union.push(range);
            origins.push((source, *post));
        }
    }
    let mut stream = union.union();

    let mut builder = fst::MapBuilder::memory();
    let mut post = Vec::new();
    let mut part = Part {
        fst: Vec::new(),
        post: Vec::new(),
        terms: 0,
        postings: 0,
        inline_postings: 0,
    };
    let mut postings: Vec<(u32, u8)> = Vec::new();
    while let Some((term, values)) = stream.next() {
        postings.clear();
        for value in values {
            let (source, data) = origins[value.index];
            let remap = sources[source].1;
            LexIndex::for_each_value_posting(value.value, data, |doc, tf| {
                if let Some(&doc) = usize::try_from(doc).ok().and_then(|doc| remap.get(doc))
                    && doc != DEAD
                {
                    postings.push((doc, tf));
                }
            });
        }
        if postings.is_empty() {
            continue;
        }
        postings.sort_unstable_by_key(|&(doc, _)| doc);
        part.postings += postings.len() as u64;
        let value = if let [(doc, tf)] = postings[..] {
            part.inline_postings += 1;
            INLINE | (u64::from(tf) << 32) | u64::from(doc)
        } else {
            let off = post.len() as u64;
            write_varint(&mut post, postings.len() as u64);
            let mut prev = 0;
            for &(doc, tf) in &postings {
                write_posting(&mut post, doc - prev, tf);
                prev = doc;
            }
            off
        };
        builder
            .insert(term, value)
            .expect("union streams terms in sorted order");
        part.terms += 1;
    }
    part.fst = builder.into_inner().expect("in-memory fst write");
    part.post = post;
    part
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_match::{is_word_start, normalize_for_search};

    fn index_of(texts: &[&str]) -> (tempfile::TempDir, LexIndex) {
        let lowered: Vec<String> = texts.iter().map(|t| normalize_for_search(t)).collect();
        let built = build(lowered.len(), |doc| Cow::Borrowed(&lowered[doc as usize]));
        let dir = tempfile::tempdir().unwrap();
        built.write(dir.path(), 0, Field::Body).unwrap();
        let index = LexIndex::open(dir.path(), 0, Field::Body, lowered.len() as u32).unwrap();
        (dir, index)
    }

    /// Word-start positions whose text starts with `word` (cut to
    /// `TERM_CAP` bytes unless it is a pair).
    fn naive_tf(lowered: &str, word: &str) -> u8 {
        let key = if word.contains(' ') {
            word
        } else {
            truncate_on_char_boundary(word, TERM_CAP)
        };
        let count = lowered
            .char_indices()
            .filter(|&(i, ch)| {
                ch != ' '
                    && lowered[..i]
                        .chars()
                        .next_back()
                        .is_none_or(|c| !c.is_alphanumeric())
                    && lowered[i..].starts_with(key)
            })
            .count();
        count.min(TF_CAP as usize) as u8
    }

    #[test]
    fn terms_are_word_start_suffixes_of_chunks() {
        let mut terms = Vec::new();
        for_each_term("foo.bar(baz) ((x  é ab x", |term| {
            terms.push(term.to_string())
        });
        assert_eq!(
            terms,
            [
                "foo.bar(baz) ((x",
                "bar(baz) ((x",
                "baz) ((x",
                "((x",
                "(x",
                "x",
                "é ab",
                "ab x",
                "x"
            ]
        );
        let mut terms = Vec::new();
        for_each_term("a 测试", |term| terms.push(term.to_string()));
        assert_eq!(terms, ["a 测", "测试"]);
        // A next chunk starting with a four-byte char contributes that char.
        let mut terms = Vec::new();
        for_each_term("fiesta 🎉 now", |term| terms.push(term.to_string()));
        assert_eq!(terms, ["fiesta 🎉", "🎉 now", "now"]);
        let mut inner = Vec::new();
        for_each_inner_term("这是测试 x会话 测 a测", |rest| {
            inner.push(rest.to_string())
        });
        assert_eq!(inner, ["是测试", "测试", "试", "会话", "话", "测"]);
        let long = "a".repeat(30) + "é";
        let mut terms = Vec::new();
        for_each_term(&long, |term| terms.push(term.len()));
        assert_eq!(terms, [TERM_CAP]);
        let mut terms = Vec::new();
        for_each_term(&format!("{}é", "a".repeat(23)), |term| {
            terms.push(term.len())
        });
        assert_eq!(terms, [23]);
    }

    #[test]
    fn prefix_tf_matches_naive_counts() {
        let texts = [
            "error handling error handled errors",
            "Hardened_Runtime config/main-worktree a.a.a",
            "",
            "e e e e e e e e e e e e e",
            "零 测试 cache 会话 café cafe 这是测试测试 x会话",
            &"x".repeat(40),
            "the then there theme the.foo",
        ];
        let lowered: Vec<String> = texts.iter().map(|t| normalize_for_search(t)).collect();
        let (_dir, index) = index_of(&texts);
        for word in [
            "e",
            "er",
            "err",
            "error",
            "errors",
            "handl",
            "a",
            "a.a",
            "x",
            "xx",
            &"x".repeat(24),
            &"x".repeat(30),
            "测",
            "测试",
            "caf",
            "café",
            "c",
            "é",
            "the",
            "the.",
            "zzz",
            "runtime",
            "main",
            "foo",
        ] {
            let tf = index.prefix_tf(word);
            for (doc, text) in lowered.iter().enumerate() {
                assert_eq!(tf[doc], naive_tf(text, word), "word {word:?} doc {doc}");
            }
        }

        for word in [
            "测",
            "测试",
            "试",
            "会话",
            "话",
            "是测试测试",
            "零",
            "测试x",
        ] {
            let inner = index.inner_tf(word);
            let tf = index.prefix_tf(word);
            for (doc, text) in lowered.iter().enumerate() {
                let count = text
                    .match_indices(word)
                    .filter(|&(i, _)| !is_word_start(text, i))
                    .count()
                    .min(TF_CAP as usize) as u8;
                assert_eq!(inner[doc], count, "word {word:?} doc {doc}");
                assert_eq!(inner[doc] > 0 || tf[doc] > 0, text.contains(word));
            }
        }
    }

    #[test]
    fn many_documents_split_into_partitions() {
        let texts: Vec<String> = (0..400)
            .map(|i| {
                (0..60)
                    .map(|j| format!("w{}x{} common", (i * 7 + j * 13) % 997, j % 11))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let lowered: Vec<String> = refs.iter().map(|t| normalize_for_search(t)).collect();
        let built = build(lowered.len(), |doc| Cow::Borrowed(&lowered[doc as usize]));
        assert!(built.stats.parts > 1);
        let (_dir, index) = index_of(&refs);
        for word in [
            "w1",
            "w12",
            "w123x",
            "common",
            "co",
            "c",
            "w9",
            "w99",
            "common w",
            "common w1",
            "w1x1 com",
        ] {
            let tf = index.prefix_tf(word);
            for (doc, text) in lowered.iter().enumerate() {
                assert_eq!(tf[doc], naive_tf(text, word), "word {word:?} doc {doc}");
            }
        }
    }

    #[test]
    fn open_rejects_corrupt_or_mismatched_files() {
        let (dir, _index) = index_of(&["alpha beta", "gamma"]);
        assert!(
            LexIndex::open(dir.path(), 0, Field::Body, 3).is_none(),
            "doc count"
        );
        for file in [FST_FILE, POST_FILE, SHORT_FILE] {
            let path = dir.path().join(file);
            let original = std::fs::read(&path).unwrap();
            let mut bytes = original.clone();
            bytes[0] = b'X';
            std::fs::write(&path, &bytes).unwrap();
            assert!(
                LexIndex::open(dir.path(), 0, Field::Body, 2).is_none(),
                "{file} magic"
            );
            std::fs::write(&path, &original[..original.len().min(20)]).unwrap();
            assert!(
                LexIndex::open(dir.path(), 0, Field::Body, 2).is_none(),
                "{file} truncated"
            );
            std::fs::write(&path, &original).unwrap();
        }
        let short = dir.path().join(SHORT_FILE);
        let mut bytes = std::fs::read(&short).unwrap();
        let row = std::mem::size_of::<FileHeader>();
        bytes[row + 16..row + 24].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&short, &bytes).unwrap();
        assert!(
            LexIndex::open(dir.path(), 0, Field::Body, 2).is_none(),
            "short row bounds"
        );
    }

    fn words_of(lowered: &[String]) -> Vec<String> {
        let mut words: Vec<String> = lowered
            .iter()
            .flat_map(|text| text.split(' ').map(str::to_owned))
            .filter(|word| !word.is_empty())
            .flat_map(|word| {
                let short: String = word.chars().take(2).collect();
                let one: String = word.chars().take(1).collect();
                [word, short, one]
            })
            .collect();
        words.extend(["测试", "试", "alpha be", "zzz"].map(str::to_owned));
        words.sort();
        words.dedup();
        words
    }

    #[test]
    fn merge_equals_build_over_the_new_order() {
        let texts: Vec<String> = (0..300)
            .map(|i| match i % 4 {
                0 => format!("alpha beta {i} common word{}", i % 7),
                1 => format!("测试{i} abc测试 gamma{} common", i % 5),
                2 => "a b c d e f alpha alpha alpha".repeat(i % 3 + 1),
                _ => format!("x{i} y{} unique{i}", i % 11),
            })
            .map(|text| normalize_for_search(&text))
            .collect();
        let (first, second) = texts.split_at(200);
        let built_first = build(first.len(), |doc| Cow::Borrowed(&first[doc as usize]));
        let built_second = build(second.len(), |doc| Cow::Borrowed(&second[doc as usize]));
        let dir = tempfile::tempdir().unwrap();
        built_first.write(dir.path(), 0, Field::Body).unwrap();
        built_second.write(dir.path(), 3, Field::Body).unwrap();
        let a = LexIndex::open(dir.path(), 0, Field::Body, first.len() as u32).unwrap();
        let b = LexIndex::open(dir.path(), 3, Field::Body, second.len() as u32).unwrap();

        // Drop every third doc of the first source and interleave the rest
        // in reverse order.
        let mut new_order: Vec<&String> = Vec::new();
        let mut remap_a = vec![DEAD; first.len()];
        let mut remap_b = vec![DEAD; second.len()];
        let mut next_b = (0..second.len()).rev();
        for (doc, text) in first.iter().enumerate().rev() {
            if doc % 3 != 0 {
                remap_a[doc] = new_order.len() as u32;
                new_order.push(text);
            }
            if doc % 2 == 0
                && let Some(other) = next_b.next()
            {
                remap_b[other] = new_order.len() as u32;
                new_order.push(&second[other]);
            }
        }
        for other in next_b {
            remap_b[other] = new_order.len() as u32;
            new_order.push(&second[other]);
        }

        let merged = merge(&[(&a, &remap_a), (&b, &remap_b)], new_order.len());
        merged.write(dir.path(), 7, Field::Body).unwrap();
        let merged = LexIndex::open(dir.path(), 7, Field::Body, new_order.len() as u32).unwrap();
        let rebuilt = build(new_order.len(), |doc| {
            Cow::Borrowed(new_order[doc as usize])
        });
        rebuilt.write(dir.path(), 8, Field::Body).unwrap();
        let rebuilt = LexIndex::open(dir.path(), 8, Field::Body, new_order.len() as u32).unwrap();

        for word in words_of(&texts) {
            assert_eq!(
                merged.prefix_tf(&word),
                rebuilt.prefix_tf(&word),
                "{word:?}"
            );
            assert_eq!(merged.inner_tf(&word), rebuilt.inner_tf(&word), "{word:?}");
            let mut acc = vec![0; new_order.len()];
            a.prefix_tf_into(&word, &mut acc, Some(&remap_a));
            b.prefix_tf_into(&word, &mut acc, Some(&remap_b));
            assert_eq!(acc, rebuilt.prefix_tf(&word), "remapped {word:?}");
        }
    }
}

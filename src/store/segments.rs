//! The index segments of one generation, queried as one index over its
//! cards.
//!
//! Every card's document lives in exactly one segment (`IndexLoc`), so a
//! card's term frequencies are those of its segment and the merged result
//! equals an index built over all cards at once. Segment docs no card points
//! at are dead and skipped.

use super::card::{IndexLoc, IndexSegRow};
use super::lexindex::{DEAD, Field, LexIndex};
use std::path::Path;

pub struct IndexSet {
    n_docs: u32,
    segments: Vec<Segment>,
}

struct Segment {
    index: LexIndex,
    /// Card row of each segment doc, or `DEAD`; `None` when doc `d` is card
    /// `d` for every card.
    remap: Option<Vec<u32>>,
}

impl IndexSet {
    /// One segment whose docs are the cards, in order.
    #[cfg(test)]
    pub fn single(index: LexIndex) -> Self {
        Self {
            n_docs: index.n_docs(),
            segments: vec![Segment { index, remap: None }],
        }
    }

    /// Opens the `field` files of the segments `segs` in `dir` for cards
    /// located at `locs`. `None` when a segment is missing or invalid, or a
    /// card points outside the segments.
    pub fn open(dir: &Path, field: Field, segs: &[IndexSegRow], locs: &[IndexLoc]) -> Option<Self> {
        let n_docs = u32::try_from(locs.len()).ok()?;
        let mut remaps: Vec<Vec<u32>> = segs
            .iter()
            .map(|seg| vec![DEAD; seg.n_docs as usize])
            .collect();
        for (row, loc) in locs.iter().enumerate() {
            let at = segs.binary_search_by_key(&loc.seg, |seg| seg.id).ok()?;
            let slot = remaps[at].get_mut(loc.doc as usize)?;
            if *slot != DEAD {
                return None;
            }
            *slot = row as u32;
        }
        let segments = segs
            .iter()
            .zip(remaps)
            .map(|(seg, remap)| {
                let index = LexIndex::open(dir, seg.id, field, seg.n_docs)?;
                let identity = segs.len() == 1
                    && remap.len() == n_docs as usize
                    && remap
                        .iter()
                        .enumerate()
                        .all(|(doc, &row)| row == doc as u32);
                Some(Segment {
                    index,
                    remap: (!identity).then_some(remap),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self { n_docs, segments })
    }

    /// Number of cards; tf vectors are indexed by card row.
    pub fn n_docs(&self) -> u32 {
        self.n_docs
    }

    /// `LexIndex::prefix_tf` per card.
    pub fn prefix_tf(&self, word: &str) -> Vec<u8> {
        let mut acc = vec![0; self.n_docs as usize];
        for segment in &self.segments {
            segment
                .index
                .prefix_tf_into(word, &mut acc, segment.remap.as_deref());
        }
        acc
    }

    /// `LexIndex::inner_tf` per card.
    pub fn inner_tf(&self, word: &str) -> Vec<u8> {
        let mut acc = vec![0; self.n_docs as usize];
        for segment in &self.segments {
            segment
                .index
                .inner_tf_into(word, &mut acc, segment.remap.as_deref());
        }
        acc
    }

    /// The segments with each one's card remap, for merging.
    pub fn sources(&self) -> Vec<(&LexIndex, Vec<u32>)> {
        self.segments
            .iter()
            .map(|segment| {
                let remap = segment
                    .remap
                    .clone()
                    .unwrap_or_else(|| (0..segment.index.n_docs()).collect());
                (&segment.index, remap)
            })
            .collect()
    }
}

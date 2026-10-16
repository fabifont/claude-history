//! Lexical search over a generation's suffix index.
//!
//! Results equal `search::lexical::search` on the same corpus: the same
//! result set, the same scores and the same order (score descending, then
//! newest first). Doc ids are in recency order, so "newest first" is "lowest
//! doc id first".
//!
//! Body term frequencies and adjacent pair counts come from the index, and
//! title, summary and project are scored from card data, so ordinary queries
//! read no document text. A candidate's text is read only to verify literals
//! and to count words or pairs the index can only bound (longer than
//! `TERM_CAP`, or able to overlap themselves). Candidates are visited by
//! upper bound, so the top of the ranking is final once its exact scores
//! beat every remaining bound; the rest is completed on demand.
//!
//! What the index cannot narrow is checked newest first, and only as far as
//! the requested results need: literals without a word to look up, and
//! documents that may match a query with CJK words only inside words, which
//! the scorer accepts with a flat score. Those are narrowed to documents
//! containing the query's ideograph-only words, found through word-start and
//! inner terms.

use super::card::card_flags;
use super::generation::Generation;
use super::lexindex::{self, TERM_CAP};
use super::segments::IndexSet;
use crate::search::lexical::{
    WEIGHT_BODY, WEIGHT_PROJECT, WEIGHT_SUMMARY, WEIGHT_TITLE, adjacency_score, adjacent_pairs,
    count_adjacent_pairs, count_prefix_matches, freshness_bonus, identifier_literals,
    normalized_query_words, tf_score,
};
use crate::search::literal::{Literal, matches_all_literals};
use crate::search::query::ParsedQuery;
use crate::text_match::{contains_cjk, normalize_for_search};
use chrono::{DateTime, Local};
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

/// Field term frequencies are capped here, as in the scorer.
const TF_CAP: usize = 10;
/// Adjacent pairs counted per field.
const ADJACENCY_CAP: usize = 3;

/// Reads a document's text for verification and adjacency.
pub trait TextSource: Sync {
    /// `Conversation::full_text` of doc `doc`.
    fn full_text(&self, doc: u32) -> Cow<'_, str>;
}

/// Card fields of one document, normalized for scoring.
pub struct DocFields {
    pub title_lower: String,
    pub summary_lower: String,
    pub project: u32,
    pub timestamp_ms: i64,
}

pub struct ProjectFields {
    /// `Conversation::project_name`, which literals also match.
    pub name: Option<String>,
    pub lower: String,
}

/// Everything besides body text that scoring reads, indexed by doc id.
pub struct SearchDocs {
    pub docs: Vec<DocFields>,
    pub projects: Vec<ProjectFields>,
}

impl SearchDocs {
    pub fn from_generation(generation: &Generation) -> Self {
        let projects = (0..generation.projects().len() as u32)
            .map(|id| {
                let name = generation.project(id).name.map(str::to_owned);
                ProjectFields {
                    lower: name
                        .as_deref()
                        .map(normalize_for_search)
                        .unwrap_or_default(),
                    name,
                }
            })
            .collect();
        let docs = generation
            .rows()
            .iter()
            .map(|row| {
                let field = |flag: u8, r| {
                    if row.flags & flag != 0 {
                        normalize_for_search(generation.str(r))
                    } else {
                        String::new()
                    }
                };
                DocFields {
                    title_lower: field(card_flags::CUSTOM_TITLE, row.custom_title),
                    summary_lower: field(card_flags::SUMMARY, row.summary),
                    project: row.project_id,
                    timestamp_ms: row.timestamp_ms,
                }
            })
            .collect();
        Self { docs, projects }
    }

    fn project(&self, doc: u32) -> &ProjectFields {
        &self.projects[self.docs[doc as usize].project as usize]
    }
}

/// One result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub doc: u32,
    pub score: f64,
}

/// Work done by one query.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryStats {
    /// Documents that passed the index filter.
    pub candidates: usize,
    /// Document texts read so far.
    pub texts_read: usize,
}

/// Ranking order: higher score first, then lower doc id (newer) first.
fn rank_cmp(a: (f64, u32), b: (f64, u32)) -> Ordering {
    b.0.partial_cmp(&a.0)
        .unwrap_or(Ordering::Equal)
        .then(a.1.cmp(&b.1))
}

/// Max-heap entry: the best-ranked hit is the greatest.
struct Ranked(Hit);

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        rank_cmp((other.0.score, other.0.doc), (self.0.score, self.0.doc))
    }
}

/// A document that may match, with an upper bound on its score.
struct Candidate {
    doc: u32,
    /// Upper bound on the score, or the exact score when `!needs_text`.
    upper: f64,
    needs_text: bool,
    /// Title, summary and project score, exact.
    fields: f64,
    fresh: f64,
    /// Adjacent pairs in the body: the count, or an upper bound on it when
    /// `!adjacent_exact`.
    adjacent: usize,
    adjacent_exact: bool,
    /// Matches only if every word occurs as a substring, with score `upper`.
    flat: bool,
}

enum Plan {
    /// Every candidate's score is its freshness; text is read to verify
    /// `literals`.
    Literal {
        literals: Vec<Literal>,
    },
    Scored(Box<ScoredPlan>),
}

struct ScoredPlan {
    words: Vec<String>,
    pairs: Vec<String>,
    literals: Vec<Literal>,
    /// Index term frequencies per word, `words.len()` per candidate in
    /// `order`.
    tfs: Vec<u8>,
    /// Word whose index term frequency is exact.
    exact: Vec<bool>,
    /// Per project, per word: the word occurs in the project name.
    in_project: Vec<Vec<bool>>,
    /// The query has CJK words, so documents where a word occurs only
    /// inside words still match, with a flat score.
    cjk: bool,
}

/// Results of one query in final order, computed as far as requested.
pub struct Ranking<'a> {
    docs: &'a SearchDocs,
    text: &'a dyn TextSource,
    plan: Plan,
    /// Candidates not yet in `hits`, by descending upper bound.
    order: Vec<Candidate>,
    /// Position in `order` of the first candidate not yet evaluated.
    next: usize,
    /// Evaluated candidates waiting until no remaining bound can beat them.
    pending: BinaryHeap<Ranked>,
    hits: Vec<Hit>,
    candidates: usize,
    texts_read: AtomicUsize,
}

/// Starts `query` against `index`. `scope(doc)` restricts the documents
/// considered, like the `scope` of `search::debug_search`.
pub fn search<'a>(
    index: &IndexSet,
    docs: &'a SearchDocs,
    text: &'a dyn TextSource,
    query: &str,
    now: DateTime<Local>,
    scope: &(dyn Fn(u32) -> bool + Sync),
) -> Ranking<'a> {
    assert_eq!(index.n_docs() as usize, docs.docs.len());
    let fresh = |doc: u32| freshness_bonus(timestamp(docs.docs[doc as usize].timestamp_ms), now);
    let mut ranking = Ranking {
        docs,
        text,
        plan: Plan::Literal {
            literals: Vec::new(),
        },
        order: Vec::new(),
        next: 0,
        pending: BinaryHeap::new(),
        hits: Vec::new(),
        candidates: 0,
        texts_read: AtomicUsize::new(0),
    };
    let in_scope = || (0..docs.docs.len() as u32).filter(|&doc| scope(doc));

    let parsed = ParsedQuery::parse(query);
    if parsed.is_effectively_empty() {
        ranking.hits = in_scope()
            .map(|doc| Hit {
                doc,
                score: fresh(doc),
            })
            .collect();
        ranking.candidates = ranking.hits.len();
        return ranking;
    }

    let intent = parsed.lexical_text().trim();
    let (literals, words) = if parsed.is_quoted_only() {
        (parsed.literals().to_vec(), Vec::new())
    } else {
        let literals: Vec<Literal> = parsed
            .literals()
            .iter()
            .cloned()
            .chain(identifier_literals(intent))
            .collect();
        let words: Vec<String> = normalized_query_words(intent)
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        (literals, words)
    };

    if words.is_empty() {
        if literals.is_empty() {
            return search(index, docs, text, "", now, scope);
        }
        let mask = literal_mask(index, docs, &literals);
        ranking.order = in_scope()
            .filter(|&doc| mask.as_ref().is_none_or(|mask| mask[doc as usize]))
            .map(|doc| Candidate {
                doc,
                upper: fresh(doc),
                needs_text: true,
                fields: 0.0,
                fresh: fresh(doc),
                adjacent: 0,
                adjacent_exact: true,
                flat: false,
            })
            .collect();
        ranking.candidates = ranking.order.len();
        ranking.plan = Plan::Literal { literals };
        return ranking;
    }

    let pairs = adjacent_pairs(&words);
    let cjk = words.iter().any(|word| contains_cjk(word));
    let flat_score = words.len() as f64 * 0.5;

    let word_tfs: Vec<Vec<u8>> = words.par_iter().map(|w| index.prefix_tf(w)).collect();
    let exact: Vec<bool> = words.iter().map(|word| is_exact_word(word)).collect();
    // Pairs whose first word is longer than `TERM_CAP` have no pair terms;
    // their count is bounded by the word counts instead.
    let pair_terms: Vec<bool> = words[..pairs.len()]
        .iter()
        .map(|word| word.len() <= TERM_CAP)
        .collect();
    let pair_tfs: Vec<Vec<u8>> = pairs
        .par_iter()
        .zip(&pair_terms)
        .map(|(pair, &terms)| {
            if terms {
                index.prefix_tf(pair)
            } else {
                Vec::new()
            }
        })
        .collect();
    let pair_exact: Vec<bool> = pairs
        .iter()
        .zip(&pair_terms)
        .map(|(pair, &terms)| terms && !overlaps_itself(pair))
        .collect();
    let in_project: Vec<Vec<bool>> = docs
        .projects
        .iter()
        .map(|project| {
            words
                .iter()
                .map(|word| count_prefix_matches(&project.lower, word, 1) > 0)
                .collect()
        })
        .collect();
    let mask = literal_mask(index, docs, &literals);
    // Documents can match only by containing a query word with CJK if they
    // contain each ideographic word, which the index finds at word starts
    // and inside runs of ideographs.
    let ideographic: Vec<usize> = (0..words.len())
        .filter(|&w| cjk && is_ideographic(&words[w]))
        .collect();
    let inner_tfs: Vec<Vec<u8>> = ideographic
        .par_iter()
        .map(|&w| index.inner_tf(&words[w]))
        .collect();
    let may_contain = |doc: usize, project: &ProjectFields| {
        ideographic.iter().zip(&inner_tfs).all(|(&w, inner)| {
            word_tfs[w][doc] > 0 || inner[doc] > 0 || project.lower.contains(words[w].as_str())
        })
    };

    let mut tfs = Vec::new();
    let mut order = Vec::new();
    for doc in in_scope() {
        let d = doc as usize;
        let project = docs.docs[d].project as usize;
        if mask.as_ref().is_some_and(|mask| !mask[d]) {
            continue;
        }
        let doc_tfs: Vec<u8> = word_tfs.iter().map(|tf| tf[d]).collect();
        let qualifies = (0..words.len()).all(|w| doc_tfs[w] > 0 || in_project[project][w]);
        if !qualifies {
            if cjk && may_contain(d, &docs.projects[project]) {
                let fresh = fresh(doc);
                order.push(Candidate {
                    doc,
                    upper: flat_score + fresh,
                    needs_text: true,
                    fields: 0.0,
                    fresh,
                    adjacent: 0,
                    adjacent_exact: true,
                    flat: true,
                });
                tfs.extend_from_slice(&doc_tfs);
            }
            continue;
        }
        let fields = field_scores(docs, doc, &words, &pairs);
        let body = body_tf_score(doc_tfs.iter().map(|&tf| tf as usize));
        let mut adjacent = 0;
        let mut adjacent_exact = true;
        for (p, tf) in pair_tfs.iter().enumerate() {
            let count = if pair_terms[p] {
                tf[d]
            } else {
                doc_tfs[p].min(doc_tfs[p + 1])
            };
            adjacent += count as usize;
            adjacent_exact &= pair_exact[p] || count == 0;
        }
        let adjacent = adjacent.min(ADJACENCY_CAP);
        let needs_text = !literals.is_empty()
            || !adjacent_exact
            || doc_tfs
                .iter()
                .zip(&exact)
                .any(|(&tf, &exact)| tf > 0 && !exact);
        let fresh = fresh(doc);
        order.push(Candidate {
            doc,
            upper: ((fields + body) + adjacency_score(WEIGHT_BODY, adjacent)) + fresh,
            needs_text,
            fields,
            fresh,
            adjacent,
            adjacent_exact,
            flat: false,
        });
        tfs.extend_from_slice(&doc_tfs);
    }

    // Sort candidates and their tf rows together by upper bound.
    let mut perm: Vec<usize> = (0..order.len()).collect();
    perm.sort_unstable_by(|&a, &b| {
        rank_cmp(
            (order[a].upper, order[a].doc),
            (order[b].upper, order[b].doc),
        )
    });
    let n = words.len();
    let mut sorted_tfs = Vec::with_capacity(tfs.len());
    for &i in &perm {
        sorted_tfs.extend_from_slice(&tfs[i * n..(i + 1) * n]);
    }
    let mut slots: Vec<Option<Candidate>> = order.into_iter().map(Some).collect();
    ranking.order = perm
        .iter()
        .map(|&i| slots[i].take().expect("permutation"))
        .collect();
    ranking.candidates = ranking.order.len();
    ranking.plan = Plan::Scored(Box::new(ScoredPlan {
        words,
        pairs,
        literals,
        tfs: sorted_tfs,
        exact,
        in_project,
        cjk,
    }));
    ranking
}

impl Ranking<'_> {
    /// The first `n` results (fewer if there are fewer), final.
    pub fn top(&mut self, n: usize) -> &[Hit] {
        let threads = rayon::current_num_threads().max(1);
        while self.hits.len() < n {
            let bound = self
                .order
                .get(self.next)
                .map(|candidate| (candidate.upper, candidate.doc));
            match (self.pending.peek(), bound) {
                (Some(best), Some(bound))
                    if rank_cmp((best.0.score, best.0.doc), bound) == Ordering::Less =>
                {
                    self.hits.push(self.pending.pop().expect("peeked").0);
                }
                (Some(_), None) => self.hits.push(self.pending.pop().expect("peeked").0),
                (None, None) => break,
                _ => {
                    // Evaluate the next candidates in parallel, as many as
                    // results are still missing (at least one per thread),
                    // counting only those that read text.
                    let want = (n - self.hits.len()).clamp(threads, 4 * threads);
                    let best = self.pending.peek().map(|best| (best.0.score, best.0.doc));
                    let mut end = self.next;
                    let mut reads = 0;
                    while end < self.order.len() && reads < want {
                        let candidate = &self.order[end];
                        if best.is_some_and(|best| {
                            rank_cmp(best, (candidate.upper, candidate.doc)) == Ordering::Less
                        }) {
                            break;
                        }
                        reads += usize::from(candidate.needs_text);
                        end += 1;
                    }
                    let end = end.max(self.next + 1);
                    self.evaluate(self.next..end);
                    self.next = end;
                }
            }
        }
        &self.hits[..n.min(self.hits.len())]
    }

    /// All results, final.
    pub fn complete(&mut self) -> &[Hit] {
        self.evaluate(self.next..self.order.len());
        self.next = self.order.len();
        let mut rest: Vec<Hit> = std::mem::take(&mut self.pending)
            .into_iter()
            .map(|ranked| ranked.0)
            .collect();
        rest.sort_unstable_by(|a, b| rank_cmp((a.score, a.doc), (b.score, b.doc)));
        self.hits.extend(rest);
        &self.hits
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn stats(&self) -> QueryStats {
        QueryStats {
            candidates: self.candidates,
            texts_read: self.texts_read.load(AtomicOrdering::Relaxed),
        }
    }

    fn evaluate(&mut self, range: std::ops::Range<usize>) {
        let start = range.start;
        let hits: Vec<Hit> = self.order[range]
            .par_iter()
            .enumerate()
            .filter_map(|(i, candidate)| {
                self.exact_score(start + i, candidate).map(|score| Hit {
                    doc: candidate.doc,
                    score,
                })
            })
            .collect();
        self.pending.extend(hits.into_iter().map(Ranked));
    }

    /// Exact score of the candidate at `position` in `order`, or `None` when
    /// it does not match after all.
    fn exact_score(&self, position: usize, candidate: &Candidate) -> Option<f64> {
        if !candidate.needs_text {
            return Some(candidate.upper);
        }
        self.texts_read.fetch_add(1, AtomicOrdering::Relaxed);
        let full_text = self.text.full_text(candidate.doc);
        let project = self.docs.project(candidate.doc);
        let plan = match &self.plan {
            Plan::Literal { literals } => {
                return literals_match(&full_text, project.name.as_deref(), literals)
                    .then_some(candidate.fresh);
            }
            Plan::Scored(plan) => plan,
        };
        if !literals_match(&full_text, project.name.as_deref(), &plan.literals) {
            return None;
        }
        let flat = || {
            // Ideographs have no case and are not separators, so they occur
            // in the normalized text exactly where they occur in the text.
            if plan
                .words
                .iter()
                .filter(|word| is_ideographic(word))
                .any(|word| {
                    !full_text.contains(word.as_str()) && !project.lower.contains(word.as_str())
                })
            {
                return None;
            }
            let body = normalize_for_search(&full_text);
            plan.words
                .iter()
                .all(|word| body.contains(word.as_str()) || project.lower.contains(word.as_str()))
                .then_some(plan.words.len() as f64 * 0.5 + candidate.fresh)
        };
        if candidate.flat {
            return flat();
        }
        let n = plan.words.len();
        let mut tfs: Vec<usize> = plan.tfs[position * n..(position + 1) * n]
            .iter()
            .map(|&tf| tf as usize)
            .collect();
        let inexact = tfs
            .iter()
            .zip(&plan.exact)
            .any(|(&tf, &exact)| tf > 0 && !exact);
        if candidate.adjacent_exact && !inexact {
            return Some(candidate.upper);
        }

        let body = normalize_for_search(&full_text);
        let in_project = &plan.in_project[self.docs.docs[candidate.doc as usize].project as usize];
        for (w, word) in plan.words.iter().enumerate() {
            if tfs[w] > 0 && !plan.exact[w] {
                tfs[w] = count_prefix_matches(&body, word, TF_CAP);
            }
            if tfs[w] == 0 && !in_project[w] {
                return if plan.cjk { flat() } else { None };
            }
        }
        let adjacent = if candidate.adjacent_exact {
            candidate.adjacent
        } else {
            count_adjacent_pairs(&body, &plan.pairs, ADJACENCY_CAP)
        };
        Some(
            ((candidate.fields + body_tf_score(tfs.into_iter()))
                + adjacency_score(WEIGHT_BODY, adjacent))
                + candidate.fresh,
        )
    }
}

fn timestamp(ms: i64) -> DateTime<Local> {
    DateTime::from_timestamp_millis(ms)
        .unwrap_or_default()
        .with_timezone(&Local)
}

/// Whether the index counts `word` exactly as `count_prefix_matches` does:
/// it is at most `TERM_CAP` bytes (longer words are looked up by a prefix)
/// and cannot overlap itself.
pub fn is_exact_word(word: &str) -> bool {
    word.len() <= TERM_CAP && !overlaps_itself(word)
}

/// Whether an occurrence of `word` can begin at a word start inside another
/// occurrence: when a proper suffix of `word` that follows a
/// non-alphanumeric char is also a prefix of it (`a.a` in `a.a.a`). The
/// index counts every word-start occurrence, while the scorer counts
/// non-overlapping occurrences left to right and keeps those at word starts,
/// so the two agree for words that cannot overlap themselves.
fn overlaps_itself(word: &str) -> bool {
    let mut prev_alnum = true;
    for (i, ch) in word.char_indices() {
        if i > 0 && !prev_alnum && word.starts_with(&word[i..]) {
            return true;
        }
        prev_alnum = ch.is_alphanumeric();
    }
    false
}

/// Title, summary and project score of `doc`, accumulated exactly as the
/// scorer does before it adds the body.
fn field_scores(docs: &SearchDocs, doc: u32, words: &[String], pairs: &[String]) -> f64 {
    let fields = &docs.docs[doc as usize];
    let mut score = 0.0;
    for (field, weight) in [
        (fields.title_lower.as_str(), WEIGHT_TITLE),
        (fields.summary_lower.as_str(), WEIGHT_SUMMARY),
        (docs.project(doc).lower.as_str(), WEIGHT_PROJECT),
    ] {
        if field.is_empty() {
            continue;
        }
        let mut tf = 0.0;
        for word in words {
            tf += tf_score(count_prefix_matches(field, word, TF_CAP));
        }
        score += weight * tf;
        let adjacent = if pairs.is_empty() {
            0
        } else {
            count_adjacent_pairs(field, pairs, ADJACENCY_CAP)
        };
        score += adjacency_score(weight, adjacent);
    }
    score
}

fn body_tf_score(tfs: impl Iterator<Item = usize>) -> f64 {
    let mut score = 0.0;
    for tf in tfs {
        score += tf_score(tf);
    }
    WEIGHT_BODY * score
}

/// Whether `word` consists of ideographs only.
fn is_ideographic(word: &str) -> bool {
    word.chars().all(lexindex::is_ideograph)
}

/// What literals match: `full_text`, then the project name, joined by a
/// space as in the literal corpus.
fn literals_match(full_text: &str, project_name: Option<&str>, literals: &[Literal]) -> bool {
    if literals.is_empty() || matches_all_literals(full_text, literals) {
        return true;
    }
    match project_name.filter(|name| !name.is_empty()) {
        Some(name) if !full_text.is_empty() => {
            matches_all_literals(&format!("{full_text} {name}"), literals)
        }
        Some(name) => matches_all_literals(name, literals),
        None => false,
    }
}

/// Documents that can contain every literal; `None` when no literal has a
/// term to look up.
///
/// A literal occurrence starts at a word start, and so does every word inside
/// it, so the normalized text holds, at each of those positions, an index term
/// that starts with the term the literal itself yields there. A document
/// lacking any of a literal's terms can still match through the text the
/// literal corpus appends: the project name, after a space.
fn literal_mask(index: &IndexSet, docs: &SearchDocs, literals: &[Literal]) -> Option<Vec<bool>> {
    let mut mask: Option<Vec<bool>> = None;
    for literal in literals {
        let lowered = normalize_for_search(literal.text());
        let mut terms = Vec::new();
        lexindex::for_each_term(&lowered, |term| terms.push(term));
        terms.sort_unstable();
        terms.dedup();
        if terms.is_empty() {
            continue;
        }
        let tfs: Vec<Vec<u8>> = terms.par_iter().map(|term| index.prefix_tf(term)).collect();
        let in_project: Vec<bool> = docs
            .projects
            .iter()
            .map(|project| project_may_hold(&lowered, project))
            .collect();
        let mask = mask.get_or_insert_with(|| vec![true; docs.docs.len()]);
        for (doc, keep) in mask.iter_mut().enumerate() {
            *keep &=
                in_project[docs.docs[doc].project as usize] || tfs.iter().all(|tf| tf[doc] > 0);
        }
    }
    mask
}

/// Whether a normalized literal can occur in, or run into, `project`'s name.
fn project_may_hold(lowered: &str, project: &ProjectFields) -> bool {
    if project.name.as_deref().is_none_or(str::is_empty) {
        return false;
    }
    project.lower.contains(lowered)
        || lowered
            .match_indices(' ')
            .any(|(i, _)| project.lower.starts_with(&lowered[i + 1..]))
}

#[cfg(test)]
impl SearchDocs {
    /// Fields of `conversations`, which must be in doc id order.
    pub(crate) fn from_conversations(conversations: &[crate::history::Conversation]) -> Self {
        let mut ids = std::collections::HashMap::new();
        let mut projects = Vec::new();
        let docs = conversations
            .iter()
            .map(|conversation| {
                let name = conversation.project_name.clone();
                let project = *ids.entry(name.clone()).or_insert_with(|| {
                    projects.push(ProjectFields {
                        lower: name
                            .as_deref()
                            .map(normalize_for_search)
                            .unwrap_or_default(),
                        name,
                    });
                    projects.len() as u32 - 1
                });
                let lower = |field: &Option<String>| {
                    field
                        .as_deref()
                        .map(normalize_for_search)
                        .unwrap_or_default()
                };
                DocFields {
                    title_lower: lower(&conversation.custom_title),
                    summary_lower: lower(&conversation.summary),
                    project,
                    timestamp_ms: conversation.timestamp.timestamp_millis(),
                }
            })
            .collect();
        Self { docs, projects }
    }
}

#[cfg(test)]
pub(crate) mod oracle {
    //! Compares index search with `search::debug_search`.

    use super::*;
    use crate::history::Conversation;
    use crate::search::{SearchableConversation, debug_search, precompute_search_text};
    use crate::store::card::{BlobRef, CardRow};
    use crate::store::lexindex;
    use crate::store::text::{self, TextSegments};
    use bytemuck::Zeroable;
    use chrono::TimeZone;

    /// Texts read from a zstd text segment, as from a generation.
    pub(crate) struct Frames {
        pub segments: TextSegments,
        pub refs: Vec<BlobRef>,
    }

    impl TextSource for Frames {
        fn full_text(&self, doc: u32) -> Cow<'_, str> {
            Cow::Owned(
                self.segments
                    .read(self.refs[doc as usize])
                    .unwrap_or_default(),
            )
        }
    }

    /// `conversations` sorted into the doc order of cards (newest first,
    /// then by path bytes), with timestamps cut to the milliseconds cards
    /// store.
    pub(crate) fn into_doc_order(mut conversations: Vec<Conversation>) -> Vec<Conversation> {
        for conversation in &mut conversations {
            let ms = conversation.timestamp.timestamp_millis();
            conversation.timestamp = Local.timestamp_millis_opt(ms).unwrap();
        }
        conversations.sort_by(|a, b| {
            b.timestamp
                .cmp(&a.timestamp)
                .then_with(|| a.path.as_os_str().cmp(b.path.as_os_str()))
        });
        for (index, conversation) in conversations.iter_mut().enumerate() {
            conversation.index = index;
        }
        conversations
    }

    pub(crate) struct Corpus {
        pub conversations: Vec<Conversation>,
        pub searchable: Vec<SearchableConversation>,
        pub docs: SearchDocs,
        pub index: IndexSet,
        pub frames: Frames,
        pub build: lexindex::BuildStats,
        pub build_time: std::time::Duration,
        _dir: tempfile::TempDir,
    }

    impl Corpus {
        pub(crate) fn new(conversations: Vec<Conversation>) -> Self {
            let conversations = into_doc_order(conversations);
            let started = std::time::Instant::now();
            let built = lexindex::build(conversations.len(), |doc| {
                Cow::Owned(normalize_for_search(&conversations[doc as usize].full_text))
            });
            let dir = tempfile::tempdir().unwrap();
            built.write(dir.path(), 0).unwrap();
            let build_time = started.elapsed();
            let index = IndexSet::single(
                lexindex::LexIndex::open(dir.path(), 0, conversations.len() as u32).unwrap(),
            );
            let texts: Vec<&str> = conversations.iter().map(|c| &*c.full_text).collect();
            let refs = text::write_segment(&text::segment_path(dir.path(), 0), 0, &texts).unwrap();
            let rows: Vec<CardRow> = refs
                .iter()
                .map(|&text| CardRow {
                    text,
                    ..CardRow::zeroed()
                })
                .collect();
            let segments = TextSegments::open(dir.path(), &rows).unwrap();
            Self {
                searchable: precompute_search_text(&conversations),
                docs: SearchDocs::from_conversations(&conversations),
                index,
                frames: Frames { segments, refs },
                build: built.stats,
                build_time,
                conversations,
                _dir: dir,
            }
        }

        pub(crate) fn expected(&self, query: &str, now: DateTime<Local>) -> Vec<Hit> {
            let mut hits: Vec<Hit> =
                debug_search(&self.conversations, &self.searchable, query, now, |_| true)
                    .results
                    .into_iter()
                    .map(|(index, debug)| Hit {
                        doc: index as u32,
                        score: debug.total,
                    })
                    .collect();
            // `search` leaves equal timestamps in any order; doc ids break
            // those ties.
            hits.sort_by(|a, b| rank_cmp((a.score, a.doc), (b.score, b.doc)));
            hits
        }

        /// Asserts that index search, reading text from zstd frames, returns
        /// exactly the in-memory results, and that its top page is a prefix of
        /// them. Returns the stats.
        pub(crate) fn check(&self, query: &str, now: DateTime<Local>, page: usize) -> QueryStats {
            let expected = self.expected(query, now);
            let mut ranking = search(&self.index, &self.docs, &self.frames, query, now, &|_| true);
            let top = ranking.top(page).to_vec();
            let all = ranking.complete().to_vec();
            assert_eq!(top[..], all[..top.len()], "query {query:?}: top page");
            assert_eq!(
                top.len(),
                page.min(expected.len()),
                "query {query:?}: page size"
            );
            assert_same_hits(query, &all, &expected);
            ranking.stats()
        }
    }

    /// Asserts that `actual` holds the `expected` docs in order with the
    /// same scores.
    pub(crate) fn assert_same_hits(query: &str, actual: &[Hit], expected: &[Hit]) {
        if let Some(at) = actual
            .iter()
            .zip(expected)
            .position(|(a, e)| a.doc != e.doc || (a.score - e.score).abs() > 1e-9)
        {
            let near = |hits: &[Hit]| {
                hits[at.saturating_sub(2)..(at + 3).min(hits.len())]
                    .iter()
                    .map(|hit| format!("{}:{:.6}", hit.doc, hit.score))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            panic!(
                "query {query:?}: results differ at {at} of {}/{}: {} != {}",
                actual.len(),
                expected.len(),
                near(actual),
                near(expected)
            );
        }
        let render = |hits: &[Hit]| {
            hits.iter()
                .map(|hit| format!("{}:{:.6}", hit.doc, hit.score))
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert_eq!(
            render(actual),
            render(expected),
            "query {query:?}: results differ"
        );
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual.score - expected.score).abs() <= 1e-9,
                "query {query:?}: doc {} score {} != {}",
                actual.doc,
                actual.score,
                expected.score
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::oracle::Corpus;
    use super::*;
    use crate::history::Conversation;
    use crate::search::test_fixtures::one_message_conversation;
    use chrono::{Duration, TimeZone};

    fn conv(
        text: &str,
        project: Option<&str>,
        title: Option<&str>,
        summary: Option<&str>,
        timestamp: DateTime<Local>,
        path: &str,
    ) -> Conversation {
        let mut conversation = one_message_conversation(text, timestamp, summary, title, project);
        conversation.path = path.into();
        conversation
    }

    #[test]
    fn exact_words_exclude_self_overlapping_and_long_words() {
        for word in ["error", "a.b", "foo.bar", "x", "café", "a.", ".a"] {
            assert!(is_exact_word(word), "{word}");
        }
        for word in ["a.a", "..", "ab.ab", "x(x", &"y".repeat(TERM_CAP + 1)] {
            assert!(!is_exact_word(word), "{word}");
        }
        // Borders after an alphanumeric char cannot start at a word start.
        assert!(is_exact_word("abab"));
    }

    /// The fixture corpus of `score_impl_golden_results`.
    fn golden_corpus(now: DateTime<Local>) -> Vec<Conversation> {
        vec![
            conv(
                "error handling in the cache layer, error handling everywhere",
                Some("workmux/cache-fix"),
                Some("Error handling cleanup"),
                Some("Refactor error paths"),
                now - Duration::hours(2),
                "/0",
            ),
            conv(
                "tmux session handling and window errors",
                Some("dotfiles"),
                None,
                None,
                now - Duration::days(3),
                "/1",
            ),
            conv(
                "unrelated chat that mentions nothing",
                Some("claude-history"),
                None,
                Some("Search ranking notes"),
                now - Duration::days(10),
                "/2",
            ),
            conv(
                "body ends with handling",
                Some("error-reporter"),
                None,
                None,
                now - Duration::days(1),
                "/3",
            ),
            conv(
                "",
                Some("tmux-config"),
                None,
                None,
                now - Duration::days(40),
                "/4",
            ),
            conv(
                "precache and subcache only as substrings",
                Some("rust/cache"),
                None,
                None,
                now - Duration::hours(30),
                "/5",
            ),
            conv(
                "\u{6D4B}\u{8BD5} cache \u{4F1A}\u{8BDD}",
                Some("\u{9879}\u{76EE}"),
                None,
                None,
                now - Duration::days(5),
                "/6",
            ),
            conv("no project here cache cache", None, None, None, now, "/7"),
            conv(
                "dialogue about tmux panes",
                Some("tools/agent-runner"),
                None,
                None,
                now - Duration::hours(3),
                "/8",
            ),
        ]
    }

    #[test]
    fn golden_fixture_queries_match_in_memory_search() {
        let now = Local.timestamp_opt(1_760_000_000, 0).unwrap();
        let corpus = Corpus::new(golden_corpus(now));
        for query in [
            "error handling",
            "tmux",
            "cache",
            "cache-fix",
            "workmux cache",
            "error reporter",
            "handling error",
            "dotfiles window",
            "\u{6D4B}\u{8BD5} \u{9879}\u{76EE}",
            "\u{9879}\u{76EE}",
            "rust",
            "\"error handling\"",
            "error \"cache layer\"",
            "warmup",
            "",
            "\"\"",
            "---",
            "e",
            "ca",
            "cache_layer",
            "\"cache\"",
            "\"Cache\"",
            "\"cache fix\"",
            "\"layer, error\"",
        ] {
            for page in [1, 3, 50] {
                corpus.check(query, now, page);
            }
        }
    }

    #[test]
    fn self_overlapping_word_is_counted_like_the_scorer() {
        let now = Local.timestamp_opt(1_760_000_000, 0).unwrap();
        let corpus = Corpus::new(vec![
            conv("a.a.a a.a.a", None, None, None, now, "/0"),
            conva("x.x.x.x", now - Duration::hours(1), "/1"),
            conva("ya.a.a z", now - Duration::hours(2), "/2"),
            conva("..x ... .", now - Duration::hours(3), "/3"),
        ]);
        fn conva(text: &str, timestamp: DateTime<Local>, path: &str) -> Conversation {
            conv(text, None, None, None, timestamp, path)
        }
        for query in ["a.a", "a.a a.a", "x.x", "x.x.x", "..", ".", "a.a z", "ya.a"] {
            corpus.check(query, now, 10);
        }
        // The index alone overcounts these, so their candidates are read.
        let texts = &corpus.frames;
        let mut ranking = search(&corpus.index, &corpus.docs, texts, "x.x", now, &|_| true);
        ranking.complete();
        assert_eq!(ranking.stats().texts_read, 1);
    }

    #[test]
    fn ordinary_queries_read_no_text_and_top_pages_read_little() {
        let now = Local.timestamp_opt(1_760_000_000, 0).unwrap();
        let conversations: Vec<Conversation> = (0..200)
            .map(|i| {
                conv(
                    if i % 2 == 0 {
                        "alpha beta filler"
                    } else {
                        "beta alpha filler"
                    },
                    None,
                    None,
                    None,
                    now - Duration::hours(i),
                    &format!("/{i:03}"),
                )
            })
            .collect();
        let corpus = Corpus::new(conversations);
        for query in ["alpha beta", "alpha be", "beta", "alpha beta filler"] {
            let stats = corpus.check(query, now, 5);
            assert_eq!(stats.texts_read, 0, "{query}");
        }

        // Literal candidates are the documents holding the literal's terms,
        // here half the corpus, verified only as far as the page needs.
        let query = "alpha \"beta filler\"";
        let stats = corpus.check(query, now, 5);
        assert_eq!((stats.candidates, stats.texts_read), (100, 100));
        let texts = &corpus.frames;
        let mut ranking = search(&corpus.index, &corpus.docs, texts, query, now, &|_| true);
        assert_eq!(ranking.top(5).len(), 5);
        let threads = rayon::current_num_threads();
        let read = ranking.stats().texts_read;
        assert!(read <= 2 * 5 + 4 * threads, "read {read}");
    }

    #[test]
    fn literals_match_at_word_starts_and_into_the_project_name() {
        let now = Local.timestamp_opt(1_760_000_000, 0).unwrap();
        let texts = [
            ("load_all_conversations_streaming()", None),
            ("subconversations_streaming", None),
            ("ship the deploy", Some("error-reporter")),
            ("deploy error later", None),
            ("", Some("error-reporter")),
        ];
        let conversations = texts
            .iter()
            .enumerate()
            .map(|(i, (text, project))| {
                conv(
                    text,
                    *project,
                    None,
                    None,
                    now - Duration::hours(i as i64),
                    &format!("/{i}"),
                )
            })
            .collect();
        let corpus = Corpus::new(conversations);
        let docs = |query: &str| -> Vec<u32> {
            corpus.check(query, now, 10);
            let texts = &corpus.frames;
            let mut ranking = search(&corpus.index, &corpus.docs, texts, query, now, &|_| true);
            ranking.complete().iter().map(|hit| hit.doc).collect()
        };

        assert_eq!(docs("\"conversations_streaming\""), [0]);
        assert_eq!(docs("\"onversations\""), [] as [u32; 0]);
        assert_eq!(docs("\"deploy error\""), [2, 3]);
        assert_eq!(docs("\"deploy error-rep\""), [2]);
        assert_eq!(docs("\"reporter\""), [2, 4]);
    }

    #[test]
    fn cjk_queries_read_only_documents_containing_their_ideographs() {
        let now = Local.timestamp_opt(1_760_000_000, 0).unwrap();
        let conversations: Vec<Conversation> = (0..100)
            .map(|i| {
                let text = match i {
                    10 => "这是测试 cache",
                    20 => "测试 cache",
                    30 => "x测试y",
                    40 => "测 试",
                    _ => "plain cache text",
                };
                conv(
                    text,
                    None,
                    None,
                    None,
                    now - Duration::hours(i),
                    &format!("/{i:03}"),
                )
            })
            .collect();
        let corpus = Corpus::new(conversations);
        for (query, read) in [("测试", 2), ("测试 cache", 2), ("试", 3), ("cache 测试", 2)] {
            let stats = corpus.check(query, now, 10);
            assert_eq!(stats.texts_read, read, "{query}");
        }
    }

    #[test]
    fn generation_cards_give_the_same_fields_as_conversations() {
        let now = Local.timestamp_opt(1_760_000_000, 0).unwrap();
        let conversations = super::oracle::into_doc_order(golden_corpus(now));
        let root = tempfile::tempdir().unwrap();
        crate::store::generation::publish(root.path(), &conversations).unwrap();
        let generation = Generation::open(root.path()).unwrap();

        let from_cards = SearchDocs::from_generation(&generation);
        let expected = SearchDocs::from_conversations(&conversations);
        assert_eq!(from_cards.docs.len(), expected.docs.len());
        for doc in 0..expected.docs.len() as u32 {
            let (a, b) = (&from_cards.docs[doc as usize], &expected.docs[doc as usize]);
            assert_eq!(a.title_lower, b.title_lower);
            assert_eq!(a.summary_lower, b.summary_lower);
            assert_eq!(a.timestamp_ms, b.timestamp_ms);
            assert_eq!(from_cards.project(doc).lower, expected.project(doc).lower);
            assert_eq!(from_cards.project(doc).name, expected.project(doc).name);
        }
    }

    #[test]
    fn scope_restricts_results() {
        let now = Local.timestamp_opt(1_760_000_000, 0).unwrap();
        let corpus = Corpus::new(golden_corpus(now));
        let texts = &corpus.frames;
        for query in [
            "cache",
            "\"cache\"",
            "",
            "\u{9879}\u{76EE}",
            "\"cache layer\"",
        ] {
            let mut ranking = search(&corpus.index, &corpus.docs, texts, query, now, &|doc| {
                doc % 2 == 0
            });
            let scoped: Vec<Hit> = ranking.complete().to_vec();
            let expected: Vec<Hit> = corpus
                .expected(query, now)
                .into_iter()
                .filter(|hit| hit.doc % 2 == 0)
                .collect();
            assert_eq!(scoped, expected, "{query}");
        }
    }

    /// Small deterministic generator, so failures reproduce from the seed.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
            items[self.below(items.len())]
        }
    }

    const VOCAB: &[&str] = &[
        "error",
        "errors",
        "err",
        "handling",
        "handle",
        "handled",
        "cache",
        "cachex",
        "precache",
        "layer",
        "a",
        "ab",
        "abc",
        "the",
        "then",
        "theme",
        "a.a",
        "a.a.a",
        "x.x",
        "..",
        "foo.bar",
        "Foo",
        "BAR",
        "café",
        "cafe",
        "İstanbul",
        "straße",
        "测试",
        "会话",
        "这是测试",
        "x会话y",
        "项目",
        "foo_bar",
        "API_KEY",
        "restaurant_signals",
        "e7d318b1",
        "superlongidentifierwithoutanybreaks42",
        "superlongidentifierwithoutanybreaks43",
        "**bold**",
        "(paren)",
        "tmux",
        "window",
        "1.2.3",
        "v1",
        "x",
        "été",
        "naïve",
    ];
    const SEPARATORS: &[&str] = &[
        " ", " ", " ", "_", "-", "/", ".", "\n", ", ", "\u{3001}", "  ",
    ];
    const PROJECTS: &[&str] = &[
        "cache",
        "error-reporter",
        "tmux-config",
        "项目",
        "foo_bar",
        "",
    ];

    fn generated_corpus(seed: u64, docs: usize) -> (Vec<Conversation>, DateTime<Local>) {
        let mut rng = Rng(seed);
        let now = Local.timestamp_opt(1_760_000_000, 0).unwrap();
        let conversations = (0..docs)
            .map(|i| {
                let len = rng.below(120);
                let mut text = String::new();
                for _ in 0..len {
                    text.push_str(rng.pick(VOCAB));
                    text.push_str(rng.pick(SEPARATORS));
                }
                let project = rng.pick(PROJECTS);
                let project = (rng.below(5) > 0).then_some(project);
                let title =
                    (rng.below(6) == 0).then(|| format!("{} {}", rng.pick(VOCAB), rng.pick(VOCAB)));
                let summary =
                    (rng.below(6) == 0).then(|| format!("{} {}", rng.pick(VOCAB), rng.pick(VOCAB)));
                // Coarse timestamps, so some tie.
                let age = Duration::hours(rng.below(24 * 30) as i64 / 6 * 6);
                conv(
                    &text,
                    project,
                    title.as_deref(),
                    summary.as_deref(),
                    now - age,
                    &format!("/{i:04}"),
                )
            })
            .collect();
        (conversations, now)
    }

    fn generated_queries(seed: u64) -> Vec<String> {
        let mut rng = Rng(seed);
        let mut queries: Vec<String> = [
            "a.a",
            "a.a a",
            "x.x",
            "..",
            "e",
            "er",
            "ca",
            "th",
            "the",
            "superlongidentifierwithout",
            "superlongidentifierwithoutanybreaks4",
            "superlongidentifierwithoutanybreaks42",
            "\"a.a\"",
            "\"error handling\"",
            "\"Foo\"",
            "\"foo\"",
            "foo_bar",
            "API_KEY",
            "restaurant_signals error",
            "error \"cache layer\"",
            "\"handling error\" cache",
            "测试",
            "测试 cache",
            "\"测试\"",
            "项目",
            "error-reporter",
            "tmux config",
            "**bold",
            "(paren",
            "1.2",
            "café",
            "İstanbul",
            "istanbul",
            "straße",
            "STRASSE",
            "été naïve",
            "",
            "\"\"",
            "_",
            "-- --",
            "cache cache cache",
            "\"cache\" \"layer\"",
            "\"err\" handling",
            "e7d318b1",
            "zzz",
            "error zzz",
            "\"rror\"",
            "\"bar\"",
            "\"_bar\"",
            "\"KEY\"",
            "\"2.3\"",
            "\"aïve\"",
            "\"试\"",
            "\"y\"",
            "\"r_bar cache\"",
            "\"superlongidentifierwithoutanybreaks42 tmux\"",
            "error \"andling\"",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        for _ in 0..60 {
            let words = 1 + rng.below(3);
            let mut query = (0..words)
                .map(|_| {
                    let word = rng.pick(VOCAB);
                    let cut = word
                        .char_indices()
                        .map(|(i, _)| i)
                        .chain([word.len()])
                        .filter(|&i| i > 0)
                        .collect::<Vec<_>>();
                    word[..cut[rng.below(cut.len())]].to_owned()
                })
                .collect::<Vec<_>>()
                .join(rng.pick(&[" ", " ", "-", "_"]));
            if rng.below(5) == 0 {
                query = format!("\"{query}\"");
            } else if rng.below(5) == 0 {
                query = format!("{} \"{}\"", query, rng.pick(VOCAB));
            }
            queries.push(query);
        }
        queries
    }

    #[test]
    fn generated_corpora_match_in_memory_search() {
        for seed in [1, 2, 3, 0x9e37_79b9_7f4a_7c15] {
            let (conversations, now) = generated_corpus(seed, 300);
            let corpus = Corpus::new(conversations);
            assert!(corpus.build.parts > 1);
            for query in generated_queries(seed) {
                for page in [1, 20] {
                    corpus.check(&query, now, page);
                }
            }
        }
    }

    /// Asserts that searching the generation `CURRENT` names under `root`
    /// returns the in-memory results over `conversations`.
    fn check_generation(
        root: &std::path::Path,
        conversations: &[Conversation],
        queries: &[&str],
        now: DateTime<Local>,
    ) {
        let corpus = Corpus::new(conversations.to_vec());
        let snapshot =
            crate::store::Snapshot::open(root, false, crate::time_filter::TimeFilter::default())
                .unwrap();
        let index = crate::store::SearchIndex::open(snapshot).unwrap();
        for query in queries {
            let mut ranking = index.search(query, now, &|_| true);
            let all = ranking.complete().to_vec();
            super::oracle::assert_same_hits(query, &all, &corpus.expected(query, now));
        }
    }

    /// Publishes `live` as the next generation under `lock`, adding a delta
    /// for the changed conversations.
    fn publish_live(lock: &crate::store::generation::PublishLock, live: &[Conversation]) {
        crate::store::generation::publish_all(lock, live).unwrap();
    }

    #[test]
    fn generations_with_deltas_and_merges_match_in_memory_search() {
        use crate::store::generation::{Generation, MergePolicy, PublishLock, compact};
        for seed in [1, 2] {
            let (all, now) = generated_corpus(seed, 400);
            let queries = generated_queries(seed);
            let queries: Vec<&str> = queries.iter().map(String::as_str).collect();
            let root = tempfile::tempdir().unwrap();
            let lock = PublishLock::try_acquire(root.path()).unwrap().unwrap();
            let mut live = all[..250].to_vec();
            publish_live(&lock, &live);
            for (round, added) in all[250..].chunks(25).take(5).enumerate() {
                for conversation in live.iter_mut().skip(round).step_by(17) {
                    conversation.full_text =
                        format!("{} {}", conversation.full_text, VOCAB[round]).into();
                }
                let removed = (round * 37) % live.len();
                live.remove(removed);
                live.extend_from_slice(added);
                publish_live(&lock, &live);
                check_generation(root.path(), &live, &queries, now);
            }
            let generation = Generation::open(root.path()).unwrap();
            assert_eq!(generation.segs().len(), 6, "seed {seed}");

            compact(&lock, &generation, MergePolicy::DEFAULT).unwrap();
            let merged = Generation::open(root.path()).unwrap();
            assert!(merged.segs().len() < 6, "seed {seed}");
            check_generation(root.path(), &live, &queries, now);

            compact(&lock, &merged, MergePolicy::FULL).unwrap();
            assert_eq!(Generation::open(root.path()).unwrap().segs().len(), 1);
            check_generation(root.path(), &live, &queries, now);
        }
    }

    /// Sample queries for the real-corpus oracle and benchmark.
    const REAL_QUERIES: &[&str] = &[
        "error handling",
        "search ranking",
        "tmux",
        "workmux",
        "cargo test",
        "rust",
        "fn",
        "the",
        "a",
        "e",
        "co",
        "con",
        "config",
        "claude history",
        "semantic search",
        "zstd",
        "fst",
        "mmap",
        "generation snapshot",
        "pull request",
        "git rebase",
        "worktree",
        "commit message",
        "typescript",
        "react component",
        "database migration",
        "sqlite",
        "api key",
        "docker",
        "nix flake",
        "neovim",
        "lua",
        "python script",
        "performance",
        "benchmark",
        "latency p95",
        "memory usage",
        "async await",
        "tokio",
        "serde json",
        "unit test",
        "integration test",
        "readme",
        "changelog",
        "release",
        "homebrew",
        "github actions",
        "ci failure",
        "how do i",
        "why does",
        "can you",
        "please fix",
        "the quick brown",
        "x",
        "1",
        "src/main.rs",
        "Cargo.toml",
        "localhost:3000",
        "http",
        "https github com",
        "claude-history",
        "configuration management system",
        "superlongidentifierthatdoesnotexistanywhere",
        "normalize for search",
        "score impl golden",
        "token usage",
        "file not found",
        "permission denied",
        "\"error handling\"",
        "\"cargo test\"",
        "\"fn main\"",
        "\"TODO\"",
        "search_text_lower",
        "normalize_for_search",
        "score_impl",
        "error \"cache layer\"",
        "\"search_text_lower\"",
        "测试",
        "日本語",
        "a.a",
        "e7d318b1",
    ];

    fn real_corpus() -> Corpus {
        let started = std::time::Instant::now();
        let conversations = crate::history::load_all_conversations(false, None).unwrap();
        eprintln!(
            "loaded {} conversations in {:.2?}",
            conversations.len(),
            started.elapsed()
        );
        let corpus = Corpus::new(conversations);
        let stats = &corpus.build;
        let text: usize = corpus
            .conversations
            .iter()
            .map(|conversation| conversation.full_text.len())
            .sum();
        eprintln!(
            "index build {:.2?} for {} docs, {:.1} MB of text: {} terms, {} postings \
             ({} inline), {} short prefixes, {} parts",
            corpus.build_time,
            stats.docs,
            text as f64 / 1e6,
            stats.terms,
            stats.postings,
            stats.inline_postings,
            stats.short_prefixes,
            stats.parts
        );
        eprintln!(
            "sizes: base.fst {:.1} MB, base.post {:.1} MB, base.short {:.1} MB, total {:.1} MB, \
             text-0000.zst {:.1} MB",
            stats.fst_bytes as f64 / 1e6,
            stats.post_bytes as f64 / 1e6,
            stats.short_bytes as f64 / 1e6,
            (stats.fst_bytes + stats.post_bytes + stats.short_bytes) as f64 / 1e6,
            corpus.frames.segments.disk_bytes() as f64 / 1e6
        );
        corpus
    }

    /// Checks index search against the in-memory search on the user's
    /// conversations. Run with
    /// `cargo test --release --bin claude-history real_corpus_oracle -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_corpus_oracle() {
        let corpus = real_corpus();
        let now = Local::now();
        for query in REAL_QUERIES {
            let stats = corpus.check(query, now, 50);
            eprintln!(
                "ok {query:?}: {} candidates, {} texts read",
                stats.candidates, stats.texts_read
            );
        }
    }

    /// The real corpus published as an older base plus a delta holding the
    /// newest sessions and edits to older ones, searched against the
    /// in-memory results.
    #[test]
    #[ignore = "reads the real history; run with --ignored"]
    fn real_corpus_generation_oracle() {
        let mut conversations = crate::history::load_all_conversations(false, None).unwrap();
        conversations.sort_by_key(|conversation| std::cmp::Reverse(conversation.timestamp));
        let now = Local::now();
        let root = tempfile::tempdir().unwrap();
        let lock = crate::store::generation::PublishLock::try_acquire(root.path())
            .unwrap()
            .unwrap();
        publish_live(&lock, &conversations[300..]);
        for conversation in conversations.iter_mut().skip(300).step_by(97) {
            conversation.full_text =
                format!("{} zstd generation snapshot", conversation.full_text).into();
        }
        let started = std::time::Instant::now();
        publish_live(&lock, &conversations);
        eprintln!("delta publish {:.2?}", started.elapsed());
        let generation = crate::store::generation::Generation::open(root.path()).unwrap();
        eprintln!("segments {:?}", generation.segs());
        assert_eq!(generation.segs().len(), 2);
        check_generation(root.path(), &conversations, REAL_QUERIES, now);
    }

    /// Counts the text bytes handed out.
    struct CountingTexts<'a> {
        texts: &'a dyn TextSource,
        bytes: std::sync::atomic::AtomicUsize,
    }

    impl TextSource for CountingTexts<'_> {
        fn full_text(&self, doc: u32) -> Cow<'_, str> {
            let text = self.texts.full_text(doc);
            self.bytes
                .fetch_add(text.len(), std::sync::atomic::Ordering::Relaxed);
            text
        }
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        sorted[((sorted.len() - 1) as f64 * p).round() as usize]
    }

    /// Times index queries (first page of 50, and the complete ranking)
    /// against the in-memory search on the user's conversations, with text
    /// decompressed from zstd frames. Run with
    /// `cargo test --release --bin claude-history real_corpus_query_latency -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_corpus_query_latency() {
        const PAGE: usize = 50;
        const RUNS: usize = 5;
        let corpus = real_corpus();
        let now = Local::now();
        let median = |mut samples: Vec<f64>| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        let mut page_ms = Vec::new();
        let mut complete_ms = Vec::new();
        let mut scan_ms = Vec::new();
        let mut page_reads = Vec::new();
        let mut complete_reads = Vec::new();
        for query in REAL_QUERIES {
            let texts = CountingTexts {
                texts: &corpus.frames,
                bytes: Default::default(),
            };
            let run = |complete: bool| {
                let started = std::time::Instant::now();
                let mut ranking =
                    search(&corpus.index, &corpus.docs, &texts, query, now, &|_| true);
                let hits = if complete {
                    ranking.complete().len()
                } else {
                    ranking.top(PAGE).len()
                };
                (started.elapsed().as_secs_f64() * 1e3, ranking.stats(), hits)
            };
            run(false);
            texts.bytes.store(0, std::sync::atomic::Ordering::Relaxed);
            let (_, page_stats, _) = run(false);
            let page_bytes = texts.bytes.load(std::sync::atomic::Ordering::Relaxed);
            let page = median((0..RUNS).map(|_| run(false).0).collect());
            texts.bytes.store(0, std::sync::atomic::Ordering::Relaxed);
            let (_, complete_stats, hits) = run(true);
            let complete_bytes = texts.bytes.load(std::sync::atomic::Ordering::Relaxed);
            let complete = median((0..RUNS).map(|_| run(true).0).collect());
            let scan = median(
                (0..3)
                    .map(|_| {
                        let started = std::time::Instant::now();
                        crate::search::search(
                            &corpus.conversations,
                            &corpus.searchable,
                            query,
                            now,
                        );
                        started.elapsed().as_secs_f64() * 1e3
                    })
                    .collect(),
            );
            eprintln!(
                "{query:?}: {hits} hits, {} candidates | page {page:.2} ms, {} reads \
                 ({:.1} MB) | complete {complete:.2} ms, {} reads ({:.1} MB) | in-memory \
                 {scan:.1} ms",
                page_stats.candidates,
                page_stats.texts_read,
                page_bytes as f64 / 1e6,
                complete_stats.texts_read,
                complete_bytes as f64 / 1e6,
            );
            page_ms.push(page);
            complete_ms.push(complete);
            scan_ms.push(scan);
            page_reads.push(page_stats.texts_read as f64);
            complete_reads.push(complete_stats.texts_read as f64);
        }
        for (name, mut values) in [
            ("page ms", page_ms),
            ("complete ms", complete_ms),
            ("in-memory ms", scan_ms),
            ("page reads", page_reads),
            ("complete reads", complete_reads),
        ] {
            values.sort_by(f64::total_cmp);
            eprintln!(
                "{name}: p50 {:.2} p95 {:.2} max {:.2}",
                percentile(&values, 0.5),
                percentile(&values, 0.95),
                values[values.len() - 1]
            );
        }
    }
}

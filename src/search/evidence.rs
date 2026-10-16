use crate::search::literal::Literal;
use crate::search::query::ParsedQuery;
use crate::text_match::{NormalizedText, contains_cjk, normalize_for_search, word_match_ranges};

/// Where a conversation's text matches a query beyond its preview.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LexicalEvidence {
    /// The text around the ranges: windows of `CONTEXT_WINDOW_CHARS` chars on
    /// each side of every range, cut from the conversation text, with each
    /// left-out stretch replaced by `ELIDED`. Any context a list row shows
    /// around a range is the same as in the whole text.
    pub text: String,
    /// Byte ranges of the matches in `text`, in order.
    pub context_ranges: Vec<(usize, usize)>,
}

/// Characters kept on each side of a context range, more than half of any
/// list row.
const CONTEXT_WINDOW_CHARS: usize = 1024;
/// Stands in for text left out of `LexicalEvidence::text`.
const ELIDED: &str = "\u{2026}";

#[derive(Clone, Copy, Debug)]
struct TermHit {
    start: usize,
    end: usize,
    term_idx: usize,
}

#[derive(Clone, Debug)]
struct HitCluster {
    start: usize,
    end: usize,
    unique_terms: u64,
    missing_terms: u64,
    adjacent_pairs: u32,
    last_hit_end: usize,
    last_term_idx: usize,
}

impl HitCluster {
    fn span(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    fn unique_count(&self) -> u32 {
        self.unique_terms.count_ones()
    }

    fn missing_count(&self) -> u32 {
        self.missing_terms.count_ones()
    }
}

#[derive(Clone)]
enum EvidenceSpec {
    /// A search-normalized query word, matching as keyword search counts it:
    /// at word starts, or `anywhere` when the query has CJK text.
    Normalized {
        term: String,
        anywhere: bool,
    },
    Literal(Literal),
}

impl EvidenceSpec {
    fn key(&self) -> (&str, bool) {
        match self {
            Self::Normalized { term, .. } => (term.as_str(), false),
            Self::Literal(literal) => (literal.text(), true),
        }
    }

    fn ranges(&self, text: &NormalizedText) -> Vec<(usize, usize)> {
        match self {
            Self::Normalized { term, anywhere } => {
                text.raw_ranges(&word_match_ranges(text.normalized(), term, *anywhere))
            }
            Self::Literal(literal) => literal.match_ranges(text.raw()),
        }
    }
}

/// Evidence that `full_text` matches `parsed` beyond `preview`, or `None`
/// when the preview already shows every match.
pub fn build_lexical_evidence(
    full_text: &str,
    preview: &str,
    parsed: &ParsedQuery,
) -> Option<LexicalEvidence> {
    let specs = evidence_specs(parsed);
    if specs.is_empty() {
        return None;
    }

    let ranges = select_context_ranges(full_text, preview, &specs)?;
    Some(excerpt(full_text, &ranges))
}

/// The ranges `build_lexical_evidence` selects, in `full_text`.
#[cfg(test)]
pub(crate) fn full_text_context_ranges(
    full_text: &str,
    preview: &str,
    parsed: &ParsedQuery,
) -> Option<Vec<(usize, usize)>> {
    select_context_ranges(full_text, preview, &evidence_specs(parsed))
}

/// Cuts the windows around `ranges` out of `full_text`.
fn excerpt(full_text: &str, ranges: &[(usize, usize)]) -> LexicalEvidence {
    let back = |at: usize| {
        full_text[..at]
            .char_indices()
            .rev()
            .nth(CONTEXT_WINDOW_CHARS - 1)
            .map_or(0, |(index, _)| index)
    };
    let forward = |at: usize| {
        full_text[at..]
            .char_indices()
            .nth(CONTEXT_WINDOW_CHARS)
            .map_or(full_text.len(), |(index, _)| at + index)
    };
    // Windows that overlap or touch are merged, so text between two of them
    // is always left out.
    let mut windows: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in ranges {
        let (from, to) = (back(start), forward(end));
        match windows.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => windows.push((from, to)),
        }
    }

    let mut text = String::new();
    let mut context_ranges = Vec::with_capacity(ranges.len());
    let mut copied_to = 0;
    let mut ranges = ranges.iter().peekable();
    for (from, to) in windows {
        if from > copied_to {
            text.push_str(ELIDED);
        }
        let offset = text.len();
        text.push_str(&full_text[from..to]);
        while let Some(&&(start, end)) = ranges.peek()
            && end <= to
        {
            context_ranges.push((start - from + offset, end - from + offset));
            ranges.next();
        }
        copied_to = to;
    }
    if copied_to < full_text.len() {
        text.push_str(ELIDED);
    }
    LexicalEvidence {
        text,
        context_ranges,
    }
}

fn evidence_specs(parsed: &ParsedQuery) -> Vec<EvidenceSpec> {
    let unquoted = parsed.unquoted();
    let unquoted_terms = unquoted.split_whitespace().collect::<Vec<_>>();
    let identifier_literals = unquoted_terms
        .iter()
        .copied()
        .filter(|term| term.contains('_'))
        .map(|term| EvidenceSpec::Literal(Literal::new(term.to_string())));
    let normalized_query = unquoted_terms
        .iter()
        .copied()
        .filter(|term| !term.contains('_'))
        .collect::<Vec<_>>()
        .join(" ");
    let normalized_query = normalize_for_search(&normalized_query);
    let anywhere = contains_cjk(&normalized_query);
    let normalized = normalized_query
        .split_whitespace()
        .map(|term| EvidenceSpec::Normalized {
            term: term.to_string(),
            anywhere,
        })
        .collect::<Vec<_>>();

    dedupe_specs(
        &normalized
            .into_iter()
            .chain(identifier_literals)
            .chain(parsed.literals().iter().cloned().map(EvidenceSpec::Literal))
            .collect::<Vec<_>>(),
    )
}

fn dedupe_specs(specs: &[EvidenceSpec]) -> Vec<EvidenceSpec> {
    let mut deduped = Vec::new();
    for spec in specs {
        let (text, is_literal) = spec.key();
        if !deduped.iter().any(|existing: &EvidenceSpec| {
            let (existing_text, existing_is_literal) = existing.key();
            existing_is_literal == is_literal && existing_text.eq_ignore_ascii_case(text)
        }) {
            deduped.push(spec.clone());
            if deduped.len() == 64 {
                break;
            }
        }
    }
    deduped
}

fn select_context_ranges(
    full_text: &str,
    preview: &str,
    specs: &[EvidenceSpec],
) -> Option<Vec<(usize, usize)>> {
    let preview = NormalizedText::new(preview);
    let full_text = NormalizedText::new(full_text);
    let mut missing_mask: u64 = 0;
    let mut missing_count = 0u32;
    for (i, spec) in specs.iter().enumerate() {
        if spec.ranges(&preview).is_empty() {
            missing_mask |= 1 << i;
            missing_count += 1;
        }
    }

    let all_hits = find_all_spec_hits(&full_text, specs);
    if all_hits.is_empty() {
        return None;
    }

    if missing_count == 0 {
        let preview_hit_count = find_all_spec_hits(&preview, specs).len();
        if all_hits.len() <= preview_hit_count {
            return None;
        }
    }

    let hit_spans: Vec<(usize, usize, usize)> = all_hits
        .iter()
        .map(|hit| (hit.start, hit.end, hit.term_idx))
        .collect();
    select_hidden_context_ranges(full_text.raw(), &hit_spans, missing_mask, missing_count)
}

pub(crate) fn select_hidden_context_ranges(
    full_text: &str,
    hits: &[(usize, usize, usize)],
    missing_mask: u64,
    missing_count: u32,
) -> Option<Vec<(usize, usize)>> {
    let merge_gap_bytes: usize = 50;
    let max_cluster_span_bytes: usize = 200;
    let max_clusters: usize = 3;

    if hits.is_empty() {
        return None;
    }

    let mut clusters: Vec<HitCluster> = Vec::new();
    for &(start, end, term_idx) in hits {
        let term_bit: u64 = 1u64 << term_idx;
        let is_missing = (missing_mask & term_bit) != 0;
        let mut extended = false;

        if let Some(last) = clusters.last_mut() {
            let close_enough = start <= last.end.saturating_add(merge_gap_bytes);
            let new_end = last.end.max(end);
            let new_span = new_end.saturating_sub(last.start);
            if close_enough && new_span <= max_cluster_span_bytes {
                if term_idx != last.last_term_idx && start >= last.last_hit_end {
                    let gap = &full_text[last.last_hit_end..start];
                    if !gap.is_empty() && gap.chars().all(|c| !c.is_alphanumeric()) {
                        last.adjacent_pairs += 1;
                    }
                }

                last.end = new_end;
                last.unique_terms |= term_bit;
                if is_missing {
                    last.missing_terms |= term_bit;
                }
                last.last_hit_end = end;
                last.last_term_idx = term_idx;
                extended = true;
            }
        }

        if !extended {
            clusters.push(HitCluster {
                start,
                end,
                unique_terms: term_bit,
                missing_terms: if is_missing { term_bit } else { 0 },
                adjacent_pairs: 0,
                last_hit_end: end,
                last_term_idx: term_idx,
            });
        }
    }

    if missing_count > 0 {
        clusters.retain(|cluster| cluster.missing_count() > 0);
    }
    if clusters.is_empty() {
        return None;
    }

    clusters.sort_unstable_by(|a, b| {
        b.missing_count()
            .cmp(&a.missing_count())
            .then_with(|| b.adjacent_pairs.cmp(&a.adjacent_pairs))
            .then_with(|| b.unique_count().cmp(&a.unique_count()))
            .then_with(|| a.span().cmp(&b.span()))
            .then_with(|| a.start.cmp(&b.start))
    });

    let mut selected: Vec<HitCluster> = Vec::new();
    let mut covered_missing: u64 = 0;

    for cluster in &clusters {
        if selected.len() >= max_clusters {
            break;
        }
        let new_missing = cluster.missing_terms & !covered_missing;
        if new_missing != 0 {
            covered_missing |= cluster.missing_terms;
            selected.push(cluster.clone());
        }
    }

    for cluster in &clusters {
        if selected.len() >= max_clusters {
            break;
        }
        if !selected
            .iter()
            .any(|selected| selected.start == cluster.start && selected.end == cluster.end)
        {
            selected.push(cluster.clone());
        }
    }

    selected.sort_unstable_by_key(|cluster| cluster.start);
    Some(
        selected
            .into_iter()
            .map(|cluster| (cluster.start, cluster.end))
            .collect(),
    )
}

fn find_all_spec_hits(text: &NormalizedText, specs: &[EvidenceSpec]) -> Vec<TermHit> {
    let mut all_hits = Vec::new();
    for (term_idx, spec) in specs.iter().enumerate() {
        all_hits.extend(spec.ranges(text).into_iter().map(|(start, end)| TermHit {
            start,
            end,
            term_idx,
        }));
    }
    all_hits.sort_unstable_by_key(|hit| hit.start);
    all_hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::Conversation;

    use chrono::{Local, TimeZone};
    use std::path::PathBuf;

    fn conversation(preview: &str, full_text: &str) -> Conversation {
        Conversation {
            source: crate::history::model::SourceKind::Claude,
            session_id: "session".to_owned(),
            path: PathBuf::from("/tmp/session.jsonl"),
            index: 0,
            timestamp: Local.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            preview: preview.to_string().into(),
            preview_first: preview.to_string().into(),
            preview_last: preview.to_string().into(),
            full_text: full_text.to_string().into(),
            own_text: "".into(),
            agent_search_text: "".into(),
            semantic_route_text: "".into(),
            semantic_turns: vec![full_text.to_string()].into(),
            semantic_turn_ranges: vec![crate::agent::refs::MessageRange::single(1)],
            project_name: None,
            project_path: None,
            cwd: None,
            message_count: 1,
            parse_errors: Vec::new(),
            summary: None,
            custom_title: None,
            model: None,
            total_tokens: 0,
            duration_minutes: None,
            provenance: Default::default(),
        }
    }

    #[test]
    fn unquoted_body_match_produces_context_range() {
        let parsed = ParsedQuery::parse("deepgram");
        let conversation = conversation(
            "unrelated preview",
            "unrelated preview followed by hidden deepgram evidence",
        );

        let evidence =
            build_lexical_evidence(&conversation.full_text, &conversation.preview, &parsed)
                .unwrap();

        assert_eq!(evidence.context_ranges.len(), 1);
        let (start, end) = evidence.context_ranges[0];
        assert!(evidence.text[start..end].contains("deepgram"));
    }

    #[test]
    fn evidence_shows_every_match_keyword_search_counts() {
        let snippets = |query: &str, full_text: &str| {
            let evidence =
                build_lexical_evidence(full_text, "unrelated preview", &ParsedQuery::parse(query))
                    .unwrap();
            evidence
                .context_ranges
                .iter()
                .map(|&(start, end)| evidence.text[start..end].to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            snippets("istanbul", "unrelated preview then the İstanbul trip"),
            ["İstanbul"]
        );
        assert_eq!(
            snippets("测试会话", "unrelated preview 中间这是测试会话内容"),
            ["测试会话"]
        );
        assert_eq!(
            snippets("strasse", "unrelated preview with Straße name"),
            ["Straße"]
        );
        assert_eq!(
            build_lexical_evidence(
                "unrelated preview asp.net",
                "unrelated preview",
                &ParsedQuery::parse(".net")
            ),
            None
        );
    }

    #[test]
    fn quoted_literal_preserves_exact_punctuation() {
        let parsed = ParsedQuery::parse("\"audio_generation\"");
        let conversation = conversation(
            "audio generation normalized preview",
            "audio generation normalized preview and exact audio_generation evidence",
        );

        let evidence =
            build_lexical_evidence(&conversation.full_text, &conversation.preview, &parsed)
                .unwrap();

        let (start, end) = evidence.context_ranges[0];
        assert_eq!(&evidence.text[start..end], "audio_generation");
    }

    #[test]
    fn unquoted_identifier_evidence_preserves_exact_punctuation() {
        let parsed = ParsedQuery::parse("api_key");
        let conversation = conversation(
            "api key normalized preview",
            "api key normalized preview and exact api_key evidence",
        );

        let evidence =
            build_lexical_evidence(&conversation.full_text, &conversation.preview, &parsed)
                .unwrap();

        let snippets = evidence
            .context_ranges
            .iter()
            .map(|(start, end)| &evidence.text[*start..*end])
            .collect::<Vec<_>>();
        assert!(snippets.iter().any(|snippet| snippet.contains("api_key")));
    }

    #[test]
    fn mixed_query_can_surface_unquoted_and_literal_ranges() {
        let parsed = ParsedQuery::parse("hidden_unquoted \"exact_literal\"");
        let full_text = format!("hidden_unquoted {} exact_literal", "x ".repeat(120));
        let conversation = conversation("visible preview", &full_text);

        let evidence =
            build_lexical_evidence(&conversation.full_text, &conversation.preview, &parsed)
                .unwrap();
        let snippets = evidence
            .context_ranges
            .iter()
            .map(|(start, end)| &evidence.text[*start..*end])
            .collect::<Vec<_>>();

        assert!(
            snippets
                .iter()
                .any(|snippet| snippet.contains("hidden_unquoted"))
        );
        assert!(
            snippets
                .iter()
                .any(|snippet| snippet.contains("exact_literal"))
        );
    }
}

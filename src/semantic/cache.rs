use crate::error::{AppError, Result};
use crate::semantic::embed::SemanticEmbedder;
use crate::semantic::matrix::{EmbeddingKey, EmbeddingMatrix, MatrixRow, MatrixSettings};
use crate::semantic::types::{
    CACHE_SCHEMA_VERSION, CachedChunk, ChunkConfig, DEFAULT_EMBEDDING_BATCH_SIZE, EmbeddedChunk,
    Embedding, EmbeddingCache, MODEL_NAME, SemanticChunk,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

pub(crate) const MAX_CACHE_ENTRIES: usize = 50_000;

#[cfg(test)]
pub fn embed_chunks_with_progress_and_save(
    embedder: &mut dyn SemanticEmbedder,
    chunks: Vec<SemanticChunk>,
    cache: &mut EmbeddingCache,
    cancellation: &crate::semantic::types::SemanticCancellationToken,
    progress: impl FnMut(usize, usize),
    save: impl FnMut(&EmbeddingCache),
) -> Result<Vec<EmbeddedChunk>> {
    embed_chunks_with_budget_and_save(embedder, chunks, cache, cancellation, None, progress, save)
}

pub fn embed_chunks_with_budget_and_save(
    embedder: &mut dyn SemanticEmbedder,
    chunks: Vec<SemanticChunk>,
    cache: &mut EmbeddingCache,
    cancellation: &crate::semantic::types::SemanticCancellationToken,
    max_new_embeddings: Option<usize>,
    mut progress: impl FnMut(usize, usize),
    mut save: impl FnMut(&EmbeddingCache),
) -> Result<Vec<EmbeddedChunk>> {
    const SAVE_INTERVAL: usize = 256;

    if cancellation.is_cancelled() {
        return Err(AppError::SemanticSearchCancelled);
    }

    let access = cache.access_counter.saturating_add(1);
    cache.access_counter = access;
    let mut embedded = Vec::with_capacity(chunks.len());
    let mut misses = Vec::<Vec<SemanticChunk>>::new();
    let mut miss_indices = HashMap::<EmbeddingKey, usize>::new();

    for chunk in chunks {
        if cancellation.is_cancelled() {
            return Err(AppError::SemanticSearchCancelled);
        }
        let key = embedding_cache_key(&chunk.text);
        if let Some(entry) = cache.entries.get_mut(&key) {
            entry.last_used = access;
            embedded.push(embedded_chunk(&chunk, entry.embedding.clone()));
        } else if let Some(index) = miss_indices.get(&key).copied() {
            misses[index].push(chunk);
        } else {
            miss_indices.insert(key, misses.len());
            misses.push(vec![chunk]);
        }
    }

    let total_misses = max_new_embeddings.map_or(misses.len(), |limit| misses.len().min(limit));
    misses.truncate(total_misses);
    let mut completed = 0;
    let mut last_saved = 0;
    for batch in misses.chunks(DEFAULT_EMBEDDING_BATCH_SIZE) {
        if cancellation.is_cancelled() {
            save_pending_cache(cache, completed, &mut last_saved, &mut save);
            return Err(AppError::SemanticSearchCancelled);
        }
        let texts = batch
            .iter()
            .map(|chunks| chunks[0].text.clone())
            .collect::<Vec<_>>();
        let embeddings = match embedder.embed_passages(&texts) {
            Ok(embeddings) => embeddings,
            Err(error) => {
                save_pending_cache(cache, completed, &mut last_saved, &mut save);
                return Err(error);
            }
        };
        if let Err(error) = validate_passage_embeddings(&texts, &embeddings) {
            save_pending_cache(cache, completed, &mut last_saved, &mut save);
            return Err(error);
        }

        for (chunks, embedding) in batch.iter().zip(embeddings) {
            let key = embedding_cache_key(&chunks[0].text);
            let embedding = Embedding::from(embedding);
            cache.entries.insert(
                key,
                CachedChunk {
                    embedding: embedding.clone(),
                    last_used: access,
                    protected: false,
                },
            );
            for chunk in chunks {
                embedded.push(embedded_chunk(chunk, embedding.clone()));
            }
        }
        completed += batch.len();
        if completed == total_misses || completed - last_saved >= SAVE_INTERVAL {
            prune_cache(cache);
            save(cache);
            last_saved = completed;
        }
        progress(completed, total_misses);
    }

    Ok(embedded)
}

fn validate_passage_embeddings(texts: &[String], embeddings: &[Vec<f32>]) -> Result<()> {
    if embeddings.len() != texts.len() {
        return Err(AppError::ConfigError(format!(
            "semantic embedder returned {} passage embeddings for {} passages",
            embeddings.len(),
            texts.len()
        )));
    }
    let Some(dimensions) = embeddings.first().map(Vec::len) else {
        return Ok(());
    };
    if dimensions == 0
        || embeddings.iter().any(|embedding| {
            embedding.len() != dimensions || embedding.iter().any(|value| !value.is_finite())
        })
    {
        return Err(AppError::ConfigError(
            "semantic embedder returned empty, inconsistent, or non-finite passage embeddings"
                .to_string(),
        ));
    }
    Ok(())
}

fn save_pending_cache(
    cache: &mut EmbeddingCache,
    completed: usize,
    last_saved: &mut usize,
    save: &mut impl FnMut(&EmbeddingCache),
) {
    if completed > *last_saved {
        prune_cache(cache);
        save(cache);
        *last_saved = completed;
    }
}

fn prune_cache(cache: &mut EmbeddingCache) {
    prune_cache_to_limit(cache, MAX_CACHE_ENTRIES);
}

fn prune_cache_to_limit(cache: &mut EmbeddingCache, max_entries: usize) {
    let excess = cache.entries.len().saturating_sub(max_entries);
    if excess == 0 {
        return;
    }
    let mut oldest = cache
        .entries
        .iter()
        .map(|(key, entry)| (entry.protected, entry.last_used, *key))
        .collect::<Vec<_>>();
    oldest.sort_unstable();
    for (_, _, key) in oldest.into_iter().take(excess) {
        cache.entries.remove(&key);
    }
}

fn embedded_chunk(chunk: &SemanticChunk, embedding: Embedding) -> EmbeddedChunk {
    EmbeddedChunk {
        conversation_index: chunk.conversation_index,
        source: chunk.source,
        session: chunk.session.clone(),
        chunk_index: chunk.chunk_index,
        text: chunk.text.clone(),
        message_range: chunk.message_range,
        embedding,
    }
}

pub fn cached_chunks(
    chunks: Vec<SemanticChunk>,
    cache: &EmbeddingCache,
    cancellation: &crate::semantic::types::SemanticCancellationToken,
) -> Result<(Vec<EmbeddedChunk>, usize)> {
    let mut embedded = Vec::with_capacity(chunks.len());
    let mut misses = HashSet::new();

    for chunk in chunks {
        if cancellation.is_cancelled() {
            return Err(AppError::SemanticSearchCancelled);
        }
        let key = embedding_cache_key(&chunk.text);
        if let Some(entry) = cache.entries.get(&key) {
            embedded.push(embedded_chunk(&chunk, entry.embedding.clone()));
        } else {
            misses.insert(key);
        }
    }

    Ok((embedded, misses.len()))
}

pub fn cache_miss_count(chunks: &[SemanticChunk], cache: &EmbeddingCache) -> usize {
    chunks
        .iter()
        .map(|chunk| embedding_cache_key(&chunk.text))
        .filter(|key| !cache.entries.contains_key(key))
        .collect::<HashSet<_>>()
        .len()
}

pub fn embedding_cache_key(text: &str) -> EmbeddingKey {
    EmbeddingKey::of(text)
}

pub fn cache_contains_text(cache: &EmbeddingCache, text: &str) -> bool {
    cache.entries.contains_key(&embedding_cache_key(text))
}

pub fn read_embedding_cache(config: ChunkConfig) -> EmbeddingCache {
    let Some(dir) = semantic_cache_dir() else {
        return empty_embedding_cache(config);
    };
    read_embedding_cache_in(&dir, config)
}

/// Reads the embedding matrix in `dir`. When there is none yet, converts
/// `embeddings-v1.bin`, the serialized map earlier versions kept, into one
/// and removes it, so its embeddings are kept.
fn read_embedding_cache_in(dir: &Path, config: ChunkConfig) -> EmbeddingCache {
    let path = dir.join(MATRIX_FILE);
    if let Some(cache) = read_matrix(&path, config) {
        if cache.entries.len() > MAX_CACHE_ENTRIES {
            let mut cache = cache;
            prune_cache(&mut cache);
            write_embedding_cache_in(&cache, dir);
            return cache;
        }
        return cache;
    }
    let legacy = dir.join(LEGACY_FILE);
    if !legacy.exists() {
        return empty_embedding_cache(config);
    }
    let Ok(_lock) = lock_cache_dir(dir) else {
        return empty_embedding_cache(config);
    };
    if let Some(cache) = read_matrix(&path, config) {
        return cache;
    }
    let Some(mut cache) = read_legacy(&legacy, config) else {
        return empty_embedding_cache(config);
    };
    prune_cache(&mut cache);
    if write_matrix(&cache, &path).is_ok() {
        let _ = std::fs::remove_file(&legacy);
        remove_stale_temp_files(dir);
    }
    read_matrix(&path, config).unwrap_or(cache)
}

fn read_matrix(path: &Path, config: ChunkConfig) -> Option<EmbeddingCache> {
    let matrix = Arc::new(EmbeddingMatrix::open(path)?);
    if matrix.settings() != matrix_settings(config) {
        return None;
    }
    let mut cache = empty_embedding_cache(config);
    cache.access_counter = matrix.access_counter();
    cache.entries = mapped_entries(&matrix).collect();
    Some(cache)
}

fn mapped_entries(
    matrix: &Arc<EmbeddingMatrix>,
) -> impl Iterator<Item = (EmbeddingKey, CachedChunk)> + '_ {
    matrix.keys().iter().enumerate().map(|(row, key)| {
        let row = row as u32;
        (
            *key,
            CachedChunk {
                embedding: Embedding::Mapped(matrix.clone(), row),
                last_used: matrix.last_used(row),
                protected: matrix.protected(row),
            },
        )
    })
}

/// `embeddings-v1.bin` as earlier versions serialized it.
#[derive(serde::Deserialize, serde::Serialize)]
struct LegacyCache {
    schema_version: u32,
    model: String,
    chunk_target_chars: usize,
    chunk_overlap_chars: usize,
    chunk_context_turns: usize,
    access_counter: u64,
    entries: HashMap<String, LegacyChunk>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct LegacyChunk {
    embedding: Vec<f32>,
    last_used: u64,
    protected: bool,
}

fn read_legacy(path: &Path, config: ChunkConfig) -> Option<EmbeddingCache> {
    let data = std::fs::read(path).ok()?;
    let legacy = bincode::deserialize::<LegacyCache>(&data).ok()?;
    let mut cache = empty_embedding_cache(config);
    let matches = legacy.schema_version == CACHE_SCHEMA_VERSION
        && legacy.model == MODEL_NAME
        && legacy.chunk_target_chars == config.target_chars
        && legacy.chunk_overlap_chars == config.overlap_chars
        && legacy.chunk_context_turns == config.context_turns;
    if !matches {
        return None;
    }
    cache.access_counter = legacy.access_counter;
    let dims = legacy
        .entries
        .values()
        .next()
        .map(|entry| entry.embedding.len());
    cache.entries = legacy
        .entries
        .into_iter()
        .filter(|(_, entry)| Some(entry.embedding.len()) == dims)
        .filter_map(|(key, entry)| {
            Some((
                EmbeddingKey::from_hex(&key)?,
                CachedChunk {
                    embedding: entry.embedding.into(),
                    last_used: entry.last_used,
                    protected: entry.protected,
                },
            ))
        })
        .collect();
    Some(cache)
}

pub fn write_embedding_cache(cache: &EmbeddingCache) {
    let Some(dir) = semantic_cache_dir() else {
        return;
    };
    write_embedding_cache_in(cache, &dir);
}

/// Writes `cache` as the matrix in `dir`. Embeddings another process wrote
/// since this one read the matrix are kept, then the result is pruned to
/// `MAX_CACHE_ENTRIES`.
fn write_embedding_cache_in(cache: &EmbeddingCache, dir: &Path) {
    let Ok(_lock) = lock_cache_dir(dir) else {
        return;
    };
    let path = dir.join(MATRIX_FILE);
    let config = ChunkConfig {
        target_chars: cache.chunk_target_chars,
        overlap_chars: cache.chunk_overlap_chars,
        context_turns: cache.chunk_context_turns,
    };
    let mut merged = EmbeddingCache {
        entries: cache.entries.clone(),
        ..empty_embedding_cache(config)
    };
    merged.access_counter = cache.access_counter;
    if let Some(disk) = EmbeddingMatrix::open(&path)
        .filter(|matrix| matrix.settings() == matrix_settings(config))
        .map(Arc::new)
    {
        merged.access_counter = merged.access_counter.max(disk.access_counter());
        for (key, entry) in mapped_entries(&disk) {
            merged.entries.entry(key).or_insert(entry);
        }
    }
    prune_cache(&mut merged);
    let _ = write_matrix(&merged, &path);
}

fn write_matrix(cache: &EmbeddingCache, path: &Path) -> std::io::Result<()> {
    let dims = cache
        .entries
        .values()
        .next()
        .map_or(0, |entry| entry.embedding.len());
    let rows = cache
        .entries
        .iter()
        .map(|(key, entry)| MatrixRow {
            key: *key,
            last_used: entry.last_used,
            protected: entry.protected,
            vector: &entry.embedding,
        })
        .collect();
    let settings = matrix_settings(ChunkConfig {
        target_chars: cache.chunk_target_chars,
        overlap_chars: cache.chunk_overlap_chars,
        context_turns: cache.chunk_context_turns,
    });
    EmbeddingMatrix::write(path, settings, cache.access_counter, dims, rows)
}

fn matrix_settings(config: ChunkConfig) -> MatrixSettings {
    let model = format!("{MODEL_NAME}\0{CACHE_SCHEMA_VERSION}");
    let hash = blake3::hash(model.as_bytes());
    MatrixSettings {
        model_hash: u64::from_le_bytes(hash.as_bytes()[..8].try_into().expect("8 bytes")),
        chunk_target_chars: config.target_chars as u32,
        chunk_overlap_chars: config.overlap_chars as u32,
        chunk_context_turns: config.context_turns as u32,
        _pad: 0,
    }
}

/// Serializes writers of the cache directory across processes.
fn lock_cache_dir(dir: &Path) -> std::io::Result<std::fs::File> {
    std::fs::create_dir_all(dir)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join("LOCK"))?;
    file.lock()?;
    Ok(file)
}

/// Removes temporary files that writers in `dir` left behind when they
/// stopped before renaming them into place. Callers hold the directory lock,
/// and live writers finish well within the age limit.
fn remove_stale_temp_files(dir: &Path) {
    const STALE_AFTER: Duration = Duration::from_secs(60 * 60);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let is_temp = entry.file_name().to_string_lossy().starts_with(".tmp");
        let stale = entry
            .metadata()
            .ok()
            .filter(|metadata| metadata.is_file())
            .and_then(|metadata| metadata.modified().ok())
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > STALE_AFTER);
        if is_temp && stale {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

pub fn embedding_cache_file_path() -> Option<PathBuf> {
    semantic_cache_dir().map(|dir| dir.join(MATRIX_FILE))
}

pub fn model_cache_dir() -> PathBuf {
    semantic_cache_dir_with_fallback().join("fastembed")
}

pub fn clear_semantic_cache_files() -> std::io::Result<bool> {
    let Some(path) = semantic_cache_dir() else {
        return Ok(false);
    };
    if !path.exists() {
        return Ok(false);
    }
    std::fs::remove_dir_all(path)?;
    Ok(true)
}

pub fn empty_embedding_cache(config: ChunkConfig) -> EmbeddingCache {
    EmbeddingCache {
        chunk_target_chars: config.target_chars,
        chunk_overlap_chars: config.overlap_chars,
        chunk_context_turns: config.context_turns,
        access_counter: 0,
        entries: HashMap::new(),
    }
}

const MATRIX_FILE: &str = "embeddings-v2.bin";
const LEGACY_FILE: &str = "embeddings-v1.bin";

/// The semantic cache directory, which tests never use.
fn semantic_cache_dir() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    home::home_dir().map(semantic_cache_dir_in)
}

fn semantic_cache_dir_with_fallback() -> PathBuf {
    semantic_cache_dir_in(home::home_dir().unwrap_or_else(|| PathBuf::from(".")))
}

fn semantic_cache_dir_in(home: PathBuf) -> PathBuf {
    home.join(".cache").join("claude-history").join("semantic")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::types::SemanticCancellationToken;

    struct FakeEmbedder {
        calls: usize,
    }

    impl SemanticEmbedder for FakeEmbedder {
        fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls += 1;
            Ok(passages
                .iter()
                .map(|text| vec![text.len() as f32, 1.0])
                .collect())
        }

        fn embed_query(&mut self, _query: &str) -> Result<Option<Vec<f32>>> {
            Ok(Some(vec![1.0, 0.0]))
        }
    }

    struct CancellingEmbedder {
        cancellation: SemanticCancellationToken,
    }

    impl SemanticEmbedder for CancellingEmbedder {
        fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
            self.cancellation.cancel();
            Ok(passages.iter().map(|_| vec![1.0, 0.0]).collect())
        }

        fn embed_query(&mut self, _query: &str) -> Result<Option<Vec<f32>>> {
            Ok(Some(vec![1.0, 0.0]))
        }
    }

    fn chunk(key: &str, text: &str) -> SemanticChunk {
        let chunk_index = key
            .rsplit_once(':')
            .and_then(|(_, index)| index.parse::<usize>().ok())
            .unwrap_or(0);
        SemanticChunk {
            conversation_index: 0,
            source: crate::semantic::types::SemanticChunkSource::VisibleDialogue,
            session: "session".to_string(),
            chunk_index,
            text: text.to_string(),
            message_range: crate::agent::refs::MessageRange::single(1),
        }
    }

    fn cached() -> CachedChunk {
        CachedChunk {
            embedding: vec![0.5, 0.5].into(),
            last_used: 0,
            protected: false,
        }
    }

    fn cache_text(cache: &mut EmbeddingCache, text: &str) {
        cache.entries.insert(embedding_cache_key(text), cached());
    }

    #[test]
    fn passage_embedding_validation_rejects_invalid_output() {
        let texts = vec!["alpha".to_string(), "beta".to_string()];

        assert!(validate_passage_embeddings(&texts, &[vec![1.0, 0.0]]).is_err());
        assert!(validate_passage_embeddings(&texts, &[vec![1.0], vec![1.0, 0.0]]).is_err());
        assert!(validate_passage_embeddings(&texts, &[vec![f32::NAN], vec![1.0]]).is_err());
    }

    #[test]
    fn cache_pruning_keeps_most_recent_entries() {
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        for (text, last_used) in [("oldest", 1), ("middle", 2), ("newest", 3)] {
            cache.entries.insert(
                embedding_cache_key(text),
                CachedChunk {
                    embedding: vec![last_used as f32].into(),
                    last_used,
                    protected: false,
                },
            );
        }

        prune_cache_to_limit(&mut cache, 2);

        assert_eq!(cache.entries.len(), 2);
        assert!(!cache_contains_text(&cache, "oldest"));
        assert!(cache_contains_text(&cache, "middle"));
        assert!(cache_contains_text(&cache, "newest"));
    }

    #[test]
    fn cache_pruning_preserves_protected_core_entries() {
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        cache_text(&mut cache, "adaptive newest");
        cache
            .entries
            .get_mut(&embedding_cache_key("adaptive newest"))
            .unwrap()
            .last_used = 10;
        cache_text(&mut cache, "core oldest");
        let core = cache
            .entries
            .get_mut(&embedding_cache_key("core oldest"))
            .unwrap();
        core.last_used = 1;
        core.protected = true;

        prune_cache_to_limit(&mut cache, 1);

        assert!(cache_contains_text(&cache, "core oldest"));
        assert!(!cache_contains_text(&cache, "adaptive newest"));
    }

    #[test]
    fn embed_chunks_respects_new_embedding_budget() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache_text(&mut cache, "cached text");
        let chunks = std::iter::once(chunk("session:0", "cached text"))
            .chain((1..=40).map(|index| {
                chunk(
                    &format!("session:{index}"),
                    &format!("missing text {index}"),
                )
            }))
            .collect::<Vec<_>>();
        let mut embedder = FakeEmbedder { calls: 0 };

        let embedded = embed_chunks_with_budget_and_save(
            &mut embedder,
            chunks,
            &mut cache,
            &SemanticCancellationToken::new(),
            Some(32),
            |_, _| {},
            |_| {},
        )
        .expect("embedding succeeds");

        assert_eq!(embedder.calls, 1);
        assert_eq!(embedded.len(), 33);
        assert_eq!(cache.entries.len(), 33);
        assert!(cache_contains_text(&cache, "missing text 32"));
        assert!(!cache_contains_text(&cache, "missing text 33"));
    }

    #[test]
    fn embed_chunks_reuses_matching_cache_entry() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache_text(&mut cache, "cached text");
        let mut embedder = FakeEmbedder { calls: 0 };

        let embedded = embed_chunks_with_progress_and_save(
            &mut embedder,
            vec![chunk("session:0", "cached text")],
            &mut cache,
            &SemanticCancellationToken::new(),
            |_, _| {},
            |_| {},
        )
        .expect("embedding succeeds");

        assert_eq!(embedder.calls, 0);
        assert_eq!(embedded[0].embedding, vec![0.5, 0.5]);
        assert_eq!(embedded[0].chunk_index, 0);
    }

    #[test]
    fn embed_chunks_embeds_cache_misses() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        let mut embedder = FakeEmbedder { calls: 0 };

        let embedded = embed_chunks_with_progress_and_save(
            &mut embedder,
            vec![chunk("session:0", "new text")],
            &mut cache,
            &SemanticCancellationToken::new(),
            |_, _| {},
            |_| {},
        )
        .expect("embedding succeeds");

        assert_eq!(embedder.calls, 1);
        assert_eq!(embedded[0].embedding, vec![8.0, 1.0]);
        assert_eq!(embedded[0].chunk_index, 0);
        assert!(cache.entries.contains_key(&embedding_cache_key("new text")));
    }

    #[test]
    fn embed_chunks_reuses_content_across_chunk_keys() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache_text(&mut cache, "shared text");
        let mut embedder = FakeEmbedder { calls: 0 };

        let embedded = embed_chunks_with_progress_and_save(
            &mut embedder,
            vec![
                chunk("first-session:0", "shared text"),
                chunk("second-session:4", "shared text"),
            ],
            &mut cache,
            &SemanticCancellationToken::new(),
            |_, _| {},
            |_| {},
        )
        .expect("embedding succeeds");

        assert_eq!(embedder.calls, 0);
        assert_eq!(embedded.len(), 2);
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn embed_chunks_embeds_duplicate_misses_once() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        let mut embedder = FakeEmbedder { calls: 0 };
        let mut progress = Vec::new();

        let embedded = embed_chunks_with_progress_and_save(
            &mut embedder,
            vec![
                chunk("first-session:0", "duplicate text"),
                chunk("second-session:0", "duplicate text"),
            ],
            &mut cache,
            &SemanticCancellationToken::new(),
            |completed, total| progress.push((completed, total)),
            |_| {},
        )
        .expect("embedding succeeds");

        assert_eq!(embedder.calls, 1);
        assert_eq!(embedded.len(), 2);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(progress, vec![(1, 1)]);
    }

    #[test]
    fn embed_chunks_checkpoints_large_runs_and_saves_the_final_batch() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        let mut embedder = FakeEmbedder { calls: 0 };
        let chunks = (0..300)
            .map(|index| chunk(&format!("session:{index}"), &format!("text {index}")))
            .collect();
        let mut saved_entry_counts = Vec::new();

        embed_chunks_with_progress_and_save(
            &mut embedder,
            chunks,
            &mut cache,
            &SemanticCancellationToken::new(),
            |_, _| {},
            |cache| saved_entry_counts.push(cache.entries.len()),
        )
        .expect("embedding succeeds");

        assert_eq!(embedder.calls, 10);
        assert_eq!(saved_entry_counts, vec![256, 300]);
    }

    #[test]
    fn embed_chunks_saves_completed_work_when_cancelled() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        let cancellation = SemanticCancellationToken::new();
        let mut embedder = CancellingEmbedder {
            cancellation: cancellation.clone(),
        };
        let chunks = (0..33)
            .map(|index| chunk(&format!("session:{index}"), &format!("text {index}")))
            .collect();
        let mut saved_entry_counts = Vec::new();

        let result = embed_chunks_with_progress_and_save(
            &mut embedder,
            chunks,
            &mut cache,
            &cancellation,
            |_, _| {},
            |cache| saved_entry_counts.push(cache.entries.len()),
        );

        assert!(matches!(result, Err(AppError::SemanticSearchCancelled)));
        assert_eq!(saved_entry_counts, vec![32]);
    }

    #[test]
    fn changed_text_creates_a_distinct_cache_entry() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache_text(&mut cache, "old text");
        let mut embedder = FakeEmbedder { calls: 0 };

        let embedded = embed_chunks_with_progress_and_save(
            &mut embedder,
            vec![chunk("session:0", "new text")],
            &mut cache,
            &SemanticCancellationToken::new(),
            |_, _| {},
            |_| {},
        )
        .expect("embedding succeeds");

        let restored = cache
            .entries
            .get(&embedding_cache_key("new text"))
            .expect("cache entry");
        assert_eq!(embedder.calls, 1);
        assert_eq!(embedded[0].embedding, vec![8.0, 1.0]);
        assert_eq!(restored.embedding, vec![8.0, 1.0]);
        assert!(cache_contains_text(&cache, "old text"));
    }

    fn legacy_cache(config: ChunkConfig, texts: &[&str]) -> LegacyCache {
        LegacyCache {
            schema_version: CACHE_SCHEMA_VERSION,
            model: MODEL_NAME.to_string(),
            chunk_target_chars: config.target_chars,
            chunk_overlap_chars: config.overlap_chars,
            chunk_context_turns: config.context_turns,
            access_counter: 9,
            entries: texts
                .iter()
                .enumerate()
                .map(|(index, text)| {
                    (
                        blake3::hash(text.as_bytes()).to_hex().to_string(),
                        LegacyChunk {
                            embedding: vec![index as f32, 1.0],
                            last_used: index as u64,
                            protected: index == 0,
                        },
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn legacy_cache_migrates_to_the_matrix() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ChunkConfig::default();
        let legacy = legacy_cache(config, &["first", "second"]);
        std::fs::write(
            dir.path().join(LEGACY_FILE),
            bincode::serialize(&legacy).unwrap(),
        )
        .unwrap();
        let stale = dir.path().join(".tmpstale");
        std::fs::write(&stale, b"partial").unwrap();
        let old = SystemTime::now() - Duration::from_secs(2 * 60 * 60);
        std::fs::File::options()
            .write(true)
            .open(&stale)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let fresh = dir.path().join(".tmpfresh");
        std::fs::write(&fresh, b"in progress").unwrap();

        let restored = read_embedding_cache_in(dir.path(), config);

        assert!(!dir.path().join(LEGACY_FILE).exists());
        assert!(dir.path().join(MATRIX_FILE).exists());
        assert!(!stale.exists());
        assert!(fresh.exists());
        assert_eq!(restored.access_counter, 9);
        let second = &restored.entries[&embedding_cache_key("second")];
        assert!(matches!(second.embedding, Embedding::Mapped(..)));
        assert_eq!(second.embedding, vec![1.0, 1.0]);
        assert_eq!((second.last_used, second.protected), (1, false));
        let first = &restored.entries[&embedding_cache_key("first")];
        assert_eq!(first.embedding, vec![0.0, 1.0]);
        assert!(first.protected);

        let reread = read_embedding_cache_in(dir.path(), config);
        assert_eq!(reread.entries, restored.entries);
    }

    #[test]
    fn legacy_cache_with_another_schema_is_not_migrated() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ChunkConfig::default();
        let mut legacy = legacy_cache(config, &["stale"]);
        legacy.schema_version -= 1;
        std::fs::write(
            dir.path().join(LEGACY_FILE),
            bincode::serialize(&legacy).unwrap(),
        )
        .unwrap();

        let restored = read_embedding_cache_in(dir.path(), config);

        assert!(restored.entries.is_empty());
        assert!(!dir.path().join(MATRIX_FILE).exists());
    }

    #[test]
    fn previous_cache_layout_is_ignored() {
        #[derive(serde::Serialize)]
        struct PreviousCache {
            schema_version: u32,
            model: String,
            chunk_target_chars: usize,
            chunk_overlap_chars: usize,
            chunk_context_turns: usize,
            entries: HashMap<String, PreviousEntry>,
        }

        #[derive(serde::Serialize)]
        struct PreviousEntry {
            file_size: u64,
            mtime_secs: u64,
            mtime_nsecs: u32,
            text: String,
            embedding: Vec<f32>,
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let config = ChunkConfig::default();
        let previous = PreviousCache {
            schema_version: CACHE_SCHEMA_VERSION - 1,
            model: MODEL_NAME.to_string(),
            chunk_target_chars: config.target_chars,
            chunk_overlap_chars: config.overlap_chars,
            chunk_context_turns: config.context_turns,
            entries: HashMap::from([(
                "session:0".to_string(),
                PreviousEntry {
                    file_size: 10,
                    mtime_secs: 20,
                    mtime_nsecs: 30,
                    text: "cached text".to_string(),
                    embedding: vec![0.5, 0.5],
                },
            )]),
        };
        std::fs::write(
            dir.path().join(LEGACY_FILE),
            bincode::serialize(&previous).unwrap(),
        )
        .unwrap();

        let restored = read_embedding_cache_in(dir.path(), config);

        assert!(restored.entries.is_empty());
    }

    #[test]
    fn corrupt_cache_is_ignored() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ChunkConfig::default();
        std::fs::write(dir.path().join(MATRIX_FILE), b"not a matrix").unwrap();

        let restored = read_embedding_cache_in(dir.path(), config);

        assert!(restored.entries.is_empty());
    }

    #[test]
    fn mismatched_config_cache_is_ignored_when_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(ChunkConfig {
            overlap_chars: config.overlap_chars + 1,
            ..config
        });
        cache_text(&mut cache, "stale");
        write_embedding_cache_in(&cache, dir.path());

        let restored = read_embedding_cache_in(dir.path(), config);

        assert_eq!(restored.chunk_overlap_chars, config.overlap_chars);
        assert!(restored.entries.is_empty());
    }

    #[test]
    fn cache_round_trips_when_config_matches() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache_text(&mut cache, "cached text");

        write_embedding_cache_in(&cache, dir.path());
        let restored = read_embedding_cache_in(dir.path(), config);

        assert!(cache_contains_text(&restored, "cached text"));
        assert_eq!(restored.entries, cache.entries);
    }

    #[test]
    fn writes_keep_embeddings_another_process_saved() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ChunkConfig::default();
        let mut first = read_embedding_cache_in(dir.path(), config);
        let mut second = read_embedding_cache_in(dir.path(), config);
        cache_text(&mut second, "from second");
        write_embedding_cache_in(&second, dir.path());
        cache_text(&mut first, "from first");
        write_embedding_cache_in(&first, dir.path());

        let restored = read_embedding_cache_in(dir.path(), config);

        assert!(cache_contains_text(&restored, "from first"));
        assert!(cache_contains_text(&restored, "from second"));
    }

    #[test]
    fn tests_never_use_the_real_cache_dir() {
        assert_eq!(semantic_cache_dir(), None);
        assert_eq!(embedding_cache_file_path(), None);
    }

    #[test]
    fn cache_miss_count_matches_cached_chunks() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache_text(&mut cache, "cached text");
        cache_text(&mut cache, "old text");
        let chunks = vec![
            chunk("session:0", "cached text"),
            chunk("session:1", "new text"),
            chunk("session:2", "uncached text"),
        ];

        let (_, cached_chunk_misses) =
            cached_chunks(chunks.clone(), &cache, &SemanticCancellationToken::new()).unwrap();

        assert_eq!(cache_miss_count(&chunks, &cache), cached_chunk_misses);
        assert_eq!(cache_miss_count(&chunks, &cache), 2);
    }

    #[test]
    fn semantic_cache_paths_preserve_existing_locations() {
        let home = PathBuf::from("/home/example");
        let cache_dir = semantic_cache_dir_in(home);

        assert_eq!(
            cache_dir.join(MATRIX_FILE),
            PathBuf::from("/home/example/.cache/claude-history/semantic/embeddings-v2.bin")
        );
        assert_eq!(
            cache_dir.join("fastembed"),
            PathBuf::from("/home/example/.cache/claude-history/semantic/fastembed")
        );
    }
}

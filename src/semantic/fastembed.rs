use crate::error::{AppError, Result};
use crate::semantic::embed::SemanticEmbedder;
use crate::semantic::types::MODEL_BATCH_SIZE;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;

#[cfg(feature = "release-dynamic-ort")]
use std::{path::Path, sync::OnceLock};

pub struct FastembedEmbedder {
    model: TextEmbedding,
}

/// How much of the machine embedding may use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbedPriority {
    /// A command the person is waiting on: every core.
    Foreground,
    /// Indexing behind the TUI: half the cores at a lower scheduling
    /// priority, so the list and lexical search stay responsive.
    Background,
}

impl EmbedPriority {
    fn threads(self) -> usize {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        match self {
            Self::Foreground => cores,
            Self::Background => (cores / 2).max(1),
        }
    }
}

impl FastembedEmbedder {
    pub fn new() -> Result<Self> {
        Self::new_with_download_progress(
            crate::semantic::cache::model_cache_dir(),
            true,
            EmbedPriority::Foreground,
        )
    }

    /// An embedder for the calling thread that shows no download progress.
    /// With `EmbedPriority::Background` it also lowers the scheduling
    /// priority of the calling thread, which the inference threads it
    /// creates inherit.
    pub fn new_quiet(priority: EmbedPriority) -> Result<Self> {
        if priority == EmbedPriority::Background {
            lower_current_thread_priority();
        }
        Self::new_with_download_progress(crate::semantic::cache::model_cache_dir(), false, priority)
    }

    pub fn cache_dir() -> PathBuf {
        crate::semantic::cache::model_cache_dir()
    }

    fn new_with_download_progress(
        cache_dir: PathBuf,
        show_download_progress: bool,
        priority: EmbedPriority,
    ) -> Result<Self> {
        // Tests use fake embedders; the model lives in the user's cache.
        if cfg!(test) {
            return Err(AppError::ConfigError(
                "the embedding model is not loaded in tests".into(),
            ));
        }
        init_onnx_runtime()?;
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(show_download_progress)
                .with_intra_threads(priority.threads()),
        )
        .map_err(to_config_error)?;
        Ok(Self { model })
    }
}

impl SemanticEmbedder for FastembedEmbedder {
    /// Embeds `passages` in small batches of similar length: a batch is
    /// padded to its longest passage, so mixing lengths wastes inference on
    /// padding, and small batches keep the activation memory small.
    fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
        let order = length_order(passages);
        let sorted = order
            .iter()
            .map(|&index| passages[index].as_str())
            .collect::<Vec<_>>();
        let embeddings = self
            .model
            .embed(prefixed_passages(&sorted), Some(MODEL_BATCH_SIZE))
            .map_err(to_config_error)?;
        Ok(unsort(&order, embeddings))
    }

    fn embed_query(&mut self, query: &str) -> Result<Option<Vec<f32>>> {
        let embeddings = self
            .model
            .embed(vec![prefixed_query(query)], Some(MODEL_BATCH_SIZE))
            .map_err(to_config_error)?;
        Ok(embeddings.first().cloned())
    }
}

/// Indices of `passages` from shortest to longest.
fn length_order(passages: &[String]) -> Vec<usize> {
    let mut order = (0..passages.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| passages[index].len());
    order
}

/// Puts `values`, computed for the passages in `order`, back in passage
/// order.
fn unsort<T: Default>(order: &[usize], values: Vec<T>) -> Vec<T> {
    let mut out = (0..values.len()).map(|_| T::default()).collect::<Vec<_>>();
    for (&index, value) in order.iter().zip(values) {
        out[index] = value;
    }
    out
}

/// Lowers the scheduling priority of the calling thread: the utility QoS
/// class on macOS, nice 10 on Linux.
fn lower_current_thread_priority() {
    #[cfg(target_os = "macos")]
    // SAFETY: changes only the calling thread's QoS class.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0);
    }
    #[cfg(target_os = "linux")]
    // SAFETY: changes only the calling thread's nice value; Linux applies
    // PRIO_PROCESS with a thread id to that thread alone.
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, 10);
    }
}

pub fn prefixed_query(query: &str) -> String {
    format!("Represent this sentence for searching relevant passages: {query}")
}

pub fn prefixed_passages(passages: &[&str]) -> Vec<String> {
    passages
        .iter()
        .map(|passage| (*passage).to_owned())
        .collect()
}

#[cfg(feature = "release-dynamic-ort")]
fn init_onnx_runtime() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    match INIT.get_or_init(initialize_bundled_onnx_runtime) {
        Ok(()) => Ok(()),
        Err(message) => Err(AppError::ConfigError(message.clone())),
    }
}

#[cfg(not(feature = "release-dynamic-ort"))]
fn init_onnx_runtime() -> Result<()> {
    Ok(())
}

#[cfg(feature = "release-dynamic-ort")]
fn initialize_bundled_onnx_runtime() -> std::result::Result<(), String> {
    let path = bundled_onnx_runtime_path().ok_or_else(|| {
        format!(
            "Bundled ONNX Runtime library was not found beside the executable; expected a compatible {}",
            runtime_library_name()
        )
    })?;
    let builder = ort::init_from(&path).map_err(|error| {
        format!(
            "Failed to load bundled ONNX Runtime from {}: {error}",
            path.display()
        )
    })?;
    if !builder.commit() {
        return Err(format!(
            "Failed to initialize bundled ONNX Runtime from {}: an ONNX Runtime environment is already configured",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(feature = "release-dynamic-ort")]
fn bundled_onnx_runtime_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    bundled_onnx_runtime_path_in_dir(exe.parent()?)
}

#[cfg(feature = "release-dynamic-ort")]
fn bundled_onnx_runtime_path_in_dir(dir: &Path) -> Option<PathBuf> {
    onnx_runtime_candidates(dir)
        .into_iter()
        .find(|candidate| candidate.is_file())
}

#[cfg(feature = "release-dynamic-ort")]
fn onnx_runtime_candidates(dir: &Path) -> Vec<PathBuf> {
    let name = runtime_library_name();
    let lib_dir = dir.join("lib");
    let mut candidates = vec![dir.join(name), lib_dir.join(name)];

    for library_dir in [dir, lib_dir.as_path()] {
        let Ok(entries) = std::fs::read_dir(library_dir) else {
            continue;
        };
        let mut versioned = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| is_versioned_runtime_name(path, name))
            .collect::<Vec<_>>();
        versioned.sort();
        candidates.extend(versioned);
    }
    candidates
}

#[cfg(feature = "release-dynamic-ort")]
fn runtime_library_name() -> &'static str {
    match std::env::consts::OS {
        "macos" => "libonnxruntime.dylib",
        "windows" => "onnxruntime.dll",
        _ => "libonnxruntime.so",
    }
}

#[cfg(feature = "release-dynamic-ort")]
fn is_versioned_runtime_name(path: &Path, unversioned_name: &str) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    match unversioned_name {
        "libonnxruntime.dylib" => {
            file_name.starts_with("libonnxruntime.")
                && file_name.ends_with(".dylib")
                && file_name != unversioned_name
        }
        "onnxruntime.dll" => false,
        _ => file_name.starts_with("libonnxruntime.so.") && file_name != unversioned_name,
    }
}

fn to_config_error(err: impl std::fmt::Display) -> AppError {
    AppError::ConfigError(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_query_for_fastembed() {
        assert_eq!(
            prefixed_query("rust cache"),
            "Represent this sentence for searching relevant passages: rust cache"
        );
    }

    #[test]
    fn leaves_passages_unprefixed_for_fastembed() {
        assert_eq!(
            prefixed_passages(&["one", "two"]),
            vec!["one".to_string(), "two".to_string()]
        );
    }

    #[test]
    fn length_order_round_trips() {
        let passages = vec!["ccc".to_string(), "a".to_string(), "bb".to_string()];
        let order = length_order(&passages);
        assert_eq!(order, vec![1, 2, 0]);
        let computed = order.iter().map(|&index| passages[index].len()).collect();
        assert_eq!(unsort(&order, computed), vec![3, 1, 2]);
    }

    #[cfg(not(feature = "release-dynamic-ort"))]
    #[test]
    fn non_release_runtime_initialization_is_a_noop() {
        assert!(init_onnx_runtime().is_ok());
    }

    #[cfg(feature = "release-dynamic-ort")]
    #[test]
    fn runtime_path_uses_the_host_library_name() {
        let temp = tempfile::tempdir().unwrap();
        let lib_dir = temp.path().join("lib");
        std::fs::create_dir(&lib_dir).unwrap();
        std::fs::write(lib_dir.join("libonnxruntime.dylib"), b"wrong host").unwrap();
        std::fs::write(lib_dir.join(runtime_library_name()), b"runtime").unwrap();

        assert_eq!(
            bundled_onnx_runtime_path_in_dir(temp.path()),
            Some(lib_dir.join(runtime_library_name()))
        );
    }

    #[cfg(feature = "release-dynamic-ort")]
    #[test]
    fn runtime_path_falls_back_to_a_versioned_library() {
        let temp = tempfile::tempdir().unwrap();
        let lib_dir = temp.path().join("lib");
        std::fs::create_dir(&lib_dir).unwrap();
        let versioned_name = if runtime_library_name() == "libonnxruntime.dylib" {
            "libonnxruntime.1.24.2.dylib"
        } else {
            "libonnxruntime.so.1.24.2"
        };
        let versioned = lib_dir.join(versioned_name);
        std::fs::write(&versioned, b"runtime").unwrap();

        assert_eq!(
            bundled_onnx_runtime_path_in_dir(temp.path()),
            Some(versioned)
        );
    }
}

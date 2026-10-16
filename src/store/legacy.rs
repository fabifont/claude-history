//! Removal of the caches that older versions kept beside the store: one
//! file per Claude project in `projects/` (also inside each `config-<hash>/`)
//! and one per Pi or OMP session root in `pi/root-<hash>/` and
//! `omp/root-<hash>/`.
//!
//! Only files named and tagged as those caches are removed, with the
//! temporary files their writers left behind, and a directory only when that
//! leaves it empty. Nothing reads these files, and none of them is mapped, so
//! removing one under a running process is safe.

use std::fs;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, SystemTime};

const PROJECT_MAGIC: [u8; 8] = *b"CLHIST01";
const PI_MAGIC: [u8; 8] = *b"PIHIST01";
const OMP_MAGIC: [u8; 8] = *b"OMHIST01";
/// Age after which a temporary file can no longer belong to a running write.
const STALE_TEMP: Duration = Duration::from_secs(60 * 60);

/// What `remove_old_caches` removed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Removed {
    pub files: usize,
    pub bytes: u64,
    pub dirs: usize,
}

/// The cache dir holding the store at `root`, when `root` is a store root in
/// the default layout: `<cache>/v3/roots-<hash>` or
/// `<cache>/config-<hash>/v3/roots-<hash>`.
pub fn cache_dir_of(root: &Path) -> Option<&Path> {
    let name = root.file_name()?.to_str()?;
    if !name.starts_with("roots-") {
        return None;
    }
    let v3 = root.parent()?;
    if v3.file_name()? != "v3" {
        return None;
    }
    let base = v3.parent()?;
    match base.file_name()?.to_str() {
        Some(name) if is_hashed(name, "config-") => base.parent(),
        _ => Some(base),
    }
}

/// Removes the old caches under the cache dir `cache`.
pub fn remove_old_caches(cache: &Path) -> Removed {
    let mut removed = Removed::default();
    remove_project_caches(&cache.join("projects"), &mut removed);
    for entry in read_dir(cache) {
        let name = entry.file_name();
        if name.to_str().is_some_and(|name| is_hashed(name, "config-")) {
            let projects = entry.path().join("projects");
            remove_project_caches(&projects, &mut removed);
            remove_empty_dir(&entry.path(), &mut removed);
        }
    }
    for (source, magic) in [("pi", PI_MAGIC), ("omp", OMP_MAGIC)] {
        let dir = cache.join(source);
        for entry in read_dir(&dir) {
            let name = entry.file_name();
            if !name.to_str().is_some_and(|name| is_hashed(name, "root-")) {
                continue;
            }
            let root = entry.path();
            for file in read_dir(&root) {
                let path = file.path();
                if file.file_name() == "sessions.bin" && has_magic(&path, magic)
                    || is_stale_temp(&file)
                {
                    remove_file(&path, &mut removed);
                }
            }
            remove_empty_dir(&root, &mut removed);
        }
        remove_empty_dir(&dir, &mut removed);
    }
    removed
}

fn remove_project_caches(dir: &Path, removed: &mut Removed) {
    for file in read_dir(dir) {
        let path = file.path();
        let is_cache = path.extension().is_some_and(|ext| ext == "bin")
            && file.file_type().is_ok_and(|kind| kind.is_file())
            && has_magic(&path, PROJECT_MAGIC);
        if is_cache || is_stale_temp(&file) {
            remove_file(&path, removed);
        }
    }
    remove_empty_dir(dir, removed);
}

/// `name` is `prefix` followed by 16 lowercase hex digits.
fn is_hashed(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix).is_some_and(|hash| {
        hash.len() == 16 && hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    })
}

fn has_magic(path: &Path, magic: [u8; 8]) -> bool {
    let mut head = [0; 8];
    fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut head))
        .is_ok()
        && head == magic
}

/// A temporary file of an old cache write that no write can still own.
fn is_stale_temp(entry: &fs::DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|name| name.starts_with(".tmp"))
        && entry.metadata().is_ok_and(|metadata| {
            metadata.is_file()
                && metadata.modified().is_ok_and(|modified| {
                    SystemTime::now()
                        .duration_since(modified)
                        .is_ok_and(|age| age > STALE_TEMP)
                })
        })
}

fn read_dir(dir: &Path) -> impl Iterator<Item = fs::DirEntry> {
    fs::read_dir(dir).into_iter().flatten().flatten()
}

fn remove_file(path: &Path, removed: &mut Removed) {
    let bytes = fs::metadata(path).map_or(0, |metadata| metadata.len());
    if fs::remove_file(path).is_ok() {
        removed.files += 1;
        removed.bytes += bytes;
    }
}

/// Removes `dir` if it is empty; `remove_dir` refuses otherwise.
fn remove_empty_dir(dir: &Path, removed: &mut Removed) {
    if fs::remove_dir(dir).is_ok() {
        removed.dirs += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write(path: PathBuf, bytes: &[u8]) -> PathBuf {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }

    fn age(path: &Path) {
        let old = SystemTime::now() - 2 * STALE_TEMP;
        fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(old)
            .unwrap();
    }

    #[test]
    fn store_roots_name_their_cache_dir() {
        let cache = Path::new("/c/claude-history");
        assert_eq!(cache_dir_of(&cache.join("v3/roots-ab")), Some(cache));
        assert_eq!(
            cache_dir_of(&cache.join("config-0123456789abcdef/v3/roots-ab")),
            Some(cache)
        );
        assert_eq!(cache_dir_of(Path::new("/tmp/x/store")), None);
        assert_eq!(cache_dir_of(&cache.join("v2/roots-ab")), None);
    }

    #[test]
    fn removes_only_old_cache_files_and_the_dirs_they_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path();
        let mut cache_bytes = b"CLHIST01".to_vec();
        cache_bytes.extend([0; 32]);
        write(cache.join("projects/-a.bin"), &cache_bytes);
        let stale = write(cache.join("projects/.tmpAbC123"), b"x");
        age(&stale);
        let fresh = write(cache.join("projects/.tmpFresh1"), b"x");
        let foreign = write(cache.join("projects/notes.bin"), b"not ours");
        write(
            cache.join("config-0123456789abcdef/projects/-b.bin"),
            b"CLHIST01",
        );
        let config_store = write(cache.join("config-fedcba9876543210/v3/roots-1/LOCK"), b"");
        write(
            cache.join("config-fedcba9876543210/projects/-c.bin"),
            b"CLHIST01",
        );
        let unhashed = write(cache.join("config-mine/projects/-d.bin"), b"CLHIST01");
        write(
            cache.join("pi/root-0000000000000001/sessions.bin"),
            b"PIHIST01",
        );
        let wrong_magic = write(
            cache.join("omp/root-0000000000000002/sessions.bin"),
            b"PIHIST01",
        );
        write(
            cache.join("omp/root-0000000000000003/sessions.bin"),
            b"OMHIST01",
        );
        let kept = [
            write(cache.join("v3/roots-1/CURRENT"), b"g"),
            write(cache.join("semantic/embeddings-v2.bin"), b"m"),
            write(cache.join("catalog/a.bin"), b"CLHIST01"),
            write(cache.join("semantic-poc/b"), b"x"),
            fresh,
            foreign,
            config_store,
            unhashed,
            wrong_magic,
        ];

        let removed = remove_old_caches(cache);
        assert_eq!(removed.files, 6);
        assert_eq!(removed.bytes, 40 + 1 + 8 + 8 + 8 + 8);
        for path in &kept {
            assert!(path.exists(), "{}", path.display());
        }
        for gone in [
            "config-0123456789abcdef",
            "config-fedcba9876543210/projects",
            "pi",
            "omp/root-0000000000000003",
        ] {
            assert!(!cache.join(gone).exists(), "{gone}");
        }
        assert_eq!(remove_old_caches(cache), Removed::default());
    }
}

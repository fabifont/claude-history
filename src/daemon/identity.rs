//! Which claude-history binary a daemon runs, so that a replaced or upgraded
//! binary is noticed both by the daemon and by the next launch.

use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BinaryIdentity {
    pub path: PathBuf,
    pub version: String,
    pub dev: u64,
    pub ino: u64,
    pub size: u64,
    /// Modification time in epoch nanoseconds.
    pub mtime_ns: i64,
}

impl BinaryIdentity {
    /// The identity of the file at `path`, as this version.
    pub fn of(path: &Path) -> io::Result<Self> {
        let metadata = std::fs::metadata(path)?;
        let mtime_ns = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |since| i64::try_from(since.as_nanos()).unwrap_or(0));
        let (dev, ino) = file_id(&metadata);
        Ok(Self {
            path: path.to_path_buf(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            dev,
            ino,
            size: metadata.len(),
            mtime_ns,
        })
    }

    /// The identity of the running executable.
    pub fn current() -> io::Result<Self> {
        Self::of(&std::env::current_exe()?)
    }

    /// Whether `other` is the same build: the same file, unchanged, of the
    /// same version. Paths may differ when one binary is reached through
    /// links.
    pub fn same_build(&self, other: &Self) -> bool {
        self.version == other.version
            && self.dev == other.dev
            && self.ino == other.ino
            && self.size == other.size
            && self.mtime_ns == other.mtime_ns
    }

    /// Whether the file this identity was taken from is still that build.
    /// A removed, replaced or rewritten file is not.
    pub fn still_installed(&self) -> bool {
        Self::of(&self.path).is_ok_and(|now| now.same_build(self))
    }
}

#[cfg(unix)]
fn file_id(metadata: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (metadata.dev(), metadata.ino())
}

#[cfg(not(unix))]
fn file_id(_metadata: &std::fs::Metadata) -> (u64, u64) {
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{Duration, SystemTime};

    #[test]
    fn a_replaced_or_rewritten_binary_is_a_different_build() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("claude-history");
        fs::write(&path, "v1").unwrap();
        let identity = BinaryIdentity::of(&path).unwrap();
        assert!(identity.still_installed());
        assert!(identity.same_build(&BinaryIdentity::of(&path).unwrap()));

        // Installed by rename, as package managers and `cargo install` do.
        let next = dir.path().join("next");
        fs::write(&next, "v2").unwrap();
        fs::rename(&next, &path).unwrap();
        assert!(!identity.still_installed());

        // Rewritten in place with the same size.
        let identity = BinaryIdentity::of(&path).unwrap();
        fs::write(&path, "v3").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(SystemTime::now() + Duration::from_secs(5))
            .unwrap();
        assert!(!identity.still_installed());

        let mut other_version = BinaryIdentity::of(&path).unwrap();
        other_version.version = "0.0.1".into();
        assert!(!other_version.same_build(&BinaryIdentity::of(&path).unwrap()));

        fs::remove_file(&path).unwrap();
        assert!(!other_version.still_installed());
    }
}

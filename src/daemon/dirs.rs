//! Modification times of the directories in a directory, which the quick
//! pass compares each time.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Modification time of each directory directly in `root`, symlinks to
/// directories included, by the time they point to.
pub fn child_dirs(root: &Path) -> io::Result<HashMap<PathBuf, Option<SystemTime>>> {
    #[cfg(target_os = "macos")]
    {
        bulk::child_dirs(root)
    }
    #[cfg(not(target_os = "macos"))]
    {
        listed(root)
    }
}

/// `child_dirs` by listing `root` and stating each directory.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub fn listed(root: &Path) -> io::Result<HashMap<PathBuf, Option<SystemTime>>> {
    let mut dirs = HashMap::new();
    for entry in fs::read_dir(root)?.flatten() {
        let path = entry.path();
        let metadata = match entry.file_type() {
            Ok(kind) if kind.is_dir() => entry.metadata(),
            Ok(kind) if kind.is_symlink() => fs::metadata(&path),
            _ => continue,
        };
        if let Ok(metadata) = metadata
            && metadata.is_dir()
        {
            dirs.insert(path, metadata.modified().ok());
        }
    }
    Ok(dirs)
}

/// `getattrlistbulk(2)` returns the names, types and times of a
/// directory's entries a buffer at a time, where stating each one makes a
/// system call per entry.
#[cfg(target_os = "macos")]
mod bulk {
    use super::*;
    use std::ffi::{CString, OsStr};
    use std::os::unix::ffi::OsStrExt;
    use std::time::Duration;

    /// `enum vtype` of `<sys/vnode.h>`.
    const VDIR: u32 = 2;
    const VLNK: u32 = 5;

    pub fn child_dirs(root: &Path) -> io::Result<HashMap<PathBuf, Option<SystemTime>>> {
        let c_root = CString::new(root.as_os_str().as_bytes())?;
        // SAFETY: `c_root` is a NUL-terminated path.
        let fd = unsafe {
            libc::open(
                c_root.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let result = read(fd, root);
        // SAFETY: `fd` is open and closed only here.
        unsafe { libc::close(fd) };
        result
    }

    fn read(fd: libc::c_int, root: &Path) -> io::Result<HashMap<PathBuf, Option<SystemTime>>> {
        let mut request = libc::attrlist {
            bitmapcount: libc::ATTR_BIT_MAP_COUNT,
            reserved: 0,
            commonattr: libc::ATTR_CMN_RETURNED_ATTRS
                | libc::ATTR_CMN_NAME
                | libc::ATTR_CMN_OBJTYPE
                | libc::ATTR_CMN_MODTIME,
            volattr: 0,
            dirattr: 0,
            fileattr: 0,
            forkattr: 0,
        };
        let mut buffer = vec![0u64; 32 * 1024 / 8];
        let size = buffer.len() * 8;
        let mut dirs = HashMap::new();
        loop {
            // SAFETY: `request` and `buffer` outlive the call, which writes
            // at most `size` bytes.
            let count = unsafe {
                libc::getattrlistbulk(
                    fd,
                    (&raw mut request).cast(),
                    buffer.as_mut_ptr().cast(),
                    size,
                    u64::from(libc::FSOPT_PACK_INVAL_ATTRS),
                )
            };
            if count < 0 {
                return Err(io::Error::last_os_error());
            }
            if count == 0 {
                return Ok(dirs);
            }
            // SAFETY: the buffer holds `count` entries as the call packs
            // them, which `entries` reads within `size` bytes.
            let bytes = unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), size) };
            entries(bytes, count as usize, root, &mut dirs)?;
        }
    }

    fn field<T: Copy>(bytes: &[u8], at: usize) -> io::Result<T> {
        let end = at + std::mem::size_of::<T>();
        let field = bytes.get(at..end).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "truncated attribute entry")
        })?;
        // SAFETY: `field` holds `size_of::<T>()` bytes, read unaligned, and
        // `T` is a plain integer struct.
        Ok(unsafe { field.as_ptr().cast::<T>().read_unaligned() })
    }

    /// Reads `count` packed entries from `bytes`. Each is its length, the
    /// attributes it has, then the name, type and time requested, in that
    /// order, each four-byte aligned.
    fn entries(
        bytes: &[u8],
        count: usize,
        root: &Path,
        dirs: &mut HashMap<PathBuf, Option<SystemTime>>,
    ) -> io::Result<()> {
        let mut start = 0;
        for _ in 0..count {
            let length: u32 = field(bytes, start)?;
            let entry = bytes
                .get(start..start + length as usize)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated entry"))?;
            start += length as usize;

            let mut at = 4;
            let returned: libc::attribute_set_t = field(entry, at)?;
            at += std::mem::size_of::<libc::attribute_set_t>();
            let name_at = at;
            let name: libc::attrreference_t = field(entry, at)?;
            at += std::mem::size_of::<libc::attrreference_t>();
            let kind: u32 = field(entry, at)?;
            at += 4;
            let time: libc::timespec = field(entry, at)?;

            let name_start = name_at as isize + name.attr_dataoffset as isize;
            let Some(name) = usize::try_from(name_start).ok().and_then(|name_start| {
                entry.get(name_start..name_start + name.attr_length as usize)
            }) else {
                continue;
            };
            // The length counts the terminating NUL.
            let name = name.strip_suffix(&[0]).unwrap_or(name);
            let path = root.join(OsStr::from_bytes(name));
            let has = |attr| returned.commonattr & attr != 0;
            if !has(libc::ATTR_CMN_OBJTYPE) {
                continue;
            }
            match kind {
                VDIR => {
                    let modified = has(libc::ATTR_CMN_MODTIME)
                        .then(|| {
                            let secs = u64::try_from(time.tv_sec).ok()?;
                            let nanos = u32::try_from(time.tv_nsec).ok()?;
                            SystemTime::UNIX_EPOCH.checked_add(Duration::new(secs, nanos))
                        })
                        .flatten();
                    dirs.insert(path, modified);
                }
                VLNK => {
                    if let Ok(metadata) = fs::metadata(&path)
                        && metadata.is_dir()
                    {
                        dirs.insert(path, metadata.modified().ok());
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_dirs_match_a_listing() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a", "b with space", "ç", "-Users-x-project"] {
            fs::create_dir(dir.path().join(name)).unwrap();
        }
        fs::write(dir.path().join("file.jsonl"), "x").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(dir.path().join("a"), dir.path().join("link")).unwrap();
            std::os::unix::fs::symlink(dir.path().join("file.jsonl"), dir.path().join("flink"))
                .unwrap();
        }
        let dirs = child_dirs(dir.path()).unwrap();
        assert_eq!(dirs, listed(dir.path()).unwrap());
        assert_eq!(dirs.len(), 5);
        assert!(dirs.values().all(Option::is_some));
        assert!(child_dirs(&dir.path().join("missing")).is_err());
    }

    #[test]
    fn many_child_dirs_span_several_reads() {
        let dir = tempfile::tempdir().unwrap();
        for index in 0..2000 {
            fs::create_dir(
                dir.path()
                    .join(format!("-Users-someone-code-project-{index:05}")),
            )
            .unwrap();
        }
        assert_eq!(child_dirs(dir.path()).unwrap(), listed(dir.path()).unwrap());
    }
}

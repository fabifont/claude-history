use crate::error::{AppError, Result};
use serde_json::Value;
use std::fs::read_dir;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PiSessionRoot {
    pub path: PathBuf,
    pub flat: bool,
}

pub fn session_root() -> Result<PiSessionRoot> {
    session_root_from(
        std::env::var_os("PI_CODING_AGENT_DIR").map(PathBuf::from),
        std::env::var_os("PI_CODING_AGENT_SESSION_DIR").map(PathBuf::from),
        home::home_dir(),
        std::env::current_dir().ok(),
    )
}

fn session_root_from(
    agent_override: Option<PathBuf>,
    session_override: Option<PathBuf>,
    home_dir: Option<PathBuf>,
    cwd: Option<PathBuf>,
) -> Result<PiSessionRoot> {
    let agent_dir = if let Some(value) = agent_override {
        expand_path_with_home(value, home_dir.as_deref())?
    } else {
        home_dir
            .clone()
            .ok_or_else(|| {
                AppError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Could not determine home directory",
                ))
            })?
            .join(".pi")
            .join("agent")
    };

    if let Some(value) = session_override {
        return Ok(PiSessionRoot {
            path: expand_path_with_home(value, home_dir.as_deref())?,
            flat: true,
        });
    }

    let global_setting = configured_session_dir(&agent_dir.join("settings.json"));
    let project_setting = cwd
        .as_deref()
        .and_then(|cwd| configured_session_dir(&cwd.join(".pi/settings.json")));
    if let Some(Some(value)) = project_setting.or(global_setting) {
        return Ok(PiSessionRoot {
            path: expand_path_with_home(value, home_dir.as_deref())?,
            flat: true,
        });
    }

    Ok(PiSessionRoot {
        path: agent_dir.join("sessions"),
        flat: false,
    })
}

fn configured_session_dir(path: &Path) -> Option<Option<PathBuf>> {
    let contents = std::fs::read_to_string(path).ok()?;
    let settings = serde_json::from_str::<Value>(&contents).ok()?;
    let value = settings.as_object()?.get("sessionDir")?;
    Some(
        value
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from),
    )
}

fn expand_path_with_home(path: PathBuf, home_dir: Option<&Path>) -> Result<PathBuf> {
    let missing_home = || {
        AppError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Could not determine home directory",
        ))
    };
    let expanded = if path == Path::new("~") {
        home_dir.map(Path::to_path_buf).ok_or_else(missing_home)?
    } else if let Ok(rest) = path.strip_prefix("~/") {
        home_dir.ok_or_else(missing_home)?.join(rest)
    } else {
        path
    };
    Ok(if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()?.join(expanded)
    })
}

pub fn discover_files(root: &PiSessionRoot) -> Result<Vec<PathBuf>> {
    if !root.path.exists() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    if root.flat {
        collect_jsonl(&root.path, &mut files)?;
    } else {
        for entry in read_dir(&root.path)? {
            let entry = entry?;
            let path = entry.path();
            // The listing's file type saves a stat per directory; a symlink
            // is followed to see whether it leads to one.
            let is_dir = match entry.file_type() {
                Ok(kind) if !kind.is_symlink() => kind.is_dir(),
                _ => path.is_dir(),
            };
            if is_dir {
                collect_jsonl(&path, &mut files)?;
            }
        }
    }
    files.sort();
    Ok(files)
}

fn collect_jsonl(directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("jsonl") {
            files.push(path);
        }
    }
    Ok(())
}

pub fn delete_session(path: &Path) -> Result<()> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl")
        || !super::pi::is_pi_file(path)
    {
        return Err(AppError::SessionNotFound(path.display().to_string()));
    }
    std::fs::remove_file(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_directory_precedence_matches_pi() {
        let home = tempfile::tempdir().unwrap();
        let agent = home.path().join("agent");
        std::fs::create_dir_all(&agent).unwrap();
        std::fs::write(
            agent.join("settings.json"),
            r#"{"sessionDir":"~/settings-sessions"}"#,
        )
        .unwrap();

        let settings = session_root_from(
            Some(agent.clone()),
            None,
            Some(home.path().to_path_buf()),
            None,
        )
        .unwrap();
        assert_eq!(settings.path, home.path().join("settings-sessions"));
        assert!(settings.flat);

        let project = home.path().join("project");
        std::fs::create_dir_all(project.join(".pi")).unwrap();
        std::fs::write(
            project.join(".pi/settings.json"),
            r#"{"sessionDir":"./project-sessions"}"#,
        )
        .unwrap();
        let project_settings = session_root_from(
            Some(agent.clone()),
            None,
            Some(home.path().to_path_buf()),
            Some(project),
        )
        .unwrap();
        assert_eq!(
            project_settings.path,
            std::env::current_dir().unwrap().join("project-sessions")
        );
        assert!(project_settings.flat);

        let environment = session_root_from(
            Some(agent),
            Some(PathBuf::from("~/environment-sessions")),
            Some(home.path().to_path_buf()),
            None,
        )
        .unwrap();
        assert_eq!(environment.path, home.path().join("environment-sessions"));
        assert!(environment.flat);
    }

    #[test]
    fn discovers_flat_and_nested_roots_and_validates_headers() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("nested");
        let project = nested.join("--tmp-project--");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pi/v1.jsonl"),
            project.join("session.jsonl"),
        )
        .unwrap();
        std::fs::write(project.join("not-pi.jsonl"), "{\"type\":\"user\"}\n").unwrap();

        let nested_files = discover_files(&PiSessionRoot {
            path: nested,
            flat: false,
        })
        .unwrap();
        assert_eq!(nested_files.len(), 2);
        assert!(
            crate::history::pi::is_pi_file(&nested_files[0])
                || crate::history::pi::is_pi_file(&nested_files[1])
        );

        let flat = directory.path().join("flat");
        std::fs::create_dir_all(&flat).unwrap();
        std::fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pi/v2.jsonl"),
            flat.join("flat.jsonl"),
        )
        .unwrap();
        assert_eq!(
            discover_files(&PiSessionRoot {
                path: flat,
                flat: true,
            })
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn delete_removes_only_the_valid_pi_jsonl() {
        let directory = tempfile::tempdir().unwrap();
        let session = directory.path().join("session.jsonl");
        std::fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pi/v1.jsonl"),
            &session,
        )
        .unwrap();
        let sibling = directory.path().join("sibling.txt");
        std::fs::write(&sibling, "keep").unwrap();

        delete_session(&session).unwrap();
        assert!(!session.exists());
        assert!(sibling.exists());
    }

    #[cfg(unix)]
    #[test]
    fn nested_discovery_follows_symlinked_project_directories() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target");
        let root = directory.path().join("sessions");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pi/v3-branched.jsonl"),
            target.join("session.jsonl"),
        )
        .unwrap();
        symlink(&target, root.join("--linked--")).unwrap();

        let files = discover_files(&PiSessionRoot {
            path: root,
            flat: false,
        })
        .unwrap();
        assert_eq!(files.len(), 1);
        assert!(crate::history::pi::is_pi_file(&files[0]));
    }
}

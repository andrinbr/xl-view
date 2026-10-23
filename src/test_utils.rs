//! Shared support for unit tests in the library and binary targets.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) fn fixture_path(name: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

pub(crate) struct TestTempDir {
    path: PathBuf,
}

impl TestTempDir {
    pub(crate) fn new(prefix: &str) -> Self {
        loop {
            let id = NEXT_TEMP_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("{prefix}-{}-{id}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Self { path },
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!(
                    "cannot create temporary test directory {}: {error}",
                    path.display()
                ),
            }
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

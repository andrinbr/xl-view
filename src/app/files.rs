use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};

use rfd::FileDialog;
use winit::data_transfer::TypedData;
use winit::window::Window;

pub(super) fn dropped_file_paths(value: &dyn TypedData) -> std::io::Result<Vec<PathBuf>> {
    // Preserve transfer errors so the caller can retry an unfinished drop.
    let uris = value.try_as_uris()?;
    Ok(uris
        .into_iter()
        .filter_map(|uri| dropped_file_path(&uri))
        .collect())
}

/// Converts a `file:` URI to a path without consulting the file system.
fn dropped_file_path(uri: &str) -> Option<PathBuf> {
    let url = url::Url::parse(uri).ok()?;
    if url.scheme() != "file" {
        return None;
    }
    let path = url.to_file_path().ok()?;
    if path.as_os_str().as_encoded_bytes().contains(&0) {
        return None;
    }
    Some(path)
}

#[derive(Debug)]
pub(super) enum FilePickerResult {
    Selected(PathBuf),
    Canceled,
    Unsupported(PathBuf),
}

/// The folder holding the current image could not be listed.
#[derive(Debug, thiserror::Error)]
#[error("failed to read {}: {source}", .folder.display())]
pub(super) struct FolderReadError {
    folder: PathBuf,
    source: io::Error,
}

impl FolderReadError {
    pub(super) fn new(folder: &Path, source: io::Error) -> Self {
        Self {
            folder: folder.to_path_buf(),
            source,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FolderDirection {
    Previous,
    Next,
}

#[derive(Debug)]
pub(super) struct NeighborPaths {
    pub(super) previous: Option<PathBuf>,
    pub(super) next: Option<PathBuf>,
}

impl FolderDirection {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Previous => "previous",
            Self::Next => "next",
        }
    }
}

pub(super) fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

fn is_supported_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case("jxl"))
}

fn is_hidden_path(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name.as_encoded_bytes().starts_with(b"."))
        || has_hidden_file_attribute(path)
}

#[cfg(target_os = "windows")]
fn has_hidden_file_attribute(path: &Path) -> bool {
    use std::os::windows::fs::MetadataExt;

    const FILE_ATTRIBUTE_HIDDEN: u32 = 0x2;
    path.metadata()
        .is_ok_and(|metadata| metadata.file_attributes() & FILE_ATTRIBUTE_HIDDEN != 0)
}

#[cfg(not(target_os = "windows"))]
const fn has_hidden_file_attribute(_path: &Path) -> bool {
    false
}

/// Makes a path from the command line absolute. Symbolic links are kept, so
/// the image stays in the folder it was opened from.
///
/// A bare file name has an empty parent, which can neither be listed for
/// folder navigation nor given to the file picker.
pub(super) fn absolute_path(path: &Path) -> PathBuf {
    // A path that cannot be made absolute is left to fail when it is opened.
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

pub(super) fn first_image_path(paths: &[PathBuf]) -> Option<PathBuf> {
    paths
        .iter()
        .find(|path| is_supported_image_path(path))
        .cloned()
}

pub(super) fn select_image_file(
    window: &dyn Window,
    current_folder: Option<PathBuf>,
) -> FilePickerResult {
    let chooser = FileDialog::new()
        .set_parent(window)
        .set_title("Open image")
        .add_filter("Images", &["jxl"]);
    let chooser = if let Some(folder) = current_folder {
        chooser.set_directory(folder)
    } else {
        chooser
    };

    match chooser.pick_file() {
        Some(path) if is_supported_image_path(&path) => FilePickerResult::Selected(path),
        Some(path) => FilePickerResult::Unsupported(path),
        None => FilePickerResult::Canceled,
    }
}

pub(super) fn adjacent_image_path(
    current_path: &Path,
    direction: FolderDirection,
) -> Result<Option<PathBuf>, FolderReadError> {
    let neighbors = neighboring_image_paths(current_path)?;
    Ok(match direction {
        FolderDirection::Previous => neighbors.previous,
        FolderDirection::Next => neighbors.next,
    })
}

pub(super) fn neighboring_image_paths(
    current_path: &Path,
) -> Result<NeighborPaths, FolderReadError> {
    let parent = current_path.parent().unwrap_or_else(|| Path::new("."));
    let current_name = current_path.file_name();
    let entries =
        std::fs::read_dir(parent).map_err(|source| FolderReadError::new(parent, source))?;
    // Once the directory is open, scan it best-effort: one entry may disappear
    // or become unreadable without hiding the other navigable images.
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            is_supported_image_path(path)
                && path.is_file()
                && (path.file_name() == current_name || !is_hidden_path(path))
        })
        .collect::<Vec<_>>();
    // The exact name orders names that differ only in case, which the
    // directory may list in any order.
    paths.sort_by_cached_key(|path| {
        let name = file_name(path);
        (name.to_lowercase(), name)
    });
    Ok(NeighborPaths {
        previous: choose_adjacent_path(&paths, current_path, FolderDirection::Previous),
        next: choose_adjacent_path(&paths, current_path, FolderDirection::Next),
    })
}

fn choose_adjacent_path(
    sorted_paths: &[PathBuf],
    current_path: &Path,
    direction: FolderDirection,
) -> Option<PathBuf> {
    let current_name = current_path.file_name()?;
    let index = sorted_paths
        .iter()
        .position(|path| path.file_name() == Some(current_name))?;
    let adjacent = match direction {
        FolderDirection::Previous => index.checked_sub(1),
        FolderDirection::Next => index
            .checked_add(1)
            .filter(|index| *index < sorted_paths.len()),
    }?;
    Some(sorted_paths[adjacent].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestTempDir;

    #[test]
    fn dropped_file_urls_decode_names_and_reject_unusable_entries() {
        #[cfg(target_os = "windows")]
        let (uri, path) = (
            "file:///C:/Pictures/space%20%E9%9B%AA%23%25+.jxl",
            r"C:\Pictures\space 雪#%+.jxl",
        );
        #[cfg(not(target_os = "windows"))]
        let (uri, path) = (
            "file:///tmp/space%20%E9%9B%AA%23%25+.jxl",
            "/tmp/space 雪#%+.jxl",
        );
        assert_eq!(dropped_file_path(uri), Some(PathBuf::from(path)));
        for invalid in ["https://localhost/tmp/image.jxl", "not a URI"] {
            assert_eq!(dropped_file_path(invalid), None);
        }
        assert_eq!(dropped_file_path(&format!("{uri}%00")), None);
    }

    #[test]
    fn dot_prefixed_names_are_hidden_on_every_platform() {
        assert!(is_hidden_path(Path::new(".image.jxl")));
        assert!(is_hidden_path(Path::new("folder/._image.jxl")));
        assert!(!is_hidden_path(Path::new("folder/image.jxl")));
    }

    #[test]
    fn folder_neighbors_exclude_hidden_images_but_retain_a_hidden_anchor() {
        let directory = TestTempDir::new("xl-view-hidden-navigation");
        let hidden = directory.path().join("._sidecar.jxl");
        let first = directory.path().join("a.jxl");
        let second = directory.path().join("b.jxl");
        for path in [&hidden, &first, &second] {
            std::fs::write(path, []).unwrap();
        }

        let visible_neighbors = neighboring_image_paths(&first).unwrap();
        assert_eq!(visible_neighbors.previous, None);
        assert_eq!(visible_neighbors.next, Some(second));

        let hidden_neighbors = neighboring_image_paths(&hidden).unwrap();
        assert_eq!(hidden_neighbors.previous, None);
        assert_eq!(hidden_neighbors.next, Some(first));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_hidden_attribute_is_recognized() {
        use std::os::windows::fs::OpenOptionsExt;

        const FILE_ATTRIBUTE_HIDDEN: u32 = 0x2;
        let directory = TestTempDir::new("xl-view-hidden-attribute");
        let path = directory.path().join("image.jxl");
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .attributes(FILE_ATTRIBUTE_HIDDEN)
            .open(&path)
            .unwrap();
        assert!(is_hidden_path(&path));
    }

    #[test]
    fn displayed_file_name_does_not_expose_parent_directories() {
        assert_eq!(
            file_name(Path::new("/home/user/private/photo.jxl")),
            "photo.jxl"
        );
    }

    #[test]
    fn bare_file_name_gets_a_folder_that_can_be_listed() {
        let path = absolute_path(Path::new("image.jxl"));
        assert!(path.is_absolute());
        assert_eq!(path.file_name(), Some(OsStr::new("image.jxl")));
        // The empty parent of the bare name cannot be read.
        assert!(neighboring_image_paths(&path).is_ok());
    }

    #[test]
    fn dropped_paths_choose_the_first_image() {
        let paths = [
            PathBuf::from("notes.txt"),
            PathBuf::from("first.JXL"),
            PathBuf::from("second.jxl"),
        ];
        assert_eq!(first_image_path(&paths), Some(PathBuf::from("first.JXL")));
        assert_eq!(first_image_path(&[PathBuf::from("notes.txt")]), None);
    }

    #[test]
    fn folder_navigation_chooses_neighbors_without_wrapping() {
        let paths = ["a.jxl", "b.jxl", "c.jxl"].map(PathBuf::from);
        assert_eq!(
            choose_adjacent_path(&paths, Path::new("b.jxl"), FolderDirection::Previous),
            Some(PathBuf::from("a.jxl"))
        );
        assert_eq!(
            choose_adjacent_path(&paths, Path::new("b.jxl"), FolderDirection::Next),
            Some(PathBuf::from("c.jxl"))
        );
        assert_eq!(
            choose_adjacent_path(&paths, Path::new("a.jxl"), FolderDirection::Previous),
            None
        );
        assert_eq!(
            choose_adjacent_path(&paths, Path::new("c.jxl"), FolderDirection::Next),
            None
        );
    }
}

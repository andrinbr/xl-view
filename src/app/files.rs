use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use rfd::FileDialog;
use winit::window::Window;

#[derive(Debug)]
pub(super) enum FilePickerResult {
    Selected(PathBuf),
    Cancelled,
    Failed(String),
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

#[cfg(windows)]
fn has_hidden_file_attribute(path: &Path) -> bool {
    use std::os::windows::fs::MetadataExt;

    const FILE_ATTRIBUTE_HIDDEN: u32 = 0x2;
    path.metadata()
        .is_ok_and(|metadata| metadata.file_attributes() & FILE_ATTRIBUTE_HIDDEN != 0)
}

#[cfg(not(windows))]
const fn has_hidden_file_attribute(_path: &Path) -> bool {
    false
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
        Some(_) => {
            FilePickerResult::Failed("the selected file is not a supported image".to_owned())
        }
        None => FilePickerResult::Cancelled,
    }
}

pub(super) fn adjacent_image_path(
    current_path: &Path,
    direction: FolderDirection,
) -> Result<Option<PathBuf>, String> {
    let neighbors = neighboring_image_paths(current_path)?;
    Ok(match direction {
        FolderDirection::Previous => neighbors.previous,
        FolderDirection::Next => neighbors.next,
    })
}

pub(super) fn neighboring_image_paths(current_path: &Path) -> Result<NeighborPaths, String> {
    let parent = current_path.parent().unwrap_or_else(|| Path::new("."));
    let current_name = current_path.file_name();
    let entries = std::fs::read_dir(parent)
        .map_err(|error| format!("cannot read {}: {error}", parent.display()))?;
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
    paths.sort_by_cached_key(|path| {
        path.file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy()
            .to_lowercase()
    });
    Ok(NeighborPaths {
        previous: choose_adjacent_path(&paths, current_path, FolderDirection::Previous),
        next: choose_adjacent_path(&paths, current_path, FolderDirection::Next),
    })
}

pub(super) fn choose_adjacent_path(
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

    #[test]
    fn dot_prefixed_names_are_hidden_on_every_platform() {
        assert!(is_hidden_path(Path::new(".image.jxl")));
        assert!(is_hidden_path(Path::new("folder/._image.jxl")));
        assert!(!is_hidden_path(Path::new("folder/image.jxl")));
    }

    #[test]
    fn folder_neighbors_exclude_hidden_images_but_retain_a_hidden_anchor() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "xl-view-hidden-navigation-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let hidden = directory.join("._sidecar.jxl");
        let first = directory.join("a.jxl");
        let second = directory.join("b.jxl");
        for path in [&hidden, &first, &second] {
            std::fs::write(path, []).unwrap();
        }

        let visible_neighbors = neighboring_image_paths(&first).unwrap();
        assert_eq!(visible_neighbors.previous, None);
        assert_eq!(visible_neighbors.next, Some(second));

        let hidden_neighbors = neighboring_image_paths(&hidden).unwrap();
        assert_eq!(hidden_neighbors.previous, None);
        assert_eq!(hidden_neighbors.next, Some(first));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_hidden_attribute_is_recognized() {
        use std::os::windows::fs::OpenOptionsExt;

        const FILE_ATTRIBUTE_HIDDEN: u32 = 0x2;
        let path = std::env::temp_dir().join(format!(
            "xl-view-hidden-attribute-{}.jxl",
            std::process::id()
        ));
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .attributes(FILE_ATTRIBUTE_HIDDEN)
            .open(&path)
            .unwrap();
        assert!(is_hidden_path(&path));
        std::fs::remove_file(path).unwrap();
    }
}

//! Foundation-backed conversion of macOS drag-and-drop file URLs.
//!
//! At the pinned winit revision, `AppKit`'s `PasteboardValue::try_as_uris()` returns
//! `NSPasteboardTypeFileURL` values as strings. Finder may encode those values as
//! file-reference URLs such as `file:///.file/id=...`, which are stable macOS
//! references rather than directly usable POSIX paths:
//! <https://github.com/rust-windowing/winit/blob/42366231b0d92480a933339bd5eaa7fba8106f42/winit-appkit/src/dnd.rs#L268-L304>
//!
//! winit's default `try_as_file_paths()` implementation parses each string with
//! the platform-independent `url` crate. That conversion is purely syntactic, so
//! a Finder reference becomes the unusable path `/.file/id=...`:
//! <https://github.com/rust-windowing/winit/blob/42366231b0d92480a933339bd5eaa7fba8106f42/winit-core/src/data_transfer.rs#L223-L237>
//!
//! Reading the original URIs here preserves the information Foundation needs.
//! `NSURL::filePathURL()` then resolves both ordinary and reference file URLs to
//! filesystem paths. This adapter can be removed once winit performs that
//! resolution in its `AppKit` backend.

use std::path::{Path, PathBuf};

use objc2_foundation::{NSString, NSURL};
use winit::data_transfer::TypedData;

pub(in crate::app) fn dropped_file_paths(value: &dyn TypedData) -> std::io::Result<Vec<PathBuf>> {
    // Keep transfer-level errors (including WouldBlock) visible to the caller,
    // but ignore individual values that are not resolvable local file URLs.
    let uris = value.try_as_uris()?;
    Ok(uris.into_iter().filter_map(|uri| file_path(&uri)).collect())
}

fn file_path(uri: &str) -> Option<PathBuf> {
    // URLWithString() does not classify a bare absolute path as a file URL.
    if Path::new(uri).is_absolute() {
        return Some(PathBuf::from(uri));
    }

    NSURL::URLWithString(&NSString::from_str(uri))
        .and_then(|url| url.filePathURL())
        .and_then(|url| url.to_file_path())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestTempDir;
    use winit::data_transfer::{TransferType, TypeHint};

    #[derive(Debug)]
    struct UriData(Vec<String>);

    impl TypedData for UriData {
        fn type_(&self) -> &dyn TransferType {
            &TypeHint::UriList
        }

        fn try_read(&self) -> Option<Box<dyn std::io::BufRead>> {
            None
        }

        fn try_as_uris(&self) -> std::io::Result<Vec<String>> {
            Ok(self.0.clone())
        }

        fn try_as_string(&self) -> std::io::Result<String> {
            Err(std::io::ErrorKind::Unsupported.into())
        }
    }

    #[test]
    fn file_urls_decode_percent_encoded_paths() {
        assert_eq!(
            file_path("file:///tmp/xl-view%20image.jxl"),
            Some(PathBuf::from("/tmp/xl-view image.jxl"))
        );
    }

    #[test]
    fn legacy_raw_paths_are_preserved() {
        assert_eq!(
            file_path("/tmp/xl-view image.jxl"),
            Some(PathBuf::from("/tmp/xl-view image.jxl"))
        );
    }

    #[test]
    fn non_file_urls_are_ignored() {
        assert_eq!(file_path("https://example.com/image.jxl"), None);
    }

    #[test]
    fn invalid_entries_do_not_discard_resolvable_file_urls() {
        let data = UriData(vec![
            "https://example.com/image.jxl".to_owned(),
            "file:///tmp/local.jxl".to_owned(),
        ]);
        assert_eq!(
            dropped_file_paths(&data).unwrap(),
            vec![PathBuf::from("/tmp/local.jxl")]
        );
    }

    #[test]
    fn file_reference_urls_resolve_to_paths() {
        let directory = TestTempDir::new("xl-view-file-reference");
        let path = directory.path().join("image.jxl");
        std::fs::write(&path, []).unwrap();

        let path_url = NSURL::from_file_path(&path).unwrap();
        let reference_url = path_url.fileReferenceURL().unwrap();
        let uri = reference_url.absoluteString().unwrap().to_string();
        let resolved = file_path(&uri).unwrap();

        assert_eq!(
            std::fs::canonicalize(resolved).unwrap(),
            std::fs::canonicalize(&path).unwrap()
        );
    }
}

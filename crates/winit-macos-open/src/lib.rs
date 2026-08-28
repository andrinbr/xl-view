#![cfg(target_os = "macos")]

//! A small bridge from macOS open-document events to a Rust callback.
//!
//! Install [`OpenFiles`] after creating winit's event loop and retain it for as
//! long as the application is running.

use std::path::PathBuf;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{NSApplication, NSApplicationDelegate};
use objc2_foundation::{NSArray, NSObject, NSObjectProtocol, NSURL};

/// Keeps the application delegate alive while open-document events are handled.
#[must_use = "retain the guard while open-document events are handled"]
pub struct OpenFiles {
    _delegate: Retained<AppDelegate>,
}

impl OpenFiles {
    /// Installs a callback for paths opened through Finder.
    ///
    /// This must be called on the main thread, after creating winit's event loop.
    /// Returns `None` if either requirement is not met.
    pub fn install<F>(handler: F) -> Option<Self>
    where
        F: Fn(Vec<PathBuf>) + 'static,
    {
        let mtm = MainThreadMarker::new()?;
        let application = NSApplication::sharedApplication(mtm);
        // Winit's "Custom NSApplicationDelegate" documentation guarantees that
        // it leaves this delegate slot available to application integrations.
        if application.delegate().is_some() {
            return None;
        }

        let delegate = AppDelegate::new(mtm, Box::new(handler));
        application.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
        Some(Self {
            _delegate: delegate,
        })
    }
}

// SAFETY: Objective-C requires these protocol implementations and message
// sends. AppDelegate is main-thread-only, and the selectors match AppKit's
// declared signatures.
define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = Box<dyn Fn(Vec<PathBuf>)>]
    #[name = "WinitMacOSOpenDelegate"]
    struct AppDelegate;

    unsafe impl NSObjectProtocol for AppDelegate {}

    unsafe impl NSApplicationDelegate for AppDelegate {
        #[unsafe(method(application:openURLs:))]
        #[allow(non_snake_case)]
        fn application_openURLs(&self, _application: &NSApplication, urls: &NSArray<NSURL>) {
            let paths = urls
                .to_vec()
                .into_iter()
                .filter_map(|url| url.filePathURL())
                .filter_map(|url| url.to_file_path())
                .collect::<Vec<_>>();
            (self.ivars())(paths);
        }

        #[unsafe(method(applicationSupportsSecureRestorableState:))]
        #[allow(non_snake_case)]
        fn applicationSupportsSecureRestorableState(
            &self,
            _application: &NSApplication,
        ) -> bool {
            true
        }
    }
);

impl AppDelegate {
    #[allow(unsafe_code)]
    fn new(mtm: MainThreadMarker, handler: Box<dyn Fn(Vec<PathBuf>)>) -> Retained<Self> {
        unsafe { msg_send![super(Self::alloc(mtm).set_ivars(handler)), init] }
    }
}

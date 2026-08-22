//! Event-driven macOS display-target updates.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSNotificationCenter, NSObjectProtocol, ns_string};
use winit_common::foundation::create_observer;

use super::{HdrTarget, macos_target};
use crate::gpu::DisplayChangeNotifier;

pub(in crate::gpu) struct DisplayTargetProvider {
    // Process-lifetime registration: NotificationCenter retains block observers until removed.
    _observer_token: Retained<ProtocolObject<dyn NSObjectProtocol>>,
}

impl DisplayTargetProvider {
    // AppKit pushes target changes, so ordinary renders use cached state.
    pub(in crate::gpu) const QUERY_BEFORE_RENDER: bool = false;

    pub(in crate::gpu) fn new(notify_changed: DisplayChangeNotifier) -> Self {
        let center = NSNotificationCenter::defaultCenter();
        // AppKit posts this for display configuration and EDR headroom changes.
        // objc2 exposes the typed AppKit constant as an unsafe extern static.
        let observer = create_observer(
            &center,
            ns_string!("NSApplicationDidChangeScreenParametersNotification"),
            move |_| notify_changed(),
        );
        Self {
            _observer_token: observer,
        }
    }

    #[allow(clippy::unused_self)] // The provider also owns the notification observer.
    pub(in crate::gpu) fn target(&self, info: &wgpu::DisplayHdrInfo) -> HdrTarget {
        // wgpu resolves the window's screen, only normalize its EDR values here.
        let headroom = info.headroom.unwrap_or_default();
        macos_target(headroom.potential, headroom.current)
    }
}

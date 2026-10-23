use super::*;

#[test]
#[cfg(target_os = "linux")]
fn wayland_display_is_accepted() {
    assert!(has_wayland_endpoint(Some(OsStr::new("wayland-0")), None));
}

#[test]
#[cfg(target_os = "linux")]
fn inherited_wayland_socket_is_accepted() {
    assert!(has_wayland_endpoint(None, Some(OsStr::new("3"))));
}

#[test]
#[cfg(target_os = "linux")]
fn missing_or_empty_wayland_endpoints_are_rejected() {
    assert!(!has_wayland_endpoint(None, None));
    assert!(!has_wayland_endpoint(Some(OsStr::new("")), None));
}

#[test]
fn fullscreen_cursor_hides_after_inactivity() {
    let now = Instant::now();
    assert_eq!(
        fullscreen_cursor_hide_deadline(FullscreenState::Fullscreen, false, now),
        Some(now + FULLSCREEN_CURSOR_HIDE_DELAY)
    );
}

#[test]
fn cursor_timeout_is_disabled_while_windowed_or_dragging() {
    let now = Instant::now();
    assert_eq!(
        fullscreen_cursor_hide_deadline(FullscreenState::Windowed, false, now),
        None
    );
    assert_eq!(
        fullscreen_cursor_hide_deadline(FullscreenState::Fullscreen, true, now),
        None
    );
}

#[test]
fn event_loop_uses_earliest_pending_deadline() {
    let now = Instant::now();
    let prefetch = now + Duration::from_millis(250);
    let cursor = now + FULLSCREEN_CURSOR_HIDE_DELAY;
    let metadata = now + DISPLAY_METADATA_REFRESH_DELAY;
    assert_eq!(
        next_wake_deadline(Some(prefetch), Some(cursor), Some(metadata)),
        Some(metadata)
    );
    assert_eq!(
        next_wake_deadline(Some(prefetch), Some(cursor), None),
        Some(prefetch)
    );
    assert_eq!(next_wake_deadline(None, Some(cursor), None), Some(cursor));
    assert_eq!(next_wake_deadline(None, None, None), None);
}

#[test]
fn dropped_file_waits_for_both_drop_confirmation_and_transfer_data() {
    let mut pending = PendingDrop {
        id: DataTransferId::from_raw(1),
        fetch_serial: AsyncRequestSerial::get(),
        dropped: false,
        data: PendingDropData::Path(PathBuf::from("image.jxl")),
    };
    assert!(pending.take_ready_data().is_none());
    pending.dropped = true;
    assert!(matches!(
        pending.take_ready_data(),
        Some(PendingDropData::Path(path)) if path == Path::new("image.jxl")
    ));

    let mut pending = PendingDrop {
        id: DataTransferId::from_raw(2),
        fetch_serial: AsyncRequestSerial::get(),
        dropped: true,
        data: PendingDropData::Awaiting,
    };
    assert!(pending.take_ready_data().is_none());
    pending.data = PendingDropData::Unsupported;
    assert!(matches!(
        pending.take_ready_data(),
        Some(PendingDropData::Unsupported)
    ));
}

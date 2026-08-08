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
fn keyboard_bindings_match_the_interaction_contract() {
    let bindings = [
        (KeyCode::KeyF, KeyboardAction::Fit),
        (KeyCode::Digit1, KeyboardAction::OneToOne),
        (KeyCode::Numpad1, KeyboardAction::OneToOne),
        (KeyCode::Equal, KeyboardAction::ZoomIn),
        (KeyCode::NumpadAdd, KeyboardAction::ZoomIn),
        (KeyCode::Minus, KeyboardAction::ZoomOut),
        (KeyCode::NumpadSubtract, KeyboardAction::ZoomOut),
        (KeyCode::KeyO, KeyboardAction::Open),
        (KeyCode::KeyQ, KeyboardAction::Quit),
        (KeyCode::BracketLeft, KeyboardAction::ExposureDown),
        (KeyCode::BracketRight, KeyboardAction::ExposureUp),
        (KeyCode::KeyR, KeyboardAction::ResetViewAndExposure),
        (KeyCode::KeyB, KeyboardAction::CycleBackground),
        (KeyCode::KeyI, KeyboardAction::ToggleMetadata),
        (KeyCode::F11, KeyboardAction::ToggleFullscreen),
        (KeyCode::Enter, KeyboardAction::ToggleFullscreen),
        (KeyCode::KeyA, KeyboardAction::PanLeft),
        (KeyCode::KeyD, KeyboardAction::PanRight),
        (KeyCode::KeyW, KeyboardAction::PanUp),
        (KeyCode::KeyS, KeyboardAction::PanDown),
        (KeyCode::ArrowLeft, KeyboardAction::PreviousImage),
        (KeyCode::ArrowRight, KeyboardAction::NextImage),
    ];
    for (key, expected) in bindings {
        assert_eq!(
            keyboard_action(
                PhysicalKey::Code(key),
                ModifiersState::empty(),
                FullscreenState::Windowed,
            ),
            Some(expected),
        );
    }

    assert_eq!(
        keyboard_action(
            PhysicalKey::Code(KeyCode::KeyO),
            ModifiersState::CONTROL,
            FullscreenState::Windowed,
        ),
        Some(KeyboardAction::Open),
    );
    assert_eq!(
        keyboard_action(
            PhysicalKey::Code(KeyCode::KeyQ),
            ModifiersState::CONTROL,
            FullscreenState::Windowed,
        ),
        Some(KeyboardAction::Quit),
    );
    assert_eq!(
        keyboard_action(
            PhysicalKey::Code(KeyCode::Escape),
            ModifiersState::empty(),
            FullscreenState::Fullscreen,
        ),
        Some(KeyboardAction::LeaveFullscreen),
    );
    assert_eq!(
        keyboard_action(
            PhysicalKey::Code(KeyCode::Escape),
            ModifiersState::empty(),
            FullscreenState::Windowed,
        ),
        None,
    );
}

#[test]
fn source_encoding_details_are_concise() {
    let sdr = xl_view::decode::SourceColorEncoding::Enumerated {
        colour_space: "Rgb".to_owned(),
        white_point: "D65".to_owned(),
        primaries: "Srgb".to_owned(),
        transfer_function: "sRGB".to_owned(),
    };
    assert_eq!(
        source_encoding_details(&sdr),
        (
            "Rgb, Srgb primaries, D65 white".to_owned(),
            "sRGB".to_owned()
        )
    );
    assert_eq!(source_range_summary(false, "sRGB"), "SDR (sRGB)");
    assert_eq!(source_range_summary(true, "HLG"), "HDR (HLG)");
    assert_eq!(source_range_summary(true, "PQ"), "HDR (PQ)");
}

#[test]
fn decode_summary_combines_time_and_memory() {
    const MIB: u64 = 1024 * 1024;
    assert_eq!(
        decode_summary(
            DecodeTiming::Measured(Duration::from_millis(611)),
            usize::try_from(16 * MIB).unwrap(),
        ),
        "611.0 ms, 16 MiB"
    );
    assert_eq!(
        decode_summary(DecodeTiming::CacheHit(Duration::from_millis(611)), 0),
        "611.0 ms (cached), 0 MiB"
    );
}

#[test]
fn dimensions_summary_includes_decimal_megapixels() {
    assert_eq!(dimensions_summary(4000, 3000), "4000 x 3000 (12.0 MP)");
    assert_eq!(dimensions_summary(1024, 1024), "1024 x 1024 (1.0 MP)");
}

#[test]
fn embedded_metadata_is_grouped_and_missing_fields_are_omitted() {
    let exif = ExifMetadata {
        aperture_f_number: Some(2.8),
        artist: Some("Ada Example".to_owned()),
        camera_make: Some("ACME".to_owned()),
        camera_model: Some("ACME Photon 1".to_owned()),
        captured_at: Some("2026:07:13 12:34:56".to_owned()),
        copyright: Some("CC0 fixture".to_owned()),
        exposure_bias_ev: Some(-1.0 / 3.0),
        exposure_time_seconds: Some(1.0 / 125.0),
        focal_length_mm: Some(50.0),
        iso_speed: Some(200),
        lens_make: None,
        lens_model: Some("Prime 50".to_owned()),
        parse_error: None,
        software: None,
    };
    assert_eq!(
        capture_metadata_rows(Some(&exif)),
        [
            ("Camera".to_owned(), "ACME Photon 1".to_owned()),
            ("Lens".to_owned(), "Prime 50".to_owned()),
            ("Captured".to_owned(), "2026-07-13 12:34:56".to_owned()),
            ("Shutter".to_owned(), "1/125 s".to_owned()),
            ("Aperture".to_owned(), "f/2.8".to_owned()),
            ("ISO".to_owned(), "200".to_owned()),
            ("Focal length".to_owned(), "50 mm".to_owned()),
            ("Exposure bias".to_owned(), "-0.3 EV".to_owned()),
        ]
    );
    assert_eq!(
        attribution_metadata_rows(Some(&exif), None),
        [
            ("Artist".to_owned(), "Ada Example".to_owned()),
            ("Copyright".to_owned(), "CC0 fixture".to_owned()),
        ]
    );
    assert!(capture_metadata_rows(None).is_empty());
    assert!(attribution_metadata_rows(None, None).is_empty());

    let mut empty = exif.clone();
    empty.aperture_f_number = None;
    empty.artist = None;
    empty.camera_make = None;
    empty.camera_model = None;
    empty.captured_at = None;
    empty.copyright = None;
    empty.exposure_bias_ev = None;
    empty.exposure_time_seconds = None;
    empty.focal_length_mm = None;
    empty.iso_speed = None;
    empty.lens_make = None;
    empty.lens_model = None;
    empty.software = None;
    assert!(capture_metadata_rows(Some(&empty)).is_empty());
    assert!(attribution_metadata_rows(Some(&empty), None).is_empty());

    assert_eq!(format_exif_datetime("not a date"), "not a date".to_owned());
}

#[test]
fn xmp_rating_is_the_first_attribution_row() {
    let xmp = XmpMetadata {
        parse_error: None,
        rating: Some(4.5),
    };
    assert_eq!(
        attribution_metadata_rows(None, Some(&xmp)).first(),
        Some(&("Rating".to_owned(), "4.5 Stars".to_owned()))
    );

    let one_star = XmpMetadata {
        parse_error: None,
        rating: Some(1.0),
    };
    assert_eq!(
        xmp_rating_row(Some(&one_star)),
        Some(("Rating".to_owned(), "1 Star".to_owned()))
    );

    let rejected = XmpMetadata {
        parse_error: None,
        rating: Some(-1.0),
    };
    assert_eq!(
        xmp_rating_row(Some(&rejected)),
        Some(("Rating".to_owned(), "Rejected".to_owned()))
    );

    let unrated = XmpMetadata {
        parse_error: None,
        rating: Some(0.0),
    };
    assert_eq!(
        xmp_rating_row(Some(&unrated)),
        Some(("Rating".to_owned(), "Unrated".to_owned()))
    );
}

#[test]
fn fullscreen_requested_state_toggles_without_waiting_for_the_compositor() {
    assert_eq!(
        FullscreenState::Windowed.toggled(),
        FullscreenState::Fullscreen
    );
    assert_eq!(
        FullscreenState::Fullscreen.toggled(),
        FullscreenState::Windowed
    );
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
    assert_eq!(
        next_wake_deadline(Some(prefetch), Some(cursor)),
        Some(prefetch)
    );
    assert_eq!(next_wake_deadline(None, Some(cursor)), Some(cursor));
    assert_eq!(next_wake_deadline(None, None), None);
}

#[test]
fn displayed_file_name_does_not_expose_parent_directories() {
    assert_eq!(
        file_name(Path::new("/home/user/private/photo.jxl")),
        "photo.jxl"
    );
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

#[test]
fn double_click_only_depends_on_the_click_interval() {
    let start = Instant::now();
    assert!(is_double_click(start, start + Duration::from_millis(250)));
    assert!(!is_double_click(start, start + Duration::from_millis(500)));
}

#[test]
fn image_cursor_shows_pan_availability_and_drag_state() {
    assert_eq!(image_cursor(false, false), CursorIcon::Default);
    assert_eq!(image_cursor(false, true), CursorIcon::Default);
    #[cfg(target_os = "windows")]
    {
        assert_eq!(image_cursor(true, false), CursorIcon::Default);
        assert_eq!(image_cursor(true, true), CursorIcon::Grabbing);
    }
    #[cfg(not(target_os = "windows"))]
    {
        assert_eq!(image_cursor(true, false), CursorIcon::Grab);
        assert_eq!(image_cursor(true, true), CursorIcon::Grabbing);
    }
}

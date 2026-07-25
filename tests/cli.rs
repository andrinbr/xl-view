use std::process::Command;

fn command_without_display() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xl-view"));
    command
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY")
        .env_remove("WAYLAND_SOCKET")
        .env_remove("RUST_LOG");
    command
}

#[test]
fn version_is_available_without_a_display() {
    let output = command_without_display().arg("--version").output().unwrap();

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        format!("xl-view {}", env!("CARGO_PKG_VERSION"))
    );
    assert!(output.stderr.is_empty());
}

#[test]
#[cfg(target_os = "linux")]
fn unsupported_session_has_an_actionable_error() {
    let output = command_without_display().output().unwrap();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("requires a native Wayland session"));
    assert!(stderr.contains("WAYLAND_DISPLAY"));
    assert!(stderr.contains("category=\"no_native_wayland\""));
    assert_eq!(
        stderr.matches("requires a native Wayland session").count(),
        1,
        "fatal error should be reported exactly once: {stderr}"
    );
}

#[test]
#[cfg(target_os = "linux")]
fn fatal_error_remains_visible_when_logging_is_disabled() {
    let output = command_without_display()
        .env("RUST_LOG", "off")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.starts_with("xl-view: "));
    assert_eq!(
        stderr.matches("requires a native Wayland session").count(),
        1
    );
}

#[test]
#[cfg(target_os = "linux")]
fn invalid_log_filter_is_reported_before_falling_back() {
    let output = command_without_display()
        .env("RUST_LOG", "[invalid")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("invalid RUST_LOG; using the default filter"));
    assert!(stderr.contains("rust_log=[invalid"));
    assert!(stderr.contains("fallback=\"xl_view=info,vulkan_hdr_metadata=warn\""));
    assert_eq!(stderr.matches("invalid RUST_LOG").count(), 1);
    assert_eq!(
        stderr.matches("requires a native Wayland session").count(),
        1
    );
}

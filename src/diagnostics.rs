use std::error::Error;
use std::io::{self, IsTerminal, Write};
use std::panic;

use rfd::{MessageDialog, MessageLevel};
use tracing_subscriber::EnvFilter;

use crate::APPLICATION_NAME;
use crate::error::{AppError, RuntimeError};
use crate::gpu::GpuInitializationError;

const DEFAULT_LOG_FILTER: &str = "xl_view=info,vulkan_hdr_metadata=warn";

pub fn initialize_logging() {
    let (filter, invalid_filter) = match std::env::var(EnvFilter::DEFAULT_ENV) {
        Ok(directives) => match EnvFilter::try_new(&directives) {
            Ok(filter) => (filter, None),
            Err(error) => (
                EnvFilter::new(DEFAULT_LOG_FILTER),
                Some((directives, error.to_string())),
            ),
        },
        Err(std::env::VarError::NotPresent) => (EnvFilter::new(DEFAULT_LOG_FILTER), None),
        Err(std::env::VarError::NotUnicode(value)) => (
            EnvFilter::new(DEFAULT_LOG_FILTER),
            Some((
                value.to_string_lossy().into_owned(),
                "value is not UTF-8".to_owned(),
            )),
        ),
    };
    let ansi = io::stderr().is_terminal();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(ansi)
        .with_writer(io::stderr)
        .with_target(true)
        .with_thread_names(true)
        .compact()
        .init();

    if let Some((directives, error)) = invalid_filter {
        tracing::warn!(
            rust_log = %directives,
            %error,
            fallback = DEFAULT_LOG_FILTER,
            "invalid RUST_LOG; using the default filter"
        );
    }
}

pub fn report_fatal(error: &AppError) {
    if tracing::enabled!(tracing::Level::ERROR) {
        if let Some(causes) = source_chain(error) {
            tracing::error!(
                category = error.category(),
                error = %error,
                %causes,
                "application stopped"
            );
        } else {
            tracing::error!(
                category = error.category(),
                error = %error,
                "application stopped"
            );
        }
    } else {
        eprintln!("xl-view: {error}");
    }
}

pub fn show_startup_error_dialog(error: &AppError) {
    if !should_show_startup_error_dialog(error, io::stderr().is_terminal(), x11_display_available())
    {
        return;
    }
    MessageDialog::new()
        .set_level(MessageLevel::Error)
        .set_title(format!("{APPLICATION_NAME} could not start"))
        .set_description(error.to_string())
        .show();
}

fn should_show_startup_error_dialog(
    error: &AppError,
    stderr_is_terminal: bool,
    x11_display_available: bool,
) -> bool {
    #[cfg(not(target_os = "linux"))]
    let _ = x11_display_available;
    if stderr_is_terminal {
        return false;
    }
    match error {
        #[cfg(target_os = "linux")]
        AppError::NoNativeWayland => x11_display_available,
        AppError::Runtime(RuntimeError::GpuInitialization(
            GpuInitializationError::BackendUnavailable { .. },
        )) => true,
        AppError::EventLoop(_) | AppError::Runtime(_) => false,
    }
}

#[cfg(target_os = "linux")]
fn x11_display_available() -> bool {
    std::env::var_os("DISPLAY").is_some_and(|display| !display.is_empty())
}

#[cfg(not(target_os = "linux"))]
const fn x11_display_available() -> bool {
    false
}

fn source_chain(error: &(dyn Error + 'static)) -> Option<String> {
    let mut causes = Vec::new();
    let mut previous = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        let message = error.to_string();
        if message != previous {
            causes.push(message.clone());
        }
        previous = message;
        source = error.source();
    }
    (!causes.is_empty()).then(|| causes.join(": "))
}

pub fn print_report(phase: &str, report: &str) {
    let output = format_report(phase, report);
    let mut stdout = io::stdout().lock();
    if let Err(error) = stdout.write_all(output.as_bytes()) {
        tracing::warn!(%error, phase, "cannot write diagnostic report");
    }
}

fn format_report(phase: &str, report: &str) -> String {
    format!("Diagnostics [{phase}]:\n{report}\n")
}

pub fn install_panic_hook() {
    panic::set_hook(Box::new(|panic_info| {
        let location = panic_info
            .location()
            .map_or_else(|| "unknown location".to_owned(), ToString::to_string);
        let message = if let Some(message) = panic_info.payload().downcast_ref::<&str>() {
            *message
        } else if let Some(message) = panic_info.payload().downcast_ref::<String>() {
            message
        } else {
            "non-string panic payload"
        };

        let report = format!(
            "xl-view stopped because of an internal error at {location}: {message}. \
             Set RUST_LOG=xl_view=debug and report the diagnostic log."
        );
        if tracing::enabled!(tracing::Level::ERROR) {
            tracing::error!(category = "panic", %location, %message, "{report}");
        } else {
            eprintln!("{report}");
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RuntimeError;

    #[test]
    fn default_filter_covers_the_application_and_hdr_bridge() {
        assert!(EnvFilter::try_new(DEFAULT_LOG_FILTER).is_ok());
        assert!(DEFAULT_LOG_FILTER.contains("xl_view=info"));
        assert!(DEFAULT_LOG_FILTER.contains("vulkan_hdr_metadata=warn"));
    }

    #[test]
    fn reports_are_labeled_and_end_with_one_newline() {
        assert_eq!(
            format_report("startup", "adapter: test"),
            "Diagnostics [startup]:\nadapter: test\n"
        );
    }

    #[test]
    fn source_chains_skip_transparent_wrappers() {
        let error = AppError::from(RuntimeError::GpuBackgroundWork(io::Error::other(
            "tile worker disconnected",
        )));

        assert_eq!(
            source_chain(&error).as_deref(),
            Some("tile worker disconnected")
        );
    }

    #[test]
    fn backend_unavailable_dialog_is_limited_to_desktop_launches() {
        let error = AppError::from(RuntimeError::from(
            GpuInitializationError::backend_unavailable(wgpu::RequestAdapterError::EnvNotSet),
        ));

        assert!(should_show_startup_error_dialog(&error, false, false));
        assert!(!should_show_startup_error_dialog(&error, true, true));
    }

    #[test]
    fn unrelated_runtime_errors_do_not_open_startup_dialogs() {
        let error = AppError::from(RuntimeError::GpuBackgroundWork(io::Error::other(
            "tile worker disconnected",
        )));

        assert!(!should_show_startup_error_dialog(&error, false, true));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn missing_wayland_dialog_requires_an_x11_desktop_launch() {
        assert!(should_show_startup_error_dialog(
            &AppError::NoNativeWayland,
            false,
            true,
        ));
        assert!(!should_show_startup_error_dialog(
            &AppError::NoNativeWayland,
            false,
            false,
        ));
        assert!(!should_show_startup_error_dialog(
            &AppError::NoNativeWayland,
            true,
            true,
        ));
    }
}

// Prevent Windows release builds from automatically opening a console
#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

mod app;
mod app_icon;
mod cli;
mod diagnostics;
mod error;
mod gpu;
mod units;

use std::process::ExitCode;

use clap::Parser;

#[cfg(target_os = "linux")]
const APPLICATION_ID: &str = "io.github.andrinbr.xl_view";
const APPLICATION_NAME: &str = "XL-View";

fn main() -> ExitCode {
    #[cfg(all(target_os = "windows", not(debug_assertions)))]
    // Attach to the parent console on windows when one is available.
    let _ = winsafe::AttachConsole(winsafe::PidParent::Parent);

    // Configure diagnostics before starting the application
    diagnostics::initialize_logging();
    diagnostics::install_panic_hook();

    // Parse command-line options
    let cli = match cli::Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let exit_code = u8::try_from(error.exit_code()).unwrap_or(1);
            let _ = error.print();
            return ExitCode::from(exit_code);
        }
    };

    // Report startup failures
    if let Err(error) = app::run(&cli) {
        diagnostics::report_fatal(&error);
        diagnostics::show_startup_error_dialog(&error);
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

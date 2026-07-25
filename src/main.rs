mod app;
mod app_icon;
mod cli;
mod diagnostics;
mod error;
mod gpu;
mod units;

use clap::Parser;

#[cfg(target_os = "linux")]
const APPLICATION_ID: &str = "io.github.andrinbr.xl_view";
const APPLICATION_NAME: &str = "XL-View";

fn main() {
    diagnostics::initialize_logging();
    diagnostics::install_panic_hook();

    let cli = cli::Cli::parse();
    if let Err(error) = app::run(&cli) {
        diagnostics::report_fatal(&error);
        diagnostics::show_startup_error_dialog(&error);
        std::process::exit(1);
    }
}

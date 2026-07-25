use std::path::PathBuf;

use clap::{Parser, ValueEnum};

const DEFAULT_GPU_MEMORY_MIB: u64 = 1024;

#[derive(Debug, Parser)]
#[command(version, about)]
pub struct Cli {
    /// Image to open.
    pub image: Option<PathBuf>,

    /// Show a test pattern and print runtime/display diagnostics.
    #[arg(long)]
    pub diagnostics: bool,

    /// Select the display output encoding.
    #[arg(long, value_enum, default_value = "auto")]
    pub output: OutputMode,

    /// Select the background used outside and behind transparent image pixels.
    #[arg(long, value_enum, default_value = "black")]
    pub background: BackgroundMode,

    /// Decoded-image cache budget in MiB [default: 25% of system RAM, at least 2048, a value of 0 disables the cache]
    #[arg(long, value_name = "MIB")]
    pub cache: Option<u64>,

    /// Maximum GPU image memory in MiB
    #[arg(long, value_name = "MIB", default_value_t = DEFAULT_GPU_MEMORY_MIB, value_parser = clap::value_parser!(u64).range(1..))]
    pub gpu_memory: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum OutputMode {
    #[default]
    Auto,
    Pq,
    Hlg,
    Scrgb,
    Sdr,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum BackgroundMode {
    #[default]
    Black,
    MiddleGray,
    White,
    Checkerboard,
}

impl BackgroundMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Black => "black",
            Self::Checkerboard => "checkerboard",
            Self::White => "white",
            Self::MiddleGray => "middle-gray",
        }
    }
}

impl OutputMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Pq => "pq",
            Self::Hlg => "hlg",
            Self::Scrgb => "scrgb",
            Self::Sdr => "sdr",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_defaults_to_auto() {
        let cli = Cli::try_parse_from(["xl-view"]).unwrap();
        assert_eq!(cli.output, OutputMode::Auto);
        assert_eq!(cli.background, BackgroundMode::Black);
        assert_eq!(cli.cache, None);
        assert_eq!(cli.gpu_memory, 1024);
        assert!(cli.image.is_none());
        assert!(!cli.diagnostics);
    }

    #[test]
    fn parses_every_output_mode() {
        for (argument, expected) in [
            ("auto", OutputMode::Auto),
            ("pq", OutputMode::Pq),
            ("hlg", OutputMode::Hlg),
            ("scrgb", OutputMode::Scrgb),
            ("sdr", OutputMode::Sdr),
        ] {
            let cli = Cli::try_parse_from(["xl-view", "--output", argument]).unwrap();
            assert_eq!(cli.output, expected);
        }
    }

    #[test]
    fn parses_every_background_mode() {
        for (argument, expected) in [
            ("black", BackgroundMode::Black),
            ("checkerboard", BackgroundMode::Checkerboard),
            ("white", BackgroundMode::White),
            ("middle-gray", BackgroundMode::MiddleGray),
        ] {
            let cli = Cli::try_parse_from(["xl-view", "--background", argument]).unwrap();
            assert_eq!(cli.background, expected);
        }
    }

    #[test]
    fn parses_development_options_and_image_path() {
        let cli = Cli::try_parse_from([
            "xl-view",
            "--diagnostics",
            "--background",
            "middle-gray",
            "--cache",
            "3072",
            "--gpu-memory",
            "768",
            "image.jxl",
        ])
        .unwrap();

        assert!(cli.diagnostics);
        assert_eq!(cli.background, BackgroundMode::MiddleGray);
        assert_eq!(cli.cache, Some(3072));
        assert_eq!(cli.gpu_memory, 768);
        assert_eq!(cli.image, Some(PathBuf::from("image.jxl")));
    }

    #[test]
    fn rejects_zero_gpu_memory() {
        assert!(Cli::try_parse_from(["xl-view", "--gpu-memory", "0"]).is_err());
    }

    #[test]
    fn zero_cache_disables_reuse() {
        let cli = Cli::try_parse_from(["xl-view", "--cache", "0"]).unwrap();
        assert_eq!(cli.cache, Some(0));
    }
}

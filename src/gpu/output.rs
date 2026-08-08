//! Pure surface-selection, format-ranking, luminance, and HDR-metadata policy
//! shared by the stateful GPU renderer.

use std::error::Error;
use std::fmt;

#[cfg(test)]
use wgpu::SurfaceColorSpaces;
use wgpu::{SurfaceColorSpace, SurfaceFormatCapabilities, TextureFormat};
use winit::dpi::PhysicalSize;

use super::hdr_metadata::HdrMetadata;
use crate::cli::{BackgroundMode, OutputMode};
use xl_view::color::HDR_REFERENCE_WHITE_NITS;
use xl_view::decode::SourceDynamicRange;

/// Absolute peak of the ST 2084/PQ encoding
pub(super) const PQ_ENCODING_PEAK_NITS: f32 = 10_000.0;
/// HLG encoding peak used for system gamma and metadata
pub(super) const HLG_ENCODING_PEAK_NITS: f32 = 1_000.0;
/// Source mastering peak used when HDR metadata is missing or invalid
pub(super) const FALLBACK_HDR_SOURCE_PEAK_NITS: f32 = 1_000.0;
/// Working output peak assumed for an extended-linear surface
pub(super) const EXTENDED_LINEAR_FALLBACK_PEAK_NITS: f32 = 1_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderingOptions {
    pub exposure_stops: f32,
    pub background: BackgroundMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SurfaceCandidate {
    pub(super) color_space: SurfaceColorSpace,
    pub(super) format: TextureFormat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SurfaceOutputError {
    NoUsablePair,
    RequestedModeUnavailable(OutputMode),
}

impl fmt::Display for SurfaceOutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoUsablePair => formatter.write_str(
                "the GPU surface advertises no usable output format/color-space pair; check the display connection and GPU driver",
            ),
            Self::RequestedModeUnavailable(OutputMode::Sdr) => formatter.write_str(
                "requested SDR output mode is unavailable on this display; use '--output auto' or check the display and GPU driver",
            ),
            Self::RequestedModeUnavailable(OutputMode::Auto) => {
                unreachable!("automatic output failure uses NoUsablePair")
            }
            Self::RequestedModeUnavailable(mode) => write!(
                formatter,
                "requested HDR output mode '{}' is unavailable on this display; use '--output auto' or '--output sdr'",
                mode.as_str()
            ),
        }
    }
}

impl Error for SurfaceOutputError {}

pub(super) fn is_hdr_color_space(color_space: SurfaceColorSpace) -> bool {
    matches!(
        color_space,
        SurfaceColorSpace::Bt2100Pq
            | SurfaceColorSpace::Bt2100Hlg
            | SurfaceColorSpace::ExtendedSrgbLinear
    )
}

pub(super) fn surface_configuration(
    candidate: SurfaceCandidate,
    size: PhysicalSize<u32>,
) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: candidate.format,
        color_space: candidate.color_space,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
    }
}

pub(super) fn surface_hdr_metadata(
    candidate: SurfaceCandidate,
    rendering_options: RenderingOptions,
    source_dynamic_range: SourceDynamicRange,
    source_peak_nits: f32,
    source_min_nits: f32,
) -> Option<HdrMetadata> {
    if !is_hdr_color_space(candidate.color_space) {
        return None;
    }

    let source_mastering_peak = if source_dynamic_range == SourceDynamicRange::Sdr {
        HDR_REFERENCE_WHITE_NITS
    } else if source_peak_nits.is_finite() && source_peak_nits > 0.0 {
        source_peak_nits
    } else {
        FALLBACK_HDR_SOURCE_PEAK_NITS
    };
    let exposure = if rendering_options.exposure_stops.is_finite() {
        rendering_options.exposure_stops.exp2()
    } else {
        1.0
    };
    // JPEG XL's intensity target is an upper bound rather than a measured
    // content maximum. Using it avoids another full pass over the decoded
    // pixels. The surface also carries viewer UI at HDR reference white.
    let max_cll = (source_mastering_peak * exposure).max(HDR_REFERENCE_WHITE_NITS);
    let encoding_peak = match candidate.color_space {
        SurfaceColorSpace::Bt2100Pq => PQ_ENCODING_PEAK_NITS,
        SurfaceColorSpace::Bt2100Hlg => HLG_ENCODING_PEAK_NITS,
        SurfaceColorSpace::ExtendedSrgbLinear => f32::INFINITY,
        _ => return None,
    };
    let max_cll = max_cll.min(encoding_peak);
    let mastering_peak = source_mastering_peak.max(max_cll).min(encoding_peak);

    Some(HdrMetadata::bt2020(
        mastering_peak,
        finite_non_negative(source_min_nits),
        max_cll,
        0.0,
    ))
}

pub(super) fn finite_non_negative(value: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

pub(super) fn shader_mode(color_space: SurfaceColorSpace) -> u32 {
    match color_space {
        SurfaceColorSpace::Srgb => 0,
        SurfaceColorSpace::ExtendedSrgbLinear => 1,
        SurfaceColorSpace::Bt2100Pq => 2,
        SurfaceColorSpace::Bt2100Hlg => 3,
        _ => unreachable!("surface selection returned an unsupported color space"),
    }
}

pub(super) fn quantization_bits(format: TextureFormat) -> u8 {
    match format {
        TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb => 8,
        TextureFormat::Rgb10a2Unorm => 10,
        _ => 0,
    }
}

pub(super) fn hdr_mapping_summary(
    hdr_surface: bool,
    source_dynamic_range: SourceDynamicRange,
) -> &'static str {
    if hdr_surface {
        "Compositor"
    } else if source_dynamic_range.is_hdr() {
        "Application tone mapping to SDR"
    } else {
        "Not required"
    }
}

pub(super) fn resolved_ui_white_nits(
    color_space: SurfaceColorSpace,
    hdr_info: &wgpu::DisplayHdrInfo,
) -> f32 {
    if !is_hdr_color_space(color_space) {
        // Image and UI share a fixed relative-white scale on non-HDR outputs.
        return HDR_REFERENCE_WHITE_NITS;
    }
    hdr_info
        .luminance
        .as_ref()
        .and_then(|luminance| luminance.sdr_white_nits)
        .filter(|white| white.is_finite() && *white > 0.0)
        .unwrap_or(HDR_REFERENCE_WHITE_NITS)
}

pub(super) fn resolved_output_peak_nits(color_space: SurfaceColorSpace) -> f32 {
    match color_space {
        // SDR surfaces carry relative values. The OS-reported SDR white is
        // where an HDR desktop places that surface, not the surface's peak.
        SurfaceColorSpace::Srgb => HDR_REFERENCE_WHITE_NITS,
        SurfaceColorSpace::Bt2100Pq => PQ_ENCODING_PEAK_NITS,
        SurfaceColorSpace::Bt2100Hlg => HLG_ENCODING_PEAK_NITS,
        SurfaceColorSpace::ExtendedSrgbLinear => EXTENDED_LINEAR_FALLBACK_PEAK_NITS,
        _ => unreachable!("surface selection returned an unsupported color space"),
    }
}

pub(super) fn select_required_surface_candidate(
    capabilities: &[SurfaceFormatCapabilities],
    output_mode: OutputMode,
    source_dynamic_range: SourceDynamicRange,
) -> Result<SurfaceCandidate, SurfaceOutputError> {
    select_surface_candidate(capabilities, output_mode, source_dynamic_range).ok_or(
        match output_mode {
            OutputMode::Auto => SurfaceOutputError::NoUsablePair,
            forced => SurfaceOutputError::RequestedModeUnavailable(forced),
        },
    )
}

fn select_surface_candidate(
    capabilities: &[SurfaceFormatCapabilities],
    output_mode: OutputMode,
    source_dynamic_range: SourceDynamicRange,
) -> Option<SurfaceCandidate> {
    const AUTO_PQ_PRIORITY: [SurfaceColorSpace; 4] = [
        SurfaceColorSpace::Bt2100Pq,
        SurfaceColorSpace::Bt2100Hlg,
        SurfaceColorSpace::ExtendedSrgbLinear,
        SurfaceColorSpace::Srgb,
    ];
    const AUTO_HLG_PRIORITY: [SurfaceColorSpace; 4] = [
        SurfaceColorSpace::Bt2100Hlg,
        SurfaceColorSpace::Bt2100Pq,
        SurfaceColorSpace::ExtendedSrgbLinear,
        SurfaceColorSpace::Srgb,
    ];
    const AUTO_SDR_PRIORITY: [SurfaceColorSpace; 4] = [
        SurfaceColorSpace::Srgb,
        SurfaceColorSpace::Bt2100Pq,
        SurfaceColorSpace::Bt2100Hlg,
        SurfaceColorSpace::ExtendedSrgbLinear,
    ];
    const PQ_ONLY: [SurfaceColorSpace; 1] = [SurfaceColorSpace::Bt2100Pq];
    const HLG_ONLY: [SurfaceColorSpace; 1] = [SurfaceColorSpace::Bt2100Hlg];
    const SCRGB_ONLY: [SurfaceColorSpace; 1] = [SurfaceColorSpace::ExtendedSrgbLinear];
    const SDR_ONLY: [SurfaceColorSpace; 1] = [SurfaceColorSpace::Srgb];

    let color_spaces = match output_mode {
        OutputMode::Auto => match source_dynamic_range {
            SourceDynamicRange::Sdr => &AUTO_SDR_PRIORITY[..],
            SourceDynamicRange::Pq => &AUTO_PQ_PRIORITY[..],
            SourceDynamicRange::Hlg => &AUTO_HLG_PRIORITY[..],
        },
        OutputMode::Pq => &PQ_ONLY[..],
        OutputMode::Hlg => &HLG_ONLY[..],
        OutputMode::Scrgb => &SCRGB_ONLY[..],
        OutputMode::Sdr => &SDR_ONLY[..],
    };

    color_spaces
        .iter()
        .copied()
        .find_map(|color_space| best_format_for_color_space(capabilities, color_space))
}

fn best_format_for_color_space(
    capabilities: &[SurfaceFormatCapabilities],
    color_space: SurfaceColorSpace,
) -> Option<SurfaceCandidate> {
    let color_space_flag = color_space
        .to_color_spaces()
        .expect("the candidate list must not contain Auto");

    capabilities
        .iter()
        .enumerate()
        .filter(|(_, capability)| capability.color_spaces.contains(color_space_flag))
        .filter_map(|(preference, capability)| {
            format_rank(capability.format, color_space)
                .map(|format_rank| (format_rank, preference, capability.format))
        })
        .min_by_key(|(format_rank, preference, _)| (*format_rank, *preference))
        .map(|(_, _, format)| SurfaceCandidate {
            color_space,
            format,
        })
}

fn format_rank(format: TextureFormat, color_space: SurfaceColorSpace) -> Option<u8> {
    match color_space {
        SurfaceColorSpace::Bt2100Pq | SurfaceColorSpace::Bt2100Hlg => match format {
            TextureFormat::Rgb10a2Unorm => Some(0),
            TextureFormat::Rgba16Float => Some(1),
            TextureFormat::Rgba16Unorm => Some(2),
            TextureFormat::Rgba32Float => Some(3),
            _ => None,
        },
        SurfaceColorSpace::ExtendedSrgbLinear => match format {
            TextureFormat::Rgba16Float => Some(0),
            TextureFormat::Rgba32Float => Some(1),
            _ => None,
        },
        SurfaceColorSpace::Srgb => {
            if format.is_srgb() {
                Some(0)
            } else {
                match format {
                    TextureFormat::Bgra8Unorm | TextureFormat::Rgba8Unorm => Some(1),
                    TextureFormat::Rgb10a2Unorm => Some(2),
                    TextureFormat::Rgba16Unorm => Some(3),
                    TextureFormat::Rgba16Float => Some(4),
                    TextureFormat::Rgba32Float => Some(5),
                    _ => None,
                }
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;

//! Pure surface-selection and presentation policy shared by the GPU renderer.

#[cfg(test)]
use wgpu::SurfaceColorSpaces;
use wgpu::{SurfaceColorSpace, SurfaceFormatCapabilities, TextureFormat};
use winit::dpi::PhysicalSize;

use super::display_target::{DISPLAY_INFORMATION_SOURCE, HdrTarget, HdrUnavailableReason};
use crate::cli::{BackgroundMode, OutputMode};
use xl_view::color::HDR_REFERENCE_WHITE_NITS;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SurfaceSelection {
    candidate: SurfaceCandidate,
    display_target: HdrTarget,
    output_peak_nits: f32,
}

impl SurfaceSelection {
    const fn pq(candidate: SurfaceCandidate, peak_nits: f32) -> Self {
        Self {
            candidate,
            display_target: Ok(peak_nits),
            output_peak_nits: peak_nits,
        }
    }

    const fn srgb(candidate: SurfaceCandidate, display_target: HdrTarget) -> Self {
        Self {
            candidate,
            display_target,
            output_peak_nits: HDR_REFERENCE_WHITE_NITS,
        }
    }

    pub(super) const fn candidate(self) -> SurfaceCandidate {
        self.candidate
    }

    pub(super) const fn display_target(self) -> HdrTarget {
        self.display_target
    }

    pub(super) const fn output_peak_nits(self) -> f32 {
        self.output_peak_nits
    }

    pub(super) fn requires_surface_reconfiguration(self, current: Self) -> bool {
        self.candidate != current.candidate
    }

    pub(super) fn with_display_target(
        self,
        display_target: HdrTarget,
    ) -> Result<Self, SurfaceOutputError> {
        if self.candidate.color_space == SurfaceColorSpace::Bt2100Pq {
            display_target
                .map(|target| Self::pq(self.candidate, target))
                .map_err(SurfaceOutputError::HdrTargetUnavailable)
        } else {
            Ok(Self::srgb(self.candidate, display_target))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[expect(clippy::enum_variant_names)] // Each planned name identifies the unavailable resource.
pub(crate) enum SurfaceOutputError {
    #[error(
        "requested PQ output is unavailable because the GPU surface does not advertise BT.2100 PQ with Rgb10a2Unorm; use '--output auto' or '--output sdr'"
    )]
    PqSurfaceUnavailable,
    #[error(
        "requested PQ output is unavailable from {provider}: {0}; use '--output auto' or '--output sdr'",
        provider = DISPLAY_INFORMATION_SOURCE
    )]
    HdrTargetUnavailable(HdrUnavailableReason),
    #[error(
        "sRGB output is unavailable because the GPU surface does not advertise an sRGB 8-bit format/color-space pair; check the display connection and GPU driver"
    )]
    SrgbSurfaceUnavailable,
}

pub(super) fn is_hdr_color_space(color_space: SurfaceColorSpace) -> bool {
    color_space == SurfaceColorSpace::Bt2100Pq
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

pub(super) fn select_required_surface(
    capabilities: &[SurfaceFormatCapabilities],
    output_mode: OutputMode,
    display_target: HdrTarget,
) -> Result<SurfaceSelection, SurfaceOutputError> {
    match output_mode {
        OutputMode::Auto => {
            if let Ok(target) = display_target
                && let Some(candidate) = pq_candidate(capabilities)
            {
                return Ok(SurfaceSelection::pq(candidate, target));
            }
            srgb_candidate(capabilities)
                .map(|candidate| SurfaceSelection::srgb(candidate, display_target))
                .ok_or(SurfaceOutputError::SrgbSurfaceUnavailable)
        }
        OutputMode::Pq => {
            let target = display_target.map_err(SurfaceOutputError::HdrTargetUnavailable)?;
            pq_candidate(capabilities)
                .map(|candidate| SurfaceSelection::pq(candidate, target))
                .ok_or(SurfaceOutputError::PqSurfaceUnavailable)
        }
        OutputMode::Sdr => srgb_candidate(capabilities)
            .map(|candidate| SurfaceSelection::srgb(candidate, display_target))
            .ok_or(SurfaceOutputError::SrgbSurfaceUnavailable),
    }
}

fn pq_candidate(capabilities: &[SurfaceFormatCapabilities]) -> Option<SurfaceCandidate> {
    candidate_with_format(
        capabilities,
        SurfaceColorSpace::Bt2100Pq,
        TextureFormat::Rgb10a2Unorm,
    )
}

fn srgb_candidate(capabilities: &[SurfaceFormatCapabilities]) -> Option<SurfaceCandidate> {
    [TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba8UnormSrgb]
        .into_iter()
        .find_map(|format| candidate_with_format(capabilities, SurfaceColorSpace::Srgb, format))
}

fn candidate_with_format(
    capabilities: &[SurfaceFormatCapabilities],
    color_space: SurfaceColorSpace,
    format: TextureFormat,
) -> Option<SurfaceCandidate> {
    let required_color_space = color_space
        .to_color_spaces()
        .expect("surface candidates never use the automatic color space");
    capabilities
        .iter()
        .any(|capability| {
            capability.format == format && capability.color_spaces.contains(required_color_space)
        })
        .then_some(SurfaceCandidate {
            color_space,
            format,
        })
}

#[cfg(test)]
mod tests;

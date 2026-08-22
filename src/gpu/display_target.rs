//! Display-target normalization shared by the platform-specific providers.

use std::fmt;

#[cfg(any(test, target_os = "windows", target_os = "linux"))]
use xl_view::color::HDR_REFERENCE_WHITE_NITS;

/// PQ reference white used to convert macOS EDR headroom to absolute luminance.
///
/// Apple defines PQ-to-EDR conversion relative to 100 cd/m², independently of
/// the application's BT.2100/BT.2408 working-space reference white.
#[cfg(any(test, target_os = "macos"))]
const MACOS_PQ_REFERENCE_WHITE_NITS: f32 = 100.0;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "linux")]
mod wayland;

#[cfg(target_os = "macos")]
pub(super) use macos::DisplayTargetProvider;
#[cfg(target_os = "linux")]
pub(super) use wayland::DisplayTargetProvider;

#[cfg(target_os = "windows")]
pub(super) const DISPLAY_INFORMATION_SOURCE: &str = "Windows DXGI";
#[cfg(target_os = "macos")]
pub(super) const DISPLAY_INFORMATION_SOURCE: &str = "macOS EDR";
#[cfg(target_os = "linux")]
pub(super) const DISPLAY_INFORMATION_SOURCE: &str = "Wayland color-management-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HdrUnavailableReason {
    DisplayReportsSdr,
    #[cfg(target_os = "linux")]
    MissingColorManagement,
    MissingDisplayInformation,
    InvalidLuminance,
    #[cfg(target_os = "linux")]
    OutputUnmatched,
}

impl fmt::Display for HdrUnavailableReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DisplayReportsSdr => "display reports no usable HDR headroom",
            #[cfg(target_os = "linux")]
            Self::MissingColorManagement => "color-management-v1 is unavailable",
            Self::MissingDisplayInformation => "display luminance information is unavailable",
            Self::InvalidLuminance => "display luminance information is invalid",
            #[cfg(target_os = "linux")]
            Self::OutputUnmatched => "window monitor does not match a Wayland output",
        })
    }
}

pub(super) type HdrTarget = Result<f32, HdrUnavailableReason>;

#[cfg(target_os = "windows")]
pub(super) struct DisplayTargetProvider;

#[cfg(target_os = "windows")]
impl DisplayTargetProvider {
    // DXGI display state is queried when a frame is rendered.
    pub(super) const QUERY_BEFORE_RENDER: bool = true;

    pub(super) fn new(_: super::DisplayChangeNotifier) -> Self {
        Self
    }

    #[allow(clippy::unused_self)] // Keeps target queries uniform across platform providers.
    pub(super) fn target(&self, info: &wgpu::DisplayHdrInfo) -> HdrTarget {
        let luminance = info.luminance.unwrap_or_default();
        windows_target(
            info.coarse.and_then(|coarse| coarse.high_dynamic_range),
            luminance.sdr_white_nits,
            luminance.max_nits,
        )
    }
}

#[cfg(any(test, target_os = "windows", target_os = "linux"))]
fn target_above_reference_white(peak_nits: f32) -> HdrTarget {
    if peak_nits <= HDR_REFERENCE_WHITE_NITS {
        return Err(HdrUnavailableReason::DisplayReportsSdr);
    }
    Ok(peak_nits)
}

#[cfg(any(test, target_os = "windows"))]
pub(super) fn windows_target(
    hdr_active: Option<bool>,
    sdr_white_nits: Option<f32>,
    max_nits: Option<f32>,
) -> HdrTarget {
    if hdr_active != Some(true) {
        return Err(HdrUnavailableReason::DisplayReportsSdr);
    }
    let (Some(sdr_white_nits), Some(max_nits)) = (sdr_white_nits, max_nits) else {
        return Err(HdrUnavailableReason::MissingDisplayInformation);
    };
    if !sdr_white_nits.is_finite()
        || sdr_white_nits <= 0.0
        || !max_nits.is_finite()
        || max_nits <= 0.0
    {
        return Err(HdrUnavailableReason::InvalidLuminance);
    }
    if max_nits <= sdr_white_nits {
        return Err(HdrUnavailableReason::DisplayReportsSdr);
    }
    target_above_reference_white(max_nits)
}

#[cfg(any(test, target_os = "macos"))]
pub(super) fn macos_target(
    potential_headroom: Option<f32>,
    current_headroom: Option<f32>,
) -> HdrTarget {
    let (Some(potential_headroom), Some(current_headroom)) = (potential_headroom, current_headroom)
    else {
        return Err(HdrUnavailableReason::MissingDisplayInformation);
    };
    if !potential_headroom.is_finite() || potential_headroom <= 0.0 || !current_headroom.is_finite()
    {
        return Err(HdrUnavailableReason::InvalidLuminance);
    }
    if potential_headroom <= 1.0 {
        return Err(HdrUnavailableReason::DisplayReportsSdr);
    }
    let peak_nits = MACOS_PQ_REFERENCE_WHITE_NITS * current_headroom.max(1.0);
    if !peak_nits.is_finite() {
        return Err(HdrUnavailableReason::InvalidLuminance);
    }
    Ok(peak_nits)
}

#[cfg(any(test, target_os = "linux"))]
#[allow(clippy::cast_precision_loss)] // Protocol luminances are far below f32's exact-integer limit.
pub(super) fn wayland_target(reference_lum: Option<u32>, max_lum: Option<u32>) -> HdrTarget {
    let (Some(reference_lum), Some(max_lum)) = (reference_lum, max_lum) else {
        return Err(HdrUnavailableReason::MissingDisplayInformation);
    };
    if reference_lum == 0 || max_lum == 0 {
        return Err(HdrUnavailableReason::InvalidLuminance);
    }
    if max_lum <= reference_lum {
        return Err(HdrUnavailableReason::DisplayReportsSdr);
    }
    target_above_reference_white(max_lum as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reason(target: HdrTarget) -> HdrUnavailableReason {
        target.unwrap_err()
    }

    #[test]
    fn windows_requires_active_hdr_and_complete_finite_headroom() {
        assert_eq!(
            reason(windows_target(None, Some(203.0), Some(600.0))),
            HdrUnavailableReason::DisplayReportsSdr
        );
        assert_eq!(
            reason(windows_target(Some(false), Some(203.0), Some(600.0))),
            HdrUnavailableReason::DisplayReportsSdr
        );
        assert_eq!(
            reason(windows_target(Some(true), None, Some(600.0))),
            HdrUnavailableReason::MissingDisplayInformation
        );
        assert_eq!(
            reason(windows_target(Some(true), Some(203.0), None)),
            HdrUnavailableReason::MissingDisplayInformation
        );
        for invalid in [f32::NAN, f32::INFINITY, 0.0] {
            assert_eq!(
                reason(windows_target(Some(true), Some(invalid), Some(600.0))),
                HdrUnavailableReason::InvalidLuminance
            );
            assert_eq!(
                reason(windows_target(Some(true), Some(203.0), Some(invalid))),
                HdrUnavailableReason::InvalidLuminance
            );
        }
        assert_eq!(
            reason(windows_target(Some(true), Some(600.0), Some(600.0))),
            HdrUnavailableReason::DisplayReportsSdr
        );
        assert_eq!(
            reason(windows_target(Some(true), Some(100.0), Some(203.0))),
            HdrUnavailableReason::DisplayReportsSdr
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // These normalization cases are exactly representable.
    fn windows_reference_white_is_validation_only() {
        let first = windows_target(Some(true), Some(100.0), Some(600.0)).unwrap();
        let second = windows_target(Some(true), Some(300.0), Some(600.0)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first, 600.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // The headroom fixtures have exact binary products.
    fn macos_bootstraps_with_potential_and_maps_pq_from_100_nits() {
        assert_eq!(
            reason(macos_target(Some(1.0), Some(2.0))),
            HdrUnavailableReason::DisplayReportsSdr
        );
        assert_eq!(
            reason(macos_target(Some(2.0), None)),
            HdrUnavailableReason::MissingDisplayInformation
        );
        assert_eq!(
            reason(macos_target(None, Some(1.0))),
            HdrUnavailableReason::MissingDisplayInformation
        );
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                reason(macos_target(Some(invalid), Some(1.0))),
                HdrUnavailableReason::InvalidLuminance
            );
            assert_eq!(
                reason(macos_target(Some(2.0), Some(invalid))),
                HdrUnavailableReason::InvalidLuminance
            );
        }
        assert_eq!(
            reason(macos_target(Some(2.0), Some(f32::MAX))),
            HdrUnavailableReason::InvalidLuminance
        );
        assert_eq!(macos_target(Some(2.0), Some(1.0)).unwrap(), 100.0);
        assert_eq!(macos_target(Some(2.0), Some(0.0)).unwrap(), 100.0);
        assert_eq!(macos_target(Some(4.0), Some(2.0)).unwrap(), 200.0);
        assert_eq!(macos_target(Some(4.0), Some(1.5)).unwrap(), 150.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Integer protocol luminances convert exactly here.
    fn wayland_uses_integer_luminance_and_fixed_white() {
        assert_eq!(
            reason(wayland_target(None, Some(600))),
            HdrUnavailableReason::MissingDisplayInformation
        );
        assert_eq!(
            reason(wayland_target(Some(200), None)),
            HdrUnavailableReason::MissingDisplayInformation
        );
        assert_eq!(
            reason(wayland_target(Some(0), Some(600))),
            HdrUnavailableReason::InvalidLuminance
        );
        assert_eq!(
            reason(wayland_target(Some(600), Some(600))),
            HdrUnavailableReason::DisplayReportsSdr
        );
        assert_eq!(
            reason(wayland_target(Some(100), Some(203))),
            HdrUnavailableReason::DisplayReportsSdr
        );
        let first = wayland_target(Some(100), Some(600)).unwrap();
        let second = wayland_target(Some(300), Some(600)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first, 600.0);
    }
}

use super::*;
use crate::gpu::display_target::HdrUnavailableReason;

fn capability(
    format: TextureFormat,
    color_spaces: SurfaceColorSpaces,
) -> SurfaceFormatCapabilities {
    SurfaceFormatCapabilities {
        format,
        color_spaces,
    }
}

fn unavailable_target() -> HdrTarget {
    Err(HdrUnavailableReason::MissingDisplayInformation)
}

#[test]
fn automatic_output_requires_both_target_and_pq_pair() {
    let capabilities = [
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ),
        capability(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
    ];
    assert_eq!(
        select_required_surface(&capabilities, OutputMode::Auto, Ok(600.0)).unwrap(),
        SurfaceSelection::pq(
            SurfaceCandidate {
                color_space: SurfaceColorSpace::Bt2100Pq,
                format: TextureFormat::Rgb10a2Unorm,
            },
            600.0,
        )
    );
    assert_eq!(
        select_required_surface(&capabilities, OutputMode::Auto, unavailable_target())
            .unwrap()
            .candidate()
            .color_space,
        SurfaceColorSpace::Srgb
    );
    assert_eq!(
        select_required_surface(&capabilities[1..], OutputMode::Auto, Ok(600.0))
            .unwrap()
            .candidate()
            .color_space,
        SurfaceColorSpace::Srgb
    );
}

#[test]
fn forced_modes_are_strict_and_report_distinct_failures() {
    let pq_only = [capability(
        TextureFormat::Rgb10a2Unorm,
        SurfaceColorSpaces::BT2100_PQ,
    )];
    let srgb_only = [capability(
        TextureFormat::Bgra8UnormSrgb,
        SurfaceColorSpaces::SRGB,
    )];
    assert_eq!(
        select_required_surface(&pq_only, OutputMode::Pq, unavailable_target()),
        Err(SurfaceOutputError::HdrTargetUnavailable(
            unavailable_target().unwrap_err()
        ))
    );
    assert_eq!(
        select_required_surface(&srgb_only, OutputMode::Pq, Ok(600.0)),
        Err(SurfaceOutputError::PqSurfaceUnavailable)
    );
    assert_eq!(
        select_required_surface(&pq_only, OutputMode::Sdr, Ok(600.0)),
        Err(SurfaceOutputError::SrgbSurfaceUnavailable)
    );
}

#[test]
fn legacy_color_spaces_and_formats_are_rejected() {
    let capabilities = [
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_HLG),
        capability(
            TextureFormat::Rgba16Float,
            SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
        ),
        capability(TextureFormat::Bgra8Unorm, SurfaceColorSpaces::SRGB),
        capability(TextureFormat::Rgba8Unorm, SurfaceColorSpaces::SRGB),
        capability(TextureFormat::Rgba16Float, SurfaceColorSpaces::BT2100_PQ),
    ];
    assert_eq!(
        select_required_surface(&capabilities, OutputMode::Auto, Ok(600.0)),
        Err(SurfaceOutputError::SrgbSurfaceUnavailable)
    );
}

#[test]
fn srgb_format_preference_is_stable() {
    let capabilities = [
        capability(TextureFormat::Rgba8UnormSrgb, SurfaceColorSpaces::SRGB),
        capability(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
    ];
    assert_eq!(
        select_required_surface(&capabilities, OutputMode::Sdr, unavailable_target())
            .unwrap()
            .candidate()
            .format,
        TextureFormat::Bgra8UnormSrgb
    );
}

#[test]
#[allow(clippy::float_cmp)] // The selected target and fixed white are exact fixtures.
fn selection_carries_a_validated_presentation_peak() {
    let pq_capabilities = [capability(
        TextureFormat::Rgb10a2Unorm,
        SurfaceColorSpaces::BT2100_PQ,
    )];
    let pq = select_required_surface(&pq_capabilities, OutputMode::Pq, Ok(600.0)).unwrap();
    assert_eq!(pq.output_peak_nits(), 600.0);
    assert_eq!(pq.display_target(), Ok(600.0));
    let refreshed_pq = pq.with_display_target(Ok(1_000.0)).unwrap();
    assert_eq!(refreshed_pq.candidate(), pq.candidate());
    assert_eq!(refreshed_pq.output_peak_nits(), 1_000.0);
    assert_eq!(
        pq.with_display_target(unavailable_target()),
        Err(SurfaceOutputError::HdrTargetUnavailable(
            unavailable_target().unwrap_err()
        ))
    );

    let srgb_capabilities = [capability(
        TextureFormat::Bgra8UnormSrgb,
        SurfaceColorSpaces::SRGB,
    )];
    let srgb =
        select_required_surface(&srgb_capabilities, OutputMode::Sdr, unavailable_target()).unwrap();
    assert_eq!(srgb.output_peak_nits(), HDR_REFERENCE_WHITE_NITS);
    assert_eq!(
        srgb.with_display_target(Ok(1_000.0))
            .unwrap()
            .output_peak_nits(),
        HDR_REFERENCE_WHITE_NITS
    );
}

#[test]
fn reselecting_the_same_candidate_does_not_require_surface_reconfiguration() {
    let capabilities = [
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ),
        capability(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
    ];
    let current = select_required_surface(&capabilities, OutputMode::Auto, Ok(600.0)).unwrap();
    let same = select_required_surface(&capabilities, OutputMode::Auto, Ok(600.0)).unwrap();
    let new_peak = select_required_surface(&capabilities, OutputMode::Auto, Ok(1_000.0)).unwrap();
    let sdr =
        select_required_surface(&capabilities, OutputMode::Auto, unavailable_target()).unwrap();

    assert!(!same.requires_surface_reconfiguration(current));
    assert!(!new_peak.requires_surface_reconfiguration(current));
    assert!(sdr.requires_surface_reconfiguration(current));
}

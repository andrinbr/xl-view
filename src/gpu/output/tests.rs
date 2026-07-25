use super::*;
use xl_view::color::HDR_REFERENCE_WHITE_NITS;

const RENDERING_OPTIONS: RenderingOptions = RenderingOptions {
    exposure_stops: 0.0,
    background: BackgroundMode::Black,
};

fn capability(
    format: TextureFormat,
    color_spaces: SurfaceColorSpaces,
) -> SurfaceFormatCapabilities {
    SurfaceFormatCapabilities {
        format,
        color_spaces,
    }
}

#[test]
fn hdr_candidate_detection_requires_a_usable_encoding() {
    let unusable = [
        capability(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
        capability(TextureFormat::Rgba8Unorm, SurfaceColorSpaces::BT2100_PQ),
    ];
    assert!(!has_hdr_encoding_candidate(&unusable));

    for candidate in [
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ),
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_HLG),
        capability(
            TextureFormat::Rgba16Float,
            SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
        ),
    ] {
        assert!(has_hdr_encoding_candidate(&[candidate]));
    }
}

#[test]
fn pq_ten_bit_beats_advertised_eight_bit_and_other_color_spaces() {
    let capabilities = [
        capability(TextureFormat::Rgba8Unorm, SurfaceColorSpaces::BT2100_PQ),
        capability(
            TextureFormat::Rgba16Float,
            SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
        ),
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ),
    ];

    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Auto, SourceDynamicRange::Pq),
        Some(SurfaceCandidate {
            format: TextureFormat::Rgb10a2Unorm,
            color_space: SurfaceColorSpace::Bt2100Pq,
        })
    );
}

#[test]
fn automatic_output_prefers_the_matching_source_encoding() {
    let capabilities = [
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ),
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_HLG),
        capability(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
    ];

    for (source_dynamic_range, format, color_space) in [
        (
            SourceDynamicRange::Sdr,
            TextureFormat::Bgra8UnormSrgb,
            SurfaceColorSpace::Srgb,
        ),
        (
            SourceDynamicRange::Pq,
            TextureFormat::Rgb10a2Unorm,
            SurfaceColorSpace::Bt2100Pq,
        ),
        (
            SourceDynamicRange::Hlg,
            TextureFormat::Rgb10a2Unorm,
            SurfaceColorSpace::Bt2100Hlg,
        ),
    ] {
        assert_eq!(
            select_surface_candidate(&capabilities, OutputMode::Auto, source_dynamic_range,),
            Some(SurfaceCandidate {
                color_space,
                format
            })
        );
    }
}

#[test]
fn hlg_falls_back_to_pq_when_hlg_has_no_suitable_format() {
    let capabilities = [
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ),
        capability(TextureFormat::Rgba8Unorm, SurfaceColorSpaces::BT2100_HLG),
    ];

    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Auto, SourceDynamicRange::Hlg),
        Some(SurfaceCandidate {
            format: TextureFormat::Rgb10a2Unorm,
            color_space: SurfaceColorSpace::Bt2100Pq,
        })
    );
}

#[test]
fn format_and_color_space_capabilities_are_never_mixed_across_pairs() {
    let capabilities = [
        capability(TextureFormat::Rgba8Unorm, SurfaceColorSpaces::BT2100_PQ),
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::SRGB),
    ];

    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Auto, SourceDynamicRange::Hlg),
        Some(SurfaceCandidate {
            format: TextureFormat::Rgb10a2Unorm,
            color_space: SurfaceColorSpace::Srgb,
        })
    );
    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Pq, SourceDynamicRange::Sdr),
        None
    );
}

#[test]
fn extended_linear_requires_a_float_format() {
    let capabilities = [
        capability(
            TextureFormat::Rgba16Unorm,
            SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
        ),
        capability(
            TextureFormat::Rgba16Float,
            SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
        ),
    ];

    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Auto, SourceDynamicRange::Hlg),
        Some(SurfaceCandidate {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
        })
    );
}

#[test]
fn srgb_is_the_final_fallback() {
    let capabilities = [capability(
        TextureFormat::Bgra8UnormSrgb,
        SurfaceColorSpaces::SRGB,
    )];

    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Auto, SourceDynamicRange::Hlg),
        Some(SurfaceCandidate {
            format: TextureFormat::Bgra8UnormSrgb,
            color_space: SurfaceColorSpace::Srgb,
        })
    );
}

#[test]
fn a_non_srgb_format_is_a_secondary_sdr_fallback() {
    let capabilities = [
        capability(TextureFormat::Rgba16Float, SurfaceColorSpaces::SRGB),
        capability(TextureFormat::Bgra8Unorm, SurfaceColorSpaces::SRGB),
    ];

    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Auto, SourceDynamicRange::Hlg),
        Some(SurfaceCandidate {
            format: TextureFormat::Bgra8Unorm,
            color_space: SurfaceColorSpace::Srgb,
        })
    );
}

#[test]
fn eight_bit_hdr_without_sdr_fallback_is_unusable() {
    let capabilities = [capability(
        TextureFormat::Rgba8Unorm,
        SurfaceColorSpaces::BT2100_PQ,
    )];

    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Auto, SourceDynamicRange::Hlg),
        None
    );
}

#[test]
fn forced_sdr_ignores_available_hdr_pairs() {
    let capabilities = [
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ),
        capability(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
    ];

    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Sdr, SourceDynamicRange::Hlg),
        Some(SurfaceCandidate {
            format: TextureFormat::Bgra8UnormSrgb,
            color_space: SurfaceColorSpace::Srgb,
        })
    );
}

#[test]
fn forced_output_fails_instead_of_falling_back() {
    let capabilities = [
        capability(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::BT2100_PQ),
        capability(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
    ];

    assert_eq!(
        select_surface_candidate(&capabilities, OutputMode::Hlg, SourceDynamicRange::Sdr),
        None
    );

    let error =
        select_required_surface_candidate(&capabilities, OutputMode::Hlg, SourceDynamicRange::Sdr)
            .unwrap_err();
    assert_eq!(
        error,
        SurfaceOutputError::RequestedModeUnavailable(OutputMode::Hlg)
    );
    assert_eq!(
        error.to_string(),
        "requested HDR output mode 'hlg' is unavailable on this display; use '--output auto' or '--output sdr'"
    );
}

#[test]
fn automatic_output_reports_a_distinct_missing_surface_pair() {
    let error = select_required_surface_candidate(&[], OutputMode::Auto, SourceDynamicRange::Sdr)
        .unwrap_err();

    assert_eq!(error, SurfaceOutputError::NoUsablePair);
    assert_eq!(
        error.to_string(),
        "the GPU surface advertises no usable output format/color-space pair; check the display connection and GPU driver"
    );
}

#[test]
fn hdr_peaks_are_container_defined_and_sdr_uses_reported_white() {
    let reported = wgpu::DisplayHdrInfo {
        luminance: Some(wgpu::DisplayLuminance {
            max_nits: Some(1_500.0),
            sdr_white_nits: Some(180.0),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert_eq!(
        resolved_output_peak_nits(SurfaceColorSpace::Bt2100Pq, &reported).to_bits(),
        PQ_ENCODING_PEAK_NITS.to_bits(),
    );
    assert_eq!(
        resolved_output_peak_nits(SurfaceColorSpace::Bt2100Hlg, &reported).to_bits(),
        HLG_ENCODING_PEAK_NITS.to_bits(),
    );
    assert_eq!(
        resolved_output_peak_nits(SurfaceColorSpace::ExtendedSrgbLinear, &reported).to_bits(),
        EXTENDED_LINEAR_FALLBACK_PEAK_NITS.to_bits(),
    );
    assert_eq!(
        resolved_output_peak_nits(SurfaceColorSpace::Srgb, &reported).to_bits(),
        180.0_f32.to_bits(),
    );
    assert_eq!(
        resolved_ui_white_nits(&reported).to_bits(),
        180.0_f32.to_bits()
    );

    let unknown = wgpu::DisplayHdrInfo::default();
    assert_eq!(
        resolved_output_peak_nits(SurfaceColorSpace::Srgb, &unknown).to_bits(),
        HDR_REFERENCE_WHITE_NITS.to_bits(),
    );
    assert_eq!(
        resolved_ui_white_nits(&unknown).to_bits(),
        HDR_REFERENCE_WHITE_NITS.to_bits()
    );
}

#[test]
fn hdr_mapping_summary_distinguishes_surface_and_source_roles() {
    assert_eq!(
        hdr_mapping_summary(true, SourceDynamicRange::Hlg),
        "Compositor"
    );
    assert_eq!(
        hdr_mapping_summary(true, SourceDynamicRange::Sdr),
        "Compositor"
    );
    assert_eq!(
        hdr_mapping_summary(false, SourceDynamicRange::Pq),
        "Application tone mapping to SDR"
    );
    assert_eq!(
        hdr_mapping_summary(false, SourceDynamicRange::Sdr),
        "Not required"
    );
}

#[test]
fn pq_surface_metadata_uses_hlg_intensity_target_as_max_cll() {
    let metadata = surface_hdr_metadata(
        SurfaceCandidate {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
        },
        RENDERING_OPTIONS,
        SourceDynamicRange::Hlg,
        1_000.0,
        0.0,
    );
    assert_eq!(
        metadata,
        Some(HdrMetadata::bt2020(1_000.0, 0.0, 1_000.0, 0.0))
    );
}

#[test]
fn pq_metadata_uses_intensity_target_for_mastering_and_max_cll() {
    let metadata = surface_hdr_metadata(
        SurfaceCandidate {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
        },
        RENDERING_OPTIONS,
        SourceDynamicRange::Pq,
        10_000.0,
        0.005,
    );
    assert_eq!(
        metadata,
        Some(HdrMetadata::bt2020(10_000.0, 0.005, 10_000.0, 0.0))
    );
}

#[test]
fn exposure_expands_the_signaled_hdr_volume() {
    let options = RenderingOptions {
        exposure_stops: 1.0,
        ..RENDERING_OPTIONS
    };
    let metadata = surface_hdr_metadata(
        SurfaceCandidate {
            color_space: SurfaceColorSpace::Bt2100Pq,
            format: TextureFormat::Rgb10a2Unorm,
        },
        options,
        SourceDynamicRange::Hlg,
        1_000.0,
        0.0,
    );
    assert_eq!(
        metadata,
        Some(HdrMetadata::bt2020(2_000.0, 0.0, 2_000.0, 0.0))
    );
}

#[test]
fn invalid_hdr_numbers_fall_back_to_safe_metadata() {
    let candidate = SurfaceCandidate {
        color_space: SurfaceColorSpace::Bt2100Pq,
        format: TextureFormat::Rgb10a2Unorm,
    };
    let expected = Some(HdrMetadata::bt2020(
        FALLBACK_HDR_SOURCE_PEAK_NITS,
        0.0,
        FALLBACK_HDR_SOURCE_PEAK_NITS,
        0.0,
    ));

    for (exposure_stops, source_peak_nits, source_min_nits) in [
        (f32::NAN, f32::NAN, f32::NAN),
        (f32::INFINITY, f32::INFINITY, f32::INFINITY),
        (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        (f32::NAN, -1.0, -1.0),
    ] {
        let options = RenderingOptions {
            exposure_stops,
            ..RENDERING_OPTIONS
        };
        assert_eq!(
            surface_hdr_metadata(
                candidate,
                options,
                SourceDynamicRange::Pq,
                source_peak_nits,
                source_min_nits,
            ),
            expected
        );
    }
}

#[test]
fn sdr_surfaces_do_not_request_vulkan_hdr_metadata() {
    assert_eq!(
        surface_hdr_metadata(
            SurfaceCandidate {
                color_space: SurfaceColorSpace::Srgb,
                format: TextureFormat::Bgra8UnormSrgb,
            },
            RENDERING_OPTIONS,
            SourceDynamicRange::Sdr,
            HDR_REFERENCE_WHITE_NITS,
            0.0,
        ),
        None
    );
}

#[test]
fn hdr_surface_classification_includes_hdr_encodings_and_scrgb() {
    for color_space in [
        SurfaceColorSpace::Bt2100Pq,
        SurfaceColorSpace::Bt2100Hlg,
        SurfaceColorSpace::ExtendedSrgbLinear,
    ] {
        assert!(is_hdr_color_space(color_space));
    }
    assert!(!is_hdr_color_space(SurfaceColorSpace::Srgb));
}

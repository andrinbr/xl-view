use std::path::PathBuf;

use xl_view::color::HDR_REFERENCE_WHITE_NITS;
use xl_view::decode::{
    CANONICAL_BYTES_PER_PIXEL, DecodeError, DecodeLimits, DecodedImage, LinearUnit,
    SourceColorEncoding, SourceDynamicRange, TILE_GUTTER, TILE_SIZE, decode_file, decode_memory,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn assert_near(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

fn pixels(image: &DecodedImage) -> &[f32] {
    assert_eq!(
        image.store.coarse_downsample(),
        1,
        "correctness fixtures should retain a full-resolution preview"
    );
    image.store.coarse_pixels()
}

#[test]
fn controlled_transfer_fixtures_decode_to_floating_point_bt2020() {
    for (name, expected_transfer, expected_range, expected_hdr) in [
        (
            "test_pattern-sRGB.jxl",
            "sRGB",
            SourceDynamicRange::Sdr,
            false,
        ),
        ("test_pattern-PQ.jxl", "PQ", SourceDynamicRange::Pq, true),
        ("test_pattern-HLG.jxl", "HLG", SourceDynamicRange::Hlg, true),
    ] {
        let decoded =
            decode_file(&fixture(name), DecodeLimits::from_memory_ceiling_mib(64)).unwrap();
        assert!(decoded.width > 0);
        assert!(decoded.height > 0);
        assert_eq!(
            pixels(&decoded).len(),
            decoded.width as usize * decoded.height as usize * 4
        );
        assert_eq!(decoded.source_dynamic_range, expected_range);
        assert_eq!(decoded.hdr_source, expected_hdr);
        assert!(pixels(&decoded).iter().all(|sample| sample.is_finite()));
        let SourceColorEncoding::Enumerated {
            transfer_function, ..
        } = &decoded.metadata.color_encoding
        else {
            panic!("fixture should use an enumerated color encoding");
        };
        assert_eq!(transfer_function, expected_transfer);
    }
}

#[test]
fn hdr_intensity_target_is_normalized_by_hdr_reference_white() {
    for (name, intensity_target) in [
        ("test_pattern-PQ.jxl", 10_000.0),
        ("test_pattern-HLG.jxl", 1_000.0),
    ] {
        let decoded =
            decode_file(&fixture(name), DecodeLimits::from_memory_ceiling_mib(64)).unwrap();
        assert_near(
            pixels(&decoded)[pixels(&decoded).len() - 4],
            intensity_target / HDR_REFERENCE_WHITE_NITS,
            0.001,
        );
        assert_near(
            decoded.metadata.tone_mapping.intensity_target_nits,
            intensity_target,
            0.5,
        );
    }

    let sdr = decode_file(
        &fixture("test_pattern-sRGB.jxl"),
        DecodeLimits::from_memory_ceiling_mib(64),
    )
    .unwrap();
    // Generated ICC colorants round to s15Fixed16. White differs by 3.1e-5
    // before final f16 packing (where it still becomes exactly 1.0).
    assert_near(pixels(&sdr)[pixels(&sdr).len() - 4], 1.0, 0.000_05);
}

#[test]
fn pq_and_hlg_present_their_measured_peak_but_keep_the_target() {
    // (fixture, target, measured peak)
    for (name, target, peak) in [
        ("test_pattern-PQ.jxl", 10_000.0_f64, 10_000.0),
        ("ramp-hlg-2000.jxl", 2_000.0, 2_000.0),
        // Content below the target,
        ("hdr-sRGB-HLG.jxl", 1_000.0, 211.2),
        ("gray-hdr-vardct-PQ.jxl", 1_000.0, 982.9),
        // or above it, which is not capped.
        ("hdr-P3-PQ.jxl", 1_000.0, 1_531.1),
        ("hlg-extended.jxl", 1_000.0, 1_927.4),
    ] {
        let image = decode_file(&fixture(name), DecodeLimits::from_memory_ceiling_mib(64)).unwrap();
        let measured = image.source_peak.nits();
        assert!((measured - peak).abs() < 0.1, "{name}: {measured}");
        // It is the brightest visible channel, before f16 storage rounding.
        let stored = pixels(&image)
            .chunks_exact(4)
            .filter(|pixel| pixel[3] > 0.0)
            .map(|pixel| pixel[0].max(pixel[1]).max(pixel[2]))
            .fold(0.0_f32, f32::max);
        let stored = f64::from(stored * HDR_REFERENCE_WHITE_NITS);
        assert!(
            (stored - measured).abs() <= measured / 2_048.0,
            "{name}: {stored}"
        );
        assert_eq!(image.intensity_target.nits().to_bits(), target.to_bits());
        // The target, not the measured peak, supports the HDR claim.
        assert!(image.hdr_source, "{name}");
    }
}

#[test]
fn hdr_brightness_ramps_reach_their_declared_peaks() {
    for (name, dynamic_range, peak_nits) in [
        ("ramp-hlg-1000.jxl", SourceDynamicRange::Hlg, 1_000.0),
        ("ramp-hlg-2000.jxl", SourceDynamicRange::Hlg, 2_000.0),
        ("ramp-pq-1000.jxl", SourceDynamicRange::Pq, 1_000.0),
        ("ramp-pq-10000.jxl", SourceDynamicRange::Pq, 10_000.0),
    ] {
        let decoded =
            decode_file(&fixture(name), DecodeLimits::from_memory_ceiling_mib(4)).unwrap();
        assert_eq!((decoded.width, decoded.height), (1_024, 64));
        assert_eq!(decoded.source_dynamic_range, dynamic_range);
        assert_near(
            decoded.metadata.tone_mapping.intensity_target_nits,
            peak_nits,
            f32::EPSILON,
        );

        let first_row = &pixels(&decoded)[..decoded.width as usize * 4];
        let mut previous = f32::NEG_INFINITY;
        for pixel in first_row.chunks_exact(4) {
            assert!(previous <= pixel[0]);
            previous = pixel[0];
        }
        let midpoint = decoded.width / 2;
        let midpoint_index = usize::try_from(midpoint).unwrap();
        let midpoint = f32::from(u16::try_from(midpoint).unwrap());
        let maximum_x = f32::from(u16::try_from(decoded.width - 1).unwrap());
        assert_near(
            first_row[midpoint_index * 4],
            peak_nits * midpoint / maximum_x / HDR_REFERENCE_WHITE_NITS,
            // Bound HDR error relative to luminance: f16 quantization occurs
            // before the steep inverse transfer, including for 10,000-nit PQ.
            peak_nits * midpoint / maximum_x / HDR_REFERENCE_WHITE_NITS * 0.0025 + 0.000_01,
        );
        let final_pixel = &first_row[(decoded.width as usize - 1) * 4..];
        for channel in &final_pixel[..3] {
            assert_near(*channel, peak_nits / HDR_REFERENCE_WHITE_NITS, 0.02);
        }
    }
}

#[test]
fn hdr_ramps_without_explicit_intensity_use_libjxl_hdr_defaults() {
    for (name, dynamic_range, decoded_peak_nits) in [
        (
            "ramp-hlg-no-intensity.jxl",
            SourceDynamicRange::Hlg,
            1_000.0,
        ),
        ("ramp-pq-no-intensity.jxl", SourceDynamicRange::Pq, 10_000.0),
    ] {
        let decoded =
            decode_file(&fixture(name), DecodeLimits::from_memory_ceiling_mib(4)).unwrap();
        assert_eq!(decoded.source_dynamic_range, dynamic_range);
        // These files were encoded without --intensity_target. libjxl selects
        // its transfer-function-specific HDR default and stores it in the
        // resulting JPEG XL metadata.
        assert_near(
            decoded.metadata.tone_mapping.intensity_target_nits,
            decoded_peak_nits,
            f32::EPSILON,
        );
        let last_pixel = (decoded.width as usize - 1) * 4;
        assert_near(
            pixels(&decoded)[last_pixel],
            decoded_peak_nits / HDR_REFERENCE_WHITE_NITS,
            0.02,
        );
    }
}

#[test]
fn representative_working_space_values_are_stable() {
    for (name, expected) in [
        ("test_pattern-sRGB.jxl", [0.014_187_302, 0.348_052_9, 1.0]),
        (
            "test_pattern-PQ.jxl",
            [0.002_870_307_3, 1.516_745_8, 49.261_086],
        ),
        (
            "test_pattern-HLG.jxl",
            [0.008_818_784, 0.478_998_93, 4.926_109],
        ),
    ] {
        let decoded =
            decode_file(&fixture(name), DecodeLimits::from_memory_ceiling_mib(64)).unwrap();
        for (x, expected) in [0, decoded.width as usize / 2, decoded.width as usize - 1]
            .into_iter()
            .zip(expected)
        {
            // Encoded samples round to f16 before linearization. Half an ULP
            // produces up to ~0.1% SDR and ~0.25% PQ/HLG luminance error away
            // from black. Keep it relative to luminance instead of granting
            // dark patches a large absolute HDR tolerance.
            // A double-precision sweep of 1,000,001 encoded values in [0.05, 1]
            // measured at most 0.2441% PQ and 0.1598% HLG error from f16 input.
            let relative = if decoded.source_dynamic_range == SourceDynamicRange::Sdr {
                0.001
            } else {
                0.0025
            };
            assert_near(
                pixels(&decoded)[x * 4],
                expected,
                expected.abs() * relative + 0.000_01,
            );
        }
    }
}

#[test]
fn srgb_primaries_are_converted_to_linear_bt2020() {
    let decoded = decode_file(
        &fixture("test_pattern-sRGB.jxl"),
        DecodeLimits::from_memory_ceiling_mib(64),
    )
    .unwrap();
    assert_eq!((decoded.width, decoded.height), (1_024, 1_024));

    // Independently evaluated columns of the standard D65
    // linear-BT.709/sRGB-to-linear-BT.2020 conversion matrix. The fixture's
    // final column contains full-intensity blue, green, and red ramp pixels.
    for (y, expected) in [
        (576, [0.043_313_067, 0.011_362_316, 0.895_595_25]),
        (640, [0.329_283_03, 0.919_540_4, 0.088_013_31]),
        (768, [0.627_403_9, 0.069_097_29, 0.016_391_44]),
    ] {
        let offset = (y * decoded.width as usize + decoded.width as usize - 1) * 4;
        let actual = &pixels(&decoded)[offset..offset + 4];
        for (actual, expected) in actual[..3].iter().zip(expected) {
            assert_near(*actual, expected, 0.000_1);
        }
        assert_near(actual[3], 1.0, f32::EPSILON);
    }
}

#[test]
fn grayscale_is_expanded_to_rgb() {
    let decoded = decode_file(
        &fixture("grayscale.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    assert_eq!((decoded.width, decoded.height), (3, 2));
    for rgba in pixels(&decoded).chunks_exact(4) {
        assert_near(rgba[0], rgba[1], f32::EPSILON);
        assert_near(rgba[1], rgba[2], f32::EPSILON);
        assert_near(rgba[3], 1.0, f32::EPSILON);
    }
}

#[test]
fn linear_bt2020_passes_through_the_canonical_cms_request() {
    let decoded = decode_file(
        &fixture("linear-bt2020.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    let SourceColorEncoding::Enumerated {
        primaries,
        transfer_function,
        ..
    } = &decoded.metadata.color_encoding
    else {
        panic!("fixture should have an enumerated color encoding");
    };
    assert_eq!(primaries.as_deref(), Some("BT.2020"));
    assert_eq!(transfer_function, "Linear");
    assert_eq!(decoded.source_dynamic_range, SourceDynamicRange::Linear);
    assert_eq!(
        decoded.linear_unit,
        Some(LinearUnit {
            nits: 203.0,
            assumed: true
        })
    );
    assert!(!decoded.hdr_source);
    assert_eq!(decoded.source_peak.nits().to_bits(), 203.0_f64.to_bits());
    assert_near(pixels(&decoded)[0], 1.0, 0.000_01);
    assert_near(pixels(&decoded)[1], 0.0, 0.000_01);
    assert_near(pixels(&decoded)[2], 0.0, 0.000_01);
}

#[test]
fn orientation_is_applied_once_by_the_decoder() {
    let decoded = decode_file(
        &fixture("oriented.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    assert_eq!((decoded.width, decoded.height), (2, 3));

    let top_left = &pixels(&decoded)[0..3];
    let bottom_left = &pixels(&decoded)[(2 * decoded.width as usize * 4)..][..3];
    assert!(top_left.iter().all(|channel| *channel > 0.99));
    assert!(bottom_left.iter().all(|channel| channel.abs() < 0.000_01));
}

#[test]
fn straight_and_associated_alpha_return_straight_working_pixels() {
    let decoded = decode_file(
        &fixture("alpha.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    assert_near(pixels(&decoded)[3], 1.0, f32::EPSILON);
    assert_near(pixels(&decoded)[7], 128.0 / 255.0, 0.000_01);
    assert_near(pixels(&decoded)[11], 0.0, f32::EPSILON);

    let associated = decode_file(
        &fixture("associated-alpha.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    for (straight, associated) in pixels(&decoded)
        .chunks_exact(4)
        .zip(pixels(&associated).chunks_exact(4))
    {
        assert_near(straight[3], associated[3], 0.000_01);
        if straight[3] > 0.0 {
            for channel in 0..3 {
                assert_near(straight[channel], associated[channel], 0.005);
            }
        } else {
            assert!(
                associated[..3]
                    .iter()
                    .all(|channel| channel.abs() < f32::EPSILON)
            );
        }
    }
}

#[test]
fn icc_profile_is_reported_and_applied() {
    let icc = decode_file(
        &fixture("icc.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    assert!(matches!(
        icc.metadata.color_encoding,
        SourceColorEncoding::Icc {
            profile_bytes: 21_012,
            ..
        }
    ));
    assert_eq!((icc.width, icc.height), (3, 2));
    for (actual, expected) in pixels(&icc).chunks_exact(4).zip([
        [0.877_19, 0.096_56, 0.022_83, 1.0],
        [0.077_64_f32, 0.891_48, 0.042_97, 1.0],
        [0.045_05, 0.011_72, 0.933_59, 1.0],
    ]) {
        for channel in 0..4 {
            assert_near(actual[channel], expected[channel], 0.001);
        }
    }
}

#[test]
fn animation_first_frame_pixels_are_stable() {
    let animation = decode_file(
        &fixture("animation.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    assert_eq!((animation.width, animation.height), (2, 2));
    for pixel in pixels(&animation).chunks_exact(4) {
        for (actual, expected) in pixel.iter().zip([0.043_32_f32, 0.011_36, 0.895_61, 1.0]) {
            assert_near(*actual, expected, 0.001);
        }
    }
}

#[test]
fn common_exif_fixture_is_parsed() {
    let decoded = decode_file(
        &fixture("exif-common.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    assert_eq!((decoded.width, decoded.height), (3, 2));
    let exif = decoded
        .metadata
        .exif
        .as_ref()
        .expect("EXIF metadata should be available");
    assert_eq!(exif.camera_make.as_deref(), Some("ACME"));
    assert_eq!(exif.camera_model.as_deref(), Some("Photon 1"));
    assert_eq!(exif.lens_make.as_deref(), Some("ACME Optics"));
    assert_eq!(exif.lens_model.as_deref(), Some("Prime 50"));
    assert_eq!(exif.captured_at.as_deref(), Some("2026:07:13 12:34:56"));
    assert_eq!(exif.exposure_time_seconds, Some(1.0 / 125.0));
    assert_eq!(exif.aperture_f_number, Some(2.8));
    assert_eq!(exif.iso_speed, Some(200));
    assert_eq!(exif.focal_length_mm, Some(50.0));
    assert_eq!(exif.exposure_bias_ev, Some(-1.0 / 3.0));
    assert_eq!(exif.artist.as_deref(), Some("Ada Example"));
    assert_eq!(exif.copyright.as_deref(), Some("CC0 fixture"));
    assert_eq!(exif.software.as_deref(), Some("Fixture Maker 1.0"));
    assert_eq!(exif.parse_error, None);
}

#[test]
fn xmp_rating_fixture_is_parsed() {
    let decoded = decode_file(
        &fixture("xmp-rating.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    let xmp = decoded
        .metadata
        .xmp
        .as_ref()
        .expect("XML box should be retained");
    assert_eq!(xmp.rating, Some(4.0));
    assert_eq!(xmp.parse_error, None);
}

#[test]
fn corrupt_and_truncated_inputs_return_typed_errors() {
    let limits = DecodeLimits::from_memory_ceiling_mib(64);
    assert!(matches!(
        decode_memory(b"not a JPEG XL image", limits),
        Err(DecodeError::Decoder(_))
    ));

    let bytes = std::fs::read(fixture("test_pattern-sRGB.jxl")).unwrap();
    assert!(matches!(
        decode_memory(&bytes[..8], limits),
        Err(DecodeError::Decoder(_))
    ));
}

#[test]
fn output_storage_accounts_for_one_canonical_rgba_buffer() {
    const RGBA_BYTES: usize = 1_024 * 1_024 * CANONICAL_BYTES_PER_PIXEL;
    let decoded = decode_file(
        &fixture("test_pattern-sRGB.jxl"),
        DecodeLimits::from_memory_ceiling_mib(64),
    )
    .unwrap();
    assert_eq!(decoded.store.canonical_storage_bytes(), RGBA_BYTES);
    assert_eq!(
        decoded.memory_cost_bytes,
        RGBA_BYTES + size_of_val(decoded.store.coarse_pixels())
    );
}

#[test]
fn unified_store_preserves_canonical_pixels_and_clamps_gutters() {
    let path = fixture("alpha.jxl");
    let decoded = decode_file(&path, DecodeLimits::from_memory_ceiling_mib(4)).unwrap();
    let store = &decoded.store;
    let expected = pixels(&decoded);
    assert_eq!(store.coarse_dimensions(), (decoded.width, decoded.height));
    assert_eq!(store.coarse_downsample(), 1);

    let mut canonical_row = vec![0.0; usize::try_from(decoded.width).unwrap() * 4];
    store
        .read_canonical_row_rgba_f32(0, &mut canonical_row)
        .unwrap();
    for (canonical, straight) in canonical_row.chunks_exact(4).zip(expected.chunks_exact(4)) {
        for channel in 0..3 {
            assert_near(canonical[channel], straight[channel] * straight[3], 0.001);
        }
        assert_near(canonical[3], straight[3], 0.001);
    }
    assert!(
        store
            .read_canonical_row_rgba_f32(decoded.height, &mut canonical_row)
            .is_err()
    );

    let tile = store.read_tile_rgba16f(0, 0).unwrap();
    let extent = usize::try_from(TILE_SIZE + TILE_GUTTER * 2).unwrap();
    let sample = |x: usize, y: usize| -> [f32; 4] {
        let offset = (y * extent + x) * 8;
        std::array::from_fn(|channel| {
            let offset = offset + channel * 2;
            half::f16::from_bits(u16::from_le_bytes(
                tile[offset..offset + 2].try_into().unwrap(),
            ))
            .to_f32()
        })
    };
    let first = sample(usize::try_from(TILE_GUTTER).unwrap(), 0);
    let first_interior = sample(
        usize::try_from(TILE_GUTTER).unwrap(),
        usize::try_from(TILE_GUTTER).unwrap(),
    );
    for (actual, expected) in first.into_iter().zip(first_interior) {
        assert_near(actual, expected, f32::EPSILON);
    }
    let second = sample(
        usize::try_from(TILE_GUTTER + 1).unwrap(),
        usize::try_from(TILE_GUTTER).unwrap(),
    );
    let second_alpha = expected[7];
    for channel in 0..3 {
        assert_near(second[channel], expected[4 + channel] * second_alpha, 0.001);
    }
    assert_near(second[3], second_alpha, 0.001);
    let final_pixel = sample(extent - 1, extent - 1);
    assert!(
        final_pixel
            .into_iter()
            .all(|channel| channel.abs() < f32::EPSILON)
    );
}

#[test]
fn hdr_with_srgb_and_p3_primaries_linearizes_before_gamut_conversion() {
    use xl_view::color::{hlg_inverse_oetf, hlg_system_gamma, pq_eotf};
    let samples = [[0.6_f32, 0.4, 0.2], [0.4, 0.6, 0.8], [0.75, 0.5, 0.25]];
    // Independent D65 source-to-BT.2020 matrices.
    let srgb = [
        [0.627_403_9_f32, 0.329_283_03, 0.043_313_067],
        [0.069_097_29, 0.919_540_4, 0.011_362_316],
        [0.016_391_44, 0.088_013_31, 0.895_595_25],
    ];
    let p3 = [
        [0.753_833_03_f32, 0.198_597_37, 0.047_569_6],
        [0.045_743_85, 0.941_777_2, 0.012_478_93],
        [-0.001_210_34, 0.017_601_72, 0.983_608_6],
    ];
    for (primaries, matrix) in [("sRGB", srgb), ("P3", p3)] {
        for transfer in ["PQ", "HLG"] {
            let image = decode_file(
                &fixture(&format!("hdr-{primaries}-{transfer}.jxl")),
                DecodeLimits::from_memory_ceiling_mib(4),
            )
            .unwrap();
            for (actual, source) in pixels(&image).chunks_exact(4).zip(samples) {
                let linear = source.map(|v| {
                    let quantized = half::f16::from_f32(v).to_f32();
                    if transfer == "PQ" {
                        pq_eotf(quantized)
                    } else {
                        hlg_inverse_oetf(quantized)
                    }
                });
                let mut expected = matrix.map(|row| {
                    row[0].mul_add(linear[0], row[1].mul_add(linear[1], row[2] * linear[2]))
                });
                let scale = if transfer == "PQ" {
                    10_000.0 / HDR_REFERENCE_WHITE_NITS
                } else {
                    let luminance = expected[0]
                        .mul_add(0.2627, expected[1].mul_add(0.6780, expected[2] * 0.0593));
                    luminance.max(0.0).powf(hlg_system_gamma(1_000.0) - 1.0) * 1_000.0
                        / HDR_REFERENCE_WHITE_NITS
                };
                for channel in &mut expected {
                    *channel *= scale;
                }
                for channel in 0..3 {
                    assert_near(
                        actual[channel],
                        expected[channel],
                        expected[channel].abs() * 0.0003 + 0.000_01,
                    );
                }
            }
        }
    }
}

#[test]
fn floating_point_hlg_preserves_headroom_and_documents_the_decoder_clamp() {
    let image = decode_file(
        &fixture("hlg-extended.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    let values = pixels(&image);
    assert!(values[4] > 1_000.0 / HDR_REFERENCE_WHITE_NITS);
    // Like jxl's XYB output, original-profile HLG keeps 109% headroom and
    // clamps only larger overshoot to 1.1 rounded to f16. Pin that limit.
    let clamped = xl_view::color::hlg_inverse_oetf(half::f16::from_f32(1.1).to_f32()).powf(1.2)
        * 1_000.0
        / HDR_REFERENCE_WHITE_NITS;
    assert_near(values[8], clamped, 0.000_01);
    assert!(values[8] > values[4]);
}

#[test]
fn vardct_uses_actual_output_profile_and_preserves_embedded_icc_metadata() {
    // The embedded Adobe RGB profile describes the source, while XYB VarDCT
    // output is linear sRGB. Reference: djxl 0.12.0, linear BT.2020 output.
    let image = decode_file(
        &fixture("icc-vardct.jxl"),
        DecodeLimits::from_memory_ceiling_mib(4),
    )
    .unwrap();
    assert!(matches!(
        image.metadata.color_encoding,
        SourceColorEncoding::Icc { .. }
    ));
    assert_eq!(image.source_dynamic_range, SourceDynamicRange::Sdr);
    assert_eq!(image.source_peak.nits().to_bits(), 203.0_f64.to_bits());
    assert_eq!(image.linear_unit, None);
    assert!(!image.hdr_source);
    let expected = [
        [0.877_220_4_f32, 0.099_047_89, 0.025_154_68],
        [0.075_885_82, 0.885_276_1, 0.052_365_69],
        [0.048_172_32_f32, 0.014_956_82, 0.895_855_8],
        [0.993_013_8, 0.999_46, 1.008_425_1],
        [0.223_782_78, 0.224_218_55, 0.199_495_75],
        [0.000_203_404_34, -0.000_251_546_47, 0.001_409_266_7],
    ];
    for (actual, expected) in pixels(&image).chunks_exact(4).zip(expected) {
        for channel in 0..3 {
            // Source-channel half rounding propagates through the gamut
            // matrix, so small destination channels need an absolute bound.
            assert_near(actual[channel], expected[channel], 0.0005);
        }
    }
}

#[test]
fn hdr_vardct_and_already_linear_grayscale_use_the_hdr_reference_white() {
    // djxl 0.12.0 linear output is relative to the 1,000-nit intensity target.
    // These reference values are rescaled to the viewer's 203-nit white.
    let cases = [
        (
            "hdr-vardct-PQ.jxl",
            SourceDynamicRange::Pq,
            [
                [0.938_474_6_f32, 0.205_928_52, 0.012_391_346],
                [0.723_969_04, 1.233_772_9, 7.563_636],
                [3.746_620_4, 0.647_619_37, 0.022_240_207],
            ],
            0.0025,
        ),
        (
            "gray-hdr-vardct-PQ.jxl",
            SourceDynamicRange::Pq,
            [[0.025_350_263; 3], [0.454_337_78; 3], [4.841_332; 3]],
            0.0005,
        ),
        (
            "gray-hdr-vardct-HLG.jxl",
            SourceDynamicRange::Hlg,
            [[0.047_418_006; 3], [0.249_529_81; 3], [1.000_829_7; 3]],
            0.0005,
        ),
    ];
    for (name, range, expected, relative_error) in cases {
        let image = decode_file(&fixture(name), DecodeLimits::from_memory_ceiling_mib(4)).unwrap();
        assert_eq!(image.source_dynamic_range, range);
        for (actual, expected) in pixels(&image).chunks_exact(4).zip(expected) {
            for channel in 0..3 {
                assert_near(
                    actual[channel],
                    expected[channel],
                    expected[channel].abs() * relative_error + 0.000_01,
                );
            }
        }
    }
}

#[test]
fn native_linear_patterns_match_the_absolute_hlg_reference() {
    let limits = DecodeLimits::from_memory_ceiling_mib(64);
    let reference = decode_file(&fixture("test_pattern-HLG.jxl"), limits).unwrap();
    for (name, expected_unit, lossy) in [
        ("test_pattern-scRGB.jxl", 80.0, false),
        ("test_pattern-scRGB-xyb.jxl", 80.0, true),
        ("test_pattern-linear.jxl", 1_000.0, false),
    ] {
        let image = decode_file(&fixture(name), limits).unwrap();
        assert_eq!(image.source_dynamic_range, SourceDynamicRange::Linear);
        let unit = image.linear_unit.unwrap();
        assert_near(unit.nits, expected_unit, 0.01);
        assert!(!unit.assumed, "{name}");
        // Calibrated by their targets, so even the XYB fixture claims HDR.
        assert!(image.hdr_source, "{name}");
        assert!(
            (image.source_peak.nits() - 1_000.0).abs() < if lossy { 10.0 } else { 1.0 },
            "{name}: {:?}",
            image.source_peak
        );
        assert_near(
            image.metadata.tone_mapping.intensity_target_nits,
            if lossy { 80.0 } else { 1_000.0 },
            0.0,
        );
        for (index, (actual, expected)) in pixels(&image)
            .chunks_exact(4)
            .zip(pixels(&reference).chunks_exact(4))
            .enumerate()
        {
            let x = index % 1024;
            let y = index / 1024;
            // Distance-1 VarDCT introduces ringing at patch edges. Compare
            // patch/ramp interiors for the lossy fixture's absolute calibration.
            if lossy
                && (x % 128 < 8
                    || x % 128 >= 120
                    || y % 64 < 8
                    || y % 64 >= 56
                    || (392..409).contains(&y))
            {
                continue;
            }
            let magnitude = expected[..3].iter().copied().fold(0.0_f32, f32::max);
            // HLG input f16 contributes <=0.25%; linear source f16 and ICC
            // round trips add <=0.1%. XYB distance 1 adds a 1% peak allowance.
            let tolerance = magnitude * 0.0035 + if lossy { 0.05 } else { 0.0005 };
            for channel in 0..3 {
                assert!(
                    (actual[channel] - expected[channel]).abs() <= tolerance,
                    "{name} pixel ({x}, {y}) channel {channel}: {} vs {}, tolerance {tolerance}",
                    actual[channel],
                    expected[channel]
                );
            }
        }
    }
}

#[test]
fn default_target_linear_peaks_are_measured_but_only_exact_samples_claim_hdr() {
    const ASSUMED_UNIT: LinearUnit = LinearUnit {
        nits: HDR_REFERENCE_WHITE_NITS,
        assumed: true,
    };
    let limits = DecodeLimits::from_memory_ceiling_mib(4);
    let exact = decode_file(&fixture("linear-highlights.jxl"), limits).unwrap();
    assert_eq!(exact.source_dynamic_range, SourceDynamicRange::Linear);
    assert_near(
        exact.metadata.tone_mapping.intensity_target_nits,
        255.0,
        0.0,
    );
    assert_eq!(exact.linear_unit, Some(ASSUMED_UNIT));
    // Lossless BT.2020 needs no CMS, so the measured 4.0 peak is exact.
    assert_eq!(exact.source_peak.nits().to_bits(), 812.0_f64.to_bits());
    assert!(exact.hdr_source);
    let neutral = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0];
    for (pixel, expected) in pixels(&exact).chunks_exact(4).zip(neutral) {
        assert_eq!(pixel, [expected, expected, expected, 1.0]);
    }

    let lossy = decode_file(&fixture("linear-xyb-sdr.jxl"), limits).unwrap();
    assert_eq!(lossy.source_dynamic_range, SourceDynamicRange::Linear);
    assert_eq!(lossy.linear_unit, Some(ASSUMED_UNIT));
    // VarDCT overshoot lifts the measured peak past the white snap, but XYB
    // with the default target has no evidence beyond those decoded values.
    assert!(
        lossy.source_peak.nits() > 203.5,
        "fixture no longer overshoots: {:?}",
        lossy.source_peak
    );
    assert!(!lossy.hdr_source);
}

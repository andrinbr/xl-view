use std::path::PathBuf;

use xl_view::color::HDR_REFERENCE_WHITE_NITS;
use xl_view::decode::{
    CANONICAL_BYTES_PER_PIXEL, DecodeError, DecodeLimits, DecodedImage, SourceColorEncoding,
    SourceDynamicRange, TILE_GUTTER, TILE_SIZE, decode_file, decode_memory,
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
    for (name, expected_transfer, expected_range) in [
        ("test_pattern-sRGB.jxl", "sRGB", SourceDynamicRange::Sdr),
        ("test_pattern-PQ.jxl", "PQ", SourceDynamicRange::Pq),
        ("test_pattern-HLG.jxl", "HLG", SourceDynamicRange::Hlg),
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
    assert_near(pixels(&sdr)[pixels(&sdr).len() - 4], 1.0, 0.000_01);
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
            0.02,
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
            assert_near(pixels(&decoded)[x * 4], expected, 0.000_1);
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
    assert_eq!(primaries, "Bt2100");
    assert_eq!(transfer_function, "Linear");
    assert_near(pixels(&decoded)[0], 1.0, 0.000_01);
    assert_near(pixels(&decoded)[1], 0.0, 0.000_01);
    assert_near(pixels(&decoded)[2], 0.0, 0.000_01);
}

#[test]
fn orientation_is_applied_once_by_the_render_stream() {
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
        [0.077_64, 0.891_48, 0.042_97, 1.0],
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
        for (actual, expected) in pixel.iter().zip([0.043_32, 0.011_36, 0.895_61, 1.0]) {
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

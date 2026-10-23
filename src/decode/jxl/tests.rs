// Exact defaults, identity values and snapped peaks are part of the contract.
#![expect(clippy::float_cmp)]

use super::*;
use crate::test_utils::fixture_path;

fn limits() -> DecodeLimits {
    DecodeLimits::from_memory_ceiling_mib(64)
}

#[test]
fn checked_dimensions_and_bounded_memory_cover_adversarial_geometry() {
    assert!(matches!(
        validate_dimensions(0, 1),
        Err(DecodeError::InvalidDimensions { .. })
    ));
    assert!(matches!(
        validate_dimensions(u32::MAX, u32::MAX),
        Err(DecodeError::SizeOverflow)
    ));
    for (width, height) in [
        (50_000, 20_000),
        (u32::MAX, 1),
        (1, u32::MAX),
        (28_000, 7_143),
    ] {
        let retained = estimated_memory_cost_bytes(width, height).unwrap();
        let scratch = estimated_render_scratch_bytes(width, height).unwrap();
        assert!(scratch <= MAX_SCRATCH_BYTES as u64);
        assert!(retained >= u64::from(width) * u64::from(height) * 8);
        assert!(matches!(
            validate_output_memory(retained + scratch, limits()),
            Err(DecodeError::OutputMemoryLimit { .. })
        ));
    }
    assert_eq!(coarse_downsample(6_000, 4_000), 3);
    assert_eq!(coarse_downsample(u32::MAX, 1), 1_048_576);
    assert!(matches!(
        allocate_vec::<CanonicalPixel>(usize::MAX, true),
        Err(DecodeError::SizeOverflow)
    ));
    assert!(matches!(
        allocate_vec::<CanonicalPixel>(usize::MAX / 8, true),
        Err(DecodeError::OutputAllocation { .. })
    ));
}

struct ChunkedInput<'a> {
    remaining: &'a [u8],
    reads: usize,
}
impl JxlBitstreamInput for ChunkedInput<'_> {
    fn available_bytes(&mut self) -> std::io::Result<usize> {
        Ok(self.remaining.len())
    }
    fn read(&mut self, buffers: &mut [IoSliceMut<'_>]) -> std::io::Result<usize> {
        self.reads += 1;
        let Some(buffer) = buffers.first_mut() else {
            return Ok(0);
        };
        let count = 7.min(buffer.len()).min(self.remaining.len());
        buffer[..count].copy_from_slice(&self.remaining[..count]);
        self.remaining = &self.remaining[count..];
        Ok(count)
    }
}

#[test]
fn decoder_handles_short_reads_and_rejects_prefetch_before_frame_decoding() {
    let bytes = fs::read(fixture_path("test_pattern-sRGB.jxl")).unwrap();
    let mut input = ChunkedInput {
        remaining: &bytes,
        reads: 0,
    };
    let decoded = decode_input(&mut input, limits(), None).unwrap();
    assert_eq!((decoded.width, decoded.height), (1_024, 1_024));
    assert!(input.reads > 1);
    assert_eq!(
        estimated_memory_cost_bytes(decoded.width, decoded.height).unwrap(),
        decoded.memory_cost_bytes as u64
    );
    let mut input = ChunkedInput {
        remaining: &bytes,
        reads: 0,
    };
    assert!(matches!(
        decode_input(&mut input, limits(), Some(1)),
        Err(DecodeError::PrefetchTooLarge { .. })
    ));
    assert!(input.remaining.len() > bytes.len() / 2);
}

#[test]
fn in_place_conversion_keeps_allocation_and_averages_straight_alpha_across_chunks() {
    // A coarse band crosses the 65,536-pixel scratch boundary and has a short
    // final coarse row and column. Use an independent scalar f64 reference.
    let (width, height, downsample) = (1_001_u32, 69_u32, 3_u32);
    let mut pixels: Vec<_> = (0..width * height)
        .map(|index| {
            let alpha = if index % 3 == 0 { 0.0 } else { 0.5 };
            let rgb = f32::from(u16::try_from(index % 128).unwrap()) / 32.0 - 1.0;
            [rgb, rgb / 2.0, 1.5, alpha]
                .map(half::f16::from_f32)
                .map(half::f16::to_bits)
        })
        .collect();
    let address = pixels.as_ptr();
    let mut sums = vec![
        [0.0_f64; 4];
        usize_from_u32(width.div_ceil(downsample) * height.div_ceil(downsample))
    ];
    let mut counts = vec![0_u32; sums.len()];
    for (index, pixel) in pixels.iter().enumerate() {
        let index = u32::try_from(index).unwrap();
        let coarse = usize_from_u32(
            (index / width / downsample) * width.div_ceil(downsample) + index % width / downsample,
        );
        let rgba = pixel.map(|v| half::f16::from_bits(v).to_f32());
        for channel in 0..3 {
            sums[coarse][channel] += f64::from(rgba[channel] * rgba[3]);
        }
        sums[coarse][3] += f64::from(rgba[3]);
        counts[coarse] += 1;
    }
    let conversion = ColorConversion::new(
        &working_profile(),
        false,
        SourceDynamicRange::Sdr,
        SourceIntensityTarget::from_jxl_metadata(1_000.0),
    )
    .unwrap();
    let mut preview = vec![[0.0; 4]; sums.len()];
    convert_band(
        &mut pixels,
        &mut preview,
        width,
        downsample,
        false,
        &conversion,
    )
    .unwrap();
    assert_eq!(address, pixels.as_ptr());
    for ((sum, count), actual) in sums.iter().zip(counts).zip(preview) {
        for channel in 0..3 {
            assert!((f64::from(actual[channel]) - sum[channel] / sum[3]).abs() < 1e-6);
        }
        assert!((f64::from(actual[3]) - sum[3] / f64::from(count)).abs() < 1e-6);
    }
    assert_eq!(pixels[0], [0; 4]);
}

#[test]
fn matrix_profiles_preserve_negative_and_above_white_values() {
    for primaries in [JxlPrimaries::SRGB, JxlPrimaries::P3] {
        let profile = JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries,
            transfer_function: JxlTransferFunction::Linear,
            rendering_intent: RenderingIntent::Relative,
        });
        let conversion = ColorConversion::new(
            &profile,
            false,
            SourceDynamicRange::Sdr,
            SourceIntensityTarget::from_jxl_metadata(1_000.0),
        )
        .unwrap();
        assert!(conversion.transform.is_some());
        let pixels = vec![
            [-0.25, -0.25, -0.25, 1.0],
            [2.0, 2.0, 2.0, 1.0],
            [0.25, 0.1, 0.5, 1.0],
            [2.0, 1.5, 1.1, 1.0],
            [0.5, 0.25, 0.75, 1.0],
        ]
        .into_iter()
        .map(|p: [f32; 4]| p.map(half::f16::from_f32).map(half::f16::to_bits))
        .collect();
        let (store, _) = convert_pixels(pixels, 5, 1, true, &conversion).unwrap();
        for value in &store.coarse_pixels()[..3] {
            assert!((*value + 0.25).abs() < 0.0001);
        }
        for value in &store.coarse_pixels()[4..7] {
            assert!((*value - 2.0).abs() < 0.0002);
        }
        // Non-neutral above-white samples still require gamut conversion.
        assert!(
            store.coarse_pixels()[12..15]
                .iter()
                .all(|value| *value > 1.0)
        );
    }
}

#[test]
fn nominal_sample_clamping_precedes_unpremultiplication() {
    for (associated, expected) in [(false, [0.0, 1.0, 0.25, 0.5]), (true, [0.0, 2.0, 0.5, 0.5])] {
        let mut conversion = linear_conversion(&working_profile(), associated);
        conversion.clamp_to_nominal_range = true;
        // Lossy integer reconstruction overshoots both bounds. Only the stored
        // samples are bounded; recovering straight color can exceed white.
        let (store, _) = convert_pixels(
            vec![packed([-0.25, 1.25, 0.25, 0.5])],
            1,
            1,
            false,
            &conversion,
        )
        .unwrap();
        assert_eq!(store.coarse_pixels(), expected);
    }
}

#[test]
fn integer_associated_alpha_preserves_above_white_color() {
    let mut conversion = linear_conversion(&working_profile(), true);
    conversion.clamp_to_nominal_range = true;
    // Nominal 8-bit samples can exceed alpha and recover above-white color.
    let pixel = packed([192_u8, 128, 64, 128].map(|value| f32::from(value) / 255.0));
    let image = calibrate_and_convert(
        vec![pixel],
        1,
        1,
        false,
        conversion,
        linear_source(255.0, false, false),
    )
    .unwrap();
    assert_eq!(&image.store.coarse_pixels()[..3], &[1.5, 1.0, 0.5]);
    assert_eq!(image.source_peak.nits(), 304.5);
    assert!(image.hdr_source);
}

#[test]
fn rendering_intent_and_unsupported_profiles_are_explicit() {
    let profile = working_profile();
    let mut icc = profile.try_as_icc().unwrap().into_owned();
    for (value, expected) in [
        (0_u32, moxcms::RenderingIntent::Perceptual),
        (1, moxcms::RenderingIntent::RelativeColorimetric),
        (2, moxcms::RenderingIntent::Saturation),
        (3, moxcms::RenderingIntent::AbsoluteColorimetric),
        (99, moxcms::RenderingIntent::RelativeColorimetric),
    ] {
        icc[64..68].copy_from_slice(&value.to_be_bytes());
        assert_eq!(profile_intent(&icc), expected);
    }
    assert!(matches!(
        ColorConversion::new(
            &JxlColorProfile::Icc(vec![0; 128]),
            false,
            SourceDynamicRange::Sdr,
            SourceIntensityTarget::from_jxl_metadata(1_000.0)
        ),
        Err(DecodeError::UnsupportedColorProfile(_))
    ));
    for transfer in [JxlTransferFunction::PQ, JxlTransferFunction::HLG] {
        let profile = JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::BT2100,
            transfer_function: transfer,
            rendering_intent: RenderingIntent::Relative,
        });
        assert!(matches!(
            dynamic_range(&JxlColorProfile::Icc(
                profile.try_as_icc().unwrap().into_owned()
            )),
            Err(DecodeError::UnsupportedColorProfile(_))
        ));
    }
}

#[test]
fn gray_matrix_transform_preserves_supported_extended_values() {
    let profile = JxlColorProfile::Simple(JxlColorEncoding::GrayscaleColorSpace {
        white_point: JxlWhitePoint::D65,
        transfer_function: JxlTransferFunction::Linear,
        rendering_intent: RenderingIntent::Relative,
    });
    let conversion = ColorConversion::new(
        &profile,
        false,
        SourceDynamicRange::Sdr,
        SourceIntensityTarget::from_jxl_metadata(1_000.0),
    )
    .unwrap();
    assert!(conversion.input_range.is_none());
    let pixels = [-0.25_f32, 2.0].map(|v| {
        [v, v, v, 1.0]
            .map(half::f16::from_f32)
            .map(half::f16::to_bits)
    });
    let (store, _) = convert_pixels(pixels.to_vec(), 2, 1, true, &conversion).unwrap();
    assert!(store.coarse_pixels()[0] < -0.24);
    assert!(store.coarse_pixels()[4] > 1.99);
    assert_eq!(
        hlg_inverse_oetf(-0.05).to_bits(),
        (-hlg_inverse_oetf(0.05)).to_bits()
    );
}

#[test]
fn publication_keeps_the_decoders_canonical_allocation() {
    let conversion = ColorConversion::new(
        &working_profile(),
        false,
        SourceDynamicRange::Sdr,
        SourceIntensityTarget::from_jxl_metadata(1_000.0),
    )
    .unwrap();
    let pixels = vec![[half::f16::ONE.to_bits(); 4]; 2];
    let address = pixels.as_ptr();
    let (store, _) = convert_pixels(pixels, 2, 1, true, &conversion).unwrap();
    assert_eq!(address, store.pixels.as_ptr());
}

#[test]
fn sampled_rgb_and_gray_curves_accept_nominal_samples_and_reject_clipping() {
    for mut profile in [
        ColorProfile::new_srgb(),
        ColorProfile::new_gray_with_gamma(2.0),
    ] {
        let curve = moxcms::ToneReprCurve::Lut(vec![0, 16384, 65535]);
        profile.cicp = None;
        if profile.color_space == DataColorSpace::Gray {
            profile.gray_trc = Some(curve);
        } else {
            profile.red_trc = Some(curve.clone());
            profile.green_trc = Some(curve.clone());
            profile.blue_trc = Some(curve);
        }
        let source = JxlColorProfile::Icc(profile.encode().unwrap());
        let conversion = ColorConversion::new(
            &source,
            false,
            SourceDynamicRange::Sdr,
            SourceIntensityTarget::from_jxl_metadata(1_000.0),
        )
        .unwrap();
        assert!(
            conversion
                .input_range
                .as_ref()
                .is_some_and(|range| { !range.contains(&-0.1) && !range.contains(&1.1) })
        );
        let pixel = |value: f32| {
            [value, value, value, 1.0]
                .map(half::f16::from_f32)
                .map(half::f16::to_bits)
        };
        let (store, _) = convert_pixels(vec![pixel(0.5)], 1, 1, true, &conversion).unwrap();
        assert!((store.coarse_pixels()[0] - 0.25).abs() < 0.0005);
        for value in [-0.1, 1.1] {
            assert!(matches!(
                convert_pixels(vec![pixel(value)], 1, 1, true, &conversion),
                Err(DecodeError::UnsupportedColorProfile(_))
            ));
        }
    }
}

#[test]
fn gamma_profiles_preserve_above_white_even_when_negative_values_are_unsupported() {
    let mut profile = ColorProfile::new_srgb();
    let curve = moxcms::ToneReprCurve::Lut(vec![512]); // u8Fixed8 gamma 2.0
    profile.cicp = None;
    profile.red_trc = Some(curve.clone());
    profile.green_trc = Some(curve.clone());
    profile.blue_trc = Some(curve);
    let conversion = ColorConversion::new(
        &JxlColorProfile::Icc(profile.encode().unwrap()),
        false,
        SourceDynamicRange::Sdr,
        SourceIntensityTarget::from_jxl_metadata(1_000.0),
    )
    .unwrap();
    assert!(
        conversion
            .input_range
            .as_ref()
            .is_none_or(|range| range.contains(&2.0))
    );
    let pixel = [2.0_f32, 2.0, 2.0, 1.0]
        .map(half::f16::from_f32)
        .map(half::f16::to_bits);
    let (store, _) = convert_pixels(vec![pixel], 1, 1, true, &conversion).unwrap();
    assert!((store.coarse_pixels()[0] - 4.0).abs() < 0.0005);
}

#[test]
fn lut_profiles_use_declared_intent_and_reject_extended_domain() {
    use moxcms::{LutDataType, LutStore, LutType, LutWarehouse};
    let table = |xyz: [u16; 3]| {
        LutWarehouse::Lut(LutDataType {
            num_input_channels: 3,
            num_output_channels: 3,
            num_clut_grid_points: 2,
            matrix: moxcms::Matrix3d {
                v: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            num_input_table_entries: 2,
            num_output_table_entries: 2,
            input_table: LutStore::Store16(vec![0, 65535, 0, 65535, 0, 65535]),
            clut_table: LutStore::Store16(xyz.repeat(8)),
            output_table: LutStore::Store16(vec![0, 65535, 0, 65535, 0, 65535]),
            lut_type: LutType::Lut16,
        })
    };
    let mut profile = ColorProfile::new_srgb();
    // PCS XYZ uses a scale of almost two. These encode neutral D50 at 50%
    // and 100%, making A2B selection observable independently of input RGB.
    profile.lut_a_to_b_perceptual = Some(table([15800, 16384, 13515]));
    profile.lut_a_to_b_colorimetric = Some(table([31600, 32768, 27030]));
    let mut results = Vec::new();
    for intent in [
        moxcms::RenderingIntent::Perceptual,
        moxcms::RenderingIntent::RelativeColorimetric,
    ] {
        profile.rendering_intent = intent;
        let source = JxlColorProfile::Icc(profile.encode().unwrap());
        let conversion = ColorConversion::new(
            &source,
            false,
            SourceDynamicRange::Sdr,
            SourceIntensityTarget::from_jxl_metadata(1_000.0),
        )
        .unwrap();
        assert!(
            conversion
                .input_range
                .as_ref()
                .is_some_and(|range| { !range.contains(&-0.1) && !range.contains(&1.1) })
        );
        let pixels = vec![
            [0.2_f32, 0.4, 0.6, 1.0]
                .map(half::f16::from_f32)
                .map(half::f16::to_bits),
        ];
        let (store, _) = convert_pixels(pixels, 1, 1, true, &conversion).unwrap();
        results.push(store.coarse_pixels()[0]);
        let pixels = vec![
            [-0.1_f32, 1.2, 0.6, 1.0]
                .map(half::f16::from_f32)
                .map(half::f16::to_bits),
        ];
        assert!(matches!(
            convert_pixels(pixels, 1, 1, true, &conversion),
            Err(DecodeError::UnsupportedColorProfile(_))
        ));
    }
    assert!((results[0] - 0.5).abs() < 0.005, "{results:?}");
    assert!((results[1] - 1.0).abs() < 0.01, "{results:?}");
}

fn linear_profile(primaries: JxlPrimaries) -> JxlColorProfile {
    JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
        white_point: JxlWhitePoint::D65,
        primaries,
        transfer_function: JxlTransferFunction::Linear,
        rendering_intent: RenderingIntent::Relative,
    })
}

fn packed(rgba: [f32; 4]) -> CanonicalPixel {
    rgba.map(half::f16::from_f32).map(half::f16::to_bits)
}

fn linear_conversion(profile: &JxlColorProfile, associated: bool) -> ColorConversion {
    ColorConversion::new(
        profile,
        associated,
        SourceDynamicRange::Linear,
        SourceIntensityTarget::FALLBACK,
    )
    .unwrap()
}

fn linear_source(raw_intensity: f32, xyb: bool, float_samples: bool) -> SourceCalibration {
    SourceCalibration {
        dynamic_range: SourceDynamicRange::Linear,
        raw_intensity,
        intensity: SourceIntensityTarget::from_jxl_metadata(raw_intensity),
        xyb,
        float_samples,
    }
}

#[test]
fn linear_unit_rule_scans_only_original_profile_floats_with_an_explicit_target() {
    // (raw target, XYB, float samples, source maximum, unit, scans)
    for (raw_t, xyb, float_samples, m, expected, scans) in [
        (255.0, false, false, 12.5, 203.0, false),
        (255.0, false, true, 12.5, 203.0, false),
        (255.0, true, false, 12.5, 203.0, false),
        (255.0, true, true, 12.5, 203.0, false),
        (1_000.0, false, false, 12.5, 1_000.0, false),
        (1_000.0, true, false, 12.5, 1_000.0, false),
        (80.0, true, true, 12.5, 80.0, false),
        (1_000.0, false, true, 0.5, 1_000.0, true),
        (1_000.0, false, true, 1.0, 1_000.0, true),
        (1_000.0, false, true, 12.5, 80.0, true),
        (80.0, false, true, 12.5, 6.4, true),
        (254.875, false, true, 1.0, 254.875, true),
    ] {
        let scanned = std::cell::Cell::new(false);
        let unit = linear_source(raw_t, xyb, float_samples)
            .linear_unit(|| {
                scanned.set(true);
                Ok(m)
            })
            .unwrap();
        assert!((unit.nits - expected).abs() < 0.0001);
        // Only the default target leaves the unit uncalibrated.
        assert_eq!(unit.assumed, raw_t == 255.0);
        assert_eq!(
            scanned.get(),
            scans,
            "T={raw_t} xyb={xyb} float={float_samples}"
        );
    }
    // The default check uses raw metadata, independently of the validated value.
    let unit = linear_source(f32::NAN, true, true)
        .linear_unit(|| unreachable!("XYB never scans"))
        .unwrap();
    assert_eq!(unit.nits, 1_000.0);
    assert!(matches!(
        linear_source(1_000.0, false, true).linear_unit(|| Err(profile_error("scan failed"))),
        Err(DecodeError::UnsupportedColorProfile(_))
    ));
}

#[test]
fn hdr_claim_needs_evidence_that_codec_overshoot_cannot_produce() {
    use SourceDynamicRange::{Hlg, Linear, Pq, Sdr};
    // (range, XYB, raw target, peak nits, HDR)
    for (dynamic_range, xyb, raw_intensity, peak, expected) in [
        // XYB with the default target: overshoot alone can exceed white.
        (Linear, true, 255.0, 206.4, false),
        (Linear, true, 255.0, 812.0, false),
        // An explicit target or exact original-profile samples support the claim.
        (Linear, true, 80.0, 1_000.0, true),
        (Linear, false, 255.0, 812.0, true),
        (Linear, false, 255.0, 203.0, false),
        (Linear, true, 1_000.0, 203.0, false),
        // PQ and HLG follow their target, whatever the measured peak.
        (Pq, true, 255.0, 203.0, true),
        (Hlg, false, 1_000.0, 211.2, true),
        (Hlg, false, 203.0, 1_927.4, false),
        (Pq, false, 150.0, 400.0, false),
        (Sdr, true, 255.0, 203.0, false),
    ] {
        let source = SourceCalibration {
            dynamic_range,
            raw_intensity,
            intensity: SourceIntensityTarget::from_jxl_metadata(raw_intensity),
            xyb,
            float_samples: true,
        };
        assert_eq!(
            source.hdr_source(SourceIntensityTarget::new(peak).unwrap()),
            expected,
            "{dynamic_range:?} xyb={xyb} T={raw_intensity} peak={peak}"
        );
    }
}

#[test]
fn hdr_signals_present_their_measured_peak_and_decode_by_their_target() {
    use JxlTransferFunction::{HLG, Linear, PQ};
    use SourceDynamicRange::{Hlg, Pq};
    let pq_nits = |signal| f64::from(pq_eotf(signal)) * 10_000.0;
    // (range, decoder output, target, sample, peak nits, HDR)
    for (dynamic_range, output, target, sample, expected, hdr) in [
        // Content far below libjxl's default PQ target.
        (Pq, PQ, 10_000.0, 0.75, pq_nits(0.75), true),
        // Content above its target is not capped,
        (Pq, PQ, 1_000.0, 0.8125, pq_nits(0.8125), true),
        // and makes no HDR claim when the target sits below white.
        (Pq, PQ, 150.0, 0.75, pq_nits(0.75), false),
        // HLG 0.5 displays at 51 nits; content below white presents at white.
        (Hlg, HLG, 1_000.0, 0.5, 203.0, true),
        (Hlg, HLG, 2_000.0, 1.0, 2_000.0, true),
        // The XYB fallback output is relative to the target.
        (Pq, Linear, 1_000.0, 0.5, 500.0, true),
    ] {
        let xyb = output == Linear;
        let profile = JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::BT2100,
            transfer_function: output,
            rendering_intent: RenderingIntent::Relative,
        });
        let intensity = SourceIntensityTarget::new(f64::from(target)).unwrap();
        let conversion = ColorConversion::new(&profile, false, dynamic_range, intensity).unwrap();
        let source = SourceCalibration {
            dynamic_range,
            raw_intensity: target,
            intensity,
            xyb,
            float_samples: true,
        };
        let pixels = vec![packed([sample, sample, sample, 1.0])];
        let calibrated = calibrate_and_convert(pixels, 1, 1, true, conversion, source).unwrap();
        let peak = calibrated.source_peak.nits();
        assert!(
            (peak - expected).abs() <= expected * 1e-5,
            "{dynamic_range:?} T={target} sample={sample}: {peak} vs {expected}"
        );
        assert_eq!(calibrated.hdr_source, hdr, "{dynamic_range:?} T={target}");
        assert_eq!(calibrated.linear_unit, None);
    }
}

#[test]
fn hdr_signals_beyond_the_decoder_range_saturate_instead_of_overflowing() {
    use JxlTransferFunction::{HLG, PQ};
    use SourceDynamicRange::{Hlg, Pq};
    // jxl clamps only converted XYB output. Unclamped original-profile floats
    // would decode to infinite PQ or f16-overflowing HLG luminance.
    let hlg_range = [-0.074_f32, 1.1].map(|bound| half::f16::from_f32(bound).to_f32());
    // (range, decoder output, signal range, out-of-range samples)
    for (dynamic_range, output, [lower, upper], samples) in [
        (Pq, PQ, [0.0, 1.0], [-0.5, 1.5, 2.5]),
        (Hlg, HLG, hlg_range, [-0.5, 1.2, 2.5]),
    ] {
        let profile = JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::BT2100,
            transfer_function: output,
            rendering_intent: RenderingIntent::Relative,
        });
        let intensity = SourceIntensityTarget::new(1_000.0).unwrap();
        let conversion = ColorConversion::new(&profile, false, dynamic_range, intensity).unwrap();
        for sample in samples {
            let bound = if sample < 0.0 { lower } else { upper };
            assert_eq!(
                conversion.unpack(packed([sample; 4]), 0),
                conversion.unpack(packed([bound; 4]), 0),
                "{dynamic_range:?} sample={sample}"
            );
        }
        let pixels = samples.map(|sample| packed([sample, sample, sample, 1.0]));
        convert_pixels(pixels.to_vec(), 3, 1, true, &conversion).unwrap();
    }
}

#[test]
fn conversion_table_rejects_inconsistent_outputs_and_keeps_sdr_fallback_relative() {
    use JxlTransferFunction::{HLG, PQ, SRGB};
    use SourceDynamicRange::{Hlg, Linear, Pq, Sdr};
    let linear = working_profile();
    assert_eq!(dynamic_range(&linear).unwrap(), Linear);
    let encoded = |transfer_function| {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::BT2100,
            transfer_function,
            rendering_intent: RenderingIntent::Relative,
        })
    };
    // The decoder cannot produce these pairs; reject them instead of guessing.
    for (range, output) in [
        (Linear, SRGB),
        (Sdr, PQ),
        (Sdr, HLG),
        (Pq, HLG),
        (Hlg, PQ),
        (Pq, SRGB),
    ] {
        assert!(
            matches!(
                ColorConversion::new(
                    &encoded(output),
                    false,
                    range,
                    SourceIntensityTarget::FALLBACK
                ),
                Err(DecodeError::UnsupportedColorProfile(_))
            ),
            "{range:?}"
        );
    }
    // An embedded ICC profile is still SDR even if its own TRC is linear.
    assert_eq!(
        dynamic_range(&JxlColorProfile::Icc(
            linear.try_as_icc().unwrap().into_owned()
        ))
        .unwrap(),
        SourceDynamicRange::Sdr
    );
    let sdr = ColorConversion::new(
        &linear,
        false,
        SourceDynamicRange::Sdr,
        SourceIntensityTarget::FALLBACK,
    )
    .unwrap();
    assert_eq!(sdr.scale, 1.0);
    for range in [SourceDynamicRange::Pq, SourceDynamicRange::Hlg] {
        let conversion =
            ColorConversion::new(&linear, false, range, SourceIntensityTarget::FALLBACK).unwrap();
        assert_eq!(conversion.unpack(packed([0.5; 4]), 0), 0.5);
        assert!(conversion.hlg_gamma.is_none());
        assert!((conversion.scale - 1_000.0 / HDR_REFERENCE_WHITE_NITS).abs() < 1e-6);
    }
}

#[test]
fn luminance_calibration_counts_hidden_straight_color_and_ignores_saturated_max_rgb() {
    let conversion = linear_conversion(&linear_profile(JxlPrimaries::SRGB), false);
    let source = linear_source(1_000.0, false, true);
    let pixels = [
        packed([12.5, 12.5, 12.5, 1.0]),
        packed([15.9, 0.0, 0.0, 1.0]),
    ];
    let maximum = conversion.maximum_luminance(&pixels).unwrap();
    // ICC matrix coefficients have s15Fixed16 precision.
    assert!((maximum - 12.5).abs() < 0.001);
    let unit = source
        .linear_unit(|| conversion.maximum_luminance(&pixels))
        .unwrap();
    assert!((unit.nits - 80.0).abs() < 0.01);
    let hidden = [packed([12.5, 12.5, 12.5, 0.0]), pixels[1]];
    assert_eq!(conversion.maximum_luminance(&hidden).unwrap(), maximum);
    let cropped = source
        .linear_unit(|| conversion.maximum_luminance(&hidden[1..]))
        .unwrap();
    assert!(cropped.nits > unit.nits * 3.0);
}

#[test]
fn luminance_pass_uses_recoverable_associated_color_and_gray_channel_zero() {
    let straight = linear_conversion(&working_profile(), false);
    let associated = linear_conversion(&working_profile(), true);
    let invalid = [
        packed([f32::NAN, 100.0, 100.0, 1.0]),
        packed([100.0, f32::INFINITY, 100.0, 1.0]),
        packed([100.0, 100.0, 100.0, f32::NAN]),
    ];
    assert_eq!(straight.maximum_luminance(&invalid).unwrap(), 0.0);
    let pixels = [
        packed([6.25, 6.25, 6.25, 0.5]),
        packed([100.0, 100.0, 100.0, 0.0]),
    ];
    assert!((associated.maximum_luminance(&pixels).unwrap() - 12.5).abs() < 1e-6);
    assert!((straight.maximum_luminance(&pixels).unwrap() - 100.0).abs() < 1e-5);
    let gray = JxlColorProfile::Simple(JxlColorEncoding::GrayscaleColorSpace {
        white_point: JxlWhitePoint::D65,
        transfer_function: JxlTransferFunction::Linear,
        rendering_intent: RenderingIntent::Relative,
    });
    let gray = linear_conversion(&gray, false);
    assert_eq!(
        gray.maximum_luminance(&[packed([12.5, 100.0, 200.0, 0.0])])
            .unwrap(),
        12.5
    );
}

#[test]
fn native_linear_peak_is_visible_full_resolution_straight_color_without_a_target_floor() {
    for (sample, unit_nits, expected, tolerance) in [
        (4.0, 203.0, 812.0, 0.001),
        (0.5, 203.0, 203.0, 0.001),
        (1_000.0, 203.0, 10_000.0, 0.001),
        (0.1, 10_000.0, 1_000.0, 0.25), // f16 quantizes 0.1 by 0.0244%.
        (0.1, 1_000.0, 203.0, 0.001),   // Also applies to integer sources.
        (1.534, 1_000.0, 1_534.0, 0.5), // XYB may decode above 1.0.
    ] {
        let mut conversion = linear_conversion(&working_profile(), false);
        conversion.scale = unit_nits / HDR_REFERENCE_WHITE_NITS;
        let pixels = vec![packed([sample, sample, sample, 1.0])];
        let (_, peak) = convert_pixels(pixels, 1, 1, true, &conversion).unwrap();
        assert!((measured_source_peak(peak).nits() - expected).abs() < tolerance);
    }
    let conversion = linear_conversion(&working_profile(), false);
    // Downsample=2, and the single highlight is in another parallel band on
    // multi-worker pools. Its full-resolution peak must survive averaging.
    let mut pixels = vec![packed([0.5, 0.5, 0.5, 1.0]); 4_097 * 18];
    pixels[0] = packed([20.0, 20.0, 20.0, 0.0]);
    *pixels.last_mut().unwrap() = packed([4.0, 4.0, 4.0, 0.25]);
    let (store, peak) = convert_pixels(pixels, 4_097, 18, false, &conversion).unwrap();
    assert_eq!(store.coarse_downsample(), 2);
    assert_eq!(measured_source_peak(peak).nits(), 812.0);
    assert!(store.coarse_pixels().iter().all(|v| *v < 4.0));
    let mut associated = linear_conversion(&working_profile(), true);
    let pixels = vec![packed([6.25, 6.25, 6.25, 0.5])];
    let unit = linear_source(1_000.0, false, true)
        .linear_unit(|| associated.maximum_luminance(&pixels))
        .unwrap();
    assert!((unit.nits - 80.0).abs() < 0.001);
    associated.scale = unit.nits / HDR_REFERENCE_WHITE_NITS;
    let (store, peak) = convert_pixels(pixels, 1, 1, false, &associated).unwrap();
    assert!((measured_source_peak(peak).nits() - 1_000.0).abs() < 0.001);
    assert!((store.coarse_pixels()[0] - 1_000.0 / HDR_REFERENCE_WHITE_NITS).abs() < 0.001);
}

#[test]
fn ordinary_linear_white_snaps_to_sdr_but_the_next_f16_step_is_hdr() {
    for primaries in [JxlPrimaries::SRGB, JxlPrimaries::P3, JxlPrimaries::BT2100] {
        for associated in [false, true] {
            let conversion = linear_conversion(&linear_profile(primaries.clone()), associated);
            let sample = if associated { 0.5 } else { 1.0 };
            let (_, peak) = convert_pixels(
                vec![packed([sample, sample, sample, 0.5])],
                1,
                1,
                false,
                &conversion,
            )
            .unwrap();
            assert_eq!(measured_source_peak(peak).nits(), 203.0);
            let sample = sample * (1.0 + 1.0 / 1024.0);
            let (_, peak) = convert_pixels(
                vec![packed([sample, sample, sample, 0.5])],
                1,
                1,
                false,
                &conversion,
            )
            .unwrap();
            assert!(measured_source_peak(peak).nits() > 203.0);
        }
    }
}

#[test]
fn conversion_rejects_nonfinite_half_output_before_publication() {
    for range in [SourceDynamicRange::Linear, SourceDynamicRange::Sdr] {
        let mut conversion = ColorConversion::new(
            &working_profile(),
            false,
            range,
            SourceIntensityTarget::FALLBACK,
        )
        .unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                convert_pixels(
                    vec![packed([value, 0.5, 0.5, 1.0])],
                    1,
                    1,
                    true,
                    &conversion
                ),
                Err(DecodeError::NonFiniteOutput)
            ));
        }
        conversion.scale = 2.0;
        assert!(matches!(
            convert_pixels(
                vec![packed([65_504.0, 1.0, 1.0, 1.0])],
                1,
                1,
                true,
                &conversion
            ),
            Err(DecodeError::NonFiniteOutput)
        ));
    }
}

#[test]
fn color_encoding_labels_use_standard_names() {
    assert_eq!(primaries_label(&JxlPrimaries::SRGB), "sRGB");
    assert_eq!(primaries_label(&JxlPrimaries::BT2100), "BT.2020");
    assert_eq!(primaries_label(&JxlPrimaries::P3), "P3");
    assert_eq!(
        primaries_label(&JxlPrimaries::Chromaticities {
            rx: 0.64,
            ry: 0.33,
            gx: 0.3,
            gy: 0.6,
            bx: 0.15,
            by: 0.06,
        }),
        "custom"
    );

    assert_eq!(white_point_label(&JxlWhitePoint::D65), "D65");
    assert_eq!(white_point_label(&JxlWhitePoint::DCI), "DCI");
    assert_eq!(white_point_label(&JxlWhitePoint::E), "E (equal-energy)");
    assert_eq!(
        white_point_label(&JxlWhitePoint::Chromaticity {
            wx: 0.345_669,
            wy: 0.358_496,
        }),
        "D50"
    );
    assert_eq!(
        white_point_label(&JxlWhitePoint::Chromaticity { wx: 0.3, wy: 0.32 }),
        "x 0.3000, y 0.3200"
    );

    for (transfer, expected) in [
        (JxlTransferFunction::SRGB, "sRGB"),
        (JxlTransferFunction::BT709, "BT.709"),
        (JxlTransferFunction::Linear, "Linear"),
        (JxlTransferFunction::PQ, "PQ"),
        (JxlTransferFunction::HLG, "HLG"),
        (JxlTransferFunction::DCI, "DCI gamma 2.6"),
        (JxlTransferFunction::Gamma(1.0 / 2.2), "Gamma 2.2"),
        (JxlTransferFunction::Gamma(0.5), "Gamma 2"),
        (JxlTransferFunction::Gamma(0.1), "Gamma 10"),
    ] {
        assert_eq!(transfer_label(&transfer), expected);
    }
}

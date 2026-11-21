// Parsed ratios and gains of whole stops are exactly representable.
#![expect(clippy::float_cmp)]

use jxl::api::{JxlTransferFunction, RenderingIntent};

use super::*;

const LIMIT: usize = 1 << 20;

fn boxed(box_type: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut bytes = u32::try_from(payload.len() + 8)
        .unwrap()
        .to_be_bytes()
        .to_vec();
    bytes.extend_from_slice(&box_type);
    bytes.extend_from_slice(payload);
    bytes
}

fn container(boxes: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = b"\0\0\0\x0cJXL \r\n\x87\n".to_vec();
    bytes.extend(boxed(*b"ftyp", b"jxl \0\0\0\0jxl "));
    bytes.extend(boxes.iter().flatten());
    bytes
}

/// Returns at most `step` bytes per read, like a slow file.
struct Trickle<'a> {
    remaining: &'a [u8],
    step: usize,
}

impl JxlBitstreamInput for Trickle<'_> {
    fn available_bytes(&mut self) -> io::Result<usize> {
        Ok(self.remaining.len())
    }

    fn read(&mut self, buffers: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let Some(buffer) = buffers.first_mut() else {
            return Ok(0);
        };
        let count = self.step.min(buffer.len()).min(self.remaining.len());
        buffer[..count].copy_from_slice(&self.remaining[..count]);
        self.remaining = &self.remaining[count..];
        Ok(count)
    }
}

/// Reads the whole container through the wrapper, checking it forwards the
/// bytes unchanged.
fn scan(bytes: &[u8], limit: usize, step: usize) -> Capture {
    let mut inner = Trickle {
        remaining: bytes,
        step,
    };
    let mut input = CaptureInput::new(&mut inner, limit);
    let mut forwarded = Vec::new();
    let mut buffer = [0_u8; 97];
    loop {
        let count = input.read(&mut [IoSliceMut::new(&mut buffer)]).unwrap();
        if count == 0 {
            break;
        }
        forwarded.extend_from_slice(&buffer[..count]);
    }
    assert_eq!(forwarded, bytes);
    input.finish(false).unwrap()
}

fn payload(capture: Capture) -> (Vec<u8>, bool) {
    match capture {
        Capture::Payload(payload) => (payload.data, payload.compressed),
        Capture::None => panic!("no gain map box was captured"),
        Capture::Skipped(reason) => panic!("the gain map box was skipped: {reason}"),
    }
}

fn skipped(capture: &Capture) -> Reason {
    match capture {
        Capture::Skipped(reason) => *reason,
        Capture::None => panic!("no gain map box was seen"),
        Capture::Payload(_) => panic!("the gain map box was captured"),
    }
}

#[test]
fn finite_boxes_are_captured_at_every_position_and_read_size() {
    let gain_map: Vec<u8> = (0..=255).cycle().take(1_000).collect();
    let codestream = boxed(*b"jxlc", &[0xff; 300]);
    for boxes in [
        vec![boxed(*b"jhgm", &gain_map), codestream.clone()],
        vec![
            boxed(*b"Exif", &[1; 77]),
            codestream.clone(),
            boxed(*b"jhgm", &gain_map),
            boxed(*b"xml ", &[2; 5]),
        ],
    ] {
        let bytes = container(&boxes);
        for step in [1, 7, 4096] {
            assert_eq!(
                payload(scan(&bytes, LIMIT, step)),
                (gain_map.clone(), false)
            );
        }
    }
}

#[test]
fn skipped_bytes_still_pass_through_the_capture() {
    let gain_map = [9_u8; 5_000];
    let bytes = container(&[boxed(*b"jhgm", &gain_map), boxed(*b"jxlc", &[0; 16])]);
    let mut inner = bytes.as_slice();
    let mut input = CaptureInput::new(&mut inner, LIMIT);

    // `jxl` skips boxes it was not asked for. A native skip of the inner
    // input would hide them.
    assert_eq!(input.skip(usize::MAX).unwrap(), bytes.len());

    assert_eq!(payload(input.finish(false).unwrap()).0, gain_map);
}

#[test]
fn compressed_and_large_size_boxes_are_recognized() {
    let mut wrapped = b"jhgm".to_vec();
    wrapped.extend_from_slice(b"compressed bundle");
    let bytes = container(&[boxed(*b"brob", &wrapped)]);
    assert_eq!(
        payload(scan(&bytes, LIMIT, 3)),
        (b"compressed bundle".to_vec(), true)
    );

    // A 64-bit size follows the type when the 32-bit size is 1.
    let mut large = 1_u32.to_be_bytes().to_vec();
    large.extend_from_slice(b"jhgm");
    large.extend_from_slice(&(16_u64 + 4).to_be_bytes());
    large.extend_from_slice(b"data");
    let bytes = container(&[large, boxed(*b"jxlc", &[0; 4])]);
    assert_eq!(payload(scan(&bytes, LIMIT, 5)), (b"data".to_vec(), false));

    // Another wrapped type is not a gain map.
    let mut other = b"xml ".to_vec();
    other.extend_from_slice(b"jhgm");
    let bytes = container(&[boxed(*b"brob", &other)]);
    assert!(matches!(scan(&bytes, LIMIT, 5), Capture::None));
}

#[test]
fn a_box_reaching_the_end_of_the_file_is_complete_only_once_it_was_read() {
    let mut bytes = container(&[boxed(*b"jxlc", &[0; 64])]);
    let prefix = bytes.len();
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(b"jhgm");
    bytes.extend_from_slice(&vec![7; 30_000]);

    let mut inner = bytes.as_slice();
    let mut input = CaptureInput::new(&mut inner, LIMIT);
    // The decoder stops reading shortly after the header of such a box.
    let mut buffer = vec![0_u8; prefix + 20];
    input.read(&mut [IoSliceMut::new(&mut buffer)]).unwrap();
    assert_eq!(payload(input.finish(true).unwrap()).0, vec![7; 30_000]);

    // Without reading to the end, the partial payload is not a gain map.
    let mut inner = bytes.as_slice();
    let mut input = CaptureInput::new(&mut inner, LIMIT);
    input.read(&mut [IoSliceMut::new(&mut buffer)]).unwrap();
    assert!(matches!(input.finish(false).unwrap(), Capture::None));

    // The same holds for a finite box that is not complete yet.
    let bytes = container(&[boxed(*b"jhgm", &[1; 100])]);
    let mut inner = &bytes[..bytes.len() - 1];
    let mut input = CaptureInput::new(&mut inner, LIMIT);
    input.skip(usize::MAX).unwrap();
    assert!(matches!(input.finish(false).unwrap(), Capture::None));
}

#[test]
fn oversized_and_repeated_boxes_are_skipped_with_a_reason() {
    let codestream = boxed(*b"jxlc", &[0; 8]);

    // The declared size already exceeds the limit, so nothing is buffered.
    let bytes = container(&[boxed(*b"jhgm", &[0; 101]), codestream.clone()]);
    assert_eq!(skipped(&scan(&bytes, 100, 16)), Reason::TooLarge);
    assert_eq!(payload(scan(&bytes, 101, 16)).0.len(), 101);

    // A box without a declared size is bounded while it streams.
    let mut open = container(std::slice::from_ref(&codestream));
    open.extend_from_slice(&0_u32.to_be_bytes());
    open.extend_from_slice(b"jhgm");
    open.extend_from_slice(&[0; 101]);
    assert_eq!(skipped(&scan(&open, 100, 16)), Reason::TooLarge);

    // The exact budget arrives after an early box was already captured.
    let mut inner = bytes.as_slice();
    let mut input = CaptureInput::new(&mut inner, LIMIT);
    input.skip(usize::MAX).unwrap();
    input.set_budget(100);
    assert_eq!(skipped(&input.finish(false).unwrap()), Reason::TooLarge);

    let bytes = container(&[
        boxed(*b"jhgm", &[1; 10]),
        codestream,
        boxed(*b"jhgm", &[2; 10]),
    ]);
    assert_eq!(skipped(&scan(&bytes, LIMIT, 16)), Reason::Unsupported);
}

#[test]
fn bare_codestreams_and_broken_framing_capture_nothing() {
    let mut bare = vec![0xff, 0x0a];
    bare.extend(boxed(*b"jhgm", &[0; 32]));
    assert!(matches!(scan(&bare, LIMIT, 4), Capture::None));

    // A box smaller than its own header ends the scan. The decoder reports it.
    let mut broken = 4_u32.to_be_bytes().to_vec();
    broken.extend_from_slice(b"junk");
    let bytes = container(&[broken, boxed(*b"jhgm", &[0; 32])]);
    assert!(matches!(scan(&bytes, LIMIT, 4), Capture::None));
}

const UNIT: [(i32, u32); 5] = [(0, 1), (0, 1), (1, 1), (0, 1), (0, 1)];

fn metadata(flags: u8, headrooms: [(u32, u32); 2], channels: &[[(i32, u32); 5]]) -> Vec<u8> {
    let mut bytes = vec![0, 0, 0, 0, flags];
    for (numerator, denominator) in headrooms {
        bytes.extend_from_slice(&numerator.to_be_bytes());
        bytes.extend_from_slice(&denominator.to_be_bytes());
    }
    for (numerator, denominator) in channels.iter().flatten() {
        bytes.extend_from_slice(&numerator.to_be_bytes());
        bytes.extend_from_slice(&denominator.to_be_bytes());
    }
    bytes
}

fn metadata_error(bytes: &[u8]) -> Reason {
    Metadata::parse(bytes)
        .err()
        .expect("metadata should be rejected")
}

#[test]
fn metadata_parses_per_channel_rationals() {
    let red = [(-3, 2), (5, 2), (5, 4), (1, 64), (1, 32)];
    let green = [(-1, 1), (2, 1), (1, 1), (0, 1), (-1, 8)];
    let blue = [(0, 1), (1, 1), (3, 4), (1, 4), (1, 2)];
    let parsed = Metadata::parse(&metadata(0xc0, [(0, 1), (23, 10)], &[red, green, blue])).unwrap();
    assert!(parsed.base_color_space);
    assert_eq!(parsed.base_headroom, 0.0);
    assert_eq!(parsed.alternate_headroom, 2.3);
    assert_eq!(parsed.channels[0].gain_min, -1.5);
    assert_eq!(parsed.channels[0].gain_max, 2.5);
    assert_eq!(parsed.channels[0].exponent, 0.8);
    assert_eq!(parsed.channels[0].base_offset, 1.0 / 64.0);
    assert_eq!(parsed.channels[0].alternate_offset, 1.0 / 32.0);
    assert_eq!(parsed.channels[1].alternate_offset, -0.125);
    assert_eq!(parsed.channels[2].exponent, 4.0 / 3.0);

    // One set of parameters applies to all three channels.
    let parsed = Metadata::parse(&metadata(0, [(1, 2), (3, 1)], &[red])).unwrap();
    assert!(!parsed.base_color_space);
    assert_eq!(parsed.base_headroom, 0.5);
    assert!(
        parsed
            .channels
            .iter()
            .all(|channel| channel.exponent == 0.8)
    );

    // A later writer version may append fields.
    let mut extended = metadata(0x40, [(0, 1), (1, 1)], &[UNIT]);
    extended[3] = 1;
    extended.extend_from_slice(&[0xaa; 8]);
    assert!(Metadata::parse(&extended).is_ok());
}

#[test]
fn metadata_rejects_reserved_flags_and_unusable_values() {
    let valid = metadata(0xc0, [(0, 1), (1, 1)], &[UNIT; 3]);
    assert!(Metadata::parse(&valid).is_ok());

    // Reserved bits include the legacy common-denominator and direction flags.
    for bit in 0..6 {
        let mut bytes = valid.clone();
        bytes[4] |= 1 << bit;
        assert_eq!(
            metadata_error(&bytes),
            Reason::Unsupported,
            "flag bit {bit}"
        );
    }
    let mut newer = valid.clone();
    newer[1] = 1;
    assert_eq!(metadata_error(&newer), Reason::Unsupported);

    for length in 0..valid.len() {
        assert_eq!(metadata_error(&valid[..length]), Reason::Malformed);
    }
    assert_eq!(
        metadata_error(&metadata(0, [(0, 0), (1, 1)], &[UNIT])),
        Reason::Malformed
    );
    for (field, value) in [
        (0, (0, 0)),    // zero denominator
        (2, (0, 1)),    // gamma of zero
        (0, (1, 1)),    // minimum above the maximum
        (1, (200, 1)),  // maximum gain overflows f32
        (0, (-200, 1)), // minimum gain underflows to zero
        (4, (1, 0)),    // offset with a zero denominator
    ] {
        let mut channel = UNIT;
        channel[field] = value;
        assert_eq!(
            metadata_error(&metadata(0, [(0, 1), (1, 1)], &[channel])),
            Reason::Malformed,
            "field {field} = {value:?}"
        );
    }

    // Large but representable gains, negative gains and nonzero offsets are valid.
    let extreme = [
        (-10, 1),
        (12_799_999, 100_000),
        (1, u32::MAX),
        (-1, 3),
        (5, 1),
    ];
    assert!(Metadata::parse(&metadata(0, [(0, 1), (1, 1)], &[extreme])).is_ok());
}

fn bundle(metadata: &[u8], color_encoding: &[u8], icc: &[u8], codestream: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0];
    bytes.extend_from_slice(&u16::try_from(metadata.len()).unwrap().to_be_bytes());
    bytes.extend_from_slice(metadata);
    bytes.push(u8::try_from(color_encoding.len()).unwrap());
    bytes.extend_from_slice(color_encoding);
    bytes.extend_from_slice(&u32::try_from(icc.len()).unwrap().to_be_bytes());
    bytes.extend_from_slice(icc);
    bytes.extend_from_slice(codestream);
    bytes
}

#[test]
fn bundles_split_into_their_fields() {
    let bytes = bundle(b"meta", &[], &[], b"codestream");
    let parsed = Bundle::parse(&bytes).unwrap();
    assert_eq!(parsed.metadata, b"meta");
    assert_eq!(parsed.codestream, b"codestream");
    assert_eq!(parsed.alternate_encoding(), Ok(None));

    // An all-default color encoding is sRGB.
    let srgb = bundle(b"meta", &[1], &[], b"codestream");
    assert!(matches!(
        Bundle::parse(&srgb).unwrap().alternate_encoding(),
        Ok(Some(JxlColorEncoding::RgbColorSpace {
            primaries: JxlPrimaries::SRGB,
            ..
        }))
    ));

    // The ICC bytes declare a 255 MiB profile. They must never be decoded.
    let icc = bundle(b"meta", &[], &[0xf8, 0xff, 0xff, 0x7f, 0], b"codestream");
    assert_eq!(
        Bundle::parse(&icc).unwrap().alternate_encoding(),
        Err(Reason::UnsupportedProfile)
    );

    let mut newer = bytes.clone();
    newer[0] = 1;
    assert_eq!(Bundle::parse(&newer).err(), Some(Reason::Unsupported));
    assert_eq!(
        Bundle::parse(&bundle(b"meta", &[], &[], b"")).err(),
        Some(Reason::Malformed)
    );
    for length in 0..bytes.len() - b"codestream".len() {
        assert_eq!(
            Bundle::parse(&bytes[..length]).err(),
            Some(Reason::Malformed)
        );
    }
}

fn rgb_encoding(primaries: JxlPrimaries) -> JxlColorEncoding {
    JxlColorEncoding::RgbColorSpace {
        white_point: JxlWhitePoint::D65,
        primaries,
        transfer_function: JxlTransferFunction::SRGB,
        rendering_intent: RenderingIntent::Relative,
    }
}

#[test]
fn gain_spaces_are_matrices_to_and_from_linear_bt2020() {
    assert!(
        GainSpace::for_encoding(&rgb_encoding(JxlPrimaries::BT2100))
            .unwrap()
            .is_none()
    );
    let p3 = GainSpace::for_encoding(&rgb_encoding(JxlPrimaries::P3))
        .unwrap()
        .unwrap();
    for (actual, expected) in multiply(&p3.from_gain, [80_000.0, 0.0, 0.0])
        .into_iter()
        .zip([60_307.0, 3_659.0, -96.7])
    {
        assert!((actual - expected).abs() < 2.0, "{actual} vs {expected}");
    }
    let srgb = GainSpace::for_encoding(&rgb_encoding(JxlPrimaries::SRGB))
        .unwrap()
        .unwrap();
    // White stays white, and BT.2020 red lies outside sRGB.
    for value in multiply(&srgb.from_gain, [1.0; 3]) {
        assert!((value - 1.0).abs() < 1e-4);
    }
    let red = multiply(&srgb.to_gain, [1.0, 0.0, 0.0]);
    assert!((red[0] - 1.6605).abs() < 1e-3 && red[1] < 0.0 && red[2] < 0.0);

    assert_eq!(
        GainSpace::for_encoding(&JxlColorEncoding::GrayscaleColorSpace {
            white_point: JxlWhitePoint::D65,
            transfer_function: JxlTransferFunction::SRGB,
            rendering_intent: RenderingIntent::Relative,
        })
        .err(),
        Some(Reason::UnsupportedProfile)
    );
}

#[test]
fn unusable_matrices_are_rejected() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(GainSpace::from_matrix(identity).is_some());

    // A singular matrix has no inverse. The identity is not a substitute.
    let singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]];
    assert!(inverse(&singular).is_none());
    assert!(GainSpace::from_matrix(singular).is_none());

    // Invertible, but too poorly conditioned to round-trip colors.
    let skewed = [[1.0, 0.0, 0.0], [1_000.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(inverse(&skewed).is_some());
    assert!(GainSpace::from_matrix(skewed).is_none());

    // Well conditioned, but its rows overflow the intermediate range.
    let huge = identity.map(|row| row.map(|value: f64| value * 1e12));
    assert!(GainSpace::from_matrix(huge).is_none());

    // The f32 inverse of this matrix is not accurate enough.
    let imprecise = [[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-6, 0.0], [0.0, 0.0, 1.0]];
    assert!(GainSpace::from_matrix(imprecise).is_none());
    assert!(GainSpace::from_matrix([[f64::NAN; 3]; 3]).is_none());
}

/// Gain range in stops, gamma, then the base and alternate offsets.
type Parameters = (f64, f64, f64, (f32, f32));

#[expect(clippy::cast_possible_truncation)] // As when metadata is parsed.
fn channel(gain_min: f64, gain_max: f64, gamma: f64, offsets: (f32, f32)) -> Channel {
    Channel {
        gain_min: gain_min as f32,
        gain_max: gain_max as f32,
        exponent: (1.0 / gamma) as f32,
        base_offset: offsets.0,
        alternate_offset: offsets.1,
    }
}

fn gain_map(
    samples: Samples,
    channels: [Channel; 3],
    space: Option<GainSpace>,
    base: (u32, u32),
) -> GainMap {
    let metadata = Metadata {
        base_color_space: true,
        base_headroom: 0.0,
        alternate_headroom: 1.0,
        channels,
    };
    let base = BaseImage {
        width: base.0,
        height: base.1,
        encoding: None,
        dynamic_range: SourceDynamicRange::Sdr,
        identity_orientation: true,
        animated: false,
    };
    GainMap::new(samples, &metadata, space, &base)
}

fn samples(width: usize, height: usize, channels: usize, data: &[u16]) -> Samples {
    assert_eq!(data.len(), width * height * channels);
    Samples {
        data: data.to_vec(),
        width,
        height,
        channels,
    }
}

/// The ISO 21496-1 formula in double precision.
fn reference(base: f64, sample: f64, parameters: Parameters) -> f64 {
    let (gain_min, gain_max, gamma, (base_offset, alternate_offset)) = parameters;
    let log_gain = gain_min + sample.powf(1.0 / gamma) * (gain_max - gain_min);
    (base + f64::from(base_offset)) * log_gain.exp2() - f64::from(alternate_offset)
}

#[test]
fn full_size_maps_follow_the_gain_formula_per_channel() {
    let parameters: [Parameters; 3] = [
        (-1.5, 2.5, 1.25, (1.0 / 64.0, 1.0 / 64.0)),
        (-2.0, 3.0, 1.0, (0.0, 0.25)),
        (0.0, 1.0, 0.75, (0.5, 0.0)),
    ];
    let channels = parameters
        .map(|(gain_min, gain_max, gamma, offsets)| channel(gain_min, gain_max, gamma, offsets));
    let codes = [
        0, 1, 255, 12_345, 40_000, 65_534, 65_535, 30_000, 77, 9, 65_000, 3,
    ];
    let map = gain_map(samples(2, 2, 3, &codes), channels, None, (2, 2));
    let base = [
        0.0, 0.25, 1.0, 0.5, 0.75, 0.125, 1.0, 1.0, 1.0, 0.01, 0.9, 0.3_f32,
    ];

    // The second row alone, to check the pixel offset.
    let mut row = base[6..].to_vec();
    map.apply(&mut row, 2);
    let mut rgb = base.to_vec();
    map.apply(&mut rgb, 0);
    assert_eq!(rgb[6..], row);

    for (index, actual) in rgb.iter().enumerate() {
        let expected = reference(
            f64::from(base[index]),
            f64::from(codes[index]) / 65_535.0,
            parameters[index % 3],
        );
        assert!(
            (f64::from(*actual) - expected).abs() <= expected.abs() * 1e-5 + 1e-6,
            "sample {index}: {actual} vs {expected}"
        );
    }
}

#[test]
fn smaller_maps_are_sampled_bilinearly_at_pixel_centers() {
    let linear = channel(0.0, 4.0, 1.0, (0.0, 0.0));
    // 2x1 map on a 4x2 base: centers fall at -0.25, 0.25, 0.75 and 1.25 map
    // pixels, clamped to the edge samples.
    let map = gain_map(samples(2, 1, 1, &[0, 65_535]), [linear; 3], None, (4, 2));
    let mut rgb = vec![1.0_f32; 4 * 2 * 3];
    map.apply(&mut rgb, 0);
    for (pixel, weight) in rgb.chunks_exact(3).zip([0.0, 0.25, 0.75, 1.0].repeat(2)) {
        let expected = (4.0_f32 * weight).exp2();
        for value in pixel {
            assert!(
                (value - expected).abs() <= expected * 2e-4,
                "{value} vs {expected}"
            );
        }
    }

    // Vertical weights use the same rule.
    let map = gain_map(samples(1, 2, 1, &[0, 65_535]), [linear; 3], None, (1, 4));
    let mut rgb = vec![1.0_f32; 4 * 3];
    map.apply(&mut rgb, 0);
    for (pixel, weight) in rgb.chunks_exact(3).zip([0.0_f32, 0.25, 0.75, 1.0]) {
        let expected = (4.0 * weight).exp2();
        assert!((pixel[0] - expected).abs() <= expected * 2e-4);
    }
}

#[test]
fn samples_between_codes_follow_the_formula_where_it_is_steep() {
    // With gamma 4, the gain rises a quarter stop between the two lowest
    // codes. Interpolating gains per code there would be 7% too dark.
    let steep: Parameters = (0.0, 4.0, 4.0, (0.0, 0.0));
    let map = gain_map(
        samples(2, 1, 1, &[0, 1]),
        [channel(steep.0, steep.1, steep.2, steep.3); 3],
        None,
        (4, 2),
    );
    let mut rgb = vec![1.0_f32; 4 * 2 * 3];
    map.apply(&mut rgb, 0);

    // Pixel centers fall at 0, 0.25, 0.75 and 1 between the two samples.
    for (pixel, code) in rgb.chunks_exact(3).zip([0.0, 0.25, 0.75, 1.0]) {
        let expected = reference(1.0, code / 65_535.0, steep);
        assert!(
            (f64::from(pixel[0]) - expected).abs() <= 1e-5,
            "{} vs {expected}",
            pixel[0]
        );
    }
    assert!((rgb[3] - 1.130_356).abs() < 1e-5);
}

#[test]
fn channel_counts_of_map_and_parameters_broadcast_both_ways() {
    let per_channel = [
        channel(0.0, 1.0, 1.0, (0.0, 0.0)),
        channel(0.0, 2.0, 1.0, (0.0, 0.0)),
        channel(0.0, 3.0, 1.0, (0.0, 0.0)),
    ];
    // One map channel drives all three parameter sets.
    let map = gain_map(samples(1, 1, 1, &[65_535]), per_channel, None, (1, 1));
    let mut rgb = [1.0_f32; 3];
    map.apply(&mut rgb, 0);
    assert_eq!(rgb, [2.0, 4.0, 8.0]);

    // One parameter set applies to all three map channels.
    let map = gain_map(
        samples(1, 1, 3, &[0, 65_535, 0]),
        [per_channel[1]; 3],
        None,
        (1, 1),
    );
    let mut rgb = [1.0_f32; 3];
    map.apply(&mut rgb, 0);
    assert_eq!(rgb, [1.0, 4.0, 1.0]);
}

#[test]
fn gains_apply_in_the_gain_color_space() {
    let space = || {
        GainSpace::for_encoding(&rgb_encoding(JxlPrimaries::P3))
            .unwrap()
            .unwrap()
    };
    // Unit gain: a valid BT.2020 color far outside P3's f16 range in gain
    // space must come back unclipped.
    let unit = channel(0.0, 0.0, 1.0, (0.0, 0.0));
    let map = gain_map(samples(1, 1, 1, &[0]), [unit; 3], Some(space()), (1, 1));
    let input = multiply(&space().from_gain, [80_000.0, 0.0, 0.0]);
    let mut rgb = input;
    map.apply(&mut rgb, 0);
    for (actual, expected) in rgb.into_iter().zip(input) {
        assert!((actual - expected).abs() <= expected.abs() * 1e-4 + 0.1);
    }

    // Doubling only P3 red is not a per-channel gain in BT.2020.
    let red_only = [channel(1.0, 1.0, 1.0, (0.0, 0.0)), unit, unit];
    let map = gain_map(samples(1, 1, 1, &[0]), red_only, Some(space()), (1, 1));
    let mut rgb = multiply(&space().from_gain, [0.5, 0.25, 0.125]);
    map.apply(&mut rgb, 0);
    for (actual, expected) in rgb
        .into_iter()
        .zip(multiply(&space().from_gain, [1.0, 0.25, 0.125]))
    {
        assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
    }
}

#[test]
fn extreme_parameters_and_pixels_stay_finite() {
    let p3 = GainSpace::for_encoding(&rgb_encoding(JxlPrimaries::P3)).unwrap();

    // A 2^120 gain on bright float samples saturates at the f16 range.
    let huge = channel(120.0, 120.0, 1.0, (0.0, 0.0));
    let map = gain_map(samples(1, 1, 1, &[65_535]), [huge; 3], p3, (1, 1));
    let mut rgb = [32.0_f32; 3];
    map.apply(&mut rgb, 0);
    assert_eq!(rgb, [MAX_FINITE_F16; 3]);

    // The largest f32 multiplier on a zero sample is zero, not NaN.
    let wide = channel(-10.0, 127.999_99, 1.0, (0.0, 0.0));
    let map = gain_map(samples(1, 1, 1, &[65_535]), [wide; 3], None, (1, 1));
    let mut rgb = [0.0_f32; 3];
    map.apply(&mut rgb, 0);
    assert_eq!(rgb, [0.0; 3]);

    // Interpolating equal maximum samples with an extreme gamma and no gain
    // range must not leave [0, 1].
    let flat = channel(3.0, 3.0, 1.0 / f64::from(u32::MAX), (0.0, 0.0));
    let map = gain_map(samples(2, 2, 1, &[65_535; 4]), [flat; 3], None, (3, 3));
    let mut rgb = [1.0_f32; 27];
    map.apply(&mut rgb, 0);
    assert_eq!(rgb, [8.0; 27]);

    // Infinite pixels are bounded. Only NaN, which the base image conversion
    // rejects as well, stays NaN.
    let unit = channel(0.0, 0.0, 1.0, (0.0, 0.0));
    let map = gain_map(samples(1, 1, 1, &[0]), [unit; 3], None, (1, 1));
    let mut rgb = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    map.apply(&mut rgb, 0);
    assert_eq!(rgb[..2], [MAX_FINITE_F16, -MAX_FINITE_F16]);
    assert!(rgb[2].is_nan());
}

#[test]
fn decompression_is_bounded_and_reports_corruption() {
    // A stored Brotli stream: WBITS 16, then one uncompressed meta-block.
    let data = b"gain map bundle";
    let header = ((u32::try_from(data.len()).unwrap() - 1) << 4) | (1 << 20);
    let mut stream = header.to_le_bytes()[..3].to_vec();
    stream.extend_from_slice(data);
    stream.push(3);

    assert_eq!(
        decompress_brotli(stream.as_slice(), data.len()).unwrap(),
        data
    );
    assert_eq!(
        decompress_brotli(stream.as_slice(), data.len() - 1),
        Err(Reason::TooLarge)
    );
    assert_eq!(
        decompress_brotli(&b"not brotli"[..], LIMIT),
        Err(Reason::Malformed)
    );
    assert_eq!(
        decompress_brotli(&stream[..stream.len() - 4], LIMIT),
        Err(Reason::Malformed)
    );
}

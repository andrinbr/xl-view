//! ISO 21496-1 gain maps: HDR reconstruction, the untouched SDR base image,
//! box placement, and fallback to the base image.

use xl_view::decode::{
    DecodeError, DecodeLimits, DecodedImage, GainMapSkipReason, GainMapStatus, Rendition,
    SourceDynamicRange, decode_file, decode_memory, decode_memory_with_rendition,
};

const SIGNATURE: &[u8] = b"\0\0\0\x0cJXL \r\n\x87\n";
/// `base_offset` and `alternate_offset` of the fixture.
const OFFSET: f32 = 1.0 / 64.0;

fn fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

fn limits() -> DecodeLimits {
    DecodeLimits::from_memory_ceiling_mib(256)
}

fn boxed(box_type: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut bytes = u32::try_from(payload.len() + 8)
        .unwrap()
        .to_be_bytes()
        .to_vec();
    bytes.extend_from_slice(&box_type);
    bytes.extend_from_slice(payload);
    bytes
}

/// A box without a declared size, which extends to the end of the file.
fn open_box(box_type: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut bytes = 0_u32.to_be_bytes().to_vec();
    bytes.extend_from_slice(&box_type);
    bytes.extend_from_slice(payload);
    bytes
}

/// Splits a container into its top-level boxes, each with its header.
fn boxes(container: &[u8]) -> Vec<Vec<u8>> {
    let mut boxes = Vec::new();
    let mut rest = container;
    while !rest.is_empty() {
        let size = u32::from_be_bytes(rest[..4].try_into().unwrap()) as usize;
        let (head, tail) = rest.split_at(size);
        boxes.push(head.to_vec());
        rest = tail;
    }
    boxes
}

fn is_gain_map(candidate: &[u8]) -> bool {
    &candidate[4..8] == b"jhgm"
}

/// The gain map fixture split into its base image and its `jhgm` payload.
struct Fixture {
    container: Vec<u8>,
    base: Vec<u8>,
    bundle: Vec<u8>,
}

impl Fixture {
    fn load() -> Self {
        let container = fixture_bytes("test_pattern-gain-map.jxl");
        let boxes = boxes(&container);
        let bundle = boxes
            .iter()
            .find(|candidate| is_gain_map(candidate))
            .unwrap()[8..]
            .to_vec();
        let base = boxes
            .iter()
            .filter(|candidate| !is_gain_map(candidate))
            .flatten()
            .copied()
            .collect();
        Self {
            container,
            base,
            bundle,
        }
    }

    /// The base image followed by `trailing` boxes.
    fn with(&self, trailing: &[Vec<u8>]) -> Vec<u8> {
        let mut bytes = self.base.clone();
        bytes.extend(trailing.iter().flatten());
        bytes
    }

    fn metadata_len(&self) -> usize {
        usize::from(u16::from_be_bytes([self.bundle[1], self.bundle[2]]))
    }

    /// Offset of the ISO 21496-1 flags within the bundle.
    const FLAGS: usize = 3 + 4;
    /// Offset of the base headroom, followed by the alternate headroom.
    const HEADROOMS: usize = 3 + 5;

    /// Offset of the gain map codestream within the bundle.
    fn codestream_offset(&self) -> usize {
        3 + self.metadata_len() + 1 + 4
    }

    /// The bundle with another gain map codestream.
    fn bundle_with_codestream(&self, codestream: &[u8]) -> Vec<u8> {
        let mut bundle = self.bundle[..self.codestream_offset()].to_vec();
        bundle.extend_from_slice(codestream);
        bundle
    }
}

/// Appends a box to any fixture, wrapping a bare codestream in a container.
fn with_trailing_box(image: &[u8], trailing: &[u8]) -> Vec<u8> {
    let mut bytes = if image.starts_with(SIGNATURE) {
        image.to_vec()
    } else {
        let mut bytes = SIGNATURE.to_vec();
        bytes.extend(boxed(*b"ftyp", b"jxl \0\0\0\0jxl "));
        bytes.extend(boxed(*b"jxlc", image));
        bytes
    };
    bytes.extend_from_slice(trailing);
    bytes
}

/// A Brotli stream of stored, uncompressed meta-blocks.
fn stored_brotli(data: &[u8]) -> Vec<u8> {
    let mut stream = Vec::new();
    for (index, block) in data.chunks(1 << 16).enumerate() {
        // ISLAST = 0, MNIBBLES = 4, MLEN - 1, ISUNCOMPRESSED = 1. The first
        // block also carries the one-bit WBITS = 16 stream header.
        let length = u32::try_from(block.len() - 1).unwrap();
        let header = if index == 0 {
            (length << 4) | (1 << 20)
        } else {
            (length << 3) | (1 << 19)
        };
        stream.extend_from_slice(&header.to_le_bytes()[..3]);
        stream.extend_from_slice(block);
    }
    // ISLAST = 1, ISLASTEMPTY = 1.
    stream.push(3);
    stream
}

fn compressed_gain_map(bundle: &[u8]) -> Vec<u8> {
    let mut payload = b"jhgm".to_vec();
    payload.extend(stored_brotli(bundle));
    payload
}

fn pixels(image: &DecodedImage) -> &[f32] {
    assert_eq!(image.store.coarse_downsample(), 1);
    image.store.coarse_pixels()
}

/// Largest difference in stops, with a floor that keeps noise near black small.
fn max_stops_between(actual: &DecodedImage, expected: &DecodedImage) -> f32 {
    pixels(actual)
        .iter()
        .zip(pixels(expected))
        .map(|(actual, expected)| {
            ((actual.max(0.0) + 0.01) / (expected.max(0.0) + 0.01))
                .log2()
                .abs()
        })
        .fold(0.0, f32::max)
}

fn assert_falls_back(bytes: &[u8], base: &DecodedImage, reason: GainMapSkipReason, case: &str) {
    for rendition in [Rendition::Hdr, Rendition::Sdr] {
        let decoded = decode_memory_with_rendition(bytes, limits(), rendition)
            .unwrap_or_else(|error| panic!("{case}: {error}"));
        assert_eq!(
            decoded.gain_map,
            Some(GainMapStatus::NotApplied(reason)),
            "{case} as {rendition:?}"
        );
        assert_eq!(pixels(&decoded), pixels(base), "{case} as {rendition:?}");
        assert_eq!(decoded.hdr_source, base.hdr_source, "{case}");
        assert_eq!(decoded.source_peak, base.source_peak, "{case}");
    }
}

#[test]
fn hdr_rendition_reconstructs_the_hlg_pattern() {
    let fixture = Fixture::load();
    let hdr = decode_memory(&fixture.container, limits()).unwrap();
    assert_eq!(
        hdr.gain_map,
        Some(GainMapStatus::Applied {
            headroom_stops: 2.3
        })
    );
    // The base image's encoding is reported, with the HDR claim of the pixels.
    assert_eq!(hdr.source_dynamic_range, SourceDynamicRange::Sdr);
    assert!(hdr.hdr_source);
    let peak = hdr.source_peak.nits();
    assert!((peak - 1_000.0).abs() < 10.0, "measured peak {peak}");

    // The 8-bit gain map spans four stops, so one code is 0.016 stops.
    let reference = decode_memory(&fixture_bytes("test_pattern-HLG.jxl"), limits()).unwrap();
    let error = max_stops_between(&hdr, &reference);
    assert!(error < 0.03, "{error} stops from the HLG pattern");
}

#[test]
fn sdr_rendition_is_the_untouched_base_image() {
    let fixture = Fixture::load();
    let base = decode_memory(&fixture.base, limits()).unwrap();
    assert_eq!(base.gain_map, None);
    assert!(!base.hdr_source);

    let sdr = decode_memory_with_rendition(&fixture.container, limits(), Rendition::Sdr).unwrap();
    assert_eq!(sdr.gain_map, Some(GainMapStatus::Available));
    assert_eq!(pixels(&sdr), pixels(&base));
    assert!(!sdr.hdr_source);
    assert_eq!(sdr.source_peak, base.source_peak);
    assert_eq!(sdr.memory_cost_bytes, base.memory_cost_bytes);
}

#[test]
fn file_decoding_defaults_to_the_hdr_rendition() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/test_pattern-gain-map.jxl");
    let decoded = decode_file(&path, limits()).unwrap();
    assert!(matches!(
        decoded.gain_map,
        Some(GainMapStatus::Applied { .. })
    ));
}

#[test]
fn gain_map_boxes_apply_wherever_they_are_stored() {
    let fixture = Fixture::load();
    let expected = decode_memory(&fixture.container, limits()).unwrap();
    let padding = boxed(*b"junk", &[0; 300]);
    for (case, bytes) in [
        (
            "after the codestream",
            fixture.with(&[boxed(*b"jhgm", &fixture.bundle), padding.clone()]),
        ),
        (
            "open at the end of the file",
            fixture.with(&[padding.clone(), open_box(*b"jhgm", &fixture.bundle)]),
        ),
        (
            "compressed",
            fixture.with(&[boxed(*b"brob", &compressed_gain_map(&fixture.bundle))]),
        ),
        (
            "compressed and open at the end of the file",
            fixture.with(&[open_box(*b"brob", &compressed_gain_map(&fixture.bundle))]),
        ),
    ] {
        let decoded =
            decode_memory(&bytes, limits()).unwrap_or_else(|error| panic!("{case}: {error}"));
        assert_eq!(decoded.gain_map, expected.gain_map, "{case}");
        assert_eq!(pixels(&decoded), pixels(&expected), "{case}");

        // Files are read through a different input than memory.
        let path = std::env::temp_dir().join(format!(
            "xl-view-gain-map-{}-{}.jxl",
            std::process::id(),
            case.replace(' ', "-")
        ));
        std::fs::write(&path, &bytes).unwrap();
        let decoded = decode_file(&path, limits());
        std::fs::remove_file(&path).unwrap();
        let decoded = decoded.unwrap_or_else(|error| panic!("{case} from a file: {error}"));
        assert_eq!(pixels(&decoded), pixels(&expected), "{case} from a file");
    }
}

#[test]
fn unusable_gain_map_metadata_falls_back_to_the_base_image() {
    let fixture = Fixture::load();
    let base = decode_memory(&fixture.base, limits()).unwrap();
    let edited = |edit: &dyn Fn(&mut Vec<u8>)| {
        let mut bundle = fixture.bundle.clone();
        edit(&mut bundle);
        fixture.with(&[boxed(*b"jhgm", &bundle)])
    };

    for (case, bytes, reason) in [
        (
            "reserved metadata flag",
            edited(&|bundle| bundle[Fixture::FLAGS] |= 0x08),
            GainMapSkipReason::Unsupported,
        ),
        (
            "newer bundle version",
            edited(&|bundle| bundle[0] = 1),
            GainMapSkipReason::Unsupported,
        ),
        (
            "HDR base image",
            edited(&|bundle| {
                let (base, alternate) =
                    bundle[Fixture::HEADROOMS..Fixture::HEADROOMS + 16].split_at_mut(8);
                base.swap_with_slice(alternate);
            }),
            GainMapSkipReason::HdrBase,
        ),
        (
            "equal headrooms",
            edited(&|bundle| {
                bundle.copy_within(
                    Fixture::HEADROOMS + 8..Fixture::HEADROOMS + 16,
                    Fixture::HEADROOMS,
                );
            }),
            GainMapSkipReason::NoHeadroom,
        ),
        (
            "zero denominator",
            edited(&|bundle| bundle[Fixture::HEADROOMS + 4..Fixture::HEADROOMS + 8].fill(0)),
            GainMapSkipReason::Malformed,
        ),
        (
            "truncated bundle",
            edited(&|bundle| bundle.truncate(100)),
            GainMapSkipReason::Malformed,
        ),
        (
            "alternate ICC profile",
            edited(&|bundle| {
                // Gains in the alternate color space, described by an ICC profile.
                bundle[Fixture::FLAGS] &= !0x40;
                let icc_length = fixture.codestream_offset() - 4;
                bundle[icc_length..icc_length + 4].copy_from_slice(&3_u32.to_be_bytes());
                bundle.splice(icc_length + 4..icc_length + 4, [1, 2, 3]);
            }),
            GainMapSkipReason::UnsupportedProfile,
        ),
    ] {
        assert_falls_back(&bytes, &base, reason, case);
    }
}

#[test]
fn unusable_gain_map_images_and_boxes_fall_back_to_the_base_image() {
    let fixture = Fixture::load();
    let base = decode_memory(&fixture.base, limits()).unwrap();
    let with_codestream = |codestream: &[u8]| {
        fixture.with(&[boxed(*b"jhgm", &fixture.bundle_with_codestream(codestream))])
    };

    for (case, bytes, reason) in [
        (
            "gain map image that is not JPEG XL",
            with_codestream(b"not a codestream"),
            GainMapSkipReason::Malformed,
        ),
        (
            "rotated gain map image",
            with_codestream(&fixture_bytes("oriented.jxl")),
            GainMapSkipReason::Unsupported,
        ),
        (
            "gain map image with alpha",
            with_codestream(&fixture_bytes("alpha.jxl")),
            GainMapSkipReason::Unsupported,
        ),
        (
            "animated gain map image",
            with_codestream(&fixture_bytes("animation.jxl")),
            GainMapSkipReason::Unsupported,
        ),
        (
            "gain map image decoded to another profile",
            with_codestream(&fixture_bytes("icc-vardct.jxl")),
            GainMapSkipReason::UnsupportedProfile,
        ),
        (
            "two gain maps",
            fixture.with(&[
                boxed(*b"jhgm", &fixture.bundle),
                boxed(*b"jhgm", &fixture.bundle),
            ]),
            GainMapSkipReason::Unsupported,
        ),
        (
            "corrupt compression",
            fixture.with(&[boxed(*b"brob", b"jhgmnot a brotli stream")]),
            GainMapSkipReason::Malformed,
        ),
    ] {
        assert_falls_back(&bytes, &base, reason, case);
    }
}

#[test]
fn a_cut_off_gain_map_image_is_found_when_it_is_decoded() {
    let fixture = Fixture::load();
    let base = decode_memory(&fixture.base, limits()).unwrap();
    let codestream = &fixture.bundle[fixture.codestream_offset()..];
    let bundle = fixture.bundle_with_codestream(&codestream[..codestream.len() / 2]);
    let bytes = fixture.with(&[boxed(*b"jhgm", &bundle)]);

    // The SDR rendition reads only the gain map's header, which is intact.
    let sdr = decode_memory_with_rendition(&bytes, limits(), Rendition::Sdr).unwrap();
    assert_eq!(sdr.gain_map, Some(GainMapStatus::Available));
    assert_eq!(pixels(&sdr), pixels(&base));

    // The HDR rendition falls back, and its status asks for no further decode.
    let hdr = decode_memory(&bytes, limits()).unwrap();
    let status = GainMapStatus::NotApplied(GainMapSkipReason::Malformed);
    assert_eq!(hdr.gain_map, Some(status));
    assert_eq!(pixels(&hdr), pixels(&base));
    assert!(!status.differs_for(Rendition::Hdr) && !status.differs_for(Rendition::Sdr));
}

#[test]
fn broken_container_framing_still_fails_the_decode() {
    let fixture = Fixture::load();
    // A gain map box that ends before its declared size is a truncated file.
    let mut truncated = fixture.with(&[boxed(*b"jhgm", &fixture.bundle)]);
    truncated.truncate(truncated.len() - 10);
    assert!(matches!(
        decode_memory(&truncated, limits()),
        Err(DecodeError::Decoder(_))
    ));
}

#[test]
fn unsupported_base_images_keep_their_own_pixels() {
    let gain_map = boxed(*b"jhgm", &Fixture::load().bundle);
    for (name, reason) in [
        ("test_pattern-PQ.jxl", GainMapSkipReason::HdrBase),
        ("test_pattern-HLG.jxl", GainMapSkipReason::HdrBase),
        ("linear-bt2020.jxl", GainMapSkipReason::Unsupported),
        ("oriented.jxl", GainMapSkipReason::Unsupported),
        ("grayscale.jxl", GainMapSkipReason::UnsupportedProfile),
        ("icc.jxl", GainMapSkipReason::UnsupportedProfile),
        ("icc-vardct.jxl", GainMapSkipReason::UnsupportedProfile),
    ] {
        let image = fixture_bytes(name);
        let base = decode_memory(&image, limits()).unwrap();
        assert_falls_back(&with_trailing_box(&image, &gain_map), &base, reason, name);
    }
}

#[test]
fn animations_keep_their_first_frame() {
    let gain_map = boxed(*b"jhgm", &Fixture::load().bundle);
    let animation = fixture_bytes("animation-multi-frame.jxl");
    let base = decode_memory(&animation, limits()).unwrap();

    let mut early = SIGNATURE.to_vec();
    early.extend(boxed(*b"ftyp", b"jxl \0\0\0\0jxl "));
    early.extend_from_slice(&gain_map);
    early.extend(boxed(*b"jxlc", &animation));
    assert_falls_back(&early, &base, GainMapSkipReason::Unsupported, "early box");

    // Boxes after the first frame are not read, so nothing is known about them.
    let late = decode_memory(&with_trailing_box(&animation, &gain_map), limits()).unwrap();
    assert_eq!(late.gain_map, None);
    assert_eq!(pixels(&late), pixels(&base));
}

const SRGB_TO_BT2020: [[f32; 3]; 3] = [
    [0.627_404, 0.329_283, 0.043_313],
    [0.069_097, 0.919_540, 0.011_362],
    [0.016_391, 0.088_013, 0.895_595],
];
const BT2020_TO_SRGB: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641, -0.072_850],
    [-0.124_550, 1.132_9, -0.008_349],
    [-0.018_151, -0.100_579, 1.118_73],
];

fn multiply(matrix: &[[f32; 3]; 3], vector: [f32; 3]) -> [f32; 3] {
    matrix.map(|row| row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2])
}

/// Checks `actual` against gains applied in linear sRGB to `base`.
///
/// The fixture applies its gains in BT.2020, so its two renditions give the
/// per-channel gain of every pixel without decoding the gain map here.
fn assert_gains_applied_in_srgb(actual: &DecodedImage, base: &DecodedImage, case: &str) {
    let fixture = Fixture::load();
    let hdr = decode_memory(&fixture.container, limits()).unwrap();
    let sdr = decode_memory_with_rendition(&fixture.container, limits(), Rendition::Sdr).unwrap();
    let mut checked = 0;
    let mut differs_from_bt2020_gains = false;
    for (((actual, base), hdr), sdr) in pixels(actual)
        .chunks_exact(4)
        .zip(pixels(base).chunks_exact(4))
        .zip(pixels(&hdr).chunks_exact(4))
        .zip(pixels(&sdr).chunks_exact(4))
        .step_by(37)
    {
        let gain: [f32; 3] =
            std::array::from_fn(|channel| (hdr[channel] + OFFSET) / (sdr[channel] + OFFSET));
        let in_srgb = multiply(&BT2020_TO_SRGB, [base[0], base[1], base[2]]);
        let expected = multiply(
            &SRGB_TO_BT2020,
            std::array::from_fn(|channel| (in_srgb[channel] + OFFSET) * gain[channel] - OFFSET),
        );
        for channel in 0..3 {
            let tolerance = expected[channel].abs() * 0.004 + 0.004;
            assert!(
                (actual[channel] - expected[channel]).abs() <= tolerance,
                "{case}: {} vs {}",
                actual[channel],
                expected[channel]
            );
            let in_bt2020 = (base[channel] + OFFSET) * gain[channel] - OFFSET;
            differs_from_bt2020_gains |= (actual[channel] - in_bt2020).abs() > 0.05;
        }
        checked += 1;
    }
    assert!(checked > 10_000);
    // Per-channel gains do not commute with the color matrix.
    assert!(differs_from_bt2020_gains, "{case}");
}

#[test]
fn gains_apply_in_the_base_images_color_space() {
    let fixture = Fixture::load();
    let image = fixture_bytes("test_pattern-sRGB.jxl");
    let base = decode_memory(&image, limits()).unwrap();
    let decoded = decode_memory(
        &with_trailing_box(&image, &boxed(*b"jhgm", &fixture.bundle)),
        limits(),
    )
    .unwrap();
    assert!(matches!(
        decoded.gain_map,
        Some(GainMapStatus::Applied { .. })
    ));
    assert!(decoded.hdr_source);
    assert_gains_applied_in_srgb(&decoded, &base, "sRGB base image");
}

#[test]
fn gains_apply_in_the_alternate_color_space_when_requested() {
    let fixture = Fixture::load();
    let expected = decode_memory(&fixture.container, limits()).unwrap();
    let base = decode_memory(&fixture.base, limits()).unwrap();
    let color_encoding_length = fixture.codestream_offset() - 5;

    // Without its own profile, the alternate image shares the base image's.
    let mut inherited = fixture.bundle.clone();
    inherited[Fixture::FLAGS] &= !0x40;
    let decoded = decode_memory(&fixture.with(&[boxed(*b"jhgm", &inherited)]), limits()).unwrap();
    assert_eq!(pixels(&decoded), pixels(&expected));

    // An all-default color encoding makes the alternate image sRGB.
    let mut srgb = inherited.clone();
    srgb[color_encoding_length] = 1;
    srgb.insert(color_encoding_length + 1, 1);
    let decoded = decode_memory(&fixture.with(&[boxed(*b"jhgm", &srgb)]), limits()).unwrap();
    assert!(matches!(
        decoded.gain_map,
        Some(GainMapStatus::Applied { .. })
    ));
    assert_gains_applied_in_srgb(&decoded, &base, "sRGB alternate image");

    // The alternate profile is ignored when gains use the base color space.
    let mut ignored = fixture.bundle.clone();
    ignored[color_encoding_length] = 1;
    ignored.insert(color_encoding_length + 1, 1);
    let decoded = decode_memory(&fixture.with(&[boxed(*b"jhgm", &ignored)]), limits()).unwrap();
    assert_eq!(pixels(&decoded), pixels(&expected));
}

#[test]
fn smaller_and_single_channel_gain_map_images_are_resampled() {
    let fixture = Fixture::load();
    let base = decode_memory(&fixture.base, limits()).unwrap();
    for name in ["grayscale.jxl", "linear-bt2020.jxl", "icc.jxl"] {
        let bundle = fixture.bundle_with_codestream(&fixture_bytes(name));
        let decoded = decode_memory(&fixture.with(&[boxed(*b"jhgm", &bundle)]), limits()).unwrap();
        assert!(
            matches!(decoded.gain_map, Some(GainMapStatus::Applied { .. })),
            "{name}: {:?}",
            decoded.gain_map
        );
        // Gains span -1.5 to +2.5 stops around the base image.
        for (actual, base) in pixels(&decoded).iter().zip(pixels(&base)) {
            let low = (base + OFFSET) * 0.353 - OFFSET - 1e-3;
            let high = (base + OFFSET) * 5.66 - OFFSET + 1e-3;
            assert!(
                (low..=high).contains(actual),
                "{name}: {actual} from {base}"
            );
        }
    }
}

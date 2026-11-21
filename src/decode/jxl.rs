use std::borrow::Cow;
use std::fs::{self, File, Metadata};
use std::io::{BufReader, IoSliceMut, Read};
use std::ops::RangeInclusive;
use std::path::Path;
use std::sync::Arc;

use jxl::api::{
    Event, ExtraChannel, JxlAuxBox, JxlAuxBoxType, JxlBasicInfo, JxlBitDepth, JxlBitstreamInput,
    JxlColorEncoding, JxlColorProfile, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    JxlParallelRunner, JxlParallelRunnerFun, JxlPixelFormat, JxlPrimaries, JxlTransferFunction,
    JxlWhitePoint, Orientation, RenderingIntent,
};
use moxcms::{ColorProfile, DataColorSpace, Layout, TransformF32Executor, TransformOptions};
use rayon::prelude::*;

use crate::color::{
    BT2020_LUMA_F32, HDR_REFERENCE_WHITE_NITS, PQ_PEAK_NITS, SourceDynamicRange,
    SourceIntensityTarget, hlg_inverse_oetf, hlg_system_gamma, pq_eotf,
};
use crate::metadata::{ExifMetadata, MAX_METADATA_BYTES, XmpMetadata, parse_exif, parse_xmp};
use crate::units::{mib_to_bytes, mib_to_usize_bytes, u64_from_usize, usize_from_u32};

use super::gain_map::{self, BaseImage, CaptureInput, GainMap};
use super::{
    CANONICAL_BYTES_PER_PIXEL, CanonicalPixel, CoarsePixel, DecodeError, DecodeLimits,
    DecodeResult, DecodedImage, DecodedTileStore, HISTOGRAM_BINS, HISTOGRAM_BINS_PER_STOP,
    HISTOGRAM_REFERENCE_WHITE_BIN, Histogram, ImageKey, ImageMetadata, LinearUnit, Rendition,
    SourceColorEncoding, ToneMappingMetadata,
};

const CHUNK_PIXELS: usize = 65_536;
const MAX_SCRATCH_BYTES: usize = mib_to_usize_bytes(32);
const MAX_COARSE_BYTES: u64 = mib_to_bytes(64);
const MAX_COARSE_DIMENSION: u32 = 4_096;
const MAX_TRAILING_METADATA_BYTES: usize = mib_to_usize_bytes(32);
const XML: JxlAuxBoxType = JxlAuxBoxType(*b"xml ");
// An Exif box starts with the big-endian offset of its TIFF header.
const EXIF_OFFSET_BYTES: usize = 4;
// jxl clamps converted HLG output to this range before f16 storage.
const HLG_SIGNAL_MIN: f32 = half::f16::from_f32_const(-0.074).to_f32_const();
const HLG_SIGNAL_MAX: f32 = half::f16::from_f32_const(1.1).to_f32_const();
// Conversion input and output: two RGB buffers.
const CONVERSION_FLOATS_PER_PIXEL: usize = 6;
// The conversion buffers plus a row of f64 preview accumulators per worker.
const WORKER_SCRATCH_BYTES: usize = CHUNK_PIXELS * CONVERSION_FLOATS_PER_PIXEL * size_of::<f32>()
    + MAX_COARSE_DIMENSION as usize * size_of::<[f64; 4]>();
const MAX_WORKERS: usize = MAX_SCRATCH_BYTES / WORKER_SCRATCH_BYTES;

/// Decodes a filesystem source within the supplied limits. A gain map
/// reconstructs the HDR rendition.
///
/// # Errors
/// Returns a typed error for I/O, malformed input, unsupported content, or a
/// configured resource limit.
pub fn decode_file(path: &Path, limits: DecodeLimits) -> DecodeResult {
    let file = open_file(path)?;
    decode_input(&mut FileInput(BufReader::new(file)), limits, None, &|| {
        Rendition::Hdr
    })
}

/// `rendition` is called once, when the decoded base image is ready for its
/// gain map, so a decode started before the display is known can still follow it.
pub(super) fn decode_file_for_key(
    key: &ImageKey,
    limits: DecodeLimits,
    maximum_retained_bytes: Option<u64>,
    rendition: &dyn Fn() -> Rendition,
) -> DecodeResult {
    let file = open_file(&key.normalized_path)?;
    let metadata_error = |source| DecodeError::SourceMetadata {
        path: key.normalized_path.clone(),
        source,
    };
    verify_source_identity(key, &file.metadata().map_err(metadata_error)?)?;
    let mut input = FileInput(BufReader::new(file));
    let decoded = decode_input(&mut input, limits, maximum_retained_bytes, rendition);
    let after = input.0.get_ref().metadata().map_err(metadata_error)?;
    let path_after = fs::metadata(&key.normalized_path).map_err(metadata_error)?;
    verify_source_identity(key, &after)?;
    verify_source_identity(key, &path_after)?;
    decoded
}

fn open_file(path: &Path) -> Result<File, DecodeError> {
    File::open(path).map_err(|source| DecodeError::Read {
        path: path.to_owned(),
        source,
    })
}

fn verify_source_identity(key: &ImageKey, metadata: &Metadata) -> Result<(), DecodeError> {
    let modified = metadata
        .modified()
        .map_err(|source| DecodeError::SourceMetadata {
            path: key.normalized_path.clone(),
            source,
        })?;
    if metadata.len() == key.source_len && modified == key.source_modified {
        Ok(())
    } else {
        Err(DecodeError::SourceChanged {
            path: key.normalized_path.clone(),
        })
    }
}

// Use the default read-based skip: seeking past EOF would hide truncated boxes.
struct FileInput(BufReader<File>);
impl JxlBitstreamInput for FileInput {
    fn available_bytes(&mut self) -> std::io::Result<usize> {
        self.0.available_bytes()
    }
    fn read(&mut self, buffers: &mut [IoSliceMut<'_>]) -> std::io::Result<usize> {
        JxlBitstreamInput::read(&mut self.0, buffers)
    }
}

/// Decodes an in-memory JPEG XL source within the supplied limits. A gain map
/// reconstructs the HDR rendition.
///
/// # Errors
/// Returns a typed error for malformed input, unsupported content, or a
/// configured resource limit.
pub fn decode_memory(bytes: &[u8], limits: DecodeLimits) -> DecodeResult {
    decode_memory_with_rendition(bytes, limits, Rendition::Hdr)
}

/// Decodes an in-memory JPEG XL source as the requested rendition.
///
/// # Errors
/// Returns a typed error for malformed input, unsupported content, or a
/// configured resource limit.
pub fn decode_memory_with_rendition(
    mut bytes: &[u8],
    limits: DecodeLimits,
    rendition: Rendition,
) -> DecodeResult {
    decode_input(&mut bytes, limits, None, &|| rendition)
}

pub(super) struct RayonRunner;
impl JxlParallelRunner for RayonRunner {
    fn run(&mut self, count: usize, fun: &JxlParallelRunnerFun<'_>) -> jxl::error::Result<()> {
        (0..count).into_par_iter().try_for_each(fun)
    }
    fn num_threads(&self) -> usize {
        rayon::current_num_threads()
    }
}

// Inputs are complete files, so a request for more input means truncation.
fn advance(
    decoder: &mut JxlDecoder,
    input: &mut impl JxlBitstreamInput,
    buffers: Option<&mut [JxlOutputBuffer<'_>]>,
    limits: DecodeLimits,
) -> Result<Event, DecodeError> {
    match decoder
        .process(input, buffers, Some(&mut RayonRunner))
        .map_err(|error| decoder_error(error, limits))?
    {
        Event::NeedMoreInput { .. } => Err(DecodeError::Decoder(Box::new(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "incomplete JPEG XL input",
        )))),
        event => Ok(event),
    }
}

fn expect_event(event: Event, expected: Event) -> Result<(), DecodeError> {
    if event == expected {
        Ok(())
    } else {
        Err(unexpected_event(event))
    }
}

fn unexpected_event(event: Event) -> DecodeError {
    DecodeError::Decoder(Box::new(std::io::Error::other(format!(
        "unexpected JPEG XL decoder event {event:?}"
    ))))
}

fn decoder_error(error: jxl::error::Error, limits: DecodeLimits) -> DecodeError {
    match error {
        jxl::error::Error::ImageSizeTooLarge(..)
        | jxl::error::Error::OutOfMemory(..)
        | jxl::error::Error::ImageOutOfMemory(..) => DecodeError::DecoderMemoryLimit {
            limit: limits.memory_ceiling_bytes,
        },
        _ => DecodeError::Decoder(Box::new(error)),
    }
}

fn decode_input(
    input: &mut impl JxlBitstreamInput,
    limits: DecodeLimits,
    maximum_retained_bytes: Option<u64>,
    rendition: &dyn Fn() -> Rendition,
) -> DecodeResult {
    // The gain map box is captured from the input instead of being requested
    // here: `jxl` would buffer it without a limit.
    let input = &mut CaptureInput::new(input, limits.memory_ceiling_bytes);
    let mut options = JxlDecoderOptions::default();
    options.sample_limit = Some(limits.memory_ceiling_bytes / size_of::<f32>());
    options.request_aux_boxes = vec![JxlAuxBoxType::EXIF, XML];
    let mut decoder = JxlDecoder::new(options);
    expect_event(
        advance(&mut decoder, input, None, limits)?,
        Event::BasicInfo,
    )?;
    let info = decoder.basic_info().expect("basic info follows BasicInfo");
    let embedded_profile = decoder
        .embedded_color_profile()
        .expect("color profile follows BasicInfo");
    // basic_info().size is already oriented when adjust_orientation is enabled.
    let width = u32::try_from(info.size.0).map_err(|_| DecodeError::SizeOverflow)?;
    let height = u32::try_from(info.size.1).map_err(|_| DecodeError::SizeOverflow)?;
    validate_dimensions(width, height)?;
    for channel in &info.extra_channels {
        if channel.ec_type != ExtraChannel::Alpha {
            return Err(DecodeError::UnsupportedExtraChannel(format!(
                "{:?}",
                channel.ec_type
            )));
        }
    }
    let opaque = info.extra_channels.is_empty();
    let identity_orientation = info.orientation == Orientation::Identity;
    let associated_alpha = info
        .extra_channels
        .first()
        .is_some_and(|channel| channel.alpha_associated);
    let mut metadata = source_metadata(info, embedded_profile);
    let source = SourceCalibration::from_header(info, embedded_profile)?;
    if associated_alpha
        && (source.xyb
            || embedded_profile.channels() != 3
            || matches!(
                source.dynamic_range,
                SourceDynamicRange::Pq | SourceDynamicRange::Hlg
            ))
    {
        return Err(DecodeError::UnsupportedAssociatedAlphaEncoding);
    }
    let image_bytes = validate_image_memory(width, height, limits, maximum_retained_bytes)?;
    let gain_map_budget = gain_map_budget(limits, image_bytes);
    input.set_budget(gain_map_budget);
    let extra_count = info.extra_channels.len();
    decoder
        .set_pixel_format(JxlPixelFormat::rgba_f16(extra_count))
        .map_err(|error| decoder_error(error, limits))?;
    let output_profile = decoder
        .output_color_profile()
        .expect("output profile follows BasicInfo");
    let mut conversion = ColorConversion::new(
        output_profile,
        associated_alpha,
        source.dynamic_range,
        source.intensity,
    )?;
    // Integer samples span [0, 1] in their own encoding. Outside it, lossy
    // reconstruction overshoot would reach SDR gamut mapping as fake color.
    // XYB sources with an ICC profile decode to linear sRGB instead, where
    // negative values can be real wide-gamut color.
    conversion.clamp_to_nominal_range =
        !source.float_samples && Some(output_profile) == decoder.embedded_color_profile();
    let mut base = BaseImage {
        width,
        height,
        encoding: gain_map::base_encoding(output_profile, decoder.embedded_color_profile()),
        dynamic_range: source.dynamic_range,
        identity_orientation,
        animated: false,
    };
    let (pixels, has_more_frames) = decode_frame(&mut decoder, input, width, height, limits)?;
    // Trailing boxes come first: a gain map may follow the codestream, and
    // conversion overwrites the pixels it applies to. This also releases the
    // main decoder before the gain map is decoded.
    finish_metadata(decoder, input, has_more_frames, &mut metadata, limits)?;
    base.animated = has_more_frames;
    // Boxes after the first frame of an animation are not read.
    let capture = input
        .finish(!has_more_frames)
        .map_err(|error| DecodeError::Decoder(Box::new(error)))?;
    // Everything that can fail for the gain map happens here, so a failure
    // leaves the base image exactly as a decode without the gain map.
    let (gain_map, gain_map_status) =
        gain_map::prepare(capture, &base, gain_map_budget, limits, rendition);
    let calibrated = calibrate_and_convert(
        pixels,
        width,
        height,
        opaque,
        conversion,
        source,
        gain_map.as_ref(),
    )?;
    let memory_cost_bytes = calibrated.store.memory_cost_bytes();
    Ok(Arc::new(DecodedImage {
        store: Arc::new(calibrated.store),
        width,
        height,
        source_dynamic_range: source.dynamic_range,
        source_peak: calibrated.source_peak,
        histogram: calibrated.histogram,
        intensity_target: source.intensity,
        linear_unit: calibrated.linear_unit,
        hdr_source: calibrated.hdr_source,
        gain_map: gain_map_status,
        metadata,
        memory_cost_bytes,
    }))
}

/// Checks that the decoded image and its conversion scratch fit the memory
/// ceiling, and that a prefetched image fits `maximum_retained_bytes`.
/// Returns the bytes the image and its scratch need together.
fn validate_image_memory(
    width: u32,
    height: u32,
    limits: DecodeLimits,
    maximum_retained_bytes: Option<u64>,
) -> Result<u64, DecodeError> {
    let required = estimated_memory_cost_bytes(width, height)?;
    let required_with_scratch = required
        .checked_add(estimated_conversion_scratch_bytes(width, height)?)
        .ok_or(DecodeError::SizeOverflow)?;
    validate_output_memory(required_with_scratch, limits)?;
    if let Some(available) = maximum_retained_bytes
        && required > available
    {
        return Err(DecodeError::PrefetchTooLarge {
            required,
            available,
        });
    }
    Ok(required_with_scratch)
}

/// What the image and its scratch leave of the memory ceiling for a gain map.
fn gain_map_budget(limits: DecodeLimits, image_bytes: u64) -> usize {
    usize::try_from(u64_from_usize(limits.memory_ceiling_bytes).saturating_sub(image_bytes))
        .unwrap_or(usize::MAX)
}

/// Decodes the first displayable frame into a new canonical buffer. Also
/// returns whether more frames follow.
fn decode_frame(
    decoder: &mut JxlDecoder,
    input: &mut impl JxlBitstreamInput,
    width: u32,
    height: u32,
    limits: DecodeLimits,
) -> Result<(Vec<CanonicalPixel>, bool), DecodeError> {
    let mut pixels = allocate_vec::<CanonicalPixel>(checked_pixel_count(width, height)?, true)?;
    // Coalescing processes hidden/reference frames internally before this frame.
    expect_event(advance(decoder, input, None, limits)?, Event::FrameHeader)?;
    let mut buffers = [JxlOutputBuffer::new(
        bytemuck::cast_slice_mut(&mut pixels),
        usize_from_u32(height),
        usize_from_u32(width) * CANONICAL_BYTES_PER_PIXEL,
    )];
    match advance(decoder, input, Some(&mut buffers), limits)? {
        Event::FrameComplete { has_more_frames } => Ok((pixels, has_more_frames)),
        event => Err(unexpected_event(event)),
    }
}

fn finish_metadata(
    mut decoder: JxlDecoder,
    input: &mut impl JxlBitstreamInput,
    has_more_frames: bool,
    metadata: &mut ImageMetadata,
    limits: DecodeLimits,
) -> Result<(), DecodeError> {
    if has_more_frames {
        extract_aux_boxes(&decoder, metadata);
        return Ok(());
    }
    // Always process, even at external EOF: decoder buffers may hold metadata.
    // The input holds the whole file, and closing it lets EOF end the boxes.
    decoder.close_input();
    expect_event(advance(&mut decoder, input, None, limits)?, Event::Complete)?;
    extract_aux_boxes(&decoder, metadata);
    if let Some(aux) = decoder.trailing_box() {
        let limit = limits.memory_ceiling_bytes.min(MAX_TRAILING_METADATA_BYTES);
        let too_large = move |_| Err(format!("the box is larger than the {limit}-byte limit"));
        let mut required = aux.raw_data().len();
        if required > limit {
            store_aux_box(metadata, aux.box_type(), too_large);
            return Ok(());
        }
        let mut remaining = Vec::new();
        let mut buffer = [0_u8; 8192];
        loop {
            // Include buffered bytes and probe only one byte past the cap.
            let read_size = buffer.len().min(limit - required + 1);
            let count = match input.read(&mut [IoSliceMut::new(&mut buffer[..read_size])]) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => result.map_err(|error| DecodeError::Decoder(Box::new(error)))?,
            };
            if count == 0 {
                break;
            }
            required += count;
            if required > limit {
                store_aux_box(metadata, aux.box_type(), too_large);
                return Ok(());
            }
            remaining
                .try_reserve(count)
                .map_err(|source| DecodeError::ScratchAllocation {
                    required: remaining.len() + count,
                    source,
                })?;
            remaining.extend_from_slice(&buffer[..count]);
        }
        store_aux_box(metadata, aux.box_type(), |limit| {
            read_aux_box(aux, &remaining, limit)
        });
    }
    Ok(())
}

fn allocate_vec<T: Default>(count: usize, canonical: bool) -> Result<Vec<T>, DecodeError> {
    let required = count
        .checked_mul(size_of::<T>())
        .ok_or(DecodeError::SizeOverflow)?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(count).map_err(|source| {
        if canonical {
            DecodeError::OutputAllocation { required, source }
        } else {
            DecodeError::ScratchAllocation { required, source }
        }
    })?;
    buffer.resize_with(count, T::default);
    Ok(buffer)
}

fn profile_error(error: impl std::fmt::Display) -> DecodeError {
    DecodeError::UnsupportedColorProfile(error.to_string())
}

pub(super) fn working_profile() -> JxlColorProfile {
    JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
        white_point: JxlWhitePoint::D65,
        primaries: JxlPrimaries::BT2100,
        transfer_function: JxlTransferFunction::Linear,
        rendering_intent: RenderingIntent::Relative,
    })
}

fn dynamic_range(profile: &JxlColorProfile) -> Result<SourceDynamicRange, DecodeError> {
    if let JxlColorProfile::Icc(bytes) = profile {
        let icc = ColorProfile::new_from_slice(bytes).map_err(profile_error)?;
        if icc.cicp.is_some_and(|cicp| {
            matches!(
                cicp.transfer_characteristics,
                moxcms::TransferCharacteristics::Smpte2084 | moxcms::TransferCharacteristics::Hlg
            )
        }) {
            // Embedded PQ/HLG ICC profiles may contain SDR tone mapping and
            // placeholder reverse tables. Reject HDR transfers declared by CICP.
            return Err(profile_error(
                "embedded PQ/HLG ICC profiles are not supported",
            ));
        }
    }
    Ok(match profile.transfer_function() {
        Some(JxlTransferFunction::PQ) => SourceDynamicRange::Pq,
        Some(JxlTransferFunction::HLG) => SourceDynamicRange::Hlg,
        Some(JxlTransferFunction::Linear) => SourceDynamicRange::Linear,
        _ => SourceDynamicRange::Sdr,
    })
}

pub(super) fn profile_intent(source_icc: &[u8]) -> moxcms::RenderingIntent {
    let value = source_icc.get(64..68).map_or(1, |bytes| {
        u32::from_be_bytes(bytes.try_into().expect("four-byte intent"))
    });
    match value {
        0 => moxcms::RenderingIntent::Perceptual,
        2 => moxcms::RenderingIntent::Saturation,
        3 => moxcms::RenderingIntent::AbsoluteColorimetric,
        _ => moxcms::RenderingIntent::RelativeColorimetric,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum InverseTransfer {
    None,
    Pq,
    Hlg,
}

// The decoder outputs the embedded encoding, or a linear fallback for XYB.
// Any other pairing is inconsistent and rejected rather than guessed.
fn output_processing(
    source_dynamic_range: SourceDynamicRange,
    transfer: Option<&JxlTransferFunction>,
    intensity: f32,
) -> Result<(InverseTransfer, f32), DecodeError> {
    Ok(match (source_dynamic_range, transfer) {
        (SourceDynamicRange::Sdr, _)
            if !matches!(
                transfer,
                Some(JxlTransferFunction::PQ | JxlTransferFunction::HLG)
            ) =>
        {
            (InverseTransfer::None, 1.0)
        }
        // Calibration sets the scale once the decoded samples are known.
        (SourceDynamicRange::Linear, Some(JxlTransferFunction::Linear)) => {
            (InverseTransfer::None, 1.0)
        }
        (SourceDynamicRange::Pq, Some(JxlTransferFunction::PQ)) => {
            (InverseTransfer::Pq, PQ_PEAK_NITS / HDR_REFERENCE_WHITE_NITS)
        }
        (SourceDynamicRange::Hlg, Some(JxlTransferFunction::HLG)) => {
            (InverseTransfer::Hlg, intensity / HDR_REFERENCE_WHITE_NITS)
        }
        // XYB fallback output is already display-linear and relative
        // to the intensity target. Do not repeat the HLG OOTF.
        (SourceDynamicRange::Pq | SourceDynamicRange::Hlg, Some(JxlTransferFunction::Linear)) => {
            (InverseTransfer::None, intensity / HDR_REFERENCE_WHITE_NITS)
        }
        _ => {
            return Err(profile_error(format!(
                "unsupported {transfer:?} output for a {source_dynamic_range:?} source"
            )));
        }
    })
}

struct ColorConversion {
    transform: Option<Arc<TransformF32Executor>>,
    gray: bool,
    associated: bool,
    inverse_transfer: InverseTransfer,
    scale: f32,
    hlg_gamma: Option<f32>,
    input_range: Option<RangeInclusive<f32>>,
    clamp_to_nominal_range: bool,
}

impl ColorConversion {
    #[expect(clippy::cast_possible_truncation)] // JXL intensity is f32 metadata.
    fn new(
        profile: &JxlColorProfile,
        associated: bool,
        source_dynamic_range: SourceDynamicRange,
        intensity: SourceIntensityTarget,
    ) -> Result<Self, DecodeError> {
        let intensity = intensity.nits() as f32;
        let (inverse_transfer, scale) =
            output_processing(source_dynamic_range, profile.transfer_function(), intensity)?;
        let hlg_gamma =
            (inverse_transfer == InverseTransfer::Hlg).then(|| hlg_system_gamma(intensity));
        let source = if inverse_transfer == InverseTransfer::None {
            profile.clone()
        } else {
            profile
                .with_linear_tf()
                .ok_or_else(|| profile_error("missing linear HDR profile"))?
        };
        let target = working_profile();
        let gray = source.channels() == 1;
        let mut bounded_below = false;
        let mut bounded_above = false;
        let transform = if source.same_color_encoding(&target) {
            None
        } else {
            let source_icc = source
                .try_as_icc()
                .ok_or_else(|| profile_error("missing source ICC profile"))?;
            let target_icc = target
                .try_as_icc()
                .ok_or_else(|| profile_error("missing target ICC profile"))?;
            let source = ColorProfile::new_from_slice(&source_icc).map_err(profile_error)?;
            let target = ColorProfile::new_from_slice(&target_icc).map_err(profile_error)?;
            if !matches!(
                source.color_space,
                DataColorSpace::Rgb | DataColorSpace::Gray
            ) {
                return Err(profile_error(
                    "only RGB and gray ICC profiles are supported",
                ));
            }
            // LUT profiles do not promise extended-range behavior.
            let lut = source.lut_a_to_b_perceptual.is_some()
                || source.lut_a_to_b_colorimetric.is_some()
                || source.lut_a_to_b_saturation.is_some();
            bounded_below = lut;
            bounded_above = lut;
            Some(
                source
                    .create_transform_f32(
                        if gray { Layout::Gray } else { Layout::Rgb },
                        &target,
                        Layout::Rgb,
                        TransformOptions {
                            rendering_intent: profile_intent(&source_icc),
                            prefer_fixed_point: false,
                            allow_extended_range_rgb_xyz: true,
                            ..Default::default()
                        },
                    )
                    .map_err(profile_error)?,
            )
        };
        if let Some(transform) = &transform {
            let support = supported_input_range(transform.as_ref(), if gray { 1 } else { 3 });
            bounded_below |= !support.below_zero;
            bounded_above |= !support.above_one;
        }
        Ok(Self {
            transform,
            gray,
            associated,
            inverse_transfer,
            scale,
            hlg_gamma,
            input_range: input_range(bounded_below, bounded_above),
            clamp_to_nominal_range: false,
        })
    }

    fn input_channels(&self) -> usize {
        if self.gray { 1 } else { 3 }
    }

    // Writes one chunk of source samples as normalized working-space RGB.
    fn convert_chunk(
        &self,
        chunk: &[CanonicalPixel],
        input: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), DecodeError> {
        let channels = self.input_channels();
        let input = &mut input[..chunk.len() * channels];
        for (pixel, samples) in chunk.iter().zip(input.chunks_exact_mut(channels)) {
            for (channel, value) in samples.iter_mut().enumerate() {
                *value = self.unpack(*pixel, channel);
            }
        }
        if let Some(range) = &self.input_range
            && input
                .iter()
                .any(|value| !value.is_finite() || !range.contains(value))
        {
            return Err(profile_error(
                "extended-range samples with a range-limited ICC profile are not supported",
            ));
        }
        if let Some(transform) = &self.transform {
            transform.transform(input, output).map_err(profile_error)?;
        } else {
            output.copy_from_slice(input);
        }
        for rgb in output.chunks_exact_mut(3) {
            self.normalize(rgb);
        }
        Ok(())
    }

    fn unpack(&self, pixel: CanonicalPixel, channel: usize) -> f32 {
        let mut value = half::f16::from_bits(pixel[channel]).to_f32();
        // The nominal range bounds stored samples. Unpremultiplication can
        // recover valid above-white color even from integer samples.
        if self.clamp_to_nominal_range {
            value = value.clamp(0.0, 1.0);
        }
        if self.associated {
            let alpha = finite_alpha(half::f16::from_bits(pixel[3]).to_f32());
            value = if alpha > 0.0 { value / alpha } else { 0.0 };
        }
        // jxl clamps only converted XYB output. Original-profile float
        // samples get the same range: overshoot beyond it is not meaningful
        // luminance and can overflow f16 storage.
        match self.inverse_transfer {
            InverseTransfer::None => value,
            InverseTransfer::Pq => pq_eotf(value.clamp(0.0, 1.0)),
            InverseTransfer::Hlg => hlg_inverse_oetf(value.clamp(HLG_SIGNAL_MIN, HLG_SIGNAL_MAX)),
        }
    }

    // Only linear RGB uses this matrix probe. Grayscale luminance is
    // channel zero. RGB weights are the Y row of source -> linear BT.2020.
    fn maximum_luminance(&self, pixels: &[CanonicalPixel]) -> Result<f32, DecodeError> {
        let weights = if self.gray {
            [1.0, 0.0, 0.0]
        } else if let Some(transform) = &self.transform {
            let mut columns = [0.0; 9];
            transform
                .transform(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &mut columns)
                .map_err(profile_error)?;
            std::array::from_fn(|channel| bt2020_luminance(&columns[channel * 3..channel * 3 + 3]))
        } else {
            BT2020_LUMA_F32
        };
        let channels = self.input_channels();
        Ok(pixels
            .par_iter()
            .map(|pixel| {
                let rgba = pixel.map(|value| half::f16::from_bits(value).to_f32());
                // Fully transparent pixels never count, in either alpha mode,
                // as in the measurements of `convert_band`. Color under
                // alpha 0 is invisible and can be arbitrary, so it must not
                // change the brightness of the visible image. Opaque images
                // decode with alpha 1.0.
                if !rgba[3].is_finite()
                    || rgba[3] <= 0.0
                    || rgba[..channels].iter().any(|value| !value.is_finite())
                {
                    return 0.0;
                }
                let alpha = if self.associated {
                    finite_alpha(rgba[3])
                } else {
                    1.0
                };
                rgba[..channels]
                    .iter()
                    .zip(weights)
                    .map(|(value, weight)| value / alpha * weight)
                    .sum::<f32>()
            })
            .reduce(|| 0.0, f32::max))
    }

    fn normalize(&self, rgb: &mut [f32]) {
        let multiplier = self.hlg_gamma.map_or(self.scale, |gamma| {
            // Black stays black, as in the shader's `hlg_input_to_canonical`.
            // Below a target of about 300 nits the system gamma is below 1,
            // and zero to the power gamma - 1 is infinite.
            let luminance = bt2020_luminance(rgb);
            if luminance > 0.0 {
                luminance.powf(gamma - 1.0) * self.scale
            } else {
                0.0
            }
        });
        for value in rgb {
            *value *= multiplier;
        }
    }
}

fn bt2020_luminance(rgb: &[f32]) -> f32 {
    rgb[0].mul_add(
        BT2020_LUMA_F32[0],
        rgb[1].mul_add(BT2020_LUMA_F32[1], rgb[2] * BT2020_LUMA_F32[2]),
    )
}

/// Header facts that resolve linear calibration and the HDR claim.
#[derive(Clone, Copy)]
struct SourceCalibration {
    dynamic_range: SourceDynamicRange,
    /// Header value before [`SourceIntensityTarget::from_jxl_metadata`].
    raw_intensity: f32,
    intensity: SourceIntensityTarget,
    xyb: bool,
    float_samples: bool,
}

impl SourceCalibration {
    fn from_header(
        info: &JxlBasicInfo,
        embedded_profile: &JxlColorProfile,
    ) -> Result<Self, DecodeError> {
        let raw_intensity = info.tone_mapping.intensity_target;
        Ok(Self {
            dynamic_range: dynamic_range(embedded_profile)?,
            raw_intensity,
            intensity: SourceIntensityTarget::from_jxl_metadata(raw_intensity),
            xyb: !info.uses_original_profile,
            float_samples: matches!(info.bit_depth, JxlBitDepth::Float { .. }),
        })
    }

    // libjxl writes 255 when the author set no target. An explicit 255 is
    // indistinguishable and is treated as the default.
    #[expect(clippy::float_cmp)] // The encoder default is an exact binary16 sentinel.
    fn default_target(self) -> bool {
        self.raw_intensity == 255.0
    }

    // Only non-XYB float sources with an explicit target call `maximum_luminance`.
    #[expect(clippy::cast_possible_truncation)] // Metadata originated as f32.
    fn linear_unit(
        self,
        maximum_luminance: impl FnOnce() -> Result<f32, DecodeError>,
    ) -> Result<LinearUnit, DecodeError> {
        let target = self.intensity.nits() as f32;
        let nits = if self.default_target() {
            // Uncalibrated: relative, like gamma-encoded SDR.
            HDR_REFERENCE_WHITE_NITS
        } else if self.xyb || !self.float_samples {
            // The XYB transform and the integer maximum-value rule define 1.0.
            target
        } else {
            // For original-profile floats the target only bounds luminance.
            target / maximum_luminance()?.max(1.0)
        };
        Ok(LinearUnit {
            nits,
            assumed: self.default_target(),
        })
    }

    fn hdr_source(self, source_peak: SourceIntensityTarget) -> bool {
        let evidence = match self.dynamic_range {
            // XYB samples with the default target carry neither a calibration nor
            // exact values. Codec overshoot alone can lift their peak above white.
            SourceDynamicRange::Linear if self.xyb && self.default_target() => return false,
            SourceDynamicRange::Sdr | SourceDynamicRange::Linear => source_peak,
            // The transfer declares HDR intent. The measured peak may sit near
            // white or include codec overshoot.
            SourceDynamicRange::Pq | SourceDynamicRange::Hlg => self.intensity,
        };
        evidence.nits() > f64::from(HDR_REFERENCE_WHITE_NITS)
    }
}

fn measured_source_peak(observed_peak_linear: f32) -> SourceIntensityTarget {
    // Half the f16 spacing above canonical white absorbs CMS coefficient error.
    // Darker content presents at white, where tone mapping is the identity.
    let peak_linear = if observed_peak_linear <= 1.0 + 1.0 / 2048.0 {
        1.0
    } else {
        observed_peak_linear
    };
    let nits = (peak_linear * HDR_REFERENCE_WHITE_NITS).min(PQ_PEAK_NITS);
    SourceIntensityTarget::new(f64::from(nits))
        .expect("measured peaks lie between reference white and 10,000 nits")
}

// Extended-range support is profile dependent even with the feature and runtime
// option enabled. Probe neutral samples on either side of the nominal domain.
// Reject out-of-domain pixels for transforms that clip or cannot evaluate them.
fn supported_input_range(transform: &TransformF32Executor, channels: usize) -> RangeSupport {
    const PROBES: [f32; 4] = [0.0, 1.0, -0.25, 2.0];
    let mut source = [0.0; 3 * PROBES.len()];
    for (pixel, value) in source.chunks_exact_mut(channels).zip(PROBES) {
        pixel.fill(value);
    }
    let mut output = [[0.0_f32; 3]; PROBES.len()];
    if transform
        .transform(
            &source[..channels * PROBES.len()],
            output.as_flattened_mut(),
        )
        .is_err()
        || output.as_flattened().iter().any(|value| !value.is_finite())
    {
        return RangeSupport::default();
    }
    let [zero, one, below_zero, above_one] = output;
    RangeSupport {
        below_zero: (0..3).all(|channel| below_zero[channel] < zero[channel] - 0.0001),
        above_one: (0..3).all(|channel| above_one[channel] > one[channel] + 0.0001),
    }
}

/// Out-of-domain inputs that a color transform preserves instead of clipping.
#[derive(Clone, Copy, Debug, Default)]
struct RangeSupport {
    below_zero: bool,
    above_one: bool,
}

fn input_range(bounded_below: bool, bounded_above: bool) -> Option<RangeInclusive<f32>> {
    (bounded_below || bounded_above).then(|| {
        let lower = if bounded_below {
            0.0
        } else {
            f32::NEG_INFINITY
        };
        let upper = if bounded_above { 1.0 } else { f32::INFINITY };
        lower..=upper
    })
}

fn conversion_workers(width: u32, height: u32) -> usize {
    rayon::current_num_threads()
        .min(MAX_WORKERS)
        .min(
            checked_pixel_count(width, height)
                .unwrap_or(usize::MAX)
                .div_ceil(CHUNK_PIXELS),
        )
        .min(usize_from_u32(
            height.div_ceil(coarse_downsample(width, height)),
        ))
        .max(1)
}

struct CalibratedImage {
    store: DecodedTileStore,
    source_peak: SourceIntensityTarget,
    histogram: Histogram,
    linear_unit: Option<LinearUnit>,
    hdr_source: bool,
}

// Linear calibration needs the decoded samples, and the measured peak exists
// only after conversion. A gain map reconstructs the HDR image on the way.
fn calibrate_and_convert(
    pixels: Vec<CanonicalPixel>,
    width: u32,
    height: u32,
    opaque: bool,
    mut conversion: ColorConversion,
    source: SourceCalibration,
    gain_map: Option<&GainMap>,
) -> Result<CalibratedImage, DecodeError> {
    let linear_unit = if source.dynamic_range == SourceDynamicRange::Linear {
        let unit = source.linear_unit(|| conversion.maximum_luminance(&pixels))?;
        conversion.scale = unit.nits / HDR_REFERENCE_WHITE_NITS;
        Some(unit)
    } else {
        None
    };
    let (store, stats) = convert_pixels(pixels, width, height, opaque, &conversion, gain_map)?;
    // The PQ and HLG target only bounds the content. Their pixels may stay far
    // below it or exceed it. HLG decoding still uses the target for its OOTF.
    let source_peak = match source.dynamic_range {
        SourceDynamicRange::Sdr if gain_map.is_none() => SourceIntensityTarget::REFERENCE_WHITE,
        SourceDynamicRange::Sdr
        | SourceDynamicRange::Linear
        | SourceDynamicRange::Pq
        | SourceDynamicRange::Hlg => measured_source_peak(stats.presentation_peak()),
    };
    Ok(CalibratedImage {
        store,
        source_peak,
        histogram: stats.histogram,
        linear_unit,
        hdr_source: source.hdr_source(source_peak),
    })
}

/// What conversion measures from the largest color channel of each visible pixel.
struct PixelStats {
    /// The exact maximum, which the bins of the histogram cannot give.
    peak: f32,
    histogram: Histogram,
}

impl PixelStats {
    const EMPTY: Self = Self {
        peak: 0.0,
        histogram: [0; HISTOGRAM_BINS],
    };

    fn add(&mut self, rgb: &[f32]) {
        let value = rgb.iter().copied().fold(0.0, f32::max);
        self.peak = self.peak.max(value);
        self.histogram[histogram_bin(value)] += 1;
    }

    fn merge(mut self, other: &Self) -> Self {
        self.peak = self.peak.max(other.peak);
        for (count, other) in self.histogram.iter_mut().zip(other.histogram) {
            *count += other;
        }
        self
    }

    /// The peak for presentation. A few glints would otherwise set the tone
    /// mapping of the whole image, so the brightest 0.01% of the pixels do not
    /// count. The histogram places them only to an eighth of a stop: the brightest
    /// bins are left out whole while they hold no more than that share, and
    /// the peak is the upper edge of the bin below them. With no bin left out
    /// it is the exact maximum.
    fn presentation_peak(&self) -> f32 {
        let glints = self.histogram.iter().sum::<u64>() / 10_000;
        let mut brighter = 0;
        for (bin, count) in self.histogram.iter().enumerate().rev() {
            if brighter + count > glints {
                return if brighter == 0 {
                    self.peak
                } else {
                    histogram_bin_end(bin)
                };
            }
            brighter += count;
        }
        self.peak
    }
}

/// The bin of a working-space value that is not negative. The exponent of its
/// eighth power counts eighths of a stop, so the bin comes from the float's
/// bits without a logarithm.
fn histogram_bin(value: f32) -> usize {
    const EXPONENT_SHIFT: u32 = f32::MANTISSA_DIGITS - 1;
    // SDR white is 1.0 only up to the rounding of its color conversion. Half a
    // step of the stored f16 lower, which is 1/256 in the eighth power, it
    // counts below reference white.
    const BELOW_EDGE: f32 = 1.0 - 1.0 / 256.0;
    // Biased exponent of 1.0, which starts the bins above reference white.
    let reference_white = usize_from_u32(1.0_f32.to_bits() >> EXPONENT_SHIFT);
    let square = value * value;
    let fourth = square * square;
    let power = fourth * fourth * BELOW_EDGE;
    let eighths = usize_from_u32(power.to_bits() >> EXPONENT_SHIFT);
    (eighths + HISTOGRAM_REFERENCE_WHITE_BIN)
        .saturating_sub(reference_white)
        .min(HISTOGRAM_BINS - 1)
}

/// The upper edge of a bin as a working-space value.
#[expect(clippy::cast_precision_loss)] // A histogram has 125 bins.
fn histogram_bin_end(bin: usize) -> f32 {
    let bins = (bin + 1) as f32 - HISTOGRAM_REFERENCE_WHITE_BIN as f32;
    (bins / HISTOGRAM_BINS_PER_STOP as f32).exp2()
}

fn convert_pixels(
    mut pixels: Vec<CanonicalPixel>,
    width: u32,
    height: u32,
    opaque: bool,
    conversion: &ColorConversion,
    gain_map: Option<&GainMap>,
) -> Result<(DecodedTileStore, PixelStats), DecodeError> {
    let downsample = coarse_downsample(width, height);
    let coarse_width = width.div_ceil(downsample);
    let coarse_height = height.div_ceil(downsample);
    let mut preview =
        allocate_vec::<CoarsePixel>(checked_pixel_count(coarse_width, coarse_height)?, false)?;
    let coarse_rows_per_worker =
        usize_from_u32(coarse_height).div_ceil(conversion_workers(width, height));
    let rows_per_worker = coarse_rows_per_worker
        .checked_mul(usize_from_u32(downsample))
        .ok_or(DecodeError::SizeOverflow)?;
    let band_pixels = rows_per_worker
        .checked_mul(usize_from_u32(width))
        .ok_or(DecodeError::SizeOverflow)?;
    let stats = pixels
        .par_chunks_mut(band_pixels)
        .zip(preview.par_chunks_mut(coarse_rows_per_worker * usize_from_u32(coarse_width)))
        .enumerate()
        .map(|(band, (pixels, preview))| {
            let gain_map = gain_map.map(|gain_map| (gain_map, band * band_pixels));
            convert_band(
                pixels, preview, width, downsample, opaque, conversion, gain_map,
            )
        })
        .try_reduce(|| PixelStats::EMPTY, |left, right| Ok(left.merge(&right)))?;
    Ok((
        DecodedTileStore::new(
            width,
            height,
            coarse_width,
            coarse_height,
            downsample,
            preview,
            pixels,
        ),
        stats,
    ))
}

// `gain_map` carries the index of the band's first pixel within the image.
fn convert_band(
    pixels: &mut [CanonicalPixel],
    preview: &mut [CoarsePixel],
    width: u32,
    downsample: u32,
    opaque: bool,
    conversion: &ColorConversion,
    gain_map: Option<(&GainMap, usize)>,
) -> Result<PixelStats, DecodeError> {
    let rows = pixels.len() / usize_from_u32(width);
    let scratch_pixels = pixels.len().min(CHUNK_PIXELS);
    let mut input = allocate_vec::<f32>(scratch_pixels * conversion.input_channels(), false)?;
    let mut output = allocate_vec::<f32>(scratch_pixels * 3, false)?;
    let mut preview = BandPreview::new(preview, width, downsample, opaque)?;
    let mut stats = PixelStats::EMPTY;
    for (chunk_index, chunk) in pixels.chunks_mut(CHUNK_PIXELS).enumerate() {
        let output = &mut output[..chunk.len() * 3];
        conversion.convert_chunk(chunk, &mut input, output)?;
        if let Some((gain_map, first_pixel)) = gain_map {
            gain_map.apply(output, first_pixel + chunk_index * CHUNK_PIXELS);
        }
        for (index, (pixel, rgb)) in chunk.iter_mut().zip(output.chunks_exact(3)).enumerate() {
            let alpha = if opaque {
                1.0
            } else {
                finite_alpha(half::f16::from_bits(pixel[3]).to_f32())
            };
            // Like the calibration in `maximum_luminance`, only visible
            // pixels count.
            if alpha > 0.0 {
                stats.add(rgb);
            }
            preview.add(chunk_index * CHUNK_PIXELS + index, rgb, alpha);
            *pixel = pack_premultiplied(rgb, alpha)?;
        }
    }
    preview.finish(rows);
    Ok(stats)
}

// Zero alpha also hides color that is not finite.
fn premultiply(rgb: &[f32], alpha: f32) -> [f32; 3] {
    if alpha > 0.0 {
        [rgb[0] * alpha, rgb[1] * alpha, rgb[2] * alpha]
    } else {
        [0.0; 3]
    }
}

fn pack_premultiplied(rgb: &[f32], alpha: f32) -> Result<CanonicalPixel, DecodeError> {
    let [red, green, blue] = premultiply(rgb, alpha).map(half::f16::from_f32);
    let pixel = [red, green, blue, half::f16::from_f32(alpha)];
    if pixel.iter().all(|value| value.is_finite()) {
        Ok(pixel.map(half::f16::to_bits))
    } else {
        Err(DecodeError::NonFiniteOutput)
    }
}

// Builds one band's coarse preview while its pixels are converted.
struct BandPreview<'a> {
    preview: &'a mut [CoarsePixel],
    width: usize,
    downsample: usize,
    coarse_width: usize,
    opaque: bool,
    // Premultiplied RGB and alpha of the coarse row being filled.
    sums: Vec<[f64; 4]>,
    coarse_y: usize,
}

impl<'a> BandPreview<'a> {
    fn new(
        preview: &'a mut [CoarsePixel],
        width: u32,
        downsample: u32,
        opaque: bool,
    ) -> Result<Self, DecodeError> {
        let width = usize_from_u32(width);
        let downsample = usize_from_u32(downsample);
        let coarse_width = width.div_ceil(downsample);
        let accumulator_width = if downsample == 1 { 0 } else { coarse_width };
        Ok(Self {
            preview,
            width,
            downsample,
            coarse_width,
            opaque,
            sums: allocate_vec(accumulator_width, false)?,
            coarse_y: 0,
        })
    }

    fn add(&mut self, position: usize, rgb: &[f32], alpha: f32) {
        if self.downsample == 1 {
            self.preview[position] = if alpha > 0.0 {
                [rgb[0], rgb[1], rgb[2], alpha]
            } else {
                [0.0; 4]
            };
            return;
        }
        if position / self.width / self.downsample != self.coarse_y {
            self.finish_row(self.downsample);
        }
        let sum = &mut self.sums[position % self.width / self.downsample];
        for (total, value) in sum[..3].iter_mut().zip(premultiply(rgb, alpha)) {
            *total += f64::from(value);
        }
        sum[3] += f64::from(alpha);
    }

    fn finish(mut self, rows: usize) {
        if self.downsample > 1 {
            self.finish_row(rows - self.coarse_y * self.downsample);
        }
    }

    fn finish_row(&mut self, rows: usize) {
        let start = self.coarse_y * self.coarse_width;
        finish_preview_row(
            &self.sums,
            &mut self.preview[start..start + self.coarse_width],
            self.width,
            self.downsample,
            rows,
            self.opaque,
        );
        self.sums.fill([0.0; 4]);
        self.coarse_y += 1;
    }
}

#[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Averaged preview intentionally uses f32.
fn finish_preview_row(
    sums: &[[f64; 4]],
    preview: &mut [CoarsePixel],
    width: usize,
    downsample: usize,
    rows: usize,
    opaque: bool,
) {
    for (x, (sum, pixel)) in sums.iter().zip(preview).enumerate() {
        let count = (width - x * downsample).min(downsample) as f64 * rows as f64;
        let alpha = if opaque { 1.0 } else { (sum[3] / count) as f32 };
        *pixel = if sum[3] > 0.0 {
            [
                (sum[0] / sum[3]) as f32,
                (sum[1] / sum[3]) as f32,
                (sum[2] / sum[3]) as f32,
                alpha,
            ]
        } else {
            [0.0, 0.0, 0.0, alpha]
        };
    }
}

fn finite_alpha(alpha: f32) -> f32 {
    if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn estimated_memory_cost_bytes(width: u32, height: u32) -> Result<u64, DecodeError> {
    let canonical = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(u64_from_usize(CANONICAL_BYTES_PER_PIXEL)))
        .ok_or(DecodeError::SizeOverflow)?;
    let downsample = coarse_downsample(width, height);
    canonical
        .checked_add(
            u64::from(width.div_ceil(downsample))
                * u64::from(height.div_ceil(downsample))
                * u64_from_usize(size_of::<CoarsePixel>()),
        )
        .ok_or(DecodeError::SizeOverflow)
}

fn estimated_conversion_scratch_bytes(width: u32, height: u32) -> Result<u64, DecodeError> {
    let pixels = checked_pixel_count(width, height)?.min(CHUNK_PIXELS);
    let downsample = coarse_downsample(width, height);
    let row = if downsample == 1 {
        0
    } else {
        usize_from_u32(width.div_ceil(downsample)) * size_of::<[f64; 4]>()
    };
    Ok(u64_from_usize(
        (pixels * CONVERSION_FLOATS_PER_PIXEL * size_of::<f32>() + row)
            * conversion_workers(width, height),
    ))
}

fn coarse_downsample(width: u32, height: u32) -> u32 {
    let mut minimum = 1_u32;
    let mut maximum = width.max(height).max(1);
    while minimum < maximum {
        let candidate = minimum + (maximum - minimum) / 2;
        let (w, h) = (width.div_ceil(candidate), height.div_ceil(candidate));
        if w <= MAX_COARSE_DIMENSION
            && h <= MAX_COARSE_DIMENSION
            && u64::from(w) * u64::from(h) * u64_from_usize(size_of::<CoarsePixel>())
                <= MAX_COARSE_BYTES
        {
            maximum = candidate;
        } else {
            minimum = candidate + 1;
        }
    }
    minimum
}

fn checked_pixel_count(width: u32, height: u32) -> Result<usize, DecodeError> {
    usize_from_u32(width)
        .checked_mul(usize_from_u32(height))
        .ok_or(DecodeError::SizeOverflow)
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), DecodeError> {
    if width == 0 || height == 0 {
        Err(DecodeError::InvalidDimensions { width, height })
    } else {
        checked_pixel_count(width, height)?
            .checked_mul(CANONICAL_BYTES_PER_PIXEL)
            .ok_or(DecodeError::SizeOverflow)
            .map(|_| ())
    }
}

fn validate_output_memory(required: u64, limits: DecodeLimits) -> Result<(), DecodeError> {
    let limit = u64_from_usize(limits.memory_ceiling_bytes);
    if required > limit {
        Err(DecodeError::OutputMemoryLimit { required, limit })
    } else {
        Ok(())
    }
}

fn source_metadata(info: &JxlBasicInfo, profile: &JxlColorProfile) -> ImageMetadata {
    let color_encoding = match profile {
        JxlColorProfile::Icc(bytes) => SourceColorEncoding::Icc {
            color_space: if profile.channels() == 1 {
                "Grayscale"
            } else {
                "RGB"
            }
            .to_owned(),
            profile_bytes: bytes.len(),
        },
        JxlColorProfile::Simple(encoding) => {
            let (space, white, primaries, transfer) = match encoding {
                JxlColorEncoding::RgbColorSpace {
                    white_point,
                    primaries,
                    transfer_function,
                    ..
                } => (
                    "RGB",
                    white_point_label(white_point),
                    Some(primaries_label(primaries)),
                    transfer_label(transfer_function),
                ),
                JxlColorEncoding::GrayscaleColorSpace {
                    white_point,
                    transfer_function,
                    ..
                } => (
                    "Grayscale",
                    white_point_label(white_point),
                    None,
                    transfer_label(transfer_function),
                ),
                JxlColorEncoding::XYB { .. } => (
                    "XYB",
                    "D65".to_owned(),
                    Some("sRGB".to_owned()),
                    "Unknown".to_owned(),
                ),
            };
            SourceColorEncoding::Enumerated {
                color_space: space.to_owned(),
                white_point: white,
                primaries,
                transfer_function: transfer,
            }
        }
    };
    ImageMetadata {
        color_encoding,
        tone_mapping: ToneMappingMetadata {
            intensity_target_nits: info.tone_mapping.intensity_target,
        },
        exif: None,
        xmp: None,
    }
}

// Display names follow ITU-R and SMPTE usage rather than the jxl crate's
// identifiers or its compact `Display` codes.
fn primaries_label(primaries: &JxlPrimaries) -> String {
    match primaries {
        JxlPrimaries::SRGB => "sRGB",
        // BT.2100 reuses the BT.2020 primaries.
        JxlPrimaries::BT2100 => "BT.2020",
        JxlPrimaries::P3 => "P3",
        JxlPrimaries::Chromaticities { .. } => "custom",
    }
    .to_owned()
}

fn white_point_label(white_point: &JxlWhitePoint) -> String {
    // ICC-derived encodings signal D50 as a custom chromaticity.
    const D50: (f32, f32) = (0.3457, 0.3585);
    const D50_TOLERANCE: f32 = 5.0e-4;
    match white_point {
        JxlWhitePoint::D65 => "D65".to_owned(),
        JxlWhitePoint::DCI => "DCI".to_owned(),
        JxlWhitePoint::E => "E (equal-energy)".to_owned(),
        JxlWhitePoint::Chromaticity { wx, wy }
            if (wx - D50.0).abs() <= D50_TOLERANCE && (wy - D50.1).abs() <= D50_TOLERANCE =>
        {
            "D50".to_owned()
        }
        JxlWhitePoint::Chromaticity { wx, wy } => format!("x {wx:.4}, y {wy:.4}"),
    }
}

fn transfer_label(transfer: &JxlTransferFunction) -> String {
    match transfer {
        JxlTransferFunction::SRGB => "sRGB".to_owned(),
        JxlTransferFunction::BT709 => "BT.709".to_owned(),
        JxlTransferFunction::Linear => "Linear".to_owned(),
        JxlTransferFunction::PQ => "PQ".to_owned(),
        JxlTransferFunction::HLG => "HLG".to_owned(),
        JxlTransferFunction::DCI => "DCI gamma 2.6".to_owned(),
        // JPEG XL stores the encoding exponent. Display gamma is its inverse.
        JxlTransferFunction::Gamma(exponent) => {
            let gamma = format!("{:.2}", 1.0 / exponent);
            format!(
                "Gamma {}",
                gamma.trim_end_matches('0').trim_end_matches('.')
            )
        }
    }
}

fn extract_aux_boxes(decoder: &JxlDecoder, metadata: &mut ImageMetadata) {
    for box_type in [JxlAuxBoxType::EXIF, XML] {
        for aux in decoder.aux_boxes(box_type) {
            store_aux_box(metadata, aux.box_type(), |limit| {
                read_aux_box(aux, &[], limit)
            });
        }
    }
}

/// Stores the first EXIF and the first XMP box. `read` gets the size limit
/// for that kind of box. A box that cannot be read becomes a parse error,
/// like invalid content: metadata never stops the image from showing, just
/// as a broken gain map falls back to the base image.
fn store_aux_box<'a>(
    metadata: &mut ImageMetadata,
    box_type: JxlAuxBoxType,
    read: impl FnOnce(usize) -> Result<Cow<'a, [u8]>, String>,
) {
    if box_type == JxlAuxBoxType::EXIF && metadata.exif.is_none() {
        let exif = read(EXIF_OFFSET_BYTES + MAX_METADATA_BYTES).and_then(|data| {
            let (offset, payload) = data
                .split_first_chunk::<EXIF_OFFSET_BYTES>()
                .ok_or_else(|| "the box is shorter than its TIFF offset".to_owned())?;
            Ok(parse_exif(payload, u32::from_be_bytes(*offset)))
        });
        metadata.exif = Some(exif.unwrap_or_else(|error| ExifMetadata {
            parse_error: Some(error),
            ..ExifMetadata::default()
        }));
    } else if box_type == XML && metadata.xmp.is_none() {
        metadata.xmp = Some(read(MAX_METADATA_BYTES).map_or_else(
            |error| XmpMetadata {
                parse_error: Some(error),
                rating: None,
            },
            |data| parse_xmp(&data),
        ));
    }
}

/// Reads a box payload, decompressing a `brob` box within `limit`.
fn read_aux_box<'a>(
    aux: &'a JxlAuxBox,
    trailing: &[u8],
    limit: usize,
) -> Result<Cow<'a, [u8]>, String> {
    if aux.is_compressed() {
        gain_map::decompress_brotli(aux.raw_data().chain(trailing), limit)
            .map(Cow::Owned)
            .map_err(|reason| format!("compressed box: {reason}"))
    } else {
        aux.data(trailing).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests;

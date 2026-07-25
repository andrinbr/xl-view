use std::fs::{self, File, Metadata};
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::Arc;

use jxl_oxide::image::ExtraChannelType;
use jxl_oxide::image::color::{
    ColourEncoding, ColourSpace, EnumColourEncoding, Primaries, TransferFunction, WhitePoint,
};
use jxl_oxide::{
    AllocTracker, AuxBoxData, HdrType, JxlImage, PixelFormat, Render, RenderingIntent,
};
use moxcms::{ColorProfile, DataColorSpace, Layout, TransformF32Executor, TransformOptions};
use rayon::prelude::*;

use crate::color::{
    HDR_REFERENCE_WHITE_NITS, SourceDynamicRange, SourceIntensityTarget, hlg_inverse_oetf,
    hlg_system_gamma, pq_eotf,
};
use crate::metadata::{parse_exif, parse_xmp};

use super::{
    CANONICAL_BYTES_PER_PIXEL, CanonicalPixel, CoarsePixel, DecodeError, DecodeLimits,
    DecodeResult, DecodedImage, DecodedTileStore, ImageKey, ImageMetadata, SourceColorEncoding,
    ToneMappingMetadata, usize_from_u32,
};

const MAX_RENDER_CHUNK_BYTES: usize = 32 * 1024 * 1024;
const MAX_COARSE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_COARSE_DIMENSION: u32 = 4_096;

/// Decodes a filesystem source within the supplied limits.
///
/// # Errors
///
/// Returns a typed error for I/O, malformed input, unsupported content, or a
/// configured resource limit.
pub fn decode_file(path: &Path, limits: DecodeLimits) -> DecodeResult {
    let file = File::open(path).map_err(|source| DecodeError::Read {
        path: path.to_owned(),
        source,
    })?;
    let image = read_image(file, limits)?;
    decode_initialized_image(image, limits, None)
}

pub(super) fn decode_file_for_key(
    key: &ImageKey,
    limits: DecodeLimits,
    maximum_retained_bytes: Option<u64>,
) -> DecodeResult {
    let path = &key.normalized_path;
    let mut file = File::open(path).map_err(|source| DecodeError::Read {
        path: path.clone(),
        source,
    })?;
    let before = file
        .metadata()
        .map_err(|source| DecodeError::SourceMetadata {
            path: path.clone(),
            source,
        })?;
    verify_source_identity(key, &before)?;

    let image = read_image(&mut file, limits);

    let after = file
        .metadata()
        .map_err(|source| DecodeError::SourceMetadata {
            path: path.clone(),
            source,
        })?;
    let path_after = fs::metadata(path).map_err(|source| DecodeError::SourceMetadata {
        path: path.clone(),
        source,
    })?;
    verify_source_identity(key, &after)?;
    verify_source_identity(key, &path_after)?;
    decode_initialized_image(image?, limits, maximum_retained_bytes)
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

/// Decodes an in-memory JPEG XL source within the supplied limits.
///
/// # Errors
///
/// Returns a typed error for malformed input, unsupported content, or a
/// configured resource limit.
pub fn decode_memory(bytes: &[u8], limits: DecodeLimits) -> DecodeResult {
    let image = read_image(Cursor::new(bytes), limits)?;
    decode_initialized_image(image, limits, None)
}

fn read_image(reader: impl Read, limits: DecodeLimits) -> Result<JxlImage, DecodeError> {
    JxlImage::builder()
        .alloc_tracker(AllocTracker::with_limit(limits.memory_ceiling_bytes))
        .read(reader)
        .map_err(|error| classify_decoder_error(error, limits.memory_ceiling_bytes))
}

fn decode_initialized_image(
    mut image: JxlImage,
    limits: DecodeLimits,
    maximum_retained_bytes: Option<u64>,
) -> DecodeResult {
    // Header metadata exists before rendering and before our output allocation.
    let width = image.width();
    let height = image.height();
    let (associated_alpha, grayscale, xyb_encoded) = {
        let source = &image.image_header().metadata;
        (
            source
                .ec_info
                .iter()
                .find_map(jxl_oxide::image::ExtraChannelInfo::alpha_associated)
                .unwrap_or(false),
            source.grayscale(),
            source.xyb_encoded,
        )
    };
    let metadata = extract_metadata(&image);
    let hdr_type = image.hdr_type();
    let source_dynamic_range = match hdr_type {
        Some(HdrType::Pq) => SourceDynamicRange::Pq,
        Some(HdrType::Hlg) => SourceDynamicRange::Hlg,
        None => SourceDynamicRange::Sdr,
    };
    validate_dimensions(width, height)?;
    validate_extra_channels(&image)?;
    let required_retained_bytes = estimated_memory_cost_bytes(width, height)?;
    let required_output_and_scratch_bytes = required_retained_bytes
        .checked_add(estimated_render_scratch_bytes(width, height)?)
        .ok_or(DecodeError::SizeOverflow)?;
    validate_output_memory(required_output_and_scratch_bytes, limits)?;
    if let Some(available) = maximum_retained_bytes {
        if required_retained_bytes > available {
            return Err(DecodeError::PrefetchTooLarge {
                required: required_retained_bytes,
                available,
            });
        }
    }
    let (render, pixel_format, transform) = if associated_alpha {
        if xyb_encoded || grayscale || hdr_type.is_some() {
            return Err(DecodeError::UnsupportedAssociatedAlphaEncoding);
        }
        let source_icc = image.rendered_icc();
        let source_format = image.pixel_format();
        let render = render_frame(&image, false, limits.memory_ceiling_bytes)?;
        request_working_space(&mut image);
        let target_icc = image.rendered_icc();
        let transform = create_rgb_icc_transform(&source_icc, &target_icc)?;
        (
            render,
            source_format,
            CanonicalTransform::AssociatedIcc(transform),
        )
    } else if let Some(hdr_type) = hdr_type {
        request_hdr_working_gamut(&mut image, hdr_type);
        let pixel_format = image.pixel_format();
        let render = render_frame(
            &image,
            matches!(metadata.color_encoding, SourceColorEncoding::Icc { .. }),
            limits.memory_ceiling_bytes,
        )?;
        (
            render,
            pixel_format,
            CanonicalTransform::Hdr {
                hdr_type,
                intensity_target: SourceIntensityTarget::from_jxl_metadata(
                    metadata.tone_mapping.intensity_target_nits,
                ),
            },
        )
    } else {
        request_working_space(&mut image);
        let pixel_format = image.pixel_format();
        let render = render_frame(
            &image,
            matches!(metadata.color_encoding, SourceColorEncoding::Icc { .. }),
            limits.memory_ceiling_bytes,
        )?;
        (render, pixel_format, CanonicalTransform::None)
    };
    let (store, memory_cost_bytes) =
        render_canonical_store(&render, pixel_format, width, height, &transform)?;
    Ok(Arc::new(DecodedImage {
        store: Arc::new(store),
        width,
        height,
        source_dynamic_range,
        metadata,
        memory_cost_bytes,
    }))
}

fn estimated_memory_cost_bytes(width: u32, height: u32) -> Result<u64, DecodeError> {
    let canonical = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(CANONICAL_BYTES_PER_PIXEL as u64))
        .ok_or(DecodeError::SizeOverflow)?;
    let downsample = coarse_downsample(width, height);
    let coarse = u64::from(width.div_ceil(downsample))
        .checked_mul(u64::from(height.div_ceil(downsample)))
        .and_then(|pixels| pixels.checked_mul(4 * size_of::<f32>() as u64))
        .ok_or(DecodeError::SizeOverflow)?;
    canonical
        .checked_add(coarse)
        .ok_or(DecodeError::SizeOverflow)
}

fn estimated_render_scratch_bytes(width: u32, height: u32) -> Result<u64, DecodeError> {
    let row_bytes = usize::try_from(width)
        .map_err(|_| DecodeError::SizeOverflow)?
        .checked_mul(4 * size_of::<f32>())
        .ok_or(DecodeError::SizeOverflow)?;
    let rows = render_chunk_rows(row_bytes, usize_from_u32(coarse_downsample(width, height)))
        .0
        .min(usize::try_from(height).map_err(|_| DecodeError::SizeOverflow)?);
    let bytes = rows
        .checked_mul(row_bytes)
        .ok_or(DecodeError::SizeOverflow)?;
    u64::try_from(bytes).map_err(|_| DecodeError::SizeOverflow)
}

enum CanonicalTransform {
    None,
    Hdr {
        hdr_type: HdrType,
        intensity_target: SourceIntensityTarget,
    },
    AssociatedIcc(Arc<TransformF32Executor>),
}

impl CanonicalTransform {
    fn apply(&self, pixels: &mut [f32]) -> Result<(), DecodeError> {
        match self {
            Self::None => Ok(()),
            Self::Hdr {
                hdr_type,
                intensity_target,
            } => {
                linearize_hdr(pixels, *hdr_type, *intensity_target);
                Ok(())
            }
            Self::AssociatedIcc(transform) => {
                unpremultiply_encoded(pixels);
                transform_rgb_icc(pixels, transform.as_ref())
            }
        }
    }
}

fn request_hdr_working_gamut(image: &mut JxlImage, hdr_type: HdrType) {
    image.request_color_encoding(EnumColourEncoding {
        colour_space: ColourSpace::Rgb,
        white_point: WhitePoint::D65,
        primaries: Primaries::Bt2100,
        tf: match hdr_type {
            HdrType::Pq => TransferFunction::Pq,
            HdrType::Hlg => TransferFunction::Hlg,
        },
        rendering_intent: RenderingIntent::Relative,
    });
}

#[allow(clippy::cast_possible_truncation)] // Canonical image storage is intentionally f32.
fn linearize_hdr(pixels: &mut [f32], hdr_type: HdrType, intensity_target: SourceIntensityTarget) {
    match hdr_type {
        HdrType::Pq => {
            let scale = 10_000.0 / HDR_REFERENCE_WHITE_NITS;
            pixels.par_chunks_exact_mut(4).for_each(|rgba| {
                for channel in &mut rgba[..3] {
                    *channel = pq_eotf(*channel) * scale;
                }
            });
        }
        HdrType::Hlg => {
            let intensity_target = intensity_target.nits() as f32;
            let gamma = hlg_system_gamma(intensity_target);
            let scale = intensity_target / HDR_REFERENCE_WHITE_NITS;
            pixels.par_chunks_exact_mut(4).for_each(|rgba| {
                let mut scene = [
                    hlg_inverse_oetf(rgba[0]),
                    hlg_inverse_oetf(rgba[1]),
                    hlg_inverse_oetf(rgba[2]),
                ];
                let luminance =
                    scene[0].mul_add(0.2627, scene[1].mul_add(0.6780, scene[2] * 0.0593));
                let multiplier = luminance.max(0.0).powf(gamma - 1.0) * scale;
                for channel in &mut scene {
                    *channel *= multiplier;
                }
                rgba[..3].copy_from_slice(&scene);
            });
        }
    }
}

fn request_working_space(image: &mut JxlImage) {
    image.request_color_encoding(EnumColourEncoding {
        colour_space: ColourSpace::Rgb,
        white_point: WhitePoint::D65,
        primaries: Primaries::Bt2100,
        tf: TransferFunction::Linear,
        rendering_intent: RenderingIntent::Relative,
    });
}

fn render_canonical_store(
    render: &Render,
    pixel_format: PixelFormat,
    width: u32,
    height: u32,
    transform: &CanonicalTransform,
) -> Result<(DecodedTileStore, usize), DecodeError> {
    if pixel_format.has_black() {
        return Err(DecodeError::UnsupportedPixelFormat(pixel_format.channels()));
    }
    let output_bytes = checked_output_bytes(width, height)?;

    let writer = TileStoreWriter::new(width, height, output_bytes, !pixel_format.has_alpha())?;
    let writer = stream_canonical_rgba(render, pixel_format, width, height, transform, writer)?;
    let store = writer.finish()?;
    let retained_bytes = store.memory_cost_bytes();
    Ok((store, retained_bytes))
}

fn render_frame(
    image: &JxlImage,
    source_uses_icc: bool,
    codec_allocation_ceiling: usize,
) -> Result<Render, DecodeError> {
    image.render_frame(0).map_err(|error| {
        if image.num_loaded_keyframes() == 0 {
            DecodeError::NoDisplayableKeyframe
        } else {
            classify_render_error(error, source_uses_icc, codec_allocation_ceiling)
        }
    })
}

fn stream_canonical_rgba(
    render: &Render,
    pixel_format: PixelFormat,
    width: u32,
    height: u32,
    transform: &CanonicalTransform,
    mut writer: TileStoreWriter,
) -> Result<TileStoreWriter, DecodeError> {
    let width_usize = usize::try_from(width).map_err(|_| DecodeError::SizeOverflow)?;
    let height_usize = usize::try_from(height).map_err(|_| DecodeError::SizeOverflow)?;
    let channels = pixel_format.channels();
    let rgba_row_samples = width_usize
        .checked_mul(4)
        .ok_or(DecodeError::SizeOverflow)?;
    let stream_row_samples = width_usize
        .checked_mul(channels)
        .ok_or(DecodeError::SizeOverflow)?;
    let row_bytes = rgba_row_samples
        .checked_mul(size_of::<f32>())
        .ok_or(DecodeError::SizeOverflow)?;
    let coarse_downsample = usize_from_u32(writer.downsample);
    let (rows_per_chunk, preview_chunks_are_independent) =
        render_chunk_rows(row_bytes, coarse_downsample);
    let rows_per_chunk = rows_per_chunk.min(height_usize);
    let mut stream = render.stream();
    let expected_samples = width_usize
        .checked_mul(height_usize)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(DecodeError::SizeOverflow)?;
    let actual_samples = usize::try_from(stream.width())
        .unwrap_or(usize::MAX)
        .saturating_mul(usize::try_from(stream.height()).unwrap_or(usize::MAX))
        .saturating_mul(usize::try_from(stream.channels()).unwrap_or(usize::MAX));
    if stream.width() != width
        || stream.height() != height
        || usize::try_from(stream.channels()).ok() != Some(channels)
    {
        return Err(DecodeError::IncompleteOutput {
            actual: actual_samples,
            expected: expected_samples,
        });
    }

    let scratch_samples = rows_per_chunk
        .checked_mul(rgba_row_samples)
        .ok_or(DecodeError::SizeOverflow)?;
    let scratch_bytes = scratch_samples
        .checked_mul(size_of::<f32>())
        .ok_or(DecodeError::SizeOverflow)?;
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(scratch_samples)
        .map_err(|source| DecodeError::ScratchAllocation {
            required: scratch_bytes,
            source,
        })?;
    scratch.resize(scratch_samples, 0.0_f32);
    let mut first_row = 0_u32;
    while first_row < height {
        let row_count = usize_from_u32(height - first_row).min(rows_per_chunk);
        let pixel_count = row_count
            .checked_mul(width_usize)
            .ok_or(DecodeError::SizeOverflow)?;
        let stream_samples = row_count
            .checked_mul(stream_row_samples)
            .ok_or(DecodeError::SizeOverflow)?;
        let rgba_samples = pixel_count
            .checked_mul(4)
            .ok_or(DecodeError::SizeOverflow)?;
        let written = stream.write_to_buffer(&mut scratch[..stream_samples]);
        if written != stream_samples {
            return Err(DecodeError::IncompleteOutput {
                actual: written,
                expected: stream_samples,
            });
        }
        expand_to_rgba_in_place(&mut scratch[..rgba_samples], pixel_format, pixel_count)?;
        transform.apply(&mut scratch[..rgba_samples])?;
        let row_count_u32 =
            u32::try_from(row_count).expect("chunk rows are bounded by the u32 image height");
        writer.write_rows(
            first_row,
            row_count_u32,
            &scratch[..rgba_samples],
            preview_chunks_are_independent,
        )?;
        first_row += row_count_u32;
    }
    Ok(writer)
}

fn render_chunk_rows(row_bytes: usize, coarse_downsample: usize) -> (usize, bool) {
    let maximum_rows = (MAX_RENDER_CHUNK_BYTES / row_bytes.max(1)).max(1);
    if maximum_rows >= coarse_downsample {
        (maximum_rows / coarse_downsample * coarse_downsample, true)
    } else {
        (maximum_rows, false)
    }
}

struct TileStoreWriter {
    pixels: Vec<CanonicalPixel>,
    width: u32,
    height: u32,
    coarse_width: u32,
    coarse_height: u32,
    downsample: u32,
    opaque: bool,
    coarse_pixels: Vec<CoarsePixel>,
    coarse_accumulator: Vec<[f64; 4]>,
    coarse_counts: Vec<u32>,
    current_coarse_row: u32,
}

impl TileStoreWriter {
    fn new(
        width: u32,
        height: u32,
        canonical_bytes: usize,
        opaque: bool,
    ) -> Result<Self, DecodeError> {
        let downsample = coarse_downsample(width, height);
        let coarse_width = width.div_ceil(downsample);
        let coarse_height = height.div_ceil(downsample);
        let coarse_width_usize =
            usize::try_from(coarse_width).map_err(|_| DecodeError::SizeOverflow)?;
        let coarse_height_usize =
            usize::try_from(coarse_height).map_err(|_| DecodeError::SizeOverflow)?;
        let coarse_pixel_count = coarse_width_usize
            .checked_mul(coarse_height_usize)
            .ok_or(DecodeError::SizeOverflow)?;
        debug_assert!(canonical_bytes.is_multiple_of(size_of::<CanonicalPixel>()));
        let canonical_pixel_count = canonical_bytes / size_of::<CanonicalPixel>();
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(canonical_pixel_count)
            .map_err(|source| DecodeError::OutputAllocation {
                required: canonical_bytes,
                source,
            })?;
        Ok(Self {
            pixels,
            width,
            height,
            coarse_width,
            coarse_height,
            downsample,
            opaque,
            coarse_pixels: Vec::with_capacity(coarse_pixel_count),
            coarse_accumulator: vec![[0.0; 4]; coarse_width_usize],
            coarse_counts: vec![0; coarse_width_usize],
            current_coarse_row: 0,
        })
    }

    fn flush_coarse_row(&mut self) {
        for (sum, count) in self
            .coarse_accumulator
            .iter_mut()
            .zip(&mut self.coarse_counts)
        {
            self.coarse_pixels
                .push(finalize_coarse_pixel(*sum, *count, self.opaque));
            *sum = [0.0; 4];
            *count = 0;
        }
    }

    fn accumulate_partial_coarse_rows(&mut self, first_row: u32, pixels: &[f32]) {
        let width = usize_from_u32(self.width);
        for (local_row, row) in pixels.chunks_exact(width * 4).enumerate() {
            let source_y = first_row
                + u32::try_from(local_row)
                    .expect("chunk-local rows are bounded by the u32 image height");
            let coarse_y = source_y / self.downsample;
            if coarse_y != self.current_coarse_row {
                self.flush_coarse_row();
                self.current_coarse_row = coarse_y;
            }
            for (x, rgba) in row.chunks_exact(4).enumerate() {
                let alpha = if self.opaque {
                    1.0
                } else {
                    finite_alpha(rgba[3])
                };
                let coarse_x = x / usize_from_u32(self.downsample);
                let sum = &mut self.coarse_accumulator[coarse_x];
                sum[0] += f64::from(rgba[0] * alpha);
                sum[1] += f64::from(rgba[1] * alpha);
                sum[2] += f64::from(rgba[2] * alpha);
                sum[3] += f64::from(alpha);
                self.coarse_counts[coarse_x] += 1;
            }
        }
    }

    fn write_rows(
        &mut self,
        first_row: u32,
        row_count: u32,
        pixels: &[f32],
        preview_chunks_are_independent: bool,
    ) -> Result<(), DecodeError> {
        let width = usize_from_u32(self.width);
        let expected = width
            .checked_mul(usize_from_u32(row_count))
            .and_then(|pixel_count| pixel_count.checked_mul(4))
            .ok_or(DecodeError::SizeOverflow)?;
        if pixels.len() != expected {
            return Err(DecodeError::IncompleteOutput {
                actual: pixels.len(),
                expected,
            });
        }
        if preview_chunks_are_independent {
            debug_assert!(first_row.is_multiple_of(self.downsample));
            debug_assert!(
                row_count.is_multiple_of(self.downsample)
                    || first_row.saturating_add(row_count) == self.height
            );
        }

        append_premultiplied_rgba16f(pixels, &mut self.pixels, self.opaque);

        if preview_chunks_are_independent {
            append_coarse_preview_rows(
                pixels,
                self.width,
                row_count,
                self.downsample,
                self.opaque,
                &mut self.coarse_pixels,
            );
        } else {
            self.accumulate_partial_coarse_rows(first_row, pixels);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<DecodedTileStore, DecodeError> {
        if self.coarse_counts.iter().any(|count| *count != 0) {
            self.flush_coarse_row();
        }
        let expected_samples = usize_from_u32(self.coarse_width)
            .checked_mul(usize_from_u32(self.coarse_height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(DecodeError::SizeOverflow)?;
        let actual_samples = self.coarse_pixels.len().saturating_mul(4);
        if actual_samples != expected_samples {
            return Err(DecodeError::IncompleteOutput {
                actual: actual_samples,
                expected: expected_samples,
            });
        }
        let expected_bytes = usize_from_u32(self.width)
            .checked_mul(usize_from_u32(self.height))
            .and_then(|pixels| pixels.checked_mul(CANONICAL_BYTES_PER_PIXEL))
            .ok_or(DecodeError::SizeOverflow)?;
        let actual_bytes = self
            .pixels
            .len()
            .saturating_mul(size_of::<CanonicalPixel>());
        if actual_bytes != expected_bytes {
            return Err(DecodeError::IncompleteOutput {
                actual: actual_bytes,
                expected: expected_bytes,
            });
        }
        Ok(DecodedTileStore::new(
            self.width,
            self.height,
            self.coarse_width,
            self.coarse_height,
            self.downsample,
            self.coarse_pixels,
            self.pixels.into_boxed_slice(),
        ))
    }
}

fn coarse_downsample(width: u32, height: u32) -> u32 {
    let mut minimum = 1_u32;
    let mut maximum = width.max(height).max(1);
    while minimum < maximum {
        let candidate = minimum + (maximum - minimum) / 2;
        if coarse_preview_fits(width, height, candidate) {
            maximum = candidate;
        } else {
            minimum = candidate + 1;
        }
    }
    minimum
}

fn coarse_preview_fits(width: u32, height: u32, downsample: u32) -> bool {
    let coarse_width = width.div_ceil(downsample);
    let coarse_height = height.div_ceil(downsample);
    if coarse_width > MAX_COARSE_DIMENSION || coarse_height > MAX_COARSE_DIMENSION {
        return false;
    }
    u64::from(coarse_width)
        .checked_mul(u64::from(coarse_height))
        .and_then(|pixels| pixels.checked_mul(size_of::<CoarsePixel>() as u64))
        .is_some_and(|bytes| bytes <= MAX_COARSE_BYTES)
}

fn finite_alpha(alpha: f32) -> f32 {
    if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn append_premultiplied_rgba16f(
    source: &[f32],
    destination: &mut Vec<CanonicalPixel>,
    opaque: bool,
) {
    assert_eq!(source.len() % 4, 0);
    if opaque {
        destination.par_extend(source.par_chunks_exact(4).map(|rgba| {
            encode_canonical_pixel([
                half::f16::from_f32(rgba[0]).to_bits(),
                half::f16::from_f32(rgba[1]).to_bits(),
                half::f16::from_f32(rgba[2]).to_bits(),
                half::f16::ONE.to_bits(),
            ])
        }));
    } else {
        destination.par_extend(source.par_chunks_exact(4).map(|rgba| {
            let alpha = finite_alpha(rgba[3]);
            encode_canonical_pixel([
                half::f16::from_f32(rgba[0] * alpha).to_bits(),
                half::f16::from_f32(rgba[1] * alpha).to_bits(),
                half::f16::from_f32(rgba[2] * alpha).to_bits(),
                half::f16::from_f32(alpha).to_bits(),
            ])
        }));
    }
}

fn encode_canonical_pixel(channels: [u16; 4]) -> CanonicalPixel {
    let red = channels[0].to_le_bytes();
    let green = channels[1].to_le_bytes();
    let blue = channels[2].to_le_bytes();
    let alpha = channels[3].to_le_bytes();
    [
        red[0], red[1], green[0], green[1], blue[0], blue[1], alpha[0], alpha[1],
    ]
}

#[derive(Clone, Copy, Debug)]
struct CoarseBlock {
    first_x: usize,
    end_x: usize,
    first_y: usize,
    end_y: usize,
}

impl CoarseBlock {
    fn at(
        coarse_index: usize,
        coarse_width: usize,
        width: usize,
        row_count: usize,
        downsample: usize,
    ) -> Self {
        let coarse_x = coarse_index % coarse_width;
        let coarse_y = coarse_index / coarse_width;
        let first_x = coarse_x * downsample;
        let first_y = coarse_y * downsample;
        Self {
            first_x,
            end_x: (first_x + downsample).min(width),
            first_y,
            end_y: (first_y + downsample).min(row_count),
        }
    }

    fn sample_count(self) -> u32 {
        u32::try_from((self.end_x - self.first_x) * (self.end_y - self.first_y))
            .expect("coarse blocks contain at most u32-sized source geometry")
    }
}

#[allow(clippy::cast_possible_truncation)] // The retained coarse preview intentionally uses f32.
fn finalize_coarse_pixel(sum: [f64; 4], count: u32, opaque: bool) -> CoarsePixel {
    let divisor = f64::from(count.max(1));
    if opaque {
        return [
            (sum[0] / divisor) as f32,
            (sum[1] / divisor) as f32,
            (sum[2] / divisor) as f32,
            1.0,
        ];
    }

    let alpha = (sum[3] / divisor) as f32;
    if sum[3] > 0.0 && sum[3].is_finite() {
        [
            (sum[0] / sum[3]) as f32,
            (sum[1] / sum[3]) as f32,
            (sum[2] / sum[3]) as f32,
            alpha,
        ]
    } else {
        [0.0, 0.0, 0.0, alpha]
    }
}

fn average_opaque_coarse_block(source: &[f32], width: usize, block: CoarseBlock) -> CoarsePixel {
    let mut sum = [0.0_f64; 4];
    for y in block.first_y..block.end_y {
        let row = &source[y * width * 4..(y + 1) * width * 4];
        for rgba in row[block.first_x * 4..block.end_x * 4].chunks_exact(4) {
            sum[0] += f64::from(rgba[0]);
            sum[1] += f64::from(rgba[1]);
            sum[2] += f64::from(rgba[2]);
        }
    }
    finalize_coarse_pixel(sum, block.sample_count(), true)
}

fn average_transparent_coarse_block(
    source: &[f32],
    width: usize,
    block: CoarseBlock,
) -> CoarsePixel {
    let mut sum = [0.0_f64; 4];
    for y in block.first_y..block.end_y {
        let row = &source[y * width * 4..(y + 1) * width * 4];
        for rgba in row[block.first_x * 4..block.end_x * 4].chunks_exact(4) {
            let alpha = finite_alpha(rgba[3]);
            sum[0] += f64::from(rgba[0] * alpha);
            sum[1] += f64::from(rgba[1] * alpha);
            sum[2] += f64::from(rgba[2] * alpha);
            sum[3] += f64::from(alpha);
        }
    }
    finalize_coarse_pixel(sum, block.sample_count(), false)
}

fn average_coarse_block(
    source: &[f32],
    width: usize,
    block: CoarseBlock,
    opaque: bool,
) -> CoarsePixel {
    if opaque {
        average_opaque_coarse_block(source, width, block)
    } else {
        average_transparent_coarse_block(source, width, block)
    }
}

fn append_coarse_preview_rows(
    source: &[f32],
    width: u32,
    row_count: u32,
    downsample: u32,
    opaque: bool,
    destination: &mut Vec<CoarsePixel>,
) {
    let width = usize_from_u32(width);
    let row_count = usize_from_u32(row_count);
    let downsample = usize_from_u32(downsample);
    let coarse_width = width.div_ceil(downsample);
    let coarse_rows = row_count.div_ceil(downsample);
    destination.par_extend(
        (0..coarse_width * coarse_rows)
            .into_par_iter()
            .map(|coarse_index| {
                let block =
                    CoarseBlock::at(coarse_index, coarse_width, width, row_count, downsample);
                average_coarse_block(source, width, block, opaque)
            }),
    );
}

fn expand_to_rgba_in_place(
    pixels: &mut [f32],
    pixel_format: PixelFormat,
    pixel_count: usize,
) -> Result<(), DecodeError> {
    match pixel_format {
        PixelFormat::Rgb => {
            for index in (0..pixel_count).rev() {
                let source = index * 3;
                let destination = index * 4;
                let rgba = [pixels[source], pixels[source + 1], pixels[source + 2], 1.0];
                pixels[destination..destination + 4].copy_from_slice(&rgba);
            }
        }
        PixelFormat::Rgba => {}
        PixelFormat::Gray => {
            for index in (0..pixel_count).rev() {
                let gray = pixels[index];
                let destination = index * 4;
                pixels[destination..destination + 4].copy_from_slice(&[gray, gray, gray, 1.0]);
            }
        }
        PixelFormat::Graya => {
            for index in (0..pixel_count).rev() {
                let source = index * 2;
                let gray = pixels[source];
                let alpha = pixels[source + 1];
                let destination = index * 4;
                pixels[destination..destination + 4].copy_from_slice(&[gray, gray, gray, alpha]);
            }
        }
        PixelFormat::Cmyk | PixelFormat::Cmyka => {
            return Err(DecodeError::UnsupportedPixelFormat(pixel_format.channels()));
        }
    }
    Ok(())
}

fn classify_render_error(
    error: Box<dyn std::error::Error + Send + Sync + 'static>,
    source_uses_icc: bool,
    codec_allocation_ceiling: usize,
) -> DecodeError {
    if error_chain_contains::<jxl_grid::OutOfMemory>(error.as_ref()) {
        DecodeError::DecoderMemoryLimit {
            limit: codec_allocation_ceiling,
        }
    } else if source_uses_icc && error_chain_contains::<moxcms::CmsError>(error.as_ref()) {
        DecodeError::UnsupportedColorProfile(error.to_string())
    } else {
        DecodeError::Decoder(error)
    }
}

fn classify_decoder_error(
    error: Box<dyn std::error::Error + Send + Sync + 'static>,
    codec_allocation_ceiling: usize,
) -> DecodeError {
    if error_chain_contains::<jxl_grid::OutOfMemory>(error.as_ref()) {
        DecodeError::DecoderMemoryLimit {
            limit: codec_allocation_ceiling,
        }
    } else {
        DecodeError::Decoder(error)
    }
}

fn error_chain_contains<T>(mut error: &(dyn std::error::Error + 'static)) -> bool
where
    T: std::error::Error + 'static,
{
    loop {
        if error.downcast_ref::<T>().is_some() {
            return true;
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}

fn unpremultiply_encoded(pixels: &mut [f32]) {
    for rgba in pixels.chunks_exact_mut(4) {
        let alpha = rgba[3];
        if alpha > 0.0 && alpha.is_finite() {
            rgba[0] /= alpha;
            rgba[1] /= alpha;
            rgba[2] /= alpha;
        } else {
            rgba[0] = 0.0;
            rgba[1] = 0.0;
            rgba[2] = 0.0;
        }
    }
}

fn create_rgb_icc_transform(
    source_icc: &[u8],
    target_icc: &[u8],
) -> Result<Arc<TransformF32Executor>, DecodeError> {
    let source = ColorProfile::new_from_slice(source_icc)
        .map_err(|error| DecodeError::UnsupportedColorProfile(error.to_string()))?;
    let target = ColorProfile::new_from_slice(target_icc)
        .map_err(|error| DecodeError::UnsupportedColorProfile(error.to_string()))?;
    if source.color_space != DataColorSpace::Rgb || target.color_space != DataColorSpace::Rgb {
        return Err(DecodeError::UnsupportedColorProfile(
            "associated alpha currently requires RGB profiles".to_owned(),
        ));
    }
    source
        .create_transform_f32(
            Layout::Rgb,
            &target,
            Layout::Rgb,
            TransformOptions {
                rendering_intent: moxcms::RenderingIntent::RelativeColorimetric,
                ..Default::default()
            },
        )
        .map_err(|error| DecodeError::UnsupportedColorProfile(error.to_string()))
}

fn transform_rgb_icc(
    pixels: &mut [f32],
    transform: &TransformF32Executor,
) -> Result<(), DecodeError> {
    let mut input = vec![0.0_f32; 1024 * 3];
    let mut output = vec![0.0_f32; 1024 * 3];
    for chunk in pixels.chunks_mut(1024 * 4) {
        let pixel_count = chunk.len() / 4;
        for (rgb, rgba) in input[..pixel_count * 3]
            .chunks_exact_mut(3)
            .zip(chunk.chunks_exact(4))
        {
            rgb.copy_from_slice(&rgba[..3]);
        }
        transform
            .transform(&input[..pixel_count * 3], &mut output[..pixel_count * 3])
            .map_err(|error| DecodeError::UnsupportedColorProfile(error.to_string()))?;
        for (rgba, rgb) in chunk
            .chunks_exact_mut(4)
            .zip(output[..pixel_count * 3].chunks_exact(3))
        {
            rgba[..3].copy_from_slice(rgb);
        }
    }
    Ok(())
}

fn checked_pixel_count(width: u32, height: u32) -> Result<usize, DecodeError> {
    let width = usize::try_from(width).map_err(|_| DecodeError::SizeOverflow)?;
    let height = usize::try_from(height).map_err(|_| DecodeError::SizeOverflow)?;
    width.checked_mul(height).ok_or(DecodeError::SizeOverflow)
}

fn checked_output_bytes(width: u32, height: u32) -> Result<usize, DecodeError> {
    let pixel_count = checked_pixel_count(width, height)?;
    pixel_count
        .checked_mul(CANONICAL_BYTES_PER_PIXEL)
        .ok_or(DecodeError::SizeOverflow)
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), DecodeError> {
    if width == 0 || height == 0 {
        return Err(DecodeError::InvalidDimensions { width, height });
    }
    Ok(())
}

fn validate_output_memory(required: u64, limits: DecodeLimits) -> Result<(), DecodeError> {
    let limit = u64::try_from(limits.memory_ceiling_bytes).unwrap_or(u64::MAX);
    if required > limit {
        Err(DecodeError::OutputMemoryLimit { required, limit })
    } else {
        Ok(())
    }
}

fn validate_extra_channels(image: &JxlImage) -> Result<(), DecodeError> {
    for channel in &image.image_header().metadata.ec_info {
        if !matches!(channel.ty, ExtraChannelType::Alpha { .. }) {
            return Err(DecodeError::UnsupportedExtraChannel(format!(
                "{:?}",
                channel.ty
            )));
        }
    }
    Ok(())
}

fn colour_space_label(colour_space: ColourSpace) -> String {
    match colour_space {
        ColourSpace::Rgb => "Rgb",
        ColourSpace::Grey => "Grey",
        ColourSpace::Xyb => "Xyb",
        ColourSpace::Unknown => "Unknown",
    }
    .to_owned()
}

fn white_point_label(white_point: WhitePoint) -> String {
    match white_point {
        WhitePoint::D65 => "D65".to_owned(),
        WhitePoint::Custom(coordinates) => format!("Custom({coordinates:?})"),
        WhitePoint::E => "E".to_owned(),
        WhitePoint::Dci => "Dci".to_owned(),
    }
}

fn primaries_label(primaries: Primaries) -> String {
    match primaries {
        Primaries::Srgb => "Srgb".to_owned(),
        Primaries::Custom { red, green, blue } => {
            format!("Custom {{ red: {red:?}, green: {green:?}, blue: {blue:?} }}")
        }
        Primaries::Bt2100 => "Bt2100".to_owned(),
        Primaries::P3 => "P3".to_owned(),
    }
}

fn transfer_function_label(transfer_function: TransferFunction) -> String {
    match transfer_function {
        TransferFunction::Gamma { g, inverted } => {
            format!("Gamma {{ g: {g}, inverted: {inverted} }}")
        }
        TransferFunction::Bt709 => "BT.709".to_owned(),
        TransferFunction::Unknown => "Unknown".to_owned(),
        TransferFunction::Linear => "Linear".to_owned(),
        TransferFunction::Srgb => "sRGB".to_owned(),
        TransferFunction::Pq => "PQ".to_owned(),
        TransferFunction::Dci => "DCI".to_owned(),
        TransferFunction::Hlg => "HLG".to_owned(),
    }
}

fn extract_metadata(image: &JxlImage) -> ImageMetadata {
    let header = image.image_header();
    let source = &header.metadata;
    let color_encoding = match &source.colour_encoding {
        ColourEncoding::Enum(encoding) => SourceColorEncoding::Enumerated {
            colour_space: colour_space_label(encoding.colour_space),
            white_point: white_point_label(encoding.white_point),
            primaries: primaries_label(encoding.primaries),
            transfer_function: transfer_function_label(encoding.tf),
        },
        ColourEncoding::IccProfile(colour_space) => SourceColorEncoding::Icc {
            colour_space: colour_space_label(*colour_space),
            profile_bytes: image.original_icc().map_or(0, <[u8]>::len),
        },
    };
    let tone_mapping = &source.tone_mapping;
    let exif = match image.aux_boxes().first_exif() {
        Ok(AuxBoxData::Data(exif)) => Some(parse_exif(exif.payload(), exif.tiff_header_offset())),
        Ok(AuxBoxData::Decoding | AuxBoxData::NotFound) | Err(_) => None,
    };
    let xmp = match image.aux_boxes().first_xml() {
        AuxBoxData::Data(xml) => Some(parse_xmp(xml)),
        AuxBoxData::Decoding | AuxBoxData::NotFound => None,
    };

    ImageMetadata {
        color_encoding,
        tone_mapping: ToneMappingMetadata {
            intensity_target_nits: tone_mapping.intensity_target,
            min_nits: tone_mapping.min_nits,
        },
        exif,
        xmp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ChunkedReader<R> {
        inner: R,
        maximum_chunk_bytes: usize,
        reads: usize,
    }

    impl<R: Read> Read for ChunkedReader<R> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            self.reads += 1;
            let chunk_bytes = buffer.len().min(self.maximum_chunk_bytes);
            self.inner.read(&mut buffer[..chunk_bytes])
        }
    }

    #[test]
    fn parallel_rgba16f_packing_matches_the_scalar_reference() {
        let source: Vec<f32> = (0..65_537)
            .flat_map(|index| {
                let code = f32::from(u16::try_from(index % 4_096).unwrap()) / 4_095.0;
                let alpha = match index % 6 {
                    0 => -0.5,
                    1 => 0.0,
                    2 => 0.5,
                    3 => 1.0,
                    4 => 1.5,
                    _ => f32::NAN,
                };
                [code * 4.0 - 1.0, code * 0.75, code * 16.0, alpha]
            })
            .collect();
        let expected = encode_premultiplied_rgba16f_scalar_reference(&source);
        let mut actual = Vec::with_capacity(expected.len());
        let split = source.len() / 8 * 4;
        append_premultiplied_rgba16f(&source[..split], &mut actual, false);
        append_premultiplied_rgba16f(&source[split..], &mut actual, false);
        assert_eq!(actual, expected);
    }

    #[test]
    fn opaque_rgba16f_packing_skips_alpha_work() {
        let source = [-1.0, 0.25, 4.0, f32::NAN, 0.5, 1.5, 65_504.0, -1.0];
        let expected = source
            .chunks_exact(4)
            .map(|rgba| {
                encode_canonical_pixel([
                    half::f16::from_f32(rgba[0]).to_bits(),
                    half::f16::from_f32(rgba[1]).to_bits(),
                    half::f16::from_f32(rgba[2]).to_bits(),
                    half::f16::ONE.to_bits(),
                ])
            })
            .collect::<Vec<_>>();
        let mut actual = Vec::with_capacity(expected.len());
        append_premultiplied_rgba16f(&source, &mut actual, true);
        assert_eq!(actual, expected);
    }

    fn encode_premultiplied_rgba16f_scalar_reference(source: &[f32]) -> Vec<CanonicalPixel> {
        source
            .chunks_exact(4)
            .map(|rgba| {
                let alpha = finite_alpha(rgba[3]);
                encode_canonical_pixel([
                    half::f16::from_f32(rgba[0] * alpha).to_bits(),
                    half::f16::from_f32(rgba[1] * alpha).to_bits(),
                    half::f16::from_f32(rgba[2] * alpha).to_bits(),
                    half::f16::from_f32(alpha).to_bits(),
                ])
            })
            .collect()
    }

    #[test]
    fn parallel_coarse_preview_matches_the_scalar_reference_across_chunks() {
        const WIDTH: u32 = 7;
        const ROW_COUNT: u32 = 5;
        const DOWNSAMPLE: u32 = 3;
        let source = (0..WIDTH * ROW_COUNT)
            .flat_map(|index| {
                let value = f32::from(u16::try_from(index).unwrap()) / 10.0 - 0.5;
                let alpha = match index % 6 {
                    0 => -0.5,
                    1 => 0.0,
                    2 => 0.25,
                    3 => 0.75,
                    4 => 1.5,
                    _ => f32::NAN,
                };
                [value, value * 0.5, value * 2.0, alpha]
            })
            .collect::<Vec<_>>();
        let second_chunk_start = usize::try_from(DOWNSAMPLE * WIDTH * 4).unwrap();

        for opaque in [false, true] {
            let expected =
                coarse_preview_scalar_reference(&source, WIDTH, ROW_COUNT, DOWNSAMPLE, opaque);
            let mut actual = Vec::with_capacity(expected.len());
            append_coarse_preview_rows(
                &source[..second_chunk_start],
                WIDTH,
                DOWNSAMPLE,
                DOWNSAMPLE,
                opaque,
                &mut actual,
            );
            append_coarse_preview_rows(
                &source[second_chunk_start..],
                WIDTH,
                ROW_COUNT - DOWNSAMPLE,
                DOWNSAMPLE,
                opaque,
                &mut actual,
            );
            assert_eq!(actual, expected);
        }
    }

    #[allow(clippy::cast_possible_truncation)] // Test dimensions bound every scalar-reference accumulator before f32 storage.
    fn coarse_preview_scalar_reference(
        source: &[f32],
        width: u32,
        row_count: u32,
        downsample: u32,
        opaque: bool,
    ) -> Vec<CoarsePixel> {
        let width = usize::try_from(width).unwrap();
        let row_count = usize::try_from(row_count).unwrap();
        let downsample = usize::try_from(downsample).unwrap();
        let coarse_width = width.div_ceil(downsample);
        let coarse_rows = row_count.div_ceil(downsample);
        let mut sums = vec![[0.0_f64; 4]; coarse_width * coarse_rows];
        let mut counts = vec![0_u32; sums.len()];

        for y in 0..row_count {
            let row = &source[y * width * 4..(y + 1) * width * 4];
            for (x, rgba) in row.chunks_exact(4).enumerate() {
                let index = y / downsample * coarse_width + x / downsample;
                let sum = &mut sums[index];
                let alpha = if opaque { 1.0 } else { finite_alpha(rgba[3]) };
                sum[0] += f64::from(rgba[0] * alpha);
                sum[1] += f64::from(rgba[1] * alpha);
                sum[2] += f64::from(rgba[2] * alpha);
                sum[3] += f64::from(alpha);
                counts[index] += 1;
            }
        }

        sums.into_iter()
            .zip(counts)
            .map(|(sum, count)| {
                let divisor = f64::from(count.max(1));
                let alpha = (sum[3] / divisor) as f32;
                if sum[3] > 0.0 && sum[3].is_finite() {
                    [
                        (sum[0] / sum[3]) as f32,
                        (sum[1] / sum[3]) as f32,
                        (sum[2] / sum[3]) as f32,
                        alpha,
                    ]
                } else {
                    [0.0, 0.0, 0.0, alpha]
                }
            })
            .collect()
    }

    #[test]
    fn dimensions_have_no_fixed_upper_bound() {
        assert!(matches!(
            validate_dimensions(0, 1),
            Err(DecodeError::InvalidDimensions { .. })
        ));
        validate_dimensions(u32::MAX, 1).unwrap();
        validate_dimensions(1, u32::MAX).unwrap();
    }

    #[test]
    fn maximum_header_dimensions_cannot_overflow_output_storage() {
        assert!(matches!(
            checked_output_bytes(u32::MAX, u32::MAX),
            Err(DecodeError::SizeOverflow)
        ));
    }

    #[test]
    fn memory_limit_bounds_output_without_a_fixed_pixel_cap() {
        let one_gigapixel_width = 50_000;
        let one_gigapixel_height = 20_000;
        let generous_limits = DecodeLimits::from_memory_ceiling_mib(8 * 1024);
        let required = estimated_memory_cost_bytes(one_gigapixel_width, one_gigapixel_height)
            .unwrap()
            .checked_add(
                estimated_render_scratch_bytes(one_gigapixel_width, one_gigapixel_height).unwrap(),
            )
            .unwrap();
        validate_output_memory(required, generous_limits).unwrap();

        let constrained_limits = DecodeLimits::from_memory_ceiling_mib(4 * 1024);
        assert!(matches!(
            validate_output_memory(required, constrained_limits),
            Err(DecodeError::OutputMemoryLimit { .. })
        ));
    }

    #[test]
    fn extreme_single_row_is_governed_by_memory() {
        let width = u32::MAX;
        let required = estimated_memory_cost_bytes(width, 1)
            .unwrap()
            .checked_add(estimated_render_scratch_bytes(width, 1).unwrap())
            .unwrap();
        validate_output_memory(required, DecodeLimits::from_memory_ceiling_mib(128 * 1024))
            .unwrap();
        assert!(matches!(
            validate_output_memory(required, DecodeLimits::from_memory_ceiling_mib(64 * 1024),),
            Err(DecodeError::OutputMemoryLimit { .. })
        ));
    }

    #[test]
    fn canonical_store_reports_reservation_failure() {
        assert!(matches!(
            TileStoreWriter::new(1, 1, usize::MAX - 7, false),
            Err(DecodeError::OutputAllocation { .. })
        ));
    }

    #[test]
    fn every_image_uses_the_same_bounded_preview_policy() {
        assert_eq!(coarse_downsample(1_024, 1_024), 1);
        assert_eq!(coarse_downsample(6_000, 4_000), 3);
        assert_eq!(coarse_downsample(20_000, 5_000), 5);
        assert_eq!(coarse_downsample(28_000, 7_143), 7);
        assert_eq!(coarse_downsample(u32::MAX, 1), 1_048_576);
        assert_eq!(coarse_downsample(1, u32::MAX), 1_048_576);
        assert_eq!(coarse_downsample(u32::MAX, u32::MAX), 2_097_152);
    }

    #[test]
    fn render_chunks_align_to_preview_bands_without_exceeding_the_byte_limit() {
        let ordinary_row_bytes = 20_000 * 4 * size_of::<f32>();
        let (ordinary_rows, ordinary_is_independent) = render_chunk_rows(ordinary_row_bytes, 8);
        assert!(ordinary_is_independent);
        assert!(ordinary_rows.is_multiple_of(8));
        assert!(ordinary_rows * ordinary_row_bytes <= MAX_RENDER_CHUNK_BYTES);

        let wide_row_bytes = 1_000_000 * 4 * size_of::<f32>();
        let (wide_rows, wide_is_independent) = render_chunk_rows(wide_row_bytes, 256);
        assert!(!wide_is_independent);
        assert!(wide_rows * wide_row_bytes <= MAX_RENDER_CHUNK_BYTES);
        assert_eq!(estimated_render_scratch_bytes(1, 1).unwrap(), 16);
    }

    #[test]
    fn retained_memory_estimate_matches_completed_storage() {
        let bytes = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_pattern-sRGB.jxl"),
        )
        .unwrap();
        let image = decode_memory(&bytes, DecodeLimits::from_memory_ceiling_mib(64)).unwrap();
        assert_eq!(
            estimated_memory_cost_bytes(image.width, image.height).unwrap(),
            u64::try_from(image.memory_cost_bytes).unwrap()
        );
    }

    #[test]
    fn decoder_accepts_streamed_input() {
        let bytes = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_pattern-sRGB.jxl"),
        )
        .unwrap();
        let limits = DecodeLimits::from_memory_ceiling_mib(64);
        let mut reader = ChunkedReader {
            inner: Cursor::new(bytes),
            maximum_chunk_bytes: 7,
            reads: 0,
        };

        let image = read_image(&mut reader, limits).unwrap();
        assert!(reader.reads > 1);
        let decoded = decode_initialized_image(image, limits, None).unwrap();
        assert_eq!((decoded.width, decoded.height), (1_024, 1_024));
    }

    #[test]
    fn oversized_prefetch_stops_after_header_inspection() {
        let bytes = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_pattern-sRGB.jxl"),
        )
        .unwrap();
        let limits = DecodeLimits::from_memory_ceiling_mib(64);
        assert!(matches!(
            decode_initialized_image(
                read_image(Cursor::new(&bytes), limits).unwrap(),
                limits,
                Some(1),
            ),
            Err(DecodeError::PrefetchTooLarge { .. })
        ));
    }

    #[test]
    fn lower_channel_formats_expand_backward_without_a_second_buffer() {
        for (pixel_format, source, expected) in [
            (
                PixelFormat::Rgb,
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..],
                &[1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 1.0][..],
            ),
            (
                PixelFormat::Gray,
                &[0.25, 0.75][..],
                &[0.25, 0.25, 0.25, 1.0, 0.75, 0.75, 0.75, 1.0][..],
            ),
            (
                PixelFormat::Graya,
                &[0.25, 0.5, 0.75, 1.0][..],
                &[0.25, 0.25, 0.25, 0.5, 0.75, 0.75, 0.75, 1.0][..],
            ),
        ] {
            let mut pixels = vec![0.0; expected.len()];
            pixels[..source.len()].copy_from_slice(source);
            expand_to_rgba_in_place(&mut pixels, pixel_format, 2).unwrap();
            assert_eq!(pixels, expected);
        }
    }

    #[test]
    fn unsupported_pixel_formats_and_incomplete_storage_are_typed() {
        for (pixel_format, channels) in [(PixelFormat::Cmyk, 4), (PixelFormat::Cmyka, 5)] {
            assert!(matches!(
                expand_to_rgba_in_place(&mut [0.0; 4], pixel_format, 1),
                Err(DecodeError::UnsupportedPixelFormat(actual)) if actual == channels
            ));
        }

        let mut writer = TileStoreWriter::new(2, 2, 32, false).unwrap();
        assert!(matches!(
            writer.write_rows(0, 1, &[0.0; 7], true),
            Err(DecodeError::IncompleteOutput {
                actual: 7,
                expected: 8,
            })
        ));
        assert!(matches!(
            writer.finish(),
            Err(DecodeError::IncompleteOutput { .. })
        ));
    }

    #[test]
    fn cms_failures_have_a_specific_profile_error() {
        let error = Box::new(moxcms::CmsError::UnsupportedProfileConnection);
        assert!(matches!(
            classify_render_error(error, true, 1024),
            DecodeError::UnsupportedColorProfile(_)
        ));

        let error = Box::new(moxcms::CmsError::UnsupportedProfileConnection);
        assert!(matches!(
            classify_render_error(error, false, 1024),
            DecodeError::Decoder(_)
        ));
    }

    #[test]
    fn decoder_allocation_failures_have_a_specific_error() {
        let tracker = AllocTracker::with_limit(0);
        let allocation_error = tracker.alloc::<u8>(1).unwrap_err();

        assert!(matches!(
            classify_decoder_error(Box::new(allocation_error), 0),
            DecodeError::DecoderMemoryLimit { limit: 0 }
        ));
    }
}

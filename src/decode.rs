//! Memory-limited, asynchronous JPEG XL decoding.
//!
//! The module boundary is one oriented canonical tile store in linear-light
//! BT.2020. Every image is streamed to one contiguous in-memory row-major
//! premultiplied RGBA16F buffer plus a bounded straight-alpha whole-image
//! preview.
//! `jxl-oxide` performs the source-to-working-space transform through its
//! `moxcms` integration. [`jxl_oxide::Render::stream`] applies orientation; raw
//! color and extra-channel accessors do not, so this module only uses the
//! stream accessor. PQ and HLG are requested in BT.2020 without changing their
//! transfer function, then linearized here so `jxl-oxide` cannot implicitly
//! tone-map them through an SDR linear destination. PQ becomes absolute
//! luminance and HLG receives the source OOTF driven by its intensity target;
//! both are normalized so working-space `1.0` represents xl-view's fixed
//! 203-nit HDR reference white. SDR remains relative, with encoded white mapped to
//! working-space `1.0`.
//!
//! Associated-alpha RGB is rendered in its declared source encoding,
//! unpremultiplied there (transparent pixels become black), and then converted
//! to the working space with `moxcms`. Associated grayscale and XYB remain a
//! typed limitation because `jxl-oxide`'s public render path performs its color
//! transform before the application can safely unpremultiply those samples.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use half::prelude::{HalfBitsSliceExt, HalfFloatSliceExt};
use sysinfo::System;

mod coordinator;
mod jxl;

use crate::metadata::{ExifMetadata, XmpMetadata};

pub use crate::color::SourceDynamicRange;
pub use coordinator::{DecodeCompletion, DecodeCoordinator, DecodePurpose, DecodeQueueDisposition};
pub use jxl::{decode_file, decode_memory};

pub const TILE_SIZE: u32 = 512;
// Eight source pixels preserve one filtering texel through three tile mip
// levels. More deeply minified views continue sampling the coarse preview.
pub const TILE_GUTTER: u32 = 8;
pub const CANONICAL_BYTES_PER_PIXEL: usize = 8;
const HALF_CONVERSION_CHUNK_SAMPLES: usize = 1024;
const AUTOMATIC_MEMORY_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
type CanonicalPixel = [u8; CANONICAL_BYTES_PER_PIXEL];
type CoarsePixel = [f32; 4];

fn usize_from_u32(value: u32) -> usize {
    usize::try_from(value).expect("u32 geometry fits usize on 32-bit-or-wider targets")
}

fn automatic_memory_ceiling_bytes() -> usize {
    let mut system = System::new();
    system.refresh_memory();
    automatic_memory_ceiling_bytes_from_total(system.total_memory())
}

fn automatic_memory_ceiling_bytes_from_total(total_memory: u64) -> usize {
    let bytes = (total_memory / 8 * 3).saturating_sub(AUTOMATIC_MEMORY_RESERVE_BYTES);
    usize::try_from(bytes).unwrap_or(usize::MAX)
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ImageKey {
    pub normalized_path: PathBuf,
    pub source_len: u64,
    pub source_modified: SystemTime,
}

impl ImageKey {
    /// Resolves the filesystem identity of an image.
    ///
    /// # Errors
    ///
    /// Returns an error when the path cannot be normalized or its metadata
    /// cannot be inspected.
    pub fn from_path(path: &Path) -> Result<Self, DecodeError> {
        let normalized_path =
            std::fs::canonicalize(path).map_err(|source| DecodeError::SourceMetadata {
                path: path.to_owned(),
                source,
            })?;
        let metadata =
            std::fs::metadata(&normalized_path).map_err(|source| DecodeError::SourceMetadata {
                path: normalized_path.clone(),
                source,
            })?;
        let source_modified =
            metadata
                .modified()
                .map_err(|source| DecodeError::SourceMetadata {
                    path: normalized_path.clone(),
                    source,
                })?;
        Ok(Self {
            normalized_path,
            source_len: metadata.len(),
            source_modified,
        })
    }
}

/// Limits applied before and during decoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeLimits {
    /// Ceiling applied independently to codec-managed allocations and to
    /// application-owned decoded storage plus render scratch.
    ///
    /// This is not a process-wide peak-memory limit.
    pub memory_ceiling_bytes: usize,
}

impl DecodeLimits {
    /// Constructs decode limits with a custom per-category memory ceiling.
    ///
    /// The same ceiling is applied independently to codec-managed allocations
    /// and to application-owned decoded storage plus render scratch. The viewer
    /// uses [`Default`]. This constructor remains useful for tests that need a
    /// small deterministic ceiling.
    #[must_use]
    pub fn from_memory_ceiling_mib(memory_ceiling_mib: u64) -> Self {
        let bytes_u64 = memory_ceiling_mib.saturating_mul(1024 * 1024);
        let bytes = usize::try_from(bytes_u64).unwrap_or(usize::MAX);
        Self {
            memory_ceiling_bytes: bytes,
        }
    }
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            memory_ceiling_bytes: automatic_memory_ceiling_bytes(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToneMappingMetadata {
    pub intensity_target_nits: f32,
    pub min_nits: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceColorEncoding {
    Enumerated {
        colour_space: String,
        white_point: String,
        primaries: String,
        transfer_function: String,
    },
    Icc {
        colour_space: String,
        profile_bytes: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageMetadata {
    pub color_encoding: SourceColorEncoding,
    pub tone_mapping: ToneMappingMetadata,
    pub exif: Option<ExifMetadata>,
    pub xmp: Option<XmpMetadata>,
}

#[derive(Debug)]
pub struct DecodedTileStore {
    pixels: Box<[CanonicalPixel]>,
    width: u32,
    height: u32,
    coarse_width: u32,
    coarse_height: u32,
    coarse_downsample: u32,
    coarse_pixels: Vec<CoarsePixel>,
}

impl DecodedTileStore {
    #[must_use]
    pub fn coarse_dimensions(&self) -> (u32, u32) {
        (self.coarse_width, self.coarse_height)
    }

    #[must_use]
    pub fn coarse_downsample(&self) -> u32 {
        self.coarse_downsample
    }

    #[must_use]
    pub fn coarse_pixels(&self) -> &[f32] {
        self.coarse_pixels.as_flattened()
    }

    #[must_use]
    pub fn canonical_storage_bytes(&self) -> usize {
        self.pixels
            .len()
            .saturating_mul(size_of::<CanonicalPixel>())
    }

    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Decodes one canonical premultiplied RGBA16F row into `f32` working values.
    ///
    /// # Errors
    ///
    /// Returns an error when the row is outside the image or the destination
    /// does not contain exactly four samples per source pixel.
    pub fn read_canonical_row_rgba_f32(
        &self,
        row: u32,
        destination: &mut [f32],
    ) -> Result<(), io::Error> {
        let expected_samples = usize::try_from(self.width)
            .map_err(|_| io::Error::other("canonical row width is not representable"))?
            .checked_mul(4)
            .ok_or_else(|| io::Error::other("canonical row sample count overflowed"))?;
        if destination.len() != expected_samples {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "canonical row expected {expected_samples} destination samples, got {}",
                    destination.len()
                ),
            ));
        }
        self.read_canonical_row_range_rgba_f32(row, 0, destination)
    }

    /// Decodes part of one canonical premultiplied RGBA16F row into `f32`
    /// working values.
    ///
    /// The destination length selects the range width and must contain exactly
    /// four samples per requested source pixel.
    ///
    /// # Errors
    ///
    /// Returns an error when the row or selected column range is outside the
    /// image, or when the destination length is not a multiple of four.
    pub fn read_canonical_row_range_rgba_f32(
        &self,
        row: u32,
        start_column: u32,
        destination: &mut [f32],
    ) -> Result<(), io::Error> {
        if row >= self.height {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "canonical row is outside the decoded image",
            ));
        }
        if !destination.len().is_multiple_of(4) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "canonical row range needs four destination samples per pixel",
            ));
        }
        let range_width = u32::try_from(destination.len() / 4)
            .map_err(|_| io::Error::other("canonical row range width is not representable"))?;
        let end_column = start_column
            .checked_add(range_width)
            .ok_or_else(|| io::Error::other("canonical row range overflowed"))?;
        if end_column > self.width {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "canonical row range is outside the decoded image",
            ));
        }
        let image_width = usize::try_from(self.width)
            .map_err(|_| io::Error::other("canonical row width is not representable"))?;
        let row_start = usize::try_from(row)
            .map_err(|_| io::Error::other("canonical row is not representable"))?
            .checked_mul(image_width)
            .ok_or_else(|| io::Error::other("canonical row offset overflowed"))?;
        let range_start = usize::try_from(start_column)
            .map_err(|_| io::Error::other("canonical row column is not representable"))?
            .checked_add(row_start)
            .ok_or_else(|| io::Error::other("canonical row range offset overflowed"))?;
        let range_pixels = usize::try_from(range_width)
            .map_err(|_| io::Error::other("canonical row range width is not representable"))?;
        let range_end = range_start
            .checked_add(range_pixels)
            .ok_or_else(|| io::Error::other("canonical row end overflowed"))?;
        let source = self
            .pixels
            .get(range_start..range_end)
            .ok_or_else(|| io::Error::other("canonical row storage is incomplete"))?;
        let mut bits = [0_u16; HALF_CONVERSION_CHUNK_SAMPLES];
        for (encoded, samples) in source
            .as_flattened()
            .chunks(HALF_CONVERSION_CHUNK_SAMPLES * size_of::<u16>())
            .zip(destination.chunks_mut(HALF_CONVERSION_CHUNK_SAMPLES))
        {
            let bits = &mut bits[..samples.len()];
            for (encoded, bits) in encoded.chunks_exact(size_of::<u16>()).zip(bits.iter_mut()) {
                *bits = u16::from_le_bytes([encoded[0], encoded[1]]);
            }
            bits.reinterpret_cast::<half::f16>()
                .convert_to_f32_slice(samples);
        }
        Ok(())
    }

    #[must_use]
    pub fn memory_cost_bytes(&self) -> usize {
        self.pixels
            .len()
            .saturating_mul(size_of::<CanonicalPixel>())
            .saturating_add(
                self.coarse_pixels
                    .len()
                    .saturating_mul(size_of::<CoarsePixel>()),
            )
    }

    #[must_use]
    pub fn tile_columns(&self) -> u32 {
        self.width.div_ceil(TILE_SIZE)
    }

    #[must_use]
    pub fn tile_rows(&self) -> u32 {
        self.height.div_ceil(TILE_SIZE)
    }

    /// Reads one fixed-size tile with clamped filtering gutters.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested coordinate is outside the image or
    /// the tile allocation size overflows.
    ///
    /// # Panics
    ///
    /// Panics only if a `u32` tile dimension cannot fit `usize`, which cannot
    /// occur on targets with at least 32-bit pointers.
    pub fn read_tile_rgba16f(&self, tile_x: u32, tile_y: u32) -> Result<Vec<u8>, io::Error> {
        if tile_x >= self.tile_columns() || tile_y >= self.tile_rows() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "tile coordinate is outside the decoded image",
            ));
        }

        let extent = TILE_SIZE + TILE_GUTTER * 2;
        let extent_usize = usize_from_u32(extent);
        let row_bytes = extent_usize
            .checked_mul(CANONICAL_BYTES_PER_PIXEL)
            .ok_or_else(|| io::Error::other("tile row size overflowed"))?;
        let mut output = vec![
            0_u8;
            row_bytes
                .checked_mul(extent_usize)
                .ok_or_else(|| io::Error::other("tile size overflowed"))?
        ];
        let origin_x = tile_x * TILE_SIZE;
        let origin_y = tile_y * TILE_SIZE;
        let width = usize_from_u32(self.width);

        for output_y in 0..extent {
            let source_y = origin_y
                .saturating_add(output_y)
                .saturating_sub(TILE_GUTTER)
                .min(self.height - 1);
            let destination_row = &mut output
                [usize_from_u32(output_y) * row_bytes..usize_from_u32(output_y + 1) * row_bytes];
            for output_x in 0..extent {
                let source_x = origin_x
                    .saturating_add(output_x)
                    .saturating_sub(TILE_GUTTER)
                    .min(self.width - 1);
                let source_index = usize_from_u32(source_y) * width + usize_from_u32(source_x);
                let destination_index = usize_from_u32(output_x) * CANONICAL_BYTES_PER_PIXEL;
                destination_row[destination_index..destination_index + CANONICAL_BYTES_PER_PIXEL]
                    .copy_from_slice(&self.pixels[source_index]);
            }
        }
        Ok(output)
    }

    fn new(
        width: u32,
        height: u32,
        coarse_width: u32,
        coarse_height: u32,
        coarse_downsample: u32,
        coarse_pixels: Vec<CoarsePixel>,
        pixels: Box<[CanonicalPixel]>,
    ) -> Self {
        Self {
            pixels,
            width,
            height,
            coarse_width,
            coarse_height,
            coarse_downsample,
            coarse_pixels,
        }
    }
}

#[derive(Debug)]
pub struct DecodedImage {
    pub store: Arc<DecodedTileStore>,
    pub width: u32,
    pub height: u32,
    pub source_dynamic_range: SourceDynamicRange,
    pub metadata: ImageMetadata,
    pub memory_cost_bytes: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("cannot inspect JPEG XL source {path}: {source}")]
    SourceMetadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot read JPEG XL source {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("JPEG XL source changed while it was being opened: {path}")]
    SourceChanged { path: PathBuf },
    #[error("invalid or unsupported JPEG XL data: {0}")]
    Decoder(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("JPEG XL decoder allocations exceed the {limit} byte working-memory limit")]
    DecoderMemoryLimit { limit: usize },
    #[error("invalid image dimensions {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error(
        "decoded image and render scratch need {required} bytes; memory limit is {limit} bytes"
    )]
    OutputMemoryLimit { required: u64, limit: u64 },
    #[error(
        "decoded prefetch needs {required} bytes but only {available} cache bytes are available"
    )]
    PrefetchTooLarge { required: u64, available: u64 },
    #[error("cannot reserve {required} bytes for the decoded RGBA16F image: {source}")]
    OutputAllocation {
        required: usize,
        #[source]
        source: std::collections::TryReserveError,
    },
    #[error("cannot reserve {required} bytes for decode scratch: {source}")]
    ScratchAllocation {
        required: usize,
        #[source]
        source: std::collections::TryReserveError,
    },
    #[error("image size arithmetic overflowed")]
    SizeOverflow,
    #[error("unsupported JPEG XL extra channel: {0}")]
    UnsupportedExtraChannel(String),
    #[error("associated alpha with XYB or grayscale input is not safely transformable yet")]
    UnsupportedAssociatedAlphaEncoding,
    #[error("unsupported color profile: {0}")]
    UnsupportedColorProfile(String),
    #[error("decoder produced no displayable keyframe")]
    NoDisplayableKeyframe,
    #[error("decoder produced an unsupported {0}-channel pixel format")]
    UnsupportedPixelFormat(usize),
    #[error("decoder produced only {actual} of {expected} expected samples")]
    IncompleteOutput { actual: usize, expected: usize },
    #[error("decode coordinator stopped before completing the request")]
    Cancelled,
}

pub type DecodeResult = Result<Arc<DecodedImage>, DecodeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceiling_conversion_saturates_safely() {
        let limits = DecodeLimits::from_memory_ceiling_mib(u64::MAX);
        assert_eq!(limits.memory_ceiling_bytes, usize::MAX);
    }

    #[test]
    fn automatic_memory_ceiling_subtracts_fixed_reserve_from_three_eighths() {
        const GIB: u64 = 1024 * 1024 * 1024;

        assert_eq!(
            automatic_memory_ceiling_bytes_from_total(8 * GIB),
            usize::try_from(3 * GIB - AUTOMATIC_MEMORY_RESERVE_BYTES).unwrap()
        );
        assert_eq!(
            automatic_memory_ceiling_bytes_from_total(128 * GIB),
            usize::try_from(48 * GIB - AUTOMATIC_MEMORY_RESERVE_BYTES).unwrap()
        );
        assert_eq!(automatic_memory_ceiling_bytes_from_total(0), 0);
    }

    #[test]
    fn canonical_row_ranges_decode_only_selected_columns() {
        let pixels = (0..6_u16)
            .map(|pixel| {
                let channels = [f32::from(pixel), 0.25, 0.5, 1.0]
                    .map(half::f16::from_f32)
                    .map(half::f16::to_le_bytes);
                [
                    channels[0][0],
                    channels[0][1],
                    channels[1][0],
                    channels[1][1],
                    channels[2][0],
                    channels[2][1],
                    channels[3][0],
                    channels[3][1],
                ]
            })
            .collect::<Vec<_>>();
        let store = DecodedTileStore::new(3, 2, 1, 1, 1, vec![[0.0; 4]], pixels.into_boxed_slice());
        let mut selected = [0.0; 8];
        store
            .read_canonical_row_range_rgba_f32(1, 1, &mut selected)
            .unwrap();
        assert_eq!(&selected[..4], &[4.0, 0.25, 0.5, 1.0]);
        assert_eq!(&selected[4..], &[5.0, 0.25, 0.5, 1.0]);
        assert!(
            store
                .read_canonical_row_range_rgba_f32(1, 2, &mut selected)
                .is_err()
        );
    }

    #[test]
    fn canonical_row_conversion_handles_full_batches_and_tail() {
        let width = 257;
        let sample_count = usize::try_from(width).unwrap() * 4;
        let half_bits = (0..sample_count)
            .map(|index| u16::try_from(index * 61 % 0x7bff).unwrap())
            .collect::<Vec<_>>();
        let pixels = half_bits
            .chunks_exact(4)
            .map(|channels| {
                let red = channels[0].to_le_bytes();
                let green = channels[1].to_le_bytes();
                let blue = channels[2].to_le_bytes();
                let alpha = channels[3].to_le_bytes();
                [
                    red[0], red[1], green[0], green[1], blue[0], blue[1], alpha[0], alpha[1],
                ]
            })
            .collect::<Vec<_>>();
        let store =
            DecodedTileStore::new(width, 1, 1, 1, 1, vec![[0.0; 4]], pixels.into_boxed_slice());
        let mut actual = vec![0.0; sample_count];

        store.read_canonical_row_rgba_f32(0, &mut actual).unwrap();

        for (bits, actual) in half_bits.into_iter().zip(actual) {
            assert_eq!(
                actual.to_bits(),
                half::f16::from_bits(bits).to_f32().to_bits()
            );
        }
    }

    #[test]
    fn image_key_includes_file_identity() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_pattern-sRGB.jxl");
        let mut bytes = std::fs::read(fixture).unwrap();
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "xl-view-image-key-{}-{unique}.jxl",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();

        let original = ImageKey::from_path(&path).unwrap();

        bytes.push(0);
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(
            jxl::decode_file_for_key(&original, DecodeLimits::from_memory_ceiling_mib(64), None,),
            Err(DecodeError::SourceChanged { .. })
        ));
        let _ = std::fs::remove_file(path);
    }
}

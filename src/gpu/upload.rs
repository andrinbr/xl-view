use std::io;
use std::ops::Range;

use crate::units::usize_from_u32;

pub(super) const MAX_STAGING_BYTES: usize = 16 * 1024 * 1024;
const RGBA16F_BYTES_PER_PIXEL: u32 = 8;

/// Validated row geometry for one tightly packed CPU image uploaded to a GPU
/// texture through a reusable, row-aligned staging allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TextureUploadLayout {
    context: &'static str,
    width: u32,
    height: u32,
    row_bytes: u32,
    padded_row_bytes: u32,
    rows_per_stripe: u32,
    source_bytes: usize,
    staging_bytes: usize,
}

impl TextureUploadLayout {
    pub(super) fn rgba16f(
        width: u32,
        height: u32,
        context: &'static str,
    ) -> Result<Self, io::Error> {
        Self::new(width, height, RGBA16F_BYTES_PER_PIXEL, context)
    }

    fn new(
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
        context: &'static str,
    ) -> Result<Self, io::Error> {
        if width == 0 || height == 0 {
            return Err(io::Error::other(format!(
                "{context} dimensions must be non-zero"
            )));
        }
        let row_bytes = width
            .checked_mul(bytes_per_pixel)
            .ok_or_else(|| io::Error::other(format!("{context} row size overflowed")))?;
        let padded_row_bytes = row_bytes
            .checked_add(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            .map(|value| value & !(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1))
            .ok_or_else(|| io::Error::other(format!("{context} row alignment overflowed")))?;
        let row_bytes_usize = usize_from_u32(row_bytes);
        let padded_row_bytes_usize = usize_from_u32(padded_row_bytes);
        let height_usize = usize_from_u32(height);
        let source_bytes = row_bytes_usize
            .checked_mul(height_usize)
            .ok_or_else(|| io::Error::other(format!("{context} source size overflowed")))?;
        let rows_per_stripe_usize = (MAX_STAGING_BYTES / padded_row_bytes_usize)
            .max(1)
            .min(height_usize);
        let rows_per_stripe = u32::try_from(rows_per_stripe_usize)
            .expect("stripe rows are bounded by the validated u32 texture height");
        let staging_bytes = padded_row_bytes_usize
            .checked_mul(rows_per_stripe_usize)
            .ok_or_else(|| io::Error::other(format!("{context} staging size overflowed")))?;
        Ok(Self {
            context,
            width,
            height,
            row_bytes,
            padded_row_bytes,
            rows_per_stripe,
            source_bytes,
            staging_bytes,
        })
    }

    #[cfg(test)]
    const fn source_bytes(self) -> usize {
        self.source_bytes
    }

    pub(super) const fn staging_bytes(self) -> usize {
        self.staging_bytes
    }

    pub(super) fn allocate_staging(self) -> Vec<u8> {
        vec![0; self.staging_bytes]
    }

    pub(super) const fn stripes(self) -> UploadStripes {
        UploadStripes {
            next_row: 0,
            height: self.height,
            rows_per_stripe: self.rows_per_stripe,
        }
    }

    pub(super) fn validate_source_len(self, actual: usize) -> Result<(), io::Error> {
        if actual == self.source_bytes {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "{} expected {} bytes, got {actual}",
                self.context, self.source_bytes
            )))
        }
    }

    pub(super) fn source_row_range(self, row: u32) -> Range<usize> {
        debug_assert!(row < self.height);
        let start = usize_from_u32(row) * usize_from_u32(self.row_bytes);
        start..start + usize_from_u32(self.row_bytes)
    }

    pub(super) fn staging_row_mut(self, staging: &mut [u8], stripe_row: u32) -> &mut [u8] {
        debug_assert!(stripe_row < self.rows_per_stripe);
        debug_assert!(staging.len() >= self.staging_bytes);
        let start = usize_from_u32(stripe_row) * usize_from_u32(self.padded_row_bytes);
        &mut staging[start..start + usize_from_u32(self.row_bytes)]
    }

    pub(super) fn copy_stripe(self, source: &[u8], stripe: UploadStripe, staging: &mut [u8]) {
        debug_assert_eq!(source.len(), self.source_bytes);
        for stripe_row in 0..stripe.row_count {
            let source_row = stripe.first_row + stripe_row;
            let source_range = self.source_row_range(source_row);
            self.staging_row_mut(staging, stripe_row)
                .copy_from_slice(&source[source_range]);
        }
    }

    pub(super) fn stripe_data(self, staging: &[u8], stripe: UploadStripe) -> &[u8] {
        let bytes = usize_from_u32(stripe.row_count) * usize_from_u32(self.padded_row_bytes);
        &staging[..bytes]
    }

    pub(super) const fn copy_buffer_layout(
        self,
        stripe: UploadStripe,
    ) -> wgpu::TexelCopyBufferLayout {
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(self.padded_row_bytes),
            rows_per_image: Some(stripe.row_count),
        }
    }

    pub(super) const fn copy_extent(self, stripe: UploadStripe) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.width,
            height: stripe.row_count,
            depth_or_array_layers: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct UploadStripe {
    first_row: u32,
    row_count: u32,
}

impl UploadStripe {
    pub(super) const fn first_row(self) -> u32 {
        self.first_row
    }

    pub(super) const fn row_count(self) -> u32 {
        self.row_count
    }
}

pub(super) struct UploadStripes {
    next_row: u32,
    height: u32,
    rows_per_stripe: u32,
}

impl Iterator for UploadStripes {
    type Item = UploadStripe;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_row == self.height {
            return None;
        }
        let row_count = self.rows_per_stripe.min(self.height - self.next_row);
        let stripe = UploadStripe {
            first_row: self.next_row,
            row_count,
        };
        self.next_row += row_count;
        Some(stripe)
    }
}

pub(super) fn upload_staging_buffer_bytes(width: u32, height: u32) -> usize {
    if width == 0 || height == 0 {
        return 0;
    }
    TextureUploadLayout::rgba16f(width, height, "RGBA16F upload")
        .map_or(usize::MAX, TextureUploadLayout::staging_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba16f_layout_aligns_rows_and_bounds_the_staging_allocation() {
        let layout = TextureUploadLayout::rgba16f(528, 528, "test upload").unwrap();

        assert_eq!(layout.source_bytes(), 528 * 528 * 8);
        assert_eq!(layout.staging_bytes(), 4_352 * 528);
        assert!(layout.staging_bytes() <= MAX_STAGING_BYTES);
        assert_eq!(
            layout.stripes().collect::<Vec<_>>(),
            [UploadStripe {
                first_row: 0,
                row_count: 528,
            }]
        );
    }

    #[test]
    fn stripe_iteration_covers_large_images_once_without_exceeding_the_buffer() {
        let layout = TextureUploadLayout::rgba16f(4_096, 1_025, "test upload").unwrap();
        let stripes = layout.stripes().collect::<Vec<_>>();

        assert_eq!(
            stripes,
            [
                UploadStripe {
                    first_row: 0,
                    row_count: 512,
                },
                UploadStripe {
                    first_row: 512,
                    row_count: 512,
                },
                UploadStripe {
                    first_row: 1_024,
                    row_count: 1,
                },
            ]
        );
        assert_eq!(layout.staging_bytes(), MAX_STAGING_BYTES);
    }

    #[test]
    fn byte_copy_preserves_rows_and_leaves_alignment_padding_zeroed() {
        let layout = TextureUploadLayout::rgba16f(3, 2, "test upload").unwrap();
        let source = (0..layout.source_bytes())
            .map(|byte| u8::try_from(byte).unwrap())
            .collect::<Vec<_>>();
        let mut staging = layout.allocate_staging();
        let stripe = layout.stripes().next().unwrap();

        layout.copy_stripe(&source, stripe, &mut staging);

        assert_eq!(&staging[..24], &source[..24]);
        assert!(staging[24..256].iter().all(|&byte| byte == 0));
        assert_eq!(&staging[256..280], &source[24..48]);
        assert!(staging[280..512].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn layout_rejects_invalid_geometry_and_source_lengths_at_the_boundary() {
        assert!(
            TextureUploadLayout::rgba16f(0, 1, "test upload")
                .unwrap_err()
                .to_string()
                .contains("dimensions must be non-zero")
        );
        assert!(
            TextureUploadLayout::rgba16f(u32::MAX, 1, "test upload")
                .unwrap_err()
                .to_string()
                .contains("row size overflowed")
        );
        let layout = TextureUploadLayout::rgba16f(1, 1, "test upload").unwrap();
        assert_eq!(
            layout.validate_source_len(7).unwrap_err().to_string(),
            "test upload expected 8 bytes, got 7"
        );
    }
}

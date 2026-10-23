//! Separable Lanczos2 convolution for the viewport resampler.
//!
//! Source rows are read once, in increasing order, and filtered horizontally.
//! Each filtered row is added to every target row whose vertical support
//! contains it, so no complete horizontally filtered image is ever stored.
//!
//! Vertical supports are monotone: as the target row increases, neither the
//! first source row nor the exclusive end source row decreases. Target rows
//! therefore become active in order, when the scan reaches their first source
//! row, and complete in the same order, once the scan reaches the last row
//! before their `end_source_y`. The active rows form a queue: rows join at the
//! back, completed rows leave from the front and are encoded into the output,
//! and their sample buffers are reused for later rows.

use std::collections::VecDeque;
use std::io;

use half::prelude::{HalfBitsSliceExt, HalfFloatSliceExt};
use rayon::prelude::*;
use xl_view::decode::{CANONICAL_BYTES_PER_PIXEL, DecodedTileStore, HALF_CONVERSION_CHUNK_SAMPLES};

use super::ViewportRequest;
use xl_view::units::usize_from_u32;

// Small batches expose row-independent work without delaying cancellation;
// the byte cap prevents unusually wide images from multiplying scratch use.
const RESAMPLING_ROWS_PER_WORKER: usize = 2;
const MAX_RESAMPLING_ROW_BATCH_BYTES: usize = 32 * 1024 * 1024;
const MAX_COEFFICIENT_BYTES: usize = 64 * 1024 * 1024;
const MAX_FINITE_F16: f32 = half::f16::MAX.to_f32_const();

#[derive(Debug)]
pub(super) enum ResampleError {
    Cancelled,
    Failed(io::Error),
}

impl From<io::Error> for ResampleError {
    fn from(error: io::Error) -> Self {
        Self::Failed(error)
    }
}

#[derive(Clone, Copy, Debug)]
struct AxisBound {
    start: u32,
    length: u32,
    weights_start: usize,
}

#[derive(Default)]
struct AxisContributions {
    bounds: Vec<AxisBound>,
    weights: Vec<f32>,
}

impl AxisContributions {
    fn clear(&mut self) {
        self.bounds.clear();
        self.weights.clear();
    }

    fn allocated_bytes(&self) -> usize {
        allocation_bytes::<AxisBound>(self.bounds.capacity())
            .saturating_add(allocation_bytes::<f32>(self.weights.capacity()))
    }
}

#[inline]
fn axis_inputs_are_invalid(
    source_size: u32,
    target_size: u32,
    view_center: f64,
    view_scale: f64,
) -> bool {
    source_size == 0
        || target_size == 0
        || !view_center.is_finite()
        || !view_scale.is_finite()
        || view_scale <= 0.0
}

#[inline]
fn validate_weight_sum(sum: f64) -> Result<(), ResampleError> {
    if !sum.is_finite() || sum == 0.0 {
        return Err(io::Error::other("Lanczos2 produced invalid coefficient weights").into());
    }
    Ok(())
}

#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Dimensions bound the non-negative pixel indices; weights intentionally use f32 storage.
fn build_axis_contributions(
    source_size: u32,
    target_size: u32,
    view_center: f64,
    view_scale: f64,
    contributions: &mut AxisContributions,
    is_cancelled: &mut impl FnMut() -> bool,
) -> Result<(), ResampleError> {
    contributions.clear();
    if axis_inputs_are_invalid(source_size, target_size, view_center, view_scale) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "resampling view must have finite coordinates and nonzero dimensions",
        )
        .into());
    }
    let source_per_target = 1.0 / view_scale;
    let filter_scale = source_per_target.max(1.0);
    let radius = 2.0 * filter_scale;
    let maximum_window = (radius.ceil() as usize)
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| io::Error::other("Lanczos2 coefficient window overflowed"))?;
    let maximum_weights = usize_from_u32(target_size)
        .checked_mul(maximum_window)
        .ok_or_else(|| io::Error::other("Lanczos2 coefficient count overflowed"))?;
    let coefficient_bytes = maximum_weights
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| io::Error::other("Lanczos2 coefficient size overflowed"))?;
    if coefficient_bytes > MAX_COEFFICIENT_BYTES {
        return Err(io::Error::other(format!(
            "Lanczos2 coefficient table needs {coefficient_bytes} bytes; limit is {MAX_COEFFICIENT_BYTES}"
        ))
        .into());
    }

    contributions
        .bounds
        .try_reserve_exact(usize_from_u32(target_size))
        .map_err(|error| io::Error::other(format!("cannot allocate Lanczos2 bounds: {error}")))?;
    contributions
        .weights
        .try_reserve_exact(maximum_weights)
        .map_err(|error| io::Error::other(format!("cannot allocate Lanczos2 weights: {error}")))?;

    for target in 0..target_size {
        if target % 32 == 0 && is_cancelled() {
            return Err(ResampleError::Cancelled);
        }
        let source_center = view_center
            + (f64::from(target) + 0.5 - f64::from(target_size) * 0.5) * source_per_target;
        if source_center < 0.0 || source_center > f64::from(source_size) {
            contributions.bounds.push(AxisBound {
                start: 0,
                length: 0,
                weights_start: contributions.weights.len(),
            });
            continue;
        }
        let first = (source_center - radius).floor().max(0.0) as u32;
        let end = (source_center + radius).ceil().min(f64::from(source_size)) as u32;
        let weights_start = contributions.weights.len();
        let mut sum = 0.0_f64;
        for source in first..end {
            let distance = (f64::from(source) + 0.5 - source_center) / filter_scale;
            let weight = lanczos2(distance);
            contributions.weights.push(weight as f32);
            sum += weight;
        }
        validate_weight_sum(sum)?;
        for weight in &mut contributions.weights[weights_start..] {
            *weight = (f64::from(*weight) / sum) as f32;
        }
        contributions.bounds.push(AxisBound {
            start: first,
            length: end - first,
            weights_start,
        });
    }
    Ok(())
}

fn contribution_source_range(contributions: &AxisContributions) -> Option<(u32, u32)> {
    contributions
        .bounds
        .iter()
        .filter(|bound| bound.length != 0)
        .fold(None, |range, bound| {
            let end = bound.start + bound.length;
            Some(range.map_or((bound.start, end), |(start, previous_end)| {
                (start.min(bound.start), previous_end.max(end))
            }))
        })
}

fn lanczos2(value: f64) -> f64 {
    let value = value.abs();
    if value >= 2.0 {
        return 0.0;
    }
    if value < f64::EPSILON {
        return 1.0;
    }
    let pi_value = std::f64::consts::PI * value;
    (pi_value.sin() / pi_value) * ((pi_value * 0.5).sin() / (pi_value * 0.5))
}

type RgbaF32 = [f32; 4];

struct ActiveOutputRow {
    target_y: u32,
    /// Exclusive end of the source rows that contribute to `target_y`.
    end_source_y: u32,
    samples: Vec<RgbaF32>,
}

#[derive(Default)]
pub(super) struct ResamplingScratch {
    source_rows: Vec<RgbaF32>,
    horizontal_rows: Vec<RgbaF32>,
    pub(super) output: Vec<u8>,
    pub(super) upload_staging: Vec<u8>,
    active_rows: VecDeque<ActiveOutputRow>,
    free_row_buffers: Vec<Vec<RgbaF32>>,
    horizontal: AxisContributions,
    vertical: AxisContributions,
}

impl ResamplingScratch {
    pub(super) fn allocated_bytes(&self) -> usize {
        let direct_buffers = allocation_bytes::<RgbaF32>(self.source_rows.capacity())
            .saturating_add(allocation_bytes::<RgbaF32>(self.horizontal_rows.capacity()))
            .saturating_add(allocation_bytes::<u8>(self.output.capacity()))
            .saturating_add(allocation_bytes::<u8>(self.upload_staging.capacity()))
            .saturating_add(self.horizontal.allocated_bytes())
            .saturating_add(self.vertical.allocated_bytes())
            .saturating_add(allocation_bytes::<ActiveOutputRow>(
                self.active_rows.capacity(),
            ))
            .saturating_add(allocation_bytes::<Vec<RgbaF32>>(
                self.free_row_buffers.capacity(),
            ));
        self.active_rows
            .iter()
            .map(|row| allocation_bytes::<RgbaF32>(row.samples.capacity()))
            .chain(
                self.free_row_buffers
                    .iter()
                    .map(|row| allocation_bytes::<RgbaF32>(row.capacity())),
            )
            .fold(direct_buffers, usize::saturating_add)
    }
}

fn allocation_bytes<T>(capacity: usize) -> usize {
    capacity.saturating_mul(std::mem::size_of::<T>())
}

fn resampling_row_batch_capacity(
    source_row_pixels: usize,
    target_row_pixels: usize,
    source_row_count: usize,
) -> usize {
    let combined_row_bytes = source_row_pixels
        .saturating_add(target_row_pixels)
        .saturating_mul(std::mem::size_of::<RgbaF32>())
        .max(1);
    let memory_limited_batch_size = (MAX_RESAMPLING_ROW_BATCH_BYTES / combined_row_bytes).max(1);
    let scheduling_batch_size = rayon::current_num_threads()
        .saturating_mul(RESAMPLING_ROWS_PER_WORKER)
        .max(1);
    source_row_count
        .min(scheduling_batch_size)
        .min(memory_limited_batch_size)
}

#[inline]
fn activate_output_rows(
    vertical: &AxisContributions,
    target_height: u32,
    target_row_pixels: usize,
    source_y: u32,
    next_target_y: &mut u32,
    active: &mut VecDeque<ActiveOutputRow>,
    free_row_buffers: &mut Vec<Vec<RgbaF32>>,
) -> Result<(), ResampleError> {
    while *next_target_y < target_height {
        let bound = vertical.bounds[usize_from_u32(*next_target_y)];
        if bound.length == 0 {
            *next_target_y += 1;
            continue;
        }
        if bound.start > source_y {
            break;
        }
        let mut samples = free_row_buffers.pop().unwrap_or_default();
        prepare_zeroed_buffer(&mut samples, target_row_pixels, "active output row")?;
        active.push_back(ActiveOutputRow {
            target_y: *next_target_y,
            end_source_y: bound.start + bound.length,
            samples,
        });
        *next_target_y += 1;
    }
    Ok(())
}

#[inline]
fn flush_completed_output_rows(
    active: &mut VecDeque<ActiveOutputRow>,
    free_row_buffers: &mut Vec<Vec<RgbaF32>>,
    source_y: u32,
    output: &mut [u8],
    target_width: u32,
) {
    while active
        .front()
        .is_some_and(|row| row.end_source_y <= source_y + 1)
    {
        let mut row = active
            .pop_front()
            .expect("the front row was checked before removal");
        encode_resampled_row(&mut row.samples, output, target_width, row.target_y);
        free_row_buffers.push(row.samples);
    }
}

#[inline]
fn skip_empty_output_rows(
    vertical: &AxisContributions,
    target_height: u32,
    next_target_y: &mut u32,
) {
    while *next_target_y < target_height
        && vertical.bounds[usize_from_u32(*next_target_y)].length == 0
    {
        *next_target_y += 1;
    }
}

#[inline]
fn accumulate_output_row_weights(
    vertical: &AxisContributions,
    active_rows: &mut VecDeque<ActiveOutputRow>,
    source_y: u32,
    horizontal_row: &[RgbaF32],
) {
    for row in active_rows {
        let bound = vertical.bounds[usize_from_u32(row.target_y)];
        debug_assert!(bound.start <= source_y && source_y < row.end_source_y);
        let offset = usize_from_u32(source_y - bound.start);
        let weight = vertical.weights[bound.weights_start + offset];
        for (destination, sample) in row
            .samples
            .as_flattened_mut()
            .iter_mut()
            .zip(horizontal_row.as_flattened())
        {
            *destination += sample * weight;
        }
    }
}

pub(super) fn resample_lanczos2(
    source: &DecodedTileStore,
    view: ViewportRequest,
    scratch: &mut ResamplingScratch,
    is_cancelled: impl FnMut() -> bool,
) -> Result<usize, ResampleError> {
    resample_lanczos2_rows(
        source.dimensions(),
        view,
        scratch,
        is_cancelled,
        |row, start, destination| source.read_canonical_row_range_rgba_f32(row, start, destination),
    )
}

#[expect(clippy::too_many_lines)] // Keeps the cancellable convolution, memory-limited output, and 64 MiB-per-axis coefficient tables in one transaction.
fn resample_lanczos2_rows(
    (source_width, source_height): (u32, u32),
    view: ViewportRequest,
    scratch: &mut ResamplingScratch,
    mut is_cancelled: impl FnMut() -> bool,
    read_source_row: impl Fn(u32, u32, &mut [f32]) -> Result<(), io::Error> + Sync,
) -> Result<usize, ResampleError> {
    let target = view.target;
    build_axis_contributions(
        source_width,
        target.width,
        view.center_x,
        view.scale,
        &mut scratch.horizontal,
        &mut is_cancelled,
    )?;
    build_axis_contributions(
        source_height,
        target.height,
        view.center_y,
        view.scale,
        &mut scratch.vertical,
        &mut is_cancelled,
    )?;
    if is_cancelled() {
        return Err(ResampleError::Cancelled);
    }

    let horizontal_source_range = contribution_source_range(&scratch.horizontal);
    let vertical_source_range = contribution_source_range(&scratch.vertical);
    let source_row_pixels =
        usize_from_u32(horizontal_source_range.map_or(0, |(start, end)| end - start));
    let target_row_pixels = usize_from_u32(target.width);
    let output_bytes = usize_from_u32(target.width)
        .checked_mul(usize_from_u32(target.height))
        .and_then(|pixels| pixels.checked_mul(CANONICAL_BYTES_PER_PIXEL))
        .ok_or_else(|| io::Error::other("resampled output size overflowed"))?;
    let (first_source_y, end_source_y) = vertical_source_range.unwrap_or((0, 0));
    let source_row_count = usize_from_u32(end_source_y - first_source_y);
    let batch_capacity =
        resampling_row_batch_capacity(source_row_pixels, target_row_pixels, source_row_count);
    let source_batch_pixels = source_row_pixels
        .checked_mul(batch_capacity)
        .ok_or_else(|| io::Error::other("source row batch size overflowed"))?;
    let horizontal_batch_pixels = target_row_pixels
        .checked_mul(batch_capacity)
        .ok_or_else(|| io::Error::other("horizontal row batch size overflowed"))?;

    prepare_reused_buffer(
        &mut scratch.source_rows,
        source_batch_pixels,
        "source row batch",
    )?;
    prepare_reused_buffer(
        &mut scratch.horizontal_rows,
        horizontal_batch_pixels,
        "horizontal row batch",
    )?;
    prepare_zeroed_buffer(&mut scratch.output, output_bytes, "resampled output")?;

    let mut next_target_y = 0_u32;
    let mut source_rows_read = 0_usize;
    let source_x_start = horizontal_source_range.map_or(0, |range| range.0);
    let mut batch_start_y = first_source_y;
    while batch_start_y < end_source_y {
        if is_cancelled() {
            return Err(ResampleError::Cancelled);
        }
        let batch_row_count = usize_from_u32(end_source_y - batch_start_y).min(batch_capacity);
        let source_batch_len = source_row_pixels * batch_row_count;
        let horizontal_batch_len = target_row_pixels * batch_row_count;
        let source_batch = &mut scratch.source_rows[..source_batch_len];
        let horizontal_batch = &mut scratch.horizontal_rows[..horizontal_batch_len];

        if source_row_pixels != 0 {
            source_batch
                .par_chunks_exact_mut(source_row_pixels)
                .enumerate()
                .try_for_each(|(row_offset, destination)| {
                    let row_offset = u32::try_from(row_offset)
                        .expect("a resampling row batch contains at most u32 rows");
                    read_source_row(
                        batch_start_y + row_offset,
                        source_x_start,
                        destination.as_flattened_mut(),
                    )
                })?;
        }
        source_rows_read = source_rows_read.saturating_add(batch_row_count);
        if source_row_pixels == 0 {
            horizontal_batch.fill(RgbaF32::default());
        } else {
            source_batch
                .par_chunks_exact(source_row_pixels)
                .zip(horizontal_batch.par_chunks_exact_mut(target_row_pixels))
                .for_each(|(source_row, horizontal_row)| {
                    horizontal_convolution(
                        source_row,
                        horizontal_row,
                        &scratch.horizontal,
                        source_x_start,
                    );
                });
        }

        for (row_offset, horizontal_row) in
            horizontal_batch.chunks_exact(target_row_pixels).enumerate()
        {
            if is_cancelled() {
                return Err(ResampleError::Cancelled);
            }
            let row_offset = u32::try_from(row_offset)
                .expect("a resampling row batch contains at most u32 rows");
            let source_y = batch_start_y + row_offset;
            activate_output_rows(
                &scratch.vertical,
                target.height,
                target_row_pixels,
                source_y,
                &mut next_target_y,
                &mut scratch.active_rows,
                &mut scratch.free_row_buffers,
            )?;
            accumulate_output_row_weights(
                &scratch.vertical,
                &mut scratch.active_rows,
                source_y,
                horizontal_row,
            );
            flush_completed_output_rows(
                &mut scratch.active_rows,
                &mut scratch.free_row_buffers,
                source_y,
                &mut scratch.output,
                target.width,
            );
        }
        batch_start_y += u32::try_from(batch_row_count)
            .expect("a resampling row batch contains at most u32 rows");
    }
    skip_empty_output_rows(&scratch.vertical, target.height, &mut next_target_y);
    ensure_output_rows_complete(next_target_y, target.height, scratch.active_rows.len())?;
    let source_bytes = source_row_pixels
        .saturating_mul(source_rows_read)
        .saturating_mul(CANONICAL_BYTES_PER_PIXEL);
    Ok(source_bytes)
}

#[inline]
fn ensure_output_rows_complete(
    next_target_y: u32,
    target_height: u32,
    active_rows: usize,
) -> Result<(), ResampleError> {
    if next_target_y != target_height || active_rows != 0 {
        return Err(io::Error::other("Lanczos2 did not complete every target row").into());
    }
    Ok(())
}

pub(super) fn prepare_reused_buffer<T: Clone + Default>(
    values: &mut Vec<T>,
    length: usize,
    label: &str,
) -> Result<(), ResampleError> {
    let additional = length.saturating_sub(values.len());
    values
        .try_reserve_exact(additional)
        .map_err(|error| io::Error::other(format!("cannot allocate {label}: {error}")))?;
    values.resize(length, T::default());
    Ok(())
}

fn prepare_zeroed_buffer<T: Clone + Default>(
    values: &mut Vec<T>,
    length: usize,
    label: &str,
) -> Result<(), ResampleError> {
    prepare_reused_buffer(values, length, label)?;
    values.fill(T::default());
    Ok(())
}

#[inline(never)] // Avoid duplicating this sizeable inner convolution loop into its streaming caller.
fn horizontal_convolution(
    source: &[RgbaF32],
    destination: &mut [RgbaF32],
    contributions: &AxisContributions,
    source_origin: u32,
) {
    debug_assert_eq!(destination.len(), contributions.bounds.len());
    for (destination, bound) in destination.iter_mut().zip(&contributions.bounds) {
        let mut sum = [0.0_f32; 4];
        if bound.length != 0 {
            debug_assert!(bound.start >= source_origin);
            let start = usize_from_u32(bound.start - source_origin);
            let end = start + usize_from_u32(bound.length);
            let weights_end = bound.weights_start + usize_from_u32(bound.length);

            for (pixel, &weight) in source[start..end]
                .iter()
                .zip(&contributions.weights[bound.weights_start..weights_end])
            {
                sum[0] += pixel[0] * weight;
                sum[1] += pixel[1] * weight;
                sum[2] += pixel[2] * weight;
                sum[3] += pixel[3] * weight;
            }
        }
        *destination = sum;
    }
}

fn encode_resampled_row(samples: &mut [RgbaF32], output: &mut [u8], width: u32, row: u32) {
    debug_assert_eq!(samples.len(), usize_from_u32(width));
    let row_bytes = usize_from_u32(width) * CANONICAL_BYTES_PER_PIXEL;
    let destination = &mut output[usize_from_u32(row) * row_bytes..][..row_bytes];
    let mut bits = [0_u16; HALF_CONVERSION_CHUNK_SAMPLES];
    for (pixels, encoded) in samples
        .chunks_mut(HALF_CONVERSION_CHUNK_SAMPLES / 4)
        .zip(destination.chunks_mut(HALF_CONVERSION_CHUNK_SAMPLES * std::mem::size_of::<u16>()))
    {
        // Completed samples are recycled immediately, so sanitize them in
        // place before converting each cache-sized chunk.
        for rgba in pixels.iter_mut() {
            let alpha = if rgba[3].is_finite() {
                rgba[3].clamp(0.0, 1.0)
            } else {
                0.0
            };
            rgba[0] = sanitize_resampled_color(rgba[0], alpha);
            rgba[1] = sanitize_resampled_color(rgba[1], alpha);
            rgba[2] = sanitize_resampled_color(rgba[2], alpha);
            rgba[3] = alpha;
        }

        let samples = pixels.as_flattened_mut();
        let bits = &mut bits[..samples.len()];
        bits.reinterpret_cast_mut::<half::f16>()
            .convert_from_f32_slice(samples);
        for (&bits, encoded) in bits
            .iter()
            .zip(encoded.chunks_exact_mut(std::mem::size_of::<u16>()))
        {
            encoded.copy_from_slice(&bits.to_le_bytes());
        }
    }
}

#[inline]
fn sanitize_resampled_color(value: f32, alpha: f32) -> f32 {
    if alpha == 0.0 || !value.is_finite() {
        0.0
    } else {
        value.clamp(-MAX_FINITE_F16, MAX_FINITE_F16)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    use super::*;
    use crate::gpu::resampling::ViewportTarget;
    use crate::gpu::resampling::tests::centered_view;

    fn resample_test_pixels(
        dimensions: (u32, u32),
        source: &[f32],
        view: ViewportRequest,
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<Vec<[f32; 4]>, ResampleError> {
        let mut scratch = ResamplingScratch::default();
        resample_test_into_scratch(dimensions, source, view, &mut scratch, is_cancelled)?;
        Ok(scratch
            .output
            .chunks_exact(CANONICAL_BYTES_PER_PIXEL)
            .map(|rgba| {
                std::array::from_fn(|channel| {
                    let start = channel * 2;
                    half::f16::from_le_bytes([rgba[start], rgba[start + 1]]).to_f32()
                })
            })
            .collect())
    }

    fn resample_test_into_scratch(
        dimensions: (u32, u32),
        source: &[f32],
        view: ViewportRequest,
        scratch: &mut ResamplingScratch,
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<usize, ResampleError> {
        let source_row_samples = usize::try_from(dimensions.0).unwrap() * 4;
        resample_lanczos2_rows(
            dimensions,
            view,
            scratch,
            is_cancelled,
            |row, source_x, destination| {
                let start = usize::try_from(row).unwrap() * source_row_samples
                    + usize::try_from(source_x).unwrap() * 4;
                destination.copy_from_slice(&source[start..start + destination.len()]);
                Ok(())
            },
        )
    }

    fn scalar_resampled_row_encoding(samples: &[RgbaF32]) -> Vec<u8> {
        let mut encoded = Vec::with_capacity(samples.len() * CANONICAL_BYTES_PER_PIXEL);
        for rgba in samples {
            let alpha = if rgba[3].is_finite() {
                rgba[3].clamp(0.0, 1.0)
            } else {
                0.0
            };
            for value in [
                sanitize_resampled_color(rgba[0], alpha),
                sanitize_resampled_color(rgba[1], alpha),
                sanitize_resampled_color(rgba[2], alpha),
                alpha,
            ] {
                encoded.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        encoded
    }

    fn checkerboard_fixture(dimensions: (u32, u32)) -> Vec<f32> {
        let mut pixels =
            Vec::with_capacity(usize::try_from(dimensions.0 * dimensions.1).unwrap() * 4);
        for y in 0..dimensions.1 {
            for x in 0..dimensions.0 {
                let value = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
                pixels.extend_from_slice(&[value, value, value, 1.0]);
            }
        }
        pixels
    }

    #[test]
    fn lanczos2_has_expected_support_and_center() {
        assert!((lanczos2(0.0) - 1.0).abs() < f64::EPSILON);
        assert!(lanczos2(2.0).abs() < f64::EPSILON);
        assert!(lanczos2(-2.0).abs() < f64::EPSILON);
        assert!(lanczos2(0.5) > 0.0);
        assert!(lanczos2(1.5) < 0.0);
    }

    #[test]
    fn axis_weights_are_normalized_and_scale_aware() {
        let mut never_cancel = || false;
        let mut axis = AxisContributions::default();
        build_axis_contributions(100, 10, 50.0, 0.1, &mut axis, &mut never_cancel).unwrap();
        assert!(axis.bounds.iter().all(|bound| bound.length > 4));
        for bound in axis.bounds {
            let start = bound.weights_start;
            let end = start + usize::try_from(bound.length).unwrap();
            let sum: f32 = axis.weights[start..end].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "{sum}");
        }
    }

    #[test]
    fn coefficient_generation_is_cancellable() {
        let mut checks = 0;
        let mut axis = AxisContributions::default();
        let result =
            build_axis_contributions(100_000, 10_000, 50_000.0, 0.1, &mut axis, &mut || {
                checks += 1;
                checks > 2
            });
        assert!(matches!(result, Err(ResampleError::Cancelled)));
    }

    #[test]
    fn active_output_rows_stay_within_their_vertical_bounds() {
        let vertical = AxisContributions {
            bounds: vec![
                AxisBound {
                    start: 0,
                    length: 0,
                    weights_start: 0,
                },
                AxisBound {
                    start: 0,
                    length: 2,
                    weights_start: 0,
                },
                AxisBound {
                    start: 1,
                    length: 3,
                    weights_start: 0,
                },
                AxisBound {
                    start: 3,
                    length: 1,
                    weights_start: 0,
                },
                AxisBound {
                    start: 0,
                    length: 0,
                    weights_start: 0,
                },
            ],
            weights: Vec::new(),
        };
        let target_height = u32::try_from(vertical.bounds.len()).unwrap();
        let mut next_target_y = 0;
        let mut active = VecDeque::new();
        let mut free_row_buffers = Vec::new();
        let mut output = vec![0; usize_from_u32(target_height) * CANONICAL_BYTES_PER_PIXEL];

        for source_y in 0..4 {
            activate_output_rows(
                &vertical,
                target_height,
                1,
                source_y,
                &mut next_target_y,
                &mut active,
                &mut free_row_buffers,
            )
            .unwrap();
            for row in &active {
                let bound = vertical.bounds[usize_from_u32(row.target_y)];
                assert!(bound.start <= source_y && source_y < row.end_source_y);
            }
            flush_completed_output_rows(
                &mut active,
                &mut free_row_buffers,
                source_y,
                &mut output,
                1,
            );
        }

        assert_eq!(next_target_y, target_height);
        assert!(active.is_empty());
    }

    #[test]
    fn scratch_byte_accounting_includes_direct_allocations() {
        let mut scratch = ResamplingScratch::default();
        assert_eq!(scratch.allocated_bytes(), 0);
        prepare_reused_buffer(&mut scratch.output, 128, "test output").unwrap();
        prepare_reused_buffer(&mut scratch.upload_staging, 64, "test staging").unwrap();
        prepare_reused_buffer(&mut scratch.source_rows, 32, "test source rows").unwrap();
        prepare_reused_buffer(&mut scratch.horizontal_rows, 16, "test horizontal rows").unwrap();
        assert!(scratch.allocated_bytes() >= 192 + 48 * std::mem::size_of::<RgbaF32>());
    }

    #[test]
    fn upscale_row_buffers_are_included_in_scratch_accounting() {
        let dimensions = (3, 2);
        let source = [0.25, 0.5, 0.75, 1.0].repeat(3 * 2);
        let mut scratch = ResamplingScratch::default();
        resample_lanczos2_rows(
            dimensions,
            centered_view(dimensions, (96, 40), 20.0),
            &mut scratch,
            || false,
            |row, source_x, destination| {
                let start =
                    usize::try_from(row).unwrap() * 3 * 4 + usize::try_from(source_x).unwrap() * 4;
                destination.copy_from_slice(&source[start..start + destination.len()]);
                Ok(())
            },
        )
        .unwrap();

        let row_buffer_bytes = scratch
            .free_row_buffers
            .iter()
            .map(|row| allocation_bytes::<RgbaF32>(row.capacity()))
            .sum::<usize>();
        assert!(row_buffer_bytes > 0);
        assert!(scratch.allocated_bytes() >= scratch.output.capacity() + row_buffer_bytes);
    }

    #[test]
    fn constant_premultiplied_hdr_values_survive_resampling() {
        let expected = [0.75, -0.125, 1.5, 0.5];
        let source = expected.repeat(7 * 5);
        let output =
            resample_test_pixels((7, 5), &source, centered_view((7, 5), (3, 2), 0.4), || {
                false
            })
            .unwrap();
        assert_eq!(output.len(), 6);
        for actual in output {
            for channel in 0..4 {
                assert!(
                    (actual[channel] - expected[channel]).abs() < 0.001,
                    "channel {channel}: {} != {}",
                    actual[channel],
                    expected[channel],
                );
            }
        }
    }

    #[test]
    fn one_pixel_source_and_transparent_edge_resample_stably() {
        let source = [0.25, 0.5, 1.0, 0.75];
        let output =
            resample_test_pixels((1, 1), &source, centered_view((1, 1), (3, 3), 3.0), || {
                false
            })
            .unwrap();
        assert_eq!(output.len(), 9);
        for pixel in output {
            for channel in 0..4 {
                assert!((pixel[channel] - source[channel]).abs() < 0.001);
            }
        }

        let opaque_to_transparent = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let output = resample_test_pixels(
            (2, 1),
            &opaque_to_transparent,
            centered_view((2, 1), (1, 1), 0.5),
            || false,
        )
        .unwrap();
        for (actual, expected) in output[0].into_iter().zip([0.5, 0.0, 0.0, 0.5]) {
            assert!((actual - expected).abs() < 0.001);
        }
    }

    #[test]
    fn odd_dimension_linear_ramp_preserves_pixel_centers() {
        let mut source = Vec::new();
        for _y in 0..3 {
            for x in 0_u16..7 {
                let value = f32::from(x);
                source.extend_from_slice(&[value, value * 2.0, value * 0.5, 1.0]);
            }
        }
        let output =
            resample_test_pixels((7, 3), &source, centered_view((7, 3), (3, 1), 1.0), || {
                false
            })
            .unwrap();

        assert_eq!(output.len(), 3);
        for (pixel, x) in output.into_iter().zip([2.0_f32, 3.0, 4.0]) {
            for (actual, expected) in pixel.into_iter().zip([x, x * 2.0, x * 0.5, 1.0]) {
                assert!((actual - expected).abs() < 0.001);
            }
        }
    }

    #[test]
    fn one_pixel_checkerboard_minifies_to_phase_stable_gray() {
        let dimensions = (128, 128);
        let source = checkerboard_fixture(dimensions);
        for phase in [0.0, 0.25, 0.5, 0.75] {
            let output = resample_test_pixels(
                dimensions,
                &source,
                ViewportRequest {
                    target: ViewportTarget {
                        width: 8,
                        height: 8,
                    },
                    center_x: f64::from(dimensions.0) * 0.5 + phase,
                    center_y: f64::from(dimensions.1) * 0.5 + phase,
                    scale: 0.125,
                },
                || false,
            )
            .unwrap();
            for (index, rgba) in output.into_iter().enumerate() {
                for (channel, value) in rgba[..3].iter().copied().enumerate() {
                    assert!(
                        (value - 0.5).abs() < 0.01,
                        "phase {phase}, pixel {index}, channel {channel}: {value}"
                    );
                }
                assert!(
                    (rgba[3] - 1.0).abs() < 0.001,
                    "phase {phase}, pixel {index}, alpha: {}",
                    rgba[3]
                );
            }
        }
    }

    #[test]
    fn partial_final_row_batch_is_consumed() {
        let dimensions = (8, 100);
        let expected = [0.25, 0.5, 0.75, 1.0];
        let source = expected.repeat(8 * 100);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let output = pool
            .install(|| {
                resample_test_pixels(
                    dimensions,
                    &source,
                    centered_view(dimensions, (2, 25), 0.25),
                    || false,
                )
            })
            .unwrap();

        assert_eq!(output.len(), 2 * 25);
        for pixel in output {
            for (actual, expected) in pixel.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 0.001);
            }
        }
    }

    #[test]
    fn viewport_center_maps_to_the_same_source_pixel_centers_as_presentation() {
        let mut source = Vec::new();
        for x in 0..8 {
            source.extend_from_slice(&[f32::from(u16::try_from(x).unwrap()), 0.0, 0.0, 1.0]);
        }
        let output = resample_test_pixels(
            (8, 1),
            &source,
            ViewportRequest {
                target: ViewportTarget {
                    width: 4,
                    height: 1,
                },
                center_x: 4.0,
                center_y: 0.5,
                scale: 1.0,
            },
            || false,
        )
        .unwrap();
        let red: Vec<f32> = output.iter().map(|rgba| rgba[0]).collect();
        assert_eq!(red, [2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn output_policy_clamps_alpha_and_keeps_rgba16f_finite() {
        let mut samples = [
            [1.0, 2.0, 3.0, -1.0],
            [f32::INFINITY, -70_000.0, 70_000.0, 2.0],
            [f32::NAN, -0.0, f32::MIN_POSITIVE, 0.5],
            [0.333_251_95, -MAX_FINITE_F16, MAX_FINITE_F16, f32::NAN],
        ];
        let scalar_reference = scalar_resampled_row_encoding(&samples);
        let mut encoded = vec![0_u8; samples.len() * CANONICAL_BYTES_PER_PIXEL];
        encode_resampled_row(&mut samples, &mut encoded, 4, 0);
        assert_eq!(encoded, scalar_reference);
        let output: Vec<f32> = encoded
            .chunks_exact(2)
            .map(|value| half::f16::from_le_bytes([value[0], value[1]]).to_f32())
            .collect();
        assert_eq!(&output[..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&output[4..8], &[0.0, -MAX_FINITE_F16, MAX_FINITE_F16, 1.0]);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn chunked_output_encoding_matches_scalar_reference() {
        let mut samples: Vec<RgbaF32> = (0..HALF_CONVERSION_CHUNK_SAMPLES / 4 + 13)
            .map(|index| {
                let value = f32::from(u16::try_from(index).unwrap()) / 127.0;
                let alpha = f32::from(u8::try_from(index % 5).unwrap()) * 0.25;
                [value, -value, value * 0.5, alpha]
            })
            .collect();
        let scalar_reference = scalar_resampled_row_encoding(&samples);
        let mut encoded = vec![0_u8; samples.len() * CANONICAL_BYTES_PER_PIXEL];
        let width = u32::try_from(samples.len()).unwrap();

        encode_resampled_row(&mut samples, &mut encoded, width, 0);

        assert_eq!(encoded, scalar_reference);
    }

    #[test]
    fn row_batch_capacity_handles_zero_widths() {
        assert_eq!(resampling_row_batch_capacity(0, 0, 0), 0);
        assert_eq!(resampling_row_batch_capacity(0, 0, 1), 1);
    }

    #[test]
    fn source_scan_stops_after_cancellation() {
        let checks = Cell::new(0_u32);
        let rows_read = AtomicU32::new(0);
        let source = vec![1.0; 64 * 64 * 4];
        let mut scratch = ResamplingScratch::default();
        let maximum_rows_read = resampling_row_batch_capacity(64, 16, 64);
        let result = resample_lanczos2_rows(
            (64, 64),
            centered_view((64, 64), (16, 16), 0.25),
            &mut scratch,
            || {
                checks.set(checks.get() + 1);
                checks.get() > 8
            },
            |row, source_x, destination| {
                rows_read.fetch_add(1, Ordering::Relaxed);
                let row_samples = 64 * 4;
                let start = usize::try_from(row).unwrap() * row_samples
                    + usize::try_from(source_x).unwrap() * 4;
                destination.copy_from_slice(&source[start..start + destination.len()]);
                Ok(())
            },
        );
        assert!(matches!(result, Err(ResampleError::Cancelled)));
        let rows_read = usize::try_from(rows_read.load(Ordering::Relaxed)).unwrap();
        assert!((1..=maximum_rows_read).contains(&rows_read));
    }

    #[test]
    fn viewport_resampling_reads_only_the_required_source_region() {
        let source = vec![1.0; 100 * 100 * 4];
        let rows_read = AtomicU32::new(0);
        let smallest_start = AtomicU32::new(u32::MAX);
        let widest_row = AtomicUsize::new(0);
        let mut scratch = ResamplingScratch::default();
        let source_bytes = resample_lanczos2_rows(
            (100, 100),
            centered_view((100, 100), (10, 10), 2.0),
            &mut scratch,
            || false,
            |row, source_x, destination| {
                rows_read.fetch_add(1, Ordering::Relaxed);
                smallest_start.fetch_min(source_x, Ordering::Relaxed);
                widest_row.fetch_max(destination.len() / 4, Ordering::Relaxed);
                let start = usize::try_from(row).unwrap() * 100 * 4
                    + usize::try_from(source_x).unwrap() * 4;
                destination.copy_from_slice(&source[start..start + destination.len()]);
                Ok(())
            },
        )
        .unwrap();
        assert!(rows_read.load(Ordering::Relaxed) < 100);
        assert!(smallest_start.load(Ordering::Relaxed) > 0);
        assert!(widest_row.load(Ordering::Relaxed) < 100);
        assert!(source_bytes < 100 * 100 * CANONICAL_BYTES_PER_PIXEL);
    }

    #[test]
    fn viewport_pixels_outside_the_image_stay_transparent() {
        let source = [1.0, 1.0, 1.0, 1.0].repeat(2 * 2);
        let output =
            resample_test_pixels((2, 2), &source, centered_view((2, 2), (6, 2), 1.0), || {
                false
            })
            .unwrap();
        let alpha: Vec<f32> = output.iter().map(|rgba| rgba[3]).collect();
        assert_eq!(alpha, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0].repeat(2));
    }
}

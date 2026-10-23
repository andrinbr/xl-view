use std::collections::{HashSet, VecDeque};
use std::io;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use wgpu::TextureFormat;
use wgpu::util::DeviceExt;

use super::mip::{LinearMipGenerator, mip_level_count, rgba16f_texture_budget_bytes};
use super::upload::TextureUploadLayout;
use super::view::{ImageBounds, ViewTransform};
use super::{WorkerContext, lock_unpoisoned};
use crate::diagnostics::panic_payload_message;
use xl_view::decode::{DecodedTileStore, TILE_GUTTER, TILE_SIZE};
use xl_view::units::usize_from_u32;

const MISSING_TILE_SLOT: u32 = u32::MAX;

/// GPU tile residency state.
///
/// `mapping[logical_tile]` names its resident slot, or
/// [`MISSING_TILE_SLOT`]. `slot_tiles[slot] = Some(logical_tile)` reserves a
/// slot for either a pending load or a resident tile; it is resident only when
/// the reverse `mapping` entry points back to that slot. `last_used` has one
/// entry per slot. Generations let the worker reject stale jobs before their
/// reserved slots can become resident.
pub(super) struct TileCache {
    texture: wgpu::Texture,
    mapping_buffer: wgpu::Buffer,
    mapping: Vec<u32>,
    slot_tiles: Vec<Option<u32>>,
    last_used: Vec<u64>,
    epoch: u64,
    hits: u64,
    misses: u64,
    tile_columns: u32,
    tile_rows: u32,
    coarse_downsample: u32,
    desired_tiles: Vec<u32>,
    request_generation: u64,
    last_view_center: Option<(f64, f64)>,
    worker: Option<TileWorker>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TileJob {
    generation: u64,
    logical_tile: u32,
    slot: u32,
}

#[derive(Debug)]
enum TileWorkResult {
    Ready(TileJob),
    Cancelled(TileJob),
    Failed(TileJob, io::Error),
}

#[derive(Debug, Default)]
struct TileWorkerState {
    generation: u64,
    pending: VecDeque<TileJob>,
    shutdown: bool,
}

struct TileWorker {
    state: Arc<(Mutex<TileWorkerState>, Condvar)>,
    results: Receiver<TileWorkResult>,
    thread: Option<JoinHandle<()>>,
}

impl TileWorker {
    fn spawn(
        source: Arc<DecodedTileStore>,
        context: WorkerContext,
        texture: wgpu::Texture,
        mip_level_count: u32,
    ) -> Result<Self, io::Error> {
        let state = Arc::new((Mutex::new(TileWorkerState::default()), Condvar::new()));
        let worker_state = Arc::clone(&state);
        let (result_sender, results) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("xl-view-tile-cache".to_owned())
            .spawn(move || {
                run_tile_worker(
                    &source,
                    &context,
                    &texture,
                    mip_level_count,
                    &worker_state,
                    &result_sender,
                );
            })?;
        Ok(Self {
            state,
            results,
            thread: Some(thread),
        })
    }

    fn replace_pending(&self, generation: u64) -> Vec<TileJob> {
        let (state, wake) = self.state.as_ref();
        let mut state = lock_unpoisoned(state);
        let cancelled = replace_pending_jobs(&mut state, generation);
        wake.notify_one();
        cancelled
    }

    fn append(&self, generation: u64, jobs: Vec<TileJob>) -> Vec<TileJob> {
        if jobs.is_empty() {
            return Vec::new();
        }
        let (state, wake) = self.state.as_ref();
        let mut state = lock_unpoisoned(state);
        if state.shutdown || state.generation != generation {
            return jobs;
        }
        state.pending.extend(jobs);
        wake.notify_one();
        Vec::new()
    }
}

fn replace_pending_jobs(state: &mut TileWorkerState, generation: u64) -> Vec<TileJob> {
    state.generation = generation;
    state.pending.drain(..).collect()
}

impl Drop for TileWorker {
    fn drop(&mut self) {
        let (state, wake) = self.state.as_ref();
        {
            let mut state = lock_unpoisoned(state);
            state.shutdown = true;
            state.pending.clear();
            wake.notify_one();
        }
        if let Some(thread) = self.thread.take()
            && let Err(payload) = thread.join()
        {
            tracing::error!(
                panic = panic_payload_message(payload.as_ref()),
                "tile-cache worker panicked"
            );
        }
    }
}

fn run_tile_worker(
    source: &DecodedTileStore,
    context: &WorkerContext,
    texture: &wgpu::Texture,
    mip_level_count: u32,
    state: &Arc<(Mutex<TileWorkerState>, Condvar)>,
    results: &mpsc::Sender<TileWorkResult>,
) {
    let WorkerContext {
        device,
        queue,
        work_ready: notify_ready,
        submission_lock,
    } = context;
    let mip_generator = LinearMipGenerator::new(device);
    loop {
        let Some(job) = next_tile_job(state) else {
            return;
        };
        let pixels = source.read_tile_rgba16f(
            job.logical_tile % source.tile_columns(),
            job.logical_tile / source.tile_columns(),
        );
        let result = if tile_job_is_current(state, job.generation) {
            match pixels {
                Ok(pixels) => {
                    let _submission_guard = lock_unpoisoned(submission_lock);
                    if let Err(error) = upload_tile_layer(queue, texture, job.slot, &pixels) {
                        TileWorkResult::Failed(job, error)
                    } else {
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("asynchronous tile mip generation encoder"),
                            });
                        mip_generator.generate(
                            device,
                            &mut encoder,
                            texture,
                            mip_level_count,
                            &[job.slot],
                        );
                        queue.submit([encoder.finish()]);
                        TileWorkResult::Ready(job)
                    }
                }
                Err(error) => TileWorkResult::Failed(job, error),
            }
        } else {
            TileWorkResult::Cancelled(job)
        };
        if results.send(result).is_err() {
            return;
        }
        notify_ready();
    }
}

fn next_tile_job(state: &Arc<(Mutex<TileWorkerState>, Condvar)>) -> Option<TileJob> {
    let (state, wake) = state.as_ref();
    let mut state = lock_unpoisoned(state);
    loop {
        if state.shutdown {
            return None;
        }
        if let Some(job) = state.pending.pop_front() {
            return Some(job);
        }
        state = wake
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn tile_job_is_current(state: &Arc<(Mutex<TileWorkerState>, Condvar)>, generation: u64) -> bool {
    let (state, _) = state.as_ref();
    let state = lock_unpoisoned(state);
    !state.shutdown && state.generation == generation
}

impl TileCache {
    pub(super) fn fallback(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("unused canonical tile binding"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mapping = vec![MISSING_TILE_SLOT];
        let mapping_buffer = create_tile_mapping_buffer(device, &mapping);
        Self {
            texture,
            mapping_buffer,
            mapping,
            slot_tiles: vec![None],
            last_used: vec![0],
            epoch: 0,
            hits: 0,
            misses: 0,
            tile_columns: 0,
            tile_rows: 0,
            coarse_downsample: 1,
            desired_tiles: Vec::new(),
            request_generation: 0,
            last_view_center: None,
            worker: None,
        }
    }

    pub(super) fn active(
        context: &WorkerContext,
        source: Arc<DecodedTileStore>,
        maximum_tile_bytes: u64,
        viewport_dimensions: (u32, u32),
    ) -> Result<Self, io::Error> {
        let WorkerContext { device, queue, .. } = context;
        let logical_tiles = source
            .tile_columns()
            .checked_mul(source.tile_rows())
            .ok_or_else(|| io::Error::other("logical tile count overflowed"))?;
        let capacity = working_set_capacity(
            logical_tiles,
            viewport_dimensions,
            maximum_tile_bytes,
            device.limits().max_texture_array_layers,
        );
        if capacity == 0 {
            return Ok(Self::fallback(device));
        }
        let texture = create_tile_array_texture(device, capacity);
        // Presentation binds the complete array. Initialize it before the worker
        // starts so lazy zero-initialization cannot interleave with a tile upload.
        initialize_tile_array_texture(device, queue, &texture);
        let tile_columns = source.tile_columns();
        let tile_rows = source.tile_rows();
        let coarse_downsample = source.coarse_downsample();
        let mip_level_count = mip_level_count(tile_texture_extent(), tile_texture_extent());
        let worker = TileWorker::spawn(source, context.clone(), texture.clone(), mip_level_count)?;
        let mapping = vec![MISSING_TILE_SLOT; usize_from_u32(logical_tiles)];
        let mapping_buffer = create_tile_mapping_buffer(device, &mapping);
        Ok(Self {
            texture,
            mapping_buffer,
            mapping,
            slot_tiles: vec![None; usize_from_u32(capacity)],
            last_used: vec![0; usize_from_u32(capacity)],
            epoch: 0,
            hits: 0,
            misses: 0,
            tile_columns,
            tile_rows,
            coarse_downsample,
            desired_tiles: Vec::new(),
            request_generation: 0,
            last_view_center: None,
            worker: Some(worker),
        })
    }

    pub(super) fn is_active(&self) -> bool {
        self.tile_columns != 0 && self.tile_rows != 0
    }

    pub(super) fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub(super) fn mapping_buffer(&self) -> &wgpu::Buffer {
        &self.mapping_buffer
    }

    pub(super) fn hits(&self) -> u64 {
        self.hits
    }

    pub(super) fn misses(&self) -> u64 {
        self.misses
    }

    pub(super) fn tile_columns(&self) -> u32 {
        self.tile_columns
    }

    pub(super) fn capacity(&self) -> u32 {
        if self.is_active() {
            slot_u32(self.slot_tiles.len())
        } else {
            0
        }
    }

    pub(super) fn gpu_bytes(&self) -> u64 {
        if self.is_active() {
            rgba16f_texture_budget_bytes(
                tile_texture_extent(),
                tile_texture_extent(),
                mip_level_count(tile_texture_extent(), tile_texture_extent()),
                self.capacity(),
            )
        } else {
            0
        }
    }

    pub(super) fn status(&self) -> String {
        if self.is_active() {
            format!(
                "{} of {} working-set slots resident",
                self.slot_tiles
                    .iter()
                    .enumerate()
                    .filter(|(slot, tile)| {
                        tile.is_some_and(|tile| self.slot_is_resident(*slot, tile))
                    })
                    .count(),
                self.slot_tiles.len()
            )
        } else {
            "coarse-only".to_owned()
        }
    }

    fn slot_is_resident(&self, slot: usize, logical_tile: u32) -> bool {
        self.mapping[usize_from_u32(logical_tile)] == slot_u32(slot)
    }

    pub(super) fn should_sample(&self, view: Option<ViewTransform>) -> bool {
        self.is_active()
            && view.is_some_and(|view| {
                let maximum_tile_minification = self.coarse_downsample.min(TILE_GUTTER);
                view.scale() > 1.0 / f64::from(maximum_tile_minification)
            })
    }

    pub(super) fn request_view(&mut self, queue: &wgpu::Queue, view: ViewTransform) {
        if !self.is_active() {
            return;
        }
        let (next_desired, next_center) = if self.should_sample(Some(view)) {
            let bounds = view.visible_image_bounds();
            let (desired, center) = prioritized_tiles(
                bounds,
                self.tile_columns,
                self.tile_rows,
                self.last_view_center,
                self.slot_tiles.len(),
            );
            (desired, Some(center))
        } else {
            (Vec::new(), None)
        };
        self.last_view_center = next_center;
        self.epoch = self.epoch.wrapping_add(1);
        if same_tile_set(&self.desired_tiles, &next_desired) {
            self.touch_resident_tiles();
            return;
        }

        self.request_generation = self.request_generation.wrapping_add(1);
        let generation = self.request_generation;
        let cancelled = self
            .worker
            .as_ref()
            .map_or_else(Vec::new, |worker| worker.replace_pending(generation));
        for job in cancelled {
            self.release_loading_job(job);
        }
        self.desired_tiles = next_desired;
        self.touch_resident_tiles();
        self.schedule_missing(queue, generation);
    }

    fn touch_resident_tiles(&mut self) {
        for &logical_tile in &self.desired_tiles {
            let slot = self.mapping[usize_from_u32(logical_tile)];
            if slot != MISSING_TILE_SLOT {
                self.hits = self.hits.saturating_add(1);
                self.last_used[usize_from_u32(slot)] = self.epoch;
            }
        }
    }

    pub(super) fn process_completions(&mut self, queue: &wgpu::Queue) -> Result<bool, io::Error> {
        let mut visual_change = false;
        let mut mapping_changed = false;
        loop {
            let result = match self.worker.as_ref().map(|worker| worker.results.try_recv()) {
                Some(Ok(result)) => result,
                Some(Err(TryRecvError::Empty)) | None => break,
                Some(Err(TryRecvError::Disconnected)) => {
                    return Err(io::Error::other(
                        "the tile-cache worker stopped unexpectedly",
                    ));
                }
            };
            match result {
                TileWorkResult::Ready(job) => {
                    let slot = usize_from_u32(job.slot);
                    if self.slot_tiles.get(slot) == Some(&Some(job.logical_tile)) {
                        self.mapping[usize_from_u32(job.logical_tile)] = job.slot;
                        self.last_used[slot] = self.epoch;
                        mapping_changed = true;
                        visual_change = true;
                    }
                }
                TileWorkResult::Cancelled(job) => self.release_loading_job(job),
                TileWorkResult::Failed(job, error) => {
                    self.release_loading_job(job);
                    if job.generation == self.request_generation {
                        return Err(error);
                    }
                }
            }
        }
        if mapping_changed {
            self.write_mapping(queue);
        }
        self.schedule_missing(queue, self.request_generation);
        Ok(visual_change)
    }

    fn schedule_missing(&mut self, queue: &wgpu::Queue, generation: u64) {
        let desired_set: HashSet<u32> = self.desired_tiles.iter().copied().collect();
        let mut jobs = Vec::new();
        let mut mapping_changed = false;
        for logical_tile in self.desired_tiles.clone() {
            if self.mapping[usize_from_u32(logical_tile)] != MISSING_TILE_SLOT
                || self.slot_tiles.contains(&Some(logical_tile))
            {
                continue;
            }
            let Some(slot) = self
                .slot_tiles
                .iter()
                .position(Option::is_none)
                .or_else(|| {
                    self.slot_tiles
                        .iter()
                        .enumerate()
                        .filter(|(slot, tile)| {
                            tile.is_some_and(|tile| {
                                self.slot_is_resident(*slot, tile) && !desired_set.contains(&tile)
                            })
                        })
                        .min_by_key(|(slot, _)| self.last_used[*slot])
                        .map(|(slot, _)| slot)
                })
            else {
                break;
            };
            if let Some(evicted) = self.slot_tiles[slot] {
                self.mapping[usize_from_u32(evicted)] = MISSING_TILE_SLOT;
                mapping_changed = true;
            }
            self.slot_tiles[slot] = Some(logical_tile);
            self.last_used[slot] = self.epoch;
            self.misses = self.misses.saturating_add(1);
            jobs.push(TileJob {
                generation,
                logical_tile,
                slot: slot_u32(slot),
            });
        }
        if mapping_changed {
            self.write_mapping(queue);
        }
        let rejected = match self.worker.as_ref() {
            Some(worker) => worker.append(generation, jobs),
            None => jobs,
        };
        for job in rejected {
            self.release_loading_job(job);
        }
    }

    fn release_loading_job(&mut self, job: TileJob) {
        let slot = usize_from_u32(job.slot);
        if self.slot_tiles.get(slot) == Some(&Some(job.logical_tile))
            && self.mapping[usize_from_u32(job.logical_tile)] == MISSING_TILE_SLOT
        {
            self.slot_tiles[slot] = None;
        }
    }

    fn write_mapping(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.mapping_buffer, 0, bytemuck::cast_slice(&self.mapping));
    }
}

// The slot table is allocated from a u32 capacity, so slot counts and indices
// always fit.
fn slot_u32(slot: usize) -> u32 {
    u32::try_from(slot).expect("tile-cache slots are bounded by a u32 capacity")
}

pub(super) fn maximum_capacity(
    logical_tiles: u32,
    maximum_tile_bytes: u64,
    maximum_array_layers: u32,
) -> u32 {
    let maximum = logical_tiles.min(maximum_array_layers);
    let mut lower = 0_u32;
    let mut upper = maximum;
    while lower < upper {
        let candidate = lower + (upper - lower).div_ceil(2);
        let required = rgba16f_texture_budget_bytes(
            tile_texture_extent(),
            tile_texture_extent(),
            mip_level_count(tile_texture_extent(), tile_texture_extent()),
            candidate,
        );
        if required <= maximum_tile_bytes {
            lower = candidate;
        } else {
            upper = candidate - 1;
        }
    }
    lower
}

/// Holds a 1:1 viewport at the worst tile alignment, plus one
/// prefetch tile on every side.
fn interactive_working_set_tiles((viewport_width, viewport_height): (u32, u32)) -> u32 {
    if viewport_width == 0 || viewport_height == 0 {
        return 0;
    }
    let columns = viewport_width.div_ceil(TILE_SIZE).saturating_add(3);
    let rows = viewport_height.div_ceil(TILE_SIZE).saturating_add(3);
    columns.saturating_mul(rows)
}

pub(super) fn working_set_capacity(
    logical_tiles: u32,
    viewport_dimensions: (u32, u32),
    maximum_tile_bytes: u64,
    maximum_array_layers: u32,
) -> u32 {
    maximum_capacity(
        logical_tiles.min(interactive_working_set_tiles(viewport_dimensions)),
        maximum_tile_bytes,
        maximum_array_layers,
    )
}

fn same_tile_set(left: &[u32], right: &[u32]) -> bool {
    left.len() == right.len() && left.iter().all(|tile| right.contains(tile))
}

#[derive(Clone, Copy, Debug)]
struct TileWindow {
    first_x: u32,
    first_y: u32,
    last_x: u32,
    last_y: u32,
}

impl TileWindow {
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Bounds are clamped to the non-empty u32 tile grid immediately after conversion.
    fn visible(bounds: ImageBounds, columns: u32, rows: u32) -> Self {
        let ImageBounds {
            left,
            top,
            right,
            bottom,
        } = bounds;
        let tile_size = f64::from(TILE_SIZE);
        Self {
            first_x: ((left / tile_size).floor() as u32).min(columns - 1),
            first_y: ((top / tile_size).floor() as u32).min(rows - 1),
            last_x: ((right / tile_size).floor() as u32).min(columns - 1),
            last_y: ((bottom / tile_size).floor() as u32).min(rows - 1),
        }
    }

    fn expanded_for_prefetch(self, direction: (f64, f64), columns: u32, rows: u32) -> Self {
        let mut expanded = Self {
            first_x: self.first_x.saturating_sub(1),
            first_y: self.first_y.saturating_sub(1),
            last_x: self.last_x.saturating_add(1).min(columns - 1),
            last_y: self.last_y.saturating_add(1).min(rows - 1),
        };
        if direction.0 > 0.0 {
            expanded.last_x = expanded.last_x.saturating_add(1).min(columns - 1);
        } else if direction.0 < 0.0 {
            expanded.first_x = expanded.first_x.saturating_sub(1);
        }
        if direction.1 > 0.0 {
            expanded.last_y = expanded.last_y.saturating_add(1).min(rows - 1);
        } else if direction.1 < 0.0 {
            expanded.first_y = expanded.first_y.saturating_sub(1);
        }
        expanded
    }

    fn contains(self, x: u32, y: u32) -> bool {
        (self.first_x..=self.last_x).contains(&x) && (self.first_y..=self.last_y).contains(&y)
    }
}

fn tile_is_ahead(x: u32, y: u32, visible: TileWindow, direction: (f64, f64)) -> bool {
    (direction.0 > 0.0 && x > visible.last_x)
        || (direction.0 < 0.0 && x < visible.first_x)
        || (direction.1 > 0.0 && y > visible.last_y)
        || (direction.1 < 0.0 && y < visible.first_y)
}

fn tile_priority(x: u32, y: u32, visible: TileWindow, direction: (f64, f64)) -> u8 {
    if visible.contains(x, y) {
        0
    } else if tile_is_ahead(x, y, visible, direction) {
        1
    } else {
        2
    }
}

fn prioritized_tiles(
    bounds: ImageBounds,
    columns: u32,
    rows: u32,
    previous_center: Option<(f64, f64)>,
    capacity: usize,
) -> (Vec<u32>, (f64, f64)) {
    let tile_size = f64::from(TILE_SIZE);
    let visible = TileWindow::visible(bounds, columns, rows);
    let center = (
        (bounds.left + bounds.right) * 0.5,
        (bounds.top + bounds.bottom) * 0.5,
    );
    let direction = previous_center.map_or((0.0, 0.0), |previous| {
        (center.0 - previous.0, center.1 - previous.1)
    });
    let desired_window = visible.expanded_for_prefetch(direction, columns, rows);
    let center_x = center.0 / tile_size;
    let center_y = center.1 / tile_size;
    let mut desired = Vec::new();
    for tile_y in desired_window.first_y..=desired_window.last_y {
        for tile_x in desired_window.first_x..=desired_window.last_x {
            let priority = tile_priority(tile_x, tile_y, visible, direction);
            let dx = f64::from(tile_x) + 0.5 - center_x;
            let dy = f64::from(tile_y) + 0.5 - center_y;
            desired.push((tile_y * columns + tile_x, priority, dx.mul_add(dx, dy * dy)));
        }
    }
    desired.sort_by(|left, right| {
        left.1
            .cmp(&right.1)
            .then_with(|| left.2.total_cmp(&right.2))
    });
    desired.truncate(capacity);
    (
        desired.into_iter().map(|(tile, _, _)| tile).collect(),
        center,
    )
}

pub(super) fn tile_texture_extent() -> u32 {
    TILE_SIZE + TILE_GUTTER * 2
}

fn create_tile_array_texture(device: &wgpu::Device, layers: u32) -> wgpu::Texture {
    let extent = tile_texture_extent();
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("canonical high-resolution tile cache"),
        size: wgpu::Extent3d {
            width: extent,
            height: extent,
            depth_or_array_layers: layers,
        },
        mip_level_count: mip_level_count(extent, extent),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn initialize_tile_array_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("canonical tile-cache initialization encoder"),
    });
    encoder.clear_texture(texture, &wgpu::ImageSubresourceRange::default());
    queue.submit([encoder.finish()]);
}

fn create_tile_mapping_buffer(device: &wgpu::Device, mapping: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("logical-to-resident tile mapping"),
        contents: bytemuck::cast_slice(mapping),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn upload_tile_layer(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layer: u32,
    pixels: &[u8],
) -> Result<(), io::Error> {
    let extent = tile_texture_extent();
    let upload_layout = TextureUploadLayout::rgba16f(extent, extent, "tile upload")?;
    upload_layout.validate_source_len(pixels.len())?;
    let mut staging = upload_layout.allocate_staging();
    for stripe in upload_layout.stripes() {
        upload_layout.copy_stripe(pixels, stripe, &mut staging);
        upload_layout.write_stripe(queue, texture, layer, &staging, stripe);
    }
    Ok(())
}

#[cfg(test)]
impl TileCache {
    /// Builds a fully resident single-row cache from canonical pixels for
    /// offscreen rendering tests.
    pub(super) fn from_canonical_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pixels: &[f32],
        width: u32,
        height: u32,
    ) -> Self {
        use super::mip::{encode_premultiplied_rgba16f_row, generate_linear_mips_for_layers};

        assert!(height <= TILE_SIZE);
        assert_eq!(pixels.len(), usize::try_from(width * height * 4).unwrap());

        let extent = usize::try_from(tile_texture_extent()).unwrap();
        let width = usize::try_from(width).unwrap();
        let height = usize::try_from(height).unwrap();
        let gutter = usize::try_from(TILE_GUTTER).unwrap();
        let tile_size = usize::try_from(TILE_SIZE).unwrap();
        let columns = width.div_ceil(tile_size);
        let mut source_row = vec![0.0_f32; extent * 4];
        let mut encoded_pixels = vec![0_u8; extent * extent * 8];
        let mapping = (0..u32::try_from(columns).unwrap()).collect::<Vec<_>>();
        let cache = Self {
            texture: create_tile_array_texture(device, u32::try_from(columns).unwrap()),
            mapping_buffer: create_tile_mapping_buffer(device, &mapping),
            mapping: mapping.clone(),
            slot_tiles: (0..u32::try_from(columns).unwrap()).map(Some).collect(),
            last_used: vec![0; columns],
            epoch: 0,
            hits: 0,
            misses: 0,
            tile_columns: u32::try_from(columns).unwrap(),
            tile_rows: 1,
            coarse_downsample: 1,
            desired_tiles: Vec::new(),
            request_generation: 0,
            last_view_center: None,
            worker: None,
        };
        for column in 0..columns {
            let origin_x = column * tile_size;
            for tile_y in 0..extent {
                let source_y = tile_y.saturating_sub(gutter).min(height - 1);
                for tile_x in 0..extent {
                    let source_x = origin_x
                        .saturating_add(tile_x)
                        .saturating_sub(gutter)
                        .min(width - 1);
                    let source_offset = (source_y * width + source_x) * 4;
                    source_row[tile_x * 4..tile_x * 4 + 4]
                        .copy_from_slice(&pixels[source_offset..source_offset + 4]);
                }
                encode_premultiplied_rgba16f_row(
                    &source_row,
                    &mut encoded_pixels[tile_y * extent * 8..(tile_y + 1) * extent * 8],
                );
            }
            upload_tile_layer(
                queue,
                &cache.texture,
                u32::try_from(column).unwrap(),
                &encoded_pixels,
            )
            .unwrap();
        }
        let mut mip_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen tiled fixture mip encoder"),
        });
        generate_linear_mips_for_layers(
            device,
            &mut mip_encoder,
            &cache.texture,
            mip_level_count(tile_texture_extent(), tile_texture_extent()),
            &mapping,
        );
        queue.submit([mip_encoder.finish()]);
        cache
    }
}

#[cfg(test)]
mod tests;

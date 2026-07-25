use std::collections::{HashSet, VecDeque};
use std::io;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use wgpu::TextureFormat;
use wgpu::util::DeviceExt;

use super::WorkReadyNotifier;
use super::mip::{LinearMipGenerator, mip_level_count, rgba16f_texture_budget_bytes};
use super::upload::TextureUploadLayout;
use super::view::ViewTransform;
use crate::units::usize_from_u32;
use xl_view::decode::{DecodedTileStore, TILE_GUTTER, TILE_SIZE};

pub(super) const MISSING_TILE_SLOT: u32 = u32::MAX;

/// GPU tile residency state.
///
/// `mapping[logical_tile]` names its resident slot, or
/// [`MISSING_TILE_SLOT`]. `slot_tiles[slot] = Some(logical_tile)` reserves a
/// slot for either a pending load or a resident tile; it is resident only when
/// the reverse `mapping` entry points back to that slot. `last_used` has one
/// entry per slot. Generations let the worker reject stale jobs before their
/// reserved slots can become resident.
pub(super) struct TileCache {
    pub(super) texture: wgpu::Texture,
    pub(super) mapping_buffer: wgpu::Buffer,
    pub(super) mapping: Vec<u32>,
    pub(super) slot_tiles: Vec<Option<u32>>,
    pub(super) last_used: Vec<u64>,
    pub(super) epoch: u64,
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) tile_columns: u32,
    pub(super) tile_rows: u32,
    pub(super) coarse_downsample: u32,
    pub(super) desired_tiles: Vec<u32>,
    pub(super) request_generation: u64,
    pub(super) last_view_center: Option<(f64, f64)>,
    pub(super) worker: Option<TileWorker>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TileJob {
    pub(super) generation: u64,
    pub(super) logical_tile: u32,
    pub(super) slot: u32,
}

#[derive(Debug)]
enum TileWorkResult {
    Ready(TileJob),
    Cancelled(TileJob),
    Failed(TileJob, io::Error),
}

#[derive(Debug, Default)]
pub(super) struct TileWorkerState {
    pub(super) generation: u64,
    pub(super) pending: VecDeque<TileJob>,
    pub(super) shutdown: bool,
}

pub(super) struct TileWorker {
    state: Arc<(Mutex<TileWorkerState>, Condvar)>,
    results: Receiver<TileWorkResult>,
    thread: Option<JoinHandle<()>>,
}

impl TileWorker {
    fn spawn(
        source: Arc<DecodedTileStore>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        texture: wgpu::Texture,
        mip_level_count: u32,
        notify_ready: WorkReadyNotifier,
        submission_lock: Arc<Mutex<()>>,
    ) -> Result<Self, io::Error> {
        let state = Arc::new((Mutex::new(TileWorkerState::default()), Condvar::new()));
        let worker_state = Arc::clone(&state);
        let (result_sender, results) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("xl-view-tile-cache".to_owned())
            .spawn(move || {
                run_tile_worker(
                    &source,
                    &device,
                    &queue,
                    &texture,
                    mip_level_count,
                    &worker_state,
                    &result_sender,
                    notify_ready.as_ref(),
                    &submission_lock,
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
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let cancelled = replace_pending_jobs(&mut state, generation);
        wake.notify_one();
        cancelled
    }

    fn append(&self, generation: u64, jobs: Vec<TileJob>) -> Vec<TileJob> {
        if jobs.is_empty() {
            return Vec::new();
        }
        let (state, wake) = self.state.as_ref();
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.shutdown || state.generation != generation {
            return jobs;
        }
        state.pending.extend(jobs);
        wake.notify_one();
        Vec::new()
    }
}

pub(super) fn replace_pending_jobs(state: &mut TileWorkerState, generation: u64) -> Vec<TileJob> {
    state.generation = generation;
    state.pending.drain(..).collect()
}

impl Drop for TileWorker {
    fn drop(&mut self) {
        let (state, wake) = self.state.as_ref();
        {
            let mut state = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.shutdown = true;
            state.pending.clear();
            wake.notify_one();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[allow(clippy::too_many_arguments)] // Worker dependencies stay explicit across this single thread boundary.
fn run_tile_worker(
    source: &DecodedTileStore,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level_count: u32,
    state: &Arc<(Mutex<TileWorkerState>, Condvar)>,
    results: &mpsc::Sender<TileWorkResult>,
    notify_ready: &(dyn Fn() + Send + Sync),
    submission_lock: &Mutex<()>,
) {
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
                    let _submission_guard = submission_lock
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    let state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: Arc<DecodedTileStore>,
        maximum_tile_bytes: u64,
        viewport_dimensions: (u32, u32),
        notify_ready: WorkReadyNotifier,
        submission_lock: Arc<Mutex<()>>,
    ) -> Result<Self, io::Error> {
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
        let worker = TileWorker::spawn(
            source,
            device.clone(),
            queue.clone(),
            texture.clone(),
            mip_level_count,
            notify_ready,
            submission_lock,
        )?;
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

    pub(super) fn tile_columns(&self) -> u32 {
        self.tile_columns
    }

    pub(super) fn capacity(&self) -> u32 {
        if self.is_active() {
            u32::try_from(self.slot_tiles.len()).unwrap_or(u32::MAX)
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
        self.mapping[usize_from_u32(logical_tile)]
            == u32::try_from(slot).expect("tile-cache slots are bounded by a u32 capacity")
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
                slot: u32::try_from(slot).expect("tile-cache slots are bounded by a u32 capacity"),
            });
        }
        if mapping_changed {
            self.write_mapping(queue);
        }
        let rejected = self
            .worker
            .as_ref()
            .map_or(jobs.clone(), |worker| worker.append(generation, jobs));
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
        queue.write_buffer(
            &self.mapping_buffer,
            0,
            self.mapping
                .iter()
                .flat_map(|slot| slot.to_ne_bytes())
                .collect::<Vec<_>>()
                .as_slice(),
        );
    }
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
pub(super) fn interactive_working_set_tiles((viewport_width, viewport_height): (u32, u32)) -> u32 {
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

pub(super) fn same_tile_set(left: &[u32], right: &[u32]) -> bool {
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
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Bounds are clamped to the non-empty u32 tile grid immediately after conversion.
    fn visible((left, top, right, bottom): (f64, f64, f64, f64), columns: u32, rows: u32) -> Self {
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

pub(super) fn prioritized_tiles(
    (left, top, right, bottom): (f64, f64, f64, f64),
    columns: u32,
    rows: u32,
    previous_center: Option<(f64, f64)>,
    capacity: usize,
) -> (Vec<u32>, (f64, f64)) {
    let tile_size = f64::from(TILE_SIZE);
    let visible = TileWindow::visible((left, top, right, bottom), columns, rows);
    let center = ((left + right) * 0.5, (top + bottom) * 0.5);
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

pub(super) fn create_tile_array_texture(device: &wgpu::Device, layers: u32) -> wgpu::Texture {
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

pub(super) fn create_tile_mapping_buffer(device: &wgpu::Device, mapping: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("logical-to-resident tile mapping"),
        contents: mapping
            .iter()
            .flat_map(|slot| slot.to_ne_bytes())
            .collect::<Vec<_>>()
            .as_slice(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

pub(super) fn upload_tile_layer(
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
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: stripe.first_row(),
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            upload_layout.stripe_data(&staging, stripe),
            upload_layout.copy_buffer_layout(stripe),
            upload_layout.copy_extent(stripe),
        );
    }
    Ok(())
}

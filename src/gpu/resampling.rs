use std::collections::VecDeque;
use std::io;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use half::prelude::{HalfBitsSliceExt, HalfFloatSliceExt};
use rayon::prelude::*;
use wgpu::TextureFormat;
use xl_view::decode::{CANONICAL_BYTES_PER_PIXEL, DecodedTileStore, HALF_CONVERSION_CHUNK_SAMPLES};

use super::mip::rgba16f_texture_budget_bytes;
use super::upload::TextureUploadLayout;
use super::{WorkReadyNotifier, panic_payload_message};
use crate::units::{bytes_to_mib, u64_from_usize, usize_from_u32};

const RESAMPLING_DEBOUNCE: Duration = Duration::from_millis(50);
// Small batches expose row-independent work without delaying cancellation;
// the byte cap prevents unusually wide images from multiplying scratch use.
const RESAMPLING_ROWS_PER_WORKER: usize = 2;
const MAX_RESAMPLING_ROW_BATCH_BYTES: usize = 32 * 1024 * 1024;
const MAX_COEFFICIENT_BYTES: usize = 64 * 1024 * 1024;
const MAX_FINITE_F16: f32 = half::f16::MAX.to_f32_const();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ViewportTarget {
    pub(super) width: u32,
    pub(super) height: u32,
}

impl ViewportTarget {
    const FALLBACK: Self = Self {
        width: 1,
        height: 1,
    };

    fn gpu_budget_bytes(self) -> u64 {
        rgba16f_texture_budget_bytes(self.width, self.height, 1, 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ViewportRequest {
    target: ViewportTarget,
    center_x: f64,
    center_y: f64,
    scale: f64,
}

impl ViewportRequest {
    fn new(target: ViewportTarget, center_x: f64, center_y: f64, scale: f64) -> Option<Self> {
        if !center_x.is_finite() || !center_y.is_finite() || !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        Some(Self {
            target,
            center_x,
            center_y,
            scale,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ResamplingJob {
    generation: u64,
    view: ViewportRequest,
    deadline: Instant,
}

struct ResampledViewport {
    job: ResamplingJob,
    texture: wgpu::Texture,
    elapsed: Duration,
    source_bytes: usize,
}

enum ResamplingWorkResult {
    Ready(ResampledViewport),
    Cancelled,
    Failed(ResamplingJob, io::Error),
}

#[derive(Debug, Default)]
struct ResamplingWorkerState {
    generation: u64,
    pending: Option<ResamplingJob>,
    scratch_peak_bytes: usize,
    shutdown: bool,
}

struct ResamplingWorker {
    state: Arc<(Mutex<ResamplingWorkerState>, Condvar)>,
    results: Receiver<ResamplingWorkResult>,
    thread: Option<JoinHandle<()>>,
}

impl ResamplingWorker {
    fn spawn(
        source: Arc<DecodedTileStore>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        notify_ready: WorkReadyNotifier,
        submission_lock: Arc<Mutex<()>>,
    ) -> Result<Self, io::Error> {
        let state = Arc::new((Mutex::new(ResamplingWorkerState::default()), Condvar::new()));
        let worker_state = Arc::clone(&state);
        let (result_sender, results) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("xl-view-viewport-resample".to_owned())
            .spawn(move || {
                run_resampling_worker(
                    &source,
                    &device,
                    &queue,
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

    fn scratch_peak_bytes(&self) -> usize {
        let (state, _) = self.state.as_ref();
        state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .scratch_peak_bytes
    }

    fn replace(&self, generation: u64, view: Option<ViewportRequest>) {
        let (state, wake) = self.state.as_ref();
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.generation = generation;
        state.pending = view.map(|view| ResamplingJob {
            generation,
            view,
            deadline: Instant::now() + RESAMPLING_DEBOUNCE,
        });
        wake.notify_one();
    }
}

impl Drop for ResamplingWorker {
    fn drop(&mut self) {
        let (state, wake) = self.state.as_ref();
        {
            let mut state = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.shutdown = true;
            state.pending = None;
            state.generation = state.generation.wrapping_add(1);
            wake.notify_one();
        }
        if let Some(thread) = self.thread.take()
            && let Err(payload) = thread.join()
        {
            tracing::error!(
                panic = panic_payload_message(payload.as_ref()),
                "viewport resampling worker panicked"
            );
        }
    }
}

enum ResamplerLifecycle {
    Unavailable,
    Available(AvailableResampler),
}

struct AvailableResampler {
    worker: ResamplingWorker,
    generation: u64,
    phase: ResamplerPhase,
}

enum ResamplerPhase {
    Inactive,
    // Pending work uses the one-pixel fallback binding.
    Pending(ViewportRequest),
    // Active work owns the completed texture represented by `ViewportResampler::texture`.
    Active(ViewportRequest),
}

impl ResamplerPhase {
    fn desired(&self) -> Option<ViewportRequest> {
        match self {
            Self::Pending(request) | Self::Active(request) => Some(*request),
            Self::Inactive => None,
        }
    }

    fn active_target(&self) -> Option<ViewportTarget> {
        match self {
            Self::Active(request) => Some(request.target),
            Self::Inactive | Self::Pending(_) => None,
        }
    }

    fn is_active(&self) -> bool {
        matches!(self, Self::Active(_))
    }

    fn replace_request(&mut self, desired: Option<ViewportRequest>) -> bool {
        let binding_changed = self.active_target().is_some();
        *self = desired.map_or(Self::Inactive, Self::Pending);
        binding_changed
    }
}

impl ResamplerLifecycle {
    fn worker_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }

    fn desired(&self) -> Option<ViewportRequest> {
        match self {
            Self::Unavailable => None,
            Self::Available(available) => available.phase.desired(),
        }
    }

    fn active_target(&self) -> Option<ViewportTarget> {
        match self {
            Self::Unavailable => None,
            Self::Available(available) => available.phase.active_target(),
        }
    }

    fn is_active(&self) -> bool {
        matches!(self, Self::Available(available) if available.phase.is_active())
    }

    fn is_current(&self, job: ResamplingJob) -> bool {
        match self {
            Self::Available(available) => {
                job.generation == available.generation
                    && matches!(
                        &available.phase,
                        ResamplerPhase::Pending(request) if *request == job.view
                    )
            }
            Self::Unavailable => false,
        }
    }

    fn replace_request(&mut self, desired: Option<ViewportRequest>) -> bool {
        let Self::Available(available) = self else {
            return false;
        };
        let binding_changed = available.phase.replace_request(desired);
        available.generation = available.generation.wrapping_add(1);
        available.worker.replace(available.generation, desired);
        binding_changed
    }

    fn mark_ready(&mut self, request: ViewportRequest) {
        if let Self::Available(available) = self {
            available.phase = ResamplerPhase::Active(request);
        }
    }

    fn mark_failed(&mut self) {
        if let Self::Available(available) = self {
            available.phase = ResamplerPhase::Inactive;
        }
    }

    fn disconnect(&mut self) -> bool {
        let binding_changed = self.active_target().is_some();
        *self = Self::Unavailable;
        binding_changed
    }

    #[cfg(test)]
    fn current_request(&self) -> Option<(u64, ViewportRequest)> {
        let Self::Available(available) = self else {
            return None;
        };
        available
            .phase
            .desired()
            .map(|request| (available.generation, request))
    }
}

pub(super) struct ViewportResampler {
    // Presentation always needs a valid binding; the lifecycle says whether this is
    // the fallback or the active resampled viewport.
    texture: wgpu::Texture,
    lifecycle: ResamplerLifecycle,
    memory_limit_bytes: u64,
    requests: u64,
    discarded_results: u64,
    completions: u64,
    last_elapsed: Option<Duration>,
    last_source_bytes: usize,
}

impl ViewportResampler {
    pub(super) fn fallback(device: &wgpu::Device, memory_limit_bytes: u64) -> Self {
        Self {
            texture: create_resampled_texture(device, ViewportTarget::FALLBACK),
            lifecycle: ResamplerLifecycle::Unavailable,
            memory_limit_bytes,
            requests: 0,
            discarded_results: 0,
            completions: 0,
            last_elapsed: None,
            last_source_bytes: 0,
        }
    }

    pub(super) fn active(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: Arc<DecodedTileStore>,
        memory_limit_bytes: u64,
        notify_ready: WorkReadyNotifier,
        submission_lock: Arc<Mutex<()>>,
    ) -> Self {
        let mut resampler = Self::fallback(device, memory_limit_bytes);
        match ResamplingWorker::spawn(
            source,
            device.clone(),
            queue.clone(),
            notify_ready,
            submission_lock,
        ) {
            Ok(worker) => {
                resampler.lifecycle = ResamplerLifecycle::Available(AvailableResampler {
                    worker,
                    generation: 0,
                    phase: ResamplerPhase::Inactive,
                });
            }
            Err(error) => {
                tracing::warn!(%error, "viewport resampling worker is unavailable");
            }
        }
        resampler
    }

    pub(super) fn is_active(&self) -> bool {
        self.lifecycle.is_active()
    }

    pub(super) fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub(super) fn gpu_bytes(&self) -> u64 {
        self.lifecycle
            .active_target()
            .map_or(0, ViewportTarget::gpu_budget_bytes)
    }

    pub(super) fn scratch_peak_bytes(&self) -> u64 {
        match &self.lifecycle {
            ResamplerLifecycle::Unavailable => 0,
            ResamplerLifecycle::Available(available) => {
                u64_from_usize(available.worker.scratch_peak_bytes())
            }
        }
    }

    pub(super) fn set_memory_limit_bytes(&mut self, memory_limit_bytes: u64) {
        self.memory_limit_bytes = memory_limit_bytes;
    }

    /// Returns true when presentation must rebuild the texture binding.
    pub(super) fn request_view(
        &mut self,
        device: &wgpu::Device,
        viewport_dimensions: (u32, u32),
        view_transform: Option<super::view::ViewTransform>,
        maximum_texture_dimension: u32,
    ) -> bool {
        let desired = view_transform
            .and_then(|view| {
                resampling_target(
                    viewport_dimensions,
                    view.scale(),
                    self.memory_limit_bytes,
                    maximum_texture_dimension,
                )
                .and_then(|target| {
                    let center = view.center();
                    ViewportRequest::new(target, center.x, center.y, view.scale())
                })
            })
            .filter(|_| self.lifecycle.worker_available());
        if desired == self.lifecycle.desired() {
            return false;
        }

        let binding_changed = self.lifecycle.replace_request(desired);
        if binding_changed {
            self.texture = create_resampled_texture(device, ViewportTarget::FALLBACK);
        }
        if desired.is_some() {
            self.requests = self.requests.saturating_add(1);
        }
        binding_changed
    }

    /// Returns true when presentation must rebuild the resampled texture binding.
    pub(super) fn process_completions(&mut self, device: &wgpu::Device) -> bool {
        let mut binding_changed = false;
        loop {
            let result = match &self.lifecycle {
                ResamplerLifecycle::Unavailable => break,
                ResamplerLifecycle::Available(available) => {
                    match available.worker.results.try_recv() {
                        Ok(result) => result,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            tracing::warn!("viewport resampling worker stopped unexpectedly");
                            if self.lifecycle.disconnect() {
                                self.texture =
                                    create_resampled_texture(device, ViewportTarget::FALLBACK);
                                binding_changed = true;
                            }
                            break;
                        }
                    }
                }
            };
            match result {
                ResamplingWorkResult::Ready(ready) if self.lifecycle.is_current(ready.job) => {
                    self.texture = ready.texture;
                    self.lifecycle.mark_ready(ready.job.view);
                    self.completions = self.completions.saturating_add(1);
                    self.last_elapsed = Some(ready.elapsed);
                    self.last_source_bytes = ready.source_bytes;
                    binding_changed = true;
                }
                ResamplingWorkResult::Failed(job, error) if self.lifecycle.is_current(job) => {
                    tracing::warn!(%error, "viewport resampling failed; keeping tile renderer");
                    self.lifecycle.mark_failed();
                }
                ResamplingWorkResult::Failed(job, error) => {
                    tracing::debug!(
                        %error,
                        generation = job.generation,
                        "discarding stale viewport resampling failure"
                    );
                    self.discarded_results = self.discarded_results.saturating_add(1);
                }
                ResamplingWorkResult::Ready(_) | ResamplingWorkResult::Cancelled => {
                    self.discarded_results = self.discarded_results.saturating_add(1);
                }
            }
        }
        binding_changed
    }

    pub(super) fn status(&self) -> String {
        let state = match &self.lifecycle {
            ResamplerLifecycle::Available(AvailableResampler {
                phase: ResamplerPhase::Active(request),
                ..
            }) => {
                format!("active {}x{}", request.target.width, request.target.height)
            }
            ResamplerLifecycle::Available(AvailableResampler {
                phase: ResamplerPhase::Pending(request),
                ..
            }) => {
                format!("pending {}x{}", request.target.width, request.target.height)
            }
            ResamplerLifecycle::Unavailable
            | ResamplerLifecycle::Available(AvailableResampler {
                phase: ResamplerPhase::Inactive,
                ..
            }) => "inactive".to_owned(),
        };
        let elapsed = self.last_elapsed.map_or_else(
            || "n/a".to_owned(),
            |value| format!("{:.2} ms", value.as_secs_f64() * 1_000.0),
        );
        format!(
            "{state} (requests {}, completions {}, discarded {}, last resample {elapsed}, source {:.2} MiB)",
            self.requests,
            self.completions,
            self.discarded_results,
            bytes_to_mib(u64_from_usize(self.last_source_bytes)),
        )
    }
}

pub(super) fn resampling_target(
    (viewport_width, viewport_height): (u32, u32),
    scale: f64,
    memory_limit_bytes: u64,
    maximum_texture_dimension: u32,
) -> Option<ViewportTarget> {
    if viewport_width == 0
        || viewport_height == 0
        || !scale.is_finite()
        || scale <= 0.0
        || scale.to_bits() == 1.0_f64.to_bits()
        || viewport_width > maximum_texture_dimension
        || viewport_height > maximum_texture_dimension
    {
        return None;
    }
    let target = ViewportTarget {
        width: viewport_width,
        height: viewport_height,
    };
    if target.gpu_budget_bytes() > memory_limit_bytes {
        return None;
    }
    Some(target)
}

pub(super) fn required_resampling_gpu_bytes(
    viewport_dimensions: (u32, u32),
    scale: f64,
    maximum_texture_dimension: u32,
) -> Option<u64> {
    resampling_target(
        viewport_dimensions,
        scale,
        u64::MAX,
        maximum_texture_dimension,
    )
    .map(ViewportTarget::gpu_budget_bytes)
}

fn run_resampling_worker(
    source: &DecodedTileStore,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    state: &Arc<(Mutex<ResamplingWorkerState>, Condvar)>,
    results: &mpsc::Sender<ResamplingWorkResult>,
    notify_ready: &(dyn Fn() + Send + Sync),
    submission_lock: &Mutex<()>,
) {
    while let Some(job) = next_debounced_job(state) {
        // Scratch is intentionally job-scoped so its potentially large allocations
        // are released before the worker returns to its idle wait.
        let mut scratch = ResamplingScratch::default();
        let result = match resample_job(
            source,
            device,
            queue,
            job,
            &mut scratch,
            state,
            submission_lock,
        ) {
            Ok(ready) => ResamplingWorkResult::Ready(ready),
            Err(ResampleError::Cancelled) => ResamplingWorkResult::Cancelled,
            Err(ResampleError::Failed(error)) => ResamplingWorkResult::Failed(job, error),
        };
        record_scratch_peak(state, scratch.allocated_bytes());
        drop(scratch);
        if results.send(result).is_err() {
            return;
        }
        notify_ready();
    }
}

fn resample_job(
    source: &DecodedTileStore,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    job: ResamplingJob,
    scratch: &mut ResamplingScratch,
    state: &Arc<(Mutex<ResamplingWorkerState>, Condvar)>,
    submission_lock: &Mutex<()>,
) -> Result<ResampledViewport, ResampleError> {
    let started = Instant::now();
    let source_bytes = resample_lanczos2(source, job.view, scratch, || {
        !resampling_job_is_current(state, job.generation)
    })?;
    if !resampling_job_is_current(state, job.generation) {
        return Err(ResampleError::Cancelled);
    }
    let texture = create_resampled_texture(device, job.view.target);
    upload_resampled_texture(
        queue,
        &texture,
        job,
        &scratch.output,
        &mut scratch.upload_staging,
        state,
        submission_lock,
    )?;
    if !resampling_job_is_current(state, job.generation) {
        return Err(ResampleError::Cancelled);
    }
    Ok(ResampledViewport {
        job,
        texture,
        elapsed: started.elapsed(),
        source_bytes,
    })
}

fn record_scratch_peak(shared: &Arc<(Mutex<ResamplingWorkerState>, Condvar)>, bytes: usize) {
    let (state, _) = shared.as_ref();
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.scratch_peak_bytes = state.scratch_peak_bytes.max(bytes);
}

enum DebounceOutcome {
    Ready(ResamplingJob),
    Superseded,
    Shutdown,
}

#[inline(never)] // Keep the debounce wait state machine distinct from the hot resampling path.
fn wait_for_debounce(
    mut state: std::sync::MutexGuard<'_, ResamplingWorkerState>,
    wake: &Condvar,
    mut job: ResamplingJob,
) -> DebounceOutcome {
    loop {
        if state.shutdown {
            return DebounceOutcome::Shutdown;
        }
        if state.generation != job.generation {
            return DebounceOutcome::Superseded;
        }
        if let Some(replacement) = state.pending.take() {
            job = replacement;
            continue;
        }
        let now = Instant::now();
        if now >= job.deadline {
            return DebounceOutcome::Ready(job);
        }
        let timeout = job.deadline.saturating_duration_since(now);
        let (next_state, _) = wake
            .wait_timeout(state, timeout)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state = next_state;
    }
}

#[inline(never)] // Keep blocking mutex/condvar state out of the worker's processing frame.
fn next_debounced_job(
    shared: &Arc<(Mutex<ResamplingWorkerState>, Condvar)>,
) -> Option<ResamplingJob> {
    let (state, wake) = shared.as_ref();
    'poll: loop {
        let mut locked = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if locked.shutdown {
                return None;
            }
            if let Some(job) = locked.pending.take() {
                match wait_for_debounce(locked, wake, job) {
                    DebounceOutcome::Ready(job) => return Some(job),
                    DebounceOutcome::Superseded => continue 'poll,
                    DebounceOutcome::Shutdown => return None,
                }
            }
            locked = wake
                .wait(locked)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

fn resampling_job_is_current(
    shared: &Arc<(Mutex<ResamplingWorkerState>, Condvar)>,
    generation: u64,
) -> bool {
    let (state, _) = shared.as_ref();
    let state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    !state.shutdown && state.generation == generation
}

#[derive(Debug)]
enum ResampleError {
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

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Dimensions bound the non-negative pixel indices; weights intentionally use f32 storage.
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
    end_source_y: u32,
    samples: Vec<RgbaF32>,
}

#[derive(Default)]
struct ResamplingScratch {
    source_rows: Vec<RgbaF32>,
    horizontal_rows: Vec<RgbaF32>,
    output: Vec<u8>,
    upload_staging: Vec<u8>,
    active_rows: VecDeque<ActiveOutputRow>,
    free_row_buffers: Vec<Vec<RgbaF32>>,
    horizontal: AxisContributions,
    vertical: AxisContributions,
}

impl ResamplingScratch {
    fn allocated_bytes(&self) -> usize {
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

fn resample_lanczos2(
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

#[allow(clippy::too_many_lines)] // Keeps the cancellable convolution, memory-limited output, and 64 MiB-per-axis coefficient tables in one transaction.
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

fn prepare_reused_buffer<T: Clone + Default>(
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

fn create_resampled_texture(device: &wgpu::Device, target: ViewportTarget) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resampled canonical viewport"),
        size: wgpu::Extent3d {
            width: target.width,
            height: target.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn upload_resampled_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    job: ResamplingJob,
    pixels: &[u8],
    staging: &mut Vec<u8>,
    state: &Arc<(Mutex<ResamplingWorkerState>, Condvar)>,
    submission_lock: &Mutex<()>,
) -> Result<(), ResampleError> {
    let upload_layout = TextureUploadLayout::rgba16f(
        job.view.target.width,
        job.view.target.height,
        "resampling upload",
    )?;
    upload_layout.validate_source_len(pixels.len())?;
    prepare_reused_buffer(
        staging,
        upload_layout.staging_bytes(),
        "resampling upload staging",
    )?;
    for stripe in upload_layout.stripes() {
        if !resampling_job_is_current(state, job.generation) {
            return Err(ResampleError::Cancelled);
        }
        upload_layout.copy_stripe(pixels, stripe, staging);
        let _submission_guard = submission_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !resampling_job_is_current(state, job.generation) {
            return Err(ResampleError::Cancelled);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: stripe.first_row(),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            upload_layout.stripe_data(staging, stripe),
            upload_layout.copy_buffer_layout(stripe),
            upload_layout.copy_extent(stripe),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    use super::*;

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

    fn centered_view(source: (u32, u32), target: (u32, u32), scale: f64) -> ViewportRequest {
        ViewportRequest {
            target: ViewportTarget {
                width: target.0,
                height: target.1,
            },
            center_x: f64::from(source.0) * 0.5,
            center_y: f64::from(source.1) * 0.5,
            scale,
        }
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

    fn available_test_lifecycle(results: Receiver<ResamplingWorkResult>) -> ResamplerLifecycle {
        ResamplerLifecycle::Available(AvailableResampler {
            worker: ResamplingWorker {
                state: Arc::new((Mutex::new(ResamplingWorkerState::default()), Condvar::new())),
                results,
                thread: None,
            },
            generation: 0,
            phase: ResamplerPhase::Inactive,
        })
    }

    fn native_test_device() -> wgpu::Device {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = crate::gpu::native_backends();
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("resampling lifecycle test requires a native GPU adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .unwrap()
            .0
    }

    #[test]
    fn lifecycle_tracks_pending_active_and_unavailable_states() {
        let (_result_sender, results) = mpsc::channel();
        let request = centered_view((100, 100), (40, 30), 0.5);
        let mut lifecycle = available_test_lifecycle(results);
        let job = ResamplingJob {
            generation: 1,
            view: request,
            deadline: Instant::now(),
        };

        assert!(!lifecycle.replace_request(Some(request)));
        assert_eq!(lifecycle.desired(), Some(request));
        assert!(lifecycle.is_current(job));
        assert!(!lifecycle.is_active());
        assert!(lifecycle.active_target().is_none());

        lifecycle.mark_ready(request);
        assert!(!lifecycle.is_current(job));
        assert!(lifecycle.is_active());
        assert_eq!(lifecycle.active_target(), Some(request.target));

        assert!(lifecycle.replace_request(Some(request)));
        assert!(!lifecycle.is_active());
        assert!(lifecycle.active_target().is_none());
        lifecycle.mark_failed();
        assert!(!lifecycle.is_active());
        assert!(lifecycle.desired().is_none());
        assert!(lifecycle.active_target().is_none());

        assert!(!lifecycle.replace_request(Some(request)));
        assert!(!lifecycle.is_active());
        assert!(lifecycle.active_target().is_none());

        assert!(!lifecycle.disconnect());
        assert!(!lifecycle.worker_available());
        assert!(lifecycle.desired().is_none());
        assert!(lifecycle.active_target().is_none());

        let (_result_sender, results) = mpsc::channel();
        let mut lifecycle = available_test_lifecycle(results);
        assert!(!lifecycle.replace_request(Some(request)));
        lifecycle.mark_ready(request);
        assert!(lifecycle.disconnect());
        assert!(!lifecycle.worker_available());
        assert!(lifecycle.active_target().is_none());
    }

    #[test]
    fn viewport_request_rejects_invalid_floating_point_inputs() {
        let target = ViewportTarget {
            width: 40,
            height: 30,
        };
        assert!(ViewportRequest::new(target, 50.0, 50.0, 0.5).is_some());
        for (center_x, center_y) in [
            (f64::NAN, 50.0),
            (50.0, f64::NAN),
            (f64::INFINITY, 50.0),
            (50.0, f64::NEG_INFINITY),
        ] {
            assert!(ViewportRequest::new(target, center_x, center_y, 0.5).is_none());
        }
        for scale in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.5] {
            assert!(ViewportRequest::new(target, 50.0, 50.0, scale).is_none());
        }
    }

    #[test]
    fn scratch_peak_preserves_the_largest_observation() {
        let state = Arc::new((Mutex::new(ResamplingWorkerState::default()), Condvar::new()));
        record_scratch_peak(&state, 1_024);
        record_scratch_peak(&state, 512);
        let (state, _) = state.as_ref();
        assert_eq!(
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .scratch_peak_bytes,
            1_024
        );
    }

    #[test]
    fn worker_panic_payloads_have_readable_messages() {
        assert_eq!(
            panic_payload_message(&"borrowed message"),
            "borrowed message"
        );
        assert_eq!(
            panic_payload_message(&"owned message".to_owned()),
            "owned message"
        );
        assert_eq!(panic_payload_message(&42_u8), "non-string panic payload");
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
    fn debounced_worker_uses_only_the_latest_request() {
        let state = Arc::new((Mutex::new(ResamplingWorkerState::default()), Condvar::new()));
        let (_result_sender, results) = mpsc::channel();
        let worker = ResamplingWorker {
            state: Arc::clone(&state),
            results,
            thread: None,
        };
        let first = centered_view((100, 100), (20, 20), 0.5);
        let latest = centered_view((100, 100), (30, 20), 0.75);
        worker.replace(1, Some(first));
        let consumer_state = Arc::clone(&state);
        let consumer = std::thread::spawn(move || next_debounced_job(&consumer_state));
        worker.replace(2, Some(latest));

        let job = consumer
            .join()
            .unwrap()
            .expect("the latest request should survive the debounce");
        assert_eq!(job.generation, 2);
        assert_eq!(job.view, latest);
        assert!(resampling_job_is_current(&state, 2));
        assert!(!resampling_job_is_current(&state, 1));
    }

    #[test]
    fn waiting_resampling_worker_stops_on_shutdown() {
        let state = Arc::new((Mutex::new(ResamplingWorkerState::default()), Condvar::new()));
        let consumer_state = Arc::clone(&state);
        let consumer = std::thread::spawn(move || next_debounced_job(&consumer_state));
        let (locked, wake) = state.as_ref();
        {
            let mut locked = locked
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            locked.shutdown = true;
            wake.notify_one();
        }

        assert!(consumer.join().unwrap().is_none());
        assert!(!resampling_job_is_current(&state, 0));
    }

    #[test]
    #[ignore = "requires a native GPU adapter for viewport texture lifecycle"]
    fn viewport_resampler_rejects_stale_results_and_handles_failures() {
        let device = native_test_device();
        let (result_sender, results) = mpsc::channel();
        let mut resampler = ViewportResampler::fallback(&device, 64 * 1024 * 1024);
        resampler.lifecycle = available_test_lifecycle(results);
        let view = super::super::view::ViewTransform::fit(
            winit::dpi::PhysicalSize::new(100, 100),
            winit::dpi::PhysicalSize::new(50, 50),
        );

        assert!(!resampler.request_view(&device, (50, 50), Some(view), 16_384));
        assert_eq!(resampler.requests, 1);
        assert!(!resampler.request_view(&device, (50, 50), Some(view), 16_384));
        assert_eq!(resampler.requests, 1);
        let (current_generation, request) = resampler
            .lifecycle
            .current_request()
            .expect("the resampler should have a pending request");
        result_sender
            .send(ResamplingWorkResult::Ready(ResampledViewport {
                job: ResamplingJob {
                    generation: current_generation.wrapping_sub(1),
                    view: request,
                    deadline: Instant::now(),
                },
                texture: create_resampled_texture(&device, request.target),
                elapsed: Duration::from_millis(1),
                source_bytes: 4,
            }))
            .unwrap();
        assert!(!resampler.process_completions(&device));
        assert_eq!(resampler.discarded_results, 1);
        assert!(!resampler.is_active());

        result_sender
            .send(ResamplingWorkResult::Ready(ResampledViewport {
                job: ResamplingJob {
                    generation: current_generation,
                    view: request,
                    deadline: Instant::now(),
                },
                texture: create_resampled_texture(&device, request.target),
                elapsed: Duration::from_millis(2),
                source_bytes: 8,
            }))
            .unwrap();
        assert!(resampler.process_completions(&device));
        assert!(resampler.is_active());
        assert_eq!(resampler.completions, 1);
        assert_eq!(resampler.last_source_bytes, 8);

        resampler.set_memory_limit_bytes(0);
        assert!(resampler.request_view(&device, (50, 50), Some(view), 16_384));
        assert!(!resampler.is_active());
        assert!(resampler.lifecycle.desired().is_none());
        assert_eq!(resampler.gpu_bytes(), 0);

        resampler.set_memory_limit_bytes(64 * 1024 * 1024);
        assert!(!resampler.request_view(&device, (50, 50), Some(view), 16_384));
        let (current_generation, request) = resampler
            .lifecycle
            .current_request()
            .expect("the resampler should have a pending request");
        result_sender
            .send(ResamplingWorkResult::Failed(
                ResamplingJob {
                    generation: current_generation.wrapping_sub(1),
                    view: request,
                    deadline: Instant::now(),
                },
                io::Error::other("stale failure"),
            ))
            .unwrap();
        assert!(!resampler.process_completions(&device));
        assert_eq!(resampler.discarded_results, 2);
        assert_eq!(resampler.lifecycle.desired(), Some(request));
        result_sender
            .send(ResamplingWorkResult::Failed(
                ResamplingJob {
                    generation: current_generation,
                    view: request,
                    deadline: Instant::now(),
                },
                io::Error::other("current failure"),
            ))
            .unwrap();
        assert!(!resampler.process_completions(&device));
        assert_eq!(resampler.discarded_results, 2);
        assert!(resampler.lifecycle.desired().is_none());

        drop(result_sender);
        assert!(!resampler.process_completions(&device));
        assert!(!resampler.lifecycle.worker_available());
    }

    #[test]
    #[ignore = "requires a native GPU adapter for viewport texture lifecycle"]
    fn worker_disconnection_releases_a_retained_viewport_texture() {
        let device = native_test_device();
        let (result_sender, results) = mpsc::channel();
        let mut resampler = ViewportResampler::fallback(&device, 64 * 1024 * 1024);
        resampler.lifecycle = available_test_lifecycle(results);
        let request = centered_view((100, 100), (50, 50), 0.5);
        assert!(!resampler.lifecycle.replace_request(Some(request)));
        resampler.texture = create_resampled_texture(&device, request.target);
        resampler.lifecycle.mark_ready(request);
        assert!(resampler.is_active());
        assert!(resampler.gpu_bytes() > 0);

        drop(result_sender);

        assert!(resampler.process_completions(&device));
        assert!(!resampler.lifecycle.worker_available());
        assert_eq!(resampler.gpu_bytes(), 0);
        assert_eq!(resampler.texture.width(), 1);
        assert_eq!(resampler.texture.height(), 1);
    }

    #[test]
    fn target_uses_the_bounded_viewport_at_every_non_unit_scale() {
        assert_eq!(
            resampling_target((1_000, 1_000), 0.25, 64 * 1024 * 1024, 16_384),
            Some(ViewportTarget {
                width: 1_000,
                height: 1_000,
            })
        );
        assert_eq!(
            resampling_target((1_000, 1_000), 2.0, 64 * 1024 * 1024, 16_384),
            Some(ViewportTarget {
                width: 1_000,
                height: 1_000,
            })
        );
        assert_eq!(
            resampling_target((1_000, 1_000), 1.0, 64 * 1024 * 1024, 16_384),
            None
        );
    }

    #[test]
    fn target_rejects_zero_oversized_and_over_budget_viewports() {
        assert_eq!(resampling_target((0, 10), 0.5, 1024, 16_384), None);
        assert_eq!(resampling_target((100, 10), 0.5, 1024, 16_384), None);
        assert_eq!(resampling_target((20_000, 10), 0.5, u64::MAX, 16_384), None);
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

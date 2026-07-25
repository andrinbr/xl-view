use std::collections::VecDeque;
use std::io;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use wgpu::TextureFormat;
use xl_view::decode::{CANONICAL_BYTES_PER_PIXEL, DecodedTileStore};

use super::WorkReadyNotifier;
use super::mip::rgba16f_texture_budget_bytes;
use super::upload::TextureUploadLayout;
use crate::units::{bytes_to_mib, u64_from_usize, usize_from_u32};

const RESAMPLING_DEBOUNCE: Duration = Duration::from_millis(50);
const MAX_COEFFICIENT_BYTES: usize = 64 * 1024 * 1024;
const MAX_FINITE_F16: f32 = 65_504.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ViewportTarget {
    pub(super) width: u32,
    pub(super) height: u32,
}

impl ViewportTarget {
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
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

enum ResamplerLifecycle {
    // A completed texture may remain bound after the worker stops. It is ignored by
    // presentation, but `retained` keeps its GPU allocation accounted for.
    Unavailable { retained: Option<ViewportTarget> },
    Available(AvailableResampler),
}

struct AvailableResampler {
    worker: ResamplingWorker,
    generation: u64,
    phase: ResamplerPhase,
}

enum ResamplerPhase {
    // A late failure can leave the previous completed texture allocated but inactive.
    Inactive { retained: Option<ViewportTarget> },
    // Pending work uses the one-pixel fallback binding.
    Pending(ViewportRequest),
    // Active work owns the completed texture represented by `ViewportResampler::texture`.
    Active(ViewportRequest),
}

impl ResamplerPhase {
    fn desired(&self) -> Option<ViewportRequest> {
        match self {
            Self::Pending(request) | Self::Active(request) => Some(*request),
            Self::Inactive { .. } => None,
        }
    }

    fn retained_target(&self) -> Option<ViewportTarget> {
        match self {
            Self::Inactive { retained } => *retained,
            Self::Pending(_) => None,
            Self::Active(request) => Some(request.target),
        }
    }

    fn is_active(&self) -> bool {
        matches!(self, Self::Active(_))
    }

    fn replace_request(&mut self, desired: Option<ViewportRequest>) -> bool {
        let binding_changed = self.retained_target().is_some();
        *self = desired.map_or(Self::Inactive { retained: None }, Self::Pending);
        binding_changed
    }

    fn mark_failed(&mut self) {
        *self = Self::Inactive {
            retained: self.retained_target(),
        };
    }
}

impl ResamplerLifecycle {
    fn worker_available(&self) -> bool {
        matches!(self, Self::Available(_))
    }

    fn desired(&self) -> Option<ViewportRequest> {
        match self {
            Self::Unavailable { .. } => None,
            Self::Available(available) => available.phase.desired(),
        }
    }

    fn retained_target(&self) -> Option<ViewportTarget> {
        match self {
            Self::Unavailable { retained } => *retained,
            Self::Available(available) => available.phase.retained_target(),
        }
    }

    fn is_active(&self) -> bool {
        matches!(self, Self::Available(available) if available.phase.is_active())
    }

    fn is_current(&self, job: ResamplingJob) -> bool {
        match self {
            Self::Available(available) => {
                job.generation == available.generation
                    && Some(job.view) == available.phase.desired()
            }
            Self::Unavailable { .. } => false,
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
            available.phase.mark_failed();
        }
    }

    fn disconnect(&mut self) {
        let retained = self.retained_target();
        *self = Self::Unavailable { retained };
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
    // the fallback, a retained inactive allocation, or the active resampled viewport.
    texture: wgpu::Texture,
    lifecycle: ResamplerLifecycle,
    memory_limit_bytes: u64,
    requests: u64,
    cancellations: u64,
    completions: u64,
    last_elapsed: Option<Duration>,
    last_source_bytes: usize,
}

impl ViewportResampler {
    pub(super) fn fallback(device: &wgpu::Device, memory_limit_bytes: u64) -> Self {
        Self {
            texture: create_resampled_texture(
                device,
                ViewportTarget {
                    width: 1,
                    height: 1,
                },
            ),
            lifecycle: ResamplerLifecycle::Unavailable { retained: None },
            memory_limit_bytes,
            requests: 0,
            cancellations: 0,
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
                    phase: ResamplerPhase::Inactive { retained: None },
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
            .retained_target()
            .map_or(0, ViewportTarget::gpu_budget_bytes)
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
                .map(|target| ViewportRequest {
                    target,
                    center_x: view.center().x,
                    center_y: view.center().y,
                    scale: view.scale(),
                })
            })
            .filter(|_| self.lifecycle.worker_available());
        if desired == self.lifecycle.desired() {
            return false;
        }

        let binding_changed = self.lifecycle.replace_request(desired);
        if binding_changed {
            self.texture = create_resampled_texture(
                device,
                ViewportTarget {
                    width: 1,
                    height: 1,
                },
            );
        }
        if desired.is_some() {
            self.requests = self.requests.saturating_add(1);
        }
        binding_changed
    }

    /// Returns true when presentation must rebuild the resampled texture binding.
    pub(super) fn process_completions(&mut self) -> bool {
        let mut binding_changed = false;
        loop {
            let result = match &self.lifecycle {
                ResamplerLifecycle::Unavailable { .. } => break,
                ResamplerLifecycle::Available(available) => {
                    match available.worker.results.try_recv() {
                        Ok(result) => result,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            tracing::warn!("viewport resampling worker stopped unexpectedly");
                            self.lifecycle.disconnect();
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
                ResamplingWorkResult::Ready(_) | ResamplingWorkResult::Cancelled => {
                    self.cancellations = self.cancellations.saturating_add(1);
                }
                ResamplingWorkResult::Failed(job, error) => {
                    if self.lifecycle.is_current(job) {
                        tracing::warn!(%error, "viewport resampling failed; keeping tile renderer");
                        self.lifecycle.mark_failed();
                    }
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
            ResamplerLifecycle::Unavailable { .. }
            | ResamplerLifecycle::Available(AvailableResampler {
                phase: ResamplerPhase::Inactive { .. },
                ..
            }) => "inactive".to_owned(),
        };
        let elapsed = self.last_elapsed.map_or_else(
            || "n/a".to_owned(),
            |value| format!("{:.2} ms", value.as_secs_f64() * 1_000.0),
        );
        format!(
            "{state} (requests {}, completions {}, cancellations {}, last resample {elapsed}, source {:.2} MiB)",
            self.requests,
            self.completions,
            self.cancellations,
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
        let started = Instant::now();
        let resized = resample_lanczos2(source, job.view, || {
            !resampling_job_is_current(state, job.generation)
        });
        let result = match resized {
            Ok(resized) if resampling_job_is_current(state, job.generation) => {
                let texture = create_resampled_texture(device, job.view.target);
                match upload_resampled_texture(
                    queue,
                    &texture,
                    job,
                    &resized.pixels,
                    state,
                    submission_lock,
                ) {
                    Ok(()) if resampling_job_is_current(state, job.generation) => {
                        ResamplingWorkResult::Ready(ResampledViewport {
                            job,
                            texture,
                            elapsed: started.elapsed(),
                            source_bytes: resized.source_bytes,
                        })
                    }
                    Ok(()) | Err(ResampleError::Cancelled) => ResamplingWorkResult::Cancelled,
                    Err(ResampleError::Failed(error)) => ResamplingWorkResult::Failed(job, error),
                }
            }
            Ok(_) | Err(ResampleError::Cancelled) => ResamplingWorkResult::Cancelled,
            Err(ResampleError::Failed(error)) => ResamplingWorkResult::Failed(job, error),
        };
        if results.send(result).is_err() {
            return;
        }
        notify_ready();
    }
}

enum DebounceOutcome {
    Ready(ResamplingJob),
    Superseded,
    Shutdown,
}

#[inline(never)] // Keep blocking mutex/condvar state out of the worker's processing frame.
fn wait_for_pending_job<'a>(
    state: &'a Mutex<ResamplingWorkerState>,
    wake: &Condvar,
) -> Option<(
    std::sync::MutexGuard<'a, ResamplingWorkerState>,
    ResamplingJob,
)> {
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if state.shutdown {
            return None;
        }
        if let Some(job) = state.pending.take() {
            return Some((state, job));
        }
        state = wake
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
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

fn next_debounced_job(
    shared: &Arc<(Mutex<ResamplingWorkerState>, Condvar)>,
) -> Option<ResamplingJob> {
    let (state, wake) = shared.as_ref();
    loop {
        let (locked, job) = wait_for_pending_job(state, wake)?;
        match wait_for_debounce(locked, wake, job) {
            DebounceOutcome::Ready(job) => return Some(job),
            DebounceOutcome::Superseded => {}
            DebounceOutcome::Shutdown => return None,
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

struct AxisContributions {
    bounds: Vec<AxisBound>,
    weights: Vec<f32>,
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
    is_cancelled: &mut impl FnMut() -> bool,
) -> Result<AxisContributions, ResampleError> {
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

    let mut bounds = Vec::new();
    bounds
        .try_reserve_exact(usize_from_u32(target_size))
        .map_err(|error| io::Error::other(format!("cannot allocate Lanczos2 bounds: {error}")))?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(maximum_weights)
        .map_err(|error| io::Error::other(format!("cannot allocate Lanczos2 weights: {error}")))?;

    for target in 0..target_size {
        if target % 32 == 0 && is_cancelled() {
            return Err(ResampleError::Cancelled);
        }
        let source_center = view_center
            + (f64::from(target) + 0.5 - f64::from(target_size) * 0.5) * source_per_target;
        if source_center < 0.0 || source_center > f64::from(source_size) {
            bounds.push(AxisBound {
                start: 0,
                length: 0,
                weights_start: weights.len(),
            });
            continue;
        }
        let first = (source_center - radius).floor().max(0.0) as u32;
        let end = (source_center + radius).ceil().min(f64::from(source_size)) as u32;
        let weights_start = weights.len();
        let mut sum = 0.0_f64;
        for source in first..end {
            let distance = (f64::from(source) + 0.5 - source_center) / filter_scale;
            let weight = lanczos2(distance);
            weights.push(weight as f32);
            sum += weight;
        }
        validate_weight_sum(sum)?;
        for weight in &mut weights[weights_start..] {
            *weight = (f64::from(*weight) / sum) as f32;
        }
        bounds.push(AxisBound {
            start: first,
            length: end - first,
            weights_start,
        });
    }
    Ok(AxisContributions { bounds, weights })
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

struct ActiveOutputRow {
    target_y: u32,
    end_source_y: u32,
    samples: Vec<f32>,
}

struct ResampledPixels {
    pixels: Vec<u8>,
    source_bytes: usize,
}

#[inline]
fn activate_output_rows(
    vertical: &AxisContributions,
    target_height: u32,
    target_row_samples: usize,
    source_y: u32,
    next_target_y: &mut u32,
    active: &mut VecDeque<ActiveOutputRow>,
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
        active.push_back(ActiveOutputRow {
            target_y: *next_target_y,
            end_source_y: bound.start + bound.length,
            samples: try_zeroed_f32(target_row_samples, "active output row")?,
        });
        *next_target_y += 1;
    }
    Ok(())
}

#[inline]
fn flush_completed_output_rows(
    active: &mut VecDeque<ActiveOutputRow>,
    source_y: u32,
    output: &mut [u8],
    target_width: u32,
) {
    while active
        .front()
        .is_some_and(|row| row.end_source_y <= source_y + 1)
    {
        let row = active
            .pop_front()
            .expect("the front row was checked before removal");
        encode_resampled_row(&row.samples, output, target_width, row.target_y);
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

fn resample_lanczos2(
    source: &DecodedTileStore,
    view: ViewportRequest,
    is_cancelled: impl FnMut() -> bool,
) -> Result<ResampledPixels, ResampleError> {
    resample_lanczos2_rows(
        source.dimensions(),
        view,
        is_cancelled,
        |row, start, destination| source.read_canonical_row_range_rgba_f32(row, start, destination),
    )
}

#[allow(clippy::too_many_lines)] // Keeps the cancellable convolution, memory-limited output, and 64 MiB-per-axis coefficient tables in one transaction.
fn resample_lanczos2_rows(
    (source_width, source_height): (u32, u32),
    view: ViewportRequest,
    mut is_cancelled: impl FnMut() -> bool,
    mut read_source_row: impl FnMut(u32, u32, &mut [f32]) -> Result<(), io::Error>,
) -> Result<ResampledPixels, ResampleError> {
    let target = view.target;
    let horizontal = build_axis_contributions(
        source_width,
        target.width,
        view.center_x,
        view.scale,
        &mut is_cancelled,
    )?;
    let vertical = build_axis_contributions(
        source_height,
        target.height,
        view.center_y,
        view.scale,
        &mut is_cancelled,
    )?;
    if is_cancelled() {
        return Err(ResampleError::Cancelled);
    }

    let horizontal_source_range = contribution_source_range(&horizontal);
    let vertical_source_range = contribution_source_range(&vertical);
    let source_row_pixels = horizontal_source_range.map_or(0, |(start, end)| end - start);
    let source_row_samples = usize_from_u32(source_row_pixels)
        .checked_mul(4)
        .ok_or_else(|| io::Error::other("source row sample count overflowed"))?;
    let target_row_samples = usize_from_u32(target.width)
        .checked_mul(4)
        .ok_or_else(|| io::Error::other("target row sample count overflowed"))?;
    let output_bytes = usize_from_u32(target.width)
        .checked_mul(usize_from_u32(target.height))
        .and_then(|pixels| pixels.checked_mul(CANONICAL_BYTES_PER_PIXEL))
        .ok_or_else(|| io::Error::other("resampled output size overflowed"))?;

    let mut source_row = try_zeroed_f32(source_row_samples, "source row")?;
    let mut horizontal_row = try_zeroed_f32(target_row_samples, "horizontal row")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_bytes)
        .map_err(|error| io::Error::other(format!("cannot allocate resampled output: {error}")))?;
    output.resize(output_bytes, 0);

    let mut active = VecDeque::<ActiveOutputRow>::new();
    let mut next_target_y = 0_u32;
    let mut source_rows_read = 0_usize;
    let (first_source_y, end_source_y) = vertical_source_range.unwrap_or((0, 0));
    for source_y in first_source_y..end_source_y {
        if is_cancelled() {
            return Err(ResampleError::Cancelled);
        }
        activate_output_rows(
            &vertical,
            target.height,
            target_row_samples,
            source_y,
            &mut next_target_y,
            &mut active,
        )?;
        if active.is_empty() {
            continue;
        }

        let source_x_start = horizontal_source_range.map_or(0, |range| range.0);
        read_source_row(source_y, source_x_start, &mut source_row)?;
        source_rows_read = source_rows_read.saturating_add(1);
        horizontal_convolution(
            &source_row,
            &mut horizontal_row,
            &horizontal,
            source_x_start,
        );
        for row in &mut active {
            let bound = vertical.bounds[usize_from_u32(row.target_y)];
            if !(bound.start..row.end_source_y).contains(&source_y) {
                continue;
            }
            let offset = usize_from_u32(source_y - bound.start);
            let weight = vertical.weights[bound.weights_start + offset];
            for (destination, sample) in row.samples.iter_mut().zip(&horizontal_row) {
                *destination += sample * weight;
            }
        }
        flush_completed_output_rows(&mut active, source_y, &mut output, target.width);
    }
    skip_empty_output_rows(&vertical, target.height, &mut next_target_y);
    ensure_output_rows_complete(next_target_y, target.height, active.len())?;
    let source_bytes = usize_from_u32(source_row_pixels)
        .saturating_mul(source_rows_read)
        .saturating_mul(CANONICAL_BYTES_PER_PIXEL);
    Ok(ResampledPixels {
        pixels: output,
        source_bytes,
    })
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

fn try_zeroed_f32(length: usize, label: &str) -> Result<Vec<f32>, ResampleError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|error| io::Error::other(format!("cannot allocate {label}: {error}")))?;
    values.resize(length, 0.0);
    Ok(values)
}

#[inline(never)] // Avoid duplicating this sizeable inner convolution loop into its streaming caller.
fn horizontal_convolution(
    source: &[f32],
    destination: &mut [f32],
    contributions: &AxisContributions,
    source_origin: u32,
) {
    for (target_x, bound) in contributions.bounds.iter().enumerate() {
        let destination = &mut destination[target_x * 4..target_x * 4 + 4];
        destination.fill(0.0);
        for offset in 0..usize_from_u32(bound.length) {
            let source_x = usize_from_u32(bound.start - source_origin) + offset;
            let source = &source[source_x * 4..source_x * 4 + 4];
            let weight = contributions.weights[bound.weights_start + offset];
            for channel in 0..4 {
                destination[channel] += source[channel] * weight;
            }
        }
    }
}

fn encode_resampled_row(samples: &[f32], output: &mut [u8], width: u32, row: u32) {
    let row_bytes = usize_from_u32(width) * CANONICAL_BYTES_PER_PIXEL;
    let destination = &mut output[usize_from_u32(row) * row_bytes..][..row_bytes];
    for (rgba, encoded) in samples
        .chunks_exact(4)
        .zip(destination.chunks_exact_mut(CANONICAL_BYTES_PER_PIXEL))
    {
        let alpha = if rgba[3].is_finite() {
            rgba[3].clamp(0.0, 1.0)
        } else {
            0.0
        };
        for channel in 0..3 {
            let value = if alpha == 0.0 || !rgba[channel].is_finite() {
                0.0
            } else {
                rgba[channel].clamp(-MAX_FINITE_F16, MAX_FINITE_F16)
            };
            encoded[channel * 2..channel * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
        }
        encoded[6..8].copy_from_slice(&half::f16::from_f32(alpha).to_bits().to_le_bytes());
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
    state: &Arc<(Mutex<ResamplingWorkerState>, Condvar)>,
    submission_lock: &Mutex<()>,
) -> Result<(), ResampleError> {
    let upload_layout = TextureUploadLayout::rgba16f(
        job.view.target.width,
        job.view.target.height,
        "resampling upload",
    )?;
    upload_layout.validate_source_len(pixels.len())?;
    let mut staging = upload_layout.allocate_staging();
    for stripe in upload_layout.stripes() {
        if !resampling_job_is_current(state, job.generation) {
            return Err(ResampleError::Cancelled);
        }
        upload_layout.copy_stripe(pixels, stripe, &mut staging);
        let _submission_guard = submission_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            upload_layout.stripe_data(&staging, stripe),
            upload_layout.copy_buffer_layout(stripe),
            upload_layout.copy_extent(stripe),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    fn resample_test_pixels(
        dimensions: (u32, u32),
        source: &[f32],
        view: ViewportRequest,
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<Vec<[f32; 4]>, ResampleError> {
        let source_row_samples = usize::try_from(dimensions.0).unwrap() * 4;
        let resized = resample_lanczos2_rows(
            dimensions,
            view,
            is_cancelled,
            |row, source_x, destination| {
                let start = usize::try_from(row).unwrap() * source_row_samples
                    + usize::try_from(source_x).unwrap() * 4;
                destination.copy_from_slice(&source[start..start + destination.len()]);
                Ok(())
            },
        )?;
        Ok(resized
            .pixels
            .chunks_exact(CANONICAL_BYTES_PER_PIXEL)
            .map(|rgba| {
                std::array::from_fn(|channel| {
                    let start = channel * 2;
                    half::f16::from_le_bytes([rgba[start], rgba[start + 1]]).to_f32()
                })
            })
            .collect())
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

    fn available_test_lifecycle(results: Receiver<ResamplingWorkResult>) -> ResamplerLifecycle {
        ResamplerLifecycle::Available(AvailableResampler {
            worker: ResamplingWorker {
                state: Arc::new((Mutex::new(ResamplingWorkerState::default()), Condvar::new())),
                results,
                thread: None,
            },
            generation: 0,
            phase: ResamplerPhase::Inactive { retained: None },
        })
    }

    #[test]
    fn lifecycle_tracks_pending_active_and_retained_texture_states() {
        let (_result_sender, results) = mpsc::channel();
        let request = centered_view((100, 100), (40, 30), 0.5);
        let mut lifecycle = available_test_lifecycle(results);

        assert!(!lifecycle.replace_request(Some(request)));
        assert_eq!(lifecycle.desired(), Some(request));
        assert!(lifecycle.is_current(ResamplingJob {
            generation: 1,
            view: request,
            deadline: Instant::now(),
        }));
        assert!(!lifecycle.is_active());
        assert!(lifecycle.retained_target().is_none());

        lifecycle.mark_ready(request);
        assert!(lifecycle.is_active());
        assert_eq!(lifecycle.retained_target(), Some(request.target));

        lifecycle.mark_failed();
        assert!(!lifecycle.is_active());
        assert!(lifecycle.desired().is_none());
        assert_eq!(lifecycle.retained_target(), Some(request.target));

        assert!(lifecycle.replace_request(Some(request)));
        assert!(!lifecycle.is_active());
        assert!(lifecycle.retained_target().is_none());

        lifecycle.disconnect();
        assert!(!lifecycle.worker_available());
        assert!(lifecycle.desired().is_none());
        assert!(lifecycle.retained_target().is_none());
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
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = crate::gpu::native_backends();
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("resampling lifecycle test requires a native GPU adapter");
        let (device, _) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
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
        assert!(!resampler.process_completions());
        assert_eq!(resampler.cancellations, 1);
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
        assert!(resampler.process_completions());
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
        assert!(!resampler.process_completions());
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
        assert!(!resampler.process_completions());
        assert!(resampler.lifecycle.desired().is_none());

        drop(result_sender);
        assert!(!resampler.process_completions());
        assert!(!resampler.lifecycle.worker_available());
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
        let axis = build_axis_contributions(100, 10, 50.0, 0.1, &mut never_cancel).unwrap();
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
        let result = build_axis_contributions(100_000, 10_000, 50_000.0, 0.1, &mut || {
            checks += 1;
            checks > 2
        });
        assert!(matches!(result, Err(ResampleError::Cancelled)));
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
        let mut encoded = [0_u8; 16];
        encode_resampled_row(
            &[1.0, 2.0, 3.0, -1.0, f32::INFINITY, -70_000.0, 70_000.0, 2.0],
            &mut encoded,
            2,
            0,
        );
        let output: Vec<f32> = encoded
            .chunks_exact(2)
            .map(|value| half::f16::from_le_bytes([value[0], value[1]]).to_f32())
            .collect();
        assert_eq!(&output[..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&output[4..], &[0.0, -MAX_FINITE_F16, MAX_FINITE_F16, 1.0]);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn source_scan_stops_after_cancellation() {
        let checks = Cell::new(0_u32);
        let rows_read = Cell::new(0_u32);
        let source = vec![1.0; 64 * 64 * 4];
        let result = resample_lanczos2_rows(
            (64, 64),
            centered_view((64, 64), (16, 16), 0.25),
            || {
                checks.set(checks.get() + 1);
                checks.get() > 8
            },
            |row, source_x, destination| {
                rows_read.set(rows_read.get() + 1);
                let row_samples = 64 * 4;
                let start = usize::try_from(row).unwrap() * row_samples
                    + usize::try_from(source_x).unwrap() * 4;
                destination.copy_from_slice(&source[start..start + destination.len()]);
                Ok(())
            },
        );
        assert!(matches!(result, Err(ResampleError::Cancelled)));
        assert!(rows_read.get() < 64);
    }

    #[test]
    fn viewport_resampling_reads_only_the_required_source_region() {
        let source = vec![1.0; 100 * 100 * 4];
        let rows_read = Cell::new(0_u32);
        let smallest_start = Cell::new(u32::MAX);
        let widest_row = Cell::new(0_usize);
        let resized = resample_lanczos2_rows(
            (100, 100),
            centered_view((100, 100), (10, 10), 2.0),
            || false,
            |row, source_x, destination| {
                rows_read.set(rows_read.get() + 1);
                smallest_start.set(smallest_start.get().min(source_x));
                widest_row.set(widest_row.get().max(destination.len() / 4));
                let start = usize::try_from(row).unwrap() * 100 * 4
                    + usize::try_from(source_x).unwrap() * 4;
                destination.copy_from_slice(&source[start..start + destination.len()]);
                Ok(())
            },
        )
        .unwrap();
        assert!(rows_read.get() < 100);
        assert!(smallest_start.get() > 0);
        assert!(widest_row.get() < 100);
        assert!(resized.source_bytes < 100 * 100 * CANONICAL_BYTES_PER_PIXEL);
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

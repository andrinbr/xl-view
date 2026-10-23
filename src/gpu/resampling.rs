use std::io;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use wgpu::TextureFormat;
use xl_view::decode::DecodedTileStore;

mod convolution;

use self::convolution::{
    ResampleError, ResamplingScratch, prepare_reused_buffer, resample_lanczos2,
};
use super::mip::rgba16f_texture_budget_bytes;
use super::upload::TextureUploadLayout;
use super::{WorkerContext, lock_unpoisoned};
use crate::diagnostics::panic_payload_message;
use xl_view::units::{bytes_to_mib, milliseconds, u64_from_usize};

const RESAMPLING_DEBOUNCE: Duration = Duration::from_millis(50);

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
    fn spawn(source: Arc<DecodedTileStore>, context: WorkerContext) -> Result<Self, io::Error> {
        let state = Arc::new((Mutex::new(ResamplingWorkerState::default()), Condvar::new()));
        let worker_state = Arc::clone(&state);
        let (result_sender, results) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("xl-view-viewport-resample".to_owned())
            .spawn(move || {
                run_resampling_worker(&source, &context, &worker_state, &result_sender);
            })?;
        Ok(Self {
            state,
            results,
            thread: Some(thread),
        })
    }

    fn scratch_peak_bytes(&self) -> usize {
        let (state, _) = self.state.as_ref();
        lock_unpoisoned(state).scratch_peak_bytes
    }

    fn replace(&self, generation: u64, view: Option<ViewportRequest>) {
        let (state, wake) = self.state.as_ref();
        let mut state = lock_unpoisoned(state);
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
            let mut state = lock_unpoisoned(state);
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
        context: &WorkerContext,
        source: Arc<DecodedTileStore>,
        memory_limit_bytes: u64,
    ) -> Self {
        let mut resampler = Self::fallback(&context.device, memory_limit_bytes);
        match ResamplingWorker::spawn(source, context.clone()) {
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
            |value| format!("{:.2} ms", milliseconds(value)),
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
    context: &WorkerContext,
    state: &Arc<(Mutex<ResamplingWorkerState>, Condvar)>,
    results: &mpsc::Sender<ResamplingWorkResult>,
) {
    let WorkerContext {
        device,
        queue,
        work_ready: notify_ready,
        submission_lock,
    } = context;
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
    let mut state = lock_unpoisoned(state);
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
        let mut locked = lock_unpoisoned(state);
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
    let state = lock_unpoisoned(state);
    !state.shutdown && state.generation == generation
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
        let _submission_guard = lock_unpoisoned(submission_lock);
        if !resampling_job_is_current(state, job.generation) {
            return Err(ResampleError::Cancelled);
        }
        upload_layout.write_stripe(queue, texture, 0, staging, stripe);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::test_utils::request_native_test_gpu;

    pub(super) fn centered_view(
        source: (u32, u32),
        target: (u32, u32),
        scale: f64,
    ) -> ViewportRequest {
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
            phase: ResamplerPhase::Inactive,
        })
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
        let (device, _) = request_native_test_gpu(
            crate::gpu::native_backends(),
            wgpu::Features::empty(),
            "resampling lifecycle test requires a native GPU adapter",
        );
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
        let (device, _) = request_native_test_gpu(
            crate::gpu::native_backends(),
            wgpu::Features::empty(),
            "resampling lifecycle test requires a native GPU adapter",
        );
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
}

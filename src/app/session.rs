//! Image-session state machine for selection, decode completion, cache reuse,
//! folder navigation, and neighbor prefetch. This module returns narrow effects
//! to the application coordinator and never owns Winit windows or GPU state.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use xl_view::decode::{
    DecodeCompletion, DecodeCoordinator, DecodeError, DecodePurpose, DecodeQueueDisposition,
    DecodedImage, ImageKey,
};

use super::decoded_cache::{CachedDecodedImage, DecodedImageCache};
use super::files::{FolderDirection, NeighborPaths, file_name};
use super::milliseconds;
use super::overlay::{DecodeTiming, LoadedImageSummary};

const PREFETCH_DELAY: Duration = Duration::from_millis(250);

trait DecodeQueue {
    fn request_demand(
        &self,
        path: PathBuf,
        key: ImageKey,
        selection_generation: u64,
    ) -> DecodeQueueDisposition;

    fn request_prefetch(
        &self,
        path: PathBuf,
        key: ImageKey,
        prefetch_generation: u64,
        neighbor_index: u8,
        maximum_retained_bytes: u64,
    ) -> DecodeQueueDisposition;

    fn cancel_queued_prefetches_except(&self, prefetch_generation: u64);
    fn contains(&self, key: &ImageKey) -> bool;
    fn shutdown(&mut self);
}

impl DecodeQueue for DecodeCoordinator {
    fn request_demand(
        &self,
        path: PathBuf,
        key: ImageKey,
        selection_generation: u64,
    ) -> DecodeQueueDisposition {
        DecodeCoordinator::request_demand(self, path, key, selection_generation)
    }

    fn request_prefetch(
        &self,
        path: PathBuf,
        key: ImageKey,
        prefetch_generation: u64,
        neighbor_index: u8,
        maximum_retained_bytes: u64,
    ) -> DecodeQueueDisposition {
        DecodeCoordinator::request_prefetch(
            self,
            path,
            key,
            prefetch_generation,
            neighbor_index,
            maximum_retained_bytes,
        )
    }

    fn cancel_queued_prefetches_except(&self, prefetch_generation: u64) {
        DecodeCoordinator::cancel_queued_prefetches_except(self, prefetch_generation);
    }

    fn contains(&self, key: &ImageKey) -> bool {
        DecodeCoordinator::contains(self, key)
    }

    fn shutdown(&mut self) {
        DecodeCoordinator::shutdown(self);
    }
}

/// Owns non-GPU state for one committed image and at most one pending open.
///
/// `current_key`, `current_path`, and `loaded_image` describe the same image
/// and change together only in `commit_install`; `decoded_image` optionally
/// retains that image until a GPU becomes available. `pending_open` is the
/// authoritative in-flight selection. Selection, navigation, and prefetch
/// generations independently invalidate their corresponding asynchronous
/// results, while `current_image_presentation` belongs to the committed image.
pub(super) struct ImageSession {
    current_key: Option<ImageKey>,
    current_image_presentation: ImagePresentationState,
    current_path: Option<PathBuf>,
    decoded_cache: DecodedImageCache,
    decode_queue: Box<dyn DecodeQueue>,
    decoded_image: Option<Arc<DecodedImage>>,
    loaded_image: Option<LoadedImageSummary>,
    navigation_generation: u64,
    pending_open: Option<ImageOpenContext>,
    prefetch_generation: u64,
    prefetch_plan: Option<PrefetchPlan>,
    selection_generation: u64,
    status_message: Option<String>,
}

/// Request identity that moves through selection, decoding, and GPU install.
///
/// A single value remains authoritative for stale-generation checks, moves
/// intact into `PendingImageInstall`, and is destructured only after the GPU
/// installation succeeds.
#[derive(Debug)]
struct ImageOpenContext {
    generation: u64,
    key: ImageKey,
    path: PathBuf,
    started: Instant,
    direction: Option<FolderDirection>,
}

#[derive(Debug)]
pub(super) struct PendingPresentation {
    pub(super) cache_hit: bool,
    pub(super) decode_time: Duration,
    pub(super) direction: Option<FolderDirection>,
    pub(super) generation: u64,
    pub(super) height: u32,
    pub(super) started: Instant,
    pub(super) width: u32,
}

impl PendingPresentation {
    pub(super) const fn request_kind(&self) -> &'static str {
        request_kind(self.direction)
    }
}

const fn request_kind(direction: Option<FolderDirection>) -> &'static str {
    match direction {
        Some(FolderDirection::Previous) => "previous-image navigation",
        Some(FolderDirection::Next) => "next-image navigation",
        None => "image open",
    }
}

#[derive(Debug)]
struct PrefetchPlan {
    generation: u64,
    anchor_key: ImageKey,
    anchor_path: PathBuf,
    deadline: Option<Instant>,
    direction: Option<FolderDirection>,
    remaining: VecDeque<PathBuf>,
    protected: Vec<ImageKey>,
    next_neighbor_index: u8,
}

#[derive(Debug)]
struct PrefetchCandidate {
    path: PathBuf,
    generation: u64,
    neighbor_index: u8,
    protected: Vec<ImageKey>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrefetchStep {
    Continue,
    Stop,
}

impl PrefetchPlan {
    fn pop_candidate(&mut self) -> Option<PrefetchCandidate> {
        let path = self.remaining.pop_front()?;
        let neighbor_index = self.next_neighbor_index;
        self.next_neighbor_index = self.next_neighbor_index.saturating_add(1);
        Some(PrefetchCandidate {
            path,
            generation: self.generation,
            neighbor_index,
            protected: self.protected.clone(),
        })
    }
}

/// First-presentation status for the installed image.
///
/// `Pending(None)` means presentation has not completed but its timing was
/// deliberately cancelled, for example during replacement or shutdown.
#[derive(Debug)]
enum ImagePresentationState {
    Pending(Option<PendingPresentation>),
    Presented,
}

pub(super) struct PendingImageInstall {
    cached: CachedDecodedImage,
    context: ImageOpenContext,
    cache_hit: bool,
}

impl PendingImageInstall {
    pub(super) fn image(&self) -> &Arc<DecodedImage> {
        &self.cached.image
    }

    pub(super) fn path(&self) -> &Path {
        &self.context.path
    }

    pub(super) fn direction(&self) -> Option<FolderDirection> {
        self.context.direction
    }

    pub(super) const fn generation(&self) -> u64 {
        self.context.generation
    }
}

#[allow(clippy::large_enum_variant)] // The payload moves once; boxing would allocate on every image open.
pub(super) enum SelectionEffect {
    Install(PendingImageInstall),
    Pending,
    StatusChanged,
}

#[allow(clippy::large_enum_variant)] // The payload moves once; boxing would allocate on every completed demand.
pub(super) enum DecodeEffect {
    Install(PendingImageInstall),
    StatusChanged,
    None,
}

pub(super) struct NavigationRequest {
    pub(super) generation: u64,
    pub(super) direction: FolderDirection,
    pub(super) source_path: PathBuf,
}

pub(super) enum NavigationEffect {
    Select {
        path: PathBuf,
        direction: FolderDirection,
    },
    StatusChanged,
    None,
}

pub(super) struct NeighborLookupRequest {
    pub(super) generation: u64,
    pub(super) anchor_key: ImageKey,
    pub(super) anchor_path: PathBuf,
}

impl ImageSession {
    pub(super) fn new(maximum_cache_bytes: u64, decode_coordinator: DecodeCoordinator) -> Self {
        Self::with_decode_queue(maximum_cache_bytes, decode_coordinator)
    }

    fn with_decode_queue(
        maximum_cache_bytes: u64,
        decode_queue: impl DecodeQueue + 'static,
    ) -> Self {
        Self {
            current_key: None,
            current_image_presentation: ImagePresentationState::Pending(None),
            current_path: None,
            decoded_cache: DecodedImageCache::new(maximum_cache_bytes),
            decode_queue: Box::new(decode_queue),
            decoded_image: None,
            loaded_image: None,
            navigation_generation: 0,
            pending_open: None,
            prefetch_generation: 0,
            prefetch_plan: None,
            selection_generation: 0,
            status_message: None,
        }
    }

    pub(super) fn select_path(
        &mut self,
        path: PathBuf,
        direction: Option<FolderDirection>,
    ) -> SelectionEffect {
        self.navigation_generation = self.navigation_generation.wrapping_add(1);
        self.cancel_pending_presentation();
        self.selection_generation = self.selection_generation.wrapping_add(1);
        self.prefetch_generation = self.prefetch_generation.wrapping_add(1);
        self.prefetch_plan = None;
        self.decode_queue
            .cancel_queued_prefetches_except(self.prefetch_generation);
        let generation = self.selection_generation;
        let display_name = file_name(&path);
        let started = Instant::now();
        let key = match ImageKey::from_path(&path) {
            Ok(key) => key,
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    purpose = "demand",
                    selection_generation = generation,
                    direction = ?direction,
                    %error,
                    "cannot identify image source"
                );
                self.status_message = Some(format!("Cannot open {display_name}: {error}"));
                self.pending_open = None;
                return SelectionEffect::StatusChanged;
            }
        };
        let context = ImageOpenContext {
            generation,
            key: key.clone(),
            path: path.clone(),
            started,
            direction,
        };
        if let Some(image) = self.decoded_cache.get(&key) {
            tracing::debug!(
                path = %path.display(),
                purpose = "demand",
                selection_generation = generation,
                direction = ?direction,
                "decoded image cache hit"
            );
            self.pending_open = Some(context);
            SelectionEffect::Install(
                self.prepare_current_install(image, true)
                    .expect("the cached selection was just installed"),
            )
        } else {
            let disposition = self.decode_queue.request_demand(path, key, generation);
            tracing::debug!(
                path = %context.path.display(),
                purpose = "demand",
                selection_generation = generation,
                direction = ?direction,
                ?disposition,
                "queued image demand"
            );
            self.pending_open = Some(context);
            self.status_message = Some(format!("Opening {display_name}…"));
            SelectionEffect::Pending
        }
    }

    fn prepare_current_install(
        &mut self,
        cached: CachedDecodedImage,
        cache_hit: bool,
    ) -> Option<PendingImageInstall> {
        Some(PendingImageInstall {
            cached,
            context: self.pending_open.take()?,
            cache_hit,
        })
    }

    pub(super) fn commit_install(
        &mut self,
        install: PendingImageInstall,
        retain_decoded_image: bool,
    ) {
        let PendingImageInstall {
            cached,
            context,
            cache_hit,
        } = install;
        let ImageOpenContext {
            generation,
            key,
            path,
            started,
            direction,
        } = context;
        let decode_time = cached.decode_time;
        let dimensions = (cached.image.width, cached.image.height);
        let decode_timing = if cache_hit {
            DecodeTiming::CacheHit(decode_time)
        } else {
            DecodeTiming::Measured(decode_time)
        };
        let summary =
            LoadedImageSummary::from_decoded(Some(path.as_path()), &cached.image, decode_timing);
        let decoded_image = retain_decoded_image.then(|| Arc::clone(&cached.image));
        tracing::info!(
            path = %path.display(),
            purpose = "demand",
            request = request_kind(direction),
            selection_generation = generation,
            cache_hit,
            width = dimensions.0,
            height = dimensions.1,
            decode_ms = milliseconds(decode_time),
            install_ready_ms = milliseconds(started.elapsed()),
            ?direction,
            "image installation completed"
        );
        if direction.is_none() {
            self.decoded_cache.clear();
        }
        self.decoded_cache.commit_current(key.clone(), cached);
        self.current_key = Some(key.clone());
        self.current_path = Some(path.clone());
        self.loaded_image = Some(summary);
        self.status_message = None;
        self.current_image_presentation =
            ImagePresentationState::Pending(Some(PendingPresentation {
                cache_hit,
                decode_time,
                direction,
                generation,
                height: dimensions.1,
                started,
                width: dimensions.0,
            }));
        self.decoded_image = decoded_image;
        self.schedule_neighbor_prefetch(key, path, direction);
    }

    pub(super) fn reject_install(
        &mut self,
        install: &PendingImageInstall,
        error: &impl std::fmt::Display,
    ) {
        self.status_message = Some(format!(
            "Cannot display {}: {error}",
            file_name(install.path())
        ));
        self.cancel_pending_presentation();
    }

    fn schedule_neighbor_prefetch(
        &mut self,
        anchor_key: ImageKey,
        anchor_path: PathBuf,
        direction: Option<FolderDirection>,
    ) {
        if self.decoded_cache.available_for_prefetch(&[]) == 0 {
            self.prefetch_plan = None;
            return;
        }
        self.prefetch_plan = Some(PrefetchPlan {
            generation: self.prefetch_generation,
            anchor_key,
            anchor_path,
            deadline: Some(Instant::now() + PREFETCH_DELAY),
            direction,
            remaining: VecDeque::new(),
            protected: Vec::new(),
            next_neighbor_index: 0,
        });
    }

    pub(super) fn prefetch_deadline(&self) -> Option<Instant> {
        self.prefetch_plan.as_ref().and_then(|plan| plan.deadline)
    }

    pub(super) fn begin_neighbor_lookup(&mut self) -> Option<NeighborLookupRequest> {
        let plan = self.prefetch_plan.as_mut()?;
        plan.deadline.take()?;
        Some(NeighborLookupRequest {
            generation: plan.generation,
            anchor_key: plan.anchor_key.clone(),
            anchor_path: plan.anchor_path.clone(),
        })
    }

    pub(super) fn neighbor_lookup_start_failed(
        &mut self,
        request: &NeighborLookupRequest,
        error: &impl std::fmt::Display,
    ) {
        tracing::warn!(
            path = %request.anchor_path.display(),
            purpose = "prefetch",
            prefetch_generation = request.generation,
            %error,
            "cannot start neighboring-image lookup"
        );
        if self.prefetch_plan.as_ref().is_some_and(|plan| {
            plan.generation == request.generation && plan.anchor_key == request.anchor_key
        }) {
            self.prefetch_plan = None;
        }
    }

    pub(super) fn handle_neighbor_paths(
        &mut self,
        generation: u64,
        anchor_key: &ImageKey,
        result: Result<NeighborPaths, String>,
    ) {
        let Some(plan) = self.prefetch_plan.as_mut() else {
            return;
        };
        if plan.generation != generation
            || &plan.anchor_key != anchor_key
            || self.current_key.as_ref() != Some(anchor_key)
        {
            return;
        }
        let neighbors = match result {
            Ok(neighbors) => neighbors,
            Err(error) => {
                tracing::debug!(
                    path = %plan.anchor_path.display(),
                    purpose = "prefetch",
                    prefetch_generation = generation,
                    %error,
                    "cannot inspect neighboring images"
                );
                self.prefetch_plan = None;
                return;
            }
        };
        plan.remaining = ordered_neighbor_paths(neighbors, plan.direction)
            .into_iter()
            .flatten()
            .collect();
        self.queue_next_neighbor_prefetch();
    }

    fn queue_next_neighbor_prefetch(&mut self) {
        loop {
            let Some(candidate) = self
                .prefetch_plan
                .as_mut()
                .and_then(PrefetchPlan::pop_candidate)
            else {
                self.prefetch_plan = None;
                return;
            };
            if self.queue_neighbor_prefetch_candidate(candidate) == PrefetchStep::Stop {
                return;
            }
        }
    }

    fn queue_neighbor_prefetch_candidate(&mut self, candidate: PrefetchCandidate) -> PrefetchStep {
        let PrefetchCandidate {
            path,
            generation,
            neighbor_index,
            protected,
        } = candidate;
        let key = match ImageKey::from_path(&path) {
            Ok(key) => key,
            Err(error) => {
                tracing::debug!(path = %path.display(), %error, "cannot identify prefetch source");
                return PrefetchStep::Continue;
            }
        };
        if self.current_key.as_ref() == Some(&key) {
            return PrefetchStep::Continue;
        }
        if self.decoded_cache.contains(&key) {
            if let Some(plan) = self.prefetch_plan.as_mut() {
                plan.protected.push(key);
            }
            return PrefetchStep::Continue;
        }
        if self.decode_queue.contains(&key) {
            return PrefetchStep::Continue;
        }
        let available = self.decoded_cache.available_for_prefetch(&protected);
        if available == 0 {
            self.prefetch_plan = None;
            return PrefetchStep::Stop;
        }
        let disposition = self.decode_queue.request_prefetch(
            path.clone(),
            key,
            generation,
            neighbor_index,
            available,
        );
        tracing::debug!(
            path = %path.display(),
            purpose = "prefetch",
            prefetch_generation = generation,
            neighbor_index,
            maximum_retained_bytes = available,
            ?disposition,
            "queued neighbor prefetch"
        );
        if matches!(disposition, DecodeQueueDisposition::Queued) {
            PrefetchStep::Stop
        } else {
            PrefetchStep::Continue
        }
    }

    pub(super) fn begin_navigation(
        &mut self,
        direction: FolderDirection,
    ) -> Option<NavigationRequest> {
        let Some(current_path) = self.current_path.clone() else {
            self.status_message = Some("Open an image before navigating its folder".to_owned());
            return None;
        };
        self.navigation_generation = self.navigation_generation.wrapping_add(1);
        let generation = self.navigation_generation;
        self.status_message = Some(format!("Looking for the {} image…", direction.as_str()));
        Some(NavigationRequest {
            generation,
            direction,
            source_path: current_path,
        })
    }

    pub(super) fn navigation_start_failed(
        &mut self,
        request: &NavigationRequest,
        error: &impl std::fmt::Display,
    ) {
        tracing::warn!(
            path = %request.source_path.display(),
            direction = ?request.direction,
            %error,
            "cannot start folder navigation"
        );
        if request.generation == self.navigation_generation {
            self.status_message = Some(format!("Cannot inspect the image folder: {error}"));
        }
    }

    pub(super) fn handle_navigation_result(
        &mut self,
        generation: u64,
        direction: FolderDirection,
        source_path: &Path,
        result: Result<Option<PathBuf>, String>,
    ) -> NavigationEffect {
        if generation != self.navigation_generation {
            return NavigationEffect::None;
        }
        match result {
            Ok(Some(path)) => NavigationEffect::Select { path, direction },
            Ok(None) => {
                self.status_message =
                    Some(format!("No {} image in this folder", direction.as_str()));
                NavigationEffect::StatusChanged
            }
            Err(error) => {
                tracing::warn!(
                    path = %source_path.display(),
                    ?direction,
                    %error,
                    "cannot inspect the image folder"
                );
                self.status_message = Some(format!("Cannot inspect the image folder: {error}"));
                NavigationEffect::StatusChanged
            }
        }
    }

    pub(super) fn handle_decode_completion(
        &mut self,
        completion: DecodeCompletion,
    ) -> DecodeEffect {
        match completion.purpose {
            DecodePurpose::Demand {
                selection_generation,
            } => self.handle_demand_decode_completion(completion, selection_generation),
            DecodePurpose::Prefetch {
                prefetch_generation,
                neighbor_index,
                maximum_retained_bytes,
            } => {
                self.handle_prefetch_decode_completion(
                    completion,
                    prefetch_generation,
                    neighbor_index,
                    maximum_retained_bytes,
                );
                DecodeEffect::None
            }
        }
    }

    fn handle_demand_decode_completion(
        &mut self,
        completion: DecodeCompletion,
        selection_generation: u64,
    ) -> DecodeEffect {
        let is_current = self.pending_open.as_ref().is_some_and(|context| {
            context.generation == selection_generation && context.key == completion.key
        });
        let direction = self
            .pending_open
            .as_ref()
            .filter(|_| is_current)
            .and_then(|context| context.direction);
        match completion.result {
            Ok(image) => {
                let request_to_decode_complete = self
                    .pending_open
                    .as_ref()
                    .filter(|_| is_current)
                    .map_or(Duration::ZERO, |context| context.started.elapsed());
                if is_current {
                    tracing::info!(
                        path = %completion.path.display(),
                        purpose = "demand",
                        selection_generation,
                        direction = ?direction,
                        width = image.width,
                        height = image.height,
                        bytes = image.memory_cost_bytes,
                        color_encoding = ?image.metadata.color_encoding,
                        decode_ms = milliseconds(completion.decode_time),
                        request_to_decode_complete_ms = milliseconds(request_to_decode_complete),
                        "image decode completed"
                    );
                } else {
                    tracing::debug!(
                        path = %completion.path.display(),
                        purpose = "demand",
                        selection_generation,
                        width = image.width,
                        height = image.height,
                        bytes = image.memory_cost_bytes,
                        color_encoding = ?image.metadata.color_encoding,
                        decode_ms = milliseconds(completion.decode_time),
                        "stale image decode completed"
                    );
                }
                let cached = CachedDecodedImage::new(image, completion.decode_time);
                if is_current {
                    self.prepare_current_install(cached, false)
                        .map_or(DecodeEffect::None, DecodeEffect::Install)
                } else {
                    let protected = self
                        .prefetch_plan
                        .as_ref()
                        .map_or_else(Vec::new, |plan| plan.protected.clone());
                    let _ = self
                        .decoded_cache
                        .admit_inactive(completion.key, cached, &protected);
                    DecodeEffect::None
                }
            }
            Err(error) if is_current => {
                self.handle_current_decode_failure(
                    &completion.path,
                    selection_generation,
                    direction,
                    completion.decode_time,
                    &error,
                );
                DecodeEffect::StatusChanged
            }
            Err(error) => {
                tracing::debug!(
                    path = %completion.path.display(),
                    purpose = "demand",
                    selection_generation,
                    decode_ms = milliseconds(completion.decode_time),
                    %error,
                    "stale demand decode failed"
                );
                DecodeEffect::None
            }
        }
    }

    fn handle_prefetch_decode_completion(
        &mut self,
        completion: DecodeCompletion,
        prefetch_generation: u64,
        neighbor_index: u8,
        maximum_retained_bytes: u64,
    ) {
        let Some(plan) = self.prefetch_plan.as_ref() else {
            return;
        };
        if prefetch_generation != self.prefetch_generation
            || plan.generation != prefetch_generation
            || self.current_key.as_ref() != Some(&plan.anchor_key)
        {
            return;
        }
        match completion.result {
            Ok(image) => {
                let protected = self
                    .prefetch_plan
                    .as_ref()
                    .map_or_else(Vec::new, |plan| plan.protected.clone());
                let key = completion.key;
                let cached = CachedDecodedImage::new(image, completion.decode_time);
                let admitted = self
                    .decoded_cache
                    .admit_inactive(key.clone(), cached, &protected);
                if admitted && let Some(plan) = self.prefetch_plan.as_mut() {
                    plan.protected.push(key);
                }
                tracing::debug!(
                    path = %completion.path.display(),
                    purpose = "prefetch",
                    prefetch_generation,
                    neighbor_index,
                    maximum_retained_bytes,
                    decode_ms = milliseconds(completion.decode_time),
                    admitted,
                    "neighbor prefetch completed"
                );
            }
            Err(error) => {
                tracing::debug!(
                    path = %completion.path.display(),
                    purpose = "prefetch",
                    prefetch_generation,
                    neighbor_index,
                    maximum_retained_bytes,
                    decode_ms = milliseconds(completion.decode_time),
                    %error,
                    "neighbor prefetch failed"
                );
            }
        }
        self.queue_next_neighbor_prefetch();
    }

    fn handle_current_decode_failure(
        &mut self,
        path: &Path,
        selection_generation: u64,
        direction: Option<FolderDirection>,
        decode_time: Duration,
        error: &DecodeError,
    ) {
        let name = file_name(path);
        tracing::warn!(
            path = %path.display(),
            purpose = "demand",
            selection_generation,
            direction = ?direction,
            decode_ms = milliseconds(decode_time),
            %error,
            "image decode failed"
        );
        self.pending_open = None;
        self.status_message = Some(format!("Cannot open {name}: {error}"));
        self.cancel_pending_presentation();
    }

    pub(super) fn loaded_image(&self) -> Option<&LoadedImageSummary> {
        self.loaded_image.as_ref()
    }

    pub(super) const fn has_loaded_image(&self) -> bool {
        self.loaded_image.is_some()
    }

    pub(super) fn current_path(&self) -> Option<&Path> {
        self.current_path.as_deref()
    }

    pub(super) fn status_message(&self) -> Option<&str> {
        self.status_message.as_deref()
    }

    pub(super) fn set_status_message(&mut self, message: impl Into<String>) {
        self.status_message = Some(message.into());
    }

    pub(super) fn take_decoded_image(&mut self) -> Option<Arc<DecodedImage>> {
        self.decoded_image.take()
    }

    pub(super) fn finish_presentation(&mut self) -> Option<(&Path, PendingPresentation)> {
        let ImagePresentationState::Pending(pending) = &mut self.current_image_presentation else {
            return None;
        };
        let opening = pending.take()?;
        let path = self.current_path.as_deref()?;
        self.current_image_presentation = ImagePresentationState::Presented;
        Some((path, opening))
    }

    pub(super) fn cancel_pending_presentation(&mut self) {
        if let ImagePresentationState::Pending(pending) = &mut self.current_image_presentation {
            *pending = None;
        }
    }

    pub(super) fn current_image_was_presented(&self) -> bool {
        matches!(
            self.current_image_presentation,
            ImagePresentationState::Presented
        )
    }

    pub(super) fn shutdown(&mut self) {
        self.decode_queue.shutdown();
    }
}

fn ordered_neighbor_paths(
    neighbors: NeighborPaths,
    direction: Option<FolderDirection>,
) -> [Option<PathBuf>; 2] {
    match direction {
        Some(FolderDirection::Previous) => [neighbors.previous, neighbors.next],
        Some(FolderDirection::Next) | None => [neighbors.next, neighbors.previous],
    }
}

#[cfg(test)]
mod tests;

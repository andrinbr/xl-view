use std::collections::HashSet;
use std::sync::Mutex;
use std::time::SystemTime;

use super::*;
use crate::test_utils::fixture_path;
use xl_view::decode::{DecodeLimits, decode_file};

fn permission_denied() -> FolderReadError {
    FolderReadError::new(
        Path::new("/photos"),
        std::io::Error::new(std::io::ErrorKind::PermissionDenied, "permission denied"),
    )
}

#[derive(Debug)]
struct FakeDecodeState {
    demand_disposition: DecodeQueueDisposition,
    demands: Vec<(PathBuf, ImageKey, u64)>,
    prefetches: Vec<(PathBuf, ImageKey, u64, u8, u64)>,
    cancellations: Vec<u64>,
    contained: HashSet<ImageKey>,
    renditions: Vec<Rendition>,
    shutdown: bool,
}

#[derive(Clone, Debug)]
struct FakeDecodeQueue {
    state: Arc<Mutex<FakeDecodeState>>,
}

impl FakeDecodeQueue {
    fn new(demand_disposition: DecodeQueueDisposition) -> Self {
        Self {
            state: Arc::new(Mutex::new(FakeDecodeState {
                demand_disposition,
                demands: Vec::new(),
                prefetches: Vec::new(),
                cancellations: Vec::new(),
                contained: HashSet::new(),
                renditions: Vec::new(),
                shutdown: false,
            })),
        }
    }
}

impl DecodeQueue for FakeDecodeQueue {
    fn request_demand(
        &self,
        path: PathBuf,
        key: ImageKey,
        selection_generation: u64,
    ) -> DecodeQueueDisposition {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.demands.push((path, key, selection_generation));
        state.demand_disposition
    }

    fn request_prefetch(
        &self,
        path: PathBuf,
        key: ImageKey,
        prefetch_generation: u64,
        neighbor_index: u8,
        maximum_retained_bytes: u64,
    ) -> DecodeQueueDisposition {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .prefetches
            .push((
                path,
                key,
                prefetch_generation,
                neighbor_index,
                maximum_retained_bytes,
            ));
        DecodeQueueDisposition::Queued
    }

    fn cancel_queued_prefetches_except(&self, prefetch_generation: u64) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cancellations
            .push(prefetch_generation);
    }

    fn contains(&self, key: &ImageKey) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contained
            .contains(key)
    }

    fn set_rendition(&self, rendition: Rendition) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .renditions
            .push(rendition);
    }

    fn shutdown(&mut self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .shutdown = true;
    }
}

fn decoded_fixture() -> (PathBuf, ImageKey, Arc<DecodedImage>) {
    let path = fixture_path("ramp-pq-1000.jxl");
    let key = ImageKey::from_path(&path).unwrap();
    let image = decode_file(&path, DecodeLimits::from_memory_ceiling_mib(3)).unwrap();
    (path, key, image)
}

fn session_with_queued_neighbors() -> (
    ImageSession,
    Arc<Mutex<FakeDecodeState>>,
    Arc<DecodedImage>,
    PathBuf,
    PathBuf,
) {
    let (anchor_path, anchor_key, image) = decoded_fixture();
    let preferred = fixture_path("alpha.jxl");
    let secondary = fixture_path("grayscale.jxl");
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    session.decoded_cache.commit_current(
        anchor_key.clone(),
        CachedDecodedImage::new(Arc::clone(&image), Duration::from_millis(5)),
    );
    session.current_key = Some(anchor_key.clone());
    session.current_path = Some(anchor_path.clone());
    session.prefetch_generation = 7;
    session.prefetch_plan = Some(PrefetchPlan {
        generation: 7,
        anchor_key: anchor_key.clone(),
        anchor_path,
        deadline: None,
        direction: Some(FolderDirection::Next),
        remaining: VecDeque::new(),
        protected: Vec::new(),
        next_neighbor_index: 0,
    });
    session.handle_neighbor_paths(
        7,
        &anchor_key,
        Ok(NeighborPaths {
            previous: Some(secondary.clone()),
            next: Some(preferred.clone()),
        }),
    );
    (session, queue_state, image, preferred, secondary)
}

fn session_with_pending_neighbor_lookup() -> (
    ImageSession,
    NeighborLookupRequest,
    Arc<Mutex<FakeDecodeState>>,
) {
    let (path, key, image) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    session.decoded_cache.commit_current(
        key.clone(),
        CachedDecodedImage::new(image, Duration::from_millis(5)),
    );
    session.current_key = Some(key.clone());
    session.current_path = Some(path.clone());
    session.schedule_neighbor_prefetch(key, path, None);
    let request = session
        .begin_neighbor_lookup()
        .expect("the current prefetch plan should start a lookup");
    (session, request, queue_state)
}

#[test]
fn current_demand_failure_clears_pending_state_and_reports_the_error() {
    let (path, key, _) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    assert!(matches!(
        session.select_path(path.clone(), None),
        SelectionEffect::Pending
    ));

    let effect = session.handle_decode_completion(DecodeCompletion {
        decode_time: Duration::from_millis(10),
        key,
        path,
        purpose: DecodePurpose::Demand {
            selection_generation: 1,
        },
        result: Err(DecodeError::Canceled),
    });

    assert!(matches!(effect, DecodeEffect::StatusChanged));
    assert!(session.pending_open.is_none());
    assert!(
        session
            .status_message()
            .is_some_and(|message| message.contains("Failed to open"))
    );
}

#[test]
fn invalid_source_path_reports_status_without_queueing_decode() {
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);

    assert!(matches!(
        session.select_path(fixture_path("missing.jxl"), None),
        SelectionEffect::StatusChanged
    ));
    assert!(session.pending_open.is_none());
    assert!(
        queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .demands
            .is_empty()
    );
}

#[test]
fn stale_demand_failure_does_not_disturb_the_current_selection() {
    let (path, key, _) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    assert!(matches!(
        session.select_path(path.clone(), None),
        SelectionEffect::Pending
    ));
    assert!(matches!(
        session.select_path(path.clone(), None),
        SelectionEffect::Pending
    ));
    let status = session.status_message.clone();

    let effect = session.handle_decode_completion(DecodeCompletion {
        decode_time: Duration::from_millis(10),
        key,
        path,
        purpose: DecodePurpose::Demand {
            selection_generation: 1,
        },
        result: Err(DecodeError::Canceled),
    });

    assert!(matches!(effect, DecodeEffect::None));
    assert_eq!(session.status_message, status);
    assert!(
        session
            .pending_open
            .as_ref()
            .is_some_and(|context| context.generation == 2)
    );
}

#[test]
fn stale_demand_completion_is_cached_without_replacing_the_current_selection() {
    let (path, current_key, image) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    session.pending_open = Some(ImageOpenContext {
        generation: 2,
        key: current_key,
        path: path.clone(),
        started: Instant::now(),
        direction: None,
        rendition_swap: false,
    });
    let stale_key = ImageKey {
        normalized_path: path.clone(),
        source_len: 1,
        source_modified: SystemTime::UNIX_EPOCH,
    };

    let effect = session.handle_decode_completion(DecodeCompletion {
        decode_time: Duration::from_millis(10),
        key: stale_key.clone(),
        path,
        purpose: DecodePurpose::Demand {
            selection_generation: 1,
        },
        result: Ok(image),
    });

    assert!(matches!(effect, DecodeEffect::None));
    assert_eq!(
        session
            .pending_open
            .as_ref()
            .map(|context| context.generation),
        Some(2)
    );
    assert!(session.decoded_cache.contains(&stale_key));
}

#[test]
fn stale_neighbor_lookup_cannot_queue_prefetch_work() {
    let (path, key, _) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    session.current_key = Some(key.clone());
    session.prefetch_plan = Some(PrefetchPlan {
        generation: 7,
        anchor_key: key.clone(),
        anchor_path: path,
        deadline: None,
        direction: None,
        remaining: VecDeque::new(),
        protected: Vec::new(),
        next_neighbor_index: 0,
    });

    session.handle_neighbor_paths(
        6,
        &key,
        Ok(NeighborPaths {
            previous: Some(PathBuf::from("previous.jxl")),
            next: Some(PathBuf::from("next.jxl")),
        }),
    );

    assert!(
        session
            .prefetch_plan
            .as_ref()
            .is_some_and(|plan| plan.remaining.is_empty())
    );
    assert!(
        queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .prefetches
            .is_empty()
    );
}

#[test]
fn neighbor_lookup_start_failure_clears_the_current_plan() {
    let (mut session, request, _) = session_with_pending_neighbor_lookup();

    session.neighbor_lookup_start_failed(&request, &"thread unavailable");

    assert!(session.prefetch_plan.is_none());
}

#[test]
fn neighbor_lookup_failure_is_silent_and_clears_the_current_plan() {
    let (mut session, request, queue_state) = session_with_pending_neighbor_lookup();
    session.set_status_message("ready");

    session.handle_neighbor_paths(
        request.generation,
        &request.anchor_key,
        Err(permission_denied()),
    );

    assert!(session.prefetch_plan.is_none());
    assert_eq!(session.status_message(), Some("ready"));
    assert!(
        queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .prefetches
            .is_empty()
    );
}

type QueuedPrefetch = (PathBuf, ImageKey, u64, u8, u64);

fn swap_neighbors() -> NeighborPaths {
    NeighborPaths {
        previous: Some(fixture_path("grayscale.jxl")),
        next: Some(fixture_path("alpha.jxl")),
    }
}

/// Shows an SDR gain map image whose plan has queued its first neighbor,
/// then swaps in the HDR rendition. Returns that first prefetch.
fn swap_after_first_prefetch(
    session: &mut ImageSession,
    queue_state: &Mutex<FakeDecodeState>,
) -> QueuedPrefetch {
    let (path, _, sdr) = gain_map_fixture(Rendition::Sdr);
    let (_, _, hdr) = gain_map_fixture(Rendition::Hdr);
    show(session, queue_state, &path, &sdr);
    let request = session.begin_neighbor_lookup().unwrap();
    session.handle_neighbor_paths(
        request.generation,
        &request.anchor_key,
        Ok(swap_neighbors()),
    );
    let first = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .prefetches[0]
        .clone();
    assert!(sync_queues_swap(session, queue_state, Rendition::Hdr));
    let DecodeEffect::Install(install) = complete_demand(session, queue_state, Ok(hdr)) else {
        panic!("the swap should be ready to install");
    };
    session.commit_install(install, false);
    first
}

fn complete_prefetch(
    session: &mut ImageSession,
    prefetch: &QueuedPrefetch,
    result: xl_view::decode::DecodeResult,
) {
    session.handle_decode_completion(DecodeCompletion {
        decode_time: Duration::from_millis(10),
        key: prefetch.1.clone(),
        path: prefetch.0.clone(),
        purpose: DecodePurpose::Prefetch {
            prefetch_generation: prefetch.2,
            neighbor_index: prefetch.3,
            maximum_retained_bytes: prefetch.4,
        },
        result,
    });
}

#[test]
fn a_replaced_plans_prefetch_before_the_new_lookup_is_discarded() {
    let (_, _, neighbor) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(256 * 1024 * 1024, queue);
    let first = swap_after_first_prefetch(&mut session, &queue_state);

    complete_prefetch(&mut session, &first, Ok(neighbor));

    // The result belongs to the replaced plan: it neither ends the new plan
    // nor enters the cache.
    let request = session
        .begin_neighbor_lookup()
        .expect("the swap's plan should still look up its neighbors");
    assert!(!session.decoded_cache.contains(&first.1));
    session.handle_neighbor_paths(
        request.generation,
        &request.anchor_key,
        Ok(swap_neighbors()),
    );
    let state = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(state.cancellations.contains(&session.prefetch_generation));
    assert_eq!(state.prefetches.len(), 2);
    assert_eq!(state.prefetches[1].0, first.0);
    assert_eq!(state.prefetches[1].2, session.prefetch_generation);
}

#[test]
fn a_replaced_plans_running_prefetch_cannot_end_the_new_plan() {
    let (_, _, neighbor) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(256 * 1024 * 1024, queue);
    let first = swap_after_first_prefetch(&mut session, &queue_state);
    // The first prefetch is still running when the new lookup returns.
    queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contained
        .insert(first.1.clone());
    let request = session.begin_neighbor_lookup().unwrap();
    session.handle_neighbor_paths(
        request.generation,
        &request.anchor_key,
        Ok(swap_neighbors()),
    );
    let second = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .prefetches[1]
        .clone();
    assert_ne!(second.0, first.0);

    complete_prefetch(&mut session, &first, Ok(Arc::clone(&neighbor)));
    complete_prefetch(&mut session, &second, Ok(neighbor));

    assert!(
        session.decoded_cache.contains(&second.1),
        "the new plan's own prefetch should be cached"
    );
    // Only the replaced plan's result is lost.
    assert!(!session.decoded_cache.contains(&first.1));
}

#[test]
fn a_rendition_change_prefetches_evicted_neighbors_again() {
    let (path, _, image) = decoded_fixture();
    let (neighbor_path, neighbor_key, sdr_neighbor) = gain_map_fixture(Rendition::Sdr);
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(256 * 1024 * 1024, queue);
    show(&mut session, &queue_state, &path, &image);
    assert!(session.decoded_cache.admit_inactive(
        neighbor_key.clone(),
        CachedDecodedImage::new(sdr_neighbor, Duration::from_millis(5)),
        &[],
    ));
    // Prefetching for the shown image has finished.
    session.prefetch_plan = None;

    // The shown image has no gain map, so nothing is swapped.
    assert!(!sync_queues_swap(
        &mut session,
        &queue_state,
        Rendition::Hdr
    ));

    assert!(!session.decoded_cache.contains(&neighbor_key));
    let request = session
        .begin_neighbor_lookup()
        .expect("the rendition change should start a new plan");
    session.handle_neighbor_paths(
        request.generation,
        &request.anchor_key,
        Ok(NeighborPaths {
            previous: None,
            next: Some(neighbor_path),
        }),
    );
    let state = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(state.prefetches.len(), 1);
    assert_eq!(state.prefetches[0].1, neighbor_key);
}

#[test]
fn a_pending_swap_holds_back_the_neighbor_lookup() {
    let (path, _, sdr) = gain_map_fixture(Rendition::Sdr);
    let (_, _, hdr) = gain_map_fixture(Rendition::Hdr);
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(256 * 1024 * 1024, queue);
    show(&mut session, &queue_state, &path, &sdr);
    // Prefetching for the shown image has finished.
    session.prefetch_plan = None;

    // The display change starts a plan and queues the swap. A lookup now
    // would queue a prefetch that the decode thread starts as soon as the
    // swap is decoded, and whose result the swap's own plan discards.
    assert!(sync_queues_swap(&mut session, &queue_state, Rendition::Hdr));
    assert!(session.prefetch_plan.is_some());
    assert!(session.prefetch_deadline().is_none());
    assert!(session.begin_neighbor_lookup().is_none());

    let DecodeEffect::Install(install) = complete_demand(&mut session, &queue_state, Ok(hdr))
    else {
        panic!("the swap should be ready to install");
    };
    session.commit_install(install, false);
    let request = session
        .begin_neighbor_lookup()
        .expect("the installed swap should look up its neighbors");
    session.handle_neighbor_paths(
        request.generation,
        &request.anchor_key,
        Ok(swap_neighbors()),
    );
    let state = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(state.prefetches.len(), 1);
    assert_eq!(state.prefetches[0].0, swap_neighbors().next.unwrap());
    assert_eq!(state.prefetches[0].2, session.prefetch_generation);
}

#[test]
fn a_failed_swap_lets_the_waiting_plan_look_up_its_neighbors() {
    let (path, _, sdr) = gain_map_fixture(Rendition::Sdr);
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(256 * 1024 * 1024, queue);
    show(&mut session, &queue_state, &path, &sdr);
    session.prefetch_plan = None;
    assert!(sync_queues_swap(&mut session, &queue_state, Rendition::Hdr));
    assert!(session.begin_neighbor_lookup().is_none());

    assert!(matches!(
        complete_demand(&mut session, &queue_state, Err(DecodeError::Canceled)),
        DecodeEffect::None
    ));

    assert!(session.prefetch_deadline().is_some());
    assert!(session.begin_neighbor_lookup().is_some());
}

#[test]
fn prefetch_failure_is_silent_and_queues_the_second_neighbor() {
    let (mut session, queue_state, _, preferred, secondary) = session_with_queued_neighbors();
    session.set_status_message("ready");
    let first = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .prefetches[0]
        .clone();

    assert!(matches!(
        session.handle_decode_completion(DecodeCompletion {
            decode_time: Duration::from_millis(10),
            key: first.1,
            path: first.0,
            purpose: DecodePurpose::Prefetch {
                prefetch_generation: first.2,
                neighbor_index: first.3,
                maximum_retained_bytes: first.4,
            },
            result: Err(DecodeError::Canceled),
        }),
        DecodeEffect::None
    ));

    assert_eq!(session.status_message(), Some("ready"));
    let state = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(state.prefetches.len(), 2);
    assert_eq!(state.prefetches[0].0, preferred);
    assert_eq!(state.prefetches[1].0, secondary);
    assert_eq!(state.prefetches[1].3, 1);
}

#[test]
fn successful_neighbor_prefetches_are_admitted_sequentially() {
    let (mut session, queue_state, image, preferred, secondary) = session_with_queued_neighbors();
    let first = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .prefetches[0]
        .clone();
    let preferred_key = first.1.clone();
    assert_eq!(first.0, preferred);

    session.handle_decode_completion(DecodeCompletion {
        decode_time: Duration::from_millis(10),
        key: first.1,
        path: first.0,
        purpose: DecodePurpose::Prefetch {
            prefetch_generation: first.2,
            neighbor_index: first.3,
            maximum_retained_bytes: first.4,
        },
        result: Ok(Arc::clone(&image)),
    });

    assert!(session.decoded_cache.contains(&preferred_key));
    let second = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .prefetches[1]
        .clone();
    assert_eq!(second.0, secondary);
    assert!(second.4 < first.4);
    let secondary_key = second.1.clone();
    session.handle_decode_completion(DecodeCompletion {
        decode_time: Duration::from_millis(10),
        key: second.1,
        path: second.0,
        purpose: DecodePurpose::Prefetch {
            prefetch_generation: second.2,
            neighbor_index: second.3,
            maximum_retained_bytes: second.4,
        },
        result: Ok(image),
    });

    assert!(session.decoded_cache.contains(&preferred_key));
    assert!(session.decoded_cache.contains(&secondary_key));
    assert!(session.prefetch_plan.is_none());
}

#[test]
fn demand_selection_preserves_prefetch_promotion() {
    let (path, key, _) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Promoted);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);

    assert!(matches!(
        session.select_path(path.clone(), Some(FolderDirection::Next)),
        SelectionEffect::Pending
    ));

    let state = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(state.cancellations, [1]);
    assert_eq!(state.demands.len(), 1);
    assert_eq!(state.demands[0].0, path);
    assert_eq!(state.demands[0].1, key);
    assert_eq!(state.demands[0].2, 1);
    assert_eq!(state.demand_disposition, DecodeQueueDisposition::Promoted);
}

#[test]
fn cached_selection_installs_shared_image_data() {
    let (path, key, image) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    session.decoded_cache.commit_current(
        key.clone(),
        CachedDecodedImage::new(Arc::clone(&image), Duration::from_millis(25)),
    );

    let SelectionEffect::Install(install) = session.select_path(path.clone(), None) else {
        panic!("the cached image should be ready to install");
    };
    assert!(Arc::ptr_eq(install.image(), &image));
    assert!(session.loaded_image().is_none());
    assert!(
        queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .demands
            .is_empty()
    );

    session.commit_install(install, true);

    assert_eq!(session.current_key.as_ref(), Some(&key));
    assert_eq!(session.current_path.as_deref(), Some(path.as_path()));
    assert!(
        session
            .decoded_image
            .as_ref()
            .is_some_and(|installed| Arc::ptr_eq(installed, &image))
    );
    assert!(!session.current_image_was_presented());
    let (presented_path, opening) = session
        .finish_presentation()
        .expect("the installed image is pending presentation");
    assert_eq!(presented_path, path);
    assert!(opening.cache_hit);
    assert!(session.current_image_was_presented());
    assert!(session.finish_presentation().is_none());

    session.shutdown();
    assert!(
        queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .shutdown
    );
}

#[test]
fn gpu_install_keeps_decoded_pixels_shared_without_a_session_arc() {
    let (path, key, image) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    session.decoded_cache.commit_current(
        key,
        CachedDecodedImage::new(Arc::clone(&image), Duration::from_millis(25)),
    );
    assert_eq!(Arc::strong_count(&image), 2);

    let SelectionEffect::Install(install) = session.select_path(path, None) else {
        panic!("the cached image should be ready to install");
    };
    assert_eq!(Arc::strong_count(&image), 3);

    session.commit_install(install, false);

    assert!(session.take_decoded_image().is_none());
    assert_eq!(Arc::strong_count(&image), 2);
    assert!(session.loaded_image().is_some());
}

#[test]
fn navigation_results_ignore_stale_work_and_report_end_or_failure() {
    let (path, _, _) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    assert!(session.begin_navigation(FolderDirection::Next).is_none());
    assert_eq!(
        session.status_message(),
        Some("Open an image before navigating its folder")
    );

    session.current_path = Some(path.clone());
    let request = session
        .begin_navigation(FolderDirection::Next)
        .expect("the current image enables navigation");
    assert!(matches!(
        session.handle_navigation_result(
            request.generation.wrapping_sub(1),
            request.direction,
            &request.source_path,
            Ok(Some(fixture_path("alpha.jxl"))),
        ),
        NavigationEffect::None
    ));
    assert!(matches!(
        session.handle_navigation_result(
            request.generation,
            request.direction,
            &request.source_path,
            Ok(Some(fixture_path("alpha.jxl"))),
        ),
        NavigationEffect::Select {
            direction: FolderDirection::Next,
            ..
        }
    ));

    let request = session.begin_navigation(FolderDirection::Previous).unwrap();
    assert!(matches!(
        session.handle_navigation_result(
            request.generation,
            request.direction,
            &request.source_path,
            Ok(None),
        ),
        NavigationEffect::StatusChanged
    ));
    assert_eq!(
        session.status_message(),
        Some("No previous image in this folder")
    );

    let request = session.begin_navigation(FolderDirection::Next).unwrap();
    assert!(matches!(
        session.handle_navigation_result(
            request.generation,
            request.direction,
            &request.source_path,
            Err(permission_denied()),
        ),
        NavigationEffect::StatusChanged
    ));
    assert_eq!(
        session.status_message(),
        Some("Failed to inspect the image folder: failed to read /photos: permission denied")
    );
}

#[test]
fn navigation_start_failure_only_updates_the_matching_request() {
    let (path, _, _) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    session.current_path = Some(path);
    let stale = session.begin_navigation(FolderDirection::Next).unwrap();
    let current = session.begin_navigation(FolderDirection::Previous).unwrap();
    let current_status = session.status_message.clone();

    session.navigation_start_failed(&stale, &"stale thread failure");
    assert_eq!(session.status_message, current_status);

    session.navigation_start_failed(&current, &"thread unavailable");
    assert_eq!(
        session.status_message(),
        Some("Failed to inspect the image folder: thread unavailable")
    );
}

#[test]
fn rejected_install_reports_the_gpu_error() {
    let (path, key, image) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    session.decoded_cache.commit_current(
        key,
        CachedDecodedImage::new(image, Duration::from_millis(25)),
    );
    let SelectionEffect::Install(install) = session.select_path(path, None) else {
        panic!("the cached image should be ready to install");
    };

    session.reject_install(&install, &"GPU allocation failed");

    assert!(
        session
            .status_message()
            .is_some_and(|message| message.contains("GPU allocation failed"))
    );
    assert!(matches!(
        &session.current_image_presentation,
        ImagePresentationState::Pending(None)
    ));
}

#[test]
fn presentation_cancellation_discards_only_the_pending_timing() {
    let (path, key, image) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    session.decoded_cache.commit_current(
        key,
        CachedDecodedImage::new(Arc::clone(&image), Duration::from_millis(5)),
    );
    let SelectionEffect::Install(install) = session.select_path(path, None) else {
        panic!("the cached image should be ready to install");
    };
    session.commit_install(install, true);

    session.cancel_pending_presentation();

    assert!(session.finish_presentation().is_none());
    assert!(!session.current_image_was_presented());
    assert!(session.has_loaded_image());
    assert!(
        session
            .take_decoded_image()
            .is_some_and(|current| Arc::ptr_eq(&current, &image))
    );
}

#[test]
fn prefetch_uses_navigation_direction_and_defaults_to_next() {
    let neighbors = || NeighborPaths {
        previous: Some(PathBuf::from("previous.jxl")),
        next: Some(PathBuf::from("next.jxl")),
    };
    assert_eq!(
        ordered_neighbor_paths(neighbors(), None),
        [
            Some(PathBuf::from("next.jxl")),
            Some(PathBuf::from("previous.jxl"))
        ]
    );
    assert_eq!(
        ordered_neighbor_paths(neighbors(), Some(FolderDirection::Previous)),
        [
            Some(PathBuf::from("previous.jxl")),
            Some(PathBuf::from("next.jxl"))
        ]
    );
    assert_eq!(
        ordered_neighbor_paths(neighbors(), Some(FolderDirection::Next)),
        [
            Some(PathBuf::from("next.jxl")),
            Some(PathBuf::from("previous.jxl"))
        ]
    );
}

fn gain_map_fixture(rendition: Rendition) -> (PathBuf, ImageKey, Arc<DecodedImage>) {
    let path = fixture_path("test_pattern-gain-map.jxl");
    let key = ImageKey::from_path(&path).unwrap();
    let image = xl_view::decode::decode_memory_with_rendition(
        &std::fs::read(&path).unwrap(),
        DecodeLimits::from_memory_ceiling_mib(256),
        rendition,
    )
    .unwrap();
    (path, key, image)
}

/// Completes the pending demand decode with `result`.
fn complete_demand(
    session: &mut ImageSession,
    queue_state: &Mutex<FakeDecodeState>,
    result: xl_view::decode::DecodeResult,
) -> DecodeEffect {
    let (path, key, selection_generation) = queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .demands
        .last()
        .expect("a demand decode was queued")
        .clone();
    session.handle_decode_completion(DecodeCompletion {
        decode_time: Duration::from_millis(10),
        key,
        path,
        purpose: DecodePurpose::Demand {
            selection_generation,
        },
        result,
    })
}

/// Opens `path` and installs `image` as its decode result.
fn show(
    session: &mut ImageSession,
    queue_state: &Mutex<FakeDecodeState>,
    path: &Path,
    image: &Arc<DecodedImage>,
) {
    assert!(matches!(
        session.select_path(path.to_owned(), None),
        SelectionEffect::Pending
    ));
    let DecodeEffect::Install(install) =
        complete_demand(session, queue_state, Ok(Arc::clone(image)))
    else {
        panic!("the current demand should be ready to install");
    };
    assert!(!install.preserves_view());
    session.commit_install(install, false);
}

fn demand_count(queue_state: &Mutex<FakeDecodeState>) -> usize {
    queue_state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .demands
        .len()
}

/// Calls `sync_rendition` and reports whether it queued a swap decode.
fn sync_queues_swap(
    session: &mut ImageSession,
    queue_state: &Mutex<FakeDecodeState>,
    rendition: Rendition,
) -> bool {
    let before = demand_count(queue_state);
    session.sync_rendition(rendition);
    demand_count(queue_state) > before
}

#[test]
fn an_hdr_display_swaps_in_the_hdr_image_and_keeps_the_view_and_neighbors() {
    let (path, key, sdr) = gain_map_fixture(Rendition::Sdr);
    let (_, _, hdr) = gain_map_fixture(Rendition::Hdr);
    let (_, neighbor_key, neighbor) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(256 * 1024 * 1024, queue);
    show(&mut session, &queue_state, &path, &sdr);
    assert!(session.decoded_cache.admit_inactive(
        neighbor_key.clone(),
        CachedDecodedImage::new(neighbor, Duration::from_millis(5)),
        &[],
    ));

    // A session starts with the SDR base image, which suits an SDR display.
    assert!(!sync_queues_swap(
        &mut session,
        &queue_state,
        Rendition::Sdr
    ));
    assert_eq!(demand_count(&queue_state), 1);

    assert!(sync_queues_swap(&mut session, &queue_state, Rendition::Hdr));
    {
        let state = queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(state.renditions, [Rendition::Hdr]);
        assert_eq!(state.demands.len(), 2);
        assert_eq!(state.demands[1].1, key);
    }
    // The shown image stays in place, without a status, while it is decoded again.
    assert!(session.status_message().is_none());
    assert!(session.has_loaded_image());
    // Repeated checks, as on every redraw, do not queue the swap again.
    assert!(!sync_queues_swap(
        &mut session,
        &queue_state,
        Rendition::Hdr
    ));
    assert_eq!(demand_count(&queue_state), 2);

    let DecodeEffect::Install(install) = complete_demand(&mut session, &queue_state, Ok(hdr))
    else {
        panic!("the swap should be ready to install");
    };
    assert!(install.preserves_view());
    session.commit_install(install, false);

    assert!(matches!(
        session.loaded_image().unwrap().gain_map,
        Some(xl_view::decode::GainMapStatus::Applied { .. })
    ));
    // Opening a file clears the cache. A swap must not.
    assert!(session.decoded_cache.contains(&neighbor_key));
    assert!(!sync_queues_swap(
        &mut session,
        &queue_state,
        Rendition::Hdr
    ));

    // Back on an SDR display, the HDR image is swapped for the base image.
    assert!(sync_queues_swap(&mut session, &queue_state, Rendition::Sdr));
    assert_eq!(demand_count(&queue_state), 3);
}

#[test]
fn images_without_a_gain_map_are_never_decoded_again() {
    let (path, _, image) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);
    show(&mut session, &queue_state, &path, &image);

    assert!(!sync_queues_swap(
        &mut session,
        &queue_state,
        Rendition::Hdr
    ));
    assert!(!sync_queues_swap(
        &mut session,
        &queue_state,
        Rendition::Sdr
    ));

    assert_eq!(demand_count(&queue_state), 1);
}

#[test]
fn a_failed_swap_keeps_the_image_and_waits_for_the_next_display_change() {
    let (path, _, sdr) = gain_map_fixture(Rendition::Sdr);
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(256 * 1024 * 1024, queue);
    show(&mut session, &queue_state, &path, &sdr);
    assert!(sync_queues_swap(&mut session, &queue_state, Rendition::Hdr));

    assert!(matches!(
        complete_demand(&mut session, &queue_state, Err(DecodeError::Canceled)),
        DecodeEffect::None
    ));

    assert!(session.has_loaded_image());
    assert!(session.status_message().is_none());
    assert!(session.pending_open.is_none());
    // No retry on the following redraws.
    assert!(!sync_queues_swap(
        &mut session,
        &queue_state,
        Rendition::Hdr
    ));
    assert_eq!(demand_count(&queue_state), 2);
    // The display changing again allows another attempt.
    assert!(!sync_queues_swap(
        &mut session,
        &queue_state,
        Rendition::Sdr
    ));
    assert!(sync_queues_swap(&mut session, &queue_state, Rendition::Hdr));
    assert_eq!(demand_count(&queue_state), 3);
}

#[test]
fn a_pending_open_takes_precedence_over_a_swap() {
    let (path, _, sdr) = gain_map_fixture(Rendition::Sdr);
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(256 * 1024 * 1024, queue);
    show(&mut session, &queue_state, &path, &sdr);
    assert!(matches!(
        session.select_path(fixture_path("alpha.jxl"), None),
        SelectionEffect::Pending
    ));

    assert!(!sync_queues_swap(
        &mut session,
        &queue_state,
        Rendition::Hdr
    ));

    assert_eq!(demand_count(&queue_state), 2);
    assert!(
        session
            .pending_open
            .as_ref()
            .is_some_and(|context| !context.rendition_swap)
    );
}

#[test]
fn cached_images_of_the_other_rendition_are_dropped_and_decoded_again() {
    let (path, key, sdr) = gain_map_fixture(Rendition::Sdr);
    let (_, ordinary_key, ordinary) = decoded_fixture();
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(256 * 1024 * 1024, queue);
    let cached = || CachedDecodedImage::new(Arc::clone(&sdr), Duration::from_millis(5));
    assert!(
        session
            .decoded_cache
            .admit_inactive(key.clone(), cached(), &[])
    );
    assert!(session.decoded_cache.admit_inactive(
        ordinary_key.clone(),
        CachedDecodedImage::new(ordinary, Duration::from_millis(5)),
        &[],
    ));

    // The display turns out to be HDR: the base image is not what it shows.
    session.sync_rendition(Rendition::Hdr);
    assert!(!session.decoded_cache.contains(&key));
    assert!(session.decoded_cache.contains(&ordinary_key));

    // One that is still cached, or arrives late, is not used either.
    assert!(
        session
            .decoded_cache
            .admit_inactive(key.clone(), cached(), &[])
    );
    assert!(matches!(
        session.select_path(path.clone(), Some(FolderDirection::Next)),
        SelectionEffect::Pending
    ));
    assert_eq!(demand_count(&queue_state), 1);

    let mut session = ImageSession::with_decode_queue(
        256 * 1024 * 1024,
        FakeDecodeQueue::new(DecodeQueueDisposition::Queued),
    );
    session.sync_rendition(Rendition::Hdr);
    session.handle_decode_completion(DecodeCompletion {
        decode_time: Duration::from_millis(10),
        key: key.clone(),
        path,
        purpose: DecodePurpose::Demand {
            selection_generation: 99,
        },
        result: Ok(sdr),
    });
    assert!(!session.decoded_cache.contains(&key));
}

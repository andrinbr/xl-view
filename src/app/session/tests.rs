use std::collections::HashSet;
use std::sync::Mutex;
use std::time::SystemTime;

use super::*;
use xl_view::decode::{DecodeLimits, decode_file};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[derive(Debug)]
struct FakeDecodeState {
    demand_disposition: DecodeQueueDisposition,
    demands: Vec<(PathBuf, ImageKey, u64)>,
    prefetches: Vec<(PathBuf, ImageKey, u64, u8, u64)>,
    cancellations: Vec<u64>,
    contained: HashSet<ImageKey>,
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

    fn shutdown(&mut self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .shutdown = true;
    }
}

fn decoded_fixture() -> (PathBuf, ImageKey, Arc<DecodedImage>) {
    let path = fixture("ramp-pq-1000.jxl");
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
    let preferred = fixture("alpha.jxl");
    let secondary = fixture("grayscale.jxl");
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
        result: Err(DecodeError::Cancelled),
    });

    assert!(matches!(effect, DecodeEffect::StatusChanged));
    assert!(session.pending_open.is_none());
    assert!(
        session
            .status_message()
            .is_some_and(|message| message.contains("Cannot open"))
    );
}

#[test]
fn invalid_source_path_reports_status_without_queueing_decode() {
    let queue = FakeDecodeQueue::new(DecodeQueueDisposition::Queued);
    let queue_state = Arc::clone(&queue.state);
    let mut session = ImageSession::with_decode_queue(64 * 1024 * 1024, queue);

    assert!(matches!(
        session.select_path(fixture("missing.jxl"), None),
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
        result: Err(DecodeError::Cancelled),
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
        Err("permission denied".to_owned()),
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
            result: Err(DecodeError::Cancelled),
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
            Ok(Some(fixture("alpha.jxl"))),
        ),
        NavigationEffect::None
    ));
    assert!(matches!(
        session.handle_navigation_result(
            request.generation,
            request.direction,
            &request.source_path,
            Ok(Some(fixture("alpha.jxl"))),
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
            Err("permission denied".to_owned()),
        ),
        NavigationEffect::StatusChanged
    ));
    assert_eq!(
        session.status_message(),
        Some("Cannot inspect the image folder: permission denied")
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
        Some("Cannot inspect the image folder: thread unavailable")
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

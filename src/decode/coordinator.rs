use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::{DecodeError, DecodeLimits, DecodeResult, ImageKey};

/// Identifies why an image was queued and carries the generation used by its
/// consumer to reject stale completion events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodePurpose {
    /// User-requested work, which is dequeued before prefetch work.
    Demand {
        /// Identifies the selection that should receive the completion.
        selection_generation: u64,
    },
    /// Speculative neighbor work with a retained-storage admission limit.
    Prefetch {
        /// Identifies the prefetch plan that should receive the completion.
        prefetch_generation: u64,
        /// Preserves the neighbor's priority within that plan.
        neighbor_index: u8,
        /// Maximum decoded bytes that may remain cached after the attempt.
        maximum_retained_bytes: u64,
    },
}

#[derive(Debug)]
pub struct DecodeCompletion {
    pub decode_time: Duration,
    pub key: ImageKey,
    pub path: PathBuf,
    pub purpose: DecodePurpose,
    pub result: DecodeResult,
}

/// Describes how a request changed the single-worker queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeQueueDisposition {
    /// A new job was appended to its priority queue.
    Queued,
    /// Work for the same image was already queued or active.
    Coalesced,
    /// Existing prefetch work was upgraded to demand priority and identity.
    Promoted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JobLocation {
    Demand,
    Prefetch,
    Active,
}

#[derive(Clone, Debug)]
struct DecodeJob {
    key: ImageKey,
    path: PathBuf,
    purpose: DecodePurpose,
}

struct DecodeAttempt {
    job: DecodeJob,
    result: DecodeResult,
    decode_time: Duration,
}

impl DecodeAttempt {
    fn into_completion(self) -> DecodeCompletion {
        DecodeCompletion {
            decode_time: self.decode_time,
            key: self.job.key,
            path: self.job.path,
            purpose: self.job.purpose,
            result: self.result,
        }
    }
}

#[derive(Debug, Default)]
/// Queue entries and `active` collectively contain each key in `jobs` exactly
/// once, and every `JobLocation` mirrors that key's current container. The
/// shared mutex protects updates to both representations as one transaction.
struct CoordinatorState {
    demand: VecDeque<DecodeJob>,
    prefetch: VecDeque<DecodeJob>,
    active: Option<DecodeJob>,
    jobs: HashMap<ImageKey, JobLocation>,
    shutdown: bool,
}

type DecodeFunction = dyn Fn(&ImageKey, Option<u64>) -> DecodeResult + Send + Sync;
type DeliverFunction = dyn Fn(DecodeCompletion) + Send + Sync;

pub struct DecodeCoordinator {
    state: Arc<(Mutex<CoordinatorState>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for DecodeCoordinator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecodeCoordinator")
            .field("thread_running", &self.thread.is_some())
            .finish_non_exhaustive()
    }
}

impl DecodeCoordinator {
    /// Starts the single long-lived decode thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the operating system cannot create the thread.
    pub fn spawn<F>(limits: DecodeLimits, deliver: F) -> Result<Self, std::io::Error>
    where
        F: Fn(DecodeCompletion) + Send + Sync + 'static,
    {
        let decode = Arc::new(move |key: &ImageKey, maximum_retained_bytes| {
            super::jxl::decode_file_for_key(key, limits, maximum_retained_bytes)
        });
        Self::spawn_with_functions(decode, Arc::new(deliver))
    }

    fn spawn_with_functions(
        decode: Arc<DecodeFunction>,
        deliver: Arc<DeliverFunction>,
    ) -> Result<Self, std::io::Error> {
        let state = Arc::new((Mutex::new(CoordinatorState::default()), Condvar::new()));
        let worker_state = Arc::clone(&state);
        let thread = std::thread::Builder::new()
            .name("xl-view-decode-coordinator".to_owned())
            .spawn(move || run_coordinator(&worker_state, decode.as_ref(), deliver.as_ref()))?;
        Ok(Self {
            state,
            thread: Some(thread),
        })
    }

    /// Queues user-requested work ahead of all prefetch work.
    ///
    /// Repeated demand is coalesced. Queued or active prefetch work for the
    /// same key is promoted and adopts this selection generation; if an active
    /// limited prefetch then fails only its admission limit, it is retried as
    /// unrestricted demand.
    pub fn request_demand(
        &self,
        path: PathBuf,
        key: ImageKey,
        selection_generation: u64,
    ) -> DecodeQueueDisposition {
        let (state, wake) = self.state.as_ref();
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let purpose = DecodePurpose::Demand {
            selection_generation,
        };
        let disposition = match state.jobs.get(&key).copied() {
            Some(JobLocation::Demand) => {
                if let Some(job) = state.demand.iter_mut().find(|job| job.key == key) {
                    job.path = path;
                    job.purpose = purpose;
                }
                DecodeQueueDisposition::Coalesced
            }
            Some(JobLocation::Prefetch) => {
                if let Some(index) = state.prefetch.iter().position(|job| job.key == key)
                    && let Some(mut job) = state.prefetch.remove(index)
                {
                    job.path = path;
                    job.purpose = purpose;
                    state.demand.push_back(job);
                    state.jobs.insert(key, JobLocation::Demand);
                }
                DecodeQueueDisposition::Promoted
            }
            Some(JobLocation::Active) => {
                // Promotion updates the authoritative active metadata while
                // the in-flight decode may still be using the prefetch limit.
                // synchronize_initial_attempt observes the new purpose and
                // retries only a limit-related failure as unrestricted demand.
                if let Some(job) = state.active.as_mut() {
                    job.path = path;
                    job.purpose = purpose;
                }
                DecodeQueueDisposition::Promoted
            }
            None => {
                state.demand.push_back(DecodeJob {
                    key: key.clone(),
                    path,
                    purpose,
                });
                state.jobs.insert(key, JobLocation::Demand);
                DecodeQueueDisposition::Queued
            }
        };
        wake.notify_one();
        disposition
    }

    /// Queues speculative neighbor work behind all demand work.
    ///
    /// `maximum_retained_bytes` bounds the decoded storage admitted for this
    /// prefetch. Any existing job for `key`, including demand or active work,
    /// is left unchanged and reported as [`DecodeQueueDisposition::Coalesced`].
    #[allow(clippy::too_many_arguments)] // Prefetch identity, priority, and admission limit form one explicit request.
    pub fn request_prefetch(
        &self,
        path: PathBuf,
        key: ImageKey,
        prefetch_generation: u64,
        neighbor_index: u8,
        maximum_retained_bytes: u64,
    ) -> DecodeQueueDisposition {
        let (state, wake) = self.state.as_ref();
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.jobs.contains_key(&key) {
            return DecodeQueueDisposition::Coalesced;
        }
        state.prefetch.push_back(DecodeJob {
            key: key.clone(),
            path,
            purpose: DecodePurpose::Prefetch {
                prefetch_generation,
                neighbor_index,
                maximum_retained_bytes,
            },
        });
        state.jobs.insert(key, JobLocation::Prefetch);
        wake.notify_one();
        DecodeQueueDisposition::Queued
    }

    /// Removes queued prefetches from older plans.
    ///
    /// Demand and active jobs are never cancelled; the retained generation is
    /// supplied by the session that owns the current prefetch plan.
    pub fn cancel_queued_prefetches_except(&self, prefetch_generation: u64) {
        let (state, _) = self.state.as_ref();
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut retained = VecDeque::new();
        let mut removed = Vec::new();
        while let Some(job) = state.prefetch.pop_front() {
            if matches!(
                job.purpose,
                DecodePurpose::Prefetch {
                    prefetch_generation: generation,
                    ..
                } if generation == prefetch_generation
            ) {
                retained.push_back(job);
            } else {
                removed.push(job.key);
            }
        }
        state.prefetch = retained;
        for key in removed {
            state.jobs.remove(&key);
        }
    }

    /// Returns whether the image is queued or actively decoding.
    #[must_use]
    pub fn contains(&self, key: &ImageKey) -> bool {
        let (state, _) = self.state.as_ref();
        let state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.jobs.contains_key(key)
    }

    /// Stops accepting work, clears queued jobs, and suppresses active delivery.
    ///
    /// The codec cannot interrupt an in-flight decode, and application shutdown
    /// must not wait for one to finish. An already-finished worker is joined;
    /// otherwise its handle is detached. The worker owns the state and callbacks
    /// it still needs, observes `shutdown` before delivery, and releases those
    /// resources when the codec call eventually returns.
    pub fn shutdown(&mut self) {
        let (state, wake) = self.state.as_ref();
        {
            let mut state = state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.shutdown = true;
            state.demand.clear();
            state.prefetch.clear();
            state
                .jobs
                .retain(|_, location| *location == JobLocation::Active);
        }
        wake.notify_one();
        if self.thread.as_ref().is_some_and(JoinHandle::is_finished)
            && let Some(thread) = self.thread.take()
        {
            let _ = thread.join();
        } else {
            let _ = self.thread.take();
        }
    }
}

impl Drop for DecodeCoordinator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn wait_for_decode_job(shared: &(Mutex<CoordinatorState>, Condvar)) -> Option<DecodeJob> {
    let (state, wake) = shared;
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while !state.shutdown && state.demand.is_empty() && state.prefetch.is_empty() {
        state = wake
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    if state.shutdown {
        return None;
    }
    let job = state
        .demand
        .pop_front()
        .or_else(|| state.prefetch.pop_front())
        .expect("a queued decode exists");
    state.jobs.insert(job.key.clone(), JobLocation::Active);
    state.active = Some(job.clone());
    Some(job)
}

fn decode_retained_byte_limit(purpose: DecodePurpose) -> Option<u64> {
    match purpose {
        DecodePurpose::Demand { .. } => None,
        DecodePurpose::Prefetch {
            maximum_retained_bytes,
            ..
        } => Some(maximum_retained_bytes),
    }
}

fn run_decode_attempt(
    job: DecodeJob,
    maximum_retained_bytes: Option<u64>,
    decode: &DecodeFunction,
) -> DecodeAttempt {
    let decode_started = Instant::now();
    let result = decode(&job.key, maximum_retained_bytes);
    DecodeAttempt {
        job,
        result,
        decode_time: decode_started.elapsed(),
    }
}

fn synchronize_initial_attempt(
    shared: &(Mutex<CoordinatorState>, Condvar),
    attempt: &mut DecodeAttempt,
) -> Option<bool> {
    // Copy any active-prefetch promotion into the completed attempt. A demand
    // that failed only because of the former prefetch admission limit is
    // retried without that limit; all other attempts leave the queue index
    // before delivery.
    let (state, _) = shared;
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(active) = state.active.as_ref() {
        attempt.job = active.clone();
    }
    if state.shutdown {
        return None;
    }
    let retry = matches!(attempt.job.purpose, DecodePurpose::Demand { .. })
        && matches!(attempt.result, Err(DecodeError::PrefetchTooLarge { .. }));
    if !retry {
        state.active = None;
        state.jobs.remove(&attempt.job.key);
    }
    Some(retry)
}

fn finish_retry(
    shared: &(Mutex<CoordinatorState>, Condvar),
    attempt: &mut DecodeAttempt,
) -> Option<()> {
    let (state, _) = shared;
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(active) = state.active.take() {
        attempt.job = active;
    }
    state.jobs.remove(&attempt.job.key);
    (!state.shutdown).then_some(())
}

fn execute_decode_job(
    shared: &(Mutex<CoordinatorState>, Condvar),
    job: DecodeJob,
    decode: &DecodeFunction,
) -> Option<DecodeCompletion> {
    let maximum_retained_bytes = decode_retained_byte_limit(job.purpose);
    let mut attempt = run_decode_attempt(job, maximum_retained_bytes, decode);
    if synchronize_initial_attempt(shared, &mut attempt)? {
        let retry = run_decode_attempt(attempt.job, None, decode);
        attempt.job = retry.job;
        attempt.result = retry.result;
        attempt.decode_time = attempt.decode_time.saturating_add(retry.decode_time);
        finish_retry(shared, &mut attempt)?;
    }
    Some(attempt.into_completion())
}

fn run_coordinator(
    shared: &(Mutex<CoordinatorState>, Condvar),
    decode: &DecodeFunction,
    deliver: &DeliverFunction,
) {
    while let Some(job) = wait_for_decode_job(shared) {
        let Some(completion) = execute_decode_job(shared, job, decode) else {
            return;
        };
        deliver(completion);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, SystemTime};

    use super::*;

    fn key(name: &str) -> ImageKey {
        ImageKey {
            normalized_path: PathBuf::from(name),
            source_len: 1,
            source_modified: SystemTime::UNIX_EPOCH,
        }
    }

    fn failed_decode() -> DecodeResult {
        Err(DecodeError::Cancelled)
    }

    #[test]
    fn demand_runs_before_queued_prefetch_and_only_one_decode_is_active() {
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let release_receiver = Arc::new(Mutex::new(release_receiver));
        let active = Arc::new(AtomicUsize::new(0));
        let maximum_active = Arc::new(AtomicUsize::new(0));
        let decode = {
            let active = Arc::clone(&active);
            let maximum_active = Arc::clone(&maximum_active);
            Arc::new(move |key: &ImageKey, _: Option<u64>| {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum_active.fetch_max(current, Ordering::SeqCst);
                started_sender.send(key.normalized_path.clone()).unwrap();
                release_receiver.lock().unwrap().recv().unwrap();
                active.fetch_sub(1, Ordering::SeqCst);
                failed_decode()
            }) as Arc<DecodeFunction>
        };
        let (completion_sender, completion_receiver) = mpsc::channel();
        let mut coordinator = DecodeCoordinator::spawn_with_functions(
            decode,
            Arc::new(move |completion| completion_sender.send(completion).unwrap()),
        )
        .unwrap();

        coordinator.request_prefetch(PathBuf::from("p1"), key("p1"), 1, 0, 100);
        assert_eq!(
            started_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            PathBuf::from("p1")
        );
        coordinator.request_prefetch(PathBuf::from("p2"), key("p2"), 1, 1, 100);
        coordinator.request_demand(PathBuf::from("d1"), key("d1"), 1);
        release_sender.send(()).unwrap();
        assert_eq!(
            started_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            PathBuf::from("d1")
        );
        release_sender.send(()).unwrap();
        assert_eq!(
            started_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            PathBuf::from("p2")
        );
        release_sender.send(()).unwrap();
        for _ in 0..3 {
            completion_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap();
        }
        assert_eq!(maximum_active.load(Ordering::SeqCst), 1);
        coordinator.shutdown();
    }

    #[test]
    fn queued_and_active_prefetches_are_promoted() {
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let release_receiver = Arc::new(Mutex::new(release_receiver));
        let decode = Arc::new(move |key: &ImageKey, _: Option<u64>| {
            started_sender.send(key.normalized_path.clone()).unwrap();
            release_receiver.lock().unwrap().recv().unwrap();
            failed_decode()
        }) as Arc<DecodeFunction>;
        let (completion_sender, completion_receiver) = mpsc::channel();
        let mut coordinator = DecodeCoordinator::spawn_with_functions(
            decode,
            Arc::new(move |completion| completion_sender.send(completion).unwrap()),
        )
        .unwrap();

        coordinator.request_prefetch(PathBuf::from("active"), key("active"), 1, 0, 100);
        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            coordinator.request_demand(PathBuf::from("active"), key("active"), 7),
            DecodeQueueDisposition::Promoted
        );
        coordinator.request_prefetch(PathBuf::from("queued"), key("queued"), 1, 1, 100);
        assert_eq!(
            coordinator.request_demand(PathBuf::from("queued"), key("queued"), 8),
            DecodeQueueDisposition::Promoted
        );

        release_sender.send(()).unwrap();
        let active = completion_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            active.purpose,
            DecodePurpose::Demand {
                selection_generation: 7
            }
        );
        assert_eq!(
            started_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            PathBuf::from("queued")
        );
        release_sender.send(()).unwrap();
        let queued = completion_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            queued.purpose,
            DecodePurpose::Demand {
                selection_generation: 8
            }
        );
        coordinator.shutdown();
    }

    #[test]
    fn promoted_oversized_prefetch_retries_without_the_admission_limit() {
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let release_receiver = Mutex::new(release_receiver);
        let limits_seen = Arc::new(Mutex::new(Vec::new()));
        let decode = {
            let limits_seen = Arc::clone(&limits_seen);
            Arc::new(move |_: &ImageKey, limit: Option<u64>| {
                limits_seen.lock().unwrap().push(limit);
                if limit.is_some() {
                    started_sender.send(()).unwrap();
                    release_receiver.lock().unwrap().recv().unwrap();
                    Err(DecodeError::PrefetchTooLarge {
                        required: 2,
                        available: 1,
                    })
                } else {
                    failed_decode()
                }
            }) as Arc<DecodeFunction>
        };
        let (completion_sender, completion_receiver) = mpsc::channel();
        let mut coordinator = DecodeCoordinator::spawn_with_functions(
            decode,
            Arc::new(move |completion| completion_sender.send(completion).unwrap()),
        )
        .unwrap();

        coordinator.request_prefetch(PathBuf::from("image"), key("image"), 1, 0, 1);
        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        coordinator.request_demand(PathBuf::from("image"), key("image"), 9);
        release_sender.send(()).unwrap();

        let completion = completion_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert!(!completion.decode_time.is_zero());
        assert_eq!(
            completion.purpose,
            DecodePurpose::Demand {
                selection_generation: 9
            }
        );
        assert_eq!(&*limits_seen.lock().unwrap(), &[Some(1), None]);
        coordinator.shutdown();
    }

    #[test]
    fn duplicate_jobs_coalesce_and_prefetch_cancellation_updates_membership() {
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let release_receiver = Arc::new(Mutex::new(release_receiver));
        let decode = Arc::new(move |key: &ImageKey, _: Option<u64>| {
            started_sender.send(key.normalized_path.clone()).unwrap();
            release_receiver.lock().unwrap().recv().unwrap();
            failed_decode()
        }) as Arc<DecodeFunction>;
        let (completion_sender, completion_receiver) = mpsc::channel();
        let mut coordinator = DecodeCoordinator::spawn_with_functions(
            decode,
            Arc::new(move |completion| completion_sender.send(completion).unwrap()),
        )
        .unwrap();

        let blocker = key("blocker");
        coordinator.request_prefetch(PathBuf::from("blocker"), blocker.clone(), 1, 0, 100);
        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        let demand = key("demand");
        assert_eq!(
            coordinator.request_demand(PathBuf::from("demand-old"), demand.clone(), 3),
            DecodeQueueDisposition::Queued
        );
        assert_eq!(
            coordinator.request_demand(PathBuf::from("demand-new"), demand.clone(), 9),
            DecodeQueueDisposition::Coalesced
        );
        let cancelled = key("cancelled");
        coordinator.request_prefetch(PathBuf::from("cancelled"), cancelled.clone(), 1, 1, 100);
        let retained = key("retained");
        coordinator.request_prefetch(PathBuf::from("retained"), retained.clone(), 2, 2, 100);
        assert_eq!(
            coordinator.request_prefetch(
                PathBuf::from("retained-duplicate"),
                retained.clone(),
                2,
                3,
                50,
            ),
            DecodeQueueDisposition::Coalesced
        );
        assert!(coordinator.contains(&demand));
        assert!(coordinator.contains(&cancelled));
        assert!(coordinator.contains(&retained));

        coordinator.cancel_queued_prefetches_except(2);
        assert!(!coordinator.contains(&cancelled));
        assert!(coordinator.contains(&retained));

        release_sender.send(()).unwrap();
        completion_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert!(!coordinator.contains(&blocker));
        assert_eq!(
            started_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            PathBuf::from("demand")
        );
        release_sender.send(()).unwrap();
        let completion = completion_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert_eq!(completion.path, PathBuf::from("demand-new"));
        assert_eq!(
            completion.purpose,
            DecodePurpose::Demand {
                selection_generation: 9
            }
        );
        assert!(!coordinator.contains(&demand));

        assert_eq!(
            started_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            PathBuf::from("retained")
        );
        release_sender.send(()).unwrap();
        completion_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert!(!coordinator.contains(&retained));
        coordinator.shutdown();
    }

    #[test]
    fn shutdown_clears_queued_work_and_suppresses_the_active_completion() {
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let release_receiver = Mutex::new(release_receiver);
        let decode = Arc::new(move |key: &ImageKey, _: Option<u64>| {
            started_sender.send(key.normalized_path.clone()).unwrap();
            release_receiver.lock().unwrap().recv().unwrap();
            failed_decode()
        }) as Arc<DecodeFunction>;
        let (completion_sender, completion_receiver) = mpsc::channel();
        let mut coordinator = DecodeCoordinator::spawn_with_functions(
            decode,
            Arc::new(move |completion| completion_sender.send(completion).unwrap()),
        )
        .unwrap();

        let active = key("active");
        coordinator.request_prefetch(PathBuf::from("active"), active, 1, 0, 100);
        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        let queued = key("queued");
        coordinator.request_demand(PathBuf::from("queued"), queued.clone(), 1);

        coordinator.shutdown();
        assert!(!coordinator.contains(&queued));
        release_sender.send(()).unwrap();
        assert!(
            completion_receiver
                .recv_timeout(Duration::from_millis(100))
                .is_err()
        );
        assert!(
            started_receiver
                .recv_timeout(Duration::from_millis(100))
                .is_err()
        );
    }

    #[test]
    fn closed_delivery_channel_does_not_stop_later_decode_work() {
        let (started_sender, started_receiver) = mpsc::channel();
        let decode = Arc::new(move |key: &ImageKey, _: Option<u64>| {
            started_sender.send(key.normalized_path.clone()).unwrap();
            failed_decode()
        }) as Arc<DecodeFunction>;
        let (completion_sender, completion_receiver) = mpsc::channel();
        drop(completion_receiver);
        let (delivery_sender, delivery_receiver) = mpsc::channel();
        let mut coordinator = DecodeCoordinator::spawn_with_functions(
            decode,
            Arc::new(move |completion| {
                let _ = completion_sender.send(completion);
                delivery_sender.send(()).unwrap();
            }),
        )
        .unwrap();

        coordinator.request_demand(PathBuf::from("first"), key("first"), 1);
        coordinator.request_demand(PathBuf::from("second"), key("second"), 2);

        assert_eq!(
            started_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            PathBuf::from("first")
        );
        assert_eq!(
            started_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            PathBuf::from("second")
        );
        for _ in 0..2 {
            delivery_receiver
                .recv_timeout(Duration::from_secs(1))
                .unwrap();
        }
        assert!(!coordinator.contains(&key("first")));
        assert!(!coordinator.contains(&key("second")));
        coordinator.shutdown();
    }
}

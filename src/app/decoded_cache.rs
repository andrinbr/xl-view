use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;

use xl_view::decode::{DecodedImage, ImageKey, Rendition};
use xl_view::units::{BYTES_PER_GIB, u64_from_usize};

const MINIMUM_AUTOMATIC_CACHE_BYTES: u64 = 2 * BYTES_PER_GIB;

pub(super) fn automatic_decoded_cache_bytes(total_memory_bytes: u64) -> u64 {
    MINIMUM_AUTOMATIC_CACHE_BYTES.max(total_memory_bytes / 4)
}

#[derive(Debug)]
struct CacheEntry<V> {
    value: V,
    cost: u64,
    last_used: u64,
}

#[derive(Debug)]
struct PinnedLru<K, V> {
    maximum_bytes: u64,
    resident_bytes: u64,
    access_clock: u64,
    current_key: Option<K>,
    entries: HashMap<K, CacheEntry<V>>,
}

impl<K, V> PinnedLru<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    fn new(maximum_bytes: u64) -> Self {
        Self {
            maximum_bytes,
            resident_bytes: 0,
            access_clock: 0,
            current_key: None,
            entries: HashMap::new(),
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        let last_used = self.next_access();
        let entry = self.entries.get_mut(key)?;
        entry.last_used = last_used;
        Some(entry.value.clone())
    }

    fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.resident_bytes = 0;
        self.current_key = None;
    }

    /// Drops every entry `stale` selects, except the pinned current one.
    fn evict_inactive_where(&mut self, mut stale: impl FnMut(&V) -> bool) {
        let current_key = self.current_key.as_ref();
        let mut freed = 0_u64;
        self.entries.retain(|key, entry| {
            let keep = current_key == Some(key) || !stale(&entry.value);
            if !keep {
                freed = freed.saturating_add(entry.cost);
            }
            keep
        });
        self.resident_bytes = self.resident_bytes.saturating_sub(freed);
    }

    fn commit_current(&mut self, key: K, value: V, cost: u64) {
        self.insert_or_replace(key.clone(), value, cost);
        self.current_key = Some(key);
        self.evict_until(self.maximum_bytes, &[]);
    }

    fn admit_inactive(&mut self, key: K, value: V, cost: u64, protected: &[K]) -> bool {
        if self.maximum_bytes == 0 || self.current_key.as_ref() == Some(&key) {
            return false;
        }
        if self.entries.contains_key(&key) {
            let _ = self.get(&key);
            return true;
        }

        let protected_cost = self.protected_cost(protected);
        if cost > self.maximum_bytes.saturating_sub(protected_cost) {
            return false;
        }

        let target = self.maximum_bytes.saturating_sub(cost);
        self.evict_until(target, protected);
        if self.resident_bytes > target {
            return false;
        }
        self.insert_or_replace(key, value, cost);
        true
    }

    fn available_for_prefetch(&self, protected: &[K]) -> u64 {
        self.maximum_bytes
            .saturating_sub(self.protected_cost(protected))
    }

    fn protected_cost(&self, protected: &[K]) -> u64 {
        self.entries
            .iter()
            .filter(|(key, _)| {
                self.current_key.as_ref() == Some(*key) || protected.iter().any(|item| item == *key)
            })
            .fold(0_u64, |total, (_, entry)| total.saturating_add(entry.cost))
    }

    fn insert_or_replace(&mut self, key: K, value: V, cost: u64) {
        if let Some(previous) = self.entries.remove(&key) {
            self.resident_bytes = self.resident_bytes.saturating_sub(previous.cost);
        }
        let last_used = self.next_access();
        self.entries.insert(
            key,
            CacheEntry {
                value,
                cost,
                last_used,
            },
        );
        self.resident_bytes = self.resident_bytes.saturating_add(cost);
    }

    fn evict_until(&mut self, target_bytes: u64, protected: &[K]) {
        while self.resident_bytes > target_bytes {
            let candidate = self
                .entries
                .iter()
                .filter(|(key, _)| {
                    self.current_key.as_ref() != Some(*key)
                        && !protected.iter().any(|item| item == *key)
                })
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone());
            let Some(candidate) = candidate else {
                break;
            };
            if let Some(entry) = self.entries.remove(&candidate) {
                self.resident_bytes = self.resident_bytes.saturating_sub(entry.cost);
            }
        }
    }

    fn next_access(&mut self) -> u64 {
        self.access_clock = self.access_clock.saturating_add(1);
        self.access_clock
    }
}

#[derive(Debug)]
pub(super) struct DecodedImageCache {
    entries: PinnedLru<ImageKey, CachedDecodedImage>,
}

#[derive(Clone, Debug)]
pub(super) struct CachedDecodedImage {
    pub(super) image: Arc<DecodedImage>,
    pub(super) decode_time: Duration,
}

impl CachedDecodedImage {
    pub(super) fn new(image: Arc<DecodedImage>, decode_time: Duration) -> Self {
        Self { image, decode_time }
    }

    fn memory_cost_bytes(&self) -> u64 {
        u64_from_usize(self.image.memory_cost_bytes)
    }

    /// Whether decoding the file for `rendition` would give another image.
    pub(super) fn differs_for(&self, rendition: Rendition) -> bool {
        self.image
            .gain_map
            .is_some_and(|status| status.differs_for(rendition))
    }
}

impl DecodedImageCache {
    pub(super) fn new(maximum_bytes: u64) -> Self {
        Self {
            entries: PinnedLru::new(maximum_bytes),
        }
    }

    pub(super) fn get(&mut self, key: &ImageKey) -> Option<CachedDecodedImage> {
        self.entries.get(key)
    }

    pub(super) fn contains(&self, key: &ImageKey) -> bool {
        self.entries.contains(key)
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
    }

    /// Drops inactive images whose gain map rendition differs from `rendition`.
    pub(super) fn evict_other_renditions(&mut self, rendition: Rendition) {
        self.entries
            .evict_inactive_where(|cached| cached.differs_for(rendition));
    }

    pub(super) fn commit_current(&mut self, key: ImageKey, image: CachedDecodedImage) {
        let cost = image.memory_cost_bytes();
        self.entries.commit_current(key, image, cost);
    }

    pub(super) fn admit_inactive(
        &mut self,
        key: ImageKey,
        image: CachedDecodedImage,
        protected: &[ImageKey],
    ) -> bool {
        let cost = image.memory_cost_bytes();
        self.entries.admit_inactive(key, image, cost, protected)
    }

    pub(super) fn available_for_prefetch(&self, protected: &[ImageKey]) -> u64 {
        self.entries.available_for_prefetch(protected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_limit_is_at_least_two_gib_and_otherwise_one_quarter() {
        const GIB: u64 = BYTES_PER_GIB;
        assert_eq!(automatic_decoded_cache_bytes(0), 2 * GIB);
        assert_eq!(automatic_decoded_cache_bytes(4 * GIB), 2 * GIB);
        assert_eq!(automatic_decoded_cache_bytes(8 * GIB), 2 * GIB);
        assert_eq!(automatic_decoded_cache_bytes(32 * GIB), 8 * GIB);
        assert_eq!(automatic_decoded_cache_bytes(128 * GIB), 32 * GIB);
    }

    #[test]
    fn current_entry_is_pinned_even_when_it_exceeds_the_budget() {
        let mut cache = PinnedLru::new(10);
        cache.commit_current("current", 1, 12);
        assert_eq!(cache.get(&"current"), Some(1));
        assert_eq!(cache.resident_bytes, 12);
        assert!(!cache.admit_inactive("neighbor", 2, 1, &[]));
    }

    #[test]
    fn inactive_entries_are_evicted_in_lru_order() {
        let mut cache = PinnedLru::new(10);
        cache.commit_current("current", 1, 4);
        assert!(cache.admit_inactive("old", 2, 3, &[]));
        assert!(cache.admit_inactive("recent", 3, 3, &[]));
        let _ = cache.get(&"recent");
        assert!(cache.admit_inactive("replacement", 4, 3, &[]));
        assert!(!cache.contains(&"old"));
        assert!(cache.contains(&"recent"));
        assert!(cache.contains(&"replacement"));
    }

    #[test]
    fn protected_neighbor_is_not_evicted_for_the_second_neighbor() {
        let mut cache = PinnedLru::new(10);
        cache.commit_current("current", 1, 4);
        assert!(cache.admit_inactive("preferred", 2, 4, &[]));
        assert!(!cache.admit_inactive("other", 3, 4, &["preferred"]));
        assert!(cache.contains(&"preferred"));
        assert!(!cache.contains(&"other"));
    }

    #[test]
    fn zero_budget_retains_only_the_current_entry() {
        let mut cache = PinnedLru::new(0);
        cache.commit_current("current", 1, 4);
        assert!(!cache.admit_inactive("neighbor", 2, 1, &[]));
        cache.commit_current("replacement", 3, 5);
        assert!(!cache.contains(&"current"));
        assert_eq!(cache.get(&"replacement"), Some(3));
    }

    #[test]
    fn selective_eviction_spares_the_current_entry_and_updates_the_total() {
        let mut cache = PinnedLru::new(10);
        cache.commit_current("current", 1, 4);
        assert!(cache.admit_inactive("stale", 1, 3, &[]));
        assert!(cache.admit_inactive("kept", 2, 2, &[]));

        cache.evict_inactive_where(|value| *value == 1);

        assert!(cache.contains(&"current"));
        assert!(!cache.contains(&"stale"));
        assert!(cache.contains(&"kept"));
        assert_eq!(cache.resident_bytes, 6);
    }

    #[test]
    fn clearing_removes_current_and_inactive_entries() {
        let mut cache = PinnedLru::new(10);
        cache.commit_current("current", 1, 4);
        assert!(cache.admit_inactive("neighbor", 2, 3, &[]));

        cache.clear();

        assert!(cache.entries.is_empty());
        assert_eq!(cache.resident_bytes, 0);
        assert_eq!(cache.current_key, None);
    }
}

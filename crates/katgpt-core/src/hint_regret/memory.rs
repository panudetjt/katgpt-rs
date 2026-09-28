//! Phase 3 — frontier ordering (Beta-LCB over [`crate::best_belief`]) and
//! the regret-scored content memory seam.
//!
//! The memory holds one entry per content item the curriculum has probed:
//! its regret estimate, CI, skill tags, and last-seen tick. Salience
//! (`r̂ · σ(−λ·Δt)` — the [`crate::tether::horizon_decay`] family)
//! time-decays entries so stale regrets fade instead of dominating.
//! Eviction is two-policy: oldest-first at capacity, and **absorbing** for
//! Intractable-classified content — once retired, a hash is never re-admitted
//! (bounded tombstone ring; "evicted, not farmed", Guide 340 §"Rollout").

use crate::best_belief::best_belief_score;
use crate::sigmoid;

use super::gate::Regime;

/// Default ε for the Beta-LCB quantile — matches `best_belief`'s documented
/// convention (0.05, the 95% lower bound) and riir-clippy's
/// `BELIEF_EPSILON` (the shipped `SelectionMode::BetaPosterior` consumer).
pub const FRONTIER_EPSILON: f32 = 0.05;

/// ε-quantile of `Beta(1 + successes, 1 + failures)` — the conservative
/// lower bound on true win rate. Thin delegation to
/// [`crate::best_belief::best_belief_score`] (Plan 576 Phase 3: "composes
/// shipped SelectionMode::BetaPosterior" — consume, don't re-implement).
#[inline]
pub fn beta_lcb(successes: u32, failures: u32, epsilon: f32) -> f32 {
    best_belief_score(successes, failures, epsilon)
}

/// Zero-alloc Beta-LCB ordering: fills `order` with candidate indices in
/// **descending** LCB order (frontier ordering — best first) and `lcbs[i]`
/// with candidate `i`'s ε-quantile (**candidate-indexed**, so consumers read
/// `lcbs[order[rank]]`; the ascending direction — weakness-slice diagnosis,
/// Research 500 row 4: weakest slice = lowest LCB — is `order` reversed).
///
/// Tie-break: ascending index — canonical and deterministic (the
/// `canonical_rank` lesson: never let backing-map order decide a rank).
/// Both output buffers are cleared and reused (caller-owned scratch);
/// `sort_unstable_by` never allocates and the comparator is a total order,
/// so determinism does not depend on sort stability.
pub fn beta_lcb_order_into(
    scores: &[(u32, u32)],
    epsilon: f32,
    order: &mut Vec<usize>,
    lcbs: &mut Vec<f32>,
) {
    order.clear();
    lcbs.clear();
    order.reserve(scores.len());
    lcbs.reserve(scores.len());
    for (i, &(s, f)) in scores.iter().enumerate() {
        order.push(i);
        lcbs.push(beta_lcb(s, f, epsilon));
    }
    order.sort_unstable_by(|&a, &b| lcbs[b].total_cmp(&lcbs[a]).then(a.cmp(&b)));
}

/// Allocating convenience wrapper over [`beta_lcb_order_into`] (cold path —
/// ordering a candidate list, not a hot loop). Returns `(order, lcbs)`
/// with `lcbs` candidate-indexed (read `lcbs[order[rank]]` for rank order).
pub fn beta_lcb_order(scores: &[(u32, u32)], epsilon: f32) -> (Vec<usize>, Vec<f32>) {
    let mut order = Vec::with_capacity(scores.len());
    let mut lcbs = Vec::with_capacity(scores.len());
    beta_lcb_order_into(scores, epsilon, &mut order, &mut lcbs);
    (order, lcbs)
}

/// One probed content item in the regret memory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegretMemoryEntry {
    /// BLAKE3 commitment of the content (quest template row hash, quest
    /// id — any 32-byte stable identity the caller commits by).
    pub content_hash: [u8; 32],
    /// Latest hint-regret estimate (module sign convention).
    pub r_hat: f32,
    /// Latest CI half-width.
    pub ci: f32,
    /// Skill-tag bitmask (KG-triple retrieval key, Guide 340 §"Connection
    /// map" — vibe skill tags).
    pub skill_tag_bits: u32,
    /// Tick of the last observation — drives salience decay.
    pub last_seen_tick: u64,
}

/// Memory salience: `r̂ · σ(−λ·Δt)` — high regret, recently confirmed.
///
/// The `decay_confidence` / `horizon_decay` family pattern: store level +
/// tick, compute the effective value on read, so beliefs FADE rather than
/// get deleted. `λ` is the decay rate per tick (e.g. `0.001`); stale
/// frontier entries sink behind fresh ones.
#[inline]
pub fn salience(r_hat: f32, last_seen_tick: u64, now_tick: u64, lambda: f32) -> f32 {
    let dt = now_tick.saturating_sub(last_seen_tick) as f32;
    r_hat * sigmoid(-lambda * dt)
}

/// Outcome of [`RegretMemory::observe`] — observable for tests and consumer
/// telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserveOutcome {
    /// New entry stored.
    Stored,
    /// Existing entry refreshed (same hash).
    Refreshed,
    /// Stored, and the oldest entry was evicted to make room.
    StoredEvictingOldest,
    /// Classified Intractable → absorbing eviction (tombstoned; any
    /// existing entry with this hash removed).
    RetiredIntractable,
    /// Refused: this hash is already tombstoned (absorbing state).
    RefusedRetired,
}

/// Regret-scored content memory — bounded, tombstone-guarded, oldest-first
/// eviction. Construction allocates the two rings once; steady state is
/// allocation-free (entries are `Copy`, eviction compacts in place).
///
/// The absorbing property: once content is classified Intractable its hash
/// enters the retired ring and is never re-admitted (until the ring itself
/// wraps — a bounded approximation, documented honestly: at retired-ring
/// capacity the OLDEST tombstone is overwritten, so "never" is really
/// "until `retired_capacity` newer retirements have passed").
pub struct RegretMemory {
    entries: Vec<RegretMemoryEntry>,
    retired: Vec<[u8; 32]>,
}

impl RegretMemory {
    /// New memory holding up to `capacity` live entries and
    /// `retired_capacity` intractable tombstones.
    pub fn new(capacity: usize, retired_capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            retired: Vec::with_capacity(retired_capacity.max(1)),
        }
    }

    /// Live entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no live entries are held.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate live entries (insertion order = age order, oldest first).
    pub fn iter(&self) -> std::slice::Iter<'_, RegretMemoryEntry> {
        self.entries.iter()
    }

    /// Look up a live entry by content hash.
    pub fn get(&self, content_hash: &[u8; 32]) -> Option<&RegretMemoryEntry> {
        self.entries
            .iter()
            .find(|e| &e.content_hash == content_hash)
    }

    /// True when this content hash is tombstoned as Intractable.
    pub fn retired(&self, content_hash: &[u8; 32]) -> bool {
        self.retired.iter().any(|h| h == content_hash)
    }

    /// Tombstone count (bounded by the retired ring's capacity).
    pub fn retired_len(&self) -> usize {
        self.retired.len()
    }

    /// Observe one probe result under its triaged regime.
    ///
    /// - [`Regime::Intractable`] → absorbing eviction: remove any live
    ///   entry with this hash, push the hash onto the retired ring
    ///   (overwriting the oldest tombstone at capacity), return
    ///   [`ObserveOutcome::RetiredIntractable`] — or
    ///   [`ObserveOutcome::RefusedRetired`] if already tombstoned.
    /// - Otherwise → insert or refresh the entry; at capacity evict the
    ///   entry with the smallest `last_seen_tick` (oldest-first).
    pub fn observe(&mut self, entry: RegretMemoryEntry, regime: Regime) -> ObserveOutcome {
        if self.retired(&entry.content_hash) {
            return ObserveOutcome::RefusedRetired;
        }
        if regime == Regime::Intractable {
            self.entries
                .retain(|e| e.content_hash != entry.content_hash);
            self.push_tombstone(entry.content_hash);
            return ObserveOutcome::RetiredIntractable;
        }
        if let Some(slot) = self
            .entries
            .iter_mut()
            .find(|e| e.content_hash == entry.content_hash)
        {
            *slot = entry;
            return ObserveOutcome::Refreshed;
        }
        let evicted = self.entries.len() >= self.entries.capacity();
        if evicted {
            // Oldest-first: drop the minimum last_seen_tick entry in place.
            if let Some((idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|&(i, e)| (e.last_seen_tick, i))
            {
                self.entries.remove(idx);
            }
        }
        self.entries.push(entry);
        if evicted {
            ObserveOutcome::StoredEvictingOldest
        } else {
            ObserveOutcome::Stored
        }
    }

    /// Most-salient live entries, into caller-owned scratch (zero-alloc
    /// steady state): fills `out` with references ordered by
    /// [`salience`] descending (ties: older `last_seen_tick` first —
    /// canonical).
    pub fn most_salient_into<'a>(
        &'a self,
        now_tick: u64,
        lambda: f32,
        out: &mut Vec<&'a RegretMemoryEntry>,
    ) {
        out.clear();
        out.extend(self.entries.iter());
        out.sort_by(|&a, &b| {
            let sa = salience(a.r_hat, a.last_seen_tick, now_tick, lambda);
            let sb = salience(b.r_hat, b.last_seen_tick, now_tick, lambda);
            sb.total_cmp(&sa)
                .then(a.last_seen_tick.cmp(&b.last_seen_tick))
        });
    }

    fn push_tombstone(&mut self, hash: [u8; 32]) {
        if self.retired.len() == self.retired.capacity() && !self.retired.is_empty() {
            // Ring overwrite: drop the oldest tombstone.
            self.retired.remove(0);
        }
        self.retired.push(hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    #[test]
    fn beta_lcb_order_descends_with_index_tiebreak() {
        // 18/20 scores above 1/1 (Bench 028's canonical pair).
        let scores = [(1u32, 1u32), (18, 2), (0, 0), (5, 5)];
        let (order, lcbs) = beta_lcb_order(&scores, FRONTIER_EPSILON);
        // 1/1 → 0.224..., 18/2 → 0.729..., 0/0 → 0.05, 5/5 → ~0.35
        assert_eq!(order[0], 1, "18/2 must lead");
        // Descending check through the candidate-indexed lcbs.
        let by_rank: Vec<f32> = order.iter().map(|&i| lcbs[i]).collect();
        for w in by_rank.windows(2) {
            assert!(
                w[0] >= w[1],
                "rank-ordered lcbs not descending: {by_rank:?}"
            );
        }
        // Sanity of the delegate: lcbs[i] matches best_belief_score directly.
        for (i, &(s, f)) in scores.iter().enumerate() {
            assert!((lcbs[i] - beta_lcb(s, f, FRONTIER_EPSILON)).abs() < 1e-7);
        }
        // Ascending view (weakness-slice diagnosis) is the exact reverse
        // under no ties.
        let mut asc: Vec<usize> = order.clone();
        asc.reverse();
        let asc_lcbs: Vec<f32> = asc.iter().map(|&i| lcbs[i]).collect();
        for w in asc_lcbs.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn beta_lcb_ties_break_by_index_ascending() {
        // Identical stats on every entry → order must be 0,1,2,...
        let scores = [(3u32, 3u32), (3, 3), (3, 3)];
        let (order, _) = beta_lcb_order(&scores, FRONTIER_EPSILON);
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn salience_decays_with_staleness_and_tracks_regret() {
        // Fresh high-regret beats fresh low-regret.
        assert!(salience(0.8, 100, 100, 0.01) > salience(0.2, 100, 100, 0.01));
        // Same regret: stale sinks below fresh.
        assert!(salience(0.8, 100, 100, 0.01) > salience(0.8, 0, 100, 0.01));
        // Very stale → salience → 0 (sigmoid(-λ·Δt) → 0), never negative.
        let s = salience(0.8, 0, 10_000, 0.01);
        assert!((0.0..1e-6).contains(&s));
    }

    #[test]
    fn memory_stores_refreshes_and_evicts_oldest_first() {
        let mut mem = RegretMemory::new(2, 4);
        let e1 = RegretMemoryEntry {
            content_hash: h(1),
            r_hat: 0.5,
            ci: 0.1,
            skill_tag_bits: 0,
            last_seen_tick: 10,
        };
        let e2 = RegretMemoryEntry {
            content_hash: h(2),
            r_hat: 0.6,
            ci: 0.1,
            skill_tag_bits: 0,
            last_seen_tick: 20,
        };
        let e3 = RegretMemoryEntry {
            content_hash: h(3),
            r_hat: 0.7,
            ci: 0.1,
            skill_tag_bits: 0,
            last_seen_tick: 30,
        };
        assert_eq!(mem.observe(e1, Regime::Frontier), ObserveOutcome::Stored);
        assert_eq!(mem.observe(e2, Regime::Frontier), ObserveOutcome::Stored);
        // Refresh e1 at a NEWER tick — must not evict anything.
        let e1b = RegretMemoryEntry {
            content_hash: h(1),
            r_hat: 0.55,
            ci: 0.05,
            skill_tag_bits: 1,
            last_seen_tick: 40,
        };
        assert_eq!(
            mem.observe(e1b, Regime::Frontier),
            ObserveOutcome::Refreshed
        );
        assert_eq!(mem.len(), 2);
        assert_eq!(mem.get(&h(1)).unwrap().r_hat, 0.55);
        // Insert e3: oldest is now e2 (tick 20) → evicted.
        assert_eq!(
            mem.observe(e3, Regime::Frontier),
            ObserveOutcome::StoredEvictingOldest
        );
        assert_eq!(mem.len(), 2);
        assert!(mem.get(&h(2)).is_none());
        assert!(mem.get(&h(1)).is_some());
        assert!(mem.get(&h(3)).is_some());
    }

    #[test]
    fn intractable_eviction_is_absorbing() {
        let mut mem = RegretMemory::new(4, 2);
        let bad = RegretMemoryEntry {
            content_hash: h(9),
            r_hat: 0.0,
            ci: 0.0,
            skill_tag_bits: 0,
            last_seen_tick: 1,
        };
        // First Intractable observation → tombstone + any live copy removed.
        assert_eq!(
            mem.observe(bad, Regime::Intractable),
            ObserveOutcome::RetiredIntractable
        );
        assert!(mem.retired(&h(9)));
        assert_eq!(mem.len(), 0);
        // Re-observation under ANY regime is refused (absorbing).
        assert_eq!(
            mem.observe(bad, Regime::Frontier),
            ObserveOutcome::RefusedRetired
        );
        assert_eq!(mem.len(), 0);
        // Live entry later classified Intractable is removed + tombstoned.
        let e = RegretMemoryEntry {
            content_hash: h(1),
            r_hat: 0.5,
            ci: 0.1,
            skill_tag_bits: 0,
            last_seen_tick: 5,
        };
        mem.observe(e, Regime::Frontier);
        assert_eq!(
            mem.observe(e, Regime::Intractable),
            ObserveOutcome::RetiredIntractable
        );
        assert_eq!(mem.len(), 0);
        assert!(mem.retired(&h(1)));
    }

    #[test]
    fn retired_ring_overwrites_oldest_at_capacity() {
        let mut mem = RegretMemory::new(2, 2);
        for (seed, tick) in [(1u8, 1u64), (2, 2), (3, 3)] {
            let e = RegretMemoryEntry {
                content_hash: h(seed),
                r_hat: 0.0,
                ci: 0.0,
                skill_tag_bits: 0,
                last_seen_tick: tick,
            };
            mem.observe(e, Regime::Intractable);
        }
        // Capacity 2: h(1) overwritten by h(3) — bounded approximation.
        assert_eq!(mem.retired_len(), 2);
        assert!(!mem.retired(&h(1)));
        assert!(mem.retired(&h(2)));
        assert!(mem.retired(&h(3)));
    }

    #[test]
    fn most_salient_orders_fresh_high_regret_first() {
        let mut mem = RegretMemory::new(8, 4);
        let mk = |seed: u8, r: f32, t: u64| RegretMemoryEntry {
            content_hash: h(seed),
            r_hat: r,
            ci: 0.05,
            skill_tag_bits: 0,
            last_seen_tick: t,
        };
        mem.observe(mk(1, 0.9, 100), Regime::Frontier); // fresh + high regret
        mem.observe(mk(2, 0.1, 100), Regime::Frontier); // fresh + low regret
        mem.observe(mk(3, 0.9, 10), Regime::Frontier); // stale + high regret
        let mut out = Vec::new();
        mem.most_salient_into(100, 0.01, &mut out);
        // Salience = r_hat * sigmoid(-lambda*dt):
        //   h(1): 0.9 * sigmoid(0)      = 0.45
        //   h(3): 0.9 * sigmoid(-0.9)   = 0.26
        //   h(2): 0.1 * sigmoid(0)      = 0.05
        // High regret dominates staleness at this λ; staleness breaks ties
        // WITHIN a regret level (h(1) ahead of h(3)).
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].content_hash, h(1));
        assert_eq!(out[1].content_hash, h(3));
        assert_eq!(out[2].content_hash, h(2));
    }
}

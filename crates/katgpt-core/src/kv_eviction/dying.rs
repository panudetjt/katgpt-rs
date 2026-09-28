//! Dying — the dual-threshold staleness death metric (Issue 873 primitive C;
//! Research 581 mechanism row #5, distilled from volotat/mini-AGI
//! `paged.py:284-325` @ `96784b7`, MIT). The delete-side complement of this
//! directory's keep-side usage-rate scoring ([`super::score`]:
//! `cum_mass/max(1,age)` ranks whom to KEEP by usage; dying ranks what to
//! STOP FEEDING and then DELETE by staleness — different signals, opposite
//! verdicts, one directory).
//!
//! # The metric (ONE definition, read at TWO thresholds)
//!
//! ```text
//! d = min(now − last_addressed, own_age) / window     (0 if on trial)
//! d ≥ BRAKE (0.75) → stop growth/feeding (Verdict::Braking)
//! d ≥ 1.0          → delete               (Verdict::Dead)
//! ```
//!
//! ONE definition at two thresholds is the whole point: brakes and prunes
//! can never disagree about "dead" because they read the same number — a
//! separate "should_prune" predicate is exactly the drift this shape
//! refuses. `d` saturates at ≥ 1.0 via the clamp (min caps the numerator at
//! `own_age`, and items older than the window read 1.0-or-more → the Dead
//! set is exactly the Braking set plus time).
//!
//! # The two newborn protections (both measured the hard way upstream)
//!
//! 1. **own-age clamp** — `min(·, own_age)`: an item whose `last_addressed`
//!    predates its bookkeeping (the never-addressed default `0`) scores by
//!    AGE, not by distance-from-epoch. Without the clamp a never-addressed
//!    newborn reads `(now − 0)/window` — dead from birth.
//! 2. **trial pinning** — `d = 0` while `now < trial_until`: a newborn on a
//!    survival trial cannot die before its trial ends (mini-AGI's
//!    speculative-growth survival trials, Research 581 row #13). The trial
//!    window is caller policy ([`DeathConfig::trial_len`]).
//!
//! # The anti-predictive-magnitude lesson (`paged.py:292-299`)
//!
//! The busiest items carry the smallest gates — historical magnitude is
//! ANTI-predictive of being wanted again. This metric deliberately reads NO
//! magnitude: [`DeathRow`] carries no usage counter, no cum_mass, no score.
//! A high-magnitude stale item and a zero-magnitude stale item read the
//! SAME death score. The G1 falsifier arm
//! ([`magnitude_is_anti_predictive_and_ignored`]) pins this: the death
//! ranking orders by recency while a magnitude ranking orders oppositely.
//!
//! # Window calibration
//!
//! mini-AGI self-calibrates the window's unit from the system's own
//! cumulative counters (`segments/step`). The primitive takes `window` in
//! TICKS (caller-scaled); [`calibrate_window`] converts an event-rate
//! calibration (cumulative events, elapsed ticks, target window in events)
//! into that tick unit — pure arithmetic, the caller owns the counters.
//!
//! # Sync boundary
//!
//! None — pure scoring over caller-owned state (the [`super`] posture).
//! `d` is a raw deterministic scalar; safe to log, gate, or sync.
//!
//! # Consumers (recorded, NOT wired here — they file their own)
//!
//! riir-neuron-db `shard_compactor` freeze gating (brake = stop
//! consolidation feeding, dead = freeze/prune candidate); riir-clippy
//! corpus rule retirement (`frontier_report` deliberately never deletes —
//! this is its missing deletion half); belief GC in the cognition stack.

/// Brake threshold: at/above this fraction of the window, stop spending on
/// the item (growth, feeding, consolidation). Source: mini-AGI `paged.py`
/// (`0.75`). Below delete — braking is the earlier, reversible verdict.
pub const BRAKE_THRESHOLD: f32 = 0.75;

/// Death config. Knobs are config, NEVER adaptive (the Issue-033 law).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeathConfig {
    /// Staleness window in TICKS: unaddressed for a full window ⇒ `d ≥ 1.0`
    /// ⇒ [`Verdict::Dead`]. Scale to the caller's tick cadence
    /// ([`calibrate_window`] converts an event-rate calibration).
    pub window: u64,
    /// Survival-trial length in ticks from admission: `d` pinned `0` until
    /// `admission_tick + trial_len` (a newborn cannot die on trial).
    pub trial_len: u64,
}

impl DeathConfig {
    /// Both knobs explicit — there is no honest default for an unknown tick
    /// cadence (the [`crate::pool_admission::PoolAdmissionConfig`] posture).
    pub const fn new(window: u64, trial_len: u64) -> Self {
        Self { window, trial_len }
    }
}

/// Per-item death bookkeeping. The CALLER owns the rows (side table over
/// whatever the items are — KV slots, shards, corpus rules, beliefs).
///
/// Deliberately carries **no magnitude field** (the anti-predictive lesson —
/// see the module doc): usage mass lives in the keep-side [`super::UsageRow`]
/// beside it, and the two rankings are kept separate on purpose.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DeathRow {
    /// Tick at which the item was born (admitted/created).
    pub admission_tick: u64,
    /// Last tick at which the item was addressed (used/touched/hit).
    /// The never-addressed default `0` is safe: the own-age clamp caps the
    /// numerator at `own_age`, so the item dies by AGE, not by epoch.
    pub last_addressed: u64,
}

impl DeathRow {
    /// A row born and immediately addressed at `tick`.
    pub fn born_at(tick: u64) -> Self {
        Self {
            admission_tick: tick,
            last_addressed: tick,
        }
    }

    /// Record an addressing at `tick` (the touch that keeps the item alive).
    /// Monotone by contract: a backwards tick is ignored (staleness must
    /// never un-stale).
    pub fn addressed(&mut self, tick: u64) {
        if tick >= self.last_addressed {
            self.last_addressed = tick;
        }
    }
}

/// The one definition at its two thresholds. `#[repr(u8)]`: field-less,
/// hot-path small.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Verdict {
    /// `d < 0.75` — alive; keep spending normally.
    Alive = 0,
    /// `0.75 ≤ d < 1.0` — stop growth/feeding/consolidation (reversible).
    Braking = 1,
    /// `d ≥ 1.0` — delete (freeze/prune/retire candidate).
    Dead = 2,
}

/// The death score `d = min(now − last_addressed, own_age) / window`, pinned
/// `0` while on trial. O(1), NaN-impossible (integer numerator, `window ≥ 1`
/// enforced by the `max(1,·)` guard). A `window` of `0` is treated as `1`
/// (fail-loud is the caller's job at config time; the metric itself never
/// divides by zero).
#[inline]
pub fn death_score(row: &DeathRow, now: u64, config: &DeathConfig) -> f32 {
    let trial_until = row.admission_tick.saturating_add(config.trial_len);
    if now < trial_until {
        return 0.0;
    }
    let own_age = now.saturating_sub(row.admission_tick);
    let unaddressed = now.saturating_sub(row.last_addressed);
    let clamped = unaddressed.min(own_age);
    clamped as f32 / (config.window.max(1)) as f32
}

/// The ONE definition read at its TWO thresholds. Brakes and prunes consume
/// the same number — they cannot disagree about "dead".
#[inline]
pub fn verdict(row: &DeathRow, now: u64, config: &DeathConfig) -> Verdict {
    let d = death_score(row, now, config);
    if d >= 1.0 {
        Verdict::Dead
    } else if d >= BRAKE_THRESHOLD {
        Verdict::Braking
    } else {
        Verdict::Alive
    }
}

/// Batch scan: verdicts for `rows[..live]` at `now` into the caller-owned
/// `out` buffer (reused — no per-step allocation; the
/// [`super::UsageScoreTable::scores`] shape). Also returns the Dead count
/// (the prune-candidate population) — the one number every consumer's
/// delete pass starts from.
pub fn verdicts_into(
    rows: &[DeathRow],
    live: usize,
    now: u64,
    config: &DeathConfig,
    out: &mut Vec<Verdict>,
) -> usize {
    out.clear();
    let mut dead = 0usize;
    for row in &rows[..live.min(rows.len())] {
        let v = verdict(row, now, config);
        if v == Verdict::Dead {
            dead += 1;
        }
        out.push(v);
    }
    dead
}

/// Window calibration from the system's own cumulative counters (the
/// mini-AGI `segments/step` self-calibration, generalized): a target window
/// of `target_events` addressing events, measured against `cum_events`
/// events observed over `elapsed_ticks` ticks, converts to a window in
/// TICKS via `target_events × (elapsed_ticks / cum_events)`. Pure
/// arithmetic; zero events ⇒ `None` (no calibration exists yet — never
/// invent a window).
pub fn calibrate_window(cum_events: u64, elapsed_ticks: u64, target_events: u64) -> Option<u64> {
    if cum_events == 0 {
        return None;
    }
    // u128 intermediate: target_events × elapsed can exceed u64 at extreme
    // ratios; the result is floored back into u64.
    let ticks = (target_events as u128 * elapsed_ticks as u128) / cum_events as u128;
    u64::try_from(ticks).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(window: u64, trial: u64) -> DeathConfig {
        DeathConfig::new(window, trial)
    }

    #[test]
    fn fresh_item_is_alive() {
        let row = DeathRow::born_at(1000);
        assert_eq!(verdict(&row, 1000, &config(100, 10)), Verdict::Alive);
        assert_eq!(death_score(&row, 1000, &config(100, 10)), 0.0);
    }

    #[test]
    fn thresholds_read_one_definition() {
        // window 100: brake at ≥75 ticks unaddressed, dead at ≥100.
        let born = 0u64;
        let row = DeathRow {
            admission_tick: born,
            last_addressed: 0,
        };
        let cfg = config(100, 0);
        assert_eq!(verdict(&row, 74, &cfg), Verdict::Alive);
        assert_eq!(verdict(&row, 75, &cfg), Verdict::Braking); // exactly at brake
        assert_eq!(verdict(&row, 99, &cfg), Verdict::Braking);
        assert_eq!(verdict(&row, 100, &cfg), Verdict::Dead); // exactly at window
        assert_eq!(verdict(&row, 10_000, &cfg), Verdict::Dead); // long stale stays dead
    }

    #[test]
    fn addressing_resets_staleness_monotonically() {
        let mut row = DeathRow::born_at(0);
        let cfg = config(100, 0);
        // stale to braking, then touched at 80 → alive again from 80.
        assert_eq!(verdict(&row, 80, &cfg), Verdict::Braking);
        row.addressed(80);
        assert_eq!(verdict(&row, 80, &cfg), Verdict::Alive);
        assert_eq!(verdict(&row, 155, &cfg), Verdict::Braking); // 75 since touch
        assert_eq!(verdict(&row, 180, &cfg), Verdict::Dead); // 100 since touch
        // A backwards touch never un-stales.
        row.addressed(3);
        assert_eq!(verdict(&row, 180, &cfg), Verdict::Dead);
    }

    #[test]
    fn never_addressed_dies_by_age_not_epoch() {
        // The own-age clamp: the never-addressed default (last_addressed=0)
        // must not read (now − 0)/window — dead from birth. It reads
        // own_age/window.
        let row = DeathRow {
            admission_tick: 5_000,
            last_addressed: 0,
        };
        let cfg = config(100, 0);
        assert_eq!(death_score(&row, 5_074, &cfg), 0.74); // by AGE (74/100), alive
        assert_eq!(verdict(&row, 5_074, &cfg), Verdict::Alive);
        assert_eq!(verdict(&row, 5_075, &cfg), Verdict::Braking); // 75/100 by age
        assert_eq!(verdict(&row, 5_100, &cfg), Verdict::Dead); // 100/100 by age
    }

    #[test]
    fn trial_pinning_a_newborn_cannot_die_on_trial() {
        // Even fully stale (never addressed) and past the window, a newborn
        // on trial reads 0 until trial end — then the metric applies at
        // once (no second ramp: the clamped staleness is already ≥ window).
        let row = DeathRow {
            admission_tick: 1_000,
            last_addressed: 1_000,
        };
        let cfg = config(100, 500);
        assert_eq!(death_score(&row, 1_499, &cfg), 0.0); // on trial, though stale
        assert_eq!(verdict(&row, 1_499, &cfg), Verdict::Alive);
        // Trial ends at 1500; own_age = 500 ≥ window ⇒ clamped = 500 → dead
        // immediately at trial end.
        assert_eq!(verdict(&row, 1_500, &cfg), Verdict::Dead);
        // A trial LONGER than the staleness: touch during the trial, then
        // the post-trial score resumes from the touch, not from birth.
        let mut row2 = DeathRow {
            admission_tick: 1_000,
            last_addressed: 1_000,
        };
        let cfg2 = config(100, 500);
        row2.addressed(1_450);
        assert_eq!(verdict(&row2, 1_500, &cfg2), Verdict::Alive); // 50 since touch
        assert_eq!(verdict(&row2, 1_525, &cfg2), Verdict::Braking); // 75 since touch
        assert_eq!(verdict(&row2, 1_550, &cfg2), Verdict::Dead); // 100 since touch
    }

    #[test]
    fn trial_end_with_recent_touch_is_alive() {
        // The pin drops at trial end but the score is computed from
        // last_addressed — an item actively used THROUGH its trial stays
        // alive at the boundary (no cliff).
        let mut row = DeathRow::born_at(0);
        let cfg = config(100, 1000);
        row.addressed(999);
        assert_eq!(verdict(&row, 1000, &cfg), Verdict::Alive); // 1 tick stale
        assert_eq!(verdict(&row, 1073, &cfg), Verdict::Alive); // 74 since touch
        assert_eq!(verdict(&row, 1074, &cfg), Verdict::Braking); // 75 since touch
    }

    #[test]
    fn magnitude_is_anti_predictive_and_ignored() {
        // C2 — THE falsifier arm. Busiest items carry the smallest gates:
        // magnitude is ANTI-predictive of being wanted again. The death
        // metric must rank by RECENCY while a magnitude ranking orders
        // OPPOSITELY. DeathRow carries no magnitude field at all — the
        // disagreement is unrepresentable in the type, and the arm pins the
        // behavior: the historically-hottest item is the dead one.
        let mut hot_but_stale = DeathRow::born_at(0); // magnitude 1e9 (imagined)
        hot_but_stale.addressed(10); // touched ONCE long ago
        let cold_but_current = DeathRow::born_at(500); // magnitude 1e-9
        let cold_but_current = {
            let mut r = cold_but_current;
            r.addressed(990); // touched just now
            r
        };
        let cfg = config(100, 0);
        // A magnitude ranker puts hot_but_stale FIRST (most used). The
        // death metric puts it DEAD and the low-magnitude item ALIVE —
        // the rankings disagree exactly as the lesson demands.
        assert_eq!(verdict(&hot_but_stale, 1000, &cfg), Verdict::Dead);
        assert_eq!(verdict(&cold_but_current, 1000, &cfg), Verdict::Alive);
    }

    #[test]
    fn batch_scan_counts_dead_into_caller_buffer() {
        let rows = vec![
            DeathRow::born_at(0), // stale forever → Dead
            DeathRow {
                admission_tick: 0,
                last_addressed: 999,
            }, // current → Alive
            DeathRow {
                admission_tick: 500,
                last_addressed: 500,
            }, // braking (age 500/window? no — see config)
        ];
        // window 1000, no trial: row0 unaddressed 1000 ≥ 1000 → Dead;
        // row1 unaddressed 1 → Alive; row2 unaddressed 500, age 500 → 0.5 → Alive.
        let cfg = config(1000, 0);
        let mut out = Vec::new();
        let dead = verdicts_into(&rows, rows.len(), 1000, &cfg, &mut out);
        assert_eq!(out, vec![Verdict::Dead, Verdict::Alive, Verdict::Alive]);
        assert_eq!(dead, 1);
        // Reused buffer: second call clears + rewrites (no growth beyond len).
        let dead2 = verdicts_into(&rows, rows.len(), 1000, &cfg, &mut out);
        assert_eq!(dead2, 1);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn zero_window_is_guarded_not_panicking() {
        let row = DeathRow::born_at(0);
        let cfg = config(0, 0);
        assert!(death_score(&row, 10, &cfg).is_finite());
        assert_eq!(verdict(&row, 10, &cfg), Verdict::Dead);
    }

    #[test]
    fn calibration_converts_events_to_ticks() {
        // 1000 events over 100 ticks → 10 events/tick; a 500-event target
        // window ⇒ 50 ticks.
        assert_eq!(calibrate_window(1000, 100, 500), Some(50));
        // No events yet: no calibration exists (never invent a window).
        assert_eq!(calibrate_window(0, 100, 500), None);
        // Extreme ratio: the u128 intermediate exceeds u64 → None (a
        // non-representable window REFUSES rather than truncates — a window
        // that huge is config nonsense anyway).
        assert_eq!(calibrate_window(1, u64::MAX, u64::MAX), None);
    }

    /// G4 — the metric + batch scan are zero-alloc steady-state (the
    /// convergence_cadence pattern; counters debug-only by design).
    #[cfg(debug_assertions)]
    #[test]
    fn g4_alloc_free_steady_state() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        const N: usize = 256;
        let cfg = config(1_000, 100);
        let mut rows = vec![DeathRow::born_at(0); N];
        for (i, r) in rows.iter_mut().enumerate() {
            r.addressed(i as u64);
        }
        let mut out = Vec::with_capacity(N);
        reset_alloc_stats();
        for t in 0..1000u64 {
            let now = 5_000 + t;
            rows[(t as usize) % N].addressed(now); // keep one item current
            let dead = verdicts_into(&rows, N, now, &cfg, &mut out);
            let _ = std::hint::black_box((dead, &out));
        }
        let (count, _bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "steady-state dying verdicts must be zero-alloc, saw {count} allocs"
        );
    }
}

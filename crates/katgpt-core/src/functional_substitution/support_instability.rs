//! Support-instability regime detection + mode-factored state accessors.
//!
//! Distilled from **LpWM** — [arXiv:2608.22764](https://arxiv.org/abs/2608.22764)
//! (Kuang, Dagade, Le Lidec, Maes, Balestriero, LeCun — *LpWM: A Case for
//! Sparse Representations in World Models*, 2026-08-24), via Research 513,
//! filed as Issue 693. The paper's mode-factoring finding: in non-negative
//! sparse latent codes the binary **support** carries the discrete dynamics
//! regime (zone decode 94–99%, contact) while the **magnitudes** carry the
//! continuous within-regime state (position R² = 0.94), and **support
//! instability** `1 − J(z_t, z_{t+1})` (soft Jaccard = Ruzicka 1958,
//! `Σmin/Σmax`) spikes at regime transitions at near-zero cost.
//!
//! ## Relationship to `iou` (the kernel this consumes)
//!
//! [`super::iou`] implements Ruzicka similarity exactly — but consumes two
//! attention rows at the SAME timestep (real head vs surrogate: the
//! functional-equivalence gate for head substitution, Plan 353 /
//! arXiv:2606.19317). This module is the **consecutive-tick temporal
//! consumer**: the same entity's non-negative latent state at `t` vs
//! `t + 1`. Kernel identical, semantic disjoint — the missing composition
//! Research 513 §Path-0 identified (four shipped detectors, four signal
//! classes; the cheap state-side one was absent).
//!
//! ## Signal class vs the shipped detector cousins
//!
//! | Detector | Signal | Cost shape |
//! |---|---|---|
//! | this module | state-side consecutive-tick support overlap | one `iou` + ring update |
//! | KARC surprise (`karc`) | forecast error (needs a fitted forecaster) | basis expand + matvec per tick |
//! | `stiff_anomaly` (katgpt-spectral) | eigenvalue window vs frozen baseline | eigendecomp + window buffering |
//! | ICT branching (riir-ai) | JS-divergence over K sampled action dists | K samples per tick |
//!
//! ## Non-negativity contract + signed-state bridge
//!
//! Inputs are non-negative latent states (ReLU/clamp0 outputs, attention
//! rows, `style_weights`). Like [`super::iou`], the hot path performs NO
//! validation — bridging signed states (DEC cochains, signed HLA deltas) is
//! a caller-side `x.max(0.0)` (or `x.abs()` when signed magnitude is the
//! intended semantics) applied BEFORE the call. The bridge is documented
//! here and costs nothing in this module.
//!
//! ## Tick-indexed configuration (the `decay_confidence` pattern)
//!
//! All detector timing is tick-indexed: one instability value pushed per
//! tick, fixed-size ring, no wall clock (`Instant` appears nowhere in the
//! logic — deterministic-replay-safe, stateless-friendly: store level +
//! tick, compute on read).
//!
//! ## GOAT verdict — Bench 685 (honest summary)
//!
//! - **G1 determinism**: PASS — bit-identical PoC timelines across
//!   independent runs (LCG-only construction).
//! - **G2 perf**: PASS — see `.benchmarks/685_support_regime_goat.md` for
//!   the measured ns/entity/tick at D=64 (well under the 100 ns budget).
//! - **G3 no-regression**: default build untouched (opt-in feature; implies
//!   `functional_substitution_gate`).
//! - **G4 alloc-free**: PASS — 0 allocations across steady-state push loops.
//! - **T3 quality PoC**: the pre-registered verdict is recorded in the bench
//!   note — whichever way it came out. The primitive ships opt-in; promotion
//!   is blocked on a riir-ai consumer (R352) per the no-default-consumer
//!   rule (the `evpi_gate` precedent).

/// Fire threshold: the debounced detector enters [`DetectorState::Firing`]
/// when the window mean of instability rises strictly above this.
///
/// Pre-registered default (Issue 693 T3). LpWM's codes are 30–65% active
/// (Table 3, not ultra-sparse), so thresholds are set for moderate-density
/// supports, not binary-sparse ones.
pub const THETA_FIRE: f32 = 0.6;

/// Calm threshold: hysteresis floor. The detector returns to
/// [`DetectorState::Calm`] only when the window mean drops strictly below
/// this — strictly below [`THETA_FIRE`] by construction (callers may assert
/// the ordering; `with_params` documents it).
pub const THETA_CALM: f32 = 0.35;

/// Pre-registered debounce window (ticks): the detector averages the last
/// K pushed instability values before comparing against the thresholds.
pub const DEFAULT_WINDOW: usize = 3;

/// Ring capacity. The window must satisfy `window <= MAX_WINDOW`;
/// [`SupportInstabilityDetector::with_params`] clamps, so no configuration
/// can panic or grow the struct.
pub const MAX_WINDOW: usize = 8;

/// Debounced detector state. `#[repr(u8)]` so it is sync/wire-cheap
/// (1 byte).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DetectorState {
    /// Window mean at or below the fire threshold (regime stable).
    Calm = 0,
    /// Window mean above the fire threshold (regime transition underway).
    Firing = 1,
}

/// Support instability between consecutive non-negative latent states:
/// `1.0 − iou(z_t, z_{t+1})` (Ruzicka similarity complement).
///
/// Returns `1.0` (maximally unstable) when the states are incomparable
/// (length mismatch) or both all-zero — the inherited
/// [`super::iou`] conventions. Inputs are non-negative by contract (see
/// module docs for the signed-state bridge); no validation on the hot path.
#[inline]
pub fn support_instability(z_t: &[f32], z_t1: &[f32]) -> f32 {
    1.0 - super::iou::iou(z_t, z_t1)
}

/// Zero-alloc debounced support-instability detector.
///
/// Fixed-size ring of the last [`MAX_WINDOW`] instability values + a
/// two-state hysteresis machine (fire on window mean > [`THETA_FIRE`] from
/// [`DetectorState::Calm`]; return to calm on window mean < [`THETA_CALM`]).
/// Tick-indexed: one [`push`](Self::push) per tick, no wall clock anywhere.
///
/// A cold detector trusts its partial window (the mean is over
/// `min(len, window)` values) — no panic on a short history, no hidden
/// warm-up counter.
#[derive(Clone, Debug)]
pub struct SupportInstabilityDetector {
    ring: [f32; MAX_WINDOW],
    head: usize,
    len: usize,
    state: DetectorState,
    theta_fire: f32,
    theta_calm: f32,
    window: usize,
}

impl SupportInstabilityDetector {
    /// Construct with the pre-registered defaults
    /// ([`THETA_FIRE`], [`THETA_CALM`], [`DEFAULT_WINDOW`]).
    pub const fn new() -> Self {
        Self {
            ring: [0.0; MAX_WINDOW],
            head: 0,
            len: 0,
            state: DetectorState::Calm,
            theta_fire: THETA_FIRE,
            theta_calm: THETA_CALM,
            window: DEFAULT_WINDOW,
        }
    }

    /// Construct with explicit thresholds + window (tests / tuning).
    ///
    /// `window` is clamped to `1..=MAX_WINDOW`; `theta_calm` is clamped to
    /// be strictly below `theta_fire` when the caller inverts the ordering
    /// (hysteresis requires `theta_calm < theta_fire`). No panic path.
    pub fn with_params(theta_fire: f32, theta_calm: f32, window: usize) -> Self {
        let theta_calm = if theta_calm < theta_fire {
            theta_calm
        } else {
            theta_fire - f32::EPSILON
        };
        Self {
            window: window.clamp(1, MAX_WINDOW),
            theta_fire,
            theta_calm,
            ..Self::new()
        }
    }

    /// Push one tick's instability value; returns the post-push state.
    ///
    /// A **fire** is the returned state being [`DetectorState::Firing`] with
    /// the pre-push state [`DetectorState::Calm`] — callers track the edge
    /// (the PoC in `.benchmarks/685` shows the two-line pattern).
    pub fn push(&mut self, instability: f32) -> DetectorState {
        self.ring[self.head] = instability;
        self.head = (self.head + 1) % MAX_WINDOW;
        if self.len < MAX_WINDOW {
            self.len += 1;
        }
        let mean = self.window_mean();
        self.state = match self.state {
            DetectorState::Calm if mean > self.theta_fire => DetectorState::Firing,
            DetectorState::Firing if mean < self.theta_calm => DetectorState::Calm,
            state => state,
        };
        self.state
    }

    /// Current state without pushing.
    pub fn state(&self) -> DetectorState {
        self.state
    }

    /// Mean over the last `min(len, window)` pushed values (0.0 when the
    /// ring is empty — no divide-by-zero, no panic).
    pub fn window_mean(&self) -> f32 {
        let n = self.len.min(self.window);
        if n == 0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        let mut idx = (self.head + MAX_WINDOW - 1) % MAX_WINDOW;
        for _ in 0..n {
            sum += self.ring[idx];
            idx = (idx + MAX_WINDOW - 1) % MAX_WINDOW;
        }
        sum / n as f32
    }
}

impl Default for SupportInstabilityDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// T2 — mode-factored accessors
// ──────────────────────────────────────────────────────────────────────────

/// Hard support bitmask over a non-negative state, `D ≤ 128` (two `u64`
/// words). The **discrete regime half** of the LpWM mode factoring: bit `i`
/// is set iff `z[i] > eps` (default `eps = 0.0` matches the exact-zero
/// sparsity the paper's (Rep)ReLU link produces).
///
/// Compare masks with exact [`jaccard`](Self::jaccard) (popcount over
/// words — cheaper and sharper than the soft [`super::iou`] when the states
/// are genuinely binary-sparse); read intensity through
/// [`magnitudes`]. Both-empty masks compare as `0.0` (the `iou` convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupportMask {
    words: [u64; 2],
    active: u32,
}

impl SupportMask {
    /// Build from a non-negative state. Returns `None` when `z` is empty or
    /// longer than 128 dims (the struct has fixed capacity; refusing is
    /// louder than truncating).
    pub fn from_state(z: &[f32], eps: f32) -> Option<Self> {
        if z.is_empty() || z.len() > 128 {
            return None;
        }
        let mut words = [0u64; 2];
        for (i, &v) in z.iter().enumerate() {
            if v > eps {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        let active = words[0].count_ones() + words[1].count_ones();
        Some(Self { words, active })
    }

    /// Number of active dims (popcount).
    pub fn active(&self) -> u32 {
        self.active
    }

    /// Active fraction relative to the 128-bit CAPACITY (the struct carries
    /// no per-call dim count). For the LpWM "codes 30–65% active" anchor on
    /// a `D < 128` state, scale by `128 / D` at the call site, or compare
    /// [`active`](Self::active) counts directly.
    pub fn active_fraction(&self) -> f32 {
        self.active as f32 / 128.0
    }

    /// Exact Jaccard `|A ∩ B| / |A ∪ B|` via popcount. Both-empty → `0.0`
    /// (the `iou` "no overlap when nothing is active" convention).
    pub fn jaccard(&self, other: &Self) -> f32 {
        let inter = (self.words[0] & other.words[0]).count_ones()
            + (self.words[1] & other.words[1]).count_ones();
        let union = (self.words[0] | other.words[0]).count_ones()
            + (self.words[1] | other.words[1]).count_ones();
        if union == 0 {
            return 0.0;
        }
        inter as f32 / union as f32
    }
}

/// Iterator over the active magnitudes of a non-negative state — the
/// **continuous half** of the mode factoring: yields `(index, value)` for
/// every `z[i] > 0.0`. Zero-alloc (a filtered slice iterator); sign-bridged
/// states should clamp before calling so inactive dims are genuinely zero.
pub fn magnitudes<'a>(z: &'a [f32]) -> impl Iterator<Item = (usize, f32)> + 'a {
    z.iter()
        .enumerate()
        .filter(|&(_, &v)| v > 0.0)
        .map(|(i, &v)| (i, v))
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── support_instability fn ──

    #[test]
    fn instability_identity_is_zero() {
        let a = [0.5f32, 0.3, 0.2];
        assert!(support_instability(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn instability_disjoint_is_one() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 0.0, 1.0];
        assert!((support_instability(&a, &b) - 1.0).abs() < 1e-6);
    }

    /// iou partial overlap = 1/3 → instability = 2/3 (hand-computed).
    #[test]
    fn instability_partial_known_value() {
        let a = [1.0f32, 1.0, 0.0, 0.0];
        let b = [1.0f32, 0.0, 1.0, 0.0];
        let got = support_instability(&a, &b);
        assert!((got - 2.0 / 3.0).abs() < 1e-6, "got {got}");
    }

    /// Length mismatch / both-empty inherit iou's 0.0 → instability 1.0
    /// (maximally unstable when incomparable — documented convention).
    #[test]
    fn instability_incomparable_is_one() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0];
        assert_eq!(support_instability(&a, &b), 1.0);
        let e: [f32; 0] = [];
        assert_eq!(support_instability(&e, &e), 1.0);
    }

    // ── detector: fire + hysteresis ──

    #[test]
    fn detector_fires_on_sustained_high_window() {
        let mut d = SupportInstabilityDetector::new();
        d.push(0.01);
        d.push(0.01);
        assert_eq!(d.state(), DetectorState::Calm);
        assert_eq!(d.push(0.95), DetectorState::Calm); // one high: mean 0.32
        assert_eq!(d.push(0.95), DetectorState::Firing); // two highs: mean 0.64
        assert_eq!(d.push(0.95), DetectorState::Firing); // mean 0.95
    }

    /// The load-bearing PoC semantics: a SINGLE 1-tick spike does not fire
    /// at the pre-registered defaults (window-3 mean over one spike + two
    /// lows ≈ 0.32 < 0.6). This is the debounce trading spike-width for
    /// noise-immunity — see Bench 685 §T3.
    #[test]
    fn detector_single_spike_does_not_fire_at_defaults() {
        let mut d = SupportInstabilityDetector::new();
        d.push(0.01);
        d.push(0.01);
        assert_eq!(d.push(0.95), DetectorState::Calm); // mean (0.01+0.01+0.95)/3
        assert_eq!(d.push(0.01), DetectorState::Calm);
    }

    /// Two consecutive spikes land in one 3-window → mean ≈ 0.64 > 0.6.
    #[test]
    fn detector_two_consecutive_spikes_fire() {
        let mut d = SupportInstabilityDetector::new();
        d.push(0.01);
        d.push(0.95);
        assert_eq!(d.push(0.95), DetectorState::Firing);
    }

    #[test]
    fn detector_hysteresis_holds_until_mean_below_calm() {
        let mut d = SupportInstabilityDetector::new();
        for _ in 0..3 {
            d.push(0.95);
        }
        assert_eq!(d.state(), DetectorState::Firing);
        // One low tick: mean (0.95+0.95+0.0)/3 = 0.633 > θ_calm → still firing.
        assert_eq!(d.push(0.0), DetectorState::Firing);
        // Second low: mean (0.95+0.0+0.0)/3 = 0.317 < 0.35 → calm.
        assert_eq!(d.push(0.0), DetectorState::Calm);
    }

    /// A cold detector trusts its partial window (documented semantic).
    #[test]
    fn detector_cold_partial_window_can_fire_immediately() {
        let mut d = SupportInstabilityDetector::new();
        assert_eq!(d.push(0.9), DetectorState::Firing);
    }

    #[test]
    fn detector_window_mean_matches_manual() {
        let mut d = SupportInstabilityDetector::new();
        d.push(0.2);
        d.push(0.4);
        d.push(0.6);
        d.push(0.8);
        // Window 3 → last three values 0.4, 0.6, 0.8.
        assert!(
            (d.window_mean() - 0.6).abs() < 1e-6,
            "got {}",
            d.window_mean()
        );
        assert!((d.window_mean() * 3.0 - 1.8).abs() < 1e-5);
    }

    #[test]
    fn detector_window_mean_empty_is_zero_no_panic() {
        let d = SupportInstabilityDetector::new();
        assert_eq!(d.window_mean(), 0.0);
        assert_eq!(d.state(), DetectorState::Calm);
    }

    #[test]
    fn detector_with_params_window_one_fires_on_single_spike() {
        let mut d = SupportInstabilityDetector::with_params(0.5, 0.2, 1);
        assert_eq!(d.push(0.01), DetectorState::Calm);
        assert_eq!(d.push(0.95), DetectorState::Firing);
    }

    /// `with_params` clamps: window > MAX_WINDOW is clamped (no panic), and
    /// an inverted threshold pair is repaired so hysteresis holds.
    #[test]
    fn detector_with_params_clamps_pathological_config() {
        let d = SupportInstabilityDetector::with_params(0.4, 0.9, 99);
        // theta_calm forced below theta_fire; window clamped to MAX_WINDOW.
        let mut d2 = d.clone();
        for _ in 0..MAX_WINDOW {
            d2.push(0.8);
        }
        assert_eq!(d2.state(), DetectorState::Firing);
        for _ in 0..MAX_WINDOW {
            d2.push(0.0);
        }
        assert_eq!(d2.state(), DetectorState::Calm);
        // Zero window clamps up to 1.
        let d3 = SupportInstabilityDetector::with_params(0.5, 0.2, 0);
        let mut d4 = d3;
        assert_eq!(d4.push(0.9), DetectorState::Firing);
    }

    /// Ring overwrite: more pushes than capacity keeps only the tail.
    #[test]
    fn detector_ring_wraps_capacity() {
        let mut d = SupportInstabilityDetector::with_params(2.0, 1.9, MAX_WINDOW);
        for i in 0..(MAX_WINDOW * 3) {
            d.push(i as f32 * 0.01);
        }
        // Last MAX_WINDOW pushes are 0.16..0.23 (i = 16..23).
        let expect = (16..24).map(|i| i as f32 * 0.01).sum::<f32>() / MAX_WINDOW as f32;
        assert!((d.window_mean() - expect).abs() < 1e-6);
    }

    #[test]
    fn detector_run_twice_bit_identical() {
        let seq: Vec<f32> = (0..64)
            .map(|i| {
                if i % 7 == 0 {
                    0.95
                } else {
                    0.02 + (i % 5) as f32 * 0.001
                }
            })
            .collect();
        let run = || {
            let mut d = SupportInstabilityDetector::new();
            let mut states = Vec::with_capacity(seq.len());
            let mut means = Vec::with_capacity(seq.len());
            for &v in &seq {
                states.push(d.push(v) as u8);
                means.push(d.window_mean().to_bits());
            }
            (states, means)
        };
        assert_eq!(run(), run());
    }

    // ── SupportMask ──

    #[test]
    fn mask_from_state_sets_bits_above_eps() {
        let z = [0.0f32, 0.5, 0.0, 2.0];
        let m = SupportMask::from_state(&z, 0.0).expect("4 dims valid");
        assert_eq!(m.active(), 2);
        assert!((m.active_fraction() - 2.0 / 128.0).abs() < 1e-6);
    }

    #[test]
    fn mask_eps_boundary_is_strictly_greater() {
        let z = [0.1f32, 0.1];
        let m = SupportMask::from_state(&z, 0.1).expect("valid");
        assert_eq!(m.active(), 0, "v == eps must NOT be active");
        let m2 = SupportMask::from_state(&z, 0.099).expect("valid");
        assert_eq!(m2.active(), 2);
    }

    #[test]
    fn mask_rejects_empty_and_over_long() {
        let e: [f32; 0] = [];
        assert!(SupportMask::from_state(&e, 0.0).is_none());
        let long = [0.0f32; 129];
        assert!(SupportMask::from_state(&long, 0.0).is_none());
        let max = [0.5f32; 128];
        assert!(SupportMask::from_state(&max, 0.0).is_some());
    }

    #[test]
    fn mask_jaccard_identity_disjoint_and_empty() {
        let a = SupportMask::from_state(&[1.0f32, 1.0, 0.0], 0.0).unwrap();
        assert!((a.jaccard(&a) - 1.0).abs() < 1e-6);
        let b = SupportMask::from_state(&[0.0f32, 0.0, 1.0], 0.0).unwrap();
        assert!(a.jaccard(&b).abs() < 1e-6);
        let e = SupportMask::from_state(&[0.0f32; 8], 0.0).unwrap();
        assert_eq!(e.jaccard(&e), 0.0, "both-empty → 0.0 (iou convention)");
    }

    /// Hand-computed: A = dims 0..16, B = dims 8..24 → |∩| = 8, |∪| = 24.
    #[test]
    fn mask_jaccard_partial_known_value() {
        let mut za = [0.0f32; 32];
        let mut zb = [0.0f32; 32];
        za.iter_mut().take(16).for_each(|v| *v = 1.0);
        zb.iter_mut().skip(8).take(16).for_each(|v| *v = 1.0);
        let a = SupportMask::from_state(&za, 0.0).unwrap();
        let b = SupportMask::from_state(&zb, 0.0).unwrap();
        let got = a.jaccard(&b);
        assert!((got - 8.0 / 24.0).abs() < 1e-6, "got {got}");
    }

    /// Cross-word bits (dim ≥ 64) are set + counted correctly.
    #[test]
    fn mask_crosses_the_second_word() {
        let mut z = [0.0f32; 100];
        z[63] = 1.0;
        z[64] = 1.0;
        z[99] = 1.0;
        let m = SupportMask::from_state(&z, 0.0).unwrap();
        assert_eq!(m.active(), 3);
        let n = SupportMask::from_state(&[0.0f32; 100], 0.0).unwrap();
        assert!(m.jaccard(&n).abs() < 1e-6);
    }

    // ── magnitudes ──

    #[test]
    fn magnitudes_yields_only_positive_with_index() {
        let z = [0.0f32, 0.25, 0.0, 1.5, 0.0];
        let got: Vec<(usize, f32)> = magnitudes(&z).collect();
        assert_eq!(got, vec![(1, 0.25), (3, 1.5)]);
    }

    #[test]
    fn magnitudes_empty_and_all_zero() {
        let e: [f32; 0] = [];
        assert_eq!(magnitudes(&e).count(), 0);
        assert_eq!(magnitudes(&[0.0f32; 16]).count(), 0);
    }

    /// The mode-factored round trip: support mask + magnitudes together
    /// reconstruct the original non-negative state exactly.
    #[test]
    fn mode_factoring_reconstructs_state() {
        let z = [0.0f32, 0.3, 0.0, 0.7, 1.2, 0.0];
        let mask = SupportMask::from_state(&z, 0.0).unwrap();
        let mut rebuilt = [0.0f32; 6];
        for (i, v) in magnitudes(&z) {
            rebuilt[i] = v;
        }
        assert_eq!(&rebuilt, &z);
        assert_eq!(mask.active() as usize, magnitudes(&z).count());
    }

    // ── G4 (debug-only: the lib test binary installs TrackingAllocator) ──

    /// G4: zero allocations across a steady-state detector loop —
    /// 10_000 pushes + the `iou` call that feeds them (D = 64).
    #[cfg(debug_assertions)]
    #[test]
    fn g4_detector_loop_alloc_free() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        // Sentinel: confirm the allocator is installed (the lib test binary
        // installs TrackingAllocator via TEST_GLOBAL_ALLOC).
        reset_alloc_stats();
        let _sentinel: Vec<u8> = vec![0u8; 256];
        let (sent_count, _) = get_alloc_stats();
        assert!(sent_count > 0, "TrackingAllocator not installed");

        let a = [0.5f32; 64];
        let b = [0.4f32; 64];
        let mut d = SupportInstabilityDetector::new();
        reset_alloc_stats();
        let mut sink = 0.0f32;
        for i in 0..10_000 {
            // Alternate the operand order so iou sees changing input without
            // any allocation (both arrays are stack-allocated).
            let inst = match i % 2 {
                0 => support_instability(&a, &b),
                _ => support_instability(&b, &a),
            };
            sink += inst + d.push(inst) as u8 as f32;
        }
        let (count, bytes) = get_alloc_stats();
        std::hint::black_box(&sink);
        assert_eq!(
            count, 0,
            "steady-state allocs leaked ({count} allocs, {bytes} B)"
        );
    }
}

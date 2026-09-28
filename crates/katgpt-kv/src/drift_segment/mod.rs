//! DriftSegmentStore — training-free drift-segmented multi-state memory (Issue 652).
//!
//! Modelless adaptation of "Dynamic Linear Attention"
//! ([arXiv:2606.10650](https://arxiv.org/abs/2606.10650)); research note:
//! [`katgpt-rs/.research/482_Dynamic_Linear_Attention_Drift_Segmented_Memory.md`](../../../.research/482_Dynamic_Linear_Attention_Drift_Segmented_Memory.md).
//!
//! DLA's mechanism is a memory **policy**, not a training contribution: a
//! per-token drift score opens a new memory state at semantic transitions, and
//! a capacity-K cache merges the adjacent lowest-density pair when full. Every
//! component has a shipped modelless analog, composed here into a capability
//! none has alone: **adaptive-resolution memory at fixed budget**.
//!
//! # Composition (three shipped primitives)
//!
//! | DLA component | Substrate consumed | Origin |
//! |---|---|---|
//! | `I_t` drift score | [`TemporalDerivativeKernel`] dual-EMA on the key stream | Plan 277 / Research 435 |
//! | capacity-K state cache | `SegmentStore`-shaped bounded slots | Plan 223b / Research 199 |
//! | adjacent pair-merge | `hope_compactor`'s pair-merge pattern (online + adjacency-restricted) | riir-neuron-db Plan 321 |
//! | query-gated readout | GRM sigmoid gating (`segment_checkpoint::gating`) | Plan 223b |
//!
//! # Deltas over the paper (Research 482 §2.3)
//!
//! 1. **Score in key space, not state space.** The paper's `I_t` is the
//!    relative Frobenius drift of the d×d state (cost ≈ the update itself).
//!    We score the relative drift of the dual-EMA key estimate — O(d) per
//!    token, and the same signal class (regime change ⇒ key-distribution
//!    shift).
//! 2. **Rising-edge boundary.** The kernel's `surprise_norm()` is a
//!    displacement signal that stays elevated for the slow EMA's timescale
//!    (~1/α_slow tokens) after a transition; the paper's per-token state
//!    *delta* spikes only at the transition itself. Firing the boundary on
//!    the rising edge (`score ≥ τ` while armed, disarm until it drops below
//!    τ) recovers single-fire-per-transition semantics from the displacement
//!    signal.
//! 3. **Modelless readout.** Learned per-slot λ → sigmoid-gated signed dot
//!    (`w_i = σ(β·(q·k̄ᵢ))·(q·k̄ᵢ)`) — the gated-linear-attention shape,
//!    sigmoid never softmax.
//!
//! # Density accounting
//!
//! Each slot carries `info_sum` (Σ per-token drift score) and `n_tokens`.
//! Slot density = `info_sum / n_tokens`. When capacity is full, the
//! **adjacent** pair with the lowest pair density
//! `(info_i + info_j)/(n_i + n_j)` is merged in place (additive states
//! compose linearly — why linear-attention states merge cheaply, unlike
//! softmax attention). Surprising spans (needles, regime heads) carry high
//! density and survive merges; boring spans merge first. Chronological order
//! is preserved by construction (merge fuses the pair's span; the tail
//! shifts left).
//!
//! # Zero-alloc
//!
//! Fixed arrays throughout: `[DriftSlot<D>; K]`, no per-token heap. Readout
//! writes into a caller-provided buffer. Merge reuses slot storage in place.

#![allow(clippy::needless_range_loop)] // const-generic index loops over [f32; D]

use katgpt_types::temporal::TemporalDerivativeKernel;

/// One memory slot: additive key/value accumulators + density bookkeeping.
///
/// `Copy` so the ring can shift/merge without allocation.
#[derive(Clone, Copy, Debug)]
pub struct DriftSlot<const D: usize> {
    /// Σ k_t over the slot's tokens (summary = `key_sum / n_tokens`).
    pub key_sum: [f32; D],
    /// Σ v_t over the slot's tokens (additive value state).
    pub val_sum: [f32; D],
    /// Tokens accumulated into this slot.
    pub n_tokens: u32,
    /// Σ per-token drift score (density numerator).
    pub info_sum: f32,
    /// First token position (inclusive).
    pub pos_start: u32,
    /// Last token position (inclusive).
    pub pos_end: u32,
}

impl<const D: usize> Default for DriftSlot<D> {
    fn default() -> Self {
        Self {
            key_sum: [0.0; D],
            val_sum: [0.0; D],
            n_tokens: 0,
            info_sum: 0.0,
            pos_start: 0,
            pos_end: 0,
        }
    }
}

impl<const D: usize> DriftSlot<D> {
    /// Mean information density of this slot (`info_sum / n_tokens`).
    ///
    /// Empty slots report `f32::INFINITY` (never the merge argmin target —
    /// only active slots are scanned).
    #[inline]
    pub fn density(&self) -> f32 {
        if self.n_tokens == 0 {
            f32::INFINITY
        } else {
            self.info_sum / self.n_tokens as f32
        }
    }

    /// Mean key (slot summary for gating), written into `out`.
    #[inline]
    pub fn mean_key_into(&self, out: &mut [f32; D]) {
        let inv = 1.0 / self.n_tokens.max(1) as f32;
        for d in 0..D {
            out[d] = self.key_sum[d] * inv;
        }
    }
}

/// Shared sigmoid-gated readout over any slot slice (Issue 652 T2).
///
/// `out = Σᵢ σ(β·(q·k̄ᵢ)) · (q·k̄ᵢ) · v̄ᵢ`
///
/// The same function serves all bench arms (single-state / fixed-LFU /
/// drift) so the GOAT comparison isolates the **slot policy**, not the
/// readout. Signed dot keeps it linear-attention-faithful (misaligned slots
/// contribute signed noise that cancels); the sigmoid gate is per-slot
/// independent — **never softmax**.
///
/// Zero-alloc: writes into caller-owned `out`.
pub fn sigmoid_gated_readout<const D: usize>(
    slots: &[DriftSlot<D>],
    query: &[f32; D],
    beta: f32,
    out: &mut [f32; D],
) {
    *out = [0.0; D];
    for s in slots.iter() {
        if s.n_tokens == 0 {
            continue;
        }
        let inv_n = 1.0 / s.n_tokens as f32;
        let mean_dot = katgpt_core::simd::simd_dot_f32(query, &s.key_sum, D) * inv_n;
        let w = katgpt_core::simd::fast_sigmoid(beta * mean_dot) * mean_dot;
        for d in 0..D {
            out[d] += w * s.val_sum[d] * inv_n;
        }
    }
}

/// Drift-segmented bounded multi-state memory.
///
/// See the [module docs](self) for the mechanism and provenance.
///
/// # Invariants
///
/// - `n_slots() <= K` always (capacity enforced by adjacent-density merge).
/// - Active slots are chronologically ordered and non-overlapping:
///   `slots[i].pos_end < slots[i+1].pos_start`.
/// - Σ `n_tokens` over active slots == tokens observed (merges conserve mass).
/// - No heap allocation on any path.
#[derive(Clone, Debug)]
pub struct DriftSegmentStore<const K: usize, const D: usize> {
    slots: [DriftSlot<D>; K],
    n_active: usize,
    kernel: TemporalDerivativeKernel<D>,
    /// Boundary threshold τ on the relative drift score.
    tau: f32,
    /// Readout gate inverse temperature β.
    beta: f32,
    /// Minimum tokens in the current slot before a boundary may fire
    /// (anti-spam cooldown for noisy scores oscillating near τ).
    min_segment: u32,
    /// Rising-edge state: `true` when the score is below τ (a new boundary
    /// may fire on the next crossing).
    armed: bool,
    /// Tokens observed so far.
    pos: u32,
    boundaries_fired: u64,
    merges_done: u64,
}

impl<const K: usize, const D: usize> DriftSegmentStore<K, D> {
    /// Construct with explicit kernel + policy parameters.
    ///
    /// # Panics
    ///
    /// Debug-asserts `K >= 2` (merge needs an adjacent pair) and
    /// `0 < alpha_slow < alpha_fast <= 1` (kernel contract).
    pub fn with_params(
        tau: f32,
        beta: f32,
        min_segment: u32,
        alpha_fast: f32,
        alpha_slow: f32,
    ) -> Self {
        debug_assert!(K >= 2, "DriftSegmentStore requires K >= 2 for merge");
        debug_assert!(D >= 1, "DriftSegmentStore requires D >= 1");
        Self {
            slots: [DriftSlot::default(); K],
            n_active: 0,
            kernel: TemporalDerivativeKernel::new(alpha_fast, alpha_slow),
            tau,
            beta,
            min_segment,
            armed: true,
            pos: 0,
            boundaries_fired: 0,
            merges_done: 0,
        }
    }

    /// Construct with calibrated defaults (Issue 652 bench configuration):
    /// α_f=0.3 / α_s=0.03 (kernel paper default), min_segment=4.
    ///
    /// τ is calibrated for the **key-relative** drift score (the paper's
    /// τ=0.6 is for its state-relative Frobenius form — different scale).
    pub fn new(tau: f32, beta: f32) -> Self {
        Self::with_params(tau, beta, 4, 0.3, 0.03)
    }

    /// Relative drift score: `‖fast − slow‖ / (‖fast‖ + ε)`.
    ///
    /// The key-space analog of the paper's `I_t` (relative state drift).
    /// Normalizing by ‖fast‖ (not ‖slow‖) avoids the startup explosion
    /// while slow is still near zero, and keeps the score scale-free.
    #[inline]
    pub fn relative_drift(&self) -> f32 {
        let diff = self.kernel.surprise_norm();
        let fast_sq = katgpt_core::simd::simd_dot_f32(&self.kernel.fast, &self.kernel.fast, D);
        let fast_norm = fast_sq.max(0.0).sqrt();
        diff / (fast_norm + 1e-6)
    }

    /// Observe one (key, value) token: update the drift kernel, maybe open a
    /// slot, accumulate.
    ///
    /// Boundary fires when the rising edge of the relative drift crosses τ
    /// and the current slot has at least `min_segment` tokens. The boundary
    /// token itself seeds the new slot (paper semantics: high-drift tokens
    /// open the next state).
    pub fn observe(&mut self, key: &[f32; D], value: &[f32; D]) {
        self.kernel.observe(key);
        let score = self.relative_drift();

        let cur_len = if self.n_active == 0 {
            u32::MAX // empty store: first token always opens slot 0
        } else {
            self.slots[self.n_active - 1].n_tokens
        };
        let boundary =
            cur_len == u32::MAX || (self.armed && score >= self.tau && cur_len >= self.min_segment);

        if boundary {
            if self.n_active == K {
                self.merge_min_density_pair();
            }
            let s = &mut self.slots[self.n_active];
            s.key_sum = *key;
            s.val_sum = *value;
            s.n_tokens = 1;
            s.info_sum = score;
            s.pos_start = self.pos;
            s.pos_end = self.pos;
            self.n_active += 1;
            self.boundaries_fired += 1;
            self.armed = false;
        } else {
            let s = &mut self.slots[self.n_active - 1];
            for d in 0..D {
                s.key_sum[d] += key[d];
                s.val_sum[d] += value[d];
            }
            s.n_tokens += 1;
            s.info_sum += score;
            s.pos_end = self.pos;
        }

        if score < self.tau {
            self.armed = true;
        }
        self.pos += 1;
    }

    /// Query-gated readout into a caller-owned buffer (zero-alloc).
    ///
    /// Delegates to [`sigmoid_gated_readout`] over the active slots.
    pub fn readout_into(&self, query: &[f32; D], out: &mut [f32; D]) {
        sigmoid_gated_readout(&self.slots[..self.n_active], query, self.beta, out);
    }

    /// Merge the adjacent pair with the lowest pair density, in place.
    ///
    /// Pair density `(info_i + info_j)/(n_i + n_j)` — the paper's capacity
    /// policy. Additive states merge by summation; the span fuses; the tail
    /// shifts left. O(K + D) — no allocation.
    fn merge_min_density_pair(&mut self) {
        debug_assert!(
            self.n_active >= 2,
            "merge requires at least two active slots"
        );
        let mut best = 0usize;
        let mut best_density = f32::INFINITY;
        for i in 0..self.n_active - 1 {
            let a = &self.slots[i];
            let b = &self.slots[i + 1];
            let density = (a.info_sum + b.info_sum) / (a.n_tokens + b.n_tokens) as f32;
            if density < best_density {
                best_density = density;
                best = i;
            }
        }
        let b = self.slots[best + 1];
        let a = &mut self.slots[best];
        for d in 0..D {
            a.key_sum[d] += b.key_sum[d];
            a.val_sum[d] += b.val_sum[d];
        }
        a.n_tokens += b.n_tokens;
        a.info_sum += b.info_sum;
        a.pos_end = b.pos_end;
        // Compact the tail (slots are Copy — plain shift, no allocation).
        for i in (best + 1)..self.n_active - 1 {
            self.slots[i] = self.slots[i + 1];
        }
        self.n_active -= 1;
        self.merges_done += 1;
    }

    /// Number of active slots (always `<= K`).
    #[inline]
    pub fn n_slots(&self) -> usize {
        self.n_active
    }

    /// Borrow an active slot by index (chronological order).
    #[inline]
    pub fn slot(&self, i: usize) -> &DriftSlot<D> {
        &self.slots[i]
    }

    /// Total tokens observed.
    #[inline]
    pub fn tokens_seen(&self) -> u32 {
        self.pos
    }

    /// Boundaries opened (diagnostic).
    #[inline]
    pub fn boundaries_fired(&self) -> u64 {
        self.boundaries_fired
    }

    /// Capacity merges performed (diagnostic).
    #[inline]
    pub fn merges_done(&self) -> u64 {
        self.merges_done
    }

    /// Configured τ.
    #[inline]
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Configured readout β.
    #[inline]
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Reset to the empty state (keeps policy params).
    pub fn reset(&mut self) {
        self.n_active = 0;
        self.kernel.reset();
        self.armed = true;
        self.pos = 0;
        self.boundaries_fired = 0;
        self.merges_done = 0;
        // Slot contents are dead once n_active == 0; re-zero for determinism
        // of debug dumps only.
        self.slots = [DriftSlot::default(); K];
    }

    /// Discard all accumulated segments, keeping the converged drift kernel
    /// (Issue 680 consumer support — the warmup-discard primitive).
    ///
    /// The kernel's slow EMA converges over ~`2/α_slow` tokens during which
    /// the relative score is elevated by construction (both EMAs near zero —
    /// the same phenomenon τ calibrators skip a warmup prefix for).
    /// Those elevated scores land in slot 0's `info_sum`, making the startup
    /// transient look like a high-density needle. A consumer that wants
    /// density to mean "moment that mattered" (recognition, salience feeds)
    /// discards the warmup prefix with this method once the kernel has
    /// converged: segments + boundary/merge counters reset, the kernel + the
    /// monotonic token position continue.
    ///
    /// After this call the store behaves as if freshly started EXCEPT the
    /// kernel is already converged — the next `observe` opens a clean slot
    /// whose density reflects only post-warmup drift.
    pub fn reset_slots(&mut self) {
        self.n_active = 0;
        self.boundaries_fired = 0;
        self.merges_done = 0;
        self.slots = [DriftSlot::default(); K];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 16;

    /// Deterministic pseudo-normal helper (sum of 3 uniforms, std ≈ 1).
    pub(crate) fn randn(rng: &mut fastrand::Rng) -> f32 {
        (rng.f32() + rng.f32() + rng.f32() - 1.5) * 2.0
    }

    /// Random unit vector in D dims.
    pub(crate) fn unit_dir(rng: &mut fastrand::Rng) -> [f32; D] {
        let mut v = [0.0f32; D];
        let mut sq = 0.0f32;
        for d in 0..D {
            v[d] = randn(rng);
            sq += v[d] * v[d];
        }
        let n = sq.sqrt().max(1e-9);
        for d in 0..D {
            v[d] /= n;
        }
        v
    }

    /// Random unit vector orthogonal to `basis` (Gram–Schmidt step).
    ///
    /// Transitions between random unit vectors are only strong if the two
    /// directions are actually far apart — with D=16 a seed can draw two
    /// nearly-parallel directions (cos ≈ 0.5+) and the drift jump stays
    /// below τ (empirically: the first version of the merge test silently
    /// missed its A→B boundary on seed 5). Orthogonalizing makes every
    /// planted change point a full-strength (√2) jump, deterministically.
    pub(crate) fn orthogonal_unit_dir(basis: &[[f32; D]], rng: &mut fastrand::Rng) -> [f32; D] {
        let mut v = unit_dir(rng);
        for b in basis.iter() {
            let mut dot = 0.0f32;
            for d in 0..D {
                dot += v[d] * b[d];
            }
            for d in 0..D {
                v[d] -= dot * b[d];
            }
        }
        let mut sq = 0.0f32;
        for d in 0..D {
            sq += v[d] * v[d];
        }
        let n = sq.sqrt().max(1e-9);
        for d in 0..D {
            v[d] /= n;
        }
        v
    }

    /// A noisy token drawn around `dir` (unit-norm mean, per-dim noise σ).
    pub(crate) fn noisy(dir: &[f32; D], sigma: f32, rng: &mut fastrand::Rng) -> [f32; D] {
        let mut k = *dir;
        for d in 0..D {
            k[d] += randn(rng) * sigma;
        }
        k
    }

    // ── T1: boundary fires at planted change points ───────────────────────

    #[test]
    fn t1_boundary_fires_at_planted_change_points() {
        // Two orthogonal regimes, 200 tokens each. The drift score must fire
        // a boundary near the change point (within the slow-EMA lag) and
        // ONLY there — no within-regime spam.
        let mut rng = fastrand::Rng::with_seed(42);
        let a = unit_dir(&mut rng);
        let b = orthogonal_unit_dir(&[a], &mut rng);
        let mut store = DriftSegmentStore::<8, D>::new(0.35, 4.0);

        for t in 0..400 {
            let dir = if t < 200 { &a } else { &b };
            let k = noisy(dir, 0.08, &mut rng);
            let v = *dir; // value = regime dir (any payload works)
            store.observe(&k, &v);
        }

        assert!(
            store.n_slots() >= 2,
            "expected >= 2 slots, got {}",
            store.n_slots()
        );
        assert!(
            store.boundaries_fired() <= 4,
            "boundary spam: {} fires for 1 change point",
            store.boundaries_fired()
        );

        // Find the slot whose span contains the change point t=200 and
        // check a boundary landed near it (within the slow-EMA lag ~ 40).
        let mut nearest = i32::MAX;
        for i in 0..store.n_slots() {
            nearest = nearest.min((store.slot(i).pos_start as i32 - 200).abs());
        }
        assert!(
            nearest <= 40,
            "no boundary within 40 tokens of the change point (nearest {nearest})"
        );
    }

    #[test]
    fn t1_stationary_stream_fires_no_boundaries() {
        // One regime, 600 tokens: the score stays at the noise floor and no
        // boundary may fire (the degenerate case ≈ single-state arm).
        let mut rng = fastrand::Rng::with_seed(7);
        let a = unit_dir(&mut rng);
        let mut store = DriftSegmentStore::<8, D>::new(0.35, 4.0);
        for _ in 0..600 {
            let k = noisy(&a, 0.08, &mut rng);
            store.observe(&k, &a);
        }
        assert_eq!(
            store.boundaries_fired(),
            1,
            "only the initial slot may open"
        );
        assert_eq!(store.n_slots(), 1);
    }

    // ── T1: capacity invariant + chronological order + mass conservation ──

    #[test]
    fn t1_capacity_never_exceeds_k_and_mass_conserved() {
        // 40 orthogonal regime switches with K=8: capacity pressure is real.
        let mut rng = fastrand::Rng::with_seed(99);
        let mut store = DriftSegmentStore::<8, D>::new(0.35, 4.0);
        let total = 40 * 60; // 60 tokens per regime
        for r in 0..40 {
            let dir = unit_dir(&mut rng);
            for _ in 0..60 {
                let k = noisy(&dir, 0.08, &mut rng);
                store.observe(&k, &dir);
            }
            assert!(
                store.n_slots() <= 8,
                "capacity exceeded at regime {}: {}",
                r,
                store.n_slots()
            );
        }
        assert!(store.merges_done() > 0, "expected merges under pressure");

        // Mass conservation: Σ n_tokens == tokens observed.
        let mut sum_n = 0u32;
        for i in 0..store.n_slots() {
            sum_n += store.slot(i).n_tokens;
        }
        assert_eq!(sum_n, total, "merge lost/duplicated tokens");
        assert_eq!(store.tokens_seen(), total);
    }

    #[test]
    fn t1_chronological_order_preserved_across_merges() {
        let mut rng = fastrand::Rng::with_seed(1234);
        let mut store = DriftSegmentStore::<6, D>::new(0.35, 4.0);
        for _ in 0..30 {
            let dir = unit_dir(&mut rng);
            for _ in 0..50 {
                let k = noisy(&dir, 0.08, &mut rng);
                store.observe(&k, &dir);
            }
        }
        for i in 0..store.n_slots() {
            let s = store.slot(i);
            assert!(s.pos_start <= s.pos_end, "slot {i} inverted span");
            if i + 1 < store.n_slots() {
                assert!(
                    s.pos_end < store.slot(i + 1).pos_start,
                    "slots {i}/{} overlap: [{},{}] vs [{},{}]",
                    i + 1,
                    s.pos_start,
                    s.pos_end,
                    store.slot(i + 1).pos_start,
                    store.slot(i + 1).pos_end
                );
            }
        }
    }

    #[test]
    fn t1_merge_picks_lowest_density_adjacent_pair() {
        // Controlled 3-slot case via the public observe() surface. Two
        // lessons baked into this construction (both empirically confirmed
        // by earlier versions of this test):
        //
        // 1. Random unit directions can be nearly parallel on unlucky seeds
        //    → the planted A→B transition stays below τ and silently never
        //    fires. Gram–Schmidt-orthogonalized directions make every
        //    change point a full-strength √2 jump.
        // 2. Slot density is dominated by CHANGE-POINT TRANSIENTS amortized
        //    over slot length — so regimes must be LONG for the noise-floor
        //    contrast to surface (short regimes make every slot
        //    transient-dominated).
        // 3. A noisy regime's score floor must stay below τ (σ=0.10 floor:
        //    mean 0.14, max 0.23 < τ=0.35 — measured) or the rising-edge
        //    detector never re-arms and the C→E boundary can't fire.
        let mut store = DriftSegmentStore::<3, D>::new(0.35, 4.0);
        let mut rng = fastrand::Rng::with_seed(5);
        // Regime A: clean (low info) — 300 tokens.
        let a = unit_dir(&mut rng);
        for _ in 0..300 {
            store.observe(&noisy(&a, 0.02, &mut rng), &a);
        }
        // Regime B: clean — 300 tokens, orthogonal to A.
        let b = orthogonal_unit_dir(&[a], &mut rng);
        for _ in 0..300 {
            store.observe(&noisy(&b, 0.02, &mut rng), &b);
        }
        // Regime C: noisy but sub-τ floor — 300 tokens. Capacity is now 3.
        let c = orthogonal_unit_dir(&[a, b], &mut rng);
        for _ in 0..300 {
            store.observe(&noisy(&c, 0.10, &mut rng), &c);
        }
        assert_eq!(
            store.n_slots(),
            3,
            "all three regimes must be separate slots"
        );
        let d0 = store.slot(0).density();
        let d1 = store.slot(1).density();
        let d2 = store.slot(2).density();
        // The noisy regime must have the highest density once transients
        // amortize equally across slots.
        assert!(
            d2 > d0 && d2 > d1,
            "densities d0={d0} d1={d1} d2={d2} — noisy regime must dominate"
        );

        // A 4th regime forces a merge: the (A,B) pair must be chosen (its
        // pair density is the minimum — C is excluded because (A,B) < (B,C)
        // when C is noisy).
        let e = orthogonal_unit_dir(&[a, b, c], &mut rng);
        let merges_before = store.merges_done();
        for _ in 0..320 {
            store.observe(&noisy(&e, 0.02, &mut rng), &e);
        }
        assert_eq!(store.merges_done(), merges_before + 1);
        assert_eq!(store.n_slots(), 3);
        // Merged slot 0 spans A+B: 600 tokens.
        assert_eq!(store.slot(0).n_tokens, 600);
        // Slot 1 is still C (300 tokens) — the merge did NOT touch C.
        assert_eq!(store.slot(1).n_tokens, 300);
    }

    #[test]
    fn t1_tau_never_fires_degenerates_to_single_state() {
        // τ = ∞: no boundary ever fires → exactly the arm-(a) single-state
        // accumulator. The slot's sums must equal the raw stream sums.
        let mut rng = fastrand::Rng::with_seed(11);
        let mut store = DriftSegmentStore::<4, D>::new(f32::INFINITY, 4.0);
        let mut key_tot = [0.0f32; D];
        let mut val_tot = [0.0f32; D];
        for _ in 0..500 {
            let k = noisy(&unit_dir(&mut rng), 0.1, &mut rng);
            let v = [0.5f32; D];
            for d in 0..D {
                key_tot[d] += k[d];
                val_tot[d] += v[d];
            }
            store.observe(&k, &v);
        }
        assert_eq!(store.n_slots(), 1);
        let s = store.slot(0);
        assert_eq!(s.n_tokens, 500);
        for d in 0..D {
            assert!((s.key_sum[d] - key_tot[d]).abs() < 1e-3);
            assert!((s.val_sum[d] - val_tot[d]).abs() < 1e-3);
        }
    }

    #[test]
    fn t1_readout_prefers_aligned_slot() {
        // Two regimes; query the second regime's direction → the readout
        // must point at regime B's value payload, not A's.
        let mut rng = fastrand::Rng::with_seed(21);
        let a = unit_dir(&mut rng);
        let b = orthogonal_unit_dir(&[a], &mut rng);
        let mut store = DriftSegmentStore::<8, D>::new(0.35, 4.0);
        for _ in 0..150 {
            store.observe(&noisy(&a, 0.05, &mut rng), &a);
        }
        for _ in 0..150 {
            store.observe(&noisy(&b, 0.05, &mut rng), &b);
        }
        let mut out = [0.0f32; D];
        store.readout_into(&b, &mut out);
        let cos_a: f32 = out.iter().zip(a.iter()).map(|(x, y)| x * y).sum();
        let cos_b: f32 = out.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            cos_b > cos_a + 0.1,
            "readout must favor the aligned regime: cos_b={cos_b} cos_a={cos_a}"
        );
    }

    #[test]
    fn reset_slots_discards_segments_keeps_converged_kernel() {
        // Warmup discard (Issue 680 consumer support): after ~2/α_slow
        // tokens of a stationary regime, reset_slots must (a) clear slots +
        // boundary counters, (b) keep the converged kernel so the next
        // stationary token's relative score is LOW (no second transient),
        // (c) keep the monotonic token position.
        let mut rng = fastrand::Rng::with_seed(31);
        let a = unit_dir(&mut rng);
        let mut store = DriftSegmentStore::<8, D>::new(0.35, 4.0);
        for _ in 0..200 {
            store.observe(&noisy(&a, 0.02, &mut rng), &a);
        }
        let pos_before = store.tokens_seen();
        let score_converged = store.relative_drift();
        store.reset_slots();
        assert_eq!(store.n_slots(), 0);
        assert_eq!(store.boundaries_fired(), 0);
        assert_eq!(store.merges_done(), 0);
        // Kernel kept: next token's score stays at the converged (low) level,
        // NOT the ~0.9 startup level a full reset() would re-produce.
        store.observe(&noisy(&a, 0.02, &mut rng), &a);
        assert_eq!(store.n_slots(), 1);
        let post = store.relative_drift();
        assert!(
            post < 0.5,
            "kernel must stay converged after reset_slots: post={post:.3} (converged={score_converged:.3})"
        );
        assert_eq!(store.tokens_seen(), pos_before + 1);
    }
}

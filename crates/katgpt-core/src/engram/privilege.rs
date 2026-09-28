//! Counterfactual privilege gating for engram fusion (Issue 656).
//!
//! Source: riir-train Research 419 §5.2 (LOPD, arXiv:2608.13040), modelless
//! corollary. Training-track sibling: riir-train Plan 340.
//!
//! # The blind spot this closes
//!
//! The shipped engram gate ([`sigmoid_fuse_into`]) is **similarity-only**:
//!
//! ```text
//! gate   = σ(dot(q_norm, k_norm) / τ)
//! out[j] = gate * v[j]
//! ```
//!
//! It answers *"is this memory relevant to the query?"* — never *"does fusing
//! this memory improve the consumer's prediction?"* The shipped test
//! `forward::tests::fuse_zero_query_does_not_corrupt_hidden_state` documents
//! the consequence: with `q = 0`, `dot = 0` → `gate = σ(0) = 0.5`, so **every
//! populated slot fuses at half strength regardless of utility**. A drifted or
//! anti-useful entry cannot be vetoed — only scaled by similarity.
//!
//! # The δ quantity (modelless, no gradient descent)
//!
//! ```text
//! δ_k = score(state + contribution_k) − score(state)    // counterfactual advantage
//! Δ_k ← (1−α)·Δ_k + α·(A · δ_k)                         // outcome-weighted EMA, per slot
//! p_k  = σ((Δ_k − m) / s)                               // privilege factor
//! out  = (base_gate · p_k) · v                          // multiplicative extension
//! ```
//!
//! Two evaluations and a comparison. No backprop, no gradients, no weight
//! mutation — the ledger is runtime latent state, exactly like a routing table
//! (see the modelless-first mandate in `CLAUDE.md`).
//!
//! LOPD's F2 finding is the empirical justification: **the same context helps
//! in one regime and hurts in another**, so utility must be measured at use
//! time and conditioned on the current query. Neither the similarity gate
//! (query-only) nor riir-clippy's `EvidenceTier` (history-only, discrete
//! 3-tier) is query-conditional.
//!
//! # Why the ledger is a side-car, not a table field
//!
//! Issue 656 scoped `Δ_slot` as a field *inside* the engram table ("table
//! layout change, versioned (freeze/thaw bump)"). This module deliberately
//! ships it as a **separate [`PrivilegeLedger`] indexed by slot** instead:
//!
//! - [`InMemoryEngramTable`](super::InMemoryEngramTable) is a **frozen,
//!   BLAKE3-committed snapshot**. Adding mutable per-slot state to it would
//!   either poison the commitment (two tables with identical patterns but
//!   different usage histories would no longer share a root) or require
//!   excluding the field from the commitment — at which point it was never
//!   really part of the table.
//! - The ledger composes with **any** [`EngramTable`] impl (in-memory,
//!   `ZipfianCacheHierarchy`, `StagingEngramTable`) with no trait change and
//!   no freeze/thaw version bump.
//! - It keeps mutable runtime state cleanly separated from immutable committed
//!   state, which is the same split the repo already enforces at the sync
//!   boundary.
//!
//! # Slot-mapping contract
//!
//! [`fuse_into_hidden_state_privileged`] derives the ledger index as
//! `hash_keys[k].0 as usize % table.num_slots()`. This is not an assumption —
//! it is the documented [`EngramTable::num_slots`] contract ("the modulus used
//! for `hash.0 as usize % num_slots`"). A table impl that violates it would
//! mis-index the ledger.
//!
//! # Cost model (the honest constraint)
//!
//! δ requires the consumer scored twice per slot. The split here keeps that
//! cost **off the hot path**:
//!
//! - **Hot path** ([`fuse_into_hidden_state_privileged`]): one extra array
//!   read, one [`fast_sigmoid`], and one scalar multiply per head. The
//!   privilege factor folds into the *scalar* gate before the `D`-element
//!   store, so there is no second pass over the vector. Zero-allocation.
//! - **Update path** (host-driven, amortized): the host re-derives the per-head
//!   contribution with [`sigmoid_fuse_scaled_into`], scores it against the
//!   base state, and calls [`PrivilegeLedger::observe`]. Run this on a sampled
//!   fraction of retrieval events — the EMA decays between updates.
//!
//! [`PrivilegeConfig::gate_floor`] is the "gate-on-gate": heads whose *base*
//! similarity gate falls below the floor are not traced, so they never consume
//! counterfactual-scoring budget.
//!
//! # CRITICAL — sigmoid, not softmax
//!
//! The privilege factor is `σ((Δ − m)/s)` — an independent per-slot scalar.
//! There is no `softmax` symbol in this file. Softmax would make slots compete
//! for a fixed budget; privilege is an **absolute** contract ("this slot must
//! keep earning ≥ m"), not a relative ranking.
//!
//! [`sigmoid_fuse_into`]: super::sigmoid_fuse_into
//! [`fast_sigmoid`]: crate::simd::fast_sigmoid

use super::{EngramConfig, EngramHash, EngramTable, K_MAX, sigmoid_fuse_scaled_into};
use crate::simd::{fast_sigmoid, simd_add_inplace, simd_sum_abs_f32};

/// Tuning for the privilege gate.
///
/// # Units
///
/// `margin` and `scale` are in **the host's score units** — the same units as
/// the `delta` reported to [`PrivilegeLedger::observe`]. There is no implicit
/// normalization: a scorer whose δ typically lands around `0.01` needs a
/// `scale` near `0.01`, not the default `1.0`. Use
/// [`PrivilegeConfig::for_delta_scale`] to derive a coherent set from one
/// measured magnitude rather than hand-tuning three coupled numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrivilegeConfig {
    /// EMA rate `α ∈ (0, 1]` for the per-slot advantage. Higher = faster
    /// adaptation, noisier. `1.0` makes the ledger memoryless (last observation
    /// wins).
    pub alpha: f32,
    /// Margin `m` — the advantage a slot must sustain to fuse at full strength.
    /// Cold start (`Δ = 0`) yields `p = σ(−m/s) < 0.5`: new slots fuse weakly
    /// and *earn* strength. `m = 0` disables the cold-start penalty.
    pub margin: f32,
    /// Sigmoid scale `s > 0` on `(Δ − m)/s`. Smaller = sharper veto (closer to
    /// a hard threshold at `Δ = m`); larger = softer.
    pub scale: f32,
    /// Gate-on-gate floor: heads whose **base** similarity gate is below this
    /// are not recorded in the [`PrivilegeTrace`], so they never consume
    /// counterfactual-scoring budget. `0.0` traces every populated head.
    pub gate_floor: f32,
    /// Dual step size `η` for the table-health accumulator
    /// `β ← [β + η(m − Δ_table)]₊`. See [`PrivilegeLedger::tick_dual`].
    pub dual_eta: f32,
    /// Veto short-circuit: heads whose privilege factor is `≤ veto_epsilon`
    /// skip the fusion kernel entirely.
    ///
    /// `0.0` (the default) means **never skip** — the output is then exactly
    /// `base_gate · p · v` for every head, which is what the G1 equivalence
    /// tests pin. Raising it trades a bounded output perturbation (at most
    /// `veto_epsilon · base_gate · ‖v‖`) for skipping the whole RMSNorm + dot +
    /// store on vetoed heads. Worth it only on tables with many vetoed slots.
    pub veto_epsilon: f32,
}

impl PrivilegeConfig {
    /// Derive a coherent config from the typical magnitude of `|A · δ|`.
    ///
    /// Sets `scale = typical_abs_delta` and `margin = 0.25 · typical_abs_delta`,
    /// so a slot with no history sits at `p = σ(−0.25) ≈ 0.438` (a mild
    /// cold-start penalty) and a slot sustaining one typical positive δ sits at
    /// `p = σ(0.75) ≈ 0.679`.
    ///
    /// `typical_abs_delta` must be finite and `> 0`; non-positive or non-finite
    /// input falls back to `1.0` so the config is never degenerate.
    ///
    /// **This tuning suppresses, it does not veto.** A slot converging to
    /// `Δ = −typical_abs_delta` lands at `p = σ(−1.25) ≈ 0.223` — roughly a 3×
    /// suppression against a useful slot's `0.679`. For a hard veto (`p → 0`),
    /// shrink `scale` below the δ magnitude: `s = 0.1 · typical` puts a
    /// consistently-harmful slot at `σ(−12.5) ≈ 4e-6`. Sharper is not free —
    /// it also makes the gate react violently to a single noisy observation.
    #[inline]
    pub fn for_delta_scale(typical_abs_delta: f32) -> Self {
        let s = if typical_abs_delta.is_finite() && typical_abs_delta > 0.0 {
            typical_abs_delta
        } else {
            1.0
        };
        Self {
            alpha: 0.15,
            margin: 0.25 * s,
            scale: s,
            gate_floor: 0.0,
            dual_eta: 0.01,
            veto_epsilon: 0.0,
        }
    }
}

impl Default for PrivilegeConfig {
    #[inline]
    fn default() -> Self {
        Self::for_delta_scale(1.0)
    }
}

/// How an aggregate δ is split across the heads that produced it.
///
/// Only relevant to [`PrivilegeLedger::observe_trace`]. When the host can
/// afford per-slot marginal scoring it should call [`PrivilegeLedger::observe`]
/// directly with an exact per-slot δ and skip this entirely.
///
/// # Known limitation — sign-opposed slots
///
/// Both variants distribute a **single scalar** δ using **unsigned** weights,
/// so every traced slot receives credit of the same sign. When a fuse mixes
/// slots whose contributions point in *opposite* directions (the LOPD F2
/// regime this whole module exists for), the aggregate δ is near zero and no
/// weighting can recover the per-slot signs. Aggregate attribution is a
/// cost-saving approximation for same-sign fuses, not a substitute for
/// per-slot scoring. Issue 656 T1 measures exactly this failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CreditAssignment {
    /// Split δ equally across traced heads. Unbiased under symmetry, and the
    /// cheapest option.
    Uniform,
    /// Split δ proportionally to each head's L1 contribution magnitude
    /// `p_k · base_gate_k · ‖v_k‖₁`. Falls back to [`Uniform`] when the weights
    /// sum to zero.
    ///
    /// [`Uniform`]: CreditAssignment::Uniform
    GateWeighted,
}

/// Which slots a privileged fuse touched, and how strongly.
///
/// Stack-only and `Copy` — fixed `K_MAX`-sized arrays, no allocation. The host
/// carries one of these from the fuse to the (later, amortized) outcome report.
#[derive(Debug, Clone, Copy)]
pub struct PrivilegeTrace {
    slots: [u32; K_MAX],
    weights: [f32; K_MAX],
    len: usize,
}

impl Default for PrivilegeTrace {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl PrivilegeTrace {
    /// An empty trace.
    #[inline]
    pub const fn new() -> Self {
        Self {
            slots: [0; K_MAX],
            weights: [0.0; K_MAX],
            len: 0,
        }
    }

    /// Reset to empty, retaining the (stack) storage.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Record one head. Silently drops entries past `K_MAX` (unreachable when
    /// driven by [`fuse_into_hidden_state_privileged`], which fuses at most
    /// `K_MAX` heads).
    #[inline]
    pub fn push(&mut self, slot: u32, weight: f32) {
        if self.len < K_MAX {
            self.slots[self.len] = slot;
            self.weights[self.len] = weight;
            self.len += 1;
        }
    }

    /// Number of recorded heads.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no head was recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Recorded slot indices, in fuse order.
    #[inline]
    pub fn slots(&self) -> &[u32] {
        &self.slots[..self.len]
    }

    /// Recorded contribution weights (`p_k · base_gate_k · ‖v_k‖₁`), in fuse
    /// order.
    #[inline]
    pub fn weights(&self) -> &[f32] {
        &self.weights[..self.len]
    }
}

/// Per-slot counterfactual-advantage state for one engram table.
///
/// Sized to the table's slot count and indexed by the same
/// `hash mod num_slots` mapping the table uses. Runtime latent state — never
/// committed, never synced (only the scalar outputs cross the sync boundary,
/// per the bridge pattern).
#[derive(Debug, Clone)]
pub struct PrivilegeLedger {
    /// Per-slot advantage EMA `Δ`. Length = `n_slots`.
    delta: Box<[f32]>,
    /// Per-slot privilege factor `σ((Δ − m)/s)`, kept in lockstep with `delta`.
    ///
    /// `Δ` changes only on [`observe`](PrivilegeLedger::observe) — an amortized,
    /// sampled path — while `privilege()` is read once per head on **every**
    /// fuse. Recomputing the sigmoid in the hot path burned ~30% of the fuse
    /// (measured, Issue 656 G2: 1.294× → 1.02× on this change alone). The
    /// sigmoid now runs on the update path, where it belongs.
    ///
    /// Every write to `delta` MUST write the matching `factor` entry. That
    /// invariant is local to this file: `observe`, `set_config`, `reset`, and
    /// `new` are the only writers.
    factor: Box<[f32]>,
    /// Table-level advantage EMA (all observations, any slot). Drives
    /// [`tick_dual`](PrivilegeLedger::tick_dual).
    table_delta: f32,
    /// Dual accumulator `β ≥ 0` — ramping pressure when the table's aggregate
    /// advantage sits below the margin.
    beta: f32,
    /// Total number of `observe` calls (diagnostics / warm-up checks).
    observations: u64,
    config: PrivilegeConfig,
}

impl PrivilegeLedger {
    /// Cold-start ledger for an `n_slots` table: every `Δ = 0`, `β = 0`.
    #[inline]
    pub fn new(n_slots: usize, config: PrivilegeConfig) -> Self {
        let cold = privilege_of(0.0, &config);
        Self {
            delta: vec![0.0f32; n_slots].into_boxed_slice(),
            factor: vec![cold; n_slots].into_boxed_slice(),
            table_delta: 0.0,
            beta: 0.0,
            observations: 0,
            config,
        }
    }

    /// Ledger sized to `table.num_slots()`.
    #[inline]
    pub fn for_table(table: &dyn EngramTable, config: PrivilegeConfig) -> Self {
        Self::new(table.num_slots(), config)
    }

    /// Number of slots tracked. Must equal the table's `num_slots()`.
    #[inline]
    pub fn n_slots(&self) -> usize {
        self.delta.len()
    }

    /// Current tuning.
    #[inline]
    pub fn config(&self) -> &PrivilegeConfig {
        &self.config
    }

    /// Retune without discarding accumulated advantage. Rebuilds the cached
    /// privilege factors, so it is `O(n_slots)` — call it off the hot path.
    #[inline]
    pub fn set_config(&mut self, config: PrivilegeConfig) {
        self.config = config;
        for (f, &d) in self.factor.iter_mut().zip(self.delta.iter()) {
            *f = privilege_of(d, &config);
        }
    }

    /// Accumulated advantage `Δ` for a slot. Out-of-range returns `0.0`
    /// (cold start).
    #[inline]
    pub fn advantage(&self, slot: usize) -> f32 {
        self.delta.get(slot).copied().unwrap_or(0.0)
    }

    /// Table-level advantage EMA across all observations.
    #[inline]
    pub fn table_advantage(&self) -> f32 {
        self.table_delta
    }

    /// Dual accumulator `β`. Rises while the table's aggregate advantage sits
    /// below `margin` — a host-facing "this table is decaying" health signal.
    #[inline]
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Number of [`observe`](PrivilegeLedger::observe) calls so far.
    #[inline]
    pub fn observations(&self) -> u64 {
        self.observations
    }

    /// Privilege factor `p = σ((Δ_slot − m) / s) ∈ (0, 1)`.
    ///
    /// A cached array read — the sigmoid was already paid on the update path.
    /// This is the only ledger call on the fusion hot path.
    ///
    /// **Fails open**: an out-of-range slot (ledger/table size mismatch)
    /// returns `1.0`, degrading to the plain similarity gate rather than
    /// silently zeroing the table's memory. A mismatch is a `debug_assert`
    /// failure in debug builds.
    #[inline]
    pub fn privilege(&self, slot: usize) -> f32 {
        if let Some(&p) = self.factor.get(slot) {
            p
        } else {
            debug_assert!(
                false,
                "PrivilegeLedger::privilege: slot {slot} out of range (n_slots = {}) — \
                     ledger and table sizes must match",
                self.factor.len()
            );
            1.0
        }
    }

    /// Record one **exact per-slot** counterfactual outcome.
    ///
    /// `Δ_slot ← (1−α)·Δ_slot + α·(advantage · delta)`, where `delta` is
    /// `score(state + contribution_slot) − score(state)` and `advantage` is the
    /// host's outcome weight (e.g. `+1` verified good, `−1` verified bad, `0`
    /// unknown). Non-finite inputs are ignored so a diverged scorer cannot
    /// poison the ledger.
    ///
    /// This is the accurate path. It costs one scorer call per slot plus one
    /// for the base state; amortize by sampling retrieval events, not by
    /// approximating the attribution (see [`CreditAssignment`]'s limitation).
    #[inline]
    pub fn observe(&mut self, slot: usize, advantage: f32, delta: f32) {
        let credit = advantage * delta;
        if !credit.is_finite() {
            return;
        }
        let alpha = self.config.alpha;
        let updated = if let Some(d) = self.delta.get_mut(slot) {
            *d = (1.0 - alpha) * *d + alpha * credit;
            *d
        } else {
            debug_assert!(
                false,
                "PrivilegeLedger::observe: slot {slot} out of range (n_slots = {})",
                self.delta.len()
            );
            return;
        };
        // Keep the cached factor in lockstep — this is the one place the
        // sigmoid is paid, and it is off the fusion hot path by construction.
        self.factor[slot] = privilege_of(updated, &self.config);
        self.table_delta = (1.0 - alpha) * self.table_delta + alpha * credit;
        self.observations = self.observations.saturating_add(1);
    }

    /// Record one **aggregate** outcome, split across a trace.
    ///
    /// Cheaper than [`observe`](PrivilegeLedger::observe) (two scorer calls
    /// total instead of `K+1`) but approximate — read
    /// [`CreditAssignment`]'s sign-opposed limitation before relying on it.
    /// Non-finite `advantage · delta` is ignored.
    pub fn observe_trace(
        &mut self,
        trace: &PrivilegeTrace,
        advantage: f32,
        delta: f32,
        credit_assignment: CreditAssignment,
    ) {
        let credit = advantage * delta;
        if trace.is_empty() || !credit.is_finite() {
            return;
        }
        let n = trace.len() as f32;
        let weight_sum: f32 = trace.weights().iter().sum();
        // GateWeighted degrades to Uniform when every weight is zero (or the
        // sum is non-finite) — otherwise the split would be 0/0.
        let use_weighted = match credit_assignment {
            CreditAssignment::GateWeighted => weight_sum.is_finite() && weight_sum > 0.0,
            CreditAssignment::Uniform => false,
        };
        for (&slot, &w) in trace.slots().iter().zip(trace.weights().iter()) {
            let share = if use_weighted {
                w / weight_sum
            } else {
                1.0 / n
            };
            // `observe` re-multiplies by `advantage`, so pass the already-split
            // delta and a unit advantage to avoid squaring the outcome weight.
            self.observe(slot as usize, 1.0, credit * share);
        }
    }

    /// Advance the dual accumulator: `β ← [β + η·(m − Δ_table)]₊`.
    ///
    /// The generic "must keep earning ≥ m" runtime contract. `β` grows while
    /// the table's aggregate advantage stays under the margin and relaxes back
    /// toward zero once it recovers. Host-facing only — `β` does **not** feed
    /// the fusion gate (it is a table-level health metric, and coupling it into
    /// a per-slot gate would punish good slots for their neighbours).
    #[inline]
    pub fn tick_dual(&mut self) {
        let next = self.beta + self.config.dual_eta * (self.config.margin - self.table_delta);
        self.beta = next.max(0.0);
    }

    /// Reset every slot's advantage, the table EMA, `β`, and the observation
    /// count. Use after a table hot-swap — the new table's slots have not
    /// earned the old table's history.
    #[inline]
    pub fn reset(&mut self) {
        self.delta.fill(0.0);
        self.factor.fill(privilege_of(0.0, &self.config));
        self.table_delta = 0.0;
        self.beta = 0.0;
        self.observations = 0;
    }

    /// Raw per-slot advantages (diagnostics, snapshotting).
    #[inline]
    pub fn advantages(&self) -> &[f32] {
        &self.delta
    }
}

/// Privilege-gated end-to-end fuse — [`fuse_into_hidden_state`] plus a
/// per-slot utility veto.
///
/// Identical to [`fuse_into_hidden_state`] except that head `k`'s scalar gate
/// is multiplied by `p_k = ledger.privilege(slot_k)` before it scales `v`. The
/// similarity-gate math is **untouched**: `p_k` folds into the scalar, so the
/// `D`-element store still happens exactly once per head and the RMSNorm + dot
/// path is byte-for-byte the shipped one.
///
/// With an all-cold ledger and `margin = 0`, every `p_k = σ(0) = 0.5`, so the
/// result is exactly half the unprivileged fuse — a uniform positive scale,
/// which preserves relevance ranking exactly. Ranking only changes once slots
/// have *earned* differing advantage, which is the point.
///
/// # Zero-allocation
///
/// Caller provides both scratch buffers. Unlike [`fuse_into_hidden_state`],
/// there is no `scratch_norm` parameter: the fused-RMSNorm path never used it,
/// and this entry point is new so it carries none of that forward-compat
/// baggage.
///
/// # Arguments
///
/// - `hidden_state` — live latent state, length `D`; gated patterns are
///   residual-added in place.
/// - `query` — query vector `q`, length `D`.
/// - `table` — frozen engram table. Looked up once.
/// - `hash_keys` — `K_MAX` slot keys (typically from
///   [`multi_head_hash`](super::multi_head_hash)).
/// - `config` — fusion tau + `k_heads`.
/// - `ledger` — per-slot advantage. `ledger.n_slots()` MUST equal
///   `table.num_slots()` (debug_asserted; mismatch fails open to `p = 1`).
/// - `trace` — **cleared on entry**, then filled with the heads that fused at
///   or above `gate_floor`. Feed it to
///   [`PrivilegeLedger::observe_trace`], or use its `slots()` to drive exact
///   per-slot [`observe`](PrivilegeLedger::observe) calls.
/// - `scratch_lookup` — size `K_MAX * D`.
/// - `scratch_out` — size `D`.
///
/// [`fuse_into_hidden_state`]: super::fuse_into_hidden_state
#[allow(clippy::too_many_arguments)] // zero-alloc hot path: 6 inputs + trace + 2 scratch buffers
pub fn fuse_into_hidden_state_privileged(
    hidden_state: &mut [f32],
    query: &[f32],
    table: &dyn EngramTable,
    hash_keys: &[EngramHash; K_MAX],
    config: &EngramConfig,
    ledger: &PrivilegeLedger,
    trace: &mut PrivilegeTrace,
    scratch_lookup: &mut [f32],
    scratch_out: &mut [f32],
) {
    let d = table.dim();
    debug_assert_eq!(
        hidden_state.len(),
        d,
        "fuse_into_hidden_state_privileged: hidden_state.len() must equal table.dim()"
    );
    debug_assert_eq!(
        query.len(),
        d,
        "fuse_into_hidden_state_privileged: query.len() must equal table.dim()"
    );
    debug_assert!(
        scratch_lookup.len() >= K_MAX * d,
        "fuse_into_hidden_state_privileged: scratch_lookup must be ≥ K_MAX*D"
    );
    debug_assert!(
        scratch_out.len() >= d,
        "fuse_into_hidden_state_privileged: scratch_out must be ≥ D"
    );
    debug_assert!(
        config.k_heads <= K_MAX,
        "fuse_into_hidden_state_privileged: config.k_heads must be ≤ K_MAX"
    );
    debug_assert_eq!(
        ledger.n_slots(),
        table.num_slots(),
        "fuse_into_hidden_state_privileged: ledger.n_slots() must equal table.num_slots()"
    );

    trace.clear();

    let n_slots = table.num_slots();
    if d == 0 || n_slots == 0 {
        return;
    }

    // Step 1: one lookup for all K_MAX heads (same as the unprivileged path).
    let _hits = table.lookup_into(hash_keys, &mut scratch_lookup[..K_MAX * d]);

    let gate_floor = config_gate_floor(ledger);
    let veto_eps = ledger.config.veto_epsilon;
    let k_active = config.k_heads.min(K_MAX);

    for k in 0..k_active {
        let e_k = &scratch_lookup[k * d..(k + 1) * d];

        // Empty-slot skip, byte-identical to the unprivileged path.
        let ek_l1 = simd_sum_abs_f32(e_k);
        if ek_l1 == 0.0 {
            continue;
        }

        // The slot mapping is the documented `EngramTable::num_slots` contract.
        let slot = (hash_keys[k].0 as usize) % n_slots;
        let p = ledger.privilege(slot);

        // Veto short-circuit — skips the whole RMSNorm + dot + store. Disabled
        // by default (`veto_epsilon = 0.0`) so the output stays exactly
        // `base_gate · p · v` for every head.
        if p <= veto_eps {
            continue;
        }

        // CRITICAL: sigmoid, not softmax (see kernel.rs). `p` scales the
        // *scalar* gate — the vector store still happens exactly once.
        let base_gate = sigmoid_fuse_scaled_into(query, e_k, e_k, scratch_out, &config.fusion, p);

        simd_add_inplace(&mut hidden_state[..d], &scratch_out[..d]);

        // Gate-on-gate: only heads whose *base* similarity gate cleared the
        // floor consume counterfactual-scoring budget.
        if base_gate >= gate_floor {
            trace.push(slot as u32, p * base_gate * ek_l1);
        }
    }
}

/// The privilege sigmoid. Single definition so the cached
/// [`PrivilegeLedger::factor`] can never drift from what `Δ` implies.
///
/// CRITICAL: sigmoid, not softmax — an absolute per-slot contract, not a
/// competition for a fixed budget.
#[inline(always)]
fn privilege_of(delta: f32, config: &PrivilegeConfig) -> f32 {
    fast_sigmoid((delta - config.margin) / config.scale)
}

/// Extracted so the hot loop reads one local instead of chasing `ledger.config`
/// through a `Box` deref on every head.
#[inline(always)]
fn config_gate_floor(ledger: &PrivilegeLedger) -> f32 {
    ledger.config.gate_floor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{EngramTableBuilder, fuse_into_hidden_state, sigmoid_fuse_into};

    fn make_table(d: usize, n_slots: usize) -> impl EngramTable {
        let mut b = EngramTableBuilder::new(n_slots, d);
        for i in 0..8u64 {
            let pat: Vec<f32> = (0..d).map(|j| (i as f32) * 0.1 + j as f32 * 0.01).collect();
            b.add_pattern(EngramHash(i), &pat);
        }
        b.build()
    }

    // ── kernel equivalence ──────────────────────────────────────────────────

    #[test]
    fn scaled_fuse_with_unit_scale_is_bit_identical() {
        // The load-bearing "existing gate math untouched" anchor.
        let d = 32;
        let cfg = super::super::SigmoidFusionConfig {
            tau: (d as f32).sqrt(),
            rmsnorm_eps: 1e-6,
            logit_bias: 0.0,
        };
        let q: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.37).sin()).collect();
        let k: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.11).cos()).collect();
        let v: Vec<f32> = (0..d).map(|i| (i as f32) * 0.05 - 0.4).collect();

        let mut out_ref = vec![0.0f32; d];
        sigmoid_fuse_into(&q, &k, &v, &mut out_ref, &cfg);

        let mut out_scaled = vec![0.0f32; d];
        let gate = sigmoid_fuse_scaled_into(&q, &k, &v, &mut out_scaled, &cfg, 1.0);

        assert_eq!(out_scaled, out_ref, "scale = 1.0 must be bit-identical");
        assert!((0.0..=1.0).contains(&gate), "gate in [0,1], got {gate}");
    }

    #[test]
    fn scaled_fuse_halves_output_at_half_scale() {
        let d = 16;
        let cfg = super::super::SigmoidFusionConfig {
            tau: (d as f32).sqrt(),
            rmsnorm_eps: 1e-6,
            logit_bias: 0.0,
        };
        let q: Vec<f32> = (1..=d).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..d).map(|i| (i as f32) * 0.1 + 1.0).collect();

        let mut full = vec![0.0f32; d];
        sigmoid_fuse_scaled_into(&q, &q, &v, &mut full, &cfg, 1.0);
        let mut half = vec![0.0f32; d];
        sigmoid_fuse_scaled_into(&q, &q, &v, &mut half, &cfg, 0.5);

        for j in 0..d {
            assert!(
                (half[j] - 0.5 * full[j]).abs() < 1e-6,
                "j={j}: {} vs {}",
                half[j],
                0.5 * full[j]
            );
        }
    }

    // ── ledger mechanics ────────────────────────────────────────────────────

    #[test]
    fn cold_ledger_privilege_is_below_half() {
        let cfg = PrivilegeConfig::for_delta_scale(1.0);
        let ledger = PrivilegeLedger::new(64, cfg);
        let p = ledger.privilege(0);
        assert!(p < 0.5, "cold start must fuse weakly, got {p}");
        // σ(-0.25) ≈ 0.4378
        assert!((p - 0.4378).abs() < 1e-3, "expected σ(-0.25), got {p}");
    }

    #[test]
    fn zero_margin_cold_privilege_is_exactly_half() {
        let cfg = PrivilegeConfig {
            margin: 0.0,
            ..PrivilegeConfig::for_delta_scale(1.0)
        };
        let ledger = PrivilegeLedger::new(8, cfg);
        assert!((ledger.privilege(3) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn positive_outcomes_raise_privilege_negative_lower_it() {
        let cfg = PrivilegeConfig::for_delta_scale(1.0);
        let mut ledger = PrivilegeLedger::new(4, cfg);
        for _ in 0..64 {
            ledger.observe(0, 1.0, 1.0); // consistently useful
            ledger.observe(1, 1.0, -1.0); // consistently harmful
        }
        let good = ledger.privilege(0);
        let bad = ledger.privilege(1);
        let cold = ledger.privilege(2);
        // At the default `for_delta_scale(1.0)` tuning (m = 0.25, s = 1.0) the
        // converged factors are σ(0.75) ≈ 0.679 and σ(−1.25) ≈ 0.223 — a ~3×
        // suppression, NOT a hard veto. A hard veto needs a smaller `scale`
        // (see `vetoed_slot_contributes_nothing`, which uses s = 0.01).
        assert!((good - 0.679).abs() < 0.01, "expected σ(0.75), got {good}");
        assert!((bad - 0.223).abs() < 0.01, "expected σ(−1.25), got {bad}");
        assert!(
            bad < cold && cold < good,
            "ordering harmful < cold < useful: {bad} / {cold} / {good}"
        );
        assert!(good > 3.0 * bad, "useful must dominate harmful by ≥3×");
    }

    #[test]
    fn negative_advantage_flips_the_sign_of_delta() {
        // A = -1 with δ = +1 means "the fuse raised the score but the outcome
        // was bad" → the slot should lose privilege.
        let mut ledger = PrivilegeLedger::new(2, PrivilegeConfig::for_delta_scale(1.0));
        for _ in 0..32 {
            ledger.observe(0, -1.0, 1.0);
        }
        assert!(ledger.advantage(0) < 0.0, "A=-1 must produce Δ<0");
        // Δ → −(1 − 0.85³²) ≈ −0.995 → σ(−1.245) ≈ 0.224.
        assert!(ledger.privilege(0) < 0.25, "got {}", ledger.privilege(0));
    }

    #[test]
    fn non_finite_credit_is_ignored() {
        let mut ledger = PrivilegeLedger::new(2, PrivilegeConfig::for_delta_scale(1.0));
        ledger.observe(0, 1.0, 0.5);
        let before = ledger.advantage(0);
        ledger.observe(0, 1.0, f32::NAN);
        ledger.observe(0, f32::INFINITY, 1.0);
        assert_eq!(
            ledger.advantage(0),
            before,
            "diverged scorer must not poison the ledger"
        );
        assert_eq!(ledger.observations(), 1);
    }

    #[test]
    fn out_of_range_slot_fails_open() {
        // debug_assert would fire in a debug build, so exercise the release
        // semantics through the public read-only accessor instead.
        let ledger = PrivilegeLedger::new(4, PrivilegeConfig::for_delta_scale(1.0));
        assert_eq!(
            ledger.advantage(999),
            0.0,
            "out-of-range advantage reads as cold"
        );
    }

    #[test]
    fn cached_factor_stays_in_lockstep_with_delta() {
        // The factor cache is the one place a bug could silently gate on stale
        // state — the fuse would keep using an old σ(Δ) while `advantage()`
        // reported the new Δ. Pin every writer: observe / set_config / reset.
        let cfg = PrivilegeConfig::for_delta_scale(0.5);
        let mut ledger = PrivilegeLedger::new(8, cfg);
        let expect = |l: &PrivilegeLedger, c: &PrivilegeConfig| {
            for s in 0..l.n_slots() {
                let want = fast_sigmoid((l.advantage(s) - c.margin) / c.scale);
                assert_eq!(
                    l.privilege(s),
                    want,
                    "slot {s}: cached factor desynced from Δ = {}",
                    l.advantage(s)
                );
            }
        };
        expect(&ledger, &cfg);

        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for i in 0..200usize {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let d = ((state >> 40) as f32 / 8_388_608.0) - 1.0;
            ledger.observe(i % 8, 1.0, d);
        }
        expect(&ledger, &cfg);

        let retuned = PrivilegeConfig {
            margin: 0.3,
            scale: 0.05,
            ..cfg
        };
        ledger.set_config(retuned);
        expect(&ledger, &retuned);

        ledger.reset();
        expect(&ledger, &retuned);
        assert!(ledger.advantages().iter().all(|&d| d == 0.0));
    }

    #[test]
    fn reset_clears_all_state() {
        let mut ledger = PrivilegeLedger::new(4, PrivilegeConfig::for_delta_scale(1.0));
        for _ in 0..10 {
            ledger.observe(1, 1.0, 1.0);
        }
        ledger.tick_dual();
        ledger.reset();
        assert_eq!(ledger.advantage(1), 0.0);
        assert_eq!(ledger.table_advantage(), 0.0);
        assert_eq!(ledger.beta(), 0.0);
        assert_eq!(ledger.observations(), 0);
    }

    // ── dual accumulator ────────────────────────────────────────────────────

    #[test]
    fn beta_ramps_when_table_advantage_decays_and_relaxes_when_it_recovers() {
        let cfg = PrivilegeConfig {
            dual_eta: 0.1,
            ..PrivilegeConfig::for_delta_scale(1.0)
        };
        let mut ledger = PrivilegeLedger::new(4, cfg);
        // Decay phase: every observation is harmful → Δ_table < m.
        for _ in 0..40 {
            ledger.observe(0, 1.0, -1.0);
            ledger.tick_dual();
        }
        let ramped = ledger.beta();
        assert!(
            ramped > 0.0,
            "β must ramp under sustained decay, got {ramped}"
        );

        // Recovery phase: strongly useful observations pull Δ_table above m.
        for _ in 0..200 {
            ledger.observe(0, 1.0, 4.0);
            ledger.tick_dual();
        }
        assert!(
            ledger.beta() < ramped,
            "β must relax on recovery: {} vs {ramped}",
            ledger.beta()
        );
    }

    #[test]
    fn beta_is_clamped_non_negative() {
        let cfg = PrivilegeConfig {
            dual_eta: 1.0,
            ..PrivilegeConfig::for_delta_scale(1.0)
        };
        let mut ledger = PrivilegeLedger::new(2, cfg);
        for _ in 0..50 {
            ledger.observe(0, 1.0, 10.0);
            ledger.tick_dual();
        }
        assert_eq!(ledger.beta(), 0.0, "β must clamp at 0, never go negative");
    }

    // ── trace + credit assignment ───────────────────────────────────────────

    #[test]
    fn trace_push_saturates_at_k_max() {
        let mut t = PrivilegeTrace::new();
        for i in 0..(K_MAX + 5) {
            t.push(i as u32, 1.0);
        }
        assert_eq!(t.len(), K_MAX);
        assert_eq!(t.slots().len(), K_MAX);
    }

    #[test]
    fn uniform_credit_splits_evenly() {
        let mut ledger = PrivilegeLedger::new(8, PrivilegeConfig::for_delta_scale(1.0));
        let mut t = PrivilegeTrace::new();
        t.push(0, 10.0);
        t.push(1, 1.0);
        ledger.observe_trace(&t, 1.0, 2.0, CreditAssignment::Uniform);
        assert!(
            (ledger.advantage(0) - ledger.advantage(1)).abs() < 1e-6,
            "Uniform must ignore weights: {} vs {}",
            ledger.advantage(0),
            ledger.advantage(1)
        );
    }

    #[test]
    fn gate_weighted_credit_follows_contribution_magnitude() {
        let mut ledger = PrivilegeLedger::new(8, PrivilegeConfig::for_delta_scale(1.0));
        let mut t = PrivilegeTrace::new();
        t.push(0, 9.0);
        t.push(1, 1.0);
        ledger.observe_trace(&t, 1.0, 2.0, CreditAssignment::GateWeighted);
        assert!(
            ledger.advantage(0) > 5.0 * ledger.advantage(1),
            "9:1 weights → ~9:1 credit, got {} vs {}",
            ledger.advantage(0),
            ledger.advantage(1)
        );
    }

    #[test]
    fn gate_weighted_falls_back_to_uniform_on_zero_weights() {
        let mut ledger = PrivilegeLedger::new(8, PrivilegeConfig::for_delta_scale(1.0));
        let mut t = PrivilegeTrace::new();
        t.push(0, 0.0);
        t.push(1, 0.0);
        ledger.observe_trace(&t, 1.0, 2.0, CreditAssignment::GateWeighted);
        assert!(ledger.advantage(0) > 0.0, "must not produce 0/0");
        assert!((ledger.advantage(0) - ledger.advantage(1)).abs() < 1e-6);
    }

    #[test]
    fn empty_trace_observation_is_a_noop() {
        let mut ledger = PrivilegeLedger::new(4, PrivilegeConfig::for_delta_scale(1.0));
        let t = PrivilegeTrace::new();
        ledger.observe_trace(&t, 1.0, 5.0, CreditAssignment::GateWeighted);
        assert_eq!(ledger.observations(), 0);
    }

    // ── privileged fuse ─────────────────────────────────────────────────────

    /// Fuse helper: run the privileged path and return the resulting state.
    fn run_privileged(
        table: &dyn EngramTable,
        query: &[f32],
        keys: &[EngramHash; K_MAX],
        ledger: &PrivilegeLedger,
    ) -> (Vec<f32>, PrivilegeTrace) {
        let d = table.dim();
        let cfg = EngramConfig::for_dim(d);
        let mut hidden = vec![0.0f32; d];
        let mut trace = PrivilegeTrace::new();
        let mut sl = vec![0.0f32; K_MAX * d];
        let mut so = vec![0.0f32; d];
        fuse_into_hidden_state_privileged(
            &mut hidden,
            query,
            table,
            keys,
            &cfg,
            ledger,
            &mut trace,
            &mut sl,
            &mut so,
        );
        (hidden, trace)
    }

    #[test]
    fn all_privilege_one_matches_unprivileged_fuse() {
        // G1 anchor: a ledger saturated to p ≈ 1 must reproduce the shipped
        // fuse to within f32 rounding.
        let d = 32;
        let table = make_table(d, 64);
        let cfg = PrivilegeConfig {
            margin: 0.0,
            scale: 0.001, // tiny scale → σ saturates to 1 for any Δ > 0
            ..PrivilegeConfig::for_delta_scale(1.0)
        };
        let mut ledger = PrivilegeLedger::new(table.num_slots(), cfg);
        for s in 0..table.num_slots() {
            ledger.observe(s, 1.0, 100.0);
        }

        let query: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.21).sin()).collect();
        let keys = [EngramHash(3); K_MAX];

        let (privileged, _) = run_privileged(&table, &query, &keys, &ledger);

        let ecfg = EngramConfig::for_dim(d);
        let mut plain = vec![0.0f32; d];
        let mut sl = vec![0.0f32; K_MAX * d];
        let mut sn = vec![0.0f32; d];
        let mut so = vec![0.0f32; d];
        fuse_into_hidden_state(
            &mut plain, &query, &table, &keys, &ecfg, &mut sl, &mut sn, &mut so,
        );

        for j in 0..d {
            assert!(
                (privileged[j] - plain[j]).abs() < 1e-4,
                "j={j}: privileged {} vs plain {}",
                privileged[j],
                plain[j]
            );
        }
    }

    #[test]
    fn uniform_cold_ledger_is_a_uniform_scale_of_the_plain_fuse() {
        // Ranking-preservation in its strongest form: with every slot at the
        // same Δ, the privileged output is the plain output times one constant.
        let d = 16;
        let table = make_table(d, 32);
        let cfg = PrivilegeConfig {
            margin: 0.0,
            ..PrivilegeConfig::for_delta_scale(1.0)
        };
        let ledger = PrivilegeLedger::new(table.num_slots(), cfg);
        let query: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.4).cos()).collect();
        let keys = [EngramHash(2); K_MAX];

        let (privileged, _) = run_privileged(&table, &query, &keys, &ledger);

        let ecfg = EngramConfig::for_dim(d);
        let mut plain = vec![0.0f32; d];
        let mut sl = vec![0.0f32; K_MAX * d];
        let mut sn = vec![0.0f32; d];
        let mut so = vec![0.0f32; d];
        fuse_into_hidden_state(
            &mut plain, &query, &table, &keys, &ecfg, &mut sl, &mut sn, &mut so,
        );

        for j in 0..d {
            assert!(
                (privileged[j] - 0.5 * plain[j]).abs() < 1e-4,
                "j={j}: expected exactly half, got {} vs {}",
                privileged[j],
                0.5 * plain[j]
            );
        }
    }

    #[test]
    fn vetoed_slot_contributes_nothing() {
        let d = 16;
        let table = make_table(d, 32);
        let cfg = PrivilegeConfig {
            margin: 0.0,
            scale: 0.01,
            ..PrivilegeConfig::for_delta_scale(1.0)
        };
        let mut ledger = PrivilegeLedger::new(table.num_slots(), cfg);
        for s in 0..table.num_slots() {
            ledger.observe(s, 1.0, -100.0); // everything is harmful
        }
        let query: Vec<f32> = vec![1.0; d];
        let keys = [EngramHash(1); K_MAX];
        let (hidden, _) = run_privileged(&table, &query, &keys, &ledger);
        assert!(
            hidden.iter().all(|h| h.abs() < 1e-5),
            "fully-vetoed table must leave the hidden state untouched: {hidden:?}"
        );
    }

    #[test]
    fn trace_records_fused_heads_and_is_cleared_on_entry() {
        let d = 8;
        let table = make_table(d, 32);
        let ledger = PrivilegeLedger::new(table.num_slots(), PrivilegeConfig::default());
        let query: Vec<f32> = vec![0.5; d];
        let keys = [EngramHash(1); K_MAX];

        let (_, trace) = run_privileged(&table, &query, &keys, &ledger);
        assert_eq!(trace.len(), K_MAX, "all heads hit populated slot 1");
        assert!(trace.slots().iter().all(|&s| s == 1));
        assert!(trace.weights().iter().all(|&w| w > 0.0));

        // Re-running against an empty slot must clear the previous contents.
        let empty_keys = [EngramHash(31); K_MAX]; // slot 31 was never written
        let (_, trace2) = run_privileged(&table, &query, &empty_keys, &ledger);
        assert!(trace2.is_empty(), "trace must be cleared on entry");
    }

    #[test]
    fn gate_floor_suppresses_low_similarity_heads_from_the_trace() {
        let d = 16;
        let mut b = EngramTableBuilder::new(32, d);
        // A pattern anti-aligned with the query → base gate ≈ σ(−√d) ≈ 0.018.
        let anti: Vec<f32> = (1..=d).map(|i| -(i as f32)).collect();
        b.add_pattern(EngramHash(5), &anti);
        let table = b.build();

        let query: Vec<f32> = (1..=d).map(|i| i as f32).collect();
        let keys = [EngramHash(5); K_MAX];

        let open = PrivilegeLedger::new(table.num_slots(), PrivilegeConfig::default());
        let (_, t_open) = run_privileged(&table, &query, &keys, &open);
        assert_eq!(t_open.len(), K_MAX, "floor 0 traces every populated head");

        let cfg = PrivilegeConfig {
            gate_floor: 0.25,
            ..PrivilegeConfig::default()
        };
        let gated = PrivilegeLedger::new(table.num_slots(), cfg);
        let (_, t_gated) = run_privileged(&table, &query, &keys, &gated);
        assert!(
            t_gated.is_empty(),
            "low-similarity heads must not consume scoring budget, got {} entries",
            t_gated.len()
        );
    }

    #[test]
    fn veto_epsilon_short_circuit_matches_the_exact_path_within_tolerance() {
        let d = 16;
        let table = make_table(d, 32);
        let query: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.3).sin()).collect();
        let keys = [EngramHash(2); K_MAX];

        let base_cfg = PrivilegeConfig {
            margin: 0.0,
            scale: 0.05,
            ..PrivilegeConfig::for_delta_scale(1.0)
        };
        let mut exact = PrivilegeLedger::new(table.num_slots(), base_cfg);
        for s in 0..table.num_slots() {
            exact.observe(s, 1.0, -1.0); // p ≈ σ(-20) ≈ 2e-9
        }
        let mut skipping = exact.clone();
        skipping.set_config(PrivilegeConfig {
            veto_epsilon: 1e-3,
            ..base_cfg
        });

        let (h_exact, _) = run_privileged(&table, &query, &keys, &exact);
        let (h_skip, _) = run_privileged(&table, &query, &keys, &skipping);
        for j in 0..d {
            assert!(
                (h_exact[j] - h_skip[j]).abs() < 1e-4,
                "j={j}: {} vs {}",
                h_exact[j],
                h_skip[j]
            );
        }
    }

    #[test]
    fn empty_table_privileged_fuse_is_a_noop() {
        let d = 8;
        let table = EngramTableBuilder::new(0, d).build();
        let ledger = PrivilegeLedger::for_table(&table, PrivilegeConfig::default());
        let query = vec![1.0f32; d];
        let keys = [EngramHash(7); K_MAX];
        let (hidden, trace) = run_privileged(&table, &query, &keys, &ledger);
        assert!(hidden.iter().all(|&h| h == 0.0));
        assert!(trace.is_empty());
    }

    #[test]
    fn for_delta_scale_rejects_degenerate_input() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let cfg = PrivilegeConfig::for_delta_scale(bad);
            assert_eq!(cfg.scale, 1.0, "degenerate {bad} must fall back to 1.0");
            assert!(cfg.scale > 0.0);
        }
    }
}

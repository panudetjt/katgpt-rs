//! `PkmEpisodicStore` — δ-rule write gate over ProductKeyMemory.
//!
//! Plan 408 Phase 5 (F1 fusion: PKM × δ-Mem). Wraps a
//! [`FrozenProductKeyMemory`] (the published, BLAKE3-committed snapshot) plus
//! a mutable working [`ProductKeyMemory`] (the write target). The `write`\n//! method performs a √N retrieval to find the top-k value rows touched by the
//! query, then applies a δ-rule update:
//!
//! ```text
//! V[idx] += gate * (target - V[idx])   for each idx in top-k(q)
//! ```
//!
//! This is bit-identical to **one** gradient-descent step at η=1 on the loss
//! `L = ½ Σ_idx ‖target - V[idx]‖²` (the gradient w.r.t. `V[idx]` is
//! `-(target - V[idx])`), scaled by `gate`. It is NOT iterated — one call is
//! one update. Iterating it N times would be gradient descent (forbidden per
//! AGENTS.md constraint #1); calling it once per surprising event is the same
//! modelless associative consolidation as [`crate::delta_mem::DeltaMemoryState::write`]
//! (Plan 053).
//!
//! # Why this is modelless (not gradient descent)
//!
//! The δ-rule is a Hebbian-style associative update: prediction error drives
//! consolidation. One step at η=1 is a closed-form move toward the target.
//! There is no loss-function evaluation, no backprop, no chain rule, no
//! learning-rate schedule — the update is a single SAXPY per value component.
//! The curiosity `gate` (the analog of the FwPKM paper's `g_t`) is sourced
//! EXTERNALLY by the caller (Temporal Derivative Kernel Plan 277, CGSP
//! Plan 274, BoM Sampler Plan 281) — this primitive is gate-agnostic.
//!
//! # Working copy vs published snapshot
//!
//! `write` mutates the working copy; [`publish`](PkmEpisodicStore::publish)
//! clones the working copy into the freeze slot (BLAKE3 commitment + atomic
//! swap). Readers of the freeze slot never see partial writes — they see
//! either the pre-write snapshot or the post-publish snapshot, never a
//! half-updated table. This mirrors the [`crate::induced_cwm::InducedCwmSlot`]
//! publish contract: write locally, publish atomically.
//!
//! # The `gate` parameter
//!
//! `gate ∈ [0, 1]` controls how far each top-k value row moves toward
//! `target`:
//!
//! | `gate` | Effect | Use case |
//! |---|---|---|
//! | `0.0` | no-op (write suppressed) | non-surprising event — skip |
//! | `1.0` | `V[idx] := target` (full overwrite) | one-shot memorization |
//! | `0.1` | 10% move toward target (EMA) | gentle consolidation over many events |
//!
//! Values outside `[0, 1]` are clamped before application. NaN is treated as
//! `0.0` (no-op).
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! - The `gate` scalar bridges the sync boundary (it's a curiosity signal
//!   sourced from synced state). It is consumed locally and never stored.
//! - The `target` vector is latent (it's a value-space pattern). It is NOT
//!   synced directly — only the BLAKE3 commitment of the resulting table
//!   crosses the sync boundary (via [`publish`](PkmEpisodicStore::publish)
//!   → [`FrozenProductKeyMemory::commit`]).
//! - The write/gate counters (`writes_total`, `writes_applied`, `gate_sum`)
//!   are raw audit scalars; consumers may sync them as telemetry.
//!
//! # Unweighted vs weighted update
//!
//! The default [`write`](PkmEpisodicStore::write) applies the same `gate`\n//! to every top-k slot, regardless of retrieval weight (the literal Plan 408
//! T5.1 formula). The [`write_weighted`](PkmEpisodicStore::write_weighted)
//! variant scales the per-slot update by the softmax retrieval weight:
//!
//! ```text
//! V[idx] += gate * weight[idx] * (target - V[idx])
//! ```
//!
//! The weighted variant is the gradient of
//! `L = ½ Σ_idx weight[idx] · ‖target - V[idx]‖²` — slots that matched the
//! query better receive a larger update. Consumers that want to preserve
//! value-row diversity (avoid all top-k slots collapsing toward `target`)
//! should prefer `write_weighted`; consumers that want uniform consolidation
//! (every retrieved slot learns the association equally) should use `write`.
//!
//! # TF-IDF write-gate selection (Issue 650 / Research 481)
//!
//! The plain `write`/`write_weighted` paths select write targets by raw
//! retrieval activation (TF-only ranking). Under dot-product scoring this
//! gravitates every write toward the handful of high-norm "magnate" slots
//! that every query retrieves — so successive fact sets overwrite the slots
//! general queries rely on. The paper's ablation ("Continual Learning via
//! Sparse Memory Finetuning", arXiv:2510.15103 §6 Fig 6) shows TF-only
//! selection forgets significantly more than TF-IDF selection, with the gap
//! widening as the write set shrinks — and our per-event write top-k is the
//! smallest write set in the stack.
//!
//! [`write_idf`](PkmEpisodicStore::write_idf) / [`write_weighted_idf`](PkmEpisodicStore::write_weighted_idf)
//! fix this: they retrieve a top-`k` candidate pool, re-rank by
//! `weight × idf(slot)` where `idf` comes from a static
//! [`BackgroundAccessStats`] table built once from a consumer-supplied
//! background query corpus, and apply the unchanged δ-rule to the top-`t`.
//! The δ-rule update itself is untouched; only selection changes.
//! [`write_selected`](PkmEpisodicStore::write_selected) is the escape hatch
//! for custom selection policies. See `.benchmarks/636_idf_write_gate.md`
//! for the measured GOAT evidence.
//!
//! # References
//!
//! - Plan: `katgpt-rs/.plans/408_Product_Key_Memory_Primitive.md` §Phase 5
//! - δ-Mem substrate: [`crate::delta_mem`] (the rank-r associative memory this
//!   primitive scales to √N slots)
//! - Freeze/thaw wrapper: [`crate::product_key_memory::FrozenProductKeyMemory`]
//! - FwPKM paper: [arXiv:2601.00671](https://arxiv.org/abs/2601.00671) — the
//!   `L_mem` GD half is forbidden; this is the modelless δ-rule analog.

use crate::product_key_memory::{FrozenProductKeyMemory, PkmScratch, ProductKeyMemory, ScoreFn};

/// Clamp `gate` into `[0, 1]`. NaN → 0.0 (no-op).
#[inline]
fn clamp_gate(gate: f32) -> f32 {
    if gate.is_nan() {
        return 0.0;
    }
    gate.clamp(0.0, 1.0)
}

/// δ-rule write gate over [`ProductKeyMemory`] (Plan 408 Phase 5, F1 fusion).
///
/// See the [module docs](self) for the full design: working copy vs published
/// snapshot, the `gate` parameter, and the modelless mandate.
///
/// # Type parameters
///
/// Inherits `SQRT_N`, `D_K`, `D_V` from the wrapped [`ProductKeyMemory`].
///
/// # Example
///
/// ```
/// use katgpt_core::product_key_memory::{
///     PkmEpisodicStore, PkmScratch, ProductKeyMemory, ScoreFn,
/// };
///
/// // Start from a random table.
/// let table: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(1);
/// let mut store = PkmEpisodicStore::new(table);
///
/// // One δ-rule write: query → top-k → move each value row 50% toward target.
/// let q = [0.5f32; 8];
/// let target = [1.0f32, 0.0, 0.0, 0.0];
/// let mut scratch = PkmScratch::<16, 4>::new();
/// let mut out = [(0usize, 0.0f32); 4];
/// let n = store.write(&q, &target, 0.5, ScoreFn::Dot, 4, &mut out, &mut scratch);
/// assert!(n > 0);
///
/// // Publish the updated working copy into the freeze slot.
/// let commit = store.publish();
/// assert_eq!(store.slot().current_version(), 1);
/// ```
pub struct PkmEpisodicStore<const SQRT_N: usize, const D_K: usize, const D_V: usize> {
    /// Working copy — mutated by [`write`](Self::write). NOT visible to readers
    /// of `slot` until [`publish`](Self::publish) clones it into the freeze slot.
    working: ProductKeyMemory<SQRT_N, D_K, D_V>,
    /// The freeze/thaw slot. Readers see `slot.current()`. `publish` swaps the
    /// working copy in atomically (BLAKE3 commitment + version bump).
    slot: FrozenProductKeyMemory<SQRT_N, D_K, D_V>,
    /// Total write calls (including gate-suppressed no-ops). Audit counter.
    writes_total: u64,
    /// Writes that actually mutated the value table (gate > 0 after clamp).
    /// Audit counter.
    writes_applied: u64,
    /// Cumulative clamped gate sum — for computing the mean lifetime gate.
    gate_sum: f64,
}

impl<const SQRT_N: usize, const D_K: usize, const D_V: usize> PkmEpisodicStore<SQRT_N, D_K, D_V> {
    /// Construct an episodic store pre-loaded with `table`.
    ///
    /// `table` becomes both the working copy (the write target) and version-0
    /// of the freeze slot (immediately readable via
    /// [`slot`](Self::slot).`current()`). The store takes ownership.
    pub fn new(table: ProductKeyMemory<SQRT_N, D_K, D_V>) -> Self {
        // Clone `table` into the freeze slot so the original becomes the working
        // copy. The freeze slot's version-0 commitment is computed lazily by
        // `current_commitment` (per the Phase 4 design).
        let slot = FrozenProductKeyMemory::new(table.clone());
        Self {
            working: table,
            slot,
            writes_total: 0,
            writes_applied: 0,
            gate_sum: 0.0,
        }
    }

    /// δ-rule write: retrieve top-k for `q`, move each top-k value row
    /// `gate` fraction toward `target`.
    ///
    /// `V[idx] += gate * (target - V[idx])` for each `idx` in `top-k(q)`.
    /// The retrieval weight is NOT used (every top-k slot receives the same
    /// `gate`-scaled move). See [`write_weighted`](Self::write_weighted) for
    /// the weight-scaled variant.
    ///
    /// # Arguments
    ///
    /// - `q`: the `D_K`-dim query (retrieval key).
    /// - `target`: the `D_V`-dim target value to consolidate into the top-k slots.
    /// - `gate`: curiosity / learning signal in `[0, 1]`. Clamped; NaN → 0.0.
    ///   See the [module docs](self#the-gate-parameter) for semantics.
    /// - `score_fn`: [`ScoreFn::Dot`] or [`ScoreFn::Idw`].
    /// - `k`: the final top-k (`<= K * K`, `<= out.len()`).
    /// - `out`: caller-allocated `(flat_index, weight)` scratch; `out[..n]` is
    ///   overwritten with the retrieval result. This is the SAME buffer
    ///   [`ProductKeyMemory::query_into`] writes — the caller can reuse it
    ///   across calls.
    /// - `scratch`: pre-allocated [`PkmScratch`] sized for per-codebook top-`K`.
    ///
    /// # Returns
    ///
    /// The number of top-k entries written into `out` (always `k` unless the
    /// table is degenerate). `0` if `gate <= 0` (write suppressed, no retrieval
    /// performed).
    ///
    /// # Panics
    ///
    /// Debug builds delegate to [`ProductKeyMemory::query_into`] which asserts
    /// `k <= K * K`, `k <= out.len()`, `K <= SQRT_N`.
    // Each parameter is independently meaningful on this hot path (see the
    // per-argument docs above); bundling them into a struct would obscure
    // call sites without reducing complexity, so the lint is allowed here.
    #[allow(clippy::too_many_arguments)]
    pub fn write<const K: usize>(
        &mut self,
        q: &[f32; D_K],
        target: &[f32; D_V],
        gate: f32,
        score_fn: ScoreFn,
        k: usize,
        out: &mut [(usize, f32)],
        scratch: &mut PkmScratch<SQRT_N, K>,
    ) -> usize {
        self.writes_total = self.writes_total.wrapping_add(1);
        let g = clamp_gate(gate);
        if g <= 0.0 {
            return 0;
        }

        // Step 1: √N retrieval on the working copy.
        let n = self.working.query_into(q, score_fn, k, out, scratch);

        // Step 2: δ-rule update on each retrieved value row.
        //   V[idx] += g * (target - V[idx])   (elementwise over D_V)
        //
        // Unweighted: every top-k slot receives the same `g`-scaled move,
        // regardless of retrieval weight. This is the literal Plan 408 T5.1
        // formula and bit-identical to one GD step at η=g on the unweighted
        // loss L = ½ Σ ‖target - V[idx]‖².
        for &(idx, _weight) in &out[..n] {
            // SAFETY: idx < SQRT_N * SQRT_N by query_into contract; the slice
            // bounds are therefore in-range. `value_mut` would also work but
            // the direct index avoids a second bounds check in debug builds.
            let row = &mut self.working.values[idx * D_V..(idx + 1) * D_V];
            for (v_j, &t_j) in row.iter_mut().zip(target.iter()) {
                *v_j += g * (t_j - *v_j);
            }
        }

        self.writes_applied = self.writes_applied.wrapping_add(1);
        self.gate_sum += g as f64;
        n
    }

    /// Weighted δ-rule write: same as [`write`](Self::write) but scales the
    /// per-slot update by the softmax retrieval weight.
    ///
    /// `V[idx] += gate * weight[idx] * (target - V[idx])` for each `idx` in
    /// `top-k(q)`. This is the gradient of the weighted loss
    /// `L = ½ Σ weight[idx] · ‖target - V[idx]‖²`, so higher-relevance slots
    /// receive a proportionally larger update. Use this variant when you want
    /// to preserve value-row diversity (avoid all top-k slots collapsing
    /// toward `target`).
    ///
    /// See [`write`](Self::write) for the argument contract. Returns the
    /// number of top-k entries written; `0` if `gate <= 0`.
    // See `write` above: each argument is independently meaningful on this
    // hot path, so bundling into a struct would not improve clarity.
    #[allow(clippy::too_many_arguments)]
    pub fn write_weighted<const K: usize>(
        &mut self,
        q: &[f32; D_K],
        target: &[f32; D_V],
        gate: f32,
        score_fn: ScoreFn,
        k: usize,
        out: &mut [(usize, f32)],
        scratch: &mut PkmScratch<SQRT_N, K>,
    ) -> usize {
        self.writes_total = self.writes_total.wrapping_add(1);
        let g = clamp_gate(gate);
        if g <= 0.0 {
            return 0;
        }

        let n = self.working.query_into(q, score_fn, k, out, scratch);

        // Weighted: scale the per-slot gate by the softmax retrieval weight.
        // weight ∈ (0, 1] (softmax-normalized by query_into).
        for &(idx, weight) in &out[..n] {
            let scaled_gate = g * weight;
            let row = &mut self.working.values[idx * D_V..(idx + 1) * D_V];
            for (v_j, &t_j) in row.iter_mut().zip(target.iter()) {
                *v_j += scaled_gate * (t_j - *v_j);
            }
        }

        self.writes_applied = self.writes_applied.wrapping_add(1);
        self.gate_sum += g as f64;
        n
    }

    /// Publish the working copy into the freeze slot.
    ///
    /// Clones the working [`ProductKeyMemory`] and atomically swaps it into
    /// the [`FrozenProductKeyMemory`] (BLAKE3 commitment + version bump). After
    /// this call, readers of the freeze slot see the post-write table; the
    /// working copy remains mutable for subsequent writes.
    ///
    /// # Returns
    ///
    /// The BLAKE3 commitment `[u8; 32]` of the newly-published table. Callers
    /// should pass this to the sync layer so other nodes can verify the swap.
    ///
    /// # Cost
    ///
    /// `O(SQRT_N² × D_V)` — one deep clone of the three flat slices plus one
    /// BLAKE3 pass. Intended for sleep-cycle cadence (seconds-scale), NOT
    /// per-tick.
    pub fn publish(&mut self) -> [u8; 32] {
        self.slot.commit(self.working.clone())
    }

    /// Read-only borrow of the working copy.
    ///
    /// Useful for inspection / debugging. The working copy reflects all writes
    /// since the last [`publish`](Self::publish); the freeze slot may lag
    /// behind.
    pub fn working(&self) -> &ProductKeyMemory<SQRT_N, D_K, D_V> {
        &self.working
    }

    /// Read-only borrow of the freeze/thaw slot.
    ///
    /// Readers concurrent with this store should call
    /// `store.slot().current()` to get a stable snapshot — they will NOT see
    /// writes until the next [`publish`](Self::publish).
    pub fn slot(&self) -> &FrozenProductKeyMemory<SQRT_N, D_K, D_V> {
        &self.slot
    }

    /// Total write calls (including gate-suppressed no-ops).
    #[inline]
    pub fn writes_total(&self) -> u64 {
        self.writes_total
    }

    /// Writes that actually mutated the value table (gate > 0 after clamp).
    #[inline]
    pub fn writes_applied(&self) -> u64 {
        self.writes_applied
    }

    /// Mean clamped gate over applied writes (`gate_sum / writes_applied`).
    ///
    /// Returns `0.0` if no writes have been applied yet.
    pub fn mean_gate(&self) -> f64 {
        if self.writes_applied == 0 {
            0.0
        } else {
            self.gate_sum / self.writes_applied as f64
        }
    }

    /// TF-IDF-ranked δ-rule write (Issue 650 / Research 481): retrieve a
    /// top-`k` candidate pool, re-rank by `weight × idf(slot)`, apply the
    /// unchanged unweighted δ-rule to the top-`t`.
    ///
    /// Selection: `score(idx) = weight[idx] * idf(idx)` where `idf` comes from
    /// a consumer-supplied [`BackgroundAccessStats`] built once from a
    /// background query corpus. Slots retrieved by many background batches
    /// (generally-hot / "magnate" slots) get low idf and are avoided — so a
    /// new fact set does not overwrite the slots general queries rely on
    /// ("Continual Learning via Sparse Memory Finetuning", arXiv:2510.15103
    /// §6 Fig 6: TF-only selection forgets significantly more, and the gap
    /// widens as the write set shrinks).
    ///
    /// Application: identical to [`write`](Self::write) — every selected slot
    /// receives the same `gate`-scaled move. The idf is a **selection
    /// statistic only** (not applied as an update scale, not a probability —
    /// not UQ-bearing; conformal floor N/A).
    ///
    /// # Arguments
    ///
    /// Same as [`write`](Self::write), plus:
    /// - `t`: the write width (`<= k`, `<= out.len()`). The top-`t` of the
    ///   candidate pool by `weight × idf` is written.
    /// - `stats`: background access statistics. `N` MUST equal
    ///   `SQRT_N * SQRT_N` (debug-asserted).
    ///
    /// # Returns
    ///
    /// The number of slots written (`min(t, n)`), or `0` if `gate <= 0`.
    ///
    /// # Degenerate stats = TF-only selection
    ///
    /// With an all-zero `slot_batch_counts`, idf is the constant `ln(|B|+1)`
    /// and selection reduces to top-`t` by retrieval weight — the baseline
    /// arm the Issue 650 bench measures against (matched candidate pool).
    #[allow(clippy::too_many_arguments)]
    pub fn write_idf<const K: usize, const N: usize>(
        &mut self,
        q: &[f32; D_K],
        target: &[f32; D_V],
        gate: f32,
        score_fn: ScoreFn,
        k: usize,
        t: usize,
        stats: &BackgroundAccessStats<N>,
        out: &mut [(usize, f32)],
        scratch: &mut PkmScratch<SQRT_N, K>,
    ) -> usize {
        debug_assert!(N == SQRT_N * SQRT_N, "stats N must equal SQRT_N^2");
        self.writes_total = self.writes_total.wrapping_add(1);
        let g = clamp_gate(gate);
        if g <= 0.0 {
            return 0;
        }

        // Step 1: candidate pool (top-k by retrieval score, softmax weights).
        let n = self.working.query_into(q, score_fn, k, out, scratch);

        // Step 2: re-rank by weight × idf, select top-t in place.
        //
        // `scratch.scores_1` is dead scratch at this point — query_into only
        // reads it back through its own top_1 selection, which has already
        // run. Reusing it for the selection scores keeps the call zero-alloc
        // with no new scratch parameter (contract: scores_1 is scratch, fully
        // rewritten by the next query_into call).
        let t_eff = select_top_t_by_idf(out, &mut scratch.scores_1, stats, n, t);

        // Step 3: unchanged unweighted δ-rule on the selected slots.
        for &(idx, _weight) in &out[..t_eff] {
            let row = &mut self.working.values[idx * D_V..(idx + 1) * D_V];
            for (v_j, &t_j) in row.iter_mut().zip(target.iter()) {
                *v_j += g * (t_j - *v_j);
            }
        }

        self.writes_applied = self.writes_applied.wrapping_add(1);
        self.gate_sum += g as f64;
        t_eff
    }

    /// TF-IDF-ranked weighted δ-rule write: same selection as
    /// [`write_idf`](Self::write_idf), but the application scales by the
    /// softmax retrieval weight (identical to
    /// [`write_weighted`](Self::write_weighted) on the selected slots).
    ///
    /// `V[idx] += gate * weight[idx] * (target - V[idx])` for each of the
    /// top-`t` slots by `weight[idx] × idf(idx)`. The idf ranks selection
    /// only — it never scales the update.
    // See `write` above: each argument is independently meaningful on this
    // hot path, so bundling them into a struct would not improve clarity.
    #[allow(clippy::too_many_arguments)]
    pub fn write_weighted_idf<const K: usize, const N: usize>(
        &mut self,
        q: &[f32; D_K],
        target: &[f32; D_V],
        gate: f32,
        score_fn: ScoreFn,
        k: usize,
        t: usize,
        stats: &BackgroundAccessStats<N>,
        out: &mut [(usize, f32)],
        scratch: &mut PkmScratch<SQRT_N, K>,
    ) -> usize {
        debug_assert!(N == SQRT_N * SQRT_N, "stats N must equal SQRT_N^2");
        self.writes_total = self.writes_total.wrapping_add(1);
        let g = clamp_gate(gate);
        if g <= 0.0 {
            return 0;
        }

        let n = self.working.query_into(q, score_fn, k, out, scratch);
        let t_eff = select_top_t_by_idf(out, &mut scratch.scores_1, stats, n, t);

        for &(idx, weight) in &out[..t_eff] {
            let scaled_gate = g * weight;
            let row = &mut self.working.values[idx * D_V..(idx + 1) * D_V];
            for (v_j, &t_j) in row.iter_mut().zip(target.iter()) {
                *v_j += scaled_gate * (t_j - *v_j);
            }
        }

        self.writes_applied = self.writes_applied.wrapping_add(1);
        self.gate_sum += g as f64;
        t_eff
    }

    /// Apply the unweighted δ-rule to a caller-selected set of slots
    /// (advanced escape hatch for custom selection policies).
    ///
    /// `V[idx] += gate * (target - V[idx])` for each `idx` in `indices`.
    /// This is the substrate under [`write_idf`](Self::write_idf)'s
    /// application step, exposed so consumers can implement their own
    /// selection policy (the Issue 650 bench's random-`t` control arm uses
    /// it) without reaching into the working table.
    ///
    /// # Contract
    ///
    /// Every `idx` MUST be `< SQRT_N * SQRT_N` (out-of-range panics via the
    /// value-row slice — fail-loud, not fail-wrong). Entries are the
    /// `(slot, weight)` pairs from [`ProductKeyMemory::query_into`]; the
    /// weight is ignored (uniform gate application).
    pub fn write_selected(
        &mut self,
        selected: &[(usize, f32)],
        target: &[f32; D_V],
        gate: f32,
    ) -> usize {
        self.writes_total = self.writes_total.wrapping_add(1);
        let g = clamp_gate(gate);
        if g <= 0.0 {
            return 0;
        }
        for &(idx, _weight) in selected {
            let row = &mut self.working.values[idx * D_V..(idx + 1) * D_V];
            for (v_j, &t_j) in row.iter_mut().zip(target.iter()) {
                *v_j += g * (t_j - *v_j);
            }
        }
        self.writes_applied = self.writes_applied.wrapping_add(1);
        self.gate_sum += g as f64;
        selected.len()
    }
}

/// Static background access-count table for TF-IDF write-gate selection
/// (Issue 650 / Research 481, distilled from arXiv:2510.15103 §6).
///
/// Built ONCE from a consumer-supplied background query corpus (the analog
/// of the paper's static background indices stored in the checkpoint).
/// `slot_batch_counts[i]` is the document frequency: the number of background
/// batches whose retrieved-slot union contained slot `i`. The IDF,
///
/// ```text
/// idf(i) = ln((|B| + 1) / (1 + slot_batch_counts[i]))
///
/// (|B| = n_batches)
/// ```
///
/// is the same smoothed log-ratio the riir-neuron-db `Bm25Index` applies to
/// term selection, applied here to slot selection. It is `ln(|B|+1)` for
/// never-retrieved slots and `0` for slots retrieved by every batch — so a
/// generally-hot "magnate" slot scores `weight × 0 = 0` and is never
/// selected ahead of a specific slot with any positive weight.
///
/// # Modelless + sync boundary
///
/// Pure static statistics — no training, no learning-rate, built once from a
/// retrieval pass. `u32` counts; fixed-size arrays (no heap). The selection
/// score is a ranking statistic, **not** a probability — not UQ-bearing,
/// conformal floor N/A. Nothing here crosses the sync boundary (the stats
/// table is local to the write path; only the BLAKE3 table commitment
/// crosses via [`PkmEpisodicStore::publish`]).
///
/// # Example
///
/// ```
/// use katgpt_core::product_key_memory::BackgroundAccessStats;
///
/// let mut stats = BackgroundAccessStats::<4>::new();
/// stats.record_batch(&[0, 3]);       // batch 1 retrieved slots 0 and 3
/// stats.record_batch(&[0]);           // batch 2 retrieved slot 0
///
/// // Slot 0 (in both batches) has lower idf than slot 3 (in one).
/// assert!(stats.idf(0) < stats.idf(3));
/// // Never-retrieved slot 2: idf = ln(|B| + 1) = ln(3).
/// assert!((stats.idf(2) - 3.0f32.ln()).abs() < 1e-6);
/// ```
pub struct BackgroundAccessStats<const N: usize> {
    /// Number of background batches recorded (|B|).
    n_batches: u32,
    /// Document frequency: number of batches whose retrieved-slot union
    /// contained slot `i`. Capped implicitly at `n_batches`.
    slot_batch_counts: [u32; N],
}

impl<const N: usize> BackgroundAccessStats<N> {
    /// Zero-initialized stats (no batches recorded). idf is uniformly `0`
    /// until the first `record_batch` — build stats before use.
    pub fn new() -> Self {
        Self {
            n_batches: 0,
            slot_batch_counts: [0; N],
        }
    }

    /// Record one background batch: bump the count for each of the batch's
    /// **distinct** retrieved slots.
    ///
    /// # Contract
    ///
    /// - `slots` must contain **distinct** indices (the constructor
    ///   [`build_background_stats`](Self::build_background_stats) dedups
    ///   internally; duplicates here would double-count).
    /// - Every index must be `< N` (out-of-range panics via the array index
    ///   — fail-loud).
    pub fn record_batch(&mut self, slots: &[usize]) {
        self.n_batches = self.n_batches.wrapping_add(1);
        for &idx in slots {
            self.slot_batch_counts[idx] += 1;
        }
    }

    /// Smoothed inverse document frequency of `slot`:
    /// `ln((n_batches + 1) / (1 + slot_batch_counts[slot]))`.
    ///
    /// `ln(|B|+1)` for never-retrieved slots, strictly decreasing in the
    /// count, `0` for slots retrieved by every batch. Bounds-checked
    /// (panics if `slot >= N`).
    #[inline]
    pub fn idf(&self, slot: usize) -> f32 {
        ((self.n_batches as f32 + 1.0) / (1.0 + self.slot_batch_counts[slot] as f32)).ln()
    }

    /// Number of background batches recorded so far (`|B|`).
    #[inline]
    pub fn n_batches(&self) -> u32 {
        self.n_batches
    }

    /// Document frequency of `slot` — number of batches whose retrieved-slot
    /// union contained it. Bounds-checked.
    #[inline]
    pub fn slot_batch_count(&self, slot: usize) -> u32 {
        self.slot_batch_counts[slot]
    }

    /// Build stats from a background query corpus against `table` (the
    /// recommended source is the published snapshot,
    /// `store.slot().current()`).
    ///
    /// Queries are grouped into consecutive batches of `batch_size`; for each
    /// batch, every query retrieves its top-`k` slots and the **union** of
    /// the retrieved slots is recorded as one batch (mirrors the paper's
    /// batch-level document frequency). The trailing partial batch counts as
    /// one batch.
    ///
    /// # Arguments
    ///
    /// - `table`: the table to retrieve against (frozen snapshot recommended
    ///   — the stats are static like the paper's checkpoint-stored indices).
    /// - `background_queries`: flat slice of `D_K`-dim queries.
    /// - `batch_size`: queries per background batch (≥ 1).
    /// - `score_fn` / `k`: retrieval config for the background pass.
    /// - `out` / `scratch`: retrieval scratch (same buffers the write path
    ///   uses; reused across the whole build).
    /// - `batch_slots`: scratch for one batch's slot union, capacity ≥
    ///   `batch_size * k` (deduped in place via sort + compaction).
    ///
    /// # Panics
    ///
    /// Panics if `batch_size == 0` or `batch_slots.len() < batch_size * k`.
    /// Debug-asserts `N == SQRT_N * SQRT_N`.
    ///
    /// # Alloc
    ///
    /// Zero-alloc (sort + compaction in place on the caller buffer).
    #[allow(clippy::too_many_arguments)]
    pub fn build_background_stats<
        const SQRT_N: usize,
        const D_K: usize,
        const D_V: usize,
        const K: usize,
    >(
        table: &ProductKeyMemory<SQRT_N, D_K, D_V>,
        background_queries: &[[f32; D_K]],
        batch_size: usize,
        score_fn: ScoreFn,
        k: usize,
        out: &mut [(usize, f32)],
        batch_slots: &mut [usize],
        scratch: &mut PkmScratch<SQRT_N, K>,
    ) -> Self {
        debug_assert!(
            N == SQRT_N * SQRT_N,
            "BackgroundAccessStats<N> with N={} but table has SQRT_N^2={} slots",
            N,
            SQRT_N * SQRT_N
        );
        assert!(batch_size >= 1, "batch_size must be >= 1");
        assert!(
            batch_slots.len() >= batch_size * k,
            "batch_slots must hold batch_size*k = {} entries (got {})",
            batch_size * k,
            batch_slots.len()
        );
        let mut stats = Self::new();
        for batch in background_queries.chunks(batch_size) {
            // Collect this batch's retrieved slots (with duplicates).
            let mut len = 0usize;
            for q in batch {
                let n = table.query_into(q, score_fn, k, out, scratch);
                for &(idx, _w) in &out[..n] {
                    batch_slots[len] = idx;
                    len += 1;
                }
            }
            // Distinct slots: sort + compaction (zero-alloc dedup). Indexed
            // access — the compaction writes while iterating the same buffer.
            batch_slots[..len].sort_unstable();
            let mut write = 0usize;
            let mut prev = usize::MAX;
            for read in 0..len {
                let idx = batch_slots[read];
                if idx != prev {
                    batch_slots[write] = idx;
                    write += 1;
                    prev = idx;
                }
            }
            stats.record_batch(&batch_slots[..write]);
        }
        stats
    }
}

impl<const N: usize> Default for BackgroundAccessStats<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-rank `out[..n]` by `weight × idf(slot)` and select the top-`t` in
/// place (partial selection sort, co-swapping `out` and `scores`).
///
/// Deterministic: ties keep the earlier entry (query_into returns `out`
/// sorted weight-descending, so ties preserve retrieval order). `scores`
/// must have length ≥ `n` — the write paths reuse the dead
/// `PkmScratch::scores_1` buffer for this.
#[inline]
fn select_top_t_by_idf<const N: usize>(
    out: &mut [(usize, f32)],
    scores: &mut [f32],
    stats: &BackgroundAccessStats<N>,
    n: usize,
    t: usize,
) -> usize {
    debug_assert!(scores.len() >= n, "scores scratch must hold n entries");
    let t = t.min(n);
    for i in 0..n {
        scores[i] = out[i].1 * stats.idf(out[i].0);
    }
    // Partial selection sort: top-t by score descending. Strict `>` keeps
    // the earliest (highest retrieval-ranked) entry on ties.
    for i in 0..t {
        let mut best = i;
        for j in (i + 1)..n {
            if scores[j] > scores[best] {
                best = j;
            }
        }
        scores.swap(i, best);
        out.swap(i, best);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test dimensions (kept small for fast tests) ───────────────────────────
    //
    // SQRT_N=16 → N=256 slots. D_K=8 (halves are 4-dim). D_V=4. K=4 (per-codebook
    // top-k), final k=4 (<= K*K=16).

    /// Build a store from `from_random(seed)` for tests.
    fn store_from_seed(seed: u64) -> PkmEpisodicStore<16, 8, 4> {
        PkmEpisodicStore::new(ProductKeyMemory::from_random(seed))
    }

    // ── Contract: gate semantics ──────────────────────────────────────────────

    #[test]
    fn write_gate_zero_is_noop() {
        let mut store = store_from_seed(1);
        let q = [0.5f32; 8];
        let target = [1.0f32, 0.0, 0.0, 0.0];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];

        // Snapshot the working values before the write.
        let before: Vec<f32> = store.working().values_flat().to_vec();

        let n = store.write(&q, &target, 0.0, ScoreFn::Dot, 4, &mut out, &mut scratch);

        // gate=0 → write suppressed, no retrieval performed.
        assert_eq!(n, 0, "gate=0 should suppress the write");
        assert_eq!(
            store.writes_total(),
            1,
            "writes_total should still count the suppressed call"
        );
        assert_eq!(
            store.writes_applied(),
            0,
            "writes_applied should NOT count the suppressed call"
        );
        assert_eq!(
            store.working().values_flat(),
            before.as_slice(),
            "gate=0 should not mutate any value row"
        );
    }

    #[test]
    fn write_gate_nan_is_noop() {
        let mut store = store_from_seed(1);
        let q = [0.5f32; 8];
        let target = [1.0f32; 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];

        let before: Vec<f32> = store.working().values_flat().to_vec();
        let n = store.write(
            &q,
            &target,
            f32::NAN,
            ScoreFn::Dot,
            4,
            &mut out,
            &mut scratch,
        );

        assert_eq!(n, 0, "NaN gate should be treated as 0 (no-op)");
        assert_eq!(
            store.working().values_flat(),
            before.as_slice(),
            "NaN gate should not mutate any value row"
        );
    }

    #[test]
    fn write_gate_one_is_full_overwrite_of_top_k() {
        let mut store = store_from_seed(1);
        let q = [0.5f32; 8];
        let target = [1.0f32, 0.5, -0.5, -1.0];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];

        let n = store.write(&q, &target, 1.0, ScoreFn::Dot, 4, &mut out, &mut scratch);
        assert!(n > 0, "gate=1 should apply the write");

        // Every top-k slot should now equal `target` exactly (gate=1 →
        // V[idx] += 1.0 * (target - V[idx]) = target).
        for &(idx, _w) in &out[..n] {
            let row = store.working().value(idx);
            for (v_j, &t_j) in row.iter().zip(target.iter()) {
                assert_eq!(
                    *v_j, t_j,
                    "gate=1 should fully overwrite top-k slot {idx} to target"
                );
            }
        }
        assert_eq!(store.writes_applied(), 1);
    }

    #[test]
    fn write_gate_half_moves_halfway() {
        let mut store = store_from_seed(1);
        let q = [0.5f32; 8];
        let target = [1.0f32; 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];

        // Snapshot before values for the top-k slots.
        let n = store.write(&q, &target, 0.5, ScoreFn::Dot, 4, &mut out, &mut scratch);
        assert!(n > 0);

        // After gate=0.5: V'[idx] = 0.5*V[idx] + 0.5*target = midpoint.
        // We can't check against the PRE-write values (already mutated), but
        // we CAN check the invariant: V'[idx] = 0.5*V_old + 0.5*target.
        // Reconstruct: V'[idx] - 0.5*target = 0.5*V_old → V_old = 2*V'[idx] - target.
        // For this to be consistent, V_old should be the from_random value.
        // Simpler check: the move is exactly half the distance to target.
        // V'[idx] = V_old + 0.5*(target - V_old) = 0.5*V_old + 0.5*target.
        // So |V'[idx] - target| = 0.5 * |V_old - target|.
        // We verify by writing AGAIN with gate=0.5 from the same query — the
        // second write should halve the distance again (geometric decay).
        let dist_after_first: f32 = out[..n]
            .iter()
            .map(|&(idx, _)| {
                let row = store.working().value(idx);
                row.iter()
                    .zip(target.iter())
                    .map(|(v, t)| (v - t).abs())
                    .sum::<f32>()
            })
            .sum();

        let n2 = store.write(&q, &target, 0.5, ScoreFn::Dot, 4, &mut out, &mut scratch);
        assert_eq!(n2, n, "same query should retrieve the same top-k");

        let dist_after_second: f32 = out[..n]
            .iter()
            .map(|&(idx, _)| {
                let row = store.working().value(idx);
                row.iter()
                    .zip(target.iter())
                    .map(|(v, t)| (v - t).abs())
                    .sum::<f32>()
            })
            .sum();

        // The second write halves the distance again (EMA with β=0.5).
        let ratio = dist_after_second / dist_after_first.max(1e-12);
        assert!(
            (ratio - 0.5).abs() < 0.01,
            "second gate=0.5 write should halve the distance (ratio≈0.5, got {ratio})"
        );
    }

    #[test]
    fn write_gate_clamped_above_one() {
        let mut store = store_from_seed(1);
        let q = [0.5f32; 8];
        let target = [1.0f32; 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];

        // gate=2.0 should be clamped to 1.0 → full overwrite (same as gate=1).
        let n = store.write(&q, &target, 2.0, ScoreFn::Dot, 4, &mut out, &mut scratch);
        assert!(n > 0);
        for &(idx, _w) in &out[..n] {
            let row = store.working().value(idx);
            for (v_j, &t_j) in row.iter().zip(target.iter()) {
                assert_eq!(*v_j, t_j, "gate=2.0 clamped to 1.0 → full overwrite");
            }
        }
        // mean_gate should be 1.0 (the clamped value), not 2.0.
        assert!(
            (store.mean_gate() - 1.0).abs() < 1e-9,
            "mean_gate should reflect the clamped gate (1.0), got {}",
            store.mean_gate()
        );
    }

    #[test]
    fn write_gate_negative_clamped_to_zero() {
        let mut store = store_from_seed(1);
        let q = [0.5f32; 8];
        let target = [1.0f32; 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];
        let before: Vec<f32> = store.working().values_flat().to_vec();

        let n = store.write(&q, &target, -0.5, ScoreFn::Dot, 4, &mut out, &mut scratch);
        assert_eq!(n, 0, "gate=-0.5 clamped to 0 → no-op");
        assert_eq!(
            store.working().values_flat(),
            before.as_slice(),
            "negative gate should not mutate values"
        );
    }

    // ── Contract: only top-k is touched ───────────────────────────────────────

    #[test]
    fn write_only_touches_top_k_slots() {
        let mut store = store_from_seed(1);
        let q = [0.5f32; 8];
        let target = [1.0f32; 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];

        let before: Vec<f32> = store.working().values_flat().to_vec();
        let k = 4;
        let n = store.write(&q, &target, 0.5, ScoreFn::Dot, k, &mut out, &mut scratch);
        assert_eq!(n, k);

        let touched: std::collections::HashSet<usize> =
            out[..n].iter().map(|&(idx, _)| idx).collect();

        let num_slots = ProductKeyMemory::<16, 8, 4>::num_slots();
        for idx in 0..num_slots {
            let row_before = &before[idx * 4..(idx + 1) * 4];
            let row_after = store.working().value(idx);
            if touched.contains(&idx) {
                // Touched slots should differ (unless target happened to equal
                // the original — extremely unlikely with from_random).
                let any_diff = row_before.iter().zip(row_after.iter()).any(|(a, b)| a != b);
                assert!(any_diff, "touched slot {idx} should have been mutated");
            } else {
                // Untouched slots should be byte-identical.
                assert_eq!(
                    row_before, row_after,
                    "untouched slot {idx} should be unchanged"
                );
            }
        }
    }

    // ── Contract: weighted variant scales by retrieval weight ─────────────────

    #[test]
    fn write_weighted_scales_by_retrieval_weight() {
        // Two stores from the same seed → identical starting tables.
        let q = [0.5f32; 8];
        let target = [1.0f32; 4];
        let gate = 1.0;

        // Unweighted.
        let mut store_u = store_from_seed(42);
        let mut scratch_u = PkmScratch::<16, 4>::new();
        let mut out_u = [(0usize, 0.0f32); 4];
        let n_u = store_u.write(
            &q,
            &target,
            gate,
            ScoreFn::Dot,
            4,
            &mut out_u,
            &mut scratch_u,
        );

        // Weighted.
        let mut store_w = store_from_seed(42);
        let mut scratch_w = PkmScratch::<16, 4>::new();
        let mut out_w = [(0usize, 0.0f32); 4];
        let n_w = store_w.write_weighted(
            &q,
            &target,
            gate,
            ScoreFn::Dot,
            4,
            &mut out_w,
            &mut scratch_w,
        );

        // Same top-k set.
        assert_eq!(n_u, n_w);
        let set_u: std::collections::HashSet<usize> =
            out_u[..n_u].iter().map(|&(i, _)| i).collect();
        let set_w: std::collections::HashSet<usize> =
            out_w[..n_w].iter().map(|&(i, _)| i).collect();
        assert_eq!(set_u, set_w, "weighted and unweighted retrieve same top-k");

        // The unweighted update applies `gate` to every slot.
        // The weighted update applies `gate * weight` to each slot.
        // Since weights ∈ (0, 1] and sum to 1, the weighted update is smaller
        // (or equal for weight=1) for every slot. The max-weight slot gets the
        // same update in both iff its weight == 1.0 (single-slot softmax peak).
        //
        // Verify: for each top-k slot, weighted_distance <= unweighted_distance
        // from the original from_random(42) value.
        let original = ProductKeyMemory::<16, 8, 4>::from_random(42);
        for &(idx, weight) in &out_w[..n_w] {
            let orig_row = original.value(idx);
            let u_row = store_u.working().value(idx);
            let w_row = store_w.working().value(idx);

            let dist_u: f32 = orig_row
                .iter()
                .zip(u_row.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let dist_w: f32 = orig_row
                .iter()
                .zip(w_row.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            // Weighted distance should be weight * unweighted_distance
            // (since scaled_gate = gate * weight, and the update is linear in
            // the gate for a single write).
            let expected_ratio = weight;
            let actual_ratio = dist_w / dist_u.max(1e-12);
            assert!(
                (actual_ratio - expected_ratio).abs() < 0.01,
                "slot {idx} weight {weight}: dist_w/dist_u should be ≈{expected_ratio}, got {actual_ratio}"
            );
        }
    }

    // ── Contract: publish swaps working into slot ─────────────────────────────

    #[test]
    fn publish_swaps_working_into_slot() {
        let mut store = store_from_seed(1);
        let commit_v0 = store.slot().current_commitment().unwrap();

        // Write a few updates.
        let q = [0.5f32; 8];
        let target = [1.0f32, 0.0, 0.0, 0.0];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];
        for _ in 0..5 {
            store.write(&q, &target, 0.3, ScoreFn::Dot, 4, &mut out, &mut scratch);
        }

        // Before publish: slot still shows the original table.
        assert_eq!(
            store.slot().current_commitment().unwrap(),
            commit_v0,
            "slot should show v0 before publish"
        );
        assert_eq!(store.slot().current_version(), 0);

        // Publish.
        let commit_v1 = store.publish();
        assert_ne!(
            commit_v1, commit_v0,
            "publish should produce a new commitment"
        );
        assert_eq!(
            store.slot().current_commitment().unwrap(),
            commit_v1,
            "slot should show v1 after publish"
        );
        assert_eq!(store.slot().current_version(), 1);

        // The published snapshot should match the working copy.
        let published = store.slot().current().unwrap();
        assert_eq!(
            published.values_flat(),
            store.working().values_flat(),
            "published snapshot should match working copy after publish"
        );
    }

    #[test]
    fn publish_does_not_break_subsequent_writes() {
        let mut store = store_from_seed(1);

        // Write + publish.
        let q = [0.5f32; 8];
        let target = [1.0f32; 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];
        store.write(&q, &target, 0.5, ScoreFn::Dot, 4, &mut out, &mut scratch);
        let _commit_v1 = store.publish();

        // Write again — should work fine, mutating the working copy.
        let before: Vec<f32> = store.working().values_flat().to_vec();
        store.write(&q, &target, 0.5, ScoreFn::Dot, 4, &mut out, &mut scratch);
        assert_ne!(
            store.working().values_flat(),
            before.as_slice(),
            "write after publish should still mutate the working copy"
        );

        // The slot should still show v1 (not yet re-published).
        assert_eq!(store.slot().current_version(), 1);

        // Re-publish → v2.
        let commit_v2 = store.publish();
        assert_eq!(store.slot().current_version(), 2);
        assert_eq!(store.slot().current_commitment().unwrap(), commit_v2);
    }

    // ── Contract: audit counters ──────────────────────────────────────────────

    #[test]
    fn audit_counters_track_writes() {
        let mut store = store_from_seed(1);
        let q = [0.5f32; 8];
        let target = [1.0f32; 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];

        // 3 applied writes + 2 suppressed.
        store.write(&q, &target, 0.5, ScoreFn::Dot, 4, &mut out, &mut scratch);
        store.write(&q, &target, 1.0, ScoreFn::Dot, 4, &mut out, &mut scratch);
        store.write(&q, &target, 0.25, ScoreFn::Dot, 4, &mut out, &mut scratch);
        store.write(&q, &target, 0.0, ScoreFn::Dot, 4, &mut out, &mut scratch); // suppressed
        store.write(
            &q,
            &target,
            f32::NAN,
            ScoreFn::Dot,
            4,
            &mut out,
            &mut scratch,
        ); // suppressed

        assert_eq!(store.writes_total(), 5);
        assert_eq!(store.writes_applied(), 3);
        // mean_gate = (0.5 + 1.0 + 0.25) / 3 = 0.583...
        let expected_mean = (0.5f64 + 1.0 + 0.25) / 3.0;
        assert!(
            (store.mean_gate() - expected_mean).abs() < 1e-9,
            "mean_gate should be {}, got {}",
            expected_mean,
            store.mean_gate()
        );
    }

    // ── Determinism (G6 substrate) ────────────────────────────────────────────

    #[test]
    fn write_is_deterministic_same_inputs_same_outputs() {
        let q = [0.5f32; 8];
        let target = [1.0f32, -0.5, 0.25, 0.75];
        let gate = 0.37;

        // Two independent stores from the same seed.
        let mut store_a = store_from_seed(99);
        let mut store_b = store_from_seed(99);

        let mut scratch_a = PkmScratch::<16, 4>::new();
        let mut scratch_b = PkmScratch::<16, 4>::new();
        let mut out_a = [(0usize, 0.0f32); 4];
        let mut out_b = [(0usize, 0.0f32); 4];

        let n_a = store_a.write(
            &q,
            &target,
            gate,
            ScoreFn::Dot,
            4,
            &mut out_a,
            &mut scratch_a,
        );
        let n_b = store_b.write(
            &q,
            &target,
            gate,
            ScoreFn::Dot,
            4,
            &mut out_b,
            &mut scratch_b,
        );

        // Same top-k set + weights.
        assert_eq!(n_a, n_b);
        assert_eq!(out_a, out_b, "top-k output should be bit-identical");

        // Same working copy after the write.
        assert_eq!(
            store_a.working().values_flat(),
            store_b.working().values_flat(),
            "working copy should be bit-identical after same write"
        );

        // Same publish commitment.
        let commit_a = store_a.publish();
        let commit_b = store_b.publish();
        assert_eq!(
            commit_a, commit_b,
            "publish commitment should be bit-identical for same inputs"
        );
    }

    // ── IDW scoring works through the write path ──────────────────────────────

    #[test]
    fn write_works_with_idw_scoring() {
        let mut store = store_from_seed(1);
        let q = [0.5f32; 8];
        let target = [1.0f32; 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut out = [(0usize, 0.0f32); 4];

        let n = store.write(
            &q,
            &target,
            0.5,
            ScoreFn::Idw { epsilon: 1e-6 },
            4,
            &mut out,
            &mut scratch,
        );
        assert!(n > 0, "IDW write should retrieve a non-empty top-k");

        // IDW top-k should generally differ from Dot top-k on a random table.
        let mut store_dot = store_from_seed(1);
        let mut scratch_dot = PkmScratch::<16, 4>::new();
        let mut out_dot = [(0usize, 0.0f32); 4];
        let n_dot = store_dot.write(
            &q,
            &target,
            0.5,
            ScoreFn::Dot,
            4,
            &mut out_dot,
            &mut scratch_dot,
        );

        // The two top-k sets may or may not overlap (depends on the table).
        // The contract here is just that both write paths run without panic
        // and produce non-empty results.
        assert!(n_dot > 0);
    }

    // ── Issue 650 / Research 481: TF-IDF write gate ─────────────────────────

    #[test]
    fn idf_monotone_decreasing_in_count() {
        let mut stats = BackgroundAccessStats::<8>::new();
        // 3 batches; slot 0 in all 3, slot 1 in 2, slot 2 in 1, slot 3 in none.
        stats.record_batch(&[0, 1, 2]);
        stats.record_batch(&[0, 1]);
        stats.record_batch(&[0]);
        assert_eq!(stats.n_batches(), 3);
        // Strictly decreasing in count.
        assert!(stats.idf(0) < stats.idf(1));
        assert!(stats.idf(1) < stats.idf(2));
        // Slots 3+ never accessed → equal idf (ln(|B|+1)).
        assert!((stats.idf(3) - stats.idf(4)).abs() < 1e-6);
        // Slot retrieved by EVERY batch: idf = ln((3+1)/(3+1)) = ln(1) = 0.
        assert!(stats.idf(0) >= 0.0);
        assert!(stats.idf(0) < 1e-6);
    }

    #[test]
    fn idf_never_accessed_is_ln_nb_plus_one() {
        let mut stats = BackgroundAccessStats::<4>::new();
        stats.record_batch(&[0]);
        stats.record_batch(&[0]);
        // |B| = 2 → never-accessed idf = ln(3).
        let expected = 3.0f32.ln();
        assert!((stats.idf(1) - expected).abs() < 1e-6);
        assert!((stats.idf(2) - expected).abs() < 1e-6);
        // Empty stats: |B| = 0 → idf = ln(1/1) = 0 (degenerate, documented).
        let empty = BackgroundAccessStats::<4>::new();
        assert!(empty.idf(0).abs() < 1e-6);
    }

    #[test]
    fn build_background_stats_counts_match_direct_queries() {
        // batch_size=1 → each query is one batch; counts must equal the
        // per-query top-k membership (top-k indices are distinct by
        // construction — cartesian pairs are unique (i,j) cells).
        let table: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(7);
        let queries: Vec<[f32; 8]> = (0..6)
            .map(|i| {
                let mut q = [0.0f32; 8];
                for (j, x) in q.iter_mut().enumerate() {
                    *x = ((i * 13 + j * 7) as f32 * 0.37).sin();
                }
                q
            })
            .collect();
        let stats = BackgroundAccessStats::<256>::build_background_stats(
            &table,
            &queries,
            1,
            ScoreFn::Dot,
            4,
            &mut [(0usize, 0.0f32); 4],
            &mut [0usize; 4],
            &mut PkmScratch::<16, 4>::new(),
        );
        assert_eq!(stats.n_batches(), 6);
        // Total counts == total (query, slot) retrievals.
        let mut out = [(0usize, 0.0f32); 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let mut direct_total = 0u32;
        for q in &queries {
            let n = table.query_into(q, ScoreFn::Dot, 4, &mut out, &mut scratch);
            direct_total += n as u32;
            for &(idx, _) in &out[..n] {
                assert!(stats.slot_batch_count(idx) >= 1);
            }
        }
        let total: u32 = (0..256).map(|i| stats.slot_batch_count(i)).sum();
        assert_eq!(total, direct_total);
    }

    #[test]
    fn build_background_stats_dedups_within_batch() {
        // Two identical queries in one batch → the union is one query's
        // top-k, not double-counted.
        let table: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(7);
        let queries = vec![[0.5f32; 8], [0.5f32; 8]];
        let stats = BackgroundAccessStats::<256>::build_background_stats(
            &table,
            &queries,
            2,
            ScoreFn::Dot,
            4,
            &mut [(0usize, 0.0f32); 4],
            &mut [0usize; 8],
            &mut PkmScratch::<16, 4>::new(),
        );
        assert_eq!(stats.n_batches(), 1);
        let mut out = [(0usize, 0.0f32); 4];
        let mut scratch = PkmScratch::<16, 4>::new();
        let n = table.query_into(&queries[0], ScoreFn::Dot, 4, &mut out, &mut scratch);
        for i in 0..256 {
            let expected = if out[..n].iter().any(|&(idx, _)| idx == i) {
                1
            } else {
                0
            };
            assert_eq!(stats.slot_batch_count(i), expected, "slot {i}");
        }
    }

    #[test]
    fn write_idf_degenerate_stats_selects_by_weight() {
        // All-zero counts → uniform idf → selection = top-t by weight from
        // the SAME candidate pool (the matched-pool TF-only baseline the
        // bench uses). Expected = first t of the pool (out is weight-desc).
        let mut store = store_from_seed(3);
        let mut stats = BackgroundAccessStats::<256>::new();
        stats.record_batch(&[]); // |B|=1, no counts → idf = ln(2) uniform

        let q = [0.3f32, -0.7, 0.1, 0.9, -0.4, 0.2, 0.8, -0.6];
        let target = [1.0f32, 0.0, 0.5, -0.25];

        // Reference pool from an identical untouched store.
        let reference = store_from_seed(3);
        let mut pool = [(0usize, 0.0f32); 8];
        let mut scratch_ref = PkmScratch::<16, 4>::new();
        let n = reference
            .working()
            .query_into(&q, ScoreFn::Dot, 8, &mut pool, &mut scratch_ref);
        let expected: Vec<usize> = pool[..n].iter().map(|&(idx, _)| idx).take(4).collect();

        let mut out = [(0usize, 0.0f32); 8];
        let mut scratch = PkmScratch::<16, 4>::new();
        let t_written = store.write_idf(
            &q,
            &target,
            0.5,
            ScoreFn::Dot,
            8,
            4,
            &stats,
            &mut out,
            &mut scratch,
        );
        assert_eq!(t_written, 4);
        let got: Vec<usize> = out[..4].iter().map(|&(idx, _)| idx).collect();
        assert_eq!(got, expected, "uniform idf must select top-t by weight");
    }

    #[test]
    fn write_idf_avoids_zero_idf_slots() {
        // "Magnate" slots retrieved by EVERY background batch score
        // weight × 0 = 0 and must never be selected ahead of any positive-idf
        // candidate.
        let mut store = store_from_seed(5);
        let q = [0.25f32, -0.5, 0.75, 0.1, -0.9, 0.4, 0.6, -0.2];
        let target = [0.5f32, 0.5, -0.5, 0.5];

        // Pool + magnate identification (top-3 by weight).
        let reference = store_from_seed(5);
        let mut pool = [(0usize, 0.0f32); 8];
        let mut scratch0 = PkmScratch::<16, 4>::new();
        let n = reference
            .working()
            .query_into(&q, ScoreFn::Dot, 8, &mut pool, &mut scratch0);
        let magnates = [pool[0].0, pool[1].0, pool[2].0];

        // 10 background batches, each retrieving exactly the magnates.
        let mut stats = BackgroundAccessStats::<256>::new();
        for _ in 0..10 {
            stats.record_batch(&magnates);
        }

        let mut out = [(0usize, 0.0f32); 8];
        let mut scratch = PkmScratch::<16, 4>::new();
        let t_written = store.write_idf(
            &q,
            &target,
            1.0,
            ScoreFn::Dot,
            8,
            4,
            &stats,
            &mut out,
            &mut scratch,
        );
        assert_eq!(t_written, 4);
        let selected: Vec<usize> = out[..4].iter().map(|&(idx, _)| idx).collect();
        for m in magnates {
            assert!(
                !selected.contains(&m),
                "zero-idf magnate {m} must not be selected (got {selected:?})"
            );
        }
        // Selected = top-4 by weight among the survivors = pool ranks 3..7
        // (weights are distinct almost surely under softmax).
        for &s in &selected {
            let rank = pool[..n].iter().position(|&(idx, _)| idx == s).unwrap();
            assert!(rank >= 3, "selected rank {rank} should be >= 3");
        }
        // Magnates' value rows are bit-identical to the reference.
        for m in magnates {
            let got = store.working().value(m);
            let old = reference.working().value(m);
            assert_eq!(got, old, "magnate {m} must be untouched");
        }
    }

    #[test]
    fn write_idf_gate_zero_is_noop() {
        let mut store = store_from_seed(11);
        let reference = store_from_seed(11);
        let mut stats = BackgroundAccessStats::<256>::new();
        for _ in 0..4 {
            stats.record_batch(&[0, 5, 10]);
        }
        let q = [0.1f32; 8];
        let target = [1.0f32; 4];
        let mut out = [(0usize, 0.0f32); 8];
        let mut scratch = PkmScratch::<16, 4>::new();
        let n = store.write_idf(
            &q,
            &target,
            0.0,
            ScoreFn::Dot,
            8,
            4,
            &stats,
            &mut out,
            &mut scratch,
        );
        assert_eq!(n, 0);
        assert_eq!(store.writes_total(), 1);
        assert_eq!(store.writes_applied(), 0);
        for i in 0..256 {
            assert_eq!(store.working().value(i), reference.working().value(i));
        }
    }

    #[test]
    fn write_idf_is_deterministic() {
        let q = [0.4f32, -0.2, 0.6, -0.8, 0.3, 0.7, -0.1, 0.5];
        let target = [0.25f32, -0.75, 0.5, 1.0];
        let mut stats = BackgroundAccessStats::<256>::new();
        for _ in 0..4 {
            stats.record_batch(&[1, 2, 3, 250]);
        }

        let run = |seed: u64| -> Vec<u8> {
            let mut store = store_from_seed(seed);
            let mut out = [(0usize, 0.0f32); 8];
            let mut scratch = PkmScratch::<16, 4>::new();
            for i in 0..8 {
                let q_i = {
                    let mut q2 = q;
                    q2[0] += i as f32 * 0.01;
                    q2
                };
                store.write_idf(
                    &q_i,
                    &target,
                    0.7,
                    ScoreFn::Dot,
                    8,
                    4,
                    &stats,
                    &mut out,
                    &mut scratch,
                );
            }
            store
                .working()
                .values
                .iter()
                .map(|b| b.to_bits() as u8)
                .collect()
        };
        assert_eq!(run(21), run(21), "same seed → bit-identical");
    }

    #[test]
    fn write_weighted_idf_scales_by_weight_on_selected() {
        // With uniform stats (selection = top-t by weight from pool),
        // write_weighted_idf(pool=8, t=4) applies the softmax-over-8 weights
        // of the pool's top-4 entries (NOTE: not bit-identical to
        // write_weighted(k=4) — softmax normalizes over the selected k, so
        // the weight VALUES differ between pool-8 and direct-k=4 despite the
        // same selected set). Pin the actual contract: same slots as the
        // pool's top-4, each scaled by its pool-8 softmax weight.
        let q = [-0.3f32, 0.8, 0.2, -0.6, 0.5, -0.4, 0.9, 0.1];
        let target = [0.75f32, -0.25, 1.0, 0.5];
        let mut stats = BackgroundAccessStats::<256>::new();
        stats.record_batch(&[]); // uniform idf = ln(2)

        // Reference pool (softmax-over-8 weights, weight-desc order).
        let reference = store_from_seed(33);
        let mut pool = [(0usize, 0.0f32); 8];
        let mut scratch_ref = PkmScratch::<16, 4>::new();
        let n = reference
            .working()
            .query_into(&q, ScoreFn::Dot, 8, &mut pool, &mut scratch_ref);
        let expected: Vec<(usize, f32)> = pool[..n].iter().copied().take(4).collect();

        let mut store = store_from_seed(33);
        let mut out = [(0usize, 0.0f32); 8];
        let mut scratch = PkmScratch::<16, 4>::new();
        let t_written = store.write_weighted_idf(
            &q,
            &target,
            0.8,
            ScoreFn::Dot,
            8,
            4,
            &stats,
            &mut out,
            &mut scratch,
        );
        assert_eq!(t_written, 4);

        // Selected slots match the pool's top-4 by weight.
        let selected: Vec<(usize, f32)> = out[..4].to_vec();
        let sel_idx: Vec<usize> = selected.iter().map(|e| e.0).collect();
        let exp_idx: Vec<usize> = expected.iter().map(|e| e.0).collect();
        assert_eq!(sel_idx, exp_idx);

        // Each selected slot moved by gate * pool-8-weight toward target.
        for &(idx, weight) in &expected {
            let got = store.working().value(idx);
            let old = reference.working().value(idx);
            let scaled = 0.8 * weight;
            for ((g, o), t) in got.iter().zip(old.iter()).zip(target.iter()) {
                let e = o + scaled * (t - o);
                assert!(
                    (g - e).abs() < 1e-6,
                    "slot {idx} weight {weight} diverged: got {got:?} expected {e}"
                );
            }
        }
        // Non-selected slots are bit-identical to the reference.
        for i in 0..256 {
            if !expected.iter().any(|&(idx, _)| idx == i) {
                assert_eq!(store.working().value(i), reference.working().value(i));
            }
        }
    }

    #[test]
    fn write_selected_applies_delta_to_caller_slots() {
        let mut store = store_from_seed(9);
        let reference = store_from_seed(9);
        let target = [2.0f32, -1.0, 0.5, 0.0];
        // Arbitrary slots + dummy weights (the weight must be ignored).
        let selected = [(3usize, 0.9f32), (17usize, 0.1f32)];
        let n = store.write_selected(&selected, &target, 0.5);
        assert_eq!(n, 2);
        assert_eq!(store.writes_total(), 1);
        assert_eq!(store.writes_applied(), 1);
        for &(idx, _) in &selected {
            let got = store.working().value(idx);
            let old = reference.working().value(idx);
            for ((g, o), t) in got.iter().zip(old.iter()).zip(target.iter()) {
                let expected = o + 0.5 * (t - o);
                assert!((g - expected).abs() < 1e-6);
            }
        }
        // Untouched slot is bit-identical.
        assert_eq!(store.working().value(100), reference.working().value(100));
    }
}

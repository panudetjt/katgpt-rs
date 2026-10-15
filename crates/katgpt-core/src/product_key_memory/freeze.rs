//! `FrozenProductKeyMemory` — atomic freeze/thaw wrapper for
//! [`ProductKeyMemory`].
//!
//! Plan 408 Phase 4 (F4 fusion). Implements the freeze/thaw contract:
//!
//! - **Readers** (the hot path, `query_into`) never see a torn snapshot —
//!   either the old table or the new one, never a half-swapped mix of
//!   `keys_1` from version N and `values` from version N+1.
//! - **Writers** (`commit`) atomically replace the entire table and return a
//!   BLAKE3 commitment over the three flat slices. The commitment is the
//!   syncable audit artifact (raw, deterministic, replayable) — the slot
//!   itself is process-local and does not cross the sync boundary.
//!
//! # Pattern: same as `InducedCwmSlot` / `micro_belief::snapshot`
//!
//! Per the established katgpt-core precedent, this wrapper uses
//! `Arc<RwLock<Arc<ProductKeyMemory>>>` rather than `arc_swap::ArcSwap`:
//!
//! 1. **katgpt-core does NOT depend on `arc-swap`**, and since riir-ai Issue
//!    950 T7 no workspace crate does at runtime (mirrors the
//!    `induced_cwm/hot_swap.rs` decision, documented at length there).
//! 2. **The hot path tolerates `RwLock` read-lock cost** (~10ns uncontended
//!    on x86_64) because readers clone the `Arc` out (one refcount bump) and
//!    release the lock immediately, then run the √N scoring on the cloned
//!    `Arc`. The read critical section is just `guard.clone()` — sub-µs even
//!    for the largest tables.
//! 3. **Writers are rare** (sleep-cycle consolidation cadence, seconds-scale)
//!    so `RwLock` writer contention is not a concern.
//!
//! **Do NOT swap to `arc-swap`'s default strategy** if a profile ever shows
//! read contention: riir-ai Issue 950 measured it handing readers
//! freed-and-reused values under load (vorner/arc-swap#210). See
//! `induced_cwm/hot_swap.rs` for the full note.
//!
//! # Why `RwLock<Arc<T>>` and not `RwLock<T>`
//!
//! Storing `RwLock<Arc<ProductKeyMemory>>` lets readers clone-out a stable
//! snapshot (one `Arc` refcount bump) and then release the read lock BEFORE
//! running the √N query. Callers that want to run many queries against a
//! consistent snapshot (e.g. a batch retrieval loop) call `current()` once,
//! hold the `Arc`, and query it repeatedly — the table cannot be swapped out
//! from under them mid-batch. This is the same "clone-on-read" contract as
//! `InducedCwmSlot::current`.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |---|---|---|
//! | Slot keys + values (latent patterns) | Latent | NO (process-local) |
//! | BLAKE3 commitment `[u8; 32]` | Raw | YES (audit event / chain sync) |
//! | `version: u64` (monotonic counter) | Raw | YES (audit event) |
//! | `FrozenProductKeyMemory.inner` | — | NO (slot is process-local; the *commitment* is the syncable artifact) |
//!
//! The top-k weights produced by `query_into` are latent scalars — they
//! bridge at the sync boundary per the parent module docs, not here.
//!
//! # Hot-path vs cold-path
//!
//! | Layer | What | Tier |
//! |---|---|---|
//! | Sleep-cycle consolidation (writer) | δ-rule / TEMP diversity → new table | Cold (background) |
//! | [`FrozenProductKeyMemory::commit`] | BLAKE3 + atomic swap | Cold (event) |
//! | [`FrozenProductKeyMemory::current`] | Read-lock + Arc clone | **Hot** (once per retrieval batch) |
//! | `query_into` on cloned Arc | √N scoring | **Hot** (per query) |
//!
//! # Determinism
//!
//! The BLAKE3 commitment is deterministic: byte-identical tables produce
//! identical commitments (G6 substrate). The hash is computed over the
//! little-endian byte representation of the three flat `&[f32]` slices via
//! `bytemuck::cast_slice` (valid on all our targets — x86_64 and aarch64 are
//! both LE). A leading domain tag `b"pkm_v1"` distinguishes this commitment
//! from other BLAKE3 digests in the system.
//!
//! # References
//!
//! - Plan: `katgpt-rs/.plans/408_Product_Key_Memory_Primitive.md` §Phase 4
//! - Precedent (katgpt-core): [`crate::induced_cwm::InducedCwmSlot`]
//!   (`induced_cwm/hot_swap.rs`) — the `Arc<RwLock<Option<...>>>` pattern
//!   this module generalizes to √N×√N tables.
//! - Precedent (riir-ai): `LoRAWeightVersion` (Issue 354) — the
//!   `concurrent_lora_no_torn_read` stress test this module's T4.2 test
//!   generalizes.
//! - Cross-repo FV: the freeze/thaw reader invariant Lean theorem T2 lives
//!   in `riir-ai/.proofs/RiirAiProof/Runtime/FreezeThaw.lean`; this wrapper
//!   is its Rust-side spec-match target (Issue 348 T2, closed).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::product_key_memory::{PkmScratch, ProductKeyMemory, ScoreFn};

/// BLAKE3 domain-separation tag. Prepended to every commitment hash input so
/// PKM commitments cannot collide with digests from other subsystems that
/// hash raw f32 slices.
const COMMITMENT_TAG: &[u8] = b"pkm_v1";

/// Atomic freeze/thaw wrapper around [`ProductKeyMemory`].
///
/// Readers ([`current`](Self::current) /
/// [`query_into`](Self::query_into)) never observe a torn snapshot; writers
/// ([`commit`](Self::commit)) atomically install a new table under a write
/// lock and return a BLAKE3 commitment. See the [module docs](self) for the
/// full pattern and rationale.
///
/// # Type parameters
///
/// Inherits `SQRT_N`, `D_K`, `D_V` from the wrapped [`ProductKeyMemory`].
///
/// # Example
///
/// ```
/// use katgpt_core::product_key_memory::{
///     FrozenProductKeyMemory, ProductKeyMemory, ScoreFn, PkmScratch,
/// };
///
/// // Build a table and freeze it.
/// let table_v1: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(1);
/// let frozen = FrozenProductKeyMemory::new(table_v1);
/// let commit_v1 = frozen.current_commitment().unwrap();
///
/// // Hot-swap to a new table.
/// let table_v2: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(2);
/// let commit_v2 = frozen.commit(table_v2);
/// assert_ne!(commit_v1, commit_v2);
/// assert_eq!(frozen.current_commitment().unwrap(), commit_v2);
/// assert_eq!(frozen.current_version(), 1); // bumped from 0 → 1 by the commit
/// ```
pub struct FrozenProductKeyMemory<const SQRT_N: usize, const D_K: usize, const D_V: usize> {
    /// Inner storage. Readers clone the `Arc` out under a read lock; writers
    /// swap the `Arc` under a write lock. The read critical section is one
    /// refcount bump.
    ///
    /// `RwLock` rather than `ArcSwap` because katgpt-core doesn't depend on
    /// `arc-swap` — see the module docs for the full rationale.
    inner: Arc<RwLock<Option<Arc<ProductKeyMemory<SQRT_N, D_K, D_V>>>>>,
    /// Monotonic contents ordinal, bumped on every successful
    /// [`commit`](Self::commit). Starts at 0 (the table passed to
    /// [`new`](Self::new) is version 0); the first `commit` moves it to 1.
    /// NOT part of the BLAKE3 input — same semantics as `CwmCommitment.version`.
    ///
    /// Wrapped in `Arc` so clones of the slot share the counter (the
    /// "fan-out" pattern — see [`Clone`]). The counter is advisory (audit /
    /// sync logging); the torn-read guarantee comes from the `RwLock<Arc>`
    /// table swap, not from this counter, so it does NOT need to be
    /// atomically consistent with the table swap. Readers may briefly observe
    /// `version = N` alongside a table installed by commit `N+1` — this is
    /// acceptable because `version` is never used for correctness gating,
    /// only for human-readable audit trails.
    version: Arc<AtomicU64>,
}

// Manual `Clone` — bumping the inner `Arc` refcount shares BOTH the
// RwLock-protected table storage AND the `AtomicU64` version counter. Two
// clones of a `FrozenProductKeyMemory` observe the same table, the same
// hot-swaps, AND the same monotonic version (the "fan-out" pattern: a
// top-level slot gets cloned into per-worker slots). This mirrors
// `InducedCwmSlot::clone`.
impl<const SQRT_N: usize, const D_K: usize, const D_V: usize> Clone
    for FrozenProductKeyMemory<SQRT_N, D_K, D_V>
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            version: Arc::clone(&self.version),
        }
    }
}

impl<const SQRT_N: usize, const D_K: usize, const D_V: usize>
    FrozenProductKeyMemory<SQRT_N, D_K, D_V>
{
    /// Construct a frozen slot pre-loaded with `table` at version 0.
    ///
    /// The slot is immediately readable. Use [`empty`](Self::empty) instead
    /// if you want a slot with no table installed yet (the lazy-load path).
    pub fn new(table: ProductKeyMemory<SQRT_N, D_K, D_V>) -> Self {
        // NOTE: we deliberately do NOT compute + cache the commitment at
        // construction. `commit` is the only path that needs the hash; the
        // `new` path leaves it to `current_commitment()` to compute lazily
        // if a caller asks. This keeps `new` allocation-free beyond the
        // `Arc` + `RwLock` construction.
        Self {
            inner: Arc::new(RwLock::new(Some(Arc::new(table)))),
            version: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Construct an empty slot (no table installed). All read methods return
    /// `None` until the first [`commit`](Self::commit).
    ///
    /// Useful for the "load from disk on startup" path where the slot exists
    /// before the first table is deserialized.
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
            version: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Hot-swap the table.
    ///
    /// Computes the BLAKE3 commitment of `table`'s three flat slices,
    /// atomically installs `table` under the write lock (readers that called
    /// [`current`](Self::current) before this call keep observing the old
    /// `Arc`; readers that call after see the new one), bumps the version
    /// counter, and returns the commitment.
    ///
    /// # Returns
    ///
    /// The freshly-computed BLAKE3 commitment `[u8; 32]` for the new table.
    /// Callers should pass this to the sync layer so other nodes can verify
    /// the swap. Returns `[0u8; 32]` is NEVER possible — `commit` always
    /// hashes a real table.
    ///
    /// # Panics
    ///
    /// Panics if the inner lock is poisoned (a previous writer panicked while
    /// holding the write lock). This is unrecoverable — the slot is corrupt.
    pub fn commit(&self, table: ProductKeyMemory<SQRT_N, D_K, D_V>) -> [u8; 32] {
        let commitment = compute_commitment(&table);
        // Write lock — blocks new readers until the swap is done. The
        // critical section is one `Option` assignment + `Arc` construction.
        // Microseconds even for the largest tables.
        let mut guard = self
            .inner
            .write()
            .expect("FrozenProductKeyMemory lock poisoned");
        *guard = Some(Arc::new(table));
        // Bump version AFTER the swap succeeds, so readers that observe
        // version N are guaranteed to see the N-th table (or later). The
        // AcqRel ordering pairs with the Acquire load in `current_version`.
        // NOTE: the version is advisory — a reader may briefly see the new
        // table but the old version. This is documented at the field decl.
        self.version.fetch_add(1, Ordering::AcqRel);
        commitment
    }

    /// Atomically read the current table, cloning the `Arc` out.
    ///
    /// Returns `None` if the slot is empty (no table committed yet, or the
    /// table has been retired). The returned `Arc` is a stable snapshot —
    /// subsequent [`commit`](Self::commit) calls do not affect it.
    ///
    /// # Hot-path note
    ///
    /// This call takes the read lock for the duration of one `Arc` clone
    /// (~5ns refcount bump uncontended). Callers running batch retrieval
    /// should call this once per batch, not per query — see the module docs.
    ///
    /// # Panics
    ///
    /// Panics if the inner lock is poisoned.
    pub fn current(&self) -> Option<Arc<ProductKeyMemory<SQRT_N, D_K, D_V>>> {
        let guard = self
            .inner
            .read()
            .expect("FrozenProductKeyMemory lock poisoned");
        guard.as_ref().map(Arc::clone)
    }

    /// Convenience: run [`ProductKeyMemory::query_into`] against the current
    /// snapshot. Takes the read lock, clones the `Arc`, releases the lock,
    /// then queries the clone.
    ///
    /// Returns `0` if the slot is empty. Otherwise delegates to
    /// [`ProductKeyMemory::query_into`] — see that method for the full
    /// contract.
    ///
    /// # When to use this vs [`current`](Self::current)
    ///
    /// Use this for one-shot queries. For batch retrieval (many queries
    /// against the same snapshot), call [`current`](Self::current) once and
    /// loop over `query_into` on the returned `Arc` — avoids the per-query
    /// lock-acquire + Arc-clone overhead.
    pub fn query_into<const K: usize>(
        &self,
        q: &[f32; D_K],
        score_fn: ScoreFn,
        k: usize,
        out: &mut [(usize, f32)],
        scratch: &mut PkmScratch<SQRT_N, K>,
    ) -> usize {
        match self.current() {
            Some(table) => table.query_into(q, score_fn, k, out, scratch),
            None => 0,
        }
    }

    /// Cheap accessor for the current table's BLAKE3 commitment.
    ///
    /// Returns `None` if the slot is empty. Recomputes the hash on every
    /// call (one BLAKE3 pass over the three flat slices) — for hot paths,
    /// cache the result of [`commit`](Self::commit) instead.
    pub fn current_commitment(&self) -> Option<[u8; 32]> {
        let guard = self
            .inner
            .read()
            .expect("FrozenProductKeyMemory lock poisoned");
        guard.as_ref().map(|arc| compute_commitment(arc))
    }

    /// Verify that the current table's BLAKE3 commitment matches `expected`.
    ///
    /// Returns `false` if the slot is empty OR the recomputed hash differs
    /// from `expected`. Returns `true` only if the slot is non-empty AND the
    /// hash matches.
    ///
    /// Cheap enough for audit paths (one BLAKE3 pass); do NOT call per-tick.
    pub fn verify(&self, expected: &[u8; 32]) -> bool {
        match self.current_commitment() {
            Some(actual) => &actual == expected,
            None => false,
        }
    }

    /// Cheap accessor for the monotonic version counter.
    ///
    /// Returns the number of [`commit`](Self::commit) calls that have
    /// succeeded on this slot. Starts at 0 (the
    /// [`new`](Self::new)-installed table); each `commit` bumps it by 1.
    /// Two clones of the same slot share the counter.
    pub fn current_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Returns `true` iff no table is currently installed.
    ///
    /// Cheap (one lock acquisition, no clone).
    pub fn is_empty(&self) -> bool {
        let guard = self
            .inner
            .read()
            .expect("FrozenProductKeyMemory lock poisoned");
        guard.is_none()
    }

    /// Get the number of strong references to the inner storage.
    ///
    /// Diagnostic / test utility. Returns 1 for a freshly-constructed slot,
    /// 2 immediately after [`Clone::clone`], etc.
    pub fn arc_strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl<const SQRT_N: usize, const D_K: usize, const D_V: usize> Default
    for FrozenProductKeyMemory<SQRT_N, D_K, D_V>
{
    fn default() -> Self {
        Self::empty()
    }
}

// ── Commitment ────────────────────────────────────────────────────────────

/// Compute the BLAKE3 commitment over a table's three flat slices.
///
/// Deterministic: byte-identical tables produce identical commitments (G6
/// substrate). The hash input is `TAG || keys_1 || keys_2 || values` where
/// each slice is cast to `&[u8]` via `bytemuck::cast_slice` (little-endian
/// on all our targets — x86_64, aarch64).
///
/// Not length-prefixed: the lengths are determined by the const generics
/// (`SQRT_N`, `D_K`, `D_V`), so two tables with the same generics have the
/// same layout and the concatenation is unambiguous. Tables with DIFFERENT
/// generics are different Rust types and cannot be compared via commitment
/// anyway — the caller picks the type at construction.
fn compute_commitment<const SQRT_N: usize, const D_K: usize, const D_V: usize>(
    table: &ProductKeyMemory<SQRT_N, D_K, D_V>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(COMMITMENT_TAG);
    // Safety / portability: `cast_slice::<f32, u8>` yields the host's native
    // byte representation. On our targets (x86_64, aarch64) this is LE, so
    // the commitment matches `f.to_le_bytes()` iterated. We do NOT commit to
    // cross-arch determinism — chain-sync assumes homogenous archs, and the
    // commitment is an audit artifact, not a serialization format.
    hasher.update(bytemuck::cast_slice::<f32, u8>(table.keys_1.as_ref()));
    hasher.update(bytemuck::cast_slice::<f32, u8>(table.keys_2.as_ref()));
    hasher.update(bytemuck::cast_slice::<f32, u8>(table.values.as_ref()));
    *hasher.finalize().as_bytes()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Build a table from `seed` with three flat slices that are each filled
    /// with a single characteristic value `fill`. Used by the bit-identity
    /// and torn-read tests — `fill` is the "version marker" that lets a
    /// reader detect cross-slice inconsistency.
    fn table_with_fill<const SQRT_N: usize, const D_K: usize, const D_V: usize>(
        fill: f32,
    ) -> ProductKeyMemory<SQRT_N, D_K, D_V> {
        let half = ProductKeyMemory::<SQRT_N, D_K, D_V>::key_half_dim();
        let keys_1 = vec![fill; SQRT_N * half].into_boxed_slice();
        let keys_2 = vec![fill; SQRT_N * half].into_boxed_slice();
        let values = vec![fill; SQRT_N * SQRT_N * D_V].into_boxed_slice();
        ProductKeyMemory::new(keys_1, keys_2, values)
    }

    /// The three flat slices (`keys_1`, `keys_2`, `values`) that make up a
    /// [`ProductKeyMemory`]'s storage.
    type TableSlices = (Box<[f32]>, Box<[f32]>, Box<[f32]>);

    /// Clone a table's three flat slices into fresh `Box<[f32]>`s. Used by
    /// the bit-identity test to construct a 1-bit-flipped copy without
    /// needing `ProductKeyMemory: Clone`.
    fn clone_table_slices<const SQRT_N: usize, const D_K: usize, const D_V: usize>(
        src: &ProductKeyMemory<SQRT_N, D_K, D_V>,
    ) -> TableSlices {
        (src.keys_1.clone(), src.keys_2.clone(), src.values.clone())
    }

    // ── T4.1 — basic freeze/thaw contract ───────────────────────────────

    #[test]
    fn new_installs_table_at_version_0() {
        let table: ProductKeyMemory<8, 4, 2> = ProductKeyMemory::from_random(42);
        let frozen = FrozenProductKeyMemory::new(table);
        assert_eq!(frozen.current_version(), 0);
        assert!(!frozen.is_empty());
        assert!(frozen.current().is_some());
        // Strong-count: just the inner Arc.
        assert_eq!(frozen.arc_strong_count(), 1);
    }

    /// Type alias to keep the test signatures readable.
    type FrozenProductMemory<const SQRT_N: usize, const D_K: usize, const D_V: usize> =
        FrozenProductKeyMemory<SQRT_N, D_K, D_V>;

    #[test]
    fn empty_slot_has_no_table() {
        let frozen: FrozenProductMemory<8, 4, 2> = FrozenProductKeyMemory::empty();
        assert!(frozen.is_empty());
        assert_eq!(frozen.current_version(), 0);
        assert!(frozen.current().is_none());
        assert!(frozen.current_commitment().is_none());
        assert!(!frozen.verify(&[0u8; 32]));
    }

    #[test]
    fn commit_installs_new_table_and_bumps_version() {
        let t1: ProductKeyMemory<8, 4, 2> = ProductKeyMemory::from_random(1);
        let frozen = FrozenProductKeyMemory::new(t1);
        let v0_commitment = frozen.current_commitment().unwrap();
        assert_eq!(frozen.current_version(), 0);

        let t2: ProductKeyMemory<8, 4, 2> = ProductKeyMemory::from_random(2);
        let v1_commitment = frozen.commit(t2);
        assert_eq!(frozen.current_version(), 1);
        assert_ne!(v0_commitment, v1_commitment);
        assert_eq!(frozen.current_commitment().unwrap(), v1_commitment);
        assert!(frozen.verify(&v1_commitment));
        assert!(!frozen.verify(&v0_commitment));
    }

    #[test]
    fn current_returns_stable_snapshot_independent_of_commit() {
        // Clone out a snapshot, commit a new table, verify the old snapshot
        // is unaffected (the Arc is a stable handle to the old table).
        let frozen = FrozenProductKeyMemory::new(table_with_fill::<8, 4, 2>(1.0));
        let snap = frozen.current().unwrap();
        // Sanity: the snapshot's first value matches what we installed.
        assert_eq!(snap.values[0], 1.0);

        frozen.commit(table_with_fill::<8, 4, 2>(2.0));
        // The new table is visible to a fresh current() call.
        let snap2 = frozen.current().unwrap();
        assert_eq!(snap2.values[0], 2.0);
        // The OLD snapshot is unchanged — this is the no-torn-read guarantee
        // for batch retrieval (callers holding `snap` across a commit).
        assert_eq!(snap.values[0], 1.0);
    }

    #[test]
    fn query_into_delegates_to_current_snapshot() {
        let frozen = FrozenProductKeyMemory::new(table_with_fill::<8, 4, 2>(0.5));
        let q = [0.5f32; 4];
        let mut scratch = PkmScratch::<8, 4>::default();
        let mut out = [(0usize, 0.0f32); 4];
        let n = frozen.query_into(&q, ScoreFn::Dot, 4, &mut out, &mut scratch);
        assert!(n > 0, "query_into on a frozen slot should return > 0");
        // Weights are softmax-normalized → sum to ~1.
        let sum: f32 = out[..n].iter().map(|(_, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "weights should sum to 1, got {sum}"
        );
    }

    #[test]
    fn query_into_on_empty_slot_returns_zero() {
        let frozen: FrozenProductMemory<8, 4, 2> = FrozenProductKeyMemory::empty();
        let q = [0.0f32; 4];
        let mut scratch = PkmScratch::<8, 4>::default();
        let mut out = [(0usize, 0.0f32); 4];
        let n = frozen.query_into(&q, ScoreFn::Dot, 4, &mut out, &mut scratch);
        assert_eq!(n, 0, "empty slot should return 0 results");
    }

    #[test]
    fn clone_shares_inner_storage() {
        let frozen = FrozenProductKeyMemory::new(table_with_fill::<8, 4, 2>(1.0));
        let cloned = frozen.clone();
        assert_eq!(frozen.arc_strong_count(), 2);
        assert_eq!(cloned.arc_strong_count(), 2);
        // Both observe the same table.
        assert_eq!(frozen.current_commitment(), cloned.current_commitment());
        // A commit on one is visible to the other.
        cloned.commit(table_with_fill::<8, 4, 2>(2.0));
        assert_eq!(frozen.current_version(), 1);
        assert_eq!(cloned.current_version(), 1);
    }

    // ── T4.3 — bit-identity test ─────────────────────────────────────────

    #[test]
    fn bit_identity_byte_identical_tables_match() {
        // Two byte-identical tables (same seed → deterministic construction)
        // must produce identical BLAKE3 commitments.
        let t_a: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(123);
        let t_b: ProductKeyMemory<16, 8, 4> = ProductKeyMemory::from_random(123);
        // Sanity: they really are byte-identical.
        assert_eq!(t_a.keys_1.as_ref(), t_b.keys_1.as_ref());
        assert_eq!(t_a.values.as_ref(), t_b.values.as_ref());

        let frozen = FrozenProductKeyMemory::new(t_a);
        let hash_a = frozen.current_commitment().unwrap();

        let hash_b = frozen.commit(t_b);
        // Commitment is deterministic over byte contents, not over version.
        assert_eq!(
            hash_a, hash_b,
            "byte-identical tables must produce identical commitments"
        );
    }

    #[test]
    fn bit_identity_one_bit_flip_differs() {
        // Build a table, clone its slices, flip ONE bit in one f32, construct
        // a new table, verify the commitments differ.
        let original: ProductKeyMemory<8, 4, 2> = ProductKeyMemory::from_random(7);
        let (mut k1, k2, v) = clone_table_slices(&original);

        // Flip the lowest bit of the first float in keys_1. Use `to_bits` /
        // `from_bits` to manipulate the IEEE-754 representation directly.
        let first = k1[0].to_bits();
        k1[0] = f32::from_bits(first ^ 1);

        let flipped = ProductKeyMemory::<8, 4, 2>::new(k1, k2, v);
        // Sanity: only one float changed, by one bit.
        assert_ne!(original.keys_1[0].to_bits(), flipped.keys_1[0].to_bits());
        assert_eq!(original.keys_1[1..], flipped.keys_1[1..]);

        let frozen = FrozenProductKeyMemory::new(original);
        let hash_original = frozen.current_commitment().unwrap();
        let hash_flipped = frozen.commit(flipped);
        assert_ne!(
            hash_original, hash_flipped,
            "a single-bit flip MUST change the BLAKE3 commitment"
        );
    }

    #[test]
    fn commitment_tag_distinguishes_from_raw_slice_hash() {
        // The domain tag means our commitment differs from a naive
        // BLAKE3(keys_1 || keys_2 || values) with no prefix. This guards
        // against cross-subsystem hash collisions.
        let table: ProductKeyMemory<8, 4, 2> = ProductKeyMemory::from_random(99);
        let our_hash = compute_commitment(&table);

        // Reference: hash the concatenation WITHOUT the tag.
        let mut naive = blake3::Hasher::new();
        naive.update(bytemuck::cast_slice::<f32, u8>(table.keys_1.as_ref()));
        naive.update(bytemuck::cast_slice::<f32, u8>(table.keys_2.as_ref()));
        naive.update(bytemuck::cast_slice::<f32, u8>(table.values.as_ref()));
        let naive_hash = *naive.finalize().as_bytes();

        assert_ne!(
            our_hash, naive_hash,
            "domain tag must distinguish PKM commitments from raw slice hashes"
        );
    }

    // ── T4.2 — concurrent no-torn-read stress test ───────────────────────
    //
    // Generalizes riir-engine's `concurrent_lora_no_torn_read` (Issue 354)
    // to the √N×√N PKM table. With `Arc<RwLock<Arc<ProductKeyMemory>>>`,
    // torn reads are impossible BY CONSTRUCTION — the reader either clones
    // the old Arc or the new Arc, never a half-swapped mix. This test is the
    // empirical complement to that invariant.

    #[test]
    fn concurrent_commit_read_no_torn_read() {
        // Small table (4×4 slots, D_K=4, D_V=2) to keep per-commit alloc
        // cost low — we want to stress the race window, not the allocator.
        const SQRT_N: usize = 4;
        const D_K: usize = 4;
        const D_V: usize = 2;
        const N_COMMITS: u32 = 100;
        const N_READS: u32 = 100_000;

        // Initial table: fill = 1.0 (version "1").
        let frozen = Arc::new(FrozenProductKeyMemory::<SQRT_N, D_K, D_V>::new(
            table_with_fill(1.0),
        ));

        let writer_frozen = Arc::clone(&frozen);
        let writer = thread::spawn(move || {
            for i in 2..=(N_COMMITS + 1) {
                // Each commit installs a table whose EVERY cell == i as f32.
                // A torn read {old keys_1, new values} would show mismatched
                // fills across the three slices.
                let fill = i as f32;
                writer_frozen.commit(table_with_fill(fill));
            }
        });

        let reader_frozen = Arc::clone(&frozen);
        let reader = thread::spawn(move || {
            let mut consistent_reads = 0u64;
            for _ in 0..N_READS {
                let snap = match reader_frozen.current() {
                    Some(s) => s,
                    None => continue, // shouldn't happen — slot starts non-empty
                };
                // The torn-read check: every cell in every slice must agree.
                // Writer filled all three slices with the same `fill` value
                // per commit, so a consistent snapshot has:
                //   keys_1[0] == keys_2[0] == values[0]
                // A torn read violates this (the Arc swap would have to be
                // non-atomic for it to happen — impossible with RwLock<Arc>).
                let k1 = snap.keys_1[0];
                let k2 = snap.keys_2[0];
                let v0 = snap.values[0];
                if k1 != k2 || k1 != v0 {
                    // Hard-fail on the first torn read — this is a correctness
                    // invariant, not a probabilistic check.
                    panic!(
                        "torn read detected: keys_1[0]={k1}, keys_2[0]={k2}, \
                         values[0]={v0} — all three must agree. The \
                         RwLock<Arc<...>> design should make this impossible."
                    );
                } else {
                    consistent_reads += 1;
                }
            }
            consistent_reads
        });

        writer.join().expect("writer panicked");
        let consistent_reads = reader.join().expect("reader panicked");

        // Sanity: the reader should have observed many consistent reads.
        assert!(
            consistent_reads > 0,
            "reader should have seen at least one consistent snapshot"
        );
        // Version should reflect all commits.
        assert_eq!(frozen.current_version(), N_COMMITS as u64);
    }

    #[test]
    fn concurrent_commit_read_version_monotonic() {
        // Companion to the above: the version counter observed by readers
        // must be monotonically non-decreasing. A torn read of the counter
        // would show it going backwards.
        const SQRT_N: usize = 4;
        const D_K: usize = 4;
        const D_V: usize = 2;
        const N_COMMITS: u32 = 50;
        const N_READS: u32 = 50_000;

        let frozen = Arc::new(FrozenProductKeyMemory::<SQRT_N, D_K, D_V>::new(
            table_with_fill(0.0),
        ));

        let writer_frozen = Arc::clone(&frozen);
        let writer = thread::spawn(move || {
            for i in 1..=N_COMMITS {
                writer_frozen.commit(table_with_fill(i as f32));
            }
        });

        let reader_frozen = Arc::clone(&frozen);
        let reader = thread::spawn(move || {
            let mut last_version = 0u64;
            for _ in 0..N_READS {
                let v = reader_frozen.current_version();
                assert!(
                    v >= last_version,
                    "version went backwards: {last_version} -> {v}"
                );
                last_version = v;
            }
            last_version
        });

        writer.join().expect("writer panicked");
        let last = reader.join().expect("reader panicked");
        // The reader may have finished its loop BEFORE the writer completed
        // all commits (the reader does 50K atomic loads; the writer does 50
        // BLAKE3-hashing commits). So `last` is the version at the reader's
        // final iteration — at most N_COMMITS, and strictly <= the final
        // state. The invariant we're testing is MONOTONICITY (asserted
        // in-loop above), not that the reader saw every commit.
        assert!(
            last <= N_COMMITS as u64,
            "reader's last version {last} should be <= {N_COMMITS} (the reader may finish before the writer)"
        );
        // After both threads join, all commits are applied.
        assert_eq!(frozen.current_version(), N_COMMITS as u64);
    }
}

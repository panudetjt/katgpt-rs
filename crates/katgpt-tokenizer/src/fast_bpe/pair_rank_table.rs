// SPDX-License-Identifier: MIT
//
// Vendored from gigatoken https://github.com/marcelroed/gigatoken (Marcel Rød).
// Upstream files: `src/bpe/mod.rs` (PairRankTable + merge cores).
// Upstream commit at vendor time: master 2026-07-25.
//
// Adaptations vs upstream:
// - Module path `crate::bpe::...` → `crate::fast_bpe::...` (this crate).
// - Removed `pretoken_cache` field references on `Tokenizer` (not vendored here).
// - Removed the SentencePiece ranked-merge variants (`ranked_merge_key`,
//   `bpe_merge_symbols_ranked`, `bpe_merge_symbols_ranked_slice`) — the katgpt
//   `BpeTokenizer` uses id-as-rank (tiktoken-style), not rank-mapped vocab.
// - Removed `simple_bpe_merge` (only used upstream by `from_ranks`, which we
//   don't expose).
// - Removed `vocab_entries` / `ranked_merge_key` / `MergeScratch::heap` typing
//   adjustments kept minimal.
//
// Algorithmic content is bit-identical to upstream — the
// `short_merges_match_vec_merge_loop` differential test in upstream
// `bpe/mod.rs` applies. We re-exercise it in `tests/fast_bpe_goat.rs::g1_*`.
//
// `allow(unused)` is intentional: `SHORT_MERGE_MAX` + the short-merge scalar/
// NEON cores are vendored substrate for the day someone adds pretokenization
// (Issue 191 Phase 1 §"What's NOT here"). They are bit-identical to upstream,
// kept here so the port is one import away when that lands. Silencing the
// warnings individually would make the upstream-sync diff noisy; the
// module-level allow is the clean choice.

#![allow(unused)]

use crate::fast_bpe::token::TokenId;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PairRankTable — flat pair-rank lookups for the merge hot path
// ---------------------------------------------------------------------------

/// Merge-pair lookup structure replacing the hashbrown `merges` map on the
/// cache-miss merge path. Two levels, both immutable after construction (so
/// forks share one copy via `Arc` instead of cloning the map):
///
/// - a dense `byte × byte` grid covering every pair whose sides are both
///   below the initial-symbol range (256 for GPT-2, 512 for DeepSeek-style
///   layouts). All round-1 lookups — roughly half of all lookups per miss,
///   since initial symbols are byte tokens — become one shift-or index and
///   one L1/L2 load: no hash, no probe.
/// - a flat open-addressed table of all merges: power-of-two slots at
///   ≤ 1/2 load, linear probing, one `u64` slot packing key and value
///   (`(key << 21) | merged_id`; keys are 42 bits and merged IDs 21, so a
///   packed slot uses 63 bits and `u64::MAX` can never collide) so a probe
///   touches one 8-byte load per step — the hit's value rides on the same
///   line as its key.
///
/// Values are merged token IDs (`u32::MAX` = no merge), matching the
/// tiktoken-style convention that merged ID order == merge priority.
#[derive(Debug)]
pub struct PairRankTable {
    /// `dense[(a << dense_log2) | b]` = merged ID for pairs with both sides
    /// `< 2^dense_log2`, `u32::MAX` when they do not merge.
    dense: Box<[u32]>,
    dense_log2: u32,
    /// Flat table slots, `((a << 21 | b) << 21) | merged_id` packed;
    /// `u64::MAX` = empty (real slots fit 63 bits, so the sentinel can
    /// never collide, and its key field `MAX >> 21` exceeds every real
    /// 42-bit key, so the hit compare rejects it for free).
    slots: Box<[u64]>,
    /// `slots.len() - 1` (length is a power of two).
    mask: usize,
    /// `64 - log2(slots.len())`: the hash keeps the top bits.
    shift: u32,
}

/// Every token ID must fit a 21-bit key lane (covers any vocab < 2M IDs).
const PAIR_ID_BITS: u32 = 21;

/// Why a `PairRankTable::build` call refused to construct a table — the
/// caller then falls back to the `HashMap` probe path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairRankTableBuildError {
    /// Vocab size exceeds the 21-bit packed-key lane (`vocab_len > 2²¹`).
    VocabTooLarge,
    /// At least one merge's `(left, right, merged)` triple has an ID that
    /// exceeds the 21-bit packed-key lane.
    MergeIdTooLarge,
    /// Pathological clustering in the flat table (>64 displacement on
    /// insert). Never seen on real vocabs at ≤1/2 load.
    Clustering,
}

impl PairRankTable {
    /// Build the two-level table from a `(left, right) → merged` map.
    ///
    /// Returns `Err(_)` when this vocabulary cannot use the table (IDs too
    /// large for the packed key, or pathological clustering in the flat
    /// table) — the caller then falls back to the `HashMap` probe.
    pub fn build(
        merges: &HashMap<(TokenId, TokenId), TokenId>,
        vocab_len: usize,
    ) -> Result<Self, PairRankTableBuildError> {
        if vocab_len > 1 << PAIR_ID_BITS {
            return Err(PairRankTableBuildError::VocabTooLarge);
        }
        let id_limit = 1u32 << PAIR_ID_BITS;
        if merges
            .iter()
            .any(|(&(a, b), &m)| a.0 >= id_limit || b.0 >= id_limit || m.0 >= id_limit)
        {
            return Err(PairRankTableBuildError::MergeIdTooLarge);
        }

        // Dense level covering every pair with both sides < 2^11 — the
        // initial byte tokens (0..255 for GPT-2, 3..258 for DeepSeek) AND
        // the earliest ~1.8k merged IDs, which by merge order are the most
        // frequent symbols in the merge loop. Round-1 lookups plus the bulk
        // of mid-merge lookups become one shift-or index and one load into
        // a 16 MiB, L3-resident grid.
        //
        // NOTE: upstream also accepts a `byte_remapping: Option<&ByteRemapping>`
        // that lets a vocab with byte tokens above the grid (e.g. DeepSeek's
        // bytes at 3..258) keep correctness with round-1 lookups falling
        // through to the flat table. The katgpt `BpeTokenizer` has no byte
        // remapping (it uses character-string vocab IDs), so the parameter
        // is dropped here. If a future caller needs it, port upstream's
        // `ByteRemapping` first.
        let dense_log2 = 11u32;
        let mut dense = vec![u32::MAX; 1usize << (2 * dense_log2)].into_boxed_slice();

        // Flat level holds all merges (including the dense subset, so it is
        // a complete map on its own) at ≤ 1/2 load.
        let n_slots = (merges.len().max(1) * 2).next_power_of_two().max(64);
        let shift = 64 - n_slots.trailing_zeros();
        let mask = n_slots - 1;
        let mut slots = vec![u64::MAX; n_slots].into_boxed_slice();
        for (&(a, b), &m) in merges {
            if (a.0 | b.0) >> dense_log2 == 0 {
                dense[((a.0 as usize) << dense_log2) | b.0 as usize] = m.0;
            }
            let key = ((a.0 as u64) << PAIR_ID_BITS) | b.0 as u64;
            let mut idx = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift) as usize;
            let mut displacement = 0usize;
            while slots[idx] != u64::MAX {
                idx = (idx + 1) & mask;
                displacement += 1;
                // Ugly clustering (never seen on real vocabs at this load
                // factor): give up rather than degrade every miss lookup.
                if displacement > 64 {
                    return Err(PairRankTableBuildError::Clustering);
                }
            }
            // Merged IDs are < 2^21 (checked above), so key and value pack
            // into 63 bits without overlap.
            slots[idx] = (key << PAIR_ID_BITS) | m.0 as u64;
        }

        Ok(Self {
            dense,
            dense_log2,
            slots,
            mask,
            shift,
        })
    }

    /// Merged token ID of the pair `(a, b)`, or `u32::MAX` when it does
    /// not merge — the same convention as probing the merges map.
    #[inline(always)]
    pub fn rank(&self, a: TokenId, b: TokenId) -> u32 {
        if (a.0 | b.0) >> self.dense_log2 == 0 {
            let idx = ((a.0 as usize) << self.dense_log2) | b.0 as usize;
            // SAFETY: both IDs < 2^dense_log2, so idx < 2^(2*dense_log2)
            // == dense.len().
            return unsafe { *self.dense.get_unchecked(idx) };
        }
        let key = ((a.0 as u64) << PAIR_ID_BITS) | b.0 as u64;
        let mut idx = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize;
        loop {
            // SAFETY: idx starts < slots.len() (shift keeps log2(len) bits)
            // and stays masked.
            let slot = unsafe { *self.slots.get_unchecked(idx) };
            // One load answers both hit and value: the key rides in the
            // slot's top 43 bits, the merged ID in the low 21. The empty
            // sentinel's key field (u64::MAX >> 21) exceeds every real
            // 42-bit key, so it never false-hits.
            if slot >> PAIR_ID_BITS == key {
                return (slot & ((1 << PAIR_ID_BITS) - 1)) as u32;
            }
            if slot == u64::MAX {
                return u32::MAX;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// `prefetcht0` the first line [`Self::rank`] would load for `(a, b)`:
    /// the dense cell when both sides sit in the grid, else the first
    /// sparse probe slot. The short-merge loop's refresh lookups sit on a
    /// serial dependency chain and their slot-load consumers are ~5-7% of
    /// cold cycles on Zen 5 (upstream `profiling/zen5_st_profile.md` §4.3);
    /// both refresh pairs are known before the list surgery, so issuing
    /// their prefetches first overlaps the load latency with the surgery
    /// and with each other. Address math mirrors `rank` exactly (same
    /// bounds: dense idx < dense.len(), slot idx < slots.len() via
    /// `shift`), so every prefetched address lies within the table's
    /// allocations.
    /// x86_64 only; a no-op elsewhere (aarch64 uses the NEON merge core).
    #[inline(always)]
    pub fn prefetch_rank(&self, a: TokenId, b: TokenId) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `_mm_prefetch` does not dereference; both index
        // computations are the bounded ones from `rank` (see its SAFETY
        // comments), so the address stays inside the live allocation.
        unsafe {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let addr = if (a.0 | b.0) >> self.dense_log2 == 0 {
                let idx = ((a.0 as usize) << self.dense_log2) | b.0 as usize;
                self.dense.as_ptr().add(idx) as *const i8
            } else {
                let key = ((a.0 as u64) << PAIR_ID_BITS) | b.0 as u64;
                let idx = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize;
                self.slots.as_ptr().add(idx) as *const i8
            };
            _mm_prefetch(addr, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (a, b);
    }
}

// ---------------------------------------------------------------------------
// Shared BPE merge functions
// ---------------------------------------------------------------------------

/// Reusable scratch buffers for [`bpe_merge_symbols_with_scratch`]. Holding one
/// of these across calls means the hot encode loop performs no per-pretoken
/// allocations for the linked list or the merge heap.
#[derive(Default)]
pub struct MergeScratch {
    next: Vec<u32>,
    prev: Vec<u32>,
    heap: Vec<std::cmp::Reverse<u64>>,
}

/// Pack a heap entry as `(merged_token << 32) | position`. `u64` ordering then
/// matches the old `(TokenId, usize)` tuple ordering (token ID first, position
/// as tie-break) while halving the element size the heap has to shuffle.
#[inline(always)]
fn pack_merge_entry(merged: TokenId, pos: u32) -> u64 {
    ((merged.0 as u64) << 32) | pos as u64
}

/// Apply BPE merges to an already-initialized symbol sequence, using a
/// `PairRankTable` for the pair-rank lookup.
///
/// Priority is determined by the merged token's ID (lower = first).
/// This is correct for tiktoken-style tokenizers where vocab ID equals merge rank.
///
/// Uses a min-heap + doubly-linked list for O(n log n) performance on long
/// sequences, dispatching to the linear-scan small-merge core for sequences
/// of ≤ `SMALL_MERGE_MAX` symbols.
pub fn bpe_merge_symbols_by_rank(
    table: &PairRankTable,
    symbols: &mut Vec<TokenId>,
    scratch: &mut MergeScratch,
) {
    bpe_merge_symbols_by_rank_inner(&|a, b| table.rank(a, b), symbols, scratch);
}

/// Same shape as [`bpe_merge_symbols_by_rank`] but with a custom pair-rank
/// lookup (kept for parity with upstream + for tests that need to drive the
/// loop against a `HashMap`).
pub fn bpe_merge_symbols_by_rank_with_lookup(
    get_rank: &impl Fn(TokenId, TokenId) -> u32,
    symbols: &mut Vec<TokenId>,
    scratch: &mut MergeScratch,
) {
    bpe_merge_symbols_by_rank_inner(get_rank, symbols, scratch);
}

// Out of line: this only runs on cache-miss paths, and inlining its bulk into
// the encode loop costs more in I-cache and register pressure there than a
// call costs here.
#[inline(never)]
fn bpe_merge_symbols_by_rank_inner(
    get_rank: &impl Fn(TokenId, TokenId) -> u32,
    symbols: &mut Vec<TokenId>,
    scratch: &mut MergeScratch,
) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = symbols.len();
    if n < 2 {
        return;
    }

    // For short sequences (the overwhelming majority of pretokens), a linear
    // rank scan has far lower constants than heap + linked list.
    if n <= SMALL_MERGE_MAX {
        bpe_merge_symbols_small(get_rank, symbols);
        return;
    }

    // Doubly-linked list via u32 index arrays (sequences are far shorter than
    // 2^32 symbols).
    const NONE: u32 = u32::MAX;
    let next = &mut scratch.next;
    let prev = &mut scratch.prev;
    next.clear();
    next.extend(1..n as u32);
    next.push(NONE);
    prev.clear();
    prev.push(NONE);
    prev.extend(0..n as u32 - 1);

    // Min-heap of packed (merged_token_id, position). Lower ID = higher
    // priority. Seed with all initial pairs, then heapify in O(n) instead of
    // pushing one at a time.
    let mut seeds = std::mem::take(&mut scratch.heap);
    seeds.clear();
    for i in 0..n - 1 {
        let m = get_rank(symbols[i], symbols[i + 1]);
        if m != u32::MAX {
            seeds.push(Reverse(pack_merge_entry(TokenId(m), i as u32)));
        }
    }
    let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::from(seeds);

    while let Some(Reverse(entry)) = heap.pop() {
        let pos = (entry & u32::MAX as u64) as usize;
        let expected_merged = (entry >> 32) as u32;
        // Validate: pos must still be active and its right neighbor must exist
        let right = next[pos];
        if right == NONE {
            continue;
        }
        let right = right as usize;
        // Check the pair still matches (it may have been invalidated by an
        // earlier merge). A no-merge result (u32::MAX) never equals the
        // expected merged ID, since only real merges are pushed.
        let merged = get_rank(symbols[pos], symbols[right]);
        if merged == expected_merged {
            // Apply the merge
            symbols[pos] = TokenId(merged);
            let right_right = next[right];
            next[pos] = right_right;
            if right_right != NONE {
                prev[right_right as usize] = pos as u32;
            }
            // `next[right] = NONE` invalidates any stale heap entries at
            // `right`; nothing reads `prev[right]` once it is unlinked.
            next[right] = NONE;

            // Re-check pair with left neighbor
            let left = prev[pos];
            if left != NONE {
                let m = get_rank(symbols[left as usize], symbols[pos]);
                if m != u32::MAX {
                    heap.push(Reverse(pack_merge_entry(TokenId(m), left)));
                }
            }
            // Re-check pair with new right neighbor
            if next[pos] != NONE {
                let m = get_rank(symbols[pos], symbols[next[pos] as usize]);
                if m != u32::MAX {
                    heap.push(Reverse(pack_merge_entry(TokenId(m), pos as u32)));
                }
            }
        }
    }
    // The pop loop drained the heap; keep its capacity for the next call.
    scratch.heap = heap.into_vec();

    // Compact surviving symbols in place. Surviving indices are strictly
    // increasing and the k-th survivor has index >= k, so writes never
    // overtake reads.
    let mut write = 0;
    let mut i = 0;
    loop {
        symbols[write] = symbols[i];
        write += 1;
        if next[i] == NONE {
            break;
        }
        i = next[i] as usize;
    }
    symbols.truncate(write);
}

/// Sequences up to this length use the linear-scan merge instead of the heap.
pub const SMALL_MERGE_MAX: usize = 32;

/// BPE merge for short sequences (n <= SMALL_MERGE_MAX), in the style of
/// tiktoken's `byte_pair_merge`: keep a per-position rank (the merged token ID
/// of the pair starting there, or `u32::MAX`), find the minimum by linear scan,
/// merge, and recompute only the two affected neighbor ranks. The scan over a
/// few stack-resident `u32`s beats a `BinaryHeap`'s sift traffic at these
/// sizes, and merge priority (lowest merged ID, then lowest position) is
/// identical to the heap version's ordering.
fn bpe_merge_symbols_small(
    get_rank: &impl Fn(TokenId, TokenId) -> u32,
    symbols: &mut Vec<TokenId>,
) {
    let n = symbols.len();
    debug_assert!((2..=SMALL_MERGE_MAX).contains(&n));
    // Stack-resident doubly-linked list, so a merge is O(1) pointer updates
    // instead of shifting the tail (which lowers to a libc memmove call).
    // Sentinels: next[last] == n, prev[0] == u8::MAX; both fail `< n` checks.
    let mut next = [0u8; SMALL_MERGE_MAX];
    let mut prev = [0u8; SMALL_MERGE_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    // ranks[i] = priority of merging the pair starting at active position i;
    // MAX when there is no merge or the position was merged away.
    let mut ranks = [u32::MAX; SMALL_MERGE_MAX];
    for i in 0..n - 1 {
        ranks[i] = get_rank(symbols[i], symbols[i + 1]);
    }
    loop {
        let mut best = u32::MAX;
        let mut best_i = 0;
        for (i, &rank) in ranks[..n - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                best_i = i;
            }
        }
        if best == u32::MAX {
            break;
        }
        let i = best_i;
        symbols[i] = TokenId::from(best);
        // Unlink the right element of the merged pair.
        let dead = next[i] as usize;
        let new_right = next[dead] as usize;
        next[i] = new_right as u8;
        ranks[dead] = u32::MAX;
        // Refresh the two pairs now touching the merged symbol.
        if new_right < n {
            prev[new_right] = i as u8;
            ranks[i] = get_rank(symbols[i], symbols[new_right]);
        } else {
            ranks[i] = u32::MAX;
        }
        let left = prev[i] as usize;
        if left < n {
            ranks[left] = get_rank(symbols[left], symbols[i]);
        }
    }
    // Compact survivors in place: list indices are strictly increasing, so
    // writes never overtake reads.
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = next[i] as usize;
    }
    symbols.truncate(write);
}

/// Symbol capacity of the stack-array short merges: short pretokens are
/// ≤ 15 bytes, so at most 15 initial symbols.
pub const SHORT_MERGE_MAX: usize = 16;

/// [`bpe_merge_symbols_small`] over a caller-owned stack array instead of
/// the `Vec` scratch (no bounds/capacity code, no extend memmove) — the
/// short-pretoken miss path's merge. Returns the merged length.
///
/// The non-aarch64 core of the short-miss domain (<= 15 symbols); aarch64 uses
/// [`bpe_merge_symbols_short_neon`] when a [`PairRankTable`] is available.
/// Kept separate from the Vec-based [`bpe_merge_symbols_small`] (16..=32
/// symbols): the domains are disjoint and unification was measured as a
/// regression risk to this tuned miss path.
///
/// `prefetch_rank` is [`PairRankTable::prefetch_rank`] when a table backs
/// `get_rank`, and a no-op closure for the HashMap fallback — the merge
/// scan itself stays identical either way (restructuring it was measured
/// negative; the prefetch only reorders when the refresh loads issue).
pub fn bpe_merge_symbols_short_scalar(
    get_rank: impl Fn(TokenId, TokenId) -> u32,
    prefetch_rank: impl Fn(TokenId, TokenId),
    symbols: &mut [TokenId; SHORT_MERGE_MAX],
    n: usize,
) -> usize {
    debug_assert!((2..=SHORT_MERGE_MAX - 1).contains(&n));
    // Stack-resident doubly-linked list; see `bpe_merge_symbols_small`.
    let mut next = [0u8; SHORT_MERGE_MAX];
    let mut prev = [0u8; SHORT_MERGE_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    // ranks[i] = priority of merging the pair starting at active position i;
    // MAX when there is no merge or the position was merged away.
    let mut ranks = [u32::MAX; SHORT_MERGE_MAX];
    for i in 0..n - 1 {
        ranks[i] = get_rank(symbols[i], symbols[i + 1]);
    }
    loop {
        let mut best = u32::MAX;
        let mut best_i = 0;
        for (i, &rank) in ranks[..n - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                best_i = i;
            }
        }
        if best == u32::MAX {
            break;
        }
        let i = best_i;
        // Both refresh pairs are fixed the moment `best` is chosen:
        // (best, symbols[new_right]) and (symbols[left], best). Their rank
        // lines are the serial chain's next loads, so request both before
        // the list surgery. List indices are strictly increasing, so
        // new_right > i and the surgery's `prev[new_right]` store cannot
        // alias `prev[i]` — reading `left` early is safe.
        let dead = next[i] as usize;
        let new_right = next[dead] as usize;
        let left = prev[i] as usize;
        if new_right < n {
            prefetch_rank(TokenId(best), symbols[new_right]);
        }
        if left < n {
            prefetch_rank(symbols[left], TokenId(best));
        }
        symbols[i] = TokenId(best);
        // Unlink the right element of the merged pair.
        next[i] = new_right as u8;
        ranks[dead] = u32::MAX;
        // Refresh the two pairs now touching the merged symbol.
        if new_right < n {
            prev[new_right] = i as u8;
            ranks[i] = get_rank(symbols[i], symbols[new_right]);
        } else {
            ranks[i] = u32::MAX;
        }
        if left < n {
            ranks[left] = get_rank(symbols[left], symbols[i]);
        }
    }
    // Compact survivors in place: list indices are strictly increasing, so
    // writes never overtake reads.
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = next[i] as usize;
    }
    write
}

/// [`bpe_merge_symbols_short_scalar`] with a branchless NEON min-rank scan.
/// Rank and position share one lane, `pr[i] = (rank << 8) | i`, so the
/// vector minimum selects the lowest rank with ties broken by the lowest
/// position — exactly the scalar scan's order. Real ranks are < 2^21 (a
/// [`PairRankTable`] build invariant), so packed lanes are < 2^29 and "no
/// merge" (`u32::MAX << 8`) sorts above every real lane. The scan covers
/// 16 lanes with inactive lanes parked at `u32::MAX` — four loads, three
/// `vmin`s, one `vminv`, zero data-dependent branches on a core where the
/// scalar scan's `rank < best` mispredicts freely.
#[cfg(target_arch = "aarch64")]
pub fn bpe_merge_symbols_short_neon(
    table: &PairRankTable,
    symbols: &mut [TokenId; SHORT_MERGE_MAX],
    n: usize,
) -> usize {
    use core::arch::aarch64::{vld1q_u32, vminq_u32, vminvq_u32};
    /// Every packed value at or above this has rank u32::MAX (no merge).
    const NO_MERGE_FLOOR: u32 = u32::MAX << 8;

    debug_assert!((2..=SHORT_MERGE_MAX - 1).contains(&n));
    let pack = |rank: u32, i: usize| (rank << 8) | i as u32;
    // Stack-resident doubly-linked list; see `bpe_merge_symbols_small`.
    let mut next = [0u8; SHORT_MERGE_MAX];
    let mut prev = [0u8; SHORT_MERGE_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    // pr[i] = packed (rank, position) of the pair starting at active
    // position i; the only array the scan reads.
    let mut pr = [u32::MAX; SHORT_MERGE_MAX];
    for i in 0..n - 1 {
        pr[i] = pack(table.rank(symbols[i], symbols[i + 1]), i);
    }
    // Lanes at index >= n stay u32::MAX forever (initial pairs sit below
    // n - 1; the loop below only writes lanes < n), so short pretokens
    // need only the first half of the scan.
    let narrow = n <= 8;
    loop {
        // SAFETY: pr is 16 contiguous u32s; vld1q_u32 has no alignment
        // requirement beyond u32's.
        let best = unsafe {
            let p = pr.as_ptr();
            let m01 = vminq_u32(vld1q_u32(p), vld1q_u32(p.add(4)));
            let m = if narrow {
                m01
            } else {
                let m23 = vminq_u32(vld1q_u32(p.add(8)), vld1q_u32(p.add(12)));
                vminq_u32(m01, m23)
            };
            vminvq_u32(m)
        };
        if best >= NO_MERGE_FLOOR {
            break;
        }
        let i = (best & 0xFF) as usize;
        symbols[i] = TokenId(best >> 8);
        // Unlink the right element of the merged pair.
        let dead = next[i] as usize;
        let new_right = next[dead] as usize;
        next[i] = new_right as u8;
        pr[dead] = u32::MAX;
        // Refresh the two pairs now touching the merged symbol.
        if new_right < n {
            prev[new_right] = i as u8;
            pr[i] = pack(table.rank(symbols[i], symbols[new_right]), i);
        } else {
            pr[i] = u32::MAX;
        }
        let left = prev[i] as usize;
        if left < n {
            pr[left] = pack(table.rank(symbols[left], symbols[i]), left);
        }
    }
    // Compact survivors in place.
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = next[i] as usize;
    }
    write
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal deterministic RNG (xorshift64*) so the differential tests
    /// need no dev-dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn random_merges(
        rng: &mut Rng,
        n_merges: usize,
        id_range: u32,
    ) -> HashMap<(TokenId, TokenId), TokenId> {
        let mut merges = HashMap::new();
        for i in 0..n_merges {
            let a = TokenId(rng.below(id_range as u64) as u32);
            let b = TokenId(rng.below(id_range as u64) as u32);
            merges.entry((a, b)).or_insert(TokenId(256 + i as u32));
        }
        merges
    }

    /// The flat table must agree with the hashbrown map on every merge pair
    /// and return the no-merge sentinel everywhere else.
    #[test]
    fn pair_rank_table_matches_map() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        let id_range = 4096u32;
        let merges = random_merges(&mut rng, 3000, id_range);
        let table = PairRankTable::build(&merges, id_range as usize).expect("build");
        for (&(a, b), &m) in &merges {
            assert_eq!(table.rank(a, b), m.0, "pair ({}, {})", a.0, b.0);
        }
        for _ in 0..100_000 {
            let a = TokenId(rng.below(id_range as u64) as u32);
            let b = TokenId(rng.below(id_range as u64) as u32);
            let expected = merges.get(&(a, b)).map_or(u32::MAX, |m| m.0);
            assert_eq!(table.rank(a, b), expected, "pair ({}, {})", a.0, b.0);
        }
        // Oversized IDs must refuse the table, not corrupt it.
        assert_eq!(
            PairRankTable::build(&merges, (1 << PAIR_ID_BITS) + 1).unwrap_err(),
            PairRankTableBuildError::VocabTooLarge,
        );
        let mut big = merges.clone();
        big.insert((TokenId(1 << PAIR_ID_BITS), TokenId(0)), TokenId(300));
        assert_eq!(
            PairRankTable::build(&big, id_range as usize).unwrap_err(),
            PairRankTableBuildError::MergeIdTooLarge,
        );
    }

    /// The stack-array short merges (scalar and NEON) must produce exactly
    /// the sequence of the Vec-based merge loop — same merges, same order,
    /// same tie-breaks — across random symbol sequences and merge tables.
    #[test]
    fn short_merges_match_vec_merge_loop() {
        let mut rng = Rng(0xDEAD_BEEF_0BAD_F00D);
        for trial in 0..2000 {
            let id_range = 300 + (trial % 7) as u32 * 500;
            let merges = random_merges(&mut rng, 200 + trial % 800, id_range);
            let table = PairRankTable::build(&merges, id_range as usize + 1024).expect("build");
            let n = 2 + rng.below(14) as usize; // 2..=15
            let init: Vec<TokenId> = (0..n)
                .map(|_| TokenId(rng.below(id_range as u64) as u32))
                .collect();

            let mut reference = init.clone();
            bpe_merge_symbols_by_rank_with_lookup(
                &|a, b| merges.get(&(a, b)).map_or(u32::MAX, |m| m.0),
                &mut reference,
                &mut MergeScratch::default(),
            );

            let mut scalar = [TokenId(0); SHORT_MERGE_MAX];
            scalar[..n].copy_from_slice(&init);
            let len = bpe_merge_symbols_short_scalar(
                |a, b| table.rank(a, b),
                |a, b| table.prefetch_rank(a, b),
                &mut scalar,
                n,
            );
            assert_eq!(&scalar[..len], &reference[..], "scalar diverged: {init:?}");

            #[cfg(target_arch = "aarch64")]
            {
                let mut neon = [TokenId(0); SHORT_MERGE_MAX];
                neon[..n].copy_from_slice(&init);
                let len = bpe_merge_symbols_short_neon(&table, &mut neon, n);
                assert_eq!(&neon[..len], &reference[..], "neon diverged: {init:?}");
            }
        }
    }
}

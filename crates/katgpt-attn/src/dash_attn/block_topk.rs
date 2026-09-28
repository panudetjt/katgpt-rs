//! BlockTopK router — simplest VortexFlow implementation.
//!
//! Routes via mean key centroids + dot-product top-k selection.
//! Feature gate: `vortex_flow` (Plan 196, Phase 1).

#![allow(clippy::needless_range_loop)]

use katgpt_core::simd::simd_argmax_f32;

use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};

// ---------------------------------------------------------------------------
// BlockTopKCache
// ---------------------------------------------------------------------------

/// Cache for BlockTopK router: mean key centroid per block.
#[derive(Debug, Clone)]
pub struct BlockTopKCache {
    /// Centroid vectors: flat `[n_blocks * head_dim]`.
    pub centroids: Vec<f32>,
    /// Number of blocks currently cached.
    pub n_blocks: usize,
    /// Dimension of each centroid.
    pub head_dim: usize,
}

impl BlockTopKCache {
    /// Create a pre-allocated cache.
    pub fn new(n_blocks_capacity: usize, head_dim: usize) -> Self {
        Self {
            centroids: vec![0.0; n_blocks_capacity * head_dim],
            n_blocks: 0,
            head_dim,
        }
    }

    /// Ensure the cache can hold at least `n` blocks.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.centroids.len() < n * self.head_dim {
            self.centroids.resize(n * self.head_dim, 0.0);
        }
    }

    /// Get centroid for block `i`.
    #[inline]
    pub fn centroid(&self, i: usize) -> &[f32] {
        let start = i * self.head_dim;
        &self.centroids[start..start + self.head_dim]
    }
}

// ---------------------------------------------------------------------------
// BlockTopKRouter
// ---------------------------------------------------------------------------

/// BlockTopK router — routes via mean key centroids + dot product top-k.
#[derive(Debug)]
pub struct BlockTopKRouter {
    /// Apply `1/sqrt(head_dim)` scaling to dot products.
    pub scale: bool,
}

impl BlockTopKRouter {
    /// Create a new router with optional scaling.
    pub fn new(scale: bool) -> Self {
        Self { scale }
    }
}

impl VortexFlow for BlockTopKRouter {
    type Cache = BlockTopKCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        _values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        cache.ensure_capacity(block_idx + 1);
        let block_size = keys.len() / head_dim;
        if block_size == 0 {
            // Zero-length block: zero centroid
            let start = block_idx * head_dim;
            cache.centroids[start..start + head_dim].fill(0.0);
            cache.n_blocks = cache.n_blocks.max(block_idx + 1);
            return;
        }

        // Compute mean of keys → centroid (SIMD accumulate + scale)
        let start = block_idx * head_dim;
        let centroid = &mut cache.centroids[start..start + head_dim];
        centroid.fill(0.0);
        for t in 0..block_size {
            let k_start = t * head_dim;
            katgpt_core::simd::simd_add_inplace(centroid, &keys[k_start..k_start + head_dim]);
        }
        let inv = 1.0 / block_size as f32;
        katgpt_core::simd::simd_scale_inplace(centroid, inv);

        cache.n_blocks = cache.n_blocks.max(block_idx + 1);
    }

    fn forward_indexer(
        &self,
        query: &[f32],
        cache: &Self::Cache,
        n_blocks: usize,
        top_k: usize,
        scratch: &mut VortexScratch,
    ) -> RoutingDecision {
        if n_blocks == 0 {
            return RoutingDecision::new();
        }

        let hd = query.len();
        let scale = if self.scale {
            1.0 / (hd as f32).sqrt()
        } else {
            1.0
        };

        scratch.ensure_capacity(n_blocks);
        let scores = &mut scratch.scores[..n_blocks];

        // Compute dot(query, centroid[i]) for each block
        // Chunked accumulation for auto-vectorization
        for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
            let centroid = cache.centroid(i);
            let mut dot0 = 0.0f32;
            let mut dot1 = 0.0f32;
            let mut dot2 = 0.0f32;
            let mut dot3 = 0.0f32;
            // `as_chunks::<4>()` over pre-sliced rows: each yielded `&[f32; 4]`
            // has a statically known length, so the per-iteration bounds checks
            // collapse. The four lane accumulators receive their addends in
            // exactly the same order as the index form and the tail covers the
            // same `main..hd` range → bit-identical.
            let main = (hd / 4) * 4;
            for (q_c, c_c) in query[..main]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(centroid[..main].as_chunks::<4>().0.iter())
            {
                dot0 += q_c[0] * c_c[0];
                dot1 += q_c[1] * c_c[1];
                dot2 += q_c[2] * c_c[2];
                dot3 += q_c[3] * c_c[3];
            }
            let mut dot = dot0 + dot1 + dot2 + dot3;
            for (&q_d, &c_d) in query[main..hd].iter().zip(centroid[main..hd].iter()) {
                dot += q_d * c_d;
            }
            *score = dot * scale;
        }

        // Partial sort to find top-k (reuses scratch buffer)
        let k = top_k.min(n_blocks);
        scratch.indices.clear();
        argtopk_with_scratch(scores, k, &mut scratch.indices, &mut scratch.argtopk_pairs);

        // Build routing decision with sigmoid weights
        let mut decision = RoutingDecision::with_capacity(k);
        for &idx in &scratch.indices[..k] {
            decision.blocks.push(idx);
            // Sigmoid normalization per-block score
            let w = sigmoid(scores[idx]);
            decision.weights.push(w);
        }

        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        BlockTopKCache::new(n_blocks_capacity, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find top-k indices from scores (partial sort, descending).
/// Uses a simple selection-based approach — O(n*k) which is fine for small k.
///
/// `Vec::new()` here starts at zero capacity (no heap allocation); it only
/// grows if the k>16 selection-sort fallback runs. The common k≤16 SIMD path
/// never touches it. Prefer [`argtopk_with_scratch`] in callers that can
/// reuse a buffer across calls to avoid the (rare) fallback allocation.
pub fn argtopk(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    argtopk_with_scratch(scores, k, indices, &mut Vec::new());
}

/// Zero-alloc variant of [`argtopk`] — reuses `pairs` scratch buffer.
///
/// Uses SIMD-optimized path for k ≤ 16 (Plan 256 Phase 1):
/// - k=1: single-pass `simd_argmax_f32`
/// - k≤16: register-based min-heap with SIMD-parallel comparison
/// - k>16: falls back to selection sort
///
/// ⚠ The k ≤ 16 arm is **per-architecture**. NEON takes it whole; on x86_64 the
/// AVX2 kernel is dispatched only up to `AVX2_ARGTOPK_K_MAX` and k above that
/// takes [`argtopk_scalar_heap`], which is measurably faster there (Issue 808
/// T1 / Bench 810). The *result* is identical on every path — this is a
/// performance dispatch, and no caller can observe which kernel ran.
///
/// (`AVX2_ARGTOPK_K_MAX` is deliberately NOT an intra-doc link here: the const
/// is `cfg(target_arch = "x86_64")` and this item is not, so a link would be a
/// broken-intra-doc-link warning on aarch64 and wasm32 — two platforms whose
/// lanes this repo runs and whose docs nobody would read the warning on.)
pub fn argtopk_with_scratch(
    scores: &[f32],
    k: usize,
    indices: &mut Vec<usize>,
    pairs: &mut Vec<(usize, f32)>,
) {
    indices.clear();
    let n = scores.len();
    let k = k.min(n);
    if k == 0 {
        return;
    }

    if k == 1 {
        // Fast path: single argmax via SIMD
        let (idx, _) = simd_argmax_f32(scores);
        indices.push(idx);
        return;
    }

    if k <= 16 {
        // SIMD-optimized register top-k for small k (Plan 256)
        argtopk_simd(scores, k, indices);
        return;
    }

    // Fallback: selection sort O(n*k) for large k
    pairs.clear();
    pairs.extend((0..n).map(|i| (i, scores[i])));

    for i in 0..k {
        let mut best = i;
        for j in (i + 1)..n {
            if pairs[j].1 > pairs[best].1 {
                best = j;
            }
        }
        pairs.swap(i, best);
    }

    indices.extend(pairs[..k].iter().map(|(idx, _)| *idx));
}

/// Insert (val, idx) into sorted register (descending). k ≤ 16.
/// Binary search + shift: O(log k + k) ≈ O(k) for small k.
#[inline(always)]
#[allow(dead_code)] // Used by argtopk_simd (NEON) and argtopk_scalar_heap (non-NEON)
fn insert_sorted(
    heap_vals: &mut [f32; 16],
    heap_idxs: &mut [usize; 16],
    k: usize,
    val: f32,
    idx: usize,
) {
    if val <= heap_vals[k - 1] {
        return;
    }
    // Binary search for insertion position (first element < val)
    let mut lo = 0usize;
    let mut hi = k;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if heap_vals[mid] < val {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    // Shift elements down to make room at position `lo`
    for j in (lo + 1..k).rev() {
        heap_vals[j] = heap_vals[j - 1];
        heap_idxs[j] = heap_idxs[j - 1];
    }
    heap_vals[lo] = val;
    heap_idxs[lo] = idx;
}

/// SIMD-accelerated top-k for k ≤ 16 on NEON.
///
/// Strategy: maintain top-k as a sorted register (descending).
/// Phase 1 optimization (Plan 256): batch NEON threshold comparison +
/// lane-indexed insertion. NEON loads 4 scores, compares against threshold
/// via vcgtq_f32, then only inserts lanes that exceed threshold.
/// The inner insertion uses NEON-parallel comparison across the entire
/// top-k register (4 lanes at a time) to find the insertion point,
/// then scalar shift + write for the final placement.
#[cfg(target_arch = "aarch64")]
fn argtopk_simd(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    use core::arch::aarch64::{vcgtq_f32, vdupq_n_f32, vld1q_f32};
    unsafe {
        let n = scores.len();
        let mut heap_vals = [f32::NEG_INFINITY; 16];
        let mut heap_idxs = [0usize; 16];

        let init = k.min(n);
        for i in 0..init {
            heap_vals[i] = *scores.get_unchecked(i);
            heap_idxs[i] = i;
        }
        // Sort initial k elements descending (insertion sort, k≤16)
        for i in 1..init {
            let val = heap_vals[i];
            let idx = heap_idxs[i];
            let mut j = i;
            while j > 0 && heap_vals[j - 1] < val {
                heap_vals[j] = heap_vals[j - 1];
                heap_idxs[j] = heap_idxs[j - 1];
                j -= 1;
            }
            heap_vals[j] = val;
            heap_idxs[j] = idx;
        }

        // Process remaining elements in NEON chunks of 4.
        // Phase 1: NEON threshold filter — skip entire chunk if no lane exceeds.
        // Phase 2: per-lane insertion using NEON-parallel comparison across
        // the sorted register to find insertion points faster than scalar binary search.
        let remaining = n - init;
        let chunks4 = remaining / 4;
        let mut pos = init;

        for _ in 0..chunks4 {
            let v = vld1q_f32(scores.as_ptr().add(pos));
            let thresh = vdupq_n_f32(heap_vals[k - 1]);
            let mask = vcgtq_f32(v, thresh);

            // Extract mask as u32 lanes — check which elements exceed threshold
            let mask_bits: u32 = core::arch::aarch64::vmaxvq_u32(mask);

            if mask_bits != 0 {
                // At least one lane exceeded threshold — process qualifying lanes
                let mask_arr: [u32; 4] = core::mem::transmute(mask);
                let vals: [f32; 4] = core::mem::transmute(v);
                for lane in 0..4 {
                    if mask_arr[lane] != 0 {
                        insert_sorted_simd_neon(
                            &mut heap_vals,
                            &mut heap_idxs,
                            k,
                            vals[lane],
                            pos + lane,
                        );
                    }
                }
            }
            pos += 4;
        }

        // Scalar tail
        while pos < n {
            let val = *scores.get_unchecked(pos);
            if val > heap_vals[k - 1] {
                insert_sorted_simd_neon(&mut heap_vals, &mut heap_idxs, k, val, pos);
            }
            pos += 1;
        }

        indices.extend(heap_idxs[..k].iter());
    }
}

/// NEON-accelerated insertion into sorted register (descending, k ≤ 16).
///
/// Uses NEON to compare the candidate value against 4 heap elements at a time,
/// finding the insertion point via SIMD-parallel comparison instead of scalar
/// binary search. The shift is still scalar (k≤16 → max 16 shifts = negligible).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn insert_sorted_simd_neon(
    heap_vals: &mut [f32; 16],
    heap_idxs: &mut [usize; 16],
    k: usize,
    val: f32,
    idx: usize,
) {
    use core::arch::aarch64::{vcgtq_f32, vdupq_n_f32, vld1q_f32};
    unsafe {
        // SIMD-parallel search: compare val against 4 heap elements at a time
        // to find the first position where heap_vals[pos] < val.
        let val_vec = vdupq_n_f32(val);
        let mut lo = 0usize;
        let mut found_in_simd = false;

        // Process 4 elements at a time via NEON
        let chunks4 = k / 4;
        for c in 0..chunks4 {
            let base = c * 4;
            let heap_chunk = vld1q_f32(heap_vals.as_ptr().add(base));
            // Find which heap elements are < val (val > heap element)
            let gt_mask = vcgtq_f32(val_vec, heap_chunk);
            let mask_arr: [u32; 4] = core::mem::transmute(gt_mask);

            // Find first position in this chunk where val > heap element
            for lane in 0..4 {
                if mask_arr[lane] != 0 {
                    lo = base + lane;
                    found_in_simd = true;
                    break;
                }
            }
            if found_in_simd {
                break;
            }
        }

        // If not found in NEON chunks, start scanning from the tail of SIMD region
        if !found_in_simd {
            lo = chunks4 * 4;
        }
        while lo < k && heap_vals[lo] >= val {
            lo += 1;
        }

        // lo is the insertion point — if it's at or beyond k, value doesn't qualify
        if lo >= k {
            return;
        }

        // Shift elements down to make room at position `lo`
        for j in (lo + 1..k).rev() {
            heap_vals[j] = heap_vals[j - 1];
            heap_idxs[j] = heap_idxs[j - 1];
        }
        heap_vals[lo] = val;
        heap_idxs[lo] = idx;
    }
}

// ---------------------------------------------------------------------------
// AVX2 path (x86_64)

/// Largest `k` for which the AVX2 kernel is dispatched on x86_64 (Issue 808 T1,
/// option 2 — owner's call, 2026-09-17).
///
/// ⛔ This is **not** the kernel's correctness limit (it is correct to k = 16);
/// it is the measured point past which it stops earning its keep. The arm is a
/// scalar sorted-insert whose *insertion-point search* is vectorised, and the
/// shift that follows is scalar either way — so the win shrinks as `k` grows
/// rather than growing, and `argtopk_scalar_heap` is itself compiled under the
/// same `+avx2` the kernel demands.
///
/// Measured ([Bench 810](../../../../.benchmarks/810_argtopk_distribution_crossover.md),
/// i7-13700K, six deterministic block-score distributions × k ∈ {1,2,4,8,16} ×
/// n ∈ {64..1024}): k ∈ {2, 4} win or tie on **every** distribution past
/// n ≈ 64–128 (up to 3.9×), while k ∈ {8, 16} lose on every distribution but
/// `early_peak` and deepen to 0.41–0.72× on `late_peak`, where **no** n-floor
/// rescues them. That is why this is a flat `k` bound and not the per-`k`
/// `N_MIN` of option 1 — Bench 810 measured `N_MIN` as distribution-sensitive
/// (k = 8: 256 on `locality`, 768 on `iid_uniform`, > 1024 on `late_peak`), and
/// the dispatch cannot observe the distribution.
///
/// ⚠ x86_64 ONLY. The NEON arm keeps the full k ≤ 16 dispatch: it is the
/// algorithm witness this kernel was transcribed from, it is not measured as a
/// loss, and narrowing it here would be a behaviour change on aarch64 that no
/// measurement asked for.
#[cfg(target_arch = "x86_64")]
pub const AVX2_ARGTOPK_K_MAX: usize = 4;

/// SIMD-accelerated top-k for k ≤ 16 on x86_64 AVX2.
///
/// Strategy: same register-based sorted top-k as NEON path, but using
/// 8-wide __m256 comparison via AVX2 intrinsics. Loads 8 scores per
/// iteration, compares against threshold, and inserts qualifying lanes
/// using AVX2-parallel search across the sorted register.
///
/// Dispatched only for `k <= AVX2_ARGTOPK_K_MAX`; larger `k` takes
/// [`argtopk_scalar_heap`], which is measurably faster there (Issue 808).
///
/// ⚠ The kernel is a free function rather than a closure nested in the
/// dispatcher **so that tests can still reach it above the dispatch bound**.
/// Issue 808 T1 narrowed the bound to 4; the kernel stays correct to k = 16,
/// and the two Issue-806 regression tests go through `argtopk_with_scratch`,
/// so at k ∈ {8, 12, 16} they would silently start asserting the SCALAR path
/// on x86_64 — a green zero over the exact arm those two defects were found
/// in. `test_argtopk_avx2_kernel_matches_reference_above_dispatch_bound`
/// calls this directly instead.
///
/// # Safety
/// Caller must have verified AVX2 is available (`is_x86_feature_detected!`).
/// `k` must be in `1..=16` and `scores` non-empty with `k <= scores.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argtopk_avx2_kernel(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    use core::arch::x86_64::{
        _CMP_GT_OS, _mm256_cmp_ps, _mm256_loadu_ps, _mm256_movemask_ps, _mm256_set1_ps,
    };

    let n = scores.len();
    let mut heap_vals = [f32::NEG_INFINITY; 16];
    let mut heap_idxs = [0usize; 16];

    let init = k.min(n);
    for i in 0..init {
        heap_vals[i] = *unsafe { scores.get_unchecked(i) };
        heap_idxs[i] = i;
    }
    // Sort initial k elements descending (insertion sort, k≤16)
    for i in 1..init {
        let val = heap_vals[i];
        let idx = heap_idxs[i];
        let mut j = i;
        while j > 0 && heap_vals[j - 1] < val {
            heap_vals[j] = heap_vals[j - 1];
            heap_idxs[j] = heap_idxs[j - 1];
            j -= 1;
        }
        heap_vals[j] = val;
        heap_idxs[j] = idx;
    }

    // Process remaining elements in AVX2 chunks of 8
    let remaining = n - init;
    let chunks8 = remaining / 8;
    let mut pos = init;

    for _ in 0..chunks8 {
        let v = unsafe { _mm256_loadu_ps(scores.as_ptr().add(pos)) };
        let thresh = _mm256_set1_ps(heap_vals[k - 1]);
        // Compare: v > thresh (ordered, signaling)
        let cmp = _mm256_cmp_ps(v, thresh, _CMP_GT_OS);
        let mask_bits = _mm256_movemask_ps(cmp) as u32;

        if mask_bits != 0 {
            // Extract qualifying lanes
            let vals_arr: [f32; 8] = unsafe { core::mem::transmute(v) };
            for lane in 0..8 {
                if mask_bits & (1 << lane) != 0 {
                    unsafe {
                        insert_sorted_simd_avx2(
                            &mut heap_vals,
                            &mut heap_idxs,
                            k,
                            vals_arr[lane],
                            pos + lane,
                        );
                    };
                }
            }
        }
        pos += 8;
    }

    // No partial-chunk SIMD step: the AVX2 port had one that loaded a FULL
    // 8-wide vector from a position with fewer than 8 scores left (it ran
    // only when `n - pos` was 4..7, so it over-read on every execution) and
    // then masked lanes 4..7 away — UB for a result it discarded. The NEON
    // sibling goes straight to the scalar tail; so does this now (Issue 806).

    // Scalar tail
    while pos < n {
        let val = *unsafe { scores.get_unchecked(pos) };
        if val > heap_vals[k - 1] {
            unsafe { insert_sorted_simd_avx2(&mut heap_vals, &mut heap_idxs, k, val, pos) };
        }
        pos += 1;
    }

    indices.extend(heap_idxs[..k].iter());
}

/// x86_64 dispatch for `k ≤ 16`: the AVX2 kernel up to
/// [`AVX2_ARGTOPK_K_MAX`], [`argtopk_scalar_heap`] above it and whenever AVX2
/// is absent at runtime.
#[cfg(target_arch = "x86_64")]
fn argtopk_simd(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    // Safety: the kernel is `#[target_feature(enable = "avx2")]`, so calling it
    // is sound only once the runtime probe below has succeeded.
    //
    // The `k` bound is a PERFORMANCE gate, not a correctness one — see
    // `AVX2_ARGTOPK_K_MAX` (Issue 808 T1). Both arms produce identical
    // indices; `test_argtopk_simd_matches_scalar` spans the boundary.
    if k <= AVX2_ARGTOPK_K_MAX && is_x86_feature_detected!("avx2") {
        unsafe { argtopk_avx2_kernel(scores, k, indices) };
    } else {
        argtopk_scalar_heap(scores, k, indices);
    }
}

/// AVX2-accelerated insertion into sorted register (descending, k ≤ 16).
///
/// Uses AVX2 to compare the candidate value against 8 heap elements at a time,
/// finding the insertion point via SIMD-parallel comparison.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn insert_sorted_simd_avx2(
    heap_vals: &mut [f32; 16],
    heap_idxs: &mut [usize; 16],
    k: usize,
    val: f32,
    idx: usize,
) {
    use core::arch::x86_64::{
        _CMP_GT_OS, _mm256_cmp_ps, _mm256_loadu_ps, _mm256_movemask_ps, _mm256_set1_ps,
    };

    // SIMD-parallel search: compare val against up to 8 heap elements at a time
    let val_vec = _mm256_set1_ps(val);
    let mut lo = 0usize;
    // The NEON sibling carries this flag and the AVX2 port replaced it with a
    // `lo == 0` test — which cannot tell "no lane qualified" from "lane 0
    // qualified", i.e. from `val` being a new MAXIMUM. That reset `lo` to `k`
    // and returned, DISCARDING every new top-1 (Issue 806). Keep the flag.
    let mut found_in_simd = false;

    // Process 8 elements at a time via AVX2
    let chunks8 = k / 8;
    for c in 0..chunks8 {
        let base = c * 8;
        let heap_chunk = unsafe { _mm256_loadu_ps(heap_vals.as_ptr().add(base)) };
        let cmp = _mm256_cmp_ps(val_vec, heap_chunk, _CMP_GT_OS);
        let mask_bits = _mm256_movemask_ps(cmp) as u32;

        if mask_bits != 0 {
            // Find first set bit = first position where val > heap element
            lo = base + mask_bits.trailing_zeros() as usize;
            found_in_simd = true;
            break;
        }
    }

    // If not found in AVX2 chunks, scan the remainder (0 when chunks8 == 0,
    // which is the k < 8 case the scalar walk below handles on its own).
    if !found_in_simd {
        lo = chunks8 * 8;
    }
    while lo < k && heap_vals[lo] >= val {
        lo += 1;
    }

    if lo >= k {
        return;
    }

    // Shift elements down to make room at position `lo`
    for j in (lo + 1..k).rev() {
        heap_vals[j] = heap_vals[j - 1];
        heap_idxs[j] = heap_idxs[j - 1];
    }
    heap_vals[lo] = val;
    heap_idxs[lo] = idx;
}

/// Scalar fallback for targets without NEON or AVX2.
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn argtopk_simd(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    argtopk_scalar_heap(scores, k, indices);
}

/// Scalar min-heap top-k: O(n * log(k)).
/// Used as fallback when SIMD is not available.
pub fn argtopk_scalar_heap(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    let n = scores.len();
    let init = k.min(n);

    let mut heap_vals = [f32::NEG_INFINITY; 16];
    let mut heap_idxs = [0usize; 16];

    for i in 0..init {
        heap_vals[i] = scores[i];
        heap_idxs[i] = i;
    }
    // Sort initial k elements descending (insertion sort)
    for i in 1..init {
        let val = heap_vals[i];
        let idx = heap_idxs[i];
        let mut j = i;
        while j > 0 && heap_vals[j - 1] < val {
            heap_vals[j] = heap_vals[j - 1];
            heap_idxs[j] = heap_idxs[j - 1];
            j -= 1;
        }
        heap_vals[j] = val;
        heap_idxs[j] = idx;
    }

    // Process remaining elements
    for i in init..n {
        insert_sorted(&mut heap_vals, &mut heap_idxs, k, scores[i], i);
    }

    indices.extend(heap_idxs[..k].iter());
}

/// Standard sigmoid function.
/// Delegates to [`katgpt_core::simd::fast_sigmoid`] (Cephes polynomial).
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
}

// ---------------------------------------------------------------------------
// PerGroupTopKRouter — per-GQA-group independent top-k (Plan 256 Phase 2)
// ---------------------------------------------------------------------------

/// Per-GQA-group independent top-k router.
///
/// Instead of one shared top-k across all blocks for a KV head,
/// each GQA group selects its own top blocks independently from the blocks
/// assigned to that group. This allows different attention heads to focus
/// on different regions of the KV cache.
///
/// Feature gate: `msa_per_group` (Plan 256 Phase 2 GOAT gate).
#[cfg(feature = "msa_per_group")]
#[derive(Debug)]
pub struct PerGroupTopKRouter {
    /// Inner router for cache management and fallback.
    pub inner: BlockTopKRouter,
    /// Number of GQA groups (query heads sharing one KV head is one group).
    pub n_groups: usize,
}

#[cfg(feature = "msa_per_group")]
impl PerGroupTopKRouter {
    /// Create a new per-group router.
    ///
    /// * `scale` — apply `1/sqrt(head_dim)` scaling to dot products
    /// * `n_groups` — number of GQA groups (must be > 0)
    pub fn new(scale: bool, n_groups: usize) -> Self {
        assert!(n_groups > 0, "n_groups must be > 0");
        Self {
            inner: BlockTopKRouter::new(scale),
            n_groups,
        }
    }
}

#[cfg(feature = "msa_per_group")]
impl VortexFlow for PerGroupTopKRouter {
    type Cache = BlockTopKCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        // Delegate to inner BlockTopKRouter — centroid computation is group-independent
        self.inner
            .forward_cache(cache, keys, values, block_idx, head_dim);
    }

    fn forward_indexer(
        &self,
        query: &[f32],
        cache: &Self::Cache,
        n_blocks: usize,
        top_k: usize,
        scratch: &mut VortexScratch,
    ) -> RoutingDecision {
        if n_blocks == 0 || top_k == 0 {
            return RoutingDecision::new();
        }

        let hd = query.len();
        let scale = if self.inner.scale {
            1.0 / (hd as f32).sqrt()
        } else {
            1.0
        };

        // Budget: distribute top_k across groups, ensuring each gets at least 1
        let per_group_k = (top_k / self.n_groups).max(1);
        let total_budget = per_group_k * self.n_groups;

        // Compute all dot-product scores once
        scratch.ensure_capacity(n_blocks);
        let scores = &mut scratch.scores[..n_blocks];
        for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
            let centroid = cache.centroid(i);
            let mut dot0 = 0.0f32;
            let mut dot1 = 0.0f32;
            let mut dot2 = 0.0f32;
            let mut dot3 = 0.0f32;
            // `as_chunks::<4>()` over pre-sliced rows: each yielded `&[f32; 4]`
            // has a statically known length, so the per-iteration bounds checks
            // collapse. The four lane accumulators receive their addends in
            // exactly the same order as the index form and the tail covers the
            // same `main..hd` range → bit-identical.
            let main = (hd / 4) * 4;
            for (q_c, c_c) in query[..main]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(centroid[..main].as_chunks::<4>().0.iter())
            {
                dot0 += q_c[0] * c_c[0];
                dot1 += q_c[1] * c_c[1];
                dot2 += q_c[2] * c_c[2];
                dot3 += q_c[3] * c_c[3];
            }
            let mut dot = dot0 + dot1 + dot2 + dot3;
            for (&q_d, &c_d) in query[main..hd].iter().zip(centroid[main..hd].iter()) {
                dot += q_d * c_d;
            }
            *score = dot * scale;
        }

        // Partition blocks into groups (round-robin assignment)
        // Group g owns blocks where block_idx % n_groups == g
        let mut decision = RoutingDecision::with_capacity(total_budget);
        // Pre-size per-group scratch: each group holds at most ceil(n_blocks/n_groups)
        // blocks. Avoids the 1-3 reallocs per group that `Vec::new()` + push
        // triggers as the allocator walks 8/16/32-byte doublings.
        let group_cap = n_blocks.div_ceil(self.n_groups);
        let mut group_indices: Vec<(usize, f32)> = Vec::with_capacity(group_cap);
        let mut group_pairs: Vec<(usize, f32)> = Vec::with_capacity(group_cap);
        let mut group_scores: Vec<f32> = Vec::with_capacity(group_cap);
        let mut local_indices: Vec<usize> = Vec::with_capacity(per_group_k);

        for g in 0..self.n_groups {
            // Collect scores for blocks belonging to this group
            group_indices.clear();
            group_pairs.clear();
            group_scores.clear();

            for i in (0..n_blocks).filter(|&i| i % self.n_groups == g) {
                group_indices.push((i, scores[i]));
            }

            let group_n = group_indices.len();
            if group_n == 0 {
                continue;
            }

            // Extract scores for this group's blocks into the reusable buffer
            group_scores.extend(group_indices.iter().map(|&(_, s)| s));
            let k = per_group_k.min(group_n);

            local_indices.clear();
            argtopk_with_scratch(&group_scores, k, &mut local_indices, &mut group_pairs);

            // Map local indices back to global block indices
            for &local_idx in &local_indices[..k] {
                let global_block = group_indices[local_idx].0;
                decision.blocks.push(global_block);
                decision.weights.push(sigmoid(scores[global_block]));
            }
        }

        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        self.inner.cache_new(n_blocks_capacity, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 4;

    fn make_router() -> BlockTopKRouter {
        BlockTopKRouter::new(true)
    }

    #[test]
    fn test_known_keys_known_centroids_known_topk() {
        let router = make_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        // Block 0: keys = [[1,0,0,0], [1,0,0,0]] → centroid [1,0,0,0]
        let keys0 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let vals0 = vec![0.0; 8];
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HEAD_DIM);

        // Block 1: keys = [[0,1,0,0], [0,1,0,0]] → centroid [0,1,0,0]
        let keys1 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let vals1 = vec![0.0; 8];
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HEAD_DIM);

        // Block 2: keys = [[0,0,1,0], [0,0,1,0]] → centroid [0,0,1,0]
        let keys2 = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let vals2 = vec![0.0; 8];
        router.forward_cache(&mut cache, &keys2, &vals2, 2, HEAD_DIM);

        // Query aligned with block 1 → should select block 1 first
        let query = vec![0.0, 1.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 3, 1, &mut scratch);
        assert_eq!(decision.blocks.len(), 1);
        assert_eq!(decision.blocks[0], 1);
    }

    #[test]
    fn test_single_block_always_selected() {
        let router = make_router();
        let mut cache = router.cache_new(1, HEAD_DIM);
        let mut scratch = VortexScratch::new(1);

        let keys = vec![1.0, 2.0, 3.0, 4.0];
        let vals = vec![5.0, 6.0, 7.0, 8.0];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        let query = vec![0.1, 0.2, 0.3, 0.4];
        let decision = router.forward_indexer(&query, &cache, 1, 1, &mut scratch);
        assert_eq!(decision.blocks, vec![0]);
    }

    #[test]
    fn test_zero_query_all_scores_equal() {
        let router = make_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let keys2 = vec![0.0, 0.0, 1.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];

        router.forward_cache(&mut cache, &keys0, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys1, &vals, 1, HEAD_DIM);
        router.forward_cache(&mut cache, &keys2, &vals, 2, HEAD_DIM);

        // Zero query → all dot products = 0
        let query = vec![0.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 3, 2, &mut scratch);
        assert_eq!(decision.blocks.len(), 2);
        // All scores are 0.0, any 2 of 3 blocks selected is valid
        assert!(decision.blocks.iter().all(|&b| b < 3));
    }

    #[test]
    fn test_empty_blocks_returns_empty() {
        let router = make_router();
        let cache = router.cache_new(0, HEAD_DIM);
        let mut scratch = VortexScratch::new(0);

        let query = vec![1.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 0, 4, &mut scratch);
        assert!(decision.is_empty());
    }

    #[test]
    fn test_topk_capped_at_n_blocks() {
        let router = make_router();
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys, &vals, 1, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        // top_k=10 but only 2 blocks
        let decision = router.forward_indexer(&query, &cache, 2, 10, &mut scratch);
        assert_eq!(decision.blocks.len(), 2);
    }

    #[test]
    fn test_cache_ensure_capacity_grows() {
        let mut cache = BlockTopKCache::new(2, HEAD_DIM);
        assert_eq!(cache.centroids.len(), 8);
        cache.ensure_capacity(5);
        assert!(cache.centroids.len() >= 20);
    }

    #[test]
    fn test_centroid_mean_pooling() {
        let router = make_router();
        let mut cache = router.cache_new(1, HEAD_DIM);

        // 3 tokens: [1,0,0,0], [0,2,0,0], [0,0,3,0] → mean = [1/3, 2/3, 1, 0]
        let keys = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let vals = vec![0.0; 12];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        let c = cache.centroid(0);
        let expected = [1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0];
        for (i, (&got, &exp)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "centroid mismatch at dim {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_argtopk_descending() {
        let scores = [0.5, 2.0, 0.1, 3.0, 1.5];
        let mut indices = Vec::new();
        argtopk(&scores, 3, &mut indices);
        assert_eq!(indices.len(), 3);
        // Descending order by score
        assert_eq!(indices[0], 3); // score 3.0
        assert_eq!(indices[1], 1); // score 2.0
        assert_eq!(indices[2], 4); // score 1.5
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    // ── Plan 256: SIMD argtopk tests ─────────────────────────────────

    #[test]
    fn test_argtopk_k1_uses_simd_path() {
        let scores = [0.1, 0.5, 0.3, 0.8, 0.2];
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 1, &mut indices, &mut pairs);
        assert_eq!(indices, vec![3]); // score 0.8
    }

    #[test]
    fn test_argtopk_k4_simd_path() {
        // k=4, n=20 — hits the SIMD register path
        let scores: Vec<f32> = (0..20).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 4, &mut indices, &mut pairs);
        assert_eq!(indices.len(), 4);
        // Verify descending order
        for w in indices.windows(2) {
            assert!(
                scores[w[0]] >= scores[w[1]],
                "not descending: scores[{}] = {} < scores[{}] = {}",
                w[0],
                scores[w[0]],
                w[1],
                scores[w[1]]
            );
        }
    }

    #[test]
    fn test_argtopk_k8_simd_path() {
        // k=8, n=32 — NEON chunk processing
        let scores: Vec<f32> = (0..32).rev().map(|i| i as f32).collect();
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 8, &mut indices, &mut pairs);
        assert_eq!(indices.len(), 8);
        // Scores are 31,30,...,0 so top-8 = 31..24 at indices 0..8
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_argtopk_k16_boundary() {
        // k=16 is the SIMD path boundary
        let scores: Vec<f32> = (0..64).map(|i| (i as f32 * 0.5).cos()).collect();
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 16, &mut indices, &mut pairs);
        assert_eq!(indices.len(), 16);
        // Verify all selected indices are valid
        assert!(indices.iter().all(|&i| i < 64));
        // Verify descending
        for w in indices.windows(2) {
            assert!(
                scores[w[0]] >= scores[w[1]],
                "not descending: scores[{}] = {} < scores[{}] = {}",
                w[0],
                scores[w[0]],
                w[1],
                scores[w[1]]
            );
        }
    }

    #[test]
    fn test_argtopk_k17_falls_back_to_selection() {
        // k=17 > 16, should use selection sort fallback
        let scores: Vec<f32> = (0..50).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut indices = Vec::new();
        let mut pairs = Vec::new();
        argtopk_with_scratch(&scores, 17, &mut indices, &mut pairs);
        assert_eq!(indices.len(), 17);
        for w in indices.windows(2) {
            assert!(
                scores[w[0]] >= scores[w[1]],
                "not descending: scores[{}] = {} < scores[{}] = {}",
                w[0],
                scores[w[0]],
                w[1],
                scores[w[1]]
            );
        }
    }

    #[test]
    fn test_argtopk_simd_matches_scalar() {
        // Verify SIMD and scalar paths produce identical results
        let scores: Vec<f32> = [
            0.5, -1.2, 3.4, 0.0, 2.1, -0.5, 1.8, 4.2, -3.0, 0.7, 1.1, -2.0, 2.9, 0.3, -1.1, 3.8,
            0.1, 1.5, -0.3, 2.6,
        ]
        .to_vec();
        for k in [1, 2, 4, 8, 16] {
            // SIMD path
            let mut simd_indices = Vec::new();
            let mut pairs = Vec::new();
            argtopk_with_scratch(&scores, k, &mut simd_indices, &mut pairs);

            // Reference: full sort
            let mut ref_pairs: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
            ref_pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            let ref_indices: Vec<usize> = ref_pairs[..k].iter().map(|(i, _)| *i).collect();

            assert_eq!(
                simd_indices, ref_indices,
                "k={k}: SIMD result {simd_indices:?} != reference {ref_indices:?}"
            );
        }
    }

    /// Issue 806 regression — a candidate that is a NEW MAXIMUM must land at
    /// position 0.
    ///
    /// The AVX2 insertion-point search used `lo == 0` as its "no lane
    /// qualified" sentinel, which cannot be told apart from "lane 0 qualified".
    /// Every new top-1 arriving after the first `k` elements was reset to `k`
    /// and DISCARDED. The NEON sibling carries an explicit `found_in_simd`
    /// flag and never had it.
    ///
    /// `test_argtopk_simd_matches_scalar` above cannot express this: its
    /// fixture's maximum (4.2) sits at index 7, inside the first `k`, so no
    /// later element is ever a new maximum and the branch is unreachable. A
    /// strictly ascending ramp makes EVERY post-init element one.
    #[test]
    fn test_argtopk_new_maximum_after_init_is_kept() {
        for n in [24usize, 32, 40, 64, 100] {
            for k in [1usize, 4, 8, 12, 16] {
                if k > n {
                    continue;
                }
                let scores: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let mut indices = Vec::new();
                let mut pairs = Vec::new();
                argtopk_with_scratch(&scores, k, &mut indices, &mut pairs);
                // Strictly ascending ⇒ the top-k is the LAST k indices, in
                // reverse order, with no ties to arbitrate.
                let expected: Vec<usize> = (n - k..n).rev().collect();
                assert_eq!(indices, expected, "n={n} k={k}");
            }
        }
    }

    /// Issue 806 regression — the tail past the last full SIMD chunk.
    ///
    /// The AVX2 port had a "remaining 4" step that loaded a FULL 8-wide vector
    /// from a position with 4..7 scores left and masked lanes 4..7 away: an
    /// out-of-bounds read on every execution, for a result it discarded. These
    /// lengths put 1..7 elements past the last full chunk for k = 8.
    #[test]
    fn test_argtopk_partial_tail_lengths_match_reference() {
        for n in [17usize, 18, 19, 20, 21, 22, 23, 25, 31, 33] {
            for k in [4usize, 8, 16] {
                if k > n {
                    continue;
                }
                let scores: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32) * 0.25).collect();
                let mut indices = Vec::new();
                let mut pairs = Vec::new();
                argtopk_with_scratch(&scores, k, &mut indices, &mut pairs);

                let mut ref_pairs: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
                ref_pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
                let expected: Vec<usize> = ref_pairs[..k].iter().map(|(i, _)| *i).collect();
                assert_eq!(indices, expected, "n={n} k={k}");
            }
        }
    }

    /// Issue 808 T1 — the AVX2 kernel ABOVE its own dispatch bound.
    ///
    /// Option 2 narrowed the x86_64 dispatch to `k ≤ AVX2_ARGTOPK_K_MAX` (= 4)
    /// on Bench 810's evidence. The kernel is still compiled and still correct
    /// to k = 16 — but the two Issue-806 regression tests reach it through
    /// `argtopk_with_scratch`, so at k ∈ {8, 12, 16} they now assert the
    /// SCALAR path on this platform and the AVX2 arm goes back to being
    /// executed by nothing. That is the precondition both 806 defects grew in
    /// (`full_gate` is macOS/aarch64; the one target that ran this kernel was
    /// a `#![cfg]` green zero), and narrowing a dispatch must not recreate it.
    ///
    /// So this calls the kernel DIRECTLY, over both 806 fixtures, for every
    /// `k` the dispatch no longer sends it: the ascending ramp that makes
    /// every post-init element a new maximum (defect 1), and the lengths that
    /// leave 1..7 elements past the last full 8-wide chunk (defect 2).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_argtopk_avx2_kernel_matches_reference_above_dispatch_bound() {
        if !is_x86_feature_detected!("avx2") {
            // Not a skip that can hide a regression: the kernel is
            // unreachable on this box in production too.
            eprintln!("avx2 not detected — kernel unreachable here, nothing to assert");
            return;
        }

        // Every k the dispatch stopped sending here, plus one inside the
        // bound as a control that the direct call agrees with the dispatch.
        let ks = [4usize, 5, 8, 12, 16];
        assert!(
            ks.iter().any(|&k| k > AVX2_ARGTOPK_K_MAX),
            "fixture must span ABOVE the dispatch bound ({AVX2_ARGTOPK_K_MAX}), \
             or this test asserts only what argtopk_with_scratch already does"
        );

        // Fixture A (806 defect 1): strictly ascending ⇒ every post-init
        // element is a new maximum, so the insertion search's "new top-1"
        // branch runs on every step.
        for n in [24usize, 32, 40, 64, 100] {
            for &k in &ks {
                if k > n {
                    continue;
                }
                let scores: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let mut indices = Vec::new();
                unsafe { argtopk_avx2_kernel(&scores, k, &mut indices) };
                let expected: Vec<usize> = (n - k..n).rev().collect();
                assert_eq!(indices, expected, "ascending ramp: n={n} k={k}");
            }
        }

        // Fixture B (806 defect 2): 1..7 elements past the last full chunk,
        // where the deleted partial-chunk step used to over-read.
        for n in [17usize, 18, 19, 20, 21, 22, 23, 25, 31, 33] {
            for &k in &ks {
                if k > n {
                    continue;
                }
                let scores: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32) * 0.25).collect();
                let mut indices = Vec::new();
                unsafe { argtopk_avx2_kernel(&scores, k, &mut indices) };

                let mut ref_pairs: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
                ref_pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
                let expected: Vec<usize> = ref_pairs[..k].iter().map(|(i, _)| *i).collect();
                assert_eq!(indices, expected, "partial tail: n={n} k={k}");
            }
        }
    }

    /// Issue 808 T2 — the bound is PINNED, and widening it needs a measurement.
    ///
    /// This is the load-immune half of T2's "the next regression is a test
    /// failure rather than a table nobody reads". The timing half lives in
    /// `tests/bench_256_simd_topk.rs`; a timing bar alone cannot be trusted to
    /// red, because this repo has measured perf bars producing four different
    /// failing sets across four runs of one commit (Bench 806 T7). A constant
    /// does not oscillate.
    ///
    /// 4 is not a round number chosen for tidiness — it is where Bench 810's
    /// six distributions agree. k ∈ {2, 4} win or tie on ALL of them past
    /// n ≈ 64–128; k ∈ {8, 16} lose on all but `early_peak` and reach
    /// 0.41–0.72× on `late_peak`, where no n-floor rescues them.
    ///
    /// ⛔ Raising this is a behaviour change on every x86_64 deployment and
    /// needs Issue 808 T2's bar first: `N_MIN` per k on **≥ 2**
    /// microarchitectures (Raptor Lake is 1 of 2), across the distribution
    /// axis, since Bench 810 measured `N_MIN` as distribution-sensitive.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_argtopk_dispatch_bound_is_pinned() {
        assert_eq!(
            AVX2_ARGTOPK_K_MAX, 4,
            "the AVX2 argtopk dispatch bound moved. Bench 810 licensed k <= 4 \
             on ONE microarchitecture; Issue 808 T2 requires >= 2 plus the \
             distribution axis before this widens. Re-measure, then re-pin."
        );
    }

    /// Issue 808 T1 — the dispatch bound is a PERFORMANCE gate, so both sides
    /// of it must return identical indices. A bound that changed results would
    /// be a behaviour change on every x86_64 deployment.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_argtopk_dispatch_bound_is_result_neutral() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for n in [20usize, 33, 64, 129] {
            for k in 1..=16usize {
                if k > n {
                    continue;
                }
                let scores: Vec<f32> = (0..n).map(|i| ((i * 61 % 97) as f32) * 0.125).collect();

                let mut via_kernel = Vec::new();
                unsafe { argtopk_avx2_kernel(&scores, k, &mut via_kernel) };

                let mut via_scalar = Vec::new();
                argtopk_scalar_heap(&scores, k, &mut via_scalar);

                assert_eq!(
                    via_kernel, via_scalar,
                    "kernel and scalar heap disagree at n={n} k={k} — the \
                     AVX2_ARGTOPK_K_MAX bound is not result-neutral"
                );
            }
        }
    }

    #[test]
    fn test_argtopk_ties_stable_first_occurrence() {
        // Duplicate scores — should prefer earlier indices
        let scores = [1.0, 2.0, 2.0, 1.0, 3.0];
        let mut indices = Vec::new();
        argtopk(&scores, 3, &mut indices);
        assert_eq!(indices[0], 4); // 3.0
        // Both 2.0 values — should pick first occurrence
        assert_eq!(indices[1], 1); // first 2.0
        assert_eq!(indices[2], 2); // second 2.0
    }

    #[test]
    fn test_argtopk_n_equals_k() {
        // n == k: all elements selected
        let scores = [3.0, 1.0, 4.0];
        let mut indices = Vec::new();
        argtopk(&scores, 3, &mut indices);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![2, 0, 1]); // 4.0, 3.0, 1.0
    }

    // -----------------------------------------------------------------------
    // PerGroupTopKRouter tests (Plan 256 Phase 2)
    // -----------------------------------------------------------------------

    #[cfg(feature = "msa_per_group")]
    fn cache_centroids(
        router: &super::PerGroupTopKRouter,
        cache: &mut BlockTopKCache,
        centroids: &[Vec<f32>],
        head_dim: usize,
    ) {
        let values = vec![0.0f32; head_dim];
        for (i, centroid) in centroids.iter().enumerate() {
            // Build keys as [centroid] * block_size (block_size=1 for simplicity)
            router.forward_cache(cache, centroid, &values, i, head_dim);
        }
    }

    /// Per-group selection returns different blocks per group when centroids differ.
    #[test]
    #[cfg(feature = "msa_per_group")]
    fn test_per_group_different_blocks_per_group() {
        use super::PerGroupTopKRouter;
        use super::VortexFlow;

        let n_groups = 2;
        let router = PerGroupTopKRouter::new(true, n_groups);
        let mut cache = router.cache_new(6, HEAD_DIM);

        // 6 blocks with distinct centroids
        // Group 0: blocks 0, 2, 4
        // Group 1: blocks 1, 3, 5
        let centroids: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0], // block 0 (group 0)
            vec![0.0, 1.0, 0.0, 0.0], // block 1 (group 1)
            vec![0.5, 0.0, 0.0, 0.0], // block 2 (group 0)
            vec![0.0, 0.5, 0.0, 0.0], // block 3 (group 1)
            vec![0.1, 0.0, 0.0, 0.0], // block 4 (group 0)
            vec![0.0, 0.1, 0.0, 0.0], // block 5 (group 1)
        ];
        cache_centroids(&router, &mut cache, &centroids, HEAD_DIM);

        // Query aligned with group 0's best (block 0) and group 1's best (block 1)
        let query = vec![1.0, 1.0, 0.0, 0.0];
        let mut scratch = VortexScratch::new(6);
        let decision = router.forward_indexer(&query, &cache, 6, 4, &mut scratch);

        // Should select blocks from both groups
        // Group 0's top: block 0 (highest), block 2
        // Group 1's top: block 1 (highest), block 3
        assert_eq!(decision.len(), 4, "expected 4 blocks (2 per group)");
        assert!(
            decision.blocks.contains(&0),
            "block 0 should be selected (group 0 best)"
        );
        assert!(
            decision.blocks.contains(&1),
            "block 1 should be selected (group 1 best)"
        );
    }

    /// Total selected blocks ≤ top_k.
    #[test]
    #[cfg(feature = "msa_per_group")]
    fn test_per_group_total_leq_topk() {
        use super::PerGroupTopKRouter;
        use super::VortexFlow;

        let n_groups = 3;
        let router = PerGroupTopKRouter::new(true, n_groups);
        let mut cache = router.cache_new(9, HEAD_DIM);

        let centroids: Vec<Vec<f32>> = (0..9)
            .map(|i| {
                let mut v = vec![0.0; HEAD_DIM];
                v[i % HEAD_DIM] = (i + 1) as f32;
                v
            })
            .collect();
        cache_centroids(&router, &mut cache, &centroids, HEAD_DIM);

        let query = vec![1.0, 1.0, 1.0, 1.0];
        let mut scratch = VortexScratch::new(9);
        let decision = router.forward_indexer(&query, &cache, 9, 6, &mut scratch);

        assert!(
            decision.len() <= 6,
            "selected {} blocks but top_k=6",
            decision.len()
        );
    }

    /// Each group gets at least 1 block when top_k >= n_groups.
    #[test]
    #[cfg(feature = "msa_per_group")]
    fn test_per_group_each_group_gets_block() {
        use super::PerGroupTopKRouter;
        use super::VortexFlow;

        let n_groups = 3;
        let router = PerGroupTopKRouter::new(true, n_groups);
        let mut cache = router.cache_new(9, HEAD_DIM);

        // 9 blocks, 3 groups → 3 blocks each
        let centroids: Vec<Vec<f32>> = (0..9)
            .map(|i| {
                let mut v = vec![0.0; HEAD_DIM];
                v[i % HEAD_DIM] = (i + 1) as f32;
                v
            })
            .collect();
        cache_centroids(&router, &mut cache, &centroids, HEAD_DIM);

        let query = vec![1.0, 1.0, 1.0, 1.0];
        let mut scratch = VortexScratch::new(9);
        let decision = router.forward_indexer(&query, &cache, 9, 6, &mut scratch);

        // Each group should contribute at least 1 block
        for g in 0..n_groups {
            let group_has_block = decision.blocks.iter().any(|&b| b % n_groups == g);
            assert!(group_has_block, "group {g} has no selected blocks");
        }
    }

    /// Backward compatible when n_groups=1 (same as BlockTopKRouter).
    #[test]
    #[cfg(feature = "msa_per_group")]
    fn test_per_group_backward_compat_single_group() {
        use super::PerGroupTopKRouter;
        use super::VortexFlow;

        let router = PerGroupTopKRouter::new(true, 1);
        let plain_router = BlockTopKRouter::new(true);
        let mut cache = router.cache_new(4, HEAD_DIM);
        let mut plain_cache = BlockTopKCache::new(4, HEAD_DIM);

        let centroids: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let values = vec![0.0f32; HEAD_DIM];
        for (i, c) in centroids.iter().enumerate() {
            router.forward_cache(&mut cache, c, &values, i, HEAD_DIM);
            plain_router.forward_cache(&mut plain_cache, c, &values, i, HEAD_DIM);
        }

        let query = vec![1.0, 0.5, 0.0, 0.0];
        let mut scratch = VortexScratch::new(4);
        let mut plain_scratch = VortexScratch::new(4);

        let decision = router.forward_indexer(&query, &cache, 4, 2, &mut scratch);
        let plain_decision =
            plain_router.forward_indexer(&query, &plain_cache, 4, 2, &mut plain_scratch);

        // n_groups=1 should produce identical block selection
        assert_eq!(
            decision.blocks, plain_decision.blocks,
            "n_groups=1 should match BlockTopKRouter selection"
        );
    }
}

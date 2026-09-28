//! AR-ness (autoregressive-ness) statistics over decode trajectories
//! (Plan 602 T1.1, distilled from arXiv:2609.20751 §decode-order metrics).
//!
//! # The trajectory π
//!
//! `π[i]` is the **unmask step** of position `i`: the decode iteration
//! (denoise step for D2F, forward-pass index for SW-SetDLM) at which the
//! position committed its final token. An AR (left-to-right) decode has
//! `π = [0, 1, 2, …]`; a fully parallel decode has `π = [s, s, …, s]`
//! (every position committed in the same step); block-decoding interleaves
//! the two. These statistics quantify where a trajectory sits on that
//! AR↔parallel axis — the measured explanation axis for hybrid-attention
//! regime studies (e.g. why SW-SetDLM's w=0.5 wins, Plans 379–384).
//!
//! # Contracts (all functions)
//!
//! - **Pure, zero-alloc** — no heap, no state; safe in hot instrumented
//!   loops and on streaming canvases.
//! - **Sentinel**: pairs touching [`UNMASKED_NEVER`] are **excluded** from
//!   both numerator and denominator (never-committed positions have no
//!   defined order — measure the committed sub-trajectory).
//! - **Ties are discordant**: `π[i] == π[j]` (parallel commit in one step)
//!   counts as NOT autoregressive. This is a deliberate divergence from
//!   Kendall tau-b (which drops ties from the denominator): parallel decode
//!   is exactly the non-AR behavior the instrument exists to detect.
//! - **Degenerate input** (no valid pairs after exclusions) returns `NaN`,
//!   never a silent 0.5/1.0 — callers must see the empty measurement.
//!
//! # Expectations (sanity anchors)
//!
//! | π | ALR | AGR |
//! |---|-----|-----|
//! | identity `[0,1,2,…]` | 1.0 | 1.0 |
//! | reversed `[L-1,…,1,0]` | 0.0 | 0.0 |
//! | random order | ≈ 0.5 | ≈ 0.5 |

/// Sentinel for a position that never committed (still masked at decode
/// end). Pairs touching this value are excluded from every statistic.
pub const UNMASKED_NEVER: u32 = u32::MAX;

/// Local AR-ness (ALR): the fraction of valid **adjacent** position pairs
/// decoded in increasing step order — `π[i] < π[i+1]`.
///
/// This is the local window-2 case of [`global_ar_ness`] and the statistic
/// the dQwen3.5 paper reports as "local AR-ness": adjacent text positions
/// decoded back-to-back in the correct order. Ties (parallel commit of
/// adjacent positions) are discordant; sentinel pairs are excluded.
///
/// Returns `NaN` when fewer than two non-sentinel positions exist (no valid
/// adjacent pair).
pub fn local_ar_ness(pi: &[u32]) -> f32 {
    let mut concordant: u32 = 0;
    let mut valid: u32 = 0;
    for pair in pi.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if a == UNMASKED_NEVER || b == UNMASKED_NEVER {
            continue;
        }
        valid += 1;
        if a < b {
            concordant += 1;
        }
    }
    if valid == 0 {
        f32::NAN
    } else {
        concordant as f32 / valid as f32
    }
}

/// Global AR-ness (AGR): the fraction of valid **all-pairs** position
/// combinations decoded in increasing step order — `π[i] < π[j]` for all
/// `i < j`.
///
/// Equivalent to 1 minus the normalized inversion count of π (the concordant
/// half of Kendall's tau, with ties counted as discordant by design — see
/// the module contract). O(L²) pair comparisons, zero alloc: at L=4096 that
/// is ~8.4M branch-light comparisons (sub-millisecond), fine for bench
/// cross-tabs; use [`global_ar_ness_windowed`] for streaming canvases or
/// when only local order structure matters.
///
/// Returns `NaN` when no valid pair exists.
pub fn global_ar_ness(pi: &[u32]) -> f32 {
    let mut concordant: u64 = 0;
    let mut valid: u64 = 0;
    for i in 0..pi.len() {
        if pi[i] == UNMASKED_NEVER {
            continue;
        }
        for j in (i + 1)..pi.len() {
            if pi[j] == UNMASKED_NEVER {
                continue;
            }
            valid += 1;
            if pi[i] < pi[j] {
                concordant += 1;
            }
        }
    }
    if valid == 0 {
        f32::NAN
    } else {
        concordant as f32 / valid as f32
    }
}

/// Windowed global AR-ness: the mean AGR over sliding windows of `window`
/// positions, in **O(L·W)** via incremental window slide (each slide drops
/// the leaving left-endpoint pairs and adds the entering right-endpoint
/// pairs — W comparisons per slide, never a W²/2 re-scan per window).
///
/// For streaming canvases and for trajectories whose global order is
/// meaningless across long range (early text vs late text decoded in
/// separate passes) but whose *local* block structure matters. Identities
/// pinned by test: `window = 2` equals [`local_ar_ness`] (one pair per
/// window); `window >= π.len()` equals [`global_ar_ness`] (single window).
/// Windows with no valid pairs are skipped; `window < 2` or a trajectory
/// with no valid window returns `NaN`.
pub fn global_ar_ness_windowed(pi: &[u32], window: usize) -> f32 {
    if window < 2 || pi.len() < 2 {
        return f32::NAN;
    }
    let w = window.min(pi.len());
    let is_valid = |x: u32| x != UNMASKED_NEVER;

    // Seed window 0: positions [0, w).
    let mut concordant: u64 = 0;
    let mut valid: u64 = 0;
    for i in 0..w {
        if !is_valid(pi[i]) {
            continue;
        }
        for j in (i + 1)..w {
            if !is_valid(pi[j]) {
                continue;
            }
            valid += 1;
            if pi[i] < pi[j] {
                concordant += 1;
            }
        }
    }

    let mut sum: f32 = 0.0;
    let mut n_valid_windows: u64 = 0;
    fn push(concordant: u64, valid: u64, sum: &mut f32, n: &mut u64) {
        if valid > 0 {
            *sum += concordant as f32 / valid as f32;
            *n += 1;
        }
    }
    push(concordant, valid, &mut sum, &mut n_valid_windows);

    // Slide: window k covers positions [k, k+w). Leaving left endpoint is
    // k-1 (its pairs had right endpoints in [k, k+w-1)); entering right
    // endpoint is k+w-1 (its pairs have left endpoints in [k, k+w-1)).
    for k in 1..=(pi.len() - w) {
        let old = k - 1;
        let new = k + w - 1;
        if is_valid(pi[old]) {
            for j in (old + 1)..(old + w) {
                if !is_valid(pi[j]) {
                    continue;
                }
                valid -= 1;
                if pi[old] < pi[j] {
                    concordant -= 1;
                }
            }
        }
        if is_valid(pi[new]) {
            for i in k..new {
                if !is_valid(pi[i]) {
                    continue;
                }
                valid += 1;
                if pi[i] < pi[new] {
                    concordant += 1;
                }
            }
        }
        push(concordant, valid, &mut sum, &mut n_valid_windows);
    }

    if n_valid_windows == 0 {
        f32::NAN
    } else {
        sum / n_valid_windows as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Fisher–Yates shuffle (xorshift32) — no external rng dep,
    /// reproducible on every platform.
    fn seeded_shuffle(n: usize, seed: u32) -> Vec<u32> {
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut v: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            let j = (next() as usize) % (i + 1);
            v.swap(i, j);
        }
        v
    }

    #[test]
    fn identity_is_perfect() {
        let pi: Vec<u32> = (0..16u32).collect();
        assert_eq!(local_ar_ness(&pi), 1.0);
        assert_eq!(global_ar_ness(&pi), 1.0);
    }

    #[test]
    fn reversed_is_zero() {
        let pi: Vec<u32> = (0..16u32).rev().collect();
        assert_eq!(local_ar_ness(&pi), 0.0);
        assert_eq!(global_ar_ness(&pi), 0.0);
    }

    #[test]
    fn degenerate_lengths_are_nan() {
        assert!(local_ar_ness(&[]).is_nan());
        assert!(global_ar_ness(&[]).is_nan());
        assert!(local_ar_ness(&[7]).is_nan());
        assert!(global_ar_ness(&[7]).is_nan());
        // All-sentinel: no valid pair survives exclusion.
        assert!(local_ar_ness(&[UNMASKED_NEVER, UNMASKED_NEVER]).is_nan());
        assert!(global_ar_ness(&[UNMASKED_NEVER, UNMASKED_NEVER]).is_nan());
        assert!(global_ar_ness_windowed(&[UNMASKED_NEVER; 8], 4).is_nan());
        // Window < 2 has no pair per window.
        assert!(global_ar_ness_windowed(&[0, 1, 2], 1).is_nan());
        assert!(global_ar_ness_windowed(&[0, 1, 2], 0).is_nan());
    }

    #[test]
    fn two_element_orders() {
        assert_eq!(local_ar_ness(&[0, 1]), 1.0);
        assert_eq!(global_ar_ness(&[0, 1]), 1.0);
        assert_eq!(local_ar_ness(&[1, 0]), 0.0);
        assert_eq!(global_ar_ness(&[1, 0]), 0.0);
        // Tie (parallel commit of both positions): discordant by contract.
        assert_eq!(local_ar_ness(&[0, 0]), 0.0);
        assert_eq!(global_ar_ness(&[0, 0]), 0.0);
    }

    #[test]
    fn ties_are_discordant() {
        // π = [0,0,1]: pair (0,1) tie→discordant, (0,2) ✓, (1,2) ✓.
        let pi = [0u32, 0, 1];
        assert!((local_ar_ness(&pi) - 0.5).abs() < 1e-6);
        assert!((global_ar_ness(&pi) - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn block_swap_hand_computed() {
        // Two blocks of 4 decoded back-to-back, in-order within each block:
        // π = [4,5,6,7, 0,1,2,3].
        // ALR: 6 concordant of 7 adjacent pairs (only the block boundary
        // breaks the run) = 6/7.
        // AGR: within-block pairs 6+6 concordant; all 16 cross-block pairs
        // discordant → 12/28 = 3/7.
        let pi = [4u32, 5, 6, 7, 0, 1, 2, 3];
        assert!((local_ar_ness(&pi) - 6.0 / 7.0).abs() < 1e-6);
        assert!((global_ar_ness(&pi) - 3.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn parallel_block_decode_is_anti_ar() {
        // Two blocks of 4, each decoded fully parallel in one step: block 1
        // at step 1, block 0 at step 0 → π = [1,1,1,1, 0,0,0,0].
        let pi = [1u32, 1, 1, 1, 0, 0, 0, 0];
        // ALR: within-block ties discordant (6), boundary (1→0) also
        // discordant → 0/7. AGR: 12 within-block ties + all 16 cross-block
        // (1 < 0 false) discordant → 0/28.
        assert_eq!(local_ar_ness(&pi), 0.0);
        assert_eq!(global_ar_ness(&pi), 0.0);
    }

    #[test]
    fn random_permutation_is_near_half() {
        // E[ALR] = E[AGR] = 0.5 for a uniform random permutation; at L=1024
        // the sampling std is ~0.016 (AGR, Kendall) / ~0.016 (ALR) — the
        // [0.4, 0.6] band is a ~6σ envelope.
        let pi = seeded_shuffle(1024, 0x602_c0de);
        let alr = local_ar_ness(&pi);
        let agr = global_ar_ness(&pi);
        assert!((0.4..0.6).contains(&alr), "ALR outside random band: {alr}");
        assert!((0.4..0.6).contains(&agr), "AGR outside random band: {agr}");
    }

    #[test]
    fn sentinel_pairs_are_excluded() {
        // Sentinel in the middle: only pair (0,2) survives → AGR 1.0; no
        // valid adjacent pair → ALR NaN.
        let pi = [0u32, UNMASKED_NEVER, 2];
        assert!(local_ar_ness(&pi).is_nan());
        assert_eq!(global_ar_ness(&pi), 1.0);

        // Sentinel on the wing: adjacent pair (1,2) survives ✓.
        let pi = [UNMASKED_NEVER, 1, 2];
        assert_eq!(local_ar_ness(&pi), 1.0);
        assert_eq!(global_ar_ness(&pi), 1.0);
    }

    #[test]
    fn windowed_full_width_equals_global() {
        let cases: Vec<Vec<u32>> = vec![
            (0..32u32).collect(),
            (0..32u32).rev().collect(),
            seeded_shuffle(32, 7),
            vec![1, 1, 1, 0, 0, 0, 2, 2],
        ];
        for pi in cases {
            let full = global_ar_ness_windowed(&pi, pi.len());
            let global = global_ar_ness(&pi);
            assert!((full - global).abs() < 1e-6, "full-width ≠ global: {pi:?}");
        }
    }

    #[test]
    fn windowed_two_equals_local() {
        // Window of 2 = one pair per window → the mean over windows is
        // exactly the adjacent-pair statistic.
        let cases: Vec<Vec<u32>> = vec![
            (0..24u32).collect(),
            seeded_shuffle(24, 99),
            vec![5, 3, 3, 9, 0, 7, 7, 1],
        ];
        for pi in cases {
            let w2 = global_ar_ness_windowed(&pi, 2);
            let alr = local_ar_ness(&pi);
            assert!((w2 - alr).abs() < 1e-6, "w2 ≠ ALR: {pi:?}");
        }
    }

    #[test]
    fn windowed_hand_computed_mid_width() {
        // π = [4,5,6,7,0,1,2,3], window 4: windows A=[4,5,6,7],
        // B=[5,6,7,0], C=[6,7,0,1], D=[7,0,1,2], E=[0,1,2,3].
        // AGRs: A=6/6, B=3/6, C=2/6, D=3/6, E=6/6 → mean = 20/30 = 2/3.
        let pi = [4u32, 5, 6, 7, 0, 1, 2, 3];
        let expected = (1.0 + 0.5 + 2.0 / 6.0 + 0.5 + 1.0) / 5.0;
        assert_eq!(expected, 2.0 / 3.0);
        assert!((global_ar_ness_windowed(&pi, 4) - expected).abs() < 1e-6);
    }

    #[test]
    fn windowed_skips_all_sentinel_windows() {
        // Windows [MAX,MAX] have no valid pairs and are skipped; the two
        // mixed windows contribute their committed sub-order.
        let pi = [UNMASKED_NEVER, UNMASKED_NEVER, 0, 1];
        let w = global_ar_ness_windowed(&pi, 2);
        // Valid windows: [1,2]→(MAX,0) excluded→NaN? No — pair touches
        // sentinel → no valid pairs → window skipped. [2,3]→(0,1) ✓ =1.0.
        assert_eq!(w, 1.0);
    }
}

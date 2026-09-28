//! Bench 839 — `kron_tile` GOAT G2: the Kronecker-factored tile apply against
//! the dense `n² × n²` matvec it replaces, and against the WHT butterfly.
//!
//! Issue 839 T4. Run:
//!
//! ```text
//! cargo test --release --features kron_tile --test bench_839_kron_tile_goat -- --nocapture
//! ```
//!
//! # Three arms, and the third is the one that can embarrass the primitive
//!
//! - **G2a** `kron_apply` vs `dense_matvec_into` at `n = 32` (so the dense
//!   operator is `1024 × 1024`, Research 569's actual stage). The FLOP ratio is
//!   `n⁴ / 2n³ = n/2 = 16×`; the measured ratio is also carrying a 4 MiB working
//!   set on the dense side against 8 KiB on ours, so it can exceed the FLOP
//!   ratio without that being a better kernel.
//! - **G2d** the **3-stage Monarch** composition — Issue 839 T4's literal ask —
//!   against the same dense matvec. Three stages with fixed permutations
//!   between them, because *without* Π the composition collapses to one stage
//!   at multiplied factors (pinned by `stages_collapse_without_a_permutation`)
//!   and the arm would measure the wrong operator while looking right. FLOP
//!   ratio `n⁴ / 3·2n³ = n/6` = 5.33×.
//! - **G2b** `wht_apply_tiles` vs `kron_apply` at the same operator. FLOP ratio
//!   `2n³ / (2n² log₂ n) = n / log₂ n = 6.4×` at `n = 32`. This is the arm that
//!   says whether the delegating fast path earns its dispatch.
//! - **G2c** `kron_apply` vs the shipped **ternary** dense matvec
//!   (`katgpt_types::simd::simd_ternary_matvec`) at the same `1024 × 1024`
//!   operator — the honest competitor, because a 1-bit-and-a-half dense matvec
//!   is the *other* way this workspace already buys a cheap 1024-channel stage.
//!   ⚠ It is **not** a matched-parameter comparison and is not reported as one:
//!   ternary dense carries `n⁴ = 1M` trits (~197 KiB), the Kronecker stage
//!   `2n² = 2048` f32 (8 KiB). Equal FLOPs would be the claim to make if the
//!   arms had equal parameters, and they are 512× apart, so this arm is
//!   reported as **throughput at a fixed operator size** and nothing more.
//!
//! # Why the interleaved harness and not two sequential loops
//!
//! Every ratio here is two timed arms compared, which is `AGENTS.md`'s
//! sequential-A/B class — measured at **+5.2% and +21.7%** for two arms of the
//! *same* work thirty seconds apart (Issue 723 T5), and the subject of a live
//! census (Issues 833/834) that found 152 instances. Writing a 153rd would be
//! perverse, so this target ADOPTS `tests/common/ab_timing.rs`: interleaved
//! `(a, b)` chunks, one ratio per pair, median across pairs, and a loud named
//! failure when an arm reads zero because the optimiser deleted it.
//!
//! ⚠ **Orientation is the thing to get right** (`AGENTS.md`: "backwards inverts
//! the bar silently"). `AbRatio::median` is `b / a` over *times*, so every arm
//! below puts the **candidate in `b`** and reports `speedup = a_ns / b_ns`. The
//! bars are stated as speedups and asserted on that quotient, never on the
//! median directly.

#![cfg(feature = "kron_tile")]

use katgpt_core::linalg::kron_tile::{
    KronScratch, dense_matvec_into, is_permutation, kron_apply, kron_dense_into, permute_into,
    wht_apply_tiles, wht_factor_into,
};
// Reached through katgpt-core's re-exports (`pub use katgpt_types::simd` /
// `pub use katgpt_types as types`): the ROOT crate has no direct katgpt-types
// dependency, and adding one for a bench would widen the dep graph to buy a
// shorter path.
use katgpt_core::simd::{simd_level, simd_ternary_matvec};
use katgpt_core::types::TernaryWeights;
use std::hint::black_box;

#[path = "common/ab_timing.rs"]
mod ab_timing;
use ab_timing::ab_median_ratio;

const N: usize = 32; // tile width — 1024 channels as a 32×32 tile
const NN: usize = N * N; // 1024 = the dense operator's dimension
const TILES: usize = 1;

/// Deterministic LCG. Seeded: `scripts/global_rng_gate.py` reds on an unseeded
/// global draw, and a benchmark whose fixture changes run to run is not a
/// benchmark.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        2.0 * (((self.0 >> 33) as f32) / (1u64 << 31) as f32) - 1.0
    }
}

fn random_vec(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next()).collect()
}

/// G2a — the Kronecker stage against the dense operator it replaces.
///
/// Bar: **≥ 8× speedup**, half the 16× FLOP ratio. The slack is for the dense
/// arm's ISA-dispatched SIMD (the baseline is `katgpt-types`' real
/// `simd_matvec`, not a scalar loop) and for the box; a bar set at the FLOP
/// ratio itself would be a bar on the memory system.
#[test]
fn g2a_kron_beats_dense_matvec() {
    let mut rng = Lcg::new(0x0839_2001);
    let a = random_vec(&mut rng, NN);
    let b = random_vec(&mut rng, NN);
    let x0 = random_vec(&mut rng, NN);

    let mut dense = vec![0.0f32; NN * NN];
    kron_dense_into(&a, &b, N, &mut dense);
    let mut dense_out = vec![0.0f32; NN];

    let mut kron_buf = x0.clone();
    let mut scratch = KronScratch::with_capacity(N);

    let r = ab_median_ratio(
        11,
        50,
        5,
        // a = baseline: the dense 1024×1024 matvec.
        |_| {
            dense_matvec_into(black_box(&dense), NN, black_box(&x0), &mut dense_out);
            black_box(dense_out[0]);
        },
        // b = candidate: the two small GEMMs.
        |_| {
            kron_buf.copy_from_slice(&x0);
            kron_apply(
                black_box(&a),
                black_box(&b),
                N,
                TILES,
                &mut kron_buf,
                &mut scratch,
            );
            black_box(kron_buf[0]);
        },
    );
    r.report("G2a kron vs dense");

    let speedup = r.a_ns_per_iter() / r.b_ns_per_iter();
    println!(
        "   G2a: speedup {speedup:.2}x (FLOP ratio n/2 = {:.1}x; params 2n^2={} vs n^4={})",
        N as f64 / 2.0,
        2 * NN,
        NN * NN
    );
    assert!(
        speedup >= 8.0,
        "G2a FAIL: kron_apply is only {speedup:.2}x the dense matvec, bar 8x \
         (median b/a {:.4}, rounds {:.4}..{:.4})",
        r.median,
        r.min(),
        r.max()
    );
}

/// G2b — does the delegating WHT fast path earn its dispatch?
///
/// Bar: **≥ 2× speedup** against the generic path at the same operator. The
/// FLOP ratio is 6.4×, but the butterfly's column pass is a strided
/// gather/scatter while the generic path is two contiguous GEMMs, so a large
/// part of the arithmetic win is spent on memory order. 2× is the threshold
/// below which the fast path would not be worth the second code path at all —
/// which is the decision this arm exists to inform, not a prediction.
#[test]
fn g2b_wht_fast_path_beats_generic() {
    let mut rng = Lcg::new(0x0839_2002);
    let x0 = random_vec(&mut rng, NN);
    let mut w = vec![0.0f32; NN];
    wht_factor_into(N, &mut w);

    let mut generic_buf = x0.clone();
    let mut fast_buf = x0.clone();
    let mut scratch = KronScratch::with_capacity(N);
    let mut scratch_b = KronScratch::with_capacity(N);

    let r = ab_median_ratio(
        11,
        200,
        20,
        // a = baseline: the generic two-GEMM path at A = B = W.
        |_| {
            generic_buf.copy_from_slice(&x0);
            kron_apply(
                black_box(&w),
                black_box(&w),
                N,
                TILES,
                &mut generic_buf,
                &mut scratch,
            );
            black_box(generic_buf[0]);
        },
        // b = candidate: the butterfly.
        |_| {
            fast_buf.copy_from_slice(&x0);
            let ok = wht_apply_tiles(N, TILES, &mut fast_buf, &mut scratch_b);
            assert!(ok, "n=32 must dispatch");
            black_box(fast_buf[0]);
        },
    );
    r.report("G2b wht vs generic");

    let speedup = r.a_ns_per_iter() / r.b_ns_per_iter();
    println!(
        "   G2b: speedup {speedup:.2}x (FLOP ratio n/log2(n) = {:.1}x)",
        N as f64 / (N as f64).log2()
    );
    assert!(
        speedup >= 2.0,
        "G2b FAIL: the WHT fast path is only {speedup:.2}x the generic path, bar 2x \
         — a second code path this close to the generic one should be deleted, not kept \
         (median b/a {:.4}, rounds {:.4}..{:.4})",
        r.median,
        r.min(),
        r.max()
    );
}

/// G2c — against the ternary dense matvec, the workspace's other cheap
/// 1024-channel stage.
///
/// **Reported, not barred.** The two arms are 512× apart in parameters (8 KiB
/// of f32 factors against ~197 KiB of trits), so a bar either way would be a
/// claim about a comparison nobody made: losing to ternary dense would not make
/// the Kronecker stage worse at its own job, and beating it is not a
/// matched-parameter result. The number is here because the *deployment*
/// question — which structured stage to spend a millisecond on — needs both
/// measured on one box, and because a GOAT record that quietly omits the
/// strongest competitor is the kind this repo has learned to distrust.
#[test]
fn g2c_kron_against_ternary_dense_reported() {
    let mut rng = Lcg::new(0x0839_2003);
    let a = random_vec(&mut rng, NN);
    let b = random_vec(&mut rng, NN);
    let x0 = random_vec(&mut rng, NN);

    // Ternary weights over the same 1024×1024 operator: sign of a random draw,
    // with a dead band so all three trits occur.
    let trits: Vec<i8> = (0..NN * NN)
        .map(|_| {
            let v = rng.next();
            if v > 0.33 {
                1
            } else if v < -0.33 {
                -1
            } else {
                0
            }
        })
        .collect();
    let tw = TernaryWeights::pack_from_i8(&trits, NN, NN).expect("ternary pack");
    let mut tern_out = vec![0.0f32; NN];

    let mut kron_buf = x0.clone();
    let mut scratch = KronScratch::with_capacity(N);

    let r = ab_median_ratio(
        11,
        50,
        5,
        // a = baseline: ternary dense.
        |_| {
            simd_ternary_matvec(black_box(&tw), black_box(&x0), &mut tern_out);
            black_box(tern_out[0]);
        },
        // b = candidate: the Kronecker stage.
        |_| {
            kron_buf.copy_from_slice(&x0);
            kron_apply(
                black_box(&a),
                black_box(&b),
                N,
                TILES,
                &mut kron_buf,
                &mut scratch,
            );
            black_box(kron_buf[0]);
        },
    );
    r.report("G2c kron vs ternary dense");
    // Disclose the BASELINE's dispatch on the verdict line. `simd_ternary_matvec`
    // selects its arm from a runtime probe, and a ternary number taken on the
    // scalar fallback is a measurement of the fallback — the "a lane compiles
    // what it names" hazard, one layer down, where the arm is chosen at run time
    // rather than at compile time. Without this line the reader cannot tell
    // which kernel produced the 124 µs.
    println!(
        "   G2c: baseline dispatch simd_level() = {:?}",
        simd_level()
    );

    let speedup = r.a_ns_per_iter() / r.b_ns_per_iter();
    println!(
        "   G2c: kron is {speedup:.2}x ternary dense at a matched OPERATOR (1024x1024); \
         parameters are NOT matched (2n^2={} f32 vs n^4={} trits) — reported, not barred",
        2 * NN,
        NN * NN
    );
    // The only assertion is that the instrument worked: a zero arm means the
    // optimiser deleted the work, which `ab_median_ratio` already fails on, and
    // a non-finite speedup means the report is meaningless.
    assert!(
        speedup.is_finite() && speedup > 0.0,
        "G2c instrument FAIL: speedup {speedup} is not a measurement"
    );
}

/// G2d — the three-stage Monarch composition against the dense operator, which
/// is what Issue 839 T4 actually asks for.
///
/// G2a measures ONE stage, which is the primitive's own claim; this measures the
/// construction Research 569 builds out of it: `M₃ Π₂ M₂ Π₁ M₁`, three
/// Kronecker stages with fixed permutations between them. The permutations are
/// not decoration — one stage mixes only within rows and within columns, and
/// `stages_collapse_without_a_permutation` pins that dropping Π turns three
/// stages into one with multiplied factors.
///
/// Bar: **≥ 3× speedup** against a 5.33× FLOP ratio. The slack is for the two
/// gather passes, which are pure memory traffic and buy no arithmetic.
#[test]
fn g2d_three_stage_monarch_beats_dense_matvec() {
    let mut rng = Lcg::new(0x0839_2005);
    let f: Vec<Vec<f32>> = (0..6).map(|_| random_vec(&mut rng, NN)).collect();
    let x0 = random_vec(&mut rng, NN);

    // Two fixed permutations of the 1024 channels. Odd stride mod an even
    // length is a bijection; asserted rather than assumed, ONCE, here at setup
    // — which is the split `is_permutation` exists for.
    let pi1: Vec<u32> = (0..NN).map(|i| ((i * 37 + 11) % NN) as u32).collect();
    let pi2: Vec<u32> = (0..NN).map(|i| ((i * 101 + 7) % NN) as u32).collect();
    assert!(is_permutation(&pi1, NN) && is_permutation(&pi2, NN));

    let mut dense = vec![0.0f32; NN * NN];
    kron_dense_into(&f[0], &f[1], N, &mut dense);
    let mut dense_out = vec![0.0f32; NN];

    let mut buf_a = x0.clone();
    let mut buf_b = vec![0.0f32; NN];
    let mut scratch = KronScratch::with_capacity(N);

    let r = ab_median_ratio(
        11,
        50,
        5,
        // a = baseline: one dense 1024x1024 matvec.
        |_| {
            dense_matvec_into(black_box(&dense), NN, black_box(&x0), &mut dense_out);
            black_box(dense_out[0]);
        },
        // b = candidate: M3 P2 M2 P1 M1, alternating the two buffers.
        |_| {
            buf_a.copy_from_slice(&x0);
            kron_apply(&f[0], &f[1], N, TILES, &mut buf_a, &mut scratch);
            permute_into(&buf_a, &pi1, &mut buf_b);
            kron_apply(&f[2], &f[3], N, TILES, &mut buf_b, &mut scratch);
            permute_into(&buf_b, &pi2, &mut buf_a);
            kron_apply(&f[4], &f[5], N, TILES, &mut buf_a, &mut scratch);
            black_box(buf_a[0]);
        },
    );
    r.report("G2d 3-stage monarch vs dense");

    let speedup = r.a_ns_per_iter() / r.b_ns_per_iter();
    println!(
        "   G2d: speedup {speedup:.2}x (FLOP ratio n/6 = {:.2}x; params 6n^2={} vs n^4={})",
        N as f64 / 6.0,
        6 * NN,
        NN * NN
    );
    assert!(
        speedup >= 3.0,
        "G2d FAIL: the 3-stage Monarch apply is only {speedup:.2}x the dense matvec, bar 3x          (median b/a {:.4}, rounds {:.4}..{:.4})",
        r.median,
        r.min(),
        r.max()
    );
}

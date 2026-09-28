//! GOAT gate — per-shape plasma dispatch (Issue 843 T4, owner call gate D1,
//! 2026-09-19): f32 below the L3 boundary, ternary above.
//!
//! Run: `cargo test --release --features plasma_path --test
//! bench_843_plasma_dispatch_goat -- --nocapture`
//!
//! - **G1 correctness** — the dispatch is bit-identical to whichever kernel
//!   the policy selects, on BOTH sides of a forced boundary (`_with_l3`
//!   forms make the test box-independent), and the public cached wrapper
//!   agrees with the explicit-boundary form.
//! - **G2 perf** — at 1024×1024 (the shape the issue was filed on, below L3
//!   on any box worth naming) with the boundary forced to 1 GiB so the f32
//!   arm is selected deterministically: the dispatch must be ≥1.5× faster
//!   than the pure ternary kernel (measured 2.26× AVX2 / 2.16× NEON in
//!   Issue 843 §T2/§T1 — the bar is half the effect, the house slack
//!   convention) and within 1.10× of the pure f32 kernel (the dispatch cost
//!   is one cached policy read + branch).
//! - **G3 no-regression** — the dispatch is a NEW policy primitive; nothing
//!   existing routes through it (the flag's kernels are untouched), so the
//!   no-regression claim is structural + the unchanged katgpt-types suite.
//! - **G4 alloc-free** — structural: slices in, slices out, no formatting,
//!   no vectors; the policy read is one OnceLock load.

#[path = "common/ab_timing.rs"]
mod ab_timing;

use katgpt_core::simd::{
    plasma_prefers_ternary, plasma_prefers_ternary_with_l3, simd_matvec,
    simd_matvec_plasma_dispatch, simd_matvec_plasma_dispatch_with_l3, simd_ternary_matvec,
};
use katgpt_core::types::TernaryWeights;

/// Deterministic fixture (the Issue-843 house shape: LCG draw thresholded
/// by quantize_from_f32 so all three trit values occur).
fn fixture(m: usize) -> (Vec<f32>, TernaryWeights, Vec<f32>) {
    let mut s = 0x843_843u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let dense: Vec<f32> = (0..m * m).map(|_| next()).collect();
    let tw = TernaryWeights::quantize_from_f32(&dense, m, m);
    let x: Vec<f32> = (0..m).map(|_| next()).collect();
    (dense, tw, x)
}

#[test]
fn g1_dispatch_is_bit_identical_to_the_selected_kernel() {
    let (dense, tw, x) = fixture(256);
    let mut y = vec![0.0f32; 256];
    let mut y_ref = vec![0.0f32; 256];

    // Below the boundary (forced generous L3): the f32 arm, bit-identical.
    simd_matvec_plasma_dispatch_with_l3(&dense, &tw, &x, &mut y, 1 << 30);
    simd_matvec(&mut y_ref, &dense, &x, tw.rows, tw.cols);
    assert_eq!(y, y_ref, "below L3 the dispatch must BE the f32 kernel");

    // Above the boundary (forced tiny L3): the ternary arm, bit-identical.
    simd_matvec_plasma_dispatch_with_l3(&dense, &tw, &x, &mut y, 1);
    katgpt_core::simd_ternary_matvec(&tw, &x, &mut y_ref);
    assert_eq!(y, y_ref, "above L3 the dispatch must BE the ternary kernel");

    // The public cached wrapper agrees with the explicit form wherever the
    // cached probe lands (256² = 256 KiB f32 operand — below every plausible
    // L3 and below the 32 MiB fallback, so the f32 arm).
    simd_matvec_plasma_dispatch(&dense, &tw, &x, &mut y);
    simd_matvec(&mut y_ref, &dense, &x, tw.rows, tw.cols);
    assert_eq!(y, y_ref, "cached wrapper on a small shape = f32 arm");
    assert!(!plasma_prefers_ternary(256, 256));
}

#[test]
fn g1_policy_boundary_is_the_f32_operand() {
    assert!(!plasma_prefers_ternary_with_l3(512, 512, 1024 * 1024));
    assert!(plasma_prefers_ternary_with_l3(512, 512, 1024 * 1024 - 1));
    // The served shape (768×3072 → 9 MiB f32 operand) is f32-side under the
    // 32 MiB fallback — the regime Issue 843 measured at 2.10×.
    assert!(!plasma_prefers_ternary_with_l3(768, 3072, 32 * 1024 * 1024));
}

#[test]
fn g2_dispatch_beats_pure_ternary_below_the_boundary() {
    use std::hint::black_box;
    let m = 1024usize;
    let (dense, tw, x) = fixture(m);

    // Force the f32 arm deterministically (1 GiB boundary) so the verdict is
    // box-independent — this is exactly the regime the issue measured.
    // One output buffer PER ARM: the harness holds both FnMut closures live
    // simultaneously, so they cannot share a `&mut`.
    let l3 = 1 << 30;
    let mut y_a = vec![0.0f32; m];
    let mut y_b = vec![0.0f32; m];
    let ab = ab_timing::ab_median_ratio(
        7,
        8,
        20,
        |_i| {
            simd_matvec_plasma_dispatch_with_l3(
                black_box(&dense),
                black_box(&tw),
                black_box(&x),
                black_box(&mut y_a),
                l3,
            );
        },
        |_i| {
            simd_ternary_matvec(black_box(&tw), black_box(&x), black_box(&mut y_b));
        },
    );
    ab.report("G2 dispatch-vs-pure-ternary @1024² (f32 arm forced)");
    // ratio = b/a = pure_ternary/dispatch — dispatch ≥1.5× faster means the
    // ratio is ≥ 1.5 (measured ~2.16–2.26× across both arches).
    let min_ratio = 1.5;
    assert!(
        ab.median >= min_ratio,
        "dispatch must be >= {min_ratio}× faster than the pure ternary kernel at 1024² \
         below the boundary (Issue 843 measured 2.16–2.26×), got ratio {:.3} \
         (range {:.3}..{:.3})",
        ab.median,
        ab.min(),
        ab.max(),
    );
}

#[test]
fn g2_dispatch_overhead_vs_pure_dense_is_one_branch() {
    use std::hint::black_box;
    let m = 1024usize;
    let (dense, tw, x) = fixture(m);
    let l3 = 1 << 30;
    let mut y_a = vec![0.0f32; m];
    let mut y_b = vec![0.0f32; m];
    let ab = ab_timing::ab_median_ratio(
        7,
        8,
        20,
        |_i| {
            simd_matvec_plasma_dispatch_with_l3(
                black_box(&dense),
                black_box(&tw),
                black_box(&x),
                black_box(&mut y_a),
                l3,
            );
        },
        |_i| {
            simd_matvec(black_box(&mut y_b), black_box(&dense), black_box(&x), m, m);
        },
    );
    ab.report("G2 dispatch-overhead-vs-pure-dense @1024²");
    // ratio = b/a = pure_dense/dispatch — the dispatch may be at most 10%
    // slower than the kernel it delegates to (one cached policy read).
    let max_overhead = 1.10;
    assert!(
        ab.median >= 1.0 / max_overhead,
        "dispatch overhead vs the pure f32 kernel must stay under 10% (ratio {:.3}, \
         range {:.3}..{:.3})",
        ab.median,
        ab.min(),
        ab.max(),
    );
}

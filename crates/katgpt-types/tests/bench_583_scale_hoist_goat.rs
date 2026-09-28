//! Issue 583 A/B: group scale **hoisted** vs **folded** in the bit-plane NEON kernel.
//!
//! Run:
//! ```bash
//! cargo test -p katgpt-types --features ternary_group_scale,plasma_path --release \
//!   --test bench_583_scale_hoist_goat -- --nocapture
//! ```
//!
//! # The question
//!
//! `simd_ternary_group_matvec` folds the group scale into every sign vector
//! (`±1 → ±scale`) — 32 `vmulq` per 128-weight group — so its 4 accumulators can
//! span the whole row with no per-group reset and no per-group horizontal sum.
//! Issue 578 chose that on the reasoning that a reset + hsum serializes the
//! pipeline, and measured the kernel at 1.29–1.31× the row-scale kernel.
//!
//! `simd_ternary_group_matvec_hoisted` does the opposite: accumulate the group
//! unscaled, one `vaddvq`, one `vmulq`. **2 ops per group instead of 32.**
//!
//! Bench 582 (the trit tier) measured the hoisted association ~10% faster on an
//! otherwise-comparable kernel, which is what motivated this A/B. This test
//! settles it on the *same* container, so decode cost cancels out entirely and
//! the only difference is where the scale is applied.
//!
//! # Gates
//!
//! - **G1** hoisted stays within 1e-6 relative of `ternary_group_matvec_scalar`
//!   on the ragged shapes, and agrees with the folded kernel within 1e-5.
//! - **G2** ≥ 1.10× over the fold on the 3-shape median, or the fold stays.
//! - **G4** 0 allocations per call.

#![cfg(all(feature = "ternary_group_scale", feature = "plasma_path"))]

use std::time::Instant;

use katgpt_types::TernaryGroupWeights;
use katgpt_types::simd::{
    simd_ternary_group_matvec_folded, simd_ternary_group_matvec_hoisted,
    ternary_group_matvec_scalar,
};

struct CountingAllocator;

thread_local! {
    static ALLOC_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOC_COUNT.with(|c| c.set(c.get().wrapping_add(1)));
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

#[inline]
fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOC_COUNT.with(|c| c.get());
    let r = f();
    let after = ALLOC_COUNT.with(|c| c.get());
    (r, after - before)
}

fn pseudo(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn dense_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..rows * cols).map(|_| pseudo(&mut s)).collect()
}

fn median_ns(reps: usize, inner: usize, mut f: impl FnMut()) -> f64 {
    const MAX_REPS: usize = 32;

    for _ in 0..inner {
        f();
    }
    assert!(reps <= MAX_REPS, "reps={reps} exceeds MAX_REPS={MAX_REPS}");
    let mut samples = [0.0f64; MAX_REPS];
    for slot in samples.iter_mut().take(reps) {
        let t = Instant::now();
        for _ in 0..inner {
            f();
        }
        *slot = t.elapsed().as_nanos() as f64 / inner as f64;
    }
    samples[..reps].sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    samples[reps / 2]
}

const SHAPES: [(usize, usize); 3] = [(512, 512), (1024, 1024), (512, 5120)];

#[test]
fn g2_hoisted_vs_folded() {
    println!("\n── Issue 583 G2: ns/call (median of 9 x 20 calls) ──");
    println!(
        "{:>12} {:>14} {:>14} {:>12}",
        "shape", "hoisted", "folded", "speedup"
    );

    let mut speedups = Vec::with_capacity(SHAPES.len());
    for &(rows, cols) in &SHAPES {
        let src = dense_matrix(rows, cols, 0x583);
        let gw = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        let mut s = 0xBEEF_u64;
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
        let mut y = vec![0.0f32; rows];

        let t_hoist = median_ns(9, 20, || simd_ternary_group_matvec_hoisted(&gw, &x, &mut y));
        let t_fold = median_ns(9, 20, || simd_ternary_group_matvec_folded(&gw, &x, &mut y));
        let speedup = t_fold / t_hoist;
        speedups.push(speedup);
        println!("{rows:>5}x{cols:<6} {t_hoist:>14.0} {t_fold:>14.0} {speedup:>11.2}x");
    }

    let median = {
        let mut s = speedups.clone();
        s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        s[s.len() / 2]
    };
    println!(
        "\nmedian speedup {median:.2}x (gate >= 1.10x)\n\
         Mechanism: 1 vmulq + 1 vaddvq per 128-weight group instead of 32 vmulq."
    );
    assert!(
        median >= 1.10,
        "G2 FAIL: {median:.2}x — keep the fold and record the negative result"
    );
}

#[test]
fn g1_hoisted_matches_scalar_and_folded() {
    // Shapes mirror the existing `neon_matches_scalar_reference` coverage:
    // ragged group, ragged block, sub-4 tail.
    for &(rows, cols) in &[
        (4usize, 128usize),
        (3, 256),
        (5, 300),
        (2, 133),
        (7, 64),
        (3, 13),
        (512, 5120),
    ] {
        let src = dense_matrix(rows, cols, 0x1583);
        let gw = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        let mut s = 0xF00D_u64;
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
        let mut y_hoist = vec![0.0f32; rows];
        let mut y_fold = vec![0.0f32; rows];
        let mut y_scalar = vec![0.0f32; rows];

        simd_ternary_group_matvec_hoisted(&gw, &x, &mut y_hoist);
        simd_ternary_group_matvec_folded(&gw, &x, &mut y_fold);
        ternary_group_matvec_scalar(&gw, &x, &mut y_scalar);

        // Error is measured against the magnitude of the computation, not the
        // possibly-cancelled result: a row of a 5120-column matvec is 40 group
        // sums of magnitude ~3, so `max(|want|, 1.0)` would flag a rare
        // near-zero row as a failure while letting large rows off lightly. See
        // the same fix in bench_582's `assert_close_rms`.
        let rms = (y_scalar
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            / y_scalar.len() as f64)
            .sqrt() as f32;
        for r in 0..rows {
            let denom = y_scalar[r].abs().max(rms).max(f32::MIN_POSITIVE);
            let d_hoist = (y_hoist[r] - y_scalar[r]).abs() / denom;
            let d_fold = (y_fold[r] - y_scalar[r]).abs() / denom;
            assert!(
                d_hoist < 1e-6,
                "{rows}x{cols} row {r}: hoisted {} vs scalar {} (rel {d_hoist:.2e})",
                y_hoist[r],
                y_scalar[r]
            );
            assert!(
                (y_hoist[r] - y_fold[r]).abs() / denom < 1e-5,
                "{rows}x{cols} row {r}: hoisted {} vs folded {}",
                y_hoist[r],
                y_fold[r]
            );
            // Informational: the hoisted association matches the reference's own
            // (one scale per group), so it should be no further from scalar than
            // the fold is. Not asserted as a gate — f32 rounding is not
            // monotone in association — but printed when it is violated.
            if d_hoist > d_fold * 4.0 && d_hoist > 1e-8 {
                println!(
                    "note {rows}x{cols} row {r}: hoisted rel {d_hoist:.2e} > folded rel {d_fold:.2e}"
                );
            }
        }
    }
}

#[test]
fn g4_hoisted_allocates_nothing() {
    let (rows, cols) = (512usize, 5120usize);
    let src = dense_matrix(rows, cols, 0x4583);
    let gw = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
    let mut s = 0xC0DE_u64;
    let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
    let mut y = vec![0.0f32; rows];

    simd_ternary_group_matvec_hoisted(&gw, &x, &mut y);
    let (_, allocs) = alloc_delta(|| {
        for _ in 0..1000 {
            simd_ternary_group_matvec_hoisted(&gw, &x, &mut y);
        }
    });
    println!("\n── Issue 583 G4 ──\nhoisted {allocs} allocs / 1000 calls");
    assert_eq!(allocs, 0);
}

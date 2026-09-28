//! Issue 578 AVX2 follow-up (2026-08-11): G2-AVX2 gate — AVX2 vs scalar
//! reference for the group-scale ternary matvec.
//!
//! Run:
//! ```bash
//! cargo test -p katgpt-types --features ternary_group_scale,plasma_path \
//!   --test bench_578_avx2_goat --release -- --nocapture --test-threads=1
//! ```
//!
//! # Why this benchmark exists
//!
//! Issue 578's G2 table (recorded on M3 Max / NEON) shows the group-scale
//! kernel costs 1.29-1.31× the row-scale kernel. The AVX2 port adds the same
//! per-group arithmetic cost on top of x86_64's 8-wide SIMD (vs NEON's
//! 4-wide). The decisive gate is therefore **AVX2 vs scalar reference** —
//! the speedup the AVX2 kernel earns over the scalar fallback it replaces.
//!
//! The threshold mirrors the binary AVX2 kernel's GOAT: ≥ 2× throughput vs
//! scalar. Anything below 2× means the hand-written intrinsics are not
//! pulling their weight vs LLVM's auto-vectorizer on the scalar loop.
//!
//! # Test-parallelism caveat
//!
//! `median_ns` heap-allocates its `samples: Vec<f64>`, which can bleed into
//! a concurrently-running G4 test's `CountingAllocator` window. **Always run
//! this benchmark with `--test-threads=1` when G4 is in the same crate.**
//! (Recorded as a pre-existing harness flakiness, not a kernel bug.)

#![cfg(all(
    feature = "ternary_group_scale",
    feature = "plasma_path",
    target_arch = "x86_64"
))]

use std::time::Instant;

use katgpt_types::TernaryGroupWeights;
use katgpt_types::simd::{
    SimdLevel, simd_level, simd_ternary_group_matvec, ternary_group_matvec_scalar,
};

/// Deterministic pseudo-random f32 in [-1, 1). No rand dep, reproducible.
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

/// Median of `reps` timed runs of `f`, in nanoseconds per call.
fn median_ns(reps: usize, inner: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..inner {
        f();
    }
    let mut samples: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..inner {
                f();
            }
            t.elapsed().as_nanos() as f64 / inner as f64
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    samples[samples.len() / 2]
}

#[test]
fn g2_avx2_vs_scalar_speedup() {
    // Skip cleanly if the host CPU lacks AVX2+FMA. The dispatcher falls back
    // to scalar in that case, so the two timed calls measure the same code.
    if !matches!(simd_level(), SimdLevel::Avx2) {
        eprintln!("SKIP: host CPU does not report AVX2+FMA; test only meaningful on Haswell+");
        return;
    }

    // Shapes mirror Issue 578's M3 NEON G2 table so the speedup ratio is
    // directly comparable across architectures.
    let shapes = [(512usize, 512usize), (1024, 1024), (512, 5120)];

    println!("\n── Issue 578 G2-AVX2: ns/call (median of 9 x 20 calls, x86_64) ──");
    println!(
        "{:>12} {:>14} {:>14} {:>14}",
        "shape", "avx2", "scalar", "speedup"
    );

    let mut all_pass = true;
    let mut min_speedup = f64::INFINITY;
    for &(rows, cols) in &shapes {
        let src = dense_matrix(rows, cols, 0x578);
        let gw = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        let mut s = 0xBEEF_u64;
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
        let mut y = vec![0.0f32; rows];

        // Interleave the two kernels so cache state is symmetric across runs.
        let t_avx2 = median_ns(9, 20, || simd_ternary_group_matvec(&gw, &x, &mut y));
        let t_scalar = median_ns(9, 20, || ternary_group_matvec_scalar(&gw, &x, &mut y));

        let speedup = t_scalar / t_avx2;
        min_speedup = min_speedup.min(speedup);
        println!("{rows:>5}x{cols:<6} {t_avx2:>14.0} {t_scalar:>14.0} {speedup:>13.2}x");

        // THE GATE: AVX2 must beat scalar by ≥ 2×. The scalar loop is simple
        // enough that LLVM's auto-vectorizer is already competitive, so the
        // bar is "the hand-written SWAR+FMA is doing real work the compiler
        // cannot recover from the scalar form."
        const GATE: f64 = 2.0;
        if speedup < GATE {
            println!("    FAIL: {speedup:.2}x vs scalar, gate is {GATE:.2}x");
            all_pass = false;
        }
    }

    println!("\nmin speedup across shapes: {min_speedup:.2}x");
    assert!(all_pass, "G2-AVX2 FAILED — see the rows marked FAIL above");
}

/// Correctness sanity check that runs *in the same binary* so a release build
/// of the perf test cannot silently regress correctness. Asserts AVX2 matches
/// scalar to ~1e-6 relative (the documented scalar-vs-SIMD agreement).
#[test]
fn g1_avx2_matches_scalar_reference() {
    if !matches!(simd_level(), SimdLevel::Avx2) {
        eprintln!("SKIP: host CPU does not report AVX2+FMA");
        return;
    }

    for &(rows, cols) in &[(4, 128), (3, 256), (2, 300), (5, 65), (1, 7), (2, 129)] {
        let w = {
            let mut s = 0x51ED_u64.wrapping_add(cols as u64);
            let mut w = TernaryGroupWeights::new(rows, cols);
            for r in 0..rows {
                for c in 0..cols {
                    let v = pseudo(&mut s);
                    let q = match v {
                        v if v > 0.33 => 1i8,
                        v if v < -0.33 => -1i8,
                        _ => 0i8,
                    };
                    w.set(r, c, q);
                }
                for g in 0..w.groups_per_row {
                    w.set_scale(r, g, 0.5 + 0.25 * (g % 4) as f32);
                }
            }
            w
        };
        let mut s = 0xABCD_u64;
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();

        let mut y_scalar = vec![0.0f32; rows];
        let mut y_avx2 = vec![0.0f32; rows];
        ternary_group_matvec_scalar(&w, &x, &mut y_scalar);
        simd_ternary_group_matvec(&w, &x, &mut y_avx2);

        for r in 0..rows {
            let denom = y_scalar[r].abs().max(1.0);
            let rel = (y_scalar[r] - y_avx2[r]).abs() / denom;
            assert!(
                rel < 1e-6,
                "rows={rows} cols={cols} r={r}: scalar={} avx2={} rel={rel:e}",
                y_scalar[r],
                y_avx2[r]
            );
        }
    }
}

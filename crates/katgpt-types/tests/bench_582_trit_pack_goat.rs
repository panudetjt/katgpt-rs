//! Issue 582 GOAT gates for the base-3 trit-packed footprint tier.
//!
//! Run:
//! ```bash
//! cargo test -p katgpt-types --features ternary_trit_pack --release \
//!   --test bench_582_trit_pack_goat -- --nocapture
//! ```
//!
//! # What is measured
//!
//! **G2 footprint (the load-bearing gate).** `encoded_bytes()` of the trit tier
//! vs the bit-plane tier at equal dims must be `<= 0.83x` — i.e. the −18.8% is
//! actually realized and not eaten by row padding. This gate is *arithmetic*,
//! not a timing, so it cannot be noisy and it cannot be argued with.
//!
//! **G2b latency (informational — no pass threshold on CPU).** The trit kernel
//! pays a LUT decode pass per group that the bit-plane kernel does not, and
//! buys 18.8% less memory traffic. At the matrix sizes measurable in a unit
//! test the weights are cache-resident, so the traffic saving has nothing to
//! pay for the decode with, and a loss here is *expected* and shippable — the
//! precedent is `binary_plasma`, which ships opt-in on storage PASS + latency
//! FAIL. What this gate does enforce is a **reject bound: > 2x is a design
//! failure**, because the whole premise is that memory traffic dominates.
//!
//! **G4 alloc-free.** The decode scratch must be a stack array. 0 allocations
//! per call under a `CountingAllocator`.
//!
//! # Honest measurement notes
//!
//! - Timings are wall-clock medians on a busy developer machine — indicative,
//!   not controlled. Treat <15% as noise.
//! - Every shape here fits in cache, which **structurally favours the bit-plane
//!   tier**. The regime this tier is built for (streaming a 5.82 GB model
//!   instead of a 7.17 GB one, or fitting 24 GB of VRAM alongside a KV cache)
//!   is not reachable from a unit test. That asymmetry is stated rather than
//!   hidden, and it is why G2b carries no pass threshold.

#![cfg(feature = "ternary_trit_pack")]

use std::time::Instant;

use katgpt_types::simd::{
    simd_ternary_group_matvec, simd_ternary_group_matvec_parallel, simd_ternary_trit_matvec,
    simd_ternary_trit_matvec_parallel, ternary_group_matvec_scalar, ternary_trit_matvec_scalar,
};
use katgpt_types::{TernaryGroupWeights, TernaryTritWeights};

// ── CountingAllocator (G4) ────────────────────────────────────
// Inlined for the same reason as bench_578: this is a separate compilation
// unit and katgpt-types has no dependency carrying a shared harness.
// Thread-local (not process-global) so sibling tests' heap work is not
// attributed to the G4 window under parallel test execution.

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

// ── Fixtures ──────────────────────────────────────────────────

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

/// Median of `reps` timed runs, ns per call. Median over mean: one scheduling
/// stall on a shared box would dominate a mean.
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

/// Assert `got ≈ want` with error measured against the **magnitude of the
/// computation**, not against the possibly-cancelled result.
///
/// A row of a 5120-column ternary matvec is the sum of 40 group sums, each of
/// magnitude ~3, so a row whose final value lands near 0 is catastrophic
/// cancellation: an absolute error of 1e-5 is ~5e-7 of the work done, but
/// ~3e-5 relative to the result. Dividing by `max(|want|, 1.0)` therefore
/// flags rare near-zero rows as failures while letting large rows off lightly —
/// exactly backwards. Dividing by `max(|want|, rms(want))` measures against the
/// typical magnitude of the operation instead.
///
/// This matters more as `rows` grows: at 512 rows a near-cancelling row is
/// unlikely, at 32768 it is near-certain. Found the hard way by
/// `g2c_streaming_regime_out_of_cache` (row 190 of 32768: -0.3007803 vs
/// -0.3007903, i.e. 1e-5 absolute on a row whose group sums are ~3 each).
fn assert_close_rms(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len());
    let rms = (want.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / want.len() as f64)
        .sqrt() as f32;
    for (r, (&g, &w)) in got.iter().zip(want).enumerate() {
        let denom = w.abs().max(rms).max(f32::MIN_POSITIVE);
        let rel = (g - w).abs() / denom;
        assert!(
            rel < tol,
            "{label} row {r}: {g} vs {w} (rel {rel:.2e} vs tol {tol:.0e}, rms {rms:.3})"
        );
    }
}

const SHAPES: [(usize, usize); 3] = [(512, 512), (1024, 1024), (512, 5120)];

// ── G2: footprint (the load-bearing gate) ─────────────────────

#[test]
fn g2_footprint_beats_bit_planes_by_at_least_17_percent() {
    println!("\n── Issue 582 G2: footprint ──");
    println!(
        "{:>12} {:>14} {:>14} {:>10} {:>12}",
        "shape", "trit B", "bit-plane B", "ratio", "bits/weight"
    );

    let mut all_pass = true;
    for &(rows, cols) in &SHAPES {
        let trit = TernaryTritWeights::new(rows, cols);
        let plane = TernaryGroupWeights::new(rows, cols);
        let ratio = trit.encoded_bytes() as f64 / plane.encoded_bytes() as f64;
        let bpw = (trit.encoded_bytes() * 8) as f64 / (rows * cols) as f64;
        let pass = ratio <= 0.83;
        all_pass &= pass;
        println!(
            "{:>5}x{:<6} {:>14} {:>14} {:>9.4} {:>12.4} {}",
            rows,
            cols,
            trit.encoded_bytes(),
            plane.encoded_bytes(),
            ratio,
            bpw,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    // The Bonsai-27B headline, computed from the same arithmetic rather than
    // quoted: 26.97B params at 1.725 vs 2.125 bits/weight.
    let params = 26.97e9;
    println!(
        "\nTernary-Bonsai-27B ({params:.2e} params): {:.2} GB trit vs {:.2} GB bit-plane",
        params * 1.725 / 8.0 / 1e9,
        params * 2.125 / 8.0 / 1e9
    );

    assert!(
        all_pass,
        "G2 FAIL: footprint ratio above 0.83 at some shape"
    );
}

// ── G2b: latency (informational, reject bound only) ───────────

#[test]
fn g2b_latency_vs_bit_plane_kernel() {
    println!("\n── Issue 582 G2b: ns/call (median of 9 x 20 calls) ──");
    println!(
        "{:>12} {:>14} {:>14} {:>10}",
        "shape", "trit", "bit-plane", "ratio"
    );

    let mut worst = 0.0f64;
    for &(rows, cols) in &SHAPES {
        let src = dense_matrix(rows, cols, 0x582);
        let plane = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        let trit = TernaryTritWeights::from_group(&plane);
        let mut s = 0xBEEF_u64;
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
        let mut y = vec![0.0f32; rows];

        let t_trit = median_ns(9, 20, || simd_ternary_trit_matvec(&trit, &x, &mut y));
        let t_plane = median_ns(9, 20, || simd_ternary_group_matvec(&plane, &x, &mut y));
        let ratio = t_trit / t_plane;
        worst = worst.max(ratio);
        println!("{rows:>5}x{cols:<6} {t_trit:>14.0} {t_plane:>14.0} {ratio:>9.2}x");
    }

    println!(
        "\nworst ratio {worst:.2}x — informational; reject bound is 2.00x.\n\
         All shapes here are cache-resident, which favours the bit-plane tier:\n\
         the trit tier's 18.8% traffic saving has nothing to pay its decode with\n\
         until the weights stream from RAM."
    );

    // ── Attribution: where does the SIMD delta come from? ──
    //
    // The trit SIMD kernel differs from the bit-plane one in TWO ways, and a
    // single ratio cannot tell them apart:
    //   (a) base-3 LUT decode instead of SWAR bit extraction
    //   (b) one hsum + one scale multiply per group, instead of folding the
    //       scale into every sign vector (the bit-plane kernel's Issue 578
    //       trick, which costs a vmulq per 4 lanes)
    //
    // The SCALAR paths of both tiers already share (b) — both accumulate a
    // group then multiply once. So scalar-vs-scalar isolates (a). If the scalar
    // ratio is ~1.0 while the SIMD ratio is ~0.8, the win is structural and is
    // ALSO available to the existing tier without any format change; if the
    // scalar ratio is itself below 1.0, base-3 decode is genuinely cheaper than
    // bit extraction.
    println!("\n── attribution: scalar-vs-scalar isolates decode cost from scale hoisting ──");
    println!(
        "{:>12} {:>14} {:>14} {:>10}",
        "shape", "trit", "bit-plane", "ratio"
    );
    for &(rows, cols) in &SHAPES {
        let src = dense_matrix(rows, cols, 0x582);
        let plane = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        let trit = TernaryTritWeights::from_group(&plane);
        let mut s = 0xBEEF_u64;
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
        let mut y = vec![0.0f32; rows];

        let t_trit = median_ns(9, 5, || ternary_trit_matvec_scalar(&trit, &x, &mut y));
        let t_plane = median_ns(9, 5, || ternary_group_matvec_scalar(&plane, &x, &mut y));
        println!(
            "{rows:>5}x{cols:<6} {t_trit:>14.0} {t_plane:>14.0} {:>9.2}x",
            t_trit / t_plane
        );
    }
    assert!(
        worst <= 2.0,
        "G2b REJECT: {worst:.2}x is past the 2x design bound — the decode is \
         too expensive to be paid for by an 18.8% traffic cut"
    );
}

// ── G2c: the streaming regime (the tier's actual rationale) ───

/// Build bit-plane weights **without** a dense f32 source.
///
/// A 32768×5120 f32 source would be 671 MB; the container itself is only ~44 MB.
/// Filling `pos_bits`/`neg_bits` directly keeps the fixture affordable, and
/// `pos = a & !b` / `neg = b & !a` guarantees the `pos & neg == 0` invariant by
/// construction (asserted below).
fn big_plane_weights(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
    let mut w = TernaryGroupWeights::new(rows, cols);
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };
    for i in 0..w.pos_bits.len() {
        let a = next();
        let b = next();
        w.pos_bits[i] = a & !b;
        w.neg_bits[i] = b & !a;
    }
    for scale in w.group_scale.iter_mut() {
        *scale = half::f16::from_f32(0.25 + ((next() >> 60) as f32) * 0.125);
    }
    assert!(w.invariant_holds(), "fixture must satisfy pos & neg == 0");
    w
}

/// Does the 18.8% traffic saving actually pay off when the weights do **not**
/// fit in cache?
///
/// Every other timing in this file is cache-resident, which the honest-caveats
/// section calls out as structurally favouring the bit-plane tier. This test
/// covers the regime the tier was built for: a 32768×5120 matrix is ~44 MB of
/// bit-planes vs ~36 MB of trits, both well past the M3 Max's 16 MB L2, so each
/// call streams the weights from RAM exactly as a 27B decode does.
///
/// Informational, not a gate — a single large shape on one host is weak evidence
/// and the ratio is reported either way. What would be *interesting* is the
/// ratio moving relative to the cache-resident 0.79–0.82×: further below means
/// the traffic saving compounds, back toward 1.0 means it does not.
#[test]
#[ignore = "allocates ~80 MB and streams it repeatedly; run explicitly"]
fn g2c_streaming_regime_out_of_cache() {
    let (rows, cols) = (32768usize, 5120usize);
    let plane = big_plane_weights(rows, cols, 0x582_5EED);
    let trit = TernaryTritWeights::from_group(&plane);

    // Correctness first — a fixture this large would otherwise hide an
    // indexing bug behind a plausible-looking timing.
    assert!(trit.is_canonical());
    assert_eq!(trit.checksum(), plane.checksum(), "cross-tier checksum");

    let mut s = 0xBEEF_u64;
    let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
    let mut y_trit = vec![0.0f32; rows];
    let mut y_plane = vec![0.0f32; rows];
    simd_ternary_trit_matvec(&trit, &x, &mut y_trit);
    simd_ternary_group_matvec(&plane, &x, &mut y_plane);
    assert_close_rms(&y_trit, &y_plane, 1e-6, "streaming trit-vs-plane");

    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!(
        "\n── Issue 582 G2c: streaming regime ({rows}x{cols}) ──\n\
         trit payload {:.1} MiB vs bit-plane {:.1} MiB",
        mib(trit.encoded_bytes()),
        mib(plane.encoded_bytes())
    );

    // Fewer reps: each call touches ~40 MB, so wall-clock per call is ~ms and
    // 5 reps of 3 calls is already several seconds.
    let t_trit = median_ns(5, 3, || simd_ternary_trit_matvec(&trit, &x, &mut y_trit));
    let t_plane = median_ns(5, 3, || simd_ternary_group_matvec(&plane, &x, &mut y_plane));
    let ratio = t_trit / t_plane;
    println!(
        "trit {:.3} ms/call vs bit-plane {:.3} ms/call → ratio {ratio:.3}x\n\
         (cache-resident ratio for reference: 0.79-0.82x)\n\
         effective bandwidth: trit {:.1} GB/s, bit-plane {:.1} GB/s",
        t_trit / 1e6,
        t_plane / 1e6,
        trit.encoded_bytes() as f64 / t_trit,
        plane.encoded_bytes() as f64 / t_plane
    );
}

// ── G1: cross-tier agreement at benchmark scale ───────────────

#[test]
fn g1_matches_bit_plane_tier_at_benchmark_scale() {
    // The unit tests cover small + ragged shapes; this covers the wide ones,
    // where an indexing bug would only show up past the first few groups.
    for &(rows, cols) in &SHAPES {
        let src = dense_matrix(rows, cols, 0x1582);
        let plane = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        let trit = TernaryTritWeights::from_group(&plane);
        assert!(trit.is_canonical(), "{rows}x{cols}");
        assert_eq!(trit.checksum(), plane.checksum(), "{rows}x{cols} checksum");

        let mut s = 0xF00D_u64;
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
        let mut y_trit = vec![0.0f32; rows];
        let mut y_plane = vec![0.0f32; rows];

        // Scalar-vs-scalar is exact: same op order, same one-scale-per-group.
        ternary_trit_matvec_scalar(&trit, &x, &mut y_trit);
        ternary_group_matvec_scalar(&plane, &x, &mut y_plane);
        assert_eq!(
            y_trit, y_plane,
            "{rows}x{cols} scalar must be bit-identical"
        );

        // SIMD-vs-SIMD differs only in summation order (~1e-6 relative).
        simd_ternary_trit_matvec(&trit, &x, &mut y_trit);
        simd_ternary_group_matvec(&plane, &x, &mut y_plane);
        assert_close_rms(&y_trit, &y_plane, 1e-6, &format!("{rows}x{cols} simd"));
    }
}

// ── G2d: row-parallel parity with the bit-plane tier ──────────

/// The row-parallel kernel is what a decode-step consumer actually calls, so the
/// tier is only usable if it keeps pace there too.
///
/// The bit-plane tier's row-parallel kernel measured **7.21×** over serial on
/// real Bonsai geometry (riir-ai Bench 582), and riir-engine's `forward_ternary`
/// depends on it. A footprint tier that saved 18.8% of bytes but lost that 7×
/// would be strictly worse for every real consumer — so this checks the parallel
/// split works and stays bit-identical.
///
/// **The timing half of this test is noisy and only the bit-identity is
/// asserted.** Measured trit/plane ratios under 16 threads ranged 0.83–1.03×
/// across runs on a working developer box, and thread scaling 2.4–4.5× — i.e.
/// the serial tier's stable 0.87× advantage is *not* reproducibly visible once
/// 16 threads contend, and this harness cannot resolve whether it survives.
/// Reported, not claimed. A quiet-box measurement is the way to settle it.
#[test]
fn g2d_row_parallel_keeps_pace() {
    // 4096 rows is past PARALLEL_ROW_MIN (256) by enough to amortize rayon.
    let (rows, cols) = (4096usize, 5120usize);
    let plane = big_plane_weights(rows, cols, 0x582D);
    let trit = TernaryTritWeights::from_group(&plane);

    let mut s = 0xBEEF_u64;
    let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
    let mut y = vec![0.0f32; rows];
    let mut y_ref = vec![0.0f32; rows];

    // Bit-identity under threads is the load-bearing claim; time is secondary.
    simd_ternary_trit_matvec_parallel(&trit, &x, &mut y);
    simd_ternary_trit_matvec(&trit, &x, &mut y_ref);
    assert_eq!(y, y_ref, "parallel must be bit-identical to serial");

    // 9 reps rather than 5: the parallel numbers move more than the serial ones.
    let t_trit_ser = median_ns(9, 3, || simd_ternary_trit_matvec(&trit, &x, &mut y));
    let t_plane_ser = median_ns(9, 3, || simd_ternary_group_matvec(&plane, &x, &mut y));
    let t_trit_par = median_ns(9, 5, || {
        simd_ternary_trit_matvec_parallel(&trit, &x, &mut y)
    });
    let t_plane_par = median_ns(9, 5, || {
        simd_ternary_group_matvec_parallel(&plane, &x, &mut y)
    });

    println!(
        "\n── Issue 582 G2d: row-parallel ({rows}x{cols}, {} threads; timings noisy) ──\n\
         {:>10} {:>12} {:>12} {:>10}\n\
         {:>10} {:>11.3} {:>11.3} {:>9.2}x\n\
         {:>10} {:>11.3} {:>11.3} {:>9.2}x\n\
         thread scaling: trit {:.2}x, plane {:.2}x",
        rayon::current_num_threads(),
        "",
        "trit ms",
        "plane ms",
        "trit/plane",
        "serial",
        t_trit_ser / 1e6,
        t_plane_ser / 1e6,
        t_trit_ser / t_plane_ser,
        "parallel",
        t_trit_par / 1e6,
        t_plane_par / 1e6,
        t_trit_par / t_plane_par,
        t_trit_ser / t_trit_par,
        t_plane_ser / t_plane_par
    );

    // Only a floor is asserted: thread scaling on a busy dev box is noisy, but
    // anything under 2x on 4096 rows would mean the parallel split is broken
    // rather than merely contended.
    assert!(
        t_trit_ser / t_trit_par > 2.0,
        "parallel speedup {:.2}x is too low to be real parallelism",
        t_trit_ser / t_trit_par
    );
}

// ── G4: alloc-free ────────────────────────────────────────────

#[test]
fn g4_matvec_allocates_nothing_in_steady_state() {
    let (rows, cols) = (512usize, 5120usize);
    let src = dense_matrix(rows, cols, 0x4582);
    let plane = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
    let trit = TernaryTritWeights::from_group(&plane);
    let mut s = 0xC0DE_u64;
    let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
    let mut y = vec![0.0f32; rows];

    // Warm up outside the counted window.
    simd_ternary_trit_matvec(&trit, &x, &mut y);
    ternary_trit_matvec_scalar(&trit, &x, &mut y);

    let (_, simd_allocs) = alloc_delta(|| {
        for _ in 0..1000 {
            simd_ternary_trit_matvec(&trit, &x, &mut y);
        }
    });
    let (_, scalar_allocs) = alloc_delta(|| {
        for _ in 0..1000 {
            ternary_trit_matvec_scalar(&trit, &x, &mut y);
        }
    });

    println!(
        "\n── Issue 582 G4 ──\nsimd  {simd_allocs} allocs / 1000 calls\nscalar {scalar_allocs} allocs / 1000 calls"
    );
    assert_eq!(simd_allocs, 0, "simd kernel must not allocate");
    assert_eq!(scalar_allocs, 0, "scalar kernel must not allocate");
}

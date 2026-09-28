//! Issue 187 T6 — d_h = 18_720 timing trial.
//!
//! Verifies the parallel Householder+QL path completes the full G
//! eigendecomp at the KARC promotion-gate target config within a feasible
//! wall time. Run with:
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/katgpt-par-test cargo test --release \
//!   --features karc_householder_eig_par \
//!   --test karc_dh_18720_timing_trial -- --ignored --nocapture
//! ```
//!
//! The test is `#[ignore]`'d because it takes ~1 hour. It is NOT a unit
//! test — it produces a wall-time measurement that informs the T6/T7
//! promotion decision.

#![cfg(feature = "karc_householder_eig_par")]

use katgpt_core::linalg::symmetric_eig::{SymmetricEigScratch, par::symmetric_eig_par};

/// Deterministic xorshift64 PRNG (matches the symmetric_eig test helper).
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn u64_to_unit_f64(x: u64) -> f64 {
    let u = x >> 11;
    let sign = (x >> 63) & 1;
    let m = (u as f64) / ((1u64 << 53) as f64);
    if sign == 1 { -m } else { m }
}

/// Build a random SPD Gram matrix of size n×n. Matches the symmetric_eig
/// tests' SPD generator recipe.
fn random_spd_gram(state: &mut u64, a_out: &mut [f64], n: usize) {
    for v in a_out.iter_mut().take(n * n) {
        *v = 0.0;
    }
    let mut m_col = vec![0.0_f64; n];
    for _k in 0..n {
        for m_col_j in m_col.iter_mut().take(n) {
            *m_col_j = u64_to_unit_f64(xorshift64(state));
        }
        for i in 0..n {
            for j in 0..n {
                a_out[i * n + j] += m_col[i] * m_col[j];
            }
        }
    }
    // SPD floor + small diagonal to break degeneracy.
    for i in 0..n {
        a_out[i * n + i] += 1.0 + 0.01 * (i as f64);
    }
}

#[test]
#[ignore = "~1 hour wall-time measurement at d_h=18720, not a unit test; run with --release --ignored"]
fn dh_18720_parallel_timing() {
    use std::time::Instant;

    let d_h = 18_720_usize;
    eprintln!("Issue 187 T6: d_h = {d_h} parallel Householder+QL timing trial");
    eprintln!(
        "rayon thread pool: {} threads",
        rayon::current_num_threads()
    );

    // Build the Gram matrix (this allocation is ~2.8 GB; takes a moment).
    let t_alloc = Instant::now();
    let mut a = vec![0.0_f64; d_h * d_h];
    eprintln!(
        "allocating {} entries ({:.2} GB)...",
        d_h * d_h,
        (d_h * d_h * 8) as f64 / 1e9
    );
    let mut state: u64 = 0x1872_dead_beef_face;
    random_spd_gram(&mut state, &mut a, d_h);
    eprintln!("Gram build: {:.2?}", t_alloc.elapsed());

    // Pre-allocate output buffers.
    let mut eigvals = vec![0.0_f64; d_h];
    let mut eigvecs = vec![0.0_f64; d_h * d_h];
    let mut scratch = SymmetricEigScratch::new();

    // Run the parallel eigendecomp.
    let t0 = Instant::now();
    symmetric_eig_par(&mut eigvals, &mut eigvecs, &a, &mut scratch, d_h, 30);
    let dt = t0.elapsed();

    eprintln!("RESULT: d_h = {d_h}, parallel wall = {dt:.2?}");
    eprintln!(
        "        ({:.2}× the ≤30 min feasibility target)",
        dt.as_secs_f64() / 1800.0
    );

    // Sanity: trace of A equals sum of eigenvalues (within f64 precision).
    let mut trace_a = 0.0_f64;
    for i in 0..d_h {
        trace_a += a[i * d_h + i];
    }
    let sum_eig: f64 = eigvals.iter().sum();
    let trace_err = (trace_a - sum_eig).abs() / trace_a.abs().max(1e-300);
    eprintln!(
        "sanity: trace(A) = {trace_a:.6e}, sum(eigvals) = {sum_eig:.6e}, rel err = {trace_err:.2e}"
    );
    assert!(
        trace_err < 1e-10,
        "trace sanity check failed: rel err = {trace_err:.2e}"
    );

    // Sanity: eigenvalues finite.
    for (i, &v) in eigvals.iter().enumerate() {
        assert!(v.is_finite(), "eigvals[{i}] = {v} not finite");
    }

    // Report verdict.
    let target_secs = 1800.0; // 30 min
    let actual_secs = dt.as_secs_f64();
    if actual_secs <= target_secs {
        eprintln!("VERDICT: T6 PASS — parallel wall {dt:.2?} ≤ 30 min target");
    } else {
        eprintln!(
            "VERDICT: T6 MISS — parallel wall {:.2?} > 30 min target ({:.2}× over)",
            dt,
            actual_secs / target_secs
        );
    }
}

//! Large-d_h ALS B-step via Jacobi eigendecomposition (Issue 185, Plan 308 T4.5).
//!
//! The Kronecker-vectorized B-step in [`super::low_rank_fit`] is exact but
//! `O((r·d_h)³)` time and `O((r·d_h)²)` space — feasible only when
//! `r·d_h ≤ ~2000`. At the KARC promotion-gate target
//! (`K=8, M=8, R=2, d_h=18_720, r=8`) the Kronecker system alone is
//! `(r·d_h)² = (8·18_720)² ≈ 2.24·10¹⁰` f64 ≈ 179 GB — infeasible.
//!
//! This module ships the large-d_h path documented in the Phase 2 rustdoc as
//! "future work": eigendecompose `G = XᵀX` (one-time, `O(d_h³)` via Jacobi),
//! then per ALS iteration the B-step becomes `O(r·d_h²)` instead of
//! `O((r·d_h)³)`. For `d_h = 18_720, r = 8` that's `~2.8·10⁹` FLOPs/iter
//! vs `~5.2·10¹³` FLOPs/iter for the Kronecker path — a 18_000× speedup.
//!
//! # Algorithm (Bartels–Stewart diagonalization of the Sylvester equation)
//!
//! The B-step normal equation `(AᵀA)·B·G + λB = Aᵀ·Covᵀ` is a Sylvester-like
//! equation. Eigendecomposing both symmetric sides:
//!
//! - `AᵀA = Q_a · Λ_a · Q_aᵀ` (r × r — small, `O(r³)` per iter)
//! - `G = Q_g · Λ_g · Q_gᵀ` (d_h × d_h — done ONCE outside the ALS loop,
//!   `O(d_h³)` via Jacobi)
//!
//! and substituting `B = Q_a · C · Q_gᵀ`, the equation decouples element-wise:
//!
//! ```text
//! Λ_a[i]·C[i,j]·Λ_g[j] + λ·C[i,j] = (Q_aᵀ · Aᵀ · Covᵀ · Q_g)[i,j]
//! ⇒ C[i,j] = (Q_aᵀ · Aᵀ · Covᵀ · Q_g)[i,j] / (Λ_a[i]·Λ_g[j] + λ)
//! ```
//!
//! Per ALS iteration cost breakdown:
//! - Eigendecompose `AᵀA`: `O(r³)` (trivial)
//! - Build `AᵀA` from current `A`: `O(D·r²)`
//! - Transform RHS `Q_aᵀ · (Cov·A)ᵀ · Q_g`: `O(r·d_h²)` (dominated by the
//!   `(Cov·A)ᵀ · Q_g` matmul)
//! - Element-wise scale: `O(r·d_h)`
//! - Recover `B = Q_a · C · Q_gᵀ`: `O(r·d_h²)` (dominated by the `C · Q_gᵀ`
//!   matmul)
//!
//! **Total per iter: `O(r·d_h²)`.** One-time (outside ALS loop): `O(d_h³)`.
//!
//! # Determinism / bit-reproducibility
//!
//! - The one-time Jacobi eigendecomp of `G` is bit-deterministic given the
//!   same `G` and `tol` — the [`super::jacobi_eigen`] implementation sweeps
//!   in fixed `(p, q)` order.
//! - Per-iter Jacobi of `AᵀA` is likewise deterministic.
//! - Element-wise division is deterministic.
//! - Therefore two calls with identical `(G, Cov, d_h, D, r, λ, max_iters, tol)`
//!   produce bit-identical `A, B`.
//!
//! On configs where both paths are feasible (e.g. `r=4, d_h=96`), the Jacobi
//! path and the Kronecker path produce solutions that agree to ~`1e-10`
//! (both solve the same normal equation to machine precision; the small
//! residual difference is the float-operation-order difference between
//! Kronecker Cholesky and the eigenbasis diagonalization). This is the T2
//! parity-test contract.

use super::{
    LowRankFitScratch, als_a_step, als_convergence_step, als_scale_rebalance, build_ata,
    build_cov_a, jacobi_eigen,
};

/// Large-d_h ALS low-rank ridge factorization with the Jacobi B-step.
///
/// Drop-in alternative to [`super::low_rank_fit`] for configs where
/// `r·d_h > ~2000`. Identical contract: same inputs, same outputs, same
/// deterministic-init convention (`B ← [I_r | 0]`, `A ← 0`), same scale
/// rebalance, same convergence check. The only difference is the B-step
/// decomposition: Jacobi eigendecomposition of `G` (one-time) + `AᵀA`
/// (per-iter) instead of the `(r·d_h) × (r·d_h)` Kronecker Cholesky.
///
/// # When to prefer this over [`super::low_rank_fit`]
///
/// - `r·d_h ≤ ~2000`: use [`super::low_rank_fit`] — Kronecker path is faster
///   (no eigendecomp overhead) and equally exact.
/// - `r·d_h > ~2000` and `d_h ≤ ~20_000`: use this function. Memory budget
///   is dominated by the `d_h × d_h` eigvecs of `G` (~2.8 GB at d_h=18_720).
/// - `d_h > ~50_000`: still infeasible without H-matrix / tensor-train
///   factorization — out of scope for Issue 185.
///
/// # Inputs
///
/// Identical to [`super::low_rank_fit`]: `gram` (`d_h × d_h`, un-regularized),
/// `cov` (`d_h × D`), `d_h`, `d_out` = `D`, `r`, `lambda`, `max_iters`, `tol`,
/// `a_out` (`D × r`), `b_out` (`r × d_h`), `scratch`.
///
/// The scratch must additionally carry the Jacobi-path buffers — call
/// [`LowRankFitScratch::ensure_jacobi_capacity`] before invoking this
/// function. (For convenience this function also calls it; pre-calling
/// avoids the first-call resize.)
///
/// # Returns
///
/// Number of ALS iterations performed (same semantics as [`super::low_rank_fit`]).
///
/// # Panics
///
/// Same input-validation panics as [`super::low_rank_fit`]: `r == 0`,
/// `r > d_h`, `λ ≤ 0`, or any buffer undersized.
///
/// # Worst-case memory
///
/// At `d_h = 18_720` (K=8/M=8/R=2): the largest single buffer is `eigvecs_g`
/// at `18_720² × 8 B ≈ 2.6 GB` f64. The function takes a caller-owned
/// `LowRankFitScratch` so the buffers live in the caller's heap, not the
/// primitive's — per AGENTS.md "Pass pre-allocated scratch buffers as `&mut`
/// parameters instead of allocating inside hot loops".
#[allow(clippy::too_many_arguments)]
pub fn low_rank_fit_jacobi_bstep(
    gram: &[f64],
    cov: &[f64],
    d_h: usize,
    d_out: usize,
    r: usize,
    lambda: f64,
    max_iters: usize,
    tol: f64,
    a_out: &mut [f64],
    b_out: &mut [f64],
    scratch: &mut LowRankFitScratch,
) -> usize {
    assert!(r > 0, "low_rank_fit_jacobi_bstep: r must be > 0");
    assert!(
        r <= d_h,
        "low_rank_fit_jacobi_bstep: r must be <= d_h (got r={r}, d_h={d_h})"
    );
    assert!(
        lambda > 0.0,
        "low_rank_fit_jacobi_bstep: lambda must be > 0"
    );
    assert!(a_out.len() >= d_out * r, "a_out too small");
    assert!(b_out.len() >= r * d_h, "b_out too small");

    // Grow the Jacobi-path scratch buffers (idempotent — Kronecker-path
    // callers never paid for these, but this function needs them).
    scratch.ensure_jacobi_capacity(d_h, r);

    // Deterministic init: B = [I_r | 0], A = 0 (identical to low_rank_fit).
    for k in 0..r {
        for j in 0..d_h {
            b_out[k * d_h + j] = if j == k { 1.0 } else { 0.0 };
        }
    }
    for v in a_out.iter_mut().take(d_out * r) {
        *v = 0.0;
    }

    low_rank_fit_jacobi_with_init(
        gram, cov, d_h, d_out, r, lambda, max_iters, tol, a_out, b_out, scratch,
    )
}

/// ALS loop body for the Jacobi B-step path (shared with the warm-start
/// variant when one is added). Mirrors [`super::low_rank_fit_with_init`]:
/// caller has initialized `a_out` and `b_out`, we run the loop.
///
/// Differs from [`super::low_rank_fit_with_init`] only in (a) the one-time
/// `G` eigendecomp that replaces the one-time `G+λI` Cholesky, and (b) the
/// per-iter Jacobi B-step that replaces the per-iter Kronecker Cholesky.
#[allow(clippy::too_many_arguments)]
fn low_rank_fit_jacobi_with_init(
    gram: &[f64],
    cov: &[f64],
    d_h: usize,
    d_out: usize,
    r: usize,
    lambda: f64,
    max_iters: usize,
    tol: f64,
    a_out: &mut [f64],
    b_out: &mut [f64],
    scratch: &mut LowRankFitScratch,
) -> usize {
    // ── One-time: eigendecompose G = Q_g · Λ_g · Q_gᵀ (d_h × d_h) ──
    // This replaces low_rank_fit_with_init's pre-compute of chol(G + λI).
    //
    // Three eigensolver paths (Issues 186 + 187):
    //   - Default (no feature): in-tree `jacobi_eigen` (Plan 308 T2.3).
    //     Classic cyclic Jacobi with sign-bug fix from Issue 185.
    //     O(d_h³·n_sweeps) cost — infeasible at d_h > ~5000.
    //   - `karc_householder_eig` feature: `linalg::symmetric_eig` (Issue 186
    //     Path B). Serial Householder tridiag + implicit-shift QL. ~5-10×
    //     faster than Jacobi at d_h ≥ 256.
    //   - `karc_householder_eig_par` feature (implies the above): row-parallel
    //     rayon variant `linalg::symmetric_eig::par::symmetric_eig_par`
    //     (Issue 187). 6-8× over serial at n ≥ 1024 on 16 cores; brings
    //     d_h = 18_720 from ~12 h projected serial to ~87 min wall.
    // All three paths produce bit-identical (eigvals, eigvecs) modulo
    // eigenvector sign — the parallel path is bit-identical to serial by
    // construction (row-parallel, no cross-row reductions); verified by
    // `tests/karc_low_rank_jacobi_vs_kronecker.rs` + `symmetric_eig/tests.rs`
    // `par_vs_serial_*` parity tests under the `karc_householder_eig_par`
    // feature flag.
    #[cfg(not(feature = "karc_householder_eig"))]
    {
        // The Jacobi sweep count is bounded: for SPD Grams, ~10 sweeps typically
        // suffices for `tol = 1e-12`. For d_h = 18_720 each sweep is ~6.5e12
        // FLOPs; allow up to 30 sweeps as a safety margin (anything still
        // off-diagonal after that is a near-defective Gram the caller should
        // regularize).
        let g_tol = 1e-12_f64;
        let g_max_sweeps = 30_usize;
        jacobi_eigen(
            &mut scratch.eigvals_g[..d_h],
            &mut scratch.eigvecs_g[..d_h * d_h],
            gram,
            &mut scratch.jacobi_scratch_g[..d_h * d_h],
            d_h,
            g_tol,
            g_max_sweeps,
        );
    }
    #[cfg(all(
        feature = "karc_householder_eig",
        not(feature = "karc_householder_eig_par")
    ))]
    {
        // Numerical Recipes default: 30 QL iterations per eigenvalue before
        // declaring non-convergence. For SPD Grams, ~2-5 iterations suffice.
        crate::linalg::symmetric_eig::symmetric_eig(
            &mut scratch.eigvals_g[..d_h],
            &mut scratch.eigvecs_g[..d_h * d_h],
            gram,
            &mut scratch.symmetric_eig,
            d_h,
            30,
        );
    }
    #[cfg(feature = "karc_householder_eig_par")]
    {
        // Same algorithm + same max-iter as the serial Householder path; only
        // the parallelism differs (row-parallel rayon — bit-identical output).
        crate::linalg::symmetric_eig::par::symmetric_eig_par(
            &mut scratch.eigvals_g[..d_h],
            &mut scratch.eigvecs_g[..d_h * d_h],
            gram,
            &mut scratch.symmetric_eig,
            d_h,
            30,
        );
    }

    // wout_old must be zeroed so the first iteration's convergence check
    // measures against the post-init Wout (not stale memory).
    for v in scratch.wout_old.iter_mut().take(d_out * d_h) {
        *v = 0.0;
    }

    // ── ALS iterations ──
    let mut iters_done = max_iters;
    for iter in 0..max_iters {
        // ── A-step: identical to the Kronecker path (shared helper) ──
        als_a_step(gram, cov, d_h, d_out, r, lambda, b_out, a_out, scratch);

        // ── B-step: Jacobi diagonalization ──
        jacobi_b_step(gram, cov, d_h, d_out, r, lambda, a_out, b_out, scratch);

        // ── Scale rebalance (shared with Kronecker path) ──
        als_scale_rebalance(a_out, b_out, d_out, d_h, r);

        // ── Convergence check (shared with Kronecker path) ──
        let diff = als_convergence_step(a_out, b_out, d_out, d_h, r, scratch);
        if diff < tol {
            iters_done = iter + 1;
            break;
        }
    }
    iters_done
}

/// One Jacobi B-step: solves `(AᵀA)·B·G + λB = Aᵀ·Covᵀ` via the
/// `AᵀA = Q_a·Λ_a·Q_aᵀ`, `G = Q_g·Λ_g·Q_gᵀ` diagonalization.
///
/// Writes the new `B` into `b_out`. Reads `a_out` and the pre-computed
/// `scratch.eigvals_g` / `scratch.eigvecs_g` (set up once by the caller).
///
/// # Math (see module docs for the full derivation)
///
/// Substitute `B = Q_a · C · Q_gᵀ` into the normal equation:
/// ```text
/// Λ_a[i]·C[i,j]·Λ_g[j] + λ·C[i,j] = (Q_aᵀ · Aᵀ · Covᵀ · Q_g)[i,j]
/// ⇒ C[i,j] = (Q_aᵀ · Aᵀ · Covᵀ · Q_g)[i,j] / (Λ_a[i]·Λ_g[j] + λ)
/// ⇒ B = Q_a · C · Q_gᵀ
/// ```
///
/// Buffers used (all pre-allocated in `LowRankFitScratch`):
/// - `eigvals_ata`, `eigvecs_ata`, `jacobi_scratch_ata` — for the r × r eigendecomp
/// - `ata` — built from `a_out` via [`build_ata`]
/// - `cov_a` — built from `cov, a_out` via [`build_cov_a`]
/// - `rhs_transformed` — `Q_aᵀ · (Cov·A)ᵀ · Q_g` (r × d_h)
/// - `c_tilde` — element-wise scaled `C`
/// - `qc_temp` — temp for the `Q_a · C` matmul
#[allow(clippy::too_many_arguments)]
fn jacobi_b_step(
    _gram: &[f64], // unused: G already eigendecomposed into scratch.eigvecs_g/eigvals_g
    cov: &[f64],
    d_h: usize,
    d_out: usize,
    r: usize,
    lambda: f64,
    a_out: &[f64],
    b_out: &mut [f64],
    scratch: &mut LowRankFitScratch,
) {
    // 1. Build AᵀA (r × r) from current A.
    build_ata(a_out, d_out, r, scratch);

    // 2. Eigendecompose AᵀA = Q_a · Λ_a · Q_aᵀ (r × r, cheap).
    //    Re-use the r-side Jacobi buffers.
    jacobi_eigen(
        &mut scratch.eigvals_ata[..r],
        &mut scratch.eigvecs_ata[..r * r],
        &scratch.ata[..r * r],
        &mut scratch.jacobi_scratch_ata[..r * r],
        r,
        1e-15_f64,
        50_usize,
    );

    // 3. Build Cov·A (d_h × r) — same as the Kronecker path's step 0.
    build_cov_a(cov, a_out, d_h, d_out, r, scratch);

    // 4. Compute rhs_transformed = Q_aᵀ · (Cov·A)ᵀ · Q_g  (r × d_h).
    //    Note: (Cov·A)ᵀ[k, j] = cov_a[j*r + k]  (cov_a is d_h × r row-major).
    //    Step 4a: temp = (Cov·A)ᵀ · Q_g  (r × d_h).
    //            temp[k, l] = Σ_j cov_a[j*r+k] · Q_g[j, l]
    //                       = Σ_j cov_a[j*r+k] · eigvecs_g[j*d_h + l]
    //    We accumulate directly into rhs_transformed, then apply Q_aᵀ.
    let rhs = &mut scratch.rhs_transformed[..r * d_h];
    for k in 0..r {
        for l in 0..d_h {
            let mut s = 0.0f64;
            let mut j = 0;
            // 4-way unrolled inner loop over d_h (matches the A-step's
            // unrolling style for SIMD-friendliness).
            while j + 4 <= d_h {
                s += scratch.cov_a[j * r + k] * scratch.eigvecs_g[j * d_h + l];
                s += scratch.cov_a[(j + 1) * r + k] * scratch.eigvecs_g[(j + 1) * d_h + l];
                s += scratch.cov_a[(j + 2) * r + k] * scratch.eigvecs_g[(j + 2) * d_h + l];
                s += scratch.cov_a[(j + 3) * r + k] * scratch.eigvecs_g[(j + 3) * d_h + l];
                j += 4;
            }
            while j < d_h {
                s += scratch.cov_a[j * r + k] * scratch.eigvecs_g[j * d_h + l];
                j += 1;
            }
            rhs[k * d_h + l] = s;
        }
    }
    //    Step 4b: rhs_transformed = Q_aᵀ · temp  (r × d_h).
    //            (Q_aᵀ · temp)[i, l] = Σ_k Q_a[k, i] · temp[k, l]
    //                              = Σ_k eigvecs_ata[k*r + i] · rhs[k*d_h + l]
    //    Compute in-place: read rhs (temp), write into c_tilde.
    let c_tilde = &mut scratch.c_tilde[..r * d_h];
    for i in 0..r {
        for l in 0..d_h {
            let mut s = 0.0f64;
            for k in 0..r {
                s += scratch.eigvecs_ata[k * r + i] * rhs[k * d_h + l];
            }
            c_tilde[i * d_h + l] = s;
        }
    }

    // 5. Element-wise scale: c_tilde[i, l] /= (Λ_a[i] · Λ_g[l] + λ).
    //    After this, c_tilde holds C (the solution in the rotated basis).
    for i in 0..r {
        let lambda_a_i = scratch.eigvals_ata[i];
        for l in 0..d_h {
            let denom = lambda_a_i * scratch.eigvals_g[l] + lambda;
            // denom > 0 guaranteed: Λ_a[i] ≥ 0 (PSD), Λ_g[l] ≥ 0 (PSD Gram),
            // λ > 0 (asserted). Defensive guard against degenerate eigenvalues.
            c_tilde[i * d_h + l] /= denom.max(f64::MIN_POSITIVE);
        }
    }

    // 6. Recover B = Q_a · C · Q_gᵀ  (r × d_h).
    //    Step 6a: qc_temp = Q_a · C  (r × d_h).
    //            qc_temp[i, l] = Σ_k Q_a[i, k] · C[k, l]
    //                          = Σ_k eigvecs_ata[i*r + k] · c_tilde[k*d_h + l]
    let qc_temp = &mut scratch.qc_temp[..r * d_h];
    for i in 0..r {
        for l in 0..d_h {
            let mut s = 0.0f64;
            for k in 0..r {
                s += scratch.eigvecs_ata[i * r + k] * c_tilde[k * d_h + l];
            }
            qc_temp[i * d_h + l] = s;
        }
    }
    //    Step 6b: B = qc_temp · Q_gᵀ  (r × d_h).
    //            B[i, j] = Σ_l qc_temp[i, l] · Q_gᵀ[l, j]
    //                    = Σ_l qc_temp[i, l] · Q_g[j, l]
    //                    = Σ_l qc_temp[i, l] · eigvecs_g[j*d_h + l]
    //    Write directly into b_out.
    for i in 0..r {
        for j in 0..d_h {
            let mut s = 0.0f64;
            let mut l = 0;
            while l + 4 <= d_h {
                s += qc_temp[i * d_h + l] * scratch.eigvecs_g[j * d_h + l];
                s += qc_temp[i * d_h + l + 1] * scratch.eigvecs_g[j * d_h + l + 1];
                s += qc_temp[i * d_h + l + 2] * scratch.eigvecs_g[j * d_h + l + 2];
                s += qc_temp[i * d_h + l + 3] * scratch.eigvecs_g[j * d_h + l + 3];
                l += 4;
            }
            while l < d_h {
                s += qc_temp[i * d_h + l] * scratch.eigvecs_g[j * d_h + l];
                l += 1;
            }
            b_out[i * d_h + j] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::jacobi_eigen;
    use super::*;

    /// Regression guard for the Issue 185 sign bug in `jacobi_eigen`. The
    /// original Plan 308 T2.3 code computed the rotation angle as
    /// `0.5 · atan(2·apq / (app − aqq))` but the working-matrix + eigvec
    /// updates use the rotation convention `J = [[c, s], [−s, c]]`, which
    /// requires `0.5 · atan(2·apq / (aqq − app))` (sign-flipped denominator).
    ///
    /// The bug was latent because `jacobi_eigen` had no callers before Issue
    /// 185. This test pins the corrected formula by checking eigenvector
    /// reconstruction on a matrix with `app ≠ aqq` — the case the original
    /// formula got wrong.
    #[test]
    fn jacobi_eigen_sign_convention_correct() {
        // 2x2 with app ≠ aqq: A = [[2, 0.3], [0.3, 2.5]].
        // True eigenvalues: (4.5 ± √0.61)/2 ≈ {2.6405, 1.8595}.
        let a = [2.0_f64, 0.3, 0.3, 2.5];
        let mut eigvals = vec![0.0_f64; 2];
        let mut eigvecs = vec![0.0_f64; 4];
        let mut scratch = vec![0.0_f64; 4];
        jacobi_eigen(&mut eigvals, &mut eigvecs, &a, &mut scratch, 2, 1e-15, 50);
        // Verify A · V[:,k] = λ[k] · V[:,k] for each k, using the documented
        // convention `eigvecs[i*r+j] = V[i, j]` (columns of V are eigenvectors).
        // For r=2: V[:,k] = [eigvecs[0*2+k], eigvecs[1*2+k]] = [eigvecs[k], eigvecs[2+k]].
        let mut max_err = 0.0_f64;
        for k in 0..2usize {
            // Eigenvector k: V[0,k], V[1,k].
            let v0k = eigvecs[k];
            let v1k = eigvecs[2 + k];
            // A · V[:,k] = [a00·v0k + a01·v1k, a10·v0k + a11·v1k].
            let av0 = a[0] * v0k + a[1] * v1k;
            let av1 = a[2] * v0k + a[3] * v1k;
            // λ[k] · V[:,k].
            let lv0 = eigvals[k] * v0k;
            let lv1 = eigvals[k] * v1k;
            let err = (av0 - lv0).abs().max((av1 - lv1).abs());
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-12,
            "jacobi_eigen sign bug regression: A·V ≠ Λ·V, max_err = {max_err:e}"
        );
    }

    /// Small parity check: the Jacobi B-step alone (single iteration, frozen A)
    /// should produce the same B as the Kronecker B-step on a tiny config
    /// where both are feasible. This is the unit-level check; the full
    /// end-to-end parity test against `low_rank_fit` is in
    /// `tests/karc_low_rank_jacobi_vs_kronecker.rs`.
    #[test]
    fn jacobi_b_step_matches_kronecker_small() {
        // Synthetic 6 × 6 Gram (SPD, well-conditioned, app ≠ aqq everywhere).
        let d_h = 6usize;
        let d_out = 2usize;
        let r = 2usize;
        let mut gram = vec![0.0f64; d_h * d_h];
        for i in 0..d_h {
            for j in 0..d_h {
                gram[i * d_h + j] = if i == j { 3.0 + (i as f64) * 0.1 } else { 0.25 };
            }
        }
        let mut cov = vec![0.0f64; d_h * d_out];
        for i in 0..d_h {
            for d in 0..d_out {
                cov[i * d_out + d] = (i as f64 + 0.1) * ((d as f64) + 0.5);
            }
        }
        let lambda = 1e-3f64;
        // Arbitrary frozen A (D × r).
        let a_frozen: Vec<f64> = (0..d_out * r).map(|i| (i as f64 + 1.0) * 0.13).collect();

        // Kronecker path: re-use the existing low_rank_fit_b_with_frozen_a.
        let mut b_kron = vec![0.0f64; r * d_h];
        let mut scr_kron = LowRankFitScratch::with_capacity(d_h, d_out, r);
        super::super::low_rank_fit_b_with_frozen_a(
            &gram,
            &cov,
            d_h,
            d_out,
            r,
            lambda,
            &a_frozen,
            &mut b_kron,
            &mut scr_kron,
        );

        // Jacobi path: eigendecompose G, then call jacobi_b_step.
        let mut b_jac = vec![0.0f64; r * d_h];
        let mut scr_jac = LowRankFitScratch::with_capacity(d_h, d_out, r);
        scr_jac.ensure_jacobi_capacity(d_h, r);
        let a_buf = a_frozen.clone();
        // Populate eigvals_g/eigvecs_g (one-time setup that
        // low_rank_fit_jacobi_with_init would normally do).
        jacobi_eigen(
            &mut scr_jac.eigvals_g[..d_h],
            &mut scr_jac.eigvecs_g[..d_h * d_h],
            &gram,
            &mut scr_jac.jacobi_scratch_g[..d_h * d_h],
            d_h,
            1e-15,
            50,
        );
        jacobi_b_step(
            &gram,
            &cov,
            d_h,
            d_out,
            r,
            lambda,
            &a_buf,
            &mut b_jac,
            &mut scr_jac,
        );

        // Both should satisfy the same normal equation (AᵀA)·B·G + λB = Aᵀ·Covᵀ
        // to machine precision. They use completely different float-operation
        // orderings (Kronecker Cholesky vs eigenbasis diagonalization) so we
        // don't require bit-identity, just agreement to ~1e-12.
        let mut max_diff = 0.0f64;
        for i in 0..r * d_h {
            let diff = (b_kron[i] - b_jac[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 1e-9,
            "Jacobi vs Kronecker B-step disagree: max_diff = {max_diff:e}"
        );
    }
}

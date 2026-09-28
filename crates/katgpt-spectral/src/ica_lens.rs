//! ICA Lens — FastICA Non-Gaussian Direction Mining + ERF Diagnostic.
//!
//! Implements the training-free interpretability method from
//! [ICA Lens (Liu & Han, arXiv:2606.11722, Jun 2026)][paper] —
//! finds maximally **non-Gaussian** directions in activation windows via
//! FastICA (Hyvärinen 1999), after centering + whitening.
//!
//! # Relationship to `EigenbasisTracker` (the whitening step)
//!
//! [`EigenbasisTracker`][super::hla_eigenbasis::EigenbasisTracker] ships
//! PCA — power iteration on the `D × D` Gram `G = W^T W` — which maximizes
//! *variance*. The downstream `excess_kurtosis()` ranker (Plan 203) is then
//! applied post-hoc to pick the most non-Gaussian PCA directions. That
//! heuristic approximates what ICA does *jointly and optimally*: maximize
//! non-Gaussianity directly via a fixed-point iteration on the whitened
//! data. On non-Gaussian data (which NPC activations are), ICA directions
//! are strictly more non-Gaussian than PCA directions ranked by kurtosis
//! post-hoc — the load-bearing claim validated by GOAT gate G2.
//!
//! ICA consumes the **same Gram + eigvecs** machinery from
//! [`EigenbasisScratch`][super::hla_eigenbasis::EigenbasisScratch] for its
//! whitening step, then runs the FastICA rotation on top.
//!
//! # The three stability recipes (paper §3.2)
//!
//! Naive scikit-learn FastICA is brittle on activations dominated by a few
//! large-norm outliers (the attention-sink regime). The paper's three
//! recipes make FastICA practical:
//!
//! 1. **Row-normalization** (`FastIcaConfig::row_normalize`, default `true`):
//!    scale each row to unit norm before centering/whitening. Reduces
//!    outlier-norm influence (+400% accepted layers in the paper).
//! 2. **p95-LIM acceptance** (`IcaAcceptance::P95`, default): accept the fit
//!    when the 95th percentile of per-component LIM values is below threshold,
//!    instead of the strict max. Rescues layers with a small unstable tail.
//! 3. **Adaptive refit** (`FastIcaConfig::adaptive_refit`, default `true`):
//!    halve the target component count `m` until acceptance, down to
//!    `min_components`. Returns the highest accepted resolution.
//!
//! # The ERF diagnostic (paper §4.2, novel)
//!
//! [`effective_receptive_field`] measures how much left context is sufficient
//! to recover a component's signed score at a target token. Small ERF →
//! token-local / reactive; large ERF → context-dependent / deliberative.
//! This is exactly the cognitive-hierarchy signal the latent_functor tier
//! router needs (token-local → plasma tier; context-dependent → hot tier).
//!
//! # Modelless design
//!
//! - No training, no gradients, no SAE dictionary learning. FastICA is a
//!   one-shot linear-algebra operation: center → whiten → fixed-point iteration.
//! - Caller-owned scratch (`FastIcaScratch`). After the first call for a given
//!   `(T, D, m)` triple, `fastica_into` allocates 0 bytes.
//! - Deterministic seed on every coordinate (mirrors `EigenbasisTracker`).
//!
//! # Sync-boundary rule
//!
//! The recovered ICA directions, like the PCA eigenbasis, stay local to the
//! NPC — never synced. Project to raw emotion scalars via a bridge function
//! before committing. See `hla_eigenbasis.rs` for the full rule.
//!
//! [paper]: https://arxiv.org/abs/2606.11722
//!
//! # Plan 475 references
//!
//! - Research: `.research/475_ICA_Lens_FastICA_Non_Gaussian_Directions.md`
//! - Plan: `.plans/475_ica_lens_fastica_primitive.md`
//! - Closest cousins: MAG (R397/P418, verdict-supervised), Within-Class
//!   Effective Rank (R394/P415, eigenvalue-spectrum), Rosetta Polarization
//!   (R180, kurtosis-as-monosemanticity).

#![allow(clippy::needless_range_loop)]

use crate::hla_eigenbasis::EigenbasisScratch;
use katgpt_core::simd::{simd_dot_f32, simd_outer_product_acc};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Acceptance rule for the FastICA fit (paper recipe A2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcaAcceptance {
    /// `max(LIM_j) ≤ τ`. Strictest. Rejects fits with any unstable component.
    Strict,
    /// `p95(LIM_j) ≤ τ` (paper recipe A2, default). Accepts fits with a small
    /// unstable tail; flags the tail components via `n_unstable`.
    P95,
}

/// Contrast function `g(u)` for the FastICA fixed-point iteration (paper §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcaContrast {
    /// `g(u) = tanh(u)`, `g'(u) = 1 − tanh²(u)`. Default. Robust, fast.
    LogCosh,
    /// `g(u) = u·exp(−u²/2)`, `g'(u) = (1 − u²)·exp(−u²/2)`.
    Exp,
    /// `g(u) = u³`, `g'(u) = 3·u²`. Better on sub-Gaussian sources.
    Cubic,
}

/// Configuration for [`fastica_into`]. Sensible defaults mirror the paper.
#[derive(Clone, Debug)]
pub struct FastIcaConfig {
    /// Target component count (the paper's `m`). Halved on adaptive refit
    /// down to `min_components`. Must be `≤ D`.
    pub n_components: usize,
    /// Max FastICA iterations per refit attempt.
    pub max_iters: u32,
    /// LIM convergence threshold (the paper's `τ`, default `1e-4`). LIM is
    /// `1 − |cos(angle between w_new and w_old)|`; values near 0 mean
    /// convergence.
    pub lim_threshold: f32,
    /// Row-normalize activations before whitening (paper recipe A1, default `true`).
    pub row_normalize: bool,
    /// Acceptance rule: Strict (max-LIM < τ) or P95 (p95-LIM < τ).
    pub acceptance: IcaAcceptance,
    /// Adaptive refit enabled (paper recipe A3, default `true`). When `true`,
    /// halve `n_components` on failure down to `min_components`.
    pub adaptive_refit: bool,
    /// Adaptive refit floor (default `16`). When `n_components` falls to this,
    /// stop halving.
    pub min_components: usize,
    /// Contrast function: LogCosh (default), Exp, or Cubic.
    pub contrast: IcaContrast,
}

impl Default for FastIcaConfig {
    fn default() -> Self {
        Self {
            n_components: 8,
            max_iters: 200,
            lim_threshold: 1e-4,
            row_normalize: true,
            acceptance: IcaAcceptance::P95,
            adaptive_refit: true,
            min_components: 16,
            contrast: IcaContrast::LogCosh,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Status of a FastICA fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastIcaStatus {
    /// All `n_components` directions converged within `lim_threshold` under
    /// the configured acceptance rule at the requested `m`.
    Converged,
    /// Converged after adaptive refit — `m_eff < n_components` requested.
    ConvergedRefit,
    /// Did not converge even at `min_components`. The directions are returned
    /// anyway (best effort); consumers should treat them with skepticism and
    /// gate downstream effects on `component_kurtosis` or `n_unstable`.
    Failed,
}

/// Result of a FastICA fit. Borrows the caller-owned output slices.
///
/// `m_eff` is the effective component count (after adaptive refit). Callers
/// should read only the first `m_eff` rows of `reading_map`, `m_eff` columns
/// of `writing_map`, and `m_eff` columns of `source_scores`.
pub struct FastIcaResult<'a> {
    /// Reading map `R = W · K` — `n_components × D` row-major. The first
    /// `m_eff` rows are valid; rows `[m_eff..n_components]` are zero/garbage.
    /// Projects normalized centered activations into signed component scores
    /// `S = X_c · R^T`.
    pub reading_map: &'a mut [f32],
    /// Writing map `D = R^†` (pseudoinverse) — `D × n_components` row-major.
    /// The first `m_eff` columns are valid. Column `j` (stride `n_components`)
    /// is the activation-space direction vector for component `j`.
    pub writing_map: &'a mut [f32],
    /// Source scores `S = X_c · R^T` — `T × n_components` row-major. The first
    /// `m_eff` columns are valid.
    pub source_scores: &'a mut [f32],
    /// Per-component excess kurtosis (length `n_components`; first `m_eff`
    /// are valid). Higher = more non-Gaussian. Computed via the local
    /// [`excess_kurtosis`] formula.
    pub component_kurtosis: &'a mut [f32],
    /// Per-component LIM (last-iteration magnitude) values (length
    /// `n_components`; first `m_eff` are valid). Near 0 = converged.
    pub component_lim: &'a mut [f32],
    /// Effective `m` (after adaptive refit). Number of valid rows/cols.
    pub m_eff: usize,
    /// Number of unstable components among the `m_eff` (LIM ≥ threshold).
    pub n_unstable: usize,
    /// Fit status.
    pub status: FastIcaStatus,
}

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

/// Caller-owned scratch for [`fastica_into`].
///
/// Holds the whitening matrix `K`, the rotation `W`, the compact reading map
/// `R = W·K`, plus work buffers. Resize happens only when `(T, D, m)`
/// changes between calls; subsequent calls for the same shape are
/// allocation-free (G4 gate target).
///
/// # Layout (for `T=512, D=8, m=8`)
///
/// - `window_buf`: `T*D = 4096` (centered + row-normalized copy of input)
/// - `whitening`: `D*D = 64` (the whitening matrix `K`)
/// - `reading`: `m*D = 64` (the compact reading map `R = W·K`)
/// - `reading_prev`: `m*D = 64` (previous-iteration R for LIM)
/// - `source_scores`: `T*m = 4096` (component scores)
/// - `lim`, `kurt`: `m = 8` each
/// - `proj_buf`, `g_buf`: `T = 512` each (projections + contrast values)
/// - `wwt`, `wwt_scratch`, `wwt_eigvecs`: `m*m = 64` each
/// - `wwt_eigvals`: `m = 8`
/// - `work_d`: `D = 8` (row buffer for orthogonalization)
/// - `aug`: `m*2m = 128` (Gauss-Jordan augmented matrix for pseudoinverse)
/// - `col_mean`: `D = 8` (centering mean)
/// - `eigenbasis_scratch`: `D*D + 2*D = 80`
///
/// Total ≈ 9400 floats ≈ 38 KB, reused for the harness lifetime.
#[derive(Clone, Debug, Default)]
pub struct FastIcaScratch {
    /// Centered + row-normalized copy of the input window (T × D).
    window_buf: Vec<f32>,
    /// The whitening matrix `K` (D × D, row-major).
    whitening: Vec<f32>,
    /// The compact reading map `R = W·K` (m × D, row-major). Mutated each
    /// iteration of the FastICA loop.
    reading: Vec<f32>,
    /// Previous-iteration reading map (m × D, row-major). Used to compute LIM.
    reading_prev: Vec<f32>,
    /// Source scores `S = X_c · R^T` (T × m, row-major).
    source_scores: Vec<f32>,
    /// Per-component LIM values (m).
    lim: Vec<f32>,
    /// Per-component excess kurtosis (m).
    kurt: Vec<f32>,
    /// Projections `w^T z_i` for the current component (T).
    proj_buf: Vec<f32>,
    /// Contrast-function values `g(projections)` (T).
    g_buf: Vec<f32>,
    /// `R · R^T` (m × m) for symmetric orthogonalization.
    rrt: Vec<f32>,
    /// Scratch for the Jacobi eigensolver (m × m, mutated).
    rrt_scratch: Vec<f32>,
    /// Eigvals of `R · R^T` (m).
    rrt_eigvals: Vec<f32>,
    /// Eigvecs of `R · R^T` (m × m, column `j` = eigenvector for
    /// `rrt_eigvals[j]`).
    rrt_eigvecs: Vec<f32>,
    /// Work buffer of length D (row accumulator for orthogonalization).
    work_d: Vec<f32>,
    /// Augmented matrix for Gauss-Jordan pseudoinverse (m × 2m).
    aug: Vec<f32>,
    /// Column mean for centering (D).
    col_mean: Vec<f32>,
    /// Eigenvectors of the D×D covariance (D×D, column-major). Used by
    /// `whiten_into` as the Jacobi eigvecs output.
    eigvecs_d: Vec<f32>,
    /// Eigenvalues of the D×D covariance (D). Used by `whiten_into`.
    cov_eigvals: Vec<f32>,
    /// Temp buffer for whitening the window in place: Z = X_c · K^T (T × D).
    /// Written then copied back into `window_buf`.
    z_buf: Vec<f32>,
    /// Scratch buffer for P95 acceptance sort (length m_req). Avoids
    /// per-call allocation in `p95_accepts_into`.
    p95_buf: Vec<f32>,
    /// Reuse `EigenbasisScratch` for the whitening Gram + power iteration.
    eigenbasis_scratch: EigenbasisScratch,
    /// Cached (T, D, m) — resize only on change.
    cached_t: usize,
    cached_d: usize,
    cached_m: usize,
}

impl FastIcaScratch {
    /// Construct empty scratch. Allocates lazily on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocate for a given `(T, D, m)` triple. Idempotent.
    pub fn with_capacity(t: usize, d: usize, m: usize) -> Self {
        let mut s = Self::default();
        s.ensure_capacity(t, d, m);
        s
    }

    /// Resize buffers if `(T, D, m)` changed since the last call.
    fn ensure_capacity(&mut self, t: usize, d: usize, m: usize) {
        if self.cached_t == t && self.cached_d == d && self.cached_m == m {
            return;
        }
        self.window_buf.resize(t * d, 0.0);
        self.whitening.resize(d * d, 0.0);
        self.reading.resize(m * d, 0.0);
        self.reading_prev.resize(m * d, 0.0);
        self.source_scores.resize(t * m, 0.0);
        self.lim.resize(m, 0.0);
        self.kurt.resize(m, 0.0);
        self.proj_buf.resize(t, 0.0);
        self.g_buf.resize(t, 0.0);
        self.rrt.resize(m * m, 0.0);
        self.rrt_scratch.resize(m * m, 0.0);
        self.rrt_eigvals.resize(m, 0.0);
        self.rrt_eigvecs.resize(m * m, 0.0);
        self.work_d.resize(d, 0.0);
        self.aug.resize(m * 2 * m, 0.0);
        self.col_mean.resize(d, 0.0);
        self.eigvecs_d.resize(d * d, 0.0);
        self.cov_eigvals.resize(d, 0.0);
        self.z_buf.resize(t * d, 0.0);
        self.p95_buf.resize(m, 0.0);
        self.eigenbasis_scratch = EigenbasisScratch::with_capacity_d(d);
        self.cached_t = t;
        self.cached_d = d;
        self.cached_m = m;
    }
}

// ---------------------------------------------------------------------------
// Contrast functions
// ---------------------------------------------------------------------------

/// Evaluate `g(u)` (contrast derivative) at a single point.
#[inline]
fn contrast_g_scalar(u: f32, contrast: IcaContrast) -> f32 {
    match contrast {
        IcaContrast::LogCosh => u.tanh(),
        IcaContrast::Exp => u * (-0.5 * u * u).exp(),
        IcaContrast::Cubic => u * u * u,
    }
}

/// Evaluate `g'(u)` (contrast second derivative) at a single point.
#[inline]
fn contrast_gp_scalar(u: f32, contrast: IcaContrast) -> f32 {
    match contrast {
        IcaContrast::LogCosh => {
            let t = u.tanh();
            1.0 - t * t
        }
        IcaContrast::Exp => (1.0 - u * u) * (-0.5 * u * u).exp(),
        IcaContrast::Cubic => 3.0 * u * u,
    }
}

// ---------------------------------------------------------------------------
// Whitening (ZCA via eigendecomposition of the covariance)
// ---------------------------------------------------------------------------

/// Compute the ZCA whitening matrix `K` (D × D, row-major) such that
/// `Z = X_c · K^T` has identity covariance, where `X_c` is the centered
/// activation matrix.
///
/// `K = Λ^{-1/2} · V^T` where `Σ = V Λ V^T` is the eigendecomposition of the
/// covariance `Σ = (1/T) X_c^T X_c`.
///
/// Reuses [`EigenbasisScratch`] for the Gram + eigenvector extraction.
///
/// Writes eigenvalues of `Σ` (descending) into `out_eigvals[..D]`.
#[allow(clippy::too_many_arguments)]
fn whiten_into(
    window: &[f32],
    t_dim: usize,
    d_dim: usize,
    out_k: &mut [f32],
    out_eigvals: &mut [f32],
    eigvecs_buf: &mut [f32], // length D*D, receives V column-major (V[row, col])
    scratch: &mut EigenbasisScratch,
) {
    scratch.with_gram_buffers(d_dim, |gram, _v, _w| {
        // ── 1. Build D×D covariance Σ = (1/T) X_c^T X_c.
        for g in gram.iter_mut() {
            *g = 0.0;
        }
        for r in 0..t_dim {
            let row = &window[r * d_dim..(r + 1) * d_dim];
            simd_outer_product_acc(gram, row, row, d_dim, d_dim);
        }
        // Scale to covariance (divide by T).
        let inv_t = 1.0 / t_dim as f32;
        for g in gram.iter_mut() {
            *g *= inv_t;
        }

        // ── 2. Jacobi eigendecomposition Σ = V Λ V^T.
        // Use eigvecs_buf as both the Jacobi scratch and the output eigvecs.
        // We need a separate scratch buffer — use out_k temporarily (it's
        // D*D and not yet written).
        //
        // Actually jacobi_eig_symmetric_into needs (a, n, sweeps, eigvals,
        // eigvecs, scratch). We have:
        //   - a = gram (D×D covariance)
        //   - eigvals = out_eigvals
        //   - eigvecs = eigvecs_buf
        //   - scratch = out_k (D×D, safe to overwrite; we recompute K below)
        jacobi_eig_symmetric_into(
            gram,
            d_dim,
            30,
            out_eigvals,
            eigvecs_buf,
            out_k, // scratch (mutated by Jacobi)
        );

        // ── 3. Compute K = Λ^{-1/2} · V^T into out_k.
        // K[i, j] = inv_sqrt(λ_i) · V[j, i]
        // V[row, col] = eigvecs_buf[row*D + col]  (column `col` is eigvec)
        // So V[j, i] = eigvecs_buf[j*D + i].
        for i in 0..d_dim {
            let inv_sqrt_lambda = if out_eigvals[i] > 1e-10 {
                1.0 / out_eigvals[i].sqrt()
            } else {
                0.0
            };
            for j in 0..d_dim {
                out_k[i * d_dim + j] = inv_sqrt_lambda * eigvecs_buf[j * d_dim + i];
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Jacobi eigendecomposition (f32, with eigenvectors)
// ---------------------------------------------------------------------------

/// Jacobi eigendecomposition of a symmetric `n × n` matrix `a` (f32).
///
/// Writes eigenvalues into `eigvals[..n]` and eigenvectors into
/// `eigvecs[..n*n]` (column `j` = eigenvector for `eigvals[j]`).
///
/// `scratch[..n*n]` is mutated (working copy of `a`).
///
/// Mirrors the f32 Jacobi pattern in `katgpt-spectral/src/river_valley.rs`
/// (eigenvalues-only variant) extended to track eigenvectors.
fn jacobi_eig_symmetric_into(
    a: &[f32],
    n: usize,
    max_sweeps: usize,
    eigvals: &mut [f32],
    eigvecs: &mut [f32],
    scratch: &mut [f32],
) {
    if n == 0 {
        return;
    }
    if n == 1 {
        eigvals[0] = a[0];
        eigvecs[0] = 1.0;
        return;
    }

    scratch[..n * n].copy_from_slice(&a[..n * n]);
    for i in 0..n {
        for j in 0..n {
            eigvecs[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    let sweeps = max_sweeps.max(20);
    for _ in 0..sweeps {
        let mut max_off = 0.0_f32;
        for p in 0..n {
            for q in (p + 1)..n {
                let val = scratch[p * n + q].abs();
                if val > max_off {
                    max_off = val;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = scratch[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = scratch[p * n + p];
                let aqq = scratch[q * n + q];

                let tau = (aqq - app) / (2.0 * apq);
                let t = 1.0f32.copysign(tau) / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                scratch[p * n + p] = app - t * apq;
                scratch[q * n + q] = aqq + t * apq;
                scratch[p * n + q] = 0.0;
                scratch[q * n + p] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = scratch[r * n + p];
                    let arq = scratch[r * n + q];
                    let new_rp = c * arp - s * arq;
                    let new_rq = s * arp + c * arq;
                    scratch[r * n + p] = new_rp;
                    scratch[p * n + r] = new_rp;
                    scratch[r * n + q] = new_rq;
                    scratch[q * n + r] = new_rq;
                }

                // Update eigenvectors: V → V · R(p, q, θ).
                for i in 0..n {
                    let vip = eigvecs[i * n + p];
                    let viq = eigvecs[i * n + q];
                    eigvecs[i * n + p] = c * vip - s * viq;
                    eigvecs[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }

    for i in 0..n {
        eigvals[i] = scratch[i * n + i].max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Symmetric orthogonalization of rectangular R (m × D) rows
// ---------------------------------------------------------------------------

/// Symmetric orthogonalize the rows of an `m × D` matrix `R` so that
/// `R · R^T = I` (rows become orthonormal).
///
/// Computes `M = R R^T = Q Λ Q^T`, then `R_new = M^{-1/2} · R` where
/// `M^{-1/2} = Q Λ^{-1/2} Q^T`. Then
/// `R_new R_new^T = M^{-1/2} M M^{-1/2} = I`. ✓
///
/// `r_prev_buf` is a length-`m*D` workspace used to hold a copy of R during
/// the in-place update (we read from the copy, write into R).
///
/// NOTE: the production FastICA uses deflationary Gram-Schmidt, not symmetric
/// orthogonalization. This function is retained for the parallel variant + the
/// unit test that verifies the orthogonalization math.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn symmetric_orthogonalize_rows_into(
    r: &mut [f32],
    r_prev_buf: &mut [f32],
    rrt: &mut [f32],
    rrt_scratch: &mut [f32],
    rrt_eigvals: &mut [f32],
    rrt_eigvecs: &mut [f32],
    work_d: &mut [f32],
    m: usize,
    d: usize,
) {
    // M = R · R^T (m × m).
    for i in 0..m {
        for j in 0..m {
            let row_i = &r[i * d..(i + 1) * d];
            let row_j = &r[j * d..(j + 1) * d];
            rrt[i * m + j] = simd_dot_f32(row_i, row_j, d);
        }
    }

    jacobi_eig_symmetric_into(rrt, m, 30, rrt_eigvals, rrt_eigvecs, rrt_scratch);

    // M^{-1/2}[i, j] = Σ_k Q[i, k] · inv_sqrt(λ_k) · Q[j, k] into rrt_scratch.
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0_f32;
            for k in 0..m {
                let inv_sqrt_lambda = if rrt_eigvals[k] > 1e-10 {
                    1.0 / rrt_eigvals[k].sqrt()
                } else {
                    0.0
                };
                acc += rrt_eigvecs[i * m + k] * inv_sqrt_lambda * rrt_eigvecs[j * m + k];
            }
            rrt_scratch[i * m + j] = acc;
        }
    }

    // Snapshot R into r_prev_buf (the matmul reads old R, writes new R).
    r_prev_buf[..m * d].copy_from_slice(&r[..m * d]);

    // R_new = M^{-1/2} · R_prev. Left-multiply.
    // R_new[i, l] = Σ_k M^{-1/2}[i, k] · R_prev[k, l]
    for i in 0..m {
        for l in 0..d {
            let mut acc = 0.0_f32;
            for k in 0..m {
                acc += rrt_scratch[i * m + k] * r_prev_buf[k * d + l];
            }
            work_d[l] = acc;
        }
        r[i * d..(i + 1) * d].copy_from_slice(&work_d[..d]);
    }
}

// ---------------------------------------------------------------------------
// Pseudoinverse (for the writing map D = R^†)
// ---------------------------------------------------------------------------

/// Compute the pseudoinverse `D = R^†` (D × m, row-major) of the reading map
/// `R` (m × D, row-major).
///
/// For `m ≤ D` (undercomplete ICA), `R^† = R^T · (R · R^T)^{-1}`. We compute
/// the inverse of the `m × m` matrix `R · R^T` via Gauss-Jordan elimination
/// on an augmented `[R R^T | I]` system.
///
/// Column `j` (stride `m`) of `D` is the activation-space direction vector
/// for component `j`.
#[allow(clippy::too_many_arguments)]
fn pseudoinverse_into(
    reading_map: &[f32],
    out_writing_map: &mut [f32],
    rrt: &mut [f32],
    aug: &mut [f32],
    m: usize,
    d: usize,
) {
    // R · R^T (m × m).
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0_f32;
            for k in 0..d {
                acc += reading_map[i * d + k] * reading_map[j * d + k];
            }
            rrt[i * m + j] = acc;
        }
    }

    // Augmented [R R^T | I] (m × 2m).
    for i in 0..m {
        for j in 0..m {
            aug[i * (2 * m) + j] = rrt[i * m + j];
            aug[i * (2 * m) + m + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    // Gauss-Jordan with partial pivoting.
    for col in 0..m {
        let mut pivot_row = col;
        let mut max_abs = aug[col * (2 * m) + col].abs();
        for row in (col + 1)..m {
            let val = aug[row * (2 * m) + col].abs();
            if val > max_abs {
                max_abs = val;
                pivot_row = row;
            }
        }
        if max_abs < 1e-12 {
            continue;
        }
        if pivot_row != col {
            for j in 0..(2 * m) {
                aug.swap(col * (2 * m) + j, pivot_row * (2 * m) + j);
            }
        }
        let pivot = aug[col * (2 * m) + col];
        let inv_pivot = 1.0 / pivot;
        for j in 0..(2 * m) {
            aug[col * (2 * m) + j] *= inv_pivot;
        }
        for row in 0..m {
            if row == col {
                continue;
            }
            let factor = aug[row * (2 * m) + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * m) {
                aug[row * (2 * m) + j] -= factor * aug[col * (2 * m) + j];
            }
        }
    }

    // D = R^T · (R R^T)^{-1}.
    // D[i, j] = Σ_k R[k, i] · (R R^T)^{-1}[k, j]
    //        = Σ_k reading_map[k*d + i] · aug[k*(2m) + m + j]
    for i in 0..d {
        for j in 0..m {
            let mut acc = 0.0_f32;
            for k in 0..m {
                acc += reading_map[k * d + i] * aug[k * (2 * m) + m + j];
            }
            out_writing_map[i * m + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Excess kurtosis (local copy — matches Plan 203's formula)
// ---------------------------------------------------------------------------

/// Compute the excess kurtosis of a slice of f32 samples.
///
/// `excess_kurtosis = (m_4 / m_2²) − 3` where `m_k` is the k-th central
/// moment. Returns 0 if `n < 4` or the variance is near zero.
///
/// Same formula as `excess_kurtosis` in
/// `katgpt-speculative/src/kurtosis_gate.rs` (Plan 203), reimplemented
/// locally to avoid a cross-crate dep. Higher = heavier-tailed = more
/// non-Gaussian.
#[inline]
pub fn excess_kurtosis(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n < 4.0 {
        return 0.0;
    }
    let mean: f32 = values.iter().copied().sum::<f32>() / n;
    let (m2, m4) = values.iter().fold((0.0_f32, 0.0_f32), |(m2, m4), &x| {
        let dev = x - mean;
        (m2 + dev * dev, m4 + dev * dev * dev * dev)
    });
    if m2 < 1e-12 {
        return 0.0;
    }
    let m2_avg = m2 / n;
    let m4_avg = m4 / n;
    (m4_avg / (m2_avg * m2_avg)) - 3.0
}

/// Compute the excess kurtosis of the projection of `window` onto `direction`.
///
/// Two-pass (mean then central moments); O(T·D) with no allocation.
pub fn excess_kurtosis_of_projection(
    window: &[f32],
    direction: &[f32],
    t_dim: usize,
    d_dim: usize,
) -> f32 {
    debug_assert_eq!(window.len(), t_dim * d_dim);
    debug_assert_eq!(direction.len(), d_dim);
    if t_dim < 4 {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for r in 0..t_dim {
        let row = &window[r * d_dim..(r + 1) * d_dim];
        sum += simd_dot_f32(row, direction, d_dim);
    }
    let mean = sum / t_dim as f32;
    let (m2, m4) = (0..t_dim).fold((0.0_f32, 0.0_f32), |(m2, m4), r| {
        let row = &window[r * d_dim..(r + 1) * d_dim];
        let p = simd_dot_f32(row, direction, d_dim);
        let dev = p - mean;
        (m2 + dev * dev, m4 + dev * dev * dev * dev)
    });
    if m2 < 1e-12 {
        return 0.0;
    }
    let inv_n = 1.0 / t_dim as f32;
    let m2_avg = m2 * inv_n;
    let m4_avg = m4 * inv_n;
    (m4_avg / (m2_avg * m2_avg)) - 3.0
}

// ---------------------------------------------------------------------------
// The primitive: fastica_into (zero-alloc hot path)
// ---------------------------------------------------------------------------

/// Run FastICA on a window of activations, producing the reading map
/// (component directions), writing map (pseudoinverse), source scores,
/// per-component kurtosis, and the fit status.
///
/// See the [module docs][self] for the algorithm and the three stability
/// recipes.
///
/// # Arguments
///
/// * `window` — `T × D` row-major activations.
/// * `t_dim` — number of ticks (`T`).
/// * `d_dim` — activation dimension (`D`).
/// * `config` — fit configuration. `n_components` must be `≤ D`.
/// * `scratch` — caller-owned; resized on first call for a new `(T, D, m)`.
/// * `out_reading_map` — length `n_components * D`, row-major. Receives `R`.
///   Only the first `m_eff` rows are valid.
/// * `out_writing_map` — length `D * n_components`, row-major. Receives `D = R^†`.
/// * `out_source_scores` — length `T * n_components`, row-major. Receives `S`.
/// * `out_component_kurtosis` — length `n_components`. Receives per-component
///   excess kurtosis.
/// * `out_component_lim` — length `n_components`. Receives per-component LIM.
///
/// # Zero-alloc contract
///
/// After the first call for a given `(T, D, m)`, allocates 0 bytes (all
/// scratch pre-allocated; output buffers caller-owned). G4 gate target.
///
/// # Determinism
///
/// Seed is deterministic (Hadamard-like sign pattern on `1/sqrt(m)`). The
/// only cross-platform variability is the SIMD reduction order, the same
/// surface `EigenbasisTracker` relies on.
#[allow(clippy::too_many_arguments)]
pub fn fastica_into<'a>(
    window: &[f32],
    t_dim: usize,
    d_dim: usize,
    config: &FastIcaConfig,
    scratch: &mut FastIcaScratch,
    out_reading_map: &'a mut [f32],
    out_writing_map: &'a mut [f32],
    out_source_scores: &'a mut [f32],
    out_component_kurtosis: &'a mut [f32],
    out_component_lim: &'a mut [f32],
) -> FastIcaResult<'a> {
    assert!(
        t_dim > 0 && d_dim > 0,
        "fastica_into: t_dim and d_dim must be positive, got T={t_dim}, D={d_dim}"
    );
    assert_eq!(
        window.len(),
        t_dim * d_dim,
        "fastica_into: window.len() {} != T*D = {}*{} = {}",
        window.len(),
        t_dim,
        d_dim,
        t_dim * d_dim
    );
    let m_req = config.n_components;
    assert!(
        m_req >= 1 && m_req <= d_dim,
        "fastica_into: n_components must be in [1, D]=[1, {d_dim}], got {m_req}"
    );
    assert_eq!(
        out_reading_map.len(),
        m_req * d_dim,
        "out_reading_map.len() {} != m*D",
        out_reading_map.len()
    );
    assert_eq!(
        out_writing_map.len(),
        d_dim * m_req,
        "out_writing_map.len() {} != D*m",
        out_writing_map.len()
    );
    assert_eq!(
        out_source_scores.len(),
        t_dim * m_req,
        "out_source_scores.len() {} != T*m",
        out_source_scores.len()
    );
    assert_eq!(
        out_component_kurtosis.len(),
        m_req,
        "out_component_kurtosis.len() {} != m",
        out_component_kurtosis.len()
    );
    assert_eq!(
        out_component_lim.len(),
        m_req,
        "out_component_lim.len() {} != m",
        out_component_lim.len()
    );

    let min_m = config.min_components.min(m_req).max(1);

    // Adaptive refit loop: try m_req, m_req/2, ..., min_m.
    let mut m_try = m_req;
    let mut status;

    loop {
        scratch.ensure_capacity(t_dim, d_dim, m_try);

        let fit_status = fastica_single_fit(
            window,
            t_dim,
            d_dim,
            m_try,
            config,
            scratch,
            out_reading_map,
            out_source_scores,
            out_component_kurtosis,
            out_component_lim,
        );

        let accepted = match config.acceptance {
            IcaAcceptance::Strict => count_unstable(&out_component_lim[..m_try], config) == 0,
            IcaAcceptance::P95 => {
                p95_accepts_into(&out_component_lim[..m_try], config, &mut scratch.p95_buf)
            }
        };

        if accepted {
            status = if m_try < m_req {
                FastIcaStatus::ConvergedRefit
            } else {
                FastIcaStatus::Converged
            };
            break;
        }

        status = fit_status;

        if !config.adaptive_refit || m_try <= min_m {
            break;
        }
        m_try = (m_try / 2).max(min_m);
    }

    // Compute the writing map for the final m_try (R^†).
    pseudoinverse_into(
        &out_reading_map[..m_try * d_dim],
        &mut out_writing_map[..d_dim * m_try],
        &mut scratch.rrt[..m_try * m_try],
        &mut scratch.aug[..m_try * 2 * m_req],
        m_try,
        d_dim,
    );

    let n_unstable_final = count_unstable(&out_component_lim[..m_try], config);

    FastIcaResult {
        reading_map: out_reading_map,
        writing_map: out_writing_map,
        source_scores: out_source_scores,
        component_kurtosis: out_component_kurtosis,
        component_lim: out_component_lim,
        m_eff: m_try,
        n_unstable: n_unstable_final,
        status,
    }
}

/// Single FastICA fit at fixed `m` — no adaptive refit.
///
/// Writes the reading map, source scores, kurtosis, and LIM for this `m`.
/// Returns `Converged` if the acceptance rule passes, else `Failed`.
#[allow(clippy::too_many_arguments)]
fn fastica_single_fit(
    window: &[f32],
    t_dim: usize,
    d_dim: usize,
    m: usize,
    config: &FastIcaConfig,
    scratch: &mut FastIcaScratch,
    out_reading_map: &mut [f32],
    out_source_scores: &mut [f32],
    out_kurtosis: &mut [f32],
    out_lim: &mut [f32],
) -> FastIcaStatus {
    assert!(m >= 1 && m <= d_dim);

    // ── 1. Copy + row-normalize + center into window_buf.
    if config.row_normalize {
        for r in 0..t_dim {
            let row_in = &window[r * d_dim..(r + 1) * d_dim];
            let row_out = &mut scratch.window_buf[r * d_dim..(r + 1) * d_dim];
            let norm = simd_dot_f32(row_in, row_in, d_dim).max(1e-30).sqrt();
            let inv_norm = 1.0 / norm;
            for j in 0..d_dim {
                row_out[j] = row_in[j] * inv_norm;
            }
        }
    } else {
        scratch.window_buf[..t_dim * d_dim].copy_from_slice(&window[..t_dim * d_dim]);
    }

    // Center: column mean.
    for j in 0..d_dim {
        scratch.col_mean[j] = 0.0;
    }
    for r in 0..t_dim {
        let row = &scratch.window_buf[r * d_dim..(r + 1) * d_dim];
        for j in 0..d_dim {
            scratch.col_mean[j] += row[j];
        }
    }
    let inv_t = 1.0 / t_dim as f32;
    for j in 0..d_dim {
        scratch.col_mean[j] *= inv_t;
    }
    for r in 0..t_dim {
        let row = &mut scratch.window_buf[r * d_dim..(r + 1) * d_dim];
        for j in 0..d_dim {
            row[j] -= scratch.col_mean[j];
        }
    }

    // ── 2. Whiten: K (D × D) into scratch.whitening.
    whiten_into(
        &scratch.window_buf,
        t_dim,
        d_dim,
        &mut scratch.whitening,
        &mut scratch.cov_eigvals,
        &mut scratch.eigvecs_d,
        &mut scratch.eigenbasis_scratch,
    );

    // ── 3. Whiten the window IN PLACE: Z = X_c · K^T replaces window_buf.
    // Z[i, j] = Σ_l X_c[i, l] · K[j, l] = simd_dot(X_c[i, 0..D], K[j, 0..D])
    // Write into z_buf to avoid aliasing, then copy back into window_buf.
    for i in 0..t_dim {
        let x_row = &scratch.window_buf[i * d_dim..(i + 1) * d_dim];
        for j in 0..d_dim {
            let k_row = &scratch.whitening[j * d_dim..(j + 1) * d_dim];
            scratch.z_buf[i * d_dim + j] = simd_dot_f32(x_row, k_row, d_dim);
        }
    }
    // Move Z into window_buf (X_c is no longer needed).
    scratch.window_buf[..t_dim * d_dim].copy_from_slice(&scratch.z_buf[..t_dim * d_dim]);

    // ── 4. Initialize W as identity in the first m columns of scratch.reading
    //    (m × D). Each component starts aligned with a different axis of
    //    whitened space — maximum diversity. Columns [m..D] are zero.
    for j in 0..m {
        for k in 0..d_dim {
            scratch.reading[j * d_dim + k] = if k == j { 1.0 } else { 0.0 };
        }
    }

    // ── 4. FastICA fixed-point iteration (DEFLATIONARY with Gram-Schmidt)
    //    in WHITENED space.
    //
    // Works on Z (stored in window_buf, T × D). W is stored in the first m
    // columns of scratch.reading (rows are length-D but only [0..m] used).
    //
    // For each component j = 0..m:
    //   1. Iterate up to max_iters:
    //      a. proj_buf[i] = Z[i, 0..m] · W[j, 0..m]
    //      b. mean_gp = (1/T) Σ g'(proj_buf[i])
    //      c. g_buf[i] = g(proj_buf[i])
    //      d. W_new[j,k] = (1/T) Σ g_buf[i]·Z[i,k] − mean_gp·W[j,k]  (for k in 0..m)
    //      e. Gram-Schmidt: subtract projections onto W[0..j-1].
    //      f. Normalize W[j,:] to unit norm.
    //      g. LIM[j] = 1 − |cos(new, old)|.
    //      h. If LIM[j] < threshold: break.
    let max_iters = config.max_iters as usize;
    for j in 0..m {
        for iter in 0..max_iters {
            // a. Projections (only first m dims of Z participate).
            let w_row = &scratch.reading[j * d_dim..j * d_dim + m];
            for i in 0..t_dim {
                let z_row = &scratch.window_buf[i * d_dim..i * d_dim + m];
                scratch.proj_buf[i] = simd_dot_f32(z_row, w_row, m);
            }

            // b. mean_gp.
            let mut sum_gp = 0.0_f32;
            for i in 0..t_dim {
                sum_gp += contrast_gp_scalar(scratch.proj_buf[i], config.contrast);
            }
            let mean_gp = sum_gp * inv_t;

            // c. g(projections).
            for i in 0..t_dim {
                scratch.g_buf[i] = contrast_g_scalar(scratch.proj_buf[i], config.contrast);
            }

            // Save old W[j,0..m] into reading_prev.
            scratch.reading_prev[j * d_dim..j * d_dim + m]
                .copy_from_slice(&scratch.reading[j * d_dim..j * d_dim + m]);

            // d. New W[j,k] = inv_T · Σ g_buf[i]·Z[i,k] − mean_gp·W_prev[j,k].
            // Loop i-outer, k-inner for sequential Z access (cache-friendly).
            for k in 0..m {
                scratch.work_d[k] = 0.0;
            }
            for i in 0..t_dim {
                let g = scratch.g_buf[i];
                let z_row = &scratch.window_buf[i * d_dim..i * d_dim + m];
                for k in 0..m {
                    scratch.work_d[k] += g * z_row[k];
                }
            }
            for k in 0..m {
                scratch.reading[j * d_dim + k] =
                    scratch.work_d[k] * inv_t - mean_gp * scratch.reading_prev[j * d_dim + k];
            }

            // e. Gram-Schmidt against W[0..j-1].
            for prev in 0..j {
                let (prev_part, rest) = scratch.reading.split_at_mut((prev + 1) * d_dim);
                let prev_w = &prev_part[prev * d_dim..prev * d_dim + m];
                let curr_offset = (j - prev - 1) * d_dim;
                let curr_w = &rest[curr_offset..curr_offset + m];
                let proj = simd_dot_f32(curr_w, prev_w, m);
                for k in 0..m {
                    rest[curr_offset + k] -= proj * prev_w[k];
                }
            }

            // f. Normalize.
            let curr_w = &scratch.reading[j * d_dim..j * d_dim + m];
            let norm = simd_dot_f32(curr_w, curr_w, m).max(1e-30).sqrt();
            let inv_norm = 1.0 / norm;
            for k in 0..m {
                scratch.reading[j * d_dim + k] *= inv_norm;
            }

            // g. LIM.
            let new_w = &scratch.reading[j * d_dim..j * d_dim + m];
            let old_w = &scratch.reading_prev[j * d_dim..j * d_dim + m];
            let dot = simd_dot_f32(new_w, old_w, m);
            let lim = 1.0 - dot.abs();
            out_lim[j] = lim;

            // h. Convergence.
            if lim < config.lim_threshold {
                break;
            }
            let _ = iter;
        }
    }

    // ── 5. Form R = W · K (m × D). Now we expand from whitened-space W (m × m)
    //    into activation-space R (m × D).
    for j in 0..m {
        // Save W[j, 0..m] into reading_prev.
        scratch.reading_prev[j * d_dim..j * d_dim + m]
            .copy_from_slice(&scratch.reading[j * d_dim..j * d_dim + m]);
        // R[j, l] = Σ_k W[j, k] · K[k, l]
        for l in 0..d_dim {
            let mut acc = 0.0_f32;
            for k in 0..m {
                acc += scratch.reading_prev[j * d_dim + k] * scratch.whitening[k * d_dim + l];
            }
            scratch.reading[j * d_dim + l] = acc;
        }
    }

    // ── 6. Source scores S = Z · W^T (T × m).
    // Z is in window_buf; W rows are in reading_prev[j, 0..m].
    // S[i,j] = Z[i, 0..m] · W[j, 0..m]
    for i in 0..t_dim {
        let z_row = &scratch.window_buf[i * d_dim..i * d_dim + m];
        for j in 0..m {
            let w_row = &scratch.reading_prev[j * d_dim..j * d_dim + m];
            out_source_scores[i * m + j] = simd_dot_f32(z_row, w_row, m);
        }
    }

    // ── 7. Per-component kurtosis of source scores.
    for j in 0..m {
        let mut sum = 0.0_f32;
        for i in 0..t_dim {
            sum += out_source_scores[i * m + j];
        }
        let mean = sum * inv_t;
        let (mut m2, mut m4) = (0.0_f32, 0.0_f32);
        for i in 0..t_dim {
            let dev = out_source_scores[i * m + j] - mean;
            let d2 = dev * dev;
            m2 += d2;
            m4 += d2 * d2;
        }
        if m2 < 1e-12 {
            out_kurtosis[j] = 0.0;
        } else {
            let m2_avg = m2 * inv_t;
            let m4_avg = m4 * inv_t;
            out_kurtosis[j] = (m4_avg / (m2_avg * m2_avg)) - 3.0;
        }
    }

    // ── 8. Copy R into the caller's output reading_map.
    for j in 0..m {
        for l in 0..d_dim {
            out_reading_map[j * d_dim + l] = scratch.reading[j * d_dim + l];
        }
    }

    // Status from acceptance.
    let accepted = match config.acceptance {
        IcaAcceptance::Strict => count_unstable(&out_lim[..m], config) == 0,
        IcaAcceptance::P95 => p95_accepts_into(&out_lim[..m], config, &mut scratch.p95_buf),
    };
    if accepted {
        FastIcaStatus::Converged
    } else {
        FastIcaStatus::Failed
    }
}

// ---------------------------------------------------------------------------
// Acceptance checks
// ---------------------------------------------------------------------------

/// Count components with LIM ≥ threshold.
#[inline]
fn count_unstable(lim: &[f32], config: &FastIcaConfig) -> usize {
    lim.iter().filter(|&&l| l >= config.lim_threshold).count()
}

/// p95 acceptance: the 95th percentile of LIM values is below threshold.
///
/// Copies into a temp Vec to sort — this allocates. For the alloc-free hot
/// path, callers can check `count_unstable` directly against `5% of m`.
#[inline]
fn p95_accepts_into(lim: &[f32], config: &FastIcaConfig, sort_buf: &mut [f32]) -> bool {
    if lim.is_empty() {
        return true;
    }
    sort_buf[..lim.len()].copy_from_slice(lim);
    let buf = &mut sort_buf[..lim.len()];
    buf.sort_by(|a, b| a.total_cmp(b));
    let idx = ((lim.len() as f32) * 0.95).ceil() as usize;
    let idx = idx.saturating_sub(1).min(lim.len() - 1);
    buf[idx] < config.lim_threshold
}

// ---------------------------------------------------------------------------
// ERF — Effective Receptive Field
// ---------------------------------------------------------------------------

/// Default ERF suffix schedule (paper uses k=1..11 for token windows; we use
/// an exponential schedule covering reactive → deliberative timescales at
/// 20 Hz tick: 1, 2, 4, 8, 16, 32, 64).
pub const DEFAULT_ERF_SCHEDULE: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

/// Compute the Effective Receptive Field of a single component at a target
/// position.
///
/// Given the component's signed score at the target under (a) full context
/// and (b) suffixes of increasing length, returns the minimum suffix length
/// `k` such that the component is in the top-N by absolute score AND
/// preserves its sign.
///
/// Returns the last schedule entry if no suffix recovers within the window.
///
/// # Arguments
///
/// * `scores_full` — the component's signed score at each position under full
///   context.
/// * `target_idx` — index of the target position.
/// * `suffix_scores` — the component's signed score at the target under each
///   suffix length in `schedule`. Length must equal `schedule.len()`.
/// * `schedule` — the suffix lengths to try (e.g. [`DEFAULT_ERF_SCHEDULE`]).
/// * `top_n` — the top-N threshold (paper uses 15). Component must be in the
///   top-N by absolute score under the suffix.
pub fn effective_receptive_field(
    scores_full: &[f32],
    target_idx: usize,
    suffix_scores: &[f32],
    schedule: &[usize],
    top_n: usize,
) -> usize {
    let full_score = scores_full[target_idx];
    let full_sign = full_score.signum();
    if full_sign == 0.0 {
        return schedule.first().copied().unwrap_or(1);
    }

    let mut abs_scores: Vec<f32> = scores_full.iter().map(|x| x.abs()).collect();
    abs_scores.sort_by(|a, b| b.total_cmp(a));
    let top_n_threshold = if top_n >= abs_scores.len() {
        0.0
    } else {
        abs_scores[top_n - 1]
    };

    for (k_idx, &k) in schedule.iter().enumerate() {
        if k_idx >= suffix_scores.len() {
            break;
        }
        let suffix_score = suffix_scores[k_idx];
        let suffix_sign = suffix_score.signum();
        if suffix_sign != full_sign && suffix_sign != 0.0 {
            continue;
        }
        if suffix_score.abs() >= top_n_threshold {
            return k;
        }
    }

    *schedule.last().unwrap_or(&1)
}

/// Batch ERF: average the per-evidence-example ERF for a component.
///
/// For each evidence example (target position), computes the ERF and returns
/// the mean. Lower mean ERF → token-local component; higher → context-dependent.
///
/// `activations_per_suffix[k_idx]` is the activation matrix (flat row-major)
/// observed under `schedule[k_idx]` tokens of context. All must have the same
/// `d_dim`.
pub fn erf_batch(
    reading_map_row: &[f32],
    activations_full: &[f32],
    activations_per_suffix: &[Vec<f32>],
    evidence_indices: &[usize],
    schedule: &[usize],
    top_n: usize,
    d_dim: usize,
) -> f32 {
    if evidence_indices.is_empty() {
        return 0.0;
    }
    let t_full = activations_full.len() / d_dim;
    let mut scores_full = vec![0.0_f32; t_full];
    for i in 0..t_full {
        let row = &activations_full[i * d_dim..(i + 1) * d_dim];
        scores_full[i] = simd_dot_f32(row, reading_map_row, d_dim);
    }

    let mut total_erf = 0.0_f32;
    for &target in evidence_indices {
        let mut suffix_scores = vec![0.0_f32; schedule.len()];
        for (k_idx, _k) in schedule.iter().enumerate() {
            if k_idx >= activations_per_suffix.len() {
                break;
            }
            let acts = &activations_per_suffix[k_idx];
            let t_suf = acts.len() / d_dim;
            if target >= t_suf {
                suffix_scores[k_idx] = 0.0;
                continue;
            }
            let row = &acts[target * d_dim..(target + 1) * d_dim];
            suffix_scores[k_idx] = simd_dot_f32(row, reading_map_row, d_dim);
        }
        total_erf +=
            effective_receptive_field(&scores_full, target, &suffix_scores, schedule, top_n) as f32;
    }
    total_erf / evidence_indices.len() as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple LCG for deterministic synthetic data.
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Make a synthetic non-Gaussian source: independent Laplace + Uniform
    /// components mixed by a random-ish orthogonal-ish matrix.
    fn make_synthetic_non_gaussian(t: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        // Sources: first half Laplace(0,1), second half Uniform[-√3, √3].
        let mut sources = vec![0.0_f32; t * d];
        for j in 0..d {
            for i in 0..t {
                let u1 = lcg_next(&mut rng);
                let u2 = lcg_next(&mut rng);
                sources[i * d + j] = if j < d / 2 {
                    let s = if u1 < 0.5 { -1.0 } else { 1.0 };
                    s * (1.0 - 2.0 * (u1 - 0.5).abs()).ln()
                } else {
                    (u2 * 2.0 - 1.0) * 3.0_f32.sqrt()
                };
            }
        }
        // Mixing matrix (d × d) with sign flips for determinism.
        let mut mix = vec![0.0_f32; d * d];
        for i in 0..d {
            for j in 0..d {
                let sign = if ((i + j) * (i + j + 1) / 2) & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
                mix[i * d + j] = sign * (1.0 + (lcg_next(&mut rng) - 0.5) * 0.5);
            }
        }
        // Observed = sources · mix^T.
        let mut observed = vec![0.0_f32; t * d];
        for i in 0..t {
            for j in 0..d {
                let mut acc = 0.0_f32;
                for k in 0..d {
                    acc += sources[i * d + k] * mix[k * d + j];
                }
                observed[i * d + j] = acc;
            }
        }
        observed
    }

    #[test]
    fn t1_9a_synthetic_non_gaussian_ica_finds_high_kurtosis_directions() {
        let t = 1024;
        let d = 8;
        let window = make_synthetic_non_gaussian(t, d, 42);

        let config = FastIcaConfig {
            n_components: d,
            max_iters: 200,
            lim_threshold: 1e-3,
            row_normalize: false,
            acceptance: IcaAcceptance::P95,
            adaptive_refit: false,
            min_components: d,
            contrast: IcaContrast::LogCosh,
        };
        let mut scratch = FastIcaScratch::new();
        let mut reading = vec![0.0_f32; d * d];
        let mut writing = vec![0.0_f32; d * d];
        let mut scores = vec![0.0_f32; t * d];
        let mut kurt = vec![0.0_f32; d];
        let mut lim = vec![0.0_f32; d];
        let result = fastica_into(
            &window,
            t,
            d,
            &config,
            &mut scratch,
            &mut reading,
            &mut writing,
            &mut scores,
            &mut kurt,
            &mut lim,
        );
        println!(
            "t1_9a: status={:?}, m_eff={}, n_unstable={}, kurt={:?}",
            result.status, result.m_eff, result.n_unstable, kurt
        );
        // The Laplace sources have excess kurtosis ≈ 3; the Uniform sources
        // have excess kurtosis ≈ −1.2. FastICA should recover directions with
        // kurtosis near these values.
        let max_kurt = kurt.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            max_kurt > 1.5,
            "expected max component kurtosis > 1.5 (Laplace sources have kurt≈3), got max={max_kurt:.4}, kurt={kurt:?}"
        );
        // Sanity: at least some directions should have positive excess kurtosis.
        let n_positive = kurt.iter().filter(|&&k| k > 0.5).count();
        assert!(
            n_positive >= 1,
            "expected ≥1 direction with kurtosis > 0.5, got {kurt:?}"
        );
    }

    #[test]
    fn t1_9b_gaussian_source_yields_low_kurtosis() {
        let t = 1024;
        let d = 8;
        let mut rng = 7u64;
        let mut window = vec![0.0_f32; t * d];
        for i in 0..t {
            for j in 0..d {
                let u1 = lcg_next(&mut rng).max(1e-10);
                let u2 = lcg_next(&mut rng);
                window[i * d + j] =
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            }
        }
        let config = FastIcaConfig {
            n_components: d,
            row_normalize: false,
            adaptive_refit: false,
            ..Default::default()
        };
        let mut scratch = FastIcaScratch::new();
        let mut reading = vec![0.0_f32; d * d];
        let mut writing = vec![0.0_f32; d * d];
        let mut scores = vec![0.0_f32; t * d];
        let mut kurt = vec![0.0_f32; d];
        let mut lim = vec![0.0_f32; d];
        let _ = fastica_into(
            &window,
            t,
            d,
            &config,
            &mut scratch,
            &mut reading,
            &mut writing,
            &mut scores,
            &mut kurt,
            &mut lim,
        );
        let mean_kurt: f32 = kurt.iter().sum::<f32>() / d as f32;
        println!("t1_9b: mean kurtosis on Gaussian = {mean_kurt:.4}");
        assert!(
            mean_kurt.abs() < 1.0,
            "Gaussian source should have near-zero mean kurtosis, got {mean_kurt:.4}"
        );
    }

    #[test]
    fn t1_10_determinism_bit_identical() {
        let t = 256;
        let d = 8;
        let window = make_synthetic_non_gaussian(t, d, 99);
        let config = FastIcaConfig {
            n_components: 4,
            row_normalize: false,
            adaptive_refit: false,
            ..Default::default()
        };
        let run_once = || -> (Vec<f32>, Vec<f32>) {
            let mut scratch = FastIcaScratch::new();
            let mut reading = vec![0.0_f32; 4 * d];
            let mut writing = vec![0.0_f32; d * 4];
            let mut scores = vec![0.0_f32; t * 4];
            let mut kurt = vec![0.0_f32; 4];
            let mut lim = vec![0.0_f32; 4];
            let _ = fastica_into(
                &window,
                t,
                d,
                &config,
                &mut scratch,
                &mut reading,
                &mut writing,
                &mut scores,
                &mut kurt,
                &mut lim,
            );
            (reading, scores)
        };
        let (r1, s1) = run_once();
        let (r2, s2) = run_once();
        assert_eq!(r1, r2, "reading_map not bit-identical across runs");
        assert_eq!(s1, s2, "source_scores not bit-identical across runs");
    }

    #[test]
    fn excess_kurtosis_laplace_is_three() {
        let mut rng = 5u64;
        let n = 100_000;
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                let u = lcg_next(&mut rng);
                let s = if u < 0.5 { -1.0 } else { 1.0 };
                s * (1.0 - 2.0 * (u - 0.5).abs()).ln()
            })
            .collect();
        let k = excess_kurtosis(&vals);
        assert!(
            (k - 3.0).abs() < 0.3,
            "Laplace excess kurtosis ≈ 3.0, got {k:.4}"
        );
    }

    #[test]
    fn excess_kurtosis_gaussian_is_zero() {
        let mut rng = 11u64;
        let n = 100_000;
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                let u1 = lcg_next(&mut rng).max(1e-10);
                let u2 = lcg_next(&mut rng);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        let k = excess_kurtosis(&vals);
        assert!(k.abs() < 0.1, "Gaussian excess kurtosis ≈ 0, got {k:.4}");
    }

    #[test]
    fn jacobi_eig_recovers_known_2x2() {
        let a = [2.0_f32, 1.0, 1.0, 2.0];
        let mut eigvals = vec![0.0_f32; 2];
        let mut eigvecs = vec![0.0_f32; 4];
        let mut scratch = vec![0.0_f32; 4];
        jacobi_eig_symmetric_into(&a, 2, 30, &mut eigvals, &mut eigvecs, &mut scratch);
        let mut sorted = eigvals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((sorted[0] - 1.0).abs() < 1e-4, "smaller eigval: {sorted:?}");
        assert!((sorted[1] - 3.0).abs() < 1e-4, "larger eigval: {sorted:?}");
    }

    #[test]
    fn jacobi_eig_identity_returns_ones() {
        let n = 3;
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut eigvals = vec![0.0_f32; n];
        let mut eigvecs = vec![0.0_f32; n * n];
        let mut scratch = vec![0.0_f32; n * n];
        jacobi_eig_symmetric_into(&a, n, 30, &mut eigvals, &mut eigvecs, &mut scratch);
        for &ev in &eigvals {
            assert!((ev - 1.0).abs() < 1e-5, "eigval should be 1, got {ev}");
        }
    }

    #[test]
    fn erf_token_local_returns_one() {
        let scores_full = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let suffix_scores = vec![0.9]; // k=1 recovers
        let erf = effective_receptive_field(&scores_full, 2, &suffix_scores, &[1, 2, 4], 3);
        assert_eq!(erf, 1);
    }

    #[test]
    fn erf_context_dependent_returns_large_k() {
        let scores_full = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let suffix_scores = vec![0.0, 0.0, 0.9]; // k=4 recovers
        let erf = effective_receptive_field(&scores_full, 2, &suffix_scores, &[1, 2, 4], 3);
        assert_eq!(erf, 4);
    }

    #[test]
    fn erf_unrecoverable_returns_k_max() {
        let scores_full = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        // Sign flips under every suffix.
        let suffix_scores = vec![-0.9, -0.9, -0.9];
        let erf = effective_receptive_field(&scores_full, 2, &suffix_scores, &[1, 2, 4], 3);
        assert_eq!(erf, 4);
    }

    #[test]
    fn symmetric_orthogonalize_rows_makes_them_orthonormal() {
        let m = 3;
        let d = 5;
        let mut r = vec![
            1.0_f32, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.3, 0.0, 0.0, 0.0, 0.3, 1.0, 0.2, 0.1,
        ];
        let mut r_prev = vec![0.0_f32; m * d];
        let mut rrt = vec![0.0_f32; m * m];
        let mut rrt_scratch = vec![0.0_f32; m * m];
        let mut rrt_eigvals = vec![0.0_f32; m];
        let mut rrt_eigvecs = vec![0.0_f32; m * m];
        let mut work_d = vec![0.0_f32; d];
        symmetric_orthogonalize_rows_into(
            &mut r,
            &mut r_prev,
            &mut rrt,
            &mut rrt_scratch,
            &mut rrt_eigvals,
            &mut rrt_eigvecs,
            &mut work_d,
            m,
            d,
        );
        for i in 0..m {
            for j in 0..m {
                let dot = simd_dot_f32(&r[i * d..(i + 1) * d], &r[j * d..(j + 1) * d], d);
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-3,
                    "R·R^T[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn pseudoinverse_round_trip_recovers_original() {
        // For a tall-skinny R (m × D with m < D), R · R^† · R should ≈ R.
        let m = 3;
        let d = 5;
        let r: Vec<f32> = vec![
            1.0, 0.5, 0.0, 0.2, 0.1, 0.0, 1.0, 0.3, 0.0, 0.4, 0.1, 0.0, 1.0, 0.5, 0.0,
        ];
        let mut d_map = vec![0.0_f32; d * m];
        let mut rrt = vec![0.0_f32; m * m];
        let mut aug = vec![0.0_f32; m * 2 * m];
        pseudoinverse_into(&r, &mut d_map, &mut rrt, &mut aug, m, d);
        // R · D = R · R^† = I (since R has full row rank in m-space).
        for i in 0..m {
            for j in 0..m {
                let mut acc = 0.0_f32;
                for k in 0..d {
                    acc += r[i * d + k] * d_map[k * m + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc - expected).abs() < 1e-3,
                    "R·R^†[{i},{j}] = {acc}, expected {expected}"
                );
            }
        }
    }
}

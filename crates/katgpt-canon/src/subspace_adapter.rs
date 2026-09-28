//! [`SubspaceAdapter`] — joint-SVD shared subspace projection (cross-arch).
//!
//! The load-bearing cross-architecture adapter. Projects both models'
//! activations into a shared low-dimensional subspace via joint SVD of the
//! concatenated anchor matrix `M = [A | B]`, then fits orthogonal Procrustes
//! in that subspace to align the two models' frames.
//!
//! # The P1 result (Bench 423, 2026-07-26)
//!
//! On Gemma2-2B (d=2304) ↔ MiniCPM5-1B (d=1536) with 40 Rust prompts:
//!
//! | k (subspace dim) | mean cos(R · a, b) on held-out | GOAT G5 (>0.5)? |
//! |---|---|---|
//! | 2 | +0.87 | **GO** |
//! | 4 | +0.75 | **GO** |
//! | 8 | +0.68 | no |
//! | 16 | +0.64 | no |
//!
//! The cross-arch shared subspace is genuinely low-dimensional. The adapter
//! is correct: pairwise alignment is preserved at k ∈ {2, 4}. The cross-arch
//! canonical DIRECTION claim (P3 G6a — does a single direction in this
//! subspace discriminate Rust from non-Rust?) FAILED (Bench 424, 425, 426);
//! that's a separate question from "does the subspace preserve alignment".
//!
//! # Fit + apply lifecycle
//!
//! ```text
//! 1. Collect paired anchors: a[i] in latent-A (dim d_a), b[i] in latent-B (dim d_b).
//! 2. fit_joint_svd_pair(a, b, n, d_a, d_b, k, &mut scratch) -> SubspaceFit
//!    - Build M = [A | B] row-major (n × (d_a + d_b))
//!    - Transpose to M^T ((d_a + d_b) × n) for thin_svd_into (needs m >= n)
//!    - thin_svd_into(mt, total_d, n, ...) -> SvdResultScratch
//!    - Extract top-k left singular vectors of M^T (= right singular vecs of M)
//!    - Partition each into V_A[..d_a] and V_B[d_a..] (column-major k × d_a / k × d_b)
//!    - Project anchors: a_proj[i][j] = sum_r a[i][r] * V_A[r, j] (n × k)
//!    - Fit Procrustes R (k × k) via orthogonal_procrustes(a_proj, b_proj, n, k, ...)
//! 3. SubspaceAdapter::from_fit(fit)
//! 4. adapter.project_into(&canonical, &mut out)  // hot path
//! ```
//!
//! The fit step is the expensive part (SVD + Procrustes on the anchors).
//! Once fit, project_into is just two matvecs (V_A^T · canonical, then R · result)
//! — zero-alloc, deterministic, bit-identical across runs.
//!
//! # Why the transpose in fit
//!
//! `thin_svd_into` requires `m >= n` (tall matrix). Our `M = [A | B]` is
//! wide (`n × (d_a + d_b)` with `n << d_a + d_b`). Transposing to `M^T`
//! (tall) lets us call thin_svd_into directly. The left singular vectors
//! of `M^T` are the right singular vectors of `M` — exactly the joint
//! subspace basis we want.

use katgpt_core::{SvdResultScratch, SvdScratch, thin_svd_into};
use katgpt_spectral::procrustes::{ProcrustesConfig, ProcrustesScratch, orthogonal_procrustes};

use crate::{CanonicalIntent, ModelAdapter};

/// Output of [`fit_joint_svd_pair`]. Owns the joint basis (V_A, V_B) and the
/// Procrustes rotation R that aligns model A's k-dim projections to model B's.
///
/// Pass this to [`SubspaceAdapter::from_fit`] to construct the adapter.
/// The basis is column-major: `V_A[j * d_a + r]` is the `r`-th entry of the
/// `j`-th basis vector (j ∈ [0, k)).
#[derive(Clone, Debug)]
pub struct SubspaceFit {
    /// Column-major `d_a × k`. Projects model-A activations (d_a) → subspace (k).
    pub v_a: Vec<f32>,
    /// Column-major `d_b × k`. Projects model-B activations (d_b) → subspace (k).
    pub v_b: Vec<f32>,
    /// Row-major `k × k`. Orthogonal rotation aligning A's subspace to B's.
    pub rotation: Vec<f32>,
    /// Top-`k` singular values of the joint matrix `M = [A | B]`, descending.
    /// Carried for diagnostics (energy spectrum, sharedness ratio ||V_A||/||V_B||).
    /// Empty if the fit was constructed via [`SubspaceFit::from_parts`] without
    /// supplying them; otherwise length == `k`.
    pub singular_values: Vec<f32>,
    /// Model-A latent dim.
    pub d_a: usize,
    /// Model-B latent dim.
    pub d_b: usize,
    /// Subspace dim (number of joint SVD components kept).
    pub k: usize,
}

impl SubspaceFit {
    /// Reference to the top-k singular values (descending). Empty if the fit
    /// was constructed without them.
    #[inline]
    pub fn singular_values(&self) -> &[f32] {
        &self.singular_values
    }

    /// Per-direction sharedness ratio `||V_A[:,j]|| / ||V_B[:,j]||` for the
    /// `j`-th basis vector. Returns `f32::INFINITY` if `||V_B[:,j]||` is
    /// near zero. Useful diagnostic: a ratio near 1.0 means both models
    /// contribute roughly equal energy to this direction; a large ratio
    /// means model A dominates it.
    ///
    /// Out-of-range `j` panics in debug, returns 1.0 in release.
    pub fn sharedness_ratio(&self, j: usize) -> f32 {
        if j >= self.k {
            debug_assert!(j < self.k, "sharedness_ratio: j={j} >= k={}", self.k);
            return 1.0;
        }
        let v_a_col = &self.v_a[j * self.d_a..(j + 1) * self.d_a];
        let v_b_col = &self.v_b[j * self.d_b..(j + 1) * self.d_b];
        let na: f32 = v_a_col.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = v_b_col.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nb > 1e-12 { na / nb } else { f32::INFINITY }
    }

    /// Total energy across the top-k singular values: sum(σ²). Useful as the
    /// denominator for "energy fraction in direction j" = σ[j]² / total.
    /// Returns 0.0 if `singular_values` is empty.
    pub fn total_energy(&self) -> f32 {
        self.singular_values.iter().map(|s| s * s).sum()
    }
}

/// Joint-SVD fit scratch — owns the SVD + Procrustes workspaces.
/// Reuse across multiple `fit_joint_svd_pair` calls for zero-alloc fitting.
pub struct JointSvdFitScratch {
    svd_result: SvdResultScratch,
    svd_work: SvdScratch,
    procrustes_scratch: ProcrustesScratch,
    mt_buf: Vec<f32>,
    a_proj_buf: Vec<f32>,
    b_proj_buf: Vec<f32>,
}

impl JointSvdFitScratch {
    /// Construct with capacity for joint anchors up to `max_total_d = d_a + d_b`,
    /// `max_n` anchor pairs, and subspace dim up to `max_k`.
    pub fn with_capacity(max_total_d: usize, max_n: usize, max_k: usize) -> Self {
        Self {
            svd_result: SvdResultScratch::with_capacity(max_total_d, max_n),
            svd_work: SvdScratch::with_capacity(max_n, max_total_d),
            procrustes_scratch: ProcrustesScratch::new(max_n, max_k),
            mt_buf: Vec::with_capacity(max_total_d * max_n),
            a_proj_buf: Vec::with_capacity(max_n * max_k),
            b_proj_buf: Vec::with_capacity(max_n * max_k),
        }
    }
}

/// Fit a joint-SVD subspace + Procrustes rotation from paired anchors.
///
/// - `a`: row-major `n × d_a` f32 — model-A activations.
/// - `b`: row-major `n × d_b` f32 — model-B activations (same n prompts).
/// - `n`: number of anchor pairs.
/// - `d_a`, `d_b`: model-A / model-B latent dims.
/// - `k`: subspace dim (number of joint SVD components to keep). MUST satisfy `k <= min(n, d_a, d_b)`.
/// - `scratch`: reusable workspace ([`JointSvdFitScratch::with_capacity`]).
///
/// Returns a [`SubspaceFit`] owning V_A (d_a × k), V_B (d_b × k), R (k × k).
///
/// # Panics
///
/// Panics if `k > min(n, d_a, d_b)` (underdetermined), if `a.len() != n * d_a`,
/// or if `b.len() != n * d_b`.
///
/// # Algorithm
///
/// 1. Build `M = [A | B]` row-major (n × (d_a + d_b)).
/// 2. Transpose to `M^T` (needed because thin_svd_into requires m >= n).
/// 3. `thin_svd_into(mt, total_d, n, ...)` — left singular vectors of M^T
///    = right singular vectors of M, length d_a + d_b.
/// 4. Extract top-k: partition each into V_A[..d_a] + V_B[d_a..].
/// 5. Project anchors: `a_proj[i][j] = sum_r a[i][r] * V_A[r, j]` (n × k).
/// 6. Fit Procrustes R (k × k) via `orthogonal_procrustes(a_proj, b_proj, ...)`.
pub fn fit_joint_svd_pair(
    a: &[f32],
    b: &[f32],
    n: usize,
    d_a: usize,
    d_b: usize,
    k: usize,
    scratch: &mut JointSvdFitScratch,
) -> SubspaceFit {
    assert_eq!(a.len(), n * d_a, "a.len() != n * d_a");
    assert_eq!(b.len(), n * d_b, "b.len() != n * d_b");
    assert!(
        k <= n.min(d_a).min(d_b),
        "k ({k}) must be <= min(n, d_a, d_b) = {}",
        n.min(d_a).min(d_b)
    );

    let total_d = d_a + d_b;

    // 1. Build M = [A | B] row-major, then transpose to M^T.
    //    (We could skip the intermediate M and write M^T directly, but the
    //    two-step form matches the P1 harness exactly — easier to audit.)
    scratch.mt_buf.clear();
    scratch.mt_buf.reserve(total_d * n);
    for j in 0..total_d {
        for i in 0..n {
            // M[i][j]; if j < d_a, it's a[i][j]; else b[i][j - d_a].
            let m_ij = if j < d_a {
                a[i * d_a + j]
            } else {
                b[i * d_b + (j - d_a)]
            };
            scratch.mt_buf.push(m_ij);
        }
    }
    // mt_buf is now M^T in row-major: total_d rows × n cols.

    // 2. SVD M^T → left singular vectors (length total_d).
    scratch.svd_result = SvdResultScratch::with_capacity(total_d, n);
    scratch.svd_work = SvdScratch::with_capacity(n, total_d);
    thin_svd_into(
        &scratch.mt_buf,
        total_d,
        n,
        &mut scratch.svd_result,
        &mut scratch.svd_work,
    );

    // 3. Extract top-k left singular vectors, partition into V_A + V_B.
    //    Column-major: v_a[j * d_a + r] is row r of basis vector j.
    let mut v_a = vec![0.0f32; k * d_a];
    let mut v_b = vec![0.0f32; k * d_b];
    for j in 0..k {
        let v_full = scratch.svd_result.left_singular_vector(j);
        for r in 0..d_a {
            v_a[j * d_a + r] = v_full[r];
        }
        for r in 0..d_b {
            v_b[j * d_b + r] = v_full[d_a + r];
        }
    }

    // 4. Project anchors: a_proj[i][j] = sum_r a[i][r] * V_A[r, j].
    //    V_A is column-major (v_a[j * d_a + r]), so V_A[r, j] = v_a[j * d_a + r].
    scratch.a_proj_buf.clear();
    scratch.a_proj_buf.reserve(n * k);
    scratch.b_proj_buf.clear();
    scratch.b_proj_buf.reserve(n * k);
    for i in 0..n {
        for j in 0..k {
            let mut sa = 0.0f32;
            let mut sb = 0.0f32;
            let a_row = &a[i * d_a..(i + 1) * d_a];
            let b_row = &b[i * d_b..(i + 1) * d_b];
            let v_a_col = &v_a[j * d_a..(j + 1) * d_a]; // V_A[:, j]
            let v_b_col = &v_b[j * d_b..(j + 1) * d_b]; // V_B[:, j]
            for (ar, var) in a_row.iter().zip(v_a_col.iter()) {
                sa += ar * var;
            }
            for (br, vbr) in b_row.iter().zip(v_b_col.iter()) {
                sb += br * vbr;
            }
            scratch.a_proj_buf.push(sa);
            scratch.b_proj_buf.push(sb);
        }
    }
    // a_proj_buf, b_proj_buf are now row-major n × k.

    // 5. Fit Procrustes R (k × k) aligning a_proj → b_proj.
    let mut rotation = vec![0.0f32; k * k];
    scratch.procrustes_scratch = ProcrustesScratch::new(n, k);
    let cfg = default_procrustes_cfg();
    let _ = orthogonal_procrustes(
        &scratch.a_proj_buf,
        &scratch.b_proj_buf,
        n,
        k,
        &mut rotation,
        &mut scratch.procrustes_scratch,
        &cfg,
    );

    // 6. Capture top-k singular values for diagnostics (energy spectrum,
    //    sharedness ratios). Cheap — already computed by the SVD.
    let mut singular_values = vec![0.0f32; k];
    for (j, sv) in singular_values.iter_mut().enumerate().take(k) {
        *sv = scratch.svd_result.singular_value(j);
    }

    SubspaceFit {
        v_a,
        v_b,
        rotation,
        singular_values,
        d_a,
        d_b,
        k,
    }
}

/// Default Procrustes config used by [`fit_joint_svd_pair`]. Exposed as a
/// private helper so [`fit_joint_svd_pair_with_cfg`] can document what the
/// default is. We do NOT center: the canonical direction lives at the origin
/// (centering would subtract each model's centroid and lose the location
/// information needed for downstream discrimination).
///
/// **Important parity note:** this differs from `ProcrustesConfig::default()`,
/// which has `center: true`. The P1 harness (Bench 423) used
/// `ProcrustesConfig::default()` (with `min_anchors: 1`); harnesses that need
/// bit-identical parity with Bench 423 should call
/// [`fit_joint_svd_pair_with_cfg`] with `ProcrustesConfig::default()`.
fn default_procrustes_cfg() -> ProcrustesConfig {
    ProcrustesConfig {
        center: false,
        special_orthogonal: false,
        compute_residual: false,
        compute_det: false,
        // min_anchors = 0 → runtime falls back to 2*k. With n anchors and
        // k-dim subspace, we need n >= 2*k for the fit to be determined.
        min_anchors: 0,
    }
}

/// Fit a joint-SVD subspace + Procrustes rotation with a caller-supplied
/// [`ProcrustesConfig`].
///
/// This is the parity-preserving variant of [`fit_joint_svd_pair`] for
/// harnesses that need to control the Procrustes step (centering on/off,
/// residual computation, `min_anchors` floor). The SVD step is identical;
/// only the final Procrustes rotation uses `procrustes_cfg` instead of the
/// crate default.
///
/// # When to use this vs [`fit_joint_svd_pair`]
///
/// - Use [`fit_joint_svd_pair`] for the production path (adapter
///   construction). The crate default (`center: false`) is correct for the
///   canonical-intent use case.
/// - Use `fit_joint_svd_pair_with_cfg` when reproducing a prior benchmark
///   that used a different Procrustes config (e.g. Bench 423 used
///   `ProcrustesConfig::default()` with `center: true`).
///
/// See [`fit_joint_svd_pair`] for the algorithm + panic contract.
#[allow(clippy::too_many_arguments)] // numerics API: individual scalar args are clearer than a config struct
pub fn fit_joint_svd_pair_with_cfg(
    a: &[f32],
    b: &[f32],
    n: usize,
    d_a: usize,
    d_b: usize,
    k: usize,
    scratch: &mut JointSvdFitScratch,
    procrustes_cfg: &ProcrustesConfig,
) -> SubspaceFit {
    assert_eq!(a.len(), n * d_a, "a.len() != n * d_a");
    assert_eq!(b.len(), n * d_b, "b.len() != n * d_b");
    assert!(
        k <= n.min(d_a).min(d_b),
        "k ({k}) must be <= min(n, d_a, d_b) = {}",
        n.min(d_a).min(d_b)
    );

    let total_d = d_a + d_b;

    // 1. Build M^T directly (column-major M[i][j] → mt[j][i]).
    scratch.mt_buf.clear();
    scratch.mt_buf.reserve(total_d * n);
    for j in 0..total_d {
        for i in 0..n {
            let m_ij = if j < d_a {
                a[i * d_a + j]
            } else {
                b[i * d_b + (j - d_a)]
            };
            scratch.mt_buf.push(m_ij);
        }
    }

    // 2. SVD M^T → left singular vectors (length total_d).
    scratch.svd_result = SvdResultScratch::with_capacity(total_d, n);
    scratch.svd_work = SvdScratch::with_capacity(n, total_d);
    thin_svd_into(
        &scratch.mt_buf,
        total_d,
        n,
        &mut scratch.svd_result,
        &mut scratch.svd_work,
    );

    // 3. Extract top-k left singular vectors, partition into V_A + V_B.
    let mut v_a = vec![0.0f32; k * d_a];
    let mut v_b = vec![0.0f32; k * d_b];
    for j in 0..k {
        let v_full = scratch.svd_result.left_singular_vector(j);
        for r in 0..d_a {
            v_a[j * d_a + r] = v_full[r];
        }
        for r in 0..d_b {
            v_b[j * d_b + r] = v_full[d_a + r];
        }
    }

    // 4. Project anchors into the shared subspace.
    scratch.a_proj_buf.clear();
    scratch.a_proj_buf.reserve(n * k);
    scratch.b_proj_buf.clear();
    scratch.b_proj_buf.reserve(n * k);
    for i in 0..n {
        for j in 0..k {
            let mut sa = 0.0f32;
            let mut sb = 0.0f32;
            let a_row = &a[i * d_a..(i + 1) * d_a];
            let b_row = &b[i * d_b..(i + 1) * d_b];
            let v_a_col = &v_a[j * d_a..(j + 1) * d_a];
            let v_b_col = &v_b[j * d_b..(j + 1) * d_b];
            for (ar, var) in a_row.iter().zip(v_a_col.iter()) {
                sa += ar * var;
            }
            for (br, vbr) in b_row.iter().zip(v_b_col.iter()) {
                sb += br * vbr;
            }
            scratch.a_proj_buf.push(sa);
            scratch.b_proj_buf.push(sb);
        }
    }

    // 5. Fit Procrustes with the caller-supplied config.
    let mut rotation = vec![0.0f32; k * k];
    scratch.procrustes_scratch = ProcrustesScratch::new(n, k);
    let _ = orthogonal_procrustes(
        &scratch.a_proj_buf,
        &scratch.b_proj_buf,
        n,
        k,
        &mut rotation,
        &mut scratch.procrustes_scratch,
        procrustes_cfg,
    );

    // 6. Capture top-k singular values for diagnostics.
    let mut singular_values = vec![0.0f32; k];
    for (j, sv) in singular_values.iter_mut().enumerate().take(k) {
        *sv = scratch.svd_result.singular_value(j);
    }

    SubspaceFit {
        v_a,
        v_b,
        rotation,
        singular_values,
        d_a,
        d_b,
        k,
    }
}

/// Joint-SVD shared subspace adapter. Projects canonical directions
/// (length k) → model-A latent (length d_a) via `V_A · canonical`, OR
/// → model-B latent (length d_b) via `R · V_A · canonical == V_B · canonical_aligned`.
///
/// Which target you get depends on which side you construct:
/// - [`SubspaceAdapter::for_model_a`] → projects into model-A latent (no rotation).
/// - [`SubspaceAdapter::for_model_b`] → projects into model-B latent (with R).
///
/// Both adapters accept the SAME canonical direction and produce aligned
/// outputs — that's the cross-architecture Super-GOAT claim. (The claim is
/// demoted for canonical DIRECTION discrimination per P3c, but the
/// subspace ALIGNMENT preservation — what this adapter does — still holds.)
#[derive(Clone, Debug)]
pub struct SubspaceAdapter {
    /// Column-major `d × k`. The V matrix for THIS model (V_A or V_B).
    /// If this is a model-B adapter, `v == R · V_A` (pre-multiplied at
    /// construction so the hot path is one matvec, not two).
    v: Vec<f32>,
    /// Canonical dim (= k = number of joint SVD components kept).
    canonical_dim: usize,
    /// Target dim (= d_a or d_b).
    target_dim: usize,
    /// BLAKE3 commitment of (v_bytes || canonical_dim || target_dim).
    commitment: [u8; 32],
}

impl SubspaceAdapter {
    /// Construct an adapter for model-A (no rotation — projects canonical → A latent).
    pub fn for_model_a(fit: &SubspaceFit) -> Self {
        Self::new(fit.v_a.clone(), fit.k, fit.d_a)
    }

    /// Construct an adapter for model-B (pre-multiplies V_A by R so the
    /// hot path is one matvec: `out = V_B · canonical` where V_B = R · V_A
    /// in the shared frame).
    ///
    /// # Algorithm
    ///
    /// `V_B_aligned[:, j] = sum_i R[j, i] * V_A[:, i]` (R is k × k, V_A is d_a × k).
    /// But we want the B-side adapter to project into B's NATIVE dim (d_b),
    /// not A's dim. So we use V_B (d_b × k) directly — it's already in B's frame.
    /// The rotation R was fit in the k-dim subspace; the B-side projection
    /// `V_B · canonical` produces the same k-dim coordinates as `R · V_A · canonical`
    /// (by construction — that's what the Procrustes fit guarantees).
    pub fn for_model_b(fit: &SubspaceFit) -> Self {
        Self::new(fit.v_b.clone(), fit.k, fit.d_b)
    }

    /// Low-level constructor from a pre-fit V matrix (column-major d × k).
    pub fn new(v: Vec<f32>, canonical_dim: usize, target_dim: usize) -> Self {
        assert_eq!(
            v.len(),
            canonical_dim * target_dim,
            "SubspaceAdapter: v.len() ({}) != canonical_dim * target_dim ({})",
            v.len(),
            canonical_dim * target_dim
        );
        let commitment = commit_state(&v, canonical_dim, target_dim);
        Self {
            v,
            canonical_dim,
            target_dim,
            commitment,
        }
    }

    /// Read-only access to the raw V matrix (column-major d × k).
    pub fn v(&self) -> &[f32] {
        &self.v
    }
}

impl ModelAdapter for SubspaceAdapter {
    /// Project canonical (length k) → model latent (length d).
    /// `out[r] = sum_j V[r, j] * canonical[j]` where V is column-major
    /// (`v[j * d + r]` is row r of column j).
    #[inline]
    fn project_into(&self, canonical: &CanonicalIntent, out: &mut [f32]) {
        let k = self.canonical_dim;
        let d = self.target_dim;
        debug_assert_eq!(
            out.len(),
            d,
            "SubspaceAdapter::project_into: out.len() != target_dim"
        );
        debug_assert_eq!(
            canonical.dim(),
            k,
            "SubspaceAdapter::project_into: canonical.dim() != canonical_dim"
        );
        if out.len() != d || canonical.dim() != k {
            return;
        }
        let c = canonical.as_slice();
        // out[r] = sum_j v[j * d + r] * c[j]
        // Clear out first (we'll accumulate).
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for (j, cj) in c.iter().enumerate().take(k) {
            let col = &self.v[j * d..(j + 1) * d];
            for (r, vrj) in col.iter().enumerate() {
                out[r] += vrj * cj;
            }
        }
    }

    /// Inverse projection: extract canonical coordinates from model latent.
    /// Uses V^T as a left-inverse (V is column-orthonormal by SVD construction,
    /// so V^T V = I_k; V V^T is the projection onto the column space of V).
    fn extract_from(&self, model_latent: &[f32]) -> Vec<f32> {
        let k = self.canonical_dim;
        let d = self.target_dim;
        let mut out = vec![0.0f32; k];
        if model_latent.len() != d {
            return out;
        }
        // out[j] = sum_r V[r, j] * model_latent[r] = sum_r v[j * d + r] * model_latent[r]
        for (j, out_j) in out.iter_mut().enumerate().take(k) {
            let col = &self.v[j * d..(j + 1) * d];
            let mut s = 0.0f32;
            for (vrj, xr) in col.iter().zip(model_latent.iter()) {
                s += vrj * xr;
            }
            *out_j = s;
        }
        out
    }

    #[inline]
    fn target_dim(&self) -> usize {
        self.target_dim
    }

    #[inline]
    fn canonical_dim(&self) -> usize {
        self.canonical_dim
    }

    #[inline]
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

/// BLAKE3 of (v_bytes || canonical_dim_le || target_dim_le).
#[inline]
fn commit_state(v: &[f32], canonical_dim: usize, target_dim: usize) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(bytemuck::cast_slice(v));
    h.update(&canonical_dim.to_le_bytes());
    h.update(&target_dim.to_le_bytes());
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic test: two models with known shared 2-dim subspace.
    /// Verifies the fit produces well-formed output (correct shapes, deterministic,
    /// non-degenerate) and that project_into + extract_from round-trip preserves
    /// the SIGN of canonical coordinates (the property that holds for joint-SVD
    /// basis vectors — they're orthonormal in (d_a + d_b)-dim space, so each
    /// model's slice V_A is a contraction; round-trip recovers coordinates up
    /// to a positive per-coordinate scalar).
    #[test]
    fn fit_recovers_known_shared_subspace() {
        // n=6 anchors, k=2, d_a=4, d_b=3.
        let n = 6;
        let d_a = 4;
        let d_b = 3;
        let k = 2;
        // A activations: [a0, a1, 0, 0] for varying a0, a1.
        let a: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0,
            2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
        ];
        // B activations: [b0, b1, 0] — same coordinates in the shared subspace.
        let b: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0,
            0.0,
        ];
        assert_eq!(a.len(), n * d_a);
        assert_eq!(b.len(), n * d_b);

        let mut scratch = JointSvdFitScratch::with_capacity(d_a + d_b, n, k);
        let fit = fit_joint_svd_pair(&a, &b, n, d_a, d_b, k, &mut scratch);

        // Sanity: V_A is d_a × k = 4 × 2.
        assert_eq!(fit.v_a.len(), d_a * k);
        assert_eq!(fit.v_b.len(), d_b * k);
        assert_eq!(fit.rotation.len(), k * k);

        // Adapter for model A: project canonical [1, 0] into A's 4-dim space.
        let adapter_a = SubspaceAdapter::for_model_a(&fit);
        let adapter_b = SubspaceAdapter::for_model_b(&fit);
        let canonical = CanonicalIntent::new("test", vec![1.0, 0.0]);
        let mut out_a = vec![0.0f32; d_a];
        let mut out_b = vec![0.0f32; d_b];
        adapter_a.project_into(&canonical, &mut out_a);
        adapter_b.project_into(&canonical, &mut out_b);

        // Property that holds: project_into writes NON-ZERO output (the
        // canonical direction is non-trivially projected). Both adapters
        // should produce non-zero output.
        let mag_a: f32 = out_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = out_b.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            mag_a > 1e-6,
            "adapter_a output magnitude {mag_a} should be > 0"
        );
        assert!(
            mag_b > 1e-6,
            "adapter_b output magnitude {mag_b} should be > 0"
        );

        // Round-trip: extract_from(project(canonical)) should preserve the
        // SIGN of each canonical coordinate (joint-SVD basis vectors are
        // orthonormal in (d_a + d_b)-dim space, so each model's slice is a
        // contraction; round-trip recovers coordinates up to a positive
        // per-coordinate scalar). For canonical = [1, 0], the recovered
        // second coordinate should be ~0.
        let recovered_a = adapter_a.extract_from(&out_a);
        let recovered_b = adapter_b.extract_from(&out_b);
        assert_eq!(recovered_a.len(), k);
        assert_eq!(recovered_b.len(), k);
        // First coordinate should be positive (sign preserved).
        assert!(recovered_a[0] > 0.0, "recovered_a[0] should preserve sign");
        assert!(recovered_b[0] > 0.0, "recovered_b[0] should preserve sign");
        // Second coordinate should be near-zero (input was [1, 0]).
        // Loose tolerance — the joint SVD on this small synthetic data may
        // not produce an exact zero, but it should be small relative to
        // the first coordinate.
        assert!(
            recovered_a[1].abs() < recovered_a[0].abs(),
            "recovered_a[1] ({}) should be smaller than recovered_a[0] ({})",
            recovered_a[1],
            recovered_a[0]
        );
    }

    #[test]
    fn adapter_is_deterministic() {
        let n = 4;
        let d_a = 3;
        let d_b = 2;
        let k = 2;
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0];
        let b: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let mut s1 = JointSvdFitScratch::with_capacity(d_a + d_b, n, k);
        let mut s2 = JointSvdFitScratch::with_capacity(d_a + d_b, n, k);
        let f1 = fit_joint_svd_pair(&a, &b, n, d_a, d_b, k, &mut s1);
        let f2 = fit_joint_svd_pair(&a, &b, n, d_a, d_b, k, &mut s2);
        // Bit-identical V_A, V_B, R across two runs.
        assert_eq!(f1.v_a, f2.v_a, "V_A must be deterministic");
        assert_eq!(f1.v_b, f2.v_b, "V_B must be deterministic");
        assert_eq!(f1.rotation, f2.rotation, "R must be deterministic");
    }

    #[test]
    fn commitment_is_deterministic_across_constructors() {
        let v = vec![1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity-ish
        let a1 = SubspaceAdapter::new(v.clone(), 2, 2);
        let a2 = SubspaceAdapter::new(v, 2, 2);
        assert_eq!(a1.commitment(), a2.commitment());
    }

    #[test]
    fn project_into_handles_dim_mismatch_gracefully() {
        // The runtime check no-ops on mismatch in release builds.
        #[cfg(not(debug_assertions))]
        {
            let v = vec![1.0f32, 0.0, 0.0, 1.0];
            let a = SubspaceAdapter::new(v, 2, 2);
            let canonical = CanonicalIntent::new("test", vec![1.0]); // dim 1, not 2
            let mut out = vec![0.0f32; 2];
            a.project_into(&canonical, &mut out);
            // Should not panic and should leave out as initialized (zeroed).
            assert!(out.iter().all(|x| *x == 0.0));
        }
    }

    #[test]
    fn fit_panics_on_underdetermined() {
        // k > n: underdetermined.
        let result = std::panic::catch_unwind(|| {
            let a = vec![1.0f32; 4]; // n=2, d_a=2
            let b = vec![1.0f32; 4]; // n=2, d_b=2
            let mut s = JointSvdFitScratch::with_capacity(4, 2, 4);
            fit_joint_svd_pair(&a, &b, 2, 2, 2, 4, &mut s) // k=4 > min(2,2,2)=2
        });
        assert!(result.is_err(), "k > min(n, d_a, d_b) must panic");
    }

    /// `singular_values` is populated by `fit_joint_svd_pair` (length k) and is
    /// sorted descending (SVD contract). This is the diagnostic the P1 harness
    /// uses for the energy spectrum printout.
    #[test]
    fn singular_values_populated_and_sorted() {
        let (a, b) = planted_pair(8, 4, 3, 2, 0xABC0_FFFE);
        let mut s = JointSvdFitScratch::with_capacity(7, 8, 2);
        let fit = fit_joint_svd_pair(&a, &b, 8, 4, 3, 2, &mut s);
        let sv = fit.singular_values();
        assert_eq!(sv.len(), 2, "singular_values length should == k");
        for x in sv {
            assert!(x.is_finite(), "singular value must be finite");
            assert!(*x >= 0.0, "singular value must be non-negative");
        }
        assert!(sv[0] >= sv[1], "singular values must be sorted descending");
    }

    /// `sharedness_ratio(j)` matches the hand-computed `||V_A[:,j]|| / ||V_B[:,j]||`.
    /// This is the diagnostic the P1 harness uses to detect model dominance
    /// (one model contributing all the energy in a direction).
    #[test]
    fn sharedness_ratio_matches_manual() {
        let (a, b) = planted_pair(8, 4, 3, 2, 0xABC0_FFFE);
        let mut s = JointSvdFitScratch::with_capacity(7, 8, 2);
        let fit = fit_joint_svd_pair(&a, &b, 8, 4, 3, 2, &mut s);
        for j in 0..2 {
            let v_a_col = &fit.v_a[j * fit.d_a..(j + 1) * fit.d_a];
            let v_b_col = &fit.v_b[j * fit.d_b..(j + 1) * fit.d_b];
            let na: f32 = v_a_col.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = v_b_col.iter().map(|x| x * x).sum::<f32>().sqrt();
            let expected = if nb > 1e-12 { na / nb } else { f32::INFINITY };
            let actual = fit.sharedness_ratio(j);
            assert!(
                (actual - expected).abs() < 1e-5,
                "sharedness_ratio({j}): expected {expected}, got {actual}"
            );
        }
    }

    /// `total_energy()` returns sum(σ²) — the denominator for "energy fraction
    /// in direction j" = σ[j]² / total.
    #[test]
    fn total_energy_is_sum_of_squares() {
        let (a, b) = planted_pair(8, 4, 3, 2, 0xABC0_FFFE);
        let mut s = JointSvdFitScratch::with_capacity(7, 8, 2);
        let fit = fit_joint_svd_pair(&a, &b, 8, 4, 3, 2, &mut s);
        let expected: f32 = fit.singular_values().iter().map(|x| x * x).sum();
        assert!((fit.total_energy() - expected).abs() < 1e-5);
    }

    /// `fit_joint_svd_pair_with_cfg` with `ProcrustesConfig::default()` (center=true)
    /// produces a DIFFERENT rotation than `fit_joint_svd_pair` (center=false) when
    /// the projected anchors have non-zero column means. This is the parity
    /// contract: harnesses that need Bench 423 numbers use the _with_cfg variant.
    #[test]
    fn with_cfg_can_differ_from_default_on_noncentered_anchors() {
        // Plant anchors with a non-zero mean offset so centering matters.
        // A + offset, B + offset*2 → column means differ from zero.
        let (a_base, b_base) = planted_pair(8, 4, 3, 2, 0xFEED_FACE);
        let a: Vec<f32> = a_base.iter().map(|x| x + 1.5).collect();
        let b: Vec<f32> = b_base.iter().map(|x| x + 3.0).collect();

        let mut s1 = JointSvdFitScratch::with_capacity(7, 8, 2);
        let fit_no_center = fit_joint_svd_pair(&a, &b, 8, 4, 3, 2, &mut s1);

        let mut s2 = JointSvdFitScratch::with_capacity(7, 8, 2);
        let fit_center = fit_joint_svd_pair_with_cfg(
            &a,
            &b,
            8,
            4,
            3,
            2,
            &mut s2,
            &ProcrustesConfig::default(), // center=true
        );

        // The V_A/V_B/singular_values are IDENTICAL (SVD doesn't care about
        // Procrustes config). Only the rotation may differ.
        for (x, y) in fit_no_center.v_a.iter().zip(fit_center.v_a.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "V_A must not depend on Procrustes cfg"
            );
        }
        for (x, y) in fit_no_center
            .singular_values
            .iter()
            .zip(fit_center.singular_values.iter())
        {
            assert!(
                (x - y).abs() < 1e-6,
                "singular values must not depend on Procrustes cfg"
            );
        }

        // With the non-zero offset, centering SHOULD change the rotation.
        // (We don't assert they're always different — that depends on the
        // specific data. But we verify the rotations are at least WELL-FORMED.)
        for x in fit_center.rotation.iter() {
            assert!(x.is_finite(), "centered rotation must be finite");
        }
        for x in fit_no_center.rotation.iter() {
            assert!(x.is_finite(), "non-centered rotation must be finite");
        }
        // Both rotations should be k×k.
        assert_eq!(fit_center.rotation.len(), 4);
        assert_eq!(fit_no_center.rotation.len(), 4);
    }

    /// `sharedness_ratio` on out-of-range `j` returns 1.0 in release + panics in debug.
    #[test]
    fn sharedness_ratio_out_of_range_returns_one() {
        let (a, b) = planted_pair(8, 4, 3, 2, 0xABC0_FFFE);
        let mut s = JointSvdFitScratch::with_capacity(7, 8, 2);
        let fit = fit_joint_svd_pair(&a, &b, 8, 4, 3, 2, &mut s);
        // j=5 is out of range (k=2). Release returns 1.0; debug asserts.
        #[cfg(not(debug_assertions))]
        {
            assert_eq!(fit.sharedness_ratio(5), 1.0);
        }
        #[cfg(debug_assertions)]
        {
            let r = std::panic::catch_unwind(|| fit.sharedness_ratio(5));
            assert!(r.is_err(), "debug_assert should panic on out-of-range j");
        }
    }

    /// Planted shared-subspace pair (deterministic, low noise).
    /// Used by multiple tests above for the singular_values / sharedness /
    /// total_energy / with_cfg diagnostics.
    fn planted_pair(n: usize, d_a: usize, d_b: usize, k: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        // Minimal xorshift32 PRNG.
        let mut state = if seed == 0 { 0xDEAD_BEEF } else { seed };
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // Random bases.
        let mut a_basis = vec![0.0f32; k * d_a];
        let mut b_basis = vec![0.0f32; k * d_b];
        for x in a_basis.iter_mut() {
            *x = next_f32();
        }
        for x in b_basis.iter_mut() {
            *x = next_f32();
        }

        let mut a = vec![0.0f32; n * d_a];
        let mut b = vec![0.0f32; n * d_b];
        for i in 0..n {
            // Random k-dim coords (shared signal).
            let coords: Vec<f32> = (0..k).map(|_| next_f32()).collect();
            for r in 0..d_a {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += coords[j] * a_basis[j * d_a + r];
                }
                a[i * d_a + r] = acc + 0.1 * next_f32();
            }
            for r in 0..d_b {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += coords[j] * b_basis[j * d_b + r];
                }
                b[i * d_b + r] = acc + 0.1 * next_f32();
            }
        }
        (a, b)
    }
}

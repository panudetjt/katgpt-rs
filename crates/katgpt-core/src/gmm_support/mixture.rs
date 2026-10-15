//! Diagonal-covariance Gaussian mixture — the density half of the support
//! gate (Plan 618 T1).
//!
//! This is the mixture-density extension of the
//! [`RegionSubspaceField`](crate::region_subspace::RegionSubspaceField)
//! class of Gaussian machinery (Plan 416): per-component diagonal
//! VARIANCES (`psi per region, not shared`), a true logsumexp MIXTURE
//! density (the shipped `membership_gates` is per-region sigmoid, not a
//! mixture), and — via [`em`](super::em) — EM refinement seeded from the
//! shipped deterministic k-means
//! ([`fit_codebook_kmeans_into`](crate::factorized_action::fit_codebook_kmeans_into)).
//! Per the Plan 618 verdict round 1 there is NO parallel GMM type with its
//! own invented commitment format: the artifact commitment reuses the
//! [`compute_field_commitment`](crate::region_subspace::compute_field_commitment)
//! convention (little-endian f32 fields in a pinned order, BLAKE3-32), with
//! the per-component variance block occupying the position the shared
//! `psi_inv` occupies there and no loadings block (a diagonal GMM has no
//! subspace).
//!
//! # Fit-space law (measured, 2026-10-04)
//!
//! The GMM must be fit/eval'd in the **PROJECTED** space (JL → E ≈ 32–64),
//! never raw hashed-bag D-space — raw space REJECTS Gaussianity universally
//! (0/154 per-label pools; riir-reflex Issue 066 PRE-CHECK, reflex
//! `bd18b86`). Fitting raw measures a distribution the data provably is
//! not. The projector lives in [`super::projector`].
//!
//! # Determinism
//!
//! The eval path is pure f32 arithmetic over fixed-order loops plus one
//! f64-accumulator logsumexp — no RNG, no HashMap, no allocation. Two
//! instances built from identical parameter bytes produce identical
//! outputs and identical commitments (pinned by tests + the GOAT bench).
//! Cross-platform bit-identity of the FIT path is limited by the platform
//! `exp`/`ln` inside the logsumexp (the Plan 618 risk row); the
//! commitment is the regression tripwire within a platform.
//!
//! # Sigmoid, not softmax
//!
//! Nothing here projects semantics through softmax: the mixture density
//! is a logsumexp (the defining normalization of a mixture — the
//! `distributional_steering` caveat-4 precedent), and downstream GATES
//! consume the log-ratio through `sigmoid` (see [`super::gate`]).

use blake3::Hasher;

/// `ln(2π)` — the per-dimension Gaussian normalizer constant.
const LN_2PI: f32 = 1.837_877_1;

/// Errors returned by [`DiagGmm::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MixtureError {
    /// A variance value `σ²[k][d]` is not strictly positive (or not finite).
    InvalidVariance,
    /// A weight (or log-weight) is not finite.
    InvalidWeight,
}

/// A diagonal-covariance Gaussian mixture over `R^E` with `K` components.
///
/// Density: `p(x) = Σ_k exp(log_weights[k]) · N(x; centroids[k],
/// diag(variances[k]))`. The per-component normalizer
/// `log_norm[k] = log_weights[k] − 0.5·Σ_d (ln 2π + ln σ²_kd)` is
/// precomputed at construction so the hot eval path is one Mahalanobis
/// fold + one logsumexp over `K`.
///
/// Zero heap allocations by construction — all fields are fixed-size
/// arrays (the [`RegionSubspaceField`](crate::region_subspace::RegionSubspaceField)
/// storage discipline).
#[derive(Debug, Clone)]
pub struct DiagGmm<const E: usize, const K: usize> {
    /// Component means `μ_k ∈ R^E`.
    pub centroids: [[f32; E]; K],
    /// Normalized mixture log-weights `log π_k` (sum of exp = 1).
    pub log_weights: [f32; K],
    /// Per-component per-dimension variances `σ²[k][d] > 0`.
    pub variances: [[f32; E]; K],
    /// Precomputed `log π_k − 0.5·Σ_d (ln 2π + ln σ²_kd)`.
    log_norm: [f32; K],
    /// `BLAKE3(centroids || log_weights || variances)` — LE f32, pinned
    /// order (the `compute_field_commitment` convention; see module docs).
    commitment: [u8; 32],
}

impl<const E: usize, const K: usize> DiagGmm<E, K> {
    /// Construct from raw parameters, validating variances and weights and
    /// NORMALIZING the weights (the density contract: `Σ exp(log π) = 1`).
    ///
    /// Normalization is log-space (logsumexp), so unnormalized but
    /// proportionally-correct weights are accepted — callers passing
    /// already-normalized weights get them back bit-identically up to the
    /// f32 logsumexp round-trip (weights of exactly `ln(1/K)` each pass
    /// through unchanged).
    pub fn new(
        centroids: [[f32; E]; K],
        mut log_weights: [f32; K],
        variances: [[f32; E]; K],
    ) -> Result<Self, MixtureError> {
        for &w in log_weights.iter() {
            if !w.is_finite() {
                return Err(MixtureError::InvalidWeight);
            }
        }
        for row in variances.iter() {
            for &v in row.iter() {
                if !v.is_finite() || v <= 0.0 {
                    return Err(MixtureError::InvalidVariance);
                }
            }
        }
        // Normalize: subtract the logsumexp (finite by the check above —
        // all-finite log-weights have a finite lse).
        let z = lse(&log_weights);
        for w in log_weights.iter_mut() {
            *w -= z;
        }
        Ok(Self::from_normalized(centroids, log_weights, variances))
    }

    /// Construct without validation, trusting that weights are normalized
    /// and variances positive. The normalizers and commitment are still
    /// computed. Used when the mixture comes from a trusted frozen artifact
    /// or from the EM fitter (which normalizes by construction).
    #[must_use]
    pub fn new_unchecked(
        centroids: [[f32; E]; K],
        log_weights: [f32; K],
        variances: [[f32; E]; K],
    ) -> Self {
        Self::from_normalized(centroids, log_weights, variances)
    }

    fn from_normalized(
        centroids: [[f32; E]; K],
        log_weights: [f32; K],
        variances: [[f32; E]; K],
    ) -> Self {
        let mut log_norm = [0f32; K];
        for (k, ln_k) in log_norm.iter_mut().enumerate() {
            let mut acc = log_weights[k];
            for &v in variances[k].iter() {
                acc -= 0.5 * (LN_2PI + v.ln());
            }
            *ln_k = acc;
        }
        let commitment = compute_mixture_commitment(&centroids, &log_weights, &variances);
        Self {
            centroids,
            log_weights,
            variances,
            log_norm,
            commitment,
        }
    }

    /// Re-compute the commitment over the current contents. Returns `false`
    /// on a mismatch (tampered or drifted artifact — the freeze/thaw
    /// tripwire).
    #[must_use]
    pub fn verify(&self) -> bool {
        // log_norm is derived; a commitment match over the source fields
        // implies parameter consistency (the RegionSubspaceField::verify
        // precedent for derived projectors).
        compute_mixture_commitment(&self.centroids, &self.log_weights, &self.variances)
            == self.commitment
    }

    /// The BLAKE3 content commitment.
    #[must_use]
    pub fn commitment(&self) -> &[u8; 32] {
        &self.commitment
    }

    /// Log-density of `x` under the mixture: `logsumexp_k` of the component
    /// log-likelihoods. Zero-alloc (stack scratch `[f32; K]`).
    #[inline]
    #[must_use]
    pub fn loglik_mixture(&self, x: &[f32; E]) -> f32 {
        let mut terms = [0f32; K];
        self.component_logliks(x, &mut terms);
        lse(&terms)
    }

    /// Component log-likelihoods into a caller stack buffer:
    /// `out[k] = log π_k − 0.5·Σ_d (ln 2π + ln σ²_kd) − 0.5·Mahalanobis_k(x)`.
    #[inline]
    pub fn component_logliks(&self, x: &[f32; E], out: &mut [f32; K]) {
        for (k, o) in out.iter_mut().enumerate() {
            let mut mahal = 0.0f32;
            for ((&xi, &cen), &var) in x
                .iter()
                .zip(self.centroids[k].iter())
                .zip(self.variances[k].iter())
            {
                let diff = xi - cen;
                mahal += diff * diff / var;
            }
            *o = self.log_norm[k] - 0.5 * mahal;
        }
    }

    /// Posterior responsibilities `p(k|x)` in log space into a caller stack
    /// buffer (the EM E-step input): `out[k] = component_loglik[k] −
    /// loglik_mixture(x)`. Not used on the gate hot path.
    pub fn log_responsibilities(&self, x: &[f32; E], out: &mut [f32; K]) {
        self.component_logliks(x, out);
        let total = lse(out);
        for o in out.iter_mut() {
            *o -= total;
        }
    }

    /// The ambient dimension `E`.
    #[inline]
    #[must_use]
    pub const fn dim(&self) -> usize {
        E
    }

    /// The number of components `K`.
    #[inline]
    #[must_use]
    pub const fn num_components(&self) -> usize {
        K
    }
}

/// BLAKE3 commitment over `centroids || log_weights || variances`
/// (little-endian f32, pinned order) — the
/// [`compute_field_commitment`](crate::region_subspace::compute_field_commitment)
/// convention extended to the mixture's parameter set (no loadings block —
/// a diagonal GMM carries no subspace; the per-component variance block
/// sits where the shared `psi_inv` sits in the field format).
#[must_use]
pub fn compute_mixture_commitment<const E: usize, const K: usize>(
    centroids: &[[f32; E]; K],
    log_weights: &[f32; K],
    variances: &[[f32; E]; K],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    for row in centroids.iter() {
        for &f in row.iter() {
            hasher.update(&f.to_le_bytes());
        }
    }
    for &f in log_weights.iter() {
        hasher.update(&f.to_le_bytes());
    }
    for row in variances.iter() {
        for &f in row.iter() {
            hasher.update(&f.to_le_bytes());
        }
    }
    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    out
}

/// Stable log-sum-exp with an f64 accumulator — the normalization
/// primitive for every weight path here (never `exp` of raw terms).
///
/// Lineage: twin of `distributional_steering::lse` (kept private there;
/// that module's feature is not implied for one helper). Cross-agreement
/// is pinned by `lse_agrees_with_distributional_steering` whenever both
/// features are compiled.
#[inline]
pub(crate) fn lse(xs: &[f32]) -> f32 {
    let mut m = f64::NEG_INFINITY;
    for &x in xs {
        let xd = x as f64;
        if xd > m {
            m = xd;
        }
    }
    if m == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let mut sum = 0.0f64;
    for &x in xs {
        sum += f64::exp((x as f64) - m);
    }
    (m + sum.ln()) as f32
}

//! Sketched Gaussianity Probe — multi-direction projection-normality for
//! embedding populations (Issue 681, Research 498 — SIGReg distilled from
//! training loss to inference-time diagnostic).
//!
//! Every shipped representation-health metric is **second-moment only**:
//! [`geometry::effective_rank`](super::geometry::effective_rank) (entropy of
//! covariance eigenvalues), `avg_cosine_similarity`,
//! `within_class_effective_rank`, riir-neuron-db `spectral_flatness`. A
//! population can be **full-rank and pass all of these while its marginals
//! are non-Gaussian**:
//!
//! - bimodal mixture `½N(−μe, σI) + ½N(+μe, σI)`: covariance `σ²I + μ²eeᵀ`
//!   → near-full erank, "healthy"; the projection onto `e` is two-point
//!   separated → any normality test rejects. (The "shard population that is
//!   two disjoint styles glued together" failure — exactly what a
//!   consolidation pipeline should catch before freezing.)
//! - heavy tails / outlier contamination (5% @ 10σ): outliers *inflate*
//!   eigenvalues — erank passes.
//! - discrete/quantized marginals (snap values): possible full-rank
//!   covariance, non-Gaussian in every direction.
//!
//! Meanwhile shipped assumptions depend on exactly this unchecked property:
//! `katgpt-band/src/band_conditioner.rs` — Fisher-z "requires approximate
//! Gaussianity of residuals" — no runtime guard.
//!
//! # The primitive (Cramér-Wold sketch)
//!
//! Project the population (n × d, caller-owned flat buffer) onto `A = 16`
//! fixed directions, run a 1D KS-vs-fitted-Gaussian per direction, aggregate.
//! The Cramér-Wold theorem: the set of ALL 1D projections determines the
//! distribution; a fixed finite sketch of 16 catches every *margin-wide*
//! departure (heavy tails, discreteness) and direction-specific departures
//! that align with a probed direction.
//!
//! ## Direction set (honest design note)
//!
//! - First `min(4, d)` directions are **coordinate axes** `e_0..e_3` — the
//!   cheap structured anchors that catch the canonical axis-aligned bimodal
//!   case at ANY d. A purely random sketch dilutes it: a Rademacher
//!   direction in d=64 has |cos| ≈ 1/8 with the mixture axis, so a moderate
//!   separation μ ≈ 2σ shrinks to 0.5σ in projection — invisible.
//! - Remaining 12 are **Rademacher ±1** directions (exact in f32),
//!   BLAKE3-derived from `(seed, direction index)` — deterministic, seedable,
//!   avalanche-mixed (the house BLAKE3-deterministic-table pattern).
//!
//! **Known blind spot:** a non-axis-aligned, moderate-strength bimodal
//! departure in high d can be missed by all 16 directions. The sketch is
//! the cheap always-on audit; [`katgpt-spectral`'s `ica_lens`](https://github.com/katopz/katgpt-rs)
//! (FastICA non-Gaussian direction mining, Plan 475) is the optimizing
//! *locator* a consumer runs when the sketch trips — they are complements,
//! and the `katgpt-core` leaf constraint (`rrq_quant`) forbids consuming it
//! from here anyway.
//!
//! ## Aggregate (statistically grounded, no tuned magic)
//!
//! `score = sigmoid(κ · ln(p_min / p₀))` where `p_min` is the Kolmogorov
//! complementary CDF at the worst direction's D (Numerical Recipes
//! formula — n-aware: the KS critical value scales 1/√n), `p₀ = 0.01` is a
//! multiple-comparison-aware center (with 16 directions, E[min p] under H₀
//! ≈ 0.06; 0.01 sits at the Bonferroni 0.05/16 ≈ 0.003 boundary's relaxed
//! side), and `κ = 1` (the log-multiple margin is already scale-free — a
//! linear-p margin would bottom out at sigmoid(−κ·p₀) for hard rejects).
//! score → 1 = Gaussian, → 0 = rejected; score = 0.5 exactly when the
//! worst direction's evidence sits at the 1% center.
//!
//! # Allocation discipline (G4)
//!
//! All state lives in [`GaussianityScratch`], allocated once at construction
//! (direction table + sort buffer) and reused across calls.
//! [`sketched_gaussianity`] performs zero heap allocation after construction.
//!
//! # Consumers (why now)
//!
//! 1. `band_conditioner` Fisher-z precondition guard (advisory field)
//! 2. riir-ai edge_lora hidden-space monitor (Issue 743)
//! 3. riir-neuron-db freeze-gate advisory (`FreezeGateReport` additive field
//!    — the bimodal-two-styles-before-freeze case)
//!
//! Feature-gated behind `gaussianity_probe` (opt-in until a consumer
//! promotes). Sigmoid, not softmax (per AGENTS.md).

use blake3::Hasher;

/// Number of probe directions (|A| in the Cramér-Wold sketch).
pub const N_DIRECTIONS: usize = 16;

/// Number of leading coordinate-axis anchor directions.
/// See module docs ("Direction set") for why the sketch is not purely random.
pub const N_AXIS_ANCHORS: usize = 4;

/// Aggregate center: worst-direction p-value at which `score = 0.5`.
/// Multiple-comparison-aware for |A| = 16 (module docs).
pub const P_CENTER: f32 = 0.01;

/// Aggregate sigmoid sharpness (applied to the LOG-multiple margin
/// `ln(p_min / P_CENTER)` — scale-free; see module docs).
pub const KAPPA: f32 = 1.0;

// ──────────────────────────────────────────────────────────────────────────
// Report
// ──────────────────────────────────────────────────────────────────────────

/// Sketched-gaussianity report for one embedding population.
///
/// All fields are plain `f32`/`usize` so the report is `Copy` and trivially
/// serializable for downstream consumers (freeze-gate advisories, training
/// monitors).
#[derive(Debug, Clone, Copy)]
pub struct GaussianityReport {
    /// Sigmoid-bounded aggregate ∈ (0, 1). 1 = Gaussian, → 0 = rejected.
    /// `sigmoid(KAPPA · ln(p_min / P_CENTER))` — see module docs.
    pub score: f32,
    /// Index of the direction with the largest KS D (0 = axis `e_0`).
    pub worst_direction: usize,
    /// Raw KS D-statistic per direction, ∈ [0, 1]. Index-aligned with the
    /// direction table (`direction(a)`); directly comparable with
    /// `katgpt_spectral::ks_d_statistic` on the same projected sample
    /// (bit-identical by construction — same algorithm).
    pub per_direction: [f32; N_DIRECTIONS],
    /// Kolmogorov complementary CDF at the worst D (n-aware p-value).
    pub min_p_value: f32,
}

// ──────────────────────────────────────────────────────────────────────────
// Scratch
// ──────────────────────────────────────────────────────────────────────────

/// Caller-owned scratch: the fixed direction table + the per-direction
/// sort buffer. Allocate once (audit cadence), reuse across calls —
/// [`sketched_gaussianity`] is zero-alloc after construction (G4).
#[derive(Debug, Clone)]
pub struct GaussianityScratch {
    /// `N_DIRECTIONS × d` Rademacher/axis direction table, row-major.
    directions: Box<[f32]>,
    /// Projection buffer for one direction (n samples), reused as the KS
    /// sort buffer (the `ks_d_statistic` single-scratch pattern).
    projections: Box<[f32]>,
    n_samples: usize,
    d: usize,
}

impl GaussianityScratch {
    /// Construct the scratch for `n_samples × d` populations at a given
    /// direction seed. Same `(n, d, seed)` → bit-identical table.
    pub fn new(n_samples: usize, d: usize, seed: u64) -> Self {
        assert!(n_samples > 0, "n_samples must be positive");
        assert!(d > 0, "d must be positive");
        let mut directions = vec![0.0f32; N_DIRECTIONS * d].into_boxed_slice();
        let n_axis = N_AXIS_ANCHORS.min(d);
        for a in 0..N_DIRECTIONS {
            let row = &mut directions[a * d..(a + 1) * d];
            if a < n_axis {
                // Axis anchor e_a: unit entry at position a.
                row[a] = 1.0;
            } else {
                // Rademacher ±1 from BLAKE3(seed ‖ a), extended by block
                // counter for d > 256 (32 output bytes = 256 signs per hash).
                let mut hasher = Hasher::new();
                hasher.update(&seed.to_le_bytes());
                hasher.update(&(a as u64).to_le_bytes());
                let mut block = hasher.finalize();
                let mut block_idx = 0usize;
                for (j, slot) in row.iter_mut().enumerate() {
                    if j > 0 && j % 256 == 0 {
                        let mut h = Hasher::new();
                        h.update(block.as_bytes());
                        h.update(&(block_idx as u64).to_le_bytes());
                        block = h.finalize();
                        block_idx += 1;
                    }
                    // 32 output bytes = 256 SIGNS (one per bit).
                    let bit_idx = j % 256;
                    let byte = block.as_bytes()[bit_idx / 8];
                    let bit = (byte >> (bit_idx % 8)) & 1;
                    *slot = if bit == 1 { 1.0 } else { -1.0 };
                }
            }
        }
        Self {
            directions,
            projections: vec![0.0f32; n_samples].into_boxed_slice(),
            n_samples,
            d,
        }
    }

    /// The direction vector for sketch index `a` (length `d`).
    ///
    /// Public so the cross-crate agreement test (katgpt-spectral) can
    /// reconstruct projections and compare against
    /// `katgpt_spectral::ks_d_statistic` directly.
    pub fn direction(&self, a: usize) -> &[f32] {
        assert!(a < N_DIRECTIONS, "direction index {a} out of range");
        &self.directions[a * self.d..(a + 1) * self.d]
    }

    /// Population size the scratch was built for.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Embedding dimension the scratch was built for.
    pub fn dim(&self) -> usize {
        self.d
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Probe
// ──────────────────────────────────────────────────────────────────────────

/// Run the sketched gaussianity probe over a flat `n × d` population.
///
/// `states.len()` must equal `scratch.n_samples() * scratch.dim()`.
/// Deterministic: a fixed `(states, scratch)` pair yields a bit-identical
/// report. Zero heap allocation (G4).
///
/// # Panics
/// Panics if `states.len()` does not match the scratch's `(n, d)` shape.
pub fn sketched_gaussianity(states: &[f32], scratch: &mut GaussianityScratch) -> GaussianityReport {
    let n = scratch.n_samples;
    let d = scratch.d;
    assert_eq!(
        states.len(),
        n * d,
        "states.len() {} != n {} × d {}",
        states.len(),
        n,
        d
    );

    let mut per_direction = [0.0f32; N_DIRECTIONS];
    let mut worst_direction = 0usize;
    let mut worst_d = 0.0f32;
    for (a, slot) in per_direction.iter_mut().enumerate() {
        let dir = &scratch.directions[a * d..(a + 1) * d];
        let proj = &mut scratch.projections[..];
        // Project: proj[i] = Σ_j states[i·d + j] · dir[j]. Axis anchors hit
        // the strided-gather degenerate case (one nonzero) — same loop keeps
        // the code branch-free; LLVM hoists what it can.
        for i in 0..n {
            let row = &states[i * d..(i + 1) * d];
            let mut acc = 0.0f64;
            for (j, &dv) in dir.iter().enumerate() {
                acc += (row[j] as f64) * (dv as f64);
            }
            proj[i] = acc as f32;
        }
        // Sort the projections in place; the KS core reads the sorted slice.
        proj.sort_by(|x, y| x.total_cmp(y));
        let d_stat = ks_d_sorted_inplace(proj);
        if d_stat > worst_d {
            worst_d = d_stat;
            worst_direction = a;
        }
        *slot = d_stat;
    }

    let p_min = kolmogorov_q(worst_d, n);
    // Log-multiple margin: ln(p_min / P_CENTER), clamped away from ln(0).
    // A hard reject (p ≈ 1e-30) → margin ≈ −67 → score ≈ 0; a clean accept
    // (p ≈ 0.5) → margin ≈ +3.9 → score ≈ 0.98.
    let p_clamped = p_min.clamp(1e-30, 1.0);
    let score = sigmoid(KAPPA * (p_clamped.ln() - P_CENTER.ln()));
    GaussianityReport {
        score,
        worst_direction,
        per_direction,
        min_p_value: p_min,
    }
}

// ──────────────────────────────────────────────────────────────────────────
// 1D KS core (verbatim port of katgpt-spectral spectral.rs ks_d_statistic)
// ──────────────────────────────────────────────────────────────────────────

/// Kolmogorov-Smirnov D-statistic of a 1D sample against a fitted Gaussian
/// `N(μ, σ)` (moments estimated from the sample itself).
///
/// Bit-identical port of `katgpt_spectral::spectral::ks_d_statistic`
/// (Plan 224 OAQG substrate): same sort, same f64 single-pass accumulation,
/// same Abramowitz-Stegun `normal_cdf`. The cross-crate agreement test
/// (katgpt-spectral `tests/gaussianity_agreement.rs`) pins the two
/// implementations together — this copy exists because the `katgpt-core`
/// leaf must not depend on `katgpt-spectral` (`rrq_quant` scalar-inversion
/// rule).
///
/// Zero-alloc: `sample` is copied into `scratch[..n]` and sorted there.
/// Returns D ∈ [0, 1] (D < 0.1 typical for Gaussian samples at n ≥ 256).
pub fn ks_d_vs_fitted_gaussian(sample: &[f32], scratch: &mut [f32]) -> f32 {
    let n = sample.len().min(scratch.len());
    if n == 0 {
        return 0.0;
    }

    scratch[..n].copy_from_slice(&sample[..n]);
    scratch[..n].sort_by(|a, b| a.total_cmp(b));
    ks_d_sorted_inplace(&mut scratch[..n])
}

/// In-place KS core: `sorted` holds the (already sorted) sample; computes
/// moments + fitted-Gaussian D over it. Shared by the public
/// [`ks_d_vs_fitted_gaussian`] and the probe loop (whose projection buffer
/// doubles as the sort scratch — the single-scratch discipline).
fn ks_d_sorted_inplace(sorted: &mut [f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &x in sorted.iter() {
        let v = x as f64;
        sum += v;
        sum_sq = v.mul_add(v, sum_sq);
    }
    let mean = sum / n as f64;
    let var = (sum_sq / n as f64 - mean * mean).max(0.0);
    let std = var.sqrt().max(1e-10);

    if std < 1e-10 {
        return 0.0; // constant sample → no distribution to compare
    }

    let mut max_d = 0.0f32;
    for (i, &x) in sorted.iter().enumerate() {
        let z = (x as f64 - mean) / std;
        let gaussian_cdf = normal_cdf(z) as f32;
        let empirical_cdf_upper = ((i + 1) as f32) / n as f32;
        let empirical_cdf_lower = (i as f32) / n as f32;

        let d_plus = (empirical_cdf_upper - gaussian_cdf).abs();
        let d_minus = (gaussian_cdf - empirical_cdf_lower).abs();
        max_d = max_d.max(d_plus).max(d_minus);
    }

    max_d
}

/// Standard normal CDF approximation (Abramowitz and Stegun).
/// Accurate to ~1e-7. (Verbatim from katgpt-spectral `spectral.rs`.)
fn normal_cdf(z: f64) -> f64 {
    if z < -8.0 {
        return 0.0;
    }
    if z > 8.0 {
        return 1.0;
    }

    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422804014327; // 1/sqrt(2π)
    let p = d
        * (-z * z / 2.0).exp()
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if z >= 0.0 { 1.0 - p } else { p }
}

/// Kolmogorov distribution complementary CDF `Q_KS(λ)` — the p-value of a
/// one-sample KS statistic `D` at sample size `n`, via the Numerical
/// Recipes asymptotic series (the same formula as the qmc test helper):
/// `λ = (√n + 0.12 + 0.11/√n) · D`, `Q = 2 Σ_{k≥1} (−1)^{k−1} e^{−2k²λ²}`.
///
/// The series only converges for λ ≳ 0.3; below that the p-value saturates
/// at 1.0 (a guard needs no precision where the verdict is a forgone
/// accept).
fn kolmogorov_q(d_stat: f32, n: usize) -> f32 {
    if n == 0 || d_stat <= 0.0 {
        return 1.0;
    }
    let sqrt_n = (n as f64).sqrt();
    let lambda = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d_stat as f64;
    if lambda < 0.3 {
        return 1.0;
    }
    let mut q = 0.0f64;
    for k in 1..=100u64 {
        let term = (-2.0 * (k * k) as f64 * lambda * lambda).exp();
        let signed = if k % 2 == 1 { term } else { -term };
        q += 2.0 * signed;
        if term < 1e-12 {
            break;
        }
    }
    q.clamp(0.0, 1.0) as f32
}

/// House sigmoid (f32). Saturates cleanly at |x| > ~88 in exp — here the
/// argument is bounded by KAPPA·(1+P_CENTER) < 9, so a plain form is exact.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ──────────────────────────────────────────────────────────────────────────
// Tests (G1 fixtures + determinism + KS sanity + G4 alloc)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rng;

    const N: usize = 1024;

    /// (i) Seeded isotropic Gaussian population, d=64.
    fn gaussian_population(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut out = vec![0.0f32; n * d];
        for v in out.iter_mut() {
            *v = rng.normal();
        }
        out
    }

    /// (ii) Bimodal mixture ½N(−μe_k, σI) + ½N(+μe_k, σI), μ=3σ, axis k.
    /// μ=3 is the operating point where BOTH halves of the blind-spot demo
    /// hold: population KS D ≈ 0.157 (hard reject) while the mixture spike
    /// consumes exactly one covariance eigenvalue (erank ≈ 80% of d — still
    /// "healthy" to a rank metric).
    fn bimodal_axis_population(n: usize, d: usize, axis: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
            for j in 0..d {
                out[i * d + j] = rng.normal();
            }
            out[i * d + axis] += sign * 3.0;
        }
        out
    }

    /// (iii) Radial heavy-tail: r=1 w.p. 0.95, r=10 w.p. 0.05; sample = r·(unit vec).
    /// Isotropic → covariance stays ~scalar·I (erank full), but EVERY 1D
    /// projection is a 0.95·N(0,1)+0.05·N(0,100) mixture — the margin-wide
    /// blind spot that passes every second-moment metric.
    fn radial_heavy_tail_population(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            let r = if rng.uniform() < 0.95 { 1.0 } else { 10.0 };
            // Random unit vector: normalize a d-dim normal sample.
            let mut norm = 0.0f64;
            for j in 0..d {
                let g = rng.normal() as f64;
                out[i * d + j] = g as f32;
                norm += g * g;
            }
            let inv = 1.0 / norm.sqrt().max(1e-12);
            for j in 0..d {
                out[i * d + j] *= r * inv as f32;
            }
        }
        out
    }

    /// (iv) Bernoulli lattice {0,1} coordinates, d=8 (CLT smoothing is weak
    /// at low d — projections stay visibly discrete).
    fn lattice_population(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut out = vec![0.0f32; n * d];
        for v in out.iter_mut() {
            *v = if rng.uniform() < 0.5 { 0.0 } else { 1.0 };
        }
        out
    }

    #[test]
    fn gaussian_fixture_accepts() {
        let d = 64;
        let states = gaussian_population(N, d, 42);
        let mut scratch = GaussianityScratch::new(N, d, 7);
        let report = sketched_gaussianity(&states, &mut scratch);
        assert!(
            report.score > 0.5,
            "Gaussian fixture must accept: score={} p_min={:.4} worst_d={:.4} dir={}",
            report.score,
            report.min_p_value,
            report.per_direction[report.worst_direction],
            report.worst_direction
        );
        for (a, &dd) in report.per_direction.iter().enumerate() {
            assert!(
                dd < 0.1,
                "Gaussian fixture direction {a} D={dd:.4} exceeds the ks_d_statistic normal band"
            );
        }
    }

    #[test]
    fn bimodal_axis0_rejects_at_anchor() {
        let d = 64;
        let states = bimodal_axis_population(N, d, 0, 43);
        let mut scratch = GaussianityScratch::new(N, d, 7);
        let report = sketched_gaussianity(&states, &mut scratch);
        assert!(
            report.score < 0.5,
            "bimodal fixture must reject: score={} p_min={:.3e}",
            report.score,
            report.min_p_value
        );
        assert_eq!(
            report.worst_direction, 0,
            "axis-0-aligned bimodal must be caught by the e_0 anchor (per_direction={:?})",
            report.per_direction
        );
        // The anchor sees the two-point separation strongly (population D
        // for μ=3σ ≈ 0.157; empirical at n=1024 within ±0.02).
        assert!(
            report.per_direction[0] > 0.12,
            "axis-0 D={:.4} should show two-point separation",
            report.per_direction[0]
        );
    }

    #[test]
    fn bimodal_axis3_rejects_at_anchor() {
        let d = 64;
        let states = bimodal_axis_population(N, d, 3, 44);
        let mut scratch = GaussianityScratch::new(N, d, 7);
        let report = sketched_gaussianity(&states, &mut scratch);
        assert!(report.score < 0.5, "score={}", report.score);
        assert_eq!(
            report.worst_direction, 3,
            "axis-3-aligned bimodal must be caught by the e_3 anchor"
        );
    }

    #[test]
    fn radial_heavy_tail_rejects_in_every_direction() {
        let d = 64;
        let states = radial_heavy_tail_population(N, d, 45);
        let mut scratch = GaussianityScratch::new(N, d, 7);
        let report = sketched_gaussianity(&states, &mut scratch);
        assert!(
            report.score < 0.5,
            "radial heavy-tail must reject: score={} p_min={:.3e}",
            report.score,
            report.min_p_value
        );
        // Margin-wide departure: at least 12 of 16 directions reject.
        let rejecting = report.per_direction.iter().filter(|&&dd| dd > 0.1).count();
        assert!(
            rejecting >= 12,
            "radial heavy-tail is margin-wide; only {rejecting}/16 directions D>0.1 ({:?})",
            report.per_direction
        );
    }

    #[test]
    fn lattice_d8_rejects() {
        let d = 8;
        let states = lattice_population(N, d, 46);
        let mut scratch = GaussianityScratch::new(N, d, 7);
        let report = sketched_gaussianity(&states, &mut scratch);
        assert!(
            report.score < 0.5,
            "lattice fixture must reject: score={} p_min={:.3e} per_direction={:?}",
            report.score,
            report.min_p_value,
            report.per_direction
        );
    }

    #[test]
    fn determinism_bit_identical_three_runs() {
        let d = 64;
        let states = radial_heavy_tail_population(N, d, 45);
        let mut scratch = GaussianityScratch::new(N, d, 7);
        let a = sketched_gaussianity(&states, &mut scratch);
        let b = sketched_gaussianity(&states, &mut scratch);
        let c = sketched_gaussianity(&states, &mut scratch);
        assert_eq!(a.per_direction, b.per_direction);
        assert_eq!(b.per_direction, c.per_direction);
        assert_eq!(a.score.to_bits(), b.score.to_bits());
        assert_eq!(b.score.to_bits(), c.score.to_bits());
        assert_eq!(a.min_p_value.to_bits(), c.min_p_value.to_bits());
        assert_eq!(a.worst_direction, c.worst_direction);
    }

    #[test]
    fn direction_table_deterministic_and_anchored() {
        let d = 32;
        let s1 = GaussianityScratch::new(16, d, 99);
        let s2 = GaussianityScratch::new(16, d, 99);
        assert_eq!(s1.direction(0), s2.direction(0), "same seed → same table");
        assert_eq!(s1.direction(10), s2.direction(10));
        let s3 = GaussianityScratch::new(16, d, 100);
        assert_ne!(
            s1.direction(10),
            s3.direction(10),
            "different seed → different Rademacher rows"
        );
        // Axis anchors: unit vectors.
        for a in 0..N_AXIS_ANCHORS {
            let dir = s1.direction(a);
            let expected: Vec<f32> = (0..d).map(|j| if j == a { 1.0 } else { 0.0 }).collect();
            assert_eq!(dir, expected.as_slice(), "anchor {a} must be e_{a}");
        }
        // Rademacher rows: every entry exactly ±1.
        for a in N_AXIS_ANCHORS..N_DIRECTIONS {
            for &v in s1.direction(a) {
                assert!(
                    v == 1.0 || v == -1.0,
                    "Rademacher direction {a} has non-±1 entry {v}"
                );
            }
        }
    }

    #[test]
    fn ks_core_two_point_rejects_and_perfect_gaussian_is_small() {
        // Two-point {−c, +c} sample: fitted σ=c, so the population D is
        // SCALE-INVARIANT at exactly 0.5 − Φ(−1) ≈ 0.341 for every c.
        // (Empirical D at n=256 sits within ±0.05 of it.)
        for &c in &[1.0f32, 5.0] {
            let sample: Vec<f32> = (0..256).map(|i| if i % 2 == 0 { -c } else { c }).collect();
            let mut scratch = vec![0.0f32; sample.len()];
            let d = ks_d_vs_fitted_gaussian(&sample, &mut scratch);
            assert!(
                (0.25..0.45).contains(&d),
                "two-point sample (c={c}) D={d:.4} outside the scale-invariant 0.341 band"
            );
        }
        // Constant sample → 0.5, NOT 0: the original spectral.rs
        // `if std < 1e-10 return 0.0` guard is DEAD CODE (std is already
        // clamped `.max(1e-10)` on the line above), so a constant sample
        // runs through with σ=1e-10, z=0 for every element → fitted CDF
        // 0.5 everywhere → D = |1 − 0.5| = 0.5 at the top order statistic.
        // The port keeps this bit-identical (agreement test trumps the
        // cleanup; noted for the spectral-side owner).
        let consts = vec![3.5f32; 64];
        let mut scratch2 = vec![0.0f32; 64];
        assert_eq!(ks_d_vs_fitted_gaussian(&consts, &mut scratch2), 0.5);
    }

    #[test]
    fn kolmogorov_q_monotone_and_saturated() {
        // p at D=0 → 1; small λ saturates at 1.
        assert_eq!(kolmogorov_q(0.0, 1024), 1.0);
        assert_eq!(kolmogorov_q(1e-6, 1024), 1.0);
        // Monotone decreasing in D.
        let p_small = kolmogorov_q(0.02, 1024);
        let p_large = kolmogorov_q(0.30, 1024);
        assert!(p_small > p_large, "{p_small} !> {p_large}");
        // Strong reject → p ≈ 0.
        assert!(kolmogorov_q(0.30, 1024) < 1e-5);
    }

    #[test]
    #[should_panic(expected = "states.len()")]
    fn shape_mismatch_panics() {
        let mut scratch = GaussianityScratch::new(8, 4, 1);
        let wrong = vec![0.0f32; 31]; // 8*4=32 ≠ 31
        let _ = sketched_gaussianity(&wrong, &mut scratch);
    }

    /// G4: zero allocations in steady state (the latent_confounder_audit
    /// pattern — the lib test binary installs `alloc::TrackingAllocator`
    /// under cfg(test, debug_assertions); skip with a message if absent).
    /// The accessors are debug_assertions-gated in alloc.rs — gate the test
    /// to match (pre-existing release-build break found by the mi_est
    /// release lib run, Plan 583; the alloc.rs module-docs convention).
    #[test]
    #[cfg(debug_assertions)]
    fn g4_zero_alloc_steady_state() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        let d = 64;
        let states = gaussian_population(N, d, 42);
        let mut scratch = GaussianityScratch::new(N, d, 7);

        // Sentinel: confirm the allocator is installed.
        reset_alloc_stats();
        let _sentinel: Vec<u8> = vec![0u8; 256];
        let (sent_count, _) = get_alloc_stats();
        if sent_count == 0 {
            eprintln!("g4_zero_alloc_steady_state: TrackingAllocator not installed — SKIPPED");
            return;
        }
        drop(_sentinel);

        // Warmup (lazy runtime state, if any).
        let _ = sketched_gaussianity(&states, &mut scratch);

        reset_alloc_stats();
        for _ in 0..100 {
            let _ = sketched_gaussianity(&states, &mut scratch);
        }
        let (count, bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "sketched_gaussianity must be alloc-free in steady state; \
             observed {count} allocations ({bytes} bytes) across 100 calls"
        );
    }
}

//! Phase 2 — sigmoid band-pass difficulty gate, three-regime triage, and
//! the Wilson CI on the learnable-share statistic.

use crate::sigmoid;

/// Sigmoid band-pass difficulty gate: `σ(κ(w−w_lo))·σ(κ(w_hi−w))`.
///
/// The house sigmoid-not-softmax band shape (Guide 340 §2): rises through
/// `w_lo`, falls through `w_hi`, peaks at the band center — a smooth,
/// monotone↑↓ membership in `[w_lo, w_hi]`. Values lie in `[0, 1]`, never
/// negative, never > 1. **Both ends of the interval are reachable in f32**
/// (the same Rust-vs-ℝ split the bridge module documents for its ranking
/// theorem — Lean pins the ideal strict `(0,1)` contract over ℝ:
/// `KatgptProof.HintRegret.bandGate_mem_Ioo`):
/// - far outside the band at large `κ`, the house `fast_sigmoid`'s ±40
///   early-exit yields exactly **0** (a usable hard-off);
/// - deep inside the band at large `κ`, both sigmoid arguments exceed ~17
///   and `1 − σ(x) < 2⁻²⁵` rounds `σ(x)` to exactly **1.0f32** (plain f32
///   rounding, no early-exit involved) — a usable hard-on.
///
/// `w` is the learner's per-composition Beta posterior mean (win rate
/// against this content composition) — the quantity the gate was designed
/// to band-pass. `κ` controls wall steepness:
/// `κ → ∞` recovers the crisp `[w_lo, w_hi]` indicator; `κ = 0` flattens to
/// `σ(0)·σ(0) = 0.25` everywhere.
///
/// Properties (property-tested in `tests`):
/// - in `[0, 1]` always; strictly inside `(0, 1)` except at the two f32
///   saturation surfaces (±40 early-exit → 0 far outside the band;
///   rounding → 1 deep inside the band at large `κ` — the Lean theorem
///   pins the ideal ℝ contract, the same Rust-vs-ℝ split the bridge
///   module documents);
/// - peaks at the band center `(w_lo + w_hi)/2`;
/// - monotone non-decreasing on `w < center`, non-increasing on
///   `w > center` (symmetric by construction: swapping the two sigmoid
///   arguments around the center maps the function onto itself);
/// - saturates toward 1 inside the band and toward 0 far outside, as `κ`
///   grows.
#[inline]
pub fn learnable_band_gate(w: f32, w_lo: f32, w_hi: f32, kappa: f32) -> f32 {
    sigmoid(kappa * (w - w_lo)) * sigmoid(kappa * (w_hi - w))
}

/// The three content regimes a curriculum triages into (Guide 340 §3).
///
/// Mutually exclusive and exhaustive over the `(r̂, R⁻)` plane — the
/// partition is property-tested (`triage` docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Regime {
    /// `r̂ ≥ τ_r` — the hint unlocks a large gain: learnable-hard content at
    /// the learner's exact frontier. **Offer it.**
    Frontier = 1,
    /// `r̂ < τ_r` and `R⁻ ≥ τ_ret` — the hint adds little AND the learner
    /// already succeeds unhinted: solved content. **Retire it** (freeze in
    /// the neuron-db sense, Guide 340 §"Rollout" P2/Mastered retirement).
    Mastered = 2,
    /// `r̂ < τ_r` and `R⁻ < τ_ret` — the hint adds little AND the learner
    /// fails unhinted: never-winnable for this learner. **Evict, don't farm
    /// it** (the CGSP-conflation failure mode this discriminator exists to
    /// fix).
    Intractable = 3,
}

/// Three-regime triage over the `(r̂, R⁻)` plane (Guide 340 §3 — the
/// authoritative partition; the landed mmorpg consumer
/// `frontier_regime_of` is the deterministic collapse of exactly this).
///
/// - `r_hat` — the estimated hint regret (this module's sign convention:
///   unhinted − hinted; high = the hint's value is high).
/// - `unhinted_return` — `R⁻`, the no-hint baseline return/win-rate.
/// - `tau_r` — regret threshold: `r̂ ≥ τ_r` → [`Regime::Frontier`].
/// - `tau_ret` — return threshold splitting the low-regret half:
///   `R⁻ ≥ τ_ret` → [`Regime::Mastered`], else [`Regime::Intractable`]
///   (named to match the landed consumer's `FRONTIER_TAU_RET`).
///
/// Boundary pins (property-tested): `r̂ == τ_r` is Frontier (`≥`);
/// `R⁻ == τ_ret` is Mastered (`≥`). Exactly one regime per input — the three
/// cells cover the plane with no overlap.
///
/// **Plan-text deviation, documented:** the Plan 576 draft signature carried
/// a fifth `r_floor` argument that the guide's partition (and the landed
/// consumer) do not use — the guide is the spec, so the primitive ships the
/// 4-argument canonical form. Noise-floor handling belongs to the estimator
/// (stop when the CI half-width exceeds `r̂`), not to the partition.
#[inline]
pub fn triage(r_hat: f32, unhinted_return: f32, tau_r: f32, tau_ret: f32) -> Regime {
    if r_hat >= tau_r {
        Regime::Frontier
    } else if unhinted_return >= tau_ret {
        Regime::Mastered
    } else {
        Regime::Intractable
    }
}

/// Wilson score confidence interval for a proportion (Brown, Cai & DasGupta
/// 2001) — the CI used on the **learnable-share** statistic (the fraction
/// of offered content whose win rate sits in the learnable band; Guide 340
/// §"Validation protocol" — UQ honesty on the signature metric).
///
/// Correct coverage at small `n` and near the 0/1 boundaries where the
/// normal approximation fails. `z` is the two-sided critical value
/// (1.96 for 95%). Returns `(low, high)` clamped to `[0, 1]`; `n == 0`
/// returns the uninformative `(0, 1)`.
///
/// Byte-identical formula to `speculative::qmc::BootstrapEstimate::wilson_ci`
/// (the f64 twin) — that one is a method over its bootstrap struct; this is
/// the free function the learnable-share statistic needs. Kept local rather
/// than extracting a shared util: the qmc form reads its struct fields and
/// the leaf constraint would turn a 10-line function into a cross-module
/// dependency for zero arithmetic difference.
#[inline]
pub fn wilson_score_ci(p_hat: f64, n: u64, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let n = n as f64;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let margin = (z / denom) * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt();
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    const W_LO: f32 = 0.2;
    const W_HI: f32 = 0.8;

    #[test]
    fn band_gate_is_strictly_inside_open_unit_interval_where_not_saturated() {
        // The house sigmoid (`fast_sigmoid`) saturates at BOTH ends of [0,1]
        // in f32: ±40 early-exit → exactly 0 far outside the band, and
        // plain rounding → exactly 1.0f32 once an argument exceeds ~17
        // (1 − σ(x) < 2⁻²⁵ is below f32 resolution). So the Rust gate is a
        // hard-off/hard-on member of [0,1]; the STRICT (0,1) contract is
        // the ideal one Lean pins over ℝ (the same Rust-vs-ℝ split the
        // bridge module documents for its ranking theorem).
        for &kappa in &[0.1f32, 1.0, 4.0] {
            let mut w = -2.0f32;
            while w <= 3.0 {
                let g = learnable_band_gate(w, W_LO, W_HI, kappa);
                assert!(
                    g > 0.0 && g < 1.0,
                    "gate {g} outside (0,1) at w={w} kappa={kappa}"
                );
                w += 0.01;
            }
        }
        // At large kappa the ±40 saturation kicks in far outside the band:
        // the gate is exactly 0 there (hard-off).
        for w in [-2.0f32, -1.5, 2.5, 3.0] {
            assert_eq!(learnable_band_gate(w, W_LO, W_HI, 64.0), 0.0);
        }
        // In and near the band at large kappa: strictly inside (0,1) EXCEPT
        // the deep band interior, where both sigmoid args (~19.2 at the
        // center) round σ to exactly 1.0f32 — the hard-on saturation, not a
        // bug. Walls sit at 0.5 (= σ(0)·1.0); just outside the band the
        // inner sigmoid is still a representable ~2.7e-6 > 0.
        for w in [0.0f32, 0.2, 0.8, 1.0] {
            let g = learnable_band_gate(w, W_LO, W_HI, 64.0);
            assert!(g > 0.0 && g < 1.0, "in/near-band gate {g} at w={w}");
        }
        assert_eq!(learnable_band_gate(0.5, W_LO, W_HI, 64.0), 1.0);
        assert_eq!(learnable_band_gate(0.2, W_LO, W_HI, 64.0), 0.5);
        assert_eq!(learnable_band_gate(0.8, W_LO, W_HI, 64.0), 0.5);
    }

    #[test]
    fn band_gate_peaks_at_band_center() {
        let center = (W_LO + W_HI) / 2.0;
        for &kappa in &[1.0f32, 4.0, 16.0] {
            let peak = learnable_band_gate(center, W_LO, W_HI, kappa);
            let mut w = -1.0f32;
            while w <= 2.0 {
                let g = learnable_band_gate(w, W_LO, W_HI, kappa);
                assert!(
                    g <= peak + 1e-6,
                    "g({w})={g} exceeds peak {peak} (kappa={kappa})"
                );
                w += 0.01;
            }
        }
    }

    #[test]
    fn band_gate_is_monotone_up_then_down() {
        let kappa = 8.0;
        let center = (W_LO + W_HI) / 2.0;
        let mut prev = learnable_band_gate(-2.0, W_LO, W_HI, kappa);
        let mut w = -1.99f32;
        while w < center {
            let g = learnable_band_gate(w, W_LO, W_HI, kappa);
            assert!(g >= prev - 1e-6, "not monotone ↑ below center at w={w}");
            prev = g;
            w += 0.01;
        }
        let mut prev = learnable_band_gate(center, W_LO, W_HI, kappa);
        let mut w = center + 0.01;
        while w <= 2.5 {
            let g = learnable_band_gate(w, W_LO, W_HI, kappa);
            assert!(g <= prev + 1e-6, "not monotone ↓ above center at w={w}");
            prev = g;
            w += 0.01;
        }
    }

    #[test]
    fn band_gate_saturates_with_kappa() {
        let center = (W_LO + W_HI) / 2.0;
        // Inside the band the gate rises toward 1; far outside it falls toward 0.
        let mid_small = learnable_band_gate(center, W_LO, W_HI, 1.0);
        let mid_big = learnable_band_gate(center, W_LO, W_HI, 128.0);
        assert!(mid_big > mid_small && mid_big > 0.9, "mid_big={mid_big}");
        let out_small = learnable_band_gate(-2.0, W_LO, W_HI, 1.0);
        let out_big = learnable_band_gate(-2.0, W_LO, W_HI, 128.0);
        assert!(out_big < out_small && out_big < 0.01, "out_big={out_big}");
        // kappa = 0 flattens to sigma(0)^2 = 0.25 everywhere.
        assert!((learnable_band_gate(0.5, W_LO, W_HI, 0.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn triage_partition_is_exclusive_exhaustive_and_boundary_pinned() {
        // Boundary pins: r̂ == τ_r → Frontier (≥); R⁻ == τ_R → Mastered (≥).
        assert_eq!(triage(0.5, 0.9, 0.5, 0.5), Regime::Frontier);
        assert_eq!(triage(0.0, 0.5, 0.5, 0.5), Regime::Mastered);
        assert_eq!(triage(0.0, 0.4999, 0.5, 0.5), Regime::Intractable);
        // Frontier wins regardless of R⁻ (high regret dominates the split).
        assert_eq!(triage(0.9, 0.0, 0.5, 0.5), Regime::Frontier);
        // Exhaustive sweep: every cell hit, each input exactly one regime.
        let mut counts = [0usize; 3];
        let mut r = -0.5f32;
        while r <= 1.5 {
            let mut ret = -0.5f32;
            while ret <= 1.5 {
                let reg = triage(r, ret, 0.5, 0.5);
                counts[reg as usize - 1] += 1;
                ret += 0.05;
            }
            r += 0.05;
        }
        assert!(
            counts.iter().all(|&c| c > 0),
            "some cell unreachable: {counts:?}"
        );
    }

    #[test]
    fn wilson_ci_matches_the_reference_formula() {
        // p̂=0.5, n=10, z=1.96 → the textbook interval ≈ (0.2366, 0.7634).
        let (lo, hi) = wilson_score_ci(0.5, 10, 1.959_963_984_540_054);
        assert!((lo - 0.236_64).abs() < 1e-4, "lo={lo}");
        assert!((hi - 0.763_36).abs() < 1e-4, "hi={hi}");
        // Degenerate: n=0 → (0,1); extreme p̂ stays clamped.
        assert_eq!(wilson_score_ci(0.5, 0, 1.96), (0.0, 1.0));
        let (lo, hi) = wilson_score_ci(1.0, 5, 1.96);
        assert!(lo > 0.0 && hi <= 1.0);
    }
}

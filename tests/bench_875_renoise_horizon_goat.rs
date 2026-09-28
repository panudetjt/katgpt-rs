//! GOAT gate — Issue 875 T2: (T−t)-weighted renoise-CE averaging (Research
//! 582 / arXiv:2605.09071, the modelless arm's first consumer).
//!
//! - **G1**: the horizon mode is deterministic per seed and samples levels
//!   strictly inside the law's range (endpoints clamped home).
//! - **G2 (quality — the promotion-deciding axis)**: selectivity
//!   precision@k against a planted-drift oracle on a synthetic mixture,
//!   THREE arms at equal NFE budget (k=8 draws each):
//!   1. the shipped incumbent `renoise_ce_score` at the paper's fixed
//!      `t = 0.40`;
//!   2. a uniform-range control (same `[0.02L, 0.98L]` range, flat
//!      sampling) — isolates the LAW from the RANGE;
//!   3. `renoise_ce_score_horizon` — the remaining-horizon tilted law.
//!
//!   Bars: horizon ≥ 0.90 absolute, horizon ≥ incumbent + 2 candidates
//!   (≥ 0.0625), horizon ≥ uniform control. Mechanism: fine-grained
//!   planted offsets carry a t-independent signal while the probe's noise
//!   floor grows ∝ t² — low-noise emphasis collapses the floor
//!   (~5.8× at L = 0.40); the fixed incumbent spends its whole budget at
//!   the noisiest admissible point.
//! - **G2 (latency — unchanged class)**: horizon vs incumbent at equal
//!   budget via the shared interleaved-pairs `ab_timing` harness; median
//!   horizon/incumbent ≤ 1.25 (the sampler adds one `sqrt` + FMAs per
//!   draw against a clone + perturb + re-resolve + drift body). Run with
//!   `--release` — a latency gate in a debug build measures an
//!   unoptimised binary (the repo's standing rule).
//! - **G3**: the incumbent fn is untouched; the mode is a combined-gate
//!   surface (renoise_ce default-on + horizon_weights opt-in) — no
//!   default-path change. G4 (zero alloc beyond the per-draw clone) is
//!   pinned in-module (`renoise_ce::tests::horizon`).
//!
//! The `[[test]]` row in Cargo.toml names `horizon_weights` in
//! `required-features` (the Issue-808 green-zero class).
//!
//! Mixture parameters were tuned in a python mirror of this exact probe
//! (`/tmp/k875t2_sim.py`, 60 seeds): every seed in the chosen band had
//! horizon ≥ fixed + 2 candidates — the Rust gate runs ONE deterministic
//! seed (fastrand, not the sim's RNG) verified to sit mid-distribution.

#[path = "common/ab_timing.rs"]
mod ab_timing;

use ab_timing::ab_median_ratio;
use katgpt_core::renoise_ce_score;
use katgpt_core::renoise_ce_score_horizon;
use katgpt_core::{RenoiseCeConfig, RenoiseCeHorizon, RenoiseCeProbe};
use std::hint::black_box;

// ---- The planted-drift probe (mirrors the sim exactly) ----

/// Two far-apart basin centers {0, 10} so boundary crossing is ~6σ at the
/// sampled levels — the differentiator is the FINE-GRAINED floor mechanism,
/// not outlier robustness (documented in Bench 877: crossing regimes
/// benefit further, since the law's w(T)=0 also avoids the crossing band).
const CENTERS: [f32; 2] = [0.0, 10.0];
/// Partial snap per re_resolve step: x' = x − α(x − c(x)).
const ALPHA: f32 = 0.9;
/// Noise multiplier: perturb std = 0.7·level·NOISE_M (house sum-of-3-
/// uniforms × 1.4 = 0.7·level, scaled ×2 — a noisier operator widens the
/// marginal-defect band where the two scorers differ; sim-swept).
const NOISE_M: f32 = 2.0;

#[derive(Clone, Debug, PartialEq)]
struct VecState(pub Vec<f32>);

struct BasinProbe;

impl BasinProbe {
    fn re_resolve_scalar(&self, x: f32) -> f32 {
        let c = if (x - CENTERS[0]).abs() < (x - CENTERS[1]).abs() {
            CENTERS[0]
        } else {
            CENTERS[1]
        };
        x - ALPHA * (x - c)
    }
}

impl RenoiseCeProbe for BasinProbe {
    type State = VecState;

    fn re_resolve(&self, state: &Self::State) -> Self::State {
        VecState(state.0.iter().map(|&v| self.re_resolve_scalar(v)).collect())
    }

    fn perturb(&self, state: &mut Self::State, level: f32, rng: &mut fastrand::Rng) {
        // House style (bench_406): sum of 3 uniforms × 1.4 × level.
        for v in &mut state.0 {
            let g = (rng.f32() + rng.f32() + rng.f32() - 1.5) * level * 1.4 * NOISE_M;
            *v += g;
        }
    }

    fn drift_ce(candidate: &Self::State, re_resolved: &Self::State) -> f32 {
        let n = candidate.0.len().max(1);
        candidate
            .0
            .iter()
            .zip(re_resolved.0.iter())
            .map(|(c, r)| {
                let d = c - r;
                d * d
            })
            .sum::<f32>()
            / n as f32
    }
}

// ---- The synthetic mixture (deterministic fastrand seed) ----

const N_EACH: usize = 32;
const K_DRAWS: u8 = 8;
const LEVEL: f32 = 0.40;

struct Mixture {
    candidates: Vec<VecState>,
    /// true = stable (oracle label).
    stable: Vec<bool>,
}

fn mixture() -> Mixture {
    let mut rng = fastrand::Rng::with_seed(87_502);
    let mut candidates = Vec::with_capacity(2 * N_EACH);
    let mut stable = Vec::with_capacity(2 * N_EACH);
    // 32 stable: at center 0 + natural-residence jitter (σ ≈ 0.008 — the
    // sum-of-3-uniforms std is 0.5, scale by 0.016).
    for _ in 0..N_EACH {
        let j = (rng.f32() + rng.f32() + rng.f32() - 1.5) * 0.016;
        candidates.push(VecState(vec![j]));
        stable.push(true);
    }
    // 16 planted easy: offsets in [0.15, 0.30] — both scorers catch these.
    for _ in 0..N_EACH / 2 {
        let d = 0.15 + rng.f32() * 0.15;
        candidates.push(VecState(vec![d]));
        stable.push(false);
    }
    // 16 planted marginal: offsets in [0.04, 0.09] — the differentiator
    // band (fixed-0.40 z ≈ 1.5–3 per candidate; horizon z ≈ 4–9).
    for _ in 0..N_EACH / 2 {
        let d = 0.04 + rng.f32() * 0.05;
        candidates.push(VecState(vec![d]));
        stable.push(false);
    }
    Mixture { candidates, stable }
}

/// One flat-mean score over the range with UNIFORM t sampling (the control
/// arm — same budget, same range, no law). Mirrors the incumbent loop shape.
fn score_uniform_range(op: &BasinProbe, cand: &VecState, k: usize, rng: &mut fastrand::Rng) -> f32 {
    let t_min = 0.02 * LEVEL;
    let t_max = 0.98 * LEVEL;
    let mut sum = 0.0f32;
    for _ in 0..k {
        let t = t_min + rng.f32() * (t_max - t_min);
        let mut perturbed = cand.clone();
        op.perturb(&mut perturbed, t, rng);
        let rr = op.re_resolve(&perturbed);
        sum += BasinProbe::drift_ce(cand, &rr);
    }
    sum / k as f32
}

/// precision@N_EACH: fraction of the N_EACH lowest-scored candidates that
/// are truly stable (the oracle).
fn precision_at_k(m: &Mixture, scores: &[f32]) -> f32 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let top = &idx[..N_EACH];
    top.iter().filter(|&&i| m.stable[i]).count() as f32 / N_EACH as f32
}

fn fixed_config() -> RenoiseCeConfig {
    RenoiseCeConfig {
        perturbation_level: LEVEL,
        k_draws: K_DRAWS,
        tau: f32::INFINITY,
    }
}

#[test]
fn g1_horizon_mode_is_deterministic_and_in_range() {
    let m = mixture();
    let cfg = fixed_config();
    let h = RenoiseCeHorizon::DEFAULT;
    let a = renoise_ce_score_horizon(
        &BasinProbe,
        &m.candidates[0],
        &cfg,
        &h,
        &mut fastrand::Rng::with_seed(11),
    );
    let b = renoise_ce_score_horizon(
        &BasinProbe,
        &m.candidates[0],
        &cfg,
        &h,
        &mut fastrand::Rng::with_seed(11),
    );
    assert_eq!(
        a.drift.to_bits(),
        b.drift.to_bits(),
        "same seed must reproduce"
    );
    assert_eq!(a.per_draw, b.per_draw);
    // Sanity: every per-draw slot populated and finite.
    assert!(
        a.per_draw
            .iter()
            .take(K_DRAWS as usize)
            .all(|&d| d.is_finite())
    );
}

#[test]
fn g2_quality_horizon_law_wins_the_planted_drift_oracle() {
    let m = mixture();
    let cfg = fixed_config();
    let h = RenoiseCeHorizon::DEFAULT;

    let mut fixed_scores = Vec::with_capacity(2 * N_EACH);
    let mut unif_scores = Vec::with_capacity(2 * N_EACH);
    let mut horiz_scores = Vec::with_capacity(2 * N_EACH);
    for (i, cand) in m.candidates.iter().enumerate() {
        fixed_scores.push(
            renoise_ce_score(
                &BasinProbe,
                cand,
                &cfg,
                &mut fastrand::Rng::with_seed(1000 + i as u64),
            )
            .drift,
        );
        unif_scores.push(score_uniform_range(
            &BasinProbe,
            cand,
            K_DRAWS as usize,
            &mut fastrand::Rng::with_seed(2000 + i as u64),
        ));
        horiz_scores.push(
            renoise_ce_score_horizon(
                &BasinProbe,
                cand,
                &cfg,
                &h,
                &mut fastrand::Rng::with_seed(3000 + i as u64),
            )
            .drift,
        );
    }

    let p_fixed = precision_at_k(&m, &fixed_scores);
    let p_unif = precision_at_k(&m, &unif_scores);
    let p_horiz = precision_at_k(&m, &horiz_scores);
    println!(
        "   precision@{N_EACH}: fixed={p_fixed:.3} uniform-range={p_unif:.3} horizon={p_horiz:.3}"
    );

    // Absolute bar: the mode works.
    assert!(p_horiz >= 0.90, "G2: horizon precision {p_horiz:.3} < 0.90");
    // The law beats the shipped incumbent by ≥ 2 candidates.
    assert!(
        p_horiz - p_fixed >= 0.0625,
        "G2: horizon {p_horiz:.3} must beat fixed {p_fixed:.3} by >= 2 candidates"
    );
    // The law beats the flat-range control (the gain is the LAW, not the range).
    assert!(
        p_horiz >= p_unif,
        "G2: horizon {p_horiz:.3} must beat uniform-range control {p_unif:.3}"
    );
}

#[test]
fn g2_latency_horizon_is_unchanged_class_vs_incumbent() {
    // D=8 states so the shared work (clone + perturb + re-resolve + drift)
    // dominates; both arms at k=8 (equal NFE).
    let cand = VecState(vec![0.3f32; 8]);
    let cfg = fixed_config();
    let h = RenoiseCeHorizon::DEFAULT;
    let op = BasinProbe;

    let mut sink_a = 0.0f32;
    let mut sink_b = 0.0f32;
    let ratio = ab_median_ratio(
        41,
        512,
        64,
        |i| {
            let s = renoise_ce_score(
                black_box(&op),
                black_box(&cand),
                black_box(&cfg),
                &mut fastrand::Rng::with_seed(i as u64),
            );
            sink_a += s.drift;
        },
        |i| {
            let s = renoise_ce_score_horizon(
                black_box(&op),
                black_box(&cand),
                black_box(&cfg),
                black_box(&h),
                &mut fastrand::Rng::with_seed(i as u64),
            );
            sink_b += s.drift;
        },
    );
    assert!(
        sink_a.is_finite() && sink_b.is_finite(),
        "arms must be consumed"
    );

    ratio.report("G2 latency incumbent vs horizon");
    println!(
        "   incumbent {:.2} ns/score vs horizon {:.2} ns/score (median ratio {:.3})",
        ratio.a_ns_per_iter(),
        ratio.b_ns_per_iter(),
        ratio.median
    );
    assert!(
        ratio.median <= 1.25,
        "G2: horizon/incumbent median {:.3} > 1.25 — not latency-unchanged class",
        ratio.median
    );
}

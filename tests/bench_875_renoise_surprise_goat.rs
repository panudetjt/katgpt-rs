//! GOAT gate — Issue 875 T4: target-anchored renoise-CE surprise (Research
//! 582 / arXiv:2605.09071, PFD's resolve-against-the-teacher variant) + the
//! consolidation-ordering consumer sketch.
//!
//! - **G1**: the surprise mode is deterministic per seed, finite, and the
//!   shell world holds (every candidate at the same distance from the prior
//!   anchor — the plain control's blindness is by construction).
//! - **G2 (quality — the promotion-deciding axis)**: flow-relative-novelty
//!   precision@32 against a planted-drift oracle on the SHELL world, THREE
//!   arms at equal NFE budget (k=8 draws each):
//!   1. the shipped incumbent `renoise_ce_score` (self-consistency);
//!   2. a plain distance control (rank the raw observation by distance to
//!      the prior anchor — no operator, no renoise);
//!   3. `renoise_ce_surprise` — the target-anchored mode.
//!
//!   The shell world isolates the mechanism the mode exists for: every
//!   candidate sits at the SAME whitened radius from the prior center
//!   (plain distance is blind by construction), stable states point away
//!   from a foreign attractor while planted states point into its capture
//!   cap (self-consistency ANTI-ranks them — a state resting in ANY basin,
//!   foreign included, is the most self-stable), and only the flow through
//!   the operator, scored against the prior anchor, separates.
//!
//!   Bars: surprise ≥ 0.90 absolute, surprise ≥ incumbent + 2 candidates,
//!   surprise ≥ plain + 2 candidates. Sim-swept 200 seeds (python mirror,
//!   `k875t4_sim.py` v7): surprise 1.000±0.002 (min 0.969), incumbent
//!   0.000, plain 0.428 — margins held 200/200; the Rust gate runs ONE
//!   deterministic fastrand seed. Regime boundary (recorded in Bench 879):
//!   in worlds where pointwise distance already sees the displacement,
//!   plain distance is a strong ranker and the mode adds little — the win
//!   is flow-relative novelty, not generic drift detection.
//! - **G2 (latency — unchanged class)**: surprise vs incumbent at equal
//!   budget via the shared interleaved-pairs `ab_timing` harness; median
//!   surprise/incumbent ≤ 1.25 (identical loop shape — the only delta is
//!   which reference the drift is measured against). Run with `--release`.
//! - **G3 (consumer sketch)**: consolidation surprise ordering — rank the
//!   shards by surprise descending; the Raven/δ-Mem sleep-cycle admission
//!   list is that ranking's head. Ordering-only: no behavior change rides
//!   the score (the sketch asserts the ordering, it does not mutate a
//!   consolidation pipeline).
//! - **G4** (zero alloc on the score path with fixed-array State) is pinned
//!   in-module (`renoise_ce::tests::surprise`).
//!
//! The `[[test]]` row in Cargo.toml names `renoise_ce_surprise` in
//! `required-features` (the Issue-808 green-zero class).

#[path = "common/ab_timing.rs"]
mod ab_timing;

use ab_timing::ab_median_ratio;
use katgpt_core::renoise_ce_score;
use katgpt_core::renoise_ce_surprise;
use katgpt_core::{RenoiseCeConfig, RenoiseCeProbe};
use std::hint::black_box;

// ---- The shell world (mirrors the python sim exactly; states are whitened
// units — the sim's per-coordinate sigma rescaled every arm identically and
// canceled, so the Rust world states it in whitened coordinates directly) ----

const D: usize = 8;
const N_EACH: usize = 32;
const K_DRAWS: u8 = 8;
const LEVEL: f32 = 0.40;
/// Foreign-center whitened distance.
const L: f32 = 2.6;
/// Shell radius as a fraction of L.
const R_FRAC: f32 = 0.9;
/// Stable directions: cos(angle to foreign axis) ≤ this.
const STABLE_MAX_COS: f32 = 0.30;
/// Planted directions: cos(angle to foreign axis) ≥ this.
const PLANTED_MIN_COS: f32 = 0.75;
/// Partial contraction per re_resolve (T2's BasinProbe shape).
const ALPHA: f32 = 0.9;
/// Noise multiplier on the house sum-of-3-uniforms perturb (T2's value).
const NOISE_M: f32 = 2.0;

#[derive(Clone, Debug, PartialEq)]
struct VecState(pub Vec<f32>);

/// Two-attractor operator: attractors {origin (the prior center), L·û_c
/// (foreign)}; re_resolve snaps to the nearest then contracts α = 0.9. The
/// operator's basins are its own dynamics — knowing them is not oracle
/// leakage (the oracle is WHICH shards are planted; the operator never sees
/// that).
struct ShellProbe {
    c: Vec<f32>,
}

impl ShellProbe {
    fn wd2(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
            / D as f32
    }
}

impl RenoiseCeProbe for ShellProbe {
    type State = VecState;

    fn re_resolve(&self, state: &Self::State) -> Self::State {
        let origin = [0.0f32; D];
        let attractor: &[f32] = if self.wd2(&state.0, &origin) <= self.wd2(&state.0, &self.c) {
            &origin
        } else {
            &self.c
        };
        VecState(
            state
                .0
                .iter()
                .zip(attractor.iter())
                .map(|(x, a)| x - ALPHA * (x - a))
                .collect(),
        )
    }

    fn perturb(&self, state: &mut Self::State, level: f32, rng: &mut fastrand::Rng) {
        for v in state.0.iter_mut() {
            let g = (rng.f32() + rng.f32() + rng.f32() - 1.5) * level * 1.4 * NOISE_M;
            *v += g;
        }
    }

    fn drift_ce(candidate: &Self::State, re_resolved: &Self::State) -> f32 {
        candidate
            .0
            .iter()
            .zip(re_resolved.0.iter())
            .map(|(c, r)| {
                let d = c - r;
                d * d
            })
            .sum::<f32>()
            / D as f32
    }
}

// ---- The synthetic shell mixture (deterministic fastrand seed) ----

struct Mixture {
    candidates: Vec<VecState>,
    /// true = stable (oracle label).
    stable: Vec<bool>,
    probe: ShellProbe,
}

fn house_unit(rng: &mut fastrand::Rng) -> f32 {
    rng.f32() + rng.f32() + rng.f32() - 1.5
}

fn mixture() -> Mixture {
    let mut rng = fastrand::Rng::with_seed(87_504);
    // Foreign axis: normalize a spherically-symmetric house-noise vector
    // (any spherically symmetric distribution gives uniform directions).
    let uc = {
        let mut v = [0.0f32; D];
        for x in v.iter_mut() {
            *x = house_unit(&mut rng);
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut u = v.iter().map(|x| x / n).collect::<Vec<f32>>();
        u.shrink_to_fit();
        u
    };
    let c: Vec<f32> = uc.iter().map(|x| L * x).collect();
    let probe = ShellProbe { c };

    let r = R_FRAC * L;
    let mut candidates = Vec::with_capacity(2 * N_EACH);
    let mut stable = Vec::with_capacity(2 * N_EACH);

    // A unit direction with cos(angle to û_c) in [lo, hi], rejection-sampled.
    let sample_dir = |rng: &mut fastrand::Rng, lo: f32, hi: f32| -> Vec<f32> {
        loop {
            let mut v = [0.0f32; D];
            for x in v.iter_mut() {
                *x = house_unit(rng);
            }
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n < 1e-6 {
                continue;
            }
            let cos = v.iter().zip(uc.iter()).map(|(a, b)| a * b).sum::<f32>() / n;
            if cos >= lo && cos <= hi {
                return v.iter().map(|x| x / n).collect();
            }
        }
    };

    // 32 stable: directions away from the foreign cap, scaled to the shell.
    for _ in 0..N_EACH {
        let u = sample_dir(&mut rng, -1.0, STABLE_MAX_COS);
        candidates.push(VecState(u.iter().map(|x| r * x).collect()));
        stable.push(true);
    }
    // 32 planted: directions inside the foreign capture cap, same shell.
    for _ in 0..N_EACH {
        let u = sample_dir(&mut rng, PLANTED_MIN_COS, 1.0);
        candidates.push(VecState(u.iter().map(|x| r * x).collect()));
        stable.push(false);
    }
    Mixture {
        candidates,
        stable,
        probe,
    }
}

/// precision@N_EACH for NOVELTY: fraction of the N_EACH highest-scored
/// candidates that are truly planted (drift injected at known shards).
fn precision_at_k(m: &Mixture, scores: &[f32]) -> f32 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let top = &idx[..N_EACH];
    top.iter().filter(|&&i| !m.stable[i]).count() as f32 / N_EACH as f32
}

fn fixed_config() -> RenoiseCeConfig {
    RenoiseCeConfig {
        perturbation_level: LEVEL,
        k_draws: K_DRAWS,
        tau: f32::INFINITY,
    }
}

#[test]
fn g1_surprise_mode_is_deterministic_and_shell_holds() {
    let m = mixture();
    let cfg = fixed_config();
    let target = VecState(vec![0.0; D]);
    let a = renoise_ce_surprise(
        &m.probe,
        &target,
        &m.candidates[0],
        &cfg,
        &mut fastrand::Rng::with_seed(11),
    );
    let b = renoise_ce_surprise(
        &m.probe,
        &target,
        &m.candidates[0],
        &cfg,
        &mut fastrand::Rng::with_seed(11),
    );
    assert_eq!(
        a.drift.to_bits(),
        b.drift.to_bits(),
        "same seed must reproduce"
    );
    assert_eq!(a.per_draw, b.per_draw);
    assert!(
        a.per_draw
            .iter()
            .take(K_DRAWS as usize)
            .all(|d| d.is_finite())
    );
    // World sanity: every candidate sits on the shell (identical whitened
    // radius) — the plain control's blindness is by construction.
    let shell = (R_FRAC * L) * (R_FRAC * L) / D as f32;
    for cand in &m.candidates {
        let d = m.probe.wd2(&cand.0, &target.0);
        assert!((d - shell).abs() < 0.01, "shell violated: {d} vs {shell}");
    }
}

#[test]
fn g2_quality_target_anchored_sees_flow_relative_novelty() {
    let m = mixture();
    let cfg = fixed_config();
    let target = VecState(vec![0.0; D]);

    let mut sur_scores = Vec::with_capacity(2 * N_EACH);
    let mut inc_scores = Vec::with_capacity(2 * N_EACH);
    let mut plain_scores = Vec::with_capacity(2 * N_EACH);
    for (i, cand) in m.candidates.iter().enumerate() {
        sur_scores.push(
            renoise_ce_surprise(
                &m.probe,
                &target,
                cand,
                &cfg,
                &mut fastrand::Rng::with_seed(1000 + i as u64),
            )
            .drift,
        );
        inc_scores.push(
            renoise_ce_score(
                &m.probe,
                cand,
                &cfg,
                &mut fastrand::Rng::with_seed(2000 + i as u64),
            )
            .drift,
        );
        plain_scores.push(m.probe.wd2(&target.0, &cand.0));
    }

    let p_sur = precision_at_k(&m, &sur_scores);
    let p_inc = precision_at_k(&m, &inc_scores);
    let p_plain = precision_at_k(&m, &plain_scores);
    println!(
        "   precision@{N_EACH}: surprise={p_sur:.3} incumbent={p_inc:.3} plain-distance={p_plain:.3}"
    );

    // Absolute bar: the mode works.
    assert!(p_sur >= 0.90, "G2: surprise precision {p_sur:.3} < 0.90");
    // Beats the shipped incumbent by ≥ 2 candidates (self-consistency
    // ANTI-ranks foreign-basin novelty — it certifies comfort in ANY basin).
    assert!(
        p_sur - p_inc >= 0.0625,
        "G2: surprise {p_sur:.3} must beat incumbent {p_inc:.3} by >= 2 candidates"
    );
    // Beats the plain-distance control (the gain is the flow through the
    // operator, not the distance).
    assert!(
        p_sur - p_plain >= 0.0625,
        "G2: surprise {p_sur:.3} must beat plain-distance control {p_plain:.3} by >= 2 candidates"
    );
}

#[test]
fn g2_latency_surprise_is_unchanged_class_vs_incumbent() {
    // D=8 states, both arms k=8 (equal NFE). Identical loop shape; the only
    // delta is the drift reference.
    let m = mixture();
    let cfg = fixed_config();
    let target = VecState(vec![0.0; D]);
    let cand = &m.candidates[0];

    let mut sink_a = 0.0f32;
    let mut sink_b = 0.0f32;
    let ratio = ab_median_ratio(
        41,
        512,
        64,
        |i| {
            let s = renoise_ce_score(
                black_box(&m.probe),
                black_box(cand),
                black_box(&cfg),
                &mut fastrand::Rng::with_seed(i as u64),
            );
            sink_a += s.drift;
        },
        |i| {
            let s = renoise_ce_surprise(
                black_box(&m.probe),
                black_box(&target),
                black_box(cand),
                black_box(&cfg),
                &mut fastrand::Rng::with_seed(i as u64),
            );
            sink_b += s.drift;
        },
    );
    assert!(
        sink_a.is_finite() && sink_b.is_finite(),
        "arms must be consumed"
    );

    ratio.report("G2 latency incumbent vs surprise");
    println!(
        "   incumbent {:.2} ns/score vs surprise {:.2} ns/score (median ratio {:.3})",
        ratio.a_ns_per_iter(),
        ratio.b_ns_per_iter(),
        ratio.median
    );
    assert!(
        ratio.median <= 1.25,
        "G2: surprise/incumbent median {:.3} > 1.25 — not latency-unchanged class",
        ratio.median
    );
}

#[test]
fn g3_consumer_sketch_consolidation_ordering_is_surprise_ranked() {
    // The consolidation-ordering consumer sketch: rank shards by surprise
    // descending; the Raven/δ-Mem sleep-cycle admission list is the head of
    // that ranking. ORDERING-ONLY — no consolidation behavior changes here;
    // this pins the consumer convention (score → sort → take head) against
    // the oracle: the first admissions must be the planted (novel) shards.
    let m = mixture();
    let cfg = RenoiseCeConfig {
        perturbation_level: LEVEL,
        k_draws: 4, // ordering pass rides existing budget at half cost
        tau: f32::INFINITY,
    };
    let target = VecState(vec![0.0; D]);

    let mut scored: Vec<(usize, f32)> = m
        .candidates
        .iter()
        .enumerate()
        .map(|(i, cand)| {
            (
                i,
                renoise_ce_surprise(
                    &m.probe,
                    &target,
                    cand,
                    &cfg,
                    &mut fastrand::Rng::with_seed(7000 + i as u64),
                )
                .drift,
            )
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // The admission list (first N_EACH) against the oracle.
    let admitted_planted = scored[..N_EACH]
        .iter()
        .filter(|(i, _)| !m.stable[*i])
        .count();
    assert!(
        admitted_planted as f32 / N_EACH as f32 >= 0.90,
        "consolidation-ordering sketch: {admitted_planted}/{N_EACH} first admissions planted"
    );
    // And the tail is the stable remainder — an ordering, not a threshold.
    let tail_stable = scored[N_EACH..]
        .iter()
        .filter(|(i, _)| m.stable[*i])
        .count();
    assert!(
        tail_stable as f32 / N_EACH as f32 >= 0.90,
        "consolidation-ordering sketch: {tail_stable}/{N_EACH} tail deferrals stable"
    );
}

//! Issue 663 PoC tests — SwitchCostTable G1/warm-up/consumer-shape coverage
//! and G2 lookup latency (release-only), plus the factorized-vs-full-table
//! A/B at the paper's own factorization-fidelity bar (82–86% partition
//! overlap, arXiv:2608.05139 §B.4 → Spearman ≥ 0.75 with identical argmax).
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features switch_cost --test switch_cost_663_poc
//! # G2 timing gate (release only):
//! cargo test --release -p katgpt-core --features switch_cost \
//!     --test switch_cost_663_poc g2 -- --nocapture
//! ```

#![cfg(feature = "switch_cost")]

use katgpt_core::switch_cost::{DEFAULT_ALPHA, FactorizedSwitchCost, SwitchCostTable, cdf_rank};
use std::hint::black_box;

/// Spearman rank correlation over paired samples (no external crate).
fn spearman(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let rank = |v: &[f32]| -> Vec<f32> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
        let mut r = vec![0.0f32; v.len()];
        for (pos, &i) in idx.iter().enumerate() {
            r[i] = pos as f32;
        }
        r
    };
    let (rx, ry) = (rank(x), rank(y));
    let n = x.len() as f32;
    let mx = rx.iter().sum::<f32>() / n;
    let my = ry.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut dx = 0.0f32;
    let mut dy = 0.0f32;
    for i in 0..x.len() {
        let a = rx[i] - mx;
        let b = ry[i] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    num / (dx * dy).sqrt()
}

/// Multiplicative ground truth (paper Eq. 7's assumption): pair success
/// `p(a,b) = clamp(s_b / (L(a, fam_b) · G(fam_a, b)), 0, 1)` with leave costs
/// L and land costs G ≥ 1, modest solo spread so the family aggregation
/// recovers the structure.
fn ground_truth() -> ([f32; 6], [[f32; 2]; 6], [[f32; 6]; 2]) {
    let solo = [0.80, 0.65, 0.75, 0.70, 0.60, 0.72];
    // L[a][f]: leave cost of a → family f. Mode 1 (in fam 0) leaving into
    // fam 1 is the designed hardest leave.
    let leave = [
        [1.00, 1.30],
        [1.10, 2.20],
        [1.20, 1.10],
        [1.40, 1.05],
        [1.00, 1.60],
        [1.25, 1.00],
    ];
    // G[f][b]: land cost of family f → b. Landing on mode 4 is hard from
    // either family; landing on mode 0 is easy.
    let land = [
        [1.00, 1.15, 1.05, 1.30, 1.70, 1.10],
        [1.05, 1.20, 1.10, 1.35, 1.85, 1.15],
    ];
    (solo, leave, land)
}

fn pair_success(solo_b: f32, leave: f32, land: f32) -> f32 {
    (solo_b / (leave * land)).clamp(0.0, 1.0)
}

/// Build BOTH tables from the SAME simulated trajectories (deterministic
/// seed) and return (exact ske matrix, factorized ske matrix).
fn simulate(trials_per_pair: u32, solo_trials: u32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let (solo, leave, land) = ground_truth();
    let family_of = [0usize, 0, 0, 1, 1, 1];
    let mut exact = SwitchCostTable::<6>::new(DEFAULT_ALPHA);
    let mut fact = FactorizedSwitchCost::<6, 2>::new(family_of, DEFAULT_ALPHA);

    for (m, &s) in solo.iter().enumerate() {
        for _ in 0..solo_trials {
            let ok = rng.f32() < s;
            exact.record_solo(m, ok);
            fact.record_solo(m, ok);
        }
    }
    for a in 0..6 {
        for b in 0..6 {
            let p = pair_success(solo[b], leave[a][family_of[b]], land[family_of[a]][b]);
            for _ in 0..trials_per_pair {
                let ok = rng.f32() < p;
                exact.record_switch(a, b, ok);
                fact.record_switch(a, b, ok);
            }
        }
    }

    let flat = |f: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
        let mut v = Vec::with_capacity(36);
        for a in 0..6 {
            for b in 0..6 {
                v.push(f(a, b));
            }
        }
        v
    };
    (flat(&|a, b| exact.ske(a, b)), flat(&|a, b| fact.ske(a, b)))
}

/// G1-A/B: the factorized variant (O(N·F) counters) must reproduce the exact
/// table's pair-hardness ORDER at the paper's own factorization fidelity —
/// Spearman ≥ 0.75 across all 36 ordered pairs, and identical argmax (the
/// single hardest switch, what an F1 trigger keys on).
#[test]
fn factorized_matches_full_table_ranking() {
    let (exact, fact) = simulate(600, 400, 663);
    let rho = spearman(&exact, &fact);
    assert!(
        rho >= 0.75,
        "Spearman {rho:.3} < 0.75 — factorization lost the ordering"
    );
    let (i_e, v_e) = exact
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.total_cmp(y.1))
        .unwrap();
    let (i_f, v_f) = fact
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.total_cmp(y.1))
        .unwrap();
    assert_eq!(
        i_e, i_f,
        "argmax pair differs: exact idx {i_e} ({v_e:.2}) vs factorized idx {i_f} ({v_f:.2})"
    );
    // Sanity on the fixture itself: the designed hard structure must be
    // visible in the exact table (hardest leave is mode 1 → fam 1, i.e.
    // pairs (1, 3..5); hardest land is onto mode 4).
    assert!(*v_e > 1.5, "fixture too soft: max exact SkE {v_e:.2}");
}

/// G1: same telemetry (same seed) → bit-identical tables; different seeds →
/// still highly correlated (the measure is an estimator, not a constant).
#[test]
fn deterministic_and_stable_across_seeds() {
    let (e1, f1) = simulate(300, 200, 7);
    let (e2, f2) = simulate(300, 200, 7);
    for i in 0..e1.len() {
        assert_eq!(e1[i].to_bits(), e2[i].to_bits(), "exact drift at {i}");
        assert_eq!(f1[i].to_bits(), f2[i].to_bits(), "fact drift at {i}");
    }
    let (e3, _) = simulate(300, 200, 8);
    let rho = spearman(&e1, &e3);
    assert!(
        rho >= 0.75,
        "cross-seed Spearman {rho:.3} — estimator unstable"
    );
}

/// Warm-up shape (Research 484 §6.1): cold = exactly neutral; the armed gate
/// opens only past the trial floor; α keeps every partially-warmed value
/// finite.
#[test]
fn warmup_cold_to_armed_progression() {
    let mut t = SwitchCostTable::<3>::new(DEFAULT_ALPHA);
    assert_eq!(t.ske(0, 1), 1.0);
    assert!(t.ske_if_armed(0, 1, 10).is_none());
    for k in 0..10 {
        t.record_switch(0, 1, k < 3);
        assert!(t.ske(0, 1).is_finite());
    }
    assert!(t.ske_if_armed(0, 1, 10).is_some());
    // 3/10 pair success with strong solos must read as a hard switch.
    for k in 0..10 {
        t.record_solo(0, k < 9);
        t.record_solo(1, k < 9);
    }
    assert!(
        t.ske(0, 1) > 2.0,
        "3/10 pair after 9/10 solos: {}",
        t.ske(0, 1)
    );
}

/// Consumer shape: `cdf_rank` makes the entropy reward scale-free (Gap 6).
/// Scaling the corpus must not change the reward.
#[test]
fn cdf_rank_reward_is_scale_free() {
    let corpus = [1.1, 1.4, 1.9, 2.6, 3.3];
    let scaled: Vec<f32> = corpus.iter().map(|v| v * 10.0).collect();
    let rho_hat = 1.9f32;
    let r_ent = |pred: f32, sample: &[f32]| 1.0 - (cdf_rank(pred, sample) - 0.6).abs();
    let a = r_ent(rho_hat, &corpus);
    let b = r_ent(rho_hat * 10.0, &scaled);
    assert!((a - b).abs() < 1e-6, "{a} vs {b}");
}

/// G2: hot `ske` lookup is single-digit nanoseconds (release-only gate —
/// debug builds pay ~10× on the division).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn g2_lookup_latency_single_digit_ns() {
    let mut t = SwitchCostTable::<8>::new(DEFAULT_ALPHA);
    for a in 0..8 {
        for k in 0..50u32 {
            t.record_solo(a, k % 3 != 0);
        }
        for b in 0..8 {
            for k in 0..50u32 {
                t.record_switch(a, b, !(k + a as u32 + b as u32).is_multiple_of(4));
            }
        }
    }
    let snap = t.snapshot();
    let mut best = f32::MAX;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let mut acc = 0.0f32;
        for i in 0..1_000_000u32 {
            let a = (i & 7) as usize;
            let b = ((i >> 3) & 7) as usize;
            acc += snap.ske(a, b);
        }
        black_box(acc);
        let ns = t0.elapsed().as_nanos() as f32 / 1_000_000.0;
        best = best.min(ns);
    }
    println!("g2 ske lookup: {best:.2} ns/op (best of 3 × 1M)");
    assert!(best < 10.0, "ske lookup {best:.2} ns ≥ 10 ns");
}

/// G2 companion: sequence entropy over a 16-mode sequence stays well under
/// the F1 per-tick budget (release-only gate).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn g2_sequence_entropy_under_300ns() {
    let mut t = SwitchCostTable::<16>::new(DEFAULT_ALPHA);
    for a in 0..16 {
        for b in 0..16 {
            t.record_switch(a, b, (a + b) % 3 != 0);
        }
    }
    let snap = t.snapshot();
    let seq: [usize; 16] = core::array::from_fn(|i| (i * 7 + 3) & 15);
    let mut best = f32::MAX;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..100_000 {
            acc += snap.sequence_entropy(&seq);
        }
        black_box(acc);
        let ns = t0.elapsed().as_nanos() as f32 / 100_000.0;
        best = best.min(ns);
    }
    println!("g2 sequence_entropy(16): {best:.2} ns/eval (best of 3 × 100k)");
    assert!(best < 300.0, "sequence_entropy {best:.2} ns ≥ 300 ns");
}

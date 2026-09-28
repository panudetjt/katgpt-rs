//! Phase 1 unit tests — pure-arithmetic properties of the estimator and the
//! Hoeffding machinery. The stochastic oracles (reveal-the-arm bandit,
//! hinted shortest path) and the coverage/variance GOAT gates live in
//! `tests/bench_576_hint_regret_goat.rs`.

use super::*;

const B: ReturnBounds = ReturnBounds { lo: 0.0, hi: 1.0 };

#[test]
fn hoeffding_k_matches_the_closed_form() {
    // K(eps, delta) = ceil((b-a)^2 / (2 eps^2) * ln(2/delta)), (b-a) = 2(hi-lo).
    // bounds [0,1] → range 2. eps=0.1, delta=0.05:
    //   K = ceil(4 / (2*0.01) * ln(40)) = ceil(200 * 3.6889) = ceil(737.78) = 738
    assert_eq!(hoeffding_k(0.1, 0.05, B), 738);
    // eps=0.25, delta=0.05: ceil(4 / (2*0.0625) * ln(40)) = ceil(32*3.6889) = ceil(118.04) = 119
    assert_eq!(hoeffding_k(0.25, 0.05, B), 119);
    // Tighter confidence: delta=0.01 → ln(200) = 5.298
    //   eps=0.25: ceil(32 * 5.298) = ceil(169.54) = 170
    assert_eq!(hoeffding_k(0.25, 0.01, B), 170);
    // Never below 1.
    assert_eq!(hoeffding_k(1e9, 0.05, B), 1);
}

#[test]
fn hoeffding_half_width_round_trips_the_schedule() {
    let eps = 0.15;
    let delta = 0.05;
    let k = hoeffding_k(eps, delta, B);
    // At n = K the guaranteed half-width is <= eps (the schedule's promise)...
    assert!(hoeffding_half_width(k, delta, B) <= eps + 1e-6);
    // ...and one pair SHORT of K it is not yet (ceil monotonicity).
    assert!(hoeffding_half_width(k - 1, delta, B) > eps);
    // 1/n^0.5 shape: quadrupling n halves the width.
    let h1 = hoeffding_half_width(64, delta, B);
    let h4 = hoeffding_half_width(256, delta, B);
    assert!((h1 / h4 - 2.0).abs() < 1e-5);
}

#[test]
fn estimator_matches_direct_arithmetic() {
    let pairs: Vec<(f32, f32)> = (0..64u32)
        .map(|i| {
            let a = (i as f32 * 0.013).sin() * 0.5 + 0.5;
            let b = (i as f32 * 0.007).cos() * 0.4 + 0.5;
            (a, b)
        })
        .collect();
    let mut est = HintRegretEstimator::new(B);
    for &(a, b) in &pairs {
        est.record_pair(a, b);
    }
    let e = est.estimate(0.05);
    let n = pairs.len() as f64;
    let mean_a: f64 = pairs.iter().map(|&(a, _)| a as f64).sum::<f64>() / n;
    let mean_b: f64 = pairs.iter().map(|&(_, b)| b as f64).sum::<f64>() / n;
    let mean_d: f64 = pairs.iter().map(|&(a, b)| (a - b) as f64).sum::<f64>() / n;
    // Outputs are f32 (cast from the f64 accumulators) — f32 tolerance.
    assert!((e.arm_means.0 as f64 - mean_a).abs() < 1e-6);
    assert!((e.arm_means.1 as f64 - mean_b).abs() < 1e-6);
    // Sign convention: r_hat = mean(hinted) - mean(unhinted).
    assert!((e.r_hat as f64 - (mean_a - mean_b)).abs() < 1e-6);
    assert!((e.r_hat as f64 - mean_d).abs() < 1e-6);
    // Unbiased sample variance of the differences.
    let var_d: f64 = pairs
        .iter()
        .map(|&(a, b)| {
            let d = (a - b) as f64 - mean_d;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    assert!((est.diff_sample_variance() as f64 - var_d).abs() < 1e-6);
    // CLT half-width = 1.96 * sqrt(var/n).
    let clt = 1.959_963_984_540_054 * (var_d / n).sqrt();
    assert!((e.empirical_half_width as f64 - clt).abs() < 1e-6);
}

#[test]
fn sign_convention_hint_gain_is_positive_when_hint_helps() {
    // The landed consumer's semantics (frontier_regime_of): a hint that
    // lifts the return 0.3 → 0.8 is r_hat = +0.5.
    let mut est = HintRegretEstimator::new(B);
    est.record_pair(0.8, 0.3);
    assert!((est.estimate(0.05).r_hat - 0.5).abs() < 1e-6);
}

#[test]
fn zero_pairs_is_uninformative_never_a_stop() {
    let est = HintRegretEstimator::new(B);
    let e = est.estimate(0.05);
    assert_eq!(e.n_pairs, 0);
    assert_eq!(e.r_hat, 0.0);
    assert_eq!(e.ci_half_width, f32::MAX);
    assert_eq!(e.eb_half_width, f32::MAX);
    assert_eq!(e.empirical_half_width, f32::MAX);
    assert!(!e.should_stop(1e9));
}

#[test]
fn single_pair_has_no_variance_and_max_eb() {
    let mut est = HintRegretEstimator::new(B);
    est.record_pair(0.6, 0.2);
    let e = est.estimate(0.05);
    assert_eq!(e.n_pairs, 1);
    assert_eq!(est.diff_sample_variance(), 0.0);
    assert_eq!(e.eb_half_width, f32::MAX);
    assert_eq!(e.empirical_half_width, f32::MAX);
    assert_eq!(e.ci_half_width, hoeffding_half_width(1, 0.05, B));
}

#[test]
fn should_stop_fires_exactly_at_the_schedule() {
    let eps = 0.2;
    let delta = 0.05;
    let k = hoeffding_k(eps, delta, B);
    let mut est = HintRegretEstimator::new(B);
    for i in 0..k {
        est.record_pair(0.5 + (i % 7) as f32 * 0.01, 0.4);
        let e = est.estimate(delta);
        assert_eq!(
            e.should_stop(eps),
            (i + 1) >= k,
            "stop must fire exactly at K"
        );
    }
}

#[test]
fn out_of_range_returns_are_clamped_before_accumulation() {
    let mut est = HintRegretEstimator::new(B);
    est.record_pair(5.0, -3.0); // wired wrong — clamps to (1.0, 0.0)
    let e = est.estimate(0.05);
    assert!((e.r_hat - 1.0).abs() < 1e-6);
    assert_eq!(e.arm_means, (1.0, 0.0));
}

#[test]
fn empirical_bernstein_sees_variance_collapse() {
    // Constant differences (variance 0): the EB sqrt term vanishes, leaving
    // only the additive 7(b-a)ln(2/δ)/(3(n-1)) term — which decays like 1/n,
    // strictly faster than the Hoeffding 1/sqrt(n) for large n.
    let var = 0.0;
    let eb = |n: u32| empirical_bernstein_half_width(n, var, 0.05, B);
    let hoeff = |n: u32| hoeffding_half_width(n, 0.05, B);
    // At n=2 EB is additive-dominated (wide); at large n it beats Hoeffding.
    assert!(eb(2) > hoeff(2));
    let n = 10_000;
    assert!(
        eb(n) < hoeff(n),
        "EB {} must beat Hoeffding {} under zero variance at n={n}",
        eb(n),
        hoeff(n)
    );
    // Monotone decreasing in n for fixed variance.
    assert!(eb(100) < eb(50));
    assert!(eb(50) < eb(10));
}

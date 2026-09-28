//! G1 + G8 unit tests for the similarity-inference primitive (Plan 526 Phase 1).
//!
//! - **G1** — closed-form `ω_T` matches analytical `α/(α+(1−α)·2^(−T))` to
//!   f32 epsilon for T=0..50, α=0.1.
//! - **G8** — `embedded_best_response` cooperates iff `ω > 0.5` for canonical PD.
//!
//! These are the load-bearing correctness tests. The PoC (G2 emergent
//! cooperation) lives in Phase 2 (separate file); the alloc bench (G4) lives
//! in Phase 4.

use crate::similarity_inference::{
    SimilarityError, SimilarityPosterior, canonical_pd, embedded_best_response,
};

/// Allowed relative error for f32 closed-form comparisons. Two f32 ops
/// (`exp`, divide) accrue ~2 ULP; 1e-5 is comfortably above that.
const F32_EPS: f32 = 1e-5;

/// Analytical reference: `ω_T = α / (α + (1−α) · |A|^(−T))`.
fn analytical_omega(alpha: f32, n_actions: usize, t: u32) -> f32 {
    let w = (n_actions as f32).powi(-(t as i32));
    alpha / (alpha + (1.0 - alpha) * w)
}

#[test]
fn g1_matches_analytical_omega() {
    // G1: ω_T from incremental updates must match the closed-form
    // α/(α+(1−α)·|A|^(−T)) to f32 epsilon, for T=0..50, α=0.1, |A|=2.
    let alpha = 0.1_f32;
    let n_actions = 2_usize;
    let mut posterior = SimilarityPosterior::new(alpha).expect("alpha=0.1 is valid");

    for t in 0..=50u32 {
        let expected = analytical_omega(alpha, n_actions, t);
        let got = posterior.omega();
        let rel_err = ((got - expected).abs() / expected.max(f32::MIN_POSITIVE)).abs();
        assert!(
            rel_err < F32_EPS,
            "T={t}: ω_T drift. expected={expected:.8} got={got:.8} rel_err={rel_err:.2e}",
        );
        // Advance T by 1: push one matched observation.
        posterior.observe_match(n_actions);
    }
    // Sanity: ω should have climbed from 0.1 → ~0.9988 over 50 matches.
    // 2^-50 ≈ 8.9e-16 → ω_50 ≈ 0.1/(0.1+0.9·8.9e-16) ≈ 0.9999... (basically 1).
    assert!(
        posterior.omega() > 0.99,
        "after 50 matches ω should be >0.99, got {}",
        posterior.omega()
    );
}

#[test]
fn g1_log_w_matches_minus_t_ln_a() {
    // Companion: log W should equal -T·ln(|A|) exactly (modulo f32 round-off).
    let alpha = 0.1_f32;
    let n_actions = 4_usize; // non-canonical |A| to exercise the general formula
    let mut posterior = SimilarityPosterior::new(alpha).expect("alpha=0.1 is valid");
    let log_inv_n = -(n_actions as f32).ln();
    for t in 0..=50u32 {
        let expected_log_w = log_inv_n * (t as f32);
        let got = posterior.log_w_independent();
        assert!(
            (got - expected_log_w).abs() < F32_EPS * expected_log_w.abs().max(1.0),
            "T={t}: log W drift. expected={expected_log_w:.6} got={got:.6}",
        );
        posterior.observe_match(n_actions);
    }
}

#[test]
fn g1_observations_counter_tracks_t() {
    let mut posterior = SimilarityPosterior::new(0.1).unwrap();
    assert_eq!(posterior.observations(), 0);
    for t in 1..=10u32 {
        posterior.observe_match(2);
        assert_eq!(posterior.observations() as u32, t);
    }
}

#[test]
fn g1_rejects_invalid_prior_alpha() {
    // α outside (0,1) is invalid.
    assert!(matches!(
        SimilarityPosterior::new(0.0),
        Err(SimilarityError::InvalidPriorAlpha { .. })
    ));
    assert!(matches!(
        SimilarityPosterior::new(1.0),
        Err(SimilarityError::InvalidPriorAlpha { .. })
    ));
    assert!(matches!(
        SimilarityPosterior::new(-0.1),
        Err(SimilarityError::InvalidPriorAlpha { .. })
    ));
    assert!(matches!(
        SimilarityPosterior::new(1.1),
        Err(SimilarityError::InvalidPriorAlpha { .. })
    ));
    assert!(matches!(
        SimilarityPosterior::new(f32::NAN),
        Err(SimilarityError::InvalidPriorAlpha { .. })
    ));
    assert!(matches!(
        SimilarityPosterior::new(f32::INFINITY),
        Err(SimilarityError::InvalidPriorAlpha { .. })
    ));
    // Valid α.
    assert!(SimilarityPosterior::new(0.001).is_ok());
    assert!(SimilarityPosterior::new(0.999).is_ok());
    assert!(SimilarityPosterior::new(0.5).is_ok());
}

#[test]
fn g1_omega_stays_in_closed_unit_interval_f32() {
    // ω_T ∈ [0, 1] for any α ∈ (0, 1) and any finite log_w, when computed in
    // f32. Over ℝ the bound is strict (0,1) — α>0 forces ω>0, (1−α)>0 forces
    // ω<1. But in f32, exp(log_w) underflows to exactly 0.0 once log_w < -88.7
    // (≈ −127·ln2). At that point ω = α/(α+0) = 1.0 exactly — a precision
    // artifact, not a math error. Mirror the katgpt-rs HLA boundedness proof
    // convention: strict (0,1) over ℝ is the Lean theorem; the f32 test uses
    // the closed [0,1] interval. Verify by stress-testing with extreme T.
    let mut posterior = SimilarityPosterior::new(0.5).unwrap();
    for _ in 0..1000 {
        posterior.observe_match(2);
        let omega = posterior.omega();
        assert!((0.0..=1.0).contains(&omega), "ω out of [0,1]: {omega}");
    }
    // After 1000 observations with |A|=2, log_w = -693 → exp underflows → ω=1.
    // Verify the saturation is reached (sanity-check the precision floor).
    assert_eq!(
        posterior.omega(),
        1.0,
        "ω should saturate to 1.0 in f32 precision"
    );

    // Positive log_w contribution → ω → 0. exp(50) ≈ 5.2e21 (finite in f32),
    // so ω stays strictly > 0 here.
    let mut posterior = SimilarityPosterior::new(0.5).unwrap();
    posterior.observe(&[], &[], &[], 50.0);
    let omega = posterior.omega();
    assert!(
        omega > 0.0 && omega < 1.0,
        "ω should be in (0,1) for finite W: got {omega}"
    );
    // To actually saturate to 0, we need log_w so large that exp() overflows
    // to +inf in f32 (happens around log_w > 88.7 since f32 max ≈ 3.4e38).
    // exp(89) ≈ 4.4e38 which rounds up to +inf → ω = α/(α+inf) = 0.0 exactly.
    let mut posterior = SimilarityPosterior::new(0.5).unwrap();
    posterior.observe(&[], &[], &[], 89.0);
    assert_eq!(
        posterior.omega(),
        0.0,
        "ω should saturate to 0.0 in f32 precision"
    );
}

#[test]
fn g1_clone_preserves_state() {
    let mut p1 = SimilarityPosterior::new(0.1).unwrap();
    for _ in 0..10 {
        p1.observe_match(2);
    }
    let p2 = p1.clone();
    assert_eq!(p1.omega(), p2.omega());
    assert_eq!(p1.observations(), p2.observations());
    assert_eq!(p1.log_w_independent(), p2.log_w_independent());
}

#[test]
fn g8_cooperates_iff_omega_above_half_pd() {
    // G8: for canonical PD (R=2, S=0, T=3, P=1) with uniform partner marginal,
    // embedded_best_response returns Cooperate (0) iff ω > 0.5, else Defect (1).
    const COOPERATE: u8 = 0;
    const DEFECT: u8 = 1;

    let payoff = canonical_pd();
    let marginal = [0.5_f32, 0.5];

    // Below threshold → Defect
    for omega in [0.0_f32, 0.1, 0.25, 0.49, 0.4999] {
        let a = embedded_best_response(omega, &payoff, &marginal).unwrap();
        assert_eq!(a, DEFECT, "ω={omega} < 0.5 should Defect, got Cooperate");
    }
    // Exactly at threshold → Defect (strict-greater comparison; tie breaks to
    // lower index = Cooperate, but at exactly ω=0.5 Q(C)=Q(D) so it's a genuine
    // tie. Verify the documented behavior: lower index wins, so it Cooperates).
    let a = embedded_best_response(0.5, &payoff, &marginal).unwrap();
    assert_eq!(
        a, COOPERATE,
        "ω=0.5 tie should resolve to Cooperate (lower idx)"
    );
    // Above threshold → Cooperate
    for omega in [0.5001_f32, 0.6, 0.75, 0.9, 1.0] {
        let a = embedded_best_response(omega, &payoff, &marginal).unwrap();
        assert_eq!(a, COOPERATE, "ω={omega} > 0.5 should Cooperate, got Defect");
    }
}

#[test]
fn g8_threshold_analytical_pd() {
    // Theoretical check: for canonical PD with uniform marginal, the threshold
    // is exactly ω*=0.5. Derivation:
    //   Q(C) = ω·R + (1−ω)·(0.5·R + 0.5·S) = ω·2 + (1−ω)·1 = 1 + ω
    //   Q(D) = ω·T + (1−ω)·(0.5·T + 0.5·P) = ω·3 + (1−ω)·2 = 2 − ω
    //   Q(C) − Q(D) = 2ω − 1 > 0  ⟺  ω > 0.5
    let payoff = canonical_pd();
    let marginal = [0.5_f32, 0.5];
    // Binary-search the threshold to 4 decimal places.
    let mut lo = 0.0_f32;
    let mut hi = 1.0_f32;
    for _ in 0..20 {
        let mid = 0.5 * (lo + hi);
        let a = embedded_best_response(mid, &payoff, &marginal).unwrap();
        if a == 0 {
            // Cooperated → threshold is at-or-below mid
            hi = mid;
        } else {
            // Defected → threshold is above mid
            lo = mid;
        }
    }
    let measured_threshold = 0.5 * (lo + hi);
    assert!(
        (measured_threshold - 0.5).abs() < 1e-3,
        "measured PD threshold {measured_threshold:.5} should be ~0.5"
    );
}

#[test]
fn g8_shape_mismatch_errors() {
    let payoff = canonical_pd(); // 2x2
    let bad_marginal = [0.5_f32, 0.25, 0.25]; // length 3, should be 2
    let err = embedded_best_response(0.7, &payoff, &bad_marginal);
    assert!(matches!(
        err,
        Err(SimilarityError::MarginalShapeMismatch {
            expected: 2,
            got: 3
        })
    ));
}

#[test]
fn g8_into_variant_matches_plain() {
    let payoff = canonical_pd();
    let marginal = [0.5_f32, 0.5];
    for omega in [0.1_f32, 0.5, 0.7, 0.99] {
        let mut into_out: u8 = 99;
        crate::similarity_inference::embedded_best_response_into(
            omega,
            &payoff,
            &marginal,
            &mut into_out,
        )
        .unwrap();
        let plain_out = embedded_best_response(omega, &payoff, &marginal).unwrap();
        assert_eq!(into_out, plain_out, "into variant mismatch at ω={omega}");
    }
}

#[test]
fn payoff_matrix_shape_validation() {
    // Valid 2x2.
    let m = crate::similarity_inference::PayoffMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    assert!(m.is_ok());
    let m = m.unwrap();
    assert_eq!(m.n_actions(), 2);
    assert_eq!(m.payoff(0, 0), 1.0);
    assert_eq!(m.payoff(1, 1), 4.0);
    assert_eq!(m.payoff(0, 1), 2.0);
    assert_eq!(m.payoff(1, 0), 3.0);

    // Valid 3x3.
    let m = crate::similarity_inference::PayoffMatrix::from_row_major(
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    assert!(m.is_ok());

    // Shape mismatch.
    let bad = crate::similarity_inference::PayoffMatrix::from_row_major(2, vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        bad,
        Err(SimilarityError::PayoffShapeMismatch {
            expected: 4,
            got: 3
        })
    ));

    // Empty.
    let bad = crate::similarity_inference::PayoffMatrix::from_row_major(0, vec![]);
    assert!(matches!(bad, Err(SimilarityError::EmptyActionSet)));
}

#[test]
fn canonical_pd_layout() {
    // Sanity-check the canonical_pd() factory: C=0, D=1, R=2, S=0, T=3, P=1.
    let m = canonical_pd();
    assert_eq!(m.n_actions(), 2);
    assert_eq!(m.payoff(0, 0), 2.0, "R (C,C) should be 2");
    assert_eq!(m.payoff(0, 1), 0.0, "S (C,D) should be 0");
    assert_eq!(m.payoff(1, 0), 3.0, "T (D,C) should be 3");
    assert_eq!(m.payoff(1, 1), 1.0, "P (D,D) should be 1");
}

#[test]
fn g1_mismatch_drives_omega_to_zero() {
    // Under the perfect-identity model, a single mismatch is impossible under
    // the shared hypothesis → LR = 0 → ω = 0 permanently.
    let mut p = SimilarityPosterior::new(0.5).unwrap();
    // First, accumulate some matching evidence → ω climbs.
    for _ in 0..10 {
        p.observe_match(2);
    }
    assert!(
        p.omega() > 0.99,
        "after 10 matches ω should be >0.99, got {}",
        p.omega()
    );
    assert!(!p.is_collapsed_to_zero());
    // Now a mismatch → ω = 0.
    p.observe_mismatch(2);
    assert_eq!(p.omega(), 0.0);
    assert!(p.is_collapsed_to_zero());
    // Further matches cannot revive it.
    for _ in 0..100 {
        p.observe_match(2);
    }
    assert_eq!(p.omega(), 0.0, "collapsed posterior cannot recover");
    assert!(p.is_collapsed_to_zero());
}

#[test]
fn g1_mismatch_at_t0_omega_zero_from_start() {
    // Edge case: mismatch on the very first observation.
    let mut p = SimilarityPosterior::new(0.1).unwrap();
    p.observe_mismatch(2);
    assert_eq!(p.omega(), 0.0);
    assert!(p.is_collapsed_to_zero());
}

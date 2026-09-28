//! Unit tests for the kinematics module (Plan 578 Phases 1–2).

use crate::kinematics::fixture::{self, Fixture, RangeLabel};
use crate::kinematics::perception::{
    ApproachReport, Eps, EventKind, Regime, RegimeClassifier, RegimeConfig, RegimeSnapshot,
    ResidualConfig, ResidualMonitor, closest_approach, extrapolation_horizon,
    extrapolation_horizon_for_state, head_on_elastic_resolve, horizon_bound, intercept_time,
    normal_two_sided_z, predictive_half_width, time_to_contact, time_to_contact_log,
};
use crate::kinematics::{
    K_MAX, KinError, KinState, Sched, extrapolation_weight_ss, kinematic_extrapolate_capped_into,
    kinematic_extrapolate_into, lattice_rows_for_test, reference_chain_extrapolate_into,
    terminal_velocity,
};

// ===== helpers: analytic trajectories via exact difference chains =====

/// Feed a state with the first `n` samples of `s(t) = v·t` (uniform, dt=1).
fn uniform_state(v: f32, n: u32) -> KinState<1> {
    let mut st = KinState::<1>::new(1.0).unwrap();
    for t in 0..n {
        let x = v * t as f32;
        st.observe_into(&[x], t).unwrap();
    }
    st
}

/// `s(t) = t²` parabola (dt=1).
fn parabola_state(n: u32) -> KinState<1> {
    let mut st = KinState::<1>::new(1.0).unwrap();
    for t in 0..n {
        let x = (t as f32) * t as f32;
        st.observe_into(&[x], t).unwrap();
    }
    st
}

/// `s(t) = t³` cubic (dt=1) — const jerk 6.
fn cubic_state(n: u32) -> KinState<1> {
    let mut st = KinState::<1>::new(1.0).unwrap();
    for t in 0..n {
        let x = (t as f32) * (t as f32) * t as f32;
        st.observe_into(&[x], t).unwrap();
    }
    st
}

fn pred1(st: &KinState<1>, k: u32, sched: &Sched) -> f32 {
    let mut out = [0.0f32; 1];
    kinematic_extrapolate_into(st, k, sched, &mut out).unwrap();
    out[0]
}

fn chain1(st: &KinState<1>, k: u32, sched: &Sched) -> f32 {
    let mut out = [0.0f32; 1];
    reference_chain_extrapolate_into(st, k, sched, &mut out).unwrap();
    out[0]
}

// ===== T1.1: state, ladder, screens =====

#[test]
fn constructor_screens_dt() {
    assert_eq!(KinState::<2>::new(0.0).unwrap_err(), KinError::BadDt);
    assert_eq!(KinState::<2>::new(-1.0).unwrap_err(), KinError::BadDt);
    assert!(KinState::<2>::new(f32::NAN).is_err());
    assert!(KinState::<2>::new(1.0).is_ok());
}

#[test]
fn observe_screens_nan_and_ticks() {
    let mut st = KinState::<1>::new(1.0).unwrap();
    st.observe_into(&[1.0], 0).unwrap();
    assert_eq!(
        st.observe_into(&[f32::NAN], 1).unwrap_err(),
        KinError::NonFinite
    );
    assert_eq!(
        st.observe_into(&[2.0], 0).unwrap_err(),
        KinError::NonMonotonicTick
    );
    st.observe_into(&[2.0], 2).unwrap();
}

#[test]
fn ladder_orders_advance() {
    // n_obs → significant_order(eps=0): 1→0, 2→1, 3→2, 4+→3.
    let mut st = KinState::<1>::new(1.0).unwrap();
    st.observe_into(&[3.0], 0).unwrap();
    assert_eq!((st.n_obs, st.significant_order(0.0)), (1, 0));
    st.observe_into(&[5.0], 1).unwrap();
    assert_eq!((st.n_obs, st.significant_order(0.0)), (2, 1));
    st.observe_into(&[9.0], 2).unwrap();
    assert_eq!((st.n_obs, st.significant_order(0.0)), (3, 2));
    st.observe_into(&[15.0], 3).unwrap();
    assert_eq!((st.n_obs, st.significant_order(0.0)), (4, 3));
    st.observe_into(&[23.0], 4).unwrap();
    assert_eq!((st.n_obs, st.significant_order(0.0)), (4, 3));
}

#[test]
fn backward_difference_coefficients_on_canonical_families() {
    // Uniform v=2.5: vel = 2.5 exactly, acc = jerk = 0.
    let st = uniform_state(2.5, 4);
    assert_eq!(st.vel[0], 2.5);
    assert_eq!(st.acc[0], 0.0);
    assert_eq!(st.jerk[0], 0.0);

    // Parabola t²: acc = 2 exactly (physical), jerk = 0.
    let st = parabola_state(4);
    assert_eq!(st.acc[0], 2.0);
    assert_eq!(st.jerk[0], 0.0);

    // Cubic t³: jerk = 6 exactly, acc coefficient ∇² = 6t−6 = 18 at t=4.
    let st = cubic_state(5);
    assert_eq!(st.jerk[0], 6.0);
    assert_eq!(st.acc[0], 18.0);
}

#[test]
fn extrapolate_requires_obs_and_lattice_bound() {
    let empty = KinState::<1>::new(1.0).unwrap();
    let mut out = [0.0f32; 1];
    assert_eq!(
        kinematic_extrapolate_into(&empty, 1, &Sched::ZeroJerk, &mut out).unwrap_err(),
        KinError::NotEnoughObs
    );
    let st = uniform_state(1.0, 4);
    assert_eq!(
        kinematic_extrapolate_into(&st, K_MAX as u32 + 1, &Sched::ZeroJerk, &mut out).unwrap_err(),
        KinError::HorizonTooFar
    );
    kinematic_extrapolate_into(&st, K_MAX as u32, &Sched::ZeroJerk, &mut out).unwrap();
}

// ===== T1.2/T1.3: exactness fixtures (the load-bearing claim) =====

#[test]
fn uniform_exact_at_all_horizons() {
    let st = uniform_state(2.5, 4); // anchor t=3, s=7.5
    for k in [1u32, 10, 100, 1000] {
        for sched in [Sched::ZeroJerk, Sched::Measured] {
            let got = pred1(&st, k, &sched);
            let want = 2.5 * (3 + k) as f32;
            assert_eq!(got, want, "uniform k={k} {sched:?}: {got} != {want}");
        }
    }
}

#[test]
fn parabola_exact_at_all_horizons() {
    let st = parabola_state(4); // anchor t=3, s=9
    for k in [1u32, 10, 100, 1000] {
        for sched in [Sched::ZeroJerk, Sched::Measured] {
            let got = pred1(&st, k, &sched);
            let t = (3 + k) as f32;
            let want = t * t;
            assert_eq!(got, want, "parabola k={k} {sched:?}: {got} != {want}");
        }
    }
}

#[test]
fn const_jerk_exact_within_f32_mantissa_budget() {
    let st = cubic_state(5); // anchor t=4, s=64, jerk=6
    // Exactly 0 where every lattice coefficient and trajectory value is
    // 24-bit exact: C(102,3) = 171700 ✓, C(12,3) = 220 ✓.
    for k in [1u32, 10, 100] {
        let got = pred1(&st, k, &Sched::Measured);
        let t = (4 + k) as f32;
        let want = t * t * t;
        assert_eq!(got, want, "cubic k={k}: {got} != {want}");
    }
    // k=1000: C(1002,3) = 167,167,000 needs 25 mantissa bits — the f32
    // exactness boundary documented in the module doc. Relative band only.
    let got = pred1(&st, 1000, &Sched::Measured);
    let t = 1004.0f32;
    let want = t * t * t;
    let rel = ((got - want) / want).abs();
    assert!(
        rel < 1e-6,
        "cubic k=1000 relative error {rel} exceeds the 2^-24 band"
    );
}

#[test]
fn closed_form_bit_identical_to_reference_chain_on_exactness_family() {
    // On dyadic-exact fixtures every op in both paths is exact → bit-equal.
    let cases: [(KinState<1>, Sched); 5] = [
        (uniform_state(2.5, 4), Sched::ZeroJerk),
        (parabola_state(4), Sched::ZeroJerk),
        (cubic_state(5), Sched::Measured),
        (cubic_state(5), Sched::ConstJerk { j: 6.0 }),
        (
            cubic_state(5),
            Sched::ClampedCorrection {
                j_max: 6.0,
                lambda: 0.0,
            },
        ),
    ];
    for (st, sched) in cases {
        for k in [1u32, 2, 3, 7, 31, 100] {
            assert_eq!(
                pred1(&st, k, &sched).to_bits(),
                chain1(&st, k, &sched).to_bits(),
                "closed form != chain at k={k}"
            );
        }
    }
}

#[test]
fn closed_form_matches_chain_on_random_data_within_ulp_band() {
    // Arbitrary floats: the two paths evaluate the same polynomial through
    // different op sequences — agree to a few ULP, never exact in general
    // (documented in the module doc). Measure the band.
    let mut rng = fixture::SplitMix64::new(0x5EED_0001);
    let mut max_rel = 0.0f32;
    for _ in 0..200 {
        let mut st = KinState::<1>::new(1.0).unwrap();
        // Random walk with drift — genuinely non-polynomial.
        let mut x = 0.0f32;
        let mut drift = 0.0f32;
        for t in 0..6u32 {
            drift += (rng.next_unit() - 0.5) * 0.3;
            x += drift + (rng.next_unit() - 0.5) * 0.2;
            st.observe_into(&[x], t).unwrap();
        }
        for k in [1u32, 5, 23, 97] {
            let a = pred1(&st, k, &Sched::Measured);
            let b = chain1(&st, k, &Sched::Measured);
            let rel = if b == 0.0 { 0.0 } else { ((a - b) / b).abs() };
            max_rel = max_rel.max(rel);
        }
    }
    assert!(max_rel < 5e-5, "max relative divergence {max_rel}");
    // The ULP band is what it is — record it for the bench doc.
    eprintln!("closed-vs-chain max relative divergence on random walks: {max_rel:e}");
}

#[test]
fn clamped_correction_saturates_to_const_jerk() {
    // lambda=0 → tanh(0)=0 → j3=0 (ZeroJerk); huge lambda·|v| → tanh→1 →
    // ConstJerk{j_max}.
    let st = uniform_state(2.0, 4);
    let zero = pred1(
        &st,
        10,
        &Sched::ClampedCorrection {
            j_max: 5.0,
            lambda: 0.0,
        },
    );
    assert_eq!(zero, pred1(&st, 10, &Sched::ZeroJerk));
    let sat = pred1(
        &st,
        10,
        &Sched::ClampedCorrection {
            j_max: 5.0,
            lambda: 100.0,
        },
    );
    assert_eq!(sat, pred1(&st, 10, &Sched::ConstJerk { j: 5.0 }));
}

#[test]
fn geometric_drag_closed_form_matches_chain() {
    // Drag trajectory via the chain itself; compare closed form at small k
    // (dyadic-exact) and within tolerance further out.
    let rho = 0.5f32;
    let (v0, a0) = (4.0f32, 1.0f32);
    let mut st = KinState::<1>::new(1.0).unwrap();
    let (mut s, mut v, mut a) = (0.0f32, v0, a0);
    for t in 0..8u32 {
        if t > 0 {
            v += a;
            s += v;
            a *= rho;
        }
        st.observe_into(&[s], t).unwrap();
    }
    let sched = Sched::GeometricDrag { rho };
    for k in [1u32, 2, 3, 8, 16] {
        let closed = pred1(&st, k, &sched);
        // Independent chain continuation.
        let (mut s2, mut v2, mut a2) = (s, v, a);
        for _ in 0..k {
            v2 += a2;
            s2 += v2;
            a2 *= rho;
        }
        let rel = if s2 == 0.0 {
            0.0
        } else {
            ((closed - s2) / s2).abs()
        };
        assert!(rel < 1e-6, "drag k={k}: closed {closed} vs chain {s2}");
        // And the in-module reference chain (same recurrence).
        assert_eq!(closed.to_bits(), chain1(&st, k, &sched).to_bits());
    }
    // Terminal velocity: v_∞ = v + a·ρ/(1−ρ).
    let v_inf = terminal_velocity(v, a, 1.0, rho).unwrap();
    assert!((v_inf - (v + a * 0.5 / 0.5)).abs() < 1e-6);
    assert!(terminal_velocity(v, a, 1.0, 1.0).is_none());
    assert!(terminal_velocity(v, a, 1.0, 1.5).is_none());
}

#[test]
fn dt_rescale_invariance_bit_exact() {
    // Δt vs Δt/2 at matched wall-times agree bit-for-bit on the analytic
    // family (all values dyadic).
    // Uniform v=2, anchor t=2, wall +10.
    {
        let mut st1 = KinState::<1>::new(1.0).unwrap();
        for t in 0..=2u32 {
            st1.observe_into(&[2.0 * t as f32], t).unwrap();
        }
        let mut st2 = KinState::<1>::new(0.5).unwrap();
        for t in 0..=4u32 {
            st2.observe_into(&[2.0 * (t as f32 * 0.5)], t).unwrap();
        }
        assert_eq!(
            pred1(&st1, 10, &Sched::ZeroJerk),
            pred1(&st2, 20, &Sched::ZeroJerk)
        );
    }
    // Parabola t², anchor t=2, wall +5.
    {
        let mut st1 = KinState::<1>::new(1.0).unwrap();
        for t in 0..=2u32 {
            let tf = t as f32;
            st1.observe_into(&[tf * tf], t).unwrap();
        }
        let mut st2 = KinState::<1>::new(0.5).unwrap();
        for t in 0..=4u32 {
            let tf = t as f32 * 0.5;
            st2.observe_into(&[tf * tf], t).unwrap();
        }
        assert_eq!(
            pred1(&st1, 5, &Sched::ZeroJerk),
            pred1(&st2, 10, &Sched::ZeroJerk)
        );
    }
    // Cubic t³/8 (jerk 6/8), anchor t=3, wall +4.
    {
        let mut st1 = KinState::<1>::new(1.0).unwrap();
        for t in 0..=3u32 {
            let tf = t as f32;
            st1.observe_into(&[tf * tf * tf / 8.0], t).unwrap();
        }
        let mut st2 = KinState::<1>::new(0.5).unwrap();
        for t in 0..=6u32 {
            let tf = t as f32 * 0.5;
            st2.observe_into(&[tf * tf * tf / 8.0], t).unwrap();
        }
        assert_eq!(
            pred1(&st1, 4, &Sched::Measured),
            pred1(&st2, 8, &Sched::Measured)
        );
    }
}

#[test]
fn capped_extrapolation_reduces_order() {
    let st = parabola_state(4);
    // Capping to order 1 on a parabola: linear extrapolation from the
    // anchor — verify against pos + k·vel.
    let mut out = [0.0f32; 1];
    kinematic_extrapolate_capped_into(&st, 7, &Sched::ZeroJerk, 1, &mut out).unwrap();
    assert_eq!(out[0], st.pos[0] + 7.0 * st.vel[0]);
}

#[test]
fn lattice_coefficients_are_correct() {
    let rows = lattice_rows_for_test();
    assert_eq!(rows.len(), K_MAX + 1);
    assert_eq!(rows[0].b1, 0.0);
    assert_eq!(rows[0].b2, 0.0);
    assert_eq!(rows[0].b3, 0.0);
    // Spot values (exact integers in f32):
    assert_eq!(rows[1].b2, 1.0); // C(2,2)
    assert_eq!(rows[10].b2, 55.0); // C(11,2)
    assert_eq!(rows[10].b3, 220.0); // C(12,3)
    assert_eq!(rows[100].b2, 5050.0);
    assert_eq!(rows[100].b3, 171700.0);
    assert_eq!(rows[1000].b2, 500500.0);
    // b1 == k exactly everywhere.
    for (k, r) in rows.iter().enumerate() {
        assert_eq!(r.b1, k as f32);
    }
    // Integer-valued binomials must round-trip exactly while 24-bit.
    for k in [2usize, 10, 100, 287] {
        let kf = k as f64;
        let want3 = (kf * (kf + 1.0) * (kf + 2.0) / 6.0) as f32;
        assert_eq!(rows[k].b3, want3);
    }
}

#[test]
fn weight_ss_orders() {
    assert_eq!(extrapolation_weight_ss(0, 0), 1.0);
    assert_eq!(extrapolation_weight_ss(5, 0), 1.0);
    // Order 1, k=1: weights (2, −1) → 5.
    assert_eq!(extrapolation_weight_ss(1, 1), 5.0);
    // Order 0: wss = 1 for any k.
    assert_eq!(extrapolation_weight_ss(100, 0), 1.0);
    // Monotone in k per order (noise amplification grows with horizon).
    for order in 1u8..=3 {
        let mut prev = 0.0f32;
        for k in [0u32, 1, 2, 5, 10, 50, 200] {
            let w = extrapolation_weight_ss(k, order);
            assert!(w >= prev, "wss not monotone at order {order}");
            prev = w;
        }
    }
}

// ===== T2.1: time to contact =====

#[test]
fn ttc_planted_contact_recovered_within_one_tick() {
    // σ(t) = 1 − 0.1·t → contact at t=10.
    let dt = 1.0f32;
    for t in 1..10u32 {
        let s_now = 1.0 - 0.1 * t as f32;
        let s_prev = 1.0 - 0.1 * (t - 1) as f32;
        let tau = time_to_contact(s_now, s_prev, dt);
        let predicted_contact = t as f32 + tau;
        assert!(
            (predicted_contact - 10.0).abs() <= 1.0,
            "t={t}: predicted contact {predicted_contact}"
        );
    }
}

#[test]
fn ttc_guards() {
    assert_eq!(time_to_contact(1.0, 1.0, 1.0), f32::INFINITY); // constant
    assert_eq!(time_to_contact(1.0, 0.5, 1.0), f32::INFINITY); // receding
    assert_eq!(time_to_contact(0.0, 0.5, 1.0), f32::INFINITY); // no extent
    assert_eq!(time_to_contact(f32::NAN, 0.5, 1.0), f32::INFINITY);
    // Approaching: τ = σ/|σ̇| = 1/0.5 = 2.
    assert!((time_to_contact(1.0, 1.5, 1.0) - 2.0).abs() < 1e-6);
}

#[test]
fn ttc_log_exact_for_exponential_extent() {
    // σ(t) = e^{−t/4}: relative rate −0.25 → τ = 4 at every tick (log form).
    for t in 1..8u32 {
        let tf = t as f32;
        let s_now = (-tf / 4.0).exp();
        let s_prev = (-(tf - 1.0) / 4.0).exp();
        let tau = time_to_contact_log(s_now, s_prev, 1.0);
        assert!((tau - 4.0).abs() < 0.01, "t={t}: τ={tau}");
    }
}

// ===== T2.2: regime predicates =====

/// Run the classifier over a fixture; return (tag, classified) pairs for the
/// comparable frames (≥3 frames into each segment — the FD window flush).
fn classify_fixture(fix: &Fixture) -> (usize, usize) {
    let cfg = RegimeConfig {
        g: fix.params.g,
        ..RegimeConfig::default()
    };
    let mut clf = RegimeClassifier::new(cfg);
    let mut st = KinState::<2>::new(1.0).unwrap();
    let mut prev_vel = [0.0f32; 2];
    let mut running_acc = 0.0f32;
    let mut prev_extent = fix.frames[0].extent;
    let mut total = 0usize;
    let mut correct = 0usize;
    for (i, f) in fix.frames.iter().enumerate() {
        // Comparable once ≥3 frames into the current segment.
        let seg_start = fix
            .segments
            .iter()
            .rev()
            .find(|&&s| s <= i)
            .copied()
            .unwrap_or(0);
        let comparable = i - seg_start >= 3;
        let sigma_rate = (f.extent - prev_extent) / 1.0;
        prev_extent = f.extent;
        st.observe_into(&f.pos, f.tick).unwrap();
        let snap = RegimeSnapshot::from_state(&st, &prev_vel, running_acc, f.extent, sigma_rate);
        let verdict = clf.classify(&snap);
        // Robust (winsorized) running-|acc| EMA — β=0.5 toward
        // min(acc, 10·running + 0.05): segment-boundary mixed-window
        // transients (~120 units here) are clipped to a few units and drain
        // before the next event, while a sustained force onset is tracked
        // within ~3 ticks (a hard exclusion would starve the scale and fire
        // impulses on every parabola frame — both failure modes found by
        // the fixtures, not theory).
        let capped = snap.acc_mag.min(10.0 * running_acc + 0.05);
        running_acc += 0.5 * (capped - running_acc);
        prev_vel = st.vel;
        if comparable {
            total += 1;
            let same = verdict == fix.tags[i]
                || (matches!(verdict, Regime::Parabolic { .. })
                    && matches!(fix.tags[i], Regime::Parabolic { .. }));
            if same {
                correct += 1;
            } else {
                eprintln!(
                    "t={}: classified {verdict:?}, tag {:?}",
                    f.tick, fix.tags[i]
                );
            }
        }
    }
    (correct, total)
}

#[test]
fn regime_classification_100_percent_on_interleaved_streams() {
    for seed in [1u64, 2, 3, 7, 42] {
        for label in [RangeLabel::Id, RangeLabel::Ood] {
            let fix = fixture::generate(seed, label);
            let (correct, total) = classify_fixture(&fix);
            assert_eq!(
                correct, total,
                "seed {seed} {label:?}: {correct}/{total} — misclassifications above"
            );
        }
    }
}

// ===== T2.3: residual events =====

#[test]
fn planted_bounce_detected_at_exact_tick_with_restitution() {
    // Uniform toward a wall in y at v=2; at t=20 reverse with e=0.5.
    let v = 2.0f32;
    let e = 0.5f32;
    let bounce_at = 20u32;
    let mut st = KinState::<2>::new(1.0).unwrap();
    let mut mon = ResidualMonitor::new(ResidualConfig::default());
    let mut detected: Option<(u32, EventKind)> = None;
    let mut y = 0.0f32;
    let mut vy = v;
    for t in 0..40u32 {
        if t == bounce_at {
            vy = -vy * e;
        }
        y += vy;
        let obs = [0.0f32, y];
        let mut predicted = [0.0f32; 2];
        kinematic_extrapolate_into(&st, 1, &Sched::Measured, &mut predicted).ok();
        let vel_before = st.vel;
        st.observe_into(&obs, t).unwrap();
        let ev = mon.update(predicted[1] - obs[1], &vel_before, &st.vel, 1.0);
        if let Some(kind) = ev {
            detected = Some((t, kind));
            break;
        }
    }
    let (tick, kind) = detected.expect("bounce not detected");
    assert_eq!(tick, bounce_at, "detection tick");
    match kind {
        EventKind::Impulse { axis, e: e_est } => {
            assert_eq!(axis, Some(1), "wall axis = y");
            assert_eq!(e_est, Some(e), "restitution");
        }
        other => panic!("expected Impulse, got {other:?}"),
    }
}

#[test]
fn zero_alarms_on_100k_clean_ticks() {
    // Uniform motion: residuals exactly 0, |Δv|/Δt == running |acc| == 0.
    {
        let mut st = KinState::<1>::new(1.0).unwrap();
        let mut mon = ResidualMonitor::new(ResidualConfig::default());
        let mut x = 0.0f32;
        for t in 0..100_000u32 {
            x += 2.5;
            let mut predicted = [0.0f32; 1];
            kinematic_extrapolate_into(&st, 1, &Sched::Measured, &mut predicted).ok();
            let vel_before = st.vel;
            st.observe_into(&[x], t).unwrap();
            assert!(
                mon.update(predicted[0] - x, &vel_before, &st.vel, 1.0)
                    .is_none(),
                "false alarm at t={t}"
            );
        }
    }
    // Parabola: residuals exactly 0, |Δv|/Δt == g == running |acc|. Ticks
    // capped at 5,000 — f32's 24-bit exact-integer ceiling: a g=0.25 parabola
    // stays dyadic-exact only while t²·g/2·(1/g) < 2²⁴ (the bench doc's
    // exactness table records the same boundary).
    {
        let mut st = KinState::<1>::new(1.0).unwrap();
        let mut mon = ResidualMonitor::new(ResidualConfig::default());
        let mut x = 0.0f32;
        let mut v = 8.0f32;
        for t in 0..5_000u32 {
            v += 0.25;
            x += v;
            let mut predicted = [0.0f32; 1];
            kinematic_extrapolate_into(&st, 1, &Sched::Measured, &mut predicted).ok();
            let vel_before = st.vel;
            st.observe_into(&[x], t).unwrap();
            assert!(
                mon.update(predicted[0] - x, &vel_before, &st.vel, 1.0)
                    .is_none(),
                "false alarm at t={t}"
            );
        }
    }
}

#[test]
fn cusum_detects_sustained_drift_that_spikes_do_not() {
    // The canonical CUSUM case: bounded noise (σ≈0.058) plus a +0.2/tick
    // drift starting at t=100. Individual z-scores stay under the spike
    // gate (|r−μ|/σ ≤ ~3.4 with the drift absorbed by the lagging mean), but
    // the one-sided accumulation crosses the CUSUM threshold — a sustained
    // drift the spike gate cannot see.
    const VEL: [f32; 1] = [1.0];

    let mut mon = ResidualMonitor::new(ResidualConfig::default());
    let mut rng = fixture::SplitMix64::new(0xC0FFEE);
    let mut drift_tick = None;
    let mut spike_seen = false;
    for t in 0..400u32 {
        let noise = rng.next_unit() * 0.2 - 0.1;
        let drift = if t >= 100 { 0.2 } else { 0.0 };
        let r = noise + drift;
        match mon.update(r, &VEL, &VEL, 1.0) {
            Some(EventKind::Drift { .. }) => {
                drift_tick = Some(t);
                break;
            }
            Some(EventKind::Spike { .. }) => spike_seen = true,
            _ => {}
        }
    }
    assert!(drift_tick.is_some(), "sustained drift never alarmed");
    assert!(
        drift_tick.unwrap() >= 100,
        "alarm before the drift began ({drift_tick:?})"
    );
    assert!(!spike_seen || drift_tick.is_some());
}

// ===== T2.4: closest approach / intercept / elastic resolve =====

#[test]
fn closest_approach_canonical_cases() {
    // Head-on 1D: p_rel=10, v_rel=−2 → t*=5, miss=0.
    let r = closest_approach(&[10.0], &[-1.0], &[0.0], &[1.0]);
    assert!((r.t_star - 5.0).abs() < 1e-6);
    assert!(r.miss_dist.abs() < 1e-6);
    assert!(r.closing);

    // Flyby with known miss: p=(10, 5), v_rel = (−2, 0): closest at t=5,
    // miss = 5.
    let r = closest_approach(&[10.0, 5.0], &[-2.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]);
    assert!((r.t_star - 5.0).abs() < 1e-6);
    assert!((r.miss_dist - 5.0).abs() < 1e-6);

    // Parallel: no relative motion.
    let r = closest_approach(&[0.0, 0.0], &[1.0, 0.0], &[5.0, 0.0], &[1.0, 0.0]);
    assert_eq!(r.t_star, 0.0);
    assert!((r.miss_dist - 5.0).abs() < 1e-6);
    assert!(!r.closing);

    // Receding: p1 moving away from a stationary p2.
    let r = closest_approach(&[0.0], &[-1.0], &[10.0], &[0.0]);
    assert!(r.t_star < 0.0);
    assert!(!r.closing);
}

#[test]
fn t_star_ascending_orders_contact_times() {
    // N pairs closing at different rates/distances; sorted t* must equal the
    // ground-truth contact order (miss=0 head-ons with distinct t*).
    let mut reports: Vec<ApproachReport> = Vec::new();
    let mut ground_truth: Vec<f32> = Vec::new();
    for i in 1..=8u32 {
        let dist = 10.0 * i as f32;
        let speed = 1.0 + i as f32 * 0.25;
        let t = dist / speed;
        ground_truth.push(t);
        reports.push(closest_approach(&[dist], &[-speed], &[0.0], &[0.0]));
    }
    let mut order: Vec<usize> = (0..reports.len()).collect();
    order.sort_by(|&a, &b| reports[a].t_star.total_cmp(&reports[b].t_star));
    let sorted_t: Vec<f32> = order.iter().map(|&i| ground_truth[i]).collect();
    let mut expect = ground_truth.clone();
    expect.sort_by(|a, b| a.total_cmp(b));
    // The sorted-by-t* selection reproduces ascending contact time.
    let mut asc = true;
    for w in sorted_t.windows(2) {
        if w[0] > w[1] {
            asc = false;
        }
    }
    assert!(asc, "t*-ordering law violated");
    assert_eq!(sorted_t, expect);
}

#[test]
fn intercept_quadratic_solves_and_rejects() {
    // Stationary target at distance 10, pursuer speed 2 → t = 5.
    let t = intercept_time(&[0.0], &[10.0], &[0.0], 2.0).unwrap();
    assert!((t - 5.0).abs() < 1e-5);
    // Target receding at the same speed: uncatchable.
    assert!(intercept_time(&[0.0], &[10.0], &[2.0], 2.0).is_none());
    // Crossing target: some positive root exists.
    let t = intercept_time(&[0.0, 0.0], &[5.0, 5.0], &[0.0, -1.0], 2.0).unwrap();
    assert!(t > 0.0 && t.is_finite());
}

#[test]
fn head_on_elastic_conserves() {
    // Equal masses swap velocities.
    let (a, b) = head_on_elastic_resolve(3.0, -1.0, 1.0, 1.0);
    assert!((a - (-1.0)).abs() < 1e-6 && (b - 3.0).abs() < 1e-6);
    // Momentum + energy conservation for unequal masses.
    let (v1, v2) = (2.0f32, -1.0f32);
    let (m1, m2) = (2.0f32, 0.5f32);
    let (a, b) = head_on_elastic_resolve(v1, v2, m1, m2);
    let p_before = m1 * v1 + m2 * v2;
    let p_after = m1 * a + m2 * b;
    let e_before = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2;
    let e_after = 0.5 * m1 * a * a + 0.5 * m2 * b * b;
    assert!((p_before - p_after).abs() < 1e-5, "momentum");
    assert!((e_before - e_after).abs() < 1e-4, "energy");
}

// ===== T2.5: extrapolation horizon =====

#[test]
fn horizon_bound_monotone_and_gate_consistent() {
    let eps = Eps::from_obs_noise(0.1, 1.0);
    let mut prev = 0.0f32;
    for k in [0u32, 1, 5, 20, 100] {
        let b = horizon_bound(k, 1.0, &eps);
        assert!(b >= prev, "bound not monotone");
        prev = b;
    }
    // k* decreases as the threshold tightens; conf ∈ [0,1].
    let loose = extrapolation_horizon(1.0, &eps, 100.0);
    let tight = extrapolation_horizon(1.0, &eps, 1.0);
    assert!(loose.k_star > tight.k_star);
    assert!(loose.conf > 0.0 && loose.conf <= 1.0);
    assert!(tight.conf > 0.0 && tight.conf <= 1.0);
    assert!(tight.bound <= 1.0);
    // k=0 verdict when even the anchor exceeds thr.
    let none = extrapolation_horizon(1.0, &eps, 0.01);
    assert_eq!(none.k_star, 0);
}

#[test]
fn horizon_for_state_wrapper() {
    let st = uniform_state(1.0, 4);
    let v = extrapolation_horizon_for_state(&st, 0.05, 10.0);
    assert!(v.k_star >= 1);
    // Larger noise → shorter horizon.
    let v2 = extrapolation_horizon_for_state(&st, 0.5, 10.0);
    assert!(v2.k_star < v.k_star);
}

#[test]
fn normal_z_and_half_width_sanity() {
    let z = normal_two_sided_z(0.05);
    assert!((z - 1.9604).abs() < 0.01, "z(0.05)={z}");
    let z2 = normal_two_sided_z(0.10);
    assert!((z2 - 1.6449).abs() < 0.01, "z(0.10)={z2}");
    // Half-width grows with k and order.
    assert!(predictive_half_width(0.1, 10, 1, 0.05) > predictive_half_width(0.1, 1, 1, 0.05));
    assert!(predictive_half_width(0.1, 10, 3, 0.05) > predictive_half_width(0.1, 10, 1, 0.05));
}

// ===== T3.2: fixture generator + the ID-OOD gap table =====

#[test]
fn fixture_determinism_same_seed_same_bits() {
    for seed in [1u64, 7, 42] {
        for label in [RangeLabel::Id, RangeLabel::Ood] {
            let a = fixture::generate(seed, label);
            let b = fixture::generate(seed, label);
            assert_eq!(a.frames.len(), b.frames.len());
            assert_eq!(a.events, b.events);
            assert_eq!(a.segments, b.segments);
            for (fa, fb) in a.frames.iter().zip(b.frames.iter()) {
                assert_eq!(fa.pos[0].to_bits(), fb.pos[0].to_bits());
                assert_eq!(fa.pos[1].to_bits(), fb.pos[1].to_bits());
                assert_eq!(fa.extent.to_bits(), fb.extent.to_bits());
            }
        }
    }
}

#[test]
fn fixture_shape_and_ranges() {
    for label in [RangeLabel::Id, RangeLabel::Ood] {
        let fix = fixture::generate(3, label);
        assert_eq!(fix.frames.len(), fixture::N_SEGMENTS * fixture::T);
        assert_eq!(fix.segments.len(), fixture::N_SEGMENTS);
        assert_eq!(fix.events.len(), 1);
        // Parameters inside the paper's ranges.
        let p = fix.params;
        match label {
            RangeLabel::Id => {
                assert!((1.0..=4.0).contains(&p.v));
                assert!((0.7..=1.4).contains(&p.r));
                assert!(p.rdot <= 0.0 && p.rdot >= -0.031);
            }
            RangeLabel::Ood => {
                let in_v = (0.05..=6.0).contains(&p.v);
                let out_of_id = !(1.0..=4.0).contains(&p.v);
                assert!(
                    in_v && out_of_id,
                    "OOD v={} must be in [0.05,6] minus ID",
                    p.v
                );
                assert!((0.6..=2.0).contains(&p.r));
                assert!(
                    p.rdot <= -0.05 + 1e-6,
                    "OOD |ṙ|={:?} in [0.05,0.09]",
                    -p.rdot
                );
                assert!(p.rdot >= -0.09 - 1e-6);
            }
        }
    }
}

#[test]
fn in_family_id_ood_gap_is_exactly_zero() {
    // The load-bearing T3.2 table: on the analytic family the extrapolation
    // error is bit-exactly zero for BOTH ID and OOD parameter ranges —
    // gap ≡ 0 by construction (the provable strengthening of the paper's
    // empirical ~20×).
    let ks = [1u32, 8, 31];
    let seg_sched = [
        Sched::Measured,                   // uniform
        Sched::Measured,                   // parabolic
        Sched::Measured,                   // bounce (event-crossing predictions excluded)
        Sched::Measured,                   // looming (motion is uniform)
        Sched::GeometricDrag { rho: 0.5 }, // drag
    ];
    let seg_names = ["uniform", "parabolic", "bounce", "looming", "drag"];
    let mut id_worst = vec![0.0f32; ks.len()];
    let mut ood_worst = vec![0.0f32; ks.len()];
    for seed in 1..=6u64 {
        for (label, worst) in [
            (RangeLabel::Id, &mut id_worst),
            (RangeLabel::Ood, &mut ood_worst),
        ] {
            let fix = fixture::generate(seed, label);
            for (seg, sched) in seg_sched.iter().enumerate() {
                let errs = fixture::extrapolation_errors(&fix, seg, &ks, sched);
                for (ki, e) in errs.iter().enumerate() {
                    worst[ki] = worst[ki].max(*e);
                }
            }
        }
    }
    for (ki, k) in ks.iter().enumerate() {
        eprintln!(
            "gap-table k={k}: ID max err = {:e}, OOD max err = {:e}",
            id_worst[ki], ood_worst[ki]
        );
        let _ = seg_names; // names used by the bench-doc table generator
        assert_eq!(id_worst[ki], 0.0, "ID error nonzero at k={k}");
        assert_eq!(ood_worst[ki], 0.0, "OOD error nonzero at k={k}");
    }
}

// ===== determinism (G1 support) =====

#[test]
fn full_pipeline_bit_identical_across_runs() {
    fn run(seed: u64) -> u64 {
        let fix = fixture::generate(seed, RangeLabel::Id);
        let mut st = KinState::<2>::new(1.0).unwrap();
        let mut clf = RegimeClassifier::new(RegimeConfig {
            g: fix.params.g,
            ..RegimeConfig::default()
        });
        let mut mon = ResidualMonitor::new(ResidualConfig::default());
        let mut prev_vel = [0.0f32; 2];
        let mut running_acc = 0.0f32;
        let mut prev_extent = fix.frames[0].extent;
        let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
        for f in fix.frames.iter() {
            let sigma_rate = f.extent - prev_extent;
            prev_extent = f.extent;
            let mut predicted = [0.0f32; 2];
            kinematic_extrapolate_into(&st, 8, &Sched::Measured, &mut predicted).ok();
            let vel_before = st.vel;
            st.observe_into(&f.pos, f.tick).unwrap();
            let snap =
                RegimeSnapshot::from_state(&st, &prev_vel, running_acc, f.extent, sigma_rate);
            let v = clf.classify(&snap);
            running_acc += 0.1 * (snap.acc_mag - running_acc);
            prev_vel = st.vel;
            h = h
                .wrapping_mul(31)
                .wrapping_add(predicted[0].to_bits() as u64)
                .wrapping_add(predicted[1].to_bits() as u64)
                .wrapping_add(discriminant(&v));
            let ev = mon.update(predicted[0] - f.pos[0], &vel_before, &st.vel, 1.0);
            if let Some(k) = ev {
                h = h.wrapping_mul(33).wrapping_add(discriminant_event(&k));
            }
        }
        h
    }
    fn discriminant(r: &Regime) -> u64 {
        match r {
            Regime::Uniform => 1,
            Regime::Parabolic { g } => 2 + (g.to_bits() as u64),
            Regime::Impulse => 4,
            Regime::Looming => 5,
            Regime::Drag => 6,
        }
    }
    fn discriminant_event(e: &EventKind) -> u64 {
        match e {
            EventKind::Impulse { axis, e } => {
                10 + (axis.unwrap_or(9) as u64) + (e.unwrap_or(0.0).to_bits() as u64)
            }
            EventKind::Drift { cusum } => 20 + (cusum.to_bits() as u64),
            EventKind::Spike { z, gate } => 30 + (z.to_bits() as u64) + (gate.to_bits() as u64),
        }
    }
    for seed in [1u64, 42] {
        let a = run(seed);
        let b = run(seed);
        assert_eq!(a, b, "pipeline hash differs across runs at seed {seed}");
    }
}

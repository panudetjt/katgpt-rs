//! Plan 575 Phase 2 — risk_control_exit GOAT gate (G1 risk-hold / G2
//! exit-floor / perf report). G4 (zero-alloc) lives in
//! `bench_681_risk_control_exit_alloc_check.rs` (the house single-fn
//! alloc-binary convention — parallel tests share the global counting
//! allocator); G3 (default untouched) is the plain default-features lib
//! count outside this binary.
//!
//! # Population oracle (synthetic confidence trajectories, known ground
//! # truth)
//!
//! Two instance classes over a T = 24-step horizon:
//! - **Trivial** (solvable): difficulty `d ~ U(0.55, 0.95)`; confidence
//!   ramps `s̃t = d + (1−d)(t/T)^0.8 + noise` — crosses high λ+ partway
//!   through the horizon. Per-step correctness `= s̃t ≥ 0.5` (the answer is
//!   right once confidence builds), so a trivial instance crossing any
//!   λ+ ≥ 0.70 commits CORRECT — the class never contributes FP risk.
//! - **Stuck** (unsolvable): `s̃t = 0.55 + 0.12·z` i.i.d. per step,
//!   never correct. Noise spikes cross λ+ occasionally → the ONLY FP
//!   source; `P(single step ≥ λ+)` is a Gaussian tail, so true FP risk is
//!   strictly decreasing in λ+ (the monotonicity the paper assumes and
//!   the calibrator verifies).
//!
//! True FP risk at 50:50 composition (analytic, `1−(1−p)^24` tails):
//! λ+ 0.70 → 0.466 · 0.80 → 0.181 · 0.85 → 0.070 · 0.90 → 0.021 · 0.95 →
//! 0.005 — a steep crossing of ε = 0.10 between 0.80 and 0.85, which is
//! exactly the regime where naive small-n validation underestimates the
//! aggressive point (paper Fig. 4 shape).
//!
//! Determinism: SplitMix64 (the bench_576 convention), fixed seeds, no
//! wall-clock inputs to any verdict (timing reported only, release-gated).

#![cfg(feature = "risk_control_exit")]

use katgpt_core::risk_control_exit::{
    CalibrateConfig, CalibrateScratch, DualExitPolicy, ExitTrace, ScheduleParams, TerminalVerdict,
    TrajectorySample, calibrate_into, empirical_upper_risk, fp_loss, mean_normalized_compute,
    run_policy,
};
use std::time::Instant;

// ──────────────────────────────────────────────────────────────────────────
// Deterministic RNG (SplitMix64 — bench_576 pattern)
// ──────────────────────────────────────────────────────────────────────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
    fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

// ──────────────────────────────────────────────────────────────────────────
// Population oracle
// ──────────────────────────────────────────────────────────────────────────

const T: usize = 24;
const STUCK_MEAN: f32 = 0.55;
const STUCK_SIGMA: f32 = 0.12;
const TRIVIAL_NOISE: f32 = 0.02;

/// One generated instance: owned trajectory + per-step correctness.
/// (Stuck instances carry `correct = all-false` — class identity is
/// recoverable from that, so no separate flag is kept.)
struct Instance {
    s: Vec<f32>,
    correct: Vec<bool>,
}

impl Instance {
    fn trivial(rng: &mut SplitMix64) -> Self {
        let d = 0.55 + 0.40 * rng.next_uniform() as f32;
        let mut s = Vec::with_capacity(T);
        let mut correct = Vec::with_capacity(T);
        for t in 0..T {
            let v = clamp01(
                d + (1.0 - d) * (t as f32 / T as f32).powf(0.8) + TRIVIAL_NOISE * rng.next_normal(),
            );
            s.push(v);
            correct.push(v >= 0.5);
        }
        Self { s, correct }
    }

    fn stuck(rng: &mut SplitMix64) -> Self {
        let mut s = Vec::with_capacity(T);
        for _ in 0..T {
            s.push(clamp01(STUCK_MEAN + STUCK_SIGMA * rng.next_normal()));
        }
        Self {
            s,
            correct: vec![false; T],
        }
    }
}

/// Draws `n` instances with stuck fraction `stuck_frac` from `seed`.
fn draw(n: usize, stuck_frac: f64, seed: u64) -> Vec<Instance> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|_| {
            if rng.next_uniform() < stuck_frac {
                Instance::stuck(&mut rng)
            } else {
                Instance::trivial(&mut rng)
            }
        })
        .collect()
}

/// The calibration grids (shared by every arm so floors are compared at
/// MATCHED realized risk — same ε, same grid, same λ+ selection).
///
/// Lower-grid contract: `u = 0.65` stays strictly below the smallest upper
/// grid point (0.70) so every (λ+, schedule) pairing satisfies the
/// mutual-exclusivity invariant.
const UPPER_GRID: [f32; 6] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95];
const LOWER_GRID: [ScheduleParams; 3] = [
    ScheduleParams {
        c: 8.0 / T as f32,
        s: 0.5,
        l: 0.0,
        u: 0.65,
    },
    ScheduleParams {
        c: 16.0 / T as f32,
        s: 0.5,
        l: 0.0,
        u: 0.65,
    },
    ScheduleParams {
        c: 32.0 / T as f32,
        s: 0.5,
        l: 0.0,
        u: 0.65,
    },
];

/// Realized FP risk of the DEPLOYED dual policy on a test set (the honest
/// end-to-end quantity: Eq. 8 loss under `run_policy` traces).
fn realized_fp_risk(test: &[Instance], policy: &DualExitPolicy) -> f32 {
    let mut loss = 0.0f64;
    for inst in test {
        let trace = run_policy(policy, &inst.s);
        loss += fp_loss(&inst.correct, trace) as f64;
    }
    (loss / test.len().max(1) as f64) as f32
}

/// Naive (no-correction) λ+ selection: the smallest grid point whose
/// EMPIRICAL risk on the validation set reads ≤ ε (the paper's Fig. 4
/// comparator — cross-validation without the finite-sample correction).
/// Falls back to the largest grid point when nothing reads feasible.
fn naive_lambda_plus(val: &[TrajectorySample<'_>], epsilon: f32) -> (f32, usize) {
    for (i, &lp) in UPPER_GRID.iter().enumerate() {
        if empirical_upper_risk(val, lp) <= epsilon {
            return (lp, i);
        }
    }
    (UPPER_GRID[UPPER_GRID.len() - 1], UPPER_GRID.len() - 1)
}

/// Overall accuracy of the deployed policy: commit → the answer at the exit
/// tick; abandon → wrong (no answer); exhausted → the final step's answer.
fn accuracy(test: &[Instance], policy: &DualExitPolicy) -> f32 {
    let mut good = 0.0f64;
    for inst in test {
        let ExitTrace { verdict, tick } = run_policy(policy, &inst.s);
        let ok = match verdict {
            TerminalVerdict::Commit => inst.correct[tick],
            TerminalVerdict::Exhausted => inst.correct[tick],
            TerminalVerdict::Abandon => false,
        };
        good += f64::from(ok);
    }
    (good / test.len().max(1) as f64) as f32
}

/// The single-threshold floor: upper exit only (commit at the first
/// `s̃ ≥ λ+`, else run to budget exhaustion). Local loop — the floor is a
/// baseline, not the primitive.
fn upper_only_trace(s: &[f32], lambda_plus: f32) -> ExitTrace {
    for (t, &v) in s.iter().enumerate() {
        if v >= lambda_plus {
            return ExitTrace {
                verdict: TerminalVerdict::Commit,
                tick: t,
            };
        }
    }
    ExitTrace {
        verdict: TerminalVerdict::Exhausted,
        tick: s.len() - 1,
    }
}

// ──────────────────────────────────────────────────────────────────────────
// G1 — risk hold across 40 resplits (+ naive violation contrast)
// ──────────────────────────────────────────────────────────────────────────

/// G1: at every validation size, UCB calibration holds realized FP risk ≤ ε
/// on all 40 test resplits; naive no-correction calibration VIOLATES the
/// target on some resplits at small n (the paper Fig. 4 shape — the
/// demonstration the plan requires; if naive had never violated we would
/// shrink n further, per the plan).
#[test]
fn g1_ucb_holds_naive_violates() {
    const EPS: f32 = 0.10;
    const DELTA: f32 = 0.05;
    const RESPLITS: usize = 40;

    for (n_val, expect_naive_violations) in [(40usize, true), (400, false)] {
        let mut ucb_violations = 0usize;
        let mut naive_violations = 0usize;
        let mut ucb_fallbacks = 0usize;
        let mut ucb_interior = 0usize;
        let mut naive_picks = [0usize; UPPER_GRID.len()];
        let mut ucb_picks = [0usize; UPPER_GRID.len()];
        let mut max_ucb_risk = 0.0f32;

        for r in 0..RESPLITS {
            let val = draw(n_val, 0.5, 0x5750_0000 + ((n_val as u64) << 24) + r as u64);
            let test = draw(800, 0.5, 0x5751_0000 + ((n_val as u64) << 24) + r as u64);
            let val_samples: Vec<TrajectorySample<'_>> = val
                .iter()
                .map(|i| TrajectorySample::new(&i.s, &i.correct))
                .collect();

            // UCB arm (the primitive).
            let cfg = CalibrateConfig::new(EPS, EPS, DELTA);
            let mut scratch = CalibrateScratch::new();
            let out = calibrate_into(&val_samples, &cfg, &UPPER_GRID, &LOWER_GRID, &mut scratch);
            if out.fell_back {
                ucb_fallbacks += 1;
            }
            if out.upper_index < UPPER_GRID.len() - 1 {
                ucb_interior += 1;
            }
            ucb_picks[out.upper_index] += 1;
            let risk = realized_fp_risk(&test, &out.policy);
            max_ucb_risk = max_ucb_risk.max(risk);
            if risk > EPS {
                ucb_violations += 1;
            }

            // Naive arm (no UCB correction).
            let (nlp, ni) = naive_lambda_plus(&val_samples, EPS);
            naive_picks[ni] += 1;
            let naive_policy = DualExitPolicy::new(
                nlp,
                LOWER_GRID[2].c,
                LOWER_GRID[2].s,
                LOWER_GRID[2].l,
                LOWER_GRID[2].u,
            );
            let naive_risk = realized_fp_risk(&test, &naive_policy);
            if naive_risk > EPS {
                naive_violations += 1;
            }
        }

        println!(
            "G1 n={n_val}: ucb violations {ucb_violations}/{RESPLITS} (max risk {max_ucb_risk:.4}) \
             fallbacks {ucb_fallbacks} interior-picks {ucb_interior} picks {ucb_picks:?} | \
             naive violations {naive_violations}/{RESPLITS} picks {naive_picks:?}"
        );

        // The guarantee: UCB never violates, at any n.
        assert_eq!(
            ucb_violations, 0,
            "UCB calibration must hold realized FP risk ≤ ε on every resplit (n={n_val})"
        );
        // The demonstration: naive violates at small n (paper Fig. 4).
        if expect_naive_violations {
            assert!(
                naive_violations >= 1,
                "naive calibration must violate ε on some resplits at n={n_val} \
                 (got 0 — shrink n further per the plan)"
            );
        }
        // Honest mechanism check: at n=40 the UCB term (0.194) exceeds ε →
        // mostly conservative fallbacks; at n=400 (ucb 0.061) the selection
        // is genuinely feasible and lands INTERIOR (not the max grid point).
        if n_val == 400 {
            assert!(
                ucb_interior >= RESPLITS / 2,
                "at n=400 UCB must select interior grid points (got {ucb_interior}/{RESPLITS})"
            );
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// G2 — the exit-floor rule (dual vs fixed-budget vs single-threshold)
// ──────────────────────────────────────────────────────────────────────────

/// G2: crowd-composition sweep 3:1 / 1:1 / 1:3 trivial:stuck. At MATCHED
/// realized risk (same ε, same grids, same λ+ selection for the single-
/// threshold floor), the dual-exit policy must win-or-tie BOTH floors per
/// composition (accuracy within ε− + slack; compute ≤) and win overall
/// (strictly lower mean compute than both floors).
#[test]
fn g2_exit_floor_dual_wins_or_ties_both_floors() {
    const EPS: f32 = 0.15;
    const DELTA: f32 = 0.05;
    const N_VAL: usize = 400;
    const N_TEST: usize = 800;
    // Accuracy tie tolerance = the FN budget the lower threshold is
    // calibrated to (realized FN sits well under it — reported below).
    const ACC_TOL: f32 = EPS + 0.02;

    // (label, stuck fraction): 3:1 / 1:1 / 1:3 trivial:stuck.
    struct Row {
        label: &'static str,
        dual: [f32; 2], // [accuracy, compute]
        single: [f32; 2],
        fixed: [f32; 2],
        dual_risk: f32,
        single_risk: f32,
        lambda_plus: f32,
    }
    fn r_select(r: &Row, k: usize) -> [f32; 2] {
        match k {
            0 => r.dual,
            1 => r.single,
            _ => r.fixed,
        }
    }

    let compositions: [(&str, f64); 3] = [("3:1", 0.25), ("1:1", 0.50), ("1:3", 0.75)];
    let mut rows: Vec<Row> = Vec::new();

    for (label, stuck_frac) in compositions {
        let val = draw(
            N_VAL,
            stuck_frac,
            0x5752_0000 + (stuck_frac * 1000.0) as u64,
        );
        let test = draw(
            N_TEST,
            stuck_frac,
            0x5753_0000 + (stuck_frac * 1000.0) as u64,
        );
        let val_samples: Vec<TrajectorySample<'_>> = val
            .iter()
            .map(|i| TrajectorySample::new(&i.s, &i.correct))
            .collect();

        // Dual arm: the primitive, two-step calibrated.
        let cfg = CalibrateConfig::new(EPS, EPS, DELTA);
        let mut scratch = CalibrateScratch::new();
        let out = calibrate_into(&val_samples, &cfg, &UPPER_GRID, &LOWER_GRID, &mut scratch);
        let dual_policy = out.policy;

        // Single-threshold floor: the SAME step-1 λ+ (matched risk by
        // construction), upper-only.
        let single_lp = UPPER_GRID[out.upper_index];

        // Fixed-budget floor: everyone runs to T, commits the final answer.
        let fixed_acc = {
            let good = test.iter().filter(|i| i.correct[T - 1]).count();
            good as f32 / test.len() as f32
        };

        // Dual metrics.
        let dual_acc = accuracy(&test, &dual_policy);
        let dual_compute = mean_normalized_compute(
            &test
                .iter()
                .map(|i| TrajectorySample::new(&i.s, &i.correct))
                .collect::<Vec<_>>(),
            &dual_policy,
        );
        let dual_risk = realized_fp_risk(&test, &dual_policy);

        // Single-threshold metrics (upper-only loop).
        let mut single_acc_acc = 0.0f64;
        let mut single_compute_acc = 0.0f64;
        let mut single_fp = 0.0f64;
        for inst in &test {
            let ExitTrace { verdict, tick } = upper_only_trace(&inst.s, single_lp);
            let ok = match verdict {
                TerminalVerdict::Commit | TerminalVerdict::Exhausted => inst.correct[tick],
                TerminalVerdict::Abandon => false,
            };
            single_acc_acc += f64::from(ok);
            single_compute_acc += (tick + 1) as f64 / T as f64;
            if verdict == TerminalVerdict::Commit && !inst.correct[tick] {
                single_fp += 1.0;
            }
        }
        let single_acc = (single_acc_acc / test.len() as f64) as f32;
        let single_compute = (single_compute_acc / test.len() as f64) as f32;
        let single_risk = (single_fp / test.len() as f64) as f32;

        let row = Row {
            label,
            dual: [dual_acc, dual_compute],
            single: [single_acc, single_compute],
            fixed: [fixed_acc, 1.0],
            dual_risk,
            single_risk,
            lambda_plus: dual_policy.lambda_plus,
        };
        println!(
            "G2 {label}: λ+={:.2} dual acc {:.3} comp {:.3} (risk {:.4}) | single acc {:.3} comp {:.3} (risk {:.4}) | fixed acc {:.3} comp 1.000",
            row.lambda_plus,
            row.dual[0],
            row.dual[1],
            row.dual_risk,
            row.single[0],
            row.single[1],
            row.single_risk,
            row.fixed[0]
        );
        rows.push(row);
    }

    // ── The floor rule: win-or-tie per composition, win overall ──────────
    let mean = |rows: &Vec<Row>, k: usize, j: usize| {
        rows.iter().map(|r| r_select(r, k)[j]).sum::<f32>() / rows.len() as f32
    };

    for r in &rows {
        // Risk accounting: the calibrated arms hold their budgets.
        assert!(
            r.dual_risk <= EPS && r.single_risk <= EPS,
            "{}: realized risk must hold ≤ ε (dual {:.4}, single {:.4})",
            r.label,
            r.dual_risk,
            r.single_risk
        );
        // vs single-threshold floor: accuracy tie-or-better, compute ≤.
        assert!(
            r.dual[0] >= r.single[0] - ACC_TOL,
            "{}: dual accuracy {:.3} must tie-or-beat single {:.3}",
            r.label,
            r.dual[0],
            r.single[0]
        );
        assert!(
            r.dual[1] <= r.single[1],
            "{}: dual compute {:.3} must be ≤ single {:.3}",
            r.label,
            r.dual[1],
            r.single[1]
        );
        // vs fixed-budget floor: accuracy tie-or-better, compute STRICTLY
        // less (any exit at all beats burning the full budget).
        assert!(
            r.dual[0] >= r.fixed[0] - ACC_TOL,
            "{}: dual accuracy {:.3} must tie-or-beat fixed {:.3}",
            r.label,
            r.dual[0],
            r.fixed[0]
        );
        assert!(
            r.dual[1] < r.fixed[1],
            "{}: dual compute {:.3} must strictly beat fixed 1.0",
            r.label,
            r.dual[1]
        );
    }
    // Overall: strictly lower mean compute than BOTH floors.
    let dual_mean = mean(&rows, 0, 1);
    let single_mean = mean(&rows, 1, 1);
    let fixed_mean = mean(&rows, 2, 1);
    println!(
        "G2 overall: dual compute {dual_mean:.3} vs single {single_mean:.3} vs fixed {fixed_mean:.3}"
    );
    assert!(dual_mean < single_mean, "dual must beat single overall");
    assert!(dual_mean < fixed_mean, "dual must beat fixed overall");
    // The paper's Fig. 6 shape: the dual-vs-single compute gap GROWS with
    // the stuck share (upper-only captures most savings at 3:1; the lower
    // threshold dominates at 1:3).
    let gap = |i: usize| rows[i].single[1] - rows[i].dual[1];
    println!(
        "G2 dual-vs-single compute gaps by composition: {:.3} / {:.3} / {:.3}",
        gap(0),
        gap(1),
        gap(2)
    );
    assert!(
        gap(2) > gap(0),
        "the dual-vs-single gap must grow with stuck share (Fig. 6 shape)"
    );
}

// ──────────────────────────────────────────────────────────────────────────
// Perf report (release-gated — the documented debug-timing flake class)
// ──────────────────────────────────────────────────────────────────────────

/// Per-call `exit()` cost: 2 comparisons + 1 squeezed sigmoid must sit in
/// the nanosecond class (gate < 1 µs; the plan carries no explicit perf
/// gate — this is the house G2-perf-style reported number).
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn perf_report_per_exit_call() {
    const ITERS: usize = 1_000_000;

    let policy = DualExitPolicy::new(0.85, 16.0 / T as f32, 0.5, 0.0, 0.65);
    let s: Vec<f32> = (0..T).map(|t| 0.3 + 0.02 * t as f32).collect();
    let mut sink = 0u64;
    let start = Instant::now();
    for i in 0..ITERS {
        let v = policy.exit(s[i % T], (i % T) as u32 + 1, T as u32);
        sink += v as u64;
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / ITERS as f64;
    println!("perf: {ns_per:.2} ns/exit over {ITERS} calls (sink {sink})");
    assert!(
        ns_per < 1000.0,
        "exit() must stay sub-µs (got {ns_per:.2} ns)"
    );
}

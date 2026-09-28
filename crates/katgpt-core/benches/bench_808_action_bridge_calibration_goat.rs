//! Bench 808 — ActionBridge confidence-calibration GOAT (riir-ai Issue 964 C2).
//!
//! The second `sigmoid_calibration` consumer (substrate: Issue 810; first
//! consumer: `clr_calibration`, Bench 807). `ActionBridge`'s
//! `sigmoid_confidence` gates the ABSTAIN threshold — but a threshold on an
//! UNCALIBRATED confidence is a threshold on a number whose meaning is proven
//! nowhere. `CalibratedActionBridge` observes
//! `(raw confidence, action_succeeded)` pairs and refits, making the ABSTAIN
//! threshold a statement about outcome probability.
//!
//! Gates (all on a decision-level corpus with a planted overconfidence the
//! bridge does not know about, train/test split):
//!
//! - **G1** decision-level ECE improves (cal < raw, cal ≤ 0.05) with
//!   planted-transform recovery.
//! - **G2** Report-the-Floor: calibrated log-loss + Brier beat BOTH the
//!   uncalibrated confidence AND the base-rate floor.
//! - **G3** (a) cold start bit-identical — selections AND ABSTAIN decisions;
//!   (b) the argmax invariant after a REAL refit — the winner never changes
//!   (one strictly monotone transform on a shared score scale);
//!   (c) fire-rate movement: the ABSTAIN operating point on the calibrated
//!   confidence is closer to the ORACLE abstain set (true-p < threshold)
//!   than the raw one.
//! - **G4** steady-state observe + select_action_calibrated adds zero heap
//!   allocations (Issue-741 predicate).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/bench808 cargo bench -p katgpt-core \
//!   --features sigmoid_calibration --no-default-features \
//!   --bench bench_808_action_bridge_calibration_goat -- --nocapture
//! ```

#![cfg(feature = "sigmoid_calibration")]

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::ActionBridge;
use katgpt_core::bridge::calibrated::CalibratedActionBridge;
use katgpt_core::sigmoid_calibration::{brier_score, expected_calibration_error, log_loss};
use std::hint::black_box;

/// Deterministic SplitMix64 (the workspace-bench idiom — no rand dep).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f32 {
        // Top 24 bits / 2^24 — uniform in [0, 1).
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn next_sym(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

// A=4 actions, D=3 latent dim (exercises a non-trivial argmax).
const N_ACTIONS: usize = 4;
const LATENT_D: usize = 3;
const TRAIN_N: usize = 4096;
const TEST_N: usize = 4096;
const ECE_BINS: usize = 10;
/// The ABSTAIN threshold under test (the pipeline's Step-8 gate).
const THRESHOLD: f32 = 0.75;

/// Planted overconfidence (the C2 analog of Issue 810's planted transform):
/// a decision whose raw confidence is `p` truly succeeds with probability
/// `sigmoid(1.4·logit(p) − 0.3)` — the bridge is overconfident everywhere.
const PLANTED_A: f32 = 1.4;
const PLANTED_B: f32 = -0.3;

fn planted_true_p(raw_conf: f32) -> f32 {
    sigmoid(PLANTED_A.mul_add(logit(raw_conf), PLANTED_B))
}

/// One decision draw: latent q → the bridge's (action, raw confidence).
fn make_bridge() -> ActionBridge<N_ACTIONS, LATENT_D> {
    ActionBridge::new([[1, 0, -1], [0, 1, 1], [-1, -1, 0], [1, 1, 1]], THRESHOLD)
}

/// Build `n` decisions: q-values in [-3, 3]^D with varied scale so raw
/// confidences span the full (0, 1) range and exercise all ECE bins.
fn build_decisions(seed: u64, n: usize) -> Vec<[f32; LATENT_D]> {
    let mut rng = SplitMix64(seed);
    (0..n)
        .map(|_| {
            let scale = 0.75 + rng.next_unit() * 2.0;
            [
                rng.next_sym() * scale,
                rng.next_sym() * scale,
                rng.next_sym() * scale,
            ]
        })
        .collect()
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Bench 808 — ActionBridge confidence-calibration GOAT");
    println!("  (riir-ai Issue 964 C2; planted overconfidence 1.4·logit − 0.3)");
    println!("═══════════════════════════════════════════════════════════════");

    let train_qs = build_decisions(0xC2A1_0001, TRAIN_N);
    let test_qs = build_decisions(0xC2A1_0002, TEST_N);

    // ── Fit on the train split ───────────────────────────────────────────
    let mut calibrated = CalibratedActionBridge::new(make_bridge(), TRAIN_N, TRAIN_N / 4);
    let mut fit_rng = SplitMix64(0xC2A1_0003);
    for q in &train_qs {
        let (_, raw_conf) = calibrated.inner().select_action(q);
        let succeeded = fit_rng.next_unit() < planted_true_p(raw_conf);
        calibrated.observe(raw_conf, succeeded);
    }
    assert!(
        calibrated.refit(),
        "planted fixture must move the parameters"
    );
    let (t, b) = calibrated.params();

    // ── Evaluate on the held-out test split ──────────────────────────────
    let mut eval_rng = SplitMix64(0xC2A1_0004);
    let mut raw_ps = Vec::with_capacity(TEST_N);
    let mut cal_ps = Vec::with_capacity(TEST_N);
    let mut ys = Vec::with_capacity(TEST_N);
    // G3(b) argmax + G3(c) abstain-rate bookkeeping.
    let mut winner_mismatches = 0usize;
    let mut raw_abstain = 0usize;
    let mut cal_abstain = 0usize;
    let mut oracle_abstain = 0usize;
    for q in &test_qs {
        let (raw_idx, raw_conf) = calibrated.inner().select_action(q);
        let (cal_idx, cal_conf) = calibrated.select_action_calibrated(q);
        if cal_idx != raw_idx {
            winner_mismatches += 1;
        }
        let true_p = planted_true_p(raw_conf);
        let succeeded = eval_rng.next_unit() < true_p;

        raw_abstain += usize::from(raw_conf < THRESHOLD);
        cal_abstain += usize::from(calibrated.should_abstain(cal_conf));
        oracle_abstain += usize::from(true_p < THRESHOLD);

        raw_ps.push(raw_conf);
        cal_ps.push(cal_conf);
        ys.push(f32::from(succeeded));
    }

    let ece_raw = expected_calibration_error(&raw_ps, &ys, ECE_BINS);
    let ece_cal = expected_calibration_error(&cal_ps, &ys, ECE_BINS);
    let ll_raw = log_loss(&raw_ps, &ys);
    let ll_cal = log_loss(&cal_ps, &ys);
    let br_raw = brier_score(&raw_ps, &ys);
    let br_cal = brier_score(&cal_ps, &ys);
    let base_rate = ys.iter().sum::<f32>() / ys.len() as f32;
    let floor_ps = vec![base_rate; TEST_N];
    let ll_floor = log_loss(&floor_ps, &ys);
    let br_floor = brier_score(&floor_ps, &ys);

    let raw_rate = raw_abstain as f32 / TEST_N as f32;
    let cal_rate = cal_abstain as f32 / TEST_N as f32;
    let oracle_rate = oracle_abstain as f32 / TEST_N as f32;

    println!();
    println!("G1 — decision-level calibration (test split n={TEST_N}, {ECE_BINS} bins)");
    println!(
        "  recovered (T, b)   = ({t:.3}, {b:.3})   planted = ({:.3}, {:.3})",
        1.0 / PLANTED_A,
        -PLANTED_B / PLANTED_A
    );
    println!("  ECE     raw → cal  = {ece_raw:.4} → {ece_cal:.4}");
    println!("  logloss raw → cal  = {ll_raw:.4} → {ll_cal:.4}   (base-rate floor {ll_floor:.4})");
    println!("  brier   raw → cal  = {br_raw:.4} → {br_cal:.4}   (base-rate floor {br_floor:.4})");

    let g1 = ece_cal <= 0.05 && ece_cal < ece_raw;
    let (t_planted, b_planted) = (1.0_f32 / PLANTED_A, -PLANTED_B / PLANTED_A);
    let g1_recovery = (t - t_planted).abs() < 0.05 && (b - b_planted).abs() < 0.05;
    let g2 = ll_cal < ll_raw && ll_cal < ll_floor && br_cal < br_raw && br_cal < br_floor;

    println!();
    println!("G3 — no regression");
    println!(
        "  (b) argmax invariant after real refit : {winner_mismatches} winner mismatches (target 0)"
    );
    println!(
        "  (c) ABSTAIN operating point (τ = {THRESHOLD}): raw {raw_rate:.4} → cal {cal_rate:.4} (oracle {oracle_rate:.4})"
    );
    let g3b = winner_mismatches == 0;
    // Fire-rate movement (the katgpt-rs G3 shape): the calibrated operating
    // point moves TOWARD the oracle abstain set, never away.
    let g3c = (cal_rate - oracle_rate).abs() <= (raw_rate - oracle_rate).abs();

    // G3(a) cold start — bit-identical selections AND abstain decisions.
    let cold = CalibratedActionBridge::new(make_bridge(), 64, 16);
    let mut cold_bit_ok = true;
    let mut cold_abstain_ok = true;
    for q in &test_qs {
        let (_, raw_conf) = cold.inner().select_action(q);
        let (cold_idx, cold_conf) = cold.select_action_calibrated(q);
        let (raw_idx, _) = cold.inner().select_action(q);
        if cold_idx != raw_idx || cold_conf.to_bits() != raw_conf.to_bits() {
            cold_bit_ok = false;
        }
        if cold.should_abstain(cold_conf) != (raw_conf < THRESHOLD) {
            cold_abstain_ok = false;
        }
    }
    println!(
        "  (a) cold-start bit-identity           : confidences {} / abstain decisions {}",
        if cold_bit_ok { "exact" } else { "DRIFTED" },
        if cold_abstain_ok { "exact" } else { "DRIFTED" }
    );
    let g3a = cold_bit_ok && cold_abstain_ok;

    // ── G4 — alloc-free observe + select (Issue-741 predicate) ───────────
    // counting_allocator!() is thread-local (Issue 714) and this bench is
    // single-threaded — the counters are ours alone. The macro registers the
    // allocator UNCONDITIONALLY, so the gate runs in every profile; the
    // Issue-741 note applies to consumers of katgpt-core's own alloc module
    // (not used here).
    let g4_delta: Option<usize>;
    {
        use std::sync::atomic::Ordering;
        {
            let _probe: Vec<u8> = vec![0u8; 64];
            black_box(&_probe);
            let live = ALLOC_COUNT.load(Ordering::Relaxed) > 0;
            assert!(live, "CountingAllocator not installed — G4 vacuous");
        }
        let before = ALLOC_COUNT.load(Ordering::Relaxed);
        let mut sink = 0.0f32;
        for i in 0..TEST_N {
            calibrated.observe(raw_ps[i], ys[i] > 0.5);
            let (_, conf) = calibrated.select_action_calibrated(&test_qs[i]);
            sink += conf;
        }
        black_box(sink);
        let after = ALLOC_COUNT.load(Ordering::Relaxed);
        g4_delta = Some(after - before);
        println!();
        println!(
            "G4 — {} × (observe + select_action_calibrated) : {} allocs (target 0)",
            TEST_N,
            g4_delta.unwrap()
        );
    }

    println!();
    println!("───────────────────────────────────────────────────────────────");
    let pass = g1 && g1_recovery && g2 && g3a && g3b && g3c && g4_delta.is_none_or(|d| d == 0);
    assert!(
        g1,
        "G1 FAILED: ECE cal {ece_cal:.4} (≤0.05, < raw {ece_raw:.4})"
    );
    assert!(
        g1_recovery,
        "G1 FAILED: planted recovery ({t:.3}, {b:.3}) vs ({t_planted:.3}, {b_planted:.3})"
    );
    assert!(
        g2,
        "G2 FAILED: logloss cal {ll_cal:.4} / raw {ll_raw:.4} / floor {ll_floor:.4}; brier cal {br_cal:.4} / raw {br_raw:.4} / floor {br_floor:.4}"
    );
    assert!(g3a, "G3a FAILED: cold start drifted");
    assert!(
        g3b,
        "G3b FAILED: {winner_mismatches} winner mismatches after refit"
    );
    assert!(
        g3c,
        "G3c FAILED: abstain rate moved AWAY from oracle (raw err {:.4} → cal err {:.4})",
        (raw_rate - oracle_rate).abs(),
        (cal_rate - oracle_rate).abs()
    );
    if let Some(d) = g4_delta {
        assert!(d == 0, "G4 FAILED: {d} allocations on observe+select");
    }

    println!(
        "Bench 808 {} — G1 {} · G1-recovery {} · G2 {} · G3a {} · G3b {} · G3c {} · G4 {}",
        if pass { "PASS ✅" } else { "FAIL ❌" },
        ok(g1),
        ok(g1_recovery),
        ok(g2),
        ok(g3a),
        ok(g3b),
        ok(g3c),
        ok(g4_delta.is_none_or(|d| d == 0))
    );
}

fn ok(v: bool) -> &'static str {
    if v { "✅" } else { "❌" }
}

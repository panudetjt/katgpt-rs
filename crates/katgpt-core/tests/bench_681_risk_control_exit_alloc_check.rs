//! Plan 575 Phase 2 T2.4 — G4 zero-alloc steady state for the
//! risk_control_exit primitive. Separate single-fn binary (the house
//! convention: parallel tests share the global counting allocator —
//! `bench_576_hint_regret_alloc_check.rs` pattern).

#![cfg(feature = "risk_control_exit")]

use katgpt_core::risk_control_exit::{
    CalibrateConfig, CalibrateScratch, DualExitPolicy, ExitVerdict, ScheduleParams,
    TrajectorySample, calibrate_into, run_policy,
};

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// G4: zero allocations in the per-call exit hot path (10⁵ `exit()` calls
/// over a pre-built policy) AND zero allocations in the calibration steady
/// state (200 `calibrate_into` passes over pre-built samples with
/// pre-sized, reused scratch). One function so the checks run serially
/// against the shared allocator.
#[test]
fn g4_zero_alloc_exit_and_calibration_steady_state() {
    // ── Per-call exit hot path ───────────────────────────────────────────
    {
        let policy = DualExitPolicy::new(0.85, 16.0 / 24.0, 0.5, 0.0, 0.65);
        let s: Vec<f32> = (0..24).map(|t| 0.3 + 0.02 * t as f32).collect();
        let (sink, delta) = alloc_delta(|| {
            let mut acc = 0u64;
            for i in 0..100_000u32 {
                let v = policy.exit(s[(i % 24) as usize], i % 24 + 1, 24);
                acc += matches!(v, ExitVerdict::Commit) as u64;
                acc += matches!(v, ExitVerdict::Abandon) as u64;
            }
            acc
        });
        assert_eq!(delta, 0, "per-call exit must be alloc-free (got {delta})");
        assert!(sink < 100_000);
    }

    // ── Calibration steady state (scratch reused) ────────────────────────
    {
        // Setup (allocations allowed): 24 owned trajectories + samples.
        let owned: Vec<(Vec<f32>, Vec<bool>)> = (0..24)
            .map(|i| {
                let s: Vec<f32> = (0..24)
                    .map(|t| (0.35 + 0.025 * (t + i % 3) as f32).min(0.97))
                    .collect();
                let c: Vec<bool> = (0..24).map(|t| t >= 14 && i % 2 == 0).collect();
                (s, c)
            })
            .collect();
        let samples: Vec<TrajectorySample<'_>> = owned
            .iter()
            .map(|(s, c)| TrajectorySample::new(s, c))
            .collect();
        let upper = [0.70f32, 0.75, 0.80, 0.85, 0.90, 0.95];
        let lower = [
            ScheduleParams {
                c: 8.0 / 24.0,
                s: 0.5,
                l: 0.0,
                u: 0.65,
            },
            ScheduleParams {
                c: 16.0 / 24.0,
                s: 0.5,
                l: 0.0,
                u: 0.65,
            },
            ScheduleParams {
                c: 32.0 / 24.0,
                s: 0.5,
                l: 0.0,
                u: 0.65,
            },
        ];
        let cfg = CalibrateConfig::new(0.15, 0.15, 0.05);
        // Pre-sized scratch, warmed once (the growth alloc happens HERE,
        // outside the measured region).
        let mut scratch = CalibrateScratch::with_capacity(upper.len(), lower.len());
        let _ = calibrate_into(&samples, &cfg, &upper, &lower, &mut scratch);

        let (n_runs, delta) = alloc_delta(|| {
            let mut checksum = 0.0f32;
            for _ in 0..200 {
                let out = calibrate_into(&samples, &cfg, &upper, &lower, &mut scratch);
                checksum += out.mean_normalized_compute;
                // Exercise the trace path alongside calibration.
                checksum += run_policy(&out.policy, &owned[0].0).tick as f32;
            }
            checksum
        });
        assert_eq!(
            delta, 0,
            "calibration steady state must be alloc-free with reused scratch (got {delta})"
        );
        assert!(n_runs > 0.0);
    }
}

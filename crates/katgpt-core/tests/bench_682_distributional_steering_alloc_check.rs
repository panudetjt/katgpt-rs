//! Plan 577 Phase 3 T3.5 — G4 zero-alloc steady state for the FK stepper
//! path over 1000 steps at N=1000. Separate single-fn binary (the house
//! convention: parallel tests share the global counting allocator —
//! `bench_576_hint_regret_alloc_check.rs` pattern).
//!
//! NOTE: bench number 682, not 577 — 577 was already allocated
//! (emotion_direction_rank; the monotonic numbering rule).

#![cfg(feature = "distributional_steering")]

use katgpt_core::distributional_steering::{FkStepper, MmdReward, SteeringScratch};

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// G4: zero allocations across 1000 begin/finish steps at N=1000, K_FP=3.
/// Construction (scratch + buffers) happens BEFORE the measured region —
/// the house `bench_576_hint_regret_alloc_check.rs` pattern.
///
/// Debug-ignored: at N=1000 the exact O(N²) kernel build makes the debug
/// run exceed 60 s, at which point libtest's slow-test warning itself
/// allocates (+2 on the shared global counter — measured at step 481,
/// harness noise, not module allocation). Run recorded with `--release`
/// (allocation behavior is build-independent).
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug run >60s trips libtest slow-warning alloc; run --release (Bench 682)"
)]
fn g4_zero_alloc_steady_state_fk_path() {
    let n = 1000usize;
    let dim = 1usize;
    let mut rng_state = 0x1234_5678_9abc_def0u64;
    let mut next_uniform = move || {
        rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    // Setup (allocations allowed) — outside the measured region.
    let mut states = vec![0.0f32; n * dim];
    for s in states.iter_mut() {
        *s = next_uniform() as f32;
    }
    let target: Vec<f32> = (0..256 * dim).map(|_| next_uniform() as f32).collect();
    let reward = MmdReward::new(0.1, target, dim);
    let stepper = FkStepper {
        steer_scale: 5.0,
        k_fp: 3,
        damping: 0.4,
        clip_log_delta: 1.0,
    };
    let mut scratch = SteeringScratch::new(n, dim);
    let mut log_w = vec![0.0f32; n];
    let b: Vec<f32> = vec![0.05; n * dim];
    let dt = 0.05f32;

    let (steps_run, delta_allocs) = alloc_delta(|| {
        let mut local_count = 0u32;
        for _ in 0..1000 {
            let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            stepper.begin_step(&reward, &states, &mut log_w, &mut scratch);
            let mid = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            let steer = scratch.steering();
            for i in 0..n * dim {
                states[i] += 0.01 * steer[i] + b[i] * dt;
            }
            stepper.finish_step(&reward, &states, &b, dt, &mut log_w, &mut scratch);
            let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            if after > before {
                println!(
                    "alloc at step {local_count}: begin +{}, finish +{}",
                    mid - before,
                    after - mid
                );
            }
            local_count += 1;
        }
        assert!(log_w.iter().all(|l| l.is_finite()));
        local_count
    });
    assert_eq!(steps_run, 1000);
    assert_eq!(
        delta_allocs, 0,
        "FK path steady state must be alloc-free over 1000 steps at N=1000 (got {delta_allocs})"
    );
}

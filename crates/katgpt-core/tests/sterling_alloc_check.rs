//! Issue 672 G4 — zero-allocation steady state for the sterling hot paths
//! (T1 logit mask, T2 decomposed GEMV readout, T3 bias-table write, T4 HSIC
//! gauge with caller scratch). The T2 scalar `decomposed_readout` and the
//! LiftTableBuilder are cold-path (they allocate by design — documented);
//! the *per-tick* surfaces are the `_into` variants gated here.
//!
//! Separate test binary (mirrors `switch_cost_alloc_check`) so the
//! `CountingAllocator` global picks up only this module's allocations.
//! **One `#[test]` function** — the counter is process-global.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features sterling_primitives \
//!     --test sterling_alloc_check --release -- --nocapture
//! ```

#![cfg(feature = "sterling_primitives")]

use katgpt_core::sterling::{
    LiftTable, decomposed_readout_gemv_into, hsic_cross_covariance_gauge, lift_set_to_bias_table,
    relu_gated_suppression_into, tau_over_peak_calibration,
};
use std::hint::black_box;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

#[test]
fn g4_zero_alloc_steady_state() {
    let v = 512usize;
    let d = 256usize;
    let logits: Vec<f32> = (0..v).map(|i| (i % 17) as f32 / 17.0 - 0.5).collect();
    let alignment: Vec<f32> = (0..v).map(|i| (i % 23) as f32 / 23.0 - 0.5).collect();
    let mut out = vec![0f32; v];

    let head: Vec<f32> = (0..v * d).map(|i| (i % 19) as f32 / 19.0 - 0.5).collect();
    let k1: Vec<f32> = (0..d).map(|i| (i % 13) as f32 / 13.0 - 0.5).collect();
    let k2: Vec<f32> = (0..d).map(|i| (i % 29) as f32 / 29.0 - 0.5).collect();
    let eps: Vec<f32> = vec![0.01; d];
    let mut gemv_out = vec![0f32; 4 * v];

    let lift = LiftTable {
        alpha: 0.5,
        entries: vec![(3, 2.5), (17, 0.4), (255, 1.1)],
    };
    let mut bias = vec![0f32; v];

    let m = 32usize;
    let psi: Vec<f32> = (0..m * d).map(|i| (i % 31) as f32 / 31.0 - 0.5).collect();
    let phi: Vec<f32> = (0..m * d).map(|i| (i % 37) as f32 / 37.0 - 0.5).collect();
    let mut sp = vec![0f32; m * d];
    let mut sp2 = vec![0f32; m * d];
    let dir: Vec<f32> = (0..d).map(|i| (i % 7) as f32 / 7.0).collect();

    // ── Measured: hot paths, 0 allocations ─────────────────────────────────
    let ((), allocs) = alloc_delta(|| {
        let mut acc = 0.0f32;
        for i in 0..10_000u32 {
            relu_gated_suppression_into(&logits, &alignment, (i % 9) as f32 / 9.0, &mut out);
            acc += out[(i as usize) % v];

            decomposed_readout_gemv_into(&head, d, &[&k1, &k2], &eps, &mut gemv_out);
            acc += gemv_out[(i as usize) % (4 * v)];

            lift_set_to_bias_table(&lift, 1.5, &mut bias);
            acc += bias[(i as usize) % v];

            acc += hsic_cross_covariance_gauge(&psi, &phi, m, d, &mut sp, &mut sp2);

            if let Some(g) = tau_over_peak_calibration(&head, &dir, v, 0.5) {
                acc += g;
            }
        }
        black_box(acc);
    });
    assert_eq!(allocs, 0, "steady-state sterling paths allocated {allocs}×");
}

//! Issue 655 G4 — zero-alloc steady state for
//! `propagate_selection_to_fixpoint_into` (separate test binary so the
//! `CountingAllocator` global picks up only the operator's allocations;
//! parallel lib tests would corrupt the deltas). Mirrors
//! `analytic_lattice_alloc_check.rs` / `rrq_quant_alloc_check`.

#![cfg(feature = "selection_propagation")]

use katgpt_core::selection_propagation::{
    PropagationBlend, PropagationConfig, SelectionPropagationScratch,
    propagate_selection_to_fixpoint_into,
};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// G4: 0 allocations across 100 steady-state calls with reused scratch, in
/// BOTH blend modes, at the G2 scale (N=1024, budget 32). Single function so
/// the checks run serially (they share the global `CountingAllocator`).
#[test]
fn g4_zero_alloc_steady_state_both_blends() {
    // Deterministic sparse graph: N=1024 nodes, ~8 out-edges each (chain +
    // distractor density comparable to the G1 fixture).
    const N: usize = 1024;
    const DEG: usize = 8;
    let mut offsets = vec![0u32; N + 1];
    let mut targets = Vec::with_capacity(N * DEG);
    let mut weights = Vec::with_capacity(N * DEG);
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        state
    };
    for off in offsets.iter_mut().take(N) {
        *off = targets.len() as u32;
        for _ in 0..DEG {
            let t = (rng() % N as u64) as u32;
            let w = ((rng() >> 33) as f32 / (1u64 << 31) as f32) * 0.9 + 0.05;
            targets.push(t);
            weights.push(w);
        }
    }
    offsets[N] = targets.len() as u32;

    let mut seed = vec![0.0f32; N];
    for s in seed.iter_mut() {
        *s = katgpt_core::sigmoid(4.0 * (((rng() >> 33) as f32 / (1u64 << 31) as f32) - 0.5));
    }

    let mut out = vec![0.0f32; N];
    let mut scratch = SelectionPropagationScratch::with_capacity(N, 32);

    for blend in [PropagationBlend::Mass, PropagationBlend::Mean] {
        const CALLS: usize = 100;

        let cfg = PropagationConfig {
            blend,
            ..Default::default()
        };
        // Warmup: settle any lazy allocations (SIMD dispatcher, etc.).
        for _ in 0..5 {
            let _ = propagate_selection_to_fixpoint_into(
                &offsets,
                &targets,
                &weights,
                &seed,
                N,
                32,
                &cfg,
                &mut out,
                &mut scratch,
            );
        }
        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
        for _ in 0..CALLS {
            let _ = propagate_selection_to_fixpoint_into(
                &offsets,
                &targets,
                &weights,
                &seed,
                N,
                32,
                &cfg,
                &mut out,
                &mut scratch,
            );
        }
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
        let deallocs = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
        assert_eq!(
            (allocs, deallocs),
            (0, 0),
            "G4 FAIL ({blend:?}): {allocs} allocs / {deallocs} deallocs across {CALLS} steady-state calls"
        );
    }
}

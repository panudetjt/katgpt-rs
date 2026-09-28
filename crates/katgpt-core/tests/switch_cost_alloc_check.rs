//! Issue 663 G4 — zero-allocation steady state for the SwitchCostTable hot
//! paths (lookup, sequence entropy, telemetry ingest, snapshot, factorized
//! lookup).
//!
//! Separate test binary (mirrors `bench_656_privilege_alloc_check`) so the
//! `CountingAllocator` global picks up only this module's allocations.
//! **One `#[test]` function, not several** — the counter is a process
//! global; parallel tests in one binary corrupt each other's deltas.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features switch_cost \
//!     --test switch_cost_alloc_check --release -- --nocapture
//! ```

#![cfg(feature = "switch_cost")]

use katgpt_core::switch_cost::{DEFAULT_ALPHA, FactorizedSwitchCost, SwitchCostTable, cdf_rank};
use std::hint::black_box;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

#[test]
fn g4_zero_alloc_steady_state() {
    // ── Warm-up (outside measurement) ────────────────────────────────────
    let mut table = SwitchCostTable::<8>::new(DEFAULT_ALPHA);
    let mut fact = FactorizedSwitchCost::<8, 3>::new([0, 0, 0, 1, 1, 1, 2, 2], DEFAULT_ALPHA);
    for a in 0..8 {
        for k in 0..100u32 {
            let ok = k % 3 != 0;
            table.record_solo(a, ok);
            fact.record_solo(a, ok);
        }
        for b in 0..8 {
            for k in 0..100u32 {
                let ok = !(k + a as u32 + b as u32).is_multiple_of(4);
                table.record_switch(a, b, ok);
                fact.record_switch(a, b, ok);
            }
        }
    }
    let seq: [usize; 16] = core::array::from_fn(|i| (i * 5 + 1) & 7);
    let sample: [f32; 8] = core::array::from_fn(|i| 1.0 + i as f32 * 0.25);

    // ── Measured: every hot path, 0 allocations ──────────────────────────
    let ((), allocs) = alloc_delta(|| {
        let mut acc = 0.0f32;
        for i in 0..1_000_000u32 {
            let a = (i & 7) as usize;
            let b = ((i >> 3) & 7) as usize;
            acc += table.ske(a, b);
            acc += fact.ske(a, b);
        }
        for _ in 0..100_000 {
            acc += table.sequence_entropy(&seq);
            acc += fact.sequence_entropy(&seq);
        }
        for k in 0..100_000u32 {
            table.record_solo((k & 7) as usize, k & 1 == 0);
            table.record_switch((k & 7) as usize, ((k >> 3) & 7) as usize, k % 3 != 0);
        }
        let snap = table.snapshot();
        acc += snap.ske(0, 7);
        acc += snap.sequence_entropy(&seq);
        acc += cdf_rank(2.5, &sample);
        black_box(acc);
    });
    assert_eq!(
        allocs, 0,
        "steady-state switch_cost paths allocated {allocs}×"
    );
}

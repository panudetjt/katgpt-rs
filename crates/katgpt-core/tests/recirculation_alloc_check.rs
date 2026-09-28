//! Issue 673 G1/G4 — recirculation operator determinism + zero-allocation
//! steady state (the capture/mix cross-step loop at production width).
//!
//! Separate test binary (mirrors `switch_cost_alloc_check`) so the
//! `CountingAllocator` global picks up only this module's allocations.
//! **One `#[test]` per binary** — the counter is process-global. The G1
//! bit-identity repeat and the G2 latency gates live in the module's unit
//! tests; this binary owns the multi-step loop alloc check.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features recirculation \
//!     --test recirculation_alloc_check --release -- --nocapture
//! ```

#![cfg(feature = "recirculation")]

use katgpt_core::recirculation::{RecircBuffer, RecircOp};
use std::hint::black_box;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

#[test]
fn g4_zero_alloc_steady_state_cross_step_loop() {
    let d = 2048usize;
    let op = RecircOp::convex(35, 16, 0.15, 10);
    let mut buf = RecircBuffer::new(d);
    let mut state: Vec<f32> = (0..d).map(|i| (i % 41) as f32 / 41.0 - 0.5).collect();
    let src_probe: Vec<f32> = (0..d).map(|i| (i % 43) as f32 / 43.0 - 0.5).collect();

    // Warm-up outside measurement.
    for step in 0..100u32 {
        op.mix_into(step, buf.as_slice(), &mut state);
        buf.capture(&src_probe);
    }

    let ((), allocs) = alloc_delta(|| {
        let mut acc = 0.0f32;
        for step in 0..50_000u32 {
            op.mix_into(step, buf.as_slice(), &mut state);
            buf.capture(&state);
            acc += state[(step as usize) % d];
        }
        black_box(acc);
    });
    assert_eq!(
        allocs, 0,
        "steady-state recirculation loop allocated {allocs}×"
    );
}

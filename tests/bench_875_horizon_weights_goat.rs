//! GOAT gate — Issue 875 T1: remaining-horizon weights + the PFD w(t)
//! committed table (Research 582 / arXiv:2605.09071).
//!
//! - **G1 (e2e)**: the committed table matches the closed-form recompute
//!   bit-for-bit through the public API, and the frozen commitment detects
//!   tampering on load.
//! - **G2**: O(1) table lookup vs the STRONG per-call baseline —
//!   `pfd_horizon_weight_at` with the caller-held cumulative integral (one
//!   `exp` + multiplies; the fair fight, NOT the O(n) re-integration the
//!   table also avoids) — measured with the shared interleaved-pairs
//!   `ab_timing` harness (Issue 723/833 treatment), median lookup/recompute
//!   ≤ 0.5 (≥ 2× faster). Run with `--release` — a latency gate in a debug
//!   build measures an unoptimised binary (the repo's standing rule).
//! - **G3**: opt-in feature, default-off — no default-path surface (the
//!   in-module tests pin G4 zero-alloc under `debug_assertions`).
//!
//! The `[[test]]` row in Cargo.toml names `horizon_weights` in
//! `required-features` (the Issue-808 green-zero class: without the row, a
//! feature-off invocation compiles an empty binary and reports a pass).

#[path = "common/ab_timing.rs"]
mod ab_timing;

use ab_timing::ab_median_ratio;
use katgpt_core::horizon_weights::{
    HORIZON_WEIGHT_GRID, HorizonWeightTable, pfd_horizon_weight_at, pfd_horizon_weights,
    remaining_horizon_weight,
};
use std::hint::black_box;

/// Arbitrary horizon in "tick" units — the law is scale-free.
const T: f32 = 1000.0;

/// VE-flavored schedules (mirror the in-module fixture): g increasing in
/// t, positive mildly-sloped drift a.
fn schedules() -> ([f32; HORIZON_WEIGHT_GRID], [f32; HORIZON_WEIGHT_GRID]) {
    let mut g = [0.0f32; HORIZON_WEIGHT_GRID];
    let mut a = [0.0f32; HORIZON_WEIGHT_GRID];
    for i in 0..HORIZON_WEIGHT_GRID {
        let t = T * i as f32 / (HORIZON_WEIGHT_GRID - 1) as f32;
        g[i] = 0.5 + 0.0008 * t;
        a[i] = 0.25 + 0.0001 * t;
    }
    (g, a)
}

/// The caller-held cumulative trapezoid integral of `a` (what the STRONG
/// G2 baseline gets for free outside the timed region).
fn cumulative_integral(a: &[f32; HORIZON_WEIGHT_GRID]) -> [f32; HORIZON_WEIGHT_GRID] {
    let dt = T / (HORIZON_WEIGHT_GRID - 1) as f32;
    let mut out = [0.0f32; HORIZON_WEIGHT_GRID];
    for i in 1..HORIZON_WEIGHT_GRID {
        out[i] = out[i - 1] + 0.5 * (a[i - 1] + a[i]) * dt;
    }
    out
}

#[test]
fn g1_table_matches_closed_form_bit_for_bit() {
    let (g, a) = schedules();
    let table = HorizonWeightTable::build(&g, &a, T);

    // e2e bit-match against the pure grid fill.
    let mut direct = [0.0f32; HORIZON_WEIGHT_GRID];
    pfd_horizon_weights(&g, &a, T, &mut direct);
    for (i, (&tw, &dw)) in table.weights().iter().zip(direct.iter()).enumerate() {
        assert_eq!(
            tw.to_bits(),
            dw.to_bits(),
            "G1 e2e bit-match failed at grid point {i}"
        );
    }

    // And at every grid point the point form (same evaluation order)
    // agrees with the table row — the lookup path lands on the frozen
    // value, not a re-rounding of it.
    let ints = cumulative_integral(&a);
    for i in 0..HORIZON_WEIGHT_GRID {
        let t = T * i as f32 / (HORIZON_WEIGHT_GRID - 1) as f32;
        let point = pfd_horizon_weight_at(t, T, g[i], ints[i]);
        assert_eq!(point.to_bits(), table.weights()[i].to_bits(), "i={i}");
    }

    // The generic law's zero-terminal-weight corollary, e2e.
    assert_eq!(
        table.weights()[HORIZON_WEIGHT_GRID - 1].to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(remaining_horizon_weight(T, T).to_bits(), 0.0f32.to_bits());
    assert_eq!(remaining_horizon_weight(0.0, T), 1.0);

    // Commitment verifies untampered.
    assert!(table.verify());
}

#[test]
fn g2_lookup_beats_strong_per_call_recompute() {
    let (g, a) = schedules();
    let ints = cumulative_integral(&a);
    let table = HorizonWeightTable::build(&g, &a, T);

    // Shared t samples, hoisted OUT of both arms so the timed bodies are
    // only the op under test (the fair fight: both arms pay the same array
    // read for t).
    let ts: [f32; HORIZON_WEIGHT_GRID] =
        core::array::from_fn(|k| T * k as f32 / (HORIZON_WEIGHT_GRID - 1) as f32);

    // Interleaved A/B: A = per-call closed form (caller-held integral —
    // one exp + multiplies) = the BASELINE; B = table lookup = the
    // CANDIDATE. `AbRatio::median` is b/a (the harness's overhead
    // convention: < 1 means the candidate is faster), so the gate is
    // median <= 0.5 — lookup at most HALF the strong per-call baseline.
    // Inputs cycle the grid so neither arm can be constant-folded;
    // results are black-boxed so the optimiser cannot delete the work
    // (Issue 855).
    let n = HORIZON_WEIGHT_GRID;
    let mut sink_a = 0.0f32;
    let mut sink_b = 0.0f32;
    let ratio = ab_median_ratio(
        41,   // rounds (odd → a true median)
        1024, // iterations per arm per round
        128,  // warmup
        |i| {
            let k = i % n;
            sink_a += black_box(pfd_horizon_weight_at(black_box(ts[k]), T, g[k], ints[k]));
        },
        |i| {
            let k = i % n;
            sink_b += black_box(table.w_at(black_box(ts[k])));
        },
    );
    assert!(
        sink_a.is_finite() && sink_b.is_finite(),
        "arms must be consumed"
    );

    ratio.report("G2 per-call-exp vs table-lookup");
    println!(
        "   recompute {:.2} ns/op vs lookup {:.2} ns/op",
        ratio.a_ns_per_iter(),
        ratio.b_ns_per_iter()
    );
    assert!(
        ratio.median <= 0.5,
        "G2 FAIL: lookup/recompute median {:.3} > 0.5 — lookup must be at least 2x the strong per-call exp baseline",
        ratio.median
    );
}

//! SE(2) Equivariant Lift — G2 Perf Gate (Plan 560).
//!
//! Measures the median latency of `se2_lift_into` at three production-scale
//! grid sizes. The gate target is < 1ms (1000µs) at the largest production
//! grid (32×32 NPC perception scale, 8 orientations, 5×5 kernel).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/plan560 cargo run -p katgpt-dec --release \
//!     --features se2_equivariant_lift --bench bench_560_se2_lift_perf -- --nocapture
//! ```

#![cfg(feature = "se2_equivariant_lift")]

use katgpt_dec::{se2_lift_into, se2_project_integrate_into, se2_project_max_into};
use std::hint::black_box;
use std::time::Instant;

const WARMUP: usize = 100;
const ITERS: usize = 1000;

fn median_ns(samples: &mut [u64]) -> f64 {
    samples.sort();
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2] as f64
    } else {
        (samples[n / 2 - 1] as f64 + samples[n / 2] as f64) * 0.5
    }
}

fn bench_shape(name: &str, w: usize, h: usize, k: usize, n_orient: usize) -> f64 {
    let field: Vec<f32> = (0..(w * h))
        .map(|i| ((i as f32) * 0.37).sin() * 0.5 + 0.5)
        .collect();
    let kernel: Vec<f32> = (0..(k * k))
        .map(|i| ((i as f32) * 0.13).sin() * 0.3 + 0.5)
        .collect();
    let mut out = vec![0.0f32; w * h * n_orient];

    // Warmup
    for _ in 0..WARMUP {
        se2_lift_into(
            black_box(&field),
            w,
            h,
            black_box(&kernel),
            k,
            n_orient,
            black_box(&mut out),
        );
    }

    let mut samples: Vec<u64> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        se2_lift_into(
            black_box(&field),
            w,
            h,
            black_box(&kernel),
            k,
            n_orient,
            black_box(&mut out),
        );
        samples.push(t.elapsed().as_nanos() as u64);
    }

    let med_ns = median_ns(&mut samples);
    let med_us = med_ns / 1000.0;
    let total_cells = w * h * n_orient;
    let per_cell_ns = med_ns / (total_cells as f64);
    println!(
        "  {name:<24} ({w}×{h}×{n_orient}×{k}×{k}): median {med_us:>8.2} µs  ({per_cell_ns:.2} ns/cell, {total_cells} total cells)"
    );
    med_us
}

fn bench_projection(name: &str, w: usize, h: usize, n_orient: usize, integr: bool) -> f64 {
    let n_cells = w * h;
    let lifted: Vec<f32> = (0..(n_cells * n_orient))
        .map(|i| ((i as f32) * 0.37).sin() * 0.5 + 0.5)
        .collect();
    let mut out = vec![0.0f32; n_cells];

    let op = |lifted: &[f32], out: &mut [f32]| {
        if integr {
            se2_project_integrate_into(lifted, n_cells, n_orient, out);
        } else {
            se2_project_max_into(lifted, n_cells, n_orient, out);
        }
    };

    for _ in 0..WARMUP {
        op(black_box(&lifted), black_box(&mut out));
    }
    let mut samples: Vec<u64> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        op(black_box(&lifted), black_box(&mut out));
        samples.push(t.elapsed().as_nanos() as u64);
    }
    let med_ns = median_ns(&mut samples);
    let med_us = med_ns / 1000.0;
    println!("  {name:<24} ({w}×{h}×{n_orient}):        median {med_us:>8.2} µs");
    med_us
}

fn main() {
    println!("\n=== SE(2) Lift G2 Perf Gate (Plan 560) ===\n");

    println!("Lift latency (target: < 1000 µs at 32×32×8×5×5):");
    let lift_16 = bench_shape("16×16×8 orientations, 5×5", 16, 16, 5, 8);
    let lift_32 = bench_shape("32×32×8 orientations, 5×5", 32, 32, 5, 8);
    let lift_64 = bench_shape("64×64×8 orientations, 5×5", 64, 64, 5, 8);

    println!();
    println!("Projection latency (cheap by design — sum/max reduction):");
    let _ = bench_projection("project_integrate 32×32×8", 32, 32, 8, true);
    let _ = bench_projection("project_max 32×32×8", 32, 32, 8, false);

    println!();
    println!("=== Verdict ===\n");
    let gate = 1000.0; // 1 ms target
    let budget_20hz = 50_000.0; // 20 Hz tick budget in µs

    let pass_16 = lift_16 < gate;
    let pass_32 = lift_32 < gate;
    let pass_64 = lift_64 < gate;

    println!(
        "  G2 16×16: {:.2} µs vs {:.0} µs target → {} ({:.0}× under)",
        lift_16,
        gate,
        if pass_16 { "PASS" } else { "FAIL" },
        gate / lift_16
    );
    println!(
        "  G2 32×32: {:.2} µs vs {:.0} µs target → {} ({:.0}× under)",
        lift_32,
        gate,
        if pass_32 { "PASS" } else { "FAIL" },
        gate / lift_32
    );
    println!(
        "  G2 64×64: {:.2} µs vs {:.0} µs target → {} ({:.0}× under)",
        lift_64,
        gate,
        if pass_64 { "PASS" } else { "FAIL" },
        gate / lift_64
    );

    println!();
    println!("  Per-NPC scale analysis (@ 32×32 lift): ");
    println!(
        "    Hero scale (1 NPC/tick):  {:.2} µs   ({:.1}% of 20Hz tick budget)",
        lift_32,
        lift_32 * 100.0 / budget_20hz
    );
    println!(
        "    Squad scale (10 NPCs/tick): {:.2} µs   ({:.1}% of 20Hz tick budget)",
        lift_32 * 10.0,
        lift_32 * 10.0 * 100.0 / budget_20hz
    );
    println!(
        "    Crowd scale (1000 NPCs/tick): {:.2} ms ({:.1}% of 20Hz tick budget — TOO EXPENSIVE for per-NPC per-tick at crowd scale; use per-zone or LoD)",
        lift_32 * 1000.0 / 1000.0,
        lift_32 * 1000.0 * 100.0 / budget_20hz
    );
    println!(
        "    Zone scale (1 lift per zone, ~16 zones): {:.2} ms ({:.1}% of 20Hz tick budget)",
        lift_32 * 16.0 / 1000.0,
        lift_32 * 16.0 * 100.0 / budget_20hz
    );

    if pass_16 && pass_32 && pass_64 {
        println!(
            "\n  G2 PERF VERDICT: PASS — se2_lift_into is well within the 1ms target at all production sizes."
        );
    } else {
        println!("\n  G2 PERF VERDICT: FAIL — one or more sizes exceed the 1ms target.");
    }
}

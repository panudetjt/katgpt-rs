//! Bench 763 — PC-ALM dual-wave GOAT gates (Issue 775 T6/T7).
//!
//! - **G2 (reach)** — anomaly injected at vertex 0 of an L-cell path graph;
//!   ticks until |h[L−1]| crosses the detectability threshold. Wave
//!   (α=1, self-calibrated η=1/4, ρ=1) vs its own α=0 diffusion twin (the
//!   SAME operators, same rate ηρ — the honest A/B: only α differs).
//!   PASS: wave arrives, within the t_infl = L/√(αη) = 2L prediction band
//!   (≤ 4L), strictly before heat at every L, and the heat/wave ratio
//!   GROWS with L (the quadratic-vs-linear signature — not just "faster").
//! - **G4 (latency + zero-alloc)** — one `wave_step_into` at K=100
//!   (10×10 grid, dim 8 — the sheaf_admm G4 workload shape) against the
//!   < 5 µs budget class, K=1024 (32×32) reported with a linear-scaling
//!   gate, and 0 allocations in steady state (counting allocator).
//!
//! The G1-adjoint gate (duals vs explicit reverse-mode, cosine ≥ 0.9 at
//! T=2L) lives in katgpt-core's `bench_775_adjoint_goat` — katgpt-dec is
//! zero-dep by contract, and the adjoint readout ships in
//! `katgpt_core::dual`.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/bench763 cargo bench -p katgpt-dec \
//!   --features dual_wave --no-default-features \
//!   --bench bench_775_dual_wave_goat -- --nocapture
//! ```

#![cfg(feature = "dual_wave")]

// Shared CountingAllocator macro (the bench_407 mirror).
#[path = "../tests/common/counting_allocator.rs"]
mod counting_allocator;

use katgpt_dec::{CellComplex, CochainField, WaveParams, WaveScratch, wave_step_into};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

counting_allocator!();

/// Detectability threshold at the far end (the injected anomaly is 1.0).
const THRESHOLD: f32 = 1e-3;

/// Ticks until |h[L−1]| ≥ THRESHOLD (None if not within `cap`).
fn ticks_to_detect(cx: &CellComplex, params: WaveParams, cap: usize) -> Option<usize> {
    let l = cx.n_vertices();
    let mut h = CochainField::zeros(0, l, 1);
    h.data[0] = 1.0;
    let mut lambda = CochainField::zeros(1, cx.n_edges(), 1);
    let mut scratch = WaveScratch::new(cx, 0, 1);
    for t in 0..cap {
        wave_step_into(cx, &mut h, &mut lambda, &params, &mut scratch);
        if h.data[l - 1].abs() >= THRESHOLD {
            return Some(t + 1);
        }
    }
    None
}

// ===========================================================================
// G2 — reach: ballistic vs diffusive on path graphs
// ===========================================================================

fn g2_reach() -> bool {
    println!(
        "─ G2 reach (inject at v0, detect |h[L−1]| ≥ {THRESHOLD}; wave α=1 vs heat α=0, same η=¼ ρ=1)"
    );
    println!(
        "  {:<6}{:>8} {:>9} {:>10} {:>9}   verdict",
        "L", "wave", "heat", "t_infl=2L", "heat/wave"
    );
    let mut all_pass = true;
    let mut ratios: Vec<Option<f64>> = Vec::new();
    for &l in &[16usize, 64, 128] {
        let edges: Vec<(usize, usize)> = (0..l - 1).map(|i| (i, i + 1)).collect();
        let cx = CellComplex::from_edges(l, &edges);
        let wave_params = WaveParams::self_calibrated();
        let heat_params = WaveParams {
            alpha: 0.0,
            ..wave_params
        };
        // Caps: wave 8L (generous over the 2L prediction); heat 20·L² (the
        // diffusive L² law — measured heat sits far below this).
        let wave = ticks_to_detect(&cx, wave_params, 8 * l);
        let heat = ticks_to_detect(&cx, heat_params, 20 * l * l);

        let row_pass = match (wave, heat) {
            (Some(w), Some(hh)) => {
                let ratio = hh as f64 / w as f64;
                ratios.push(Some(ratio));
                let ok = w <= 4 * l && w < hh;
                println!(
                    "  {l:<6}{w:>8} {hh:>9} {:<10} {ratio:>9.2}   {}",
                    2 * l,
                    if ok { "PASS ✅" } else { "FAIL ❌" }
                );
                ok
            }
            (Some(w), None) => {
                // Heat did not arrive within its L² cap — the honest limit
                // case (report the wave arrival; ratio is a lower bound).
                ratios.push(Some((20 * l * l) as f64 / w as f64));
                let ok = w <= 4 * l;
                println!(
                    "  {l:<6}{w:>8} {:>9} {:<10} {:>9}   {}",
                    ">cap",
                    2 * l,
                    "—",
                    if ok {
                        "PASS ✅ (heat > cap)"
                    } else {
                        "FAIL ❌"
                    }
                );
                ok
            }
            (None, _) => {
                ratios.push(None);
                println!(
                    "  {l:<6}{:>8} {:>9} {:<10} {:>9}   FAIL ❌ (wave never arrived)",
                    "—",
                    "—",
                    2 * l,
                    "—"
                );
                false
            }
        };
        all_pass &= row_pass;
    }
    // The scaling signature: heat/wave must grow with L (quadratic vs
    // linear — the paper's Eq 23 law, not a constant-factor win).
    if let (Some(Some(r16)), Some(Some(r128))) = (ratios.first().copied(), ratios.last().copied()) {
        let grows = r128 > 2.0 * r16;
        println!(
            "  scaling: heat/wave ratio {:.2} → {:.2} (gate: ×2 growth) → {}",
            r16,
            r128,
            if grows { "PASS ✅" } else { "FAIL ❌" }
        );
        all_pass &= grows;
    }
    all_pass
}

// ===========================================================================
// G4 — latency + zero-alloc
// ===========================================================================

fn build_grid_workload(w: usize) -> (CellComplex, CochainField, CochainField, WaveScratch) {
    let cx = CellComplex::grid_2d(w, w);
    let dim = 8usize;
    let mut h = CochainField::zeros(0, cx.n_vertices(), dim);
    let mut s = 0x775_6026_0000_0001u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (((s >> 40) as f32) / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    for v in h.data.iter_mut() {
        *v = next();
    }
    let lambda = CochainField::zeros(1, cx.n_edges(), dim);
    let scratch = WaveScratch::new(&cx, 0, dim);
    (cx, h, lambda, scratch)
}

fn g4_latency(w: usize, iters: usize) -> (f64, usize) {
    let (cx, mut h, mut lambda, mut scratch) = build_grid_workload(w);
    let params = WaveParams::self_calibrated();
    // Warmup.
    for _ in 0..10 {
        wave_step_into(&cx, &mut h, &mut lambda, &params, &mut scratch);
    }
    // Zero-alloc snapshot (steady state, 100 steps).
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..100 {
        wave_step_into(&cx, &mut h, &mut lambda, &params, &mut scratch);
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - before;

    let start = Instant::now();
    for _ in 0..iters {
        wave_step_into(
            black_box(&cx),
            black_box(&mut h),
            black_box(&mut lambda),
            black_box(&params),
            black_box(&mut scratch),
        );
    }
    let per_call_us = start.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
    (per_call_us, allocs)
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS ✅" } else { "FAIL ❌" }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Issue 775 — Dual-Wave GOAT (G2 reach, G4 latency + zero-alloc)  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut all_pass = true;

    all_pass &= g2_reach();
    println!();

    // K=100 workload (10×10 grid, dim 8 — the sheaf_admm G4 shape).
    let (us100, allocs100) = g4_latency(10, 2000);
    let gate100 = us100 < 5.0 && allocs100 == 0;
    println!(
        "G4 latency  K=100  (10×10, dim 8): mean = {:.3} µs  (gate < 5.0)  → {}",
        us100,
        verdict(us100 < 5.0)
    );
    println!(
        "G4 zero-alloc K=100 (100 steps, steady state): allocs = {allocs100}  (gate = 0)  → {}",
        verdict(allocs100 == 0)
    );
    all_pass &= gate100;

    // K=1024 (32×32): linear-scaling budget (10× vertices ⇒ ≤10× the K=100
    // gate as the class bound; reported against the same budget family).
    let (us1024, allocs1024) = g4_latency(32, 500);
    let gate1024 = us1024 < 50.0 && allocs1024 == 0;
    println!(
        "G4 latency  K=1024 (32×32, dim 8): mean = {:.3} µs  (gate < 50.0, linear-class)  → {}",
        us1024,
        verdict(us1024 < 50.0)
    );
    println!(
        "G4 zero-alloc K=1024 (100 steps): allocs = {allocs1024}  (gate = 0)  → {}",
        verdict(allocs1024 == 0)
    );
    all_pass &= gate1024;

    println!();
    if all_pass {
        println!("══ DUAL-WAVE PERF GATES PASS — G2 + G4 both pass ══");
    } else {
        println!("══ DUAL-WAVE PERF GATES FAIL — see above ══");
    }
    std::process::exit(if all_pass { 0 } else { 1 });
}

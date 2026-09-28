//! Issue 681 — Sketched Gaussianity Probe GOAT gate.
//!
//! G1   falsifiable fixtures (i)–(iv) + the non-redundancy pin:
//!      `effective_rank` computed on the SAME fixtures must be HIGH — the
//!      blind-spot claim pinned against the shipped second-moment metric
//!      (the `p415_g2_nonredundancy_vs_global` pattern applied to
//!      shape-vs-rank).
//! G2   latency vs `effective_rank` at d=64, n=1024 (erank pays the O(d³)
//!      Jacobi sweep; the probe is O(B·d·|A| + |A|·B log B) — no eigensolve).
//! G5   determinism: three runs bit-identical.
//!
//! std::time::Instant + harness=false (repo bench convention). The G4
//! alloc gate lives in the module's own test (lib test binary installs the
//! TrackingAllocator).
//!
//! Run: cargo bench -p katgpt-core --bench bench_681_gaussianity_goat
//!      (or `cargo test --features gaussianity_probe,sink_aware_attn
//!       --bench bench_681_gaussianity_goat -- --nocapture`)

use katgpt_core::data_probe::gaussianity::{GaussianityScratch, sketched_gaussianity};
use katgpt_core::data_probe::geometry::effective_rank;
use katgpt_core::types::Rng;

const N: usize = 1024;
const D: usize = 64;

fn gaussian_population(seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0f32; N * D];
    for v in out.iter_mut() {
        *v = rng.normal();
    }
    out
}

fn bimodal_axis_population(axis: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0f32; N * D];
    for i in 0..N {
        let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        for j in 0..D {
            out[i * D + j] = rng.normal();
        }
        out[i * D + axis] += sign * 3.0;
    }
    out
}

fn radial_heavy_tail_population(seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0f32; N * D];
    for i in 0..N {
        let r = if rng.uniform() < 0.95 { 1.0 } else { 10.0 };
        let mut norm = 0.0f64;
        for j in 0..D {
            let g = rng.normal() as f64;
            out[i * D + j] = g as f32;
            norm += g * g;
        }
        let inv = 1.0 / norm.sqrt().max(1e-12);
        for j in 0..D {
            out[i * D + j] *= r * inv as f32;
        }
    }
    out
}

fn lattice_population_d8(seed: u64) -> (Vec<f32>, usize) {
    let d = 8;
    let mut rng = Rng::new(seed);
    let mut out = vec![0.0f32; N * d];
    for v in out.iter_mut() {
        *v = if rng.uniform() < 0.5 { 0.0 } else { 1.0 };
    }
    (out, d)
}

/// `effective_rank` over a flat population (the geometry API takes
/// `&[Vec<f32>]` — test-side assembly, not probe cost).
fn erank_of(states: &[f32], d: usize) -> f32 {
    let rows: Vec<Vec<f32>> = states.chunks(d).map(|c| c.to_vec()).collect();
    effective_rank(&rows)
}

fn main() {
    let mut failures = 0usize;

    // ── G1: fixtures (i)–(iv) ────────────────────────────────────────────
    println!("═══ G1 — falsifiable fixtures (n={N}) ═══");

    // (i) Gaussian d=64: accept + erank high.
    {
        let states = gaussian_population(42);
        let mut scratch = GaussianityScratch::new(N, D, 7);
        let rep = sketched_gaussianity(&states, &mut scratch);
        let er = erank_of(&states, D);
        println!(
            "(i) Gaussian      d={D}: score={:.4} p_min={:.4} erank={:.1}/{}",
            rep.score, rep.min_p_value, er, D
        );
        if rep.score <= 0.5 {
            failures += 1;
            println!("  FAIL: Gaussian fixture must accept");
        }
        if er < 0.85 * D as f32 {
            failures += 1;
            println!("  FAIL: erank {er:.1} below 0.85·d — fixture invalid");
        }
    }

    // (ii) Bimodal axis-aligned d=64, μ=3σ: reject at anchor + erank high.
    //      The NON-REDUNDANCY PIN: erank says healthy, the probe says sick.
    //      μ=3σ consumes exactly one covariance eigenvalue (theoretical
    //      erank ≈ 80% of d — pinned at ≥ 0.75·d: still "healthy" to a rank
    //      metric, where an actual collapse reads 1-10% of d).
    {
        let states = bimodal_axis_population(0, 43);
        let mut scratch = GaussianityScratch::new(N, D, 7);
        let rep = sketched_gaussianity(&states, &mut scratch);
        let er = erank_of(&states, D);
        println!(
            "(ii) Bimodal e_0  d={D}: score={:.2e} p_min={:.2e} erank={:.1}/{} worst_dir={} D_0={:.4}",
            rep.score, rep.min_p_value, er, D, rep.worst_direction, rep.per_direction[0]
        );
        if rep.score >= 0.5 {
            failures += 1;
            println!("  FAIL: bimodal fixture must reject");
        }
        if rep.worst_direction != 0 {
            failures += 1;
            println!("  FAIL: axis-0 bimodal must be caught by the e_0 anchor");
        }
        if er < 0.75 * D as f32 {
            failures += 1;
            println!(
                "  FAIL: NON-REDUNDANCY broken — erank {er:.1} collapsed too; \
                 the fixture no longer demonstrates the blind spot"
            );
        }
    }

    // (iii) Radial heavy-tail d=64: reject margin-wide + erank ~full.
    {
        let states = radial_heavy_tail_population(45);
        let mut scratch = GaussianityScratch::new(N, D, 7);
        let rep = sketched_gaussianity(&states, &mut scratch);
        let er = erank_of(&states, D);
        let rejecting = rep.per_direction.iter().filter(|&&d| d > 0.1).count();
        println!(
            "(iii) Radial 5%@10 d={D}: score={:.2e} dirs D>0.1: {rejecting}/16 erank={:.1}/{}",
            rep.score, er, D
        );
        if rep.score >= 0.5 {
            failures += 1;
            println!("  FAIL: radial heavy-tail must reject");
        }
        if rejecting < 12 {
            failures += 1;
            println!("  FAIL: radial departure must be margin-wide (≥12/16)");
        }
        if er < 0.85 * D as f32 {
            failures += 1;
            println!("  FAIL: NON-REDUNDANCY broken — erank {er:.1} collapsed too");
        }
    }

    // (iv) Bernoulli lattice d=8: reject + erank ~full.
    {
        let (states, d) = lattice_population_d8(46);
        let mut scratch = GaussianityScratch::new(N, d, 7);
        let rep = sketched_gaussianity(&states, &mut scratch);
        let er = erank_of(&states, d);
        println!(
            "(iv) Lattice {{0,1}} d={d}: score={:.2e} p_min={:.2e} erank={:.2}/{}",
            rep.score, rep.min_p_value, er, d
        );
        if rep.score >= 0.5 {
            failures += 1;
            println!("  FAIL: lattice fixture must reject");
        }
        if er < 0.85 * d as f32 {
            failures += 1;
            println!("  FAIL: NON-REDUNDANCY broken — erank {er:.2} below 0.85·{d}");
        }
    }

    // ── G5: determinism ×3 ───────────────────────────────────────────────
    {
        let states = radial_heavy_tail_population(45);
        let mut scratch = GaussianityScratch::new(N, D, 7);
        let a = sketched_gaussianity(&states, &mut scratch);
        let b = sketched_gaussianity(&states, &mut scratch);
        let c = sketched_gaussianity(&states, &mut scratch);
        let ok = a.per_direction == b.per_direction
            && b.per_direction == c.per_direction
            && a.score.to_bits() == c.score.to_bits()
            && a.min_p_value.to_bits() == c.min_p_value.to_bits();
        println!(
            "═══ G5 — determinism ×3 bit-identical: {} ═══",
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            failures += 1;
        }
    }

    // ── G2: latency vs effective_rank (d=64, n=1024) ─────────────────────
    {
        const REPS: usize = 50;

        let states = gaussian_population(42);
        let mut scratch = GaussianityScratch::new(N, D, 7);

        // Warmup.
        for _ in 0..3 {
            let _ = sketched_gaussianity(&states, &mut scratch);
            let _ = erank_of(&states, D);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            let _ = sketched_gaussianity(&states, &mut scratch);
        }
        let probe_us = t0.elapsed().as_secs_f64() / REPS as f64 * 1e6;

        // erank re-allocates per call by design (the geometry API) — the
        // row-assembly is included in its arm (that IS the shipped surface).
        let t1 = std::time::Instant::now();
        for _ in 0..REPS {
            let _ = erank_of(&states, D);
        }
        let erank_us = t1.elapsed().as_secs_f64() / REPS as f64 * 1e6;

        let ratio = erank_us / probe_us;
        println!(
            "═══ G2 — latency @ n={N} d={D}: probe {probe_us:.1}µs vs erank {erank_us:.1}µs \
             (erank/probe = {ratio:.2}×; target probe ≤ erank) ═══"
        );
        if probe_us > erank_us {
            failures += 1;
            println!("FAIL: probe slower than effective_rank");
        }
    }

    println!();
    if failures == 0 {
        println!("Issue 681 GOAT: ALL GATES PASS (opt-in; promotion deferred to a consumer)");
    } else {
        println!("Issue 681 GOAT: {failures} FAILURES");
        std::process::exit(1);
    }
}

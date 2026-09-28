//! Issue 158 Phase 1 — Interpolation Geometry GOAT gate bench.
//!
//! Exercises the four GOAT gates for the `interpolation_geometry` primitive
//! (Issue 158, Research 445 — Prabhudesai & Geng, *Latent Thought Flows with
//! Text Compression*, Jun 2026).
//!
//! # Gates
//!
//! - **G1 (correctness)** — `imauve_score` strictly distinguishes a known-good
//!   geometry (1D manifold) from a known-bad one (radial clustering, the
//!   paper's "length clustering" failure mode analog). Good score > bad score
//!   AND good > 0.9 AND bad < 0.95.
//!
//! - **G2 (perf)** — `imauve_score` at the audit-cadence reference scale
//!   (n=256 anchors × 256 candidates × 64 dims) completes in < 50ms. This is
//!   the paper's reference scale (their TinyStories n≈256); 50ms is a
//!   generous audit-cadence budget (vs the 5µs hot-path budget of Plan 342).
//!
//! - **G4 (alloc-free)** — `imauve_score` allocates ZERO bytes on the hot
//!   path (caller-supplied midpoint scratch is reused across anchors).
//!   Measured via the CountingAllocator over 100 calls.
//!
//! - **G4-intervention** — `intervention_battery` is also zero-alloc.
//!
//! # Non-goal
//!
//! - **G3 (no-regression)** — covered by `cargo clippy --all-features`
//!   passing on the module's lib target (no other feature depends on
//!   `interpolation_geometry` yet, so there's nothing to regress).
//!
//! # Run
//!
//! ```bash
//! cargo run --release --bench bench_456_interpolation_geometry_goat \
//!     --features interpolation_geometry -- --nocapture
//! ```

#![cfg(feature = "interpolation_geometry")]

use katgpt_core::interpolation_geometry::{
    EuclideanLatentSpace, FixtureRng, GaussianMixtureSpace, imauve_score, intervention_battery,
};
use std::time::{Duration, Instant};

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── Config ────────────────────────────────────────────────────────────────

/// The reference audit-cadence scale: 256 anchors × 256 candidates × 64
/// dims. This is the paper's TinyStories n≈256 analog at shard style_weights
/// dimensionality.
const N_ANCHORS: usize = 256;
const DIM: usize = 64;

/// Warmup iterations (untimed).
const WARMUP: usize = 10;

/// Number of timed runs to take the median over.
const TIMED_RUNS: usize = 21;

/// G2 perf target: < 50 ms at the audit-cadence reference scale.
const G2_TARGET_MS: u64 = 50;

// ─── Helpers ───────────────────────────────────────────────────────────────

fn format_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns:>5} ns")
    } else if ns < 1_000_000 {
        format!("{:>5.2} µs", ns as f64 / 1_000.0)
    } else {
        format!("{:>5.2} ms", ns as f64 / 1_000_000.0)
    }
}

// ─── G1: correctness (good vs bad geometry) ────────────────────────────────

fn gate_g1_correctness() -> bool {
    println!("── G1: correctness (good > bad geometry) ─────────────────────");
    let good = GaussianMixtureSpace::good_along_manifold(8);
    let bad = GaussianMixtureSpace::bad_radial_clustering(8);

    let good_points = good.centers.clone();
    let bad_points = bad.centers.clone();

    let mut scratch = [0.0f32; 2];
    let max_possible = 10.0f32;
    let good_score = imauve_score(
        &good,
        &good_points,
        &good_points,
        &mut scratch,
        max_possible,
    );
    let bad_score = imauve_score(&bad, &bad_points, &bad_points, &mut scratch, max_possible);

    println!(
        "  good (1D manifold):  score = {:.6}  (n_anchors = {})",
        good_score.score, good_score.n_anchors
    );
    println!(
        "  bad  (radial cluster): score = {:.6}  (n_anchors = {})",
        bad_score.score, bad_score.n_anchors
    );

    let strict_order = good_score.score > bad_score.score;
    let good_high = good_score.score > 0.9;
    let bad_low = bad_score.score < 0.95;
    println!(
        "  good > bad:    {}",
        if strict_order { "PASS" } else { "FAIL" }
    );
    println!(
        "  good > 0.9:    {}  (got {})",
        if good_high { "PASS" } else { "FAIL" },
        good_score.score
    );
    println!(
        "  bad  < 0.95:   {}  (got {})",
        if bad_low { "PASS" } else { "FAIL" },
        bad_score.score
    );

    strict_order && good_high && bad_low
}

// ─── G2: perf at the audit-cadence reference scale ─────────────────────────

fn gate_g2_perf() -> bool {
    println!();
    println!("── G2: perf at n={N_ANCHORS} × d={DIM} ─────────────────────────────");

    let space = EuclideanLatentSpace::<DIM>;
    let mut rng = FixtureRng::new(42);

    // Build a synthetic point cloud at the audit-cadence reference scale.
    // 8 clusters along a 1D manifold embedded in DIM dimensions.
    let mut points: Vec<[f32; DIM]> = Vec::with_capacity(N_ANCHORS);
    let per_cluster = N_ANCHORS / 8;
    for cluster_t in [0.0f32, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1] {
        for _ in 0..per_cluster {
            let mut p = [0.0f32; DIM];
            for coord in &mut p {
                *coord = cluster_t + rng.range(-0.01, 0.01);
            }
            points.push(p);
        }
    }

    let mut scratch = [0.0f32; DIM];
    let max_possible = 4.0f32;

    // Warmup.
    for _ in 0..WARMUP {
        let _ = imauve_score(&space, &points, &points, &mut scratch, max_possible);
    }

    // Timed.
    let mut samples: Vec<Duration> = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let s = imauve_score(&space, &points, &points, &mut scratch, max_possible);
        let dur = t0.elapsed();
        samples.push(dur);
        // Side-effect to prevent elision.
        if s.score.is_nan() {
            std::process::abort();
        }
    }
    samples.sort();
    let median = samples[TIMED_RUNS / 2];

    println!(
        "  median over {} runs: {}  (target: < {})",
        TIMED_RUNS,
        format_duration(median),
        format_duration(Duration::from_millis(G2_TARGET_MS))
    );

    median.as_millis() as u64 <= G2_TARGET_MS
}

// ─── G4: zero-alloc hot path ───────────────────────────────────────────────

fn gate_g4_zero_alloc() -> bool {
    use std::sync::atomic::Ordering;

    println!();
    println!("── G4: zero-alloc hot path ───────────────────────────────────");

    let space = EuclideanLatentSpace::<DIM>;
    let mut rng = FixtureRng::new(7);
    let mut points: Vec<[f32; DIM]> = Vec::with_capacity(64);
    for cluster_t in [0.0f32, 0.5, 1.0, 1.5] {
        for _ in 0..16 {
            let mut p = [0.0f32; DIM];
            for coord in &mut p {
                *coord = cluster_t + rng.range(-0.01, 0.01);
            }
            points.push(p);
        }
    }
    let mut midpoint_scratch = [0.0f32; DIM];
    let max_possible = 4.0f32;

    // Reset counters, then call imauve_score 100 times.
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);
    for _ in 0..100 {
        let _ = imauve_score(
            &space,
            &points,
            &points,
            &mut midpoint_scratch,
            max_possible,
        );
    }
    let imauve_allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let imauve_deallocs = DEALLOC_COUNT.load(Ordering::Relaxed);
    println!(
        "  imauve_score × 100 calls: {imauve_allocs} allocs, {imauve_deallocs} deallocs  (target: 0, 0)"
    );

    // intervention_battery: build donors + scratch.
    let donors: Vec<[f32; DIM]> = (0..4)
        .map(|i| {
            let mut p = [0.0f32; DIM];
            let mut lrng = FixtureRng::new(100 + i as u64);
            for coord in &mut p {
                *coord = lrng.range(-1.0, 1.0);
            }
            p
        })
        .collect();
    let anchor = [0.5f32; DIM];
    let mut z = [0.0f32; DIM];
    let mut m = [0.0f32; DIM];
    let mut n = [0.0f32; DIM];

    ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);
    for seed in 0..100u64 {
        let _ = intervention_battery(&space, &anchor, &donors, seed, &mut z, &mut m, &mut n);
    }
    let battery_allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let battery_deallocs = DEALLOC_COUNT.load(Ordering::Relaxed);
    println!(
        "  intervention_battery × 100 calls: {battery_allocs} allocs, {battery_deallocs} deallocs  (target: 0, 0)"
    );

    imauve_allocs == 0 && battery_allocs == 0
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Issue 158 — Interpolation Geometry GOAT Gate (G1+G2+G4)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Config: n={N_ANCHORS} anchors, dim={DIM} (NeuronShard::style_weights scale)");
    println!("       {TIMED_RUNS} timed runs (median), {WARMUP} warmup, seed=42");

    let g1 = gate_g1_correctness();
    let g2 = gate_g2_perf();
    let g4 = gate_g4_zero_alloc();

    println!();
    println!("──────────────────────────────────────────────────────────────");
    println!("Gate summary:");
    println!("  G1 correctness:    {}", if g1 { "PASS" } else { "FAIL" });
    println!(
        "  G2 perf (n={}, d={}, < {}):  {}",
        N_ANCHORS,
        DIM,
        format_duration(Duration::from_millis(G2_TARGET_MS)),
        if g2 { "PASS" } else { "FAIL" }
    );
    println!("  G4 zero-alloc:     {}", if g4 { "PASS" } else { "FAIL" });
    println!();

    let all_pass = g1 && g2 && g4;
    println!(
        "Verdict: {}",
        if all_pass {
            "ALL GATES PASS — primitive meets the audit-cadence contract."
        } else {
            "ONE OR MORE GATES FAILED — see above."
        }
    );
    if !all_pass {
        std::process::exit(1);
    }
}

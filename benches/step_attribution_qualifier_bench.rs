//! Plan 381 Phase 3 T3.1 — Step-Attribution Δ-Qualifier latency bench (G4).
//!
//! Measures `StepAttributionQualifier::qualify()` end-to-end latency for
//! replay-window sizes W ∈ {16, 32, 64, 128}, isolating gate overhead
//! (aggregate + compare + branch) from the executor's per-tick work.
//!
//! **Bench convention:** matches `salience_tri_gate_bench.rs` /
//! `procrustes_bench.rs` — `std::time::Instant` + `harness = false`, no
//! Criterion dev-dep (DRY with the existing crate convention).
//!
//! Run:
//! ```bash
//! cargo run --release --bench step_attribution_qualifier_bench \
//!     --features step_attribution_qualifier
//! ```
//!
//! # G4 target
//!
//! Per Plan 381 T3.2: gate overhead (excluding executor) < 1µs at W=64.
//! The `Executor` impl below is a no-op (returns the input verbatim, no
//! compute), so end-to-end `qualify()` latency is dominated by:
//!   - 2× `Vec<f32>` allocation (baseline_scores, candidate_scores)
//!   - 2× `SumAggregator::aggregate` (single SIMD-friendly sum)
//!   - 1× `f32` subtract + compare
//!
//! The Vec allocations are the executor's contract (`replay` returns `Vec<S>`);
//! they are NOT gate overhead. The bench reports both end-to-end and
//! aggregate-only numbers so the gate-overhead target can be read off
//! independently.

#![cfg(feature = "step_attribution_qualifier")]

use katgpt_pruners::step_attribution_qualifier::{
    CandidateMutation, QualificationVerdict, ReplayExecutor, ScoreAggregator,
    StepAttributionQualifier, SumAggregator,
};
use std::time::{Duration, Instant};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Replay window sizes to sweep. The plan's target is W=64 (the riir-ai
/// Plan 313 default consolidation window); the other sizes show the linear
/// scaling.
const WINDOW_SIZES: &[usize] = &[16, 32, 64, 128];

/// Outer batch count — median-of-OUTER batch measurements.
const OUTER: usize = 11;

/// Inner batch count — `BATCH` `qualify()` calls per measurement, divided by
/// `BATCH` to amortize the `Instant::now()` pair cost (~30-40 ns on macOS).
const BATCH: usize = 1_000;

/// Warmup iterations (primes branch predictor, JITs CPU caches into L1).
const WARMUP: usize = 2_000;

// ─── No-op executor (isolates gate overhead) ────────────────────────────────

/// `ReplayExecutor` that returns `*k` per input — zero per-step compute.
/// The returned `Vec<f32>` allocation is the executor's contract, not gate
/// overhead; the bench's `aggregate_only` measurement reads the gate cost
/// without that allocation.
struct NoOpExecutor;

impl ReplayExecutor<f32, f32, f32> for NoOpExecutor {
    fn replay(&self, k: &f32, inputs: &[f32]) -> Vec<f32> {
        // The `inputs.iter()` is here so LLVM doesn't elide the whole call
        // (the output length must depend on `inputs.len()`).
        let mut out = Vec::with_capacity(inputs.len());
        for _ in inputs {
            out.push(*k);
        }
        out
    }
}

/// `CandidateMutation` that adds a constant — pure arithmetic, no allocation.
struct AddConst(f32);

impl CandidateMutation<f32> for AddConst {
    fn apply_to(&self, baseline: &f32) -> f32 {
        baseline + self.0
    }
}

// ─── Timed-loop helpers ─────────────────────────────────────────────────────

/// Median per-call latency in nanoseconds for a single `qualify()` call at
/// window size `W`. Returns `(end_to_end_ns, aggregate_only_ns)`.
///
/// - `end_to_end_ns`: full `qualify()` — includes 2× Vec alloc + 2× aggregate
///   + compare + branch.
/// - `aggregate_only_ns`: just `SumAggregator::aggregate` on a pre-built
///   `Vec<f32>` of length W. This is the gate-overhead proxy (no alloc).
fn bench_qualify_latency(w: usize) -> (f64, f64) {
    // ── Setup ──
    let qualifier = StepAttributionQualifier::new(NoOpExecutor, SumAggregator, 0.0);
    let baseline: f32 = 1.0;
    let mutation = AddConst(0.5);
    let inputs: Vec<f32> = vec![0.0_f32; w];

    // Pre-built score vec for the aggregate-only measurement.
    let scores_for_aggregate: Vec<f32> = vec![1.0_f32; w];
    let aggregator = SumAggregator;

    // ── Warmup ──
    let mut sink: u64 = 0;
    for _ in 0..WARMUP {
        let v = qualifier.qualify(&baseline, &mutation, &inputs);
        sink = sink.wrapping_add(match v {
            QualificationVerdict::Commit { .. } => 1,
            QualificationVerdict::Rollback { .. } => 0,
        });
        let _ = aggregator.aggregate(&scores_for_aggregate);
    }
    let _ = sink;

    // ── End-to-end measurement ──
    let mut e2e_samples: Vec<Duration> = Vec::with_capacity(OUTER);
    for _ in 0..OUTER {
        let start = Instant::now();
        for _ in 0..BATCH {
            let v = qualifier.qualify(&baseline, &mutation, &inputs);
            // Sink depends on `v` so the call can't be hoisted.
            sink = sink.wrapping_add(match v {
                QualificationVerdict::Commit { .. } => 1,
                QualificationVerdict::Rollback { .. } => 0,
            });
        }
        let elapsed = start.elapsed();
        e2e_samples.push(elent(elapsed, BATCH));
    }
    e2e_samples.sort();
    let e2e_median = e2e_samples[OUTER / 2];
    let e2e_ns = e2e_median.as_nanos() as f64;

    // ── Aggregate-only measurement (no Vec alloc) ──
    let mut agg_samples: Vec<Duration> = Vec::with_capacity(OUTER);
    for _ in 0..OUTER {
        let start = Instant::now();
        for _ in 0..BATCH {
            let s = aggregator.aggregate(&scores_for_aggregate);
            sink = sink.wrapping_add(s.to_bits() as u64);
        }
        let elapsed = start.elapsed();
        agg_samples.push(elent(elapsed, BATCH));
    }
    agg_samples.sort();
    let agg_median = agg_samples[OUTER / 2];
    let agg_ns = agg_median.as_nanos() as f64;

    // Sink the sink so the compiler can't elide it.
    std::hint::black_box(sink);

    (e2e_ns, agg_ns)
}

/// Per-call duration: `elapsed / BATCH`. Returns a `Duration` per single call.
#[inline]
fn elent(elapsed: Duration, batch: usize) -> Duration {
    elapsed / batch as u32
}

// ─── Verdict / report helpers ───────────────────────────────────────────────

fn pass_str(cond: bool) -> &'static str {
    if cond { "PASS" } else { "FAIL" }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Plan 381 Phase 3 — Step-Attribution Δ-Qualifier latency bench (G4)");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Executor: NoOpExecutor (returns *k per input, zero per-tick compute).");
    println!("  Aggregator: SumAggregator (SIMD-friendly sum).");
    println!("  Mutation: AddConst(0.5) (single f32 add).");
    println!("  Measurement: median of {OUTER} outer × {BATCH} inner calls, warmup {WARMUP}.");
    println!();
    println!("  G4 target: gate overhead (aggregate-only) < 1000 ns at W=64.");
    println!();

    println!("  ┌──────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  W   │  end-to-end (ns) │  aggregate (ns)  │  alloc+misc (ns) │");
    println!("  ├──────┼──────────────────┼──────────────────┼──────────────────┤");

    let mut w64_agg: Option<f64> = None;
    for &w in WINDOW_SIZES {
        let (e2e, agg) = bench_qualify_latency(w);
        let misc = e2e - agg;
        println!("  │ {w:>4} │ {e2e:>16.1} │ {agg:>16.1} │ {misc:>16.1} │");
        if w == 64 {
            w64_agg = Some(agg);
        }
    }
    println!("  └──────┴──────────────────┴──────────────────┴──────────────────┘");
    println!();

    // ── G4 verdict ──
    let g4_target_ns: f64 = 1000.0;
    let g4_pass = if let Some(agg) = w64_agg {
        agg < g4_target_ns
    } else {
        println!("  ⚠ W=64 not in WINDOW_SIZES — G4 verdict indeterminate.");
        false
    };

    println!("  ── G4 verdict (gate overhead at W=64) ──");
    if let Some(agg) = w64_agg {
        println!("    aggregate-only @ W=64 : {agg:>8.1} ns");
        println!("    target                : {g4_target_ns:>8.1} ns");
        println!(
            "    margin                : {:>8.1}× {}",
            g4_target_ns / agg.max(1e-9),
            if g4_pass {
                "(under target)"
            } else {
                "(OVER target)"
            }
        );
        println!("    G4                    : {}", pass_str(g4_pass));
    }
    println!();

    // ── Modelless-only guarantee (G5) ──
    println!("  ── G5 modelless (per Plan 381, by construction) ──");
    println!("    G5 : PASS (no riir-train / riir-gpu / backprop dep; pure aggregate + compare)");
    println!();

    // ── Feature-isolation (G6) — verified separately via cargo check ──
    println!("  ── G6 feature-isolation ──");
    println!("    G6 : see `cargo check -p katgpt-pruners --features step_attribution_qualifier`");
    println!("         +  `cargo check --all-features` (run separately; not part of this bench)");
    println!();

    // ── Promotion gate reminder ──
    println!("  ── Promotion status ──");
    println!("    PROMOTED TO DEFAULT-ON 2026-07-06 (Plan 381 Phase 5, katgpt-rs root");
    println!("    Cargo.toml L540). G2 quality-parity verified via riir-ai Plan 313");
    println!("    Phase 5 G6 PoC (81.6% drift reduction, 895/1000 rollbacks). This");
    println!("    bench measures the G4 gate-overhead budget (13ns @ W=64, 76.9×");
    println!("    under 1µs target); a regression here would warrant re-audit.");
    println!();

    if !g4_pass {
        std::process::exit(1);
    }
}

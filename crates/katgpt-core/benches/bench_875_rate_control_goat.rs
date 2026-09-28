//! Issue 873 primitive B — the rate_control GOAT gate (G2 latency; the
//! dual-EWLS effect-size controller from Research 581 / mini-AGI @
//! `96784b7` `plasticity.py:63-358`, MIT).
//!
//! | Gate | Instrument | Where |
//! |---|---|---|
//! | G1 | Signal-series known answers (plateau/trend/asymmetry/jump confirm+discard/window-edge absence/guards/determinism/save-restore/clamps) | in-module `#[cfg(test)]` suite (run via the `katgpt-core:<n>:rate_control` test-gate row) |
//! | G2 | ns/observe ceiling | THIS bench |
//! | G3 | Additive leaf: default-feature surface unchanged (measured at landing) | test_gate default row |
//! | G4 | Alloc-free observe | in-module debug-assertion suite over the lib test binary's TrackingAllocator |
//!
//! Issue 855 law: the timed loop consumes its result through `black_box`
//! into a checksum returned to the harness. Issue 723/831 discipline:
//! absolute ceiling, best-of-3 (never a ratio of sequential arms).

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::rate_control::RateController;

/// Minimal deterministic xorshift — the benches/tests house pattern.
struct SimpleLcg(u64);
impl SimpleLcg {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn signed(&mut self, scale: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * scale
    }
}

fn main() {
    const ITERS: u64 = 100_000;
    let mut rng = SimpleLcg::new(0x875B);

    let mut best = f64::INFINITY;
    let mut final_sink = 0u64;
    for _ in 0..3 {
        let mut c = RateController::new();
        let mut rng_run = SimpleLcg::new(0x875B);
        let t = Instant::now();
        let mut sink = 0u64;
        for i in 0..ITERS {
            let y = 0.001 * i as f32 + rng_run.signed(0.2);
            if c.observe(black_box(y), black_box(0.1)).is_some() {
                sink += 1;
            }
            sink = sink.wrapping_add(c.factor().to_bits() as u64);
        }
        sink = sink.wrapping_add(c.factor().to_bits() as u64);
        let elapsed = t.elapsed().as_nanos() as f64;
        black_box(sink);
        final_sink = sink;
        if elapsed < best {
            best = elapsed;
        }
    }
    let per_observe = best / ITERS as f64;
    let _ = rng.next_f32(); // keep the helper honest (both methods used at compile time)

    println!("bench_875 — rate_control GOAT (Issue 873 primitive B / Research 581)");
    println!();
    println!(
        "[B rate_ctrl ] {per_observe:9.1} ns/observe (dual EWLS + tanh/exp nudge + jump detect)"
    );
    println!("  sink = {final_sink} (Issue 855: the timed work is consumed)");
    println!();
    if per_observe < 1_000.0 {
        println!("✅ G2 PASS");
        println!();
        println!("NOT PROMOTED to default — the Issue-873 C4 decision: no consumer-");
        println!("  measured gain exists (the stack-slot rule). First consumer A/B is");
        println!("  riir-train Plan 416 Phase 2 (vs cosine at fixed budget + regime-");
        println!("  change arm). Constants pinned (Issue-033 law); report-first.");
        std::process::exit(0);
    } else {
        println!("⛔ G2 FAIL: ns/observe ceiling is 1000");
        std::process::exit(1);
    }
}

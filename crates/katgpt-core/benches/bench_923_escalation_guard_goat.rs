//! Issue 923 — the escalation_guard GOAT gate (G2 ns/observe).
//!
//! | Gate | Instrument | Where |
//! |---|---|---|
//! | G1 | Pinned-semantics suite (rate-None-until-full, latch reasons + stickiness, window roll, kill-switch truth table incl. the tpr polarity cross-pin, ESC receipt bytes, bounds grammar, constructor panics) | in-module `#[cfg(test)]` suite (run via the `katgpt-core:<n>:escalation_guard` test-gate row) |
//! | G2 | ns/observe absolute ceiling | THIS bench |
//! | G3 | Additive leaf: default-feature surface unchanged (opt-in feature; measured at landing) | test_gate default row |
//! | G4 | Alloc-free observe (post-construction) | in-module debug-assertion suite over the lib test binary's TrackingAllocator |
//!
//! Issue 855 law: the timed loop consumes its result through `black_box`
//! into a checksum returned to the harness. Issue 723/831 discipline:
//! absolute ceiling, best-of-3 (never a ratio of sequential arms).

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::escalation_guard::RollingRateLatch;

/// Minimal deterministic xorshift — the benches/tests house pattern.
struct SimpleLcg(u64);
impl SimpleLcg {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn main() {
    const ITERS: u64 = 1_000_000;
    let mut final_sink = 0u64;
    let mut best = f64::INFINITY;

    for _ in 0..3 {
        let mut g = RollingRateLatch::new(0.15, 0.60);
        // Pre-fill so every timed observe runs the ROLLING path (the
        // steady-state cost — the fill path is strictly cheaper).
        for i in 0..200 {
            g.observe(i % 2 == 0);
        }
        let mut rng = SimpleLcg(0x923_5EED);
        let mut sink = 0u64;
        let t = Instant::now();
        for _ in 0..ITERS {
            let escalated = (rng.next_u64() & 1) == 1;
            g.observe(black_box(escalated));
            if g.latched().is_some() {
                sink += 1;
            }
            sink = sink.wrapping_add(g.rate().map(f64::to_bits).unwrap_or(0));
        }
        let elapsed = t.elapsed().as_nanos() as f64;
        black_box(sink);
        final_sink = sink;
        if elapsed < best {
            best = elapsed;
        }
    }
    let per_observe = best / ITERS as f64;

    println!("bench_923 — escalation_guard GOAT (Issue 923 / riir-refine Plan 202 R1)");
    println!();
    println!(
        "[B esc_guard] {per_observe:9.2} ns/observe (rolling-window replace + count + latch evaluate)"
    );
    println!("  sink = {final_sink} (Issue 855: the timed work is consumed)");
    println!();
    if per_observe < 20.0 {
        println!("✅ G2 PASS (ceiling 20 ns/observe)");
        println!();
        println!("NOT PROMOTED to default — Feature Flag Discipline: promotion needs a");
        println!("  consumer-measured gain. First consumers: riir-refine's Plan-202 R1");
        println!("  escalation manifest (cost-ceiling shape) and the instinct/rethink");
        println!("  migration onto the shared receipt tiers.");
        std::process::exit(0);
    } else {
        println!("⛔ G2 FAIL: ns/observe ceiling is 20");
        std::process::exit(1);
    }
}

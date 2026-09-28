//! DefaultFaithfulnessProbe benchmark (Plan 278 Phase 2, T2.9).
//!
//! Measures the full intervention suite cost at **audit cadence** (every N
//! ticks, NOT per-tick). Target: **<1ms per segment** per Plan 278 §Expected
//! Performance.
//!
//! Follows the existing katgpt-rs bench convention (`std::time::Instant`,
//! `harness = false`, `fn main()`) — criterion is not a katgpt-rs dev-dep.
//!
//! Run with:
//! ```bash
//! cargo run --release --bench faithfulness_probe_bench --features faithfulness_probe
//! ```

use fastrand::Rng;
use katgpt_core::faithfulness::probe::{DefaultFaithfulnessProbe, FaithfulnessProbe};
use katgpt_core::faithfulness::types::ConsumerContext;

/// Synthetic faithful consumer: behavior = weighted dot product.
/// Mirrors the faithful consumer from probe.rs unit tests.
struct BenchConsumer {
    weights: Vec<f32>,
}

impl ConsumerContext for BenchConsumer {
    type Behavior = f32;
    type Delta = f32;
    type Memory = Vec<f32>;

    fn baseline_behavior(&self) -> f32 {
        0.0
    }

    fn behavior_with_memory(&self, memory: &Vec<f32>) -> f32 {
        memory
            .iter()
            .zip(self.weights.iter())
            .map(|(&v, &w)| v * w)
            .sum()
    }

    fn behavior_delta(&self, a: &f32, b: &f32) -> f32 {
        (a - b).abs()
    }
}

fn main() {
    println!("=== DefaultFaithfulnessProbe Benchmark (Plan 278 T2.9) ===\n");
    println!("Target: <1ms per `faithfulness_profile` call (audit cadence).\n");

    println!(
        "{:>6} {:>14} {:>14} {:>14}",
        "n_dim", "us/call", "verdict", "is_faithful"
    );

    for &n in &[16_usize, 64, 256, 1024, 4096] {
        // Position-dependent weights.
        let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let memory: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let irrelevant_pool: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];

        let consumer = BenchConsumer { weights };
        let mut probe = DefaultFaithfulnessProbe::new(consumer, irrelevant_pool, 1.0_f32);

        // Warm up.
        let mut rng = Rng::with_seed(42);
        let _ = probe.faithfulness_profile(&memory, &mut rng);

        // Timed: 100 calls, report mean.
        let n_iters = 100;
        let start = std::time::Instant::now();
        let mut last_verdict = false;
        for _ in 0..n_iters {
            let profile = probe.faithfulness_profile(&memory, &mut rng);
            last_verdict = profile.is_faithfully_used(0.5);
        }
        let elapsed = start.elapsed();
        let us_per_call = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

        let verdict = if us_per_call < 1000.0 {
            "PASS ✅"
        } else {
            "FAIL ❌ (over 1ms)"
        };
        println!("{n:>6} {us_per_call:>14.2} {verdict:>14} {last_verdict:>14}");
    }

    println!("\nNote: this is audit-cadence cost (every N ticks), NOT hot-path.");
    println!("The hot-path gate benchmark lives in `triggered_injection_bench.rs`.");

    // ── Issue 776 (Research 555 / CVRR) — contrastive intervention cost rows.
    // Same audit-cadence target (<1ms per segment); the two new probe methods
    // each cost one copy + one perturb + one consumer call (no rng for the
    // swap; Box-Muller per element for the noise).
    println!("\n=== Issue 776 contrastive interventions (audit cadence) ===\n");
    println!(
        "{:>6} {:>18} {:>18} {:>14} {:>14}",
        "n_dim", "us/matched_swap", "us/norm_noise", "swap PASS", "noise PASS"
    );

    for &n in &[16_usize, 64, 256, 1024, 4096] {
        let weights: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let memory: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let donor: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.5).collect();

        let consumer = BenchConsumer { weights };
        let mut probe = DefaultFaithfulnessProbe::new(consumer, vec![0.1_f32], 1.0_f32);

        // Warm up.
        let mut rng = Rng::with_seed(42);
        let mut scratch = vec![0.0_f32; n];
        let _ = probe.probe_matched_swap_into(&memory, &donor, &mut scratch);
        let _ = probe.probe_norm_noise_into(&memory, &mut scratch, &mut rng);

        let n_iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..n_iters {
            let _ = probe.probe_matched_swap_into(&memory, &donor, &mut scratch);
        }
        let us_swap = start.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

        let start = std::time::Instant::now();
        for _ in 0..n_iters {
            let _ = probe.probe_norm_noise_into(&memory, &mut scratch, &mut rng);
        }
        let us_noise = start.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

        let (swap_v, noise_v) = if us_swap < 1000.0 && us_noise < 1000.0 {
            ("PASS ✅", "PASS ✅")
        } else {
            ("FAIL ❌", "FAIL ❌")
        };
        println!("{n:>6} {us_swap:>18.2} {us_noise:>18.2} {swap_v:>14} {noise_v:>14}");
    }
}

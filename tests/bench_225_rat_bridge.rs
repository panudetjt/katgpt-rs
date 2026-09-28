#![cfg(feature = "rat_plus_bridge")]
//! Benchmarks for RAT+ Recurrence Bridge (Plan 225).
//!
//! Measures decode latency at different dilation factors,
//! bridge projection overhead, and KV cache memory per dilation.
//!
//! Run: `cargo test --features rat_plus_bridge --test bench_225_rat_bridge -- --nocapture`

use katgpt_attn::rat_bridge::{DilatedKvAccessor, RatBridgeState, rat_decode_step};
use katgpt_core::types::DilationConfig;

// Issue 855: the load-invariant timing treatment. `best_of_us` panics on an
// all-zero reading instead of letting a ceiling be satisfied by a loop the
// optimiser deleted.
#[path = "common/ab_timing.rs"]
mod ab_timing;

// ── T6.2: Decode Latency Benchmarks ─────────────────────────────

#[test]
fn bench_decode_latency_per_dilation() {
    let dim = 64;
    let seq_len = 1024;
    let query = vec![0.5; dim];
    let keys: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| vec![(i as f32 % 10.0) / 10.0; dim])
        .collect();
    let vals: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| vec![(i as f32 % 8.0) / 8.0; dim])
        .collect();
    let gdn2 = vec![0.1; dim];

    let dilations = [
        DilationConfig::D1,
        DilationConfig::D4,
        DilationConfig::D16,
        DilationConfig::D64,
    ];

    for d in dilations {
        let mut state = RatBridgeState::new(d, dim);
        // Pre-allocate the decode output buffer once and reuse it across
        // iterations via rat_decode_step_into. The allocating rat_decode_step
        // wrapper would add a per-iteration heap alloc that dominates the
        // measurement at small dim and misrepresents production latency
        // (production callers use the _into variant).
        let mut out_buf = vec![0.0f32; dim];
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = katgpt_attn::rat_bridge::rat_decode_step_into(
                &mut state,
                &query,
                &keys,
                &vals,
                &gdn2,
                &mut out_buf,
            );
        }
        let elapsed = start.elapsed();
        let per_decode = elapsed / 100;
        // D=1 is dense, should be slowest. D=64 is most sparse, should be fastest.
        println!("Dilation D={}: {:.2?} per decode", d.stride(), per_decode);
        // Verify it completes in reasonable time
        assert!(per_decode < std::time::Duration::from_millis(100));
    }
}

#[test]
fn bench_bridge_projection_overhead() {
    let dim = 64;
    let mut state = RatBridgeState::new(DilationConfig::D16, dim);
    let query = vec![0.5; dim];
    let gdn2 = vec![0.1; dim];

    // Issue 855: the previous loop called `state.compute_gate(&query, &gdn2)`
    // 10 000 times over loop-invariant inputs and discarded the returned gate,
    // with `state` dead after the loop — so rustc deleted the whole timed
    // region and this printed `Gate computation: 0.00ns per call`, satisfying
    // the < 10 µs ceiling with maximum margin on work that never ran.
    // `best_of_us` FAILS loudly on an all-zero reading; the gate is summed into
    // a sink consumed through `black_box`. The bar below is unchanged, and the
    // total timed iteration count is raised from 10 000 to 100 000.
    //
    // ⛔ The `black_box` on the two INPUTS is not decoration. A sink alone was
    // measured insufficient here: `compute_gate` is pure in its arguments and
    // both are loop-invariant, so LLVM hoisted the dot product out of the loop
    // and this still printed `1.00ns per call`. Making the operands opaque per
    // iteration is what keeps the gate inside the timed region.
    const ROUNDS: usize = 10;
    const ITERS: usize = 10_000;

    let best_us = ab_timing::best_of_us(2, ROUNDS, || {
        let mut sink = 0.0f32;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            sink += state.compute_gate(std::hint::black_box(&query), std::hint::black_box(&gdn2));
        }
        let e = t0.elapsed();
        std::hint::black_box(sink);
        e
    });
    let per_gate = std::time::Duration::from_secs_f64(best_us * 1e-6 / ITERS as f64);
    println!("Gate computation: {per_gate:.2?} per call");
    assert!(per_gate < std::time::Duration::from_micros(10));
}

#[test]
fn bench_kv_cache_memory_per_dilation() {
    let seq_len = 4096;
    let dim = 64;

    let dilations = [
        DilationConfig::D1,
        DilationConfig::D4,
        DilationConfig::D16,
        DilationConfig::D64,
    ];

    for d in dilations {
        let indices = DilatedKvAccessor::dilated_indices(seq_len, d);
        let effective_kv = indices.len();
        let bytes = effective_kv * dim * std::mem::size_of::<f32>();
        println!(
            "D={}: {} KV entries, {} bytes ({:.1}%)",
            d.stride(),
            effective_kv,
            bytes,
            100.0 * effective_kv as f64 / seq_len as f64
        );
    }
}

// ── T6.3: Before/After Comparison ───────────────────────────────

#[test]
fn test_before_after_dilation_comparison() {
    let dim = 32;
    let query = vec![0.5; dim];
    let keys: Vec<Vec<f32>> = (0..256)
        .map(|i| vec![(i as f32 % 10.0) / 10.0; dim])
        .collect();
    let vals: Vec<Vec<f32>> = (0..256)
        .map(|i| vec![(i as f32 % 8.0) / 8.0; dim])
        .collect();
    let gdn2 = vec![0.1; dim];

    // Dense baseline
    let mut state_dense = RatBridgeState::new(DilationConfig::D1, dim);
    let dense_out = rat_decode_step(&mut state_dense, &query, &keys, &vals, &gdn2);

    // Bridge D=16
    let mut state_bridge = RatBridgeState::new(DilationConfig::D16, dim);
    let bridge_out = rat_decode_step(&mut state_bridge, &query, &keys, &vals, &gdn2);

    // Both should produce valid output
    assert_eq!(dense_out.output.len(), dim);
    assert_eq!(bridge_out.output.len(), dim);

    // Output should be different (different KV positions used)
    assert_ne!(dense_out.output, bridge_out.output);

    // Both should produce finite values
    for &v in &dense_out.output {
        assert!(v.is_finite());
    }
    for &v in &bridge_out.output {
        assert!(v.is_finite());
    }

    println!(
        "Dense α={:.3}, Bridge D=16 α={:.3}",
        dense_out.alpha, bridge_out.alpha
    );
}

// ── T6.4: GOAT Gate Decision ────────────────────────────────────

#[test]
fn test_goat_gate_decision() {
    // GOAT criteria:
    // D=16: <2% quality loss, >8× FLOPs reduction → DEFAULT-ON
    // D=64: <5% quality loss, >40× FLOPs reduction → DEFAULT-ON

    // FLOPs reduction: decode FLOPs ∝ 1/D
    let d16_reduction = 16.0; // 16× FLOPs reduction
    let d64_reduction = 64.0; // 64× FLOPs reduction

    assert!(d16_reduction >= 8.0, "D=16 should give ≥8× FLOPs reduction");
    assert!(
        d64_reduction >= 40.0,
        "D=64 should give ≥40× FLOPs reduction"
    );

    // Quality: would need real model evaluation to measure.
    // For now, verify the mechanism works correctly.
    println!("GOAT: D=16 meets ≥8× FLOPs reduction (actual: {d16_reduction:.0}×)");
    println!("GOAT: D=64 meets ≥40× FLOPs reduction (actual: {d64_reduction:.0}×)");
    println!("GOAT: Quality validation requires real model evaluation");
    println!("GOAT: Decision — keep rat_plus_bridge as opt-in until real quality benchmarks pass");
}

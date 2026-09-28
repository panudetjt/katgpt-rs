#![cfg(all(feature = "gdn2_attention", feature = "hla_attention"))]
//! GOAT Benchmark Test — Gated DeltaNet-2 Recurrent Attention (Plan 105)
//!
//! Validates all 6 success criteria from Plan 105:
//! 1. All unit tests pass (including GQA variant) — verified by `cargo test`
//! 2. GDN2 within 10% of AHLA throughput
//! 3. GDN2 memory < flat KV memory at all configs
//! 4. No NaN/Inf in logits at any position
//! 5. Gate ablation: EraseOnly within 5% of Full quality (cosine sim)
//! 6. Context scaling: flat throughput profile (O(1) per step)
//!
//! Run: `cargo test --features "gdn2_attention,hla_attention" --test bench_105_gdn2_goat -- --nocapture`

#[path = "common/ab_timing.rs"]
mod ab_timing;

use std::hint::black_box;
use std::time::Instant;

use katgpt_rs::gdn2::{Gdn2GateConfig, MultiLayerGdn2Cache, forward_gdn2, generate_gdn2_into};
use katgpt_rs::hla::{MultiLayerAhlaCache, forward_ahla};
use katgpt_rs::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use katgpt_rs::types::{Config, Rng, kv_dim};

const WARMUP: usize = 50;
const ITERS: usize = 5000;
const POSITIONS: usize = 8;
/// Interleaved rounds for GOAT 2 (Issue 833). ITERS splits across these.
///
/// Raised 500 -> 5000 on the measurement, not on taste: at the original 500 the
/// per-round chunk was ~153 us and the printed per-round RANGE came back
/// 0.73 .. 2.99 -- a 4x swing around a 1.0028 median. The median was already
/// doing its job, but a bar read off rounds that noisy is one preemption away
/// from being interesting, and the whole target costs 0.01 s. At 200 iters per
/// round each chunk is ~1.5 ms and the range closes. The RANGE is the quantity
/// that told us this; a median alone would have looked finished.
const ROUNDS: usize = 25;

// ── Helpers ───────────────────────────────────────────────────

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ── Criterion 2: GDN2 within 10% of AHLA throughput ──────────

#[test]
fn goat_2_gdn2_within_10pct_of_ahla_throughput() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // ⛔ This criterion used to time GDN2 to completion, then AHLA to
    // completion, and assert on the single resulting ratio. That is Issue 723
    // **Class A** — a gate whose verdict the BOX decides rather than the code —
    // and it is exactly how the 4090's `scripts/x86_64_execution_matrix.sh`
    // caught it on 2026-09-18: FAILED in cell 8, then PASSED 3/3 when re-run
    // alone. Two sequential arms of the same work measured +5.2% and +21.7%
    // thirty seconds apart on a loaded box (Issue 723 T5), and a 10% bar cannot
    // survive that.
    //
    // `common/ab_timing.rs` is the treatment AGENTS.md already prescribes, and
    // this target was simply missed by the Issue-723 census (Issue 833):
    // interleaved `(a-chunk, b-chunk)` pairs, so a load drift moves BOTH arms
    // and cancels in the ratio instead of landing entirely on whichever arm ran
    // second, and the MEDIAN across pairs, so one preemption spike is discarded.
    //
    // ⚠ `a` is AHLA — the baseline this criterion is *stated against* — and `b`
    // is GDN2, so `ab.median` is a **time** ratio gdn2/ahla and the criterion's
    // **throughput** ratio is its reciprocal. Getting that backwards inverts the
    // gate silently, which is the defect riir-ai Issue 977 was filed for one
    // instrument over.
    let mut ctx_ahla = ForwardContext::new(&config);
    let mut cache_ahla = MultiLayerAhlaCache::new(&config);
    let mut ctx_gdn2 = ForwardContext::new(&config);
    let mut cache_gdn2 = MultiLayerGdn2Cache::new(&config);

    let iters_per_round = ITERS / ROUNDS;

    let ab = ab_timing::ab_median_ratio(
        ROUNDS,
        iters_per_round,
        WARMUP,
        // a = BASELINE: AHLA, one cache pass over POSITIONS.
        |_i| {
            cache_ahla.reset();
            for pos in 0..POSITIONS {
                black_box(forward_ahla(
                    &mut ctx_ahla,
                    &weights,
                    &mut cache_ahla,
                    0,
                    pos,
                    &config,
                ));
            }
        },
        // b = CANDIDATE: GDN2, the same shape.
        |_i| {
            cache_gdn2.reset();
            for pos in 0..POSITIONS {
                black_box(forward_gdn2(
                    &mut ctx_gdn2,
                    &weights,
                    &mut cache_gdn2,
                    0,
                    pos,
                    &config,
                ));
            }
        },
    );

    // ns-per-ITER covers POSITIONS steps, so per-step is that over POSITIONS.
    let ahla_us = ab.a_ns_per_iter() / POSITIONS as f64 / 1000.0;
    let gdn2_us = ab.b_ns_per_iter() / POSITIONS as f64 / 1000.0;
    let ahla_tps = 1e6 / ahla_us;
    let gdn2_tps = 1e6 / gdn2_us;
    let ratio = 1.0 / ab.median;

    println!();
    println!(
        "┌── GOAT 2: GDN2 vs AHLA Throughput (micro, {ROUNDS}×{iters_per_round}×{POSITIONS} pos, interleaved) ──┐"
    );
    println!("│ {:<18} {:>10} {:>12} │", "Method", "tok/s", "µs/step");
    println!("│ {} │", "-".repeat(42));
    println!("│ {:<18} {:>10.1} {:>12.2} │", "GDN2", gdn2_tps, gdn2_us);
    println!("│ {:<18} {:>10.1} {:>12.2} │", "AHLA", ahla_tps, ahla_us);
    println!(
        "│ {:<18} {:>10.1}%{:>13} │",
        "GDN2/AHLA ratio",
        ratio * 100.0,
        ""
    );
    println!("└{}┘", "─".repeat(45));
    // The per-round RANGE, not just the median: a median inside 0.98..1.02 and
    // a median inside 0.6..1.7 are not the same claim at the same number.
    ab.report("GOAT 2 ahla(a)-vs-gdn2(b), time ratio");

    assert!(
        ratio >= 0.90,
        "GDN2 throughput ({gdn2_tps:.1} tok/s) must be within 10% of AHLA ({ahla_tps:.1} tok/s), \
         got {ratio:.3} (median of {} interleaved rounds, per-round time ratio {:.4} .. {:.4})",
        ab.ratios.len(),
        ab.min(),
        ab.max(),
    );
    println!("  ✅ GOAT 2 PASSED: GDN2/AHLA = {ratio:.3} (≥ 0.90)");
}

// ── Criterion 3: GDN2 memory < flat KV memory at all configs ──

#[test]
fn goat_3_gdn2_memory_less_than_flat_kv() {
    let configs: [(&str, Config); 4] = [
        ("micro", Config::micro()),
        ("game", Config::game()),
        ("bpe", Config::bpe()),
        ("gqa_draft", Config::gqa_draft()),
    ];

    println!();
    println!("┌── GOAT 3: GDN2 Memory vs Flat KV ─────────────────────┐");
    println!(
        "│ {:<12} {:>10} {:>10} {:>8} │",
        "Config", "Flat KV", "GDN2", "Saved"
    );
    println!("│ {} │", "-".repeat(44));

    for (name, cfg) in &configs {
        let kvd = kv_dim(cfg);
        let flat_bytes = cfg.block_size * kvd * 2 * 4; // key + value, f32
        let gdn2_bytes = MultiLayerGdn2Cache::new(cfg).memory_bytes();
        let saved = (1.0 - gdn2_bytes as f64 / flat_bytes as f64) * 100.0;

        println!("│ {name:<12} {flat_bytes:>8} B {gdn2_bytes:>8} B {saved:>6.1}% │");

        assert!(
            gdn2_bytes < flat_bytes,
            "GDN2 ({gdn2_bytes} B) must be < flat KV ({flat_bytes} B) for {name}"
        );
    }
    println!("└{}┘", "─".repeat(47));
    println!("  ✅ GOAT 3 PASSED: GDN2 memory < flat KV at all configs");
}

// ── Criterion 4: No NaN/Inf in logits at any position ────────

#[test]
fn goat_4_no_nan_inf_at_any_position() {
    let configs: [(&str, Config); 4] = [
        ("micro", Config::micro()),
        ("game", Config::game()),
        ("bpe", Config::bpe()),
        ("gqa_draft", Config::gqa_draft()),
    ];

    let positions: [usize; 5] = [0, 8, 64, 128, 255];

    println!();
    println!("┌── GOAT 4: No NaN/Inf in Logits ───────────────────────┐");

    for (cfg_name, config) in &configs {
        let max_pos = config.block_size - 1;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(config, &mut rng);
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerGdn2Cache::new(config);

        for &pos in &positions {
            if pos > max_pos {
                continue;
            }

            let logits = forward_gdn2(
                &mut ctx,
                &weights,
                &mut cache,
                config.bos_token,
                pos,
                config,
            );

            assert!(
                logits.iter().all(|&l| l.is_finite()),
                "Non-finite logits at {cfg_name} pos={pos}"
            );
        }
        println!("  ✅ {cfg_name}: all positions finite (up to pos={max_pos})");
    }

    // Also test multi-token streaming generation with all gate configs
    for gate_config in [
        Gdn2GateConfig::EraseOnly,
        Gdn2GateConfig::Full,
        Gdn2GateConfig::Kda,
    ] {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::with_gate_config(&config, gate_config);
        let mut tokens = Vec::new();

        generate_gdn2_into(
            &mut ctx,
            &mut cache,
            &weights,
            &config,
            &mut rng,
            16,
            &mut tokens,
        );

        assert_eq!(
            tokens.len(),
            16,
            "Gate {gate_config:?}: should generate 16 tokens"
        );
        for &t in &tokens {
            assert!(
                t < config.vocab_size,
                "Token {t} out of vocab range for {gate_config:?}"
            );
        }
        println!("  ✅ {gate_config:?}: 16-token streaming generation stable");
    }

    println!("└{}┘", "─".repeat(56));
    println!("  ✅ GOAT 4 PASSED: No NaN/Inf in logits at any position or gate config");
}

// ── Criterion 5: Gate ablation — EraseOnly within 5% of Full ──

#[test]
fn goat_5_erase_only_within_5pct_of_full_quality() {
    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Compare final logits from a single forward pass (same input, different gate configs)
    let mut ctx_e = ForwardContext::new(&config);
    let mut cache_e = MultiLayerGdn2Cache::with_gate_config(&config, Gdn2GateConfig::EraseOnly);
    let logits_erase = forward_gdn2(
        &mut ctx_e,
        &weights,
        &mut cache_e,
        config.bos_token,
        0,
        &config,
    )
    .to_vec();

    let mut ctx_f = ForwardContext::new(&config);
    let mut cache_f = MultiLayerGdn2Cache::with_gate_config(&config, Gdn2GateConfig::Full);
    let logits_full = forward_gdn2(
        &mut ctx_f,
        &weights,
        &mut cache_f,
        config.bos_token,
        0,
        &config,
    )
    .to_vec();

    let cos_sim = cosine_sim(&logits_erase, &logits_full);

    println!();
    println!("┌── GOAT 5: Gate Ablation (cosine similarity) ──────────┐");
    println!("│ EraseOnly vs Full cosine sim: {cos_sim:.6}                   │");
    println!(
        "│ Threshold (1 - 0.05):         {threshold:.6}                   │",
        threshold = 0.95
    );
    println!("└{}┘", "─".repeat(57));

    assert!(
        cos_sim >= 0.95,
        "EraseOnly/Full cosine sim ({cos_sim:.4}) must be ≥ 0.95 (within 5%)"
    );
    println!("  ✅ GOAT 5 PASSED: EraseOnly/Full cosine sim = {cos_sim:.4} (≥ 0.95)");
}

// ── Criterion 6: Context scaling — flat throughput O(1) ───────
//
// Key insight: we measure ONLY the single decode step at target_pos,
// NOT the prefill. Prefill is O(N) for both methods (sequential token loop).
// The O(1) claim is about per-step decode cost:
//   - GDN2: O(dk × dv) regardless of position (recurrent state is fixed-size)
//   - Flat KV: O(N × dk) at position N (scans all N stored keys)

#[test]
fn goat_6_context_scaling_flat_o1() {
    // Use game() config — block_size=170, allows positions [0..169]
    let config = Config::game();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let single_step_iters = 2000;

    let positions: [usize; 4] = [1, 8, 64, 128];

    // ── GDN2 scaling (should be flat O(1) per step) ──
    // Issue 833's class, found by execution (x86_64 matrix cell 8, 2026-09-20:
    // spread 0.306 against the 0.30 bar in-cell, 3/3 alone) — the original
    // loop measured each position in ONE sequential window, so load drift
    // between windows landed on single positions. Treatment: the ab_timing
    // defenses composed for N arms — round-robin interleave (adjacent samples
    // share a load window) + the per-arm minimum (contention only ever adds
    // time). The GOAT 2 repair in this file, one arm-count over.
    const SPREAD_ROUNDS: usize = 5;

    let mut gdn2_states: Vec<(ForwardContext, MultiLayerGdn2Cache)> = Vec::new();
    for &target_pos in &positions {
        // Prefill once to reach target_pos
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::new(&config);
        for pos in 0..target_pos {
            black_box(forward_gdn2(
                &mut ctx, &weights, &mut cache, 0, pos, &config,
            ));
        }

        // Warmup single-step at target_pos
        for _ in 0..WARMUP {
            black_box(forward_gdn2(
                &mut ctx, &weights, &mut cache, 0, target_pos, &config,
            ));
        }
        gdn2_states.push((ctx, cache));
    }

    // Measure ONLY the single step at target_pos (each state keeps its
    // position's recurrent state resident across rounds — the per-call
    // semantics are unchanged, only the sampling order interleaves).
    let gdn2_us_per_step: Vec<f64> =
        ab_timing::best_of_arms(positions.len(), 1, SPREAD_ROUNDS, |arm| {
            let (ctx, cache) = &mut gdn2_states[arm];
            let target_pos = positions[arm];
            let start = Instant::now();
            for _ in 0..single_step_iters {
                black_box(forward_gdn2(ctx, &weights, cache, 0, target_pos, &config));
            }
            start.elapsed()
        });

    // ── Flat KV scaling (should grow linearly with position) ──
    let mut flat_us_per_step: Vec<f64> = Vec::new();
    for &target_pos in &positions {
        // Prefill once to reach target_pos
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        for pos in 0..target_pos {
            black_box(forward(&mut ctx, &weights, &mut cache, 0, pos, &config));
        }

        // Warmup single-step at target_pos
        for _ in 0..WARMUP.min(20) {
            black_box(forward(
                &mut ctx, &weights, &mut cache, 0, target_pos, &config,
            ));
        }

        // Measure ONLY the single step at target_pos (cache has N entries)
        let start = Instant::now();
        for _ in 0..single_step_iters {
            black_box(forward(
                &mut ctx, &weights, &mut cache, 0, target_pos, &config,
            ));
        }
        let elapsed = start.elapsed();
        let us_per_step = elapsed.as_micros() as f64 / single_step_iters as f64;
        flat_us_per_step.push(us_per_step);
    }

    // ── Variance analysis ──
    let gdn2_min = gdn2_us_per_step.iter().fold(f64::MAX, |a, &b| a.min(b));
    let gdn2_max = gdn2_us_per_step.iter().fold(0.0f64, |a, &b| a.max(b));
    let gdn2_mean: f64 = gdn2_us_per_step.iter().sum::<f64>() / gdn2_us_per_step.len() as f64;
    let gdn2_max_spread = (gdn2_max - gdn2_min) / gdn2_mean; // relative spread

    let flat_growth = flat_us_per_step.last().unwrap() / flat_us_per_step.first().unwrap();

    println!();
    println!("┌── GOAT 6: Context Scaling — Single Step Decode (game) ┐");
    println!(
        "│ {:<10} {:>14} {:>14} │",
        "Position", "GDN2 (µs)", "Flat KV (µs)"
    );
    println!("│ {} │", "-".repeat(42));
    for (i, &pos) in positions.iter().enumerate() {
        println!(
            "│ {:<10} {:>14.2} {:>14.2} │",
            pos, gdn2_us_per_step[i], flat_us_per_step[i]
        );
    }
    println!("│ {} │", "-".repeat(42));
    println!("│ GDN2  spread (max-min)/mean: {gdn2_max_spread:.3}                │");
    println!("│ Flat  growth (last/first):   {flat_growth:.2}x                │");
    println!("└{}┘", "─".repeat(46));

    // GDN2 single-step cost should be nearly constant — max spread < 30% of mean
    assert!(
        gdn2_max_spread < 0.30,
        "GDN2 single-step scaling not flat: spread={gdn2_max_spread:.3}, expected < 0.30"
    );

    // Flat KV single-step should grow (at least 1.5× from first to last)
    assert!(
        flat_growth > 1.5,
        "Flat KV should show O(N) growth, got {flat_growth:.2}× (expected > 1.5×)"
    );

    println!(
        "  ✅ GOAT 6 PASSED: GDN2 spread={gdn2_max_spread:.3} (< 0.30), Flat growth={flat_growth:.1}× (> 1.5×)"
    );
}

// ── Summary ───────────────────────────────────────────────────

#[test]
fn summary_goat_105_gdn2_benchmarks() {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  🐐 GOAT Benchmarks: Gated DeltaNet-2 (Plan 105)");
    println!("  Features: gdn2_attention + hla_attention");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  Criterion 1: All unit tests pass          — see `cargo test --features gdn2_attention`"
    );
    println!(
        "  Criterion 2: GDN2 within 10% of AHLA      — goat_2_gdn2_within_10pct_of_ahla_throughput"
    );
    println!("  Criterion 3: GDN2 mem < flat KV all config — goat_3_gdn2_memory_less_than_flat_kv");
    println!("  Criterion 4: No NaN/Inf in logits          — goat_4_no_nan_inf_at_any_position");
    println!(
        "  Criterion 5: EraseOnly within 5% of Full   — goat_5_erase_only_within_5pct_of_full_quality"
    );
    println!("  Criterion 6: Flat O(1) context scaling      — goat_6_context_scaling_flat_o1");
    println!();
    println!("  Run all: cargo test --features \"gdn2_attention,hla_attention\" \\");
    println!("             --test bench_105_gdn2_goat -- --nocapture");
    println!("═══════════════════════════════════════════════════════════════");
}

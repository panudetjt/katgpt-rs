//! Plan 600 G4 — the DBTM fill path's per-round allocation gate.
//!
//! The DBTM commit floor (Issue 811) added `round_candidates` to the fill
//! loop. G4 requires the fill path to be alloc-free ACROSS ROUNDS: the
//! candidate Vec must be pre-allocated (capacity = block size, the provable
//! per-round upper bound — one proposal per masked position) and reused via
//! the push/retain/clear cycle.
//!
//! The probe: with τ ≈ 1 the threshold commits nothing and the floor drives
//! every round, so budget 4 converges in exactly 4 rounds (2 tokens/round)
//! and budget 8 in exactly 8 (1 token/round) — 2× the rounds through the
//! identical per-call buffer set. Per-call allocations appear in BOTH
//! windows; a per-ROUND allocation appears only in the 8-round window. The
//! gate asserts the two allocation counts are EQUAL.
//!
//! Per-thread counting allocator (Issue 714): `cargo test` runs a binary's
//! tests on parallel threads, so process-global counters would measure this
//! gate's hot path plus whatever a sibling test happened to allocate. This
//! file has one test; the per-thread counter makes that safe by construction.
//!
//! # Run
//!
//! ```bash
//! cargo test --features flashar_anchor --test bench_600_flashar_fill_alloc_gate -- --nocapture
//! ```

#![cfg(feature = "flashar_anchor")]

#[path = "../crates/katgpt-core/tests/common/mod.rs"]
mod common;

counting_allocator!();

use katgpt_core::{Config, Rng};
use katgpt_forward::d2f::D2fDecodeConfig;
use katgpt_forward::{D2fContext, anchor_fill_with_prefilled};
use katgpt_rs::dllm::{generate_pattern_dataset, train_mini_dllm};
use katgpt_rs::transformer::TransformerWeights;
use std::sync::atomic::Ordering;

const BLOCK: usize = 8;

/// Trained just enough that proposals are real tokens — the alloc probe
/// exercises the full candidate path (push/sort/retain/clear per round)
/// regardless of model quality.
fn make_weights() -> (Config, TransformerWeights) {
    let config = Config::micro_dllm();
    let mut train_rng = Rng::new(123);
    let train_data =
        generate_pattern_dataset(&mut train_rng, 20, config.block_size, config.vocab_size - 1);
    let test_data =
        generate_pattern_dataset(&mut train_rng, 5, config.block_size, config.vocab_size - 1);
    let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 30, 0.01, 0.3, 42);
    (config, weights)
}

fn probe_config(budget: usize) -> D2fDecodeConfig {
    D2fDecodeConfig {
        denoise_steps: budget,
        // τ ≈ 1: threshold never commits — the DBTM floor alone drives every
        // round, which is exactly the code path G4 gates.
        confidence_threshold: 0.9999,
        block_size: BLOCK,
        ..D2fDecodeConfig::default()
    }
}

#[test]
fn g4_fill_round_loop_is_allocation_free() {
    assert_counter_is_live();
    let (config, weights) = make_weights();
    let mask = config.mask_token;
    let anchors = [mask; BLOCK];
    let mut dctx = D2fContext::new(&config);

    // Warmup both shapes once, outside the measured windows (deterministic:
    // identical seeds reproduce identical runs).
    for &budget in &[4usize, 8] {
        let mut rng = Rng::new(7);
        let r = anchor_fill_with_prefilled(
            &mut dctx,
            &weights,
            &config,
            &probe_config(budget),
            &anchors,
            &mut rng,
            Some(budget),
        );
        assert_eq!(
            r.steps_used, budget,
            "probe shape broken at budget {budget}: the floor must drive every round (steps_used {})",
            r.steps_used
        );
    }

    let mut rng4 = Rng::new(7);
    let before4 = ALLOC_COUNT.load(Ordering::Relaxed);
    let r4 = anchor_fill_with_prefilled(
        &mut dctx,
        &weights,
        &config,
        &probe_config(4),
        &anchors,
        &mut rng4,
        Some(4),
    );
    let allocs4 = ALLOC_COUNT.load(Ordering::Relaxed) - before4;

    let mut rng8 = Rng::new(7);
    let before8 = ALLOC_COUNT.load(Ordering::Relaxed);
    let r8 = anchor_fill_with_prefilled(
        &mut dctx,
        &weights,
        &config,
        &probe_config(8),
        &anchors,
        &mut rng8,
        Some(8),
    );
    let allocs8 = ALLOC_COUNT.load(Ordering::Relaxed) - before8;

    assert_eq!(
        r4.steps_used, 4,
        "budget-4 window must run exactly 4 rounds"
    );
    assert_eq!(
        r8.steps_used, 8,
        "budget-8 window must run exactly 8 rounds"
    );
    println!(
        "G4: budget=4 ({} rounds) → {allocs4} allocs | budget=8 ({} rounds) → {allocs8} allocs",
        r4.steps_used, r8.steps_used
    );
    assert_eq!(
        allocs4, allocs8,
        "per-round allocation detected in the DBTM fill path: 8 rounds allocated {allocs8} vs 4 rounds {allocs4}"
    );

    // Informational: the production config (τ=0.9, floor=8) per-call count.
    let prod = D2fDecodeConfig {
        denoise_steps: 8,
        confidence_threshold: 0.9,
        block_size: BLOCK,
        ..D2fDecodeConfig::default()
    };
    let mut rng = Rng::new(7);
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let r = anchor_fill_with_prefilled(
        &mut dctx,
        &weights,
        &config,
        &prod,
        &anchors,
        &mut rng,
        Some(8),
    );
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    println!(
        "G4 (info): production config (τ=0.9, floor=8, {} rounds) → {allocs} allocs/call",
        r.steps_used
    );
}

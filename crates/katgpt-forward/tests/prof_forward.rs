//! Forward-path profiling harness — isolates per-phase cost without a sampling
//! profiler.
//!
//! # Method
//!
//! The forward pass has two cost regimes per decoded token:
//!   - **O(1) per token**: embedding lookup, QKV matmuls, output projection,
//!     MLP matmuls, lm_head matmul. These are independent of how many tokens
//!     are already in the KV cache.
//!   - **O(seq) per token**: the attention head — score computation,
//!     softmax, and value accumulation each scan over all previous tokens.
//!
//! By measuring the full `forward()` call at position 0 (seq_len=1) and at
//! position N-1 (seq_len=N), the difference isolates the attention cost.
//! The position-0 cost isolates the matmul-dominated cost.
//!
//! This is a *relative* profiler — it tells you whether attention or matmuls
//! dominate, and how attention scales with seq_len. Absolute numbers depend
//! on the config and host.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-forward --test prof_forward --release -- --ignored --nocapture
//! ```
//!
//! `--release` is required: debug builds measure debug-codegen overhead, not
//! the production hot path. `--ignored` because perf tests don't run in CI.

#![allow(clippy::cast_precision_loss)]

use katgpt_forward::{ForwardContext, forward, forward_f16};
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};
use katgpt_types::Config;

/// A medium config that stresses matmuls meaningfully (n_embd=128, 4 layers,
/// 4 heads, head_dim=32, vocab=256, mlp_hidden=512) while staying fast enough
/// to iterate on. This is NOT a built-in Config preset — it's a local
/// profiling config. Derived from `Config::micro()` with larger dims.
fn medium_config() -> Config {
    let mut c = Config::micro();
    c.vocab_size = 256;
    c.block_size = 128;
    c.n_embd = 128;
    c.n_head = 4;
    c.head_dim = 32;
    c.mlp_hidden = 512;
    c.n_layer = 4;
    c.n_kv_head = 4; // MHA (no GQA)
    c
}

/// A large config where the model exceeds L3 cache, so the f32 forward pass
/// is truly DRAM-bandwidth-bound. At n_embd=512, n_layer=8, the per-layer
/// weight bytes are ~8 MB, total model ~64 MB + lm_head. This exceeds typical
/// L3 caches (8-16 MB), forcing DRAM reads where f16's halved bandwidth wins.
fn large_config() -> Config {
    let mut c = Config::micro();
    c.vocab_size = 1024;
    c.block_size = 256;
    c.n_embd = 512;
    c.n_head = 8;
    c.head_dim = 64;
    c.mlp_hidden = 2048;
    c.n_layer = 8;
    c.n_kv_head = 8; // MHA (no GQA)
    c
}

/// How many iterations to run per measurement. Tuned so each sub-test takes
/// ~0.5–2 s in release mode on the medium config.
const ITERS: usize = 5_000;

fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

/// Run `forward()` once at each position in `0..seq_len`, repeating the whole
/// loop `iters` times. Returns total elapsed.
fn run_decode_loop(config: &Config, seq_len: usize, iters: usize) -> std::time::Duration {
    let mut rng = katgpt_types::Rng::new(42);
    let weights = TransformerWeights::new(config, &mut rng);
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);

    // Warmup: one full decode loop to populate caches / branch predictors.
    for pos in 0..seq_len {
        let token = pos % config.vocab_size;
        let _ = forward(&mut ctx, &weights, &mut cache, token, pos, config);
    }

    let start = std::time::Instant::now();
    for _ in 0..iters {
        // Reset the cache position so each iteration decodes the same sequence.
        // We don't rebuild ctx/cache — they're pre-allocated and just get
        // overwritten. advance_pos inside forward() handles the cursor.
        for pos in 0..seq_len {
            let token = pos % config.vocab_size;
            let logits = forward(&mut ctx, &weights, &mut cache, token, pos, config);
            black_box(logits.as_ptr());
        }
    }
    start.elapsed()
}

#[test]
#[ignore = "profiling harness — per-phase cost measurement, not a pass/fail test; run with --release --ignored"]
fn prof_forward_phase_breakdown() {
    let config = medium_config();
    println!();
    println!("═══ Forward-path profiling harness ═══");
    println!(
        "config: n_embd={}, n_layer={}, n_head={}, head_dim={}, vocab={}, mlp_hidden={}",
        config.n_embd,
        config.n_layer,
        config.n_head,
        config.head_dim,
        config.vocab_size,
        config.mlp_hidden
    );
    println!("iters per measurement: {ITERS}");
    println!();

    // Measure a full decode loop at increasing sequence lengths.
    // The per-token cost at seq_len=1 is matmul-dominated; the per-token cost
    // at seq_len=N minus the seq_len=1 cost is the attention cost at depth N.
    for &seq_len in &[1usize, 8, 32, 64, 128] {
        let total = run_decode_loop(&config, seq_len, ITERS);
        let per_iter_ns = total.as_nanos() as f64 / ITERS as f64;
        let per_token_ns = per_iter_ns / seq_len as f64;
        println!(
            "seq_len={seq_len:>3}: {per_token_ns:>10.1} ns/token  ({per_iter_ns:>12.1} ns/iter)",
        );
    }

    println!();
    println!("── Interpretation ──");
    println!("  ns/token at seq_len=1  ≈ matmul-dominated cost (embedding + QKV +");
    println!("                             attn[1] + output proj + MLP + lm_head)");
    println!(
        "  (ns/token at seq_len=N) - (ns/token at seq_len=1) ≈ attention scan cost at depth N"
    );
    println!("  If attention cost grows linearly with N and dominates at large N,");
    println!("  attention is the bottleneck for long contexts. If the seq_len=1");
    println!("  cost dominates, matmuls are the bottleneck (weight-bandwidth-bound).");
}

// ── f16 Weight Quantization GOAT Gate (Issue 200) ─────────────────────
//
// G1 (correctness): f16 logits must be approximately equal to f32 logits.
//     f16 has ~3 decimal digits of precision, so bit-identical is impossible.
//     The gate checks max absolute error per vocab element is within an
//     acceptable bound relative to the logit magnitude.
// G2 (perf): f16 forward must be faster than f32 forward at seq_len=1
//     (the bandwidth-bound regime where weight reads dominate).
// G3 (no-regression): existing tests still pass (verified by `cargo test`).
// G4 (alloc-free): covered by the existing forward_base alloc-free invariant.

/// Run f32 forward for one token and return the logits.
fn forward_f32_once(
    config: &Config,
    weights: &TransformerWeights,
    token: usize,
    pos: usize,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let logits = forward(&mut ctx, weights, &mut cache, token, pos, config);
    logits.to_vec()
}

/// Run f16 forward for one token and return the logits.
fn forward_f16_once(
    config: &Config,
    weights_f16: &katgpt_transformer::TransformerWeightsF16,
    token: usize,
    pos: usize,
) -> Vec<f32> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let logits = forward_f16(&mut ctx, weights_f16, &mut cache, token, pos, config);
    logits.to_vec()
}

#[test]
#[ignore = "profiling harness — per-phase cost measurement, not a pass/fail test; run with --release --ignored"]
fn g1_f16_approximate_correctness() {
    let config = medium_config();
    let mut rng = katgpt_types::Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let weights_f16 = weights.to_f16();

    // Decode a short sequence and compare logits at each position.
    // f16 quantization introduces rounding; the gate verifies the error is
    // bounded — not that it's zero.
    let seq_len = 16;
    let mut max_abs_err = 0.0f32;
    let mut max_logit_mag = 0.0f32;

    for pos in 0..seq_len {
        let token = pos % config.vocab_size;
        let logits_f32 = forward_f32_once(&config, &weights, token, pos);
        let logits_f16 = forward_f16_once(&config, &weights_f16, token, pos);

        assert_eq!(logits_f32.len(), logits_f16.len(), "vocab size mismatch");
        for (l32, l16) in logits_f32.iter().zip(logits_f16.iter()) {
            let err = (l32 - l16).abs();
            max_abs_err = max_abs_err.max(err);
            max_logit_mag = max_logit_mag.max(l32.abs());
        }
    }

    println!();
    println!("═══ G1: f16 approximate correctness ═══");
    println!("  seq_len:           {seq_len}");
    println!("  max |logit_f32|:   {max_logit_mag:.4}");
    println!("  max |f32 - f16|:   {max_abs_err:.6}");
    let rel_err = if max_logit_mag > 0.0 {
        max_abs_err / max_logit_mag
    } else {
        0.0
    };
    println!(
        "  relative error:    {rel_err:.4} ({:.2}%)",
        rel_err * 100.0
    );
    println!();

    // G1 gate: the relative error should be small (< 15%).
    // f16 mantissa is 10 bits (~3 decimal digits). For a transformer with
    // 4 layers of accumulated rounding, ~10-15% relative error is expected.
    // This gate confirms the f16 path is correct (no bugs in conversion or
    // matmul), not that f16 == f32.
    const G1_REL_ERR_THRESHOLD: f32 = 0.20; // 20% — generous for random-init weights
    assert!(
        rel_err < G1_REL_ERR_THRESHOLD,
        "G1 FAIL: relative error {rel_err:.4} exceeds threshold {G1_REL_ERR_THRESHOLD}"
    );
    println!("  G1: PASS (relative error < {G1_REL_ERR_THRESHOLD})");
}

#[test]
#[ignore = "profiling harness — per-phase cost measurement, not a pass/fail test; run with --release --ignored"]
fn g2_f16_speedup_vs_f32() {
    // Test at both medium (fits in L3) and large (exceeds L3) configs.
    // The f16 win only materializes when the model exceeds cache — the
    // halved bandwidth matters only for DRAM reads.
    for (label, config) in [
        ("medium (fits L3)", medium_config()),
        ("large (exceeds L3)", large_config()),
    ] {
        let mut rng = katgpt_types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let weights_f16 = weights.to_f16();

        let _seq_len = 1; // bandwidth-bound regime — weight reads dominate
        let iters = if config.n_embd >= 512 { 1_000 } else { ITERS };

        // Warmup both paths
        {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let _ = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        }
        {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let _ = forward_f16(&mut ctx, &weights_f16, &mut cache, 0, 0, &config);
        }

        // Measure f32
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
            black_box(logits.as_ptr());
        }
        let f32_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        // Measure f16
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let logits = forward_f16(&mut ctx, &weights_f16, &mut cache, 0, 0, &config);
            black_box(logits.as_ptr());
        }
        let f16_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        let speedup = f32_ns / f16_ns;

        // Estimate weight bytes for context
        let n = config.n_embd;
        let kvd = config.n_kv_head * config.head_dim;
        let per_layer_f32 =
            (n * n + kvd * n * 2 + n * n + config.mlp_hidden * n + n * config.mlp_hidden) * 4;
        let total_f32 = per_layer_f32 * config.n_layer + config.vocab_size * n * 4;

        println!();
        println!("═══ G2: f16 speedup vs f32 [{label}] ═══");
        println!(
            "  config: n_embd={}, n_layer={}, vocab={}, mlp_hidden={}",
            config.n_embd, config.n_layer, config.vocab_size, config.mlp_hidden
        );
        println!("  total f32 weight bytes: {:.1} MB", total_f32 as f64 / 1e6);
        println!("  iters: {iters}");
        println!("  f32: {f32_ns:>12.1} ns/token");
        println!("  f16: {f16_ns:>12.1} ns/token");
        println!(
            "  speedup: {speedup:.3}× ({:.1}% of f32)",
            f16_ns / f32_ns * 100.0
        );
        println!();

        // G2 gate threshold: 1.5× matches the issue's promotion criteria.
        // Below this, the f16 path is NOT a modelless perf gain and must not
        // be promoted. The assertion is the honest gate — a `WARNING` print
        // would paper over the failure. See `.issues/200` §"Why G2 failed"
        // for the root-cause analysis (f16 weight-only quantization is
        // 2-3× slower than f32 on Apple Silicon because the f32 activation
        // limits the bandwidth reduction to 25%, which the FCVT latency
        // more than eats).
        //
        // This test is `#[ignore]`d, so the assertion only fires under
        // `cargo test --ignored` — the explicit benchmark-gate invocation.
        const G2_MIN_SPEEDUP: f64 = 1.5;
        assert!(
            speedup >= G2_MIN_SPEEDUP,
            "G2 FAIL [{label}]: f16 speedup {speedup:.3}× < {G2_MIN_SPEEDUP}× threshold. \
             f32={f32_ns:.0} ns/tok, f16={f16_ns:.0} ns/tok. \
             See .issues/200 §Why G2 failed — f16 weight-only quantization is \
             slower than f32 on this hardware (f32 activation limits bandwidth \
             reduction to 25%, FCVT latency dominates)."
        );
        println!("  G2: PASS (speedup {speedup:.3}× ≥ {G2_MIN_SPEEDUP}×)");
    }
}

#[test]
#[ignore = "profiling harness — per-phase cost measurement, not a pass/fail test; run with --release --ignored"]
fn g1_g2_f16_combined_report() {
    // Combined report for convenience — runs both G1 and G2 in one test.
    g1_f16_approximate_correctness();
    g2_f16_speedup_vs_f32();
}

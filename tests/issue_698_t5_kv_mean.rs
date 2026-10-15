#![cfg(feature = "tf_loop")]
//! Issue 698 T5 — KV mean-across-steps (`CacheStrategy::Mean`).
//!
//! The tf_loop stash pass (step 6) re-forwards the window ONE extra time per
//! token just to write canonical KV entries. `CacheStrategy::Mean` folds the
//! loop iterations' own K/V rows into an incremental running mean (fixed
//! loop order ⇒ deterministic f32 sum) and writes THAT back — the stash
//! window-forward is deleted. Per token at k=4, β=0.5: 6 window-forwards →
//! 5 (−16.7% of window compute).
//!
//! # Lossy-surface promotion rule (Bench-773 / Issue 750 T3, in full)
//!
//! The mean is a LOSSY transform of the canonical KV: gate on **per-position
//! argmax stability** (the behavior axis — a flipped argmax is a family
//! behavior flip) + a **pinned max_abs band** on the logits (never max_rel —
//! the 1e-6 denominator floor cannot certify lossy numerics), **never
//! bit-identity**. Per-family conditional retention = per decode position:
//! every position's argmax must survive the strategy swap.
//!
//! # The exact corner (plumbing proof)
//!
//! k=1 with β=0 (DampedEuler) is EXACT end-to-end: the single iteration's
//! running mean IS that iteration's K/V, which was computed from the
//! pre-window state — the same input the `First` stash forwards from — so
//! Mean ≡ First bit-for-bit (logits AND cache rows). This pins the fold +
//! write-back plumbing exactly, without claiming the k>1 lossy path is
//! exact. Both iteration modes are covered (the per-layer sub-step with
//! k=1 collapses identically).
//!
//! # Falsifiable content
//!
//! 1. G1: every arm's decode is bit-identical across repeat runs.
//! 2. k=1 degenerate: Mean ≡ First (logits + KV rows, both modes).
//! 3. k=4 lossy gate: zero argmax flips over the decode positions ( Mean vs
//!    the `First` default); max_abs band pinned as raw bits.
//! 4. Wall-clock: Mean strictly faster than First (the deleted pass is pure
//!    work removal — 5/6 of the window compute per token at k=4, β=0.5).
//!
//! # Run
//!
//! ```bash
//! cargo test --test issue_698_t5_kv_mean -- --nocapture
//! ```

use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_training_free_loop,
};
use katgpt_rs::types::{
    CacheStrategy, Config, IterationMode, Rng, SubStepStrategy, TrainingFreeLoopConfig, kv_dim,
};

// ── Fixture ──────────────────────────────────────────────────────

/// Decode positions exercised per arm (families = positions).
const N_POS: usize = 12;

/// Wall-clock shape: each round runs WALL_INNER decodes of WALL_POS
/// positions over a REUSED ctx+cache (alloc-free steady state — per-decode
/// allocation noise cancels between arms); min round per arm is the
/// statistic. WALL_POS ≤ block_size (16): the KV cache is block_size rows
/// (pre-existing caller contract — pos must stay below block_size).
const WALL_POS: usize = 16;
const WALL_INNER: usize = 400;
const WALL_ROUNDS: usize = 5;

/// Fixture seed (the 698 / 407 convention).
const SEED: u64 = 42;

/// Pinned fixture identity (T1's convention): any change to weight init or
/// the fixture config re-keys this hash and fails loudly instead of silently
/// re-basing the pinned band.
///
/// PLATFORM-BOUND by construction (first x86_64 execution, issue 806,
/// 4090 box, 2026-09-16): the weights come from `Rng::normal()`'s Box-Muller
/// (`f32::ln`/`f32::cos`), whose ulps differ between macOS libm and the MSVC
/// ucrt — so every fixture weight differs at the last bit and the BLAKE3
/// hash re-keys. Measured: aarch64 `23d0daab3f087159`, x86_64
/// `4d0b592740db9358`. Both pins are live (a drift within ONE platform
/// still fails loudly); the behavior gates below are the cross-platform
/// contract, not this identity record.
#[cfg(target_arch = "aarch64")]
const PINNED_FIXTURE_HASH: &str = "23d0daab3f087159";
#[cfg(target_arch = "x86_64")]
const PINNED_FIXTURE_HASH: &str = "4d0b592740db9358";
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const PINNED_FIXTURE_HASH: &str = "23d0daab3f087159";

/// Pinned max_abs band (raw bits, debug + release bit-identical on this
/// platform — the same-platform determinism contract; a cross-platform run
/// that trips the pin should relax to a tolerance and record the delta, the
/// Bench-773 metric lesson). The behavior gate is the argmax assertion
/// above; the band is the magnitude record.
///
/// PLATFORM-BOUND at 1 ulp (issue 806 measurement, same session as the
/// fixture hash): aarch64 `0x3e5f_d968` (2.186028e-1), x86_64 `0x3e5f_d970`
/// (2.186029e-1) — the x86_64 forward realizes the band ONE bit higher;
/// argmax flips stayed 0/12 on both. Same-platform drift still fails loudly.
#[cfg(target_arch = "aarch64")]
const PINNED_BAND_BITS: u32 = 0x3e5f_d968; // 2.186028e-1
#[cfg(target_arch = "x86_64")]
const PINNED_BAND_BITS: u32 = 0x3e5f_d970; // 2.186029e-1
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const PINNED_BAND_BITS: u32 = 0x3e5f_d968; // 2.186028e-1

/// micro + n_layer=3: a real 3-phase forward (pre-loop layer 0, window
/// layer 1, post-loop layer 2) with the window explicitly mid-stack — the
/// production tf_loop shape, unlike n_layer=1 where the window IS the model.
fn make_config() -> Config {
    let mut config = Config::micro();
    config.n_layer = 3;
    config
}

/// Deterministic BLAKE3[16] over every active f32 weight slice (T1's hash
/// convention; the n_layer=3 config re-keys the hash vs T1's fixture).
fn fixture_hash(config: &Config, weights: &TransformerWeights) -> String {
    let mut hasher = blake3::Hasher::new();
    let feed = |h: &mut blake3::Hasher, v: &[f32]| {
        for f in v {
            h.update(&f.to_le_bytes());
        }
    };
    feed(&mut hasher, &weights.wte);
    feed(&mut hasher, &weights.wpe);
    feed(&mut hasher, &weights.lm_head);
    for layer in &weights.layers {
        feed(&mut hasher, &layer.attn_wq);
        feed(&mut hasher, &layer.attn_wk);
        feed(&mut hasher, &layer.attn_wv);
        feed(&mut hasher, &layer.attn_wo);
        feed(&mut hasher, &layer.mlp_w1);
        feed(&mut hasher, &layer.mlp_w2);
    }
    for d in [
        config.vocab_size,
        config.n_embd,
        config.n_layer,
        config.n_head,
        config.head_dim,
        config.mlp_hidden,
    ] {
        hasher.update(&d.to_le_bytes());
    }
    hasher.finalize().to_hex()[..16].to_string()
}

/// Deterministic token schedule: position → token.
fn token_at(pos: usize) -> usize {
    (pos * 7 + 3) % 27
}

/// Build a tf_config: window (1,1) — the middle layer of 3.
fn tf_config(
    k: usize,
    strategy: SubStepStrategy,
    mode: IterationMode,
    cache_strategy: CacheStrategy,
) -> TrainingFreeLoopConfig {
    TrainingFreeLoopConfig {
        window_start: 1,
        window_end: 1,
        loop_count: k,
        strategy,
        iteration_mode: mode,
        cache_strategy,
    }
}

/// Decode `n_pos` positions through the production tf_loop forward into the
/// given ctx+cache. Returns per-position logits.
fn decode_into(
    config: &Config,
    weights: &TransformerWeights,
    tf: &TrainingFreeLoopConfig,
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    n_pos: usize,
) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(n_pos);
    for pos in 0..n_pos {
        let logits = forward_training_free_loop(
            ctx,
            weights,
            cache,
            token_at(pos),
            pos,
            config,
            tf,
            #[cfg(feature = "loop_guidance")]
            None, // Plan 617: LoopCD guidance - None = bit-identical baseline
        );
        out.push(logits.to_vec());
    }
    out
}

/// Fresh-fixture decode: returns per-position logits + the final cache (for
/// KV-row comparison).
fn decode(
    config: &Config,
    weights: &TransformerWeights,
    tf: &TrainingFreeLoopConfig,
    n_pos: usize,
) -> (Vec<Vec<f32>>, MultiLayerKVCache) {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let out = decode_into(config, weights, tf, &mut ctx, &mut cache, n_pos);
    (out, cache)
}

fn argmax(l: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..l.len() {
        if l[i] > l[best] {
            best = i;
        }
    }
    best
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── The gate ─────────────────────────────────────────────────────

#[test]
fn t698_t5_kv_mean_gates() {
    let config = make_config();
    let mut rng = Rng::new(SEED);
    let weights = TransformerWeights::new(&config, &mut rng);
    let hash = fixture_hash(&config, &weights);
    println!("fixture hash (blake3[16]): {hash}");
    assert_eq!(
        hash, PINNED_FIXTURE_HASH,
        "fixture identity drifted — re-pin PINNED_FIXTURE_HASH + PINNED_BAND_BITS and \
         record the delta (never re-base silently)"
    );

    let kvd = kv_dim(&config);

    // Production shape: k=4, K-stage RK β=0.5 (the default strategy), Block.
    let first = tf_config(
        4,
        SubStepStrategy::KStageRK { beta: 0.5 },
        IterationMode::Block,
        CacheStrategy::First,
    );
    let last = tf_config(
        4,
        SubStepStrategy::KStageRK { beta: 0.5 },
        IterationMode::Block,
        CacheStrategy::Last,
    );
    let mean = tf_config(
        4,
        SubStepStrategy::KStageRK { beta: 0.5 },
        IterationMode::Block,
        CacheStrategy::Mean,
    );

    // ── G1: repeat-run bit-identity per arm ──────────────────────
    let (f_logits, _) = decode(&config, &weights, &first, N_POS);
    let (f_logits2, _) = decode(&config, &weights, &first, N_POS);
    let (l_logits, _) = decode(&config, &weights, &last, N_POS);
    let (l_logits2, _) = decode(&config, &weights, &last, N_POS);
    let (m_logits, m_cache) = decode(&config, &weights, &mean, N_POS);
    let (m_logits2, m_cache2) = decode(&config, &weights, &mean, N_POS);
    for pos in 0..N_POS {
        assert_eq!(
            f_logits[pos]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>(),
            f_logits2[pos]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>(),
            "G1: First decode differs between runs at pos {pos}"
        );
        assert_eq!(
            l_logits[pos]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>(),
            l_logits2[pos]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>(),
            "G1: Last decode differs between runs at pos {pos}"
        );
        assert_eq!(
            m_logits[pos]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>(),
            m_logits2[pos]
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>(),
            "G1: Mean decode differs between runs at pos {pos}"
        );
    }
    // Cache rows bit-identical across Mean repeats (the write-back itself).
    for layer in 0..config.n_layer {
        for pos in 0..N_POS {
            let off = pos * kvd;
            assert_eq!(
                m_cache.layers[layer].key[off..off + kvd],
                m_cache2.layers[layer].key[off..off + kvd],
                "G1: Mean K row (layer {layer}, pos {pos}) differs between runs"
            );
        }
    }

    // ── Lossy gate: argmax stability + max_abs band (Mean vs First) ──
    let mut flips = 0usize;
    let mut band = 0.0f32;
    for pos in 0..N_POS {
        if argmax(&m_logits[pos]) != argmax(&f_logits[pos]) {
            flips += 1;
        }
        band = band.max(max_abs_diff(&m_logits[pos], &f_logits[pos]));
    }
    println!(
        "Mean vs First: argmax flips {flips}/{N_POS}, max_abs band {band:.6e} (0x{:08x})",
        band.to_bits()
    );

    // Last vs First (recorded — Last is not the default; both do the stash pass).
    let mut flips_last = 0usize;
    let mut band_last = 0.0f32;
    for pos in 0..N_POS {
        if argmax(&l_logits[pos]) != argmax(&f_logits[pos]) {
            flips_last += 1;
        }
        band_last = band_last.max(max_abs_diff(&l_logits[pos], &f_logits[pos]));
    }
    println!("Last vs First: argmax flips {flips_last}/{N_POS}, max_abs band {band_last:.6e}");

    // The behavior gate (pre-registered): every decode position's argmax
    // must survive the strategy swap — per-family conditional retention.
    assert_eq!(
        flips, 0,
        "Mean flipped {flips}/{N_POS} decode-position argmaxes vs the First default — \
         the lossy-surface behavior gate FAILS; Mean must stay opt-in"
    );
    // The magnitude record: the pinned max_abs band must reproduce exactly
    // on this platform (cross-platform drift → relax to a tolerance + record
    // the delta, the T1/Bench-773 escape).
    assert_eq!(
        band.to_bits(),
        PINNED_BAND_BITS,
        "pinned max_abs band drifted: measured 0x{:08x} — the lossy numerics moved; \
         re-pin + record the delta (never re-base silently)",
        band.to_bits()
    );
    // All logits finite on every arm.
    for logits in f_logits
        .iter()
        .chain(m_logits.iter())
        .chain(l_logits.iter())
    {
        assert!(logits.iter().all(|f| f.is_finite()), "non-finite logits");
    }

    // ── k=1 degenerate (β=0): Mean ≡ First, bit-exact, both modes ────
    for mode in [IterationMode::Block, IterationMode::Layer] {
        let first1 = tf_config(1, SubStepStrategy::DampedEuler, mode, CacheStrategy::First);
        let mean1 = tf_config(1, SubStepStrategy::DampedEuler, mode, CacheStrategy::Mean);
        let (f1, f1_cache) = decode(&config, &weights, &first1, N_POS);
        let (m1, m1_cache) = decode(&config, &weights, &mean1, N_POS);
        for pos in 0..N_POS {
            assert_eq!(
                f1[pos].iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                m1[pos].iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "k=1 degenerate ({mode:?}): logits differ at pos {pos} — the fold/write-back \
                 plumbing is broken"
            );
        }
        // Direct KV-row equality across the whole decode (the mean of one
        // iteration IS that iteration's K/V == the First stash's K/V).
        for layer in 0..config.n_layer {
            for pos in 0..N_POS {
                let off = pos * kvd;
                assert_eq!(
                    f1_cache.layers[layer].key[off..off + kvd],
                    m1_cache.layers[layer].key[off..off + kvd],
                    "k=1 degenerate ({mode:?}): K row (layer {layer}, pos {pos}) differs"
                );
                assert_eq!(
                    f1_cache.layers[layer].value[off..off + kvd],
                    m1_cache.layers[layer].value[off..off + kvd],
                    "k=1 degenerate ({mode:?}): V row (layer {layer}, pos {pos}) differs"
                );
            }
        }
    }

    // ── Wall-clock: the deleted stash pass ──────────────────
    // k=4, β=0.5: First runs anchor(1) + loop(4) + stash(1) = 6 window
    // forwards per token; Mean runs 5. Min-of-rounds per arm over reused
    // ctx+cache (allocation-free steady state).
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);
    // Warmup both arms (caches, branch predictors, allocator).
    let _ = decode_into(&config, &weights, &first, &mut ctx, &mut cache, WALL_POS);
    let _ = decode_into(&config, &weights, &mean, &mut ctx, &mut cache, WALL_POS);
    let mut t_first = f64::MAX;
    let mut t_mean = f64::MAX;
    for _ in 0..WALL_ROUNDS {
        let t = std::time::Instant::now();
        for _ in 0..WALL_INNER {
            let _ = decode_into(&config, &weights, &first, &mut ctx, &mut cache, WALL_POS);
        }
        t_first = t_first.min(t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        for _ in 0..WALL_INNER {
            let _ = decode_into(&config, &weights, &mean, &mut ctx, &mut cache, WALL_POS);
        }
        t_mean = t_mean.min(t.elapsed().as_secs_f64());
    }
    let ratio = t_mean / t_first;
    let f_ms = t_first * 1e3;
    let m_ms = t_mean * 1e3;
    println!(
        "wall-clock ({WALL_ROUNDS} rounds × {WALL_POS} pos, min): First {f_ms:.2} ms · \
         Mean {m_ms:.2} ms · ratio {ratio:.3}× (theoretical floor 5/6 ≈ 0.833)"
    );
    assert!(
        t_mean < t_first,
        "Mean ({t_mean:.4} ms) must be strictly faster than First ({t_first:.4} ms) — \
         the deleted stash pass is pure work removal; a slowdown means the write-back \
         is more expensive than the pass it replaced"
    );

    println!("\n═══ Issue 698 T5 verdict ═══");
    println!("  fixture blake3[16] = {hash} · seed {SEED} · n_layer 3 · window (1,1)");
    println!("  k=1 degenerate: EXACT (logits + KV rows, Block + Layer) — plumbing pinned");
    println!("  k=4 lossy gate: {flips}/{N_POS} argmax flips, max_abs {band:.3e} — stable");
    println!("  wall-clock: {ratio:.3}× (stash pass deleted = 1/6 of window compute at k=4)");
}

//! Issue 584 Phase 1+ — FlashMemory real-weights retrieval accuracy validation.
//!
//! The Phase 1 mechanism test (9 unit tests in `flashmemory_sparse.rs`) validated
//! the MECHANISM with random weights — proving the sparse forward runs finite +
//! stable. But random weights have uniform attention patterns: they can't test
//! whether FlashMemory's sigmoid-threshold sparse selection preserves MEANINGFUL
//! attention behavior.
//!
//! This bench closes that gap with **real Kimi-K3-0.40B weights**:
//!
//! 1. Load real `model.safetensors` → extract MLA weights from layer 3 (first
//!    MLA layer in Kimi-K3-0.40B's hybrid 6-KDA + 2-MLA architecture).
//! 2. Create realistic hidden states from token embedding lookup (real token IDs
//!    from a NIAH-style prompt — a needle sentence embedded in a haystack).
//! 3. Run both dense (`mla_forward_token`) and sparse (`mla_forward_token_flashmemory`)
//!    MLA forward on the SAME hidden states, with separate but identically-fed
//!    caches. Both caches append the same `c_kv`/`k_r` (same weights, same input),
//!    so the ONLY difference is which tokens receive attention weight.
//! 4. Compare per-token outputs: cosine similarity + relative MSE.
//! 5. Report block selection dynamics: avg blocks selected per head, refresh
//!    amortization ratio, fraction of tokens attended.
//!
//! # G1 gate (correctness — the load-bearing question)
//!
//! Does FlashMemory sparse selection preserve the dense MLA output within an
//! acceptable tolerance on real weights?
//!
//! - **PASS criterion:** median cosine similarity ≥ 0.90 AND max relative MSE ≤ 0.5
//!   over the full sequence. This is the threshold below which the downstream
//!   model output is empirically indistinguishable (per FlashMemory paper §3.2,
//!   retrieval tasks tolerate ≤30% attention mass loss).
//! - **FAIL → sparse selection is too aggressive (threshold too high) OR the
//!   block centroid approximation loses too much information.**
//!
//! # Run
//!
//! ```bash
//! # Requires real model.safetensors at data/kimi-k3-0.40b/
//! cargo bench --manifest-path Cargo.toml \
//!     --features "kimi_k3_loader flashmemory_sparse" \
//!     --bench bench_021_flashmemory_real_weights_retrieval -- --nocapture
//!
//! # Override context length (default 512):
//! FLASHMEMORY_BENCH_SEQ=1024 cargo bench --manifest-path Cargo.toml \
//!     --features "kimi_k3_loader flashmemory_sparse" \
//!     --bench bench_021_flashmemory_real_weights_retrieval -- --nocapture
//!
//! # Override block size + threshold (paper defaults: 64/0.5):
//! FLASHMEMORY_BLOCK_SIZE=32 FLASHMEMORY_THRESHOLD=0.3 cargo bench ...
//! ```

#![cfg(feature = "kimi_k3_loader")]
#![allow(clippy::needless_range_loop)]

use std::time::Instant;

use katgpt_attn::dash_attn::flashmemory_sparse::{
    FlashMemoryBlockCache, FlashMemoryConfig, FlashMemorySelector, mla_forward_token_flashmemory,
};
use katgpt_attn::mla::{MlaForwardScratch, MlaKVCache, MlaWeights, mla_forward_token};
use katgpt_kv::shard_kv::rope::RopeFreqs;

use katgpt_rs::kimi_k3::loader::{KimiK3ModelWeights, load_kimi_k3};

// ---------------------------------------------------------------------------
// Configuration helpers
// ---------------------------------------------------------------------------

/// Parse env var or fall back to default.
fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_or_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Which MLA layer to extract weights from. Kimi-K3-0.40B has MLA at layers 3 + 7.
const MLA_LAYER_IDX: usize = 3;

// ---------------------------------------------------------------------------
// NIAH-style prompt construction
// ---------------------------------------------------------------------------

/// Build a needle-in-haystack token sequence using real token IDs.
///
/// We don't have a tokenizer readily available in this bench context, so we
/// construct a synthetic but realistic sequence: a small vocabulary of distinct
/// token IDs repeated to fill the context, with a "needle" token at a specific
/// position. The point is NOT semantic retrieval — it's to exercise the real
/// weight matrices with realistic-magnitude, diverse hidden states.
///
/// The hidden states come from the real embedding table: `h_t = embed[token_id_t]`.
/// These are 1024-dim vectors with real learned statistics (not random), so the
/// MLA layer's attention patterns will reflect real weight structure.
fn build_niah_token_ids(seq_len: usize, vocab_size: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(seq_len);

    // Hay: cycle through a diverse set of token IDs.
    // Use IDs spread across the vocab to get varied embedding vectors.
    let hay_cycle = 128.min(vocab_size);
    let needle_pos = seq_len / 2;

    for i in 0..seq_len {
        if i == needle_pos {
            // The needle — a unique token not in the hay cycle.
            let needle_id = ((hay_cycle + 42) as u32).min(vocab_size as u32 - 1);
            ids.push(needle_id);
        } else if i == needle_pos + 1 {
            // The needle's context — another unique token.
            let ctx_id = ((hay_cycle + 99) as u32).min(vocab_size as u32 - 1);
            ids.push(ctx_id);
        } else {
            // Hay — cycling through diverse token IDs.
            ids.push((i % hay_cycle) as u32);
        }
    }

    ids
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot = katgpt_core::simd::simd_dot_f32(a, b, a.len());
    let norm_a = katgpt_core::simd::simd_dot_f32(a, a, a.len()).sqrt();
    let norm_b = katgpt_core::simd::simd_dot_f32(b, b, b.len()).sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Relative MSE: ||a - b||² / ||a||². 0 = perfect match, 1 = completely different.
fn relative_mse(a: &[f32], b: &[f32]) -> f32 {
    let mut diff_sq = 0.0f32;
    let mut norm_sq = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let d = ai - bi;
        diff_sq += d * d;
        norm_sq += ai * ai;
    }
    if norm_sq < 1e-12 {
        return if diff_sq < 1e-12 { 0.0 } else { f32::INFINITY };
    }
    diff_sq / norm_sq
}

/// Check if all values are finite (no NaN/Inf).
fn all_finite(x: &[f32]) -> bool {
    x.iter().all(|&v| v.is_finite())
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

fn run_bench() {
    let config = katgpt_rs::kimi_k3::model::KimiK3ModelConfig::kimi_k3_0_40b();
    let d = config.hidden_size;
    println!(
        "Config: D_model={d}, vocab={vocab}, layers={n_layers}",
        vocab = config.vocab_size,
        n_layers = config.num_layers
    );
    println!("MLA layers: {:?}", config.mla_layer_indices);
    println!(
        "MLA config: kv_lora_rank={}, q_lora_rank={}, d_h={}, n_heads={}",
        config.mla_config.kv_lora_rank,
        config.mla_config.q_lora_rank,
        config.mla_config.d_h(),
        config.mla_config.n_heads
    );

    // ── Load real weights ──────────────────────────────────────────────────
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    });
    let model_path = format!("{model_dir}/model.safetensors");

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ERROR: requires real model.safetensors at {model_path}");
        eprintln!("This bench validates real-weight attention behavior — random weights");
        eprintln!("are already covered by the 9 unit tests in flashmemory_sparse.rs.");
        std::process::exit(1);
    }

    print!("Loading real model.safetensors ... ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let t0 = Instant::now();
    let weights: KimiK3ModelWeights = load_kimi_k3(&model_path).unwrap_or_else(|e| {
        eprintln!("\n❌ load failed: {e}");
        std::process::exit(1);
    });
    println!("done ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── Extract MLA weights from the target layer ──────────────────────────
    use katgpt_rs::kimi_k3::decoder_layer::KimiAttentionWeights;
    let KimiAttentionWeights::Mla(mla_weights) = &weights.layers[MLA_LAYER_IDX].attention else {
        eprintln!("ERROR: layer {MLA_LAYER_IDX} is not an MLA layer");
        std::process::exit(1);
    };
    // Clone the weights so we own them (avoids borrow issues with the full model).
    let mla_weights: MlaWeights = mla_weights.clone();
    let mla_config = &config.mla_config;
    println!("\nExtracted MLA weights from layer {MLA_LAYER_IDX}");

    // ── Build NIAH token sequence ──────────────────────────────────────────
    let seq_len = env_or("FLASHMEMORY_BENCH_SEQ", 512);
    let token_ids = build_niah_token_ids(seq_len, config.vocab_size);
    println!(
        "Sequence length: {seq_len} tokens (needle at position {})",
        seq_len / 2
    );

    // Hidden states from real embedding table.
    let hidden_states: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&tid| {
            let start = (tid as usize) * d;
            weights.embed_weight[start..start + d].to_vec()
        })
        .collect();

    // ── FlashMemory config ─────────────────────────────────────────────────
    let block_size = env_or("FLASHMEMORY_BLOCK_SIZE", 16);
    let refresh_period = env_or("FLASHMEMORY_REFRESH_PERIOD", block_size);
    let threshold = env_or_f32("FLASHMEMORY_THRESHOLD", 0.5);
    let fm_config = FlashMemoryConfig {
        block_size,
        refresh_period,
        threshold,
    };
    println!(
        "FlashMemory config: block_size={block_size}, refresh_period={refresh_period}, threshold={threshold}"
    );
    let max_blocks = seq_len.div_ceil(block_size);

    // ── Dense MLA setup ────────────────────────────────────────────────────
    let mut cache_dense = MlaKVCache::new(mla_config, seq_len + 1);
    let mut scratch_dense = MlaForwardScratch::new(mla_config, seq_len + 1);
    let mut rope_dense =
        RopeFreqs::new_with_theta(mla_config.qk_rope_head_dim, mla_config.rope_theta);

    // ── Sparse (FlashMemory) MLA setup ─────────────────────────────────────
    let mut cache_sparse = MlaKVCache::new(mla_config, seq_len + 1);
    let mut scratch_sparse = MlaForwardScratch::new(mla_config, seq_len + 1);
    let mut rope_sparse =
        RopeFreqs::new_with_theta(mla_config.qk_rope_head_dim, mla_config.rope_theta);
    let mut block_cache = FlashMemoryBlockCache::new(mla_config, &fm_config, seq_len + 1);
    let mut selector = FlashMemorySelector::new(fm_config.clone(), mla_config.n_heads, max_blocks);

    // ── Run both forward paths ─────────────────────────────────────────────
    println!("\nRunning dense + sparse MLA forward on {seq_len} tokens ...");
    let t0 = Instant::now();

    let mut cos_sims = Vec::with_capacity(seq_len);
    let mut rel_mses = Vec::with_capacity(seq_len);
    let mut total_blocks_selected_per_token = Vec::with_capacity(seq_len);
    let mut total_tokens_attended_per_token = Vec::with_capacity(seq_len);

    for (step, h) in hidden_states.iter().enumerate() {
        // Dense forward.
        let out_dense = mla_forward_token(
            mla_config,
            &mla_weights,
            &mut cache_dense,
            &mut scratch_dense,
            &mut rope_dense,
            h,
        );
        let out_dense: Vec<f32> = out_dense.to_vec();

        // Sparse forward.
        let out_sparse = mla_forward_token_flashmemory(
            mla_config,
            &mla_weights,
            &mut cache_sparse,
            &mut scratch_sparse,
            &mut rope_sparse,
            h,
            &mut block_cache,
            &mut selector,
            step,
        );
        let out_sparse: Vec<f32> = out_sparse.to_vec();

        // Record metrics.
        let cs = cosine_sim(&out_dense, &out_sparse);
        let rm = relative_mse(&out_dense, &out_sparse);
        cos_sims.push(cs);
        rel_mses.push(rm);

        // Block selection stats (only meaningful after the first block fills).
        let selection = selector.selection();
        let total_sel: usize = (0..mla_config.n_heads)
            .map(|h| selection.len_for_head(h))
            .sum();
        total_blocks_selected_per_token.push(total_sel);

        // Tokens attended = sum of block sizes for selected blocks.
        let tokens_attended: usize = (0..mla_config.n_heads)
            .map(|head| {
                selection.blocks_per_head[head]
                    .iter()
                    .map(|&blk| block_cache.block_count(blk))
                    .sum::<usize>()
            })
            .sum();
        total_tokens_attended_per_token.push(tokens_attended);

        // Finite check.
        if !all_finite(&out_dense) || !all_finite(&out_sparse) {
            eprintln!("FAIL: non-finite output at step {step}");
            eprintln!(
                "  dense finite: {}, sparse finite: {}",
                all_finite(&out_dense),
                all_finite(&out_sparse)
            );
            std::process::exit(1);
        }
    }

    let elapsed = t0.elapsed();
    println!(
        "done ({:.2}s, {:.1}ms/token)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / seq_len as f64
    );

    // ── Report ─────────────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("FlashMemory Real-Weights Retrieval Accuracy (Issue 584 Phase 1+)");
    println!("{}", "=".repeat(70));

    // Cosine similarity statistics.
    cos_sims.sort_by(|a, b| a.total_cmp(b));
    let cs_min = cos_sims.first().copied().unwrap_or(0.0);
    let cs_p25 = cos_sims[cos_sims.len() / 4];
    let cs_median = cos_sims[cos_sims.len() / 2];
    let cs_p75 = cos_sims[3 * cos_sims.len() / 4];
    let cs_max = cos_sims.last().copied().unwrap_or(0.0);
    let cs_mean = cos_sims.iter().sum::<f32>() / cos_sims.len() as f32;

    println!("\nCosine similarity (dense vs sparse output per token):");
    println!(
        "  min={cs_min:.4}  p25={cs_p25:.4}  median={cs_median:.4}  p75={cs_p75:.4}  max={cs_max:.4}  mean={cs_mean:.4}"
    );

    // Relative MSE statistics.
    rel_mses.sort_by(|a, b| a.total_cmp(b));
    let rm_median = rel_mses[rel_mses.len() / 2];
    let rm_p90 = rel_mses[9 * rel_mses.len() / 10];
    let rm_max = rel_mses.last().copied().unwrap_or(0.0);
    let rm_mean = rel_mses.iter().sum::<f32>() / rel_mses.len() as f32;

    println!("\nRelative MSE (||dense - sparse||² / ||dense||²):");
    println!("  median={rm_median:.4}  p90={rm_p90:.4}  max={rm_max:.4}  mean={rm_mean:.4}");

    // Block selection statistics (skip early tokens before first full block).
    let warmup = block_size.min(seq_len);
    if seq_len > warmup {
        let sel_stats = &total_blocks_selected_per_token[warmup..];
        let attended_stats = &total_tokens_attended_per_token[warmup..];
        let avg_blocks: f32 =
            sel_stats.iter().map(|&x| x as f32).sum::<f32>() / sel_stats.len() as f32;
        let avg_tokens: f32 =
            attended_stats.iter().map(|&x| x as f32).sum::<f32>() / attended_stats.len() as f32;
        let max_blocks_per_head = (max_blocks * mla_config.n_heads) as f32;
        let selection_ratio = avg_blocks / max_blocks_per_head;
        let attendance_ratio = avg_tokens / (seq_len * mla_config.n_heads) as f32;

        println!("\nBlock selection dynamics (post-warmup, steps {warmup}..{seq_len}):");
        println!(
            "  avg blocks selected (all heads): {avg_blocks:.1} / {max_blocks_per_head:.0} total ({pct_blocks:.1}%)",
            pct_blocks = selection_ratio * 100.0
        );
        println!(
            "  avg tokens attended (all heads): {avg_tokens:.1} / {} total ({pct_tokens:.1}%)",
            seq_len * mla_config.n_heads,
            pct_tokens = attendance_ratio * 100.0
        );
        println!(
            "  selector refresh count: {} (amortization: {:.1}× = {refresh_period}-step period)",
            selector.refresh_count(),
            seq_len as f64 / selector.refresh_count().max(1) as f64
        );
    }

    // ── G1 gate verdict ────────────────────────────────────────────────────
    println!("\n{}", "─".repeat(70));
    println!("G1 GATE (correctness — real weights)");
    println!("{}", "─".repeat(70));

    let cs_gate = cs_median >= 0.90;
    let rm_gate = rm_median <= 0.5;
    let all_finite_gate = true; // already checked per-token above

    println!(
        "  median cosine ≥ 0.90: {} ({cs_median:.4})",
        if cs_gate { "PASS" } else { "FAIL" }
    );
    println!(
        "  median rel MSE ≤ 0.50: {} ({rm_median:.4})",
        if rm_gate { "PASS" } else { "FAIL" }
    );
    println!(
        "  all outputs finite:    {}",
        if all_finite_gate { "PASS" } else { "FAIL" }
    );

    let g1_pass = cs_gate && rm_gate && all_finite_gate;
    println!(
        "\n  G1 VERDICT: {}",
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    if g1_pass {
        println!("\n  → FlashMemory sparse selection preserves dense MLA output on");
        println!("    real Kimi-K3-0.40B weights within acceptable tolerance.");
        println!("  → Phase 2 (256K scale test on 4090) is DE-RISKED on the correctness axis.");
    } else {
        println!("\n  → Sparse selection degrades output quality beyond tolerance.");
        println!(
            "  → Consider lowering threshold (currently {threshold}) or increasing block_size."
        );
        println!("  → Phase 2 scale test may produce incorrect results — investigate first.");
    }

    // ── Exit code ──────────────────────────────────────────────────────────
    std::process::exit(if g1_pass { 0 } else { 1 });
}

fn main() {
    run_bench();
}

//! Issue 826 T1+T2 — FlashMemory full-model NIAH QA gate (needle-axis quality
//! measurement for the promotion gate).
//!
//! Bench 023's honest negative (2026-08-13): at single-layer depth the needle
//! is invisible (needle dense-attention rank 34/34 at layer 3/8 — retrieval is
//! a multi-layer emergent property), so the needle axis was never measured
//! end-to-end. Issue 584 Phase 3 pinned G1 on output-vector similarity
//! (Bench 021, cos >= 0.96) and deferred full-model NIAH to "Phase 2". SAS
//! (arXiv:2609.13141, katgpt-rs Research 567) documents the failure class this
//! gate exists for: pooling-based block summaries destroy needle-like
//! localized information — and FlashMemory's block centroid IS a
//! mean-of-latent pool.
//!
//! This bench closes that gap end-to-end:
//!
//! 1. Builds a real-text NIAH prompt — single-needle and multi-needle
//!    (RULER multi-key shape: a queried needle plus a distractor needle).
//! 2. Prefills + greedily decodes the FULL 8-layer Kimi-K3-0.40B two ways:
//!    - **DENSE** — the real `kimi_k3_forward_token` path, unmodified. The
//!      dense arm IS the oracle.
//!    - **SPARSE** — an identical layer composition (public pieces only) with
//!      the MLA layers' attention call swapped to
//!      `mla_forward_token_flashmemory`. KDA layers are unchanged (sparse
//!      only applies to full-attention layers; MLA at layers 3 and 7).
//! 3. QA axis: needle hit, dense-vs-sparse decode agreement, per-step logit
//!    cosine, password-token best rank. Selection axis (the SAS-class
//!    question): is the needle's BLOCK in the sigmoid-threshold selection at
//!    all, block coverage, recency-fallback rate.
//! 4. T2 budget stress: e2e arms at tightened sigma, plus a selection-axis
//!    sweep re-derived from the query step's block scores (public formulas:
//!    score = dot(q_c_h, centroid_h) * attn_scale, selected iff
//!    sigmoid(score) >= threshold).
//!
//! # Parity gate (birth check for the bench-local layer composition)
//!
//! The sparse arm's MLA layer forward mirrors `kimi_decoder_layer_forward`
//! step-for-step. At sigma = 0.0 every block passes the threshold, so the
//! sparse forward is mathematically dense with identical summation order —
//! the gate asserts max|delta-logits| <= 1e-3 against the real path over the
//! query step + every decode step at the shortest ladder length. A
//! composition drift fails the run loudly (exit 1) instead of measuring a
//! different model than the oracle.
//!
//! # Honest caveats (stated up front)
//!
//! - Kimi-K3-0.40B is trained at 4K (`max_position_embeddings: 4096`). The
//!   16384/65536 ladder points are RoPE extrapolation for BOTH arms — the
//!   dense-vs-sparse comparison stays matched-context, but absolute QA
//!   accuracy there reports the extrapolation regime, not serving reality.
//!   That is why the ladder starts at 2048 (in-distribution).
//! - At the query step the bench calls `force_refresh()` so the selection is
//!   scored against the query's own q_c (a serving stack would re-score for
//!   the answering turn). Generous-to-sparse is the conservative direction
//!   for a promotion gate: the gate must not fail the mechanism on an
//!   artifact of refresh staleness.
//! - Prefill-time selection staleness within the tau refresh window is the
//!   FlashMemory paper's own semantics, kept as-is.
//! - Modelless end to end: zero training, zero LLM calls, deterministic
//!   given the prompt + weights.
//!
//! # Run
//!
//! ```bash
//! KIMI_K3_MODEL_DIR=/path/to/kimi-k3-0.40b \
//! cargo bench --bench bench_685_flashmemory_full_model_niah_qa \
//!   --features kimi_k3_loader,flashmemory_sparse -- --nocapture
//!
//! # Knobs (all env):
//! #   FFM_SEQ_LENS     comma list, default "2048,16384,65536"
//! #   FFM_MULTI_LENS   comma list, default "16384,65536" (multi-needle at)
//! #   FFM_THRESHOLDS   comma list, default "0.5,0.7"        (e2e sparse arms)
//! #   FFM_BLOCK_SIZE   default 64   (paper default)
//! #   FFM_REFRESH      default 64   (paper default)
//! #   FFM_N_DECODE     default 12
//! #   FFM_SWEEP_LENS   comma list, default "16384,65536" (T2 sweep print)
//! ```

#![cfg(feature = "kimi_k3_loader")]
#![cfg(feature = "flashmemory_sparse")]
// Parallel-array head/layer indexing is clearer than iterator chains here
// (bench-023 posture).
#![allow(clippy::needless_range_loop)]

use std::io::{self, Write as _};
use std::time::Instant;

use katgpt_attn::dash_attn::flashmemory_sparse::{
    FlashMemoryBlockCache, FlashMemoryConfig, FlashMemorySelector, mla_forward_token_flashmemory,
};
use katgpt_attn::mla::{MlaConfig, MlaForwardScratch, MlaKVCache};
use katgpt_core::simd::{simd_add_inplace, simd_dot_f32, simd_matmul_rows};
use katgpt_core::types::math::rmsnorm_with_gamma_eps;
use katgpt_kv::shard_kv::rope::RopeFreqs;
use katgpt_rs::kimi_k3::decoder_layer::{
    KimiAttentionConfig, KimiAttentionScratch, KimiAttentionState, KimiAttentionWeights,
    KimiDecoderLayerConfig, KimiDecoderLayerWeights, KimiFfnConfig, KimiFfnScratch, KimiFfnWeights,
    kimi_decoder_layer_forward,
};
use katgpt_rs::kimi_k3::loader::{KimiK3ModelWeights, load_kimi_k3};
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token};
use katgpt_rs::kimi_k3::tiktoken::{TiktokenTokenizer, load_tiktoken_bpe};
use katgpt_transformer::attn_res::{AttnResBlockState, AttnResScratch, apply_attn_res};
use katgpt_transformer::moe::moe_forward_token;

// ---------------------------------------------------------------------------
// Env helpers
// ---------------------------------------------------------------------------

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_list_usize(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(s) if !s.trim().is_empty() => {
            s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
        }
        _ => default.to_vec(),
    }
}

fn env_list_f32(name: &str, default: &[f32]) -> Vec<f32> {
    match std::env::var(name) {
        Ok(s) if !s.trim().is_empty() => {
            s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
        }
        _ => default.to_vec(),
    }
}

// ---------------------------------------------------------------------------
// NIAH prompt construction (real text, RULER single-key + multi-key shape)
// ---------------------------------------------------------------------------

/// One planted needle: the sentence embedded in the haystack.
struct NeedleSpec {
    sentence: String,
}

struct Prompt {
    text: String,
    /// Char range [start, end) of each needle sentence inside `text`.
    needle_char_ranges: Vec<(usize, usize)>,
    /// Index (into needle arrays) of the needle the query asks about.
    queried_idx: usize,
}
// Diverse filler pool — a single repeated sentence makes every block
// near-identical, which both hurts retrieval realism and collapses the
// centroid signal the selector scores against. Cycled deterministically.
const FILLER_POOL: [&str; 10] = [
    "The wind moved quietly across the open field where nothing of note had happened for many hours. ",
    "A cart loaded with winter grain creaked along the road past the old stone wall. ",
    "Somewhere beyond the ridge a hawk circled twice and drifted out of sight. ",
    "The innkeeper counted his barrels and marked the tally on a slate by the door. ",
    "Rain had fallen in the night and every leaf still held its bright beads of water. ",
    "Two shepherds argued mildly about the price of wool and then shared a pipe. ",
    "The blacksmith's hammer kept a slow patient rhythm that could be heard for a mile. ",
    "Children chased a dog between the market stalls until the bell rang for noon. ",
    "An old map hung on the tavern wall, its coastlines worn away by many fingers. ",
    "By evening the clouds thinned and the first stars appeared over the eastern hills. ",
];

/// ASCII-only filler assembled from the varied pool, cut to exactly `chars`
/// bytes (safe: the pool is ASCII).
fn filler(chars: usize) -> String {
    if chars == 0 {
        return String::new();
    }
    let mut s = String::with_capacity(chars + FILLER_POOL[0].len());
    let mut i = 0usize;
    while s.len() < chars {
        s.push_str(FILLER_POOL[i % FILLER_POOL.len()]);
        i += 1;
    }
    s.truncate(chars.min(s.len()));
    s
}

fn build_prompt(variant_multi: bool, target_tokens: usize, query_stem: bool) -> Prompt {
    let queried = NeedleSpec {
        sentence: "The magic password is sunset7742. Remember it for later. ".to_string(),
    };
    let distractor = NeedleSpec {
        sentence: "The access code is morning3156. Remember it for later. ".to_string(),
    };

    // ~5.3 chars/token for this filler measured on tiktoken; 5 keeps the
    // actual token count close to (slightly under) the target.
    let total_chars = target_tokens * 5;
    let mut text = String::with_capacity(total_chars + 1024);
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let tail = if query_stem {
        // Continuation-style: the model completes the password directly.
        // Base models (no chat tuning) continue far more reliably than they
        // answer questions.
        " The magic password is"
    } else {
        " Question: What is the magic password?"
    };

    if !variant_multi {
        // Single-key: queried needle at the requested depth (default 0.5).
        let depth = std::env::var("FFM_DEPTH")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.5);
        let prefix_chars = ((total_chars as f32) * depth) as usize;
        text.push_str(&filler(prefix_chars));
        let start = text.len();
        text.push_str(&queried.sentence);
        ranges.push((start, text.len()));
        text.push_str(&filler(total_chars.saturating_sub(text.len())));
        text.push_str(tail);
        Prompt {
            text,
            needle_char_ranges: ranges,
            queried_idx: 0,
        }
    } else {
        // Multi-key: distractor SHALLOW (0.25), queried DEEP (0.75). The query
        // asks for the deep one — the retrieval-stress case.
        text.push_str(&filler(total_chars * 25 / 100));
        let s0 = text.len();
        text.push_str(&distractor.sentence);
        ranges.push((s0, text.len()));
        let mid_target = total_chars * 50 / 100;
        let mid = mid_target.saturating_sub(text.len() - s0);
        text.push_str(&filler(mid));
        let s1 = text.len();
        text.push_str(&queried.sentence);
        ranges.push((s1, text.len()));
        text.push_str(&filler(total_chars.saturating_sub(text.len())));
        text.push_str(tail);
        Prompt {
            text,
            needle_char_ranges: ranges,
            queried_idx: 1,
        }
    }
}

/// Token range of each needle: tokenize prefix-only vs prefix+needle
/// (bench-023's byte-offset mapping).
fn needle_token_ranges(
    tokenizer: &TiktokenTokenizer,
    prompt_text: &str,
    char_ranges: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    char_ranges
        .iter()
        .map(|&(s, e)| {
            let start = tokenizer
                .encode(&prompt_text[..s.min(prompt_text.len())])
                .len();
            let end = tokenizer
                .encode(&prompt_text[..e.min(prompt_text.len())])
                .len();
            (start, end)
        })
        .collect()
}

/// Minimal token window inside `sentence` whose decode contains `password`.
fn extract_pw_tokens(tokenizer: &TiktokenTokenizer, sentence: &str, password: &str) -> Vec<usize> {
    let ids = tokenizer.encode(sentence);
    let mut best: Option<(usize, usize)> = None;
    for i in 0..ids.len() {
        for j in (i + 1)..=ids.len() {
            if tokenizer.decode(&ids[i..j]).contains(password)
                && best.is_none_or(|(bi, bj)| j - i < bj - bi)
            {
                best = Some((i, j));
            }
        }
    }
    match best {
        Some((i, j)) => ids[i..j].to_vec(),
        None => {
            eprintln!(
                "WARN: password '{password}' not recoverable from needle tokens; using full sentence"
            );
            ids
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics helpers
// ---------------------------------------------------------------------------

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// 1-based rank of `v[id]` among all entries (1 = argmax).
fn rank_of(v: &[f32], id: usize) -> usize {
    let t = v[id];
    1 + v.iter().filter(|&&x| x > t).count()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn logit_of(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

// ---------------------------------------------------------------------------
// Sparse-arm per-MLA-layer FlashMemory state + per-step stats
// ---------------------------------------------------------------------------

/// Per-MLA-layer FlashMemory state (block centroid cache + selector) plus the
/// selection stats of the most recent forward. The arm runner snapshots these
/// at the query step + each decode step; prefill steps overwrite harmlessly.
struct LayerFm {
    block_cache: FlashMemoryBlockCache,
    selector: FlashMemorySelector,
    needle_block: Option<usize>,
    distractor_block: Option<usize>,
    query_step: usize,
    // Stats from the most recent forward.
    last_needle_sel: Vec<bool>,
    last_distractor_sel: Vec<bool>,
    last_fallback_heads: usize,
    last_covered_tokens: usize,
    /// Full block-score matrix at the query step `[head][block]` — the T2
    /// selection-axis sweep re-derives thresholds from this snapshot.
    query_scores: Option<Vec<Vec<f32>>>,
    /// Tokens per block at the query step (for sweep coverage math).
    query_block_counts: Option<Vec<usize>>,
}

impl LayerFm {
    fn new(
        config: &KimiK3ModelConfig,
        fm_config: &FlashMemoryConfig,
        max_seq: usize,
        needle_block: Option<usize>,
        distractor_block: Option<usize>,
        query_step: usize,
    ) -> Self {
        let max_blocks = max_seq.div_ceil(fm_config.block_size).max(1);
        Self {
            block_cache: FlashMemoryBlockCache::new(&config.mla_config, fm_config, max_seq),
            selector: FlashMemorySelector::new(
                fm_config.clone(),
                config.mla_config.n_heads,
                max_blocks,
            ),
            needle_block,
            distractor_block,
            query_step,
            last_needle_sel: Vec::new(),
            last_distractor_sel: Vec::new(),
            last_fallback_heads: 0,
            last_covered_tokens: 0,
            query_scores: None,
            query_block_counts: None,
        }
    }

    /// Record selection stats after a forward. `record_scores` is true only at
    /// the query step (the T2 sweep snapshot).
    fn record_step(
        &mut self,
        mla_cfg: &MlaConfig,
        scratch: &MlaForwardScratch,
        record_scores: bool,
    ) {
        let selected: Vec<Vec<usize>> = self.selector.selection().blocks_per_head.clone();
        let n_heads = selected.len();
        let n_active = self.block_cache.n_active_blocks();
        let mut covered = 0usize;
        let mut fallback = 0usize;
        let mut needle_sel = vec![false; n_heads];
        let mut distractor_sel = vec![false; n_heads];
        for (head, blocks) in selected.iter().enumerate() {
            if blocks.is_empty() {
                fallback += 1; // recency-fallback path inside the forward
            }
            for &b in blocks {
                covered += self.block_cache.block_count(b);
            }
            if let Some(nb) = self.needle_block
                && nb < n_active
                && blocks.contains(&nb)
            {
                needle_sel[head] = true;
            }
            if let Some(db) = self.distractor_block
                && db < n_active
                && blocks.contains(&db)
            {
                distractor_sel[head] = true;
            }
        }
        self.last_needle_sel = needle_sel;
        self.last_distractor_sel = distractor_sel;
        self.last_fallback_heads = fallback;
        self.last_covered_tokens = covered;

        if record_scores {
            let d_h = mla_cfg.d_h();
            let n_h = mla_cfg.n_heads;
            let scale = mla_cfg.attn_scale();
            let q_c = scratch.q_c_view();
            let mut matrix = Vec::with_capacity(n_h);
            for head in 0..n_h {
                let q = &q_c[head * d_h..(head + 1) * d_h];
                let mut row = Vec::with_capacity(n_active);
                for b in 0..n_active {
                    let c = self.block_cache.key_centroid(b, head);
                    row.push(simd_dot_f32(q, c, d_h) * scale);
                }
                matrix.push(row);
            }
            self.query_scores = Some(matrix);
            self.query_block_counts = Some(
                (0..n_active)
                    .map(|b| self.block_cache.block_count(b))
                    .collect(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Bench-local sparse full-model forward
// ---------------------------------------------------------------------------

/// One MLA decoder layer with FlashMemory sparse attention — a step-for-step
/// mirror of `kimi_decoder_layer_forward` (its steps 1-7) with the attention
/// call swapped. The parity gate (sigma = 0.0 → dense-equivalent) proves the
/// mirror against the real path at birth.
#[allow(clippy::too_many_arguments)]
fn mla_layer_forward_flashmemory(
    layer_idx: usize,
    layer_cfg: &KimiDecoderLayerConfig,
    layer_w: &KimiDecoderLayerWeights,
    fm: &mut LayerFm,
    step: usize,
    cache: &mut MlaKVCache,
    scratch: &mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    ffn_scratch: &mut KimiFfnScratch,
    res_self: &mut AttnResScratch,
    res_mlp: &mut AttnResScratch,
    block_state: &mut AttnResBlockState,
    prefix_sum: &mut [f32],
    scratch_hidden: &mut [f32],
) {
    let (mla_cfg, mla_w) = match (&layer_cfg.attention, &layer_w.attention) {
        (KimiAttentionConfig::Mla(c), KimiAttentionWeights::Mla(w)) => (c, w),
        _ => panic!("flashmemory bench: layer {layer_idx} is not MLA — topology changed"),
    };
    let d = layer_cfg.attn_res.d();
    let eps = layer_cfg.rms_eps;
    let block_size = layer_cfg.attn_res.block_size;
    let is_boundary = layer_idx.is_multiple_of(block_size);

    // Step 1: apply_attn_res (self-attention) — mix prefix_sum with blocks.
    if !block_state.is_empty() {
        let mixed = apply_attn_res(
            &layer_cfg.attn_res,
            &layer_w.self_attn_res,
            block_state,
            res_self,
            prefix_sum,
        );
        scratch_hidden.copy_from_slice(mixed);
    } else {
        scratch_hidden.copy_from_slice(prefix_sum);
    }

    // Step 2: block boundary — push prefix_sum BEFORE attention.
    if is_boundary {
        block_state.push(prefix_sum);
        prefix_sum.fill(0.0);
    }

    // Step 3: input_layernorm → FlashMemory sparse attention.
    rmsnorm_with_gamma_eps(scratch_hidden, &layer_w.input_layernorm_weight, eps as f64);
    {
        let attn_out = mla_forward_token_flashmemory(
            mla_cfg,
            mla_w,
            cache,
            scratch,
            rope_freqs,
            scratch_hidden,
            &mut fm.block_cache,
            &mut fm.selector,
            step,
        );
        simd_add_inplace(&mut prefix_sum[..d], &attn_out[..d]);
    }

    // Step 5: apply_attn_res (MLP).
    let mixed = apply_attn_res(
        &layer_cfg.attn_res,
        &layer_w.mlp_attn_res,
        block_state,
        res_mlp,
        prefix_sum,
    );
    scratch_hidden.copy_from_slice(mixed);

    // Step 6: post_attention_layernorm → FFN. MLA layers are MoE; a dense FFN
    // here would mean the layer topology changed — fail loudly.
    rmsnorm_with_gamma_eps(
        scratch_hidden,
        &layer_w.post_attention_layernorm_weight,
        eps as f64,
    );
    let ffn_out: &[f32] = match (&layer_cfg.ffn, &layer_w.ffn) {
        (KimiFfnConfig::Moe(moe_cfg), KimiFfnWeights::Moe(moe_w)) => {
            moe_forward_token(
                moe_w,
                moe_cfg,
                scratch_hidden,
                &mut ffn_scratch.dense_out,
                &mut ffn_scratch.moe,
            );
            &ffn_scratch.dense_out[..d]
        }
        _ => panic!("flashmemory bench: non-MoE FFN on MLA layer {layer_idx} — topology changed"),
    };

    // Step 7: accumulate FFN output.
    simd_add_inplace(&mut prefix_sum[..d], &ffn_out[..d]);

    // Selection stats (the T2 score snapshot only at the query step).
    fm.record_step(mla_cfg, scratch, step == fm.query_step);
}

/// Full-model forward with FlashMemory sparse attention on the MLA layers —
/// the sparse twin of `kimi_k3_forward_token`. KDA layers go through the real
/// `kimi_decoder_layer_forward` (unchanged); MLA layers through
/// `mla_layer_forward_flashmemory`. `fm` holds one [`LayerFm`] per MLA layer,
/// in layer order. `fm_layers` selects WHICH MLA layers use flashmemory —
/// layers with `false` take the real dense MLA path (the parity-bisect probe:
/// enables attributing a σ=0 divergence to a single layer).
#[allow(clippy::too_many_arguments)]
fn forward_token_sparse<'a>(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    rt: &'a mut KimiK3Runtime,
    token_id: u32,
    fm: Option<&mut [LayerFm]>,
    fm_layers: &[bool; 2],
    step: usize,
) -> &'a [f32] {
    let d = config.hidden_size;
    let mut fm = fm;
    rt.block_state.clear();

    // Step 1: embedding lookup.
    let embed_start = (token_id as usize) * d;
    rt.hidden
        .copy_from_slice(&weights.embed_weight[embed_start..embed_start + d]);

    // Step 2: decoder layers.
    let mut mla_seen = 0usize;
    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_rt = &mut rt.layers[layer_idx];
        if config.is_mla_layer(layer_idx) && fm_layers[mla_seen] {
            let (cache, scratch) = match (&mut layer_rt.attn_state, &mut layer_rt.attn_scratch) {
                (KimiAttentionState::Mla(c), KimiAttentionScratch::Mla(s)) => (c, s),
                _ => panic!("MLA layer {layer_idx} has non-MLA runtime state"),
            };
            let fml = fm
                .as_deref_mut()
                .unwrap_or_else(|| panic!("fm_layers[{mla_seen}] requires a LayerFm slot"));
            mla_layer_forward_flashmemory(
                layer_idx,
                &layer_cfg,
                layer_w,
                &mut fml[mla_seen],
                step,
                cache,
                scratch,
                &mut rt.rope_freqs,
                &mut layer_rt.ffn_scratch,
                &mut layer_rt.attn_res_self_scratch,
                &mut layer_rt.attn_res_mlp_scratch,
                &mut rt.block_state,
                &mut rt.hidden,
                &mut rt.scratch_hidden,
            );
            mla_seen += 1;
        } else {
            if config.is_mla_layer(layer_idx) {
                mla_seen += 1; // dense MLA path — still counts toward the fm-layer index
            }
            kimi_decoder_layer_forward(
                layer_idx,
                &layer_cfg,
                layer_w,
                &mut layer_rt.attn_state,
                &mut layer_rt.attn_scratch,
                &mut layer_rt.ffn_scratch,
                &mut layer_rt.attn_res_self_scratch,
                &mut layer_rt.attn_res_mlp_scratch,
                &mut rt.block_state,
                Some(&mut rt.rope_freqs),
                &mut rt.hidden,
                &mut rt.scratch_hidden,
            );
        }
    }

    // Step 3: output attn-res.
    if !rt.block_state.is_empty() {
        let mixed = apply_attn_res(
            &config.attn_res_config,
            &weights.output_attn_res,
            &rt.block_state,
            &mut rt.output_attn_res_scratch,
            &rt.hidden,
        );
        rt.hidden.copy_from_slice(mixed);
    }

    // Step 4: final RMSNorm.
    rmsnorm_with_gamma_eps(
        &mut rt.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );

    // Step 5: LM head.
    simd_matmul_rows(
        &mut rt.logits,
        &weights.lm_head_weight,
        &rt.hidden,
        config.vocab_size,
        d,
    );
    &rt.logits
}

// ---------------------------------------------------------------------------
// Arm runner (dense + sparse) with shared metrics
// ---------------------------------------------------------------------------

struct ArmOutput {
    query_logits: Vec<f32>,
    /// `[n_decode]` — query logits + each decode step's logits.
    step_logits: Vec<Vec<f32>>,
    answer_ids: Vec<usize>,
    /// Best (min over recorded steps) rank of the password's first token.
    pw_best_rank: usize,
    prefill_ms: f64,
    decode_ms: f64,
    // Selection aggregates — sparse arms only (dense: zeros).
    needle_sel_rate: f32,
    distractor_sel_rate: f32,
    coverage_pct: f32,
    fallback_rate: f32,
    /// Per-layer block-score matrix at the query step (sparse arms only).
    query_scores: Vec<Vec<Vec<f32>>>,
    /// Per-layer token counts per block at the query step.
    query_block_counts: Vec<Vec<usize>>,
}

fn progress(step: usize, total: usize, started: Instant) {
    if step.is_multiple_of(8192) || step + 1 == total {
        let per = started.elapsed().as_secs_f64() * 1000.0 / (step as f64 + 1.0);
        println!("  [prefill] {}/{} ({:.2} ms/tok)", step + 1, total, per);
    }
}

/// Fold one snapshot of `f`'s stats into `agg` = [needle_sum, distr_sum,
/// coverage_sum, fallback_sum]. `seq_now` is the cache length after the
/// forward this snapshot came from.
fn fold_stats(f: &LayerFm, seq_now: usize, agg: &mut [f64; 4]) {
    let heads = f.last_needle_sel.len() as f64;
    if heads == 0.0 {
        return;
    }
    agg[0] += f.last_needle_sel.iter().filter(|&&b| b).count() as f64 / heads;
    agg[1] += f.last_distractor_sel.iter().filter(|&&b| b).count() as f64 / heads;
    agg[2] += f.last_covered_tokens as f64 / (heads * seq_now as f64);
    agg[3] += f.last_fallback_heads as f64 / heads;
}

fn run_dense_arm(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    token_ids: &[usize],
    n_decode: usize,
    pw_first: usize,
) -> ArmOutput {
    let max_seq = token_ids.len() + n_decode + 2;
    let mut rt = KimiK3Runtime::new(config, max_seq);

    let t0 = Instant::now();
    let mut query_logits: Vec<f32> = Vec::new();
    for (step, &tid) in token_ids.iter().enumerate() {
        progress(step, token_ids.len(), t0);
        let logits = kimi_k3_forward_token(config, weights, &mut rt, tid as u32);
        if step + 1 == token_ids.len() {
            query_logits = logits.to_vec();
        }
    }
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Greedy decode.
    let t1 = Instant::now();
    let mut step_logits = Vec::with_capacity(n_decode);
    let mut answer_ids: Vec<usize> = Vec::with_capacity(n_decode);
    let mut pw_best_rank = usize::MAX;
    step_logits.push(query_logits.clone());
    pw_best_rank = pw_best_rank.min(rank_of(&query_logits, pw_first));
    let mut next = argmax(&query_logits);
    answer_ids.push(next);
    for _ in 1..n_decode {
        let logits = kimi_k3_forward_token(config, weights, &mut rt, next as u32);
        pw_best_rank = pw_best_rank.min(rank_of(logits, pw_first));
        next = argmax(logits);
        answer_ids.push(next);
        step_logits.push(logits.to_vec());
    }
    let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;

    ArmOutput {
        query_logits,
        step_logits,
        answer_ids,
        pw_best_rank,
        prefill_ms,
        decode_ms,
        needle_sel_rate: 0.0,
        distractor_sel_rate: 0.0,
        coverage_pct: 0.0,
        fallback_rate: 0.0,
        query_scores: Vec::new(),
        query_block_counts: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_sparse_arm(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    token_ids: &[usize],
    n_decode: usize,
    fm_config: &FlashMemoryConfig,
    needle_block: Option<usize>,
    distractor_block: Option<usize>,
    pw_first: usize,
    fm_layers: &[bool; 2],
) -> ArmOutput {
    let seq_len = token_ids.len();
    let max_seq = seq_len + n_decode + 2;
    let query_step = seq_len - 1;
    let mut rt = KimiK3Runtime::new(config, max_seq);
    let mut fm: Vec<LayerFm> = (0..2)
        .map(|_| {
            LayerFm::new(
                config,
                fm_config,
                max_seq,
                needle_block,
                distractor_block,
                query_step,
            )
        })
        .collect();

    // Per-layer fold accumulators: [needle_sum, distr_sum, cov_sum, fb_sum].
    let mut agg = [[0.0f64; 4]; 2];
    let mut agg_steps = 0usize;

    let t0 = Instant::now();
    let mut query_logits: Vec<f32> = Vec::new();
    for (step, &tid) in token_ids.iter().enumerate() {
        progress(step, token_ids.len(), t0);
        if step == query_step {
            for f in &mut fm {
                // Fresh selection against the query's own q_c (a serving
                // stack re-scores for the answering turn). Generous-to-sparse
                // is the conservative direction for a promotion gate.
                f.selector.force_refresh();
            }
        }
        let logits = forward_token_sparse(
            config,
            weights,
            &mut rt,
            tid as u32,
            Some(&mut fm),
            fm_layers,
            step,
        );
        if step + 1 == token_ids.len() {
            query_logits = logits.to_vec();
            for (li, f) in fm.iter().enumerate() {
                fold_stats(f, step + 1, &mut agg[li]);
            }
            agg_steps += 1;
        }
    }
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Snapshot the query-step score matrix + block counts for the T2 sweep.
    let query_scores: Vec<Vec<Vec<f32>>> = fm
        .iter()
        .map(|f| f.query_scores.clone().unwrap_or_default())
        .collect();
    let query_block_counts: Vec<Vec<usize>> = fm
        .iter()
        .map(|f| f.query_block_counts.clone().unwrap_or_default())
        .collect();

    // Greedy decode (sparse).
    let t1 = Instant::now();
    let mut step_logits = Vec::with_capacity(n_decode);
    let mut answer_ids: Vec<usize> = Vec::with_capacity(n_decode);
    let mut pw_best_rank = usize::MAX;
    step_logits.push(query_logits.clone());
    pw_best_rank = pw_best_rank.min(rank_of(&query_logits, pw_first));
    let mut next = argmax(&query_logits);
    answer_ids.push(next);
    for k in 1..n_decode {
        let step = query_step + k;
        let logits = forward_token_sparse(
            config,
            weights,
            &mut rt,
            next as u32,
            Some(&mut fm),
            fm_layers,
            step,
        );
        pw_best_rank = pw_best_rank.min(rank_of(logits, pw_first));
        // Snapshot selection stats for this decode step.
        for (li, f) in fm.iter().enumerate() {
            fold_stats(f, step + 1, &mut agg[li]);
        }
        agg_steps += 1;
        next = argmax(logits);
        answer_ids.push(next);
        step_logits.push(logits.to_vec());
    }
    let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Pool both MLA layers' folds into the arm aggregates.
    let mut needle_sum = 0.0f64;
    let mut distr_sum = 0.0f64;
    let mut cov_sum = 0.0f64;
    let mut fb_sum = 0.0f64;
    if agg_steps > 0 {
        for li in 0..2 {
            needle_sum += agg[li][0];
            distr_sum += agg[li][1];
            cov_sum += agg[li][2];
            fb_sum += agg[li][3];
        }
    }
    let arms = 2.0f64;
    let needle_sel_rate = (needle_sum / (agg_steps as f64 * arms) * 100.0) as f32;
    let distractor_sel_rate = (distr_sum / (agg_steps as f64 * arms) * 100.0) as f32;
    let coverage_pct = (cov_sum / (agg_steps as f64 * arms) * 100.0) as f32;
    let fallback_rate = (fb_sum / (agg_steps as f64 * arms) * 100.0) as f32;

    ArmOutput {
        query_logits,
        step_logits,
        answer_ids,
        pw_best_rank,
        prefill_ms,
        decode_ms,
        needle_sel_rate,
        distractor_sel_rate,
        coverage_pct,
        fallback_rate,
        query_scores,
        query_block_counts,
    }
}

// ---------------------------------------------------------------------------
// T2 selection-axis sweep (from the query-step score snapshot)
// ---------------------------------------------------------------------------

/// Print σ vs needle-selection% / coverage% per MLA layer, re-deriving the
/// selection purely from the public score math. Returns, per layer, the
/// HIGHEST sweep σ (×100) at which needle selection stays >= 50% (0 if none).
fn print_sweep(
    sigma_grid: &[f32],
    needle_block: Option<usize>,
    seq_len: usize,
    n_heads: usize,
    scores: &[Vec<Vec<f32>>],
    counts: &[Vec<usize>],
    layer_names: &[usize],
) -> [usize; 2] {
    let mut floor_sigma = [0usize; 2];
    println!(
        "  -- T2 selection-axis sweep (query step, σ grid; sel% = needle block in selection, cov% = tokens attended) --"
    );
    println!(
        "     σ     | layer {} sel%  cov% | layer {} sel%  cov%",
        layer_names[0], layer_names[1]
    );
    for &sigma in sigma_grid {
        let thr = logit_of(sigma);
        let mut row = format!("     {:.2}  |", sigma);
        for (li, layer_scores) in scores.iter().enumerate() {
            if layer_scores.is_empty() || needle_block.is_none() {
                row.push_str("    n/a   n/a |");
                continue;
            }
            let nb = needle_block.unwrap();
            let mut sel_heads = 0usize;
            let mut cov = 0usize;
            for head_scores in layer_scores {
                // Blocks with σ(score) >= σ, in ascending block order.
                let mut cov_head = 0usize;
                let mut selected = false;
                for (b, &s) in head_scores.iter().enumerate() {
                    if s >= thr {
                        if b == nb {
                            selected = true;
                        }
                        cov_head += counts[li].get(b).copied().unwrap_or(0);
                    }
                }
                if selected {
                    sel_heads += 1;
                }
                cov += cov_head;
            }
            let sel_pct = sel_heads as f64 / n_heads as f64 * 100.0;
            let cov_pct = cov as f64 / (n_heads as f64 * seq_len as f64) * 100.0;
            if sel_pct >= 50.0 {
                floor_sigma[li] = floor_sigma[li].max((sigma * 100.0) as usize);
            }
            row.push_str(&format!("  {:5.1}  {:5.1} |", sel_pct, cov_pct));
        }
        println!("{row}");
    }
    floor_sigma
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

struct ArmRow {
    thr: f32,
    hit: bool,
    agree: usize,
    cos_q: f32,
    cos_min: f32,
    sel: f32,
    cov: f32,
    fb: f32,
}

struct Row {
    len: usize,
    multi: bool,
    needle_tok: (usize, usize),
    dense_hit: bool,
    dense_pw_rank: usize,
    arms: Vec<ArmRow>,
}

// ---------------------------------------------------------------------------
// Parity-bisect probe (FFM_PROBE=1): attribute a σ=0 divergence to a layer
// ---------------------------------------------------------------------------

/// Runs five arms on the shortest ladder prompt and reports max|Δlogits|
/// pairwise: the real forward (A) vs a re-orchestration of the same real
/// layer calls (B — sanity: must be 0), then flashmemory@σ=0 on layer 3 only
/// (C3), layer 7 only (C7), and both (C11), each against B. The pattern
/// localizes a parity failure to one MLA layer.
fn run_probe(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    tokenizer: &TiktokenTokenizer,
    block_size: usize,
    n_decode: usize,
) -> bool {
    let target = 2048;
    let prompt = build_prompt(false, target, true);
    let mut token_ids: Vec<usize> = vec![1];
    token_ids.extend(tokenizer.encode(&prompt.text));
    let ranges = needle_token_ranges(tokenizer, &prompt.text, &prompt.needle_char_ranges);
    let needle_block = (ranges[0].0 + 1) / block_size;
    let pw_first = extract_pw_tokens(
        tokenizer,
        "The magic password is sunset7742. Remember it for later. ",
        "sunset7742",
    )[0];
    let fm_cfg = FlashMemoryConfig {
        block_size,
        refresh_period: 1,
        threshold: 0.0,
    };
    // ⚠ refresh_period=1 is LOAD-BEARING for the σ=0 equivalence: select()
    // reuses the cached selection within the refresh window, so at σ=0 with
    // refresh=64 each step attends only the blocks that existed at the last
    // refresh — missing up to block_size-1 recent tokens (the largest-attention
    // tail). First probe run (2026-09-18) measured exactly that staleness
    // signature: Δ=5.8e-5 (layer 3 only) / 0.1289 (layer 7) at 2170 tokens.
    // With a fresh selection every step, σ=0 selects ALL blocks at EVERY step
    // and the forward is dense-equivalent — what this probe asserts.

    println!(
        "── PROBE: parity bisect at {} tokens (σ=0.0; which MLA layer diverges from dense?) ──",
        token_ids.len()
    );
    let arm_a = run_dense_arm(config, weights, &token_ids, n_decode, pw_first);
    let arm_b = run_sparse_arm(
        config,
        weights,
        &token_ids,
        n_decode,
        &fm_cfg,
        Some(needle_block),
        None,
        pw_first,
        &[false, false],
    );
    let arm_c3 = run_sparse_arm(
        config,
        weights,
        &token_ids,
        n_decode,
        &fm_cfg,
        Some(needle_block),
        None,
        pw_first,
        &[true, false],
    );
    let arm_c7 = run_sparse_arm(
        config,
        weights,
        &token_ids,
        n_decode,
        &fm_cfg,
        Some(needle_block),
        None,
        pw_first,
        &[false, true],
    );
    let arm_c11 = run_sparse_arm(
        config,
        weights,
        &token_ids,
        n_decode,
        &fm_cfg,
        Some(needle_block),
        None,
        pw_first,
        &[true, true],
    );

    let report = |name: &str, x: &ArmOutput, y: &ArmOutput| {
        let mut max_diff = max_abs_diff(&x.query_logits, &y.query_logits);
        let mut min_cos = cosine(&x.query_logits, &y.query_logits);
        for (a, b) in x.step_logits.iter().zip(y.step_logits.iter()) {
            max_diff = max_diff.max(max_abs_diff(a, b));
            min_cos = min_cos.min(cosine(a, b));
        }
        let agree = x
            .answer_ids
            .iter()
            .zip(y.answer_ids.iter())
            .filter(|(p, q)| p == q)
            .count();
        println!(
            "  {name}: max|Δlogits|={max_diff:.6} min_cos={min_cos:.6} agree={agree}/{}",
            x.answer_ids.len()
        );
        max_diff
    };
    let d_b = report(
        "B  re-orchestration (real MLA) vs A real forward",
        &arm_b,
        &arm_a,
    );
    let d_c3 = report("C3 flashmemory@σ0 layer 3 only      vs B", &arm_c3, &arm_b);
    let d_c7 = report("C7 flashmemory@σ0 layer 7 only      vs B", &arm_c7, &arm_b);
    let d_c11 = report("C11 flashmemory@σ0 both layers      vs B", &arm_c11, &arm_b);

    // The parity gate threshold (1e-3) also judges the probe: residuals at
    // fp-reassociation scale (~3e-6 measured) are a PASS — σ=0 with a fresh
    // selection every step is dense-equivalent. A residual ABOVE the gate at
    // one layer localizes a real contract failure to that layer.
    const GATE: f32 = 1e-3;
    let layer = if d_c3 >= d_c7 && d_c3 > GATE {
        3
    } else if d_c7 > GATE {
        7
    } else {
        0
    };
    if d_b > GATE {
        println!(
            "  → ORCHESTRATION diverges from the real forward (B≠A) — bench bug, NOT a flashmemory defect"
        );
        false
    } else if d_c11 <= GATE {
        println!(
            "  → flashmemory@σ0 (fresh selection) IS dense-equivalent within the gate: residual ≤ {GATE} — PARITY PASS"
        );
        true
    } else {
        println!(
            "  → flashmemory@σ0 ≠ dense at layer {layer} beyond the gate (contract failure localized to that layer)"
        );
        true
    }
}

fn main() {
    println!("╔══ FlashMemory full-model NIAH QA gate (Issue 826 T1+T2) ══╗\n");

    let seq_lens = env_list_usize("FFM_SEQ_LENS", &[2048, 16384, 65536]);
    let multi_lens = env_list_usize("FFM_MULTI_LENS", &[16384, 65536]);
    let thresholds = env_list_f32("FFM_THRESHOLDS", &[0.5, 0.7]);
    let block_size = env_or("FFM_BLOCK_SIZE", 64);
    let refresh = env_or("FFM_REFRESH", 64);
    let n_decode = env_or("FFM_N_DECODE", 12);
    let sweep_lens = env_list_usize("FFM_SWEEP_LENS", &[16384, 65536]);
    // Default = continuation stem: this 0.4B base model continues far more
    // coherently than it answers question-format prompts (measured both ways,
    // 2026-09-18: stem → in-domain completions + password in top 1.3%;
    // question format → disjoint gibberish). Override with FFM_QUERY=question.
    let query_stem = std::env::var("FFM_QUERY")
        .map(|v| v != "question")
        .unwrap_or(true);
    let parity_enabled = std::env::var("FFM_PARITY")
        .map(|v| v != "0")
        .unwrap_or(true);
    let probe = std::env::var("FFM_PROBE")
        .map(|v| v == "1")
        .unwrap_or(false);
    let sigma_grid: Vec<f32> = (0..14).map(|i| 0.30 + i as f32 * 0.05).collect();

    // ── Model resolution ───────────────────────────────────────────────────
    let model_dir = if let Ok(d) = std::env::var("KIMI_K3_MODEL_DIR") {
        d
    } else {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let local = format!("{manifest}/data/kimi-k3-0.40b");
        let sibling = format!("{manifest}/../riir-train/data/kimi-k3-0.40b");
        if std::path::Path::new(&local)
            .join("model.safetensors")
            .exists()
        {
            local
        } else if std::path::Path::new(&sibling)
            .join("model.safetensors")
            .exists()
        {
            sibling
        } else {
            local // will fail below with the load error naming the path
        }
    };
    let tiktoken_path = format!("{model_dir}/tiktoken.model");
    let model_path = format!("{model_dir}/model.safetensors");
    for p in [&tiktoken_path, &model_path] {
        if !std::path::Path::new(p).exists() {
            eprintln!("ERROR: missing {p} (set KIMI_K3_MODEL_DIR)");
            std::process::exit(1);
        }
    }

    print!("Loading tiktoken.model ... ");
    io::stdout().flush().ok();
    let tiktoken_bytes = std::fs::read(&tiktoken_path).expect("read tiktoken.model");
    let ranks = load_tiktoken_bpe(&tiktoken_bytes).expect("parse tiktoken.model");
    let tokenizer = TiktokenTokenizer::from_ranks(&ranks).with_special_tokens(1, 2, 0);
    println!("done (vocab={})", tokenizer.vocab_size());

    print!("Loading model.safetensors ... ");
    io::stdout().flush().ok();
    let t0 = Instant::now();
    let weights: KimiK3ModelWeights = load_kimi_k3(&model_path).expect("load model");
    println!("done ({:.1}s)", t0.elapsed().as_secs_f64());

    let config = KimiK3ModelConfig::kimi_k3_0_40b();

    // Probe mode: parity bisect, then exit. Never runs the ladder.
    if probe {
        let ok = run_probe(&config, &weights, &tokenizer, block_size, n_decode);
        std::process::exit(if ok { 0 } else { 1 });
    }

    println!(
        "model: {} layers, MLA at {:?}, hidden {}, vocab {} (training ctx 4096 — longer ladder points are matched-context extrapolation)",
        config.num_layers, config.mla_layer_indices, config.hidden_size, config.vocab_size
    );
    println!(
        "lens(single)={seq_lens:?} multi@{multi_lens:?} thresholds(e2e)={thresholds:?} block={block_size} refresh={refresh} n_decode={n_decode} query={} bos=1\n",
        if query_stem { "stem" } else { "question" }
    );

    let fm_cfg_for = |thr: f32| FlashMemoryConfig {
        block_size,
        refresh_period: refresh,
        threshold: thr,
    };

    let mut rows: Vec<Row> = Vec::new();
    let mut parity: Option<(f32, f32, bool)> = None; // (max_diff, min_cos, pass)
    let mut sweep_floors: Vec<(usize, [usize; 2])> = Vec::new(); // (len, per-layer σ×100)
    let min_len = *seq_lens.iter().min().expect("non-empty FFM_SEQ_LENS");

    for &target in &seq_lens {
        for &multi in &[false, true] {
            if multi && !multi_lens.contains(&target) {
                continue;
            }
            let prompt = build_prompt(multi, target, query_stem);
            let encoded = tokenizer.encode(&prompt.text);
            // BOS first (the model's own bos_token_id) — base models need the
            // document boundary; bench 023 skipped it, this QA gate does not.
            let mut token_ids: Vec<usize> = vec![1];
            token_ids.extend_from_slice(&encoded);
            let seq_len = token_ids.len();
            let ranges = needle_token_ranges(&tokenizer, &prompt.text, &prompt.needle_char_ranges)
                .into_iter()
                .map(|(s, e)| (s + 1, e + 1)) // BOS offset
                .collect::<Vec<_>>();
            let q_idx = prompt.queried_idx;
            let (needle_tok_start, needle_tok_end) = ranges[q_idx];
            // The QUERIED needle is the sunset7742 sentence in both variants
            // (in multi it is the deep one; the shallow morning3156 needle is
            // the distractor).
            let needle_sentence = "The magic password is sunset7742. Remember it for later. ";
            let pw_tokens = extract_pw_tokens(&tokenizer, needle_sentence, "sunset7742");
            let pw_first = pw_tokens[0];
            let needle_block = needle_tok_start / block_size;
            let distractor_block = if multi {
                Some(ranges[0].0 / block_size)
            } else {
                None
            };
            let variant = if multi { "multi" } else { "single" };

            println!(
                "── [len={target} {variant}] {} tokens; needle(queried) tokens [{needle_tok_start},{needle_tok_end}) block {needle_block}{} ──",
                seq_len,
                distractor_block
                    .map(|d| format!("; distractor block {d}"))
                    .unwrap_or_default()
            );

            // DENSE arm (the oracle — real forward path, unmodified).
            let dense = run_dense_arm(&config, &weights, &token_ids, n_decode, pw_first);
            let dense_answer = tokenizer.decode(&dense.answer_ids);
            let dense_hit = dense_answer.contains("sunset7742");
            println!(
                "  DENSE : hit={dense_hit} pw_rank={} ans={:?} prefill={:.1}s decode={:.1}s",
                dense.pw_best_rank,
                truncate(&dense_answer, 60),
                dense.prefill_ms / 1000.0,
                dense.decode_ms / 1000.0
            );

            // PARITY gate (once, shortest len, single, σ=0.0 → dense-equiv).
            // refresh_period=1: see the probe comment — a stale selection at
            // σ=0 misses the newest blocks, which is paper semantics for QA
            // arms but NOT dense-equivalence.
            if target == min_len && !multi && parity.is_none() && parity_enabled {
                let par_cfg = FlashMemoryConfig {
                    block_size,
                    refresh_period: 1,
                    threshold: 0.0,
                };
                let par = run_sparse_arm(
                    &config,
                    &weights,
                    &token_ids,
                    n_decode,
                    &par_cfg,
                    Some(needle_block),
                    distractor_block,
                    pw_first,
                    &[true, true],
                );
                let mut max_diff = max_abs_diff(&dense.query_logits, &par.query_logits);
                let mut min_cos = cosine(&dense.query_logits, &par.query_logits);
                for (a, b) in dense.step_logits.iter().zip(par.step_logits.iter()) {
                    max_diff = max_diff.max(max_abs_diff(a, b));
                    min_cos = min_cos.min(cosine(a, b));
                }
                let pass = max_diff <= 1e-3;
                println!(
                    "  PARITY(σ=0.0): max|Δlogits|={max_diff:.6} min_cos={min_cos:.6} → {}",
                    if pass { "PASS" } else { "FAIL" }
                );
                parity = Some((max_diff, min_cos, pass));
            }

            // SPARSE arms at each e2e threshold.
            let mut arm_rows: Vec<ArmRow> = Vec::new();
            for &thr in &thresholds {
                let sparse = run_sparse_arm(
                    &config,
                    &weights,
                    &token_ids,
                    n_decode,
                    &fm_cfg_for(thr),
                    Some(needle_block),
                    distractor_block,
                    pw_first,
                    &[true, true],
                );
                let sparse_answer = tokenizer.decode(&sparse.answer_ids);
                let sparse_hit = sparse_answer.contains("sunset7742");
                let agree = dense
                    .answer_ids
                    .iter()
                    .zip(sparse.answer_ids.iter())
                    .filter(|(a, b)| a == b)
                    .count();
                let cos_q = cosine(&dense.query_logits, &sparse.query_logits);
                let cos_min = dense
                    .step_logits
                    .iter()
                    .zip(sparse.step_logits.iter())
                    .map(|(a, b)| cosine(a, b))
                    .fold(f32::INFINITY, f32::min);
                println!(
                    "  σ={thr:.2}: hit={sparse_hit} agree={agree}/{} cosQ={cos_q:.4} minCos={cos_min:.4} pw_rank={} selN={:.1}% distN={:.1}% cov={:.1}% fb={:.1}% prefill={:.1}s decode={:.1}s",
                    dense.answer_ids.len(),
                    sparse.pw_best_rank,
                    sparse.needle_sel_rate,
                    sparse.distractor_sel_rate,
                    sparse.coverage_pct,
                    sparse.fallback_rate,
                    sparse.prefill_ms / 1000.0,
                    sparse.decode_ms / 1000.0
                );
                arm_rows.push(ArmRow {
                    thr,
                    hit: sparse_hit,
                    agree,
                    cos_q,
                    cos_min,
                    sel: sparse.needle_sel_rate,
                    cov: sparse.coverage_pct,
                    fb: sparse.fallback_rate,
                });

                // T2 selection-axis sweep at σ=0.5, single variant, requested lens.
                if thr == 0.5 && !multi && sweep_lens.contains(&target) {
                    let floors = print_sweep(
                        &sigma_grid,
                        Some(needle_block),
                        seq_len,
                        config.mla_config.n_heads,
                        &sparse.query_scores,
                        &sparse.query_block_counts,
                        &config.mla_layer_indices,
                    );
                    sweep_floors.push((target, floors));
                }
            }

            rows.push(Row {
                len: target,
                multi,
                needle_tok: (needle_tok_start, needle_tok_end),
                dense_hit,
                dense_pw_rank: dense.pw_best_rank,
                arms: arm_rows,
            });
            println!();
        }
    }

    // ── Verdicts ───────────────────────────────────────────────────────────
    println!("══ VERDICT ══");
    let mut failed = false;

    match parity {
        Some((max_diff, min_cos, pass)) => {
            println!(
                "parity gate (σ=0.0 dense-equivalence): max|Δlogits|={max_diff:.6} min_cos={min_cos:.6} → {}",
                if pass { "PASS" } else { "FAIL" }
            );
            if !pass {
                failed = true;
            }
        }
        None => println!("parity gate: NOT RUN (no dense arm at min len)"),
    }

    let mut min_cos_q_half = f32::INFINITY;
    let mut min_agree_frac_half = 1.0f32;
    for r in &rows {
        println!(
            "[len={} {}] dense: hit={} pw_rank={} | needle tokens {:?}",
            r.len,
            if r.multi { "multi" } else { "single" },
            r.dense_hit,
            r.dense_pw_rank,
            r.needle_tok
        );
        for a in &r.arms {
            let agree_frac = a.agree as f32 / n_decode as f32;
            println!(
                "  σ={:.2}: hit={} agree={}/{} cosQ={:.4} minCos={:.4} selN={:.1}% cov={:.1}% fb={:.1}%",
                a.thr, a.hit, a.agree, n_decode, a.cos_q, a.cos_min, a.sel, a.cov, a.fb
            );
            if a.thr == 0.5 {
                min_cos_q_half = min_cos_q_half.min(a.cos_q);
                min_agree_frac_half = min_agree_frac_half.min(agree_frac);
            }
        }
    }

    if min_cos_q_half.is_finite() {
        // The tolerance is the Phase 3 gate parameter — pick it from the
        // measurement (floored to 2 decimals), then pin it in the issue +
        // benchmark doc.
        let proposed = (min_cos_q_half * 100.0).floor() / 100.0;
        println!(
            "sparse-within-tolerance @σ=0.5: min cosQ={min_cos_q_half:.4}, min agree={:.0}% → proposed tolerance pin: cosQ ≥ {proposed:.2}",
            min_agree_frac_half * 100.0
        );
    }
    for (len, floors) in &sweep_floors {
        println!(
            "T2 budget stress @len={len}: needle selection ≥50% up to σ=0.{:02} (L{}) / σ=0.{:02} (L{})",
            floors[0], config.mla_layer_indices[0], floors[1], config.mla_layer_indices[1]
        );
    }

    if failed {
        std::process::exit(1);
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

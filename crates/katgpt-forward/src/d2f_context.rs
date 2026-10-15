//! D2F Inference Substrate — Plan 398 (2026-07-05).
//!
//! Extracted from root `src/dllm.rs`. This module hosts the **inference-only**
//! substrate shared by the d2f cluster (`speculative/d2f.rs`,
//! `speculative/d2f_verifier.rs`, `speculative/diffusion_sampler.rs`):
//!
//! - `D2fContext` — pre-allocated flat buffers for zero-alloc D2F denoising
//! - `forward_block_causal_with` — block-causal attention forward kernel that
//!   writes into `D2fContext::logits_flat`
//! - `attention_forward_safe_into` — zero-alloc attention helper called by
//!   `forward_block_causal_with` (and re-exported back to root's training
//!   code via `pub use katgpt_forward::...`)
//! - `denoising_accuracy` — fraction of correctly recovered tokens
//!
//! ## Why this lives here
//!
//! Root's `src/dllm.rs` (4782 LOC) mixes training infrastructure
//! (`train_mini_dllm`, `train_mini_set_causal`, `evaluate_set_causal_nelbo`,
//! `generate_pattern_dataset`, `evaluate_accuracy`) with the inference
//! substrate above. The training code must stay in root (it's a research
//! concern, not a published-API concern, and per the modelless-first mandate
//! doesn't belong in a published leaf crate). The inference substrate, on
//! the other hand, blocks the entire d2f cluster from leaving root — so it
//! moves here.
//!
//! Root re-exports these via `pub use katgpt_forward::d2f_context::{...}`,
//! preserving every historical `katgpt_rs::dllm::D2fContext` /
//! `katgpt_rs::dllm::forward_block_causal_with` /
//! `katgpt_rs::dllm::denoising_accuracy` import path.
//!
//! ## DRY note
//!
//! `attention_forward_safe_into` has 5 callers in the codebase:
//! - 4 stay in root (`forward_bidirectional_positions_into`, `forward_save`,
//!   `forward_block_causal_positions`, `attention_forward_safe`)
//! - 1 moves with this file (`forward_block_causal_with`)
//!
//! The function is therefore `pub` here (workspace-internal — this crate is
//! `publish = false`) and root imports it via `use katgpt_forward::...`. This
//! preserves a single source of truth rather than duplicating the body.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use katgpt_core::simd;
use katgpt_core::types::{Config, kv_dim, matmul, matmul_relu, rmsnorm};
use katgpt_transformer::TransformerWeights;

// ═══════════════════════════════════════════════════════════════
// Zero-Alloc D2F Context + Forward
// ═══════════════════════════════════════════════════════════════

/// Pre-allocated buffers for zero-alloc D2F denoising.
///
/// Unlike `SpeculativeContext` (single-token autoregressive), D2F processes
/// all positions in a block simultaneously, so we use flat 2D buffers
/// indexed by `[p * dim..(p+1) * dim]`.
pub struct D2fContext {
    /// Flat KV cache: `[max_seq * kv_dim]`.
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    /// Normalized embeddings per position: `[max_seq * n_embd]`.
    pub x_norm: Vec<f32>,
    /// Residual (pre-norm) embeddings per position: `[max_seq * n_embd]`.
    pub xr: Vec<f32>,
    /// Flat logits: `[max_seq * vocab_size]`.
    pub logits_flat: Vec<f32>,
    /// Temp buffer for per-position embedding: `[n_embd]`.
    pub x_buf: Vec<f32>,
    /// Temp buffer for query: `[n_embd]`.
    pub q_buf: Vec<f32>,
    /// Temp buffer for key: `[kv_dim]`.
    pub k_buf: Vec<f32>,
    /// Temp buffer for value: `[kv_dim]`.
    pub v_buf: Vec<f32>,
    /// Temp buffer for attention projection: `[n_embd]`.
    pub x_proj_buf: Vec<f32>,
    /// Temp buffer for MLP hidden: `[mlp_hidden]`.
    pub hidden_buf: Vec<f32>,
    /// Temp buffer for MLP output: `[n_embd]`.
    pub x_mlp_buf: Vec<f32>,
    /// Temp buffer for single-position logits: `[vocab_size]`.
    pub logits_buf: Vec<f32>,
    /// Attention output buffer: `[n_head * head_dim]` (reused per position).
    pub attn_out_buf: Vec<f32>,
    /// Attention weights buffer: `[n_head * max_seq]` (reused per position).
    pub attn_weights_buf: Vec<f32>,
    /// Attention scores buffer: `[max_seq]` (reused per position).
    pub attn_scores_buf: Vec<f32>,
    /// Cached logits from previous denoising step: `[max_seq * vocab_size]`.
    /// Used by DPM-Solver++(2M) multistep extrapolation (Plan 078 T10.5).
    pub prev_logits_flat: Vec<f32>,
    /// Cached logits from two steps ago: `[max_seq * vocab_size]`.
    /// Second cache for multistep logit extrapolation (Plan 078 T10.5).
    pub prev_prev_logits_flat: Vec<f32>,
    /// Residual embeddings for RCD injection: `[max_seq * n_embd]`.
    /// Stores interpolated residual embeddings for masked positions.
    /// Written after token commitment, read during next step's input construction.
    #[cfg(feature = "rcd_residual")]
    pub residual_embeddings: Vec<f32>,
    /// Entropy weights per position: `[max_seq]`.
    /// α_i values computed from marginal distributions.
    #[cfg(feature = "rcd_residual")]
    pub entropy_weights: Vec<f32>,
    /// Softmax scratch buffer for RCD: `[vocab_size]`.
    #[cfg(feature = "rcd_residual")]
    pub rcd_softmax_scratch: Vec<f32>,
    /// Probe-guidance scratch (Issue 865 T1): weak-side probe logits
    /// `[max_seq * vocab_size]`. Written by the installed probe each
    /// denoising step, consumed by [`apply_probe_guidance`] — always fully
    /// written before read, never reset (same contract as the other temp
    /// buffers).
    #[cfg(feature = "probe_guidance")]
    pub probe_logits_flat: Vec<f32>,
    /// Probe-guidance strength λ (Issue 865 T1). 1.0 = off — the combine is
    /// skipped entirely, which is what makes guided and unguided decode
    /// bit-identical at the default (G1).
    #[cfg(feature = "probe_guidance")]
    pub guidance_lambda: f32,
    /// Installed weak-side probe, if any.
    #[cfg(feature = "probe_guidance")]
    pub weak_probe: Option<Box<dyn WeakLogitProbe>>,
    /// Post-attention-residual taps (Issue 865 T2 / 869 T2), LAYERED: one
    /// plane per configured tap layer ([`D2fContext::probe_tap_layers`],
    /// slot order), each `[max_seq * n_embd]` — attention output + input
    /// residual of that layer, BEFORE the MLP refinement. These are the
    /// earliest context-carrying points per layer: at masked positions the
    /// pre-layer input residual (`xr`) is just the mask-token embedding plus
    /// position — no context — so weak probes read HERE. Written only when
    /// [`D2fContext::probe_tap_capture`] is armed (the decode core arms it
    /// exactly when guidance is active); a pure copy, so arming never moves
    /// a logit. Default tap set `[0]` = exactly the pre-869 single-tap
    /// buffer (slot 0 at offset 0 — the layer-0 post-attention residual).
    #[cfg(feature = "probe_guidance")]
    pub probe_tap_flat: Vec<f32>,
    /// Arm the per-position tap copy in the forward kernel. Off by default;
    /// the decode core sets it before the forward when guidance is live, so
    /// the G1 (λ = 1) path pays nothing.
    #[cfg(feature = "probe_guidance")]
    pub probe_tap_capture: bool,
    /// Tap planes to capture (Issue 869 T2): the POST-ATTENTION residual is
    /// captured for every layer index in this set, into
    /// [`D2fContext::probe_tap_flat`]'s slot of the same order. Default
    /// `[0]` — exactly the pre-869 single-tap behavior. Maintain via
    /// [`D2fContext::set_probe_tap_layers`] (sorts, dedups, validates against
    /// the decode depth, and resizes the buffer); a hand-written value
    /// desyncs the slot count from the buffer size.
    #[cfg(feature = "probe_guidance")]
    pub probe_tap_layers: Vec<usize>,
    // usize fields after all Vec<f32> fields to eliminate inter-field padding.
    /// Elements per tap plane (`max_seq * n_embd`) — the slot stride of
    /// `probe_tap_flat` (Issue 869 T2).
    #[cfg(feature = "probe_guidance")]
    pub probe_tap_plane: usize,
    /// Total layer capacity (the config's `n_layer`) — the KV planes and the
    /// `set_decode_layers` ceiling are sized by this at construction
    /// (Issue 869).
    pub n_layer_total: usize,
    /// How many trunk layers the decode kernel actually runs (Issue 869 T1).
    /// **Defaults to `n_layer_total`** since Issue 869 T5 (the training-side
    /// per-layer migration landed — every layer of a `train_mini_dllm`
    /// trained model is trained, so decoding through all of them is the
    /// consistent semantics; the pre-T5 lane was single-layer end-to-end and
    /// defaulted to 1). Shrink explicitly with [`D2fContext::set_decode_layers`]
    /// for a truncated-trunk decode.
    pub decode_n_layer: usize,
    /// Number of positions with committed KV cache entries.
    /// Positions `[0..committed_len)` are valid and won't be recomputed.
    pub committed_len: usize,
}

impl D2fContext {
    /// Create a new context with buffers sized for the given config.
    pub fn new(config: &Config) -> Self {
        let n = config.n_embd;
        let kvd = kv_dim(config);
        let max_seq = config.block_size;
        let vocab = config.vocab_size;
        let hidden = config.mlp_hidden;

        Self {
            // Issue 869: per-layer KV planes — `layers[l]`'s plane lives at
            // `l * (block_size * kvd)`. Plane 0's layout is byte-identical
            // to the pre-869 single-plane cache (bit-identity pinned by the
            // Issue-865 gates, which re-ran on that change); T5 made the
            // decode depth default to `n_layer`, so all planes are live for
            // multi-layer configs.
            k_cache: vec![0.0f32; config.n_layer * max_seq * kvd],
            v_cache: vec![0.0f32; config.n_layer * max_seq * kvd],
            x_norm: vec![0.0f32; max_seq * n],
            xr: vec![0.0f32; max_seq * n],
            logits_flat: vec![0.0f32; max_seq * vocab],
            x_buf: vec![0.0f32; n],
            q_buf: vec![0.0f32; n],
            k_buf: vec![0.0f32; kvd],
            v_buf: vec![0.0f32; kvd],
            x_proj_buf: vec![0.0f32; n],
            hidden_buf: vec![0.0f32; hidden],
            x_mlp_buf: vec![0.0f32; n],
            logits_buf: vec![0.0f32; vocab],
            attn_out_buf: vec![0.0f32; n],
            attn_weights_buf: vec![0.0f32; config.n_head * max_seq],
            attn_scores_buf: vec![0.0f32; max_seq],
            prev_logits_flat: vec![0.0f32; max_seq * vocab],
            prev_prev_logits_flat: vec![0.0f32; max_seq * vocab],
            #[cfg(feature = "rcd_residual")]
            residual_embeddings: vec![0.0f32; max_seq * n],
            #[cfg(feature = "rcd_residual")]
            entropy_weights: vec![0.0f32; max_seq],
            #[cfg(feature = "rcd_residual")]
            rcd_softmax_scratch: vec![0.0f32; vocab],
            #[cfg(feature = "probe_guidance")]
            probe_logits_flat: vec![0.0f32; max_seq * vocab],
            #[cfg(feature = "probe_guidance")]
            guidance_lambda: 1.0,
            #[cfg(feature = "probe_guidance")]
            weak_probe: None,
            #[cfg(feature = "probe_guidance")]
            probe_tap_flat: vec![0.0f32; max_seq * n],
            #[cfg(feature = "probe_guidance")]
            probe_tap_capture: false,
            #[cfg(feature = "probe_guidance")]
            probe_tap_layers: vec![0],
            #[cfg(feature = "probe_guidance")]
            probe_tap_plane: max_seq * n,
            n_layer_total: config.n_layer,
            decode_n_layer: config.n_layer,
            committed_len: 0,
        }
    }

    /// Reset flat buffers for a new forward pass.
    ///
    /// Temp buffers (`x_buf`, `q_buf`, etc.) need not be reset since they are
    /// always fully written before being read.
    pub fn reset(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.x_norm.fill(0.0);
        self.xr.fill(0.0);
        self.logits_flat.fill(0.0);
        self.committed_len = 0;
        self.prev_logits_flat.fill(0.0);
        self.prev_prev_logits_flat.fill(0.0);
        #[cfg(feature = "rcd_residual")]
        {
            self.residual_embeddings.fill(0.0);
            self.entropy_weights.fill(0.0);
        }
    }

    /// Commit KV cache entries for positions `[0..len)`.
    /// After calling this, subsequent forward passes will skip KV computation
    /// for these positions.
    #[inline]
    pub fn commit(&mut self, len: usize) {
        self.committed_len = len;
    }

    /// Set how many trunk layers the decode kernel runs (Issue 869 T1).
    ///
    /// `n` must be in `[1, n_layer_total]`. Panics loudly (never silently
    /// truncates) outside that range, and when any configured probe tap layer
    /// would fall outside the new depth — shrink the tap set first in that
    /// case (`set_probe_tap_layers`).
    pub fn set_decode_layers(&mut self, n: usize) {
        assert!(n >= 1, "decode depth must be at least 1, got {n}");
        assert!(
            n <= self.n_layer_total,
            "decode depth {n} exceeds the config's layer count {} — the weights/context disagree with the request",
            self.n_layer_total
        );
        #[cfg(feature = "probe_guidance")]
        if let Some(&tl) = self.probe_tap_layers.iter().find(|&&tl| tl >= n) {
            panic!(
                "decode depth {n} would orphan tap layer {tl} (taps {:?}) — call set_probe_tap_layers with layers < {n} first",
                self.probe_tap_layers
            );
        }
        self.decode_n_layer = n;
    }

    /// Set which layer indices the forward captures taps for (Issue 869 T2).
    ///
    /// Every index must be `< decode_n_layer` (raise it first with
    /// [`D2fContext::set_decode_layers`] if needed — a tap deeper than the
    /// decode depth can never be written, so asking for one is a loud error,
    /// never a silent zero plane). The set is sorted + deduplicated, and
    /// `probe_tap_flat` is resized to `layers.len()` planes. The slot order
    /// afterwards is the sorted set order; `ProbeCtx::tap_layers` carries it
    /// to probes so artifacts find their plane by layer index.
    #[cfg(feature = "probe_guidance")]
    pub fn set_probe_tap_layers(&mut self, layers: &[usize]) {
        assert!(!layers.is_empty(), "the tap set must not be empty");
        if let Some(&bad) = layers.iter().find(|&&l| l >= self.decode_n_layer) {
            panic!(
                "tap layer {bad} is at or beyond the decode depth {} — call set_decode_layers({}) first",
                self.decode_n_layer,
                bad + 1
            );
        }
        let mut set = layers.to_vec();
        set.sort_unstable();
        set.dedup();
        self.probe_tap_flat
            .resize(set.len() * self.probe_tap_plane, 0.0);
        self.probe_tap_layers = set;
    }
}

// ═══════════════════════════════════════════════════════════════
// Attention helper — shared by 5 root callers + 1 in-file caller
// ═══════════════════════════════════════════════════════════════

/// Zero-alloc fused attention head with GQA support.
///
/// Writes attention output (`[n_head * head_dim]`) into `attn_out`, attention
/// weights (`[n_head * seq_len]`) into `all_weights`, and reuses `scores`
/// (`[seq_len]`) as scratch. All buffers must be pre-sized by the caller.
///
/// `pub` (not `pub(crate)`) so root's `dllm.rs` training code can import it
/// via `use katgpt_forward::attention_forward_safe_into;` — preserves a
/// single source of truth across the 5 callers in the codebase.
pub fn attention_forward_safe_into(
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    kv_dim: usize,
    seq_len: usize,
    scale: f32,
    attn_out: &mut [f32],
    all_weights: &mut [f32],
    scores: &mut [f32],
) {
    debug_assert!(attn_out.len() >= n_head * head_dim);
    debug_assert!(all_weights.len() >= n_head * seq_len);
    debug_assert!(scores.len() >= seq_len);

    attn_out[..n_head * head_dim].fill(0.0);

    // Pre-slice the K/V planes to exactly `seq_len` rows once, so the per-head /
    // per-t walks use `chunks_exact` instead of re-deriving `t * kv_dim + kv_off`
    // (two bounds checks per position per head). The `zip`s below are length-safe
    // by construction: `scores[..seq_len]` has exactly `seq_len` elements and
    // `k_rows`/`v_rows` have exactly `seq_len * kv_dim` bytes, hence exactly
    // `seq_len` chunks of `kv_dim` — so no iteration is silently dropped.
    let k_rows = &k_all[..seq_len * kv_dim];
    let v_rows = &v_all[..seq_len * kv_dim];

    for h in 0..n_head {
        let kv_group = h * n_kv_head / n_head;
        let q_off = h * head_dim;
        let kv_off = kv_group * head_dim;
        // `q` row and the `attn_out` row are loop-invariant across `t`: hoist the
        // slicing (and its bounds check) out of the inner loops.
        let q_head = &q[q_off..q_off + head_dim];

        // Compute scores (reuse buffer across heads)
        let mut max_score = f32::NEG_INFINITY;
        for (dst, k_row) in scores[..seq_len]
            .iter_mut()
            .zip(k_rows.chunks_exact(kv_dim))
        {
            let s = simd::simd_dot_f32(q_head, &k_row[kv_off..kv_off + head_dim], head_dim) * scale;
            *dst = s;
            if s > max_score {
                max_score = s;
            }
        }

        // Softmax (SIMD batch exp + sum)
        simd::simd_add_scalar_inplace(&mut scores[..seq_len], -max_score);
        simd::simd_exp_inplace(&mut scores[..seq_len]);
        let sum_exp = simd::simd_sum_f32(&scores[..seq_len]);
        let inv_sum = 1.0 / sum_exp;
        simd::simd_scale_inplace(&mut scores[..seq_len], inv_sum);
        all_weights[h * seq_len..h * seq_len + seq_len].copy_from_slice(&scores[..seq_len]);

        // Weighted value sum: accumulate per-position scaled value rows (SIMD-friendly)
        // Loop order: t outer → contiguous v_all row access, better cache locality.
        // Previous d-outer/t-inner order touched a different cache line per t for each d.
        // Accumulation order over `t` is unchanged (ascending), so the float sum is
        // bit-identical to the indexed version.
        let out_head = &mut attn_out[q_off..q_off + head_dim];
        for (&s, v_row) in scores[..seq_len].iter().zip(v_rows.chunks_exact(kv_dim)) {
            simd::simd_fused_scale_acc(out_head, &v_row[kv_off..kv_off + head_dim], s, head_dim);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Block-causal forward (zero-alloc, writes into D2fContext)
// ═══════════════════════════════════════════════════════════════

/// Zero-alloc block-causal forward — writes logits into `ctx.logits_flat`.
///
/// Writes logits into `ctx.logits_flat[p * vocab..(p+1) * vocab]` instead of
/// returning `Vec<Vec<f32>>`. Attention weights are not computed since D2F
/// denoising only needs logits.
///
/// # Multi-layer decode (Issue 869)
///
/// Runs `D2fContext::decode_n_layer` trunk layers over a chained residual
/// stream (`h_0 = rmsnorm(embedding)` — the lane's double-norm quirk; then
/// `h_{l+1} = h_l + MLP_l(rmsnorm_l(h_l + Attn_l(rmsnorm_l(h_l))))`), each
/// layer reading its own KV plane at `l * block_size * kvd`. Depth 1 (the
/// default) executes exactly the pre-869 single-layer op sequence — the
/// Issue-865 gates pin that bit-identity. Per-layer taps: see
/// [`D2fContext::set_probe_tap_layers`].
///
/// Returns `seq_len` (the actual number of positions processed).
pub fn forward_block_causal_with(
    ctx: &mut D2fContext,
    weights: &TransformerWeights,
    tokens: &[usize],
    config: &Config,
    causal_block_size: usize,
) -> usize {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let seq_len = tokens.len().min(config.block_size);
    let scale = 1.0 / (hd as f32).sqrt();
    let vocab = config.vocab_size;
    assert!(
        ctx.decode_n_layer <= weights.layers.len(),
        "decode depth {} exceeds the weights' layer count {} — mismatched weights/context",
        ctx.decode_n_layer,
        weights.layers.len()
    );

    let committed = ctx.committed_len;
    let kv_plane = config.block_size * kvd;

    // No logits pre-clear needed: the final layer's Phase B iterates exactly
    // `committed..seq_len` and ends each iteration with a full-width
    // `logits_flat[p * vocab..(p + 1) * vocab].copy_from_slice(&ctx.logits_buf)`
    // (logits_buf is exactly `vocab` long, so copy_from_slice covers every slot).
    // The loop has no `break`/`continue`, so every slot the old `fill(0.0)` touched
    // is unconditionally overwritten before anyone can read it. Committed positions
    // keep their previous logits, exactly as before.

    for (l, layer) in weights.layers.iter().take(ctx.decode_n_layer).enumerate() {
        let kv_base = l * kv_plane;
        let last = l + 1 == ctx.decode_n_layer;

        // Issue 869 T2: resolve this layer's tap slot once per layer (outside
        // the position loops) — `None` when capture is disarmed or this layer
        // is not in the tap set.
        #[cfg(feature = "probe_guidance")]
        let tap_slot = if ctx.probe_tap_capture {
            ctx.probe_tap_layers.iter().position(|&tl| tl == l)
        } else {
            None
        };

        // Phase A: fill layer-l K/V for UNCOMMITTED positions from the
        // current residual stream (`xr`). Layer 0 additionally builds the
        // stream from the embeddings — the double rmsnorm (xr then x_norm) is
        // the pre-869 layer-0 sequence, preserved op-for-op.
        for (p, &token) in tokens.iter().enumerate().take(seq_len).skip(committed) {
            if l == 0 {
                // Embedding = wte[token] + wpe[position]
                simd::simd_add_into(
                    &mut ctx.x_buf,
                    &weights.wte[token * n..(token + 1) * n],
                    &weights.wpe[p * n..(p + 1) * n],
                );
                // First rmsnorm → residual (xr)
                rmsnorm(&mut ctx.x_buf);
                ctx.xr[p * n..(p + 1) * n].copy_from_slice(&ctx.x_buf);
                // Second rmsnorm → normalized embedding (x_norm)
                rmsnorm(&mut ctx.x_buf);
            } else {
                // Layer l ≥ 1: the stream `h_l` already sits in `xr` (written
                // by layer l-1's Phase B); the layer input norm is a single
                // rmsnorm of it.
                ctx.x_buf.copy_from_slice(&ctx.xr[p * n..(p + 1) * n]);
                rmsnorm(&mut ctx.x_buf);
            }
            ctx.x_norm[p * n..(p + 1) * n].copy_from_slice(&ctx.x_buf);

            // K, V projections (layer l's plane)
            matmul(&mut ctx.k_buf, &layer.attn_wk, &ctx.x_buf, kvd, n);
            matmul(&mut ctx.v_buf, &layer.attn_wv, &ctx.x_buf, kvd, n);
            ctx.k_cache[kv_base + p * kvd..kv_base + (p + 1) * kvd].copy_from_slice(&ctx.k_buf);
            ctx.v_cache[kv_base + p * kvd..kv_base + (p + 1) * kvd].copy_from_slice(&ctx.v_buf);
        }

        // Phase B: Block-causal attention + MLP for UNCOMMITTED positions only
        for p in committed..seq_len {
            // Load normalized embedding
            ctx.x_buf.copy_from_slice(&ctx.x_norm[p * n..(p + 1) * n]);

            // Query projection
            matmul(&mut ctx.q_buf, &layer.attn_wq, &ctx.x_buf, n, n);

            // Block-causal: attend to positions [0..end_of_current_block]
            let block_end = (p / causal_block_size + 1) * causal_block_size;
            let t_n = block_end.min(seq_len);

            // Zero-alloc attention using pre-allocated buffers in D2fContext.
            // The plane tail from `kv_base` is length-safe by construction:
            // the remaining planes are ≥ one full `block_size * kvd` plane,
            // and the helper only reads `[..t_n * kvd]` of what it is given.
            attention_forward_safe_into(
                &ctx.q_buf,
                &ctx.k_cache[kv_base..],
                &ctx.v_cache[kv_base..],
                config.n_head,
                config.n_kv_head,
                hd,
                kvd,
                t_n,
                scale,
                &mut ctx.attn_out_buf,
                &mut ctx.attn_weights_buf,
                &mut ctx.attn_scores_buf,
            );

            // Attention output projection + residual connection
            matmul(&mut ctx.x_proj_buf, &layer.attn_wo, &ctx.attn_out_buf, n, n);
            simd::simd_add_inplace(&mut ctx.x_proj_buf, &ctx.xr[p * n..(p + 1) * n]);

            // Issue 865 T2 / 869 T2: post-attention-residual tap capture, at
            // THIS layer's plane when armed (a pure copy: arming it never
            // moves a logit, so the G1 (λ = 1) decode path stays bit-identical
            // with or without, at any depth).
            #[cfg(feature = "probe_guidance")]
            if let Some(slot) = tap_slot {
                let base = slot * ctx.probe_tap_plane;
                ctx.probe_tap_flat[base + p * n..base + (p + 1) * n]
                    .copy_from_slice(&ctx.x_proj_buf);
            }

            // Save residual before rmsnorm by reusing x_buf (no longer needed this iteration)
            ctx.x_buf.copy_from_slice(&ctx.x_proj_buf);

            // Post-attention rmsnorm
            rmsnorm(&mut ctx.x_proj_buf);

            // MLP: relu hidden → output projection + residual
            matmul_relu(
                &mut ctx.hidden_buf,
                &layer.mlp_w1,
                &ctx.x_proj_buf,
                config.mlp_hidden,
                n,
            );
            matmul(
                &mut ctx.x_mlp_buf,
                &layer.mlp_w2,
                &ctx.hidden_buf,
                n,
                config.mlp_hidden,
            );
            simd::simd_add_inplace(&mut ctx.x_mlp_buf, &ctx.x_buf[..n]);

            if last {
                // Logits (final layer only — the pre-869 position)
                matmul(
                    &mut ctx.logits_buf,
                    &weights.lm_head,
                    &ctx.x_mlp_buf,
                    vocab,
                    n,
                );
                ctx.logits_flat[p * vocab..(p + 1) * vocab].copy_from_slice(&ctx.logits_buf);
            } else {
                // Feed the stream: h_{l+1} becomes the next layer's residual
                // base + norm input. Read-then-write within the same
                // iteration — Phase A_{l+1} runs only after this whole loop.
                ctx.xr[p * n..(p + 1) * n].copy_from_slice(&ctx.x_mlp_buf);
            }
        }
    }

    seq_len
}

// ═══════════════════════════════════════════════════════════════
// Accuracy metric
// ═══════════════════════════════════════════════════════════════

/// Measure denoising accuracy: fraction of correctly recovered tokens.
pub fn denoising_accuracy(predicted: &[usize], target: &[usize]) -> f32 {
    let len = predicted.len().min(target.len());
    if len == 0 {
        return 0.0;
    }
    let correct = (0..len).filter(|&i| predicted[i] == target[i]).count();
    correct as f32 / len as f32
}

// ═══════════════════════════════════════════════════════════
// Probe guidance (Issue 865 T1) — weak-side probe seam + affine combine
// ═══════════════════════════════════════════════════════════
//
// The autoguidance form (Research 68 §7.2 / arXiv:2609.19356): extrapolate
// along the strong−weak prediction difference,
//
//   logits' = logits + (λ−1)·(logits − probe_logits)
//           = λ·logits + (1−λ)·probe_logits
//
// The strong side is the trunk's own logits; the weak side is a cheap probe
// (T2 trains a `LatentDynamicsMLP`-class artifact on a frozen trunk — the
// class already ships in `katgpt-speculative/src/belief_drafter.rs`, Plan
// 217). This module owns the MECHANISM only: the seam a probe plugs into and
// the zero-alloc combine. λ = 1.0 (the default) skips all work, so a context
// without guidance is bit-identical to pre-Issue-865 decode (G1).

/// Everything a weak-side probe may read for one denoising step (Issue 865
/// T1). Field slices, not a whole-context borrow: the probe contract names
/// exactly what it may read, and disjoint field borrows are what let the
/// caller hand these out while writing the probe's own scratch buffer.
/// T2 extended this with the tap-point field (`tap`) — the early-layer hidden
/// state the trained connector reads — without breaking probe call sites.
/// Issue 869 T2 layered the tap: `tap` now holds one plane per configured
/// tap layer (`tap_layers`, slot order), and probes locate their plane by
/// layer index.
#[cfg(feature = "probe_guidance")]
pub struct ProbeCtx<'a> {
    /// Input residual embeddings (the layer-0 residual stream the kernel
    /// writes in Phase A), `[seq_len * n_embd]` (per position). NOT the tap
    /// point — at masked positions this carries no context (see `tap`).
    pub xr: &'a [f32],
    /// Normalized embeddings, `[seq_len * n_embd]` (per position).
    pub x_norm: &'a [f32],
    /// Hidden states at the tapped points, LAYERED (Issue 869 T2): one plane
    /// per entry of `tap_layers` (slot order), each plane holding the
    /// post-ATTENTION residual of that layer — attention output + input
    /// residual, pre-MLP refinement — for every position (`[max_seq * n_embd]`
    /// per plane, row `p` at `[plane + p * n_embd..]`). At masked positions
    /// the pre-layer input residual (`xr`) is just the mask-token embedding
    /// plus position — no context — so probes read HERE. Populated only when
    /// the decode core arms capture (guidance active); an artifact declaring
    /// a layer outside `tap_layers` is rejected loudly at guidance install
    /// (`D2fContext::set_guidance`), never silently misread.
    pub tap: &'a [f32],
    /// The layer indices captured into `tap`, in slot order (Issue 869 T2):
    /// `tap_layers[s]` is the source layer of the plane at
    /// `tap[s * tap_plane..]`. Probes that care which layer they read (e.g.
    /// `MlpWeakProbe`, whose artifact pins `tap_layer`) find their slot by
    /// index here; layer-agnostic probes read slot 0.
    pub tap_layers: &'a [usize],
    /// Elements per tap plane (`max_seq * n_embd`) — the slot stride of `tap`
    /// (Issue 869 T2).
    pub tap_plane: usize,
    /// Current token state (prompt + block; uncommitted positions hold the
    /// mask token).
    pub tokens: &'a [usize],
    /// Valid KV prefix length — positions `[0..committed_len)` are committed.
    pub committed_len: usize,
    /// First denoised (block) position in `tokens`.
    pub block_start: usize,
    /// Total sequence length this step (prompt + block).
    pub seq_len: usize,
    /// Vocabulary size (the probe's `out` holds `count * vocab` logits).
    pub vocab: usize,
    /// Embedding width, `[position * n_embd..]` indexing into `xr`/`x_norm`.
    pub n_embd: usize,
    /// Denoising step index (0-based) this probe call serves.
    pub step: usize,
}

/// Weak-side probe: produces the cheap logits the guidance extrapolates
/// AGAINST (Issue 865 T1). Implementations write exactly
/// `(seq_len − block_start) * vocab` logits into `out`, ordered by position
/// then vocab — the same layout as `D2fContext::logits_flat`'s block range.
/// A trained artifact loads via the freeze/thaw wire (BLAKE3-checked) and
/// stays consistent with the modelless consumption rule. `Send + Sync` is
/// required so `D2fContext` keeps its auto-`Send + Sync`: the tri_mode
/// verifier structs (`SpeculativeVerifier: Send + Sync`) embed the context,
/// and the probe payload — a frozen MLP or plain fixture data — is plain
/// data anyway.
#[cfg(feature = "probe_guidance")]
pub trait WeakLogitProbe: Send + Sync {
    fn probe(&mut self, input: ProbeCtx<'_>, out: &mut [f32]);

    /// The tap layer this probe's artifact was trained against (Issue 869
    /// T2): `None` (the default) = layer-agnostic — reads slot 0, whatever
    /// layer that is; `Some(l)` = the probe must find layer `l` in
    /// `ProbeCtx::tap_layers`, and `D2fContext::set_guidance` validates that
    /// at INSTALL time (a probe whose layer is not being captured is a loud
    /// panic with the remedy, never a silent misread of the wrong plane).
    fn tap_layer(&self) -> Option<usize> {
        None
    }
}

#[cfg(feature = "probe_guidance")]
impl D2fContext {
    /// Install probe guidance with strength `lambda` (Issue 865 T1).
    ///
    /// `lambda = 1.0` leaves decode bit-identical to unguided (the combine is
    /// skipped, the probe is never invoked); values above 1.0 extrapolate
    /// away from the weak side, below 1.0 blend toward it. Calling again
    /// replaces the previous probe and strength.
    ///
    /// Issue 869 T2: a probe declaring a tap layer (see
    /// [`WeakLogitProbe::tap_layer`]) is validated against the context's tap
    /// set HERE — the one place both sides are known. A mismatch panics with
    /// the exact remedy (`set_probe_tap_layers` + possibly
    /// `set_decode_layers`), replacing the old constructor-time `tap_layer
    /// != 0` rejection that froze every artifact to the single-layer era.
    pub fn set_guidance(&mut self, lambda: f32, probe: Box<dyn WeakLogitProbe>) {
        if let Some(tl) = probe.tap_layer() {
            assert!(
                self.probe_tap_layers.contains(&tl),
                "probe reads tap layer {tl} but the context captures {:?} — \
                 call set_probe_tap_layers(&[{tl}]) (and set_decode_layers({}) if the \
                 decode depth does not reach it) before set_guidance",
                self.probe_tap_layers,
                tl + 1
            );
        }
        self.guidance_lambda = lambda;
        self.weak_probe = Some(probe);
    }

    /// Remove any installed probe and restore the no-op default (λ = 1.0).
    pub fn clear_guidance(&mut self) {
        self.guidance_lambda = 1.0;
        self.weak_probe = None;
    }
}

/// Affine probe-guidance combine (Issue 865 T1):
/// `logits' = λ·logits + (1−λ)·probe_logits` over the denoised block's
/// positions.
///
/// Applied AFTER the (optional) multistep blend and BEFORE per-position
/// sampling, so the sampler reads guided logits exactly as it reads raw ones
/// and the two logit transforms never fight over ordering. No-ops when
/// λ == 1.0 (G1 bit-identity — not even the probe is invoked) or when no
/// probe is installed. Zero-alloc: reads/writes the context's flat buffers
/// only; the combine loop is chunked 8-wide and branch-free so LLVM
/// auto-vectorizes it.
#[cfg(feature = "probe_guidance")]
pub(crate) fn apply_probe_guidance(
    dctx: &mut D2fContext,
    tokens: &[usize],
    block_start: usize,
    seq_len: usize,
    vocab: usize,
    n_embd: usize,
    step: usize,
) {
    if dctx.guidance_lambda == 1.0 {
        return;
    }
    // take()/put-back: the probe must not hold a borrow of the context while
    // the probe scratch buffer is handed out mutably.
    let Some(mut probe) = dctx.weak_probe.take() else {
        return;
    };
    let range = block_start * vocab..seq_len * vocab;
    {
        let input = ProbeCtx {
            xr: &dctx.xr,
            x_norm: &dctx.x_norm,
            // Issue 869 T2: the layered tap — one plane per configured layer
            // (`tap_layers`, slot order), each the post-attention residual of
            // its layer, captured by the forward when the decode core armed
            // `probe_tap_capture`. Probes locate their plane by layer index;
            // install-time validation (`set_guidance`) guarantees a
            // declared layer is present.
            tap: &dctx.probe_tap_flat,
            tap_layers: &dctx.probe_tap_layers,
            tap_plane: dctx.probe_tap_plane,
            tokens,
            committed_len: dctx.committed_len,
            block_start,
            seq_len,
            vocab,
            n_embd,
            step,
        };
        probe.probe(input, &mut dctx.probe_logits_flat[range.clone()]);
    }
    dctx.weak_probe = Some(probe);

    // Plan 617 T1.2: the arithmetic lives in katgpt-core's contrast_combine
    // (ONE home for the strong−weak affine family — the LoopCD lane consumes
    // the same kernel at λ = 1+ω). Bit-identical to the pre-extraction
    // in-place loop: same op order, (1−λ) hoisted once, 8-wide chunk + tail.
    let lambda = dctx.guidance_lambda;
    katgpt_core::contrast_combine::affine_combine(
        &mut dctx.logits_flat[range.clone()],
        &dctx.probe_logits_flat[range],
        lambda,
    );
}

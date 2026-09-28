//! Per-layer and global transformer weight structures.
//!
//! Pure data types — no forward logic. The forward kernels live in the
//! `katgpt-rs` root crate because they compose cognitive primitives
//! (`crate::hla`, `crate::sleep`, `crate::tf_loop`, etc.) that do not exist
//! in this substrate crate.

use half::f16;
use katgpt_core::types::{self, Config, Rng};

// ── f16 Weight Storage (Issue 200) ─────────────────────────────
//
// Parallel weight structs storing projection weights as `half::f16`.
// Halves memory bandwidth for weight reads vs f32 storage, attacking the
// GEMV bandwidth ceiling identified by the forward-pass profiling.
//
// Pattern mirrors riir-engine's `GemmaTransformerWeightsF16` (Plan 095):
// separate struct + separate forward function, NOT an enum-wrapped dispatch.
// This is additive (zero breakage to existing f32 path).

/// Per-layer transformer weights with f16 storage for projection matrices.
///
/// Embedding / norm / gate vectors stay f32 (tiny, non-matmul, negligible
/// bandwidth). Only the projection weights that dominate the GEMV bandwidth
/// budget are stored as f16.
#[derive(Clone)]
pub struct LayerWeightsF16 {
    pub attn_wq: Vec<f16>, // [n_embd, n_embd]
    pub attn_wk: Vec<f16>, // [kv_dim, n_embd]
    pub attn_wv: Vec<f16>, // [kv_dim, n_embd]
    pub attn_wo: Vec<f16>, // [n_embd, n_embd]
    pub mlp_w1: Vec<f16>,  // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f16>,  // [n_embd, mlp_hidden]
}

/// Transformer weights with f16 projection storage (Issue 200).
///
/// Embeddings (`wte`, `wpe`) and `lm_head` stay f32 — they are accessed via
/// random-row lookup (embedding) or benefit from f32 precision at the vocab
/// projection (lm_head produces logits where small numerical differences
/// could affect token sampling). The bandwidth-dominant per-layer projection
/// matrices are f16. This captures ~95% of the bandwidth win (the per-layer
/// projections) while keeping the logits numerically stable.
///
/// Construct via [`TransformerWeights::to_f16`] — a one-time conversion at
/// model load time.
#[derive(Clone)]
pub struct TransformerWeightsF16 {
    pub wte: Vec<f32>,                // [vocab_size, n_embd] — f32 (embedding lookup)
    pub wpe: Vec<f32>,                // [block_size, n_embd] — f32 (embedding lookup)
    pub lm_head: Vec<f32>,            // [vocab_size, n_embd] — f32 (logit precision)
    pub layers: Vec<LayerWeightsF16>, // [n_layer]
}

/// Per-layer transformer weights.
/// Each layer has its own attention and MLP parameters.
///
/// `Clone` derived (Issue 374 / Plan 301): consumers that fork frozen base weights
/// (e.g. `riir-train-gpu`'s CPU LoRA fallback trainer) need `.clone()` without
/// rebuilding from a seed. All fields are `Vec<f32>` / `Option<Vec<f32>>` — clone
/// is a shallow vector copy, no deep aliasing.
#[derive(Clone)]
pub struct LayerWeights {
    pub attn_wq: Vec<f32>, // [n_embd, n_embd]
    pub attn_wk: Vec<f32>, // [kv_dim, n_embd] where kv_dim = n_kv_head * head_dim
    pub attn_wv: Vec<f32>, // [kv_dim, n_embd]
    pub attn_wo: Vec<f32>, // [n_embd, n_embd]
    pub mlp_w1: Vec<f32>,  // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,  // [n_embd, mlp_hidden]
    // Gated MLP (Issue 377): the "up" projection for SwiGLU.
    // When `gated_mlp` is enabled, mlp_w1 becomes W_gate, mlp_w_up is W_up,
    // and mlp_w2 becomes W_down. Forward: SiLU(W_gate·h) ⊙ W_up·h → W_down·.
    #[cfg(feature = "gated_mlp")]
    pub mlp_w_up: Vec<f32>, // [mlp_hidden, n_embd]
    // Kog CPU fusion (Plan 160): RMSNorm gamma vectors — only present when the
    // consumer enables `kog_cpu_fusion`. Consumers that don't use Kog fusion
    // (e.g. riir-engine) get the compact 6-field struct and avoid ~2×n
    // floats/layer of dead weight.
    #[cfg(feature = "kog_cpu_fusion")]
    pub attn_norm_gamma: Vec<f32>, // [n_embd] pre-attention RMSNorm gamma (identity=1.0)
    #[cfg(feature = "kog_cpu_fusion")]
    pub mlp_norm_gamma: Vec<f32>, // [n_embd] pre-MLP RMSNorm gamma (identity=1.0)
    // Kog CPU fusion (Plan 160): fused QKV weight storage
    #[cfg(feature = "kog_cpu_fusion")]
    pub attn_qkv_fused: Option<Vec<f32>>, // [(n_embd + 2*kv_dim), n_embd] interleaved
    // Wall Attention gate projection weights (Plan 173)
    #[cfg(feature = "wall_attention")]
    pub attn_wg: Vec<f32>, // [kv_dim] gate projection per KV head dimension
}

impl LayerWeights {
    /// Base-6 construction with every optional (cfg-gated) field at its inert
    /// value. Lives HERE because the optional-field set is this crate's
    /// feature set as unified across the graph — a consumer-crate struct
    /// literal cannot know it and breaks under feature unification (Issue
    /// 558's `into_generic` conversion was the measured casualty).
    ///
    /// The optional fields are left EMPTY (not ones-init): callers that enable
    /// their features construct them through this crate's own paths.
    pub fn from_parts(
        attn_wq: Vec<f32>, // [outs, ins]
        attn_wk: Vec<f32>, // [kv_dim, ins]
        attn_wv: Vec<f32>, // [kv_dim, ins]
        attn_wo: Vec<f32>, // [ins, outs]
        mlp_w1: Vec<f32>,  // [mlp_hidden, ins]
        mlp_w2: Vec<f32>,  // [ins, mlp_hidden]
    ) -> Self {
        Self {
            attn_wq,
            attn_wk,
            attn_wv,
            attn_wo,
            mlp_w1,
            mlp_w2,
            #[cfg(feature = "gated_mlp")]
            mlp_w_up: Vec::new(),
            #[cfg(feature = "kog_cpu_fusion")]
            attn_norm_gamma: Vec::new(),
            #[cfg(feature = "kog_cpu_fusion")]
            mlp_norm_gamma: Vec::new(),
            #[cfg(feature = "kog_cpu_fusion")]
            attn_qkv_fused: None,
            #[cfg(feature = "wall_attention")]
            attn_wg: Vec::new(),
        }
    }
}

/// All transformer weights: embeddings, per-layer weights, and LM head.
/// Layout preserves init order for backward compat: wte, wpe, layers…, lm_head.
///
/// # Future: f16 Storage
///
/// For memory-constrained deployments, weights can be stored as `f16` (half-precision)
/// and quantized on-the-fly during matmul. This would halve memory usage with minimal
/// accuracy loss for inference-only workloads. The migration path:
///
/// 1. Add a `StorageFormat` enum: `F32`, `F16`, `Q4_0`, `Q8_0`
/// 2. Replace `Vec<f32>` with a `WeightTensor` enum that stores the chosen format
/// 3. Add `dequantize_row()` that converts to `f32` on-the-fly during matmul
/// 4. The `forward()` kernel remains unchanged — it operates on `f32` buffers
///    populated by dequantization
///
/// Key insight: only storage changes; compute stays in `f32`. This avoids the need
/// for f16 arithmetic hardware support and keeps the attention kernel simple.
/// Estimated memory savings: ~50% for f16, ~75% for 4-bit quantized.
// `Clone` derived (Issue 374 / Plan 301): the CPU LoRA fallback trainer needs to
// fork frozen base weights. All fields are `Vec` / `Option<Vec>` — clone is a
// shallow vector copy, no deep aliasing.
#[derive(Clone)]
pub struct TransformerWeights {
    pub wte: Vec<f32>,             // [vocab_size, n_embd]
    pub wpe: Vec<f32>,             // [block_size, n_embd]
    pub lm_head: Vec<f32>,         // [vocab_size, n_embd]
    pub layers: Vec<LayerWeights>, // [n_layer]
    // MTP Drafter weights (Plan 055: Gemma 4 MTP)
    /// Target→Draft activation projection: [draft_n_embd, target_n_embd + embed_dim]
    /// Only loaded when Config mtp_activation_threshold is met and weights file exists.
    /// Falls back to truncate/pad when absent.
    pub mtp_activation_proj: Option<Vec<f32>>,
    /// Cluster classifier: [num_clusters, n_embd]
    /// Only loaded when vocab_size > mtp_cluster_vocab_threshold.
    pub mtp_cluster_classifier: Option<Vec<f32>>,
    /// Cluster membership table: `[num_clusters]` → `Vec<usize>` (token indices)
    pub mtp_cluster_map: Option<Vec<Vec<usize>>>,
    // Delta routing weights (Plan 097: Delta Attention Residuals)
    #[cfg(feature = "delta_routing")]
    pub delta_routing_query: Vec<Vec<f32>>, // [n_layer][n_embd] per-layer query vectors
    #[cfg(feature = "delta_routing")]
    pub delta_routing_norm: Vec<Vec<f32>>, // [n_layer][n_embd] per-layer RMSNorm weights (gamma)
    /// Final RMSNorm gamma between the residual stream and `lm_head`:
    /// `[n_embd]` f32 (riir-train Issue 538 / Plan 398). The plain forward
    /// paths of this crate IGNORE it (additive, bit-identical); the DFlash/
    /// DFlare training lanes apply it — logit scale no longer ∝ residual
    /// norm (the measured confidence-ceiling mechanism). Ones-init = identity
    /// gain (the `delta_routing_norm` precedent).
    pub final_norm_gamma: Vec<f32>,
}

impl TransformerWeights {
    /// Non-gated construction with every optional (cfg-gated) field at its
    /// inert value, from the required base parts (Issue 558's `into_generic`
    /// conversion). Lives HERE for the same reason as
    /// [`LayerWeights::from_parts`]: the optional-field set is this crate's
    /// feature set as unified across the graph — a consumer-crate literal
    /// cannot know it and breaks under feature unification (measured:
    /// `delta_routing` on at infer-core's default via another consumer's
    /// forwarding while the consumer's own arm compiled the field out).
    ///
    /// `cfg!` (runtime-const) rather than `#[cfg]` keeps this ONE code path —
    /// the inert defaults are empty/zeroed regardless of which half compiles.
    #[allow(clippy::field_reassign_with_default)]
    pub fn from_parts(
        wte: Vec<f32>,             // [vocab_size, n_embd]
        wpe: Vec<f32>,             // [block_size, n_embd] (or a placeholder)
        lm_head: Vec<f32>,         // [vocab_size, n_embd]
        layers: Vec<LayerWeights>, // [n_layer]
        n_embd: usize,
        n_layer: usize,
    ) -> Self {
        Self {
            wte,
            wpe,
            lm_head,
            layers,
            mtp_activation_proj: None,
            mtp_cluster_classifier: None,
            mtp_cluster_map: None,
            // This crate's own cfg arms — they match the struct layout by
            // construction (the unification-proof form).
            #[cfg(feature = "delta_routing")]
            delta_routing_query: vec![vec![0.0; n_embd]; n_layer],
            #[cfg(feature = "delta_routing")]
            delta_routing_norm: vec![vec![1.0; n_embd]; n_layer],
            // Identity gain — Issue 538's ones-init convention.
            final_norm_gamma: vec![1.0; n_embd],
        }
    }

    pub fn new(config: &Config, rng: &mut Rng) -> Self {
        let n = config.n_embd;
        let kvd = types::kv_dim(config);
        let embd_scale = (2.0 / n as f32).sqrt();
        let layer_scale = (2.0 / (n as f32 * config.n_layer as f32)).sqrt();

        // Embeddings first (same order as original single-layer code)
        // Pre-allocate to avoid repeated re-allocation during collect().
        let wte_len = config.vocab_size * n;
        let mut wte = Vec::with_capacity(wte_len);
        wte.extend((0..wte_len).map(|_| rng.normal() * embd_scale));

        let wpe_len = config.block_size * n;
        let mut wpe = Vec::with_capacity(wpe_len);
        wpe.extend((0..wpe_len).map(|_| rng.normal() * embd_scale));

        // Per-layer weights: same field order as original per n_layer iterations
        // Pre-allocate each weight vector to avoid repeated reallocation.
        let mut layers = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            layers.push(LayerWeights {
                attn_wq: {
                    let len = n * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                attn_wk: {
                    let len = kvd * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                attn_wv: {
                    let len = kvd * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                attn_wo: {
                    let len = n * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                mlp_w1: {
                    let len = config.mlp_hidden * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                mlp_w2: {
                    let len = n * config.mlp_hidden;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                #[cfg(feature = "gated_mlp")]
                mlp_w_up: {
                    let len = config.mlp_hidden * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                #[cfg(feature = "kog_cpu_fusion")]
                attn_norm_gamma: vec![1.0f32; n],
                #[cfg(feature = "kog_cpu_fusion")]
                mlp_norm_gamma: vec![1.0f32; n],
                #[cfg(feature = "kog_cpu_fusion")]
                attn_qkv_fused: None,
                #[cfg(feature = "wall_attention")]
                attn_wg: vec![0.0; kvd], // Initialized to zeros; gate not active unless wall_config is Some
            });
        }

        // LM head last
        let lm_len = config.vocab_size * n;
        let mut lm_head = Vec::with_capacity(lm_len);
        lm_head.extend((0..lm_len).map(|_| rng.normal() * embd_scale));

        Self {
            wte,
            wpe,
            lm_head,
            layers,
            mtp_activation_proj: None,
            mtp_cluster_classifier: None,
            mtp_cluster_map: None,
            #[cfg(feature = "delta_routing")]
            delta_routing_query: {
                let mut v = Vec::with_capacity(config.n_layer);
                for _ in 0..config.n_layer {
                    v.push(vec![0.0; config.n_embd]); // Zero-init: safe additive start
                }
                v
            },
            #[cfg(feature = "delta_routing")]
            delta_routing_norm: {
                let mut v = Vec::with_capacity(config.n_layer);
                for _ in 0..config.n_layer {
                    v.push(vec![1.0f32; config.n_embd]); // Ones: identity RMSNorm
                }
                v
            },
            // Issue 538 (riir-train): ones-init identity gain — the DFlash/
            // DFlare lanes consume it; every other path ignores it.
            final_norm_gamma: vec![1.0f32; config.n_embd],
        }
    }

    /// Convert f32 projection weights to f16 storage (Issue 200).
    ///
    /// One-time conversion at model load time. Embeddings (`wte`, `wpe`) and
    /// `lm_head` stay f32 (embedding lookup + logit precision). Only per-layer
    /// projection matrices are converted to f16 — these dominate the GEMV
    /// bandwidth budget (~95% of per-forward weight reads).
    ///
    /// The returned `TransformerWeightsF16` is consumed by `forward_base_f16`,
    /// which dispatches matmuls to `matmul_f16` (f16 weights × f32 activations,
    /// dequant-on-load inside the dot kernel).
    pub fn to_f16(&self) -> TransformerWeightsF16 {
        let layers = self
            .layers
            .iter()
            .map(|lw| LayerWeightsF16 {
                attn_wq: lw.attn_wq.iter().map(|&v| f16::from_f32(v)).collect(),
                attn_wk: lw.attn_wk.iter().map(|&v| f16::from_f32(v)).collect(),
                attn_wv: lw.attn_wv.iter().map(|&v| f16::from_f32(v)).collect(),
                attn_wo: lw.attn_wo.iter().map(|&v| f16::from_f32(v)).collect(),
                mlp_w1: lw.mlp_w1.iter().map(|&v| f16::from_f32(v)).collect(),
                mlp_w2: lw.mlp_w2.iter().map(|&v| f16::from_f32(v)).collect(),
            })
            .collect();
        TransformerWeightsF16 {
            wte: self.wte.clone(),
            wpe: self.wpe.clone(),
            lm_head: self.lm_head.clone(),
            layers,
        }
    }

    /// Initialize Wall Attention gate weights with random values (Plan 173).
    /// Call when `wall_config` is `Some` — populates `attn_wg` with proper scaling.
    /// This is separate from `new()` to avoid consuming RNG when Wall is disabled.
    #[cfg(feature = "wall_attention")]
    pub fn init_wall_gates(&mut self, config: &Config, rng: &mut Rng) {
        let kvd = types::kv_dim(config);
        let layer_scale = (2.0 / (config.n_embd as f32 * config.n_layer as f32)).sqrt();
        for layer in &mut self.layers {
            let mut v = Vec::with_capacity(kvd);
            v.extend((0..kvd).map(|_| rng.normal() * layer_scale));
            layer.attn_wg = v;
        }
    }

    /// Fold RMSNorm gamma into projection weights (Plan 160: Kog CPU fusion).
    ///
    /// For each projection preceded by RMSNorm with gamma:
    ///   `weight[row * n_embd + col] *= gamma[col]`
    ///
    /// After folding, gamma is set to 1.0 (identity), so runtime rmsnorm_with_gamma
    /// becomes a no-op. This eliminates per-token gamma memory reads.
    ///
    /// **Attention gamma**: NOT folded because the residual connection (`xr`) captures
    /// the post-norm value (`x * inv_rms * gamma`). Folding would change the residual.
    /// The attention gamma remains at runtime for `rmsnorm_with_gamma`.
    ///
    /// **MLP gamma**: Folded into `mlp_w1` because the residual (`xr2`) is saved
    /// BEFORE the norm, so gamma only affects the projection path.
    #[cfg(feature = "kog_cpu_fusion")]
    pub fn fold_gamma(&mut self, config: &Config) {
        let n = config.n_embd;

        for layer in &mut self.layers {
            // Fold mlp_norm_gamma into mlp_w1
            // (Safe: xr2 is saved before rmsnorm, so residual is pre-norm)
            let mlp_gamma = &layer.mlp_norm_gamma;
            for row in 0..config.mlp_hidden {
                for (col, g) in mlp_gamma.iter().enumerate() {
                    layer.mlp_w1[row * n + col] *= g;
                }
            }
            // Set mlp_norm_gamma to identity
            layer.mlp_norm_gamma.fill(1.0f32);

            // Note: attn_norm_gamma is NOT folded because xr (attention residual)
            // captures the post-norm value. It remains for runtime rmsnorm_with_gamma.
        }
    }

    /// Repack Q/K/V weights into a single contiguous buffer (Plan 160: Kog CPU fusion).
    ///
    /// Layout: [Q rows | K rows | V rows] × `[n_embd]`, where:
    ///   Q rows = `[n_embd]`, K rows = `[kv_dim]`, V rows = `[kv_dim]`
    ///
    /// The fused weight is stored in `attn_qkv_fused` (Some when populated).
    /// Original weights are preserved — fused is an additional allocation.
    /// Cache locality win: single contiguous memory region instead of 3 scattered buffers.
    #[cfg(feature = "kog_cpu_fusion")]
    pub fn interleave_qkv(&mut self, config: &Config) {
        let n = config.n_embd;
        let kvd = types::kv_dim(config);
        let q_rows = n;
        let k_rows = kvd;
        let v_rows = kvd;
        let total_rows = q_rows + k_rows + v_rows;

        for layer in &mut self.layers {
            let mut fused = Vec::with_capacity(total_rows * n);
            fused.extend_from_slice(&layer.attn_wq);
            fused.extend_from_slice(&layer.attn_wk);
            fused.extend_from_slice(&layer.attn_wv);
            layer.attn_qkv_fused = Some(fused);
        }
    }
}

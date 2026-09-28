//! Small configuration enums and feature-config structs.

// Shared configuration, RNG, and math utilities.
// Superset of types from both katgpt-rs and riir-engine projects.

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Adaptive depth tier mapping to layer count (Plan 284).
/// Reuses ThermalPath naming convention from FlashAR Consensus (Plan 166).
///
/// | Tier     | Layers     | When                              |
/// |----------|------------|-----------------------------------|
/// | Plasma   | 1          | High entropy, easy positions      |
/// | Hot      | 2          | Medium entropy, standard tactics  |
/// | Warm     | all        | Low entropy, complex positions    |
/// | Cold     | all+verify | Critical, full verification       |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DepthTier {
    /// Easy positions: empty board, forced moves. 1 layer.
    Plasma = 0,
    /// Moderate: standard tactics. 2 layers.
    Hot = 1,
    /// Complex: all layers + spot-check verification.
    Warm = 2,
    /// Critical: all layers + full verification.
    Cold = 3,
}

impl DepthTier {
    /// Returns the maximum number of transformer layers to execute for this tier.
    pub fn max_layers(&self, total_layers: usize) -> usize {
        match self {
            Self::Plasma => 1.min(total_layers),
            Self::Hot => 2.min(total_layers),
            Self::Warm => total_layers,
            Self::Cold => total_layers,
        }
    }
}

/// Attention mode for HLA (Higher-order Linear Attention).
///
/// - `Standard`: SDPA with KV cache (default, backward-compatible).
/// - `Hla`: Symmetric second-order linear attention — O(1) per-token memory.
/// - `Ahla`: Asymmetric second-order linear attention — lower state cost than symmetric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum HlaMode {
    #[default]
    Standard,
    /// Symmetric second-order: SK, CQV, mQ accumulators.
    Hla,
    /// Asymmetric second-order: PKV, mK accumulators.
    Ahla,
}

/// Attention mode for forward passes.
///
/// - `Causal`: Standard autoregressive — only attend to positions ≤ current (default).
/// - `Bidirectional`: Attend to ALL positions — used for dLLM masked prediction (Plan 066).
/// - `BlockCausal`: Bidirectional within current block, causal across blocks — D2F student.
/// - `SpKv`: SP-KV self-pruned key-value attention (Plan 070).
/// - `SpKvQuant`: SP-KV + Quantized KV fusion (Plan 070 Phase 3, Task T12).
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionMode {
    #[default]
    Causal,
    /// Full bidirectional: all positions see all positions (teacher mode).
    Bidirectional,
    /// Block-causal: bidirectional within block, causal across blocks (student mode).
    BlockCausal,
    /// SP-KV self-pruned key-value attention (Plan 070).
    /// Learns which KV pairs to retain via utility prediction.
    /// Gate bias = log(u) during training, 0|-inf during inference.
    SpKv,
    /// SP-KV + Quantized KV fusion (Plan 070 Phase 3, Task T12).
    /// Selective write (SP-KV utility gating) + lossy quantize (any QuantizedKVCache backend).
    /// Two-stage compression: only useful KV pairs kept, those compressed to 2-4 bits/coord.
    SpKvQuant,
    /// DashAttention: adaptive sparse hierarchical attention via α-entmax routing (Plan 106).
    /// Replaces fixed-budget top-k block selection with adaptive support selection.
    /// Learned chunk summaries via head_cls vectors.
    DashAttn,
}

/// Model architecture selector for forward pass dispatch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelArchitecture {
    #[default]
    Generic,
    Gemma2,
    Llama,
    /// Hybrid DeltaNet/Attention model (e.g., Qwen 3.5, Kimi Linear).
    /// Uses per-layer config to determine DeltaNet vs standard attention.
    /// Plan 182: Luce Megakernel Distill — DeltaNet GPU Inference.
    #[cfg(feature = "deltanet_inference")]
    QwenDeltaNet,
    /// Gemma 4 unified text model (Issue 577 — baseline loader for Plan 318).
    /// Alternating sliding-window attention (5 layers, window=1024) +
    /// full attention (1 layer) repeating; per-layer head_dim and KV-head
    /// count vary. Native 256K context. GGUF-loaded; opt-in `gemma4_inference`.
    #[cfg(feature = "gemma4_inference")]
    Gemma4,
    /// Ternary-weight transformer (Plan 333 Phase 2).
    /// Ternary {-1,0,+1} weights with per-128 f16 group scale (Q2_0_g128).
    /// The weight substrate ships behind `ternary_group_scale` (Issue 578:
    /// `TernaryGroupWeights` + `simd_ternary_group_matvec`); this variant
    /// gates only the forward dispatch in riir-engine. GGUF-loaded via the
    /// `Q2_0` tensor type (Plan 333 T2.1).
    ///
    /// **Not named `BitNet`** (renamed 2026-08-10): the format is `Q2_0_g128`,
    /// not BitNet's `i2_s`, and Ternary-Bonsai-27B is `general.architecture =
    /// qwen35`, not the BitNet b1.58 family. Every piece of the substrate is
    /// named `ternary`; this variant now matches.
    ///
    /// **Known modelling tension** (riir-ai Issue 594): "ternary" is a *weight
    /// container*, not an architecture, so it is orthogonal to the other
    /// variants rather than parallel to them. It earns a variant here only
    /// because this repo's dispatch pattern is one variant per weights struct.
    /// Running a ternary qwen35 hybrid will need a container-vs-architecture
    /// split, not a second variant.
    #[cfg(feature = "ternary_inference")]
    Ternary,
}

/// Per-layer attention type for Gemma 4 (Issue 577).
///
/// Gemma 4 alternates between sliding-window attention layers (5 of every 6)
/// and full-attention layers (1 of every 6, at index `% 6 == 5`). The two layer
/// kinds differ in head_dim, n_kv_head, RoPE base, and whether sliding-window
/// masking applies — see `Config::gemma4_12b` for the concrete shape.
#[cfg(feature = "gemma4_inference")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Gemma4LayerType {
    /// Sliding-window attention: head_dim=256, n_kv_head=8 (GQA 2:1),
    /// rope_theta=10000, sliding_window=1024, full RoPE rotation.
    #[default]
    Sliding,
    /// Full attention: head_dim=512, n_kv_head=1 (MQA),
    /// rope_theta=1_000_000, partial_rotary_factor=0.25, no sliding window.
    Full,
}

/// Attention projection configuration.
/// Controls whether K and V projections share weights (Q-K=V tying).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum AttentionProjection {
    /// Standard Q, K, V (3 projections, full KV cache)
    #[default]
    Full,
    /// Q-K=V: K and V share projection (2 projections, K-only cache).
    /// 50% KV cache reduction, ~3% perplexity cost.
    /// Post-hoc weight merging: W_kv = (W_k + W_v) / 2.
    SharedKV,
}

/// KV cache layout (derived from AttentionProjection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CacheLayout {
    /// Store both K and V (standard)
    KV,
    /// Store K only, V = K at read (SharedKV)
    K,
}

/// Weight storage dtype (affects loading and dequantization).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum WeightDtype {
    #[default]
    F32,
    F16,
    BF16,
}

// ---------------------------------------------------------------------------
// Delta Routing (Plan 097, Research 061)
// ---------------------------------------------------------------------------

/// Delta routing mode — cross-layer information flow via delta vectors.
/// Research 061: Delta Attention Residuals (Plan 097).
///
/// Kept compiled even when `delta_routing` is off so config round-trips
/// serialize identically across feature sets. Reachable via `Config` defaults
/// once the routing backend lands.
#[allow(dead_code)]
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeltaRoutingMode {
    /// No delta routing (default).
    #[default]
    Off,
    /// Delta Block: accumulate deltas within blocks of `block_size` layers.
    /// B+1 sources per routing decision. ~20% throughput overhead.
    DeltaBlock,
    /// Delta Attention Residuals: per-sublayer delta routing.
    /// 2L sources. 69% throughput reduction at L=36. Use only for research.
    DeltaAttnRes,
}

/// Configuration for delta routing (Plan 097, Research 061).
///
/// Fields ordered by descending alignment to minimize padding:
/// usize (8B) → repr(u8) enum (1B) — 16 bytes total, no wasted padding.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct DeltaRoutingConfig {
    /// Block size for DeltaBlock mode (number of layers per block).
    /// Default: 4. Paper recommends B=4.
    pub block_size: usize,
    /// Routing mode.
    pub mode: DeltaRoutingMode,
}

impl Default for DeltaRoutingConfig {
    fn default() -> Self {
        Self {
            block_size: 4,
            mode: DeltaRoutingMode::Off,
        }
    }
}

// ---------------------------------------------------------------------------
// DeltaNet Inference (Plan 182: Luce Megakernel Distill)
// ---------------------------------------------------------------------------

/// Per-layer type for hybrid DeltaNet/Attention models.
/// Each layer is either a standard attention layer or a DeltaNet recurrent layer.
#[cfg(feature = "deltanet_inference")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum DeltaNetLayerType {
    /// Standard multi-head attention with KV cache.
    #[default]
    Attention,
    /// DeltaNet linear recurrent layer (fast recurrent update, no KV cache needed).
    DeltaNet,
}

// DeltaRoutingConfig::delta_block / is_enabled are intended for the
// delta_routing backend (Plan 097) which is still being wired up. Silence
// dead-code until callers land.
#[allow(dead_code)]
impl DeltaRoutingConfig {
    pub fn delta_block(block_size: usize) -> Self {
        Self {
            mode: DeltaRoutingMode::DeltaBlock,
            block_size,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.mode != DeltaRoutingMode::Off
    }
}

// ---------------------------------------------------------------------------
// DashAttention Config (Plan 106, Research 68)
// ---------------------------------------------------------------------------

/// Configuration for DashAttention adaptive sparse hierarchical attention.
/// Controls α-entmax routing, chunk summarization, and routing bias.
///
/// Fields ordered by descending alignment to minimize padding:
/// usize (8B) → f32 (4B) → bool (1B) — 24 bytes total, no wasted padding.
#[derive(Clone, Copy, Debug)]
pub struct DashAttnConfig {
    /// Chunk size for block-level attention (default: 64).
    pub chunk_size: usize,
    /// α parameter for entmax. Only α=1.5 supported (quadratic, closed-form).
    pub alpha: f32,
    /// Scaling factor γ applied to chunk logits before entmax (default: 1.0).
    pub scaling_factor: f32,
    /// Prior strength σ for routing bias (default: 1e6, weak prior).
    pub sigma: f32,
    /// Whether to estimate diagonal attention contribution (default: true).
    /// Tail-packed after f32 group to avoid bool-between-f32 padding.
    pub estimate_diagonal: bool,
}

impl Default for DashAttnConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64,
            alpha: 1.5,
            scaling_factor: 1.0,
            sigma: 1e6,
            estimate_diagonal: true,
        }
    }
}

// ---------------------------------------------------------------------------
// RTPurbo Retrieval Head Sparse Decode (Plan 126, Research 86)
// ---------------------------------------------------------------------------

/// Head role classification for RTPurbo sparse decode.
///
/// Only ~15% of attention heads ("retrieval heads") need full long-context access.
/// The remaining ~85% ("local heads") attend only to local context + attention sinks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum RetrievalHeadRole {
    /// Local head — sliding window + sink tokens only, no full KV scan.
    #[default]
    Local,
    /// Retrieval head — low-dim projection + dynamic top-p token selection.
    Retrieval,
}

/// Configuration for RTPurbo retrieval head sparse decode.
///
/// Feature gate: `rt_turbo` (opt-in, requires `dash_attn`).
/// Adds head-wise retrieval/local classification + dynamic top-p token selection
/// for decode-phase sparse attention. Complements DashAttention's α-entmax block
/// routing with per-head specialization.
///
/// Must pass 6/6 GOAT proofs before default-on promotion.
///
/// Fields ordered by descending alignment to minimize padding:
/// usize (8B) → f32 (4B) → CalibrationMode (1B) — no padding between groups.
///
/// # Calibration mode (Plan 358)
///
/// [`CalibrationMode::AttentionMass`] is the default (cheaper: 1 forward pass).
/// [`CalibrationMode::CausalNecessity`] is opt-in — strictly stronger on
/// workloads with correlated bystanders but ~10–100× more expensive to
/// calibrate. See `calibrate_from_causal_scores` in `rt_turbo::calibration`.
#[derive(Clone, Copy, Debug)]
pub struct RtTurboConfig {
    /// Low-dimensional projection size for pre-RoPE scoring (default: 16).
    /// Paper ablation: dim=16 is the sweet spot for low-frequency retrieval.
    pub low_dim: usize,
    /// Sliding window size for local heads (default: 8192).
    pub sliding_window: usize,
    /// Number of attention sink tokens always retained for local heads (default: 4).
    pub sink_tokens: usize,
    /// Block size for block-level top-p variant (default: 64).
    /// Should match `DashAttnConfig::chunk_size` for consistent routing.
    pub block_size: usize,
    /// Fraction of heads classified as retrieval heads (default: 0.15).
    /// Paper ablation: 15% is optimal balance of accuracy vs sparsity.
    pub retrieval_head_ratio: f32,
    /// Cumulative attention mass threshold for dynamic top-p selection (default: 0.9).
    /// Paper ablation: top-p=0.9 preserves >93% attention mass at 97% sparsity.
    pub top_p: f32,
    /// Which score semantics to use for head calibration (Plan 358). Default:
    /// `AttentionMass` (cheaper). `CausalNecessity` is opt-in — strictly
    /// stronger on bystander-heavy workloads but ~10–100× more expensive.
    /// `AdaptiveCausal` (Proposal 004) is opt-in — cheap-proxy escalate,
    /// unvalidated, requires per-head OV norms from the caller.
    pub calibration_mode: CalibrationMode,
}

/// Head-calibration score source (Plan 358, Research 362).
///
/// `#[repr(u8)]` for sync-friendly 1-byte representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CalibrationMode {
    /// Observational needle attention-mass (RTPurbo Plan 126 default).
    /// Cheaper: a single forward pass + per-head mass scan.
    #[default]
    AttentionMass = 0,
    /// Causal necessity via activation/path patching IE score (Plan 358).
    /// Strictly stronger — excludes correlated bystanders — but requires
    /// `n_heads × n_calibration_samples` patched forward passes. Requires the
    /// `causal_head_importance` feature on the consuming crate.
    CausalNecessity = 1,
    /// Adaptive cheap-proxy escalate (Proposal 004 — OUR INVENTION, not from
    /// HydraHead). Uses an OV-circuit proxy (`attention_mass / ||OV_out||`)
    /// to detect bystander suspects, then escalates to Plan 358's causal
    /// patching only on those `k` suspects instead of all `n_heads`. Pays zero
    /// patched forwards when there are no bystanders (degenerates to
    /// `AttentionMass`). Requires the `adaptive_causal_calibration` feature.
    ///
    /// **UNVALIDATED.** Promotion to default is blocked on G1 (proxy precision)
    /// and G2 (cost reduction), both deferred to riir-engine. Unlike the other
    /// two modes, the caller must supply per-head OV output norms (from a real
    /// transformer forward) — see `calibrate_from_adaptive_causal`.
    AdaptiveCausal = 2,
}

impl Default for RtTurboConfig {
    fn default() -> Self {
        Self {
            low_dim: 16,
            sliding_window: 8192,
            sink_tokens: 4,
            block_size: 64,
            retrieval_head_ratio: 0.15,
            top_p: 0.9,
            calibration_mode: CalibrationMode::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// LT2 Looped Inference (Plan 108, Research 73)
// ---------------------------------------------------------------------------

/// Looped transformer mode — weight-shared layer repetition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum LoopMode {
    /// Standard single-pass (no looping).
    #[default]
    None,
    /// Weight-shared looping: same layers applied T times.
    /// Effective depth = n_layer × loop_count.
    WeightShared { loop_count: usize },
    /// Training-free loop: ODE-refined sub-stepping over a window of layers.
    /// No extra parameters — pure inference-time retrofit (Plan 136).
    TrainingFree,
}

/// Hybrid attention pattern for looped inference.
/// Controls which layers use full SDPA vs linear attention.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum HybridPattern {
    /// All layers use the same attention mode.
    #[default]
    Uniform,
    /// Depth-level interleave: every Nth layer uses full SDPA.
    /// e.g., Interleave { full_ratio: 5 } = every 5th layer is full.
    /// Paper optimal: 1:4 ratio (full_ratio=5).
    Interleave { full_ratio: usize },
    /// Bookend: first and last layers are full, middle is linear.
    Bookend,
}

/// Loop stability mode for weight-shared looped inference (Plan 428).
///
/// Parameter-free architectural fixes for T-pass loop stability, validated
/// via the §3.6 defend-wrong PoC benchmark.
///
/// **`InterLoopNorm` ships** — the PoC proved it's the sole fix that
/// controls residual norm growth. FLA-res (direct residual addition of
/// `prev_h` at every layer) caused catastrophic norm explosion (~2.2B× at
/// T=12), and Attention Injection was a no-op for single-position attention
/// (softmax of 1 element = 1.0, so Q doesn't affect the output). Both were
/// dropped per the defend-wrong verdict.
///
/// **`FixedAnchor`** (Issue 698 T2, GRT arXiv:2608.15062 Table 11): composes
/// the inter-loop norm (GRT's gate consumes LN inputs — normalization is a
/// prerequisite, not a competitor) with a FROZEN loop anchor: the state
/// after the first loop iteration (h^(0)) is hoisted once into a dedicated
/// buffer, and every gated iteration re-injects `ρ_τ ⊙ anchor` instead of
/// the drifting `ρ_τ ⊙ h^(τ-1)`. Paper ordering (trained anchor-gate
/// weights): frozen prelude output 2.68 < drifting h(r−1) 3.38 (+0.70 nats)
/// < raw embedding 3.73 < zeros 8.08. On our prelude-less arch the tau==0
/// pre-pass state IS the raw embedding — the paper's distinct, worse arm —
/// so the anchor is hoisted once the FIRST iteration completes (h^(0) is
/// (the prelude-output interpretant). Random-weight caveat: the paper's
/// numbers are anchor-trained; the ORDERING is the structural claim under
/// test (`tests/issue_698_t2_fixed_anchor.rs`). Zero cost when not selected.
///
/// Issue 698 T6 added `StateNoise { scale }` — an `f32` payload, so `Eq` is
/// no longer derived (only `==`/PartialEq comparisons exist; the payload
/// makes the enum 8 bytes, no `#[repr(u8)]` POD contract is relied on).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(u8)]
pub enum LoopStabilityMode {
    /// No inter-loop stabilization (byte-identical to pre-Plan-428 behavior).
    #[default]
    None,
    /// Inter-loop RMSNorm: normalize the hidden state between loop iterations
    /// (tau > 0), before the inner layer pass. PoC: norm ratio 3.34× vs
    /// baseline 11.19×, KL 0.0008, step-size trend converging (14.9 → 2.05).
    InterLoopNorm,
    /// GRT fixed-anchor loop (Issue 698 T2): the inter-loop norm PLUS a
    /// frozen anchor hoisted once at the end of the first loop iteration
    /// (= h^(0)); every gated iteration re-injects `ρ_τ ⊙ anchor` instead
    /// of the drifting `ρ_τ ⊙ h^(τ-1)`. GRT Table 11: fixed prelude output
    /// beats drifting h(r−1) by +0.70 nats; zeros are catastrophic (8.08).
    FixedAnchor,
    /// Issue 698 T6 — per-step state noise (GRT arXiv:2608.15062, the
    /// paper's smallest ablation): the inter-loop norm PLUS a BLAKE3-seeded
    /// Gaussian perturbation of the loop input at every iteration (tau > 0).
    /// `scale` is RELATIVE to the state's own RMS (config-independent;
    /// 0.05 = 5% noise). `scale == 0.0` skips the injection entirely → the
    /// mode is bit-identical to `InterLoopNorm` (the flag-off pin). The seed
    /// hashes only `(pos, tau)` so the noise field is call-independent;
    /// Box–Muller over BLAKE3-XOF uniform words, zero allocation.
    StateNoise { scale: f32 },
}

/// Head-specific sigmoid gate after SDPA, before Wo.
/// Zero-initialized → starts at sigmoid(0) = 0.5 (neutral multiplicative identity).
#[derive(Clone)]
pub struct SdpaOutputGate {
    /// Gate weights: [n_heads * head_dim, dim].
    /// Zero-init so gate starts at sigmoid(0) = 0.5.
    pub w_gate: Vec<f32>,
}

impl SdpaOutputGate {
    /// Allocate zeroed gate weights.
    pub fn new(n_heads: usize, head_dim: usize, dim: usize) -> Self {
        Self {
            w_gate: vec![0.0; n_heads * head_dim * dim],
        }
    }

    /// Apply sigmoid-gated projection to attention output.
    ///
    /// Computes: `gate[i] = sigmoid(W_gate[i] · attn_out)`, then `attn_out[i] *= gate[i]`.
    /// Zero-init weights produce sigmoid(0) = 0.5 for all (neutral half-pass).
    /// Paper reference: +0.3–0.5 avg points on zero-shot benchmarks.
    pub fn forward(&self, attn_out: &mut [f32], dim: usize, temp: &mut [f32]) {
        let n = attn_out.len();
        debug_assert!(temp.len() >= n, "temp buffer too small");
        debug_assert!(self.w_gate.len() >= n * dim, "gate weights too small");

        // Step 1: Compute gate signal = sigmoid(W_gate @ attn_out)
        // Batch matvec then batch sigmoid avoids per-element loop overhead
        crate::simd::simd_matvec(temp, &self.w_gate, attn_out, n, dim);

        // SIMD sigmoid: temp = -temp, exp, then 1/(1+exp)
        crate::simd::simd_scale_inplace(&mut temp[..n], -1.0);
        crate::simd::simd_exp_inplace(&mut temp[..n]);
        crate::simd::simd_add_scalar_inplace(&mut temp[..n], 1.0);
        // temp now = 1 + exp(-x), invert: temp = 1/temp = sigmoid
        crate::simd::simd_reciprocal_inplace(&mut temp[..n]);

        // Step 2: Apply gate elementwise via SIMD scale-mul (fused)
        // attn_out[i] *= temp[i] is element-wise multiply
        // Use simd_scale_mul_inplace with scale=1.0: attn[i] = temp[i] * attn[i] * 1.0
        crate::simd::simd_scale_mul_inplace(attn_out, &temp[..n], 1.0);
    }
}

/// Interpolation shape of a convex copy-late gate schedule (Issue 698 T3).
///
/// All shapes interpolate the per-loop copy weight `g_τ` from `g0` (loop 0,
/// write-open) to `gR` (final loop, copy-closed). GRT arXiv:2608.15062: the
/// trained gate's effective openness declines monotonically (0.182 → 0.066,
/// §5) — write-early, copy-late. The shape controls HOW the closure is
/// distributed across loops; the endpoints control how much.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum CopyLateShape {
    /// Linear: `g_τ = g0 + (gR − g0) · τ / (L − 1)`.
    #[default]
    Linear,
    /// Quadratic ease toward `gR`: `g_τ = g0 + (gR − g0) · (1 − (1 − t)²)`
    /// with `t = τ / (L − 1)` — closes FAST early, flattens near `gR`.
    /// Fastest route to the contraction regime (updates ∝ 1 − g_τ shrink
    /// early).
    EaseOutClose,
    /// Piecewise step at the midpoint: `g0` for `τ < L/2`, `gR` after —
    /// the coarsest 2-phase proxy for the paper's write/copy phases.
    StepMid,
}

/// Per-loop residual scaling gate.
/// h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)
/// Zero-init so first iteration is h̃^(1) (no residual from "previous").
///
/// **Convex copy-late mode** (Issue 698 T3, GRT arXiv:2608.15062 C1a/C4):
/// when `convex_schedule` is `Some`, the per-loop combine switches from the
/// additive form above to the convex blend
/// `h^(τ) = g_τ ⊙ src + (1 − g_τ) ⊙ h̃^(τ)` with a SCALAR per-loop `g_τ`.
/// Two free properties the additive form lacks: (1) boundedness — a scalar
/// convex blend gives ‖h^(τ)‖ ≤ max(‖src‖, ‖h̃^(τ)‖) by norm convexity (the
/// additive `h̃ + ρ⊙src` has NO bound — the exact instability Plan 428
/// fights); (2) contraction — the update magnitude is ∝ (1 − g_τ), so a
/// copy-late schedule (g_τ → 1) drives the update to zero and the loop
/// converges to a fixed point (T2 measured the constant-ρ additive arm
/// NEVER settling: ref drift 2.404 vs 3.2e-8 zeros — the closing schedule is
/// the required complement). Per-channel gates would forfeit property (1)'s
/// exactness (p=(1,0), o=(0,1), per-channel g=(1,0) → ‖h‖=√2 > 1); the
/// scalar schedule is the strongest modelless interpretant.
#[derive(Clone)]
pub struct ResidualGate {
    /// Per-loop gates: [loop_count, dim].
    /// Each ρ_τ is element-wise, zero-init.
    /// Empty under the convex constructors (the convex path never reads it).
    pub gates: Vec<f32>,
    /// Per-loop scalar copy weights g_τ: [loop_count] (entry τ = gate at loop
    /// τ; entry 0 is never read — gating starts at τ = 1). `None` = additive
    /// path (all classic constructors). Values are clamped to [0, 1] at
    /// construction — the free bound requires g ∈ [0, 1].
    pub convex_schedule: Option<Vec<f32>>,
    /// Issue 698 T8 — hand conditional gate (the paper's contrastive
    /// projection finding, open on divergence): the per-loop copy weight is
    /// computed AT RUNTIME from the trajectory itself,
    /// `g_τ = σ(β·(cos(S(τ−1), S(τ−2)) − θ) + b)`, instead of a pre-built
    /// schedule. Present → the adaptive path replaces both the convex
    /// schedule and the additive form (the same convex blend, adaptive g).
    /// Mechanism gate (T8 probe): divergence co-locates with marginal loop
    /// gain at the pre-registered rank thresholds in the anchored context
    /// (`tests/issue_698_t8_gate_probe.rs`, 13/15 per-r positive).
    pub conditional: Option<ConditionalGate>,
}

/// Issue 698 T8 — the hand conditional gate constants.
///
/// `g = σ(β·(cos − θ) + b)`: OPEN (small g → take the new loop output) on
/// DIVERGENCE (cos(S(τ−1), S(τ−2)) below θ), CLOSED (g → 1 → freeze/copy)
/// once consecutive states align past θ. The mechanism gate probe showed
/// the co-location grows with r (r=2 ≈ 0, r≥7 substantial) — the gate
/// discriminates exactly where a copy-late schedule must decide when to
/// close, per token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditionalGate {
    /// Sigmoid steepness (nats per unit cos). 400 gives a sharp transition
    /// (|Δcos| = 0.01 swings g by ~σ(4)−σ(−4)).
    pub beta: f32,
    /// Cosine threshold: the divergence→convergence decision point.
    pub theta: f32,
    /// Bias (nats) — shifts the transition; b = +2 puts g ≈ 0.88 AT θ.
    pub bias: f32,
    /// Hard-freeze clamp (Issue 698 T8 A/B finding): when the soft g exceeds
    /// [`ConditionalGate::FREEZE_SOFT`], snap to EXACTLY 1.0. The bare
    /// sigmoid never reaches 1, leaving a permanent (1−g) ≈ 0.25% update
    /// kick — the T2 constant-ρ never-settles class in miniature (measured
    /// ref drift 1.9 vs the static schedule's 1.7e-6). The clamp preserves
    /// the adaptivity (the trigger is still the per-token cosine) while
    /// restoring exact contraction: g = 1 is a bit-exact full copy (T3's
    /// spec-test degenerate), the state freezes, cos ≡ 1, the gate stays
    /// shut — contraction by construction.
    pub hard_freeze: bool,
}

impl ConditionalGate {
    /// Soft-g level above which the hard-freeze clamp snaps to 1.0.
    pub const FREEZE_SOFT: f32 = 0.9;
}

impl ResidualGate {
    /// Allocate zeroed residual gates.
    pub fn new(loop_count: usize, dim: usize) -> Self {
        Self {
            gates: vec![0.0; loop_count * dim],
            convex_schedule: None,
            conditional: None,
        }
    }

    /// Deterministic loop-stable residual gates (Plan 483 T2.1, §3.5 path 2).
    ///
    /// Sets gates to a constant `decay` factor for τ > 0, enabling information
    /// carry-forward between T passes. The first loop (τ=0) has zero gates
    /// (no previous state to carry forward).
    ///
    /// This is a **modelless** construction — no training, no gradient descent.
    /// The `decay` factor controls the trade-off between carry-forward strength
    /// and stability. Conservative values (0.1–0.3) are safe for most weight
    /// matrices; larger values risk divergence.
    ///
    /// Rationale: the zero-init default (`new()`) makes every T-pass effectively
    /// independent — no hidden state carries forward between loops. This
    /// undermines the LT2 paper's "effective depth T×n_layer" claim. A non-zero
    /// deterministic gate restores the residual connection across loops without
    /// requiring trained gate parameters.
    ///
    /// # Arguments
    /// * `loop_count` - Number of T-passes (T)
    /// * `dim` - Hidden dimension (n_embd)
    /// * `decay` - Constant gate value for τ > 0 (e.g., 0.1 = conservative)
    #[inline]
    pub fn new_loop_stable(loop_count: usize, dim: usize, decay: f32) -> Self {
        let mut gates = vec![0.0f32; loop_count * dim];
        // τ=0: zero (no previous state). τ>0: constant decay.
        for tau in 1..loop_count {
            let offset = tau * dim;
            gates[offset..offset + dim].fill(decay);
        }
        Self {
            gates,
            convex_schedule: None,
            conditional: None,
        }
    }

    /// Deterministic loop-stable residual gates with exponential decay
    /// (Plan 483 T2.1, §3.5 path 2 variant).
    ///
    /// ρ_τ = `base`^(τ-1) for τ > 0 — later loops contribute exponentially less.
    /// This schedule mirrors the spectral-radius-based stabilization where the
    /// residual contribution decays as the hidden state converges.
    ///
    /// # Arguments
    /// * `loop_count` - Number of T-passes (T)
    /// * `dim` - Hidden dimension (n_embd)
    /// * `base` - Decay base (e.g., 0.5 → ρ_1=1.0, ρ_2=0.5, ρ_3=0.25, ...)
    #[inline]
    pub fn new_loop_stable_exp_decay(loop_count: usize, dim: usize, base: f32) -> Self {
        let mut gates = vec![0.0f32; loop_count * dim];
        for tau in 1..loop_count {
            let offset = tau * dim;
            let val = base.powi((tau - 1) as i32);
            gates[offset..offset + dim].fill(val);
        }
        Self {
            gates,
            convex_schedule: None,
            conditional: None,
        }
    }

    /// Convex copy-late schedule with LINEAR interpolation (Issue 698 T3 —
    /// the issue-pinned entry point; see [`Self::copy_late_schedule_shaped`]
    /// for the shape sweep).
    ///
    /// # Arguments
    /// * `loop_count` - Number of T-passes (T). The schedule has exactly this
    ///   many entries; entry 0 is `g0` (never read — gating starts at τ = 1).
    /// * `g0` - Copy weight at loop 0 (write-open end; low = writes more).
    /// * `gR` - Copy weight at the final loop (copy-closed end; GRT: g > 0.95
    ///   is the copy-saturated band).
    ///
    /// The `gates` elementwise buffer is left EMPTY — the convex path never
    /// reads it, and a scalar schedule carries no per-channel data.
    /// `g0`/`gR` are clamped into [0, 1] (the free norm bound requires it).
    /// `gR` keeps the paper's notation (g at loop R−1) over snake_case.
    #[inline]
    #[allow(non_snake_case)]
    pub fn copy_late_schedule(loop_count: usize, g0: f32, gR: f32) -> Self {
        Self::copy_late_schedule_shaped(loop_count, g0, gR, CopyLateShape::Linear)
    }

    /// Convex copy-late schedule with an explicit interpolation shape
    /// (Issue 698 T3).
    ///
    /// Builds `convex_schedule[τ] = clamp01(g0 + (gR − g0) · shape(τ/(L−1)))`
    /// — a SCALAR per-loop copy weight consumed by `forward_looped`'s convex
    /// blend path: `h^(τ) = g_τ ⊙ src + (1 − g_τ) ⊙ h̃^(τ)`. Modelless: no
    /// training, deterministic, allocation once at construction (the loop
    /// reads `schedule[τ]` — zero per-forward allocation).
    ///
    /// Form-mismatch caveat (issue 698 T3): existing checkpoints are trained
    /// under the ADDITIVE form — switching the combine form on shared weights
    /// is an OOD intervention; the fixture A/B arbitrates
    /// (`tests/issue_698_t3_copy_late.rs`).
    ///
    /// A `loop_count == 0` or `1` schedule is degenerate (no gated loop ever
    /// reads entry ≥ 1); it is accepted and returns a well-formed schedule.
    /// `gR` keeps the paper's notation (g at loop R−1) over snake_case.
    #[inline]
    #[allow(non_snake_case)]
    pub fn copy_late_schedule_shaped(
        loop_count: usize,
        g0: f32,
        gR: f32,
        shape: CopyLateShape,
    ) -> Self {
        let clamp01 = |v: f32| v.clamp(0.0, 1.0);
        let (a, b) = (clamp01(g0), clamp01(gR));
        let mut schedule = Vec::with_capacity(loop_count);
        match shape {
            CopyLateShape::StepMid => {
                let mid = loop_count / 2;
                for tau in 0..loop_count {
                    schedule.push(if tau < mid { a } else { b });
                }
            }
            CopyLateShape::Linear | CopyLateShape::EaseOutClose => {
                for tau in 0..loop_count {
                    let t = if loop_count <= 1 {
                        0.0
                    } else {
                        tau as f32 / (loop_count - 1) as f32
                    };
                    let s = match shape {
                        // 1 − (1 − t)²: derivative 2(1 − t) — closes fast early.
                        CopyLateShape::EaseOutClose => 1.0 - (1.0 - t) * (1.0 - t),
                        _ => t,
                    };
                    schedule.push(clamp01(a + (b - a) * s));
                }
            }
        }
        Self {
            gates: Vec::new(),
            convex_schedule: Some(schedule),
            conditional: None,
        }
    }

    /// Issue 698 T8 — the hand conditional gate: an ADAPTIVE convex blend
    /// whose copy weight is computed per loop from the trajectory itself,
    /// `g_τ = σ(β·(cos(S(τ−1), S(τ−2)) − θ) + b)`, clamped to [0, 1] by the
    /// sigmoid's range. Open on divergence (moving state → take the new
    /// loop output), closed on convergence (aligned states → freeze on the
    /// injected source) — the per-token interpretant of T3's copy-late
    /// schedule, closing WHEN the state settles instead of on a fixed
    /// timetable.
    ///
    /// The same convex-blend path as [`Self::copy_late_schedule_shaped`]
    /// applies the value, so the free bound ‖h^(τ)‖ ≤ max(‖src‖, ‖h̃^(τ)‖)
    /// holds for every g ∈ [0, 1] (T3's spec test covers the blend; the
    /// conditional only changes WHERE g comes from).
    ///
    /// Modelless caveat (the issue's own "honest coin-flip" label, now
    /// bounded by the mechanism-gate PASS): the constants (β, θ, b) are
    /// hand-set — weights that never learned contrastive reading may prefer
    /// different operating points; the A/B bench arbitrates the direction
    /// (`tests/issue_698_t8_conditional_ab.rs`).
    ///
    /// The `gates` elementwise buffer is left EMPTY (the adaptive path is a
    /// scalar gate — it never reads per-channel data), mirroring the convex
    /// constructors. `loop_count`/`dim` are accepted for constructor
    /// symmetry and ignored.
    #[inline]
    #[allow(non_snake_case)]
    pub fn new_conditional(
        _loop_count: usize,
        _dim: usize,
        beta: f32,
        theta: f32,
        bias: f32,
    ) -> Self {
        Self {
            gates: Vec::new(),
            convex_schedule: None,
            conditional: Some(ConditionalGate {
                beta,
                theta,
                bias,
                hard_freeze: false,
            }),
        }
    }

    /// [`Self::new_conditional`] with the hard-freeze clamp: once the soft
    /// copy weight exceeds `ConditionalGate::FREEZE_SOFT` (0.9), g snaps to
    /// EXACTLY 1.0 — a bit-exact full copy that freezes the carried state
    /// (cos ≡ 1 keeps it frozen). The A/B measured why: the bare sigmoid's
    /// residual (1−g) kick never settles (ref drift 1.9 vs static 1.7e-6);
    /// the clamp restores exact contraction while keeping the per-token
    /// adaptive trigger (`tests/issue_698_t8_conditional_ab.rs`).
    #[inline]
    #[allow(non_snake_case)]
    pub fn new_conditional_hard(
        _loop_count: usize,
        _dim: usize,
        beta: f32,
        theta: f32,
        bias: f32,
    ) -> Self {
        Self {
            gates: Vec::new(),
            convex_schedule: None,
            conditional: Some(ConditionalGate {
                beta,
                theta,
                bias,
                hard_freeze: true,
            }),
        }
    }

    /// The copy weight at loop `tau` for the CONVEX path, clamped to the
    /// schedule's last entry when `tau` runs past the end (elastic override
    /// executing more loops than the schedule was built for: stay at the
    /// closed end — the contraction regime — rather than going inert).
    /// Returns `None` when no schedule is installed (additive path).
    #[inline]
    pub fn convex_gate_at(&self, tau: usize) -> Option<f32> {
        let s = self.convex_schedule.as_ref()?;
        Some(
            s.get(tau)
                .copied()
                .unwrap_or_else(|| *s.last().unwrap_or(&1.0)),
        )
    }

    /// The adaptive copy weight at loop `tau` (Issue 698 T8):
    /// `σ(β·(cos(prev, prev_prev) − θ) + b)` — `None` unless the conditional
    /// gate is installed AND `tau ≥ 2` (the cosine needs TWO carried states;
    /// τ ∈ {0, 1} have at most one — those loops stay write-open, the
    /// paper's gate also starts open).
    #[inline]
    pub fn conditional_gate_at(&self, tau: usize, prev: &[f32], prev_prev: &[f32]) -> Option<f32> {
        let c = self.conditional.as_ref()?;
        if tau < 2 {
            return None;
        }
        let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..prev.len() {
            dot += prev[i] * prev_prev[i];
            na += prev[i] * prev[i];
            nb += prev_prev[i] * prev_prev[i];
        }
        if na == 0.0 || nb == 0.0 {
            // Degenerate zero state: nothing has converged — stay open.
            return Some(0.0);
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        let g = crate::simd::fast_sigmoid(c.beta * (cos - c.theta) + c.bias).clamp(0.0, 1.0);
        if c.hard_freeze && g > ConditionalGate::FREEZE_SOFT {
            Some(1.0)
        } else {
            Some(g)
        }
    }
}

// ---------------------------------------------------------------------------
// SR²AM Configurator Bandit (Plan 112, Research 076)
// ---------------------------------------------------------------------------

/// SR²AM Configurator decision — learned per-turn planning regulation.
///
/// The configurator selects one of these arms per inference turn based on
/// context (domain + entropy bin). UCB1 balances exploration vs exploitation.
///
/// - `PlanNew`: reset tree, full budget allocation (high uncertainty, new sub-problem)
/// - `PlanExtend`: keep tree, extend depth by one level (moderate uncertainty, continuing)
/// - `PlanSkip`: skip tree search, direct token sampling (low uncertainty, confident)
#[cfg(feature = "sr2am_configurator")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum PlanningDecision {
    /// Reset tree, full budget allocation (high uncertainty, new sub-problem).
    PlanNew,
    /// Keep tree, extend depth by one level (moderate uncertainty, continuing).
    PlanExtend,
    /// Skip tree search, direct token sampling (low uncertainty, confident).
    PlanSkip,
    /// Activate SpecHop continuous speculation with k speculative threads (Plan 131).
    /// Selected when speculator latency α is low and tool ratio β is moderate.
    SpecHop { k: usize },
    /// Harness update: AbsorbCompress promote + HotSwapPruner reload (Plan 163 T5).
    /// Selected when harness has plateaued and a compressed arm set may improve.
    #[cfg(feature = "sia_feedback")]
    HarnessUpdate,
    /// Weight update: trigger riir-gpu training step on accumulated TrialLog (Plan 163 T6).
    /// Selected when stall detection fires — reward plateau suggests weights need updating.
    #[cfg(feature = "sia_feedback")]
    WeightUpdate,
}

/// Context key for configurator bandit — coarse entropy binning.
///
/// Entropy is discretized into 10 bins via `floor(entropy * 10.0)` clamped to 0..9.
/// Combined with domain index, this provides context-aware arm selection.
#[cfg(feature = "sr2am_configurator")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfiguratorContext {
    /// Domain index from bandit infrastructure.
    pub domain: usize,
    /// Coarse entropy bin: `floor(entropy * 10.0)`, clamped to 0..9.
    /// u8 — values are 0..9, packed after usize to avoid padding.
    pub entropy_bin: u8,
    /// Coarse desperation bin: `floor(desperation * 10.0)`, clamped to 0..9.
    /// Plan 162 T11: emotion vector desperation score as additional context.
    /// 0 = not desperate, 9 = highly desperate.
    pub desperation_bin: u8,
    /// Coarse epiplexity bin: `floor(epiplexity * 10.0)`, clamped to 0..9.
    /// Plan 130 T4: structural information content (S_T) as additional context.
    /// 0 = no structure detectable, 9 = highly structured.
    pub epiplexity_bin: u8,
}

#[cfg(feature = "sr2am_configurator")]
impl ConfiguratorContext {
    /// Create context without desperation information (legacy compatibility).
    ///
    /// Sets `desperation_bin` to 0 (not desperate). Use `with_desperation()`
    /// when emotion vector data is available.
    pub fn new(domain: usize, entropy_bin: usize) -> Self {
        Self {
            domain,
            entropy_bin: (entropy_bin.min(9)) as u8,
            desperation_bin: 0,
            epiplexity_bin: 0,
        }
    }

    /// Set the desperation bin from a raw desperation score.
    ///
    /// `floor(desperation * 10.0)`, clamped to 0..9.
    pub fn with_desperation(mut self, desperation: f32) -> Self {
        self.desperation_bin = ((desperation * 10.0).floor() as u8).min(9);
        self
    }

    /// Set the epiplexity bin from a raw epiplexity score (S_T).
    ///
    /// `floor(epiplexity * 10.0)`, clamped to 0..9.
    /// S_T measures structural information content — higher values indicate
    /// more structure that a bounded observer can extract from the data.
    pub fn with_epiplexity(mut self, epiplexity: f32) -> Self {
        self.epiplexity_bin = ((epiplexity * 10.0).floor() as u8).min(9);
        self
    }

    /// Create context from entropy and epiplexity signals.
    ///
    /// Convenience constructor that bins both entropy (H_T proxy) and
    /// epiplexity (S_T structural information) in one call.
    /// `desperation_bin` defaults to 0.
    pub fn from_entropy_epiplexity(domain: usize, entropy: f32, epiplexity: f32) -> Self {
        let entropy_bin = ((entropy * 10.0).floor() as u8).min(9);
        let epiplexity_bin = ((epiplexity * 10.0).floor() as u8).min(9);
        Self {
            domain,
            entropy_bin,
            desperation_bin: 0,
            epiplexity_bin,
        }
    }

    /// Discretize epiplexity (S_T) into a coarse bin index.
    ///
    /// `floor(epiplexity * 10.0)`, clamped to 0..9.
    pub fn epiplexity_bin(epiplexity: f32) -> u8 {
        ((epiplexity * 10.0).floor() as u8).min(9)
    }
}

// ---------------------------------------------------------------------------
// EqR Convergence Selection (Plan 119)
// ---------------------------------------------------------------------------

/// Selection strategy for width-scaled rollouts (EqR convergence-based selection).
///
/// Maps to `WidthSelectionMode` (in `crate::speculative::dd_tree`) at runtime.
/// This enum lives in `katgpt-core` so Config can reference it without depending on
/// the speculative decode module.
///
/// - `BestQ`: Highest cumulative relevance (PTRM default, no behavior change)
/// - `MajorityVote`: Most common path across rollouts (mode@K)
/// - `Top1Converged`: Smallest final residual ∥p_{d+1} − p_d∥ (EqR proxy)
/// - `BtRank`: Pairwise Bradley-Terry ranking (requires `bt_rank` feature)
///
/// **Precondition:** `Top1Converged` is only reliable after landscape shaping
/// (RI + NI training). See Research 079 (EqR) for theoretical justification.
#[repr(u8)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConvergenceSelector {
    /// Select rollout with highest cumulative relevance score (PTRM Q-head analog).
    #[default]
    BestQ,
    /// Select the most frequent path across all rollouts (mode@K, majority vote).
    MajorityVote,
    /// Select rollout with smallest marginal-change residual ∥p_{d+1} − p_d∥ (EqR proxy).
    Top1Converged,
    /// Pairwise Bradley-Terry ranking across rollouts (requires `bt_rank` feature).
    BtRank,
}

// ---------------------------------------------------------------------------
// Wall Attention — Diagonal Forget Gates Replacing RoPE (Plan 173)
// ---------------------------------------------------------------------------

/// Wall Attention configuration (Plan 173, Research: Wall Attention paper).
///
/// Wall replaces RoPE with diagonal forget gates applied as factorized Q/K rescaling:
/// `q̃_i = exp(P_i) ⊙ q_i`, `k̃_j = exp(-P_j) ⊙ k_j`.
/// This means attention kernels are UNCHANGED — they receive pre-rescaled Q and K.
///
/// Only applicable to Wall-trained models (requires W_g gate projection weights).
///
/// The wall on/off switch lives at the parent `Config.wall_config: Option<WallConfig>`
/// level — `None` means use RoPE/fallback, `Some(_)` means Wall is active. There is
/// no `use_wall` field on the struct itself (the canonical design since Plan 173).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[cfg(feature = "wall_attention")]
pub struct WallConfig {
    /// Gate bias initialization value. Default 6.0 = open gate (vanilla attention behavior).
    /// Lower values → more active forgetting (gate_bias=0 → retention ≈ 0.62).
    pub gate_bias: f32,
    /// Maximum gate log-sigmoid clamp value. Default 0.87 (matches paper).
    /// Gates are clamped to (-gate_max, 0] after log-sigmoid.
    pub gate_max: f32,
    /// Use key-projected gate variant (derive gate from K projection).
    /// Preferred for zero KV cache overhead — gate is computed from key, not hidden state.
    pub use_key_projected: bool,
    /// Gate projection dimension = n_kv_heads * head_dim.
    /// Default 0 = compute from model dims via [`Self::with_dims`] or
    /// [`Self::validate`]. Set explicitly only when the consumer already knows
    /// the dim and wants to skip the derivation.
    pub gate_proj_dim: usize,
}

#[cfg(feature = "wall_attention")]
impl Default for WallConfig {
    fn default() -> Self {
        Self {
            gate_bias: 6.0,
            gate_max: 0.87,
            use_key_projected: true,
            gate_proj_dim: 0,
        }
    }
}

#[cfg(feature = "wall_attention")]
impl WallConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate config consistency against model dimensions.
    ///
    /// Checks:
    /// - `gate_max` is in the open interval (0, 1) — soft-clamp must produce
    ///   finite non-degenerate log-gates.
    /// - `gate_proj_dim` is either unset (0, meaning "derive from dims") or
    ///   exactly `n_kv_heads * head_dim`.
    pub fn validate(&self, n_kv_heads: usize, head_dim: usize) -> Result<(), String> {
        if self.gate_max <= 0.0 || self.gate_max >= 1.0 {
            return Err(format!("gate_max must be in (0, 1), got {}", self.gate_max));
        }
        let expected_dim = n_kv_heads * head_dim;
        if self.gate_proj_dim != 0 && self.gate_proj_dim != expected_dim {
            return Err(format!(
                "gate_proj_dim ({}) must equal n_kv_heads * head_dim ({})",
                self.gate_proj_dim, expected_dim
            ));
        }
        Ok(())
    }

    /// Builder: derive `gate_proj_dim` from model dimensions.
    /// Consumes and returns `self` for chaining.
    pub fn with_dims(mut self, n_kv_heads: usize, head_dim: usize) -> Self {
        self.gate_proj_dim = n_kv_heads * head_dim;
        self
    }
}

// ---------------------------------------------------------------------------
// Collapse-Aware Adaptive Thinking (Plan 212)
// ---------------------------------------------------------------------------

/// Per-instance adaptive budget for collapse-aware thinking.
///
/// Controls when mid-reasoning early exit triggers and how efficiency rewards
/// are shaped. Feature-gated behind `collapse_aware_thinking`.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct ThinkingBudget {
    /// Maximum thinking tokens before forced termination.
    pub max_tokens: u32,
    /// Hesitation count threshold τ — collapse triggers when exceeded.
    pub collapse_threshold: u32,
    /// Efficiency–accuracy trade-off for reward shaping.
    /// Higher γ penalizes longer traces more aggressively.
    /// Range: [0.0, 1.0].
    pub efficiency_gamma: f32,
}

#[cfg(feature = "collapse_aware_thinking")]
impl Default for ThinkingBudget {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            collapse_threshold: 3,
            efficiency_gamma: 0.5,
        }
    }
}

#[cfg(test)]
mod issue698_conditional_gate_tests {
    use super::ResidualGate;

    #[test]
    fn conditional_gate_contract() {
        let g = ResidualGate::new_conditional(32, 8, 400.0, 0.99, 2.0);
        let a = [0.5f32; 8];
        let b = [0.25f32; 8];
        // No conditional installed → None (plain gates stay additive).
        assert!(
            ResidualGate::new(32, 8)
                .conditional_gate_at(5, &a, &b)
                .is_none()
        );
        // τ < 2 → None (the cosine needs two carried states).
        assert!(g.conditional_gate_at(0, &a, &b).is_none());
        assert!(g.conditional_gate_at(1, &a, &b).is_none());
        // Identical states → cos 1 → σ(400·0.01 + 2) ≈ 0.9975 (closed).
        let closed = g.conditional_gate_at(2, &a, &a).expect("some at τ=2");
        assert!(closed > 0.99, "identical states must close: {closed}");
        // Orthogonal states → cos 0 → σ(−396) = 0 (wide open).
        let mut c = [0.0f32; 8];
        c[0] = 1.0;
        let open = g.conditional_gate_at(2, &a, &c).expect("some");
        assert!(open < 1e-10, "orthogonal states must open: {open}");
        // Zero-norm degenerate → open (nothing has converged).
        let z = [0.0f32; 8];
        let deg = g.conditional_gate_at(2, &z, &a).expect("some");
        assert_eq!(deg, 0.0);
    }

    #[test]
    fn hard_freeze_clamps_to_exactly_one() {
        let soft = ResidualGate::new_conditional(32, 8, 400.0, 0.99, 2.0);
        let hard = ResidualGate::new_conditional_hard(32, 8, 400.0, 0.99, 2.0);
        let a = [0.5f32; 8];
        let s = soft.conditional_gate_at(2, &a, &a).expect("some");
        let h = hard.conditional_gate_at(2, &a, &a).expect("some");
        // Same trigger region (identical states close), but the hard clamp
        // snaps the blend weight to EXACTLY 1.0 — the bit-exact full-copy
        // degenerate (frozen state ⇒ cos ≡ 1 ⇒ stays frozen).
        assert!(s > 0.9);
        assert_eq!(h, 1.0);
        // Below the clamp threshold both arms agree (the soft value).
        let mut lo = [0.0f32; 8];
        lo[0] = 1.0;
        let s2 = soft.conditional_gate_at(2, &a, &lo).expect("some");
        let h2 = hard.conditional_gate_at(2, &a, &lo).expect("some");
        assert_eq!(s2, h2);
        assert!(s2 < 1e-10);
    }
}

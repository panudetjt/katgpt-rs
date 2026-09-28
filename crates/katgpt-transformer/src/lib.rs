//! katgpt-transformer: Transformer substrate types shared between katgpt-rs root
//! and riir-engine.
//!
//! Pure data + loaders — no forward logic. The forward kernels stay in the
//! katgpt-rs root crate because they compose cognitive primitives
//! (`crate::hla`, `crate::sleep`, `crate::tf_loop`, `crate::gdn2`, etc.)
//! that do not exist in this substrate crate.
//!
//! See `README.md` for the architectural rationale (why this is a separate
//! crate from `katgpt-core`).

mod context;
mod contiguous;
mod kv_cache;
#[cfg(feature = "transformer_moe")]
pub mod moe;
// MoE analytic backward pass (Plan 318 Phase C C4). CPU reference for the
// GPU backward. Gated behind `moe_backward` (implies `transformer_moe`).
// katgpt-rs is modelless-by-mandate; this is the training-time reference
// consumed by riir-train, never on the production inference path.
#[cfg(feature = "transformer_attn_res")]
pub mod attn_res;
#[cfg(feature = "moe_backward")]
pub mod moe_backward;
mod mtp;
mod weights;

// Phase 9 absorptions (Proposal 003) — modules moved from katgpt-rs root.
#[cfg(feature = "dense_mesh")]
pub mod dense_mesh;
#[cfg(feature = "kog_cpu_fusion")]
pub mod mbu;
#[cfg(feature = "swir_switch_thinking")]
pub mod swir;
#[cfg(feature = "tf_loop")]
pub mod tf_loop;
// Phase 12 T4.6 (2026-07-04): module moved from katgpt-rs root.
// thinking_cot hosts the ThinkingStrategy trait + StepContext + StepDirective
// wiring types consumed by swir/strategy_adapter (Plan 194).
#[cfg(feature = "thinking_cot")]
pub mod thinking_cot;

pub use context::PrefillContext;
#[cfg(feature = "wall_attention")]
pub use context::{GateStatistics, WallPrefixState};
pub use contiguous::ContiguousWeights;
/// Page size in tokens for [`PagedKVCache`] (tuneable, must be power of 2).
pub use kv_cache::PAGE_SIZE;
pub use kv_cache::{
    KVCache, KVLayerSnapshot, KVSnapshot, MultiLayerKVCache, PagedKVCache, RavenKVCache,
    preload_kv_cache,
};
pub use mtp::{MtpProjection, load_mtp_projection, project_target_activation};
pub use weights::{LayerWeights, LayerWeightsF16, TransformerWeights, TransformerWeightsF16};

// Contiguous ternary loader (Plan 148, gated `plasma_path`).
#[cfg(feature = "plasma_path")]
pub use contiguous::load_ternary_bits;

// Contiguous binary loader (Issue 145, gated `binary_plasma`).
#[cfg(feature = "binary_plasma")]
pub use contiguous::load_binary_bits;
// Contiguous group-scale ternary loader — the Q2_0_g128 container (Issue 578).
#[cfg(feature = "ternary_group_scale")]
pub use contiguous::load_ternary_group_bits;

// Decode stage for specialized forward paths (Plan 102: TileRT pipeline).
/// Different stages have different optimization opportunities:
/// - Draft: can skip screening, reduced KV writes, approximate attention
/// - Verify: exact attention, full KV write, enable screening
/// - Sample: SIMD-only, no attention needed
/// - BeliefDraft: MLP-only forward via BeliefDrafter, no attention needed (Plan 217)
#[cfg(feature = "decode_specialize")]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeStage {
    /// Batch-friendly, attention-heavy, needs full KV write.
    Prefill,
    /// Small batch, can skip screening, matmul-heavy.
    Draft,
    /// Single batch, needs exact attention, KV read-heavy.
    Verify,
    /// SIMD-only, no attention needed.
    Sample,
    /// MLP-only forward via BeliefDrafter — no attention, no KV (Plan 217).
    /// The belief drafter uses a lightweight MLP to predict next hidden states
    /// instead of running the full transformer forward pass.
    BeliefDraft,
}

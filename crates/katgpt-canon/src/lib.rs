//! # katgpt-canon — Canonical Intent Space
//!
//! Architecture-neutral intent direction space + per-model adapters.
//! Proposal 009 / Research 459.
//!
//! ## TL;DR
//!
//! A canonical intent is a unit-norm f32 direction in *canonical space*
//! (architecture-neutral). Each base model carries a [`ModelAdapter`] that
//! projects canonical directions into its model-specific latent space.
//! Plug any frozen base model into the system without retraining overlays.
//!
//! ```text
//!         ┌────────────────────────────────────────────────────────────┐
//!         │ Canonical Intent Space (architecture-neutral)              │
//!         │   d_Rust_idiom  = normalized direction in canonical space  │
//!         │   d_curiosity   = ...                                      │
//!         │   d_valence     = ...                                      │
//!         │   d_Rosetta_universal = joint subspace basis               │
//!         └─────────────────────────┬──────────────────────────────────┘
//!                                   │
//!                   ┌───────────────┼───────────────┐
//!                   ▼               ▼               ▼
//!          ProcrustesAdapter  SubspaceAdapter   MaskAdapter
//!          (same-arch)        (cross-arch)      (lottery ticket)
//!          R_Gemma            V_Gemma           m_Gemma
//!          R_Llama            V_Llama           m_Llama
//!          R_MiniCPM5         V_MiniCPM5        m_MiniCPM5
//!                   │               │               │
//!                   └───────────────┼───────────────┘
//!                                   ▼
//!                         model_specific_latent
//!                                   │
//!                                   ▼
//!                         frozen_base_model.decode()
//! ```
//!
//! ## What ships where
//!
//! | Feature | Adapter | Status |
//! |---|---|---|
//! | `canon` (P0) | [`CanonicalIntent`] + [`ModelAdapter`] + [`ProcrustesAdapter`] | opt-in |
//! | `canon_subspace` (P1) | [`SubspaceAdapter`] (cross-arch joint-SVD) | opt-in |
//! | `canon_mask` (P4) | [`MaskAdapter`] (lottery-ticket application) | opt-in |
//! | `canon_source_features` (Issue 867 P1+P2) | [`source_features`] AST histogram extractor + [`SourceFeatureAdapter`] ridge-fit steering adapter | opt-in |
//!
//! ## The P1 result (Bench 423, G5 GO)
//!
//! On Gemma2-2B (d=2304) ↔ MiniCPM5-1B (d=1536) with 40 Rust prompts,
//! the joint-SVD SubspaceAdapter preserves pairwise alignment on held-out
//! prompts at mean cos +0.87 (k=2) and +0.75 (k=4). The cross-arch
//! shared subspace is genuinely low-dimensional — this is a real
//! cross-model covariance result.
//!
//! ## The P3c verdict (Bench 426, modelless path exhausted)
//!
//! The cross-arch canonical DIRECTION claim (does a single direction in
//! the shared subspace discriminate Rust from non-Rust on BOTH models?)
//! FAILED: three converging failure lines (centroid agreement −0.33,
//! layer 0 discriminates best = vocabulary signal, length-detrending
//! reverses Python discrimination). Cross-arch Super-GOAT is demoted.
//! The intra-arch path ([`ProcrustesAdapter`] for same-dim model pairs)
//! is unaffected.
//!
//! ## The G1/G2/G4 GOAT stamp (Bench 562, 2026-07-28)
//!
//! The substrate's hot-path gates are measured and pass:
//!
//! | Adapter | G1 correctness | G2 perf (target 50µs) | G4 alloc-free |
//! |---|---|---|---|
//! | [`ProcrustesAdapter`] | ✓ (residual ≤ 1%, round-trip ≤ 1e-4) | ✓ at d=256 (29µs); d=2304 is 3.9ms diagnostic (O(d²)) | ✓ (0 allocs/1000 calls) |
//! | [`SubspaceAdapter`] | ✓ (heldout cosine > 0, frac pos ≥ 0.6) | ✓ at k=4, d=1536 (417ns) | ✓ (0 allocs/1000 calls) |
//! | [`MaskAdapter`] | ✓ (all-ones = identity, half-zero correct) | ✓ at d=2304 (1.38µs) | ✓ (0 allocs/1000 calls) |
//!
//! See [Bench 562](../../.benchmarks/562_katgpt_canon_goat.md) for the full
//! gate matrix + the ProcrustesAdapter d=2304 scaling limitation note.
//!
//! ## Layering note
//!
//! This crate lives at `crates/katgpt-canon/` (NOT `crates/katgpt-core/src/canon/`
//! as Proposal 009 originally specified) because the substrate split since
//! the proposal was written: `orthogonal_procrustes` lives in
//! [`katgpt_spectral`] (depends on katgpt-core) and `thin_svd_into` lives
//! in [`katgpt_core`]. A new crate that depends on both avoids the dep
//! cycle. Matches the Issue 007 crate-split pattern.

#![cfg_attr(not(feature = "canon"), no_std)]
// We need `alloc` for Vec in every adapter. katgpt-core already pulls alloc
// transitively; spell it out so this crate compiles under no_std + alloc.
// Issue 867 T2: `canon_source_features` joins the gate — the SourceFeatureAdapter
// owns weight Vecs while keeping the feature standalone (no implied `canon`).
#![cfg_attr(not(feature = "canon"), allow(dead_code, unused_imports))]

// blake3 is non-optional in Cargo.toml — it's used by every adapter for
// commitment. The feature gates below only gate the ADAPTER IMPLS, not
// blake3 itself.

#[cfg(any(feature = "canon", feature = "canon_source_features"))]
extern crate alloc;

/// Architecture-neutral intent direction + adapter trait.
#[cfg(feature = "canon")]
pub mod intent;
#[cfg(feature = "canon")]
pub use intent::{CanonicalIntent, ModelAdapter};

/// Orthogonal Procrustes rotation (same-arch).
#[cfg(feature = "canon")]
pub mod procrustes_adapter;
#[cfg(feature = "canon")]
pub use procrustes_adapter::ProcrustesAdapter;

/// Joint-SVD shared subspace (cross-arch).
#[cfg(feature = "canon_subspace")]
pub mod subspace_adapter;
#[cfg(feature = "canon_subspace")]
pub use subspace_adapter::{
    JointSvdFitScratch, SubspaceAdapter, SubspaceFit, fit_joint_svd_pair,
    fit_joint_svd_pair_with_cfg,
};

/// Lottery-ticket mask application (modelless apply; discovery in riir-train).
#[cfg(feature = "canon_mask")]
pub mod mask_adapter;
#[cfg(feature = "canon_mask")]
pub use mask_adapter::MaskAdapter;

/// Deterministic AST node-type histogram extractor (Issue 867 Phase 1 —
/// Proposal 010 §Feature 1). Architecture-independent source features; the
/// fixture corpus lives riir-train-side (`data/canon_rust_contrastive/`).
#[cfg(feature = "canon_source_features")]
pub mod source_features;
#[cfg(feature = "canon_source_features")]
pub use source_features::{AstBin, AstHistogram, N_AST_BINS, ast_histogram};

/// Issue 867 Phase 2 — `SourceFeatureAdapter`: the ridge-fit linear map from
/// the Phase 1 AST histogram into a model's latent steering space (Proposal
/// 010 §Feature 2). Setup-time f64 fit, zero-alloc f32 apply. Deliberately
/// NOT a `ModelAdapter` impl — the input space is source features, not
/// canonical intent (this is the construction side of Proposal 010).
#[cfg(feature = "canon_source_features")]
pub mod source_adapter;
#[cfg(feature = "canon_source_features")]
pub use source_adapter::{
    SourceAdapterFit, SourceFeatureAdapter, fit_linear_adapter, fit_source_adapter,
};

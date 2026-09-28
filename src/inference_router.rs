//! InferenceRouter — combines TriggerGate load monitoring with backend selection.
//!
//! Routes inference requests to the appropriate compute backend based on live
//! load metrics. At low load everything runs on CPU; as QPS increases the
//! TriggerGate promotes to GPU / ANE tiers. Tier-down releases accelerators
//! and returns to CPU-only.
//!
//! GPU and ANE backends are optional (`Option<Box<dyn InferenceBackend>>`).
//! When a backend is `None` the router falls back to CPU transparently.
//!
//! _Root-resident by design (Issue 033 §C, Option C)._ Depends on root-only
//! `katgpt_core::trigger_gate`, `katgpt_core::dllm_solver`, and `crate::pruners::acceptance_variance`
//! for dynamic tier routing.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::{Config, Rng};
use katgpt_backend::InferenceBackend;
use katgpt_core::trigger_gate::{ComputeTier, TriggerGate, TriggerGateConfig};

#[cfg(feature = "rv_gated_routing")]
use crate::pruners::acceptance_variance::AcceptanceVarianceTracker;

#[cfg(feature = "rv_gated_routing")]
use katgpt_core::trigger_gate::RvThresholds;

#[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
use katgpt_core::dllm_solver::{
    CriticalIntervalConfig, CriticalTierDecision, critical_tier_decision,
};

#[cfg(feature = "rcd_residual")]
use katgpt_core::dllm_solver::{ResidualMode, tier_to_residual_mode};

// Plan 267 — Thicket Variance Probe (TVP) decoding-space density signal.
// Composes with RV (Plan 202) for the G4 ablation gate: TVP+RV ≥ max(TVP, RV).
//
// The `TvpConfig` / `TvpSignal` / `TvpTierDecision` types are imported here so
// the inline `tier_after_tvp` block in `forward()` and the `router_tests`
// submodule (via `super::*`) can refer to them unqualified. The actual
// TVP-facing impl methods live in `router_tvp.rs` (Issue 018 split).
#[cfg(feature = "thicket_variance_probe")]
use crate::pruners::thicket_variance_probe::{TvpConfig, TvpSignal, TvpTierDecision};

// Plan 269 — CHIAR (Chiaroscuro Attention) router observation hook.
// Observation-only: exposes CHIAR KV strategy utilization and regime gate
// via RouterStats. Does NOT influence tier routing (CHIAR is per-token).
#[cfg(feature = "chiaroscuro")]
use katgpt_attn::chiaroscuro::{ChiarRouterHook, ChiarRouterStats};

#[cfg(feature = "modality_pruned_load")]
use katgpt_core::pipeline_pruner::QueryClassifier;

// ---------------------------------------------------------------------------
// Issue 018 — sibling-module split.
//
// Cohesive sub-systems that used to live inline here have been moved to
// sibling files to keep `inference_router.rs` under the 2048-line ceiling
// in the user's AGENTS.md Rust rules. Every moved item keeps its original
// `#[cfg(feature = ...)]` gate; the public API surface is preserved via
// `pub use` re-exports below so downstream callers (e.g.
// `examples/module_aware_routing.rs`) can still write
// `katgpt::inference_router::{ComputeTarget, route_by_module_energy, ...}`.
//
// Layout:
//   router_compute_target.rs — ComputeTarget + ModuleEnergyProfile +
//                              route_by_module_energy + their unit tests
//                              (Plan 264, gated on `module_energy_route`).
//   router_tvp.rs            — TVP-facing impl block on `InferenceRouter`
//                              (Plan 267, gated on `thicket_variance_probe`).
//   router_tests.rs          — integration/unit tests for the router
//                              (cfg(test) only).
// ---------------------------------------------------------------------------
#[cfg(feature = "module_energy_route")]
mod router_compute_target;
#[cfg(feature = "module_energy_route")]
pub use router_compute_target::{ComputeTarget, ModuleEnergyProfile, route_by_module_energy};

#[cfg(feature = "thicket_variance_probe")]
mod router_tvp;
// TVP types (`TvpSignal`, `TvpConfig`, `TvpTierDecision`) are NOT re-exported
// here — they remain accessible only at their canonical path
// `crate::pruners::thicket_variance_probe::*`, matching the pre-split
// behavior. The private `use` import at the top of this file still brings
// them into module scope so `router_tests` (via `super::*`) and the inline
// `tier_after_tvp` block in `forward()` can refer to them unqualified.

#[cfg(test)]
mod router_tests;

// ---------------------------------------------------------------------------
// RouterStats
// ---------------------------------------------------------------------------

/// Statistics snapshot from the router.
#[derive(Debug, Clone)]
pub struct RouterStats {
    /// Current compute tier.
    pub current_tier: ComputeTier,
    /// Total inferences routed since last reset.
    pub total_inferences: u64,
    /// Estimated QPS at time of snapshot.
    pub estimated_qps: f64,
    /// Name of the backend used for last forward pass.
    pub last_backend: &'static str,
    /// Number of tier transitions since creation.
    pub tier_transitions: u64,
    /// Current trust signal (0.0 = low trust, 1.0 = high trust).
    pub trust_signal: f32,
    /// Current RV signal (Plan 202). -1.0 if unavailable.
    #[cfg(feature = "rv_gated_routing")]
    pub rv_signal: f64,
    /// Current Lodestar completion distance (0 if never observed).
    #[cfg(feature = "lodestar")]
    pub lodestar_distance: u32,
    /// Current Lodestar budget remaining (-1 if unavailable).
    #[cfg(feature = "lodestar")]
    pub lodestar_budget_remaining: i32,
    /// Breakeven routing stats (Plan 250).
    #[cfg(feature = "breakeven_routing")]
    pub breakeven: katgpt_core::breakeven::BreakevenStats,
    /// Current TVP signal (Plan 267). `None` if `thicket_variance_probe` disabled.
    #[cfg(feature = "thicket_variance_probe")]
    pub tvp_signal: Option<TvpSignal>,
    /// CHIAR (Plan 269) router observation stats. `None` if `chiaroscuro` disabled
    /// or no keys observed yet.
    #[cfg(feature = "chiaroscuro")]
    pub chiar_stats: Option<ChiarRouterStats>,
}

// ---------------------------------------------------------------------------
// InferenceRouter
// ---------------------------------------------------------------------------

/// Router that combines [`TriggerGate`] load monitoring with backend selection.
///
/// At low load: routes everything to CPU.
/// As QPS increases: the [`TriggerGate`] promotes to GPU/ANE tiers.
/// On tier-up: attempts compilation; falls back to CPU on failure.
/// On tier-down: releases GPU/ANE, returns to CPU-only.
pub struct InferenceRouter {
    // NOTE: `gpu` and `gate` are `pub(crate)` so the TVP impl block in
    // `router_tvp.rs` (Issue 018 split) can read gate config / GPU presence
    // without exposing them on the public API.
    pub(crate) gpu: Option<Box<dyn InferenceBackend>>,
    ane: Option<Box<dyn InferenceBackend>>,
    pub(crate) gate: TriggerGate,
    config: Config,
    /// Monotonically increasing inference counter (atomic for borrow-checker compatibility).
    total_inferences: AtomicU64,
    /// Number of tier transitions since creation. Bounded by total_inferences,
    /// so u32 (4B cap) is more than enough — saves 4 bytes vs AtomicU64.
    tier_transitions: AtomicU32,
    last_backend: &'static str,
    /// Trust signal from speculative verification (0.0 = low trust, 1.0 = high trust).
    /// Updated externally via `update_trust()`. Influences tier transitions.
    trust_signal: f32,
    /// RV tracker for acceptance variance signal (Plan 202).
    /// `None` when `rv_gated_routing` feature is disabled → zero cost.
    #[cfg(feature = "rv_gated_routing")]
    rv_tracker: Option<AcceptanceVarianceTracker>,
    /// RV thresholds for tier promotion/demotion (Plan 202).
    #[cfg(feature = "rv_gated_routing")]
    rv_thresholds: RvThresholds,
    /// Last observed Lodestar completion distance d(root) (Plan 207).
    #[cfg(feature = "lodestar")]
    lodestar_distance: u32,
    /// Last observed Lodestar budget remaining (Plan 207).
    #[cfg(feature = "lodestar")]
    lodestar_budget_remaining: i32,
    /// Query classifier for modality-pruned pipeline selection (Plan 227 Phase 3).
    #[cfg(feature = "modality_pruned_load")]
    query_classifier: QueryClassifier,
    /// Critical interval config for entropy-triggered tier decisions (Plan 222 T15).
    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    critical_interval_config: CriticalIntervalConfig,
    /// Last observed critical interval entropy (Plan 222 T15).
    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    last_critical_entropy: f32,
    /// Breakeven bandit for cost-aware tier routing (Plan 250).
    #[cfg(feature = "breakeven_routing")]
    breakeven: katgpt_core::breakeven::BreakevenBandit,
    /// TVP signal (Plan 267) — decoding-space disagreement from K parallel probes.
    /// Starts as `None` (no probes run yet). Updated via `update_tvp()` after
    /// the probe-runner completes. When `None`, has zero routing impact.
    ///
    /// `pub(crate)` so the TVP impl block in `router_tvp.rs` (Issue 018 split)
    /// can read/mutate it without exposing it on the public API.
    #[cfg(feature = "thicket_variance_probe")]
    pub(crate) tvp_signal: Option<TvpSignal>,
    /// TVP config (Plan 267) — promote/demote thresholds + probe knobs.
    /// `pub(crate)` for the same reason as `tvp_signal` above.
    #[cfg(feature = "thicket_variance_probe")]
    pub(crate) tvp_config: TvpConfig,
    /// CHIAR observation hook (Plan 269 T15) — KV strategy utilization + regime gate.
    /// Observation-only; does NOT influence tier routing.
    #[cfg(feature = "chiaroscuro")]
    chiar_hook: ChiarRouterHook,
}

impl InferenceRouter {
    /// Create a new router.
    ///
    /// Starts at [`ComputeTier::CpuOnly`] with a `CpuBackend`.
    /// GPU backend is initialised if `gpu_available` and the `gpu_inference` feature
    /// is enabled with a Metal device present. ANE backend uses the same pattern.
    pub fn new(
        gate_config: TriggerGateConfig,
        model_config: Config,
        gpu_available: bool,
        ane_available: bool,
    ) -> Self {
        let gpu = if gpu_available {
            Self::try_create_gpu_backend()
        } else {
            None
        };
        let ane = if ane_available {
            Self::try_create_ane_backend()
        } else {
            None
        };

        Self {
            gpu,
            ane,
            gate: TriggerGate::new(gate_config, gpu_available, ane_available),
            config: model_config,
            total_inferences: AtomicU64::new(0),
            tier_transitions: AtomicU32::new(0),
            last_backend: "CPU",
            trust_signal: 1.0,
            #[cfg(feature = "rv_gated_routing")]
            rv_tracker: Some(AcceptanceVarianceTracker::new()),
            #[cfg(feature = "rv_gated_routing")]
            rv_thresholds: RvThresholds::default(),
            #[cfg(feature = "lodestar")]
            lodestar_distance: 0,
            #[cfg(feature = "lodestar")]
            lodestar_budget_remaining: -1,
            #[cfg(feature = "modality_pruned_load")]
            query_classifier: QueryClassifier::new(),
            #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
            critical_interval_config: CriticalIntervalConfig::default(),
            #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
            last_critical_entropy: 0.0,
            #[cfg(feature = "breakeven_routing")]
            breakeven: katgpt_core::breakeven::BreakevenBandit::with_defaults(),
            #[cfg(feature = "thicket_variance_probe")]
            tvp_signal: None,
            #[cfg(feature = "thicket_variance_probe")]
            tvp_config: TvpConfig::default(),
            #[cfg(feature = "chiaroscuro")]
            chiar_hook: ChiarRouterHook::new(),
        }
    }

    /// Try to create a GPU backend.
    #[cfg(all(target_os = "macos", feature = "gpu_inference"))]
    fn try_create_gpu_backend() -> Option<Box<dyn InferenceBackend>> {
        match crate::gpu_backend::GpuBackend::new() {
            Ok(backend) => {
                log::info!("InferenceRouter: GPU backend created (awaiting compile)");
                Some(Box::new(backend))
            }
            Err(e) => {
                log::info!("InferenceRouter: GPU unavailable ({e})");
                None
            }
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "gpu_inference")))]
    fn try_create_gpu_backend() -> Option<Box<dyn InferenceBackend>> {
        None
    }

    /// Try to create an ANE backend.
    #[cfg(all(target_os = "macos", feature = "ane"))]
    fn try_create_ane_backend() -> Option<Box<dyn InferenceBackend>> {
        log::info!("InferenceRouter: ANE backend created (awaiting compile)");
        Some(Box::new(crate::ane_backend::AneBackend::new()))
    }

    #[cfg(not(all(target_os = "macos", feature = "ane")))]
    fn try_create_ane_backend() -> Option<Box<dyn InferenceBackend>> {
        None
    }

    fn signal_recompile_for_tier(&mut self, tier: ComputeTier) {
        if matches!(tier, ComputeTier::CpuGpu | ComputeTier::CpuGpuAne)
            && let Some(ref mut gpu) = self.gpu
        {
            gpu.recompile_hint();
        }
        if matches!(tier, ComputeTier::CpuGpuAne)
            && let Some(ref mut ane) = self.ane
        {
            ane.recompile_hint();
        }
    }

    /// Run one forward pass, routing to the appropriate backend.
    ///
    /// Checks the [`TriggerGate`] for tier changes, selects the backend, and
    /// records inference timing for future load estimation.
    #[inline]
    pub fn forward<'a>(
        &mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
    ) -> &'a [f32] {
        // Evaluate tier change (returns Some(new_tier) if changed).
        if let Some(new_tier) = self.gate.evaluate() {
            log::info!("Router tier transition → {new_tier}");
            self.tier_transitions.fetch_add(1, Ordering::Relaxed);
            self.signal_recompile_for_tier(new_tier);
        }

        let start = Instant::now();

        // Snapshot the current tier before routing.
        let tier = self.gate.current_tier();

        // Trust-triggered tier adjustment (Plan 182)
        let tier_after_trust = if self.trust_signal < 0.4 && tier == ComputeTier::CpuOnly {
            // Low trust on CPU → tier up to GPU if available
            if self.gpu.is_some() {
                log::info!(
                    "Router trust-triggered tier-up: trust={:.2}, CPU→CPU+GPU",
                    self.trust_signal
                );
                ComputeTier::CpuGpu
            } else {
                tier
            }
        } else if self.trust_signal > 0.8 && tier == ComputeTier::CpuGpu {
            // High trust on GPU → allow tier down to CPU.
            // Snapshot gate config once to avoid repeated method calls.
            let cfg = self.gate.config();
            let low_load = self.gate.estimated_qps() < cfg.gpu_activate_qps * cfg.hysteresis_factor;
            if low_load {
                log::info!(
                    "Router trust-triggered tier-down: trust={:.2}, CPU+GPU→CPU",
                    self.trust_signal
                );
                ComputeTier::CpuOnly
            } else {
                tier
            }
        } else {
            tier
        };

        // RV-gated tier adjustment (Plan 202)
        // Overrides trust/QPS routing when RV signal is available.
        #[cfg(feature = "rv_gated_routing")]
        let tier_after_rv = {
            let rv_signal = self.rv_tracker.as_ref().map_or(-1.0, |t| t.rv());
            match self.gate.rv_tier_boost(rv_signal, &self.rv_thresholds) {
                Some(rv_tier) => {
                    if rv_tier != tier_after_trust {
                        log::info!(
                            "Router RV-gated tier override: rv={rv_signal:.4}, {tier_after_trust}→{rv_tier}"
                        );
                    }
                    rv_tier
                }
                None => tier_after_trust,
            }
        };
        #[cfg(not(feature = "rv_gated_routing"))]
        let tier_after_rv = tier_after_trust;

        // Critical-interval tier adjustment (Plan 222 T15)
        // Entropy-triggered override: promote to GPU for q-sample refinement
        // when marginals are multimodal AND load is low.
        #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
        let tier_after_critical = match self.observe_critical_entropy(self.last_critical_entropy) {
            CriticalTierDecision::PromoteGpu if tier_after_rv == ComputeTier::CpuOnly => {
                log::info!(
                    "Router critical-interval override: {tier_after_rv}→CpuGpu (entropy={:.4})",
                    self.last_critical_entropy
                );
                ComputeTier::CpuGpu
            }
            _ => tier_after_rv,
        };
        #[cfg(not(all(feature = "critical_interval_gate", feature = "rv_gated_routing")))]
        let tier_after_critical = tier_after_rv;

        // TVP tier adjustment (Plan 267 T10) — decoding-space disagreement.
        // Sits AFTER critical-interval (entropy-driven) and BEFORE breakeven
        // (cost-amortization veto). The decision logic is extracted into
        // [`crate::pruners::thicket_variance_probe::tvp_tier_decision`] so it
        // can be unit-tested without running the full forward pass.
        //
        // Format-only disagreement (TvpSignal.format_disagreement) is intentionally
        // routed to canonicalization, NOT to compute promotion — see G5.
        #[cfg(feature = "thicket_variance_probe")]
        let tier_after_tvp = match self.observe_tvp_decision(tier_after_critical) {
            TvpTierDecision::PromoteGpu => ComputeTier::CpuGpu,
            TvpTierDecision::DemoteCpu => ComputeTier::CpuOnly,
            _ => tier_after_critical,
        };
        #[cfg(not(feature = "thicket_variance_probe"))]
        let tier_after_tvp = tier_after_critical;

        // Breakeven tier adjustment (Plan 250)
        // Cost-aware override: promote when tier upgrade has amortized, defer when not.
        #[cfg(feature = "breakeven_routing")]
        let tier_final = match self.breakeven.select_tier(tier_after_tvp) {
            Some(breakeven_tier) if breakeven_tier != tier_after_tvp => {
                log::info!("Router breakeven tier override: {tier_after_tvp}→{breakeven_tier}");
                breakeven_tier
            }
            _ => tier_after_tvp,
        };
        #[cfg(not(feature = "breakeven_routing"))]
        let tier_final = tier_after_tvp;

        // Route to the appropriate backend.
        //
        // We populate ctx.logits via forward(), then return a borrow of ctx.logits
        // (not from self) to satisfy the lifetime constraint that the returned slice
        // borrows from `ctx`.
        //
        // CpuGpu and CpuGpuAne both route through dispatch_gpu_or_cpu: the ANE
        // compile path is not yet implemented, so ANE falls back to GPU dispatch.
        let backend_name = match tier_final {
            ComputeTier::CpuOnly => {
                crate::transformer::forward(ctx, weights, cache, token, pos, &self.config);
                "CPU"
            }
            ComputeTier::CpuGpu | ComputeTier::CpuGpuAne => {
                self.dispatch_gpu_or_cpu(ctx, weights, cache, token, pos)
            }
        };

        // Record timing using atomics (no mutable borrow of self needed).
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.gate.record_inference(elapsed_us);
        self.total_inferences.fetch_add(1, Ordering::Relaxed);
        self.last_backend = backend_name;

        // Feed timing into breakeven bandit (Plan 250).
        #[cfg(feature = "breakeven_routing")]
        {
            use katgpt_core::breakeven::BreakevenTierPair;
            match tier_final {
                ComputeTier::CpuOnly => {
                    // CPU is the baseline for CpuToGpu pair.
                    self.breakeven
                        .observe_baseline(BreakevenTierPair::CpuToGpu, elapsed_us);
                }
                ComputeTier::CpuGpu => {
                    // GPU is the upgraded tier for CpuToGpu pair.
                    self.breakeven
                        .observe_tier(BreakevenTierPair::CpuToGpu, elapsed_us);
                }
                ComputeTier::CpuGpuAne => {
                    // ANE is the upgraded tier for GpuToAne pair.
                    self.breakeven
                        .observe_tier(BreakevenTierPair::GpuToAne, elapsed_us);
                }
            }
        }

        // Return logits borrowed from ctx (not from self).
        &ctx.logits[..self.config.vocab_size]
    }

    /// Update trust signal from verifier (called after each speculative decode).
    #[inline]
    pub fn update_trust(&mut self, trust: f32) {
        self.trust_signal = trust;
    }

    /// Get current trust signal.
    #[inline]
    pub fn trust_signal(&self) -> f32 {
        self.trust_signal
    }

    // ── RV-Gated Compute Routing API (Plan 202) ───────────────────

    /// Observe an acceptance event for RV tracking.
    ///
    /// Call after each speculative decode verification.
    /// No-op when `rv_gated_routing` is disabled.
    #[cfg(feature = "rv_gated_routing")]
    #[inline]
    pub fn observe_acceptance(&mut self, accepted: bool) {
        if let Some(ref mut tracker) = self.rv_tracker {
            tracker.observe(accepted);
        }
    }

    /// Get current RV signal. Returns -1.0 if tracking is unavailable.
    ///
    /// RV ∈ [0.0, 0.25] for Bernoulli acceptance data.
    /// 0.0 = all accept/reject (confident). 0.25 = 50/50 (uncertain).
    #[cfg(feature = "rv_gated_routing")]
    #[inline]
    pub fn rv_signal(&self) -> f64 {
        self.rv_tracker.as_ref().map_or(-1.0, |t| t.rv())
    }

    /// Reset the RV tracker (call at query boundaries).
    /// No-op when `rv_gated_routing` is disabled.
    #[cfg(feature = "rv_gated_routing")]
    #[inline]
    pub fn reset_rv(&mut self) {
        if let Some(ref mut tracker) = self.rv_tracker {
            tracker.reset();
        }
    }

    /// Update RV thresholds at runtime.
    #[cfg(feature = "rv_gated_routing")]
    pub fn set_rv_thresholds(&mut self, thresholds: RvThresholds) {
        self.rv_thresholds = thresholds;
    }

    /// Get current RV thresholds.
    #[cfg(feature = "rv_gated_routing")]
    pub fn rv_thresholds(&self) -> &RvThresholds {
        &self.rv_thresholds
    }

    // ── Critical Interval Tier Routing (Plan 222 T15) ────────────

    /// Observe entropy from DDTree build and decide whether to override tier.
    ///
    /// Call during each DDTree depth with the Shannon entropy of marginals.
    /// Returns the tier decision:
    /// - `Defer` — no override, use current routing
    /// - `PromoteGpu` — critical interval + low load, promote to GPU
    /// - `StayCpu` — critical interval + high load, stay on CPU with fast solver
    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    #[inline]
    pub fn observe_critical_entropy(&mut self, entropy: f32) -> CriticalTierDecision {
        self.last_critical_entropy = entropy;
        let current_tier = self.gate.current_tier();
        let gpu_available = self.gate.gpu_available();
        let decision = critical_tier_decision(
            entropy,
            &self.critical_interval_config,
            current_tier,
            gpu_available,
        );
        if !matches!(decision, CriticalTierDecision::Defer) {
            log::info!(
                "Router critical-interval tier: entropy={entropy:.4}, decision={decision:?}"
            );
        }
        decision
    }

    /// Update CriticalInterval config at runtime.
    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    pub fn set_critical_interval_config(&mut self, config: CriticalIntervalConfig) {
        self.critical_interval_config = config;
    }

    /// Get current CriticalInterval config.
    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    pub fn critical_interval_config(&self) -> &CriticalIntervalConfig {
        &self.critical_interval_config
    }

    /// Get last observed critical interval entropy.
    #[cfg(all(feature = "critical_interval_gate", feature = "rv_gated_routing"))]
    pub fn last_critical_entropy(&self) -> f32 {
        self.last_critical_entropy
    }

    // ── Lodestar Distance/Budget Routing Hook (Plan 207) ────────────

    /// Observe Lodestar distance and budget for routing decisions.
    ///
    /// Call after each tree build with the root distance and remaining budget.
    /// High d + tight budget → prefer CPU for deterministic guarantee.
    #[cfg(feature = "lodestar")]
    #[inline]
    pub fn observe_lodestar(&mut self, d_root: u32, budget_remaining: usize) {
        self.lodestar_distance = d_root;
        self.lodestar_budget_remaining = budget_remaining as i32;
    }

    /// Get current Lodestar distance (0 if never observed).
    #[cfg(feature = "lodestar")]
    #[inline]
    pub fn lodestar_distance(&self) -> u32 {
        self.lodestar_distance
    }

    /// Get current Lodestar budget remaining (-1 if never observed).
    #[cfg(feature = "lodestar")]
    #[inline]
    pub fn lodestar_budget_remaining(&self) -> i32 {
        self.lodestar_budget_remaining
    }

    /// Whether Lodestar suggests CPU fallback.
    ///
    /// Returns `true` when completion is far and budget is tight:
    /// `d_root > 4 && budget_remaining < d_root * 2`
    ///
    /// This means: we're far from done AND we don't have 2× the minimum
    /// budget needed — so the deterministic CPU path is safer.
    #[cfg(feature = "lodestar")]
    #[inline]
    pub fn lodestar_suggests_cpu(&self) -> bool {
        if self.lodestar_budget_remaining < 0 {
            return false;
        }
        let d = self.lodestar_distance;
        let br = self.lodestar_budget_remaining;
        d > 4 && br < (d as i32 * 2)
    }

    /// Reset Lodestar state (call at query boundaries).
    #[cfg(feature = "lodestar")]
    #[inline]
    pub fn reset_lodestar(&mut self) {
        self.lodestar_distance = 0;
        self.lodestar_budget_remaining = -1;
    }

    /// Signal that weights have changed; GPU/ANE backends should recompile.
    pub fn update_weights(&mut self, _weights: &TransformerWeights) {
        if let Some(ref mut gpu) = self.gpu {
            gpu.recompile_hint();
        }
        if let Some(ref mut ane) = self.ane {
            ane.recompile_hint();
        }
    }

    /// Try GPU forward, fall back to CPU. Returns the backend name used.
    ///
    /// Central routing point for GPU dispatch:
    /// 1. Auto-compiles weights on first use (lazy compile)
    /// 2. Dispatches to GPU if compiled, else falls back to CPU
    #[inline]
    fn dispatch_gpu_or_cpu(
        &mut self,
        ctx: &mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
    ) -> &'static str {
        if let Some(ref mut gpu) = self.gpu {
            // Single is_compiled() probe: if not yet compiled, attempt compile
            // once and capture readiness, avoiding a redundant probe afterwards.
            let ready = if gpu.is_compiled() {
                true
            } else {
                match gpu.compile(weights, &self.config) {
                    Ok(()) => {
                        log::info!("TriggerGate: CPU → CPU+GPU (compiled)");
                        true
                    }
                    Err(e) => {
                        log::info!("Router: GPU compile failed ({e}), falling back to CPU");
                        false
                    }
                }
            };
            if ready {
                gpu.forward(ctx, weights, cache, token, pos, &self.config);
                return "GPU";
            }
        }
        crate::transformer::forward(ctx, weights, cache, token, pos, &self.config);
        "CPU"
    }

    /// Return a snapshot of router statistics.
    pub fn stats(&self) -> RouterStats {
        RouterStats {
            current_tier: self.gate.current_tier(),
            total_inferences: self.total_inferences.load(Ordering::Relaxed),
            estimated_qps: self.gate.estimated_qps(),
            last_backend: self.last_backend,
            tier_transitions: self.tier_transitions.load(Ordering::Relaxed) as u64,
            trust_signal: self.trust_signal,
            #[cfg(feature = "rv_gated_routing")]
            rv_signal: self.rv_signal(),
            #[cfg(feature = "lodestar")]
            lodestar_distance: self.lodestar_distance,
            #[cfg(feature = "lodestar")]
            lodestar_budget_remaining: self.lodestar_budget_remaining,
            #[cfg(feature = "breakeven_routing")]
            breakeven: self.breakeven.stats(),
            #[cfg(feature = "thicket_variance_probe")]
            tvp_signal: self.tvp_signal,
            #[cfg(feature = "chiaroscuro")]
            chiar_stats: {
                let s = self.chiar_hook.stats();
                if s.tokens_observed > 0 { Some(s) } else { None }
            },
        }
    }

    /// Run a batch of forward passes, amortizing tier evaluation across all items.
    ///
    /// For GPU/ANE backends, batch mode allows a single kernel dispatch for multiple
    /// tokens, reducing per-inference overhead. On CPU, this is equivalent to calling
    /// `forward()` in a loop but with a single tier evaluation.
    ///
    /// Returns a flat buffer of logits with `vocab_size` stride per token.
    /// Unlike `forward()`, this returns owned `Vec<f32>` because the borrow checker
    /// doesn't allow holding multiple mutable borrows of `ctx` across loop iterations.
    ///
    /// Layout: `[token0_logits, token1_logits, ...]` where each segment is `config.vocab_size` elements.
    #[inline]
    pub fn forward_batch(
        &mut self,
        ctx: &mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        tokens: &[(usize, usize)],
    ) -> Vec<f32> {
        if tokens.is_empty() {
            return Vec::new();
        }

        // Single tier evaluation for the entire batch.
        if let Some(new_tier) = self.gate.evaluate() {
            log::info!("Router batch tier transition → {new_tier}");
            self.tier_transitions.fetch_add(1, Ordering::Relaxed);
            self.signal_recompile_for_tier(new_tier);
        }

        let tier = self.gate.current_tier();
        let config = &self.config;
        let vocab_size = config.vocab_size;
        let batch_len = tokens.len();
        let mut flat = Vec::with_capacity(batch_len * vocab_size);

        match tier {
            ComputeTier::CpuOnly | ComputeTier::CpuGpu | ComputeTier::CpuGpuAne => {
                // CPU path: iterate through tokens sequentially.
                // GPU/ANE TODO: when backends exist, dispatch entire batch at once.
                let batch_start = Instant::now();
                for &(token, pos) in tokens {
                    let logits =
                        crate::transformer::forward(ctx, weights, cache, token, pos, config);
                    flat.extend_from_slice(&logits[..vocab_size]);
                }
                let elapsed_us = batch_start.elapsed().as_micros() as u64;
                // Record total batch time as a single inference for QPS estimation.
                self.gate.record_inference(elapsed_us);
            }
        }

        self.total_inferences
            .fetch_add(batch_len as u64, Ordering::Relaxed);
        self.last_backend = "CPU";

        flat
    }

    /// Borrow the inner [`TriggerGate`].
    pub fn gate(&self) -> &TriggerGate {
        &self.gate
    }

    /// Delegate queue-depth recording to the gate.
    #[inline]
    pub fn record_queue_depth(&self, depth: usize) {
        self.gate.record_queue_depth(depth);
    }

    /// Borrow the breakeven bandit (Plan 250).
    #[cfg(feature = "breakeven_routing")]
    pub fn breakeven(&self) -> &katgpt_core::breakeven::BreakevenBandit {
        &self.breakeven
    }

    // NOTE: TVP API (`update_tvp`, `tvp_signal`, `set_tvp_config`, `tvp_config`,
    // `observe_tvp_decision`) moved to `router_tvp.rs` (Issue 018 split).

    // ── CHIAR Observation API (Plan 269 T15) ────────────────────

    /// Observe a key embedding for CHIAR KV strategy classification (Plan 269).
    ///
    /// Updates the τ calibrator and dispatches the key to a storage strategy
    /// (DctTruncated / Quantized / FullPrecision). Call this for each key
    /// entering the KV cache when the `chiaroscuro` feature is enabled.
    ///
    /// Observation-only — does NOT influence tier routing.
    #[cfg(feature = "chiaroscuro")]
    #[inline]
    pub fn observe_chiar_key(&mut self, key: &[f32]) {
        self.chiar_hook.observe_key(key);
    }

    /// Observe a prompt token's spectral entropy for CHIAR regime classification (Plan 269).
    ///
    /// Updates the Welford variance tracker inside the regime gate.
    /// Observation-only — does NOT influence tier routing.
    #[cfg(feature = "chiaroscuro")]
    #[inline]
    pub fn observe_chiar_prompt_token(&mut self, h: f32) {
        self.chiar_hook.observe_prompt_token(h);
    }

    /// Get the current CHIAR router stats snapshot (Plan 269).
    /// Returns `None` if no keys have been observed yet.
    #[cfg(feature = "chiaroscuro")]
    pub fn chiar_stats(&self) -> Option<ChiarRouterStats> {
        let s = self.chiar_hook.stats();
        if s.tokens_observed > 0 { Some(s) } else { None }
    }

    /// Classify a query and select the optimal pipeline configuration (Plan 227 Phase 3).
    /// Only available when `modality_pruned_load` feature is enabled.
    #[cfg(feature = "modality_pruned_load")]
    #[inline]
    pub fn select_pipeline(&self, prompt: &str) -> katgpt_core::pipeline_pruner::PipelineConfig {
        self.query_classifier.classify_prompt(prompt)
    }

    /// Get the current residual mode based on the active compute tier (Plan 258).
    ///
    /// Plasma path returns `Skip` for zero overhead.
    /// Higher tiers return progressively more expensive residual modes.
    #[cfg(feature = "rcd_residual")]
    #[inline]
    pub fn residual_mode(&self) -> ResidualMode {
        tier_to_residual_mode(self.gate.current_tier())
    }

    /// Generate tokens autoregressively using the routed forward path.
    ///
    /// Mirrors [`crate::transformer::generate_into`] but routes each forward pass
    /// through [`Self::forward`], recording queue depth for load estimation.
    pub fn generate_routed(
        &mut self,
        ctx: &mut ForwardContext,
        cache: &mut MultiLayerKVCache,
        weights: &TransformerWeights,
        rng: &mut Rng,
        max_tokens: usize,
        tokens: &mut Vec<usize>,
    ) {
        tokens.clear();
        let mut token = self.config.bos_token;
        let mut pos = 0;

        for _ in 0..max_tokens {
            if pos >= self.config.block_size {
                cache.reset();
                pos = 0;
                token = self.config.bos_token;
            }

            self.record_queue_depth(1);
            self.forward(ctx, weights, cache, token, pos);

            let next_token = ctx.sample_next_token(self.config.temperature, rng);
            tokens.push(next_token);

            if next_token == self.config.bos_token {
                cache.reset();
                pos = 0;
                token = self.config.bos_token;
            } else {
                token = next_token;
                pos += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SpeculativeGenerator routing (Plan 193 T13)
// ---------------------------------------------------------------------------

#[cfg(feature = "speculative_generator")]
use katgpt_core::{GenerativeConstraintPruner, SpeculativeGenerator};

#[cfg(feature = "speculative_generator")]
impl InferenceRouter {
    /// Generate candidates via any [`SpeculativeGenerator`] and validate with
    /// a [`GenerativeConstraintPruner`].
    ///
    /// For token generators: routes to GPU/ANE when load is high.
    /// For action generators: always CPU (WASM validation is CPU-bound).
    ///
    /// Returns validated candidates only (invalid ones pruned).
    pub fn generate_validated<G, P>(
        &mut self,
        generator: &mut G,
        pruner: &P,
        condition: &G::Condition,
        rng: &mut fastrand::Rng,
    ) -> Vec<G::Output>
    where
        G: SpeculativeGenerator,
        P: GenerativeConstraintPruner<G::Output>,
    {
        let candidates = generator.generate(condition, rng).unwrap_or_default();
        let validity = pruner.batch_is_valid(&candidates);
        // Pre-allocate with exact valid count to avoid incremental Vec growth
        // during collect (filter_map yields unknown lower-bound size hint).
        let valid_count = validity.iter().filter(|&&v| v).count();
        let mut result = Vec::with_capacity(valid_count);
        for (c, v) in candidates.into_iter().zip(validity) {
            if v {
                result.push(c);
            }
        }
        result.shrink_to_fit();
        result
    }
}

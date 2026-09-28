//! Proposal 011 Phase 5 follow-up — value-sensitive encoder probe.
//!
//! **The question this bench answers:** Bench 015 showed the
//! `GeometrySummaryEncoder` (length + curvature + cosine + n_steps) cannot
//! discriminate perturbed vs original Kimi-K3 weights — accuracy stays at
//! ~50% (coin flip) even at σ=0.5 (50% relative noise). The root cause was
//! identified: the geometry encoder captures SHAPE features that are
//! invariant to value perturbation but sensitive to structural change.
//!
//! This bench tests whether a **value-sensitive encoder** — one that captures
//! per-layer displacement statistics rather than aggregate trajectory shape —
//! can discriminate perturbed vs original weights.
//!
//! # Key architectural insight
//!
//! The depth trajectory captures the RAW residual stream (prefix_sum is never
//! normalized inside the decoder layer — RMSNorm is applied to scratch_hidden,
//! not prefix_sum). So the displacement `h_{l+1} - h_l = attn_out + ffn_out`
//! IS the raw per-layer delta, directly computed from the layer's weights.
//!
//! The geometry encoder collapses 8 displacement vectors into 4 scalar
//! features (length = sum of norms, etc.), losing most per-layer detail.
//! A value-sensitive encoder preserving per-displacement information should
//! be more discriminative — IF the per-layer deltas change differently
//! under perturbation (which they should, since layers have different
//! architectures: KDA vs MLA, Dense vs MoE).
//!
//! # Encoders tested
//!
//! 1. **Geometry** (baseline) — same `GeometrySummaryEncoder` as bench_015.
//! 2. **DispNorms** — per-displacement L2 norms (8 features, replicated to D=32).
//! 3. **DispStats** — per-displacement [L2, mean, var, max] (8×4 = 32 features).
//! 4. **StateNorms** — per-state L2 norms (9 features, replicated to D=32).
//! 5. **DispRatios** — per-displacement L2 / total L2 (8 features, replicated).
//!
//! # Run
//!
//! ```bash
//! cargo bench --features "kimi_k3_loader swe_trajectory_freeze" \
//!     --bench bench_016_value_sensitive_encoder -- --nocapture
//! ```

#![cfg(all(feature = "kimi_k3_loader", feature = "swe_trajectory_freeze"))]
#![allow(clippy::needless_range_loop)]

use katgpt_attn::gdn2::kda_forward::KdaWeights;
use katgpt_attn::mla::MlaWeights;
use katgpt_core::latent_trajectory_geometry::from_states_into;
use katgpt_core::swe_trajectory_freeze::GeometrySummaryEncoder;
use katgpt_rs::kimi_k3::decoder_layer::{
    KimiAttentionWeights, KimiDecoderLayerWeights, KimiFfnWeights,
};
use katgpt_rs::kimi_k3::loader::KimiK3ModelWeights;
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token_traced};
use katgpt_transformer::attn_res::AttnResWeights;
use katgpt_transformer::moe::{MoeWeights, SwiGluExpertWeights};

// ─── Constants ─────────────────────────────────────────────────────────────

/// Summary dimension D (matches bench_014/015).
const D: usize = 32;

/// Number of archetype modes (Model A vs Model B(σ)).
const N: usize = 2;

/// Total tokens to extract trajectories for.
const N_TOKENS: usize = 32;

/// Training split size (per model).
const N_TRAIN: usize = 12;

/// Truncated vocab.
const BENCH_VOCAB: usize = 512;

/// Perturbation σ levels.
const SIGMA_LEVELS: &[f32] = &[0.0, 0.001, 0.01, 0.05, 0.1, 0.5];

// ─── Deterministic LCG (copied from bench_015) ─────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

// ─── Weight perturbation (copied from bench_015) ───────────────────────────

#[inline]
fn perturb_vec(v: &mut [f32], rng: &mut Lcg, sigma: f32) {
    if sigma == 0.0 {
        return;
    }
    for w in v.iter_mut() {
        let noise = rng.next_f32();
        *w *= 1.0 + sigma * noise;
    }
}

fn perturb_attn_res(w: &mut AttnResWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.norm_weight, rng, sigma);
    perturb_vec(&mut w.proj_weight, rng, sigma);
}

fn perturb_mla(w: &mut MlaWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.w_dkv, rng, sigma);
    perturb_vec(&mut w.w_dq, rng, sigma);
    perturb_vec(&mut w.w_uq, rng, sigma);
    perturb_vec(&mut w.w_qr, rng, sigma);
    perturb_vec(&mut w.w_uk, rng, sigma);
    perturb_vec(&mut w.w_uv, rng, sigma);
    perturb_vec(&mut w.w_kr, rng, sigma);
    perturb_vec(&mut w.w_o, rng, sigma);
    perturb_vec(&mut w.q_a_norm_weight, rng, sigma);
    perturb_vec(&mut w.kv_a_norm_weight, rng, sigma);
    if let Some(w_g) = w.w_g.as_mut() {
        perturb_vec(w_g, rng, sigma);
    }
}

fn perturb_kda(w: &mut KdaWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.q_proj, rng, sigma);
    perturb_vec(&mut w.k_proj, rng, sigma);
    perturb_vec(&mut w.v_proj, rng, sigma);
    perturb_vec(&mut w.q_conv_weight, rng, sigma);
    perturb_vec(&mut w.k_conv_weight, rng, sigma);
    perturb_vec(&mut w.v_conv_weight, rng, sigma);
    perturb_vec(&mut w.a_log, rng, sigma);
    perturb_vec(&mut w.f_a_proj, rng, sigma);
    perturb_vec(&mut w.f_b_proj, rng, sigma);
    perturb_vec(&mut w.dt_bias, rng, sigma);
    perturb_vec(&mut w.beta_proj, rng, sigma);
    perturb_vec(&mut w.g_proj, rng, sigma);
    perturb_vec(&mut w.o_norm_weight, rng, sigma);
    perturb_vec(&mut w.o_proj, rng, sigma);
}

fn perturb_swiglu(w: &mut SwiGluExpertWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.gate_proj, rng, sigma);
    perturb_vec(&mut w.up_proj, rng, sigma);
    perturb_vec(&mut w.down_proj, rng, sigma);
}

fn perturb_moe(w: &mut MoeWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.router_weight, rng, sigma);
    perturb_vec(&mut w.e_score_correction_bias, rng, sigma);
    for expert in w.experts.iter_mut() {
        perturb_swiglu(expert, rng, sigma);
    }
    for expert in w.shared_experts.iter_mut() {
        perturb_swiglu(expert, rng, sigma);
    }
    if let Some(p) = w.routed_expert_down_proj.as_mut() {
        perturb_vec(p, rng, sigma);
    }
    if let Some(p) = w.routed_expert_up_proj.as_mut() {
        perturb_vec(p, rng, sigma);
    }
    if let Some(p) = w.routed_expert_norm_weight.as_mut() {
        perturb_vec(p, rng, sigma);
    }
}

fn perturb_layer(w: &mut KimiDecoderLayerWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.input_layernorm_weight, rng, sigma);
    perturb_vec(&mut w.post_attention_layernorm_weight, rng, sigma);
    match &mut w.attention {
        KimiAttentionWeights::Mla(m) => perturb_mla(m, rng, sigma),
        KimiAttentionWeights::Kda(k) => perturb_kda(k, rng, sigma),
    }
    match &mut w.ffn {
        KimiFfnWeights::Dense(s) => perturb_swiglu(s, rng, sigma),
        KimiFfnWeights::Moe(m) => perturb_moe(m, rng, sigma),
    }
    perturb_attn_res(&mut w.self_attn_res, rng, sigma);
    perturb_attn_res(&mut w.mlp_attn_res, rng, sigma);
}

fn perturb_model(w: &mut KimiK3ModelWeights, sigma: f32) {
    let seed = (sigma * 1_000_000.0) as u64 | 0xA15E_0000;
    let mut rng = Lcg::new(seed);
    perturb_vec(&mut w.embed_weight, &mut rng, sigma);
    for layer in w.layers.iter_mut() {
        perturb_layer(layer, &mut rng, sigma);
    }
    perturb_vec(&mut w.final_norm_weight, &mut rng, sigma);
    if !w.lm_head_weight.is_empty() {
        perturb_vec(&mut w.lm_head_weight, &mut rng, sigma);
    }
    perturb_attn_res(&mut w.output_attn_res, &mut rng, sigma);
}

// ─── Trajectory extraction ─────────────────────────────────────────────────

struct ExtractScratch {
    disp_curr: Vec<f32>,
    disp_prev: Vec<f32>,
    traj_buf: Vec<Vec<f32>>,
}

impl ExtractScratch {
    fn new(hidden_dim: usize) -> Self {
        Self {
            disp_curr: Vec::with_capacity(hidden_dim),
            disp_prev: Vec::with_capacity(hidden_dim),
            traj_buf: Vec::with_capacity(9),
        }
    }

    /// Run the traced forward pass; leaves trajectory in `traj_buf`.
    fn extract_traj(
        &mut self,
        config: &KimiK3ModelConfig,
        weights: &KimiK3ModelWeights,
        runtime: &mut KimiK3Runtime,
        token_id: u32,
    ) {
        runtime.reset();
        self.traj_buf.clear();
        let _ =
            kimi_k3_forward_token_traced(config, weights, runtime, token_id, &mut self.traj_buf);
    }
}

// ─── Value-sensitive encoders ──────────────────────────────────────────────
//
// Each encoder writes a fixed-D summary from the raw trajectory states.
// The classification pipeline (derive_directions + nearest-centroid) is
// encoder-agnostic — it only sees the D-dim summary vectors.

/// Baseline: the shipped GeometrySummaryEncoder (same as bench_015).
fn encode_geometry(
    states: &[&[f32]],
    scratch: &mut ExtractScratch,
    encoder: &GeometrySummaryEncoder,
    out: &mut [f32; D],
) {
    let geom = from_states_into(states, &mut scratch.disp_curr, &mut scratch.disp_prev);
    encoder.encode_into(&geom, out);
}

/// Per-displacement L2 norms. Captures the magnitude of each layer's
/// (attn_out + ffn_out) delta — the most direct weight-dependent signal.
/// 8 features (one per displacement step), replicated to fill D=32.
fn encode_disp_norms(states: &[&[f32]], out: &mut [f32; D]) {
    let n_disps = states.len().saturating_sub(1);
    let mut norms = [0.0_f32; 8]; // max 8 displacements (9 states)

    for l in 0..n_disps.min(8) {
        let mut sum_sq = 0.0_f32;
        for i in 0..states[l].len() {
            let diff = states[l + 1][i] - states[l][i];
            sum_sq += diff * diff;
        }
        norms[l] = sum_sq.sqrt();
    }

    // Replicate the 8 values across D=32 (4 copies).
    let n_blocks = D / 8;
    for block in out.chunks_mut(8).take(n_blocks) {
        block.copy_from_slice(&norms);
    }
    // Zero remaining slots.
    for j in (n_blocks * 8)..D {
        out[j] = 0.0;
    }
}

/// Per-displacement full statistics: [L2, mean, variance, max_abs] per layer.
/// 8 layers × 4 stats = 32 features = D exactly.
fn encode_disp_stats(states: &[&[f32]], out: &mut [f32; D]) {
    let n_disps = states.len().saturating_sub(1);

    for l in 0..n_disps.min(8) {
        let dim = states[l].len();
        let mut sum_sq = 0.0_f32;
        let mut sum = 0.0_f32;
        let mut max_abs = 0.0_f32;

        for i in 0..dim {
            let diff = states[l + 1][i] - states[l][i];
            sum_sq += diff * diff;
            sum += diff;
            let abs_diff = diff.abs();
            if abs_diff > max_abs {
                max_abs = abs_diff;
            }
        }

        let l2 = sum_sq.sqrt();
        let mean = sum / dim as f32;
        // Variance = E[x^2] - E[x]^2
        let var = (sum_sq / dim as f32) - (mean * mean);

        let base = l * 4;
        if base + 3 < D {
            out[base] = l2;
            out[base + 1] = mean;
            out[base + 2] = var;
            out[base + 3] = max_abs;
        }
    }

    // Zero any remaining slots.
    let used = (n_disps.min(8)) * 4;
    for j in used..D {
        out[j] = 0.0;
    }
}

/// Per-state L2 norms. Captures the growth profile of the accumulated
/// residual stream. 9 features (embed + 8 layers), replicated to fill D=32.
fn encode_state_norms(states: &[&[f32]], out: &mut [f32; D]) {
    let n_states = states.len().min(9);
    let mut norms = [0.0_f32; 9];

    for l in 0..n_states {
        let mut sum_sq = 0.0_f32;
        for i in 0..states[l].len() {
            sum_sq += states[l][i] * states[l][i];
        }
        norms[l] = sum_sq.sqrt();
    }

    // Replicate across D=32: floor(32/9) = 3 full copies + 5 trailing.
    let n_blocks = D / 9;
    for block in out.chunks_mut(9).take(n_blocks) {
        block.copy_from_slice(&norms);
    }
    // Partial trailing block.
    let trailing_start = n_blocks * 9;
    let trailing_len = D - trailing_start;
    out[trailing_start..trailing_start + trailing_len].copy_from_slice(&norms[..trailing_len]);
}

/// Per-displacement L2 norm ratios: ||disp_l|| / sum(||disp_l||).
/// Captures the DISTRIBUTION of delta magnitudes across layers.
/// Scale-invariant — isolates the per-layer profile from overall scale.
/// 8 features, replicated to fill D=32.
fn encode_disp_ratios(states: &[&[f32]], out: &mut [f32; D]) {
    let n_disps = states.len().saturating_sub(1);
    let mut norms = [0.0_f32; 8];
    let mut total = 0.0_f32;

    for l in 0..n_disps.min(8) {
        let mut sum_sq = 0.0_f32;
        for i in 0..states[l].len() {
            let diff = states[l + 1][i] - states[l][i];
            sum_sq += diff * diff;
        }
        norms[l] = sum_sq.sqrt();
        total += norms[l];
    }

    if total > 0.0 {
        for n in norms.iter_mut() {
            *n /= total;
        }
    }

    let n_blocks = D / 8;
    for block in out.chunks_mut(8).take(n_blocks) {
        block.copy_from_slice(&norms);
    }
    for j in (n_blocks * 8)..D {
        out[j] = 0.0;
    }
}

// ─── Nearest-centroid classification (encoder-agnostic) ────────────────────
//
// Same math as SweTrajectoryFreezer::derive_directions + freeze, but
// without the FAME commit envelope. Works on any D-dim summary vectors.

/// Derive archetype direction vectors from cluster centroids.
/// direction_k = normalize(centroid_k - global_centroid)
fn derive_directions_and_centroid(
    train_summaries: &[[[f32; D]; N_TRAIN]; N],
    directions: &mut [[f32; D]; N],
    global_centroid: &mut [f32; D],
) {
    // Global centroid = mean of all training summaries.
    for j in 0..D {
        global_centroid[j] = 0.0;
    }
    for mode in 0..N {
        for s in &train_summaries[mode] {
            for j in 0..D {
                global_centroid[j] += s[j];
            }
        }
    }
    let total = (N * N_TRAIN) as f32;
    for j in 0..D {
        global_centroid[j] /= total;
    }

    // Per-mode direction = normalize(centroid_k - global_centroid).
    for mode in 0..N {
        let mut centroid = [0.0_f32; D];
        for s in &train_summaries[mode] {
            for j in 0..D {
                centroid[j] += s[j];
            }
        }
        for j in 0..D {
            centroid[j] /= N_TRAIN as f32;
        }

        let mut norm_sq = 0.0_f32;
        for j in 0..D {
            directions[mode][j] = centroid[j] - global_centroid[j];
            norm_sq += directions[mode][j] * directions[mode][j];
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for j in 0..D {
                directions[mode][j] /= norm;
            }
        }
    }
}

/// Classify a summary via nearest-centroid dot product.
/// Returns the mode index with the highest dot(summary, direction_k).
fn classify(summary: &[f32; D], directions: &[[f32; D]; N], global_centroid: &[f32; D]) -> usize {
    let mut centered = [0.0_f32; D];
    for j in 0..D {
        centered[j] = summary[j] - global_centroid[j];
    }

    let mut best_mode = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for mode in 0..N {
        let mut dot = 0.0_f32;
        for j in 0..D {
            dot += centered[j] * directions[mode][j];
        }
        if dot > best_dot {
            best_dot = dot;
            best_mode = mode;
        }
    }
    best_mode
}

/// Compute the L2 distance between the two mode centroids (diagnostic).
fn centroid_distance(train_summaries: &[[[f32; D]; N_TRAIN]; N]) -> f32 {
    let mut centroids = [[0.0_f32; D]; N];
    for mode in 0..N {
        for s in &train_summaries[mode] {
            for j in 0..D {
                centroids[mode][j] += s[j];
            }
        }
        for j in 0..D {
            centroids[mode][j] /= N_TRAIN as f32;
        }
    }

    let mut dist_sq = 0.0_f32;
    for j in 0..D {
        let diff = centroids[0][j] - centroids[1][j];
        dist_sq += diff * diff;
    }
    dist_sq.sqrt()
}

// ─── Per-encoder, per-σ discrimination test ────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum EncoderKind {
    Geometry,
    DispNorms,
    DispStats,
    StateNorms,
    DispRatios,
}

impl EncoderKind {
    fn name(self) -> &'static str {
        match self {
            Self::Geometry => "Geometry",
            Self::DispNorms => "DispNorms",
            Self::DispStats => "DispStats",
            Self::StateNorms => "StateNorms",
            Self::DispRatios => "DispRatios",
        }
    }
}

struct TestResult {
    encoder: EncoderKind,
    sigma: f32,
    accuracy: f32,
    centroid_dist: f32,
    /// Mean within-class standard deviation across D dimensions.
    /// High relative to centroid_dist → classes overlap → accuracy ~50%.
    within_class_std: f32,
    /// Signal-to-noise ratio = centroid_dist / within_class_std.
    /// >2.0 → classes should be separable; <1.0 → hopelessly overlapped.
    snr: f32,
    /// Centroid-of-test-tokens classification: average all test summaries
    /// per model into one fingerprint, then classify. Eliminates token
    /// variance. If this is 100% but per-token accuracy is ~50%, the signal
    /// exists but is too weak per-token.
    centroid_accuracy: f32,
}

/// Run the full discrimination test for one encoder at one σ level.
#[allow(clippy::too_many_arguments)]
fn test_encoder_sigma(
    config: &KimiK3ModelConfig,
    weights_a: &KimiK3ModelWeights,
    weights_b: &KimiK3ModelWeights,
    tokens: &[u32],
    encoder_kind: EncoderKind,
    geom_encoder: &GeometrySummaryEncoder,
    runtime_a: &mut KimiK3Runtime,
    runtime_b: &mut KimiK3Runtime,
    scratch: &mut ExtractScratch,
) -> TestResult {
    // Extract all summaries.
    let mut summaries: [[[f32; D]; N_TOKENS]; N] = [[[0.0_f32; D]; N_TOKENS]; N];

    for (idx, &tok) in tokens.iter().enumerate() {
        // Model A: extract trajectory, then clone states for encoding
        // (avoids borrow conflict: traj_buf is borrowed by refs, but
        // the geometry encoder path also needs scratch for disp buffers).
        scratch.extract_traj(config, weights_a, runtime_a, tok);
        let states_a: Vec<Vec<f32>> = scratch.traj_buf.clone();
        let refs_a: Vec<&[f32]> = states_a.iter().map(|v| v.as_slice()).collect();
        encode(
            encoder_kind,
            &refs_a,
            scratch,
            geom_encoder,
            &mut summaries[0][idx],
        );

        // Model B
        scratch.extract_traj(config, weights_b, runtime_b, tok);
        let states_b: Vec<Vec<f32>> = scratch.traj_buf.clone();
        let refs_b: Vec<&[f32]> = states_b.iter().map(|v| v.as_slice()).collect();
        encode(
            encoder_kind,
            &refs_b,
            scratch,
            geom_encoder,
            &mut summaries[1][idx],
        );
    }

    // Stage 1: Fit directions from the training split.
    let mut train_summaries: [[[f32; D]; N_TRAIN]; N] = [[[0.0_f32; D]; N_TRAIN]; N];
    for mode in 0..N {
        for (train_idx, token_idx) in (0..N_TRAIN).enumerate() {
            train_summaries[mode][train_idx] = summaries[mode][token_idx];
        }
    }
    let mut directions: [[f32; D]; N] = [[0.0_f32; D]; N];
    let mut global_centroid = [0.0_f32; D];
    derive_directions_and_centroid(&train_summaries, &mut directions, &mut global_centroid);

    let centroid_dist = centroid_distance(&train_summaries);

    // Within-class standard deviation (mean across D dimensions).
    let mut within_var_sum = 0.0_f32;
    for mode in 0..N {
        for j in 0..D {
            let mean = train_summaries[mode].iter().map(|s| s[j]).sum::<f32>() / N_TRAIN as f32;
            let var = train_summaries[mode]
                .iter()
                .map(|s| {
                    let d = s[j] - mean;
                    d * d
                })
                .sum::<f32>()
                / N_TRAIN as f32;
            within_var_sum += var;
        }
    }
    let within_class_std = (within_var_sum / (N * D) as f32).sqrt();
    let snr = if within_class_std > 1e-12 {
        centroid_dist / within_class_std
    } else {
        f32::INFINITY
    };

    // Stage 2: Classify held-out trajectories.
    let mut n_correct = 0usize;
    let mut n_total = 0usize;

    for tok_idx in N_TRAIN..N_TOKENS {
        for mode in 0..N {
            let predicted = classify(&summaries[mode][tok_idx], &directions, &global_centroid);
            if predicted == mode {
                n_correct += 1;
            }
            n_total += 1;
        }
    }

    let accuracy = n_correct as f32 / n_total.max(1) as f32;

    // Centroid-of-test-tokens classification: average all test summaries per
    // model, then classify the averaged fingerprint.
    let mut centroid_correct = 0usize;
    let mut centroid_total = 0usize;
    for mode in 0..N {
        let mut test_centroid = [0.0_f32; D];
        let n_test = N_TOKENS - N_TRAIN;
        for tok_idx in N_TRAIN..N_TOKENS {
            for j in 0..D {
                test_centroid[j] += summaries[mode][tok_idx][j];
            }
        }
        for j in 0..D {
            test_centroid[j] /= n_test as f32;
        }
        let predicted = classify(&test_centroid, &directions, &global_centroid);
        if predicted == mode {
            centroid_correct += 1;
        }
        centroid_total += 1;
    }
    let centroid_accuracy = centroid_correct as f32 / centroid_total.max(1) as f32;

    TestResult {
        encoder: encoder_kind,
        sigma: 0.0, // filled by caller
        accuracy,
        centroid_dist,
        within_class_std,
        snr,
        centroid_accuracy,
    }
}

/// Dispatch to the appropriate encoder.
#[allow(clippy::too_many_arguments)]
fn encode(
    kind: EncoderKind,
    states: &[&[f32]],
    scratch: &mut ExtractScratch,
    geom_encoder: &GeometrySummaryEncoder,
    out: &mut [f32; D],
) {
    match kind {
        EncoderKind::Geometry => encode_geometry(states, scratch, geom_encoder, out),
        EncoderKind::DispNorms => encode_disp_norms(states, out),
        EncoderKind::DispStats => encode_disp_stats(states, out),
        EncoderKind::StateNorms => encode_state_norms(states, out),
        EncoderKind::DispRatios => encode_disp_ratios(states, out),
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  P011 follow-up — value-sensitive encoder discrimination probe     ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let d = config.hidden_size;
    println!("Config: D={d}, layers={}", config.num_layers);
    println!("Summary dim: {D}, archetypes: {N} (original vs perturbed)");
    println!(
        "Tokens: {N_TOKENS} ({N_TRAIN} train + {} test per model)",
        N_TOKENS - N_TRAIN
    );
    println!("Sigma levels: {SIGMA_LEVELS:?}");
    println!();

    // ── Load real model ───────────────────────────────────────────────────
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    });
    let model_path = format!("{model_dir}/model.safetensors");

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ERROR: this experiment requires real model.safetensors at {model_path}");
        std::process::exit(1);
    }

    print!("Loading real model.safetensors ... ");
    let t0 = std::time::Instant::now();
    let weights_a = katgpt_rs::kimi_k3::loader::load_kimi_k3(&model_path).unwrap_or_else(|e| {
        eprintln!("\n  load failed: {e}");
        std::process::exit(1);
    });
    println!("done ({:.1}s)", t0.elapsed().as_secs_f64());
    println!();

    // ── Shared setup ──────────────────────────────────────────────────────
    let max_seq_len = 64;
    let mut runtime_a = KimiK3Runtime::new(&config, max_seq_len);
    let mut runtime_b = KimiK3Runtime::new(&config, max_seq_len);
    let geom_encoder = GeometrySummaryEncoder::default_depth_trajectory();
    let tokens: Vec<u32> = (1..=N_TOKENS as u32)
        .map(|i| (i * 7 + 3) % (BENCH_VOCAB as u32))
        .collect();
    let mut scratch = ExtractScratch::new(d);

    let encoders = [
        EncoderKind::Geometry,
        EncoderKind::DispNorms,
        EncoderKind::DispStats,
        EncoderKind::StateNorms,
        EncoderKind::DispRatios,
    ];

    // ── Run the sweep ─────────────────────────────────────────────────────
    let mut all_results: Vec<TestResult> = Vec::new();

    for &sigma in SIGMA_LEVELS {
        // Clone Model A + perturb at σ.
        let mut weights_b = weights_a.clone();
        perturb_model(&mut weights_b, sigma);

        println!("── σ = {sigma} ──────────────────────────────────────────");
        println!(
            "  {:>12}  {:>8}  {:>10}  {:>8}  {:>6}  {:>8}  {:>8}",
            "encoder", "acc", "centroid_d", "within_σ", "SNR", "centroid", "verdict"
        );
        println!("  {}", "-".repeat(78));

        for &ek in &encoders {
            let mut result = test_encoder_sigma(
                &config,
                &weights_a,
                &weights_b,
                &tokens,
                ek,
                &geom_encoder,
                &mut runtime_a,
                &mut runtime_b,
                &mut scratch,
            );
            result.sigma = sigma;

            let verdict = if sigma == 0.0 {
                if result.accuracy <= 0.60 {
                    "OK"
                } else {
                    "WARN"
                }
            } else if result.accuracy >= 0.80 {
                "PASS"
            } else if result.centroid_accuracy >= 1.0 {
                "SIGNAL"
            } else {
                ""
            };

            println!(
                "  {:>12}  {:>7.1}%  {:>10.4}  {:>8.4}  {:>6.2}  {:>7.1}%  {:>8}",
                ek.name(),
                result.accuracy * 100.0,
                result.centroid_dist,
                result.within_class_std,
                result.snr,
                result.centroid_accuracy * 100.0,
                verdict,
            );
            all_results.push(result);
        }
        println!();
    }

    // ── Analysis ─────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("Cross-encoder comparison at σ=0.5 (max perturbation):");
    println!();

    for &ek in &encoders {
        if let Some(r) = all_results
            .iter()
            .find(|r| r.encoder == ek && r.sigma == 0.5)
        {
            let bar_len = (r.accuracy * 40.0) as usize;
            let bar: String = "█".repeat(bar_len);
            println!(
                "  {:>12}: acc={:>5.1}%  centroid_d={:>10.4}  within_σ={:>10.4}  SNR={:>6.2}  centroid_acc={:>5.1}%  {}",
                ek.name(),
                r.accuracy * 100.0,
                r.centroid_dist,
                r.within_class_std,
                r.snr,
                r.centroid_accuracy * 100.0,
                bar,
            );
        }
    }
    println!();

    // ── Discrimination floor per encoder ──────────────────────────────────
    println!("Discrimination floor (smallest σ where accuracy ≥ 80%):");
    println!();
    for &ek in &encoders {
        let floor = all_results
            .iter()
            .filter(|r| r.encoder == ek)
            .find(|r| r.sigma > 0.0 && r.accuracy >= 0.80)
            .map(|r| r.sigma);

        if let Some(f) = floor {
            println!("  {:>12}: σ* = {:.4} ✅", ek.name(), f)
        } else {
            let max_acc = all_results
                .iter()
                .filter(|r| r.encoder == ek)
                .map(|r| r.accuracy)
                .fold(0.0_f32, f32::max);
            println!(
                "  {:>12}: no floor (max acc = {:.1}%) ❌",
                ek.name(),
                max_acc * 100.0
            );
        }
    }
    println!();

    // ── Verdict ───────────────────────────────────────────────────────────
    let any_value_encoder_works = encoders.iter().any(|&ek| {
        ek != EncoderKind::Geometry
            && all_results
                .iter()
                .any(|r| r.encoder == ek && r.sigma > 0.0 && r.accuracy >= 0.80)
    });

    let signal_exists_but_too_weak = encoders.iter().any(|&ek| {
        ek != EncoderKind::Geometry
            && all_results.iter().any(|r| {
                r.encoder == ek && r.sigma > 0.0 && r.centroid_accuracy >= 1.0 && r.accuracy < 0.80
            })
    });

    println!("══════════════════════════════════════════════════════════════════");
    if any_value_encoder_works {
        println!("VERDICT: At least one value-sensitive encoder discriminates");
        println!("perturbed vs original weights at ≥80% per-token accuracy. The");
        println!("bench_015 negative was ENCODER-SPECIFIC, not fundamental.");
    } else if signal_exists_but_too_weak {
        println!("VERDICT: Signal EXISTS but is too weak for per-token classification.");
        println!("Centroid-of-test-tokens classification succeeds (centroid_acc=100%),");
        println!("but per-token accuracy stays at ~50%. The perturbation changes the");
        println!("MEAN feature vector (centroids separate) but token-to-token variance");
        println!("is much larger than the perturbation signal (SNR << 1). Individual");
        println!("token trajectories cannot discriminate; a multi-token aggregate");
        println!("or covariance-aware classifier (Mahalanobis/LDA) would be needed.");
        println!();
        println!("This is a RESOLUTION FLOOR for per-token trajectory classification,");
        println!("not an information deficit. The depth trajectory captures the signal");
        println!("but at insufficient SNR for nearest-centroid per-token decisions.");
    } else {
        println!("VERDICT: No value-sensitive encoder discriminates at ANY level.");
        println!("The bench_015 negative is CONFIRMED as fundamental.");
    }
    println!("══════════════════════════════════════════════════════════════════");
}

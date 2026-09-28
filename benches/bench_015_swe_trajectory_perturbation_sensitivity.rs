//! Proposal 011 Phase 5 follow-up — perturbation sensitivity analysis.
//!
//! **The question this bench answers:** T5.6 (Bench 014) proved the
//! SweTrajectoryFreezer discriminates real Kimi-K3 from RANDOM weights at
//! 100% accuracy — the EXTREME case. But how ROBUST is this signal? At what
//! weight-perturbation magnitude does discrimination emerge?
//!
//! This is a **cross-snapshot proxy**: two real model checkpoints from
//! different training steps differ by structured gradient drift. We don't
//! have a second real checkpoint (only one Kimi-K3 model.safetensors), so
//! we perturb the real weights by additive relative noise at increasing σ
//! and measure the discrimination accuracy curve.
//!
//! # Why this matters
//!
//! - If discrimination emerges at σ=0.001 (0.1% relative noise — typical
//!   FP16 quantization level), the signal is VERY robust → cross-snapshot
//!   discrimination at training-step granularity is likely to work.
//! - If discrimination only emerges at σ≥0.1 (10% relative noise), the
//!   signal is FRAGILE → cross-snapshot discrimination may require many
//!   training steps to produce detectable trajectory drift.
//! - If discrimination never emerges below σ=0.5, depth trajectories alone
//!   are insufficient for fine-grained snapshot discrimination → the
//!   iterative refinement trajectory (T5.4 path 2) becomes the necessary
//!   substrate.
//!
//! **A negative result is valuable** — it documents the freezer's resolution
//! floor honestly, informing whether cross-snapshot is worth pursuing with
//! real checkpoints (separate proposal).
//!
//! # Method
//!
//! Model A = real `model.safetensors` (loaded once).
//! Model B(σ) = clone of Model A with every weight perturbed:
//!   `w' = w * (1 + σ * noise)` where `noise ~ Uniform(-0.5, +0.5)`.
//!
//! Relative perturbation preserves each weight's magnitude distribution
//! (unlike absolute noise, which would be dominated by large-magnitude
//! tensors like the embedding). The LCG noise is seeded deterministically
//! per σ level so results are reproducible.
//!
//! For each σ ∈ {0.0, 0.001, 0.01, 0.05, 0.1, 0.5}:
//!   1. Perturb a fresh clone of Model A at σ → Model B(σ).
//!   2. Extract 32-token depth trajectories for both models (12 train + 20 test).
//!   3. Fit directions from the train split (`derive_directions_and_centroid`).
//!   4. Classify the held-out test split via `freeze_attempt_into`.
//!   5. Record accuracy.
//!
//! # Run
//!
//! ```bash
//! # Requires real model.safetensors at data/kimi-k3-0.40b/:
//! cargo bench --features "kimi_k3_loader swe_trajectory_freeze" \
//!     --bench bench_015_swe_trajectory_perturbation_sensitivity -- --nocapture
//! ```

#![cfg(all(feature = "kimi_k3_loader", feature = "swe_trajectory_freeze"))]
#![allow(clippy::needless_range_loop)]

use katgpt_attn::gdn2::kda_forward::KdaWeights;
use katgpt_attn::mla::MlaWeights;
use katgpt_core::committed_field_blend::ArchetypeFieldSource;
use katgpt_core::latent_trajectory_geometry::from_states_into;
use katgpt_core::swe_trajectory_freeze::{GeometrySummaryEncoder, SweTrajectoryFreezer};
use katgpt_rs::kimi_k3::decoder_layer::{
    KimiAttentionWeights, KimiDecoderLayerWeights, KimiFfnWeights,
};
use katgpt_rs::kimi_k3::loader::KimiK3ModelWeights;
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token_traced};
use katgpt_transformer::attn_res::AttnResWeights;
use katgpt_transformer::moe::{MoeWeights, SwiGluExpertWeights};

// ─── Constants ─────────────────────────────────────────────────────────────

/// Summary dimension D (matches bench_014).
const D: usize = 32;

/// Number of archetype modes (Model A vs Model B(σ)).
const N: usize = 2;

/// Total tokens to extract trajectories for (split into train + test).
const N_TOKENS: usize = 32;

/// Training split size (per model). Must be ≤ N_TOKENS / 2.
const N_TRAIN: usize = 12;

/// Truncated vocab for the bench (matches bench_014).
const BENCH_VOCAB: usize = 512;

/// Perturbation σ levels to probe (relative noise magnitude).
/// σ=0.0 is the sanity check (identical weights → ~50% accuracy).
const SIGMA_LEVELS: &[f32] = &[0.0, 0.001, 0.01, 0.05, 0.1, 0.5];

// ─── Deterministic LCG (from bench_012/014) ────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Uniform(-0.5, +0.5) — bounded noise for relative perturbation.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

// ─── Weight perturbation (relative additive noise) ─────────────────────────
//
// `w' = w * (1 + σ * noise)` where `noise ~ Uniform(-0.5, +0.5)`.
// This preserves each weight tensor's magnitude distribution (unlike absolute
// noise which would be dominated by large tensors). The LCG is seeded
// deterministically per σ level so results are reproducible.

#[inline]
fn perturb_vec(v: &mut [f32], rng: &mut Lcg, sigma: f32) {
    if sigma == 0.0 {
        return;
    }
    for w in v.iter_mut() {
        let noise = rng.next_f32(); // Uniform(-0.5, +0.5)
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

/// Perturb every weight in the model by relative additive noise at magnitude σ.
/// Seeds the LCG deterministically from σ so each level is reproducible.
fn perturb_model(w: &mut KimiK3ModelWeights, sigma: f32) {
    // Deterministic seed from σ (avoids floating-point bit-cast issues).
    let seed = (sigma * 1_000_000.0) as u64 | 0xA15E_0000;
    let mut rng = Lcg::new(seed);
    perturb_vec(&mut w.embed_weight, &mut rng, sigma);
    for layer in w.layers.iter_mut() {
        perturb_layer(layer, &mut rng, sigma);
    }
    perturb_vec(&mut w.final_norm_weight, &mut rng, sigma);
    // lm_head_weight is empty in the bench config — skip.
    if !w.lm_head_weight.is_empty() {
        perturb_vec(&mut w.lm_head_weight, &mut rng, sigma);
    }
    perturb_attn_res(&mut w.output_attn_res, &mut rng, sigma);
}

// ─── Stub archetype fields (for FAME commit — only commitment() is read) ────

struct StubField {
    idx: u8,
}
impl StubField {
    const fn new(idx: u8) -> Self {
        Self { idx }
    }
}
impl ArchetypeFieldSource<D> for StubField {
    fn evolve<'a>(&self, _z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        for x in dz_scratch.iter_mut().take(D) {
            *x = 0.0;
        }
        &mut dz_scratch[..D]
    }
    fn commitment(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"stub_field:");
        h.update(&[self.idx]);
        h.finalize().into()
    }
}

fn make_fields() -> [&'static dyn ArchetypeFieldSource<D>; N] {
    static S0: StubField = StubField::new(0);
    static S1: StubField = StubField::new(1);
    [&S0 as &'static dyn ArchetypeFieldSource<D>, &S1]
}

// ─── Trajectory extraction (from bench_014) ────────────────────────────────

struct ExtractScratch {
    disp_curr: Vec<f32>,
    disp_prev: Vec<f32>,
    traj_buf: Vec<Vec<f32>>,
    summary: [f32; D],
}

impl ExtractScratch {
    fn new(hidden_dim: usize) -> Self {
        Self {
            disp_curr: Vec::with_capacity(hidden_dim),
            disp_prev: Vec::with_capacity(hidden_dim),
            traj_buf: Vec::with_capacity(9),
            summary: [0.0; D],
        }
    }
}

fn extract_summary(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    token_id: u32,
    encoder: &GeometrySummaryEncoder,
    scratch: &mut ExtractScratch,
) {
    runtime.reset();
    scratch.traj_buf.clear();
    let _ = kimi_k3_forward_token_traced(config, weights, runtime, token_id, &mut scratch.traj_buf);
    let refs: Vec<&[f32]> = scratch.traj_buf.iter().map(|v| v.as_slice()).collect();
    let geom = from_states_into(&refs, &mut scratch.disp_curr, &mut scratch.disp_prev);
    encoder.encode_into(&geom, &mut scratch.summary);
}

// ─── Per-σ discrimination test ─────────────────────────────────────────────

struct SigmaResult {
    sigma: f32,
    accuracy: f32,
    n_correct: usize,
    n_total: usize,
    /// L2 distance between the two mode centroids in the D-dim summary space.
    /// Near-zero → centroids overlap → directions are degenerate → no signal.
    centroid_dist: f32,
}

/// Run the full discrimination test (fit + classify) for Model A vs Model B(σ).
/// Returns the accuracy on the held-out test split.
#[allow(clippy::too_many_arguments)]
fn test_sigma(
    config: &KimiK3ModelConfig,
    weights_a: &KimiK3ModelWeights,
    weights_b: &KimiK3ModelWeights,
    tokens: &[u32],
    encoder: &GeometrySummaryEncoder,
    runtime_a: &mut KimiK3Runtime,
    runtime_b: &mut KimiK3Runtime,
    scratch: &mut ExtractScratch,
) -> SigmaResult {
    let d = config.hidden_size;

    // Extract all summaries.
    let mut summaries: [[[f32; D]; N_TOKENS]; N] = [[[0.0_f32; D]; N_TOKENS]; N];
    for (idx, &tok) in tokens.iter().enumerate() {
        extract_summary(config, weights_a, runtime_a, tok, encoder, scratch);
        summaries[0][idx] = scratch.summary;
        extract_summary(config, weights_b, runtime_b, tok, encoder, scratch);
        summaries[1][idx] = scratch.summary;
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
    katgpt_core::swe_trajectory_freeze::derive_directions_and_centroid(
        &train_summaries,
        &mut directions,
        &mut global_centroid,
    );

    // Diagnostic: L2 distance between the two mode centroids.
    // If this is near-zero, the directions are degenerate (centroids overlap).
    let mut centroid_dist_sq = 0.0_f32;
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
        if mode == 0 {
            // Accumulate (centroid_0 - global_centroid)²
            for j in 0..D {
                let diff = centroid[j] - global_centroid[j];
                centroid_dist_sq += diff * diff;
            }
        }
    }
    // centroid_dist = ||centroid_0 - global_centroid|| = half the inter-centroid dist.
    let centroid_dist = 2.0 * centroid_dist_sq.sqrt();

    // Stage 2: Freeze + classify held-out trajectories.
    let freezer =
        SweTrajectoryFreezer::<N, D>::with_centroid(directions, global_centroid, *encoder);
    let fields: [&dyn ArchetypeFieldSource<D>; N] = make_fields();

    let mut freeze_disp_curr = Vec::<f32>::with_capacity(d);
    let mut freeze_disp_prev = Vec::<f32>::with_capacity(d);

    let mut n_correct = 0usize;
    let mut n_total = 0usize;

    for tok_idx in N_TRAIN..N_TOKENS {
        for mode in 0..N {
            let weights = if mode == 0 { weights_a } else { weights_b };
            let runtime = if mode == 0 {
                &mut *runtime_a
            } else {
                &mut *runtime_b
            };
            runtime.reset();
            scratch.traj_buf.clear();
            let _ = kimi_k3_forward_token_traced(
                config,
                weights,
                runtime,
                tokens[tok_idx],
                &mut scratch.traj_buf,
            );
            let refs: Vec<&[f32]> = scratch.traj_buf.iter().map(|v| v.as_slice()).collect();
            let frozen = freezer.freeze_attempt_into(
                &refs,
                &fields,
                1,
                &mut freeze_disp_curr,
                &mut freeze_disp_prev,
            );
            let argmax_k = frozen.argmax_archetype();
            if argmax_k == mode {
                n_correct += 1;
            }
            n_total += 1;
        }
    }

    let accuracy = n_correct as f32 / n_total.max(1) as f32;
    SigmaResult {
        sigma: 0.0, // filled by caller
        accuracy,
        n_correct,
        n_total,
        centroid_dist,
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  P011 follow-up — perturbation sensitivity (cross-snapshot proxy)  ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let d = config.hidden_size;
    println!(
        "Config: D={d}, layers={}, MLA@[3,7], KDA@[0..6], MoE@[1..7]",
        config.num_layers
    );
    println!("Summary dim: {D}, archetypes: {N} (original vs perturbed)");
    println!(
        "Tokens: {N_TOKENS} ({N_TRAIN} train + {} test per model)",
        N_TOKENS - N_TRAIN
    );
    println!("Sigma levels: {SIGMA_LEVELS:?}");
    println!();

    // ── Load real model (REQUIRED — no fallback for this experiment) ──────
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    });
    let model_path = format!("{model_dir}/model.safetensors");

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ERROR: this experiment requires real model.safetensors at {model_path}");
        eprintln!("The perturbation sensitivity probe is meaningless on random weights");
        eprintln!("(perturbing random weights produces more random weights — the");
        eprintln!("extreme case already tested in bench_014).");
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
    let encoder = GeometrySummaryEncoder::default_depth_trajectory();
    let tokens: Vec<u32> = (1..=N_TOKENS as u32)
        .map(|i| (i * 7 + 3) % (BENCH_VOCAB as u32))
        .collect();
    let mut scratch = ExtractScratch::new(d);

    // ── Run the sensitivity sweep ─────────────────────────────────────────
    println!("── Sensitivity sweep: σ vs discrimination accuracy ──");
    println!(
        "  {:>10}  {:>10}  {:>12}  {:>14}  {:>10}",
        "sigma", "accuracy", "correct/total", "centroid_dist", "verdict"
    );
    println!("  {}", "-".repeat(66));

    let mut results: Vec<SigmaResult> = Vec::with_capacity(SIGMA_LEVELS.len());

    for &sigma in SIGMA_LEVELS {
        // Clone Model A + perturb at σ.
        let mut weights_b = weights_a.clone();
        perturb_model(&mut weights_b, sigma);

        let mut result = test_sigma(
            &config,
            &weights_a,
            &weights_b,
            &tokens,
            &encoder,
            &mut runtime_a,
            &mut runtime_b,
            &mut scratch,
        );
        result.sigma = sigma;

        let verdict = if sigma == 0.0 {
            // Sanity: identical weights → expect ~50% (no signal).
            if result.accuracy <= 0.60 {
                "OK (sanity)"
            } else {
                "WARN (>60%)"
            }
        } else if result.accuracy >= 0.80 {
            "PASS"
        } else {
            "below 80%"
        };

        println!(
            "  {:>10.4}  {:>10.2}  {:>6}/{:<6}    {:>14.6}  {:>10}",
            sigma, result.accuracy, result.n_correct, result.n_total, result.centroid_dist, verdict
        );
        results.push(result);
    }
    println!();

    // ── Analysis ──────────────────────────────────────────────────────────
    println!("── Analysis ──");

    // Find the discrimination floor: smallest σ where accuracy ≥ 80%.
    let floor = results
        .iter()
        .find(|r| r.sigma > 0.0 && r.accuracy >= 0.80)
        .map(|r| r.sigma);

    if let Some(sigma_floor) = floor {
        println!("   Discrimination floor (σ* where accuracy first ≥ 80%): {sigma_floor}");
        println!();
        if sigma_floor <= 0.001 {
            println!("   INTERPRETATION: signal is VERY ROBUST — discrimination");
            println!("   emerges at 0.1% relative noise (FP16 quantization level).");
            println!("   Cross-snapshot discrimination at training-step granularity");
            println!("   is LIKELY to work (subtle weight drift is detectable).");
        } else if sigma_floor <= 0.01 {
            println!("   INTERPRETATION: signal is MODERATELY ROBUST — discrimination");
            println!("   emerges at 1% relative noise. Cross-snapshot discrimination");
            println!("   should work for checkpoints separated by enough training");
            println!("   steps to produce ≥1% average weight drift.");
        } else if sigma_floor <= 0.1 {
            println!("   INTERPRETATION: signal is MODERATE — discrimination emerges");
            println!("   at 10% relative noise. Cross-snapshot discrimination requires");
            println!("   significant weight divergence (many training steps).");
        } else {
            println!("   INTERPRETATION: signal is FRAGILE — discrimination only");
            println!("   emerges above 10% relative noise. Fine-grained cross-snapshot");
            println!("   discrimination is unlikely via depth trajectories alone;");
            println!("   the iterative refinement trajectory (T5.4 path 2) becomes");
            println!("   the necessary substrate.");
        }
    } else {
        // Check if even σ=0.5 didn't reach 80%.
        let max_acc = results.iter().map(|r| r.accuracy).fold(0.0_f32, f32::max);
        if max_acc < 0.80 {
            println!("   NO discrimination floor found — accuracy never reached 80%");
            println!("   even at σ=0.5 (50% relative noise). The perturbation does NOT");
            println!("   produce discriminative depth trajectories via the freezer.");
            println!("   NOTE: this is unexpected given bench_014's 100% on real-vs-random;");
            println!("   it suggests the discrimination signal lives in the STRUCTURE");
            println!("   of real weights, not just their values. Relative perturbation");
            println!("   preserves structure; random weights destroy it.");
        }
    }
    println!();

    // ── Sanity check: σ=0.0 should be ~50% ─────────────────────────────────
    let sanity = &results[0];
    println!("── Sanity check (σ=0.0) ──");
    if sanity.accuracy <= 0.60 {
        println!(
            "   σ=0.0 accuracy = {:.2} (≤60%) ✅ — identical weights produce",
            sanity.accuracy
        );
        println!("   no discriminative signal, confirming the test is well-calibrated.");
    } else {
        println!(
            "   σ=0.0 accuracy = {:.2} (>60%) ⚠️ — unexpected signal at zero",
            sanity.accuracy
        );
        println!("   perturbation. This may indicate overfitting (directions derived");
        println!("   from the same weights that produce the test trajectories).");
    }
    println!();

    // ── Summary ───────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("Perturbation sensitivity curve:");
    for r in &results {
        let bar_len = (r.accuracy * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "   σ={:<8.4}  {:>5.1}%  {}",
            r.sigma,
            r.accuracy * 100.0,
            bar
        );
    }
    println!();
    println!("This is a CROSS-SNAPSHOT PROXY (additive relative noise, not");
    println!("structured gradient drift). The discrimination floor informs whether");
    println!("real cross-snapshot discrimination is worth pursuing with multiple");
    println!("real checkpoints (separate proposal).");
}

//! Proposal 011 Phase 5 — T5.6 G5 gate: cross-model discrimination on real
//! Kimi-K3 depth trajectories.
//!
//! **The load-bearing question for Layer 4:** does trajectory geometry
//! discriminate across snapshots/models? T5.4 PARTIAL documented that the
//! per-token DEPTH trajectory is only 29% distinct across tokens (same model,
//! different inputs) — depth geometry is dominated by the LAYER WEIGHT
//! STRUCTURE, which is the same across tokens. **This is actually good news
//! for G5:** if depth trajectories are model-specific rather than input-
//! specific, different MODELS should produce measurably different depth
//! trajectory geometries — exactly what G5 needs.
//!
//! # Experiment design
//!
//! Two "models" (snapshots) are loaded:
//! - **Model A** — real `model.safetensors` if present; else random seed=42.
//! - **Model B** — random weights with a different seed (137).
//!
//! For each of N_TOKENS diverse token IDs, both models produce a 9-state depth
//! trajectory (embed + 8 post-layer hidden states). The trajectory geometry is
//! computed + encoded into a D-dimensional summary. The `SweTrajectoryFreezer`
//! is fit on a training split (derive_directions from the two model centroids)
//! and tested on a held-out split (freeze + classify).
//!
//! **G5 PASS criterion:** classification accuracy ≥ 80% on the held-out split.
//!
//! If G5 PASSES → depth trajectory geometry DOES discriminate across models
//! → the modelless Layer 4 path is validated for snapshot/model discrimination.
//! If G5 FAILS → even extreme weight differences don't produce discriminative
//! depth trajectory geometry → T5.7 (document + file Layer 4b).
//!
//! # Run
//!
//! ```bash
//! # Random weights only (always runnable):
//! cargo bench --features "kimi_k3_loader swe_trajectory_freeze" \
//!     --bench bench_014_swe_trajectory_freezer_g5 -- --nocapture
//!
//! # With real model.safetensors (if present at data/kimi-k3-0.40b/):
//! KIMI_K3_MODEL_DIR=/path/to/model cargo bench --features "kimi_k3_loader swe_trajectory_freeze" \
//!     --bench bench_014_swe_trajectory_freezer_g5 -- --nocapture
//! ```

#![cfg(all(feature = "kimi_k3_loader", feature = "swe_trajectory_freeze"))]
#![allow(clippy::needless_range_loop)]

use katgpt_attn::gdn2::kda_forward::KdaWeights;
use katgpt_attn::mla::MlaWeights;
use katgpt_core::committed_field_blend::ArchetypeFieldSource;
use katgpt_core::latent_trajectory_geometry::from_states_into;
use katgpt_core::swe_trajectory_freeze::{GeometrySummaryEncoder, SweTrajectoryFreezer};
use katgpt_rs::kimi_k3::decoder_layer::{
    KimiAttentionWeights, KimiDecoderLayerWeights, KimiFfnConfig, KimiFfnWeights,
};
use katgpt_rs::kimi_k3::loader::KimiK3ModelWeights;
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token_traced};
use katgpt_transformer::attn_res::AttnResWeights;
use katgpt_transformer::moe::{MoeWeights, SwiGluExpertWeights};

// ─── Constants ─────────────────────────────────────────────────────────────

/// Summary dimension D. The geometry encoder writes 4 features (length,
/// curvature, min_cosine, n_steps) replicated across D/4 blocks.
const D: usize = 32;

/// Number of archetype modes (Model A vs Model B).
const N: usize = 2;

/// Total tokens to extract trajectories for (split into train + test).
const N_TOKENS: usize = 32;

/// Training split size (per model). Must be ≤ N_TOKENS / 2.
const N_TRAIN: usize = 12;

/// Truncated vocab for random-weight embedding.
const BENCH_VOCAB: usize = 512;

const MODE_NAMES: [&str; N] = ["model_a", "model_b"];

// ─── Deterministic LCG + random weights (from bench_012) ───────────────────

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
    #[inline]
    fn xavier_vec(&mut self, n: usize, fan_in: usize) -> Vec<f32> {
        let scale = (2.0 / fan_in as f32).sqrt();
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

/// Build a full `KimiK3ModelWeights` with random but structurally valid weights.
/// (Copied from bench_012 — same Xavier-scaled random constructor.)
fn build_random_weights(config: &KimiK3ModelConfig, seed: u64) -> KimiK3ModelWeights {
    let d = config.hidden_size;
    let mut rng = Lcg::new(seed);

    let embed_weight = rng.xavier_vec(BENCH_VOCAB * d, d);

    let layers: Vec<KimiDecoderLayerWeights> = (0..config.num_layers)
        .map(|layer_idx| {
            let layer_seed = seed.wrapping_add((layer_idx as u64) * 7919);

            let input_layernorm_weight: Vec<f32> =
                (0..d).map(|_| 1.0 + rng.next_f32() * 0.02).collect();
            let post_attention_layernorm_weight: Vec<f32> =
                (0..d).map(|_| 1.0 + rng.next_f32() * 0.02).collect();

            let is_mla = layer_idx == 3 || layer_idx == 7;
            let attention = if is_mla {
                KimiAttentionWeights::Mla(MlaWeights::random(&config.mla_config, layer_seed))
            } else {
                KimiAttentionWeights::Kda(KdaWeights::random(&config.kda_config, layer_seed))
            };

            let is_dense = layer_idx == 0;
            let ffn = if is_dense {
                let intermediate = match &config.dense_ffn_config {
                    KimiFfnConfig::Dense {
                        intermediate_size, ..
                    } => *intermediate_size,
                    _ => unreachable!("layer 0 must be Dense"),
                };
                let mut expert_rng = Lcg::new(layer_seed.wrapping_add(1));
                let gate_proj = expert_rng.xavier_vec(intermediate * d, d);
                let up_proj = expert_rng.xavier_vec(intermediate * d, d);
                let down_proj = expert_rng.xavier_vec(d * intermediate, intermediate);
                KimiFfnWeights::Dense(SwiGluExpertWeights {
                    gate_proj,
                    up_proj,
                    down_proj,
                })
            } else {
                KimiFfnWeights::Moe(MoeWeights::random(&config.moe_config, layer_seed))
            };

            let self_attn_res = AttnResWeights::random(d, layer_seed.wrapping_add(2));
            let mlp_attn_res = AttnResWeights::random(d, layer_seed.wrapping_add(3));

            KimiDecoderLayerWeights {
                input_layernorm_weight,
                post_attention_layernorm_weight,
                attention,
                ffn,
                self_attn_res,
                mlp_attn_res,
            }
        })
        .collect();

    let final_norm_weight: Vec<f32> = (0..d).map(|_| 1.0).collect();
    let lm_head_weight: Vec<f32> = Vec::new();
    let output_attn_res = AttnResWeights::random(d, seed.wrapping_add(999));

    KimiK3ModelWeights {
        embed_weight,
        layers,
        final_norm_weight,
        lm_head_weight,
        output_attn_res,
    }
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
        // Should never be called by freeze_attempt (only commitment() is read).
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

// ─── Trajectory extraction ─────────────────────────────────────────────────

/// Reusable scratch buffers for trajectory extraction + geometry computation.
/// Avoids per-call allocation in the hot loop.
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
            traj_buf: Vec::with_capacity(9), // embed + 8 layers
            summary: [0.0; D],
        }
    }
}

/// Extract a per-token depth trajectory from the model + return its geometry
/// summary via `from_states_into` with reusable scratch buffers (zero-alloc
/// steady state).
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

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Proposal 011 T5.6 — SweTrajectoryFreezer G5: cross-model disc.   ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let d = config.hidden_size;
    println!(
        "Config: D={d}, layers={}, MLA@[3,7], KDA@[0..6], MoE@[1..7]",
        config.num_layers
    );
    println!("Summary dim: {D}, archetypes: {N} (model_a vs model_b)");
    println!(
        "Tokens: {N_TOKENS} ({N_TRAIN} train + {} test per model)",
        N_TOKENS - N_TRAIN
    );
    println!();

    // ── Load weights: Model A (real if available, else random seed=42) ─────
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    });
    let model_path = format!("{model_dir}/model.safetensors");
    let has_real = std::path::Path::new(&model_path).exists();

    let (weights_a, label_a) = if has_real {
        println!("Model A: REAL model.safetensors ({model_path})");
        print!("  Loading ... ");
        let t0 = std::time::Instant::now();
        let w = katgpt_rs::kimi_k3::loader::load_kimi_k3(&model_path).unwrap_or_else(|e| {
            eprintln!("\n  load failed: {e}");
            std::process::exit(1);
        });
        println!("done ({:.1}s)", t0.elapsed().as_secs_f64());
        (w, "REAL".to_string())
    } else {
        println!("Model A: random weights (seed=42) — no model.safetensors found");
        (build_random_weights(&config, 42), "random-42".to_string())
    };

    // ── Model B: always random (seed=137) ─────────────────────────────────
    println!("Model B: random weights (seed=137)");
    let weights_b = build_random_weights(&config, 137);
    println!();

    // ── Runtimes ──────────────────────────────────────────────────────────
    let max_seq_len = 64;
    let mut runtime_a = KimiK3Runtime::new(&config, max_seq_len);
    let mut runtime_b = KimiK3Runtime::new(&config, max_seq_len);

    // ── Encoder tuned for depth trajectories ──────────────────────────────
    let encoder = GeometrySummaryEncoder::default_depth_trajectory();

    // ── Token IDs: diverse spread across the vocab ────────────────────────
    let tokens: Vec<u32> = (1..=N_TOKENS as u32)
        .map(|i| (i * 7 + 3) % (BENCH_VOCAB as u32))
        .collect();

    // ── Extract all summaries ────────────────────────────────────────
    println!("Extracting depth trajectories for {N_TOKENS} tokens × 2 models ...");

    // summaries[mode][token_idx] = [f32; D]
    let mut summaries: [[[f32; D]; N_TOKENS]; N] = [[[0.0_f32; D]; N_TOKENS]; N];

    // Reusable scratch (zero-alloc steady state).
    let mut scratch = ExtractScratch::new(d);

    for (idx, &tok) in tokens.iter().enumerate() {
        // Model A
        extract_summary(
            &config,
            &weights_a,
            &mut runtime_a,
            tok,
            &encoder,
            &mut scratch,
        );
        summaries[0][idx] = scratch.summary;

        // Model B
        extract_summary(
            &config,
            &weights_b,
            &mut runtime_b,
            tok,
            &encoder,
            &mut scratch,
        );
        summaries[1][idx] = scratch.summary;
    }
    println!("Done.");
    println!();

    // ── Stage 1: Fit — derive directions from training split ──────────────
    // train_summaries[mode][train_idx] = [f32; D], shape [N][N_TRAIN][D]
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

    // ── Diagnostic: centroids + raw geometry ──────────────────────────────
    println!("── Diagnostic: training centroids + geometry features ──");
    for mode in 0..N {
        // Compute centroid of this mode's training summaries.
        let mut centroid = [0.0_f32; D];
        for s in &train_summaries[mode] {
            for j in 0..D {
                centroid[j] += s[j];
            }
        }
        for j in 0..D {
            centroid[j] /= N_TRAIN as f32;
        }
        // Print the first 4 features (one block — they're replicated).
        println!(
            "   {} centroid: length_norm={:.4}, curvature_norm={:.4}, cosine_norm={:.4}, n_steps_norm={:.4}",
            MODE_NAMES[mode], centroid[0], centroid[1], centroid[2], centroid[3]
        );
    }
    // Print a sample test summary from each model.
    println!("   Sample test summaries:");
    for mode in 0..N {
        let s = &summaries[mode][N_TRAIN]; // first test token
        println!(
            "   {} test[0]: length_norm={:.4}, curvature_norm={:.4}, cosine_norm={:.4}, n_steps_norm={:.4}",
            MODE_NAMES[mode], s[0], s[1], s[2], s[3]
        );
    }
    // Print dot products with direction_0 for test summaries (centered).
    let mut dot_a_sum = 0.0_f32;
    let mut dot_b_sum = 0.0_f32;
    for tok_idx in N_TRAIN..N_TOKENS {
        let mut da = 0.0_f32;
        let mut db = 0.0_f32;
        for j in 0..D {
            let sa = summaries[0][tok_idx][j] - global_centroid[j];
            let sb = summaries[1][tok_idx][j] - global_centroid[j];
            da += sa * directions[0][j];
            db += sb * directions[0][j];
        }
        dot_a_sum += da;
        dot_b_sum += db;
    }
    let n_test = (N_TOKENS - N_TRAIN) as f32;
    println!(
        "   mean dot(model_a_test, dir_0) = {:.4}",
        dot_a_sum / n_test
    );
    println!(
        "   mean dot(model_b_test, dir_0) = {:.4}",
        dot_b_sum / n_test
    );
    println!();

    // ── G1: directions non-degenerate ────────────────────────────────
    println!("── G1: directions non-degenerate ──");
    let mut g1_pass = true;
    for k in 0..N {
        let mut norm_sq = 0.0_f32;
        for j in 0..D {
            norm_sq += directions[k][j] * directions[k][j];
        }
        let norm = norm_sq.sqrt();
        let is_unit = (norm - 1.0).abs() < 1e-4;
        if !is_unit {
            println!("   direction {k}: norm={norm:.6} (expected 1.0) ❌");
            g1_pass = false;
        }
    }
    // Pairwise cosine. For N≥3, directions should be distinct (not parallel).
    // For N=2, antiparallel directions (cos = ±1) are MATHEMATICALLY CORRECT:
    // 2 centroids define a line, the midpoint is on it, and the directions
    // from midpoint to each centroid are opposite. This is a valid binary
    // classifier axis, not a degeneracy. The G5 accuracy gate is the
    // load-bearing test; G1 here just verifies the directions are unit-norm.
    let mut dot01 = 0.0_f32;
    for j in 0..D {
        dot01 += directions[0][j] * directions[1][j];
    }
    if N > 2 {
        let distinct = dot01.abs() < 0.99;
        if !distinct {
            println!("   pairwise cosine(dir_0, dir_1) = {dot01:.6} (expected |cos| < 0.99) ❌");
            g1_pass = false;
        }
    } else {
        println!("   N=2: antiparallel directions expected (cos=±1 is valid binary axis)");
    }
    println!("   pairwise cosine(dir_0, dir_1) = {dot01:.6}");
    println!("   G1 verdict: {}", if g1_pass { "PASS" } else { "FAIL" });
    println!();

    // ── Stage 2: Freeze + classify held-out trajectories ────────────────
    let freezer = SweTrajectoryFreezer::<N, D>::with_centroid(directions, global_centroid, encoder);
    let fields: [&dyn ArchetypeFieldSource<D>; N] = make_fields();

    // Test split: tokens [N_TRAIN .. N_TOKENS)
    let test_tokens: Vec<usize> = (N_TRAIN..N_TOKENS).collect();

    println!("── G5: cross-model discrimination on held-out tokens ──");
    println!(
        "  {:>8}  {:>10}  {:>10}  {:>12}  {:>12}  {:>8}",
        "token_idx", "true_mode", "argmax_k", "gate_a", "gate_b", "correct"
    );
    println!("  {}", "-".repeat(76));

    let mut n_correct = 0usize;
    let mut n_total = 0usize;

    // Reusable scratch for freeze_attempt_into (separate from extract scratch
    // since freeze runs at a different point in the loop).
    let mut freeze_disp_curr = Vec::<f32>::with_capacity(d);
    let mut freeze_disp_prev = Vec::<f32>::with_capacity(d);

    for &tok_idx in &test_tokens {
        for mode in 0..N {
            // Re-extract trajectory refs for this token+model.
            let weights = if mode == 0 { &weights_a } else { &weights_b };
            let runtime = if mode == 0 {
                &mut runtime_a
            } else {
                &mut runtime_b
            };
            runtime.reset();
            scratch.traj_buf.clear();
            let _ = kimi_k3_forward_token_traced(
                &config,
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
            let gates = frozen.gates();
            let argmax_k = frozen.argmax_archetype();
            let correct = argmax_k == mode;

            if correct {
                n_correct += 1;
            }
            n_total += 1;

            println!(
                "  {:>8}  {:>10}  {:>10}  {:>12.4}  {:>12.4}  {:>8}",
                tok_idx,
                MODE_NAMES[mode],
                argmax_k,
                gates[0],
                gates[1],
                if correct { "YES" } else { "NO" }
            );
        }
    }

    let accuracy = n_correct as f32 / n_total.max(1) as f32;
    let g5_pass = accuracy >= 0.80;
    println!();
    println!("   accuracy: {accuracy:.2} ({n_correct}/{n_total}) (target ≥0.80)");
    println!("   G5 verdict: {}", if g5_pass { "PASS" } else { "FAIL" });
    println!();

    // ── G2: freeze_attempt latency (real model scale) ────────────────────
    println!("── G2: freeze_attempt latency (D={d} hidden, 9-state trajectory) ──");
    let n_iters = 1000;
    runtime_a.reset();
    scratch.traj_buf.clear();
    let _ = kimi_k3_forward_token_traced(
        &config,
        &weights_a,
        &mut runtime_a,
        tokens[0],
        &mut scratch.traj_buf,
    );
    let refs: Vec<&[f32]> = scratch.traj_buf.iter().map(|v| v.as_slice()).collect();

    // Warmup.
    for _ in 0..100 {
        let _ = freezer.freeze_attempt_into(
            &refs,
            &fields,
            1,
            &mut freeze_disp_curr,
            &mut freeze_disp_prev,
        );
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n_iters {
        let _ = freezer.freeze_attempt_into(
            &refs,
            &fields,
            1,
            &mut freeze_disp_curr,
            &mut freeze_disp_prev,
        );
    }
    let elapsed_ns = t0.elapsed().as_nanos() as u64;
    let per_call_ns = elapsed_ns / n_iters as u64;
    let g2_pass = per_call_ns < 20_000; // 20µs target (real-model D=1024, 9-state depth traj)
    println!("   per_call: {per_call_ns} ns (target < 20000 ns)");
    println!("   G2 verdict: {}", if g2_pass { "PASS" } else { "FAIL" });
    println!();

    // ── Summary ───────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("T5.6 SweTrajectoryFreezer G5 gate (cross-model discrimination):");
    println!("  Model A : {label_a}");
    println!("  Model B : random-137");
    println!(
        "  G1 directions non-degenerate : {}",
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G2 freeze_attempt latency    : {}",
        if g2_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G5 cross-model discrimination: {}",
        if g5_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!();
    let all_pass = g1_pass && g2_pass && g5_pass;
    if all_pass {
        println!("ALL GATES PASS — depth trajectory geometry DOES discriminate");
        println!("across model snapshots on real Kimi-K3 architecture.");
        println!();
        println!("Interpretation: T5.4 found depth trajectories are 29% distinct");
        println!("across TOKENS (input-dependent signal is weak), but they ARE");
        println!("distinct across MODELS (weight-dependent signal is strong). The");
        println!("SweTrajectoryFreezer amplifies this model-specific signal into");
        println!("a usable discrimination gate via data-derived directions + FAME.");
        println!();
        println!("Layer 4 modelless path validated for snapshot/model discrimination.");
    } else {
        let failed: Vec<&str> = [("G1", !g1_pass), ("G2", !g2_pass), ("G5", !g5_pass)]
            .iter()
            .filter(|(_, f)| *f)
            .map(|(g, _)| *g)
            .collect();
        println!("GATES FAILED: {} — see analysis above.", failed.join(", "));
        println!();
        if !g5_pass {
            println!("G5 FAIL: even extreme weight differences (real vs random OR");
            println!("random-42 vs random-137) do not produce discriminative depth");
            println!("trajectory geometry via the SweTrajectoryFreezer pipeline.");
            println!("This challenges the Layer 4 hypothesis for depth trajectories.");
            println!("→ T5.7: document why modelless was insufficient, file Layer 4b");
            println!("  (riir-train LoRA fallback) with explicit §3.5 documentation.");
        }
    }
}

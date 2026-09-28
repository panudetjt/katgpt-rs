//! Proposal 011 Phase 5 T5.4 — Kimi-K3 trajectory geometry on real architecture.
//!
//! Defend-or-refute the T5.4 question: does `latent_trajectory_geometry::from_states`
//! produce sensible, non-degenerate, discriminative geometry when applied to
//! per-layer hidden states from the real Kimi-K3 architecture (D=1024, 8 layers,
//! MLA/KDA/MoE/attn-res)?
//!
//! This is Option 2 from the T5.4 wiring investigation: bypass `tf_loop` (which
//! is architecturally incompatible with Kimi-K3's hybrid layer type) and extract
//! trajectories directly from the production forward path via
//! `kimi_k3_forward_token_traced`.
//!
//! # What this bench tests
//!
//! 1. **Substrate numerical stability at D=1024** — does `from_states` produce
//!    finite, in-range geometry (no NaN/Inf, curvature in [0, π], cosine in [-1, 1])?
//! 2. **Non-degeneracy** — is the trajectory length > 0 (layers actually transform
//!    the hidden state) and is the geometry non-trivial (not all-same values)?
//! 3. **Discriminative power** — does geometry vary across different token inputs
//!    and across different sequence positions?
//!
//! # Weight strategy
//!
//! Uses **random weights** (real architecture, random parameters) — NOT the 1.5GB
//! `model.safetensors`. This tests the substrate + architecture, not semantic
//! model behavior. Random weights exercise the real MLA/KDA/MoE/attn-res forward
//! path with structurally valid (Xavier-scaled) parameters. The hidden states are
//! "random-ish" but flow through the real decoder stack.
//!
//! This is sufficient for the substrate validation question. Semantic validation
//! (does geometry discriminate real failure modes?) requires real weights — a
//! follow-up gated on model file availability.
//!
//! # Run
//!
//! ```bash
//! cargo bench --manifest-path Cargo.toml \
//!     --features "kimi_k3_loader latent_trajectory_geometry" \
//!     --bench bench_012_kimi_k3_trajectory_geometry -- --nocapture
//! ```

#![cfg(feature = "kimi_k3_loader")]
#![allow(clippy::needless_range_loop)]

use katgpt_attn::gdn2::kda_forward::KdaWeights;
use katgpt_attn::mla::MlaWeights;
use katgpt_core::latent_trajectory_geometry::{LatentTrajectoryGeometry, from_states};
use katgpt_rs::kimi_k3::decoder_layer::{
    KimiAttentionWeights, KimiDecoderLayerWeights, KimiFfnConfig, KimiFfnWeights,
};
use katgpt_rs::kimi_k3::loader::{KimiK3ModelWeights, load_kimi_k3};
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token_traced};
use katgpt_transformer::attn_res::AttnResWeights;
use katgpt_transformer::moe::{MoeWeights, SwiGluExpertWeights};

// ─── Random weights constructor ────────────────────────────────────────────

/// Truncated vocab size for the bench. Only tokens with ID < this value are
/// valid. Keeps the embedding table small (512 × 1024 = 2 MB instead of
/// 163840 × 1024 = 670 MB).
const BENCH_VOCAB: usize = 512;

/// LCG RNG — deterministic, no external dep.
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
        (((self.0 >> 40) as f32) / ((1u64 << 24) as f32)) * 2.0 - 1.0
    }
    /// Xavier-scaled random vector of length `n`.
    fn xavier_vec(&mut self, n: usize, fan_in: usize) -> Vec<f32> {
        let scale = 1.0 / (fan_in as f32).sqrt().max(1e-6);
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

/// Build a full `KimiK3ModelWeights` with random but structurally valid weights.
///
/// Uses the real Kimi-K3-0.40B config (D=1024, 8 layers, MLA/KDA/MoE/attn-res).
/// Embedding is truncated to `BENCH_VOCAB` rows; lm_head is empty (the traced
/// forward skips it).
fn build_random_weights(config: &KimiK3ModelConfig, seed: u64) -> KimiK3ModelWeights {
    let d = config.hidden_size;
    let mut rng = Lcg::new(seed);

    // Embedding: [BENCH_VOCAB × hidden_size], Xavier-scaled.
    let embed_weight = rng.xavier_vec(BENCH_VOCAB * d, d);

    // Per-layer decoder weights.
    let layers: Vec<KimiDecoderLayerWeights> = (0..config.num_layers)
        .map(|layer_idx| {
            let layer_seed = seed.wrapping_add((layer_idx as u64) * 7919);

            // RMSNorm gammas — near-1.0 (standard initialization).
            let input_layernorm_weight: Vec<f32> =
                (0..d).map(|_| 1.0 + rng.next_f32() * 0.02).collect();
            let post_attention_layernorm_weight: Vec<f32> =
                (0..d).map(|_| 1.0 + rng.next_f32() * 0.02).collect();

            // Attention: MLA (layers 3, 7) or KDA (others).
            let is_mla = layer_idx == 3 || layer_idx == 7;
            let attention = if is_mla {
                KimiAttentionWeights::Mla(MlaWeights::random(&config.mla_config, layer_seed))
            } else {
                KimiAttentionWeights::Kda(KdaWeights::random(&config.kda_config, layer_seed))
            };

            // FFN: Dense (layer 0) or MoE (layers 1-7).
            let is_dense = layer_idx == 0;
            let ffn = if is_dense {
                // Dense layer uses SiTU MLP. Get intermediate_size from config.
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

            // Attn-res weights.
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

    // Final norm — ones (identity-ish).
    let final_norm_weight: Vec<f32> = (0..d).map(|_| 1.0).collect();

    // LM head — empty (traced forward skips it).
    let lm_head_weight: Vec<f32> = Vec::new();

    // Output attn-res.
    let output_attn_res = AttnResWeights::random(d, seed.wrapping_add(999));

    KimiK3ModelWeights {
        embed_weight,
        layers,
        final_norm_weight,
        lm_head_weight,
        output_attn_res,
    }
}

// ─── Geometry validation ───────────────────────────────────────────────────

#[derive(Clone, Debug)]
#[allow(dead_code)] // token_id/seq_pos are diagnostic fields
struct TokenGeometry {
    token_id: u32,
    seq_pos: usize,
    geom: LatentTrajectoryGeometry,
    all_finite: bool,
}

/// Validate that geometry values are finite and in expected ranges.
fn validate_geometry(geom: &LatentTrajectoryGeometry) -> bool {
    geom.length.is_finite()
        && geom.mean_curvature.is_finite()
        && geom.min_adjacent_cosine.is_finite()
        && geom.mean_curvature >= 0.0
        && geom.mean_curvature <= std::f32::consts::PI + 1e-5
        && geom.min_adjacent_cosine >= -1.0 - 1e-5
        && geom.min_adjacent_cosine <= 1.0 + 1e-5
}

/// Compute geometry statistics across a set of token geometries.
struct GeometryStats {
    mean_length: f32,
    std_length: f32,
    mean_curvature: f32,
    std_curvature: f32,
    min_length: f32,
    max_length: f32,
    n_distinct_pairs: usize,
    n_total_pairs: usize,
}

fn compute_stats(geoms: &[TokenGeometry]) -> GeometryStats {
    let n = geoms.len();
    let lengths: Vec<f32> = geoms.iter().map(|g| g.geom.length).collect();
    let curvs: Vec<f32> = geoms.iter().map(|g| g.geom.mean_curvature).collect();

    let mean_length = lengths.iter().sum::<f32>() / n as f32;
    let std_length = (lengths
        .iter()
        .map(|&l| (l - mean_length).powi(2))
        .sum::<f32>()
        / n as f32)
        .sqrt();

    let mean_curvature = curvs.iter().sum::<f32>() / n as f32;
    let std_curvature = (curvs
        .iter()
        .map(|&c| (c - mean_curvature).powi(2))
        .sum::<f32>()
        / n as f32)
        .sqrt();

    // Count distinct pairs: curvature differs by > 0.1 rad OR length differs by > 20%.
    let mut n_distinct = 0;
    let mut n_total = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            n_total += 1;
            let dc = (geoms[i].geom.mean_curvature - geoms[j].geom.mean_curvature).abs();
            let max_l = geoms[i].geom.length.max(geoms[j].geom.length).max(1e-6);
            let rl = (geoms[i].geom.length - geoms[j].geom.length).abs() / max_l;
            if dc > 0.1 || rl > 0.20 {
                n_distinct += 1;
            }
        }
    }

    GeometryStats {
        mean_length,
        std_length,
        mean_curvature,
        std_curvature,
        min_length: lengths.iter().cloned().fold(f32::INFINITY, f32::min),
        max_length: lengths.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        n_distinct_pairs: n_distinct,
        n_total_pairs: n_total,
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Proposal 011 T5.4 — Kimi-K3 Trajectory Geometry (real arch)      ║");
    println!("║  Option 2: bypass tf_loop, extract from production forward         ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let d = config.hidden_size;
    let n_layers = config.num_layers;
    println!("Config: D={d}, layers={n_layers}, MLA@[3,7], KDA@[0..6], MoE@[1..7]");

    // ── Load weights: real model.safetensors if available, else random ────
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    });
    let model_path = format!("{model_dir}/model.safetensors");
    let use_real = std::path::Path::new(&model_path).exists();

    let weights = if use_real {
        println!("Weights: REAL model.safetensors ({model_path})");
        print!("Loading ... ");
        let t0 = std::time::Instant::now();
        let w = load_kimi_k3(&model_path).unwrap_or_else(|e| {
            eprintln!("\n❌ load failed: {e}");
            std::process::exit(1);
        });
        println!("done ({:.1}s)", t0.elapsed().as_secs_f64());
        w
    } else {
        println!("Weights: random (Xavier-scaled, seed=42) — NOT real model.safetensors");
        print!("Building random Kimi-K3 weights ... ");
        let w = build_random_weights(&config, 42);
        println!(
            "done (embed={}k floats, {} layers)",
            w.embed_weight.len() / 1024,
            w.layers.len()
        );
        w
    };
    println!();

    // ── Create runtime ────────────────────────────────────────────────────
    let max_seq_len = 64;
    let mut runtime = KimiK3Runtime::new(&config, max_seq_len);

    // ── Test 1: per-token depth trajectory (single token, no context) ─────
    println!();
    println!("── Test 1: per-token depth trajectory (no KV context) ────────");
    println!("  Each token processed in isolation (reset → 1 traced forward).");
    println!(
        "  Trajectory = [embed → layer0 → ... → layer{}] ({} states, D={})",
        n_layers - 1,
        n_layers + 1,
        d
    );
    println!();

    let test_tokens: Vec<u32> = vec![1, 2, 5, 10, 42, 100, 200, 500];
    let mut geoms_t1: Vec<TokenGeometry> = Vec::with_capacity(test_tokens.len());

    println!(
        "  {:>8}  {:>10}  {:>12}  {:>12}  {:>12}  {:>8}",
        "token", "n_steps", "length", "mean_curv", "min_cos", "finite"
    );
    println!("  {}", "-".repeat(70));

    for &tok in &test_tokens {
        runtime.reset();
        let mut traj: Vec<Vec<f32>> = Vec::new();
        let _hidden = kimi_k3_forward_token_traced(&config, &weights, &mut runtime, tok, &mut traj);

        let refs: Vec<&[f32]> = traj.iter().map(|v| v.as_slice()).collect();
        let geom = from_states(&refs);
        let all_finite = validate_geometry(&geom);

        geoms_t1.push(TokenGeometry {
            token_id: tok,
            seq_pos: 0,
            geom,
            all_finite,
        });

        println!(
            "  {:>8}  {:>10}  {:>12.4}  {:>12.4}  {:>12.4}  {:>8}",
            tok,
            geom.n_steps,
            geom.length,
            geom.mean_curvature,
            geom.min_adjacent_cosine,
            if all_finite { "YES" } else { "NO" }
        );
    }

    let stats_t1 = compute_stats(&geoms_t1);
    println!();
    println!(
        "  length:   mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
        stats_t1.mean_length, stats_t1.std_length, stats_t1.min_length, stats_t1.max_length
    );
    println!(
        "  curvature: mean={:.4} rad, std={:.4} rad",
        stats_t1.mean_curvature, stats_t1.std_curvature
    );
    println!(
        "  distinct pairs: {}/{} ({:.0}%)",
        stats_t1.n_distinct_pairs,
        stats_t1.n_total_pairs,
        if stats_t1.n_total_pairs > 0 {
            stats_t1.n_distinct_pairs as f32 / stats_t1.n_total_pairs as f32 * 100.0
        } else {
            0.0
        }
    );

    // ── Test 2: trajectory variation across sequence positions ───────────
    println!();
    println!("── Test 2: trajectory variation across sequence positions ────");
    println!("  Same token sequence, traced forward at each position.");
    println!("  Tests whether KV context affects trajectory geometry.");
    println!();

    let prompt: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let mut geoms_t2: Vec<TokenGeometry> = Vec::with_capacity(prompt.len());

    println!(
        "  {:>8}  {:>8}  {:>10}  {:>12}  {:>12}  {:>12}  {:>8}",
        "seq_pos", "token", "n_steps", "length", "mean_curv", "min_cos", "finite"
    );
    println!("  {}", "-".repeat(80));

    runtime.reset();
    for (pos, &tok) in prompt.iter().enumerate() {
        let mut traj: Vec<Vec<f32>> = Vec::new();
        let _hidden = kimi_k3_forward_token_traced(&config, &weights, &mut runtime, tok, &mut traj);

        let refs: Vec<&[f32]> = traj.iter().map(|v| v.as_slice()).collect();
        let geom = from_states(&refs);
        let all_finite = validate_geometry(&geom);

        geoms_t2.push(TokenGeometry {
            token_id: tok,
            seq_pos: pos,
            geom,
            all_finite,
        });

        println!(
            "  {:>8}  {:>8}  {:>10}  {:>12.4}  {:>12.4}  {:>12.4}  {:>8}",
            pos,
            tok,
            geom.n_steps,
            geom.length,
            geom.mean_curvature,
            geom.min_adjacent_cosine,
            if all_finite { "YES" } else { "NO" }
        );
    }

    let stats_t2 = compute_stats(&geoms_t2);
    println!();
    println!(
        "  length:   mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
        stats_t2.mean_length, stats_t2.std_length, stats_t2.min_length, stats_t2.max_length
    );
    println!(
        "  curvature: mean={:.4} rad, std={:.4} rad",
        stats_t2.mean_curvature, stats_t2.std_curvature
    );

    // ── Test 3: trajectory with untraced prefix (realistic context) ──────
    println!();
    println!("── Test 3: traced forward after 16-token context prefix ──────");
    println!("  Process 16 tokens untraced (build KV cache), then trace.");
    println!("  Tests realistic generation context.");
    println!();

    let context: Vec<u32> = (1..=16).collect();
    let trace_tokens: Vec<u32> = vec![100, 200, 300];

    let mut geoms_t3: Vec<TokenGeometry> = Vec::with_capacity(trace_tokens.len());

    println!(
        "  {:>8}  {:>8}  {:>10}  {:>12}  {:>12}  {:>12}  {:>8}",
        "seq_pos", "token", "n_steps", "length", "mean_curv", "min_cos", "finite"
    );
    println!("  {}", "-".repeat(80));

    runtime.reset();
    // Build context with traced forward (discarding trajectory). The traced
    // variant skips the LM head, so it works with both empty (random) and
    // populated (real) lm_head_weight.
    let mut dummy_traj: Vec<Vec<f32>> = Vec::new();
    for &tok in &context {
        let _ = kimi_k3_forward_token_traced(&config, &weights, &mut runtime, tok, &mut dummy_traj);
    }
    // Trace subsequent tokens.
    for (i, &tok) in trace_tokens.iter().enumerate() {
        let mut traj: Vec<Vec<f32>> = Vec::new();
        let _hidden = kimi_k3_forward_token_traced(&config, &weights, &mut runtime, tok, &mut traj);

        let refs: Vec<&[f32]> = traj.iter().map(|v| v.as_slice()).collect();
        let geom = from_states(&refs);
        let all_finite = validate_geometry(&geom);

        geoms_t3.push(TokenGeometry {
            token_id: tok,
            seq_pos: context.len() + i,
            geom,
            all_finite,
        });

        println!(
            "  {:>8}  {:>8}  {:>10}  {:>12.4}  {:>12.4}  {:>12.4}  {:>8}",
            context.len() + i,
            tok,
            geom.n_steps,
            geom.length,
            geom.mean_curvature,
            geom.min_adjacent_cosine,
            if all_finite { "YES" } else { "NO" }
        );
    }

    let stats_t3 = compute_stats(&geoms_t3);
    println!();
    println!(
        "  length:   mean={:.4}, std={:.4}",
        stats_t3.mean_length, stats_t3.std_length
    );
    println!(
        "  curvature: mean={:.4} rad, std={:.4} rad",
        stats_t3.mean_curvature, stats_t3.std_curvature
    );

    // ── Verdicts ──────────────────────────────────────────────────────────
    println!();
    println!("──────────────────────────────────────────────────────────────────");
    println!("┌───────┬──────────────────────────────────────────────────┬────────┐");
    println!("│ Gate  │ Claim                                             │ Verdict│");
    println!("├───────┼──────────────────────────────────────────────────┼────────┤");

    // G1: all finite + in-range.
    let all_finite = geoms_t1
        .iter()
        .chain(&geoms_t2)
        .chain(&geoms_t3)
        .all(|g| g.all_finite);
    print_verdict("G1", "all geometry finite + in-range (D=1024)", all_finite);

    // G2: non-degenerate (length > 0 — layers actually transform hidden).
    let non_degenerate = geoms_t1
        .iter()
        .chain(&geoms_t2)
        .chain(&geoms_t3)
        .all(|g| g.geom.length > 0.0);
    print_verdict("G2", "non-degenerate (length > 0)", non_degenerate);

    // G3: discriminative across tokens (Test 1).
    let discrim_t1 = stats_t1.n_distinct_pairs as f32 / stats_t1.n_total_pairs.max(1) as f32;
    let g3_pass = discrim_t1 > 0.3;
    print_verdict(
        "G3",
        "discriminative across tokens (>30% distinct)",
        g3_pass,
    );

    // G4: varies across sequence positions (Test 2).
    let varies_pos = stats_t2.std_length > 0.0 || stats_t2.std_curvature > 0.0;
    print_verdict("G4", "varies across sequence positions", varies_pos);

    println!("└───────┴──────────────────────────────────────────────────┴────────┘");
    println!();

    // ── Overall verdict ───────────────────────────────────────────────────
    let all_pass = all_finite && non_degenerate && g3_pass && varies_pos;
    if all_pass {
        println!(
            "═ T5.4 VALIDATED on {} Kimi-K3 weights ═",
            if use_real { "REAL" } else { "random" }
        );
        println!();
        println!("`latent_trajectory_geometry::from_states` produces finite,");
        println!("non-degenerate, discriminative geometry at D=1024 on the real");
        println!("MLA/KDA/MoE/attn-res decoder stack. The substrate works at");
        println!("production-model scale.");
        if use_real {
            println!();
            println!("Next: T5.5 (SweTrajectoryFreezer impl) — the substrate is");
            println!("validated on real model hidden states.");
        } else {
            println!();
            println!("NOTE: random weights pass all gates including discrimination.");
            println!("For stronger validation, re-run with real model.safetensors.");
        }
    } else {
        let failed: Vec<&str> = [
            ("G1", !all_finite),
            ("G2", !non_degenerate),
            ("G3", !g3_pass),
            ("G4", !varies_pos),
        ]
        .iter()
        .filter(|(_, f)| *f)
        .map(|(g, _)| *g)
        .collect();
        println!(
            "═ T5.4 PARTIAL ({}) — {} failed: {} ═",
            if use_real {
                "REAL weights"
            } else {
                "random weights"
            },
            failed.len(),
            failed.join(", ")
        );
        println!();
        if !all_finite {
            println!("  G1 FAIL: NaN/Inf or out-of-range geometry at D=1024.");
            println!("    This indicates a numerical issue in from_states at scale.");
        }
        if !non_degenerate {
            println!("  G2 FAIL: zero-length trajectory — layers are identity.");
            println!("    Check if weights produce valid forward passes.");
        }
        if !g3_pass {
            println!(
                "  G3 FAIL: geometry does not discriminate across tokens ({:.0}%).",
                discrim_t1 * 100.0
            );
            if !use_real {
                println!("    EXPECTED with random weights — random transformations are");
                println!("    input-invariant. Re-test with real model.safetensors to");
                println!("    validate semantic discrimination.");
            } else {
                println!("    Real weights still don't discriminate — this challenges");
                println!("    the Layer 4 hypothesis for depth-wise trajectories.");
            }
        }
        if !varies_pos {
            println!("  G4 FAIL: geometry does not vary across sequence positions.");
        }
    }
}

fn print_verdict(gate: &str, claim: &str, pass: bool) {
    let verdict = if pass { "✅ PASS" } else { "❌ FAIL" };
    println!("│ {gate:5} │ {claim:<50} │ {verdict} │");
}

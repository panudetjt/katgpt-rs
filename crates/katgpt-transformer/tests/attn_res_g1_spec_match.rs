//! G1 spec-match tests for the Attention Residual Block (Kimi-K3).
//!
//! These tests verify the f32 `apply_attn_res` implementation against an
//! independent f64 reference written directly from the actual
//! `modeling_kimi_k3_linear.py` `_apply_attn_res` function (Research 330 §5).
//!
//! The f64 reference is written WITHOUT reusing the f32 impl's code — it's a
//! from-scratch implementation of the same math. If both agree within f32
//! precision, we have confidence the f32 impl correctly encodes the equations.
//!
//! # What's tested
//!
//! The reference mirrors the actual model code:
//! - Concatenation: `v = [block_residual..., prefix_sum]`
//! - RMSNorm WITHOUT gamma (gamma is folded into score_weight)
//! - Score weight: `score_weight = norm_weight ⊙ proj_weight`
//! - Scores: `scores[i] = dot(rmsnorm(v[i]), score_weight)`
//! - Softmax mixing: `out = Σ softmax(scores)[i] · v[i]` (original v, not normed)
//!
//! The TRUE correctness gate is Phase 6 (logits match real PyTorch weights).
//! This G1 gate catches transcription errors in our own code before we get to
//! Phase 6.

#![cfg(feature = "transformer_attn_res")]

use katgpt_transformer::attn_res::{
    AttnResBlockState, AttnResConfig, AttnResScratch, AttnResWeights, apply_attn_res,
};

// ─── f64 weight mirror ─────────────────────────────────────────────────────

/// f64 mirror of `AttnResWeights`.
struct AttnResWeightsF64 {
    norm_weight: Vec<f64>,
    proj_weight: Vec<f64>,
}

fn weights_to_f64(w: &AttnResWeights) -> AttnResWeightsF64 {
    AttnResWeightsF64 {
        norm_weight: w.norm_weight.iter().map(|&x| x as f64).collect(),
        proj_weight: w.proj_weight.iter().map(|&x| x as f64).collect(),
    }
}

// ─── f64 reference implementation ──────────────────────────────────────────

/// Independent f64 reference for `_apply_attn_res`.
///
/// Written from scratch from the PyTorch pseudocode (Research 330 §5):
/// ```python
/// def _apply_attn_res(prefix_sum, block_residual, proj, norm):
///     v = cat(block_residual, prefix_sum.unsqueeze(1))
///     k = v * rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
///     score_weight = norm.weight * proj.weight.squeeze(0)
///     scores = (k * score_weight).sum(-1)
///     probs = scores.softmax(-1).unsqueeze(1)
///     return matmul(probs, v).squeeze(1)
/// ```
#[allow(clippy::needless_range_loop)]
fn apply_attn_res_f64(
    hidden_size: usize,
    eps: f64,
    weights: &AttnResWeightsF64,
    block_residuals: &[Vec<f64>],
    prefix_sum: &[f64],
) -> Vec<f64> {
    let d = hidden_size;

    // Build v: [block_residuals..., prefix_sum]
    let num_entries = block_residuals.len() + 1;
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(num_entries);
    for r in block_residuals {
        v.push(r.clone());
    }
    v.push(prefix_sum.to_vec());

    // Compute score_weight = norm_weight ⊙ proj_weight
    let mut score_weight = vec![0.0; d];
    for i in 0..d {
        score_weight[i] = weights.norm_weight[i] * weights.proj_weight[i];
    }

    // Compute scores: scores[i] = dot(rmsnorm(v[i]), score_weight)
    let mut scores = vec![0.0; num_entries];
    for i in 0..num_entries {
        // RMSNorm (no gamma): rms = sqrt(mean(v[i]^2) + eps)
        let mut sum_sq = 0.0;
        for j in 0..d {
            sum_sq += v[i][j] * v[i][j];
        }
        let inv_rms = 1.0 / (sum_sq / d as f64 + eps).sqrt();

        // score = dot(v[i] / rms, score_weight) = inv_rms * dot(v[i], score_weight)
        let mut dot = 0.0;
        for j in 0..d {
            dot += v[i][j] * score_weight[j];
        }
        scores[i] = inv_rms * dot;
    }

    // Softmax over scores
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exp_sum = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
        exp_sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= exp_sum;
    }

    // Output: weighted average of ORIGINAL v
    let mut out = vec![0.0; d];
    for i in 0..num_entries {
        let prob = scores[i];
        for j in 0..d {
            out[j] += prob * v[i][j];
        }
    }

    out
}

// ─── Tests ──────────────────────────────────────────────────────────────────

fn max_diff_f32_f64(a: &[f32], b: &[f64]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x as f64) - y).abs() as f32)
        .fold(0.0f32, f32::max)
}

#[test]
fn g1_attn_res_matches_reference_empty_block() {
    // No block residuals — single entry (prefix_sum only).
    // softmax of 1 element = [1.0], output = prefix_sum.
    let config = AttnResConfig::kimi_k3_0_40b();
    let weights = AttnResWeights::random(config.d(), 42);
    let weights_f64 = weights_to_f64(&weights);

    let block_state = AttnResBlockState::new(config.d());
    let mut scratch = AttnResScratch::new(&config, 8);

    let prefix_sum: Vec<f32> = (0..config.d()).map(|i| (i as f32).sin() * 0.1).collect();
    let prefix_f64: Vec<f64> = prefix_sum.iter().map(|&x| x as f64).collect();

    let out_f32 = apply_attn_res(&config, &weights, &block_state, &mut scratch, &prefix_sum);
    let out_f64 = apply_attn_res_f64(
        config.d(),
        config.rms_eps as f64,
        &weights_f64,
        &[],
        &prefix_f64,
    );

    let diff = max_diff_f32_f64(out_f32, &out_f64);
    assert!(
        diff < 1e-3,
        "G1 FAIL empty block: max_diff = {diff:.2e} (tol 1e-3)"
    );
    eprintln!("g1_attn_res_empty_block: max_diff = {diff:.2e}");
}

#[test]
fn g1_attn_res_matches_reference_one_block_entry() {
    let config = AttnResConfig::kimi_k3_0_40b();
    let weights = AttnResWeights::random(config.d(), 42);
    let weights_f64 = weights_to_f64(&weights);

    let mut block_state = AttnResBlockState::new(config.d());
    let r1: Vec<f32> = (0..config.d()).map(|i| (i as f32).cos() * 0.1).collect();
    block_state.push(&r1);

    let r1_f64: Vec<f64> = r1.iter().map(|&x| x as f64).collect();

    let mut scratch = AttnResScratch::new(&config, 8);

    let prefix_sum: Vec<f32> = (0..config.d()).map(|i| (i as f32).sin() * 0.1).collect();
    let prefix_f64: Vec<f64> = prefix_sum.iter().map(|&x| x as f64).collect();

    let out_f32 = apply_attn_res(&config, &weights, &block_state, &mut scratch, &prefix_sum);
    let out_f64 = apply_attn_res_f64(
        config.d(),
        config.rms_eps as f64,
        &weights_f64,
        &[r1_f64],
        &prefix_f64,
    );

    let diff = max_diff_f32_f64(out_f32, &out_f64);
    assert!(
        diff < 1e-3,
        "G1 FAIL one block entry: max_diff = {diff:.2e} (tol 1e-3)"
    );
    eprintln!("g1_attn_res_one_entry: max_diff = {diff:.2e}");
}

#[test]
fn g1_attn_res_matches_reference_two_block_entries() {
    let config = AttnResConfig::kimi_k3_0_40b();
    let weights = AttnResWeights::random(config.d(), 77);
    let weights_f64 = weights_to_f64(&weights);

    let mut block_state = AttnResBlockState::new(config.d());
    let r1: Vec<f32> = (0..config.d()).map(|i| (i as f32).sin() * 0.1).collect();
    let r2: Vec<f32> = (0..config.d()).map(|i| (i as f32).cos() * 0.1).collect();
    block_state.push(&r1);
    block_state.push(&r2);

    let r1_f64: Vec<f64> = r1.iter().map(|&x| x as f64).collect();
    let r2_f64: Vec<f64> = r2.iter().map(|&x| x as f64).collect();

    let mut scratch = AttnResScratch::new(&config, 8);

    let prefix_sum: Vec<f32> = vec![0.05; config.d()];
    let prefix_f64: Vec<f64> = prefix_sum.iter().map(|&x| x as f64).collect();

    let out_f32 = apply_attn_res(&config, &weights, &block_state, &mut scratch, &prefix_sum);
    let out_f64 = apply_attn_res_f64(
        config.d(),
        config.rms_eps as f64,
        &weights_f64,
        &[r1_f64, r2_f64],
        &prefix_f64,
    );

    let diff = max_diff_f32_f64(out_f32, &out_f64);
    assert!(
        diff < 1e-3,
        "G1 FAIL two block entries: max_diff = {diff:.2e} (tol 1e-3)"
    );
    eprintln!("g1_attn_res_two_entries: max_diff = {diff:.2e}");
}

#[test]
fn g1_attn_res_matches_reference_kimi_k3_0_40b_dims() {
    // Full 0.40B config dimensions at max block accumulation (2 entries for 8 layers).
    let config = AttnResConfig::kimi_k3_0_40b();
    let weights = AttnResWeights::random(config.d(), 777);
    let weights_f64 = weights_to_f64(&weights);

    let mut block_state = AttnResBlockState::new(config.d());
    let r1: Vec<f32> = (0..config.d())
        .map(|i| ((i + 1) as f32).sin() * 0.01)
        .collect();
    let r2: Vec<f32> = (0..config.d())
        .map(|i| ((i + 2) as f32).cos() * 0.01)
        .collect();
    block_state.push(&r1);
    block_state.push(&r2);

    let r1_f64: Vec<f64> = r1.iter().map(|&x| x as f64).collect();
    let r2_f64: Vec<f64> = r2.iter().map(|&x| x as f64).collect();

    let mut scratch = AttnResScratch::new(&config, 8);

    let prefix_sum: Vec<f32> = (0..config.d())
        .map(|i| ((i + 3) as f32).sin() * 0.01)
        .collect();
    let prefix_f64: Vec<f64> = prefix_sum.iter().map(|&x| x as f64).collect();

    let out_f32 = apply_attn_res(&config, &weights, &block_state, &mut scratch, &prefix_sum);
    let out_f64 = apply_attn_res_f64(
        config.d(),
        config.rms_eps as f64,
        &weights_f64,
        &[r1_f64, r2_f64],
        &prefix_f64,
    );

    let diff = max_diff_f32_f64(out_f32, &out_f64);
    assert!(
        diff < 1e-3,
        "G1 FAIL 0.40B dims: max_diff = {diff:.2e} (tol 1e-3)"
    );
    eprintln!("g1_attn_res_kimi_k3_0_40b_dims: max_diff = {diff:.2e}");
}

#[test]
fn g1_softmax_weights_sum_to_one() {
    // The output probabilities must sum to 1.0 (softmax property).
    // We verify this indirectly: the output must be a convex combination
    // of the input entries.
    let config = AttnResConfig::kimi_k3_0_40b();
    let weights = AttnResWeights::random(config.d(), 42);

    let mut block_state = AttnResBlockState::new(config.d());
    let r1: Vec<f32> = vec![1.0; config.d()];
    let r2: Vec<f32> = vec![3.0; config.d()];
    block_state.push(&r1);
    block_state.push(&r2);

    let mut scratch = AttnResScratch::new(&config, 8);

    let prefix_sum: Vec<f32> = vec![2.0; config.d()];

    let out = apply_attn_res(&config, &weights, &block_state, &mut scratch, &prefix_sum);

    // All entries are in {1.0, 2.0, 3.0}. Convex combination must be in [1.0, 3.0].
    for &v in out.iter() {
        assert!(
            (1.0 - 1e-5..=3.0 + 1e-5).contains(&v),
            "output {v} outside convex hull [1, 3]"
        );
    }
}

#[test]
fn g1_output_uses_original_v_not_normed_k() {
    // Verify the output is a weighted combination of the ORIGINAL values,
    // not the RMSNorm'd values. If we set norm_weight=ones, proj_weight=ones,
    // then score_weight=ones, and each score = dot(rmsnorm(v[i]), ones) = sum(rmsnorm(v[i])) / d.
    // The output is NOT the rmsnorm'd values — it's the original values weighted
    // by the softmax of those scores.
    let config = AttnResConfig {
        hidden_size: 4,
        block_size: 4,
        rms_eps: 1e-5,
    };
    let weights = AttnResWeights {
        norm_weight: vec![1.0; 4],
        proj_weight: vec![1.0; 4],
    };

    let mut block_state = AttnResBlockState::new(4);
    let r1 = vec![10.0, 0.0, 0.0, 0.0]; // large magnitude in one dim
    block_state.push(&r1);

    let mut scratch = AttnResScratch::new(&config, 8);
    let prefix_sum = vec![0.0, 0.0, 0.0, 1.0]; // different magnitude distribution

    let out = apply_attn_res(&config, &weights, &block_state, &mut scratch, &prefix_sum);

    // If output used normed k, dim 0 of r1 would be ~1.0 (rmsnorm of [10,0,0,0]
    // gives ~2.0 for dim 0). But the output uses original v, so dim 0 of r1
    // is 10.0 * prob[0]. Since both entries contribute, dim 0 will be > 1.0
    // if prob[0] > 0.1 (which it will be, since r1 has higher RMS score).
    assert!(
        out[0] > 1.0,
        "output[0]={} should be >1.0 if using original v (10.0 * prob[0])",
        out[0]
    );
}

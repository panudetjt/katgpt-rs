//! G1 spec-match tests for MLA (Multi-head Latent Attention).
//!
//! These tests verify the f32 `mla_forward_token` implementation against an
//! independent f64 reference written directly from the DeepSeek-V2 equations
//! (Research 327 §2, paper Appendix C, equations 37–47).
//!
//! The f64 reference is written WITHOUT reusing the f32 impl's code — it's a
//! from-scratch implementation of the same math. If both agree within f32
//! precision, we have confidence the f32 impl correctly encodes the equations.
//!
//! The TRUE correctness gate is Phase 6 (logits match real PyTorch weights).
//! This G1 gate catches transcription errors in our own code before we get to
//! Phase 6.

#![cfg(feature = "mla_attention")]

use katgpt_attn::mla::{MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights, mla_forward_token};
use katgpt_kv::shard_kv::rope::RopeFreqs;

// ─── f64 reference implementation ───────────────────────────────────────────
// Completely independent of the f32 impl. Implements equations 37–47 from
// DeepSeek-V2 Appendix C (Research 327 §2) using plain f64 arithmetic.

/// RoPE application on f64: rotate dim pairs at the given position.
/// `inv_freq[i] = 1.0 / (theta ^ (2i / d_r))` for i in 0..d_r/2.
fn rope_f64(x: &mut [f64], pos: usize, d_r: usize, theta: f64) {
    let half = d_r / 2;
    for i in 0..half {
        let exp = 2.0 * i as f64 / d_r as f64;
        let inv_freq = 1.0 / theta.powf(exp);
        let theta_angle = pos as f64 * inv_freq;
        let (sin_t, cos_t) = theta_angle.sin_cos();
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = cos_t * x0 - sin_t * x1;
        x[2 * i + 1] = sin_t * x0 + cos_t * x1;
    }
}

/// f64 matmul: `out = W · x` where W is `[rows][cols]` row-major.
fn matmul_f64(out: &mut [f64], w: &[f64], x: &[f64], rows: usize, cols: usize) {
    for r in 0..rows {
        let mut sum = 0.0;
        for c in 0..cols {
            sum += w[r * cols + c] * x[c];
        }
        out[r] = sum;
    }
}

/// f64 dot product.
fn dot_f64(a: &[f64], b: &[f64], len: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// f64 RMSNorm with gamma, applied in-place.
/// Mirrors the actual model's `KimiRMSNorm` (Research 330 §2).
fn rmsnorm_f64_inplace(x: &mut [f64], gamma: &[f64], eps: f64) {
    let n = x.len();
    let sum_sq = dot_f64(x, x, n);
    let inv_rms = 1.0 / (sum_sq / n as f64 + eps).sqrt();
    for i in 0..n {
        x[i] = x[i] * inv_rms * gamma[i];
    }
}

/// Independent f64 reference for the full MLA forward pass.
///
/// Matches the actual `modeling_kimi_k3_linear.py` forward path (Research 330
/// §2): applies `q_a_layernorm` to the query latent and `kv_a_layernorm` to the
/// KV latent BEFORE up-projection + caching. The normed latent is what gets
/// cached + up-projected.
fn mla_forward_f64_reference(
    config: &MlaConfig,
    weights_f64: &MlaWeightsF64,
    tokens: &[Vec<f64>],
) -> Vec<f64> {
    let d = config.hidden_size;
    let d_c = config.kv_lora_rank;
    let d_qc = config.q_lora_rank;
    let d_h = config.qk_nope_head_dim;
    let d_r = config.qk_rope_head_dim;
    let v_h = config.v_head_dim;
    let n_h = config.n_heads;
    let theta = config.rope_theta as f64;
    let eps = config.rms_norm_eps as f64;
    let scale = 1.0 / ((d_h + d_r) as f64).sqrt();

    // Cache: store per-token normed latent + shared rope key.
    let mut c_kv_cache: Vec<Vec<f64>> = Vec::new();
    let mut k_r_cache: Vec<Vec<f64>> = Vec::new();

    let mut output = vec![0.0f64; d];

    for (pos, h) in tokens.iter().enumerate() {
        // ── Step 1: Down-projections + latent RMSNorm ───────────────────────
        let mut c_kv = vec![0.0; d_c];
        let mut c_q = vec![0.0; d_qc];
        matmul_f64(&mut c_kv, &weights_f64.w_dkv, h, d_c, d);
        matmul_f64(&mut c_q, &weights_f64.w_dq, h, d_qc, d);
        // Apply latent RMSNorms (actual model: q_a_layernorm, kv_a_layernorm)
        rmsnorm_f64_inplace(&mut c_q, &weights_f64.q_a_norm_weight, eps);
        rmsnorm_f64_inplace(&mut c_kv, &weights_f64.kv_a_norm_weight, eps);

        // ── Step 2: Query up-projections + decoupled RoPE ─────────────────
        let mut q_c = vec![0.0; d_h * n_h];
        let mut q_r = vec![0.0; d_r * n_h];
        matmul_f64(&mut q_c, &weights_f64.w_uq, &c_q, d_h * n_h, d_qc);
        matmul_f64(&mut q_r, &weights_f64.w_qr, &c_q, d_r * n_h, d_qc);
        if !config.use_nope {
            for head in 0..n_h {
                let start = head * d_r;
                rope_f64(&mut q_r[start..start + d_r], pos, d_r, theta);
            }
        }

        // ── Step 3: Shared RoPE key (NOT normed — outside kv_a_layernorm) ──
        let mut k_r = vec![0.0; d_r];
        matmul_f64(&mut k_r, &weights_f64.w_kr, h, d_r, d);
        if !config.use_nope {
            rope_f64(&mut k_r, pos, d_r, theta);
        }

        // Cache this token's normed latent + rope key.
        c_kv_cache.push(c_kv.clone());
        k_r_cache.push(k_r.clone());

        // ── Step 4: Attention per head ─────────────────────────────────────
        // Up-project k_c/v_c from the normed latent at attention time.
        let seq = c_kv_cache.len();
        let mut attn_out = vec![0.0f64; v_h * n_h];

        for head in 0..n_h {
            let q_c_h = &q_c[head * d_h..(head + 1) * d_h];
            let q_r_h = &q_r[head * d_r..(head + 1) * d_r];

            let mut scores = vec![0.0f64; seq];
            let mut max_score = f64::NEG_INFINITY;
            for j in 0..seq {
                // k_c_j_h = W_UK[head] · c_kv_j  (normed latent)
                let mut k_c_h = vec![0.0; d_h];
                matmul_f64(
                    &mut k_c_h,
                    &weights_f64.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
                    &c_kv_cache[j],
                    d_h,
                    d_c,
                );
                let k_r_j = &k_r_cache[j];
                let content_dot = dot_f64(q_c_h, &k_c_h, d_h);
                let rope_dot = dot_f64(q_r_h, k_r_j, d_r);
                let score = (content_dot + rope_dot) * scale;
                scores[j] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            let mut sum_exp = 0.0;
            for s in scores.iter_mut().take(seq) {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }

            for (j, &s) in scores.iter().enumerate().take(seq) {
                // v_c_j_h = W_UV[head] · c_kv_j  (normed latent)
                let mut v_c_h = vec![0.0; v_h];
                matmul_f64(
                    &mut v_c_h,
                    &weights_f64.w_uv[head * v_h * d_c..(head + 1) * v_h * d_c],
                    &c_kv_cache[j],
                    v_h,
                    d_c,
                );
                let weight = s / sum_exp;
                for vi in 0..v_h {
                    attn_out[head * v_h + vi] += weight * v_c_h[vi];
                }
            }
        }

        // ── Step 5: Output gate (Kimi-K3 extension) — BEFORE o_proj ────────
        // Gate applied to attn_out (shape [v_h*n_h]) BEFORE output projection.
        // g_proj shape: [v_h*n_h, d].
        let proj_size = v_h * n_h;
        if config.use_output_gate
            && let Some(ref w_g) = weights_f64.w_g
        {
            let mut gate = vec![0.0; proj_size];
            matmul_f64(&mut gate, w_g, h, proj_size, d);
            for i in 0..proj_size {
                attn_out[i] *= 1.0 / (1.0 + (-gate[i]).exp());
            }
        }

        // ── Step 6: Output projection ──────────────────────────────────────
        matmul_f64(&mut output, &weights_f64.w_o, &attn_out, d, proj_size);
    }

    output
}

/// f64 version of MlaWeights for the reference impl.
struct MlaWeightsF64 {
    w_dkv: Vec<f64>,
    w_dq: Vec<f64>,
    w_uq: Vec<f64>,
    w_qr: Vec<f64>,
    w_uk: Vec<f64>,
    w_uv: Vec<f64>,
    w_kr: Vec<f64>,
    w_o: Vec<f64>,
    q_a_norm_weight: Vec<f64>,
    kv_a_norm_weight: Vec<f64>,
    w_g: Option<Vec<f64>>,
}

/// Convert f32 MlaWeights to f64.
fn weights_to_f64(w: &MlaWeights) -> MlaWeightsF64 {
    let conv = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    MlaWeightsF64 {
        w_dkv: conv(&w.w_dkv),
        w_dq: conv(&w.w_dq),
        w_uq: conv(&w.w_uq),
        w_qr: conv(&w.w_qr),
        w_uk: conv(&w.w_uk),
        w_uv: conv(&w.w_uv),
        w_kr: conv(&w.w_kr),
        w_o: conv(&w.w_o),
        q_a_norm_weight: conv(&w.q_a_norm_weight),
        kv_a_norm_weight: conv(&w.kv_a_norm_weight),
        w_g: w.w_g.as_ref().map(|g| conv(g)),
    }
}

// ─── Test helpers ───────────────────────────────────────────────────────────

fn small_config() -> MlaConfig {
    MlaConfig {
        kv_lora_rank: 8,
        q_lora_rank: 12,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 4,
        v_head_dim: 4,
        n_heads: 2,
        hidden_size: 16,
        use_output_gate: true,
        use_nope: false,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    }
}

/// Run the f32 impl on a token sequence and return the final output.
fn run_f32(config: &MlaConfig, weights: &MlaWeights, tokens: &[Vec<f32>]) -> Vec<f32> {
    let max_seq = tokens.len();
    let mut cache = MlaKVCache::new(config, max_seq);
    let mut scratch = MlaForwardScratch::new(config, max_seq);
    let mut rope_freqs = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    let mut last_out = Vec::new();
    for h in tokens {
        let out = mla_forward_token(
            config,
            weights,
            &mut cache,
            &mut scratch,
            &mut rope_freqs,
            h,
        );
        last_out = out.to_vec();
    }
    last_out
}

fn max_abs_diff_f32_f64(f32_out: &[f32], f64_out: &[f64]) -> f64 {
    f32_out
        .iter()
        .zip(f64_out.iter())
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0.0f64, f64::max)
}

// ─── G1 tests ───────────────────────────────────────────────────────────────

#[test]
fn g1_single_token_zero_position_matches_reference() {
    // At position 0, RoPE is identity. Isolates projection + attention math.
    let config = small_config();
    let weights = MlaWeights::random(&config, 42);
    let weights_f64 = weights_to_f64(&weights);

    let h_f32: Vec<f32> = (0..config.hidden_size)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let h_f64: Vec<f64> = h_f32.iter().map(|&v| v as f64).collect();

    let f32_out = run_f32(&config, &weights, &[h_f32]);
    let f64_out = mla_forward_f64_reference(&config, &weights_f64, &[h_f64]);

    let max_diff = max_abs_diff_f32_f64(&f32_out, &f64_out);
    assert!(
        max_diff < 1e-4,
        "G1 FAIL at pos=0: max_diff = {max_diff:.2e} (tol 1e-4)"
    );
    eprintln!("g1_single_token_zero_position: max_diff = {max_diff:.2e}");
}

#[test]
fn g1_single_token_nonzero_position_matches_reference() {
    // Position 5 — RoPE active. Verifies decoupled RoPE (q_r/k_r rotated,
    // q_c/k_c NOT rotated).
    let config = small_config();
    let weights = MlaWeights::random(&config, 7);
    let weights_f64 = weights_to_f64(&weights);

    let mut tokens_f32: Vec<Vec<f32>> = Vec::new();
    for t in 0..6 {
        let h: Vec<f32> = (0..config.hidden_size)
            .map(|i| ((i + t * 3) as f32).sin() * 0.3 + 0.5)
            .collect();
        tokens_f32.push(h);
    }
    let tokens_f64: Vec<Vec<f64>> = tokens_f32
        .iter()
        .map(|h| h.iter().map(|&v| v as f64).collect())
        .collect();

    let f32_out = run_f32(&config, &weights, &tokens_f32);
    let f64_out = mla_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_abs_diff_f32_f64(&f32_out, &f64_out);
    assert!(
        max_diff < 1e-4,
        "G1 FAIL at pos=5 (RoPE active): max_diff = {max_diff:.2e} (tol 1e-4)"
    );
    eprintln!("g1_single_token_nonzero_position: max_diff = {max_diff:.2e}");
}

#[test]
fn g1_multi_token_sequence_matches_reference() {
    // 3 tokens: each attends to its prefix.
    let config = small_config();
    let weights = MlaWeights::random(&config, 99);
    let weights_f64 = weights_to_f64(&weights);

    let tokens_f32: Vec<Vec<f32>> = (0..3)
        .map(|t| {
            (0..config.hidden_size)
                .map(|i| ((i + t * 7) as f32).cos() * 0.4 + 0.1)
                .collect()
        })
        .collect();
    let tokens_f64: Vec<Vec<f64>> = tokens_f32
        .iter()
        .map(|h| h.iter().map(|&v| v as f64).collect())
        .collect();

    let f32_out = run_f32(&config, &weights, &tokens_f32);
    let f64_out = mla_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_abs_diff_f32_f64(&f32_out, &f64_out);
    assert!(
        max_diff < 1e-4,
        "G1 FAIL multi-token: max_diff = {max_diff:.2e} (tol 1e-4)"
    );
    eprintln!("g1_multi_token_sequence: max_diff = {max_diff:.2e}");
}

#[test]
fn g1_output_gate_on_matches_reference() {
    let config = small_config();
    let weights = MlaWeights::random(&config, 11);
    let weights_f64 = weights_to_f64(&weights);

    let h_f32: Vec<f32> = (0..config.hidden_size)
        .map(|i| (i as f32) * 0.05 - 0.4)
        .collect();
    let h_f64: Vec<f64> = h_f32.iter().map(|&v| v as f64).collect();

    let f32_out = run_f32(&config, &weights, &[h_f32]);
    let f64_out = mla_forward_f64_reference(&config, &weights_f64, &[h_f64]);

    let max_diff = max_abs_diff_f32_f64(&f32_out, &f64_out);
    assert!(
        max_diff < 1e-4,
        "G1 FAIL output gate: max_diff = {max_diff:.2e} (tol 1e-4)"
    );
    eprintln!("g1_output_gate_on: max_diff = {max_diff:.2e}");
}

#[test]
fn g1_output_gate_off_matches_reference() {
    let mut config = small_config();
    config.use_output_gate = false;
    let weights = MlaWeights::random(&config, 11);
    let weights_f64 = weights_to_f64(&weights);

    let h_f32: Vec<f32> = (0..config.hidden_size)
        .map(|i| (i as f32) * 0.05 - 0.4)
        .collect();
    let h_f64: Vec<f64> = h_f32.iter().map(|&v| v as f64).collect();

    let f32_out = run_f32(&config, &weights, &[h_f32]);
    let f64_out = mla_forward_f64_reference(&config, &weights_f64, &[h_f64]);

    let max_diff = max_abs_diff_f32_f64(&f32_out, &f64_out);
    assert!(
        max_diff < 1e-4,
        "G1 FAIL ungated: max_diff = {max_diff:.2e} (tol 1e-4)"
    );
    eprintln!("g1_output_gate_off: max_diff = {max_diff:.2e}");
}

#[test]
fn g1_attention_scale_is_sqrt_dh_plus_dr() {
    // Implicitly tested by all above (they'd fail with wrong scale), but made
    // explicit with a high-attention case where scale error is amplified.
    let config = small_config();
    let weights = MlaWeights::random(&config, 123);
    let weights_f64 = weights_to_f64(&weights);

    let tokens_f32: Vec<Vec<f32>> = vec![
        (0..config.hidden_size).map(|i| (i as f32) * 0.1).collect(),
        (0..config.hidden_size)
            .map(|i| 1.0 - (i as f32) * 0.05)
            .collect(),
    ];
    let tokens_f64: Vec<Vec<f64>> = tokens_f32
        .iter()
        .map(|h| h.iter().map(|&v| v as f64).collect())
        .collect();

    let f32_out = run_f32(&config, &weights, &tokens_f32);
    let f64_out = mla_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_abs_diff_f32_f64(&f32_out, &f64_out);
    assert!(
        max_diff < 1e-4,
        "G1 FAIL scale test: max_diff = {max_diff:.2e} — scale may be wrong"
    );
    eprintln!("g1_attention_scale: max_diff = {max_diff:.2e}");
}

#[test]
fn g1_kimi_k3_0_40b_full_dims_match_reference() {
    // Full 0.40B dimensions. Slower but verifies real-scale math.
    let config = MlaConfig::kimi_k3_0_40b();
    let weights = MlaWeights::random(&config, 777);
    let weights_f64 = weights_to_f64(&weights);

    let tokens_f32: Vec<Vec<f32>> = (0..3)
        .map(|t| {
            (0..config.hidden_size)
                .map(|i| ((i + t * 13) as f32).sin() * 0.01)
                .collect()
        })
        .collect();
    let tokens_f64: Vec<Vec<f64>> = tokens_f32
        .iter()
        .map(|h| h.iter().map(|&v| v as f64).collect())
        .collect();

    let f32_out = run_f32(&config, &weights, &tokens_f32);
    let f64_out = mla_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_abs_diff_f32_f64(&f32_out, &f64_out);
    // Looser tolerance for full 1024-dim (accumulated float error).
    assert!(
        max_diff < 1e-3,
        "G1 FAIL 0.40B dims: max_diff = {max_diff:.2e} (tol 1e-3)"
    );
    eprintln!("g1_kimi_k3_0_40b_full_dims: max_diff = {max_diff:.2e}");
}

#[test]
fn g1_latent_rmsnorm_changes_output() {
    // Verify that the q_a_layernorm + kv_a_layernorm are actually applied
    // (not no-ops). If we replace the norm weights with all-ones (pure RMS
    // normalization, no gamma scaling) vs random gammas, the outputs should
    // differ.
    let config = small_config();
    let weights_gamma = MlaWeights::random(&config, 42);
    // Replace norm weights with all-ones for the comparison baseline
    let mut weights_ones = weights_gamma.clone();
    weights_ones.q_a_norm_weight.fill(1.0);
    weights_ones.kv_a_norm_weight.fill(1.0);

    let h: Vec<f32> = (0..config.hidden_size)
        .map(|i| (i as f32).sin() * 0.3 + 0.5)
        .collect();

    let out_gamma = run_f32(&config, &weights_gamma, std::slice::from_ref(&h));
    let out_ones = run_f32(&config, &weights_ones, std::slice::from_ref(&h));

    // With random gammas ≠ 1.0, the outputs should differ.
    let any_diff = out_gamma
        .iter()
        .zip(out_ones.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(any_diff, "latent RMSNorm gammas had no effect on output");
}

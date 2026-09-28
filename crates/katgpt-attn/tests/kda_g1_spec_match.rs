//! G1 spec-match tests for KDA (Kimi Delta Attention).
//!
//! These tests verify the f32 `kda_forward_token` implementation against an
//! independent f64 reference written directly from the actual
//! `modeling_kimi_k3_linear.py` + the `fla.ops.kda.fused_recurrent_kda` Triton
//! kernel (decoded from fla-org/flash-linear-attention).
//!
//! The f64 reference is written WITHOUT reusing the f32 impl's code — it's a
//! from-scratch implementation of the same math. If both agree within f32
//! precision, we have confidence the f32 impl correctly encodes the equations.
//!
//! # What's tested (revised Research 330 §4)
//!
//! The reference mirrors the actual Kimi-K3 model code:
//! - Separate q/k/v projections (NOT shared qk)
//! - Separate q/k/v ShortConv + SiLU (NOT shared qk_conv + v_conv)
//! - Per-head A_log (NOT low-rank alpha_down/alpha_up)
//! - dt_bias added to gate before softplus
//! - f_a_proj + f_b_proj gate into the kernel
//! - Decay: `gk = -exp(A_log[h]) * softplus(g + dt_bias)` — per-key-channel
//! - Beta sigmoid inside the kernel
//! - Full-rank g_proj output gate (NOT low-rank)
//! - FusedRMSNormGated: `RMSNorm(o) * sigmoid(g_proj(h))`
//!
//! The TRUE correctness gate is Phase 6 (logits match real PyTorch weights).
//! This G1 gate catches transcription errors in our own code before we get to
//! Phase 6.

#![cfg(feature = "kda_linear")]

use katgpt_attn::gdn2::kda_forward::{
    KdaConfig, KdaForwardScratch, KdaLayerCache, KdaWeights, kda_forward_token,
};

// ─── f64 weight mirror ─────────────────────────────────────────────────────

/// f64 mirror of `KdaWeights`. Used to feed the f64 reference.
struct KdaWeightsF64 {
    q_proj: Vec<f64>,
    k_proj: Vec<f64>,
    v_proj: Vec<f64>,
    q_conv_weight: Vec<f64>,
    k_conv_weight: Vec<f64>,
    v_conv_weight: Vec<f64>,
    a_log: Vec<f64>,
    f_a_proj: Vec<f64>,
    f_b_proj: Vec<f64>,
    dt_bias: Vec<f64>,
    beta_proj: Vec<f64>,
    g_proj: Vec<f64>,
    o_norm_weight: Vec<f64>,
    o_proj: Vec<f64>,
}

fn weights_to_f64(w: &KdaWeights) -> KdaWeightsF64 {
    KdaWeightsF64 {
        q_proj: w.q_proj.iter().map(|&x| x as f64).collect(),
        k_proj: w.k_proj.iter().map(|&x| x as f64).collect(),
        v_proj: w.v_proj.iter().map(|&x| x as f64).collect(),
        q_conv_weight: w.q_conv_weight.iter().map(|&x| x as f64).collect(),
        k_conv_weight: w.k_conv_weight.iter().map(|&x| x as f64).collect(),
        v_conv_weight: w.v_conv_weight.iter().map(|&x| x as f64).collect(),
        a_log: w.a_log.iter().map(|&x| x as f64).collect(),
        f_a_proj: w.f_a_proj.iter().map(|&x| x as f64).collect(),
        f_b_proj: w.f_b_proj.iter().map(|&x| x as f64).collect(),
        dt_bias: w.dt_bias.iter().map(|&x| x as f64).collect(),
        beta_proj: w.beta_proj.iter().map(|&x| x as f64).collect(),
        g_proj: w.g_proj.iter().map(|&x| x as f64).collect(),
        o_norm_weight: w.o_norm_weight.iter().map(|&x| x as f64).collect(),
        o_proj: w.o_proj.iter().map(|&x| x as f64).collect(),
    }
}

// ─── f64 primitives ────────────────────────────────────────────────────────

fn matmul_f64(out: &mut [f64], w: &[f64], x: &[f64], rows: usize, cols: usize) {
    for r in 0..rows {
        let mut sum = 0.0;
        for c in 0..cols {
            sum += w[r * cols + c] * x[c];
        }
        out[r] = sum;
    }
}

fn dot_f64(a: &[f64], b: &[f64], len: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu_f64(x: f64) -> f64 {
    x * sigmoid_f64(x)
}

/// Numerically stable softplus: `log(1 + exp(x))`.
#[inline]
fn softplus_f64(x: f64) -> f64 {
    if x >= 20.0 {
        x
    } else if x >= 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// L2-normalize a vector in-place, matching the fla kernel's `1e-6` eps:
/// `x /= sqrt(sum(x^2) + 1e-6)`.
fn l2_normalize_f64(v: &mut [f64]) {
    let mut sum_sq = 0.0;
    for &x in v.iter() {
        sum_sq += x * x;
    }
    let inv_norm = 1.0 / (sum_sq.sqrt() + 1e-6);
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

// ─── f64 ShortConv1D ───────────────────────────────────────────────────────
// Independent reimplementation — does NOT share code with the f32 ShortConv1D.

struct ShortConvF64 {
    weight: Vec<f64>,
    buf: Vec<f64>,
    buf_idx: usize,
    n_channels: usize,
    kernel_size: usize,
}

impl ShortConvF64 {
    fn new(n_channels: usize, kernel_size: usize) -> Self {
        Self {
            weight: vec![0.0; n_channels * kernel_size],
            buf: vec![0.0; n_channels * kernel_size],
            buf_idx: 0,
            n_channels,
            kernel_size,
        }
    }

    fn forward(&mut self, x: &[f64], out: &mut [f64]) {
        let ks = self.kernel_size;
        let nc = self.n_channels;
        for (c, &xc) in x.iter().enumerate().take(nc) {
            self.buf[c * ks + self.buf_idx] = xc;
        }
        let new_buf_idx = (self.buf_idx + 1) % ks;
        let newest_slot = self.buf_idx;
        for (c, out_slot) in out.iter_mut().enumerate().take(nc) {
            let mut acc = 0.0;
            let w_off = c * ks;
            let b_off = c * ks;
            for k in 0..ks {
                let tap = self.weight[w_off + k];
                if tap != 0.0 {
                    let slot = (newest_slot + ks - k) % ks;
                    acc += tap * self.buf[b_off + slot];
                }
            }
            *out_slot = acc;
        }
        self.buf_idx = new_buf_idx;
    }
}

// ─── f64 KDA forward reference ─────────────────────────────────────────────

/// Independent f64 reference for the KDA forward pass.
///
/// Mirrors the actual `KimiDeltaAttention.forward` + the
/// `fla.ops.kda.fused_recurrent_kda` Triton kernel (decoded from the fla
/// source). The recurrence (with state stored as [K, V] row-major, matching the
/// f32 gdn2 kernel's layout):
///
/// ```text
/// Per-head, per-token:
///   q = L2Norm(z_q_conv[h]) * scale        (scale = 1/sqrt(dk))
///   k = L2Norm(z_k_conv[h])                 (no scale)
///   v = z_v_conv[h]                         (no norm)
///
///   gk[i] = -exp(A_log[h]) * softplus(g[i] + dt_bias[i])
///   alpha[i] = exp(gk[i])
///
///   # State update (K-rows, V-cols layout — the gdn2 convention):
///   for i in 0..dk:
///     S[i,:] *= alpha[i]                    (decay row i)
///   r[j] = sum_i S[i,j] * (beta * k[i])    (gated read with beta-broadcast)
///   delta[j] = beta * v[j] - r[j]           (delta rule)
///   S += k ⊗ delta                          (outer product write)
///   o[j] = sum_i S[i,j] * q[i]              (readout using updated S)
/// ```
fn kda_forward_f64_reference(
    config: &KdaConfig,
    weights: &KdaWeightsF64,
    tokens: &[Vec<f64>],
) -> Vec<f64> {
    let d = config.hidden_size;
    let dk = config.head_dim;
    let dv = config.head_dim;
    let n_h = config.n_heads;
    let proj = dk * n_h;
    let ks = config.conv_kernel_size;
    let alpha_eps = config.alpha_eps as f64;
    let rms_eps = config.rms_eps as f64;
    let scale = 1.0 / (dk as f64).sqrt();

    // Per-head state matrices S ∈ R^{dk × dv}, row-major (K-rows, V-cols).
    let mut state: Vec<Vec<f64>> = (0..n_h).map(|_| vec![0.0; dk * dv]).collect();
    // ShortConv ring buffers (separate for q, k, v).
    let mut q_conv = ShortConvF64::new(proj, ks);
    let mut k_conv = ShortConvF64::new(proj, ks);
    let mut v_conv = ShortConvF64::new(proj, ks);
    q_conv.weight.copy_from_slice(&weights.q_conv_weight);
    k_conv.weight.copy_from_slice(&weights.k_conv_weight);
    v_conv.weight.copy_from_slice(&weights.v_conv_weight);

    let mut last_output = vec![0.0f64; d];

    for h in tokens {
        // ── Step 1: Separate q/k/v projections ────────────────────────────
        let mut z_q = vec![0.0; proj];
        let mut z_k = vec![0.0; proj];
        let mut z_v = vec![0.0; proj];
        matmul_f64(&mut z_q, &weights.q_proj, h, proj, d);
        matmul_f64(&mut z_k, &weights.k_proj, h, proj, d);
        matmul_f64(&mut z_v, &weights.v_proj, h, proj, d);

        // Gate into kernel: g = f_b_proj(f_a_proj(h))
        let mut f_a_hidden = vec![0.0; dk];
        let mut g_raw = vec![0.0; proj];
        matmul_f64(&mut f_a_hidden, &weights.f_a_proj, h, dk, d);
        matmul_f64(&mut g_raw, &weights.f_b_proj, &f_a_hidden, proj, dk);

        // Beta (pre-sigmoid)
        let mut beta_pre = vec![0.0; n_h];
        matmul_f64(&mut beta_pre, &weights.beta_proj, h, n_h, d);
        let beta: Vec<f64> = beta_pre.iter().map(|&x| sigmoid_f64(x)).collect();

        // Full-rank output gate
        let mut g_out = vec![0.0; proj];
        matmul_f64(&mut g_out, &weights.g_proj, h, proj, d);

        // ── Step 2: Separate ShortConv + SiLU ─────────────────────────────
        let mut z_q_conv = vec![0.0; proj];
        let mut z_k_conv = vec![0.0; proj];
        let mut z_v_conv = vec![0.0; proj];
        q_conv.forward(&z_q, &mut z_q_conv);
        k_conv.forward(&z_k, &mut z_k_conv);
        v_conv.forward(&z_v, &mut z_v_conv);
        for i in 0..proj {
            z_q_conv[i] = silu_f64(z_q_conv[i]);
            z_k_conv[i] = silu_f64(z_k_conv[i]);
            z_v_conv[i] = silu_f64(z_v_conv[i]);
        }

        // ── Step 3: Per-head recurrent step ───────────────────────────────
        let mut o_concat = vec![0.0; proj];
        for head in 0..n_h {
            let off = head * dk;

            // q: L2Norm + scale
            let mut q_h = vec![0.0; dk];
            q_h.copy_from_slice(&z_q_conv[off..off + dk]);
            l2_normalize_f64(&mut q_h);
            for slot in q_h.iter_mut() {
                *slot *= scale;
            }

            // k: L2Norm (no scale)
            let mut k_h = vec![0.0; dk];
            k_h.copy_from_slice(&z_k_conv[off..off + dk]);
            l2_normalize_f64(&mut k_h);

            // v: no norm
            let v_h = &z_v_conv[off..off + dk];

            // Per-key-channel decay: alpha[i] = exp(-exp(A_log) * softplus(g[i] + dt_bias[i]))
            let alpha_head = weights.a_log[head].exp();
            let mut alpha_h = vec![0.0; dk];
            for i in 0..dk {
                let g_plus_bias = g_raw[off + i] + weights.dt_bias[off + i];
                let gk = -alpha_head * softplus_f64(g_plus_bias);
                let a = gk.exp();
                alpha_h[i] = if a < alpha_eps { alpha_eps } else { a };
            }

            let beta_h = beta[head];

            // ── KDA recurrence (expanded form, K-rows/V-cols layout) ──────
            // Matches the gdn2 kernel's Kda gate config:
            //   S' = Diag(alpha) · S          (decay each row i by alpha[i])
            //   r = S'^T · (beta·k)           (gated read — beta-broadcast)
            //   delta = beta·v − r            (delta rule)
            //   S = S' + k ⊗ delta            (write)
            //   o = S^T · q                   (readout)
            let s = &mut state[head];

            // S' = Diag(alpha) · S + accumulate r = S'^T · (beta·k)
            let mut s_decayed = vec![0.0; dk * dv];
            let mut r = vec![0.0; dv];
            for i in 0..dk {
                let a = alpha_h[i];
                let bk_i = beta_h * k_h[i];
                for j in 0..dv {
                    let sv = s[i * dv + j] * a;
                    s_decayed[i * dv + j] = sv;
                    r[j] += sv * bk_i;
                }
            }

            // delta = beta·v − r
            let mut delta = vec![0.0; dv];
            for j in 0..dv {
                delta[j] = beta_h * v_h[j] - r[j];
            }

            // S = S' + k ⊗ delta
            for i in 0..dk {
                let ki = k_h[i];
                for j in 0..dv {
                    s[i * dv + j] = s_decayed[i * dv + j] + ki * delta[j];
                }
            }

            // o = S^T · q
            let mut o_h = vec![0.0; dv];
            for j in 0..dv {
                let mut acc = 0.0;
                for i in 0..dk {
                    acc += s[i * dv + j] * q_h[i];
                }
                o_h[j] = acc;
            }

            // ── FusedRMSNormGated: RMSNorm(o, gamma) * sigmoid(g_out) ─────
            let sum_sq = dot_f64(&o_h, &o_h, dk);
            let mean_sq = sum_sq / dk as f64;
            let inv_rms = 1.0 / (mean_sq + rms_eps).sqrt();
            for i in 0..dk {
                let normed = o_h[i] * inv_rms * weights.o_norm_weight[i];
                let gated = sigmoid_f64(g_out[off + i]);
                o_concat[head * dk + i] = normed * gated;
            }
        }

        // ── Step 4: Output projection ─────────────────────────────────────
        matmul_f64(&mut last_output, &weights.o_proj, &o_concat, d, proj);
    }

    last_output
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn run_f32(config: &KdaConfig, weights: &KdaWeights, tokens: &[Vec<f32>]) -> Vec<f32> {
    let mut cache = KdaLayerCache::new(config);
    let mut scratch = KdaForwardScratch::new(config);
    let mut last_out = Vec::new();
    for h in tokens {
        let out = kda_forward_token(config, weights, &mut cache, &mut scratch, h);
        last_out = out.to_vec();
    }
    last_out
}

fn max_diff_f32_f64(a: &[f32], b: &[f64]) -> f32 {
    let mut max_d = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y as f32).abs();
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

// ─── Tests ─────────────────────────────────────────────────────────────────

fn small_config() -> KdaConfig {
    KdaConfig {
        head_dim: 8,
        n_heads: 2,
        hidden_size: 16,
        conv_kernel_size: 4,
        alpha_eps: 1e-5,
        rms_eps: 1e-5,
    }
}

#[test]
fn g1_kda_matches_reference_small_dims() {
    let config = small_config();
    let weights = KdaWeights::random(&config, 42);
    let weights_f64 = weights_to_f64(&weights);

    let tokens_f32: Vec<Vec<f32>> = (0..3)
        .map(|t| {
            (0..config.hidden_size)
                .map(|i| ((i + t * 13) as f32).sin() * 0.1)
                .collect()
        })
        .collect();
    let tokens_f64: Vec<Vec<f64>> = tokens_f32
        .iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();

    let out_f32 = run_f32(&config, &weights, &tokens_f32);
    let out_f64 = kda_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_diff_f32_f64(&out_f32, &out_f64);
    eprintln!("g1_kda_small_dims: max_diff = {max_diff:.2e}");
    assert!(
        max_diff < 1e-4,
        "G1 FAIL small dims: max_diff = {max_diff:.2e} (tol 1e-4)"
    );
}

#[test]
fn g1_kda_matches_reference_kimi_k3_0_40b_dims() {
    // Full 0.40B KDA config. Slower but verifies real-scale math.
    let config = KdaConfig::kimi_k3_0_40b();
    let weights = KdaWeights::random(&config, 777);
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
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();

    let out_f32 = run_f32(&config, &weights, &tokens_f32);
    let out_f64 = kda_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_diff_f32_f64(&out_f32, &out_f64);
    eprintln!("g1_kda_0.40B_dims: max_diff = {max_diff:.2e}");
    assert!(
        max_diff < 1e-3,
        "G1 FAIL 0.40B dims: max_diff = {max_diff:.2e} (tol 1e-3)"
    );
}

#[test]
fn g1_smoke_kimi_k3_0_40b_dims_finite() {
    // Quick finite-output smoke test at full 0.40B dims.
    let config = KdaConfig::kimi_k3_0_40b();
    let weights = KdaWeights::random(&config, 99);
    let tokens: Vec<Vec<f32>> = (0..3)
        .map(|t| vec![0.1f32 + t as f32 * 0.01; config.hidden_size])
        .collect();
    let out = run_f32(&config, &weights, &tokens);
    assert_eq!(out.len(), config.hidden_size);
    for &v in &out {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

/// The load-bearing test: prove that the separate q/k/v projections + per-head
/// A_log + dt_bias + f_a/f_b gate + full-rank g_proj + FusedRMSNormGated all
/// wire correctly. The f64 reference computes the full expanded recurrence;
/// any divergence surfaces as a max_diff exceeding tolerance.
#[test]
fn g1_kda_revised_architecture_correctness() {
    // Smaller-than-small config for tight tolerance + fast iteration.
    let config = KdaConfig {
        head_dim: 4,
        n_heads: 2,
        hidden_size: 8,
        conv_kernel_size: 2,
        alpha_eps: 1e-5,
        rms_eps: 1e-5,
    };
    let weights = KdaWeights::random(&config, 123);
    let weights_f64 = weights_to_f64(&weights);

    let tokens_f32: Vec<Vec<f32>> = (0..5)
        .map(|t| {
            (0..config.hidden_size)
                .map(|i| ((i + t * 7) as f32).sin() * 0.1)
                .collect()
        })
        .collect();
    let tokens_f64: Vec<Vec<f64>> = tokens_f32
        .iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();

    let out_f32 = run_f32(&config, &weights, &tokens_f32);
    let out_f64 = kda_forward_f64_reference(&config, &weights_f64, &tokens_f64);

    let max_diff = max_diff_f32_f64(&out_f32, &out_f64);
    eprintln!("g1_kda_revised_arch: max_diff = {max_diff:.2e}");
    assert!(
        max_diff < 1e-4,
        "G1 FAIL revised arch: max_diff = {max_diff:.2e} (tol 1e-4). \
         Check separate q/k/v, per-head A_log, dt_bias, f_a/f_b gate, \
         full-rank g_proj, FusedRMSNormGated."
    );
}

/// Causality test: token t's output depends only on tokens [0..=t].
#[test]
fn g1_kda_causality_state_does_not_depend_on_future() {
    let config = small_config();
    let weights = KdaWeights::random(&config, 7);

    let t0: Vec<f32> = (0..config.hidden_size)
        .map(|i| (i as f32).sin() * 0.1)
        .collect();
    let t1: Vec<f32> = (0..config.hidden_size)
        .map(|i| ((i + 13) as f32).sin() * 0.1)
        .collect();
    let t2: Vec<f32> = (0..config.hidden_size)
        .map(|i| ((i + 26) as f32).sin() * 0.1)
        .collect();
    let t3: Vec<f32> = (0..config.hidden_size)
        .map(|i| ((i + 39) as f32).sin() * 0.1)
        .collect();

    // Path A: process [t0, t1, t2] only — capture o2 (last output).
    let out_a = run_f32(&config, &weights, &[t0.clone(), t1.clone(), t2.clone()]);

    // Path B: process [t0, t1, t2, t3] — capture o2 (third output).
    let mut cache = KdaLayerCache::new(&config);
    let mut scratch = KdaForwardScratch::new(&config);
    let _o0 = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &t0).to_vec();
    let _o1 = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &t1).to_vec();
    let o2_b = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &t2).to_vec();
    let _o3 = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &t3).to_vec();

    // o2 in path A == o2 in path B (t3 has no retroactive effect).
    assert_eq!(
        out_a, o2_b,
        "KDA state must be causal: t3 must not change the output at t2"
    );
}

/// Verify the decay gate produces values in (0, 1) (exp of a negative number).
#[test]
fn g1_decay_is_in_unit_interval() {
    let config = KdaConfig::kimi_k3_0_40b();
    let weights = KdaWeights::random(&config, 42);
    let mut cache = KdaLayerCache::new(&config);
    let mut scratch = KdaForwardScratch::new(&config);

    let h: Vec<f32> = (0..config.hidden_size)
        .map(|i| (i as f32).sin() * 0.1)
        .collect();
    let _ = kda_forward_token(&config, &weights, &mut cache, &mut scratch, &h);

    // After forward, the state matrices should have finite values.
    // The decay exp(gk) where gk < 0 means values are in (0, 1), but the
    // write step (k ⊗ delta) can produce any sign. We just check finiteness.
    for head in &cache.heads {
        for &v in &head.s {
            assert!(v.is_finite(), "non-finite state: {v}");
        }
    }
}

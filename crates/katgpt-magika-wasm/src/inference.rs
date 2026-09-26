//! SIMD128-accelerated forward pass for the Magika model (F32, no quantization).

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;

use crate::model::{
    get_weights, CONV_OUT, EMB_DIM, HIDDEN, KERNEL, NUM_LABELS, TIME,
};

/// Full forward pass. `features` is a 2048-element i32 slice.
/// Returns 214 softmax probabilities.
pub fn forward(features: &[i32]) -> Vec<f32> {
    debug_assert_eq!(features.len(), crate::model::FEATURES_SIZE);
    let w = get_weights();

    // 1. Embedding lookup + bias + GELU, reshaped to (512, 256)
    let mut x0 = vec![0f32; HIDDEN * TIME];
    for t in 0..2048 {
        let tok = features[t] as usize;
        let src = &w.emb_w[tok * EMB_DIM..][..EMB_DIM];
        let dst = &mut x0[t * EMB_DIM..][..EMB_DIM];
        for c in 0..EMB_DIM {
            let xi = src[c] + w.emb_b[c];
            let inner = 0.797_884_6f32 * (xi + 0.044_715f32 * xi * xi * xi);
            dst[c] = 0.5 * xi * (1.0 + inner.tanh());
        }
    }

    // 2. LayerNorm 0 across the 512 channels for each of 256 columns
    let mut mean = [0f32; TIME];
    let mut second = [0f32; TIME];
    for t in 0..TIME {
        let mut s = 0f32;
        let mut ss = 0f32;
        for h in 0..HIDDEN {
            let v = x0[h * TIME + t];
            s += v;
            ss += v * v;
        }
        mean[t] = s / HIDDEN as f32;
        second[t] = ss / HIDDEN as f32;
    }
    let mut x1 = vec![0f32; HIDDEN * TIME];
    for h in 0..HIDDEN {
        let g = w.ln0_gamma[h];
        let b = w.ln0_beta[h];
        for t in 0..TIME {
            let v = x0[h * TIME + t];
            let var = (second[t] - mean[t] * mean[t]).max(0.0);
            x1[h * TIME + t] = (v - mean[t]) * (1.0 / (var + 1e-6).sqrt()) * g + b;
        }
    }

    // 3. Conv1d: kernel 5, in 256, out 512 → (512, 508) + GELU
    let mut conv_gelu = vec![0f32; HIDDEN * CONV_OUT];
    for oc in 0..HIDDEN {
        let w_oc = &w.conv_w[oc * KERNEL * TIME..][..KERNEL * TIME];
        for t in 0..CONV_OUT {
            let mut acc = w.conv_b[oc];
            for k in 0..KERNEL {
                let x = &x1[(t + k) * TIME..][..TIME];
                let w_k = &w_oc[k * TIME..][..TIME];
                acc += dot_f32(x, w_k);
            }
            let inner = 0.797_884_6f32 * (acc + 0.044_715f32 * acc * acc * acc);
            conv_gelu[oc * CONV_OUT + t] = 0.5 * acc * (1.0 + inner.tanh());
        }
    }

    // 4. GlobalMaxPool over 508
    let mut pooled = vec![0f32; HIDDEN];
    for c in 0..HIDDEN {
        let row = &conv_gelu[c * CONV_OUT..][..CONV_OUT];
        pooled[c] = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }

    // 5. LayerNorm 1 over 512 channels
    let mut p_sum = 0f32;
    let mut p_ss = 0f32;
    for v in pooled.iter() {
        p_sum += v;
        p_ss += v * v;
    }
    let m1 = p_sum / HIDDEN as f32;
    let var1 = (p_ss / HIDDEN as f32 - m1 * m1).max(0.0);
    let inv1 = 1.0 / (var1 + 1e-6).sqrt();
    let mut x3 = vec![0f32; HIDDEN];
    for i in 0..HIDDEN {
        x3[i] = (pooled[i] - m1) * inv1 * w.ln1_gamma[i] + w.ln1_beta[i];
    }

    // 6. Dense 1: (214, 512) x (512,) + bias → (214,)
    let mut logits = vec![0f32; NUM_LABELS];
    for (l, logit) in logits.iter_mut().enumerate() {
        let row = &w.dense1_w[l * HIDDEN..][..HIDDEN];
        *logit = dot_f32(row, &x3) + w.dense1_b[l];
    }

    // 7. Softmax
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_s = 0f32;
    let mut probs = vec![0f32; NUM_LABELS];
    for (prob, &logit) in probs.iter_mut().zip(logits.iter()) {
        *prob = (logit - max_l).exp();
        exp_s += *prob;
    }
    for prob in probs.iter_mut() {
        *prob /= exp_s;
    }
    probs
}

/// Dot product of two f32 slices. SIMD128-accelerated on wasm32,
/// 4-lane unrolled elsewhere.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { dot_f32_simd128(a, b) }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        dot_f32_scalar(a, b)
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_f32_simd128(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = f32x4_splat(0.0);
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let idx = i * 4;
        unsafe {
            let va = v128_load(a[idx..].as_ptr() as *const v128);
            let vb = v128_load(b[idx..].as_ptr() as *const v128);
            acc = f32x4_add(acc, f32x4_mul(va, vb));
        }
    }
    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    for i in chunks * 4..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
#[allow(dead_code)]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let idx = i * 4;
        acc0 += a[idx] * b[idx];
        acc1 += a[idx + 1] * b[idx + 1];
        acc2 += a[idx + 2] * b[idx + 2];
        acc3 += a[idx + 3] * b[idx + 3];
    }
    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    for i in chunks * 4..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

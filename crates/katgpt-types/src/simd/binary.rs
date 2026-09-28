//! Binary bit-plane matvec — single sign bit-plane, group-wise FP16 scale
//! (`binary_plasma` feature, Issue 145). Scalar / NEON / AVX2 paths.
//!
//! Simpler than ternary: one bit-plane instead of two, no zero-skip branch.
//! Each weight is `{-1, +1}` — bit set → +1, bit clear → -1. Group-wise
//! FP16 scale (per 128 weights) replaces ternary's single row_scale.
//!
//! # Kernel design (Gate A fix)
//!
//! The group scale is folded INTO the sign vector (`±scale` instead of `±1`),
//! so the 4 accumulators span the ENTIRE row (like ternary) — no per-group
//! resets, no per-group horizontal sums. This keeps the OoO pipeline fed
//! across all groups in a row.
//!
//! `scaled_sign = fma(neg_2scale, bit_set_f, neg_scale)`:
//! - bit set (bit_set_f = -1.0): 2*scale - scale = +scale
//! - bit clear (bit_set_f = 0.0): -scale

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

#[cfg(feature = "binary_plasma")]
use super::simd_level;
// `SimdLevel` is only referenced inside target_arch-gated match arms (NEON/
// AVX2). On targets where none of those arms compile (e.g. wasm32, or scalar
// fallbacks), the bare import would trigger an unused-import warning under
// `-D warnings`. Gate it to match the arms.
#[cfg(all(
    feature = "binary_plasma",
    any(target_arch = "aarch64", target_arch = "x86_64"),
))]
use super::SimdLevel;

#[cfg(all(feature = "binary_plasma", target_arch = "x86_64"))]
use super::horizontal::horizontal_sum_256;

#[cfg(feature = "binary_plasma")]
use crate::{BinaryWeights, GROUP_SIZE};

/// Scalar reference binary matvec:
/// `y[r] = Σ_g group_scale[r,g] * Σ_{col in g} sign(bit[r,col]) * x[col]`
///
/// `sign = +1` if bit set, `-1` if clear. No zero state.
#[cfg(feature = "binary_plasma")]
pub fn binary_matvec_scalar(w: &BinaryWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    for r in 0..w.rows {
        let mut row_sum = 0.0f32;
        let sign_base = r * w.blocks64;
        for g in 0..w.groups_per_row {
            let g_start = g * GROUP_SIZE;
            let g_end = (g_start + GROUP_SIZE).min(w.cols);
            let scale = w.group_scale[r * w.groups_per_row + g].to_f32();
            let mut group_acc = 0.0f32;
            for col in g_start..g_end {
                let block = col >> 6;
                let bit = col & 63;
                let mask = 1u64 << bit;
                let sign = if (w.sign_bits[sign_base + block] & mask) != 0 {
                    1.0f32
                } else {
                    -1.0f32
                };
                group_acc += sign * unsafe { *x.get_unchecked(col) };
            }
            row_sum += scale * group_acc;
        }
        y[r] = row_sum;
    }
}

#[cfg(all(feature = "binary_plasma", target_arch = "aarch64"))]
unsafe fn neon_binary_matvec(w: &BinaryWeights, x: &[f32], y: &mut [f32]) {
    // Safety: caller guarantees x.len()==w.cols and y.len()==w.rows
    unsafe {
        use core::arch::aarch64::{float32x4_t, uint32x4_t, *};
        assert_eq!(x.len(), w.cols);
        assert_eq!(y.len(), w.rows);

        // SWAR bit-position masks for low/high nibble.
        let mask_lo_arr: [u32; 4] = [1, 2, 4, 8];
        let mask_hi_arr: [u32; 4] = [16, 32, 64, 128];
        let mask_lo: uint32x4_t = vld1q_u32(mask_lo_arr.as_ptr());
        let mask_hi: uint32x4_t = vld1q_u32(mask_hi_arr.as_ptr());
        let one_u: uint32x4_t = vdupq_n_u32(1);

        for r in 0..w.rows {
            let sign_base = r * w.blocks64;
            let group_base = r * w.groups_per_row;

            // 4 accumulators span the ENTIRE row (no per-group reset).
            let mut acc0: float32x4_t = vdupq_n_f32(0.0);
            let mut acc1: float32x4_t = vdupq_n_f32(0.0);
            let mut acc2: float32x4_t = vdupq_n_f32(0.0);
            let mut acc3: float32x4_t = vdupq_n_f32(0.0);

            for g in 0..w.groups_per_row {
                let g_start = g * GROUP_SIZE;
                let g_end = (g_start + GROUP_SIZE).min(w.cols);
                let remaining = g_end - g_start;

                // Group scale folded into sign vector: ±scale instead of ±1.
                let scale = w.group_scale[group_base + g].to_f32();
                let neg_2scale: float32x4_t = vdupq_n_f32(-2.0 * scale);
                let neg_scale: float32x4_t = vdupq_n_f32(-scale);

                let chunks32 = remaining / 32 * 32;
                let mut col = 0usize;
                while col < chunks32 {
                    fma_scaled_nibble8(
                        &mut acc0,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_lo,
                        mask_hi,
                        one_u,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                    fma_scaled_nibble8(
                        &mut acc1,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_lo,
                        mask_hi,
                        one_u,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                    fma_scaled_nibble8(
                        &mut acc2,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_lo,
                        mask_hi,
                        one_u,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                    fma_scaled_nibble8(
                        &mut acc3,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_lo,
                        mask_hi,
                        one_u,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                }
                while col + 8 <= remaining {
                    fma_scaled_nibble8(
                        &mut acc0,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_lo,
                        mask_hi,
                        one_u,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                }
                if col + 4 <= remaining {
                    fma_scaled_nibble4(
                        &mut acc0,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_lo,
                        mask_hi,
                        one_u,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 4;
                }

                // Scalar tail
                let mut scalar_acc = 0.0f32;
                while col < remaining {
                    let abs_col = g_start + col;
                    let block = abs_col >> 6;
                    let bit = abs_col & 63;
                    let mask = 1u64 << bit;
                    let sign = if (w.sign_bits[sign_base + block] & mask) != 0 {
                        scale
                    } else {
                        -scale
                    };
                    scalar_acc += sign * *x.get_unchecked(abs_col);
                    col += 1;
                }
                if scalar_acc != 0.0 {
                    acc0 = vaddq_f32(acc0, vsetq_lane_f32(scalar_acc, vdupq_n_f32(0.0), 0));
                }
            }

            // Single horizontal sum at the end (like ternary).
            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            y[r] = vaddvq_f32(acc0);
        }
    }
}

/// NEON inner-loop: process 8 elements with group-scaled sign.
/// `scaled_sign = fma(neg_2scale, bit_set_f, neg_scale)` → ±scale.
#[cfg(all(feature = "binary_plasma", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn fma_scaled_nibble8(
    acc: &mut core::arch::aarch64::float32x4_t,
    sign_bits: &[u64],
    sign_base: usize,
    col: usize,
    g_start: usize,
    x: &[f32],
    mask_lo: core::arch::aarch64::uint32x4_t,
    mask_hi: core::arch::aarch64::uint32x4_t,
    one_u: core::arch::aarch64::uint32x4_t,
    neg_2scale: core::arch::aarch64::float32x4_t,
    neg_scale: core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let abs_col = g_start + col;
        let block = abs_col >> 6;
        let byte_off = (abs_col & 63) / 8;
        let sign_byte = ((sign_bits[sign_base + block] >> (byte_off * 8)) & 0xFF) as u32;
        let sign_splat = vdupq_n_u32(sign_byte);

        // vcgeq_u32(and, 1) → all-ones (=-1 as i32) where bit set, 0 where clear.
        let bs_lo = vcvtq_f32_s32(vreinterpretq_s32_u32(vcgeq_u32(
            vandq_u32(sign_splat, mask_lo),
            one_u,
        )));
        let bs_hi = vcvtq_f32_s32(vreinterpretq_s32_u32(vcgeq_u32(
            vandq_u32(sign_splat, mask_hi),
            one_u,
        )));

        // scaled_sign = neg_2scale * bit_set_f + neg_scale → +scale where set, -scale where clear.
        let sign_lo = vfmaq_f32(neg_scale, neg_2scale, bs_lo);
        let sign_hi = vfmaq_f32(neg_scale, neg_2scale, bs_hi);

        let x_lo = vld1q_f32(x.as_ptr().add(abs_col));
        let x_hi = vld1q_f32(x.as_ptr().add(abs_col + 4));

        *acc = vfmaq_f32(*acc, sign_lo, x_lo);
        *acc = vfmaq_f32(*acc, sign_hi, x_hi);
    }
}

/// NEON 4-element variant (half-nibble tail).
#[cfg(all(feature = "binary_plasma", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn fma_scaled_nibble4(
    acc: &mut core::arch::aarch64::float32x4_t,
    sign_bits: &[u64],
    sign_base: usize,
    col: usize,
    g_start: usize,
    x: &[f32],
    mask_lo: core::arch::aarch64::uint32x4_t,
    _mask_hi: core::arch::aarch64::uint32x4_t,
    one_u: core::arch::aarch64::uint32x4_t,
    neg_2scale: core::arch::aarch64::float32x4_t,
    neg_scale: core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let abs_col = g_start + col;
        let block = abs_col >> 6;
        let byte_off = (abs_col & 63) / 8;
        let sign_byte = ((sign_bits[sign_base + block] >> (byte_off * 8)) & 0xFF) as u32;
        let sign_splat = vdupq_n_u32(sign_byte);

        let bs_lo = vcvtq_f32_s32(vreinterpretq_s32_u32(vcgeq_u32(
            vandq_u32(sign_splat, mask_lo),
            one_u,
        )));
        let sign_lo = vfmaq_f32(neg_scale, neg_2scale, bs_lo);
        let x_lo = vld1q_f32(x.as_ptr().add(abs_col));
        *acc = vfmaq_f32(*acc, sign_lo, x_lo);
    }
}

#[cfg(all(feature = "binary_plasma", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_binary_matvec(w: &BinaryWeights, x: &[f32], y: &mut [f32]) {
    // Safety: caller guarantees x.len()==w.cols and y.len()==w.rows
    unsafe {
        use core::arch::x86_64::*;
        assert_eq!(x.len(), w.cols);
        assert_eq!(y.len(), w.rows);

        let mask_byte: __m256i = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let zero_i: __m256i = _mm256_setzero_si256();

        for r in 0..w.rows {
            let sign_base = r * w.blocks64;
            let group_base = r * w.groups_per_row;

            let mut acc0: __m256 = _mm256_setzero_ps();
            let mut acc1: __m256 = _mm256_setzero_ps();
            let mut acc2: __m256 = _mm256_setzero_ps();
            let mut acc3: __m256 = _mm256_setzero_ps();

            for g in 0..w.groups_per_row {
                let g_start = g * GROUP_SIZE;
                let g_end = (g_start + GROUP_SIZE).min(w.cols);
                let remaining = g_end - g_start;

                let scale = w.group_scale[group_base + g].to_f32();
                let neg_2scale = _mm256_set1_ps(-2.0 * scale);
                let neg_scale = _mm256_set1_ps(-scale);

                let chunks32 = remaining / 32 * 32;
                let mut col = 0usize;
                while col < chunks32 {
                    fma_scaled_byte8_avx2(
                        &mut acc0,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_byte,
                        zero_i,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                    fma_scaled_byte8_avx2(
                        &mut acc1,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_byte,
                        zero_i,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                    fma_scaled_byte8_avx2(
                        &mut acc2,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_byte,
                        zero_i,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                    fma_scaled_byte8_avx2(
                        &mut acc3,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_byte,
                        zero_i,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                }
                while col + 8 <= remaining {
                    fma_scaled_byte8_avx2(
                        &mut acc0,
                        &w.sign_bits,
                        sign_base,
                        col,
                        g_start,
                        x,
                        mask_byte,
                        zero_i,
                        neg_2scale,
                        neg_scale,
                    );
                    col += 8;
                }

                // Scalar tail
                let mut scalar_acc = 0.0f32;
                while col < remaining {
                    let abs_col = g_start + col;
                    let block = abs_col >> 6;
                    let bit = abs_col & 63;
                    let mask = 1u64 << bit;
                    let sign = if (w.sign_bits[sign_base + block] & mask) != 0 {
                        scale
                    } else {
                        -scale
                    };
                    scalar_acc += sign * *x.get_unchecked(abs_col);
                    col += 1;
                }
                if scalar_acc != 0.0 {
                    acc0 = _mm256_add_ps(
                        acc0,
                        _mm256_setr_ps(scalar_acc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    );
                }
            }

            acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            y[r] = horizontal_sum_256(acc0);
        }
    }
}

/// AVX2 inner-loop: 8 elements with group-scaled sign.
/// `scaled_sign = fma(neg_2scale, bit_set_f, neg_scale)` → ±scale.
#[cfg(all(feature = "binary_plasma", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fma_scaled_byte8_avx2(
    acc: &mut core::arch::x86_64::__m256,
    sign_bits: &[u64],
    sign_base: usize,
    col: usize,
    g_start: usize,
    x: &[f32],
    mask_byte: core::arch::x86_64::__m256i,
    zero_i: core::arch::x86_64::__m256i,
    neg_2scale: core::arch::x86_64::__m256,
    neg_scale: core::arch::x86_64::__m256,
) {
    use core::arch::x86_64::*;
    unsafe {
        let abs_col = g_start + col;
        let block = abs_col >> 6;
        let byte_off = (abs_col & 63) / 8;
        let sign_byte = ((sign_bits[sign_base + block] >> (byte_off * 8)) & 0xFF) as i32;
        let sign_splat = _mm256_set1_epi32(sign_byte);

        // cmpgt(and, 0) → all-ones (-1) where set, 0 where clear.
        let bs_f = _mm256_cvtepi32_ps(_mm256_cmpgt_epi32(
            _mm256_and_si256(sign_splat, mask_byte),
            zero_i,
        ));

        // scaled_sign = neg_2scale * bs_f + neg_scale → +scale where set, -scale where clear.
        let scaled_sign = _mm256_fmadd_ps(neg_2scale, bs_f, neg_scale);

        let x_v = _mm256_loadu_ps(x.as_ptr().add(abs_col));
        *acc = _mm256_fmadd_ps(scaled_sign, x_v, *acc);
    }
}

/// Dispatch binary matvec to the best available SIMD backend.
#[cfg(feature = "binary_plasma")]
pub fn simd_binary_matvec(w: &BinaryWeights, x: &[f32], y: &mut [f32]) {
    match simd_level() {
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe { neon_binary_matvec(w, x, y) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe { avx2_binary_matvec(w, x, y) },
        _ => binary_matvec_scalar(w, x, y),
    }
}

/// Batch binary matvec (sequential for small batches, rayon for large).
/// Mirrors `simd_ternary_matmul_batch`.
#[cfg(feature = "binary_plasma")]
pub fn simd_binary_matmul_batch(w: &BinaryWeights, x: &[f32], batch: usize, y: &mut [f32]) {
    const PARALLEL_BATCH_MIN: usize = 4;

    use rayon::prelude::*;

    if batch < PARALLEL_BATCH_MIN {
        for b in 0..batch {
            let x_off = b * w.cols;
            let y_off = b * w.rows;
            // Slice EXACTLY — see the note in simd_ternary_matmul_batch. The
            // open-ended form panics the length assert for 2 <= batch < 4.
            simd_binary_matvec(w, &x[x_off..x_off + w.cols], &mut y[y_off..y_off + w.rows]);
        }
        return;
    }
    y.par_chunks_mut(w.rows)
        .zip(x.par_chunks(w.cols))
        .enumerate()
        .for_each(|(b, (y_chunk, x_chunk))| {
            if b < batch {
                simd_binary_matvec(w, x_chunk, y_chunk);
            }
        });
}

#[cfg(test)]
#[cfg(all(feature = "binary_plasma", feature = "plasma_path"))]
mod tests {
    use super::*;
    use crate::TernaryWeights;

    fn make_random_binary_weights(rows: usize, cols: usize, seed: u64) -> BinaryWeights {
        let mut bw = BinaryWeights::new(rows, cols);
        let mut rng = seed;
        for r in 0..rows {
            let sign_base = r * bw.blocks64;
            for b in 0..bw.blocks64 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                bw.sign_bits[sign_base + b] = rng;
            }
        }
        bw
    }

    fn make_random_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        (0..len)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng as f32) / (u64::MAX as f32) - 0.5) * 2.0
            })
            .collect()
    }

    #[test]
    fn test_scalar_vs_simd_parity_64() {
        let bw = make_random_binary_weights(4, 64, 42);
        let x = make_random_vec(64, 99);
        let mut y_scalar = vec![0.0f32; 4];
        let mut y_simd = vec![0.0f32; 4];
        binary_matvec_scalar(&bw, &x, &mut y_scalar);
        simd_binary_matvec(&bw, &x, &mut y_simd);
        let max_diff = y_scalar
            .iter()
            .zip(y_simd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "scalar vs simd max_diff = {max_diff}");
    }

    #[test]
    fn test_scalar_vs_simd_parity_1024() {
        let bw = make_random_binary_weights(8, 1024, 42);
        let x = make_random_vec(1024, 99);
        let mut y_scalar = vec![0.0f32; 8];
        let mut y_simd = vec![0.0f32; 8];
        binary_matvec_scalar(&bw, &x, &mut y_scalar);
        simd_binary_matvec(&bw, &x, &mut y_simd);
        let max_diff = y_scalar
            .iter()
            .zip(y_simd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "scalar vs simd max_diff = {max_diff}");
    }

    /// The core T0.1 property test: binary matvec matches ternary matvec
    /// bit-identically when the ternary weights have no zeros (binary subset).
    #[test]
    fn test_binary_subset_of_ternary() {
        let rows = 4;
        let cols = 256;

        // Construct ternary weights with no zeros (binary subset):
        // pos_bits = random, neg_bits = !pos_bits.
        let mut tw = TernaryWeights::new(rows, cols);
        let mut rng = 12345u64;
        for r in 0..rows {
            let base = r * tw.blocks64;
            for b in 0..tw.blocks64 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                tw.pos_bits[base + b] = rng;
                tw.neg_bits[base + b] = !rng; // no zeros: XOR = all-ones
            }
            tw.row_scale[r] = 1.5; // arbitrary non-trivial scale
        }

        // Build equivalent binary weights
        let bw = BinaryWeights::from_ternary_no_zeros(&tw).expect("no zeros in ternary");

        let x = make_random_vec(cols, 99);

        // Ternary matvec
        let mut y_ternary = vec![0.0f32; rows];
        crate::simd::simd_ternary_matvec(&tw, &x, &mut y_ternary);

        // Binary matvec (SIMD)
        let mut y_binary = vec![0.0f32; rows];
        simd_binary_matvec(&bw, &x, &mut y_binary);

        // They must match (both compute row_scale * Σ sign * x)
        let max_diff = y_ternary
            .iter()
            .zip(y_binary.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "binary vs ternary matvec max_diff = {max_diff}\n  ternary: {y_ternary:?}\n  binary:  {y_binary:?}"
        );
    }

    #[test]
    fn test_binary_subset_rejects_zeros() {
        // Construct ternary weights with a zero: both pos and neg clear for some bits.
        // pos = 0xAAAAAAAA, neg = 0x0 → XOR = 0xAAAAAAAA which has zero bits.
        let mut tw = TernaryWeights::new(1, 64);
        tw.pos_bits[0] = 0xAAAAAAAAAAAAAAAA;
        tw.neg_bits[0] = 0x0; // XOR != all-ones → has zeros
        assert!(BinaryWeights::from_ternary_no_zeros(&tw).is_none());
    }

    #[test]
    fn test_non_128_aligned_cols() {
        // cols = 100 (not a multiple of 64 or 128)
        let bw = make_random_binary_weights(2, 100, 7);
        let x = make_random_vec(100, 3);
        let mut y_scalar = vec![0.0f32; 2];
        let mut y_simd = vec![0.0f32; 2];
        binary_matvec_scalar(&bw, &x, &mut y_scalar);
        simd_binary_matvec(&bw, &x, &mut y_simd);
        let max_diff = y_scalar
            .iter()
            .zip(y_simd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "non-128-aligned max_diff = {max_diff}");
    }

    #[test]
    fn test_batch_parity() {
        let bw = make_random_binary_weights(4, 64, 42);
        let batch = 8;
        let x = make_random_vec(64 * batch, 99);
        let mut y_seq = vec![0.0f32; 4 * batch];
        let mut y_par = vec![0.0f32; 4 * batch];

        // Sequential
        for b in 0..batch {
            let x_off = b * bw.cols;
            let y_off = b * bw.rows;
            simd_binary_matvec(
                &bw,
                &x[x_off..x_off + bw.cols],
                &mut y_seq[y_off..y_off + bw.rows],
            );
        }
        // Batch (parallel)
        simd_binary_matmul_batch(&bw, &x, batch, &mut y_par);

        let max_diff = y_seq
            .iter()
            .zip(y_par.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "batch parity max_diff = {max_diff}");
    }
}

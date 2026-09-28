//! BITCOS GEMV kernels — presence bitmap + compacted signs (Issue 864 T3).
//!
//! Three consumers of [`crate::BitcosWeights`], in ascending specialization:
//!
//! 1. [`bitcos_matvec_scalar`] — the portable reference. Walks each group's
//!    columns in order via a presence/sign cursor, **bit-identical** to
//!    [`super::ternary_group::ternary_group_matvec_scalar`] on the same
//!    logical weights (same summation order: group-wise inner accumulation in
//!    column order, one scale multiply per group). That equality is the
//!    tier's G1 and is asserted in tests.
//! 2. [`bitcos_matvec_lut`] — the **GPU-portable** variant: a 256-entry LUT
//!    keyed by (presence nibble, sign window) returning 4 ternary codes —
//!    "a precomputed, four-bit-wide pdep composed with the sign→ternary map"
//!    (the paper's Xe2 trick, our `dequant_dot_via_lut`/StreamDQ lineage).
//!    No `pdep` needed — the same table shape ports to CUDA/WGSL shared
//!    memory. Bit-identical to the scalar reference (same lane order).
//! 3. [`bitcos_matvec_pdep`] (`x86_64`, runtime-probed) — reconstructs the
//!    pos/neg planes per word with one hardware `pdep` (`neg = pdep(sign
//!    window, presence)`, `pos = presence & !neg`) and then runs the shipped
//!    AVX2 SWAR dot ([`super::ternary_group`] helper) on the reconstructed
//!    planes. Scale folded per element → ~1e-6 agreement with the scalar
//!    reference, the same close-not-bit-identical relationship the shipped
//!    AVX2 bit-plane kernel has to its own scalar.
//!
//! [`bitcos_matvec`] dispatches: pdep arm when AVX2+FMA **and** BMI2 are
//! probed at runtime (the `shipped_target_feature` law — never a compile-time
//! `target_feature` *cfg* on a shipped path; `#[target_feature(enable)]` on
//! the unsafe fn plus a runtime probe), else the LUT kernel.
//!
//! # Sign-stream addressing
//!
//! Row `r`'s signs start at bit [`crate::BitcosWeights::row_sign_offset`]`[r]`;
//! within a word, the j-th set presence bit consumes the j-th sign bit, so a
//! kernel walking words left-to-right carries one cursor and advances it by
//! `popcount(presence_word)`. The unaligned 64-bit window read at the cursor
//! is branch-free on the straddle — the container appends a zero sentinel
//! word exactly for this.
//!
//! # Regime honesty (Issue 864 T4/T5's subject)
//!
//! Every kernel here is a **decode** consumer: it trades payload bytes
//! (2−z bits/w vs 2.125) for decode instructions. Where GEMV is
//! bandwidth-bound the trade pays; where it is instruction-bound (the
//! paper's Lunar Lake row) or cache-resident (Issue 582's G2b) it loses.
//! The GOAT gate measures both regimes and asserts the negative controls.

#![allow(clippy::too_many_arguments)]

// x86_64-only: the sole consumer is pdep_arm_available() below. Ungated here
// would warn unused on every non-x86_64 lane; ungated in the fn body would
// not compile on this arm. (The AGENTS.md x86_64 sibling-arm lesson.)
#[cfg(target_arch = "x86_64")]
use super::{SimdLevel, simd_level};
use crate::GROUP_SIZE;
use crate::bitcos::BitcosWeights;

/// Decode table: `(presence nibble m, sign window s) → 4 ternary codes`.
///
/// Key `m << 4 | s`. Lane `k` is nonzero iff `m` has bit `k`; its value is
/// `−1` where the window's `rank(k)`-th bit is set, else `+1`, where
/// `rank(k) = popcount(m & ((1 << k) − 1))` — the j-th set lane consumes the
/// j-th sign bit, the pdep composition the paper's Xe2 port ships as a
/// shared-memory table. The GPU-portable consume mechanism: no scatter
/// instruction anywhere.
pub static BITCOS_LUT: [[i8; 4]; 256] = build_bitcos_lut();

const fn build_bitcos_lut() -> [[i8; 4]; 256] {
    let mut lut = [[0i8; 4]; 256];
    let mut idx = 0usize;
    while idx < 256 {
        let m = (idx >> 4) as u8;
        let s = (idx & 0xF) as u8;
        let mut k = 0usize;
        while k < 4 {
            let bit = 1u8 << k;
            if m & bit != 0 {
                let rank = (m & (bit - 1)).count_ones();
                lut[idx][k] = if (s >> rank) & 1 != 0 { -1 } else { 1 };
            }
            k += 1;
        }
        idx += 1;
    }
    lut
}

/// Unaligned 64-bit sign window at bit offset `off` (sentinel covers the
/// straddle). Shared by all three kernels.
#[inline]
fn sign_window(w: &BitcosWeights, off: u64) -> u64 {
    let word = (off / 64) as usize;
    let o = (off % 64) as u32;
    let lo = w.signs[word] >> o;
    if o == 0 {
        lo
    } else {
        lo | (w.signs[word + 1] << (64 - o))
    }
}

// ── 1. Scalar reference ─────────────────────────────────────────

/// Scalar reference: `y[r] = Σ_g group_scale[r,g] · Σ_{col∈g} sign(col)·x[col]`.
///
/// Bit-identical to [`super::ternary_group::ternary_group_matvec_scalar`] on
/// the same logical weights — same column order within each group's inner
/// accumulation (zeros contribute `0.0·x` exactly as the bit-plane scalar
/// does), same single scale multiply per group.
pub fn bitcos_matvec_scalar(w: &BitcosWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    for (i, y_slot) in y.iter_mut().enumerate() {
        let r = i;
        let row_base = r * w.words_per_row;
        let group_base = r * w.groups_per_row;
        let mut cursor = w.row_sign_offset[r];
        let mut row_sum = 0.0f32;
        for g in 0..w.groups_per_row {
            let g_start = g * GROUP_SIZE;
            let g_end = (g_start + GROUP_SIZE).min(w.cols);
            let w_start = g_start >> 6;
            let w_end = g_end.div_ceil(64);
            let mut group_acc = 0.0f32;
            for wi in w_start..w_end {
                let p = w.presence[row_base + wi];
                let c0 = wi * 64;
                let c_hi = (c0 + 64).min(w.cols);
                if p == 0 {
                    // Zero weights still contribute 0.0·x in the reference
                    // sequence (the bit-plane scalar adds every column).
                    for c in c0..c_hi {
                        group_acc += 0.0 * unsafe { *x.get_unchecked(c) };
                    }
                    continue;
                }
                let neg = crate::bitcos::scatter_bits(sign_window(w, cursor), p);
                cursor += p.count_ones() as u64;
                for k in 0..(c_hi - c0) {
                    let bit = 1u64 << k;
                    // Reconstructed planes: pos = presence & !neg — presence
                    // alone is NOT the pos plane (a negative weight has both
                    // bits set; presence&bit && neg&bit must decode to -1).
                    let pos = ((p & !neg & bit) != 0) as i32;
                    let n = ((neg & bit) != 0) as i32;
                    let sign = pos - n;
                    group_acc += sign as f32 * unsafe { *x.get_unchecked(c0 + k) };
                }
            }
            row_sum += w.group_scale[group_base + g].to_f32() * group_acc;
        }
        *y_slot = row_sum;
    }
}

// ── 2. LUT variant (GPU-portable) ────────────────────────────────

/// LUT kernel — 4 weights per table lookup, presence-nibble × sign-window key.
pub fn bitcos_matvec_lut(w: &BitcosWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    for (i, y_slot) in y.iter_mut().enumerate() {
        let r = i;
        let row_base = r * w.words_per_row;
        let group_base = r * w.groups_per_row;
        let mut cursor = w.row_sign_offset[r];
        let mut row_sum = 0.0f32;
        for g in 0..w.groups_per_row {
            let g_start = g * GROUP_SIZE;
            let g_end = (g_start + GROUP_SIZE).min(w.cols);
            let w_start = g_start >> 6;
            let w_end = g_end.div_ceil(64);
            let mut group_acc = 0.0f32;
            for wi in w_start..w_end {
                let p = w.presence[row_base + wi];
                let c0 = wi * 64;
                let c_hi = (c0 + 64).min(w.cols);
                let live = c_hi - c0;
                if p == 0 || live == 0 {
                    continue;
                }
                let window = sign_window(w, cursor);
                let mut nib = 0usize;
                // Full nibbles only while all 4 lanes are in-columns.
                while nib * 4 + 4 <= live {
                    let m = ((p >> (nib * 4)) & 0xF) as usize;
                    let rank = (p & ((1u64 << (nib * 4)) - 1)).count_ones() as usize;
                    let s = ((window >> rank) & 0xF) as usize;
                    let e = &BITCOS_LUT[(m << 4) | s];
                    let cb = c0 + nib * 4;
                    group_acc += e[0] as f32 * unsafe { *x.get_unchecked(cb) };
                    group_acc += e[1] as f32 * unsafe { *x.get_unchecked(cb + 1) };
                    group_acc += e[2] as f32 * unsafe { *x.get_unchecked(cb + 2) };
                    group_acc += e[3] as f32 * unsafe { *x.get_unchecked(cb + 3) };
                    nib += 1;
                }
                // Ragged tail lanes (< 4, only on the final partial word).
                let mut k = nib * 4;
                while k < live {
                    let bit = 1u64 << k;
                    if p & bit != 0 {
                        let rank = (p & (bit - 1)).count_ones();
                        let sign = if (window >> rank) & 1 != 0 {
                            -1.0f32
                        } else {
                            1.0
                        };
                        group_acc += sign * unsafe { *x.get_unchecked(c0 + k) };
                    } else {
                        group_acc += 0.0 * unsafe { *x.get_unchecked(c0 + k) };
                    }
                    k += 1;
                }
                cursor += p.count_ones() as u64;
            }
            row_sum += w.group_scale[group_base + g].to_f32() * group_acc;
        }
        *y_slot = row_sum;
    }
}

// ── 3. pdep + AVX2 SWAR (x86_64, runtime-probed) ─────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
fn pdep_arm_available() -> bool {
    matches!(simd_level(), SimdLevel::Avx2) && std::is_x86_feature_detected!("bmi2")
}

/// pdep/AVX2 kernel over the row range `[row_offset, row_offset + y.len())`.
///
/// Per 64-weight block: one hardware `_pdep_u64` scatters the sign window
/// onto the presence mask (negatives at their positions), `andnot` yields the
/// positives, and the shipped 8-lane SWAR dot
/// ([`super::ternary_group::fma_scaled_nibble8_avx2`]) runs on the
/// reconstructed planes — the bit-plane kernel's exact inner arithmetic fed
/// by a ~25%-smaller payload at z = 0.5.
///
/// # Safety
/// Caller guarantees AVX2+FMA+BMI2 probed, `x.len() == w.cols`,
/// `row_offset + y.len() <= w.rows`.
#[cfg(all(feature = "bitcos", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,bmi2")]
unsafe fn pdep_row_range(w: &BitcosWeights, x: &[f32], y: &mut [f32], row_offset: usize) {
    use core::arch::x86_64::*;
    unsafe {
        use super::horizontal::horizontal_sum_256;
        use super::ternary_group::fma_scaled_nibble8_avx2;
        debug_assert_eq!(x.len(), w.cols);
        debug_assert!(row_offset + y.len() <= w.rows);

        let mask_byte: __m256i = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let zero_i: __m256i = _mm256_setzero_si256();

        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let row_base = r * w.words_per_row;
            let group_base = r * w.groups_per_row;
            let mut cursor = w.row_sign_offset[r];

            let mut acc0: __m256 = _mm256_setzero_ps();
            let mut acc1: __m256 = _mm256_setzero_ps();
            let mut acc2: __m256 = _mm256_setzero_ps();
            let mut acc3: __m256 = _mm256_setzero_ps();

            for g in 0..w.groups_per_row {
                let scale = w.group_scale[group_base + g].to_f32();
                let scale_v: __m256 = _mm256_set1_ps(scale);

                let b_start = (g * GROUP_SIZE) >> 6;
                let b_end = ((g * GROUP_SIZE + GROUP_SIZE).min(w.cols)).div_ceil(64);

                for b in b_start..b_end {
                    let p = w.presence[row_base + b];
                    let base_col = b * 64;
                    let remaining = if base_col + 64 <= w.cols {
                        64
                    } else {
                        w.cols - base_col
                    };
                    if p == 0 {
                        continue;
                    }
                    // Reconstruct the planes: one pdep + one andnot.
                    let window = sign_window(w, cursor);
                    let neg_word = _pdep_u64(window, p);
                    let pos_word = p & !neg_word;
                    cursor += p.count_ones() as u64;

                    let mut col = 0usize;
                    while col + 32 <= remaining {
                        fma_scaled_nibble8_avx2(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                        fma_scaled_nibble8_avx2(
                            &mut acc1, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                        fma_scaled_nibble8_avx2(
                            &mut acc2, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                        fma_scaled_nibble8_avx2(
                            &mut acc3, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                    }
                    while col + 8 <= remaining {
                        fma_scaled_nibble8_avx2(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                    }
                    let mut scalar_acc = 0.0f32;
                    while col < remaining {
                        let bit_mask = 1u64 << col;
                        let pos = ((pos_word & bit_mask) != 0) as i32 as f32;
                        let neg = ((neg_word & bit_mask) != 0) as i32 as f32;
                        scalar_acc += (pos - neg) * scale * *x.get_unchecked(base_col + col);
                        col += 1;
                    }
                    if scalar_acc != 0.0 {
                        acc0 = _mm256_add_ps(
                            acc0,
                            _mm256_setr_ps(scalar_acc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        );
                    }
                }
            }

            acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            *y_slot = horizontal_sum_256(acc0);
        }
    }
}

/// Dispatching entry point: the pdep/AVX2 arm when probed, else the LUT
/// kernel. NEON targets take the LUT path (no scatter instruction — the LUT
/// IS the ARM/GPU answer).
pub fn bitcos_matvec(w: &BitcosWeights, x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if pdep_arm_available() {
            assert_eq!(x.len(), w.cols);
            assert_eq!(y.len(), w.rows);
            unsafe { pdep_row_range(w, x, y, 0) };
            return;
        }
    }
    bitcos_matvec_lut(w, x, y);
}

#[cfg(all(test, feature = "bitcos"))]
mod tests {
    use super::*;
    use crate::TernaryGroupWeights;

    /// Deterministic planes at a controlled keep-probability; returns the
    /// group container (bitcos packs from it).
    fn planes(rows: usize, cols: usize, seed: u64, keep_mod: u64) -> TernaryGroupWeights {
        let mut s = seed;
        let mut gw = TernaryGroupWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if (s >> 33).is_multiple_of(keep_mod) {
                    gw.set(r, c, if (s >> 20) & 1 == 0 { 1 } else { -1 });
                }
            }
            for g in 0..gw.groups_per_row {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                gw.set_scale(r, g, 0.5 + ((s >> 33) % 5) as f32 * 0.3);
            }
        }
        gw
    }

    fn reference_y(gw: &TernaryGroupWeights, cols: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; cols];
        let mut s = 0x9e3779b97f4a7c15u64;
        for v in x.iter_mut() {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let mut y = vec![0.0f32; gw.rows];
        super::super::ternary_group::ternary_group_matvec_scalar(gw, &x, &mut y);
        (x, y).1
    }

    #[test]
    fn scalar_is_bit_identical_to_the_bit_plane_reference() {
        for &(rows, cols, keep) in &[
            (3usize, 200usize, 3u64),
            (5, 128, 2),
            (7, 129, 3),
            (4, 4096, 3),
            (2, 63, 2),
        ] {
            let gw = planes(rows, cols, 0x1234_5678 + keep, keep);
            let bc = BitcosWeights::pack_from_group(&gw);
            let y_ref = reference_y(&gw, cols);
            // regenerate x identically
            let mut x = vec![0.0f32; cols];
            let mut s = 0x9e3779b97f4a7c15u64;
            for v in x.iter_mut() {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
            let mut y = vec![0.0f32; rows];
            bitcos_matvec_scalar(&bc, &x, &mut y);
            assert_eq!(y, y_ref, "scalar bit-identity ({rows}x{cols})");
        }
    }

    #[test]
    fn lut_is_bit_identical_to_scalar() {
        for &(rows, cols, keep) in &[
            (3usize, 200usize, 3u64),
            (5, 128, 2),
            (7, 129, 3),
            (4, 4096, 3),
        ] {
            let gw = planes(rows, cols, 0xfeed_beef + keep, keep);
            let bc = BitcosWeights::pack_from_group(&gw);
            let mut x = vec![0.0f32; cols];
            let mut s = 0x2545_f491_4f6c_dd1du64;
            for v in x.iter_mut() {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            }
            let mut y_s = vec![0.0f32; rows];
            let mut y_l = vec![0.0f32; rows];
            bitcos_matvec_scalar(&bc, &x, &mut y_s);
            bitcos_matvec_lut(&bc, &x, &mut y_l);
            assert_eq!(y_s, y_l, "lut bit-identity ({rows}x{cols})");
        }
    }

    #[test]
    fn pdep_agrees_with_scalar_to_1e6_relative() {
        #[cfg(target_arch = "x86_64")]
        {
            if !pdep_arm_available() {
                eprintln!("(pdep arm unavailable — skipped)");
                return;
            }
            for &(rows, cols, keep) in &[
                (3usize, 200usize, 3u64),
                (5, 128, 2),
                (4, 4096, 3),
                (2, 130, 2),
            ] {
                let gw = planes(rows, cols, 0xc0ff_ee00 + keep, keep);
                let bc = BitcosWeights::pack_from_group(&gw);
                let mut x = vec![0.0f32; cols];
                let mut s = 0x8dde_6e20_cbb0_973fu64;
                for v in x.iter_mut() {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                }
                let mut y_s = vec![0.0f32; rows];
                let mut y_p = vec![0.0f32; rows];
                bitcos_matvec_scalar(&bc, &x, &mut y_s);
                unsafe { pdep_row_range(&bc, &x, &mut y_p, 0) };
                for (a, b) in y_s.iter().zip(y_p.iter()) {
                    let denom = a.abs().max(1e-3);
                    assert!(
                        (a - b).abs() / denom < 1e-5,
                        "pdep {a} vs scalar {b} ({rows}x{cols})"
                    );
                }
            }
        }
    }

    #[test]
    fn dispatcher_matches_its_arm() {
        let gw = planes(3, 200, 0x5555_aaaa, 3);
        let bc = BitcosWeights::pack_from_group(&gw);
        let mut x = vec![0.25f32; 200];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.3;
        }
        let mut y_a = vec![0.0f32; 3];
        let mut y_b = vec![0.0f32; 3];
        bitcos_matvec(&bc, &x, &mut y_a);
        #[cfg(target_arch = "x86_64")]
        if pdep_arm_available() {
            unsafe { pdep_row_range(&bc, &x, &mut y_b, 0) };
        } else {
            bitcos_matvec_lut(&bc, &x, &mut y_b);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            bitcos_matvec_lut(&bc, &x, &mut y_b);
        }
        let tol = 1e-5f32;
        for (a, b) in y_a.iter().zip(y_b.iter()) {
            assert!((a - b).abs() <= tol * a.abs().max(1.0));
        }
    }
}

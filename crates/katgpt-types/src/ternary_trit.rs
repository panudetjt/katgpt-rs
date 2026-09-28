//! Base-3 trit-packed ternary weights with group-wise f16 scale (Issue 582).
//!
//! The footprint tier. Same alphabet and same scale granularity as
//! [`crate::TernaryGroupWeights`], but the weights are packed **5 trits per
//! byte** in base 3 instead of two 1-bit planes.
//!
//! ## Why
//!
//! A trit carries `log2(3) = 1.585` bits, but two bit-planes spend 2 bits on it
//! and leave the fourth codepoint (`pos & neg`) forbidden by the representation
//! invariant. `3^5 = 243 <= 256`, so five trits fit in one byte with 13 unused
//! codes instead of one wasted bit per weight:
//!
//! | Layout | per 128-weight group | bits/weight | Bonsai-27B |
//! |---|---|---|---|
//! | two bit-planes ([`crate::TernaryGroupWeights`]) | 32 B + 2 B scale = 34 B | 2.125 | 7.17 GB |
//! | `Q2_0` 2-bit slots (GGUF on disk) | 32 B + 2 B scale = 34 B | 2.125 | 7.17 GB |
//! | **5 trits per byte** (this tier) | 26 B + 2 B scale = 28 B | **1.75** | **5.90 GB** |
//!
//! −17.6%, which is the "~5.9 GB ideal" Issue 578 recorded as unreachable with
//! 2-bit slots. Ternary GEMV is memory-bound on every backend we ship, so the
//! traffic cut is the point; the arithmetic is unchanged.
//!
//! ## Encoding
//!
//! Byte `i` holds weights `5i .. 5i+5`, least-significant trit first:
//!
//! ```text
//! byte = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4     where t = weight + 1 in {0,1,2}
//! ```
//!
//! so `weight = t - 1` in `{-1, 0, +1}`. A byte `>= 243` is **not** a fourth
//! state — it is corruption, and [`TernaryTritWeights::is_canonical`] rejects
//! it. Trailing trits in the final byte of a row are zero-padded (encoding
//! `0`, i.e. weight `0`), so canonical bytes stay `< 243` even on a ragged row.
//!
//! ## Alignment hazard — groups split bytes
//!
//! This is the one structural difference from the bit-plane tier, and every
//! kernel has to respect it. There, `GROUP_SIZE = 128` is exactly two 64-bit
//! blocks, so a group boundary never splits a word. Here `128 % 5 == 3`, so
//! **group boundaries land mid-byte**: group 0 owns weights `0..128` = bytes
//! `0..25` plus the low 3 trits of byte 25, and group 1 picks up at the high 2
//! trits of that same byte.
//!
//! Consequence: a byte can straddle two groups, and the two halves may need
//! *different* scales. Kernels therefore walk **weights within a group** (using
//! the LUT-decoded byte as a 5-lane window) rather than walking whole bytes.
//! [`TernaryTritWeights::group_byte_span`] returns the byte range a group
//! touches, inclusive of shared end bytes.

use crate::GROUP_SIZE;
use half::f16;

/// Trits packed per byte. `3^5 = 243 <= 256`; a sixth would need `3^6 = 729`.
pub const TRITS_PER_BYTE: usize = 5;

/// `3^5` — one past the largest canonical byte value.
pub const TRIT_CODE_LIMIT: u8 = 243;

/// Powers of three for the five trit positions within a byte.
pub const TRIT_POW3: [u8; TRITS_PER_BYTE] = [1, 3, 9, 27, 81];

/// Ternary `{-1, 0, +1}` weights packed 5-per-byte in base 3, with a
/// per-128-weight f16 scale.
///
/// `trits[r * bytes_per_row + i]` holds weights `5i .. 5i+5` of row `r`.
/// `group_scale[r * groups_per_row + g]` rescales group `g` of row `r`.
///
/// **Representation invariant:** every byte is `< 243`
/// ([`Self::is_canonical`]). Unlike the bit-plane tier there is no
/// "impossible pair" to check — the invariant is a range check, and it is the
/// corruption signal a mis-parsed load trips.
#[cfg(feature = "ternary_trit_pack")]
#[derive(Clone, Debug)]
pub struct TernaryTritWeights {
    pub rows: usize,
    pub cols: usize,
    pub bytes_per_row: usize,  // cols.div_ceil(TRITS_PER_BYTE)
    pub groups_per_row: usize, // cols.div_ceil(GROUP_SIZE)
    pub trits: Vec<u8>,        // [rows * bytes_per_row]
    pub group_scale: Vec<f16>, // [rows * groups_per_row]
}

/// Decode table: byte → its 5 signed trit values, padded to 8 lanes.
///
/// Padding to 8 makes each entry an aligned 8-byte copy, so the hot path can
/// splat a whole entry with one `u64`-width move instead of five byte writes.
/// The three pad lanes are `0` and are never read as weights (the caller knows
/// only 5 are live), so a stray read contributes `0 * x = 0`.
///
/// Non-canonical bytes (`>= 243`) decode to all-zero rather than garbage —
/// defence in depth behind [`TernaryTritWeights::is_canonical`].
pub static TRIT_LUT: [[i8; 8]; 256] = build_trit_lut();

const fn build_trit_lut() -> [[i8; 8]; 256] {
    let mut lut = [[0i8; 8]; 256];
    let mut byte = 0usize;
    while byte < 243 {
        let mut rem = byte;
        let mut k = 0usize;
        while k < TRITS_PER_BYTE {
            lut[byte][k] = (rem % 3) as i8 - 1;
            rem /= 3;
            k += 1;
        }
        byte += 1;
    }
    // 243..=255 stay all-zero: non-canonical, never produced by `set`.
    lut
}

#[cfg(feature = "ternary_trit_pack")]
impl TernaryTritWeights {
    /// Create all-zero weights with unit group scale.
    ///
    /// The zero weight encodes as trit `1`, so an all-zero *weight* matrix is
    /// **not** an all-zero byte buffer — every byte is `1+3+9+27+81 = 121`.
    pub fn new(rows: usize, cols: usize) -> Self {
        let bytes_per_row = cols.div_ceil(TRITS_PER_BYTE);
        let groups_per_row = cols.div_ceil(GROUP_SIZE);
        // All-zero weights: every live trit is 1. Pad trits are also 1 here,
        // which keeps `set`'s read-modify-write uniform and every byte < 243.
        let all_zero_byte: u8 = TRIT_POW3.iter().sum();
        Self {
            rows,
            cols,
            bytes_per_row,
            groups_per_row,
            trits: vec![all_zero_byte; rows * bytes_per_row],
            group_scale: vec![f16::ONE; rows * groups_per_row],
        }
    }

    /// Byte index and trit position within that byte for column `col`.
    #[inline]
    fn locate(&self, row: usize, col: usize) -> (usize, usize) {
        (
            row * self.bytes_per_row + col / TRITS_PER_BYTE,
            col % TRITS_PER_BYTE,
        )
    }

    /// Set the ternary value at `(row, col)`. Panics if out of bounds or if
    /// `value` is not in `{-1, 0, +1}`.
    ///
    /// Read-modify-write on the containing byte: subtract the old trit's
    /// contribution, add the new one. Keeps every other trit in the byte
    /// untouched and the byte canonical by construction.
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        assert!(
            (-1..=1).contains(&value),
            "ternary value must be -1, 0, or +1"
        );
        let (idx, k) = self.locate(row, col);
        let pow = TRIT_POW3[k];
        let old = TRIT_LUT[self.trits[idx] as usize][k] + 1; // 0..=2
        let new = (value + 1) as u8;
        self.trits[idx] = self.trits[idx] - old as u8 * pow + new * pow;
    }

    /// Get the ternary value at `(row, col)`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let (idx, k) = self.locate(row, col);
        TRIT_LUT[self.trits[idx] as usize][k]
    }

    /// Scale applied to group `g` of row `r`.
    #[inline]
    pub fn scale_at(&self, row: usize, group: usize) -> f32 {
        self.group_scale[row * self.groups_per_row + group].to_f32()
    }

    /// Set the scale for group `g` of row `r`.
    #[inline]
    pub fn set_scale(&mut self, row: usize, group: usize, scale: f32) {
        self.group_scale[row * self.groups_per_row + group] = f16::from_f32(scale);
    }

    /// Half-open byte range that group `g` touches, and the trit offset of the
    /// group's first weight within the first byte.
    ///
    /// The end byte is **inclusive of sharing**: when `GROUP_SIZE % 5 != 0` the
    /// last byte is shared with the next group, and it is still returned here
    /// because the decode has to read it. Callers must clamp by weight index,
    /// not by byte index — see the module docs' alignment hazard.
    #[inline]
    pub fn group_byte_span(&self, group: usize) -> (usize, usize, usize) {
        let w_start = group * GROUP_SIZE;
        let w_end = (w_start + GROUP_SIZE).min(self.cols);
        let b_start = w_start / TRITS_PER_BYTE;
        let b_end = w_end.div_ceil(TRITS_PER_BYTE);
        (b_start, b_end, w_start % TRITS_PER_BYTE)
    }

    /// Check the representation invariant: every byte is a canonical base-3
    /// encoding (`< 243`).
    ///
    /// G1 gate helper and the corruption signal for a mis-parsed load — the
    /// analogue of [`crate::TernaryGroupWeights::invariant_holds`].
    pub fn is_canonical(&self) -> bool {
        self.trits.iter().all(|&b| b < TRIT_CODE_LIMIT)
    }

    /// Repack from the bit-plane tier. Lossless in both weights and scale —
    /// same alphabet, same group granularity, same f16 scale values.
    #[cfg(feature = "ternary_group_scale")]
    pub fn from_group(gw: &crate::TernaryGroupWeights) -> Self {
        let mut out = Self::new(gw.rows, gw.cols);
        for r in 0..gw.rows {
            for c in 0..gw.cols {
                out.set(r, c, gw.get(r, c));
            }
        }
        out.group_scale.copy_from_slice(&gw.group_scale);
        out
    }

    /// Repack back into the bit-plane tier. Lossless — the inverse of
    /// [`Self::from_group`], so `from_group(w).to_group()` is bit-identical to
    /// `w` (G1).
    #[cfg(feature = "ternary_group_scale")]
    pub fn to_group(&self) -> crate::TernaryGroupWeights {
        let mut out = crate::TernaryGroupWeights::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(r, c, self.get(r, c));
            }
        }
        out.group_scale.copy_from_slice(&self.group_scale);
        out
    }

    /// Quantize f32 weights with group-wise error compensation.
    ///
    /// Identical arithmetic to
    /// [`crate::TernaryGroupWeights::quantize_from_f32`] — same mean-abs scale,
    /// same `0.5 * scale` threshold, same carry — so the two tiers quantize the
    /// *same* input to the *same* weights. Only the storage differs.
    pub fn quantize_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(
            weights.len(),
            rows * cols,
            "weights slice must be rows*cols"
        );
        let mut out = Self::new(rows, cols);

        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let group_base = r * out.groups_per_row;

            for g in 0..out.groups_per_row {
                let g_start = g * GROUP_SIZE;
                let g_end = (g_start + GROUP_SIZE).min(cols);
                let group = &row[g_start..g_end];

                let abs_sum: f32 = group.iter().map(|v| v.abs()).sum();
                let scale = if abs_sum > 0.0 {
                    abs_sum / group.len() as f32
                } else {
                    1.0
                };
                out.group_scale[group_base + g] = f16::from_f32(scale);
                // Quantize against the f16-rounded scale the kernel applies,
                // not the f32 ideal — same reasoning as the bit-plane tier.
                let scale = out.group_scale[group_base + g].to_f32();

                let threshold = 0.5 * scale;
                let mut carry = 0.0f32;
                for (i, &val) in group.iter().enumerate() {
                    let adjusted = val + carry;
                    let q = match adjusted {
                        a if a > threshold => 1i8,
                        a if a < -threshold => -1i8,
                        _ => 0i8,
                    };
                    out.set(r, g_start + i, q);
                    carry = adjusted - (q as f32 * scale);
                }
            }
        }

        out
    }

    /// Checksum over all values: `Σ_r Σ_g scale[r,g] * Σ_{col∈g} weight`.
    ///
    /// Same definition as [`crate::TernaryGroupWeights::checksum`], so the two
    /// are directly comparable for cross-tier verification.
    pub fn checksum(&self) -> f32 {
        let mut total = 0.0f32;
        for r in 0..self.rows {
            let group_base = r * self.groups_per_row;
            for g in 0..self.groups_per_row {
                let w_start = g * GROUP_SIZE;
                let w_end = (w_start + GROUP_SIZE).min(self.cols);
                let mut sum: i32 = 0;
                for c in w_start..w_end {
                    sum += self.get(r, c) as i32;
                }
                total += self.group_scale[group_base + g].to_f32() * sum as f32;
            }
        }
        total
    }

    /// Bytes of weight payload (trits + scales), excluding `Vec` overhead.
    ///
    /// `8/5 bits/weight + 16 bits per GROUP_SIZE weights` = **1.75
    /// bits/weight** at `GROUP_SIZE = 128`, vs the bit-plane tier's 2.125.
    pub fn encoded_bytes(&self) -> usize {
        self.trits.len() + self.group_scale.len() * 2
    }
}

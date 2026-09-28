//! Bit-plane packed ternary weights.

/// Why a dense `[i8]` buffer could not be packed into [`TernaryWeights`].
#[cfg(feature = "plasma_path")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryPackError {
    /// The buffer is not `rows * cols` long.
    LengthMismatch { got: usize, want: usize },
    /// A value outside `{-1, 0, +1}` — this container has no representation for
    /// it, and truncating would silently corrupt the weights.
    NotTernary { index: usize, value: i8 },
}

#[cfg(feature = "plasma_path")]
impl core::fmt::Display for TernaryPackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { got, want } => {
                write!(f, "i8 buffer has {got} values, expected {want}")
            }
            Self::NotTernary { index, value } => {
                write!(f, "value {value} at index {index} is not in {{-1, 0, +1}}")
            }
        }
    }
}

#[cfg(feature = "plasma_path")]
impl core::error::Error for TernaryPackError {}

/// Bit-plane packed ternary weights: each element is {-1, 0, +1}.
///
/// 64 weights per block stored as two u64 bitmasks:
/// - `pos_bits[block]` bit k set → `weight[row][k] = +1`
/// - `neg_bits[block]` bit k set → `weight[row][k] = -1`
/// - both zero → weight = 0 (implicit skip, no storage needed)
///
/// `row_scale[r]` rescales the accumulated sum back toward original float magnitudes.
/// Memory: 2 bits/weight (two bit-planes — the log₂3 ≈ 1.585 figure is the
/// information-theoretic floor for ternary symbols, not this layout's rate;
/// see ternary_trit.rs for the container that approaches it), plus one f32
/// per row for scale.
#[cfg(feature = "plasma_path")]
#[derive(Clone, Debug)]
pub struct TernaryWeights {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,     // (cols + 63) / 64
    pub pos_bits: Vec<u64>,  // [rows * blocks64]
    pub neg_bits: Vec<u64>,  // [rows * blocks64]
    pub row_scale: Vec<f32>, // [rows]
}

#[cfg(feature = "plasma_path")]
impl TernaryWeights {
    /// Create zeroed ternary weights.
    pub fn new(rows: usize, cols: usize) -> Self {
        let blocks64 = cols.div_ceil(64);
        Self {
            rows,
            cols,
            blocks64,
            pos_bits: vec![0u64; rows * blocks64],
            neg_bits: vec![0u64; rows * blocks64],
            row_scale: vec![1.0f32; rows],
        }
    }

    /// Set a single ternary value at (row, col). Panics if out of bounds or value not in {-1, 0, +1}.
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        assert!(
            (-1..=1).contains(&value),
            "ternary value must be -1, 0, or +1"
        );
        let block = col >> 6;
        let bit = col & 63;
        let mask = 1u64 << bit;
        let idx = row * self.blocks64 + block;
        match value {
            1 => {
                self.pos_bits[idx] |= mask;
                self.neg_bits[idx] &= !mask;
            }
            -1 => {
                self.pos_bits[idx] &= !mask;
                self.neg_bits[idx] |= mask;
            }
            0 => {
                self.pos_bits[idx] &= !mask;
                self.neg_bits[idx] &= !mask;
            }
            _ => unreachable!(),
        }
    }

    /// Get the ternary value at (row, col).
    pub fn get(&self, row: usize, col: usize) -> i8 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let block = col >> 6;
        let bit = col & 63;
        let mask = 1u64 << bit;
        let idx = row * self.blocks64 + block;
        let pos = (self.pos_bits[idx] & mask) != 0;
        let neg = (self.neg_bits[idx] & mask) != 0;
        pos as i8 - neg as i8
    }

    /// Quantize f32 weights to ternary with row-wise error compensation.
    ///
    /// For each row:
    ///   scale = mean(|row|)
    ///   threshold = 0.5 * scale
    ///   for each weight: adjusted = value + carry
    ///     if adjusted > threshold → +1
    ///     if adjusted < -threshold → -1
    ///     else → 0
    ///     carry = adjusted - (q * scale)
    pub fn quantize_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(
            weights.len(),
            rows * cols,
            "weights slice must be rows*cols"
        );
        let mut tw = Self::new(rows, cols);

        for r in 0..rows {
            let row_start = r * cols;
            let row = &weights[row_start..row_start + cols];

            // Compute scale = mean(|row|)
            let abs_sum = crate::simd::simd_sum_abs_f32(row);
            let scale = abs_sum / cols as f32;
            tw.row_scale[r] = if scale > 0.0 { scale } else { 1.0 };

            let threshold = 0.5 * tw.row_scale[r];
            let mut carry = 0.0f32;

            // Inline bit manipulation to avoid per-element bounds checks in set()
            let row_base = r * tw.blocks64;
            for (c, &val) in row.iter().enumerate() {
                let adjusted = val + carry;
                let q = if adjusted > threshold {
                    1i8
                } else if adjusted < -threshold {
                    -1i8
                } else {
                    0i8
                };
                let block = c >> 6;
                let bit = c & 63;
                let mask = 1u64 << bit;
                let idx = row_base + block;
                // Branch-free: clear both bits, then set the one that matches q
                tw.pos_bits[idx] &= !mask;
                tw.neg_bits[idx] &= !mask;
                // q is 1 or -1 or 0; only set the relevant bit
                tw.pos_bits[idx] |= (q == 1) as u64 * mask;
                tw.neg_bits[idx] |= (q == -1) as u64 * mask;
                carry = adjusted - (q as f32 * tw.row_scale[r]);
            }
        }

        tw
    }

    /// Bulk-unpack the bit-planes into a dense row-major `[i8]` buffer.
    ///
    /// The fast path for `[i8]`-shaped consumers — notably LoTA-QAF's ternary
    /// LoRA (`riir-train-engine::lota_ternary`), whose `ternary_sign_sgd_step` /
    /// `ternary_merge` / `apply_fol_rule` all operate on dense `&mut [i8]`.
    /// [`Self::get`] is the correct-but-slow reference path (two bounds asserts
    /// and full index arithmetic per element); this decodes a whole 64-weight
    /// block at a time.
    ///
    /// `row_scale` is **not** applied — the output is the raw sign alphabet
    /// `{-1, 0, +1}`, which is what the `[i8]` consumers expect.
    ///
    /// Allocation-free: writes into a caller-owned buffer.
    ///
    /// # Panics
    /// Panics if `dst.len() != rows * cols`.
    pub fn unpack_into(&self, dst: &mut [i8]) {
        assert_eq!(
            dst.len(),
            self.rows * self.cols,
            "dst must be rows*cols i8 values"
        );
        for r in 0..self.rows {
            let row_base = r * self.blocks64;
            let out_row = &mut dst[r * self.cols..(r + 1) * self.cols];
            for (b, chunk) in out_row.chunks_mut(64).enumerate() {
                let idx = row_base + b;
                let pos = self.pos_bits[idx];
                let neg = self.neg_bits[idx];
                // Branch-free per element; the chunk is 64 wide except possibly
                // the last, so LLVM can unroll the common case.
                for (bit, slot) in chunk.iter_mut().enumerate() {
                    let p = ((pos >> bit) & 1) as i8;
                    let n = ((neg >> bit) & 1) as i8;
                    *slot = p - n;
                }
            }
        }
    }

    /// Allocating convenience wrapper over [`Self::unpack_into`].
    pub fn to_i8(&self) -> Vec<i8> {
        let mut out = vec![0i8; self.rows * self.cols];
        self.unpack_into(&mut out);
        out
    }

    /// Overwrite the bit-planes from a dense row-major `[i8]` buffer, keeping
    /// `rows`, `cols`, and `row_scale` untouched.
    ///
    /// The return leg of the `[i8]` bridge: unpack → mutate in `[i8]` space
    /// (t-SignSGD step, FOL rule injection) → pack back, without disturbing the
    /// scales the matvec kernel reads.
    ///
    /// # Errors
    /// [`TernaryPackError::LengthMismatch`] if `src.len() != rows * cols`, and
    /// [`TernaryPackError::NotTernary`] if any value is outside `{-1, 0, +1}`.
    /// The latter is the load-bearing check: `lota_ternary::ternary_merge`
    /// accumulates `base + Σ_k a_k·b_k` and clamps to the **full int8 range**,
    /// so a merged base is *not* ternary and must not be silently truncated
    /// back into this container. Rejecting it loudly is the point.
    ///
    /// On error, `self` is left unmodified.
    pub fn overwrite_from_i8(&mut self, src: &[i8]) -> Result<(), TernaryPackError> {
        if src.len() == self.rows * self.cols {
        } else {
            return Err(TernaryPackError::LengthMismatch {
                got: src.len(),
                want: self.rows * self.cols,
            });
        }
        // Validate before writing so a rejected buffer cannot leave `self`
        // half-updated.
        if let Some(pos) = src.iter().position(|&v| !(-1..=1).contains(&v)) {
            return Err(TernaryPackError::NotTernary {
                index: pos,
                value: src[pos],
            });
        }
        for r in 0..self.rows {
            let row_base = r * self.blocks64;
            let in_row = &src[r * self.cols..(r + 1) * self.cols];
            for (b, chunk) in in_row.chunks(64).enumerate() {
                let mut pos = 0u64;
                let mut neg = 0u64;
                for (bit, &v) in chunk.iter().enumerate() {
                    pos |= ((v == 1) as u64) << bit;
                    neg |= ((v == -1) as u64) << bit;
                }
                let idx = row_base + b;
                self.pos_bits[idx] = pos;
                self.neg_bits[idx] = neg;
            }
        }
        Ok(())
    }

    /// Build fresh bit-planes from a dense row-major `[i8]` buffer, with
    /// `row_scale` defaulted to `1.0`.
    ///
    /// # Errors
    /// See [`Self::overwrite_from_i8`].
    pub fn pack_from_i8(src: &[i8], rows: usize, cols: usize) -> Result<Self, TernaryPackError> {
        let mut tw = Self::new(rows, cols);
        tw.overwrite_from_i8(src)?;
        Ok(tw)
    }

    /// Compute a checksum over all values (sum of `row_scale[r]` * sum of signs in row r).
    /// Used for cross-implementation verification.
    pub fn checksum(&self) -> f32 {
        let mut total = 0.0f32;
        for r in 0..self.rows {
            // Accumulate as integer to avoid per-element f32 conversion overhead.
            let mut row_sum: i32 = 0;
            let row_base = r * self.blocks64;
            for b in 0..self.blocks64 {
                let idx = row_base + b;
                row_sum += self.pos_bits[idx].count_ones() as i32;
                row_sum -= self.neg_bits[idx].count_ones() as i32;
            }
            total += self.row_scale[r] * row_sum as f32;
        }
        total
    }
}

#[cfg(all(test, feature = "plasma_path"))]
mod tests {
    use super::*;

    /// Not a multiple of 64, so the last block is partial — the case a
    /// chunk-of-64 loop gets wrong if it reads past `cols`.
    const ROWS: usize = 3;
    const COLS: usize = 70;

    fn seeded() -> TernaryWeights {
        let mut tw = TernaryWeights::new(ROWS, COLS);
        for r in 0..ROWS {
            for c in 0..COLS {
                // Deterministic mix of all three states, phase-shifted per row.
                let v = match (r + c) % 3 {
                    0 => 0i8,
                    1 => 1,
                    _ => -1,
                };
                tw.set(r, c, v);
            }
            tw.row_scale[r] = 0.5 + r as f32;
        }
        tw
    }

    #[test]
    fn unpack_matches_get_elementwise() {
        let tw = seeded();
        let dense = tw.to_i8();
        assert_eq!(dense.len(), ROWS * COLS);
        for r in 0..ROWS {
            for c in 0..COLS {
                assert_eq!(
                    dense[r * COLS + c],
                    tw.get(r, c),
                    "bulk unpack disagrees with get() at ({r}, {c})"
                );
            }
        }
    }

    #[test]
    fn round_trip_is_bit_identical() {
        let tw = seeded();
        let dense = tw.to_i8();
        let packed = TernaryWeights::pack_from_i8(&dense, ROWS, COLS).expect("ternary values");
        assert_eq!(packed.pos_bits, tw.pos_bits, "pos_bits diverged");
        assert_eq!(packed.neg_bits, tw.neg_bits, "neg_bits diverged");
    }

    #[test]
    fn partial_last_block_leaves_padding_bits_clear() {
        // cols = 70 → block 1 holds 6 live bits; bits 6..64 must stay zero or
        // checksum()/popcount kernels would count phantom weights.
        let tw = seeded();
        let packed = TernaryWeights::pack_from_i8(&tw.to_i8(), ROWS, COLS).unwrap();
        let live = COLS - 64;
        let pad = !0u64 << live;
        for r in 0..ROWS {
            let idx = r * packed.blocks64 + 1;
            assert_eq!(packed.pos_bits[idx] & pad, 0, "pos padding set in row {r}");
            assert_eq!(packed.neg_bits[idx] & pad, 0, "neg padding set in row {r}");
        }
    }

    #[test]
    fn overwrite_preserves_row_scale() {
        let mut tw = seeded();
        let scales = tw.row_scale.clone();
        let mut dense = tw.to_i8();
        // Flip every sign — the mutation an `[i8]`-space optimizer step makes.
        for v in dense.iter_mut() {
            *v = -*v;
        }
        tw.overwrite_from_i8(&dense).expect("still ternary");
        assert_eq!(
            tw.row_scale, scales,
            "row_scale must survive the round trip"
        );
        for r in 0..ROWS {
            for c in 0..COLS {
                assert_eq!(tw.get(r, c), dense[r * COLS + c]);
            }
        }
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let mut tw = seeded();
        let short = vec![0i8; ROWS * COLS - 1];
        assert_eq!(
            tw.overwrite_from_i8(&short),
            Err(TernaryPackError::LengthMismatch {
                got: ROWS * COLS - 1,
                want: ROWS * COLS,
            })
        );
    }

    /// The load-bearing guard for Plan 333 T3.3b: `lota_ternary::ternary_merge`
    /// accumulates `base + Σ_k a_k·b_k` and clamps to the *full* int8 range, so
    /// a merged base is int8, not ternary. It must not be silently truncated
    /// back into this container.
    #[test]
    fn merged_int8_values_are_rejected_not_truncated() {
        let mut tw = seeded();
        let before_pos = tw.pos_bits.clone();
        let before_neg = tw.neg_bits.clone();
        let mut dense = tw.to_i8();
        dense[5] = 7; // what a rank-16 merge can produce

        assert_eq!(
            tw.overwrite_from_i8(&dense),
            Err(TernaryPackError::NotTernary { index: 5, value: 7 })
        );
        assert_eq!(tw.pos_bits, before_pos, "self must be left unmodified");
        assert_eq!(tw.neg_bits, before_neg, "self must be left unmodified");
    }

    #[test]
    fn unpack_into_rejects_a_wrong_sized_buffer() {
        let tw = seeded();
        let mut dst = vec![0i8; ROWS * COLS + 1];
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tw.unpack_into(&mut dst);
        }));
        assert!(
            err.is_err(),
            "unpack_into must assert on a mis-sized buffer"
        );
    }

    /// The onward composition Plan 333 T3.3b needs: `[i8]` → row-scale tier →
    /// group-scale tier (the Bonsai `Q2_0_g128` container) via the already
    /// shipped `TernaryGroupWeights::from_ternary`.
    #[cfg(feature = "ternary_group_scale")]
    #[test]
    fn bridges_onward_to_the_group_scale_tier() {
        let tw = seeded();
        let packed = TernaryWeights::pack_from_i8(&tw.to_i8(), ROWS, COLS).unwrap();
        let group = crate::TernaryGroupWeights::from_ternary(&packed);
        assert!(group.invariant_holds(), "group tier invariant");
        for r in 0..ROWS {
            for c in 0..COLS {
                assert_eq!(group.get(r, c), tw.get(r, c), "sign lost at ({r}, {c})");
            }
        }
    }
}

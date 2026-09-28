//! BITCOS — distribution-adaptive ternary layout: presence bitmap + compacted
//! signs (Issue 864, Research 577 / arXiv:2609.16338).
//!
//! The fourth tier of the ternary container ladder, and the first whose rate
//! adapts to the weight **distribution** instead of being fixed:
//!
//! | Layout | bits/weight (incl. f16 scale @128) | Bonsai-27B |
//! |---|---|---|
//! | two bit-planes ([`crate::TernaryGroupWeights`], Issue 578) | 2.125 | 7.17 GB |
//! | 5 trits/byte ([`crate::TernaryTritWeights`], Issue 582) | 1.75 | 5.90 GB |
//! | **presence bitmap + compacted signs** (this tier) | **2−z+0.125** | 6.16 GB @ z=0.297 |
//!
//! where `z` is the **zero density** of the tensor — real ternary checkpoints
//! measure z = 0.297–0.515 across 29 SOTA models (the paper's founding
//! observation), so the rate spans 1.625–1.875 bits/weight depending on the
//! checkpoint.
//!
//! # Layout
//!
//! - **Presence bitmap**: dense, positional, 1 bit per weight (bit set =
//!   nonzero). Word `j`, bit `k` of row `r` ↔ column `j*64+k` — the same
//!   word geometry as the bit-plane tier, so `presence = pos | neg` is a
//!   per-word OR and the bitmap is block-loadable.
//! - **Compacted sign vector**: 1 bit per nonzero, in tensor order (row-major,
//!   columns ascending). The j-th set bit of the presence bitmap takes the j-th
//!   sign bit. **Sign bit 0 = +1, 1 = −1.** [`BitcosWeights::row_sign_offset`]
//!   holds each row's starting bit offset (the only per-row state the bitmap
//!   alone does not give away); a zero sentinel word is appended so kernels
//!   can read an unaligned 64-bit window without a bounds branch.
//!
//! # Honest scoping (inherited from Research 577 §2.4 — read before wiring)
//!
//! - **Regime-gated**: the layout wins only where GEMV is **bandwidth-bound**.
//!   Where it is instruction-bound (bandwidth-rich / few-core — the paper's
//!   Lunar Lake row, 0.47–0.70×) or **cache-resident** (Issue 582's G2b: the
//!   traffic saving has nothing to pay the decode with), it **loses**.
//!   [`should_use_bitcos`] is the roofline dispatch; the GOAT gate asserts both
//!   negative controls.
//! - **Tier-relative**: it beats the bit-plane tier at every z, but beats the
//!   trit tier only for **z > 0.375** — at Bonsai-27B (z = 0.297) it is a
//!   0.96× footprint **regression** vs our own best tier. The dispatch
//!   predicate therefore requires `z > 0.375` unconditionally: it must never
//!   select bitcos where the shipped trit tier is smaller.
//! - **Not** a league lever (tg128 is GPU-busy-bound, prefill compute-bound)
//!   and **not** for cache-resident consumers (the healer drafter class).
//!
//! Conversion from the bit-plane tier is bit-exact and nearly free — the
//! planes ARE an uncompressed BITCOS (presence = `pos|neg`, signs = compacted
//! `neg` bits: sign 0 = +1, 1 = −1).

use crate::{GROUP_SIZE, TernaryGroupWeights};
use half::f16;

/// Paper-metric scale overhead: 16 bits per [`GROUP_SIZE`] weights.
const SCALE_BITS_PER_WEIGHT: f64 = 16.0 / GROUP_SIZE as f64;

/// Trit-tier crossover. Below this zero density the shipped 5-trit container
/// (1.75 bits/w) is strictly smaller than BITCOS (2−z+0.125 > 1.75 ⟺ z < 0.375)
/// and the dispatch must never select this tier on footprint grounds.
pub const TRIT_CROSSOVER_Z: f64 = 0.375;

/// Extract `src`'s bits at `mask`'s set positions into the low bits (the
/// `pext` semantic — the pack-side mirror of [`scatter_bits`]).
///
/// The compacted sign stream stores the j-th nonzero's sign at bit j, so
/// packing extracts the neg plane onto the presence mask; decoding scatters
/// back. Portable scalar emulation (pack is offline; kernels probe for the
/// hardware instruction).
#[inline]
fn extract_bits(src: u64, mask: u64) -> u64 {
    let mut out = 0u64;
    let mut bit = 0u32;
    let mut m = mask;
    while m != 0 {
        let lsb = m.isolate_lowest_one();
        m ^= lsb;
        if src & lsb != 0 {
            out |= 1 << bit;
        }
        bit += 1;
    }
    out
}

/// Scatter `src`'s low bits onto `mask`'s set positions (the `pdep` semantic).
///
/// Portable scalar emulation used at **pack time only** — pack is offline, and
/// keeping it probe-free means `pack_from_group` behaves identically on every
/// target. The hot kernels use the hardware `pdep` behind a runtime probe
/// where available (see [`crate::simd::bitcos`]). `pub(crate)`: the scalar
/// GEMV reference decodes words the same way (its per-lane walk needs the
/// scattered negatives, not the compacted stream).
#[inline]
pub(crate) fn scatter_bits(src: u64, mask: u64) -> u64 {
    let mut out = 0u64;
    let mut s = src;
    let mut m = mask;
    while m != 0 {
        let lsb = m.isolate_lowest_one();
        m ^= lsb;
        if s & 1 != 0 {
            out |= lsb;
        }
        s >>= 1;
    }
    out
}

/// Append the low `n` bits of `bits` to `buf` at bit cursor `cursor`.
///
/// `buf` is grown as needed; callers append one final zero word as the
/// unaligned-read sentinel after the stream is complete.
fn push_bits(buf: &mut Vec<u64>, cursor: &mut usize, bits: u64, n: u32) {
    debug_assert!(n <= 64);
    let word = *cursor / 64;
    let off = *cursor % 64;
    while buf.len() <= word + 1 {
        buf.push(0);
    }
    buf[word] |= bits << off;
    // off == 0 with n == 64 fills the word exactly; only a straddle carries.
    if off > 0 && off + n as usize > 64 {
        buf[word + 1] |= bits >> (64 - off);
    }
    *cursor += n as usize;
}

/// Ternary `{-1, 0, +1}` weights as a presence bitmap + compacted signs, with
/// a per-[`GROUP_SIZE`] f16 scale (same alphabet, same scale granularity, and
/// bit-exact round-trip against [`crate::TernaryGroupWeights`]).
///
/// Built by packing (from planes, from the group container, or from f32 via
/// the shared error-compensated quantizer); there is no `set()` — the sign
/// stream's offsets are frozen at pack time, so in-place mutation would shift
/// every later row. Mutate via `to_group()` → edit → `pack_from_group()`.
#[cfg(feature = "bitcos")]
#[derive(Clone, Debug)]
pub struct BitcosWeights {
    pub rows: usize,
    pub cols: usize,
    pub words_per_row: usize,      // cols.div_ceil(64)
    pub groups_per_row: usize,     // cols.div_ceil(GROUP_SIZE)
    pub presence: Vec<u64>,        // [rows * words_per_row]
    pub signs: Vec<u64>,           // compacted stream + 1 sentinel zero word
    pub row_sign_offset: Vec<u64>, // bit offset of row r's first sign bit
    pub group_scale: Vec<f16>,     // [rows * groups_per_row]
}

/// Per-tensor zero-density measurement (Issue 864 T2 — the z-meter).
///
/// The cheap popcount report that gates every downstream decision: `z` decides
/// the footprint tier and (with the roofline) the layout dispatch. `z_row_min`
/// / `z_row_max` bound the per-row spread — a dispatch fed by the overall mean
/// can be wrong per row, and the spread says by how much.
#[cfg(feature = "bitcos")]
#[derive(Clone, Copy, Debug)]
pub struct ZeroDensityReport {
    pub rows: usize,
    pub cols: usize,
    /// Nonzero weights.
    pub nnz: u64,
    /// `rows * cols`.
    pub total: u64,
    /// Zero fraction overall — the paper's z.
    pub z_overall: f64,
    pub z_row_min: f64,
    pub z_row_max: f64,
}

/// z-meter over a bit-plane blob: popcount `pos | neg` per row (Issue 864 T2).
///
/// Works directly on [`TernaryGroupWeights`] so an existing checkpoint's z is
/// measurable **before** any repack decision.
#[cfg(feature = "bitcos")]
pub fn zero_density_report(w: &TernaryGroupWeights) -> ZeroDensityReport {
    let mut nnz: u64 = 0;
    let mut z_min = f64::MAX;
    let mut z_max = f64::MIN;
    for r in 0..w.rows {
        let base = r * w.blocks64;
        let mut row_nnz = 0u64;
        for b in 0..w.blocks64 {
            row_nnz += (w.pos_bits[base + b] | w.neg_bits[base + b]).count_ones() as u64;
        }
        let live = w.cols as f64;
        let z = 1.0 - row_nnz as f64 / live;
        z_min = z_min.min(z);
        z_max = z_max.max(z);
        nnz += row_nnz;
    }
    let total = (w.rows * w.cols) as u64;
    ZeroDensityReport {
        rows: w.rows,
        cols: w.cols,
        nnz,
        total,
        z_overall: 1.0 - nnz as f64 / total as f64,
        z_row_min: z_min,
        z_row_max: z_max,
    }
}

/// Payload bytes per weight at zero density `z`: `(2 − z) / 8`.
///
/// The paper's `B(z)` per-32-weight block is `8.5 − 4z` bytes — the same
/// quantity per weight. Excludes scale (constant across tiers).
#[cfg(feature = "bitcos")]
#[inline]
pub fn bitcos_payload_bytes_per_weight(z: f64) -> f64 {
    (2.0 - z) / 8.0
}

/// Paper-metric rate incl. the f16 group scale: `2 − z + 0.125` bits/weight.
#[cfg(feature = "bitcos")]
#[inline]
pub fn bitcos_bits_per_weight(z: f64) -> f64 {
    (2.0 - z) + SCALE_BITS_PER_WEIGHT
}

/// Roofline dispatch (Issue 864 T4).
///
/// `ebw = min(β, B(z)/γ)` — effective payload bandwidth is the smaller of the
/// per-core share `beta_bytes_per_cycle` and the decode-limited rate
/// `B(z)/gamma_cycles_per_byte` (γ measured on an L1-resident loop of the
/// consuming kernel: cycles per payload byte when memory is free). BITCOS is
/// selected iff **both** hold:
///
/// 1. `B(z)/γ > β` — the kernel is **bandwidth-bound**, so the smaller payload
///    converts to time savings. When the decode is the limiter the time per
///    weight is γ regardless of payload size, and the layout's only remaining
///    virtue is footprint — which the trit tier beats below the crossover.
/// 2. `z > 0.375` ([`TRIT_CROSSOVER_Z`]) — never select bitcos where the
///    shipped trit tier is smaller.
///
/// The honest full-ladder choice compares `max(B_f/β, γ_f)` per format; this
/// predicate is the bitcos arm of that comparison as specced by Issue 864 —
/// the losing-regime refusals it encodes are exactly the paper's Lunar Lake
/// row and Issue 582's G2b cache-resident note, and both are asserted as
/// negative controls in the GOAT gate.
#[cfg(feature = "bitcos")]
pub fn should_use_bitcos(z: f64, gamma_cycles_per_byte: f64, beta_bytes_per_cycle: f64) -> bool {
    let b = bitcos_payload_bytes_per_weight(z);
    b / gamma_cycles_per_byte > beta_bytes_per_cycle && z > TRIT_CROSSOVER_Z
}

#[cfg(feature = "bitcos")]
impl BitcosWeights {
    /// Pack from raw pos/neg planes (`rows * cols.div_ceil(64)` words each,
    /// bit `k` of word `j` ↔ column `j*64+k` — the [`TernaryGroupWeights`]
    /// geometry).
    ///
    /// `presence = pos | neg` per word; the sign stream carries the `pos` bits
    /// compacted at presence-set positions (sign 0 = +1, 1 = −1). Pad bits
    /// past `cols` in the final word of a ragged row are masked off, so the
    /// packed form never consumes signs for them.
    ///
    /// # Panics
    /// If `pos` and `neg` overlap (the bit-plane invariant `pos & neg == 0`
    /// makes both-set unreachable) or the slice lengths disagree with the dims.
    pub fn pack_from_planes(
        pos: &[u64],
        neg: &[u64],
        rows: usize,
        cols: usize,
        group_scale: &[f16],
    ) -> Self {
        let words_per_row = cols.div_ceil(64);
        let groups_per_row = cols.div_ceil(GROUP_SIZE);
        assert_eq!(pos.len(), rows * words_per_row, "pos plane length mismatch");
        assert_eq!(neg.len(), rows * words_per_row, "neg plane length mismatch");
        assert_eq!(
            group_scale.len(),
            rows * groups_per_row,
            "group scale length mismatch"
        );

        let tail_bits = cols % 64;
        let mut presence = vec![0u64; rows * words_per_row];
        let mut signs: Vec<u64> = Vec::new();
        let mut row_sign_offset = Vec::with_capacity(rows);
        let mut cursor = 0usize;

        for r in 0..rows {
            row_sign_offset.push(cursor as u64);
            for j in 0..words_per_row {
                let i = r * words_per_row + j;
                let mask = if j + 1 == words_per_row && tail_bits != 0 {
                    (1u64 << tail_bits) - 1
                } else {
                    u64::MAX
                };
                let p_word = pos[i] & mask;
                let n_word = neg[i] & mask;
                assert!(
                    p_word & n_word == 0,
                    "pos/neg overlap at word {i} — corrupt planes"
                );
                let p = p_word | n_word;
                presence[i] = p;
                // Sign of the j-th set position: sign 1 = -1, so the stream
                // carries the NEG bits EXTRACTED from p's set positions into
                // the low bits (the pext semantic — `scatter_bits`'s mirror).
                // NOTE: Research 577 §2.2 words this as "compacted pos bits"
                // against its own 0=+1/1=-1 convention — the self-consistent
                // form is neg-compaction, pinned here.
                let s = extract_bits(n_word, p);
                let n = p.count_ones();
                if n > 0 {
                    push_bits(&mut signs, &mut cursor, s, n);
                }
            }
        }
        // Sentinel: kernels read an unaligned 64-bit window at any live cursor.
        signs.push(0);

        Self {
            rows,
            cols,
            words_per_row,
            groups_per_row,
            presence,
            signs,
            row_sign_offset,
            group_scale: group_scale.to_vec(),
        }
    }

    /// Pack from the bit-plane tier — bit-exact in weights and scale
    /// (the Issue 582 `from_group` repack precedent, third instance).
    #[cfg(feature = "ternary_group_scale")]
    pub fn pack_from_group(gw: &TernaryGroupWeights) -> Self {
        Self::pack_from_planes(
            &gw.pos_bits,
            &gw.neg_bits,
            gw.rows,
            gw.cols,
            &gw.group_scale,
        )
    }

    /// Pack from f32 with the shared error-compensated group quantizer.
    ///
    /// Delegates to [`TernaryGroupWeights::quantize_from_f32`] then
    /// [`Self::pack_from_group`]: the quantize arithmetic is deliberately the
    /// bit-plane tier's own (same mean-abs scale, same `0.5·scale` threshold,
    /// same carry), so every tier quantizes the same input to the same
    /// weights and only the storage differs. Pack is offline; the transient
    /// intermediate is the price of that single-source-of-truth.
    #[cfg(feature = "ternary_group_scale")]
    pub fn pack_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self {
        let gw = TernaryGroupWeights::quantize_from_f32(weights, rows, cols);
        Self::pack_from_group(&gw)
    }

    /// Unaligned 64-bit read at bit offset `off` (sentinel word covers the
    /// straddle; `off` must address a live or sentinel bit).
    #[inline]
    fn sign_window(&self, off: u64) -> u64 {
        let w = (off / 64) as usize;
        let o = (off % 64) as u32;
        let lo = self.signs[w] >> o;
        if o == 0 {
            lo
        } else {
            lo | (self.signs[w + 1] << (64 - o))
        }
    }

    /// The ternary value at `(row, col)`: `0` where presence is clear, else
    /// `+1`/`−1` by the row's compacted sign bit.
    ///
    /// Sign index = row offset + popcount of preceding presence bits — the
    /// paper's per-column cursor. Random access costs a masked popcount over
    /// the row's earlier words; kernels avoid it with a running cursor.
    pub fn get(&self, row: usize, col: usize) -> i8 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let w = col >> 6;
        let bit = 1u64 << (col & 63);
        let pw = self.presence[row * self.words_per_row + w];
        if pw & bit == 0 {
            return 0;
        }
        let mut prior: u64 = self.row_sign_offset[row];
        let base = row * self.words_per_row;
        for j in 0..w {
            prior += self.presence[base + j].count_ones() as u64;
        }
        prior += (pw & (bit - 1)).count_ones() as u64;
        if (self.sign_window(prior) & 1) == 1 {
            -1
        } else {
            1
        }
    }

    /// Decode row `row` into `out` (`out.len() == cols`). Zero allocations.
    ///
    /// Absent positions are written as `0` — zeros are implicit in the
    /// layout, so a fully dense word skips the zeroing pass entirely.
    pub fn unpack_row_into(&self, row: usize, out: &mut [i8]) {
        assert_eq!(out.len(), self.cols, "out length must equal cols");
        assert!(row < self.rows, "row out of bounds");
        let base = row * self.words_per_row;
        let mut cursor = self.row_sign_offset[row];
        for j in 0..self.words_per_row {
            let p = self.presence[base + j];
            let c0 = j * 64;
            let c_end = (c0 + 64).min(self.cols);
            let live = p.count_ones() as usize;
            if live == 0 {
                out[c0..c_end].fill(0);
                continue;
            }
            let window = self.sign_window(cursor);
            let neg = scatter_bits(window, p);
            let mut m = p;
            while m != 0 {
                let lsb = m.isolate_lowest_one();
                let c = c0 + lsb.trailing_zeros() as usize;
                out[c] = if (neg & lsb) != 0 { -1 } else { 1 };
                m ^= lsb;
            }
            if live != 64 {
                // Implicit zeros: clear the absent bits of this word's span.
                let absent = !p;
                let mut z = absent;
                while z != 0 {
                    let lsb = z.isolate_lowest_one();
                    let c = c0 + lsb.trailing_zeros() as usize;
                    if c < c_end {
                        out[c] = 0;
                    }
                    z ^= lsb;
                }
            }
            cursor += live as u64;
        }
    }

    /// Repack into the bit-plane tier. Bit-exact — the inverse of
    /// [`Self::pack_from_group`], so `pack_from_group(w).to_group()` is
    /// identical to `w` in every field (G1).
    #[cfg(feature = "ternary_group_scale")]
    pub fn to_group(&self) -> TernaryGroupWeights {
        let mut out = TernaryGroupWeights::new(self.rows, self.cols);
        for r in 0..self.rows {
            let base = r * self.words_per_row;
            let mut cursor = self.row_sign_offset[r];
            for j in 0..self.words_per_row {
                let p = self.presence[base + j];
                if p != 0 {
                    let window = self.sign_window(cursor);
                    let neg = scatter_bits(window, p);
                    out.neg_bits[base + j] = neg;
                    out.pos_bits[base + j] = p & !neg;
                    cursor += p.count_ones() as u64;
                }
            }
        }
        out.group_scale.copy_from_slice(&self.group_scale);
        out
    }

    /// Representation invariant (the corruption signal for a mis-parsed load):
    /// pad bits past `cols` are clear, sign bits past the stream end are zero,
    /// and the scale slice is fully populated.
    pub fn is_canonical(&self) -> bool {
        let tail_bits = self.cols % 64;
        for r in 0..self.rows {
            let last = r * self.words_per_row + self.words_per_row - 1;
            if tail_bits != 0 && (self.presence[last] >> tail_bits) != 0 {
                return false;
            }
        }
        // Total live sign bits = last row's offset + its nnz.
        let mut used = 0u64;
        for r in 0..self.rows {
            let base = r * self.words_per_row;
            let mut nnz = 0u32;
            for j in 0..self.words_per_row {
                nnz += self.presence[base + j].count_ones();
            }
            if r + 1 == self.rows {
                used = self.row_sign_offset[r] + nnz as u64;
            }
        }
        let total_bits = (self.signs.len() * 64) as u64;
        if used > total_bits {
            return false;
        }
        // Every bit at or past `used` must be zero.
        let w = (used / 64) as usize;
        let o = (used % 64) as u32;
        if (self.signs.get(w).copied().unwrap_or(0)) >> o != 0 {
            return false;
        }
        for word in self.signs.iter().skip(w + 1) {
            if *word != 0 {
                return false;
            }
        }
        self.group_scale.len() == self.rows * self.groups_per_row
    }

    /// Zero density `z` of the packed tensor (popcount over presence).
    pub fn zero_density(&self) -> f64 {
        let nnz: u64 = self.presence.iter().map(|p| p.count_ones() as u64).sum();
        1.0 - nnz as f64 / (self.rows * self.cols) as f64
    }

    /// Nonzero count.
    pub fn nnz(&self) -> u64 {
        self.presence.iter().map(|p| p.count_ones() as u64).sum()
    }

    /// Checksum over all values — same definition as the other two tiers
    /// (`Σ_r Σ_g scale[r,g] · Σ_{col∈g} weight`), so cross-tier verification
    /// compares like with like.
    pub fn checksum(&self) -> f32 {
        let mut total = 0.0f32;
        let mut row_vals = vec![0i8; self.cols.max(1)];
        for r in 0..self.rows {
            self.unpack_row_into(r, &mut row_vals);
            let group_base = r * self.groups_per_row;
            for g in 0..self.groups_per_row {
                let w_start = g * GROUP_SIZE;
                let w_end = (w_start + GROUP_SIZE).min(self.cols);
                let mut sum: i32 = 0;
                for &v in &row_vals[w_start..w_end] {
                    sum += v as i32;
                }
                total += self.group_scale[group_base + g].to_f32() * sum as f32;
            }
        }
        total
    }

    /// Bytes of weight payload actually stored (presence + signs + offsets +
    /// scales), excluding `Vec` overhead.
    ///
    /// Note the offsets row (`8 B/row`) — the honest true cost the paper rate
    /// [`bitcos_bits_per_weight`] abstracts away. At real shapes it is noise
    /// (4096²: 32 KB on a ~3.5 MB payload).
    pub fn encoded_bytes(&self) -> usize {
        self.presence.len() * 8
            + self.signs.len() * 8
            + self.row_sign_offset.len() * 8
            + self.group_scale.len() * 2
    }

    /// Scale applied to group `g` of row `r`.
    #[inline]
    pub fn scale_at(&self, row: usize, group: usize) -> f32 {
        self.group_scale[row * self.groups_per_row + group].to_f32()
    }
}

#[cfg(all(test, feature = "bitcos"))]
mod tests {
    use super::*;

    fn group_from_pairs(pairs: &[(usize, i8)], rows: usize, cols: usize) -> TernaryGroupWeights {
        let mut w = TernaryGroupWeights::new(rows, cols);
        for &(i, v) in pairs {
            w.set(i / cols, i % cols, v);
        }
        w
    }

    #[test]
    fn exhaustive_small_roundtrip_every_bitmap_sign_pair() {
        // cols = 4 (single nibble), rows = 1: enumerate all 3^4 = 81 ternary
        // rows via the group container and assert planes+scale survive pack.
        for seed in 0..81u32 {
            let mut v = seed;
            let mut gw = TernaryGroupWeights::new(1, 4);
            for c in 0..4 {
                let t = v % 3;
                v /= 3;
                gw.set(0, c, t as i8 - 1);
            }
            gw.set_scale(0, 0, 0.5 + seed as f32 * 0.01);
            let bc = BitcosWeights::pack_from_group(&gw);
            let rt = bc.to_group();
            assert_eq!(rt.pos_bits, gw.pos_bits, "pos planes differ (seed {seed})");
            assert_eq!(rt.neg_bits, gw.neg_bits, "neg planes differ (seed {seed})");
            assert_eq!(rt.group_scale, gw.group_scale);
            assert!(bc.is_canonical());
            for c in 0..4 {
                assert_eq!(bc.get(0, c), gw.get(0, c), "get({c}) seed {seed}");
            }
        }
    }

    #[test]
    fn roundtrip_seeded_random_planes_multi_word_multi_row() {
        let mut s = 0x853c49e6748fea9bu64;
        let mut pseudo = move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            s >> 33
        };
        for &(rows, cols) in &[(3usize, 200usize), (5, 128), (2, 63), (7, 129), (1, 256)] {
            let mut gw = TernaryGroupWeights::new(rows, cols);
            for r in 0..rows {
                for c in 0..cols {
                    let v = pseudo() % 3;
                    gw.set(r, c, v as i8 - 1);
                }
                for g in 0..gw.groups_per_row {
                    gw.set_scale(r, g, 1.0 + (pseudo() % 7) as f32 * 0.25);
                }
            }
            let bc = BitcosWeights::pack_from_group(&gw);
            assert!(bc.is_canonical(), "canonical ({rows}x{cols})");
            let rt = bc.to_group();
            assert_eq!(rt.pos_bits, gw.pos_bits, "pos ({rows}x{cols})");
            assert_eq!(rt.neg_bits, gw.neg_bits, "neg ({rows}x{cols})");
            assert_eq!(rt.group_scale, gw.group_scale, "scale ({rows}x{cols})");
        }
    }

    #[test]
    fn unpack_row_matches_get_and_zero_pads_are_implicit() {
        let gw = group_from_pairs(
            &[(0, 1), (3, -1), (5, 1), (70, -1), (71, -1), (129, 1)],
            2,
            130,
        );
        let bc = BitcosWeights::pack_from_group(&gw);
        for r in 0..2 {
            let mut out = vec![7i8; 130];
            bc.unpack_row_into(r, &mut out);
            for (c, &v) in out.iter().enumerate() {
                assert_eq!(v, bc.get(r, c), "row {r} col {c}");
            }
        }
    }

    #[test]
    fn all_zero_tensor_packs_to_empty_sign_stream() {
        let gw = TernaryGroupWeights::new(4, 300);
        let bc = BitcosWeights::pack_from_group(&gw);
        assert_eq!(bc.nnz(), 0);
        assert!(bc.signs.iter().all(|&w| w == 0));
        assert!(bc.is_canonical());
        assert!((bc.zero_density() - 1.0).abs() < 1e-12);
        let rt = bc.to_group();
        assert!(rt.pos_bits.iter().all(|&w| w == 0));
        assert!(rt.neg_bits.iter().all(|&w| w == 0));
    }

    #[test]
    fn z_meter_matches_known_density() {
        // 3x64 = 192 weights, 12 nonzero → z = 180/192 = 0.9375.
        let mut gw = TernaryGroupWeights::new(3, 64);
        for i in 0..12 {
            gw.set(i / 64, i % 64, 1);
        }
        let rep = zero_density_report(&gw);
        assert_eq!(rep.nnz, 12);
        assert!((rep.z_overall - 0.9375).abs() < 1e-12);
        assert!((rep.z_row_min - (1.0 - 12.0 / 64.0)).abs() < 1e-12); // row 0 densest
        assert!((rep.z_row_max - 1.0).abs() < 1e-12); // rows 1-2 all zero
    }

    #[test]
    fn paper_rate_and_crossover_arithmetic() {
        // z = 0.515 → 1.485 + 0.125 = 1.610 bits/w incl. scale; trit = 1.75.
        assert!((bitcos_bits_per_weight(0.515) - 1.61).abs() < 1e-9);
        // B(z)/w at z = 0.5: 1.5 bits / 8 = 0.1875 bytes.
        assert!((bitcos_payload_bytes_per_weight(0.5) - 0.1875).abs() < 1e-12);
        // At the crossover the rates are equal (2 − 0.375 + 0.125 = 1.75).
        assert!((bitcos_bits_per_weight(TRIT_CROSSOVER_Z) - 1.75).abs() < 1e-12);
    }

    #[test]
    fn dispatch_negative_controls() {
        // Bandwidth-bound, high z → selected.
        // B(0.5) = 0.1875 B/w; γ = 0.05 cyc/B → 3.75 B/cyc > β = 1.0.
        assert!(should_use_bitcos(0.5, 0.05, 1.0));
        // Instruction-bound (the paper's Lunar Lake shape): decode-limited
        // 0.1875/0.5 = 0.375 B/cyc < β = 1.0 → refuse, even at CAT-Q z.
        assert!(!should_use_bitcos(0.515, 0.5, 1.0));
        // Bandwidth-bound but below the trit crossover (Bonsai-27B z):
        // trit is smaller — refuse.
        assert!(!should_use_bitcos(0.297, 0.05, 1.0));
        // Boundary: exactly at the crossover is NOT selected (strict >).
        assert!(!should_use_bitcos(0.375, 0.01, 0.01));
    }

    #[test]
    fn footprint_tracks_z_between_the_shipped_tiers() {
        // Dense-ish random planes at controlled z by thresholding.
        let mut s = 0x243f6a8885a308d3u64;
        let mut pseudo = move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            s >> 33
        };
        let rows = 64usize;
        let cols = 4096usize;
        let mut gw = TernaryGroupWeights::new(rows, cols);
        // Keep every 3rd weight → z ≈ 2/3.
        for r in 0..rows {
            for c in 0..cols {
                if pseudo() % 3 == 0 {
                    let v = pseudo() % 2;
                    gw.set(r, c, if v == 0 { 1 } else { -1 });
                }
            }
        }
        let bc = BitcosWeights::pack_from_group(&gw);
        let z = bc.zero_density();
        assert!(z > 0.60 && z < 0.70, "z={z}");
        // bit-plane bytes (planes + scales), same shape.
        let plane_bytes = gw.pos_bits.len() * 8 + gw.neg_bits.len() * 8 + gw.group_scale.len() * 2;
        assert!(
            bc.encoded_bytes() < plane_bytes,
            "bitcos {} vs planes {} at z={z}",
            bc.encoded_bytes(),
            plane_bytes
        );
        // vs trit at this z (> 0.375): trit bytes = ceil(cols/5) * rows + scales.
        let trit_bytes = rows * cols.div_ceil(5) + gw.group_scale.len() * 2;
        assert!(
            bc.encoded_bytes() < trit_bytes,
            "bitcos {} vs trit {} at z={z}",
            bc.encoded_bytes(),
            trit_bytes
        );
    }

    #[test]
    fn checksum_agrees_with_the_bit_plane_tier() {
        let mut gw = TernaryGroupWeights::new(3, 200);
        let mut s = 0x13198a2e03707344u64;
        for r in 0..3 {
            for c in 0..200 {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                gw.set(r, c, ((s >> 33) % 3) as i8 - 1);
            }
            gw.set_scale(r, 0, 1.25);
            gw.set_scale(r, 1, 0.75);
        }
        let bc = BitcosWeights::pack_from_group(&gw);
        assert!((bc.checksum() - gw.checksum()).abs() < 1e-4);
    }
}

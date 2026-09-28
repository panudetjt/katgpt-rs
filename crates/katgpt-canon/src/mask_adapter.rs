//! [`MaskAdapter`] — lottery-ticket mask application (modelless).
//!
//! Applies a precomputed lottery-ticket mask to a canonical direction
//! before projection. The mask is a bit-packed vector of length `target_dim`;
//! bit `i` selects whether the `i`-th latent coordinate participates.
//!
//! # What's modelless vs training-only
//!
//! - **Mask application at inference** — elementwise multiply `m ⊙ W`.
//!   Modelless. Shipped here.
//! - **Mask discovery (Iterative Magnitude Pruning, IMP)** — iterative
//!   training. **NOT modelless** — lives in riir-train (Research 459 §1.3).
//!
//! # Mask transfer across canonical-aligned models
//!
//! If model B is Procrustes-aligned to model A (via [`crate::ProcrustesAdapter`]),
//! A's winning-ticket mask transfers (under appropriate basis change).
//! This is the novel modelless residue: the mask is discovered once on A,
//! then transferred to B without re-discovery.
//!
//! # Bit packing
//!
//! The mask is bit-packed: 32 latents per `u32`. Bit `i` of `mask[i / 32]`
//! corresponds to latent `i`. A set bit = "participates" (1.0); a clear bit
//! = "masked out" (0.0). This matches the wire format used by
//! `riir-neuron-db/src/spectral_flatness.rs` (lottery-ticket init) and
//! keeps the adapter size at `target_dim / 32` u32s (compact for sync/commit).

use crate::{CanonicalIntent, ModelAdapter};

/// Lottery-ticket mask adapter. Applies a bit-packed mask to the canonical
/// direction's projection.
///
/// NOTE: the mask applies to the OUTPUT dim (target_dim), not the canonical
/// dim. This means a `MaskAdapter` wraps another adapter (composition):
/// the inner adapter projects canonical → model latent, then the mask
/// zeroes out the masked coordinates.
///
/// For now, P0 ships a standalone mask that operates on canonical ==
/// target (same-dim identity case). The composition-with-Procrustes case
/// (ProcrustesAdapter + MaskAdapter) is a P4 follow-up — the trait
/// composition is straightforward but the API needs a `composed` helper.
#[derive(Clone, Debug)]
pub struct MaskAdapter {
    mask: Vec<u32>,
    target_dim: usize,
    commitment: [u8; 32],
}

impl MaskAdapter {
    /// Construct from a bit-packed mask. `mask.len()` MUST be
    /// `target_dim.div_ceil(32)`. Bit `i` of `mask[i / 32]` selects latent `i`.
    #[inline]
    pub fn new(mask: Vec<u32>, target_dim: usize) -> Self {
        let expected_words = target_dim.div_ceil(32);
        assert_eq!(
            mask.len(),
            expected_words,
            "MaskAdapter: mask.len() ({}) != ceil(target_dim / 32) ({})",
            mask.len(),
            expected_words
        );
        let commitment = commit_mask(&mask, target_dim);
        Self {
            mask,
            target_dim,
            commitment,
        }
    }

    /// Construct an "all-ones" mask (no masking — identity behavior).
    /// Useful as the default / control in comparisons.
    pub fn all_ones(target_dim: usize) -> Self {
        let words = target_dim.div_ceil(32);
        let mask = vec![u32::MAX; words];
        // The last word may have bits beyond target_dim set — that's fine,
        // project_into clamps to target_dim.
        Self::new(mask, target_dim)
    }

    /// Construct from a boolean slice (one bool per latent). Convenience
    /// constructor — packs into u32 words.
    pub fn from_bools(bits: &[bool]) -> Self {
        let target_dim = bits.len();
        let words = target_dim.div_ceil(32);
        let mut mask = vec![0u32; words];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                mask[i / 32] |= 1u32 << (i % 32);
            }
        }
        Self::new(mask, target_dim)
    }

    /// Read-only access to the bit-packed mask.
    pub fn mask(&self) -> &[u32] {
        &self.mask
    }

    /// Returns true if latent `i` participates (bit set).
    #[inline]
    pub fn participates(&self, i: usize) -> bool {
        debug_assert!(
            i < self.target_dim,
            "index {i} >= target_dim {}",
            self.target_dim
        );
        if i >= self.target_dim {
            return false;
        }
        let word = self.mask[i / 32];
        let bit = 1u32 << (i % 32);
        (word & bit) != 0
    }
}

impl ModelAdapter for MaskAdapter {
    /// Apply the mask to `canonical` directly (canonical must already be
    /// length `target_dim` — MaskAdapter is a same-dim adapter for now).
    /// `out[i] = canonical[i] if participates(i) else 0.0`.
    #[inline]
    fn project_into(&self, canonical: &CanonicalIntent, out: &mut [f32]) {
        let d = self.target_dim;
        debug_assert_eq!(
            out.len(),
            d,
            "MaskAdapter::project_into: out.len() != target_dim"
        );
        debug_assert_eq!(
            canonical.dim(),
            d,
            "MaskAdapter::project_into: canonical.dim() != target_dim (same-dim adapter)"
        );
        if out.len() != d {
            return;
        }
        let src = canonical.as_slice();
        let n = src.len().min(d);
        // Process 32 latents at a time (one u32 word).
        let mut i = 0;
        while i + 32 <= n {
            let word = self.mask[i / 32];
            for bit_idx in 0..32 {
                let participates = (word >> bit_idx) & 1 == 1;
                out[i + bit_idx] = if participates { src[i + bit_idx] } else { 0.0 };
            }
            i += 32;
        }
        // Tail: remaining < 32 latents.
        if i < n {
            let word = self.mask[i / 32];
            for bit_idx in 0..(n - i) {
                let participates = (word >> bit_idx) & 1 == 1;
                out[i + bit_idx] = if participates { src[i + bit_idx] } else { 0.0 };
            }
        }
    }

    /// Inverse: the mask is its own inverse on the supported set
    /// (coordinates that participate survive; masked-out coordinates are
    /// unrecoverable). Returns a Vec of length `canonical_dim`.
    fn extract_from(&self, model_latent: &[f32]) -> Vec<f32> {
        let d = self.target_dim;
        if model_latent.len() != d {
            return vec![0.0f32; d];
        }
        let mut out = vec![0.0f32; d];
        // Mask is idempotent on the supported set: extract == project.
        let canonical = CanonicalIntent::from_parts(self.commitment, model_latent.to_vec());
        // Note: from_parts doesn't normalize, so canonical is model_latent as-is.
        self.project_into(&canonical, &mut out);
        out
    }

    #[inline]
    fn target_dim(&self) -> usize {
        self.target_dim
    }

    #[inline]
    fn canonical_dim(&self) -> usize {
        // MaskAdapter is same-dim — canonical == target.
        self.target_dim
    }

    #[inline]
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[inline]
fn commit_mask(mask: &[u32], target_dim: usize) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    // Cast u32 slice to bytes (little-endian on most platforms; BLAKE3
    // is byte-order agnostic, but commitment is reproducible only on the
    // same endianness. For cross-arch sync, both sides must use the same
    // byte order — we use native byte order via cast_slice, matching
    // how katgpt-core's cognitive_architecture_root commits u64s.)
    let mask_bytes: &[u8] = bytemuck::cast_slice(mask);
    h.update(mask_bytes);
    h.update(&target_dim.to_le_bytes());
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ones_preserves_direction() {
        let m = MaskAdapter::all_ones(64);
        let d = CanonicalIntent::new("test", vec![1.0f32; 64]);
        let mut out = vec![0.0f32; 64];
        m.project_into(&d, &mut out);
        // All bits set → all coordinates survive.
        for (got, want) in out.iter().zip(d.as_slice().iter()) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn from_bools_packs_correctly() {
        // 64-bit mask with every other bit set.
        let bits: Vec<bool> = (0..64).map(|i| i % 2 == 0).collect();
        let m = MaskAdapter::from_bools(&bits);
        // Even indices participate.
        for i in 0..64 {
            assert_eq!(m.participates(i), i % 2 == 0);
        }
    }

    #[test]
    fn mask_zeroes_unselected_coordinates() {
        let mut bits = vec![true; 8];
        bits[3] = false;
        bits[7] = false;
        let m = MaskAdapter::from_bools(&bits);
        // Use from_parts to skip normalization — we want the literal values
        // preserved through the mask, not the unit-norm version.
        let tag = [0u8; 32];
        let d = CanonicalIntent::from_parts(tag, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut out = vec![0.0f32; 8];
        m.project_into(&d, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 7.0, 0.0]);
    }

    #[test]
    fn commitment_differs_for_different_masks() {
        let m1 = MaskAdapter::all_ones(64);
        let bits = vec![false; 64];
        let m2 = MaskAdapter::from_bools(&bits);
        assert_ne!(m1.commitment(), m2.commitment());
    }

    #[test]
    fn commitment_deterministic_across_constructors() {
        let bits: Vec<bool> = (0..64).map(|i| i % 3 == 0).collect();
        let m1 = MaskAdapter::from_bools(&bits);
        let m2 = MaskAdapter::from_bools(&bits);
        assert_eq!(m1.commitment(), m2.commitment());
    }

    #[test]
    fn handles_non_multiple_of_32() {
        // target_dim = 50, not a multiple of 32.
        let bits: Vec<bool> = (0..50).map(|i| i % 2 == 0).collect();
        let m = MaskAdapter::from_bools(&bits);
        // from_parts skips normalization so we can assert exact 1.0 values.
        let tag = [0u8; 32];
        let d = CanonicalIntent::from_parts(tag, vec![1.0f32; 50]);
        let mut out = vec![0.0f32; 50];
        m.project_into(&d, &mut out);
        for (i, out_i) in out.iter().enumerate().take(50) {
            let expected = if i % 2 == 0 { 1.0 } else { 0.0 };
            assert!(
                (out_i - expected).abs() < 1e-6,
                "i={i}: expected {expected}, got {out_i}"
            );
        }
    }
}

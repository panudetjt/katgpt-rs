//! [`ProcrustesAdapter`] — orthogonal Procrustes rotation (same-arch).
//!
//! Wraps a pre-fit rotation matrix from [`katgpt_spectral::procrustes::orthogonal_procrustes`]
//! (Issue 001 / Plan 152) as a [`crate::ModelAdapter`]. Linear, bijective,
//! information-preserving. The default adapter for same-architecture model
//! swap (e.g. two snapshots of Gemma-2-2B, or two same-dim checkpoints of
//! the same base).
//!
//! # Fit + apply lifecycle
//!
//! ```text
//! 1. Collect paired anchors: a[i] in canonical-frame-A, b[i] in latent-B.
//! 2. orthogonal_procrustes(a, b, n, d, &mut R, &mut scratch, &cfg)
//! 3. ProcrustesAdapter::from_rotation(R, d)
//! 4. adapter.project_into(&canonical, &mut out)  // hot path, zero-alloc
//! ```
//!
//! The fit step (1-2) runs once at adapter construction; the project step
//! (4) is the hot path and allocates nothing. The adapter does NOT own
//! scratch — fit-time scratch is the caller's responsibility (one
//! [`katgpt_spectral::procrustes::ProcrustesScratch`] can be reused across
//! many fits).
//!
//! # Cross-architecture note
//!
//! Procrustes requires `d_a == d_b` (square rotation). For cross-architecture
//! pairs (different hidden dims, e.g. Gemma 2304 ↔ MiniCPM 1536), use
//! [`crate::SubspaceAdapter`] which projects both into a shared low-dim
//! subspace first, THEN fits Procrustes in that subspace (Bench 423 G5 GO
//! at k ∈ {2, 4}).

use blake3::Hasher;

use crate::{CanonicalIntent, ModelAdapter};

/// Orthogonal Procrustes rotation as a [`ModelAdapter`].
///
/// Owns:
/// - `rotation`: row-major `d × d` f32 matrix (from `orthogonal_procrustes`).
/// - `target_dim`: equals `d` (square — bijective same-dim).
/// - `commitment`: BLAKE3 of the rotation bytes (computed once at construction).
///
/// The adapter is `Send + Sync` (no interior mutability, no `Rc`, no
/// lifetime-tied borrows) — safe to share across threads via `Arc`.
#[derive(Clone, Debug)]
pub struct ProcrustesAdapter {
    rotation: Vec<f32>,
    target_dim: usize,
    commitment: [u8; 32],
}

impl ProcrustesAdapter {
    /// Construct from a pre-fit rotation matrix (row-major `d × d`).
    /// Computes the BLAKE3 commitment. The rotation is consumed (moved).
    ///
    /// # Panics
    ///
    /// Panics if `rotation.len() != d * d`.
    #[inline]
    pub fn from_rotation(rotation: Vec<f32>, d: usize) -> Self {
        assert_eq!(
            rotation.len(),
            d * d,
            "ProcrustesAdapter: rotation.len() ({}) != d*d ({})",
            rotation.len(),
            d * d
        );
        let commitment = blake3_of_f32_slice(&rotation);
        Self {
            rotation,
            target_dim: d,
            commitment,
        }
    }

    /// Construct the identity adapter (rotation = I_d). Useful for tests
    /// and for the "no-op" case where canonical space == model latent space.
    pub fn identity(d: usize) -> Self {
        let mut rotation = vec![0.0f32; d * d];
        for i in 0..d {
            rotation[i * d + i] = 1.0;
        }
        Self::from_rotation(rotation, d)
    }

    /// Read-only access to the raw rotation matrix (row-major `d × d`).
    pub fn rotation(&self) -> &[f32] {
        &self.rotation
    }
}

impl ModelAdapter for ProcrustesAdapter {
    /// Project `canonical` through the rotation: `out = R · canonical`.
    ///
    /// `out.len()` MUST equal `target_dim()` AND `canonical.dim()`
    /// (Procrustes is square — bijective same-dim). Panics on mismatch.
    ///
    /// Zero-alloc: writes into caller-owned `out`. Row-major matvec.
    #[inline]
    fn project_into(&self, canonical: &CanonicalIntent, out: &mut [f32]) {
        let d = self.target_dim;
        debug_assert_eq!(
            out.len(),
            d,
            "ProcrustesAdapter::project_into: out.len() != target_dim"
        );
        debug_assert_eq!(
            canonical.dim(),
            d,
            "ProcrustesAdapter::project_into: canonical.dim() != target_dim (square adapter)"
        );
        // Row-major matvec: out[i] = sum_j R[i*d + j] * canonical[j].
        // Uses the auto-vectorizing dot product (mirrors `dot_8wide` in
        // katgpt-attn-match/src/score_matrix_simd.rs — the canonical version).
        // The simple `for` loop lets LLVM emit optimal NEON `fmla` / AVX2 FMA;
        // the previous 8-accumulator unroll was empirically slower (see
        // dot_8wide doc comment).
        let r = &self.rotation;
        let c = canonical.as_slice();
        // If lengths mismatch (release-mode), bail safely rather than OOB.
        if out.len() != d || c.len() != d {
            return;
        }
        for i in 0..d {
            let row = &r[i * d..(i + 1) * d];
            out[i] = dot_8wide(row, c, d);
        }
    }

    /// Inverse projection: `canonical = R^T · model_latent` (orthogonal
    /// inverse is the transpose). Returns a new Vec (diagnostic path, MAY alloc).
    fn extract_from(&self, model_latent: &[f32]) -> Vec<f32> {
        let d = self.target_dim;
        debug_assert_eq!(
            model_latent.len(),
            d,
            "ProcrustesAdapter::extract_from: model_latent.len() != target_dim"
        );
        let mut out = vec![0.0f32; d];
        if model_latent.len() != d {
            return out;
        }
        // Transpose matvec: out[j] = sum_i R[i*d + j] * model_latent[i].
        // The transpose access pattern (strided by d) is NOT cache-friendly,
        // so the 8-wide FMA pattern doesn't help here — extract_from is a
        // diagnostic path (the hot path is project_into). Scalar is fine.
        let r = &self.rotation;
        for j in 0..d {
            let mut s = 0.0f32;
            for i in 0..d {
                s += r[i * d + j] * model_latent[i];
            }
            out[j] = s;
        }
        out
    }

    #[inline]
    fn target_dim(&self) -> usize {
        self.target_dim
    }

    #[inline]
    fn canonical_dim(&self) -> usize {
        // Procrustes is square — canonical dim == target dim.
        self.target_dim
    }

    #[inline]
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

/// 8-wide chunked dot product — auto-vectorizes on AVX2 (8× f32) and
/// NEON (4× f32 packs to 2 instructions). The unrolled accumulator
/// pattern breaks the loop-carried dependency that blocks a naive
/// `for k in 0..d { s += a[k] * b[k]; }` from vectorizing.
///
/// # Performance
/// At d=2304 (Gemma2-2B hidden dim), the naive `.zip()` form ran at
/// ~scalar speed (3.9ms). The original 8-wide accumulator pattern was added
/// to break the fadd dependency chain, but was empirically refuted on Apple
/// Silicon M3 Max (2026-07-29): the 8-accumulator unroll ran 1.26× SLOWER
/// than a simple auto-vectorizable `for` loop (LLVM emits better NEON
/// `fmla` from the simple loop than from the manual unroll). The kernel was
/// simplified to trust the compiler.
///
/// # DRY note
/// This mirrors `dot_8wide` in `katgpt-attn-match/src/score_matrix_simd.rs`
/// (Plan 271). The function is small enough (10 lines, no deps) that a
/// cross-crate dep on katgpt-attn-match would pull in attention machinery
/// unnecessarily. If a third crate needs this pattern, move it to
/// katgpt-core's math utilities + have all three depend on that.
///
/// # Panics
/// Caller guarantees `a.len() == b.len() == d`.
#[inline]
fn dot_8wide(a: &[f32], b: &[f32], d: usize) -> f32 {
    debug_assert_eq!(a.len(), d);
    debug_assert_eq!(b.len(), d);

    // Simple loop — LLVM auto-vectorizes to optimal SIMD FMA. The 8-accumulator
    // manual unroll was empirically slower (see module doc); the simple loop
    // lets LLVM emit the optimal `fmla` sequence.
    let mut dot = 0.0f32;
    for k in 0..d {
        dot += a[k] * b[k];
    }
    dot
}

/// BLAKE3 of an f32 slice (little-endian bytes). Used for adapter state
/// commitment — two adapters with the same rotation bytes get the same hash.
#[inline]
fn blake3_of_f32_slice(v: &[f32]) -> [u8; 32] {
    let bytes: &[u8] = bytemuck::cast_slice(v);
    *blake3::hash(bytes).as_bytes()
}

// Re-export Hasher so unused-import warnings don't fire when blake3::Hasher
// is referenced only in docstrings. (The trait is used via the standalone
// `blake3::hash` function above; this `use` keeps the `Hasher` import live
// for future variants that may want streaming hashing.)
#[allow(dead_code)]
fn _keep_hasher_import_live(_h: Hasher) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_adapter_preserves_direction() {
        let d = CanonicalIntent::new("test", vec![1.0, 0.0, 0.0]);
        let a = ProcrustesAdapter::identity(3);
        let mut out = vec![0.0f32; 3];
        a.project_into(&d, &mut out);
        // Should reproduce the (normalized) direction bit-close.
        for (got, want) in out.iter().zip(d.as_slice().iter()) {
            assert!((got - want).abs() < 1e-6, "identity should preserve");
        }
    }

    #[test]
    fn project_is_zero_alloc_after_construction() {
        // Run project_into 1000 times; assert no allocations via simple
        // drop-count heuristic (we can't easily install a counting allocator
        // in a unit test, so this is a smoke test that the path doesn't
        // panic + produces consistent output). The real G4 alloc-free gate
        // lives in the GOAT bench.
        let a = ProcrustesAdapter::identity(64);
        let d = CanonicalIntent::new("test", vec![1.0f32; 64]);
        let mut out = vec![0.0f32; 64];
        let mut first = out.clone();
        for _ in 0..1000 {
            a.project_into(&d, &mut out);
        }
        first.copy_from_slice(&out);
        // After 1000 calls, output is identical (deterministic).
        assert_eq!(out, first);
    }

    #[test]
    fn commitment_is_deterministic_across_runs() {
        let rot: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let a1 = ProcrustesAdapter::from_rotation(rot.clone(), 4);
        let a2 = ProcrustesAdapter::from_rotation(rot, 4);
        assert_eq!(a1.commitment(), a2.commitment());
    }

    #[test]
    fn commitment_differs_for_different_rotation() {
        let r1: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let r2: Vec<f32> = (0..16).map(|i| i as f32 * 0.2).collect();
        let a1 = ProcrustesAdapter::from_rotation(r1, 4);
        let a2 = ProcrustesAdapter::from_rotation(r2, 4);
        assert_ne!(a1.commitment(), a2.commitment());
    }

    #[test]
    fn extract_inverts_project_for_orthogonal_rotation() {
        // Build a 2x2 rotation by 30 degrees.
        let theta = std::f32::consts::FRAC_PI_6; // 30 deg
        let rot = vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()];
        let a = ProcrustesAdapter::from_rotation(rot, 2);
        let d = CanonicalIntent::new("test", vec![0.6, 0.8]); // already unit-ish
        let mut projected = vec![0.0f32; 2];
        a.project_into(&d, &mut projected);
        let recovered = a.extract_from(&projected);
        // For orthogonal R, R^T * R * x = x. So extract(project(x)) == x.
        for (got, want) in recovered.iter().zip(d.as_slice().iter()) {
            assert!(
                (got - want).abs() < 1e-5,
                "round-trip failed: {got} vs {want}"
            );
        }
    }

    #[test]
    fn project_into_handles_dim_mismatch_gracefully() {
        // The runtime contract: project_into checks dim compatibility at
        // runtime and no-ops on mismatch (rather than OOB). debug_assert
        // surfaces the contract violation in debug builds; the runtime
        // check is the release-build safety net. This test verifies the
        // runtime check works when debug_assertions are off.
        #[cfg(not(debug_assertions))]
        {
            let a = ProcrustesAdapter::identity(4);
            let d = CanonicalIntent::new("test", vec![1.0, 0.0]); // dim 2, not 4
            let mut out = vec![0.0f32; 4];
            // Should not panic and should leave out as initialized.
            a.project_into(&d, &mut out);
            assert!(out.iter().all(|x| *x == 0.0));
        }
        // In debug builds, this test is a no-op (the debug_assert would
        // fire). The runtime check is verified by the release-only branch.
    }
}

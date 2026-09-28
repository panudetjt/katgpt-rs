//! CLR trait surface — extractor, verifier, direction source (Plan 284).
//!
//! These three traits are the seams between the CLR runtime and the
//! domain-specific code that supplies claim embeddings and direction vectors.
//! The runtime itself is generic over `T` (the claim payload type) and is
//! deliberately decoupled from any specific model or tokenizer.

use crate::clr::types::{Claim, Trajectory, Verdict};

/// Extracts exactly M claims from a trajectory.
///
/// Returns exactly `M` claims (where `M` is configured on the implementing
/// type — see [`crate::clr::extractor::FnClaimExtractor`]). The caller asserts
/// length; mis-sized output is a programmer error and trips a `debug_assert`.
///
/// # Zero-alloc hot path
///
/// Hot-path callers (the per-NPC CLR vote cycle in `clr_vote_minimal`)
/// should override [`Self::extract_embeddings_into`] to write claim embeddings
/// directly into a flat `&mut [f32]` buffer, avoiding the `Vec<Claim<T>>`
/// allocation entirely. The default `extract_embeddings_into` delegates to
/// `extract` + copy (still allocates) — override it for true zero-alloc.
pub trait ClaimExtractor<T> {
    /// Extract claims from `trajectory`. Length must equal the configured `M`.
    ///
    /// **Allocates** — returns a fresh `Vec<Claim<T>>` per call. Used by the
    /// `clr_vote` audit-trail path + test helpers. Hot-path callers should
    /// override [`Self::extract_embeddings_into`] instead.
    fn extract(&self, trajectory: &Trajectory<T>) -> Vec<Claim<T>>;

    /// Extract `M` claim embeddings into the flat buffer `out`, row-major.
    ///
    /// `out.len()` must be >= `M * k`. Row `m` occupies `out[m*k..(m+1)*k]`.
    /// The extractor writes each embedding in-place — no `Vec` allocation.
    ///
    /// # Default impl
    ///
    /// Delegates to [`Self::extract`] + `copy_from_slice` — still allocates
    /// (via `extract`). Override this method for the zero-alloc hot path by
    /// writing embeddings directly into `out` without going through
    /// `Vec<Claim<T>>`.
    ///
    /// # Arguments
    ///
    /// * `trajectory` — the trajectory to extract claims from.
    /// * `out` — flat embedding buffer, length >= `M * k`. Overwritten in-place.
    /// * `k` — embedding dimension (== `ClrConfig::k`). Each row is `k` f32s.
    fn extract_embeddings_into(&self, trajectory: &Trajectory<T>, out: &mut [f32], k: usize) {
        let claims = self.extract(trajectory);
        debug_assert!(
            claims.len() * k <= out.len(),
            "extract_embeddings_into: claims={} x k={} > out.len()={}",
            claims.len(),
            k,
            out.len(),
        );
        for (i, claim) in claims.iter().enumerate() {
            out[i * k..(i + 1) * k].copy_from_slice(&claim.embedding[..k]);
        }
    }
}

/// Verifies a single claim against one projection direction.
///
/// Returns `sigmoid(dot(claim.embedding, direction_vec[direction_idx]))`.
/// `direction_idx` must be in `[0, M)`. The scalar output is bounded in
/// `(0, 1)` and is the atomic unit of reliability aggregation — and, with
/// the `clr_calibration` feature, it can be made CALIBRATED against
/// recorded outcomes by wrapping any implementor in
/// [`crate::clr::calibration::CalibratedVerifier`] (Bench 807: a drifted
/// verifier's ECE 0.0924 → 0.0164; an already-calibrated one undisturbed).
/// Boundedness says the output lives in `(0,1)`; calibration is what makes
/// a 0.8 mean "~80% of such claims come true".
///
/// # Zero-alloc hot path
///
/// Implementors MUST implement [`Self::verify_embedding`] (the raw-slice core).
/// The default [`Self::verify`] delegates to it via `claim.embedding`. Hot-path
/// callers (`clr_vote_minimal`) call `verify_embedding` directly to avoid
/// constructing `Claim<T>` values.
pub trait ClaimVerifier<T> {
    /// Verdict for a raw embedding slice projected onto direction `direction_idx`.
    ///
    /// This is the core computation — implementors MUST provide this.
    fn verify_embedding(&self, embedding: &[f32], direction_idx: usize) -> Verdict;

    /// Verdict for `claim` projected onto direction `direction_idx`.
    ///
    /// Default: delegates to [`Self::verify_embedding`] via `claim.embedding`.
    #[inline(always)]
    fn verify(&self, claim: &Claim<T>, direction_idx: usize) -> Verdict {
        self.verify_embedding(&claim.embedding, direction_idx)
    }
}

/// Freeze/thaw-versioned direction vector pool.
///
/// Supplies the `M` projection directions used by [`ClaimVerifier`]. The
/// `blake3` + `version` pair lets downstream consumers detect direction-vector
/// drift across freeze/thaw cycles without re-reading the full pool.
///
/// Implementors MUST guarantee that `direction(idx)` for a fixed `version`
/// returns a byte-identical slice — verdict reproducibility depends on it.
pub trait DirectionVectorSource {
    /// Borrow the direction vector at `idx`. Length must equal configured `k`.
    fn direction(&self, idx: usize) -> &[f32];
    /// BLAKE3 hash of the full direction pool (all `M` vectors concatenated).
    fn blake3(&self) -> [u8; 32];
    /// Monotonic freeze/thaw version. Bumps on every direction-vector update.
    fn version(&self) -> u64;
}

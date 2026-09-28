//! [`SourceFeatureAdapter`] — Issue 867 Phase 2 (Proposal 010 §Feature 2):
//! the ridge-fit linear map from the Phase 1 AST histogram
//! ([`crate::source_features`]) into a model's latent steering space.
//!
//! # The construction
//!
//! ```text
//!   Rust item ──syn::visit──▶ AstHistogram (38 bins)
//!                                 │ L1-normalize (length-invariant)
//!                                 ▼
//!                    SourceFeatureAdapter (W: 38 → d_model)
//!                                 │ zero-alloc apply
//!                                 ▼
//!                       latent steering direction
//! ```
//!
//! `fit_source_adapter` fits W by ridge regression over (histogram, latent)
//! pairs; `apply_into` maps a histogram to the steering space. Two models
//! carry two INDEPENDENTLY fit adapters (Gemma-2-2B first, MiniCPM5-1B at
//! T3); the G5 gate compares CONTRASTED directions (idiomatic vs
//! non-idiomatic) across them — never absolute structure (the proposal's
//! false-positive defense).
//!
//! # Deliberately NOT a `ModelAdapter` impl
//!
//! The `ModelAdapter` seam (`crate::intent`) projects `CanonicalIntent`
//! directions (latent → latent). This adapter's input space is source
//! features, not the canonical intent space — it is the CONSTRUCTION side of
//! Proposal 010, not a projection side. Forcing the trait would be a category
//! error; the commitment/accessor conventions (BLAKE3 state hash, zero-alloc
//! hot path) are mirrored instead.
//!
//! # Length invariance
//!
//! Both fit and apply L1-normalize the histogram (`AstHistogram::normalized`)
//! so the steering direction is invariant to item length. Recipe D measured
//! the hazard this avoids: length-detrending at the DIRECTION level reversed
//! a discrimination — normalizing at the FEATURE level keeps the detrend out
//! of the learned map entirely, which is also what T3's detrend sanity check
//! presumes.
//!
//! # Solver (local f64 Cholesky — substrate note)
//!
//! The fit is the KARC shape (cold f64 accumulate + solve, f32 hot apply;
//! `katgpt-core::karc` is the reference): Gram and cross-covariance
//! accumulate in f64, factor via a local f64 Cholesky, solve the two
//! triangular systems in place, cast `Wᵀ` to f32 for the adapter. The
//! shipped substrate (`katgpt-core::linalg::ridge_solve`) was NOT consumed
//! because `pub mod linalg` there is gated behind heavyweight model features
//! (`karc_forecaster`-class); forwarding any of them into
//! `canon_source_features` would compile unrelated machinery to reach a
//! ~70-line kernel. This is the documented `katgpt-attn-match::value_fitter`
//! precedent (same workspace, same trade). Promotion trigger: a THIRD
//! consumer of small-ridge should lift the kernel to a shared
//! katgpt-core feature and convert all three call sites.
//!
//! λ > 0 is a hard precondition (the `+λI` diagonal is what makes the Gram
//! strictly PD — the katgpt-core `ridge_solve` contract). With finite input
//! and λ > 0 the factorization cannot fail, so the jitter-retry ladder
//! value_fitter needs (f32 Grams, λ = 0 allowed) is not carried here.
//!
//! # Determinism
//!
//! Pure sequential f64/f32 arithmetic in a fixed order — same inputs give
//! bit-identical weights and commitments (the KARC G4 determinism contract).

// Matrix kernels are index loops over row slices by nature; the zip form
// loses the row-slice shape. Same module-level allow as
// `katgpt-attn-match/src/value_fitter.rs`.
#![allow(clippy::needless_range_loop)]

use crate::source_features::{AstHistogram, N_AST_BINS};
use alloc::vec;
use alloc::vec::Vec;

/// The ridge-fit adapter: `ŷ = h · W` with `W` stored row-major **transposed**
/// (`d_in × d_out`) so the hot path streams contiguous rows per feature bin.
///
/// Construct via [`SourceFeatureAdapter::from_weights`] or a fit
/// ([`fit_source_adapter`] / [`fit_linear_adapter`]). The BLAKE3 commitment
/// over the weight bytes is computed once at construction (the
/// `ProcrustesAdapter` convention).
pub struct SourceFeatureAdapter {
    /// Row-major `Wᵀ`: `w_t[j * d_out + o]` is the weight from feature
    /// bin `j` to latent coordinate `o`.
    w_t: Vec<f32>,
    d_in: usize,
    d_out: usize,
    commitment: [u8; 32],
}

impl SourceFeatureAdapter {
    /// Construct from pre-fit weights (row-major `Wᵀ`, `d_in × d_out`).
    /// Computes the BLAKE3 commitment. The weights are consumed (moved).
    ///
    /// # Panics
    ///
    /// Panics if `w_t.len() != d_in * d_out` or any dim is 0.
    #[inline]
    pub fn from_weights(w_t: Vec<f32>, d_in: usize, d_out: usize) -> Self {
        assert!(
            d_in > 0 && d_out > 0,
            "SourceFeatureAdapter: dims must be > 0"
        );
        assert_eq!(
            w_t.len(),
            d_in * d_out,
            "SourceFeatureAdapter: w_t.len() ({}) != d_in * d_out ({})",
            w_t.len(),
            d_in * d_out
        );
        let commitment = blake3_of_f32_slice(&w_t);
        Self {
            w_t,
            d_in,
            d_out,
            commitment,
        }
    }

    /// Input feature dim (histogram bins for the typed path).
    #[inline]
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Output latent dim (steering space width).
    #[inline]
    pub fn d_out(&self) -> usize {
        self.d_out
    }

    /// Read-only access to the raw weights (row-major `Wᵀ`, `d_in × d_out`).
    pub fn weights_t(&self) -> &[f32] {
        &self.w_t
    }

    /// BLAKE3 commitment of the weight bytes — freeze/thaw attestation +
    /// cross-node verify. Two adapters with the same commitment produce
    /// bit-identical applies.
    #[inline]
    pub fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    /// Apply to a RAW feature slice — NO normalization. This is the generic
    /// seam: the typed [`Self::apply_into`] normalizes first, and the
    /// round-trip back-map (latent → features, T3's information-preservation
    /// instrument) consumes latents that must NOT be L1-normalized.
    ///
    /// Overwrites `out` (zero-fill + accumulate — not an accumulator).
    /// Zero-allocation after construction (G4): fixed iteration, no
    /// intermediate buffers; exact-zero feature bins are skipped (real
    /// histograms are sparse in the bins a given item never touches).
    ///
    /// # Panics
    ///
    /// Panics if `h.len() != d_in` or `out.len() != d_out`.
    pub fn apply_slice_into(&self, h: &[f32], out: &mut [f32]) {
        assert_eq!(
            h.len(),
            self.d_in,
            "apply_slice_into: h.len() ({}) != d_in ({})",
            h.len(),
            self.d_in
        );
        assert_eq!(
            out.len(),
            self.d_out,
            "apply_slice_into: out.len() ({}) != d_out ({})",
            out.len(),
            self.d_out
        );
        for v in out.iter_mut() {
            *v = 0.0;
        }
        for (j, &hj) in h.iter().enumerate() {
            if hj == 0.0 {
                continue;
            }
            let w_row = &self.w_t[j * self.d_out..(j + 1) * self.d_out];
            for (o, &wv) in w_row.iter().enumerate() {
                out[o] += hj * wv;
            }
        }
    }

    /// Typed apply: L1-normalizes the histogram (via
    /// [`AstHistogram::normalized`] — a fixed-size stack array, no
    /// allocation) then applies. Length-invariant by construction: a scaled
    /// histogram steers to the same direction.
    ///
    /// Zero-allocation after construction (G4). Overwrites `out`.
    #[inline]
    pub fn apply_into(&self, hist: &AstHistogram, out: &mut [f32]) {
        let hn = hist.normalized();
        self.apply_slice_into(&hn, out);
    }
}

/// A successful fit: the adapter plus its in-sample diagnostic.
pub struct SourceAdapterFit {
    /// The fitted adapter (`Wᵀ` cast to f32 — the shipped hot path).
    pub adapter: SourceFeatureAdapter,
    /// `‖X·W − Y‖_F / ‖Y‖_F` over the FIT rows, computed from the f64
    /// solution before the f32 cast. IN-SAMPLE diagnostic only — a ridge
    /// fit memorizes its training rows given enough capacity; held-out
    /// generalization is T3's measurement, never this number.
    pub relative_error: f32,
}

/// Fit `Wᵀ = (XᵀX + λI)⁻¹ XᵀY` over raw row-major features — the generic
/// core. `x` is `n × d_in`, `y` is `n × d_out`, both row-major.
///
/// NO normalization here: L1 normalization is the histogram wrapper's
/// semantics ([`fit_source_adapter`]); latent-space inputs (the round-trip
/// back-map) must arrive as-is.
///
/// Returns `None` only on non-finite input (a scan, never a solve failure —
/// λ > 0 keeps the Gram strictly PD). See the module doc for the solver's
/// substrate note.
///
/// # Panics
///
/// Panics on shape mismatches, empty dims, or `ridge_lambda ≤ 0 / NaN`.
pub fn fit_linear_adapter(
    x: &[f32],
    y: &[f32],
    n: usize,
    d_in: usize,
    d_out: usize,
    ridge_lambda: f32,
) -> Option<SourceAdapterFit> {
    assert!(n > 0, "fit_linear_adapter: n must be > 0");
    assert!(
        d_in > 0 && d_out > 0,
        "fit_linear_adapter: dims must be > 0"
    );
    assert_eq!(
        x.len(),
        n * d_in,
        "fit_linear_adapter: x.len() ({}) != n * d_in ({})",
        x.len(),
        n * d_in
    );
    assert_eq!(
        y.len(),
        n * d_out,
        "fit_linear_adapter: y.len() ({}) != n * d_out ({})",
        y.len(),
        n * d_out
    );
    assert!(
        ridge_lambda.is_finite() && ridge_lambda > 0.0,
        "fit_linear_adapter: ridge λ must be finite and > 0 — the +λI diagonal is \
         what makes the Gram strictly PD (the katgpt-core ridge_solve contract); \
         got {ridge_lambda}"
    );
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Gram (d_in²) + cross-covariance XᵀY (d_in × d_out), f64-accumulated —
    // the KARC cold-fit shape (f64 fit, f32 apply). The Gram accumulates the
    // lower triangle only (symmetry); one mirror at the end.
    let mut gram = vec![0.0f64; d_in * d_in];
    let mut rhs = vec![0.0f64; d_in * d_out]; // XᵀY, solved in place into Wᵀ
    for i in 0..n {
        let xr = &x[i * d_in..(i + 1) * d_in];
        let yr = &y[i * d_out..(i + 1) * d_out];
        for (j, &xj) in xr.iter().enumerate() {
            let xj = xj as f64;
            if xj == 0.0 {
                continue;
            }
            for k in j..d_in {
                gram[j * d_in + k] += xj * xr[k] as f64;
            }
            let cov_row = &mut rhs[j * d_out..(j + 1) * d_out];
            for (o, &yv) in yr.iter().enumerate() {
                cov_row[o] += xj * yv as f64;
            }
        }
    }
    for j in 0..d_in {
        for k in 0..j {
            gram[j * d_in + k] = gram[k * d_in + j];
        }
        gram[j * d_in + j] += ridge_lambda as f64;
    }

    let mut l = vec![0.0f64; d_in * d_in];
    cholesky_f64(&mut l, &gram, d_in);
    solve_lower_in_place(&mut rhs, &l, d_in, d_out); // rhs ← L⁻¹ XᵀY  (= Z)
    solve_upper_transposed_in_place(&mut rhs, &l, d_in, d_out); // rhs ← L⁻ᵀ Z (= Wᵀ)

    // In-sample relative error ‖X·W − Y‖_F / ‖Y‖_F from the f64 solution.
    let mut y_norm_sq = 0.0f64;
    let mut resid_sq = 0.0f64;
    let mut row_out = vec![0.0f64; d_out];
    for i in 0..n {
        let xr = &x[i * d_in..(i + 1) * d_in];
        let yr = &y[i * d_out..(i + 1) * d_out];
        row_out.fill(0.0);
        for (j, &xj) in xr.iter().enumerate() {
            let xj = xj as f64;
            if xj == 0.0 {
                continue;
            }
            let w_row = &rhs[j * d_out..(j + 1) * d_out];
            for (o, &wv) in w_row.iter().enumerate() {
                row_out[o] += xj * wv;
            }
        }
        for o in 0..d_out {
            let yv = yr[o] as f64;
            let r = row_out[o] - yv;
            resid_sq += r * r;
            y_norm_sq += yv * yv;
        }
    }
    let relative_error = if y_norm_sq > 0.0 {
        (resid_sq.sqrt() / y_norm_sq.sqrt()) as f32
    } else {
        0.0
    };

    let w_t: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
    Some(SourceAdapterFit {
        adapter: SourceFeatureAdapter::from_weights(w_t, d_in, d_out),
        relative_error,
    })
}

/// Fit the typed histogram → latent adapter (Issue 867 T2's
/// `fit_source_adapter(X, Y, λ)`).
///
/// Every histogram row is L1-normalized before the fit (length invariance —
/// see the module doc), so `y` must be built from the SAME normalized
/// features (e.g. T3's paired mean activations conditioned the same way).
///
/// `x.len()` rows × [`N_AST_BINS`] features; `y` is `x.len() × d_out`
/// row-major. Returns `None` only on non-finite `y` (zero-count histograms
/// normalize to all-zero rows and are benign fit inputs).
///
/// # Panics
///
/// Panics on shape mismatch or `ridge_lambda ≤ 0 / NaN`.
pub fn fit_source_adapter(
    x: &[AstHistogram],
    y: &[f32],
    d_out: usize,
    ridge_lambda: f32,
) -> Option<SourceAdapterFit> {
    let n = x.len();
    assert_eq!(
        y.len(),
        n * d_out,
        "fit_source_adapter: y.len() ({}) != n * d_out ({})",
        y.len(),
        n * d_out
    );
    let mut xn = Vec::with_capacity(n * N_AST_BINS);
    for h in x {
        xn.extend_from_slice(&h.normalized());
    }
    fit_linear_adapter(&xn, y, n, N_AST_BINS, d_out, ridge_lambda)
}

/// f64 Cholesky `A = L·Lᵀ` (lower-triangular, row-major `k×k`), reading only
/// the lower triangle of `a`.
///
/// Panics on a non-positive pivot. With λ > 0 and finite input the Gram is
/// strictly PD in exact arithmetic and pivots sit well above f64 rounding,
/// so the panic is reachable only via non-finite input — which the fit
/// already screened. This is a deliberate divergence from
/// `katgpt-core::cholesky_f64`'s relative-tolerance clamp: that kernel must
/// survive caller-supplied near-singular matrices, ours is pre-guaranteed.
fn cholesky_f64(l: &mut [f64], a: &[f64], k: usize) {
    for v in l.iter_mut().take(k * k) {
        *v = 0.0;
    }
    for j in 0..k {
        let j_row = j * k;
        let mut sum = a[j_row + j];
        for col in 0..j {
            sum -= l[j_row + col] * l[j_row + col];
        }
        assert!(
            sum > 0.0,
            "source_adapter::cholesky_f64: non-positive pivot {sum} at ({j},{j}) — \
             the Gram + λI is strictly PD for λ > 0 with finite input"
        );
        let diag = sum.sqrt();
        l[j_row + j] = diag;
        for i in (j + 1)..k {
            let i_row = i * k;
            let mut s = a[i_row + j];
            for col in 0..j {
                s -= l[i_row + col] * l[j_row + col];
            }
            l[i_row + j] = s / diag;
        }
    }
}

/// In-place forward substitution `B ← L⁻¹B` for `L` lower-triangular `k×k`
/// and `B` `k×m` row-major. Row-wise: each output row reads only already-
/// scaled earlier rows, so the two triangular solves share the in-place
/// buffer with no scratch.
fn solve_lower_in_place(b: &mut [f64], l: &[f64], k: usize, m: usize) {
    for i in 0..k {
        for j in 0..i {
            let lij = l[i * k + j];
            if lij == 0.0 {
                continue;
            }
            for o in 0..m {
                b[i * m + o] -= lij * b[j * m + o];
            }
        }
        let inv = 1.0 / l[i * k + i];
        for o in 0..m {
            b[i * m + o] *= inv;
        }
    }
}

/// In-place back substitution `B ← L⁻ᵀB` (`Lᵀ` upper-triangular), bottom-up.
fn solve_upper_transposed_in_place(b: &mut [f64], l: &[f64], k: usize, m: usize) {
    for i in (0..k).rev() {
        for j in (i + 1)..k {
            let lji = l[j * k + i]; // Lᵀ[i][j] = L[j][i]
            if lji == 0.0 {
                continue;
            }
            for o in 0..m {
                b[i * m + o] -= lji * b[j * m + o];
            }
        }
        let inv = 1.0 / l[i * k + i];
        for o in 0..m {
            b[i * m + o] *= inv;
        }
    }
}

/// BLAKE3 of an f32 slice (little-endian bytes) — adapter state commitment.
///
/// DRY note: a 4-line copy of `procrustes_adapter::blake3_of_f32_slice`,
/// which is private to a DIFFERENT feature gate (`canon`) — sharing would
/// need a feature-independent home, and a cross-module dependency on a
/// sibling adapter's private helper is worse than 4 lines. If a third copy
/// appears, lift it next to the `ModelAdapter` trait (`crate::intent`).
#[inline]
fn blake3_of_f32_slice(v: &[f32]) -> [u8; 32] {
    let bytes: &[u8] = bytemuck::cast_slice(v);
    *blake3::hash(bytes).as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Deterministic xorshift64* — no dev-dep RNG; fixed seeds make every
    /// assertion below a known-answer check.
    struct XorShift64(u64);
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self(seed.max(1))
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Uniform [0, 1).
        fn next_f32(&mut self) -> f32 {
            ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
        }
    }

    /// Planted linear ground truth: `Y = X·M` for a known random `M`.
    /// Returns `(x_train, y_train, x_heldout, y_heldout)`; rows are strictly
    /// positive (histogram-like) and never zero.
    fn planted_fixture(
        n: usize,
        d_in: usize,
        d_out: usize,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = XorShift64::new(seed);
        let mut m = vec![0.0f32; d_in * d_out];
        for v in m.iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        let gen_rows = |count: usize, rng: &mut XorShift64| -> (Vec<f32>, Vec<f32>) {
            let mut x = vec![0.0f32; count * d_in];
            for v in x.iter_mut() {
                *v = 0.1 + rng.next_f32();
            }
            let mut y = vec![0.0f32; count * d_out];
            for i in 0..count {
                let xr = &x[i * d_in..(i + 1) * d_in];
                let yr = &mut y[i * d_out..(i + 1) * d_out];
                for (j, &xv) in xr.iter().enumerate() {
                    let m_row = &m[j * d_out..(j + 1) * d_out];
                    for (o, &mv) in m_row.iter().enumerate() {
                        yr[o] += xv * mv;
                    }
                }
            }
            (x, y)
        };
        let (x, y) = gen_rows(n, &mut rng);
        let (xh, _yh) = gen_rows(128, &mut rng);
        (x, y, xh, _yh)
    }

    fn rel_err(got: &[f32], want: &[f32]) -> f32 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (g, w) in got.iter().zip(want.iter()) {
            let r = (*g - *w) as f64;
            num += r * r;
            den += (*w as f64) * (*w as f64);
        }
        (num.sqrt() / den.sqrt()) as f32
    }

    #[test]
    fn fit_recovers_planted_linear_map() {
        let (d_in, d_out, n) = (N_AST_BINS, 128, 512);
        let (x, y, _, _) = planted_fixture(n, d_in, d_out, 0x867_0001);
        let fit = fit_linear_adapter(&x, &y, n, d_in, d_out, 1e-6).expect("fit succeeds");
        assert!(
            fit.relative_error < 1e-3,
            "planted map not recovered: relative_error {}",
            fit.relative_error
        );
        // The f32 hot path reproduces the planted targets row-wise.
        let mut out = vec![0.0f32; d_out];
        for i in 0..n {
            fit.adapter
                .apply_slice_into(&x[i * d_in..(i + 1) * d_in], &mut out);
            let rel = rel_err(&out, &y[i * d_out..(i + 1) * d_out]);
            assert!(rel < 1e-3, "row {i}: f32 apply rel err {rel}");
        }
    }

    #[test]
    fn round_trip_information_preserved_on_heldout() {
        // Forward: features → latent. Back: latent → features, fit by the
        // SAME generic core with the roles swapped. Held-out rows must
        // survive the round trip — the information-preservation property T3
        // leans on when it reads steering directions back as features.
        let (d_in, d_mid, n) = (N_AST_BINS, 128, 512);
        let (x, y, xh, _) = planted_fixture(n, d_in, d_mid, 0x867_0002);
        let fwd = fit_linear_adapter(&x, &y, n, d_in, d_mid, 1e-6)
            .expect("fwd fit")
            .adapter;
        let back = fit_linear_adapter(&y, &x, n, d_mid, d_in, 1e-6)
            .expect("back fit")
            .adapter;

        let mut v = vec![0.0f32; d_mid];
        let mut h_rec = vec![0.0f32; d_in];
        let held_rows = xh.len() / d_in;
        for i in 0..held_rows {
            fwd.apply_slice_into(&xh[i * d_in..(i + 1) * d_in], &mut v);
            back.apply_slice_into(&v, &mut h_rec);
            let rel = rel_err(&h_rec, &xh[i * d_in..(i + 1) * d_in]);
            assert!(rel < 1e-2, "held-out row {i}: round-trip rel err {rel}");
        }
    }

    #[test]
    fn typed_fit_applies_scale_invariant() {
        let d_out = 64usize;
        let n = 256usize;
        let mut rng = XorShift64::new(0x867_0003);
        let hists: Vec<AstHistogram> = (0..n)
            .map(|_| AstHistogram {
                counts: core::array::from_fn(|_| (rng.next_u64() as u32) % 60 + 1),
            })
            .collect();
        // Planted targets from the NORMALIZED features — exactly what the
        // typed fit sees after its own normalization.
        let mut m = vec![0.0f32; N_AST_BINS * d_out];
        for v in m.iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        let mut y = vec![0.0f32; n * d_out];
        for (i, h) in hists.iter().enumerate() {
            let hn = h.normalized();
            let yr = &mut y[i * d_out..(i + 1) * d_out];
            for (j, &hv) in hn.iter().enumerate() {
                if hv == 0.0 {
                    continue;
                }
                let m_row = &m[j * d_out..(j + 1) * d_out];
                for (o, &mv) in m_row.iter().enumerate() {
                    yr[o] += hv * mv;
                }
            }
        }
        let fit = fit_source_adapter(&hists, &y, d_out, 1e-6).expect("typed fit");
        assert!(
            fit.relative_error < 1e-3,
            "typed fit rel err {}",
            fit.relative_error
        );

        // Scale invariance: ×7 counts → the same steering direction (within
        // f32 rounding of the normalize path; NOT bit-identical — (7c)/(7t)
        // and c/t take different division roundings).
        let mut out_a = vec![0.0f32; d_out];
        let mut out_b = vec![0.0f32; d_out];
        fit.adapter.apply_into(&hists[7], &mut out_a);
        let scaled = AstHistogram {
            counts: core::array::from_fn(|b| hists[7].counts[b] * 7),
        };
        fit.adapter.apply_into(&scaled, &mut out_b);
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!(
                (a - b).abs() <= 1e-4 * a.abs().max(1.0),
                "scale variance: {a} vs {b}"
            );
        }

        // Fit rows reproduce their planted targets through the typed apply.
        let mut out = vec![0.0f32; d_out];
        for i in 0..n {
            fit.adapter.apply_into(&hists[i], &mut out);
            let rel = rel_err(&out, &y[i * d_out..(i + 1) * d_out]);
            assert!(rel < 1e-3, "row {i}: typed apply rel err {rel}");
        }
    }

    #[test]
    fn zero_histogram_is_zero_direction_and_benign_fit_row() {
        let d_out = 16usize;
        let empty = AstHistogram {
            counts: [0; N_AST_BINS],
        };
        // Pre-filled garbage — apply OVERWRITES, it does not accumulate.
        let mut out = vec![7.5f32; d_out];
        let w_t: Vec<f32> = (0..N_AST_BINS * d_out).map(|i| i as f32 * 0.01).collect();
        let a = SourceFeatureAdapter::from_weights(w_t, N_AST_BINS, d_out);
        a.apply_into(&empty, &mut out);
        assert!(
            out.iter().all(|&v| v == 0.0),
            "zero histogram must steer to zero"
        );

        // A fit whose rows include zero histograms still succeeds: the row
        // contributes nothing to the Gram. Zero targets → relative error 0
        // by the ‖Y‖ = 0 convention.
        let hists = vec![empty; 4];
        let y = vec![0.0f32; 4 * d_out];
        let fit = fit_source_adapter(&hists, &y, d_out, 1e-3).expect("zero-row fit");
        assert_eq!(fit.relative_error, 0.0);
    }

    #[test]
    fn rank_deficient_fit_still_succeeds_under_ridge() {
        // All rows identical → rank-1 X. +λI keeps the Gram strictly PD, so
        // the fit returns the ridge-shrunk solution — a finite (large-ish)
        // in-sample error is the honest outcome, never a panic. This is the
        // behavior T3 relies on when a family's histograms saturate bins.
        let (d_in, d_out, n) = (4usize, 3usize, 16usize);
        let x = [0.5f32, 1.0, 0.25, 2.0].repeat(n);
        let y = vec![1.0f32; n * d_out];
        let fit = fit_linear_adapter(&x, &y, n, d_in, d_out, 1e-3).expect("ridge keeps PD");
        assert!(fit.relative_error.is_finite());
    }

    #[test]
    #[should_panic(expected = "ridge λ must be finite and > 0")]
    fn ridge_lambda_must_be_positive() {
        let x = vec![1.0f32; 8];
        let y = vec![1.0f32; 12];
        let _ = fit_linear_adapter(&x, &y, 4, 2, 3, 0.0);
    }

    #[test]
    #[should_panic(expected = "x.len()")]
    fn fit_dim_mismatch_panics() {
        let x = vec![1.0f32; 10]; // n * d_in would be 8
        let y = vec![1.0f32; 12];
        let _ = fit_linear_adapter(&x, &y, 4, 2, 3, 1e-3);
    }

    #[test]
    #[should_panic(expected = "h.len()")]
    fn apply_input_dim_mismatch_panics() {
        let a = SourceFeatureAdapter::from_weights(vec![0.5f32; 6], 2, 3);
        let mut out = vec![0.0f32; 3];
        a.apply_slice_into(&[1.0, 2.0, 3.0], &mut out); // d_in is 2
    }

    #[test]
    #[should_panic(expected = "out.len()")]
    fn apply_output_dim_mismatch_panics() {
        let a = SourceFeatureAdapter::from_weights(vec![0.5f32; 6], 2, 3);
        let mut out = vec![0.0f32; 4];
        a.apply_slice_into(&[1.0, 2.0], &mut out);
    }

    #[test]
    fn non_finite_input_returns_none() {
        let x = vec![1.0f32, f32::NAN, 1.0, 1.0];
        let y = vec![1.0f32; 4];
        assert!(fit_linear_adapter(&x, &y, 2, 2, 2, 1e-3).is_none());
        let x2 = vec![1.0f32; 4];
        let y2 = vec![1.0f32, f32::INFINITY, 1.0, 1.0];
        assert!(fit_linear_adapter(&x2, &y2, 2, 2, 2, 1e-3).is_none());
    }

    #[test]
    fn fit_is_deterministic_across_runs() {
        let (d_in, d_out, n) = (N_AST_BINS, 32, 128);
        let (x, y, _, _) = planted_fixture(n, d_in, d_out, 0x867_0008);
        let a = fit_linear_adapter(&x, &y, n, d_in, d_out, 1e-3).expect("fit");
        let b = fit_linear_adapter(&x, &y, n, d_in, d_out, 1e-3).expect("fit");
        assert_eq!(a.adapter.weights_t(), b.adapter.weights_t());
        assert_eq!(a.adapter.commitment(), b.adapter.commitment());
        assert_eq!(a.relative_error, b.relative_error);
    }

    #[test]
    fn commitment_deterministic_and_discriminating() {
        let w1: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let a1 = SourceFeatureAdapter::from_weights(w1.clone(), 4, 4);
        let a2 = SourceFeatureAdapter::from_weights(w1, 4, 4);
        assert_eq!(a1.commitment(), a2.commitment());
        let w2: Vec<f32> = (0..16).map(|i| i as f32 * 0.2).collect();
        let a3 = SourceFeatureAdapter::from_weights(w2, 4, 4);
        assert_ne!(a1.commitment(), a3.commitment());
    }
}

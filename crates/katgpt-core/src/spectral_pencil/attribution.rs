//! `attribution` — Hellmann–Feynman exact local feature influence
//! (Issue 676 T6; paper §5.2).
//!
//! At a **simple** eigenvalue λk with unit eigenvector `v`, the model is
//! differentiable and
//!
//! ```text
//! ∂f/∂xᵢ = vᵀ Aᵢ v
//! ```
//!
//! — one quadratic form per feature, closed-form, exact (not a
//! finite-difference estimate). At **repeated** eigenvalues the Clarke
//! subdifferential is a convex hull and per-feature influence is only
//! bounded: `|gᵢ| ≤ ‖VᵀAᵢV‖₂` over the eigenspace basis `V` (paper
//! Lemma 1) — strictly tighter than the global bound because
//! `‖VᵀAᵢV‖₂ ≤ ‖Aᵢ‖₂`.
//!
//! The γk certificate (from the same solve) flags low-trust attribution:
//! Davis–Kahan pins eigenvector sensitivity at ~2·‖ΔA‖/γk, so a small
//! eigengap means `v` itself is unstable and the attribution inherits
//! that instability. [`AttributionReport::trust`] carries the sigmoid
//! readout; consumers gate on it rather than raw gap values.
//!
//! All computation is fixed-array, f64-accumulated, zero heap.

use crate::spectral_pencil::dense::{DenseScratch, jacobi_eigen};
use crate::spectral_pencil::sym::SymPacked;

/// Per-feature attribution for one evaluation.
#[derive(Clone, Debug)]
pub struct AttributionReport<const N: usize> {
    /// `vᵀAᵢv` per feature — the exact gradient at simple eigenvalues,
    /// ONE valid subgradient element at repeated ones.
    pub influence: [f32; N],
    /// `‖Aᵢ‖₂`-scale trust per feature: `true` when γk was healthy
    /// enough that the eigenvector (hence the influence) is stable.
    pub trusted: bool,
    /// γk at the evaluation (the certificate).
    pub eigengap: f32,
}

/// Compute `vᵀAᵢv` for every feature, given the eigenvector of λk in
/// `v` (column k of the `eval_with_vectors` scratch).
///
/// Quadratic form on the packed representation: diagonal terms direct,
/// off-diagonals ×2 (both (i,j) and (j,i) cells of the full matrix
/// contribute; the packed scale convention cancels via `get`).
#[must_use]
pub fn feature_influences<const D: usize, const N: usize>(
    a: &[SymPacked<D>; N],
    v: &[f32; D],
) -> [f32; N] {
    let mut out = [0.0_f32; N];
    for (m, o) in a.iter().zip(out.iter_mut()) {
        // q = Σ_i A[i][i]·v[i]² + 2·Σ_{i<j} A[i][j]·v[i]·v[j]
        let mut q = 0.0_f64;
        for (i, &vi) in v.iter().enumerate() {
            q += f64::from(m.get(i, i)) * f64::from(vi) * f64::from(vi);
            for (j, &vj) in v.iter().enumerate().skip(i + 1) {
                q += 2.0 * f64::from(m.get(i, j)) * f64::from(vi) * f64::from(vj);
            }
        }
        *o = q as f32;
    }
    out
}

/// The full attribution path: materialize `A(x)`, solve with
/// eigenvectors, extract `vᵀAᵢv` per feature, flag trust by γk.
///
/// `trust_tau` is the eigengap threshold below which attribution is
/// flagged untrusted (default guidance: `0.1·median-gap-at-init` — for
/// seeded pencils the init guarantees γk ≥ ½, so 0.1 is a natural
/// floor).
#[must_use]
pub fn attribute<const D: usize, const N: usize>(
    a0: &SymPacked<D>,
    a: &[SymPacked<D>; N],
    x: &[f32; N],
    k: usize,
    trust_tau: f32,
    scratch: &mut DenseScratch<D>,
) -> AttributionReport<N> {
    // A(x)
    let mut ax = *a0;
    for (m, &xi) in a.iter().zip(x.iter()) {
        ax.add_scaled_into(m, xi);
    }
    let full = ax.to_full();
    jacobi_eigen(&full, true, scratch);
    let kk = k.min(D - 1);
    // Eigenvector of λ_kk (column kk).
    let mut v = [0.0_f32; D];
    for (r, vr) in v.iter_mut().enumerate() {
        *vr = scratch.v[r][kk];
    }
    let influence = feature_influences(a, &v);
    // γk (simple-gap form; exact multiplicity via float == is moot —
    // near-repeats show as small gaps, which is the honest signal).
    let alpha = scratch.values[kk];
    let below = if kk == 0 {
        f32::INFINITY
    } else {
        alpha - scratch.values[kk - 1]
    };
    let above = if kk + 1 == D {
        f32::INFINITY
    } else {
        scratch.values[kk + 1] - alpha
    };
    let eigengap = below.min(above);
    AttributionReport {
        influence,
        trusted: eigengap >= trust_tau,
        eigengap,
    }
}

/// The repeated-eigenvalue bound `‖VᵀAᵢV‖₂` over a caller-supplied
/// eigenspace basis (columns; `basis.len() == D × m` row-major) —
/// Lemma 1's tighter-than-global influence cap. Uses the Jacobi exact
/// norm on the m×m projected matrix.
#[must_use]
pub fn subdifferential_bound<const D: usize, const M: usize>(
    a_i: &SymPacked<D>,
    basis: &[[f32; D]; M], // M orthonormal columns spanning the eigenspace
    scratch: &mut DenseScratch<M>,
) -> f32 {
    // P = Bᵀ A_i B (M×M symmetric)
    let a_full = a_i.to_full();
    let mut p = [[0.0_f32; M]; M];
    for (c1, b1) in basis.iter().enumerate() {
        for (c2, b2) in basis.iter().enumerate().take(c1 + 1) {
            // entry = b1ᵀ A b2
            let mut acc = 0.0_f64;
            for (r1, &b1r) in b1.iter().enumerate() {
                if b1r == 0.0 {
                    continue;
                }
                for (r2, &b2r) in b2.iter().enumerate() {
                    acc += f64::from(b1r) * f64::from(a_full[r1][r2]) * f64::from(b2r);
                }
            }
            let e = acc as f32;
            p[c1][c2] = e;
            p[c2][c1] = e;
        }
    }
    jacobi_eigen(&p, false, scratch);
    scratch.values[M - 1].abs().max(scratch.values[0].abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_pencil::init::seeded_dense;

    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / 2.0_f32.powi(31)) * 2.0 - 1.0
        }
    }

    /// T6 G1: attribution matches central finite differences at simple
    /// eigenvalues — 10⁵ random simple-eigenvalue probes (release tier;
    /// debug tier 1k).
    #[test]
    fn attribution_matches_central_finite_difference() {
        const D: usize = 6;
        const N: usize = 4;
        const PROBES: usize = if cfg!(debug_assertions) {
            1_000
        } else {
            100_000
        };
        let init = seeded_dense::<D, N>(b"fd-probe", 3);
        let mut scratch = DenseScratch::<D>::new();
        let mut rng = Lcg(123);
        let h = 1e-3_f32;
        let mut skipped_repeated = 0_usize;
        let val_at = |xx: &[f32; N], scratch: &mut DenseScratch<D>| -> f32 {
            let mut ax = init.a0;
            for (m, &xi) in init.a.iter().zip(xx.iter()) {
                ax.add_scaled_into(m, xi);
            }
            let f = ax.to_full();
            jacobi_eigen(&f, false, scratch);
            scratch.values[3]
        };
        for _ in 0..PROBES {
            let mut x = [0.0_f32; N];
            for v in x.iter_mut() {
                *v = rng.next_f32() * 3.0;
            }
            let rep = attribute(&init.a0, &init.a, &x, 3, 0.05, &mut scratch);
            if !rep.trusted {
                skipped_repeated += 1; // repeated/near-repeated: FD ill-posed
                continue;
            }
            for i in 0..N {
                let mut xp = x;
                let mut xm = x;
                xp[i] += h;
                xm[i] -= h;
                let fd = (val_at(&xp, &mut scratch) - val_at(&xm, &mut scratch)) / (2.0 * h);
                let err = (rep.influence[i] - fd).abs();
                assert!(
                    err < 5e-3,
                    "attribution vs FD drifted {err} at feature {i} (closed {} vs fd {fd})",
                    rep.influence[i]
                );
            }
        }
        // The seeded init guarantees γk ≥ ½ on the box, so almost every
        // probe is trusted; tolerate only rare near-degeneracies.
        assert!(
            skipped_repeated <= PROBES / 10,
            "{skipped_repeated} untrusted of {PROBES}"
        );
    }

    /// T6 law: |vᵀAᵢv| ≤ ‖Aᵢ‖₂ always (Lemma 1 with ‖vvᵀ‖* = 1).
    #[test]
    fn attribution_never_exceeds_spectral_norm() {
        const D: usize = 6;
        const N: usize = 4;
        let init = seeded_dense::<D, N>(b"norm-cap-probe", 2);
        let mut scratch = DenseScratch::<D>::new();
        let mut rng = Lcg(7);
        for _ in 0..512 {
            let mut x = [0.0_f32; N];
            for v in x.iter_mut() {
                *v = rng.next_f32() * 5.0;
            }
            let rep = attribute(&init.a0, &init.a, &x, 2, 0.0, &mut scratch);
            for (i, &inf) in rep.influence.iter().enumerate() {
                let norm =
                    crate::spectral_pencil::bounds::norm_jacobi_exact(&init.a[i], &mut scratch);
                assert!(
                    inf.abs() <= norm + 1e-4,
                    "|{inf}| > ‖A{i}‖ = {norm} (Lemma 1 violated)"
                );
            }
        }
    }
}

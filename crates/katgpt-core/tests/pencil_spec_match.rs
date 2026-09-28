//! Spec-match test for the spectral pencil Lean 4 proofs (Issue 678 P3).
//!
//! Asserts the Rust substrate (`crates/katgpt-core/src/spectral_pencil/`)
//! matches the Lean 4 spec at `katgpt-rs/.proofs/KatgptProof/Pencil/`:
//!
//! * **T1 (Sym.lean)** — the `sym` isometry: `‖sym(v)‖_F == ‖v‖₂` and
//!   `⟨sym(u), sym(v)⟩_F == ⟨u, v⟩` for the mirrored-√2 packing. Pinned on
//!   the SAME hand instance as the Lean SpecTests (`!![1, 2√2; 2√2, 3]`,
//!   Frobenius square exactly 18) plus randomized sweeps — the Rust
//!   `frobenius_norm`/`frobenius_dot` must match an independent f64
//!   computation of the full-matrix Frobenius norm.
//!
//! * **T2 (Weyl.lean)** — eigenvalues are 1-Lipschitz in the spectral
//!   norm: `|λᵢ(A) − λᵢ(B)| ≤ ‖A − B‖₂` sampled over seeded pencils
//!   (Jacobi eigenvalues, exact spectral norms).
//!
//! * **T4 (Eigengap.lean, shift core)** — the scalar shift is
//!   gap-invariant: `A + c·I` has the same eigengaps as `A` (the shift
//!   lemma `eigval_add_smul_one` on real data).
//!
//! * **T4 (Eigengap.lean, ladder closeout)** — the ladder's unit gap is
//!   EXACT (both sides of the zero: −1→0 and 0→+1), and the seeded
//!   pencil `A(x) = A₀ + ΣxᵢAᵢ` at ‖x‖∞ ≤ BOX_R keeps the Lean-proven
//!   slot (`eigengap_ladder_ge_half`: the gap toward the −1 rung) at
//!   ≥ ½ — Lemma 2 measured on the production constructor.
//!
//! Sort-order note: Rust `jacobi_eigen` sorts ASCENDING (`values[0]` =
//! smallest); the Lean `eigval` array is ANTITONE (decreasing). The
//! correspondence is `lean eigval i = rust values[D−1−i]`, so the Lean
//! gap `eigval k₀ − eigval (k₀+1)` (zero minus the −1 below it) reads as
//! the ASCENDING gap `values[k] − values[k−1]` at the zero's position k.
//!
//! If these fail, the Lean spec and the Rust substrate have drifted;
//! reconcile before merging.
//!
//! Run: `cargo test --features spectral_pencil --test pencil_spec_match`
//!
//! Cross-references:
//! - Issue: `.issues/678_lean_spectral_pencil_package.md`
//! - Lean: `.proofs/KatgptProof/Pencil/{Sym,Weyl,Loewner,Eigengap}.lean`
//! - Rust: `crates/katgpt-core/src/spectral_pencil/`

#![cfg(feature = "spectral_pencil")]

use katgpt_core::spectral_pencil::dense::{DenseScratch, jacobi_eigen};
use katgpt_core::spectral_pencil::init::{BOX_R, seeded_dense};
use katgpt_core::spectral_pencil::sym::SymPacked;

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

/// T1: the Rust `frobenius_norm` matches an independent full-matrix
/// computation (the isometry lets the packed loop equal the true
/// Frobenius norm — this is `sym_isometry_norm` on real data).
#[test]
fn t1_frobenius_norm_matches_full_matrix() {
    const D: usize = 6;
    let mut rng = Lcg(20260822);
    for _ in 0..128 {
        #[allow(clippy::needless_range_loop)]
        let full = {
            let mut full = [[0.0_f32; D]; D];
            for i in 0..D {
                for j in i..D {
                    let v = rng.next_f32() * 3.0;
                    full[i][j] = v;
                    full[j][i] = v;
                }
            }
            full
        };
        let packed = SymPacked::<D>::pack_from_full(&full);
        // independent: the actual matrix's Frobenius norm (offs as-is)
        let mut sq = 0.0_f64;
        for row in &full {
            for &v in row {
                sq += f64::from(v) * f64::from(v);
            }
        }
        let want = (sq as f32).sqrt();
        let got = packed.frobenius_norm();
        let ulps = ((want.to_bits() as i64) - (got.to_bits() as i64)).abs();
        assert!(
            ulps <= 2,
            "isometry broke: full ‖A‖_F {want} vs packed {got} ({ulps} ulps)"
        );
    }
}

/// T1: the Rust `frobenius_dot` matches the independent full-matrix
/// Frobenius inner product (`sym_isometry_dot` on real data).
#[test]
fn t1_frobenius_dot_matches_full_matrix() {
    const D: usize = 5;
    let mut rng = Lcg(781);
    for _ in 0..128 {
        let mut fa = [[0.0_f32; D]; D];
        let mut fb = [[0.0_f32; D]; D];
        for i in 0..D {
            for j in i..D {
                let a = rng.next_f32();
                let b = rng.next_f32();
                fa[i][j] = a;
                fa[j][i] = a;
                fb[i][j] = b;
                fb[j][i] = b;
            }
        }
        #[allow(clippy::needless_range_loop)]
        let pa = SymPacked::<D>::pack_from_full(&fa);
        let pb = SymPacked::<D>::pack_from_full(&fb);
        let mut want = 0.0_f64;
        for i in 0..D {
            for j in 0..D {
                want += f64::from(fa[i][j]) * f64::from(fb[i][j]);
            }
        }
        let got = pa.frobenius_dot(&pb);
        assert!(
            (want as f32 - got).abs() < 1e-4 * got.abs().max(1.0),
            "inner-product isometry broke: {want} vs {got}"
        );
    }
}

/// T1: the Lean SpecTest hand instance — storage `!![1, 2√2; 2√2, 3]`
/// carries `!![1, 2; 2, 3]`, Frobenius square exactly 18 (1+4+4+9).
/// Cross-checks the Lean instance at `Pencil/SpecTests.lean`.
#[test]
fn t1_hand_instance_matches_lean_spec_test() {
    const D: usize = 2;
    let mut data = [[0.0_f32; D]; D];
    let s2 = 2.0_f32 * core::f32::consts::SQRT_2;
    data[0][0] = 1.0;
    data[0][1] = s2;
    data[1][0] = s2;
    data[1][1] = 3.0;
    let v = SymPacked::<D> { data };
    let n = v.frobenius_norm();
    assert!(
        (n * n - 18.0).abs() < 1e-4,
        "hand instance: ‖v‖² = {} want 18",
        n * n
    );
    // to_full recovers the intended matrix
    let full = v.to_full();
    assert!(
        (full[0][1] - 2.0).abs() < 1e-6,
        "off-scale wrong: {}",
        full[0][1]
    );
}

/// T2: Weyl 1-Lipschitz on sampled pencil matrices — |λᵢ(A) − λᵢ(B)|
/// ≤ ‖A−B‖₂ with exact Jacobi eigenvalues and spectral norms.
#[test]
fn t2_weyl_lipschitz_sampled() {
    const D: usize = 6;
    let mut rng = Lcg(678678);
    let mut sa = DenseScratch::<D>::new();
    let mut sb = DenseScratch::<D>::new();
    let mut sd = DenseScratch::<D>::new();
    for _ in 0..64 {
        let mut fa = [[0.0_f32; D]; D];
        let mut fb = [[0.0_f32; D]; D];
        for i in 0..D {
            for j in i..D {
                let a = rng.next_f32();
                let b = rng.next_f32();
                fa[i][j] = a;
                fa[j][i] = a;
                fb[i][j] = b;
                fb[j][i] = b;
            }
        }
        #[allow(clippy::needless_range_loop)]
        let pa = SymPacked::<D>::pack_from_full(&fa);
        let pb = SymPacked::<D>::pack_from_full(&fb);
        jacobi_eigen(&fa, false, &mut sa);
        jacobi_eigen(&fb, false, &mut sb);
        // A − B
        let mut fd = fa;
        for i in 0..D {
            for j in 0..D {
                fd[i][j] -= fb[i][j];
            }
        }
        jacobi_eigen(&fd, false, &mut sd);
        let norm_d = sd.values.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        let _ = (&pa, &pb); // packing path exercised by T1 tests
        for k in 0..D {
            let diff = (sa.values[k] - sb.values[k]).abs();
            assert!(
                diff <= norm_d * (1.0 + 1e-3) + 1e-5,
                "Weyl violated at k={k}: |{:.6} - {:.6}| = {diff:.6} > ‖A-B‖ {norm_d:.6}",
                sa.values[k],
                sb.values[k]
            );
        }
    }
}

/// T4 (shift core): `A + c·I` preserves every eigengap — the shift lemma
/// `eigval_add_smul_one` (gap invariance under a scalar shift) on real
/// Jacobi spectra.
#[test]
#[allow(clippy::needless_range_loop)]
fn t4_shift_preserves_eigengaps() {
    const D: usize = 5;
    let mut rng = Lcg(4711);
    let mut s0 = DenseScratch::<D>::new();
    let mut s1 = DenseScratch::<D>::new();
    for trial in 0..32 {
        let c = rng.next_f32() * 4.0;
        let mut f0 = [[0.0_f32; D]; D];
        for i in 0..D {
            for j in i..D {
                let v = rng.next_f32() * 2.0;
                f0[i][j] = v;
                f0[j][i] = v;
            }
        }
        let mut f1 = f0;
        for i in 0..D {
            f1[i][i] += c;
        }
        jacobi_eigen(&f0, false, &mut s0);
        jacobi_eigen(&f1, false, &mut s1);
        for k in 0..D {
            // every eigenvalue shifted by exactly c (within Jacobi tolerance)
            assert!(
                (s1.values[k] - (s0.values[k] + c)).abs() < 1e-3,
                "trial {trial} k={k}: shift moved λ by {}, want {c}",
                s1.values[k] - s0.values[k]
            );
        }
        // gaps identical
        for k in 1..D {
            let g0 = s0.values[k - 1] - s0.values[k];
            let g1 = s1.values[k - 1] - s1.values[k];
            assert!(
                (g0 - g1).abs() < 1e-3,
                "trial {trial} gap {k}: {g1} vs {g0}"
            );
        }
    }
}

/// T4 (ladder closeout): the pure diagonal ladder's eigengaps are EXACTLY
/// 1 on both sides of the zero — the input gap of Lemma 2
/// (`ladder_unit_gap` in Lean). Ascending convention: the Lean-proven
/// slot (zero minus the −1 below it, `eigval k₀ − eigval (k₀+1)` with
/// k₀ = D−1−k) reads `values[k] − values[k−1]`.
#[test]
#[allow(clippy::needless_range_loop)]
fn t4_ladder_unit_gap_exact() {
    const D: usize = 6;
    let mut s = DenseScratch::<D>::new();
    for k in 1..D - 1 {
        // the Rust-orientation ladder: −1 at i<k, 0@k, +1 at i>k
        let mut ladder = [[0.0_f32; D]; D];
        for i in 0..D {
            ladder[i][i] = if i < k {
                -1.0
            } else if i == k {
                0.0
            } else {
                1.0
            };
        }
        jacobi_eigen(&ladder, false, &mut s);
        // ascending: −1×k, 0@k, +1×(D−1−k)
        let gap_below = s.values[k] - s.values[k - 1];
        let gap_above = s.values[k + 1] - s.values[k];
        assert!(
            (gap_below - 1.0).abs() < 1e-5,
            "k={k} gap toward −1 rung: {gap_below} want 1"
        );
        assert!(
            (gap_above - 1.0).abs() < 1e-5,
            "k={k} gap toward +1 rung: {gap_above} want 1"
        );
        // the zero itself sits exactly at ascending index k
        assert!(
            (s.values[k] - 0.0).abs() < 1e-5,
            "k={k} zero moved: {}",
            s.values[k]
        );
    }
}

/// T4 (ladder closeout, Lemma 2 measured): the seeded dense pencil
/// `A(x) = A₀ + ΣxᵢAᵢ` (conjugated ladder + `αᵢI + diag(εᵢ)` features)
/// keeps the Lean-proven slot at ≥ ½ for every sampled input with
/// `‖x‖∞ ≤ BOX_R` — `eigengap_ladder_ge_half` on the production
/// constructor (scalar shift = Σxᵢαᵢ, jitter ‖E‖ ≤ R·n·(1/20n) = ¼).
#[test]
#[allow(clippy::needless_range_loop)]
fn t4_seeded_pencil_gap_survives_jitter() {
    const D: usize = 8;
    const N: usize = 4;
    const K: usize = 1; // zero at ascending index 1 (one −1 rung below)
    let init = seeded_dense::<D, N>(b"pencil-spec-match/ladder-gap", K);
    let a0 = init.a0.to_full();
    let mut rng = Lcg(678_678);
    let mut s = DenseScratch::<D>::new();
    for trial in 0..64 {
        let mut ax = a0;
        for ai in &init.a {
            let x = rng.next_f32() * BOX_R; // |x| ≤ R
            let full = ai.to_full();
            for i in 0..D {
                for j in 0..D {
                    ax[i][j] += x * full[i][j];
                }
            }
        }
        jacobi_eigen(&ax, false, &mut s);
        // the Lean-proven slot: ascending gap toward the −1 rung
        let gap = s.values[K] - s.values[K - 1];
        assert!(
            gap >= 0.5 - 1e-3,
            "trial {trial}: jittered gap {gap:.4} < ½ — Lemma 2 violated"
        );
    }
}

//! Cross-module property tests for `spectral_pencil` (Issue 676 T1–T5
//! G1 gates). Per-module unit tests live beside their code; the gates
//! here assert the paper-level invariants that span modules.

use crate::spectral_pencil::dense::{DenseScratch, jacobi_eigen};
use crate::spectral_pencil::init::{seeded_dense, seeded_tridiag};
use crate::spectral_pencil::sym::SymPacked;
use crate::spectral_pencil::tridiag::{TriScratch, count_below};
use crate::spectral_pencil::{DensePencil, TridiagPencil, eigengap_at};

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / 2.0_f32.powi(31)) * 2.0 - 1.0 // [-1, 1)
    }
}

/// T4 G1 (dense): γk ≥ ½ for seeded pencils on the box ‖x‖∞ ≤ 5, the
/// paper's Lemma 2 gate. Frozen seed set + box-corner + interior sweep.
#[test]
fn seeded_dense_eigengap_ge_half_on_box() {
    const D: usize = 8;
    const N: usize = 6;
    for k in [1_usize, 2, 4, 6] {
        for seed_idx in 0..8_u64 {
            let seed = format!("gap-probe/{seed_idx}/{k}");
            let init = seeded_dense::<D, N>(seed.as_bytes(), k);
            let pencil = DensePencil::<D, N> {
                a0: init.a0,
                a: init.a,
            };
            let mut scratch = DenseScratch::<D>::new();
            let mut rng = Lcg(seed_idx ^ (k as u64) << 8);

            // Box corners (±5) + interior draws.
            for trial in 0..64 {
                let x = if trial < 2 {
                    [if trial == 0 { 5.0 } else { -5.0 }; N]
                } else {
                    let mut x = [0.0_f32; N];
                    for v in x.iter_mut() {
                        *v = rng.next_f32() * 5.0;
                    }
                    x
                };
                let ev = pencil.eval(&x, k, &mut scratch);
                let gap = ev.eigengap.expect("D >= 2");
                assert!(
                    gap >= 0.5 - 1e-4,
                    "seed {seed_idx} k {k} trial {trial}: γk {gap} < ½ (Lemma 2 violated)"
                );
            }
        }
    }
}

/// T4 G1 (tridiag): same gate for the tridiagonal family (rescaled ε).
#[test]
fn seeded_tridiag_eigengap_ge_half_on_box() {
    const D: usize = 8;
    const N: usize = 6;
    for k in [1_usize, 3, 5] {
        for seed_idx in 0..8_u64 {
            let seed = format!("tri-gap/{seed_idx}/{k}");
            let init = seeded_tridiag::<D, N>(seed.as_bytes(), k);
            let pencil = TridiagPencil::<D, N> {
                a0: init.a0,
                a: init.a,
            };
            let mut scratch = TriScratch::<D>::new();
            let mut rng = Lcg(seed_idx ^ 0xBEEF ^ (k as u64));

            // γk via the fused-matrix dense solve (Sturm gives values, not
            // gaps — cross-check the two kernels while we're here).
            for trial in 0..64 {
                let x = if trial < 2 {
                    [if trial == 0 { 5.0 } else { -5.0 }; N]
                } else {
                    let mut x = [0.0_f32; N];
                    for v in x.iter_mut() {
                        *v = rng.next_f32() * 5.0;
                    }
                    x
                };
                // Fused tridiag → full → Jacobi (independent of the Sturm
                // path; validates both).
                let mut full = [[0.0_f32; D]; D];
                for (i, &d) in pencil.a0.diag.iter().enumerate() {
                    full[i][i] = d;
                }
                for i in 0..(D - 1) {
                    full[i][i + 1] = pencil.a0.off[i];
                    full[i + 1][i] = pencil.a0.off[i];
                }
                for (m, &xi) in pencil.a.iter().zip(x.iter()) {
                    for (i, d) in m.diag.iter().enumerate() {
                        full[i][i] += xi * d;
                    }
                    for i in 0..(D - 1) {
                        full[i][i + 1] += xi * m.off[i];
                        full[i + 1][i] += xi * m.off[i];
                    }
                }
                let mut ds = DenseScratch::<D>::new();
                jacobi_eigen(&full, false, &mut ds);
                let gap = eigengap_at(&ds.values, k).expect("D >= 2");
                assert!(
                    gap >= 0.5 - 1e-3,
                    "seed {seed_idx} k {k} trial {trial}: γk {gap} < ½"
                );

                // Sturm-vs-Jacobi cross-check on the same fused matrix.
                let sturm_val = pencil.eval(&x, k, &mut scratch);
                assert!(
                    (sturm_val - ds.values[k]).abs() < 1e-3,
                    "seed {seed_idx} k {k}: sturm {sturm_val} vs jacobi {}",
                    ds.values[k]
                );
            }
        }
    }
}

/// T5 law: mirror duality λk(−A) = −λ_{d−k+1}(A).
#[test]
fn mirror_duality_holds() {
    const D: usize = 7;
    let mut rng = Lcg(5);
    for _ in 0..64 {
        let mut full = [[0.0_f32; D]; D];
        for (i, j) in (0..D).flat_map(|i| (i..D).map(move |j| (i, j))) {
            let v = rng.next_f32();
            full[i][j] = v;
            full[j][i] = v;
        }
        let packed = SymPacked::<D>::pack_from_full(&full);
        let mut neg = packed;
        neg.negate();
        let mut s1 = DenseScratch::<D>::new();
        let mut s2 = DenseScratch::<D>::new();
        jacobi_eigen(&packed.to_full(), false, &mut s1);
        jacobi_eigen(&neg.to_full(), false, &mut s2);
        for k in 0..D {
            let lhs = s2.values[k]; // λk(−A)
            let rhs = -s1.values[D - 1 - k]; // −λ_{d−k+1}(A)
            assert!(
                (lhs - rhs).abs() < 1e-5,
                "k {k}: λk(−A) {lhs} vs −λ_(d−k+1)(A) {rhs}"
            );
        }
    }
}

/// Shape law sanity: k=1 concave, k=d convex on a line sweep (paper
/// Claim 1 — the structural property the T7 DSL will build on).
#[test]
fn extremal_k_concave_convex_on_line_sweeps() {
    const D: usize = 5;
    const N: usize = 2;
    // A generic seeded pencil; sweep x[0] ∈ [−4, 4] at fixed x[1].
    let init = seeded_dense::<D, N>(b"shape-sweep", 2);
    let pencil = DensePencil::<D, N> {
        a0: init.a0,
        a: init.a,
    };
    let mut scratch = DenseScratch::<D>::new();
    let xs: Vec<f32> = (0..41).map(|i| -4.0 + 8.0 * (i as f32) / 40.0).collect();
    let mut f_at = |x0: f32, k: usize| -> f32 {
        let ev = pencil.eval(&[x0, 0.3], k, &mut scratch);
        ev.value
    };
    // Concavity of k=0: f(mid) ≥ midpoint of f(lo), f(hi) on every triple.
    for w in xs.windows(3) {
        let mid_f = f_at(w[1], 0);
        let avg = 0.5 * (f_at(w[0], 0) + f_at(w[2], 0));
        assert!(
            mid_f >= avg - 1e-3,
            "k=0 concavity violated: f(mid) {mid_f} < avg {avg}"
        );
        // Convexity of k=d−1.
        let mid_g = f_at(w[1], D - 1);
        let avg_g = 0.5 * (f_at(w[0], D - 1) + f_at(w[2], D - 1));
        assert!(
            mid_g <= avg_g + 1e-3,
            "k=d convexity violated: f(mid) {mid_g} > avg {avg_g}"
        );
    }
}

/// Determinism gate: whole-module repeat evaluation bit-identical.
#[test]
fn evaluations_are_bit_reproducible() {
    const D: usize = 6;
    const N: usize = 4;
    let init = seeded_dense::<D, N>(b"repro", 3);
    let pencil = DensePencil::<D, N> {
        a0: init.a0,
        a: init.a,
    };
    let mut s1 = DenseScratch::<D>::new();
    let mut s2 = DenseScratch::<D>::new();
    let mut rng = Lcg(9);
    for _ in 0..32 {
        let mut x = [0.0_f32; N];
        for v in x.iter_mut() {
            *v = rng.next_f32() * 3.0;
        }
        let e1 = pencil.eval(&x, 3, &mut s1);
        let e2 = pencil.eval(&x, 3, &mut s2);
        assert_eq!(e1.value.to_bits(), e2.value.to_bits());
        assert_eq!(e1.jacobi, e2.jacobi);
    }
}

/// The 10⁶-tridiag Sturm == full-solve count gate (release tier — the
/// debug tier lives in tridiag.rs at 10k). Literally the issue's spec:
/// 10⁶ random tridiags, θ at interior midpoints of each matrix's own
/// solved spectrum.
#[test]
#[cfg_attr(debug_assertions, ignore = "release tier: 1M Jacobi solves")]
fn sturm_matches_full_solve_1m_release() {
    const D: usize = 8;
    let mut work = [0.0_f32; D];
    let mut rng = Lcg(0xABCD);
    let mut checked = 0_u64;
    for _ in 0..1_000_000_u64 {
        let mut diag = [0.0_f32; D];
        let mut off = [0.0_f32; D];
        for v in diag.iter_mut() {
            *v = rng.next_f32() * 2.0;
        }
        for v in off.iter_mut().take(D - 1) {
            *v = rng.next_f32();
        }
        let mut full = [[0.0_f32; D]; D];
        for i in 0..D {
            full[i][i] = diag[i];
        }
        for i in 0..(D - 1) {
            full[i][i + 1] = off[i];
            full[i + 1][i] = off[i];
        }
        let mut ds = DenseScratch::<D>::new();
        jacobi_eigen(&full, false, &mut ds);
        for w in ds.values.windows(2) {
            if w[0] == w[1] {
                continue;
            }
            let theta = 0.5 * (w[0] + w[1]);
            let sturm = count_below(&diag, &off, theta, &mut work);
            let dense = ds.values.iter().filter(|&&v| v < theta).count() as u32;
            assert_eq!(sturm, dense, "theta {theta}");
            checked += 1;
        }
    }
    // 10⁶ matrices × ~7 interior midpoints each — the issue's full
    // population, one release run.
    assert!(checked >= 6_000_000, "population too small: {checked}");
}

/// TridiagPencil.eval end-to-end against the dense solve on the same
/// fused matrix (cross-kernel agreement at pencil level).
#[test]
fn tridiag_pencil_eval_matches_dense() {
    const D: usize = 6;
    const N: usize = 3;
    let init = seeded_tridiag::<D, N>(b"pencil-xcheck", 2);
    let pencil = TridiagPencil::<D, N> {
        a0: init.a0,
        a: init.a,
    };
    let mut ts = TriScratch::<D>::new();
    let mut rng = Lcg(3);
    for _ in 0..128 {
        let mut x = [0.0_f32; N];
        for v in x.iter_mut() {
            *v = rng.next_f32() * 4.0;
        }
        for k in 0..D {
            let sturm = pencil.eval(&x, k, &mut ts);
            // dense on the fused matrix
            let mut full = [[0.0_f32; D]; D];
            for (i, d) in pencil.a0.diag.iter().enumerate() {
                full[i][i] = *d;
            }
            for i in 0..(D - 1) {
                full[i][i + 1] = pencil.a0.off[i];
                full[i + 1][i] = pencil.a0.off[i];
            }
            for (m, &xi) in pencil.a.iter().zip(x.iter()) {
                for (i, d) in m.diag.iter().enumerate() {
                    full[i][i] += xi * d;
                }
                for i in 0..(D - 1) {
                    full[i][i + 1] += xi * m.off[i];
                    full[i + 1][i] += xi * m.off[i];
                }
            }
            let mut ds = DenseScratch::<D>::new();
            jacobi_eigen(&full, false, &mut ds);
            assert!(
                (sturm - ds.values[k]).abs() < 1e-3,
                "k {k}: sturm {sturm} vs dense {}",
                ds.values[k]
            );
        }
    }
}

/// The exact-integer-count property the chain seam cares about: counts
/// are small non-negative integers, monotone in μ, and stable across
/// repeat evaluation (bit-identical).
#[test]
fn sturm_counts_are_monotone_and_reproducible() {
    const D: usize = 7;
    const N: usize = 2;
    let init = seeded_tridiag::<D, N>(b"count-probe", 3);
    let pencil = TridiagPencil::<D, N> {
        a0: init.a0,
        a: init.a,
    };
    let mut ts = TriScratch::<D>::new();
    let x = [1.5_f32, -2.0];
    let mut prev = 0_u32;
    for step in 0..40 {
        let mu = -3.0 + 6.0 * (step as f32) / 39.0;
        let c = pencil.count_below(&x, mu, &mut ts);
        let c2 = pencil.count_below(&x, mu, &mut ts);
        assert_eq!(c, c2, "not reproducible at mu {mu}");
        assert!(c >= prev, "not monotone at mu {mu}: {c} < {prev}");
        assert!(c <= D as u32);
        prev = c;
    }
}

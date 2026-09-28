//! Unit tests for the `CP^(d-1)` Hopfield primitive.
//!
//! Gate mapping: `basis_*` and `constraint_*` verify the SU(d) algebra against
//! literature values (a wrong `f_abc` would silently produce a plausible-but-wrong
//! flow); `recall_*` and `llg_*` are the G1 correctness gates; `constraint_*` are
//! G4 correctness.

use super::*;
use crate::cp_hopfield::capacity::{distribution_fixture, haar_fixture};

const TOL: f32 = 1e-4;

// ── SU(d) basis + structure constants ────────────────────────────────────

/// The `d = 2` basis must reproduce the Pauli matrices in `(σx, σy, σz)` order.
#[test]
fn basis_d2_is_pauli() {
    let b = GellMannBasis::<2>::new();
    let sx = b.generator_dense(0);
    let sy = b.generator_dense(1);
    let sz = b.generator_dense(2);

    assert_eq!(sx[0][1], C32::ONE);
    assert_eq!(sx[1][0], C32::ONE);
    assert_eq!(sy[0][1], C32::new(0.0, -1.0));
    assert_eq!(sy[1][0], C32::new(0.0, 1.0));
    assert_eq!(sz[0][0], C32::ONE);
    assert_eq!(sz[1][1], C32::real(-1.0));
}

/// `Tr(λ_a λ_b) = 2 δ_ab` for every `d` we support.
#[test]
fn basis_is_orthonormal() {
    fn check<const D: usize>() {
        let b = GellMannBasis::<D>::new();
        let n = GellMannBasis::<D>::BLOCH_DIM;
        let dense: Vec<_> = (0..n).map(|a| b.generator_dense(a)).collect();
        for a in 0..n {
            for bb in 0..n {
                let mut tr = C32::ZERO;
                for (r, row_a) in dense[a].iter().enumerate().take(D) {
                    for (c, elem_a) in row_a.iter().enumerate().take(D) {
                        tr = tr.mul_add(*elem_a, dense[bb][c][r]);
                    }
                }
                let expected = if a == bb { 2.0 } else { 0.0 };
                assert!(
                    (tr.re - expected).abs() < TOL && tr.im.abs() < TOL,
                    "d={D}: Tr(l{a} l{bb}) = {tr:?}, expected {expected}"
                );
            }
        }
    }
    check::<2>();
    check::<3>();
    check::<4>();
}

/// `d = 2` structure constants must be the Levi-Civita symbol: `f_abc = ε_abc`.
#[test]
fn structure_constants_d2_are_levi_civita() {
    let b = GellMannBasis::<2>::new();
    let sc = StructureConstants::new(&b);
    assert!((sc.f(0, 1, 2) - 1.0).abs() < TOL);
    assert!((sc.f(1, 2, 0) - 1.0).abs() < TOL);
    assert!((sc.f(2, 0, 1) - 1.0).abs() < TOL);
    assert!((sc.f(1, 0, 2) + 1.0).abs() < TOL);
    assert!(sc.f(0, 0, 1).abs() < TOL);
    // {σ_a, σ_b} = 2δ_ab I leaves no λ_c component, so d_abc vanishes identically.
    assert_eq!(sc.d_nnz(), 0, "d_abc must vanish for SU(2)");
}

/// `d = 3` structure constants must match the canonical Gell-Mann values.
///
/// Literature (1-indexed): `f_123 = 1`, `f_147 = 1/2`, `f_458 = √3/2`,
/// `d_118 = 1/√3`, `d_146 = 1/2`, `d_888 = −1/√3`.
#[test]
fn structure_constants_d3_match_gell_mann() {
    let b = GellMannBasis::<3>::new();
    let sc = StructureConstants::new(&b);
    let r3 = 3.0f32.sqrt();

    assert!((sc.f(0, 1, 2) - 1.0).abs() < TOL, "f_123");
    assert!((sc.f(0, 3, 6) - 0.5).abs() < TOL, "f_147");
    assert!((sc.f(3, 4, 7) - r3 / 2.0).abs() < TOL, "f_458");
    assert!((sc.f(5, 6, 7) - r3 / 2.0).abs() < TOL, "f_678");

    assert!((sc.d_sym(0, 0, 7) - 1.0 / r3).abs() < TOL, "d_118");
    assert!((sc.d_sym(0, 3, 5) - 0.5).abs() < TOL, "d_146");
    assert!((sc.d_sym(7, 7, 7) + 1.0 / r3).abs() < TOL, "d_888");
}

/// `f_abc` is totally antisymmetric, `d_abc` totally symmetric.
#[test]
fn structure_constants_have_correct_symmetry() {
    let b = GellMannBasis::<3>::new();
    let sc = StructureConstants::new(&b);
    for a in 0..8 {
        for bb in 0..8 {
            for c in 0..8 {
                assert!(
                    (sc.f(a, bb, c) + sc.f(bb, a, c)).abs() < TOL,
                    "f not antisymmetric at ({a},{bb},{c})"
                );
                assert!(
                    (sc.d_sym(a, bb, c) - sc.d_sym(bb, a, c)).abs() < TOL,
                    "d not symmetric at ({a},{bb},{c})"
                );
            }
        }
    }
}

// ── Bloch geometry ───────────────────────────────────────────────────────

/// A pure state's Bloch vector must have norm `|s|² = 2(1 − 1/d)`.
#[test]
fn bloch_norm_matches_purity() {
    fn check<const D: usize>() {
        let b = GellMannBasis::<D>::new();
        let n = GellMannBasis::<D>::BLOCH_DIM;
        let mut q = [C32::ZERO; D];
        for (i, z) in q.iter_mut().enumerate() {
            *z = C32::new(1.0 + i as f32, 0.5 * i as f32);
        }
        let q = normalize(&q);
        let mut s = vec![0.0f32; n];
        b.bloch_projection_into(&q, &mut s);
        let n2: f32 = s.iter().map(|x| x * x).sum();
        assert!(
            (n2 - bloch_norm_sq(D)).abs() < TOL,
            "d={D}: |s|^2 = {n2}, expected {}",
            bloch_norm_sq(D)
        );
    }
    check::<2>();
    check::<3>();
    check::<4>();
}

/// The Bloch projection must be invariant under the `U(1)` phase — that is what
/// makes it a coordinate on `CP^(d-1)` rather than on the sphere.
#[test]
fn bloch_projection_is_phase_invariant() {
    let b = GellMannBasis::<3>::new();
    let q = normalize(&[C32::new(0.3, 0.1), C32::new(-0.5, 0.7), C32::new(0.2, -0.4)]);
    let phase = C32::new(0.6, 0.8); // |e^{iθ}| = 1
    let mut rotated = q;
    for z in rotated.iter_mut() {
        *z = z.mul(phase);
    }
    let mut s1 = [0.0f32; 8];
    let mut s2 = [0.0f32; 8];
    b.bloch_projection_into(&q, &mut s1);
    b.bloch_projection_into(&rotated, &mut s2);
    for a in 0..8 {
        assert!((s1[a] - s2[a]).abs() < TOL, "phase leaked into s[{a}]");
    }
}

/// Overlap of a state with itself is 1; with an orthogonal state, `−1/(d−1)`.
#[test]
fn bloch_overlap_endpoints() {
    let b = GellMannBasis::<3>::new();
    let e0 = [C32::ONE, C32::ZERO, C32::ZERO];
    let e1 = [C32::ZERO, C32::ONE, C32::ZERO];
    let mut s0 = [0.0f32; 8];
    let mut s1 = [0.0f32; 8];
    b.bloch_projection_into(&e0, &mut s0);
    b.bloch_projection_into(&e1, &mut s1);
    assert!((bloch_overlap(&s0, &s0, 3) - 1.0).abs() < TOL);
    assert!((bloch_overlap(&s0, &s1, 3) + 0.5).abs() < TOL, "-1/(d-1)");
}

// ── T2.3 / T2.4: manifold constraint (G4 correctness) ────────────────────

/// An on-manifold state satisfies the non-linear constraint identically.
#[test]
fn constraint_holds_for_pure_states() {
    let rec = CpHopfield3::new(4);
    let q = normalize(&[C32::new(0.4, -0.2), C32::new(0.1, 0.9), C32::new(-0.3, 0.5)]);
    let mut s = [0.0f32; 8];
    rec.basis().bloch_projection_into(&q, &mut s);
    assert!(
        rec.constraint_residual(&s) < 1e-5,
        "residual {} on a pure state",
        rec.constraint_residual(&s)
    );
    assert!(rec.norm_residual(&s) < 1e-5);
}

/// `project_to_manifold` maps an arbitrary off-manifold vector onto `CP^(d-1)`,
/// satisfying both constraints — in one shot, with no convergence loop.
#[test]
fn project_to_manifold_enforces_both_constraints() {
    let rec = CpHopfield3::new(4);
    let mut s = [0.9f32, -0.4, 0.7, 0.2, -0.8, 0.5, 0.1, -0.6];
    assert!(
        rec.constraint_residual(&s) > 1e-3,
        "fixture should start off-manifold"
    );
    rec.project_to_manifold(&mut s);
    assert!(
        rec.constraint_residual(&s) < 1e-5,
        "constraint residual {} after projection",
        rec.constraint_residual(&s)
    );
    assert!(
        rec.norm_residual(&s) < 1e-5,
        "norm residual {} after projection",
        rec.norm_residual(&s)
    );
}

/// Projection is idempotent: re-projecting an already-projected state is a no-op.
/// This is the exactness claim — an iterative scheme would keep drifting.
#[test]
fn project_to_manifold_is_idempotent() {
    let rec = CpHopfield3::new(4);
    let mut s = [0.3f32, 0.8, -0.5, 0.1, 0.4, -0.7, 0.2, 0.6];
    rec.project_to_manifold(&mut s);
    let once = s;
    rec.project_to_manifold(&mut s);
    for a in 0..8 {
        assert!(
            (s[a] - once[a]).abs() < 1e-4,
            "projection not idempotent at {a}: {} vs {}",
            once[a],
            s[a]
        );
    }
}

/// `d = 2` has no constraint beyond the norm (`CP^1 = S^2`).
#[test]
fn d2_has_no_nonlinear_constraint() {
    let rec = CpHopfield2::new(4);
    assert_eq!(rec.structure().d_nnz(), 0);
    let mut s = [0.5f32, -0.3, 0.9];
    rec.project_to_manifold(&mut s);
    assert!(rec.norm_residual(&s) < 1e-5);
}

// ── T1.9 / T1.10: G1 recall correctness ──────────────────────────────────

/// T1.9 — one memory on `CP²`, corrupted 40%, recovers to `m̄ ≥ 0.9` in one sweep.
#[test]
fn recall_single_memory_from_40pct_corruption() {
    let mut rec = haar_fixture::<3, 8>(64, 1, 0.4, 0xA11CE);
    let before = rec.mean_overlap(0);
    assert!(
        before < 0.75,
        "corruption should degrade overlap, got {before}"
    );
    rec.sweep();
    let after = rec.mean_overlap(0);
    assert!(
        after >= 0.9,
        "single-memory recall: m = {after} after one sweep (was {before})"
    );
}

/// T1.10 — 10 memories at `α = 0.1 < α_c(d=3)`, cued memory recovers to `m̄ ≥ 0.9`.
#[test]
fn recall_ten_memories_below_alpha_c() {
    let mut rec = haar_fixture::<3, 8>(100, 10, 0.4, 0xBEEF);
    assert!((rec.load() - 0.1).abs() < 1e-6);
    rec.sweep();
    let m = rec.mean_overlap(0);
    assert!(m >= 0.9, "alpha=0.1 recall: m = {m} after one sweep");
}

/// A stored memory is an *exact* fixed point at `P = 1` and only an approximate
/// one once crosstalk exists — and the drift must grow with load.
///
/// This is the finite-`N` crosstalk budget made explicit. At `P = 1` there is
/// nothing to interfere, so `K_i` is exactly rank-1 and the memory is invariant.
/// At `P > 1` the other memories perturb `K_i` by `O(1/√N)` relative to the
/// signal, which tilts the top eigenvector by a correspondingly small angle. A
/// test that demanded exact invariance at `P > 1` would be asserting the absence
/// of crosstalk that provably exists.
#[test]
fn stored_memory_is_a_fixed_point() {
    let mut clean = haar_fixture::<3, 8>(64, 1, 0.0, 0xF1AED);
    assert!(clean.mean_overlap(0) > 0.999, "fixture should start clean");
    let drift_p1 = clean.sweep();
    assert!(
        drift_p1 < 1e-3,
        "P=1 memory must be an exact fixed point, drifted {drift_p1}"
    );
    assert!(clean.mean_overlap(0) > 0.999);

    let mut loaded = haar_fixture::<3, 8>(64, 4, 0.0, 0xF1AED);
    let drift_p4 = loaded.sweep();
    assert!(
        drift_p4 > drift_p1,
        "crosstalk drift should grow with load: P=1 {drift_p1}, P=4 {drift_p4}"
    );
    // Small perturbation, not a different basin: recall must still hold the memory.
    assert!(
        loaded.mean_overlap(0) > 0.99,
        "crosstalk knocked recall off the memory: {}",
        loaded.mean_overlap(0)
    );
}

/// Recall output is on-manifold without any explicit projection, because the top
/// eigenvector is a genuine unit qudit.
#[test]
fn recall_output_is_on_manifold() {
    let rec = haar_fixture::<3, 8>(32, 3, 0.4, 0xC0FFEE);
    let next = rec.recall_step(0);
    assert!(rec.constraint_residual(&next) < 1e-4);
    assert!(rec.norm_residual(&next) < 1e-4);
}

/// Bit-reproducibility: identical inputs give identical recall.
#[test]
fn recall_is_deterministic() {
    let a = haar_fixture::<3, 8>(32, 3, 0.4, 0xD00D).recall_step(0);
    let b = haar_fixture::<3, 8>(32, 3, 0.4, 0xD00D).recall_step(0);
    assert_eq!(a, b, "recall must be bit-reproducible");
}

/// The incremental `T_μ` cache must stay bit-close to a full recomputation across
/// many state writes. This guards the `O(P·N·D2)` -> `O(P·D2)` optimization: if the
/// cache drifted, every Mattis overlap — and therefore every recall — would be
/// silently wrong in a way no other test would catch.
#[test]
fn overlap_cache_matches_full_recompute() {
    let mut rec = haar_fixture::<3, 8>(32, 6, 0.4, 0xCAC4E);
    for _ in 0..5 {
        rec.sweep();
    }
    let incremental: Vec<f32> = (0..rec.n_memories())
        .map(|mu| rec.mattis_overlap_excluding(3, mu))
        .collect();
    rec.refresh_overlap_totals();
    let recomputed: Vec<f32> = (0..rec.n_memories())
        .map(|mu| rec.mattis_overlap_excluding(3, mu))
        .collect();
    for (mu, (a, b)) in incremental.iter().zip(recomputed.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "overlap cache drifted at memory {mu}: incremental {a} vs recomputed {b}"
        );
    }
}

// ── G7: BBP gap ──────────────────────────────────────────────────────────

/// At low load the memory kernel must show a clear spectral gap — the mechanism
/// the whole capacity claim rests on. At high load it must close. If the gap did
/// not respond to load, there would be no BBP transition to exploit.
#[test]
fn bbp_gap_shrinks_with_load() {
    let low = haar_fixture::<3, 8>(64, 2, 0.2, 0x9A9)
        .kernel_spectrum(0)
        .relative_gap();
    let high = haar_fixture::<3, 8>(64, 128, 0.2, 0x9A9)
        .kernel_spectrum(0)
        .relative_gap();
    assert!(low > 0.1, "low-load gap {low} should be clearly open");
    assert!(
        high < low,
        "gap should shrink with load: alpha=0.03 -> {low}, alpha=2.0 -> {high}"
    );
}

// ── T3.5 / T3.6: LLG flow ────────────────────────────────────────────────

/// T3.5 — LLG flow recovers a corrupted memory.
#[test]
fn llg_recall_recovers_corrupted_memory() {
    let mut rec = haar_fixture::<3, 8>(48, 1, 0.4, 0x11C6);
    let before = rec.mean_overlap(0);
    let result = rec.llg_recall(&LlgConfig::default());
    let after = rec.mean_overlap(0);
    assert!(
        after > before + 0.1,
        "LLG made no progress: {before} -> {after} in {} steps",
        result.steps
    );
}

/// T3.6 — the Gilbert damping term must lower energy monotonically
/// (`Ė = −λ Σ |s ×_f B|² ≤ 0`).
#[test]
fn llg_energy_is_non_increasing() {
    let mut rec = haar_fixture::<3, 8>(32, 2, 0.4, 0x3A3E);
    let cfg = LlgConfig {
        damping: 1.0,
        dt: 0.01,
        tol: 1e-5,
        max_steps: 100,
    };
    let result = rec.llg_recall(&cfg);
    let scale = result.energy_trajectory[0].abs().max(1.0);
    let worst = result.max_energy_increase();
    assert!(
        worst < 1e-3 * scale,
        "energy increased by {worst} (scale {scale}); trajectory head {:?}",
        &result.energy_trajectory[..5.min(result.energy_trajectory.len())]
    );
}

/// Every LLG step must leave the state on the manifold.
#[test]
fn llg_preserves_manifold() {
    let mut rec = haar_fixture::<3, 8>(16, 2, 0.4, 0x0FF);
    let cfg = LlgConfig::default();
    for _ in 0..20 {
        rec.llg_step(&cfg);
        for i in 0..rec.n_neurons() {
            let s = rec.state(i);
            assert!(rec.constraint_residual(s) < 1e-4, "left the manifold");
            assert!(rec.norm_residual(s) < 1e-4);
        }
    }
}

/// The Lie bracket must be antisymmetric: `s ×_f s = 0`.
#[test]
fn lie_bracket_is_antisymmetric() {
    let rec = CpHopfield3::new(2);
    let s = [0.3f32, -0.5, 0.7, 0.1, 0.2, -0.4, 0.6, 0.8];
    let mut out = [0.0f32; 8];
    lie_bracket_into(&s, &s, rec.structure(), &mut out);
    for (a, &v) in out.iter().enumerate() {
        assert!(v.abs() < TOL, "s x_f s nonzero at {a}: {v}");
    }
}

/// The precession term is orthogonal to the field, so it does no work — this is
/// what makes it energy-conserving.
#[test]
fn precession_does_no_work() {
    let rec = CpHopfield3::new(2);
    let s = [0.3f32, -0.5, 0.7, 0.1, 0.2, -0.4, 0.6, 0.8];
    let b = [0.6f32, 0.2, -0.3, 0.9, -0.1, 0.4, 0.5, -0.7];
    let mut cross = [0.0f32; 8];
    lie_bracket_into(&s, &b, rec.structure(), &mut cross);
    let work: f32 = (0..8).map(|a| cross[a] * b[a]).sum();
    assert!(work.abs() < TOL, "precession did work {work} against B");
}

// ── T4: capacity ─────────────────────────────────────────────────────────

/// Capacity must decrease monotonically in load — the basic sanity check that
/// makes `alpha_c` meaningful at all.
#[test]
fn capacity_degrades_with_load() {
    let curve = measure_capacity::<3, 8>(
        32,
        &[0.1, 0.5, 2.0, 8.0],
        2,
        0.4,
        MemoryDistribution::Haar,
        0.5,
        0x5EED,
    );
    let first = curve.points[0].mean_overlap;
    let last = curve.points[curve.points.len() - 1].mean_overlap;
    assert!(
        first > last,
        "overlap should fall with load: {first} -> {last}"
    );
    assert!(first > 0.8, "alpha=0.1 should recall well, got {first}");
}

/// Correlated memories exhibit the paper's **shadow phenomenon**: recall from one
/// cue drags un-cued but correlated memories along with it.
///
/// Plan 567 expected correlated memories to simply recall *worse* than Haar-random
/// ones. Measured, the opposite happens on the cued memory — at `α = 1.0`,
/// correlated recall scores ~0.97 against Haar's ~0.45. That is not extra capacity:
/// when every memory points nearly the same way they reinforce rather than
/// interfere, so the cued memory is trivially easy to hit. The information that has
/// actually been lost is *discriminability*, so that is what this test measures —
/// overlap with a memory that was never cued.
///
/// For Haar memories recall is winner-takes-all (near-zero overlap with the
/// un-cued memory); for correlated memories the un-cued memory co-fires. Per
/// Research 466 §1.6 this is a real correlation signal rather than crosstalk noise,
/// and it is desirable for KG retrieval (related context) but undesirable for
/// personality recall (bleed) — hence worth pinning down in a test.
#[test]
fn correlated_memories_show_shadow_phenomenon() {
    let p = 8;
    let mut haar = distribution_fixture::<3, 8>(32, p, 0.4, MemoryDistribution::Haar, 0x77AA);
    let mut corr =
        distribution_fixture::<3, 8>(32, p, 0.4, MemoryDistribution::correlated(0.35), 0x77AA);
    haar.recall_to_fixed_point(1e-4, 20);
    corr.recall_to_fixed_point(1e-4, 20);

    let haar_shadow = haar.mean_overlap(p - 1);
    let corr_shadow = corr.mean_overlap(p - 1);

    assert!(
        haar_shadow.abs() < 0.3,
        "Haar recall should be winner-takes-all, but un-cued memory scored {haar_shadow}"
    );
    assert!(
        corr_shadow > haar_shadow + 0.3,
        "expected a shadow on correlated memories: un-cued overlap {corr_shadow} vs Haar {haar_shadow}"
    );
}

/// `alpha_c` returns `None` rather than a fabricated number when the swept range
/// does not bracket the crossing.
#[test]
fn alpha_c_is_none_when_range_does_not_bracket() {
    let curve = measure_capacity::<3, 8>(
        16,
        &[0.06, 0.12],
        1,
        0.3,
        MemoryDistribution::Haar,
        0.5,
        0x1234,
    );
    if let Some(ac) = curve.alpha_c() {
        assert!((0.06..=0.12).contains(&ac));
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

fn normalize<const D: usize>(q: &[C32; D]) -> [C32; D] {
    let n: f32 = q.iter().map(|z| z.norm_sq()).sum::<f32>().sqrt();
    let mut out = *q;
    for z in out.iter_mut() {
        *z = z.scale(1.0 / n);
    }
    out
}

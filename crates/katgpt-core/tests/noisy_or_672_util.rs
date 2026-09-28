//! Issue 672 T4 — noisy-OR core util tests (the civ delegation's gate).
//!
//! `noisy_or` is UNGATED (crate-root util next to `sigmoid`) because the
//! riir-games-civ salience gate delegates to it under DEFAULT features —
//! so these tests run ungated too.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --test noisy_or_672_util
//! ```

use katgpt_core::{noisy_or, noisy_or_stable};

/// Boundary identities (the issue's "all-0 → 0; any-1 → 1").
#[test]
fn boundary_identities() {
    assert_eq!(noisy_or(&[]), 0.0);
    assert_eq!(noisy_or(&[0.0, 0.0, 0.0]), 0.0);
    assert_eq!(noisy_or(&[0.0, 1.0, 0.0]), 1.0);
    assert_eq!(noisy_or(&[1.0, 1.0]), 1.0);
    // Out-of-range inputs are clamped (probabilities-style weights).
    assert_eq!(noisy_or(&[-3.0, 0.5]), noisy_or(&[0.0, 0.5]));
    assert_eq!(noisy_or(&[2.0, 0.5]), 1.0);
}

/// Monotone non-decreasing in every k.
#[test]
fn monotone_in_each_term() {
    let ks = [0.1f32, 0.3, 0.05];
    let base = noisy_or(&ks);
    for i in 0..ks.len() {
        let mut bumped = ks;
        bumped[i] += 0.2;
        assert!(noisy_or(&bumped) >= base);
    }
}

/// **Bit-compatibility with the civ salience-gate formula it replaces**
/// (Issue 672 T4 delegation): for the two-term case the util performs
/// exactly `(1.0−k₀)·(1.0−k₁)` then `1.0−·` — same ops, same order, clamp
/// is identity on in-range inputs. Pinned over a dense grid so the
/// delegation cannot drift.
#[test]
fn bit_identical_to_civ_two_term_formula() {
    let mut failures = 0u32;
    let n = 64;
    for i in 0..n {
        for j in 0..n {
            let c = i as f32 / n as f32;
            let boost = j as f32 / n as f32;
            let civ = 1.0f32 - (1.0f32 - c) * (1.0f32 - boost);
            let util = noisy_or(&[c, boost]);
            if civ.to_bits() != util.to_bits() {
                failures += 1;
            }
        }
    }
    assert_eq!(
        failures, 0,
        "noisy_or must be bit-identical to the civ formula"
    );
}

/// The log1p-stable variant: both forms agree with an f64 ground truth
/// within their respective fp natures, and the stable form keeps full
/// relative accuracy where the direct form's `1 − product` cancels.
#[test]
fn stable_variant_matches_and_keeps_resolution() {
    // f64 reference for small spans.
    for ks in [
        [0.1f32, 0.2, 0.3].as_slice(),
        &[0.5, 0.5],
        &[0.001, 0.002],
        &[0.02, 0.01, 0.03],
    ] {
        let truth = 1.0 - ks.iter().map(|&k| (1.0f64) - k as f64).product::<f64>();
        let d = noisy_or(ks) as f64;
        let s = noisy_or_stable(ks) as f64;
        // Both within 1e-4 ABSOLUTE of truth (probabilities scale)…
        assert!((d - truth).abs() < 1e-4, "direct {d} vs truth {truth}");
        assert!((s - truth).abs() < 1e-4, "stable {s} vs truth {truth}");
        // …and for small outputs the STABLE form keeps relative accuracy
        // where the direct form's cancellation eats digits (the motivating
        // regime): stable's relative error ≤ direct's ×3, or both tiny.
        let rel_d = (d - truth).abs() / truth.max(1e-12);
        let rel_s = (s - truth).abs() / truth.max(1e-12);
        assert!(
            rel_s <= rel_d * 3.0 + 1e-6,
            "stable rel {rel_s} vs direct rel {rel_d}"
        );
    }
    // Boundary identities hold for the stable form too.
    assert!(noisy_or_stable(&[0.0, 0.0]).abs() < 1e-7);
    assert!(noisy_or_stable(&[0.0, 1.0]) > 0.9999999);
    // Resolution: 1000 terms of 0.001 → true ≈ 1 − e⁻¹ ≈ 0.632.
    let many = [0.001f32; 1000];
    let s = noisy_or_stable(&many);
    assert!(
        (s - 1.0 + std::f32::consts::E.recip()).abs() < 1e-3,
        "stable {s}"
    );
    // The direct form agrees here too (no underflow yet at these
    // magnitudes) — the stable form is a strict generalization, not a
    // behavior change, on spans where direct is numerically fine.
    let d = noisy_or(&many);
    assert!((d - s).abs() < 1e-4);
}

/// Determinism: identical inputs → identical bits, both forms.
#[test]
fn deterministic_bits() {
    let ks: Vec<f32> = (0..33).map(|i| (i % 7) as f32 / 23.0).collect();
    let a = noisy_or(&ks);
    let b = noisy_or(&ks);
    assert_eq!(a.to_bits(), b.to_bits());
    let sa = noisy_or_stable(&ks);
    let sb = noisy_or_stable(&ks);
    assert_eq!(sa.to_bits(), sb.to_bits());
}

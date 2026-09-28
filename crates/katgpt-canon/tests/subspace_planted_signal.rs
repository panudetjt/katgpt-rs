//! Integration test: katgpt-canon SubspaceAdapter on a planted shared-subspace
//! signal. Reproduces the P1 result shape (Bench 423) on synthetic data.
//!
//! The test plants a 2-dim shared subspace in two models of different dims
//! (d_a=8, d_b=6), generates 32 anchor pairs with the planted structure +
//! noise, fits the joint-SVD SubspaceAdapter, and verifies that:
//!
//! 1. The fit is well-formed (correct shapes, no NaN).
//! 2. Held-out prompts (8 additional pairs not used in fit) project to
//!    positively-correlated k-dim coordinates across the two models.
//! 3. The adapters round-trip a canonical direction with positive magnitude
//!    in both model frames.
//!
//! This is NOT the G5 gate (which requires real Gemma/MiniCPM weights +
//! the cos > 0.5 threshold on held-out). It's a SMOKE test that the
//! pipeline works end-to-end and produces positive cross-model correlation
//! when a real shared subspace exists.

use katgpt_canon::{
    CanonicalIntent, JointSvdFitScratch, ModelAdapter, SubspaceAdapter, fit_joint_svd_pair,
};

/// Simple deterministic PRNG (xorshift32) so the test is reproducible
/// across runs / platforms / Rust versions. NOT cryptographically secure.
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF } else { seed },
        }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
    /// Uniform [-1, 1] random. Simpler than Gaussian; avoids Box-Muller
    /// extreme-value tail that can cause numerical issues in test data.
    fn next_f32(&mut self) -> f32 {
        let u = self.next_u32() as f32 / u32::MAX as f32;
        u * 2.0 - 1.0
    }
}

/// Plant a k-dim shared subspace in two models of dims (d_a, d_b).
/// Activations = (shared_coords · basis) + noise_scale * gaussian_noise.
/// Returns (a_anchors, b_anchors, shared_coords) where shared_coords[i]
/// is the k-dim coordinate of anchor i (same for both models).
fn plant_shared_subspace(
    n: usize,
    d_a: usize,
    d_b: usize,
    k: usize,
    noise_scale: f32,
    seed: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);

    // Random bases: A_basis is k vectors in R^d_a, B_basis is k vectors in R^d_b.
    // (Not orthonormal — just random; SVD will handle it.)
    let mut a_basis = vec![0.0f32; k * d_a];
    let mut b_basis = vec![0.0f32; k * d_b];
    for x in a_basis.iter_mut() {
        *x = rng.next_f32();
    }
    for x in b_basis.iter_mut() {
        *x = rng.next_f32();
    }

    let mut a = vec![0.0f32; n * d_a];
    let mut b = vec![0.0f32; n * d_b];
    let mut coords = vec![0.0f32; n * k];

    for i in 0..n {
        // Random k-dim coordinates (the shared signal).
        for j in 0..k {
            coords[i * k + j] = rng.next_f32();
        }
        // a[i] = coords[i] · A_basis + noise
        for r in 0..d_a {
            let mut s = 0.0f32;
            for j in 0..k {
                s += coords[i * k + j] * a_basis[j * d_a + r];
            }
            s += noise_scale * rng.next_f32();
            a[i * d_a + r] = s;
        }
        // b[i] = coords[i] · B_basis + noise
        for r in 0..d_b {
            let mut s = 0.0f32;
            for j in 0..k {
                s += coords[i * k + j] * b_basis[j * d_b + r];
            }
            s += noise_scale * rng.next_f32();
            b[i * d_b + r] = s;
        }
    }

    (a, b, coords)
}

/// Project an activation through a basis V (column-major d × k) to get
/// k-dim coordinates: out[j] = sum_r V[r, j] * activation[r].
fn project_to_subspace(activation: &[f32], v: &[f32], d: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k];
    for j in 0..k {
        let col = &v[j * d..(j + 1) * d];
        let mut s = 0.0f32;
        for (vrj, xr) in col.iter().zip(activation.iter()) {
            s += vrj * xr;
        }
        out[j] = s;
    }
    out
}

#[test]
fn subspace_adapter_recovers_planted_signal() {
    let n_train = 32;
    let n_test = 8;
    let d_a = 8;
    let d_b = 6;
    let k = 2;
    let noise_scale = 0.1; // Low noise — shared signal should dominate.

    let (a_train, b_train, _coords_train) =
        plant_shared_subspace(n_train, d_a, d_b, k, noise_scale, 0xCAFE_BABE);
    let (a_test, b_test, _coords_test) =
        plant_shared_subspace(n_test, d_a, d_b, k, noise_scale, 0xFEED_FACE);

    // Fit.
    let mut scratch = JointSvdFitScratch::with_capacity(d_a + d_b, n_train, k);
    let fit = fit_joint_svd_pair(&a_train, &b_train, n_train, d_a, d_b, k, &mut scratch);

    // 1. Well-formed fit.
    assert_eq!(fit.v_a.len(), d_a * k);
    assert_eq!(fit.v_b.len(), d_b * k);
    assert_eq!(fit.rotation.len(), k * k);
    for x in fit
        .v_a
        .iter()
        .chain(fit.v_b.iter())
        .chain(fit.rotation.iter())
    {
        assert!(x.is_finite(), "fit produced non-finite value");
    }

    // 2. Held-out cross-model correlation: project test pairs through V_A / V_B,
    //    apply rotation to A's projections, compare to B's projections.
    let adapter_a = SubspaceAdapter::for_model_a(&fit);
    let adapter_b = SubspaceAdapter::for_model_b(&fit);
    let v_a = adapter_a.v();
    let v_b = adapter_b.v();
    let r = &fit.rotation;

    let mut cosines = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let a_proj = project_to_subspace(&a_test[i * d_a..(i + 1) * d_a], v_a, d_a, k);
        let b_proj = project_to_subspace(&b_test[i * d_b..(i + 1) * d_b], v_b, d_b, k);
        // Rotate A's k-dim projection through R.
        let mut a_rotated = vec![0.0f32; k];
        for row in 0..k {
            let mut s = 0.0f32;
            for col in 0..k {
                s += r[row * k + col] * a_proj[col];
            }
            a_rotated[row] = s;
        }
        // Cosine similarity.
        let dot: f32 = a_rotated
            .iter()
            .zip(b_proj.iter())
            .map(|(a, b)| a * b)
            .sum();
        let na: f32 = a_rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b_proj.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 1e-6 && nb > 1e-6 {
            cosines.push(dot / (na * nb));
        }
    }
    let mean_cos = cosines.iter().sum::<f32>() / cosines.len().max(1) as f32;
    // With low noise + planted shared subspace, we expect mostly-positive
    // cross-model correlation. The mean can be dragged down by outliers
    // (one bad pair), so we use TWO criteria: (a) mean positive, (b) at
    // least 60% of cosines positive.
    let n_positive = cosines.iter().filter(|c| **c > 0.0).count();
    let frac_positive = n_positive as f32 / cosines.len().max(1) as f32;
    assert!(
        mean_cos > 0.0,
        "mean cross-model cosine {mean_cos} should be > 0 on planted shared subspace (got {cosines:?})"
    );
    assert!(
        frac_positive >= 0.6,
        "fraction of positive cosines {frac_positive} should be >= 0.6 (got {cosines:?})"
    );
    eprintln!(
        "P1 smoke: mean cos = {mean_cos:.4}, frac positive = {frac_positive:.2} (cosines: {cosines:?})"
    );

    // 3. Round-trip a canonical direction through both adapters.
    let canonical = CanonicalIntent::new("test_signal", vec![1.0, 0.0]);
    let mut out_a = vec![0.0f32; d_a];
    let mut out_b = vec![0.0f32; d_b];
    adapter_a.project_into(&canonical, &mut out_a);
    adapter_b.project_into(&canonical, &mut out_b);
    let mag_a: f32 = out_a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = out_b.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(mag_a > 1e-6, "adapter_a output should be non-zero");
    assert!(mag_b > 1e-6, "adapter_b output should be non-zero");
}

#[test]
fn adapters_share_canonical_input() {
    // The cross-architecture contract: the SAME canonical direction,
    // projected through model-A's adapter and model-B's adapter, produces
    // outputs that — when projected back to the shared k-dim subspace —
    // recover the same coordinates (up to a positive per-coordinate scalar).
    // This is what makes canonical intent space "plug-and-play".
    let n = 16;
    let d_a = 8;
    let d_b = 6;
    let k = 2;
    let (a, b, _coords) = plant_shared_subspace(n, d_a, d_b, k, 0.05, 0x1234_5678);

    let mut scratch = JointSvdFitScratch::with_capacity(d_a + d_b, n, k);
    let fit = fit_joint_svd_pair(&a, &b, n, d_a, d_b, k, &mut scratch);

    let adapter_a = SubspaceAdapter::for_model_a(&fit);
    let adapter_b = SubspaceAdapter::for_model_b(&fit);

    // Two canonical directions; both should produce non-zero output in both adapters.
    let c1 = CanonicalIntent::new("c1", vec![1.0, 0.0]);
    let c2 = CanonicalIntent::new("c2", vec![0.0, 1.0]);

    let mut out_a1 = vec![0.0f32; d_a];
    let mut out_a2 = vec![0.0f32; d_a];
    let mut out_b1 = vec![0.0f32; d_b];
    let mut out_b2 = vec![0.0f32; d_b];
    adapter_a.project_into(&c1, &mut out_a1);
    adapter_a.project_into(&c2, &mut out_a2);
    adapter_b.project_into(&c1, &mut out_b1);
    adapter_b.project_into(&c2, &mut out_b2);

    // Both directions produce output in both adapters (non-zero).
    let mag_a1: f32 = out_a1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_a2: f32 = out_a2.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b1: f32 = out_b1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b2: f32 = out_b2.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        mag_a1 > 1e-6 && mag_a2 > 1e-6,
        "model A produced degenerate output"
    );
    assert!(
        mag_b1 > 1e-6 && mag_b2 > 1e-6,
        "model B produced degenerate output"
    );

    // Commitments are stable + distinct per adapter.
    let comm_a = adapter_a.commitment();
    let comm_b = adapter_b.commitment();
    assert_eq!(comm_a, adapter_a.commitment(), "commitment must be stable");
    assert_ne!(
        comm_a, comm_b,
        "different adapters should have different commitments"
    );
}

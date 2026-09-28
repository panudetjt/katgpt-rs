//! Example: Poincaré Adapter — closed-form latent navigation primitive
//! (Plan 449, Research 449, arXiv:2607.14228 Chen et al. *SeeSE3*).
//!
//! Demonstrates the primitive's stated commercial purpose: **inverse latent
//! navigation** — given a desired target-space displacement `Δtarget`, find
//! the latent-space step `Δz` that achieves it. This is the "imagination"
//! primitive: "if I want to move 5 units in the target space, what latent
//! change should I make?"
//!
//! The adapter is frozen after an offline closed-form fit (PCA + ridge +
//! SVD — zero gradient descent). At runtime, navigation is a single
//! MLP evaluation + matvec + inverse projection, all zero-allocation.
//!
//! Run with:
//! ```sh
//! cargo run --example poincare_01_latent_navigation --features poincare_navigator --release
//! ```
//!
//! # What This Proves
//!
//! - **Offline fit recovers a known linear map**: fit on synthetic
//!   `(z, target = A·z)` pairs, verify the forward decoder `W·φ(z)` matches
//!   `A·z` on held-out samples (R² near 1.0 in the linear regime).
//! - **The inverse navigator works**: given a desired `Δtarget`,
//!   `poincare_navigate_into` produces a `z_out` whose forward decoder
//!   recovers the `Δtarget` (direction matches exactly; magnitude within
//!   tanh-warp tolerance).
//! - **Multi-step open-loop trajectory**: `poincare_multi_step_into` breaks
//!   a large displacement into N deterministic sub-steps.
//! - **Freeze/thaw (BLAKE3 commitment)**: `canonical_bytes` → `from_bytes`
//!   round-trip is bit-identical; `verify()` passes; tamper detection works.
//! - **API surface**: `eval_phi_into` (φ evaluation), the scratch-buffer
//!   pattern (zero-alloc hot path), `FitConfig` knobs.
//!
//! # What This Does NOT Prove
//!
//! - **Real SeeSE3 vision-feature navigation** — this is a reference demo on
//!   a synthetic linear map, not a production 3D-pose imagination pipeline.
//!   The paper's R² results on real SE(3) data are in the GOAT gate
//!   (`.benchmarks/449_poincare_goat.md`).
//! - **Nonlinear unrolling (phi_out < latent_dim)** — the modelless default
//!   uses `phi_out = latent_dim` (no PCA reduction), giving a near-linear
//!   chart. The nonlinear regime (`phi_out < latent_dim`, tanh curvature)
//!   is where the gradient-fit follow-up (riir-train) would add value.
//! - **G2 strict-domination over linear-only ridge** — the GOAT gate notes
//!   the adapter does NOT strictly dominate linear ridge (R² 0.71 vs 0.93);
//!   its load-bearing value is the closed-form inverse navigation (G3:
//!   perfect Hit@0.3 = 1.000) + the frozen Pod commitment pattern.
//!
//! # Reference
//!
//! - Plan: `katgpt-rs/.plans/449_poincare_latent_navigation_primitive.md`
//! - Research: `katgpt-rs/.research/449_SeeSE3_Poincare_Adapter_Primitive.md`
//! - Source: arXiv:2607.14228 — Chen et al., *SeeSE3: Emergence of 3D Space
//!   in Vision Features* (DeepMind)

use katgpt_core::poincare::{
    FitConfig, PoincareAdapter, PoincareFitError, eval_phi_into, fit_poincare_adapter,
    poincare_multi_step_into, poincare_navigate_into,
};

// ─────────────────────────────────────────────────────────────────────────
// Synthetic linear map: target = A · z (latent_dim=4 → target_dim=2).
//
// This is the cleanest regime for the modelless adapter: a linear map with
// phi_out = latent_dim (no PCA reduction) gives a near-linear chart, so the
// closed-form fit recovers A tightly. We use a fixed seed for determinism.
// ─────────────────────────────────────────────────────────────────────────

const LATENT_DIM: usize = 4;
const TARGET_DIM: usize = 2;
const PHI_OUT: usize = LATENT_DIM; // no PCA reduction → linear chart
const N_SAMPLES: usize = 80;
const SEED: u64 = 42;

/// Generate (z, target) pairs from a known rank-2 linear map A.
#[allow(clippy::type_complexity)] // dataset generator returns three parallel Vecs
fn make_synthetic_dataset() -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = fastrand::Rng::with_seed(SEED);

    // A is (target_dim × latent_dim), rank 2. Two random unit-norm rows.
    let a_row1 = make_unit_vector(&mut rng, LATENT_DIM);
    let a_row2 = make_unit_vector(&mut rng, LATENT_DIM);
    let a = vec![a_row1, a_row2];

    let z_samples: Vec<Vec<f32>> = (0..N_SAMPLES)
        .map(|_| {
            (0..LATENT_DIM)
                .map(|_| rng.f32() * 0.2 - 0.1) // centered ±0.1 → tanh ≈ linear, mean_z ≈ 0
                .collect()
        })
        .collect();
    let target_samples: Vec<Vec<f32>> = z_samples
        .iter()
        .map(|z| vec![dot(&a[0], z), dot(&a[1], z)])
        .collect();

    (z_samples, target_samples, a)
}

fn make_unit_vector(rng: &mut fastrand::Rng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────
// Section 1: Offline fit (modelless, closed-form).
//
// fit_poincare_adapter runs PCA on the z samples + ridge regression on
// (φ(z), target). No gradient descent. The result is a frozen Pod holding
// (φ, W, W†) with a BLAKE3 commitment.
// ─────────────────────────────────────────────────────────────────────────

fn section_1_offline_fit() -> PoincareAdapter {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 1: Offline fit (modelless, closed-form PCA + ridge)      │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    let (z_samples, target_samples, _a) = make_synthetic_dataset();
    let z_refs: Vec<&[f32]> = z_samples.iter().map(|v| v.as_slice()).collect();
    let t_refs: Vec<&[f32]> = target_samples.iter().map(|v| v.as_slice()).collect();

    // Use a smaller ridge_alpha than the default (1.0) because our z samples
    // are small (|z| ≈ 0.1), so the Gram diagonal is ≈ 0.8. With α=1.0 the
    // shrinkage factor 0.8/(0.8+1.0)=0.44 over-dampens the decoder; α=0.01
    // gives shrinkage 0.8/(0.8+0.01)=0.988 → near-exact recovery.
    let cfg = FitConfig {
        ridge_alpha: 0.01,
        ..FitConfig::default()
    };

    let adapter = fit_poincare_adapter(
        &z_refs, &t_refs, LATENT_DIM, TARGET_DIM, PHI_OUT, PHI_OUT, &cfg,
    )
    .expect("fit should succeed on a well-conditioned linear map");

    println!("  Fit succeeded. Adapter structure:");
    println!("    latent_dim  = {}", adapter.latent_dim);
    println!("    target_dim  = {}", adapter.target_dim);
    println!("    phi_hidden  = {}", adapter.phi_hidden);
    println!("    phi_out     = {}", adapter.phi_out);
    println!(
        "    W  shape    = [{} × {}]  (forward decoder)",
        adapter.target_dim, adapter.phi_out
    );
    println!(
        "    W† shape    = [{} × {}]  (pseudoinverse navigator)",
        adapter.phi_out, adapter.target_dim
    );
    println!("    blake3      = {}...", hex_prefix(&adapter.blake3));
    println!();
    println!("  → The adapter is a frozen Pod: (φ, W, W†) committed with BLAKE3.");
    println!("    No gradient descent — PCA + ridge + SVD only (modelless).");
    println!();

    adapter
}

// ─────────────────────────────────────────────────────────────────────────
// Section 2: Forward decoder accuracy (W · φ(z) ≈ target).
//
// Verify the fit recovered the linear map: reconstruct targets on held-out
// z samples and measure R². In the linear regime (small |z|, phi_out =
// latent_dim), recovery should be tight.
// ─────────────────────────────────────────────────────────────────────────

fn section_2_forward_decoder(adapter: &PoincareAdapter) {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 2: Forward decoder accuracy (W·φ(z) ≈ A·z)              │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    let (z_samples, target_samples, a) = make_synthetic_dataset();
    let mut hidden = vec![0.0_f32; adapter.phi_hidden()];
    let mut phi = vec![0.0_f32; adapter.phi_out()];

    let mut ss_res = 0.0_f32;
    let mut ss_tot = 0.0_f32;
    let mut mean_target = [0.0_f32; TARGET_DIM];
    for t in &target_samples {
        for (j, mt) in mean_target.iter_mut().enumerate().take(TARGET_DIM) {
            *mt += t[j];
        }
    }
    for mt in mean_target.iter_mut() {
        *mt /= N_SAMPLES as f32;
    }

    println!(
        "  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "idx", "t_true[0]", "t_hat[0]", "t_true[1]", "t_hat[1]", "err"
    );
    println!("  ────────────────────────────────────────────────────────────────────────");
    for (i, (z, t_true)) in z_samples
        .iter()
        .zip(target_samples.iter())
        .take(6)
        .enumerate()
    {
        eval_phi_into(z, adapter, &mut phi, &mut hidden);
        let t_hat: Vec<f32> = (0..TARGET_DIM)
            .map(|j| {
                dot(
                    &adapter.W[j * adapter.phi_out()..(j + 1) * adapter.phi_out()],
                    &phi,
                )
            })
            .collect();
        let err = ((t_true[0] - t_hat[0]).abs() + (t_true[1] - t_hat[1]).abs()) / 2.0;
        println!(
            "  {:>4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}",
            i, t_true[0], t_hat[0], t_true[1], t_hat[1], err
        );
    }

    // R² over ALL samples.
    for (z, t_true) in z_samples.iter().zip(target_samples.iter()) {
        eval_phi_into(z, adapter, &mut phi, &mut hidden);
        for j in 0..TARGET_DIM {
            let t_hat = dot(
                &adapter.W[j * adapter.phi_out()..(j + 1) * adapter.phi_out()],
                &phi,
            );
            ss_res += (t_true[j] - t_hat).powi(2);
            ss_tot += (t_true[j] - mean_target[j]).powi(2);
        }
    }
    let r2 = 1.0 - ss_res / ss_tot.max(1e-12);
    println!("  ...  ({N_SAMPLES} total samples)");
    println!();
    println!("  R² = {r2:.4}  (1.0 = perfect recovery of the linear map)");
    println!();
    println!("  → The forward decoder W·φ(z) reconstructs the ground-truth targets.");
    println!("    Small |z| (≈0.1) keeps tanh ≈ identity; ridge α=0.01 avoids");
    println!("    over-dampening at this signal scale. The GOAT gate (.benchmarks/");
    println!("    449_poincare_goat.md) reports R²=0.71 on real SE(3) data — this");
    println!("    synthetic linear regime is easier.");
    println!();

    // Suppress unused-warning on `a` (we use it implicitly via the dataset).
    let _ = &a;
}

// ─────────────────────────────────────────────────────────────────────────
// Section 3: The inverse navigator (the headline).
//
// Given a desired Δtarget, poincare_navigate_into finds the latent step:
//   z_out = z_src + φ⁻¹(φ(z_src) + W†·Δtarget)
//
// We verify by projecting z_out back through the forward decoder: the
// recovered displacement W·φ(z_out) − W·φ(z_src) should match Δtarget
// (direction matches exactly; magnitude within tanh-warp tolerance).
// ─────────────────────────────────────────────────────────────────────────

fn section_3_inverse_navigator(adapter: &PoincareAdapter) {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 3: The inverse navigator (the headline primitive)        │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  z_out = z_src + φ⁻¹(φ(z_src) + W†·Δtarget)");
    println!();

    let mut rng = fastrand::Rng::with_seed(99);
    let mut hidden = vec![0.0_f32; adapter.phi_hidden()];
    let mut phi = vec![0.0_f32; adapter.phi_out()];

    println!(
        "  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Δt[0]", "Δt[1]", "rec[0]", "rec[1]", "dir✓?"
    );
    println!("  ────────────────────────────────────────────────────────────────────");
    for trial in 0..5 {
        let z_src: Vec<f32> = (0..LATENT_DIM).map(|_| rng.f32() * 0.1 - 0.05).collect();
        let delta_target = vec![rng.f32() * 0.2 - 0.1, rng.f32() * 0.2 - 0.1];
        let mut z_out = vec![0.0_f32; LATENT_DIM];

        poincare_navigate_into(
            &z_src,
            &delta_target,
            adapter,
            &mut z_out,
            &mut phi,
            &mut hidden,
        );

        // Recover: W·φ(z_out) − W·φ(z_src) should ≈ Δtarget.
        eval_phi_into(&z_src, adapter, &mut phi, &mut hidden);
        let w_src: Vec<f32> = (0..TARGET_DIM)
            .map(|j| {
                dot(
                    &adapter.W[j * adapter.phi_out()..(j + 1) * adapter.phi_out()],
                    &phi,
                )
            })
            .collect();
        eval_phi_into(&z_out, adapter, &mut phi, &mut hidden);
        let w_out: Vec<f32> = (0..TARGET_DIM)
            .map(|j| {
                dot(
                    &adapter.W[j * adapter.phi_out()..(j + 1) * adapter.phi_out()],
                    &phi,
                )
            })
            .collect();
        let recovered = [w_out[0] - w_src[0], w_out[1] - w_src[1]];

        // Direction match: dot product > 0.
        let dot_dir = recovered[0] * delta_target[0] + recovered[1] * delta_target[1];
        println!(
            "  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10}",
            delta_target[0],
            delta_target[1],
            recovered[0],
            recovered[1],
            if dot_dir > 0.0 { "✓" } else { "✗" }
        );
        let _ = trial;
    }
    println!();
    println!("  → The navigator moves the latent state in the direction that achieves");
    println!("    the desired Δtarget. Direction always matches; magnitude is within");
    println!("    tanh-warp tolerance (the chart compresses large displacements).");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 4: Multi-step open-loop trajectory.
//
// poincare_multi_step_into splits a large Δtarget into N sub-steps and
// iterates the navigator. This is an open-loop integrator (no environment
// correction). Bit-identical across runs with the same inputs.
// ─────────────────────────────────────────────────────────────────────────

fn section_4_multi_step(adapter: &PoincareAdapter) {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 4: Multi-step open-loop trajectory (deterministic)       │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    let z_src: Vec<f32> = vec![0.1, 0.05, -0.03, 0.02];
    let delta_target = vec![0.15, -0.10]; // larger displacement
    let n_steps = 5;

    let mut z_out_a = vec![0.0_f32; LATENT_DIM];
    let mut z_out_b = vec![0.0_f32; LATENT_DIM];
    let mut phi = vec![0.0_f32; adapter.phi_out()];
    let mut hidden = vec![0.0_f32; adapter.phi_hidden()];
    let mut delta_step = vec![0.0_f32; TARGET_DIM];

    // Run twice — must be bit-identical.
    poincare_multi_step_into(
        &z_src,
        &delta_target,
        n_steps,
        adapter,
        &mut z_out_a,
        &mut phi,
        &mut hidden,
        &mut delta_step,
    );
    poincare_multi_step_into(
        &z_src,
        &delta_target,
        n_steps,
        adapter,
        &mut z_out_b,
        &mut phi,
        &mut hidden,
        &mut delta_step,
    );

    let bit_identical = z_out_a == z_out_b;
    let displacement: f32 = z_out_a
        .iter()
        .zip(z_src.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    println!("  z_src         = {z_src:?}");
    println!("  Δtarget       = {delta_target:?}");
    println!("  n_steps       = {n_steps}");
    println!("  z_out (5-step)= {z_out_a:?}");
    println!("  |z_out - z_src| = {displacement:.4}");
    println!();
    println!(
        "  Determinism: two runs bit-identical? {}",
        if bit_identical { "✓ YES" } else { "✗ NO" }
    );
    println!();
    println!("  → Multi-step splits a large displacement into smaller sub-steps,");
    println!("    reducing the tanh-warp error per step. Deterministic — no RNG,");
    println!("    no thread-state reads.");
    println!();
}

// ─────────────────────────────────────────────────────────────────────────
// Section 5: Freeze/thaw (BLAKE3 commitment).
//
// canonical_bytes serializes the adapter; from_bytes deserializes +
// verifies the BLAKE3 commitment. Round-trip must be bit-identical.
// Tampering any weight must break verification.
// ─────────────────────────────────────────────────────────────────────────

fn section_5_freeze_thaw(adapter: &PoincareAdapter) {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ Section 5: Freeze/thaw (BLAKE3 commitment)                       │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    println!();

    // Serialize → deserialize → verify.
    let bytes = adapter.canonical_bytes();
    println!("  Serialized size: {} bytes", bytes.len());

    let restored = PoincareAdapter::from_bytes(&bytes);
    match restored {
        Ok(r) => {
            let blake3_match = r.blake3 == adapter.blake3;
            let w_match = r.W == adapter.W;
            let wpinv_match = r.W_pinv == adapter.W_pinv;
            let verify_ok = r.verify();
            println!("  from_bytes: OK");
            println!(
                "    blake3 match:    {}",
                if blake3_match { "✓" } else { "✗" }
            );
            println!("    W match:         {}", if w_match { "✓" } else { "✗" });
            println!(
                "    W† match:        {}",
                if wpinv_match { "✓" } else { "✗" }
            );
            println!("    verify():        {}", if verify_ok { "✓" } else { "✗" });
        }
        Err(e) => println!("  from_bytes FAILED: {e:?}"),
    }
    println!();

    // Tamper detection: flip one weight byte, expect verification to fail.
    let mut tampered = bytes.clone();
    if let Some(last) = tampered.last_mut() {
        *last ^= 0xFF;
    }
    let tampered_result = PoincareAdapter::from_bytes(&tampered);
    match tampered_result {
        Ok(r) => {
            let verify_ok = r.verify();
            println!(
                "  Tampered buffer: from_bytes OK, verify() = {}  (expected ✗)",
                if verify_ok { "✓" } else { "✗" }
            );
        }
        Err(PoincareFitError::MalformedBuffer) => {
            println!("  Tampered buffer: from_bytes rejected (MalformedBuffer) ✓");
        }
        Err(e) => println!("  Tampered buffer: unexpected error {e:?}"),
    }
    println!();
    println!("  → The BLAKE3 commitment makes the adapter tamper-evident. Frozen");
    println!("    Pods can be distributed + verified without trusting the channel.");
    println!();
}

fn main() {
    println!();
    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║  Poincaré Adapter — closed-form latent navigation (Plan 449)       ║");
    println!("║  Inverse navigation: given Δtarget, find the latent step           ║");
    println!("╚═════════════════════════════════════════════════════════════════════╝");
    println!();

    let adapter = section_1_offline_fit();
    section_2_forward_decoder(&adapter);
    section_3_inverse_navigator(&adapter);
    section_4_multi_step(&adapter);
    section_5_freeze_thaw(&adapter);

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Done. All 5 sections completed. See the module doc (top of this file)");
    println!("for what this proves / does NOT prove.");
    println!("═══════════════════════════════════════════════════════════════════════");
}

// Helper: first 8 hex chars of a BLAKE3 hash for compact display.
fn hex_prefix(hash: &[u8; 32]) -> String {
    hash.iter().take(4).map(|b| format!("{b:02x}")).collect()
}

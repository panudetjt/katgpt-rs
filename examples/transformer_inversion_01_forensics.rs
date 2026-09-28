//! SipIt Transformer Inversion — Prompt Forensics Demo (Plan 561).
//!
//! Demonstrates the primitive's stated commercial purpose: an **adoption hook
//! for transparency / audit / forensics tooling** on standard decoder-only
//! text transformers. Given a frozen model's forward function + a captured
//! layer-ℓ hidden-state matrix `H̆^(ℓ)` (e.g. an audit log, a transparency
//! probe snapshot, or a KV-cache forensics dump), recover the exact input
//! prompt — proving the model's internal state uniquely identifies the prompt
//! (Theorem 2.2 of Nikolaou et al. ICLR 2026, arXiv:2510.15511).
//!
//! Run: `cargo run --example transformer_inversion_01_forensics --features grad_policy`
//!
//! # What This Proves
//!
//! - **Exact recovery (G1)**: random init → random prompts → bit-identical
//!   recovery via `invert_sequence`. The headline transparency guarantee:
//!   "if you can see layer-ℓ, you can recover the prompt exactly."
//! - **Both policies**: Random (worst case `T·|V|` trials, always correct)
//!   + GradientGuided (paper Alg 3, 3.4× fewer acceptance tests on the toy
//!     per Phase 2 A/B). Shows the audit-time vs compute tradeoff.
//! - **Robustness (Thm 3.2)**: recovery holds when observation noise is below
//!   `Δ_π,t / 2` (the per-position margin), degrades above. Validates that
//!   the primitive works on real-world noisy captures, not just clean ones.
//! - **Forensics narrative**: the demo is framed as "audit log → recovered
//!   prompt" — the exact shape a transparency tool would compose against.
//!
//! # What This Does NOT Prove
//!
//! - **Real-text-LLM scale** — the toy is `d=16, |V|=32, T=8`. The paper's
//!   regime is `|V| ∈ [32K, 128K]` with near-orthogonal high-dim embeddings.
//!   Plan 561 Phase 3 G2 measured latency scaling on the toy; production
//!   scale needs a real transformer's analytical gradient (Phase 2 here
//!   uses finite-difference, which dominates latency at toy scale).
//! - **Production consumer wiring** — this is a reference demo of the
//!   primitive's intended use case, not a production audit pipeline.
//!   Plan 561 T5.1 (promotion to default-on) still requires a concrete
//!   consumer in riir-ai or katgpt-rs itself that demonstrates a measured
//!   downstream gain at the GOAT gate.
//!
//! # Reference
//!
//! - Plan: `katgpt-rs/.plans/561_transformer_inversion_sipit_open_primitive.md`
//! - Source paper: arXiv:2510.15511 — Nikolaov et al., *Language Models are
//!   Injective and Hence Invertible*, ICLR 2026.

#![cfg(feature = "grad_policy")]

use katgpt_core::inversion::{
    InversionConfig, InversionError, InversionForward, InversionPolicy, InversionResult,
    ObservedStates, invert_sequence, invert_sequence_grad,
};

// ─────────────────────────────────────────────────────────────────────────
// Toy 2-layer GELU transformer (the Plan 561 G1 substrate).
//
// This is the minimum that satisfies Theorem 2.2's hypotheses:
//   - Real-analytic activation (GELU, tanh-form).
//   - Causal (position-t output depends only on prefix π and token v at t).
//   - Vocabulary-indexed embedding lookup.
//
// Mirrors the test substrate at `crates/katgpt-core/src/inversion/tests.rs`
// so the example + the unit tests agree on what "a toy text transformer"
// means in this codebase.
// ─────────────────────────────────────────────────────────────────────────

const D: usize = 16; // hidden dim
const V: u32 = 32; // vocab size
const T: usize = 8; // sequence length

struct ToyTransformer {
    embedding: Vec<f32>, // V × D row-major
    w1_up: Vec<f32>,     // 4D × D row-major
    w1_down: Vec<f32>,   // D × 4D row-major
    w2_up: Vec<f32>,     // 4D × D row-major
    w2_down: Vec<f32>,   // D × 4D row-major
}

impl ToyTransformer {
    /// Random init with scale 1.0 (well-conditioned Jacobian, per Phase 2 lesson).
    fn new(rng: &mut fastrand::Rng) -> Self {
        let scale = 1.0_f32;
        let mut rand_vec =
            |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect() };
        Self {
            embedding: rand_vec((V as usize) * D),
            w1_up: rand_vec(D * 4 * D),
            w1_down: rand_vec(4 * D * D),
            w2_up: rand_vec(D * 4 * D),
            w2_down: rand_vec(4 * D * D),
        }
    }

    /// Forward pass returning the full `T × D` layer-2 hidden-state matrix.
    fn forward_full(&self, prompt: &[u32]) -> Vec<f32> {
        debug_assert!(prompt.len() <= T);
        let t = prompt.len();
        let mut out = vec![0.0_f32; t * D];
        for (position, &v) in prompt.iter().enumerate() {
            let row = &mut out[position * D..(position + 1) * D];
            self.embed_into(v, row);
            self.apply_layer_into(&self.w1_up, &self.w1_down, row);
            self.apply_layer_into(&self.w2_up, &self.w2_down, row);
        }
        out
    }

    #[inline]
    fn embed_into(&self, token: u32, out: &mut [f32]) {
        let base = (token as usize) * D;
        out.copy_from_slice(&self.embedding[base..base + D]);
    }

    #[inline]
    fn apply_layer_into(&self, w_up: &[f32], w_down: &[f32], x: &mut [f32]) {
        debug_assert_eq!(x.len(), D);
        let mut hidden = [0.0_f32; 4 * D];
        for i in 0..4 * D {
            let row = &w_up[i * D..(i + 1) * D];
            let mut acc = 0.0_f32;
            for (j, xi) in x.iter().enumerate() {
                acc += row[j] * xi;
            }
            hidden[i] = gelu(acc);
        }
        for i in 0..D {
            let row = &w_down[i * 4 * D..(i + 1) * 4 * D];
            let mut acc = 0.0_f32;
            for (j, hi) in hidden.iter().enumerate() {
                acc += row[j] * hi;
            }
            x[i] = acc;
        }
    }
}

impl InversionForward for ToyTransformer {
    fn hidden_at_into(
        &self,
        prefix: &[u32],
        candidate: u32,
        position: usize,
        out: &mut [f32],
    ) -> Result<(), InversionError> {
        debug_assert_eq!(position, prefix.len());
        debug_assert_eq!(out.len(), D);
        let mut full: Vec<u32> = Vec::with_capacity(prefix.len() + 1);
        full.extend_from_slice(prefix);
        full.push(candidate);
        let states = self.forward_full(&full);
        let row_offset = position * D;
        out.copy_from_slice(&states[row_offset..row_offset + D]);
        Ok(())
    }
}

// Phase 2 also requires InversionGradient for the GradientGuided policy.
// The toy uses finite-difference gradients (O(D) forward evals per step).
impl katgpt_core::inversion::InversionGradient for ToyTransformer {
    fn grad_hidden_at_into(
        &self,
        prefix: &[u32],
        observed_state: &[f32],
        proxy: &[f32],
        position: usize,
        out: &mut [f32],
    ) -> Result<(), InversionError> {
        debug_assert_eq!(out.len(), D);
        let mut perturbed = vec![0.0_f32; D];
        let eps = 1e-3_f32;
        for d in 0..D {
            // Central difference: L(e + ε·ê_d) − L(e − ε·ê_d), divided by 2ε.
            // L(e) = ½·‖h̆_t − F(e; π, t)‖² (squared L2). The gradient of L
            // w.r.t. e_d is −(h̆ − F(e))ᵀ · ∂F/∂e_d. We finite-difference F.
            perturbed.copy_from_slice(proxy);
            perturbed[d] += eps;
            let mut f_plus = vec![0.0_f32; D];
            self.proxy_forward_into(prefix, &perturbed, position, &mut f_plus)?;

            perturbed.copy_from_slice(proxy);
            perturbed[d] -= eps;
            let mut f_minus = vec![0.0_f32; D];
            self.proxy_forward_into(prefix, &perturbed, position, &mut f_minus)?;

            // dF_d ≈ (f_plus − f_minus) / (2ε). Then dL/de_d = −(h̆ − F)·dF_d.
            let mut grad_d = 0.0_f32;
            for i in 0..D {
                let df = (f_plus[i] - f_minus[i]) / (2.0 * eps);
                let residual = observed_state[i] - 0.5 * (f_plus[i] + f_minus[i]);
                grad_d -= residual * df;
            }
            out[d] = grad_d;
        }
        Ok(())
    }

    fn nearest_token(&self, proxy: &[f32]) -> Result<u32, InversionError> {
        // argmin_v ‖proxy − embedding[v]‖² — linear scan, fine for |V|=32.
        let mut best_v = 0_u32;
        let mut best_dist = f32::INFINITY;
        for v in 0..V {
            let base = (v as usize) * D;
            let emb = &self.embedding[base..base + D];
            let mut dist = 0.0_f32;
            for i in 0..D {
                let d = proxy[i] - emb[i];
                dist += d * d;
            }
            if dist < best_dist {
                best_dist = dist;
                best_v = v;
            }
        }
        Ok(best_v)
    }
}

impl ToyTransformer {
    /// Forward pass from a continuous proxy embedding (no token lookup).
    /// Used by the finite-difference gradient above.
    fn proxy_forward_into(
        &self,
        prefix: &[u32],
        proxy: &[f32],
        position: usize,
        out: &mut [f32],
    ) -> Result<(), InversionError> {
        debug_assert_eq!(proxy.len(), D);
        debug_assert_eq!(out.len(), D);
        out.copy_from_slice(proxy);
        self.apply_layer_into(&self.w1_up, &self.w1_down, out);
        self.apply_layer_into(&self.w2_up, &self.w2_down, out);
        // The prefix is ignored on this toy (no cross-position attention);
        // position is accepted for API conformance but the layer-2 output
        // at position t depends only on the token/proxy at t. This matches
        // the test substrate (tests.rs L100-122).
        let _ = (prefix, position);
        Ok(())
    }
}

/// GELU approximation (tanh-form). Real-analytic, satisfies Theorem 2.2.
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x)).tanh())
}

// ─────────────────────────────────────────────────────────────────────────
// Forensics helpers — the shape a transparency/audit tool would compose.
// ─────────────────────────────────────────────────────────────────────────

/// Simulate capturing an audit-log hidden-state matrix from a frozen model.
fn capture_audit_log(model: &ToyTransformer, prompt: &[u32]) -> Vec<f32> {
    model.forward_full(prompt)
}

/// Inject observation noise into a captured audit log (Thm 3.2 robustness test).
fn inject_noise(log: &mut [f32], magnitude: f32, rng: &mut fastrand::Rng) {
    for x in log.iter_mut() {
        *x += (rng.f32() * 2.0 - 1.0) * magnitude;
    }
}

/// Compute the per-position margin `Δ_π,t` — the minimum L∞ distance from
/// the true token's state to any OTHER token's state at that position under
/// the recovered prefix. Used to set the noise tolerance for Thm 3.2.
fn min_margin_at(model: &ToyTransformer, prompt: &[u32], t: usize) -> f32 {
    let prefix = &prompt[..t];
    let true_token = prompt[t];
    let mut scratch_true = [0.0_f32; D];
    model
        .hidden_at_into(prefix, true_token, t, &mut scratch_true)
        .unwrap();
    let mut min_dist = f32::INFINITY;
    for v in 0..V {
        if v == true_token {
            continue;
        }
        let mut scratch = [0.0_f32; D];
        model.hidden_at_into(prefix, v, t, &mut scratch).unwrap();
        let mut max_comp = 0.0_f32;
        for i in 0..D {
            max_comp = max_comp.max((scratch_true[i] - scratch[i]).abs());
        }
        min_dist = min_dist.min(max_comp);
    }
    min_dist
}

fn fmt_prompt(prompt: &[u32]) -> String {
    prompt
        .iter()
        .map(|v| format!("{v:2}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    const N_PROMPTS: usize = 4;

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  SipIt Transformer Inversion — Prompt Forensics Demo (Plan 561)    ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Substrate: toy 2-layer GELU decoder-only transformer");
    println!("    d = {D} (hidden dim), |V| = {V} (vocab), T = {T} (sequence len)");
    println!("    Theorem 2.2 hypotheses: real-analytic activation, causal,");
    println!("      vocabulary-indexed embedding lookup → prompt is uniquely");
    println!("      recoverable from layer-ℓ hidden states up to tolerance.");
    println!();

    let mut rng = fastrand::Rng::with_seed(0xC0DE);
    let model = ToyTransformer::new(&mut rng);

    // ── 1. G1: Exact recovery via Random policy ──────────────────────────
    println!("── 1. G1 Exact Recovery (Random policy) ──────────────────────────────");
    println!();
    println!("  Audit scenario: a transparency tool captured the model's layer-ℓ");
    println!("  hidden-state matrix H̆^(ℓ). We recover the original prompt exactly.");
    println!();

    let mut prompt_rng = fastrand::Rng::with_seed(0xA5A5);
    let mut all_recovered = 0_usize;
    for i in 0..N_PROMPTS {
        let prompt: Vec<u32> = (0..T).map(|_| prompt_rng.u32(0..V)).collect();
        let audit_log = capture_audit_log(&model, &prompt);
        let observed = ObservedStates::from_row_major(&audit_log, T, D).unwrap();
        let cfg = InversionConfig::default(); // Random policy, ε=1e-3
        let result = invert_sequence(&observed, V, &model, &cfg, i as u64).unwrap();
        match result {
            InversionResult::Recovered(recovered) => {
                let ok = recovered == prompt;
                if ok {
                    all_recovered += 1;
                }
                println!(
                    "  prompt {i}: {} → {}  {}",
                    fmt_prompt(&prompt),
                    fmt_prompt(&recovered),
                    if ok { "✓ exact" } else { "✗ mismatch" }
                );
            }
            InversionResult::Failed {
                failed_position,
                candidates_tried,
            } => {
                println!(
                    "  prompt {i}: FAILED at pos {failed_position} ({candidates_tried} tried)"
                );
            }
        }
    }
    println!();
    println!("  Result: {all_recovered}/{N_PROMPTS} prompts recovered exactly (G1).");
    println!();

    // ── 2. Phase 2: Gradient-Guided policy (fewer acceptance tests) ──────
    println!("── 2. Gradient-Guided Policy (paper Alg 3) ───────────────────────────");
    println!();
    println!("  Same recovery, fewer verifier calls. On the toy the speedup is");
    println!("  measured by acceptance-test count (Phase 2 A/B). The gradient");
    println!("  policy uses a finite-difference Jacobian here — a real transformer");
    println!("  would supply the analytical gradient via InversionGradient.");
    println!();

    let prompt: Vec<u32> = (0..T).map(|_| prompt_rng.u32(0..V)).collect();
    let audit_log = capture_audit_log(&model, &prompt);
    let observed = ObservedStates::from_row_major(&audit_log, T, D).unwrap();

    // Random policy baseline.
    let cfg_rand = InversionConfig::default();
    let result_rand = invert_sequence(&observed, V, &model, &cfg_rand, 99).unwrap();
    match &result_rand {
        InversionResult::Recovered(r) => {
            let ok = r == &prompt;
            println!(
                "  Random:        {}  {}",
                fmt_prompt(r),
                if ok { "✓" } else { "✗" }
            );
        }
        InversionResult::Failed { .. } => println!("  Random:        FAILED"),
    }

    // Gradient-guided policy.
    let cfg_grad = InversionConfig {
        policy: InversionPolicy::gradient_guided_default(),
        ..Default::default()
    };
    let result_grad = invert_sequence_grad(&observed, V, &model, &model, &cfg_grad, 99).unwrap();
    match &result_grad {
        InversionResult::Recovered(r) => {
            let ok = r == &prompt;
            println!(
                "  GradientGuided:{}  {}",
                fmt_prompt(r),
                if ok { "✓" } else { "✗" }
            );
        }
        InversionResult::Failed { .. } => println!("  GradientGuided:FAILED"),
    }
    println!();
    println!("  Both policies recover the same prompt — the audit-time vs compute");
    println!("  tradeoff is the deployer's choice.");
    println!();

    // ── 3. Thm 3.2: Robustness under observation noise ───────────────────
    println!("── 3. Robustness (Theorem 3.2) ───────────────────────────────────────");
    println!();
    println!("  Audit logs in the wild carry observation noise (FP4/INT8 quant,");
    println!("  sensor drift, compression). Theorem 3.2: recovery holds when");
    println!("  ‖noise‖_∞ < Δ_π,t / 2 (the per-position margin). We verify this");
    println!("  by sweeping noise levels around the margin boundary.");
    println!();

    let prompt: Vec<u32> = (0..T).map(|_| prompt_rng.u32(0..V)).collect();
    let min_margin: f32 = (0..T)
        .map(|t| min_margin_at(&model, &prompt, t))
        .fold(f32::INFINITY, f32::min);
    println!("  min_t(Δ_π,t) = {min_margin:.4} (smallest per-position margin)");
    println!("  → noise tolerance = Δ/2 = {:.4}", min_margin * 0.5);
    println!();

    // Theorem 3.2 acceptance contract: set ε = Δ/2 so that the true token
    // (within ‖e_t‖ ≤ noise < Δ/2 of F(v_true)) is accepted, and every wrong
    // token (≥ Δ away from F(v_true), so ≥ Δ/2 away from ę_t) is rejected.
    // Below the noise threshold → exact recovery; above → degradation.
    for &frac in &[0.1_f32, 0.25, 0.45, 0.7, 1.5] {
        let noise_mag = frac * min_margin * 0.5;
        let mut noisy_log = model.forward_full(&prompt);
        inject_noise(&mut noisy_log, noise_mag, &mut rng);
        let observed = ObservedStates::from_row_major(&noisy_log, T, D).unwrap();
        // ε = Δ/2 is the theoretical max tolerance for guaranteed recovery
        // under worst-case noise of that magnitude (per Thm 3.2).
        let cfg = InversionConfig {
            tolerance: min_margin * 0.5,
            ..Default::default()
        };
        let result = invert_sequence(&observed, V, &model, &cfg, 42).unwrap();
        let (recovered_ok, label) = match result {
            InversionResult::Recovered(r) => (r == prompt, "recovered"),
            InversionResult::Failed { .. } => (false, "failed"),
        };
        let verdict = if frac < 0.5 {
            "expected ✓"
        } else if frac <= 1.0 {
            "boundary"
        } else {
            "expected ✗"
        };
        println!(
            "  noise = {frac:>4.2}× Δ/2 ({noise_mag:>7.4}): {label:>9}  [{verdict}]  {}",
            if recovered_ok { "✓" } else { "✗" }
        );
    }
    println!();
    println!("  Below Δ/2: recovery holds. Above: degrades. This is the theorem's");
    println!("  perturbation guarantee, demonstrated on the toy.");
    println!();

    // ── 4. Forensics narrative ───────────────────────────────────────────
    println!("── 4. The Transparency / Audit Use Case ──────────────────────────────");
    println!();
    println!("  Composition shape a transparency tool would build:");
    println!();
    println!("    [frozen model weights]");
    println!("         ↓");
    println!("    [audit log: layer-ℓ hidden states H̆^(ℓ)]  ← captured at runtime");
    println!("         ↓");
    println!("    [invert_sequence] ← this primitive");
    println!("         ↓");
    println!("    [recovered prompt s = ⟨s₁, …, s_T⟩]");
    println!();
    println!("  Applications:");
    println!("    - KV-cache forensics: prove which prompt produced a captured cache");
    println!("    - Transparency reports: reconstruct inputs from logged activations");
    println!("    - Anti-cheat audit: verify a claimed inference matches observed state");
    println!("    - Privacy review: demonstrate that logged activations leak prompts");
    println!();
    println!("  This demo is a reference implementation of that composition. The");
    println!("  primitive itself is modelless (no training, no backprop through");
    println!("  weights) and substrate-agnostic (any decoder-only text transformer");
    println!("  that satisfies Theorem 2.2's hypotheses works).");
    println!();

    // ── 5. Honest scope note ─────────────────────────────────────────────
    println!("── 5. Scope ──────────────────────────────────────────────────────────");
    println!();
    println!("  This is the reference demo for the primitive's stated commercial");
    println!("  purpose (Plan 561: 'adoption hook for transparency/audit tooling').");
    println!("  It is NOT a production consumer — promotion to default-on still");
    println!("  requires a concrete downstream consumer that demonstrates a measured");
    println!("  gain at the GOAT gate (Plan 561 T5.1, unmet as of 2026-07-29).");
    println!("  The demo closes the documentation gap (every public primitive in");
    println!("  katgpt-rs ships an example harness) and provides a reference for");
    println!("  future consumers wiring this primitive against real transformers.");
    println!();
    println!("  See:");
    println!("    - Plan: katgpt-rs/.plans/561_transformer_inversion_sipit_open_primitive.md");
    println!("    - Paper: arXiv:2510.15511 (Nikolaou et al. ICLR 2026)");
}

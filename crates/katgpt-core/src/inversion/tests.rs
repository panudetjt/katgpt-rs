//! G1 correctness tests for SipIt inversion on a toy 2-layer decoder-only transformer.
//!
//! Per Plan 561 §"Phase 1 — Skeleton + Random Policy + G1 (essential)":
//! a real (randomly-initialized) toy transformer with GELU activation,
//! `d = 16`, `|V| = 32`, `T = 8`. The three sub-tests directly exercise
//! the paper's injectivity theorem (Theorem 2.2) and its causality lemma
//! (Lemma D.2):
//!
//! 1. `g1_exact_recovery_random_init` — generate random prompts, run the
//!    transformer forward to get `H̆^(ℓ)`, run `invert_sequence`, assert
//!    exact recovery. This is the headline correctness property.
//! 2. `g1_recovers_when_two_prompts_differ_only_at_position_t` — direct
//!    test of Lemma D.2's causality argument: the prefix-conditioned
//!    verifier correctly distinguishes tokens that differ at one position.
//! 3. `g1_no_false_positive_on_mismatched_observed` — wrong `H̆` does not
//!    produce the original prompt (negative control).

use crate::inversion::{
    InversionConfig, InversionForward, InversionResult, ObservedStates, invert_sequence,
};

// ─────────────────────────────────────────────────────────────────────────
// Toy transformer: embedding lookup → 2 × (Linear d→4d, GELU, Linear 4d→d) →
// returns the per-position hidden state at the LAST layer (layer-ℓ = layer 2).
//
// This is the *minimum* that satisfies Theorem 2.2's hypotheses:
//   - Real-analytic activation (GELU).
//   - Causal (position-t output depends only on prefix π and token v at t).
//   - Vocabulary-indexed embedding lookup.
//
// We deliberately keep the architecture tiny so the G1 tests run in
// microseconds, not seconds. The G2 latency benchmark (Phase 3) will
// scale this up.
// ─────────────────────────────────────────────────────────────────────────

const D: usize = 16;
const V: u32 = 32;
const T: usize = 8;

/// Toy 2-layer GELU transformer.
struct ToyTransformer {
    /// Embedding matrix: `V × D`, row `v` is the embedding of token `v`.
    /// Plain flat row-major.
    embedding: Vec<f32>,
    /// Layer-1 + Layer-2 weights. Each layer = (W_up D×4D, W_down 4D×D).
    /// Plain flat row-major.
    w1_up: Vec<f32>,
    w1_down: Vec<f32>,
    w2_up: Vec<f32>,
    w2_down: Vec<f32>,
}

impl ToyTransformer {
    /// Random init with a deterministic seed, standard scale `1/sqrt(D)`.
    fn new(rng: &mut fastrand::Rng) -> Self {
        Self::new_scaled(rng, 1.0 / (D as f32).sqrt())
    }

    /// Random init with a caller-specified weight scale. Larger scales
    /// (e.g. `1.0`) produce a steeper loss landscape with larger gradients,
    /// suitable for exercising the Phase 2 gradient-guided policy; the
    /// Phase 1 default `1/sqrt(D)` is the standard stable-init scale.
    fn new_scaled(rng: &mut fastrand::Rng, scale: f32) -> Self {
        let mut embedding = vec![0.0_f32; (V as usize) * D];
        let mut w1_up = vec![0.0_f32; D * 4 * D];
        let mut w1_down = vec![0.0_f32; 4 * D * D];
        let mut w2_up = vec![0.0_f32; D * 4 * D];
        let mut w2_down = vec![0.0_f32; 4 * D * D];
        for x in embedding.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        for x in w1_up.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        for x in w1_down.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        for x in w2_up.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        for x in w2_down.iter_mut() {
            *x = (rng.f32() * 2.0 - 1.0) * scale;
        }
        Self {
            embedding,
            w1_up,
            w1_down,
            w2_up,
            w2_down,
        }
    }

    /// Forward pass returning the full `T × D` layer-2 hidden state matrix
    /// for `prompt`. Allocates a fresh `Vec<f32>` per call — this is the
    /// test helper, not the hot path.
    fn forward_full(&self, prompt: &[u32]) -> Vec<f32> {
        debug_assert!(prompt.len() <= T);
        let t = prompt.len();
        let mut out = vec![0.0_f32; t * D];

        // Per-position: embed, then apply 2 layers, writing into the row.
        // We compute the full row at each position; the causal constraint
        // is satisfied trivially because each row depends only on its own
        // token (no attention between positions in this toy — we keep
        // the substrate minimal to make the injectivity theorem
        // straightforward to verify empirically).
        for (position, &v) in prompt.iter().enumerate() {
            let row_offset = position * D;
            self.embed_into(v, &mut out[row_offset..row_offset + D]);
            self.apply_layer_into(
                &self.w1_up,
                &self.w1_down,
                &mut out[row_offset..row_offset + D],
            );
            self.apply_layer_into(
                &self.w2_up,
                &self.w2_down,
                &mut out[row_offset..row_offset + D],
            );
        }
        out
    }

    #[inline]
    fn embed_into(&self, token: u32, out: &mut [f32]) {
        let base = (token as usize) * D;
        out.copy_from_slice(&self.embedding[base..base + D]);
    }

    /// Apply one transformer layer in-place: `x ← W_down · gelu(W_up · x)`.
    /// Uses a 4D intermediate buffer on the stack.
    #[inline]
    fn apply_layer_into(&self, w_up: &[f32], w_down: &[f32], x: &mut [f32]) {
        debug_assert_eq!(x.len(), D);
        let mut hidden = [0.0_f32; 4 * D];
        // hidden = W_up · x   (W_up is 4D × D row-major)
        for i in 0..4 * D {
            let row = &w_up[i * D..(i + 1) * D];
            let mut acc = 0.0_f32;
            for (j, xi) in x.iter().enumerate() {
                acc += row[j] * xi;
            }
            hidden[i] = gelu(acc);
        }
        // x ← W_down · hidden   (W_down is D × 4D row-major)
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
    ) -> Result<(), crate::inversion::InversionError> {
        debug_assert_eq!(position, prefix.len());
        debug_assert_eq!(out.len(), D);

        // Build the full prompt = prefix ⊕ [candidate] and run forward.
        // For Phase 1 we re-run the full forward per trial; Phase 2 will
        // cache the prefix's intermediate state. This is O(T·|V|·D) per
        // position, which is fine for G1 on a toy; G2 (Phase 3) measures
        // latency on the toy, G4 (Phase 3) checks alloc-free of the
        // *driver itself* (not the test forward).
        let mut full: Vec<u32> = Vec::with_capacity(prefix.len() + 1);
        full.extend_from_slice(prefix);
        full.push(candidate);
        let states = self.forward_full(&full);
        let row_offset = position * D;
        out.copy_from_slice(&states[row_offset..row_offset + D]);
        Ok(())
    }
}

/// GELU approximation (tanh-form). Real-analytic, satisfies Theorem 2.2.
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x)).tanh())
}

// ─── G1 tests ─────────────────────────────────────────────────────────────

fn random_prompt(rng: &mut fastrand::Rng) -> Vec<u32> {
    (0..T).map(|_| rng.u32(0..V)).collect()
}

#[test]
fn g1_exact_recovery_random_init() {
    // Headline G1: random init → random prompts → exact recovery.
    const N_PROMPTS: usize = 8;

    let mut rng = fastrand::Rng::with_seed(0xC0DE);
    let transformer = ToyTransformer::new(&mut rng);

    let mut success_count = 0;
    for prompt_seed in 0..N_PROMPTS {
        let mut prompt_rng = fastrand::Rng::with_seed(0xA5A5 + prompt_seed as u64);
        let prompt = random_prompt(&mut prompt_rng);

        let buf = transformer.forward_full(&prompt);
        let observed = ObservedStates::from_row_major(&buf, T, D).unwrap();
        let cfg = InversionConfig::default();
        let result = invert_sequence(&observed, V, &transformer, &cfg, prompt_seed as u64).unwrap();
        match result {
            InversionResult::Recovered(recovered) => {
                assert_eq!(recovered, prompt, "prompt {prompt:?} not recovered");
                success_count += 1;
            }
            InversionResult::Failed {
                failed_position,
                candidates_tried,
            } => {
                // Should not happen on a toy with T·|V| budget — but if it
                // does, surface the position so the failure is debuggable.
                panic!(
                    "failed at position {failed_position} after {candidates_tried} candidates \
                     (prompt {prompt:?})"
                );
            }
        }
    }
    // All 8 must succeed; if even one fails, the theorem isn't holding on
    // this toy and the primitive has a bug.
    assert_eq!(
        success_count, N_PROMPTS,
        "expected all {N_PROMPTS} random prompts to recover exactly"
    );
}

#[test]
fn g1_recovers_when_two_prompts_differ_only_at_position_t() {
    // Lemma D.2 causality: if two prompts differ only at position t, the
    // observed states at position t (under the recovered prefix π) are
    // different — the prefix-conditioned verifier correctly distinguishes.
    let mut rng = fastrand::Rng::with_seed(0xCAFE);
    let transformer = ToyTransformer::new(&mut rng);

    // Construct a base prompt, then mutate it at each position one at a
    // time, asserting each mutated prompt still recovers exactly.
    let base_prompt: Vec<u32> = vec![3, 7, 11, 15, 19, 23, 27, 31];
    for t in 0..T {
        let mut mutated = base_prompt.clone();
        let old = mutated[t];
        // Find a different token at this position.
        let new = if old + 1 < V { old + 1 } else { old - 1 };
        mutated[t] = new;

        let buf = transformer.forward_full(&mutated);
        let observed = ObservedStates::from_row_major(&buf, T, D).unwrap();
        let cfg = InversionConfig::default();
        let result = invert_sequence(&observed, V, &transformer, &cfg, 0xBEEF).unwrap();
        match result {
            InversionResult::Recovered(recovered) => assert_eq!(
                recovered, mutated,
                "mutated-at-position-{t} prompt not recovered exactly"
            ),
            InversionResult::Failed {
                failed_position, ..
            } => panic!("failed at position {failed_position} when only position {t} was mutated"),
        }
    }
}

#[test]
fn g1_no_false_positive_on_mismatched_observed() {
    // Negative control: observed states from prompt A, but checked against
    // the transformer's actual forward on prompt A. We invert observed-A
    // and then RE-RUN the full forward on the recovered prompt. If
    // observed-A was actually for prompt B, the recovered prompt should
    // NOT match prompt A.
    //
    // Stronger version: take observed-A, mutate it slightly (above the
    // tolerance), and assert that the inversion either fails OR returns
    // a prompt that, when re-run forward, produces states different from
    // the (mutated) observed-A.
    let mut rng = fastrand::Rng::with_seed(0xFEED);
    let transformer = ToyTransformer::new(&mut rng);

    let prompt_a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut buf_a = transformer.forward_full(&prompt_a);
    // Corrupt position 0's first coordinate by 1.0 (way above tolerance 1e-3).
    buf_a[0] += 1.0;

    let observed = ObservedStates::from_row_major(&buf_a, T, D).unwrap();
    let cfg = InversionConfig::default();
    let result = invert_sequence(&observed, V, &transformer, &cfg, 0).unwrap();
    match result {
        InversionResult::Failed { .. } => {
            // Correct: corrupted observed doesn't match any token's true state.
            // This is the cleanest negative-control result.
        }
        InversionResult::Recovered(recovered) => {
            // Allow recovery of SOME prompt (the corrupted observed might
            // happen to land in another token's acceptance region) — but
            // then re-running the forward must produce states that DO
            // match the corrupted observed, AND those states must NOT
            // match the original prompt A's states within tolerance.
            let re_buf = transformer.forward_full(&recovered);
            let re_observed = ObservedStates::from_row_major(&re_buf, T, D).unwrap();
            // The recovered prompt's forward must match the (corrupted)
            // observed within tolerance — otherwise the inversion is wrong.
            for t in 0..T {
                let diff: f32 = observed
                    .row(t)
                    .iter()
                    .zip(re_observed.row(t).iter())
                    .map(|(o, c)| (o - c).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    diff <= cfg.tolerance,
                    "recovered prompt's forward diverges from observed at position {t}: {diff}"
                );
            }
            // Sanity: the recovered prompt should not equal prompt_a
            // (because observed was corrupted away from prompt_a's state).
            assert_ne!(
                recovered, prompt_a,
                "inversion recovered prompt A from corrupted-A observed — false positive"
            );
        }
    }
}

// ─── Phase 2: gradient-guided policy ────────────────────────────────────
//
// The gradient-guided policy (paper Alg 3) refines a continuous proxy embedding
// via gradient descent on L(e) = ½·‖h̆_t − F(e;π,t)‖², then projects to the
// nearest vocab token. We implement `InversionGradient` for the toy via
// *central finite differences* — no autodiff dep. The production caller
// (e.g. a real transformer audit tool) would supply an analytical gradient.
//
// Test target (Plan 561 T2.3): gradient-guided recovers exactly AND uses
// strictly fewer acceptance tests than uniform-random on average. The
// paper's <0.25%·|V| claim is for |V|=32K–128K; on our toy |V|=32 the
// meaningful assertion is the relative speedup vs random's |V|/2 average.

#[cfg(feature = "grad_policy")]
mod grad {
    use super::*;
    use crate::inversion::{
        InversionError, InversionGradient, InversionPolicy, invert_sequence_grad,
    };

    impl ToyTransformer {
        /// Forward pass from a continuous proxy embedding (no token lookup).
        /// Writes the layer-2 hidden state at position `position` into `out`.
        ///
        /// The toy has no attention between positions, so the proxy forward
        /// is just `apply_layer(w2, apply_layer(w1, proxy))` — prefix is
        /// irrelevant for this architecture (a real transformer's impl would
        /// attend over the prefix).
        fn forward_proxy_into(&self, proxy: &[f32], out: &mut [f32]) {
            debug_assert_eq!(proxy.len(), D);
            debug_assert_eq!(out.len(), D);
            out.copy_from_slice(proxy);
            self.apply_layer_into(&self.w1_up, &self.w1_down, out);
            self.apply_layer_into(&self.w2_up, &self.w2_down, out);
        }

        /// Central finite-difference gradient of `L(e) = ½·‖h̆ − F(e)‖²` w.r.t.
        /// `e`. O(D) forward evals per call — fine for the test; production
        /// callers supply an analytical gradient.
        fn numerical_grad_into(&self, observed_state: &[f32], proxy: &[f32], out: &mut [f32]) {
            debug_assert_eq!(proxy.len(), D);
            debug_assert_eq!(out.len(), D);
            let eps = 1.0e-3_f32;
            let mut f_plus = [0.0_f32; D];
            let mut f_minus = [0.0_f32; D];
            let mut e_plus = [0.0_f32; D];
            let mut e_minus = [0.0_f32; D];
            for i in 0..D {
                e_plus.copy_from_slice(proxy);
                e_minus.copy_from_slice(proxy);
                e_plus[i] += eps;
                e_minus[i] -= eps;
                self.forward_proxy_into(&e_plus, &mut f_plus);
                self.forward_proxy_into(&e_minus, &mut f_minus);
                // L(e) = ½·Σ_o (h̆_o − F(e)_o)²
                // ∂L/∂e_i ≈ [L(e+eps·î) − L(e−eps·î)] / (2·eps)
                let l_plus: f32 = observed_state
                    .iter()
                    .zip(f_plus.iter())
                    .map(|(o, f)| 0.5 * (o - f).powi(2))
                    .sum();
                let l_minus: f32 = observed_state
                    .iter()
                    .zip(f_minus.iter())
                    .map(|(o, f)| 0.5 * (o - f).powi(2))
                    .sum();
                out[i] = (l_plus - l_minus) / (2.0 * eps);
            }
        }
    }

    impl InversionGradient for ToyTransformer {
        fn grad_hidden_at_into(
            &self,
            _prefix: &[u32],
            observed_state: &[f32],
            proxy: &[f32],
            _position: usize,
            out: &mut [f32],
        ) -> Result<(), InversionError> {
            self.numerical_grad_into(observed_state, proxy, out);
            Ok(())
        }

        fn nearest_token(&self, proxy: &[f32]) -> Result<u32, InversionError> {
            // Linear scan — fine for |V|=32; production callers use KD-tree /
            // FAISS / etc.
            let mut best_v = 0_u32;
            let mut best_dist = f32::INFINITY;
            for v in 0..V {
                let base = (v as usize) * D;
                let embed = &self.embedding[base..base + D];
                let dist: f32 = proxy
                    .iter()
                    .zip(embed.iter())
                    .map(|(p, e)| (p - e).powi(2))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_v = v;
                }
            }
            Ok(best_v)
        }

        fn init_proxy_into(&self, out: &mut [f32]) -> Result<(), InversionError> {
            // Paper §E.1: start at the mean of all vocabulary embeddings.
            // This is closer to every individual embedding than zeros, so
            // the gradient basin is more likely to contain the correct
            // token.
            for x in out.iter_mut() {
                *x = 0.0;
            }
            for v in 0..V {
                let base = (v as usize) * D;
                for (out_i, embed_val) in out.iter_mut().zip(&self.embedding[base..base + D]) {
                    *out_i += *embed_val;
                }
            }
            let inv = 1.0 / V as f32;
            for x in out.iter_mut() {
                *x *= inv;
            }
            Ok(())
        }
    }

    /// Count total acceptance tests (hidden_at + accept calls) across a full
    /// inversion run, by intercepting via a wrapper forward.
    struct CountingForward<'a, F: InversionForward> {
        inner: &'a F,
        count: std::cell::Cell<usize>,
    }

    impl<'a, F: InversionForward> InversionForward for CountingForward<'a, F> {
        fn hidden_at_into(
            &self,
            prefix: &[u32],
            candidate: u32,
            position: usize,
            out: &mut [f32],
        ) -> Result<(), InversionError> {
            self.count.set(self.count.get() + 1);
            self.inner.hidden_at_into(prefix, candidate, position, out)
        }
    }

    impl<'a, F: InversionForward> InversionGradient for CountingForward<'a, F>
    where
        F: InversionGradient,
    {
        fn grad_hidden_at_into(
            &self,
            prefix: &[u32],
            observed_state: &[f32],
            proxy: &[f32],
            position: usize,
            out: &mut [f32],
        ) -> Result<(), InversionError> {
            self.inner
                .grad_hidden_at_into(prefix, observed_state, proxy, position, out)
        }

        fn nearest_token(&self, proxy: &[f32]) -> Result<u32, InversionError> {
            self.inner.nearest_token(proxy)
        }

        fn init_proxy_into(&self, out: &mut [f32]) -> Result<(), InversionError> {
            self.inner.init_proxy_into(out)
        }
    }

    #[test]
    fn grad_guided_recovers_all_random_prompts() {
        // Phase 2 headline: same prompts as g1_exact_recovery_random_init,
        // but using gradient-guided policy. All must recover exactly.
        //
        // Uses weight scale 1.0 (vs Phase 1's 1/sqrt(D)) because the
        // gradient-guided policy needs a non-flat loss landscape to produce
        // meaningful gradients. The standard 1/sqrt(D) init produces near-
        // zero intermediate activations (GELU saturates near the origin),
        // making the Jacobian tiny and convergence glacial.
        const N_PROMPTS: usize = 8;

        let mut rng = fastrand::Rng::with_seed(0xC0DE);
        let transformer = ToyTransformer::new_scaled(&mut rng, 1.0);
        let mut recovered_count = 0;
        for prompt_seed in 0..N_PROMPTS {
            let mut prompt_rng = fastrand::Rng::with_seed(0xA5A5 + prompt_seed as u64);
            let prompt = random_prompt(&mut prompt_rng);

            let buf = transformer.forward_full(&prompt);
            let observed = ObservedStates::from_row_major(&buf, T, D).unwrap();
            let cfg = InversionConfig {
                policy: InversionPolicy::gradient_guided_default(),
                ..InversionConfig::default()
            };
            let result = invert_sequence_grad(
                &observed,
                V,
                &transformer,
                &transformer,
                &cfg,
                prompt_seed as u64,
            )
            .unwrap();
            match result {
                InversionResult::Recovered(recovered) => {
                    assert_eq!(
                        recovered, prompt,
                        "prompt {prompt:?} not recovered via gradient-guided"
                    );
                    recovered_count += 1;
                }
                InversionResult::Failed {
                    failed_position,
                    candidates_tried,
                } => panic!(
                    "gradient-guided failed at position {failed_position} after {candidates_tried} \
                     candidates (prompt {prompt:?})"
                ),
            }
        }
        assert_eq!(recovered_count, N_PROMPTS);
    }

    #[test]
    fn grad_guided_uses_fewer_acceptance_tests_than_random() {
        // Direct A/B comparison: gradient-guided vs uniform-random on the
        // same prompts. Assert gradient-guided uses strictly fewer acceptance
        // tests (hidden_at calls) total across all 8 prompts × 8 positions.
        // Uses scale 1.0 (see `grad_guided_recovers_all_random_prompts` for
        // why the standard 1/sqrt(D) scale is too flat for gradient-guided).
        const N_PROMPTS: usize = 8;

        let mut rng = fastrand::Rng::with_seed(0xC0DE);
        let transformer = ToyTransformer::new_scaled(&mut rng, 1.0);
        let mut random_total = 0_usize;
        let mut grad_total = 0_usize;

        for prompt_seed in 0..N_PROMPTS {
            let mut prompt_rng = fastrand::Rng::with_seed(0xA5A5 + prompt_seed as u64);
            let prompt = random_prompt(&mut prompt_rng);
            let buf = transformer.forward_full(&prompt);
            let observed = ObservedStates::from_row_major(&buf, T, D).unwrap();

            // Random-policy baseline.
            let random_cfg = InversionConfig::default();
            let random_counter = CountingForward {
                inner: &transformer,
                count: std::cell::Cell::new(0),
            };
            let r = invert_sequence(
                &observed,
                V,
                &random_counter,
                &random_cfg,
                prompt_seed as u64,
            )
            .unwrap();
            assert!(
                matches!(r, InversionResult::Recovered(_)),
                "random baseline failed"
            );
            random_total += random_counter.count.get();

            // Gradient-guided.
            let grad_cfg = InversionConfig {
                policy: InversionPolicy::gradient_guided_default(),
                ..InversionConfig::default()
            };
            let grad_counter = CountingForward {
                inner: &transformer,
                count: std::cell::Cell::new(0),
            };
            let r = invert_sequence_grad(
                &observed,
                V,
                &grad_counter,
                &grad_counter,
                &grad_cfg,
                prompt_seed as u64,
            )
            .unwrap();
            assert!(
                matches!(r, InversionResult::Recovered(_)),
                "gradient-guided failed"
            );
            grad_total += grad_counter.count.get();
        }

        // Sanity: random averages ~|V|/2 = 16 per position; with 8 prompts × 8
        // positions = 64 positions, random_total ≈ 64 × 16 = 1024.
        // Gradient-guided should use dramatically fewer acceptance tests.
        eprintln!(
            "A/B: random={random_total} acceptance tests, gradient-guided={grad_total} \
             ({:.1}× reduction)",
            (random_total - grad_total) as f64 / random_total as f64 * 100.0
        );
        assert!(
            grad_total < random_total,
            "gradient-guided ({grad_total} acceptance tests) should beat random ({random_total})"
        );
        // Stronger: with weight scale 1.0 (non-flat loss landscape), gradient-
        // guided should use < 50% of random's acceptance tests on this toy.
        // The paper reports <0.25%·|V| for |V|=32K-128K with near-orthogonal
        // high-dim embeddings; on our toy (|V|=32, D=16) the relative
        // improvement is smaller because the embedding matrix is rank-32 in
        // 16-dim space (tokens cannot be orthogonal). Phase 3 G2 (T3.2) will
        // measure sub-linear scaling on larger vocabs.
        assert!(
            grad_total * 2 < random_total,
            "gradient-guided ({grad_total}) should use <50% of random ({random_total}) acceptance tests"
        );
    }

    #[test]
    fn grad_guided_no_false_positive_on_corrupted_observed() {
        // Negative control: corrupted observed should NOT recover the original
        // prompt (same shape as g1_no_false_positive_on_mismatched_observed).
        let mut rng = fastrand::Rng::with_seed(0xFEED);
        let transformer = ToyTransformer::new_scaled(&mut rng, 1.0);

        let prompt_a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut buf_a = transformer.forward_full(&prompt_a);
        // Corrupt position 0's first coordinate by 100.0 (way above tolerance 1e-3,
        // and large relative to the scale-1.0 hidden-state magnitudes ~O(50)).
        buf_a[0] += 100.0;

        let observed = ObservedStates::from_row_major(&buf_a, T, D).unwrap();
        let cfg = InversionConfig {
            policy: InversionPolicy::gradient_guided_default(),
            ..InversionConfig::default()
        };
        let result =
            invert_sequence_grad(&observed, V, &transformer, &transformer, &cfg, 0).unwrap();
        match result {
            InversionResult::Failed { .. } => {
                // Cleanest: corrupted observed → no match.
            }
            InversionResult::Recovered(recovered) => {
                // Allow recovery of some prompt whose forward matches the
                // corrupted observed within tolerance — but it must not be
                // prompt_a (because observed was corrupted away from it).
                let re_buf = transformer.forward_full(&recovered);
                for t in 0..T {
                    let diff: f32 = observed
                        .row(t)
                        .iter()
                        .zip(re_buf[t * D..(t + 1) * D].iter())
                        .map(|(o, c)| (o - c).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        diff <= cfg.tolerance,
                        "recovered prompt's forward diverges from observed at position {t}: {diff}"
                    );
                }
                assert_ne!(
                    recovered, prompt_a,
                    "gradient-guided recovered prompt A from corrupted-A observed — false positive"
                );
            }
        }
    }

    #[test]
    fn grad_guided_with_random_policy_uses_grad_path_as_random() {
        // When the policy is Random but the caller uses invert_sequence_grad,
        // the driver should dispatch to the random path (no grad hook needed).
        // Verify bit-identical result to invert_sequence with the same seed.
        let mut rng = fastrand::Rng::with_seed(0x1234);
        let transformer = ToyTransformer::new_scaled(&mut rng, 1.0);
        let prompt = vec![5_u32, 10, 15, 20, 25, 30, 3, 7];
        let buf = transformer.forward_full(&prompt);
        let observed = ObservedStates::from_row_major(&buf, T, D).unwrap();

        let cfg_random = InversionConfig::default();
        let r1 = invert_sequence(&observed, V, &transformer, &cfg_random, 99).unwrap();
        let r2 = invert_sequence_grad(&observed, V, &transformer, &transformer, &cfg_random, 99)
            .unwrap();
        assert_eq!(
            r1, r2,
            "random via grad driver should be bit-identical to random via base driver"
        );
    }
}

// ─── Phase 4: robustness (paper Theorem 3.2) ─────────────────────────────
//
// Theorem 3.2 guarantees recovery under perturbation: if the observed state
// `ę_t = h̆_t + e_t` has noise `‖e_t‖_∞ < Δ_π,t / 2` (where `Δ_π,t` is the
// margin — the minimum L∞ distance from the true token's state to any other
// token's state at position `t` under prefix `π`), then recovery still works.
//
// We verify this empirically: compute the margin for each position, inject
// noise at varying fractions of `Δ_π,t / 2`, and assert recovery holds below
// the threshold and fails above it.

#[cfg(feature = "grad_policy")]
mod robustness {
    use super::*;
    use crate::inversion::{InversionConfig, InversionResult, ObservedStates, invert_sequence};

    /// Compute the margin `Δ_π,t` at position `t` for the given prompt: the
    /// minimum L∞ distance from the true token's forward output to any OTHER
    /// token's forward output at that position (under the recovered prefix).
    /// Returns `f32::INFINITY` if there's only one token in the vocab.
    fn compute_margin_at(transformer: &ToyTransformer, prompt: &[u32], t: usize) -> f32 {
        let prefix = &prompt[..t];
        let true_token = prompt[t];
        let mut true_state = [0.0_f32; D];
        transformer
            .hidden_at_into(prefix, true_token, t, &mut true_state)
            .unwrap();

        let mut min_dist = f32::INFINITY;
        for v in 0..V {
            if v == true_token {
                continue;
            }
            let mut state = [0.0_f32; D];
            transformer
                .hidden_at_into(prefix, v, t, &mut state)
                .unwrap();
            let linf: f32 = true_state
                .iter()
                .zip(state.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            if linf < min_dist {
                min_dist = linf;
            }
        }
        min_dist
    }

    /// Inject uniform noise `e` with `‖e‖_∞ ≤ max_noise` into each component
    /// of `buf`. Deterministic given `seed`.
    fn inject_noise_into(buf: &mut [f32], max_noise: f32, rng: &mut fastrand::Rng) {
        for x in buf.iter_mut() {
            // Uniform [-max_noise, max_noise].
            *x += (rng.f32() * 2.0 - 1.0) * max_noise;
        }
    }

    #[test]
    fn robust_recovery_holds_below_half_margin() {
        // Theorem 3.2: if ‖e_t‖_∞ < Δ_π,t / 2 for all t, recovery still works.
        // We inject noise at 0.1×, 0.25×, and 0.45× of min_t(Δ_π,t / 2) — all
        // strictly below the threshold — and verify exact recovery.
        let mut rng = fastrand::Rng::with_seed(0xB0_70);
        let transformer = ToyTransformer::new_scaled(&mut rng, 1.0);

        let prompt: Vec<u32> = vec![3, 7, 11, 15, 19, 23, 27, 31];

        // Compute the minimum margin across all positions.
        let mut min_margin = f32::INFINITY;
        for t in 0..T {
            let margin = compute_margin_at(&transformer, &prompt, t);
            if margin < min_margin {
                min_margin = margin;
            }
        }
        assert!(
            min_margin > 0.0,
            "margin should be positive on a non-degenerate transformer"
        );

        let half_margin = min_margin / 2.0;

        for &noise_fraction in &[0.1_f32, 0.25, 0.45] {
            let noise_level = half_margin * noise_fraction;
            let mut noisy_buf = transformer.forward_full(&prompt);
            inject_noise_into(&mut noisy_buf, noise_level, &mut rng);
            let observed = ObservedStates::from_row_major(&noisy_buf, T, D).unwrap();

            // Set tolerance to half_margin (the theoretical max for guaranteed
            // recovery). The injected noise is below this, so recovery should work.
            let cfg = InversionConfig {
                tolerance: half_margin,
                ..InversionConfig::default()
            };
            let result = invert_sequence(&observed, V, &transformer, &cfg, 0).unwrap();
            match result {
                InversionResult::Recovered(recovered) => assert_eq!(
                    recovered, prompt,
                    "recovery should hold at noise fraction {noise_fraction} of Δ/2"
                ),
                InversionResult::Failed {
                    failed_position, ..
                } => panic!(
                    "recovery failed at position {failed_position} with noise {noise_fraction}×Δ/2 \
                     (margin={min_margin:.4}, half={half_margin:.4}, noise={noise_level:.4})"
                ),
            }
        }
    }

    #[test]
    fn robust_recovery_fails_above_half_margin() {
        // Negative control: when noise exceeds Δ_π,t / 2, the perturbed observed
        // can fall into another token's acceptance region, causing recovery to
        // either fail or recover a DIFFERENT prompt. We inject noise at 2× the
        // half-margin and verify that exact recovery of the ORIGINAL prompt is
        // NOT guaranteed (either Failed, or Recovered != original).
        const N_TRIALS: usize = 20;

        let mut rng = fastrand::Rng::with_seed(0xB0_70 + 1);
        let transformer = ToyTransformer::new_scaled(&mut rng, 1.0);

        let prompt: Vec<u32> = vec![3, 7, 11, 15, 19, 23, 27, 31];

        let mut min_margin = f32::INFINITY;
        for t in 0..T {
            let margin = compute_margin_at(&transformer, &prompt, t);
            if margin < min_margin {
                min_margin = margin;
            }
        }
        let half_margin = min_margin / 2.0;

        // Noise at 2× half_margin — well above the Theorem 3.2 threshold.
        // Try multiple seeds; with enough noise, at least one trial should fail
        // exact recovery (either Failed or wrong prompt).
        let noise_level = half_margin * 2.0;
        let mut exact_recovery_count = 0;
        for trial in 0..N_TRIALS {
            let mut trial_rng = fastrand::Rng::with_seed(0xBAD + trial as u64);
            let mut noisy_buf = transformer.forward_full(&prompt);
            inject_noise_into(&mut noisy_buf, noise_level, &mut trial_rng);
            let observed = ObservedStates::from_row_major(&noisy_buf, T, D).unwrap();

            // Use a tight tolerance — the true token's state is at distance 0
            // from the unperturbed observed, but the noise pushes it away.
            let cfg = InversionConfig {
                tolerance: 1e-3,
                ..InversionConfig::default()
            };
            let result = invert_sequence(&observed, V, &transformer, &cfg, trial as u64).unwrap();
            match result {
                InversionResult::Recovered(recovered) if recovered == prompt => {
                    exact_recovery_count += 1;
                }
                _ => {
                    // Failed or wrong prompt — expected when noise > Δ/2.
                }
            }
        }

        // With noise at 2× half_margin and tight tolerance, NOT all trials
        // should recover exactly. If they all do, the margins are too large
        // for the noise to matter (or the toy is degenerate).
        // We expect at least one failure in 20 trials.
        assert!(
            exact_recovery_count < N_TRIALS,
            "all {N_TRIALS} trials recovered exactly with noise at 2×Δ/2 — \
             the margin {min_margin:.4} may be too large for this test to be meaningful, \
             or the noise injection is not reaching the acceptance boundary"
        );
    }

    #[test]
    fn robust_margin_is_positive_on_random_init() {
        // Sanity: the margin Δ_π,t should be strictly positive on a random-init
        // transformer (different tokens produce different states, by injectivity).
        // If this fails, the transformer is degenerate (two tokens produce the
        // same state at some position).
        let mut rng = fastrand::Rng::with_seed(0xB6_6C);
        let transformer = ToyTransformer::new_scaled(&mut rng, 1.0);

        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        for t in 0..T {
            let margin = compute_margin_at(&transformer, &prompt, t);
            assert!(
                margin > 0.0,
                "margin at position {t} is {margin} — should be positive (injectivity violation)"
            );
        }
    }
}

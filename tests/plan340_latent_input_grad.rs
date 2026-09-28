//! Plan 340 — latent-position gradients must reach the caller, not an embedding row.
//!
//! `kimi_k3_forward_token_hidden_saved` takes its input hidden state from the
//! caller and has no token to record, so it stores the *iteration index* in
//! `saved.token_id`. The embedding backward used that field as a row index
//! unconditionally, which meant a prefix/latent-tuning run (Plan 340 LOPD arms D
//! and E) would:
//!
//!   1. receive **no** gradient for its injected latents — nothing returned
//!      `d_prefix`; and
//!   2. **corrupt** `embed_weight` rows `0..n_latents` of a table the caller
//!      believes is frozen.
//!
//! Neither symptom raised anything, which is what made it dangerous: such a run
//! trains, logs plausible losses, and produces a composer that never moved.
//!
//! These tests fail on that old behaviour by construction — the first asserts the
//! rows stay clean, and the second asks for a value the old signature could not
//! return.

#![cfg(feature = "kimi_k3_backward")]

use katgpt_rs::kimi_k3::backward::{
    KimiK3ModelGradients, TokenSavedActivations, kimi_k3_backward_sequence,
    kimi_k3_backward_sequence_with_input_grad, kimi_k3_forward_token_hidden_saved,
    kimi_k3_forward_token_saved,
};
use katgpt_rs::kimi_k3::loader::KimiK3ModelWeights;
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime};

/// Smallest shape that still exercises both layer kinds.
fn tiny() -> (KimiK3ModelConfig, KimiK3ModelWeights) {
    let mut config = KimiK3ModelConfig::kimi_k3_0_40b();
    config.vocab_size = 256;
    config.num_layers = 2;
    config.mla_layer_indices = vec![1];
    let weights = KimiK3ModelWeights::random_train_init(&config, 340);
    (config, weights)
}

fn unit_d_logits(n: usize, vocab: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| vec![1.0f32 / vocab as f32; vocab]).collect()
}

/// Forward `n` caller-supplied hidden states through the latent seam.
fn forward_latents(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    n: usize,
) -> Vec<TokenSavedActivations> {
    let d = config.hidden_size;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Distinct, non-degenerate hidden states so a zero gradient cannot be an
        // artifact of a zero input.
        let latent: Vec<f32> = (0..d)
            .map(|j| ((i * 31 + j) as f32 * 0.017).sin() * 0.1 + 0.05)
            .collect();
        runtime.hidden[..d].copy_from_slice(&latent);
        let mut saved = TokenSavedActivations::new();
        kimi_k3_forward_token_hidden_saved(config, weights, runtime, i, i, &mut saved);
        out.push(saved);
    }
    out
}

#[test]
fn latent_positions_are_flagged_and_do_not_touch_embedding_rows() {
    let (config, weights) = tiny();
    let d = config.hidden_size;
    let v = config.vocab_size;
    let n = 3;
    let mut rt = KimiK3Runtime::new(&config, 8);

    let saved = forward_latents(&config, &weights, &mut rt, n);
    assert!(
        saved.iter().all(|s| s.is_latent),
        "the hidden-input seam must mark its positions latent"
    );
    // The field it repurposes is still the iteration index — that is precisely why
    // it must not be used as an embedding row.
    assert_eq!(
        saved.iter().map(|s| s.token_id).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );

    let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
    let mut d_in: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
    kimi_k3_backward_sequence_with_input_grad(
        &config,
        &weights,
        &rt,
        &saved,
        &unit_d_logits(n, v),
        &mut grads,
        Some(&mut d_in),
    );

    // THE REGRESSION: rows 0..n used to receive the latents' input gradient.
    for row in 0..n {
        let base = row * d;
        let dirty = grads.embed_weight[base..base + d].iter().any(|g| *g != 0.0);
        assert!(
            !dirty,
            "embed_weight row {row} was written by a latent position — the frozen \
             table is being corrupted (this is the Plan 340 misattribution)"
        );
    }
    // And nothing else in the table either: no latent position has an owner row.
    let any_embed = grads.embed_weight.iter().any(|g| *g != 0.0);
    assert!(
        !any_embed,
        "a latent-only backward must leave embed_weight entirely untouched"
    );

    // The gradient must arrive where a composer can consume it.
    for (i, g) in d_in.iter().enumerate() {
        assert_eq!(g.len(), d, "d_input_hidden[{i}] must be d-wide");
        assert!(
            g.iter().all(|x| x.is_finite()),
            "d_input_hidden[{i}] must be finite"
        );
        assert!(
            g.iter().any(|x| *x != 0.0),
            "d_input_hidden[{i}] is all zero — the composer would receive no signal"
        );
    }
}

#[test]
fn token_positions_still_scatter_to_their_own_embedding_row() {
    // The fix must not cost the token path its gradient.
    let (config, weights) = tiny();
    let d = config.hidden_size;
    let v = config.vocab_size;
    let ids: [u32; 2] = [5, 9];
    let mut rt = KimiK3Runtime::new(&config, 8);

    let mut saved = Vec::new();
    for (pos, &t) in ids.iter().enumerate() {
        let mut s = TokenSavedActivations::new();
        kimi_k3_forward_token_saved(&config, &weights, &mut rt, t, pos, &mut s);
        assert!(!s.is_latent, "the token path must not be flagged latent");
        saved.push(s);
    }

    let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
    kimi_k3_backward_sequence(
        &config,
        &weights,
        &rt,
        &saved,
        &unit_d_logits(2, v),
        &mut grads,
    );

    for &t in &ids {
        let base = t as usize * d;
        assert!(
            grads.embed_weight[base..base + d].iter().any(|g| *g != 0.0),
            "token {t}'s embedding row must receive gradient"
        );
    }
    // Rows that took no part stay clean.
    for row in [0usize, 1, 2] {
        let base = row * d;
        assert!(
            grads.embed_weight[base..base + d].iter().all(|g| *g == 0.0),
            "row {row} took no part and must stay zero"
        );
    }
}

#[test]
fn wrapper_matches_the_out_param_variant_bitwise_on_the_token_path() {
    // `kimi_k3_backward_sequence` now delegates. Pin that delegating changed
    // nothing for existing callers — bitwise, since the docs promise the SIMD add
    // is bit-identical.
    let (config, weights) = tiny();
    let v = config.vocab_size;
    let ids: [u32; 3] = [7, 11, 200];
    let mut rt = KimiK3Runtime::new(&config, 8);

    let mut saved = Vec::new();
    for (pos, &t) in ids.iter().enumerate() {
        let mut s = TokenSavedActivations::new();
        kimi_k3_forward_token_saved(&config, &weights, &mut rt, t, pos, &mut s);
        saved.push(s);
    }
    let dl = unit_d_logits(ids.len(), v);

    let mut g_wrapper = KimiK3ModelGradients::zeros_like(&config, &weights);
    kimi_k3_backward_sequence(&config, &weights, &rt, &saved, &dl, &mut g_wrapper);

    let mut g_explicit = KimiK3ModelGradients::zeros_like(&config, &weights);
    kimi_k3_backward_sequence_with_input_grad(
        &config,
        &weights,
        &rt,
        &saved,
        &dl,
        &mut g_explicit,
        None,
    );

    assert_eq!(
        g_wrapper
            .embed_weight
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>(),
        g_explicit
            .embed_weight
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>(),
        "delegating must be bit-identical for existing callers"
    );
    assert_eq!(
        g_wrapper
            .lm_head_weight
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>(),
        g_explicit
            .lm_head_weight
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn mixed_span_credits_tokens_and_latents_separately() {
    // The realistic LOPD shape: injected latents followed by real tokens. Each
    // position's gradient must go to exactly one destination.
    let (config, weights) = tiny();
    let d = config.hidden_size;
    let v = config.vocab_size;
    let mut rt = KimiK3Runtime::new(&config, 8);

    let mut saved = forward_latents(&config, &weights, &mut rt, 2);
    let tok: u32 = 42;
    let mut s = TokenSavedActivations::new();
    kimi_k3_forward_token_saved(&config, &weights, &mut rt, tok, 2, &mut s);
    saved.push(s);

    let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
    let mut d_in: Vec<Vec<f32>> = (0..saved.len()).map(|_| Vec::new()).collect();
    kimi_k3_backward_sequence_with_input_grad(
        &config,
        &weights,
        &rt,
        &saved,
        &unit_d_logits(saved.len(), v),
        &mut grads,
        Some(&mut d_in),
    );

    // Only the real token's row is credited.
    let base = tok as usize * d;
    assert!(
        grads.embed_weight[base..base + d].iter().any(|g| *g != 0.0),
        "the real token's row must be credited"
    );
    for row in [0usize, 1] {
        let b = row * d;
        assert!(
            grads.embed_weight[b..b + d].iter().all(|g| *g == 0.0),
            "latent iteration index {row} must not be treated as an embedding row"
        );
    }
    // Every position, latent or not, reports its input gradient.
    assert!(
        d_in.iter()
            .all(|g| g.len() == d && g.iter().any(|x| *x != 0.0))
    );
}

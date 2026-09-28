//! SW-SetDLM set-causal training + evaluation (Research 376 Phase 4) and the
//! custom-order reveal seam (Issue 813).
//!
//! Moved out of `dllm/mod.rs` (2026-09-17, Issue 813): mod.rs sat at the
//! 2048-line soft limit and the seam grows this surface. Re-exports in
//! `mod.rs` preserve every historical
//! `crate::dllm::{train_mini_set_causal, evaluate_set_causal_nelbo}` path
//! (the Plan 402/403 move-and-re-export precedent).
//!
//! # The custom-order seam (Issue 813)
//!
//! [`train_mini_set_causal`] / [`evaluate_set_causal_nelbo`] keep their
//! signatures and stay byte-identical: they delegate to the generation-steps
//! cores with a schedule closure whose rng draw sequence is exactly the
//! pre-seam path (`seq_len` uniforms per sequence, after the epoch shuffle —
//! pinned bitwise by `tests/bench_809_set_causal_order_sweep.rs` G1).
//!
//! The new [`train_mini_set_causal_with_gen_steps`] /
//! [`evaluate_set_causal_nelbo_with_gen_steps`] accept a per-sequence
//! generation-step source — a `(seq_len, tokens, rng) -> Vec<u32>` closure —
//! so a reveal arm can feed ANY ordering: a deterministic permutation
//! (`katgpt_core::ar_order`, confidence-descending
//! `katgpt_core::probability_order`), the all-at-once MDLM endpoint
//! (`katgpt_core::mdlm_gen_steps`), or a sampled schedule.
//!
//! A `PositionOffsetSchedule::Custom` variant was considered and rejected:
//! the schedule is a plain `w`/`k` struct consumed cross-repo (riir-train
//! samples it with `fastrand::Rng`), a struct→enum rewrite would touch every
//! construction site, and a fixed `Vec<usize>` order cannot express
//! per-sequence lengths or the MDLM endpoint. The closure seam changes no
//! existing caller and generalizes over both.

use super::{
    BackwardContext, ForwardSaveContext, LossAveraging, backward, corrupt_block_into,
    forward_save_set_causal, masked_loss_into, sgd_update,
};
use crate::transformer::TransformerWeights;
use crate::types::{Config, Rng};
use katgpt_core::PositionOffsetSchedule;
use katgpt_core::order_to_gen_steps;

/// Per-sequence generation-step source — the Issue 813 custom-order seam.
///
/// Given `(seq_len, tokens, rng)` return the per-position generation steps
/// the set-causal kernel consumes (see
/// `katgpt_forward::forward_set_causal_positions`: position `t` is visible to
/// query `q` iff `gen_step[t] <= gen_step[q]`). Deterministic orderings
/// ignore `rng`; schedule-sampled orderings draw from it.
///
/// The rng is a PARAMETER, not a capture, so one core serves both
/// deterministic and sampled arms — and the schedule wrapper's draw sequence
/// (the seq_len uniforms) is exactly the legacy path's, which is what makes
/// the byte-identity pin hold.
pub type SetCausalGenStepsFn<'a> = &'a mut dyn FnMut(usize, &[usize], &mut Rng) -> Vec<u32>;

/// The incumbent schedule reveal as a gen-steps closure.
///
/// Draw-for-byte identical to the pre-seam inline path: `seq_len` uniforms
/// via `sample_order_with`, consumed after the epoch shuffle, before the
/// forward pass.
fn schedule_gen_steps(
    schedule: &PositionOffsetSchedule,
) -> impl FnMut(usize, &[usize], &mut Rng) -> Vec<u32> + '_ {
    move |len: usize, _tokens: &[usize], rng: &mut Rng| {
        order_to_gen_steps(&schedule.sample_order_with(len, || rng.uniform()))
    }
}

// ═══════════════════════════════════════════════════════════════
// Research 376 Phase 4: SW-SetDLM Training (set-causal attention)
// ═══════════════════════════════════════════════════════════════
//
// The functions in this module reference `crate::speculative::set_diffusion`
// (via the set-causal forward), which is gated behind the `set_diffusion`
// feature — the module carries that gate so `cargo build -p katgpt-rs` still
// works when `dllm` is enabled but `set_diffusion` is not (feature
// unification via dev-deps + dev-dep defaults transitively enables `dllm`).

/// Train a mini transformer with **set-causal attention** (SW-SetDLM training).
///
/// This is the set-causal counterpart of [`super::train_mini_dllm`]. Each training
/// step samples a generation ordering σ from the [`PositionOffsetSchedule`],
/// converts it to generation steps, runs a set-causal forward pass via
/// `forward_save_set_causal`, computes the NELBO loss (mean cross-entropy
/// over ALL positions — the all-L-conditionals estimator, Eq. 9), and runs
/// backprop + SGD update.
///
/// # Why this exists (the GOAT-gate unblock)
///
/// The set-diffusion decoder substrate (Phase 4 T4.1–T4.3) is validated
/// against bidirectionally-trained models, but a bidirectional model shows
/// NO GAIN over direct bidirectional decode at the MDLM endpoint (they're the
/// same thing). To pass the GOAT gate, the decoder needs a model that was
/// TRAINED to exploit set-causal attention's flexibility. This function
/// produces such a model on CPU, mirroring the `train_mini_dllm` precedent.
///
/// # Key difference from `train_mini_dllm`
///
/// | Aspect | `train_mini_dllm` | `train_mini_set_causal` |
/// |--------|-------------------|-------------------------|
/// | Attention | Bidirectional (all ↔ all) | Set-causal (gen-step masked) |
/// | Input | Corrupted (mask_token substituted) | Clean tokens (no corruption) |
/// | Targets | Masked positions only | ALL positions |
/// | Ordering | Fixed (bidirectional) | Sampled from schedule each step |
///
/// In SW-SetDLM training, the model sees CLEAN tokens and predicts each token
/// conditioned on its set-causal context (positions revealed earlier in σ).
/// The loss is the mean cross-entropy over all L positions — this is the
/// all-L-conditionals estimator that gives ~3× lower gradient variance than
/// single-position estimation (paper Table 5, verified in
/// `riir-train/tests/set_diffusion_variance_376.rs`).
///
/// # Arguments
/// - `schedule`: the position-offset schedule to sample orderings from.
///   Use [`PositionOffsetSchedule::default`] for the SW-SetDLM setting (w=0.5).
/// - All other args mirror `train_mini_dllm`.
///
/// # Returns
/// `(weights, loss_history)` — the trained model and per-epoch mean NELBO.
pub fn train_mini_set_causal(
    config: &Config,
    train_data: &[Vec<usize>],
    test_data: &[Vec<usize>],
    n_epochs: usize,
    lr: f32,
    schedule: &PositionOffsetSchedule,
    seed: u64,
) -> (TransformerWeights, Vec<f32>) {
    let mut gen_steps_for = schedule_gen_steps(schedule);
    train_mini_set_causal_with_gen_steps(
        config,
        train_data,
        test_data,
        n_epochs,
        lr,
        &mut gen_steps_for,
        seed,
    )
}

/// [`train_mini_set_causal`] under a CUSTOM per-sequence reveal source
/// (Issue 813 seam).
///
/// `gen_steps_for` replaces the schedule: each training step calls it with
/// `(seq_len, tokens, &mut rng)` and consumes the returned generation steps
/// verbatim. Everything else — epoch shuffle, clean-token set-causal forward,
/// all-L-conditionals NELBO, backprop — is the incumbent path, byte-identical
/// when the closure reproduces the schedule's sampling (the G1 pin).
///
/// Deterministic orderings ignore the rng param; sampled orderings draw from
/// it. `tokens` carries the SEQUENCE (not the corrupted input — set-causal
/// training is clean-token), so law-based confidence ordering is expressible.
pub fn train_mini_set_causal_with_gen_steps(
    config: &Config,
    train_data: &[Vec<usize>],
    test_data: &[Vec<usize>],
    n_epochs: usize,
    lr: f32,
    gen_steps_for: SetCausalGenStepsFn<'_>,
    seed: u64,
) -> (TransformerWeights, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut weights = TransformerWeights::new(config, &mut rng);
    let mut loss_history = Vec::with_capacity(n_epochs);
    let mut fwd_ctx = ForwardSaveContext::new(config);
    let mut bwd_ctx = BackwardContext::new(config);

    // In SW-SetDLM training, ALL positions are targets (no masking/corruption).
    // The model sees clean tokens and predicts each given its set-causal context.
    let mut is_masked_all: Vec<bool> = vec![true; config.block_size];

    let mut indices: Vec<usize> = (0..train_data.len()).collect();
    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0f32;
        let mut n_samples = 0usize;

        // Shuffle training data in-place
        for i in (1..indices.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let tokens = &train_data[idx];
            let seq_len = tokens.len().min(config.block_size);
            is_masked_all[..seq_len].fill(true);

            // Issue 813 seam: the reveal source produces the generation steps
            // (schedule-sampled order, deterministic permutation, or the MDLM
            // endpoint) in place of the inline schedule sampling.
            let gen_steps = gen_steps_for(seq_len, tokens, &mut rng);

            // Set-causal forward with activation saving.
            let act = forward_save_set_causal(&weights, tokens, config, &gen_steps, &mut fwd_ctx);

            // NELBO loss: mean cross-entropy over all positions.
            let loss = masked_loss_into(
                act.logits,
                tokens,
                &is_masked_all[..seq_len],
                config.vocab_size,
                LossAveraging::Global,
                &mut bwd_ctx.loss_exp_buf,
            );

            // Backward + SGD update (same as train_mini_dllm — the mask is
            // encoded in the attention weights, so backward() works as-is).
            backward(
                &act,
                &weights,
                tokens,
                &is_masked_all[..seq_len],
                config,
                &mut bwd_ctx,
            );
            sgd_update(&mut weights, &bwd_ctx.grads, lr);

            epoch_loss += loss;
            n_samples += 1;
        }

        let avg_loss = if n_samples > 0 {
            epoch_loss / n_samples as f32
        } else {
            0.0
        };
        loss_history.push(avg_loss);

        if epoch % 100 == 0 || epoch == n_epochs - 1 {
            // Evaluate NELBO on test data at the training reveal source.
            let test_nelbo = evaluate_set_causal_nelbo_with_gen_steps_internal(
                &weights,
                test_data,
                config,
                gen_steps_for,
                &mut rng,
                &mut fwd_ctx,
            );
            eprintln!(
                "Epoch {epoch:>4}/{n_epochs}: train_nelbo={avg_loss:.4} test_nelbo={test_nelbo:.4}",
            );
        }
    }

    (weights, loss_history)
}

/// Evaluate mean NELBO of a model under set-causal attention at a given schedule.
///
/// Samples one ordering per test sequence (matching the training distribution)
/// and computes the mean NELBO. Allocates its own forward context — use
/// `evaluate_set_causal_nelbo_with_gen_steps_internal` in hot paths to reuse a
/// context.
///
/// Used by the GOAT gate test for cross-model comparison (set-causal vs
/// bidirectional models at various schedule endpoints).
pub fn evaluate_set_causal_nelbo(
    weights: &TransformerWeights,
    data: &[Vec<usize>],
    config: &Config,
    schedule: &PositionOffsetSchedule,
    rng: &mut Rng,
) -> f32 {
    let mut gen_steps_for = schedule_gen_steps(schedule);
    evaluate_set_causal_nelbo_with_gen_steps(weights, data, config, &mut gen_steps_for, rng)
}

/// [`evaluate_set_causal_nelbo`] under a CUSTOM per-sequence reveal source
/// (Issue 813 seam).
///
/// One ordering per test sequence via `gen_steps_for` — the eval counterpart
/// of [`train_mini_set_causal_with_gen_steps`], so a sweep arm trains and
/// evaluates under the SAME reveal law. Allocates its own forward context.
pub fn evaluate_set_causal_nelbo_with_gen_steps(
    weights: &TransformerWeights,
    data: &[Vec<usize>],
    config: &Config,
    gen_steps_for: SetCausalGenStepsFn<'_>,
    rng: &mut Rng,
) -> f32 {
    let mut fwd_ctx = ForwardSaveContext::new(config);
    evaluate_set_causal_nelbo_with_gen_steps_internal(
        weights,
        data,
        config,
        gen_steps_for,
        rng,
        &mut fwd_ctx,
    )
}

/// Internal allocation-free variant — caller provides the forward context.
fn evaluate_set_causal_nelbo_with_gen_steps_internal(
    weights: &TransformerWeights,
    data: &[Vec<usize>],
    config: &Config,
    gen_steps_for: SetCausalGenStepsFn<'_>,
    rng: &mut Rng,
    fwd_ctx: &mut ForwardSaveContext,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    let mut is_masked_all: Vec<bool> = vec![true; config.block_size];
    let mut exp_buf: Vec<f32> = vec![0.0f32; config.vocab_size];
    for tokens in data {
        let seq_len = tokens.len().min(config.block_size);
        if seq_len == 0 {
            continue;
        }
        is_masked_all[..seq_len].fill(true);
        let gen_steps = gen_steps_for(seq_len, tokens, rng);
        let act = forward_save_set_causal(weights, tokens, config, &gen_steps, fwd_ctx);
        let loss = masked_loss_into(
            act.logits,
            tokens,
            &is_masked_all[..seq_len],
            config.vocab_size,
            LossAveraging::Global,
            &mut exp_buf,
        );
        total += loss;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

// ═══════════════════════════════════════════════════════════════
// Issue 816: the masked-target (denoiser) set-causal trainer
// ═══════════════════════════════════════════════════════════════
//
// Bench 809 measured the clean-token objective as SELF-COPY TRIVIAL: the
// copy path is the RESIDUAL stream (Phase C adds `after_norm1` =
// norm(wte[token] + wpe) back before the logits), not just self-attention —
// a strict-self attention variant alone leaves the leak intact (measured:
// every arm still converges to ~0 NELBO). The fix is the paper-faithful
// denoiser shape: corrupt the input (mask_ratio), restrict the loss to the
// masked positions — their residual carries `wte[mask_token]`, not the
// answer — and let the set-causal mask decide how much CONTEXT each masked
// position gets. Every mechanism (corrupt_block_into, masked_loss_into's
// is_masked restriction, the zero-weight ⇒ zero-gradient backward invariant)
// already shipped; this is their composition under the Issue-813 seam.

/// Train a set-causal denoiser under a CUSTOM reveal source (Issue 816).
///
/// Per sample: corrupt the sequence at `mask_ratio` (the
/// `train_mini_dllm` corruption model), draw the reveal ordering via
/// `gen_steps_for` (the Issue-813 seam), run the INCUMBENT set-causal
/// forward (same-set + earlier context), and take the loss over the MASKED
/// positions only — so the target token never enters the prediction path
/// and the reveal-order axis becomes measurable (Bench 809's blocked cell).
///
/// The order arms are comparable under paired seeds: the corruption draw is
/// order-independent, so every arm sees the same corrupted inputs and
/// differs only in which positions get context.
pub fn train_mini_set_causal_denoiser_with_gen_steps(
    config: &Config,
    train_data: &[Vec<usize>],
    test_data: &[Vec<usize>],
    n_epochs: usize,
    lr: f32,
    mask_ratio: f32,
    gen_steps_for: SetCausalGenStepsFn<'_>,
    seed: u64,
) -> (TransformerWeights, Vec<f32>) {
    assert!(
        (0.0..=1.0).contains(&mask_ratio),
        "mask_ratio must be in [0, 1], got {mask_ratio}"
    );
    let mut rng = Rng::new(seed);
    let mut weights = TransformerWeights::new(config, &mut rng);
    let mut loss_history = Vec::with_capacity(n_epochs);
    let mut fwd_ctx = ForwardSaveContext::new(config);
    let mut bwd_ctx = BackwardContext::new(config);
    let mut corrupted_buf = Vec::with_capacity(config.block_size);
    let mut is_masked_buf = Vec::with_capacity(config.block_size);
    let mut positions_buf = Vec::with_capacity(config.block_size);

    let mut indices: Vec<usize> = (0..train_data.len()).collect();
    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0f32;
        let mut n_samples = 0usize;

        // Shuffle training data in-place (the shared trainer convention).
        for i in (1..indices.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            indices.swap(i, j);
        }

        for &idx in &indices {
            let tokens = &train_data[idx];
            let seq_len = tokens.len().min(config.block_size);

            let n_mask = corrupt_block_into(
                tokens,
                mask_ratio,
                config.mask_token,
                &mut rng,
                &mut corrupted_buf,
                &mut is_masked_buf,
                &mut positions_buf,
            );
            // Skip if nothing masked — the loss set is empty.
            if n_mask == 0 {
                continue;
            }

            // Issue 813 seam: the reveal source produces the generation steps
            // over the CLEAN length (the corruption does not change it).
            let gen_steps = gen_steps_for(seq_len, tokens, &mut rng);

            // Incumbent set-causal forward over the CORRUPTED input.
            let act =
                forward_save_set_causal(&weights, &corrupted_buf, config, &gen_steps, &mut fwd_ctx);

            // Denoiser loss: mean cross-entropy over the MASKED positions
            // only (the leak-free loss set — unmasked positions carry their
            // own token in the residual and are excluded).
            let loss = masked_loss_into(
                act.logits,
                tokens,
                &is_masked_buf,
                config.vocab_size,
                LossAveraging::Global,
                &mut bwd_ctx.loss_exp_buf,
            );

            // Backward + SGD: the zero-weight ⇒ zero-gradient attention
            // invariant carries the mask; the loss set carries the target.
            backward(&act, &weights, tokens, &is_masked_buf, config, &mut bwd_ctx);
            sgd_update(&mut weights, &bwd_ctx.grads, lr);

            epoch_loss += loss;
            n_samples += 1;
        }

        let avg_loss = if n_samples > 0 {
            epoch_loss / n_samples as f32
        } else {
            0.0
        };
        loss_history.push(avg_loss);

        if epoch % 100 == 0 || epoch == n_epochs - 1 {
            let test_nll = evaluate_set_causal_denoiser_nll_with_gen_steps_internal(
                &weights,
                test_data,
                config,
                mask_ratio,
                gen_steps_for,
                &mut rng,
                &mut fwd_ctx,
            );
            eprintln!(
                "Epoch {epoch:>4}/{n_epochs}: train_masked_nll={avg_loss:.4} test_masked_nll={test_nll:.4}",
            );
        }
    }

    (weights, loss_history)
}

/// Evaluate mean NLL over masked positions for a set-causal denoiser under a
/// CUSTOM reveal source (Issue 816) — one corruption + ordering draw per
/// sequence. The eval companion of
/// [`train_mini_set_causal_denoiser_with_gen_steps`], so a sweep arm trains
/// and evaluates under the same objective.
pub fn evaluate_set_causal_denoiser_nll_with_gen_steps(
    weights: &TransformerWeights,
    data: &[Vec<usize>],
    config: &Config,
    mask_ratio: f32,
    gen_steps_for: SetCausalGenStepsFn<'_>,
    rng: &mut Rng,
) -> f32 {
    let mut fwd_ctx = ForwardSaveContext::new(config);
    evaluate_set_causal_denoiser_nll_with_gen_steps_internal(
        weights,
        data,
        config,
        mask_ratio,
        gen_steps_for,
        rng,
        &mut fwd_ctx,
    )
}

fn evaluate_set_causal_denoiser_nll_with_gen_steps_internal(
    weights: &TransformerWeights,
    data: &[Vec<usize>],
    config: &Config,
    mask_ratio: f32,
    gen_steps_for: SetCausalGenStepsFn<'_>,
    rng: &mut Rng,
    fwd_ctx: &mut ForwardSaveContext,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    let mut corrupted_buf: Vec<usize> = Vec::with_capacity(config.block_size);
    let mut is_masked_buf: Vec<bool> = Vec::with_capacity(config.block_size);
    let mut positions_buf: Vec<usize> = Vec::with_capacity(config.block_size);
    let mut exp_buf: Vec<f32> = vec![0.0f32; config.vocab_size];
    for tokens in data {
        let seq_len = tokens.len().min(config.block_size);
        if seq_len == 0 {
            continue;
        }
        let n_mask = corrupt_block_into(
            tokens,
            mask_ratio,
            config.mask_token,
            rng,
            &mut corrupted_buf,
            &mut is_masked_buf,
            &mut positions_buf,
        );
        if n_mask == 0 {
            continue;
        }
        let gen_steps = gen_steps_for(seq_len, tokens, rng);
        let act = forward_save_set_causal(weights, &corrupted_buf, config, &gen_steps, fwd_ctx);
        let loss = masked_loss_into(
            act.logits,
            tokens,
            &is_masked_buf,
            config.vocab_size,
            LossAveraging::Global,
            &mut exp_buf,
        );
        total += loss;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

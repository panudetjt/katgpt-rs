//! FlashAR Strided Anchor-Then-Fill D2F Decoding — root re-export shim.
//!
//! Plan 400 (2026-07-05): the production code moved to
//! `crates/katgpt-forward/src/flashar_anchor.rs`. This file is now a thin
//! re-export shim that preserves the historical
//! `crate::speculative::flashar_anchor::*` import path, plus the 2
//! training-coupled tests that depend on `crate::dllm::{train_mini_dllm,
//! generate_pattern_dataset}` (root-only training code).

#![allow(clippy::too_many_arguments)]

pub use katgpt_forward::flashar_anchor::{
    AnchorConfig, AnchorFillResult, ConfidenceAnchorConfig, anchor_fill_with_prefilled,
    anchor_then_fill, anchor_then_fill_with, dbtm_floor, select_confidence_anchors,
    select_confidence_anchors_into,
};

// ---------------------------------------------------------------------------
// Tests — 2 training-coupled tests that cannot move to katgpt-forward.
//
// These tests call `crate::dllm::{generate_pattern_dataset, train_mini_dllm}`
// which is root-only training code. The 6 inference-only tests moved with
// the production file to `katgpt-forward/src/flashar_anchor.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::dllm::D2fContext;
    use crate::speculative::d2f::D2fDecodeConfig;
    use crate::speculative::flashar_anchor::{AnchorConfig, anchor_then_fill};
    use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
    use crate::types::{Config, Rng};

    fn make_config() -> Config {
        Config::micro_dllm()
    }

    /// Train a mini D2F model on pattern data (same recipe as
    /// `speculative::d2f::tests::test_decode_with_trained_model`).
    ///
    /// The anchor-then-fill tests assert properties that only hold for a
    /// trained model — a random `TransformerWeights::new` produces a
    /// degenerate all-mask baseline that trivially converges in 1 step by
    /// emitting the same token at every position, which inverts the
    /// step-reduction comparison the tests are trying to verify.
    fn make_trained_weights() -> (Config, TransformerWeights) {
        use crate::dllm::{generate_pattern_dataset, train_mini_dllm};
        let config = make_config();
        let mut train_rng = Rng::new(123);
        let train_data =
            generate_pattern_dataset(&mut train_rng, 20, config.block_size, config.vocab_size - 1);
        let test_data =
            generate_pattern_dataset(&mut train_rng, 5, config.block_size, config.vocab_size - 1);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);
        (config, weights)
    }

    #[test]
    fn test_anchor_then_fill_produces_valid_output() {
        // Uses a trained mini D2F model — random weights produce a degenerate
        // all-same-token output that doesn't exercise the fill path.
        let (config, weights) = make_trained_weights();
        let mut rng = Rng::new(42);
        let block_size = 8;

        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(2);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut dctx = D2fContext::new(&config);

        let result = anchor_then_fill(
            &mut ctx,
            &mut cache,
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0, // seed_token
            0, // start_pos
            &mut rng,
        );

        // Should produce block_size tokens
        assert_eq!(result.tokens.len(), block_size);
        // All tokens should be valid (non-mask)
        let mask = config.mask_token;
        for (i, &t) in result.tokens.iter().enumerate() {
            assert_ne!(t, mask, "token at position {i} should not be mask");
        }
        // Should have anchors
        assert!(result.n_anchors > 0, "should have at least 1 anchor");
    }

    #[test]
    fn test_anchor_then_fill_reduces_steps() {
        // The "anchors reduce denoising steps" property only holds for a
        // trained model — with random weights, the all-mask baseline
        // degenerately converges in 1 step (same token at every position),
        // inverting the comparison. See `make_trained_weights` doc.
        let (config, weights) = make_trained_weights();
        let mut rng = Rng::new(42);
        let block_size = 8;

        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(2);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut dctx = D2fContext::new(&config);

        let result = anchor_then_fill(
            &mut ctx,
            &mut cache,
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0,
            0,
            &mut rng,
        );

        // With anchors pre-filled, fill steps should be ≤ baseline steps
        // (anchors reduce the denoising search space)
        assert!(
            result.fill_steps_used <= result.baseline_steps_used,
            "anchor fill ({}) should use ≤ baseline ({}) steps",
            result.fill_steps_used,
            result.baseline_steps_used,
        );

        println!(
            "  anchors={}, fill_steps={}, baseline_steps={}, reduction={}",
            result.n_anchors,
            result.fill_steps_used,
            result.baseline_steps_used,
            result.step_reduction,
        );
    }

    // =========================================================================
    // Issue 811 PoC — DBTM confidence-commit anchor rule (κ ∪ floor)
    // vs the strided incumbent, on ONE trained mini-D2F. Research 563 §2.3.
    // =========================================================================
    //
    // All arms share ONE Round-1 AR walk per (sequence, row): the walk is
    // identical to the production `predict_anchors` shape (same rng stream,
    // sampled-token context propagation), and additionally records the
    // per-position argmax proposal + its max-softmax confidence q — the
    // signal DBTM's rule consumes and the stride rule ignores. Selection
    // differs per arm; the fill runs through the SHARED production path
    // (`anchor_fill_with_prefilled`), which is what makes the comparison
    // apples-to-apples. `test_issue811_harness_matches_production` pins the
    // harness walk to byte-identical parity with the production decode.

    use super::ConfidenceAnchorConfig;
    use super::anchor_fill_with_prefilled;
    use super::anchor_then_fill_with;
    use super::select_confidence_anchors;
    use crate::dllm::generate_pattern_dataset;
    use katgpt_core::simd::simd_argmax_f32;
    use katgpt_core::softmax_scaled;
    use katgpt_core::speculative::sampling::sample_from_distribution;

    const N_EVAL_SEQUENCES: usize = 24;
    const BLOCK: usize = 8;

    /// Round-1 AR walk (production `predict_anchors` shape) + the
    /// confidence record the DBTM rule consumes.
    #[allow(clippy::too_many_arguments)]
    fn round1_walk(
        ctx: &mut ForwardContext,
        cache: &mut MultiLayerKVCache,
        weights: &TransformerWeights,
        config: &Config,
        seed_token: usize,
        block_size: usize,
        sampled: &mut [usize],
        argmax: &mut [usize],
        probs: &mut [f32],
        rng: &mut Rng,
    ) {
        let vocab = config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        let mut cur = seed_token;
        for pos in 0..block_size {
            let l = katgpt_forward::forward(ctx, weights, cache, cur, pos, config);
            logits.copy_from_slice(l);
            softmax_scaled(&mut logits, 1.0 / config.temperature);
            sampled[pos] = sample_from_distribution(&logits, rng);
            let (best, best_prob) = simd_argmax_f32(&logits);
            argmax[pos] = best;
            probs[pos] = best_prob;
            cur = sampled[pos];
        }
    }

    struct ArmRow {
        label: &'static str,
        /// Selection rule for Round-1 anchors.
        stride: Option<usize>,
        kappa: f32,
        /// DBTM per-round commit budget (None = threshold-only fill).
        floor_budget: Option<usize>,
    }

    fn select_anchors(
        row: &ArmRow,
        sampled: &[usize],
        argmax: &[usize],
        probs: &[f32],
        mask: usize,
    ) -> Vec<usize> {
        let mut buf = vec![mask; BLOCK];
        match row.stride {
            Some(s) => {
                for p in 0..BLOCK {
                    if (p + 1) % s == 0 {
                        buf[p] = sampled[p];
                    }
                }
            }
            None => {
                for p in 0..BLOCK {
                    if probs[p] >= row.kappa && argmax[p] != mask {
                        buf[p] = argmax[p];
                    }
                }
            }
        }
        buf
    }

    struct ArmOutcome {
        accuracy: f32,
        mean_fill_steps: f32,
        mean_anchors: f32,
        mean_wall_us: f32,
        all_terminated: bool,
    }

    fn run_row(
        row: &ArmRow,
        config: &Config,
        weights: &TransformerWeights,
        test_data: &[Vec<usize>],
        budget: usize,
    ) -> ArmOutcome {
        let mask = config.mask_token;
        let decode_config = D2fDecodeConfig {
            denoise_steps: budget,
            confidence_threshold: row.kappa,
            block_size: BLOCK,
            ..D2fDecodeConfig::default()
        };
        let (mut acc_sum, mut step_sum, mut anchor_sum, mut wall_sum) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut all_terminated = true;
        let mut trials = 0usize;
        for (si, seq) in test_data.iter().take(N_EVAL_SEQUENCES).enumerate() {
            // Paired runs: every row draws the identical rng stream per
            // sequence, so the walk (and thus the selection input) is
            // byte-identical across rows and only the rule differs.
            let mut rng = Rng::new(9_000 + si as u64);
            let mut ctx = ForwardContext::new(config);
            let mut cache = MultiLayerKVCache::new(config);
            let mut dctx = D2fContext::new(config);

            let mut sampled = [0usize; BLOCK];
            let mut argmax = [0usize; BLOCK];
            let mut probs = [0.0f32; BLOCK];
            round1_walk(
                &mut ctx,
                &mut cache,
                weights,
                config,
                seq[0],
                BLOCK,
                &mut sampled,
                &mut argmax,
                &mut probs,
                &mut rng,
            );
            let anchors = select_anchors(row, &sampled, &argmax, &probs, mask);
            let n_anchors = anchors.iter().filter(|&&t| t != mask).count();

            let t0 = std::time::Instant::now();
            let result = anchor_fill_with_prefilled(
                &mut dctx,
                weights,
                config,
                &decode_config,
                &anchors,
                &mut rng,
                row.floor_budget,
            );
            wall_sum += t0.elapsed().as_secs_f32() * 1e6;

            // Ground truth: slot p carries the successor of position p,
            // i.e. seq[p + 1] (the pattern is [a, b, a, b, ...]).
            let mut correct = 0usize;
            let mut terminated = true;
            for (p, &got) in result.tokens.iter().enumerate() {
                if got == mask {
                    terminated = false;
                }
                if got == seq[p + 1] {
                    correct += 1;
                }
            }
            all_terminated &= terminated;
            acc_sum += correct as f32 / BLOCK as f32;
            step_sum += result.steps_used as f32;
            anchor_sum += n_anchors as f32;
            trials += 1;
        }
        ArmOutcome {
            accuracy: acc_sum / trials as f32,
            mean_fill_steps: step_sum / trials as f32,
            mean_anchors: anchor_sum / trials as f32,
            mean_wall_us: wall_sum / trials as f32,
            all_terminated,
        }
    }

    fn build_rows() -> Vec<ArmRow> {
        let ks = [0.5f32, 0.9, 0.99];
        // usize::MAX stride never divides (p+1) — the all-mask baseline.
        const ALL_MASK: usize = usize::MAX;
        let mut rows = vec![ArmRow {
            label: "all-mask D2F baseline",
            stride: Some(ALL_MASK),
            kappa: 0.7,
            floor_budget: None,
        }];
        rows.push(ArmRow {
            label: "stride1 (pure AR anchors)",
            stride: Some(1),
            kappa: 0.7,
            floor_budget: None,
        });
        rows.push(ArmRow {
            label: "stride2 tau0.70 (incumbent)",
            stride: Some(2),
            kappa: 0.7,
            floor_budget: None,
        });
        rows.push(ArmRow {
            label: "stride4 tau0.70",
            stride: Some(4),
            kappa: 0.7,
            floor_budget: None,
        });
        // Matched-threshold stride reference: isolates the SELECTION delta
        // (stride vs confidence) at each swept kappa.
        for &k in &ks {
            rows.push(ArmRow {
                label: "stride2 tau=kappa (matched ref)",
                stride: Some(2),
                kappa: k,
                floor_budget: None,
            });
        }
        // The two DBTM arms.
        for &k in &ks {
            rows.push(ArmRow {
                label: "conf kappa (no floor)",
                stride: None,
                kappa: k,
                floor_budget: None,
            });
        }
        for &k in &ks {
            rows.push(ArmRow {
                label: "conf kappa + floor (DBTM)",
                stride: None,
                kappa: k,
                floor_budget: Some(0), // patched per budget at run time
            });
        }
        rows
    }

    #[test]
    fn test_issue811_harness_matches_production() {
        // Parity guard: the harness stride-arm at incumbent defaults must
        // reproduce the production decode byte-identically (same walk, same
        // anchor buffer, same fill path).
        let (config, weights) = make_trained_weights();
        let seq = {
            let mut rng = Rng::new(555);
            generate_pattern_dataset(&mut rng, 1, config.block_size, config.vocab_size - 1)
                .pop()
                .unwrap()
        };
        let decode_config = D2fDecodeConfig::with_block_size(BLOCK);

        let mut rng_prod = Rng::new(9_000);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut dctx_prod = D2fContext::new(&config);
        let prod = anchor_then_fill(
            &mut ctx,
            &mut cache,
            &mut dctx_prod,
            &weights,
            &config,
            &decode_config,
            &AnchorConfig::with_stride(2),
            seq[0],
            0,
            &mut rng_prod,
        );

        let row = ArmRow {
            label: "parity",
            stride: Some(2),
            kappa: 0.7,
            floor_budget: None,
        };
        let mut rng = Rng::new(9_000);
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let mut dctx = D2fContext::new(&config);
        let mut sampled = [0usize; BLOCK];
        let mut argmax = [0usize; BLOCK];
        let mut probs = [0.0f32; BLOCK];
        round1_walk(
            &mut ctx2,
            &mut cache2,
            &weights,
            &config,
            seq[0],
            BLOCK,
            &mut sampled,
            &mut argmax,
            &mut probs,
            &mut rng,
        );
        let anchors = select_anchors(&row, &sampled, &argmax, &probs, config.mask_token);
        let mine = anchor_fill_with_prefilled(
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &anchors,
            &mut rng,
            None,
        );

        assert_eq!(
            prod.tokens, mine.tokens,
            "harness stride arm must equal production"
        );
    }

    #[test]
    fn test_issue600_entry_matches_poc_conf_arm() {
        // The production confidence-commit entry must reproduce the PoC
        // harness's conf+floor arm exactly: same walk (the observation
        // out-params are behavior-neutral, pinned by the parity test above),
        // same selector, same fill path, same budget.
        let (config, weights) = make_trained_weights();
        let test_data = {
            let mut rng = Rng::new(777);
            generate_pattern_dataset(&mut rng, 4, config.block_size, config.vocab_size - 1)
        };
        let decode_config = D2fDecodeConfig {
            denoise_steps: 8,
            confidence_threshold: 0.9,
            block_size: BLOCK,
            ..D2fDecodeConfig::default()
        };
        let ccfg = ConfidenceAnchorConfig::new(0.9, true);
        let mask = config.mask_token;
        for (si, seq) in test_data.iter().enumerate() {
            // Production entry.
            let mut rng_entry = Rng::new(9_000 + si as u64);
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let mut dctx_entry = D2fContext::new(&config);
            let entry = anchor_then_fill_with(
                &mut ctx,
                &mut cache,
                &mut dctx_entry,
                &weights,
                &config,
                &decode_config,
                &ccfg,
                seq[0],
                0,
                &mut rng_entry,
            );

            // PoC harness arm (walk + selector + fill seam).
            let mut rng = Rng::new(9_000 + si as u64);
            let mut ctx2 = ForwardContext::new(&config);
            let mut cache2 = MultiLayerKVCache::new(&config);
            let mut dctx = D2fContext::new(&config);
            let mut sampled = [0usize; BLOCK];
            let mut argmax = [0usize; BLOCK];
            let mut probs = [0.0f32; BLOCK];
            round1_walk(
                &mut ctx2,
                &mut cache2,
                &weights,
                &config,
                seq[0],
                BLOCK,
                &mut sampled,
                &mut argmax,
                &mut probs,
                &mut rng,
            );
            let (anchors, n_anchors) = select_confidence_anchors(&argmax, &probs, mask, ccfg.kappa);
            let mine = anchor_fill_with_prefilled(
                &mut dctx,
                &weights,
                &config,
                &decode_config,
                &anchors,
                &mut rng,
                Some(8),
            );

            assert_eq!(
                entry.tokens, mine.tokens,
                "entry vs PoC arm diverged at seq {si}"
            );
            assert_eq!(entry.n_anchors, n_anchors);
            assert_eq!(entry.fill_steps_used, mine.steps_used);
        }
    }

    #[test]
    fn test_issue811_arm_table() {
        let (config, weights) = make_trained_weights();
        let mut train_rng = Rng::new(777);
        let test_data = generate_pattern_dataset(
            &mut train_rng,
            N_EVAL_SEQUENCES,
            config.block_size,
            config.vocab_size - 1,
        );
        let budgets = [1usize, 2, 4, 8];
        let mut rows = build_rows();
        println!(
            "\n== Issue 811 arm table ({} seqs, block={}) ==",
            N_EVAL_SEQUENCES, BLOCK
        );
        println!(
            "{:<32} {:>6} {:>8} {:>7} {:>8} {:>7} {:>9}",
            "arm", "kappa", "NFE", "acc", "steps", "anchors", "wall_us"
        );
        for &budget in &budgets {
            for row in rows.iter_mut() {
                if row.label.contains("DBTM") {
                    row.floor_budget = Some(budget);
                }
                let out = run_row(row, &config, &weights, &test_data, budget);
                assert!(
                    out.all_terminated || !row.label.contains("DBTM"),
                    "DBTM arm must terminate within budget: {} k={}",
                    row.label,
                    budget
                );
                println!(
                    "{:<32} {:>6.2} {:>8} {:>7.3} {:>8.2} {:>7.2} {:>9.1}",
                    row.label,
                    row.kappa,
                    budget,
                    out.accuracy,
                    out.mean_fill_steps,
                    out.mean_anchors,
                    out.mean_wall_us
                );
            }
        }
    }

    #[test]
    fn test_issue811_termination_property() {
        // The DBTM floor's termination guarantee, on the real model: at
        // EVERY (kappa, budget) cell the block must be fully unmasked within
        // `budget` fill rounds — including adversarial kappas the sweep does
        // not visit.
        let (config, weights) = make_trained_weights();
        let mut train_rng = Rng::new(777);
        let test_data =
            generate_pattern_dataset(&mut train_rng, 6, config.block_size, config.vocab_size - 1);
        let mask = config.mask_token;
        for &kappa in &[0.5f32, 0.9, 0.99, 0.999] {
            for budget in 1..=8usize {
                let decode_config = D2fDecodeConfig {
                    denoise_steps: budget,
                    confidence_threshold: kappa,
                    block_size: BLOCK,
                    ..D2fDecodeConfig::default()
                };
                for (si, seq) in test_data.iter().enumerate() {
                    let mut rng = Rng::new(9_000 + si as u64);
                    let mut ctx = ForwardContext::new(&config);
                    let mut cache = MultiLayerKVCache::new(&config);
                    let mut dctx = D2fContext::new(&config);
                    let mut sampled = [0usize; BLOCK];
                    let mut argmax = [0usize; BLOCK];
                    let mut probs = [0.0f32; BLOCK];
                    round1_walk(
                        &mut ctx,
                        &mut cache,
                        &weights,
                        &config,
                        seq[0],
                        BLOCK,
                        &mut sampled,
                        &mut argmax,
                        &mut probs,
                        &mut rng,
                    );
                    let row = ArmRow {
                        label: "term",
                        stride: None,
                        kappa,
                        floor_budget: Some(budget),
                    };
                    let anchors = select_anchors(&row, &sampled, &argmax, &probs, mask);
                    let result = anchor_fill_with_prefilled(
                        &mut dctx,
                        &weights,
                        &config,
                        &decode_config,
                        &anchors,
                        &mut rng,
                        Some(budget),
                    );
                    assert!(
                        result.tokens.iter().all(|&t| t != mask),
                        "DBTM floor left a mask at kappa={kappa} budget={budget} seq={si}"
                    );
                }
            }
        }
    }
}

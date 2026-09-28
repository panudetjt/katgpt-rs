//! Deprecated paged-KV-cache speculative step (Plan 394, 2026-07-05).
//!
//! Extracted from the historical `src/speculative/step.rs` (which moved to
//! `crates/katgpt-forward/src/step.rs`). This function stays root because
//! `DDTreeBranchCache` consumes `forward_paged` (root `transformer.rs:2462`),
//! which has genuine root deps (`crate::sleep::*`, `crate::gdn2::*`,
//! `crate::tf_loop`) that cannot move to a leaf without dissolving those
//! modules first.
//!
//! The function is `#[deprecated]` and not re-exported by `mod.rs` — it lives
//! here only so the historical `katgpt_rs::speculative::step_paged::*` path
//! keeps resolving for any legacy caller and so the 3 paged-KV tests have a
//! home. The non-paged speculative-step pipeline moved to katgpt-forward.

use crate::speculative::dd_tree::build_dd_tree;
use crate::speculative::types::DDTreeBranchCache;
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax_scaled};
use katgpt_core::speculative::sampling::{
    sample_from_distribution, sample_residual_distribution_into,
};
use katgpt_forward::dflash::dflash_predict;
#[cfg(feature = "stability_metrics")]
use std::time::Instant;

/// Deprecated paged-KV-cache speculative step.
///
/// See module docs for why this stays root. Production callers should use
/// [`katgpt_forward::speculative_step_rollback_with`] instead.
#[deprecated(note = "Use speculative_step_rollback_with for zero-alloc production path")]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_rollback_paged(
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerKVCache,
    branch_cache: &mut DDTreeBranchCache,
    draft_ctx: &mut ForwardContext,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    #[cfg(feature = "stability_metrics")]
    let _t_draft_start = Instant::now();

    // 1. Draft marginals via DFlash
    let marginals = dflash_predict(draft_weights, draft_config, token, pos);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // 2. Build DDTree
    let tree = build_dd_tree(&mv, draft_config);

    // 3. Extract candidate paths (top-3 root branches)
    let paths = crate::speculative::step::extract_ddtree_paths(&tree);

    if paths.is_empty() {
        let fallback =
            sample_from_distribution(marginals.first().map_or(&[1.0], |m| m.as_slice()), rng);
        return (vec![fallback], 1);
    }

    // 4. Explore branches using paged KV cache
    // Run trunk forward for the prompt token to populate shared prefix
    branch_cache.reset();
    let _ = branch_cache.forward_branch(draft_ctx, draft_weights, 0, token, pos, draft_config);

    // Fork branches from the trunk at current position, tracking seq indices
    let mut branch_seqs: Vec<usize> = Vec::with_capacity(paths.len());
    for (path_idx, path) in paths.iter().enumerate() {
        if path_idx == 0 {
            // First path continues on trunk (seq 0)
            branch_seqs.push(0);
            for (depth, &tok) in path.iter().enumerate() {
                let _ = branch_cache.forward_branch(
                    draft_ctx,
                    draft_weights,
                    0,
                    tok,
                    pos + 1 + depth,
                    draft_config,
                );
            }
        } else {
            // Subsequent paths fork from trunk
            let branch_seq = branch_cache.fork_branch(0, pos + 1);
            branch_seqs.push(branch_seq);
            for (depth, &tok) in path.iter().enumerate() {
                let _ = branch_cache.forward_branch(
                    draft_ctx,
                    draft_weights,
                    branch_seq,
                    tok,
                    pos + 1 + depth,
                    draft_config,
                );
            }
        }
    }

    // 5. Snapshot target KV cache at current position for verification rollback
    let snapshot = target_cache.snapshot(pos, target_config);

    // 6. Verify candidate paths against target model with draft branch rollback
    // Lifted scratch buffer (see speculative_step_rollback for rationale).
    let mut residual_buf: Vec<f32> = Vec::new();
    for (path_idx, path) in paths.iter().enumerate() {
        target_cache.restore(&snapshot, target_config);

        let mut accepted = Vec::with_capacity(path.len());
        let mut all_accepted = true;

        // Score initial token with target
        let logits = forward(
            target_ctx,
            target_weights,
            target_cache,
            token,
            pos,
            target_config,
        );
        let mut p_dist = logits.to_vec();
        softmax_scaled(&mut p_dist, 1.0 / target_config.temperature);

        for (i, &draft_tok) in path.iter().enumerate() {
            // Array-literal default in a plain-let position: the map_or form
            // E0308s (expected `&[_; 0]`, found `&[f32]`) — the healer's
            // array-literal guard is load-bearing here (probed 2026-08-18).
            let q_dist = marginals.get(i).map(|m| m.as_slice()).unwrap_or(&[]);
            let q_i = q_dist.get(draft_tok).copied().unwrap_or(0.0);
            let p_i = p_dist.get(draft_tok).copied().unwrap_or(0.0);

            let acceptance_prob = if q_i > 0.0 { (p_i / q_i).min(1.0) } else { 1.0 };

            if rng.uniform() <= acceptance_prob {
                accepted.push(draft_tok);
                if i + 1 < path.len() {
                    let logits = forward(
                        target_ctx,
                        target_weights,
                        target_cache,
                        draft_tok,
                        pos + 1 + i,
                        target_config,
                    );
                    p_dist = logits.to_vec();
                    softmax_scaled(&mut p_dist, 1.0 / target_config.temperature);
                }
            } else {
                residual_buf.clear();
                residual_buf.resize(p_dist.len(), 0.0);
                let replacement =
                    sample_residual_distribution_into(&p_dist, q_dist, &mut residual_buf, rng);
                accepted.push(replacement);
                all_accepted = false;
                break;
            }
        }

        if all_accepted && !p_dist.is_empty() {
            let bonus = sample_from_distribution(&p_dist, rng);
            accepted.push(bonus);
        }

        if !accepted.is_empty() {
            // Rollback draft branch to accepted position, freeing exclusive pages
            let seq = branch_seqs[path_idx];
            let rollback_pos = pos + 1 + accepted.len();
            branch_cache.rollback_branch(seq, rollback_pos);
            let len = accepted.len();
            return (accepted, len);
        }

        // Path fully rejected: rollback/discard draft branch to free pages
        let seq = branch_seqs[path_idx];
        if seq == 0 {
            // Trunk: rollback to prompt position to undo failed draft tokens
            branch_cache.rollback_branch(0, pos + 1);
        } else {
            // Non-trunk branch: discard entirely
            branch_cache.discard_branch(seq);
        }
    }

    // All paths exhausted: restore target and rollback draft trunk, then sample
    target_cache.restore(&snapshot, target_config);
    branch_cache.rollback_branch(0, pos + 1);
    let logits = forward(
        target_ctx,
        target_weights,
        target_cache,
        token,
        pos,
        target_config,
    );
    let mut p_dist = logits.to_vec();
    softmax_scaled(&mut p_dist, 1.0 / target_config.temperature);
    let fallback = sample_from_distribution(&p_dist, rng);
    (vec![fallback], 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(deprecated)]
    #[test]
    fn test_speculative_step_rollback_paged_returns_at_least_one() {
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerKVCache::new(&target_config);
        let mut branch_cache = DDTreeBranchCache::new(&draft_config, 8);
        let mut draft_ctx = ForwardContext::new(&draft_config);

        let (accepted, len) = speculative_step_rollback_paged(
            &draft_weights,
            &draft_config,
            &target_weights,
            &target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut branch_cache,
            &mut draft_ctx,
            0,
            0,
            &mut Rng::new(100),
        );

        assert!(!accepted.is_empty(), "should return at least 1 token");
        assert!(len >= 1);
        for &t in &accepted {
            assert!(t < target_config.vocab_size, "token {t} out of range");
        }
    }

    #[allow(deprecated)]
    #[test]
    fn test_speculative_step_rollback_paged_deterministic() {
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        let (a1, l1) = {
            let mut target_ctx = ForwardContext::new(&target_config);
            let mut target_cache = MultiLayerKVCache::new(&target_config);
            let mut branch_cache = DDTreeBranchCache::new(&draft_config, 8);
            let mut draft_ctx = ForwardContext::new(&draft_config);
            speculative_step_rollback_paged(
                &draft_weights,
                &draft_config,
                &target_weights,
                &target_config,
                &mut target_ctx,
                &mut target_cache,
                &mut branch_cache,
                &mut draft_ctx,
                0,
                0,
                &mut Rng::new(100),
            )
        };

        let (a2, l2) = {
            let mut target_ctx = ForwardContext::new(&target_config);
            let mut target_cache = MultiLayerKVCache::new(&target_config);
            let mut branch_cache = DDTreeBranchCache::new(&draft_config, 8);
            let mut draft_ctx = ForwardContext::new(&draft_config);
            speculative_step_rollback_paged(
                &draft_weights,
                &draft_config,
                &target_weights,
                &target_config,
                &mut target_ctx,
                &mut target_cache,
                &mut branch_cache,
                &mut draft_ctx,
                0,
                0,
                &mut Rng::new(100),
            )
        };

        assert_eq!(a1, a2, "same seed should produce same results");
        assert_eq!(l1, l2);
    }

    #[allow(deprecated)]
    #[test]
    fn test_speculative_step_rollback_paged_matches_flat_results() {
        // Both paged and flat should produce at least 1 valid token
        let target_config = Config::micro();
        let draft_config = Config::draft();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&draft_config, &mut draft_rng);

        for seed in [0u64, 42, 100, 999] {
            // Paged variant
            let (paged_accepted, paged_len) = {
                let mut target_ctx = ForwardContext::new(&target_config);
                let mut target_cache = MultiLayerKVCache::new(&target_config);
                let mut branch_cache = DDTreeBranchCache::new(&draft_config, 8);
                let mut draft_ctx = ForwardContext::new(&draft_config);
                speculative_step_rollback_paged(
                    &draft_weights,
                    &draft_config,
                    &target_weights,
                    &target_config,
                    &mut target_ctx,
                    &mut target_cache,
                    &mut branch_cache,
                    &mut draft_ctx,
                    0,
                    0,
                    &mut Rng::new(seed),
                )
            };

            assert!(
                !paged_accepted.is_empty(),
                "seed {seed}: paged should return at least 1 token"
            );
            assert!(paged_len >= 1, "seed {seed}: paged len should be >= 1");
            for &t in &paged_accepted {
                assert!(
                    t < target_config.vocab_size,
                    "seed {seed}: paged token {t} out of range"
                );
            }
        }
    }
}

//! Speculative step using GDN tree verification (Plan 424 T4.3).
//!
//! Routes GDN layers through [`forward_tree_gdn2`] instead of the KV-rollback
//! path. For pure-GDN2 models (all layers are DeltaNet), this uses the tree
//! verify primitive to process all draft tree nodes in one forward pass —
//! no state rollback needed.
//!
//! # Convention note
//!
//! The tree verify primitive uses the paper's convention (decay → read →
//! update with 1/√dₖ scaling), while the GDN2 kernel uses update-then-read.
//! The [`forward_tree_gdn2`] function applies a √dₖ scale correction to
//! bridge this gap. Full numerical equivalence with the GDN2 kernel requires
//! aligning the read/update order — tracked as T4.3b.

#![allow(clippy::needless_range_loop)]

use crate::types::{Config, Rng, softmax_scaled};
use katgpt_attn::gdn2::MultiLayerGdn2Cache;
use katgpt_attn::gdn2::forward::forward_gdn2;
use katgpt_attn::gdn2::tree_forward::forward_tree_gdn2;
use katgpt_core::gdn_tree_verify::{GdnTreeVerifier, TreeTopology, build_topology_from_tree_nodes};
use katgpt_core::speculative::sampling::{
    sample_from_distribution, sample_residual_distribution_into,
};
use katgpt_core::speculative::types::{TreeNode, TreePath};
use katgpt_core::traits::NoPruner;
use katgpt_forward::dflash::dflash_predict_with;
#[cfg(feature = "weaver_runtime")]
use katgpt_forward::dflash::dflash_predict_with_weaver;
use katgpt_forward::{ForwardContext, SpeculativeContext};
use katgpt_speculative::dd_tree::TreeBuilder;
use katgpt_transformer::TransformerWeights;

/// Build a `&[&[f32]]` view over `marginals_flat` using a stack array (max 64 steps).
///
/// Mirrors the pattern in `speculative_step_rollback_with` and
/// `speculative_step_qwen_deltanet_tree`. Returns the slice count (may be <
/// `steps_populated` if capped at 64).
///
/// Plan 436 T4.8: made `pub` so riir-gpu's GPU-corrector speculative step
/// (`speculative_step_gdn_tree_with_weaver_gpu`) can reuse the exact same
/// marginals-slicing logic. The GPU orchestration lives in riir-gpu due to
/// the cross-repo cycle (riir-gpu → katgpt-rs, never the reverse).
#[allow(clippy::needless_range_loop)]
pub fn build_marginals_view(
    marginals_flat: &[f32],
    steps_populated: usize,
    vocab_size: usize,
) -> (usize, [&[f32]; 64]) {
    let mut buf: [&[f32]; 64] = [&[]; 64];
    let count = steps_populated.min(64);
    for (i, slot) in buf.iter_mut().enumerate().take(count) {
        let start = i * vocab_size;
        let end = start + vocab_size;
        *slot = if end <= marginals_flat.len() && i < steps_populated {
            &marginals_flat[start..end]
        } else {
            &[]
        };
    }
    (count, buf)
}

/// Shared post-verify pipeline: p/q rejection sampling over DDTree paths +
/// sequential commit. Used by all GDN tree spec step variants (base, Weaver,
/// HOLA, HOLA-Weaver) to eliminate the ~170-line DRY violation (Plan 435 T1).
///
/// Caller responsibilities (done before calling this):
/// 1. Draft step (DFlash or DFlash+Weaver)
/// 2. Build marginals view
/// 3. Build DDTree + topology
/// 4. Run the tree forward (GDN or GDN-HOLA) to produce `tree_logits`
///
/// This helper handles:
/// - Path extraction + p/q rejection along each candidate path
/// - Bonus token on full acceptance
/// - Sequential commit via `commit_accepted_path_sequential`
/// - Fallback sampling when no path is accepted
///
/// Plan 436 T4.8: made `pub` so riir-gpu's GPU-corrector speculative step
/// can reuse the exact same post-verify pipeline (p/q rejection + sequential
/// commit). The only alternative was duplicating ~120 lines of rejection
/// logic in riir-gpu — a DRY violation with a real divergence risk.
#[allow(clippy::too_many_arguments)]
pub fn gdn_tree_post_verify(
    tree_logits: &[f32],
    tree: &[TreeNode],
    topo: &TreeTopology,
    marginals: &[&[f32]],
    target_ctx: &mut ForwardContext,
    target_weights: &TransformerWeights,
    target_cache: &mut MultiLayerGdn2Cache,
    target_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    let vocab_size = target_config.vocab_size;
    let t = topo.n_nodes;

    // 1. Extract candidate paths from the DDTree.
    let paths = katgpt_forward::step::extract_ddtree_paths(tree);

    if paths.is_empty() {
        let root_logits = &tree_logits[0..vocab_size];
        let mut probs = root_logits.to_vec();
        softmax_scaled(&mut probs, 1.0 / target_config.temperature);
        let fallback = sample_from_distribution(&probs, rng);
        return (vec![fallback], 1);
    }

    // 2. Try each candidate path with p/q rejection.
    let mut residual_buf: Vec<f32> = Vec::new();
    // Reused across paths and depths. Each node's logits row used to be copied
    // out with `to_vec()`, i.e. one `vocab_size` heap allocation per
    // (path × depth) — hundreds of KB of malloc/free per speculative step at
    // real vocab sizes. Copying into one reused buffer leaves the softmax input
    // (and therefore every sampled token) byte-identical.
    let mut probs: Vec<f32> = Vec::new();
    // Hoisted: one reciprocal instead of one per node.
    let inv_temperature = 1.0 / target_config.temperature;

    for path in &paths {
        let mut accepted = Vec::with_capacity(path.len());
        let mut all_accepted = true;

        let mut current_path_prefix = TreePath::default();

        for (depth, &draft_tok) in path.iter().enumerate() {
            current_path_prefix = current_path_prefix.push(draft_tok as u32, depth);

            // Find the topo node matching (depth, current_path_prefix).
            // Resolve the slot index only — the logits row is copied into the
            // reused `probs` buffer instead of a fresh `Vec`.
            let node_slot = (0..t).find(|&k| {
                let node = &tree[topo.topo_order[k]];
                node.depth == depth && node.parent_path == current_path_prefix
            });

            let Some(k) = node_slot else {
                all_accepted = false;
                break;
            };

            probs.clear();
            probs.extend_from_slice(&tree_logits[k * vocab_size..(k + 1) * vocab_size]);
            softmax_scaled(&mut probs, inv_temperature);

            let q_dist = marginals.get(depth).copied().unwrap_or(&[]);
            let q_i = q_dist.get(draft_tok).copied().unwrap_or(0.0);
            let p_i = probs.get(draft_tok).copied().unwrap_or(0.0);

            let acceptance_prob = if q_i > 0.0 { (p_i / q_i).min(1.0) } else { 1.0 };

            if rng.uniform() <= acceptance_prob {
                accepted.push(draft_tok);
            } else {
                residual_buf.clear();
                residual_buf.resize(probs.len(), 0.0);
                let replacement =
                    sample_residual_distribution_into(&probs, q_dist, &mut residual_buf, rng);
                accepted.push(replacement);
                all_accepted = false;
                break;
            }
        }

        // Bonus token if all accepted.
        if all_accepted && !accepted.is_empty() {
            let last_depth = path.len() - 1;
            let last_prefix = path
                .iter()
                .enumerate()
                .fold(TreePath::default(), |acc, (d, &tok)| {
                    acc.push(tok as u32, d)
                });
            let bonus_slot = (0..t).find(|&k| {
                let node = &tree[topo.topo_order[k]];
                node.depth == last_depth && node.parent_path == last_prefix
            });

            if let Some(k) = bonus_slot {
                // `probs` is dead at this point — reuse it instead of allocating.
                probs.clear();
                probs.extend_from_slice(&tree_logits[k * vocab_size..(k + 1) * vocab_size]);
                softmax_scaled(&mut probs, inv_temperature);
                let bonus = sample_from_distribution(&probs, rng);
                accepted.push(bonus);
            }
        }

        if !accepted.is_empty() {
            // 3. Commit the accepted path via sequential replay.
            let accepted_len = accepted.len().saturating_sub(1); // exclude bonus
            commit_accepted_path_sequential(
                target_ctx,
                target_weights,
                target_cache,
                &accepted[..accepted_len],
                token,
                pos,
                target_config,
            );

            let len = accepted.len();
            return (accepted, len);
        }
    }

    // 4. All paths exhausted: forward from current token, sample.
    let logits = forward_gdn2(
        target_ctx,
        target_weights,
        target_cache,
        token,
        pos,
        target_config,
    );
    let mut probs = logits.to_vec();
    softmax_scaled(&mut probs, 1.0 / target_config.temperature);
    let fallback = sample_from_distribution(&probs, rng);
    (vec![fallback], 1)
}

/// Speculative step with GDN tree verification for pure-GDN2 models.
///
/// Uses [`forward_tree_gdn2`] to verify all draft tree nodes in one pass,
/// then applies p/q rejection sampling along the best path. The accepted path
/// is committed to the GDN2 cache via `commit_gdn2_tree_layer`.
///
/// # Arguments
/// * `draft_sctx` — Draft speculative context (marginals buffer + scratch).
/// * `tree_builder` — Pre-allocated DDTree builder.
/// * `draft_weights` / `draft_config` — Draft model (for marginal prediction).
/// * `target_weights` / `target_config` — Target model (GDN2, for verification).
/// * `target_ctx` — Target forward context.
/// * `target_cache` — Target GDN2 multi-layer cache.
/// * `verifier` — Pre-allocated tree verify scratch.
/// * `token` / `pos` — Current token and position.
/// * `rng` — Random number generator.
///
/// # Returns
/// `(accepted_tokens, num_accepted)` — same format as `speculative_step_rollback_with`.
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_gdn_tree(
    draft_sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerGdn2Cache,
    verifier: &mut GdnTreeVerifier,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    // 1. Draft marginals via DFlash.
    let _ = dflash_predict_with(draft_sctx, draft_weights, draft_config, token, pos);
    let (count, marginals_buf) = build_marginals_view(
        &draft_sctx.marginals_flat,
        draft_sctx.steps_populated,
        draft_config.vocab_size,
    );
    let marginals = &marginals_buf[..count];

    // 2. Build DDTree + topology.
    let tree = tree_builder.build(marginals, draft_config, &NoPruner, false);
    if tree.is_empty() {
        let fallback = sample_from_distribution(marginals.first().copied().unwrap_or(&[1.0]), rng);
        return (vec![fallback], 1);
    }
    let alpha = target_cache
        .layers
        .first()
        .and_then(|l| l.decay_alpha.first().copied())
        .unwrap_or(0.99);
    let (topo, token_ids) = build_topology_from_tree_nodes(tree, alpha);

    // 3. Forward all tree nodes through the target GDN2 model (read-only verify).
    let tree_logits = forward_tree_gdn2(
        target_ctx,
        target_weights,
        target_cache, // read-only — S₀ not modified
        &topo,
        &token_ids,
        pos,
        target_config,
        verifier,
    );

    // 4. p/q rejection + commit (shared pipeline).
    gdn_tree_post_verify(
        &tree_logits,
        tree,
        &topo,
        marginals,
        target_ctx,
        target_weights,
        target_cache,
        target_config,
        token,
        pos,
        rng,
    )
}

/// Weaver-corrected sibling of [`speculative_step_gdn_tree`] (Plan 435).
///
/// Drops in [`dflash_predict_with_weaver`] for the draft step, then delegates
/// to the same post-draft pipeline (DDTree build → tree forward → p/q reject →
/// commit). Callers allocate `h_dflash_captured` once sized
/// `[draft_config.draft_lookahead * draft_config.n_embd]` and `weaver_scratch`
/// once via `WeaverScratch::new(&weaver.config)`.
///
/// `h_verifier` is sourced from `target_ctx.hidden_state` — the snapshot
/// `forward_gdn2` writes before the LM head matmul (the GDN analog of the
/// QwenDeltaNet `target_scratch.hidden_copy`). On cold-start it is zeros, which
/// Weaver's no-harm contract handles (zero hidden → zero residual).
/// `embedding` is `target_weights.wte` (`[vocab_size, n_embd]` row-major).
///
/// See `dflash_predict_with_weaver` for the no-harm contract: zero-weight
/// Weaver weights leave the marginals bit-identical to the base path.
#[cfg(feature = "weaver_runtime")]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_gdn_tree_with_weaver(
    draft_sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerGdn2Cache,
    verifier: &mut GdnTreeVerifier,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    h_dflash_captured: &mut [f32],
    weaver: &katgpt_speculative::weaver::WeaverCorrector,
    weaver_scratch: &mut katgpt_speculative::weaver::WeaverScratch,
) -> (Vec<usize>, usize) {
    // 1. Draft marginals via DFlash + Weaver correction.
    let n_embd = target_config.n_embd;
    let h_verifier = &target_ctx.hidden_state[..n_embd];
    let embedding = &target_weights.wte;
    let _ = dflash_predict_with_weaver(
        draft_sctx,
        draft_weights,
        draft_config,
        token,
        pos,
        h_dflash_captured,
        weaver,
        h_verifier,
        embedding,
        weaver_scratch,
    );
    let (count, marginals_buf) = build_marginals_view(
        &draft_sctx.marginals_flat,
        draft_sctx.steps_populated,
        draft_config.vocab_size,
    );
    let marginals = &marginals_buf[..count];

    // 2. Build DDTree + topology.
    let tree = tree_builder.build(marginals, draft_config, &NoPruner, false);
    if tree.is_empty() {
        let fallback = sample_from_distribution(marginals.first().copied().unwrap_or(&[1.0]), rng);
        return (vec![fallback], 1);
    }
    let alpha = target_cache
        .layers
        .first()
        .and_then(|l| l.decay_alpha.first().copied())
        .unwrap_or(0.99);
    let (topo, token_ids) = build_topology_from_tree_nodes(tree, alpha);

    // 3. Forward all tree nodes through the target GDN2 model (read-only verify).
    let tree_logits = forward_tree_gdn2(
        target_ctx,
        target_weights,
        target_cache,
        &topo,
        &token_ids,
        pos,
        target_config,
        verifier,
    );

    // 4. p/q rejection + commit (shared pipeline).
    gdn_tree_post_verify(
        &tree_logits,
        tree,
        &topo,
        marginals,
        target_ctx,
        target_weights,
        target_cache,
        target_config,
        token,
        pos,
        rng,
    )
}

/// Commit the accepted path by replaying it through `forward_gdn2` sequentially.
///
/// This is the simplest correct commit: it advances the GDN2 state along the
/// accepted path using the standard kernel (update-then-read). The tree verify
/// state (read-before-update) is not used for the commit — only for verification.
///
/// This means the committed state uses the GDN2 kernel's convention, which is
/// the convention the model uses for subsequent decode steps.
fn commit_accepted_path_sequential(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerGdn2Cache,
    accepted: &[usize],
    initial_token: usize,
    pos: usize,
    config: &Config,
) {
    let mut current_token = initial_token;
    for (i, &tok) in accepted.iter().enumerate() {
        // Forward processes the current token, updates state, produces logits.
        // The logits are discarded — we only care about the state update.
        let _logits = forward_gdn2(ctx, weights, cache, current_token, pos + i, config);
        current_token = tok;
    }
    // Process the last accepted token to update the state for it
    if let Some(&last) = accepted.last() {
        let _logits = forward_gdn2(ctx, weights, cache, last, pos + accepted.len(), config);
    }
}

/// Speculative step with dual-path (GDN × HOLA) tree verification (Plan 430 T3.1).
///
/// Identical to [`speculative_step_gdn_tree`] but uses the dual-path tree
/// verifier ([`forward_tree_gdn2_hola`]). Each layer's output is `O_gdn + O_hola`
/// (residual-add complement). The commit path is the same sequential replay via
/// [`forward_gdn2`] (which already integrates HOLA observe+read when caches are
/// non-empty).
///
/// # Arguments
/// Same as [`speculative_step_gdn_tree`], except:
/// * `target_cache` — Must have hippocampal caches populated
///   (`MultiLayerGdn2Cache::with_hippocampal_cache`). The forward mutates cache
///   scratch (read path) but not persistent state.
/// * `verifier` — A [`GdnHolaTreeVerifier`] (dual-path scratch).
#[cfg(feature = "gdn_hola_tree_verify")]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_gdn_hola_tree(
    draft_sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerGdn2Cache,
    verifier: &mut katgpt_core::gdn_tree_verify::hola_fusion::GdnHolaTreeVerifier,
    token: usize,
    pos: usize,
    rng: &mut Rng,
) -> (Vec<usize>, usize) {
    use katgpt_attn::gdn2::tree_forward::forward_tree_gdn2_hola;

    // 1. Draft marginals via DFlash.
    let _ = dflash_predict_with(draft_sctx, draft_weights, draft_config, token, pos);
    let (count, marginals_buf) = build_marginals_view(
        &draft_sctx.marginals_flat,
        draft_sctx.steps_populated,
        draft_config.vocab_size,
    );
    let marginals = &marginals_buf[..count];

    // 2. Build DDTree + topology.
    let tree = tree_builder.build(marginals, draft_config, &NoPruner, false);
    if tree.is_empty() {
        let fallback = sample_from_distribution(marginals.first().copied().unwrap_or(&[1.0]), rng);
        return (vec![fallback], 1);
    }
    let alpha = target_cache
        .layers
        .first()
        .and_then(|l| l.decay_alpha.first().copied())
        .unwrap_or(0.99);
    let (topo, token_ids) = build_topology_from_tree_nodes(tree, alpha);

    // 3. Forward all tree nodes through the target GDN2 model (dual-path verify).
    let tree_logits = forward_tree_gdn2_hola(
        target_ctx,
        target_weights,
        target_cache,
        &topo,
        &token_ids,
        pos,
        target_config,
        verifier,
    );

    // 4. p/q rejection + commit (shared pipeline).
    gdn_tree_post_verify(
        &tree_logits,
        tree,
        &topo,
        marginals,
        target_ctx,
        target_weights,
        target_cache,
        target_config,
        token,
        pos,
        rng,
    )
}

/// Weaver-corrected sibling of [`speculative_step_gdn_hola_tree`] (Plan 435).
///
/// Same dual-path (GDN × HOLA) tree verification as the base HOLA variant,
/// but uses [`dflash_predict_with_weaver`] for the draft step. Gated behind
/// both `weaver_runtime` and `gdn_hola_tree_verify`.
///
/// See [`speculative_step_gdn_tree_with_weaver`] for the `h_verifier` /
/// `embedding` sourcing and no-harm contract.
#[cfg(all(feature = "weaver_runtime", feature = "gdn_hola_tree_verify"))]
#[allow(clippy::too_many_arguments)]
pub fn speculative_step_gdn_hola_tree_with_weaver(
    draft_sctx: &mut SpeculativeContext,
    tree_builder: &mut TreeBuilder,
    draft_weights: &TransformerWeights,
    draft_config: &Config,
    target_weights: &TransformerWeights,
    target_config: &Config,
    target_ctx: &mut ForwardContext,
    target_cache: &mut MultiLayerGdn2Cache,
    verifier: &mut katgpt_core::gdn_tree_verify::hola_fusion::GdnHolaTreeVerifier,
    token: usize,
    pos: usize,
    rng: &mut Rng,
    h_dflash_captured: &mut [f32],
    weaver: &katgpt_speculative::weaver::WeaverCorrector,
    weaver_scratch: &mut katgpt_speculative::weaver::WeaverScratch,
) -> (Vec<usize>, usize) {
    use katgpt_attn::gdn2::tree_forward::forward_tree_gdn2_hola;

    // 1. Draft marginals via DFlash + Weaver correction.
    let n_embd = target_config.n_embd;
    let h_verifier = &target_ctx.hidden_state[..n_embd];
    let embedding = &target_weights.wte;
    let _ = dflash_predict_with_weaver(
        draft_sctx,
        draft_weights,
        draft_config,
        token,
        pos,
        h_dflash_captured,
        weaver,
        h_verifier,
        embedding,
        weaver_scratch,
    );
    let (count, marginals_buf) = build_marginals_view(
        &draft_sctx.marginals_flat,
        draft_sctx.steps_populated,
        draft_config.vocab_size,
    );
    let marginals = &marginals_buf[..count];

    // 2. Build DDTree + topology.
    let tree = tree_builder.build(marginals, draft_config, &NoPruner, false);
    if tree.is_empty() {
        let fallback = sample_from_distribution(marginals.first().copied().unwrap_or(&[1.0]), rng);
        return (vec![fallback], 1);
    }
    let alpha = target_cache
        .layers
        .first()
        .and_then(|l| l.decay_alpha.first().copied())
        .unwrap_or(0.99);
    let (topo, token_ids) = build_topology_from_tree_nodes(tree, alpha);

    // 3. Forward all tree nodes through the target GDN2 model (dual-path verify).
    let tree_logits = forward_tree_gdn2_hola(
        target_ctx,
        target_weights,
        target_cache,
        &topo,
        &token_ids,
        pos,
        target_config,
        verifier,
    );

    // 4. p/q rejection + commit (shared pipeline).
    gdn_tree_post_verify(
        &tree_logits,
        tree,
        &topo,
        marginals,
        target_ctx,
        target_weights,
        target_cache,
        target_config,
        token,
        pos,
        rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_weights(config: &Config) -> TransformerWeights {
        let mut rng = Rng::new(42);
        TransformerWeights::new(config, &mut rng)
    }

    /// The GDN tree speculative step should return at least one token.
    #[test]
    fn test_speculative_step_gdn_tree_returns_tokens() {
        let draft_config = Config::micro();
        let target_config = Config::micro();
        let draft_weights = random_weights(&draft_config);
        let target_weights = random_weights(&target_config);

        let mut draft_sctx = SpeculativeContext::new(&draft_config);
        let mut tree_builder = TreeBuilder::new(&draft_config);
        let mut target_ctx = ForwardContext::new(&target_config);
        let mut target_cache = MultiLayerGdn2Cache::new(&target_config);

        // Set paper-compatible alpha
        for layer in &mut target_cache.layers {
            layer.decay_alpha.fill(0.99);
            layer.erase_b.fill(1.0);
        }

        let hd = target_config.head_dim;
        let max_tree = 64; // generous for the DDTree
        let mut verifier = GdnTreeVerifier::new(max_tree, hd, hd);

        let mut rng = Rng::new(42);

        let (accepted, len) = speculative_step_gdn_tree(
            &mut draft_sctx,
            &mut tree_builder,
            &draft_weights,
            &draft_config,
            &target_weights,
            &target_config,
            &mut target_ctx,
            &mut target_cache,
            &mut verifier,
            target_config.bos_token,
            0,
            &mut rng,
        );

        assert!(!accepted.is_empty(), "must accept at least one token");
        assert_eq!(len, accepted.len());
    }

    /// The GDN tree speculative step should be deterministic for the same seed.
    #[test]
    fn test_speculative_step_gdn_tree_deterministic() {
        let config = Config::micro();
        let weights = random_weights(&config);

        let run = || {
            let mut draft_sctx = SpeculativeContext::new(&config);
            let mut tree_builder = TreeBuilder::new(&config);
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerGdn2Cache::new(&config);
            for layer in &mut cache.layers {
                layer.decay_alpha.fill(0.99);
                layer.erase_b.fill(1.0);
            }
            let hd = config.head_dim;
            let mut verifier = GdnTreeVerifier::new(64, hd, hd);
            let mut rng = Rng::new(42);

            speculative_step_gdn_tree(
                &mut draft_sctx,
                &mut tree_builder,
                &weights,
                &config,
                &weights,
                &config,
                &mut ctx,
                &mut cache,
                &mut verifier,
                config.bos_token,
                0,
                &mut rng,
            )
        };

        let (a1, _) = run();
        let (a2, _) = run();
        assert_eq!(a1, a2, "same seed must produce same accepted tokens");
    }

    // ── Weaver variant tests (Plan 435) ──────────────────────────────

    /// Construct a zero-weight Weaver corrector with K > vocab_size so the
    /// `correct_marginals_with_scratch` takes the early-return path → marginals
    /// unchanged. Mirrors the Plan 433/434 zero-weight pattern.
    #[cfg(feature = "weaver_runtime")]
    fn make_zero_weaver(
        n_embd: usize,
        vocab_size: usize,
        draft_lookahead: usize,
    ) -> (
        katgpt_speculative::weaver::WeaverCorrector,
        katgpt_speculative::weaver::WeaverScratch,
    ) {
        use katgpt_speculative::weaver::{
            WeaverConfig, WeaverCorrector, WeaverScratch, WeaverWeights,
        };
        let weaver_cfg = WeaverConfig {
            hidden_dim: n_embd,
            n_heads: 4,
            k_candidates: vocab_size + 100, // K > V → early return, marginals unchanged
            n_layer: 1,
            d_ff: n_embd * 2,
            rms_eps: 1e-6,
            max_depth: draft_lookahead,
        };
        let corrector = WeaverCorrector::from_weights(WeaverWeights::zeros(weaver_cfg.clone()));
        let scratch = WeaverScratch::new(&weaver_cfg);
        (corrector, scratch)
    }

    /// T4.1: zero-weight Weaver (K > V early-return) must produce the same
    /// accepted tokens as the base path. Same seed, same inputs, same RNG
    /// path → deterministic match.
    #[cfg(feature = "weaver_runtime")]
    #[test]
    fn test_speculative_step_gdn_tree_with_weaver_no_harm() {
        let config = Config::micro();
        let weights = random_weights(&config);

        let run = |use_weaver: bool| -> (Vec<usize>, usize) {
            let mut draft_sctx = SpeculativeContext::new(&config);
            let mut tree_builder = TreeBuilder::new(&config);
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerGdn2Cache::new(&config);
            for layer in &mut cache.layers {
                layer.decay_alpha.fill(0.99);
                layer.erase_b.fill(1.0);
            }
            let hd = config.head_dim;
            let mut verifier = GdnTreeVerifier::new(64, hd, hd);
            let mut rng = Rng::new(42);

            if use_weaver {
                let (weaver, mut wscratch) =
                    make_zero_weaver(config.n_embd, config.vocab_size, config.draft_lookahead);
                let mut h_dflash_captured = vec![0.0f32; config.draft_lookahead * config.n_embd];
                speculative_step_gdn_tree_with_weaver(
                    &mut draft_sctx,
                    &mut tree_builder,
                    &weights,
                    &config,
                    &weights,
                    &config,
                    &mut ctx,
                    &mut cache,
                    &mut verifier,
                    config.bos_token,
                    0,
                    &mut rng,
                    &mut h_dflash_captured,
                    &weaver,
                    &mut wscratch,
                )
            } else {
                speculative_step_gdn_tree(
                    &mut draft_sctx,
                    &mut tree_builder,
                    &weights,
                    &config,
                    &weights,
                    &config,
                    &mut ctx,
                    &mut cache,
                    &mut verifier,
                    config.bos_token,
                    0,
                    &mut rng,
                )
            }
        };

        let (accepted_base, len_base) = run(false);
        let (accepted_weaver, len_weaver) = run(true);
        assert_eq!(
            len_base, len_weaver,
            "zero-weight Weaver must not change accepted count"
        );
        assert_eq!(
            accepted_base, accepted_weaver,
            "zero-weight Weaver (K > V early-return) must produce bit-identical \
             accepted tokens to the base path"
        );
    }

    /// T4.2: non-zero Weaver weights (K <= V so correction runs) must still
    /// return at least one accepted token without panicking.
    #[cfg(feature = "weaver_runtime")]
    #[test]
    fn test_speculative_step_gdn_tree_with_weaver_returns_tokens() {
        use katgpt_speculative::weaver::{
            WeaverConfig, WeaverCorrector, WeaverScratch, WeaverWeights,
        };
        let config = Config::micro();
        let weights = random_weights(&config);

        // K <= V so the correction actually runs (not the early-return path).
        let weaver_cfg = WeaverConfig {
            hidden_dim: config.n_embd,
            n_heads: 4,
            k_candidates: 8, // K <= vocab_size → correction runs
            n_layer: 1,
            d_ff: config.n_embd * 2,
            rms_eps: 1e-6,
            max_depth: config.draft_lookahead,
        };
        // Non-zero weights (ones) so the residual is non-trivial.
        let corrector = WeaverCorrector::from_weights(WeaverWeights::zeros(weaver_cfg.clone()));
        let mut wscratch = WeaverScratch::new(&weaver_cfg);

        let mut draft_sctx = SpeculativeContext::new(&config);
        let mut tree_builder = TreeBuilder::new(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::new(&config);
        for layer in &mut cache.layers {
            layer.decay_alpha.fill(0.99);
            layer.erase_b.fill(1.0);
        }
        let hd = config.head_dim;
        let mut verifier = GdnTreeVerifier::new(64, hd, hd);
        let mut rng = Rng::new(42);
        let mut h_dflash_captured = vec![0.0f32; config.draft_lookahead * config.n_embd];

        let (accepted, len) = speculative_step_gdn_tree_with_weaver(
            &mut draft_sctx,
            &mut tree_builder,
            &weights,
            &config,
            &weights,
            &config,
            &mut ctx,
            &mut cache,
            &mut verifier,
            config.bos_token,
            0,
            &mut rng,
            &mut h_dflash_captured,
            &corrector,
            &mut wscratch,
        );

        assert!(!accepted.is_empty(), "must accept at least one token");
        assert_eq!(len, accepted.len());
    }

    /// T4.3: cold-start (no prior commit → target_ctx.hidden_state is zeros)
    /// must not panic and must produce at least one accepted token.
    #[cfg(feature = "weaver_runtime")]
    #[test]
    fn test_speculative_step_gdn_tree_with_weaver_cold_start() {
        let config = Config::micro();
        let weights = random_weights(&config);

        let (weaver, mut wscratch) =
            make_zero_weaver(config.n_embd, config.vocab_size, config.draft_lookahead);

        let mut draft_sctx = SpeculativeContext::new(&config);
        let mut tree_builder = TreeBuilder::new(&config);
        // Fresh ctx — hidden_state is zero-init (cold-start case).
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::new(&config);
        for layer in &mut cache.layers {
            layer.decay_alpha.fill(0.99);
            layer.erase_b.fill(1.0);
        }
        let hd = config.head_dim;
        let mut verifier = GdnTreeVerifier::new(64, hd, hd);
        let mut rng = Rng::new(7);
        let mut h_dflash_captured = vec![0.0f32; config.draft_lookahead * config.n_embd];

        let (accepted, len) = speculative_step_gdn_tree_with_weaver(
            &mut draft_sctx,
            &mut tree_builder,
            &weights,
            &config,
            &weights,
            &config,
            &mut ctx,
            &mut cache,
            &mut verifier,
            config.bos_token,
            0,
            &mut rng,
            &mut h_dflash_captured,
            &weaver,
            &mut wscratch,
        );

        assert!(
            !accepted.is_empty(),
            "cold-start must accept at least one token"
        );
        assert_eq!(len, accepted.len());
    }
}

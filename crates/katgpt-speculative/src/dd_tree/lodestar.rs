//! Lodestar completion-distance pruning (extracted from mod.rs by Issue 178).
//!
//! Plan 207, Research 183 — A* completion-distance-pruned DDTree builder.
//! The tree builder lives here; the pruner + automaton (LodestarPruner,
//! LodestarAutomaton) live in katgpt-pruners. LodestarConfig lives here
//! because it configures the builder (A* lambda + jump-ahead).

use std::collections::BinaryHeap;

use katgpt_core::speculative::types::{TreeNode, TreePath};
use katgpt_core::traits::CompletionHorizon;

use super::extract_parent_tokens_into;

// ── Lodestar completion-distance pruning (Plan 207, Research 183) ────────
//
// The tree builder lives in this crate (it composes the heap walk); the
// pruner + automaton (`LodestarPruner`, `LodestarAutomaton`) live in
// `katgpt-pruners`. `LodestarConfig` lives here because it configures the
// *builder* (A* lambda + jump-ahead), not the pruner. katgpt-pruners
// re-exports it for back-compat with the historical `katgpt_rs::pruners::`
// path. Gated behind the `lodestar` feature.

/// Configuration for [`build_dd_tree_lodestar`] — controls A* ordering and jump-ahead.
///
/// Default reproduces pure log-prob best-first (λ = 0, jump-ahead disabled).
#[cfg(feature = "lodestar")]
#[derive(Clone, Copy, Debug)]
pub struct LodestarConfig {
    /// A* distance weight λ. Heap key = `score − λ·d(s)`.
    /// λ = 0 (default) → pure log-prob ordering, byte-identical to `build_dd_tree_pruned`.
    /// λ > 0 → prefer branches closer to completion (A* admissible heuristic).
    pub astar_lambda: f32,
    /// Enable jump-ahead: collapse singular spans into one tree node.
    /// When `true`, deterministic forced paths are emitted as a single expansion step
    /// instead of per-token, reducing tree nodes and speeding up traversal.
    pub jump_ahead: bool,
}

#[cfg(feature = "lodestar")]
impl Default for LodestarConfig {
    fn default() -> Self {
        Self {
            astar_lambda: 0.0,
            jump_ahead: false,
        }
    }
}

#[cfg(feature = "lodestar")]
impl LodestarConfig {
    /// Pure log-prob ordering, no jump-ahead (default).
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// A* ordering with jump-ahead enabled.
    pub fn thinking(lambda: f32) -> Self {
        Self {
            astar_lambda: lambda,
            jump_ahead: true,
        }
    }
}

/// DDTree with Lodestar Completion-Distance Pruning (Plan 207, Research 183).
///
/// Identical to [`build_dd_tree_pruned`] (best-first, no chain seed) **plus**:
/// - **(A) Budget-aware mask**: a candidate token is admitted only if, after placing it,
///   the [`CompletionHorizon`]'s shortest-accepting-distance fits within the remaining
///   sequence slots. Token budget = `marginals.len()`.
/// - **(B) Jump-ahead** (when `lode_config.jump_ahead`): deterministic singular spans
///   are collapsed into a single tree node, reducing tree nodes and speeding traversal.
/// - **(C) A\* ordering** (when `lode_config.astar_lambda > 0`): heap key = `score − λ·d(s)`,
///   preferring branches closer to completion. λ = 0 reproduces pure log-prob.
///
/// # Guarantee (TRUNCPROOF)
///
/// Every branch the tree retains can be completed to a valid output within the
/// sequence length — no branch is "painted into a corner". When the horizon is a
/// [`NoPruner`] (or any pruner using the default-0 `min_completion_distance`),
/// the mask is a no-op and this reduces to [`build_dd_tree_pruned`].
///
/// Feature-gated behind `lodestar`.
#[cfg(feature = "lodestar")]
pub fn build_dd_tree_lodestar(
    marginals: &[&[f32]],
    config: &katgpt_types::Config,
    horizon: &dyn CompletionHorizon,
    lode_config: &LodestarConfig,
) -> Vec<TreeNode> {
    let lambda = lode_config.astar_lambda;
    let jump_ahead = lode_config.jump_ahead;
    let mut heap: BinaryHeap<TreeNode> = BinaryHeap::new();
    let mut tree: Vec<TreeNode> = Vec::with_capacity(config.tree_budget);
    if marginals.is_empty() {
        return tree;
    }
    let seq_len = marginals.len();
    let mut parent_buf: Vec<usize> = vec![0usize; seq_len];
    // Reusable scratch for jump-ahead span walk; avoids per-iteration allocation.
    let mut span_parents_buf: Vec<usize> = Vec::with_capacity(seq_len);
    // Lazily-filled per-depth log-marginal cache (same shape as
    // `TreeBuilder::cache_log_marginals`). The expansion loop below calls
    // `prob.ln()` once per vocab entry on EVERY heap-pop, but the value depends
    // only on `(depth, token)` — never on the popped node — so with a 32K vocab
    // and a 64-node budget that is ~2M redundant `ln()` calls per build. Filling
    // one table per depth on first expansion of that depth makes it one `ln()`
    // per `(depth, token)`. Filled with the SAME `if p > 0.0 { p.ln() } else
    // { 0.0 }` rule, and read only under the identical `prob > 0.0` guard, so
    // the `0.0` filler for non-positive probs is never observed.
    let mut log_marginals: Vec<Vec<f32>> = vec![Vec::new(); seq_len];

    // Seed root children (depth 0, empty parent). After placing a token at depth
    // 0, remaining slots = seq_len - 1.
    for_each_candidate(horizon, 0, &[], marginals[0], |i, prob| {
        let d = horizon.min_completion_distance(0, i, &[]);
        if d != u32::MAX && (d as usize) < seq_len {
            let score = prob.ln();
            heap.push(TreeNode {
                score: a_star_score(score, lambda, d),
                depth: 0,
                token_idx: i,
                parent_path: TreePath::root(i as u32),
            });
        }
    });

    // Best-first expansion with budget mask + optional jump-ahead + A*.
    while tree.len() < config.tree_budget {
        let Some(best) = heap.pop() else {
            break;
        };
        tree.push(best);

        if best.depth + 1 < seq_len {
            let start_depth = best.depth + 1;
            let parent_tokens =
                extract_parent_tokens_into(best.parent_path, best.depth + 1, &mut parent_buf);

            // (B) Jump-ahead: if the horizon reports a singular span, collapse it.
            if jump_ahead {
                let span = horizon.singular_span_len(start_depth, parent_tokens);
                // Cap span to fit the 8-slot TreePath (one token per level).
                let max_span = 8usize.saturating_sub(best.depth + 1);
                let span = span.min(max_span as u32);
                if span > 0 {
                    let end_depth = start_depth + span as usize;
                    if end_depth <= seq_len {
                        // Walk the span: collect forced tokens, accumulate log-prob.
                        // Reuse span_parents_buf: clear + refill avoids per-pop allocation.
                        let mut span_score = best.score;
                        let mut span_path = best.parent_path;
                        let mut span_depth = start_depth;
                        span_parents_buf.clear();
                        span_parents_buf.extend_from_slice(parent_tokens);
                        let mut valid = true;

                        for _ in 0..span {
                            if span_depth >= seq_len {
                                valid = false;
                                break;
                            }
                            let forced = find_forced_token(
                                marginals,
                                span_depth,
                                &span_parents_buf,
                                horizon,
                                seq_len - span_depth - 1,
                            );
                            if let Some((token, prob)) = forced {
                                let d = horizon.min_completion_distance(
                                    span_depth,
                                    token,
                                    &span_parents_buf,
                                );
                                if d == u32::MAX || (d as usize) > seq_len - span_depth - 1 {
                                    valid = false;
                                    break;
                                }
                                span_score += prob.ln();
                                span_path = span_path.push(token as u32, span_depth);
                                span_parents_buf.push(token);
                                span_depth += 1;
                            } else {
                                valid = false;
                                break;
                            }
                        }

                        if valid && span_depth <= seq_len {
                            let d = if span_depth < seq_len {
                                let post_parents = extract_parent_tokens_into(
                                    span_path,
                                    span_depth,
                                    &mut parent_buf,
                                );
                                horizon.min_completion_distance(span_depth, 0, post_parents)
                            } else {
                                0
                            };
                            heap.push(TreeNode {
                                score: a_star_score(span_score, lambda, d),
                                depth: span_depth - 1,
                                token_idx: span_path.token_at(span_depth - 1) as usize,
                                parent_path: span_path,
                            });
                        }
                        continue;
                    }
                }
            }

            // Standard per-token expansion with budget mask.
            let remaining_after = seq_len - start_depth - 1;
            let marginal = marginals[start_depth];
            let log_m = &mut log_marginals[start_depth];
            if log_m.is_empty() {
                log_m.reserve(marginal.len());
                for &p in marginal {
                    log_m.push(if p > 0.0 { p.ln() } else { 0.0 });
                }
            }
            // NEURO-SYMBOLIC INTERCEPT + LODESTAR BUDGET MASK
            for_each_candidate(horizon, start_depth, parent_tokens, marginal, |i, _prob| {
                let d = horizon.min_completion_distance(start_depth, i, parent_tokens);
                if d != u32::MAX && (d as usize) <= remaining_after {
                    // `log_m[i]` == `prob.ln()` here (the guard is `prob > 0.0`).
                    let score = best.score + log_m[i];
                    heap.push(TreeNode {
                        score: a_star_score(score, lambda, d),
                        depth: start_depth,
                        token_idx: i,
                        parent_path: best.parent_path.push(i as u32, start_depth),
                    });
                }
            });
        }
    }

    // A* offset is left in the scores: downstream consumers use relative
    // comparisons (heap ordering), so the offset is harmless and consistent
    // with build_dd_tree_pruned's score convention. When λ = 0 this is a no-op.
    tree.shrink_to_fit();
    tree
}

/// Compute A*-adjusted score: `score − λ·d`.
/// When λ = 0, returns `score` unchanged.
#[cfg(feature = "lodestar")]
#[inline]
fn a_star_score(score: f32, lambda: f32, d: u32) -> f32 {
    if lambda == 0.0 || d == u32::MAX {
        score
    } else {
        score - lambda * d as f32
    }
}

/// Visit every `(token, prob)` this position could expand into.
///
/// Two implementations of one question. The fallback asks the horizon about
/// each of the `vocab_size` tokens in turn — and `is_valid` re-walks the
/// prefix inside every one of those calls, so a heap pop costs
/// O(vocab × prefix). The fast path asks the horizon to NAME its legal set
/// instead, which a structural pruner can do in O(deg) from its own state.
///
/// **Bit-identical, not merely equivalent.** Both visit ascending token ids —
/// the scan by construction, the enumeration by the
/// [`ConstraintPruner::for_each_legal`] contract — so the heap receives the
/// same pushes in the same order, and `f32` scores accumulate the same way.
/// The `prob > 0.0` guard and the `marginal` bound stay on BOTH paths: the
/// legal set is a property of the grammar and says nothing about this
/// position's marginal, which may be shorter than the automaton's alphabet.
///
/// [`ConstraintPruner::for_each_legal`]: katgpt_core::traits::ConstraintPruner::for_each_legal
#[cfg(all(feature = "lodestar", feature = "legal_token_set"))]
#[inline]
fn for_each_candidate<F: FnMut(usize, f32)>(
    horizon: &dyn CompletionHorizon,
    depth: usize,
    parent_tokens: &[usize],
    marginal: &[f32],
    mut f: F,
) {
    let Some(degree) = horizon.legal_degree(depth, parent_tokens) else {
        for (i, &prob) in marginal.iter().enumerate() {
            if prob > 0.0 && horizon.is_valid(depth, i, parent_tokens) {
                f(i, prob);
            }
        }
        return;
    };
    let mut yielded = 0usize;
    horizon.for_each_legal(depth, parent_tokens, &mut |i| {
        yielded += 1;
        if let Some(&prob) = marginal.get(i)
            && prob > 0.0
        {
            f(i, prob);
        }
    });
    debug_assert_eq!(
        yielded, degree,
        "ConstraintPruner contract: legal_degree said {degree}, for_each_legal yielded {yielded}"
    );
}

/// The scan, when the legal-set seam is not compiled in.
#[cfg(all(feature = "lodestar", not(feature = "legal_token_set")))]
#[inline]
fn for_each_candidate<F: FnMut(usize, f32)>(
    horizon: &dyn CompletionHorizon,
    depth: usize,
    parent_tokens: &[usize],
    marginal: &[f32],
    mut f: F,
) {
    for (i, &prob) in marginal.iter().enumerate() {
        if prob > 0.0 && horizon.is_valid(depth, i, parent_tokens) {
            f(i, prob);
        }
    }
}

/// Find the single forced token at `depth` given `parent_tokens`.
/// Returns `Some((token_idx, prob))` if there is exactly one valid token
/// that passes both the validity and budget checks; `None` otherwise.
#[cfg(feature = "lodestar")]
fn find_forced_token(
    marginals: &[&[f32]],
    depth: usize,
    parent_tokens: &[usize],
    horizon: &dyn CompletionHorizon,
    budget_remaining: usize,
) -> Option<(usize, f32)> {
    let marginal = marginals.get(depth)?;
    let mut found: Option<(usize, f32)> = None;
    let mut branched = false;
    for_each_candidate(horizon, depth, parent_tokens, marginal, |i, prob| {
        if branched {
            return;
        }
        let d = horizon.min_completion_distance(depth, i, parent_tokens);
        if d != u32::MAX && (d as usize) <= budget_remaining {
            match found {
                // More than one valid token — not forced.
                Some(_) => branched = true,
                None => found = Some((i, prob)),
            }
        }
    });
    match branched {
        true => None,
        false => found,
    }
}

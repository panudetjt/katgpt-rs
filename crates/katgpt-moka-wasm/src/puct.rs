//! PUCT (Predictor + UCB applied to Trees) search — the AlphaZero recipe,
//! ported from `katgpt_pruners::go::moka_net::GoPuctMokaPlayer` (Plan 205) for
//! the standalone WASM build (Issue 204). Same algorithm, same vendored
//! weights, same forward pass — only the board type changed (this crate's
//! minimal `Board` instead of the full `GoState` engine).
//!
//! Exists to answer the measurement question the prior doc sidestepped:
//! does combining PUCT (98% strength, native) with WASM SIMD (0.5 ms/pass,
//! browser) produce a build that is BOTH strong AND fast, or does the
//! combination invert both headlines? See `.issues/204` and the
//! `go_arena.md` Table-A-vs-Table-B framing.
//!
//! Kept in sync by construction with the native PUCT implementation: this
//! is the same search loop, not a reimplementation. The native crate's
//! parity testing (Bench 205) covers the algorithm; this crate's job is
//! the latency measurement under WASM.

use crate::board::{AREA as BOARD_AREA, Board, SIZE as BOARD_SIZE, flood_group};
use crate::moka::{self, MokaBatchScratch, MokaScratch, MokaWeights};
use crate::moka_int8::{MokaScratchInt8, MokaWeightsInt8};

/// A move: `Some(idx)` = play at board index `idx`; `None` = pass.
/// Replaces `GoAction` from the native crate.
type Move = Option<usize>;

struct PuctNode {
    /// Move that led to this node. `None` (= pass) for root by convention.
    action: Move,
    /// Board state at this node (cloned on expansion — the dominant non-SIMD
    /// cost in WASM, since `Vec<Cell>` heap-allocates per clone).
    state: Board,
    /// Visit count.
    visits: u32,
    /// Accumulated value from the perspective of the player who MOVED INTO
    /// this node (i.e., the parent's to_play). Negamax: negate at each level.
    total_value: f32,
    /// Policy prior P(s,a) from the parent's policy head evaluation.
    prior: f32,
    /// Arena indices of children.
    children: Vec<usize>,
    /// Arena index of parent. `None` for root.
    parent: Option<usize>,
    /// Whether this node has been expanded (policy+value evaluated, children
    /// created, or flagged terminal).
    expanded: bool,
    /// Virtual loss applied during batched selection (Issue 205). Non-zero
    /// only while a leaf is in the current batch's queue — drives diverse
    /// leaf selection within a single batch by temporarily penalizing the
    /// path another in-flight simulation already chose. Cleared at backprop.
    /// Zero in the K=1 sequential path (never touched).
    virtual_loss: f32,
}

impl PuctNode {
    fn new_root(state: Board) -> Self {
        Self {
            action: None,
            state,
            visits: 0,
            total_value: 0.0,
            prior: 1.0,
            children: Vec::new(),
            parent: None,
            expanded: false,
            virtual_loss: 0.0,
        }
    }

    /// Mean value including the virtual loss penalty. Used during batched
    /// selection so the same path isn't picked twice in one batch: each
    /// in-flight leaf adds `VIRTUAL_LOSS` to its ancestors' `virtual_loss`,
    /// which makes the node look like it was visited with a WINNING result
    /// for the child (= bad for the parent to keep exploring, since negamax
    /// flips the sign at selection time). The parent's `q = -effective_mean_value`
    /// therefore drops, steering the next selection within the batch to a
    /// different sibling. Cleared at backprop.
    #[inline]
    fn effective_mean_value(&self) -> f32 {
        let denom = self.visits as f32 + self.virtual_loss;
        if denom == 0.0 {
            0.0
        } else {
            // +virtual_loss (not -): inflates the child's apparent value,
            // which the parent's `q = -eff` then reads as a penalty
            // (avoid this child for now).
            (self.total_value + self.virtual_loss) / denom
        }
    }
}

/// Standalone PUCT player. Mirrors `GoPuctMokaPlayer` field-for-field minus
/// the `GoPlayer` trait plumbing (this crate has no such trait).
pub struct PuctPlayer {
    weights: MokaWeights,
    scratch: MokaScratch,
    /// int8 weights + scratch — `None` for f32-only mode, `Some` when int8
    /// forward path is enabled (Issue 206 T5). When present, `expand` uses
    /// `forward_int8_with_scratch` instead of the f32 path. Only the K=1
    /// sequential path supports int8 — batched MCTS (K>1) always uses f32
    /// (batched int8 forward is not implemented).
    weights_i8: Option<MokaWeightsInt8>,
    scratch_i8: Option<MokaScratchInt8>,
    /// Batched scratch — only allocated if `batch_k > 1` (lazy, on first
    /// batched `select_move`). Sized for `batch_k`.
    batch_scratch: Option<MokaBatchScratch>,
    budget: usize,
    c_puct: f32,
    top_k: usize,
    /// Batch size K for batched MCTS (Issue 205). K=1 disables batching and
    /// runs the original sequential loop (the wasmi parity path — bit
    /// identical move choices vs the pre-batch code). K>1 enables the
    /// leaf-queue + virtual-loss + batched forward pass.
    batch_k: usize,
    /// Reused across `select_move` calls — avoids re-allocating the arena.
    arena: Vec<PuctNode>,
    /// Reused scratch for feature encoding (avoids per-expansion alloc).
    /// Sized `batch_k * INPUT_ELEMENT_COUNT` so the batched path can encode
    /// K leaves in one contiguous buffer without re-allocating.
    features_buf: Vec<f32>,
    /// Per-sample output buffers for the batched forward pass — owned by the
    /// player so they're allocated once at construction, not per `select_move`.
    policy_batch_buf: Vec<f32>,
    value_batch_buf: Vec<f32>,
    nodes_evaluated: usize,
    /// Corrected-forward state (Issue 565 G5). When present, `expand` uses
    /// `research::forward_corrected_with_scratch` instead of the f32/int8
    /// path. This enables the quant-error-compensating LoRA win-rate test:
    /// the player simulates the ternary+LoRA path by running f32 weights +
    /// a correction matrix that encodes the net (ternary + SVD_r(E)) weights.
    /// Gated behind `research` (PoC-only — production never enables this).
    #[cfg(feature = "research")]
    corrected: Option<crate::research::OwnedCorrections>,
    #[cfg(feature = "research")]
    lora_scratch: Vec<f32>,
    /// Engram-fused memory (Issue 868 / Plan 605). `None` = the feature-off
    /// posture — bit-identical search. Only [`with_engram`](Self::with_engram)
    /// sets it, and only at K=1 (batched + engram is out of POC scope, the
    /// int8-at-K>1 precedent).
    #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
    engram: Option<crate::engram_fuse::EngramPuctMemory>,
}

/// Virtual loss magnitude applied during batched selection (Issue 205).
/// A single in-flight leaf subtracts this much from each of its ancestors'
/// effective Q, discouraging the next selection within the same batch from
/// walking the same path. The value is large enough to dominate the UCB
/// exploration term (typical magnitudes ~1.0) but small enough not to
/// permanently distort the tree — it's cleared at backprop.
const VIRTUAL_LOSS: f32 = 1.0;

impl PuctPlayer {
    pub fn new(budget: usize, c_puct: f32, top_k: usize) -> Self {
        Self::with_batch_k(budget, c_puct, top_k, 1)
    }

    /// Construct with explicit batch size K (Issue 205). K=1 uses the int8
    /// forward path by default (Issue 207 promotion — GOAT-validated: 1.17–1.39×
    /// speedup, 85% win-rate parity vs f32's 94% native reference at n=20).
    /// K>1 enables batched MCTS (virtual loss + leaf queue + batched forward
    /// pass, always f32 — batched int8 is not implemented).
    ///
    /// To force the f32 path at K=1 (regression testing, platforms without
    /// int8 dot support), use [`with_f32`](Self::with_f32).
    pub fn with_batch_k(budget: usize, c_puct: f32, top_k: usize, batch_k: usize) -> Self {
        let k = batch_k.max(1);
        // Issue 207 promotion: int8 is the default for the sequential (K=1)
        // path. The win-rate parity gate (wasmi_puct_int8_winrate, 85% at
        // n=20) confirmed this is a modelless gain, not a speedup of a worse
        // result. K>1 still uses f32 (batched int8 forward is unimplemented).
        Self::with_options(budget, c_puct, top_k, batch_k, k == 1)
    }

    /// Construct with the **f32** forward path (K=1 sequential). Use this when
    /// you need the original f32 path explicitly — regression tests against
    /// the pre-int8 baseline, platforms without int8 dot kernels, or A/B
    /// comparison. For normal use, prefer [`new`](Self::new) (which defaults
    /// to int8 at K=1 per Issue 207).
    pub fn with_f32(budget: usize, c_puct: f32, top_k: usize) -> Self {
        Self::with_options(budget, c_puct, top_k, 1, false)
    }

    /// Construct with int8 forward path enabled explicitly. After the Issue 207
    /// promotion this is equivalent to [`new`](Self::new) — kept as an explicit
    /// alias for code that wants to spell out "int8 path" at the call site
    /// (e.g. A/B test harnesses, perf benches that compare against `with_f32`).
    ///
    /// Uses platform-native int8 dot kernels (SDOT on aarch64, extmul on
    /// wasm32). Only the K=1 sequential path supports int8 — if `batch_k > 1`,
    /// int8 is silently ignored (batched MCTS always uses f32 forward).
    pub fn with_int8(budget: usize, c_puct: f32, top_k: usize) -> Self {
        Self::with_options(budget, c_puct, top_k, 1, true)
    }

    fn with_options(
        budget: usize,
        c_puct: f32,
        top_k: usize,
        batch_k: usize,
        use_int8: bool,
    ) -> Self {
        let k = batch_k.max(1);
        let enable_int8 = use_int8 && k == 1;
        Self {
            weights: MokaWeights::load(),
            scratch: MokaScratch::new(),
            weights_i8: enable_int8.then(MokaWeightsInt8::load),
            scratch_i8: enable_int8.then(MokaScratchInt8::new),
            batch_scratch: (k > 1).then(|| MokaBatchScratch::new(k)),
            budget: budget.max(1),
            c_puct,
            top_k: top_k.max(1),
            batch_k: k,
            arena: Vec::new(),
            features_buf: vec![0.0; k * moka::INPUT_ELEMENT_COUNT],
            policy_batch_buf: vec![0.0; k * moka::POLICY_MOVES],
            value_batch_buf: vec![0.0; k],
            nodes_evaluated: 0,
            #[cfg(feature = "research")]
            corrected: None,
            #[cfg(feature = "research")]
            lora_scratch: Vec::new(),
            #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
            engram: None,
        }
    }

    /// Construct with the corrected-forward path enabled (Issue 565 G5).
    /// The player runs `research::forward_corrected_with_scratch` on every
    /// leaf evaluation, applying the supplied correction bundle. This is the
    /// PoC vehicle for the quant-error-compensating LoRA win-rate test: the
    /// caller builds the corrections (encoding ternary + weight-space SVD of
    /// the error) and the player simulates the net behavior end-to-end.
    ///
    /// Always K=1 (sequential — the corrected path is not batched). The int8
    /// path is disabled (the corrected path uses f32 weights + corrections).
    /// Gated behind `research` — PoC-only, never in the WASM browser build.
    #[cfg(feature = "research")]
    pub fn with_corrected(
        budget: usize,
        c_puct: f32,
        top_k: usize,
        corrections: crate::research::OwnedCorrections,
    ) -> Self {
        let mut player = Self::with_options(budget, c_puct, top_k, 1, false);
        // Size lora_scratch for the largest correction matrix's rank. The
        // Correction::Full path uses out_dim×in_dim but doesn't touch
        // lora_scratch; the Dense/Sparse paths do. A safe default: 64 (the
        // max rank the PoC ever uses). The Full path ignores this buffer.
        player.lora_scratch = vec![0.0f32; 64];
        player.corrected = Some(corrections);
        player
    }

    #[inline]
    pub fn nodes_evaluated(&self) -> usize {
        self.nodes_evaluated
    }

    /// Construct with engram-fused memory (Issue 868 / Plan 605 — Proposal
    /// 013's offline-mined value/prior table read at the expansion seam).
    /// Forces the K=1 sequential path (batched + engram is out of POC scope,
    /// the int8-at-K>1 precedent) with the default int8 forward.
    ///
    /// Fusion rule (see `engram_fuse`): at each expansion, ONE engram read
    /// keyed on the expanded node's own `(board, ko, to_play)` —
    /// Q-init as one damped pseudo-visit (`visits = 1`,
    /// `total_value = gate · v̄`) + prior sharpening of the child softmax by
    /// the bounded γ. `hits == 0` behaves exactly like the feature-off
    /// player (G1).
    #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
    pub fn with_engram(
        budget: usize,
        c_puct: f32,
        top_k: usize,
        memory: crate::engram_fuse::EngramPuctMemory,
    ) -> Self {
        let mut player = Self::with_options(budget, c_puct, top_k, 1, true);
        player.engram = Some(memory);
        player
    }

    /// Fusion telemetry: `(lookups, fires)` from the fused memory, if
    /// present. Diagnostics for the arena report ("memory never fired" is
    /// the diagnosability the T1.3 gates exist to protect).
    #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
    pub fn engram_telemetry(&self) -> Option<(u64, u64)> {
        self.engram.as_ref().map(|m| m.telemetry())
    }

    /// Root children's `(action, visit_count)` pairs from the most recent
    /// `select_move`, in child order. Mining/diagnostics only — this
    /// allocates and is not a hot-path API. The engram miner consumes it
    /// for the concentration statistic `b` (Issue 868 T1).
    #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
    pub fn root_child_visits(&self) -> Vec<(Option<usize>, u32)> {
        match self.arena.first() {
            Some(root) => root
                .children
                .iter()
                .map(|&c| (self.arena[c].action, self.arena[c].visits))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Expand a node: run policy+value, create children for top_k legal moves.
    /// Returns the value-head evaluation [-1,1] from this node's to_play
    /// perspective (or the exact terminal reward mapped onto the same range).
    ///
    /// K=1 path — the original sequential loop. The batched path uses
    /// `prepare_leaf_for_eval` + `expand_with_policy_value` instead.
    fn expand(&mut self, node_idx: usize) -> f32 {
        // Collect parent chain BEFORE mutable borrow (Moka needs last-2-plies
        // history). Each ancestor's `action` is the move that produced it.
        let mut hist: Vec<Option<(usize, usize)>> = Vec::with_capacity(2);
        {
            let mut chain_actions: Vec<Move> = Vec::with_capacity(2);
            let mut cur = Some(node_idx);
            while let Some(idx) = cur {
                if chain_actions.len() >= 2 {
                    break;
                }
                let n = &self.arena[idx];
                if n.parent.is_some() {
                    chain_actions.push(n.action);
                }
                cur = n.parent;
            }
            for a in chain_actions.iter().rev() {
                hist.push((*a).map(|idx| (idx / BOARD_SIZE, idx % BOARD_SIZE)));
            }
        }

        let node = &mut self.arena[node_idx];
        node.expanded = true;

        if node.state.is_game_over() {
            // Map {0,1} reward onto [-1,+1] for the value-head's tanh range.
            return 2.0 * node.state.reward(node.state.to_play) - 1.0;
        }

        // Snapshot everything we need from `node`, then drop the mutable borrow
        // before re-entering the arena (children push needs `&mut self.arena`).
        // `Board` is `Copy` ([Cell; 81] not Vec) — this is now a stack copy, no alloc.
        let parent_state = node.state;

        // Encode features from the parent position + reconstructed history,
        // then run ONE forward pass. Reuses the persistent `features_buf` so
        // no allocation happens here (Board is Copy, so parent_state was a
        // stack copy above — zero heap alloc in this hot path).
        moka::encode_features_into(&parent_state, &hist, &mut self.features_buf);
        let (policy, value) = self.forward_leaf();
        self.nodes_evaluated += 1;

        // Issue 868 fusion (Plan 605): ONE engram read for this node's own
        // state — Q-init as one damped pseudo-visit + the child-softmax
        // sharpening γ. `hits == 0` ⇒ exactly the feature-off path (G1).
        // Native-gated only; the fused player is forced to K=1, and this is
        // the K=1 expansion path.
        #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
        let gamma = match self.engram.as_mut() {
            Some(mem) => match mem.read(&self.arena[node_idx].state) {
                Some(row) => {
                    let node = &mut self.arena[node_idx];
                    node.visits = 1;
                    node.total_value = row.gate * row.value_mean;
                    row.gamma
                }
                None => 1.0,
            },
            None => 1.0,
        };

        #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
        self.expand_with_policy_value(node_idx, &policy, value, gamma);
        #[cfg(not(all(feature = "engram_puct", not(target_arch = "wasm32"))))]
        self.expand_with_policy_value(node_idx, &policy, value);
        value
    }

    /// Run the forward pass on `features_buf` using the active path (corrected /
    /// int8 / f32). The corrected path takes priority when present (Issue 565 G5);
    /// otherwise the int8 path is used when int8 weights are loaded; otherwise
    /// the f32 path. Returns `(policy_logits[POLICY_MOVES], value_tanh)`.
    #[inline]
    fn forward_leaf(&mut self) -> ([f32; moka::POLICY_MOVES], f32) {
        #[cfg(feature = "research")]
        if let Some(corrections) = self.corrected.as_ref() {
            return crate::research::forward_corrected_with_scratch(
                &self.weights,
                &self.features_buf,
                &mut self.scratch,
                &corrections.forward_corrections(),
                &mut self.lora_scratch,
            );
        }
        if let (Some(w), Some(s)) = (self.weights_i8.as_ref(), self.scratch_i8.as_mut()) {
            crate::moka_int8::forward_int8_with_scratch(w, &self.features_buf, s)
        } else {
            moka::forward_with_scratch(&self.weights, &self.features_buf, &mut self.scratch)
        }
    }

    /// Prepare a single leaf for batched evaluation: mark expanded, handle
    /// the terminal short-circuit, encode features into the per-sample slice
    /// of `features_buf`. Returns `Some(())` if the leaf needs a forward pass
    /// (the caller batches all such leaves), or `None` if the leaf is terminal
    /// (value already known — recorded via `terminal_value_out`).
    ///
    /// `sample_idx` selects the slice of `features_buf` to write into
    /// (`[sample_idx * INPUT_ELEMENT_COUNT ..]`).
    fn prepare_leaf_for_eval(
        &mut self,
        node_idx: usize,
        sample_idx: usize,
        terminal_value_out: &mut [Option<f32>],
    ) {
        // Collect parent chain (last-2-plies history) BEFORE mutable borrow.
        let mut hist: Vec<Option<(usize, usize)>> = Vec::with_capacity(2);
        {
            let mut chain_actions: Vec<Move> = Vec::with_capacity(2);
            let mut cur = Some(node_idx);
            while let Some(idx) = cur {
                if chain_actions.len() >= 2 {
                    break;
                }
                let n = &self.arena[idx];
                if n.parent.is_some() {
                    chain_actions.push(n.action);
                }
                cur = n.parent;
            }
            for a in chain_actions.iter().rev() {
                hist.push((*a).map(|idx| (idx / BOARD_SIZE, idx % BOARD_SIZE)));
            }
        }

        let node = &mut self.arena[node_idx];
        node.expanded = true;

        if node.state.is_game_over() {
            // Terminal: value is the exact reward (no forward pass needed).
            terminal_value_out[sample_idx] =
                Some(2.0 * node.state.reward(node.state.to_play) - 1.0);
            return;
        }

        let parent_state = node.state;
        let feat_off = sample_idx * moka::INPUT_ELEMENT_COUNT;
        moka::encode_features_into(&parent_state, &hist, &mut self.features_buf[feat_off..]);
        terminal_value_out[sample_idx] = None;
    }

    /// Expand a node's children given its already-evaluated policy + value.
    /// Shared between the K=1 `expand` path and the batched path — identical
    /// child-creation logic so both paths produce bit-identical tree shapes
    /// given the same (policy, value) inputs.
    #[inline]
    fn expand_with_policy_value(
        &mut self,
        node_idx: usize,
        policy: &[f32],
        _value: f32,
        // Issue 868: prior-sharpening factor for the child softmax, fused
        // path only. γ > 0 preserves the top-k sort order — only the
        // softmax below is scaled.
        #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))] gamma: f32,
    ) {
        let parent_state = self.arena[node_idx].state;

        // Rank moves by raw policy logit, keep top_k (always including pass as
        // a candidate — Moka's policy index BOARD_AREA is pass). Build `scored`
        // directly from the legal-move iterator (no intermediate Vec allocation).
        let mut scored: Vec<(f32, Move)> = (0..BOARD_AREA)
            .filter(|&i| parent_state.is_legal(i))
            .map(|i| (policy[i], Some(i)))
            .collect();
        scored.push((policy[BOARD_AREA], None));
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(self.top_k);

        // Softmax the top_k priors for normalized P(s,a).
        // Issue 868: the paired arms below are cfg-TWINS — identical modulo
        // the γ multiply. The γ-free arm must stay the exact pre-fusion
        // expression (the T2.1 byte-identity claim rests on it).
        let max_logit = scored
            .iter()
            .map(|(l, _)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
        let exp_sum: f32 = scored
            .iter()
            .map(|(l, _)| (gamma * (l - max_logit)).exp())
            .sum();
        #[cfg(not(all(feature = "engram_puct", not(target_arch = "wasm32"))))]
        let exp_sum: f32 = scored.iter().map(|(l, _)| (l - max_logit).exp()).sum();
        let inv_exp_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 1.0 };

        // Push children.
        let children_start = self.arena.len();
        for (logit, action) in &scored {
            #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
            let prior = (gamma * (logit - max_logit)).exp() * inv_exp_sum;
            #[cfg(not(all(feature = "engram_puct", not(target_arch = "wasm32"))))]
            let prior = (logit - max_logit).exp() * inv_exp_sum;
            let mut child_state = parent_state;
            match *action {
                Some(idx) => child_state.play(idx),
                None => child_state.pass(),
            }
            self.arena.push(PuctNode {
                action: *action,
                state: child_state,
                visits: 0,
                total_value: 0.0,
                prior,
                children: Vec::new(),
                parent: Some(node_idx),
                expanded: false,
                virtual_loss: 0.0,
            });
        }
        let children_end = self.arena.len();
        self.arena[node_idx]
            .children
            .extend(children_start..children_end);
    }

    /// Selection: traverse from root to first unexpanded leaf using PUCT.
    /// Returns the leaf node index.
    fn select(&self, root: usize) -> usize {
        let mut cur = root;
        loop {
            let node = &self.arena[cur];
            if !node.expanded || node.children.is_empty() {
                return cur;
            }
            cur = self.pick_best_child(cur);
        }
    }

    /// Pick the child with the highest PUCT score from node `cur`. Used by
    /// both the K=1 `select` and the batched selection. Honors `virtual_loss`
    /// through `effective_mean_value` — nodes penalized by an in-flight leaf
    /// in the current batch score lower.
    #[inline]
    fn pick_best_child(&self, cur: usize) -> usize {
        let node = &self.arena[cur];
        let parent_visits = node.visits.max(1) as f32;
        let sqrt_parent = parent_visits.sqrt();
        let mut best_idx = node.children[0];
        let mut best_score = f32::NEG_INFINITY;
        for &child_idx in &node.children {
            let child = &self.arena[child_idx];
            // Q is negated because child.total_value is from child.to_play's
            // perspective, but we're selecting from parent's perspective.
            let q = -child.effective_mean_value();
            let u = self.c_puct * child.prior * sqrt_parent / (1.0 + child.visits as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_idx = child_idx;
            }
        }
        best_idx
    }

    /// Backpropagate value from leaf to root. Negamax: negate at each level.
    fn backprop(&mut self, leaf_idx: usize, mut value: f32) {
        let mut cur = Some(leaf_idx);
        while let Some(idx) = cur {
            let node = &mut self.arena[idx];
            node.visits += 1;
            node.total_value += value;
            value = -value; // negate for parent (opponent's perspective)
            cur = node.parent;
        }
    }

    /// Batched selection: walk from root to first unexpanded leaf, applying
    /// `VIRTUAL_LOSS` to every ancestor of the leaf as we go. The virtual
    /// loss depresses the next selection's Q for these nodes, so the next
    /// leaf in the batch is less likely to walk the same path. Returns the
    /// leaf index. Caller backprops to clear the loss.
    fn select_with_virtual_loss(&mut self, root: usize) -> usize {
        let mut cur = root;
        loop {
            // Apply virtual loss to `cur` BEFORE descending (root included —
            // a batch with all leaves under the same root child still needs
            // the root itself penalized so the second selection diverges).
            self.arena[cur].virtual_loss += VIRTUAL_LOSS;
            let next = {
                let node = &self.arena[cur];
                if !node.expanded || node.children.is_empty() {
                    break;
                }
                self.pick_best_child(cur)
            };
            cur = next;
        }
        cur
    }

    /// Backprop + clear virtual loss along the path. Used by the batched
    /// path — each leaf's path had `VIRTUAL_LOSS` added per ancestor during
    /// `select_with_virtual_loss`; we subtract it back here as we propagate
    /// the real value. The real `visits` + `total_value` updates use the
    /// evaluated value (terminal reward or value-head output).
    fn backprop_clearing_virtual_loss(&mut self, leaf_idx: usize, mut value: f32) {
        let mut cur = Some(leaf_idx);
        while let Some(idx) = cur {
            let node = &mut self.arena[idx];
            node.virtual_loss = (node.virtual_loss - VIRTUAL_LOSS).max(0.0);
            node.visits += 1;
            node.total_value += value;
            value = -value;
            cur = node.parent;
        }
    }

    /// Run a full PUCT search from `state` and return the chosen move
    /// (`Some(idx)` or `None` for pass). Mirrors `GoPuctMokaPlayer::select_move`
    /// minus the trait plumbing. If `state` has no legal non-pass move AND pass
    /// would be pointless, returns `None` (pass).
    ///
    /// Dispatches on `batch_k`: K=1 runs the original sequential loop
    /// (wasmi parity path — bit-identical move choices vs pre-batch code).
    /// K>1 runs the batched loop (leaf queue + virtual loss + batched
    /// forward pass).
    pub fn select_move(&mut self, state: &Board) -> Move {
        // Reset arena for this move's search.
        self.arena.clear();
        self.arena.push(PuctNode::new_root(*state));
        let root = 0;

        if self.batch_k <= 1 {
            self.select_move_sequential(root);
        } else {
            self.select_move_batched(root);
        }

        // Pick most-visited child at root (AlphaZero convention).
        let root_node = &self.arena[root];
        let mut best_action: Move = None;
        let mut best_visits = 0u32;
        for &child_idx in &root_node.children {
            let child = &self.arena[child_idx];
            if child.visits > best_visits {
                best_visits = child.visits;
                best_action = child.action;
            }
        }

        best_action
    }

    /// K=1 path — the original sequential loop. Bit-identical to the
    /// pre-batch code (wasmi parity guarantee).
    fn select_move_sequential(&mut self, root: usize) {
        for _ in 0..self.budget {
            let leaf = self.select(root);
            let value = self.expand(leaf);
            self.backprop(leaf, value);
        }
    }

    /// K>1 path — batched MCTS. Collects up to `batch_k` leaves (each with
    /// virtual loss applied), runs ONE batched forward pass for all
    /// non-terminal leaves, then expands + backprops each leaf. The batch
    /// count is the ceiling of `budget / batch_k` rounded to fill the budget
    /// exactly (final batch may be partial).
    ///
    /// **Root-first convention:** the root is expanded synchronously BEFORE
    /// the batch loop starts (1 forward pass). This is required because an
    /// unexpanded root would be returned by every selection in the first
    /// batch (virtual loss has no effect — the root has no siblings to
    /// compete with). Expanding root first means every batched leaf is at
    /// depth >= 1, where virtual loss can drive diverse sibling selection.
    /// This mirrors the standard AlphaZero batched-MCTS implementation.
    fn select_move_batched(&mut self, root: usize) {
        let k = self.batch_k;

        // Phase 0: expand root synchronously. Counts as 1 budget unit.
        let root_value = self.expand(root);
        self.backprop(root, root_value);
        let mut remaining = self.budget.saturating_sub(1);

        // Scratch: per-sample terminal-value slot (Some = terminal short-
        // circuit, None = needs forward pass). Sized once, reused per batch.
        let mut terminal_values: Vec<Option<f32>> = vec![None; k];
        // Leaf indices for this batch, paired with sample_idx.
        let mut leaves: Vec<(usize, usize)> = Vec::with_capacity(k);

        while remaining > 0 {
            let batch = k.min(remaining);
            remaining -= batch;

            // 1. Collect `batch` leaves via virtual-loss selection. Each leaf
            //    gets its features encoded into `features_buf` at sample_idx,
            //    OR is marked terminal in `terminal_values`.
            leaves.clear();
            for sample_idx in 0..batch {
                let leaf = self.select_with_virtual_loss(root);
                self.prepare_leaf_for_eval(leaf, sample_idx, &mut terminal_values);
                leaves.push((leaf, sample_idx));
            }

            // 2. Run ONE batched forward pass. Terminal samples' features
            //    are stale (from a previous batch) but their output is
            //    ignored — only the non-terminal samples' policy/value are
            //    used. This wastes some FLOPs on terminal samples but keeps
            //    the code simple; compaction is a deferred optimization.
            let bs = self
                .batch_scratch
                .as_mut()
                .expect("batch_scratch allocated when batch_k > 1");
            moka::forward_batch_with_scratch(
                &self.weights,
                &self.features_buf,
                batch,
                bs,
                &mut self.policy_batch_buf,
                &mut self.value_batch_buf,
            );
            self.nodes_evaluated += batch;

            // 3. Expand + backprop each leaf. If two leaves converged on the
            //    same node (rare — virtual loss usually prevents it), the
            //    second expansion sees `expanded == true` and skips child
            //    creation, but still backprops the value.
            for &(leaf_idx, sample_idx) in &leaves {
                let value = match terminal_values[sample_idx] {
                    Some(tv) => tv, // terminal: value is exact reward
                    None => {
                        // Non-terminal: use the batched policy + value.
                        // Copy the per-sample policy into a stack array so we
                        // can call `expand_with_policy_value` (which borrows
                        // `&mut self.arena`) without holding an immutable
                        // borrow of `self.policy_batch_buf`.
                        let mut policy_buf = [0f32; moka::POLICY_MOVES];
                        policy_buf.copy_from_slice(
                            &self.policy_batch_buf[sample_idx * moka::POLICY_MOVES
                                ..(sample_idx + 1) * moka::POLICY_MOVES],
                        );
                        let v = self.value_batch_buf[sample_idx];
                        // Guard: if the node was already expanded by a prior
                        // leaf in this batch (convergence), skip child creation.
                        if !self.arena[leaf_idx].children.is_empty()
                            || self.arena[leaf_idx].visits > 0
                        {
                            // Already expanded — just use the value.
                        } else {
                            // Issue 868: batched expansion is fusion-free (the
                            // fused player forces K=1) — γ = 1 keeps the two
                            // arms shape-identical.
                            #[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]
                            self.expand_with_policy_value(leaf_idx, &policy_buf, v, 1.0);
                            #[cfg(not(all(
                                feature = "engram_puct",
                                not(target_arch = "wasm32")
                            )))]
                            self.expand_with_policy_value(leaf_idx, &policy_buf, v);
                        }
                        v
                    }
                };
                self.backprop_clearing_virtual_loss(leaf_idx, value);
            }
        }
    }
}

// `flood_group` is re-exported-by-use from `board`; reference it so the unused
// import warning stays quiet if future changes remove the only other user.
#[allow(dead_code)]
fn _ensure_flood_group_linked() {
    let _ = flood_group;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Cell;

    #[test]
    fn puct_returns_a_legal_move_from_empty_board() {
        let mut player = PuctPlayer::new(50, 1.5, 8);
        let board = Board::new();
        let mv = player.select_move(&board);
        // First move on an empty 9×9 must be a real placement (not pass) —
        // the policy head is trained on real openings, never pass on move 1.
        assert!(mv.is_some(), "PUCT on empty board should not pass");
        assert!(board.is_legal(mv.unwrap()));
        // At least one forward pass per budget iteration (terminal short-
        // circuits don't fire on an empty board).
        assert!(player.nodes_evaluated() > 0);
    }

    #[test]
    fn puct_search_is_deterministic_given_same_board() {
        // PUCT with no RNG in the loop (deterministic selection + argmax
        // final pick) must return the same move for the same input across
        // two runs. This is the WASM-side analog of the native G1 gate.
        let mut player = PuctPlayer::new(50, 1.5, 8);
        let mut board = Board::new();
        // Play a few moves to reach a non-trivial mid-game-ish position.
        board.play(40); // center-ish
        board.play(41);
        board.play(31);
        board.play(50);
        let first = player.select_move(&board);
        // Re-run from identical state — arena was cleared by the second call.
        let second = player.select_move(&board);
        assert_eq!(
            first, second,
            "PUCT must be deterministic given fixed input"
        );
    }

    #[test]
    fn puct_budget_scales_nodes_evaluated() {
        // Sanity that the budget knob actually controls simulation count:
        // budget=100 should evaluate roughly 2× the nodes of budget=50
        // (not exact — terminals short-circuit — but same order of magnitude).
        let mut board = Board::new();
        board.play(40);
        board.play(41);

        let mut p50 = PuctPlayer::new(50, 1.5, 8);
        let _ = p50.select_move(&board);
        let n50 = p50.nodes_evaluated();

        let mut p100 = PuctPlayer::new(100, 1.5, 8);
        let _ = p100.select_move(&board);
        let n100 = p100.nodes_evaluated();

        assert!(
            n100 > n50,
            "budget=100 ({n100}) must exceed budget=50 ({n50})"
        );
        // Allow generous slack for terminal short-circuits in late positions.
        assert!(
            n100 >= 2 * n50 - 30,
            "budget scaling off: 100→{n100}, 50→{n50}"
        );
    }

    #[test]
    fn board_terminal_detection_after_double_pass() {
        let mut board = Board::new();
        assert!(!board.is_game_over());
        board.pass();
        assert!(!board.is_game_over()); // single pass
        board.pass();
        assert!(board.is_game_over()); // double pass ends game
        // A real move resets the consecutive-pass counter.
        let mut board2 = Board::new();
        board2.pass();
        board2.play(0);
        assert!(!board2.is_game_over());
    }

    #[test]
    fn board_reward_signs_are_consistent() {
        // On an empty board with komi 7.5 favoring White, both colors score
        // 0 stones and 0 territory, so White (7.5) is ahead → reward(White)=1,
        // reward(Black)=0.
        let board = Board::new();
        assert_eq!(
            board.reward(Cell::White),
            1.0,
            "empty board: White wins on komi"
        );
        assert_eq!(
            board.reward(Cell::Black),
            0.0,
            "empty board: Black loses on komi"
        );
    }

    // ── Batched MCTS tests (Issue 205) ─────────────────────────────

    #[test]
    fn batched_puct_returns_a_legal_move_from_empty_board() {
        // K=8 batched path must still produce a legal, non-pass move from
        // an empty board — same invariant as the sequential test above.
        let mut player = PuctPlayer::with_batch_k(50, 1.5, 8, 8);
        let board = Board::new();
        let mv = player.select_move(&board);
        assert!(mv.is_some(), "batched PUCT on empty board should not pass");
        assert!(board.is_legal(mv.unwrap()));
        assert!(player.nodes_evaluated() > 0);
    }

    #[test]
    fn batched_puct_search_is_deterministic_given_same_board() {
        // Batched PUCT must also be deterministic — same board → same move
        // across two runs. Virtual loss is cleared between batches, so the
        // tree state at end-of-search is a pure function of the input.
        let mut player = PuctPlayer::with_batch_k(50, 1.5, 8, 8);
        let mut board = Board::new();
        board.play(40);
        board.play(41);
        board.play(31);
        board.play(50);
        let first = player.select_move(&board);
        let second = player.select_move(&board);
        assert_eq!(
            first, second,
            "batched PUCT must be deterministic given fixed input"
        );
    }

    #[test]
    fn batched_puct_budget_scales_nodes_evaluated() {
        // Same scaling invariant as the sequential test, but for K=8 batches.
        // budget=100 should evaluate roughly 2× the nodes of budget=50.
        let mut board = Board::new();
        board.play(40);
        board.play(41);

        let mut p50 = PuctPlayer::with_batch_k(50, 1.5, 8, 8);
        let _ = p50.select_move(&board);
        let n50 = p50.nodes_evaluated();

        let mut p100 = PuctPlayer::with_batch_k(100, 1.5, 8, 8);
        let _ = p100.select_move(&board);
        let n100 = p100.nodes_evaluated();

        assert!(
            n100 > n50,
            "batched budget=100 ({n100}) must exceed budget=50 ({n50})"
        );
        assert!(
            n100 >= 2 * n50 - 30,
            "batched budget scaling off: 100→{n100}, 50→{n50}"
        );
    }

    #[test]
    fn batched_puct_handles_partial_final_batch() {
        // budget=50, K=8 → 6 full batches (48 leaves) + 1 partial batch (2 leaves).
        // The partial batch must not panic and must produce a legal move.
        let mut player = PuctPlayer::with_batch_k(50, 1.5, 8, 8);
        let mut board = Board::new();
        board.play(40);
        board.play(41);
        let mv = player.select_move(&board);
        assert!(
            mv.is_some() || board.is_game_over(),
            "batched PUCT should return a move (or game is over)"
        );
    }

    #[test]
    fn batched_puct_handles_terminal_leaves_in_batch() {
        // A position where some batched leaves hit terminals (double-pass
        // subtrees) must not panic — the terminal short-circuit path uses
        // the exact-reward value, the non-terminal leaves use the forward pass.
        // We construct a near-terminal position: both players have passed once.
        let mut player = PuctPlayer::with_batch_k(50, 1.5, 8, 8);
        let mut board = Board::new();
        board.play(40);
        board.pass();
        board.pass(); // now consecutive_passes == 2 → game over at root
        // Root itself is terminal — the search must handle this without panic
        // and return None (pass) since there are no children to expand.
        let mv = player.select_move(&board);
        assert_eq!(mv, None, "terminal root must return pass (no children)");
    }

    #[test]
    fn batched_puct_explores_diverse_leaves_via_virtual_loss() {
        // Virtual loss should cause the batch to explore DIFFERENT root
        // children, not all pile onto the same one. With K=8 and budget=8
        // (a single batch), the root should have multiple children with
        // visits > 0 (not all 8 visits on one child).
        //
        // We can't assert the exact distribution (depends on priors), but
        // we CAN assert that at least 2 root children received visits —
        // that's the load-bearing virtual-loss invariant. Without virtual
        // loss, all 8 leaves in the first batch would walk the same path
        // (the highest-prior child) and only 1 root child would have visits.
        let mut player = PuctPlayer::with_batch_k(8, 1.5, 8, 8);
        let mut board = Board::new();
        board.play(40);
        board.play(41);
        let _ = player.select_move(&board);

        let root = &player.arena[0];
        let visited_children = root
            .children
            .iter()
            .filter(|&&idx| player.arena[idx].visits > 0)
            .count();
        assert!(
            visited_children >= 2,
            "virtual loss must cause diverse exploration: only {visited_children} root child(ren) visited (expected ≥2)"
        );
    }

    // ── int8 PUCT tests (Issue 206 T5) ───────────────────────────────

    #[test]
    fn int8_puct_returns_a_legal_move_from_empty_board() {
        let mut player = PuctPlayer::with_int8(50, 1.5, 8);
        let board = Board::new();
        let mv = player.select_move(&board);
        assert!(mv.is_some(), "int8 PUCT on empty board should not pass");
        assert!(board.is_legal(mv.unwrap()));
        assert!(player.nodes_evaluated() > 0);
    }

    #[test]
    fn int8_puct_search_is_deterministic_given_same_board() {
        // Two searches from the same board must produce the same move —
        // the int8 forward path is deterministic (no RNG in quantization).
        let mut board = Board::new();
        board.play(40);
        board.play(41);

        let mut p1 = PuctPlayer::with_int8(50, 1.5, 8);
        let mv1 = p1.select_move(&board);

        let mut p2 = PuctPlayer::with_int8(50, 1.5, 8);
        let mv2 = p2.select_move(&board);

        assert_eq!(mv1, mv2, "int8 PUCT must be deterministic");
    }

    /// G1 gate: int8 PUCT should produce the same move as f32 PUCT on
    /// representative positions. This is the end-to-end version of
    /// `moka_int8::g1_int8_matches_f32_baseline` — it checks that the
    /// quantization noise doesn't change move selection through the full
    /// MCTS search (not just the raw argmax).
    #[test]
    fn g1_int8_puct_matches_f32_move_selection() {
        let positions: &[&[usize]] = &[&[40, 41, 31, 50], &[0, 1, 9, 10, 18], &[80, 79, 71, 70]];

        for &moves in positions {
            let mut board = Board::new();
            for &mv in moves {
                if board.is_legal(mv) {
                    board.play(mv);
                }
            }

            // Issue 207 promotion note: `PuctPlayer::new` now defaults to
            // int8 at K=1, so this test must use `with_f32` explicitly to
            // exercise the f32 path. The comparison still validates that the
            // int8 quantization noise doesn't change move selection.
            let mut f32_player = PuctPlayer::with_f32(50, 1.5, 8);
            let f32_mv = f32_player.select_move(&board);

            let mut i8_player = PuctPlayer::with_int8(50, 1.5, 8);
            let i8_mv = i8_player.select_move(&board);

            assert_eq!(
                f32_mv, i8_mv,
                "int8 PUCT move {i8_mv:?} != f32 PUCT move {f32_mv:?} on position after {moves:?}"
            );
        }
    }
}

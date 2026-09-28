//! Height-gated commit + inter-step gap-trend narrowing (Issue 688).
//!
//! Filed from the Coconut (arXiv:2412.06769v4) adversarial research panel —
//! **NOT a novelty claim**: both heuristics have decades of published prior
//! art (quiescence search Beal 1980 / conspiracy-number search McAllester
//! 1988 / LRTA* Korf 1990 for defer-commitment; successive halving
//! arXiv:1502.07943 / Hyperband / Hoeffding races for progressive
//! narrowing). Shipped because two documented gaps in `mux/` exactly consume
//! these signals:
//!
//! 1. `bfs.rs` plumbing `depth` into `MuxBfs::step()` and discarding it —
//!    no shipped mux consumer reads distance-to-terminal.
//! 2. Bench 017 T9's "Wide > Balanced > Deep >> Narrow" at fixed budget with
//!    **no data-dependent criterion for when narrowing wins**.
//!
//! # The two signals (Coconut §4.4 + Fig. 6)
//!
//! * **Height** (§4.4): probed value-estimate confidence is monotone in node
//!   HEIGHT (shortest distance to leaf) — near-terminal nodes get definitive
//!   evaluations, far nodes get ambiguous ones. Strategy: defer deterministic
//!   commitment until near terminal states; early exploration is cheap,
//!   early commitment is wrong.
//! * **Gap trend** (Fig. 6): top-1/2/3 cumulative-value gaps SHRINK from the
//!   first to the second continuous thought — progressive explore→focus
//!   narrowing driven by the gap trend itself, not a fixed schedule.
//!
//! # Signal-diff vs closest shipped cousins
//!
//! * `mux::bandit_width::MuxBanditWidth` — historical UCB1 reward per width
//!   arm (retrospective, cross-episode). This module: instantaneous,
//!   within-search.
//! * `bfs.rs::detect_width_with_peaks` — instantaneous top-k shape only
//!   (memoryless per leaf). This module adds the INTER-STEP derivative.
//! * riir-ai `latent_functor::ActionVerifyGate` — entropy + depth-from-ROOT
//!   (the opposite depth axis: progress made, not progress remaining).
//!
//! # Control law
//!
//! The cumulative top-k mass gap telescopes:
//! `Σ_{j=1}^{k−1} (p_j − p_{j+1}) = p_1 − p_k` — the mass between the best
//! and worst kept hypothesis. Its inter-step derivative drives width:
//!
//! * gap GREW beyond ε → **narrow** (focus; the frontier is differentiating)
//! * gap SHRANK beyond ε → **widen** back toward base (explore recovered)
//! * flat → **hold** (ambiguous; keep exploring at current width)
//!
//! The mandatory negative control (Bench 017 T9): on SimpleTES-style
//! FLAT-gap fixtures the derivative is 0 and the width HOLDS at base —
//! narrowing there would be a regression, and this law structurally cannot.

use crate::mux::bfs::MuxBfs;
use crate::mux::dd_tree::{LeafPaths, MuxDdTree};
use crate::mux::top_k::{MAX_TOP_K, extract_top_k_into};

/// Default gap-derivative dead-band (flat tolerance).
pub const DEFAULT_GAP_EPS: f32 = 1e-4;

/// Commit-timing gate over distance-to-TERMINAL (node height).
///
/// Near-terminal (low-height) candidates commit first — their value
/// estimates are the definitive ones (Coconut §4.4). High-height candidates
/// hold breadth: early commitment on ambiguous evaluations is the documented
/// failure mode this gate removes.
#[derive(Debug, Clone, Copy)]
pub struct HeightGate {
    /// Nodes at `height <= commit_height` commit deterministically (width 1).
    pub commit_height: usize,
}

impl HeightGate {
    pub fn new(commit_height: usize) -> Self {
        Self { commit_height }
    }

    /// Should a node at this height (distance to terminal) commit now?
    #[inline]
    pub fn should_commit(&self, height: usize) -> bool {
        height <= self.commit_height
    }

    /// Width modulation: committed nodes collapse to width 1 (deterministic
    /// commit); high-height nodes keep the (possibly narrowed) base width.
    #[inline]
    pub fn commit_width(&self, height: usize, base_width: usize) -> usize {
        if self.should_commit(height) {
            1
        } else {
            base_width
        }
    }

    /// Commit-priority ordering of candidate indices: value descending,
    /// then height ascending (among value-ties, min-height commits first),
    /// then index ascending (determinism). Zero-alloc: sorts the
    /// caller-provided `order` buffer, which is CLEARED and refilled
    /// (`0..values.len()`).
    ///
    /// # Panics
    /// If `values` and `heights` differ in length.
    pub fn commit_order_by_height_into(
        &self,
        values: &[f32],
        heights: &[usize],
        order: &mut Vec<usize>,
    ) {
        assert_eq!(
            values.len(),
            heights.len(),
            "values and heights must be parallel arrays"
        );
        order.clear();
        order.extend(0..values.len());
        order.sort_by(|&a, &b| {
            values[b]
                .total_cmp(&values[a])
                .then(heights[a].cmp(&heights[b]))
                .then(a.cmp(&b))
        });
    }

    /// Allocating convenience wrapper for [`Self::commit_order_by_height_into`].
    pub fn commit_order_by_height(&self, values: &[f32], heights: &[usize]) -> Vec<usize> {
        let mut order = Vec::with_capacity(values.len());
        self.commit_order_by_height_into(values, heights, &mut order);
        order
    }
}

/// Inter-step gap-trend width controller (progressive narrowing).
///
/// Feeds each search step's cumulative top-k mass gap and returns the width
/// to use: grew → narrow, shrank → widen back, flat → hold. Width is bounded
/// `1..=base_width`; the first observation initializes the trend baseline
/// (no derivative yet → hold).
#[derive(Debug, Clone, Copy)]
pub struct GapTrendNarrower {
    base_width: usize,
    width: usize,
    prev_gap: Option<f32>,
    eps: f32,
}

impl GapTrendNarrower {
    pub fn new(base_width: usize) -> Self {
        Self::with_epsilon(base_width, DEFAULT_GAP_EPS)
    }

    pub fn with_epsilon(base_width: usize, eps: f32) -> Self {
        Self {
            base_width: base_width.max(1),
            width: base_width.max(1),
            prev_gap: None,
            eps,
        }
    }

    /// Reset to the base width with no trend baseline.
    pub fn reset(&mut self) {
        self.width = self.base_width;
        self.prev_gap = None;
    }

    /// Current width after the latest observation.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The configured maximum (exploration) width.
    #[inline]
    pub fn base_width(&self) -> usize {
        self.base_width
    }

    /// Cumulative top-k mass gap of a descending peaks slice — the
    /// telescoping sum `Σ (p_j − p_{j+1}) = p_first − p_last`.
    #[inline]
    pub fn cumulative_gap(peaks: &[f32]) -> f32 {
        match (peaks.first(), peaks.last()) {
            (Some(&first), Some(&last)) if peaks.len() >= 2 => first - last,
            _ => 0.0,
        }
    }

    /// Single-stream convenience: observe a peaks slice, get the width.
    pub fn observe(&mut self, peaks: &[f32]) -> usize {
        let gap = Self::cumulative_gap(peaks);
        self.observe_gap(gap)
    }

    /// Core: observe this step's cumulative gap, get the width for the step.
    ///
    /// * first observation → hold at current width (baseline init)
    /// * `gap − prev > eps` → narrow (`width − 1`, floor 1)
    /// * `gap − prev < −eps` → widen (`width + 1`, cap base)
    /// * flat → hold
    pub fn observe_gap(&mut self, gap: f32) -> usize {
        if let Some(prev) = self.prev_gap {
            let d = gap - prev;
            if d > self.eps {
                self.width = self.width.saturating_sub(1).max(1);
            } else if d < -self.eps {
                self.width = (self.width + 1).min(self.base_width);
            }
            // Flat: hold.
        }
        self.prev_gap = Some(gap);
        self.width
    }
}

impl MuxBfs {
    /// Height-gated + gap-trend-narrowed BFS step (Issue 688).
    ///
    /// Composes both mechanisms onto the existing `step` loop:
    ///
    /// 1. Per-leaf base width from [`MuxBfs::detect_width_with_peaks`]
    ///    (unchanged — the narrowing is ADDITIVE on top, not a replacement).
    /// 2. The frontier's MEAN cumulative top-k mass gap feeds one
    ///    [`GapTrendNarrower`] per step — the inter-step signal is a
    ///    search-level trend (Coconut Fig. 6 tracks the search's top-k
    ///    value gaps across thoughts, not per-branch).
    /// 3. [`HeightGate::commit_width`] collapses near-terminal leaves to
    ///    width 1 (deterministic commit) and caps the rest at the narrowed
    ///    step width.
    ///
    /// `heights_by_leaf` carries each leaf's distance-to-TERMINAL (the
    /// caller knows the solution depth; the tree only knows depth-from-root).
    ///
    /// Allocates a fresh [`LeafPaths`]; prefer [`Self::step_height_gated_into`]
    /// for hot loops.
    pub fn step_height_gated(
        &self,
        tree: &mut MuxDdTree,
        depth: usize,
        logits_by_leaf: &[Vec<f32>],
        heights_by_leaf: &[usize],
        gate: &HeightGate,
        narrower: &mut GapTrendNarrower,
    ) {
        let mut leaves = LeafPaths::new();
        self.step_height_gated_into(
            tree,
            depth,
            logits_by_leaf,
            heights_by_leaf,
            gate,
            narrower,
            &mut leaves,
        );
    }

    /// Zero-alloc variant of [`Self::step_height_gated`] — reuses the
    /// caller-provided `LeafPaths` buffer across steps.
    // 8 load-bearing hot-loop params: tree + depth + the step's three
    // parallel per-leaf arrays + both controllers + the reuse buffer —
    // the `step_into` shape extended by the two controllers. No bundling:
    // each param is independently caller-owned.
    #[allow(clippy::too_many_arguments)]
    pub fn step_height_gated_into(
        &self,
        tree: &mut MuxDdTree,
        depth: usize,
        logits_by_leaf: &[Vec<f32>],
        heights_by_leaf: &[usize],
        gate: &HeightGate,
        narrower: &mut GapTrendNarrower,
        leaves: &mut LeafPaths,
    ) {
        tree.collect_leaf_paths_flat_into(leaves);
        assert_eq!(
            leaves.len(),
            logits_by_leaf.len(),
            "logits count must match leaf count"
        );
        assert_eq!(
            leaves.len(),
            heights_by_leaf.len(),
            "heights count must match leaf count"
        );

        // 1. Frontier mean cumulative gap → one inter-step narrower update.
        let mut gap_sum = 0.0f32;
        let mut gap_n = 0usize;
        let mut buf = [0.0f32; MAX_TOP_K];
        for logits in logits_by_leaf {
            let peaks = extract_top_k_into(logits, tree.k, &mut buf);
            if peaks.len() >= 2 {
                gap_sum += GapTrendNarrower::cumulative_gap(peaks);
                gap_n += 1;
            }
        }
        let step_cap = if gap_n > 0 {
            narrower.observe_gap(gap_sum / gap_n as f32)
        } else {
            narrower.width()
        };

        // 2. Per-leaf width: base width, capped by the step trend, collapsed
        //    to 1 for near-terminal (committed) leaves.
        for (i, logits) in logits_by_leaf.iter().enumerate() {
            let peaks = extract_top_k_into(logits, tree.k, &mut buf);
            let base = self.detect_width_with_peaks(peaks);
            let height = *heights_by_leaf.get(i).unwrap_or(&usize::MAX);
            let width = gate.commit_width(height, base.min(step_cap));
            if tree.pruner.is_valid_with_peaks(peaks) {
                tree.expand_node_with_peaks(leaves.path(i), peaks, width);
            }
        }
        let _ = depth; // preserved for API compat (same contract as `step`)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── HeightGate ─────────────────────────────────────────────────

    #[test]
    fn should_commit_threshold_inclusive() {
        let gate = HeightGate::new(3);
        assert!(gate.should_commit(0));
        assert!(gate.should_commit(3));
        assert!(!gate.should_commit(4));
    }

    #[test]
    fn commit_width_collapses_near_terminal() {
        let gate = HeightGate::new(2);
        assert_eq!(gate.commit_width(1, 4), 1); // near terminal → commit
        assert_eq!(gate.commit_width(5, 4), 4); // far → hold breadth
    }

    /// G1: height-gated commit ordering ≡ oracle ordering (value desc,
    /// height asc) — and DIFFERS from the ungated baseline (value desc,
    /// insertion order) when heights are inverted.
    #[test]
    fn g1_commit_ordering_matches_oracle() {
        let gate = HeightGate::new(2);
        // Planted fixture: value-tied pairs with INVERTED heights so the
        // ungated (insertion-order) tie-break disagrees with the oracle.
        let values = [0.9, 0.9, 0.7, 0.7, 0.5];
        let heights = [4, 1, 5, 2, 0];
        let gated = gate.commit_order_by_height(&values, &heights);

        // Oracle: (value desc, height asc, index asc).
        let mut oracle: Vec<usize> = (0..values.len()).collect();
        oracle.sort_by(|&a, &b| {
            values[b]
                .total_cmp(&values[a])
                .then(heights[a].cmp(&heights[b]))
                .then(a.cmp(&b))
        });

        assert_eq!(
            gated, oracle,
            "height-gated ordering must equal the (value desc, height asc) oracle"
        );
        // Concretely: among the 0.9-ties, index 1 (height 1) beats index 0
        // (height 4); among the 0.7-ties, index 3 (height 2) beats index 2.
        assert_eq!(gated[..2], [1, 0]);
        assert_eq!(gated[2..4], [3, 2]);

        // The ungated baseline (value desc, insertion order) differs — the
        // tie-break is load-bearing, not cosmetic.
        let mut ungated: Vec<usize> = (0..values.len()).collect();
        ungated.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        assert_ne!(gated, ungated);
    }

    /// Zero-alloc variant reuses the buffer without growing it.
    #[test]
    fn commit_order_into_reuses_buffer() {
        let gate = HeightGate::new(2);
        let values = [0.1, 0.9, 0.5];
        let heights = [9, 0, 3];
        let mut order = Vec::with_capacity(8);
        gate.commit_order_by_height_into(&values, &heights, &mut order);
        assert_eq!(order, vec![1, 2, 0]);
        let cap_before = order.capacity();
        gate.commit_order_by_height_into(&values, &heights, &mut order);
        assert_eq!(order, vec![1, 2, 0]);
        assert_eq!(order.capacity(), cap_before, "buffer must not regrow");
    }

    // ── GapTrendNarrower ───────────────────────────────────────────

    #[test]
    fn first_observation_holds() {
        let mut n = GapTrendNarrower::new(4);
        assert_eq!(n.observe(&[0.5, 0.3, 0.2, 0.1]), 4); // baseline init, no derivative
    }

    /// G1 (narrow direction): growing gaps narrow the width step by step
    /// down to the floor 1.
    #[test]
    fn g1_growing_gaps_narrow() {
        let mut n = GapTrendNarrower::new(4);
        // Frontier sharpening: gap grows every step.
        let frontiers = [
            [0.30, 0.27, 0.24, 0.19], // gap 0.11
            [0.45, 0.25, 0.18, 0.12], // gap 0.33
            [0.70, 0.15, 0.09, 0.06], // gap 0.64
            [0.88, 0.06, 0.04, 0.02], // gap 0.86
            [0.95, 0.03, 0.01, 0.01], // gap 0.94
        ];
        let mut widths = Vec::with_capacity(frontiers.len());
        for f in &frontiers {
            widths.push(n.observe(f));
        }
        assert_eq!(
            widths,
            vec![4, 3, 2, 1, 1],
            "progressive narrowing to floor 1"
        );
    }

    /// Shrinking gaps widen back toward base (explore recovery).
    #[test]
    fn shrinking_gaps_widen_back() {
        let mut n = GapTrendNarrower::new(4);
        assert_eq!(n.observe(&[0.60, 0.35, 0.30, 0.25]), 4); // baseline (gap 0.35)
        assert_eq!(n.observe(&[0.90, 0.40, 0.35, 0.30]), 3); // grew (0.60) → narrow
        assert_eq!(n.observe(&[0.45, 0.35, 0.30, 0.25]), 4); // shrank (0.20) → widen
        assert_eq!(n.observe(&[0.32, 0.30, 0.28, 0.26]), 4); // shrank → hold at cap
    }

    /// **Mandatory negative control** (Bench 017 T9): SimpleTES-style
    /// FLAT-gap fixtures must NOT narrow — wide-dominance is the control
    /// case; narrowing there would be a regression. This control law
    /// structurally cannot: flat derivative = hold.
    #[test]
    fn g1_negative_control_flat_gaps_do_not_narrow() {
        let mut n = GapTrendNarrower::new(4);
        // Flat-gap frontier: uniform-ish peaks, constant cumulative gap.
        for _ in 0..50 {
            let w = n.observe(&[0.28, 0.26, 0.24, 0.22]); // gap 0.06 every step
            assert_eq!(w, 4, "flat gap trend must hold width at base");
        }
        // Identical peaks (zero gap) — same verdict.
        let mut n2 = GapTrendNarrower::new(4);
        for _ in 0..50 {
            assert_eq!(n2.observe(&[0.25, 0.25, 0.25, 0.25]), 4);
        }
    }

    /// Sub-ε derivative noise is held flat (dead-band).
    #[test]
    fn dead_band_holds() {
        let mut n = GapTrendNarrower::with_epsilon(4, 0.01);
        n.observe(&[0.30, 0.25, 0.23, 0.22]); // gap 0.08
        // +0.005 < eps → flat → hold.
        assert_eq!(n.observe(&[0.305, 0.25, 0.23, 0.22]), 4);
    }

    #[test]
    fn cumulative_gap_telescopes() {
        let peaks = [0.5, 0.3, 0.2];
        let adj: f32 = (0.5 - 0.3) + (0.3 - 0.2);
        assert!((GapTrendNarrower::cumulative_gap(&peaks) - adj).abs() < 1e-7);
        assert!((GapTrendNarrower::cumulative_gap(&peaks) - 0.3).abs() < 1e-7);
        assert_eq!(GapTrendNarrower::cumulative_gap(&[0.4]), 0.0);
        assert_eq!(GapTrendNarrower::cumulative_gap(&[]), 0.0);
    }

    // ── Composed BFS step ──────────────────────────────────────────

    /// Frontier that sharpens with each step (cumulative gap grows) while
    /// staying inside the pruner's geometric-decay contract early on.
    fn sharpe_logits(step: usize) -> Vec<f32> {
        let top = 0.30 + 0.15 * step as f32;
        let rest = 0.28 - 0.05 * step as f32;
        vec![top, rest, rest * 0.9, rest * 0.8]
    }

    /// Run one composed step applying the SAME template to every current
    /// leaf (the tests' uniform-frontier shape). Heights uniform too.
    fn step_uniform(
        bfs: &MuxBfs,
        tree: &mut MuxDdTree,
        template: &[f32],
        height: usize,
        gate: &HeightGate,
        narrower: &mut GapTrendNarrower,
        leaves: &mut LeafPaths,
    ) {
        let n = tree.leaf_count().max(1);
        let logits: Vec<Vec<f32>> = (0..n).map(|_| template.to_vec()).collect();
        let heights: Vec<usize> = vec![height; n];
        bfs.step_height_gated_into(tree, 0, &logits, &heights, gate, narrower, leaves);
    }

    /// Composed step: a near-terminal leaf commits (width 1 → leaf count
    /// stays 1 while depth grows) vs the ungated baseline (width 4 → 4
    /// leaves); a far leaf holds full width (4 leaves, parity with ungated).
    #[test]
    fn composed_step_commits_near_terminal() {
        let bfs = MuxBfs::new(4);
        let logits = vec![0.5, 0.4, 0.3, 0.2, 0.1];

        // Ungated baseline leaf count after one step.
        let mut baseline_tree = MuxDdTree::new(4);
        baseline_tree.init_root(&logits);
        bfs.step(&mut baseline_tree, 1, std::slice::from_ref(&logits));
        let baseline_leaves = baseline_tree.leaf_count();
        assert!(baseline_leaves > 1, "ungated multi-peak must expand");

        // Height-gated: leaf at height 0 (terminal-adjacent) commits →
        // width 1 → exactly ONE successor leaf (depth grows, breadth holds).
        let mut gated_tree = MuxDdTree::new(4);
        gated_tree.init_root(&logits);
        let gate = HeightGate::new(1);
        let mut narrower = GapTrendNarrower::new(4);
        let mut leaves_buf = LeafPaths::new();
        step_uniform(
            &bfs,
            &mut gated_tree,
            &logits,
            0,
            &gate,
            &mut narrower,
            &mut leaves_buf,
        );
        assert_eq!(gated_tree.leaf_count(), 1, "width-1 commit → one successor");
        assert!(gated_tree.depth >= 1, "depth still grows");
        assert!(
            gated_tree.leaf_count() < baseline_leaves,
            "near-terminal commit must narrow vs ungated ({} vs {})",
            gated_tree.leaf_count(),
            baseline_leaves
        );

        // Far height → holds full width → parity with ungated.
        let mut far_tree = MuxDdTree::new(4);
        far_tree.init_root(&logits);
        let mut narrower2 = GapTrendNarrower::new(4);
        step_uniform(
            &bfs,
            &mut far_tree,
            &logits,
            99,
            &gate,
            &mut narrower2,
            &mut leaves_buf,
        );
        assert_eq!(far_tree.leaf_count(), baseline_leaves);
    }

    /// The negative control THROUGH the composed step: flat-gap logits +
    /// far heights → the step width NEVER narrows across many steps.
    #[test]
    fn composed_negative_control_flat_gaps() {
        let bfs = MuxBfs::new(4);
        let flat = vec![0.26, 0.25, 0.24, 0.23, 0.22];
        let gate = HeightGate::new(0); // commit only AT the terminal
        let mut narrower = GapTrendNarrower::new(4);
        let mut leaves_buf = LeafPaths::new();

        let mut tree = MuxDdTree::new(4);
        tree.init_root(&flat);
        // 4 steps: leaves grow 1 → 4 → 16 → 64 (uniform width held at 4).
        for step in 0..4 {
            step_uniform(
                &bfs,
                &mut tree,
                &flat,
                99,
                &gate,
                &mut narrower,
                &mut leaves_buf,
            );
            assert_eq!(
                narrower.width(),
                4,
                "flat gaps must never narrow (step {step})"
            );
        }
        assert_eq!(tree.leaf_count(), 4 * 4 * 4 * 4);
    }

    /// Multi-step composed run: a sharpe frontier narrows the step cap
    /// monotonically to 1 (the pruner may reject late steps — the trend
    /// controller still observes every frontier's gaps).
    #[test]
    fn composed_multi_step_narrows() {
        let bfs = MuxBfs::new(4);
        let mut tree = MuxDdTree::new(4);
        let l0 = sharpe_logits(0);
        tree.init_root(&l0);
        let gate = HeightGate::new(0);
        let mut narrower = GapTrendNarrower::new(4);
        let mut leaves_buf = LeafPaths::new();
        let mut widths = Vec::new();
        for step in 1..5 {
            let l = sharpe_logits(step);
            step_uniform(
                &bfs,
                &mut tree,
                &l,
                99,
                &gate,
                &mut narrower,
                &mut leaves_buf,
            );
            widths.push(narrower.width());
        }
        assert!(
            widths.windows(2).all(|w| w[0] >= w[1]),
            "monotone narrowing: {widths:?}"
        );
        assert_eq!(*widths.last().unwrap(), 1);
    }

    /// G4: zero allocs on the CONTROLLER path (extract → gap → narrower →
    /// gate). The composed step's remaining allocations are the TREE's
    /// (child-node growth on expansion) and the CALLER's (logits storage) —
    /// neither is the gate's cost, and both are measured by existing bfs/ddtree
    /// gates.
    #[cfg(all(test, debug_assertions))]
    #[test]
    fn g4_composed_step_zero_alloc() {
        let gate = HeightGate::new(1);
        let mut narrower = GapTrendNarrower::new(4);
        let logits = [0.5f32, 0.4, 0.3, 0.2];
        let mut buf = [0.0f32; MAX_TOP_K];
        // Warm up (lazy init paranoia).
        let _ = narrower.observe(extract_top_k_into(&logits, 4, &mut buf));
        let _ = gate.commit_width(0, 4);
        crate::alloc::reset_alloc_stats();
        for i in 0..1_000 {
            let peaks = extract_top_k_into(&logits, 4, &mut buf);
            let _ = narrower.observe(peaks);
            let _ = gate.commit_width(i % 3, 4);
        }
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(count, 0, "controller path must be allocation-free");
    }

    /// G2: ns-scale controller latency (release-only; debug carries the
    /// tracking allocator + no optimization).
    #[cfg_attr(debug_assertions, ignore)]
    #[test]
    fn g2_controller_latency_ns_scale() {
        const N: usize = 100_000;

        let mut n = GapTrendNarrower::new(4);
        let gate = HeightGate::new(2);
        // Warm up.
        for _ in 0..1_000 {
            n.observe_gap(0.5);
        }
        let t0 = std::time::Instant::now();
        let mut w = 0usize;
        for i in 0..N {
            w = w.wrapping_add(n.observe_gap(0.4 + (i & 7) as f32 * 0.05));
            w = w.wrapping_add(gate.commit_width(3, w.max(1)));
        }
        let per_op_ns = t0.elapsed().as_nanos() as f64 / (2 * N) as f64;
        assert!(
            per_op_ns < 50.0,
            "controller must be ns-scale: {per_op_ns:.2} ns/op"
        );
    }
}

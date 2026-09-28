//! Bigram Markov head — the modelless sequential drafter (Issue 659, Research 316 §3.5 path 2).
//!
//! DSpark's sequential head, constructed deterministically from corpus bigram
//! co-occurrence statistics — no training, no gradient descent. This is the
//! Metal-viable alternative drafter for the case where the Prism-ML-trained
//! DSpark 6-layer drafter is switched off (Bench 656 failure mode 2: a separate
//! multi-layer drafter pays its own forward every step, which batch-1
//! verification cannot amortise on Apple Silicon).
//!
//! A bigram head is a **table lookup, not a forward pass** — it does not incur
//! mode 2 at all. Emission cost is O(steps × top_m) sparse writes (~128 writes
//! at top_m=16, lookahead 8 ≈ sub-µs) versus a 6-layer forward per step.
//!
//! # Construction (deterministic)
//!
//! - `BigramMarkovBuilder::add_sequence(&[u32])` accumulates `(prev, next)`
//!   pairs packed as `u64` (`prev << 32 | next`).
//! - `build(vocab_size, top_m)` sorts the pairs (total order on u64 — no
//!   reliance on sort stability), run-length counts each distinct bigram, and
//!   keeps the top-`m` successors per prev token by `(count desc, next asc)`.
//!   Same corpus → bit-identical table.
//! - Out-of-vocabulary pairs (either side ≥ `vocab_size`) are discarded
//!   deterministically during the build pass; row totals count in-vocab
//!   successors only, so stored probabilities are true conditionals over the
//!   in-vocab support and the stored top-m mass is ≤ 1.0 (the truncated tail
//!   is dropped mass, never renormalised).
//!
//! # Emission (zero-alloc, mirrors `dflash_predict_with`)
//!
//! `bigram_predict` fills a caller-owned [`BigramMarginalBuffer`] with
//! `steps × vocab` dense marginals, step-major (step `i` occupies
//! `[i*vocab .. (i+1)*vocab]`) — the exact layout `dflash_predict_with`
//! produces and [`crate::dd_tree::build_dd_tree`] consumes.
//!
//! The sparse rows are scattered into the buffer, and only the touched
//! entries are reset on the next call (the scoped-init-at-consume-site
//! pattern): steady-state emission performs **zero allocation and zero
//! O(vocab) work** — O(steps × top_m) writes plus O(|previous touched|)
//! resets. A dense fill would cost `steps × vocab × 4 B` (~4 MB at Bonsai
//! scale, lookahead 8) per draft cycle, which would dwarf the target-model
//! forward.
//!
//! ## Greedy-chain conditioning
//!
//! The dd_tree seam takes static per-depth marginals (`&[&[f32]]`), not
//! path-conditioned ones. The head therefore emits the **greedy-path
//! conditioning**: marginal at depth `k` is `P(· | greedy_token_{k-1})`
//! where `greedy_token_0 = last_token` and `greedy_token_{k}` is the argmax
//! successor of `greedy_token_{k-1}` (= `successors[0]`, ties broken by
//! token id ascending). Deeper branches of the tree still explore siblings
//! of the greedy path under the bigram distribution around the greedy
//! predecessor — the same position-indexed refinement DSpark's Markov head
//! applies over the parallel backbone.
//!
//! ## Unseen prev token → zero row (not uniform)
//!
//! A prev token with no in-vocab bigram evidence emits an **all-zero row**.
//! The tree builder skips `prob <= 0.0` candidates at every expansion site,
//! so a zero row proposes nothing — the honest drafter behaviour when the
//! head has no information (a uniform row would spend verification budget
//! on arbitrary tokens). Under best-first log-prob scoring a zero row is
//! rank-equivalent to uniform for every candidate it would admit anyway;
//! it simply admits none. The greedy chain stalls on the same prev token.
//!
//! # Memory bound (G5)
//!
//! Worst case (every row full): `(V+1 + V·m·2 + V) × 4 B`. At Bonsai scale
//! (`V = 131_072`, `m = 16`) that is **17,825,796 B ≈ 17 MB** — versus a
//! dense `V × V` table at ~68 GB and a DSpark low-rank `r=256` head at
//! ~268 MB.
//!
//! # Composition (Issue 659 design sketch)
//!
//! - Bebop entropy confidence (`crate::acceptance_forecast`) — shipped.
//! - Hardware-Aware Prefix Scheduler (`crate::prefix_scheduler`, Plan 339) — shipped.
//! - The Bonsai consumer + GOAT G2/G3 wall-clock gate belongs in riir-ai
//!   (Plan 528); this crate ships the primitive.
//!
//! Feature-gated behind `bigram_markov` — off by default until the riir-ai
//! Bonsai consumer gate proves the gain (Issue 659 T4).

use katgpt_core::speculative::types::TreeNode;

// ── Table ──────────────────────────────────────────────────────

/// CSR-style top-`m` bigram transition table: `P(next | prev)`.
///
/// Built once at load from corpus counts (deterministic); read-only on the
/// hot path. Rows are sorted by `(count desc, next asc)` — `successors[0]`
/// is the argmax successor. Note: rows are NOT token-sorted, so use
/// [`BigramMarkovTable::probability`] (linear scan, `m ≤ 64`) for point
/// lookups, not `binary_search`.
#[derive(Debug, Clone, PartialEq)]
pub struct BigramMarkovTable {
    vocab_size: usize,
    top_m: usize,
    /// `vocab_size + 1` entries; successors of `prev` live at
    /// `successors[row_offsets[prev]..row_offsets[prev + 1]]`.
    row_offsets: Vec<u32>,
    /// Successor token ids, row-major.
    successors: Vec<u32>,
    /// `P(next | prev)` aligned with `successors`. True conditional
    /// probabilities (count / row_total), NOT renormalised after top-m
    /// truncation — the stored mass is ≤ 1.0.
    probs: Vec<f32>,
    /// Total in-vocab bigram count per prev token (the normaliser,
    /// including the truncated tail).
    row_totals: Vec<u32>,
}

impl BigramMarkovTable {
    /// Top-`m` successors of `prev` and their probabilities, or `None` if
    /// `prev` is out of vocab or has no in-vocab bigram evidence.
    #[inline]
    pub fn successors(&self, prev: u32) -> Option<(&[u32], &[f32])> {
        let prev = prev as usize;
        if prev >= self.vocab_size {
            return None;
        }
        let start = self.row_offsets[prev] as usize;
        let end = self.row_offsets[prev + 1] as usize;
        if start == end {
            return None;
        }
        Some((&self.successors[start..end], &self.probs[start..end]))
    }

    /// `P(next | prev)` — `0.0` when the pair is absent (out of vocab,
    /// no evidence, or truncated out of top-m). Linear scan over the row
    /// (`m ≤ 64` entries, count-sorted — binary search is not applicable).
    #[inline]
    pub fn probability(&self, prev: u32, next: u32) -> f32 {
        let Some((succs, probs)) = self.successors(prev) else {
            return 0.0;
        };
        for (&s, &p) in succs.iter().zip(probs.iter()) {
            if s == next {
                return p;
            }
        }
        0.0
    }

    /// Total in-vocab bigram count for `prev` (0 when unseen).
    #[inline]
    pub fn row_total(&self, prev: u32) -> u32 {
        let prev = prev as usize;
        if prev >= self.vocab_size {
            return 0;
        }
        self.row_totals[prev]
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[inline]
    pub fn top_m(&self) -> usize {
        self.top_m
    }

    /// Exact heap footprint: CSR offsets + successors + probs + row totals.
    ///
    /// At Bonsai scale (`vocab = 131_072`, `top_m = 16`, worst case every
    /// row full): 17,825,796 B ≈ 17 MB — versus a dense `V × V` table at
    /// ~68 GB and a low-rank `r=256` head at ~268 MB.
    pub fn memory_bytes(&self) -> usize {
        (self.row_offsets.len() + self.successors.len() + self.probs.len() + self.row_totals.len())
            * std::mem::size_of::<u32>()
    }
}

// ── Builder ────────────────────────────────────────────────────

/// Deterministic bigram-count builder. Accumulates `(prev, next)` pairs
/// packed as `u64`, then [`build`](BigramMarkovBuilder::build)s the CSR table
/// via sort + two pointer passes.
///
/// Build is offline (once at load); the 8-byte-per-pair working set is the
/// only memory cost. `add_sequence` contributes nothing for sequences
/// shorter than 2 tokens.
#[derive(Debug, Default, Clone)]
pub struct BigramMarkovBuilder {
    pairs: Vec<u64>,
}

impl BigramMarkovBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-reserve for `n_pairs` bigram pairs (a corpus of `T` tokens across
    /// `S` sequences yields `T - S` pairs).
    pub fn with_capacity(n_pairs: usize) -> Self {
        Self {
            pairs: Vec::with_capacity(n_pairs),
        }
    }

    /// Add one token sequence's bigrams: `(t[0],t[1]), (t[1],t[2]), …`.
    pub fn add_sequence(&mut self, tokens: &[u32]) {
        for w in tokens.windows(2) {
            let packed = (w[0] as u64) << 32 | w[1] as u64;
            self.pairs.push(packed);
        }
    }

    /// Number of accumulated bigram pairs (pre-dedup).
    pub fn len_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Build the CSR top-`m` table. Out-of-vocab pairs (either side ≥
    /// `vocab_size`) are discarded deterministically (sorted prev-major,
    /// everything at prev ≥ `vocab_size` is never visited).
    ///
    /// Same corpus + same `(vocab_size, top_m)` → bit-identical table:
    /// `sort_unstable` on a total order, run-length counting, and
    /// `(count desc, next asc)` top-m insertion are all deterministic.
    pub fn build(self, vocab_size: usize, top_m: usize) -> BigramMarkovTable {
        let mut pairs = self.pairs;
        pairs.sort_unstable();

        // Pass 1: per-prev in-vocab totals (the probability normaliser).
        let mut row_totals = vec![0u32; vocab_size];
        {
            let mut i = 0usize;
            while i < pairs.len() {
                let prev = (pairs[i] >> 32) as usize;
                if prev >= vocab_size {
                    break; // prev-major sorted: the rest is all OOV
                }
                while i < pairs.len() && (pairs[i] >> 32) as usize == prev {
                    if ((pairs[i] & 0xFFFF_FFFF) as usize) < vocab_size {
                        row_totals[prev] += 1;
                    }
                    i += 1;
                }
            }
        }

        // Pass 2: per-row top-m selection by (count desc, next asc).
        // `row_offsets[prev]` is advanced only by rows that store entries;
        // empty rows (unseen / fully-OOV prevs) get equal boundaries.
        let mut row_offsets = Vec::with_capacity(vocab_size + 1);
        let mut successors: Vec<u32> = Vec::new();
        let mut probs: Vec<f32> = Vec::new();
        // Running top-m buffer: (count, next), kept sorted count desc / next asc.
        let mut best: Vec<(u32, u32)> = Vec::with_capacity(top_m);

        let mut prefix = 0u32;
        row_offsets.push(prefix);
        let mut i = 0usize;
        for (prev, total) in row_totals.iter().enumerate() {
            if i < pairs.len() && (pairs[i] >> 32) as usize == prev {
                best.clear();
                while i < pairs.len() && (pairs[i] >> 32) as usize == prev {
                    let next = pairs[i] & 0xFFFF_FFFF;
                    let run_start = i;
                    while i < pairs.len() && pairs[i] == pairs[run_start] {
                        i += 1;
                    }
                    if (next as usize) < vocab_size {
                        let count = (i - run_start) as u32;
                        insert_top_m(&mut best, (count, next as u32), top_m);
                    }
                }
                for &(count, next) in &best {
                    successors.push(next);
                    probs.push(count as f32 / *total as f32);
                }
                prefix += best.len() as u32;
            }
            row_offsets.push(prefix);
        }

        BigramMarkovTable {
            vocab_size,
            top_m,
            row_offsets,
            successors,
            probs,
            row_totals,
        }
    }
}

/// Insert `(count, next)` into the running top-`m` buffer, keeping it sorted
/// by `(count desc, next asc)`. Deterministic: equal counts order by next asc.
fn insert_top_m(best: &mut Vec<(u32, u32)>, cand: (u32, u32), top_m: usize) {
    if top_m == 0 {
        return;
    }
    // Insertion position: before the first held entry that is strictly worse
    // than `cand` (lower count, or equal count and higher next).
    let mut pos = best.len();
    for (k, &(c, n)) in best.iter().enumerate() {
        if (cand.0 > c) || (cand.0 == c && cand.1 < n) {
            pos = k;
            break;
        }
    }
    if pos == best.len() {
        if best.len() < top_m {
            best.push(cand);
        }
        return;
    }
    if best.len() < top_m {
        best.insert(pos, cand);
    } else {
        // Full: shift the strictly-worse tail right, dropping the old worst.
        let last = best.len() - 1;
        best.copy_within(pos..last, pos + 1);
        best[pos] = cand;
        best.truncate(top_m);
    }
}

// ── Emission buffer ────────────────────────────────────────────

/// Caller-owned persistent marginal buffer enforcing the zero-outside-touched
/// invariant: every entry not written by the last [`bigram_predict`] call is
/// `0.0`.
///
/// Allocate once per session (`steps × vocab × 4 B`, ~4 MB at Bonsai scale,
/// lookahead 8); reuse across draft cycles. The buffer starts zeroed and
/// [`bigram_predict`] resets only the entries it previously touched, so
/// steady-state emission is allocation-free and O(steps × top_m).
#[derive(Debug, Clone)]
pub struct BigramMarginalBuffer {
    flat: Vec<f32>,
    /// Flat indices written by the last predict call (to reset next call).
    touched: Vec<u32>,
    steps: usize,
    vocab_size: usize,
}

impl BigramMarginalBuffer {
    /// Zeroed buffer for `steps` draft positions × `vocab_size` tokens.
    /// Pre-reserves `steps × 16` touched slots (the typical top_m); a larger
    /// table top_m grows it once on first predict.
    pub fn new(steps: usize, vocab_size: usize) -> Self {
        Self {
            flat: vec![0.0; steps * vocab_size],
            touched: Vec::with_capacity(steps * 16),
            steps,
            vocab_size,
        }
    }

    /// The full flat marginal slab, step-major: step `i` occupies
    /// `[i*vocab .. (i+1)*vocab]`.
    #[inline]
    pub fn marginals(&self) -> &[f32] {
        &self.flat
    }

    /// One step's dense marginal row.
    #[inline]
    pub fn row(&self, step: usize) -> &[f32] {
        let v = self.vocab_size;
        &self.flat[step * v..(step + 1) * v]
    }

    #[inline]
    pub fn steps(&self) -> usize {
        self.steps
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of sparse entries the last predict call wrote (diagnostic).
    #[inline]
    pub fn touched_len(&self) -> usize {
        self.touched.len()
    }
}

// ── Emission ───────────────────────────────────────────────────

/// Emit greedy-chain bigram marginals into `buf` (zero-alloc steady state).
///
/// Fills `steps` dense rows (step `i` = `P(· | greedy_token_{i-1})`, greedy
/// chain seeded at `last_token`; see the [module docs](self) for the
/// conditioning + zero-row semantics). Returns the number of rows populated
/// (= `max_steps.min(buf.steps())`).
///
/// O(steps × top_m) writes + O(previous |touched|) resets after the first
/// call. `buf` must have been created with the same `vocab_size` as `table`
/// (debug-asserted).
pub fn bigram_predict(
    table: &BigramMarkovTable,
    last_token: u32,
    max_steps: usize,
    buf: &mut BigramMarginalBuffer,
) -> usize {
    debug_assert_eq!(
        buf.vocab_size, table.vocab_size,
        "BigramMarginalBuffer vocab must match the table"
    );
    let vocab = buf.vocab_size;
    let steps = max_steps.min(buf.steps);

    // Reset only the entries the previous call touched (the buffer's
    // zero-outside-touched invariant). First call: touched is empty.
    for &idx in buf.touched.iter() {
        buf.flat[idx as usize] = 0.0;
    }
    buf.touched.clear();

    let mut prev = last_token;
    for step in 0..steps {
        let row_start = step * vocab;
        if let Some((succs, probs)) = table.successors(prev) {
            for (&next, &p) in succs.iter().zip(probs.iter()) {
                let idx = row_start + next as usize;
                buf.flat[idx] = p;
                buf.touched.push(idx as u32);
            }
            // Rows are (count desc, next asc): successors[0] is the argmax.
            prev = succs[0];
        }
        // Unseen prev → zero row (proposes nothing; see module docs). The
        // greedy chain stalls on the same token.
    }
    steps
}

/// Build a DDTree verification tree from bigram marginals (Issue 659 T3 —
/// the `build_dd_tree` seam wiring).
///
/// Emits `config.draft_lookahead` greedy-chain marginals into `buf`, then
/// feeds them to [`crate::dd_tree::build_dd_tree`] as `&[&[f32]]` rows.
/// The row-slice `Vec` is allocated here (tree building allocates anyway —
/// this is not the per-token hot path; [`bigram_predict`] is).
///
/// Tokens proposed by the returned tree are exactly the tokens the seam's
/// best-first search admits from the bigram rows: siblings of the greedy
/// chain under `P(· | greedy predecessor)`. Zero rows (unseen prev) end the
/// draft — no candidates are proposed past that depth.
pub fn bigram_build_tree(
    table: &BigramMarkovTable,
    last_token: u32,
    config: &katgpt_types::Config,
    buf: &mut BigramMarginalBuffer,
) -> Vec<TreeNode> {
    let steps = bigram_predict(table, last_token, config.draft_lookahead, buf);
    let vocab = buf.vocab_size;
    let rows: Vec<&[f32]> = buf.marginals().chunks_exact(vocab).take(steps).collect();
    crate::dd_tree::build_dd_tree(&rows, config)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference (brute-force) construction: HashMap counting + full sort.
    fn reference_table(corpus: &[&[u32]], vocab_size: usize, top_m: usize) -> BigramMarkovTable {
        use std::collections::HashMap;
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for seq in corpus {
            for w in seq.windows(2) {
                if (w[0] as usize) < vocab_size && (w[1] as usize) < vocab_size {
                    *counts.entry((w[0], w[1])).or_insert(0) += 1;
                }
            }
        }
        let mut row_offsets = vec![0u32; vocab_size + 1];
        let mut successors = Vec::new();
        let mut probs = Vec::new();
        let mut row_totals = vec![0u32; vocab_size];
        for prev in 0..vocab_size as u32 {
            let mut row: Vec<(u32, u32)> = counts
                .iter()
                .filter(|(k, _)| k.0 == prev)
                .map(|(k, &c)| (c, k.1))
                .collect();
            row_totals[prev as usize] = row.iter().map(|&(c, _)| c).sum();
            row.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            row.truncate(top_m);
            for &(c, n) in &row {
                successors.push(n);
                probs.push(c as f32 / row_totals[prev as usize] as f32);
            }
            row_offsets[prev as usize + 1] = successors.len() as u32;
        }
        BigramMarkovTable {
            vocab_size,
            top_m,
            row_offsets,
            successors,
            probs,
            row_totals,
        }
    }

    fn toy_corpus() -> Vec<Vec<u32>> {
        vec![
            vec![0, 1, 2, 3, 0, 1, 2, 0, 1],
            vec![1, 2, 3, 4, 1, 2, 3, 4, 0],
            vec![5, 0, 1, 2, 0, 1, 2, 3],
        ]
    }

    fn build_toy(vocab: usize, top_m: usize) -> BigramMarkovTable {
        let mut b = BigramMarkovBuilder::new();
        for seq in &toy_corpus() {
            b.add_sequence(seq);
        }
        b.build(vocab, top_m)
    }

    #[test]
    fn g1_build_deterministic_bit_identical() {
        let a = build_toy(16, 4);
        let b = build_toy(16, 4);
        assert_eq!(a, b, "same corpus must produce a bit-identical table");
    }

    #[test]
    fn g1_top_m_matches_bruteforce_reference() {
        for top_m in [1usize, 2, 4, 8] {
            let corpus_vec = toy_corpus();
            let corpus: Vec<&[u32]> = corpus_vec.iter().map(|v| v.as_slice()).collect();
            let mine = build_toy(8, top_m);
            let reference = reference_table(&corpus, 8, top_m);
            assert_eq!(mine, reference, "top_m={top_m}");
        }
    }

    #[test]
    fn g1_row_offsets_sparse_prevs() {
        // Toy corpus prevs: 0,1,2,3,4,5 all appear; rows 6,7 must be empty
        // with equal boundaries.
        let t = build_toy(8, 4);
        assert_eq!(t.row_offsets_len(), 9);
        for prev in [6usize, 7] {
            let (s, e) = t.row_bounds(prev);
            assert_eq!(s, e, "row {prev} must be empty");
            assert!(t.successors(prev as u32).is_none());
        }
        // Every non-empty row's range is exactly the stored slice.
        for prev in 0..8u32 {
            if let Some((succs, _)) = t.successors(prev) {
                let (s, e) = t.row_bounds(prev as usize);
                assert_eq!(e - s, succs.len());
            }
        }
    }

    #[test]
    fn g1_oov_pairs_discarded() {
        // vocab 4: tokens 4+ are OOV.
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 9, 1, 2, 9, 3]);
        let t = b.build(4, 4);
        // In-vocab pairs only: (0,1), (1,2). (1,9),(9,1),(2,9),(9,3) dropped.
        assert_eq!(t.row_total(0), 1);
        assert_eq!(t.row_total(1), 1);
        assert_eq!(t.row_total(2), 0, "(2,9) is OOV — no evidence");
        assert_eq!(t.row_total(9), 0, "OOV prev discards everything");
        assert_eq!(t.successors(2), None);
        assert_eq!(t.successors(0).unwrap().0, &[1]);
    }

    #[test]
    fn g1_tie_break_count_desc_then_token_asc() {
        // prev 0 → next 3 (×2), next 1 (×2), next 2 (×1): top-2 must be
        // (2, token 1), (2, token 3) — count desc, then token asc.
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 3, 0, 1, 0, 3, 0, 1, 0, 2]);
        let t = b.build(8, 2);
        let (succs, probs) = t.successors(0).unwrap();
        assert_eq!(succs, &[1, 3]);
        assert!((probs[0] - 2.0 / 5.0).abs() < 1e-7);
        assert!((probs[1] - 2.0 / 5.0).abs() < 1e-7);
    }

    #[test]
    fn g2_marginal_rows_sub_stochastic() {
        let t = build_toy(8, 2);
        let mut buf = BigramMarginalBuffer::new(4, 8);
        bigram_predict(&t, 0, 4, &mut buf);
        for step in 0..4 {
            let row = buf.row(step);
            let sum: f32 = row.iter().sum();
            assert!(
                sum <= 1.0 + 1e-6,
                "step {step}: stored mass {sum} must be ≤ 1.0 (truncated tail dropped)"
            );
        }
    }

    #[test]
    fn g2_greedy_chain_follows_argmax_successor() {
        let t = build_toy(8, 4);
        let mut buf = BigramMarginalBuffer::new(3, 8);
        bigram_predict(&t, 0, 3, &mut buf);

        let mut prev = 0u32;
        for step in 0..3 {
            let (succs, probs) = t.successors(prev).expect("toy corpus covers 0,1,2");
            let row = buf.row(step);
            // Every stored successor appears with its true probability.
            for (&next, &p) in succs.iter().zip(probs.iter()) {
                assert_eq!(row[next as usize], p, "step {step} token {next}");
            }
            // The row argmax is successors[0] (count desc, token asc; the
            // toy corpus has no count ties on the 0→1→2 chain).
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(argmax as u32, succs[0], "step {step}");
            prev = succs[0];
        }
    }

    #[test]
    fn g2_unseen_prev_emits_zero_row_and_chain_stalls() {
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 2, 3]);
        let t = b.build(8, 4);
        let mut buf = BigramMarginalBuffer::new(3, 8);
        // Token 6 unseen: all rows zero, chain stalls, nothing written.
        bigram_predict(&t, 6, 3, &mut buf);
        for step in 0..3 {
            assert!(buf.row(step).iter().all(|&p| p == 0.0), "step {step}");
        }
        assert_eq!(buf.touched_len(), 0, "zero rows write nothing");

        // Chain 0→1→2→3: token 3 has no successors → row 3 is zero, stall.
        let mut buf2 = BigramMarginalBuffer::new(4, 8);
        bigram_predict(&t, 0, 4, &mut buf2);
        assert!(buf2.row(0).iter().any(|&p| p > 0.0));
        assert!(buf2.row(3).iter().all(|&p| p == 0.0));
    }

    #[test]
    fn g2_probability_accessor() {
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 0, 1, 0, 2]);
        let t = b.build(4, 4);
        // prev 0: next 1 ×2, next 2 ×1 → total 3.
        assert!((t.probability(0, 1) - 2.0 / 3.0).abs() < 1e-7);
        assert!((t.probability(0, 2) - 1.0 / 3.0).abs() < 1e-7);
        assert_eq!(t.probability(0, 3), 0.0, "no evidence");
        assert_eq!(t.probability(9, 1), 0.0, "OOV prev");
        // (1,0) ×2 is the only pair with prev 1 → P(0|1) = 1.0.
        assert!((t.probability(1, 0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn g3_tree_wiring_smoke() {
        let t = build_toy(8, 4);
        // Config::micro: draft_lookahead 8, tree_budget 16, vocab 27.
        let config = katgpt_types::Config::micro();
        let mut buf = BigramMarginalBuffer::new(config.draft_lookahead, 8);
        let tree = bigram_build_tree(&t, 0, &config, &mut buf);
        assert!(
            !tree.is_empty(),
            "toy corpus has strong 0→1 signal; tree must propose"
        );
        // Depth-0 nodes must be successors of token 0.
        let (succs, _) = t.successors(0).unwrap();
        for node in tree.iter().filter(|n| n.depth == 0) {
            assert!(succs.contains(&(node.token_idx as u32)));
        }
        // The greedy chain (argmax successor per depth) must be present.
        let mut prev = 0u32;
        for depth in 0..4 {
            let Some((s, _)) = t.successors(prev) else {
                break;
            };
            let chain_token = s[0] as usize;
            assert!(
                tree.iter()
                    .any(|n| n.depth == depth && n.token_idx == chain_token),
                "greedy token {chain_token} missing at depth {depth}"
            );
            prev = s[0];
        }
    }

    #[test]
    fn g3_buffer_reuse_content_correct() {
        // Two predicts into the same buffer: entries from call 1 that are not
        // rewritten by call 2 must be reset to 0 (the touched invariant).
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 0, 2, 1, 3, 2, 3]);
        let t = b.build(4, 4);
        let mut buf = BigramMarginalBuffer::new(1, 4);
        bigram_predict(&t, 0, 1, &mut buf);
        assert!(buf.row(0)[1] > 0.0 && buf.row(0)[2] > 0.0);
        bigram_predict(&t, 1, 1, &mut buf);
        // Call 2 writes token 3 only; tokens 1, 2 must be reset.
        assert_eq!(buf.row(0)[1], 0.0);
        assert_eq!(buf.row(0)[2], 0.0);
        assert!(buf.row(0)[3] > 0.0);
    }

    #[test]
    fn g3_build_tree_unseen_root_empty_tree() {
        // Unseen root → all-zero rows → the seam proposes nothing.
        let t = build_toy(8, 4);
        let config = katgpt_types::Config::micro();
        let mut buf = BigramMarginalBuffer::new(config.draft_lookahead, 8);
        let tree = bigram_build_tree(&t, 7, &config, &mut buf);
        assert!(tree.is_empty(), "zero rows must yield an empty tree");
    }

    /// G4 — steady-state emission performs zero allocations. A thread-local
    /// tracking allocator counts allocations made by THIS thread while the
    /// probe is armed (immune to parallel-test noise from other threads).
    #[test]
    fn g4_predict_zero_alloc_steady_state() {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        struct Counting;
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                if TRACK.with(|t| t.get()) {
                    COUNT.with(|c| c.set(c.get() + 1));
                }
                unsafe { System.alloc(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                unsafe { System.dealloc(ptr, layout) }
            }
        }
        #[global_allocator]
        static A: Counting = Counting;

        thread_local! {
            static TRACK: Cell<bool> = const { Cell::new(false) };
            static COUNT: Cell<usize> = const { Cell::new(0) };
        }

        let t = build_toy(8, 4);
        let mut buf = BigramMarginalBuffer::new(4, 8);
        bigram_predict(&t, 0, 4, &mut buf); // warm-up (touched may grow)

        TRACK.with(|t| t.set(true));
        COUNT.with(|c| c.set(0));
        let steps = bigram_predict(&t, 1, 4, &mut buf);
        TRACK.with(|t| t.set(false));
        let allocs = COUNT.with(|c| c.get());
        assert_eq!(steps, 4);
        assert_eq!(
            allocs, 0,
            "steady-state bigram_predict must be allocation-free"
        );
        assert!(buf.touched_len() <= 4 * t.top_m());
    }

    #[test]
    fn g5_memory_bytes_formula() {
        let mut b = BigramMarkovBuilder::new();
        b.add_sequence(&[0, 1, 2, 3]);
        let t = b.build(8, 4);
        // Exact accounting: offsets (vocab+1) + successors + probs + totals.
        let n_stored: usize = (0..8u32)
            .map(|p| t.successors(p).map_or(0, |s| s.0.len()))
            .sum();
        let exact = (9 + n_stored + n_stored + 8) * 4;
        assert_eq!(n_stored, 3, "(0,1),(1,2),(2,3)");
        assert_eq!(t.memory_bytes(), exact);

        // Bonsai-scale worst-case projection (documented, not built):
        // vocab 131_072, top_m 16, every row full.
        let bonsai = (131_073 + 131_072 * 16 * 2 + 131_072) * 4;
        assert_eq!(bonsai, 17_825_796, "≈17.0 MiB — the G5 bound claim");
    }

    /// Release-mode cost probe: the mode-2 avoidance made concrete. Sparse
    /// emission at Bonsai scale (vocab 131_072, top_m 16, lookahead 8) must
    /// run in ~µs — versus a 6-layer drafter forward per step (~129 µs/step
    /// per token at the Bench 661 E overhead scale).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn release_bonsai_scale_emission_cost_probe() {
        // Deterministic synthetic corpus: concentrated successor locality.
        let vocab: usize = 131_072;
        let seqs = 64;
        let len = 4_096;
        let mut b = BigramMarkovBuilder::with_capacity(seqs * len);
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15; // golden-ratio seed
        let mut xorshift = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..seqs {
            let mut tokens = Vec::with_capacity(len);
            let mut cur = (xorshift() % vocab as u64) as u32;
            tokens.push(cur);
            for _ in 1..len {
                let r = xorshift();
                let next = if r % 4 == 0 {
                    (xorshift() % vocab as u64) as u32
                } else {
                    cur.wrapping_add(1 + (r % 17) as u32) % vocab as u32
                };
                tokens.push(next);
                cur = next;
            }
            b.add_sequence(&tokens);
        }
        let t = b.build(vocab, 16);
        let mut buf = BigramMarginalBuffer::new(8, vocab);
        bigram_predict(&t, 0, 8, &mut buf); // warm-up

        let iters = 1_000;
        let start = std::time::Instant::now();
        for i in 0..iters {
            bigram_predict(&t, (i % 128) as u32, 8, &mut buf);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        let per_step = per_call / 8.0;
        println!(
            "bigram emission @ vocab={vocab} top_m=16 lookahead=8: \
             {per_call:.0} ns/call ({per_step:.0} ns/step, {} touched)",
            buf.touched_len()
        );
        // Generous regression bound: sparse emission must stay µs-scale (a
        // dense fill alone would be ~500 µs/call at this scale).
        assert!(
            per_call < 100_000.0,
            "sparse emission must stay µs-scale, got {per_call:.0} ns/call"
        );
    }

    // ── Issue 659 T4 (hardware-independent half): acceptance gate ───
    //
    // G2 asks for "acceptance rate vs the DFlash baseline at equal draft
    // depth". The *literal* DFlash arm needs TRAINED low-rank weights
    // (Prism-ML's DSpark head) and the Bonsai target to verify against, so it
    // belongs to the riir-ai consumer gate — as does G3 wall-clock. What IS
    // measurable modellessly, here, today, is the structural question the
    // DFlash pattern raises: DFlash is a FACTORIZED head — its per-depth
    // marginals do not condition on the drafted prefix, which is exactly the
    // deep-position dilution Plan 424 T6.2 records. The bigram head
    // conditions every depth on its greedy predecessor. So at equal draft
    // depth, equal tree budget, equal vocabulary, on a held-out split of a
    // fixed corpus, we measure three arms:
    //
    //   arm F (factorized) — position-independent unigram marginals: the
    //                        modelless FLOOR for a non-conditioning head.
    //                        Not trained DFlash; a floor, and labelled one.
    //   arm B (bigram)     — this primitive, via `bigram_build_tree`.
    //   arm C (chain)      — the greedy argmax chain alone, no tree, to
    //                        separate "conditioning" from "tree branching".
    //
    // Metric: mean acceptance length = tokens of the drafted best chain that
    // match the held-out continuation before the first mismatch (the count a
    // lossless verifier would commit per draft cycle).

    /// Fixed corpus (embedded so the numbers are reproducible forever — a
    /// repo file would drift commit to commit). Byte-level tokens, vocab 256.
    const ACCEPTANCE_CORPUS: &str = "\
The assessment of a river valley begins with the water itself. A river carries \
sediment from the highlands toward the sea, and the sediment settles wherever \
the current slows. Where the channel widens the water slows and the coarse sand \
drops first; the finer clay travels farther and settles in the still reaches \
near the mouth. Over many seasons this sorting builds a delta, a fan of low \
islands separated by shallow channels that shift from one year to the next. \
The surveyor who returns to the same bend after a decade finds the bank moved, \
the gravel bar grown, the side channel closed and a new one opened downstream. \
Nothing in the valley is fixed except the slope, and even the slope yields to \
the patient work of the water over a long enough interval of time. \
A map of such a valley is therefore a claim about a moment, not about the \
ground. The careful reader of maps asks when the survey was made before asking \
where the channel ran. Two maps of the same river drawn thirty years apart will \
disagree about the position of every bar and bend, and both may be correct. \
The disagreement is the signal: it measures how much sediment the river moved \
between the surveys, and from that the surveyor estimates the load the river \
carries in an ordinary year. Where the maps agree, the bank is armoured by rock \
or by the roots of old trees, and the river has spent its energy elsewhere. \
Downstream of the delta the water spreads and loses the last of its load. The \
sea returns some of it to the shore as a beach, and the wind carries the driest \
of the sand inland to build low dunes behind the beach. Grasses root in the \
dunes and hold them, and behind that shelter a marsh forms in the still water. \
The marsh traps more sediment than the beach and rises faster, and in time the \
marsh becomes meadow and the meadow becomes forest, and the shore has moved \
seaward by the width of the whole sequence. Every stage of that succession is \
readable in a core taken through the ground beneath the forest floor, where the \
sand of the old beach lies under the peat of the old marsh under the soil. \
The same reasoning applies to the survey of a mountain front, where the rock \
rather than the water sets the pace of change. A glacier grinds the valley into \
the shape of a trough, and when the ice retreats the trough remains, too wide \
and too deep for the small stream that now runs along its floor. The surveyor \
who measures the stream and the valley together sees at once that the one did \
not cut the other, and looks for the ice that did. Above the trough the rock \
walls are polished where the ice pressed against them and shattered where the \
water froze in the joints, and the difference between the polish and the \
shatter marks the highest level the ice reached in the last cold season. \
Below the front of the old glacier a ridge of unsorted rubble marks where the \
ice stood longest, and beyond the ridge a plain of sorted gravel marks where \
the meltwater ran. The rubble is the work of the ice, which carries every size \
of rock without regard to weight, and the sorted gravel is the work of the \
water, which cannot. That single contrast lets the surveyor read the boundary \
between ice and water on the ground long after both have gone. Where a later \
stream has cut through the ridge, the cut exposes the unsorted rubble in \
section, and the section shows how many times the ice advanced and withdrew, \
one layer of rubble for each advance, separated by the soils of the warm \
intervals between them. Counting the layers counts the cold seasons. \
The valley and the shore therefore record the same history in different hands. \
The shore writes in sand and peat and reads forward from the sea; the valley \
writes in rubble and gravel and reads down from the ice. A survey that uses \
only one of the two records will date the history correctly and explain it \
wrongly, because each record preserves the events that moved its own material \
and is silent about the rest. The surveyor who walks from the ice front to the \
sea in one season, and writes down the slope and the sediment at every bend \
along the way, holds the whole of it, and can say not merely when the ground \
changed but which of the several agents of change was at work at each place \
and in what order the agents took their turns upon the ground.";

    /// Tokens a **tree** verifier commits: the longest root-to-leaf path in
    /// the tree that matches `target` from depth 0. This — not the
    /// highest-*scored* chain — is the acceptance a tree-structured
    /// speculative decoder realises, because the verifier checks every path
    /// in one batched forward and commits the longest match.
    ///
    /// A node's `parent_path` holds its whole ancestor chain including
    /// itself, one `u32` token per level, root at slot 0
    /// (`tree_builder`: `node_path = parent_path.push(token, depth)`).
    fn tree_acceptance(tree: &[TreeNode], target: &[u32]) -> usize {
        let mut best = 0usize;
        for n in tree {
            if n.depth + 1 > target.len() || n.depth < best {
                continue;
            }
            let matches = (0..=n.depth).all(|k| n.parent_path.token_at(k) == target[k]);
            if matches {
                best = n.depth + 1;
            }
        }
        best
    }

    /// Greedy argmax chain, no tree (arm C).
    fn greedy_chain(table: &BigramMarkovTable, last: u32, steps: usize) -> Vec<usize> {
        let mut prev = last;
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            match table.successors(prev) {
                Some((s, _)) => {
                    out.push(s[0] as usize);
                    prev = s[0];
                }
                None => break,
            }
        }
        out
    }

    /// Tokens matching the target before the first mismatch.
    fn acceptance_len(path: &[usize], target: &[u32]) -> usize {
        path.iter()
            .zip(target.iter())
            .take_while(|(a, b)| **a == **b as usize)
            .count()
    }

    /// Word-level tokenisation (whitespace split, first-appearance ids). The
    /// tokenizer is built over the whole corpus — standard practice; only the
    /// bigram *table* is fitted on the train split, so held-out words the
    /// train split never saw become zero rows, as they would in production.
    fn word_tokens(corpus: &str) -> (Vec<u32>, usize) {
        let mut ids = std::collections::HashMap::new();
        let mut out = Vec::new();
        for w in corpus.split_whitespace() {
            let n = ids.len() as u32;
            let id = *ids.entry(w).or_insert(n);
            out.push(id);
        }
        let vocab = ids.len();
        (out, vocab)
    }

    /// One (budget × top_m) sweep.
    /// Returns `(budget, top_m, bigram, chain, floor, zero_row_pct)`.
    fn acceptance_sweep(
        tokens: &[u32],
        vocab: usize,
        label: &str,
    ) -> Vec<(usize, usize, f64, f64, f64, f64)> {
        let split = tokens.len() * 4 / 5;
        let (train, test) = tokens.split_at(split);

        let mut config = katgpt_types::Config::draft();
        config.vocab_size = vocab;
        config.draft_lookahead = 8;
        let look = config.draft_lookahead;

        // Arm F: unigram marginals from the SAME train split, repeated at
        // every depth (a non-conditioning head proposes the same chain
        // everywhere — that is the property being measured, not a bug).
        let mut uni = vec![0.0f32; vocab];
        for &t in train {
            uni[t as usize] += 1.0;
        }
        let total: f32 = uni.iter().sum();
        for p in uni.iter_mut() {
            *p /= total;
        }
        let uni_rows: Vec<&[f32]> = (0..look).map(|_| uni.as_slice()).collect();

        println!(
            "--- {label}: {} tokens (train {} / held-out {}), vocab {vocab}, lookahead {look} ---",
            tokens.len(),
            train.len(),
            test.len(),
        );
        println!(
            "  {:>6} {:>7} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "budget", "top_m", "B (bigram)", "C (chain)", "F (floor)", "B/F", "0-row%"
        );
        println!("  {}", "-".repeat(73));

        let mut results = Vec::new();
        for &budget in &[16usize, 64, 256] {
            config.tree_budget = budget;
            let fac_tree = crate::dd_tree::build_dd_tree(&uni_rows, &config);
            for &top_m in &[1usize, 4, 16] {
                let mut b = BigramMarkovBuilder::new();
                b.add_sequence(train);
                let table = b.build(vocab, top_m);
                let mut buf = BigramMarginalBuffer::new(look, vocab);

                let (mut sb, mut sc, mut sf, mut n, mut zero) = (0usize, 0usize, 0usize, 0, 0usize);
                for i in 0..test.len().saturating_sub(look + 1) {
                    let prev = test[i];
                    let target = &test[i + 1..i + 1 + look];
                    let tree = bigram_build_tree(&table, prev, &config, &mut buf);
                    sb += tree_acceptance(&tree, target);
                    sc += acceptance_len(&greedy_chain(&table, prev, look), target);
                    sf += tree_acceptance(&fac_tree, target);
                    // Held-out prev the train split never saw → zero row →
                    // the head proposes nothing. High rate = data starvation,
                    // which confounds any quality claim.
                    if table.successors(prev).is_none() {
                        zero += 1;
                    }
                    n += 1;
                }
                assert!(n >= 150, "held-out split too small, got {n}");
                let (mb, mc, mf) = (
                    sb as f64 / n as f64,
                    sc as f64 / n as f64,
                    sf as f64 / n as f64,
                );
                let zr = 100.0 * zero as f64 / n as f64;
                let ratio = if mf > 0.0 {
                    format!("{:.2}x", mb / mf)
                } else {
                    "inf".to_string()
                };
                println!(
                    "  {budget:>6} {top_m:>7} {mb:>12.4} {mc:>12.4} {mf:>12.4} {ratio:>8} {zr:>7.1}"
                );
                results.push((budget, top_m, mb, mc, mf, zr));
            }
            println!();
        }
        results.shrink_to_fit();
        results
    }

    #[test]
    fn g2_acceptance_bigram_vs_factorized_floor_heldout() {
        println!(
            "=== Issue 659 T4 (G2, hardware-independent): mean acceptance length ===\n\
             metric: mean tokens a lossless TREE verifier commits per draft cycle\n\
             F = factorized floor (position-independent unigram marginals),\n\
             NOT trained DFlash — that arm needs riir-ai's Bonsai target.\n"
        );

        let bytes: Vec<u32> = ACCEPTANCE_CORPUS.bytes().map(u32::from).collect();
        let dense = acceptance_sweep(&bytes, 256, "byte-level (DENSE vocab, 256)");

        let (words, wvocab) = word_tokens(ACCEPTANCE_CORPUS);
        let sparse = acceptance_sweep(&words, wvocab, "word-level (SPARSE vocab)");

        // ── Structural invariants — must hold in BOTH regimes ──
        for (label, results) in [("byte", &dense), ("word", &sparse)] {
            for &(budget, top_m, mb, mc, _, _) in results {
                let at = format!("{label} budget={budget} top_m={top_m}");
                // The tree contains the greedy chain, so it cannot accept less.
                assert!(
                    mb >= mc - 1e-9,
                    "{at}: tree ({mb:.4}) lost to the bare chain ({mc:.4})"
                );
                // At top_m=1 the table has no siblings: the tree IS the chain.
                if top_m == 1 {
                    assert!(
                        (mb - mc).abs() < 1e-9,
                        "{at}: tree ({mb:.4}) must equal the chain ({mc:.4})"
                    );
                }
            }
            // More successors cannot reduce what a wide budget can match.
            let wide = results.iter().find(|r| r.0 == 256 && r.1 == 16).unwrap();
            let narrow = results.iter().find(|r| r.0 == 256 && r.1 == 1).unwrap();
            assert!(
                wide.2 >= narrow.2 - 1e-9,
                "{label}: at budget=256, top_m=16 ({:.4}) must not lose to top_m=1 ({:.4})",
                wide.2,
                narrow.2
            );
        }

        // ── G2, byte-level: the one regime this corpus can actually fit ──
        //
        // 3.4 k training tokens over a 256-symbol vocabulary is a well-fitted
        // bigram model (the 0-row column confirms it: ~0% of held-out prevs
        // are unseen). Here the head has a real, measured edge over the
        // coverage floor at its intended operating point.
        for &(budget, top_m, mb, mc, mf, zr) in dense.iter() {
            let at = format!("byte budget={budget} top_m={top_m}");
            assert!(zr < 5.0, "{at}: {zr:.1}% zero rows — corpus no longer fits");
            assert!(mb > 0.4, "{at}: no held-out signal ({mb:.4})");
            assert!(mc > 0.4, "{at}: chain shows no held-out signal ({mc:.4})");
            if top_m == 16 {
                assert!(
                    mb > mf,
                    "{at}: bigram ({mb:.4}) must beat the factorized floor ({mf:.4}) \
                     at the intended operating point"
                );
            }
        }
        // MEASURED SCOPE LIMIT (pinned so it cannot rot): the floor is a
        // *coverage* strategy — spend the budget enumerating the globally most
        // frequent tokens at every depth. Its strength scales with budget,
        // while a top_m=1 head proposes exactly one chain no matter the
        // budget. So at top_m=1 the floor WINS, and the head's margin shrinks
        // as budget grows (1.39x at budget 16 → 1.09x at budget 256).
        // Consequence for the consumer: this head must be run with a wide
        // top_m, and it is most valuable under TIGHT budget — which is the
        // Metal batch-1 regime it is proposed for.
        let d1 = dense.iter().find(|r| r.0 == 256 && r.1 == 1).unwrap();
        assert!(
            d1.2 < d1.4,
            "byte budget=256 top_m=1: floor no longer wins ({:.4} vs {:.4}) — \
             re-derive the scope note",
            d1.2,
            d1.4
        );
        let (tight, loose) = (
            dense.iter().find(|r| r.0 == 16 && r.1 == 16).unwrap(),
            dense.iter().find(|r| r.0 == 256 && r.1 == 16).unwrap(),
        );
        assert!(
            tight.2 / tight.4 > loose.2 / loose.4,
            "the head's edge over the floor must shrink as budget grows \
             (tight {:.2}x vs loose {:.2}x)",
            tight.2 / tight.4,
            loose.2 / loose.4
        );

        // ── Word-level: DATA-STARVED, NOT a quality verdict ──
        //
        // 636 training words over a 356-word vocabulary cannot fit a bigram
        // model: most held-out prevs were never seen, so the head emits a zero
        // row and proposes nothing (that is the honest designed behaviour, not
        // a bug — see the module docs). The floor needs no conditioning and so
        // is unaffected. This arm therefore measures CORPUS SIZE, not the
        // sparse-vocabulary question it was meant to probe, and NO quality
        // conclusion is drawn from it. Answering the sparse-vocab question
        // needs a Bonsai-scale corpus + the Bonsai target — the riir-ai
        // consumer gate (Issue 659 T4). The assertion below pins the confound
        // itself, so this note cannot be mistaken for a measured verdict.
        let w = sparse.iter().find(|r| r.1 == 16).unwrap();
        assert!(
            w.5 > 25.0,
            "word-level arm was expected to be data-starved (zero-row rate \
             {:.1}%); if the corpus now fits, this arm has become a real \
             measurement and the note above must be rewritten",
            w.5
        );
    }

    // Small accessors used by g1_row_offsets_sparse_prevs.
    impl BigramMarkovTable {
        fn row_offsets_len(&self) -> usize {
            self.row_offsets.len()
        }
        fn row_bounds(&self, prev: usize) -> (usize, usize) {
            (
                self.row_offsets[prev] as usize,
                self.row_offsets[prev + 1] as usize,
            )
        }
    }
}

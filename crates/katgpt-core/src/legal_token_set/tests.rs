//! Arms for the legal-token-set seam (Issue 841).
//!
//! The load-bearing ones are t04 (ascending order — the property that makes an
//! enumerating consumer bit-identical rather than merely equivalent), t09 (the
//! `None`-vs-`Some(0)` split, the silent direction), and t14 (the measured
//! crossover is honoured, so a small set does not automatically gather).

use super::*;
use crate::traits::ConstraintPruner;

const NO_EDGE: usize = usize::MAX;

/// A DFA whose δ is a dense row-major table, i.e. the shipped
/// `LodestarAutomaton` layout — the source `from_dense` exists for.
fn dense(n_states: usize, vocab: usize, edges: &[(usize, usize, usize)]) -> Vec<usize> {
    let mut t = vec![NO_EDGE; n_states * vocab];
    for &(s, tok, next) in edges {
        t[s * vocab + tok] = next;
    }
    t
}

// ── CsrLegalSet ────────────────────────────────────────────────────────

#[test]
fn t01_from_dense_recovers_exactly_the_edges() {
    let t = dense(3, 8, &[(0, 2, 1), (0, 5, 2), (1, 0, 2), (2, 7, 0)]);
    let csr = CsrLegalSet::from_dense(&t, 3, 8, NO_EDGE);
    assert_eq!(csr.row(0), &[2, 5]);
    assert_eq!(csr.row(1), &[0]);
    assert_eq!(csr.row(2), &[7]);
    assert_eq!(csr.n_edges(), 4);
    assert_eq!(csr.n_states(), 3);
    assert_eq!(csr.vocab_size(), 8);
}

#[test]
fn t02_degree_matches_row_len_for_every_state() {
    let t = dense(4, 6, &[(0, 1, 1), (0, 3, 1), (0, 5, 1), (2, 0, 3)]);
    let csr = CsrLegalSet::from_dense(&t, 4, 6, NO_EDGE);
    for s in 0..4 {
        assert_eq!(csr.degree(s), csr.row(s).len(), "state {s}");
    }
    assert_eq!(csr.degree(0), 3);
    assert_eq!(csr.degree(1), 0);
    assert_eq!(csr.degree(3), 0);
}

#[test]
fn t03_csr_agrees_with_a_dense_row_scan_on_every_cell() {
    // The equivalence the whole optimisation rests on, asserted over the
    // FULL product rather than sampled: a row scan of the dense table and a
    // `contains` on the index must never disagree.
    let t = dense(
        5,
        16,
        &[
            (0, 0, 1),
            (0, 15, 2),
            (1, 7, 1),
            (3, 1, 4),
            (3, 2, 4),
            (3, 3, 4),
            (4, 8, 0),
        ],
    );
    let csr = CsrLegalSet::from_dense(&t, 5, 16, NO_EDGE);
    for s in 0..5 {
        for tok in 0..16 {
            let dense_says = t[s * 16 + tok] != NO_EDGE;
            assert_eq!(csr.contains(s, tok), dense_says, "state {s} token {tok}");
        }
    }
}

#[test]
fn t04_rows_are_strictly_ascending() {
    // The contract clause that buys bit-identity, not just equivalence.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for tok in [9usize, 1, 4, 0, 7, 2] {
        edges.push((0, tok));
    }
    let csr = CsrLegalSet::from_edges(1, 10, edges);
    assert_eq!(csr.row(0), &[0, 1, 2, 4, 7, 9]);
    assert!(csr.row(0).windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn t05_from_edges_dedups_and_drops_out_of_range() {
    let csr = CsrLegalSet::from_edges(2, 4, [(0, 1), (0, 1), (0, 3), (5, 0), (1, 9), (1, 2)]);
    assert_eq!(csr.row(0), &[1, 3]);
    assert_eq!(csr.row(1), &[2]);
    assert_eq!(csr.n_edges(), 3);
}

#[test]
fn t06_out_of_range_state_reads_empty_not_panic() {
    let csr = CsrLegalSet::from_edges(2, 4, [(0, 1)]);
    assert!(csr.row(99).is_empty());
    assert_eq!(csr.degree(99), 0);
    assert!(!csr.contains(99, 1));
    assert!(!csr.contains(0, usize::MAX));
}

#[test]
fn t07_truncated_dense_table_yields_fewer_edges_never_wrong_ones() {
    // A short table must never index into the NEXT state's row.
    let full = dense(3, 4, &[(0, 1, 1), (1, 2, 2), (2, 3, 0)]);
    let truncated = &full[..6]; // state 0 whole, state 1 half
    let csr = CsrLegalSet::from_dense(truncated, 3, 4, NO_EDGE);
    assert_eq!(csr.row(0), &[1]);
    assert!(csr.row(1).is_empty(), "token 2 of state 1 is past the cut");
    assert!(csr.row(2).is_empty());
}

#[test]
fn t08_memory_beats_the_dense_table_it_replaces() {
    // The second, independent reason to hold an index — asserted, because the
    // type docs claim it.
    let n_states = 64;
    let vocab = 32_768;
    let edges: Vec<(usize, usize)> = (0..n_states)
        .flat_map(|s| (0..8).map(move |k| (s, s * 11 + k)))
        .collect();
    let csr = CsrLegalSet::from_edges(n_states, vocab, edges);
    let dense_bytes = 8 * n_states * vocab;
    assert!(
        csr.memory_bytes() * 100 < dense_bytes,
        "csr {} vs dense {dense_bytes}",
        csr.memory_bytes()
    );
}

// ── The trait seam ─────────────────────────────────────────────────────

/// A grammar-shaped pruner: legal set comes from the automaton, not a scan.
struct GrammarPruner {
    csr: CsrLegalSet,
    /// `parent_tokens.len() % n_states` stands in for a real δ* walk — the
    /// arms here are about the SEAM, not about path following.
    n_states: usize,
}

impl GrammarPruner {
    fn state(&self, parent_tokens: &[usize]) -> usize {
        parent_tokens.len() % self.n_states
    }
}

impl ConstraintPruner for GrammarPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        self.csr.contains(self.state(parent_tokens), token_idx)
    }

    fn legal_degree(&self, _depth: usize, parent_tokens: &[usize]) -> Option<usize> {
        Some(self.csr.degree(self.state(parent_tokens)))
    }

    fn for_each_legal(&self, _depth: usize, parent_tokens: &[usize], f: &mut dyn FnMut(usize)) {
        self.csr.for_each(self.state(parent_tokens), f);
    }
}

/// A pruner that cannot enumerate — every pre-existing implementor.
struct OpaquePruner;
impl ConstraintPruner for OpaquePruner {
    fn is_valid(&self, _d: usize, token_idx: usize, _p: &[usize]) -> bool {
        token_idx.is_multiple_of(3)
    }
}

fn grammar() -> GrammarPruner {
    GrammarPruner {
        csr: CsrLegalSet::from_edges(3, 32, [(0, 4), (0, 9), (0, 30), (1, 7), (2, 0), (2, 1)]),
        n_states: 3,
    }
}

#[test]
fn t09_default_hooks_mean_no_answer_not_the_empty_set() {
    // ⛔ The silent direction: the default `for_each_legal` yields nothing,
    // and nothing must be distinguishable from "no token is legal".
    let p = OpaquePruner;
    assert_eq!(p.legal_degree(0, &[]), None);
    let mut seen = Vec::new();
    p.for_each_legal(0, &[], &mut |t| seen.push(t));
    assert!(seen.is_empty());
    // The distinguishing read is the plan, not the yield count.
    let plan = plan_projection(p.legal_degree(0, &[]), 32, &RestrictionPolicy::default());
    assert!(plan.is_unenumerable());
    assert_ne!(plan, ProjectionPlan::Dead);
}

#[test]
fn t10_enumeration_and_is_valid_agree_over_the_whole_vocabulary() {
    // Exactness clause 1 of the contract, over the full product.
    let p = grammar();
    for parents_len in 0..6 {
        let parents: Vec<usize> = vec![0; parents_len];
        let mut enumerated = Vec::new();
        p.for_each_legal(0, &parents, &mut |t| enumerated.push(t));
        let scanned: Vec<usize> = (0..32).filter(|&t| p.is_valid(0, t, &parents)).collect();
        assert_eq!(enumerated, scanned, "parents_len {parents_len}");
    }
}

#[test]
fn t11_degree_matches_the_yield_count() {
    // Cardinality clause 2 — the one a caller re-checks in debug.
    let p = grammar();
    for parents_len in 0..6 {
        let parents: Vec<usize> = vec![0; parents_len];
        let mut n = 0;
        p.for_each_legal(0, &parents, &mut |_| n += 1);
        assert_eq!(p.legal_degree(0, &parents), Some(n));
    }
}

#[test]
fn t12_first_legal_only_answers_at_degree_one() {
    let p = grammar();
    assert_eq!(first_legal(&p, 0, &[]), None, "state 0 has 3 legal");
    assert_eq!(first_legal(&p, 0, &[0]), Some(7), "state 1 is forced");
    assert_eq!(first_legal(&p, 0, &[0, 0]), None, "state 2 has 2 legal");
    assert_eq!(first_legal(&OpaquePruner, 0, &[]), None);
}

#[test]
fn t13_collect_leaves_the_buffer_untouched_when_unenumerable() {
    // A cleared buffer would read as "no token is legal" to the next caller.
    let mut buf = vec![11usize, 22, 33];
    assert_eq!(collect_legal_into(&OpaquePruner, 0, &[], &mut buf), None);
    assert_eq!(buf, vec![11, 22, 33]);

    assert_eq!(collect_legal_into(&grammar(), 0, &[], &mut buf), Some(3));
    assert_eq!(buf, vec![4, 9, 30]);
}

// ── ProjectionPlan ─────────────────────────────────────────────────────

#[test]
fn t14_a_small_set_does_not_automatically_gather() {
    // The measured-crossover rule. 3000 of 32768 is 9.2% — under the 10%
    // default; 4000 is 12.2% — over it, and takes the dense pass even though
    // it is "much smaller than the vocabulary".
    let pol = RestrictionPolicy::default();
    assert_eq!(
        plan_projection(Some(3000), 32_768, &pol),
        ProjectionPlan::Restricted { degree: 3000 }
    );
    assert_eq!(
        plan_projection(Some(4000), 32_768, &pol),
        ProjectionPlan::Full {
            unenumerable: false
        }
    );
}

#[test]
fn t15_plan_separates_dead_forced_restricted_full() {
    let pol = RestrictionPolicy::default();
    assert_eq!(plan_projection(Some(0), 1000, &pol), ProjectionPlan::Dead);
    assert_eq!(plan_projection(Some(1), 1000, &pol), ProjectionPlan::Forced);
    assert_eq!(
        plan_projection(Some(10), 1000, &pol),
        ProjectionPlan::Restricted { degree: 10 }
    );
    assert_eq!(
        plan_projection(None, 1000, &pol),
        ProjectionPlan::Full { unenumerable: true }
    );
}

#[test]
fn t16_skip_forced_false_keeps_the_logits_of_a_forced_step() {
    let pol = RestrictionPolicy {
        skip_forced: false,
        ..Default::default()
    };
    // Degree 1 of 1000 is far under the fraction, so it gathers one row
    // rather than skipping — the caller still gets a number.
    assert_eq!(
        plan_projection(Some(1), 1000, &pol),
        ProjectionPlan::Restricted { degree: 1 }
    );
}

#[test]
fn t17_dense_policy_never_gathers_and_always_is_the_control() {
    assert_eq!(
        plan_projection(Some(1), 1000, &RestrictionPolicy::DENSE),
        ProjectionPlan::Full {
            unenumerable: false
        }
    );
    assert_eq!(
        plan_projection(Some(999), 1000, &RestrictionPolicy::ALWAYS),
        ProjectionPlan::Restricted { degree: 999 }
    );
    // Even ALWAYS refuses a degree that is the whole vocabulary — a "gather"
    // of every row is the dense pass with an index in front of it.
    assert_eq!(
        plan_projection(Some(1000), 1000, &RestrictionPolicy::ALWAYS),
        ProjectionPlan::Full {
            unenumerable: false
        }
    );
}

#[test]
fn t18_zero_vocab_falls_through_to_full_not_to_a_gather() {
    // `degree / 0` would be an infinity that compares true against every
    // budget; the multiply form yields a 0 budget instead.
    assert_eq!(
        plan_projection(Some(5), 0, &RestrictionPolicy::ALWAYS),
        ProjectionPlan::Full {
            unenumerable: false
        }
    );
}

#[test]
fn t19_rows_scored_accounts_each_arm() {
    assert_eq!(ProjectionPlan::Dead.rows_scored(32_768), 0);
    assert_eq!(ProjectionPlan::Forced.rows_scored(32_768), 0);
    assert_eq!(
        ProjectionPlan::Restricted { degree: 12 }.rows_scored(32_768),
        12
    );
    assert_eq!(
        ProjectionPlan::Full { unenumerable: true }.rows_scored(32_768),
        32_768
    );
}

#[test]
fn t20_full_from_too_large_is_not_pooled_with_full_from_unenumerable() {
    let a = plan_projection(Some(30_000), 32_768, &RestrictionPolicy::default());
    let b = plan_projection(None, 32_768, &RestrictionPolicy::default());
    assert!(!a.is_unenumerable());
    assert!(b.is_unenumerable());
    assert_ne!(a, b);
}

//! Correctness tests for the LIMIT fixture generator (Issue 580 T1).
//!
//! These assert the *construction* is faithful — that the fixture really is the
//! adversarial structure the paper describes. Retrieval quality measured on it
//! lives in `tests/limit_recall.rs` and in riir-neuron-db.

use super::*;

#[test]
fn paper_small_shape() {
    let f = build_limit(&LimitConfig::paper_small(7));
    assert_eq!(f.n_relevant, 46);
    assert_eq!(f.docs.len(), 46, "small variant has no distractors");
    assert_eq!(f.queries.len(), 1000, "paper emits 1000 queries");
    assert!(!f.synonyms);
    // C(46,2) = 1035 is the smallest binomial above 1000 — the paper's sizing
    // rule. Assert we could not have asked for many more.
    assert_eq!(combinations(46, 2, usize::MAX).len(), 1035);
}

#[test]
fn every_query_has_exactly_k_distinct_relevant_docs() {
    let f = build_limit(&LimitConfig::paper_small(11));
    for (qi, q) in f.queries.iter().enumerate() {
        assert_eq!(q.relevant.len(), 2, "query {qi}");
        assert_ne!(
            q.relevant[0], q.relevant[1],
            "query {qi} has a duplicate target"
        );
        for &d in &q.relevant {
            assert!(d < f.n_relevant, "query {qi} targets a distractor");
        }
    }
}

/// The defining property of the construction: no two queries share a relevant
/// set. If they did, the qrel matrix would be less dense than intended and the
/// fixture would be easier than LIMIT.
#[test]
fn relevant_sets_are_unique_across_queries() {
    let f = build_limit(&LimitConfig::paper_small(13));
    let mut seen: Vec<Vec<usize>> = f.queries.iter().map(|q| q.relevant.clone()).collect();
    let before = seen.len();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), before, "duplicate relevant sets found");
}

/// Ground truth must be *exactly* the documents carrying the query attribute —
/// otherwise recall is measured against the wrong answer key.
#[test]
fn attribute_appears_in_exactly_the_relevant_docs() {
    let f = build_limit(&LimitConfig::paper_small(17));
    for (qi, q) in f.queries.iter().enumerate() {
        let carriers: Vec<usize> = f
            .docs
            .iter()
            .enumerate()
            .filter(|(_, d)| d.attributes.contains(&q.attribute))
            .map(|(i, _)| i)
            .collect();
        let mut want = q.relevant.clone();
        want.sort_unstable();
        assert_eq!(
            carriers, want,
            "query {qi}: attribute carriers != relevant set"
        );
    }
}

/// Padding must not leak signal: every document carries the same number of
/// attributes, so attribute count cannot be used to rank.
#[test]
fn attribute_counts_are_uniform() {
    let f = build_limit(&LimitConfig::paper_small(19));
    let n = f.docs[0].attributes.len();
    assert!(
        n >= 4,
        "expected at least attrs_per_doc attributes, got {n}"
    );
    for (i, d) in f.docs.iter().enumerate() {
        assert_eq!(
            d.attributes.len(),
            n,
            "doc {i} has a different attribute count"
        );
    }
}

/// The synonym variant must preserve the relevance structure byte-identically
/// while removing the lexical overlap. If either half failed, the plain-vs-synonym
/// comparison would not isolate lexical matching.
#[test]
fn synonym_variant_preserves_structure_but_removes_overlap() {
    let plain = build_limit(&LimitConfig::paper_small(23));
    let syn = build_limit(&LimitConfig::paper_small(23).with_synonyms());

    assert_eq!(plain.queries.len(), syn.queries.len());
    for (p, s) in plain.queries.iter().zip(&syn.queries) {
        assert_eq!(
            p.relevant, s.relevant,
            "relevance structure must be identical"
        );
        assert_eq!(p.attribute, s.attribute, "query surface must be identical");
    }

    // Plain: the query attribute token appears in its relevant documents.
    for q in &plain.queries {
        for &d in &q.relevant {
            assert!(
                plain.docs[d].text.contains(&q.attribute),
                "plain variant must have lexical overlap"
            );
        }
    }
    // Synonym: it does not appear anywhere in the corpus.
    for q in &syn.queries {
        for &d in &q.relevant {
            assert!(
                !syn.docs[d].text.contains(&q.attribute),
                "synonym variant must NOT have lexical overlap for '{}'",
                q.attribute
            );
        }
    }
}

#[test]
fn full_variant_adds_distractors_relevant_to_nothing() {
    // Trimmed distractor count keeps the test fast; the property is the same.
    let cfg = LimitConfig {
        n_distractors: 500,
        ..LimitConfig::paper_small(29)
    };
    let f = build_limit(&cfg);
    assert_eq!(f.docs.len(), 546);
    let all_relevant: Vec<usize> = f.queries.iter().flat_map(|q| q.relevant.clone()).collect();
    for (i, d) in f.docs.iter().enumerate().skip(f.n_relevant) {
        assert!(d.is_distractor, "doc {i} should be flagged a distractor");
        assert!(
            !all_relevant.contains(&i),
            "distractor {i} is relevant to a query"
        );
    }
}

#[test]
fn generation_is_deterministic() {
    let a = build_limit(&LimitConfig::paper_small(31));
    let b = build_limit(&LimitConfig::paper_small(31));
    assert_eq!(a.docs, b.docs);
    assert_eq!(a.queries, b.queries);
    // A different seed must actually change the fixture.
    let c = build_limit(&LimitConfig::paper_small(32));
    assert_ne!(a.queries, c.queries, "seed must affect the fixture");
}

#[test]
fn recall_at_k_semantics() {
    // Both relevant docs in the top 2 → 1.0.
    assert_eq!(recall_at_k(&[3, 7, 1], &[3, 7], 2), 1.0);
    // One of two → 0.5.
    assert_eq!(recall_at_k(&[3, 9, 7], &[3, 7], 2), 0.5);
    // Present but below the cut → 0.5 at k=2, 1.0 at k=3.
    assert_eq!(recall_at_k(&[3, 9, 7], &[3, 7], 3), 1.0);
    // Neither → 0.0.
    assert_eq!(recall_at_k(&[1, 2], &[3, 7], 2), 0.0);
    // k beyond the ranking length must not panic.
    assert_eq!(recall_at_k(&[3], &[3, 7], 10), 0.5);
}

#[test]
fn modelless_embedder_is_deterministic_unit_norm_and_discriminative() {
    let a = modelless_embed_8("who likes brand0");
    let b = modelless_embed_8("who likes brand0");
    assert_eq!(a, b, "must be deterministic");

    let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "must be L2-normalised, got {norm}"
    );

    let c = modelless_embed_8("who likes clendz1");
    assert_ne!(a, c, "different text must embed differently");
    // BLAKE3 decorrelates, so unrelated strings should not be near-duplicates.
    assert!(
        cosine(&a, &c).abs() < 0.999,
        "distinct strings collapsed together"
    );
}

#[test]
fn combinations_is_correct() {
    assert_eq!(
        combinations(4, 2, usize::MAX),
        vec![
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 3]
        ]
    );
    assert_eq!(combinations(5, 1, usize::MAX).len(), 5);
    assert_eq!(combinations(5, 5, usize::MAX), vec![vec![0, 1, 2, 3, 4]]);
    assert!(combinations(3, 4, usize::MAX).is_empty(), "k > n → none");
    assert_eq!(combinations(46, 2, 10).len(), 10, "limit must cap");
}

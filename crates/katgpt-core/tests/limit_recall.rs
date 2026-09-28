//! Issue 580 T2/T3 (public legs) — recall@k on the LIMIT fixture.
//!
//! Measures the two legs that live in the public engine:
//!
//! - **dense 8-D cosine** over `modelless_embed_8` — the modelless single-vector
//!   regime that `ShardIndex`, `ItemEmbedIndex` and `riir-rag` all operate in.
//! - **lexical token overlap** — an idealised stand-in for the `Bm25Index` leg,
//!   used here only to establish the plain-vs-synonym asymmetry the paper reports.
//!   The real BM25 measurement belongs in riir-neuron-db, where `Bm25Index` ships.
//!
//! Reported **per leg and per k, never averaged across legs** (Issue 580 T2).
//!
//! Run with output:
//! ```text
//! cargo test -p katgpt-core --features limit_fixture --test limit_recall -- --nocapture
//! ```

#![cfg(feature = "limit_fixture")]

use katgpt_core::limit_fixture::{
    LimitConfig, LimitFixture, cosine, modelless_embed_8, recall_at_k,
};

/// Mean recall@k over all queries for a ranking function.
fn mean_recall<F>(fixture: &LimitFixture, k: usize, mut rank: F) -> f32
where
    F: FnMut(usize) -> Vec<usize>,
{
    let mut acc = 0.0f32;
    for (qi, q) in fixture.queries.iter().enumerate() {
        acc += recall_at_k(&rank(qi), &q.relevant, k);
    }
    acc / fixture.queries.len() as f32
}

/// Rank all documents by cosine against the query, best first.
fn dense_ranking(fixture: &LimitFixture) -> impl Fn(usize) -> Vec<usize> + '_ {
    let doc_emb: Vec<[f32; 8]> = fixture
        .docs
        .iter()
        .map(|d| modelless_embed_8(&d.text))
        .collect();
    let q_emb: Vec<[f32; 8]> = fixture
        .queries
        .iter()
        .map(|q| modelless_embed_8(&q.text))
        .collect();
    move |qi: usize| {
        let mut scored: Vec<(usize, f32)> = doc_emb
            .iter()
            .enumerate()
            .map(|(i, d)| (i, cosine(&q_emb[qi], d)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.into_iter().map(|(i, _)| i).collect()
    }
}

/// Rank by count of shared whitespace tokens between query and document text.
/// Ties broken by document index, so no positional luck.
fn lexical_ranking(fixture: &LimitFixture) -> impl Fn(usize) -> Vec<usize> + '_ {
    move |qi: usize| {
        let q_tokens: Vec<&str> = fixture.queries[qi].text.split_whitespace().collect();
        let mut scored: Vec<(usize, i32)> = fixture
            .docs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let hits = d
                    .text
                    .split_whitespace()
                    .filter(|t| q_tokens.contains(t))
                    .count() as i32;
                (i, hits)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.into_iter().map(|(i, _)| i).collect()
    }
}

/// Expected recall@k for a uniformly random ranking — the chance floor a leg
/// must beat to be doing anything at all.
fn chance_recall(n_docs: usize, k: usize) -> f32 {
    (k as f32 / n_docs as f32).min(1.0)
}

#[test]
fn limit_small_recall_by_leg() {
    let plain = katgpt_core::limit_fixture::build_limit(&LimitConfig::paper_small(2026));
    let syn =
        katgpt_core::limit_fixture::build_limit(&LimitConfig::paper_small(2026).with_synonyms());

    println!(
        "\n=== Issue 580 T2/T3 — LIMIT-small recall by leg ===\n\
         46 relevant docs, {} queries, k=2. Chance floor at k=2 is {:.3}.\n",
        plain.queries.len(),
        chance_recall(plain.docs.len(), 2)
    );
    println!(
        "{:<22} {:>9} {:>9} {:>9}",
        "leg / variant", "R@2", "R@10", "R@20"
    );
    println!("{}", "-".repeat(54));

    let cap_iter = [("plain", &plain), ("synonym", &syn)];
    let mut results = Vec::with_capacity(cap_iter.len());
    for (label, fx) in cap_iter {
        let dense = dense_ranking(fx);
        let d2 = mean_recall(fx, 2, &dense);
        let d10 = mean_recall(fx, 10, &dense);
        let d20 = mean_recall(fx, 20, dense);
        println!(
            "{:<22} {d2:>9.3} {d10:>9.3} {d20:>9.3}",
            format!("dense 8-D / {label}")
        );

        let lex = lexical_ranking(fx);
        let l2 = mean_recall(fx, 2, &lex);
        let l10 = mean_recall(fx, 10, &lex);
        let l20 = mean_recall(fx, 20, lex);
        println!(
            "{:<22} {l2:>9.3} {l10:>9.3} {l20:>9.3}",
            format!("lexical / {label}")
        );
        results.push((label, d2, d10, d20, l2, l10, l20));
    }

    let (_, pd2, pd10, pd20, pl2, _, _) = results[0];
    let (_, sd2, _, _, sl2, _, _) = results[1];

    println!(
        "\nDense 8-D vs chance at k=2: {pd2:.3} vs {:.3}.\n\
         Lexical asymmetry (paper: BM25 −89%, dense −38.9%):\n  \
         lexical  {pl2:.3} → {sl2:.3}\n  dense    {pd2:.3} → {sd2:.3}\n",
        chance_recall(plain.docs.len(), 2)
    );

    // ── Assertions ──

    // 1. The lexical leg must near-solve the plain variant. LIMIT is
    //    linguistically trivial by design — a leg that matches the attribute
    //    token can simply read off the answer (paper: BM25 97.8 recall@2).
    assert!(
        pl2 > 0.9,
        "lexical leg should near-solve plain LIMIT, got R@2 = {pl2:.3}"
    );

    // 2. …and must collapse on the synonym variant, since the shared token is
    //    gone. This is the paper's central caveat: lexical is not a panacea.
    assert!(
        sl2 < 0.2,
        "lexical leg should collapse on synonyms, got R@2 = {sl2:.3}"
    );
    assert!(
        sl2 < pl2 * 0.3,
        "expected a >70% lexical drop, got {pl2:.3} → {sl2:.3}"
    );

    // 3. The modelless dense leg is structural, not semantic — it has no
    //    mechanism for noticing a shared attribute token, so it should sit at
    //    the chance floor on BOTH variants. Asserted as a band rather than an
    //    exact value so the test is not brittle.
    let floor = chance_recall(plain.docs.len(), 2);
    for (label, r2) in [("plain", pd2), ("synonym", sd2)] {
        assert!(
            r2 < floor * 3.0,
            "dense 8-D on {label} scored {r2:.3}, unexpectedly far above the \
             chance floor {floor:.3} — the modelless embedder has no semantic \
             channel, so this would need explaining"
        );
    }

    // 4. Recall must be monotone in k for a fixed ranking (sanity on the metric).
    assert!(pd2 <= pd10 && pd10 <= pd20, "recall must be monotone in k");
}

/// Distractor mass must not help. Adding documents relevant to nothing can only
/// dilute a ranking, so recall must not improve.
#[test]
fn distractors_do_not_improve_recall() {
    // Trimmed from the paper's ~50k to keep the test fast; the property is the same.
    let cfg = LimitConfig {
        n_distractors: 2_000,
        n_queries: 200,
        ..LimitConfig::paper_small(99)
    };
    let full = katgpt_core::limit_fixture::build_limit(&cfg);

    let small_cfg = LimitConfig {
        n_queries: 200,
        ..LimitConfig::paper_small(99)
    };
    let small_200 = katgpt_core::limit_fixture::build_limit(&small_cfg);
    assert_eq!(small_200.queries.len(), full.queries.len());

    let lex_small = lexical_ranking(&small_200);
    let lex_full = lexical_ranking(&full);
    let s2 = mean_recall(&small_200, 2, lex_small);
    let f2 = mean_recall(&full, 2, lex_full);

    println!(
        "lexical R@2: {s2:.3} at 46 docs → {f2:.3} at {} docs\n",
        full.docs.len()
    );
    assert!(
        f2 <= s2 + 1e-4,
        "adding {} distractors improved recall ({s2:.3} → {f2:.3}), which is impossible",
        full.docs.len() - small_200.docs.len()
    );
}

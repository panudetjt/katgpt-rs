//! Issue 867 T1 corpus gate — the AST histogram extractor against the
//! riir-train Issue 567 whole-item corpus (plus inline idiom pairs so the
//! gate is non-vacuous in a bare clone where the private sibling is absent).
//!
//! The sibling read is LOUD-SKIP: katgpt-rs is the public upstream and must
//! stay green without the private checkout — a skip prints a named reason
//! line, never a silent pass. Inline assertions always run.

use katgpt_canon::source_features::{AstBin, AstHistogram, N_AST_BINS, ast_histogram};
use std::path::PathBuf;

fn bin_of(h: &AstHistogram, bin: AstBin) -> u32 {
    h.counts[bin as usize]
}

/// The sibling corpus, if this box has the riir-train checkout beside it.
fn sibling_corpus_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../riir-train/data/canon_rust_contrastive/pairs.jsonl");
    if p.is_file() { Some(p) } else { None }
}

#[test]
fn inline_idiom_pairs_separate() {
    // Iterator chain vs index loop — the `iter` family's contrast axis.
    let chain = ast_histogram(
        "fn total(prices: &[f64]) -> f64 {\n    prices.iter().map(|p| p * 1.2).sum()\n}",
    )
    .expect("chain parses");
    let looped = ast_histogram(
        "fn total(prices: &[f64]) -> f64 {\n    let mut t = 0.0;\n    for i in 0..prices.len() {\n        t += prices[i] * 1.2;\n    }\n    t\n}",
    )
    .expect("loop parses");
    assert!(bin_of(&chain, AstBin::MethodCall) > bin_of(&looped, AstBin::MethodCall));
    assert!(bin_of(&looped, AstBin::ForLoop) >= 1 && bin_of(&chain, AstBin::ForLoop) == 0);
    assert!(bin_of(&looped, AstBin::Range) >= 1 && bin_of(&chain, AstBin::Range) == 0);

    // `?` propagation vs sentinel returns — the `error` family's axis.
    let question = ast_histogram(
        "fn f(raw: &str) -> Result<u32, E> {\n    let n: u32 = raw.parse()?;\n    Ok(n)\n}",
    )
    .expect("question parses");
    let sentinel = ast_histogram(
        "fn f(raw: &str) -> u32 {\n    match raw.parse::<u32>() {\n        Ok(n) => n,\n        Err(_) => 0,\n    }\n}",
    )
    .expect("sentinel parses");
    assert_eq!(bin_of(&question, AstBin::Try), 1);
    assert_eq!(bin_of(&sentinel, AstBin::Try), 0);
    assert!(bin_of(&sentinel, AstBin::MatchExpr) >= 1 && bin_of(&question, AstBin::MatchExpr) == 0);

    // Trait dispatch vs tag match — the `traits` family's axis.
    let tag_dispatch = ast_histogram(
        "fn area(kind: u8, r: f64) -> f64 {\n    if kind == 1 {\n        3.14 * r * r\n    } else {\n        r\n    }\n}",
    )
    .expect("tag parses");
    let traitful = ast_histogram(
        "fn show(shapes: &[Box<dyn S>]) -> f64 {\n    shapes.iter().map(|s| s.area()).sum()\n}",
    )
    .expect("trait parses");
    assert!(
        bin_of(&traitful, AstBin::Trait) == 0,
        "no trait def in the consumer fn"
    );
    assert!(bin_of(&traitful, AstBin::MethodCall) > bin_of(&tag_dispatch, AstBin::MethodCall));
    assert!(
        bin_of(&tag_dispatch, AstBin::If) >= 1,
        "if/else is one ExprIf node"
    );
}

#[test]
fn determinism_bit_identical() {
    let src = "trait S { fn area(&self) -> f64; }\nimpl S for C { fn area(&self) -> f64 { 3.0 * self.r } }\n";
    let a = ast_histogram(src).expect("parses");
    let b = ast_histogram(src).expect("parses");
    assert_eq!(a.counts, b.counts);
}

#[test]
fn parse_failure_is_none_never_panic() {
    assert!(ast_histogram("fn {").is_none());
    assert!(ast_histogram("struct struct struct").is_none());
}

#[test]
fn sibling_corpus_histograms_parse_and_carry_signal() {
    let Some(path) = sibling_corpus_path() else {
        // Loud skip — the public repo must stay green in a bare clone, but
        // the skip is NAMED so a reader never mistakes it for a pass over
        // data.
        eprintln!(
            "SKIP sibling-corpus arms: riir-train checkout not present at {} (bare-clone posture)",
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../riir-train")
                .display()
        );
        return;
    };

    let raw = std::fs::read_to_string(&path).expect("sibling corpus readable");
    let mut n_pairs = 0usize;
    // Aggregate (summed over the corpus) arm separations along the
    // families' contrast axes. Per-pair direction is NOT guaranteed — the
    // contrast is statistical — so the assertions are on the sums, with the
    // concrete totals printed for the record.
    let mut idio_loop = 0u64;
    let mut non_loop = 0u64;
    let mut idio_index = 0u64;
    let mut non_index = 0u64;
    let mut idio_try = 0u64;
    let mut non_try = 0u64;
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).expect("corpus line is valid JSON");
        let idiomatic = v["idiomatic_item"].as_str().expect("idiomatic_item field");
        let non_idiomatic = v["non_idiomatic_item"]
            .as_str()
            .expect("non_idiomatic_item field");
        let h_a = ast_histogram(idiomatic).expect("corpus items are syn-validated (issue 567 T1)");
        let h_b =
            ast_histogram(non_idiomatic).expect("corpus items are syn-validated (issue 567 T1)");
        assert!(
            h_a.total() > 0 && h_b.total() > 0,
            "corpus item histograms are non-empty"
        );
        idio_loop += (bin_of(&h_a, AstBin::ForLoop)
            + bin_of(&h_a, AstBin::While)
            + bin_of(&h_a, AstBin::WhileLet)) as u64;
        non_loop += (bin_of(&h_b, AstBin::ForLoop)
            + bin_of(&h_b, AstBin::While)
            + bin_of(&h_b, AstBin::WhileLet)) as u64;
        idio_index += bin_of(&h_a, AstBin::Index) as u64;
        non_index += bin_of(&h_b, AstBin::Index) as u64;
        idio_try += bin_of(&h_a, AstBin::Try) as u64;
        non_try += bin_of(&h_b, AstBin::Try) as u64;
        n_pairs += 1;
    }
    assert!(
        n_pairs >= 100,
        "corpus scale target: ≥100 pairs, got {n_pairs}"
    );
    eprintln!(
        "sibling corpus: {n_pairs} pairs — loops idio {idio_loop} vs non {non_loop}; \
         index idio {idio_index} vs non {non_index}; try idio {idio_try} vs non {non_try}"
    );
    assert!(
        non_loop > idio_loop,
        "anti-idiomatic arm must be loop-heavier in aggregate ({non_loop} vs {idio_loop})"
    );
    assert!(
        non_index > idio_index,
        "anti-idiomatic arm must be index-heavier in aggregate ({non_index} vs {idio_index})"
    );
    assert!(
        idio_try > non_try,
        "idiomatic arm must be `?`-heavier in aggregate ({idio_try} vs {non_try})"
    );
}

#[test]
fn vocabulary_frozen() {
    // Bin indices ARE the feature-vector contract — this arm trips on any
    // insert/reorder. New bins must append after Unsafe.
    assert_eq!(AstBin::Fn as u8, 0);
    assert_eq!(AstBin::Unsafe as u8, (N_AST_BINS - 1) as u8);
}

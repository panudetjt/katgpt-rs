//! Probe: verify that whitespace-based pretokenization produces bit-identical
//! output to whole-text `BpeTokenizerImpl::encode`, for tokenizers trained by
//! `BpeTrainer::train`.
//!
//! # Why this should hold
//!
//! `BpeTrainer::train` (see `src/bpe.rs` lines 319-323) learns merges via:
//! ```ignore
//! let words: Vec<Vec<String>> = corpus
//!     .split_whitespace()
//!     .map(|w| w.chars().map(|c| c.to_string()).collect())
//!     .collect();
//! ```
//! Merges are learned WITHIN whitespace-delimited words only. Therefore:
//! 1. No learned merge rule ever has a whitespace character in `left` or `right`.
//! 2. No learned merge rule ever crosses a whitespace boundary.
//! 3. In `encode`, whitespace chars can never be part of any merge — they
//!    pass through as inert single-char tokens.
//!
//! Consequence: splitting the input on whitespace, encoding each non-whitespace
//! chunk independently, and interleaving the whitespace chars back in should
//! produce the exact same token sequence as whole-text encode.
//!
//! # Why this matters
//!
//! If this hypothesis holds, it unlocks the vendored `ShortPretokenCache`
//! substrate (Issue 191 Phase 1 §"What's NOT here") without changing BPE
//! semantics — the cache hit rate on natural language is high (words repeat),
//! and the per-pretoken BPE is bit-identical to the whole-text path. This
//! would fix deferral reasons #1 (substrate not wired) and #3 (1000× claim
//! not honest) from `.benchmarks/191_fast_bpe_goat.md`.
//!
//! This test is the G1 correctness floor for the pretokenization direction.
//! If it FAILS, the hypothesis is wrong and pretokenization cannot be added
//! to `fast_bpe` without changing `BpeTokenizer`'s semantics — the work
//! would belong in a separate `PretokenizedBpeTokenizer` instead.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-tokenizer --features fast_bpe \
//!     --test fast_bpe_pretok_hypothesis --release -- --nocapture
//! ```

#![cfg(feature = "fast_bpe")]

use katgpt_tokenizer::{BpeTokenizer, BpeTokenizerImpl, BpeTrainer};

/// Encode `text` by splitting on whitespace (matching `BpeTrainer::train`'s
/// `split_whitespace()`), encoding each non-whitespace run independently,
/// and reassembling the token sequence with the whitespace chars re-injected
/// at their original positions.
///
/// This mirrors what a pretokenized `FastBpeEncoder` would do, minus the cache.
fn encode_with_whitespace_pretokenization(tokenizer: &BpeTokenizer, text: &str) -> Vec<usize> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(text.chars().count());
    // `split_whitespace` collapses runs and ignores leading/trailing — we
    // need the EXACT char positions, so iterate manually and classify each
    // char as whitespace or not. A whitespace char is inert (never in any
    // merge rule by the trainer's construction); a non-whitespace run is a
    // pretoken that can be encoded independently.
    let mut current_run = String::new();
    for c in text.chars() {
        if c.is_whitespace() {
            // Flush the current non-ws run (if any).
            if !current_run.is_empty() {
                let ids = BpeTokenizerImpl::encode(tokenizer, &current_run);
                out.extend(ids);
                current_run.clear();
            }
            // Emit the whitespace char as its own token (same as `encode`
            // would — it can never merge with anything).
            let ids = BpeTokenizerImpl::encode(tokenizer, &c.to_string());
            out.extend(ids);
        } else {
            current_run.push(c);
        }
    }
    if !current_run.is_empty() {
        let ids = BpeTokenizerImpl::encode(tokenizer, &current_run);
        out.extend(ids);
    }
    out
}

#[test]
fn g1_whitespace_pretokenization_bit_identical_synthetic() {
    let corpus = "the cat sat on the mat the cat the mat the test hello world the test split \
                  the cat the mat the test hello world the cat the mat the test";
    let tokenizer = BpeTrainer::train(corpus, 64);
    let texts = [
        "hello",
        "the cat",
        "the cat sat on the mat",
        "world hello test",
        "xyzzy",
        "",
        "the the the the the the the the the the",
        "ca",
        "  leading spaces",
        "trailing spaces  ",
        "multiple   internal   spaces",
        "\ttab\tseparated\twords",
        "newline\nseparated\nlines",
        "mixed\t whitespace\n with various  separators",
    ];
    for text in &texts {
        let whole = BpeTokenizerImpl::encode(&tokenizer, text);
        let pretok = encode_with_whitespace_pretokenization(&tokenizer, text);
        assert_eq!(
            whole, pretok,
            "Hypothesis FAILED on synthetic corpus, text={text:?}\n  whole={whole:?}\n  pretok={pretok:?}"
        );
    }
}

#[test]
fn g1_whitespace_pretokenization_bit_identical_medium_vocab() {
    // Smaller vocab than the original probe to keep the test fast — the
    // hypothesis is independent of vocab size (it's a property of the
    // trainer's `split_whitespace()` boundary, not of how many merges fire).
    let corpus = "fn foo() -> usize { 42 } use std collections HashMap;";
    let tokenizer = BpeTrainer::train(corpus, 128);
    let texts = [
        "fn foo() -> usize { 42 }",
        "use std::collections::HashMap;",
        "the quick brown fox jumps over the lazy dog",
        "struct PairRankTable { dense: Box<[u32]> }",
        "  // comment with leading spaces\n    fn body() {\n        let x = 1;\n    }\n",
        corpus, // whole corpus — the real test
    ];
    for text in &texts {
        let whole = BpeTokenizerImpl::encode(&tokenizer, text);
        let pretok = encode_with_whitespace_pretokenization(&tokenizer, text);
        let first_diff = whole.iter().zip(pretok.iter()).position(|(a, b)| a != b);
        assert_eq!(
            whole,
            pretok,
            "Hypothesis FAILED on medium vocab (len={}): first divergence at {:?}\n  whole[{}..{}]={:?}\n  pretok[{}..{}]={:?}",
            text.len(),
            first_diff,
            first_diff.unwrap_or(0).saturating_sub(3),
            first_diff.unwrap_or(0) + 3,
            &whole[first_diff.unwrap_or(0).saturating_sub(3)
                ..(first_diff.unwrap_or(0) + 3).min(whole.len())],
            first_diff.unwrap_or(0).saturating_sub(3),
            first_diff.unwrap_or(0) + 3,
            &pretok[first_diff.unwrap_or(0).saturating_sub(3)
                ..(first_diff.unwrap_or(0) + 3).min(pretok.len())],
        );
    }
}

#[test]
fn g1_whitespace_pretokenization_bit_identical_corpus_repeated() {
    // Smaller corpus repeat — the full bpe.rs × 4 is too slow to train on
    // for a probe. The hypothesis is structural; it doesn't need scale.
    let corpus = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                  alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
    let tokenizer = BpeTrainer::train(corpus, 128);
    let whole = BpeTokenizerImpl::encode(&tokenizer, corpus);
    let pretok = encode_with_whitespace_pretokenization(&tokenizer, corpus);
    assert_eq!(
        whole,
        pretok,
        "Hypothesis FAILED on corpus_repeat ({} chars): sequences differ in length (whole={}, pretok={})",
        corpus.len(),
        whole.len(),
        pretok.len()
    );
}

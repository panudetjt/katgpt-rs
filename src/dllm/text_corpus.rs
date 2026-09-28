//! Plan 601: the real-text corpus substrate for the dllm lane.
//!
//! Plan 600's promotion bar ("T8 all-green + T9 **on real text**") could not be
//! met because this repo's dllm lane was pattern-only: `generate_pattern_dataset`
//! draws an analytic [a, b, a, b] family, and every gate ran against the law
//! that family was generated from. This module is the missing half — a
//! committed, public-domain English text fixture plus the smallest policy-free
//! vocabulary around it:
//!
//! - [`TEXT_CORPUS`] — the fixture (Jane Austen, *Pride and Prejudice* tail,
//!   Project Gutenberg eBook #1342, public domain in the US).
//! - [`encode_text`] — char-level tokenizer onto a 31-symbol compact alphabet
//!   (lowercase a–z + space + period + comma + apostrophe; every other real
//!   character — digits, quotes, dashes, newlines, illustration captions —
//!   folds to space). Lowercasing is a documented tokenizer decision, the same
//!   normalization class as every real tokenizer; it doubles the data density
//!   a micro model sees.
//! - [`slice_blocks`] — disjoint fixed-length windows of a token stream
//!   (the block shape the anchor-then-fill harness consumes).
//! - [`bigram_law_smoothed`] / [`unigram_entropy_nats`] — nonparametric
//!   estimates of the corpus law from held-out token counts, the real-text
//!   counterpart of the pattern lane's analytic `corpus_law()` (the T9 KL
//!   reference).
//!
//! Policy lives with the measurement: the train/eval window offsets, epoch
//! count, and gate thresholds are pinned in
//! `tests/bench_601_flashar_realtext_goat.rs`, not here.

/// The committed real-text fixture: novel body only, ~107k chars, aligned to a
/// chapter boundary (`CHAPTER LIII.` — the final chapters of the novel).
/// Provenance + trim documented in the file header itself.
pub const TEXT_CORPUS: &str = include_str!("../../tests/data/austen_pride_prejudice_tail.txt");

/// Number of TEXT token ids (mask lives at `vocab_size - 1` = 30 and is never
/// emitted by [`encode_text`]). Alphabet: a–z (0–25), space (26), period (27),
/// comma (28), apostrophe (29), padding slot (30, unused — keeps the ids
/// stable if a symbol is ever added).
pub const TEXT_ALPHABET: usize = 31;

/// Map one char of real text to its token id (the compact alphabet). Private
/// single point of truth for the mapping documented on [`TEXT_ALPHABET`].
fn encode_char(c: char) -> usize {
    match c {
        'a'..='z' => (c as u8 - b'a') as usize,
        'A'..='Z' => (c as u8 - b'A') as usize,
        ' ' => 26,
        '.' => 27,
        ',' => 28,
        // Straight + typographic apostrophes (the fixture uses U+2019 —
        // "don’t" is everywhere in Austen; folding it to space would shred
        // the most frequent bigram-bearing token).
        '\'' | '\u{2019}' => 29,
        // Digits, quotes, dashes, newlines, brackets, illustration captions:
        // every other real character folds to space.
        _ => 26,
    }
}

/// Tokenize real text onto the compact alphabet. Char-level (not byte-level)
/// so the fixture's UTF-8 typographic characters map whole — see
/// [`encode_char`].
pub fn encode_text(text: &str) -> Vec<usize> {
    text.chars().map(encode_char).collect()
}

/// `n_blocks` disjoint windows of `block_len` tokens starting at token offset
/// `start`. Panics if the stream is too short (a silent short eval set would
/// shrink the paired-Δ resolution the G1 gate asserts).
pub fn slice_blocks(
    tokens: &[usize],
    start: usize,
    n_blocks: usize,
    block_len: usize,
) -> Vec<Vec<usize>> {
    assert!(
        start + n_blocks * block_len <= tokens.len(),
        "token stream too short: need {} tokens from offset {}, have {}",
        n_blocks * block_len,
        start,
        tokens.len()
    );
    (0..n_blocks)
        .map(|b| tokens[start + b * block_len..start + (b + 1) * block_len].to_vec())
        .collect()
}

/// Bigram transition counts over [`TEXT_ALPHABET`]: `counts[c1 * alphabet + c2]`.
pub fn bigram_counts(tokens: &[usize], alphabet: usize) -> Vec<u64> {
    assert!(tokens.len() >= 2, "bigram counts need >= 2 tokens");
    let mut counts = vec![0u64; alphabet * alphabet];
    for w in tokens.windows(2) {
        let (c1, c2) = (w[0], w[1]);
        assert!(c1 < alphabet && c2 < alphabet, "token id out of alphabet");
        counts[c1 * alphabet + c2] += 1;
    }
    counts
}

/// Row-normalized bigram law P(c2 | c1) with add-0.5 smoothing (the same
/// smoothing the pattern lane's MC-KL uses, so the two T9 estimates are
/// method-identical). `counts` from [`bigram_counts`].
pub fn bigram_law_smoothed(counts: &[u64], alphabet: usize) -> Vec<f64> {
    assert_eq!(counts.len(), alphabet * alphabet, "counts shape mismatch");
    let mut law = vec![0.0f64; alphabet * alphabet];
    for row in 0..alphabet {
        let row_slice = &counts[row * alphabet..(row + 1) * alphabet];
        let total: u64 = row_slice.iter().sum();
        let denom = total as f64 + 0.5 * alphabet as f64;
        for (col, &c) in row_slice.iter().enumerate() {
            law[row * alphabet + col] = (c as f64 + 0.5) / denom;
        }
    }
    law
}

/// Empirical unigram entropy of a token stream, in nats/token — the reference
/// baseline the trained model's masked NLL must beat (a model at unigram
/// entropy has learned the marginal and nothing else).
pub fn unigram_entropy_nats(tokens: &[usize], alphabet: usize) -> f64 {
    let mut counts = vec![0u64; alphabet];
    for &t in tokens {
        assert!(t < alphabet, "token id out of alphabet");
        counts[t] += 1;
    }
    let n = tokens.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_encodes_into_the_alphabet() {
        let tokens = encode_text(TEXT_CORPUS);
        assert!(
            tokens.len() > 90_000,
            "fixture unexpectedly small: {}",
            tokens.len()
        );
        assert!(tokens.iter().all(|&t| t < TEXT_ALPHABET));
        // The fixture file carries a provenance header above a `====` divider;
        // the NOVEL body must start at a chapter boundary right after it.
        let body = TEXT_CORPUS
            .split_once("================================================================================\n")
            .expect("fixture missing its provenance divider")
            .1
            .trim_start();
        assert!(
            body.starts_with("CHAPTER"),
            "fixture misaligned: {:?}",
            &body[..40.min(body.len())]
        );
    }

    #[test]
    fn encode_char_mapping_roundtrips_the_documented_alphabet() {
        assert_eq!(encode_char('a'), 0);
        assert_eq!(encode_char('z'), 25);
        assert_eq!(encode_char('Q'), 16); // lowercased
        assert_eq!(encode_char(' '), 26);
        assert_eq!(encode_char('\n'), 26);
        assert_eq!(encode_char('7'), 26);
        assert_eq!(encode_char('.'), 27);
        assert_eq!(encode_char(','), 28);
        assert_eq!(encode_char('\''), 29);
        assert_eq!(encode_char('\u{2019}'), 29);
        assert_eq!(encode_char('\u{2014}'), 26); // em dash
    }

    #[test]
    fn slice_blocks_is_disjoint_and_panics_short() {
        let toks: Vec<usize> = (0..100).collect();
        let blocks = slice_blocks(&toks, 10, 5, 8);
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[0][0], 10);
        assert_eq!(blocks[4][7], 49);
        let _ = std::panic::catch_unwind(|| slice_blocks(&toks, 90, 5, 8))
            .expect_err("short stream must panic");
    }

    #[test]
    fn bigram_law_rows_sum_to_one_and_entropy_is_sane() {
        let toks = encode_text("the the the. a cat, a hat.");
        let counts = bigram_counts(&toks, TEXT_ALPHABET);
        let law = bigram_law_smoothed(&counts, TEXT_ALPHABET);
        for row in 0..TEXT_ALPHABET {
            let s: f64 = law[row * TEXT_ALPHABET..(row + 1) * TEXT_ALPHABET]
                .iter()
                .sum();
            assert!((s - 1.0).abs() < 1e-9, "row {row} sums to {s}");
        }
        let h = unigram_entropy_nats(&toks, TEXT_ALPHABET);
        assert!(h > 0.5 && h < 4.0, "unigram entropy {h} out of sane range");
    }
}

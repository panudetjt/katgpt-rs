//! GOAT gate for the SIMD bitstream splitter wiring (Issue 872) — the
//! `encode_into_pretok` end-to-end identity over the Unicode surface the
//! splitter's scalar fallback must classify bit-identically to
//! `char::is_whitespace`.
//!
//! # Gates
//!
//! - **G1 (correctness)**: `encode_into_pretok` (SIMD scan) produces
//!   bit-identical token IDs to `BpeTokenizerImpl::encode` (the scalar
//!   whole-text reference) on ASCII/code text AND on the full Unicode
//!   `White_Space` set (every multibyte ws char), near-miss non-ws chars
//!   (U+180E, U+200B, U+FEFF, …), CJK/emoji, and long identical-byte ws
//!   runs (the `AsciiWsRun` amortization).
//! - **G2 (perf)**: lives in the unit tests (`src/fast_bpe/simd_split.rs`,
//!   `g2_scan_ab_simd_vs_scalar`) — scan-only A/B, interleaved rounds +
//!   median ratio, available to `pub(crate)` levels there.
//! - **G3 (no-regression)**: the pre-existing pretok GOAT
//!   (`fast_bpe_goat_pretok.rs`) + hypothesis guard
//!   (`fast_bpe_pretok_hypothesis.rs`) cover the plain-ASCII surface; this
//!   file adds the Unicode surface they never exercised.

#![cfg(feature = "fast_bpe")]

use katgpt_tokenizer::{BpeTokenizer, BpeTokenizerImpl, BpeTrainer, FastBpeEncoder};

/// Every Unicode `White_Space` char (25 total).
const ALL_WS_CHARS: [char; 25] = [
    '\u{0009}', '\u{000A}', '\u{000B}', '\u{000C}', '\u{000D}', '\u{0020}', '\u{0085}', '\u{00A0}',
    '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}',
    '\u{2007}', '\u{2008}', '\u{2009}', '\u{200A}', '\u{2028}', '\u{2029}', '\u{202F}', '\u{205F}',
    '\u{3000}',
];

/// Near-miss chars that are NOT `White_Space` — must stay inside words.
const NOT_WS_NEAR_MISSES: [char; 9] = [
    '\u{0008}', '\u{000E}', '\u{007F}', '\u{0084}', '\u{00AD}', '\u{180E}', '\u{200B}', '\u{200C}',
    '\u{FEFF}',
];

fn assert_identity(tokenizer: &BpeTokenizer, text: &str) {
    let reference = BpeTokenizerImpl::encode(tokenizer, text);
    let mut encoder = FastBpeEncoder::from_tokenizer(tokenizer);
    let mut out = Vec::new();
    encoder.encode_into_pretok(text, &mut out);
    assert_eq!(
        out,
        reference,
        "SIMD pretok divergence vs encode on {text:?} (len={}): first diff at {}",
        text.len(),
        out.iter()
            .zip(reference.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(out.len().min(reference.len()))
    );
}

#[test]
fn g1_simd_split_bit_identical_ascii_and_code() {
    let corpus = "the cat sat on the mat the cat the mat the test hello world the test split \
                  the cat the mat the test hello world the cat the mat the test";
    let tokenizer = BpeTrainer::train(corpus, 64);
    for text in [
        "hello",
        "",
        "the cat sat on the mat",
        "  leading spaces",
        "trailing spaces  ",
        "multiple   internal   spaces",
        "\ttab\tseparated\twords",
        "mixed\t whitespace\n with various  separators",
        "vertical\t\x0b\ttab is whitespace",
    ] {
        assert_identity(&tokenizer, text);
    }

    // Self-hosted larger vocab on real code.
    let code_corpus = include_str!("../src/bpe.rs");
    let code_tokenizer = BpeTrainer::train(code_corpus, 1024);
    for text in [
        "fn foo() -> usize { 42 }",
        "let x = foo.bar.baz(qux, quux);",
        code_corpus,
    ] {
        assert_identity(&code_tokenizer, text);
    }
}

#[test]
fn g1_simd_split_bit_identical_unicode_ws_set() {
    // Train on a corpus that CONTAINS the multibyte ws chars + CJK so the
    // vocab holds them as single-char tokens (the other path — unk fallback —
    // is also identity-preserving, but this exercises the vocab-hit arm).
    let mut corpus = String::from(
        "word separator tests for the tokenizer: the cat the mat the test hello world ",
    );
    for w in ALL_WS_CHARS {
        corpus.push_str(&format!("alpha{w}beta{w}gamma "));
    }
    corpus.push_str("中文 测试 分词 性能 hello 世界 naïve café résumé ");
    let tokenizer = BpeTrainer::train(&corpus, 2048);

    // Every ws char as separator, in every position shape.
    for w in ALL_WS_CHARS {
        for text in [
            format!("aa{w}bb"),
            format!("{w}"),
            format!("x{w}"),
            format!("{w}x"),
            format!("a{w}b{w}c"),
            format!("a{w}{w}b"), // two consecutive multibyte ws chars
            format!("日本{w}語{w}テスト"),
            format!("café{w}naïve{w}résumé"),
        ] {
            assert_identity(&tokenizer, &text);
        }
    }

    // Near-miss non-ws chars stay INSIDE words.
    for c in NOT_WS_NEAR_MISSES {
        for text in [
            format!("a{c}b"),
            format!("x {c} y{c}z"),
            format!("日本{c}語"),
        ] {
            assert_identity(&tokenizer, &text);
        }
    }

    // CJK/emoji words (no ASCII at all — the scalar multibyte path carries
    // the whole scan) + mixed with ASCII chunk boundaries.
    for text in [
        "中文 测试 分词器的 性能 很好".to_string(),
        "a😀b c🇺🇸d é f̃ g🄰h".to_string(),
        "中文\u{3000}全角空格 NBSP\u{00A0}here LSEP\u{2028}next PSEP\u{2029}end".to_string(),
        format!("{}中{}文{}", "a".repeat(15), "b".repeat(17), "c".repeat(31)),
    ] {
        assert_identity(&tokenizer, &text);
    }
}

#[test]
fn g1_simd_split_ws_run_amortization() {
    // Long identical-byte ws runs (the AsciiWsRun coalescing) and mixed ws
    // runs (never coalesced), at lengths that straddle the 16/32-byte chunks.
    let corpus = "run of spaces then tabs then words the cat the mat hello world test";
    let tokenizer = BpeTrainer::train(corpus, 128);
    for k in [1usize, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1000] {
        assert_identity(&tokenizer, &format!("a{}b", " ".repeat(k)));
        assert_identity(&tokenizer, &format!("{}\n", "\t".repeat(k)));
        assert_identity(&tokenizer, &format!("x{}y", " \t ".repeat(k / 3 + 1)));
        // Vertical-tab run — the byte is_ascii_whitespace misses.
        assert_identity(&tokenizer, &format!("a{}b", "\x0b".repeat(k)));
    }
}

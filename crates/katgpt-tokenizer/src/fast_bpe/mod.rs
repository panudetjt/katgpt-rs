//! `fast_bpe` — gigatoken-derived fast BPE encode path (Issue 191, Research 456).
//!
//! Vendored from [gigatoken](https://github.com/marcelroed/gigatoken) (MIT,
//! Marcel Rød, ~2.5k★) per the Issue 191 Phase 0 verdict. Only the pure-Rust
//! `bpe/` core is vendored — the upstream repo also ships a nightly-Rust
//! `pretokenize/` module (uses `#![feature(portable_simd)]`) and unconditional
//! `pyo3`/`numpy`/`parquet`/`arrow` deps that are incompatible with this
//! crate's stable-Rust + leaf-clean constraints. See `.issues/191_*.md` §"Phase
//! 0 verdict" for the portability proof.
//!
//! # What's here
//!
//! Two of gigatoken's four substrate-independent techniques (per Research 456):
//!
//! 1. **`PairRankTable`** — two-level merge-rank lookup replacing the existing
//!    `HashMap<(usize, usize), usize>` probe on every merge step. Dense
//!    `byte × byte` grid covers every round-1 lookup with one shift-or index
//!    and one L1/L2 load; a flat open-addressed packed table covers the rest.
//! 2. **Branchless merge cores** — small (≤32 symbols) + short (≤15 symbols)
//!    scalar + aarch64-NEON + x86_64-AVX2/AVX512 reference variants of the
//!    BPE merge loop, all with stack-resident doubly-linked lists.
//!
//! # What's NOT here (deferred to a follow-up issue if pretokenization lands)
//!
//! 3. **Pretoken cache hierarchy** — `ShortPretokenCache` is shipped in
//!    `pretoken_cache.rs` AND WIRED (Phase 2.7, 2026-07-25) into
//!    `FastBpeEncoder::flush_pretoken` (the per-pretoken path of
//!    `encode_into_pretok`). The hot path uses `get_or_slot` + `insert_at`;
//!    the chunk-level `ProbeView` + `prefetch_l2` + `probe_pair` prefetch API
//!    remains unwired — it's the future SIMD-batched pretokenization pipeline
//!    that would stage hundreds of lookups ahead of demand. (Research 456 §2.2
//!    names the prefetch path as the genuinely novel technique; it stays
//!    vendored so the cross-cutting port to Engram's `ZipfianCacheHierarchy`
//!    is a one-import change when that lands.)
//!
//!    **History:** Phase 2.6 (2026-07-25) first wired whitespace
//!    pretokenization with a plain `HashMap<Vec<u8>, Vec<TokenId>>` stand-in
//!    (bit-identical correctness, 2.71× speedup on 381-char natural
//!    language). Phase 2.7 (2026-07-25) replaced the HashMap with the
//!    vendored `ShortPretokenCache` substrate after the corpus-scale
//!    benchmark showed the HashMap's hash overhead was the bottleneck
//!    (gain plateaued at 6.38× at 1M chars with 100% cache coverage).
//!    See `.benchmarks/191_fast_bpe_goat.md` §G5 (Phase 2.6) + §G6 (Phase 2.7).
//! 4. **SIMD pretokenization** — the *scan half* shipped (Issue 872,
//!    2026-09-22): `simd_split.rs` classifies whitespace/bitstream-style via
//!    SSE2/AVX2/NEON masks (runtime-probed per the `shipped_target_feature_gate`
//!    law) with a scalar per-char fallback for multibyte chars, wired into
//!    `FastBpeEncoder::encode_into_pretok`. What remains out of scope is the
//!    *regex-grammar* class (GPT-2/cl100k/… bitstream grammars à la HF
//!    tokenizers v1 "bitcannon") — those belong to tokenizer formats this
//!    crate does not load; the katgpt `BpeTokenizer`'s only pretokenizer is
//!    the whitespace class its own trainer invariant defines.
//!
//! # Attribution
//!
//! Every file in this module records the upstream MIT license header. The
//! techniques ship verbatim where possible (the merge cores are bit-identical
//! to upstream so the `short_merges_match_vec_merge_loop` differential test
//! in upstream applies); adaptation is limited to (a) the `PairRankTable`
//! adapter from katgpt's `usize` IDs to gigatoken's `TokenId(u32)` and
//! (b) removal of the `madvise_hugepage` Linux-only hint from the cache slot
//! allocation path (the hint is a no-op off Linux anyway, and on Linux it
//! would force a `libc` dep this leaf-clean crate forbids).

mod pair_rank_table;
mod pretoken_cache;
mod pretokenize_keys;
mod simd_split;
mod token;

pub use pair_rank_table::{
    PairRankTable, PairRankTableBuildError, bpe_merge_symbols_by_rank,
    bpe_merge_symbols_by_rank_with_lookup,
};
pub use token::TokenId;

// SIMD bitstream whitespace splitter (Issue 872) — the scan half of
// `FastBpeEncoder::encode_into_pretok`.
pub(crate) use simd_split::{SplitEvent, WhitespaceSplitter};

// Re-export the merge cores + scratch so `encode_fast` can drive them directly.
#[allow(unused_imports)]
// SHORT_MERGE_MAX + short_scalar are substrate for future pretokenization work.
pub use pair_rank_table::{MergeScratch, SHORT_MERGE_MAX, bpe_merge_symbols_short_scalar};

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)] // substrate for future pretokenization work
pub use pair_rank_table::bpe_merge_symbols_short_neon;

// Pretoken-cache substrate (Issue 191 Phase 2.7 — wired into
// `FastBpeEncoder::flush_pretoken`). Re-exported `pub(crate)` so `bpe.rs`
// can build the two-tier (short inline + long spill) cache.
pub(crate) use pretoken_cache::ShortPretokenCache;
pub(crate) use pretokenize_keys::{pack_pretoken_key, pretoken_key_hash};

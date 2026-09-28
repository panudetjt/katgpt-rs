//! G4 alloc-free audit for `FastBpeEncoder::encode_into` (Issue 191 Phase 2.5).
//!
//! This file is **separate** from `tests/fast_bpe_goat.rs` because the
//! CountingAllocator audit needs to be the only test in the process — other
//! tests' allocations would otherwise be counted by the global allocator
//! counter during the audit window. This mirrors the
//! `variable_rank_domain_expert_alloc.rs` convention in `katgpt-core`.
//!
//! **Single alloc-audit test in this file.** Other correctness tests
//! live in `tests/fast_bpe_goat.rs` — they would otherwise run in
//! parallel with this audit and pollute the global allocator counter
//! (Rust's default test harness runs all `#[test]`s in a file in parallel).
//!
//! # Contract
//!
//! `FastBpeEncoder::encode_into` performs **zero heap allocations** in steady
//! state (after warmup). The `symbols` buffer + `MergeScratch` are reused
//! across calls; only reallocates when an input exceeds the prior peak. The
//! per-call `BpeTokenizerImpl::encode_fast` function + the
//! `FastBpeEncoder::encode` convenience wrapper are NOT alloc-free (they
//! rebuild / return a fresh `Vec<usize>`); the zero-alloc contract is on
//! `encode_into` only.
//!
//! # Run
//!
//! ```sh
//! cargo test -p katgpt-tokenizer --features fast_bpe \
//!   --test fast_bpe_goat_g4_alloc -- --nocapture
//! ```

#![cfg(feature = "fast_bpe")]

use katgpt_tokenizer::{BpeTokenizerImpl, BpeTrainer, FastBpeEncoder};

// ─── CountingAllocator (inlined; mirrors katgpt-core's macro pattern) ───────

struct CountingAllocator;

static ALLOC_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

/// Single combined G4 audit. Sequential inside one `#[test]` so the global
/// allocator counter is uncontended.
#[test]
fn g4_zero_alloc_audit_combined() {
    // ----- Phase A: small-vocab path (n <= SMALL_MERGE_MAX = 32) -----
    {
        let corpus = "the cat sat on the mat the cat the mat the test hello world";
        let tokenizer = BpeTrainer::train(corpus, 128);
        let texts = [
            "the cat sat on the mat",
            "hello world the test",
            "the cat the mat",
            "xyzzy the cat",
        ];

        let mut encoder = FastBpeEncoder::from_tokenizer(&tokenizer);
        let mut out: Vec<usize> = Vec::new();

        // Pre-compute references BEFORE warmup (the reference `encode` path
        // allocates per call — we don't want those allocs in the audit window).
        let references: Vec<Vec<usize>> = texts
            .iter()
            .map(|t| BpeTokenizerImpl::encode(&tokenizer, t))
            .collect();

        // Warmup with the longest input so scratch reaches peak capacity.
        encoder.encode_into(corpus, &mut out);
        eprintln!(
            "g4 small-path warmup: symbols capacity = {}",
            encoder.symbols_capacity()
        );

        ALLOC_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);

        let n = 100;
        for i in 0..n {
            let text = texts[i % texts.len()];
            encoder.encode_into(text, &mut out);
            assert_eq!(
                out,
                references[i % references.len()],
                "G4 small-path divergence on text={text:?}"
            );
        }

        let allocs = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("g4 small-path steady-state: {allocs} allocations across {n} encode_into calls");
        assert_eq!(
            allocs, 0,
            "G4 small-path alloc-free violation: encode_into allocated {allocs} times in steady state (expected 0)"
        );
    }

    // ----- Phase B: long-input path (n > SMALL_MERGE_MAX → BinaryHeap merge) -----
    {
        let corpus = include_str!("../src/bpe.rs");
        let tokenizer = BpeTrainer::train(corpus, 1024);

        // Long paragraph from bpe.rs (forces n > 32 char count → heap path).
        let text = "    /// Encode a string into token IDs using BPE merge rules.\n    ///\n    /// Hot-path design: operates on `Vec<usize>` (token IDs) end-to-end.";

        let mut encoder = FastBpeEncoder::from_tokenizer(&tokenizer);
        let mut out: Vec<usize> = Vec::new();

        let reference = BpeTokenizerImpl::encode(&tokenizer, text);

        // Warmup with the corpus itself (much longer than `text`) so scratch
        // reaches peak capacity.
        encoder.encode_into(corpus, &mut out);
        eprintln!(
            "g4 long-path warmup: symbols capacity = {}",
            encoder.symbols_capacity()
        );

        ALLOC_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);

        let n = 20;
        for _ in 0..n {
            encoder.encode_into(text, &mut out);
            assert_eq!(out, reference, "G4 long-path divergence");
        }

        let allocs = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("g4 long-path steady-state: {allocs} allocations across {n} encode_into calls");
        assert_eq!(
            allocs, 0,
            "G4 long-path alloc-free violation: {allocs} allocations in steady state (expected 0)"
        );
    }
}

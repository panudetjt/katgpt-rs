//! Issue 584 Phase 3 G4 — FlashMemory sparse MLA forward alloc-free steady state.
//!
//! GOAT gate G4: the per-token decode hot path (`mla_forward_token_flashmemory`)
//! must allocate ZERO bytes in steady state (after warm-up). This is the
//! production decode contract — a chat server decoding 1000s of tokens/call
//! cannot afford per-token heap growth (GC pressure, allocator contention,
//! cache pollution).
//!
//! # What this bench measures
//!
//! The `CountingAllocator` wraps `std::alloc::System` and atomically counts
//! every `alloc` call. We:
//!
//! 1. Build a small MLA config + random weights (G4 is about ALLOCATION, not
//!    correctness — random weights exercise the identical code path).
//! 2. **Warm up**: pre-fill the KV cache + block cache + selector to steady
//!    state (enough tokens that the selector has run at least one refresh,
//!    so `PerHeadSelection` inner Vec capacities are stable).
//! 3. **Measure**: decode N_STEADY tokens, counting allocations.
//! 4. Assert `allocs == 0`.
//!
//! # Allocation sites eliminated (Issue 584 G4 fix)
//!
//! Before the fix, two per-token allocations existed:
//!
//! 1. **`blocks_to_attend: Vec<usize>`** — `mla_forward_token_flashmemory` allocated
//!    a new `Vec` every token (either `vec![last_block]` fallback or
//!    `selected_blocks.clone()`). **Fixed**: stack array fallback + direct slice
//!    reference (zero allocation).
//! 2. **`PerHeadSelection::push`** — inner Vecs could reallocate on early
//!    refreshes. **Fixed**: `PerHeadSelection::new(n_heads, max_blocks)` pre-reserves
//!    `Vec::with_capacity(max_blocks)` per head — `push` never reallocates.
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench bench_022_flashmemory_alloc_free --features flashmemory_sparse
//! ```

#![cfg(feature = "flashmemory_sparse")]

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

use katgpt_attn::dash_attn::flashmemory_sparse::{
    FlashMemoryBlockCache, FlashMemoryConfig, FlashMemorySelector, mla_forward_token_flashmemory,
};
use katgpt_attn::mla::{MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights};
use katgpt_kv::shard_kv::rope::RopeFreqs;

// ── CountingAllocator (matches bench_013 pattern) ───────────────────────────

struct CountingAllocator;
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: layout is valid (caller's contract); System.alloc is sound.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr was allocated by System.alloc with this layout.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

// ── Config ──────────────────────────────────────────────────────────────────

/// Small MLA config matching the unit-test `small_mla_config` (fast, no model load).
fn small_mla_config() -> MlaConfig {
    MlaConfig {
        kv_lora_rank: 32,
        q_lora_rank: 64,
        qk_nope_head_dim: 16,
        qk_rope_head_dim: 8,
        v_head_dim: 16,
        n_heads: 4,
        hidden_size: 128,
        use_output_gate: true,
        use_nope: false,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    }
}

fn main() {
    println!("╔══ FlashMemory G4: alloc-free steady state (Issue 584 Phase 3) ══╗");
    println!();

    let config = small_mla_config();
    let weights = MlaWeights::random(&config, 42);

    // FlashMemory config: small blocks so we get enough blocks to exercise selection.
    // refresh_period < warmup so at least one refresh fires during warmup.
    let fm_config = FlashMemoryConfig {
        block_size: 16,
        refresh_period: 8,
        threshold: 0.5,
    };

    // Sequence length: long enough to have multiple blocks + multiple refresh cycles.
    const WARMUP_TOKENS: usize = 128; // 8 blocks of 16; 16 refresh periods at τ=8
    const STEADY_TOKENS: usize = 256; // measure these after warmup
    const TOTAL_TOKENS: usize = WARMUP_TOKENS + STEADY_TOKENS;
    let max_blocks = TOTAL_TOKENS.div_ceil(fm_config.block_size);

    let mut cache = MlaKVCache::new(&config, TOTAL_TOKENS);
    let mut scratch = MlaForwardScratch::new(&config, TOTAL_TOKENS);
    let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
    let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, TOTAL_TOKENS);
    let mut selector = FlashMemorySelector::new(fm_config.clone(), config.n_heads, max_blocks);

    // Hidden states: deterministic pseudo-random (seeded), NOT allocated per-token.
    // We pre-build the full sequence so the measurement loop is pure forward calls.
    let mut hidden_states = vec![vec![0.0f32; config.hidden_size]; TOTAL_TOKENS];
    let mut rng_state = 12345u64;
    for h in &mut hidden_states {
        for v in h.iter_mut() {
            // xorshift64* — fast, deterministic, no dep.
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            *v = ((rng_state % 1000) as f32) / 1000.0 - 0.5;
        }
    }

    println!(
        "Config: hidden={}, n_heads={}, kv_lora_rank={}",
        config.hidden_size, config.n_heads, config.kv_lora_rank
    );
    println!(
        "FlashMemory: block_size={}, refresh_period={}, threshold={}",
        fm_config.block_size, fm_config.refresh_period, fm_config.threshold
    );
    println!("Tokens: warmup={WARMUP_TOKENS}, steady={STEADY_TOKENS}, total={TOTAL_TOKENS}");
    println!();

    // ── Warmup: decode WARMUP_TOKENS to stabilize capacities ─────────
    // This primes: PerHeadSelection inner Vec capacities, selector scores_buf,
    // block_cache buffers, MLA cache. After this, the steady-state loop should
    // allocate nothing.
    for (step, h) in hidden_states.iter().enumerate().take(WARMUP_TOKENS) {
        let _ = mla_forward_token_flashmemory(
            &config,
            &weights,
            &mut cache,
            &mut scratch,
            &mut rope,
            h,
            &mut block_cache,
            &mut selector,
            step,
        );
    }

    let warmup_refreshes = selector.refresh_count();
    println!("Warmup complete: {WARMUP_TOKENS} tokens, {warmup_refreshes} selector refreshes");
    assert!(
        warmup_refreshes > 0,
        "warmup must trigger at least one selector refresh"
    );

    // ── Measure: decode STEADY_TOKENS, counting allocations ───────────
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for (i, h) in hidden_states
        .iter()
        .enumerate()
        .skip(WARMUP_TOKENS)
        .take(STEADY_TOKENS)
    {
        let step = WARMUP_TOKENS + i;
        let _ = mla_forward_token_flashmemory(
            &config,
            &weights,
            &mut cache,
            &mut scratch,
            &mut rope,
            h,
            &mut block_cache,
            &mut selector,
            step,
        );
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    let total_allocs = after - before;
    let per_token = total_allocs / STEADY_TOKENS;
    let steady_refreshes = selector.refresh_count() - warmup_refreshes;

    println!();
    println!("── G4 Result ──");
    println!("  Steady tokens decoded : {STEADY_TOKENS}");
    println!("  Total allocations     : {total_allocs}");
    println!("  Per-token allocations : {per_token}");
    println!("  Steady refreshes      : {steady_refreshes} (each refresh re-scores all blocks)");
    println!();

    let g4_pass = total_allocs == 0;
    println!(
        "  G4 verdict: {}",
        if g4_pass {
            "✅ PASS (0 allocations in steady state)"
        } else {
            "❌ FAIL"
        }
    );
    println!();

    if !g4_pass {
        eprintln!(
            "FAIL: FlashMemory sparse MLA forward allocated {total_allocs} bytes across {STEADY_TOKENS} steady tokens."
        );
        eprintln!("      Expected 0 (alloc-free steady state per GOAT G4).");
        eprintln!("      This means a per-token or per-refresh allocation was introduced.");
        std::process::exit(1);
    }

    println!("╚══ G4 PASS — FlashMemory sparse MLA forward is alloc-free in steady state ══╝");
}

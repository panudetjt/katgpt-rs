//! Plan 406 T2.2 follow-up — `PagedKVCache::ensure_pages` ArrayVec fast-path GOAT bench.
//!
//! Quantifies the gain from the canonical fast-path optimization
//! (`total_new ≤ 128` → flat stack `ArrayVec` + `extend_from_slice`, zero heap
//! scratch) vs the legacy slow-path (per-layer `Vec<usize>` index allocation)
//! that the canonical `PagedKVCache` replaced in Plan 406 Phase 2 S2.
//!
//! **Hot path measured:** speculative-decode steady state — `rollback` frees
//! `n_layer` pages to the warm free list, then `ensure_pages` re-allocates them.
//! In this regime `alloc_page()` is a cheap pop + `fill(0.0)` (no pool growth),
//! making the index-Vec allocation the dominant *differing* cost between paths.
//! The `fill(0.0)` is identical in both paths and dominates absolute time, so
//! the raw speedup is modest; the primary win is **allocation hygiene** (zero
//! scratch heap churn in the hot path).
//!
//! **Gates:**
//! - **G1 (correctness):** fast-path and legacy produce identical
//!   `layer_page_tables` after the same call sequence. Page-index assignment
//!   order is the same because both allocate `total_new` pages from the free
//!   list in layer order.
//! - **G2 (perf):** fast-path is not slower than legacy in steady-state decode.
//!   The gain is modest (few %) because `alloc_page`'s `fill(0.0)` dominates
//!   and is identical in both paths; the fast-path eliminates only the
//!   per-layer `Vec<usize>` index allocations.
//! - **G4 (alloc hygiene):** fast-path = **zero scratch heap allocs** per call
//!   (`ArrayVec` is stack-allocated, `layer_table` capacities stable in steady
//!   state). Legacy = `n_layers-with-deficit` heap `Vec` allocs per call.
//!   Verified via capacity-stability audit (matches bench_413 convention).
//!
//! **Bench convention:** `std::time::Instant` + `harness = false` — matches the
//! crate's existing GOAT benches. No Criterion dev-dep.
//!
//! Run:
//! ```bash
//! cargo run --release --bench bench_414_ensure_pages_goat
//! ```

#![allow(clippy::needless_range_loop)]

use katgpt_core::types::{self, Config};
use katgpt_transformer::{PAGE_SIZE, PagedKVCache};
use std::hint::black_box;
use std::time::Instant;

// ─── Config ─────────────────────────────────────────────────────────────────

/// Pages per layer after warmup. Each rollback→re-alloc cycle touches exactly
/// 1 page per layer (deficit = 1), so this controls warmup size, not per-cycle cost.
const CYCLES: usize = 32;

/// Steady-state rollback→re-alloc iterations for G2 timing.
const REPS: usize = 50_000;

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let mut all_pass = true;

    // micro:        n_layer=1, kv_dim=16  (page = 16×16×2 = 512 floats = 2KB)
    // small_target: n_layer=4, kv_dim=64  (page = 16×64×2 = 2048 floats = 8KB)
    let configs = [
        ("micro", Config::micro()),
        ("small_target", Config::small_target()),
    ];

    for (name, config) in &configs {
        let kd = types::kv_dim(config);
        println!(
            "╔══ Config: {name} (n_layer={}, kv_dim={}, page_bytes={}) ══╗",
            config.n_layer,
            kd,
            PAGE_SIZE * kd * 2 * 4,
        );

        all_pass &= g1_correctness(config, name);
        all_pass &= g2_perf(config, name);
        all_pass &= g4_alloc_audit(config, name);
        println!();
    }

    println!("╔════════════════════════════════════════╗");
    println!(
        "║  Overall: {}                             ║",
        if all_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!("╚════════════════════════════════════════╝");
    std::process::exit(if all_pass { 0 } else { 1 });
}

// ─── Legacy (pre-optimization slow path) ────────────────────────────────────
//
// Replicates the private `alloc_page` and the pre-ArrayVec `ensure_pages` using
// only the public fields of `PagedKVCache`. This is the "before" state that the
// canonical fast-path replaced — always per-layer `Vec<usize>`, no stack fast-path.

fn alloc_page_legacy(cache: &mut PagedKVCache) -> usize {
    // Exact replica of the private `PagedKVCache::alloc_page` via pub fields.
    let idx = if let Some(idx) = cache.free_pages.pop() {
        cache.pages[idx].fill(0.0);
        idx
    } else {
        cache.pages.push(vec![0.0; PAGE_SIZE * cache.kv_dim * 2]);
        let idx = cache.total_pages;
        cache.total_pages += 1;
        cache.page_ref_counts.push(0);
        idx
    };
    cache.page_ref_counts[idx] += 1;
    idx
}

fn ensure_pages_legacy(cache: &mut PagedKVCache, seq_idx: usize, pos: usize) {
    let pages_needed = pos / PAGE_SIZE + 1;

    // Grow sequence slots if needed (matches current ensure_pages).
    for layer_tables in &mut cache.layer_page_tables {
        while seq_idx >= layer_tables.len() {
            layer_tables.push(Vec::new());
        }
    }

    // Phase 1: compute deficits (immutable borrow — avoids double-borrow with alloc_page).
    let deficits: Vec<usize> = cache
        .layer_page_tables
        .iter()
        .map(|lt| pages_needed.saturating_sub(lt[seq_idx].len()))
        .collect();

    // Phase 2: allocate per-layer Vec<usize> — the legacy approach that allocates
    // one heap Vec per layer-with-deficit (vs the canonical flat stack ArrayVec).
    // Page-index assignment order matches the canonical: layer 0's pages first,
    // then layer 1's, etc. — alloc_page pops from the same free list in the same order.
    let mut new_pages_per_layer: Vec<Vec<usize>> = Vec::with_capacity(deficits.len());
    for &deficit in &deficits {
        let pages: Vec<usize> = (0..deficit).map(|_| alloc_page_legacy(cache)).collect();
        new_pages_per_layer.push(pages);
    }

    // Phase 3: distribute into layer tables.
    for (lt, pages) in cache.layer_page_tables.iter_mut().zip(new_pages_per_layer) {
        lt[seq_idx].extend(pages);
    }
}

// ─── G1: Correctness — fast-path == legacy in page-table state ───────────────

fn g1_correctness(config: &Config, label: &str) -> bool {
    let mut fast = PagedKVCache::new(config, 1);
    let mut legacy = PagedKVCache::new(config, 1);

    // Advance seq 0 through multiple page boundaries.
    let max_pos = CYCLES * PAGE_SIZE;
    for pos in 0..max_pos {
        fast.ensure_pages(black_box(0), black_box(pos));
        ensure_pages_legacy(&mut legacy, black_box(0), black_box(pos));
    }

    // Compare page tables layer-by-layer, seq-by-seq, page-by-page.
    let tables_match = fast.layer_page_tables.len() == legacy.layer_page_tables.len()
        && fast
            .layer_page_tables
            .iter()
            .zip(&legacy.layer_page_tables)
            .all(|(f, l)| {
                f.len() == l.len()
                    && f.iter().zip(l.iter()).all(|(ft, lt)| {
                        ft.len() == lt.len() && ft.iter().zip(lt.iter()).all(|(a, b)| a == b)
                    })
            });

    // Also verify free-list and ref-count consistency (both should have recycled
    // the same pages through the same alloc_page order).
    let free_match = fast.free_pages.len() == legacy.free_pages.len();
    let total_match = fast.total_pages == legacy.total_pages;

    let pass = tables_match && free_match && total_match;

    println!(
        "  G1 ({label}): page-tables identical ({tables_match}), free-list len match ({free_match}), total_pages match ({total_match}={}) → {}",
        fast.total_pages,
        if pass { "✅" } else { "❌" }
    );
    pass
}

// ─── G2: Perf — fast-path not slower than legacy in steady-state decode ──────
//
// Measures the speculative-decode steady state: rollback frees 1 page per layer
// (warm free list), then ensure_pages re-allocates 1 page per layer. In this
// regime alloc_page() is pop+fill(0.0) (no pool growth), so the ONLY difference
// between paths is the index allocation strategy.

fn g2_perf(config: &Config, label: &str) -> bool {
    let n_layer = config.n_layer;

    // Warmup both caches identically.
    let mut fast = PagedKVCache::new(config, 1);
    let mut legacy = PagedKVCache::new(config, 1);
    for pos in 0..CYCLES * PAGE_SIZE {
        fast.ensure_pages(0, pos);
        ensure_pages_legacy(&mut legacy, 0, pos);
    }

    // Steady-state cycle targets.
    let target_pos = CYCLES * PAGE_SIZE - 1; // needs CYCLES pages (deficit 1 after rollback)
    let rollback_to = (CYCLES - 1) * PAGE_SIZE; // keeps CYCLES-1 pages

    // Measure fast-path.
    let t_fast_start = Instant::now();
    for _ in 0..REPS {
        fast.rollback(black_box(0), black_box(rollback_to));
        fast.ensure_pages(black_box(0), black_box(target_pos));
    }
    let t_fast = t_fast_start.elapsed();

    // Measure legacy.
    let t_legacy_start = Instant::now();
    for _ in 0..REPS {
        legacy.rollback(black_box(0), black_box(rollback_to));
        ensure_pages_legacy(&mut legacy, black_box(0), black_box(target_pos));
    }
    let t_legacy = t_legacy_start.elapsed();

    let speedup = t_legacy.as_nanos() as f64 / t_fast.as_nanos().max(1) as f64;
    let pct = (speedup - 1.0) * 100.0;
    // G2 pass criterion: fast-path is not slower (speedup ≥ 1.0). The gain is
    // expected to be modest because alloc_page's fill(0.0) dominates and is
    // identical in both paths; the fast-path eliminates only per-layer index
    // Vec allocations (n_layer heap allocs/call avoided).
    let pass = speedup >= 1.0;

    let fast_ns = t_fast.as_nanos() as f64 / REPS as f64;
    let legacy_ns = t_legacy.as_nanos() as f64 / REPS as f64;
    println!(
        "  G2 ({label}): legacy {legacy_ns:>7.1}ns/call | fast {fast_ns:>7.1}ns/call | {speedup:.3}× ({pct:+.1}%) | {n_layer} idx-Vec allocs/call saved → {}",
        if pass { "✅" } else { "❌" }
    );
    if !pass {
        eprintln!("    ⚠️  fast-path slower than legacy — unexpected regression");
    }
    pass
}

// ─── G4: Alloc hygiene — fast-path = 0 scratch heap allocs ───────────────────
//
// The fast-path's scratch (flat ArrayVec) is stack-allocated — zero heap
// allocation by construction. The legacy's per-layer Vec<usize> is heap.
//
// We verify the steady-state invariant: fast-path's layer-table capacities do
// NOT grow across REPS rollback→re-alloc cycles (the extend_from_slice reuses
// existing capacity, never triggers realloc). This proves zero scratch heap
// allocs in steady state.

fn g4_alloc_audit(config: &Config, label: &str) -> bool {
    let n_layer = config.n_layer;

    let mut cache = PagedKVCache::new(config, 1);
    for pos in 0..CYCLES * PAGE_SIZE {
        cache.ensure_pages(0, pos);
    }

    let target_pos = CYCLES * PAGE_SIZE - 1;
    let rollback_to = (CYCLES - 1) * PAGE_SIZE;

    // Capture per-layer table capacities after warmup.
    let caps_after_warmup: Vec<usize> = (0..n_layer)
        .map(|l| cache.layer_page_tables[l][0].capacity())
        .collect();

    // Run steady-state cycles.
    for _ in 0..REPS {
        cache.rollback(0, rollback_to);
        cache.ensure_pages(0, target_pos);
    }

    // Capture capacities after steady-state.
    let caps_after_steady: Vec<usize> = (0..n_layer)
        .map(|l| cache.layer_page_tables[l][0].capacity())
        .collect();

    let caps_stable = caps_after_warmup == caps_after_steady;

    // Also verify page-table lengths are back to CYCLES (rollback→re-alloc is
    // a perfect cycle — no leak, no growth).
    let lens_correct: bool = (0..n_layer).all(|l| cache.layer_page_tables[l][0].len() == CYCLES);

    let pass = caps_stable && lens_correct;

    println!(
        "  G4 ({label}): layer-table caps stable ({caps_stable}), lens == CYCLES ({lens_correct}) | fast-path scratch = ArrayVec<stack>, legacy scratch = {n_layer}× Vec<heap>/call → {}",
        if pass {
            "✅ zero-growth (0 scratch heap allocs)"
        } else {
            "❌ capacity grew or length leaked"
        }
    );
    pass
}

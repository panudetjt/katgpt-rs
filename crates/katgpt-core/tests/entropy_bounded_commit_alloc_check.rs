#![cfg(feature = "entropy_bounded_commit")]
//! Issue 917 G4 — zero-allocation steady state for the EB commit policy.
//!
//! Separate single-fn binary (the `regime_probe_alloc_check` convention: a
//! CountingAllocator would pick up allocations from parallel tests).
//!
//! Audited paths, on caller-owned scratch after warmup:
//! - `position_stats_into` over a 64 × 4096 logits block (×100)
//! - `entropy_bounded_commit_stats` full-sort and capped (partial-selection)
//!   paths over 64 candidates (×1000 each)
//!
//! Then reports best-of-N latency through the shared `best_of_us` harness
//! (loud zero when the optimiser deletes the work); not gated.
//!
//! ```sh
//! cargo test -p katgpt-core --features entropy_bounded_commit \
//!   --test entropy_bounded_commit_alloc_check --release -- --nocapture
//! ```

use katgpt_core::entropy_bounded_commit::{
    EbCommitConfig, ErrorProxy, PositionStats, entropy_bounded_commit_stats, position_stats_into,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;
use ab_timing::best_of_us;

struct CountingAllocator {
    inner: System,
    allocated: AtomicU64,
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocated
            .fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { self.inner.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.inner.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.allocated.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator {
    inner: System,
    allocated: AtomicU64::new(0),
};

fn allocated_bytes() -> u64 {
    GLOBAL.allocated.load(Ordering::SeqCst)
}

#[test]
fn eb_commit_steady_state_is_alloc_free() {
    const POS: usize = 64;
    const VOCAB: usize = 4096;
    let mut rng = fastrand::Rng::with_seed(917);
    // Mixed sharpness rows so the commit prefix is non-trivial.
    let logits: Vec<f32> = (0..POS * VOCAB)
        .map(|i| {
            let sharp = ((i / VOCAB) % 4) as f32 * 3.0;
            rng.f32() * (1.0 + sharp)
        })
        .collect();
    let mut stats = vec![PositionStats::default(); POS];
    let mut hs = vec![0.0f32; POS];
    let mut ks = vec![0.0f32; POS];
    let mut cand: Vec<u32> = (0..POS as u32).collect();
    let full = EbCommitConfig {
        gamma: 0.5,
        max_commit: usize::MAX,
        proxy: ErrorProxy::Entropy,
    };
    let capped = EbCommitConfig {
        max_commit: 8,
        proxy: ErrorProxy::Margin,
        ..full
    };

    // Warmup (and liveness: the commit set must be non-empty).
    position_stats_into(&logits, POS, VOCAB, &mut stats);
    let k0 = entropy_bounded_commit_stats(&mut cand, &stats, &mut hs, &mut ks, &full);
    assert!(k0 >= 1, "no-stall liveness");

    // Allocation audit: plain loops, no timing inside the counted region.
    let mut acc = 0usize;
    let before = allocated_bytes();
    for _ in 0..100 {
        position_stats_into(black_box(&logits), POS, VOCAB, &mut stats);
    }
    for cfg in [&full, &capped] {
        for _ in 0..1000 {
            for (i, c) in cand.iter_mut().enumerate() {
                *c = i as u32;
            }
            acc +=
                entropy_bounded_commit_stats(&mut cand, black_box(&stats), &mut hs, &mut ks, cfg);
        }
    }
    let after = allocated_bytes();
    black_box(acc);
    assert!(
        acc >= 2000,
        "every pass committed ≥ 1 (liveness of the audited loops)"
    );
    assert_eq!(
        after - before,
        0,
        "EB commit steady state must not allocate"
    );

    // Latency (reported, not gated): the shared `best_of_us` harness panics
    // if a timed call measures 0 ns (the optimiser deleting the work).
    let stats_us = best_of_us(3, 20, || {
        let t = Instant::now();
        position_stats_into(black_box(&logits), POS, VOCAB, &mut stats);
        black_box(&stats);
        t.elapsed()
    });
    let mut time_commit = |cfg: &EbCommitConfig| {
        best_of_us(100, 2000, || {
            for (i, c) in cand.iter_mut().enumerate() {
                *c = i as u32;
            }
            let t = Instant::now();
            let k =
                entropy_bounded_commit_stats(&mut cand, black_box(&stats), &mut hs, &mut ks, cfg);
            black_box(k);
            t.elapsed()
        })
    };
    let full_ns = time_commit(&full) * 1000.0;
    let capped_ns = time_commit(&capped) * 1000.0;
    println!(
        "position_stats_into {POS}x{VOCAB}: {stats_us:.0} µs/block (best of 20); \
         commit full: {full_ns:.0} ns; capped(8): {capped_ns:.0} ns (best of 2000); k0 = {k0}"
    );
}

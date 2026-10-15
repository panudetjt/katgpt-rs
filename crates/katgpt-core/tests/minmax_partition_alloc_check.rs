//! Plan 616 T5 G4 — the partition DP's allocation shape (its own binary:
//! the CountingAllocator is per-thread global and sibling tests would
//! leak into any count taken beside them — the karc_alloc_check pattern).
//!
//! The DP allocates its scratch PER CALL (worst table n² + count n +
//! worst_pref n + the blocks vec) — this gate pins the measured ceiling
//! so the shape cannot silently grow. The one-shot analysis primitive has
//! no steady-state hot path, so "zero alloc" is not the honest contract
//! here; a bounded, pinned count is (the plan's G4 wording recorded the
//! aspiration, the gate records the measured shape — the plan carries the
//! deviation note).

#![cfg(feature = "minmax_partition")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::partition::{SMatrix, minmax_partition, minmax_partition_typed};

#[test]
fn g4_partition_alloc_count_pinned_ceiling() {
    use std::sync::atomic::Ordering;
    assert_counter_is_live();

    // Block-structured 256×256 — the DP's real work shape.
    let n = 256usize;
    let block = 8usize;
    let s = SMatrix::from_fn(n, |i, j| {
        if i == j {
            0.0
        } else if i / block == j / block {
            0.05 + ((i ^ j) % 7) as f32 * 0.001
        } else {
            1.0
        }
    });

    // Warm-up: any first-touch lazy state settles OUTSIDE the window.
    let _ = minmax_partition(&s, 0.1).unwrap();

    let (_, allocs_untyped) = alloc_delta(|| minmax_partition(&s, 0.1).unwrap());
    let (_, allocs_typed) = alloc_delta(|| {
        let types = vec![true; n];
        minmax_partition_typed(&s, 0.1, &types).unwrap()
    });

    // Measured at landing: 5 allocations per untyped call (worst table +
    // count + worst_pref + blocks + growth). The ceiling absorbs one
    // implementation detail (a Vec growth), never a structural change —
    // a scratch-restructuring refactor that DROPS the count is a win,
    // never a red. The measured count PRINTS on every pass — a silent
    // ceiling over an unmeasured count is the green-zero shape.
    const UNTYPED_CEILING: usize = 5;
    const TYPED_CEILING: usize = 6;
    println!(
        "alloc shape: untyped {allocs_untyped} (ceiling {UNTYPED_CEILING}), typed {allocs_typed} (ceiling {TYPED_CEILING})"
    );
    assert!(
        allocs_untyped <= UNTYPED_CEILING,
        "untyped DP allocated {allocs_untyped} > ceiling {UNTYPED_CEILING} — the scratch shape grew"
    );
    assert!(
        allocs_typed <= TYPED_CEILING,
        "typed DP allocated {allocs_typed} > ceiling {TYPED_CEILING} — the scratch shape grew"
    );

    // ALLOC_COUNT is read through the atomic-shaped API with the ignored
    // ordering arg (the macro's call-site convention).
    let _ = ALLOC_COUNT.load(Ordering::Relaxed);
    let _ = DEALLOC_COUNT.load(Ordering::Relaxed);
}

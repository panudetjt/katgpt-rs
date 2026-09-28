//! G4 — allocation-free steady-state tick (Issue 763 T5).
//!
//! The tick loop must be allocation-free in steady state: active-set churn,
//! delay-ring buckets, spike collection, and inject all reuse preallocated
//! capacity. The house G4 pattern (KARC G3 et al.): the fixture is
//! **stationary** — a strictly periodic drive whose per-period dynamics are
//! IDENTICAL every period.
//!
//! The single-fire regime is what makes exact periodicity achievable: mag 10
//! PSP fires the driven node exactly once (the post-refractory asymptote
//! 6.3 mV stays under the 7 mV threshold gap — no re-fires), and the period
//! (1400 ticks) exceeds the full transient tail (g snaps below `g_floor` at
//! ~660 ticks; the v perturbation reaches its bitwise fixed point at
//! ~650 more), so every drive lands on a bit-identical converged node.
//! Quasi-periodic fixtures (period 20 vs refrac 22 vs delay 18 —
//! incommensurate) keep drifting their burst re-fire boundaries forever and
//! legitimately grow a ring bucket on rare new peaks; that amortized-growth
//! behavior is documented in `LifReservoir`, not gated here.
//!
//! The counting allocator (tests/common) is per-thread (Issue 714) — this
//! test is single-threaded by construction.

#![cfg(feature = "lif_graph")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::lif_graph::{LifParams, LifReservoir, SignedAdjacency};
use std::sync::atomic::Ordering;

/// Snapshot of the current thread's allocation count (the macro-emitted
/// per-thread counter — Issue 714).
#[inline]
fn thread_alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

/// The stationary drive: the fixed driven set, one single-fire magnitude,
/// every `PERIOD` ticks.
///
/// The single-fire window (Shiu constants): peak Δv = g0·0.1575 with
/// g0 = w·(t_mbr/tau_syn), threshold gap 7 mV → fires iff w > 11.1; the
/// post-refractory residue g0·e^(−0.44) must stay subthreshold → w < 17.2.
/// w = 14 sits mid-window: exactly one fire per drive (at ~42 ticks),
/// deterministic.
///
/// `PERIOD` MUST be a multiple of the ring length (18 buckets): the fire
/// tick's schedule slot is (ring_head + 17) mod 18, and a period not
/// divisible by 18 shifts the slot every period — each shift lands in a
/// fresh bucket whose first use grows it 4→8→…→128 (6 allocs, measured).
/// A slot-aligned period uses ONE bucket, primed during warmup. (Long
/// irregular workloads eventually prime every bucket — amortized — but the
/// G4 gate measures the stationary orbit.) PERIOD = 1404 = 18×78 also
/// exceeds the full transient tail (g-snap ~350 + v fixed-point ~650).
const PERIOD: usize = 1404;
const MAG: f32 = 14.0;
const DRIVEN: [u32; 20] = [
    7, 93, 134, 208, 317, 429, 555, 666, 777, 888, 41, 172, 259, 380, 501, 612, 733, 844, 921, 975,
];

fn drive_periodic(r: &mut LifReservoir, tick: usize) {
    if tick.is_multiple_of(PERIOD) {
        for node in DRIVEN {
            r.inject(node, MAG);
        }
    }
}

#[test]
fn steady_state_tick_is_alloc_free() {
    assert_counter_is_live();
    // Subthreshold edges (w=1): no cascade — activity is exactly the driven
    // set plus decaying one-hop targets. The sparse regime the primitive
    // exists for.
    let adj = SignedAdjacency::er_matched(1000, 5000, 0.2, 1.0, 1.0, 2026);
    let mut r = LifReservoir::new(adj, LifParams::shiu());

    // Warmup: 2 periods — the first drive plus its full transient tail; the
    // second drive confirms the periodic orbit (state bit-identical at each
    // drive boundary).
    for t in 0..2 * PERIOD {
        drive_periodic(&mut r, t);
        r.step();
    }

    // Measured window: 4 periods of the identical orbit, zero allocations.
    let before = thread_alloc_count();
    let mut total_spikes = 0usize;
    for t in 0..4 * PERIOD {
        drive_periodic(&mut r, t + 2 * PERIOD);
        total_spikes += r.step().len();
    }
    let delta = thread_alloc_count() - before;
    assert_eq!(delta, 0, "steady-state tick allocated {delta} times");
    // Non-vacuity: every driven node fires once per period.
    assert!(
        total_spikes >= DRIVEN.len() * 3,
        "measured window fired only {total_spikes} times"
    );
}

#[test]
fn dense_reference_tick_is_alloc_free() {
    assert_counter_is_live();
    // step_dense rebuilds the active list every tick but also allocates
    // nothing in steady state (clear() retains capacity). Same n=1000 so
    // the shared DRIVEN set is in bounds.
    let adj = SignedAdjacency::er_matched(1000, 5000, 0.2, 1.0, 1.0, 99);
    let mut r = LifReservoir::new(adj, LifParams::shiu());
    for t in 0..2 * PERIOD {
        drive_periodic(&mut r, t);
        r.step_dense();
    }
    let before = thread_alloc_count();
    let mut total_spikes = 0usize;
    for t in 0..3 * PERIOD {
        drive_periodic(&mut r, t + 2 * PERIOD);
        total_spikes += r.step_dense().len();
    }
    let delta = thread_alloc_count() - before;
    assert_eq!(delta, 0, "dense steady-state tick allocated {delta} times");
    assert!(total_spikes > 0);
}

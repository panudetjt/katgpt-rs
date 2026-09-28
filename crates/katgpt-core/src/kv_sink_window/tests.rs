//! Arms for sinks + the bounded window (Issue 841).
//!
//! The load-bearing ones are w05 (a sink far outside the window is still a
//! sink — the classification ORDER, at exactly the sequence length this
//! exists for), w10 (the ceiling holds over an adversarial admission
//! sequence), and w14 (the UNBOUNDED policy reduces EXACTLY to the shipped
//! selector, index for index).

use super::*;
use crate::kv_eviction::select_evict_into;

fn positions(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

// ── capacity + fidelity ────────────────────────────────────────────────

#[test]
fn w01_capacity_is_the_ceiling() {
    assert_eq!(SinkWindowPolicy::new(4, 256).capacity(), 260);
    assert_eq!(SinkWindowPolicy::new(0, 0).capacity(), 0);
}

#[test]
fn w02_an_unbounded_window_does_not_wrap_to_something_small() {
    // The silent direction: a wrapping add would make the ceiling tiny and
    // evict everything on the policy that means "evict nothing".
    let p = SinkWindowPolicy::UNBOUNDED;
    assert_eq!(p.capacity(), usize::MAX);
    assert_eq!(p.evictions_needed(1_000_000), 0);
    assert_eq!(SinkWindowPolicy::new(4, usize::MAX).capacity(), usize::MAX);
}

#[test]
fn w03_evictions_needed_is_zero_inside_the_ceiling() {
    let p = SinkWindowPolicy::new(4, 16);
    assert_eq!(p.evictions_needed(0), 0);
    assert_eq!(p.evictions_needed(20), 0);
    assert_eq!(p.evictions_needed(21), 1);
    assert_eq!(p.evictions_needed(100), 80);
}

#[test]
fn w04_no_d_max_is_lossy_not_a_free_pass() {
    // Not knowing d_max is not evidence the window clears it.
    let p = SinkWindowPolicy::new(4, 1024);
    assert_eq!(p.fidelity(None), WindowFidelity::Lossy);
    assert!(!p.fidelity(None).is_lossless());
    assert_eq!(
        p.fidelity(Some(512)),
        WindowFidelity::Lossless { d_max: 512 }
    );
    assert_eq!(
        p.fidelity(Some(1024)),
        WindowFidelity::Lossless { d_max: 1024 }
    );
    assert_eq!(p.fidelity(Some(1025)), WindowFidelity::Lossy);
}

// ── classification ─────────────────────────────────────────────────────

#[test]
fn w05_a_sink_far_outside_the_window_is_still_a_sink() {
    // ⛔ The classification ORDER. A window-first predicate evicts the system
    // prompt at exactly the sequence length this policy exists for.
    let p = SinkWindowPolicy::new(4, 8);
    for pos in 0..4u64 {
        assert_eq!(p.classify(pos, 10_000), SlotClass::Sink, "pos {pos}");
    }
    assert_eq!(p.classify(4, 10_000), SlotClass::Stale);
}

#[test]
fn w06_the_window_is_a_trailing_distance() {
    let p = SinkWindowPolicy::new(0, 4);
    assert_eq!(p.classify(100, 100), SlotClass::Window, "distance 0");
    assert_eq!(p.classify(97, 100), SlotClass::Window, "distance 3");
    assert_eq!(p.classify(96, 100), SlotClass::Stale, "distance 4 is out");
}

#[test]
fn w07_early_positions_do_not_underflow_under_a_large_window() {
    // `current_pos - window` underflows for every early position; the
    // implementation compares a DISTANCE instead.
    let p = SinkWindowPolicy::new(0, usize::MAX);
    assert_eq!(p.classify(0, 0), SlotClass::Window);
    // ⚑ This one found a real defect: the literal `distance < window`
    // comparison classified the single extreme distance `Stale` under the
    // policy that means "nothing is stale". usize::MAX is a SENTINEL.
    assert_eq!(p.classify(0, u64::MAX), SlotClass::Window);
    assert_eq!(
        SinkWindowPolicy::UNBOUNDED.classify(0, u64::MAX),
        SlotClass::Window
    );
    let q = SinkWindowPolicy::new(0, 1_000_000);
    assert_eq!(q.classify(0, 5), SlotClass::Window);
}

#[test]
fn w08_a_zero_window_keeps_only_the_sinks() {
    let p = SinkWindowPolicy::new(2, 0);
    assert_eq!(p.classify(0, 9), SlotClass::Sink);
    assert_eq!(p.classify(1, 9), SlotClass::Sink);
    assert_eq!(p.classify(9, 9), SlotClass::Stale, "even the current token");
    assert_eq!(p.capacity(), 2);
}

#[test]
fn w09_pin_mask_is_true_exactly_for_sinks() {
    let p = SinkWindowPolicy::new(3, 4);
    let pos = positions(12);
    let mut mask = Vec::new();
    sink_pin_mask_into(&p, &pos, 11, &mut mask);
    assert_eq!(mask.len(), 12);
    for (i, &m) in mask.iter().enumerate() {
        assert_eq!(m, i < 3, "slot {i}");
    }
}

// ── the ceiling, end to end ────────────────────────────────────────────

#[test]
fn w10_the_ceiling_holds_over_an_adversarial_admission_sequence() {
    // The product claim, asserted as a loop rather than as a formula: admit
    // 500 tokens one at a time, evict per the policy, and require the live
    // set to stay under the ceiling AND to always contain every sink.
    let p = SinkWindowPolicy::new(4, 16);
    let mut live: Vec<u64> = Vec::new();
    let mut evict = Vec::new();

    for step in 0..500u64 {
        live.push(step);
        // A deliberately hostile scorer: the sinks look WORST and the oldest
        // rows look best, so anything ordering by score alone evicts exactly
        // the wrong rows.
        let scores: Vec<f32> = live
            .iter()
            .map(|&pos| match pos < 4 {
                true => -1.0,
                false => 1000.0 - pos as f32,
            })
            .collect();
        let k = p.evictions_needed(live.len());
        select_evict_windowed(&p, &live, step, &scores, k, &mut evict);
        assert_eq!(evict.len(), k, "step {step}: asked for {k}");

        let mut doomed: Vec<usize> = evict.clone();
        doomed.sort_unstable();
        for &i in doomed.iter().rev() {
            live.remove(i);
        }

        assert!(
            live.len() <= p.capacity(),
            "step {step}: {} live over a ceiling of {}",
            live.len(),
            p.capacity()
        );
        // Only the sinks ADMITTED so far — at step 0 there is one token, and
        // demanding four would be the test asserting its own arithmetic
        // rather than the policy's.
        for s in 0..4u64.min(step + 1) {
            assert!(live.contains(&s), "step {step}: sink {s} was evicted");
        }
    }
    // And the steady state is actually AT the ceiling, not trivially small —
    // a policy that evicted everything would satisfy the bound above.
    assert_eq!(live.len(), p.capacity());
}

#[test]
fn w11_a_sink_is_never_evicted_at_any_budget() {
    let p = SinkWindowPolicy::new(3, 2);
    let pos = positions(10);
    let scores = vec![f32::NEG_INFINITY; 10]; // every row looks maximally evictable
    let mut out = Vec::new();
    for k in 0..20 {
        select_evict_windowed(&p, &pos, 9, &scores, k, &mut out);
        assert!(
            out.iter().all(|&i| i >= 3),
            "k {k}: evicted a sink — {out:?}"
        );
        assert!(out.len() <= k);
    }
}

#[test]
fn w12_stale_rows_go_before_window_rows_whatever_their_score() {
    // Out of POLICY, not merely low-scoring. A blended key would keep the
    // high-scoring stale row and break the ceiling.
    let p = SinkWindowPolicy::new(0, 3);
    let pos = positions(8); // current 7 => window is {5,6,7}, stale {0..4}
    let mut scores = vec![0.0f32; 8];
    scores[0] = 1e9; // the oldest row looks the most valuable there is
    let mut out = Vec::new();
    select_evict_windowed(&p, &pos, 7, &scores, 1, &mut out);
    assert_eq!(out, vec![0], "the stale row goes first despite its score");
}

#[test]
fn w13_stale_rows_are_ordered_oldest_first() {
    let p = SinkWindowPolicy::new(0, 2);
    let pos: Vec<u64> = vec![30, 10, 20, 41]; // current 41 => window {40,41}
    let scores = vec![0.0f32; 4];
    let mut out = Vec::new();
    select_evict_windowed(&p, &pos, 41, &scores, 3, &mut out);
    assert_eq!(out, vec![1, 2, 0], "positions 10, 20, 30 in that order");
}

// ── the reduction identity ─────────────────────────────────────────────

#[test]
fn w14_unbounded_reduces_exactly_to_the_shipped_selector() {
    // ⛔ An "off" switch that is a second code path is not an off switch.
    // Index for index, over a score vector with ties, zeros and a NaN.
    let scores = vec![0.5f32, 0.1, 0.1, f32::NAN, 0.9, 0.0, 0.3, 0.1];
    let pos = positions(scores.len());
    let unpinned = vec![false; scores.len()];
    let mut a = Vec::new();
    let mut b = Vec::new();
    for k in 0..=scores.len() + 2 {
        select_evict_windowed(&SinkWindowPolicy::UNBOUNDED, &pos, 7, &scores, k, &mut a);
        select_evict_into(&scores, k, &unpinned, &mut b);
        assert_eq!(a, b, "k {k}: windowed {a:?} vs shipped {b:?}");
    }
}

#[test]
fn w15_sink_only_reduces_to_the_shipped_selector_under_its_own_pin_mask() {
    // The other composition: this module's sink rule + the shipped scorer.
    let scores = vec![0.5f32, 0.1, 0.9, 0.0, 0.3];
    let pos = positions(scores.len());
    let p = SinkWindowPolicy::new(2, usize::MAX);
    let mut mask = Vec::new();
    sink_pin_mask_into(&p, &pos, 4, &mut mask);
    let mut a = Vec::new();
    let mut b = Vec::new();
    for k in 0..=6 {
        select_evict_windowed(&p, &pos, 4, &scores, k, &mut a);
        select_evict_into(&scores, k, &mask, &mut b);
        assert_eq!(a, b, "k {k}");
    }
}

#[test]
fn w16_a_corrupt_score_is_never_evicted_first() {
    // NaN orders LAST under min-ordering — the conservative direction.
    let p = SinkWindowPolicy::UNBOUNDED;
    let pos = positions(3);
    let scores = vec![f32::NAN, 0.5, 0.1];
    let mut out = Vec::new();
    select_evict_windowed(&p, &pos, 2, &scores, 1, &mut out);
    assert_eq!(out, vec![2], "the real minimum, not the NaN");
}

#[test]
fn w17_an_unscored_row_is_evicted_first_not_skipped() {
    // `scores` shorter than `positions`: the missing entries read 0.0. The
    // conservative direction for a row the caller has not scored yet.
    let p = SinkWindowPolicy::UNBOUNDED;
    let pos = positions(4);
    let scores = vec![0.9f32, 0.8]; // slots 2 and 3 unscored
    let mut out = Vec::new();
    select_evict_windowed(&p, &pos, 3, &scores, 2, &mut out);
    assert_eq!(out, vec![2, 3]);
}

#[test]
fn w18_empty_and_zero_budget_cases() {
    let p = SinkWindowPolicy::default();
    let mut out = vec![99usize];
    select_evict_windowed(&p, &[], 0, &[], 5, &mut out);
    assert!(out.is_empty(), "the buffer is cleared, never left stale");
    out.push(99);
    select_evict_windowed(&p, &positions(4), 3, &[0.0; 4], 0, &mut out);
    assert!(out.is_empty());
}

#[test]
fn w19_the_output_is_a_set_no_index_twice() {
    // The two phases append to one buffer; an overlap in the predicates
    // would evict the same slot twice and under-deliver the ceiling.
    let p = SinkWindowPolicy::new(2, 4);
    let pos = positions(20);
    let scores: Vec<f32> = (0..20).map(|i| (i % 7) as f32).collect();
    let mut out = Vec::new();
    select_evict_windowed(&p, &pos, 19, &scores, 20, &mut out);
    let mut sorted = out.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), out.len(), "duplicate index in {out:?}");
    assert!(out.iter().all(|&i| i >= 2), "no sink");
    // 20 slots, 2 sinks, k = 20: every non-sink row is planned — 14 stale
    // (positions 2..=15) then the 4 window rows (16..=19), in that order.
    assert_eq!(out.len(), 18, "everything but the sinks");
    assert!(
        out[..14].iter().all(|&i| (2..16).contains(&i)),
        "the stale block comes first: {out:?}"
    );
}

#[test]
fn w20_determinism_same_input_same_plan() {
    let p = SinkWindowPolicy::new(2, 5);
    let pos = positions(30);
    let scores: Vec<f32> = (0..30).map(|i| ((i * 13) % 11) as f32 / 11.0).collect();
    let mut a = Vec::new();
    let mut b = Vec::new();
    select_evict_windowed(&p, &pos, 29, &scores, 9, &mut a);
    select_evict_windowed(&p, &pos, 29, &scores, 9, &mut b);
    assert_eq!(a, b);
}

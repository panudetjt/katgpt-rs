//! Bench 845 — Issue 845: `channel_aware::simd_dot_f32` must stay a
//! DELEGATION and never regrow a private kernel.
//!
//! ```text
//! cargo test -p katgpt-attn --release --features dash_attn \
//!     --test bench_845_channel_aware_dot_dispatch -- --nocapture
//!
//! # the second configuration is part of the claim, not a variation on it:
//! RUSTFLAGS="-C target-feature=+avx2" cargo test -p katgpt-attn --release \
//!     --features dash_attn --test bench_845_channel_aware_dot_dispatch -- --nocapture
//! ```
//!
//! # What it was, and why the guard is what survives
//!
//! Until Issue 845 this file's `simd_dot_f32` carried ~200 lines of
//! hand-written NEON **and** AVX2 intrinsics — a same-named duplicate of
//! `katgpt_types::simd::simd_dot_f32`, which was already in this crate's
//! dependency graph through `katgpt-core`'s wholesale re-export, and which
//! **seven other files in this crate — four of them in this very `dash_attn/`
//! directory — already called**.
//!
//! ⛔ The duplicate's AVX2 arm was gated `#[cfg(target_feature = "avx2")]`, a
//! **compile-time** predicate that is OFF by default on x86_64. So on every
//! ordinary build the shipped path fell through to a 4-accumulator scalar loop
//! while the kernel it duplicated probed CPUID once and used AVX2. Measured in
//! both configurations before the deletion, since the local arm only existed
//! in one of them:
//!
//! | dot length | default build | `+avx2` build |
//! |---|---|---|
//! | 32 | local **2.72×** slower | local **1.83×** slower |
//! | 64 | 4.12× | 2.68× |
//! | 128 | 6.54× | 3.86× |
//! | 256 | 7.76× | 4.78× |
//! | 1024 | 6.37× | 4.00× |
//!
//! Agreement was `≤ 1e-6` in the default build and **bit-identical** under
//! `+avx2`, so the substitution was lossless as well as faster. After it:
//! 0.97–1.02× and bit-identical at every length — the 2-argument wrapper costs
//! nothing measurable.
//!
//! # So this is a GATE, not a report
//!
//! The measurement above is history and lives in Issue 845 and in the
//! function's own doc. What a target can still assert every time it runs is the
//! property that repair established: **the local entry point and the shipped
//! kernel are the same code path.** That is pinned two ways, and the first is
//! the one that matters:
//!
//! 1. **Bit-identity.** A delegation returns exactly what it delegates to. Any
//!    re-introduced private kernel — intrinsics, a chunked loop, a different
//!    accumulator count — changes the summation order and reds here. Arch-free
//!    by construction: it compares a function against itself through a wrapper,
//!    so it means the same thing on NEON, AVX2 and scalar.
//! 2. **A loose timing band** (0.80 – 1.30). It cannot catch what bit-identity
//!    already catches, and it is not redundant: it catches the wrapper itself
//!    becoming expensive — a length recomputation, a bounds-check cascade, a
//!    lost `#[inline]`.
//!
//! The band is deliberately wide. A tight one would be a bar on the box (see
//! AGENTS.md on sequential A/B ratios), and the interesting failure here is 2.7×
//! or worse, not 10%.

#![cfg(feature = "dash_attn")]

use katgpt_attn::dash_attn::channel_aware::simd_dot_f32 as local_dot;
use katgpt_core::simd::{simd_dot_f32 as shipped_dot, simd_level};
use std::hint::black_box;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;
use ab_timing::ab_median_ratio;

/// The routing dimensions the function's own doc used to name ("For
/// routing_dim=32 … For routing_dim=64") plus the head dims a dash_attn score
/// row runs at.
const LENS: [(usize, usize); 5] = [(32, 2000), (64, 1500), (128, 800), (256, 400), (1024, 120)];

/// Dots per timed call — a routing query against several blocks, so the fixed
/// per-call cost is amortised the way it is in `forward_indexer`.
const ROWS: usize = 16;

/// Timing band for a pure delegation. Wide on purpose: the regression this
/// exists to catch measured 2.7×–7.8×.
const BAND: (f64, f64) = (0.80, 1.30);

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        2.0 * (((self.0 >> 33) as f32) / (1u64 << 31) as f32) - 1.0
    }
}

#[test]
fn channel_aware_dot_is_a_delegation_not_a_transcription() {
    println!("\n   Issue 845 — channel_aware::simd_dot_f32 must BE katgpt_types' kernel");
    println!(
        "   arch = {}, cfg!(target_feature=\"avx2\") = {}, shipped dispatch = {:?}",
        std::env::consts::ARCH,
        cfg!(target_feature = "avx2"),
        simd_level()
    );
    println!(
        "   {:>6}  {:>12}  {:>12}  {:>12}  {:>13}",
        "len", "local ns/dot", "shipped ns/dot", "local/shipped", "bit-identical"
    );

    for &(len, iters) in &LENS {
        let mut rng = Lcg::new(0x0845_0000 + len as u64);
        let a: Vec<f32> = (0..len).map(|_| rng.next()).collect();
        let b: Vec<f32> = (0..len * ROWS).map(|_| rng.next()).collect();
        let mut sink_local = vec![0.0f32; ROWS];
        let mut sink_shipped = vec![0.0f32; ROWS];

        for (j, s) in sink_local.iter_mut().enumerate() {
            *s = local_dot(&a, &b[j * len..j * len + len]);
        }
        for (j, s) in sink_shipped.iter_mut().enumerate() {
            *s = shipped_dot(&a, &b[j * len..j * len + len], len);
        }

        // GATE 1 — the load-bearing one. Exact equality is the right assertion
        // for a delegation, and it is exactly what a re-introduced private
        // kernel cannot satisfy.
        let identical = sink_local == sink_shipped;
        let maxd = sink_local
            .iter()
            .zip(&sink_shipped)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);

        let r = ab_median_ratio(
            11,
            iters,
            iters / 2,
            |_| {
                for (j, s) in sink_shipped.iter_mut().enumerate() {
                    *s = shipped_dot(black_box(&a), black_box(&b[j * len..j * len + len]), len);
                }
                black_box(sink_shipped[0]);
            },
            |_| {
                for (j, s) in sink_local.iter_mut().enumerate() {
                    *s = local_dot(black_box(&a), black_box(&b[j * len..j * len + len]));
                }
                black_box(sink_local[0]);
            },
        );

        println!(
            "   {len:>6}  {:>12.1}  {:>12.1}  {:>12.2}  {:>13}",
            r.b_ns_per_iter() / ROWS as f64,
            r.a_ns_per_iter() / ROWS as f64,
            r.median,
            identical
        );

        assert!(
            identical,
            "GATE 1 FAIL at len={len}: channel_aware::simd_dot_f32 no longer returns \
             bit-identical results to katgpt_core::simd::simd_dot_f32 (max |Δ| = {maxd:e}). \
             That means it has grown a kernel of its own again — the Issue 845 regression. \
             It is not a tolerance question: a delegation is bit-identical by definition."
        );
        assert_eq!(
            r.ratios.len(),
            r.rounds,
            "instrument FAIL at len={len}: {} of {} rounds survived",
            r.ratios.len(),
            r.rounds
        );
        assert!(
            r.median >= BAND.0 && r.median <= BAND.1,
            "GATE 2 FAIL at len={len}: the wrapper measures {:.2}x the shipped kernel, \
             outside the {:.2}–{:.2} band. Bit-identity still holds, so this is the \
             WRAPPER becoming expensive rather than a new kernel — check `#[inline]`, \
             the `min(len)` computation and bounds checks (rounds {:.4}..{:.4})",
            r.median,
            BAND.0,
            BAND.1,
            r.min(),
            r.max()
        );
    }

    println!(
        "\n   PASS: the local entry point is the shipped kernel, bit for bit, at every \
         length — and costs nothing measurable for the 2-argument form.\n"
    );
}

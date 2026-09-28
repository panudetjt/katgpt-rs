//! Bench 844 — Issue 844: at what LENGTH does delegating to `simd_dot_f32`
//! start paying, and where does it cost?
//!
//! ```text
//! cargo test --release --test bench_844_dot_delegation_crossover -- --nocapture
//! ```
//!
//! # The question, and why it is not obvious in either direction
//!
//! This workspace holds **221** `fn *dot*`-shaped functions over 2,471 tracked
//! `.rs` files. 41 already delegate to the shipped ISA-dispatched
//! `katgpt_types::simd::simd_dot_f32`; a couple of dozen are plain
//! single-accumulator loops. It is tempting to read the second group as a
//! backlog — and two of the first three read by hand say otherwise, each with
//! its reason written at the site:
//!
//! - `katgpt-attn-match/src/score_matrix_simd.rs::dot_8wide` records that an
//!   8-accumulator hand-rolled pattern measured **1.26× SLOWER** than the
//!   simple loop on M3/NEON, because the single accumulator is what lets LLVM
//!   emit the dot-product idiom.
//! - `katgpt-core/src/similarity.rs::dot_8` is deliberately simple *"so recos
//!   stays bit-deterministic across platforms — the Phase 2 GOAT G1 gate
//!   depends on it"*. Converting it would break a stated contract.
//!
//! ⛔ **Those two facts are about DIFFERENT comparisons and conflating them is
//! the trap.** `dot_8wide`'s note is scalar-vs-scalar: a hand-rolled
//! multi-accumulator loop against a plain one. The question here is
//! scalar-vs-**intrinsic**: a plain loop against a real per-ISA kernel. The
//! first says *do not hand-roll a chunked dot*; it says nothing about whether
//! to call one that already exists, and the answer to that turns out to depend
//! entirely on the length.
//!
//! # What this measures
//!
//! `rows` independent dots of length `len`, so the kernel's fixed per-call
//! cost is amortised exactly as it is inside a real GEMM inner loop — not
//! measured once in isolation, which would flatter the plain loop. Both arms
//! go through `tests/common/ab_timing.rs` (interleaved chunks, median of
//! per-pair ratios, loud zero) for the usual reason: two sequential loops over
//! arms that differ by 20× in cost is where the box decides the verdict.
//!
//! A **report** — instrument health only, no bar. The quantity is an input to
//! a per-site engineering decision, and the crossover is a property of this
//! box's ISA rather than of any one call site.

use katgpt_core::simd::{simd_dot_f32, simd_level};
use std::hint::black_box;

#[path = "common/ab_timing.rs"]
mod ab_timing;
use ab_timing::ab_median_ratio;

/// `(len, iters_per_round)`. The interesting region is 8..64; 4 and 256 are
/// the anchors that show the two asymptotes.
const LENS: [(usize, usize); 6] = [
    (4, 4000),
    (8, 4000),
    (16, 2500),
    (32, 1500),
    (64, 800),
    (256, 250),
];

/// Dots per timed call — one row of a 32×32 tile GEMM, the shape
/// `linalg::kron_tile`'s step 2 actually runs.
const ROWS: usize = 32;

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

/// Arm A — delegate to the shipped kernel, once per dot.
#[inline]
fn dots_delegating(t: &[f32], b: &[f32], len: usize, y: &mut [f32]) {
    for (j, yv) in y.iter_mut().enumerate() {
        *yv = simd_dot_f32(&t[..len], &b[j * len..j * len + len], len);
    }
}

/// Arm B — a plain single-accumulator loop, the form most of this workspace's
/// private dots take.
#[inline]
fn dots_plain(t: &[f32], b: &[f32], len: usize, y: &mut [f32]) {
    let t_row = &t[..len];
    for (j, yv) in y.iter_mut().enumerate() {
        let b_row = &b[j * len..j * len + len];
        let mut acc = 0.0f32;
        for (&x, &bb) in t_row.iter().zip(b_row.iter()) {
            acc += x * bb;
        }
        *yv = acc;
    }
}

#[test]
fn dot_delegation_crossover_by_length() {
    println!("\n   Issue 844 — simd_dot_f32 vs a plain loop, by dot length");
    println!(
        "   dispatch: simd_level() = {:?}   ({ROWS} dots per timed call)",
        simd_level()
    );
    println!(
        "   {:>5}  {:>12}  {:>12}  {:>11}  {:>13}",
        "len", "simd ns/dot", "plain ns/dot", "plain/simd", "max |delta|"
    );

    let mut rows: Vec<(usize, f64)> = Vec::new();

    for &(len, iters) in &LENS {
        let mut rng = Lcg::new(0x0844_0000 + len as u64);
        let t: Vec<f32> = (0..len).map(|_| rng.next()).collect();
        let b: Vec<f32> = (0..len * ROWS).map(|_| rng.next()).collect();
        let mut y_a = vec![0.0f32; ROWS];
        let mut y_b = vec![0.0f32; ROWS];

        // The arms must agree before their speeds mean anything. They are NOT
        // bit-identical — different summation orders — so this is a tolerance,
        // and it is printed rather than merely asserted.
        dots_delegating(&t, &b, len, &mut y_a);
        dots_plain(&t, &b, len, &mut y_b);
        let maxd = y_a
            .iter()
            .zip(&y_b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        assert!(
            maxd <= 1e-4,
            "arms disagree at len={len} by {maxd:e} — that is a correctness \
             problem, not a timing one"
        );

        let r = ab_median_ratio(
            11,
            iters,
            iters / 2,
            |_| {
                dots_delegating(black_box(&t), black_box(&b), len, &mut y_a);
                black_box(y_a[0]);
            },
            |_| {
                dots_plain(black_box(&t), black_box(&b), len, &mut y_b);
                black_box(y_b[0]);
            },
        );

        println!(
            "   {len:>5}  {:>12.1}  {:>12.1}  {:>11.2}  {maxd:>13e}",
            r.a_ns_per_iter() / ROWS as f64,
            r.b_ns_per_iter() / ROWS as f64,
            r.median
        );
        assert!(
            r.median.is_finite() && r.median > 0.0,
            "instrument FAIL at len={len}"
        );
        assert_eq!(
            r.ratios.len(),
            r.rounds,
            "instrument FAIL at len={len}: {} of {} rounds survived",
            r.ratios.len(),
            r.rounds
        );
        rows.push((len, r.median));
    }

    println!();
    match rows
        .windows(2)
        .find(|w| w[0].1 < 1.0 && w[1].1 >= 1.0)
        .map(|w| (w[0].0, w[1].0))
    {
        Some((lo, hi)) => {
            println!(
                "   READING: crossover between len={lo} and len={hi}. BELOW it the plain \
                 loop wins — the kernel's fixed per-call cost is not amortised by that \
                 few elements. ABOVE it the delegation wins and keeps widening."
            );
            println!(
                "   So the ~23 private single-accumulator dots in this workspace are NOT a \
                 backlog: at small fixed D a plain loop is the correct choice, and the \
                 candidates are the runtime-length, large-D sites only (Issue 844 T2/T3)."
            );
        }
        None => {
            // Two opposite causes print identically as "no crossover", and the
            // aarch64/NEON run (Issue 844 T4, 2026-09-19) hit the first one:
            // delegation winning THROUGHOUT is the fixed-overhead model working
            // (the intersection sits below the measured range), not failing.
            if rows.iter().all(|&(_, r)| r >= 1.0) {
                println!(
                    "   READING: no crossover in {}..={} — delegation wins THROUGHOUT \
                     (every plain/simd ratio ≥ 1.0; measured on aarch64/NEON, Issue \
                     844 T4). The crossover sits BELOW the measured range, not absent.",
                    LENS[0].0,
                    LENS[LENS.len() - 1].0
                );
            } else {
                println!(
                    "   READING: no crossover in {}..={} — the plain loop wins \
                     throughout, which contradicts the fixed-overhead model and \
                     should be investigated before being quoted.",
                    LENS[0].0,
                    LENS[LENS.len() - 1].0
                );
            }
        }
    }
    println!(
        "   ⚠ Per-ISA property — both arches measured: x86_64/AVX2 crossover \
         between 16 and 32; aarch64/NEON (M3, Issue 844 T4, 2026-09-19): \
         delegation wins at EVERY length ≥ 4 — compile-time dispatch gives the \
         kernel a ~0.8 ns floor against x86_64's ~3.3 ns runtime-probe floor.\n"
    );
}

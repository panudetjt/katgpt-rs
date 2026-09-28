//! Issue 817 — `simd_argmax_f32` dispatch A/B: the single-pass kernel vs the
//! two-pass idiom on the x86_64 path.
//!
//! **VERDICT (2026-09-17, measured): NEGATIVE — the single-pass AVX2 kernel
//! does NOT earn the dispatch slot.** It loses `iid` 2.8–3.5× and `early`
//! 3–4.7× at n ≥ 256 on both profile arms; it only wins when the maximum
//! sits in the last few percent of the scan (1.5–1.9×) and at n=64
//! (fixed-overhead, ~20 ns). The two-pass structure is fundamentally strong
//! on x86_64: the ILP'd `max_ps` reduce + a vectorized early-exiting
//! `position()` beat a latency-bound cmp/blend chain. The port was REVERTED
//! (demote-on-loss); full tables: `/.benchmarks/812_argmax_avx2_single_pass_goat.md`.
//!
//! This harness is KEPT as the reopen instrument: arm b measures the LIVE
//! dispatch (currently the two-pass itself, so a healthy run reads ~1.00 —
//! that control is the point). Re-run it after a major toolchain or CPU
//! change; if the two-pass absolute columns regress or a re-landed kernel
//! beats it on the common shapes, the dispatch question reopens.
//!
//! Found by [Bench 810](../../.benchmarks/810_argtopk_distribution_crossover.md):
//! the two-pass structure (`simd_max_f32` reduce + `position(== max)`) costs
//! TWO full scans when the maximum sits late in the array, and ~one scan when
//! it sits early — its cost depends on WHERE the maximum is, which is not a
//! property an argmax primitive's caller controls. The single-pass kernel
//! tracked (max, index) per SIMD lane in one traversal.
//!
//! Arms:
//! - **a (baseline)** — the two-pass idiom, exactly what the x86_64 branch
//!   executes (`simd_max_f32` + `position`).
//! - **b (candidate)** — `simd_argmax_f32`, i.e. the real dispatch (currently
//!   the same two-pass after the revert; during the Issue-817 measurement it
//!   was the single-pass AVX2 kernel).
//!
//! `speedup = 1 / median_ratio = two_pass ÷ dispatch`; during the measured
//! negative, **> 1.00 meant the (then-live) kernel won**. Instrument: the
//! shared Issue-723 interleaved median-of-ratios (`ab_median_ratio`), with
//! per-iteration input churn and sink consumption on BOTH arms (fat-LTO
//! anti-hoist).
//!
//! Max-position shapes: the whole question is where the max sits, so the
//! distribution axis is reduced to that: `iid` (uniform — max position
//! uniform), `early` (max in the first 5%), `mid` (max at n/2),
//! `late` (max in the last 5%). All deterministic (seeded splitmix64).
//!
//! Run:
//!
//! ```bash
//! cargo test -p katgpt-types --test bench_817_argmax_dispatch_ab \
//!   --release -- --nocapture --test-threads=1
//! # profile axis (auto-vectorizes the two-pass arm — the other "old" config):
//! RUSTFLAGS="-C target-feature=+avx2" cargo test -p katgpt-types \
//!   --test bench_817_argmax_dispatch_ab --release -- --nocapture --test-threads=1
//! ```

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use std::hint::black_box;

use katgpt_types::simd::{simd_argmax_f32, simd_max_f32};

/// The incumbent x86_64 path, verbatim.
fn two_pass(x: &[f32]) -> (usize, f32) {
    let m = simd_max_f32(x);
    (x.iter().position(|&v| v == m).unwrap_or(0), m)
}

/// Seeded splitmix64 — deterministic, no global RNG state.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Max-position shapes — the axis the two-pass cost actually depends on.
#[derive(Clone, Copy)]
enum Shape {
    Iid,
    Early,
    Mid,
    Late,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::Iid => "iid",
            Shape::Early => "early",
            Shape::Mid => "mid",
            Shape::Late => "late",
        }
    }

    fn make(self, n: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64(seed ^ 0x2545_F491_4F6C_DD1D);
        let mut v: Vec<f32> = (0..n).map(|_| 0.9 * rng.next_f32()).collect();
        let pos = match self {
            Shape::Iid => n, // no planted max
            Shape::Early => n / 20,
            Shape::Mid => n / 2,
            Shape::Late => n - 1 - n / 20,
        };
        if pos < n {
            v[pos] = 1.0; // strictly above the 0.9-floor noise
        }
        v
    }
}

/// Per-iteration churn (fat-LTO anti-hoist): perturb four positions with a
/// golden-ratio delta. Identical in both arms; cancels in the ratio.
#[inline]
fn churn(buf: &mut [f32], i: usize, base: &[f32]) {
    let n = buf.len();
    let quarter = (n / 4).max(1);
    for p in 0..4usize {
        let pos = (i + p * quarter) % n;
        let t = (i as u64).wrapping_add(p as u64);
        let delta = ((t as f64 * 0.618_033_988_749_894_9).fract() - 0.5) as f32 * 2.0e-3;
        buf[pos] = base[pos] + delta;
    }
}

#[test]
fn bench_817_argmax_dispatch_ab() {
    let n_values = [64usize, 256, 1024, 4096];
    let shapes = [Shape::Iid, Shape::Early, Shape::Mid, Shape::Late];
    let rounds = 9;

    println!(
        "\n== Issue 817 argmax dispatch A/B (speedup = two_pass/dispatch, >1.00 dispatch wins) =="
    );
    println!(
        "{:<8} {:>6} {:>12} {:>12} {:>9}  {:>17}",
        "shape", "n", "twopass ns", "dispatch ns", "speedup", "round band"
    );

    for &shape in &shapes {
        for &n in &n_values {
            let base = shape.make(n, 0x0807_0000 + (n as u64) * 7919);

            // Correctness on the base scores: the dispatch must equal the
            // two-pass on finite inputs (this also pins the no-AVX2 sanity
            // control — there the two arms are the same code).
            let got = simd_argmax_f32(&base);
            let want = two_pass(&base);
            assert_eq!(got, want, "dispatch mismatch at {n}, {}", shape.name());

            let iters = ((4_000_000.0 / n.max(1) as f64) as usize).clamp(256, 100_000);
            let mut scratch_a = base.clone();
            let mut sink_a = 0u64;
            let mut scratch_b = base.clone();
            let mut sink_b = 0u64;

            let ab = ab_timing::ab_median_ratio(
                rounds,
                iters,
                iters,
                |i| {
                    churn(&mut scratch_a, i, &base);
                    let r = two_pass(&scratch_a);
                    sink_a = sink_a.wrapping_add(r.0 as u64).rotate_left(1);
                },
                |i| {
                    churn(&mut scratch_b, i, &base);
                    let r = simd_argmax_f32(&scratch_b);
                    sink_b = sink_b.wrapping_add(r.0 as u64).rotate_left(1);
                },
            );
            black_box((sink_a, sink_b));

            println!(
                "{:<8} {:>6} {:>12.1} {:>12.1} {:>8.2}x  {:>6.2}..{:<6.2}",
                shape.name(),
                n,
                ab.a_ns_per_iter(),
                ab.b_ns_per_iter(),
                1.0 / ab.median,
                1.0 / ab.max(),
                1.0 / ab.min(),
            );
        }
    }
}

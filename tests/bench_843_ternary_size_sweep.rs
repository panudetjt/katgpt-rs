//! Bench 843 — Issue 843 T2: does the `plasma_path` ternary dense matvec beat
//! the f32 one at ANY size, and where is the crossover?
//!
//! ```text
//! cargo test --release --features plasma_path --test bench_843_ternary_size_sweep -- --nocapture
//! ```
//!
//! # Why a sweep and not another single point
//!
//! Issue 843 filed one measurement — at `1024 × 1024`, `simd_ternary_matvec`
//! took 122.7 µs against `simd_matvec`'s 63.3 µs, i.e. **1.94× slower than the
//! multiply it removes**, with the runtime dispatch disclosed as `Avx2` and the
//! kernel a healthy 4.32× its own scalar arm. One point is not a verdict on a
//! kernel whose argument is about **memory**: at `m = 1024` the f32 operand is
//! 4 MiB and still fits this box's L3, which is exactly the regime where the
//! ternary footprint advantage buys nothing. The honest question is where the
//! f32 operand stops fitting, and whether the ternary arm wins there.
//!
//! # It prices Issue 839 T7 as a side effect, and that is deliberate
//!
//! Issue 839 T7 (ternary-factor fusion) assumes `A`, `B` quantized to
//! `{-1,0,+1}` will be cheaper than f32 factors. The factor widths there are
//! **32 and 64**, which are the bottom of this very curve — so the rows below
//! `m = 256` are not padding, they are the measurement T7's premise needs. It
//! is a **proxy**, stated as one: a Kronecker stage is two 32×32 *matrix*
//! products, not one matvec, so the arithmetic intensity differs. What the
//! `m = 32` row settles is the narrower question underneath — whether a
//! multiplication-free 32×32 multiply is cheaper than an f32 one on this box
//! at all.
//!
//! Footprints, which are the whole hypothesis:
//!
//! | `m` | f32 operand | packed ternary (2 bitplanes) | ratio |
//! |---|---|---|---|
//! | 32 | 4 KiB | 0.25 KiB | 16× |
//! | 64 | 16 KiB | 1 KiB | 16× |
//! | 128 | 64 KiB | 4 KiB | 16× |
//! | 256 | 256 KiB | 16 KiB | 16× |
//! | 512 | 1 MiB | 64 KiB | 16× |
//! | 1024 | 4 MiB | 256 KiB | 16× |
//! | 2048 | 16 MiB | 1 MiB | 16× |
//! | 4096 | 64 MiB | 4 MiB | 16× |
//!
//! # This is a REPORT, and it says so rather than pretending to gate
//!
//! It asserts only **instrument health** — every round survived, every ratio
//! finite and positive. It deliberately pins no bar: the quantity under
//! measurement is the thing Issue 843 T4 has to decide about a **default-on**
//! flag, and a bar written before the sweep would be a bar written from the
//! hypothesis. AGENTS.md's own count of targets that "assert nothing" is a
//! finding about gates that *look* like gates; this one is labelled.
//!
//! The A/B arms go through `tests/common/ab_timing.rs` (interleaved chunks,
//! median of per-pair ratios, loud zero) because a size sweep is the *worst*
//! place for two sequential loops: the arms differ in working set by 16×, so a
//! drift that lands on whichever arm ran second is indistinguishable from the
//! effect being measured.

#![cfg(feature = "plasma_path")]

use katgpt_core::simd::{simd_level, simd_matvec, simd_ternary_matvec};
use katgpt_core::types::TernaryWeights;
use std::hint::black_box;

#[path = "common/ab_timing.rs"]
mod ab_timing;
use ab_timing::ab_median_ratio;

/// `(m, iters_per_round)` — iterations shrink with `m³`-ish cost so every row
/// spends a comparable wall budget. Hand-chosen rather than derived: a derived
/// count would be one more thing to get wrong in a target whose only job is to
/// report.
const SWEEP: [(usize, usize); 8] = [
    // The low end is not padding: 32 and 64 are the Kronecker FACTOR widths of
    // Issue 839, whose T7 assumes ternary factors will be cheaper than f32
    // ones. This sweep is the cheapest instrument that can price that
    // assumption, so it reaches down to where the answer lives.
    (32, 4000),
    (64, 2000),
    (128, 800),
    (256, 200),
    (512, 60),
    (1024, 20),
    (2048, 6),
    (4096, 2),
];

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
fn t2_ternary_vs_f32_dense_size_sweep() {
    println!("\n   Issue 843 T2 — ternary vs f32 dense matvec, size sweep");
    println!("   runtime dispatch: simd_level() = {:?}", simd_level());
    println!(
        "   {:>6}  {:>12}  {:>12}  {:>9}  {:>9}  {:>16}",
        "m", "f32 ns/call", "tern ns/call", "tern/f32", "rounds", "f32 operand"
    );

    let mut any_ternary_win = false;
    let mut rows: Vec<(usize, f64)> = Vec::new();

    for &(m, iters) in &SWEEP {
        let mut rng = Lcg::new(0x0843_0000 + m as u64);
        let mat: Vec<f32> = (0..m * m).map(|_| rng.next()).collect();
        let x: Vec<f32> = (0..m).map(|_| rng.next()).collect();
        // Thresholded at ±0.33 so all three trit values occur (~1/3 zeros); a
        // sign-only draw would make the zero branch dead and flatter the kernel.
        let trits: Vec<i8> = (0..m * m)
            .map(|_| {
                let v = rng.next();
                if v > 0.33 {
                    1
                } else if v < -0.33 {
                    -1
                } else {
                    0
                }
            })
            .collect();
        let tw = TernaryWeights::pack_from_i8(&trits, m, m).expect("ternary pack");
        drop(trits); // 16 MiB at m=4096, and nothing reads it again

        let mut y_f32 = vec![0.0f32; m];
        let mut y_tern = vec![0.0f32; m];

        let r = ab_median_ratio(
            11,
            iters,
            iters.max(2) / 2,
            // a = the f32 kernel the ternary path claims to replace
            |_| {
                simd_matvec(&mut y_f32, black_box(&mat), black_box(&x), m, m);
                black_box(y_f32[0]);
            },
            // b = the multiplication-free path
            |_| {
                simd_ternary_matvec(black_box(&tw), black_box(&x), &mut y_tern);
                black_box(y_tern[0]);
            },
        );

        let f32_ns = r.a_ns_per_iter();
        let tern_ns = r.b_ns_per_iter();
        let operand_mib = (m * m * 4) as f64 / (1024.0 * 1024.0);
        println!(
            "   {m:>6}  {f32_ns:>12.0}  {tern_ns:>12.0}  {:>9.2}  {:>4} of {:>2}  {operand_mib:>12.1} MiB",
            r.median,
            r.ratios.len(),
            r.rounds
        );
        assert!(
            r.median.is_finite() && r.median > 0.0,
            "instrument FAIL at m={m}: median {} is not a measurement",
            r.median
        );
        assert_eq!(
            r.ratios.len(),
            r.rounds,
            "instrument FAIL at m={m}: {} of {} rounds survived",
            r.ratios.len(),
            r.rounds
        );
        if r.median < 1.0 {
            any_ternary_win = true;
        }
        rows.push((m, r.median));
    }

    // The reading, printed rather than asserted — see the module doc.
    println!();
    match rows
        .windows(2)
        .find(|w| w[0].1 >= 1.0 && w[1].1 < 1.0)
        .map(|w| (w[0].0, w[1].0))
    {
        Some((lo, hi)) => println!(
            "   READING: crossover between m={lo} and m={hi} — ternary becomes the \
             cheaper arm there. Issue 843 T4 has a size threshold to work with."
        ),
        None if any_ternary_win => println!(
            "   READING: ternary wins at some size but not monotonically — read the \
             column, the sweep is not a single crossover."
        ),
        None => println!(
            "   READING: NO crossover in {}..={} — ternary is the slower arm at every \
             size measured, so the 1024-point in Issue 843 is not an L3 artifact and \
             the footprint advantage does not convert into latency anywhere on this box.",
            SWEEP[0].0,
            SWEEP[SWEEP.len() - 1].0
        ),
    }
    println!(
        "   ⚠ One box (x86_64/AVX2), one thread. Issue 843 T1 (aarch64/NEON) is a \
         DIFFERENT kernel and is not measured here.\n"
    );
}

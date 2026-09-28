//! Issue 700 G2 gate: cost of the entry-point reslice on the hot dot kernel.
//! Interleaved A/B is done by the RUNNER (alternating binaries); this binary
//! just reports ns/call per size. Inputs are black_box'd, not results.
use katgpt_types::simd::simd_dot_f32;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    // matmul-row shape: `rows` calls of width `d` — per-call overhead matters
    // most at small d, which is exactly where a per-call check could show up.
    let cases: &[(usize, usize)] = &[
        (8, 200_000),
        (64, 100_000),
        (256, 50_000),
        (1024, 20_000),
        (4096, 5_000),
    ];
    for &(d, iters) in cases {
        let a = vec![1.000_001f32; d];
        let b = vec![0.999_999f32; d];
        // warm
        let mut acc = 0.0f32;
        for _ in 0..iters / 10 {
            acc += simd_dot_f32(black_box(&a), black_box(&b), black_box(d));
        }
        let t = Instant::now();
        for _ in 0..iters {
            acc += simd_dot_f32(black_box(&a), black_box(&b), black_box(d));
        }
        let el = t.elapsed();
        println!(
            "d={d} ns_per_call={:.3} sink={acc}",
            el.as_nanos() as f64 / iters as f64
        );
    }
}

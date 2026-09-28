//! Kernel-level microbenchmark: f32×f32 vs f16×f32 vs f16×f16 dot products.
//!
//! Issue 201 Phase 1 decision gate: validates whether the f16×f16 widening-FMA
//! kernel can beat f32×f32 at the **raw dot-product level**, before investing
//! in a full `ForwardContextF16` forward path.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-types --test prof_f16_f16_kernel --release -- --ignored --nocapture
//! ```

#![allow(clippy::cast_precision_loss)]

use half::f16;
use katgpt_types::simd::{simd_dot_f16_f16, simd_dot_f16_f32, simd_dot_f32};

fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

/// Generate a deterministic pseudo-random f32 vector in [-1, 1].
fn gen_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = seed;
    (0..len)
        .map(|_| {
            // xorshift64
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Map to [-1, 1]
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 1.0
        })
        .collect()
}

/// Convert an f32 vector to f16.
fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

#[test]
fn f16_f16_correctness() {
    // Verify simd_dot_f16_f16 produces approximately the same result as
    // the f32 reference (within f16 rounding precision).
    for &len in &[16usize, 32, 64, 128, 256, 1024] {
        let a_f32 = gen_f32(len, 42);
        let b_f32 = gen_f32(len, 99);
        let a_f16 = to_f16(&a_f32);
        let b_f16 = to_f16(&b_f32);

        let ref_dot = simd_dot_f32(&a_f32, &b_f32, len);
        let f16_dot = simd_dot_f16_f16(&a_f16, &b_f16, len);

        let rel_err = (ref_dot - f16_dot).abs() / ref_dot.abs().max(1e-6);
        assert!(
            rel_err < 0.05,
            "f16×f16 correctness FAIL at len={len}: ref={ref_dot:.6}, f16={f16_dot:.6}, rel_err={rel_err:.4}"
        );
    }
    println!("✅ f16×f16 correctness: all lengths within 5% relative error");
}

#[test]
#[ignore]
fn prof_f16_f16_kernel_speedup() {
    const ITERS: usize = 10_000;
    const GATE: f64 = 1.5;

    println!();
    println!("═══ f16×f16 Kernel Microbenchmark (Issue 201 Phase 1) ═══");
    println!("CPU: Apple Silicon (fp16 + fhm target features)");
    println!();

    // Test at multiple vector lengths:
    // - Small (fits in L1): measures pure instruction throughput
    // - Medium (fits in L2/L3): measures L2/L3 bandwidth
    // - Large (exceeds L3): measures DRAM bandwidth (the f16 win should materialize here)
    let configs: &[(usize, &str)] = &[
        (64, "L1 (64 elem, 256B f32 / 128B f16)"),
        (256, "L1 (256 elem, 1KB f32 / 512B f16)"),
        (1024, "L2 (1K elem, 4KB f32 / 2KB f16)"),
        (4096, "L2/L3 (4K elem, 16KB f32 / 8KB f16)"),
        (16384, "L3 (16K elem, 64KB f32 / 32KB f16)"),
        (65536, "L3+ (64K elem, 256KB f32 / 128KB f16)"),
        (262144, "DRAM (256K elem, 1MB f32 / 512KB f16)"),
        (1048576, "DRAM (1M elem, 4MB f32 / 2MB f16)"),
    ];

    println!(
        "{:<40} {:>10} {:>10} {:>10} {:>9} {:>9}",
        "Config", "f32xf32 ns", "f16xf32 ns", "f16xf16 ns", "wf16/f32", "f16f16/f32"
    );
    println!("{}", "─".repeat(110));

    let mut large_f16f16_speedup = 0.0f64;

    for &(len, label) in configs {
        let a_f32 = gen_f32(len, 42);
        let b_f32 = gen_f32(len, 99);
        let a_f16 = to_f16(&a_f32);
        let b_f16 = to_f16(&b_f32);

        // Warmup
        for _ in 0..100 {
            black_box(simd_dot_f32(&a_f32, &b_f32, len));
            black_box(simd_dot_f16_f32(&a_f16, &b_f32, len));
            black_box(simd_dot_f16_f16(&a_f16, &b_f16, len));
        }

        // Measure f32×f32
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..ITERS {
            sum += black_box(simd_dot_f32(&a_f32, &b_f32, len));
        }
        let f32_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

        // Measure f16×f32 (weight-only, Issue 200 path)
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            sum += black_box(simd_dot_f16_f32(&a_f16, &b_f32, len));
        }
        let f16_f32_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

        // Measure f16×f16 (full f16, Issue 201 path)
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            sum += black_box(simd_dot_f16_f16(&a_f16, &b_f16, len));
        }
        let f16_f16_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

        // Prevent optimizer from removing the loop
        if sum == 999_999.0 {
            println!("impossible");
        }

        let wf16_speedup = f32_ns / f16_f32_ns;
        let f16f16_speedup = f32_ns / f16_f16_ns;

        if len >= 65536 {
            large_f16f16_speedup = large_f16f16_speedup.max(f16f16_speedup);
        }

        println!(
            "{label:<40} {f32_ns:>10.1} {f16_f32_ns:>10.1} {f16_f16_ns:>10.1} {wf16_speedup:>8.3}x {f16f16_speedup:>8.3}x"
        );
    }

    println!();
    println!("── Decision gate (Issue 201 Phase 1) ──");
    println!("  f16xf16 best speedup at L3+ sizes (>=65536): {large_f16f16_speedup:.3}x");
    if large_f16f16_speedup >= GATE {
        println!("  PASS — f16xf16 kernel is >={GATE}x faster at L3+ sizes");
        println!("  -> Proceed to Phase 2: implement ForwardContextF16 + full forward path");
    } else {
        println!("  FAIL — f16xf16 kernel is NOT >={GATE}x faster at L3+ sizes");
        println!(
            "  -> The full-f16 path won't beat f32 either. Close Issue 201 with negative result."
        );
    }
}

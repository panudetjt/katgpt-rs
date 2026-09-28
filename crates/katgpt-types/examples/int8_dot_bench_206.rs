//! int8×int8 vs f32×f32 dot product microbenchmark (Issue 206).
//!
//! Tests the hypothesis from Bench 563 L146-149: "INT8 with INT8 activations
//! (the quantized-inference literature's regime) ... a different dequant path
//! entirely."
//!
//! Same methodology as Bench 563 (f16×f16 FHM investigation):
//! - Baseline: f32 dot (4-accumulator FMA, matches `simd_dot_f32`)
//! - Challenger: int8 dot using ARM SDOT (inline asm — stable, unlike
//!   `vdotq_s32` which needs nightly `stdarch_neon_dotprod`), or NEON
//!   `vmull_s8`+`vpaddlq_s16` (stable on all ARMv8), or scalar fallback
//! - Sizes: representative Moka conv dot sizes (16, 32, 108, 144, 162, 288, 324)
//!
//! # Run
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo run --release -p katgpt-types --example int8_dot_bench_206
//! ```

use std::hint::black_box;
use std::time::Instant;

const SIZES: &[usize] = &[16, 32, 108, 144, 162, 288, 324];

// ── f32 baseline ────────────────────────────────────────────────────

#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = [0.0f32; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            acc[0] = (*a.get_unchecked(i)).mul_add(*b.get_unchecked(i), acc[0]);
            acc[1] = (*a.get_unchecked(i + 1)).mul_add(*b.get_unchecked(i + 1), acc[1]);
            acc[2] = (*a.get_unchecked(i + 2)).mul_add(*b.get_unchecked(i + 2), acc[2]);
            acc[3] = (*a.get_unchecked(i + 3)).mul_add(*b.get_unchecked(i + 3), acc[3]);
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < len {
        unsafe {
            sum = (*a.get_unchecked(i)).mul_add(*b.get_unchecked(i), sum);
        }
        i += 1;
    }
    sum
}

// ── int8 scalar dot (auto-vectorizable) ─────────────────────────────

#[inline(always)]
fn dot_i8_scalar(a: &[i8], b: &[i8], len: usize) -> i32 {
    let mut acc0: i32 = 0;
    let mut acc1: i32 = 0;
    let mut acc2: i32 = 0;
    let mut acc3: i32 = 0;
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            acc0 += (*a.get_unchecked(i) as i32) * (*b.get_unchecked(i) as i32);
            acc1 += (*a.get_unchecked(i + 1) as i32) * (*b.get_unchecked(i + 1) as i32);
            acc2 += (*a.get_unchecked(i + 2) as i32) * (*b.get_unchecked(i + 2) as i32);
            acc3 += (*a.get_unchecked(i + 3) as i32) * (*b.get_unchecked(i + 3) as i32);
        }
        i += 4;
    }
    let mut sum = acc0 + acc1 + acc2 + acc3;
    while i < len {
        unsafe {
            sum += (*a.get_unchecked(i) as i32) * (*b.get_unchecked(i) as i32);
        }
        i += 1;
    }
    sum
}

// ── aarch64 NEON int8 kernels ───────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        int16x8_t, int32x4_t, vaddq_s32, vaddvq_s32, vdupq_n_s32, vget_high_s8, vget_low_s8,
        vld1q_s8, vmull_s8, vpaddlq_s16,
    };

    /// SDOT via inline assembly (ARMv8.2-A dotprod).
    ///
    /// `vdotq_s32` needs nightly `stdarch_neon_dotprod`; inline asm is stable.
    /// The `sdot` instruction does 16 i8 multiplies + 4 i32 accumulates in
    /// ONE instruction.
    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn dot_i8_sdot(a: &[i8], b: &[i8], len: usize) -> i32 {
        unsafe {
            let mut acc: int32x4_t = vdupq_n_s32(0);
            let mut i = 0;
            let chunks16 = len / 16;

            for _ in 0..chunks16 {
                let va = vld1q_s8(a.as_ptr().add(i));
                let vb = vld1q_s8(b.as_ptr().add(i));
                // sdot v_acc.4s, v_a.16b, v_b.16b
                core::arch::asm!(
                    "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                    acc = inout(vreg) acc,
                    a = in(vreg) va,
                    b = in(vreg) vb,
                    options(pure, nomem, nostack, preserves_flags),
                );
                i += 16;
            }

            let mut sum = vaddvq_s32(acc);
            while i < len {
                sum += (*a.get_unchecked(i) as i32) * (*b.get_unchecked(i) as i32);
                i += 1;
            }
            sum
        }
    }

    /// SMULL+VPADDL int8 dot (all ARMv8 NEON targets, no dotprod required).
    #[inline]
    pub unsafe fn dot_i8_vmull(a: &[i8], b: &[i8], len: usize) -> i32 {
        unsafe {
            let mut acc: int32x4_t = vdupq_n_s32(0);
            let mut i = 0;
            let chunks16 = len / 16;

            for _ in 0..chunks16 {
                let va = vld1q_s8(a.as_ptr().add(i));
                let vb = vld1q_s8(b.as_ptr().add(i));

                let lo_a = vget_low_s8(va);
                let lo_b = vget_low_s8(vb);
                let mul_lo: int16x8_t = vmull_s8(lo_a, lo_b);

                let hi_a = vget_high_s8(va);
                let hi_b = vget_high_s8(vb);
                let mul_hi: int16x8_t = vmull_s8(hi_a, hi_b);

                // Pairwise-add i16x8 → i32x4, accumulate
                let sum_lo = vpaddlq_s16(mul_lo);
                let sum_hi = vpaddlq_s16(mul_hi);
                acc = vaddq_s32(acc, vaddq_s32(sum_lo, sum_hi));

                i += 16;
            }

            let mut sum = vaddvq_s32(acc);
            while i < len {
                sum += (*a.get_unchecked(i) as i32) * (*b.get_unchecked(i) as i32);
                i += 1;
            }
            sum
        }
    }

    /// Check if dotprod is available at runtime.
    pub fn has_dotprod() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
}

// ── Activation quantization ─────────────────────────────────────────

#[inline]
fn quantize_f32_to_i8(input: &[f32], output: &mut [i8]) -> f32 {
    debug_assert_eq!(input.len(), output.len());
    let max_abs = input.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if max_abs < 1e-30 {
        output.fill(0);
        return 1.0;
    }
    let inv_scale = 127.0 / max_abs;
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let q = (*inp * inv_scale).round();
        *out = q.clamp(-128.0, 127.0) as i8;
    }
    max_abs / 127.0
}

// ── Bench helpers ───────────────────────────────────────────────────

struct BenchResult {
    f32_ns: f64,
    i8_scalar_ns: f64,
    i8_vmull_ns: f64,
    i8_sdot_ns: f64,
    full_path_ns: f64,
    amortized_ns: f64,
    f32_amortized_ns: f64,
    rel_err: f32,
}

fn bench_size(size: usize) -> BenchResult {
    const ITERS: usize = 2_000_000;
    const WARMUP: usize = 50_000;

    let a_f32: Vec<f32> = (0..size)
        .map(|i| (i as f32).sin() * 1.7 + (i as f32 * 0.3).cos() * 0.8)
        .collect();
    let b_f32: Vec<f32> = (0..size)
        .map(|i| (i as f32 * 0.7 + 1.0).cos() * 1.5 + (i as f32 * 0.13).sin() * 0.6)
        .collect();

    let mut a_i8 = vec![0i8; size];
    let mut b_i8 = vec![0i8; size];
    let a_scale = quantize_f32_to_i8(&a_f32, &mut a_i8);
    let b_scale = quantize_f32_to_i8(&b_f32, &mut b_i8);

    // Correctness
    let f32_result = dot_f32(&a_f32, &b_f32, size);
    let i8_result = dot_i8_scalar(&a_i8, &b_i8, size) as f32 * a_scale * b_scale;
    let rel_err = if f32_result.abs() > 1e-6 {
        ((f32_result - i8_result).abs() / f32_result.abs()) * 100.0
    } else {
        0.0
    };

    // f32 baseline
    let mut sink = 0.0f32;
    for _ in 0..WARMUP {
        sink = black_box(dot_f32(&a_f32, &b_f32, size));
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        sink = black_box(dot_f32(
            black_box(&a_f32),
            black_box(&b_f32),
            black_box(size),
        ));
    }
    let f32_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

    // int8 scalar
    let mut sink_i = 0i32;
    for _ in 0..WARMUP {
        sink_i = black_box(dot_i8_scalar(&a_i8, &b_i8, size));
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        sink_i = black_box(dot_i8_scalar(
            black_box(&a_i8),
            black_box(&b_i8),
            black_box(size),
        ));
    }
    let i8_scalar_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

    // int8 NEON vmull
    #[cfg(target_arch = "aarch64")]
    let i8_vmull_ns = {
        let mut _s = 0i32;
        for _ in 0..WARMUP {
            _s = black_box(unsafe { neon::dot_i8_vmull(&a_i8, &b_i8, size) });
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            _s = black_box(unsafe {
                neon::dot_i8_vmull(black_box(&a_i8), black_box(&b_i8), black_box(size))
            });
        }
        start.elapsed().as_nanos() as f64 / ITERS as f64
    };
    #[cfg(not(target_arch = "aarch64"))]
    let i8_vmull_ns = 0.0;

    // int8 NEON SDOT (if dotprod available)
    #[cfg(target_arch = "aarch64")]
    let i8_sdot_ns = if neon::has_dotprod() {
        let mut _s = 0i32;
        for _ in 0..WARMUP {
            _s = black_box(unsafe { neon::dot_i8_sdot(&a_i8, &b_i8, size) });
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            _s = black_box(unsafe {
                neon::dot_i8_sdot(black_box(&a_i8), black_box(&b_i8), black_box(size))
            });
        }
        start.elapsed().as_nanos() as f64 / ITERS as f64
    } else {
        0.0
    };
    #[cfg(not(target_arch = "aarch64"))]
    let i8_sdot_ns = 0.0;

    // Full path: quantize both inputs + int8 dot (per-dot, pessimistic)
    let mut scratch_a = vec![0i8; size];
    let mut scratch_b = vec![0i8; size];
    for _ in 0..WARMUP {
        let s1 = quantize_f32_to_i8(&a_f32, &mut scratch_a);
        let s2 = quantize_f32_to_i8(&b_f32, &mut scratch_b);
        sink_i = black_box(dot_i8_scalar(&scratch_a, &scratch_b, size));
        let _ = (s1, s2);
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let s1 = quantize_f32_to_i8(black_box(&a_f32), black_box(&mut scratch_a));
        let s2 = quantize_f32_to_i8(black_box(&b_f32), black_box(&mut scratch_b));
        sink_i = black_box(dot_i8_scalar(
            black_box(&scratch_a),
            black_box(&scratch_b),
            black_box(size),
        ));
        let _ = (s1, s2);
    }
    let full_path_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

    // Amortized path: simulate a realistic conv layer.
    // Quantize the activation ONCE, then do OUT_CH dot products against
    // OUT_CH pre-quantized weight rows. This is the production pattern:
    // weights are int8 on disk (zero quantization cost), activations are
    // quantized once per forward pass.
    //
    // We use OUT_CH=32 (Moka's trunk channel count) to match the workload.
    const OUT_CH: usize = 32;
    let weight_rows: Vec<Vec<i8>> = (0..OUT_CH)
        .map(|oc| {
            let mut w = vec![0i8; size];
            quantize_f32_to_i8(
                &(0..size)
                    .map(|j| (oc as f32 * 0.1 + j as f32 * 0.05).sin() * 2.0)
                    .collect::<Vec<_>>(),
                &mut w,
            );
            w
        })
        .collect();
    let _weight_scales: Vec<f32> = (0..OUT_CH).map(|oc| 0.01 + oc as f32 * 0.001).collect();

    // f32 amortized baseline: OUT_CH f32 dots (no quantization)
    let weight_rows_f32: Vec<Vec<f32>> = (0..OUT_CH)
        .map(|oc| {
            (0..size)
                .map(|j| (oc as f32 * 0.1 + j as f32 * 0.05).sin() * 2.0)
                .collect()
        })
        .collect();

    for _ in 0..WARMUP {
        for wrow in &weight_rows_f32 {
            sink = black_box(dot_f32(&a_f32, wrow, size));
        }
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        for wrow in &weight_rows_f32 {
            sink = black_box(dot_f32(black_box(&a_f32), black_box(wrow), black_box(size)));
        }
    }
    let f32_amortized_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

    // int8 amortized: quantize input once + OUT_CH SDOT dots
    for _ in 0..WARMUP {
        let act_scale = quantize_f32_to_i8(&a_f32, &mut scratch_a);
        for wrow in &weight_rows {
            #[cfg(target_arch = "aarch64")]
            {
                if neon::has_dotprod() {
                    sink_i = black_box(unsafe { neon::dot_i8_sdot(&scratch_a, wrow, size) });
                } else {
                    sink_i = black_box(dot_i8_scalar(&scratch_a, wrow, size));
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                sink_i = black_box(dot_i8_scalar(&scratch_a, wrow, size));
            }
            let _ = act_scale;
        }
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        let act_scale = quantize_f32_to_i8(black_box(&a_f32), black_box(&mut scratch_a));
        for wrow in &weight_rows {
            #[cfg(target_arch = "aarch64")]
            {
                if neon::has_dotprod() {
                    sink_i = black_box(unsafe {
                        neon::dot_i8_sdot(black_box(&scratch_a), black_box(wrow), black_box(size))
                    });
                } else {
                    sink_i = black_box(dot_i8_scalar(
                        black_box(&scratch_a),
                        black_box(wrow),
                        black_box(size),
                    ));
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                sink_i = black_box(dot_i8_scalar(
                    black_box(&scratch_a),
                    black_box(wrow),
                    black_box(size),
                ));
            }
            let _ = act_scale;
        }
    }
    let amortized_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

    let _ = (sink, sink_i);

    BenchResult {
        f32_ns,
        i8_scalar_ns,
        i8_vmull_ns,
        i8_sdot_ns,
        full_path_ns,
        amortized_ns,
        f32_amortized_ns,
        rel_err,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  int8×int8 vs f32×f32 dot product microbenchmark (Issue 206)   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    #[cfg(target_arch = "aarch64")]
    {
        let has_dp = neon::has_dotprod();
        println!(
            "Arch: aarch64 (NEON ✓, dotprod={}, target-cpu=native)",
            if has_dp { "✓" } else { "✗" }
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("Arch: non-aarch64 (only f32 baseline + int8 scalar available)");
    }
    println!();

    println!(
        "┌──────┬─────────┬──────────┬──────────┬──────────┬───────────────────────┬────────┐"
    );
    println!(
        "│ size │ f32 ns  │ i8scal   │ i8sdot   │ fullpath │ amortized (32 OC)     │ relerr │"
    );
    println!(
        "│      │ per dot │ per dot  │ per dot  │ per dot  │ f32 ns   i8 ns   sdup │        │"
    );
    println!(
        "├──────┼─────────┼──────────┼──────────┼──────────┼───────────────────────┼────────┤"
    );

    let mut dot_passes = 0;
    let mut full_passes = 0;

    for &size in SIZES {
        let r = bench_size(size);

        let best_i8 = {
            let mut best = r.i8_scalar_ns;
            if r.i8_vmull_ns > 0.0 && r.i8_vmull_ns < best {
                best = r.i8_vmull_ns;
            }
            if r.i8_sdot_ns > 0.0 && r.i8_sdot_ns < best {
                best = r.i8_sdot_ns;
            }
            best
        };

        let dot_speedup = if best_i8 > 0.0 {
            r.f32_ns / best_i8
        } else {
            0.0
        };
        let full_speedup = r.f32_ns / r.full_path_ns;

        if dot_speedup >= 2.0 {
            dot_passes += 1;
        }
        if full_speedup >= 1.5 {
            full_passes += 1;
        }

        // Amortized: compare f32 amortized (OUT_CH f32 dots) vs int8 amortized
        // (quantize once + OUT_CH SDOT dots)
        let amort_speedup = r.f32_amortized_ns / r.amortized_ns;

        if amort_speedup >= 1.5 {
            full_passes += 1;
        }

        let vmull_s = if r.i8_vmull_ns > 0.0 {
            format!("{:.1}", r.i8_vmull_ns)
        } else {
            "—".into()
        };
        let sdot_s = if r.i8_sdot_ns > 0.0 {
            format!("{:.1}", r.i8_sdot_ns)
        } else {
            "—".into()
        };
        let _ = vmull_s;

        println!(
            "│ {:>4} │ {:>7.1} │ {:>8.1} │ {:>8} │ {:>8.1} │ {:>7.0} {:>7.0} {:>5.2}×│ {:>5.2}% │",
            size,
            r.f32_ns,
            r.i8_scalar_ns,
            sdot_s,
            r.full_path_ns,
            r.f32_amortized_ns,
            r.amortized_ns,
            amort_speedup,
            r.rel_err
        );
    }

    println!(
        "└──────┴─────────┴──────────┴──────────┴──────────┴───────────────────────┴────────┘"
    );
    println!();

    println!("═══ Decision Gate ═══");
    println!(
        "  T1 (dot-only ≥2.0× at any size):           {}",
        if dot_passes > 0 {
            format!("✅ PASS ({}/{})", dot_passes, SIZES.len())
        } else {
            "❌ FAIL".to_string()
        }
    );
    println!(
        "  T2 (amortized conv ≥1.5× at any size):     {}",
        if full_passes > 0 {
            format!("✅ PASS ({}/{})", full_passes, SIZES.len())
        } else {
            "❌ FAIL".to_string()
        }
    );
    println!();

    if dot_passes > 0 && full_passes > 0 {
        println!("VERDICT: PROMISING — proceed to full conv2d_int8 + WASM port.");
    } else if dot_passes > 0 {
        println!("VERDICT: PARTIAL — dot kernel fast but amortized overhead still significant.");
    } else {
        println!("VERDICT: NEGATIVE — int8 dot not significantly faster. Close Issue 206.");
    }
}

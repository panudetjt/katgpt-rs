#![cfg(feature = "vortex_flow")]
//! Benchmark — SIMD Register `TopK` for k≤16 (Plan 256 Phase 1)
//!
//! Compares the `argtopk` dispatch against the `argtopk_scalar_heap` baseline
//! across k=4, 8, 16 and n=64, 128, 256, 512, 1024 block counts.
//!
//! Which kernel `argtopk` actually dispatches is **arch-dependent** (Issue 808
//! T1): NEON takes the whole k ≤ 16 range; `x86_64` AVX2 only k ≤
//! `AVX2_ARGTOPK_K_MAX` (= 4), so the k ∈ {8, 16} rows time the SAME kernel
//! twice on `x86_64` — dispatch overhead + noise, not a kernel contrast. The
//! table prints the per-row dispatch label so no column can be misread
//! (Issue 874 T3: the pre-808 "SIMD (ns/call)" header was this file's one
//! stale surface; every other section below already documents the dispatch).
//!
//! Run: `cargo test --features vortex_flow --test bench_256_simd_topk -- --nocapture`
//!
//! Issue-808 section (distribution matrix + per-k crossover) additionally
//! carries the profile axis: run it a second time under
//! `RUSTFLAGS="-C target-feature=+avx2"` — the scalar fallback auto-vectorizes
//! under that flag, so the comparison is profile-shaped, not just arch-shaped.

use katgpt_rs::dash_attn::block_topk::{argtopk, argtopk_scalar_heap};

// ── Helpers ───────────────────────────────────────────────────

/// Deterministic pseudo-random score generator (index-based seed).
fn make_scores(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i.wrapping_mul(2_654_435_761)).wrapping_add(seed.wrapping_mul(40503))) as f32;
            (x * 0.000_1).sin() * 0.5 + 0.5
        })
        .collect()
}

/// Reference scalar argtopk — full sort + take top-k.
fn argtopk_reference(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// The kernel `argtopk` actually dispatches to at `k`, per architecture.
///
/// Mirrors `argtopk_with_scratch`'s dispatch exactly: k=1 → `simd_argmax_f32`;
/// k>16 → selection sort; the k ≤ 16 register path is NEON-whole on aarch64
/// and bound to `AVX2_ARGTOPK_K_MAX` (= 4) on `x86_64`, with k above the bound
/// routing to `argtopk_scalar_heap` (Issue 808 T1).
fn dispatch_kernel(k: usize) -> &'static str {
    if k > 16 {
        return "selection_sort";
    }
    if k == 1 {
        return "simd_argmax";
    }
    {
        #[cfg(target_arch = "x86_64")]
        {
            if k <= katgpt_rs::dash_attn::block_topk::AVX2_ARGTOPK_K_MAX {
                "avx2_sorted_register"
            } else {
                "scalar_heap (dispatch)"
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            "neon_sorted_register"
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            "scalar_heap (dispatch)"
        }
    }
}

// ── Correctness check ─────────────────────────────────────────

#[test]
fn bench_simd_topk_correctness_and_speed() {
    let k_values = [4, 8, 16];
    let n_values = [64, 128, 256, 512, 1024];
    let seed = 42;
    let iters = 1000;

    println!(
        "\nPlan 256 — argtopk dispatch vs scalar_heap baseline (arch = {})",
        std::env::consts::ARCH
    );
    #[cfg(target_arch = "x86_64")]
    println!(
        "note (Issue 808 T1): x86_64 dispatches AVX2 only for k <= {} — \
         larger k rows are SAME-kernel (scalar_heap vs scalar_heap); their \
         speedup column is dispatch overhead + timing noise, not a kernel contrast.",
        katgpt_rs::dash_attn::block_topk::AVX2_ARGTOPK_K_MAX
    );
    println!(
        "{:<3} {:<6} {:<24} {:>14} {:>14} {:>9}",
        "k", "n", "kernel", "argtopk ns", "scalar ns", "speedup"
    );

    for &k in &k_values {
        for &n in &n_values {
            let scores = make_scores(n, seed);

            // Verify correctness first
            let mut simd_indices = Vec::with_capacity(k);
            argtopk(&scores, k, &mut simd_indices);
            let ref_indices = argtopk_reference(&scores, k);
            assert_eq!(
                simd_indices, ref_indices,
                "SIMD mismatch at k={k}, n={n}: simd={simd_indices:?} != ref={ref_indices:?}"
            );

            // Benchmark the argtopk dispatch path (kernel per `dispatch_kernel`)
            let mut simd_indices = Vec::with_capacity(k);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                simd_indices.clear();
                argtopk(&scores, k, &mut simd_indices);
            }
            let simd_ns = start.elapsed().as_nanos() as f64 / iters as f64;

            // Benchmark scalar path
            let mut scalar_indices = Vec::with_capacity(k);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                scalar_indices.clear();
                argtopk_scalar_heap(&scores, k, &mut scalar_indices);
            }
            let scalar_ns = start.elapsed().as_nanos() as f64 / iters as f64;

            let speedup = scalar_ns / simd_ns;
            println!(
                "{:<3} {:<6} {:<24} {:>14.1} {:>14.1} {:>8.2}x",
                k,
                n,
                dispatch_kernel(k),
                simd_ns,
                scalar_ns,
                speedup
            );
        }
    }
}

// ── k=32 fallback benchmark (scalar path) ─────────────────────

#[test]
fn bench_simd_topk_k32_scalar_fallback() {
    let n_values = [64, 128, 256, 512, 1024];
    let seed = 99;
    let iters = 1000;
    let k = 32;

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  k=32 scalar fallback (selection sort) — n sweep              ║");
    println!("╠════════════════╦════════════════════════════════════════════════╣");
    println!("║       n        ║     ns/call                                    ║");
    println!("╠════════════════╬════════════════════════════════════════════════╣");

    for &n in &n_values {
        let scores = make_scores(n, seed);

        // Verify correctness
        let mut indices = Vec::with_capacity(k);
        argtopk(&scores, k, &mut indices);
        let ref_indices = argtopk_reference(&scores, k);
        assert_eq!(
            indices, ref_indices,
            "Scalar fallback mismatch at k={k}, n={n}"
        );

        // Benchmark
        let mut indices = Vec::with_capacity(k);
        let mut pairs = Vec::new();
        let start = std::time::Instant::now();
        for _ in 0..iters {
            indices.clear();
            pairs.clear();
            katgpt_rs::dash_attn::block_topk::argtopk_with_scratch(
                &scores,
                k,
                &mut indices,
                &mut pairs,
            );
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;

        println!("║ n={n:<12}║ {ns:>12.1} ns                                ║",);
    }

    println!("╚════════════════╩════════════════════════════════════════════════╝");
}

// ── Detailed sweep: fixed n=256, k sweep ──────────────────────

#[test]
fn bench_simd_topk_k_sweep_n256() {
    let k_values = [1, 2, 4, 8, 12, 16];
    let n = 256;
    let seed = 77;
    let iters = 2000;

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  n=256 — k sweep (SIMD path k≤16 vs scalar)                   ║");
    println!("╠═════════╦════════════════╦════════════════╦═════════════════════╣");
    println!("║    k    ║  SIMD (ns/call)║ Scalar(ns/call)║ Speedup             ║");
    println!("╠═════════╬════════════════╬════════════════╬═════════════════════╣");

    for &k in &k_values {
        let scores = make_scores(n, seed);

        // Correctness check
        let mut simd_indices = Vec::with_capacity(k);
        argtopk(&scores, k, &mut simd_indices);
        let ref_indices = argtopk_reference(&scores, k);
        assert_eq!(simd_indices, ref_indices, "Mismatch at k={k}");

        // SIMD benchmark
        let mut simd_indices = Vec::with_capacity(k);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            simd_indices.clear();
            argtopk(&scores, k, &mut simd_indices);
        }
        let simd_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        // Scalar benchmark
        let mut scalar_indices = Vec::with_capacity(k);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            scalar_indices.clear();
            argtopk_scalar_heap(&scores, k, &mut scalar_indices);
        }
        let scalar_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        let speedup = scalar_ns / simd_ns;
        println!(
            "║ k={k:<5}║ {simd_ns:>12.1}  ║ {scalar_ns:>12.1}  ║ {speedup:>7.2}x              ║"
        );
    }

    println!("╚═════════╩════════════════╩════════════════╩═════════════════════╝");
}

// ── Issue 808 addendum: distribution matrix + per-k crossover ────────────────
//
// The table recorded in Issue 808 was measured on ONE input distribution —
// `make_scores` above, quasi-random i.i.d. uniform via a hashed sinusoid.
// Option 4 in the issue flags exactly that: a real DashAttn block-score
// distribution may not look like it. This section measures the same
// SIMD-vs-scalar question across six score distributions and produces the
// per-k N_MIN crossover data option 1 needs (T2's measurement half).
//
// ⛍ It does NOT gate anything: the issue's own bar is that a dispatch change
// is decided by the owner on ≥ 2 microarchitectures, and this file provides
// one. The numbers land in `.benchmarks/810_argtopk_distribution_crossover.md`
// as the Raptor-Lake row of that decision table.
//
// Instrument: `tests/common/ab_timing.rs` (Issue-723 interleaved
// median-of-ratios). Arm a = `argtopk_scalar_heap` (baseline), arm b =
// `argtopk` (the SIMD dispatch — AVX2 via runtime detection on x86_64, NEON
// on aarch64), so `speedup = 1/median_ratio` and **< 1.00 is a loss** — the
// issue's convention. `[profile.release]` is `lto = "fat"` +
// `codegen-units = 1` (the Issue-723 Class-A2 folding regime): both arms
// churn the same four input positions per iteration and consume an index
// sink, and `ab_median_ratio` asserts loudly on a vanished arm.

#[path = "common/ab_timing.rs"]
mod ab_timing;

use std::hint::black_box;

/// Seeded splitmix64 — deterministic, no global RNG state (the Issue-809 class).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let v = z ^ (z >> 31);
        ((v >> 40) as f32) / (1u32 << 24) as f32
    }
}

/// Block-score distributions the routing layer actually plausibly sees.
///
/// i.i.d. shapes are distribution-free in insertion count (the streaming
/// top-k work is a function of the rank sequence), so the interesting arms
/// are the STRUCTURED ones: positional locality, early/late peak placement,
/// and cluster separation.
#[derive(Clone, Copy)]
enum Dist {
    /// i.i.d. uniform — the recorded table's family, for continuity.
    IidUniform,
    /// Gaussian-ish logits through sigmoid — clustered near 0.5 with tails.
    GaussSigmoid,
    /// AR(1) positional correlation — adjacent blocks score similarly
    /// (attention locality over contiguous token blocks).
    Locality,
    /// The top-scoring blocks sit at the START of the scan (attention sink):
    /// the heap threshold is set high immediately, later insertions rare.
    EarlyPeak,
    /// The top-scoring blocks sit at the END: maximal insertion work during
    /// the sweep.
    LatePeak,
    /// Block-sparse routing story: a small high cluster, a large low cluster.
    BimodalSparse,
}

const ALL_DISTS: [Dist; 6] = [
    Dist::IidUniform,
    Dist::GaussSigmoid,
    Dist::Locality,
    Dist::EarlyPeak,
    Dist::LatePeak,
    Dist::BimodalSparse,
];

impl Dist {
    fn name(self) -> &'static str {
        match self {
            Self::IidUniform => "iid_uniform",
            Self::GaussSigmoid => "gauss_sigmoid",
            Self::Locality => "locality",
            Self::EarlyPeak => "early_peak",
            Self::LatePeak => "late_peak",
            Self::BimodalSparse => "bimodal_sparse",
        }
    }

    /// `k` is needed so the peak arms always carry at least `k` top blocks —
    /// a peak smaller than k cannot set the threshold early and proves nothing.
    fn make(self, n: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64(seed ^ 0xD1B5_4A32_D192_ED03);
        let peak = (n / 20).max(k);
        match self {
            Self::IidUniform => (0..n).map(|_| rng.next_f32()).collect(),
            Self::GaussSigmoid => (0..n)
                .map(|_| {
                    // Irwin–Hall(3) ≈ Gaussian logit, scaled to σ≈1
                    let s = rng.next_f32() + rng.next_f32() + rng.next_f32();
                    let logit = (s - 1.5) * 4.0;
                    1.0 / (1.0 + (-logit).exp())
                })
                .collect(),
            Self::Locality => {
                let mut s = rng.next_f32();
                (0..n)
                    .map(|_| {
                        s = 0.65 * s + 0.35 * rng.next_f32();
                        s
                    })
                    .collect()
            }
            Self::EarlyPeak => (0..n)
                .map(|i| {
                    if i < peak {
                        0.85 + 0.15 * rng.next_f32()
                    } else {
                        0.35 * rng.next_f32()
                    }
                })
                .collect(),
            Self::LatePeak => (0..n)
                .map(|i| {
                    if i >= n - peak {
                        0.85 + 0.15 * rng.next_f32()
                    } else {
                        0.35 * rng.next_f32()
                    }
                })
                .collect(),
            Self::BimodalSparse => (0..n)
                .map(|_| {
                    if rng.next_f32() < 0.12 {
                        0.75 + 0.25 * rng.next_f32()
                    } else {
                        0.35 * rng.next_f32()
                    }
                })
                .collect(),
        }
    }
}

/// Per-iteration input churn: perturb four base positions by a golden-ratio
/// delta in [-1e-3, +1e-3]. Small enough to leave the distribution's shape
/// (and top-k membership, except at true near-ties) intact; large enough that
/// every iteration's input differs — the anti-hoist requirement under fat LTO.
/// Identical work in both arms, so it cancels in the ratio.
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

struct CellResult {
    scalar_ns: f64,
    simd_ns: f64,
    /// 1 / median per-round ratio (scalar ÷ SIMD) — the issue's speedup.
    speedup: f64,
    /// Smallest and largest per-round speedup — the box-agreement band.
    speedup_lo: f64,
    speedup_hi: f64,
}

/// One interleaved A/B measurement of `argtopk` (SIMD dispatch) vs
/// `argtopk_scalar_heap`, with per-cell correctness verification against a
/// full-sort reference first.
fn measure_cell(base: &[f32], k: usize, rounds: usize) -> CellResult {
    let n = base.len();
    let iters = ((4_000_000.0 / (n.max(1) as f64 * 1.5)) as usize).clamp(256, 50_000);

    // Correctness on the base scores before any timing.
    let mut idx_simd = Vec::with_capacity(k);
    let mut idx_scalar = Vec::with_capacity(k);
    argtopk(base, k, &mut idx_simd);
    argtopk_scalar_heap(base, k, &mut idx_scalar);
    let ref_idx = argtopk_reference(base, k);
    assert_eq!(idx_simd, ref_idx, "SIMD mismatch at k={k}, n={n}");
    assert_eq!(idx_scalar, ref_idx, "scalar-heap mismatch at k={k}, n={n}");

    let mut scratch_a = base.to_vec();
    let mut idx_a = Vec::with_capacity(k);
    let mut sink_a = 0u64;
    let mut scratch_b = base.to_vec();
    let mut idx_b = Vec::with_capacity(k);
    let mut sink_b = 0u64;

    let ab = ab_timing::ab_median_ratio(
        rounds,
        iters,
        iters,
        |i| {
            churn(&mut scratch_a, i, base);
            idx_a.clear();
            argtopk_scalar_heap(&scratch_a, k, &mut idx_a);
            sink_a = sink_a.wrapping_add(idx_a[0] as u64).rotate_left(1);
        },
        |i| {
            churn(&mut scratch_b, i, base);
            idx_b.clear();
            argtopk(&scratch_b, k, &mut idx_b);
            sink_b = sink_b.wrapping_add(idx_b[0] as u64).rotate_left(1);
        },
    );
    black_box((sink_a, sink_b));

    CellResult {
        scalar_ns: ab.a_ns_per_iter(),
        simd_ns: ab.b_ns_per_iter(),
        speedup: 1.0 / ab.median,
        speedup_lo: 1.0 / ab.max(),
        speedup_hi: 1.0 / ab.min(),
    }
}

/// The full distribution × (k, n) grid — does the AVX2/NEON dispatch still
/// lose where the recorded table says it does, once the input looks like
/// block scores instead of i.i.d. uniform?
///
/// ⚠ POST-T1 the answer differs by architecture, and reading one grid for the
/// other is the trap. On **aarch64** this is still the question it was: NEON
/// dispatches the whole k ≤ 16 range and the grid measures a real A/B. On
/// **`x86_64`** Issue 808 T1 narrowed the dispatch to k ≤ 4, so the k ∈ {8, 16}
/// rows now compare `argtopk_scalar_heap` against ITSELF and read ~1.00 —
/// that is the fix, not a measurement of the kernel. The kernel's own numbers
/// above the bound are the ones recorded in Bench 810; this grid can no longer
/// reach them on `x86_64`. `bench_simd_topk_issue808_t2_above_bound_is_not_a_loss`
/// is the assertion built on that ~1.00.
#[test]
fn bench_simd_topk_issue808_distribution_matrix() {
    let k_values = [1usize, 2, 4, 8, 16];
    let n_values = [64usize, 128, 256, 512, 1024];
    let rounds = 9;

    println!("\n== Issue 808 distribution matrix (speedup = scalar/SIMD, <1.00 is a loss) ==");
    println!(
        "{:<15} {:>4} {:>6} {:>12} {:>12} {:>9}  {:>17}",
        "distribution", "k", "n", "scalar ns", "simd ns", "speedup", "round band"
    );

    for &dist in &ALL_DISTS {
        for &k in &k_values {
            for &n in &n_values {
                let base = dist.make(n, k, 0x5EED_600D + (n as u64) * 7919);
                let r = measure_cell(&base, k, rounds);
                println!(
                    "{:<15} {:>4} {:>6} {:>12.1} {:>12.1} {:>8.2}x  {:>6.2}..{:<6.2}",
                    dist.name(),
                    k,
                    n,
                    r.scalar_ns,
                    r.simd_ns,
                    r.speedup,
                    r.speedup_lo,
                    r.speedup_hi,
                );
            }
        }
    }
}

/// The per-k `N_MIN` crossover sweep — option 1's data half. Sweeps n upward at
/// each k until the SIMD dispatch reaches parity (speedup ≥ 1.00), on two
/// distribution bookends (i.i.d. and the locality shape a real block-score
/// stream plausibly has). One box, one microarchitecture — the Raptor-Lake
/// row of the table the owner's T2 needs before any dispatch change.
///
/// ⚠ The owner picked option 2, not option 1 (Issue 808 T1, 2026-09-17), so on
/// `x86_64` this sweep no longer measures what it was built to measure above
/// k = 4: both arms are the same code there and it crosses at the first n.
/// KEPT anyway, and deliberately — it is the instrument that would have to
/// re-run, on a second microarchitecture, before `AVX2_ARGTOPK_K_MAX` could be
/// widened toward option 1. Deleting it would delete the reopen path. Its
/// aarch64 run is unaffected.
#[test]
fn bench_simd_topk_issue808_crossover_nmin() {
    let k_values = [2usize, 4, 8, 12, 16];
    let n_sweep = [
        64usize, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
    ];
    let rounds = 9;

    println!("\n== Issue 808 per-k crossover (first n with speedup ≥ 1.00) ==");
    println!(
        "{:<15} {:>4} {:>6} {:>12} {:>12} {:>9}  {:>17}",
        "distribution", "k", "n", "scalar ns", "simd ns", "speedup", "round band"
    );

    for &dist in &[Dist::IidUniform, Dist::Locality] {
        for &k in &k_values {
            let mut n_min = None;
            let mut n_min_05 = None;
            for &n in &n_sweep {
                let base = dist.make(n, k, 0xC0FF_EE01 + (n as u64) * 104_729);
                let r = measure_cell(&base, k, rounds);
                println!(
                    "{:<15} {:>4} {:>6} {:>12.1} {:>12.1} {:>8.2}x  {:>6.2}..{:<6.2}",
                    dist.name(),
                    k,
                    n,
                    r.scalar_ns,
                    r.simd_ns,
                    r.speedup,
                    r.speedup_lo,
                    r.speedup_hi,
                );
                if n_min_05.is_none() && r.speedup >= 1.05 {
                    n_min_05 = Some(n);
                }
                if n_min.is_none() && r.speedup >= 1.00 {
                    n_min = Some(n);
                    break;
                }
            }
            match (n_min, n_min_05) {
                (Some(n1), Some(n2)) => println!(
                    "--> N_MIN[{}, k={k}] = {n1} (speedup ≥ 1.05 at {n2})",
                    dist.name()
                ),
                (Some(n1), None) => {
                    println!(
                        "--> N_MIN[{}, k={k}] = {n1} (no ≥1.05 point in sweep)",
                        dist.name()
                    );
                }
                (None, _) => println!(
                    "--> N_MIN[{}, k={k}] = >{} (no crossing in sweep)",
                    dist.name(),
                    n_sweep[n_sweep.len() - 1]
                ),
            }
        }
    }
}

// ── Issue 808 T1/T2 — the dispatch bound, asserted ─────────────────────────
//
// T1 landed option 2 (owner's call, 2026-09-17): on x86_64 the AVX2 kernel is
// dispatched only for `k <= AVX2_ARGTOPK_K_MAX` (= 4), and larger k takes
// `argtopk_scalar_heap`. The two tests above REPORT the grid; this one is the
// timing half of T2's "the next regression is a test failure rather than a
// table nobody reads".
//
// ⚠ The bar is deliberately far from the thing it detects. The defect was
// 0.41–0.72× on `late_peak` at k = 8; post-dispatch both arms reach the same
// kernel and the floor sits at 0.85.
//
// ⚠ The expected value is **~0.95, not 1.00**, and that is structural rather
// than noise: arm a calls `argtopk_scalar_heap` directly, while arm b goes
// through `argtopk` → `argtopk_with_scratch` → `argtopk_simd` and pays a
// `is_x86_feature_detected!` probe and two extra calls before landing on the
// same code. Measured over three consecutive runs on a quiet 4090: worst cell
// 0.95 / 0.94 / 0.94 — the worst VALUE is stable to 0.01 even though which
// cell is worst moves, which is what near-parity noise looks like. Do not
// "fix" a 0.95 by raising the floor toward 1.00; that would gate the dispatch
// overhead, not the kernel.
// That margin is not timidity — this repo has measured perf bars producing four
// different failing sets across four runs of ONE commit (Bench 806 T7), and a
// bar that cries wolf is a bar nobody keeps. The load-immune half of the same
// assertion is `test_avx2_argtopk_dispatch_bound_is_pinned` in the unit tests,
// which reds on a widened bound without timing anything at all.
//
// ⛔ A red here is a BOX-CONDITIONS question first (Bench 806 T7's discipline):
// re-run it alone before calling it a regression.

/// Issue 808 T2 — above the dispatch bound, the recorded loss must be GONE.
#[cfg(target_arch = "x86_64")]
#[test]
fn bench_simd_topk_issue808_t2_above_bound_is_not_a_loss() {
    use katgpt_rs::dash_attn::block_topk::AVX2_ARGTOPK_K_MAX;

    // The floor. See the module comment above for why it is this far out.
    const FLOOR: f64 = 0.85;

    // Exactly the cells Bench 810 measured the loss in: k above the bound, at
    // the sizes where `late_peak` reached 0.41–0.72×.
    let k_values: Vec<usize> = [8usize, 16]
        .into_iter()
        .filter(|&k| k > AVX2_ARGTOPK_K_MAX)
        .collect();
    assert!(
        !k_values.is_empty(),
        "AVX2_ARGTOPK_K_MAX = {AVX2_ARGTOPK_K_MAX} is at or above every k this \
         test measures, so it asserts NOTHING. Widen the k list to span the \
         bound, or this is a green zero over the exact arm Issue 808 is about."
    );
    let n_values = [64usize, 128, 256, 512];
    let rounds = 9;

    println!("\n== Issue 808 T2 — above the dispatch bound (floor {FLOOR:.2}x) ==");
    let mut worst: Option<(String, usize, usize, f64)> = None;

    for &dist in &ALL_DISTS {
        for &k in &k_values {
            for &n in &n_values {
                let base = dist.make(n, k, 0x808_7A2 + (n as u64) * 7919);
                let r = measure_cell(&base, k, rounds);
                println!(
                    "{:<15} k={:<3} n={:<6} {:>8.2}x  band {:>5.2}..{:<5.2}",
                    dist.name(),
                    k,
                    n,
                    r.speedup,
                    r.speedup_lo,
                    r.speedup_hi,
                );
                if worst.as_ref().is_none_or(|w| r.speedup < w.3) {
                    worst = Some((dist.name().to_string(), k, n, r.speedup));
                }
            }
        }
    }

    let (d, k, n, s) = worst.expect("the grid is non-empty");
    println!("--> worst cell: {d} k={k} n={n} at {s:.2}x (floor {FLOOR:.2}x)");
    assert!(
        s >= FLOOR,
        "above the dispatch bound the AVX2 arm should not run at all, so both \
         arms are the same code and the ratio should be ~1.00 — measured \
         {s:.2}x at {d} k={k} n={n}, below the {FLOOR:.2}x floor. Either \
         AVX2_ARGTOPK_K_MAX was widened past {AVX2_ARGTOPK_K_MAX} without \
         Issue 808 T2's >=2-microarchitecture measurement, or this box is \
         loaded. RE-RUN THIS TEST ALONE before calling it a regression."
    );
}

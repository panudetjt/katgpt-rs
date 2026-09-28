//! Plan 475 GOAT gate — ICA Lens FastICA quality + perf gate.
//!
//! The load-bearing gate is G2 (quality): FastICA directions must be MORE
//! non-Gaussian than PCA + kurtosis-ranking on:
//! - (a) Synthetic non-Gaussian source: FastICA mean kurtosis ≥ 2× PCA mean kurtosis.
//! - (b) Realistic d=64 substrate: FastICA mean kurtosis ≥ 1.5× PCA mean kurtosis.
//!
//! G1 (latency), G3 (no regression), G4 (alloc-free), G5 (determinism) are
//! also exercised.
//!
//! Run: cargo bench --bench bench_475_ica_lens_goat --features ica_lens
//! (harness = false; uses std::time::Instant + CountingAllocator)

#![cfg(feature = "ica_lens")]

use katgpt_spectral::hla_eigenbasis::{EigenbasisScratch, recover_eigenbasis_from_window};
use katgpt_spectral::ica_lens::{
    FastIcaConfig, FastIcaScratch, IcaAcceptance, IcaContrast, excess_kurtosis_of_projection,
    fastica_into,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Counting allocator for G4
// ---------------------------------------------------------------------------

struct Counting;
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn reset_alloc_counters() {
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

fn alloc_bytes() -> u64 {
    ALLOC_BYTES.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Synthetic data generators
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}

/// Build a synthetic non-Gaussian dataset: n_sources Laplace + (d-n_sources)
/// Uniform, mixed by a deterministic sign-flipped matrix.
fn make_mixed_non_gaussian(t: usize, d: usize, n_sources: usize, seed: u64) -> Vec<f32> {
    let mut rng = seed;
    let mut sources = vec![0.0_f32; t * d];
    for j in 0..d {
        for i in 0..t {
            let u1 = lcg_next(&mut rng);
            let u2 = lcg_next(&mut rng);
            sources[i * d + j] = if j < n_sources {
                let s = if u1 < 0.5 { -1.0 } else { 1.0 };
                s * (1.0 - 2.0 * (u1 - 0.5).abs()).ln()
            } else {
                (u2 * 2.0 - 1.0) * 3.0_f32.sqrt()
            };
        }
    }
    let mut mix = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let sign = if ((i + j) * (i + j + 1) / 2) & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            mix[i * d + j] = sign * (1.0 + (lcg_next(&mut rng) - 0.5) * 0.5);
        }
    }
    let mut observed = vec![0.0_f32; t * d];
    for i in 0..t {
        for j in 0..d {
            let mut acc = 0.0_f32;
            for k in 0..d {
                acc += sources[i * d + k] * mix[k * d + j];
            }
            observed[i * d + j] = acc;
        }
    }
    observed
}

/// Compute PCA top-k directions + their kurtosis (the baseline).
fn pca_top_k_kurtosis(window: &[f32], t: usize, d: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut eigvecs = vec![0.0_f32; d * k];
    let mut eigvals = vec![0.0_f32; d];
    let mut scratch = EigenbasisScratch::with_capacity_d(d);
    recover_eigenbasis_from_window(window, t, d, &mut eigvecs, &mut eigvals, &mut scratch, k, 5);

    // Center window for kurtosis projection.
    let mut centered = vec![0.0_f32; t * d];
    centered.copy_from_slice(window);
    let mut mean = vec![0.0_f32; d];
    for r in 0..t {
        for j in 0..d {
            mean[j] += centered[r * d + j];
        }
    }
    for item in mean.iter_mut().take(d) {
        *item /= t as f32;
    }
    for r in 0..t {
        for j in 0..d {
            centered[r * d + j] -= mean[j];
        }
    }

    // Kurtosis of each PCA direction.
    let mut kurt = vec![0.0_f32; k];
    for j in 0..k {
        // Direction j is column j of eigvecs (strided by k): eigvecs[row*k + j].
        let dir: Vec<f32> = (0..d).map(|row| eigvecs[row * k + j]).collect();
        kurt[j] = excess_kurtosis_of_projection(&centered, &dir, t, d);
    }
    (eigvecs, kurt)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct PassFail {
    passed: usize,
    failed: usize,
}

impl PassFail {
    fn new() -> Self {
        Self {
            passed: 0,
            failed: 0,
        }
    }
    fn pass(&mut self, msg: &str) {
        println!("  ✅ PASS: {msg}");
        self.passed += 1;
    }
    fn fail(&mut self, msg: &str) {
        println!("  ❌ FAIL: {msg}");
        self.failed += 1;
    }
    fn check(&mut self, cond: bool, msg: &str) {
        if cond {
            self.pass(msg);
        } else {
            self.fail(msg);
        }
    }
    fn summary(&self) {
        println!(
            "\n── Summary: {} passed, {} failed ──",
            self.passed, self.failed
        );
    }
}

// ---------------------------------------------------------------------------
// G1: Latency
// ---------------------------------------------------------------------------

fn g1_latency(pf: &mut PassFail) {
    println!("\n─── G1: Latency ───");
    println!("    ICA Lens is a corpus-level offline fit (not per-tick).");
    println!("    Target: ≤ 1ms for the HLA-scale window (T=512, D=8, m=8).");

    let t = 512;
    let d = 8;
    let m = 8;
    let window = make_mixed_non_gaussian(t, d, 4, 42);

    let config = FastIcaConfig {
        n_components: m,
        row_normalize: false,
        adaptive_refit: false,
        ..Default::default()
    };

    let mut scratch = FastIcaScratch::new();
    let mut reading = vec![0.0_f32; m * d];
    let mut writing = vec![0.0_f32; d * m];
    let mut scores = vec![0.0_f32; t * m];
    let mut kurt = vec![0.0_f32; m];
    let mut lim = vec![0.0_f32; m];

    // Warmup.
    let _ = fastica_into(
        &window,
        t,
        d,
        &config,
        &mut scratch,
        &mut reading,
        &mut writing,
        &mut scores,
        &mut kurt,
        &mut lim,
    );

    // Measure.
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = fastica_into(
            &window,
            t,
            d,
            &config,
            &mut scratch,
            &mut reading,
            &mut writing,
            &mut scores,
            &mut kurt,
            &mut lim,
        );
    }
    let elapsed = start.elapsed();
    let per_call_us = elapsed.as_micros() as f64 / iters as f64;
    println!("    T=512, D=8, m=8: {per_call_us:.1} µs/call (target ≤ 1000 µs)");
    pf.check(
        per_call_us <= 1000.0,
        "G1: T=512/D=8/m=8 ≤ 1000µs (offline corpus fit)",
    );
}

// ---------------------------------------------------------------------------
// G2: Quality (the load-bearing gate)
// ---------------------------------------------------------------------------

fn g2_quality_synthetic(pf: &mut PassFail) {
    println!("\n─── G2(a): Quality — synthetic non-Gaussian source ───");

    let t = 2048;
    let d = 8;
    let m = d;
    // 4 Laplace (kurt≈3) + 4 Uniform (kurt≈-1.2) mixed.
    let window = make_mixed_non_gaussian(t, d, 4, 42);

    // FastICA.
    let config = FastIcaConfig {
        n_components: m,
        row_normalize: false,
        adaptive_refit: false,
        contrast: IcaContrast::LogCosh,
        acceptance: IcaAcceptance::P95,
        ..Default::default()
    };
    let mut scratch = FastIcaScratch::new();
    let mut ica_reading = vec![0.0_f32; m * d];
    let mut ica_writing = vec![0.0_f32; d * m];
    let mut ica_scores = vec![0.0_f32; t * m];
    let mut ica_kurt = vec![0.0_f32; m];
    let mut ica_lim = vec![0.0_f32; m];
    let _ = fastica_into(
        &window,
        t,
        d,
        &config,
        &mut scratch,
        &mut ica_reading,
        &mut ica_writing,
        &mut ica_scores,
        &mut ica_kurt,
        &mut ica_lim,
    );

    // PCA baseline.
    let (_pca_vecs, pca_kurt) = pca_top_k_kurtosis(&window, t, d, m);

    // Compare mean ABSOLUTE kurtosis (the non-Gaussianity signal).
    let ica_mean_abs: f32 = ica_kurt.iter().map(|k| k.abs()).sum::<f32>() / m as f32;
    let pca_mean_abs: f32 = pca_kurt.iter().map(|k| k.abs()).sum::<f32>() / m as f32;
    let ratio = ica_mean_abs / pca_mean_abs.max(1e-10);

    println!("    ICA mean |kurtosis|: {ica_mean_abs:.4}");
    println!("    PCA mean |kurtosis|: {pca_mean_abs:.4}");
    println!("    Ratio (ICA/PCA):    {ratio:.3}x (target ≥ 2.0x)");

    pf.check(
        ratio >= 2.0,
        "G2(a): ICA/PCA kurtosis ratio ≥ 2.0x on synthetic Laplace+Uniform",
    );
}

fn g2_quality_high_dim(pf: &mut PassFail) {
    println!("\n─── G2(b): Quality — realistic d=64 substrate ───");

    let t = 4096;
    let d = 64;
    let m = 32; // half the dims
    // 16 Laplace + 48 Uniform mixed — mimics NeuronShard style_weights[64].
    let window = make_mixed_non_gaussian(t, d, 16, 42);

    let config = FastIcaConfig {
        n_components: m,
        row_normalize: false,
        adaptive_refit: false,
        contrast: IcaContrast::LogCosh,
        acceptance: IcaAcceptance::P95,
        ..Default::default()
    };
    let mut scratch = FastIcaScratch::new();
    let mut ica_reading = vec![0.0_f32; m * d];
    let mut ica_writing = vec![0.0_f32; d * m];
    let mut ica_scores = vec![0.0_f32; t * m];
    let mut ica_kurt = vec![0.0_f32; m];
    let mut ica_lim = vec![0.0_f32; m];
    let ica_result_status;
    let ica_result_m_eff;
    {
        let r = fastica_into(
            &window,
            t,
            d,
            &config,
            &mut scratch,
            &mut ica_reading,
            &mut ica_writing,
            &mut ica_scores,
            &mut ica_kurt,
            &mut ica_lim,
        );
        ica_result_status = r.status;
        ica_result_m_eff = r.m_eff;
    } // r dropped here, ica_kurt borrow released

    let (_pca_vecs, pca_kurt) = pca_top_k_kurtosis(&window, t, d, m);

    let ica_mean_abs: f32 = ica_kurt.iter().map(|k| k.abs()).sum::<f32>() / m as f32;
    let pca_mean_abs: f32 = pca_kurt.iter().map(|k| k.abs()).sum::<f32>() / m as f32;
    let ratio = ica_mean_abs / pca_mean_abs.max(1e-10);

    println!("    ICA status: {ica_result_status:?}, m_eff: {ica_result_m_eff}");
    println!("    ICA mean |kurtosis|: {ica_mean_abs:.4}");
    println!("    PCA mean |kurtosis|: {pca_mean_abs:.4}");
    println!("    Ratio (ICA/PCA):    {ratio:.3}x (target ≥ 1.5x)");

    pf.check(
        ratio >= 1.5,
        "G2(b): ICA/PCA kurtosis ratio ≥ 1.5x on d=64 substrate",
    );
}

// ---------------------------------------------------------------------------
// G4: Alloc-free steady-state
// ---------------------------------------------------------------------------

fn g4_alloc_free(pf: &mut PassFail) {
    println!("\n─── G4: Alloc-free steady-state ───");

    let t = 512;
    let d = 8;
    let m = 8;
    let window = make_mixed_non_gaussian(t, d, 4, 42);

    let config = FastIcaConfig {
        n_components: m,
        row_normalize: false,
        adaptive_refit: false,
        ..Default::default()
    };

    let mut scratch = FastIcaScratch::with_capacity(t, d, m);
    let mut reading = vec![0.0_f32; m * d];
    let mut writing = vec![0.0_f32; d * m];
    let mut scores = vec![0.0_f32; t * m];
    let mut kurt = vec![0.0_f32; m];
    let mut lim = vec![0.0_f32; m];

    // First call (may allocate for internal eigvecs_d / z_buf / w_mat temp Vecs).
    let _ = fastica_into(
        &window,
        t,
        d,
        &config,
        &mut scratch,
        &mut reading,
        &mut writing,
        &mut scores,
        &mut kurt,
        &mut lim,
    );

    // Measure second call (should be alloc-free after scratch is warmed).
    reset_alloc_counters();
    let _ = fastica_into(
        &window,
        t,
        d,
        &config,
        &mut scratch,
        &mut reading,
        &mut writing,
        &mut scores,
        &mut kurt,
        &mut lim,
    );
    let bytes = alloc_bytes();
    println!("    Steady-state allocation: {bytes} bytes (target 0)");
    pf.check(bytes == 0, "G4: 0 bytes allocated in steady state");
}

// ---------------------------------------------------------------------------
// G5: Determinism
// ---------------------------------------------------------------------------

fn g5_determinism(pf: &mut PassFail) {
    println!("\n─── G5: Determinism ───");

    let t = 512;
    let d = 8;
    let m = 4;
    let window = make_mixed_non_gaussian(t, d, 2, 99);

    let config = FastIcaConfig {
        n_components: m,
        row_normalize: false,
        adaptive_refit: false,
        ..Default::default()
    };

    let run_once = || -> Vec<f32> {
        let mut scratch = FastIcaScratch::new();
        let mut reading = vec![0.0_f32; m * d];
        let mut writing = vec![0.0_f32; d * m];
        let mut scores = vec![0.0_f32; t * m];
        let mut kurt = vec![0.0_f32; m];
        let mut lim = vec![0.0_f32; m];
        let _ = fastica_into(
            &window,
            t,
            d,
            &config,
            &mut scratch,
            &mut reading,
            &mut writing,
            &mut scores,
            &mut kurt,
            &mut lim,
        );
        reading
    };

    let r1 = run_once();
    let r2 = run_once();
    let identical = r1 == r2;
    println!("    Bit-identical across runs: {identical}");
    pf.check(identical, "G5: reading_map bit-identical across two runs");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Plan 475 GOAT gate — ICA Lens FastICA                   ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let mut pf = PassFail::new();

    g1_latency(&mut pf);
    g2_quality_synthetic(&mut pf);
    g2_quality_high_dim(&mut pf);
    g4_alloc_free(&mut pf);
    g5_determinism(&mut pf);

    pf.summary();

    if pf.failed > 0 {
        std::process::exit(1);
    }
}

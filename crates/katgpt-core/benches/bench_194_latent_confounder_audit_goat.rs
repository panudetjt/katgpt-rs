//! Issue 194 T4 G2 — Latent Confounder Audit perf bench.
//!
//! Measures `audit_confounders` latency on three realistic encoder dims:
//!
//! - **G2 target**: HLA scale `d=8` < 1 µs per audit call (the gate target —
//!   sub-µs per check at the HLA-scale encoder that's the primary audit target).
//! - Sweep: `d=8` (HLA), `d=32` (diagnostic scale), `d=64` (style_weights).
//!
//! Convention: `std::time::Instant` + `harness = false` (mirrors
//! `bench_342_latent_trajectory_geometry_goat.rs`, no Criterion dev-dep).
//!
//! Run:
//! ```bash
//! cargo run --release --bench bench_194_latent_confounder_audit_goat \
//!     --features latent_confounder_audit
//! ```

#![cfg(feature = "latent_confounder_audit")]
#![allow(clippy::type_complexity)] // Pair-of-pairs type is inherent to the API.

use katgpt_core::latent_confounder_audit::{AuditScratch, audit_confounders};
use std::time::{Duration, Instant};

// ─── Config ────────────────────────────────────────────────────────────────

/// Encoder dims to sweep.
/// - 8: HLA per-NPC affect (the primary audit target — most consumers run here)
/// - 32: typical diagnostic / cluster-PCA scale
/// - 64: NeuronShard style_weights scale
const DIMS: &[(usize, &str)] = &[
    (8, "HLA (gate target)"),
    (32, "diag"),
    (64, "style_weights"),
];

/// Warmup iterations (untimed).
const WARMUP: usize = 50;

/// Number of timed runs to take the median over.
const TIMED_RUNS: usize = 200;

// ─── Deterministic LCG ─────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

/// Build a deterministic encoder for dim `d`: the mean-subtracted displacement
/// (a clean encoder that exercises the same code paths as the real audit
/// consumers). Returns a closure capturing no state — trivially inlinable.
#[allow(clippy::needless_range_loop)] // both a[i] and b[i] are read by index.
fn make_clean_encoder(d: usize) -> impl Fn(&[f32], &[f32], &mut [f32]) {
    move |a: &[f32], b: &[f32], out: &mut [f32]| {
        let mut mean = 0.0_f32;
        for i in 0..d {
            out[i] = b[i] - a[i];
            mean += out[i];
        }
        mean /= d as f32;
        for v in out.iter_mut().take(d) {
            *v -= mean;
        }
    }
}

/// Build the audit input pairs once, outside the timed loop.
struct AuditInputs {
    zero_pairs: Vec<([f32; 64], [f32; 64])>,
    shift_pairs: Vec<([f32; 64], [f32; 64])>,
    ordinary_pairs: Vec<([f32; 64], [f32; 64])>,
    same_action_pairs: Vec<(([f32; 64], [f32; 64]), ([f32; 64], [f32; 64]))>,
    diff_action_pairs: Vec<(([f32; 64], [f32; 64]), ([f32; 64], [f32; 64]))>,
}

impl AuditInputs {
    #[allow(clippy::needless_range_loop)] // bench setup; explicit indexing reads clearer.
    fn build(d: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);

        // Zero-transition: (x, x) — second is a clone of first.
        let mut zero_pairs: Vec<([f32; 64], [f32; 64])> = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut a = [0.0_f32; 64];
            for i in 0..d {
                a[i] = rng.next_f32();
            }
            zero_pairs.push((a, a));
        }
        // Shift: (x, x + constant_offset). The clean encoder is invariant.
        let mut shift_pairs: Vec<([f32; 64], [f32; 64])> = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut a = [0.0_f32; 64];
            let mut b = [0.0_f32; 64];
            for i in 0..d {
                a[i] = rng.next_f32();
                b[i] = a[i] + 5.0; // constant nuisance offset
            }
            shift_pairs.push((a, b));
        }
        // Ordinary: (x, x') — typical transitions for normalization.
        let mut ordinary_pairs: Vec<([f32; 64], [f32; 64])> = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut a = [0.0_f32; 64];
            let mut b = [0.0_f32; 64];
            for i in 0..d {
                a[i] = rng.next_f32();
                b[i] = rng.next_f32();
            }
            ordinary_pairs.push((a, b));
        }
        // Same-action / diff-context: same displacement, different starting x.
        let mut same_action_pairs: Vec<_> = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut a1 = [0.0_f32; 64];
            let mut b1 = [0.0_f32; 64];
            let mut a2 = [0.0_f32; 64];
            let mut b2 = [0.0_f32; 64];
            // Shared displacement.
            let mut disp = [0.0_f32; 64];
            for i in 0..d {
                a1[i] = rng.next_f32();
                a2[i] = rng.next_f32();
                disp[i] = rng.next_f32();
                b1[i] = a1[i] + disp[i];
                b2[i] = a2[i] + disp[i];
            }
            same_action_pairs.push(((a1, b1), (a2, b2)));
        }
        // Diff-action / same-context: same x, different displacement.
        let mut diff_action_pairs: Vec<_> = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut a = [0.0_f32; 64];
            let mut b1 = [0.0_f32; 64];
            let mut b2 = [0.0_f32; 64];
            for i in 0..d {
                a[i] = rng.next_f32();
                b1[i] = a[i] + rng.next_f32();
                b2[i] = a[i] + rng.next_f32();
            }
            diff_action_pairs.push(((a, b1), (a, b2)));
        }

        Self {
            zero_pairs,
            shift_pairs,
            ordinary_pairs,
            same_action_pairs,
            diff_action_pairs,
        }
    }

    /// Build the slice-view arrays the audit accepts. These are cheap slice
    /// borrows over the owned arrays — the timed loop only measures the audit
    /// call, not these allocations.
    fn as_slices(&self) -> AuditSlices<'_> {
        let zero: Vec<(&[f32], &[f32])> = self
            .zero_pairs
            .iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();
        let shift: Vec<(&[f32], &[f32])> = self
            .shift_pairs
            .iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();
        let ordinary: Vec<(&[f32], &[f32])> = self
            .ordinary_pairs
            .iter()
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
            .collect();
        let same: Vec<((&[f32], &[f32]), (&[f32], &[f32]))> = self
            .same_action_pairs
            .iter()
            .map(|((a1, b1), (a2, b2))| {
                (
                    (a1.as_slice(), b1.as_slice()),
                    (a2.as_slice(), b2.as_slice()),
                )
            })
            .collect();
        let diff: Vec<((&[f32], &[f32]), (&[f32], &[f32]))> = self
            .diff_action_pairs
            .iter()
            .map(|((a1, b1), (a2, b2))| {
                (
                    (a1.as_slice(), b1.as_slice()),
                    (a2.as_slice(), b2.as_slice()),
                )
            })
            .collect();
        AuditSlices {
            zero,
            shift,
            ordinary,
            same,
            diff,
        }
    }
}

struct AuditSlices<'a> {
    zero: Vec<(&'a [f32], &'a [f32])>,
    shift: Vec<(&'a [f32], &'a [f32])>,
    ordinary: Vec<(&'a [f32], &'a [f32])>,
    same: Vec<((&'a [f32], &'a [f32]), (&'a [f32], &'a [f32]))>,
    diff: Vec<((&'a [f32], &'a [f32]), (&'a [f32], &'a [f32]))>,
}

/// Measure `audit_confounders` median latency. Inputs are pre-built; only
/// the audit call is timed.
fn bench_audit<E>(encoder: &E, slices: &AuditSlices<'_>, scratch: &mut AuditScratch) -> Duration
where
    E: Fn(&[f32], &[f32], &mut [f32]),
{
    // Warmup.
    for _ in 0..WARMUP {
        let _ = audit_confounders(
            encoder,
            &slices.zero,
            &slices.shift,
            &slices.ordinary,
            &slices.same,
            &slices.diff,
            scratch,
        );
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let r = audit_confounders(
            encoder,
            &slices.zero,
            &slices.shift,
            &slices.ordinary,
            &slices.same,
            &slices.diff,
            scratch,
        );
        samples.push(t0.elapsed());
        // Prevent the compiler from eliding the call.
        if r.normalization_denominator.is_nan() {
            std::process::abort();
        }
    }
    samples.sort();
    samples[TIMED_RUNS / 2]
}

fn format_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns:>5} ns")
    } else if ns < 1_000_000 {
        format!("{:>5.2} µs", ns as f64 / 1_000.0)
    } else {
        format!("{:>5.2} ms", ns as f64 / 1_000_000.0)
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Issue 194 — Latent Confounder Audit GOAT Gate (G2 perf)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Config: {TIMED_RUNS} timed runs (median), {WARMUP} warmup, seed=42");
    println!();

    // ── G2: audit_confounders latency ──────────────────────────────────
    println!("── G2: audit_confounders latency ──────────────────────────────");
    println!("{:>22}  {:>6}  {:>14}", "encoder_dim", "label", "median");
    println!("{}", "-".repeat(50));

    const G2_TARGET_NS: u64 = 1_000; // 1 µs per audit call at the gate dim (HLA d=8).
    let mut g2_target_passes = false;

    for &(dim, label) in DIMS {
        let inputs = AuditInputs::build(dim, 42);
        let slices = inputs.as_slices();
        let encoder = make_clean_encoder(dim);
        let mut scratch = AuditScratch::new(dim);
        let dur = bench_audit(&encoder, &slices, &mut scratch);
        // Re-compute once for the output row.
        let r = audit_confounders(
            &encoder,
            &slices.zero,
            &slices.shift,
            &slices.ordinary,
            &slices.same,
            &slices.diff,
            &mut scratch,
        );
        println!(
            "{:>22}  {:>6}  {:>14}  (R_0={:.3e}, R_sh={:.3e}, L={:.3e})",
            dim,
            label,
            format_duration(dur),
            r.zero_transition_response,
            r.shift_invariance_response,
            r.shortcut_leakage,
        );
        if dim == 8 && dur.as_nanos() as u64 <= G2_TARGET_NS {
            g2_target_passes = true;
        }
    }

    println!();
    println!(
        "G2 audit_confounders (HLA d=8 <= {}):  {}",
        format_duration(Duration::from_nanos(G2_TARGET_NS)),
        if g2_target_passes { "PASS" } else { "FAIL" }
    );
    println!();

    // ── Verdict ─────────────────────────────────────────────────────────
    println!("──────────────────────────────────────────────────────────────");
    println!(
        "Verdict: {}",
        if g2_target_passes {
            "G2 perf PASS — primitive meets the <1µs target at the gate dim."
        } else {
            "G2 perf FAIL — primitive exceeds the <1µs budget at the gate dim."
        }
    );
    if !g2_target_passes {
        std::process::exit(1);
    }
}

//! Bench 802 — NaN-safe bounded top-K JSD kernel gate (Issue 802 item 2).
//!
//! G1 known-answer gates live in the module's #[cfg(test)] self-tests
//! (disjoint supports → exactly ln 2, identical → exactly 0, the KL-trap
//! vectors, degenerate sides, scale-invariance, ~10⁴-pair bound sweep).
//! THIS bench is the G2 throughput axis + the G4 zero-alloc witness:
//!
//! - **kernel k∈{8,64,1024}** — the INTO form on caller-owned scratch
//!   (allocated once, outside the timed loop).
//! - **select-only k=64** — breakdown arm: iota fill + partial selection +
//!   top-K sum, NO union walk. The difference kernel−select-only is the
//!   bounded sum phase the restriction buys.
//! - **full-support** — the same JSD formula with no restriction (K=len
//!   semantics without selection overhead), the reference for the × column.
//! - **convenience k=64** — the allocating `jsd_topk` form, so the cost of
//!   two `Vec`s per call is on the record.
//!
//! # Honesty notes (Bench 800 lesson: no promised wins)
//!
//! The claim is BOUNDED WORK, not raw speed: the kernel pays O(n) iota +
//! O(n) introselect per side (that dominates at vocab scale), and the
//! restriction bounds only the SUM phase to ≤ 2k entries × ≤ 3 `f32::ln`.
//! There is NO explicit SIMD — `f32::ln` is the scalar libm call and the
//! iota fill is the only lane the auto-vectorizer can plausibly take;
//! numbers below are as measured on this box, nothing extrapolated.
//!
//! Convention: `std::time::Instant` + `harness = false` (the bench_801 /
//! bench_342 / bench_324 house pattern).
//!
//! Run:
//! ```bash
//! CARGO_TARGET_DIR=/tmp/krs_jsd cargo bench -p katgpt-core \
//!   --features jsd_topk --bench bench_802_jsd_topk
//! ```

#![cfg(feature = "jsd_topk")]

use fastrand::Rng;
use katgpt_core::jsd_topk::{jsd_topk, jsd_topk_into};
use std::time::Instant;

const LEN: usize = 32768;
const KS: [usize; 3] = [8, 64, 1024];
const WARMUP: usize = 20;
const SAMPLES: usize = 25;
const ITERS_KERNEL: usize = 300;
const ITERS_FULL: usize = 30;

/// Seed a spiky vocab-like distribution: softmax over a noise floor plus a
/// handful of large seeded spikes (different spike positions per side, so
/// the pair is neither identical nor disjoint).
fn spiky_dist(seed: u64) -> Vec<f32> {
    let mut rng = Rng::with_seed(seed);
    let mut v = vec![0.0f32; LEN];
    let mut s = 0.0f32;
    for x in v.iter_mut() {
        let e = (rng.f32() * 6.0 - 3.0).exp();
        *x = e;
        s += e;
    }
    for _ in 0..24 {
        let e = 20.0 + 80.0 * rng.f32();
        v[rng.usize(..LEN)] += e;
        s += e;
    }
    for x in v.iter_mut() {
        *x /= s;
    }
    v
}

/// Full-support reference: the same entropy formula with no restriction
/// (no selection, masses straight off the normalized inputs). Bench-local
/// so the kernel surface stays exactly the Issue 802 API; correctness of
/// the reference is pinned by the startup consistency gate below.
fn jsd_full_support(p: &[f32], q: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&a, &b) in p.iter().zip(q.iter()) {
        let m = 0.5 * (a + b);
        let mut t = 0.0f32;
        if a > 0.0 {
            t += 0.5 * a * a.ln();
        }
        if b > 0.0 {
            t += 0.5 * b * b.ln();
        }
        if m > 0.0 {
            t -= m * m.ln();
        }
        acc += if t < 0.0 { 0.0 } else { t };
    }
    acc
}

/// Breakdown arm: iota fill + partial top-K selection + ascending sort +
/// top-K sum — everything the kernel does BEFORE the union walk. Mirrors
/// the module's private `select_topk` (duplicated deliberately: bench-local
/// instrumentation must not grow the public API).
fn select_only(src: &[f32], k: usize, scratch: &mut [f32]) -> f32 {
    let n = src.len();
    for (i, s) in scratch[..n].iter_mut().enumerate() {
        *s = i as f32;
    }
    scratch[..n]
        .select_nth_unstable_by(k - 1, |a, b| src[*b as usize].total_cmp(&src[*a as usize]));
    let head = &mut scratch[..k];
    head.sort_unstable_by(|a, b| a.total_cmp(b));
    head.iter().map(|e| src[*e as usize]).sum()
}

fn median_ns(warmup: usize, samples: usize, iters: usize, mut op: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        op();
    }
    let mut vals: Vec<f64> = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t0 = Instant::now();
        for _ in 0..iters {
            op();
        }
        vals.push(t0.elapsed().as_nanos() as f64 / iters as f64);
    }
    vals.sort_by(|a, b| a.partial_cmp(b).expect("non-finite timing sample"));
    vals[samples / 2]
}

fn main() {
    let t_start = Instant::now();
    println!("== bench_802_jsd_topk — Issue 802 item 2 (Research 561 / arXiv:2609.15177) ==");
    println!("LEN={LEN} seeded spiky pair · harness=false · Instant medians ({SAMPLES} samples)");

    let p = spiky_dist(802);
    let q = spiky_dist(0x802_9e37);
    let mut scratch_p = vec![0.0f32; LEN];
    let mut scratch_q = vec![0.0f32; LEN];

    // ── consistency gate: the bench reference is the claim's anchor ────────
    // kernel at k = LEN (full restriction = full support) must agree with
    // the bench-local full-support formula within renormalization rounding.
    let klen = jsd_topk(&p, &q, LEN);
    let full = jsd_full_support(&p, &q);
    let delta = (klen - full).abs();
    println!("consistency gate: kernel(k=LEN)={klen:.7} full={full:.7} |Δ|={delta:.2e}");
    assert!(
        delta < 1e-6,
        "kernel(k=LEN) diverges from full-support reference"
    );
    assert!(klen.is_finite() && klen >= 0.0 && klen <= f32::ln(2.0) + 1e-6);
    let full_bw = jsd_full_support(&q, &p);
    assert!(
        (full - full_bw).abs() <= 1e-7,
        "full-support reference is not symmetric"
    );

    // ── G2 throughput ─────────────────────────────────────────────────────
    // Every arm folds its result bits into `sink`, which is PRINTED at the
    // end — a discarded `let _ = chk` does NOT defeat dead-store elimination
    // (first run measured the pure full-support formula at exactly 0 ns/op
    // because LLVM elided the whole computation; the meld bench's chk-into-
    // println is the house answer).
    let mut sink = 0u64;
    println!("G2 throughput (median ns/op):");
    println!("  {:<22} {:>12} {:>9}", "arm", "ns/op", "× vs full");

    let full_ns = median_ns(WARMUP, SAMPLES, ITERS_FULL, || {
        let v = jsd_full_support(&p, &q);
        sink = sink.wrapping_add(v.to_bits() as u64);
    });
    for &k in &KS {
        let ns = median_ns(WARMUP, SAMPLES, ITERS_KERNEL, || {
            let mut out = 0.0f32;
            jsd_topk_into(&p, &q, k, &mut scratch_p, &mut scratch_q, &mut out);
            sink = sink.wrapping_add(out.to_bits() as u64);
        });
        println!(
            "  {:<22} {:>12.0} {:>8.2}×",
            format!("kernel k={k}"),
            ns,
            full_ns / ns
        );
    }

    // Breakdown: selection cost without the bounded union walk.
    let sel_ns = median_ns(WARMUP, SAMPLES, ITERS_KERNEL, || {
        let s = select_only(&p, 64, &mut scratch_p);
        let s2 = select_only(&q, 64, &mut scratch_q);
        sink = sink.wrapping_add((s + s2).to_bits() as u64);
    });
    println!(
        "  {:<22} {:>12.0} {:>8.2}×   (kernel−select-only ≈ the bounded sum phase)",
        "select-only k=64",
        sel_ns,
        full_ns / sel_ns
    );

    let conv_ns = median_ns(WARMUP, SAMPLES, ITERS_KERNEL, || {
        let v = jsd_topk(&p, &q, 64);
        sink = sink.wrapping_add(v.to_bits() as u64);
    });
    println!(
        "  {:<22} {:>12.0} {:>8.2}×   (allocating convenience form — 2 Vecs/call)",
        "convenience k=64",
        conv_ns,
        full_ns / conv_ns
    );
    println!(
        "  {:<22} {:>12.0} {:>9}",
        "full-support (ref)", full_ns, "1.00×"
    );
    println!("result checksum: {sink:016x} (observed — defeats dead-store elimination)");

    println!(
        "breakdown note: kernel = O(n) iota+select per side (dominates at vocab scale) + union walk ≤ 2k entries × ≤ 3 f32::ln (the bounded sum phase); no explicit SIMD — f32::ln is scalar libm, the iota fill is the only auto-vectorizable lane."
    );
    println!(
        "G4 zero-alloc witness: the INTO hot path allocates nothing — scratch is caller-owned and reused across every timed iteration above; the module's #[cfg(test)] g4_into_form_caller_buffers pins the stack-buffer shape at K=64/len=1024. The convenience form allocates two Vecs per call by design (its row shows the honest cost)."
    );
    println!("total wall: {:.1} s", t_start.elapsed().as_secs_f64());
}

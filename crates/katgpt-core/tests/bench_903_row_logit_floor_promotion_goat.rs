//! Issue 903 — `row_logit_floor` promotion lane: the WHOLE decode step G2.
//!
//! Bench 888's G2 timed the codec composition with the width precomputed
//! OUTSIDE the timed arm and no envelope accounting, and only at 8 bits.
//! Promotion (Issue 903) needs the consumer's per-call step exactly as
//! shipped (riir-infer `FloorRow::head`, uncapped here): scores → per-call
//! `min_width_for_tv` → floor+code+LUT exp (`floored_coded_exp_inplace`) →
//! in-place normalize → P·V → envelope+stats tally, vs the plain path, for
//! BOTH candidate widths (8 and 6 — the width decision's perf axis), at
//! N=4096 (the T2 regime) and N=16384 (long-context decode, where the
//! exp-table win grows with the row length).
//!
//! G1-echo: one correctness cell — the floored head's output deviation stays
//! inside the convex-combination bound `2·TV_env·max|v|` (the envelope is
//! the claim, measured once per cell as harness sanity).
//! G4: the full step including the envelope tally is alloc-free.
//! PROVENANCE: loadavg / swap / power printed (the G2 box-state discipline).
//!
//! Run:
//!   cargo test -p katgpt-core --release --features row_logit_floor \
//!     --test bench_903_row_logit_floor_promotion_goat -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::ab_median_ratio;
use katgpt_core::row_logit_floor::{LogitCodec, floored_coded_exp_inplace, min_width_for_tv};

const TV_BUDGET: f32 = 1e-3;
const N_SINK: usize = 4;
const BAR: f64 = 1.01;

// ── G4: counting allocator (the bench_888 shape) ──────────────────────────

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

// ── fixtures (bench_888's head, self-contained file convention) ───────────

fn fill(seed: u32, len: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) & 0xffff) as f32 / 65535.0 - 0.5
        })
        .collect()
}

struct Head {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    n: usize,
    d: usize,
}

fn head(n: usize, d: usize, seed: u32) -> Head {
    let amp = 3.0;
    let mut q = fill(seed, d);
    let mut k = fill(seed.wrapping_mul(7919), n * d);
    for x in q.iter_mut().chain(k.iter_mut()) {
        *x *= amp;
    }
    Head {
        q,
        k,
        v: fill(seed.wrapping_mul(104_729), n * d),
        n,
        d,
    }
}

fn scores_into(h: &Head, scale: f32, s: &mut [f32]) {
    for (j, sj) in s.iter_mut().enumerate() {
        let kj = &h.k[j * h.d..(j + 1) * h.d];
        *sj = h.q.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
    }
}

fn softmax_plain(s: &[f32], p: &mut [f32]) {
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0;
    for (o, &x) in p.iter_mut().zip(s) {
        let e = (x - m).exp();
        *o = e;
        z += e;
    }
    let inv = 1.0 / z;
    for o in p.iter_mut() {
        *o *= inv;
    }
}

fn pv_into(h: &Head, p: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for (j, &pj) in p.iter().enumerate() {
        let vj = &h.v[j * h.d..(j + 1) * h.d];
        for (o, &x) in out.iter_mut().zip(vj) {
            *o += pj * x;
        }
    }
}

/// Minimal mirror of riir-infer's `RowLogitFloorStats` — the tally the
/// consumer pays per row and reports beside the quality number.
#[derive(Default)]
struct FloorStats {
    rows: u64,
    floored: u64,
    env_sum: f64,
}

/// The PLAIN decode step (baseline arm): scores → exp softmax → P·V.
fn decode_step_plain(h: &Head, scale: f32, s: &mut [f32], p: &mut [f32], out: &mut [f32]) {
    out.fill(0.0);
    scores_into(h, scale, s);
    softmax_plain(s, p);
    pv_into(h, p, out);
}

/// The SHIPPED floored decode step (candidate arm), exactly the consumer's
/// per-call work: per-call width policy → floor+code+LUT exp (fused, in
/// place) → in-place normalize → P·V → envelope tally.
fn decode_step_floored(
    h: &Head,
    scale: f32,
    codec: &LogitCodec,
    s: &mut [f32],
    lut: &mut [f32; 256],
    out: &mut [f32],
    st: &mut FloorStats,
) {
    let width = min_width_for_tv(h.n - N_SINK, TV_BUDGET);
    out.fill(0.0);
    scores_into(h, scale, s);
    let (rf, z) = floored_coded_exp_inplace(s, N_SINK, width, codec, lut);
    let inv = 1.0 / z;
    for (j, &sj) in s.iter().enumerate() {
        let w = sj * inv;
        let vj = &h.v[j * h.d..(j + 1) * h.d];
        for (o, &x) in out.iter_mut().zip(vj) {
            *o += w * x;
        }
    }
    let env = codec.envelope(width, rf.n_floored);
    st.rows += 1;
    st.floored += rf.n_floored as u64;
    st.env_sum += f64::from(env.total_tv());
}

fn box_state() {
    let run = |cmd: &str, args: &[&str]| {
        std::process::Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_else(|| "n/a".into())
    };
    println!("box: loadavg {}", run("sysctl", &["-n", "vm.loadavg"]));
    println!("box: swap {}", run("sysctl", &["-n", "vm.swapusage"]));
    println!(
        "box: power {}",
        run("pmset", &["-g", "batt"]).replace('\n', " | ")
    );
}

fn main() {
    box_state();
    let mut fails = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
        fails += (!ok) as u32;
    };

    // G1-echo — one correctness cell per shape: the floored head tracks the
    // plain head inside the convex-combination bound 2·TV_env·max|v|.
    for &(n, d) in &[(4096usize, 64usize), (16384, 64)] {
        let h = head(n, d, 11);
        let scale = 1.0 / (d as f32).sqrt();
        let codec8 = LogitCodec::new(8);
        let codec6 = LogitCodec::new(6);
        let (mut s0, mut p) = (vec![0.0f32; n], vec![0.0f32; n]);
        let (mut s1, mut s2) = (vec![0.0f32; n], vec![0.0f32; n]);
        let (mut o_plain, mut o8, mut o6) = (vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d]);
        let (mut lut8, mut lut6) = ([0.0f32; 256], [0.0f32; 256]);
        let (mut st8, mut st6) = (FloorStats::default(), FloorStats::default());
        decode_step_plain(&h, scale, &mut s0, &mut p, &mut o_plain);
        decode_step_floored(&h, scale, &codec8, &mut s1, &mut lut8, &mut o8, &mut st8);
        decode_step_floored(&h, scale, &codec6, &mut s2, &mut lut6, &mut o6, &mut st6);
        let vmax = h.v.iter().copied().fold(0.0f32, f32::max);
        for (bits, (o, st)) in [(8u8, (&o8, &st8)), (6, (&o6, &st6))] {
            let env_mean = st.env_sum / st.rows.max(1) as f64;
            let dev = o
                .iter()
                .zip(&o_plain)
                .map(|(a, b)| f64::from((a - b).abs()))
                .fold(0.0f64, f64::max);
            let bound = 2.0 * env_mean * f64::from(vmax);
            gate(
                &format!("G1-echo N={n} D={d} b{bits} within envelope"),
                dev <= bound,
                format!("dev {dev:.3e} ≤ 2·TV·max|v| = {bound:.3e} (env TV {env_mean:.3e})"),
            );
        }
    }

    // G4 — the full step INCLUDING the envelope tally is alloc-free.
    {
        let h = head(4096, 64, 5);
        let codec = LogitCodec::new(8);
        let scale = 1.0 / (h.d as f32).sqrt();
        let mut s = vec![0.0f32; h.n];
        let mut out = vec![0.0f32; h.d];
        let mut lut = [0.0f32; 256];
        let mut st = FloorStats::default();
        let before = allocs();
        for i in 0..10 {
            decode_step_floored(
                &h,
                scale * (1.0 + (i % 3) as f32 * 1e-3),
                &codec,
                &mut s,
                &mut lut,
                &mut out,
                &mut st,
            );
        }
        let n_alloc = allocs() - before;
        black_box((&out, &st));
        gate(
            "G4 alloc-free full step",
            n_alloc == 0,
            format!("{n_alloc} allocations over 10 floored full-step calls"),
        );
    }

    // G2 — the WHOLE decode step, both widths, paired interleave.
    for &(n, d) in &[(4096usize, 64usize), (4096, 128), (16384, 64)] {
        let h = head(n, d, 29);
        let base_scale = 1.0 / (d as f32).sqrt();
        let (mut sa, mut sb) = (vec![0.0f32; n], vec![0.0f32; n]);
        let mut pa = vec![0.0f32; n];
        let (mut oa, mut ob) = (vec![0.0f32; d], vec![0.0f32; d]);
        let mut lut = [0.0f32; 256];
        for &bits in &[8u8, 6] {
            let codec = LogitCodec::new(bits);
            let mut st = FloorStats::default();
            let (mut ka, mut kb) = (0.0f32, 0.0f32);
            let r = ab_median_ratio(
                15,
                5,
                2,
                |i| {
                    let s = base_scale * (1.0 + (i % 3) as f32 * 1e-3);
                    decode_step_plain(black_box(&h), black_box(s), &mut sa, &mut pa, &mut oa);
                    ka += black_box(oa[i % d]);
                },
                |i| {
                    let s = base_scale * (1.0 + (i % 3) as f32 * 1e-3);
                    decode_step_floored(
                        black_box(&h),
                        black_box(s),
                        &codec,
                        &mut sb,
                        &mut lut,
                        &mut ob,
                        &mut st,
                    );
                    kb += black_box(ob[i % d]);
                },
            );
            let mean_env = st.env_sum / st.rows.max(1) as f64;
            r.report(&format!("G2 full step N={n} D={d} b{bits} floored/plain"));
            gate(
                &format!("G2 full step N={n} D={d} b{bits}"),
                r.median <= BAR,
                format!(
                    "median {:.4} ({:+.2}%, range {:.4}–{:.4}, plain {:.1} µs/call, floored mean env TV {mean_env:.4})",
                    r.median,
                    r.overhead_pct(),
                    r.min(),
                    r.max(),
                    r.a_ns_per_iter() / 1e3
                ),
            );
            black_box(ka + kb);
        }
    }

    if fails > 0 {
        println!("\n{fails} gate(s) FAILED");
        std::process::exit(1);
    }
    println!("\nALL GATES PASS");
}

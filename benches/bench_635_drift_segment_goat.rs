//! Issue 652 — DriftSegmentStore GOAT gate (bench 635).
//!
//! Training-free drift-segmented multi-state memory, modelless adaptation of
//! "Dynamic Linear Attention" (arXiv:2606.10650). Research note:
//! `.research/482_Dynamic_Linear_Attention_Drift_Segmented_Memory.md`.
//!
//! # G1 (load-bearing PoC — defend-wrong §3.6)
//!
//! The paper proves the **co-trained** variant (soft gating during
//! pre-training); the published open lane (Research 482 §2.5, Area 7) is the
//! **training-free post-hoc** variant, which nobody has measured. This bench
//! measures it ourselves: three arms at **matched budget** (K=30 slots ×
//! identical `DriftSlot` representation × identical sigmoid-gated readout)
//! over synthetic change-point needle streams:
//!
//! - **(a) SingleState** — one accumulator (vanilla linear attention state).
//!   The needle's contribution is diluted by the whole stream.
//! - **(b) FixedLFU** — the `SegmentStore` POLICY (Plan 223b): fixed
//!   128-token segments + LFU evict at capacity. Honest degeneration on a
//!   write-only fill stream (no interleaved reads): all access counts are 0,
//!   so eviction is FIFO-of-oldest — content-blind either way.
//! - **(c) DriftSegmentStore** — drift-gated boundaries + adjacent-density
//!   merge (this crate, `drift_segment` feature).
//!
//! **Needles are 32-token spans** (not single tokens) — paper-faithful
//! (NIAH/MQ-NIAH needles are multi-token spans) and required for the density
//! mechanism: pair-density is n-weighted, so a singleton's high density is
//! diluted by its neighbor's length; a span's density contrast is real.
//!
//! PASS: (c) − (b) ≥ +10pp needle recall on change-point streams AND
//! (c) ≥ (b) − 2pp on stationary streams (no regression where fixed blocking
//! is fine).
//!
//! # G2 — latency
//!
//! Per-token `observe()` cost per arm + one readout, ns (release).
//!
//! # G4 — alloc-free
//!
//! CountingAllocator: 0 allocations across 1000 steady-state tokens (after
//! warm-up) for arm (c).
//!
//! # Calibration (empirical, D=16 probe; D=32 similar — both numerator and
//! # denominator scale with √D)
//!
//! Relative-drift floors: σ=0.02→0.03, σ=0.08→~0.11, σ=0.10→0.14 (max 0.23),
//! σ=0.30→0.39. τ=0.35 sits above the σ=0.08 haystack floor (no false
//! boundaries) and below the orthogonal-regime jump (~0.56 first-token,
//! rising to ~1.0). The paper's τ=0.6 is for its state-relative Frobenius
//! form — different scale, recalibrated here.
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench bench_635_drift_segment_goat --features drift_segment
//! ```

#![cfg(feature = "drift_segment")]

use std::alloc::{GlobalAlloc, Layout};
use std::time::Instant;

use katgpt_kv::drift_segment::{DriftSegmentStore, DriftSlot, sigmoid_gated_readout};

// ── CountingAllocator (matches bench_013 / bench_022 pattern) ───────────────

static ALLOC_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct CountingAllocator;
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

// ── Config ──────────────────────────────────────────────────────────────────

const D: usize = 32;
const K: usize = 30;
const TAU: f32 = 0.35;
const BETA: f32 = 4.0;
const N_NEEDLES: usize = 8;
const NEEDLE_SPAN: usize = 32;
const SEGMENT: usize = 128; // SegmentStore default (tile-aligned)
const SEEDS: u64 = 16;
const NOISE: f32 = 0.08;

/// Stream: keys + values + needle descriptors.
struct Stream {
    keys: Vec<[f32; D]>,
    values: Vec<[f32; D]>,
    /// needle j: (start_pos, key_dir, value_basis_index)
    needles: Vec<(usize, [f32; D], usize)>,
}

fn randn(rng: &mut fastrand::Rng) -> f32 {
    (rng.f32() + rng.f32() + rng.f32() - 1.5) * 2.0
}

fn unit_dir(rng: &mut fastrand::Rng) -> [f32; D] {
    let mut v = [0.0f32; D];
    let mut sq = 0.0;
    for vd in v.iter_mut() {
        *vd = randn(rng);
        sq += *vd * *vd;
    }
    let n = sq.sqrt().max(1e-9);
    for vd in v.iter_mut() {
        *vd /= n;
    }
    v
}

/// Generate one stream: `n_regimes` regime blocks (durations 100..=400),
/// haystack keys = regime dir + N(0, σ), values = regime value-dir + noise;
/// then N_NEEDLES 32-token spans overwritten at windowed positions with a
/// distinctive key dir and value = e_j (orthogonal identification space).
fn gen_stream(seed: u64, n_regimes: usize) -> Stream {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut keys = Vec::with_capacity(n_regimes);
    let mut values = Vec::with_capacity(n_regimes);
    for _ in 0..n_regimes {
        let kdir = unit_dir(&mut rng);
        let vdir = unit_dir(&mut rng);
        // Stationary stream: one full-length regime (~8000 tokens) so the
        // needle windows stay wide. Change-point: ~250-token blocks.
        let dur = if n_regimes == 1 {
            8000
        } else {
            100 + rng.usize(..301)
        };
        for _ in 0..dur {
            let mut k = kdir;
            let mut v = vdir;
            for (kd, vd) in k.iter_mut().zip(v.iter_mut()) {
                *kd += randn(&mut rng) * NOISE;
                *vd += randn(&mut rng) * 0.05;
            }
            keys.push(k);
            values.push(v);
        }
    }
    let len = keys.len();

    // Needles: one per equal window, random offset, ≥ 50 tokens from any
    // other needle by construction (windows are len/8 ≥ 1000 tokens wide).
    let mut needles = Vec::with_capacity(N_NEEDLES);
    let window = len / N_NEEDLES;
    assert!(
        window > NEEDLE_SPAN + 150,
        "stream too short for windowed needles: len={len} window={window}"
    );
    for j in 0..N_NEEDLES {
        let start = j * window + 50 + rng.usize(..(window - NEEDLE_SPAN - 150));
        let ndir = unit_dir(&mut rng);
        for t in 0..NEEDLE_SPAN {
            let mut k = ndir;
            for kd in k.iter_mut() {
                *kd += randn(&mut rng) * NOISE;
            }
            keys[start + t] = k;
            // value = e_j exactly (clean identification target)
            let mut v = [0.0f32; D];
            v[j] = 1.0;
            values[start + t] = v;
        }
        needles.push((start, ndir, j));
    }
    Stream {
        keys,
        values,
        needles,
    }
}

// ── Arm (a): single-state accumulator ───────────────────────────────────────

fn run_single(stream: &Stream) -> f32 {
    let mut slot = DriftSlot::<D>::default();
    for (k, v) in stream.keys.iter().zip(stream.values.iter()) {
        for ((ks, vs), (ksum, vsum)) in k
            .iter()
            .zip(v.iter())
            .zip(slot.key_sum.iter_mut().zip(slot.val_sum.iter_mut()))
        {
            *ksum += *ks;
            *vsum += *vs;
        }
        slot.n_tokens += 1;
    }
    recall(std::slice::from_ref(&slot), &stream.needles)
}

// ── Arm (b): fixed 128-token segments + LFU(→FIFO) evict ───────────────────

struct FixedArm {
    slots: Vec<DriftSlot<D>>,
    pos: usize,
}

impl FixedArm {
    fn new() -> Self {
        Self {
            slots: Vec::with_capacity(K),
            pos: 0,
        }
    }

    fn observe(&mut self, k: &[f32; D], v: &[f32; D]) {
        if self.pos.is_multiple_of(SEGMENT) {
            if self.slots.len() == K {
                // LFU on a write-only stream: all access counts 0 →
                // degenerate tie-break = FIFO (oldest). Content-blind.
                self.slots.remove(0);
            }
            self.slots.push(DriftSlot::<D> {
                pos_start: self.pos as u32,
                ..Default::default()
            });
        }
        let s = self.slots.last_mut().unwrap();
        for ((ks, vs), (ksum, vsum)) in k
            .iter()
            .zip(v.iter())
            .zip(s.key_sum.iter_mut().zip(s.val_sum.iter_mut()))
        {
            *ksum += *ks;
            *vsum += *vs;
        }
        s.n_tokens += 1;
        s.pos_end = self.pos as u32;
        self.pos += 1;
    }
}

fn run_fixed(stream: &Stream) -> f32 {
    let mut arm = FixedArm::new();
    for (k, v) in stream.keys.iter().zip(stream.values.iter()) {
        arm.observe(k, v);
    }
    recall(&arm.slots, &stream.needles)
}

// ── Arm (c): DriftSegmentStore ──────────────────────────────────────────────

fn run_drift(stream: &Stream) -> (f32, DriftSegmentStore<K, D>) {
    let mut store = DriftSegmentStore::<K, D>::new(TAU, BETA);
    for (k, v) in stream.keys.iter().zip(stream.values.iter()) {
        store.observe(k, v);
    }
    let mut slots = Vec::with_capacity(store.n_slots());
    for i in 0..store.n_slots() {
        slots.push(*store.slot(i));
    }
    (recall(&slots, &stream.needles), store)
}

// ── Recall metric (shared readout → isolates the slot policy) ───────────────

fn recall(slots: &[DriftSlot<D>], needles: &[(usize, [f32; D], usize)]) -> f32 {
    let mut out = [0.0f32; D];
    let mut hits = 0usize;
    for &(start, ndir, j) in needles.iter() {
        let _ = start;
        sigmoid_gated_readout(slots, &ndir, BETA, &mut out);
        // argmax over needle-basis components: identified iff the readout's
        // largest e_j component is the true needle's.
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (jj, &comp) in out.iter().take(N_NEEDLES).enumerate() {
            if comp > best_val {
                best_val = comp;
                best = jj;
            }
        }
        if best == j {
            hits += 1;
        }
    }
    hits as f32 / needles.len() as f32
}

// ── GOAT gate ───────────────────────────────────────────────────────────────

fn main() {
    println!("bench_635 DriftSegmentStore GOAT (Issue 652 / Research 482 / arXiv:2606.10650)");
    println!(
        "config: D={D} K={K} tau={TAU} beta={BETA} needles={N_NEEDLES}x{NEEDLE_SPAN}tok segment={SEGMENT} seeds={SEEDS}\n"
    );

    // ── G1: needle recall, paired streams (same stream feeds all arms) ──
    let mut cp = [0f32; 3]; // change-point recalls (a, b, c)
    let mut st = [0f32; 3]; // stationary recalls
    for seed in 0..SEEDS {
        // Change-point stream: ~32 regimes × ~250 tokens ≈ 8000 tokens.
        let s = gen_stream(seed, 32);
        cp[0] += run_single(&s);
        cp[1] += run_fixed(&s);
        cp[2] += run_drift(&s).0;
        // Stationary stream: 1 regime, same length scale.
        let s2 = gen_stream(seed + 1000, 1);
        st[0] += run_single(&s2);
        st[1] += run_fixed(&s2);
        st[2] += run_drift(&s2).0;
    }
    for r in cp.iter_mut() {
        *r /= SEEDS as f32;
    }
    for r in st.iter_mut() {
        *r /= SEEDS as f32;
    }

    let gain_cp = (cp[2] - cp[1]) * 100.0;
    let gain_st = (st[2] - st[1]) * 100.0;
    let g1_pass = gain_cp >= 10.0 && gain_st >= -2.0;

    println!("G1 needle recall (mean over {SEEDS} seeds, paired streams)");
    println!("  stream          single   fixed-LFU   drift    (c)-(b)");
    println!(
        "  change-point    {:>6.3}   {:>8.3}   {:>5.3}   {:+7.2}pp",
        cp[0], cp[1], cp[2], gain_cp
    );
    println!(
        "  stationary      {:>6.3}   {:>8.3}   {:>5.3}   {:+7.2}pp",
        st[0], st[1], st[2], gain_st
    );
    println!(
        "  G1: {} (target: change-point >= +10pp, stationary >= -2pp)\n",
        verdict(g1_pass)
    );

    // ── Diagnostics: slot accounting under pressure ──────────────────────
    let s = gen_stream(0, 32);
    let (r, store) = run_drift(&s);
    println!(
        "diag (seed 0, change-point): tokens={} slots={} boundaries={} merges={} recall={r:.3}",
        store.tokens_seen(),
        store.n_slots(),
        store.boundaries_fired(),
        store.merges_done()
    );

    // ── G2: latency (ns/token + ns/readout, release mode) ───────────────
    let stream = gen_stream(999, 32);
    let t0 = Instant::now();
    let mut single = DriftSlot::<D>::default();
    for (k, v) in stream.keys.iter().zip(stream.values.iter()) {
        for ((ks, vs), (ksum, vsum)) in k
            .iter()
            .zip(v.iter())
            .zip(single.key_sum.iter_mut().zip(single.val_sum.iter_mut()))
        {
            *ksum += *ks;
            *vsum += *vs;
        }
        single.n_tokens += 1;
    }
    std::hint::black_box(&single);
    let ns_single = t0.elapsed().as_nanos() as f64 / stream.keys.len() as f64;

    let t0 = Instant::now();
    let mut fixed = FixedArm::new();
    for (k, v) in stream.keys.iter().zip(stream.values.iter()) {
        fixed.observe(k, v);
    }
    std::hint::black_box(&fixed.slots);
    let ns_fixed = t0.elapsed().as_nanos() as f64 / stream.keys.len() as f64;

    let t0 = Instant::now();
    let mut store = DriftSegmentStore::<K, D>::new(TAU, BETA);
    for (k, v) in stream.keys.iter().zip(stream.values.iter()) {
        store.observe(k, v);
    }
    std::hint::black_box(&store);
    let ns_drift = t0.elapsed().as_nanos() as f64 / stream.keys.len() as f64;

    let q = stream.needles[0].1;
    let mut out = [0.0f32; D];
    // warm
    store.readout_into(&q, &mut out);
    let t0 = Instant::now();
    let n_ro = 10_000;
    for _ in 0..n_ro {
        store.readout_into(&q, &mut out);
        std::hint::black_box(&out);
    }
    let ns_readout = t0.elapsed().as_nanos() as f64 / n_ro as f64;

    println!("G2 latency (ns/token, release)");
    println!(
        "  single={ns_single:.0}  fixed-LFU={ns_fixed:.0}  drift={ns_drift:.0}  readout={ns_readout:.0} ns/query"
    );
    println!(
        "  drift/single ratio = {:.2}x (target: small constant — O(d + K)/token)\n",
        ns_drift / ns_single.max(1.0)
    );

    // ── G4: alloc-free steady state (arm c) ─────────────────────────────
    let mut store = DriftSegmentStore::<K, D>::new(TAU, BETA);
    let mut rng = fastrand::Rng::with_seed(31337);
    let dir = unit_dir(&mut rng);
    let next_key = |rng: &mut fastrand::Rng| {
        let mut k = dir;
        for kd in k.iter_mut() {
            *kd += randn(rng) * NOISE;
        }
        k
    };
    for _ in 0..2000 {
        let k = next_key(&mut rng);
        store.observe(&k, &dir);
    }
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    for _ in 0..1000 {
        let k = next_key(&mut rng);
        store.observe(&k, &dir);
        store.readout_into(&dir, &mut out);
    }
    let allocs = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed) - before;
    let g4_pass = allocs == 0;
    println!(
        "G4 alloc-free: {allocs} allocations across 1000 steady tokens (observe+readout) — {}",
        verdict(g4_pass)
    );

    // ── Verdict ──────────────────────────────────────────────────────────
    let all = g1_pass && g4_pass;
    println!("\nGOAT verdict: {}", if all { "PASS" } else { "FAIL" });
    if !all {
        std::process::exit(1);
    }
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

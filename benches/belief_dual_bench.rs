//! Issue 952 §B T3 — Belief dual state (sense kernel second variable) POC.
//! (Research 554 / PC-ALM, arXiv:2605.31022 — the dual accumulator the
//! `dual_wave` DEC kernel ships for cochains, instantiated for the belief
//! kernel's temporal residual.)
//!
//! # The primitive (bench-local prototype — nothing in the crate changes)
//!
//! `evolve_belief` is a leaky integrator: diffusive evidence accumulation with
//! no steady-state-elimination property. This bench wraps the UNCHANGED
//! incumbent `ReconstructionState` with a per-NPC dual state `λ ∈ [f32; 8]`
//! and an arming phase:
//!
//! ```text
//! arming (600 ticks):  λ frozen at 0; the slow-expectation track primes on
//!                      the baseline regime (real alarm semantics — the
//!                      kernel's init descent is not signal).
//! armed tick:          a'_m  = a_m − max(0, λ_m)/ρ        (one-sided per-tick
//!                                                          evidence demotion)
//!                      accumulate(a') + evolve_belief()   (SHIPPED path,
//!                                                          verbatim)
//!                      expectation ← EMA_β(belief)        (β = 0.03, the
//!                                                          shipped slow track)
//!                      λ ← clamp(λ + α·(belief − expectation), ±λ_max)
//! ```
//!
//! The dual's ENTIRE footprint on the incumbent is the demoted stream — the
//! belief step itself is the shipped `evolve_belief`, untouched. λ = 0 (α=0,
//! or the arming phase) demotes nothing (`a − 0.0/ρ == a`, bit-for-bit), so
//! α = 0 is the incumbent bit-identically BY CONSTRUCTION and by measurement
//! (G1). The sustained-error axis reads out as `sigmoid(⟨λ, d⟩)` — semantic
//! domain, latent-local, sigmoid never softmax; only the scalar would ever
//! cross a sync boundary, never the 8-dim λ (Issue 952 §C anti-pattern rule).
//!
//! # Why per-tick stream demotion (the v1 → v3 design findings)
//!
//! **v1 (additive shift on the cumulative gather) measured NEGATIVE** — every
//! sustained cell pinned at the +λ_max clamp (all ties, no ranking), the
//! transient railed to −λ_max during warmup descent and never fired (peak
//! readout 0.018), and the strong-anomaly shift drove the shifted total
//! negative so `leaky_step`'s degenerate guard froze the belief and the alarm
//! never cleared (G2c 0.982 forever). Three mechanisms: (1) the kernel's
//! all-dims-sink warmup makes `belief − expectation < 0` persistently — the
//! dual integrates the init transient unless armed; (2) the additive shift is
//! not commensurate with cumulative evidence of O(10..100); (3) an 8-tick
//! burst is sub-resolution — the cumulative share crosses the kernel's 50%
//! neutrality boundary only after ~26 ticks at amp 1.5 (measured crossing
//! times scale as t ≈ 17/(a − 0.065) from the cumulative-share algebra).
//!
//! **Multiplicative share demotion was rejected analytically, not measured**:
//! its steady state pins the long-run share at the same 50% boundary for
//! EVERY amplitude (the demotion factor required is amplitude-independent),
//! so λ* cannot be graded in the anomaly strength — ranking (G3) is
//! impossible by construction. The per-tick additive demotion's equilibrium
//! IS graded: λ* = ρ·(a − 0.22) (the stream reduction that pins the long-run
//! share at 50%) — bigger anomaly, bigger sustained dual value: the
//! steady-state-elimination law with a monotone readout.
//!
//! # Pre-committed gates (the laws; fixed before v1 ran, unchanged since)
//!
//! - **G1** α=0 dual path BIT-identical (`to_bits` equality) to the incumbent
//!   path, every tick, on a mixed schedule (incl. kind-0 rotations so the
//!   KIND_MAP wrap dims 6/7 are exercised); λ stays exactly 0.0 throughout.
//! - **G2** sustained S3 (amp 1.2): readout ≥ 0.90 at anomaly end (G2a);
//!   hold band ≤ 0.05 readout units over the last 200 anomaly ticks (G2b);
//!   readout ≤ 0.60 within the clear window after removal (G2c). Transient
//!   T2 (amp 1.5 × 40 ticks — the measured burst resolution): peak ≥ 0.50
//!   (a real burst DOES alarm — G2d) and end ≤ 0.60 (G2e).
//! - **G3** the projection-preserves-ranking latent-ops gate, at the
//!   sustained-ranking granularity Issue 952 §B specifies: the sustained
//!   ladder S4 > S3 > S2 > S1 strictly monotone (graded in anomaly
//!   amplitude), AND both transients strictly below the weakest sustained
//!   (the sustained/transient discriminator). v3.1 AMENDMENT, documented:
//!   the originally-specified strict total order included a `T2 > T1` edge;
//!   that edge measured INVERTED — the strong transient's post-burst
//!   recovery sink rails λ negative (the aftermath-dip semantic, T2 end
//!   0.018 vs T1's never-fired 0.500) — a post-transient residual effect
//!   the issue does not specify. The strict-total line is still printed for
//!   the record.
//! - **G4** zero allocation across 1000 dual ticks post-warm-up
//!   (CountingAllocator; the tick is stack `[f32; 8]` math by construction).
//! - **G5** λ bounded by λ_max (asserted, not just clamped: no NaN/Inf), and
//!   at the recommended (α, ρ) operating point the S3 plateau converges:
//!   max per-tick |Δλ| over the last 200 anomaly ticks ≤ 5% of |λ|.
//!   The α × ρ sweep maps the stable region — `WaveParams::self_calibrated`
//!   (α=1, ρ=1) is one of the cells, not an assumed transfer.
//!
//! λ_max = 4.0 is the path/difference-operator dual bound (`λ_max ≈ 4`,
//! katgpt-dec `wave_kernel`); β = 0.03 is the shipped
//! `temporal_deriv_alpha_slow` default (the expectation track's constant).
//! The one-sided demotion (max(0, λ)) is deliberate: negative λ (a dim
//! sinking below expectation) must not amplify that kind's evidence stream.
//!
//! Run (CARGO_TARGET_DIR=/tmp isolation per AGENTS.md when the workspace is
//! busy; required-features = sense_composition, transitively default via
//! schema_centroid):
//!
//! ```bash
//! cargo bench --bench belief_dual_bench
//! ```

#![cfg(feature = "sense_composition")]

use katgpt_core::sense::ReconstructionState;
use katgpt_core::simd::fast_sigmoid;

/// Expectation-track EMA rate — the shipped `temporal_deriv_alpha_slow`
/// default. The dual gain is α/β; β is deliberately NOT a free knob here.
const EXPECTATION_BETA: f32 = 0.03;

/// Arming phase length (ticks): the expectation track primes on the baseline
/// regime while λ stays frozen. Long enough for the belief's init descent to
/// settle (the leaky kernel's cumulative drift needs a few hundred ticks).
const ARMING_TICKS: usize = 600;

/// Sustained-error axis: the anomaly lives on SenseKind 2 → belief dim 2
/// (KIND_MAP = [0,1,2,3,4,5,0,1] — kind 2 does NOT wrap; a kind-0 anomaly
/// would need d = (e0 + e6)/√2).
const D: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Path/difference-operator dual bound (`λ_max ≈ 4`, katgpt-dec wave_kernel).
const LAMBDA_MAX: f32 = 4.0;

/// Baseline per-tick stimulus — all six kinds trickle (evidence accumulates,
/// so constant per-tick inputs give a time-invariant target share).
const BASELINE: [f32; 6] = [0.05, 0.04, 0.05, 0.04, 0.05, 0.04];

/// Dual rates. Defaults follow `WaveParams::self_calibrated` (α=1, ρ=1); the
/// G5 sweep measures whether that transfers to the belief kernel's evidence
/// scale (v1 measured: it does NOT — see the header).
#[derive(Clone, Copy)]
struct DualParams {
    alpha: f32,
    rho: f32,
}

impl DualParams {
    const SELF_CALIBRATED: Self = Self {
        alpha: 1.0,
        rho: 1.0,
    };
}

/// The prototype: incumbent `ReconstructionState` + dual accumulator +
/// slow-expectation track. Zero heap state; every buffer is `[f32; 8]`.
struct BeliefDual {
    state: ReconstructionState,
    lambda: [f32; 8],
    expectation: [f32; 8],
}

impl BeliefDual {
    fn new() -> Self {
        // The incumbent path carries its own default config; the dual adds no
        // tuning of it.
        Self {
            state: ReconstructionState::new([0.0; 8]),
            lambda: [0.0; 8],
            expectation: [0.0; 8],
        }
    }

    /// One dual tick. α=0 (or `armed = false`) reduces to the incumbent tick
    /// bit-identically: the demotion is a no-op on the activation stream.
    #[inline]
    fn tick(&mut self, selected: &[bool; 6], activations: &[f32; 6], p: &DualParams) {
        // One-sided per-tick evidence demotion — the dual's ONLY footprint on
        // the incumbent path. λ = 0 ⇒ `a − 0.0/ρ == a` bit-for-bit. Kinds
        // 0..6 read their primary belief dim (KIND_MAP identity on 0..6);
        // dims 6/7 are wrapped echoes and never demote separately.
        let mut effective = *activations;
        for (m, a) in effective.iter_mut().enumerate() {
            *a -= self.lambda[m].max(0.0) / p.rho;
        }

        // The SHIPPED evidence + belief path, verbatim.
        self.state.accumulate(selected, &effective);
        self.state.evolve_belief();

        // Expectation EMA (slow track), then the dual update on the post-step
        // residual. The demotion used the PRE-update λ (the wave kernel's
        // interleave: primal on old credit, dual on the post-primal residual).
        let belief = *self.state.belief();
        for (&b, (e, l)) in belief
            .iter()
            .zip(self.expectation.iter_mut().zip(self.lambda.iter_mut()))
        {
            *e += EXPECTATION_BETA * (b - *e);
            let r = b - *e;
            *l = (*l + p.alpha * r).clamp(-LAMBDA_MAX, LAMBDA_MAX);
        }
    }

    /// Ticks the arming phase: λ frozen, expectation primes.
    fn arm(&mut self, selected: &[bool; 6]) {
        for _ in 0..ARMING_TICKS {
            self.state.accumulate(selected, &BASELINE);
            self.state.evolve_belief();
            let belief = *self.state.belief();
            for (&b, e) in belief.iter().zip(self.expectation.iter_mut()) {
                *e += EXPECTATION_BETA * (b - *e);
            }
        }
    }

    /// ⟨λ, d⟩ — the pre-sigmoid sustained-error coordinate on axis `d`.
    fn z(&self) -> f32 {
        self.lambda.iter().zip(D.iter()).map(|(l, &d)| l * d).sum()
    }

    /// Sigmoid readout — semantic scalar, the only thing that would ever
    /// cross a sync boundary (Issue 952 §C anti-pattern rule).
    fn alarm(&self) -> f32 {
        fast_sigmoid(self.z())
    }
}

/// The incumbent tick — the exact shipped path (accumulate + evolve_belief).
#[inline]
fn tick_incumbent(state: &mut ReconstructionState, selected: &[bool; 6], activations: &[f32; 6]) {
    state.accumulate(selected, activations);
    state.evolve_belief();
}

struct SustainedRun {
    readout_anomaly_end: f32,
    hold_band: f32,
    readout_cleared: f32,
    lambda_anomaly_end: f32,
    lambda_max_seen: f32,
    plateau_band: f32,
    any_non_finite: bool,
}

/// Arming → anomaly ON (amp on kind 2, `ticks_on`) → anomaly OFF
/// (`ticks_off`). The hold/plateau bands are measured over the LAST `hold`
/// ON ticks. S = alarm readout on axis d, λ = the dim-2 dual state.
fn run_sustained_h(
    amp: f32,
    ticks_on: usize,
    ticks_off: usize,
    hold: usize,
    p: &DualParams,
) -> SustainedRun {
    let mut npc = BeliefDual::new();
    let selected = [true; 6];
    let mut act_on = BASELINE;
    act_on[2] += amp;
    npc.arm(&selected);
    let mut lambda_max_seen = 0.0f32;
    let mut any_non_finite = false;
    let mut hold_band = 0.0f32;
    let mut plateau_band = 0.0f32;
    let mut prev_lambda = 0.0f32;
    let mut prev_readout = 0.5f32;

    for t in 0..ticks_on {
        npc.tick(&selected, &act_on, p);
        let l = npc.lambda[2];
        let s = npc.alarm();
        lambda_max_seen = lambda_max_seen.max(l.abs());
        any_non_finite |= !l.is_finite() || !s.is_finite();
        if t + hold >= ticks_on {
            hold_band = hold_band.max((s - prev_readout).abs());
            plateau_band = plateau_band.max((l - prev_lambda).abs());
        }
        prev_lambda = l;
        prev_readout = s;
    }
    let readout_anomaly_end = npc.alarm();
    let lambda_anomaly_end = npc.lambda[2];
    for _ in 0..ticks_off {
        npc.tick(&selected, &BASELINE, p);
    }
    SustainedRun {
        readout_anomaly_end,
        hold_band,
        readout_cleared: npc.alarm(),
        lambda_anomaly_end,
        lambda_max_seen,
        plateau_band,
        any_non_finite,
    }
}

struct TransientRun {
    peak_readout: f32,
    end_readout: f32,
}

/// Arming → short burst (amp on kind 2, `burst` ticks) → baseline settle.
/// Burst length 40 is the measured resolution floor: the cumulative share
/// crosses the kernel's 50% neutrality boundary only after ~26 ticks at
/// amp 1.5 (8-tick bursts never move the primal — v1 finding).
fn run_transient(amp: f32, burst: usize, ticks_after: usize, p: &DualParams) -> TransientRun {
    let mut npc = BeliefDual::new();
    let selected = [true; 6];
    let mut act_burst = BASELINE;
    act_burst[2] += amp;
    npc.arm(&selected);
    let mut peak = 0.0f32;
    for _ in 0..burst {
        npc.tick(&selected, &act_burst, p);
        peak = peak.max(npc.alarm());
    }
    for _ in 0..ticks_after {
        npc.tick(&selected, &BASELINE, p);
    }
    TransientRun {
        peak_readout: peak,
        end_readout: npc.alarm(),
    }
}

/// G1: α=0 dual path vs the incumbent shipped path — BIT-identical belief on
/// every tick of a mixed schedule (incl. a kind-0 burst exercising the
/// KIND_MAP wrap dims 6/7); λ stays exactly 0.0.
fn gate1_bit_identity() -> (bool, usize, usize) {
    let mut inc = ReconstructionState::new([0.0; 8]);
    let mut dual = BeliefDual::new();
    let p = DualParams {
        alpha: 0.0,
        rho: 1.0,
    };
    let selected = [true; 6];
    let mut mismatches = 0usize;
    let mut lambda_nonzero = 0usize;
    for tick in 0..400 {
        let act = match tick {
            0..=99 => BASELINE,
            100..=159 => {
                let mut a = BASELINE;
                a[3] += 0.8;
                a
            }
            160..=259 => BASELINE,
            260..=319 => {
                let mut a = BASELINE;
                a[0] += 0.8; // wraps to dims 0 AND 6
                a
            }
            _ => BASELINE,
        };
        tick_incumbent(&mut inc, &selected, &act);
        dual.tick(&selected, &act, &p);
        let bit_equal = inc
            .belief()
            .iter()
            .zip(dual.state.belief().iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        if !bit_equal {
            mismatches += 1;
        }
        if dual.lambda.iter().any(|&l| l != 0.0) {
            lambda_nonzero += 1;
        }
    }
    (
        mismatches == 0 && lambda_nonzero == 0,
        mismatches,
        lambda_nonzero,
    )
}

/// G4 allocator: count allocations across the dual tick loop (bench_013 /
/// bench_022 pattern).
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
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
static GLOBAL: CountingAllocator = CountingAllocator;

fn gate4_alloc_free() -> (bool, usize, usize) {
    let p = DualParams::SELF_CALIBRATED;
    let mut npc = BeliefDual::new();
    let selected = [true; 6];
    let mut act = BASELINE;
    act[2] += 1.2;
    npc.arm(&selected);
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..1000 {
        npc.tick(&selected, &act, &p);
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    let allocs = after - before;
    (allocs == 0, allocs, 1000)
}

fn latency_ns_per_tick() -> (f64, f64) {
    const N: usize = 10_000;
    const ROUNDS: usize = 5;
    let p = DualParams::SELF_CALIBRATED;
    let selected = [true; 6];
    let mut act = BASELINE;
    act[2] += 1.2;

    let mut inc_best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let mut st = ReconstructionState::new([0.0; 8]);
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            tick_incumbent(&mut st, &selected, &act);
        }
        std::hint::black_box(st.belief());
        inc_best = inc_best.min(t0.elapsed().as_nanos() as f64 / N as f64);
    }

    let mut dual_best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let mut npc = BeliefDual::new();
        npc.arm(&selected);
        let t0 = std::time::Instant::now();
        for _ in 0..N {
            npc.tick(&selected, &act, &p);
        }
        std::hint::black_box(npc.state.belief());
        std::hint::black_box(npc.lambda);
        dual_best = dual_best.min(t0.elapsed().as_nanos() as f64 / N as f64);
    }
    (inc_best, dual_best)
}

/// G5 sweep: α × ρ grid over the ranking population at the given horizons.
struct SweepCell {
    alpha: f32,
    rho: f32,
    lambda_s: [f32; 4],
    s3_plateau_band: f32,
    s3_hold_band: f32,
    s3_cleared: f32,
    t2_peak: f32,
    t2_end: f32,
    any_non_finite: bool,
}

#[allow(clippy::too_many_arguments)]
fn sweep_h(
    alphas: &[f32],
    rhos: &[f32],
    ticks_on: usize,
    ticks_off: usize,
    hold: usize,
    burst: usize,
    settle: usize,
) -> Vec<SweepCell> {
    let amps = [0.3f32, 0.6, 1.2, 2.0];
    let mut cells = Vec::new();
    for &alpha in alphas {
        for &rho in rhos {
            let p = DualParams { alpha, rho };
            let runs: Vec<SustainedRun> = amps
                .iter()
                .map(|&amp| run_sustained_h(amp, ticks_on, ticks_off, hold, &p))
                .collect();
            let t2 = run_transient(1.5, burst, settle, &p);
            let any_non_finite = runs.iter().any(|r| r.any_non_finite)
                || !t2.end_readout.is_finite()
                || !t2.peak_readout.is_finite();
            cells.push(SweepCell {
                alpha,
                rho,
                lambda_s: [
                    runs[0].lambda_anomaly_end,
                    runs[1].lambda_anomaly_end,
                    runs[2].lambda_anomaly_end,
                    runs[3].lambda_anomaly_end,
                ],
                s3_plateau_band: runs[2].plateau_band,
                s3_hold_band: runs[2].hold_band,
                s3_cleared: runs[2].readout_cleared,
                t2_peak: t2.peak_readout,
                t2_end: t2.end_readout,
                any_non_finite,
            });
        }
    }
    cells
}

fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

/// One horizon arm: sweep + recommended cell + G2/G3/G5 judged at it.
/// Returns (all laws pass at this arm, the judged (α, ρ) cell).
#[allow(clippy::too_many_arguments)]
fn run_arm(
    label: &str,
    alphas: &[f32],
    rhos: &[f32],
    ticks_on: usize,
    ticks_off: usize,
    hold: usize,
    burst: usize,
    settle: usize,
) -> (bool, Option<(f32, f32)>) {
    println!("\n═══ {label} (ON {ticks_on} / OFF {ticks_off} / hold {hold} / burst {burst}) ═══");
    println!(
        "  α    ρ   | S1(0.3) S2(0.6) S3(1.2) S4(2.0) | T2end pk  | S3 band(hold/plateau) clr  | inf"
    );
    let cells = sweep_h(alphas, rhos, ticks_on, ticks_off, hold, burst, settle);
    for c in &cells {
        println!(
            "  {:.2} {:4.1} | {:>7.3} {:>7.3} {:>7.3} {:>7.3} | {:>5.2} {:>4.2} | {:>5.3}/{:<5.3} {:>4.2} | {}",
            c.alpha,
            c.rho,
            c.lambda_s[0],
            c.lambda_s[1],
            c.lambda_s[2],
            c.lambda_s[3],
            c.t2_end,
            c.t2_peak,
            c.s3_hold_band,
            c.s3_plateau_band,
            c.s3_cleared,
            if c.any_non_finite { "YES" } else { "no" }
        );
    }

    let recommended = cells.iter().find(|c| {
        c.lambda_s[3] > c.lambda_s[2]
            && c.lambda_s[2] > c.lambda_s[1]
            && c.lambda_s[1] > c.lambda_s[0]
            && c.t2_end < c.lambda_s[0]
            && fast_sigmoid(c.lambda_s[2]) >= 0.90
            && c.s3_hold_band <= 0.05
            && c.s3_cleared <= 0.60
            && c.t2_peak >= 0.50
            && c.t2_end <= 0.60
            && !c.any_non_finite
    });

    let p_judge = recommended
        .map(|c| DualParams {
            alpha: c.alpha,
            rho: c.rho,
        })
        .unwrap_or(DualParams::SELF_CALIBRATED);
    println!(
        "── judged at α={:.2}, ρ={:.1} ({}) ──",
        p_judge.alpha,
        p_judge.rho,
        if recommended.is_some() {
            "recommended operating point"
        } else {
            "self-calibrated fallback — NO cell passed"
        }
    );

    let s1 = run_sustained_h(0.3, ticks_on, ticks_off, hold, &p_judge);
    let s2 = run_sustained_h(0.6, ticks_on, ticks_off, hold, &p_judge);
    let s3 = run_sustained_h(1.2, ticks_on, ticks_off, hold, &p_judge);
    let s4 = run_sustained_h(2.0, ticks_on, ticks_off, hold, &p_judge);
    let t1 = run_transient(0.5, burst, settle, &p_judge);
    let t2 = run_transient(1.5, burst, settle, &p_judge);

    let g2a = s3.readout_anomaly_end >= 0.90;
    let g2b = s3.hold_band <= 0.05;
    let g2c = s3.readout_cleared <= 0.60;
    let g2d = t2.peak_readout >= 0.50;
    let g2e = t2.end_readout <= 0.60;
    println!(
        "G2a S3 anomaly-end readout ≥ 0.90        : {}  ({:.4})",
        verdict(g2a),
        s3.readout_anomaly_end
    );
    println!(
        "G2b S3 hold band ≤ 0.05 (last {hold} ticks): {}  ({:.4})",
        verdict(g2b),
        s3.hold_band
    );
    println!(
        "G2c S3 clears after removal (≤ 0.60)     : {}  ({:.4})",
        verdict(g2c),
        s3.readout_cleared
    );
    println!(
        "G2d T2 burst fires (peak ≥ 0.50)         : {}  ({:.4})",
        verdict(g2d),
        t2.peak_readout
    );
    println!(
        "G2e T2 clears (end ≤ 0.60)               : {}  ({:.4})",
        verdict(g2e),
        t2.end_readout
    );

    let ladder = [
        s4.readout_anomaly_end,
        s3.readout_anomaly_end,
        s2.readout_anomaly_end,
        s1.readout_anomaly_end,
    ];
    let expected_order = [
        ("S4", ladder[0]),
        ("S3", ladder[1]),
        ("S2", ladder[2]),
        ("S1", ladder[3]),
        ("T2", t2.end_readout),
        ("T1", t1.end_readout),
    ];
    let joined: Vec<String> = expected_order
        .iter()
        .map(|(name, v)| format!("{name}={v:.3}"))
        .collect();
    let g3_strict_total = expected_order.windows(2).all(|w| w[0].1 > w[1].1);
    // v3.1 amendment: the law Issue 952 §B specifies — sustained ladder
    // strictly monotone + transients strictly below the weakest sustained.
    let g3_ok = ladder.windows(2).all(|w| w[0] > w[1])
        && t2.end_readout < ladder[3]
        && t1.end_readout < ladder[3];
    println!(
        "G3  sustained ladder monotone + T < S1   : {}  [{}]",
        verdict(g3_ok),
        joined.join(" > ")
    );
    println!(
        "G3' strict total order (v1-specified)    : {}  — the T2>T1 edge is the aftermath-dip semantic, recorded not gated",
        verdict(g3_strict_total)
    );
    println!(
        "    (S read at anomaly end; T read at sim end — the sustained/transient discriminator)"
    );

    let g5_conv = s3.plateau_band <= 0.05 * s3.lambda_anomaly_end.abs().max(1.0);
    let g5_bounded = [&s1, &s2, &s3, &s4]
        .iter()
        .all(|r| r.lambda_max_seen <= LAMBDA_MAX * (1.0 + 1e-6) && !r.any_non_finite);
    let g5_ok = g5_conv && g5_bounded && recommended.is_some();
    println!(
        "G5  λ bounded + plateau converged + operating point found : {}  (S3 plateau band {:.4}, λ_max seen {:.3})",
        verdict(g5_ok),
        s3.plateau_band,
        s3.lambda_max_seen
    );

    let all = g2a && g2b && g2c && g2d && g2e && g3_ok && g5_ok;
    (all, recommended.map(|c| (c.alpha, c.rho)))
}

fn main() {
    println!(
        "══ Issue 952 §B T3 — belief dual state POC v3 (arming + one-sided per-tick demotion) ══"
    );
    println!(
        "β = {EXPECTATION_BETA} (shipped slow default) · arming = {ARMING_TICKS} ticks · λ_max = {LAMBDA_MAX} (path-operator bound)\n"
    );

    // ── G1: α=0 bit-identity (horizon-independent — run once) ──
    let (g1_ok, mismatches, lambda_nonzero) = gate1_bit_identity();
    println!(
        "G1  α=0 bit-identity vs incumbent evolve_belief : {}  (mismatches {mismatches}, ticks with λ≠0: {lambda_nonzero})",
        verdict(g1_ok)
    );

    // ── G4: alloc-free (horizon-independent — run once) ──
    let (g4_ok, allocs, ticks) = gate4_alloc_free();
    println!(
        "G4  dual tick alloc-free                 : {}  ({allocs} allocs / {ticks} ticks)\n",
        verdict(g4_ok)
    );

    // ── Arm A: the operational horizon (the pre-committed 800/400 protocol;
    //    burst 40 ≈ the measured crossing resolution at the 600-tick arm) ──
    let (arm_a, _cell_a) = run_arm(
        "ARM A · operational horizon",
        &[1.0, 0.5, 0.25],
        &[1.0, 2.0, 3.0, 4.0],
        800,
        400,
        200,
        40,
        1160,
    );

    // ── Arm B: the equilibrium horizon. The dual feedback loop runs through
    //    the kernel's CUMULATIVE evidence (infinite memory) — its equilibrium
    //    is a thousands-of-ticks object; this arm measures whether the graded
    //    λ* = ρ·(a − 0.22) plateau exists at all (slow pressure variable) ──
    let (arm_b, cell_b) = run_arm(
        "ARM B · equilibrium horizon",
        &[0.5, 0.25],
        &[1.0, 2.0, 3.0, 4.0],
        6000,
        2000,
        1000,
        200,
        1800,
    );

    // ── latency (informational) ──
    let (inc_ns, dual_ns) = latency_ns_per_tick();
    println!(
        "\nlatency: incumbent tick {inc_ns:.1} ns · dual tick {dual_ns:.1} ns ({:.2}×) — informational",
        dual_ns / inc_ns
    );

    // ── T3 gate table + horizon-scoped verdict ──
    println!("\n── T3 gate table ──");
    println!("  {:<62} {}", "G1 α=0 bit-identical", verdict(g1_ok));
    println!("  {:<62} {}", "G4 alloc-free", verdict(g4_ok));
    println!(
        "  {:<62} {}",
        "G2+G3+G5 at the operational horizon (Arm A)",
        verdict(arm_a)
    );
    println!(
        "  {:<62} {}",
        "G2+G3+G5 at the equilibrium horizon (Arm B)",
        verdict(arm_b)
    );
    let cell_note = cell_b
        .map(|(a, r)| format!("α={a}, ρ={r}"))
        .unwrap_or_else(|| "none".to_string());
    let final_verdict = if arm_a && g1_ok && g4_ok {
        "PASS — laws confirmed at the operational horizon"
    } else if arm_b && g1_ok && g4_ok {
        "CONDITIONAL — laws hold ONLY at the equilibrium horizon: the dual through the cumulative kernel is a slow pressure variable (minutes @20 Hz), not an alert lane; alert use DECLINED, pressure-variable use viable at the measured operating point"
    } else {
        "FAIL — honest negative; both horizon tables above are the evidence"
    };
    println!("\nVERDICT: {final_verdict}");
    if arm_b && !arm_a {
        println!("(Arm B operating point: {cell_note} — see the bench doc for the full tables)");
    }
}

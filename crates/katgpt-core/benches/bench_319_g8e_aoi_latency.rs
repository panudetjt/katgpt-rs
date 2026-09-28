//! Plan 319 — G8e: AOI-Scored Pairwise Complementarity Latency Gate.
//!
//! Validates the perf budget claim made by Research 299's Super-GOAT gate Q3
//! ("product selling point"): that the Clifford wedge complementarity signal
//! (`riir-engine/cgsp_runtime/clifford_bridge.rs`, Plan 319 Phase 4 T4.3) can
//! be evaluated for **every NPC's AOI partner set every tick** within a
//! real-time game budget.
//!
//! # The gate
//!
//! Simulate a worst-case-density crowd: **1000 NPCs**, each with **20 AOI
//! partner candidates** (a generous AOI for a social hub zone), at **D=64**
//! (the CGSP `DEFAULT_BELIEF_DIRECTION_DIM`). Each tick, every NPC scores every partner via
//! `geometric_product_wedge_into` + L1 norm + sigmoid + tau gate — the exact
//! computation performed by `complementarity_target` in the shipped bridge.
//!
//! **PASS criterion:** mean tick wall-clock < **5 ms** AND zero allocations
//! per tick in the steady state (scratch reused). The 5 ms budget is the
//! 60Hz-frame headroom slice (~30% of a 16.67 ms tick) that a social-signal
//! bridge is allowed to consume; a 1Hz sandbox tick (1000 ms) gives 200×
//! headroom, so 5 ms is the *tight* (60Hz) budget.
//!
//! # Why this lives in katgpt-core (not riir-engine)
//!
//! The bench measures the **public primitive's** latency under the bridge's
//! workload pattern. `katgpt-core` cannot depend on `riir-engine` (the dep
//! arrow points the other way), so the bridge logic is reproduced inline —
//! `wedge → L1 → sigmoid → tau compare`. This is fair: the bench verifies the
//! primitive's perf, and the bridge is a thin wrapper that adds no hot-path
//! overhead beyond what's measured here.
//!
//! # Single-threaded baseline
//!
//! The shipped `complementarity_targets_batch` is single-threaded (sequential
//! per-NPC loop with one reused `CliffordScratch`). This bench matches that —
//! it reports the *single-threaded* latency. If the gate fails single-threaded,
//! the report includes the rayon-parallelizable note (the workload is
//! embarrassingly parallel; a `par_iter` split would recover near-linear
//! speedup), but promotion is gated on the single-threaded number because that
//! is what the shipped bridge does today.
//!
//! # Reference
//!
//! - Plan: [katgpt-rs/.plans/319_geometric_product_latent_interaction.md](../../../.plans/319_geometric_product_latent_interaction.md)
//! - Research: [katgpt-rs/.research/299_Clifford_Geometric_Product_Latent_Interaction.md](../../../.research/299_Clifford_Geometric_Product_Latent_Interaction.md)
//! - Bridge (private): `riir-ai/crates/riir-engine/src/cgsp_runtime/clifford_bridge.rs`
//! - Primitive: [`katgpt_core::linalg::geometric_product_wedge_into`]

use std::io::{self, Write};
use std::time::{Duration, Instant};

use katgpt_core::linalg::geometric_product_wedge_into;

// ─── Constants (mirror clifford_bridge.rs exactly) ─────────────────────────

/// Belief direction dimension (CGSP `DEFAULT_BELIEF_DIRECTION_DIM`).
const DIM: usize = 64;

/// Cyclic shifts for D=64 (clifford_bridge `DEFAULT_SHIFTS`).
const SHIFTS: &[usize] = &[1, 2, 4, 8, 16, 32];

/// Sigmoid sharpness (clifford_bridge `DEFAULT_BETA = 1.0`).
///
/// Calibrated for unit-norm HLA directions: the wedge L1 norm of an orthogonal
/// unit pair is O(1), not O(D), so beta must be O(1) (not 1/D) for the sigmoid
/// to produce a meaningful [0,1] complementarity score.
const BETA: f32 = 1.0;

/// Complementarity threshold (clifford_bridge default `tau_complementarity`).
/// Strictly above `sigmoid(0) = 0.5` so only genuinely orthogonal pairs pass.
const TAU: f32 = 0.6;

// ─── Crowd simulation constants ────────────────────────────────────────────

/// Number of NPCs in the simulated zone (worst-case social hub density).
const NPC_COUNT: usize = 1000;

/// AOI partner candidates per NPC. 20 is a generous AOI for a social hub;
/// real AOIs are typically smaller and spatially coherent (cheaper).
const PARTNERS_PER_NPC: usize = 20;

/// Pairs scored per tick = NPC_COUNT × PARTNERS_PER_NPC.
const PAIRS_PER_TICK: usize = NPC_COUNT * PARTNERS_PER_NPC;

/// Measured ticks (after warmup). 100 ticks × 20K pairs = 2M wedge calls —
/// more than enough for a stable mean (the G4 micro-bench used 100K iters
/// of a single call; this is 20× more total work, spread across realistic
/// cache-miss patterns). Kept modest so the full bench runs in <10s.
const MEASURED_TICKS: usize = 100;

/// Warmup ticks (cache + branch predictor stabilization, not measured).
const WARMUP_TICKS: usize = 20;

/// G8e PASS budget: mean tick wall-clock must be below this.
const TICK_BUDGET_MS: f64 = 5.0;

// ─── Deterministic RNG (xorshift32, same as bench_319) ─────────────────────

struct Rng {
    state: u32,
}

impl Rng {
    #[inline]
    fn new(seed: u32) -> Self {
        // Avoid the xorshift32 fixed point at state=0.
        Self {
            state: if seed == 0 { 0xDEAD_BEEF } else { seed },
        }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0f32 / (1u32 << 24) as f32)
    }

    /// Standard-normal f32 via Box-Muller (cos branch).
    #[inline]
    fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

// ─── L2 normalize a slice in place ─────────────────────────────────────────

#[inline]
fn normalize_in_place(v: &mut [f32]) {
    let mut sum_sq = 0.0f32;
    for &x in v.iter() {
        sum_sq += x * x;
    }
    let norm = sum_sq.sqrt().max(1e-10);
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

// ─── Sigmoid (matches katgpt_core::cgsp::types::sigmoid) ───────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

#[inline]
fn alloc_count() -> usize {
    ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

// ─── Crowd state ───────────────────────────────────────────────────────────

/// Per-NPC HLA direction (unit-normalized 64-dim latent).
struct Crowd {
    /// `directions[npc][0..DIM]` — unit-norm HLA direction.
    directions: Vec<f32>,
    /// `partners[npc * PARTNERS_PER_NPC + p]` — partner NPC index.
    partners: Vec<usize>,
}

impl Crowd {
    fn new(seed: u32) -> Self {
        let mut rng = Rng::new(seed);

        // Generate unit-normalized Gaussian directions (matches the CGSP
        // direction-pool distribution: isotropic in 64-dim).
        let mut directions = vec![0.0f32; NPC_COUNT * DIM];
        for npc in 0..NPC_COUNT {
            let base = npc * DIM;
            for d in 0..DIM {
                directions[base + d] = rng.gaussian();
            }
            normalize_in_place(&mut directions[base..base + DIM]);
        }

        // Assign each NPC PARTNERS_PER_NPC random partners (excluding self).
        // Real AOI is spatially coherent; for a perf bench the partner pattern
        // does not affect wedge cost — only the count does.
        let mut partners = vec![0usize; PAIRS_PER_TICK];
        for npc in 0..NPC_COUNT {
            let base = npc * PARTNERS_PER_NPC;
            for p in 0..PARTNERS_PER_NPC {
                let mut partner = rng.next_u32() as usize % NPC_COUNT;
                if partner == npc {
                    partner = (partner + 1) % NPC_COUNT;
                }
                partners[base + p] = partner;
            }
        }

        Self {
            directions,
            partners,
        }
    }

    /// Simulate one tick: every NPC scores every AOI partner.
    ///
    /// Returns the number of complementary pairs (those whose complementarity
    /// score exceeded `TAU`). The scratch buffers are reused across all pairs
    /// — zero allocation in the steady state.
    ///
    /// This is a faithful inline reproduction of `clifford_bridge::
    /// complementarity_target`: `wedge → L1 norm → sigmoid(beta * l1) → tau`.
    #[inline]
    fn simulate_tick(
        &self,
        scratch_wedge: &mut [f32],
        scratch_su: &mut [f32],
        scratch_sv: &mut [f32],
    ) -> usize {
        let mut complementary = 0usize;
        for npc in 0..NPC_COUNT {
            let h_self_base = npc * DIM;
            let partner_base = npc * PARTNERS_PER_NPC;
            for p in 0..PARTNERS_PER_NPC {
                let partner = self.partners[partner_base + p];
                let h_partner_base = partner * DIM;

                geometric_product_wedge_into(
                    &self.directions[h_self_base..h_self_base + DIM],
                    &self.directions[h_partner_base..h_partner_base + DIM],
                    DIM,
                    SHIFTS,
                    scratch_wedge,
                    scratch_su,
                    scratch_sv,
                );

                // L1 norm of the wedge output (anti-symmetric structure mass).
                let wedge_l1: f32 = scratch_wedge[..DIM].iter().map(|x| x.abs()).sum();

                let complementarity = sigmoid(BETA * wedge_l1);
                if complementarity > TAU {
                    complementary += 1;
                }
            }
        }
        complementary
    }
}

// ─── Percentile helper ─────────────────────────────────────────────────────

/// Compute the `q`-th percentile (q in [0, 100]) of a non-empty sorted slice.
fn percentile(sorted: &[Duration], q: f64) -> Duration {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=100.0).contains(&q));
    let idx = ((q / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Plan 319 — G8e: AOI Pairwise Complementarity Latency       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  Crowd: {NPC_COUNT} NPCs × {PARTNERS_PER_NPC} partners = {PAIRS_PER_TICK} pairs/tick, D={DIM}"
    );
    println!(
        "  Shifts: {:?} (|S|={}), beta={}, tau={}",
        SHIFTS,
        SHIFTS.len(),
        BETA,
        TAU
    );
    println!(
        "  Budget: mean tick < {TICK_BUDGET_MS:.1} ms ({MEASURED_TICKS} ticks measured, {WARMUP_TICKS} warmup)"
    );
    println!();
    let _ = io::stdout().flush();

    // Build the crowd. Allocations here are NOT measured.
    let crowd = Crowd::new(0x68EA_0149); // "G8e AOI" mnemonic (valid hex)

    // Pre-allocate scratch (reused across ALL pairs and ALL ticks — zero
    // steady-state allocation, matching `complementarity_targets_batch`).
    let mut scratch_wedge = vec![0.0f32; DIM];
    let mut scratch_su = vec![0.0f32; DIM];
    let mut scratch_sv = vec![0.0f32; DIM];

    // Pre-allocate the per-tick duration log (capacity = MEASURED_TICKS so
    // push never reallocs during measurement).
    let mut tick_durations: Vec<Duration> = Vec::with_capacity(MEASURED_TICKS);

    // ── Warmup ──
    let mut warmup_complementary = 0usize;
    let warmup_start = Instant::now();
    for _ in 0..WARMUP_TICKS {
        warmup_complementary =
            crowd.simulate_tick(&mut scratch_wedge, &mut scratch_su, &mut scratch_sv);
    }
    let warmup_elapsed = warmup_start.elapsed();
    let _ = warmup_complementary;
    eprintln!(
        "  [warmup: {} ticks in {:.1} ms, {:.3} ms/tick]",
        WARMUP_TICKS,
        warmup_elapsed.as_secs_f64() * 1.0e3,
        warmup_elapsed.as_secs_f64() * 1.0e3 / WARMUP_TICKS as f64
    );

    // ── Reset alloc counter just before measurement ──
    ALLOC_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    let alloc_before = alloc_count();

    // ── Measured ticks ──
    let mut last_complementary = 0usize;
    let measure_start = Instant::now();
    for _ in 0..MEASURED_TICKS {
        let tick_start = Instant::now();
        last_complementary =
            crowd.simulate_tick(&mut scratch_wedge, &mut scratch_su, &mut scratch_sv);
        tick_durations.push(tick_start.elapsed());
    }
    let total_elapsed = measure_start.elapsed();

    let alloc_after = alloc_count();
    let alloc_delta = alloc_after.saturating_sub(alloc_before);

    // ── Stats ──
    tick_durations.sort_unstable();
    let mean_ns = total_elapsed.as_nanos() as f64 / MEASURED_TICKS as f64;
    let mean_ms = mean_ns / 1.0e6;
    let p50 = percentile(&tick_durations, 50.0);
    let p99 = percentile(&tick_durations, 99.0);
    let max = tick_durations[tick_durations.len() - 1];
    let min = tick_durations[0];
    let per_pair_ns = mean_ns / PAIRS_PER_TICK as f64;
    let hit_rate = last_complementary as f64 / PAIRS_PER_TICK as f64;

    println!("── Results ──");
    println!(
        "  mean tick:    {:>8.3} ms  (target < {:.1} ms)  {}",
        mean_ms,
        TICK_BUDGET_MS,
        if mean_ms < TICK_BUDGET_MS {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!("  min tick:     {:>8.3} ms", min.as_secs_f64() * 1.0e3);
    println!("  p50 tick:     {:>8.3} ms", p50.as_secs_f64() * 1.0e3);
    println!("  p99 tick:     {:>8.3} ms", p99.as_secs_f64() * 1.0e3);
    println!("  max tick:     {:>8.3} ms", max.as_secs_f64() * 1.0e3);
    println!("  per-pair:     {per_pair_ns:>8.1} ns  ({PAIRS_PER_TICK} pairs/tick)");
    println!(
        "  allocs/tick:  {:>8}   (target 0)  {}",
        alloc_delta / MEASURED_TICKS,
        if alloc_delta == 0 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!(
        "  complementarity hit rate: {:>5.1}%  ({}/{} pairs above tau={})",
        hit_rate * 100.0,
        last_complementary,
        PAIRS_PER_TICK,
        TAU
    );
    println!();

    // ── Verdict ──
    let perf_pass = mean_ms < TICK_BUDGET_MS;
    let alloc_pass = alloc_delta == 0;

    println!("════════════════════════════════════════════════════════════════");
    println!(
        "  G8e VERDICT:  perf={}  alloc={}",
        if perf_pass { "PASS" } else { "FAIL" },
        if alloc_pass { "PASS" } else { "FAIL" }
    );
    if perf_pass && alloc_pass {
        println!("  → G8e PASS. AOI-scored pairwise complementarity fits the 5 ms");
        println!(
            "    60Hz budget with {:.2}× headroom ({:.3} ms / {:.1} ms).",
            TICK_BUDGET_MS / mean_ms,
            mean_ms,
            TICK_BUDGET_MS
        );
        println!("  → Unblocks G8c/G8d runtime sims (perf budget is non-blocking).");
    } else {
        if !perf_pass {
            println!(
                "  → perf FAIL: mean {:.3} ms exceeds {:.1} ms budget by {:.2}×.",
                mean_ms,
                TICK_BUDGET_MS,
                mean_ms / TICK_BUDGET_MS
            );
            println!(
                "    The workload ({PAIRS_PER_TICK} pairs × ~{per_pair_ns:.0} ns/pair) is embarrassingly"
            );
            println!("    parallel; a rayon `par_iter` over NPC rows would recover");
            println!(
                "    near-linear speedup (8× on 8 cores → {:.2} ms). But the",
                mean_ms / 8.0
            );
            println!("    shipped `complementarity_targets_batch` is single-threaded,");
            println!("    so the gate is gated on the single-threaded number.");
        }
        if !alloc_pass {
            println!(
                "  → alloc FAIL: {alloc_delta} allocs across {MEASURED_TICKS} ticks (expected 0)."
            );
            println!("    Scratch buffers must be reused, not reallocated per pair.");
        }
    }
    println!("════════════════════════════════════════════════════════════════");

    if !(perf_pass && alloc_pass) {
        std::process::exit(1);
    }
}

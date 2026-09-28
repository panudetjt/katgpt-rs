//! Plan 571 Phase 3 — Lonely Runner Conjecture (LRC) demonstration.
//!
//! Seven entities with integer cycle speeds `{1, 2, 3, 4, 5, 6, 7}` advance
//! on a shared phase circle. For each entity, we find the tick where it is
//! *loneliest* (maximum `phase_separation` from every peer) and confirm the
//! LRC guarantee: every entity hits `phase_separation ≥ 1/N` at least once.
//!
//! This is the runnable companion to the `g1_lrc_bound_n7` unit test — same
//! setup, but prints the trajectory so the theorem is visible.
//!
//! **Run:** `cargo run --example phase_separation_demo --features phase_separation`
//!
//! **Source:** Barajas & Serra, *The Lonely Runner with Seven Runners*,
//! [arXiv:0710.4495](https://arxiv.org/abs/0710.4495) (2007). Proven for
//! N ≤ 7; conjectured (open) for N > 7.

use katgpt_core::{from_speeds_and_tick, phase_separation_sorted};

// ── Setup ─────────────────────────────────────────────────────────────────

/// Seven runners, integer speeds, gcd = 1. This is the canonical LRC instance
/// proven by Barajas & Serra (2007).
const SPEEDS: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];

/// `lcm(1..=7) = 420` — the joint orbit period. Scanning `k ∈ [0, 420)`
/// covers the entire reachable configuration (since `gcd(1, 420) = 1`, the
/// orbit has full period).
const PERIOD: u32 = 420;

/// The LRC bound for N=7 entities: every entity is guaranteed a tick where
/// `phase_separation ≥ 1/7 ≈ 0.142857`. Discrete sampling at granularity
/// `1/420` means the true continuous lonely time may fall between two sample
/// points, so we accept a small epsilon slack (`5 / 420 ≈ 0.012`).
const BOUND: f32 = 1.0 / SPEEDS.len() as f32; // 1/7
const EPS: f32 = 5.0 / PERIOD as f32; // discrete-sampling slack

fn main() {
    let n = SPEEDS.len();

    // Caller-provided scratch — zero allocation across the whole scan (G4).
    let mut phases = [0.0_f32; SPEEDS.len()];
    let mut scratch_perm = [0_usize; SPEEDS.len()];
    let mut sep = [0.0_f32; SPEEDS.len()];

    // Per-entity: the loneliest tick observed + the separation at that tick.
    let mut max_sep = [0.0_f32; SPEEDS.len()];
    let mut loneliest_tick = [0_u64; SPEEDS.len()];

    // Scan the full discrete orbit k = 0..PERIOD.
    for k in 0..PERIOD as u64 {
        // Raw time-phase bridge (sync-safe): φ_i(k) = (s_i · k mod P) / P.
        from_speeds_and_tick(&SPEEDS, k, PERIOD, &mut phases);
        // O(N log N) production scan — sort + adjacent-neighbor.
        phase_separation_sorted(&phases, &mut scratch_perm, &mut sep);
        for i in 0..n {
            if sep[i] > max_sep[i] {
                max_sep[i] = sep[i];
                loneliest_tick[i] = k;
            }
        }
    }

    // ── Report ────────────────────────────────────────────────────────────
    println!("Lonely Runner Conjecture — N = {n} entities");
    println!("Speeds: {SPEEDS:?}");
    println!("Orbit period P = lcm(1..=7) = {PERIOD}");
    println!("LRC bound: 1/N = 1/{n} ≈ {BOUND:.6} (eps slack ±{EPS:.6})");
    println!();
    println!(
        "{:<10}{:<16}{:<16}MaxSeparation",
        "Entity", "Speed", "LoneliestTick"
    );
    println!("{}", "-".repeat(58));

    let mut all_hit_bound = true;
    for i in 0..n {
        let hit = max_sep[i] >= BOUND - EPS;
        if !hit {
            all_hit_bound = false;
        }
        let marker = if hit { " ✓" } else { " ✗ BELOW BOUND" };
        println!(
            "{:<10}{:<16}{:<16}{:.6}{}",
            format!("entity {}", i),
            format!("s = {}", SPEEDS[i]),
            format!("k = {}", loneliest_tick[i]),
            max_sep[i],
            marker,
        );
    }

    println!();
    if all_hit_bound {
        println!(
            "✓ LRC CONFIRMED: all {} entities reached phase_separation ≥ {:.6}",
            n,
            BOUND - EPS
        );
    } else {
        println!(
            "✗ LRC VIOLATED: at least one entity stayed below {:.6} across the full orbit",
            BOUND - EPS
        );
        std::process::exit(1);
    }
}

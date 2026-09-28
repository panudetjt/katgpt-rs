//! Issue 663 T7 — SwitchCostTable demo: toy 5-mode FSM telemetry → directed
//! switch-difficulty table → "which incoming switch is hardest" readout.
//!
//! Simulates a behavior-FSM agent (Idle/Hunt/Flee/Tame/Sleep) whose switches
//! carry a designed difficulty structure — the canonical hard case is
//! **Flee → Hunt** (carry-over: the agent keeps fleeing instead of
//! re-engaging), mirroring the Issue-054 border-piling failure mode.
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features switch_cost --example switch_cost_demo
//! ```

#![cfg(feature = "switch_cost")]

use fastrand::Rng;
use katgpt_core::switch_cost::{DEFAULT_ALPHA, FactorizedSwitchCost, SwitchCostTable};

const IDLE: usize = 0;
const HUNT: usize = 1;
const FLEE: usize = 2;
const TAME: usize = 3;
const SLEEP: usize = 4;
const MODES: [&str; 5] = ["Idle", "Hunt", "Flee", "Tame", "Sleep"];

/// Designed ground truth: solo success rates + a hard Flee→Hunt switch
/// (carry-over). Everything else is moderate.
const SOLO_P: [f32; 5] = [0.95, 0.80, 0.90, 0.60, 0.99];

/// Pair success rate for the ordered switch `a → b` (designed, not learned).
fn switch_p(a: usize, b: usize) -> f32 {
    let base = 0.5 * (SOLO_P[a] + SOLO_P[b]);
    match (a, b) {
        // The carry-over failure: re-engaging after fleeing is HARD.
        (FLEE, HUNT) => base * 0.25,
        // Mild difficulty: hunt → tame loses focus; flee → tame misses cues.
        (HUNT, TAME) | (FLEE, TAME) => base * 0.7,
        // Same-mode continuation is easy.
        (x, y) if x == y => base * 0.95,
        _ => base * 0.85,
    }
}

fn main() {
    let mut rng = Rng::with_seed(663);
    let mut table = SwitchCostTable::<5>::new(DEFAULT_ALPHA);
    // Families: {execution: Idle, Hunt, Flee} vs {social: Tame, Sleep} —
    // the factorized variant's coarse partition.
    let mut fact = FactorizedSwitchCost::<5, 2>::new([0, 0, 0, 1, 1], DEFAULT_ALPHA);

    // ── Telemetry ingest ─────────────────────────────────────────────────
    for (m, &p) in SOLO_P.iter().enumerate() {
        for _ in 0..400 {
            let ok = rng.f32() < p;
            table.record_solo(m, ok);
            fact.record_solo(m, ok);
        }
    }
    for a in 0..5 {
        for b in 0..5 {
            let p = switch_p(a, b);
            for _ in 0..400 {
                let ok = rng.f32() < p;
                table.record_switch(a, b, ok);
                fact.record_switch(a, b, ok);
            }
        }
    }

    println!("SwitchCostTable demo — 5-mode FSM, 400 trials/pair (α = {DEFAULT_ALPHA})\n");

    // ── Directed pair table ──────────────────────────────────────────────
    println!("Directed SkE (rows = from, cols = to):");
    print!("        ");
    for name in MODES.iter() {
        print!("{name:>7}");
    }
    println!();
    for (a, name_a) in MODES.iter().enumerate() {
        print!("{name_a:>7}");
        for b in 0..5 {
            print!("{:>7.2}", table.ske(a, b));
        }
        println!();
    }

    // ── Hardest incoming switch per mode (the F1 trigger preview) ───────
    println!("\nHardest INCOMING switch per mode (mean SkE over predecessors):");
    for (b, name_b) in MODES.iter().enumerate() {
        let mut best = (f32::MIN, 0usize);
        let mut sum = 0.0f32;
        for a in 0..5 {
            if a == b {
                continue;
            }
            let v = table.ske(a, b);
            sum += v;
            if v > best.0 {
                best = (v, a);
            }
        }
        let mean = sum / 4.0;
        println!(
            "  {:>5}: hardest from {:>5} (SkE {:.2}, mean incoming {:.2})",
            name_b, MODES[best.1], best.0, mean
        );
    }

    // ── Warm-up gating (Research 484 §6.1) ───────────────────────────────
    println!("\nWarm-up gate (ske_if_armed, min 50 pair trials):");
    for (a, b) in [(FLEE, HUNT), (IDLE, SLEEP)] {
        match table.ske_if_armed(a, b, 50) {
            Some(v) => println!("  {:>5} → {:>5}: armed, SkE = {v:.2}", MODES[a], MODES[b]),
            None => println!("  {:>5} → {:>5}: not armed yet", MODES[a], MODES[b]),
        }
    }

    // ── Sequence entropy (quest-difficulty dial, Eq. 4) ─────────────────
    let calm_day = [IDLE, HUNT, IDLE, TAME, SLEEP];
    let panic_day = [IDLE, HUNT, FLEE, HUNT, FLEE, TAME, SLEEP];
    println!("\nSequence entropy (Eq. 4):");
    println!("  calm routine  {:>5.2}", table.sequence_entropy(&calm_day));
    println!(
        "  panic routine {:>5.2}",
        table.sequence_entropy(&panic_day)
    );

    // ── Factorized (Eq. 7) cross-check ───────────────────────────────────
    println!("\nFactorized (2 families) vs exact on the designed hard pair:");
    println!(
        "  Flee → Hunt: exact {:+.2}  factorized {:+.2}",
        table.ske(FLEE, HUNT),
        fact.ske(FLEE, HUNT)
    );
}

//! Bench 684 — Issue 692 T5 GOAT: head-to-head prover selection
//! (strength-ranked vs D+Al-ranked) on a controlled PAV harness
//! (Research 509 §5 defend-wrong obligation; arXiv:2410.08146 Setlur et al.).
//!
//! The falsifiable question: does ranking provers by Theorem 3.1's
//! complementarity bound γ·(D+Al) — instead of by strength (mean success) —
//! pick the prover that actually improves a cross-state beam search
//! r_eff = Q^π + α·A^μ (the paper's §5 search result, α ∈ [0.2, 0.6])?
//!
//! Three arms (Research 509 §5's exact shape):
//! - **A0 frozen baseline / shipped analog** — retention by raw Q̂^π alone
//!   (the strength-only selector the stack ships everywhere: dd_tree
//!   `WidthSelectionMode::BestQ`, drafter mean-acceptance ranking, Elo).
//! - **A1 strength arm** — wire the prover with the highest mean logged
//!   success (the paper's documented failure mode: a flat 0.95-solver has
//!   top strength but zero distinguishability).
//! - **A2 paper arm** — wire the prover with the highest
//!   `theorem_bound(D, Al, γ)` computed from the SAME logs by T1's
//!   estimators (the predicted-gain pre-gate).
//!
//! Controlled harness (all seeded, deterministic, modelless):
//! - Ground truth θ(s,a) uniform iid over 64 states × 8 actions.
//! - Base log Q̂^π(s,a): mean of n_mc=16 Bernoulli(θ) draws.
//! - Prover logs (independent RNG streams, same (s,a) support):
//!   `strong_flat` p=0.95 const (solves any prefix → A^μ ≈ 0),
//!   `peer_independent` p=θ (equal-competence peer, fresh MC noise),
//!   `intermediate_ranked` p=0.30+0.50·within-state-θ-rank (weaker overall,
//!   complementary profile), `anti_aligned` p mirrored (informational).
//! - End task: retain the top-32 of the 512 (s,a) pairs by arm score;
//!   quality = mean true θ of the retained set (cross-state retention —
//!   the direction where per-state centering is NOT rank-invariant,
//!   Research 509 §2.2).
//!
//! Gates (G1/G2/G5 per-seed — their margins are 10–50×; G3/G4 on the
//! 16-seed aggregate mean + ≥75% cell win-rate — per-(seed × α) margins
//! carry 32-slot selection quantization noise of ~±0.003):
//! - G1 determinism: identical tables on re-run per seed.
//! - G2 the inversion is real: strength-pick == strong_flat AND the bound
//!   ranks strong_flat far below the bound-pick (pre-gate flags it ≈ no
//!   gain: bound(strong) < 0.25 · bound(pick)).
//! - G3 the headline: quality(A2) > quality(A1) — complementarity-ranked
//!   pick beats strength-ranked pick at equal wiring cost.
//! - G4 no-harm at adequate tilt: quality(A2) > quality(A0) at α ∈ {0.4, 0.6}
//!   (the α=0.2 row is reported, not gated: at tilt 0.2 a peer-strength
//!   prover's n_mc=16 MC noise is not paid back — BOTH wired arms sit below
//!   the no-prover baseline there, while selection stays correct).
//! - G5 direction sensitivity (soft table row): bound(anti_aligned) <
//!   bound(peer_independent) — anti-alignment drags the bound down.
//!
//! Scope (honest): a controlled harness validates the MECHANISM (the bound
//! predicts wired gain; strength does not), not real-world drafter
//! superiority — real-log head-to-heads are riir-train/riir-poc territory.
//!
//! Run:
//! ```bash
//! cargo bench --features prover_selection --bench bench_684_prover_selection_goat -- --nocapture
//! ```

#![cfg(feature = "prover_selection")]

use katgpt_core::prover_selection::{alignment, distinguishability, selection_gate, theorem_bound};

const N_STATES: usize = 64;
const N_ACTIONS: usize = 8;
const N_MC: u32 = 16;
const BEAM: usize = 32;
const GAMMA: f32 = 1.0;
const ALPHAS: [f32; 3] = [0.2, 0.4, 0.6];
/// 16 seeds (house PoC convention): per-(seed × α) cells carry 32-slot
/// selection quantization noise (~±0.003), so quality gates run on the
/// aggregate mean + win-rate, not on every cell.
const SEEDS: [u64; 16] = [
    42, 1337, 20260827, 7, 91, 31337, 2024, 65537, 7919, 104729, 1299709, 15485863, 32452843,
    49979687, 67867967, 86028121,
];

/// xorshift64* — platform-independent u64 arithmetic, deterministic.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Mean of `n` Bernoulli(p) draws — a logged MC estimate.
    fn bernoulli_mean(&mut self, p: f32, n: u32) -> f32 {
        let mut hits = 0u32;
        for _ in 0..n {
            if self.next_f32() < p {
                hits += 1;
            }
        }
        hits as f32 / n as f32
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ProverId {
    StrongFlat,
    PeerIndependent,
    IntermediateRanked,
    AntiAligned,
}

const PROVER_NAMES: [&str; 4] = [
    "strong_flat",
    "peer_independent",
    "intermediate_ranked",
    "anti_aligned",
];

/// Prover success probability at (s,a) given the ground-truth table and the
/// within-state θ ranks (for the rank-shaped provers).
fn prover_p(id: ProverId, theta: &[&[f32]], s: usize, a: usize, rank: usize) -> f32 {
    match id {
        // Solves from any prefix — the strength trap (A^μ ≈ 0 everywhere).
        ProverId::StrongFlat => 0.95,
        // Equal-competence peer: fresh MC samples of the same θ signal.
        ProverId::PeerIndependent => theta[s][a],
        // Weaker overall (~0.55 mean) but within-state profile tracks θ rank.
        ProverId::IntermediateRanked => 0.30 + 0.50 * rank as f32 / (N_ACTIONS - 1) as f32,
        // Mirrored: succeeds where the base's action is weak (informational).
        ProverId::AntiAligned => 0.80 - 0.50 * rank as f32 / (N_ACTIONS - 1) as f32,
    }
}

/// One full harness run at a seed → all per-prover stats + per-arm qualities
/// at every α. Deterministic in (seed, α).
struct RunResult {
    strength: [f32; 4],
    d: [f32; 4],
    al: [f32; 4],
    bound: [f32; 4],
    gate: [f32; 4],
    quality_baseline: f32,
    /// [prover][alpha] wired quality.
    quality_wired: [[f32; 3]; 4],
}

fn run_seed(seed: u64) -> RunResult {
    // Ground truth + within-state ranks (ties broken by index — deterministic).
    let mut rng = Rng::new(seed);
    let mut theta_rows = vec![vec![0.0f32; N_ACTIONS]; N_STATES];
    for row in &mut theta_rows {
        for v in row.iter_mut() {
            *v = rng.next_f32();
        }
    }
    let theta: Vec<Vec<f32>> = theta_rows.clone();
    let theta_ref: Vec<&[f32]> = theta.iter().map(|r| r.as_slice()).collect();

    // Within-state ranks of θ (0 = weakest action in the state).
    let mut ranks = vec![vec![0usize; N_ACTIONS]; N_STATES];
    for s in 0..N_STATES {
        let mut idx: Vec<usize> = (0..N_ACTIONS).collect();
        idx.sort_by(|&a, &b| theta[s][a].total_cmp(&theta[s][b]));
        for (r, &a) in idx.iter().enumerate() {
            ranks[s][a] = r;
        }
    }

    // Base log: MC means at rate θ.
    let mut base_rows = vec![vec![0.0f32; N_ACTIONS]; N_STATES];
    for s in 0..N_STATES {
        for a in 0..N_ACTIONS {
            base_rows[s][a] = rng.bernoulli_mean(theta[s][a], N_MC);
        }
    }
    let base: Vec<Vec<f32>> = base_rows;
    let base_ref: Vec<&[f32]> = base.iter().map(|r| r.as_slice()).collect();

    // Prover logs — independent RNG streams per (seed, prover).
    let ids = [
        ProverId::StrongFlat,
        ProverId::PeerIndependent,
        ProverId::IntermediateRanked,
        ProverId::AntiAligned,
    ];
    let mut logs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(4);
    for (i, &id) in ids.iter().enumerate() {
        let mut prng = Rng::new(seed ^ (0x5DEE_CE66_D000_0000 + i as u64));
        let mut log = vec![vec![0.0f32; N_ACTIONS]; N_STATES];
        for s in 0..N_STATES {
            for a in 0..N_ACTIONS {
                let p = prover_p(id, &theta_ref, s, a, ranks[s][a]);
                log[s][a] = prng.bernoulli_mean(p, N_MC);
            }
        }
        logs.push(log);
    }

    // Per-prover stats from T1's estimators on the SAME logs.
    let mut strength = [0.0f32; 4];
    let mut d = [0.0f32; 4];
    let mut al = [0.0f32; 4];
    let mut bound = [0.0f32; 4];
    let mut gate = [0.0f32; 4];
    for i in 0..4 {
        let log_ref: Vec<&[f32]> = logs[i].iter().map(|r| r.as_slice()).collect();
        let mut sum = 0.0f32;
        for row in &logs[i] {
            sum += row.iter().sum::<f32>();
        }
        strength[i] = sum / (N_STATES * N_ACTIONS) as f32;
        d[i] = distinguishability(&base_ref, &log_ref);
        al[i] = alignment(&base_ref, &log_ref);
        bound[i] = theorem_bound(d[i], al[i], GAMMA);
        gate[i] = selection_gate(d[i], al[i], GAMMA);
    }

    // Wired quality: top-BEAM by Q̂^π + α·(Q̂^μ − V̂^μ(s)) → mean true θ.
    let wired_quality = |log: &[Vec<f32>], alpha: f32| -> f32 {
        let mut scored: Vec<(f32, f32)> = Vec::with_capacity(N_STATES * N_ACTIONS);
        for s in 0..N_STATES {
            let v = log[s].iter().sum::<f32>() / N_ACTIONS as f32;
            for a in 0..N_ACTIONS {
                scored.push((base[s][a] + alpha * (log[s][a] - v), theta[s][a]));
            }
        }
        scored.sort_by(|x, y| y.0.total_cmp(&x.0));
        scored[..BEAM].iter().map(|x| x.1).sum::<f32>() / BEAM as f32
    };

    let quality_baseline = wired_quality(&base, 0.0);
    let mut quality_wired = [[0.0f32; 3]; 4];
    for (i, log) in logs.iter().enumerate() {
        for (j, &alpha) in ALPHAS.iter().enumerate() {
            quality_wired[i][j] = wired_quality(log, alpha);
        }
    }

    RunResult {
        strength,
        d,
        al,
        bound,
        gate,
        quality_baseline,
        quality_wired,
    }
}

fn main() {
    println!("=== Bench 684 — Issue 692 T5: prover selection GOAT (D+Al vs strength) ===\n");

    // Aggregates over seeds (quality gates run on these — per-cell margins
    // carry 32-slot quantization noise; selection/stats gates stay per-seed).
    let mut sum_q0 = 0.0f32;
    let mut sum_q1 = [0.0f32; 3]; // strength arm per α
    let mut sum_q2 = [0.0f32; 3]; // paper arm per α
    let mut wins_21 = [0u32; 3]; // A2 > A1 cells per α
    let mut wins_20 = [0u32; 3]; // A2 > A0 cells per α

    for &seed in &SEEDS {
        let r = run_seed(seed);

        // Selections.
        let strength_pick = r
            .strength
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap();
        let bound_pick = r
            .bound
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap();

        println!("--- seed {seed} ---");
        println!(
            "{:<20} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "prover", "strength", "D", "Al", "bound", "gate"
        );
        for (i, name) in PROVER_NAMES.iter().enumerate() {
            println!(
                "{:<20} {:>9} {:>9} {:>9} {:>9} {:>9}",
                name, r.strength[i], r.d[i], r.al[i], r.bound[i], r.gate[i]
            );
        }
        println!(
            "strength-pick: {}   bound-pick: {}   (γ={GAMMA})",
            PROVER_NAMES[strength_pick], PROVER_NAMES[bound_pick]
        );
        println!(
            "retained-beam mean θ ({} of {} pairs):",
            BEAM,
            N_STATES * N_ACTIONS
        );
        println!("  A0 baseline (Q̂^π only):      {:.4}", r.quality_baseline);
        for (j, &alpha) in ALPHAS.iter().enumerate() {
            println!(
                "  α={:.1}: A1 strength→{} {:.4} | A2 bound→{} {:.4}",
                alpha,
                PROVER_NAMES[strength_pick],
                r.quality_wired[strength_pick][j],
                PROVER_NAMES[bound_pick],
                r.quality_wired[bound_pick][j]
            );
        }
        println!();

        // G1 determinism — re-run must be bit-identical.
        let r2 = run_seed(seed);
        let identical = r.strength == r2.strength
            && r.d == r2.d
            && r.al == r2.al
            && r.bound == r2.bound
            && r.gate == r2.gate
            && r.quality_baseline == r2.quality_baseline
            && r.quality_wired == r2.quality_wired;
        assert!(identical, "G1 FAIL seed {seed}: re-run diverged");

        // G2 the inversion is real + the pre-gate flags the strength winner
        // (per-seed: these margins are 10–50×, robust).
        assert_eq!(
            strength_pick, 0,
            "G2a FAIL seed {seed}: strength ranking must pick strong_flat (got {})",
            PROVER_NAMES[strength_pick]
        );
        assert_ne!(
            bound_pick, 0,
            "G2b FAIL seed {seed}: bound ranking must NOT pick strong_flat"
        );
        assert!(
            r.bound[0] < 0.25 * r.bound[bound_pick],
            "G2c FAIL seed {seed}: pre-gate must flag strong_flat far below the pick ({} vs {})",
            r.bound[0],
            r.bound[bound_pick]
        );

        // G5 direction sensitivity (anti-aligned drags the bound below peer).
        assert!(
            r.bound[3] < r.bound[1],
            "G5 FAIL seed {seed}: bound(anti) {} must sit below bound(peer) {}",
            r.bound[3],
            r.bound[1]
        );

        // Accumulate quality aggregates.
        sum_q0 += r.quality_baseline;
        for (j, _) in ALPHAS.iter().enumerate() {
            let q1 = r.quality_wired[strength_pick][j];
            let q2 = r.quality_wired[bound_pick][j];
            sum_q1[j] += q1;
            sum_q2[j] += q2;
            if q2 > q1 {
                wins_21[j] += 1;
            }
            if q2 > r.quality_baseline {
                wins_20[j] += 1;
            }
        }
    }

    let n = SEEDS.len() as f32;
    println!("=== aggregate over {n} seeds ===");
    println!("mean A0 baseline: {:.4}", sum_q0 / n);
    for (j, &alpha) in ALPHAS.iter().enumerate() {
        println!(
            "α={:.1}: mean A1(strength) {:.4} | mean A2(bound) {:.4} | A2>A1 {}/{} | A2>A0 {}/{}",
            alpha,
            sum_q1[j] / n,
            sum_q2[j] / n,
            wins_21[j],
            SEEDS.len(),
            wins_20[j],
            SEEDS.len()
        );
    }

    // G3 (selection correctness): mean A2 > mean A1 at every α, and A2 wins
    // the majority of cells (≥ 75%).
    for (j, &alpha) in ALPHAS.iter().enumerate() {
        let m2 = sum_q2[j] / n;
        let m1 = sum_q1[j] / n;
        assert!(
            m2 > m1,
            "G3a FAIL α={alpha}: mean paper arm {m2:.4} must beat mean strength arm {m1:.4}"
        );
        assert!(
            wins_21[j] * 4 >= SEEDS.len() as u32 * 3,
            "G3b FAIL α={alpha}: A2>A1 in only {}/{} cells (need ≥75%)",
            wins_21[j],
            SEEDS.len()
        );
    }

    // G4 (no-harm vs the shipped baseline) at adequate tilt α ∈ {0.4, 0.6} —
    // the α=0.2 row is reported, not gated (a peer-strength prover's n_mc=16
    // MC noise is not paid back at tilt weight 0.2; BOTH wired arms tend
    // below the no-prover baseline there, while selection stays correct).
    for (j, &alpha) in ALPHAS.iter().enumerate().skip(1) {
        let m2 = sum_q2[j] / n;
        let m0 = sum_q0 / n;
        assert!(
            m2 > m0,
            "G4 FAIL α={alpha}: mean paper arm {m2:.4} must beat mean baseline {m0:.4}"
        );
    }

    // Asserts above are the failure path (panic with the gate id + numbers).
    println!(
        "\nVERDICT: PASS — G1/G2/G5 per-seed, G3/G4 aggregate, {} seeds × α ∈ {:?}.",
        SEEDS.len(),
        ALPHAS
    );
}

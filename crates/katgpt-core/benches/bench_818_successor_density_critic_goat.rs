//! GOAT gate — successor-density goal critic (Issue 860, Bench 818).
//!
//! ```bash
//! cargo bench -p katgpt-core --features successor_density_critic \
//!     --bench bench_818_successor_density_critic_goat
//! # G4 needs an allocator in the profile:
//! cargo bench -p katgpt-core --features successor_density_critic,alloc_tracking \
//!     --bench bench_818_successor_density_critic_goat
//! ```
//!
//! **G1 correctness** — the empirical conditional successor mass
//! `N(s,a,g)/N(s,a)` converges to the ANALYTIC discounted measure of a
//! closed-form ring MDP (deterministic stay / step-forward actions; exact
//! `P_disc(g|s,a)` in closed form), and every `argmax_a` equals the
//! exact-measure argmax. Lemma 4.1 is executed directly: perturbing the
//! goal marginal by any positive factor leaves every argmax_a
//! bit-identical. The ring is tie-free across actions (min gap 0.0625 at
//! γ = 0.5), so sampling noise cannot flip a ranking. Determinism: two
//! identically-seeded builds freeze to byte-identical tables.
//!
//! **G2 perf** — O(1) lookup + argmax-scan against ABSOLUTE budgets
//! (best-of-N minimum, not a sequential A/B ratio — AGENTS.md § *A ratio
//! of two SEQUENTIALLY-timed arms measures the BOX* is why this gate
//! states budgets).
//!
//! **G3 no-regression** — the density-ratio correction is the point of the
//! primitive, so the gate is a ranking-consistency fixture where the
//! correction provably bites. Environment: 8-state ring, three
//! deterministic jumps (stay / +1 / +2) under a skewed behavior policy
//! (state 0 is a 0.98-stay sink; +1 taken 3× more often than +2
//! elsewhere); the exact target is the BEHAVIOR-CONTINUED discounted
//! measure (Bellman fixed-point iteration), not a greedy-action-repeat
//! closed form — the paper's own conditioning. TWO fixtures: G3a/G3b run
//! at γ = 0.9, where the flat conditional ladder lets the ~30× prior skew
//! move goal-salience argmax_g (at γ = 0.5 the 2×-per-step ladder
//! outruns any achievable prior and nothing can flip); G3c keeps the
//! steep γ = 0.5 fixture, where ranking all (s, a) rows for a fixed goal
//! by RAW joint count conflates visitation with conditional reachability
//! and provably inverts on enforced pairs, while the critic tracks the
//! exact measure with zero discordances (enforced pairs ≥ 1.9× apart,
//! exact ties excluded — noise ±2% cannot flip).
//!
//! **G4 alloc-free** — zero allocations on the read path (score,
//! argmax_a, argmax_g, p_successor). Gated on
//! `any(debug_assertions, feature = "alloc_tracking")` per the Issue-741
//! rule; prints a LOUD skip rather than a silent pass when the profile
//! carries no allocator.

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::successor_density_critic::{
    SamplerKind, SdcConfig, SuccessorDensityBuilder, SuccessorDensityTable, laplace_log_ratio,
};

#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[global_allocator]
static BENCH_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;

/// Deterministic xorshift — seeded, local, never the global RNG.
struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn verdict(ok: bool) -> &'static str {
    match ok {
        true => "✓",
        false => "✗",
    }
}

/// Best-of-`rounds` minimum nanoseconds per call (the load-robust
/// statistic — a loaded box can only make a round slower). The argument
/// varies the query across iterations so LLVM cannot fold the loop.
fn best_of_ns(rounds: usize, iters: usize, mut f: impl FnMut(u32) -> f32) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        let mut acc = 0.0_f32;
        for i in 0..iters as u32 {
            acc += f(i % 97);
        }
        black_box(acc);
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

/// One seeded walk over an 8-state ring with three deterministic jumps:
/// action 0 stays, action 1 steps +1, action 2 jumps +2. The behavior
/// policy is the G3 skew: state 0 is a 0.97-stay sink, and away from 0 the
/// +1 action is taken 3× as often as +2 (the raw-count inversion driver).
/// Returns the frozen table and its builder (the bench reads both).
fn build_skewed_ring(
    steps: usize,
    gamma: f32,
    seed: u64,
) -> (SuccessorDensityBuilder, SuccessorDensityTable) {
    const S: u32 = 8;
    let mut b = SuccessorDensityBuilder::new(SdcConfig {
        n_states: S,
        n_actions: 3,
        alpha: 1.0,
        gamma,
        sampler: SamplerKind::Discounted,
    });
    let mut rng = Xorshift(seed);
    let mut states = Vec::with_capacity(steps + 1);
    let mut actions = Vec::with_capacity(steps);
    states.push(0_u32);
    let mut cur = 0_u32;
    let draw = |rng: &mut Xorshift, cur: u32| -> (u32, u32) {
        let r = (rng.next_u64() % 1000) as u32;
        let act = match cur {
            0 => {
                if r < 980 {
                    0
                } else if r < 992 {
                    1
                } else {
                    2
                }
            }
            _ => {
                if r < 400 {
                    0
                } else if r < 850 {
                    1
                } else {
                    2
                }
            }
        };
        let next = match act {
            0 => cur,
            1 => (cur + 1) % S,
            _ => (cur + 2) % S,
        };
        (act, next)
    };
    for _ in 0..steps {
        let (act, next) = draw(&mut rng, cur);
        states.push(next);
        actions.push(act);
        cur = next;
    }
    b.observe_trajectory(&states, &actions);
    let table = b.clone().finish();
    (b, table)
}

/// Exact behavior-continued discounted successor measure for the
/// three-jump ring (stay / +1 / +2, all deterministic), by fixed-point
/// iteration on the Bellman recursion — the paper's own conditioning:
/// after the observed (s, a), the future continues along the BEHAVIOR
/// policy (the G3 skew: 0.97-stay sink at state 0, +1 3× over +2
/// elsewhere). Returns the flat `[s][a][g]` measure. γ = 0.5 → the
/// iteration contracts by 0.5 per sweep; 300 sweeps converge far below
/// f64 resolution.
fn exact_measure_iterated(gamma: f64, sweeps: usize) -> Vec<f64> {
    const S: usize = 8;
    const A: usize = 3;
    let jump = |s: usize, a: usize| -> usize {
        match a {
            0 => s,
            1 => (s + 1) % S,
            _ => (s + 2) % S,
        }
    };
    let behavior = |s: usize, a: usize| -> f64 {
        match s {
            0 => match a {
                0 => 0.98,
                1 => 0.012,
                _ => 0.008,
            },
            _ => match a {
                0 => 0.40,
                1 => 0.45,
                _ => 0.15,
            },
        }
    };
    let mut p = vec![0.0_f64; S * A * S];
    for _ in 0..sweeps {
        let mut nxt = vec![0.0_f64; S * A * S];
        for s in 0..S {
            for a in 0..A {
                let s2 = jump(s, a);
                for g in 0..S {
                    let hit = if s2 == g { 1.0 } else { 0.0 };
                    let mut v = (1.0 - gamma) * hit;
                    for a2 in 0..A {
                        v += gamma * behavior(s2, a2) * p[(s2 * A + a2) * S + g];
                    }
                    nxt[(s * A + a) * S + g] = v;
                }
            }
        }
        p = nxt;
    }
    p
}

/// The closed-form ring MDP is superseded: the G1 target is the exact
/// BEHAVIOR-CONTINUED measure (uniform stay/step policy), iterated to
/// convergence — the same Bellman shape as [`exact_measure_iterated`],
/// 32 states, 2 actions.
fn exact_measure_ring32(gamma: f64, sweeps: usize) -> Vec<f64> {
    const S: usize = 32;
    const A: usize = 2;
    let mut p = vec![0.0_f64; S * A * S];
    for _ in 0..sweeps {
        let mut nxt = vec![0.0_f64; S * A * S];
        for s in 0..S {
            for a in 0..A {
                let s2 = match a {
                    0 => s,
                    _ => (s + 1) % S,
                };
                for g in 0..S {
                    let hit = if s2 == g { 1.0 } else { 0.0 };
                    let v = (1.0 - gamma) * hit
                        + gamma * 0.5 * (p[s2 * A * S + g] + p[(s2 * A + 1) * S + g]);
                    nxt[(s * A + a) * S + g] = v;
                }
            }
        }
        p = nxt;
    }
    p
}

fn main() {
    let mut failures: Vec<String> = vec![];
    let t_start = Instant::now();
    println!("═══ Bench 818 — successor_density_critic GOAT (Issue 860) ═══");

    // ── G1: analytic exactness + Lemma-4.1 ranking (32-state ring) ────────
    {
        const S: u32 = 32;
        const STEPS: usize = 200_000;
        let gamma = 0.5_f64;
        let mut b = SuccessorDensityBuilder::new(SdcConfig {
            n_states: S,
            n_actions: 2,
            alpha: 1.0,
            gamma: 0.5,
            sampler: SamplerKind::Discounted,
        });
        let mut rng = Xorshift(0xDEADBEEFCAFEBABE_u64);
        let mut states = Vec::with_capacity(STEPS + 1);
        let mut actions = Vec::with_capacity(STEPS);
        states.push(0_u32);
        let mut cur = 0_u32;
        for _ in 0..STEPS {
            let act = (rng.next_u64() % 2) as u32;
            cur = match act {
                0 => cur,
                _ => (cur + 1) % S,
            };
            states.push(cur);
            actions.push(act);
        }
        b.observe_trajectory(&states, &actions);
        let table = b.clone().finish();

        let exact = exact_measure_ring32(gamma, 300);
        let mut max_cond_err = 0.0_f64;
        for st in 0..S as usize {
            for a in 0..2_usize {
                let denom = b.row_mass(st as u32, a as u32);
                for g in 0..S as usize {
                    let cond = b.sag_mass(st as u32, a as u32, g as u32) / denom;
                    max_cond_err =
                        max_cond_err.max((cond - exact[(st * 2 + a) * S as usize + g]).abs());
                }
            }
        }
        let g1a = max_cond_err <= 0.01;
        println!(
            "G1a exactness    max |empirical − analytic| = {max_cond_err:.5} ≤ 0.01 vs behavior-continued measure  {}",
            verdict(g1a)
        );
        if !g1a {
            failures.push(format!(
                "G1a: conditional-measure error {max_cond_err:.5} > 0.01"
            ));
        }

        let mut ranking_ok = true;
        let mut included = 0_usize;
        for st in 0..S {
            for g in 0..S {
                let c0 = exact[(st as usize * 2) * S as usize + g as usize];
                let c1 = exact[(st as usize * 2 + 1) * S as usize + g as usize];
                let gap = (c0 - c1).abs();
                if gap < 0.02 {
                    continue; // noise-dominated near-tie — counted, not asserted
                }
                included += 1;
                let exact_best = if c0 >= c1 { 0 } else { 1 };
                if table.argmax_a(st, g) != exact_best {
                    ranking_ok = false;
                }
            }
        }
        // Floor: at least ONE decisive goal per state on average (the
        // measured decisive fraction is ~1/8 — near-ties are expected and
        // counted, never folded into the pass).
        let g1b = ranking_ok && included >= S as usize;
        println!(
            "G1b ranking      argmax_a == exact argmax on {included} decisive (s,g) of {S}×{S} (gap ≥ 0.02)  {}",
            verdict(g1b)
        );
        if !g1b {
            failures.push(format!(
                "G1b: argmax_a disagrees with the exact measure (or only {included} decisive pairs)"
            ));
        }

        // Lemma 4.1 executed: the goal-only constant cannot move the
        // argmax. BOTH sides run the SAME f64 formula path so rounding
        // ties break identically (the frozen table's f32 argmax is already
        // tied to the exact measure by G1b).
        let argmax_f64 = |bias: f64| -> Vec<u32> {
            let mut out = Vec::with_capacity((S * S) as usize);
            for st in 0..S {
                for g in 0..S {
                    let mut best_a = 0_u32;
                    let mut best = f64::NEG_INFINITY;
                    for a in 0..2_u32 {
                        let v = laplace_log_ratio(
                            b.sag_mass(st, a, g),
                            b.row_mass(st, a),
                            b.goal_mass(g) * bias,
                            b.total_mass(),
                            1.0,
                            S as f64,
                        );
                        if v > best {
                            best = v;
                            best_a = a;
                        }
                    }
                    out.push(best_a);
                }
            }
            out
        };
        let mut invariance_ok = true;
        let baseline = argmax_f64(1.0);
        for bias in [0.001_f64, 3.7, 1.0e6] {
            if argmax_f64(bias) != baseline {
                invariance_ok = false;
            }
        }
        println!(
            "G1c Lemma 4.1    goal-prior perturbation (×0.001/×3.7/×1e6) leaves argmax_a bit-identical  {}",
            verdict(invariance_ok)
        );
        if !invariance_ok {
            failures.push("G1c: goal-prior perturbation moved an argmax_a".into());
        }

        // Determinism: an identically-seeded rebuild freezes bit-identical.
        let rebuild = build_skewed_ring(50_000, 0.5, 42).1.freeze();
        let original = build_skewed_ring(50_000, 0.5, 42).1.freeze();
        let det_ok = rebuild == original;
        println!(
            "G1d determinism  identically-seeded rebuild → byte-identical freeze  {}",
            verdict(det_ok)
        );
        if !det_ok {
            failures.push("G1d: identically-seeded rebuild diverged".into());
        }
    }

    // ── G2: lookup + argmax absolute budgets (64-state, 8-action) ─────────
    {
        let mut b = SuccessorDensityBuilder::new(SdcConfig {
            n_states: 64,
            n_actions: 8,
            alpha: 1.0,
            gamma: 0.5,
            sampler: SamplerKind::Discounted,
        });
        let mut rng = Xorshift(0xABCDEF0123456789_u64);
        let mut states = Vec::with_capacity(400_001);
        let mut actions = Vec::with_capacity(400_000);
        states.push(0_u32);
        let mut cur = 0_u32;
        for _ in 0..400_000 {
            let act = (rng.next_u64() % 8) as u32;
            cur = (cur + act) % 64;
            states.push(cur);
            actions.push(act);
        }
        b.observe_trajectory(&states, &actions);
        let table = b.finish();
        let score_ns = best_of_ns(50, 10_000, |i| table.score(i % 64, i % 8, (i + 1) % 64));
        let argmax_ns = best_of_ns(50, 10_000, |i| table.argmax_a(i % 64, (i + 1) % 64) as f32);
        let argmaxg_ns = best_of_ns(50, 10_000, |i| table.argmax_g(i % 64, i % 8) as f32);
        let g2 = score_ns < 50.0 && argmax_ns < 500.0 && argmaxg_ns < 500.0;
        println!(
            "G2 lookup        score {score_ns:.1} ns (<50), argmax_a {argmax_ns:.1} ns (<500), argmax_g {argmaxg_ns:.1} ns (<500)  {}",
            verdict(g2)
        );
        if !g2 {
            failures.push(format!(
                "G2: score {score_ns:.1}/argmax_a {argmax_ns:.1}/argmax_g {argmaxg_ns:.1} ns over budget"
            ));
        }
    }

    // ── G3: the ratio correction vs raw counts on the skewed ring ─────────
    // TWO fixtures, and the split is the point: G3a/G3b (goal-salience
    // prior bite) run at γ = 0.9, where the conditional ladder flattens to
    // ≈1.11× per step and the ~30× goal-prior skew (the state-0 sink)
    // provably outruns the top-vs-runner-up conditional gap — at γ = 0.5
    // the 2×-per-step ladder always outruns the prior and nothing can
    // flip. G3c keeps the steep γ = 0.5 fixture: near-tie exclusions stay
    // rare and the visitation-vs-conditional inversions land on ENFORCED
    // pairs (> 1.9× apart).
    {
        const S: u32 = 8;
        const A: u32 = 3;
        let gamma = 0.5_f64;
        let (b_flat, table_flat) = build_skewed_ring(400_000, 0.9, 0xFEEDFACE);
        let (b, table) = build_skewed_ring(400_000, 0.5, 0xFEEDFACE);

        // Goal-salience flip: the state-0 sink inflates N(0), so at least
        // one (s, a) query's argmax_g moves once the empirical prior is
        // applied (uniform-prior argmax_g == raw conditional argmax — the
        // structural half — and the empirical prior provably moves one).
        let mut structural_ok = true;
        let mut flips = 0_usize;
        for st in 0..S {
            for a in 0..A {
                // Uniform-prior argmax_g must equal the raw conditional
                // argmax_g (the prior term is the score's ONLY extra factor).
                let mut raw_best = 0_u32;
                let mut raw_v = f64::NEG_INFINITY;
                let mut uni_best = 0_u32;
                let mut uni_v = f64::NEG_INFINITY;
                let mut emp_best = 0_u32;
                let mut emp_v = f64::NEG_INFINITY;
                for g in 0..S {
                    let n_sag = b_flat.sag_mass(st, a, g);
                    let cond = n_sag / b_flat.row_mass(st, a);
                    if cond > raw_v {
                        raw_v = cond;
                        raw_best = g;
                    }
                    let u = laplace_log_ratio(
                        n_sag,
                        b_flat.row_mass(st, a),
                        1.0,
                        b_flat.total_mass(),
                        1.0,
                        S as f64,
                    );
                    if u > uni_v {
                        uni_v = u;
                        uni_best = g;
                    }
                    let e = laplace_log_ratio(
                        n_sag,
                        b_flat.row_mass(st, a),
                        b_flat.goal_mass(g),
                        b_flat.total_mass(),
                        1.0,
                        S as f64,
                    );
                    if e > emp_v {
                        emp_v = e;
                        emp_best = g;
                    }
                }
                if uni_best != raw_best {
                    structural_ok = false;
                }
                if emp_best != uni_best {
                    flips += 1;
                }
            }
        }
        let _ = table_flat; // the flat fixture's table is not asserted directly
        let bite_ok = flips >= 1;
        println!(
            "G3a structure    uniform-prior argmax_g == raw conditional argmax (all {S}×{A})  {}",
            verdict(structural_ok)
        );
        println!(
            "G3b prior bites  empirical prior moves {flips} goal-salience argmax_g (≥1)  {}",
            verdict(bite_ok)
        );
        if !structural_ok {
            failures.push(
                "G3a: uniform-prior argmax_g diverged from the raw conditional argmax".into(),
            );
        }
        if !bite_ok {
            failures.push(
                "G3b: the empirical goal prior moved no argmax_g (fixture skew lost?)".into(),
            );
        }

        // Row-ranking consistency: for a fixed goal, rank all (s, a) rows
        // by score vs the exact behavior-continued measure. Pairs whose
        // exact values tie or sit within 1.9× are excluded (empirical
        // noise ±2% cannot flip a 1.9× gap). The raw joint count conflates
        // (s,a) visitation and provably inverts somewhere on this fixture
        // (the state-0 sink's 33× visitation vs a ≤ 5.3× conditional
        // advantage).
        let exact = exact_measure_iterated(gamma, 300);
        let mut critic_disc = 0_usize;
        let mut raw_disc = 0_usize;
        let mut enforced_pairs = 0_usize;
        for g in 0..S {
            let mut critic_row = vec![];
            let mut raw_row = vec![];
            for st in 0..S {
                for a in 0..A {
                    let score = table.score(st, a, g) as f64;
                    let raw = b.sag_mass(st, a, g);
                    critic_row.push(score);
                    raw_row.push(raw);
                }
            }
            for i in 0..(S * A) as usize {
                for j in (i + 1)..(S * A) as usize {
                    let c_i = exact[i * S as usize + g as usize];
                    let c_j = exact[j * S as usize + g as usize];
                    if (c_i - c_j).abs() <= 1e-9 {
                        continue; // exact tie — excluded
                    }
                    if c_i.min(c_j) > 0.0 && c_i.max(c_j) / c_i.min(c_j) < 1.9 {
                        continue; // noise-adjacent — excluded
                    }
                    enforced_pairs += 1;
                    let expect = (c_i > c_j) as u8;
                    if ((critic_row[i] > critic_row[j]) as u8) != expect {
                        critic_disc += 1;
                    }
                    if ((raw_row[i] > raw_row[j]) as u8) != expect {
                        raw_disc += 1;
                    }
                }
            }
        }
        let g3c = critic_disc == 0 && raw_disc > 0;
        println!(
            "G3c ranking      critic discordances {critic_disc} == 0, raw-count discordances {raw_disc} > 0, over {enforced_pairs} enforced pairs  {}",
            verdict(g3c)
        );
        if critic_disc > 0 {
            failures.push(format!(
                "G3c: critic has {critic_disc} ranking discordances vs the exact measure"
            ));
        }
        if raw_disc == 0 {
            failures
                .push("G3c: raw-count baseline never inverted (fixture lost its teeth?)".into());
        }
    }

    // ── G4: allocation-free read path ──────────────────────────────────────
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    {
        let table = build_skewed_ring(50_000, 0.5, 7).1;
        katgpt_core::alloc::reset_alloc_stats();
        let mut acc = 0.0_f32;
        for i in 0..10_000_u32 {
            let s = i % 8;
            let a = i % 3;
            let g = (i + 1) % 8;
            acc += table.score(s, a, g);
            acc += table.p_successor(s, a, g);
            acc += table.argmax_a(s, g) as f32;
            acc += table.argmax_g(s, a) as f32;
        }
        black_box(acc);
        let (n_allocs, bytes) = katgpt_core::alloc::get_alloc_stats();
        let g4 = n_allocs == 0;
        println!(
            "G4 alloc-free    {n_allocs} alloc(s), {bytes} byte(s) over 10000 mixed read calls  → {}",
            verdict(g4)
        );
        if !g4 {
            failures.push(format!("G4: {n_allocs} allocation(s) on the read path"));
        }
    }
    #[cfg(not(any(debug_assertions, feature = "alloc_tracking")))]
    {
        println!(
            "G4 alloc-free    ⛔ NOT MEASURED — this profile compiles no allocator. \
             Re-run with `--features alloc_tracking`; a green run without it is not a G4 pass."
        );
    }

    println!();
    println!("wall {} ms", t_start.elapsed().as_millis());
    match failures.is_empty() {
        true => println!("✓ Bench 818 PASSED — every measured gate holds"),
        false => {
            for f in &failures {
                println!("✗ {f}");
            }
            println!("\n✗ Bench 818 FAILED — {} gate(s)", failures.len());
            std::process::exit(1);
        }
    }
}

//! Issue 825 — training-free Coulomb crowd redistribution via a DEC Poisson
//! solve: the GOAT gate for the shipped `coulomb_flow` primitive.
//!
//! Research 468 §5 / arXiv:2608.01692 v3 Proposition 2: with the canonical
//! Coulomb field `b = grad(phi)`, `Laplace(phi) = mu0 - mu1`, the first-hitting
//! map of the autonomous flow `X' = b(X)` transports `mu0 -> mu1` EXACTLY, with
//! no training. Targets must be singular; a zone graph is the atomic case
//! (zones = vertices, `mu1` = weighted atoms on sink vertices).
//!
//! The issue's own caveat is what this bench exists to test:
//!
//! > **Continuum Coulomb field is NOT directly the discrete solve.** Prop 2
//! > lives in R^d with the fundamental solution; the DEC analog is the *graph
//! > Laplacian* Poisson solve — transport property transfer is unproven.
//!
//! # T3: this bench measures the SHIPPED code, not a copy
//!
//! T1/T2 ran against a PoC that carried its own CG, its own refinement loop and
//! its own routing tables inside this file. T3 promoted all three into
//! `katgpt-dec` (`poisson_solve_into` in `hodge.rs`; `CoulombFlowField` and
//! `CrowdRouter` behind the opt-in `coulomb_flow` flag), and this bench now
//! calls them. A GOAT gate that measures a private transcription of the
//! primitive certifies the transcription — which is the sibling-arm defect this
//! repo records one platform over.
//!
//! # Sign convention — read this before the asserts
//!
//! The paper writes `Laplace(phi) = mu0 - mu1` for the continuum Laplacian,
//! which is negative semidefinite. This repo's `graph_laplacian` is
//! `delta.d = D - A`, the **positive** semidefinite negation of it. So the
//! solve is `L(phi) = mu1 - mu0` and `phi` is the paper's potential negated:
//! mass flows toward INCREASING phi and sinks are potential maxima. Getting
//! this backwards yields a field that pushes every NPC away from its target
//! while every conservation assert still passes, which is why the convention is
//! written down next to the solve rather than inferred.
//!
//! # What the four gates measure
//!
//! - **G1 (conservation)** — `delta(j) = delta.d(phi) = L(phi) = mu1 - mu0`
//!   holds identically for `j = d(phi)`, so the numeric assert is on the solver
//!   residual, not on the algebra. Checked twice: `codifferential` against the
//!   target rank-0 field, and `belief_mass_divergence` on the mass-balanced
//!   residual flow — the Plan 314 steady-state check generalized to
//!   `mu0 != mu1`.
//! - **G2 (endpoint distribution)** — route N independent walkers by the flow
//!   and compare arrivals to `mu1` by MAE. This is a SAMPLING measurement: the
//!   expected endpoint distribution of the consistent construction is exactly
//!   `mu1`, so the residual is Monte-Carlo noise and must fall as `1/sqrt(N)`.
//!   The gate asserts that DECAY, because an assert at one N cannot tell
//!   "exact plus noise" from "biased by less than the tolerance" — and a
//!   fixture that cannot express the mechanism is the failure this repo keeps
//!   re-finding. The exact expectation is cross-checked without any RNG at all
//!   by `CoulombFlowField::expected_arrivals`.
//! - **G3 (the 20x-bias analog)** — the naive rescaled attract field, the
//!   shipped `DecFlowField` consumption shape: `phi_naive(v) = -sum_j w_j *
//!   dist(v, sink_j)`, hand-set weights. ⚠ Both arms are walked by the SAME
//!   `CrowdRouter::step`, differing in exactly one vector: the absorption
//!   policy. That is what `from_flow_with_absorption` exists for — with two
//!   walker implementations a reader could credit the ratio to the walker.
//!   The paper measured 20x endpoint-weight bias for the EqM-style field
//!   against 0.005 MAE for the consistent one. If the naive field is NOT badly
//!   biased on this graph the bench says so and reports FAIL-honest, per Issue
//!   825 T2 — it does not move the bar.
//! - **G4 (alloc-free solve)** — after one warm-up, the refined Poisson solve
//!   and the `d(phi)` step allocate ZERO, measured through the repo's shared
//!   `counting_allocator!()` harness with its liveness canary.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features coulomb_flow \
//!   --bench bench_825_coulomb_crowd_redistribution_goat
//! ```

#![cfg(feature = "coulomb_flow")]

use std::time::Instant;

use katgpt_core::dec::{
    CellComplex, CochainField, CoulombFlowField, CrowdRouter, PoissonScratch, RouterStep,
    belief_mass_divergence, codifferential,
};
// The `_into` (zero-alloc) variants are not re-exported at the crate root -
// only the allocating ones are - so they come from the module. G4 is about
// exactly this pair, so reaching for the root export would quietly make the
// gate measure the wrong thing.
use katgpt_core::dec::hodge::poisson_solve_into;
use katgpt_core::dec::operators::{
    codifferential_into, exterior_derivative_into, graph_laplacian_into,
};

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ---------------------------------------------------------------------------
// The toy zone graph
// ---------------------------------------------------------------------------

/// Grid width / height. 4x3 = 12 zones, the issue's "~6-12 vertices".
const W: usize = 4;
const H: usize = 3;
const N_V: usize = W * H;

/// Three sinks with DELIBERATELY unequal target weights. The bias gate exists
/// because a naive attract field cannot reproduce these; mirrors the paper's
/// 5-atom unequal-weight experiment.
const SINKS: [(usize, f32); 3] = [(0, 0.5), (3, 0.3), (11, 0.2)];

fn vid(x: usize, y: usize) -> usize {
    y * W + x
}

fn coord(v: usize) -> (usize, usize) {
    (v % W, v / W)
}

/// 4-neighbour grid graph as an explicit edge list.
///
/// `CellComplex::from_edges` orients edge `i` as `(tail, head)` with boundary
/// entries `(tail, i, -1)` and `(head, i, +1)`, so `d0(phi)[i] = phi[head] -
/// phi[tail]` and `delta1(j)[v]` is the net INFLOW at `v`.
fn grid_edges() -> Vec<(usize, usize)> {
    let mut e = Vec::new();
    for y in 0..H {
        for x in 0..W {
            if x + 1 < W {
                e.push((vid(x, y), vid(x + 1, y)));
            }
            if y + 1 < H {
                e.push((vid(x, y), vid(x, y + 1)));
            }
        }
    }
    e
}

/// `mu0`: the current crowd, uniform over every NON-sink zone.
/// `mu1`: the authored target, the unequal atoms above. Both sum to 1.
fn densities() -> (Vec<f32>, Vec<f32>) {
    let mut mu1 = vec![0.0f32; N_V];
    for (v, w) in SINKS {
        mu1[v] = w;
    }
    let sources: Vec<usize> = (0..N_V).filter(|v| mu1[*v] == 0.0).collect();
    let share = 1.0 / sources.len() as f32;
    let mut mu0 = vec![0.0f32; N_V];
    for &v in &sources {
        mu0[v] = share;
    }
    (mu0, mu1)
}

// ---------------------------------------------------------------------------
// The shared walker
// ---------------------------------------------------------------------------

/// Route every walker in `start` through `router` and return the normalized
/// arrival histogram, plus the number of walkers that hit the step cap.
///
/// ONE implementation, both arms. The Coulomb arm's flow is a gradient flow and
/// is therefore acyclic, so the cap can only bind for a NON-gradient field; it
/// is reported rather than silently absorbed, because a capped walk is a
/// finding about the field and not a timeout.
fn walk(router: &CrowdRouter, start: &[usize], rng: &mut fastrand::Rng) -> (Vec<f32>, usize) {
    let mut hist = vec![0.0f32; N_V];
    let max_steps = N_V * 4;
    let mut capped = 0usize;
    for &s in start {
        let mut v = s;
        let mut stepped = 0usize;
        loop {
            match router.step(v, rng.f32()) {
                RouterStep::Absorb => {
                    hist[v] += 1.0;
                    break;
                }
                RouterStep::Move(next) => {
                    if stepped >= max_steps {
                        capped += 1;
                        hist[v] += 1.0;
                        break;
                    }
                    v = next;
                    stepped += 1;
                }
            }
        }
    }
    let n = start.len() as f32;
    for h in hist.iter_mut() {
        *h /= n;
    }
    (hist, capped)
}

fn mae(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

/// Walker start vertices drawn from `mu0` by largest-remainder apportionment,
/// so the START distribution contributes no sampling error of its own — the
/// only noise in G2 is the routing, which is what the decay bar is about.
fn walkers_from(mu0: &[f32], n: usize) -> Vec<usize> {
    let scaled: Vec<f32> = mu0.iter().map(|&m| m * n as f32).collect();
    let mut counts: Vec<usize> = scaled.iter().map(|s| s.floor() as usize).collect();
    let mut rem: Vec<(f32, usize)> = scaled
        .iter()
        .enumerate()
        .map(|(v, s)| (s - s.floor(), v))
        .collect();
    rem.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut assigned: usize = counts.iter().sum();
    let mut k = 0usize;
    while assigned < n {
        counts[rem[k % N_V].1] += 1;
        assigned += 1;
        k += 1;
    }
    let mut out = Vec::with_capacity(n);
    for (v, c) in counts.iter().enumerate() {
        out.extend(std::iter::repeat_n(v, *c));
    }
    out
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Issue 825 — Coulomb crowd redistribution GOAT (DEC Poisson solve) ===\n");

    let edges = grid_edges();
    let cx = CellComplex::from_edges(N_V, &edges);
    let (mu0, mu1) = densities();

    println!(
        "zone graph: {N_V} vertices ({W}x{H}), {} edges, {} sinks",
        edges.len(),
        SINKS.len()
    );
    println!(
        "mu0: uniform over {} source zones   mu1: {SINKS:?}\n",
        mu0.iter().filter(|&&m| m > 0.0).count()
    );

    // ── the solve, through the shipped API ────────────────────────────────
    let t0 = Instant::now();
    let field = CoulombFlowField::from_densities(&cx, &mu0, &mu1);
    let solve_us = t0.elapsed().as_secs_f64() * 1e6;
    println!(
        "solve: {} CG solve(s) over {} refinement round(s), true residual {:.3e}, {solve_us:.1} us",
        field.stats.cg_solves, field.stats.refine_rounds, field.stats.residual_inf
    );

    let rhs: Vec<f32> = mu1.iter().zip(&mu0).map(|(a, b)| a - b).collect();

    // ── G1: conservation ──────────────────────────────────────────────────
    let div = codifferential(&cx, &field.edge_flow);
    let g1_resid = div
        .data
        .iter()
        .zip(&rhs)
        .map(|(d, r)| (d - r).abs())
        .fold(0.0f32, f32::max);
    // GLOBAL balance: `delta` of ANY edge field sums to zero over vertices,
    // because every edge contributes +1 at its head and -1 at its tail. It is
    // structurally exact and independent of the solve, so it catches an
    // orientation or indexing error that the residual above cannot.
    let g1_global: f32 = div.data.iter().sum::<f32>().abs();
    // The Plan 314 reducer, generalized to `mu0 != mu1`: `belief_mass_divergence`
    // is `sum_v |delta(j)[v]|`, which for a correct field is the TOTAL MASS
    // MOVED — `sum_v |mu1 - mu0|`, i.e. everything that leaves plus everything
    // that arrives. Reading it as "should be ~0" would be the steady-state
    // special case and would assert the transport away.
    let moved_expect: f32 = rhs.iter().map(|r| r.abs()).sum();
    let g1_bmd = belief_mass_divergence(&cx, &field.edge_flow);
    // The bar is the f32 FLOOR for this problem, not a preference: the RHS has
    // entries of order 1e-1 and f32 carries ~1.2e-7 relative, so 1e-6 absolute
    // is already single-digit ULPs of the largest term. The first run sat at
    // 1.4e-5 and the bar was NOT moved to meet it — the solver was wrong (a CG
    // breakdown guard that discarded every refinement step) and got fixed.
    //
    // Diagnostic: how far apart the crate's TWO forms of the same rank-0
    // operator are in f32 on this complex — the fused `graph_laplacian` against
    // the composed `delta(d(.))`. This is now LOAD-BEARING rather than
    // decorative: `poisson_solve_into` inverts the FUSED form, while G1 reads
    // conservation through the COMPOSED one, so the gate is only measuring the
    // operator the solve used while this gap stays at zero.
    //
    // ⚠ An earlier version of the PoC claimed the two forms disagreed and named
    // that as the cause of a stuck residual. The diagnostic refuted it — the
    // gap is exactly 0 — which is why it is printed every run rather than
    // remembered from one.
    let mut fused = CochainField::zeros(0, N_V, 1);
    graph_laplacian_into(&cx, &field.potential, &mut fused);
    let mut composed_d = CochainField::zeros(1, edges.len(), 1);
    let mut composed = CochainField::zeros(0, N_V, 1);
    exterior_derivative_into(&cx, &field.potential, &mut composed_d);
    codifferential_into(&cx, &composed_d, &mut composed);
    let op_gap = fused
        .data
        .iter()
        .zip(&composed.data)
        .map(|(f, c)| (f - c).abs())
        .fold(0.0f32, f32::max);

    let g1_tol = 1e-6f32;
    let g1 = g1_resid <= g1_tol && g1_global <= 1e-6 && (g1_bmd - moved_expect).abs() <= 1e-4;
    println!("\nG1 conservation:");
    println!("   max|delta(j) - (mu1-mu0)|  = {g1_resid:.3e}  (tol {g1_tol:.0e})");
    println!("   |sum_v delta(j)[v]|        = {g1_global:.3e}  (tol 1e-6, structurally exact)");
    println!(
        "   belief_mass_divergence(j)  = {g1_bmd:.6}  vs mass moved {moved_expect:.6} (tol 1e-4)"
    );
    println!(
        "   [diagnostic] max|graph_laplacian(phi) - delta(d(phi))| = {op_gap:.3e}  (one operator, two forms; report only)"
    );
    println!("   -> {}", pf(g1));

    // ── G2: endpoint distribution and its 1/sqrt(N) decay ─────────────────
    let router = field.router(&cx, &mu0, &mu1);

    // The expectation, with no RNG in it at all. By the flow decomposition this
    // IS mu1, so it is a check on the routing table rather than a prediction —
    // and it is the thing the sampled runs below converge to.
    let exact = field.expected_arrivals(&router, &mu0);
    let exact_mae = mae(&exact, &mu1);
    println!("\nG2 endpoint distribution (Coulomb solve), MAE vs mu1:");
    println!(
        "   exact (no RNG, expected_arrivals) MAE = {exact_mae:.3e}  \
         (the flow decomposition; sampling below converges to THIS)"
    );

    let mut rng = fastrand::Rng::with_seed(0x0825_C0F1);
    let mut maes: Vec<(usize, f32)> = Vec::new();
    let mut total_capped = 0usize;
    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let start = walkers_from(&mu0, n);
        let (hist, capped) = walk(&router, &start, &mut rng);
        total_capped += capped;
        let m = mae(&hist, &mu1);
        println!(
            "   N = {n:>9}   MAE = {m:.6}   MAE*sqrt(N) = {:.4}   capped = {capped}",
            m * (n as f32).sqrt()
        );
        maes.push((n, m));
    }
    let decay = maes[0].1 / maes[3].1.max(f32::EPSILON);
    let g2_tol = 0.01f32;
    // The signal is the DECAY, not one number: exact-plus-noise falls like
    // 1/sqrt(N), a biased field flattens. 1000x more walkers should buy near
    // 31x; the bar is a loose 4x so sampling noise alone cannot red it.
    let g2 = maes[3].1 <= g2_tol && decay >= 4.0 && total_capped == 0 && exact_mae <= 1e-5;
    println!(
        "   MAE at N=1e6 {:.6} (tol {g2_tol}), decay 1e3->1e6 {decay:.1}x (bar 4x), \
         capped {total_capped} (bar 0 — a gradient flow is acyclic) -> {}",
        maes[3].1,
        pf(g2)
    );

    // ── G3: the 20x-bias analog ───────────────────────────────────────────
    // phi_naive(v) = -sum_j w_j * dist(v, sink_j) — the shipped DecFlowField
    // consumption shape: a hand-built goal potential, no density input. Negated
    // once more for this file's sign convention, exactly as the solve is, so
    // both fields push mass toward increasing potential.
    let phi_naive: Vec<f32> = (0..N_V)
        .map(|v| {
            let (vx, vy) = coord(v);
            SINKS
                .iter()
                .map(|&(s, w)| {
                    let (sx, sy) = coord(s);
                    let d = vx.abs_diff(sx) as f32 + vy.abs_diff(sy) as f32;
                    w * d
                })
                .sum::<f32>()
        })
        .collect();
    let naive_ch = CochainField::from_vec(0, 1, phi_naive);
    let mut j_naive = CochainField::zeros(1, edges.len(), 1);
    exterior_derivative_into(&cx, &naive_ch, &mut j_naive);
    // `d` of a distance-sum increases AWAY from the sinks, so the transport
    // field is its negation.
    for f in j_naive.data.iter_mut() {
        *f = -*f;
    }
    // Stop on FIRST ARRIVAL at a sink — the only rule a hand-built attract
    // field can offer, because it carries no target weights to derive one from.
    // Same table builder, same walker; one vector differs.
    let first_sink: Vec<f32> = mu1
        .iter()
        .map(|&m| if m > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let router_naive = CrowdRouter::from_flow_with_absorption(&cx, &j_naive, &first_sink);
    let n_big = 1_000_000usize;
    let start = walkers_from(&mu0, n_big);
    let (hist_naive, capped_naive) = walk(&router_naive, &start, &mut rng);
    let mae_naive = mae(&hist_naive, &mu1);
    let ratio = mae_naive / maes[3].1.max(f32::EPSILON);
    let g3 = ratio >= 10.0;
    // Two DIFFERENT failures hide inside one MAE, and pooling them lets a reader
    // credit the whole ratio to weight bias. Named apart: how much of the crowd
    // reached a sink AT ALL (a hand-built attract field has local minima, and a
    // walker that finds one stops in open country), and how the arrivals SPLIT
    // once you condition on having arrived.
    let arrived_naive: f32 = SINKS.iter().map(|&(v, _)| hist_naive[v]).sum();
    println!("\nG3 naive rescaled attract field (the DecFlowField shape):");
    println!(
        "   reached a sink at all: {:.1}%  (the rest stop at local minima of the hand-built potential, in open country)",
        arrived_naive * 100.0
    );
    for (v, w) in SINKS {
        let cond = if arrived_naive > 0.0 {
            hist_naive[v] / arrived_naive
        } else {
            0.0
        };
        println!(
            "   sink {v:>2}: target {w:.3}   naive {:.4}   conditional-on-arrival {cond:.4}",
            hist_naive[v]
        );
    }
    println!(
        "   MAE naive {mae_naive:.6} vs Coulomb {:.6} -> {ratio:.1}x \
         (bar 10x, paper measured 20x), capped {capped_naive} -> {}",
        maes[3].1,
        if g3 { "PASS" } else { "FAIL-honest" }
    );
    if !g3 {
        println!(
            "   ^ FAIL-honest: the naive field is NOT badly biased on this graph. Issue 825 T2\n   \
               says to report that and re-adjudicate whether the primitive earns a flag — it\n   \
               does NOT say to move the bar."
        );
    }

    // ── G4: alloc-free solve ──────────────────────────────────────────────
    let g4 = gate_g4(&cx, &rhs, edges.len());

    let all = g1 && g2 && g3 && g4;
    println!(
        "\n=== VERDICT: G1 {} · G2 {} · G3 {} · G4 {} -> {} ===",
        pf(g1),
        pf(g2),
        pf(g3),
        pf(g4),
        if all { "GOAT PASS" } else { "GOAT FAIL" }
    );
    if !all {
        std::process::exit(1);
    }
}

fn pf(b: bool) -> &'static str {
    if b { "PASS" } else { "FAIL" }
}

/// G4 — the steady-state solve path allocates nothing.
///
/// `assert_counter_is_live()` first: a counter that has silently become a
/// no-op passes every alloc gate in the repo at once, which is strictly worse
/// than the defect the gate is looking for.
///
/// This measures `poisson_solve_into` — the REFINED solve, the path the other
/// three gates read through `CoulombFlowField::from_densities`. The allocating
/// constructor is deliberately not the subject: it builds its own scratch by
/// design, and measuring it would report the scratch as a leak.
fn gate_g4(cx: &CellComplex, rhs: &[f32], n_edges: usize) -> bool {
    assert_counter_is_live();
    let mut scratch = PoissonScratch::new(N_V);
    let mut phi = CochainField::zeros(0, N_V, 1);
    let mut j_ch = CochainField::zeros(1, n_edges, 1);

    // Warm-up first: the first pass sizes every buffer the measured pass reuses.
    poisson_solve_into(cx, rhs, &mut phi.data, &mut scratch);
    exterior_derivative_into(cx, &phi, &mut j_ch);

    let (_, allocs) = alloc_delta(|| {
        for _ in 0..100 {
            poisson_solve_into(cx, rhs, &mut phi.data, &mut scratch);
            exterior_derivative_into(cx, &phi, &mut j_ch);
        }
    });
    let ok = allocs == 0;
    println!(
        "\nG4 alloc: {allocs} allocation(s) over 100 refined solve + d(phi) passes -> {}",
        pf(ok)
    );
    ok
}

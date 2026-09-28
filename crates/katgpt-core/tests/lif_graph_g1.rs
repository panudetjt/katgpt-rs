//! G1 — event-driven vs dense LIF parity (Issue 763 T3).
//!
//! The load-bearing claim of `lif_graph`: the event-driven `step` and the
//! dense reference `step_dense` produce **bit-identical trajectories** —
//! same spike trains (as sets), same `(v, g, refrac)` state (as bits) —
//! because quiescence is defined as the bitwise fixed point of the shared
//! per-node update, so skipping a quiescent node IS the identity.
//!
//! Also pins: run-twice determinism, ring wraparound (T ≫ delay), refrac
//! overlap under burst drive, and the delay semantics on a cascade graph.

#![cfg(feature = "lif_graph")]

use katgpt_core::lif_graph::{FixtureRng, LifParams, LifReservoir, SignedAdjacency};

/// Drive pattern shared by both arms: `n_drive` nodes from the first
/// `drive_pool` nodes get a PSP injection every `period` ticks, magnitude
/// alternating to exercise refrac overlap.
fn drive(
    r: &mut LifReservoir,
    tick: usize,
    n_drive: u32,
    period: usize,
    mag: f32,
    rng: &mut FixtureRng,
    drive_pool: u64,
) {
    if tick.is_multiple_of(period) {
        for _ in 0..n_drive {
            let node = rng.below(drive_pool) as u32;
            // Alternate a big and a small drive: bursts + partial responses.
            let w = if rng.next_u64() & 1 == 0 {
                mag
            } else {
                mag * 0.4
            };
            r.inject(node, w);
        }
    }
}

/// One parity arm: run `ticks` with the given graph/params/drive, collecting
/// the per-tick sorted spike sets + the final state.
fn run_arm(
    adj: &SignedAdjacency,
    params: &LifParams,
    ticks: usize,
    drive_cfg: (u32, usize, f32),
    dense: bool,
    seed: u64,
) -> (Vec<Vec<u32>>, Vec<f32>, Vec<f32>, Vec<u32>) {
    let mut r = LifReservoir::new(adj.clone(), params.clone());
    let mut rng = FixtureRng::new(seed);
    let mut spike_sets = Vec::with_capacity(ticks);
    let pool = r.n() as u64;
    for t in 0..ticks {
        drive(
            &mut r,
            t,
            drive_cfg.0,
            drive_cfg.1,
            drive_cfg.2,
            &mut rng,
            pool,
        );
        let mut s: Vec<u32> = if dense {
            r.step_dense().to_vec()
        } else {
            r.step().to_vec()
        };
        s.sort_unstable();
        spike_sets.push(s);
    }
    let (v, g, refrac) = r.state();
    (spike_sets, v.to_vec(), g.to_vec(), refrac.to_vec())
}

fn assert_parity(
    name: &str,
    n: u32,
    edges: &[(u32, u32, f32)],
    drive_cfg: (u32, usize, f32),
    ticks: usize,
    seed: u64,
    min_total_spikes: usize,
) {
    let adj = SignedAdjacency::from_edges(n, edges);
    let params = LifParams::shiu();
    let ev = run_arm(&adj, &params, ticks, drive_cfg, false, seed);
    let de = run_arm(&adj, &params, ticks, drive_cfg, true, seed);
    // Per-tick spike sets.
    for t in 0..ticks {
        assert_eq!(
            ev.0[t], de.0[t],
            "{name}: tick {t} spike set diverged (event vs dense)"
        );
    }
    // Final state, bitwise.
    let total: usize = ev.0.iter().map(|s| s.len()).sum();
    assert!(
        total >= min_total_spikes,
        "{name}: vacuous parity — only {total} total spikes; the fixture must fire"
    );
    for i in 0..n as usize {
        assert_eq!(
            ev.1[i].to_bits(),
            de.1[i].to_bits(),
            "{name}: v[{i}] diverged ({:+e} vs {:+e})",
            ev.1[i],
            de.1[i]
        );
        assert_eq!(
            ev.2[i].to_bits(),
            de.2[i].to_bits(),
            "{name}: g[{i}] diverged ({:+e} vs {:+e})",
            ev.2[i],
            de.2[i]
        );
        assert_eq!(ev.3[i], de.3[i], "{name}: refrac[{i}] diverged");
    }
}

/// Random ER graph arm helper (seeds the RNG once, builds edges).
fn er_edges(n: u32, n_edges: usize, p_inh: f32, w: f32, seed: u64) -> Vec<(u32, u32, f32)> {
    let mut rng = FixtureRng::new(seed);
    (0..n_edges)
        .map(|_| {
            let s = rng.below(n as u64) as u32;
            let t = rng.below(n as u64) as u32;
            let w = if rng.uniform() < p_inh { -w } else { w };
            (s, t, w)
        })
        .collect()
}

// ── 1. Sparse regime: subthreshold edges, driven bursts, no cascade ─────────

#[test]
fn parity_sparse_driven_bursts() {
    for seed in [1u64, 2, 3] {
        let edges = er_edges(500, 2500, 0.2, 1.0, seed * 977);
        // Drive 25 nodes/period with suprathreshold bursts (400 PSP = instant
        // class); edges subthreshold (1 mV asymptote < 7 mV gap) → no cascade.
        assert_parity(
            "sparse-bursts",
            500,
            &edges,
            (25, 10, 400.0),
            600, // ≫ 18-tick delay: ring wraparound × ~33
            seed,
            50,
        );
    }
}

// ── 2. Cascade regime: every edge suprathreshold → sustained activity ───────

#[test]
fn parity_cascade_saturation() {
    for seed in [11u64, 12] {
        let edges = er_edges(300, 1200, 0.2, 400.0, seed * 31);
        // Any spike cascades; inhibition pushes back — both paths must agree
        // through the full churn.
        assert_parity("cascade", 300, &edges, (5, 5, 400.0), 400, seed, 200);
    }
}

// ── 3. Deterministic chain: exact delay semantics ────────────────────────────

#[test]
fn parity_deterministic_chain() {
    let edges = vec![(0u32, 1u32, 400.0), (1, 2, 400.0), (2, 0, 400.0)];
    // Ring of 3 with instant-suprathreshold edges: fires circulate forever —
    // exercises ring wraparound + sustained re-fire without any randomness.
    // One spike per 18-tick hop → ~17 spikes per node-window; 20 is the
    // non-vacuity bar, not a rate claim.
    assert_parity("chain-ring", 3, &edges, (1, 1, 400.0), 300, 42, 20);
}

// ── 4. Run-twice determinism (both paths) ────────────────────────────────────

#[test]
fn determinism_run_twice() {
    let edges = er_edges(200, 800, 0.25, 1.0, 4242);
    for dense in [false, true] {
        let a = run_arm(
            &SignedAdjacency::from_edges(200, &edges),
            &LifParams::shiu(),
            300,
            (10, 7, 400.0),
            dense,
            7,
        );
        let b = run_arm(
            &SignedAdjacency::from_edges(200, &edges),
            &LifParams::shiu(),
            300,
            (10, 7, 400.0),
            dense,
            7,
        );
        assert_eq!(
            a.0, b.0,
            "spike trains must be reproducible (dense={dense})"
        );
        assert_eq!(a.1, b.1, "v must be reproducible (dense={dense})");
        assert_eq!(a.2, b.2, "g must be reproducible (dense={dense})");
        assert_eq!(a.3, b.3, "refrac must be reproducible (dense={dense})");
    }
}

// ── 5. Quiescent nodes stay bitwise at rest through the whole run ────────────

#[test]
fn never_driven_nodes_never_move() {
    // Nodes 150..200 receive no input and no edges target them by
    // construction: build edges only within 0..150.
    let edges = er_edges(150, 600, 0.2, 1.0, 99);
    let adj = SignedAdjacency::from_edges(200, &edges);
    let mut r = LifReservoir::new(adj, LifParams::shiu());
    let mut rng = FixtureRng::new(5);
    for t in 0..400 {
        // Drive pool = first 150 nodes ONLY: nodes 150..200 are provably
        // never driven and never edge-targeted (edges live within 0..150).
        drive(&mut r, t, 10, 8, 400.0, &mut rng, 150);
        r.step();
    }
    let (v, g, refrac) = r.state();
    for i in 150..200 {
        assert_eq!(v[i].to_bits(), (-52.0_f32).to_bits(), "v[{i}] moved");
        assert_eq!(g[i].to_bits(), 0.0_f32.to_bits(), "g[{i}] moved");
        assert_eq!(refrac[i], 0, "refrac[{i}] moved");
    }
}

// ── 6. The controls produce different-but-valid graphs (Maslov–Sneppen) ─────

#[test]
fn maslov_control_changes_topology_not_degrees() {
    let base = SignedAdjacency::from_edges(200, &er_edges(200, 1000, 0.2, 1.0, 31337));
    let swapped = base.maslov_sneppen(2000, 5);
    // Degrees preserved…
    let mut out_base = vec![0u32; 200];
    let mut out_swap = vec![0u32; 200];
    for row in 0..200u32 {
        out_base[row as usize] = (base.row_range(row).1 - base.row_range(row).0) as u32;
        out_swap[row as usize] = (swapped.row_range(row).1 - swapped.row_range(row).0) as u32;
    }
    assert_eq!(out_base, out_swap);
    // …but the edge SET changed (the control is not the identity).
    let mut diff = 0;
    for row in 0..200u32 {
        let (a, b) = base.row_range(row);
        let (c, d) = swapped.row_range(row);
        let set_a: std::collections::BTreeSet<u32> = base.targets()[a..b].iter().copied().collect();
        let set_b: std::collections::BTreeSet<u32> =
            swapped.targets()[c..d].iter().copied().collect();
        diff += set_a.symmetric_difference(&set_b).count();
    }
    assert!(diff > 0, "Maslov–Sneppen must change the edge set");
}

// ── 7. Non-vacuity: the parity fixtures actually exercise the active set ────

#[test]
fn parity_fixtures_exercise_active_set_churn() {
    // Phase A (cascade fixture): the event path's active set must churn
    // non-trivially mid-run.
    let edges = er_edges(300, 1200, 0.2, 400.0, 11 * 31);
    let mut r = LifReservoir::new(SignedAdjacency::from_edges(300, &edges), LifParams::shiu());
    let mut rng = FixtureRng::new(3);
    let mut max_active = 0;
    for t in 0..200 {
        drive(&mut r, t, 5, 5, 400.0, &mut rng, 300);
        r.step();
        max_active = max_active.max(r.active_count());
    }
    assert!(max_active > 50, "active set must churn (max={max_active})");
    // Phase B (subthreshold fixture): stop driving → everything decays to
    // bitwise quiescence (the fixed-point dropout works). The cascade
    // fixture would sustain itself forever — 80% excitatory suprathreshold
    // edges never let the ring go quiet — so the drain phase uses w=1.
    let edges_b = er_edges(300, 1200, 0.2, 1.0, 11 * 31);
    let mut r2 = LifReservoir::new(
        SignedAdjacency::from_edges(300, &edges_b),
        LifParams::shiu(),
    );
    let mut rng2 = FixtureRng::new(3);
    for t in 0..200 {
        drive(&mut r2, t, 5, 5, 400.0, &mut rng2, 300);
        r2.step();
    }
    assert!(r2.active_count() > 0);
    for _ in 0..4000 {
        r2.step();
    }
    assert_eq!(
        r2.active_count(),
        0,
        "quiescent dropout must drain the active set once drive stops"
    );
}

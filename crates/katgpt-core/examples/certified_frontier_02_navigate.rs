//! Example: grow-THEN-navigate end to end (Plan 580 T4.1).
//!
//! ```sh
//! cargo run --release --features certified_frontier \
//!   --example certified_frontier_02_navigate
//! ```
//!
//! [`certified_frontier`](katgpt_core::certified_frontier) answers *which
//! latent cells are provably valid* and
//! [`viable_manifold_graph`](katgpt_core::viable_manifold_graph) answers *how
//! to move between them without leaving the viable set*. Neither is useful
//! alone: navigation needs a node source, and a certified set you cannot
//! traverse is a list. This runs the join.
//!
//! Phase 0 (Bench 687) already showed the growth half beats passive sampling
//! 51.4x. What this adds is the check that matters for the composition: **every
//! node on the returned geodesic is a certified cell**, so a path produced by
//! the navigator never leaves the set the verifier vouched for.

use katgpt_core::certified_frontier::{
    CertifiedFrontier, FrontierConfig, SIGMOID_LIPSCHITZ, beta_union_bound,
    certified_manifold_graph,
};
use katgpt_core::subspace_phase_gate::JacobianSvdScratch;
use katgpt_core::viable_manifold_graph::{GraphBuildConfig, VolumeFieldConfig, manifold_geodesic};

const GRID: usize = 32;
const CELLS: usize = GRID * GRID;
const H: f32 = 0.6;
const AMP: f32 = 3.0;
const FREQ: f32 = 0.5;
const ROUNDS: u32 = 120_000;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % n
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Ground-truth validity probability. The algorithm only ever sees Bernoulli
/// draws from this, never the value.
fn p_true(x: f32, y: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    sigmoid(AMP * (tau * FREQ * x).cos() * (tau * FREQ * y).cos())
}

/// The decoder whose pullback metric defines navigability: `R^2 -> R^3`.
fn decode(z: &[f32], out: &mut [f32]) {
    out[0] = z[0];
    out[1] = z[1];
    out[2] = (2.0 * z[0]).sin() * (2.0 * z[1]).cos();
}

fn main() {
    let cfg = FrontierConfig {
        h: H,
        lipschitz: SIGMOID_LIPSCHITZ
            * AMP
            * std::f32::consts::TAU
            * FREQ
            * std::f32::consts::SQRT_2,
        cell_spacing: 1.0 / (GRID - 1) as f32,
        acquire_radius: 1.5 / (GRID - 1) as f32,
        ..FrontierConfig::default()
    };

    // ── grow ───────────────────────────────────────────────────────────────
    let mut f = Box::new(CertifiedFrontier::<CELLS, 2>::new());
    let mut truth = Vec::with_capacity(CELLS);
    for i in 0..CELLS {
        let (r, c) = (i / GRID, i % GRID);
        let (x, y) = (c as f32 / (GRID - 1) as f32, r as f32 / (GRID - 1) as f32);
        f.push_cell([x, y]).expect("capacity");
        truth.push(p_true(x, y));
    }
    // The tighter of the two widths — Bench 688 measured +33% growth at zero
    // violations against the paper's kernel-derived schedule.
    let beta = beta_union_bound(CELLS, ROUNDS, cfg.delta);
    let mut rng = Lcg::new(0x60A7);
    for t in 1..=ROUNDS {
        let i = rng.below(CELLS);
        f.observe(i, rng.next_f32() < truth[i]);
        f.expand_certified(&cfg, beta);
        if t % 30_000 == 0 {
            f.reachability_dilation(&cfg, 1);
        }
    }
    let violations = f
        .cells()
        .iter()
        .zip(truth.iter())
        .filter(|(c, p)| c.certified && **p < H)
        .count();
    println!(
        "grow: {} / {CELLS} cells certified ({} by dilation), {violations} violations, beta {beta:.3}",
        f.certified_count(),
        f.dilated_count()
    );
    assert_eq!(violations, 0, "certified an invalid cell");

    // ── navigate ───────────────────────────────────────────────────────────
    let volume_cfg = VolumeFieldConfig::default();
    let build_cfg = GraphBuildConfig {
        volume_threshold: 2.0,
        edge_midpoint_check: true,
        k_nearest: 6,
    };
    let mut scratch = JacobianSvdScratch::with_capacity(2, 3);
    let cmg = certified_manifold_graph(&f, decode, &volume_cfg, &build_cfg, &mut scratch);
    println!(
        "graph: {} nodes, {} edges ({} certified cells rejected by the volume threshold)",
        cmg.graph.n_nodes(),
        cmg.graph.edges().len(),
        cmg.rejected_by_volume
    );

    // Pick the farthest node REACHABLE from node 0, not the farthest node
    // outright. The valid region of this field has two disjoint lobes (the
    // product cos(pi x) cos(pi y) is positive in opposite corners), so the
    // globally farthest pair is guaranteed to be unreachable and the
    // demonstration would be vacuous — the geodesic assertions would never run.
    let src = 0u32;
    let n_nodes = cmg.graph.n_nodes();
    let (mut dst, mut best, mut reachable) = (None, -1.0f32, 0usize);
    for b in 1..n_nodes {
        if manifold_geodesic(&cmg.graph, src, b as u32).is_none() {
            continue;
        }
        reachable += 1;
        let (pa, pb) = (cmg.graph.node_latent(src), cmg.graph.node_latent(b as u32));
        let d = (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2);
        if d > best {
            best = d;
            dst = Some(b as u32);
        }
    }
    println!(
        "reachability from node 0: {} of {} nodes — the certified set has more \n\
         than one component, which is a fact about the field, not a failure.",
        reachable + 1,
        n_nodes
    );

    let dst = dst.expect("node 0 has no neighbours — the graph is all isolated nodes");
    let path = manifold_geodesic(&cmg.graph, src, dst).expect("just verified reachable");

    // THE point of the composition: navigation must never step outside what
    // the verifier vouched for.
    let all_certified = path
        .iter()
        .all(|n| f.cells()[cmg.node_to_cell[*n as usize] as usize].certified);
    let worst_cb = path
        .iter()
        .map(|n| f.cells()[cmg.node_to_cell[*n as usize] as usize].cb)
        .fold(f32::INFINITY, f32::min);
    let worst_p = path
        .iter()
        .map(|n| truth[cmg.node_to_cell[*n as usize] as usize])
        .fold(f32::INFINITY, f32::min);
    println!(
        "\ngeodesic: {} hops from node {src} to {dst} (latent distance {:.3})\n\
         every node certified      : {all_certified}\n\
         weakest certified bound   : {worst_cb:.4}  (threshold h = {H})\n\
         weakest TRUE p on the path: {worst_p:.4}  (ground truth, never read by the algorithm)",
        path.len() - 1,
        best.sqrt()
    );
    assert!(all_certified, "geodesic left the certified set");
    assert!(worst_cb >= H, "a path node was below its certified bound");
    assert!(
        worst_p >= H,
        "a path node was actually invalid — soundness breach"
    );
    println!("\nOK — the navigator stayed inside the verifier\'s certified set.");
}

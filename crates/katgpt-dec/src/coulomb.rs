//! Training-free Coulomb crowd redistribution (Issue 825 T3, opt-in
//! `coulomb_flow`).
//!
//! Distilled from Bhaskara–Torres–Menon, *Beckmann Transport with a Divergence
//! Constraint* (arXiv:2608.01692) Proposition 2: with the canonical Coulomb
//! field `b = ∇φ`, `Δφ = μ₀ − μ₁`, the first-hitting map of the autonomous flow
//! `Ẋ = b(X)` transports `μ₀ → μ₁` exactly, with **zero training**, provided
//! the target is singular. A zone graph is exactly the atomic case: zones are
//! vertices, current crowd density is `μ₀`, the authored target density is `μ₁`.
//!
//! # What this ships, and what the guarantee actually is
//!
//! The continuous first-hitting argument does not have to be transcribed. On a
//! graph with atomic targets the statement is shorter and it is an identity:
//!
//! 1. `δ(dφ) = Δφ` holds by definition of the rank-0 Laplacian, so solving
//!    `Δφ = μ₁ − μ₀` gives an edge flow `j = dφ` whose divergence is the
//!    density difference **by construction** — mass conservation is not
//!    approximated, it is the equation.
//! 2. Mass balance at a vertex reads `inflow + μ₀ = outflow + μ₁`. So the
//!    fraction of the mass passing through `v` that stops there is
//!    `μ₁(v) / (inflow(v) + μ₀(v))` — [`CrowdRouter::absorb_prob`]. A transit
//!    zone has `μ₁ = 0` and never absorbs; a pure sink has no outflow and
//!    absorbs everything.
//! 3. Routing out of `v` in proportion to the positive outgoing flow then sends
//!    **exactly `j_e`** along edge `e`: the un-absorbed mass is
//!    `avail − μ₁(v) = outflow(v)`, and `outflow(v) · j_e / out_total(v) = j_e`
//!    because `out_total(v) = outflow(v)`. The arrival distribution is `μ₁`,
//!    exactly, as a flow decomposition — no hitting-time argument is needed.
//! 4. A gradient flow is **acyclic** (every step strictly increases `φ`, since
//!    `j_e = φ(head) − φ(tail) > 0` is what makes `e` traversable), so routing
//!    terminates without a step cap.
//!
//! The only error in a per-NPC simulation is therefore **sampling** error,
//! which decays as `1/√N`. That is what the Issue 825 GOAT gate measures, and
//! it is why the gate is a decay rate rather than a single MAE.
//!
//! # Separation of concerns
//!
//! This module owns the math and the routing TABLE. It does not own the RNG:
//! [`CrowdRouter::step`] takes a caller-supplied uniform in `[0, 1)`, so the
//! consumer keeps its own seeded stream. katgpt-dec has zero dependencies and
//! an unseeded global draw is a gated defect class in this workspace
//! (`scripts/global_rng_gate.py`).

use crate::flow::DecFlowField;
use crate::hodge::{PoissonScratch, PoissonStats, poisson_solve_into};
use crate::operators::exterior_derivative_into;
use crate::types::{CellComplex, CochainField};

// ---------------------------------------------------------------------------
// CoulombFlowField
// ---------------------------------------------------------------------------

/// A zone-graph transport field solved from a current/target density pair.
///
/// Construct with [`CoulombFlowField::from_densities`]; consume per NPC with
/// [`CoulombFlowField::router`], or as per-vertex 2D velocities with
/// [`CoulombFlowField::to_flow_vectors`] when the zone layout is a grid.
pub struct CoulombFlowField {
    /// The solved mean-zero potential `φ` (rank-0 cochain).
    pub potential: CochainField,
    /// The transport flow `j = dφ` (rank-1 cochain, one signed value per edge).
    /// Positive = along edge orientation, tail → head.
    pub edge_flow: CochainField,
    /// What the Poisson solve did. `residual_inf` IS the conservation
    /// guarantee: `δ(j) − (μ₁ − μ₀)` is `Δφ − rhs` and nothing else.
    pub stats: PoissonStats,
}

impl CoulombFlowField {
    /// Solve `Δφ = μ₁ − μ₀` and build the transport flow `j = dφ`.
    ///
    /// `mu0` is the current crowd density over zones, `mu1` the authored
    /// target. Both are rank-0 quantities indexed by vertex.
    ///
    /// # Sign convention
    ///
    /// This crate's rank-0 Laplacian is `δd`, and its codifferential at rank 1
    /// is `δj(v) = inflow(v) − outflow(v)`. Mass balance therefore wants
    /// `δ(j) = μ₁ − μ₀`, which is the RHS used here. The paper writes
    /// `Δφ = μ₀ − μ₁` under the opposite Laplacian sign; the transported
    /// direction is the same.
    ///
    /// # Total mass
    ///
    /// `Δ` is singular on constants, so the system is consistent only when
    /// `μ₀` and `μ₁` carry equal total mass. Unequal totals are the caller's
    /// modelling error, not a solver failure — the mean of the RHS is
    /// projected out and the returned [`PoissonStats::residual_inf`] measures
    /// what is left. Normalise both sides first if that matters.
    ///
    /// # Panics
    ///
    /// Panics if `mu0` and `mu1` have different lengths, or if either does not
    /// match the vertex count of `cx`.
    #[must_use]
    pub fn from_densities(cx: &CellComplex, mu0: &[f32], mu1: &[f32]) -> Self {
        let n = cx.n_vertices();
        assert_eq!(mu0.len(), n, "from_densities: mu0 must be one per vertex");
        assert_eq!(mu1.len(), n, "from_densities: mu1 must be one per vertex");

        let mut potential = CochainField::zeros(0, n, 1);
        let mut scratch = PoissonScratch::new(n);
        let mut rhs = vec![0.0f32; n];
        for (r, (&a, &b)) in rhs.iter_mut().zip(mu1.iter().zip(mu0.iter())) {
            *r = a - b;
        }
        let stats = poisson_solve_into(cx, &rhs, &mut potential.data, &mut scratch);

        let mut edge_flow = CochainField::zeros(1, cx.n_edges(), 1);
        exterior_derivative_into(cx, &potential, &mut edge_flow);

        Self {
            potential,
            edge_flow,
            stats,
        }
    }

    /// Build the per-vertex routing table a crowd consumes.
    ///
    /// One solve, then `N` NPCs walk the same table — the table is the thing
    /// worth caching per redistribution event.
    #[must_use]
    pub fn router(&self, cx: &CellComplex, mu0: &[f32], mu1: &[f32]) -> CrowdRouter {
        CrowdRouter::from_flow(cx, &self.edge_flow, mu0, mu1)
    }

    /// Per-vertex 2D velocity vectors, via the shipped [`DecFlowField`] bridge.
    ///
    /// Returns `None` unless `cx` is a 2D grid complex — see
    /// [`DecFlowField::from_exact_flow`] for why a general zone graph cannot
    /// use this bridge. Use [`CoulombFlowField::router`] there instead.
    #[must_use]
    pub fn to_flow_vectors(&self, cx: &CellComplex) -> Option<Vec<[f32; 2]>> {
        DecFlowField::from_exact_flow(cx, &self.edge_flow).map(|f| f.to_flow_vectors())
    }

    /// Expected arrival distribution, computed exactly — no RNG, no sampling.
    ///
    /// Propagates mass along the flow DAG in ascending-`φ` order (a valid
    /// topological order, because a traversable edge has
    /// `j_e = φ(head) − φ(tail) > 0`). By the identity in the module docs this
    /// returns `μ₁` whenever the solve converged, so it is a **verification**
    /// utility rather than a prediction: a deviation means the flow's
    /// divergence is wrong, which shows up as a clamped absorption.
    ///
    /// `O(V log V + E)`, allocating. The per-NPC path is
    /// [`CrowdRouter::step`].
    ///
    /// ⛔ `router` MUST be the one built from THIS field's `edge_flow` — take
    /// it from [`CoulombFlowField::router`]. The topological order comes from
    /// `self.potential`, so a router built from some other flow (a hand-made
    /// attract field, say) would be walked in an order that is not a
    /// topological order of ITS graph, and the result would be a well-formed
    /// wrong number rather than an error. It is a parameter only so a caller
    /// that already built the table does not build it twice.
    #[must_use]
    pub fn expected_arrivals(&self, router: &CrowdRouter, mu0: &[f32]) -> Vec<f32> {
        let n = mu0.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let phi = &self.potential.data;
        order.sort_by(|&a, &b| {
            phi[a as usize]
                .partial_cmp(&phi[b as usize])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut carried = mu0.to_vec();
        let mut arrivals = vec![0.0f32; n];
        for &v in &order {
            let v = v as usize;
            let avail = carried[v];
            if avail <= 0.0 {
                continue;
            }
            let absorbed = avail * router.absorb[v];
            arrivals[v] += absorbed;
            let forward = avail - absorbed;
            let total = router.out_total[v];
            if forward <= 0.0 || total <= 0.0 {
                // Nowhere to go — the remainder stops here rather than
                // evaporating. Mass conservation is the claim; silently
                // dropping the residue would forfeit it.
                arrivals[v] += forward;
                continue;
            }
            let (lo, hi) = router.span(v);
            for (&d, &w) in router.dst[lo..hi].iter().zip(router.weight[lo..hi].iter()) {
                carried[d as usize] += forward * (w / total);
            }
        }
        arrivals
    }
}

// ---------------------------------------------------------------------------
// CrowdRouter
// ---------------------------------------------------------------------------

/// What an NPC standing on a zone should do this step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterStep {
    /// Stop here — this zone is the NPC's destination.
    Absorb,
    /// Move to the given vertex.
    Move(usize),
}

/// Per-vertex routing table derived from a transport flow.
///
/// CSR-flattened: no `Vec<Vec<_>>`, one allocation per array, so a crowd
/// walking it touches contiguous memory.
pub struct CrowdRouter {
    /// CSR row starts, length `n_vertices + 1`.
    starts: Vec<u32>,
    /// Destination vertex per outgoing route.
    dst: Vec<u32>,
    /// Positive flow magnitude per outgoing route.
    weight: Vec<f32>,
    /// Total outgoing flow per vertex (the CSR row sum, cached).
    out_total: Vec<f32>,
    /// Absorption probability per vertex, `μ₁(v) / (inflow(v) + μ₀(v))`.
    absorb: Vec<f32>,
}

impl CrowdRouter {
    /// Build the table from a transport flow and the density pair that produced it.
    ///
    /// Absorption is the CONSISTENT rule derived from the field itself — see
    /// [`consistent_absorption`](Self::consistent_absorption).
    ///
    /// # Panics
    ///
    /// Panics if `mu0`/`mu1` do not match the vertex count of `cx`, or if
    /// `edge_flow` is not a rank-1 single-channel cochain.
    #[must_use]
    pub fn from_flow(cx: &CellComplex, edge_flow: &CochainField, mu0: &[f32], mu1: &[f32]) -> Self {
        let table = RouteTable::build(cx, edge_flow);
        let absorb = consistent_absorption_from(&table.in_total, mu0, mu1);
        table.into_router(absorb)
    }

    /// Build the table with a caller-supplied absorption policy.
    ///
    /// Separating the routing TABLE from the stopping RULE is what makes an
    /// honest baseline comparison possible: a hand-built attract field can be
    /// walked by this exact code with `absorb[v] = 1.0` on the target zones
    /// (stop on first arrival — the only rule such a field can offer, since it
    /// carries no target weights to derive one from), so the two arms differ in
    /// one vector and nothing else. That asymmetry is the finding, and pooling
    /// it with a second walker implementation would let a reader credit the
    /// difference to the walker.
    ///
    /// # Panics
    ///
    /// Panics if `absorb` does not match the vertex count of `cx`, or if
    /// `edge_flow` is not a rank-1 single-channel cochain.
    #[must_use]
    pub fn from_flow_with_absorption(
        cx: &CellComplex,
        edge_flow: &CochainField,
        absorb: &[f32],
    ) -> Self {
        assert_eq!(
            absorb.len(),
            cx.n_vertices(),
            "from_flow_with_absorption: absorb must be one per vertex"
        );
        RouteTable::build(cx, edge_flow).into_router(absorb.to_vec())
    }

    /// The absorption vector implied by the field: `mu1(v) / (inflow(v) + mu0(v))`.
    ///
    /// Exposed because it is the whole of the endpoint guarantee and belongs in
    /// a test, not in a comment. See the module docs for the derivation.
    ///
    /// # Panics
    ///
    /// Panics if `mu0`/`mu1` do not match the vertex count of `cx`.
    #[must_use]
    pub fn consistent_absorption(
        cx: &CellComplex,
        edge_flow: &CochainField,
        mu0: &[f32],
        mu1: &[f32],
    ) -> Vec<f32> {
        let table = RouteTable::build(cx, edge_flow);
        consistent_absorption_from(&table.in_total, mu0, mu1)
    }

    /// CSR row bounds for vertex `v`.
    #[inline]
    fn span(&self, v: usize) -> (usize, usize) {
        (self.starts[v] as usize, self.starts[v + 1] as usize)
    }

    /// Probability that an NPC passing through `v` stops there.
    #[inline]
    #[must_use]
    pub fn absorb_prob(&self, v: usize) -> f32 {
        self.absorb[v]
    }

    /// Total outgoing flow at `v`. Zero means `v` is a terminal zone.
    #[inline]
    #[must_use]
    pub fn out_flow(&self, v: usize) -> f32 {
        self.out_total[v]
    }

    /// Number of vertices the table covers.
    #[inline]
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.out_total.len()
    }

    /// Advance one NPC standing on `v`, given a uniform draw `u01 ∈ [0, 1)`.
    ///
    /// The caller owns the RNG — this crate has no dependencies and takes no
    /// position on the consumer's stream or seed.
    ///
    /// The draw is used for the absorption decision first, then re-mapped onto
    /// the outgoing-flow CDF, so **one** uniform drives the whole step. That is
    /// deliberate: two draws per NPC per tick is the dominant cost at crowd
    /// scale, and the two decisions are conditionally independent given `v`
    /// (the remap `(u01 − p) / (1 − p)` is uniform on `[0, 1)` exactly when the
    /// absorb branch was not taken).
    ///
    /// # Panics
    ///
    /// Panics if `v` is out of range.
    #[must_use]
    pub fn step(&self, v: usize, u01: f32) -> RouterStep {
        let p = self.absorb[v];
        if u01 < p {
            return RouterStep::Absorb;
        }
        let total = self.out_total[v];
        let (lo, hi) = self.span(v);
        if total <= 0.0 || lo == hi {
            // Terminal zone: nowhere to go, so this IS the destination.
            return RouterStep::Absorb;
        }
        // Re-map the unused tail of the draw onto [0, 1).
        let u = if p < 1.0 { (u01 - p) / (1.0 - p) } else { 0.0 };
        let pick = u * total;
        let mut acc = 0.0f32;
        for (&d, &w) in self.dst[lo..hi].iter().zip(self.weight[lo..hi].iter()) {
            acc += w;
            if pick < acc {
                return RouterStep::Move(d as usize);
            }
        }
        // f32 rounding can leave `pick` a hair above `acc`; the last route is
        // the correct fallback, never a panic.
        RouterStep::Move(self.dst[hi - 1] as usize)
    }
}

// ---------------------------------------------------------------------------
// Route table construction (shared by both constructors)
// ---------------------------------------------------------------------------

/// The CSR routing table before a stopping policy is attached.
struct RouteTable {
    starts: Vec<u32>,
    dst: Vec<u32>,
    weight: Vec<f32>,
    out_total: Vec<f32>,
    in_total: Vec<f32>,
}

impl RouteTable {
    fn build(cx: &CellComplex, edge_flow: &CochainField) -> Self {
        assert_eq!(edge_flow.rank, 1, "RouteTable: edge_flow must be rank-1");
        assert_eq!(
            edge_flow.dim, 1,
            "RouteTable: edge_flow must be single-channel"
        );
        let n = cx.n_vertices();
        let n_edges = cx.n_edges();

        // Reassemble (tail, head) per edge from the rank-0 boundary triplets:
        // the boundary of an edge is (tail, e, -1) and (head, e, +1).
        let mut tail = vec![u32::MAX; n_edges];
        let mut head = vec![u32::MAX; n_edges];
        for &(v, e, sign) in cx.boundary_entries(0) {
            if sign < 0 {
                tail[e] = v as u32;
            } else {
                head[e] = v as u32;
            }
        }

        // A route is the POSITIVE direction of a non-zero edge flow.
        let route_of = |e: usize| -> Option<(u32, u32, f32)> {
            let (t, h) = (tail[e], head[e]);
            if t == u32::MAX || h == u32::MAX {
                return None; // a removed or malformed edge carries no route
            }
            let f = edge_flow.data[e];
            if f > 0.0 {
                Some((t, h, f))
            } else if f < 0.0 {
                Some((h, t, -f))
            } else {
                None
            }
        };

        // Pass 1: count per vertex and accumulate both totals.
        let mut counts = vec![0u32; n];
        let mut out_total = vec![0.0f32; n];
        let mut in_total = vec![0.0f32; n];
        for e in 0..n_edges {
            if let Some((from, to, mag)) = route_of(e) {
                counts[from as usize] += 1;
                out_total[from as usize] += mag;
                in_total[to as usize] += mag;
            }
        }

        // Prefix sum -> CSR row starts.
        let mut starts = vec![0u32; n + 1];
        let mut acc = 0u32;
        for (s, &c) in starts.iter_mut().zip(counts.iter()) {
            *s = acc;
            acc += c;
        }
        starts[n] = acc;

        // Pass 2: scatter.
        let mut cursor = starts.clone();
        let mut dst = vec![0u32; acc as usize];
        let mut weight = vec![0.0f32; acc as usize];
        for e in 0..n_edges {
            if let Some((from, to, mag)) = route_of(e) {
                let slot = cursor[from as usize] as usize;
                cursor[from as usize] += 1;
                dst[slot] = to;
                weight[slot] = mag;
            }
        }

        Self {
            starts,
            dst,
            weight,
            out_total,
            in_total,
        }
    }

    fn into_router(self, absorb: Vec<f32>) -> CrowdRouter {
        CrowdRouter {
            starts: self.starts,
            dst: self.dst,
            weight: self.weight,
            out_total: self.out_total,
            absorb,
        }
    }
}

/// `mu1(v) / (inflow(v) + mu0(v))`, clamped.
fn consistent_absorption_from(in_total: &[f32], mu0: &[f32], mu1: &[f32]) -> Vec<f32> {
    let n = in_total.len();
    assert_eq!(
        mu0.len(),
        n,
        "consistent_absorption: mu0 must be one per vertex"
    );
    assert_eq!(
        mu1.len(),
        n,
        "consistent_absorption: mu1 must be one per vertex"
    );
    (0..n)
        .map(|v| {
            let avail = in_total[v] + mu0[v];
            if avail <= 0.0 {
                // Nothing reaches `v`, so the ratio is undefined rather than 0
                // or 1. Zero is the safe reading: an NPC that is never there
                // never consults this entry.
                0.0
            } else {
                (mu1[v] / avail).clamp(0.0, 1.0)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::codifferential;

    /// The Issue 825 zone graph: a 4x3 lattice, three sinks with unequal
    /// authored weights — the discrete analog of the paper's 5-atom bias
    /// experiment.
    const W: usize = 4;
    const H: usize = 3;
    const N_V: usize = W * H;
    const SINKS: [(usize, f32); 3] = [(0, 0.5), (3, 0.3), (11, 0.2)];

    fn grid_edges() -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for y in 0..H {
            for x in 0..W {
                let v = y * W + x;
                if x + 1 < W {
                    e.push((v, y * W + x + 1));
                }
                if y + 1 < H {
                    e.push((v, (y + 1) * W + x));
                }
            }
        }
        e.shrink_to_fit();
        e
    }

    fn densities() -> (Vec<f32>, Vec<f32>) {
        let mut mu1 = vec![0.0f32; N_V];
        for (v, w) in SINKS {
            mu1[v] = w;
        }
        let sources: Vec<usize> = (0..N_V).filter(|v| mu1[*v] == 0.0).collect();
        let each = 1.0 / sources.len() as f32;
        let mut mu0 = vec![0.0f32; N_V];
        for v in sources {
            mu0[v] = each;
        }
        (mu0, mu1)
    }

    fn fixture() -> (CellComplex, Vec<f32>, Vec<f32>, CoulombFlowField) {
        let cx = CellComplex::from_edges(N_V, &grid_edges());
        let (mu0, mu1) = densities();
        let f = CoulombFlowField::from_densities(&cx, &mu0, &mu1);
        (cx, mu0, mu1, f)
    }

    #[test]
    fn poisson_solve_hits_the_f32_floor() {
        let (_, _, _, f) = fixture();
        assert!(
            f.stats.residual_inf <= 1e-6,
            "true residual {} above the f32 floor for this problem",
            f.stats.residual_inf
        );
    }

    #[test]
    fn refinement_is_not_inert() {
        // The PoC's breakdown guard silently discarded every refinement step,
        // and a solver doing nothing looks exactly like one that converged.
        // Assert the reported work is self-consistent.
        let (_, _, _, f) = fixture();
        assert_eq!(
            f.stats.cg_solves,
            1 + f.stats.refine_rounds,
            "one CG solve per refinement round, plus the initial one"
        );
    }

    #[test]
    fn divergence_of_the_flow_is_the_density_difference() {
        let (cx, mu0, mu1, f) = fixture();
        let div = codifferential(&cx, &f.edge_flow);
        let worst = (0..N_V)
            .map(|v| (div.data[v] - (mu1[v] - mu0[v])).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1e-6, "conservation residual {worst}");
    }

    #[test]
    fn divergence_sums_to_zero_structurally() {
        // Every edge contributes +1 at its head and -1 at its tail, so this
        // holds for ANY edge field and catches an orientation or indexing
        // error the residual above cannot.
        let (cx, _, _, f) = fixture();
        let div = codifferential(&cx, &f.edge_flow);
        assert!(div.data.iter().sum::<f32>().abs() <= 1e-6);
    }

    #[test]
    fn expected_arrivals_reproduce_the_authored_target() {
        let (cx, mu0, mu1, f) = fixture();
        let router = f.router(&cx, &mu0, &mu1);
        let arrivals = f.expected_arrivals(&router, &mu0);
        let worst = (0..N_V)
            .map(|v| (arrivals[v] - mu1[v]).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1e-5, "arrival deviation {worst} from mu1");
    }

    #[test]
    fn expected_arrivals_conserve_total_mass() {
        let (cx, mu0, mu1, f) = fixture();
        let router = f.router(&cx, &mu0, &mu1);
        let arrivals = f.expected_arrivals(&router, &mu0);
        let total: f32 = arrivals.iter().sum();
        assert!(
            (total - 1.0).abs() <= 1e-5,
            "total arrived mass {total}, expected 1.0"
        );
    }

    #[test]
    fn transit_zones_never_absorb() {
        let (cx, mu0, mu1, f) = fixture();
        let router = f.router(&cx, &mu0, &mu1);
        for (v, &m) in mu1.iter().enumerate() {
            if m == 0.0 {
                assert_eq!(router.absorb_prob(v), 0.0, "transit zone {v} absorbs");
            }
        }
    }

    #[test]
    fn routing_is_acyclic_in_the_potential() {
        // Every traversable edge strictly increases phi, which is what makes a
        // step cap unnecessary.
        let (cx, mu0, mu1, f) = fixture();
        let router = f.router(&cx, &mu0, &mu1);
        let phi = &f.potential.data;
        for v in 0..router.n_vertices() {
            let (lo, hi) = router.span(v);
            for &dst in &router.dst[lo..hi] {
                let d = dst as usize;
                assert!(
                    phi[d] > phi[v],
                    "route {v} -> {d} does not increase phi ({} -> {})",
                    phi[v],
                    phi[d]
                );
            }
        }
    }

    #[test]
    fn step_absorbs_below_the_absorption_probability() {
        let (cx, mu0, mu1, f) = fixture();
        let router = f.router(&cx, &mu0, &mu1);
        let sink = SINKS[0].0;
        let p = router.absorb_prob(sink);
        assert!(p > 0.0, "sink should absorb");
        assert_eq!(router.step(sink, 0.0), RouterStep::Absorb);
    }

    #[test]
    fn step_moves_along_a_positive_flow_edge() {
        let (cx, mu0, mu1, f) = fixture();
        let router = f.router(&cx, &mu0, &mu1);
        // Find a transit vertex with outgoing flow; it must never absorb and
        // must move to a vertex it actually has a route to.
        let v = (0..N_V)
            .find(|&v| mu1[v] == 0.0 && router.out_flow(v) > 0.0)
            .expect("the redistribution must have transit zones");
        let (lo, hi) = router.span(v);
        let targets: Vec<usize> = (lo..hi).map(|k| router.dst[k] as usize).collect();
        for &u in &[0.0f32, 0.25, 0.5, 0.75, 0.999] {
            match router.step(v, u) {
                RouterStep::Move(d) => assert!(targets.contains(&d)),
                RouterStep::Absorb => panic!("transit zone {v} absorbed at u={u}"),
            }
        }
    }

    #[test]
    fn step_at_a_terminal_zone_absorbs_whatever_the_draw() {
        let (cx, mu0, mu1, f) = fixture();
        let router = f.router(&cx, &mu0, &mu1);
        let terminal = (0..N_V).find(|&v| router.out_flow(v) <= 0.0);
        if let Some(v) = terminal {
            assert_eq!(router.step(v, 0.999_999), RouterStep::Absorb);
        }
    }

    #[test]
    fn sampled_arrivals_converge_on_the_target() {
        // The only error is sampling error. 20k walkers on a 12-zone graph
        // should land well inside 0.02 MAE; the 1/sqrt(N) decay itself is the
        // Bench 825 gate, not this test.
        let (cx, mu0, mu1, f) = fixture();
        let router = f.router(&cx, &mu0, &mu1);
        let mut rng = SplitMix64::new(0x825);
        let n = 20_000usize;
        let mut hist = [0.0f32; N_V];
        // Deal walkers over the sources in proportion to mu0.
        let mut starts = Vec::with_capacity(n);
        let mut acc = 0.0f32;
        for (v, &m) in mu0.iter().enumerate() {
            acc += m * n as f32;
            while starts.len() < acc.round() as usize && starts.len() < n {
                starts.push(v);
            }
        }
        while starts.len() < n {
            starts.push(starts[starts.len() - 1]);
        }
        for &s in &starts {
            let mut v = s;
            loop {
                match router.step(v, rng.next_f32()) {
                    RouterStep::Absorb => {
                        hist[v] += 1.0;
                        break;
                    }
                    RouterStep::Move(d) => v = d,
                }
            }
        }
        let mae = (0..N_V)
            .map(|v| (hist[v] / n as f32 - mu1[v]).abs())
            .sum::<f32>()
            / N_V as f32;
        assert!(mae <= 0.02, "sampled arrival MAE {mae}");
    }

    #[test]
    fn consistent_absorption_is_what_from_flow_uses() {
        let (cx, mu0, mu1, f) = fixture();
        let router = f.router(&cx, &mu0, &mu1);
        let direct = CrowdRouter::consistent_absorption(&cx, &f.edge_flow, &mu0, &mu1);
        for (v, &a) in direct.iter().enumerate() {
            assert_eq!(a, router.absorb_prob(v), "absorption differs at {v}");
        }
    }

    #[test]
    fn supplied_absorption_overrides_the_derived_rule() {
        // The G3 baseline shape: same table, same walker, a first-arrival
        // stopping rule instead of the derived one.
        let (cx, mu0, mu1, f) = fixture();
        let first_sink: Vec<f32> = (0..N_V)
            .map(|v| if mu1[v] > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let router = CrowdRouter::from_flow_with_absorption(&cx, &f.edge_flow, &first_sink);
        for (v, &a) in first_sink.iter().enumerate() {
            assert_eq!(router.absorb_prob(v), a);
        }
        // The routing table itself is untouched by the policy.
        let derived = f.router(&cx, &mu0, &mu1);
        for v in 0..router.n_vertices() {
            assert_eq!(
                router.out_flow(v),
                derived.out_flow(v),
                "table differs at {v}"
            );
            assert_eq!(router.span(v), derived.span(v), "CSR span differs at {v}");
        }
    }

    #[test]
    fn to_flow_vectors_refuses_a_non_grid_complex() {
        // `from_edges` builds a lattice-shaped zone graph, but the complex does
        // not KNOW it is a grid, and `to_flow_vectors` indexes by the `grid_2d`
        // edge layout. Returning vectors here would be well-formed and wrong.
        let (cx, _, _, f) = fixture();
        assert!(f.to_flow_vectors(&cx).is_none());
    }

    #[test]
    fn to_flow_vectors_bridges_a_real_grid() {
        let cx = CellComplex::grid_2d(W, H);
        let (mu0, mu1) = densities();
        let f = CoulombFlowField::from_densities(&cx, &mu0, &mu1);
        let v = f.to_flow_vectors(&cx).expect("a grid_2d complex bridges");
        assert_eq!(v.len(), N_V);
        assert!(v.iter().all(|c| c[0].is_finite() && c[1].is_finite()));
    }

    /// A seeded stream local to the tests. The crate itself has no RNG and no
    /// dependencies; the consumer owns the stream (see `step`).
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
        }
    }
}

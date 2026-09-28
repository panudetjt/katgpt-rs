//! Example: certified_frontier — Plan 580 Phase 0 PoC (T0.1 + T0.2).
//!
//! Source: [Research 510](../../../.research/510_ActFlow_Certified_Frontier_Expansion.md)
//! · De Santi et al., *Active Flow Expansion for Out-of-Distribution
//! Discovery*, arXiv:2606.08802 (safe-set expansion = SAFEOPT lineage,
//! Sui et al. 2015/2018).
//!
//! Run with:
//! ```sh
//! cargo run --release --example certified_frontier_01_basic
//! ```
//!
//! **Phase 0 is deliberately NOT feature-gated and depends on nothing** —
//! std only, hand-rolled LCG. Its whole job is to answer *"is the primitive
//! worth building?"* BEFORE Plan 580 Phase 1 spends 22 tasks on the real
//! fixed-capacity module. The exit criterion is stated in the plan:
//! certified growth with ZERO violations on the dense world, and a visible
//! passive-vs-frontier separation on the sparse world.
//!
//! # What This Proves
//!
//! - **Soundness of the certified set** (the paper's Lemma E.2 in miniature):
//!   every cell the algorithm marks certified really does satisfy
//!   `p(z) >= h`. Counted, not asserted — the run prints the violation count.
//! - **Monotone growth** (T2.3's property, here by construction): the
//!   certified lower bound `cb` only ever increases, so the certified set
//!   never shrinks. Checked each round.
//! - **Certification WITHOUT querying**: the Lipschitz relaxation admits
//!   cells that were never queried. That is the entire economic argument for
//!   the primitive — if every certified cell had to be queried, a plain
//!   per-cell Beta bound would do and no frontier machinery is needed. The
//!   run reports how many certified cells have zero queries.
//! - **The acquisition separation** (T0.2 / Prop 1): on a narrow-corridor
//!   world, frontier-targeted acquisition vs uniform passive sampling at an
//!   identical query budget.
//!
//! # What This Does NOT Prove
//!
//! - **Not the shipped primitive.** This is a heap-allocating grid mock-up.
//!   Phase 1's `CertifiedFrontier<const MAX_CELLS>` is fixed-capacity and
//!   zero-alloc, with the exact Eq 10 incremental-Cholesky posterior
//!   variance. Here the posterior is per-cell Beta-Bernoulli (plan item 2's
//!   "honest closed-form substitute"), which is why no kernel/Cholesky
//!   appears.
//! - **Not the paper's confidence schedule.** `beta_t` below is a plain
//!   union bound over (cells x rounds), monotone in t. Eq 31/37's
//!   `4·L_s·B + 2·L_s·sqrt(2κ/λ·(γ_t + log(1/δ)))` is Phase 1 T1.4 and its
//!   monotonicity pin is T2.4.
//! - **Not a Lipschitz-estimation result.** `L` is measured exactly on the
//!   grid (max adjacent |Δp| / d) rather than derived as `L_s·L_g`. That is
//!   deliberate: it isolates the question Phase 0 asks (does frontier
//!   acquisition beat passive sampling?) from a separate question (can you
//!   estimate L safely?). A real deployment must bound L a priori — with a
//!   too-small L the dilation is unsound, and THAT failure mode is invisible
//!   here by construction. Flagged, not measured.

use std::fmt::Write as _;

// ── deterministic RNG (std-only, no dev-dep) ───────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

// ── the world ──────────────────────────────────────────────────────────────

const H: f32 = 0.60; // validity threshold on p(z)

#[derive(Clone, Copy, PartialEq, Eq)]
enum World {
    /// Smooth block checkerboard — large contiguous valid regions.
    ///
    /// NOTE: a *literal* alternating checkerboard would be pointless here —
    /// single-cell alternation has unbounded Lipschitz constant, so no
    /// dilation could ever be sound and the certified set could only ever be
    /// the queried set. The paper's illustrative setup is a smooth field
    /// thresholded, which is what this is.
    Checkerboard,
    /// A narrow valid corridor: small measure, so uniform sampling rarely
    /// lands inside it. This is the Prop-1 separation setup.
    Corridor,
}

fn cell_xy(i: usize, j: usize, n: usize) -> (f32, f32) {
    (i as f32 / (n - 1) as f32, j as f32 / (n - 1) as f32)
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// The latent score field `g`; validity probability is `p = sigmoid(g)`.
fn g_field(w: World, x: f32, y: f32) -> f32 {
    match w {
        World::Checkerboard => {
            const AMP: f32 = 3.0;
            const K: f32 = 1.0; // one full period across the unit square
            AMP * (std::f32::consts::TAU * K * x).cos() * (std::f32::consts::TAU * K * y).cos()
        }
        World::Corridor => {
            // A diagonal corridor of half-width `w` — smooth, and narrow
            // enough that its area is a small fraction of the square.
            const AMP: f32 = 3.0;
            const WIDTH: f32 = 0.10;
            let perp = (y - x) / std::f32::consts::SQRT_2; // signed distance to y = x
            AMP * (1.0 - 2.0 * (perp / WIDTH).powi(2))
        }
    }
}

// ── the certified frontier (Phase 0 mock-up) ───────────────────────────────

struct Frontier {
    /// True validity probability per cell (the world; NEVER read by the
    /// algorithm — only sampled through `query`).
    p_true: Vec<f32>,
    valid: Vec<u32>,
    invalid: Vec<u32>,
    /// Certified lower bound on `p(z)`. Monotone non-decreasing by
    /// construction — this is what makes the certified set monotone (T2.3).
    cb: Vec<f32>,
    /// Grid-exact Lipschitz constant of `p` in probability space.
    lipschitz: f32,
    /// Distance between 4-neighbours.
    spacing: f32,
    n: usize,
    /// Certification events by CAUSE. Counting end-state "certified but
    /// never queried" is confounded: the frontier policy hands max posterior
    /// sd to any freshly certified cell, so it gets queried moments later
    /// and the end-state count reads 0 even when dilation did all the work.
    /// These count the moment `cb` first crosses `H`.
    cert_direct: usize,
    cert_dilated: usize,
}

impl Frontier {
    fn new(w: World, n: usize) -> Self {
        let mut p_true = vec![0.0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                let (x, y) = cell_xy(i, j, n);
                p_true[j * n + i] = sigmoid(g_field(w, x, y));
            }
        }
        // Measure L exactly on the grid: the largest probability change
        // between 4-neighbours, per unit distance. Dilation only ever hops
        // between neighbours, so this constant makes the hop rule sound by
        // construction (see the module doc's caveat).
        let spacing = 1.0 / (n - 1) as f32;
        let mut max_step = 0.0f32;
        for j in 0..n {
            for i in 0..n {
                let p = p_true[j * n + i];
                if i + 1 < n {
                    max_step = max_step.max((p - p_true[j * n + i + 1]).abs());
                }
                if j + 1 < n {
                    max_step = max_step.max((p - p_true[(j + 1) * n + i]).abs());
                }
            }
        }
        Self {
            p_true,
            valid: vec![0; n * n],
            invalid: vec![0; n * n],
            cb: vec![f32::NEG_INFINITY; n * n],
            lipschitz: max_step / spacing,
            spacing,
            n,
            cert_direct: 0,
            cert_dilated: 0,
        }
    }

    /// The per-hop cost of the Lipschitz relaxation: how much certified
    /// lower bound a cell forfeits per grid step. Equals the largest
    /// adjacent probability change on this grid.
    fn hop_decrement(&self) -> f32 {
        self.lipschitz * self.spacing
    }

    /// Beta-Bernoulli posterior mean and standard deviation (plan item 2).
    fn beta_mean_sd(&self, c: usize) -> (f32, f32) {
        let a = 1.0 + self.valid[c] as f32;
        let b = 1.0 + self.invalid[c] as f32;
        let n = a + b;
        (a / n, ((a * b) / (n * n * (n + 1.0))).sqrt())
    }

    /// Union bound over (cells x rounds) — monotone in `t`. A stand-in for
    /// the paper's Eq 31; see the module doc.
    fn beta_t(&self, t: usize, delta: f32) -> f32 {
        let t = t.max(1) as f32;
        (2.0 * ((self.n * self.n) as f32 * t * t / delta).ln()).sqrt()
    }

    fn lcb(&self, c: usize, beta: f32) -> f32 {
        let (m, s) = self.beta_mean_sd(c);
        m - beta * s
    }

    /// Query the verifier once at `c`. The verifier is BINARY and
    /// stochastic — the algorithm never sees `p_true`.
    fn query(&mut self, c: usize, rng: &mut Lcg) {
        if rng.next_f32() < self.p_true[c] {
            self.valid[c] += 1;
        } else {
            self.invalid[c] += 1;
        }
    }

    fn neighbours(&self, z: usize) -> ([usize; 4], usize) {
        let (i, j, n) = (z % self.n, z / self.n, self.n);
        let mut out = [usize::MAX; 4];
        let mut k = 0;
        if i > 0 {
            out[k] = z - 1;
            k += 1;
        }
        if i + 1 < n {
            out[k] = z + 1;
            k += 1;
        }
        if j > 0 {
            out[k] = z - n;
            k += 1;
        }
        if j + 1 < n {
            out[k] = z + n;
            k += 1;
        }
        (out, k)
    }

    /// Eq 15 / Eq 32: relax the certified lower bound outward from `c`.
    /// `cb` only ever increases, so this terminates and the certified set is
    /// monotone.
    fn relax_from(&mut self, c: usize) {
        let dec = self.hop_decrement();
        let mut stack = vec![c];
        while let Some(z) = stack.pop() {
            let base = self.cb[z];
            if !base.is_finite() {
                continue;
            }
            let (nb, k) = self.neighbours(z);
            for &w in &nb[..k] {
                let cand = base - dec;
                if cand > self.cb[w] {
                    let was = self.cb[w] >= H;
                    self.cb[w] = cand;
                    if !was && cand >= H {
                        self.cert_dilated += 1;
                    }
                    stack.push(w);
                }
            }
        }
    }

    fn observe(&mut self, c: usize, rng: &mut Lcg, t: usize, delta: f32) {
        self.query(c, rng);
        let l = self.lcb(c, self.beta_t(t, delta));
        if l > self.cb[c] {
            let was = self.cb[c] >= H;
            self.cb[c] = l;
            if !was && l >= H {
                self.cert_direct += 1;
            }
            self.relax_from(c);
        }
    }

    fn is_certified(&self, c: usize) -> bool {
        self.cb[c] >= H
    }
    fn cells(&self) -> usize {
        self.n * self.n
    }
    fn certified_count(&self) -> usize {
        (0..self.cells()).filter(|&c| self.is_certified(c)).count()
    }
    /// Cells the algorithm claims are valid but which actually are not —
    /// the soundness scoreboard.
    fn violations(&self) -> usize {
        (0..self.cells())
            .filter(|&c| self.is_certified(c) && self.p_true[c] < H)
            .count()
    }
    fn truly_valid_count(&self) -> usize {
        (0..self.cells()).filter(|&c| self.p_true[c] >= H).count()
    }
    /// The best certified lower bound reached anywhere — the "headroom"
    /// numerator in the dilation-feasibility law.
    fn best_cb(&self) -> f32 {
        (0..self.cells()).fold(f32::NEG_INFINITY, |m, c| m.max(self.cb[c]))
    }

    /// Candidate query set: the certified set plus its immediate boundary.
    /// "Safe uncertainty sampling" (Eq 33) — only look where you can already
    /// stand, or one hop out.
    fn candidates(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for c in 0..self.cells() {
            if self.is_certified(c) {
                out.push(c);
                continue;
            }
            let (nb, k) = self.neighbours(c);
            if nb[..k].iter().any(|&w| self.is_certified(w)) {
                out.push(c);
            }
        }
        out.shrink_to_fit();
        out
    }

    fn ascii_map(&self) -> String {
        let mut s = String::new();
        for j in (0..self.n).rev() {
            for i in 0..self.n {
                let c = j * self.n + i;
                s.push(match (self.is_certified(c), self.p_true[c] >= H) {
                    (true, true) => '#',
                    (true, false) => 'X', // VIOLATION
                    (false, true) => '.',
                    (false, false) => ' ',
                });
            }
            s.push('\n');
        }
        let _ = write!(
            s,
            "  legend: '#' certified-correct  'X' VIOLATION  '.' valid-uncertified"
        );
        s
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Strategy {
    /// argmax posterior sd over the safe set + its boundary.
    Frontier,
    /// Uniform over the whole grid — the passive baseline.
    Passive,
}

struct Run {
    certified: usize,
    violations: usize,
    truly_valid: usize,
    monotone_ok: bool,
    hop_decrement: f32,
    best_cb: f32,
    cert_direct: usize,
    cert_dilated: usize,
    map: String,
}

fn run(w: World, n: usize, strategy: Strategy, budget: usize, seed: u64, delta: f32) -> Run {
    let mut f = Frontier::new(w, n);
    let mut rng = Lcg::new(seed);

    // Seed set: one known-valid cell, per the plan. Give it enough queries
    // to clear the bound so the frontier has somewhere to stand.
    let seed_cell =
        (0..f.cells()).fold(0usize, |b, c| if f.p_true[c] > f.p_true[b] { c } else { b });
    for t in 1..=40 {
        f.observe(seed_cell, &mut rng, t, delta);
    }

    let mut monotone_ok = true;
    let mut prev = f.certified_count();
    for t in 1..=budget {
        let pick = match strategy {
            Strategy::Passive => rng.below(f.cells()),
            Strategy::Frontier => {
                let cands = f.candidates();
                if cands.is_empty() {
                    rng.below(f.cells())
                } else {
                    let mut best = cands[0];
                    let mut best_sd = -1.0f32;
                    for &c in &cands {
                        let (_, sd) = f.beta_mean_sd(c);
                        if sd > best_sd {
                            best_sd = sd;
                            best = c;
                        }
                    }
                    best
                }
            }
        };
        f.observe(pick, &mut rng, t + 40, delta);
        let now = f.certified_count();
        if now < prev {
            monotone_ok = false;
        }
        prev = now;
    }

    Run {
        certified: f.certified_count(),
        violations: f.violations(),
        truly_valid: f.truly_valid_count(),
        monotone_ok,
        hop_decrement: f.hop_decrement(),
        best_cb: f.best_cb(),
        cert_direct: f.cert_direct,
        cert_dilated: f.cert_dilated,
        map: f.ascii_map(),
    }
}

fn main() {
    const DELTA: f32 = 0.05;
    const BUDGET: usize = 6000;
    const N0: usize = 32;

    println!("=== Plan 580 Phase 0 — certified frontier PoC ===");
    println!("h = {H}, delta = {DELTA}, budget = {BUDGET} queries\n");

    // ── T0.1: dense world, soundness + growth ──────────────────────────────
    println!("--- T0.1  dense (smooth block checkerboard), grid {N0}x{N0} ---");
    let d = run(
        World::Checkerboard,
        N0,
        Strategy::Frontier,
        BUDGET,
        0xC0FFEE,
        DELTA,
    );
    println!(
        "certified {}/{} truly-valid | violations {} | by-cause: direct {} dilated {} | monotone {}",
        d.certified, d.truly_valid, d.violations, d.cert_direct, d.cert_dilated, d.monotone_ok
    );
    println!("{}", d.map);

    // ── T0.2: sparse corridor, passive vs frontier at equal budget ─────────
    println!("\n--- T0.2  sparse (narrow corridor): passive vs frontier, grid {N0}x{N0} ---");
    let (mut pt, mut ft, mut pv, mut fv) = (0usize, 0usize, 0usize, 0usize);
    let seeds = [1u64, 2, 3, 4, 5];
    for &s in &seeds {
        let p = run(World::Corridor, N0, Strategy::Passive, BUDGET, s, DELTA);
        let fr = run(World::Corridor, N0, Strategy::Frontier, BUDGET, s, DELTA);
        println!(
            "  seed {s}: passive {:>4} (viol {})   frontier {:>4} (viol {})",
            p.certified, p.violations, fr.certified, fr.violations
        );
        pt += p.certified;
        ft += fr.certified;
        pv += p.violations;
        fv += fr.violations;
    }
    let k = seeds.len() as f32;
    let (pass_avg, front_avg) = (pt as f32 / k, ft as f32 / k);
    println!(
        "\n  mean certified @ {BUDGET}: passive {pass_avg:.1}  frontier {front_avg:.1}  \
         separation {:.1}x",
        if pass_avg > 0.0 {
            front_avg / pass_avg
        } else {
            f32::INFINITY
        }
    );
    println!("  total violations: passive {pv}, frontier {fv}");

    // ── T0.3 (added): does the Lipschitz dilation contribute anything? ─────
    //
    // The plan's exit criterion does not ask this, but it decides whether
    // functions 4 + 5 (reachability_dilation / expand_certified) are worth
    // building at all: if every certified cell had to be queried anyway, a
    // plain per-cell Beta bound is the whole primitive.
    //
    // The law under test: a hop is admissible iff the achievable certified
    // lower bound clears the threshold by at least one hop's decrement,
    //     best_cb - H  >=  L * spacing,
    // and L*spacing is just the largest adjacent |dp| on the grid — so it
    // shrinks with resolution while the headroom does not.
    println!("\n--- T0.3  resolution sweep: is dilation feasible? ---");
    println!(
        "  {:>7} {:>10} {:>10} {:>10} {:>8} {:>9} {:>6}",
        "grid", "hop_decr", "headroom", "certified", "direct", "dilated", "viol"
    );
    for &n in &[16usize, 32, 64, 96] {
        let r = run(
            World::Checkerboard,
            n,
            Strategy::Frontier,
            BUDGET,
            0xC0FFEE,
            DELTA,
        );
        println!(
            "  {:>7} {:>10.4} {:>10.4} {:>10} {:>8} {:>9} {:>6}",
            format!("{n}x{n}"),
            r.hop_decrement,
            r.best_cb - H,
            r.certified,
            r.cert_direct,
            r.cert_dilated,
            r.violations
        );
    }

    // ── the Phase 0 exit criterion, evaluated ──────────────────────────────
    println!("\n=== Phase 0 exit criterion (plan-stated) ===");
    let sound = d.violations == 0 && fv == 0 && pv == 0;
    println!(
        "  zero violations ............. {}",
        if sound { "PASS" } else { "FAIL" }
    );
    println!(
        "  certified growth, monotone .. {}",
        if d.certified > 1 && d.monotone_ok {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "  passive/frontier separation . {}",
        if front_avg > pass_avg { "PASS" } else { "FAIL" }
    );
}

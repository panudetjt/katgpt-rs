//! Plan 580 Phase 3 — certified-frontier GOAT gate (bench 688).
//!
//! - **G2 perf** (T3.1): batch acquisition + expansion at crowd scale, plus the
//!   exact Eq-10 posterior variance at the plan's stated buffer size
//!   (N=256, D=8). Release-only assertions.
//! - **G3 no-regression** (T3.3): the feature is opt-in and this file is
//!   `#![cfg]`-gated; the default surface is untouched until promotion.
//! - **T3.4 Report-the-Floor**: the primitive claims a coverage guarantee, so
//!   it is UQ-bearing and must be benchmarked against the naive floor —
//!   **adjacency-only expansion**: certify any lattice neighbour of a cell
//!   whose tally leans valid, with no uncertainty model at all.
//!
//! ```sh
//! cargo test --release -p katgpt-core --features certified_frontier \
//!   --test bench_688_certified_frontier_goat -- --nocapture
//! ```
#![cfg(feature = "certified_frontier")]

use katgpt_core::certified_frontier::{
    CertifiedFrontier, FrontierConfig, PosteriorBuffer, SIGMOID_LIPSCHITZ, beta_union_bound,
    confidence_schedule,
};
use std::hint::black_box;
use std::time::Instant;

// ── shared world (identical to the Phase 2 suite) ──────────────────────────

const GRID: usize = 48;
const CELLS: usize = GRID * GRID;
const H: f32 = 0.6;
const AMP: f32 = 3.0;
const FREQ: f32 = 1.0;
const ROUNDS: u32 = 200_000;
const DILATE_EVERY: u32 = 50_000;
const SEEDS: u64 = 5;
/// The configured failure probability. A UQ-bearing primitive is calibrated
/// only if its measured violation rate stays under this.
const DELTA: f32 = 0.05;

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
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

fn world() -> Vec<f32> {
    let tau = std::f32::consts::TAU;
    (0..CELLS)
        .map(|i| {
            let (r, c) = (i / GRID, i % GRID);
            let (x, y) = (c as f32 / (GRID - 1) as f32, r as f32 / (GRID - 1) as f32);
            sigmoid(AMP * (tau * FREQ * x).cos() * (tau * FREQ * y).cos())
        })
        .collect()
}

fn cfg() -> FrontierConfig {
    FrontierConfig {
        h: H,
        lipschitz: SIGMOID_LIPSCHITZ
            * AMP
            * std::f32::consts::TAU
            * FREQ
            * std::f32::consts::SQRT_2,
        cell_spacing: 1.0 / (GRID - 1) as f32,
        delta: DELTA,
        ..FrontierConfig::default()
    }
}

// ── T3.4 — the two arms ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct ArmResult {
    certified: usize,
    violations: usize,
}

impl ArmResult {
    fn violation_rate(&self) -> f64 {
        match self.certified {
            0 => 0.0,
            n => self.violations as f64 / n as f64,
        }
    }
    /// The plan's stated composite: `growth * (1 - violation_rate)`.
    fn product(&self) -> f64 {
        self.certified as f64 * (1.0 - self.violation_rate())
    }
}

/// The primitive: LCB expansion + periodic Lipschitz dilation.
fn arm_primitive(seed: u64, truth: &[f32]) -> ArmResult {
    let c = cfg();
    let mut f = Box::new(CertifiedFrontier::<CELLS, 2>::new());
    for i in 0..CELLS {
        let (r, col) = (i / GRID, i % GRID);
        f.push_cell([col as f32 / (GRID - 1) as f32, r as f32 / (GRID - 1) as f32])
            .expect("capacity");
    }
    let mut rng = Lcg::new(seed);
    for t in 1..=ROUNDS {
        let i = rng.below(CELLS);
        f.observe(i, rng.next_f32() < truth[i]);
        let beta = confidence_schedule(t, c.delta, c.lambda, c.b_rkhs, 2);
        f.expand_certified(&c, beta);
        if t % DILATE_EVERY == 0 {
            f.reachability_dilation(&c, 1);
        }
    }
    ArmResult {
        certified: f.certified_count() as usize,
        violations: f
            .cells()
            .iter()
            .zip(truth.iter())
            .filter(|(cell, p)| cell.certified && **p < H)
            .count(),
    }
}

/// The floor: adjacency-only expansion. No posterior, no `beta`, no Lipschitz —
/// a cell is certified if it or any 4-neighbour has a tally that leans valid.
///
/// Given the SAME query sequence as the primitive, so the arms differ only in
/// the certification rule.
fn arm_adjacency_floor(seed: u64, truth: &[f32]) -> ArmResult {
    let mut valid = vec![0u32; CELLS];
    let mut invalid = vec![0u32; CELLS];
    let mut rng = Lcg::new(seed);
    for _ in 1..=ROUNDS {
        let i = rng.below(CELLS);
        if rng.next_f32() < truth[i] {
            valid[i] += 1
        } else {
            invalid[i] += 1
        }
    }
    let leans_valid = |i: usize| valid[i] > invalid[i] && valid[i] + invalid[i] > 0;
    let mut certified = vec![false; CELLS];
    for (i, cert) in certified.iter_mut().enumerate() {
        let (r, c) = (i / GRID, i % GRID);
        let mut ok = leans_valid(i);
        if !ok {
            if r > 0 {
                ok |= leans_valid(i - GRID);
            }
            if r + 1 < GRID {
                ok |= leans_valid(i + GRID);
            }
            if c > 0 {
                ok |= leans_valid(i - 1);
            }
            if c + 1 < GRID {
                ok |= leans_valid(i + 1);
            }
        }
        *cert = ok;
    }
    ArmResult {
        certified: certified.iter().filter(|c| **c).count(),
        violations: certified
            .iter()
            .zip(truth.iter())
            .filter(|(c, p)| **c && **p < H)
            .count(),
    }
}

#[test]
fn t3_4_report_the_floor_adjacency_only_expansion() {
    let truth = world();
    let n_valid = truth.iter().filter(|p| **p >= H).count();
    println!(
        "\n=== T3.4 Report-the-Floor: certified frontier vs adjacency-only ===\n\
         world {GRID}x{GRID} = {CELLS} cells, {n_valid} truly valid (h={H}), \
         {ROUNDS} queries/seed, delta={DELTA}\n"
    );
    println!(
        "{:>4} | {:>9} {:>6} {:>8} {:>9} | {:>9} {:>6} {:>8} {:>9} | {:>7}",
        "seed",
        "prim.cert",
        "viol",
        "rate",
        "product",
        "floor.cert",
        "viol",
        "rate",
        "product",
        "ratio"
    );

    let mut prim_wins = 0usize;
    let mut ratios = Vec::with_capacity(SEEDS as usize);
    let (mut prim_viol, mut floor_viol) = (0usize, 0usize);
    let (mut prim_cert, mut floor_cert) = (0usize, 0usize);

    for seed in 0..SEEDS {
        let p = arm_primitive(seed, &truth);
        let fl = arm_adjacency_floor(seed, &truth);
        // Paired per-seed ratio — never median(A)/median(B), which can invert
        // a verdict when the arms are correlated across seeds.
        let ratio = p.product() / fl.product().max(1e-9);
        ratios.push(ratio);
        prim_wins += usize::from(ratio > 1.0);
        prim_viol += p.violations;
        floor_viol += fl.violations;
        prim_cert += p.certified;
        floor_cert += fl.certified;
        println!(
            "{seed:>4} | {:>9} {:>6} {:>8.4} {:>9.1} | {:>9} {:>6} {:>8.4} {:>9.1} | {ratio:>7.3}",
            p.certified,
            p.violations,
            p.violation_rate(),
            p.product(),
            fl.certified,
            fl.violations,
            fl.violation_rate(),
            fl.product(),
        );
    }

    let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let prim_rate = prim_viol as f64 / prim_cert.max(1) as f64;
    let floor_rate = floor_viol as f64 / floor_cert.max(1) as f64;
    println!(
        "\nmean paired product ratio (primitive / floor): {mean_ratio:.3} \
         — primitive wins {prim_wins}/{SEEDS}"
    );
    println!(
        "pooled violation rate: primitive {prim_rate:.5}, floor {floor_rate:.5}, delta {DELTA}"
    );

    // The claim under test is a COVERAGE GUARANTEE, so calibration is the gate
    // that cannot be traded away: a certified set that breaches delta is not a
    // certified set, however large it grew. This is the one hard assertion.
    assert!(
        prim_rate <= DELTA as f64,
        "primitive breached its own delta: {prim_rate} > {DELTA}"
    );

    // The plan's stated composite is REPORTED, not asserted, because it prices
    // a guarantee breach linearly against growth — a floor that certifies
    // everything and violates 20% of the time scores 0.8x a sound arm rather
    // than failing outright. The verdict is written up in .benchmarks/688
    // against both numbers; see the file for the scope call.
    println!(
        "\nverdict inputs: product-metric {} | calibration {}",
        if mean_ratio > 1.0 {
            "primitive DOMINATES floor"
        } else {
            "floor DOMINATES primitive"
        },
        if floor_rate <= DELTA as f64 {
            "floor is ALSO calibrated"
        } else {
            "floor BREACHES delta (primitive is the only deployable arm)"
        }
    );
}

/// The primitive at a scaled confidence width — a DIAGNOSTIC, not a shipping
/// mode. Scaling `beta` below 1.0 breaks the union bound the soundness proof
/// rests on; this exists only to locate where `delta` actually binds, so the
/// T3.4 verdict can say whether the primitive's modest growth is a loose bound
/// or a hard budget limit.
fn arm_primitive_scaled(seed: u64, truth: &[f32], scale: f32) -> ArmResult {
    let c = cfg();
    let mut f = Box::new(CertifiedFrontier::<CELLS, 2>::new());
    for i in 0..CELLS {
        let (r, col) = (i / GRID, i % GRID);
        f.push_cell([col as f32 / (GRID - 1) as f32, r as f32 / (GRID - 1) as f32])
            .expect("capacity");
    }
    let mut rng = Lcg::new(seed);
    for t in 1..=ROUNDS {
        let i = rng.below(CELLS);
        f.observe(i, rng.next_f32() < truth[i]);
        let beta = scale * confidence_schedule(t, c.delta, c.lambda, c.b_rkhs, 2);
        f.expand_certified(&c, beta);
        if t % DILATE_EVERY == 0 {
            f.reachability_dilation(&c, 1);
        }
    }
    ArmResult {
        certified: f.certified_count() as usize,
        violations: f
            .cells()
            .iter()
            .zip(truth.iter())
            .filter(|(cell, p)| cell.certified && **p < H)
            .count(),
    }
}

#[test]
fn t3_4b_where_delta_actually_binds() {
    // The question the floor comparison raises: is the primitive's 300-cell
    // growth a LOOSE BOUND (fixable) or a BUDGET LIMIT (not)? Sweep the
    // confidence width and watch when violations appear.
    let truth = world();
    let n_valid = truth.iter().filter(|p| **p >= H).count();
    println!(
        "\n=== T3.4b confidence-width sweep (diagnostic — scale < 1 is UNSOUND) ===\n\
         {n_valid} truly valid of {CELLS}, {ROUNDS} queries/seed, delta={DELTA}\n"
    );
    println!(
        "{:>6} | {:>9} {:>6} {:>8} | calibrated?",
        "scale", "certified", "viol", "rate"
    );
    for &scale in &[1.0f32, 0.75, 0.5, 0.25, 0.1] {
        let (mut cert, mut viol) = (0usize, 0usize);
        for seed in 0..SEEDS {
            let r = arm_primitive_scaled(seed, &truth, scale);
            cert += r.certified;
            viol += r.violations;
        }
        let rate = viol as f64 / cert.max(1) as f64;
        println!(
            "{scale:>6.2} | {:>9} {:>6} {:>8.5} | {}",
            cert / SEEDS as usize,
            viol,
            rate,
            if rate <= DELTA as f64 { "yes" } else { "NO" }
        );
    }
    println!(
        "\nRead: if growth stays far below {n_valid} even where calibration first breaks,\n\
         the limit is the query budget (~{} observations/cell), not the bound's tightness.",
        ROUNDS as usize / CELLS
    );

    // The shipped alternative: a width derived from the comparison count
    // instead of an RKHS norm. Reported next to the sweep so the reader can
    // see WHERE on the curve it lands, and measured for calibration — the
    // check its own doc comment says a caller owes.
    let paper = confidence_schedule(ROUNDS, DELTA, 1.0, 1.0, 2);
    let union = beta_union_bound(CELLS, ROUNDS, DELTA);
    let (mut cert, mut viol) = (0usize, 0usize);
    for seed in 0..SEEDS {
        let r = arm_primitive_scaled(seed, &truth, union / paper);
        cert += r.certified;
        viol += r.violations;
    }
    let rate = viol as f64 / cert.max(1) as f64;
    println!(
        "\nbeta_union_bound: {union:.3} vs paper schedule {paper:.3} \
         (= scale {:.2}) -> {} certified, {viol} violations, rate {rate:.5} [{}]",
        union / paper,
        cert / SEEDS as usize,
        if rate <= DELTA as f64 {
            "calibrated"
        } else {
            "BREACHES delta"
        }
    );
}

// ── T3.1 — G2 perf ─────────────────────────────────────────────────────────

#[test]
fn t3_1_g2_perf_batch_acquisition_and_expansion_at_crowd_scale() {
    const CROWD: usize = 1000;
    const D: usize = 8;
    const POOL: usize = 1024;

    let c = FrontierConfig {
        h: H,
        acquire_radius: 0.35,
        ..FrontierConfig::default()
    };
    let mut f = Box::new(CertifiedFrontier::<POOL, D>::new());
    let mut rng = Lcg::new(0x9A7E);
    for _ in 0..POOL {
        f.push_cell(std::array::from_fn(|_| rng.next_f32()))
            .unwrap();
    }
    // Warm the frontier so acquisition has a realistic candidate population.
    for i in 0..POOL {
        for _ in 0..24 {
            f.observe(i, rng.next_f32() < 0.85);
        }
    }
    f.expand_certified(&c, 2.0);
    let certified = f.certified_count();
    assert!(certified > 0, "warmup certified nothing — bench is vacuous");

    // The deployed shape: every NPC acquires and reports, then ONE expansion
    // pass folds the batch in. Amortise the pass over the batch.
    let mut sink = 0usize;
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        for _ in 0..CROWD {
            let j = f.acquire_frontier_target(&c).unwrap();
            f.observe(j, rng.next_f32() < 0.85);
            sink += j;
        }
        black_box(f.expand_certified(&c, 2.0));
    }
    let per_query = t0.elapsed().as_secs_f64() / (iters * CROWD) as f64;
    black_box(sink);

    // The exact Eq-10 posterior variance at the plan's stated buffer size.
    let mut buf = Box::new(PosteriorBuffer::<256, D>::new(1.0));
    for _ in 0..256 {
        buf.append_observation(&std::array::from_fn(|_| rng.next_f32()), rng.next_f32());
    }
    let mut scratch = [0.0f32; 256];
    let probe: [f32; D] = std::array::from_fn(|_| rng.next_f32());
    let t1 = Instant::now();
    let mut vsink = 0.0f32;
    for _ in 0..2000 {
        vsink += buf.posterior_variance_linear(&probe, &mut scratch);
    }
    let per_variance = t1.elapsed().as_secs_f64() / 2000.0;
    black_box(vsink);

    // One dilation pass over the full pool, for the record.
    let t2 = Instant::now();
    black_box(f.reachability_dilation(&c, 1));
    let per_dilation = t2.elapsed().as_secs_f64();

    println!(
        "\n=== T3.1 G2 perf (pool {POOL} cells, D={D}, {certified} certified at warmup) ===\n\
         acquire+observe+amortised expand : {:>8.3} us/query  (target < 1 us)\n\
         posterior_variance_linear N=256  : {:>8.3} us/call\n\
         reachability_dilation (1 hop)    : {:>8.3} us/pass",
        per_query * 1e6,
        per_variance * 1e6,
        per_dilation * 1e6,
    );

    if cfg!(debug_assertions) {
        println!("(debug build — perf assertions skipped)");
        return;
    }
    assert!(
        per_query < 1e-6,
        "G2 FAIL: {:.3} us/query exceeds the 1 us budget",
        per_query * 1e6
    );
    // O(n^2) in the buffer depth: 256^2 forward substitution. Budgeted
    // generously — this is the opt-in kernel path, not the default Beta path.
    assert!(
        per_variance < 100e-6,
        "posterior variance regressed: {:.3} us/call",
        per_variance * 1e6
    );
}

// ── T5.3 — the dual form's regime claim, measured in-tree ──────────────────
//
// Plan 580 T5.3 shipped `DualPosteriorBuffer` on the consumer's measurement
// (riir-train Bench 563, D = 32: 79.6x at n = 256, O(1) in n). That is the
// consumer's box, not ours, so the claim is re-measured here. The bars below
// are deliberately far looser than the quoted figures: what is being gated is
// the **scaling class**, which is a property of the algorithm, not the
// microsecond count, which is a property of the machine.
//
// Release-only, like the rest of this file — a debug build measures an
// unoptimised binary and has manufactured false perf reds in this repo before
// (`.docs/10_audits/cfg_gated_silent_zero_pass.md` T2b).
mod t53_dual_perf {
    use katgpt_core::certified_frontier::{DualPosteriorBuffer, PosteriorBuffer};
    use std::hint::black_box;
    use std::time::Instant;

    const DD: usize = 32;
    const PRIMAL_CAP: usize = 256;
    const QUERIES: usize = 2_000;

    fn feat(i: usize) -> [f32; DD] {
        let mut f = [0.0f32; DD];
        for (k, v) in f.iter_mut().enumerate() {
            // Deterministic, full-rank-ish, and not axis-aligned.
            *v = ((i * 31 + k * 17) % 97) as f32 / 97.0 - 0.5;
        }
        f
    }

    /// ns per `posterior_variance_linear` query, both arms warmed first (a cold
    /// arm has produced a 174x-wrong verdict in this workspace before).
    fn time_dual(n: usize) -> f64 {
        let mut p = DualPosteriorBuffer::<DD>::new(1e-2);
        for i in 0..n {
            p.append_observation(&feat(i), 1.0);
        }
        let probe = feat(7);
        for _ in 0..QUERIES / 4 {
            black_box(p.posterior_variance_linear(black_box(&probe)));
        }
        let t = Instant::now();
        for _ in 0..QUERIES {
            black_box(p.posterior_variance_linear(black_box(&probe)));
        }
        t.elapsed().as_secs_f64() * 1e9 / QUERIES as f64
    }

    fn time_primal(n: usize) -> f64 {
        assert!(n <= PRIMAL_CAP);
        let mut p = Box::new(PosteriorBuffer::<PRIMAL_CAP, DD>::new(1e-2));
        for i in 0..n {
            p.append_observation(&feat(i), 1.0);
        }
        let probe = feat(7);
        let mut scratch = vec![0.0f32; PRIMAL_CAP];
        for _ in 0..QUERIES / 4 {
            black_box(p.posterior_variance_linear(black_box(&probe), &mut scratch));
        }
        let t = Instant::now();
        for _ in 0..QUERIES {
            black_box(p.posterior_variance_linear(black_box(&probe), &mut scratch));
        }
        t.elapsed().as_secs_f64() * 1e9 / QUERIES as f64
    }

    /// **G2-dual.** Three claims, each two-sided against its own arm rather
    /// than against an absolute latency budget:
    ///
    /// 1. the dual is `O(1)` in `n` — 4096 observations cost the same as 16;
    /// 2. the primal is not — it grows superlinearly over the same span;
    /// 3. at the crossover the plan cites (`n = 256`, `D = 32`) the dual wins
    ///    by a wide margin.
    #[test]
    fn t53_g2_dual_is_o1_in_n_and_beats_the_primal_past_the_crossover() {
        let d16 = time_dual(16);
        let d256 = time_dual(256);
        let d4096 = time_dual(4096);
        let p16 = time_primal(16);
        let p64 = time_primal(64);
        let p256 = time_primal(256);

        println!("T5.3 G2 (D={DD}, ns/query)");
        println!("  n        dual      primal");
        println!("  16    {d16:8.1}   {p16:8.1}");
        println!("  64         —      {p64:8.1}");
        println!(
            "  256   {d256:8.1}   {p256:8.1}   speedup {:.1}x",
            p256 / d256
        );
        println!("  4096  {d4096:8.1}          —   (primal would be 64 MiB of state)");
        println!(
            "  scaling 16->4096: dual {:.3}x   |   primal 16->256: {:.1}x",
            d4096 / d16,
            p256 / p16
        );

        assert!(
            d4096 / d16 < 2.0,
            "dual is not O(1) in n: {d16:.1} -> {d4096:.1} ns ({:.2}x)",
            d4096 / d16
        );
        assert!(
            p256 / p16 > 4.0,
            "primal did not grow with n ({:.2}x) — the arms are not measuring what we think",
            p256 / p16
        );
        assert!(
            p256 / d256 > 5.0,
            "dual only {:.1}x faster at the crossover (n=256, D={DD})",
            p256 / d256
        );
    }
}

//! Bench 760 — signed-graph LIF reservoir GOAT G2 (Issue 763 T4).
//!
//! Three arms per cell:
//! 1. `event`     — `LifReservoir::step` (active set + spiker edge writes).
//! 2. `dense_csr` — `LifReservoir::step_dense` (CSR full node scan; same
//!    dynamics, no active-set skip — bit-identical by the G1 gate).
//! 3. `matvec`    — the classic dense-reservoir baseline: a dense N×N weight
//!    matrix, per-tick propagation `g_in += W · spike_indicator` (bench-local
//!    naive reference, the O(N²) form every textbook reservoir runs).
//!
//! Cells: N ∈ {1k, 10k} × regimes {sparse, saturated}.
//!
//! - **sparse**: single-fire periodic drive (w=14, slot-aligned period 1404)
//!   on a subthreshold-edge ER graph (K≈5, w=1) — the fly-connectome regime:
//!   no cascade, activity = driven bursts + decaying one-hop tails
//!   (measured active fraction printed per run).
//! - **saturated**: every node driven every period — the honesty line where
//!   the active-set skip buys nothing (RuVector measured 1.01× there; the
//!   event arm's remaining edge over `matvec` is then purely the
//!   sparse-vs-dense representation).
//!
//! Gate (the issue's T4 bar): `event ≥ 3× matvec` at N=10k/sparse.
//! Run: `cargo bench -p katgpt-core --features lif_graph --bench bench_760_lif_graph_goat`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::lif_graph::{LifParams, LifReservoir, SignedAdjacency};

/// Slot-aligned single-fire period (see tests/lif_graph_g4_alloc.rs for why
/// the period must be a multiple of the 18-bucket ring).
const PERIOD: usize = 1404;
const MAG: f32 = 14.0;

fn make_graph(n: u32, edges: usize, seed: u64) -> SignedAdjacency {
    // Subthreshold excitatory/inhibitory edges (w=1 PSP): no cascade.
    SignedAdjacency::er_matched(n, edges, 0.2, 1.0, 1.0, seed)
}

fn driven_set(n: u32, n_driven: u32) -> Vec<u32> {
    (0..n_driven).map(|i| (i * 37) % n).collect()
}

/// The classic dense-W·spike-matvec tick (bench-local naive baseline).
///
/// Same current-based LIF update math as the library (exact decays), but
/// propagation is `g_in += W · spike_indicator` over a dense N×N row-major
/// matrix — the O(N²) textbook form. One-tick propagation (no delay ring):
/// this arm measures REPRESENTATION cost, not dynamics parity (the G1 gate
/// owns parity between the library's own two paths).
struct DenseMatvecLif {
    w: Vec<f32>, // N×N row-major, W[j·N + i] = weight(i→j) in g units
    v: Vec<f32>,
    g: Vec<f32>,
    refrac: Vec<u32>,
    flags: Vec<f32>, // spike indicator, reused across ticks
    spikes: Vec<u32>,
    n: usize,
    params: LifParams,
}

impl DenseMatvecLif {
    fn new(adj: &SignedAdjacency, params: &LifParams) -> Self {
        let n = adj.n() as usize;
        let mut w = vec![0.0_f32; n * n];
        for row in 0..adj.n() {
            let (a, b) = adj.row_range(row);
            for e in a..b {
                w[adj.targets()[e] as usize * n + row as usize] =
                    adj.weights()[e] * params.w_scale();
            }
        }
        Self {
            w,
            v: vec![params.v0; n],
            g: vec![0.0; n],
            refrac: vec![0; n],
            flags: vec![0.0; n],
            spikes: Vec::with_capacity(n),
            n,
            params: params.clone(),
        }
    }

    /// One tick: dense propagation from last tick's spikes + drive + update.
    fn tick(&mut self, driven: &[u32], mag: f32) -> usize {
        let n = self.n;
        let p = &self.params;
        let (a_m, a_s, c_gs) = p.decays();
        // Spike indicator from last tick's spikes.
        for f in self.flags.iter_mut() {
            *f = 0.0;
        }
        for &s in self.spikes.iter() {
            self.flags[s as usize] = 1.0;
        }
        // W·f row dots, fused with the node update.
        let mut fired = Vec::with_capacity(n);
        for j in 0..n {
            let row = &self.w[j * n..(j + 1) * n];
            let flags = &self.flags;
            let mut acc = 0.0_f32;
            let mut i = 0;
            while i + 4 <= n {
                acc = row[i].mul_add(flags[i], acc);
                acc = row[i + 1].mul_add(flags[i + 1], acc);
                acc = row[i + 2].mul_add(flags[i + 2], acc);
                acc = row[i + 3].mul_add(flags[i + 3], acc);
                i += 4;
            }
            while i < n {
                acc = row[i].mul_add(flags[i], acc);
                i += 1;
            }
            self.g[j] += acc;
        }
        for &d in driven {
            self.g[d as usize] += mag * p.w_scale();
        }
        for j in 0..n {
            if self.refrac[j] > 0 {
                self.refrac[j] -= 1;
                self.v[j] = p.v_rst;
                let g_next = self.g[j] * a_s;
                self.g[j] = if g_next.abs() < p.g_floor {
                    0.0
                } else {
                    g_next
                };
                continue;
            }
            let v_new = p.v0 + (self.v[j] - p.v0) * a_m + self.g[j] * c_gs;
            let g_next = self.g[j] * a_s;
            self.g[j] = if g_next.abs() < p.g_floor {
                0.0
            } else {
                g_next
            };
            if v_new >= p.v_th {
                self.v[j] = p.v_rst;
                self.refrac[j] = p.refrac_ticks;
                fired.push(j as u32);
            } else {
                self.v[j] = v_new;
            }
        }
        self.spikes = fired;
        self.spikes.len()
    }
}

/// Measure + print the regime's steady-state activity profile (non-vacuity
/// context for the numbers). Returns (mean active fraction, spikes/tick).
fn profile(n: u32, edges: usize, n_driven: u32, seed: u64) -> (f64, f64) {
    let adj = make_graph(n, edges, seed);
    let mut r = LifReservoir::new(adj, LifParams::shiu());
    let driven = driven_set(n, n_driven);
    let mut active_sum = 0usize;
    let mut spike_sum = 0usize;
    let ticks = 3 * PERIOD;
    for t in 0..ticks {
        if t.is_multiple_of(PERIOD) {
            for &d in &driven {
                r.inject(d, MAG);
            }
        }
        let spikes = r.step().len();
        if t >= PERIOD {
            active_sum += r.active_count();
            spike_sum += spikes;
        }
    }
    let measured = (ticks - PERIOD) as f64;
    (
        active_sum as f64 / measured / n as f64,
        spike_sum as f64 / measured,
    )
}

fn bench_cell(c: &mut Criterion, name: &str, n: u32, edges: usize, n_driven: u32, seed: u64) {
    let driven = driven_set(n, n_driven);
    let (active_frac, spikes_per_tick) = profile(n, edges, n_driven, seed);
    println!(
        "{name}: N={n} E={edges} driven={n_driven}/period — steady-state active fraction {active_frac:.4}, spikes/tick {spikes_per_tick:.2}"
    );

    // Arm 1: event-driven (warmed onto the periodic orbit).
    {
        let adj = make_graph(n, edges, seed);
        let mut r = LifReservoir::new(adj, LifParams::shiu());
        for t in 0..2 * PERIOD {
            if t.is_multiple_of(PERIOD) {
                for &d in &driven {
                    r.inject(d, MAG);
                }
            }
            r.step();
        }
        let mut tick = 2 * PERIOD;
        c.bench_with_input(BenchmarkId::new(format!("{name}/event"), n), &n, |b, _| {
            b.iter(|| {
                if tick.is_multiple_of(PERIOD) {
                    for &d in &driven {
                        r.inject(d, MAG);
                    }
                }
                tick += 1;
                black_box(r.step().len())
            })
        });
    }

    // Arm 2: CSR full scan.
    {
        let adj = make_graph(n, edges, seed);
        let mut r = LifReservoir::new(adj, LifParams::shiu());
        for t in 0..2 * PERIOD {
            if t.is_multiple_of(PERIOD) {
                for &d in &driven {
                    r.inject(d, MAG);
                }
            }
            r.step_dense();
        }
        let mut tick = 2 * PERIOD;
        c.bench_with_input(
            BenchmarkId::new(format!("{name}/dense_csr"), n),
            &n,
            |b, _| {
                b.iter(|| {
                    if tick.is_multiple_of(PERIOD) {
                        for &d in &driven {
                            r.inject(d, MAG);
                        }
                    }
                    tick += 1;
                    black_box(r.step_dense().len())
                })
            },
        );
    }

    // Arm 3: dense-W matvec baseline. 10k → 400 MB matrix (workstation OK).
    {
        let adj = make_graph(n, edges, seed);
        let mut m = DenseMatvecLif::new(&adj, &LifParams::shiu());
        let mut tick = 0usize;
        c.bench_with_input(BenchmarkId::new(format!("{name}/matvec"), n), &n, |b, _| {
            b.iter(|| {
                let driven_now: &[u32] = if tick.is_multiple_of(PERIOD) {
                    &driven
                } else {
                    &[]
                };
                tick += 1;
                black_box(m.tick(driven_now, MAG))
            })
        });
    }
}

fn bench_lif_graph(c: &mut Criterion) {
    // Sparse regime (the fly-connectome shape), tuned to the issue's ~1–5%
    // active-fraction cell: each drive leaves ≈6×driven nodes active per
    // tail (~1000-tick tails / 1404-tick period), so driven ≈ N·0.02/6.
    // Measured fraction printed per run.
    bench_cell(c, "sparse", 1_000, 5_000, 5, 2026);
    bench_cell(c, "sparse", 10_000, 50_000, 50, 2027);
    // Saturated: every node driven every period.
    bench_cell(c, "saturated", 1_000, 5_000, 1_000, 2028);
    bench_cell(c, "saturated", 10_000, 50_000, 10_000, 2029);
}

criterion_group!(benches, bench_lif_graph);
criterion_main! { benches }

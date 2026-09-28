//! Proposal 011 Phase 5 — T5.5 `SweTrajectoryFreezer` GOAT gate.
//!
//! Exercises the full two-stage pipeline of
//! [`katgpt_core::swe_trajectory_freeze`]:
//!
//! 1. **Fit** — `derive_directions` from a labeled corpus of synthetic
//!    failure-mode trajectory summaries.
//! 2. **Freeze** — `SweTrajectoryFreezer::freeze_attempt` on held-out test
//!    trajectories.
//! 3. **Classify** — `FrozenAttempt::gates` + `argmax_archetype` for
//!    failure-mode identification.
//!
//! This bench is the substrate-level GOAT gate. It does NOT run the cross-
//! snapshot G5 gate (that's T5.6, which requires real-model trajectories —
//! T5.4 PARTIAL documented that depth trajectories alone are insufficient).
//! The substrate-level gate here only asserts the primitive works end-to-end
//! on the synthetic regime T5.3b already validated.
//!
//! # Run
//!
//! ```bash
//! cargo bench --manifest-path Cargo.toml \
//!     --features swe_trajectory_freeze \
//!     --bench bench_013_swe_trajectory_freezer_goat -- --nocapture
//! ```

#![cfg(feature = "swe_trajectory_freeze")]
#![allow(clippy::needless_range_loop)]

use katgpt_core::committed_field_blend::ArchetypeFieldSource;
use katgpt_core::latent_trajectory_geometry::from_states;
use katgpt_core::swe_trajectory_freeze::{
    GeometrySummaryEncoder, SweTrajectoryFreezer, derive_directions,
};
use std::time::Instant;

// ─── Constants ─────────────────────────────────────────────────────────────

const DIM: usize = 8;
const N_STEPS: usize = 100;
const D: usize = 32;
const N: usize = 3;

const TRAJ_PER_MODE: usize = 5;
const TRAIN_SEEDS: usize = 3;

const MODE_NAMES: [&str; N] = ["oscillation", "committed_wrong", "converged_correct"];

// ─── Deterministic LCG (matches bench_011) ─────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

// ─── Synthetic trajectory builders (mirror bench_011 exactly) ──────────────

fn build_committed_wrong(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let mut state: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 0.1).collect();
    let mut direction: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let norm = direction
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    for x in direction.iter_mut() {
        *x /= norm;
    }
    let step_size = 0.15;
    let mut traj = Vec::with_capacity(N_STEPS + 1);
    traj.push(state.clone());
    for _ in 0..N_STEPS {
        for j in 0..DIM {
            state[j] += step_size * direction[j];
        }
        traj.push(state.clone());
    }
    traj.shrink_to_fit();
    traj
}

fn build_oscillation(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let attractor_a: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let attractor_b: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let mut traj = Vec::with_capacity(N_STEPS + 1);
    for i in 0..=N_STEPS {
        let target = if i % 2 == 0 {
            &attractor_a
        } else {
            &attractor_b
        };
        traj.push(target.clone());
    }
    traj.shrink_to_fit();
    traj
}

fn build_converged_correct(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let target: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let mut state: Vec<f32> = (0..DIM).map(|_| rng.next_f32() * 0.1).collect();
    let mut traj = Vec::with_capacity(N_STEPS + 1);
    traj.push(state.clone());
    for _ in 0..N_STEPS {
        for j in 0..DIM {
            state[j] += 0.05 * (target[j] - state[j]);
        }
        traj.push(state.clone());
    }
    traj.shrink_to_fit();
    traj
}

fn build_trajectory_for_mode(mode_idx: usize, seed: u64) -> Vec<Vec<f32>> {
    match mode_idx {
        0 => build_oscillation(seed),
        1 => build_committed_wrong(seed),
        2 => build_converged_correct(seed),
        _ => unreachable!(),
    }
}

fn build_refs(traj: &[Vec<f32>]) -> Vec<&[f32]> {
    traj.iter().map(|v| v.as_slice()).collect()
}

fn encode_summary(traj: &[Vec<f32>]) -> [f32; D] {
    let refs = build_refs(traj);
    let geom = from_states(&refs);
    let mut summary = [0.0_f32; D];
    GeometrySummaryEncoder::default_synthetic().encode_into(&geom, &mut summary);
    summary
}

// ─── Stub archetype field (for FAME commit, which only reads commitment()) ──

struct StubField {
    commitment: [u8; 32],
}

impl StubField {
    const fn new(commitment: [u8; 32]) -> Self {
        Self { commitment }
    }
}

impl ArchetypeFieldSource<D> for StubField {
    fn evolve<'a>(&self, _z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        for x in dz_scratch.iter_mut().take(D) {
            *x = 0.0;
        }
        &mut dz_scratch[..D]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

fn make_fields() -> [&'static dyn ArchetypeFieldSource<D>; N] {
    // Pre-compute commitments at startup (BLAKE3 of the index).
    use std::sync::OnceLock;
    static F0: OnceLock<StubField> = OnceLock::new();
    static F1: OnceLock<StubField> = OnceLock::new();
    static F2: OnceLock<StubField> = OnceLock::new();
    let f0 = F0.get_or_init(|| {
        let mut h = blake3::Hasher::new();
        h.update(b"bench_013_stub_field:");
        h.update(&[0u8]);
        StubField::new(h.finalize().into())
    });
    let f1 = F1.get_or_init(|| {
        let mut h = blake3::Hasher::new();
        h.update(b"bench_013_stub_field:");
        h.update(&[1u8]);
        StubField::new(h.finalize().into())
    });
    let f2 = F2.get_or_init(|| {
        let mut h = blake3::Hasher::new();
        h.update(b"bench_013_stub_field:");
        h.update(&[2u8]);
        StubField::new(h.finalize().into())
    });
    [f0 as &'static dyn ArchetypeFieldSource<D>, f1, f2]
}

// ─── GOAT gates ────────────────────────────────────────────────────────────

struct ProbeResult {
    mode_name: &'static str,
    matching_gate: f32,
    argmax_k: usize,
    correct: bool,
}

fn run_g3_cross_mode_discrimination(directions: &[[f32; D]; N]) -> (Vec<ProbeResult>, f32, bool) {
    let freezer = SweTrajectoryFreezer::<N, D>::new(*directions);
    let fields: [&dyn ArchetypeFieldSource<D>; N] = make_fields();

    let mut all_trajs: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(N);
    for mode_idx in 0..N {
        let mut mode_trajs = Vec::with_capacity(TRAJ_PER_MODE);
        for seed in 0..TRAJ_PER_MODE {
            mode_trajs.push(build_trajectory_for_mode(mode_idx, seed as u64 * 100 + 7));
        }
        all_trajs.push(mode_trajs);
    }

    let n_test = TRAJ_PER_MODE - TRAIN_SEEDS;
    let mut probes: Vec<ProbeResult> = Vec::with_capacity(N * n_test);
    let mut n_correct = 0usize;
    let total = N * n_test;

    for mode_idx in 0..N {
        for seed in TRAIN_SEEDS..TRAJ_PER_MODE {
            let refs = build_refs(&all_trajs[mode_idx][seed]);
            let frozen = freezer.freeze_attempt(&refs, &fields, 1);
            let gates = frozen.gates();
            let matching_gate = gates[mode_idx];
            let argmax_k = frozen.argmax_archetype();
            let correct = matching_gate > 0.6 && argmax_k == mode_idx;
            if correct {
                n_correct += 1;
            }
            probes.push(ProbeResult {
                mode_name: MODE_NAMES[mode_idx],
                matching_gate,
                argmax_k,
                correct,
            });
        }
    }

    let accuracy = n_correct as f32 / total as f32;
    let pass = accuracy >= 0.8;
    (probes, accuracy, pass)
}

fn run_g2_perf(directions: &[[f32; D]; N]) -> (u64, bool) {
    // Measure freeze_attempt latency per call (steady state).
    let freezer = SweTrajectoryFreezer::<N, D>::new(*directions);
    let fields: [&dyn ArchetypeFieldSource<D>; N] = make_fields();
    let traj = build_trajectory_for_mode(0, 999);
    let refs = build_refs(&traj);

    // Warmup.
    for _ in 0..100 {
        let _ = freezer.freeze_attempt(&refs, &fields, 1);
    }

    // Measure: 1000 freezes.
    const N_ITERS: usize = 1000;
    let start = Instant::now();
    for _ in 0..N_ITERS {
        let _ = freezer.freeze_attempt(&refs, &fields, 1);
    }
    let total_ns = start.elapsed().as_nanos() as u64;
    let per_call_ns = total_ns / (N_ITERS as u64);

    // Target: < 5 µs/call (same budget as latent_trajectory_geometry's G2;
    // the substrate's hot path is from_states + FAME commit + BLAKE3 envelope).
    let target_ns: u64 = 5_000;
    let pass = per_call_ns < target_ns;
    (per_call_ns, pass)
}

fn run_g4_zero_alloc(directions: &[[f32; D]; N]) -> (G4Result, G4Result) {
    // CountingAllocator audit of the steady-state hot path.
    //
    // Two variants measured:
    //   (1) `freeze_attempt`       — convenience wrapper, allocates 2 Vec scratch.
    //   (2) `freeze_attempt_into`  — zero-alloc when caller reuses scratch.
    //
    // The freeze pipeline itself (encode_into + FAME commit + envelope) is
    // zero-alloc in both variants; the only difference is the scratch-buffer
    // allocation in `from_states` vs `from_states_into`.
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingAllocator;
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            // SAFETY: layout is valid (caller's contract); System.alloc is sound.
            unsafe { std::alloc::System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: ptr was allocated by System.alloc with this layout.
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }
    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[global_allocator]
    static A: CountingAllocator = CountingAllocator;

    const N_CALLS: usize = 100;

    let freezer = SweTrajectoryFreezer::<N, D>::new(*directions);
    let fields: [&dyn ArchetypeFieldSource<D>; N] = make_fields();
    let traj = build_trajectory_for_mode(0, 999);
    let refs = build_refs(&traj);

    // Variant 1: freeze_attempt (allocating) — target ≤2 allocs/call.
    let before1 = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..N_CALLS {
        let _ = freezer.freeze_attempt(&refs, &fields, 1);
    }
    let after1 = ALLOC_COUNT.load(Ordering::Relaxed);
    let total1 = after1 - before1;
    let per_call1 = total1 / N_CALLS;
    let pass1 = per_call1 <= 2;

    // Variant 2: freeze_attempt_into (zero-alloc steady state) — target 0.
    // Pre-allocate scratch once; reuse across all calls.
    let mut disp_curr = Vec::<f32>::with_capacity(DIM);
    let mut disp_prev = Vec::<f32>::with_capacity(DIM);
    // Warmup the scratch buffers to their steady-state capacity (first call
    // resizes from 0 to DIM; subsequent calls reuse).
    let _ = freezer.freeze_attempt_into(&refs, &fields, 1, &mut disp_curr, &mut disp_prev);

    let before2 = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..N_CALLS {
        let _ = freezer.freeze_attempt_into(&refs, &fields, 1, &mut disp_curr, &mut disp_prev);
    }
    let after2 = ALLOC_COUNT.load(Ordering::Relaxed);
    let total2 = after2 - before2;
    let per_call2 = total2 / N_CALLS;
    let pass2 = per_call2 == 0;

    (
        G4Result {
            per_call: per_call1,
            pass: pass1,
            variant: "freeze_attempt",
        },
        G4Result {
            per_call: per_call2,
            pass: pass2,
            variant: "freeze_attempt_into",
        },
    )
}

struct G4Result {
    per_call: usize,
    pass: bool,
    variant: &'static str,
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  Proposal 011 Phase 5 — T5.5 SweTrajectoryFreezer GOAT Gate       ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Config: DIM={DIM}, N_STEPS={N_STEPS}, N archetypes={N}, D summary={D}");
    println!(
        "        {TRAJ_PER_MODE} trajs/mode, {TRAIN_SEEDS} train + {} test",
        TRAJ_PER_MODE - TRAIN_SEEDS
    );
    println!();

    // Stage 1: fit — derive directions from labeled training summaries.
    let mut all_trajs: Vec<Vec<Vec<Vec<f32>>>> = Vec::with_capacity(N);
    for mode_idx in 0..N {
        let mut mode_trajs = Vec::with_capacity(TRAJ_PER_MODE);
        for seed in 0..TRAJ_PER_MODE {
            mode_trajs.push(build_trajectory_for_mode(mode_idx, seed as u64 * 100 + 7));
        }
        all_trajs.push(mode_trajs);
    }

    let mut train: [[[f32; D]; TRAIN_SEEDS]; N] = [[[0.0; D]; TRAIN_SEEDS]; N];
    for mode_idx in 0..N {
        for seed in 0..TRAIN_SEEDS {
            train[mode_idx][seed] = encode_summary(&all_trajs[mode_idx][seed]);
        }
    }

    let mut directions = [[0.0_f32; D]; N];
    derive_directions(&train, &mut directions);

    println!("── Stage 1 (fit): derive_directions ──");
    for k in 0..N {
        let norm_sq: f32 = directions[k].iter().map(|x| x * x).sum();
        println!(
            "  direction[{k}] ({:>18}): norm = {:.4}",
            MODE_NAMES[k],
            norm_sq.sqrt()
        );
    }
    println!();

    // ── G1: directions are non-degenerate ────────────────────────────────
    println!("── G1: directions non-degenerate ──");
    let mut g1_pass = true;
    for k in 0..N {
        let norm_sq: f32 = directions[k].iter().map(|x| x * x).sum();
        if (norm_sq - 1.0).abs() > 1e-4 {
            println!("   FAIL: direction {k} norm {} != 1.0", norm_sq.sqrt());
            g1_pass = false;
        }
    }
    for i in 0..N {
        for j in (i + 1)..N {
            let dot: f32 = (0..D).map(|k| directions[i][k] * directions[j][k]).sum();
            if dot > 0.99 {
                println!("   FAIL: directions {i} and {j} near-identical (cos={dot:.4})");
                g1_pass = false;
            }
        }
    }
    println!("   G1 verdict: {}", if g1_pass { "PASS" } else { "FAIL" });
    println!();

    // ── G3: cross-mode discrimination ────────────────────────────────────
    println!("── G3: cross-mode discrimination (substrate-level gate) ──");
    let (probes, accuracy, g3_pass) = run_g3_cross_mode_discrimination(&directions);
    println!(
        "  {:>18}  {:>10}  {:>14}  {:>10}",
        "mode", "argmax_k", "matching_gate", "correct"
    );
    println!("  {}", "-".repeat(58));
    for p in &probes {
        println!(
            "  {:>18}  {:>10}  {:>14.4}  {:>10}",
            p.mode_name, p.argmax_k, p.matching_gate, p.correct
        );
    }
    println!();
    println!("   accuracy: {accuracy:.2} (target ≥0.80)");
    println!("   G3 verdict: {}", if g3_pass { "PASS" } else { "FAIL" });
    println!();

    // ── G2: perf ─────────────────────────────────────────────────────────
    println!("── G2: freeze_attempt latency ──");
    let (per_call_ns, g2_pass) = run_g2_perf(&directions);
    println!("   per_call: {per_call_ns} ns (target < 5000 ns)");
    println!("   G2 verdict: {}", if g2_pass { "PASS" } else { "FAIL" });
    println!();

    // ── G4: zero-alloc steady state ─────────────────────────────────
    println!("── G4: alloc-free steady state ──");
    let (g4_alloc, g4_into) = run_g4_zero_alloc(&directions);
    println!(
        "   {}: {} allocs/call (target ≤2; from_states substrate)",
        g4_alloc.variant, g4_alloc.per_call
    );
    println!(
        "   {}: {} allocs/call (target 0; from_states_into + reused scratch)",
        g4_into.variant, g4_into.per_call
    );
    let g4_pass = g4_alloc.pass && g4_into.pass;
    println!("   G4 verdict: {}", if g4_pass { "PASS" } else { "FAIL" });
    println!();

    // ── Summary ─────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("T5.5 SweTrajectoryFreezer substrate-level gate:");
    println!(
        "  G1 directions non-degenerate : {}",
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G2 freeze_attempt latency    : {}",
        if g2_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G3 cross-mode discrimination : {}",
        if g3_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G4 alloc-free steady state   : {}",
        if g4_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!();
    let all_pass = g1_pass && g2_pass && g3_pass && g4_pass;
    if all_pass {
        println!("ALL GATES PASS — T5.5 substrate validated on synthetic regime.");
        println!();
        println!("This is the SUBSTRATE-level gate. T5.6 G5 (cross-snapshot/model");
        println!("discrimination) is the open question — it requires real-model");
        println!("trajectories. T5.4 PARTIAL documented that depth trajectories");
        println!("alone are insufficient (G3 FAIL at 29%); the discriminative");
        println!("signal likely lives in iterative refinement trajectories, not");
        println!("depth trajectories of a single forward pass.");
    } else {
        println!("ONE OR MORE GATES FAILED — see details above.");
    }
    println!();
}

//! G1 gate for `mb_value` (Issue 767 T5): bounds-by-construction, canonical
//! top-k parity, determinism, calibration sanity, and the toy-corridor value
//! gate vs the **ridge-batch floor on the same code** (the issue's mandatory
//! baseline) plus the distribution-shift arm where online adapts and a batch
//! fit cannot.
//!
//! Run: `cargo test -p katgpt-core --features mb_value,linalg --test mb_value_g1 -- --nocapture`

use katgpt_core::linalg::ridge_solve_direct_f64;
use katgpt_core::mb_value::{MbCircuit, MbCircuitConfig, MbScratch};

// ── stats helpers (test-local; no new deps) ─────────────────────────────────

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f64;
    let ma: f64 = a.iter().sum::<f32>() as f64 / n;
    let mb: f64 = b.iter().sum::<f32>() as f64 / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        let da = a[i] as f64 - ma;
        let db = b[i] as f64 - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    (cov / (va * vb).sqrt().max(1e-12)) as f32
}

/// Spearman = Pearson on ranks (average ranks not needed: values distinct
/// with probability 1 on these fixtures).
fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let rank = |v: &[f32]| -> Vec<f32> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        let mut r = vec![0.0f32; v.len()];
        for (pos, &i) in idx.iter().enumerate() {
            r[i] = pos as f32;
        }
        r
    };
    pearson(&rank(a), &rank(b))
}

/// Deterministic test RNG (the house xorshift64* fixture pattern).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 1 } else { seed })
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f32 {
        let bits = ((self.next() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }
    fn normal(&mut self) -> f32 {
        self.uniform().max(1e-12) * 12.0 - 6.0 // clipped approx-N(0,1); fixtures only
    }
}

// ── 1. construction determinism ─────────────────────────────────────────────

#[test]
fn construction_is_seed_deterministic() {
    let cfg = MbCircuitConfig::toy();
    let a = MbCircuit::new(&cfg);
    let b = MbCircuit::new(&cfg);
    assert_eq!(a.w0(), b.w0());
    assert_eq!(a.w(), b.w());
    for kc in 0..a.n_kc() {
        assert_eq!(a.kc_mbon_targets(kc), b.kc_mbon_targets(kc));
    }
    // Different seed → different wiring (with overwhelming probability):
    // the initial w0 is all-1.0 by design (calibration rescales per-MBON),
    // so the seed difference shows in the KC→MBON targets.
    let mut cfg2 = cfg.clone();
    cfg2.seed = 99;
    let c = MbCircuit::new(&cfg2);
    let wiring_differs = (0..a.n_kc()).any(|kc| a.kc_mbon_targets(kc) != c.kc_mbon_targets(kc));
    assert!(
        wiring_differs,
        "different seeds must produce different wiring"
    );
}

// ── 2. canonical top-k parity (fast path == full-sort reference) ────────────

#[test]
fn fast_top_k_matches_full_sort_reference() {
    let cfg = MbCircuitConfig::toy();
    let mut circuit = MbCircuit::new(&cfg);
    // Calibrate on random data so thresholds are non-trivial.
    let mut rng = Rng::new(7);
    let n = 400;
    let mut feats = vec![0.0f32; n * cfg.n_features];
    for f in feats.iter_mut() {
        *f = rng.normal();
    }
    let actions = vec![1.0f32, 0.0, 0.0, 1.0]; // 2 × 2 one-hot
    circuit.calibrate(&feats, &actions, n, 0.5);

    let mut scratch = MbScratch::new(&circuit);
    let mut fast = vec![0u32; circuit.kc_active()];
    let mut reference = vec![0u32; circuit.kc_active()];
    for i in 0..200 {
        let mut f = vec![0.0f32; cfg.n_features];
        for x in f.iter_mut() {
            *x = rng.normal();
        }
        let action = if i % 2 == 0 {
            &actions[0..2]
        } else {
            &actions[2..4]
        };
        circuit.code_into(&f, action, &mut scratch, &mut fast);
        circuit.code_reference_into(&f, action, &mut scratch, &mut reference);
        assert_eq!(fast, reference, "fast top-k diverged from reference at {i}");
        // Canonical order: drives descending, index ascending — the reference
        // is sorted; verify sortedness of the fast path by re-deriving drives
        // is internal, so ordering parity is pinned via the reference output
        // equality above.
    }
}

#[test]
fn fast_top_k_matches_reference_on_all_ties() {
    // Degenerate circuit forcing EQUAL drives: n_pn = 1 → every KC reads the
    // same PN with the same normalized weight → all drives identical; the
    // canonical set is then the LOWEST indices (tie broken by index).
    let cfg = MbCircuitConfig {
        n_features: 2,
        n_action_dims: 0,
        n_pn: 1,
        pn_fanin: 1,
        n_kc: 64,
        kc_fanin: 1,
        kc_active: 8,
        n_mbon: 4,
        kc_mbon_fanin: 2,
        n_approach: 2,
        pn_action_fraction: 0.0,
        pn_quantile: 0.5,
        gain: 40.0,
        alpha: 0.02,
        seed: 3,
    };
    let circuit = MbCircuit::new(&cfg);
    let mut scratch = MbScratch::new(&circuit);
    let mut fast = vec![0u32; 8];
    let mut reference = vec![0u32; 8];
    let f = [0.7, -0.3];
    circuit.code_into(&f, &[], &mut scratch, &mut fast);
    circuit.code_reference_into(&f, &[], &mut scratch, &mut reference);
    assert_eq!(fast, reference);
    assert_eq!(
        fast,
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        "ties must break by index"
    );
}

// ── 3. bounds by construction, adversarial RPE streams ──────────────────────

#[test]
fn weights_stay_bounded_under_adversarial_rpe() {
    let cfg = MbCircuitConfig::toy();
    let mut circuit = MbCircuit::new(&cfg);
    let mut rng = Rng::new(11);
    let n = 400;
    let mut feats = vec![0.0f32; n * cfg.n_features];
    for f in feats.iter_mut() {
        *f = rng.normal();
    }
    let actions = vec![1.0f32, 0.0, 0.0, 1.0];
    circuit.calibrate(&feats, &actions, n, 0.5);
    assert!(
        circuit.eta() > 0.0,
        "calibration must derive a positive eta"
    );

    let mut scratch = MbScratch::new(&circuit);
    let mut code = vec![0u32; circuit.kc_active()];
    let adversarial = [
        0.0f32,
        1.0,
        -1.0,
        1e30,
        -1e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        5.0,
        -5.0,
        1e38,
        -1e-38,
    ];
    for (i, &rpe) in adversarial.iter().enumerate().cycle().take(10_000) {
        let mut f = vec![0.0f32; cfg.n_features];
        for x in f.iter_mut() {
            *x = rng.normal();
        }
        circuit.code_into(
            &f,
            &actions[(i % 2) * 2..(i % 2) * 2 + 2],
            &mut scratch,
            &mut code,
        );
        let before = circuit.w().to_vec();
        circuit.dopamine_update(&code, rpe);
        if !rpe.is_finite() {
            assert_eq!(
                before,
                circuit.w().to_vec(),
                "non-finite RPE must be a no-op"
            );
        }
        for k in 0..circuit.w().len() {
            assert!(
                circuit.w()[k] >= 0.0 && circuit.w()[k] <= circuit.w0()[k],
                "bound violated at synapse {k} under rpe {rpe}"
            );
        }
    }
}

// ── 4. saturation + recovery dynamics ───────────────────────────────────────

#[test]
fn depression_then_recovery() {
    let cfg = MbCircuitConfig::toy();
    let mut circuit = MbCircuit::new(&cfg);
    let mut rng = Rng::new(13);
    let n = 400;
    let mut feats = vec![0.0f32; n * cfg.n_features];
    for f in feats.iter_mut() {
        *f = rng.normal();
    }
    let actions = vec![1.0f32, 0.0, 0.0, 1.0];
    circuit.calibrate(&feats, &actions, n, 0.5);

    let mut scratch = MbScratch::new(&circuit);
    let mut code = vec![0u32; circuit.kc_active()];
    let f = [0.3, -0.1, 0.5, 0.2];
    circuit.code_into(&f, &actions[0..2], &mut scratch, &mut code);

    // Toy-circuit constants (kc_mbon_fanin=6, n_approach=8) in the closure below.
    let mean_code_side = |c: &MbCircuit, want_avoid: bool| -> f32 {
        let mut sum = 0.0f32;
        let mut n = 0u32;
        for &kc in &code {
            let base = kc as usize * 6;
            for e in 0..6 {
                let m = c.kc_mbon_targets(kc as usize)[e] as usize;
                let is_avoid = m >= 8; // toy: n_approach = 8
                if is_avoid == want_avoid {
                    sum += c.w()[base + e];
                    n += 1;
                }
            }
        }
        sum / n.max(1) as f32
    };
    let avoid_start = mean_code_side(&circuit, true);

    // Positive RPE depresses the avoid-compartment synapses of the code
    // (and lifts the approach side toward w0). 10k updates at toy eta (alpha
    // 0.01 → eta ≈ 3.3e-4) give cumulative Δ ≈ 3.3 > w0 ≈ 2.7 — saturation.
    for _ in 0..10_000 {
        circuit.dopamine_update(&code, 1.0);
    }
    let (floor, ceiling) = circuit.saturation();
    let avoid_down = mean_code_side(&circuit, true);
    println!(
        "after +RPE: floor={floor:.4} ceiling={ceiling:.4} avoid-side mean {avoid_start:.4} -> {avoid_down:.4}"
    );
    assert!(
        floor > 0.02,
        "expected measurable depression, floor={floor}"
    );
    assert!(
        ceiling > 0.02,
        "expected measurable approach recovery, ceiling={ceiling}"
    );
    assert!(avoid_down < 0.2 * avoid_start, "avoid side must depress");

    // Below-baseline dopamine (negative RPE) lets the avoid side recover
    // toward w0 (while the approach side depresses — the mirror image).
    for _ in 0..50_000 {
        circuit.dopamine_update(&code, -1.0);
    }
    let avoid_back = mean_code_side(&circuit, true);
    println!("after -RPE: avoid-side mean -> {avoid_back:.4}");
    assert!(
        avoid_back > avoid_down + 0.5 * (avoid_start - avoid_down),
        "avoid-side recovery expected: {avoid_back} from {avoid_down} vs {avoid_start}"
    );
}

// ── 5. calibration sanity ───────────────────────────────────────────────────

#[test]
fn calibration_report_is_sane_and_value_starts_at_zero() {
    let cfg = MbCircuitConfig::toy();
    let mut circuit = MbCircuit::new(&cfg);
    let mut rng = Rng::new(17);
    let n = 400;
    let mut feats = vec![0.0f32; n * cfg.n_features];
    for f in feats.iter_mut() {
        *f = rng.normal();
    }
    let actions = vec![1.0f32, 0.0, 0.0, 1.0];
    let report = circuit.calibrate(&feats, &actions, n, 0.5);
    println!("calibration: {report:?}");
    assert!(report.eta > 0.0);
    assert!(
        report.action_overlap <= 0.62,
        "bisection should land near 0.5"
    );
    assert!(
        report.mbons_connected > 0 && report.approach_mbons_used > 0 && report.avoid_mbons_used > 0,
        "both compartments must be connected: {report:?}"
    );
    // V = 0 at calibration IN THE MEAN over the CALIBRATION distribution
    // (random action codes — the drive the w0 normalization measured; a
    // fixed-action probe is a biased sample of that distribution).
    let mut scratch = MbScratch::new(&circuit);
    let mut code = vec![0u32; circuit.kc_active()];
    let mut rng2 = Rng::new(59);
    let mut sum = 0.0f32;
    let mut worst = 0.0f32;
    for s in 0..64 {
        let a = if rng2.next() & 1 == 0 {
            &actions[0..2]
        } else {
            &actions[2..4]
        };
        circuit.code_into(
            &feats[s * cfg.n_features..(s + 1) * cfg.n_features],
            a,
            &mut scratch,
            &mut code,
        );
        let v = circuit.value(&code, &mut scratch);
        sum += v;
        worst = worst.max(v.abs());
    }
    let mean = sum / 64.0;
    println!("V at start: mean {mean:.4} worst {worst:.4}");
    assert!(mean.abs() < 0.5, "mean V must start ≈0, got {mean}");
}

// ── 6. the toy corridor: value formation vs the ridge floor ────────────────

/// 1-D corridor: state s ∈ [0,1), actions move ±`step` (clamped), reward is
/// progress-shaped (`SCALE·Δs`, the source's reward family), γ = 0.98.
struct Corridor {
    step: f32,
    scale: f32,
    gamma: f32,
    sign: f32, // +1: right pays; −1: left pays (the shift arm flips this)
}

impl Corridor {
    fn features(s: f32) -> [f32; 4] {
        [s, s * s, (std::f32::consts::TAU * 0.7 * s).sin(), 0.5]
    }
    fn step_state(&self, s: f32, action: usize) -> (f32, f32) {
        let dir = if action == 0 { 1.0 } else { -1.0 };
        let s2 = (s + dir * self.step).clamp(0.0, 0.999);
        // Progress-shaped reward: `sign` flips the polarity (the shift arm);
        // direction enters through Δs itself (multiplying by `dir` again
        // would double-count and make every step pay the same sign).
        (s2, self.sign * self.scale * (s2 - s))
    }
    /// Exact value of the always-optimal-direction policy from `s`.
    fn true_value(&self, s: f32) -> f32 {
        let mut v = 0.0f32;
        let mut disc = 1.0f32;
        let mut cur = s;
        for _ in 0..200 {
            let (s2, r) = self.step_state(cur, if self.sign > 0.0 { 0 } else { 1 });
            v += disc * r;
            disc *= self.gamma;
            cur = s2;
            if cur >= 0.998 || cur <= 0.0 {
                break;
            }
        }
        v
    }
}

struct CorridorHarness {
    circuit: MbCircuit,
    scratch: MbScratch,
    code: Vec<u32>,
    code_next: Vec<u32>,
}

impl CorridorHarness {
    fn new(circuit: MbCircuit) -> Self {
        let scratch = MbScratch::new(&circuit);
        let code = vec![0u32; circuit.kc_active()];
        let code_next = vec![0u32; circuit.kc_active()];
        Self {
            circuit,
            scratch,
            code,
            code_next,
        }
    }

    /// State value (state-only circuit: no action code).
    fn v(&mut self, s: f32) -> f32 {
        self.circuit.code_into(
            &Corridor::features(s),
            &[],
            &mut self.scratch,
            &mut self.code,
        );
        self.circuit.value(&self.code, &mut self.scratch)
    }

    /// Run episodes under the deterministic follow-the-paying-direction
    /// policy; TD(0) state-value updates (the source's drive, state-only —
    /// the (s,a) shared-code interference is the source's own documented
    /// negative and is deliberately not this gate's axis).
    fn run_episodes(
        &mut self,
        corridor: &Corridor,
        rng: &mut Rng,
        episodes: usize,
        mut collect: Option<&mut Vec<(Vec<f32>, f32)>>, // (code-as-floats, realized return)
    ) {
        let mut rewards: Vec<f32> = Vec::with_capacity(64);
        for _ in 0..episodes {
            let mut s = rng.uniform();
            rewards.clear();
            let mut codes: Vec<Vec<f32>> = Vec::with_capacity(64);
            let a = if corridor.sign > 0.0 { 0 } else { 1 };
            for _ in 0..40 {
                self.circuit.code_into(
                    &Corridor::features(s),
                    &[],
                    &mut self.scratch,
                    &mut self.code,
                );
                let v_sa = self.circuit.value(&self.code, &mut self.scratch);
                if collect.is_some() {
                    let mut cf = vec![0.0f32; self.circuit.n_kc()];
                    for &kc in &self.code {
                        cf[kc as usize] = 1.0;
                    }
                    codes.push(cf);
                }
                let (s2, r) = corridor.step_state(s, a);
                rewards.push(r);
                // TD(0): the successor value goes through `code_next` —
                // `self.code` keeps the DECISION code for credit assignment.
                self.circuit.code_into(
                    &Corridor::features(s2),
                    &[],
                    &mut self.scratch,
                    &mut self.code_next,
                );
                let v_next = self.circuit.value(&self.code_next, &mut self.scratch);
                let rpe = r + corridor.gamma * v_next - v_sa;
                self.circuit.dopamine_update(&self.code, rpe);
                s = s2;
            }
            if let Some(out) = collect.as_deref_mut() {
                // Realized discounted return per decision, walking backward
                // from the episode end: at rev step t, g is exactly
                // Σ_{k≥t} γ^{k−t} r_k — the return from decision t.
                let mut g = 0.0f32;
                for t in (0..rewards.len()).rev() {
                    g = rewards[t] + corridor.gamma * g;
                    out.push((codes[t].clone(), g));
                }
            }
        }
    }
}

fn fit_ridge_readout(samples: &[(Vec<f32>, f32)], n_kc: usize, lambda: f64) -> Vec<f64> {
    let d = n_kc;
    let n_out = 1usize;
    let mut gram = vec![0.0f64; d * d];
    let mut cov = vec![0.0f64; d];
    let mut actives: Vec<usize> = Vec::with_capacity(64);
    for (code, g) in samples {
        actives.clear();
        for (i, &c) in code.iter().enumerate() {
            if c != 0.0 {
                actives.push(i);
                cov[i] += c as f64 * *g as f64;
            }
        }
        for &i in &actives {
            let row = &mut gram[i * d..(i + 1) * d];
            for &j in &actives {
                row[j] += 1.0;
            }
        }
    }
    for i in 0..d {
        gram[i * d + i] += lambda;
    }
    let mut w_t = vec![0.0f64; d * n_out];
    let mut l = vec![0.0f64; d * d];
    let mut z = vec![0.0f64; d * n_out];
    ridge_solve_direct_f64(&mut w_t, &mut l, &mut z, &gram, &cov, d, n_out);
    w_t
}

#[test]
fn toy_corridor_value_formation_vs_ridge_floor() {
    // State-only circuit: the (s,a) shared-code interference is the source's
    // own measured negative ("the value differs little between actions") —
    // this gate pins the primitive's STATED purpose, value formation feeding
    // EXTERNAL selection, on the state-value axis.
    let mut cfg = MbCircuitConfig::toy();
    cfg.n_action_dims = 0;
    let mut circuit = MbCircuit::new(&cfg);
    let mut rng = Rng::new(23);
    // Calibration batch: uniform corridor states (no action code).
    let n = 400;
    let mut feats = vec![0.0f32; n * cfg.n_features];
    for s in 0..n {
        let st = (s as f32 + 0.5) / n as f32;
        let f = Corridor::features(st);
        feats[s * cfg.n_features..(s + 1) * cfg.n_features].copy_from_slice(&f);
    }
    let report = circuit.calibrate(&feats, &[], n, 0.5);
    println!("corridor calibration: {report:?}");

    // ── Phase 1: right pays. Reward scale 4, γ 0.98. V* ∈ [0.4, ~3.3]
    // inside the representable window.
    let corridor = Corridor {
        step: 0.05,
        scale: 4.0,
        gamma: 0.98,
        sign: 1.0,
    };
    let mut harness = CorridorHarness::new(circuit);
    let mut samples: Vec<(Vec<f32>, f32)> = Vec::new();
    harness.run_episodes(&corridor, &mut rng, 2_000, Some(&mut samples));
    // Learning curve: correlation after each 1000-episode block.
    let eval: Vec<f32> = (0..64).map(|i| (i as f32 + 0.5) / 64.0).collect();
    let v_star: Vec<f32> = eval.iter().map(|&s| corridor.true_value(s)).collect();
    {
        let v: Vec<f32> = eval.iter().map(|&s| harness.v(s)).collect();
        println!("curve@2000: r={:.4}", pearson(&v, &v_star));
    }
    for _ in 0..6 {
        harness.run_episodes(&corridor, &mut rng, 1_000, None);
        let v: Vec<f32> = eval.iter().map(|&s| harness.v(s)).collect();
        println!("curve: r={:.4}", pearson(&v, &v_star));
    }
    harness.run_episodes(&corridor, &mut rng, 50_000, None);

    let v_dopa: Vec<f32> = eval.iter().map(|&s| harness.v(s)).collect();

    // Ridge floor: batch fit on the SAME code features → realized returns.
    let samples: Vec<(Vec<f32>, f32)> = samples.into_iter().take(40_000).collect();
    let u = fit_ridge_readout(&samples, cfg.n_kc, 1.0);
    // In-sample ridge sanity: does the fit explain its own targets?
    {
        let sub: Vec<(Vec<f32>, f32)> = samples.iter().take(2_000).cloned().collect();
        let pred: Vec<f32> = sub
            .iter()
            .map(|(code, _)| {
                code.iter()
                    .zip(u.iter())
                    .map(|(&c, &w)| c * w as f32)
                    .sum::<f32>()
            })
            .collect();
        let targ: Vec<f32> = sub.iter().map(|(_, g)| *g).collect();
        println!("ridge in-sample r={:.4}", pearson(&pred, &targ));
    }
    let mut ridge_scratch = MbScratch::new(&harness.circuit);
    let mut ridge_code = vec![0u32; harness.circuit.kc_active()];
    let mut ridge_val = |s: f32, circuit: &MbCircuit, scratch: &mut MbScratch| -> f32 {
        circuit.code_into(&Corridor::features(s), &[], scratch, &mut ridge_code);
        let mut acc = 0.0f64;
        for &kc in &ridge_code {
            acc += u[kc as usize];
        }
        acc as f32
    };
    let v_ridge: Vec<f32> = eval
        .iter()
        .map(|&s| ridge_val(s, &harness.circuit, &mut ridge_scratch))
        .collect();

    let r_dopa = pearson(&v_dopa, &v_star);
    let r_dopa_sp = spearman(&v_dopa, &v_star);
    let r_ridge = pearson(&v_ridge, &v_star);
    println!(
        "phase1: dopamine r={r_dopa:.4} (spearman {r_dopa_sp:.4}) vs ridge-batch r={r_ridge:.4}"
    );
    for i in (0..64).step_by(9) {
        println!(
            "  s={:.3} v*={:.3} dopa={:.3} ridge={:.3}",
            eval[i], v_star[i], v_dopa[i], v_ridge[i]
        );
    }
    assert!(
        r_dopa >= 0.85,
        "dopamine value formation too weak: r={r_dopa}"
    );
    assert!(
        r_dopa >= r_ridge - 0.05,
        "dopamine must match the ridge-batch floor: {r_dopa} vs {r_ridge}"
    );

    // ── Phase 2 (the shift arm): LEFT pays now. Online adapts; the frozen
    // batch fit cannot. ─────────────────────────────────────────────────────
    let corridor2 = Corridor {
        step: 0.05,
        scale: 4.0,
        gamma: 0.98,
        sign: -1.0,
    };
    let v_star2: Vec<f32> = eval.iter().map(|&s| corridor2.true_value(s)).collect();
    harness.run_episodes(&corridor2, &mut rng, 2_500, None);

    let v_dopa2: Vec<f32> = eval.iter().map(|&s| harness.v(s)).collect();
    let v_ridge2: Vec<f32> = eval
        .iter()
        .map(|&s| ridge_val(s, &harness.circuit, &mut ridge_scratch))
        .collect(); // frozen phase-1 fit
    let r_dopa2 = pearson(&v_dopa2, &v_star2);
    let r_ridge2 = pearson(&v_ridge2, &v_star2);
    println!("phase2 (shift): dopamine r={r_dopa2:.4} vs FROZEN ridge r={r_ridge2:.4}");
    assert!(r_dopa2 >= 0.5, "online must adapt post-shift: r={r_dopa2}");
    assert!(
        r_ridge2 <= 0.0,
        "frozen batch fit must break post-shift: r={r_ridge2}"
    );
}

// ── 7. trajectory determinism (bit-identical repeat) ────────────────────────

#[test]
fn identical_training_streams_are_bit_identical() {
    let run = |seed: u64| -> Vec<f32> {
        let mut cfg = MbCircuitConfig::toy();
        cfg.n_action_dims = 0;
        let mut circuit = MbCircuit::new(&cfg);
        let mut rng = Rng::new(seed);
        let n = 400;
        let mut feats = vec![0.0f32; n * cfg.n_features];
        for s in 0..n {
            let st = (s as f32 + 0.5) / n as f32;
            let f = Corridor::features(st);
            feats[s * cfg.n_features..(s + 1) * cfg.n_features].copy_from_slice(&f);
        }
        circuit.calibrate(&feats, &[], n, 0.5);
        let corridor = Corridor {
            step: 0.05,
            scale: 5.0,
            gamma: 0.98,
            sign: 1.0,
        };
        let mut harness = CorridorHarness::new(circuit);
        harness.run_episodes(&corridor, &mut rng, 200, None);
        harness.circuit.w().to_vec()
    };
    let a = run(29);
    let b = run(29);
    assert_eq!(a, b, "same seed + same stream must be bit-identical");
}

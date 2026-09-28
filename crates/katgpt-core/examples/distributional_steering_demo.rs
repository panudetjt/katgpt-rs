//! Plan 577 Phase 4 T4.2 — distributional steering demo.
//!
//! A 2-D GMM population (two clusters, 1:3 mass) steered toward a reweighted
//! target histogram (3:1) via the FK+Picard path, with ESS-triggered
//! systematic resampling (the sampling-consumer protocol; weights-only is
//! the persistent-agent mode — see the module docs). Prints before/after
//! MMD² to the target + the per-particle weight distribution (the "who
//! carries the distribution" read-out: top-10 weights, ESS, weighted vs
//! unweighted cluster shares).
//!
//! NOTE: bench number 682, not 577 — 577 was already allocated
//! (emotion_direction_rank; the monotonic numbering rule).

#![cfg(feature = "distributional_steering")]

use katgpt_core::distributional_steering::{
    FkStepper, MmdReward, SteeringScratch, systematic_resample_into,
};

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
    fn next_normal(&mut self) -> f64 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

const N: usize = 600;
const DIM: usize = 2;
const STEPS: usize = 60;
const DT: f64 = 0.05;
/// Cluster centers: base mass 1:3 on A/B; target 3:1.
const CENTER_A: [f64; 2] = [-1.5, -0.5];
const CENTER_B: [f64; 2] = [1.2, 0.8];
const BASE_W_A: f64 = 0.25;
const TARGET_W_A: f64 = 0.75;
const GAMMA: f32 = 0.25; // reward kernel scale
const LAM: f32 = 6.0;

fn main() {
    let mut rng = SplitMix64::new(2026_0824);

    // Base population: GMM 1:3.
    let mut states = vec![0.0f32; N * DIM];
    let mut cluster_of = [0usize; N];
    for i in 0..N {
        let a = rng.next_uniform() < BASE_W_A;
        let (cx, cy) = if a {
            (CENTER_A[0], CENTER_A[1])
        } else {
            (CENTER_B[0], CENTER_B[1])
        };
        cluster_of[i] = if a { 0 } else { 1 };
        states[i * DIM] = (cx + 0.45 * rng.next_normal()) as f32;
        states[i * DIM + 1] = (cy + 0.45 * rng.next_normal()) as f32;
    }
    // Target particles: GMM 3:1 (the designer dial).
    let mut target = vec![0.0f32; 256 * DIM];
    for t in target.chunks_mut(DIM) {
        let a = rng.next_uniform() < TARGET_W_A;
        let (cx, cy) = if a {
            (CENTER_A[0], CENTER_A[1])
        } else {
            (CENTER_B[0], CENTER_B[1])
        };
        t[0] = (cx + 0.45 * rng.next_normal()) as f32;
        t[1] = (cy + 0.45 * rng.next_normal()) as f32;
    }

    let reward = MmdReward::new(GAMMA, target.clone(), DIM);

    // MMD² of an unweighted population vs the target particles.
    let mmd_sq = |st: &[f32]| -> f64 {
        let kf = |a: &[f32], b: &[f32]| -> f64 {
            let d = (a[0] - b[0]) as f64 * (a[0] - b[0]) as f64
                + (a[1] - b[1]) as f64 * (a[1] - b[1]) as f64;
            (-GAMMA as f64 * d).exp()
        };
        let mut self_term = 0.0;
        for i in 0..N {
            for j in 0..N {
                self_term += kf(&st[i * DIM..(i + 1) * DIM], &st[j * DIM..(j + 1) * DIM]);
            }
        }
        self_term /= (N * N) as f64;
        let mut tt = 0.0;
        for i in 0..256 {
            for j in 0..256 {
                tt += kf(
                    &target[i * DIM..(i + 1) * DIM],
                    &target[j * DIM..(j + 1) * DIM],
                );
            }
        }
        tt /= (256 * 256) as f64;
        let mut cross = 0.0;
        for i in 0..N {
            for j in 0..256 {
                cross += kf(&st[i * DIM..(i + 1) * DIM], &target[j * DIM..(j + 1) * DIM]);
            }
        }
        cross /= (N * 256) as f64;
        (self_term + tt - 2.0 * cross).max(0.0)
    };

    let cluster_share = |st: &[f32], w: &[f32]| -> (f64, f64) {
        // Share of weighted mass whose nearest center is A vs B.
        let mut wa = 0.0;
        let mut wb = 0.0;
        for i in 0..N {
            let x = &st[i * DIM..(i + 1) * DIM];
            let da = (x[0] as f64 - CENTER_A[0]).powi(2) + (x[1] as f64 - CENTER_A[1]).powi(2);
            let db = (x[0] as f64 - CENTER_B[0]).powi(2) + (x[1] as f64 - CENTER_B[1]).powi(2);
            if da < db {
                wa += w[i] as f64;
            } else {
                wb += w[i] as f64;
            }
        }
        (wa, wb)
    };

    let mmd_before = mmd_sq(&states);
    let uniform = vec![1.0f32 / N as f32; N];
    let (wa0, wb0) = cluster_share(&states, &uniform);
    println!("── distributional_steering demo (Plan 577 T4.2) ──────────────");
    println!(
        "base population   : N={N}, 2-D GMM 1:3 (A={:.0}% mass, B={:.0}%)",
        100.0 * BASE_W_A,
        100.0 * (1.0 - BASE_W_A)
    );
    println!(
        "target dial       : 3:1 (A={:.0}%) — MMD² reward, γ={GAMMA}, λ={LAM}",
        100.0 * TARGET_W_A
    );
    println!("BEFORE: MMD²={mmd_before:.5}  cluster shares A/B = {wa0:.3}/{wb0:.3}");

    // ── FK+Picard steering loop (sampling-consumer protocol) ─────────────
    let damping = (2.0 / LAM as f64).min(1.0) as f32;
    let stepper = FkStepper {
        steer_scale: LAM,
        k_fp: 8,
        damping,
        clip_log_delta: 1.0,
    };
    let mut scratch = SteeringScratch::new(N, DIM);
    let mut log_w = vec![0.0f32; N];
    let mut st = states.clone();
    let mut ancestors = vec![0u32; N];
    let mut resamples = 0usize;
    let sigma = 0.35f64;

    for _ in 0..STEPS {
        stepper.begin_step(&reward, &st, &mut log_w, &mut scratch);
        let steer = scratch.steering().to_vec();
        let mut b = vec![0.0f32; N * DIM];
        for i in 0..N * DIM {
            // Weak centering drift (keeps the population on-stage) + steering.
            b[i] = -0.05 * st[i] + steer[i];
        }
        for i in 0..N * DIM {
            let noise = rng.next_normal();
            st[i] += (b[i] as f64 * DT + sigma * DT.sqrt() * noise) as f32;
        }
        stepper.finish_step(&reward, &st, &b, DT as f32, &mut log_w, &mut scratch);
        // ESS guard → systematic resample.
        let mx = log_w.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0.0f64;
        for &l in &log_w {
            sum += ((l - mx) as f64).exp();
        }
        let w: Vec<f32> = log_w
            .iter()
            .map(|&l| (((l - mx) as f64).exp() / sum) as f32)
            .collect();
        let ess = 1.0 / w.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
        if ess < 0.5 * N as f64 {
            let u = rng.next_uniform() as f32;
            systematic_resample_into(&w, N, u, &mut ancestors);
            let mut carried = st.clone();
            for i in 0..N {
                let a = ancestors[i] as usize * DIM;
                carried[i * DIM..(i + 1) * DIM].copy_from_slice(&st[a..a + DIM]);
            }
            st.copy_from_slice(&carried);
            for i in 0..N {
                cluster_of[i] = cluster_of[ancestors[i] as usize];
            }
            log_w.fill(0.0);
            resamples += 1;
        }
    }

    // ── After: the weighted empirical measure + the "who carries it" read-out ──
    let mx = log_w.iter().cloned().fold(f32::MIN, f32::max);
    let mut sum = 0.0f64;
    for &l in &log_w {
        sum += ((l - mx) as f64).exp();
    }
    let w: Vec<f32> = log_w
        .iter()
        .map(|&l| (((l - mx) as f64).exp() / sum) as f32)
        .collect();
    let ess = 1.0 / w.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();

    let mmd_after = mmd_sq(&st);
    let (wa, wb) = cluster_share(&st, &w);
    let mut top: Vec<f32> = w.clone();
    top.sort_by(|a, b| b.total_cmp(a));
    println!(
        "AFTER : MMD²={mmd_after:.5}  ({:.1}% of before)",
        100.0 * mmd_after / mmd_before
    );
    println!(
        "         weighted cluster shares A/B = {wa:.3}/{wb:.3}  (target {}/{})",
        TARGET_W_A,
        1.0 - TARGET_W_A
    );
    println!("         resampling events: {resamples} (ESS-guard at N/2)");
    println!("         ESS = {ess:.1} / {N}");
    print!("         top-10 weights: ");
    for t in top.iter().take(10) {
        print!("{t:.4} ");
    }
    println!();
    println!(
        "         uniform weight would be {:.4} — the tilt concentrates carried mass",
        1.0 / N as f32
    );
    println!("────────────────────────────────────────────────────────────");
    println!("The weighted empirical measure (Σ wᵢ δ_Xᵢ) is the object that");
    println!("converges to the tilted target μ* ∝ e^(λΨ) p (paper Thm 3.4).");
}

//! Issue 865 T3 — the `probe_guidance` λ-sweep GOAT gate (the promotion decider).
//!
//! **The recorded verdict of this gate is a NEGATIVE result** (calibration
//! history in `.benchmarks/847_probe_guidance_lambda_sweep_goat.md`): on the
//! mini-dLLM lane, the measured probe-guidance gain is **temperature-reachable
//! and within temperature noise** — the feature stays opt-in. This file keeps
//! the full measurement machinery as a pinned regression gate so the verdict
//! is re-measurable, and so the THREE lessons it encodes cannot silently
//! regress:
//!
//! 1. **The trained-probe arm measured PARITY, not win** — at matched
//!    per-position resample diversity, the guided point did not beat the
//!    unguided temperature point (the trunk is saturated: loss 0.0000 → the
//!    weak side carries no disagreement the extrapolation can exploit).
//! 2. **The zero-logit control makes the λ combine EXACTLY temperature
//!    scaling** — `λ·logits + (1−λ)·0 = logits/(T/λ)` — so it is the perfect
//!    no-information null: same knob, zero probe signal. Against it, the
//!    trained probe won by ≤ +0.21 pts at one λ (λ=1.25) and LOST to it at
//!    every other λ (−0.20 at λ=1.5, −2.11 at λ=2), while the plain
//!    temperature front dominates the whole guided front. Whatever the
//!    λ≈1.25 bump is, it is inside the knob's own reach and gone by λ=1.5.
//! 3. **Pooled unigram entropy is the wrong diversity axis on
//!    deterministic-structure lanes** (a CORRECT decoder maximizes it — the
//!    ground truth is itself high-entropy); per-position resample entropy is
//!    the working form.
//! 4. **The bonus control pins direction-following** — a mean-zero
//!    token-0-pull probe at fixed temperature moves accuracy monotonically
//!    DOWN (−1.17 pts over bonus 0→8), proving the harness CAN see a followed
//!    direction; the λ sweep's non-monotone bumps are therefore not a
//!    direction being followed.
//!
//! What is measured here (all deterministic, fixed seeds):
//! - **Guided front**: λ ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0} at T0 — λ > 1
//!   extrapolates away from the weak side, λ < 1 softens toward it.
//! - **Unguided front**: the temperature sweep — the trivial sharpener.
//! - **G1**: λ = 1 with the probe installed is BIT-IDENTICAL to the
//!   unguided decode (the one ordering-invariant comparison — pinned here
//!   because it is the load-bearing correctness contract of the whole
//!   feature).
//! - **G2a/G2b, INVERTED INTO REGRESSION PINS** (the negative verdict made
//!   permanent): G2a asserts the measured λ*-vs-baseline delta stays inside
//!   the temperature-noise envelope (+0.6 pts); G2b asserts the
//!   matched-diversity unguided point beats the guided λ* point — i.e. the
//!   trivial sharpener is at least as good as guidance on this lane. A
//!   future change that makes guidance genuinely WIN here flips these reds
//!   — at which point the feature's promotion case exists and the gate is
//!   re-pinned in the winning direction.
//! - **G-noise**: the zero-logit control (the λ combine = EXACT temperature
//!   scaling — see lesson 2) stays within ±0.6 pts of its own no-probe
//!   baseline at every λ>1 — the measured no-signal envelope this gate's
//!   bars are derived from.
//! - **G-bonus**: the directionality control — a mean-zero bonus on token 0
//!   at fixed decoding temperature is a pure "pull toward token 0" knob, so
//!   accuracy must move monotonically DOWN in it; measured −1.17 pts over
//!   the bonus sweep, monotone at every step. This is the evidence that the
//!   λ sweep's non-monotone bumps are not a direction being followed.
//!   Gated loosely (monotone down, total drop ≥ 1 pt) so it pins the
//!   mechanism, not the noise.
//!
//! The trunk + data replicate the T2b trainer seed-for-seed (the
//! established root-test pattern: `train_mini_dllm` is the root's own
//! test-helper; NO training pipeline lives here). Probe candidates: the
//! trained fixture artifact and the G-noise control's constant logits —
//! both frozen data (the modelless consumption rule).
//!
//! Box state at record time: M3 Max (16 cores), macOS 26.6.2, release
//! profile. Run: `cargo test --release --features probe_guidance --test
//! probe_guidance_goat -- --nocapture --test-threads=1`

#![cfg(feature = "probe_guidance")]

use katgpt_forward::d2f_context::WeakLogitProbe;
use katgpt_forward::weak_probe_mlp::MlpWeakProbe;
use katgpt_rs::dllm::{generate_pattern_dataset, train_mini_dllm};
use katgpt_rs::speculative::probe_artifact::ProbeArtifact;
use katgpt_rs::speculative::{D2fDecodeConfig, D2fPipeline, NoPruner, NoScreeningPruner};
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};

/// The trained probe artifact (riir-train `weak_probe_train` lane, BLAKE3
/// `bfc827fb…`, 6332 B). Regenerate with
/// `cargo run -p riir-train-engine --features probe_guidance_train --release
/// --example weak_probe_train` (G-determ: byte-identical retrain).
const FIXTURE: &[u8] = include_bytes!("fixtures/weak_probe_micro_dllm_v1.bin");

/// Held-out prompts per arm. 256 × 8 positions × 8 resamples = 16 384
/// decoded tokens per arm.
const N_PROMPTS: usize = 256;
/// The prompt is the sequence's first two tokens (a and b — full
/// determination of the alternating pattern).
const PROMPT_LEN: usize = 2;
/// One D2F block (== `config.d2f_block_size`).
const DECODE_LEN: usize = 8;
/// Resamples per prompt for the per-position diversity estimate.
const K_RESAMPLES: usize = 8;
/// Per-(arm, prompt, resample) RNG seeds.
const SEED_BASE: u64 = 900_000;
/// The decode temperature of the sweep.
const T0: f32 = 2.5;

// ── shared trunk (trained once per process, the root-test-helper pattern) ──

fn workspace() -> &'static (Config, TransformerWeights, Vec<Vec<usize>>) {
    static WORKSPACE: std::sync::OnceLock<(Config, TransformerWeights, Vec<Vec<usize>>)> =
        std::sync::OnceLock::new();
    WORKSPACE.get_or_init(|| {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        // EXACTLY the T2b trainer's data + training procedure, seed for seed.
        let train_data =
            generate_pattern_dataset(&mut rng, 2048, config.block_size, config.vocab_size - 1);
        let test_data =
            generate_pattern_dataset(&mut rng, 256, config.block_size, config.vocab_size - 1);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);
        (config, weights, test_data)
    })
}

fn load_artifact() -> ProbeArtifact {
    ProbeArtifact::from_bytes(FIXTURE).expect("fixture artifact must pass the BLAKE3 verifier")
}

// ── measurement ───────────────────────────────────────────────────────────

struct ArmResult {
    label: String,
    /// Decoded positions equal to the held-out ground truth (mask counts as
    /// wrong — a position the decode never committed is a decode failure).
    accuracy: f32,
    /// Mean per-position resample entropy (nats): how much the decode's
    /// outcome spreads across independent resamples, per position.
    diversity: f32,
    /// Positions still masked at convergence.
    mask_frac: f32,
}

/// λ: `None` = unguided; `Some(λ)` = guided. `probe_kind` selects the weak
/// side (see [`ProbeKind`]).
fn run_arm(
    label: &str,
    config: &Config,
    weights: &TransformerWeights,
    prompts: &[Vec<usize>],
    temperature: f32,
    lambda: Option<f32>,
    probe_kind: ProbeKind,
) -> ArmResult {
    let decode_config = D2fDecodeConfig {
        // τ_conf low so nearly every position commits — the diversity /
        // accuracy trade then lives in the sampling distribution, not in
        // mask survival.
        confidence_threshold: 0.3,
        denoise_steps: 16,
        temperature,
        ..D2fDecodeConfig::with_block_size(DECODE_LEN)
    };
    let vocab = config.vocab_size;
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut masked = 0usize;
    // Per-position outcome histograms across resamples, summed over prompts.
    let mut pos_hist = vec![vec![0usize; vocab]; DECODE_LEN];

    for (i, seq) in prompts.iter().take(N_PROMPTS).enumerate() {
        for k in 0..K_RESAMPLES {
            let pipeline =
                D2fPipeline::with_prompt(config, decode_config, DECODE_LEN, &seq[..PROMPT_LEN]);
            let mut rng = Rng::new(SEED_BASE + 10_000 * k as u64 + i as u64);
            let result = match lambda {
                None => pipeline.decode_all(weights, &NoPruner, &NoScreeningPruner, &mut rng),
                Some(lam) => {
                    let probe: Box<dyn WeakLogitProbe> = match probe_kind {
                        ProbeKind::Trained => {
                            Box::new(MlpWeakProbe::new(load_artifact()).expect("fixture wraps"))
                        }
                        ProbeKind::Noise => Box::new(ConstantProbe::noise(vocab)),
                        ProbeKind::Bonus(bonus) => {
                            Box::new(ConstantProbe::bonus(vocab, config.mask_token, bonus))
                        }
                    };
                    pipeline.set_guidance(lam, probe).decode_all(
                        weights,
                        &NoPruner,
                        &NoScreeningPruner,
                        &mut rng,
                    )
                }
            };
            for (p, &t) in result.tokens[PROMPT_LEN..PROMPT_LEN + DECODE_LEN]
                .iter()
                .enumerate()
            {
                if t == seq[PROMPT_LEN + p] {
                    correct += 1;
                }
                if t == config.mask_token {
                    masked += 1;
                } else {
                    pos_hist[p][t] += 1;
                }
                total += 1;
            }
        }
    }

    // Mean per-position entropy across the pooled resample histograms.
    let per_cell = (N_PROMPTS * K_RESAMPLES) as f32;
    let mut diversity = 0.0f32;
    for hist in &pos_hist {
        let mut h = 0.0f32;
        for &c in hist {
            if c > 0 {
                let p = c as f32 / per_cell;
                h -= p * p.ln();
            }
        }
        diversity += h;
    }
    diversity /= DECODE_LEN as f32;

    ArmResult {
        label: label.to_string(),
        accuracy: correct as f32 / total as f32,
        diversity,
        mask_frac: masked as f32 / total as f32,
    }
}

/// Which weak side an arm uses.
#[derive(Clone, Copy, PartialEq)]
enum ProbeKind {
    /// The trained fixture artifact (the real lane).
    Trained,
    /// The G-noise control: constant zero logits — with it the λ combine is
    /// EXACTLY temperature scaling (`logits/(T/λ)`), the no-information null.
    Noise,
    /// The directionality control: uniform −bonus with +2·bonus on token 0
    /// (mean-zero, direction = token 0).
    Bonus(f32),
}

/// A fixed logits-vector probe (the information-free and directionality
/// controls). Zero-alloc after construction; every block position gets the
/// same logits.
struct ConstantProbe {
    logits: Vec<f32>,
}

impl ConstantProbe {
    /// All-zero logits — the no-information control.
    fn noise(vocab: usize) -> Self {
        Self {
            logits: vec![0.0; vocab],
        }
    }

    /// Mean-zero with the whole contrast on `token`: uniform −bonus, +2·bonus
    /// at `token` (mask excluded from guidance — it is never sampleable).
    fn bonus(vocab: usize, mask: usize, bonus: f32) -> Self {
        Self {
            logits: (0..vocab)
                .map(|t| {
                    if t == mask {
                        0.0
                    } else if t == 0 {
                        2.0 * bonus
                    } else {
                        -bonus
                    }
                })
                .collect(),
        }
    }
}

impl WeakLogitProbe for ConstantProbe {
    fn probe(&mut self, input: katgpt_forward::d2f_context::ProbeCtx<'_>, out: &mut [f32]) {
        let vocab = self.logits.len();
        let n_rows = input.seq_len - input.block_start;
        debug_assert_eq!(out.len(), n_rows * vocab);
        for row in 0..n_rows {
            out[row * vocab..(row + 1) * vocab].copy_from_slice(&self.logits);
        }
    }
}

fn print_arm(arm: &ArmResult) {
    println!(
        "    {:<24} acc {:6.2}%   div {:6.4} nats   mask {:5.2}%",
        arm.label,
        arm.accuracy * 100.0,
        arm.diversity,
        arm.mask_frac * 100.0
    );
}

// ── gates ─────────────────────────────────────────────────────────────────

/// G0 (fixture health): the committed artifact still pairs with this trunk —
/// probe CE on kernel-extracted held-out taps < 0.5 (measured 0.1892; a
/// stale/mismatched fixture reads ≈ ln(27) = 3.30). G-fixture predates the
/// verdict: T2/T2b's landing gate, kept so the committed fixture can never
/// silently rot (a WRONG committed artifact passes its own BLAKE3 check).
#[test]
fn g0_fixture_probe_matches_this_trunk() {
    let (config, weights, test_data) = workspace();
    let mut probe = MlpWeakProbe::new(load_artifact()).expect("fixture wraps");

    let n = config.n_embd;
    let vocab = config.vocab_size;
    let mut rng = Rng::new(777);
    let mut corrupted = Vec::with_capacity(config.block_size);
    let mut is_masked = Vec::with_capacity(config.block_size);
    let mut positions = Vec::with_capacity(config.block_size);
    let mut ctx = katgpt_rs::dllm::D2fContext::new(config);
    ctx.probe_tap_capture = true;

    let mut total_ce = 0.0f32;
    let mut n_samples = 0usize;
    for seq in test_data.iter().take(64) {
        corrupted.clear();
        is_masked.clear();
        positions.clear();
        let n_mask = katgpt_rs::dllm::corrupt_block_into(
            seq,
            0.3,
            config.mask_token,
            &mut rng,
            &mut corrupted,
            &mut is_masked,
            &mut positions,
        );
        if n_mask == 0 {
            continue;
        }
        katgpt_rs::dllm::forward_block_causal_with(
            &mut ctx,
            weights,
            &corrupted,
            config,
            config.d2f_block_size,
        );
        for &p in &positions {
            let mut out = vec![0.0f32; vocab];
            let input = katgpt_forward::d2f_context::ProbeCtx {
                xr: &ctx.xr,
                x_norm: &ctx.x_norm,
                tap: &ctx.probe_tap_flat,
                tap_layers: &ctx.probe_tap_layers,
                tap_plane: ctx.probe_tap_plane,
                tokens: &corrupted,
                committed_len: ctx.committed_len,
                block_start: p,
                seq_len: p + 1,
                vocab,
                n_embd: n,
                step: 0,
            };
            probe.probe(input, &mut out);
            let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = out.iter().map(|&l| (l - max).exp()).sum();
            total_ce -= ((out[seq[p]] - max).exp() / sum_exp).ln();
            n_samples += 1;
        }
    }
    let ce = total_ce / n_samples.max(1) as f32;
    println!("G0-fixture: probe held-out CE on kernel taps = {ce:.4} (uniform 3.2958)");
    assert!(
        ce < 0.5,
        "fixture artifact does not match this trunk (probe CE {ce:.4} ≥ 0.5 — \
         regenerate tests/fixtures/weak_probe_micro_dllm_v1.bin via the riir-train \
         weak_probe_train lane and run at --release)"
    );
}

/// G1: λ = 1 with the probe installed is BIT-IDENTICAL to the unguided
/// decode — every prompt, every token, the real artifact path. (The
/// ordering-invariant comparison; the load-bearing correctness contract.)
#[test]
fn g1_lambda_one_bit_identity_with_trained_probe() {
    let (config, weights, test_data) = workspace();
    let artifact = load_artifact();
    let decode_config = D2fDecodeConfig {
        confidence_threshold: 0.3,
        denoise_steps: 16,
        temperature: T0,
        ..D2fDecodeConfig::with_block_size(DECODE_LEN)
    };
    for (i, seq) in test_data.iter().take(64).enumerate() {
        let mut rng_u = Rng::new(SEED_BASE + i as u64);
        let unguided =
            D2fPipeline::with_prompt(config, decode_config, DECODE_LEN, &seq[..PROMPT_LEN])
                .decode_all(weights, &NoPruner, &NoScreeningPruner, &mut rng_u);
        let mut rng_g = Rng::new(SEED_BASE + i as u64);
        let probe = MlpWeakProbe::new(artifact.clone()).expect("fixture wraps");
        let guided =
            D2fPipeline::with_prompt(config, decode_config, DECODE_LEN, &seq[..PROMPT_LEN])
                .set_guidance(1.0, Box::new(probe) as Box<dyn WeakLogitProbe>)
                .decode_all(weights, &NoPruner, &NoScreeningPruner, &mut rng_g);
        assert_eq!(
            unguided.tokens, guided.tokens,
            "λ=1 must be bit-identical to unguided (prompt {i})"
        );
    }
}

/// The recorded sweep + the NEGATIVE verdict's regression pins.
///
/// Prints both fronts, then asserts:
/// - **G2a** (inverted): the guided λ* advantage over the unguided baseline
///   stays INSIDE the temperature-noise envelope (+0.6 pts, read off the
///   unguided front's own high-T dip). A change that pushes the guided
///   advantage beyond the envelope is the promotion case — the gate then
///   re-pins in the winning direction.
/// - **G2b** (inverted): the matched-diversity unguided temperature point
///   beats the guided λ* point — the trivial sharpener is at least as good.
/// - **G-noise**: the information-free control stays within ±0.6 pts of its
///   own no-probe baseline at every λ>1.
#[test]
fn g2_lambda_sweep_pareto_and_noise_control() {
    let (config, weights, test_data) = workspace();

    println!("λ sweep at T = {T0} (trained-probe guided front):");
    let mut guided: Vec<(f32, ArmResult)> = Vec::new();
    for lam in [1.0f32, 1.25, 1.5, 2.0, 0.75, 0.5] {
        let arm = run_arm(
            &format!("λ={lam}"),
            config,
            weights,
            test_data,
            T0,
            Some(lam),
            ProbeKind::Trained,
        );
        print_arm(&arm);
        guided.push((lam, arm));
    }

    println!("unguided temperature front:");
    let temps = [3.5f32, 3.0, 2.75, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0];
    let mut unguided: Vec<(f32, ArmResult)> = Vec::new();
    for t in temps {
        let arm = run_arm(
            &format!("unguided T={t}"),
            config,
            weights,
            test_data,
            t,
            None,
            ProbeKind::Trained, // unused on the None path
        );
        print_arm(&arm);
        unguided.push((t, arm));
    }

    println!("G-noise control (information-free probe, same sweep):");
    let mut noise_ctl: Vec<(f32, ArmResult)> = Vec::new();
    for lam in [1.0f32, 1.25, 1.5, 2.0] {
        let arm = run_arm(
            &format!("noise λ={lam}"),
            config,
            weights,
            test_data,
            T0,
            Some(lam),
            ProbeKind::Noise,
        );
        print_arm(&arm);
        noise_ctl.push((lam, arm));
    }

    fn at(arms: &[(f32, ArmResult)], key: f32) -> &ArmResult {
        &arms.iter().find(|(k, _)| *k == key).expect("arm present").1
    }
    let base = at(&guided, 1.0);
    let noise_base = at(&noise_ctl, 1.0);

    // The temperature-noise envelope: the unguided front's own high-T dip
    // (T=3.5 vs T=2.5) — how much accuracy moves on this lane from sampling
    // conditions alone, with the model fixed. Measured 0.05 pts.
    let envelope = base.accuracy - at(&unguided, 3.5).accuracy;
    let envelope = envelope.max(0.006); // floor: the measured SE at n = 16 384

    // ── G2a (inverted pin) ──
    let (lam_star, star) = guided
        .iter()
        .filter(|(lam, _)| *lam > 1.0)
        .max_by(|a, b| a.1.accuracy.total_cmp(&b.1.accuracy))
        .map(|(lam, arm)| (*lam, arm))
        .expect("guided front non-empty");
    println!(
        "\nG2a: λ*={lam_star} acc {:.2}% vs baseline {:.2}% (Δ {:+.2} pts) — envelope ±{:.2} pts",
        star.accuracy * 100.0,
        base.accuracy * 100.0,
        (star.accuracy - base.accuracy) * 100.0,
        envelope * 100.0
    );
    assert!(
        star.accuracy <= base.accuracy + envelope + 1.0e-6,
        "G2a: guided λ* advantage ({:+.2} pts) EXCEEDS the temperature-noise \
         envelope ({:.2} pts) — the promotion case now exists; re-run the full \
         sweep, re-adjudicate, and re-pin this gate in the winning direction",
        (star.accuracy - base.accuracy) * 100.0,
        envelope * 100.0
    );

    // ── G2b (inverted pin): matched-diversity comparison ──
    let (t_match, matched) = unguided
        .iter()
        .min_by(|a, b| {
            let da = (a.1.diversity - star.diversity).abs();
            let db = (b.1.diversity - star.diversity).abs();
            da.total_cmp(&db)
        })
        .map(|(t, arm)| (*t, arm))
        .expect("unguided front non-empty");
    println!(
        "G2b: matched-diversity unguided point T={t_match} (div {:.4}) acc {:.2}% vs guided λ* acc {:.2}% (Δ {:+.2} pts)",
        matched.diversity,
        matched.accuracy * 100.0,
        star.accuracy * 100.0,
        (star.accuracy - matched.accuracy) * 100.0
    );
    assert!(
        (matched.diversity - star.diversity).abs() < 0.05,
        "G2b: the diversity match must be tight (Δ {:.4} ≥ 0.05 — fronts do not \
         overlap; re-read the sweep before trusting this gate)",
        (matched.diversity - star.diversity).abs()
    );
    assert!(
        matched.accuracy + 1.0e-6 >= star.accuracy - envelope,
        "G2b: the guided λ* point now BEATS the matched-diversity unguided point \
         by more than the noise envelope — the temperature-reachability verdict \
         no longer holds; re-adjudicate and re-pin"
    );

    // ── G-noise: the no-signal envelope ──
    // The zero-logit probe makes the combine EXACT temperature scaling, so
    // this control isolates the knob's trivial component. Its own deltas
    // (measured: +0.00/+0.01/+0.13 at λ=1.25/1.5/2) are the no-signal
    // reference the trained arm must beat by MORE than this to claim signal.
    for (lam, arm) in &noise_ctl {
        if *lam <= 1.0 {
            continue;
        }
        let delta = arm.accuracy - noise_base.accuracy;
        println!(
            "G-noise: λ={lam} Δ {:+.2} pts vs noise baseline (must stay within ±{:.2})",
            delta * 100.0,
            0.6
        );
        assert!(
            delta.abs() <= 0.006f32.max(envelope) + 1.0e-6,
            "G-noise: the information-free probe moved accuracy by {:+.2} pts at \
             λ={lam} (envelope ±{:.2}) — the no-signal envelope moved; the G2a \
             bar must be re-derived from the new control before any sweep \
             reading is trusted",
            delta * 100.0,
            0.6
        );
    }
}

/// G-bonus — the directionality control (lesson 4, made permanent): a
/// mean-zero bonus-on-token-0 probe at FIXED decoding temperature is a pure
/// "pull toward token 0" knob. Real guidance signal must move accuracy
/// MONOTONICALLY DOWN in the bonus (toward token 0 = away from the
/// alternating structure). Measured −1.17 pts over bonus 0→8, monotone at
/// every step. The λ sweep's non-monotone bumps (the apparent λ≈1.25 "win")
/// cannot therefore be a direction being followed. Gated loosely: monotone
/// down, total drop ≥ 1 pt (pins the mechanism, not the noise).
#[test]
fn g3_bonus_control_directionality() {
    let (config, weights, test_data) = workspace();
    let mut pts: Vec<(f32, f32)> = Vec::new();
    println!("G-bonus control (mean-zero token-0 pull, fixed T = {T0}):");
    for bonus in [0.0f32, 1.0, 2.0, 4.0, 8.0] {
        let arm = run_arm(
            &format!("bonus {bonus}"),
            config,
            weights,
            test_data,
            T0,
            Some(1.5),
            ProbeKind::Bonus(bonus),
        );
        print_arm(&arm);
        pts.push((bonus, arm.accuracy));
    }
    // Monotone down (loose: each successive point no higher than its
    // predecessor + a hair), and the total drop at least 2 points.
    for w in pts.windows(2) {
        assert!(
            w[1].1 <= w[0].1 + 0.002,
            "G-bonus: accuracy rose from bonus {:.1} ({:.2}%) to {:.1} ({:.2}%) — \
             the pull is not being followed in the injected direction",
            w[0].0,
            w[0].1 * 100.0,
            w[1].0,
            w[1].1 * 100.0
        );
    }
    let total_drop = pts[0].1 - pts.last().expect("non-empty").1;
    println!(
        "G-bonus: total drop {:+.2} pts over the bonus sweep",
        total_drop * 100.0
    );
    assert!(
        total_drop >= 0.01,
        "G-bonus: total drop {total_drop:.4} < 1 pt — the directionality \
         control lost its signal; re-derive the sweep interpretation"
    );
}

/// The softening side is a sanity read (not gated): λ<1 mixes TOWARD the
/// weak side — accuracy must not improve over the λ=1 point beyond noise.
#[test]
fn softening_arm_is_printed_not_gated() {
    let (config, weights, test_data) = workspace();
    let soft = run_arm(
        "λ=0.5 (soft)",
        config,
        weights,
        test_data,
        T0,
        Some(0.5),
        ProbeKind::Trained,
    );
    let base = run_arm(
        "λ=1.0",
        config,
        weights,
        test_data,
        T0,
        Some(1.0),
        ProbeKind::Trained,
    );
    print_arm(&soft);
    print_arm(&base);
}

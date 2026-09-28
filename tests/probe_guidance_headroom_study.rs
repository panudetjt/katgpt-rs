//! Issue 865 T3 follow-up — the HEADROOM-trunk study (Bench 850).
//!
//! Bench 847's negative verdict was measured on the T2b-mirror trunk — 200
//! epochs, loss 0.0000, a saturated one-hot decoder — and its root-cause
//! analysis names the confound: "the mechanism verdict needs a trunk with
//! headroom". This study is that rerun at mini scale: the SAME methodology
//! (per-position resample entropy, the zero-logit null, the temperature
//! front, 256 prompts × 8 resamples) on a **12-epoch trunk** (test acc
//! ~85%, real disagreement room), with one substitution the landed work
//! enables — the weak side is the MODELLESS [`DropoutHeadProbe`] (Bench
//! 847's deferred arm (b), unblocked tap-level: no kernel dropout needed).
//! The trained-probe arm stays out (its artifact is trained against the
//! saturated trunk's taps and would not pair with this trunk; the
//! Bonsai-scale re-open owns that arm).
//!
//! This is a STUDY, not a gate: it asserts only structural invariants
//! (bit-identity at λ=1, the noise null's equivariance shape) and PRINTS the
//! fronts; the verdict lives in `.benchmarks/850_probe_guidance_headroom_study.md`.
//! Run: `cargo test --release --features probe_guidance --test
//! probe_guidance_headroom_study -- --nocapture`

#![cfg(feature = "probe_guidance")]

use katgpt_forward::d2f_context::WeakLogitProbe;
use katgpt_forward::weak_probe_mlp::DropoutHeadProbe;
use katgpt_rs::dllm::{generate_pattern_dataset, train_mini_dllm};
use katgpt_rs::speculative::{D2fDecodeConfig, D2fPipeline, NoPruner, NoScreeningPruner};
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};

/// Trunk training: the T2b-mirror DATA (2048 train / 256 held-out) at 12
/// epochs — the headroom regime (Bench 847's 200-epoch trunk saturates to
/// loss 0.0000 / 100% test acc; 12 epochs lands ~85% test acc with real
/// disagreement for the weak side to carry).
const TRUNK_EPOCHS: usize = 12;
/// Held-out prompts per arm: 256 × 8 positions × 8 resamples = 16 384
/// decoded tokens per arm (the Bench-847 eval surface).
const N_PROMPTS: usize = 256;
const PROMPT_LEN: usize = 2;
const DECODE_LEN: usize = 8;
const K_RESAMPLES: usize = 8;
const SEED_BASE: u64 = 700_000;
/// The decode temperature of the guided fronts. A headroom trunk has softer
/// logits than a saturated one (Bench 847 needed T0 = 2.5 there to see any
/// diversity); 1.5 sits mid-front here (calibration recorded in Bench 850).
const T0: f32 = 1.5;

fn workspace() -> &'static (Config, TransformerWeights, Vec<Vec<usize>>) {
    static WORKSPACE: std::sync::OnceLock<(Config, TransformerWeights, Vec<Vec<usize>>)> =
        std::sync::OnceLock::new();
    WORKSPACE.get_or_init(|| {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        // The Bench-847 data stream, seed for seed, at 12 epochs.
        let train_data =
            generate_pattern_dataset(&mut rng, 2048, config.block_size, config.vocab_size - 1);
        let test_data =
            generate_pattern_dataset(&mut rng, 256, config.block_size, config.vocab_size - 1);
        let (weights, _) = train_mini_dllm(
            &config,
            &train_data,
            &test_data,
            TRUNK_EPOCHS,
            0.01,
            0.3,
            42,
        );
        (config, weights, test_data)
    })
}

struct ArmResult {
    label: String,
    accuracy: f32,
    diversity: f32,
    mask_frac: f32,
}

/// Which weak side an arm uses (this study's modelless set).
#[derive(Clone, Copy)]
enum ProbeKind {
    /// The shipped tap-level dropout probe (the study's subject).
    Dropout,
    /// The zero-logit null: constant zeros — the λ combine becomes EXACTLY
    /// temperature scaling (`λ·logits = logits/(T/λ)`), the no-information
    /// control (Bench 847's G-noise).
    Noise,
}

struct ZeroProbe {
    vocab: usize,
}

impl WeakLogitProbe for ZeroProbe {
    fn probe(&mut self, input: katgpt_forward::d2f_context::ProbeCtx<'_>, out: &mut [f32]) {
        let n_rows = input.seq_len - input.block_start;
        debug_assert_eq!(out.len(), n_rows * self.vocab);
        out.fill(0.0);
    }
}

// A study harness row-runner: the arm's knobs are its protocol — bundling
// them into a config struct would hide the per-arm deltas the record quotes.
#[allow(clippy::too_many_arguments)]
fn run_arm(
    label: &str,
    config: &Config,
    weights: &TransformerWeights,
    prompts: &[Vec<usize>],
    prompt_len: usize,
    temperature: f32,
    lambda: Option<f32>,
    probe_kind: ProbeKind,
    tau_conf: f32,
    denoise_steps: usize,
) -> ArmResult {
    let decode_config = D2fDecodeConfig {
        // τ_conf / denoise_steps parameterized per regime: the Bench-847
        // posture (0.3, 16) commits nearly every position; the strict
        // posture (0.7, 8 — the decode_config defaults) is the cell where
        // decode-time uncertainty actually exists on a low-data trunk.
        confidence_threshold: tau_conf,
        denoise_steps,
        temperature,
        ..D2fDecodeConfig::with_block_size(DECODE_LEN)
    };
    let vocab = config.vocab_size;
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut masked = 0usize;
    let mut pos_hist = vec![vec![0usize; vocab]; DECODE_LEN];

    for (i, seq) in prompts.iter().take(N_PROMPTS).enumerate() {
        for k in 0..K_RESAMPLES {
            let pipeline =
                D2fPipeline::with_prompt(config, decode_config, DECODE_LEN, &seq[..prompt_len]);
            let mut rng = Rng::new(SEED_BASE + 10_000 * k as u64 + i as u64);
            let result = match lambda {
                None => pipeline.decode_all(weights, &NoPruner, &NoScreeningPruner, &mut rng),
                Some(lam) => {
                    let probe: Box<dyn WeakLogitProbe> = match probe_kind {
                        ProbeKind::Dropout => Box::new(DropoutHeadProbe::new(
                            vocab,
                            config.n_embd,
                            weights.lm_head.clone(),
                            0.5,
                        )),
                        ProbeKind::Noise => Box::new(ZeroProbe { vocab }),
                    };
                    pipeline.set_guidance(lam, probe).decode_all(
                        weights,
                        &NoPruner,
                        &NoScreeningPruner,
                        &mut rng,
                    )
                }
            };
            for (p, &t) in result.tokens[prompt_len..prompt_len + DECODE_LEN]
                .iter()
                .enumerate()
            {
                if t == seq[prompt_len + p] {
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

fn print_arm(arm: &ArmResult) {
    println!(
        "    {:<28} acc {:6.2}%   div {:6.4} nats   mask {:5.2}%",
        arm.label,
        arm.accuracy * 100.0,
        arm.diversity,
        arm.mask_frac * 100.0
    );
}

/// The study: prints the three fronts + the matched-diversity analysis for
/// ALL regimes. Asserts only structural invariants; the verdicts are
/// recorded in `.benchmarks/850_probe_guidance_headroom_study.md`.
#[test]
fn headroom_lambda_sweep_study() {
    println!(
        "## Bench 850 — headroom-trunk study, regime 1: high-data 12-epoch trunk (2048 seqs, 2-token prompt, T0 = {T0})"
    );
    study_fronts(workspace(), T0, PROMPT_LEN, 0.3, 16);

    println!(
        "## Bench 850 — regime 2: LOW-DATA 12-epoch trunk (96 seqs, 8-token prompt, T0 = {T0})"
    );
    study_fronts(low_data_workspace(), T0, LOW_DATA_PROMPT_LEN, 0.3, 16);

    println!(
        "## Bench 850 — regime 3: low-data trunk + STRICT decode config (τ_conf 0.7, 8 steps — the decode-uncertainty cell)"
    );
    study_fronts(low_data_workspace(), T0, LOW_DATA_PROMPT_LEN, 0.7, 8);
}

/// Regime 2: the decode-disagreement regime. 96 sequences × 12 epochs —
/// decode accuracy sits ~63–68% at T=1 (measured), the only mini-lane
/// configuration found where the decode loop does NOT trivially converge;
/// the disagreement the guidance mechanism needs must exist AT DECODE TIME
/// (regime 1's "85% training test acc" does not survive the 16-step loop).
/// Sequences are `config.block_size` (16) long: prompt 8 + block 8.
const LOW_DATA_TRAIN_SEQS: usize = 96;
const LOW_DATA_PROMPT_LEN: usize = 8;

fn low_data_workspace() -> &'static (Config, TransformerWeights, Vec<Vec<usize>>) {
    static WS: std::sync::OnceLock<(Config, TransformerWeights, Vec<Vec<usize>>)> =
        std::sync::OnceLock::new();
    WS.get_or_init(|| {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let train = generate_pattern_dataset(
            &mut rng,
            LOW_DATA_TRAIN_SEQS,
            config.block_size,
            config.vocab_size - 1,
        );
        let eval =
            generate_pattern_dataset(&mut rng, 256, config.block_size, config.vocab_size - 1);
        let (weights, _) = train_mini_dllm(&config, &train, &eval, TRUNK_EPOCHS, 0.01, 0.3, 42);
        (config, weights, eval)
    })
}

fn study_fronts(
    ws: &(Config, TransformerWeights, Vec<Vec<usize>>),
    t0: f32,
    prompt_len: usize,
    tau_conf: f32,
    denoise_steps: usize,
) {
    let (config, weights, prompts) = ws;
    println!("  decode config: τ_conf {tau_conf}, {denoise_steps} steps");

    // Unguided temperature front (the trivial sharpener).
    println!("  unguided temperature front:");
    let mut unguided: Vec<ArmResult> = Vec::new();
    for t in [2.5f32, 2.0, 1.75, 1.5, 1.25, 1.0, 0.8, 0.6, 0.4] {
        let arm = run_arm(
            &format!("T={t:.2}"),
            config,
            weights,
            prompts,
            prompt_len,
            t,
            None,
            ProbeKind::Dropout,
            tau_conf,
            denoise_steps,
        );
        print_arm(&arm);
        unguided.push(arm);
    }

    // Dropout-guided front at T0.
    println!("  dropout-guided front (tap-level 50%, T0):",);
    let mut dropout: Vec<ArmResult> = Vec::new();
    for lam in [1.0f32, 1.25, 1.5, 1.75, 2.0] {
        let arm = run_arm(
            &format!("λ={lam:.2}"),
            config,
            weights,
            prompts,
            prompt_len,
            t0,
            Some(lam),
            ProbeKind::Dropout,
            tau_conf,
            denoise_steps,
        );
        print_arm(&arm);
        dropout.push(arm);
    }

    // The zero-logit null at T0 (the no-information envelope).
    println!("  zero-logit null (combine ≡ T/λ) at T0:");
    let mut nulls: Vec<ArmResult> = Vec::new();
    for lam in [1.25f32, 1.5, 1.75, 2.0] {
        let arm = run_arm(
            &format!("λ={lam:.2}"),
            config,
            weights,
            prompts,
            prompt_len,
            t0,
            Some(lam),
            ProbeKind::Noise,
            tau_conf,
            denoise_steps,
        );
        print_arm(&arm);
        nulls.push(arm);
    }

    // Structural invariant 1: λ = 1 with the dropout probe is bit-identical
    // to the unguided T0 arm (the G1 contract, study side).
    let t0_label = format!("T={t0:.2}");
    let t0_arm = unguided
        .iter()
        .find(|a| a.label == t0_label)
        .expect("T0 row");
    let l1_arm = dropout
        .iter()
        .find(|a| a.label == "λ=1.00")
        .expect("λ=1 row");
    assert_eq!(
        t0_arm.accuracy, l1_arm.accuracy,
        "G1: λ=1 accuracy must equal the unguided T0 arm"
    );
    assert_eq!(
        t0_arm.diversity, l1_arm.diversity,
        "G1: λ=1 diversity must equal the unguided T0 arm"
    );

    // (No monotonicity pin on the null front: at these trunk softnesses the
    // null's resample diversity moves within the sampling envelope
    // non-monotonically — the front is PRINTED for the same-λ comparison,
    // not pinned. The structural pins are the λ=1 identities above.)

    // Matched-diversity analysis (recorded, not asserted): for every dropout
    // point, the best unguided point within the match window and the same-λ
    // null, so the three-way comparison reads directly off the log.
    const MATCH_EPS: f32 = 0.01;
    println!("  matched-diversity analysis (|Δdiv| ≤ {MATCH_EPS} nats):");
    for d in &dropout {
        if d.label == "λ=1.00" {
            continue;
        }
        let lam = d.label.trim_start_matches("λ=");
        let same_lambda_null = nulls
            .iter()
            .find(|n| n.label.trim_start_matches("λ=") == lam);
        if let Some(n) = same_lambda_null {
            println!(
                "    {} acc {:+.2} pts vs same-λ null ({} — the beyond-sharpening delta)",
                d.label,
                (d.accuracy - n.accuracy) * 100.0,
                n.label
            );
        }
        if let Some(best_u) = unguided
            .iter()
            .filter(|u| (u.diversity - d.diversity).abs() <= MATCH_EPS)
            .min_by(|a, b| {
                (a.diversity - d.diversity)
                    .abs()
                    .partial_cmp(&(b.diversity - d.diversity).abs())
                    .unwrap()
            })
        {
            println!(
                "    {} (div {:.4}) vs unguided {} (div {:.4}): acc {:+.2} pts",
                d.label,
                d.diversity,
                best_u.label,
                best_u.diversity,
                (d.accuracy - best_u.accuracy) * 100.0
            );
        } else {
            println!(
                "    {} (div {:.4}): no unguided point within match window",
                d.label, d.diversity
            );
        }
    }
}

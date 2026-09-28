//! Plan 600 T8+T9 GOAT gates — the DBTM confidence-commit anchor rule
//! (κ ∪ floor, Issue 811) measured on a NON-SATURATED corpus, on the
//! production entries only.
//!
//! The Issue 811 PoC arm table (200-epoch toy, 24 sequences) passed on the
//! steps/termination axis at quality PARITY — but its quality axis could not
//! separate arms (anchor-dominated toy). Plan 600 T8 re-measures on a
//! non-saturated corpus (40-epoch training on the same Plan-116/381 pattern
//! lane, 64 held-out sequences) so quality has resolution, and gates:
//!
//! - **G1 quality non-inferiority**: conf+floor ≥ matched-threshold stride
//!   − 2·SE(paired Δ) at every κ, NFE=8. Paired seeds: every arm draws the
//!   identical rng stream per sequence, so the Round-1 walk — and therefore
//!   the selection input — is byte-identical across arms and only the rule
//!   differs. Resolution is asserted (SE(Δ) ≤ 0.05): a gate run on a corpus
//!   too noisy to detect a 2σ difference is not a measurement.
//! - **G2 steps+wall**: the DBTM floor must beat (κ ∈ {0.9, 0.99}) or tie
//!   (κ = 0.5) the matched-threshold stride arm on fill steps at NFE=8,
//!   with wall non-inferior (≤ 1.25×; debug-profile tolerance, ratio
//!   reported).
//! - Corpus liveness: all-mask baseline must sit measurably BELOW every
//!   anchored arm (the anchor round carries signal — otherwise the corpus
//!   is degenerate and G1 proves nothing).
//!
//! T9 (UGC cross-check, Issue 811 T5 / Caveat #1): the KL certificate
//! covers random-order reveal, NOT confidence-greedy reveal. The gate asks
//! the empirical question on the same corpus: does the confidence-commit
//! decode distort the block law MORE than the strided incumbent? Realized
//! KL(P_Z ‖ P̂) is Monte-Carlo measured for both arms over the analytic
//! corpus law; the UGC certificate bound 4Ĉ/N (computed for this model via
//! a `UgcDenoiser` adapter over the bidirectional D2F posterior) is printed
//! alongside as the random-order reference. PASS = the confidence path's
//! realized KL is within 10% of the incumbent's (the incumbent's fixed-order
//! stride reveal is ITSELF outside the certificate's random-order premise,
//! so it is the honest comparator, not the bound).
//!
//! # Run
//!
//! ```bash
//! cargo test --features flashar_anchor --test bench_600_flashar_confidence_commit_goat -- --nocapture
//! ```
//!
//! G3 (incumbent byte-identity with the config off) is asserted by the
//! existing parity tests — `test_issue811_harness_matches_production` and
//! `test_issue600_entry_matches_poc_conf_arm` in `src/speculative/` — which
//! run in the same feature lane; this file adds no second pin.

#![cfg(feature = "flashar_anchor")]

use katgpt_core::ugc_schedule::{
    UGC_MASK, UgcDenoiser, UgcScratch, certified_block_plan, estimate_interval,
};
use katgpt_core::{Config, Rng};
use katgpt_forward::d2f::{D2fBlockResult, D2fDecodeConfig};
use katgpt_forward::{
    AnchorConfig, BidirectionalContext, ConfidenceAnchorConfig, D2fContext, ForwardContext,
    anchor_fill_with_prefilled, anchor_then_fill, anchor_then_fill_with,
    forward_bidirectional_positions_into,
};
use katgpt_rs::dllm::{generate_pattern_dataset, train_mini_dllm};
use katgpt_rs::transformer::{MultiLayerKVCache, TransformerWeights};
use std::cell::RefCell;
use std::time::Instant;

const BLOCK: usize = 8;
const N_EVAL: usize = 64;
/// Non-saturated on purpose: the Issue 811 PoC trained 200 epochs, which
/// saturates the toy and blinds the quality axis (Plan 600 T8 premise).
const TRAIN_EPOCHS: usize = 40;
const KAPPAS: [f32; 3] = [0.5, 0.9, 0.99];
const NFE_CELLS: [usize; 2] = [4, 8];

fn make_nonsaturated_weights() -> (Config, TransformerWeights) {
    let config = Config::micro_dllm();
    let mut train_rng = Rng::new(123);
    let train_data =
        generate_pattern_dataset(&mut train_rng, 20, config.block_size, config.vocab_size - 1);
    let test_data =
        generate_pattern_dataset(&mut train_rng, 5, config.block_size, config.vocab_size - 1);
    let (weights, _) = train_mini_dllm(
        &config,
        &train_data,
        &test_data,
        TRAIN_EPOCHS,
        0.01,
        0.3,
        42,
    );
    (config, weights)
}

fn eval_set(config: &Config) -> Vec<Vec<usize>> {
    let mut rng = Rng::new(777);
    generate_pattern_dataset(&mut rng, N_EVAL, config.block_size, config.vocab_size - 1)
}

// ── Arms — production entries only ──────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum ArmKind {
    /// Context row: the all-mask D2F baseline (no anchors).
    AllMask,
    /// The strided incumbent (`anchor_then_fill`).
    Stride { stride: usize },
    /// The confidence-commit rule (`anchor_then_fill_with`).
    Conf { floor: bool },
}

struct Arm {
    label: &'static str,
    /// The fill round's confidence threshold — matched across arms at each κ
    /// (the PoC's matched-threshold discipline; τ = 0.7 for the context rows).
    kappa: f32,
    kind: ArmKind,
}

struct ArmOutcome {
    mean_acc: f32,
    se_acc: f32,
    per_seq_acc: Vec<f32>,
    mean_steps: f32,
    mean_anchors: f32,
    mean_wall_us: f32,
    all_terminated: bool,
}

struct DecodeOut {
    tokens: Vec<usize>,
    steps: usize,
    n_anchors: usize,
}

#[allow(clippy::too_many_lines)]
fn run_arm(
    arm: &Arm,
    config: &Config,
    weights: &TransformerWeights,
    eval: &[Vec<usize>],
    budget: usize,
) -> ArmOutcome {
    let mask = config.mask_token;
    let decode_config = D2fDecodeConfig {
        denoise_steps: budget,
        confidence_threshold: arm.kappa,
        block_size: BLOCK,
        ..D2fDecodeConfig::default()
    };
    let (mut acc_sum, mut step_sum, mut anchor_sum, mut wall_sum) =
        (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut all_terminated = true;
    let mut per_seq_acc = Vec::with_capacity(N_EVAL);

    for (si, seq) in eval.iter().take(N_EVAL).enumerate() {
        // Paired runs: identical rng stream per sequence across arms, so the
        // walk (and the selection input) is byte-identical and only the rule
        // differs. Pinned for the production entries by the parity tests in
        // `src/speculative/flashar_anchor.rs`.
        let mut rng = Rng::new(9_000 + si as u64);
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let mut dctx = D2fContext::new(config);

        let t0 = Instant::now();
        let out: DecodeOut = match arm.kind {
            ArmKind::AllMask => {
                let anchors = [mask; BLOCK];
                let D2fBlockResult {
                    tokens, steps_used, ..
                } = anchor_fill_with_prefilled(
                    &mut dctx,
                    weights,
                    config,
                    &decode_config,
                    &anchors,
                    &mut rng,
                    None,
                );
                DecodeOut {
                    tokens,
                    steps: steps_used,
                    n_anchors: 0,
                }
            }
            ArmKind::Stride { stride } => {
                let r = anchor_then_fill(
                    &mut ctx,
                    &mut cache,
                    &mut dctx,
                    weights,
                    config,
                    &decode_config,
                    &AnchorConfig::with_stride(stride),
                    seq[0],
                    0,
                    &mut rng,
                );
                DecodeOut {
                    tokens: r.tokens,
                    steps: r.fill_steps_used,
                    n_anchors: r.n_anchors,
                }
            }
            ArmKind::Conf { floor } => {
                let r = anchor_then_fill_with(
                    &mut ctx,
                    &mut cache,
                    &mut dctx,
                    weights,
                    config,
                    &decode_config,
                    &ConfidenceAnchorConfig::new(arm.kappa, floor),
                    seq[0],
                    0,
                    &mut rng,
                );
                DecodeOut {
                    tokens: r.tokens,
                    steps: r.fill_steps_used,
                    n_anchors: r.n_anchors,
                }
            }
        };
        wall_sum += t0.elapsed().as_secs_f32() * 1e6;

        // Ground truth: the pattern is [a, b, a, b, ...]; the block decodes
        // positions 1..=BLOCK given seed seq[0], so slot p expects seq[p+1].
        let mut correct = 0usize;
        for (p, &got) in out.tokens.iter().enumerate() {
            if got == seq[p + 1] {
                correct += 1;
            }
        }
        if out.tokens.contains(&mask) {
            all_terminated = false;
        }
        per_seq_acc.push(correct as f32 / BLOCK as f32);
        acc_sum += per_seq_acc[si];
        step_sum += out.steps as f32;
        anchor_sum += out.n_anchors as f32;
    }

    let n = per_seq_acc.len() as f32;
    let mean = acc_sum / n;
    let var = per_seq_acc
        .iter()
        .map(|&a| (a - mean) * (a - mean))
        .sum::<f32>()
        / n;
    ArmOutcome {
        mean_acc: mean,
        se_acc: var.sqrt() / n.sqrt(),
        per_seq_acc,
        mean_steps: step_sum / n,
        mean_anchors: anchor_sum / n,
        mean_wall_us: wall_sum / n,
        all_terminated,
    }
}

fn build_arms() -> Vec<Arm> {
    let mut arms = vec![
        Arm {
            label: "all-mask D2F baseline",
            kappa: 0.7,
            kind: ArmKind::AllMask,
        },
        Arm {
            label: "stride1 tau0.70",
            kappa: 0.7,
            kind: ArmKind::Stride { stride: 1 },
        },
        Arm {
            label: "stride2 tau0.70 (incumbent)",
            kappa: 0.7,
            kind: ArmKind::Stride { stride: 2 },
        },
        Arm {
            label: "stride4 tau0.70",
            kappa: 0.7,
            kind: ArmKind::Stride { stride: 4 },
        },
    ];
    for &k in &KAPPAS {
        arms.push(Arm {
            label: "stride2 tau=kappa (matched ref)",
            kappa: k,
            kind: ArmKind::Stride { stride: 2 },
        });
    }
    for &k in &KAPPAS {
        arms.push(Arm {
            label: "conf kappa (no floor)",
            kappa: k,
            kind: ArmKind::Conf { floor: false },
        });
    }
    for &k in &KAPPAS {
        arms.push(Arm {
            label: "conf kappa + floor (DBTM)",
            kappa: k,
            kind: ArmKind::Conf { floor: true },
        });
    }
    arms
}

#[test]
#[allow(clippy::too_many_lines)]
fn t8_nonsaturated_corpus_g1_g2_goat() {
    let (config, weights) = make_nonsaturated_weights();
    let eval = eval_set(&config);
    let mut arms = build_arms();

    println!(
        "\n== Plan 600 T8 arm table (non-saturated: {TRAIN_EPOCHS} epochs, {N_EVAL} seqs, block={BLOCK}) =="
    );
    println!(
        "{:<32} {:>6} {:>4} {:>7} {:>7} {:>8} {:>7} {:>9}",
        "arm", "kappa", "NFE", "acc", "se", "steps", "anchors", "wall_us"
    );

    // outcomes[(arm_idx, budget_cell)]
    let mut outcomes: Vec<Vec<ArmOutcome>> = Vec::with_capacity(arms.len());
    for arm in arms.iter_mut() {
        let mut per_budget = Vec::with_capacity(NFE_CELLS.len());
        for &budget in &NFE_CELLS {
            let out = run_arm(arm, &config, &weights, &eval, budget);
            assert!(
                out.all_terminated || !arm.label.contains("DBTM"),
                "DBTM arm must terminate within budget: {} k={} NFE={budget}",
                arm.label,
                arm.kappa
            );
            println!(
                "{:<32} {:>6.2} {:>4} {:>7.3} {:>7.3} {:>8.2} {:>7.2} {:>9.1}",
                arm.label,
                arm.kappa,
                budget,
                out.mean_acc,
                out.se_acc,
                out.mean_steps,
                out.mean_anchors,
                out.mean_wall_us
            );
            per_budget.push(out);
        }
        outcomes.push(per_budget);
    }

    let nfe8 = NFE_CELLS.iter().position(|&b| b == 8).unwrap();
    let find = |label: &str, k: f32| -> &ArmOutcome {
        let idx = arms
            .iter()
            .position(|a| a.label == label && a.kappa == k)
            .unwrap_or_else(|| panic!("arm not found: {label} k={k}"));
        &outcomes[idx][nfe8]
    };

    // ── Corpus liveness: anchors must carry signal (else G1 proves nothing) ──
    let am = find("all-mask D2F baseline", 0.7);
    let sm09 = find("stride2 tau=kappa (matched ref)", 0.9);
    let gap_se = am.se_acc.hypot(sm09.se_acc);
    assert!(
        sm09.mean_acc - am.mean_acc > 2.0 * gap_se,
        "corpus is degenerate for T8: anchored arm ({:.3}) not measurably above all-mask ({:.3})",
        sm09.mean_acc,
        am.mean_acc
    );

    // ── G1 + G2 at every κ, NFE=8 ──
    for &k in &KAPPAS {
        let cf = find("conf kappa + floor (DBTM)", k);
        let sm = find("stride2 tau=kappa (matched ref)", k);

        // G1: paired non-inferiority with asserted resolution.
        let deltas: Vec<f32> = cf
            .per_seq_acc
            .iter()
            .zip(sm.per_seq_acc.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let n = deltas.len() as f32;
        let mean_d = deltas.iter().sum::<f32>() / n;
        let var_d = deltas
            .iter()
            .map(|&d| (d - mean_d) * (d - mean_d))
            .sum::<f32>()
            / n;
        let se_d = var_d.sqrt() / n.sqrt();
        println!(
            "G1 kappa={k}: paired Δ(acc) = {mean_d:+.4} ± {se_d:.4} | conf+floor {:.3} vs matched-stride {:.3}",
            cf.mean_acc, sm.mean_acc
        );
        assert!(
            se_d <= 0.05,
            "paired resolution {se_d:.4} > 0.05 at κ={k} — the corpus cannot detect a 2σ quality difference; non-saturation failed"
        );
        assert!(
            cf.mean_acc >= sm.mean_acc - 2.0 * se_d,
            "G1 FAIL at κ={k}: conf+floor {:.4} below matched-stride {:.4} by more than 2·SE ({:.4})",
            cf.mean_acc,
            sm.mean_acc,
            se_d
        );

        // G2: steps + wall.
        if k > 0.5 {
            assert!(
                cf.mean_steps < sm.mean_steps,
                "G2 FAIL at κ={k}: DBTM steps {:.2} must beat matched-stride {:.2}",
                cf.mean_steps,
                sm.mean_steps
            );
        } else {
            assert!(
                cf.mean_steps <= sm.mean_steps,
                "G2 FAIL at κ={k}: DBTM steps {:.2} must tie or beat matched-stride {:.2}",
                cf.mean_steps,
                sm.mean_steps
            );
        }
        let wall_ratio = cf.mean_wall_us / sm.mean_wall_us;
        println!(
            "G2 kappa={k}: steps {:.2} vs {:.2} | wall ratio {wall_ratio:.2}× | anchors {:.1}",
            cf.mean_steps, sm.mean_steps, cf.mean_anchors
        );
        assert!(
            wall_ratio <= 1.25,
            "G2 FAIL at κ={k}: DBTM wall {:.0}µs is {wall_ratio:.2}× matched-stride {:.0}µs (> 1.25× non-inferiority bar)",
            cf.mean_wall_us,
            sm.mean_wall_us
        );
    }
}

// ── T9: UGC KL cross-check (Caveat #1) ──────────────────────────────────

/// `UgcDenoiser` adapter over the trained bidirectional D2F: the exact
/// single-site posterior at position i given a partially-observed block,
/// renormalized over the 26 non-mask classes (matching the fill path's mask
/// exclusion). Amortized-once construction cost is fine here: T9 is a
/// measurement, not the audited zero-alloc hot path.
struct D2fUgcDenoiser<'a> {
    weights: &'a TransformerWeights,
    config: &'a Config,
    bctx: RefCell<BidirectionalContext>,
}

impl UgcDenoiser for D2fUgcDenoiser<'_> {
    fn dim(&self) -> usize {
        BLOCK
    }

    fn alphabet(&self) -> usize {
        self.config.vocab_size - 1
    }

    fn posterior_into(&self, i: usize, x: &[usize], out: &mut [f32]) {
        let a = self.alphabet();
        let vocab = self.config.vocab_size;
        let mut block = [0usize; BLOCK];
        for (dst, &src) in block.iter_mut().zip(x.iter()) {
            *dst = if src == UGC_MASK {
                self.config.mask_token
            } else {
                src
            };
        }
        let mut bctx = self.bctx.borrow_mut();
        forward_bidirectional_positions_into(self.weights, &block, self.config, &mut bctx);
        let row = &bctx.all_logits[i * vocab..i * vocab + vocab];
        let mut m = f32::NEG_INFINITY;
        for &l in &row[..a] {
            m = m.max(l);
        }
        let mut s = 0.0f32;
        for (t, o) in out.iter_mut().enumerate().take(a) {
            let e = (row[t] - m).exp();
            *o = e;
            s += e;
        }
        for o in out.iter_mut().take(a) {
            *o /= s;
        }
    }
}

/// Analytic corpus law over alternating blocks: the training corpus draws
/// a ~ U{0..25}, b ~ U{0..25} bumped to (a+1)%26 when b == a (the
/// `generate_pattern_dataset` rule). The decoded block is positions 1..=8,
/// so v0 = b, v1 = a. Support: the 650 pairs with v0 != v1.
fn corpus_law() -> Vec<f64> {
    let a_size = 26usize;
    let mut law = vec![0.0f64; a_size * a_size];
    for v1 in 0..a_size {
        for v0 in 0..a_size {
            if v0 == v1 {
                continue;
            }
            let bump = if v0 == (v1 + 1) % a_size { 1.0 } else { 0.0 };
            law[v0 * a_size + v1] = (1.0 + bump) / (a_size * a_size) as f64;
        }
    }
    law
}

/// Monte-Carlo realized KL(P_Z ‖ P̂) for one arm: decode N_MC corpus draws
/// through the production entry, histogram the (alternating) outputs with
/// add-0.5 smoothing, and integrate the KL against the analytic law. Also
/// returns the total-variation distance and the fraction of samples that
/// left the alternating support entirely.
fn mc_realized_kl(
    arm: &Arm,
    config: &Config,
    weights: &TransformerWeights,
    budget: usize,
    n_mc: usize,
    base_seed: u64,
) -> (f64, f64, f64) {
    let a_size = 26usize;
    let pz = corpus_law();
    let mut counts = vec![0u64; a_size * a_size];
    let mut leak = 0u64;
    let mask = config.mask_token;
    let decode_config = D2fDecodeConfig {
        denoise_steps: budget,
        confidence_threshold: arm.kappa,
        block_size: BLOCK,
        ..D2fDecodeConfig::default()
    };

    for i in 0..n_mc {
        let mut rng = Rng::new(base_seed + i as u64);
        // The decode's only corpus input is the seed = seq[0] = a, and a's
        // marginal is uniform 1/26 (a is the first draw, never bumped). The
        // bump asymmetry lives entirely on the P side (`corpus_law`).
        let seed = (rng.next() as usize) % a_size;
        // The block decodes positions 1..=8.
        let mut ctx = ForwardContext::new(config);
        let mut cache = MultiLayerKVCache::new(config);
        let mut dctx = D2fContext::new(config);
        let tokens: Vec<usize> = match arm.kind {
            ArmKind::AllMask => unreachable!("T9 compares anchored arms only"),
            ArmKind::Stride { stride } => {
                let r = anchor_then_fill(
                    &mut ctx,
                    &mut cache,
                    &mut dctx,
                    weights,
                    config,
                    &decode_config,
                    &AnchorConfig::with_stride(stride),
                    seed,
                    0,
                    &mut rng,
                );
                r.tokens
            }
            ArmKind::Conf { floor } => {
                let r = anchor_then_fill_with(
                    &mut ctx,
                    &mut cache,
                    &mut dctx,
                    weights,
                    config,
                    &decode_config,
                    &ConfidenceAnchorConfig::new(arm.kappa, floor),
                    seed,
                    0,
                    &mut rng,
                );
                r.tokens
            }
        };
        // Map to the law: alternating, non-mask, v0 != v1. The parity walk
        // starts at p=2 (tokens[1] has no p−2 predecessor).
        let alternating = (2..BLOCK).all(|p| tokens[p] == tokens[p - 2]) && tokens[0] != tokens[1];
        if alternating && tokens[0] != mask && tokens[1] != mask {
            counts[tokens[0] * a_size + tokens[1]] += 1;
        } else {
            leak += 1;
        }
    }

    // Add-0.5 smoothed law estimate; KL + TV against the analytic P_Z.
    let total = n_mc as f64;
    let alpha = 0.5f64;
    let denom = total + alpha * pz.len() as f64;
    let (mut kl, mut tv) = (0.0f64, 0.0f64);
    for (p, &c) in pz.iter().zip(counts.iter()) {
        let q = (c as f64 + alpha) / denom;
        if *p > 0.0 {
            kl += p * (p / q).ln();
        }
        tv += (p - q).abs();
    }
    (kl, tv / 2.0, leak as f64 / total)
}

#[test]
fn t9_ugc_kl_cross_check() {
    let (config, weights) = make_nonsaturated_weights();

    // ── Certificate side: 4Ĉ/N for this model, at the NFE=8 budget ──
    // The g1-cert canonical-halves pattern (katgpt-core ugc_664_poc).
    let dz = D2fUgcDenoiser {
        weights: &weights,
        config: &config,
        bctx: RefCell::new(BidirectionalContext::new(&config)),
    };
    let mut rng = Rng::new(2026);
    let mut scratch = UgcScratch::new(BLOCK, config.vocab_size - 1, 24, 64);
    let est_lo = estimate_interval(
        &dz,
        1.0 / BLOCK as f32,
        0.5,
        24,
        0.05,
        &mut rng,
        &mut scratch,
    );
    let est_hi = estimate_interval(
        &dz,
        0.5,
        1.0 - 1.0 / BLOCK as f32,
        24,
        0.05,
        &mut rng,
        &mut scratch,
    );
    let plan = certified_block_plan(
        &[1.0 / BLOCK as f32, 0.5, 1.0 - 1.0 / BLOCK as f32],
        &[est_lo.upper, est_hi.upper],
        8,
    );
    let bound = 4.0 * plan.chat_partition_complexity as f64 / 8.0;
    println!(
        "T9 certificate: Ĉ = {:.4}, bound 4Ĉ/8 = {:.5} (random-order reveal reference — the greedy reveal is OUTSIDE its premise, Caveat #1)",
        plan.chat_partition_complexity, bound
    );

    // ── Realized side: MC the two production decode laws ──
    let budget = 8;
    let n_mc = 8192;
    let sm = Arm {
        label: "stride2 tau=kappa (matched ref)",
        kappa: 0.9,
        kind: ArmKind::Stride { stride: 2 },
    };
    let cf = Arm {
        label: "conf kappa + floor (DBTM)",
        kappa: 0.9,
        kind: ArmKind::Conf { floor: true },
    };

    let seeds = [500_000u64, 600_000];
    let mut kls_sm = Vec::with_capacity(seeds.len());
    let mut kls_cf = Vec::with_capacity(seeds.len());
    for &base in &seeds {
        let (kl_sm, tv_sm, leak_sm) = mc_realized_kl(&sm, &config, &weights, budget, n_mc, base);
        let (kl_cf, tv_cf, leak_cf) = mc_realized_kl(&cf, &config, &weights, budget, n_mc, base);
        println!(
            "T9 seed {base}: stride-matched KL={kl_sm:.5} TV={tv_sm:.5} leak={leak_sm:.3} | conf+floor KL={kl_cf:.5} TV={tv_cf:.5} leak={leak_cf:.3}"
        );
        kls_sm.push(kl_sm);
        kls_cf.push(kl_cf);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (m_sm, m_cf) = (mean(&kls_sm), mean(&kls_cf));
    let spread = (kls_cf.iter().map(|k| (k - m_cf).abs()).sum::<f64>() / kls_cf.len() as f64
        + kls_sm.iter().map(|k| (k - m_sm).abs()).sum::<f64>() / kls_sm.len() as f64)
        / 2.0;
    let ratio = m_cf / m_sm;
    println!(
        "T9 verdict inputs: mean KL conf+floor={m_cf:.5} vs stride-matched={m_sm:.5} (ratio {ratio:.3}, seed spread ±{spread:.5}) | certificate bound {bound:.5}"
    );

    // The gate: the confidence-greedy reveal must not distort the block law
    // more than the incumbent's fixed-order stride reveal (both outside the
    // certificate's random-order premise — Caveat #1 — so they race each
    // other, and the bound is context). Tolerance 10% + the measured seed
    // spread, so MC noise cannot flip the verdict.
    assert!(
        m_cf <= m_sm * 1.10 + spread,
        "T9 FAIL: confidence-commit realized KL {m_cf:.5} exceeds the incumbent's {m_sm:.5} by more than 10% + spread — the greedy reveal distorts the output law more than the incumbent (Caveat #1 bites); stays opt-in"
    );
}

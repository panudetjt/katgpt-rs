//! Plan 601 GOAT gates — the DBTM confidence-commit anchor rule (Issue 811 /
//! Plan 600) measured ON REAL TEXT, closing Plan 600's single remaining
//! promotion precondition.
//!
//! Plan 600's acceptance demanded "T8 all-green + T9 on real text" and its
//! 2026-09-17 resolution recorded that the repo's dllm lane was pattern-only —
//! `generate_pattern_dataset`'s analytic [a, b, a, b] family — so the bar
//! could not be met. This bench is the missing instrument:
//!
//! - **Corpus**: `src/dllm/text_corpus.rs`'s committed fixture (Jane Austen,
//!   *Pride and Prejudice* tail, Project Gutenberg eBook #1342, public
//!   domain), char-encoded onto a 31-symbol compact alphabet. Train and eval
//!   windows are disjoint slices of the same novel.
//! - **Model**: `Config::micro_dllm_text()` trained with the SAME
//!   `train_mini_dllm` the pattern lane uses (mask-corruption SGD) — a
//!   text-trained D2F, not a ported one.
//!
//! Gates (the Plan-600 gate list, re-measured on this corpus):
//!
//! - **Corpus honesty (T8 premise)**: the trained model's held-out masked NLL
//!   must sit measurably BELOW the empirical unigram entropy of the eval
//!   window — a model at unigram entropy learned the marginal and nothing
//!   else, and no decode-rule comparison over it would mean anything.
//! - **G1 quality non-inferiority**: conf+floor ≥ matched-threshold stride −
//!   2·SE(paired Δ) at every κ, NFE=8, exact-match over 512 held-out blocks;
//!   paired seeds (identical rng stream per sequence across arms — the walk
//!   is byte-identical, only the rule differs); resolution asserted
//!   (SE(Δ) ≤ 0.05).
//! - **G2 steps + wall**: DBTM floor beats (κ ∈ {0.9, 0.99}) or ties (κ = 0.5)
//!   matched-stride fill steps, wall ≤ 1.25×.
//! - **Liveness**: all-mask baseline measurably below the anchored arms (the
//!   anchor round carries signal on text, not just on the pattern family).
//! - **T9′ realized-KL cross-check (Caveat #1, real-text form)**: the
//!   analytic pattern law does not exist here, so `P_Z` is estimated
//!   nonparametrically — the add-0.5-smoothed empirical bigram joint
//!   P(c1, c2) from held-out corpus counts (the same smoothing the pattern
//!   lane's T9 uses, so the two estimates are method-identical). Realized
//!   `KL(P_Z` ‖ P̂) is Monte-Carlo measured for both production decode rules
//!   over `n_mc` corpus draws; the model's UGC certificate 4Ĉ/N (random-order
//!   reveal reference — the greedy reveal is OUTSIDE its premise) is printed
//!   alongside. PASS = the confidence path's realized KL within 10% of the
//!   incumbent's.
//!
//! G3 (incumbent byte-identity with the config off) and G4 (alloc-free fill
//! path) are corpus-INDEPENDENT properties of the seam: G3 is pinned by the
//! parity tests in `src/speculative/flashar_anchor.rs`
//! (`test_issue811_harness_matches_production`,
//! `test_issue600_entry_matches_poc_conf_arm`), G4 by
//! `tests/bench_600_flashar_fill_alloc_gate.rs` — both run in this same
//! feature lane; this file adds no second pin.
//!
//! # Run
//!
//! ```bash
//! cargo test --release --features flashar_anchor --test bench_601_flashar_realtext_goat -- --nocapture
//! ```
//!
//! RELEASE is the measurement posture: the text model's training loop is
//! ~80k sample updates, and a debug-profile wall number measures an
//! unoptimised binary (the armed-gates rule). Env overrides for scale
//! (`BENCH601_EPOCHS`, `BENCH601_MC`) exist for CI tuning; the GATE
//! thresholds do not move with them.

#![cfg(feature = "flashar_anchor")]

use katgpt_core::ugc_schedule::{
    UGC_MASK, UgcDenoiser, UgcScratch, certified_block_plan, estimate_interval,
};
use katgpt_core::{Config, Rng};
use katgpt_forward::d2f::{D2fBlockResult, D2fDecodeConfig};
use katgpt_forward::{
    AnchorConfig, BidirectionalContext, ConfidenceAnchorConfig, D2fContext, ForwardContext,
    anchor_fill_with_prefilled, anchor_then_fill, anchor_then_fill_with,
};
use katgpt_rs::dllm::text_corpus::{
    TEXT_ALPHABET, TEXT_CORPUS, bigram_counts, encode_text, slice_blocks, unigram_entropy_nats,
};
use katgpt_rs::dllm::{evaluate_accuracy, evaluate_masked_nll, train_mini_dllm};
use katgpt_rs::transformer::{MultiLayerKVCache, TransformerWeights};
use std::cell::RefCell;
use std::time::Instant;

const BLOCK: usize = 8;
const N_EVAL: usize = 512;
const TRAIN_BLOCKS: usize = 2048;
const EVAL_WINDOW_START: usize = 80_000; // disjoint from the train window 0..16_384
const TRAIN_EPOCHS_DEFAULT: usize = 40;
const MC_DEFAULT: usize = 4096;
const KAPPAS: [f32; 3] = [0.5, 0.9, 0.99];
const NFE_CELLS: [usize; 2] = [4, 8];

fn train_epochs() -> usize {
    std::env::var("BENCH601_EPOCHS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TRAIN_EPOCHS_DEFAULT)
}

fn n_mc() -> usize {
    std::env::var("BENCH601_MC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MC_DEFAULT)
}

/// Independent MC base seeds for the realized-KL mean (the T9 variance read).
const KL_SEEDS: [u64; 2] = [500_000, 600_000];

fn seed_count() -> usize {
    KL_SEEDS.len()
}

/// The text-trained D2F + the disjoint corpus windows, deterministic end to
/// end (fixed train seed, fixed window offsets). Windows are `BLOCK + 1`
/// tokens: `seq[0]` is the decode's seed, `seq[1..=BLOCK]` the ground truth
/// the block slots are scored against (the bench-600 protocol, where the
/// 16-token pattern seqs carried the same headroom).
fn train_text_model() -> (Config, TransformerWeights, Vec<Vec<usize>>) {
    let config = Config::micro_dllm_text();
    let tokens = encode_text(TEXT_CORPUS);
    let train = slice_blocks(&tokens, 0, TRAIN_BLOCKS, BLOCK + 1);
    let eval = slice_blocks(&tokens, EVAL_WINDOW_START, N_EVAL, BLOCK + 1);
    let (weights, _) = train_mini_dllm(&config, &train, &eval, train_epochs(), 0.01, 0.3, 42);
    (config, weights, eval)
}

// ── Arms — production entries only (identical protocol to bench 600) ────

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
    let mut per_seq_acc = Vec::with_capacity(eval.len());

    for (si, seq) in eval.iter().enumerate() {
        // Paired runs: identical rng stream per sequence across arms.
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

        // Ground truth: the block decodes positions 1..=BLOCK given seed
        // seq[0], so slot p expects the corpus char seq[p+1].
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

// ── T9′ pieces ──────────────────────────────────────────────────────────

/// `UgcDenoiser` adapter over the text-trained bidirectional D2F (same shape
/// as bench 600's — the certificate is model-side and transfers unchanged).
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
        katgpt_forward::forward_bidirectional_positions_into(
            self.weights,
            &block,
            self.config,
            &mut bctx,
        );
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

/// Empirical unigram law (add-0.5) as a cumulative table for sampling the
/// decode's seed char c1 from the corpus marginal — the law's own marginal,
/// so the KL integrates over draws `P_Z` actually produces.
struct UnigramCdf {
    cdf: Vec<u64>,
    total: u64,
}

fn unigram_cdf(tokens: &[usize]) -> UnigramCdf {
    let mut counts = vec![0u64; TEXT_ALPHABET];
    for &t in tokens {
        counts[t] += 1;
    }
    let mut cdf = Vec::with_capacity(TEXT_ALPHABET);
    let mut run = 0u64;
    for &c in &counts {
        run += c + 1; // +1 = add-0.5 smoothing baked into the sampler
        cdf.push(run);
    }
    UnigramCdf { cdf, total: run }
}

impl UnigramCdf {
    fn sample(&self, rng: &mut Rng) -> usize {
        let r = rng.next() % self.total;
        match self.cdf.binary_search(&r) {
            Ok(i) => i,
            Err(i) => i.min(TEXT_ALPHABET - 1),
        }
    }
}

/// Monte-Carlo realized `KL(P_Z` ‖ P̂) for one arm against the EMPIRICAL
/// bigram joint law P(c1, c2) estimated from held-out corpus counts
/// (add-0.5 smoothing over the full table — method-identical to bench 600's
/// analytic-law KL). `tokens` is the held-out token stream the law is
/// estimated from.
#[allow(clippy::too_many_lines)]
fn mc_realized_kl(
    arm: &Arm,
    config: &Config,
    weights: &TransformerWeights,
    law_tokens: &[usize],
    budget: usize,
    n_mc: usize,
    base_seed: u64,
) -> (f64, f64, f64) {
    // P side: empirical bigram joint from the held-out stream.
    let counts = bigram_counts(law_tokens, TEXT_ALPHABET);
    let total: u64 = counts.iter().sum();
    let alpha = 0.5f64;
    let support = (TEXT_ALPHABET * TEXT_ALPHABET) as f64;
    let pz_denom = total as f64 + alpha * support;
    let pz: Vec<f64> = counts
        .iter()
        .map(|&c| (c as f64 + alpha) / pz_denom)
        .collect();

    let cdf = unigram_cdf(law_tokens);
    let mut q_counts = vec![0u64; TEXT_ALPHABET * TEXT_ALPHABET];
    let mut leak = 0u64;
    let decode_config = D2fDecodeConfig {
        denoise_steps: budget,
        confidence_threshold: arm.kappa,
        block_size: BLOCK,
        ..D2fDecodeConfig::default()
    };

    for i in 0..n_mc {
        let mut rng = Rng::new(base_seed + i as u64);
        let seed = cdf.sample(&mut rng);
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
        // tokens[0] is the corpus char AFTER the seed → the (c1, c2) bigram.
        let c2 = tokens[0];
        if c2 < TEXT_ALPHABET {
            q_counts[seed * TEXT_ALPHABET + c2] += 1;
        } else {
            leak += 1; // the decode emitted the mask at the first slot
        }
    }

    // Q side: add-0.5 smoothed histogram; KL + TV against P_Z.
    let q_denom = n_mc as f64 + alpha * support;
    let (mut kl, mut tv) = (0.0f64, 0.0f64);
    for (p, &c) in pz.iter().zip(q_counts.iter()) {
        let q = (c as f64 + alpha) / q_denom;
        if *p > 0.0 {
            kl += p * (p / q).ln();
        }
        tv += (p - q).abs();
    }
    (kl, tv / 2.0, leak as f64 / n_mc as f64)
}

// ── Gates ───────────────────────────────────────────────────────────────

#[test]
fn t1_corpus_and_model_honesty() {
    let tokens = encode_text(TEXT_CORPUS);
    let (config, weights, eval) = train_text_model();

    // Corpus honesty: held-out masked NLL must beat the empirical unigram
    // entropy of the eval window by a real margin. A model at unigram
    // entropy has learned the marginal and nothing else.
    let eval_tokens = &tokens[EVAL_WINDOW_START..EVAL_WINDOW_START + N_EVAL * (BLOCK + 1)];
    let uni_h = unigram_entropy_nats(eval_tokens, TEXT_ALPHABET);
    let mut rng = Rng::new(2026_0917);
    let nll = f64::from(evaluate_masked_nll(&weights, &eval, &config, 0.3, &mut rng));
    println!(
        "T1 corpus honesty: held-out masked NLL = {nll:.4} nats/token vs empirical unigram entropy = {uni_h:.4} nats/token (margin {:.4})",
        uni_h - nll
    );
    assert!(
        nll < uni_h - 0.15,
        "corpus honesty FAIL: model NLL {nll:.4} not measurably below unigram entropy {uni_h:.4} — the text model learned nothing beyond the marginal; the decode gates would measure noise"
    );

    // Token-level sanity over the SAME eval set: argmax accuracy exists
    // (above the 1/31 chance line) — the decode gates have a non-degenerate
    // target. `evaluate_accuracy` is the pattern lane's own instrument.
    let mut acc_rng = Rng::new(555);
    let acc = evaluate_accuracy(&weights, &eval, &config, 0.3, &mut acc_rng);
    println!(
        "T1 held-out masked-token accuracy = {acc:.3} (chance = {:.3})",
        1.0 / TEXT_ALPHABET as f32
    );
    assert!(
        acc > 3.0 / TEXT_ALPHABET as f32,
        "text model accuracy {acc:.3} near chance — training failed"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn t2_realtext_g1_g2_goat() {
    let (config, weights, eval) = train_text_model();
    let arms = build_arms();

    println!(
        "\n== Plan 601 arm table (real text: Austen tail, {} train blocks × {} epochs, {} held-out blocks, block={BLOCK}) ==",
        TRAIN_BLOCKS,
        train_epochs(),
        N_EVAL
    );
    println!(
        "{:<32} {:>6} {:>4} {:>7} {:>7} {:>8} {:>7} {:>9}",
        "arm", "kappa", "NFE", "acc", "se", "steps", "anchors", "wall_us"
    );

    let mut outcomes: Vec<Vec<ArmOutcome>> = Vec::with_capacity(arms.len());
    for arm in &arms {
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

    // ── Liveness: anchors must carry signal on real text ──
    let am = find("all-mask D2F baseline", 0.7);
    let sm09 = find("stride2 tau=kappa (matched ref)", 0.9);
    let gap_se = am.se_acc.hypot(sm09.se_acc);
    assert!(
        sm09.mean_acc - am.mean_acc > 2.0 * gap_se,
        "corpus liveness FAIL: anchored arm ({:.3}) not measurably above all-mask ({:.3}) — the anchor round carries no signal on this text model",
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
            "paired resolution {se_d:.4} > 0.05 at κ={k} — the corpus cannot detect a 2σ quality difference"
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

#[test]
fn t3_realtext_ugc_kl_cross_check() {
    let (config, weights, _eval) = train_text_model();

    // Certificate side: 4Ĉ/N for the TEXT model (random-order reveal
    // reference — the greedy reveal is OUTSIDE its premise, Caveat #1).
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
    println!(
        "T9 certificate (text model): Ĉ = {:.4}, bound 4Ĉ/8 = {:.5} (random-order reference — the greedy reveal is OUTSIDE its premise)",
        plan.chat_partition_complexity,
        4.0 * plan.chat_partition_complexity as f64 / 8.0
    );

    // Realized side: the law is the EMPIRICAL bigram joint of a held-out
    // corpus stretch. Gate at κ=0.9 (the bench-600 T9 cell, for continuity);
    // κ=0.5 is printed as context — the cell where both rules terminate
    // cleanly, so the KL comparison is between two LIVE laws, not between a
    // live law and an incumbent that left the slot masked.
    let tokens = encode_text(TEXT_CORPUS);
    let law_tokens = &tokens[EVAL_WINDOW_START..EVAL_WINDOW_START + 16_384];

    let budget = 8;
    let n_mc = n_mc();
    let arms_at = |k: f32| {
        [
            Arm {
                label: "stride2 tau=kappa (matched ref)",
                kappa: k,
                kind: ArmKind::Stride { stride: 2 },
            },
            Arm {
                label: "conf kappa + floor (DBTM)",
                kappa: k,
                kind: ArmKind::Conf { floor: true },
            },
        ]
    };

    for k in KAPPAS {
        let [sm, cf] = arms_at(k);
        let mut kls_sm = Vec::with_capacity(seed_count());
        let mut kls_cf = Vec::with_capacity(seed_count());
        for &base in &KL_SEEDS {
            let (kl_sm, tv_sm, leak_sm) =
                mc_realized_kl(&sm, &config, &weights, law_tokens, budget, n_mc, base);
            let (kl_cf, tv_cf, leak_cf) =
                mc_realized_kl(&cf, &config, &weights, law_tokens, budget, n_mc, base);
            println!(
                "T9 κ={k} seed {base}: stride-matched KL={kl_sm:.5} TV={tv_sm:.5} leak={leak_sm:.3} | conf+floor KL={kl_cf:.5} TV={tv_cf:.5} leak={leak_cf:.3}"
            );
            kls_sm.push(kl_sm);
            kls_cf.push(kl_cf);
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (m_sm, m_cf) = (mean(&kls_sm), mean(&kls_cf));
        let ratio = m_cf / m_sm;
        println!(
            "T9 κ={k} mean over {} seeds: stride-matched KL={m_sm:.5} | conf+floor KL={m_cf:.5} | ratio {ratio:.3}",
            seed_count()
        );
        if k == 0.9 {
            assert!(
                ratio <= 1.10,
                "T9 FAIL: the greedy confidence reveal distorts the REAL-TEXT block law MORE than the strided incumbent ({ratio:.3} > 1.10) — Caveat #1 is live on text"
            );
        }
    }
}

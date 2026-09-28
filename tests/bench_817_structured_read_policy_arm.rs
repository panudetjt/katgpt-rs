//! Bench 817 — Issue 859 T5: the entropy-gated re-read policy arm (M3),
//! measured on the `micro_dllm_text` trained fixture.
//!
//! The T5 question (the issue's own wording): **do agreement bars beat
//! single-read entropy as a confidence proxy?** The reference protocol
//! (Research 574 §8) fires reads=4 where H1 > 0.1 and reports p(want) ±
//! stderr + agreement as first-class outputs. This bench instantiates that
//! protocol on OUR primitive (`structured_read` + `sample_label_index`,
//! temperature 1, N=4 — the reference's numbers) against the trained
//! text fixture, with GROUND TRUTH from the held-out Austen eval window.
//!
//! # Pre-registered prediction (the defend-wrong shape)
//!
//! On a deterministic forward, correctness is a DETERMINISTIC function of
//! the readout (`argmax == truth`), and `sample_label_index` draws are
//! conditionally independent of correctness given the readout. Agreement is
//! therefore a noisy finite-sample estimator of functionals the readout
//! already carries exactly (`argmax_label_prob`, `label_entropy`):
//!
//! - predicted: AUROC(agreement) < AUROC(maxprob) ≈ AUROC(−H1), with the
//!   gap shrinking as N grows — the M3 gate buys error BARS, not
//!   discrimination;
//! - genuinely open (and measured): entropy vs maxprob as the confidence
//!   functional — they rank differently on multi-modal subsets;
//! - the sampled-answer deployment regime (the deployed answer IS the
//!   temperature-1 sample): sampled accuracy + whether maxprob predicts
//!   sampled correctness.
//!
//! A prediction VIOLATION (agreement beating both analytic signals with a
//! CI excluding 0) falsifies the control-arm reading and would mean the
//! re-read machinery carries information beyond the readout — the case the
//! reference's designers argue for. Either result closes T5 honestly.
//!
//! # Protocol
//!
//! - Fixture: bench-601's exact training recipe — `Config::micro_dllm_text()`,
//!   2048 train / 512 held-out eval blocks (windows disjoint), 40 epochs,
//!   lr 0.01, mask ratio 0.3, seed 42. Corpus honesty asserted (held-out
//!   masked NLL < eval-window unigram entropy — a model at the unigram
//!   floor learned nothing the marginals don't contain).
//! - Canvas shapes: S1 = one masked position (9 per block, p ∈ 0..=8) and
//!   S3 = three contiguous masked positions (starts {0,3,6} — 33% masked,
//!   exactly the training corruption density). Both in-distribution.
//! - Label arms: A = 4-option per slot (truth + 3 uniform distractors,
//!   seeded — the corpora-mirror shape; S3 canvases read over the UNION
//!   option set, 3..=12 options, exactly a multi-field decision form);
//!   B = full 31-token text alphabet (the free-decode shape).
//! - Signals per item: H1 (`label_entropy`), maxprob (`argmax_label_prob`),
//!   agreement (fraction of the 4 samples equal to the first-read argmax),
//!   correctness (argmax label == held-out truth), sampled correctness.
//! - Metrics: midrank AUROC (tie-safe — agreement has 5 levels); paired
//!   bootstrap over items (5k reps, seeded) for ΔAUROC CIs; selective
//!   accuracy at 80/90% coverage; the M3 gate table at τ ∈ {0.05, 0.1
//!   (the reference's), 0.2, 0.4}.
//!
//! This is a MEASUREMENT bench, not a gate: no threshold asserts on the
//! AUROC comparison (that is the research question, recorded either way in
//! Bench 817 / Research 574). Non-vacuity IS asserted (item counts, both
//! correctness classes present, entropy within the label-count ceiling).
//!
//! # Run
//!
//! ```bash
//! cargo test --release --features structured_reads --test bench_817_structured_read_policy_arm -- --nocapture
//! ```
//!
//! RELEASE is the measurement posture (the training loop is ~80k updates).
//! `BENCH817_EPOCHS` scales training for smoke runs (assertions that depend
//! on a trained model are relaxed to printed observations below 10 epochs).

#![cfg(feature = "structured_reads")]

use katgpt_core::{Config, Rng};
use katgpt_forward::structured_read::{
    MAX_LABELS, SlotReadout, StructuredReadScratch, sample_label_index, structured_read_into,
};
use katgpt_rs::dllm::text_corpus::{
    TEXT_ALPHABET, TEXT_CORPUS, encode_text, slice_blocks, unigram_entropy_nats,
};
use katgpt_rs::dllm::{evaluate_masked_nll, train_mini_dllm};
use katgpt_rs::transformer::TransformerWeights;

const BLOCK: usize = 8;
const N_EVAL: usize = 512;
const TRAIN_BLOCKS: usize = 2048;
const EVAL_WINDOW_START: usize = 80_000;
const TRAIN_EPOCHS_DEFAULT: usize = 40;
const N_REREADS: usize = 4;
const TEMPERATURE: f32 = 1.0;
const BOOTSTRAP_REPS: usize = 5_000;
const TAUS: [f32; 4] = [0.05, 0.1, 0.2, 0.4];
const COVERAGES: [f64; 2] = [0.8, 0.9];
const SEED_TRAIN: u64 = 42;
const SEED_DISTRACT: u64 = 7_000_000;
const SEED_SAMPLE: u64 = 9_000_000;
const SEED_BOOT: u64 = 11_000_000;
const N_OPTIONS_ARM_A: usize = 4;
const S3_STARTS: [usize; 3] = [0, 3, 6];

fn train_epochs() -> usize {
    std::env::var("BENCH817_EPOCHS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TRAIN_EPOCHS_DEFAULT)
}

/// One decision item's measured row.
#[derive(Clone, Copy)]
struct Item {
    h1: f32,
    maxprob: f32,
    agreement: f32,
    correct: bool,
    sampled_correct_mean: f32,
}

struct ArmRows {
    name: &'static str,
    /// Entropy ceiling for the non-vacuity assert: ln(max options in this arm).
    ln_labels_ceiling: f64,
    items: Vec<Item>,
    s1_rows: (usize, usize), // (n, n_correct) for the S1 canvas shape
    s3_rows: (usize, usize),
}

fn train_text_model() -> (Config, TransformerWeights, Vec<Vec<usize>>) {
    let config = Config::micro_dllm_text();
    let tokens = encode_text(TEXT_CORPUS);
    let train = slice_blocks(&tokens, 0, TRAIN_BLOCKS, BLOCK + 1);
    let eval = slice_blocks(&tokens, EVAL_WINDOW_START, N_EVAL, BLOCK + 1);
    let (weights, _) = train_mini_dllm(
        &config,
        &train,
        &eval,
        train_epochs(),
        0.01,
        0.3,
        SEED_TRAIN,
    );
    (config, weights, eval)
}

/// One slot's option set: truth + 3 uniform distractors over the text
/// alphabet (0..=30; mask token 31 excluded), strictly ascending.
fn option_set(truth: usize, rng: &mut Rng) -> Vec<u32> {
    let mut set = vec![truth as u32];
    while set.len() < N_OPTIONS_ARM_A {
        let cand = (rng.next() % TEXT_ALPHABET as u64) as u32;
        if !set.contains(&cand) {
            set.push(cand);
        }
    }
    set.sort_unstable();
    set
}

/// The union option set for an S3 canvas: each masked slot's truth + its 3
/// distractors, unioned + ascending (3..=12 options — the multi-field
/// decision form; one read call covers all slots).
fn union_option_set(truths: &[usize], rng: &mut Rng) -> Vec<u32> {
    let mut set: Vec<u32> = Vec::with_capacity(N_OPTIONS_ARM_A * truths.len());
    for &t in truths {
        set.extend(option_set(t, rng));
    }
    set.sort_unstable();
    set.dedup();
    set
}

/// The full text alphabet (arm B), ascending.
fn alphabet_labels() -> Vec<u32> {
    (0..TEXT_ALPHABET as u32).collect()
}

/// Read one canvas against one label arm: first read + N stochastic
/// re-reads per free slot. Returns one item per free slot.
fn measure_canvas(
    weights: &TransformerWeights,
    config: &Config,
    canvas: &[usize],
    ground_truth: &[usize],
    label_ids: &[u32],
    scratch: &mut StructuredReadScratch,
    sample_rng: &mut Rng,
) -> Vec<Item> {
    let n_free = canvas.iter().filter(|&&t| t == config.mask_token).count();
    let mut out = vec![
        SlotReadout {
            argmax_logprob: 0.0,
            argmax_label_prob: 0.0,
            label_entropy: 0.0,
            vocab_argmax_token: 0,
            position: 0,
            argmax_index: 0,
            n_labels: 0,
            label_logprobs: [0.0; MAX_LABELS],
        };
        n_free.max(1)
    ];
    let n = structured_read_into(&mut out, weights, config, canvas, label_ids, scratch)
        .expect("structured_read_into");
    out.truncate(n);

    let mut items = Vec::with_capacity(n);
    for r in &out {
        let truth_tok = ground_truth[r.position as usize] as u32;
        let correct = label_ids[r.argmax_index as usize] == truth_tok;
        let mut agree = 0usize;
        let mut sampled_correct = 0usize;
        for _ in 0..N_REREADS {
            let s = sample_label_index(r, TEMPERATURE, sample_rng);
            if s == r.argmax_index {
                agree += 1;
            }
            if label_ids[s as usize] == truth_tok {
                sampled_correct += 1;
            }
        }
        items.push(Item {
            h1: r.label_entropy,
            maxprob: r.argmax_label_prob,
            agreement: agree as f32 / N_REREADS as f32,
            correct,
            sampled_correct_mean: sampled_correct as f32 / N_REREADS as f32,
        });
    }
    items
}

// ── Statistics ───────────────────────────────────────────────────────────

/// Midrank AUROC of `values` for predicting `correct` (higher => more likely
/// correct). Tie-safe (agreement has 5 levels; entropy/maxprob are
/// continuous). Returns NaN when a class is empty.
fn auroc_midrank(values: &[f32], correct: &[bool]) -> f64 {
    let n = values.len();
    debug_assert_eq!(n, correct.len());
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        let mid = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            ranks[k] = mid;
        }
        i = j + 1;
    }
    let n_pos = correct.iter().filter(|&&c| c).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return f64::NAN;
    }
    let sum_pos: f64 = (0..n).filter(|&k| correct[k]).map(|k| ranks[k]).sum();
    (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// Accuracy of the top-`coverage` fraction ranked by `conf` (higher first).
fn selective_accuracy(conf: &[f32], correct: &[bool], coverage: f64) -> f64 {
    let n = conf.len();
    let k = ((coverage * n as f64).ceil() as usize).clamp(1, n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        conf[b]
            .partial_cmp(&conf[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let hits = idx[..k].iter().filter(|&&i| correct[i]).count();
    hits as f64 / k as f64
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let m = sorted.len() as f64 - 1.0;
    let pos = p * m;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
    }
}

/// Paired bootstrap over items for the three AUROC signals; returns
/// (n_valid, Δ agreement−negH1, Δ agreement−maxprob, Δ maxprob−negH1), each
/// a 95% CI [lo, hi].
#[allow(clippy::type_complexity)]
fn bootstrap_deltas(rows: &ArmRows) -> (usize, [f64; 2], [f64; 2], [f64; 2]) {
    let n = rows.items.len();
    let neg_h1: Vec<f32> = rows.items.iter().map(|it| -it.h1).collect();
    let maxp: Vec<f32> = rows.items.iter().map(|it| it.maxprob).collect();
    let agree: Vec<f32> = rows.items.iter().map(|it| it.agreement).collect();
    let correct: Vec<bool> = rows.items.iter().map(|it| it.correct).collect();

    let mut rng = Rng::new(SEED_BOOT);
    let mut d_agree_h1 = Vec::with_capacity(BOOTSTRAP_REPS);
    let mut d_agree_maxp = Vec::with_capacity(BOOTSTRAP_REPS);
    let mut d_maxp_h1 = Vec::with_capacity(BOOTSTRAP_REPS);
    let mut idx_buf: Vec<usize> = vec![0; n];
    let (mut v_buf, mut c_buf) = (vec![0.0f32; n], vec![false; n]);
    for _ in 0..BOOTSTRAP_REPS {
        for slot in idx_buf.iter_mut() {
            *slot = (rng.next() as usize) % n;
        }
        let mut any_nan = false;
        let mut aucs = [0.0f64; 3];
        for (sig, auc) in [&neg_h1, &maxp, &agree].into_iter().zip(aucs.iter_mut()) {
            for (i, &j) in idx_buf.iter().enumerate() {
                v_buf[i] = sig[j];
                c_buf[i] = correct[j];
            }
            *auc = auroc_midrank(&v_buf, &c_buf);
            if auc.is_nan() {
                any_nan = true;
                break;
            }
        }
        if any_nan {
            continue; // degenerate resample (single class) — skipped, counted
        }
        d_agree_h1.push(aucs[2] - aucs[0]);
        d_agree_maxp.push(aucs[2] - aucs[1]);
        d_maxp_h1.push(aucs[1] - aucs[0]);
    }
    let n_valid = d_agree_h1.len();
    for v in [&mut d_agree_h1, &mut d_agree_maxp, &mut d_maxp_h1] {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    (
        n_valid,
        [
            percentile(&d_agree_h1, 0.025),
            percentile(&d_agree_h1, 0.975),
        ],
        [
            percentile(&d_agree_maxp, 0.025),
            percentile(&d_agree_maxp, 0.975),
        ],
        [percentile(&d_maxp_h1, 0.025), percentile(&d_maxp_h1, 0.975)],
    )
}

// ── The measurement ──────────────────────────────────────────────────────

#[test]
fn t5_policy_arm_measurement() {
    let epochs = train_epochs();
    let started = std::time::Instant::now();
    let (config, weights, eval) = train_text_model();
    println!(
        "[model] trained {TRAIN_BLOCKS} blocks × {epochs} epochs in {:.1}s (seed {SEED_TRAIN})",
        started.elapsed().as_secs_f32()
    );

    // Corpus honesty (bench-601 T8 premise): below the unigram floor or the
    // model learned only marginals. Asserted at full training; printed (not
    // asserted) on smoke runs.
    let mut rng = Rng::new(SEED_TRAIN + 1);
    let nll = f64::from(evaluate_masked_nll(&weights, &eval, &config, 0.3, &mut rng));
    let eval_flat: Vec<usize> = eval.iter().flatten().copied().collect();
    let uni = unigram_entropy_nats(&eval_flat, TEXT_ALPHABET);
    println!("[model] held-out masked NLL {nll:.4} vs unigram entropy {uni:.4} nats");
    if epochs >= 10 {
        assert!(
            nll < uni,
            "fixture honesty: masked NLL {nll:.4} must sit below unigram entropy {uni:.4}"
        );
    }

    let mut scratch = StructuredReadScratch::new(&config);
    let mut distract_rng = Rng::new(SEED_DISTRACT);
    let mut sample_rng = Rng::new(SEED_SAMPLE);
    let mask = config.mask_token;

    let mut arm_a = ArmRows {
        name: "A option-set (corpora mirror; S3 = union set)",
        ln_labels_ceiling: (N_OPTIONS_ARM_A * S3_STARTS.len()) as f64,
        items: Vec::new(),
        s1_rows: (0, 0),
        s3_rows: (0, 0),
    };
    let mut arm_b = ArmRows {
        name: "B full-alphabet (31 options)",
        ln_labels_ceiling: TEXT_ALPHABET as f64,
        items: Vec::new(),
        s1_rows: (0, 0),
        s3_rows: (0, 0),
    };

    for block in &eval {
        // S1: one masked position per canvas, both label arms per canvas.
        for p in 0..=BLOCK {
            let mut canvas = block.clone();
            canvas[p] = mask;
            let a_labels = option_set(block[p], &mut distract_rng);
            for (rows, labels) in [(&mut arm_a, &a_labels), (&mut arm_b, &alphabet_labels())] {
                let got = measure_canvas(
                    &weights,
                    &config,
                    &canvas,
                    block,
                    labels,
                    &mut scratch,
                    &mut sample_rng,
                );
                rows.s1_rows.0 += got.len();
                rows.s1_rows.1 += got.iter().filter(|it| it.correct).count();
                rows.items.extend(got);
            }
        }
        // S3: three contiguous masked positions — arm A reads over the
        // union option set, arm B over the alphabet; one call each.
        for &s in &S3_STARTS {
            let mut canvas = block.clone();
            for c in &mut canvas[s..s + 3] {
                *c = mask;
            }
            let truths: Vec<usize> = (s..s + 3).map(|q| block[q]).collect();
            let a_labels = union_option_set(&truths, &mut distract_rng);
            for (rows, labels) in [(&mut arm_a, &a_labels), (&mut arm_b, &alphabet_labels())] {
                let got = measure_canvas(
                    &weights,
                    &config,
                    &canvas,
                    block,
                    labels,
                    &mut scratch,
                    &mut sample_rng,
                );
                rows.s3_rows.0 += got.len();
                rows.s3_rows.1 += got.iter().filter(|it| it.correct).count();
                rows.items.extend(got);
            }
        }
    }

    for rows in [&arm_a, &arm_b] {
        report_arm(rows);
    }

    // Non-vacuity (the frontier-report law): the counts and class presence
    // must hold at ANY epoch count, or the instrument is measuring nothing.
    let expected = N_EVAL * ((BLOCK + 1) + S3_STARTS.len() * 3);
    for rows in [&arm_a, &arm_b] {
        assert_eq!(rows.items.len(), expected, "item count in {}", rows.name);
        assert!(
            rows.items.iter().any(|it| it.correct),
            "no correct item in {}",
            rows.name
        );
        assert!(
            !rows.items.iter().all(|it| it.correct),
            "no incorrect item in {} — no discriminability question",
            rows.name
        );
        let max_h = rows.items.iter().map(|it| it.h1).fold(0.0f32, f32::max) as f64;
        let ceiling = rows.ln_labels_ceiling.ln() + 1e-3;
        assert!(
            max_h <= ceiling,
            "entropy {max_h:.4} exceeds ln(max options) {ceiling:.4} in {}",
            rows.name
        );
    }
}

fn report_arm(rows: &ArmRows) {
    let n = rows.items.len();
    let n_correct = rows.items.iter().filter(|it| it.correct).count();
    let neg_h1: Vec<f32> = rows.items.iter().map(|it| -it.h1).collect();
    let maxp: Vec<f32> = rows.items.iter().map(|it| it.maxprob).collect();
    let agree: Vec<f32> = rows.items.iter().map(|it| it.agreement).collect();
    let correct: Vec<bool> = rows.items.iter().map(|it| it.correct).collect();

    let au_h1 = auroc_midrank(&neg_h1, &correct);
    let au_maxp = auroc_midrank(&maxp, &correct);
    let au_agree = auroc_midrank(&agree, &correct);
    let mean_h1: f64 = rows.items.iter().map(|it| f64::from(it.h1)).sum::<f64>() / n as f64;
    let sampled_acc: f64 = rows
        .items
        .iter()
        .map(|it| f64::from(it.sampled_correct_mean))
        .sum::<f64>()
        / n as f64;

    println!("\n== arm {} — {n} items ==", rows.name);
    println!(
        "  shapes: S1 {}/{} correct ({:.1}%) · S3 {}/{} correct ({:.1}%)",
        rows.s1_rows.1,
        rows.s1_rows.0,
        100.0 * rows.s1_rows.1 as f64 / rows.s1_rows.0 as f64,
        rows.s3_rows.1,
        rows.s3_rows.0,
        100.0 * rows.s3_rows.1 as f64 / rows.s3_rows.0 as f64,
    );
    println!(
        "  argmax acc {n_correct}/{n} = {:.1}% · sampled(t=1) acc {:.1}% · mean H1 {mean_h1:.4} nats",
        100.0 * n_correct as f64 / n as f64,
        100.0 * sampled_acc,
    );
    println!(
        "  AUROC(−H1) {au_h1:.4} · AUROC(maxprob) {au_maxp:.4} · AUROC(agreement) {au_agree:.4}"
    );

    let (n_valid, d_ah, d_am, d_mh) = bootstrap_deltas(rows);
    println!(
        "  bootstrap {n_valid}/{} valid · Δ(agree−(−H1)) [{:+.4}, {:+.4}] · Δ(agree−maxp) [{:+.4}, {:+.4}] · Δ(maxp−(−H1)) [{:+.4}, {:+.4}] (95%)",
        BOOTSTRAP_REPS, d_ah[0], d_ah[1], d_am[0], d_am[1], d_mh[0], d_mh[1],
    );

    for &cov in &COVERAGES {
        println!(
            "  selective acc @{}%: −H1 {:.1}% · maxprob {:.1}% · agreement {:.1}%",
            100.0 * cov,
            100.0 * selective_accuracy(&neg_h1, &correct, cov),
            100.0 * selective_accuracy(&maxp, &correct, cov),
            100.0 * selective_accuracy(&agree, &correct, cov),
        );
    }

    // The M3 gate table: τ | gate rate | mean reads | on-gate-subset AUROCs.
    println!(
        "  M3 gate (re-reads fire where H1 > τ; cost 1 + {} × gate rate):",
        N_REREADS - 1
    );
    for &tau in &TAUS {
        let gated: Vec<usize> = (0..n).filter(|&i| rows.items[i].h1 > tau).collect();
        if gated.is_empty() {
            println!("    τ={tau:.2}: gate 0.0% — no items");
            continue;
        }
        let g_h1: Vec<f32> = gated.iter().map(|&i| neg_h1[i]).collect();
        let g_maxp: Vec<f32> = gated.iter().map(|&i| maxp[i]).collect();
        let g_agree: Vec<f32> = gated.iter().map(|&i| agree[i]).collect();
        let g_correct: Vec<bool> = gated.iter().map(|&i| correct[i]).collect();
        let g_acc = 100.0 * g_correct.iter().filter(|&&c| c).count() as f64 / gated.len() as f64;
        let a1 = auroc_midrank(&g_h1, &g_correct);
        let a2 = auroc_midrank(&g_maxp, &g_correct);
        let a3 = auroc_midrank(&g_agree, &g_correct);
        let mean_reads = 1.0 + (N_REREADS - 1) as f64 * gated.len() as f64 / n as f64;
        println!(
            "    τ={tau:.2}: gate {:5.1}% · mean reads {mean_reads:.2} · gated acc {g_acc:.1}% · AUROC on gated: −H1 {a1:.4} · maxprob {a2:.4} · agreement {a3:.4}",
            100.0 * gated.len() as f64 / n as f64,
        );
    }
}

// ── Instrument self-check (known-answer vectors — an arm that certifies
//    nothing is the failure mode) ─────────────────────────────────────────

#[test]
fn metrics_selfcheck() {
    // Perfect separation.
    let v = vec![0.1, 0.2, 0.8, 0.9];
    let c = vec![false, false, true, true];
    assert!((auroc_midrank(&v, &c) - 1.0).abs() < 1e-12);
    // Perfectly anti-separated.
    let c2 = vec![true, true, false, false];
    assert!((auroc_midrank(&v, &c2) - 0.0).abs() < 1e-12);
    // All-tied => 0.5 exactly.
    let v3 = vec![1.0, 1.0, 1.0, 1.0];
    assert!((auroc_midrank(&v3, &c) - 0.5).abs() < 1e-12);
    // Midrank tie case: values [0,0,1,1], pos at ranks {1.5, 3.5} => 0.5.
    let v4 = vec![0.0, 0.0, 1.0, 1.0];
    let c4 = vec![true, false, true, false];
    assert!((auroc_midrank(&v4, &c4) - 0.5).abs() < 1e-12);
    // Partial: values [0,1,1], one positive mid-ranked => 0.75.
    let v5 = vec![0.0, 1.0, 1.0];
    let c5 = vec![false, true, false];
    assert!((auroc_midrank(&v5, &c5) - 0.75).abs() < 1e-12);
    // Single class => NaN.
    assert!(auroc_midrank(&v, &[true; 4]).is_nan());

    // Selective accuracy: full coverage = base rate; top-1/3 picks the max.
    let conf = vec![0.1, 0.9, 0.5];
    let corr = vec![false, true, false];
    assert!((selective_accuracy(&conf, &corr, 1.0) - 1.0 / 3.0).abs() < 1e-12);
    assert!((selective_accuracy(&conf, &corr, 1.0 / 3.0) - 1.0).abs() < 1e-12);

    // Percentile interpolation.
    let s = vec![0.0, 1.0, 2.0, 3.0];
    assert!((percentile(&s, 0.5) - 1.5).abs() < 1e-12);
    assert!((percentile(&s, 0.0)).abs() < 1e-12);
    assert!((percentile(&s, 1.0) - 3.0).abs() < 1e-12);
}

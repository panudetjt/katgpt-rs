//! K3 pause-token logit-divergence diagnostic (riir-train Issue 407 T4 precondition).
//!
//! **Question:** does `kimi_k3_inject_pause` change the output logits AT ALL on
//! random weights? This is the precondition for the modelless G5-K3 quality gate
//! (T4). If pause tokens produce near-zero logit change on random weights, the
//! mechanism may require trained weights to have any effect — which would be an
//! honest negative result for the modelless G5-K3 claim.
//!
//! **Why random weights:** T4's re-scope is to test the G5 claim ("latent
//! thinking before answering improves the answer") modellessly — without a
//! trained checkpoint. The `latent_thought_flow_scorer_bench` proved this works
//! for the LINEAR `LatentThoughtKernel` (K=0 12.45% → K=3 39.00%). K3's forward
//! is non-linear (MLA + KDA + Dense FFN); this diagnostic checks whether the
//! non-linear forward still propagates pause-induced state changes to the logits.
//!
//! **Metric:** symmetric KL divergence (Jensen-Shannon-like) between the
//! softmax(logits_N0) and softmax(logits_Nk) distributions over the full vocab.
//! We use a numerically-stable formulation. A divergence of 0.0 means pause had
//! no effect; a divergence >0.01 nats means the KDA state change reached the
//! output distribution.
//!
//! **What this does NOT prove:** even if divergence is high, that only means
//! pause CHANGES the output, not that it IMPROVES it. The quality gate (T4
//! proper) needs a task with a known correct answer. This diagnostic is the
//! precondition — "does the mechanism do anything observable?"
//!
//! # Run
//!
//! ```bash
//! cargo run --release --bench k3_pause_logit_divergence_bench --features kimi_k3_loader
//! ```

#![cfg(feature = "kimi_k3_loader")]

use katgpt_rs::kimi_k3::{
    loader::KimiK3ModelWeights,
    model::{
        KimiK3ModelConfig, KimiK3Runtime, PauseConfig, PauseStrategy, kimi_k3_forward_token,
        kimi_k3_inject_pause,
    },
};

/// Test prompts as raw token IDs. We don't need a tokenizer for this diagnostic
/// — the question is whether pause changes the logits, not whether the output is
/// meaningful. We use a few structurally different prompts to avoid
/// prompt-specific artifacts:
/// - Prompt A: BOS + 4 arbitrary tokens (short, generic)
/// - Prompt B: BOS + 8 tokens (longer context for KDA to accumulate)
/// - Prompt C: BOS + 2 repeated tokens (structured, tests repeat-sensitivity)
const PROMPT_A: &[u32] = &[1, 100, 200, 300, 400];
const PROMPT_B: &[u32] = &[1, 10, 20, 30, 40, 50, 60, 70, 80];
const PROMPT_C: &[u32] = &[1, 42, 42, 42];

const PAUSE_COUNTS: &[usize] = &[0, 1, 2, 4, 8];

/// Strategies to test. TokenId(10) is a low, benign token ID.
const STRATEGIES: &[(&str, PauseStrategy)] = &[
    ("ZeroEmbedding", PauseStrategy::ZeroEmbedding),
    ("RepeatLast", PauseStrategy::RepeatLast),
    ("TokenId(10)", PauseStrategy::TokenId(10)),
];

/// Numerically stable softmax (no overflow, sigmoid-style subtraction of max).
fn softmax_stable(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &l in logits {
        // Clamp to avoid inf from exp(large positive). Max is subtracted so
        // the largest exponent is exp(0)=1; negatives are fine.
        let e = (l - max).exp();
        exps.push(e);
        sum += e;
    }
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

/// Symmetric KL divergence: 0.5 * (KL(P||Q) + KL(Q||P)).
/// Returns nats. 0.0 = identical distributions.
/// Uses the standard KL formula with epsilon clamping for numerical stability.
fn symmetric_kl(p: &[f32], q: &[f32]) -> f32 {
    debug_assert_eq!(p.len(), q.len());
    let mut total = 0.0f32;
    let eps = 1e-12f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let pi = pi.max(eps);
        let qi = qi.max(eps);
        // KL(P||Q) = sum pi * ln(pi/qi)
        total += pi * (pi / qi).ln();
        // KL(Q||P) = sum qi * ln(qi/pi)
        total += qi * (qi / pi).ln();
    }
    total * 0.5
}

/// Feed a prompt through the model, return the logits after the last token.
/// Uses a fresh runtime for each call (the diagnostic needs independent
/// measurements — KDA state accumulates, so reusing runtime would confound).
fn feed_prompt_get_logits(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    prompt: &[u32],
    pause: &PauseConfig,
) -> Vec<f32> {
    let mut runtime = KimiK3Runtime::new(config, 32);
    runtime.reset();

    let mut last_logits: Vec<f32> = Vec::new();
    let mut last_token: u32 = 0;
    for &tok in prompt {
        let logits = kimi_k3_forward_token(config, weights, &mut runtime, tok);
        last_logits = logits.to_vec();
        last_token = tok;
    }

    kimi_k3_inject_pause(
        config,
        weights,
        &mut runtime,
        pause,
        last_token,
        &last_logits,
    )
    .to_vec()
}

fn main() {
    println!("=== K3 pause-token logit-divergence diagnostic (Issue 407 T4 precondition) ===");
    println!();
    println!("Question: does kimi_k3_inject_pause change the output logits on random weights?");
    println!("If divergence ~0: the mechanism has no observable effect modellessly.");
    println!("If divergence >0.01 nats: the KDA state change reaches the output distribution");
    println!("                        (precondition met for T4 quality gate design).");
    println!();

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    println!(
        "Model: kimi_k3_0_40b (hidden={}, vocab={}, layers={})",
        config.hidden_size, config.vocab_size, config.num_layers
    );
    println!("Weights: random (seed=42)");
    let weights = KimiK3ModelWeights::random(&config, 42);
    println!();

    let prompts: &[(&str, &[u32])] = &[
        ("Prompt A (5 tok)", PROMPT_A),
        ("Prompt B (9 tok)", PROMPT_B),
        ("Prompt C (4 tok, repeated)", PROMPT_C),
    ];

    // Threshold: if ALL strategy × prompt combos produce < this divergence at
    // N=8, the mechanism is effectively inert on random weights.
    const DIVERGENCE_THRESHOLD: f32 = 0.01;

    let mut max_divergence_any = 0.0f32;
    let mut all_inert = true;

    for &(prompt_label, prompt) in prompts {
        println!("--- {prompt_label} {prompt:?} ---");

        // Compute N=0 baseline logits (the prompt's natural next-token distribution).
        let pause_zero = PauseConfig {
            n_pause: 0,
            strategy: PauseStrategy::ZeroEmbedding, // irrelevant when n_pause=0
        };
        let logits_n0 = feed_prompt_get_logits(&config, &weights, prompt, &pause_zero);
        let probs_n0 = softmax_stable(&logits_n0);

        for &(strategy_name, strategy) in STRATEGIES {
            print!("  {strategy_name:>14}:");
            for &n in PAUSE_COUNTS {
                let pause = PauseConfig {
                    n_pause: n,
                    strategy,
                };
                let logits_n = feed_prompt_get_logits(&config, &weights, prompt, &pause);

                if n == 0 {
                    // N=0 should match the baseline exactly (sanity check).
                    let probs_n = softmax_stable(&logits_n);
                    let kl = symmetric_kl(&probs_n0, &probs_n);
                    // Expected ~0 (identical computation). Report for sanity.
                    print!("  N={n:<2} KL={kl:.6}");
                } else {
                    let probs_n = softmax_stable(&logits_n);
                    let kl = symmetric_kl(&probs_n0, &probs_n);
                    if kl > max_divergence_any {
                        max_divergence_any = kl;
                    }
                    if kl >= DIVERGENCE_THRESHOLD {
                        all_inert = false;
                    }
                    print!("  N={n:<2} KL={kl:.6}");
                }
            }
            println!();
        }
        println!();
    }

    println!("=== Diagnostic verdict ===");
    println!("Max symmetric KL divergence observed: {max_divergence_any:.6} nats");
    println!("Divergence threshold (precondition for T4): {DIVERGENCE_THRESHOLD:.6} nats");
    if all_inert {
        println!(
            "VERDICT: MECHANISM INERT on random weights. Pause tokens produce < {DIVERGENCE_THRESHOLD:.6} nats"
        );
        println!("divergence across all strategy x prompt x N configs. The KDA state change");
        println!("from pause tokens does NOT reach the output distribution on random weights.");
        println!("This suggests the mechanism requires trained weights to have an observable");
        println!("effect — an honest negative result for the modelless G5-K3 claim. T4's");
        println!("quality gate may not be designable without at least a partially-trained base.");
    } else {
        println!(
            "VERDICT: MECHANISM ACTIVE. At least one config produced >= {DIVERGENCE_THRESHOLD:.6} nats divergence."
        );
        println!("The KDA state change from pause tokens DOES reach the output distribution");
        println!("on random weights. The precondition for T4 (modelless G5-K3 quality gate) is");
        println!("MET — proceed to design the quality task (see Issue 407 T4 analysis for");
        println!("candidate metrics: input-output alignment, copy task, etc.).");
    }
}

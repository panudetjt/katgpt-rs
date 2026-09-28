// Issue 859 GOAT gates — the latency half (G2) + the three-competitor
// informational arm.
//
// G2: read-only read ≤ full-loop latency on the same fixture. The full-loop
// competitor is the REAL substrate `denoise_loop` (same seq shape, ≥1
// bidirectional forward per step + per-step commit scans). Measured with the
// Issue-723 treatment: interleaved (read, loop) chunk PAIRS, median of
// per-pair ratios — never two sequential arms — because sequential arms of
// identical work drifted +5.2%/+21.7% on a loaded box.
//
// Competitor arm (informational, printed for the bench doc): read-argmax at
// step 1 vs the full-loop-then-parse answer (iterate read→commit until the
// canvas fills) — agreement % on random-weight fixtures is reported, not
// asserted: random init has no ground truth, so this arm records the
// mechanism's agreement profile, and the 4090 reference run (Issue 859 T1,
// deferred) owns the accuracy axis.
//
// Run in release for honest latency: `cargo test --release -p katgpt-forward
// --features structured_reads --test structured_read_goat -- --nocapture`.

#![cfg(feature = "structured_reads")]

use katgpt_forward::NoConstraint;
use katgpt_forward::denoise_loop;
use katgpt_forward::structured_read::{
    MAX_LABELS, SlotReadout, StructuredReadScratch, structured_read_into,
};
use katgpt_transformer::TransformerWeights;
use katgpt_types::{Config, Rng};
use std::hint::black_box;
use std::time::Instant;

fn zero_readout() -> SlotReadout {
    SlotReadout {
        argmax_logprob: 0.0,
        argmax_label_prob: 0.0,
        label_entropy: 0.0,
        vocab_argmax_token: 0,
        position: 0,
        argmax_index: 0,
        n_labels: 0,
        label_logprobs: [0.0; MAX_LABELS],
    }
}

fn fixture(canvas_len: usize, n_free: usize) -> (TransformerWeights, Config, Vec<usize>) {
    let config = Config::micro_dllm();
    let mut rng = Rng::new(0x0859_6005);
    let weights = TransformerWeights::new(&config, &mut rng);
    // Deterministic canvas: fixed tokens, every 3rd position free up to n_free.
    let mut canvas = Vec::with_capacity(canvas_len);
    let mut free = 0;
    for i in 0..canvas_len {
        if free < n_free && i % 3 == 1 {
            canvas.push(config.mask_token);
            free += 1;
        } else {
            canvas.push((i * 7 + 3) % (config.vocab_size - 1));
        }
    }
    (weights, config, canvas)
}

/// G2 — interleaved-pairs median ratio (Issue 723 discipline).
#[test]
fn g2_read_only_not_slower_than_full_loop() {
    let (weights, config, canvas) = fixture(12, 4);
    let labels: Vec<u32> = vec![1, 4, 9, 16, 22];
    let mut out = [zero_readout(); 4];
    let mut scratch = StructuredReadScratch::new(&config);

    // Warmup both arms.
    for _ in 0..3 {
        structured_read_into(&mut out, &weights, &config, &canvas, &labels, &mut scratch).unwrap();
        let _ = denoise_loop(
            &weights,
            &canvas,
            &config,
            8,
            0.3,
            &mut NoConstraint,
            &mut Rng::new(1),
        );
    }

    const PAIRS: usize = 33;
    const CHUNK: usize = 8;
    let mut ratios: Vec<f64> = Vec::with_capacity(PAIRS);
    for _ in 0..PAIRS {
        let t = Instant::now();
        for _ in 0..CHUNK {
            let n =
                structured_read_into(&mut out, &weights, &config, &canvas, &labels, &mut scratch)
                    .unwrap();
            black_box(n);
        }
        let read_ns = t.elapsed().as_nanos() as f64;

        let t = Instant::now();
        for _ in 0..CHUNK {
            let r = denoise_loop(
                &weights,
                black_box(&canvas),
                &config,
                8,
                0.3,
                &mut NoConstraint,
                &mut Rng::new(1),
            );
            black_box(r);
        }
        let loop_ns = t.elapsed().as_nanos() as f64;

        ratios.push(read_ns / loop_ns);
    }
    ratios.sort_by(|a, b| a.total_cmp(b));
    let median = ratios[PAIRS / 2];
    println!(
        "g2: read/full-loop median ratio = {median:.4} (min {:.4}, max {:.4}) over {PAIRS} interleaved pairs x{CHUNK}",
        ratios[0],
        ratios[PAIRS - 1]
    );
    assert!(
        median <= 1.0,
        "read-only structured read must not be slower than the full denoise loop (median ratio {median:.4})"
    );
}

/// Competitor arm (informational): step-1 read argmax vs the full-loop-then-
/// parse answer built on the same primitive (iterate read→commit until the
/// canvas fills). Printed for the bench record; nothing asserted — random
/// weights carry no ground truth (the 4090 reference run owns accuracy).
#[test]
fn competitor_read_vs_full_loop_then_parse_informational() {
    let (weights, config, canvas) = fixture(12, 4);
    let labels: Vec<u32> = vec![1, 4, 9, 16, 22];
    let mut scratch = StructuredReadScratch::new(&config);
    let mut out = [zero_readout(); 4];

    // Step-1 read.
    structured_read_into(&mut out, &weights, &config, &canvas, &labels, &mut scratch).unwrap();
    let step1: Vec<u32> = out
        .iter()
        .map(|r| labels[r.argmax_index as usize])
        .collect();

    // Full loop: commit the label argmax at each free slot, re-read, repeat.
    let mut full = canvas.clone();
    let mut rounds = 0;
    loop {
        let n_free = full.iter().filter(|&&t| t == config.mask_token).count();
        if n_free == 0 || rounds > 16 {
            break;
        }
        let mut outs = vec![zero_readout(); n_free];
        structured_read_into(&mut outs, &weights, &config, &full, &labels, &mut scratch).unwrap();
        for r in &outs {
            full[r.position as usize] = labels[r.argmax_index as usize] as usize;
        }
        rounds += 1;
    }
    let final_answer: Vec<u32> = out
        .iter()
        .map(|r| full[r.position as usize] as u32)
        .collect();
    let agree = step1
        .iter()
        .zip(&final_answer)
        .filter(|(a, b)| a == b)
        .count();
    let entropy_mean: f32 = out.iter().map(|r| r.label_entropy).sum::<f32>() / out.len() as f32;
    println!(
        "competitor: step-1 read vs full-loop-then-parse agreement {agree}/{} ({:.0}%) after {rounds} commit rounds; mean label entropy {entropy_mean:.4} nats",
        step1.len(),
        100.0 * agree as f32 / step1.len() as f32
    );
}

//! Plan 598 T6 — `coarse_grain` latency bench (the G2 overhead gate).
//!
//! The premise under test: the single-pass byte-marginal conversion is a
//! thin scatter-add ON TOP of the softmax the forward pass already paid —
//! the plan's bar is <5% of the softmax cost at serving vocab sizes.
//! Also measures the checkpoint-time table build (bar: <1 s per 100K vocab)
//! and, under the `alloc_tracking` feature, asserts the streaming loop's
//! steady-state alloc count is zero (the G4 half; the harness pattern is
//! the Issue-741 class — the gate is compiled to nothing without the
//! feature, so its green HERE is dev-profile-only; the release
//! alloc-tracking run is a separate invocation).
//!
//! Convention: `std::time::Instant` self-timing (criterion is not a
//! katgpt-rs dev-dep; the repo bench convention — see
//! `benches/fpcg_probe_forecast_bench.rs`). Median of 51 after 7 warmups.

#![cfg(feature = "refinement_marginal")]
#![cfg(not(target_arch = "wasm32"))]

use katgpt_core::refinement_marginal::{
    CoarseGrainScratch, CoarseRecord, coarse_grain_first, coarse_grain_step,
};
use katgpt_rs::refinement_bridge::refinement_table_from_bpe;
use katgpt_tokenizer::BpeTokenizer;

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

/// Softmax over `logits` (the already-paid cost the conversion is compared
/// against) — subtract-max, exp, normalize. Allocates its output once
/// outside the timed loop; the exp pass is the measured work.
fn softmax_into(logits: &[f32], out: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &l in logits {
        m = m.max(l);
    }
    let mut z = 0.0_f64;
    for (o, &l) in out.iter_mut().zip(logits) {
        let e = (l - m) as f64;
        *o = e as f32;
        z += e;
    }
    let inv = 1.0 / z;
    for o in out.iter_mut() {
        *o *= inv as f32;
    }
}

/// Synthetic vocab: `n` tokens whose byte strings mimic BPE shape — mostly
/// 1–8 bytes with shared first bytes (deterministic LCG-derived).
fn synth_tokenizer(n: usize, seed: u64) -> BpeTokenizer {
    let mut s = seed | 1;
    let mut lcg = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut id_to_vocab = Vec::with_capacity(n);
    for i in 0..n {
        let len = 1 + (lcg() % 8) as usize;
        let bytes: Vec<u8> = (0..len).map(|_| (lcg() % 256) as u8).collect();
        id_to_vocab.push(String::from_utf8_lossy(&bytes).into_owned());
        let _ = i;
    }
    BpeTokenizer {
        vocab_to_id: Default::default(),
        id_to_vocab,
        merges: vec![],
        merge_ranks: Default::default(),
        merge_ranks_id: Default::default(),
        merge_target_id: vec![],
        bos_id: 0,
        eos_id: 0,
        pad_id: 0,
    }
}

fn main() {
    println!(
        "{:>9} {:>12} {:>12} {:>8} {:>12}",
        "vocab", "softmax ns", "grain ns", "ratio", "table ms"
    );

    #[cfg(feature = "alloc_tracking")]
    let mut steady_allocs = 0usize;

    for &n in &[2_048usize, 32_768, 131_072] {
        let tok = synth_tokenizer(n, 0x5EED_0598);

        // ── table build (checkpoint-time) ──
        let t0 = std::time::Instant::now();
        let table = refinement_table_from_bpe(&tok);
        let table_ms = t0.elapsed().as_secs_f64() * 1e3;

        let logits: Vec<f32> = (0..n)
            .map(|i| ((i % 97) as f32).ln() - 0.01 * i as f32)
            .collect();
        let mut probs = vec![0.0_f32; n];

        // ── softmax baseline ──
        let mut sm: Vec<f64> = Vec::with_capacity(51);
        for _ in 0..7 {
            softmax_into(&logits, &mut probs);
        }
        for _ in 0..51 {
            let t = std::time::Instant::now();
            softmax_into(&logits, &mut probs);
            sm.push(t.elapsed().as_secs_f64());
        }
        let softmax_ns = median(&mut sm) * 1e9;

        // ── one full streaming conversion (depth-0 + 7 steps), with the
        //    depth-0 / step-loop SPLIT measured separately — depth-0 is
        //    the always-paid exact marginal; the steps are per-byte. ──
        let mut rec = CoarseRecord::zeroed();
        let mut scratch = CoarseGrainScratch::new(n);

        #[cfg(feature = "alloc_tracking")]
        let loop_start_allocs = katgpt_core::alloc::get_alloc_stats().0;

        let mut first_only: Vec<f64> = Vec::with_capacity(51);
        for _ in 0..7 {
            coarse_grain_first(&probs, &table, &mut rec, &mut scratch);
        }
        for _ in 0..51 {
            let t = std::time::Instant::now();
            coarse_grain_first(&probs, &table, &mut rec, &mut scratch);
            first_only.push(t.elapsed().as_secs_f64());
        }
        let first_ns = median(&mut first_only) * 1e9;

        let mut gr: Vec<f64> = Vec::with_capacity(51);
        for _ in 0..7 {
            coarse_grain_first(&probs, &table, &mut rec, &mut scratch);
            let mut depth = 0;
            let mut b = 0u8;
            while depth < 8 {
                // greedy pick: first non-zero bin (deterministic, cheap)
                while b < 255 && rec.bins[b as usize] == 0.0 {
                    b += 1;
                }
                coarse_grain_step(&probs, &table, b, &mut rec, &mut scratch);
                depth += 1;
            }
        }
        for _ in 0..51 {
            let t = std::time::Instant::now();
            coarse_grain_first(&probs, &table, &mut rec, &mut scratch);
            let mut depth = 0;
            let mut b = 0u8;
            while depth < 8 {
                while b < 255 && rec.bins[b as usize] == 0.0 {
                    b += 1;
                }
                coarse_grain_step(&probs, &table, b, &mut rec, &mut scratch);
                depth += 1;
            }
            gr.push(t.elapsed().as_secs_f64());
        }
        let grain_ns = median(&mut gr) * 1e9;

        #[cfg(feature = "alloc_tracking")]
        {
            let after = katgpt_core::alloc::get_alloc_stats().0;
            steady_allocs = steady_allocs.max(after.saturating_sub(loop_start_allocs));
        }

        println!(
            "{:>9} {:>12.0} {:>12.0} {:>7.2}x {:>12.1}  (depth-0 alone {:>9.0} ns = {:>5.2}x softmax)",
            n,
            softmax_ns,
            grain_ns,
            grain_ns / softmax_ns,
            table_ms,
            first_ns,
            first_ns / softmax_ns
        );
    }

    #[cfg(feature = "alloc_tracking")]
    {
        println!("\nG4: steady-state allocs across the streaming loop: {steady_allocs}");
        assert_eq!(steady_allocs, 0, "the streaming loop allocated");
    }
    #[cfg(not(feature = "alloc_tracking"))]
    println!("\nG4 alloc gate: run with --features alloc_tracking to assert 0 allocs");

    println!(
        "\nBars: ratio < 0.05 (the <5%-of-softmax plan bar); table build < 1 s per 100K vocab."
    );
}

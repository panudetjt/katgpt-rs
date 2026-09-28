//! Plan 598 T5 — integration: the BPE bridge over a real trained vocabulary,
//! streaming conversion vs an independent two-pass reference, and the
//! certificate bound holding on every sampled position (the G1 "never
//! under-reports" integration half).
//!
//! # Deviation from the plan text (recorded, deliberate)
//!
//! The plan named "a real GGUF vocab"; `katgpt-tokenizer` ships no GGUF
//! reader (and is a categorical leaf — adding one there is its own project).
//! The substitute: a REAL corpus (repeated English sentences, ~1.5 KB) run
//! through the in-tree deterministic `BpeTrainer` → a real BPE vocabulary
//! with real merge structure (multi-byte tokens, shared prefixes, a
//! boundary-heavy short-token tail) — the properties T5 needs from "real".
//!
//! # Two-pass reference
//!
//! For prompt prefix `p` at depth `k`, the reference independently computes
//! the byte marginal by rescanning the FULL vocabulary with the paper's
//! Marginalize-It semantics (prefix filter + renormalize) — a different
//! code path from the streaming frontier (`naive_reference` below). The
//! streaming record's surviving bins, scaled by the same renormalization,
//! must match the reference to fp tolerance; its terminal bin must match
//! the reference's dropped mass exactly.
//!
//! # The bound gate
//!
//! For every sampled (prompt, depth) cell: `error_bound(record) ≥ TV`
//! between the renormalized approximate readout and the ADVERSARIAL exact
//! continuation (all dropped mass on the argmin byte). Asserted 100% over
//! the sample — a single under-report fails the test.

#![cfg(feature = "refinement_marginal")]
#![cfg(not(target_arch = "wasm32"))]

use katgpt_core::refinement_marginal::{
    CoarseGrainScratch, CoarseRecord, TERMINAL_BIN, coarse_grain_first, coarse_grain_step,
    error_bound, expected_escalation_cost,
};
use katgpt_rs::refinement_bridge::{decode_argmax, refinement_table_from_bpe};
use katgpt_tokenizer::BpeTrainer;

const CORPUS: &str = "\
the quick brown fox jumps over the lazy dog\n\
the cat sat on the mat and the dog ate the fat rat\n\
a quick brown cat and a lazy dog sat on the mat\n\
the fox ran over the dog and the cat ran over the mat\n\
the lazy dog saw the quick fox and the fat cat ate the rat\n\
over the mat the cat sat and the dog ran to the fox\n\
the quick cat and the brown dog ate a fat rat on the mat\n\
a dog a cat a fox a rat the mat the hat the fat cat sat\n\
the brown fox jumps and the lazy cat runs over the mat\n\
the fat rat ran and the quick dog ate and the cat sat\n";

const N_PROMPTS: usize = 24;
const MAX_DEPTH: usize = 6;

/// Deterministic LCG (house fixture pattern).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The paper's Marginalize-It + terminal bin, written as an independent
/// full-vocab rescan (a different code path from the streaming frontier).
fn naive_reference(probs: &[f32], id_bytes: &[&[u8]], prefix: &[u8]) -> (Vec<f32>, f32, f32) {
    let k = prefix.len();
    let mut bins = vec![0.0_f32; 257];
    let mut alive = 0.0_f32;
    let mut terminal = 0.0_f32;
    for (s, bytes) in id_bytes.iter().enumerate() {
        if bytes.len() < k || (0..k).any(|i| bytes[i] != prefix[i]) {
            continue;
        }
        let p = probs[s];
        alive += p;
        if bytes.len() == k {
            bins[TERMINAL_BIN] += p;
            terminal += p;
        } else {
            bins[bytes[k] as usize] += p;
        }
    }
    (bins, terminal, alive)
}

#[test]
fn tokenizer_bridge_streaming_matches_reference_and_bound_holds() {
    let tok = BpeTrainer::train(CORPUS, 96);
    let table = refinement_table_from_bpe(&tok);
    let n = table.n_symbols();
    assert!(n >= 64, "vocab too small for the integration run: {n}");

    // Snapshot the byte sequences for the reference path.
    let id_bytes: Vec<&[u8]> = tok.id_to_vocab.iter().map(|s| s.as_bytes()).collect();

    let mut rng = Lcg(0x5EED_0598);
    let mut scratch = CoarseGrainScratch::new(n);
    let mut rec = CoarseRecord::zeroed();
    let mut cells = 0usize;
    let mut max_slack = f32::INFINITY;

    for _ in 0..N_PROMPTS {
        // A random probability vector over the vocab (normalized) — the
        // stand-in for one forward pass's softmax output.
        let mut probs: Vec<f32> = (0..n).map(|_| 0.01 + rng.unit()).collect();
        let total: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= total;
        }

        // Stream the greedy path (the consumer's real loop), recording the
        // certificate at every depth; ALSO verify each record against the
        // independent reference computed at the same prefix.
        coarse_grain_first(&probs, &table, &mut rec, &mut scratch);
        let mut prefix: Vec<u8> = Vec::new();
        let mut records: Vec<CoarseRecord> = Vec::new();
        for depth in 0..MAX_DEPTH {
            // ── reference agreement ──
            let (ref_bins, ref_terminal, ref_alive) = naive_reference(&probs, &id_bytes, &prefix);
            assert!(
                (rec.alive_mass - ref_alive).abs() < 1e-4,
                "depth {depth}: alive {} vs reference {ref_alive}",
                rec.alive_mass
            );
            assert!((rec.terminal_mass - ref_terminal).abs() < 1e-4);
            for (b, (&got, &want)) in rec.bins.iter().zip(ref_bins.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-4,
                    "depth {depth} bin {b}: {got} vs {want}"
                );
            }

            // ── the bound gate (the 100% assertion) ──
            let t = rec.terminal_mass;
            let c = rec.alive_mass - t;
            let bound = error_bound(&rec);
            if c > 0.0 && t > 0.0 {
                // Adversarial exact: ALL dropped mass on the argmin byte.
                let argmin = (0..TERMINAL_BIN)
                    .min_by(|&a, &b| rec.bins[a].total_cmp(&rec.bins[b]))
                    .unwrap();
                let mut tv = 0.0_f32;
                for b in 0..TERMINAL_BIN {
                    let approx = rec.bins[b] / c;
                    let m = if b == argmin { t } else { 0.0 };
                    tv += (approx - (rec.bins[b] + m) / rec.alive_mass).abs();
                }
                tv *= 0.5;
                assert!(
                    bound >= tv - 1e-5,
                    "depth {depth}: certificate {bound} under-reported TV {tv}"
                );
                max_slack = max_slack.min(bound - tv);
            }
            cells += 1;
            records.push(rec);

            // ── advance the greedy path ──
            match decode_argmax(&rec) {
                Some(b) => {
                    prefix.push(b);
                    coarse_grain_step(&probs, &table, b, &mut rec, &mut scratch);
                }
                None => break, // terminal won: the refinement ended
            }
        }

        // The escalation cost aggregates the per-step terminal masses.
        let cost = expected_escalation_cost(&records);
        let sum_terminal: f32 = records.iter().map(|r| r.terminal_mass).sum();
        assert!((cost - sum_terminal).abs() < 1e-6);
    }

    assert!(cells >= N_PROMPTS * 2, "too few cells: {cells}");
    println!(
        "plan598 T5: {cells} (prompt, depth) cells over vocab {n}; \
         min bound-slack {max_slack:.6} (≥ 0 ⇒ bound never under-reported)"
    );
    assert!(
        max_slack.is_finite() && max_slack >= 0.0,
        "bound slack went negative somewhere: {max_slack}"
    );
}

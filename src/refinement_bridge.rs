//! refinement_bridge — the BPE instantiation of katgpt-core's
//! `refinement_marginal` (Plan 598 / Research 559; arXiv:2609.12303).
//!
//! Builds a [`RefinementTable`] from a [`BpeTokenizer`]'s vocabulary (each
//! token's UTF-8 bytes are its child sequence) and exposes the per-position
//! streaming loop that converts token-space logits into the 257-bin
//! byte-space record with the terminal-mass certificate.
//!
//! # Why this lives in the ROOT crate
//!
//! `katgpt-tokenizer` is a categorical leaf ("no `katgpt-*` dependencies" —
//! its lib doc), and `RefinementTable` lives in `katgpt-core`. The bridge
//! needs both types, so the aggregator crate — which already depends on
//! both non-optionally — is the only home that honors the leaf constraint.
//!
//! # The per-position loop (the whole consumer API)
//!
//! ```ignore
//! let table = refinement_table_from_bpe(&tok);
//! let mut scratch = CoarseGrainScratch::new(table.n_symbols());
//! let mut rec = CoarseRecord::zeroed();
//! let mut records = Vec::new();
//!
//! // one forward pass: probs = softmax(logits) over the vocab
//! coarse_grain_first(&probs, &table, &mut rec, &mut scratch);
//! records.push(rec);
//! let first_byte = decode_argmax(&rec);          // consumer picks a byte
//! coarse_grain_step(&probs, &table, first_byte, &mut rec, &mut scratch);
//! records.push(rec);
//! // ... until the terminal bin wins or the token completes
//! let cost = expected_escalation_cost(&records); // Σ_k M_k
//! if escalation_sigmoid(cost, THRESHOLD, 1.0) > 0.5 { /* escalate exact */ }
//! ```
//!
//! Depth-0 is the EXACT first-byte marginal (the paper's one lossless
//! half); each later step is approximate with a closed-form certificate —
//! serve-while-small, escalate-when-not.

#![allow(dead_code)]

use katgpt_core::refinement_marginal::{CoarseRecord, RefinementTable, TERMINAL_BIN};
use katgpt_tokenizer::BpeTokenizer;

/// Build the checkpoint-time refinement table from a BPE vocabulary: token
/// `id`'s child sequence is its UTF-8 byte string (`id_to_vocab[id]`).
///
/// One-time cost per checkpoint (`O(Σ token bytes)`); the table is
/// immutable afterward and shared across every position and request.
pub fn refinement_table_from_bpe(tok: &BpeTokenizer) -> RefinementTable {
    let seqs: Vec<&[u8]> = tok.id_to_vocab.iter().map(|s| s.as_bytes()).collect();
    RefinementTable::from_sequences(&seqs)
}

/// Static geometry profile of a tokenizer against its refinement table —
/// computed once per checkpoint (offline), never per request.
///
/// - `mean_len` — E[bytes per token]; the BPT↔BPB converter (R559 §2.3
///   last row: cross-tokenizer throughput rows are confounded without it).
/// - `max_len` / `max_len_id` — the deepest refinement (the longest token).
/// - `breadth_at_depth` — distinct child symbols observed at each depth
///   (the trie's fan-out shape; byte instantiation caps at 256).
/// - `dead_prefix_mass` — total probability mass landing in the terminal
///   bin across a uniform-depth-1 pass: the fraction of one-hot vocab mass
///   carried by tokens of length ≤ 1 (the boundary-heavy tail).
pub struct TokenizerGeometry {
    pub mean_len: f64,
    pub max_len: usize,
    pub max_len_id: usize,
    pub breadth_at_depth: Vec<usize>,
    pub dead_prefix_mass: f32,
}

/// Compute the geometry profile over a table (+ optional vocab probs for
/// `dead_prefix_mass`; pass `None` to skip that row).
pub fn tokenizer_geometry(table: &RefinementTable, probs: Option<&[f32]>) -> TokenizerGeometry {
    let n = table.n_symbols();
    let mut total_len = 0usize;
    let mut max_len = 0usize;
    let mut max_len_id = 0usize;
    for s in 0..n {
        let len = table.symbol_len(s);
        total_len += len;
        if len > max_len {
            max_len = len;
            max_len_id = s;
        }
    }
    // A len-≥(d+1) symbol contributes a child at depth d; the DISTINCT
    // child symbols per depth need an exact per-depth 256-bit bitmap.
    let mut bitmaps: Vec<[u64; 4]> = vec![[0u64; 4]; max_len];
    for s in 0..n {
        let len = table.symbol_len(s);
        // `d` drives both the bitmap row and the table lookup — enumerate
        // keeps the row borrow alive without the range-loop pattern.
        for (d, bm) in bitmaps.iter_mut().enumerate().take(len) {
            let b = table.byte_at(s, d) as usize;
            bm[b / 64] |= 1u64 << (b % 64);
        }
    }
    let breadth_at_depth: Vec<usize> = bitmaps
        .iter()
        .map(|bm| bm.iter().map(|w| w.count_ones() as usize).sum())
        .collect();
    let dead_prefix_mass = probs.map(|p| {
        let mut t = 0.0_f32;
        for (s, &p_s) in p.iter().enumerate().take(n) {
            if table.symbol_len(s) <= 1 {
                t += p_s;
            }
        }
        t
    });
    TokenizerGeometry {
        mean_len: total_len as f64 / n.max(1) as f64,
        max_len,
        max_len_id,
        breadth_at_depth,
        dead_prefix_mass: dead_prefix_mass.unwrap_or(0.0),
    }
}

/// Pick the argmax child symbol from a record, returning `None` when the
/// terminal bin wins (the refinement ENDED — the consumer emits the token
/// boundary, not a byte).
#[inline]
pub fn decode_argmax(record: &CoarseRecord) -> Option<u8> {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (b, &v) in record.bins[..TERMINAL_BIN].iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = b;
        }
    }
    if record.terminal_mass > best_v {
        None
    } else {
        Some(best as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::refinement_marginal::{CoarseGrainScratch, coarse_grain_first};

    /// The bridge produces the same table as the raw byte sequences.
    #[test]
    fn bridge_table_matches_raw_sequences() {
        let tok = BpeTokenizer {
            vocab_to_id: Default::default(),
            id_to_vocab: vec![
                "<pad>".to_string(),
                "a".to_string(),
                "ab".to_string(),
                "abc".to_string(),
            ],
            merges: vec![],
            merge_ranks: Default::default(),
            merge_ranks_id: Default::default(),
            merge_target_id: vec![],
            bos_id: 0,
            eos_id: 0,
            pad_id: 0,
        };
        let table = refinement_table_from_bpe(&tok);
        assert_eq!(table.n_symbols(), 4);
        assert_eq!(table.symbol_len(1), 1);
        assert_eq!(table.byte_at(1, 0), b'a');
        assert_eq!(table.symbol_len(3), 3);
        assert_eq!(table.byte_at(3, 2), b'c');
    }

    /// Geometry on the tiny vocab: E[len], max, breadth, dead mass.
    #[test]
    fn geometry_profile_rows() {
        let tok = BpeTokenizer {
            vocab_to_id: Default::default(),
            id_to_vocab: vec!["a".to_string(), "ab".to_string(), "b".to_string()],
            merges: vec![],
            merge_ranks: Default::default(),
            merge_ranks_id: Default::default(),
            merge_target_id: vec![],
            bos_id: 0,
            eos_id: 0,
            pad_id: 0,
        };
        let table = refinement_table_from_bpe(&tok);
        let probs = vec![0.5, 0.3, 0.2];
        let geo = tokenizer_geometry(&table, Some(&probs));
        assert!((geo.mean_len - 4.0 / 3.0).abs() < 1e-9);
        assert_eq!(geo.max_len, 2);
        assert_eq!(geo.max_len_id, 1);
        // depth 0: {a, ab, b} → 2 distinct bytes; depth 1: {ab} → 1.
        assert_eq!(geo.breadth_at_depth, vec![2, 1]);
        // tokens of len ≤ 1 hold 0.5 + 0.2 = 0.7 of the mass.
        assert!((geo.dead_prefix_mass - 0.7).abs() < 1e-6);
    }

    /// decode_argmax honors the terminal bin (None when the refinement
    /// ended) and picks the max byte otherwise.
    #[test]
    fn decode_argmax_terminal_aware() {
        let mut rec = CoarseRecord::zeroed();
        rec.bins[b'a' as usize] = 0.4;
        rec.bins[b'b' as usize] = 0.5;
        assert_eq!(decode_argmax(&rec), Some(b'b'));
        rec.terminal_mass = 0.6;
        assert_eq!(decode_argmax(&rec), None);
    }

    /// End-to-end smoke through the REAL BPE vocabulary path: the depth-0
    /// record over the trained vocab reproduces the byte histogram of the
    /// first characters of the vocab's own tokens (mass-weighted).
    #[test]
    fn depth0_over_real_bpe_vocab() {
        // A tiny real corpus → real BPE vocab (deterministic trainer).
        let corpus = "the cat sat on the mat the cat ate the fat rat on the mat\n\
                      the mat sat on the rat the cat ate the fat cat\n\
                      a bat a hat a rat the cat the mat the fat rat sat";
        let tok = katgpt_tokenizer::BpeTrainer::train(corpus, 48);
        let table = refinement_table_from_bpe(&tok);
        let n = table.n_symbols();
        assert!(n >= 24, "trained vocab too small: {n}");

        // One-hot on the token " cat" (if merged) else its first token id —
        // simplest deterministic pick: the LAST vocab entry.
        let mut probs = vec![0.0_f32; n];
        probs[n - 1] = 1.0;
        let mut rec = CoarseRecord::zeroed();
        let mut scratch = CoarseGrainScratch::new(n);
        coarse_grain_first(&probs, &table, &mut rec, &mut scratch);

        // The record must be one-hot at the last token's first byte.
        let first = tok.id_to_vocab[n - 1].as_bytes()[0];
        assert!((rec.bins[first as usize] - 1.0).abs() < 1e-6);
        assert_eq!(rec.terminal_mass, 0.0);
    }
}

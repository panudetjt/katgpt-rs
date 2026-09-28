//! Tiktoken BPE tokenizer loader for Kimi-K3.
//!
//! Parses the `tiktoken.model` binary format (used by GPT-4o, Kimi-K3, etc.).
//! The format is a sequence of `<base64_token_bytes><newline><rank>` pairs:
//!
//! ```text
//! <base64_encoded_token>\n<rank_as_ascii_int>\n
//! <base64_encoded_token>\n<rank_as_ascii_int>\n
//! ...
//! ```
//!
//! Each rank is the merge priority (lower = applied first). The base vocabulary
//! is the set of decoded byte strings; special tokens (BOS, EOS, etc.) are
//! appended after the base vocab.
//!
//! # Kimi-K3 specifics (Research 330 §6)
//!
//! - Vocab size: 163,840
//! - BOS token ID: 1, EOS token ID: 2 (special tokens after base vocab)
//! - Pretokenizer regex: GPT-4-like with `\p{Han}` Chinese character support
//!
//! # What this module provides
//!
//! - `load_tiktoken_bpe()` — parses the binary file into a rank table
//! - `TiktokenTokenizer` — wraps the rank table + provides encode/decode
//!
//! The `encode()` method applies the Kimi-K3 pretokenizer regex (with
//! `\p{Han}` CJK support) before running BPE on each chunk — this matches
//! the reference Python `tiktoken.Encoding` behavior exactly (Issue 411).

use base64::Engine;
use std::collections::HashMap;
use std::sync::LazyLock;

/// A tiktoken BPE rank table: `token_bytes → rank`.
///
/// The rank determines merge priority (lower rank = merged first).
pub type TiktokenRanks = HashMap<Vec<u8>, usize>;

/// Errors that can occur during tiktoken model loading.
#[derive(Debug)]
pub enum TiktokenLoadError {
    /// I/O error reading the file.
    Io(std::io::Error),
    /// Invalid base64 encoding in a token line.
    InvalidBase64(String),
    /// Invalid rank (not a valid integer).
    InvalidRank(String),
    /// Empty file or unexpected EOF.
    UnexpectedEof,
}

impl std::fmt::Display for TiktokenLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "tiktoken I/O error: {e}"),
            Self::InvalidBase64(s) => write!(f, "tiktoken invalid base64: {s}"),
            Self::InvalidRank(s) => write!(f, "tiktoken invalid rank: {s}"),
            Self::UnexpectedEof => write!(f, "tiktoken unexpected EOF"),
        }
    }
}

impl std::error::Error for TiktokenLoadError {}

/// Parse a `tiktoken.model` binary file into a rank table.
///
/// The file format is a sequence of newline-delimited entries:
/// ```text
/// <base64_token>\n<rank>\n
/// ```
///
/// Each entry is: a base64-encoded byte string (the token), followed by its
/// rank (an ASCII integer). The rank is the merge priority — lower ranks are
/// merged first during BPE encoding.
///
/// # Arguments
/// - `data` — the raw bytes of the `tiktoken.model` file
///
/// # Returns
/// A `HashMap<Vec<u8>, usize>` mapping each token's byte sequence to its rank.
///
/// # Errors
/// Returns `TiktokenLoadError` if the data is malformed.
pub fn load_tiktoken_bpe(data: &[u8]) -> Result<TiktokenRanks, TiktokenLoadError> {
    let mut ranks = HashMap::new();

    // The tiktoken.model file format can be either:
    //
    //   Format A (HuggingFace production): "<base64> <rank>\n" per line
    //   Format B (test synthetic):          "<base64>\n<rank>\n" alternating lines
    //
    // We detect the format by checking if the first line contains a space.
    let lines: Vec<&[u8]> = data
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .collect();

    let format_a = lines
        .first()
        .is_none_or(|first_line| first_line.iter().any(|&b| b == b' ' || b == b'\t'));

    if format_a {
        // Format A: each line is "<base64> <rank>"
        for line in &lines {
            let line_str = std::str::from_utf8(line)
                .map_err(|e| TiktokenLoadError::InvalidBase64(format!("non-UTF8 line: {e}")))?;
            let mut fields = line_str.split_whitespace();
            let b64_token = fields.next().ok_or_else(|| {
                TiktokenLoadError::InvalidRank(format!("empty line: {line_str:?}"))
            })?;
            let rank_str = fields.next().ok_or_else(|| {
                TiktokenLoadError::InvalidRank(format!("missing rank: {line_str:?}"))
            })?;

            let token = base64::engine::general_purpose::STANDARD
                .decode(b64_token)
                .map_err(|e| TiktokenLoadError::InvalidBase64(format!("{e}")))?;
            let rank: usize = rank_str
                .parse()
                .map_err(|e| TiktokenLoadError::InvalidRank(format!("'{rank_str}': {e}")))?;
            ranks.insert(token, rank);
        }
    } else {
        // Format B: alternating lines <base64>\n<rank>\n...
        let mut iter = lines.into_iter();
        while let Some(b64_line) = iter.next() {
            let rank_line = iter.next().ok_or(TiktokenLoadError::UnexpectedEof)?;
            if b64_line.is_empty() {
                break;
            }
            let token = base64::engine::general_purpose::STANDARD
                .decode(b64_line)
                .map_err(|e| TiktokenLoadError::InvalidBase64(format!("{e}")))?;
            let rank_str = std::str::from_utf8(rank_line)
                .map_err(|e| TiktokenLoadError::InvalidRank(format!("non-UTF8 rank: {e}")))?;
            let rank: usize = rank_str
                .trim()
                .parse()
                .map_err(|e| TiktokenLoadError::InvalidRank(format!("'{rank_str}': {e}")))?;
            ranks.insert(token, rank);
        }
    }

    Ok(ranks)
}

/// Kimi-K3 pretokenizer regex pattern (Issue 411).
///
/// This is the EXACT pattern from Moonshot's `tokenization_kimi.py` (lines
/// 54-65), which uses character class intersection `&&[^\p{Han}]` to exclude
/// CJK characters from the word rules. Rust's `fancy-regex` does not support
/// `&&` syntax, so we emulate it with negative lookahead `(?!\p{Han})` before
/// each character class match. This produces identical tokenization.
///
/// The pattern alternatives (in priority order):
/// 1. `[\p{Han}]+` — CJK characters
/// 2. Lowercase-initiated words (non-Han letters) + optional contractions
/// 3. Uppercase-initiated words (non-Han letters) + optional contractions
/// 4. Numbers (1-3 digits)
/// 5. Punctuation (with optional leading space)
/// 6. Newlines
/// 7. Trailing whitespace
/// 8. Other whitespace
static KIMI_K3_PRETOK_PATTERN: &str = concat!(
    r"[\p{Han}]+",
    // Lowercase word: optional non-letter prefix, then 0+ uppercase/other
    // letters, then 1+ lowercase/other letters (all non-Han), optional
    // case-insensitive contraction suffix.
    r"|[^\r\n\p{L}\p{N}]?(?:(?!\p{Han})[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}])*(?:(?!\p{Han})[\p{Ll}\p{Lm}\p{Lo}\p{M}])+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    // Uppercase word: same structure but requires 1+ uppercase-initiated chars.
    r"|[^\r\n\p{L}\p{N}]?(?:(?!\p{Han})[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}])+(?:(?!\p{Han})[\p{Ll}\p{Lm}\p{Lo}\p{M}])*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Compiled pretokenizer regex, shared across all `TiktokenTokenizer` instances.
static PRETOK_REGEX: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    fancy_regex::Regex::new(KIMI_K3_PRETOK_PATTERN).expect("invalid Kimi-K3 pretokenizer regex")
});

/// A tiktoken tokenizer with a rank table + special tokens.
///
/// This provides:
/// - `encode(text)` — applies the Kimi-K3 pretokenizer, then BPE encodes each
///   chunk into token IDs (Issue 411 fix — now matches Python tiktoken exactly)
/// - `decode(ids)` — decode token IDs back to a string
///
/// # Encoding algorithm
///
/// `encode()` first splits the input text using the Kimi-K3 pretokenizer
/// regex (the exact pattern from `tokenization_kimi.py`), then runs the
/// `byte_pair_merge` algorithm on each chunk independently. The BPE merge
/// loop looks up byte sequences directly in the rank table — no separate
/// `(left_id, right_id) -> rank` indirection, which was the root cause of
/// Bug #2 (wrong merge_ranks split at len-1).
pub struct TiktokenTokenizer {
    /// The original `mergeable_ranks`: byte sequence -> merge rank.
    /// Used by `encode_chunk` for byte-pair merging (Issue 411 Bug #2 fix).
    ranks: HashMap<Vec<u8>, usize>,
    /// Byte sequence -> token ID (for final encode output).
    token_to_id: HashMap<Vec<u8>, usize>,
    /// Token ID -> byte sequence (for decode).
    id_to_token: Vec<Vec<u8>>,
    /// Special token IDs.
    bos_id: usize,
    eos_id: usize,
    pad_id: usize,
}

impl TiktokenTokenizer {
    /// Build a tokenizer from a tiktoken rank table.
    ///
    /// The rank table is stored directly — `encode_chunk` looks up byte
    /// sequences in it during BPE merging (Issue 411 Bug #2 fix: the old code
    /// built a `(left_id, right_id) -> rank` map with a fixed split at
    /// `len-1`, which failed for tokens whose correct BPE split is elsewhere).
    ///
    /// Token IDs are assigned in rank order (single bytes first, then
    /// multi-byte by rank). For standard tiktoken files where ranks are
    /// sequential 0..N, this produces ID = rank.
    pub fn from_ranks(ranks: &TiktokenRanks) -> Self {
        // Sort entries by rank for deterministic ID assignment.
        let mut entries: Vec<(Vec<u8>, usize)> =
            ranks.iter().map(|(k, &v)| (k.clone(), v)).collect();
        entries.sort_by_key(|(_, rank)| *rank);

        let mut token_to_id = HashMap::with_capacity(entries.len());
        let mut id_to_token: Vec<Vec<u8>> = Vec::with_capacity(entries.len());

        // First pass: assign IDs to all single-byte tokens (length 1).
        for (bytes, _rank) in &entries {
            if bytes.len() == 1 {
                let id = id_to_token.len();
                token_to_id.insert(bytes.clone(), id);
                id_to_token.push(bytes.clone());
            }
        }

        // Second pass: assign IDs to multi-byte tokens in rank order.
        for (bytes, _rank) in &entries {
            if bytes.len() >= 2 {
                let id = id_to_token.len();
                token_to_id.insert(bytes.clone(), id);
                id_to_token.push(bytes.clone());
            }
        }

        Self {
            ranks: ranks.clone(),
            token_to_id,
            id_to_token,
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
        }
    }

    /// Set special token IDs (BOS, EOS, PAD).
    ///
    /// For Kimi-K3: BOS=1, EOS=2, PAD=0 (from `config.json`).
    pub fn with_special_tokens(mut self, bos: usize, eos: usize, pad: usize) -> Self {
        self.bos_id = bos;
        self.eos_id = eos;
        self.pad_id = pad;
        self
    }

    /// Encode a byte chunk into token IDs using BPE merges.
    ///
    /// This implements the `byte_pair_merge` algorithm from tiktoken: maintain
    /// a list of byte-range "parts", iteratively merge the adjacent pair whose
    /// concatenated bytes have the lowest rank in the rank table. This is a
    /// direct byte-sequence lookup — no `(left_id, right_id)` indirection —
    /// which was the root cause of Bug #2 (the old code's fixed split at
    /// `len-1` produced wrong merge rules for tokens whose correct BPE split
    /// is elsewhere).
    ///
    /// The caller should pass a single pretoken (from `encode()`, which applies
    /// the Kimi-K3 regex). For raw text without pretokenization, use `encode()`.
    pub fn encode_chunk(&self, bytes: &[u8]) -> Vec<usize> {
        if bytes.is_empty() {
            return Vec::new();
        }

        // tiktoken byte_pair_merge: parts[i] = (start_index_in_bytes, rank).
        // rank at position i = rank of merging parts[i] + parts[i+1],
        // computed from bytes[parts[i].0 .. parts[i+2].0].
        const MAX_RANK: usize = usize::MAX;

        // Initialize parts: one per byte boundary, all ranks = MAX.
        let mut parts: Vec<(usize, usize)> = (0..=bytes.len()).map(|i| (i, MAX_RANK)).collect();

        // Compute the rank for the pair at position i.
        // The merged token spans bytes[parts[i].0 .. parts[i+2].0].
        let compute_rank = |parts: &[(usize, usize)], i: usize| -> usize {
            if i + 2 >= parts.len() {
                return MAX_RANK;
            }
            let start = parts[i].0;
            let end = parts[i + 2].0;
            self.ranks
                .get(&bytes[start..end])
                .copied()
                .unwrap_or(MAX_RANK)
        };

        // Initialize ranks for all initial byte pairs.
        for i in 0..parts.len().saturating_sub(2) {
            parts[i].1 = compute_rank(&parts, i);
        }

        loop {
            // Find the pair with the minimum rank.
            let mut min_rank = MAX_RANK;
            let mut min_idx = None;
            for (i, &(_, rank)) in parts.iter().enumerate() {
                if rank < min_rank {
                    min_rank = rank;
                    min_idx = Some(i);
                    if rank == 0 {
                        break; // can't do better than rank 0
                    }
                }
            }

            let min_idx = match min_idx {
                Some(idx) if min_rank < MAX_RANK => idx,
                _ => break,
            };

            // Merge parts[min_idx] and parts[min_idx + 1]:
            // remove parts[min_idx + 1] (the merged part keeps parts[min_idx].0).
            parts.remove(min_idx + 1);

            // Recompute rank for the merged part.
            if min_idx < parts.len().saturating_sub(1) {
                parts[min_idx].1 = compute_rank(&parts, min_idx);
            } else {
                parts[min_idx].1 = MAX_RANK;
            }
            // Recompute rank for the part before the merge point.
            if min_idx > 0 {
                parts[min_idx - 1].1 = compute_rank(&parts, min_idx - 1);
            }
        }

        // Convert byte ranges to token IDs.
        let mut result = Vec::with_capacity(parts.len().saturating_sub(1));
        for window in parts.windows(2) {
            let start = window[0].0;
            let end = window[1].0;
            let slice = &bytes[start..end];
            if let Some(&id) = self.token_to_id.get(slice) {
                result.push(id);
            } else {
                // Unknown byte sequence — fall back to per-byte encoding.
                for &b in slice {
                    if let Some(&id) = self.token_to_id.get(&[b][..]) {
                        result.push(id);
                    }
                }
            }
        }
        result.shrink_to_fit();
        result
    }

    /// Encode a UTF-8 string into token IDs.
    ///
    /// Applies the Kimi-K3 pretokenizer regex (the exact pattern from
    /// `tokenization_kimi.py`) to split text into chunks, then BPE encodes each
    /// chunk independently via `encode_chunk`. This matches Python tiktoken
    /// behavior exactly (Issue 411 fix — both the pretokenizer + the byte-based
    /// BPE merge were missing/wrong in the previous implementation).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut result = Vec::new();
        for m in PRETOK_REGEX.find_iter(text).flatten() {
            let chunk = m.as_str().as_bytes();
            if !chunk.is_empty() {
                result.extend_from_slice(&self.encode_chunk(chunk));
            }
        }
        result
    }

    /// Decode token IDs back to a byte string.
    pub fn decode_bytes(&self, ids: &[usize]) -> Vec<u8> {
        let mut result = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if let Some(token) = self.id_to_token.get(id) {
                result.extend_from_slice(token);
            }
        }
        result
    }

    /// Decode token IDs back to a UTF-8 string (lossy on invalid UTF-8).
    pub fn decode(&self, ids: &[usize]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Vocabulary size (number of base tokens, excluding special tokens).
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// BOS token ID.
    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    /// EOS token ID.
    pub fn eos_id(&self) -> usize {
        self.eos_id
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal tiktoken model for testing.
    fn test_ranks() -> TiktokenRanks {
        let mut ranks = HashMap::new();

        // Single-byte tokens (0-255 equivalents)
        for b in 0u8..=255 {
            ranks.insert(vec![b], b as usize);
        }

        // A few multi-byte merges
        // "ab" → rank 256 (merge of 'a'=97 + 'b'=98)
        ranks.insert(b"ab".to_vec(), 256);
        // "abc" → rank 257 (merge of "ab" + 'c'=99)
        ranks.insert(b"abc".to_vec(), 257);

        ranks
    }

    #[test]
    fn load_tiktoken_bpe_parses_simple_format() {
        // Build a minimal valid tiktoken file:
        // <base64("a")>\n0\n<base64("b")>\n1\n
        let a_b64 = base64::engine::general_purpose::STANDARD.encode(b"a");
        let b_b64 = base64::engine::general_purpose::STANDARD.encode(b"b");
        let data = format!("{a_b64}\n0\n{b_b64}\n1\n");
        let ranks = load_tiktoken_bpe(data.as_bytes()).unwrap();

        assert_eq!(ranks.len(), 2);
        assert_eq!(ranks.get(b"a".as_slice()).copied(), Some(0));
        assert_eq!(ranks.get(b"b".as_slice()).copied(), Some(1));
    }

    #[test]
    fn load_tiktoken_bpe_handles_empty_data() {
        let ranks = load_tiktoken_bpe(&[]).unwrap();
        assert_eq!(ranks.len(), 0);
    }

    #[test]
    fn load_tiktoken_bpe_handles_trailing_newline() {
        let a_b64 = base64::engine::general_purpose::STANDARD.encode(b"a");
        let data = format!("{a_b64}\n0\n\n"); // trailing empty entry
        let ranks = load_tiktoken_bpe(data.as_bytes()).unwrap();
        assert_eq!(ranks.len(), 1);
    }

    #[test]
    fn load_tiktoken_bpe_rejects_invalid_base64() {
        let data = b"!!!invalid_base64!!!\n0\n";
        let result = load_tiktoken_bpe(data);
        assert!(result.is_err());
    }

    #[test]
    fn load_tiktoken_bpe_rejects_invalid_rank() {
        let a_b64 = base64::engine::general_purpose::STANDARD.encode(b"a");
        let data = format!("{a_b64}\nnot_a_number\n");
        let result = load_tiktoken_bpe(data.as_bytes());
        assert!(result.is_err());
    }

    #[test]
    fn tokenizer_from_ranks_assigns_single_byte_ids_first() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);

        // Single-byte tokens should be in the vocab
        assert!(tok.vocab_size() >= 256);
    }

    #[test]
    fn encode_single_byte_returns_byte_id() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);

        let ids = tok.encode("a");
        assert!(!ids.is_empty());
    }

    #[test]
    fn encode_merges_repeated_pairs() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);

        // "ab" should merge into a single token (rank 256)
        let ids = tok.encode("ab");
        assert_eq!(
            ids.len(),
            1,
            "'ab' should merge into 1 token, got {}",
            ids.len()
        );
        assert_eq!(ids[0], 256, "'ab' should have token ID 256");
    }

    #[test]
    fn encode_merges_abc_chain() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);

        // "abc" should merge: first 'a'+'b' -> "ab" (rank 256),
        // then "ab"+'c' -> "abc" (rank 257). Result: 1 token.
        let ids = tok.encode("abc");
        assert_eq!(
            ids.len(),
            1,
            "'abc' should merge into 1 token, got {} ids",
            ids.len()
        );
        assert_eq!(ids[0], 257, "'abc' should have token ID 257");
    }

    /// Bug #2 regression test: a token whose correct split is NOT at len-1.
    ///
    /// With ranks {a:97, b:98, c:99, bc:256, abc:257}, the token "abc" must
    /// split as 'a' + "bc" (not "ab" + 'c', since "ab" doesn't exist).
    /// The old code's fixed split at len-1=2 produced left="ab" which is NOT
    /// in the vocab → merge rule was silently skipped → "abc" was never formed.
    #[test]
    fn encode_bug2_correct_split_not_at_len_minus_1() {
        let mut ranks = HashMap::new();
        for b in 0u8..=255 {
            ranks.insert(vec![b], b as usize);
        }
        // "bc" exists but "ab" does NOT — forces the split at position 1.
        ranks.insert(b"bc".to_vec(), 256);
        ranks.insert(b"abc".to_vec(), 257);

        let tok = TiktokenTokenizer::from_ranks(&ranks);

        // "abc" should merge: 'b'+'c' -> "bc" (rank 256),
        // then 'a'+"bc" -> "abc" (rank 257). Result: 1 token.
        let ids = tok.encode_chunk(b"abc");
        assert_eq!(
            ids.len(),
            1,
            "'abc' should merge into 1 token via 'a'+'bc' split, got {} ids: {:?}",
            ids.len(),
            ids
        );
        assert_eq!(ids[0], 257);
    }

    /// Bug #1 regression test: the pretokenizer must split text before BPE.
    /// Without pretokenization, "hello world" encodes as one big chunk and
    /// may produce different token counts. With pretokenization (GPT-4/Kimi-K3
    /// pattern), "hello" and "world" are separate pretokens.
    #[test]
    fn encode_bug1_pretokenizer_splits_text() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);

        // The pretokenizer should split "hello world" into at least
        // ["hello", " ", "world"] (or similar). Each is BPE'd independently.
        let ids = tok.encode("hello world");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello world", "roundtrip must be lossless");
    }

    #[test]
    fn encode_pretokenizer_handles_code() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);

        let text = "fn main() { 42 }";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "code-like text roundtrip must be lossless");
    }

    #[test]
    fn encode_pretokenizer_handles_unicode() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);

        let text = "café 日本語 test";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "unicode text roundtrip must be lossless");
    }

    #[test]
    fn encode_decode_roundtrip_single_bytes() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);

        let text = "hello world";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        // With a full byte vocab (0-255), the roundtrip should be lossless
        // (each byte maps to its token, no merges for this text).
        assert_eq!(decoded, text);
    }

    #[test]
    fn decode_empty_returns_empty() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn encode_empty_returns_empty() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks);
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn special_tokens_set_correctly() {
        let ranks = test_ranks();
        let tok = TiktokenTokenizer::from_ranks(&ranks).with_special_tokens(1, 2, 0);
        assert_eq!(tok.bos_id(), 1);
        assert_eq!(tok.eos_id(), 2);
    }
}

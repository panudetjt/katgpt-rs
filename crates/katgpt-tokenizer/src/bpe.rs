use super::types::{BpeTokenizer, MergeRule};
use std::collections::HashMap;

/// BPE encoder/decoder implementation.
pub struct BpeTokenizerImpl;

impl BpeTokenizerImpl {
    /// Encode a string into token IDs using BPE merge rules.
    ///
    /// Hot-path design: operates on `Vec<usize>` (token IDs) end-to-end. The
    /// merge-rank lookup uses `merge_ranks_id: HashMap<(usize, usize), usize>`
    /// — no `String` allocation per pair. The replacement ID is resolved via
    /// `merge_target_id[rank]` — no `vocab_to_id` lookup per merge pass.
    ///
    /// Per AGENTS.md hot-loop rules: no allocation inside the merge loop.
    /// The only allocation is the initial char→ID map and the two ping-pong
    /// token buffers, both pre-sized.
    pub fn encode(tokenizer: &BpeTokenizer, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        // Map each char to its token ID up front. Unknown chars map to `unk`.
        // Uses a fixed-size stack buffer (`encode_utf8` writes ≤4 bytes) —
        // zero heap allocation for the entire char→ID map step.
        let unk = tokenizer.unk_id();
        let char_count = text.chars().count();
        let mut tokens: Vec<usize> = Vec::with_capacity(char_count);
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let s = c.encode_utf8(&mut buf);
            let id = tokenizer.vocab_to_id.get(s).copied().unwrap_or(unk);
            tokens.push(id);
        }

        // Fast path: no merges configured (or tables not rebuilt).
        if tokenizer.merge_ranks_id.is_empty() {
            return tokens;
        }

        let mut new_tokens: Vec<usize> = Vec::with_capacity(tokens.len());

        // Iteratively merge the highest-priority (lowest-rank) pair.
        loop {
            // Find the lowest-rank applicable merge across all adjacent pairs.
            // `windows(2)` lets LLVM drop the per-iteration bounds check on
            // `tokens[i + 1]` that the manual index loop forces.
            let mut best: Option<(usize, usize)> = None; // (rank, left_idx)
            for (i, w) in tokens.windows(2).enumerate() {
                if let Some(&rank) = tokenizer.merge_ranks_id.get(&(w[0], w[1])) {
                    match best {
                        Some((best_rank, _)) if best_rank <= rank => {}
                        _ => best = Some((rank, i)),
                    }
                }
            }

            let Some((best_rank, left_idx)) = best else {
                break;
            };

            // Resolve the merged token ID via the rank-indexed table — no
            // hashmap lookup, just a slice index.
            let merged_id = tokenizer.merge_target_id[best_rank];
            let left_id = tokens[left_idx];
            let right_id = tokens[left_idx + 1];

            // Apply the merge to all adjacent occurrences of (left, right).
            // Indices are `usize` (Copy) — zero allocation in this loop.
            new_tokens.clear();
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == left_id && tokens[i + 1] == right_id {
                    new_tokens.push(merged_id);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i]);
                    i += 1;
                }
            }
            std::mem::swap(&mut tokens, &mut new_tokens);
        }

        tokens.shrink_to_fit();
        tokens
    }

    /// Decode token IDs back to string.
    pub fn decode(tokenizer: &BpeTokenizer, ids: &[usize]) -> String {
        let mut result = String::with_capacity(ids.len() * 4); // estimate ~4 bytes per token
        for &id in ids {
            match tokenizer.id_to_vocab.get(id) {
                Some(s) => result.push_str(s),
                None => result.push('\u{fffd}'), // replacement character
            }
        }
        result
    }

    /// Fast BPE encode path backed by gigatoken's pure-Rust merge cores
    /// (Issue 191, Research 456).
    ///
    /// **Prefer [`FastBpeEncoder`] over this function** — it caches the
    /// `PairRankTable` once per tokenizer and reuses it across calls. This
    /// function rebuilds the table per call, which dominates the cost on
    /// short inputs (see `.benchmarks/191_fast_bpe_goat.md` G2 short-input
    /// failure). Use this only when you can't amortize a single encoder
    /// across calls.
    ///
    /// Equivalent in semantics to [`encode`]: same input text → same output
    /// token ID sequence. Differs in throughput: builds a [`PairRankTable`]
    /// (dense grid + flat packed open-addressed table) from the tokenizer's
    /// merge rules once on first call, then runs the heap + doubly-linked-list
    /// merge loop with stack-resident scratch. The single-pass table build is
    /// NOT amortized across calls in this function — see [`FastBpeEncoder`]
    /// for the amortized variant.
    ///
    /// # When to use this vs [`encode`] vs [`FastBpeEncoder`]
    ///
    /// - **Use [`FastBpeEncoder`]** when you call `encode_fast` more than
    ///   once with the same tokenizer. Pays the table build once, reuses it
    ///   across calls. This is the production path.
    /// - **Use `encode_fast`** (this function) for one-shot calls where
    ///   the input is long enough (≥~1KB) that the algorithmic win covers
    ///   the table rebuild. The GOAT gate (`tests/fast_bpe_goat.rs`)
    ///   measures 162× speedup on 64KB inputs even WITH the per-call table
    ///   rebuild — so this is a real win on long inputs.
    /// - **Use [`encode`]** for one-shot short inputs (≤~16 chars): the
    ///   iterative-merge loop has lower constants on tiny inputs and there's
    ///   no table build cost.
    ///
    /// # Fallback path
    ///
    /// If the tokenizer's vocab exceeds the 21-bit packed-key lane
    /// (`vocab > 2²¹`), the table build refuses and `encode_fast` falls back
    /// to the slow HashMap-keyed merge loop (still using the heap +
    /// linked-list core, which is faster than [`encode`]'s iterative merge on
    /// long sequences). The fallback is exercised by
    /// `tests/fast_bpe_goat.rs::g1_bit_identical_to_encode_with_table_fallback`.
    #[cfg(feature = "fast_bpe")]
    pub fn encode_fast(tokenizer: &BpeTokenizer, text: &str) -> Vec<usize> {
        FastBpeEncoder::from_tokenizer(tokenizer).encode(text)
    }
}

/// Cached fast-BPE encoder (Issue 191 Phase 2.5 — amortized `PairRankTable`).
///
/// Wraps a reference to a [`BpeTokenizer`] with a pre-built
/// [`PairRankTable`] + reusable merge scratch. Construct once per tokenizer
/// (the table build is the dominant cost on short inputs — see
/// `.benchmarks/191_fast_bpe_goat.md` G2 short-input failure for the
/// per-call `encode_fast` path), then call [`Self::encode`] as many times
/// as needed.
///
/// Semantics are bit-identical to [`BpeTokenizerImpl::encode`]. The win is
/// throughput on long inputs and across many calls.
///
/// # Fallback
///
/// If the tokenizer's vocab exceeds the 21-bit packed-key lane
/// (`vocab > 2²¹`), [`Self::from_tokenizer`] keeps `pair_ranks = None` and
/// `encode` falls back to the HashMap-keyed merge loop (still using the
/// heap + linked-list core, which is faster than the iterative-merge loop
/// on long sequences).
#[cfg(feature = "fast_bpe")]
pub struct FastBpeEncoder<'tok> {
    tokenizer: &'tok BpeTokenizer,
    /// Built once; reused across calls. `None` if the vocab is too large
    /// for the packed-key lane — `encode` falls back to the HashMap probe.
    pair_ranks: Option<crate::fast_bpe::PairRankTable>,
    /// `(TokenId, TokenId) → TokenId` map, kept for the fallback path and
    /// for table build. Owned here so the fallback closure can borrow it.
    merges: HashMap<(crate::fast_bpe::TokenId, crate::fast_bpe::TokenId), crate::fast_bpe::TokenId>,
    /// Reused across `encode` calls — no per-call allocation for the
    /// linked-list or merge heap after the first call.
    scratch: crate::fast_bpe::MergeScratch,
    /// Reused across `encode`/`encode_into` calls — the per-call `symbols`
    /// buffer (char→ID map input to the merge loop). Cleared + refilled
    /// each call; only reallocates when an input exceeds the prior peak.
    /// This is the Phase 2.5 G4 unblocker (Issue 191) — v1 allocated a
    /// fresh `Vec` per call.
    symbols: Vec<crate::fast_bpe::TokenId>,
    /// Short-pretoken cache (Issue 191 Phase 2.7 — replaces the Phase 2.6
    /// `HashMap<Vec<u8>, Vec<TokenId>>` stand-in with the vendored
    /// [`crate::fast_bpe::ShortPretokenCache`] substrate). Open-addressed +
    /// 2 MiB-aligned + prefetched; probes one cache line per lookup. Keys
    /// are packed `u128` (≤ 15 pretoken bytes + length tag) via
    /// [`crate::fast_bpe::pack_pretoken_key`]; hashes via
    /// [`crate::fast_bpe::pretoken_key_hash`] (hardware CRC32 where available).
    ///
    /// Values are inline-packed for the common case (≤ 2 merged tokens per
    /// pretoken, ~98% per upstream measurement on OWT): `val = (t0 << 32) |
    /// t1`, `ext = count` (1 or 2). Longer merged sequences spill into
    /// [`Self::long_values`]; pretokens > 15 bytes (too long for the inline
    /// key) spill into [`Self::long_pretokens`].
    short_cache: crate::fast_bpe::ShortPretokenCache,
    /// Spill storage for merged sequences longer than 2 tokens. Indexed by
    /// `ext` in the short-cache entry (sentinel `ext = u64::MAX` flags
    /// "spilled, look up here"). Monotonically grows; indices are stable.
    long_values: Vec<Box<[u32]>>,
    /// Spill storage for pretokens longer than 15 bytes (whose packed key
    /// doesn't fit `u128`). Plain `HashMap` — the long-pretoken case is rare
    /// (< 0.1% of natural-language pretokens), so the SipHash overhead is
    /// immaterial here. The fast path is [`Self::short_cache`].
    long_pretokens: HashMap<Vec<u8>, Box<[u32]>>,
    /// Reused across [`Self::encode_pretoken_tokens`] calls — the merged token
    /// IDs for the current pretoken. v1 returned a freshly allocated
    /// `Vec<u32>` on every short-cache miss.
    pretok_tokens: Vec<u32>,
}

#[cfg(feature = "fast_bpe")]
impl<'tok> FastBpeEncoder<'tok> {
    /// Build the encoder + `PairRankTable` (or fallback map) once. Pay this
    /// cost once per tokenizer; reuse across many `encode` calls.
    pub fn from_tokenizer(tokenizer: &'tok BpeTokenizer) -> Self {
        let mut merges: HashMap<
            (crate::fast_bpe::TokenId, crate::fast_bpe::TokenId),
            crate::fast_bpe::TokenId,
        > = HashMap::with_capacity(tokenizer.merge_ranks_id.len());
        for (&(l, r), &rank) in tokenizer.merge_ranks_id.iter() {
            // `merge_target_id[rank]` is the merged ID resolved at
            // `rebuild_ranks` time. `merge_ranks_id`'s rank values are
            // dense `0..merges.len()` by construction (see `rebuild_ranks`),
            // so the slice index is safe.
            let merged = tokenizer.merge_target_id[rank];
            merges.insert(
                (
                    crate::fast_bpe::TokenId(l as u32),
                    crate::fast_bpe::TokenId(r as u32),
                ),
                crate::fast_bpe::TokenId(merged as u32),
            );
        }
        let pair_ranks =
            crate::fast_bpe::PairRankTable::build(&merges, tokenizer.id_to_vocab.len()).ok();
        FastBpeEncoder {
            tokenizer,
            pair_ranks,
            merges,
            scratch: crate::fast_bpe::MergeScratch::default(),
            symbols: Vec::new(),
            // Start small (256 slots = 8 KB, fits in L1) and grow on demand
            // at the 3/4-load threshold. For corpus-scale use the table grows
            // to millions of slots via the vendored doubling logic; for small
            // tokenizers the upfront cost stays bounded.
            short_cache: crate::fast_bpe::ShortPretokenCache::with_pow2_capacity(256),
            long_values: Vec::new(),
            long_pretokens: HashMap::new(),
            pretok_tokens: Vec::new(),
        }
    }

    /// Encode `text` to token IDs, reusing the cached `PairRankTable` + scratch.
    /// Bit-identical to [`BpeTokenizerImpl::encode`].
    ///
    /// This allocates the returned `Vec<usize>` per call. For zero-alloc
    /// steady-state (e.g. hot loops, batched encoding), use
    /// [`Self::encode_into`] with a caller-owned reusable output buffer.
    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    /// Zero-alloc encode path — writes token IDs into the caller-owned `out`
    /// buffer (cleared first). After warmup (the first call seeds `symbols`
    /// and `scratch` to the prior peak), steady-state `encode_into` performs
    /// **zero heap allocations** for any input ≤ the prior peak size — the
    /// G4 gate (`tests/fast_bpe_goat.rs::g4_zero_alloc_steady_state`) audits
    /// this with a `CountingAllocator`.
    ///
    /// Bit-identical to [`BpeTokenizerImpl::encode`] + [`Self::encode`].
    pub fn encode_into(&mut self, text: &str, out: &mut Vec<usize>) {
        out.clear();
        if text.is_empty() {
            return;
        }

        // Map each char to its token ID up front — same as `encode`. The
        // `symbols` buffer is reused across calls; only reallocates if this
        // input exceeds the prior peak length.
        let unk = self.tokenizer.unk_id();
        let char_count = text.chars().count();
        self.symbols.clear();
        self.symbols.reserve(char_count);
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let s = c.encode_utf8(&mut buf);
            let id = self.tokenizer.vocab_to_id.get(s).copied().unwrap_or(unk);
            self.symbols.push(crate::fast_bpe::TokenId(id as u32));
        }

        // Fast path: no merges configured.
        if self.tokenizer.merge_ranks_id.is_empty() {
            out.extend(self.symbols.iter().map(|t| t.0 as usize));
            return;
        }

        if let Some(table) = &self.pair_ranks {
            crate::fast_bpe::bpe_merge_symbols_by_rank(table, &mut self.symbols, &mut self.scratch);
        } else {
            crate::fast_bpe::bpe_merge_symbols_by_rank_with_lookup(
                &|a, b| self.merges.get(&(a, b)).map_or(u32::MAX, |m| m.0),
                &mut self.symbols,
                &mut self.scratch,
            );
        }

        out.extend(self.symbols.iter().map(|t| t.0 as usize));
    }

    /// Pretokenized encode path.
    ///
    /// Issue 191 Phase 2.6 (2026-07-25) shipped whitespace pretokenization
    /// with a HashMap cache. Phase 2.7 (2026-07-25, same day) replaced the
    /// HashMap stand-in with the vendored `ShortPretokenCache` substrate
    /// (open-addressed + 2 MiB-aligned + prefetched). Issue 872 (2026-09-22)
    /// replaced the scalar per-char scan with the SIMD bitstream splitter
    /// ([`crate::fast_bpe::WhitespaceSplitter`] — SSE2/AVX2/NEON masks with a
    /// scalar per-char fallback for multibyte chars), removing the per-char
    /// accumulation buffer entirely: word events borrow their byte ranges
    /// straight from the input.
    ///
    /// **Bit-identical to [`BpeTokenizerImpl::encode`] + [`Self::encode`] +
    /// [`Self::encode_into`]** for any tokenizer trained by [`BpeTrainer`].
    /// The correctness invariant is structural: [`BpeTrainer::train`] learns
    /// merges via `corpus.split_whitespace()`, so no learned merge rule ever
    /// crosses a whitespace boundary or contains a whitespace char. Therefore
    /// encoding each non-whitespace run independently and emitting whitespace
    /// chars as inert single-char tokens produces the exact same sequence as
    /// whole-text encode. See `tests/fast_bpe_pretok_hypothesis.rs` +
    /// `tests/fast_bpe_goat_simd_split.rs` for the regression guards.
    ///
    /// # Wins vs [`Self::encode_into`]
    ///
    /// 1. **Structural win (always)**: each non-whitespace run is encoded
    ///    independently with a much smaller heap. The total merge work is
    ///    sum of O(k log k) per pretoken vs O(n log n) on the whole text —
    ///    a substantial reduction on natural language where pretokens are
    ///    short (avg ~5 chars).
    /// 2. **Cache-hit win (repeated pretokens)**: repeated words like "the",
    ///    "function", "return" hit the cache after first encoding, skipping
    ///    the merge loop entirely.
    ///
    /// # Cache hierarchy (Phase 2.7)
    ///
    /// The cache is the vendored [`crate::fast_bpe::ShortPretokenCache`]
    /// (open-addressed + 2 MiB-aligned + prefetched; one cache line per probe).
    /// Keys are packed `u128` (≤ 15 pretoken bytes + length tag); hashes use
    /// hardware CRC32 where available. Values are inline-packed for the
    /// common case (≤ 2 merged tokens, ~98% of pretokens on OWT):
    /// `val = (t0 << 32) | t1`, `ext = count`. Longer sequences spill into a
    /// side [`Vec<Box<[u32]>>`][Self::long_values]; pretokens > 15 bytes
    /// spill into a side [`HashMap`][Self::long_pretokens] (rare).
    ///
    /// # Allocation note
    ///
    /// This path is NOT zero-alloc in steady state on novel inputs: cache
    /// misses allocate the merged-token `Vec<u32>` (then moves it into the
    /// spill store or drops it after inlining). On repeated inputs (e.g.
    /// encoding the same document many times, or natural language with high
    /// word repetition) the cache hit rate climbs and allocation drops toward
    /// zero. For a guaranteed-zero-alloc path use [`Self::encode_into`].
    pub fn encode_into_pretok(&mut self, text: &str, out: &mut Vec<usize>) {
        out.clear();
        if text.is_empty() {
            return;
        }

        // Fast path: no merges configured — degenerates to char-by-char emit,
        // same as `encode_into`. Skip the pretokenization overhead.
        if self.tokenizer.merge_ranks_id.is_empty() {
            self.encode_into(text, out);
            return;
        }

        let unk = self.tokenizer.unk_id();
        let mut buf = [0u8; 4];

        // SIMD bitstream scan (Issue 872): maximal non-ws runs arrive as
        // borrowed byte slices (no per-char accumulation), identical-byte
        // ASCII whitespace runs amortize the vocab lookup to one per run,
        // and multibyte whitespace chars are classified by the scalar
        // per-char path using the exact `char::is_whitespace` predicate.
        // Event semantics == the previous scalar `chars()` loop; see the
        // `simd_split` module docs for the correctness contract.
        for event in crate::fast_bpe::WhitespaceSplitter::new(text) {
            match event {
                crate::fast_bpe::SplitEvent::Word(run) => {
                    self.flush_pretoken(run, out, &mut buf, unk);
                }
                crate::fast_bpe::SplitEvent::AsciiWsRun { byte, count } => {
                    // `byte < 0x80`, so `byte as char` encodes to exactly
                    // that one byte — the same single-char token the scalar
                    // loop emitted, with one lookup for the whole run.
                    let s = (byte as char).encode_utf8(&mut buf);
                    let id = self.tokenizer.vocab_to_id.get(s).copied().unwrap_or(unk);
                    for _ in 0..count {
                        out.push(id);
                    }
                }
                crate::fast_bpe::SplitEvent::MultibyteWs(c) => {
                    let s = c.encode_utf8(&mut buf);
                    let id = self.tokenizer.vocab_to_id.get(s).copied().unwrap_or(unk);
                    out.push(id);
                }
            }
        }
    }

    /// Encode one non-whitespace pretoken through the two-tier cache.
    ///
    /// Tier 1 (fast path, common case): pretoken ≤ 15 bytes → pack key into
    /// `u128` → probe [`Self::short_cache`]. On hit, decode the inline value
    /// (count 1 or 2) or follow the spill index into [`Self::long_values`].
    /// On miss, encode via the merge loop, then inline-pack (count ≤ 2) or
    /// spill (count ≥ 3) and insert.
    ///
    /// Tier 2 (rare): pretoken > 15 bytes → key doesn't fit `u128` → fall
    /// back to [`Self::long_pretokens`] (plain `HashMap`).
    ///
    /// `buf` is the caller's per-iter scratch for `char::encode_utf8`.
    #[inline]
    fn flush_pretoken(
        &mut self,
        pretoken_bytes: &[u8],
        out: &mut Vec<usize>,
        buf: &mut [u8; 4],
        unk: usize,
    ) {
        // Try the short-cache fast path (pretoken ≤ 15 bytes).
        if let Some(key) = crate::fast_bpe::pack_pretoken_key(pretoken_bytes) {
            // Key 0 (empty pretoken) is reserved as the short-cache empty
            // sentinel — pack_pretoken_key returns Some(0) for the empty
            // case. We never reach here with an empty `pretoken_bytes`
            // (flush_run! guards on `!is_empty()`), but the comment documents
            // why pack_pretoken_key's `Some(0)` return for empty input is
            // safe to pass straight to the short cache: the encode loop's
            // `!pretoken_bytes.is_empty()` check routes empty input to the
            // merge path's early return, not here.
            let h = crate::fast_bpe::pretoken_key_hash(key);
            match self.short_cache.get_or_slot(key, h) {
                Ok((val, ext)) => {
                    // Hit. Decode inline value.
                    self.emit_cached(val, ext, out);
                    return;
                }
                Err(slot) => {
                    // Miss. Encode the pretoken via the merge loop into the
                    // reusable `pretok_tokens` scratch (no per-miss alloc).
                    self.encode_pretoken_tokens(pretoken_bytes, buf, unk);
                    // Emit first (borrow the scratch), then store.
                    out.extend(self.pretok_tokens.iter().map(|&t| t as usize));
                    // `pretok_tokens` and `long_values` are disjoint fields, so
                    // the shared + unique borrows coexist.
                    let (val, ext) = Self::pack_value(&self.pretok_tokens, &mut self.long_values);
                    self.short_cache.insert_at(slot, key, h, val, ext);
                    return;
                }
            }
        }
        // Tier 2: long pretoken (> 15 bytes) → HashMap spill.
        if let Some(cached) = self.long_pretokens.get(pretoken_bytes) {
            out.extend(cached.iter().map(|&t| t as usize));
            return;
        }
        self.encode_pretoken_tokens(pretoken_bytes, buf, unk);
        out.extend(self.pretok_tokens.iter().map(|&t| t as usize));
        // Store in the long-pretoken map. Clone the key (pretoken_bytes is
        // borrowed from the caller's reuse buffer).
        let key_owned = pretoken_bytes.to_vec();
        let val_owned: Box<[u32]> = self.pretok_tokens.as_slice().into();
        self.long_pretokens.insert(key_owned, val_owned);
    }

    /// Encode a single pretoken's bytes through the merge loop, leaving the
    /// merged token IDs in [`Self::pretok_tokens`]. Reuses [`Self::symbols`] +
    /// [`Self::scratch`] + [`Self::pretok_tokens`] (no per-call allocation
    /// after warmup — v1 returned a freshly allocated `Vec<u32>`).
    ///
    /// # Panics
    ///
    /// Panics if `pretoken_bytes` is not valid UTF-8. The pretoken
    /// accumulation loop builds it from `char::encode_utf8` outputs, so it's
    /// always valid UTF-8 in practice.
    #[inline]
    fn encode_pretoken_tokens(&mut self, pretoken_bytes: &[u8], buf: &mut [u8; 4], unk: usize) {
        // SAFETY: `pretoken_bytes` is built from `char`s' UTF-8 encodings
        // in the caller's accumulation loop; qed.
        let s =
            std::str::from_utf8(pretoken_bytes).expect("pretoken_bytes is built from chars; qed");
        self.symbols.clear();
        for c in s.chars() {
            let cs = c.encode_utf8(buf);
            let id = self.tokenizer.vocab_to_id.get(cs).copied().unwrap_or(unk);
            self.symbols.push(crate::fast_bpe::TokenId(id as u32));
        }
        if let Some(table) = &self.pair_ranks {
            crate::fast_bpe::bpe_merge_symbols_by_rank(table, &mut self.symbols, &mut self.scratch);
        } else {
            crate::fast_bpe::bpe_merge_symbols_by_rank_with_lookup(
                &|a, b| self.merges.get(&(a, b)).map_or(u32::MAX, |m| m.0),
                &mut self.symbols,
                &mut self.scratch,
            );
        }
        // Same element order as the previous `.collect()`; writes into the
        // reused scratch instead of a fresh allocation.
        self.pretok_tokens.clear();
        self.pretok_tokens.extend(self.symbols.iter().map(|t| t.0));
    }

    /// Pack merged tokens into the (val, ext) inline encoding, spilling to
    /// `long_values` when the sequence doesn't fit inline (count ≥ 3).
    ///
    /// # Encoding
    ///
    /// - `count = 1`: `val = t0 as u64`, `ext = 1`.
    /// - `count = 2`: `val = (t0 << 32) | t1`, `ext = 2`.
    /// - `count ≥ 3`: `val = u64::MAX` (sentinel), `ext = index` into
    ///   `long_values`. The full sequence is copied into `long_values`.
    ///
    /// `ext` values 1 and 2 (the inline counts) never collide with spill
    /// indices because we always allocate spill indices ≥ 0 and disambiguate
    /// via `val == u64::MAX`. (A count-1 inline value with `t0 = u32::MAX`
    /// produces `val = u32::MAX as u64`, not `u64::MAX`, so no collision.)
    #[inline]
    fn pack_value(tokens: &[u32], long_values: &mut Vec<Box<[u32]>>) -> (u64, u64) {
        match tokens.len() {
            1 => (tokens[0] as u64, 1),
            2 => (((tokens[0] as u64) << 32) | tokens[1] as u64, 2),
            _ => {
                let idx = long_values.len();
                long_values.push(tokens.to_vec().into_boxed_slice());
                (u64::MAX, idx as u64)
            }
        }
    }

    /// Emit a cached (val, ext) pair into `out`. The inverse of
    /// [`Self::pack_value`].
    ///
    /// Disambiguates the spill sentinel FIRST: `val == u64::MAX` means
    /// "spilled, `ext` is the index into `long_values`", regardless of
    /// `ext`'s value. Without this check, a spill at index 1 or 2 would
    /// collide with the inline-count encoding (`ext == 1` or `ext == 2`).
    #[inline]
    fn emit_cached(&self, val: u64, ext: u64, out: &mut Vec<usize>) {
        if val == u64::MAX {
            // Spill: ext is the index into long_values.
            let idx = ext as usize;
            let tokens = &self.long_values[idx];
            out.extend(tokens.iter().map(|&t| t as usize));
            return;
        }
        match ext {
            1 => out.push(val as u32 as usize),
            2 => {
                out.push((val >> 32) as u32 as usize);
                out.push(val as u32 as usize);
            }
            _ => unreachable!(
                "ext is 1, 2, or spill-flagged via val==u64::MAX; got val={val:#x} ext={ext}"
            ),
        }
    }

    /// Diagnostic: number of entries in the pretoken cache (Issue 191 Phase 2.6).
    /// After encoding a natural-language corpus this should grow toward the
    /// unique-word count of the corpus. The hit rate is the perf signal: high
    /// hit rate = the pretokenized path is paying off.
    ///
    /// Counts both the short-cache entries (≤ 15-byte pretokens) and the
    /// long-pretoken spill-map entries (> 15-byte pretokens, rare).
    #[doc(hidden)]
    pub fn pretoken_cache_len(&self) -> usize {
        self.short_cache.len() + self.long_pretokens.len()
    }

    /// Diagnostic: current capacity of the internal `symbols` scratch buffer.
    /// After warmup this reflects the peak input size the encoder has seen.
    /// Exposed so callers in hot loops can confirm steady-state capacity has
    /// been reached before relying on the G4 zero-alloc contract.
    #[doc(hidden)]
    pub fn symbols_capacity(&self) -> usize {
        self.symbols.capacity()
    }
}

/// BPE trainer: learns merge rules from a corpus.
pub struct BpeTrainer;

impl BpeTrainer {
    /// Train a BPE tokenizer from a text corpus.
    ///
    /// `vocab_size`: target vocabulary size (including special tokens).
    /// `corpus`: training text.
    ///
    /// # Algorithm
    ///
    /// Standard greedy BPE: count adjacent pairs, merge the most frequent,
    /// repeat until `vocab_size` is reached or no pair appears ≥ 2 times.
    ///
    /// # Tie-breaking (deterministic)
    ///
    /// When multiple pairs tie at the winning count, the **lexicographically
    /// smallest `(left, right)`** is selected. This is an explicit, stable
    /// rule (Issue 192 — the previous implementation used
    /// `HashMap::drain().max_by_key()` which depended on HashMap's random
    /// seed and produced different merge sequences across process runs).
    ///
    /// # Complexity
    ///
    /// `O(N · W · T)` where `N = num_merges`, `W = word count`,
    /// `T = avg tokens/word`. Each round scans the (memoized) per-word
    /// tokenization once and applies only the new merge — Issue 192 fixed
    /// the prior O(N² · W · T) implementation that re-applied all prior
    /// merges from scratch on every round.
    ///
    /// # ID-indexed training loop
    ///
    /// The hot path operates on `Vec<Vec<usize>>` (token IDs), not
    /// `Vec<Vec<String>>`. Pair counting and merge application are pure
    /// integer arithmetic — no `String` allocation or `clone()` per pair.
    /// `String`s are only materialized for the final `MergeRule` table and
    /// `vocab_to_id` map (one allocation per learned merge + per unique
    /// char, not per pair-per-round). On the Issue 192 perf-smoke corpus
    /// (~4.3 KB / 512 merges), this cuts allocations in the inner loops
    /// from O(N·W·T·L) (L = avg string length) to O(N·W·T) integers copied.
    pub fn train(corpus: &str, vocab_size: usize) -> BpeTokenizer {
        // Pre-allocate: 4 special tokens + up to 256 unique byte-chars + merges.
        let cap = 4usize.saturating_add(vocab_size).min(corpus.len() + 4);
        let mut vocab_to_id: HashMap<String, usize> = HashMap::with_capacity(cap);
        let mut id_to_vocab: Vec<String> = Vec::with_capacity(cap);

        // Special tokens: <pad>=0, <bos>=1, <eos>=2, <unk>=3
        const SPECIAL_TOKENS: [&str; 4] = ["<pad>", "<bos>", "<eos>", "<unk>"];
        for (i, tok) in SPECIAL_TOKENS.iter().enumerate() {
            vocab_to_id.insert((*tok).to_string(), i);
            id_to_vocab.push((*tok).to_string());
        }

        // Add all unique characters from corpus.
        // Use the `entry` API to avoid the double-lookup (contains_key + insert).
        for ch in corpus.chars() {
            vocab_to_id.entry(ch.to_string()).or_insert_with(|| {
                let id = id_to_vocab.len();
                id_to_vocab.push(ch.to_string());
                id
            });
        }

        let mut merges: Vec<MergeRule> = Vec::new();
        let num_merges = vocab_size.saturating_sub(id_to_vocab.len());

        // Memoized per-word tokenization state — ID-indexed (NOT String).
        // Each token is the `id_to_vocab` index of the char/subword. Pair
        // counting and merge application become integer arithmetic — no
        // String clone() per pair, no String allocation in the merge loop.
        // This is the Issue 192 fix (each round applies only the NEW merge
        // in-place) layered on top of the ID-indexing refactor.
        let mut words_tok: Vec<Vec<usize>> = corpus
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .map(|c| {
                        // Char tokens were inserted above; lookup never misses.
                        // Inline the buffer to avoid the per-lookup alloc.
                        let mut buf = [0u8; 4];
                        let s = c.encode_utf8(&mut buf);
                        vocab_to_id[s]
                    })
                    .collect()
            })
            .collect();

        // Per-merge scratch buffers (reused across rounds — G4 alloc discipline).
        let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
        let mut scratch: Vec<usize> = Vec::new();

        for _ in 0..num_merges {
            // Count all adjacent pairs in the current memoized state.
            // IDs are `usize` (Copy) — zero allocation per pair.
            pair_counts.clear();
            for word in &words_tok {
                if word.len() < 2 {
                    continue;
                }
                for i in 0..word.len() - 1 {
                    let pair = (word[i], word[i + 1]);
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            // Pick the winning pair with an EXPLICIT tie-break rule:
            // highest count wins; on tie, lexicographically smallest
            // `(left_str, right_str)` wins. This is deterministic regardless
            // of HashMap iteration order (Issue 192).
            //
            // IDs map 1:1 to vocab strings (id_to_vocab[id]); since IDs are
            // assigned in insertion order, two tokens that share an ID always
            // share a string — comparing by string recovers lexicographic
            // order. (We can't compare by ID directly: ID order is insertion
            // order, not lexicographic order.)
            //
            // Hot-loop optimization: the prior `min_by_key` form built a
            // `(Reverse(c), l_str, r_str)` tuple — and thus read two strings
            // from `id_to_vocab` — for EVERY entry, including entries whose
            // count was already beaten by an earlier one. The explicit loop
            // below skips the string reads entirely when `count < best_count`,
            // which is the common case after the first few iterations of the
            // outer training loop establish a high best_count. The string
            // reads are deferred to the tie-break branch only.
            let mut best_pair = (0usize, 0usize);
            let mut best_count = 0usize;
            let mut best_key: (&str, &str) = ("", "");
            let mut found = false;
            for (&pair, &count) in &pair_counts {
                // Fast reject: any count below the current best cannot win,
                // and we don't need the strings to know that. This skips two
                // id_to_vocab slice reads on the common path.
                if found && count < best_count {
                    continue;
                }
                let (l_id, r_id) = pair;
                // Safe direct index: ids in `pair_counts` were inserted from
                // `words_tok`, whose ids are valid `id_to_vocab` indices by
                // construction. Skip the bounds check that `.get()` would add.
                let l_str = id_to_vocab[l_id].as_str();
                let r_str = id_to_vocab[r_id].as_str();
                let take = if !found || count > best_count {
                    true
                } else {
                    // Same count → lexicographic tie-break on (l_str, r_str).
                    (l_str, r_str) < best_key
                };
                if take {
                    best_pair = pair;
                    best_count = count;
                    best_key = (l_str, r_str);
                    found = true;
                }
            }
            // `pair_counts` was checked non-empty above; the loop sets `found`.
            debug_assert!(found, "pair_counts was non-empty but no best found");
            let (left_id, right_id) = best_pair;
            let count = best_count;

            if count < 2 {
                break; // Stop if no pair appears more than once
            }

            // Resolve merged token string + add to vocabulary (one String
            // allocation per learned merge — NOT per pair-per-round).
            let left = id_to_vocab[left_id].clone();
            let right = id_to_vocab[right_id].clone();
            let merged = format!("{left}{right}");

            let merged_id = *vocab_to_id.entry(merged.clone()).or_insert_with(|| {
                let id = id_to_vocab.len();
                id_to_vocab.push(merged.clone());
                id
            });

            merges.push(MergeRule {
                left: left.clone(),
                right: right.clone(),
                merged: merged.clone(),
            });

            // Apply ONLY this merge in-place to each word's tokenization.
            // Pure integer writes — no String clone() per token.
            //
            // Hot-loop optimization: skip words that don't contain `left_id`
            // at all. The full copy-to-scratch + swap path costs O(word_len)
            // even for words that won't change; `contains(&left_id)` short-
            // circuits on first match. For most merges the majority of words
            // don't contain the left token, so this skips most of the copies.
            for word in &mut words_tok {
                if word.len() < 2 || !word.contains(&left_id) {
                    continue;
                }
                scratch.clear();
                let mut i = 0;
                let len = word.len();
                while i < len {
                    if i + 1 < len && word[i] == left_id && word[i + 1] == right_id {
                        scratch.push(merged_id);
                        i += 2;
                    } else {
                        scratch.push(word[i]);
                        i += 1;
                    }
                }
                std::mem::swap(word, &mut scratch);
            }
        }

        let mut tokenizer = BpeTokenizer {
            vocab_to_id,
            id_to_vocab,
            merges,
            merge_ranks: HashMap::new(),
            merge_ranks_id: HashMap::new(),
            merge_target_id: Vec::new(),
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
        };
        tokenizer.rebuild_ranks();
        tokenizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let corpus = "hello world hello rust";
        let tokenizer = BpeTrainer::train(corpus, 64);
        let text = "hello";
        let ids = BpeTokenizerImpl::encode(&tokenizer, text);
        let decoded = BpeTokenizerImpl::decode(&tokenizer, &ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_special_tokens() {
        let tokenizer = BpeTrainer::train("abc", 32);
        assert_eq!(tokenizer.pad_id, 0);
        assert_eq!(tokenizer.bos_id, 1);
        assert_eq!(tokenizer.eos_id, 2);
        // unk_id is the last vocab entry
        assert!(tokenizer.unk_id() >= 3);
        // Verify special tokens in vocab
        assert_eq!(tokenizer.vocab_to_id["<pad>"], 0);
        assert_eq!(tokenizer.vocab_to_id["<bos>"], 1);
        assert_eq!(tokenizer.vocab_to_id["<eos>"], 2);
        assert_eq!(tokenizer.vocab_to_id["<unk>"], 3);
    }

    #[test]
    fn test_bpe_train_produces_merges() {
        // Use a corpus with repeated patterns to guarantee merges
        let corpus = "ab ab ab ab ab ab ab ab ab ab";
        let tokenizer = BpeTrainer::train(corpus, 64);
        // "a" + "b" → "ab" should be learned as a merge
        assert!(
            !tokenizer.merges.is_empty(),
            "Expected at least one merge rule from repeated 'ab' patterns"
        );
        // Verify the merge exists
        let has_ab_merge = tokenizer
            .merges
            .iter()
            .any(|m| m.left == "a" && m.right == "b" && m.merged == "ab");
        assert!(has_ab_merge, "Expected merge rule 'a'+'b'→'ab'");
    }

    #[test]
    fn test_bpe_vocab_coverage() {
        let corpus = "hello world";
        let tokenizer = BpeTrainer::train(corpus, 64);
        // All characters from the corpus must be in the vocabulary
        for ch in corpus.chars() {
            let s = ch.to_string();
            assert!(
                tokenizer.vocab_to_id.contains_key(&s),
                "Character '{s}' missing from vocabulary"
            );
        }
    }

    #[test]
    fn test_bpe_encode_empty() {
        let tokenizer = BpeTrainer::train("hello", 32);
        let ids = BpeTokenizerImpl::encode(&tokenizer, "");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_bpe_decode_unknown_id() {
        let tokenizer = BpeTrainer::train("hello", 32);
        // Use an out-of-range ID
        let decoded = BpeTokenizerImpl::decode(&tokenizer, &[9999]);
        assert_eq!(decoded, "�");
    }

    // ─── Issue 192 tests: O(N²) perf fix + tie-break determinism ────────────

    /// Frozen reference: corpora with NO ties at the winning count should
    /// produce the exact same merge sequence every time. These two corpora
    /// were verified deterministic on the PRE-Issue-192 implementation too
    /// (run 5× each in a probe — all runs agreed), so they're a safe
    /// regression guard for the rewrite.
    #[test]
    fn test_bpe_train_frozen_reference_no_ties() {
        // 'a'+'b' is the only pair that appears ≥ 2 times → uniquely maximal.
        let tokenizer = BpeTrainer::train("ab ab ab ab ab ab ab ab ab ab", 64);
        assert_eq!(tokenizer.merges.len(), 1);
        assert_eq!(tokenizer.merges[0].left, "a");
        assert_eq!(tokenizer.merges[0].right, "b");
        assert_eq!(tokenizer.merges[0].merged, "ab");

        // 'a'+'b' (count 5) dominates 'x'+'y' (count 2) → unique first pick.
        // After the merge, 'x'+'y' (count 2) is the only remaining pair.
        let tokenizer = BpeTrainer::train("ab ab ab ab ab xy xy", 32);
        assert_eq!(tokenizer.merges.len(), 2);
        assert_eq!(tokenizer.merges[0].left, "a");
        assert_eq!(tokenizer.merges[0].right, "b");
        assert_eq!(tokenizer.merges[0].merged, "ab");
        assert_eq!(tokenizer.merges[1].left, "x");
        assert_eq!(tokenizer.merges[1].right, "y");
        assert_eq!(tokenizer.merges[1].merged, "xy");
    }

    /// Tie-break determinism: when multiple pairs tie at the winning count,
    /// the trainer MUST pick the lexicographically smallest `(left, right)`.
    /// This is the Issue 192 fix — the prior implementation picked whichever
    /// HashMap happened to iterate last, which varied per process.
    ///
    /// Corpus "ab ab ab cd cd cd" has two pairs at count 3: ('a','b') and
    /// ('c','d'). Lexicographically, ('a','b') < ('c','d'), so ('a','b')
    /// MUST be picked first.
    #[test]
    fn test_bpe_train_tie_break_is_lexicographic() {
        let tokenizer = BpeTrainer::train("ab ab ab cd cd cd", 32);
        assert!(!tokenizer.merges.is_empty());
        assert_eq!(tokenizer.merges[0].left, "a");
        assert_eq!(tokenizer.merges[0].right, "b");
        assert_eq!(tokenizer.merges[0].merged, "ab");
    }

    /// Cross-run stability: train the same corpus 5 times and verify ALL
    /// 5 runs produce the exact same merge sequence. This is the property
    /// the Issue 192 fix establishes — the prior implementation would
    /// intermittently fail this on tie-bearing corpora.
    #[test]
    fn test_bpe_train_deterministic_across_runs() {
        // A corpus that has ties on multiple rounds (the 'hello hello hello'
        // probe showed ties at idx 0 — could pick ('e','l') OR ('l','l')).
        let corpus = "hello hello hello hello";
        let runs: Vec<_> = (0..5)
            .map(|_| {
                let tok = BpeTrainer::train(corpus, 64);
                (
                    tok.merges
                        .iter()
                        .map(|m| (m.left.clone(), m.right.clone(), m.merged.clone()))
                        .collect::<Vec<_>>(),
                    tok.vocab_to_id.clone(),
                )
            })
            .collect();
        for r in runs.iter().skip(1) {
            assert_eq!(r.0, runs[0].0, "merge sequence diverged across runs");
            assert_eq!(r.1, runs[0].1, "vocab diverged across runs");
        }
    }

    /// Issue 192 perf smoke: training on a moderately-sized corpus with a
    /// 1024-vocab target should complete in well under a second. The prior
    /// O(N²) implementation took ~50ms on this corpus; the new O(N) one
    /// should take <5ms. The gate is generous (5s) to avoid CI flakiness
    /// on slow machines — the goal is to catch an accidental O(N²)
    /// regression, not to measure perf precisely.
    #[test]
    fn test_bpe_train_perf_smoke_o_n_not_o_n2() {
        let corpus = "the quick brown fox jumps over the lazy dog ".repeat(100); // ~4.3 KB
        let start = std::time::Instant::now();
        let tokenizer = BpeTrainer::train(&corpus, 512);
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 5,
            "train took {elapsed:?} on a 4.3 KB corpus + 512-vocab target; \
             likely an O(N²) regression (Issue 192 baseline ~50ms)",
        );
        // Sanity: the corpus has enough repetition to produce many merges.
        assert!(!tokenizer.merges.is_empty());
    }
}

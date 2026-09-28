//! SIMD bitstream whitespace splitter (Issue 872) — the scan half of
//! [`crate::FastBpeEncoder::encode_into_pretok`].
//!
//! # Technique class
//!
//! Bit-parallel byte classification over SIMD registers: a chunk of 16/32
//! bytes is classified in a handful of vector ops and reduced to a bitmask
//! (`movemask`), and run boundaries fall out of `trailing_zeros`/`trailing_ones`
//! on the mask instead of a per-char branchy scan. The technique class is
//! Parabix / simdjson; the immediate inspiration is the "bitcannon" splitter
//! shipped in HuggingFace `tokenizers` v1 (blog "tokenizers-v1: encode, decode
//! and scaling, measured", 2026-09-21), which credits gigatoken — the same
//! upstream this crate vendored its merge cores from (Issue 191, Research 456).
//! This module is original code implementing the *technique* for the ONE
//! splitter class this crate defines (Unicode whitespace), not a port of
//! bitcannon's regex grammars (GPT-2/cl100k/o200k/…), which belong to
//! tokenizer formats this crate does not load.
//!
//! # Correctness contract
//!
//! Event semantics are **identical** to the previous scalar loop
//! (`for c in text.chars() { if c.is_whitespace() .. }`):
//!
//! - [`SplitEvent::Word`] is a MAXIMAL run of non-whitespace bytes. Merges
//!   can cross ASCII/multibyte boundaries inside a word (the trainer's
//!   `split_whitespace()` words may contain any non-ws chars), so words are
//!   never split into sub-events.
//! - Every whitespace char becomes its own token; a run of IDENTICAL ASCII
//!   whitespace bytes coalesces into one [`SplitEvent::AsciiWsRun`] because
//!   the same byte encodes to the same 1-char token (one vocab lookup
//!   amortized over the run). Mixed whitespace (`" \t"`) never coalesces.
//! - Any byte ≥ 0x80 is classified by the scalar per-char path using the
//!   exact `char::is_whitespace` predicate, so the full Unicode
//!   `White_Space` set (U+00A0, U+1680, U+2000–200A, U+2028/9, U+202F,
//!   U+205F, U+3000, …) classifies bit-identically to the old loop.
//!
//! # The six-byte trap
//!
//! ASCII `White_Space` (Unicode) is **six** bytes: `{0x09..=0x0D, 0x20}` —
//! including vertical tab `0x0B`. `u8::is_ascii_whitespace` is FIVE (it
//! excludes `0x0B`) and must NOT be used here; the scalar reference and the
//! SIMD predicate both use [`is_ascii_ws_byte`], which matches
//! `char::is_whitespace` exactly on the ASCII range.
//!
//! # Dispatch (the `shipped_target_feature_gate` law)
//!
//! Never select a fast arm on a compile-time `target_feature`. x86_64:
//! SSE2 is baseline (always safe) and AVX2 is chosen at runtime via
//! `std::is_x86_feature_detected!`; the AVX2 kernel carries
//! `#[target_feature(enable = "avx2")]` so it compiles on every ordinary
//! build and is only *called* on cores that have it. aarch64: NEON is
//! baseline. Every other target (including wasm32) runs the scalar level
//! through the same iterator — identical semantics, today's speed.

/// ASCII whitespace predicate matching `char::is_whitespace` exactly on the
/// ASCII range: Unicode `White_Space` ∩ ASCII = `{U+0009..=U+000D, U+0020}`.
///
/// NOT `u8::is_ascii_whitespace` — that one excludes vertical tab `0x0B`.
#[inline]
pub(crate) fn is_ascii_ws_byte(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(0x09) <= 4
}

/// One split event — a maximal non-ws byte run, or a whitespace emission.
pub(crate) enum SplitEvent<'a> {
    /// Maximal run of non-whitespace bytes. Valid UTF-8 (sliced from the
    /// input `&str`); may mix ASCII and multibyte chars.
    Word(&'a [u8]),
    /// Run of `count` copies of one ASCII whitespace byte (`byte < 0x80`).
    /// Coalescing identical bytes is exact: the same byte is the same
    /// 1-char token repeated `count` times.
    AsciiWsRun { byte: u8, count: u32 },
    /// One multibyte whitespace char (classified by `char::is_whitespace`
    /// on the scalar path).
    MultibyteWs(char),
}

/// Scan level. Variants exist only where their kernel exists; the probe
/// ([`probe_level`]) picks the widest safe level for the current core.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SplitLevel {
    /// 16-wide scalar mask fill — the reference semantics + fallback for
    /// targets with no SIMD. On SIMD targets it is constructed only by the
    /// differential tests (`cfg(test)`); the allow covers the non-test lib
    /// build there.
    #[allow(dead_code)]
    Scalar,
    /// 16-byte SSE2 chunks (x86_64 baseline — no runtime probe needed).
    #[cfg(target_arch = "x86_64")]
    Sse2,
    /// 32-byte AVX2 chunks (x86_64, runtime-probed).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// 16-byte NEON chunks (aarch64 baseline).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl SplitLevel {
    #[inline]
    fn chunk_len(self) -> usize {
        match self {
            SplitLevel::Scalar => 16,
            #[cfg(target_arch = "x86_64")]
            SplitLevel::Sse2 => 16,
            #[cfg(target_arch = "x86_64")]
            SplitLevel::Avx2 => 32,
            #[cfg(target_arch = "aarch64")]
            SplitLevel::Neon => 16,
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn probe_level() -> SplitLevel {
    // `is_x86_feature_detected!` caches its answer in a std static; calling
    // per splitter construction is fine.
    if std::is_x86_feature_detected!("avx2") {
        SplitLevel::Avx2
    } else {
        SplitLevel::Sse2
    }
}

#[cfg(target_arch = "aarch64")]
fn probe_level() -> SplitLevel {
    SplitLevel::Neon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn probe_level() -> SplitLevel {
    SplitLevel::Scalar
}

/// Splits `text` into [`SplitEvent`]s without allocating: word events borrow
/// their byte ranges from the input.
pub(crate) struct WhitespaceSplitter<'a> {
    bytes: &'a [u8],
    pos: usize,
    level: SplitLevel,
}

impl<'a> WhitespaceSplitter<'a> {
    /// Split at the widest safe SIMD level for this core.
    pub(crate) fn new(text: &'a str) -> Self {
        Self::with_level(text, probe_level())
    }

    /// Split at an explicit level (differential tests + the scalar
    /// reference).
    pub(crate) fn with_level(text: &'a str, level: SplitLevel) -> Self {
        WhitespaceSplitter {
            bytes: text.as_bytes(),
            pos: 0,
            level,
        }
    }

    /// Bit i (little-endian over the chunk) = `is_ascii_ws_byte(b) || b >= 0x80`
    /// for chunk byte i. `chunk.len() == self.level.chunk_len()`.
    fn stop_mask(&self, chunk: &[u8]) -> u32 {
        match self.level {
            SplitLevel::Scalar => {
                let mut m = 0u32;
                for (i, &b) in chunk.iter().enumerate() {
                    if is_ascii_ws_byte(b) || b >= 0x80 {
                        m |= 1 << i;
                    }
                }
                m
            }
            #[cfg(target_arch = "x86_64")]
            SplitLevel::Sse2 => sse2_stop_mask16(chunk),
            #[cfg(target_arch = "x86_64")]
            SplitLevel::Avx2 => unsafe { avx2_stop_mask32(chunk) },
            #[cfg(target_arch = "aarch64")]
            SplitLevel::Neon => neon_stop_mask16(chunk),
        }
    }

    /// Bit i = `chunk[i] == byte`. `chunk.len() == self.level.chunk_len()`.
    fn eq_mask(&self, chunk: &[u8], byte: u8) -> u32 {
        match self.level {
            SplitLevel::Scalar => {
                let mut m = 0u32;
                for (i, &b) in chunk.iter().enumerate() {
                    if b == byte {
                        m |= 1 << i;
                    }
                }
                m
            }
            #[cfg(target_arch = "x86_64")]
            SplitLevel::Sse2 => sse2_eq_mask16(chunk, byte),
            #[cfg(target_arch = "x86_64")]
            SplitLevel::Avx2 => unsafe { avx2_eq_mask32(chunk, byte) },
            #[cfg(target_arch = "aarch64")]
            SplitLevel::Neon => neon_eq_mask16(chunk, byte),
        }
    }

    /// Decode the char at `self.pos` and advance past it.
    ///
    /// # SAFETY (caller)
    ///
    /// `self.pos < self.bytes.len()` and `self.pos` is a char boundary.
    /// Both hold by construction: `pos` starts at 0 and only ever advances
    /// past ASCII bytes (< 0x80 ⇒ boundary) or by `len_utf8()` past a
    /// decoded char.
    #[inline]
    fn decode_char_at(&mut self) -> char {
        // SAFETY: see the contract above — the slice is a suffix of a valid
        // `&str` starting at a char boundary.
        let rest = unsafe { std::str::from_utf8_unchecked(&self.bytes[self.pos..]) };
        let c = rest.chars().next().expect("pos < len checked by caller");
        self.pos += c.len_utf8();
        c
    }

    /// Extend a whitespace run of the identical ASCII `byte` from `self.pos`.
    fn extend_ws_run(&mut self, byte: u8) {
        let w = self.level.chunk_len();
        // u64 arithmetic: `1u32 << 32` (the AVX2 width) overflows — in
        // release it wraps to `1 << 0`, making `full == 0` and every
        // NON-matching chunk read as all-matching.
        let full: u32 = ((1u64 << w) - 1) as u32;
        loop {
            if self.bytes.len() - self.pos < w {
                break;
            }
            let eq = self.eq_mask(&self.bytes[self.pos..self.pos + w], byte);
            if eq == full {
                self.pos += w;
            } else {
                // Leading ones of `eq` = run length within this chunk.
                self.pos += (!eq).trailing_zeros() as usize;
                break;
            }
        }
        // Scalar tail (remainder < w, or the run ended inside a chunk).
        while self.bytes.get(self.pos) == Some(&byte) {
            self.pos += 1;
        }
    }

    /// Extend a word from `self.pos` (currently at/inside a non-ws byte or
    /// just past a non-ws multibyte char) to the next whitespace char or EOF.
    /// `start` is the word's first byte index.
    fn extend_word(&mut self, start: usize) -> SplitEvent<'a> {
        loop {
            // SIMD section: consume whole chunks containing no stop bytes.
            let w = self.level.chunk_len();
            while self.bytes.len() - self.pos >= w {
                let mask = self.stop_mask(&self.bytes[self.pos..self.pos + w]);
                if mask == 0 {
                    self.pos += w;
                } else {
                    self.pos += mask.trailing_zeros() as usize;
                    break; // at a stop byte (ws or non-ASCII) — scalar step decides
                }
            }
            // Scalar section: exactly one byte/char, then loop back to SIMD.
            match self.bytes.get(self.pos) {
                None => return SplitEvent::Word(&self.bytes[start..self.pos]),
                Some(&b) if b < 0x80 => {
                    if is_ascii_ws_byte(b) {
                        return SplitEvent::Word(&self.bytes[start..self.pos]);
                    }
                    self.pos += 1;
                }
                Some(_) => {
                    let ws_start = self.pos;
                    let c = self.decode_char_at();
                    if c.is_whitespace() {
                        // Rewind: `decode_char_at` advanced past the ws char,
                        // but its event belongs to `next()`'s multibyte
                        // classifier. The word ends BEFORE its first byte.
                        self.pos = ws_start;
                        return SplitEvent::Word(&self.bytes[start..ws_start]);
                    }
                }
            }
        }
    }
}

impl<'a> Iterator for WhitespaceSplitter<'a> {
    type Item = SplitEvent<'a>;

    #[inline]
    fn next(&mut self) -> Option<SplitEvent<'a>> {
        let b = *self.bytes.get(self.pos)?;
        if b < 0x80 {
            if is_ascii_ws_byte(b) {
                let start = self.pos;
                self.pos += 1;
                self.extend_ws_run(b);
                let count = (self.pos - start) as u32;
                Some(SplitEvent::AsciiWsRun { byte: b, count })
            } else {
                let start = self.pos;
                self.pos += 1;
                Some(self.extend_word(start))
            }
        } else {
            let start = self.pos;
            let c = self.decode_char_at();
            if c.is_whitespace() {
                Some(SplitEvent::MultibyteWs(c))
            } else {
                Some(self.extend_word(start))
            }
        }
    }
}

// ── x86_64 kernels ─────────────────────────────────────────────────────────
//
// Unsigned byte compares via the classic xor-0x80 flip: `_mm_cmpgt_epi8` is
// signed, so `b ^ 0x80` re-centers the unsigned order around 0.

/// stop-mask over exactly 16 bytes (SSE2 baseline — safe on every x86_64).
#[cfg(target_arch = "x86_64")]
#[inline]
fn sse2_stop_mask16(bytes: &[u8]) -> u32 {
    debug_assert!(bytes.len() >= 16);
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm_loadu_si128(bytes.as_ptr().cast::<__m128i>());
        let bx = _mm_xor_si128(v, _mm_set1_epi8(0x80u8 as i8));
        // unsigned b > 8  ⇔ signed cmpgt(b ^ 0x80, (0x08 ^ 0x80) as i8)
        let ge9 = _mm_cmpgt_epi8(bx, _mm_set1_epi8((0x08u8 ^ 0x80) as i8));
        // unsigned b < 14 ⇔ signed cmpgt((0x0E ^ 0x80) as i8, b ^ 0x80)
        let le13 = _mm_cmpgt_epi8(_mm_set1_epi8((0x0Eu8 ^ 0x80) as i8), bx);
        let eq20 = _mm_cmpeq_epi8(v, _mm_set1_epi8(b' ' as i8));
        // b >= 0x80 ⇔ the raw byte is negative as i8 (high bit set).
        // NB: NOT `cmpgt(bx, 0)` — that misses b == 0x80 exactly (bx == 0).
        let nonascii = _mm_cmpgt_epi8(_mm_set1_epi8(0), v);
        let stop = _mm_or_si128(_mm_or_si128(_mm_and_si128(ge9, le13), eq20), nonascii);
        _mm_movemask_epi8(stop) as u32
    }
}

/// eq-mask over exactly 16 bytes (SSE2 baseline).
#[cfg(target_arch = "x86_64")]
#[inline]
fn sse2_eq_mask16(bytes: &[u8], byte: u8) -> u32 {
    debug_assert!(bytes.len() >= 16);
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm_loadu_si128(bytes.as_ptr().cast::<__m128i>());
        _mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_set1_epi8(byte as i8))) as u32
    }
}

/// stop-mask over exactly 32 bytes. AVX2 — caller must have probed
/// `is_x86_feature_detected!("avx2")` (the `SplitLevel::Avx2` invariant).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_stop_mask32(bytes: &[u8]) -> u32 {
    debug_assert!(bytes.len() >= 32);
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm256_loadu_si256(bytes.as_ptr().cast::<__m256i>());
        let bx = _mm256_xor_si256(v, _mm256_set1_epi8(0x80u8 as i8));
        let ge9 = _mm256_cmpgt_epi8(bx, _mm256_set1_epi8((0x08u8 ^ 0x80) as i8));
        let le13 = _mm256_cmpgt_epi8(_mm256_set1_epi8((0x0Eu8 ^ 0x80) as i8), bx);
        let eq20 = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(b' ' as i8));
        // b >= 0x80 ⇔ the raw byte is negative as i8 — NOT `cmpgt(bx, 0)`, which
        // misses b == 0x80 exactly (bx == 0).
        let nonascii = _mm256_cmpgt_epi8(_mm256_setzero_si256(), v);
        let stop = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(ge9, le13), eq20), nonascii);
        _mm256_movemask_epi8(stop) as u32
    }
}

/// eq-mask over exactly 32 bytes (AVX2 — same caller contract as
/// [`avx2_stop_mask32`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_eq_mask32(bytes: &[u8], byte: u8) -> u32 {
    debug_assert!(bytes.len() >= 32);
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm256_loadu_si256(bytes.as_ptr().cast::<__m256i>());
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(byte as i8))) as u32
    }
}

// ── aarch64 NEON kernels ────────────────────────────────────────────────────
//
// NEON u8 compares are unsigned — no xor-0x80 flip needed. NEON has no
// movemask; the bitmask is built by shifting each byte's bit 7 down to 0/1
// and SWAR-compressing the two u64 lanes (stride-8 → stride-1 via the
// shift-or chain — see `compress_bits8`, correct by construction: each
// source bit lands on exactly its own target bit, verified exhaustively in
// the unit tests).

/// stop-mask over exactly 16 bytes (NEON baseline on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_stop_mask16(bytes: &[u8]) -> u32 {
    debug_assert!(bytes.len() >= 16);
    unsafe {
        use core::arch::aarch64::*;
        let v = vld1q_u8(bytes.as_ptr());
        let ge9 = vcgeq_u8(v, vdupq_n_u8(0x09));
        let le13 = vcleq_u8(v, vdupq_n_u8(0x0D));
        let eq20 = vceqq_u8(v, vdupq_n_u8(b' '));
        let nonascii = vcgeq_u8(v, vdupq_n_u8(0x80));
        let stop = vorrq_u8(vorrq_u8(vandq_u8(ge9, le13), eq20), nonascii);
        neon_movemask16(stop)
    }
}

/// eq-mask over exactly 16 bytes (NEON baseline).
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_eq_mask16(bytes: &[u8], byte: u8) -> u32 {
    debug_assert!(bytes.len() >= 16);
    unsafe {
        use core::arch::aarch64::*;
        let v = vld1q_u8(bytes.as_ptr());
        neon_movemask16(vceqq_u8(v, vdupq_n_u8(byte)))
    }
}

/// bit i of result = bit 7 of byte i of `v`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_movemask16(v: core::arch::aarch64::uint8x16_t) -> u32 {
    unsafe {
        use core::arch::aarch64::*;
        let ones = vshrq_n_u8(v, 7); // 0/1 per byte
        let u = vreinterpretq_u64_u8(ones); // byte j of lane k at bit 8j
        let c0 = compress_bits8(vgetq_lane_u64(u, 0));
        let c1 = compress_bits8(vgetq_lane_u64(u, 1));
        c0 as u32 | ((c1 as u32) << 8)
    }
}

/// Compress 8 bits at stride 8 (bits 0, 8, …, 56) to bits 0..=7.
/// Each cumulative-shift path lands a source bit on exactly its own index:
/// the possible cumulative shifts {7, 14, 21, 28, 35, 42, 49} map source j
/// (bit 8j) to target 8j − shift, which is in 0..=7 only for j = target.
#[cfg(target_arch = "aarch64")]
#[inline]
fn compress_bits8(mut x: u64) -> u64 {
    x |= x >> 7;
    x |= x >> 14;
    x |= x >> 28;
    x & 0xFF
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every Unicode `White_Space` char (25 total).
    const ALL_WS_CHARS: [char; 25] = [
        '\u{0009}', '\u{000A}', '\u{000B}', '\u{000C}', '\u{000D}', '\u{0020}', '\u{0085}',
        '\u{00A0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}',
        '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200A}', '\u{2028}',
        '\u{2029}', '\u{202F}', '\u{205F}', '\u{3000}',
    ];

    /// Near-miss chars that are NOT `White_Space` (regression traps:
    /// U+180E was ws before Unicode 6.3; ZWSP/FEFF are "space-like" but not
    /// ws; 0x0E/0x08 bracket the ASCII ws range; U+0084 is a C1 control).
    const NOT_WS_NEAR_MISSES: [char; 9] = [
        '\u{0008}', '\u{000E}', '\u{007F}', '\u{0084}', '\u{00AD}', '\u{180E}', '\u{200B}',
        '\u{200C}', '\u{FEFF}',
    ];

    fn levels_under_test() -> Vec<(SplitLevel, &'static str)> {
        let mut v = vec![(SplitLevel::Scalar, "scalar")];
        #[cfg(target_arch = "x86_64")]
        {
            v.push((SplitLevel::Sse2, "sse2"));
            if std::is_x86_feature_detected!("avx2") {
                v.push((SplitLevel::Avx2, "avx2"));
            }
        }
        #[cfg(target_arch = "aarch64")]
        v.push((SplitLevel::Neon, "neon"));
        v
    }

    /// Deterministic xorshift for property inputs (no rand dep).
    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            let mut s = self.0 | 1;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            self.0 = s;
            s
        }
        fn next_byte(&mut self) -> u8 {
            (self.next_u64() >> 32) as u8
        }
    }

    // ------------------------------------------------------------------
    // G1a — mask equivalence: SIMD masks == scalar masks, per level, over
    // every byte value and random buffers.
    // ------------------------------------------------------------------

    #[test]
    fn masks_match_scalar_all_byte_values() {
        // 256 consecutive bytes = 16 chunks of 16 (SSE2/NEON) + 8 of 32 (AVX2).
        let all: Vec<u8> = (0..=255u8).collect();
        for (level, name) in levels_under_test() {
            let w = level.chunk_len();
            let spl = WhitespaceSplitter::with_level("", level);
            for (ci, chunk) in all.chunks_exact(w).enumerate() {
                let want: u32 = chunk
                    .iter()
                    .enumerate()
                    .filter(|&(_, &b)| is_ascii_ws_byte(b) || b >= 0x80)
                    .fold(0u32, |m, (i, _)| m | (1 << i));
                assert_eq!(
                    spl.stop_mask(chunk),
                    want,
                    "{name}: stop_mask chunk {ci} (bytes {chunk:02x?})"
                );
                for probe in [0u8, b' ', 0x09, 0x0B, 0x41, 0x80, 0xFF] {
                    let want_eq: u32 = chunk
                        .iter()
                        .enumerate()
                        .filter(|&(_, &b)| b == probe)
                        .fold(0u32, |m, (i, _)| m | (1 << i));
                    assert_eq!(
                        spl.eq_mask(chunk, probe),
                        want_eq,
                        "{name}: eq_mask({probe:02x}) chunk {ci}"
                    );
                }
            }
        }
    }

    #[test]
    fn masks_match_scalar_random_buffers() {
        let mut rng = Lcg(0xDEAD_BEEF_CAFE_F00D);
        for (level, name) in levels_under_test() {
            let w = level.chunk_len();
            let spl = WhitespaceSplitter::with_level("", level);
            for _ in 0..64 {
                let buf: Vec<u8> = (0..w).map(|_| rng.next_byte()).collect();
                let want: u32 = buf
                    .iter()
                    .enumerate()
                    .filter(|&(_, &b)| is_ascii_ws_byte(b) || b >= 0x80)
                    .fold(0u32, |m, (i, _)| m | (1 << i));
                assert_eq!(
                    spl.stop_mask(&buf),
                    want,
                    "{name}: random buffer {buf:02x?}"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // G1b — event-stream differential: SIMD levels == scalar level == a
    // straightforward semantic reference (chars + is_whitespace).
    // ------------------------------------------------------------------

    /// The semantic oracle: exactly the old scalar loop's behavior.
    fn reference_events(text: &str) -> Vec<(Vec<u8>, Option<char>)> {
        // (word bytes, None) for a word; (empty, Some(ws char)) per ws char.
        let mut out = Vec::new();
        let mut word: Vec<u8> = Vec::new();
        for c in text.chars() {
            if c.is_whitespace() {
                if !word.is_empty() {
                    out.push((std::mem::take(&mut word), None));
                }
                out.push((Vec::new(), Some(c)));
            } else {
                let mut buf = [0u8; 4];
                word.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
        if !word.is_empty() {
            out.push((word, None));
        }
        out
    }

    fn split_events(text: &str, level: SplitLevel) -> Vec<(Vec<u8>, Option<char>)> {
        // Expand AsciiWsRun to per-char entries so all levels + the
        // reference share one comparable shape.
        let mut out = Vec::new();
        for ev in WhitespaceSplitter::with_level(text, level) {
            match ev {
                SplitEvent::Word(b) => out.push((b.to_vec(), None)),
                SplitEvent::AsciiWsRun { byte, count } => {
                    for _ in 0..count {
                        out.push((Vec::new(), Some(byte as char)));
                    }
                }
                SplitEvent::MultibyteWs(c) => out.push((Vec::new(), Some(c))),
            }
        }
        out
    }

    fn assert_all_levels_agree(text: &str) {
        let want = reference_events(text);
        for (level, name) in levels_under_test() {
            // Level's own stream vs the semantic reference…
            assert_eq!(
                split_events(text, level),
                want,
                "{name}: event stream diverges from reference on {text:?}"
            );
            // …and (belt+braces) vs the scalar level, coalesced form.
            if level != SplitLevel::Scalar {
                let a = WhitespaceSplitter::with_level(text, SplitLevel::Scalar)
                    .map(format_event)
                    .collect::<Vec<_>>();
                let b = WhitespaceSplitter::with_level(text, level)
                    .map(format_event)
                    .collect::<Vec<_>>();
                assert_eq!(
                    a, b,
                    "{name}: coalesced stream differs from scalar on {text:?}"
                );
            }
        }
    }

    fn format_event(ev: SplitEvent<'_>) -> (Vec<u8>, Option<char>, u32) {
        match ev {
            SplitEvent::Word(b) => (b.to_vec(), None, 0),
            SplitEvent::AsciiWsRun { byte, count } => (vec![byte], Some(byte as char), count),
            SplitEvent::MultibyteWs(c) => {
                let mut buf = [0u8; 4];
                (c.encode_utf8(&mut buf).as_bytes().to_vec(), Some(c), 1)
            }
        }
    }

    #[test]
    fn events_agree_on_every_unicode_ws_char() {
        for w in ALL_WS_CHARS {
            for text in [
                format!("aa{w}bb"),
                format!("{w}"),
                format!("x{w}"),
                format!("{w}x"),
                format!("a{w}b{w}c"),
                format!("a{w}{w}b"),
                format!("日本{w}語"),
            ] {
                assert_all_levels_agree(&text);
            }
        }
    }

    #[test]
    fn events_agree_on_near_miss_non_ws() {
        for c in NOT_WS_NEAR_MISSES {
            assert_all_levels_agree(&format!("a{c}b {c} {c}a"));
        }
    }

    #[test]
    fn events_agree_on_ws_runs_and_chunk_boundaries() {
        for k in [0usize, 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 70] {
            assert_all_levels_agree(&format!("a{}b", " ".repeat(k)));
            assert_all_levels_agree(&format!("{}\n", "\t".repeat(k)));
        }
        assert_all_levels_agree(" \t \t  mixed ws runs   ");
        // Words straddling chunk boundaries at every offset around 16/32.
        for n in [14usize, 15, 16, 17, 18, 30, 31, 32, 33, 34] {
            for m in [0usize, 1, 15, 16, 17] {
                assert_all_levels_agree(&format!("{} {}{}", "x".repeat(n), "y".repeat(m), " z"));
                // ws at the exact chunk boundary
                assert_all_levels_agree(&format!("{} {}{}", "x".repeat(n), " ".repeat(m + 1), "w"));
            }
        }
    }

    #[test]
    fn events_agree_on_multibyte_text() {
        // CJK (3-byte), emoji (4-byte), regional-indicator pairs, combining
        // marks — heavy non-ASCII with embedded ws.
        assert_all_levels_agree("中文 测试 分词器的 性能 hello 世界");
        assert_all_levels_agree("a😀b c🇺🇸d é f̃ g🄰h");
        assert_all_levels_agree("中文\u{3000}全角空格 NBSP\u{00A0}here LSEP\u{2028}next");
        assert_all_levels_agree("héllo wörld naïve café résumé");
        // Non-ASCII word adjacent to ws at a chunk boundary.
        for n in [14usize, 15, 16, 31, 32] {
            assert_all_levels_agree(&format!("{}中{}", "a".repeat(n), "文 test"));
        }
    }

    #[test]
    fn events_agree_on_random_ascii() {
        let mut rng = Lcg(0x1234_5678_9ABC_DEF0);
        let alphabet: Vec<u8> = Vec::from_iter(
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.,!?()'\" \t\n\r\x0b\x0c"
                .iter()
                .copied(),
        );
        for len in [1usize, 7, 15, 16, 17, 31, 33, 64, 100, 257, 1000] {
            let text: String = (0..len)
                .map(|_| alphabet[(rng.next_u64() as usize) % alphabet.len()] as char)
                .collect();
            assert_all_levels_agree(&text);
        }
    }

    #[test]
    fn words_are_maximal_runs_including_multibyte() {
        // "héllo" must arrive as ONE word (merges can cross the ASCII
        // boundary inside a word) — not Word("h")+Word("éllo").
        let mut n_words = 0;
        for ev in WhitespaceSplitter::new("héllo wörld") {
            if let SplitEvent::Word(b) = ev {
                n_words += 1;
                assert!(b.len() >= 6, "word split mid-way: {b:02x?}");
            }
        }
        assert_eq!(n_words, 2);
    }

    #[test]
    fn ascii_ws_predicate_is_six_bytes() {
        // The predicate's contract is the ASCII range only — `char::from(b)`
        // for b >= 0x80 is a Latin-1 char, and U+0085/U+00A0 there ARE
        // Unicode whitespace (classified by the multibyte path, not this
        // predicate).
        for b in 0..0x80u8 {
            let want = char::from(b).is_whitespace();
            assert_eq!(
                is_ascii_ws_byte(b),
                want,
                "byte {b:#04x}: predicate must match char::is_whitespace on ASCII"
            );
        }
    }

    // ------------------------------------------------------------------
    // G2 — scan-only A/B: SIMD level vs Scalar on ASCII-dominant text.
    // Interleaved rounds + median ratio (the repo's hand-rolled treated
    // shape; the shared tests/common harness is root-package-only).
    // ------------------------------------------------------------------

    #[test]
    fn g2_scan_ab_simd_vs_scalar() {
        let (level, name) = match probe_level() {
            #[cfg(target_arch = "x86_64")]
            SplitLevel::Sse2 => (SplitLevel::Sse2, "sse2"),
            #[cfg(target_arch = "x86_64")]
            SplitLevel::Avx2 => (SplitLevel::Avx2, "avx2"),
            #[cfg(target_arch = "aarch64")]
            SplitLevel::Neon => (SplitLevel::Neon, "neon"),
            // Unreachable via the probe on SIMD targets; keeps the match
            // exhaustive there. On scalar-only targets it skips the A/B.
            SplitLevel::Scalar => return,
        };
        // ~1 MB of varied natural-language-ish ASCII (varied so the cache
        // doesn't trivially shortcut — scan-only counts events, no vocab).
        let sentence = "the quick brown fox jumps over the lazy dog and then \
                        returns to the function that called it, for the test \
                        of the splitter throughput on ascii dominant text; ok";
        let text = sentence.repeat(4_096);
        std::hint::black_box(&text);

        let count = |level| {
            let mut n = 0usize;
            let mut bytes = 0usize;
            for ev in WhitespaceSplitter::with_level(std::hint::black_box(&text), level) {
                match ev {
                    SplitEvent::Word(b) => {
                        n += 1;
                        bytes += b.len();
                    }
                    SplitEvent::AsciiWsRun { count, .. } => n += count as usize,
                    SplitEvent::MultibyteWs(_) => n += 1,
                }
            }
            n + bytes
        };

        // The OLD scalar loop's shape (pre-Issue-872 `encode_into_pretok`):
        // per-char UTF-8 decode + `is_whitespace` + `encode_utf8` + per-char
        // accumulation into a byte buffer. The vocab lookup is omitted to
        // match `count` (both sides skip it).
        let old_loop = |text: &str| {
            let mut n = 0usize;
            let mut bytes = 0usize;
            let mut buf = [0u8; 4];
            let mut word: Vec<u8> = Vec::new();
            for c in text.chars() {
                if c.is_whitespace() {
                    if !word.is_empty() {
                        n += 1;
                        bytes += word.len();
                        word.clear();
                    }
                    n += 1;
                } else {
                    word.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
            if !word.is_empty() {
                n += 1;
                bytes += word.len();
            }
            n + bytes
        };

        // Warm both paths.
        let a = count(SplitLevel::Scalar);
        let b = count(level);
        let o = old_loop(&text);
        assert_eq!(a, b, "event totals must match across levels");
        assert_eq!(a, o, "event totals must match the old loop");

        let mut ratios: Vec<f64> = Vec::new();
        let mut ratios_old: Vec<f64> = Vec::new();
        for _ in 0..9 {
            let t = std::time::Instant::now();
            let ro = old_loop(std::hint::black_box(&text));
            let d_old = t.elapsed();
            let t = std::time::Instant::now();
            let rs = count(SplitLevel::Scalar);
            let d_scalar = t.elapsed();
            let t = std::time::Instant::now();
            let ri = count(level);
            let d_simd = t.elapsed();
            assert_eq!(ro, rs, "old-loop total must match scalar level");
            assert_eq!(rs, ri, "scalar level total must match simd level");
            ratios.push(d_scalar.as_nanos() as f64 / d_simd.as_nanos() as f64);
            ratios_old.push(d_old.as_nanos() as f64 / d_simd.as_nanos() as f64);
        }
        ratios.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        ratios_old.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        let median = ratios[4];
        let median_old = ratios_old[4];
        println!(
            "g2_scan_ab[{name}]: median scalar-level/simd = {median:.2}x, old-loop/simd = {median_old:.2}x ({ratios:?})"
        );
        // Generous floor: expected 3-10x; 1.25 catches a broken dispatch
        // (SIMD level silently falling back to scalar) without flaking on
        // a loaded box.
        assert!(
            median >= 1.25,
            "{name}: scan median ratio {median:.2}x below the 1.25 floor — dispatch regression?"
        );
    }
}

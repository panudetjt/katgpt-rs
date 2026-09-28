//! Computational Irreducibility Gate — Kolmogorov complexity proxy via compression ratio.
//!
//! Wolfram's key finding: winning strategies are simple but can only be found by
//! running the game. This gate detects when a game IS predictable (low irreducibility)
//! and allows skipping expensive simulation.
//!
//! Uses run-length encoding (RLE) as a cheap compression proxy for Kolmogorov complexity.
//! If the win matrix compresses well (low ratio), the game has predictable structure
//! and analytical shortcuts may exist. If the ratio is high, full simulation is required.
//!
//! Plan 188 Phase 4.

use crate::types::WinMatrix;

// ── IrreducibilityResult ──────────────────────────────────────

/// Result of irreducibility analysis.
#[derive(Debug, Clone, Copy)]
pub struct IrreducibilityResult {
    /// Compression ratio of the win matrix (0.0 = fully compressible, 1.0 = fully random).
    /// Ratio = compressed_size / raw_size.
    pub compression_ratio: f32,
    /// Whether the game is considered irreducible (ratio above threshold).
    pub is_irreducible: bool,
    /// Mean absolute payoff in the matrix (indicator of game dynamics).
    pub mean_abs_payoff: f64,
    /// Payoff variance (high variance = complex dynamics).
    pub payoff_variance: f64,
}

// ── IrreducibilityGate ────────────────────────────────────────

/// Gate that determines if a game/strategy space is computationally irreducible.
///
/// Uses a simple run-length encoding (RLE) compression as a Kolmogorov complexity
/// proxy. If the win matrix compresses well (low ratio), the game has predictable
/// structure and analytical shortcuts may exist.
///
/// Threshold: compression_ratio > 0.7 → irreducible (must simulate)
///            compression_ratio ≤ 0.7 → reducible (shortcuts possible)
pub struct IrreducibilityGate {
    /// Compression ratio threshold above which we consider the game irreducible.
    pub threshold: f32,
}

impl Default for IrreducibilityGate {
    /// Default gate with 0.7 threshold.
    #[inline]
    fn default() -> Self {
        Self::new(0.7)
    }
}

impl IrreducibilityGate {
    /// Create a new gate with the given compression ratio threshold.
    #[inline]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Analyze a win matrix for irreducibility.
    ///
    /// Returns compression ratio, irreducibility verdict, and payoff statistics.
    ///
    /// Uses Shannon entropy of the quantized byte distribution as the primary
    /// Kolmogorov complexity proxy. Low entropy = low complexity = reducible.
    /// For high-entropy matrices, falls back to RLE compression ratio.
    ///
    /// # Single-pass fusion
    ///
    /// The prior implementation iterated `matrix.payoffs` three times
    /// (quantize → payoff_stats → freq). Fused into one pass: quantize each
    /// payoff into a byte, accumulate `freq[byte]`, and accumulate
    /// `sum_abs` + `sum_abs_sq` for the (mean, variance) computation. The
    /// output is `Vec<u8>` plus the two scalar accumulators; the entropy
    /// + variance math runs once over the freq table + scalars, not over
    ///   the matrix again.
    pub fn analyze(&self, matrix: &WinMatrix) -> IrreducibilityResult {
        let n = matrix.payoffs.len();
        // Upper bound on raw byte count — actual may be less if rows are
        // short, but this avoids grow calls during the fused push loop.
        let mut raw = Vec::with_capacity(n.saturating_mul(n));
        let mut freq = [0u32; 256];
        // Payoff-stat accumulators (mean + variance of |payoff|).
        let mut sum_abs = 0.0f64;
        let mut sum_abs_sq = 0.0f64;
        let mut count = 0usize;

        for row in &matrix.payoffs {
            for &val in row {
                // Quantize [-1, 1] → [0, 255].
                let normalized = ((val + 1.0) * 127.5).clamp(0.0, 255.0);
                let q = normalized as u8;
                raw.push(q);
                freq[q as usize] += 1;
                // Payoff stats — |val| not quantized.
                let abs_val = val.abs();
                sum_abs += abs_val;
                sum_abs_sq += abs_val * abs_val;
                count += 1;
            }
        }

        let total = raw.len() as u32;

        // Shannon entropy of byte distribution (bits).
        let entropy = if total == 0 {
            0.0f32
        } else {
            let mut h = 0.0f32;
            let inv_total = 1.0 / total as f32;
            for &cnt in &freq {
                if cnt > 0 {
                    let p = cnt as f32 * inv_total;
                    h -= p * p.log2();
                }
            }
            h
        };

        // Normalized entropy: 0.0 = all same byte, 1.0 = uniform distribution.
        // Max entropy for byte data = 8 bits.
        let normalized_entropy = entropy / 8.0;

        // RLE compression ratio as secondary signal.
        // `rle_compressed_len` is the zero-alloc variant of `rle_compress`:
        // the analyzer only needs the LENGTH of the RLE output (to compute
        // the ratio), not the output bytes themselves. The original path
        // built a full Vec<u8> every call just to read its .len() afterward.
        let compressed_len = rle_compressed_len(&raw);
        let rle_ratio = if raw.is_empty() {
            0.0
        } else {
            compressed_len as f32 / raw.len() as f32
        };

        // Effective compression ratio: use entropy when it's low (structured data),
        // RLE when entropy is high (potentially compressible despite high entropy).
        let compression_ratio = if entropy < 4.0 {
            // Low entropy → highly structured, use normalized entropy as ratio.
            normalized_entropy
        } else if rle_ratio < normalized_entropy {
            // High entropy but RLE compresses → some structure exists.
            rle_ratio
        } else {
            // High entropy, RLE doesn't help → likely irreducible.
            normalized_entropy
        };

        let (mean, variance) = if count == 0 {
            (0.0, 0.0)
        } else {
            let mean = sum_abs / count as f64;
            let variance = sum_abs_sq / count as f64 - mean * mean;
            (mean, variance.max(0.0)) // numerical guard
        };

        IrreducibilityResult {
            compression_ratio,
            is_irreducible: compression_ratio > self.threshold,
            mean_abs_payoff: mean,
            payoff_variance: variance,
        }
    }

    /// Quick check: is the game irreducible?
    pub fn is_irreducible(&self, matrix: &WinMatrix) -> bool {
        self.analyze(matrix).is_irreducible
    }
}

// ── RLE Compression ───────────────────────────────────────────

/// Simple run-length encoding compression.
/// Returns (value, count) pairs as a flat byte sequence.
///
/// Kept for callers that need the actual compressed bytes. The irreducibility
/// analyzer uses [`rle_compressed_len`] instead — zero allocation, same length.
/// Compiled only under `cfg(test)` because the production analyzer is the
/// sole caller of the length variant; tests still need the byte-accurate form
/// to lock the parity invariant.
#[cfg(test)]
fn rle_compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len());
    let mut current = data[0];
    let mut count: u8 = 1;

    for &byte in &data[1..] {
        if byte == current && count < 255 {
            count += 1;
        } else {
            result.push(current);
            result.push(count);
            current = byte;
            count = 1;
        }
    }
    result.push(current);
    result.push(count);

    result.shrink_to_fit();
    result
}

/// Length of `rle_compress(data)` without allocating the output Vec.
///
/// Each RLE run produces exactly 2 output bytes (value + count). The length
/// is just `2 * number_of_runs`. Walks `data` once, counting run boundaries —
/// the same predicate `rle_compress` uses to decide when to flush a run. Used
/// by [`IrreducibilityGate::analyze`] which only needs the ratio
/// (compressed_len / raw_len), not the bytes.
#[inline]
fn rle_compressed_len(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }
    // One run for the first element, +1 for every boundary where byte changes
    // OR the run hits the 255 cap (matches rle_compress's flush predicate).
    let mut runs = 1usize;
    let mut current = data[0];
    let mut count: u8 = 1;
    for &byte in &data[1..] {
        if byte == current && count < 255 {
            count += 1;
        } else {
            runs += 1;
            current = byte;
            count = 1;
        }
    }
    runs * 2
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FsmEnumerator, matching_pennies};
    use std::time::Instant;

    /// Matching pennies with 2-state FSMs should have lots of structure → reducible.
    #[test]
    fn test_irreducibility_simple_game_reducible() {
        let strategies = FsmEnumerator::enumerate(2);
        let matrix = FsmEnumerator::tournament(&strategies, 100, &matching_pennies);
        let gate = IrreducibilityGate::default();
        let result = gate.analyze(&matrix);

        // Matching pennies with 2-state FSMs produces payoffs clustered around
        // a few distinct values, so value diversity is low → reducible.
        assert!(
            !result.is_irreducible,
            "matching pennies should be reducible, ratio={}",
            result.compression_ratio
        );
    }

    /// A random matrix should have a high compression ratio (near 1.0).
    #[test]
    fn test_irreducibility_random_matrix_irreducible() {
        // Build a matrix with pseudo-random payoffs that don't compress well.
        let n = 22;
        let mut payoffs = Vec::with_capacity(n);
        // Use a simple LCG for reproducibility.
        let mut state: u64 = 42;
        for _ in 0..n {
            let mut row = Vec::with_capacity(n);
            for _ in 0..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
                row.push(val);
            }
            payoffs.push(row);
        }

        let ids: Vec<u64> = (0..n as u64).collect();
        let matrix = WinMatrix::new(payoffs, ids);
        let gate = IrreducibilityGate::default();
        let result = gate.analyze(&matrix);

        assert!(
            result.is_irreducible,
            "random matrix should be irreducible, ratio={}",
            result.compression_ratio
        );
        assert!(
            result.compression_ratio > 0.7,
            "random matrix should have high compression ratio, got {}",
            result.compression_ratio
        );
    }

    /// A uniform matrix should compress very well (all same values → 2 bytes).
    #[test]
    fn test_irreducibility_uniform_matrix_reducible() {
        let n = 10;
        let payoffs = vec![vec![0.5; n]; n];
        let ids: Vec<u64> = (0..n as u64).collect();
        let matrix = WinMatrix::new(payoffs, ids);

        let gate = IrreducibilityGate::default();
        let result = gate.analyze(&matrix);

        assert!(
            !result.is_irreducible,
            "uniform matrix should be reducible, ratio={}",
            result.compression_ratio
        );
        // All same value quantizes to same byte → RLE produces 2 bytes for n*n elements.
        assert!(
            result.compression_ratio < 0.1,
            "uniform matrix should compress very well, got {}",
            result.compression_ratio
        );
    }

    /// Verify RLE on known data.
    #[test]
    fn test_rle_compress_basic() {
        // [1, 1, 2, 2, 3] → [1, 2, 2, 2, 3, 1]
        let data = [1u8, 1, 2, 2, 3];
        let compressed = rle_compress(&data);
        assert_eq!(compressed, vec![1, 2, 2, 2, 3, 1]);
    }

    /// All same values should compress to exactly 2 bytes.
    #[test]
    fn test_rle_compress_all_same() {
        let data = [42u8; 100];
        let compressed = rle_compress(&data);
        assert_eq!(compressed.len(), 2, "all-same should compress to 2 bytes");
        assert_eq!(compressed[0], 42);
        assert_eq!(compressed[1], 100);
    }

    /// `rle_compressed_len` MUST agree with `rle_compress(..).len()` on every
    /// input. The irreducibility analyzer's hot path uses the zero-alloc
    /// variant; this parity test guards the invariant against drift.
    #[test]
    fn test_rle_compressed_len_matches_rle_compress() {
        // Empty, single-element, all-same, mixed, and a case that exercises
        // the 255-cap flush boundary.
        let cases: &[&[u8]] = &[
            &[],
            &[7],
            &[42; 1],
            &[42; 100],
            &[1, 1, 2, 2, 3],
            &[1, 2, 3, 4, 5],
            &[0; 256], // 255 + 1 → one full run + a 1-byte tail run
            &[0; 510], // 255 + 255 → two full runs
            &[0; 255], // exactly one full run
        ];
        for (i, data) in cases.iter().enumerate() {
            let expected = rle_compress(data).len();
            let got = rle_compressed_len(data);
            assert_eq!(
                got,
                expected,
                "case {i} (len={}): rle_compressed_len={} vs rle_compress.len()={}",
                data.len(),
                got,
                expected,
            );
        }

        // Randomized cross-check: 200 random byte sequences with random
        // lengths and run distributions.
        let mut state: u64 = 0xC0FFEE;
        for _ in 0..200 {
            // Length in [0, 600] — covers empty, short, and 255-cap cases.
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let len = (state >> 40) as usize % 601;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Bias toward repeats: 70% chance of repeating the prior byte.
                let byte = if !data.is_empty() && (state & 7) < 5 {
                    *data.last().unwrap()
                } else {
                    (state >> 33) as u8
                };
                data.push(byte);
            }
            let expected = rle_compress(&data).len();
            let got = rle_compressed_len(&data);
            assert_eq!(
                got,
                expected,
                "random case (len={}): rle_compressed_len={} vs rle_compress.len()={}",
                data.len(),
                got,
                expected,
            );
        }
    }

    /// Benchmark: analyze() should be sub-millisecond for a 22x22 matrix.
    #[test]
    fn test_gate_overhead() {
        // Build a 22x22 matrix (typical FSM(2) tournament size).
        let strategies = FsmEnumerator::enumerate(2);
        let matrix = FsmEnumerator::tournament(&strategies, 100, &matching_pennies);
        let gate = IrreducibilityGate::default();

        // Warm up.
        let _ = gate.analyze(&matrix);

        // Measure.
        let iterations = 1000u64;
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(gate.analyze(std::hint::black_box(&matrix)));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iterations as u32;

        assert!(
            per_call.as_micros() < 1000,
            "gate overhead should be <1ms per call, got {per_call:?}"
        );
    }

    /// Verify all IrreducibilityResult fields are populated correctly.
    #[test]
    fn test_irreducibility_result_fields() {
        let payoffs = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
        let ids = vec![1u64, 2];
        let matrix = WinMatrix::new(payoffs, ids);

        let gate = IrreducibilityGate::new(0.5);
        let result = gate.analyze(&matrix);

        // All fields should be populated (no zeros from bugs).
        assert!(result.compression_ratio >= 0.0 && result.compression_ratio <= 2.0);
        assert!(
            result.mean_abs_payoff > 0.0,
            "mean_abs_payoff should be positive"
        );
        assert!(
            result.payoff_variance >= 0.0,
            "variance should be non-negative"
        );

        // Matrix has only 2 distinct quantized values (0 and 255).
        // log2(2)/8 = 0.125, which is below the 0.5 threshold → not irreducible.
        assert!(
            !result.is_irreducible,
            "binary-valued matrix should be reducible, ratio={}",
            result.compression_ratio
        );
        assert!(
            (result.compression_ratio - 0.125).abs() < 0.01,
            "2 distinct values should give ratio ~0.125, got {}",
            result.compression_ratio
        );
    }
}

// TL;DR: IrreducibilityGate — RLE compression ratio as Kolmogorov complexity proxy. Low ratio = game is predictable (skip simulation), high ratio = irreducible (must simulate). Default threshold 0.7. Sub-millisecond overhead for 22x22 matrices.

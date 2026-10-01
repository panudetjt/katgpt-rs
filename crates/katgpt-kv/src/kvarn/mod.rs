//! KVarN — Variance-Normalized KV-Cache Quantization (Research 159).
//!
//! Phase 1 core implementation: Sinkhorn-style iterative dual-scaling variance
//! normalization combined with asymmetric RTN quantization for KV cache compression.
//!
//! Pipeline per tile:
//!   Key [D, group]:  Hadamard → variance normalize → RTN with dual scales (per-channel × per-token)
//!   Value [group, D]: Hadamard → variance normalize → RTN with dual scales (per-token × per-channel)
//!
//! The variance normalization equalizes per-row and per-column standard deviations
//! via iterative Sinkhorn-style log-space scaling, reducing quantization error from
//! heterogenous magnitude distributions.
//!
//! GOAT Status (Plan 179):
//!   ✅ 4-bit cosine ≥ 0.98: measured 0.9979 (no-Hadamard)
//!   ✅ Error accumulation ratio < 1.5: measured 1.0116
//!   ✅ Dequant overhead ≤ 1% of gen time: measured 0.57% (no-Hadamard)
//!   ⚠ Dequant vs RTN: +272% (inherent dual-scale cost; traded for ~1.0 accum ratio)
//!
//! Hadamard is optional (default: off). VarN alone provides better quality
//! (cosine 0.9988 vs 0.9974 with Hadamard) because Sinkhorn already equalizes
//! magnitudes. Enable hadamard only if profiling shows correlated channel errors.
//!
//! Binary bloat verification:
//!   cargo build --release 2>/dev/null && ls -la target/release/katgpt-rs
//!   cargo build --release --features kvarn 2>/dev/null && ls -la target/release/katgpt-rs
//!   The two binary sizes should be identical when kvarn is off by default.
//!
//! ⚠ Bit-arm NON-INTERPOLATION (Issue 907, Bench 903): the bit arms are
//! DIFFERENT quantizers, not one quantizer at three widths — `with_config`
//! derives the machinery from `bits` (skip-varn + grouped-4 RTN at b ≤ 2 vs
//! per-tile var-norm at b ≥ 3), and on REAL gemma-2 V rows the arms measured
//! non-monotone: b2 rel-MSE 24.1 < b4 28.8 < b3 132.5 (b3 cosine 0.9416 vs
//! b2's 0.9886 — 2-bit BEATS 4-bit on cosine). The crossed arms are worse
//! than every plain arm (b3-on-b2-machinery ≈ 1351, b2-on-varn ≈ 828) — the
//! machinery classes don't compose; each is tuned to its width class. Within
//! the var-norm machinery, 3 bits cost 4.6× the error of 4 bits — the 3-bit
//! var-norm scale-field handling is the recorded defect site. Consumers:
//! NEVER interpolate V-row quality across bit arms; pick arms by measurement.

mod dequant;
#[cfg(test)]
mod dequant_oracle_tests;
pub mod eval;
pub mod hadamard;
#[cfg(test)]
mod issue_896_tests;
pub mod kv_cache;
pub mod var_norm;

pub use dequant::{KVarNKeyColView, KVarNValueRowView};
pub use eval::pseudo_decode_eval;
pub use kv_cache::{
    KVarNKVCache, pack_value, packed_bytes_per_row, rtn_quantize_rows, rtn_quantize_rows_grouped,
    unpack_row, unpack_value,
};
pub use var_norm::{
    VarNormConfig, VarianceNormScales, variance_normalize, variance_normalize_into,
};

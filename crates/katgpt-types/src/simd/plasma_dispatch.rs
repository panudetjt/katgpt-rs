//! Per-shape plasma dispatch — f32 below the L3 boundary, ternary above
//! (katgpt-rs Issue 843 T4, owner call 2026-09-19).
//!
//! ## The measured basis (Issue 843, both arches, interleaved harness)
//!
//! The `plasma_path` ternary dense matvec is **1.9–3.0× SLOWER** than the
//! f32 `simd_matvec` over the same operator at every shape whose f32 operand
//! still fits L3 — which on the measured boxes includes every shape this
//! workspace serves (768×3072 → 9 MiB f32 operand reads 2.10× on NEON,
//! 1.5–1.9× on AVX2). The ternary kernel's win is its **21× weight
//! footprint** (two bitplanes vs f32), which only converts to latency
//! beyond the L3 boundary — and on bandwidth-starved boxes only (the AVX2
//! 13700K crossed at 64 MiB; NEON never crossed up to a 1 GiB operand).
//!
//! ## The owner call (gate D1, 2026-09-19)
//!
//! **Per-shape dispatch: f32 below the L3 boundary, ternary above.** Option
//! (b) of Issue 843 §T3 (pre-expanded i8 sign rows, ~1.7–2× back at 8× the
//! bitplane footprint ≈ 5.3× vs f32) is NOT taken — the bitplane footprint
//! is not the product at served shapes.
//!
//! The dispatch is a POLICY primitive for callers that hold BOTH
//! representations (the quantize-from-dense seam): it picks the measured
//! winner per shape and never changes what either kernel computes. Below
//! the boundary the output is bit-identical to `simd_matvec`; above it,
//! bit-identical to `simd_ternary_matvec`.
//!
//! ## The L3 boundary
//!
//! [`l3_cache_bytes`] is a cached best-effort probe: `KATGPT_PLASMA_L3_BYTES`
//! env override → macOS `sysctl hw.l3cachesize` → Linux
//! `/sys/devices/system/cpu/cpu0/cache/index3/size` → **32 MiB default**.
//! The default errs toward f32 deliberately: over-estimating L3 costs at
//! most the bounded above-boundary gap (measured ≤1.35× on the one box
//! where f32 collapsed), while under-estimating re-introduces the 1.9–3.0×
//! losing regime the dispatch exists to remove. The finding that travels
//! is "the crossover is the L3 boundary" (Issue 843 §T2), not any fixed
//! matrix size.

use super::dot::simd_matvec;
use super::ternary::simd_ternary_matvec;
use crate::TernaryWeights;

/// Cached L3 (or SLC) size in bytes, best-effort (see the module docs).
pub fn l3_cache_bytes() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(probe_l3_cache_bytes)
}

/// Default when detection is unavailable (Windows/unknown): 32 MiB —
/// representative of the measured boxes (13700K = 30 MB L3) and biased
/// toward the f32 arm per the module docs.
pub const DEFAULT_L3_BYTES: usize = 32 * 1024 * 1024;

fn probe_l3_cache_bytes() -> usize {
    // Explicit override wins everywhere it is set (tests, odd boxes, CI pins).
    if let Ok(s) = std::env::var("KATGPT_PLASMA_L3_BYTES") {
        match s.trim().parse::<usize>() {
            Ok(v) if v > 0 => return v,
            _ => {
                // A typo'd override must not silently read as the default.
                eprintln!(
                    "katgpt-types: ignoring invalid KATGPT_PLASMA_L3_BYTES={s:?} \
                     (expected a positive byte count); falling back to probing"
                );
            }
        }
    }
    if let Some(n) = probe_os_l3()
        && n > 0
    {
        return n;
    }
    DEFAULT_L3_BYTES
}

#[cfg(target_os = "macos")]
fn probe_os_l3() -> Option<usize> {
    // hw.l3cachesize exists on Apple Silicon (the unified SLC) and on
    // Intel Macs with L3; absent on some small-core chips → None → default.
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let name = b"hw.l3cachesize\0";
    let mut value: usize = 0;
    let mut len = core::mem::size_of::<usize>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr().cast(),
            core::ptr::addr_of_mut!(value).cast(),
            core::ptr::addr_of_mut!(len),
            core::ptr::null_mut(),
            0,
        )
    };
    (rc == 0).then_some(value)
}

#[cfg(target_os = "linux")]
fn probe_os_l3() -> Option<usize> {
    // index3 = the (first) L3 by sysfs convention; size reads like "32768K".
    let s = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size").ok()?;
    let s = s.trim();
    let (num, mult) = match s.as_bytes().last() {
        Some(b'K') | Some(b'k') => (&s[..s.len() - 1], 1024usize),
        Some(b'M') | Some(b'm') => (&s[..s.len() - 1], 1024 * 1024),
        _ => (s, 1),
    };
    num.parse::<usize>().ok().map(|n| n * mult)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn probe_os_l3() -> Option<usize> {
    // Windows has no std surface for cache topology; the FFI walk
    // (GetLogicalProcessorInformationEx) is not worth it for a policy probe.
    None
}

/// The dispatch rule, pure over an explicit boundary (the testable form):
/// ternary wins only where the f32 operand `rows × cols × 4` no longer fits
/// the cache.
#[inline]
pub fn plasma_prefers_ternary_with_l3(rows: usize, cols: usize, l3_bytes: usize) -> bool {
    // Checked against overflow: shapes whose f32 operand cannot even be
    // counted in usize are above any boundary by construction.
    (rows as u64) * (cols as u64) * 4 > l3_bytes as u64
}

/// [`plasma_prefers_ternary_with_l3`] over the cached [`l3_cache_bytes`].
#[inline]
pub fn plasma_prefers_ternary(rows: usize, cols: usize) -> bool {
    plasma_prefers_ternary_with_l3(rows, cols, l3_cache_bytes())
}

/// Per-shape dispatch over both weight representations (the quantize seam):
/// below the L3 boundary the f32 `simd_matvec` (measured 1.9–3.0× faster
/// there on both arches), above it `simd_ternary_matvec` (the 21× footprint
/// form). Output is bit-identical to whichever kernel is selected.
///
/// Zero-allocation; the policy read is one cached load.
pub fn simd_matvec_plasma_dispatch_with_l3(
    dense: &[f32],
    ternary: &TernaryWeights,
    x: &[f32],
    y: &mut [f32],
    l3_bytes: usize,
) {
    debug_assert_eq!(
        dense.len(),
        ternary.rows * ternary.cols,
        "dense/ternary shape mismatch"
    );
    if plasma_prefers_ternary_with_l3(ternary.rows, ternary.cols, l3_bytes) {
        simd_ternary_matvec(ternary, x, y);
    } else {
        simd_matvec(y, dense, x, ternary.rows, ternary.cols);
    }
}

/// [`simd_matvec_plasma_dispatch_with_l3`] over the cached [`l3_cache_bytes`].
pub fn simd_matvec_plasma_dispatch(
    dense: &[f32],
    ternary: &TernaryWeights,
    x: &[f32],
    y: &mut [f32],
) {
    let l3 = l3_cache_bytes();
    simd_matvec_plasma_dispatch_with_l3(dense, ternary, x, y, l3);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_layer() -> (Vec<f32>, TernaryWeights, Vec<f32>) {
        let rows = 64usize;
        let cols = 128usize;
        let dense: Vec<f32> = (0..rows * cols)
            .map(|i| (((i * 1103515245) as f32) * 0.001) % 1.0)
            .collect();
        let tw = TernaryWeights::quantize_from_f32(&dense, rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();
        (dense, tw, x)
    }

    #[test]
    fn boundary_is_the_f32_operand_bytes() {
        // 512×512 → 1 MiB f32 operand.
        assert!(
            !plasma_prefers_ternary_with_l3(512, 512, 1024 * 1024),
            "fits → f32"
        );
        assert!(
            plasma_prefers_ternary_with_l3(512, 512, 1024 * 1024 - 1),
            "over → ternary"
        );
        // The served shape (9 MiB operand) sits firmly below a 32 MiB L3.
        assert!(!plasma_prefers_ternary_with_l3(768, 3072, 32 * 1024 * 1024));
    }

    #[test]
    fn dispatch_matches_the_selected_kernel_on_both_sides() {
        let (dense, tw, x) = small_layer();
        let mut y_dispatch = vec![0.0f32; tw.rows];
        let mut y_ref = vec![0.0f32; tw.rows];

        // Below the boundary (generous L3): f32 arm — bit-identical.
        simd_matvec_plasma_dispatch_with_l3(&dense, &tw, &x, &mut y_dispatch, 1 << 30);
        simd_matvec(&mut y_ref, &dense, &x, tw.rows, tw.cols);
        assert_eq!(
            y_dispatch, y_ref,
            "below L3 the dispatch must be the f32 kernel"
        );

        // Above the boundary (tiny L3): ternary arm — bit-identical to the
        // ternary kernel (and numerically DIFFERENT from the f32 arm, since
        // quantization is lossy — that difference is the arm-selection proof).
        simd_matvec_plasma_dispatch_with_l3(&dense, &tw, &x, &mut y_dispatch, 1);
        simd_ternary_matvec(&tw, &x, &mut y_ref);
        assert_eq!(
            y_dispatch, y_ref,
            "above L3 the dispatch must be the ternary kernel"
        );
        assert_ne!(y_dispatch.len(), 0);
    }

    #[test]
    fn l3_probe_is_sane_and_cached() {
        let a = l3_cache_bytes();
        assert!(
            a > 0,
            "the probe must never answer zero (the fallback is 32 MiB)"
        );
        assert_eq!(a, l3_cache_bytes(), "cached");
        assert_eq!(DEFAULT_L3_BYTES, 32 * 1024 * 1024);
    }
}

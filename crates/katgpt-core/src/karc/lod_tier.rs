//! KARC LOD Tier — config type + tier-promotion projection (Plan 556 Phase 3).
//!
//! # The problem this addresses
//!
//! At crowd scale (10k+ NPCs), uniform-LOD1 KARC is wasteful: background NPCs
//! (off-screen, idle) don't need the same forecast quality as hero NPCs
//! (on-screen, in combat). The LOD-tier abstraction lets the runtime tag each
//! NPC with an importance tier and map that to a `KarcForecaster` config:
//!
//! - **LOD0** (background): D=8, M=4, K=2, R=1 → d_h=64 (4× cheaper than LOD1).
//! - **LOD1** (midground):  D=8, M=8, K=4, R=1 → d_h=256 (the HLA config).
//! - **LOD2** (hero):       D=8, M=8, K=8, R=1 → d_h=512 (2× LOD1).
//!
//! When an NPC's importance changes (e.g. enters combat → hero tier), its
//! forecaster must be promoted to the new tier. This module provides the
//! pure-matrix projection logic — the runtime constructs the new
//! `KarcForecaster` from the projected `Wout` + a fresh delay ring.
//!
//! # The tier-subset structure (key invariant)
//!
//! The three tiers are nested feature subsets:
//!
//! - LOD0 features (M=4 modes, K=2 lags) are a strict prefix of LOD1 features
//!   (M=8 modes, K=4 lags): the first 4 modes of LOD1 = LOD0's modes; the
//!   first K=2 lags of LOD1 = LOD0's lags.
//! - LOD1 features (M=8, K=4) are a strict prefix of LOD2 features (M=8, K=8):
//!   same M (8 modes), but LOD2 has more lags (K=8 vs 4).
//!
//! This nested structure means the Wout projection is pure index remapping:
//!
//! - **Down-tier** (LOD2 → LOD1, LOD1 → LOD0, LOD2 → LOD0): for each
//!   `(output_dim, surviving_lag, surviving_coord, surviving_mode)` quad in
//!   the destination, copy the corresponding Wout element from the source.
//!   The dropped features are simply not represented.
//! - **Up-tier** (LOD0 → LOD1, LOD1 → LOD2, LOD0 → LOD2): the destination
//!   Wout is zero-initialized; then for each `(output_dim, surviving_lag,
//!   surviving_coord, surviving_mode)` quad in the source, write to the
//!   corresponding destination position. The new features start with zero
//!   weights (no contribution to the forecast until the next fit).
//! - **Same-tier**: identity (clone).
//!
//! The R=2 higher-order case (Lod2 from Plan 556 spec, d_h=18_720 with paper-Par
//! config) is deferred — pair-product features aren't a simple nested subset
//! of the first-order features. R=2 is a separate concern (promotion-gate
//! config, Issue 185/186/187).
//!
//! # Forecast preservation (honest G1 analysis)
//!
//! - **Down-tier**: the surviving features' Wout columns are preserved
//!   bit-identically. The forecast on a test signal will differ from the
//!   source because the dropped features' contributions are lost. For
//!   background NPCs that don't need high-fidelity forecasts, this is
//!   acceptable. The dominant Fourier modes (lowest frequencies) are
//!   preserved, so the long-term dynamics survive; only the high-frequency
//!   detail is lost.
//! - **Up-tier**: the source forecast is preserved bit-identically on the
//!   first tick (the new features have zero weights → zero contribution).
//!   The runtime is expected to re-fit at the next training opportunity to
//!   populate the new features. This is the "warm-start from trajectory ring"
//!   pattern from the Plan 556 G1 gate.
//!
//! # Modelless
//!
//! Pure linear-algebra matrix projection. No training, no learned params.
//!
//! # Allocation
//!
//! `project_wout_lod_into` takes borrowed slices — zero allocation on the
//! hot path. The caller owns the destination buffer (typically pre-allocated
//! at tier-promotion time, which is one-time per NPC).
//!
//! # Plan 556 GOAT gate
//!
//! - **G1**: same-tier roundtrip (LOD1 → LOD1) = identity. Down-tier
//!   preserves the dominant-mode Wout columns bit-identically. Up-tier
//!   preserves the source Wout columns bit-identically and zero-fills the
//!   new columns.
//! - **G2**: tier promotion ≤ 10 µs (one-time, not per-tick).
//! - **G3**: enabling `karc_lod_tier` does not perturb the existing default
//!   (LOD1) path (separate code path).
//! - **G4**: per-tick dispatch is zero-alloc (the runtime holds one
//!   `KarcForecaster` per NPC, sized to its tier). Tier promotion may
//!   allocate (one-time, the new `Wout` buffer).
//!
//! # References
//!
//! - **Plan:** `katgpt-rs/.plans/556_karc_mitigations_open_primitives.md` (Phase 3)
//! - **Runtime consumer:** `riir-ai/.plans/514_karc_mitigations_runtime.md`
//!   Phase 2 (LOD tier dispatch — the crowd-scale use case).

/// LOD tier tag. Maps to a specific `KarcForecaster` const-generic
/// monomorphization.
///
/// See the module docs for the nested-subset structure that makes tier
/// promotion a pure index remap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KarcLodTier {
    /// Background NPCs. D=8, M=4, K=2, R=1 → d_h=64.
    Lod0 = 0,
    /// Midground NPCs (the default HLA config). D=8, M=8, K=4, R=1 → d_h=256.
    Lod1 = 1,
    /// Hero / combat NPCs. D=8, M=8, K=8, R=1 → d_h=512.
    Lod2 = 2,
}

impl KarcLodTier {
    /// Observation dimension `D` (constant across tiers — HLA dim).
    #[inline]
    pub const fn d(self) -> usize {
        8
    }

    /// Basis order `M` (Fourier modes × 2 = features per coordinate).
    #[inline]
    pub const fn m(self) -> usize {
        match self {
            Self::Lod0 => 4,
            Self::Lod1 | Self::Lod2 => 8,
        }
    }

    /// Delay embedding depth `K` (lags).
    #[inline]
    pub const fn k(self) -> usize {
        match self {
            Self::Lod0 => 2,
            Self::Lod1 => 4,
            Self::Lod2 => 8,
        }
    }

    /// Higher-order feature order `R` (1 = first-order only).
    ///
    /// Phase 3 ships R=1 only — the R=2 case (paper-Par promotion-gate config,
    /// d_h=18_720) is deferred per the module docs.
    #[inline]
    pub const fn r(self) -> usize {
        1
    }

    /// First-order feature dimension `d_h_1 = K · D · M`.
    #[inline]
    pub const fn d_h_1(self) -> usize {
        self.k() * self.d() * self.m()
    }

    /// Full feature dimension `d_h` (= `d_h_1` for R=1; R=2 deferred).
    #[inline]
    pub const fn d_h(self) -> usize {
        // R=1 path only for Phase 3.
        self.d_h_1()
    }

    /// All tiers in increasing-fidelity order.
    pub const ALL: [Self; 3] = [Self::Lod0, Self::Lod1, Self::Lod2];
}

/// Project a source-tier `Wout` matrix into a destination-tier `Wout`.
///
/// # Layout
///
/// - `src_wout`: `[D · src_tier.d_h()]` row-major.
/// - `dst_wout`: `[D · dst_tier.d_h()]` row-major. **Must be zero-initialized
///   by the caller** for up-tier promotions (the new feature columns start
///   with zero weights). For down-tier and same-tier, the caller can either
///   zero-init or leave uninitialized — only the surviving positions are
///   written.
///
/// # Forecast preservation
///
/// See the module docs for the full analysis. Summary:
/// - **Down-tier**: surviving columns copied bit-identically. Forecast
///   changes because dropped features' contributions are lost.
/// - **Up-tier**: surviving columns copied bit-identically; new columns
///   retain their caller-initialized value (typically zero). Forecast on
///   the first tick after promotion is bit-identical to the source if the
///   caller zero-initialized (the new features contribute nothing).
/// - **Same-tier**: identity copy.
///
/// # Panics
///
/// In debug builds: panics on shape mismatch (`src_wout.len() !=
/// src_tier.d() * src_tier.d_h()` or `dst_wout.len() != dst_tier.d() *
/// dst_tier.d_h()`).
///
/// # Allocation
///
/// Zero. Pure index remap into caller-owned slices.
pub fn project_wout_lod_into(
    src_wout: &[f32],
    src_tier: KarcLodTier,
    dst_wout: &mut [f32],
    dst_tier: KarcLodTier,
) {
    let d = src_tier.d();
    let src_d_h = src_tier.d_h();
    let dst_d_h = dst_tier.d_h();
    let src_m = src_tier.m();
    let dst_m = dst_tier.m();
    let src_k = src_tier.k();
    let dst_k = dst_tier.k();
    debug_assert_eq!(
        src_wout.len(),
        d * src_d_h,
        "project_wout_lod_into: src_wout.len() mismatch (expected {} = {}*{})",
        d * src_d_h,
        d,
        src_d_h,
    );
    debug_assert_eq!(
        dst_wout.len(),
        d * dst_d_h,
        "project_wout_lod_into: dst_wout.len() mismatch (expected {} = {}*{})",
        d * dst_d_h,
        d,
        dst_d_h,
    );
    debug_assert_eq!(src_tier.d(), dst_tier.d(), "D must match across tiers");
    debug_assert_eq!(
        src_tier.r(),
        dst_tier.r(),
        "R must match across tiers (R=2 deferred)"
    );

    // The surviving features are the intersection of the source and destination
    // feature spaces. Per the nested-subset invariant (module docs):
    // - modes: src_modes ∩ dst_modes = first min(src_m, dst_m) modes
    // - lags:  src_lags ∩ dst_lags = first min(src_k, dst_k) lags
    // Each lag has D coordinates; each coordinate has M features. The feature
    // index for (lag, coord, mode) is `(lag * D + coord) * M + mode`.
    let surviving_m = src_m.min(dst_m);
    let surviving_k = src_k.min(dst_k);

    // For each output row (D outputs), copy the surviving columns.
    for out_row in 0..d {
        let src_row_off = out_row * src_d_h;
        let dst_row_off = out_row * dst_d_h;
        for lag in 0..surviving_k {
            for coord in 0..d {
                let src_feat_off = (lag * d + coord) * src_m;
                let dst_feat_off = (lag * d + coord) * dst_m;
                // Copy the surviving modes for this (lag, coord) pair.
                dst_wout[dst_row_off + dst_feat_off..dst_row_off + dst_feat_off + surviving_m]
                    .copy_from_slice(
                        &src_wout
                            [src_row_off + src_feat_off..src_row_off + src_feat_off + surviving_m],
                    );
            }
        }
    }
}

/// Convenience: same-tier "promotion" is identity. Returns true iff the call
/// would be a no-op (caller can skip the projection).
#[inline]
pub fn is_identity_projection(src_tier: KarcLodTier, dst_tier: KarcLodTier) -> bool {
    src_tier == dst_tier
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FourierBasis, KarcForecaster};

    /// Build a fitted `KarcForecaster` at a specific tier.
    fn make_fitted_tier(tier: KarcLodTier) -> (Vec<f32>, Vec<f32>) {
        // Dispatch to the right const-generic monomorphization.
        match tier {
            KarcLodTier::Lod0 => make_fitted_const::<8, 4, 2>(),
            KarcLodTier::Lod1 => make_fitted_const::<8, 8, 4>(),
            KarcLodTier::Lod2 => make_fitted_const::<8, 8, 8>(),
        }
    }

    fn make_fitted_const<const D: usize, const M: usize, const K: usize>() -> (Vec<f32>, Vec<f32>) {
        let n_train = 200;
        let traj: Vec<f32> = (0..n_train)
            .flat_map(|i| {
                let t = i as f32 * 0.05;
                let mut row = [0.0f32; 32];
                let n = D.min(32);
                for (d, row_d) in row.iter_mut().enumerate().take(n) {
                    let freq = 0.3 + 0.2 * d as f32;
                    *row_d = (freq * t).sin() + 0.5 * ((freq + 1.0) * t).cos();
                }
                row[..D].to_vec()
            })
            .collect();
        let n_total = traj.len() / D;
        let mut f = KarcForecaster::<FourierBasis<M>, D, M, K>::with_capacity(
            FourierBasis::new(4.0),
            n_total,
        );
        let kd = K * D;
        for t in (K - 1)..(n_total - 1) {
            let mut delay = vec![0.0f32; kd];
            for lag in 0..K {
                let idx = t - lag;
                for d in 0..D {
                    delay[lag * D + d] = traj[idx * D + d];
                }
            }
            let mut target = vec![0.0f32; D];
            for d in 0..D {
                target[d] = traj[(t + 1) * D + d];
            }
            f.accumulate_pair(&delay, target.as_slice().try_into().unwrap());
        }
        f.fit_ridge(1e-2).expect("fit_ridge");
        let mut seed = vec![0.0f32; kd];
        for lag in 0..K {
            let idx = (n_total - 1) - lag;
            for d in 0..D {
                seed[lag * D + d] = traj[idx * D + d];
            }
        }
        (f.wout.clone(), seed)
    }

    #[test]
    fn tier_dim_accessors() {
        assert_eq!(KarcLodTier::Lod0.d(), 8);
        assert_eq!(KarcLodTier::Lod0.m(), 4);
        assert_eq!(KarcLodTier::Lod0.k(), 2);
        assert_eq!(KarcLodTier::Lod0.d_h(), 64);

        assert_eq!(KarcLodTier::Lod1.d(), 8);
        assert_eq!(KarcLodTier::Lod1.m(), 8);
        assert_eq!(KarcLodTier::Lod1.k(), 4);
        assert_eq!(KarcLodTier::Lod1.d_h(), 256);

        assert_eq!(KarcLodTier::Lod2.d(), 8);
        assert_eq!(KarcLodTier::Lod2.m(), 8);
        assert_eq!(KarcLodTier::Lod2.k(), 8);
        assert_eq!(KarcLodTier::Lod2.d_h(), 512);
    }

    #[test]
    fn same_tier_projection_is_identity() {
        // G1: same-tier roundtrip must be identity.
        for &tier in &KarcLodTier::ALL {
            let (src_wout, _) = make_fitted_tier(tier);
            let mut dst_wout = vec![f32::NAN; src_wout.len()];
            project_wout_lod_into(&src_wout, tier, &mut dst_wout, tier);
            for (i, (&s, &d)) in src_wout.iter().zip(dst_wout.iter()).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    d.to_bits(),
                    "same-tier {tier:?} projection not identity at index {i}",
                );
            }
        }
    }

    #[test]
    fn down_tier_preserves_surviving_columns() {
        // G1: down-tier (LOD1 → LOD0) preserves the surviving Wout columns
        // bit-identically. The surviving columns are those for (lag, coord,
        // mode) quads where lag < K_dst, mode < M_dst.
        let (src_wout_lod1, _) = make_fitted_tier(KarcLodTier::Lod1);
        let mut dst_wout_lod0 = vec![f32::NAN; KarcLodTier::Lod0.d() * KarcLodTier::Lod0.d_h()];
        project_wout_lod_into(
            &src_wout_lod1,
            KarcLodTier::Lod1,
            &mut dst_wout_lod0,
            KarcLodTier::Lod0,
        );

        // Verify: for each surviving (out_row, lag, coord, mode) in LOD0, the
        // destination value equals the source value at the corresponding LOD1
        // position.
        let d = 8usize;
        let m_src = 8usize;
        let m_dst = 4usize;
        let _k_src = 4usize;
        let k_dst = 2usize;
        let d_h_src = KarcLodTier::Lod1.d_h();
        let d_h_dst = KarcLodTier::Lod0.d_h();
        for out_row in 0..d {
            for lag in 0..k_dst {
                for coord in 0..d {
                    for mode in 0..m_dst {
                        let src_idx = out_row * d_h_src + (lag * d + coord) * m_src + mode;
                        let dst_idx = out_row * d_h_dst + (lag * d + coord) * m_dst + mode;
                        assert_eq!(
                            src_wout_lod1[src_idx].to_bits(),
                            dst_wout_lod0[dst_idx].to_bits(),
                            "down-tier LOD1→LOD0 column mismatch at (row={out_row}, lag={lag}, coord={coord}, mode={mode})",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn up_tier_preserves_source_columns_and_zero_fills_new() {
        // G1: up-tier (LOD0 → LOD1) preserves the source Wout columns
        // bit-identically, and the new columns retain their caller-initialized
        // value (zero in this test).
        let (src_wout_lod0, _) = make_fitted_tier(KarcLodTier::Lod0);
        let mut dst_wout_lod1 = vec![0.0f32; KarcLodTier::Lod1.d() * KarcLodTier::Lod1.d_h()];
        project_wout_lod_into(
            &src_wout_lod0,
            KarcLodTier::Lod0,
            &mut dst_wout_lod1,
            KarcLodTier::Lod1,
        );

        // Verify: source columns are preserved bit-identically at the
        // corresponding LOD1 positions.
        let d = 8usize;
        let m_src = 4usize;
        let m_dst = 8usize;
        let k_src = 2usize;
        let k_dst = 4usize;
        let d_h_src = KarcLodTier::Lod0.d_h();
        let d_h_dst = KarcLodTier::Lod1.d_h();
        for out_row in 0..d {
            for lag in 0..k_src {
                for coord in 0..d {
                    for mode in 0..m_src {
                        let src_idx = out_row * d_h_src + (lag * d + coord) * m_src + mode;
                        let dst_idx = out_row * d_h_dst + (lag * d + coord) * m_dst + mode;
                        assert_eq!(
                            src_wout_lod0[src_idx].to_bits(),
                            dst_wout_lod1[dst_idx].to_bits(),
                            "up-tier LOD0→LOD1 source column not preserved at (row={out_row}, lag={lag}, coord={coord}, mode={mode})",
                        );
                    }
                    // The 4 new modes (4..8) should be zero (caller-initialized).
                    for mode in m_src..m_dst {
                        let dst_idx = out_row * d_h_dst + (lag * d + coord) * m_dst + mode;
                        assert_eq!(
                            dst_wout_lod1[dst_idx], 0.0,
                            "up-tier LOD0→LOD1 new column not zero at (row={out_row}, lag={lag}, coord={coord}, mode={mode})",
                        );
                    }
                }
            }
            // The new lags (k_src..k_dst) should be entirely zero.
            for lag in k_src..k_dst {
                for coord in 0..d {
                    for mode in 0..m_dst {
                        let dst_idx = out_row * d_h_dst + (lag * d + coord) * m_dst + mode;
                        assert_eq!(
                            dst_wout_lod1[dst_idx], 0.0,
                            "up-tier LOD0→LOD1 new-lag column not zero",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn down_then_up_tier_roundtrip_preserves_surviving() {
        // LOD1 → LOD0 → LOD1: the surviving LOD0 features (which are a subset
        // of LOD1) should roundtrip bit-identically. The dropped LOD1 features
        // are lost.
        let (src_wout_lod1, _) = make_fitted_tier(KarcLodTier::Lod1);
        let mut dst_wout_lod0 = vec![0.0f32; KarcLodTier::Lod0.d() * KarcLodTier::Lod0.d_h()];
        project_wout_lod_into(
            &src_wout_lod1,
            KarcLodTier::Lod1,
            &mut dst_wout_lod0,
            KarcLodTier::Lod0,
        );
        let mut roundtrip_wout_lod1 = vec![0.0f32; KarcLodTier::Lod1.d() * KarcLodTier::Lod1.d_h()];
        project_wout_lod_into(
            &dst_wout_lod0,
            KarcLodTier::Lod0,
            &mut roundtrip_wout_lod1,
            KarcLodTier::Lod1,
        );

        // Check: the LOD0-shaped region of LOD1 (lags 0..2, modes 0..4) should
        // be preserved bit-identically through the roundtrip.
        let d = 8usize;
        let m = 4usize; // shared with LOD0
        let k = 2usize; // shared with LOD0
        let d_h_lod1 = KarcLodTier::Lod1.d_h();
        for out_row in 0..d {
            for lag in 0..k {
                for coord in 0..d {
                    for mode in 0..m {
                        let idx = out_row * d_h_lod1 + (lag * d + coord) * 8 + mode;
                        assert_eq!(
                            src_wout_lod1[idx].to_bits(),
                            roundtrip_wout_lod1[idx].to_bits(),
                            "LOD1→LOD0→LOD1 roundtrip not identity on surviving features at idx {idx}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn is_identity_projection_detects_same_tier() {
        assert!(is_identity_projection(KarcLodTier::Lod0, KarcLodTier::Lod0));
        assert!(is_identity_projection(KarcLodTier::Lod1, KarcLodTier::Lod1));
        assert!(is_identity_projection(KarcLodTier::Lod2, KarcLodTier::Lod2));
        assert!(!is_identity_projection(
            KarcLodTier::Lod0,
            KarcLodTier::Lod1
        ));
        assert!(!is_identity_projection(
            KarcLodTier::Lod1,
            KarcLodTier::Lod2
        ));
        assert!(!is_identity_projection(
            KarcLodTier::Lod0,
            KarcLodTier::Lod2
        ));
    }

    #[test]
    fn lod2_to_lod0_extreme_down_tier() {
        // Extreme case: LOD2 → LOD0 (drop 75% of features). The surviving
        // LOD0 columns should still be preserved bit-identically.
        let (src_wout_lod2, _) = make_fitted_tier(KarcLodTier::Lod2);
        let mut dst_wout_lod0 = vec![f32::NAN; KarcLodTier::Lod0.d() * KarcLodTier::Lod0.d_h()];
        project_wout_lod_into(
            &src_wout_lod2,
            KarcLodTier::Lod2,
            &mut dst_wout_lod0,
            KarcLodTier::Lod0,
        );

        let d = 8usize;
        let m_src = 8usize;
        let m_dst = 4usize;
        let _k_src = 8usize;
        let k_dst = 2usize;
        let d_h_src = KarcLodTier::Lod2.d_h();
        let d_h_dst = KarcLodTier::Lod0.d_h();
        for out_row in 0..d {
            for lag in 0..k_dst {
                for coord in 0..d {
                    for mode in 0..m_dst {
                        let src_idx = out_row * d_h_src + (lag * d + coord) * m_src + mode;
                        let dst_idx = out_row * d_h_dst + (lag * d + coord) * m_dst + mode;
                        assert_eq!(
                            src_wout_lod2[src_idx].to_bits(),
                            dst_wout_lod0[dst_idx].to_bits(),
                            "extreme down-tier LOD2→LOD0 column mismatch",
                        );
                    }
                }
            }
        }
    }

    /// G2 perf target: tier promotion ≤ 10 µs (one-time, not per-tick). The
    /// worst case is the largest up-tier (LOD0 → LOD2: 64 → 512 cols × D=8 rows).
    /// We don't run a criterion bench here; instead this unit test asserts the
    /// upper bound via a manual timing measurement (mirrors the karc_alloc_check
    /// pattern).
    ///
    /// Run with `cargo test --features karc_lod_tier test_project_wout_lod_perf -- --ignored --nocapture`.
    #[test]
    #[ignore = "G2 perf bench — run explicitly"]
    fn test_project_wout_lod_perf() {
        let src_d_h = KarcLodTier::Lod0.d_h(); // 64
        let dst_d_h = KarcLodTier::Lod2.d_h(); // 512 (worst case up-tier)
        let d = KarcLodTier::Lod2.d();
        let src_wout = vec![0.5f32; d * src_d_h];
        let mut dst_wout = vec![0.0f32; d * dst_d_h];

        // Warmup
        for _ in 0..10 {
            project_wout_lod_into(
                &src_wout,
                KarcLodTier::Lod0,
                &mut dst_wout,
                KarcLodTier::Lod2,
            );
        }

        // Measure: 1000 projections, expect total < 10ms (10µs each).
        let n_calls = 1000;
        let start = std::time::Instant::now();
        for _ in 0..n_calls {
            project_wout_lod_into(
                &src_wout,
                KarcLodTier::Lod0,
                &mut dst_wout,
                KarcLodTier::Lod2,
            );
            std::hint::black_box(&dst_wout);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() / n_calls as u128;
        eprintln!("project_wout_lod_into Lod0→Lod2 (64→512 cols × D=8): {per_call} ns/call",);
        assert!(
            per_call < 10_000,
            "G2 FAIL: project_wout_lod_into took {per_call} ns/call, target ≤ 10_000 ns (10 µs)",
        );
    }
}

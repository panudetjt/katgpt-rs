//! bench_694 — Plan 583 T3.5: offline MI quant-fidelity probe for KVarN
//! (audit-only; katgpt-rs `.plans/583` T3.5).
//!
//! Measures retained information I(W; Ŵ) between the original and
//! dequantized populations through the SHIPPED KVarN round-trip
//! (`store_key` → tile quantize → `dequantize_key_into`), reported alongside
//! the existing reconstruction metrics (`pseudo_decode_eval`: MSE + cosine).
//! The estimator is katgpt-core's `mi_est` — CONSUMED, not reimplemented:
//!
//! - **verdict + magnitude** = the dCor² permutation test (`run_dcor`) —
//!   the characteristic dependence detector, distribution-free p AND a
//!   low-variance magnitude in [0, 1] that ORDERS monotone in the noise
//!   level;
//! - **effect size** = the DV λ-family tuple (LOO + SMILE-LOO + spread) on
//!   the FrozenProj path, REPORTED with its gauge caveat but never asserted
//!   against: a fixed critic's DV value carries a population- and
//!   noise-level-dependent null gauge (the T1.4 calibration), so its
//!   cross-width ordering is not a law — measured en-route: a seed change
//!   flipped the 2-bit arm ABOVE the 4-bit arm while dCor² ordered cleanly.
//!   That measurement is WHY the magnitude axis is dCor², not DV.
//!
//! **Audit-only by design: no gate flips.** The falsifiable assertions pin
//! the PROBE's own behavior (monotone dependence in the bit width,
//! significance at production widths, a degenerate control that must fail),
//! not a quality verdict on KVarN — a promotion gate needs a re-gate showing
//! decision value first (the plan's own rule).
//!
//! Gates:
//! - **ordering** — dCor² strictly decreases as bits decrease 8 → 4 → 2.
//! - **retention** — dependence significant at 8, 4 AND 2 bits (2-bit +
//!   VarNorm is the shipped production setting).
//! - **non-vacuity control** — a degenerate (constant-row) "dequantized"
//!   population reports dependence-lost: the probe can fail.
//! - **determinism** — fixed seeds ⇒ bit-identical probe records.

use katgpt_core::mi::dv::{dv_report, dv_smile_in_place};
use katgpt_core::mi::perm::PermTest;
use katgpt_core::mi::{Critic, MiScratch, PermSource};
use katgpt_kv::kvarn::eval::pseudo_decode_eval;
use katgpt_kv::kvarn::kv_cache::{KVarNConfig, KVarNKVCache};

/// Deterministic small PRNG (the kv module fixtures' shape).
fn rng_next(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) & 0x7fff_ffff) as u32
}

fn rand_f32(state: &mut u64) -> f32 {
    (rng_next(state) as f32 / 0x7fff_ffff as f32) * 2.0 - 1.0
}

/// Factor-structured K-row population (n × d, flat row-major): each row is a
/// shared latent direction scaled per-row plus per-element noise — the
/// low-dimensional structure real K/V populations carry, so quantization
/// destroying it is measurable information loss (not noise-vs-noise).
fn population(n: usize, d: usize, seed: &mut u64) -> Vec<f32> {
    let latent: Vec<f32> = (0..d).map(|_| rand_f32(seed)).collect();
    let mut flat = vec![0.0f32; n * d];
    for i in 0..n {
        let a = rand_f32(seed);
        for j in 0..d {
            flat[i * d + j] = a * latent[j] + 0.3 * rand_f32(seed);
        }
    }
    flat
}

/// One probe record: the dCor² verdict/magnitude + the DV effect-size tuple
/// (reported, gauge-caveated) + the existing reconstruction pair.
#[derive(Debug, Clone)]
struct FidelityRecord {
    bits: u8,
    /// dCor² dependence magnitude ∈ [0, 1] — the ordering axis.
    dcor2: f32,
    /// dCor permutation p — the verdict field.
    p: f32,
    /// DV leave-one-out estimate in nats (null gauge critic- AND
    /// population-dependent — reported, never asserted against).
    loo_nats: f32,
    /// SMILE-clipped LOO (τ = 0.025).
    smile_loo_nats: f32,
    /// 8-fold spread of the λ=0 estimate.
    spread: f32,
    /// The existing reconstruction harness's tile MSE for the same population.
    tile_mse: f32,
    /// The existing reconstruction harness's tile cosine for the same population.
    tile_cosine: f32,
}

/// Probe one bit width through the shipped KVarN round-trip.
fn probe(bits: u8, w: &[f32], n: usize, d: usize, tile_size: usize, seed: u64) -> FidelityRecord {
    // Production round-trip: store every row, dequantize every row.
    let cfg = KVarNConfig {
        bits,
        kv_dim: d,
        tile_size,
        ..KVarNConfig::default()
    };
    let mut cache = KVarNKVCache::with_config(&cfg);
    for i in 0..n {
        cache.store_key(0, i, &w[i * d..(i + 1) * d]);
    }
    let mut wh = vec![0.0f32; n * d];
    for i in 0..n {
        cache.dequantize_key_into(0, i, &mut wh[i * d..(i + 1) * d]);
    }

    // Verdict + magnitude: the dCor² permutation test (distribution-free,
    // characteristic detector). Reseeded per run.
    let mut s = MiScratch::new(n, d, seed);
    let test = PermTest::new(128, seed);
    let dr = test.run_dcor(w, &wh, n, d, None, &mut s);

    // Effect size: one FrozenProj joint pass + one permutation pass → the DV
    // λ-family, then the SMILE variance control. Reported, not asserted.
    s.score_joint(Critic::FrozenProj, w, &wh, n, d);
    s.next_perm(n);
    s.score_perm(Critic::FrozenProj, w, &wh, n, d, PermSource::Current);
    let rep = dv_report(&s.joint[..n], &s.perm[..n]);
    let mut sort_buf = vec![0.0f64; n];
    let (_l0, smile_loo) =
        dv_smile_in_place(&mut s.joint[..n], &mut s.perm[..n], 0.025, &mut sort_buf);

    // The existing reconstruction metrics, same population (audit-only —
    // keys are probed; the harness takes keys+values, so keys ride both).
    let rows: Vec<Vec<f32>> = (0..n).map(|i| w[i * d..(i + 1) * d].to_vec()).collect();
    let eval = pseudo_decode_eval(&rows, &rows, tile_size, bits, &cfg.var_norm);

    FidelityRecord {
        bits,
        dcor2: dr.observed,
        p: dr.p,
        loo_nats: rep.loo,
        smile_loo_nats: smile_loo,
        spread: rep.spread,
        tile_mse: eval.per_tile_mse.first().copied().unwrap_or(f32::NAN),
        tile_cosine: eval.per_tile_cosine.first().copied().unwrap_or(f32::NAN),
    }
}

/// Ordering + retention: dependence magnitude strictly decreases with
/// coarser quantization and stays significant at the production 2-bit width.
#[test]
fn mi_probe_orders_information_retention_by_bit_width() {
    let n = 64;
    let d = 64;
    let tile_size = 64;
    let mut seed = 0x0051_7000u64;
    let w = population(n, d, &mut seed);

    let recs: Vec<FidelityRecord> = [8u8, 4, 2]
        .iter()
        .map(|&b| probe(b, &w, n, d, tile_size, 0x0051_7000))
        .collect();

    for r in &recs {
        println!(
            "bits={:>2}  dCor²={:.5}  p={:.4}  |  DV loo={:>8.3} nats (gauge-caveated)  smile={:>8.3}  spread={:.3}  |  mse={:.3e}  cos={:.5}",
            r.bits, r.dcor2, r.p, r.loo_nats, r.smile_loo_nats, r.spread, r.tile_mse, r.tile_cosine
        );
    }

    // Retention: significant at every production width (2-bit + VarNorm is
    // the shipped setting).
    for r in &recs {
        assert!(
            r.p < 0.05,
            "bits={}: dependence lost unexpectedly: {r:?}",
            r.bits
        );
    }
    // Ordering: strict monotone dependence magnitude (dCor², not DV — the
    // DV gauge is population/noise-dependent; see the module doc).
    assert!(
        recs[0].dcor2 > recs[1].dcor2 && recs[1].dcor2 > recs[2].dcor2,
        "dCor² must strictly decrease 8→4→2 bits: {:?}",
        recs.iter().map(|r| r.dcor2).collect::<Vec<_>>()
    );
    // All finite (a NaN anywhere means the probe instrument broke).
    for r in &recs {
        assert!(r.dcor2.is_finite() && r.p.is_finite() && r.tile_mse.is_finite());
    }
}

/// Non-vacuity control: a degenerate "dequantized" population (constant
/// rows) must report dependence lost — a probe that cannot fail is
/// documentation, not instrumentation.
#[test]
fn mi_probe_flags_degenerate_reconstruction() {
    let n = 64;
    let d = 64;
    let mut seed = 0x0005_1700u64;
    let w = population(n, d, &mut seed);
    let wh = vec![0.0f32; n * d]; // every row identical

    let mut s = MiScratch::new(n, d, 0x0005_1700);
    let test = PermTest::new(128, 0x0005_1700);
    let dr = test.run_dcor(&w, &wh, n, d, None, &mut s);
    assert!(
        dr.p >= 0.05,
        "constant reconstruction must lose dependence: p={:.4} dCor²={:.4}",
        dr.p,
        dr.observed
    );
}

/// Determinism: the probe record is bit-identical across runs.
#[test]
fn mi_probe_is_deterministic() {
    let n = 64;
    let d = 64;
    let tile_size = 64;
    let mut seed = 0x0000_7700u64;
    let w = population(n, d, &mut seed);

    let a = probe(4, &w, n, d, tile_size, 0x0005_1700);
    let b = probe(4, &w, n, d, tile_size, 0x0005_1700);
    assert_eq!(a.dcor2.to_bits(), b.dcor2.to_bits());
    assert_eq!(a.p.to_bits(), b.p.to_bits());
    assert_eq!(a.loo_nats.to_bits(), b.loo_nats.to_bits());
    assert_eq!(a.tile_mse.to_bits(), b.tile_mse.to_bits());
    assert_eq!(a.tile_cosine.to_bits(), b.tile_cosine.to_bits());
}

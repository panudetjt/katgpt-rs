#![cfg(feature = "set_admission")]
//! Plan 599 Phase 2 gate tests — the fan-out construction side.
//!
//! Covers the frozen direction bank (regen pin + unit/separation +
//! count agreement with the [`katgpt_core::sphere_exclusion_coverage`]
//! scoreboard), the tangent-cap θ-ladder (grounding + corpus identity +
//! T3.1 monotone separation), the local-PCA fan, the bounded
//! collapse→re-fan-out recovery loop, and the deterministic exp-tilt
//! resample arm.
//!
//! Whole-file `#![cfg]` + the `required-features` row in Cargo.toml: run
//! without the feature the target is SKIPPED (loud), never a green zero.

use katgpt_core::set_admission::{
    AdmissionScratch, BANK_OVERSAMPLE, BANK_SIZE, BANK_THRESHOLD, CAP_RUNGS, DIM, DIRECTION_BANK,
    FanOutReport, FanScratch, PcaScratch, SetAdmissionConfig, admit_into, admit_with_recovery_into,
    bank_sample_pool, certify_set, exclusion_centers_into, exp_tilt_resample_into,
    fan_cap_ladder_into, fan_pca_ladder_into, generate_direction_bank,
};
use katgpt_core::sphere_exclusion_coverage;

// ── local helpers (op-identical to the module's private ones) ────────

fn normalize(x: &[f32; DIM]) -> [f32; DIM] {
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mut out = [0.0_f32; DIM];
    for (o, v) in out.iter_mut().zip(x) {
        *o = v / n;
    }
    out
}

fn dot(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b).map(|(u, v)| u * v).sum()
}

fn unit(x: [f32; DIM]) -> [f32; DIM] {
    normalize(&x)
}

fn default_cfg() -> SetAdmissionConfig {
    SetAdmissionConfig::default()
}
/// Seeded random unit vectors (fastrand is a normal dep — same stream on
/// every platform).
fn random_unit_corpus(n: usize, seed: u64) -> Vec<[f32; DIM]> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut out = Vec::with_capacity(n);
    'next: while out.len() < n {
        let mut x = [0.0_f32; DIM];
        for v in x.iter_mut() {
            *v = rng.f32() * 2.0 - 1.0;
        }
        let acc: f32 = x.iter().map(|v| v * v).sum();
        if acc <= 1e-6 || !acc.is_finite() {
            continue 'next;
        }
        out.push(unit(x));
    }
    out.shrink_to_fit();
    out
}

/// Mean pairwise cosine of the returned fan rows (the spread instrument).
fn mean_pairwise_cos(rows: &[[f32; DIM]], n: usize) -> f32 {
    let mut total = 0.0_f32;
    let mut pairs = 0_usize;
    for i in 0..n {
        for j in i + 1..n {
            total += dot(&rows[i], &rows[j]);
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        total / pairs as f32
    }
}

fn n_slots(out: &[u16]) -> usize {
    out.iter().take_while(|&&i| i != u16::MAX).count()
}

/// The L2-ρ gaming family: `normalize(e0 ± 0.25·e_j)` alternating signs —
/// pairwise cos 0.9412 (passes the 0.95 cap) but spectrally one dominant
/// direction (the certificate flags it at defaults).
fn gaming_pool() -> Vec<[f32; DIM]> {
    let mut pool = Vec::with_capacity(DIM);
    for j in 1..DIM {
        for sign in [1.0_f32, -1.0] {
            if pool.len() == DIM {
                break;
            }
            let mut x = [0.0_f32; DIM];
            x[0] = 1.0;
            x[j] = 0.25 * sign;
            pool.push(unit(x));
        }
    }
    pool
}

// ── the frozen bank ──────────────────────────────────────────────────

#[test]
fn frozen_bank_matches_its_generator() {
    let (bank, written) = generate_direction_bank::<DIM>();
    assert_eq!(
        bank, DIRECTION_BANK,
        "the frozen table must be the generator's exact output"
    );
    assert!(
        written >= BANK_SIZE,
        "written {written} — the table must fill"
    );
    // Count agreement with the certified_frontier scoreboard: the
    // generator's picker mirrors that exact scan, and this pins it.
    let samples = bank_sample_pool::<DIM>();
    let cov = sphere_exclusion_coverage(&samples, BANK_THRESHOLD);
    let mut picked = [0_usize; 256];
    let picked_count = exclusion_centers_into(&samples, BANK_THRESHOLD, &mut picked);
    assert_eq!(
        picked_count, cov.centers,
        "the picker and the scoreboard must agree"
    );
    assert_eq!(
        cov.centers, written,
        "the picker and the scoreboard must agree"
    );
    assert!(
        !cov.saturated,
        "256-sample scan must not hit the center cap"
    );
}

#[test]
fn frozen_bank_rows_are_unit_and_separated() {
    let cos_cap = 1.0 - BANK_THRESHOLD * BANK_THRESHOLD / 2.0;
    for row in &DIRECTION_BANK {
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "row norm {norm} off the unit shell"
        );
    }
    let mut max_pair = 0.0_f32;
    for (i, a) in DIRECTION_BANK.iter().enumerate() {
        for b in &DIRECTION_BANK[i + 1..] {
            max_pair = max_pair.max(dot(a, b));
        }
    }
    println!("[bank] max pairwise cos {max_pair:.4} (exclusion cap {cos_cap:.4})");
    assert!(
        max_pair < cos_cap + 1e-4,
        "bank rows must respect the exclusion radius"
    );
}

#[test]
fn bank_pool_is_bit_deterministic() {
    let a = bank_sample_pool::<DIM>();
    let b = bank_sample_pool::<DIM>();
    assert_eq!(a.len(), BANK_OVERSAMPLE);
    assert_eq!(a, b, "the oversample pool must be bit-exact");
}

// ── the tangent-cap θ-ladder ─────────────────────────────────────────

#[test]
fn cap_fan_grounds_with_exact_corpus_identity() {
    let corpus = random_unit_corpus(64, 0x0599_0001);
    let query = unit([1.0; DIM]);
    let mut out_idx = [u16::MAX; 8];
    let mut out_pool = [[0.0_f32; DIM]; 8];
    let mut fan = FanScratch::new();
    let report = fan_cap_ladder_into(
        &query,
        &corpus,
        &CAP_RUNGS,
        0.3,
        &mut out_idx,
        &mut out_pool,
        &mut fan,
    );
    println!(
        "[cap] rung {} θ={:.2} grounded={} min_ground={:.3} distinct={}",
        report.rung, report.theta, report.grounded, report.min_ground, report.distinct_snaps
    );
    assert!(
        report.grounded,
        "a 64-point dense corpus must ground the ladder"
    );
    assert_eq!(n_slots(&out_idx), 8, "every slot must snap when grounded");
    assert!(report.min_ground >= 0.3, "grounded rung must clear τ");
    assert!(report.distinct_snaps >= 2, "a fan onto one point is no fan");
    for slot in 0..8 {
        let idx = out_idx[slot] as usize;
        assert!(idx < corpus.len(), "snapped index must be in-corpus");
        assert_eq!(
            out_pool[slot],
            normalize(&corpus[idx]),
            "the fan row must be the corpus point by identity"
        );
    }
}

#[test]
fn cap_fan_theta_ladder_monotone_separation() {
    // T3.1 monotonicity: a larger cone angle spreads the fan — the mean
    // pairwise angle of the returned pool strictly grows with θ.
    let corpus = random_unit_corpus(256, 0x0599_0002);
    let query = unit([1.0; DIM]);
    let fan_at = |theta: f32| {
        let mut out_idx = [u16::MAX; 8];
        let mut out_pool = [[0.0_f32; DIM]; 8];
        let mut fan = FanScratch::new();
        let report = fan_cap_ladder_into(
            &query,
            &corpus,
            &[theta],
            0.2,
            &mut out_idx,
            &mut out_pool,
            &mut fan,
        );
        (report, out_pool)
    };
    let (r1, p1) = fan_at(0.35);
    let (r2, p2) = fan_at(0.95);
    assert!(
        r1.grounded && r2.grounded,
        "both rungs must ground on a 256-point corpus"
    );
    let m1 = mean_pairwise_cos(&p1, 8);
    let m2 = mean_pairwise_cos(&p2, 8);
    println!("[cap] mean pairwise cos: θ=0.35 → {m1:.4}, θ=0.95 → {m2:.4}");
    assert!(
        m2 < m1,
        "larger θ must strictly spread the fan ({m2} !< {m1})"
    );
}

#[test]
fn cap_fan_degenerate_inputs_fail_clean() {
    let corpus = random_unit_corpus(16, 0x0599_0003);
    let mut out_idx = [u16::MAX; 4];
    let mut out_pool = [[0.0_f32; DIM]; 4];
    let mut fan = FanScratch::new();

    // Zero query: non-normalizable → failed report, outs untouched.
    let zero = [0.0_f32; DIM];
    let r = fan_cap_ladder_into(
        &zero,
        &corpus,
        &CAP_RUNGS,
        0.3,
        &mut out_idx,
        &mut out_pool,
        &mut fan,
    );
    assert_eq!(
        r,
        FanOutReport {
            rung: 0,
            theta: CAP_RUNGS[0],
            grounded: false,
            min_ground: 0.0,
            distinct_snaps: 0
        },
        "zero query fails clean"
    );
    assert!(
        out_idx.iter().all(|&i| i == u16::MAX),
        "failed fan leaves sentinels"
    );

    // Empty corpus: nothing snaps → never grounds, last rung's report.
    let query = unit([1.0; DIM]);
    let empty: Vec<[f32; DIM]> = Vec::new();
    let r = fan_cap_ladder_into(
        &query,
        &empty,
        &CAP_RUNGS,
        0.3,
        &mut out_idx,
        &mut out_pool,
        &mut fan,
    );
    assert!(!r.grounded);
    assert_eq!(
        r.rung,
        CAP_RUNGS.len() - 1,
        "the walk must exhaust the ladder"
    );

    // Single-point corpus: everything snaps to it, distinct = 1 < 2 —
    // the collapse-honesty bar keeps the report ungrounded.
    let single = vec![unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])];
    let r = fan_cap_ladder_into(
        &query,
        &single,
        &CAP_RUNGS,
        0.3,
        &mut out_idx,
        &mut out_pool,
        &mut fan,
    );
    assert!(
        !r.grounded,
        "a one-point fan is a collapse, never a grounding"
    );
    assert_eq!(r.distinct_snaps, 1);
}

// ── the local-PCA fan ────────────────────────────────────────────────

#[test]
fn pca_fan_grounds_on_clustered_corpus() {
    // A corpus whose local second moment is DOMINANT on one axis (e3):
    // a t-ladder of 24 points spread along e3 about e1 (the corpus
    // samples its own covariance axis — the anisotropic-cluster case the
    // local-PCA fan exists for), small noise elsewhere, + a diverse ring.
    // The anchor sits at t=0; the fan opens along the measured axis and
    // each slot snaps to its own ladder rung.
    let mut rng = fastrand::Rng::with_seed(0x0599_0004);
    let mut corpus = Vec::new();
    corpus.push([1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // anchor, t=0
    for i in 1..24 {
        let t = -1.5 + 3.0 * (i as f32) / 23.0;
        let mut x = [0.0_f32; DIM];
        x[0] = 1.0;
        x[2] = t;
        for v in x.iter_mut().skip(3) {
            *v = (rng.f32() * 2.0 - 1.0) * 0.05;
        }
        corpus.push(unit(x));
    }
    corpus.extend(random_unit_corpus(16, 0x0599_0005));
    let anchor = corpus[0];
    let mut out_idx = [u16::MAX; 4];
    let mut out_pool = [[0.0_f32; DIM]; 4];
    let mut fan = FanScratch::new();
    let mut pca = PcaScratch::new();
    let report = fan_pca_ladder_into(
        &anchor,
        &corpus,
        8,
        &CAP_RUNGS,
        0.3,
        &mut out_idx,
        &mut out_pool,
        &mut fan,
        &mut pca,
    );
    println!(
        "[pca] rung {} θ={:.2} grounded={} min_ground={:.3} distinct={}",
        report.rung, report.theta, report.grounded, report.min_ground, report.distinct_snaps
    );
    assert!(
        report.grounded,
        "the clustered world must ground the PCA ladder"
    );
    assert_eq!(n_slots(&out_idx), 4);
    for slot in 0..4 {
        let idx = out_idx[slot] as usize;
        assert_eq!(out_pool[slot], normalize(&corpus[idx]), "corpus identity");
    }

    // Determinism: the same inputs produce the same fan.
    let mut idx2 = [u16::MAX; 4];
    let mut pool2 = [[0.0_f32; DIM]; 4];
    let mut fan2 = FanScratch::new();
    let mut pca2 = PcaScratch::new();
    let report2 = fan_pca_ladder_into(
        &anchor, &corpus, 8, &CAP_RUNGS, 0.3, &mut idx2, &mut pool2, &mut fan2, &mut pca2,
    );
    assert_eq!(report, report2);
    assert_eq!(out_idx, idx2);
    assert_eq!(out_pool, pool2);
}

// ── the bounded collapse→re-fan-out recovery loop ────────────────────

#[test]
fn recovery_skips_when_initial_certificate_is_clean() {
    // Five orthogonal members: the certificate is clean — the loop must
    // not fire and the corpus output must stay sentinel-filled.
    let pool: Vec<[f32; DIM]> = (0..5)
        .map(|j| {
            let mut x = [0.0_f32; DIM];
            x[j] = 1.0;
            x
        })
        .collect();
    let quality = vec![1.0_f32; 5];
    let query = unit([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let corpus = random_unit_corpus(16, 0x0599_0006);
    let mut pool_idx = [u16::MAX; 5];
    let mut corpus_idx = [u16::MAX; 5];
    let mut scratch = AdmissionScratch::new();
    let mut fan = FanScratch::new();
    let outcome = admit_with_recovery_into(
        &default_cfg(),
        &pool,
        &quality,
        &query,
        &corpus,
        5,
        &mut pool_idx,
        &mut corpus_idx,
        &mut scratch,
        &mut fan,
    );
    assert!(!outcome.fired);
    assert!(outcome.recovered, "no fire is trivially recovered");
    assert_eq!(outcome.rungs_tried, 0);
    assert_eq!(outcome.admitted, 5);
    assert!(!outcome.certificate.collapsed);
    assert!(
        corpus_idx.iter().all(|&i| i == u16::MAX),
        "no fire leaves the corpus out sentinel"
    );

    // The pool output is the plain admission's output.
    let mut scratch2 = AdmissionScratch::new();
    let mut direct = [u16::MAX; 5];
    admit_into(
        &default_cfg(),
        &pool,
        &quality,
        &query,
        &mut direct,
        &mut scratch2,
    );
    assert_eq!(pool_idx, direct);
}

#[test]
fn recovery_fires_and_refans_a_collapsed_pool() {
    // The gaming pool passes the cap but the certificate flags it — the
    // loop fires, fans over a diverse corpus, and re-admits the snapped
    // corpus points alone (the collapsed pool is replaced, never merged).
    let pool = gaming_pool();
    let quality = vec![1.0_f32; pool.len()];
    let query = unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    // The world must actually be collapsed before the loop is trusted.
    assert!(
        certify_set(&default_cfg(), &pool).collapsed,
        "test world must be collapsed"
    );
    let corpus = random_unit_corpus(64, 0x0599_0007);
    const K: usize = 8;
    let mut pool_idx = [u16::MAX; K];
    let mut corpus_idx = [u16::MAX; K];
    let mut scratch = AdmissionScratch::new();
    let mut fan = FanScratch::new();
    let outcome = admit_with_recovery_into(
        &default_cfg(),
        &pool,
        &quality,
        &query,
        &corpus,
        K,
        &mut pool_idx,
        &mut corpus_idx,
        &mut scratch,
        &mut fan,
    );
    println!(
        "[recovery] fired={} recovered={} rungs={} admitted={} vendi={:.3}",
        outcome.fired,
        outcome.recovered,
        outcome.rungs_tried,
        outcome.admitted,
        outcome.certificate.vendi
    );
    assert!(outcome.fired, "the collapsed pool must fire the loop");
    assert!(outcome.rungs_tried >= 1);
    assert!(outcome.recovered, "a diverse corpus must recover the fan");
    assert!(!outcome.certificate.collapsed);
    assert!(
        pool_idx.iter().all(|&i| i == u16::MAX),
        "recovery replaces the pool"
    );
    let slots = n_slots(&corpus_idx);
    assert!(slots >= 2, "the re-admitted set must come from the fan");
    let mut distinct = 0_usize;
    for i in 0..slots {
        assert!(
            (corpus_idx[i] as usize) < corpus.len(),
            "recovery indices are corpus indices"
        );
        if !(0..i).any(|p| corpus_idx[p] == corpus_idx[i]) {
            distinct += 1;
        }
    }
    assert!(distinct >= 2, "the re-admitted fan must be a real fan");
}

#[test]
fn recovery_is_bounded_when_nothing_grounds() {
    // A one-point corpus can never pass the distinct ≥ 2 bar — the loop
    // walks every rung and gives up deterministically with both outputs
    // sentinel-filled.
    let pool = gaming_pool();
    let quality = vec![1.0_f32; pool.len()];
    let query = unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let corpus = vec![unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])];
    const K: usize = 8;
    let mut pool_idx = [u16::MAX; K];
    let mut corpus_idx = [u16::MAX; K];
    let mut scratch = AdmissionScratch::new();
    let mut fan = FanScratch::new();
    let outcome = admit_with_recovery_into(
        &default_cfg(),
        &pool,
        &quality,
        &query,
        &corpus,
        K,
        &mut pool_idx,
        &mut corpus_idx,
        &mut scratch,
        &mut fan,
    );
    assert!(outcome.fired);
    assert!(!outcome.recovered);
    assert_eq!(
        outcome.rungs_tried,
        CAP_RUNGS.len(),
        "the loop is bounded by the ladder"
    );
    assert!(pool_idx.iter().all(|&i| i == u16::MAX));
    assert!(
        corpus_idx.iter().all(|&i| i == u16::MAX),
        "no re-admission leaves sentinels"
    );
}

// ── the deterministic exp-tilt resample arm ──────────────────────────

#[test]
fn exp_tilt_uniform_at_zero_lambda_resamples_identity() {
    let psi = [0.3_f32, -1.0, 2.0, 0.0];
    let mut weights = [0.0_f32; 4];
    let mut ancestors = [0_u32; 4];
    exp_tilt_resample_into(&psi, 0.0, 0.5, &mut weights, &mut ancestors);
    for w in &weights {
        assert!((w - 0.25).abs() < 1e-6, "λ=0 is the uniform arm ({w})");
    }
    assert_eq!(
        ancestors,
        [0, 1, 2, 3],
        "uniform resample at u∈(0,1) is the identity"
    );
}

#[test]
fn exp_tilt_concentrates_on_max_psi() {
    let psi = [0.0_f32, 0.0, 0.0, 10.0];
    let mut weights = [0.0_f32; 4];
    let mut ancestors = [0_u32; 4];
    exp_tilt_resample_into(&psi, 5.0, 0.5, &mut weights, &mut ancestors);
    let total: f32 = weights.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "weights normalize ({total})");
    assert!(
        weights[3] > 0.999,
        "the max-ψ carrier takes the mass ({})",
        weights[3]
    );
    assert!(
        ancestors.iter().all(|&a| a == 3),
        "every resampled slot inherits the carrier"
    );
}

#[test]
fn exp_tilt_handles_non_finite_and_degenerate() {
    // A non-finite Ψ tilts to exactly zero weight; the rest still
    // normalizes and resamples deterministically.
    let psi = [3.0_f32, f32::NAN, 1.0, 2.0];
    let mut weights = [0.0_f32; 4];
    let mut ancestors = [0_u32; 4];
    exp_tilt_resample_into(&psi, 2.0, 0.5, &mut weights, &mut ancestors);
    assert_eq!(weights[1], 0.0, "NaN Ψ carries zero weight");
    let total: f32 = weights.iter().sum();
    assert!((total - 1.0).abs() < 1e-6);
    assert!(
        weights[0] > weights[3] && weights[3] > weights[2],
        "tilt order follows Ψ order"
    );
    assert!(
        ancestors.iter().all(|&a| (a as usize) < 4),
        "ancestors stay in range"
    );

    // All-degenerate: the uniform fallback (never NaN, never zero-total).
    let bad = [f32::NAN; 4];
    let mut w2 = [0.0_f32; 4];
    let mut a2 = [0_u32; 4];
    exp_tilt_resample_into(&bad, 2.0, 0.5, &mut w2, &mut a2);
    assert!(
        w2.iter().all(|w| (w - 0.25).abs() < 1e-6),
        "degenerate input falls back to uniform"
    );
    assert_eq!(a2, [0, 1, 2, 3]);

    // Deterministic at a fixed u.
    let mut w3 = [0.0_f32; 4];
    let mut a3 = [0_u32; 4];
    exp_tilt_resample_into(&psi, 2.0, 0.5, &mut w3, &mut a3);
    assert_eq!(weights, w3);
    assert_eq!(ancestors, a3);
}

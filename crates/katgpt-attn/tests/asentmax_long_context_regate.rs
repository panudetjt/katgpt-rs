//! Issue 762 T0 — long-context ASEntmax re-measure (the P0.7 caveat closer).
//!
//! The P0.7 re-gate (`asentmax_p07_realmodel_regate.rs`, Bench 713 addendum)
//! measured real-model routing only up to n = 32 blocks — its σ̂ ≈ 0.14
//! verdict and its "no over-sparsification on real rows" gate are therefore
//! unproven at the context lengths the ASEntmax schedule was designed for.
//! The paper's failure mode (fixed-α entmax support collapse) grows with the
//! candidate set, so the verdict must be re-earned at large n.
//!
//! This test replays the committed long-context fixture
//! (`tests/data/asentmax_long_context.fixture`, captured by
//! `examples/asentmax_p07_gen_fixture.rs --prompt-file
//! tests/data/asentmax_long_context_prompt.txt`) — ≈10.9k real-model tokens /
//! ≈170 blocks, 8 layers × 4 gqa-spread q-heads, log-spaced k-ends 4→170 —
//! through the production router in both arms (None vs
//! `with_asentmax_schedule()`), same harness as the P0.7 gate:
//!
//! 1. **σ̂ trend per n** (the headline): fresh estimator per n-bucket — does
//!    real routing σ̂ stay near the measured 0.14 as context grows, or climb
//!    toward the σ ≥ 1 over-sparsification regime where T4.3/T4.4 earn their
//!    keep?
//! 2. **Support vs n** — raw support must not collapse as n grows 32 → 170;
//!    scheduled must hold ≥ raw support; oracle-mass parity within −0.05.
//! 3. **Needle retention (deep)** — the planted access-code sentence at ~60%
//!    depth; rows whose oracle top-1 IS the needle block must keep it.
//! 4. **Latency at large n** — the schedule's O(n) estimator scan costs more
//!    at 170 blocks; the ≤ +50% P0.7 bound must still hold.
//!
//! Content caveat (honest scope): the long prompt is the P0.7 baseline with
//! its needle removed plus `.research/549` + `.research/489` appended — blocks
//! ≤ 32 are content-matched to the baseline fixture (direct comparability of
//! σ̂ at n ≤ 32); blocks > 32 are technical markdown, a distribution shift
//! the σ̂(n) trend read must acknowledge.
//!
//! Measured tables + verdict: `.benchmarks/713` (long-context addendum).

#![cfg(feature = "asentmax_schedule")]

use std::collections::HashMap;
use std::time::Instant;

use katgpt_attn::dash_attn::entmax_router::{EntmaxCache, EntmaxRouter};
use katgpt_attn::dash_attn::vortex_flow::{VortexFlow, VortexScratch};

mod common;
use common::{cache_for, mean, parse_fixture, replay, stream_n_blocks};

const FIXTURE: &[u8] = include_bytes!("data/asentmax_long_context.fixture");

// ── tests ───────────────────────────────────────────────────────────────────

/// Provenance + non-vacuity for the long-context surface. (Regenerate:
/// `cargo run --release -p katgpt-attn --features asentmax_schedule --example
/// asentmax_p07_gen_fixture -- --capture --prompt-file
/// tests/data/asentmax_long_context_prompt.txt --out
/// tests/data/asentmax_long_context.fixture`.)
#[test]
fn long_fixture_shape_and_provenance() {
    let fx = parse_fixture(FIXTURE);
    assert_eq!(fx.head_dim, 128, "qwen3 head_dim");
    assert_eq!(fx.block, 64, "DashAttnConfig default chunk_size");
    assert!(
        fx.n_tokens >= 10_000,
        "long-context prefill, got {} tokens",
        fx.n_tokens
    );

    // long capture profile: 8 layers × kv-heads {0,2,4,7} streams
    assert_eq!(
        fx.streams.len(),
        32,
        "8 layers × 4 kv-heads, got {}",
        fx.streams.len()
    );
    let layers: std::collections::HashSet<usize> = fx.streams.iter().map(|s| s.layer).collect();
    assert_eq!(
        layers,
        [1, 5, 10, 14, 19, 23, 28, 33].into_iter().collect(),
        "layer sample"
    );

    // rows: 8 layers × 4 q-heads × log-spaced k-ends
    let q_heads: std::collections::HashSet<usize> = fx.rows.iter().map(|r| r.q_head).collect();
    assert_eq!(
        q_heads,
        [0, 9, 17, 31].into_iter().collect(),
        "gqa-spread q-head sample"
    );
    assert!(
        fx.rows.len() >= 2_000,
        "routing rows, got {}",
        fx.rows.len()
    );

    // n-axis: log-spaced ends reach the full context
    let n_max = fx.rows.iter().map(|r| r.n_blocks).max().unwrap_or(0);
    assert!(n_max >= 160, "n-axis reaches ≥160 blocks, got {n_max}");
    let mut ns: Vec<usize> = fx.rows.iter().map(|r| r.n_blocks).collect();
    ns.sort_unstable();
    ns.dedup();
    assert!(
        ns.len() >= 60,
        "n-axis coverage (log-spaced), got {} points",
        ns.len()
    );
    // the first 29 ends are every-4 dense (the baseline-comparable prefix)
    assert!(
        (4..=32usize).step_by(4).all(|n| ns.contains(&n)),
        "dense n≤32 prefix for baseline comparison"
    );

    // provenance: captured from THIS committed prompt
    let prompt: &str = include_str!("data/asentmax_long_context_prompt.txt");
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in prompt.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    assert_eq!(
        fx.prompt_fnv, h,
        "long fixture prompt provenance drifted — regenerate"
    );

    // needle planted at ~60% depth (its own paragraph at the 549/489 boundary)
    assert!(
        fx.needle_block > n_max * 2 / 5 && fx.needle_block < n_max * 3 / 4,
        "needle block {} vs n_max {n_max} (expected ~60% depth)",
        fx.needle_block
    );

    // oracle masses sit on the simplex
    for row in fx.rows.iter().take(64) {
        let s: f32 = row.masses.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "masses sum {s}");
        assert!(row.masses.iter().all(|&m| m >= 0.0));
    }
}

/// σ̂ observed from a FRESH scheduled router per n-bucket — the headline
/// measurement. Each bucket's estimator sees only that bucket's rows (32 = 8
/// layers × 4 q-heads), so the trend attributes σ̂ to n rather than to EMA
/// carry-over. Measured (deterministic fixture): σ̂ CLIMBS from 0.140 at n=8
/// (the baseline's regime) to 0.347 at n=173 — but SATURATES far below the
/// σ ≥ 1 over-sparsification regime; the gate pins that band so a fixture
/// drift that changes the routing scale is caught.
#[test]
fn long_sigma_hat_trend_per_n() {
    let fx = parse_fixture(FIXTURE);
    let n_max = fx.rows.iter().map(|r| r.n_blocks).max().unwrap_or(0);
    let present: std::collections::HashSet<usize> = fx.rows.iter().map(|r| r.n_blocks).collect();
    let mut buckets: Vec<usize> = [8, 16, 32, 64, 126]
        .into_iter()
        .filter(|&n| present.contains(&n))
        .collect();
    if n_max > 128 {
        buckets.push(n_max);
    }
    println!("n     |σ̂(n)   |rows");
    for n in buckets {
        let idx: Vec<usize> = fx
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.n_blocks == n)
            .map(|(i, _)| i)
            .collect();
        assert!(idx.len() >= 16, "bucket n={n}: {} rows", idx.len());
        let router = EntmaxRouter::default_router().with_asentmax_schedule();
        let est = router.asentmax.as_ref().expect("estimator");
        let mut caches: HashMap<(usize, usize), EntmaxCache> = HashMap::new();
        let mut scratch = VortexScratch::new(256);
        for &i in &idx {
            let row = &fx.rows[i];
            let key = (row.layer, row.kv_head);
            let cache = caches.entry(key).or_insert_with(|| {
                cache_for(
                    &fx,
                    row.layer,
                    row.kv_head,
                    stream_n_blocks(&fx, row.layer, row.kv_head),
                )
            });
            let _ =
                router.forward_indexer(&row.query, cache, row.n_blocks, row.n_blocks, &mut scratch);
        }
        let sigma = est.resolve_sigma();
        println!("{n:5} |{sigma:6.3} |{}", idx.len());
        assert!(
            (0.05..=0.6).contains(&sigma),
            "σ̂({n}) = {sigma} outside the measured long-context band [0.05, 0.6] (climb-and-saturate below the σ≥1 regime)"
        );
        // the estimator observed real rows (not the virgin warm start)
        assert!(
            sigma.to_bits() != 1.0f32.to_bits(),
            "σ̂({n}) never left the warm start"
        );
    }
}

/// G2: support-vs-n + mass coverage at the full context — the P0.7 gate was
/// only measured to n = 32; the paper's over-sparsification failure mode
/// grows with the candidate set, so the no-collapse property must re-hold at
/// n = n_max. Gates: raw support does not collapse from n=32 → n_max,
/// scheduled holds ≥ raw support at n_max, and scheduled mean oracle-mass is
/// at parity-or-better vs raw (measured: +0.034 — the P0.7 "no gain"
/// verdict REVERSES at long context; the scheduled arm wins every n ≥ 24).
#[test]
fn long_g2_support_vs_n() {
    let fx = parse_fixture(FIXTURE);
    let raw = replay(&fx, false);
    let sched = replay(&fx, true);
    assert_eq!(raw.len(), sched.len());

    let support_at = |arm: &[common::Decision], n: usize| -> f32 {
        let s: Vec<f32> = fx
            .rows
            .iter()
            .zip(arm.iter())
            .filter(|(r, _)| r.n_blocks == n)
            .map(|(_, d)| d.blocks.len() as f32)
            .collect();
        mean(&s)
    };

    let n_max = fx.rows.iter().map(|r| r.n_blocks).max().unwrap_or(0);
    let present: std::collections::HashSet<usize> = fx.rows.iter().map(|r| r.n_blocks).collect();
    let report: Vec<usize> = [4, 8, 16, 24, 32, 48, 64, 94, 126, n_max]
        .into_iter()
        .filter(|&n| present.contains(&n))
        .collect();
    println!("n     |rawS |schdS |raw mass |schd mass");
    for &n in &report {
        let idx: Vec<usize> = fx
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.n_blocks == n)
            .map(|(i, _)| i)
            .collect();
        let rs: Vec<f32> = idx.iter().map(|&i| raw[i].blocks.len() as f32).collect();
        let ss: Vec<f32> = idx.iter().map(|&i| sched[i].blocks.len() as f32).collect();
        let mass = |arm: &Vec<common::Decision>| -> f32 {
            idx.iter()
                .map(|&i| {
                    let row = &fx.rows[i];
                    row.masses
                        .iter()
                        .enumerate()
                        .filter(|(b, _)| arm[i].blocks.contains(b))
                        .map(|(_, m)| m)
                        .sum::<f32>()
                })
                .sum::<f32>()
                / idx.len() as f32
        };
        let (rm, sm) = (mass(&raw), mass(&sched));
        println!(
            "{n:5} |{:5.1}|{:5.1} |{rm:8.4} |{sm:8.4}",
            mean(&rs),
            mean(&ss)
        );
    }

    // 1. no raw support collapse: n=32 → n_max (allow a small dip; the claim
    //    is "no over-sparsification collapse", not monotone growth).
    let raw32 = support_at(&raw, 32);
    let raw_max = support_at(&raw, n_max);
    let sched_max = support_at(&sched, n_max);
    println!(
        "raw support: n=32 {raw32:.2} → n={n_max} {raw_max:.2}; scheduled at n_max {sched_max:.2}"
    );
    assert!(
        raw_max >= raw32 - 1.0,
        "raw support collapses: {raw32:.2} at n=32 → {raw_max:.2} at n={n_max}"
    );
    assert!(raw_max >= 1.0, "raw support degenerate at n={n_max}");
    // 2. the schedule must not lose support vs raw at the full context
    assert!(
        sched_max + 1e-3 >= raw_max,
        "scheduled support {sched_max:.2} below raw {raw_max:.2} at n={n_max}"
    );
    // 3. mean oracle-mass parity over the whole long surface
    let raw_mean_mass: f32 = fx
        .rows
        .iter()
        .zip(raw.iter())
        .map(|(row, dec)| {
            row.masses
                .iter()
                .enumerate()
                .filter(|(b, _)| dec.blocks.contains(b))
                .map(|(_, m)| m)
                .sum::<f32>()
        })
        .sum::<f32>()
        / fx.rows.len() as f32;
    let sched_mean_mass: f32 = fx
        .rows
        .iter()
        .zip(sched.iter())
        .map(|(row, dec)| {
            row.masses
                .iter()
                .enumerate()
                .filter(|(b, _)| dec.blocks.contains(b))
                .map(|(_, m)| m)
                .sum::<f32>()
        })
        .sum::<f32>()
        / fx.rows.len() as f32;
    println!(
        "mean oracle mass: raw {raw_mean_mass:.4}, scheduled {sched_mean_mass:.4} (Δ {:+.4})",
        sched_mean_mass - raw_mean_mass
    );
    assert!(
        sched_mean_mass >= raw_mean_mass - 0.01,
        "scheduled arm loses mean oracle mass: {sched_mean_mass:.4} vs {raw_mean_mass:.4} (measured: scheduled +0.034 OVER raw)"
    );
    // weights sanity on a sample
    for arm in [&raw, &sched] {
        for dec in arm.iter().step_by(97) {
            let w: f32 = dec.weights.iter().sum();
            assert!(w <= 1.0 + 1e-4, "weights sum {w} > 1");
        }
    }
}

/// G2c: deep needle (block 94, ~55% depth) — MEASURED ORACLE-LIMITED. Unlike
/// the P0.7 baseline (needle at block 13: 54 rows top-1-concentrated, 96.3%
/// retention in both arms), the deep needle NEVER wins oracle top-1 on this
/// fixture — max mass 0.105, measured on BOTH prompt variants (needle
/// unreferenced by following content AND with an explicit retrieval-query
/// suffix). The 8B ternary model does not solve indirect-anaphora needle
/// retrieval at ~11k context on this content; the oracle itself sets the
/// ceiling, so router retention is measured, not gated, at top-1. The test
/// pins what IS measurable:
///
/// 1. fixture honesty — the diffuse-needle band (max needle mass in
///    [0.02, 0.60]); a fixture regen that suddenly concentrates the oracle
///    changes this axis and must re-pin.
/// 2. the rank axis — needle oracle-rank distribution over rows past the
///    needle, plus support-retention among rows whose oracle ranks the
///    needle in its top-8, against each arm's random-selection baseline
///    (mean support / n). Retention far above random means the router keys
///    on needle-relevant summaries even without oracle concentration.
#[test]
fn long_g2_deep_needle_oracle_limited() {
    let fx = parse_fixture(FIXTURE);
    let raw = replay(&fx, false);
    let sched = replay(&fx, true);
    let needle = fx.needle_block;
    let mut needle_mass_max = 0f32;
    let mut past = 0usize;
    let mut rank_sum = 0usize;
    let mut rank_min = usize::MAX;
    let mut top8 = 0usize;
    let mut top16 = 0usize;
    let mut raw_kept8 = 0usize;
    let mut sched_kept8 = 0usize;
    let mut raw_rand = 0f32; // Σ random-selection baseline, raw
    let mut sched_rand = 0f32;
    for (i, row) in fx.rows.iter().enumerate() {
        if row.n_blocks <= needle {
            continue;
        }
        past += 1;
        needle_mass_max = needle_mass_max.max(row.masses[needle]);
        // needle's oracle rank (0 = top-1)
        let mut rank = 0usize;
        for (b, &m) in row.masses.iter().enumerate() {
            if b != needle && m > row.masses[needle] {
                rank += 1;
            }
        }
        rank_sum += rank;
        rank_min = rank_min.min(rank);
        if rank < 8 {
            top8 += 1;
            if raw[i].blocks.contains(&needle) {
                raw_kept8 += 1;
            }
            if sched[i].blocks.contains(&needle) {
                sched_kept8 += 1;
            }
            raw_rand += raw[i].blocks.len() as f32 / row.n_blocks as f32;
            sched_rand += sched[i].blocks.len() as f32 / row.n_blocks as f32;
        }
        if rank < 16 {
            top16 += 1;
        }
    }
    assert!(past >= 200, "rows past the needle, got {past}");
    println!(
        "deep needle block {needle}: {past} rows past it; oracle rank min {rank_min} mean {:.1}; top-8 {top8}, top-16 {top16}; max needle mass {needle_mass_max:.3}",
        rank_sum as f64 / past as f64
    );
    if top8 > 0 {
        println!(
            "top-8 retention: raw {raw_kept8}/{top8} ({:.1}%, random baseline {:.1}%), sched {sched_kept8}/{top8} ({:.1}%, random baseline {:.1}%)",
            raw_kept8 as f64 / top8 as f64 * 100.0,
            raw_rand as f64 / top8 as f64 * 100.0,
            sched_kept8 as f64 / top8 as f64 * 100.0,
            sched_rand as f64 / top8 as f64 * 100.0
        );
    }
    // 1. fixture honesty: the diffuse-needle finding (measured 0.105 on both
    //    prompt variants). Out of band ⇒ the fixture content drifted.
    assert!(
        (0.02..=0.60).contains(&needle_mass_max),
        "deep-needle oracle regime drifted: max mass {needle_mass_max:.3} outside the measured diffuse band [0.02, 0.60]"
    );
    // 2. the oracle-limited verdict itself: top-1 concentration does not
    //    materialize at depth on this model/content — pinning it prevents a
    //    future regen from silently changing the axis's meaning.
    assert!(
        top8 >= 8,
        "needle never reaches oracle top-8 either ({top8} rows) — axis fully vacuous, re-pin"
    );
    // 3. the retention axis, pinned on the deterministic fixture (measured:
    //    raw 8/27 = 29.6% vs random 4.8%; sched 16/27 = 59.3% vs random
    //    9.6%) — both arms key on needle-relevant summaries far above
    //    chance, and the schedule's larger support retains ≥ the raw rate.
    //    A schedule change that loses deep-needle retention reds here even
    //    if the mass gates stay green.
    assert!(
        raw_kept8 * 20 >= top8 * 3,
        "raw deep-needle top-8 retention collapsed: {raw_kept8}/{top8} (measured 8/27 = 29.6%)"
    );
    assert!(
        sched_kept8 >= raw_kept8,
        "scheduled deep-needle top-8 retention below raw: {sched_kept8}/{top8} vs raw {raw_kept8}/{top8} (measured 16/27 vs 8/27)"
    );
}

/// G3: latency at large n — the schedule adds one powf + an O(n) estimator
/// scan per call, so its cost grows with the block count the P0.7 gate never
/// saw (n ≤ 32). Bound generously (≤ +50%) at n ≥ 96 while printing the
/// measured ratio.
#[test]
fn long_g3_latency_at_large_n() {
    let fx = parse_fixture(FIXTURE);
    let rows: Vec<&common::Row> = fx.rows.iter().filter(|r| r.n_blocks >= 96).collect();
    assert!(
        rows.len() >= 200,
        "large-n latency rows, got {}",
        rows.len()
    );
    let caches: Vec<EntmaxCache> = rows
        .iter()
        .map(|r| cache_for(&fx, r.layer, r.kv_head, r.n_blocks))
        .collect();

    let run = |scheduled: bool| -> f64 {
        let router = if scheduled {
            EntmaxRouter::default_router().with_asentmax_schedule()
        } else {
            EntmaxRouter::default_router()
        };
        let mut scratch = VortexScratch::new(256);
        for (row, cache) in rows.iter().zip(caches.iter()) {
            let _ =
                router.forward_indexer(&row.query, cache, row.n_blocks, row.n_blocks, &mut scratch);
        }
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            for (row, cache) in rows.iter().zip(caches.iter()) {
                let _ = router.forward_indexer(
                    &row.query,
                    cache,
                    row.n_blocks,
                    row.n_blocks,
                    &mut scratch,
                );
            }
        }
        t0.elapsed().as_secs_f64() / (iters * rows.len()) as f64
    };
    let raw_us = run(false) * 1e6;
    let sched_us = run(true) * 1e6;
    println!(
        "router latency/row at n≥96: raw {raw_us:.2} µs, scheduled {sched_us:.2} µs (ratio {:.3})",
        sched_us / raw_us
    );
    assert!(
        sched_us <= raw_us * 1.5 + 2.0,
        "scheduled arm latency {sched_us:.2} µs vs raw {raw_us:.2} µs at n≥96"
    );
}

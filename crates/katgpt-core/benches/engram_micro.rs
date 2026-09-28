//! Plan 299 T2.6 — Engram micro-benchmarks.
//!
//! Canonical criterion wiring for the G1 latency gate and the lower-level
//! primitives that feed it. Mirrors the convention of
//! [`micro_belief_bench.rs`] (`criterion_group!` + `criterion_main!`).
//!
//! # Benches covered
//!
//! - **T2.6 lookup** — `EngramTable::lookup_into` on a 1M-slot table with
//!   D=128 (paper's large-table regime). Target: < 50 ns amortized per K=16
//!   retrieval (i.e., the whole `lookup_into` call should take < 800 ns). The
//!   G1 gate in `tests/bench_299_engram_goat.rs` is the strict check; this
//!   bench is the regression watch.
//! - **multi_head_hash** — the suffix→K_MAX keys step. Should be a few ns;
//!   dominant cost is the K_MAX prime moduli.
//! - **sigmoid_fuse_into** — the per-pattern gate. Target: < 30 ns at D=128
//!   with SIMD engaged (NEON/AVX2).
//! - **hotswap_with_table** — `EngramHotSwap::with_table` steady-state read
//!   latency. Target: < 10 ns (single relaxed load + indirect call).
//! - **staging/update_slot** — `StagingEngramTable::update_slot` per-call
//!   latency (Plan 360 T2.6, target: < 50 ns at D=128). Regression watch for
//!   the G4 alloc gate (1 alloc/call) that lives in
//!   `tests/bench_360_engram_staging_goat.rs`.
//! - **staging/delete_slot** — `StagingEngramTable::delete_slot` per-call
//!   latency (target: < 10 ns, 0 allocs — just a `Vec::push(None)`).
//! - **staging/commit** — `StagingEngramTable::commit` latency at varying
//!   pending counts (1/10/100/1000) on a 4096-slot × D=64 table (~1 MB COW
//!   copy). Characterizes the fixed COW copy cost + per-mutation overhead.
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench engram_micro --features engram
//! ```
//!
//! # Feature gate
//!
//! Requires `engram` (Plan 299).

#![cfg(feature = "engram")]
#![allow(clippy::needless_range_loop)]

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::engram::{
    CanonicalId, EngramConfig, EngramHash, EngramTable, EngramTableBuilder, HashHead,
    InMemoryEngramTable, K_MAX, SigmoidFusionConfig, StagingEngramTable, fuse_into_hidden_state,
    multi_head_hash, sigmoid_fuse_into,
};

/// Large-table regime per Plan 299 T2.6 — 1M slots.
const LOOKUP_N_SLOTS: usize = 1_000_000;

/// Hidden-state dimension used by the G1 gate and the lookup bench. Matches
/// the paper's "large model" config (D=128).
const LOOKUP_D: usize = 128;

// ─── T2.6: lookup_into @ 1M slots × D=128 ───────────────────────────────────

fn bench_lookup_into_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/lookup_into");
    // criterion default sample_size is 100; bump to 500 for tighter CI on the
    // mean. Each iteration is one `lookup_into` call (~50 ns), so 500 iters
    // is well under criterion's per-bench time budget.
    group.sample_size(500);

    // Build the table outside the timed region. Populate ~1% of slots so hit
    // rate is realistic (matches G1 gate setup).
    let mut builder = EngramTableBuilder::new(LOOKUP_N_SLOTS, LOOKUP_D);
    let mut state = 0x00C0_FFEE_1234u64;
    let mut lcg = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    let n_populated = LOOKUP_N_SLOTS / 100;
    for _ in 0..n_populated {
        let slot = (lcg() as usize) % LOOKUP_N_SLOTS;
        let pat: Vec<f32> = (0..LOOKUP_D)
            .map(|_| (lcg() >> 40) as f32 / (1u64 << 24) as f32)
            .collect();
        builder.add_pattern(EngramHash(slot as u64), &pat);
    }
    let table = builder.build();

    // Pre-compute one set of K_MAX hash keys for the steady-state measurement.
    // (Inside the bench closure, we want to time ONLY `lookup_into`, not the
    // hash — `bench_multi_head_hash` measures that part separately.)
    let heads: [HashHead; K_MAX] = *table.heads();
    let suffix = [CanonicalId(1), CanonicalId(2), CanonicalId(3)];
    let keys = multi_head_hash(&suffix, &heads);

    // Pre-allocated output buffer — `lookup_into` is zero-allocation, and the
    // bench should reflect that.
    let mut out = vec![0.0f32; K_MAX * LOOKUP_D];

    group.bench_function("1m_slots_d128_k16", |b| {
        b.iter(|| {
            let hits = table.lookup_into(black_box(&keys), black_box(&mut out));
            black_box(hits);
        });
    });

    group.finish();
}

// ─── multi_head_hash ────────────────────────────────────────────────────────

fn bench_multi_head_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/multi_head_hash");
    group.sample_size(500);

    // Re-use the table's heads so the bench reflects realistic head configs.
    let table = EngramTableBuilder::new(1024, 16).build();
    let heads = *table.heads();

    // Three suffix lengths to characterize the fold cost: 1 (unigram),
    // 3 (trigram — the paper default), 8 (long context).
    let suffix_1 = vec![CanonicalId(7)];
    let suffix_3 = vec![CanonicalId(7), CanonicalId(11), CanonicalId(13)];
    let suffix_8: Vec<CanonicalId> = (0..8).map(|i| CanonicalId(100 + i as u64)).collect();

    group.bench_function("suffix_1", |b| {
        b.iter(|| multi_head_hash(black_box(&suffix_1), black_box(&heads)));
    });
    group.bench_function("suffix_3", |b| {
        b.iter(|| multi_head_hash(black_box(&suffix_3), black_box(&heads)));
    });
    group.bench_function("suffix_8", |b| {
        b.iter(|| multi_head_hash(black_box(&suffix_8), black_box(&heads)));
    });

    group.finish();
}

// ─── sigmoid_fuse_into ──────────────────────────────────────────────────────

fn bench_sigmoid_fuse_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/sigmoid_fuse_into");
    group.sample_size(500);

    // D=128 matches the G1 gate's table dim. The SIMD path engages when
    // D % 8 == 0; 128 satisfies that.
    let d = 128usize;
    let cfg = SigmoidFusionConfig {
        tau: (d as f32).sqrt(),
        rmsnorm_eps: 1e-6,
        logit_bias: 0.0,
    };

    let q: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..d).map(|i| (i as f32 * 0.07).cos()).collect();
    let v: Vec<f32> = (0..d).map(|i| (i as f32 * 0.05).tan() * 0.1).collect();
    let mut out = vec![0.0f32; d];

    group.bench_function("d128", |b| {
        b.iter(|| {
            sigmoid_fuse_into(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&mut out),
                black_box(&cfg),
            );
        });
    });

    group.finish();
}

// ─── fuse_into_hidden_state (end-to-end, K=16 retrievals + K gates) ─────────

fn bench_fuse_into_hidden_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/fuse_into_hidden_state");
    group.sample_size(500);

    let d = 128usize;
    let mut builder = EngramTableBuilder::new(4096, d);
    let mut state = 0xDEAD_BEEFu64;
    let mut lcg = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    for _ in 0..64 {
        let slot = (lcg() as usize) % 4096;
        let pat: Vec<f32> = (0..d)
            .map(|_| (lcg() >> 40) as f32 / (1u64 << 24) as f32)
            .collect();
        builder.add_pattern(EngramHash(slot as u64), &pat);
    }
    let table = builder.build();
    let heads = *table.heads();
    let keys = multi_head_hash(&[CanonicalId(1), CanonicalId(2), CanonicalId(3)], &heads);

    let mut hidden = vec![0.0f32; d];
    let query: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
    let cfg = EngramConfig::for_dim(d);

    let mut scratch_lookup = vec![0.0f32; K_MAX * d];
    let mut scratch_norm = vec![0.0f32; d];
    let mut scratch_out = vec![0.0f32; d];

    group.bench_function("d128_k16", |b| {
        b.iter(|| {
            // Reset hidden to keep the gate math identical across iterations.
            for x in hidden.iter_mut() {
                *x = 0.0;
            }
            fuse_into_hidden_state(
                black_box(&mut hidden),
                black_box(&query),
                black_box(&table),
                black_box(&keys),
                black_box(&cfg),
                black_box(&mut scratch_lookup),
                black_box(&mut scratch_norm),
                black_box(&mut scratch_out),
            );
        });
    });

    group.finish();
}

// ─── Plan 360 T2.6: StagingEngramTable micro-benchmarks ─────────────────────
//
// Regression watch for the GOAT gate in tests/bench_360_engram_staging_goat.rs.
// The GOAT gate is the load-bearing measurement; these criterion benches
// provide per-call latency characterization for CI regression tracking.

/// Table dim for the staging update/delete benches. Matches LOOKUP_D (D=128)
/// so the pattern memcpy cost is consistent with the rest of the engram suite.
const STAGING_D: usize = 128;

/// Table size for all staging benches. Small enough for many criterion
/// iterations without memory pressure; large enough that the COW copy in
/// `commit` is realistic (4096 × 64 × 4 = 1 MB).
const STAGING_N_SLOTS: usize = 4096;

/// Build a small engram table for the staging micro-benches. Populates ~10%
/// of slots with distinct patterns so the table has realistic non-zero
/// content (catches accidental zero-init shortcuts in the COW copy).
fn build_staging_table(n_slots: usize, d: usize) -> InMemoryEngramTable {
    let mut builder = EngramTableBuilder::new(n_slots, d);
    for i in 0..(n_slots / 10) {
        let pat: Vec<f32> = (0..d).map(|j| (i as f32 + j as f32 * 0.1).sin()).collect();
        builder.add_pattern(EngramHash(i as u64), &pat);
    }
    builder.build()
}

/// `update_slot` per-call latency at D=128.
///
/// Each iteration creates a fresh staging table with `with_capacity(1)` (so
/// the pending Vec never reallocs) and queues a single UPDATE. The measured
/// cost is: bounds check + length check + `to_vec` (1 alloc + D-f32 memcpy)
/// + `Vec::push`. Target: < 50 ns.
fn bench_staging_update_slot(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/staging/update_slot");
    group.sample_size(500);

    let table = build_staging_table(STAGING_N_SLOTS, STAGING_D);
    let pattern: Vec<f32> = (0..STAGING_D).map(|i| (i as f32 * 0.1).sin()).collect();

    group.bench_function("d128_per_call", |b| {
        b.iter_batched(
            || StagingEngramTable::with_capacity(&table, 1),
            |mut staging| {
                staging
                    .update_slot(black_box(0), black_box(&pattern))
                    .expect("in bounds, correct len");
                staging
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// `delete_slot` per-call latency.
///
/// Same structure as `update_slot` but queues a DELETE (None — no Vec<f32>
/// alloc). The measured cost is: bounds check + `Vec::push(None)`. Target:
/// < 10 ns. G4 proved 0 allocs/call.
fn bench_staging_delete_slot(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/staging/delete_slot");
    group.sample_size(500);

    let table = build_staging_table(STAGING_N_SLOTS, STAGING_D);

    group.bench_function("per_call", |b| {
        b.iter_batched(
            || StagingEngramTable::with_capacity(&table, 1),
            |mut staging| {
                staging.delete_slot(black_box(0)).expect("in bounds");
                staging
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// `commit` latency at varying pending counts.
///
/// Uses D=64 so the slots array is 1 MB (4096 × 64 × 4), isolating the
/// per-pending-mutation overhead on top of the fixed COW copy cost. The G2
/// gate in `tests/bench_360` characterizes the 1M-slot regime; this bench
/// characterizes the small-table regime typical of GM-edit workloads.
///
/// Pending counts: 1, 10, 100, 1000 — spanning the realistic GM-edit range.
/// Each iteration pre-loads N mutations in the untimed setup, then times
/// only `commit()` (COW slots clone + heads clone + N mutation applications).
fn bench_staging_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("engram/staging/commit");
    group.sample_size(100);

    let d = 64usize;
    let table = build_staging_table(STAGING_N_SLOTS, d);

    for &n_pending in &[1usize, 10, 100, 1000] {
        // Pre-build N distinct patterns so the setup doesn't re-derive them
        // each iteration. Update slots 0..N (wraps at n_slots, which is 4096
        // here so no wrap for n_pending ≤ 1000).
        let patterns: Vec<Vec<f32>> = (0..n_pending)
            .map(|i| vec![(i as f32 * 0.1).sin(); d])
            .collect();

        let name = format!("p{n_pending:04}_d{d:03}_{STAGING_N_SLOTS}slots");

        group.bench_function(&name, |b| {
            b.iter_batched(
                || {
                    let mut staging = StagingEngramTable::with_capacity(&table, n_pending);
                    for (i, p) in patterns.iter().enumerate() {
                        staging.update_slot(i, p).expect("in bounds, correct len");
                    }
                    staging
                },
                |mut staging| {
                    let _new = staging.commit().expect("has pending mutations");
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lookup_into_1m,
    bench_multi_head_hash,
    bench_sigmoid_fuse_into,
    bench_fuse_into_hidden_state,
    bench_staging_update_slot,
    bench_staging_delete_slot,
    bench_staging_commit,
);
criterion_main!(benches);

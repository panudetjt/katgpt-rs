//! KARC Batched MatVec — Plan 556 Phase 2 G2 perf bench.
//!
//! Target: N=8 batched forecast ≤ 575 ns (= 1.5× single-forecast latency at
//! the HLA config, D=8/M=8/K=4). The amortization target is ≥5.3× over N
//! sequential forecasts. The win comes from contiguous `Wout` layout +
//! loop hoisting, NOT from rayon parallelism (N=8 is below the rayon
//! threshold; ~5µs scheduling overhead dwarfs the entire batched budget).
//!
//! Three sub-benches:
//! 1. `pure_matvec` — `karc_batched_matvec_into` alone (no feature_expand).
//!    The pure matvec amortization — the headline perf number.
//! 2. `batched_forecast` — full `KarcBatchForecaster::forecast_into` (with
//!    feature_expand). The end-to-end perf including per-NPC basis eval.
//! 3. `sequential_n` — N sequential `KarcForecaster::forecast_into` calls.
//!    The baseline. Same `Wout` matrices + delay states as the batched path.
//!
//! Compares sequential-vs-batched at N=1, N=4, N=8, N=16, N=32.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::{FourierBasis, KarcBatchForecaster, KarcForecaster, karc_batched_matvec_into};

/// Build a fitted `KarcForecaster` with a deterministic synthetic trajectory.
/// Returns the forecaster + the seed delay state.
fn make_fitted<const D: usize, const M: usize, const K: usize>(
    basis: FourierBasis<M>,
    n_train: usize,
    seed_offset: usize,
) -> (KarcForecaster<FourierBasis<M>, D, M, K>, Vec<f32>) {
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
    let mut f = KarcForecaster::with_capacity(basis, n_total);
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
    f.fit_ridge(1e-4).expect("fit_ridge");
    let mut seed = vec![0.0f32; kd];
    for lag in 0..K {
        let idx = (n_total - 1 - seed_offset) - lag;
        for d in 0..D {
            seed[lag * D + d] = traj[idx * D + d];
        }
    }
    (f, seed)
}

const D: usize = 8;
const M: usize = 8;
const K: usize = 4;
// d_h = K * D * M = 256

/// Build N fitted forecasters with different seed offsets (so each NPC has a
/// different Wout + delay state), plus the batched forecaster that wraps them.
fn make_batch(
    n: usize,
) -> (
    Vec<KarcForecaster<FourierBasis<M>, D, M, K>>,
    KarcBatchForecaster<FourierBasis<M>, D, M, K>,
    Vec<f32>, // stacked delay states [N*K*D]
    Vec<f32>, // stacked features [N*d_h] (precomputed for the pure-matvec bench)
) {
    let mut singles: Vec<KarcForecaster<FourierBasis<M>, D, M, K>> = Vec::with_capacity(n);
    let mut seeds: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let (f, seed) = make_fitted::<D, M, K>(FourierBasis::new(4.0), 200, i);
        singles.push(f);
        seeds.push(seed);
    }
    let mut batch =
        KarcBatchForecaster::<FourierBasis<M>, D, M, K>::with_capacity(FourierBasis::new(4.0), n);
    for (i, f) in singles.iter().enumerate() {
        batch.set_wout(i, f.wout.clone());
    }
    let mut delay_states = vec![0.0f32; n * K * D];
    for (i, seed) in seeds.iter().enumerate() {
        delay_states[i * K * D..(i + 1) * K * D].copy_from_slice(seed);
    }
    // Precompute features for the pure-matvec bench by running one batched
    // forecast (which populates `features_buf` internally — we replicate the
    // same expansion here).
    let d_h = K * D * M;
    let mut features = vec![0.0f32; n * d_h];
    // We can't access the batch's internal features_buf, but for the pure-
    // matvec bench we just need *some* features. Use a deterministic pattern.
    for (i, feat) in features.iter_mut().enumerate() {
        *feat = (i as f32 * 0.001) - 0.5;
    }
    (singles, batch, delay_states, features)
}

fn bench_pure_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_556_karc_batched_matvec_pure");
    group.throughput(Throughput::Elements(1));

    for &n in &[1usize, 4, 8, 16, 32] {
        let (_, _, _, features) = make_batch(n);
        let d_h = K * D * M;
        // Build stacked Wouts matching the singles.
        let mut wouts = Vec::with_capacity(n * D * d_h);
        let mut singles_collector: Vec<KarcForecaster<FourierBasis<M>, D, M, K>> =
            Vec::with_capacity(n);
        for i in 0..n {
            let (f, _) = make_fitted::<D, M, K>(FourierBasis::new(4.0), 200, i);
            wouts.extend_from_slice(&f.wout);
            singles_collector.push(f);
        }
        let mut out = vec![0.0f32; n * D];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                karc_batched_matvec_into(
                    black_box(&wouts),
                    black_box(&features),
                    black_box(&mut out),
                    n,
                    d_h,
                    D,
                );
            });
        });
        std::hint::black_box(&singles_collector); // sink — prevent drop optimization
    }

    group.finish();
}

fn bench_batched_forecast(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_556_karc_batched_forecast_full");
    group.throughput(Throughput::Elements(1));

    for &n in &[1usize, 4, 8, 16, 32] {
        let (_, mut batch, delay_states, _) = make_batch(n);
        let mut out = vec![0.0f32; n * D];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                batch.forecast_into(black_box(&delay_states), black_box(&mut out));
            });
        });
    }

    group.finish();
}

fn bench_sequential_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_556_karc_sequential_baseline");
    group.throughput(Throughput::Elements(1));

    for &n in &[1usize, 4, 8, 16, 32] {
        let (mut singles, _, delay_states, _) = make_batch(n);
        let mut out = vec![0.0f32; n * D];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                for i in 0..n {
                    let ok = singles[i].forecast_into(
                        black_box(&delay_states[i * K * D..(i + 1) * K * D]),
                        black_box(&mut out[i * D..(i + 1) * D]),
                    );
                    black_box(ok);
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pure_matvec,
    bench_batched_forecast,
    bench_sequential_n
);
criterion_main!(benches);

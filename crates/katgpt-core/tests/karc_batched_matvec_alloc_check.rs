//! KARC Batched MatVec zero-allocation test — GOAT gate G4 (Plan 556 Phase 2).
//!
//! `KarcBatchForecaster::forecast_into` must not allocate heap memory on the
//! hot path. The feature scratch is pre-allocated at construction and reused
//! via indexing. Verified with a manual `GlobalAlloc` counter.

use katgpt_core::{FourierBasis, KarcBatchForecaster, KarcForecaster};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

const D: usize = 8;
const M: usize = 8;
const K: usize = 4;
const N: usize = 8;

/// Fit N single forecasters with different seed offsets (deterministic — same
/// as the bench). Returns the forecasters (for Wout reuse) + their seed delay
/// states.
fn make_n_fitted_singles() -> (Vec<KarcForecaster<FourierBasis<M>, D, M, K>>, Vec<Vec<f32>>) {
    let mut singles: Vec<KarcForecaster<FourierBasis<M>, D, M, K>> = Vec::with_capacity(N);
    let mut seeds: Vec<Vec<f32>> = Vec::with_capacity(N);
    for i in 0..N {
        let traj: Vec<f32> = (0..200)
            .flat_map(|j| {
                let t = j as f32 * 0.05;
                let mut row = [0.0f32; 32];
                for (d, row_d) in row.iter_mut().enumerate().take(D) {
                    let freq = 0.3 + 0.2 * d as f32;
                    *row_d = (freq * t).sin() + 0.5 * ((freq + 1.0) * t).cos();
                }
                row[..D].to_vec()
            })
            .collect();
        let n_total = traj.len() / D;
        let mut f = KarcForecaster::with_capacity(FourierBasis::new(4.0), n_total);
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
            let idx = (n_total - 1 - i) - lag;
            for d in 0..D {
                seed[lag * D + d] = traj[idx * D + d];
            }
        }
        singles.push(f);
        seeds.push(seed);
    }
    (singles, seeds)
}

#[test]
fn g4_batched_forecast_into_zero_alloc_after_warmup() {
    const N_CALLS: usize = 1000;

    let (singles, seeds) = make_n_fitted_singles();

    let mut batch =
        KarcBatchForecaster::<FourierBasis<M>, D, M, K>::with_capacity(FourierBasis::new(4.0), N);
    for (i, f) in singles.iter().enumerate() {
        batch.set_wout(i, f.wout.clone());
    }

    let mut delay_states = vec![0.0f32; N * K * D];
    for (i, seed) in seeds.iter().enumerate() {
        delay_states[i * K * D..(i + 1) * K * D].copy_from_slice(seed);
    }
    let mut out = vec![0.0f32; N * D];

    // Warmup: settle any lazy allocations.
    for _ in 0..10 {
        batch.forecast_into(&delay_states, &mut out);
    }

    // Measure: snapshot alloc/dealloc counts, run N_CALLS batched forecasts,
    // expect zero delta.
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    let mut total: f32 = 0.0;
    for _ in 0..N_CALLS {
        batch.forecast_into(&delay_states, &mut out);
        total += out[0]; // sink
    }

    let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_delta = alloc_after - alloc_before;
    let dealloc_delta = dealloc_after - dealloc_before;

    std::hint::black_box(total);

    assert_eq!(
        alloc_delta, 0,
        "G4 FAIL: KarcBatchForecaster::forecast_into allocated {alloc_delta} times in {N_CALLS} calls"
    );
    assert_eq!(
        dealloc_delta, 0,
        "G4 FAIL: KarcBatchForecaster::forecast_into deallocated {dealloc_delta} times in {N_CALLS} calls"
    );
}

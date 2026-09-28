#![cfg(feature = "set_admission")]
//! Plan 599 T3.4 — set-admission zero-allocation gate.
//!
//! `admit_into` + `certify_set` + `fan_cap_ladder_into` must not allocate
//! heap memory after warmup: admission works in the fixed
//! `AdmissionScratch` ([f32; 64] Sherman–Morrison scratch + reusable
//! admitted list), the certificate in fixed d×d Jacobi scratch, the fan in
//! `FanScratch` (the corpus list reserves ONCE then refills — cleared per
//! call, so steady-state cycles are alloc-free). Separate single-fn test
//! binary (global counting allocator convention — `#[global_allocator]` is
//! binary-unique).

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::set_admission::{
    AdmissionScratch, CAP_RUNGS, DIM, FanScratch, SetAdmissionConfig, admit_into, certify_scratch,
    fan_cap_ladder_into,
};
use std::sync::atomic::Ordering;

fn unit(x: [f32; DIM]) -> [f32; DIM] {
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mut out = [0.0_f32; DIM];
    for (o, v) in out.iter_mut().zip(x) {
        *o = v / n;
    }
    out
}

fn seeded(n: usize, seed: u64) -> Vec<[f32; DIM]> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let z = s ^ (s >> 32);
        ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    (0..n)
        .map(|_| {
            let mut x = [0.0_f32; DIM];
            for v in x.iter_mut() {
                *v = next();
            }
            unit(x)
        })
        .collect()
}

#[test]
fn set_admission_zero_alloc_over_1000_cycles() {
    let cfg = SetAdmissionConfig::default();
    let pool = seeded(512, 0x0599_0C01);
    let corpus = seeded(64, 0x0599_0C02);
    let query = unit({
        let mut x = [0.0_f32; DIM];
        for v in x.iter_mut() {
            *v = 0.5;
        }
        x
    });
    let quality: Vec<f32> = (0..pool.len())
        .map(|i| 0.9 + 0.1 * (i % 3) as f32 / 2.0)
        .collect();

    let mut scratch = AdmissionScratch::new();
    let mut fan = FanScratch::new();
    let mut out = [u16::MAX; 8];
    let mut fan_idx = [u16::MAX; 8];
    let mut fan_pool = [[0.0_f32; DIM]; 8];

    // Warmup: the scratch admitted list + fan corpus list grow to capacity
    // here (reserve-once), prime the Jacobi scratch.
    admit_into(&cfg, &pool, &quality, &query, &mut out, &mut scratch);
    let _ = certify_scratch(&cfg, &scratch);
    let _ = fan_cap_ladder_into(
        &query,
        &corpus,
        &CAP_RUNGS,
        0.3,
        &mut fan_idx,
        &mut fan_pool,
        &mut fan,
    );

    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    const CYCLES: usize = 1000;
    let mut sink = 0_usize;
    for cycle in 0..CYCLES {
        let n = admit_into(&cfg, &pool, &quality, &query, &mut out, &mut scratch);
        let cert = certify_scratch(&cfg, &scratch);
        // Alternate cycles exercise the fan too (steady-state refills).
        if cycle % 2 == 0 {
            let _ = fan_cap_ladder_into(
                &query,
                &corpus,
                &CAP_RUNGS,
                0.3,
                &mut fan_idx,
                &mut fan_pool,
                &mut fan,
            );
            sink += fan_idx[0] as usize;
        }
        sink += n + cert.vendi.to_bits() as usize;
    }
    std::hint::black_box(sink);

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
    assert_eq!(
        alloc_delta, 0,
        "set admission allocated {alloc_delta} times over {CYCLES} cycles (expected 0)"
    );
    assert_eq!(
        dealloc_delta, 0,
        "set admission deallocated {dealloc_delta} times over {CYCLES} cycles (expected 0)"
    );
}

//! Issue 664 G4 — zero-allocation steady-state audit for the UGC estimator +
//! sampler hot paths (separate binary so the CountingAllocator global picks
//! up only this test's allocations — the karc_alloc_check pattern).
//!
//! `estimate_interval` and `bernoulli_unmask_with_grid` must perform ZERO
//! heap allocations after `UgcScratch` construction (the amortized-once
//! constructor is exempt by design — G2 amortization).

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::types::Rng;
use katgpt_core::ugc_schedule::*;

/// Noisy repeated bit (test-side copy, alphabet 2).
struct NoisyBit {
    d: usize,
    eta: f64,
}

impl UgcDenoiser for NoisyBit {
    fn dim(&self) -> usize {
        self.d
    }
    fn alphabet(&self) -> usize {
        2
    }
    fn posterior_into(&self, i: usize, x: &[usize], out: &mut [f32]) {
        let mut m0 = 0usize;
        let mut j = 0usize;
        for (l, &v) in x.iter().enumerate() {
            if l != i && v != UGC_MASK {
                j += 1;
                if v == 0 {
                    m0 += 1;
                }
            }
        }
        let odds = ((1.0 - self.eta) / self.eta).powi((2 * m0 as i32) - j as i32);
        let p0 = odds / (1.0 + odds);
        out[0] = ((1.0 - self.eta) * p0 + self.eta * (1.0 - p0)) as f32;
        out[1] = 1.0 - out[0];
    }
}

#[test]
fn g4_zero_alloc_steady_state() {
    use std::sync::atomic::Ordering;
    let d = 32usize;
    let dz = NoisyBit { d, eta: 0.2 };
    let mut rng = Rng::new(1);
    let m = 32usize;
    let mut scratch = UgcScratch::new(d, 2, m, 64);
    let mut out = vec![0usize; d];

    // Warm-up (any lazy growth settles).
    for _ in 0..3 {
        let _ = estimate_interval(&dz, 0.1, 0.9, m, 0.1, &mut rng, &mut scratch);
        bernoulli_unmask_with_grid(
            &dz,
            &[0.2, 0.5, 0.8],
            &mut rng,
            &mut scratch,
            &mut out,
            None,
        );
    }

    let a0 = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..50 {
        let _ = estimate_interval(&dz, 0.1, 0.9, m, 0.1, &mut rng, &mut scratch);
        bernoulli_unmask_with_grid(
            &dz,
            &[0.2, 0.5, 0.8],
            &mut rng,
            &mut scratch,
            &mut out,
            None,
        );
    }
    let delta = ALLOC_COUNT.load(Ordering::Relaxed) - a0;
    eprintln!("G4: {delta} allocations across 50 estimate_interval + 50 sampler calls");
    assert_eq!(delta, 0, "steady-state allocations detected: {delta}");
}

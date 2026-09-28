//! G4 — allocation-free steady-state tick for `mb_value` (Issue 767 T7).
//!
//! `code_into` + `value` + `dopamine_update` must be allocation-free in
//! steady state: the top-k selection, drive computation, readout, and the
//! bounded update all reuse preallocated scratch. The fixture is stationary
//! — a fixed cycling stream of (features, action, rpe) triples whose
//! per-cycle behavior is identical every cycle.
//!
//! The counting allocator (tests/common) is per-thread (Issue 714) — this
//! test is single-threaded by construction.

#![cfg(feature = "mb_value")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::mb_value::{MbCircuit, MbCircuitConfig, MbScratch};
use std::sync::atomic::Ordering;

#[inline]
fn thread_alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

/// A small local fixture RNG (deterministic; the counting allocator forbids
/// nothing here but determinism keeps the stream stationary).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f32 {
        let bits = ((self.next() >> 40) as u32 & 0x007f_ffff) | 0x3f80_0000;
        f32::from_bits(bits) - 1.0
    }
}

#[test]
fn steady_state_tick_is_alloc_free() {
    let cfg = MbCircuitConfig::toy();
    let mut circuit = MbCircuit::new(&cfg);
    let mut rng = Rng(31);
    let n = 400;
    let mut feats = vec![0.0f32; n * cfg.n_features];
    for f in feats.iter_mut() {
        *f = rng.uniform() * 2.0 - 1.0;
    }
    let actions = vec![1.0f32, 0.0, 0.0, 1.0];
    circuit.calibrate(&feats, &actions, n, 0.5);

    // A stationary cycling stream: 32 fixed feature vectors, 2 actions,
    // rpe cycling over a small fixed set.
    let mut stream: Vec<(Vec<f32>, usize, f32)> = Vec::new();
    for i in 0..32 {
        let mut f = vec![0.0f32; cfg.n_features];
        for (j, x) in f.iter_mut().enumerate() {
            *x = ((i as f32 * 0.137) + j as f32 * 0.711).sin();
        }
        let rpe = [1.0f32, -0.5, 0.25, -1.0][i % 4];
        stream.push((f, i % 2, rpe));
    }

    let mut scratch = MbScratch::new(&circuit);
    let mut code = vec![0u32; circuit.kc_active()];

    // Warmup: one full cycle primes every path.
    for (f, a, rpe) in &stream {
        let a_slice: &[f32] = if *a == 0 { &[1.0, 0.0] } else { &[0.0, 1.0] };
        circuit.code_into(f, a_slice, &mut scratch, &mut code);
        let _ = circuit.value(&code, &mut scratch);
        circuit.dopamine_update(&code, *rpe);
    }

    let before = thread_alloc_count();
    for _cycle in 0..64 {
        for (f, a, rpe) in &stream {
            let a_slice: &[f32] = if *a == 0 { &[1.0, 0.0] } else { &[0.0, 1.0] };
            circuit.code_into(f, a_slice, &mut scratch, &mut code);
            let _ = circuit.value(&code, &mut scratch);
            circuit.dopamine_update(&code, *rpe);
        }
    }
    let after = thread_alloc_count();
    assert_eq!(
        before,
        after,
        "steady-state code/value/update must not allocate (delta {})",
        after - before
    );
}

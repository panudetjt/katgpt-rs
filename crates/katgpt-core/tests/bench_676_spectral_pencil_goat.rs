//! Issue 676 — spectral_pencil T10 GOAT gate (G1 determinism + G4
//! zero-alloc; G2 latency lives in `benches/spectral_pencil_bench.rs`).
//!
//! Separate test binary (the CountingAllocator global pattern —
//! `analytic_lattice_alloc_check.rs` / `karc_alloc_check.rs` precedent).
//! **All checks live in ONE test function**: the tests within a binary
//! run in parallel and share the global counter — thread-spawn
//! bookkeeping allocations from a sibling test would land in the
//! measurement window (caught live: 12 phantom "leaks" that bisected to
//! zero on every path in isolation). One function = serial by
//! construction.

#![cfg(feature = "spectral_pencil")]

use katgpt_core::spectral_pencil::dense::DenseScratch;
use katgpt_core::spectral_pencil::init::{seeded_dense, seeded_tridiag};
use katgpt_core::spectral_pencil::tridiag::TriScratch;
use katgpt_core::spectral_pencil::{DensePencil, TridiagPencil};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// T10 GOAT: G4 zero-alloc steady state on every hot path (dense eval,
/// tridiag eval, Sturm count, attribution) + G1 bit-identical
/// determinism across independent constructions + headroom-shape sanity
/// — one function, serial by construction.
#[test]
fn spectral_pencil_t10_goat_serial() {
    // ── G4: zero steady-state allocs ─────────────────────────────────
    {
        const D: usize = 16;
        const N: usize = 8;
        const CALLS: usize = 1_000;

        let init = seeded_dense::<D, N>(b"goat-alloc", 6);
        let pencil = DensePencil::<D, N> {
            a0: init.a0,
            a: init.a,
        };
        let tri_init = seeded_tridiag::<D, N>(b"goat-alloc-tri", 6);
        let tri_pencil = TridiagPencil::<D, N> {
            a0: tri_init.a0,
            a: tri_init.a,
        };
        let mut dscratch = DenseScratch::<D>::new();
        let mut tscratch = TriScratch::<D>::new();
        let mut x = [0.5_f32; N];

        // Warmup: settle any lazy allocations.
        for i in 0..8 {
            x[0] = (i % 5) as f32 * 0.3;
            let _ = pencil.eval(&x, 6, &mut dscratch);
            let _ = tri_pencil.eval(&x, 6, &mut tscratch);
            let _ = tri_pencil.count_below(&x, 0.25, &mut tscratch);
            let _ = katgpt_core::spectral_pencil::attribution::attribute(
                &pencil.a0,
                &pencil.a,
                &x,
                6,
                0.1,
                &mut dscratch,
            );
        }

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
        let mut sink = 0.0_f32;
        for i in 0..CALLS {
            x[1] = (i % 7) as f32 * 0.2 - 0.6;
            sink += pencil.eval(&x, 6, &mut dscratch).value;
            sink += tri_pencil.eval(&x, 6, &mut tscratch);
            sink += tri_pencil.count_below(&x, 0.25, &mut tscratch) as f32;
            let rep = katgpt_core::spectral_pencil::attribution::attribute(
                &pencil.a0,
                &pencil.a,
                &x,
                6,
                0.1,
                &mut dscratch,
            );
            sink += rep.influence[0];
        }

        let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
        let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
        std::hint::black_box(&sink);
        assert_eq!(
            alloc_delta, 0,
            "steady-state allocs leaked on the hot paths ({alloc_delta} allocs / {dealloc_delta} deallocs)"
        );
        assert_eq!(dealloc_delta, 0);
    }

    // ── G1: bit-identical across independent constructions ───────────
    {
        const D: usize = 12;
        const N: usize = 6;
        let mut folds = [0_u64; 2];
        for fold in folds.iter_mut() {
            let init = seeded_dense::<D, N>(b"goat-repro/run", 4);
            let pencil = DensePencil::<D, N> {
                a0: init.a0,
                a: init.a,
            };
            let mut scratch = DenseScratch::<D>::new();
            let mut h = 0xfeed_beef_u64;
            for t in 0..128 {
                let mut x = [0.0_f32; N];
                for (j, v) in x.iter_mut().enumerate() {
                    *v = (((t * 31 + j * 7) % 23) as f32 / 23.0) - 0.5;
                }
                let ev = pencil.eval(&x, 4, &mut scratch);
                h = h.wrapping_mul(31).wrapping_add(ev.value.to_bits() as u64);
            }
            *fold = h;
        }
        assert_eq!(folds[0], folds[1]);
    }

    // ── headroom shapes evaluate (d ∈ {8,16,32}, dense + tridiag) ────
    {
        macro_rules! shape_check {
            ($d:expr, $n:expr) => {{
                let init = seeded_dense::<$d, $n>(b"headroom", $d / 2);
                let pencil = DensePencil::<$d, $n> {
                    a0: init.a0,
                    a: init.a,
                };
                let mut scratch = DenseScratch::<$d>::new();
                let x = [0.5_f32; $n];
                let ev = pencil.eval(&x, $d / 2, &mut scratch);
                assert!(ev.value.is_finite());
                assert!(ev.jacobi.converged, "d={} Jacobi hit the sweep cap", $d);
            }};
        }
        shape_check!(8, 8);
        shape_check!(16, 16);
        shape_check!(32, 8);
        let tri = seeded_tridiag::<32, 8>(b"headroom-tri", 16);
        let tp = TridiagPencil::<32, 8> {
            a0: tri.a0,
            a: tri.a,
        };
        let mut ts = TriScratch::<32>::new();
        let x = [0.5_f32; 8];
        assert!(tp.eval(&x, 16, &mut ts).is_finite());
    }
}

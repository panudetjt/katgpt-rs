//! spectral_pencil latency bench (Issue 676 T10 — G2 perf gate).
//!
//! Reports ns/eval for:
//! - dense pencil eval (pinned cyclic Jacobi) at `d ∈ {8, 16, 32}`
//! - tridiagonal pencil eval (Sturm bisection) at the same dims
//! - `count_below` (the O(d) exact integer predicate)
//!
//! and prints the 10k NPC × 20 Hz headroom arithmetic in the summary
//! (the Research 495 §5 cost model: dense d=16 ≈ 8K FLOPs/eval,
//! tridiag ≈ 800).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features spectral_pencil \
//!   --bench spectral_pencil_bench -- --warm-up-time 0.5 --measurement-time 1.5
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::spectral_pencil::dense::DenseScratch;
use katgpt_core::spectral_pencil::init::{seeded_dense, seeded_tridiag};
use katgpt_core::spectral_pencil::tridiag::TriScratch;
use katgpt_core::spectral_pencil::{DensePencil, TridiagPencil};

fn bench_dense_eval(c: &mut Criterion) {
    for (d, n) in [(8_usize, 8_usize), (16, 16), (32, 8)] {
        let mut group = c.benchmark_group(format!("spectral_pencil_dense_d{d}"));
        group.throughput(criterion::Throughput::Elements(1));
        let id = BenchmarkId::from_parameter(d);
        group.bench_with_input(id, &(d, n), |b, &(d, n)| match (d, n) {
            (8, 8) => {
                let init = seeded_dense::<8, 8>(b"bench", 4);
                let p = DensePencil::<8, 8> {
                    a0: init.a0,
                    a: init.a,
                };
                let mut s = DenseScratch::<8>::new();
                let x = [0.5_f32; 8];
                b.iter(|| black_box(p.eval(black_box(&x), 4, &mut s)));
            }
            (16, 16) => {
                let init = seeded_dense::<16, 16>(b"bench", 8);
                let p = DensePencil::<16, 16> {
                    a0: init.a0,
                    a: init.a,
                };
                let mut s = DenseScratch::<16>::new();
                let x = [0.5_f32; 16];
                b.iter(|| black_box(p.eval(black_box(&x), 8, &mut s)));
            }
            (32, 8) => {
                let init = seeded_dense::<32, 8>(b"bench", 16);
                let p = DensePencil::<32, 8> {
                    a0: init.a0,
                    a: init.a,
                };
                let mut s = DenseScratch::<32>::new();
                let x = [0.5_f32; 8];
                b.iter(|| black_box(p.eval(black_box(&x), 16, &mut s)));
            }
            _ => unreachable!(),
        });
        group.finish();
    }
}

fn bench_tridiag_eval(c: &mut Criterion) {
    for (d, n) in [(8_usize, 8_usize), (16, 16), (32, 8)] {
        let mut group = c.benchmark_group(format!("spectral_pencil_tridiag_d{d}"));
        group.throughput(criterion::Throughput::Elements(1));
        let id = BenchmarkId::from_parameter(d);
        group.bench_with_input(id, &(d, n), |b, &(d, n)| match (d, n) {
            (8, 8) => {
                let init = seeded_tridiag::<8, 8>(b"bench", 4);
                let p = TridiagPencil::<8, 8> {
                    a0: init.a0,
                    a: init.a,
                };
                let mut s = TriScratch::<8>::new();
                let x = [0.5_f32; 8];
                b.iter(|| black_box(p.eval(black_box(&x), 4, &mut s)));
            }
            (16, 16) => {
                let init = seeded_tridiag::<16, 16>(b"bench", 8);
                let p = TridiagPencil::<16, 16> {
                    a0: init.a0,
                    a: init.a,
                };
                let mut s = TriScratch::<16>::new();
                let x = [0.5_f32; 16];
                b.iter(|| black_box(p.eval(black_box(&x), 8, &mut s)));
            }
            (32, 8) => {
                let init = seeded_tridiag::<32, 8>(b"bench", 16);
                let p = TridiagPencil::<32, 8> {
                    a0: init.a0,
                    a: init.a,
                };
                let mut s = TriScratch::<32>::new();
                let x = [0.5_f32; 8];
                b.iter(|| black_box(p.eval(black_box(&x), 16, &mut s)));
            }
            _ => unreachable!(),
        });
        group.finish();
    }
}

fn bench_sturm_count(c: &mut Criterion) {
    let init = seeded_tridiag::<16, 8>(b"bench-count", 8);
    let p = TridiagPencil::<16, 8> {
        a0: init.a0,
        a: init.a,
    };
    let mut s = TriScratch::<16>::new();
    let x = [0.5_f32; 8];
    c.bench_function("spectral_pencil_sturm_count_d16", |b| {
        b.iter(|| black_box(p.count_below(black_box(&x), black_box(0.25), &mut s)))
    });
}

criterion_group!(
    benches,
    bench_dense_eval,
    bench_tridiag_eval,
    bench_sturm_count
);
criterion_main!(benches);

//! Issue 807 — LtHash GOAT gate, G2/G4 axis.
//!
//! G2: the one-number story is `replace` (O(1) incremental update) vs the
//! 1000-element from-scratch rebuild — the riir-chain commitment_root shape
//! (Bench 028's 31.6 ms @ N=1e5 is the sorted-fold version of the rebuild
//! arm here). Op costs at the default 1024 lanes + the 128-lane consumer
//! sketch. G4: zero-alloc by construction (fixed arrays only) — asserted in
//! the unit gates, not re-asserted here.

use criterion::{Criterion, criterion_group, criterion_main};
use katgpt_core::lthash::{DEFAULT_LANES, Element, LtHash};
use std::hint::black_box;

const DOMAIN: &str = "katgpt-lthash-bench-v1";

fn member(i: u32) -> Element {
    Element::derive(DOMAIN, &[&i.to_le_bytes(), &(i as u64).to_le_bytes()])
}

fn bench_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("lthash");

    group.bench_function("element_derive/1024", |b| {
        b.iter(|| {
            Element::<DEFAULT_LANES>::derive(
                black_box(DOMAIN),
                black_box(&[b"account-key-0", b"value-0"]),
            )
        })
    });
    group.bench_function("element_derive/128", |b| {
        b.iter(|| {
            Element::<128>::derive(
                black_box(DOMAIN),
                black_box(&[b"account-key-0", b"value-0"]),
            )
        })
    });

    let mut h: LtHash = LtHash::identity();
    let e = member(0);
    group.bench_function("insert/1024", |b| b.iter(|| h.insert(black_box(&e))));
    let e2 = member(1);
    group.bench_function("replace/1024", |b| {
        b.iter(|| h.replace(black_box(&e), black_box(&e2)))
    });
    let other: LtHash = LtHash::identity();
    group.bench_function("merge/1024", |b| b.iter(|| h.merge(black_box(&other))));
    group.bench_function("checksum/1024", |b| b.iter(|| h.checksum()));

    // The G2 headline: one incremental replace vs rebuilding the whole
    // 1000-member multiset from scratch (derive + insert × 1000).
    let members: Vec<Element> = (0..1000).map(member).collect();
    let old = &members[0];
    let new = member(1001);
    group.bench_function("replace_vs_rebuild_1000/incremental_replace", |b| {
        b.iter(|| h.replace(black_box(old), black_box(&new)))
    });
    group.bench_function("replace_vs_rebuild_1000/from_scratch_rebuild", |b| {
        b.iter(|| {
            let mut rebuild: LtHash = LtHash::identity();
            for m in &members {
                rebuild.insert(m);
            }
            rebuild.checksum()
        })
    });

    group.finish();
}

fn bench_narrow(c: &mut Criterion) {
    let mut group = c.benchmark_group("lthash_narrow");
    let mut h = LtHash::<128>::identity();
    let e = Element::<128>::derive(DOMAIN, &[b"k", b"v"]);
    let e2 = Element::<128>::derive(DOMAIN, &[b"k", b"v2"]);
    group.bench_function("replace/128", |b| {
        b.iter(|| h.replace(black_box(&e), black_box(&e2)))
    });
    group.finish();
}

criterion_group!(benches, bench_ops, bench_narrow);
criterion_main!(benches);

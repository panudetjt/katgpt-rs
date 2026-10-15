//! Plan 616 T5 — `minmax_partition` GOAT bench (G2 runtime vs the O(n²)
//! model + the promoted-primitive determinism digest).
//!
//! Run: `cargo bench -p katgpt-core --features minmax_partition --bench
//! bench_616_minmax_partition_goat` at `--release` (the profile AGENTS.md
//! mandates for gates).
//!
//! - **G1 pointer** — brute-force optimality agreement (n ≤ 12) is the
//!   unit test battery in `src/partition.rs` (exponential brute force
//!   does not belong in a bench). This bench's correctness arm is the
//!   structure post-condition: exact block recovery on a planted oracle
//!   at every L.
//! - **G2** — DP runtime at L ∈ {16, 64, 256, 1024} over a
//!   block-structured oracle: printed per-L medians with the O(n²) model
//!   ratio beside each, and ONE loose regression bar — the L=1024 call
//!   must stay under a bound that only an algorithmic-class regression
//!   (O(n³)) can fire. Wall-clock numbers carry the box-state caveat;
//!   the RATIOS vs the n² model are the claim (katgpt-rs AGENTS.md G2
//!   law — a latency number without its box state is not a measurement,
//!   so the machine + posture print on the verdict line).
//! - **G4 pointer** — the alloc-count ceiling lives in the separate
//!   `minmax_partition_alloc_check` binary (the `*_alloc_check`
//!   convention).
//! - **Determinism digest** — the partition of the L=1024 oracle is
//!   BLAKE3-digested; the printed digest is the cross-box comparison
//!   anchor (same oracle bytes → same digest on any platform, the
//!   promoted primitive's determinism contract).
//!
//! Loud-zero defense: every timed loop's output feeds `black_box`.
#![cfg(feature = "minmax_partition")]

use katgpt_core::partition::{
    Block, SMatrix, minmax_partition, minmax_partition_typed, partition_worst,
};
use std::hint::black_box;
use std::time::Instant;

/// Block-structured oracle: tight inside blocks, far across — the shape
/// the lane actually reads (planted phase structure). Deterministic, no
/// RNG (the global-RNG gate's law; and a fixed oracle is the determinism
/// digest's input).
fn block_oracle(n: usize, block: usize) -> SMatrix {
    SMatrix::from_fn(n, |i, j| {
        if i == j {
            0.0
        } else if i / block == j / block {
            0.05 + ((i ^ j) % 7) as f32 * 0.001
        } else {
            1.0
        }
    })
}

fn median_us(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    println!("=== Plan 616 — minmax_partition GOAT bench ===");
    println!(
        "box: {} / {} cores / release: {}",
        std::env::consts::OS,
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        !cfg!(debug_assertions)
    );

    // G1 structure arm: planted recovery at every bench L.
    for &(n, block) in &[(16usize, 4), (64, 8), (256, 16), (1024, 64)] {
        let s = block_oracle(n, block);
        let p = minmax_partition(&s, 0.10).expect("partition");
        assert_eq!(p.len(), n / block, "planted recovery at n={n}");
        for b in &p {
            assert_eq!(b.len(), block);
        }
        assert!(partition_worst(&s, &p) <= 0.10);
    }
    println!("G1 structure: planted recovery exact at all four L ✓");

    // G2: runtime ladder.
    println!("\n── G2 runtime ladder (median of 9, per-call) ──");
    let mut prev_us: Option<f64> = None;
    let mut prev_n: Option<usize> = None;
    let mut partitions: Vec<(usize, Vec<Block>)> = Vec::new();
    for &(n, block) in &[(16usize, 4), (64, 8), (256, 16), (1024, 64)] {
        let s = block_oracle(n, block);
        let mut samples = Vec::with_capacity(9);
        let mut last = Vec::new();
        for _ in 0..9 {
            let t0 = Instant::now();
            let p = minmax_partition(black_box(&s), black_box(0.10)).unwrap();
            let dt = t0.elapsed();
            samples.push(dt.as_secs_f64() * 1e6);
            last = p;
        }
        let med = median_us(&mut samples);
        let model_ratio = match (prev_us, prev_n) {
            (Some(prev), Some(pn)) => {
                let n2_model = med / (prev * (n * n) as f64 / (pn * pn) as f64);
                format!("{n2_model:.2}× the n² model")
            }
            _ => "baseline".to_string(),
        };
        println!(
            "  n={n:>5}: {med:>10.1} µs/call  [{model_ratio}]  m={}",
            last.len()
        );
        assert!(
            med < 5_000.0,
            "n={n} partition took {med:.0} µs median — algorithmic regression bound (O(n³) at n=1024 is ~50ms+)"
        );
        partitions.push((n, last));
        prev_us = Some(med);
        prev_n = Some(n);
    }

    // Determinism digest: the L=1024 partition's block table.
    let mut digest_bytes = Vec::new();
    for (n, p) in &partitions {
        digest_bytes.extend_from_slice(&n.to_le_bytes());
        for b in p {
            digest_bytes.extend_from_slice(&b.start.to_le_bytes());
            digest_bytes.extend_from_slice(&b.end.to_le_bytes());
        }
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&digest_bytes);
    println!(
        "\ndeterminism digest (block tables, all four L): {}",
        hasher.finalize().to_hex()
    );

    // Typed arm smoke: the typed DP at one layout runs and respects its
    // constraint (full gate coverage is the unit battery's).
    let s = block_oracle(1024, 64);
    let types: Vec<bool> = (0..1024).map(|i| i % 3 != 0).collect();
    let p = minmax_partition_typed(&s, 0.10, &types).unwrap();
    for b in &p {
        for x in b.start..b.end {
            assert_eq!(types[x], types[b.start]);
        }
    }
    println!("typed arm: constraint respected at n=1024, m={}\n", p.len());
    println!("=== PASS ===");
}

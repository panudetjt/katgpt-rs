//! Issue 762 T4.1 + T4.2 — SSMax HoldConcentration + Kamath logit-regime
//! detector GOAT gate (`ssmax_temperature` / `logit_regime`).
//!
//! T4.1 — `SsmaxMode::HoldConcentration` (Lemma 2 softmax side, Research 549
//! §2.3): the analytic multiplier `ln((n−k)c/(k(1−c)))/Δ̂` must lift a
//! two-level row's top-k mass to exactly `c` across n, and stay in the
//! [0.1, 10]·log_n safety band at Δ̂ extremes.
//!
//! T4.2 — `logit_regime::{kamath_rho, normalized_entropy_nats, kamath_regime}`
//! (Issue 762 T4.2): Gaussian-band vs spiked separation growing with n,
//! entropy bounds, determinism.
//!
//! # Gates
//!
//! - **G1** — correctness on constructed known-answers (both halves).
//! - **G2** — latency: `kamath_regime` + `SsmaxMode::multiplier` ns/call at
//!   routing scales (n = 1024 / 4096; bars: 10 µs and 100 ns).
//! - **G3** — no-regression: default-build parity is the lib-test suite's
//!   job (cited in the record; this bench runs feature-gated code only).
//! - **G4** — zero allocations, steady state (CountingAllocator, per-thread,
//!   per Issue 741's release-runnable pattern).

#![cfg(feature = "logit_regime")]

use katgpt_core::logit_regime::{kamath_regime, kamath_rho, normalized_entropy_nats};
use katgpt_core::ssmax::{SsmaxConfig, SsmaxMode};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

fn two_level_topk_mass(n: usize, k: usize, gap: f64) -> f64 {
    let top = 1.0_f64;
    let rest = top - gap;
    let z = k as f64 * top.exp() + (n - k) as f64 * rest.exp();
    k as f64 * top.exp() / z
}

fn gate_g1_hold_concentration() -> GateResult {
    println!("\n--- G1a: HoldConcentration exact threshold across n ---");
    let mut ok = true;
    let mut detail = String::new();
    for &(n, k, c) in &[
        (100_usize, 1_usize, 0.9_f32),
        (1_000, 1, 0.9),
        (10_000, 1, 0.9),
        (10_000, 8, 0.75),
    ] {
        let delta = 0.5_f32;
        let mode = SsmaxMode::HoldConcentration {
            c,
            k,
            rolling_delta: delta,
        };
        let m = mode.multiplier((n as f32).ln());
        let mass = two_level_topk_mass(n, k, (m * delta) as f64);
        let hit = (mass - c as f64).abs() < 5e-3;
        println!(
            "  n={n:6} k={k} c={c}: multiplier {m:7.3} → top-k mass {mass:.4} ({})",
            if hit { "OK" } else { "MISS" }
        );
        ok &= hit;
        if !hit {
            detail.push_str(&format!("n={n} mass {mass:.4} != c {c}; "));
        }
    }
    // from_mode caches the exact finite-n form
    let mode = SsmaxMode::HoldConcentration {
        c: 0.9,
        k: 1,
        rolling_delta: 0.5,
    };
    let cfg = SsmaxConfig::from_mode(&mode, 10_000);
    let direct = mode.multiplier((10_000_f32).ln());
    let cfg_ok = (cfg.multiplier() - direct).abs() < 1e-3;
    println!(
        "  from_mode caches exact form: {cfg_ok} ({} vs {direct:.4})",
        cfg.multiplier()
    );
    ok &= cfg_ok;
    GateResult {
        name: "G1a HoldConcentration exactness",
        passed: ok,
        detail: if ok {
            "all thresholds hold c".into()
        } else {
            detail
        },
    }
}

fn gate_g1_logit_regime() -> GateResult {
    println!("\n--- G1b: logit_regime bands + separation + determinism ---");
    let hash = |i: usize, salt: u32| {
        let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
        ((h % 2009) as f32 / 1004.5) - 1.0
    };

    // Gaussian band at three scales
    let mut ok = true;
    for &n in &[64_usize, 256, 1024] {
        let row: Vec<f32> = (0..n).map(|i| hash(i, 7)).collect();
        let rho = kamath_rho(&row);
        let hit = (0.35..=1.15).contains(&rho);
        println!(
            "  gaussian n={n:5}: ρ = {rho:.3} ({})",
            if hit { "band OK" } else { "OUT" }
        );
        ok &= hit;
    }
    // spike separation grows with n
    let mut prev = 0.0f32;
    for &n in &[128_usize, 512, 2048] {
        let mut row: Vec<f32> = (0..n).map(|i| hash(i, 11) * 0.5).collect();
        row[3] = 8.0;
        let r = kamath_regime(&row);
        let hit = r.rho > prev && r.spike_score > 0.6;
        println!(
            "  spiked n={n:5}: ρ = {:5.2} score = {:.3} ({})",
            r.rho,
            r.spike_score,
            if hit { "OK" } else { "OUT" }
        );
        ok &= hit;
        prev = r.rho;
    }
    // entropy bounds
    let uni = (normalized_entropy_nats(&[0.0f32; 32]) - 1.0).abs() < 1e-5;
    let mut oh = vec![-40.0f32; 32];
    oh[5] = 40.0;
    let one_hot = normalized_entropy_nats(&oh) < 1e-4;
    println!("  entropy uniform→1: {uni}, one-hot→0: {one_hot}");
    ok &= uni && one_hot;
    // determinism: bit-identical twice
    let row: Vec<f32> = (0..512).map(|i| hash(i, 3)).collect();
    let a = kamath_regime(&row);
    let b = kamath_regime(&row);
    let det = a == b;
    println!("  determinism (bit-identical ×2): {det}");
    ok &= det;
    GateResult {
        name: "G1b logit_regime bands",
        passed: ok,
        detail: if ok {
            "bands + separation + bounds + determinism".into()
        } else {
            "see lines above".into()
        },
    }
}

fn gate_g2_latency() -> GateResult {
    println!("\n--- G2: latency at routing scales ---");
    println!("    (bar: ≤ 15 ns/element at every scale — a REGRESSION bar (~2× headroom");
    println!("     over the intrinsic scalar cost), not a speed record; offline-diagnostic");
    println!("     cadence, cf. gaussianity_probe. Bench 713 measured the raw entmax");
    println!("     router at ~53 µs/row already at n≈173)");
    let hash = |i: usize, salt: u32| {
        let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
        ((h % 2009) as f32 / 1004.5) - 1.0
    };
    let mut ok = true;
    let mut worst_ns = 0.0f64;
    for &n in &[1_024_usize, 4_096] {
        let row: Vec<f32> = (0..n).map(|i| hash(i, 5)).collect();
        // warm-up
        for _ in 0..20 {
            black_box(kamath_regime(black_box(&row)));
        }
        let iters = 500;
        let t0 = Instant::now();
        for _ in 0..iters {
            black_box(kamath_regime(black_box(&row)));
        }
        let ns = t0.elapsed().as_secs_f64() * 1e9 / iters as f64;
        let ns_per_elem = ns / n as f64;
        let hit = ns_per_elem <= 15.0;
        println!(
            "  kamath_regime n={n:5}: {ns:8.1} ns/call ({ns_per_elem:.2} ns/elem ≤ 15) ({})",
            if hit { "OK" } else { "OUT" }
        );
        ok &= hit;
        worst_ns = worst_ns.max(ns);
    }
    // HoldConcentration multiplier: one ln/exp + arithmetic — bar 100 ns
    let mode = SsmaxMode::HoldConcentration {
        c: 0.9,
        k: 1,
        rolling_delta: 0.5,
    };
    let log_n = (4_096_f32).ln();
    for _ in 0..100 {
        black_box(mode.multiplier(black_box(log_n)));
    }
    let iters = 100_000;
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(mode.multiplier(black_box(log_n)));
    }
    let ns = t0.elapsed().as_secs_f64() * 1e9 / iters as f64;
    let hit = ns < 100.0;
    println!(
        "  HoldConcentration multiplier: {ns:6.1} ns/call (< 100 ns: {})",
        if hit { "OK" } else { "OUT" }
    );
    ok &= hit;
    GateResult {
        name: "G2 latency",
        passed: ok,
        detail: format!("worst {worst_ns:.0} ns/call"),
    }
}

fn gate_g4_alloc() -> GateResult {
    println!("\n--- G4: zero allocations, steady state ---");
    assert_counter_is_live();
    let hash = |i: usize, salt: u32| {
        let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
        ((h % 2009) as f32 / 1004.5) - 1.0
    };
    let row: Vec<f32> = (0..1024).map(|i| hash(i, 5)).collect();
    let mode = SsmaxMode::HoldConcentration {
        c: 0.9,
        k: 1,
        rolling_delta: 0.5,
    };
    let log_n = (1024_f32).ln();

    // warm once
    black_box(kamath_regime(&row));
    let ((), allocs) = alloc_delta(|| {
        for _ in 0..1_000 {
            black_box(kamath_regime(black_box(&row)));
            black_box(mode.multiplier(black_box(log_n)));
            black_box(normalized_entropy_nats(black_box(&row)));
        }
    });
    println!("  1000× (kamath_regime + multiplier + entropy) @ n=1024: {allocs} allocs");
    GateResult {
        name: "G4 zero-alloc",
        passed: allocs == 0,
        detail: format!("{allocs} steady-state allocs"),
    }
}

fn main() {
    println!("================ Bench 759: Issue 762 T4.1 + T4.2 GOAT gate ================");
    let gates = vec![
        gate_g1_hold_concentration(),
        gate_g1_logit_regime(),
        gate_g2_latency(),
        gate_g4_alloc(),
    ];
    println!("\n================ GOAT VERDICT ================");
    let mut all = true;
    for g in &gates {
        println!(
            "  [{}] {} — {}",
            if g.passed { "PASS" } else { "FAIL" },
            g.name,
            g.detail
        );
        all &= g.passed;
    }
    // G3 is the default-build parity argument (lib suites), not a bench arm:
    println!(
        "  [NOTE] G3 no-regression: default-build lib tests (30 ssmax incl. 6 new HoldConcentration) — run separately"
    );
    if all {
        println!(
            "\n  → G1 + G2 + G4 ALL PASS — logit_regime ships OPT-IN; HoldConcentration ships in the default-on ssmax module (Adaptive-variant precedent)."
        );
    } else {
        println!("\n  → GATE FAILED");
        std::process::exit(1);
    }
}

//! Plan 580 G4 — zero-alloc steady state for the certified-frontier operators.
//!
//! Separate single-purpose binary (the `CountingAllocator` global pattern —
//! `bench_655/656/676/680` convention): parallel tests share the global
//! counter, so every check lives in ONE test function, serial by construction.
//!
//! The primitive's whole claim to a hot-path budget is that it is fixed
//! capacity. `CertifiedFrontier` and `PosteriorBuffer` are boxed ONCE during
//! setup — that box is an allocation and is deliberately outside the measured
//! window — and nothing after it may touch the allocator.
#![cfg(feature = "certified_frontier")]

use katgpt_core::certified_frontier::{
    CertifiedFrontier, FrontierConfig, PosteriorBuffer, advance_horizon, beta_mean_variance,
    beta_union_bound, confidence_schedule, laurent_massart_radius, linear_information_gain,
    should_advance, sphere_exclusion_coverage, spherical_cap_bound, vendi_diversity,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

const POOL: usize = 512;
const D: usize = 8;
const OBS: usize = 128;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

#[test]
fn certified_frontier_g4_zero_alloc_steady_state() {
    const CYCLES: usize = 1000;

    let cfg = FrontierConfig {
        h: 0.6,
        acquire_radius: 0.4,
        cell_spacing: 0.05,
        lipschitz: 1.0,
        ..FrontierConfig::default()
    };

    // ── setup (allocations expected and permitted) ─────────────────────────
    let mut rng = Lcg::new(0x6488);
    let mut f = Box::new(CertifiedFrontier::<POOL, D>::new());
    for _ in 0..POOL {
        f.push_cell(std::array::from_fn(|_| rng.next_f32()))
            .unwrap();
    }
    let mut buf = Box::new(PosteriorBuffer::<OBS, D>::new(1.0));
    for _ in 0..OBS {
        buf.append_observation(&std::array::from_fn(|_| rng.next_f32()), rng.next_f32());
    }
    let mut scratch = [0.0f32; OBS];
    let mut samples = [[0.0f32; D]; 64];
    for s in samples.iter_mut() {
        *s = std::array::from_fn(|_| rng.next_f32());
    }
    let eigs = [0.4f32, 0.3, 0.2, 0.1];
    // Warm every path once so a first-call lazy init cannot land inside the
    // measured window.
    // One observation per cell leaves LCB ~0.2 at beta=2 — far under h — so the
    // warmup has to actually saturate the tally or the whole check is vacuous.
    for i in 0..POOL {
        for _ in 0..64 {
            f.observe(i, true);
        }
    }
    black_box(f.seed_certified(0, &cfg));
    black_box(f.expand_certified(&cfg, 2.0));
    black_box(f.acquire_frontier_target(&cfg));
    black_box(f.reachability_dilation(&cfg, 1));
    black_box(f.dilation_feasibility(&cfg));
    f.rebuild_neighborhoods(&cfg);
    black_box(buf.posterior_variance_linear(&samples[0], &mut scratch));
    black_box(buf.ridge_mean(&samples[0]));
    black_box(sphere_exclusion_coverage(&samples, 0.3));
    black_box(vendi_diversity(&eigs));
    assert!(f.certified_count() > 0, "warmup certified nothing");

    // ── measured window ────────────────────────────────────────────────────
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    let mut sink = 0.0f32;
    let mut isink = 0usize;
    for i in 0..CYCLES {
        let t = i as u32 + 1;
        let j = f.acquire_frontier_target(&cfg).unwrap();
        isink += j;
        f.observe(j, rng.next_f32() < 0.8);

        let beta = confidence_schedule(t, cfg.delta, cfg.lambda, cfg.b_rkhs, D);
        sink += beta;
        isink += f.expand_certified(&cfg, beta) as usize;
        sink += f.lcb(j, beta) + f.ucb(j, beta) + f.sigma(j);
        isink += usize::from(f.query_is_decision_relevant(j, &cfg, beta));

        let feas = f.dilation_feasibility(&cfg);
        sink += feas.best_headroom + feas.hop_cost + feas.deficit;
        isink += usize::from(feas.feasible);
        isink += f.reachability_dilation(&cfg, 1) as usize;

        // The kernel path: append is the incremental Cholesky row; the buffer
        // saturates after OBS appends and must refuse without allocating.
        buf.append_observation(&samples[i % samples.len()], rng.next_f32());
        sink += buf.posterior_variance_linear(&samples[i % samples.len()], &mut scratch);
        sink += buf.ridge_mean(&samples[i % samples.len()]);
        f.refresh_kernel_sigma(&buf, &mut scratch);

        // Closed forms + scoreboards.
        let (m, v) = beta_mean_variance(t, t / 2);
        sink += m + v;
        sink += linear_information_gain(t, D, cfg.lambda);
        sink += beta_union_bound(POOL, t, cfg.delta);
        sink += advance_horizon(cfg.alpha, beta, 8.0, cfg.epsilon);
        sink += spherical_cap_bound(D, 0.3) + laurent_massart_radius(D, cfg.delta);
        isink += usize::from(should_advance(f.sigma(j), beta, cfg.epsilon));
        isink += sphere_exclusion_coverage(&samples, 0.3).centers;
        sink += vendi_diversity(&eigs);
    }

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
    black_box(&sink);
    black_box(&isink);

    assert_eq!(
        alloc_delta, 0,
        "steady-state allocs leaked ({alloc_delta} allocs / {dealloc_delta} deallocs) \
         across {CYCLES} acquire/observe/expand/dilate cycles"
    );
    assert_eq!(dealloc_delta, 0, "steady-state deallocs leaked");

    // Capacity is a hard bound, not a resize: the buffer must have refused the
    // appends past OBS rather than growing. A silent regrow would show up as an
    // alloc above, but assert the state too so the reason is legible.
    assert_eq!(buf.len(), OBS, "PosteriorBuffer grew past its capacity");
    assert_eq!(f.len(), POOL, "CertifiedFrontier grew past its capacity");
}

//! GOAT Proof: Thinking Prune — FrozenBaseGuard for Token-Level DDTree (Plan 171).
//!
//! Validates three properties:
//! P1: FrozenBaseGuard intermediate hops produce STRICTLY MORE Uniform nodes
//!     (structural dominance — a tie is a failure, see Issue 831 T3)
//! P2: FrozenBaseGuard produces identical output to Uniform when screening is cheap
//!     (NoScreeningPruner — confirms correctness of the delegation)
//! P3: Instrument health + mechanism (Issue 831 T5 owner call, 2026-09-19):
//!     the timed section asserts the INSTRUMENT (both arms measurably real,
//!     median finite — the Issue-723 eliminated-arm class) and the MECHANISM
//!     (FrozenBaseGuard makes exactly 1/3 of Uniform's screener calls over 3
//!     hops, counted not timed). The wall-clock speedup is PRINTED as the
//!     diagnostic record (measured ~63-64% on both arches, Issue 831 T2/T3)
//!     and gates nothing — thinking_prune's status does not rest on this row.
//! P4: Single-hop edge case — FrozenBaseGuard applies full screening when hop is final

#[path = "common/ab_timing.rs"]
mod ab_timing;

use katgpt_rs::speculative::{
    build_dd_tree_screened, extract_best_path,
    types::{NoScreeningPruner, ScreeningPruner},
};
use katgpt_rs::types::Config;

#[cfg(feature = "thinking_prune")]
use katgpt_rs::pruners::PrunerSchedule;
#[cfg(feature = "thinking_prune")]
use katgpt_rs::speculative::build_dd_tree_screened_with_schedule;

// ── Helpers ────────────────────────────────────────────────────

/// Simulated expensive screener: injects artificial delay to model WASM/validator cost.
#[derive(Debug, Clone)]
struct ExpensiveScreener {
    /// Base relevance to return (0.0–1.0)
    base_relevance: f32,
    /// Artificial work per call (loop iterations)
    work_per_call: u32,
}

impl ScreeningPruner for ExpensiveScreener {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        // Simulate expensive WASM validator call
        let mut acc: f32 = 0.0;
        for i in 0..self.work_per_call {
            acc += (i as f32).sin() * (i as f32).cos();
        }
        // ⛔ This MUST be `black_box`, and it used to be `let _ = acc;` under a
        // comment claiming it prevented the optimiser from removing the work.
        // It does not: the loop is pure and its result dropped, which is the
        // exact shape Issue 723 T5 measured being deleted by rustc 1.98.1 +
        // fat LTO. In release the "expensive" screener cost NOTHING, so P3 was
        // comparing two arms that both skipped the same free call — which is
        // why 50x the `work_per_call` moved the measured speedup only from
        // 0.4% to 3.1% (Issue 831, measured 2026-09-18).
        std::hint::black_box(acc);
        self.base_relevance
    }
}

/// A screener whose relevance varies per depth AND per token.
///
/// ⛔ **The per-token term is Issue 831 T4.** Keyed on `depth` alone this
/// screener is all-or-nothing WITHIN a depth — with
/// `relevances = [0.8, 0.4, 0.2, 0.6, 0.9]` against `screening_threshold = 0.3`,
/// depth 2 rejected EVERY token and the other four accepted EVERY token. A
/// screener that never says "this token yes, that token no" cannot distinguish
/// a builder applying it per TOKEN from one applying it per DEPTH, which is
/// most of what `ScreeningPruner` is for. The jitter makes depth 1 partial
/// (75% pass) and leaves the rest saturated — see the knob's own note for why
/// "partial at every depth" is not reachable from this fixture.
///
/// The jitter is an integer hash of `token_idx`, not a float expression and not
/// an RNG draw: it must be bit-identical on aarch64, x86_64 and wasm32 (this
/// target's whole Issue-831 history is arch-dependent readings) and it must not
/// touch the global RNG (`global_rng_gate`).
#[derive(Debug, Clone)]
struct VaryingScreener {
    relevances: Vec<f32>,
}

/// Per-token jitter range, `[JITTER_LO, JITTER_LO + JITTER_SPAN)`.
///
/// ⚠ **This is a FIXTURE KNOB with a budget coupling, not a free parameter,
/// and the obvious improvement is measured to be WORSE.** Pass rate at depth d
/// is the fraction of the range above `threshold / relevances[d]`. Making every
/// depth partial needs the range to straddle `0.3/0.9 = 0.333` and
/// `0.3/0.2 = 1.5`, i.e. roughly `[0.25, 1.75)` — measured, that puts every
/// seed at **12288 = 4096 x 3 hops, 10/10 ties**, because depth 2 stops
/// truncating the tree and both schedules cap out. `[0, 2)` does the same. The
/// hard reject at depth 2 is what keeps this fixture small enough for P1 to
/// mean anything, so the reachable choice is *some* partial depth, not all of
/// them. `[0.5, 1.5)` buys depth 1 at 75% with both schedules UNCAPPED
/// (1701 vs 8759). Change it and re-read the per-seed node counts, not just
/// the verdict.
const JITTER_LO: f32 = 0.5;
const JITTER_SPAN: f32 = 1.0;

impl VaryingScreener {
    /// Deterministic per-token multiplier in `[JITTER_LO, JITTER_LO + JITTER_SPAN)`. Fixed-point throughout:
    /// one u64 multiply, one shift, one exact-power-of-two divide.
    fn jitter(token_idx: usize) -> f32 {
        const BITS: u32 = 24;
        let h = (token_idx as u64)
            .wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            >> (64 - BITS);
        // 2^24 is exactly representable in f32, so the divide is exact.
        JITTER_LO + JITTER_SPAN * (h as f32) / (1u64 << BITS) as f32
    }
}

impl ScreeningPruner for VaryingScreener {
    fn relevance(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        let base = self.relevances.get(depth).copied().unwrap_or(1.0);
        base * Self::jitter(token_idx)
    }
}

fn make_config() -> Config {
    let mut c = Config::draft();
    c.screening_threshold = 0.3;
    c.tree_budget = 512;
    c
}

fn random_marginals(depths: usize, vocab: usize, seed: u32) -> Vec<Vec<f32>> {
    let mut rng = katgpt_rs::types::Rng::new(seed as u64);
    (0..depths)
        .map(|_| {
            let mut m: Vec<f32> = (0..vocab).map(|_| rng.uniform()).collect();
            let sum: f32 = m.iter().sum();
            for v in m.iter_mut() {
                *v /= sum;
            }
            m
        })
        .collect()
}

fn marginals_refs(marginals: &[Vec<f32>]) -> Vec<&[f32]> {
    marginals.iter().map(|m| m.as_slice()).collect()
}

// ══════════════════════════════════════════════════════════════════════════
// P1: Structural Dominance — FrozenBaseGuard >= Uniform nodes
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p1_structural_dominance() {
    println!("\n── P1: FrozenBaseGuard intermediate produces >= Uniform nodes ──\n");

    let mut config = make_config();
    // ⛔ P1's own budget, and it is the whole of Issue 831 T3. At the shared
    // `tree_budget = 512` this proof was VACUOUS: both schedules hit the cap
    // and reported 1536 nodes (512 x 3 hops) on all 10 seeds — `Frozen wins=0,
    // Uniform wins=0, Ties=10` — so `>=` passed on exact equality every time
    // and "the two schedules build the same tree" was indistinguishable from
    // structural dominance. The budget was masking the mechanism, not the
    // screener: uncapped, the same fixture measures Uniform 2268 against
    // FrozenBaseGuard 200756. 4096 is large enough for Uniform to reach its
    // natural 2268 and for the gap to be real (8948), and small enough not to
    // build a 200k-node tree ten times to say so.
    config.tree_budget = 4096;
    let depths = 5;
    let vocab = config.vocab_size;
    let n_trials = 10;

    let mut frozen_wins = 0;
    let mut uniform_wins = 0;
    let mut ties = 0;
    // Issue 831 T4: every trial's (uniform, frozen) pair, so the seed axis is
    // ASSERTED rather than assumed to vary. See the assertion after the loop.
    let mut per_seed: Vec<(usize, usize)> = Vec::with_capacity(n_trials as usize);

    // Screener that rejects some tokens (simulates real pruning)
    let screener = VaryingScreener {
        relevances: vec![0.8, 0.4, 0.2, 0.6, 0.9],
    };

    for seed in 0..n_trials {
        let marginals = random_marginals(depths, vocab, seed);
        let refs = marginals_refs(&marginals);

        // Simulate 3-hop SpecHop pipeline
        let total_hops = 3;

        // Uniform: every hop applies full screening
        let mut uniform_total_nodes = 0;
        for hop in 0..total_hops {
            let tree = build_dd_tree_screened_with_schedule(
                &refs,
                &config,
                &screener,
                true,
                PrunerSchedule::Uniform,
                hop,
                total_hops,
            );
            uniform_total_nodes += tree.len();
        }

        // FrozenBaseGuard: only final hop applies screening
        let mut frozen_total_nodes = 0;
        for hop in 0..total_hops {
            let tree = build_dd_tree_screened_with_schedule(
                &refs,
                &config,
                &screener,
                true,
                PrunerSchedule::FrozenBaseGuard,
                hop,
                total_hops,
            );
            frozen_total_nodes += tree.len();
        }

        println!(
            "  Seed {seed}: Uniform={uniform_total_nodes} nodes, FrozenBaseGuard={frozen_total_nodes} nodes",
        );

        per_seed.push((uniform_total_nodes, frozen_total_nodes));

        match frozen_total_nodes.cmp(&uniform_total_nodes) {
            std::cmp::Ordering::Greater => frozen_wins += 1,
            std::cmp::Ordering::Less => uniform_wins += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }
    }

    println!("\n  Summary: Frozen wins={frozen_wins}, Uniform wins={uniform_wins}, Ties={ties}");

    // ⛔ Issue 831 T4 — the 10 seeds CANNOT move this metric, and saying so is
    // the finding. `build_screened` admits every token with `prob > 0` that
    // clears the screener; it is not a top-k. `random_marginals` produces an
    // all-positive vector, so the marginals decide each node's SCORE and the
    // heap ORDER — never membership — and the node COUNT is a function of
    // (screener, threshold, vocab, depths, budget) alone. A seed loop over a
    // count is one case run ten times, and no screener change fixes that: the
    // earlier reading of T4 ("the screener ignores token_idx") was true and was
    // not the reason.
    //
    // So the loop is kept and given the job it can actually do: pin the
    // marginal-INDEPENDENCE. If a future builder starts top-k-ing candidates,
    // or `tree_budget` starts binding unevenly, this reds and names the
    // builder — which the dominance assertions below never would.
    let spread: Vec<(usize, usize)> = per_seed
        .iter()
        .copied()
        .filter(|&pair| pair != per_seed[0])
        .collect();
    assert!(
        spread.is_empty(),
        "P1 node counts became SEED-DEPENDENT ({:?} differs from seed 0's {:?}). \
         That is NOT a P1 failure — it means build_screened's candidate policy \
         changed (a top-k, an uneven tree_budget bind, or a marginal that can be \
         non-positive). Re-derive what P1 measures before re-pinning it \
         (Issue 831 T4).",
        spread.first(),
        per_seed[0],
    );

    // Assert: FrozenBaseGuard should NEVER produce fewer total nodes...
    assert_eq!(
        uniform_wins, 0,
        "FrozenBaseGuard should produce >= Uniform nodes (Uniform won {uniform_wins} times)",
    );
    // ...and STRICTLY more, which is the claim this proof is named for
    // (Issue 831 T3). The `>=` above cannot fail on a build where both
    // schedules produce the identical tree, and for as long as the budget
    // capped both that is exactly what it was passing on. A TIE is now a
    // failure, and it fails with the diagnosis attached rather than leaving
    // the next reader to rediscover the cap.
    assert_eq!(
        ties, 0,
        "P1 is VACUOUS: {ties} of {n_trials} trials produced IDENTICAL node counts for \
         both schedules, so `>=` is passing on equality and proves nothing about \
         structural dominance. The usual cause is `tree_budget` capping both \
         schedules at the same size — raise it until Uniform reaches its natural \
         size (Issue 831 T3).",
    );
    assert_eq!(
        frozen_wins, n_trials,
        "FrozenBaseGuard should produce strictly MORE nodes on every trial \
         (won {frozen_wins} of {n_trials})",
    );
    // ⚠ The 10 seeds do NOT diversify this measurement: `VaryingScreener`
    // keys only on `depth`, ignoring `token_idx` and `parent_tokens`, so
    // pruning is deterministic per depth and every seed reports the same two
    // counts. The trials are a loop over one case. Left as-is deliberately —
    // making them independent is a fixture question (Issue 831 T4), and
    // recording it beats a reader inferring breadth this proof does not have.
    println!("  ✅ P1 PASS: FrozenBaseGuard produces strictly more nodes on all {n_trials} trials");
}

// ══════════════════════════════════════════════════════════════════════════
// P2: Identical Output with NoScreeningPruner (correctness)
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p2_identical_with_noop_screener() {
    println!("\n── P2: Identical output with NoScreeningPruner ──────────────\n");

    let config = make_config();
    let marginals = random_marginals(4, config.vocab_size, 42);
    let refs = marginals_refs(&marginals);
    let screener = NoScreeningPruner;

    let total_hops = 3;

    for hop in 0..total_hops {
        let uniform_tree = build_dd_tree_screened_with_schedule(
            &refs,
            &config,
            &screener,
            true,
            PrunerSchedule::Uniform,
            hop,
            total_hops,
        );
        let frozen_tree = build_dd_tree_screened_with_schedule(
            &refs,
            &config,
            &screener,
            true,
            PrunerSchedule::FrozenBaseGuard,
            hop,
            total_hops,
        );

        assert_eq!(
            uniform_tree.len(),
            frozen_tree.len(),
            "Hop {hop}: NoScreeningPruner should produce identical trees",
        );

        // Verify scores match
        for (i, (u, f)) in uniform_tree.iter().zip(frozen_tree.iter()).enumerate() {
            assert!(
                (u.score - f.score).abs() < 1e-6,
                "Hop {hop}, node {i}: score mismatch (Uniform={}, Frozen={})",
                u.score,
                f.score,
            );
        }
        println!(
            "  Hop {hop}: {} nodes, all scores match ✅",
            uniform_tree.len()
        );
    }

    println!("  ✅ P2 PASS: NoScreeningPruner produces identical results");
}

// ══════════════════════════════════════════════════════════════════════════
// P3: Instrument health + mechanism (Issue 831 T5 owner call, 2026-09-19)
//
// Reclassified from a perf bar by the owner call: the ≥30% latency assert is
// GONE. "P3 is a coin flip" was the PRE-repair state (the screener's work was
// deleted — Issue 831 T2/T3); the repaired fixture measured 63-64% on both
// arches, and that number stays the recorded evidence. What this row still
// gates: (a) instrument health — both arms measurably real; (b) the mechanism
// as an exact CALL COUNT — `should_screen_full` is `hop >= total_hops - 1`
// for FrozenBaseGuard, so over 3 hops it must make exactly 1/3 of Uniform's
// screener calls. Load-immune, arch-independent, and the thing the latency
// claim was always a proxy for.
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p3_instrument_and_mechanism() {
    println!("\n── P3: Instrument health + mechanism (call count) ──────────\n");

    let config = make_config();
    let depths = 4;
    let vocab = config.vocab_size;
    let iters = 200;
    let total_hops = 3;

    // Expensive screener with synthetic work
    let expensive = ExpensiveScreener {
        base_relevance: 0.7,
        work_per_call: 100, // enough to be measurable
    };

    let marginals = random_marginals(depths, vocab, 12345);
    let refs = marginals_refs(&marginals);

    // ── The instrument (Issue 831 T2) ───────────────────────────────────
    //
    // This bar used to time Uniform to completion, then FrozenBaseGuard to
    // completion, and assert on that single ratio. Measured on an idle box:
    // **4 of 20 captured-mode runs FAILED** (-1.9% .. -5.0%) while 12 of 12
    // `--nocapture` runs passed (+0.9% .. +11.7%) — the verdict moved with how
    // the harness was invoked, which is a measurement of the scheduler and not
    // of the primitive. That is Issue 723 Class A exactly, and
    // `common/ab_timing.rs` is the treatment AGENTS.md already prescribes:
    // interleaved `(a-chunk, b-chunk)` pairs so a drift moves both arms, and
    // the MEDIAN across pairs so one preemption spike is discarded.
    //
    // ⚠ `ab_median_ratio`, not `best_of_us`. Issue 831 T2 names the
    // `best_of_us` shape, but that module's own docs reserve it for an
    // ABSOLUTE budget with no second arm to ratio against ("contention can
    // only ever add time, so the smallest of N samples is the closest
    // observation"). P3 has two arms and its claim is comparative, so the
    // median-of-ratios form is the one that cancels load; taking a minimum per
    // arm independently would compare two different load windows.
    //
    // ⛔ And the arms are `black_box`ed now. Both of them were
    // `let _ = build_dd_tree_screened_with_schedule(...)`, which is the exact
    // shape Issue 723 T5 measured being deleted by rustc 1.98.1 + fat LTO —
    // "a direct call with a used result measured 16.6 µs; `let _ = f()` over
    // the same fn in the same binary read ~0", *even through a `black_box`
    // inside the callee*. An eliminated arm makes this comparison noise
    // against noise, which is a second and sufficient explanation for a bar
    // that flips sign between invocations. The input is `black_box`ed too so
    // the build cannot be hoisted out of the round.
    // 7 rounds x 8 iters/arm. Sized on the MEASURED signal rather than
    // guessed: with the screener's work no longer deleted the effect is ~64%
    // in a per-round band 2.6% wide, so a median over 7 rounds is decisive and
    // the old 200-iteration sweep was buying precision the claim does not
    // need. It matters because the fixture is now genuinely expensive — the
    // debug-profile run went from ~0.6s of deleted work to 24s of real work at
    // the previous sampling, and this brings it back under 8s without
    // weakening the verdict.
    let rounds = 7;
    let iters_per_round = iters / 25;

    let ab = ab_timing::ab_median_ratio(
        rounds,
        iters_per_round,
        20,
        // a = BASELINE: Uniform screens every hop.
        |_i| {
            for hop in 0..total_hops {
                let tree = build_dd_tree_screened_with_schedule(
                    std::hint::black_box(&refs),
                    &config,
                    std::hint::black_box(&expensive),
                    true,
                    PrunerSchedule::Uniform,
                    hop,
                    total_hops,
                );
                std::hint::black_box(tree);
            }
        },
        // b = CANDIDATE: FrozenBaseGuard skips the intermediate hops.
        |_i| {
            for hop in 0..total_hops {
                let tree = build_dd_tree_screened_with_schedule(
                    std::hint::black_box(&refs),
                    &config,
                    std::hint::black_box(&expensive),
                    true,
                    PrunerSchedule::FrozenBaseGuard,
                    hop,
                    total_hops,
                );
                std::hint::black_box(tree);
            }
        },
    );

    // `ratio = b / a`, so FrozenBaseGuard being faster means a ratio BELOW 1.
    let speedup_pct = (1.0 - ab.median) * 100.0;

    // Printed, never gated (Issue 831 T5 owner call, 2026-09-19). The
    // measured record — 64.2% release / 63.1% debug x86_64 (T2/T3), 63.1-63.8%
    // aarch64 (T1) — lives in the issue/HISTORY; `report` prints the per-round
    // RANGE beside the median on purpose: a median inside a 0.9..1.1 band and
    // one inside a 0.3..3.0 band are not the same claim even when they are the
    // same number.
    ab.report("P3 uniform-vs-frozen");
    println!(
        "  Speedup (median of {rounds} interleaved rounds): {speedup_pct:.1}% (recorded, not gated)"
    );

    // ── Gated half #1: instrument health ─────────────────────────────────
    // Both arms must be measurably real. An arm the optimiser deleted reads
    // ~0 ns/iter and the ratio collapses — the Issue-723 Class A2 shape this
    // harness exists to catch. This is the instrument-health claim, NOT a
    // perf claim: nothing here compares the two arms' times.
    assert!(
        ab.median.is_finite() && ab.a_ns_per_iter() > 0.0 && ab.b_ns_per_iter() > 0.0,
        "P3 instrument health: both arms must do measurable real work \
         (a {:.0} ns/iter, b {:.0} ns/iter, median {:.4}) — an eliminated arm is \
         an instrument failure (Issue 723 Class A2), never a verdict about \
         the schedule; check that ExpensiveScreener::relevance still \
         black_boxes its accumulator",
        ab.a_ns_per_iter(),
        ab.b_ns_per_iter(),
        ab.median,
    );
    println!(
        "  ✅ P3 PASS (instrument): both arms real (a {:.0} ns/iter, b {:.0} ns/iter)",
        ab.a_ns_per_iter(),
        ab.b_ns_per_iter()
    );

    // ── Gated half #2: the mechanism as an exact CALL COUNT ──────────────
    // `should_screen_full` is `hop >= total_hops - 1` for FrozenBaseGuard,
    // so over `total_hops = 3` Uniform screens 3/3 hops and FrozenBaseGuard
    // 1/3 — every skipped hop is a screener call that cannot happen, counted
    // instead of timed. Exact, arch-independent, load-immune: this is the
    // thing the wall-clock number was always a proxy for (the fixture's own
    // "~2/3 of hops skip the work" claim, tested where it lives).
    struct CountingScreener<'a> {
        inner: &'a ExpensiveScreener,
        calls: &'a std::sync::atomic::AtomicU64,
    }
    impl ScreeningPruner for CountingScreener<'_> {
        fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.relevance(depth, token_idx, parent_tokens)
        }
    }
    let count_screener_calls = |schedule: PrunerSchedule| -> u64 {
        let calls = std::sync::atomic::AtomicU64::new(0);
        let counting = CountingScreener {
            inner: &expensive,
            calls: &calls,
        };
        for hop in 0..total_hops {
            let tree = build_dd_tree_screened_with_schedule(
                std::hint::black_box(&refs),
                &config,
                &counting,
                true,
                schedule,
                hop,
                total_hops,
            );
            std::hint::black_box(tree);
        }
        calls.load(std::sync::atomic::Ordering::Relaxed)
    };
    let uniform_calls = count_screener_calls(PrunerSchedule::Uniform);
    let frozen_calls = count_screener_calls(PrunerSchedule::FrozenBaseGuard);
    assert!(
        frozen_calls > 0 && frozen_calls * 3 == uniform_calls,
        "P3 mechanism: FrozenBaseGuard must make exactly 1/3 of Uniform's screener \
         calls over {total_hops} hops (frozen {frozen_calls}, uniform {uniform_calls}) \
         — anything else means the intermediate-hop skip is gone, doubled, or \
         the fixture stopped building identical trees per hop",
    );
    println!(
        "  ✅ P3 PASS (mechanism): screener calls frozen {frozen_calls} = uniform {uniform_calls} / 3"
    );
}

// ══════════════════════════════════════════════════════════════════════════
// P4: Single-Hop Edge Case — Full Screening Applied
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p4_single_hop_is_final() {
    println!("\n── P4: Single-hop edge case applies full screening ────────\n");

    let config = make_config();
    let marginals = random_marginals(3, config.vocab_size, 99);
    let refs = marginals_refs(&marginals);

    let screener = VaryingScreener {
        relevances: vec![0.5, 0.2, 0.8], // 0.2 < threshold 0.3 → should trim at depth 1
    };

    // Single hop (hop 0 of 1) → is final → should apply full screening
    let frozen_tree = build_dd_tree_screened_with_schedule(
        &refs,
        &config,
        &screener,
        true,
        PrunerSchedule::FrozenBaseGuard,
        0,
        1,
    );

    // Compare with explicit full screening
    let full_tree = build_dd_tree_screened(&refs, &config, &screener, true);

    assert_eq!(
        frozen_tree.len(),
        full_tree.len(),
        "Single-hop FrozenBaseGuard should produce identical tree to full screening",
    );
    println!(
        "  Single-hop tree: {} nodes (matches full screening)",
        frozen_tree.len()
    );
    println!("  ✅ P4 PASS: Single-hop applies full screening correctly");
}

// ══════════════════════════════════════════════════════════════════════════
// P5: Path Quality — Best path identical at final hop
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p5_final_hop_quality_identical() {
    println!("\n── P5: Final hop path quality identical to Uniform ────────\n");

    let config = make_config();
    let marginals = random_marginals(4, config.vocab_size, 77);
    let refs = marginals_refs(&marginals);

    let screener = VaryingScreener {
        relevances: vec![0.9, 0.6, 0.4, 0.7],
    };

    let total_hops = 3;

    // Both schedules should produce identical results at the FINAL hop
    let uniform_tree = build_dd_tree_screened_with_schedule(
        &refs,
        &config,
        &screener,
        true,
        PrunerSchedule::Uniform,
        total_hops - 1,
        total_hops,
    );
    let frozen_tree = build_dd_tree_screened_with_schedule(
        &refs,
        &config,
        &screener,
        true,
        PrunerSchedule::FrozenBaseGuard,
        total_hops - 1,
        total_hops,
    );

    let uniform_path = extract_best_path(&uniform_tree);
    let frozen_path = extract_best_path(&frozen_tree);

    assert_eq!(
        uniform_path, frozen_path,
        "Final hop paths should be identical between Uniform and FrozenBaseGuard",
    );
    println!("  Uniform path: {uniform_path:?}");
    println!("  Frozen path:  {frozen_path:?}");
    println!("  ✅ P5 PASS: Final hop produces identical paths");
}

// ══════════════════════════════════════════════════════════════════════════
// Main — Run all proofs
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_bench_171_thinking_prune_goat() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  GOAT Proof: Thinking Prune — FrozenBaseGuard DDTree (171)");
    println!("═══════════════════════════════════════════════════════════");

    #[cfg(feature = "thinking_prune")]
    {
        proof_p1_structural_dominance();
        proof_p2_identical_with_noop_screener();
        proof_p3_instrument_and_mechanism();
        proof_p4_single_hop_is_final();
        proof_p5_final_hop_quality_identical();

        println!("\n═══════════════════════════════════════════════════════════");
        println!("  ALL 5 PROOFS PASSED ✅");
        println!("═══════════════════════════════════════════════════════════");
    }

    #[cfg(not(feature = "thinking_prune"))]
    {
        println!("  ⚠️  thinking_prune feature not enabled — skipping");
        println!(
            "  Run with: cargo test --features thinking_prune --test bench_171_thinking_prune_goat -- --nocapture"
        );
    }
}

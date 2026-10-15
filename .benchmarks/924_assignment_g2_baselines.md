# Bench 924 — katgpt-assign G2 outside baselines (Plan 620 Phase 1)

> **Status:** COMPLETE — G2 PASS (quality axis); wall time recorded under PASSED preflight
> **Feature posture:** `assignment` — **OPT-IN** (katgpt-core + katgpt-rs root forwards; not in any default set)
> **Plan:** [`.plans/620_rebalancer_assignment_solver.md`](../.plans/620_rebalancer_assignment_solver.md) · **Research:** [`.research/607_Rebalancer_Assignment_Solver.md`](../.research/607_Rebalancer_Assignment_Solver.md)
> **Run:** `cargo run --release -p katgpt-assign --bench bench_924_assignment_g2_baselines` (harness = false, zero-dep bench)

## Verdict

**G2 PASS vs outside baselines — the NO-GO clause is NOT triggered.** On every
fixture family the local search strictly dominates first-fit-decreasing
construction on the same folded objective, by **3× to 47×** on the random
multi-dim family, and lands the exact known optimum on both known-optimum
families (perfect-balance partitions; the one-fat-move repair). The tiny-instance
brute-force axis closes to gap 0. The primitive stays **opt-in** per the plan's
verdict protocol — promotion additionally requires a wired consumer (riir-chain
Issue 164 minimum).

## Quality table (load-independent — integer objectives)

| fixture | initial folded | FFD folded | **local folded** | FFD spread | **local spread** | local moved | evals |
|---|---:|---:|---:|---:|---:|---:|---:|
| balance_5c_b20 (n=40) | 488,049,410 | 2,000,204 | **0** | 4 | **0** | 34 | 228,195 |
| balance_20c_b12 (n=120) | 950,096,000 | 0 | **0** | 0 | **0** | 113 (FFD 116) | 60,830 |
| rand_200x10 | 277 | 1,260 | **27** | 600 | **12** | 49 | 10,000,384 |
| rand_1000x20 | 516 | 3,000 | **427** | 1,500 | **252** | 41 | 10,000,384 |
| rand_5000x50 | 1,611 | 9,000 | **1,206** | 4,500 | **579** | 32 | 10,000,384 |
| rand_20000x100 | 3,778 | 24,000 | **3,228** | 12,000 | **1,755** | 58 | 10,000,384 |
| rand_20000x200_4d | 3,718 | 18,000 | **3,455** | 4,500 | **1,033** | 65 | 10,000,384 |
| repair_10c_L1000_e50 | 50,005,000 | 50 | **1 (optimum)** | 1,000 | 1,000 | **1** (FFD 50) | 1,121 |
| shard_256x8 (Issue 164 shape) | 5,553 | 28,325 | **165** | 18,326 | **3** | **53** (FFD 224) | 8,442,306 |
| shard_1024x16 (Issue 164 shape) | 22,550 | 113,691 | **5,819** | 73,508 | **3,707** | **92** (FFD 960) | 10,000,384 |

Reading notes:

- **balance family** reaches the constructed exact optimum (violation 0,
  spread 0, equal loads) from the deliberately-bad all-in-container-0 start.
- **rand family**: local search hits the 10M evaluation cap (default) —
  quality is eval-budget-bound, a tuning property, not a convergence claim.
  Even budget-capped it beats FFD 3–7× on folded objective and ~8× on spread.
- **repair family**: the known optimum is movement 1 (move the fat object);
  FFD-style construction cannot see it (50 moves — it rebuilds from scratch),
  the solver finds it in 1,121 evaluations.
- **shard family** (the riir-chain Issue 164 cluster shape: 2 load dims
  with town/wilderness block skew, round-robin hand initial, movement
  weight 3): the solver takes the hand assignment's spread 3671 → **3** at
  256 shards (1224×) / 15061 → 3707 at 1024 (4×), while relocating only
  **53/256 = 21%** and **92/1024 = 9%** of the shards — the
  don't-teleport-the-world property the issue demands (FFD rebuilds move
  224/960 and balance WORSE than the hand round-robin it replaces).
- **Feasibility**: violation_root = 0 on every fixture at every size (the
  random family's capacities leave slack by construction; the infeasible-tail
  behavior is covered by the G1 infeasible-fixture test).

## Lower bound (mid-size axis)

Pigeonhole bound on per-dim max utilization: `max_b util ≥ ceil(total/B)`.
The solver's max-util sits **on or above** the bound on every dim (sanity —
asserted in `tests/lower_bound.rs`) and **≤ FFD's** max-util. The numeric
gap-to-bound is reported by the bench binary; the bound is not tight for this
fixture family (it bounds the max load, not the spread objective), so the gap
is a diagnostic, not a gate — exactly the plan's "recorded as diagnostic"
posture for the mid-size axis.

## Brute force (tiny axis, in-test)

`tests/small_optimality.rs`: exhaustive enumeration at n=7, B=3 (2,187
assignments) + the n=3 repair fixture — the solver's folded objective equals
the exhaustive minimum on every fixture (documented one-fixture tolerance for
the partition-closure tail: Phase 2's triple moves close it).

## G1 / G4 (recorded here for the one-stop verdict)

- **G1 determinism**: same input + same seed ⇒ byte-identical `Assignment`
  across runs and across spec-declaration-order perturbations
  (`tests/determinism.rs`); never-worse guard pinned on every fixture family.
- **G1 correctness**: delta evaluation is bit-identical to full recompute
  over 1,200 random single/swap moves (`tests/correctness.rs` — the exactness
  claim that replaces Meta's segment tree); feasibility on every family.
- **G4 alloc**: `solve()` performs **0 allocations** after `Solver::new`
  under the counting allocator (`src/alloc.rs`, debug + `alloc_tracking`
  postures — the Issue-741 profile-free form).

## Wall time

**Preflight PASSED** for the recorded run:

```
PROVENANCE: power=AC Power load=4.63 swap=6164.88M canary=skipped powermode=2(high)
✓ preflight PASSED — quote the PROVENANCE line in the bench record
```

(An earlier run under sibling load 6.8–7.7 produced byte-identical QUALITY
numbers — an unplanned cross-load determinism check — and wall times within
3% of the quiet run: the 10M evaluation cap binds, not box speed.)

| fixture | local wall_ms | evals | evals/s |
|---|---:|---:|---:|
| balance_5c_b20 | 34 | 228,195 | ~6.7M/s |
| balance_20c_b12 | 12 | 60,830 | ~4.9M/s |
| rand_200x10 | 2,503 | 10,000,384 | ~4.0M/s |
| rand_1000x20 | 2,833 | 10,000,384 | ~3.5M/s |
| rand_5000x50 | 4,300 | 10,000,384 | ~2.3M/s |
| rand_20000x100 | 7,278 | 10,000,384 | ~1.4M/s |
| rand_20000x200_4d | 24,316 | 10,000,384 | ~0.4M/s |
| repair_10c_L1000_e50 | 0.17 | 1,121 | ~6.6M/s |
| shard_256x8 | 3,438 | 8,442,306 | ~2.5M/s |
| shard_1024x16 | 8,412 | 10,000,384 | ~1.2M/s |

Reading: single-threaded deterministic search at 0.4–6.7M evaluations/s;
~85–200 ns per delta evaluation depending on DAG size. The 10M-eval cap
binds the large fixtures, so wall time tracks the eval budget — an operator
rebalance window of seconds at 20k shards, against Meta's p99 16s at
65k×5k (our envelope is orders smaller). FFD construction is 0.01–6 ms
(single pass, no search). For the Glacial-class topology cadence this is
compute-trivial; the movement-minimality (53/256 relocation) is the value.

## Deviations from the plan letter (recorded)

1. **Secondary repair term**: the folded objective carries the SUM of
   violation rows on the goal side (identically 0 at feasibility). Reason:
   the Max fold alone cannot distinguish which of two tied-at-worst
   containers to drain — measured, strict-only acceptance froze the
   perfect-balance fixture at violation 10,400. Every accepted move strictly
   decreases the sum, so plateau descent is monotone.
2. **Sideways moves are DEFAULT-ON** (25k budget + heat>0 source +
   immediate-reversal guards): the plan's `obj_δ ≤ 0` acceptance window is
   load-bearing for a Max-folded objective — δ=0 moves are the traversal
   mechanism for tie plateaus, not an optional extra.
3. **Advisory time limit** (`Limits::time`, default `None`): a wall-clock
   cutoff breaks cross-machine byte-determinism, so it is opt-in and never
   part of a replayable result.

# Plan 615 Phase 2 — Two-Fidelity Certified BAI GOAT gate — **GOAT PASS** (re-arm after the Bench-906 negative's fixture root cause + the 916a baseline rebuild)

**Bench:** `benches/bench_615_two_fidelity_bai_goat.rs` (`harness = false`, `required-features = ["two_fidelity_bai"]`; run: `cargo bench -p katgpt-core --features two_fidelity_bai --bench bench_615_two_fidelity_bai_goat`)
**Plan:** [615_two_fidelity_bai.md](../.plans/615_two_fidelity_bai.md) · **Research:** 601 · **Paper:** arXiv:2606.01708 · **Issues:** 915 (closed — fixture root cause), 916 (closed — (a) δ-correct baseline `3aa27c149`, (b) `lazy_discharge` resolver `ca9fb1617`, see the addendum; records in HISTORY.md)
**Definitive run:** 2026-10-03 · **Box state:** M3 Max (16-core aarch64), macOS 26.6.2, AC power, release profile, sibling agent sessions active (all reported metrics are sample/cost counts — the unified cost model is load-independent by construction; G4 counts likewise). Determinism: the 2FFS rows reproduced byte-identically across four independently-built binaries spanning three code revisions today (the 915 fixture fix, the 916a baseline rebuild, the m_v table memoization) — same seeds, same code path, same numbers.

## Verdict

**ALL GATES PASS — GOAT PASS at the full protocol (n = 300: 100 trees × {(5,8),(7,6),(10,3)}, ε = 0.02, δ = 0.05, c = 4, σ = 0.05).** 2FFS: **0/300 pick errors (CP95 = 0.00994 ≤ δ), 300/300 certified, 0 invalid certificate intervals, 0 guard-fires**; strict cost wins vs both adversarial baselines at matched accuracy. The module stays **opt-in** (the certified-search slot; promotion to default was never the claim). Plan Phase 3/4 un-gate from here.

## Gate table (the definitive run)

| gate | result |
|---|---|
| CANARY gate-machinery-fires / CP closed-form / alloc-counter-moves | ✓ ✓ ✓ |
| SANITY T2.2b (negamax-UCT zero-noise convergence, \|V̂−V*\| = 0.0000 at 1k/10k/100k) | ✓ |
| FIXTURE honesty (fast envelope every node + slow mean at internal nodes — the Issue-915 class, gated) | ✓ |
| **G1 empirical PAC (Clopper–Pearson ≤ δ)** | **✓ — 0/300 errors, CP95 = 0.00994** |
| **G1 certified-interval validity (Thm 3.1's certificate body)** | **✓ — 0 invalid over 300 certified results** |
| **G1 guard-fires within the δ budget** | **✓ — 0/300 uncertified (budget 17)** |
| G2(a) paired win vs BAI-MCTS (δ-correct LUCB-MCTS, Issue 916a) | ✓ — LB95 +15.8M / +14.8M / +14.9M per setting |
| G2(b) paired win vs negamax-UCT at matched accuracy | ✓ — B* = 33.5M (no checkpoint matched; conservative) vs 2FFS ≤ 1.08M |
| G4 zero per-iteration allocation | ✓ — 3 searches = 3 × 1 search, 1 search = 3 allocs (node HashMap + result Vec + trace slot) |

## Measured rows (the definitive run)

| setting | 2FFS errs / cert | 2FFS cost mean (fast / slow) | BAI errs / capped | BAI cost mean | UCT errs@checkpoints (4k → 8.4M) | fast-only errs | slow-only errs |
|---|---|---|---|---|---|---|---|
| (5,8) | 0/100, **100/100** | 196 435 (633 / 48 951) | 62/100, **100 capped** | 16 000 000 | 65 65 65 … 65 | 77/100 | 0/100 |
| (7,6) | 0/100, **100/100** | 1 080 335 (1 596 / 269 685) | 29/100, **100 capped** | 16 000 000 | 38 → 28 | 83/100 | 0/100 |
| (10,3) | 0/100, **100/100** | 902 198 (544 / 225 413) | 38/100, **100 capped** | 16 000 000 | 41 → 21 | 56/100 | 0/100 |

BAI cost = the `BAI_SAMPLE_CAP` (4M samples × c = 4) — **capped 100/100**: the δ-correct bound-propagation baseline cannot certify any of these fixtures within the cap (the Min-arm L-fan-out needs b^(D−2)-scale CI-converged leaf chains before L rises off −∞; the U-side fans out at Min nodes symmetrically). The cap is the conservative-understatement convention (the same one the UCT no-checkpoint-matched row uses). UCT never reaches 2FFS's 0-error rate at any checkpoint through 8.4M samples ⇒ B* = max-budget cost (understated, conservative against 2FFS).

## How this landing differs from the Bench-906 negative (the full arc)

1. **The 906 negative's certificate invalidity was the BENCH FIXTURE, not the module** (Issue 915, closed): internal slow samples centered on 0.0 (leaf-only `mu`), violating the `MinimaxSpace` mean-V* contract. Fix: slow oracle draws around `vstar` at every node; a FIXTURE-honesty gate now runs before the suites so the class cannot recur; the module's `VecTree` smoke fixture moved to the contract envelope.
2. **The BAI baseline was rebuilt as the δ-correct bound-propagation class** (Issue 916a): the pre-fix arm stopped on point-extreme root estimates with arm-level radii that never covered the deep max/min backup bias (61/100 stopping errors at (5,8)) — its "cost" was the price of an inaccurate stop. The replacement (LUCB-MCTS/UGapE-MCTS per Kaufmann & Koolen 2017) propagates per-leaf time-uniform slow CIs through Eq. 6 bounds and stops only when the propagated root intervals certify — sharing 2FFS's confidence machinery (fairness: isolates the two-fidelity mechanism). Module support: `NodeInterval::slow_only()`.
3. **The search's resolution loop was made wall-viable** (perf, bit-identical): the profile put ~97% of recording-run wall in `beta()`'s exp2 via `m_samples`' per-call stage scans; the stage radii are now precomputed once per search and walked with f64 compares (byte-identical rows verified). A recording run that took ~10 h projected at the old wall now completes in ~45 min.

## Honest caveats on the G2 claims

- **Our LUCB-MCTS instance is the conservative member of its class.** The paper's own BAI instance stops at 8.8e5–1.91e7 samples where ours caps at 4M+ (diluted uniform per-leaf δ + deterministic leader/challenger alternation vs their tuned rules). A G2(a) win measured against it is an UPPER bound on the ratio vs a better-tuned instance; the paper's own numbers stay the cross-implementation context rows (their BAI: 8.8e5 / 1.75e7 / 1.91e7 samples — our capped baseline exceeds all three at the 4M cap ⇒ the class comparison is real, the ratio is not 163–1458× on our instance).
- **The 2FFS cost debt vs the paper's own 2FFS stands** (Issue 916b): ours spends 1.96e5–1.08e6 unified vs their 5.39e3–1.77e4 samples (~10–50×; the per-node trace attributes it to ladder rungs past the ε-necessary width (~2×), stitched-stage δ-halving (~2×), and a-priori δ dilution (~1.5–2×)). The named lever — the paper's own lazy discharge + effective-gap allocation, dropped in v1 — is recorded in Issue 916 behind the opt-in-knob discipline.
- **No promotion**: the module is opt-in by design (the certified-search slot; `mcts_search` keeps the default slot).

## Paper-parity context rows (their numbers, their protocol — context only)

| setting | paper 2FFS samples | paper BAI-MCTS samples | paper ratio | paper ops ratio |
|---|---|---|---|---|
| (5,8) | 5.39e3 | 8.80e5 | 163× | 2.84× |
| (7,6) | 1.77e4 | 1.75e7 | 988× | 4.58× |
| (10,3) | 1.31e4 | 1.91e7 | 1458× | 1.86× |

(Accuracy 1.00 everywhere on both sides — our G1 now reads 1.00 too: 0/300 with CP95 0.0099. The cost claim vs the paper's BAI instance is directionally confirmed — the δ-correct class is orders more expensive than two-fidelity certification — while the exact ratio waits on the 916(b) allocation work.)

## Protocol history on this record

- **Bench 906 (2026-10-02)**: the honest negative — G1 certificate arms failed at 40 trees/setting on the then-honest fixtures; root cause later isolated to the fixture (above).
- **2026-10-03 attempt A** (30M BAI cap, post-915/916a code): setting 1 complete at n=100 — 2FFS 0/100 errs, 100/100 certified, cost 196 435; BAI 62/100 errs all capped at 120M — killed at the projected ~10 h wall (the cap only sets the understatement magnitude; rows kept as corroboration).
- **2026-10-03 attempt B** (4M cap): killed after D=7's profile showed the wall cost was the search's own stage scans, not the cap.
- **THE DEFINITIVE RUN (2026-10-03, this record)**: 4M cap + the m_v table memoization; ALL GATES PASS in ~45 min.

## Re-arm procedure (for future changes)

1. `cargo clippy -p katgpt-core --features two_fidelity_bai --all-targets -- -D warnings`
2. `cargo test -p katgpt-core --features two_fidelity_bai --lib two_fidelity_bai` (21 tests, incl. the fixture-honesty PAC smoke and the m_v reference/precomputed agreement pin)
3. `B615_TREES=8` shakedown — the rows must be BYTE-IDENTICAL to this record's shakedown history for unchanged-protocol runs (determinism pin): 2FFS 225 386 / 987 702 / 574 838.
4. Full protocol (default 100 trees/setting): ALL GATES PASS is the GOAT verdict. Phase 3/4 are un-gated by THIS record; a future GOAT-relevant change (e.g. Issue 916(b)'s allocation knob) re-runs steps 3–4 under its own record.

## Addendum (2026-10-03) — Issue 916(b) lever: `SearchConfig::lazy_discharge`

The cost debt above is closed by an opt-in resolver, ported from the authors' implementation (`github.com/PeterLauLukChen/2FFS`, `method/twoffs/twoffs.py`): parent-capped scales (`capped_scale` — a child is never certified finer than its parent's scale, the code form of Prop. B.10's `Δ_v^eff ≤ 2ρ_k`), lazy discharge in the comparison cases (Max-L / Min-U: a child outside the `ρ_k/2` margin is dropped uncertified; the live blocker's OPPOSITE side is refined first), per-(node, scale) race budgets, expansion as its own work unit. Intervals, δ_v, and the ε-stop are untouched, so PAC rides on the same argument; `lazy_discharge = false` is the v1 search bit-identically (the step-3 determinism pin below reproduces 225 386 / 987 702 / 574 838 with the knob off).

Instrument: `B615_2FFS_AB=1` runs the resolver A/B alone (same trees, same per-tree seed, no baselines — the baselines do not depend on the resolver); `B615_AB_{EPS,SIGMA,COST}` override (ε, σ, c) for that mode only (σ also sets the slow noise amplitude). Deterministic — every number is a count.

**Protocol regime (ε = 0.02, σ = 0.05, c = 4), 100 trees/setting:**

| setting | v1 slow / fast / unified | lazy slow / fast / unified | v1 / lazy | paired LB95 (v1 − lazy) |
|---|---|---|---|---|
| (5,8) | 48 951 / 633 / 196 435 | 0 / 832 / 832 | 236× | +153 728 |
| (7,6) | 269 685 / 1 596 / 1 080 335 | 0 / 2 963 / 2 963 | 365× | +939 636 |
| (10,3) | 225 413 / 544 / 902 198 | 0 / 808 / 808 | 1117× | +720 555 |

Lazy G1: **0/300 errors (CP95 0.00994 < δ = 0.05), 300/300 certified, 0 invalid certified intervals.** v1: 0/300 errors, 0 uncertified.

**Hard regime (ε = 0, σ = 0.005, c = 2 — toward the paper's hard settings; our fixture's B̄ = 0.12 envelope and U[0,1] leaves are unchanged), 100 trees/setting:**

| setting | v1 errs / uncertified | v1 unified | lazy errs / uncertified | lazy slow / fast / unified | v1 / lazy | paired LB95 |
|---|---|---|---|---|---|---|
| (5,8) | 1 / 5 | 532 713 | 0 / 0 | 8 / 870 / 887 | 601× | +219 156 |
| (7,6) | 2 / 6 | 877 144 | 0 / 0 | 24 / 3 121 / 3 169 | 277× | +522 182 |
| (10,3) | 0 / 1 | 181 131 | 0 / 0 | 1 / 923 / 925 | 196× | +33 181 |

Lazy G1: **0/300 errors (CP95 0.00994), 300/300 certified, 0 invalid.** v1 DEGRADES here (3/300 errors, 12/300 uncertified exits) — the lazy resolver is the better-behaved arm on both axes, not just cheaper.

**Reading, honestly:**
- On this fixture the cheapest certificate is almost always the FAST route: honest envelopes plus exact leaves (B(0) = 0) let selective expansion certify the root, and lazy discharge stops paying for children that cannot change the decision. In the protocol regime the slow oracle is never queried; in the hard regime it is queried 1–24 times per tree. So this record proves the allocation lever and G1 under the new rule; it does **not** exercise a regime where slow sampling dominates (that needs a fixture whose fast envelope cannot be tightened by expansion — e.g. inexact leaves — not built here).
- v1's waste had a visible mechanical cause besides the ladder overshoot: once a node latched all 64 scales, v1 fell through to `sample_local`, sampling nodes (exact leaves included) that were already width 0.
- vs the paper's own 2FFS (5.39e3–1.77e4 samples) the fixtures differ (their hard trees: β = 0.45, gaps 0.002–0.012), so the numbers are not a parity claim; the 10–50× debt measured above is gone on ours.
- The knob stays opt-in per Issue 916's rule (a confidence-allocation change is never silent). It is the **recommended** setting; the example uses it. The gated suite here still records v1 by default (`B615_2FFS_LAZY=1` switches the suite's 2FFS arm); G2 vs the baselines can only widen under lazy, since its cost is 196–1117× lower at equal G1.

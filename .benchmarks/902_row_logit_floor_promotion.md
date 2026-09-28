# Bench 902 — `row_logit_floor` promotion lane: full-step G2, per-family retention walk, width decision (Issue 903)

**Status:** COMPLETE — all four promotion-lane boxes measured and PASS (2026-09-27). GOAT holds at **8-bit**; 6-bit demoted from default candidacy (admissible, policy-selectable). The `ForwardContext.logit_floor: None` default flip stays the owner act (Issue 903 non-goals; sibling WIP live in riir-infer this session).

Primitive: katgpt-core `row_logit_floor` (Bench 888, Issue 882 P2). Model-bound G1: riir-infer Bench 003 (T2/T3/T4 COMPLETE, PASS at 8/6-bit at 4K/16K/64K). This record is the promotion lane katgpt-rs owns on top.

## Box state (PROVENANCE, both G2 runs)

- M3 Max (16 cores, 64 GB), macOS 26.6.2, **AC power, battery 100% charged**.
- Run 1: loadavg 8.74 / swap 2.3 of 4.0 GB — shared with sibling agent sessions.
- Run 2 (confirm): loadavg 8.52 / same swap.
- Release profile, `--features row_logit_floor`, interleaved `ab_median_ratio(15 rounds, 5 iters/arm, 2 warmup)`, black_box on inputs, scale perturbation `(i % 3)·1e-3`, one output read per iteration consumed.

## G2 — the WHOLE decode step, both candidate widths

The promotion-lane delta over Bench 888's G2: the timed candidate arm is the consumer's per-call work exactly as shipped (riir-infer `FloorRow::head`, uncapped) — **per-call width policy (`min_width_for_tv`) → floor+code+LUT exp (`floored_coded_exp_inplace`) → in-place normalize → P·V → envelope+stats tally**. Bench 888 precomputed the width outside its timed arm and skipped the accounting; this closes that gap, and adds the 6-bit arm (the width decision's perf axis) and an N=16384 cell (long-context decode, where the exp-table win should grow with row length).

| cell (N, D) | b8 median (range) | b6 median (range) | bar |
|---|---|---|---|
| 4096, 64 | **0.9893** / 0.9943 (0.93–1.04 / 0.99–1.11) | **0.9925** / 0.9902 (0.92–1.03 / 0.89–1.00) | ≤ 1.01 |
| 4096, 128 | **0.9950** / 1.0002 (0.94–1.17 / 0.68–1.10) | **0.9958** / 0.9977 (0.93–1.06 / 0.39–1.81) | ≤ 1.01 |
| 16384, 64 | **0.9911** / 0.9891 (0.91–1.06 / 0.77–1.23) | **0.9901** / 0.9950 (0.93–1.09 / 0.92–1.75) | ≤ 1.01 |

Both runs, all six cells × both widths: **PASS** (two independent interleaved runs; medians 0.989–1.000). The whole step is latency-neutral-to-positive at BOTH widths — the LUT exp win (Bench 888's softmax-only −13.4% to −14.8%) pays for floor + code + width policy + envelope accounting with room to spare, and q·K + P·V dominate.

**Width perf verdict: 8-bit and 6-bit are indistinguishable at the whole-step level** (deltas between the two widths are inside the run-to-run spread). The LUT-fill difference (256 vs 64 `exp` per row) is noise against the dot products. The width decision therefore rests on QUALITY alone, not perf.

- G1-echo (harness sanity, one cell per shape): floored head output deviation within the convex-combination bound `2·TV_env·max|v|` — N=4096 D=64: b8 dev 2.7e-4 ≤ 3.1e-2, b6 dev 1.2e-3 ≤ 1.4e-1; N=16384: b8 1.4e-4, b6 6.1e-4. Mean env TV: 0.031 (b8) / 0.139–0.154 (b6), matching Bench 888's per-row bounds.
- G4 (alloc): **0 allocations** over 10 full-step calls including the envelope tally.

## G3 — no-regression, feature on

- `cargo test -p katgpt-core --features row_logit_floor --lib`: **2075 passed / 0 failed** (8 ignored), debug profile.
- Bench 888's full GOAT re-run on this tree (release): **ALL GATES PASS** — 144 envelope cells 0 over, trap-3 negative pinned (2.8×), masked-keys exact-0, width=+∞ bit-identical, 0 allocs, its own G2 head rows −0.02% / +0.65% ≤ 1.01.

## Per-family retention walk (the lossy-surface rule, riir-ai Issue 750 T3)

Instrument: riir-infer `row_logit_floor_ppl --families true` (landed this session, `ae3cd1e` + label fix `d4b39c5`) — two conditional views of the same paired per-token data: per-chunk families and base-margin buckets (`top1−top2`; the confident-flip class is `[8, ∞)` nats). Run: gemma-2-2b-it f16, 4096 tokens, 4 chunks, arms `base,b8,b6` (the admissible pair; b4 is inadmissible per Bench 003 T2 and b6s0's verdict rests on T4). Log: `/tmp/ri903run/t2_families.log` (~34 min, box at load 8–10).

**Aggregate cross-check — exact reproduction:** base ppl 33.7416, b8 +0.033% / 0.17% flips, b6 +0.066% / 0.90% flips — byte-identical to Bench 003 T2's recorded row (same corpus, chunking, tokens; deterministic fixture). The walk re-derives the aggregates it conditions on.

**Per-chunk families** (flips / 1024 tokens, mean |ΔNLL|):

| chunk | b8 flips | b8 flip% | b8 mean\|ΔNLL\| | b6 flips | b6 flip% | b6 mean\|ΔNLL\| |
|---|---|---|---|---|---|---|
| 0 | 1 | 0.10% | 0.00429 | 10 | 0.98% | 0.01858 |
| 1 | 2 | 0.20% | 0.00434 | 6 | 0.59% | 0.01626 |
| 2 | 2 | 0.20% | 0.00416 | 9 | 0.88% | 0.01617 |
| 3 | 2 | 0.20% | 0.00418 | 12 | 1.17% | 0.01824 |

No family concentrates the flips: b8 spans 0.10–0.20% (max/mean 1.2×), b6 spans 0.59–1.17% (max/mean 1.3×). The Orthrus per-prompt axis is likewise on record from Bench 003: T3a 6/6, T3b 3/3, T3c 3/3 seq-exact at every arm including b4 — 0 behavioral flips on any prompt.

**Base-margin buckets** (the confident-flip class is the last row):

| bucket (top1−top2, nats) | n | b8 flips | b8 flip% | b6 flips | b6 flip% |
|---|---|---|---|---|---|
| [0, 0.5) | 1385 | 7 | 0.51% | 37 | 2.67% |
| [0.5, 2.0) | 1784 | 0 | 0.00% | 0 | 0.00% |
| [2.0, 8.0) | 875 | 0 | 0.00% | 0 | 0.00% |
| [8.0, ∞) | 52 | 0 | 0.00% | 0 | 0.00% |

**Every flip at BOTH widths lives in the near-tie bucket [0, 0.5).** The disqualifying class — a flip on a confident prediction — is empty at 8-bit and 6-bit (52 confident tokens, 0 flips; b6's mean |ΔNLL| there is 1e-5, i.e. the code does not move those rows at all). This is the failure shape the lossy-surface rule warns about, answered negative: the coded softmax's errors are confined to coin-flip argmaxes whose downstream cost is a re-roll, not a behavior change.

## Width decision

Perf axis: **dead heat** — the whole-step G2 medians differ by ≤ 0.6 pp between the widths (inside the run-to-run spread); the LUT-fill difference (256 vs 64 exp) is noise against q·K + P·V. Quality axis decides:

| | b8 | b6 |
|---|---|---|
| T2 aggregate flips (4096 tok) | 0.17% | 0.90% |
| family-walk flip concentration | 0.10–0.20% per chunk | 0.59–1.17% per chunk |
| confident flips ([8,∞), n=52) | **0** | **0** |
| mean env TV (full step) | 0.031–0.034 | 0.139–0.154 |
| T3 needle (16K/64K) | PASS, 0 flips | PASS, 0 flips |

**Decision: 8-bit is the width of record for promotion; 6-bit is DEMOTED from default candidacy** — admissible (every gate it faces passes, all flips near-tie), but it pays 5.3× the aggregate flip rate for no measured perf return. 6-bit stays selectable via `RowLogitFloorPolicy.bits = 6` for a throughput-critical consumer; it is not the promoted default. (4-bit is out on Bench 003 T2: 4.32% flips, m_Y perturbation at 16K, vacuous envelope at 64K.)

## Verdict

All four Issue 903 boxes hold: per-family retention walk (no family concentration, zero confident flips), full-step G2 (latency-neutral-to-positive at both widths, two reproducing runs), G3 (2075 lib tests + Bench 888's full GOAT green at the feature), G4 (0 allocs including the envelope tally). **The GOAT holds at 8-bit.** The primitive stays opt-in: the default flip is the owner act, verdict-reviewed AGREE (2026-09-27) — and it is a TWO-part change (riir-infer's first default feature + the `#[cfg]`-gated field default), with an open scope question: `attend_row` is shared by the gemma-2 and llama forwards and every model-level walk on record is gemma-2-2b, so a global default would reach llama with no model-level walk (per-architecture scope, or a llama walk, first). What the flip needs is recorded in Issue 903's non-goals; the measured evidence here is final.

## Reading it honestly

- The G2 ranges are wide (up to 0.39–1.81 on one b6 round-set) because the box carries sibling sessions at load 8–9; the medians are the asserted quantity and they reproduced across two runs. Both runs' PROVENANCE printed by the bench itself (`box:` lines).
- The G1-echo bound is a harness sanity check (the envelope is Bench 888's claim, re-verified per shape here), not a new quality result.
- The family walk re-runs T2 with 3 arms instead of 5: b4 (inadmissible) and b6s0 (T4-settled) are skipped to bound the measurement at ~35 min on the shared box. Aggregate b8/b6 numbers remain comparable to Bench 003 T2 (same corpus, same chunking, same tokens) and the walk's base arm re-derives them as a cross-check.

Session: issue903-promotion-lane (idle cycle, 2026-09-27)

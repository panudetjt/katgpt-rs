# Plan 612 PISA Pyramid GOAT Gate Report — G2 ISO-QUALITY NEGATIVE RESULT (latency slope CONFIRMED)

**Date:** 2026-09-30
**Plan:** [`.plans/612_pisa_pyramid_lse_block_selection.md`](../.plans/612_pisa_pyramid_lse_block_selection.md)
**Research:** [`.research/595_PISA_Pyramid_Sparse_Attention.md`](../.research/595_PISA_Pyramid_Sparse_Attention.md)
**Paper:** [arXiv:2609.31093](https://arxiv.org/abs/2609.31093) — PISA, Tang et al., Sep 2026
**Tests:** `crates/katgpt-attn/tests/bench_612_pyramid_goat.rs` + `bench_612_alloc_check.rs`
(run: `PYRAMID_612_FULL_DIR=/Volumes/SDXC1TB/pyramid_612/full cargo test --release -p katgpt-attn --features pyramid_topk --test bench_612_pyramid_goat --test bench_612_alloc_check -- --nocapture`)
**Verdict:** **G2 iso-quality FAIL — the pyramid walk loses −0.133 Recall@8 vs the single-level exact-LSE scan at 32K+ (−0.091 over the full set). The latency-complexity claim CONFIRMS exactly (aggregate slope 1.07 vs 1.99). `pyramid_topk` STAYS OPT-IN, no promotion — the MSA/HGA slot discipline.**

---

## Executive summary

The first real-tensor replay of the slot delivered **two Defect-found-by-the-gate
fixes** to the Phase-1 primitive (both invisible to the structural tests), then
measured the head-to-head the plan pre-registered. The complexity claim is
real: the aggregate selection slope is **1.07 vs 1.99** (log-log, N queries ×
per-query cost vs N) — the paper's log-linear selection story reproduced on
our tensors. The quality claim is not: at every length the pyramid walk's
per-level top-8 pruning loses recall against a single-level **exact-LSE**
scan, and the loss WIDENS with depth (−0.045 at ~4K committed geometry,
−0.091 full-set overall, **−0.133 at 32K+**). The paper's own framing
(PISA vs mean-scoring BSA) still holds — pyramid_lse beats the mean scan by
+4.1 pt (committed) / +4.1 pt (full) — but the plan's bar was iso-quality vs
the LSE scan, and that bar fails.

The feature stays opt-in with the slot-ledger negative. No consumer exists
today (league pins ≤4K where single-level wins per the paper's own Table 4 —
and per our latency table below, emphatically).

## Defects the gate caught before any number was quoted (T2.2 pre-work)

Both shipped in Phase 1 (2026-09-29); both passed the module's structural
tests; both were caught by the gate's unit pins within minutes:

1. **The walk expanded the WRONG NODES** — `coarse_to_fine_select` pushed
   candidate-ARRAY POSITIONS into the retained set instead of node indices
   (`scratch.nonforced.push(ci)` vs `push(c)`). Any call whose candidate
   order diverges from identity (i.e. every non-trivial call after the first
   argtopk returns score-ordered picks) expanded wrong subtrees and emitted
   candidate positions as leaf ids. Invisible to the module tests because
   the forced leaves are appended BY VALUE at the end and the structural
   asserts (sorted/bounded/forced-present) hold over garbage. Caught by the
   new needle canary (`exact_lse_selects_needle_block_mean_rung_misses_it`).
2. **The leaf `ExactLse` arm returned `ln_z` without `+max`** — the
   max-shifted `logsumexp_parts` tail is not comparable across blocks (each
   block's max is its own subtraction constant), and it deletes exactly the
   needle signal exact-LSE exists to preserve — the MSA/HGA dilution class.
   The internal-level arm already computed true LSE; the leaf arm now does
   too. Caught by the new Jensen pin
   (`leaf_ladder_jensen_pin_mean_plus_ln_c_le_lse`, the T2.2 theorem pin).

Both fixes are in `pyramid_topk.rs` with regression tests; 10/10 module
tests green.

## Gate setup

- **Tensors:** qwen38-27B Q4_K_M FA layers (Issue 908 capture; Bonsai PQ2_0
  blocked — no ternary whole-model lane). Committed subset: 16 K bins (2
  layers × 2 kv heads × 4 source lengths, stride-sampled to ~4 097 rows) +
  30 Q mats, always runs in CI. Full set (1.2 GB, all 4 kv heads, TRUE
  contiguous 4 096/16 384/32 768/65 536): env-gated, every file
  BLAKE3-verified against the manifest.
- **Family** = layer × kv head × source length × query position; the query
  sees only the causal prefix. Reference = full softmax attention mass per
  q-head (std `exp`), group-summed over the 6 q-heads sharing the kv head,
  per C=64 block. True top-8 = argmax block mass. All arms share the
  group-summed query, K=8, and the NSA forced leaves.
- **Pins (theorems, hard-asserted per family):** Jensen
  (`mean + ln cnt ≤ true LSE` per block) and captured-mass
  (`mass(sel) ≤ mass(top-|sel|)` for every arm). Both held everywhere
  after the fixes.

## G1 — selection quality (real tensors, per-family never pooled)

Committed subset (64 non-trivial families, ~4K-row stride geometry):

| arm | recall@8 | mass-ratio |
|---|---|---|
| single_mean (BSA class) | 0.7695 | 0.8753 |
| **single_lse** (exact scan) | **0.8906** | **0.9483** |
| pyramid_mean | 0.7695 | 0.8735 |
| pyramid_halfvar | 0.7637 | 0.8903 |
| pyramid_lse | 0.8457 | 0.9176 |

Full set (128 non-trivial families, TRUE lengths 4K–64K):

| arm | recall@8 | mass-ratio |
|---|---|---|
| single_mean (BSA class) | 0.6475 | 0.8030 |
| **single_lse** (exact scan) | **0.7793** | **0.8988** |
| pyramid_mean | 0.6113 | 0.7833 |
| pyramid_halfvar | 0.6084 | 0.7964 |
| pyramid_lse | 0.6885 | 0.8246 |

Paired recall diff (pyramid_lse − single_lse): **−0.0449** (sd 0.1369,
n 64) committed; **−0.0908** (sd 0.2342, n 128) full; **−0.1328** (sd
0.2741, n 64) at L ≥ 32K — the pre-registered iso-quality bar (≥ −0.01)
**FAILS**, and the loss grows with depth exactly as the walk-pruning
mechanism predicts. Per-group tables live in the gate output; the honest
per-family reading is that no group at any length shows the pyramid ABOVE
the exact scan.

Readings worth keeping:

- **pyramid_mean == single_mean at ~4K** (0.7695 both) — with a consistent
  scorer the walk itself is near-lossless at shallow depth; the pyramid_lse
  gap is the internal-level LSE-over-child-SUMMARIES approximation plus
  per-level K pruning, not a walk bug.
- The paper's OWN comparison class (mean-scoring BSA) still loses to
  pyramid_lse: +4.1 pt on both sources — the LSE ladder's quality gain is
  real. It does not clear the plan's bar (vs the LSE scan).
- `single_lse` itself degrades with length (0.89 → 0.78) — more leaves,
  more near-tie mass; both selectors degrade, the pyramid faster.

## G2 — latency (load-borne claim)

Per-query selection latency (µs, median of 30, release, M3 Max, AC,
loadavg ≈ 6–8 during the run — sibling sessions active; medians robust,
disclose-only):

| N | pyramid_lse | single_lse | single_mean |
|---|---|---|---|
| 4 096 | 22.6 | 2.0 | 0.0 |
| 16 384 | 25.5 | 7.7 | 0.1 |
| 32 768 | 26.4 | 15.3 | 0.2 |
| 65 536 | 27.9 | 30.5 | 0.5 |

- **Aggregate slope (t·N vs N, log-log): pyramid 1.07, single_lse 1.99,
  single_mean 1.89** — the paper's log-linear selection claim reproduces
  almost exactly (asserted in release: slope gap ≥ 0.5 ✓).
- **Per-query crossover N\* ≈ 65K** — pyramid_lse wins only at 65 536
  (1.10×). Below that the exact scan is cheaper AND better. The paper's
  "BSA wins ≤16K" is confirmed emphatically.
- Per-query constants reflect the current per-dot dispatch (~600 dot-256
  calls ≈ 20 µs fixed for the walk; a fused kernel would drop it ~10×) —
  the SLOPE is the claim, not the constant.

**The G2 verdict is the combination: at the only length where the pyramid
wins latency (≥64K), it loses quality (−0.13 recall); below that it loses
both. Iso-quality FAIL ⇒ documented negative.**

## G4 — alloc

`bench_612_alloc_check`: **0 allocations** across 10 warmed selections at
gate geometry (8 192 keys, head_dim 256, GQA group 6) — PASS.

## T2.5 — canaries

Position-0 real-tensor family (prefix = 1 key): all five arms select
exactly `[0]` — the N ≤ C degenerate contract on live data. PASS. (The
constant-keys and N ≤ C synthetic canaries live in the module tests,
both green.)

## Verdict + disposition (T3.1)

| Gate | Result |
|---|---|
| G1 pins (Jensen + captured mass) | ✅ PASS (after the two defect fixes) |
| G1 quality vs BSA-mean | ✅ +4.1 pt (the paper's own comparison — the ladder gain is real) |
| **G2 iso-quality vs single-level-LSE (load-bearing)** | ❌ **FAIL −0.133 at 32K+ (documented negative)** |
| G2 latency slope | ✅ 1.07 vs 1.99 (asserted in release) |
| G4 zero-alloc | ✅ 0 |
| T2.5 canaries | ✅ |

**`pyramid_topk` stays opt-in. No default-on promotion (no head-to-head win
AND no consumer).** The slot-ledger row (README) records the negative. T\*
calibration for `meta_router` dispatch: the measured crossover ≈ 65K on
this implementation — far above every live serving shape; the dispatch
stays unarmed. Re-open triggers: a fused walk kernel AND a long-context
consumer, re-run this gate.

## Cross-repo pointers (T3.3)

- riir-train Issue 586 (training proof lane): Item A's gate (ii) reads
  "trained-sparse retrieval > MGATE-0 re-measured on our dense twin" —
  this gate's harness is the MGATE instrument; the pyramid-vs-scan gap
  measured here is the honest expectation for selector-side recall at
  length. Item B's floor measurement reuses the same replay.
- riir-ai Issue 1017 (limelight Wave-2 zone-salience LSE pyramid):
  game-side consumer of the same math — the Jensen ladder's QUALITY gain
  (mean → LSE) is confirmed here at every length; the pyramid WALK's
  pruning loss is selection-specific and does not transfer to the salience
  pooling shape (no walk there — LSE over fixed zone summaries).

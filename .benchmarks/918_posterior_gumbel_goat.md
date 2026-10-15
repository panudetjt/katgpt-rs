# Bench 918 — posterior (truncated-Gumbel) inverse sampler GOAT gate (Issue 918)

**Status:** COMPLETE — ALL GATES PASS (2026-10-05).

- **Primitive:** `keyed_posterior_gumbel_noise` in the DEFAULT-ON
  `katgpt-core` feature `ac_prefix` (an addition to the Plan-614 keyed
  Gumbel substrate, not a new feature). Given logits `ℓ` and a REALIZED
  pick `y`, samples noise `ξ*` from the exact posterior
  `p(ξ | argmax(ℓ+ξ) = y)` via the MAX-FIRST construction: `M ~
  Gumbel(LSE(ℓ))` at a reserved keyed slot (`u32::MAX`, winner-independent
  — two draws at one key with different winners share their max, the
  coupling the residual-ordering property holds on), winner pinned
  `ξ*_y = M − ℓ_y`, losers `ξ*_k ~ Gumbel(0)` truncated above at
  `M − ℓ_k` by the inverse CDF `−log(−log(u_k·F(M−ℓ_k)))` over the SAME
  keyed uniform stream the prior noise maps (common random numbers).
- **Source:** Issue 918 (execution-class, zero novelty claims) — the
  posterior construction is Zhang et al. 2026 via arXiv:2610.00497 App E
  (GSF); the trick is Gumbel 1954; the OT identification implicit in
  Galichon 2016. Distill note: riir-train Research 466 §3. The declared
  consumer is the GSF training coupling (riir-train Plan 437 Phase
  1b/4b); NOT a replay tool for our own keyed sampler (a recorded seed
  already replays exactly — the issue's retraction clause, carried in the
  fn's docs).
- **Run:** `cargo bench -p katgpt-core --bench bench_posterior_gumbel --
  --nocapture` (release bench profile; `ac_prefix` is default-on).

## Box state (the G2 law)

M3 Max, AC (battery 83% charging), loadavg **2.17 / 2.53 / 2.61** at run
start (`uptime`: up 19 days). A shared box (sibling sessions active) —
every latency below is a REGRESSION CEILING, not a quiet-box figure.

## Gates

| Gate | Verdict | Reading |
|---|---|---|
| G1 identity | **PASS** | `argmax(ℓ + ξ*) == winner` on 100% of fixture draws (lib: 2,000 positions × 4 winners × 4-vocab; bench live rerun: 500 × 5 winners × 64-vocab). Red arm: the same noise against foreign logits breaks the identity (the identity belongs to the construction). |
| G1 determinism | **PASS** | same key → bit-identical draw (pinned in lib + bench). |
| G1b joint distribution | **PASS** | winner AND loser coordinate marginals match the rejection-filtered prior (n=6,000 accepted draws, V=5, winner p≈0.28): KS winner **0.0078** · pooled losers **0.0126** · top-loser **0.0135**, all ≪ the 0.05 bound (α=0.001 critical ≈ 0.036 at this n); means within 4 joint SEs. **Red arm fires**: the biased construction (free losers, winner pinned above their max) FAILS the loser checks — pooled KS **0.1158** / top-loser KS **0.1753**, 2–3.5× the bound. The gate can catch the coupling the issue forbids. |
| Residual ordering (Thm 1) | **PASS** | coupled pairs (shared key → shared max), distinct winners: **0 violations over 10⁴ pairs**. Independent-key posteriors: also 0 — measured SHARPER than the issue's spec: ANY two valid posterior draws satisfy the inequality structurally (pinned winner vs truncated loser ⇒ d1 > M¹−M² > d2), so the discrimination arm contrasts the posterior against UNCONDITIONAL noise, where the violation fraction is **0.5023** (both sides iid keyed Gumbel). The one-posterior-side variant measured 0.2325 — the winner-pinning asymmetry biases holds; recorded, not used as the control. |
| G2 perf | **PASS** | same order as `keyed_gumbel_max_sample`, interleaved median-of-ratios over the shared `ab_timing` harness (Issue 855's loud-zero defence — adopted after the gate flagged this bench's hand-rolled loops): V=256 **b/a median 2.08×** (rounds 1.19..2.27; 11,741 vs 6,096 ns/iter) · V=32,000 **median 2.03×** (rounds 1.94..2.09; 1.16 µs vs 0.58 µs) — bar ≤ 5×; the ~2× matches the transcendental op count (4 per loser vs 2 per token). The bench's first run dead-code-eliminated the comparison arm (`let _ =` sink, 0 ns) — the very failure class the harness exists to make LOUD. |
| G4 alloc | **PASS** | 0 allocations / 1,000 calls (V=1,024), counting-allocator canary live (`assert_counter_is_live`). |

## Refactor rider (bit-pinned)

The uniform construction under `keyed_gumbel_noise` was extracted into
`keyed_unit_interval` (the posterior consumes `u`, not its Gumbel image).
The extraction is pinned **bit-identical** to the Plan-614 inline
arithmetic by a dedicated test over a 80-key grid (seeds × positions ×
tokens incl. `u32::MAX`); the categorical-distribution pin
(100k draws, ±1%) also still passes. The verify-loop consumers ride those
exact bits.

## En-route defects caught by the gates

1. **The max's sign** — `M = LSE + g` (Gumbel addition), not `LSE − g`.
   The first build flipped it; the argmax identity STILL passed (it is
   structural — winner pinned at M, losers truncated below M, regardless
   of M's law) while the winner-marginal KS read **0.293**. The joint
   gate caught what the identity gate cannot: a construction whose
   conditioning event has the wrong law still reproduces every pick.
2. **The residual-ordering control** — the issue's "decoupled ~50%" arm
   does not fire against independent-key posteriors (0 violations;
   structural, see above). The control that discriminates is
   unconditional noise on BOTH sides (0.5007 measured). Recorded here so
   the next reader does not re-derive it.

## Posture

`ac_prefix` is DEFAULT-ON (Plan 313); this addition rides it — no feature
count change, no catalog entry. The bench row declares
`required-features = ["ac_prefix"]` (runs at default features; skipped
loud under `--no-default-features`, the E0601 guard the row carries by
convention). Lib suite: **2087 passed / 0 failed** at default features
(+5 tests: bit-pin, G1, G1b, residual ordering, defensive conventions).

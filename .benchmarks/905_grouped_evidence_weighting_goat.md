# Bench 905 — Grouped-Evidence Noise-Weighting GOAT (Issue 913 T1–T4, riir-train Research 463)

**Status:** RECORD — T1–T4 landed (`9ad5e21ce`), GOAT G1–G4 executed, opt-in (no default-on claim; consumers unscheduled)

Source: riir-train Research 463 (`463_EasyPPO_Critic_Stabilization.md`) ← arXiv:2609.36802
("EasyPPO"); issue: Issue 913 (closed; record in HISTORY.md). Module:
`crates/katgpt-core/src/grouped_evidence.rs`, feature `grouped_evidence_weighting` (default-off,
implies `best_belief` + `rating`). Substrate consumed, not rebuilt: the `best_belief` Newton/Lentz
Beta-quantile solver (extracted as `beta_quantile_cf` for fractional shapes — the integer path
calls it with the identical float ops, so the LUT is bit-unchanged) and `rating::update_scored` /
`update_f32` (the T4 variants are one-line delegations at `K_eff`).

## The primitives

| Operator | Law | Gate |
|---|---|---|
| `variance_floor(Δ, n)` (T1) | `Δ/(2√n)` — Popoviciu over an n-draw mean; tight at the two-point `{0, Δ}`, p=½ | G1 |
| `filter_bias_bound(Γ, P(¬C))` (T2) | `2Γ·P(¬C)` — censoring decomposition; tight at `R=+Γ` on C, `−Γ` on ¬C | G1 |
| `best_belief_score_weighted(groups, ExogenousSigma, floor, ε)` (T3) | Beta LCB at `α = 1+Σw̃k`, `β = 1+Σw̃(n−k)`, weights `1/max(σ̂,ε)` mean-one per observation; Kish `n_eff` + `Estimand` + `SigmaProvenance` on every readout | G1–G4 |
| `noise_scaled_k` + `update_scored_noise_scaled` / `update_f32_noise_scaled` (T4) | `K_eff = K·ε/max(σ̂,ε)`; `ε/ε ≡ 1.0` in IEEE ⇒ bit-identical to fixed K at the floor | G1, G3 |

The endogeneity law is enforced in the TYPE: `ExogenousSigma` has `prior_epoch` / `leave_one_out` /
`design` constructors and no plug-in constructor; the provenance rides the readout.

## G1 — correctness + the two negative controls (14 tests release / 15 dev, all passing; seeded, deterministic across 3 runs)

- **T1**: two-point extremal attains the floor analytically at 4 (Δ, n) cells (≤1e-6 rel) and
  empirically (16-draw means, n=40k, SD within 2% of `Δ/(2√n)`); 2000 random 5-point laws on
  `[0, Δ]` all sit at or below it; strictly decreasing in n; `n=0` reads as `n=1`.
- **T2**: tight to ≤1e-6·Γ across the P(¬C) ∈ {0, 0.05, …, 1} sweep × 3 Γ; dominates 20k random
  bounded filters; clamps out-of-range P(¬C).
- **T3 extraction**: `beta_quantile(1+S, 1+F, ε)` is bitwise `best_belief_score_cf(S, F, ε)` over
  a 6×5×4 grid (incl. off-LUT S=40/120 and off-grid ε).
- **T3 normalisation**: `successes_w + failures_w = n_raw` (≤1e-4); Kish `n_eff` matches the
  hand formula and is `< n_raw` under unequal weights; lower ε strictly more conservative.
- **GOAT (a) — plug-in NEGATIVE CONTROL** (homogeneous p=0.15, 40 groups × n=8, 3000 trials,
  floor = `variance_floor(1, 8)`): pooled-rate bias **plug-in σ̂ −0.0418** (pulled toward 0: k=0
  groups have σ̂=0 → floored → the LARGEST weight), **prior-epoch σ̂ +0.0009**, unweighted
  +0.0008. The gate bites, in the direction the issue predicted.
- **GOAT (b) — best-arm identification** (4 arms p = 0.40/0.43/0.46/0.50, 12 groups × n=20,
  group noise σ_g ∈ {0.02, 0.20} 50/50 zero-mean uniform — the same-rate condition holds;
  design σ̂ = `√(variance_floor(1,n)² + σ_g²)`, ε=0.05, 4000 paired trials): P(correct)
  **weighted 0.6853 vs unweighted 0.6172**, paired Δ **+0.0680, LB95 +0.0574 > 0** — PASS.
- **T4 GOAT — MSE dominance** (true gap 200, mixed 64-round / 1-round matches, design σ̂ =
  `variance_floor(1, r)`, ε = `variance_floor(1, 64)`, 400 seeds × 600 updates, time-averaged
  over the last 400): MSE **adaptive 161.3** vs fixed K=32 2998.4 (Δ +2837, LB95 +2767) **and vs
  fixed K = adaptive's own mean step 18.0: 1667.7** (Δ +1506, LB95 +1456) — the win is the
  weighting, not a smaller step. PASS.
- **T4 plug-in NEGATIVE CONTROL / unbiasedness pin** (4-round matches only, 200 seeds × 3000
  updates): asymptote **design σ̂ 200.4** (true 200) vs **plug-in σ̂ 331.8** — the match's own
  `√(s(1−s)/r)` gives a clean sweep (s=1) the full step and inflates the gap by +66%.

## Report-the-Floor disclosure — coverage MEASURED, no UQ claim made

The T3 readout is an ε-quantile, so its parameter coverage was measured in GOAT (b) (nominal
P(p ≥ LCB_0.05) = 0.95): **weighted 0.8832, weighted-at-Kish-n_eff 0.8948, unweighted 0.8454**.
All three UNDER-cover: the Beta-Bernoulli posterior models binomial noise only, and the fixture's
between-group overdispersion is outside it. Weighting narrows the gap; Kish scaling buys +1.2pp,
which is not worth an API variant (not shipped; `n_eff` is disclosed instead). **Verdict: the
readout ships as a RANKING score (GOAT (b) is the claim), NOT a calibrated interval** — no
coverage claim is made, so the conformal-naive floor (`ConformalIntervalCalibrator<
SeasonalNaiveForecaster>`) has nothing to bind. A consumer wanting a coverage claim must add an
overdispersion term and beat that floor first. This is the same refusal shape as the existing
integer `best_belief_score` under overdispersed evidence.

## G2 — latency

`best_belief_score_weighted` at G=16 groups (fractional cold path — Newton + Lentz):
**360.6 / 376.0 / 385.9 / 372.5 ns/call** across 4 release runs (best-of-5 × 20k calls,
`black_box` at both ends, loud-zero assert). Bar 5 µs — **13× headroom**. The uniform-weight path
inherits the integer LUT (a few ns). Box: M3 Max, AC, 64 GB (30 GB used), concurrent sibling
agent sessions + one riir-ai GPU A/B in flight (CPU 54% at session start) — a loaded-box reading,
an upper bound.

## G3 — no regression

- Uniform weights (equal σ̂, all floored, NaN σ̂, empty-group weight ignored, zero evidence) →
  bitwise `best_belief_score(Σk, Σ(n−k), ε)` at 8 ε values incl. the LUT five, an off-grid ε and
  both extremes; `Estimand::Unweighted`, `n_eff = n_raw`.
- T4 at σ̂ ≤ ε (and NaN) → bitwise `update_scored` / `update_f32`.
- `best_belief.rs` refactor: the integer cold path now calls `beta_quantile_cf` with identical
  float ops; `best_belief` tests 18/18 at default features; full katgpt-core lib at
  `--features grouped_evidence_weighting --release`: **2078 passed / 0 failed / 7 ignored**.
- Clippy `-D warnings` clean at both postures (`-p katgpt-core --lib --tests`, default and
  feature); the fractional `beta_quantile` wrapper is feature-gated so default builds carry no
  dead code. The module compiles to nothing at default features.

## G4 — allocation

`goat_g4_alloc_free` (gated `any(debug_assertions, alloc_tracking)`, Issue-741 shape — executes
in dev; compiles out of a bare release run, disclosed rather than counted): T1 + T2 + a weighted
and a uniform T3 readout + both T4 updates — **0 allocations** (TrackingAllocator).

## Non-claims

No default-on promotion (consumers unscheduled — riir-clippy Issue 139, riir-dao, riir-reflex,
riir-instinct, riir-ai rows in the issue). No coverage claim (above). The negative controls prove
the API cannot stop a mislabelled plug-in σ̂ — the provenance type makes the mistake visible at
the call site, it does not make it impossible.

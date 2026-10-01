# Bench 904 — Dirichlet-Distribution Primitives GOAT (Issue 912 T2+T3, Research 596)

**Status:** RECORD — T2+T3 landed, GOAT G1–G4 executed, opt-in (no default-on claim)

Source: [Research 596](../.research/596_Simplex_Diffusion_Models.md) (arXiv:2609.35553 "Simplex Diffusion
Models"); issue: `.issues/912_dirichlet_dist_primitives.md`. Module:
`crates/katgpt-core/src/dirichlet_dist.rs`, feature `dirichlet_dist` (default-off). T1 (the
`data_probe/markov.rs` α-honoring repair, commit `7745cfb9c`) is this module's premise: its docstring
names the linear-f32 small-α underflow (α ≲ 0.02 → uniform fallback) that this module's log-space
sampler exists to avoid.

## The primitives

| Operator | Law | Gate |
|---|---|---|
| `sample_conc_into(p, c, seed, out)` | `Y ~ Dir(c·p)`, mean exact by construction; `Var = p(1−p)/(c+1)`, `Cov = −p_i p_j/(c+1)` | G1 + G2 + G4 |
| `thinning_into(x, alpha, rho, seed, out)` | Prop A.2: `B_i ~ Beta(ρα_i,(1−ρ)α_i)` reweight ⇒ `Y ~ Dir(ρα)` under Dirichlet input; NOT mean-preserving on fixed vectors | G1 |
| `DirichletEma<const M, const K>` | Prop E.1: recursive mean path (ε=∞ routing, bit-identical to a plain EMA) + drawn path `L ~ Dir(ε·shares)`; ring truncation named | G1 + G4 |

## G1 — correctness (closed-form + exactness pins, 13 tests, all passing)

- **Mean/variance/covariance closed forms** at three grid cases:
  - A: `p=(0.5,0.3,0.2)`, `c=20` (α_min=4): mean tol 4e-3, Var₀ 0.0119±0.0015, Cov₀₁ −0.00714±0.0015 (n=40k)
  - B: `p=(0.6,0.3,0.1)`, `c=1` (α_min=0.1): mean tol 1e-2 (n=40k)
  - **C — the underflow regime: `p=(0.5,0.3,0.2)`, `c=0.05` → α_min=0.01**, where the T1-documented
    linear sampler collapses to uniform: mean tol 1.5e-2 AND mean row-max > 0.8 (near-vertex peaked;
    uniform would read ≈0.52). The log-space sampler is exact where the linear one cannot go.
- **Determinism**: same seed → bitwise-identical draw.
- **Thinning variance ratio** on seeded Dir(8p) draws, ρ=0.5: sample ratio ≡ `(c+1)/(ρc+1) = 1.8` ±6%
  — NOT ρ (the round-1 issue wording's false claim, corrected there). Post-transition mean ≈ p and
  covariance ≈ `−p_i p_j/(ρc+1)` — the distribution matches Dir(ρα), not merely its variance.
- **ρ=1 bit-identity**: `out` bitwise == `x`, no rng consumed.
- **Fixed-vector Jensen bias pinned NEGATIVE** (`p=(0.9,0.07,0.03)`, `c=2`, `ρ=0.5`, n=40k):
  measured `E[Y] = (0.8473, 0.1018, 0.0509)` vs the paper's simulated `(0.847, 0.103, 0.051)` —
  three-decimal agreement, biased away from p by >3e-2 on the dominant coordinate. The non-claim
  is a measurement, not prose.
- **DirichletEma**: mean path bit-identical to a naive EMA at the frozen op order (scale-all then
  add) at every push; pre-wrap `shares` ≡ normalized `h` (≤1e-6); drawn expectation ≡ ring shares
  (±1e-2 at n=8000) while diverging from the recursive `h` by >5e-2 post-wrap — the `L_{i,M}`
  truncation is real and pinned, not papered over; `h` ≡ the raw infinite-EMA weighted sum of the
  full history (≤1e-5); `Var[L_j] ≡ share_j(1−share_j)/(C+1)` with `C = ε(1−β^M)` ±8%; ε=1e-3
  commit frequency >0.99; ε=1e6 draws within 1e-2 of the deterministic shares.

## G2 — latency

`sample_conc_into` at N=8: **183 ns/call** (release, best-of-5 rounds × 20k calls, black_box at
both ends — the loud-zero defence). Bar: ≤ 1 µs. **5.5× headroom.** The assert is release-postured
(`#[cfg(not(debug_assertions))]`): debug builds run the call for correctness without the latency
bar (a debug-profile bar would measure the unoptimized binary — the house rule).

## G3 — no regression

The feature is default-off: default-features builds compile the module to nothing; every existing
stream is untouched (T1 kept the α=1 exponential path verbatim, and this module adds new code only —
no existing call site changes). Full katgpt-core lib at `--features dirichlet_dist`, release:
**2076 passed / 0 failed**. Clippy `-D warnings` clean at both postures (default + feature).

## G4 — allocation

`goat_g4_alloc_free` (gated `any(debug_assertions, alloc_tracking)` per the Issue-741 gate shape —
the pin keeps an executing lane in dev and any alloc-tracking build; it compiles out of a bare
release run, disclosed here rather than counted as a pass): one `sample_conc_into` + one
`thinning_into` + `DirichletEma<8,3>` construction + 10 pushes + mean/shares reads +
`drawn_weights` — **0 allocations** (TrackingAllocator, reset→draw→read). All buffers are
caller-owned or const-generic stack arrays; the log-space scratch IS the output buffer.

## En-route findings (paid for, recorded)

1. **Two-term LSE bug (real, caught by the pins)**: the first thinning draft computed
   `lse = lg_a + ln_1p(exp(lg_a − mx))` — after a max shift the OTHER term is exactly 1, so the
   `ln_1p` argument must be the MIN's shift. The wrong form gives `lse = lg_a + ln 2` whenever
   `lg_a > lg_b`, forcing `B = 1/2` for half the draws — a bimodal, Beta-incorrect weight. The
   closed-form pins caught it in the first red run (variance ratio 1.445 vs 1.8; Jensen bias
   reading 0.889 WITH the bug — the corrected operator reads 0.8473 against the paper's 0.847).
   Fixed in `thinning_into` with the derivation commented at the site.
2. **Drawn-variance closed form**: the class-wise concentration of the aggregated draw is
   `C = ε·Σw_raw = ε(1−β^M)`, NOT `ε` — the raw geometric weights sum to `1−β^M < 1` (the
   truncation). The first test formula used `ε`, measured 0.0568 vs predicted 0.1483; the corrected
   form predicts 0.0566 — matching. Module doc corrected to the `C`-form.
3. **`thinning_into` signature**: the issue's one-line signature (`x, rho, seed, out`) omitted α;
   exactness REQUIRES it (the `B_i` shapes depend on `α_i` — that IS Prop A.2). The shipped
   signature carries `alpha: &[f32]`; recorded in the module docs and the catalog.

## Non-claims (restated)

No coverage/prediction-interval claim → the conformal-naive floor does not bind these primitives.
No default-on promotion; consumers unscheduled (T4 notes live in the module docs). Thinning is NOT
a general mean-preserving operator — the bias is pinned, not just documented.

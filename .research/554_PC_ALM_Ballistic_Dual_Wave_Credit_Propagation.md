# Research 554: PC-ALM — Ballistic Dual-Wave Credit Propagation (Augmented Lagrangian Predictive Coding)

> **Status:** Active — **Gain/GOAT** (modelless wave-mode dual primitive → [Issue 775](../HISTORY.md)) + training track ([riir-train Plan 401](../../riir-train/.plans/401_pcalm_nextlat_local_training_ab.md)) + healer consumer ([riir-clippy Issue 105](../../riir-refine/HISTORY.md)) + game consumer (`riir-ai Issue 952`). **NOT Super-GOAT — Q3 fails** (§3.1). Not a PASS — new-to-stack mechanism with closed-form laws (§2).
> **Source:** Seely & Gould — "Augmented Lagrangian Predictive Coding" (Sakana AI), [arXiv:2605.31022](https://arxiv.org/abs/2605.31022), 2026-05-29. Code: github.com/SakanaAI/pc-alm @ `660747f61a8a7e547c0ecd2c48c8883380a7d1f6` (MIT, JAX).
> **Date:** 2026-09-14
> **Related Research:** **438 (Sheaf-ADMM — the SAME FIRST AUTHOR'S SIBLING PAPER**, arXiv:2605.31005 → 31022, 17 numbers apart, both Sakana May-2026; R438's substrate is what this note diffs against)**, 219 (DEC substrate), 296 (Stokes vocabulary crosswalk), 354 (set attention — crowd single-state prior art)
> **Related Plans:** katgpt-rs 407 (sheaf_admm — shipped), riir-train 401 (PC-ALM NextLat A/B — this note's training track)
> **Classification:** Public (math + open-primitive direction). Selling-point wiring stays private (riir-ai 952).

---

## TL;DR

PC-ALM trains deep nets without a global backward pass by giving every layer a **dual variable** λ_i that integrates the local constraint residual (`λ_i ← λ_i + α·r_i`, one dual step interleaved per primal step). At the KKT point the activations return to forward-pass values and **λ converges to the exact backprop adjoints** — via layer-local updates only. The headline for us is NOT the training result: it is the **dynamics theory our shipped `sheaf_admm` (R438/Plan 407) lacks** — penalty-only relaxation (our diffusion-mode z-update, our `heat_kernel`, our leaky `evolve_belief`) propagates credit **diffusively** (heat equation, O(√T) reach), while the interleaved primal-dual pair propagates **ballistically** (damped wave / telegraph equation, dispersion `μ±(k) ≈ exp(−½ρη_h k² ± i√(αη_h)k)`, **group velocity √(αη_h)**, O(T) reach, BP-alignment snaps at `t_infl ≈ L/√(αη_h)`), with a per-mode **Jury stability bound** `η_h σ²(2ρ+α) < 4` and an eigenvalue **annulus law** (α = pure phase knob; damping set by η_hρσ²) that turn "add an integrator" from a tuning hazard into a certified construction.

**Distilled for katgpt-rs (modelless, inference-time):** a `dual` accumulator module + wave-mode interleave on chain/cochain constraint operators (the ballistic twin of `heat_kernel` / `sheaf_admm`'s diffusion mode), closed-form safe-rate setters (`jury_eta_max`, `jury_alpha_max`), arrival-time budget laws (`t_infl`, `T = 2L`, `α_reach ≈ 4(L/T)²`), and the **exact-adjoint sensitivity readout** (duals at equilibrium = reverse-mode derivatives of a terminal objective w.r.t. every node of a frozen DAG — "backprop without backprop", freeze/thaw-compatible, no autodiff). α=0 is bit-identical to the incumbent — the clean feature-gate story.

**Pinned claim (§4 precondition):** *Wave-mode dual interleave (λ ← λ + α·r alternating 1:1 with primal gradient steps) on chain/cochain constraint operators, consuming layer-local residuals, turning diffusive credit propagation ballistic (group velocity √(αη_h), reach O(T) vs O(√T)), distinguished from shipped `sheaf_admm` (R438: diffusion-mode z-update, hand-tuned ρ/η/T, no dynamics laws, no adjoint guarantee) by the closed-form Jury/annulus/arrival laws and the exact-adjoint-at-KKT guarantee.*

---

## 1. Paper Core Findings

### 1.1 The algorithm

Feedforward training written as constrained optimization: `min ½‖y − W_L h_{L−1}‖² s.t. h_i = σ(W_i h_{i−1})`. PC is the quadratic-penalty relaxation (energy `F_PC`); PC-ALM is **finite-inference dual ascent on the augmented Lagrangian** (method of multipliers with a 1-step inner solve):

```
forward init: h_i ← σ(W_i h_{i−1});  λ_i ← 0
repeat T−1 times:                    # T = 2L suffices
  primal:  h_i ← h_i − η_h ∇_{h_i} L_ρ    # layer-local: needs h_{i±1}, λ_{i±1} only
  dual:    λ_i ← λ_i + α · r_i             # r_i = h_i − σ(W_i h_{i−1})
final primal step, then the weight update on L_ρ.
```

α=0 recovers PC exactly; α=ρ with exact inner solve recovers method-of-multipliers. Cost: 2× activation memory (h + λ), "modest additional compute".

### 1.2 The identities (the math, not the loop)

- **Completing the square (Eq 8–9):** `λᵀr + (ρ/2)‖r‖² = (ρ/2)‖h − (σ(Wh) − λ/ρ)‖² − ‖λ‖²/(2ρ)`. The augmented Lagrangian **is** the PC energy with every prediction target shifted by `−λ/ρ`. At convergence **h returns to forward-pass values** — all correction lives in λ (a latent accumulator; the raw plane is untouched — exactly our raw/latent sync discipline).
- **LeCun 1988 (Appendix A):** at any feasible KKT point the multipliers equal the **BP adjoints** (`λ_i = −δ_i`), the weight gradient equals the BP gradient, and the augmentation changes nothing at the endpoint. Extends to **arbitrary feedforward DAGs** (reverse-topological stationarity = reverse-mode accumulation).
- **Composite credit (Eq 10–11):** the layer signal is `e_i = λ_i + ρ·r_i` (integral history + current residual) pulled back through `Wᵀdiag(σ′)` — a fixed linear map on **frozen** weights. No autodiff graph needed.

### 1.3 The dynamics laws (the part we don't ship anywhere)

| Law | Statement | Note |
|---|---|---|
| Linear convergence | spr(M) < 1 ⇒ (h,λ) → KKT point (exact BP adjoints) from any init | Prop 3 |
| Jury/Schur bound | `η_h σ_i²(2ρ + α) < 4` per singular mode of the constraint operator A | Cor 1; α=0 limit recovers PC's `ηρσ² < 2` |
| Annulus law | complex eigenvalues on circles `\|μ±\| = √(1 − η_hρσ_i²)` — **α slides phase only, never magnitude** | C.6: accelerate by pushing η_hρσ²_max to the ceiling, not by α |
| Dispersion | `μ±(k) ≈ exp(−½ρη_h k² ± i√(αη_h)k)` — **damped wave**; PC is `exp(−ρη_h k²)` — **pure heat** | σ(k) ≈ 2sin(k/2): A is literally a first-difference (cochain coboundary) operator + noise |
| Ballistic reach | `R_ALM(T) ≈ T√(αη_h)` vs `R_PC(T) = O(√(ρη_h T))` | Eq 23 |
| Arrival/inflection | `t_infl ≈ L/√(αη_h)`; `α_reach ≈ 4(L/T)²` at η_h=1/λ_max (λ_max≈4 for difference operators) | Eq 24 — self-calibrates: T=2L ⇒ α≈1 |

### 1.4 Results

Matches BP at budget T=2L across the full (N,L) ∈ {8..128}² grid (identity/tanh/ReLU, MNIST/Fashion-MNIST), closing the deep-narrow gap where PC fails; robust across parameterization sweeps; at fixed width an order of magnitude cheaper than width-scaling PC to the same BP-cosine. Limitations (paper's own): linear+bias-free theory only; **MLPs only (attention = future work)**; MNIST-scale, one epoch; equilibrium guarantee only (finite-T misaligns); does not solve weight transport.

---

## 2. Distillation

### 2.1 Path 0 decomposition (training-target math → modelless components)

| Paper component | Ships in stack? | Modelless extraction |
|---|---|---|
| Dual accumulator `λ ← λ + α·r` | **Partial** — `sheaf_admm` dual `u ← u + x − z` (same accumulator shape, R438) | YES: pure arithmetic, O(d), zero-alloc |
| Completing-the-square target shift `ŷ − λ/ρ` | No | YES: one FMA; ships beside the accumulator |
| Exact-adjoint-at-KKT readout | No (R438's u has no sensitivity guarantee) | YES: reverse-mode derivatives on frozen DAGs, no autodiff |
| Jury bound + annulus law | No (benches pin hand-tuned ρ=1.0, η=0.2, T=5) | YES: closed-form rate setters + regime classifier |
| Wave dispersion / ballistic reach / t_infl | No — **the stack's propagation is all diffusive** (`heat_kernel`, `bom/nonlinear_heat_kernel`, sheaf_admm's diffusion z-update, leaky `evolve_belief`) | YES: wave-mode twin kernel on `CochainField` pairs |
| The training loop itself | riir-train (Plan 401) | N/A — training track |

### 2.2 Signal-diff vs shipped cousins (§3.6 discipline — one read each)

- **`katgpt-dec::sheaf_admm_step`** (R438/Plan 407): consumes **primal-vs-consensus disagreement** (`u ← u + x − z`); z-update = **T diffusion steps** (heat mode). Diff vs PC-ALM: **interleaved 1:1 primal-dual = wave mode**; no Jury/annulus/arrival laws (constants hand-tuned); no adjoint guarantee. Same-lab sibling paper — the authors themselves shipped the graph-consensus instance (31005) then the deep-chain dynamics theory (31022). **The upgrade is mechanism-level, not a rename.**
- **`heat_kernel.rs`** (+`bom_`, `nonlinear_` variants): computes diffusion (heat) trajectories via precomputed Laplacian spectra. The **wave twin is absent** — the family is entirely parabolic; PC-ALM supplies the hyperbolic member.
- **`katgpt-sense evolve_belief`**: leaky integrator over `[f32;8]` = **first-order diffusive** evidence accumulation (transients decay; nothing propagates). No dual state, no phase, no steady-state-elimination guarantee.
- **`delta_mem mean_prediction_error`**: scalar **mean over history** — aggregated retrospective statistic; PC-ALM's λ is a **direction-shaped accumulator that feeds back into targets** (closed loop, not a readout).
- **riir-clippy `select_best_candidate` / `EvolveRecorder` / fixseq ring**: selection consumes **terminal outcomes** (`W_EVO·evolution + W_RATE·reliability`); the ring already records per-attempt `errors [b,a]` — the raw material for an **integral-of-journey** credit `λ_T = α·Σ_t r_t` exists, but nothing accumulates it. This is the documented strand-keep poison (Bench 089/090): a KEEP ending at a masked E0614 scores as success — terminal residual ≈ 0 while the integral is large. → [riir-clippy Issue 105](../../riir-refine/HISTORY.md).
- **Homeostasis / MAPE-K / re-estimation loops**: proportional-style feedback with hand-tuned rates; PC-ALM supplies the loop-gain ceiling (`η ≤ 4/(σ̂²(2ρ+α))` on measured signal power) and the endpoint-invariance property (retune transients without moving setpoints).

### 2.3 The modelless extractions (→ Issue 775)

1. **`dual` module** (katgpt-core): accumulator, target-shift, composite credit, Jury setters, regime classifier. α=0 = incumbent bit-identical.
2. **Wave-mode kernel** (katgpt-dec, beside `heat_kernel`): joint (h, λ) recurrence on cochain pairs; per-mode eigenvalues = the annulus. GOAT gate = **reach bench**: injected anomaly at one end of an L-cell chain, ticks-to-detectable at the far end — wave `≈ L/√(αη_h)` vs heat `≈ L²/(ρη_h)` (paper Eq 23), on `CellComplex` path graphs the benches already build.
3. **Exact-adjoint sensitivity service**: duals at equilibrium = dJ/d(node) for a frozen layered DAG. Gate: cosine ≥ 0.9 vs explicit reverse-mode at T=2L on a toy frozen MLP (the paper's Fig 5 diagnostic, replicated modellessly).

### 2.4 Fusion (what none of the parts does alone)

- **Wave kernel × sheaf_admm (R438)**: the consensus z-update's diffusion mode becomes wave-mode — tier-chain alerts at O(T) reach with Jury-certified rates. `λ_max ≈ 4` for path/difference operators ⇒ the paper's default constants (η_h=1/λ_max, ρ=1, α∈(0,2)) transfer nearly verbatim.
- **Belief duals × emotion bridge**: `desperation` (one of the 5 synced affect scalars) is semantically **integrated unmet-need error** — the λ accumulator gives it a principled formulation with stability guarantees (sigmoid-bounded projection of λ, never softmax; only the scalar crosses the sync boundary). Fusion idea, novelty TBD → riir-ai 952.
- **Fix-trajectory credit × self_evolve**: composite `sigmoid(λ_T + ρ·r_T)` as a third evidence axis orthogonal to evolution/reliability — the integral-vs-terminal distinction is exactly the strand-keep repair. Measurable on the recorded Bench 089/090 populations → riir-clippy 105.
- **Hodge triage (the paper's own discussion, made load-bearing)**: dual ascent absorbs the exact component of a residual; the **harmonic class is the part no local dual can ever fix** (no local cause) — `hodge_decompose(r)` = exact (keep iterating) ⊕ harmonic (escalate/redesign) ⊕ coexact (circulating credit — detect and break). A repair-path classifier for constraint violations, on operators that already ship. d≤3 caveat respected (zone graphs, belief regions — not high-dim shards).

---

## 3. Verdict

**One verdict per track (TTPO lesson).**

### 3.1 Modelless track: **GOAT** (not Super-GOAT)

| Novelty gate | Verdict | Evidence |
|---|---|---|
| Q1 no prior art | **YES** | External: ballistic-wave credit propagation NOT covered (§4b; one verbatim query never returned — caveat stands). Internal: shipped substrate is entirely diffusive-mode (§2.2). Multi-agent dual-accumulator angle **is covered — by our own R438** (same-lab sibling); the claim lives in the dynamics laws, not the accumulator pattern. |
| Q2 new behavior class | **YES** | O(T) vs O(√T) reach through chains; **stable oscillatory transients** (annulus) as a feature; phase control via α; exact-adjoint readout without autodiff. |
| Q3 product selling point | **NO** | The honest arithmetic: game zone hierarchy is L≈4 tiers — wave reach 6 ticks vs diffusive ~32 ticks @20 Hz = 0.3 s vs 1.6 s. A real but modest feel improvement, not a "no competitor can" claim. The paper's depth regime (L=64–128) exists in the stack only as *reasoning/fix chains*, not as the world hierarchy. |
| Q4 force multiplier | **YES** | sheaf_admm + heat_kernel + hodge + belief kernels + fixseq ring + emotion bridge (≥2 pillars). |

3/4 ⇒ **GOAT**: feature flag + reach/alignment benches before any promotion (Issue 775). "Candidate" escape not used — the gate table is the verdict.

### 3.2 Training track: **Gain → riir-train Plan 401** (Path 0.5 applicable)

Scale-matched target found: the **NextLat belief MLP** (64-dim, unrolled to L_eff ≈ 3R constraint nodes at R=8–16 = the paper's deep-narrow regime) with an **in-tree analytic BP reference** (self-validating cosine gate) and a defined modelless baseline (`TernaryDraftModel`). ~2–6 h. Secondary/backlog candidates (μPC recipe-only extraction, edge_lora per-vertex duals, maglev FFN-block PoC, Kimi-K3 backward-kernel eliminator) + honest exclusions (27B SFT/GRPO/DPO/attention-distill: outside the paper's mechanism class; quest_grammar: shallow-wide, gap doesn't bind) recorded in Plan 401.

### 3.3 Consumer tracks (fusion ladder priorities #1/#2)

- **Healer (#2)**: riir-clippy Issue 105 — integral-vs-terminal credit on fix trajectories; measurable on existing fixtures.
- **Game (#1)**: riir-ai Issue 952 — ballistic tier alerts, belief duals, desperation-as-λ. Fusion ideas, novelty TBD (Q3 unproven at game-surface depth).

### 3.4 MOAT gate

katgpt-dec/katgpt-core: generic math, no game semantics — correct (public). riir-ai 952: runtime wiring — correct. riir-clippy 105: healer-consumed selection change — correct. riir-train 401: training-method plan — correct (active moat).

---

## 4. Prior art (adversarial search, 16 queries — verdicts)

| Claim | Verdict | Blocking prior art |
|---|---|---|
| (a) layer-local multiplier accumulation → exact BP adjoints at equilibrium | **PARTIALLY COVERED** (result old, mechanism new) | LeCun 1988 "A Theoretical Framework for Back-Propagation" (BP = multiplier solution); Whittington & Bogacz 2017; Z-IL ("Predictive Coding Can Do Exact Backpropagation…", 2020); FPA-PC ("On the relationship between predictive coding and backpropagation", 2022) |
| **(b) ballistic O(T) vs diffusive O(√T) credit propagation, group velocity √(αη)** | **NOT COVERED** — the claimable novelty | Nearest misses: "Sample as you Infer" (ICML 2024 — momentum for inference convergence, not spatial); Faye et al. 2023 (waves in neural fields, not credit); "Error Highways" (arXiv:2606.22744 — architectural, post-dates PC-ALM); wave-like dopamine (Curr Bio 2021 — RL neuromodulation) |
| (c) dual accumulators aligning local updates to global objective in multi-agent/distributed systems | **COVERED** | **Sheaf-ADMM (same lab, arXiv:2605.31005 — already distilled as R438/Plan 407)**; Hansen & Ghrist IEEE TCNS 2019; "Distributed Multi-agent Coordination over Cellular Sheaves" (arXiv:2504.02049); PDMM/Prox-PDA/DC-ADMM canon |
| (d) multiplier = integral control eliminating steady-state error | **COVERED** | "On PI Controllers for Updating Lagrange Multipliers in Constrained Optimization" (ν-PI, 2024); dual-PID regret analysis (arXiv:2202.06152); Recht "Integral Action" |

Corollary: any fusion pitch must live in the **wave/dynamics substrate + local-learning-rule instantiation**, never in "dual variables align agents" — that sentence is R438's, and before it the consensus-ADMM canon's.

---

## 5. Public vs private

| Piece | Location |
|---|---|
| Dual accumulator, target-shift, Jury/annulus/arrival laws, wave kernel, adjoint readout | **katgpt-rs** (katgpt-core/katgpt-dec) — public math |
| Hodge triage of residuals (exact/harmonic/coexact repair classes) | **katgpt-rs** (katgpt-dec operators) — public |
| Zone-tier alert wiring, belief duals, desperation-as-λ, per-NPC credit | **riir-ai** (952) — private |
| Fix-trajectory integral credit, self_evolve third axis | **riir-clippy** (105) — private |
| NextLat/maglev/kimi PC-ALM training recipes | **riir-train** (401) — private |

## Caveats (honest)

- Q3 fail is the verdict's load-bearing fact: at game-hierarchy depth (L≈4) the absolute latency win is ~1.3 s; the paper's regime needs deep chains — which the stack has in *reasoning/fix sequences*, not world tiers. If Issue 952's POC shows a crowd-feel win, re-gate.
- The (b) novelty verdict carries a tool caveat: one verbatim query ("wave versus diffusion error propagation neural networks") never returned; substitutes did. A Scholar sweep for "second-order predictive coding"/"hyperbolic PC" before publishing any external claim.
- Training track inherits the paper's own gaps: MLP-only, MNIST-scale, one epoch, attention uncovered. Plan 401 gates on our own A/B, not the paper's numbers.
- Oscillatory transients are stable only inside the Jury bound — any consumer MUST clamp via the closed-form setters (that is precisely what the laws buy).
- λ doubles per-entity state for every tracked quantity; at d=8 that is 32 B/NPC — fine at MMORPG scale, stated so nobody rediscovers it.

## Landing (Issue 775, 2026-09-14)

Issue 775 CLOSED — the modelless extraction landed behind the opt-in
`dual_wave` flag in BOTH crates (`katgpt-core::dual` T1–T4+T8,
`katgpt-dec::wave_kernel` T5+T9), GOAT ALL PASS ([Bench
763](../.benchmarks/763_dual_wave_goat.md)). Landing commit:
`3a59abe1` (docs close-out in the follow-up commit). Measured on the
way: the per-layer-rate blowup (the Jury bound is per singular mode of the
STACKED operator), the Gershgorin overshrink (~2× on η), the ‖AᵀAv‖-vs-‖AᵀAv‖²
power-iteration bug, and the finite-T physics (arrival ballistic at 2L;
settled-readout settling low-mode-limited ~L² — the paper's own limitation,
made precise). Training-track follow-up unchanged (riir-train Plan 401);
consumer tracks unchanged (riir-clippy 105, riir-ai 952).

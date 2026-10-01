# Research 596: Simplex Diffusion Models (SDMs)

> **Source:** "Simplex Diffusion Models" — [arXiv:2609.35553](https://arxiv.org/abs/2609.35553) — Deschenaux, Galashov, Campbell, Wenliang, Thornton, Doucet, De Bortoli (Google DeepMind / EPFL / UCL Gatsby), 2026-09-28
> **Date:** 2026-09-30 (rev 2 — verdict-review corrections applied; rev 1 carried two false math claims in the thinning extraction and a false "no diffusion substrate" discard)
> **Status:** DISTILLED — issue filed (katgpt-rs Issue 912); training-track pilots withdrawn/dropped on review (see §3c); owner gates execution
> **Related Research:** katgpt-core `perturbation_ensemble` (arXiv:2609.33803 DRM decision-layer lineage); riir-train `.research/462_DRM_Diffusion_Reward_Models.md` (the sibling paper that already spawned Plan 429); riir-reflex Issue 055 / Bench 092 (the perturbation-UQ negative that bounds one consumer class)
> **Related Plans:** riir-train Plan 357 (ActFlow on the dllm/dflash drafter lane); riir-train Plan 429 (diffusion distributional critic for tetris — the SDM fusion row's host); riir-reflex Bench 091 (corpus synthesis V5 PASS)
> **Classification:** Public

---

## TL;DR

SDMs lift discrete diffusion to the probability simplex so the generative state is a **belief distribution over categories carried across denoising steps**, instead of a sampled token (whose draw discards the denoiser's uncertainty — "information collapse"). The inference surface is unusually modelless-friendly: closed-form Beta/Dirichlet draws, **no ODE integration** (unlike Dirichlet Flow Matching). For our stack: (1) a **substrate repair with a discovered GOAT-premise defect** — the shipped `sample_dirichlet_into` ignores its α, and `typical_set.rs`'s G3 test passes α=0.05 "for very peaked transitions" that never happen (flat Dir(1) runs instead); (2) **exact concentration-parameterized belief sampling** `Dir(c·p)` — mean-exact by construction — as the exploration dial ("return the mean" ↔ "sample a hypothesis"); (3) a **Dirichlet-randomized schedule-weighted memory over prediction streams** whose expected weights are provably ε-independent (an EMA under a geometric schedule), with ε controlling memory determinism. Training-side: the workspace ALREADY has a diffusion lane (riir-infer dllm/D2F, riir-train Plans 357/429) — SDM is recorded as an architecture **candidate** there, not a redirect target.

**Distilled for katgpt-rs (modelless, inference-time):** `Dir(c·p)` exact belief sampling, thinning as a Dirichlet-to-Dirichlet transition (NOT mean-preserving on fixed vectors — see §2 row 2), the ε-orthogonal memory identity (Prop E.1/Cor E.1: E[L_j] deterministic, Var = w(1−w)/(εC+1)), the temperature-limit theorem, the churn-knobbed closed-form reverse step, the Gaussian-fluctuation covariance bound (spectrum ∈ [min π_i, 2] ∀t).

---

## 1. Paper Core Findings

1. **Simplex belief state**: P_t ~ Dir(β_t(P_0,π)), β_t = c_t(α_t P_0 + (1−α_t)π); mean ᾱ_t, Cov = (diag(ᾱ)−ᾱᾱᵀ)/(c_t+1). Mean and concentration set **separately** (c_t = inverse temperature).
2. **Dirichlet thinning** (Prop A.2; classical, paper disclaims originality): X ~ Dir(α), B_i ~ Beta(ρα_i,(1−ρ)α_i) ⇒ Y_i = B_iX_i/ΣB_jX_j ~ Dir(ρα). Exact **only under Dirichlet input**. (Round-1 correction: thinning a FIXED vector p is biased toward uniform — E[B_i·p_i/ΣB_jp_j] ≠ p_i, simulated 0.9/0.07/0.03 → 0.847/0.103/0.051 at c=2, ρ=0.5; and the per-coordinate Var ratio is (c+1)/(ρc+1), not ρ.)
3. **Closed-form reverse bridge** (Prop 3.3): thin → innovate (Dir(β_s−ρβ_t)) → Beta-mix; κ = churn ∈[0,1] (trust in current belief ↔ independent resampling); no ODE — beats their DFM re-implementation (Sudoku 88.4/99.1 vs 76.7; TinyGSM 45.8 vs 6.1). κ=1 empirically best nearly everywhere (12.6→45.8 TinyGSM @512/T=1).
4. **Temperature limits** (Prop 4.1): ε→0 ⇒ vertex collapse = discrete diffusion exactly; ε→∞ ⇒ deterministic interpolation with Gaussian fluctuations (covariance V_t = V_π + α_t(P_0−π)(P_0−π)ᵀ, projected spectrum ∈ [min π_i, 2] ∀t).
5. **Belief-memory identity** (Prop E.1 + Cor E.1): under exact DDIM, P_{t_i} = Σ_j L_{i,j} e_{x̂_j} + L_{i,M} P_{t_M} — a Dirichlet-weighted combination of ALL past posterior draws; **E[L_{i,j}] = c̃_j/C_i deterministic and ε-independent** (a schedule-weighted average; an EMA under a geometric schedule); Var → 0 as ε→∞ (deterministic weights) / → single-categorical-draw as ε→0. Intrinsic self-conditioning without training.
6. **Normalized variance** ν = 1/(c+1) ∈ (0,1) — bounded designer-facing knob.
7. **Training**: plain cross-entropy (= −ELBO/M); adaptive time sampler: density ∝ dL/dt from a ring-buffer bucketed piecewise-linear fit + EMA 0.9 + inverse-CDF; the same CDF reused as the inference grid (+9–11 pts @512, +18–19 @64 on TinyGSM).
8. **Distillation** (Simplex DMD): exact-gradient DM distillation through the Gamma reparameterization (Dirichlet Tweedie, Lemma G.2); **8 NFE ⇒ 32.1% GSM8K vs IDLM-128-step 21.4%** — the few-step regime win.
9. **Auto-guidance**: ζ_final + w(ζ_final − ζ_earlyK), best K ∈ [20k,50k] of 250k — logit arithmetic over two checkpoints, no dropout.
10. **Systems**: fast Marsaglia–Tsang Gamma (small-α boost); counter-based PRNG vectorization (3.8× vs key-splitting).
11. **Evidence scale**: ≤168M params; paper flags scale as open. OWT: all families tie after logit-shaping; AR frontier beats real data (GenPPL-frontier critique).

## 2. Distillation — Path 0 inventory (coverage / extraction, two questions per component)

| # | Component | Modelless? | Ships today (coverage — verified) | Extraction |
|---|---|---|---|---|
| 1 | Simplex belief (mean+concentration stats) | YES | Partial — `ReconstructionState` belief = mean point-estimate (leaky-step); `MultiHypothesisBoMMinimaxPlanner` K-hypothesis planning via Gaussian/QMC noise | `SimplexBelief { mean, c }` — sample only to act, never to know |
| 2 | **Exact belief sampling Dir(c·p)** (the corrected explore dial) | YES | No — nothing; cousin `perturbation_ensemble::bucket_dropout_into` = input-hash Bernoulli dropout (different signal; AND reflex Bench 092 killed its confidence use) | `sample_conc_into(p, c)` — mean exact by construction |
| 2b | Dirichlet thinning (transition op) | YES | No | `thinning_into` — exact ONLY Dir→Dir; documented NOT mean-preserving on fixed vectors |
| 3 | Interpolation path (Beta blend to prior) | YES | Scalar analog: `GenericSpatialBelief` confidence decay | Distributional "forget toward prior" |
| 4 | Churn-knobbed reverse step | YES | No | Belief-revision operator for refine loops |
| 5 | Temperature limits (commitment dial) | YES (theorem) | Gumbel-τ-class knobs; no proven endpoint identities | Endpoint pins make it designer-safe |
| 6 | **Dirichlet-randomized memory** over prediction streams | YES — headline | No — `leaky_step`/`evolve_belief_additive` are deterministic EMAs; no randomized memory, no diversity readout | `DirichletEma<M>`: geometric-schedule EMA (recursive mean path, bit-pinnable) + ε-randomized weights; truncation named |
| 7 | Gaussian covariance + spectrum bound | YES | `katgpt-dec` has no Dirichlet-covariance crossing | Error-budget generator; certified margins |
| 8 | ν = 1/(c+1) | YES (bijection) | House sigmoid-reparam idiom | API surface for the family |
| 9 | Auto-guidance (snapshot-pair logits) | YES — two frozen artifacts | No — freeze/thaw swaps, never extrapolates | `guided_logits`; corpus-snapshot guidance fusion idea |
| 10 | Logit shaping (freq penalty etc.) | YES | No frequency/repetition penalty in katgpt-core `sampling.rs` | Commodity; local belief-coupled weight is the distinctive bit |
| 11 | Dirichlet Tweedie | YES (identity) | No consumer | Waits for a Dirichlet-mixture observation model |
| 12 | General Gamma/Dirichlet sampler | YES | **DEFECT — REPAIRED same session** (Issue 912 T1 landed): `sample_dirichlet_into` ignored `_alpha` (flat only) while `typical_set.rs:198` G3 test passed α=0.05 "very peaked" — the premise was silently not in force. Fix honors α (Marsaglia–Tsang), keeps the α=1 exponential path verbatim (bit-identical for existing callers); goat_g3/goat_g5 re-verified on truly peaked chains, 26/26 ×3 release runs, full lib 2064/0 | **Landed**; the general non-uniform-α sampler in new `dirichlet_dist.rs` remains T2 (unscheduled) |

## 3. Verdict — per track

### Track (a) modelless inference — **GAIN** (katgpt-rs Issue 912, revised)

§1.5 score: **Q1 NO** (math is classical — paper disclaims thinning originality; simplex belief = POMDP belief + conjugate Dirichlet; randomized memory = Bayesian bootstrap with schedule prior; auto-guidance = Karras 2024; §4 searches confirm). Q2 partial, Q3 not crisp, Q4 moderate → Gain, not Super-GOAT.

Actionability (reverse-grep evidence): the α-unused defect **with a GOAT test running on its false premise** (found by the review); the riir-neuron-db documented gap ("uniform averaging has no attention-shift mechanism by design"); existing exact-law-noise consumers (`perturbation_ensemble`, `bom_arena`). Signal-diffs: `bucket_dropout_into` = input-hash dropout vs output-distribution exact law (and Bench 092 kills the confidence use — exploration only); `evolve_belief_additive` IS the ε=∞ limit (the safety case, not a kill); Raven/δ-Mem uniform merge vs mean-exact randomized attention.

### Track (c) training — **RECORD + one fusion row** (no new plans/issues; rev-1 pilots withdrawn & dropped on review)

**The workspace HAS diffusion substrate** (rev-1 discard reason was factually wrong): `riir-infer/src/transformer/dllm.rs` (D2F teacher/student forwards, `masked_cross_entropy`, `set_diffusion` feature), `riir-infer-gpu/gemma2_d2f/` (GPU denoising decode loop), riir-train Plan 357 (ActFlow on dllm/dflash: sampler + GP + WDCE trainer DONE, T1.4 gate blocked on a pre-trained base), riir-train Plan 429 (pre-registered diffusion distributional critic for tetris, from the sibling DRM paper arXiv:2609.33803). Mechanism-level discard of *wholesale SDM adoption*: the D2F denoiser consumes mask tokens; SDM's state is the expectation embedding of a simplex belief — adopting it means retraining the drafter under a different input representation, and Plan 357's lane is currently blocked upstream (no pre-trained base). **Recorded candidates**: (i) SDM's CE-only training + closed-form no-ODE sampler as a candidate architecture if Plan 357 T1.4 unblocks; (ii) the §5 Plan-429 fusion row.

**Withdrawn: rev-1 pilot B (tetris loss-density curriculum).** Three independent kills: (a) *mathematical* — sampling batches ∝ w(x) equals weighting the loss by w(x) in expectation, i.e. round 2's tail-weighted objective that went backwards (Bench 017); "untried axis" was false. (b) *lane discipline* — the lane's re-open bar excludes same-class row reshaping (and Plan 429 calls itself a round-5 lever — see the §5 row for the owner-call framing). (c) stale facts (5 seed reads not 4; round-4 critic G2 breached at 3.4–3.8 ms — 0.228 ms was round 1's).

**Dropped: rev-1 pilot A (corpus-synthesis loss-density allocation).** The host lane is modelless (Bench 091: "ZERO trained weights") — no student epoch losses exist to fit; V5 already PASSED (0.7800→0.8133, LB95 +0.0110) under E0-rumor allocation; the transfer collapses to "allocate ∝ per-label error" over 59 unordered labels (the piecewise-linear fit on an ordered axis doesn't survive); the real cost is openthai veto hours (~4 h teacher wall per 2048 candidates). Residual note-row for the reflex lane, attribution dropped: *per-label error-weighted vs E0-rumor allocation, priced in veto hours* — thin marginal value since E0-rumor already approximates it.

## 4. Prior art (§4 searches — run before verdict)

- **"Dirichlet thinning"**: arXiv:2506.18223 is a *different* thinning (zeroing stick-breaking atoms). Beta-multiplicative Dir→Dir thinning is textbook neutrality folklore; the paper disclaims originality. Q1 stands.
- **Stochastic/Dirichlet EMA**: nothing direct; bounded by Rubin's Bayesian bootstrap (weights ~ Dirichlet over the sample). The ε-orthogonality theorem is this paper's.
- Simplex-diffusion lineage (paper's related work): Richemond 2022, Stark 2024 DFM (closest; ODE-sampled), Simplax (concurrent; categorical-sample states), Boget & Kalousis 2026, Chandra 2026. No novelty claimed by us on the family.
- POMDP belief tracking (decades): bounds Q1 for belief states; the paper's "information collapse" is the exact-belief-vs-sampling point restated for diffusion samplers.

## 5. Fusion ideas (recorded; owner-gated)

- **Plan 429 × SDM (evidence-weak, owner-gated)**: SDM as the sampler architecture for the tetris distributional critic. **Prerequisite stated**: Plan 429 models p(v|s,a) over a **continuous scalar value (K=1)** with seeded DDIM — it is not masked resampling, and SDM runs on a simplex, which needs categories. SDM applies only if the value head is first **discretized into bins** (a C51-style categorical support) — a separate design choice. The 8-NFE LM win (32.1% vs 21.4%) is weak evidence for a 1-D value head at S ∈ {4,6}. Plan 429 calls itself a round-5 lever; whether it clears riir-instinct Issue 009's re-open bar (whose closure names the blend A/B, not Plan 429) is the owner's call. A candidate input to Plan 429's Phase-1 architecture choice — the plan is pre-registered, so this row does not edit it.
- **D1 — Uncertainty-carrying ReconstructionState** (rows 1/2/5/6): mean-path bit-identical at defaults; diversity readout for curiosity/limelight; per-NPC temperament via ε.
- **D4 — DEC × Dirichlet covariance**: variance-weighted belief-mass conservation on zone cochains via `katgpt-dec::codifferential`, spectrum bound as certified error budget (d≤3 regions only — curse-of-dim law).
- **Corpus-snapshot guidance** (row 9 over freeze/thaw): suggestions guided by (current − older) frozen-corpus difference.

## 6. Routing + files

| Item | Repo | File |
|---|---|---|
| Sampler fix + exact belief sampling + thinning + Dirichlet-EMA (feature-gated primitives; T1 ungated fix) | katgpt-rs | `.issues/912_dirichlet_dist_primitives.md` |
| ~~Pilot A corpus synthesis~~ DROPPED (modelless host; V5 passed; note-row only, §3c) | — | — |
| ~~Pilot B tetris curriculum~~ WITHDRAWN (see §3c three kills) | — | — |

MOAT: generic probability → katgpt-core public leaf (`dirichlet.rs` taken by Dirichlet-energy — new module `dirichlet_dist.rs`). Consumers adopt in their own repos. **UQ floor note:** the primitives claim no coverage/prediction interval — G1 is exact-law pinning; the conformal floor binds future consumers that claim calibrated uncertainty.

## 7. Caveats + lesson

- Small-model evidence (≤168M); scale flagged open by the authors.
- Bench 092 bounds thinning/belief-sampling to exploration uses, not confidence ranking.
- κ=1 (near-full resampling) best in the paper is mildly *against* the belief-carrying story at high churn; extract the operators, don't over-claim the philosophy.
- **Pre-flight lesson (the rev-1 miss)**: the skill's suggested training-code grep patterns (`*_training.rs|LoRA|SFT|GRPO|DPO|ternary|self_evolve`) match NOTHING in the dllm/dflash/diffusion lane — the workspace's diffusion substrate ships under domain vocabulary none of those patterns reach. Lesson: pre-flight #5 must also grep the PAPER's domain words (diffusion/denois/masked-CE/dllm), not only the skill's generic training vocabulary. This is the arXiv:2511.18538 canonical failure reproduced with a different vocabulary set.
- **Verdict-review lesson**: round 1 carried two false exact-law claims (thinning mean-preservation on fixed vectors; Var ratio ≡ ρ) that a 400k-draw simulation refuted in minutes. Exact-law claims need the simulation BEFORE the issue text, not after the review.

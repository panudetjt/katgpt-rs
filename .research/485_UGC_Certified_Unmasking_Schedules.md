# Research 485: UGC — Certified-Optimal Unmasking Schedules for Masked Diffusion

> **Source:** "The data geometry of masking diffusion: Certified-optimal schedules via unmasking growth complexity" — Martin J. Wainwright (MIT LIDS / EECS / Math), [arXiv:2608.13520](https://arxiv.org/abs/2608.13520), 2026-08-13. Companion (Gaussian diffusion): "Denoising growth complexity" [arXiv:2607.26285](https://arxiv.org/abs/2607.26285).
> **Date:** 2026-08-16
> **Status:** CLOSED 2026-08-17 (commit `3bb115da`) — Issue 664 executed: G1/G1-cert/G4 PASS (paper numbers reproduced to 3 digits; coverage 32/32; zero-alloc), **G1b FAIL (honest negative result — the confidence-threshold d2f loop offers the certified schedule nothing to reclaim; early-exit already adapts passes; certificate N undefined at measured ε≈0)**. Substrate landed always-on as `katgpt-core::ugc_schedule` (diagnostic-only, no feature flag). Record: [Bench 659](../.benchmarks/659_ugc_certified_schedule_poc.md). Re-open trigger: a d2f random-order-reveal variant.
> **Related Research:** 034 (D2F — shipped decode), 430 (DiffusionBlocks — `EquiProbability` clock absorbed), 383 (Latent Forcing scheduling — riir-train), 271 (diffusion vocabulary crosswalk), 072 (DMax SPD), 316 (DSpark), 428 (PFlash budget), 119 (KPop KL masking — no-gain precedent)
> **Related Plans:** none — the Issue 664 G1b gate FAILED (Bench 659, 2026-08-17); no feature-flag plan opens. The `ugc_schedule` substrate landed always-on as diagnostic-only.
> **Classification:** Public

---

## TL;DR

The paper supplies the missing **cost/quality theory for masked-diffusion decode**: a path-resolved information measure — the **unmasking growth complexity (UGC)** — whose local increments bound the KL error of each unmasking step; the natural sampling clock is **log-reveal-odds** λ = log(t/(1−t)); optimal schedules place **equal √q-mass per step**; and — the headline — the schedule-governing increments are **estimable from samples** via KL increments along coupled reveal trajectories (Monte Carlo over the denoiser's own posteriors + truncated empirical Bernstein), yielding **certified-optimal samplers**: a prescribed KL error ε achieved with high probability, iteration complexity within a constant factor of the oracle schedule. **No gradient step anywhere — modelless-validable end to end.**

For this stack: both live D2F decode loops (`riir-gpu/gemma2_d2f` GPU + `katgpt-forward/d2f` CPU) run **hardcoded step presets (8/12/4) with confidence-threshold reveal and zero quality certificate**. This paper provides (a) a principled, data-geometry-adaptive step scheduler, and (b) the first *measurable* G1-style KL quality gate for the decode path.

**Distilled for katgpt-rs (modelless, inference-time):**
1. **UGC increment estimator** — forced-mask coupled trajectories, reveal-odds-dyadic grid, factor-2 sandwich, truncated empirical-Bernstein tail control.
2. **Log-reveal-odds clock** — one `logit()` transform away from the shipped `ScheduleKind::LogitNormal`.
3. **Equal-√q-mass grid + K-block geometric multipliers + DP block boundaries** — closed-form construction, pure math.
4. **Certificate statistic** — `KL ≤ 4Ĉ/N` w.p. ≥ 1−η, with explicit confidence radii.
5. **Exact toy test vectors** — closed-form densities for repeated-bit/parity (paper Eq 24a/24b) + published Ratio values for mixture ensembles.

---

## 1. Paper Core Findings

Setup: sample Z ∈ A^d via the **reveal process** X_t (each coordinate independently revealed by time t). Denoiser = single-site posteriors μ_i(·, x). Two samplers: **Bernoulli** (random-cardinality subset per step) and **fixed-cardinality** (uniform subset of given size) — both **random-order** reveal.

**UGC (the measure).** Bernoulli unmasking gain h(t) = Σ_i Info(Z_i; X_t | i still masked), with h′(t) = −d²/dt² Info(Z; X_t) (information curvature). Path complexity **H(p,q) = ∫_p^q t(1−t)h′(t)dt, additive: H(p,r) = H(p,q) + H(q,r)** — the property that makes decomposition, estimation, and adaptive scheduling possible.

**Theorem 1 (error control).** KL(P_Z ‖ P_Ẑ) ≤ Σ_j (ψ(t_{j+1})/ψ(t_j) − 1)·H(t_j, t_{j+1}) + init + completion, where ψ(t) = t/(1−t) is **reveal odds**. Estimated denoisers add an exact per-site KL term (their Eq 18) — the bound degrades gracefully with denoiser error.

**Log-reveal-odds clock.** λ = log(t/(1−t)); UGC density q(λ) = r²(1−r)²h′(r). Single-block (one geometric multiplier) complexity C_UGC ignores where the mass sits; geometry-aware schemes take fine steps where q is large. Potential gain Ratio = C_UGC / P_UGC ≥ 1 (Cauchy–Schwarz), equality iff q constant.

**Proposition 1 (K-block schedules).** For blocks with log-odds lengths S_k and UGC masses H_k: partition complexity **C(P) = (Σ_k √(S_k·H_k))²**, near-optimal multipliers **ρ_k = min(1, 4·√(C/N)·√(S_k/H_k))** — spend iterations where H_k is large. Block boundaries optimizable by a small DP (their §4.4.2).

**Estimation from samples (Lemma 1 + Prop 2 — the load-bearing novelty).** Along **coupled forced-mask reveal trajectories** (coordinate i held masked, others revealed by shared uniforms), the statistic D(p,q) = (q−p)·Σ_i E[KL(π_{i,q} ‖ π_{i,p})] satisfies **D(p,q)/(q−p) = h(q) − h(p) exactly**, and a factor-2 sandwich 1/c·D ≤ H ≤ (1+c)/c·D with c = odds-ratio − 1 (dyadic odds intervals ⇒ exact factor 2). A truncated estimator + empirical Bernstein gives **H ≤ Ĥ_m + r̂_m ≤ 2(H + r̂_m) w.p. ≥ 1−η** — Monte Carlo over the denoiser's posteriors, **no training**.

**Theorem 2 (certified-optimal sampler).** Data-dependent multipliers ρ̂_k yield KL ≤ 4Ĉ_UGC(P)/N + init + completion w.p. ≥ 1−η; choosing N ≥ 8Ĉ/ε **certifies** the ε. Within constant factor of the DP oracle.

**Theorem 3 (fine-partition limit).** inf over partitions of C(P) = **(∫√q dλ)²**, which is exactly twice the sharp leading-order optimal-Euler KL constant. The optimal grid has **equal √q-mass per step**.

**Gains.** Repeated-bit/parity: Ratio ≍ log d. Discrete mixtures (M = 2^{d/4}): Ratio = e^{Ω(√d)}, and **K = 3 adaptively placed blocks capture most of it** (65.0 → 23.5 vs fine-partition 16.9 at d=64). Random XORSAT (α=0.5): Ω(√d log d) with K = 3.

**Connections.** Aggregate UGC H(0,1) = 2/(d+1)·**TSE complexity** (Tononi–Sporns–Edelman neural complexity); H(0,1) ≤ min(TC, DTC); sandwiched with DHW26's effective total correlation; DHW τ-leaping ≈ Bernoulli unmasking with exact scores (their Appendix C.2).

**Fetch caveat (verified):** the PDF-to-markdown render mangles inline fractions — the prose "H(0,1) = (d−1)(d+1)log 2" for repeated-bit contradicts the paper's own bound (61c); recomputed via the TSE identity: **H(0,1) = ((d−1)/(d+1))·log 2 ≤ min(TC, DTC) = log 2** — consistent. Theorem/Definition formulas are unaffected; treat rendered scalar fractions with care.

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent | Status |
|---|---|---|
| unmasking growth complexity / information profile | — (zero MI machinery repo-wide) | **net-new** |
| log-reveal-odds clock λ = logit(t) | `ScheduleKind::LogitNormal` samples t = sigmoid(N(μ,σ)) — a Gaussian *in* logit(t); the clock is one transform away | 1-step extension |
| reveal-time grid, geometric multipliers | `NoiseSchedule::monotonic_ratios` (training, fixed); RePlaid `AdaptiveNoiseSchedule` (loss-variance-driven, training-side) | inference-side variant net-new |
| equal-√q-mass grid | `ScheduleKind::EquiProbability` (equal-CDF-mass, Research 430) | geometry-adaptive generalization |
| coupled / forced-mask reveal trajectories | `dllm_solver.rs` `TransitionType` + `classify_transitions` (reveal bookkeeping, no KL) | KL-increment layer net-new |
| empirical-Bernstein certificate | — (policy precedent: Report-the-Floor, `.benchmarks/010`) | net-new |
| per-position reveal times | `set_diffusion_schedule.rs` `PositionOffsetSchedule` (inverse-CDF reveal sampling, Research 376) | composes |

### 2.2 Signal-diff on closest cousins (§3.6 discipline)

- **`CriticalIntervalConfig`** (`dllm_solver.rs`): consumes **per-position softmax entropy at the current step** to switch solvers. UGC consumes **KL increments between posteriors at coupled reveal times** (path-resolved conditional-MI curvature). Instantaneous uncertainty vs information *growth* — different signal, not covered.
- **RePlaid `AdaptiveNoiseSchedule`**: consumes **per-block loss variance** (training signal) to equalize training cost. UGC consumes **inference-time posterior-KL increments**. Different track, not covered.
- **FLARE/`SoftmaxArgmax` (Issue 587)**: per-token accept/reject law — the *which-tokens* axis. UGC schedules the *how-many-steps / reveal-fraction* axis. Orthogonal, composable.
- **Confidence-threshold reveal** (both shipped loops): greedy per-token; the paper's theory covers **random-order** Bernoulli/fixed-cardinality subsets. See caveat 1.

### 2.3 Fusion

- **Fusion A (primary): UGC × `CriticalIntervalConfig`.** The UGC density is the *principled* version of entropy-thresholded critical intervals — replace heuristic interval boundaries with estimated UGC-mass boundaries as a new interval source for solver switching.
- **Fusion B: UGC × `ScheduleKind`.** `LogRevealOdds` clock + equal-√q-mass step generation; composes with the DPM-Solver++(2M) log-SNR step ratios already shipped.
- **Fusion C: UGC × `PositionOffsetSchedule`** (`set_diffusion_schedule.rs`): a global adaptive grid over per-position reveal clocks (Research 376 consumers).
- **Fusion D (training crossref):** h(t) profile → RePlaid `noise_min/max_ratio` bounds. riir-train follow-up only if the PoC shows signal.
- **Fusion E (speculative, novelty TBD):** aggregate UGC = TSE/Tononi complexity on NPC crowd joint-action distributions as a principled **crowd-coherence scalar** (latent, per-tick, sigmoid-gated). TSE has a neuroscience literature; the crowd application is a fusion *idea*, not a claim — file separately if pursued.

### 2.4 Game-context reframe

- **Quest-grammar hierarchy ↔ paper's hierarchical mixtures (their Fig 3):** UGC modes = reveal times where successive hierarchy levels resolve. Generating (quest_type → sub_goal → params) has exactly that tree-coupling shape; K=3 blocks concentrate compute at hierarchy transitions. Honest: quest_grammar is constraint-based today, not diffusion — this is the shape match, not a shipped consumer.
- **Certified generation at scale:** "KL-certified within ε of the model distribution" is a quality SLA no incumbent decode path offers — for discrete content blocks (item stats, quest params) sampled at MMO scale.
- **Honest scope:** our D2F decode is experiments-track (`gemma2_d2f` GPU, `katgpt-forward` CPU), block_size 8–16. The Ω(√d) asymptotics need d ≥ 32 with sharp mixture structure. Realistic near-term value: (a) principled step counts vs hardcoded 8/12/4 presets, (b) the first real KL quality gate for decode benches (today: perf gates only), (c) Fusion A.

### 2.5 Modelless Path 0 decomposition

| Paper component | Modelless analog | Verdict |
|---|---|---|
| UGC increment estimation | MC over trained-denoiser posteriors along coupled forced-mask trajectories | exists-as-construction — no GD |
| Log-odds clock + equal-√q grid | closed-form transform + 1-D quantile construction | pure math |
| K-block multipliers + DP boundaries | closed-form + small DP | pure math |
| Bernstein certificate | truncated empirical-Bernstein statistic | pure statistics |

**MODELLESS-VALIDABLE — no riir-train deferral.** (The denoiser is a frozen trained model; the schedule layer never touches weights. Freeze/thaw applies naturally: an estimated schedule is a committed, versioned artifact.)

## 3. Verdict

**Gain (GOAT-track pending PoC).** The certified-schedule machinery ships nowhere in the stack and is modelless-validable end-to-end, but its direct wins are confined to the D2F decode path whose block sizes (8–16) sit far below the asymptotic regime (d ≥ 32, M = 2^{d/4}) where the paper's √d gains materialize — distill the primitive + run the PoC before any feature-flag plan.

**Novelty gate:** Q1 prior art — YES (zero codebase hits on the core machinery; publication novelty of estimation+certification verified against CCL25/LZ25/DHW26 + 18 additional works; no published log-odds reveal clock). Q2 new behavior class — partial (a KL certificate is a capability no incumbent has, but bounded to the decode path). Q3 product selling point — weak-moderate (D2F decode is not yet a product NPC pillar). Q4 force multiplier — YES (ScheduleKind, dllm_solver critical intervals, set_diffusion_schedule, RePlaid crossref, Report-the-Floor discipline). **Not all-4-YES → not Super-GOAT; no guide.**

**MOAT gate:** katgpt-rs in-scope — sampling/schedule substrate for the masked-diffusion stack (the "sampling" slot). Private consumers later via `riir-gpu/gemma2_d2f`.

**Honest caveats (binding on Issue 664):**
1. **Sampler mismatch.** The theory covers random-order Bernoulli/fixed-cardinality reveal; our loops use confidence-threshold (greedy per-token) reveal. The certified bound covers the steps × reveal-fraction axis, NOT confidence ordering. The estimator itself is policy-agnostic (it measures data geometry). The PoC must implement the paper's random-subset variant for the certificate; extending the bound to confidence-ordered reveal is open.
2. **Scale.** Gains demonstrated at d ≥ 32 with sharp structure; block_size 8–16 ⇒ expect modest gains. The PoC gate must test both the paper's ensembles (d 32–128) and our real decode shapes.
3. **UQ rule.** The KL certificate is a sampler-fidelity bound, not a forecasting interval — the conformal floor does not directly apply, but the Report-the-Floor *discipline* does: certificate coverage (the 1−η claim) must be validated empirically on toys, never asserted.
4. **CCL25 impossibility tension.** CCL25 (arXiv:2511.04647) proves you cannot compete with the optimal schedule without a-priori distribution knowledge; UGC sidesteps by competing within *constant factors* with high probability via estimation. Cite both whenever "certified-optimal" is claimed.
5. **Prior-art lineage to cite:** CCL25 (oracle info-profile schedules, TC/DTC bounds), LZ25 (arXiv:2510.25544 — continuum √-profile rule; √-form attribution from UGC's text, not independently verified from LZ25's body), DHW26 (arXiv:2602.15008, COLT 2026 — aggregate effective total correlation), Entropic Time Schedulers (arXiv:2504.13612 — equal-conditional-entropy clock, optimality only *conjectured*; UGC proves the sharper √q form in this setting), EDS (arXiv:2602.06849), JYS (arXiv:2410.07761 — upper-bound search, no certificates), KLASS (arXiv:2511.05664), RADD (arXiv:2406.03736 — scalar time reparam). Game/PCG application of unmasking-schedule theory: **confirmed absent** (searched).
6. **Fetch caveat** (§1): rendered fractions unreliable — re-derive scalars from theorem forms when implementing.

## 4. Validation protocol (EXECUTED by Issue 664 — see [Bench 659](../.benchmarks/659_ugc_certified_schedule_poc.md))

- **G1 exact checks: PASS** — closed-form q_rep(λ)/q_par(λ) (Eq 24a/24b) rel < 1e-9; H(0,1) = ((d−1)/(d+1))·ln2 via both direct integration and the TSE identity; noisy-bit Ratios **4.511/2.165/1.653** vs paper {4.51, 2.16, 1.65}; mixture Ratios **2.230/3.373/4.039** vs {2.19, 3.13, 3.85} (1.8%/7.8%/4.9%).
- **G1 certificate coverage: PASS** — measured KL (exact 3^d enumeration of the sampler output law) ≤ 4Ĉ/N in **32/32 seeds** across both cells.
- **G1b (the falsifiable promotion gate): FAIL — honest negative result.** At block sizes 8–15 on the real decode path: reductions −8.6%…+1.4% (bar ≥20%); the loop's confidence early-exit already adapts passes; ε(N) flat at the noise floor for N ≥ 3 (the certificate's N = 8Ĉ/ε is undefined at measured ε ≈ 0); the one +75% cell is degenerate (N-invariant outputs, N* from the scan not the certificate). Caveat 1 (random-order vs confidence-threshold reveal) is the confirmed structural cause.
- **G4: PASS** — 0 allocations steady state (CountingAllocator).
- **Feature flag:** `ugc_schedule` NOT created (G1b failed). Substrate = always-on diagnostic.

> **PASS-Redirects (synthesis):** Yang et al. [arXiv:2610.02665 "Large Language Continuous Diffusion Models"] — NVIDIA's Sigma (3B/8B continuous dLM) independently converges on this note's schedule axis from the continuous side: their inference-optimal discretization is **uniform in γ = −log SNR** (App. G.6 — it beats learned/EDM-ρ grids at low NFE), the continuous twin of UGC's equal-√q-mass grid in λ = log reveal-odds; two independent papers land on uniform spacing in the log-odds coordinate. Cross-domain validation only — no code action: our lane is discrete-masked (their machinery operates on Gaussian-corrupted embeddings).

# Research 598: PSSA (Plastic State-Space Architecture) — Fusion Verdict

> **Source:** [Sparticle62ops/pssa](https://github.com/Sparticle62ops/pssa) @ `2d5b105a686c9a0cb76320ab41da289e154c3595` (pinned, clone deleted after verdict) — "PSSA: a plastic state-space architecture", **license GPL-3.0**
> **Date:** 2026-10-01
> **Status:** RECORD — **NO-FUSE** (external fusion proposal adjudicated against shipped substrate; no adoption, no mining batch, no lane) — §4 license scope amended 2026-10-01
> **Provenance prompt:** Gemini-generated fusion proposal (4 katgpt-rs ideas + 4 riir-reflex ideas) evaluated on user request; Gemini's claims about OUR stack were independently fact-checked (5 of 6 TRUE — see §5).
> **Classification:** Public

---

## TL;DR

PSSA is a 1.5M-parameter from-scratch Rust research prototype: a Mamba-family selective diagonal SSM + a 512-slot Poincaré episodic memory bank + a rank-16 "plastic" adapter with a fast/slow coefficient split. It beats a self-implemented parameter-matched transformer baseline 3.997 vs 4.429 held-out CE over 12.7M WikiText-103 tokens and generates ~12× faster on CPU. **Every one of the eight proposed fusion points lands on substrate this workspace already ships in equal-or-stronger form** — a 27B GDN+FA hybrid in production league, `poincare_navigator` (default-on), a PKM episodic store with δ-rule write gate, `delta_mem`, closed-form ridge critics, and a mandate-compliant calibrator-only `/feedback`. Separately, PSSA's own README overstates three mechanisms relative to its code (§3). GPL-3.0 limits the implementation path (no vendoring; clean-room only) but is not a verdict ground — §4.

**Distilled for katgpt-rs (modelless, inference-time): nothing above the action threshold.** The three genuinely shipped micro-patterns (state-conditioned retrieval query, refractory write gate, exact-sum-preserving consolidation transfer) are recorded in §6 with re-open triggers; each is a hand-tuned heuristic whose ablation value is unmeasured even inside PSSA.

---

## 1. What PSSA is (verified at the pinned sha)

One layer per token: selective diagonal SSM → episodic memory read → sigmoid gate + rank-16 SiLU adapter → SiLU MLP. Defaults `d_m=256`, `d_s=16`, rank-16, 512-slot bank.

- **Recurrence** (`pssa.rs` L43-85 README; `forward_continuous_inference` L634-737): textbook selective SSM — `delta=softplus(W_δx)`, `A=−softplus(A_raw)` (diagonal, per (channel,state)), ZOH discretization `Ā=exp(δ·A)`, `h←Āh+B̄x`, `y=C·h`. HiPPO-flavored init: log-spaced timescales τ∈[1.5, 200] tokens (`new_with_rng` L505-521). README itself: "claims no novelty … same family as S4 and Mamba".
- **Memory read** (`memory.rs`): query conditioned on BOTH token and recurrent state — `q = W_qx·x + W_qh·y` (L683-688, genuinely shipped), Euclidean→Poincaré diffeomorphic projection (radially saturated, 8-ulp headroom, f64 accumulation), distance `2·asinh(√(s/denom))` (numerically stable acosh form), softmax at temperature `tau_mem`.
- **Write path** (`defense.rs` + `insert_protected`): ring-buffer bank; when full, overwrite gated by a refractory rule — `gain = max((1−exp(−Δt/τ))², 0.005)`, `damage = 0.08·surprise·gain`; if slot confidence > damage → absorb ("Defended"), else overwrite and reset confidence to 1.0. Reinforcement path adds `eta·max(gain, 0.20)` capped at 5.0. Constants τ=30/60 hand-tuned.
- **"Plastic" adapter** (`adapter.rs`): rank-16 `down_proj`/`up_proj` with a THIRD store `consolidated_up` (slow, deliberately not Adam-updated). `consolidate(α)`: `slow += α·fast; fast = (1−α)·fast` — an exact-mass-preserving transfer (their own comment: "preserving their effective sum exactly").
- **Engineering**: hand-written linalg (AVX2+FMA runtime-detected dot, x86_64-only), associative scan over chunks via a persistent rayon worker handoff (`scan_executor.rs` — process-wide worker, borrowed job slot, avoids rayon's injector allocations), wgpu 0.19 backend, optional cudarc/cuBLAS, twin_check scalar-vs-batched gradient verification agreeing to ~3e-8 (discipline mirrors our G5 parity gates).

**Measured claims (their numbers, self-administered):** held-out CE 3.997 vs 4.429 (PPL 54.4 vs 83.8, next-token acc 24.1% vs 18.0%) on a 198,939-token unseen slice; train CE 3.98 vs 4.43 over 12.7M tokens, 29,243 updates; generation 226 ms vs 2,735 ms per 200 tokens (12×, same CPU); training throughput 4.13× but explicitly "not hardware-matched". Both models 1.5M params, same tokenizer/schedule/seed. The README's own honesty section: "research prototype, not a competitor"; text quality poor for both; **retention-after-corpus-switch and memory-bank ablation unmeasured**.

## 2. Why each proposed fusion point fails (the cousin table)

| # | Gemini proposal | Shipped cousin (verified) | Verdict |
|---|---|---|---|
| 1 | PSSA as speculative drafter for katgpt-rs | `katgpt-speculative` (verifier trait, dflash, acceptance_forecast) + `TernaryDraftModel` (`riir-games-quest/plasma_draft.rs`, `.bits`) | **NO.** Drafter must match target distribution; PSSA is 1.5M with its own tokenizer and poor text quality — acceptance would be dismal after a full retrain we have no lane for. Deterministic ternary drafter already occupies the slot. (Code reuse would additionally be GPL-foreclosed — §4 — but the NO stands without it.) |
| 2 | Hybrid SSM-Transformer interleaving (Jamba/Griffin-style) | **Bonsai-2-27B is a GDN+FA hybrid live in the 4090 league** (riir-infer `deltanet/` + `riir-infer-gpu` GDN kernel family; `att_pf_fa` default-on, Plan 605 T4) | **Already shipped, strictly stronger.** GDN's matrix-state delta-rule recurrence ⊃ PSSA's diagonal SSM. We operate the hybrid at 27B with a full kernel stack. |
| 3 | Hyperbolic memory crate for long-range recall | `katgpt-core/poincare.rs` (`poincare_navigator`, **default-on**, Bench 449 G1–G7) + `product_key_memory/episodic.rs` (PkmEpisodicStore, δ-rule write gate) + `delta_mem` | **Substrate exists.** The quality axis for hyperbolic retrieval in OUR use was **measured REFUTED** (riir-ai Bench 497 §4, `poincare_imagination` stays opt-in permanently). PSSA adds no evidence: its own ablation is unmeasured, and its read is NOT bounded (§3). |
| 4 | Plastic weights + symbolic-gated updates | Freeze/thaw + `MerkleFrozenEnvelope` + dendritic LoRA + `memory_soup_lora`; runtime base-weight mutation is **prohibited by the modelless mandate** (constraint 3) | **NO.** What PSSA actually ships is training-time fast/slow EMA bookkeeping (§3), not closed-form ridge. The runtime variant would need recasting as a deterministic freeze/thaw overlay with zero measured upside. |
| 5 | `Lane::PSSA` in riir-reflex | reflex lanes = classification comparison arenas (laya, CLM, GLiNER, AgentJev, PAW, openthai) | **NO.** A 1.5M text generator is not a decision classifier; it would sit far below every floor the arena measures. |
| 6 | PSSA drafter + neuro-symbolic shield + abstention | `distance_abstain`/`CorpusDistanceGate` + `SigmoidGateCalibrator` already guard the modelless engine | **NO** (rests on #1). |
| 7 | Hyperbolic memory routing for game/domain heads | `poincare_navigator` is already the shipped routing/navigation primitive (default-on) | **Already shipped.** |
| 8 | `/feedback` → plastic weight updates | `/feedback` EXISTS (`riir-reflex/src/serve.rs` L1029) but feeds the sigmoid calibrator (`{p, outcome}` → observe/refit) — mandate-compliant by design | **NO.** Wiring weight mutation from it violates the modelless mandate, and PSSA performs no inference-time weight updates anyway (§3). |

The Super-GOAT question never opens: Q1 (prior art) fails against our own substrate on every axis; Q2-Q4 are moot.

## 3. Source-honesty audit (README vs code at `2d5b105`)

Three headline mechanisms are stated stronger than they ship — load-bearing for any future re-evaluation:

1. **"Closed-form ridge-regression consolidation … folds the fast plastic updates back into the base transition matrix"** → `ema_consolidate_plasticity` (pssa.rs L1196-1198) calls `PlasticAdapterV2::consolidate(α)` (adapter.rs L65-72): an exact-transfer EMA between the adapter's fast and slow coefficient stores. **No `(H^T H + λI)^{-1} H^T dH` exists anywhere in the tree; nothing touches `A_base`** — confirmed by the repo's own benchmark audit string (feature_benchmark.rs L281: *"no ridge regression is implemented by this API"*) and by `Matrix::invert()` (linalg.rs L452) having **zero callers**. The authors' own instrument disowns the README equation.
2. **"The read is bounded at four slots … a fixed cost per token regardless of how much the bank holds"** → `retrieve_soft_into` (memory.rs L184-237) computes distances and softmax over **ALL** `count` slots — O(count × d_k) per token, up to the full 512-slot bank. No top-4 selection exists in the read path.
3. **"Rewrites part of its own weights while it runs" / "a memory bank written during the forward pass"** → the bank is written only in the training loop, **after backward, before AdamW** (their own `feature_benchmark.rs` L280: "enabled calls insert_training_memory after backward, before AdamW"); `src/inference.rs` contains zero write calls, their training log states `memory_writes=after_batch` (training.rs L86) and the evaluation harness is annotated "Forward only: no backward, optimizer, consolidation or memory-write path" (evaluation.rs L190). Inference is read-only over a frozen bank; "plasticity" is a training-time two-timescale optimizer scheme, and the fast store IS Adam-updated by backprop.

Also noted: the parameter-matched transformer baseline is their own hand-written implementation, not a reference one — the 0.45-nat learning-efficiency gap is self-administered and unattributed across (SSM inductive bias / memory bank / plasticity), the last two unmeasured.

## 4. Distill sub-verdict (riir-clippy axis) — NO / MARGINAL-D

- **License:** GPL-3.0 — corpus work requires strictly original fixtures (already the house rule); no vendoring ever.
  - *License scope (clarified 2026-10-01, same-day owner challenge):* GPL-3.0 governs the *code*, not the mechanisms — copyright covers expression, not ideas; the mechanism-relevant regime is patents (none indicated, and the constituent mechanisms are prior-art-rich: selective SSM ← S4/Mamba, Poincaré retrieval ← Nickel & Kiela, fast/slow weight EMA ← fast-weights literature, confidence-gated overwrite ← refractory-period / synaptic-consolidation gating, EWC-style overwrite protection — engineering-risk framing, not a formal clearance). **Concept-level distillation was therefore never license-blocked** — §6 records the concept extractions and judges them on merit. What the license actually forecloses: (a) vendoring/adapting source into our MIT/commercial trees — the cheapest adoption path an MIT/Apache source would have offered; (b) close structural reimplementation (non-literal-copying exposure is higher under GPL — clean-room discipline + original decomposition required, our default anyway); (c) verbatim quotes beyond short evidentiary citations as used in §3. **The NO-FUSE verdict is license-independent**: it stands on §2 substrate grounds and §3 source-honesty grounds alone, and would be the same had PSSA been MIT.
- **Candidates found:** persistent-worker/borrowed-job-slot rayon handoff (`scan_executor.rs` — real allocation-avoidance, niche + `unsafe`), value-dirty-check cached transform (`refresh_ssm_rates` — recompute softplus only when the raw value changed, comparing values not versions), exact-sum-preserving EMA consolidation transfer (`consolidate`). All thin: no measured GOAT, no two-regime decision content, no bounded fix-space worth a rule.
- **Lane-intel axis:** no mapping — our GDN lanes are plateau-accepted/closed and PSSA's diagonal SSM is a weaker class than what we run. **Product-coverage axis:** not a famous-lib family.
- Per the distill skill (NO/MARGINAL, no lane mapping): conversation only — no queue entry, no riir-clippy artifacts. This note is the durable record.

## 5. Gemini fact-check (for the record)

Independent verification of Gemini's claims about OUR stack: `katgpt-personality` crate **TRUE**; katgpt-speculative **TRUE**; `compression_drafter`/`template_decode` in katgpt-core consumed by reflex **TRUE**; reflex `/feedback` endpoint **TRUE** (serve.rs L1029, calibrator observe/refit); "signed sigmoid dynamic gating" **PARTIAL** ("signed" hallucinated — plain sigmoid gates ship, extensively); `distance_abstain`/`sigmoid_calibration` **TRUE**. The fusion proposal was grounded in real surfaces — its failure is not hallucination but *substrate déjà vu* plus PSSA's own §3 gaps.

## 6. Micro-patterns recorded (below action threshold; re-open triggers)

1. **State-conditioned retrieval key** (`q = W_qx·x + W_qh·y`): retrieval keyed on input AND accumulated recurrent state. Cousin gap: our retrieval surfaces key on the current query alone. *Below threshold:* no measured win; hyperbolic retrieval quality already refuted for our use (Bench 497).
2. **Refractory write gate** (confidence budget + exponential recovery + surprise-scaled damage): overwrite protection for stabilized memory slots. *Below threshold:* five hand-tuned constants, unvalidated in isolation; PKM's δ-rule write gate occupies the slot.
3. **Exact-sum-preserving consolidation transfer** (`slow += α·fast; fast ← (1−α)·fast`): folding a fast overlay into a slow store while provably preserving the effective function. *Below threshold vs. freeze/thaw's atomic versioned swap — but the "consolidation preserves the effective function exactly" invariant is a nice phrasing of a freeze/thaw-overlay law we already honor by construction.*

**Re-open triggers (any one):** (a) PSSA-class result at ≥100M params with **attributed** ablations isolating memory-bank and plasticity contributions; (b) the shipped read becomes actually bounded top-k; (c) recurrence upgraded to delta-rule/matrix-state; (d) license change away from GPL — reopens only the vendoring/adaptation path for the §4/§6 micro-patterns, not the NO-FUSE verdict. Even then the evaluation target is riir-infer/riir-train as a comparison lane, not an engine fusion.

## 7. Verdict

**NO-FUSE — Pass-tier for every proposal, RECORD for the source.** No files beyond this note; no issues, plans, batches, or lanes. Cleanup: `.raw/pssa` deleted at commit time (sha + license pinned above).

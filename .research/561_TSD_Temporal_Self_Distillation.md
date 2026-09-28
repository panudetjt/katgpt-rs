# Research 561: TSD — Temporal Self-Distillation for Faster dLLM Inference

> **Source:** "Temporal Self-Distillation: Faster Inference in Discrete Diffusion Language Models" — Xu, Miele, Jazbec, Roth, Nalisnick, Bogunovic (Basel/Amsterdam/JHU), arXiv:2609.15177, 2026-09-14
> **Date:** 2026-09-16
> **Status:** Done — GAIN (model-based plan filed; modelless class killed by prior art, residue filed) — residue SETTLED 2026-09-16: [Bench 802 calibration-rig](../.benchmarks/802_commitment_gap_calibration_rig.md) — DEAD-BY-DOMINATION at micro scale (issue closed+removed; trail in HISTORY.md)
> **Related Research:** 151 (GDSD — this paper's own init checkpoint for planning tasks), 149 (FlashAR — decode-strategy cousin), 085 RMSD, 107 ZEDA, 122 EDGE-OPD, 160 SDPG
> **Related Plans:** riir-train/.plans/409 (TSD micro GOAT gate), katgpt-rs .issues/802 (commitment-gap calibration residue)
> **Classification:** Public

---

## TL;DR

TSD trains a masked diffusion LM so its **early denoising-timestep predictions anticipate its commitment-time predictions** (the distribution at the final timestep `t_ℓ` where position `ℓ` is unmasked). Loss: `V(x₀) · Σ_ℓ JSD₀.₅(p_θ(·|c,x_t̂)_ℓ, sg[p_θ(·|c,x_{t_ℓ})]_ℓ)` — on-policy, single-stage, no offline teacher. Result: speed–quality Pareto frontier shifts hard into the low-NFE regime (4.5× NFE on Countdown, 2.3× on MBPP), composes after RL post-training (inits from GDSD checkpoints — our distilled note 151).

**Distilled for katgpt-rs (modelless, inference-time):** the loss is a *measurement* of a runtime-observable quantity — the early-vs-commit distribution gap. Without training we can still measure it, tabulate it, and gate on it (offline commitment-gap calibration tables → issue 802). The raw runtime gate itself (argmax-persistence ∧ inter-step JSD unmasking) is **published prior art (LESS, arXiv:2606.16908)** and is NOT claimed here.

---

## 1. Paper Core Findings

- **Objective:** distill each position's early-step distribution toward the model's own distribution at that position's **commitment timestep** `t_ℓ` (unique per position under unmasking-only decoding). Teacher = stop-gradient self-target; student carries the gradient at a uniformly sampled single timestep `t̂` (unbiased MC of the trajectory average → gradients at one step per rollout).
- **Divergence:** JSD β=0.5. Ablation: reverse-KL causes **reward collapse + NFE saturation** (unbounded, teacher-probability blowup); JSD ≤ ln 2 stays stable across 3000 steps. Same bounded-divergence lesson as GDSD's squared-loss (note 151). (Sibling evidence 2026-09-28, riir-train [R460](../../riir-train/.research/460_DCE_SRCL_Co_Evolving_Privileged_Teacher.md) / arXiv:2609.30652 DCE+SRCL: in *gold-conditioned on-policy* self-distillation even bounded JSD collapses — 17.99% vs 65.97% for the refreshed (dynamic) forward-KL schedule, and even the frozen forward-KL arm reads 41.04% — via premature shortening; forward-KL is the only direction of the three that transfers privileged revision guidance there.)
- **Verifier weighting `V(x₀)`:** task-checker reward multiplies the loss — graded forms preferred (Countdown 1.0/0.1/0.0, Sudoku fraction-of-correct-cells, code test-pass+format). Zero-reward rollouts contribute zero gradient.
- **Length-aware `V_len`:** `V·(1 − γ·(|x₀|−L_target)₊/(L−L_target))`, γ=1 for code — prevents **length collapse** (canvas-filling verbose correct outputs), load-bearing for cross-canvas generalization (L∈{128,256,512}).
- **Training recipe:** LoRA r=64/α=32/do 0.05 on q,k,v,up_proj of all blocks (83.9M = 1.05% of 8B), AdamW lr 3e-6 (identical to our GDSD config), ~2000–3000 steps, 4 rollouts/prompt, Fast-dLLM block-32 decode λ=0.9 during rollout, 2×A100/H200.
- **Results:** Countdown 80% acc @ ~20 NFE (vs ~90 for GDSD-post-trained base) = 4.5×; MBPP 2.3× fewer NFE than LLaDA base, 1.5× fewer than offline dParallel; works **after** RL post-training (GDSD init) — RL builds capability, TSD accelerates; complementary axes.
- **Limitations (paper's own):** no capability gain — makes existing behavior earlier, never better; assumes unmasking-only decoding (unique `t_ℓ`; remasking samplers break the target definition); needs programmatic verifiers (open-ended generation out of scope); `L_target` per task.

## 2. Distillation

### 2.1 Path 0 component inventory (three-track merge)

| # | Paper component | Track | Stack analog (verified to ship) | Verdict |
|---|---|---|---|---|
| 1 | JSD₀.₅ early→commit distillation loss | Model-based | `riir-train-engine/src/loss_gdsd.rs` (`gdsd_loss_grad`, `gdsd_loss_with_tlc`, feature `gdsd_training`) + `riir-train-gpu/src/distill_gdsd.rs` 4-forward harness | **New sibling loss `loss_tsd.rs`** — strictly simpler than GDSD (2 forwards + 1 LoRA backward; no ref model, no old-policy refresh, no clipping). → Plan 409 |
| 2 | Verifier `V(x₀)` reward weighting | Model-based | `riir-train-gpu/src/dllm/actflow_verify.rs` (two-tier syn→cargo-check, `Verdict::{SynInvalid,CheckInvalid,Valid}`), `actflow_grammar.rs::verify` — both ship; Countdown expression evaluator to build (~200 LOC) | Plan 409 Phase 2 |
| 3 | Length-aware `V_len` | Both | Training: scalar in `loss_tsd.rs`. Modelless: selection weight among best-of-n candidates in `d2f_verifier.rs` (item 5 below) | Plan 409 + issue 802 |
| 4 | Single-timestep MC `t̂ ~ Unif` | Model-based | distill loop cadence — trivial | Plan 409 |
| 5 | JSD bounded divergence | Both | NaN-safe bounded top-K JSD kernel (`JSD = H(M)−½H(P)−½H(Q)`, disjoint supports → exactly ln 2, never NaN) — needed by drift scoring + tri_mode graded verify | katgpt-core infra → issue 802 |
| 6 | Early-vs-commit gap as runtime unmasking signal | Modelless | `DiffusionSampler` (`SamplerFeatures`: top1_prob, margin, top3_mass, entropy, step_norm, pos_norm — **no cross-step consistency feature today**, signal-diff confirmed) | **Class killed: LESS arXiv:2606.16908 ships it** (below). Residue → issue 802 |
| 7 | On-policy Fast-dLLM-threshold rollouts | Model-based | D2F block-causal decode (`riir-infer-core/src/transformer/dllm.rs`, `katgpt-forward/src/d2f/`) | Plan 409 |

### 2.2 Modelless prior-art verdict (adversarial panel, No-GD advocate)

**Kill: the raw stability gate is published.** LESS — "LESS Is More: Mutual-Stability Sampling for Diffusion Language Models" (arXiv:2606.16908, 2026-06) — ships training-free joint unmasking: top-1 confidence ∧ **argmax persistence across recent reverse steps** ∧ **top-K inter-step JSD stability**, framed as online stopping; 72.1% reverse-step reduction on Dream-7B/LLaDA. That is candidate #1's exact class and components. Landscape verified around it: Fast-dLLM (single-step confidence), Prophet/EDIT (answer-level convergence, not per-position), EB-Sampler/KLASS (entropy/KL, single-step), Learn2PD + Jazbec 2026 (training-track learned policies), S2D2 (self-speculative two-pass class), FlashDLM FreeCache (temporal stability for KV caching).

**Surviving modelless residue (what LESS does NOT claim):**

1. **Offline commitment-gap calibration tables** — TSD's loss is literally `E[V·JSD(p_t ‖ p_{t_commit})]`. Run vanilla decode over a calibration corpus, record `JSD₂(p_t(ℓ), p_{t_commit}(ℓ))` bucketed by (task class, t, t_commit−t), invert the measured conditional `P(z_t = z_{t_commit} | gate fires) ≥ p*` into gate thresholds with **stated precision**, BLAKE3-commit the table. LESS uses hand-set thresholds; confidence-threshold theory exists but no commitment-gap-calibrated threshold selection surfaced. Coverage unverified → **novelty TBD → issue 802** (not claimed as Super-GOAT).
2. **NaN-safe bounded top-K JSD kernel** — infrastructure, not novelty; prerequisite for 1 and for graded tri_mode verdicts (JSD-agreement scoring replaces binary prefix-match accept in `d2f_verifier.rs`).
3. **Commit-horizon difficulty map** — `horizon(ℓ) = t_commit(ℓ) − t_first_stable(ℓ)` from free stability counters; high-horizon positions are the model's per-position uncertainty map → reallocate verify budget there. Fuses with tri_mode (AR-verify rejection labels are free AUC labels). → issue 802.
4. **Refuted with reasons:** naive two-pass commitment-context rehearsal (reinstates the +NFE cost TSD removes; S2D2 owns the class); unmask-schedule length bias beyond candidate selection (long-answer regression risk, no TSD grounding); V_len as a *training* reward outside checker tasks.
5. **Boundary laws** (correctness conditions, not polish): arm stability counters only above a probability mass floor (near-fully-masked argmax is noise); reset stability state on any remask (FSM invariant — TSD's unique-`t_ℓ` assumption); stride-s measurement is an inductive bridge (TSD's training-time subsampling does not prove the measurement analog — gate it empirically).

### 2.3 Fusion

- **TSD × GDSD (note 151):** shipped GDSD harness is the on-ramp — TSD inits *from* GDSD checkpoints in the paper. Order: capability (GDSD) → acceleration (TSD). Shared 4-forward harness; the DRY obligation (extract the common loop) lands with `distill_tsd.rs`.
- **TSD × FlashAR consensus (note 149):** TSD-trained checkpoints make D2F drafts more commit-safe early → raises the dual-path consensus acceptance rate that note 149's `flashar_consensus` consumes. Independent gains, shared decode path.
- **TSD × DiffusionSampler (Plan 089 T6):** the sampler's 6-feature logistic predictor has no cross-step axis; the calibration tables + stability counters supply the missing features (trained-predictor fusion ≠ LESS's fixed rule).

## 3. Verdict

**Per-track (one verdict per track):**

- **Model-based: GOAT (adoption).** Not Super-GOAT — the on-policy dLLM self-distillation class is crowded (SDTT, dParallel, CForce, COPSD, dOPSD ×2, SD·RL, TSD — six concurrent works); TSD is a strong recipe, not a new capability class. Adoption on our shipped harness is provable-gain territory → **riir-train Plan 409**, micro-scale GOAT gate first (0.5–2 GPU-h; kill gate before any 2B commitment), 2B chain (~10–30 GPU-h on 1×4090) gated on it. Honest prerequisites: Stage 0 dLLM base adaptation is the real long pole (~30–80 GPU-h) and the paying consumer is the **D2F draft arm**, not the DFlash primary serving path.
- **Modelless: Gain → issue 802.** Raw stability gate killed by LESS (auditable discard: class + both components published). Surviving residue (calibration tables, JSD kernel, horizon map, tri_mode graded verdicts) filed as POC/optimization issue — calibration-table novelty explicitly TBD pending coverage pass.
- **GOAT gate spec (Plan 409, summary):** λ-sweep frontier protocol, base vs TSD checkpoint, G2 requires ≥2× NFE reduction at matched accuracy (paper: 4.5×/2.3× at 8B; 2× is the honest 2B-class bar) **and G2′ must beat the best *tuned-modelless* baseline** (threshold tuning + `set_diffusion_schedule` orderings) — trained weights must buy frontier movement, not what schedule tuning fakes. G3 stability gates: JSD ≤ ln 2 throughout, no length collapse, no NFE saturation (the KL-collapse signature is the abort signal).

**MOAT gate (§1.6):** `katgpt-rs` scope confirmed — dLLM sampling/decode is this repo's transformer-stack slot (precedent: note 151 GDSD filed here; `katgpt-forward/src/{diffusion_sampler,denoise_loops,d2f*}.rs` ship here). Training how → `riir-train` (Plan 409). No game/chain/shard angle; no re-route.

**Panel record:** two advocate briefs (No-GD + Model-based), one spawn round, run 2026-09-16. Discard reasons above are auditable per §3.5 discipline. Model-based advocate verified in-code: `loss_gdsd.rs`/`distill_gdsd.rs` ship, `actflow_verify.rs`/`actflow_grammar.rs` ship, `GpuLoraBuffers` up_proj targeting is the one wiring gap. No-GD advocate verified: `DiffusionSampler::decide()` seam + `d2f_decode_block_*_with_sampler(Option<&DiffusionSampler>)` extension points are real, and `SamplerFeatures` carries no consistency feature (signal-diff for row 6).

## References

- Xu et al., "Temporal Self-Distillation: Faster Inference in Discrete Diffusion Language Models," arXiv:2609.15177 (2026).
- Tang et al., "GDSD," arXiv:2605.29398 — note 151 (distilled 2026-06-02; implementation ships in riir-train).
- "LESS Is More: Mutual-Stability Sampling," arXiv:2606.16908 — the modelless-class kill.
- Wu et al., "Fast-dLLM," arXiv/ICLR 2026 — the decoding baseline TSD trains against.
- Concurrent on-policy dLLM self-distillation: COPSD (ACL 2026 Findings), dOPSD arXiv:2607.04428, d-OPSD arXiv:2606.18195, CForce arXiv:2608.13925, dParallel ICLR 2026, SDTT arXiv:2410.21035.

# Research 559: Byte Marginal + Terminal-Mass Certificate (Breaking the Token Ceiling)

> **Source:** "Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models" — arXiv:2609.12303 (Marathe, Pagnoni, Limisiewicz, Li, Lewis, Zettlemoyer, Iyer; Meta FAIR + UW, 2026-09-11)
> **Status:** Done — GOAT verdict; Plan 598 EXECUTED 2026-09-16 (Bench 770): the
> certificate machinery is correct and alloc-free (G1/G4), but the G2 <5%-of-softmax
> premise is REFUTED (depth-0 gather ≈ 1.97× softmax, full loop 2.63×) —
> `refinement_marginal` stays OPT-IN, no promotion. The never-under-reports bound
> validated on-stack (98 adversarial cells) and is inherited by any future
> exact-conversion lane.
> **Related Research:** 017 (Fast BLT — byte *architecture* negative verdict; this note is the byte *interface* on token models, complementary), 548 (TriSpec margin-gated escalation — the certificate fuses here), 087 (ConvexTok — tokenization-side), 137 (Pplx datrie — trie substrate), 392 (attention dilution — unrelated mass-shape family)
> **Related Plans:** 598 (katgpt-rs byte marginal primitive)
> **Cross-ref (riir-ai):** Issue 961 (perf league BPB normalization); (riir-train) training rows in §7
> **Classification:** Public (katgpt-rs)
>
> **PASS-Redirects (synthesis):** Csillag, Schuller, Dall'Antonia, Guibas, Velho, Novello [arXiv:2606.16926 "Functional Gradient Descent with Adaptive Representations"] — certificate-gated representation refinement in function-space FGD (refine while `(1+ε)U ≥ ε‖g‖`, U closed-form); the control-law shape is this note's certificate family with a **gradient-approximation signal** instead of boundary mass — verdict **Pass (both tracks)**: the modelless extractions (Sobolev tail bounds, refinement trigger, descent certificate) have no in-stack consumer (no function-fitting loop ships), and the training-side residue is covered by a shipped analog — the measured k_V rung law ships as a gate-passed opt-in key (riir-train Issue 459 T2, `sample_measured`; the hand-set ladder remains `train()`'s default) — wiring it is Issue 459's residue, independent of this paper; the FGD residual criterion adds no theorem at that site (subspace projection ≠ function-space FGD) and is not free to measure (the anchor config computes no same-data full/active gradient pair).

---

## TL;DR

The paper's training study (byte students overtake token students at 6.33e22 FLOPs) is **134 GPU-years away on our 4090 and priced out by its own 4.5× inference-position tax** — honest NO-GO on the byte-model lane. The durable yield is the interface math: **single-pass token→byte logit conversion** (Marginalize-It approximate, End-Of-Token exact-with-training), plus the law underneath it — *coarse-graining a categorical over a variable-depth refinement tree is lossy exactly at terminals, unless an explicit terminal symbol absorbs the boundary mass*. Neither the paper nor its exact-conversion predecessors expose the dropped mass; we can compute it **closed-form from the same single pass** as a per-position error certificate `M_k`, gate escalation on it, and keep it in a lossless 257-bin record. That certificate is the uncovered delta in a crowded field (Phan/Hayase/BLD/OmniDraft).

**Distilled for katgpt-rs (modelless, inference-time):** refinement-tree categorical coarse-graining with (a) exact first-symbol marginal, (b) prefix-conditioned approximate continuation, (c) closed-form terminal-mass certificate `M_k` = per-position bound on conversion error, (d) virtual-terminal-bin lossless record — all zero-training on any existing GGUF token model, consuming only logits we already compute + the in-tree BPE vocab table.

---

## 1. Paper Core Findings

1. **Marginalize-It** (approximate, single teacher pass): first-byte distribution = exact scatter-add of token probs by first byte; subsequent byte k = restrict vocab to tokens whose byte-prefix matches the realized prefix, renormalize. Silent loss: tokens that *terminate* at or before byte k drop their mass (redistributed by renormalization).
2. **End-Of-Token** (exact, single pass, needs training): append `<eot>` to every token's byte decoding (vocab 260→261), train student with `<eot>` after every BPE token. Dropped mass lands in the `<eot>` bin → teacher distribution preserved exactly, no extra teacher passes. +30.94% FLOPs/unit (one eot per ~4.5 bytes) — which itself *improves* asymptotic BPB (more compute per unit data).
3. **Scaling study** (6 arms: {Tokens, Bytes, EOT} × {CE, KD}, layer-matched 1.28B, ≤1T bytes, 64×H200): token models win low-FLOP and plateau; byte models start worse, higher asymptote. EOT-KD asymptote beats Token-KD by ~4% downstream; byte students match token students with 1/6 the data. Crossover ≈ **6.33e22 FLOPs** (EOT, lr 4e-3).
4. **Storage**: full 260-dim byte-logit dump ≈ 1/5 the size of top-k≈600 token logits, with no truncation loss (boundary mass lives exactly in the tail top-k discards).
5. **BPB non-comparability** (App. G): BPB sees only p(target); downstream accuracy depends on argmax *ranking* of residual mass. Same BPB → 100% vs 0% exact match (worked example). BPT/BPB comparisons across tokenizers/objectives are invalid as model rankings.
6. **Inference caveat** (paper's own §10): byte models remain significantly more expensive at inference (~4.5× positions per text unit); matched-inference-cost comparison is open.

## 2. Distillation

### 2.1 The general law (substrate-independent)

**Coarse-graining a categorical distribution over a variable-depth refinement tree is exact iff an explicit terminal symbol absorbs boundary mass.** Without the terminal bin, renormalize-after-prefix-filter silently deletes exactly the mass of refinements that end within the filter window — a *signed* bias toward longer continuations, magnitude (1−M_k)⁻¹ on surviving mass. This applies to every variable-depth scorer, not just token→byte.

### 2.2 The four-quadrant field (where this paper sits)

| | Exact | Approximate |
|---|---|---|
| **Multi-pass** | Phan 2024 (arXiv:2410.09303, ensembles via byte space); Hayase 2025 ByteSampler (arXiv:2506.14123, ICML'26 — optimizes # of extra passes) | — |
| **Single-pass** | This paper's EOT (needs trained student w/ `<eot>`) | This paper's Marginalize-It (error unquantified in paper — renormalized away) |
| **Ours (Plan 598)** | single-pass + certificate-gated batched escalation (boundary crossings batched into ONE extra pass — they're mutually independent after pass 1) | single-pass **with closed-form M_k certificate** — the bound the paper computes but never exposes |

### 2.3 Fusion (paper × our substrate — none of the three alone can do this)

- **Terminal-mass certificate × TriSpec margin-gating (548)**: `M_k` is a per-position, zero-extra-cost bound on distribution error — the same currency as our verification margins. Gate: serve approximate byte marginal while Σ_k M_k below threshold; escalate to batched-exact above. Sigmoid gate per house rule.
- **Virtual terminal bin × DDTree path scoring**: any variable-depth tree scorer that renormalizes over surviving branches is Marginalize-It-shaped. Audit spec-decode path selection for silent boundary-mass drops; the repair is a scored accept-and-stop bin fed by target EOS/boundary mass.
- **257-bin record × logit-dump storage**: lossless serialization of the single pass (approx distribution + boundary mass kept separate); per paper's arithmetic ≈1/5 of top-k token dumps — and unlike top-k, a *complete* statistic of what the pass computed.
- **BPB unit conversion × perf league**: E[len] per tokenizer converts BPT↔BPB; cross-model throughput rows in the league (different tokenizers) are confounded without it → riir-ai Issue 961.

### 2.4 Precision the advocate brief glossed (kept honest)

The **virtual-EOT 257-bin record is lossless w.r.t. what the single pass computed** — it is NOT the exact byte distribution. Exactness at a boundary crossing requires the continuation distribution *after* the partial token (a second pass). The record's value is: nothing computed is thrown away, the error is *quantified* (M_k), and escalation is schedulable (batched, ~1.2× amortized on batched serving).

## 3. Path 0 Inventory (modelless unblock)

| Component | Coverage in-stack | Extraction (no GD?) |
|---|---|---|
| Token→byte logit conversion | ✗ nothing ships (017 records byte *models* as N/A; SynPruner decodes tokens for validation, never converts distributions) | **YES** — scatter-add over vocab table (`katgpt-tokenizer` BPE decode provides the table) |
| Terminal-mass certificate M_k | ✗ nobody (incl. the paper) exposes it | **YES** — one extra accumulator per trie depth, same pass |
| Virtual-terminal 257-bin record | ✗ | **YES** — scatter-add destination + tag |
| Certificate-gated batched-exact escalation | ✗ (TriSpec gates verification margins, not conversion error) | **YES** — scheduling, not modeling |
| Cross-tokenizer interchange | ✗ (draft/target share tokenizer today) | **YES** — capability published (OmniDraft/vLLM intersection); byte-space-with-certificate is an alternative mechanism |
| Tokenizer geometry profile (E[len], trie breadth, dead-prefix mass) | ✗ | **YES** — offline, once per checkpoint |
| EOT-trained byte student | ✗ | **NO** — training track; NO-GO (§7) |
| KD-beats-CE at same tokenization | Partial: riir-train has KD pipelines (`distill_attention.rs`, DPO/GRPO lanes) | Recipe confirmation — generic-KD provenance (Hinton/Beyer/Busbridge), not this paper's novelty |

## 4. Published Prior Art (§4 gate — searched 2026-09-15)

| Claim | Prior art | Verdict |
|---|---|---|
| Exact token→byte conversion | Phan 2024 (ICLR'25 poster, fb-research code); Hayase 2025 ByteSampler (ICML'26) — both multi-pass, both optimize efficiency of extra passes | Family covered; single-pass is this paper's delta |
| Byte-space model ensembling | Phan 2024 explicitly | Covered |
| Cross-tokenizer distillation via byte interface | ACL 2026 BLD; Minixhofer 2026 approximate likelihood matching | Covered |
| Cross-vocab speculative decoding | OmniDraft (arXiv:2507.02659, n-gram online adaptation); vLLM token-intersection; HF UAG | Capability covered (not via byte space) |
| Byte-level constrained decoding | xgrammar / outlines / llama.cpp grammars (byte→token masks) | Capability covered; terminal-bin *exactness law* is the cleaner framing, not a new capability |
| **Dropped-mass certificate as exposed error bound** | **none found** — paper renormalizes it away; exact methods don't have it | **Uncovered — our fusion lane** |
| **Certificate-gated approx→exact escalation** | **none found** in this domain | **Uncovered** |
| Single-pass byte conversion (Marginalize-It/EOT) | This paper (Sep 2026) | New; we consume, not re-claim |

## 5. Novelty Gate (pinned claims first)

- **Claim A (primitive):** "Single-pass token→byte distribution conversion with closed-form terminal-mass error certificate, for serving/spec-decode/storage consumers on existing token GGUF models, distinguished from Hayase/Phan (multi-pass exact) by O(1) passes and from the paper's Marginalize-It by exposing — not renormalizing — the boundary mass." → Q1 ✅ (uncovered), Q2 ✅ in-stack (byte-exact serving layer + certificate is a new capability *here*), Q3 ~weak-moderate (engine differentiator, not product feature), Q4 ✅ (serving + spec decode + league + training storage). **Not all-4 → not Super-GOAT** (Q3 honest).
- **Claim B (fusion, cross-tokenizer spec decode in byte space):** capability published (OmniDraft/vLLM/UAG) → mechanism-level alternative only. **GOAT/Gain, not Super-GOAT.**

## 6. Verdict

| Tier | Call | One-line reason |
|---|---|---|
| **GOAT** | Single-pass byte marginal + M_k certificate + 257-bin record + gated escalation (Plan 598) | Provable latency/quality/security-of-answer axis (certificate) on a mechanism new-to-stack with published-math backing; feature flag + bench, promote if it wins |
| Gain | BPT↔BPB league normalization (riir-ai Issue 961); logit-dump storage format (riir-train row) | Cheap, real, actionable now |
| Pass | Feather plots (plotting convention); BPB non-comparability as observation (already our league's benchmark-not-perplexity stance) | No mechanism to ship |
| Redirect (justified) | Byte students, byte drafts, byte verifiers, EOT training | Crossover 6.33e22 FLOPs ≈ 134 GPU-years on the 4090 (~3,500× below at max 2-week run); 4.5× position tax on the tok/s league metric; paper's own §10 leaves matched-inference-cost open |

**MOAT gate (katgpt-rs):** fundamental/inference primitive via fusion ✅ — generic refinement-tree math stays in `katgpt-core` (public), token→byte instantiation in `katgpt-tokenizer` (public), no game/chain/shard semantics. Consumer-first (healer) check: no healer surface — this is vocabulary-layer math, correctly out of riir-clippy scope.

## 7. Training-track rows (riir-train; panel adjudicated)

| Row | Verdict | Note |
|---|---|---|
| Byte student replication (0.4B–1.3B) | **NO-GO** | 10B-byte run = 19 days AND ~2,600× below crossover — pre-confirms the paper's own negative region |
| `<eot>` insertion probe | GO-lite (~1 day, arch-test 0.4B only) | We already get the landmark benefit free: our models train on token streams with EOS separators |
| Byte-logit cache format | GO as format note (0 GPU-h) | Adopt the *principle* (small-vocab full dumps beat top-k) for token students if 27B-teacher KD is ever adopted: top-k=64–256 fp8+u16 ≈ 26–100 GB / 200M tokens |
| alpha-KD sweep (1–2B teacher → 0.4B) | GO (4–5 days) — **provenance caveat: generic KD literature (Hinton/Beyer/Busbridge/Peng), this paper only re-confirms at 1B scale** | Teacher models already in `riir-train/data/` |
| Ternary-KD (Bonsai-27B teacher → ternary student) | Candidate, owner call — same provenance caveat | ~1.5–2 wk; improves the Bonsai lane if KD>CE holds for ternary students; file on adoption |

## 8. Honest caveats

- Single-pass claim taken from the paper's own related-work contrast; Hayase ByteSampler's efficiency frontier is adjacent and moving.
- The certificate's tight constant (TV ≤ M_k/(1−M_k) worst-case) needs on-stack calibration — G1 gate requires the certificate never under-reports.
- DDTree terminal-bin audit is a hypothesis about our verifier scoring, not a measured defect — Plan 598 T5 audits before any fix.
- Scaling-law constants (6.35×, crossover) are the paper's, on their mixture/rigs — consumed as priors only.

## Citation

```bibtex
@article{marathe2026breaking_token_ceiling,
  title  = {Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models},
  author = {Marathe, Kalyani and Pagnoni, Artidoro and Limisiewicz, Tomasz and Li, Margaret and Lewis, Mike and Zettlemoyer, Luke and Iyer, Srinivasan},
  journal= {arXiv preprint arXiv:2609.12303},
  year   = {2026}
}
```

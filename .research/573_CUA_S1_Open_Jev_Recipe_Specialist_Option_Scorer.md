# Research 573: CUA-S1 — the Open Jev Recipe (Specialist One-Pass Option Scorer)

> **Source:** trycua/cua `libs/cua-s1` @ `83f142c4290a0f7d9ed545ae8532858c6e4f8145` (MIT, sparse clone — `.raw/` deleted after pinning) + HF model card `cua-ai/cua-s1-forms` + HF dataset `cua-ai/cua-s1-forms` (~234k rows, MIT, procedural values)
> **Date:** 2026-09-19 (addendum 2026-09-20 — PoC verdict below)
> **Status:** Done — **Gain** (files riir-clippy `.issues/125` files the PoC; **PoC verdict: REFUTED at current corpus scale — see the addendum**) — Research 562's reopen trigger FIRED, addendum there
> **Related Research:** 562 (parent — TypeSafe Jev; this is its "independent replication with numbers"), 322 (Report-the-Floor — binds the PoC's gates), 278-lineage (Engram)
> **Related Issues:** katgpt-rs Issue 810 (the calibration pair — "CalibratedSigmoidGate" is a prose phantom; shipped as `SigmoidGateCalibrator` + `CalibratedActionBridge` — the calibration cousin this PoC must NOT inherit CUA-S1's skipping of), riir-clippy Issue 125 (this note's actionable half)
> **Classification:** Public sources; distillation internal.

---

## TL;DR

CUA-S1 (trycua, MIT) is the first **fully-open implementation of the Jev "System One" contract** Research 562 analyzed: a **706,048-param byte-level one-pass option scorer** for GUI form filling. Given a UI-element context (≤224 bytes) and N typed options (≤96 bytes each — `fill <entity>` pointers + `check`/`click`/`skip`), it emits one probability per option in a single forward pass; a deterministic planner orders and executes with fail-closed safety. It closes 562's two open rows — **architecture** (options-as-queries attention head, adapted from `jevlike`; 562's "undisclosed, GLiNER-shaped guess" confirmed and sharpened) and **recipe** (plain supervised cross-entropy on 10k confuser-laden synthetic episodes — **no RL needed for the contract**, answering 562's "no RLCD math to transcribe" discard) — and **prices the specialist-vs-general gap: 99.7% vs 83.6%** for the hosted general model on the same task. Our modelless healer retrieval composite top-1 is ≈85.2% — the same gap class, on an isomorphic task (context=span, options=rules). **Gain:** riir-clippy Issue 125 files the track-c PoC (GPU-minutes, GOAT-gated against `RerankMode::Structural` on ID **and** withheld-pair OOD) + the eval-taxonomy rows worth adopting (`wrong_rule`/`wrong_span`, acceptable-alternatives, coverage/selective split).

---

## 1. What it is (measured facts)

| Fact | Value |
|---|---|
| Task | Form filling behind cua-driver: score every actionable element independently (one batch, one pass); downstream code orders (fills → checkboxes → the one submit) and executes |
| IO contract | context ≤224 B (`TASK…FORM…ELEMENT…`) + N options ≤96 B each → one probability per option; argmax consumed by executor |
| Params / size | 706,048 trainable; 2.8 MB checkpoint (safetensors-only; pickle rejected) |
| Training | **Supervised CE only**: 10k synthetic episodes (~150k train rows), AdamW, cosine+warmup, 6 epochs, batch 128 |
| Splits | Disjoint by **form concept signature** (a test form's field set never appears in train) — compositional OOD by construction |
| Results | 99.95% top-1 synthetic test (~15k decisions); **100%** real demo (196 decisions, nothing synthetic); **37%** shuffled-context control |
| Head-to-head | vs hosted Jev `jev-latest` (zero fine-tune, same task): **99.7% vs 83.6%** overall; Jev 96% on judgment decisions, **74% on already-filled→no-op** (a convention it was never trained on) |
| Safety | Dry-run default; `execute`/`submit` independent opt-ins; one unambiguous target window; snapshot-bound element tokens; reobserve after each mutation; submit gate = exact label `Submit`/`Submit Form` on Button/AXButton only |
| Honest limits | Chooses only among extractor-found entities (cannot invent values); synthetic-trained, real eval is 196 decisions; byte-level English-centric; **not calibrated** (their own admission — RLCD not used) |

## 2. The architecture (closes 562's "undisclosed" row)

`AttentionHead` (adapted from **jevlike** @ `94f5fd1`, Minimal Labs, MIT — properly attributed in-source):

```
q_n = W_q(option_n)                    # each OPTION is a query
K = W_k(context_tokens), V = W_v(context_tokens)
attended_n = softmax(q_n·Kᵀ/√r) · V    # per-option, query-conditioned pooling
logit_n = q_n · attended_n / √r        # self-referential dot: the option scores
                                       # itself against ITS OWN view of the context
```

Byte embedding (257 entries, UTF-8+1, pad 0) → 2-layer Transformer encoder (width 128, 4 heads) over the context, 1-layer over each option → mean-pool options → the head. Dead options masked to −∞. `shuffle_context` is **built into `forward()`** — the control ablation ships as an architecture hook.

**Softmax note (pre-empting review friction):** the final softmax is over **mutually exclusive options** — the categorical-choice exception to the house sigmoid rule (same class as the sampling layer; sigmoid governs independent gates/projections). Note CUA-S1's **abstention lives INSIDE the softmax as a trained `skip` option** — a design alternative to our threshold-gated `ActionBridge` abstain (see §5).

**The planner separation validates our shape:** model scores, deterministic code orders/executes, fail-closed everywhere — the same separation as our suggest/fix/bench-guard lanes and the `--clippy-json` oracle-anchored feed.

## 3. What is new vs Research 562

| 562 open row | CUA-S1 answer |
|---|---|
| "Architecture undisclosed; HN guess: small encoder + calibrated readout (GLiNER2-shaped)" | **Confirmed and minimal**: byte embed + 2-layer encoder + options-as-queries attention head; 706k params suffice for the contract at 33 options |
| "RLCD: no paper, no math — nothing to transcribe" (audited discard) | **You do not need RL at all**: supervised CE on confuser-laden synthetic data reaches the contract. RLCD remains TypeSafe's (unpublished) path, not a prerequisite |
| Reopen trigger: "independent replication with numbers" / "any plan that trains a decision head" | **Fired** — open code + data + training + eval, with numbers |
| Jev "cannot abstain" (their structural weakness, our edge) | CUA-S1's fix: **train abstention as an option** (`skip` competes in the softmax). Our `ActionBridge`/`CalibratedActionBridge` abstain by threshold — an unshipped design alternative worth one PoC arm |
| Arena row #15 (zero-shot breadth — our LOSS) | Priced from the other side: a **specialist** at 1/3000th the size beats the general model in-domain (99.7 vs 83.6) and is brittle OOD (their own scope warnings). Specialization buys the niche; it does not buy breadth |

## 4. Path-0 inventory + three-track verdicts

| Component | Track | Extraction / analog | Verdict |
|---|---|---|---|
| Options-as-queries scoring over runtime-N text options | a | **Partial**: `pick_domain`/`pick_domains_top_k` (fixed N, fixed directions, fixed-dim activity — no text options, no per-option attention pool). The text-option form ≈ our BM25+KNN over options (riir-rag) — but no single learned binding head. The pool primitive itself is new only as a *trained* artifact | Recorded (ships as the PoC's model class, not as katgpt-core substrate — it is track-c by construction) |
| Abstention as trained option | a/c | Ships differently: `ActionBridge` threshold + Issue 810 calibrator + ARG `ABSTAIN`/`CLARIFY`. Trained-abstain = the PoC's second arm | Gain (PoC arm) |
| **Metric taxonomy** (coverage, abstention_rate, selective_accuracy, wrong_action vs wrong_target, unsafe_action, acceptable-alternatives gold, missing-prediction-counts-as-abstain) | eval | **Partial**: top-1/3/5 + score-bench `persist/broken/created` (`broken` ≈ `unsafe_action` analog). NOT shipped: wrong-rule vs wrong-span decomposition, acceptable-alternatives, coverage/selective split | **Gain — adopt** (Issue 125 task) |
| Signature-disjoint splits + shuffled-context control | eval | **We ship deeper**: Bench 062/063/065 withheld-pair + `shuffled_role_control` with vacuity modes, NOROLE canary, leak detectors, preregistered predictions. CUA-S1's 37% shuffle control is the simpler ancestor | Covered (ahead) |
| Specialist-beats-general empirical pricing | c | 706k vs hosted frontier API, same task: +16.1pp. Our parallel datum: healer composite top-1 85.2% modelless | Gain (prices the PoC) |
| Supervised specialist recipe (confuser curriculum: 35% hard negatives, forced look-alike pairs, 60% pre-filled→skip, 12% missing entity, 30% partial forms, 5% stale values, 20% title dropout) | c | Full generator ships (`synth.py`, seeded/deterministic, 55-concept catalogue with synonym tables) | Gain (recipe for Issue 125) |
| Safety boundary (dry-run default, reobserve-after-mutation, exact-label submit gate) | a | Analogous postures ship (bench-guard decline-or-flag, fail-closed gates everywhere) | Covered (validates) |

**Track-c affordability (Path 0.5):** 706k params × ~200k rows × 6 epochs ≈ **GPU-minutes on the 4090** (CPU-trainable in a pinch). **Serving-envelope check (TTPO rule):** the heal path is NOT the 20 Hz tick — spans are already chunked and embedded; a ~1 ms/span byte-scorer sits comfortably beside the embedder at per-fix budgets that dwarf it. The game tick keeps `pick_domain` (ns-tier) untouched. This is the rare track-c case the envelope rule **favors** rather than blocks.

## 5. Substrate mapping (signal-diffs)

- **`pick_domain` / `pick_domains_top_k`** (`katgpt-core`, opt-in Plan 558): scores a fixed-dim activity against FIXED direction vectors — CUA-S1's delta is (a) options are runtime-supplied **text** (byte-encoded, arbitrary N with masking) and (b) the score is a query-conditioned attention pool, not a plain dot. Name-level cousin ≠ coverage: no shipped primitive scores arbitrary text options against a text context in one head.
- **`ActionBridge` + `CalibratedActionBridge`** (`katgpt-core`): fixed-arity action selection with threshold abstain and (Issue 810, pending) calibrated confidence. CUA-S1's trained-`skip`-in-softmax is a genuine design alternative — the PoC compares them empirically.
- **riir-clippy retrieval lane**: `AstChunker → ModellessEmbedder → RuleIndex (KNN+BM25) → RerankMode::Structural → canonical merge` is a modelless option scorer; composite top-1 ≈ **85.2%** (Bench 096 control). The 99.7-vs-83.6 pricing says the trained-specialist class is worth one measured attempt on OUR corpus — with Bench-062's withheld-pair OOD as the anti-memorization gate.
- **Eval family**: Bench 062/063/065 already exceed CUA-S1's eval discipline (preregistration, vacuity instrumentation, leak detectors). Adopt only the **metric taxonomy** rows (§4).
- **Freeze/thaw**: a 2.8 MB checkpoint is trivially `MerkleFrozenEnvelope`-compatible — if the PoC wins, the specialist ships as a freeze/thaw artifact (BLAKE3, versioned), never an in-place mutation.

## 6. Game-context reframe (riir-ai) — brief, the arena stands

Research 562's addendum (11 W / 2 SPLIT / 2 LOSS) is unchanged; CUA-S1 adds one pricing row to the #15 LOSS column: the zero-shot-judgment niche they win is buyable **per-domain** with GPU-minutes and a labeled corpus — exactly what the healer has and a game domain mostly lacks (our NPC action sets are fixed-arity and stay modelless). No game-surface plan follows from this source.

## 7. Consumer reframe (riir-clippy — the main event)

Healer rule selection IS the CUA-S1 task shape: context = code span, options = candidate rules + decline, label = oracle ground truth — and we hold labeled corpora CUA-S1's authors did not have to synthesize (retrieval_eval fixtures, fixseq rings with oracle-anchored spans, `--clippy-json` feeds). Their confuser curriculum maps to our known look-alike lint pairs (`redundant_closure` vs `redundant_closure_for_method_calls` family). The modelless lane stays default; the specialist is a challenger arm. → **riir-clippy Issue 125** (the PoC, gates below).

## 8. Fusion

**CUA-S1 recipe × healer oracle-labeled corpora × Bench-062 withheld-pair harness → specialist rule scorer PoC:**

1. Render choice rows: context = span bytes (≤224), options = canonical-merge candidates for that span (N≈7–33) + `skip`, label = oracle rule or decline.
2. Confuser augmentation: force look-alike rule pairs into every option set; shuffled-context control arm must read ≈ chance (reuse the Bench-063 control discipline — a structured verdict with a non-degrading shuffle is a harness defect).
3. Model: tinyx clone (~700k params). Train CE/AdamW 6 epochs — GPU-minutes.
4. **GOAT gates (Report-the-Floor bound):** (G1) composite top-1 ≥ shipped structural baseline on the SAME fixtures; (G1-OOD) withheld-pair (rule, span-family) gap ≤10pp where structural holds (Bench-062 P4 bar) — a Bench-065 memorization signature = REFUTED, no widen-and-retry; (G2) ≤~1 ms/span batched; (control) shuffle ≈ chance; (calibration) decision-level ECE/Brier reported vs sigmoid floor — do not inherit CUA-S1's uncalibrated posture (Issue 810 tie-in).
5. If GOAT: freeze/thaw the checkpoint; wire as a rerank arm behind a feature flag; promote on measured gain, demote the loser per house rule.

## 9. Verdict

| Tier | This source |
|---|---|
| Super-GOAT | **No** — Q1 fails (562 lineage + jevlike + GLiNER family are the prior art; the delta is confirmation + recipe, not a new mechanism class); Q2 no new behavior class for our stack |
| GOAT | Not yet — nothing lands without the PoC's measured win on our corpus |
| **Gain** | **Yes** — reopen trigger fired with extractable recipe; prices and specifies the healer track-c PoC; eval-taxonomy adoptions |

**Files:** this note; riir-clippy `.issues/125_specialist_option_scorer_rule_selection_poc.md` (PoC + metric-taxonomy task); Research 562 addendum pointer (same commits).

**MOAT:** katgpt-rs — no new open primitive (the head is track-c; the taxonomy is eval methodology); riir-clippy — consumer-first moat (measured healer-quality gain is the bar); riir-train — recipe recorded here, plan only if the PoC scales beyond the healer.

**Next triggers:** PoC win → riir-clippy plan (rerank-arm wiring + GOAT promotion gate); PoC loss → record numbers here, keep modelless default, reopen only on a corpus-order-of-magnitude change; TypeSafe publishes RLCD math → its own distill (562's standing trigger).

---

## Addendum 2026-09-20 — the PoC verdict: REFUTED at current corpus scale (riir-clippy Bench 098)

The filed PoC ran end-to-end (deterministic harness, 719k-param recipe-faithful tinyx, FD-gradient-gated hand-rolled backward, rule-disjoint OOD split, confuser curriculum, all five gates). **The challenger lost**: composite top-1 **27.7% vs the modelless structural baseline's 88.0%** in-dist (n=83); **5.6% vs 100%** OOD (n=18, signature-disjoint); **8868 µs/span** vs the ~1 ms serving bar (single-thread CPU). Controls validated the harness rather than the model: shuffled-context dropped to 12.0% (PASS — context is read), the frozen head scored 7.2% (training genuinely learned at 27.7%), and calibration after Platt-T was still ECE 0.175 (reported per the Issue-810 tie-in; CUA-S1's uncalibrated posture NOT inherited silently).

**The honest reading is corpus scale, not architecture**: our shipped labeled corpus is ~101 gold contexts (~3.6k episodes) against CUA-S1's ~150k-row compositional generator — three orders of magnitude less signal — and the opponent was not a hosted general model but a structural rerank carrying per-rule hand-derived proposer knowledge (88% on the same pools). The specialist class's pricing (99.7 vs 83.6) does not transfer to a regime where the specialist starves; the deepest failure, G1-OOD 5.6% (zero-shot binding of an UNSEEN rule's pattern bytes to a span), is exactly what a hundred contexts cannot teach.

Also recorded for any rerun: v1's curriculum (25% gold-dropped-same-context "decline" rows) was contradictory label noise and collapsed the model to always-skip (0/83, skip p≈1.00 — WORSE than the frozen control); v2 sources decline honestly (clean spans + gold-absent pools) and weights production-pool episodes ×8.

**Standing after the verdict:** modelless default unchanged, nothing wired into heal paths; the deterministic harness + model land as the re-run lane (feature `choice_scorer_poc`). **Reopen trigger unchanged and now measured**: a corpus-order-of-magnitude change (~10⁴–10⁵ labeled contexts — fleet-scale fixseq rings or frontier-miner oracle-verified spans). Full numbers: [riir-clippy Bench 098](../../riir-clippy/.benchmarks/098_choice_scorer_poc.md). The metric-taxonomy half of the issue (ChoiceTaxonomy rows) had already landed and is unaffected.

---

*Provenance: sparse clone of `libs/cua-s1` only (29 files, ~4.5k LOC Python) pinned at the sha above; `.raw/cua` deleted after this note was written; all quotes re-verifiable at that sha. The HF model card's reference to `docs/RESULTS.md` is broken in-tree (no `docs/` dir at this sha) — the results table above is from the model card.*

> **PASS-Redirects (synthesis):** Sebastian Raschka ["Language Models for Text Classification: From Bag-of-Words to Jev" (magazine.sebastianraschka.com/p/classifier-history-and-jev, 2026-09-29)] — his DIY Jev-clone head (shared scalar si = w·hi + b per option, softmax across candidates, supervised CE) is the third independent derivation of the open-recipe class this note extracted from jevlike/CUA-S1 (options-as-queries attention head); class confirmed, no change to the REFUTED PoC verdict or its measured corpus-order-of-magnitude reopen trigger.

# Research 353: Program-Synthesized Attention Head Surrogates (Causal Substitution)

> **Source:** Hayes, Li, Andreas. *Explaining Attention with Program Synthesis*. arXiv:2606.19317. MIT CSAIL / NJIT. 30 Jun 2026.
> **Date:** 2026-06-30
> **Status:** Active — **REVISED: Gain (was GOAT).** Verdict revised after corpus review identified FuncAttn (R257) and Percepta (`katgpt-percepta` crate) as the missing prior-art layer. See §3.3 for the revision log.
> **Related Research:** **257 (Functional Attention — spectral transport operator, the surrogate representation)**, **031/032 (Percepta deep dive / distillation — programs-as-attention paradigm)**, 229 (ProgramAsWeights), 244 (Self-Evolver Faithfulness — causal intervention), 178 (Rosetta Neurons — best program per head), 277 (SmearClassifier), 295 (AC-GPT arbitrary-conditional attention), 233 (Attention Matching)
> **Related Plans:** **286 (Functional Attention spectral transport)**, **064 (Percepta full riir)**, 278 (FaithfulnessProbe — intervention primitive), 298 (SmearClassifier — hallucinated-feature detector), 313 (AC-Prefix), 271 (Attention Matching), 259 (spec_compile modelless), 353 (revised — gate only, not new primitive)
> **Classification:** Public — generic inference-time primitive. No game IP, no chain IP, no neuron-shard IP.
> **PASS-Redirects (synthesis):** Yin & Wang [arXiv:2501.15857 "Are Transformers Able to Reason by Connecting Separated Knowledge in Training Data? (FTCT: Fragmented at Training, Chained at Testing)"] — PASS. FTCT's Lemma 5.3 (2-layer Transformer constructively simulates the underlying program: induction head + parent retrieving) is a theoretical existence proof in the same family as Hayes et al.'s "25-40% of heads are programmable" — but FTCT's construct is closed-form analytical (not a deployable surrogate) and proves only that the program is expressible, not how to extract it. The extraction/synthesis pipeline that makes Hayes et al. actionable is exactly what FTCT lacks. The λ≥0.3 training-data phase transition is → riir-train.

---

## TL;DR

Hayes et al. distill attention heads in pretrained transformers (BERT-base, GPT-2-small, TinyLlama-1.1B, Llama-3.2-3B) into **executable Python programs** synthesized by an LM (Claude Sonnet 4, ~$150 total). For each head, an explainer LM produces candidate `π: tokens → attention_matrix` functions ranked by Jensen-Shannon distance, with the final selection maximizing Intersection-over-Union (IoU) against real attention. **Up to 25–40% of heads can be causally substituted** (interchange intervention: replace `A` with `π(X)` in the live forward pass) with only ~16% perplexity increase and no significant drop on six downstream QA benchmarks. A library of ~1,664 programs (one per head × four models) serves as a reusable surrogate set; the *best program for each head* often comes from a different head's library entry, suggesting generalizable functional primitives (first-token, syntactic, sequential, coreference categories).

**Distilled for katgpt-rs (modelless, inference-time):**

The paper is **fully modelless**: no gradient descent, no weight mutation, no fine-tuning. The pipeline is (1) extract attention maps → (2) prompt an external LM to synthesize Python programs → (3) score by JSD/IoU → (4) causally validate by substitution.

**After corpus review (see §3.3 revision log):** the distilled primitive is NOT a new `ProgramSynthesizedHead` struct. The "tokens → attention via an externally-supplied operator" shape is exactly what **`FuncAttn`** (R257, `crates/katgpt-core/src/funcattn/mod.rs`) already ships — `FuncAttn` solves for the operator C via closed-form Tikhonov; the paper's Python program is a strictly more general operator representation (Turing-complete callable vs closed-form matrix). The "programs become the attention mechanism" paradigm is exactly what **Percepta** (R031/032, the `katgpt-percepta` crate) ships — Percepta compiles C → WASM → lowered bytecode → transformer weights at compile time; the paper does the same via runtime callable substitution. Same paradigm, different point on the compile-vs-runtime spectrum.

**What's actually novel in this paper for us** is not the primitive — it's three empirical findings + one gate:

1. **IoU is a cheap proxy for expensive causal substitution cost** (paper §3, `r > 0.9` Spearman across all three causal models). This is a gate-design fact we did not have.
2. **25–40% of attention heads in real LMs are causally substitutable** with ~16% perplexity cost. This is a measurement of how much of a transformer is "programmable" — directly relevant to how aggressively we apply FuncAttn/Percepta.
3. **Library-search (MAP-Elites) over a small program library beats per-head synthesis** — the globally-best program for a head frequently comes from a different head's entry. Same pattern as Rosetta Neurons (R178) best-buddies alignment.
4. **The gate itself**: `IoU ≥ τ_iou AND cached FaithfulnessProbe ≤ τ_behavior`. This is a new control loop — IoU cheap-proxy gating the expensive FaithfulnessProbe validation, using the Plan 287 cached-cadence pattern.

The right output is **not a new primitive** but (a) a gate that wraps the existing FuncAttn/Percepta primitives, and (b) the empirical findings recorded as facts that update how we use those primitives. Plan 353 is revised accordingly: it ships `HeadSubstitutionGate` as a small wrapper around FuncAttn's existing trait, not a new `ProgramSynthesizedHead`.

---

## 1. Paper Core Findings

### 1.1 The pipeline (paper §2)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Extract attention maps {A₁..Aₙ} from target head on corpus X    │
│  2. Filter top-2.5% attention weights, format as token-pair summary │
│  3. LM synthesis agent S → candidate Python π : X → Â               │
│  4. Score by JSD(A, Â); refine on worst examples (1 round)          │
│  5. Select π* = argmax_π IoU(A_held_out, π(X_held_out))             │
│  6. Causal validation: replace A with π*(X) in forward pass          │
│     → measure perplexity Δ and downstream task accuracy              │
└─────────────────────────────────────────────────────────────────────┘
```

Cost: ~4,000 candidate programs via Claude Sonnet 4 across all four models, ~$150 total API spend. Final library: 1,664 programs (one per head).

### 1.2 The empirical results (paper §3)

| Model | Heads | Mean best-program IoU | 25% replacement perplexity Δ | Downstream QA at 30–40% replacement |
|---|---|---|---|---|
| BERT-base (bidirectional) | 144 | lowest of the four | n/a (poorly characterized) | degrades faster |
| GPT-2-small (causal) | 144 | 69% | ~16% | preserved |
| TinyLlama-1.1B (causal) | 704 | 74% | ~16% | preserved |
| Llama-3.2-3B (causal) | 672 | 79% | ~16% | preserved |

Three observations matter for us:

1. **Causal substitution works for a substantial fraction of heads.** Programs aren't just correlational proxies — they preserve model behavior when swapped in. This is the strongest possible evidence that the program captured the head's *function*, not just its output statistics.
2. **IoU is a valid proxy for causal substitutability.** Spearman correlation between per-head IoU and per-head perplexity-increase-when-replaced is `r > 0.9` for all three causal models. **This means IoU (cheap, no intervention) can gate substitution (expensive, requires forward pass).** This is exactly the cheap-proxy → expensive-validation pattern `FaithfulnessProbe` already uses (IG surrogate in §2.2 of R244).
3. **"Best program" often comes from a different head's library entry.** The globally-best program for a head frequently beats the head's own intended program. This is the MAP-Elites / quality-diversity pattern — a small library of general functional primitives (first-token, syntactic, sequential, coreference) covers most heads. Same shape as **Rosetta Neurons (R178)** "best buddies" cross-system alignment.

### 1.3 The failure mode (paper §3, Fig 2 BERT example)

Synthesized programs sometimes **hallucinate structural features**. The BERT L10H1 example shows the program inventing a diagonal attention pattern that the real head does not exhibit. This is the exact failure mode **SmearClassifier (Plan 298, R277)** was built to detect: "is this latent distribution reflecting a real single feature, or is it smeared across many tokens / hallucinated onto a structural prior?" The paper notes this explicitly as a coverage gap; we already ship the detector.

### 1.4 Decoder > encoder (paper §3.1)

Causal models are far more amenable to symbolic approximation than bidirectional ones. The paper attributes this to (a) the causal mask reducing the search space, (b) functional specialization increasing with scale (more heads = narrower roles per head). For us: **autoregressive game-state transformers and dialogue engines are the right target**, not bidirectional encoders.

---

## 2. Distillation (modelless, inference-time)

### 2.1 What's already shipped (the prior-art surface — five granularities)

**MANDATORY vocabulary translation** before novelty claim. Paper vocabulary → codebase vocabulary:

| Paper term | Codebase equivalent | Where it ships |
|---|---|---|
| "causal head replacement" / "interchange intervention" | causal intervention, `FaithfulnessProbe`, `Intervention::Empty/Shuffle/Corrupt/Irrelevant/Filler` | R244, Plan 278, `crates/katgpt-core/src/faithfulness/probe.rs` |
| "program synthesis" / "executable program approximation" | spec → compiled rule, `SpecPruner`, `SpecChain`, `SpecProof` | R229, Plan 259, `katgpt-core/src/spec_compile` |
| "best program per head" / "library re-ranking" | "best buddies" cross-system alignment, MAP-Elites library | R178, Plan 199/201 |
| "hallucinated structural features" | `SmearClass::TokenSmear / SequenceSmear`, `SmearClassifier` | R277, Plan 298, `crates/katgpt-core/src/faithfulness/smear.rs` |
| "attention map" / "token-pair weight" | `AcPrefix::attends`, `AttentionMatching`, `FuncAttn` | R295/R233, Plan 313/271 |
| "IoU similarity" | cosine / IoU / edit-distance (R244 §2.1 already lists IoU as a generic behavioral-delta metric) | n/a (primitive op) |
| "JSD ranking" | KL-divergence / JSD — already used in collapse-aware entropy gates | R212, Plan 212 |

**Six layers of prior art (the FuncAttn + Percepta layers were missed in the initial verdict — see §3.3):**

1. **R257 / Plan 286 — FuncAttn** ships the **surrogate representation itself**. `FuncAttn(Q, K, V) = Φ · C* · Ṽ` where `C*` is a closed-form Tikhonov-regularized operator between learned bases. The paper's `π: tokens → attention_matrix` is `FuncAttn` where the operator C is **any executable program** instead of a closed-form spectral solve. **Strictly more general program representation, identical primitive shape.** The `Box<dyn SynthesizedAttentionFn>` trait I proposed in §2.2 below is structurally `dyn FuncAttnKernel` — I was reinventing FuncAttn's trait surface.

2. **R031 / R032 / Plan 064 — Percepta** ships the **programs-as-attention paradigm**. The `katgpt-percepta` crate implements `C program → WASM → lowered bytecode → token prefix → transformer weights` (see `crates/katgpt-percepta/src/compile.rs`, `crates/katgpt-percepta/src/gates.rs`). Percepta bakes the program into weights at compile time via Futamura projection; the paper substitutes the program as a runtime callable. Same paradigm, different point in the compile↔runtime spectrum. Percepta's gates (`reglu`, `stepglu`, `multiply`, `persist`, `fetch`, `fetch_sum`) are the executable primitives the paper's Python programs compose at a higher level.

3. **R244 / Plan 278 — FaithfulnessProbe** ships the **causal-intervention validation paradigm**: substitute a component, measure behavior delta. The paper's §2.3 "causal head replacement" is `FaithfulnessProbe` applied to an attention head with `Intervention::ReplaceWith(surrogate)`. The probe's intervention enum is a strict superset of the paper's only intervention.

4. **R229 / Plan 259 — ProgramAsWeights / SpecCompile** ships the **spec → executable program** compilation. The paper's "Python program synthesized by an LM" is one instance of `SpecPruner`-style compilation. The paper's verification step (causal substitution preserves behavior) is the same shape as `SpecProof::Soundness`.

5. **R178 — Rosetta Neurons** ships the **"best program per head"** library-search pattern. The paper's "globally best program frequently beats the head's own intended program" is the MAP-Elites / quality-diversity pattern that `RosettaPruner` already implements.

6. **R277 / Plan 298 — SmearClassifier** ships the **hallucinated-feature detector** the paper explicitly flags as a coverage gap (Fig 2 BERT diagonal example).

**Honest prior-art accounting (revised):** the paper does not introduce a new primitive. The surrogate representation ships as FuncAttn. The programs-as-attention paradigm ships as Percepta. The causal validation ships as FaithfulnessProbe. The spec→program compilation ships as SpecCompile. The library search ships as Rosetta. The hallucinated-feature detector ships as SmearClassifier. **What does not ship is the gate** — the IoU cheap-proxy → FaithfulnessProbe expensive-validation control loop. That is the only novel piece, and it is small enough to be Gain-tier, not GOAT-tier.

### 2.2 ~~The novel modelless primitive~~ — `ProgramSynthesizedHead` + `HeadSubstitutionGate` (SUPERSEDED — see §3.3)

> **Revision note (see §3.3):** the `ProgramSynthesizedHead` struct and `Box<dyn SynthesizedAttentionFn>` trait proposed below are structurally identical to FuncAttn's existing trait surface. The right output is a **gate that wraps FuncAttn/Percepta**, not a new primitive. The code sketch below is retained for traceability of the revision; do not implement it as written — see Plan 353 for the revised gate-only approach.

```rust
/// A compiled attention-head surrogate: tokens → attention pattern.
///
/// Distilled from Hayes et al. 2026 (arXiv:2606.19317). The program is
/// synthesized offline by an external LM (or hand-authored) and committed
/// as a deterministic callable. At inference time, `produce_attention_into`
/// writes the surrogate attention matrix into a caller-provided buffer —
/// zero allocation inside the hot path.
///
/// **SUPERSEDED**: this struct is structurally `FuncAttn` with an externally-
/// supplied operator. Use `katgpt_core::funcattn::FuncAttn` directly instead.
pub struct ProgramSynthesizedHead {
    /// BLAKE3 hash of the program source (for cache verification + chain
    /// commitment if this surrogate is ever committed to a NeuronShard).
    program_hash: [u8; 32],
    /// IoU score on the held-out validation set. Per paper §3, r > 0.9
    /// correlation with per-head perplexity increase → safe gate signal.
    held_out_iou: f32,
    /// The actual callable. Box<dyn> is acceptable here because substitution
    /// is an audit-cadence operation, not per-token.
    program: Box<dyn SynthesizedAttentionFn>,
}

pub trait SynthesizedAttentionFn: Send + Sync {
    /// Write the surrogate attention pattern into `out` given input `tokens`.
    /// `out` is `n * n` row-major. Returns `Err` if the program fails
    /// (matches paper §2.2: "Non-well-formed functions are assigned maximal
    /// divergence").
    fn produce_attention_into(
        &self,
        tokens: &[Token],
        out: &mut [f32], // n*n, row-major
    ) -> Result<(), SynthesisError>;
}

impl ProgramSynthesizedHead {
    /// IoU gate — paper §3 Fig 5b shows r > 0.9 between IoU and substitution
    /// cost. If `held_out_iou < tau_iou`, substitution is not attempted.
    pub fn passes_iou_gate(&self, tau_iou: f32) -> bool {
        self.held_out_iou >= tau_iou
    }
}

/// Gate that decides whether to substitute a real head with its surrogate
/// during a forward pass. Combines the paper's IoU gate with R244's
/// FaithfulnessProbe for live causal validation.
pub struct HeadSubstitutionGate {
    /// Per-head surrogate library (paper's MAP-Elites structure).
    surrogates: Vec<ProgramSynthesizedHead>,
    /// IoU threshold for attempting substitution (paper default: ~0.4).
    tau_iou: f32,
    /// Behavioral-tolerance threshold for accepting substitution
    /// (FaithfulnessProbe-measured, paper's perplexity delta ≤ ~16%).
    tau_behavior: f32,
    /// Cached FaithfulnessProbe results per head — re-measured at audit
    /// cadence, not per-token. Same pattern as Plan 287 SinkAware cadence=16.
    cached_faithfulness: Vec<FaithfulnessProfile>,
}

impl HeadSubstitutionGate {
    /// Hot-path decision: should head `h` be replaced by its surrogate on
    /// this forward pass? Returns the surrogate to use, or None.
    ///
    /// This is the *cheap* decision — IoU gate + cached faithfulness. The
    /// expensive re-measurement happens off the hot path on audit cadence.
    pub fn substitute(&self, h: usize) -> Option<&ProgramSynthesizedHead> {
        let s = &self.surrogates[h];
        if !s.passes_iou_gate(self.tau_iou) {
            return None;
        }
        let f = &self.cached_faithfulness[h];
        if f.behavior_delta_when_replaced <= self.tau_behavior {
            Some(s)
        } else {
            None
        }
    }
}
```

**Why this is the right shape:**

- `Box<dyn SynthesizedAttentionFn>` keeps the program representation **opaque** — it can be a hand-authored Rust closure, a WASM-compiled Python fragment, or (eventually) a Cranelift-JIT'd function. The paper's contribution is the *pipeline* (synthesize + rank + substitute), not the program representation.
- `program_hash: [u8; 32]` (BLAKE3) is the commitment hook. If a surrogate ever crosses the sync boundary (chain commit, shard freeze), the hash is already in place. This follows the LatCal commitment discipline.
- `cached_faithfulness: Vec<FaithfulnessProfile>` reuses R244's exact type — substitution is gated by the *existing* FaithfulnessProbe primitive, not a new measurement.
- The gate has the **same cadence pattern** as Plan 287 SinkAware (cache the decision, recompute every N calls). Per the SinkAware G3 gate lesson, per-call faithfulness measurement is structurally infeasible; the cached-cadence pattern is what makes the gate affordable.

### 2.3 Latent-space reframing (mandatory per workflow step 3)

The paper operates on **token-level attention maps** (raw, discrete). The latent-space reframing for our seven Super-GOAT factory modules:

| Reframing axis | Operation | Where it lands |
|---|---|---|
| **HLA per-NPC latent state** (`sense/`) | A "head" in HLA is one of the 8 affect projections (valence/arousal/desperation/calm/fear + 3). The analog of "program-synthesized head" is a **deterministic projection rule** that replaces a learned direction vector with a hand-authored one (validated by IoU between the projected scalar time series and the original). | `crates/katgpt-sense/src/reconstruction.rs` — `evolve_hla` becomes auditable for "is this projection direction load-bearing or could a constant replace it?" |
| **Latent functor** (`riir-engine/src/latent_functor/`) | A "head" is a functor application. The analog is a **symbolic functor** (closed-form arithmetic op) replacing a learned functor, gated by coherence preservation. Direct fit with `quality_gate.rs` / `reestimation.rs`. | riir-ai runtime |
| **CGSP runtime** (`riir-engine/src/cgsp_runtime/`) | A "head" is a curiosity branch. The analog is a **deterministic exploration policy** replacing a learned curiosity signal, validated by collapse-rate preservation. | riir-ai runtime |
| **NeuronShard** (`riir-neuron-db/src/shard/mod.rs`) | A "head" is one of `style_weights[64]`. The analog is a **deterministic weight column** replacing a learned one, validated by downstream-task preservation. **This is the most interesting reframing** — it gives NeuronShard consolidation an audit step: "which style weight dimensions are load-bearing, which could be replaced by a constant?" | riir-neuron-db |
| **DEC operators** (`katgpt-dec/src/`) | A "head" is a DEC operator application (d / δ / Δ / hodge_decompose). These are *already* deterministic programs — the paper's framework is trivially satisfied. Not interesting. | n/a |
| **LatCal** (`riir-chain/src/encoding/`) | A "head" is a LatCal matrix entry. LatCal is *already* a deterministic 2×2 matrix program. Same as DEC — trivially satisfied. | n/a |

**The most valuable latent reframing is the NeuronShard one.** Today NeuronShard consolidation (Raven/δ-Mem, `consolidation.rs`) decides *which* shards to freeze, but does not ask *which dimensions within a shard are load-bearing*. The paper's framework gives us a modelless audit: for each `style_weights[i]`, synthesize a constant replacement, measure IoU on a held-out downstream task set, and substitute where IoU is high. This is a **coverage compression** — and it pairs directly with the existing `phase_transition_subspace_phase_gate`'s "intrinsic_dim" measurement.

### 2.4 Fusion — what novel combination does this enable?

The standalone primitive is GOAT (next section). The **fusion** pushes toward Super-GOAT candidate territory but does not clear the bar (Q3 weak). The two highest-value fusions:

**Fusion A — Cognitive Integrity Surrogate Library (R244 × R229 × this paper × R277).**
Today `FaithfulnessProbe` (R244) can *detect* that an attention head is unfaithful (its output doesn't causally bind to behavior), but it cannot *prescribe* what to do about it. Fusing with this paper: when `FaithfulnessProbe` flags a head as "ignored" (low behavior delta on perturbation), the system automatically **synthesizes a program surrogate** (R229 spec compile + this paper's library) and substitutes it — turning a diagnostic into a self-repair. `SmearClassifier` (R277) gates the synthesis: if the head's failure mode is "smear" (hallucinated structural prior), don't bother synthesizing — it's noise; if it's "coherent single feature", synthesis will likely succeed.

**Fusion B — Per-NPC Personality Compression (this paper × R302 FAME × `latent_functor/`).**
For each NPC's committed personality (`CommittedFieldBlend`, R302), the per-NPC HLA state has 8 projection directions. The paper's framework says: many of these directions are functionally replaceable by deterministic constants. A per-NPC audit (run at sleep consolidation cadence) produces a **personality compression report**: "NPC #42's `fear` direction is load-bearing (IoU 0.12 — keep learned), but its `calm` direction is replaceable by a constant (IoU 0.81 — substitute, save 64-dim storage)". This is a **storage win** at MMORPG scale (thousands of NPCs) AND an **audit** (which personalities are actually load-bearing vs decorative).

Neither fusion clears Q3 ("can you finish the sentence: our NPCs do X no competitor can?"). Fusion A is "self-repairing attention" — strong, but FaithfulnessProbe already does the detection half. Fusion B is "compressed personalities" — strong, but CommittedFieldBlend already does the storage half. **Both are amplifier candidates (supergoat_candidates/), not new pillars.**

---

## 3. Verdict: **Gain (revised — was GOAT)**

| Tier | Criteria | Routing |
|---|→|→|
| **Super-GOAT** | Novel mechanism + new capability class + selling point + ≥2 pillars | n/a |
| **GOAT** | Provable gain over existing approach, but not a new class. | n/a (revised down — see §3.3) |
| **Gain** | Incremental improvement, useful but not headline-worthy. | ✅ **Empirical findings + small gate wrapper, recorded for FuncAttn/Percepta consumers. Plan 353 revised to gate-only.** |
| Pass | Not relevant | n/a |

### 3.1 Novelty Gate (Q1–Q4) — revised

**Q1 — No prior art? NO (strong, after revision).** The surrogate representation itself ships as FuncAttn (R257). The programs-as-attention paradigm ships as Percepta (R031/032). The causal validation ships as FaithfulnessProbe (R244). Only the IoU-gate wrapper does not ship.

**Q2 — New class of behavior? NO.** "Attention computed by an external operator" is the existing FuncAttn class. The paper's contribution is empirical (measuring what fraction of real heads are functionally replaceable), not a new class.

**Q3 — Product selling point? NO.** No standalone selling point. The empirical findings (25–40% heads programmable) update how aggressively we apply FuncAttn/Percepta but don't constitute a new capability.

**Q4 — Force multiplier? YES.** The gate connects FuncAttn + FaithfulnessProbe + Plan 287 cadence pattern.

**Verdict: 1/4 YES → Gain, not GOAT.** The initial GOAT verdict was wrong because it underweighted FuncAttn and Percepta as prior art. The vocabulary translation in §2.1 caught FaithfulnessProbe/SpecCompile/SmearClassifier/Rosetta but missed FuncAttn/Percepta — because the paper operates at a higher abstraction level ("arbitrary Python program") than either of those primitives ("closed-form operator", "compiled WASM bytecode"). The user-prompted re-review caught the miss.

### 3.2 Why the verdict dropped from GOAT to Gain

Two prior-art instances were missed in the initial vocabulary translation:

1. **FuncAttn (R257, `funcattn.rs`).** The paper's `π: tokens → attention_matrix` and FuncAttn's `FuncAttn(Q,K,V) = Φ · C* · Ṽ` are the same shape — "attention computed by an externally-supplied operator." FuncAttn's operator is a closed-form Tikhonov solve; the paper's is an arbitrary Python program. The paper is strictly more general in *operator representation* but identical in *primitive shape*. The proposed `Box<dyn SynthesizedAttentionFn>` trait is structurally `dyn FuncAttnKernel`.

2. **Percepta (`katgpt-percepta` crate).** Percepta already does `C program → WASM → lowered bytecode → transformer weights`. The paper does `Python program → runtime callable → attention substitution`. Same programs-as-attention paradigm, different point in the compile↔runtime spectrum. Percepta's gates (`reglu`, `stepglu`, `multiply`, `persist`, `fetch`, `fetch_sum`) are the executable primitives the paper's Python programs compose at a higher level.

With these two layers included, the only novel piece remaining is the **gate** (IoU cheap-proxy → FaithfulnessProbe expensive-validation, Plan 287 cadence pattern). That is a small wrapper around existing primitives — Gain-tier, not GOAT-tier.

### 3.3 Revision log

**2026-06-30 (same day, post-commit):** User prompted "sound like percepta? and a bit of funtional attention?" — corpus re-review confirmed both:
- FuncAttn (`crates/katgpt-core/src/funcattn/mod.rs`, R257 / Plan 286) ships the exact `tokens → attention via external operator` primitive shape.
- Percepta (`katgpt-percepta` crate, R031 / R032 / Plan 064) ships the programs-as-attention paradigm.

Verdict revised GOAT → Gain. The proposed `ProgramSynthesizedHead` primitive is structurally redundant with FuncAttn; Plan 353 is revised to ship only `HeadSubstitutionGate` as a wrapper around FuncAttn's existing trait. This is a **vocabulary-translation failure** in the initial pass: the paper uses "arbitrary Python program" which reads as novel, but the underlying primitive shape (external operator producing attention) is FuncAttn's existing surface. The skill's standing vocabulary block does not include "external operator → attention map" as an explicit translation entry; this case suggests it should ("functional attention", "operator-valued attention", "programmatic attention" → `FuncAttn`, `dyn FuncAttnKernel`).

**Lesson for the research workflow:** the seven Super-GOAT factory modules list in the skill should be checked against ** FuncAttn's existing surface** whenever a paper proposes "replace attention with X". `funcattn.rs` is the canonical home for that primitive shape in this codebase.

---

## 4. What This Is NOT

- **Not a training method.** No gradient descent, no backprop. Fully modelless. The "program synthesis" step uses an external LM offline; the runtime is pure substitution. → stays in katgpt-rs, does NOT redirect to riir-train.
- **Not a replacement for FaithfulnessProbe.** The primitive *uses* FaithfulnessProbe for its causal-validation step. R244 remains the umbrella.
- **Not a Super-GOAT.** The novelty gate fails Q1 (prior art ships) and Q3 (no standalone selling point). No `riir-ai/.research/` guide is created.
- **Not a NeuronShard primitive.** The latent reframing (§2.3) suggests an *application* to NeuronShard consolidation audits, but that is a fusion candidate for a future note, not part of this distillation.

---

## 5. Cross-references

- **R229 ProgramAsWeights** — spec → executable program compilation (the "compile" half).
- **R244 FaithfulnessProbe** — causal intervention paradigm (the "validate" half).
- **R277 SmearClassifier** — hallucinated-feature detector (the failure-mode the paper flags).
- **R295 AC-Prefix** — arbitrary-conditional attention evaluation (the closest attention-mask-shaping neighbor).
- **R233 Attention Matching** — KV compaction via attention-pattern matching (a different application of the same "attention maps are summarizable" insight).
- **R178 Rosetta Neurons** — best-buddies cross-system alignment (the "best program per head" library-search pattern).
- **R302 FAME / CommittedFieldBlend** — per-entity MoE; the latent-reframing target for per-NPC personality compression (Fusion B).
- **Plan 278 / 298 / 313 / 271 / 259** — shipped code for the cousin primitives.
- **Plan 353** — the implementation plan paired with this note.

## TL;DR

Hayes et al. distill attention heads into executable Python programs and show 25–40% can be causally substituted with ~16% perplexity cost. The mechanism is **fully modelless** (offline LM synthesis + runtime substitution, no training). **User-prompted re-review identified two prior-art layers the initial pass missed**: FuncAttn (R257, `funcattn.rs`) ships the exact `tokens → attention via external operator` primitive shape; Percepta (`katgpt-percepta` crate, R031/032) ships the programs-as-attention paradigm. **Verdict revised GOAT → Gain.** The only novel piece is the gate (IoU cheap-proxy → FaithfulnessProbe expensive-validation, Plan 287 cadence pattern). The empirical findings (IoU r>0.9 with substitution cost, 25-40% heads programmable, MAP-Elites library search beats per-head synthesis) are recorded as facts that update how aggressively we apply FuncAttn/Percepta — they do not motivate a new primitive. Plan 353 revised to ship `HeadSubstitutionGate` as a small wrapper around FuncAttn's existing trait, not the redundant `ProgramSynthesizedHead`. Latent-reframing bonus (NeuronShard per-dimension audit) remains a fusion candidate for a future note.

> **PASS-Redirects (synthesis):** EverMind AI [arXiv:2609.33439 "Raven: The Harness of Harnesses for Composable Agentic Intelligence"] — its "semantic gene bank" (one elite harness per edit-category × diagnosed-pathology cell, MAP-Elites quality-diversity with donor-lineage recombination) is this note's library-search pattern applied to agent-harness policies; the healer-side analog is killed by measured evidence, not coverage (riir-refine Bench-047/R135 selection sophistication ≤0.04pp at pool sparsity; set_rerank Issue-121 promotion REFUTED; demonstrability Issue-134 degenerate flat pool axis) — PASS.

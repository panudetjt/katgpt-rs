# Research 605: The Spike, the Sparse and the Sink — Causal Anatomy, Weight-Derived Spike Census, and Delimiter-Sink Constancy

> **Source:** "The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks" — Shangwen Sun, Alfredo Canziani, Yann LeCun, Jiachen Zhu (NYU). [arXiv:2603.05498](https://arxiv.org/abs/2603.05498), 2026-03-05, ICML 2026.
> **Date:** 2026-10-05
> **Status:** Adjudicated — katgpt-rs Issue 919 CLOSED 2026-10-07, nothing promoted: census channel key REFUTED (T2 P@4 = 0 both cells); measured-diagonal KV-exemption NO-GO (T3 — QK-norm squashes per-channel scale upstream of the cache, so the 487 Q8KV-gap premise does not apply on QK-normed archs); delimiter-sink folding NO-GO (T4 — pre-RoPE K context-dominated, cos 0.80–0.89 vs the 0.99 bar). Census sidecars remain the T6 merge-drift artifact at riir-train Issue 614 T4. Verdicts + hashes: HISTORY.md (Issue 919). riir-train Issue 614 recipe levers remain open there.
> **Related Research:** 487 (Massive Activations in HLA — PAS/ISP morphology, sink-position quant prior, Issue 716), 258 (Attention Sink Dual Mechanism NOP/Broadcast, Plan 287), 286 (Attention Drift / depth invariance — magnitude accumulation, Plan 306), 566 (EOT attention priors + sink-margin forecast, Bench 813), 159 (KVarN), 538 (GDN W4A4 quant survival), 200 (quantization outlier collapse), 570 (SAN attention-only FFN deletion — the paper's attention-only ablation is SAN's premise, confirmed at 7B), 095 (MGR multi-gate residuals — "massive activations eliminated", the gating finding's architectural cousin), 531/523 (KV eviction — sink preservation)
> **Related Plans:** 287 (sink_aware_attn — per-call G3 structural FAIL, cached variant deferred), 306 (depth_invariance diagnostic), 585 (usage-rate KV eviction), 135 (Parallax sigmoid attention — sink-free by construction)
> **Cross-ref (riir-ai / riir-train):** riir-ai Issue 716 (sink guard, from 487); **riir-train Issue 614** (recipe levers + sink-ratio health metric, this paper); riir-infer quant lanes (EXL3/GDN census consumers)
> **Classification:** Public

---

## TL;DR

The paper gives the first **causal** (not just descriptive) anatomy of why massive activations and attention sinks co-occur in pre-norm Transformers: one-or-two early FFN **step-up** blocks inject spikes through a **directional quadratic amplifier** (SwiGLU ≈ quadratic form with a rank-1-dominated gain matrix sharing ONE trigger direction), the additive residual stream carries them, late **step-down** blocks cancel them, and **RMSNorm converts spike tokens into sparse near-constant vectors** whose keys collapse into a 1–2-dimensional subspace of `W_K`'s row space — the geometric substrate sinks exploit. Ablations **decouple** the phenomena: sandwich-norm/QKNorm/DynamicTanh kill spikes at parity perplexity while sinks persist (often stronger); head dimension (not head count) is the sink driver; input-conditioned per-channel/per-head gating eliminates sinks; short-context training creates them.

**Distilled for our stack (modelless, inference-time):** the spike structure is **predictable from weights alone** (block-local: only step-up/step-down FFN blocks; channel-local: a few channels; direction-shared: one trigger direction per model) → a cheap data-free **spike census** sidecar; and sink-token representations are **near-constant** post-normalization — position-0 sink rows are exact per-model constants *by construction* (causal mask ⇒ position 0 sees only itself; RoPE at position 0 is identity), and the paper-independent open question is whether *delimiter-class* sink rows fold as constants pre-RoPE. Both feed quant policy (per-channel exemption exactly where outliers live) and KV policy (ShardKV/KVarN/Q8KV substrate — 487's Q8KV absmax gap).

---

## 1. Paper Core Findings

1. **Spike life cycle** (Table 1): step-up blocks inject extreme values in 1–2 early FFN blocks; the additive residual stream (`H_{i+1} = H_1 + Σ F_j(RMSNorm(H_j))`) carries them through intermediate blocks (intermediate block contributions are 2–3 orders smaller); late step-down blocks add the additive inverse. Verified across Llama 2/3, Qwen2.5/3 (7B–14B): e.g. Llama 2 7B step-up 4 / step-down 62.
2. **Directional quadratic amplifier**: under near-identity SiLU gating for spike tokens, each FFN output coordinate is a quadratic form `F_k ≈ h̃ᵀ U_k h̃` with `U_k = Σ_i W_down(k,i)·W_gate(i)·W_up(i)ᵀ`. Spike channels ⇔ anomalously large `‖U_k‖_F`, appearing **exclusively** in step-up/step-down blocks. `S_k = (U_k+U_kᵀ)/2` is rank-1 dominated (`λ⋆ ≫` rest); all spike channels share **nearly the same principal eigenvector s⋆** → inputs aligned with s⋆ fire ALL spike channels simultaneously with fixed ratios (explains co-spiking + invariant inter-channel ratios). Mechanistic correlate in weights: `W_down` has anomalously large entries `W_down(k,i)` with highly collinear `W_gate(i)`, `W_up(i)` rows (consistent with Yu et al. super weights).
3. **What makes a spike token**: >98% of the vocabulary becomes a spike token at position 0 (Table 2) — at position 0 attention collapses to the static linear map `W_VO = Σ_h W_V(h)W_O(h)`, which consistently steers the first token's representation toward s⋆. Delimiter tokens emulate it via **self-sinking** (embeddings near-collinear with RMSNorm's learned scale → inflated post-norm magnitude → self-attention weight → isolated-token linear regime).
4. **Norm is the bridge**: RMSNorm bounds spike coords (≤ √d_model, Theorem B.3), **sparsifies** (only spike channels survive the division), and **near-constifies** (fixed inter-channel ratios ⇒ token-invariant after norm; post-step-up cosine ≈ 1.0). Sink keys `k(s) = W_Kᵀh̃(s)` collapse to the span of **1–2 rows of `W_K`** (vs full `d_head` for normal tokens) and are nearly prompt-invariant. Sink heads = those whose query subspace sits closer to the fixed sink-key subspace than to the non-sink-key subspace (large stable logit gaps).
5. **Ablations (7B, 100B DCLM tokens)**:
   - Sink ratio tracks **optimization health** (LR, β2, tokens — e.g. 200B tokens: 63.3% vs 46.0% baseline; β2=0.999: 20.9%); spike magnitude varies **independently** (weight decay 0 → spikes 12,275 with no sink/perplexity gain).
   - FFN design (GeLU/Linear/attention-only): both phenomena emerge in ALL designs — SwiGLU/GeLU are amplifiers, not prerequisites. (Attention-only: sink ratio 73.9% — cross-validates Research 570's SAN premise at 7B.)
   - **Normalization decouples**: Sandwich norm spikes 3818→520 at sink 44.7% (vs 46.0%); QKNorm measured **on top of sandwich** (row "Sandwich (QK)") spikes→92 at sink 42.0%; DynamicTanh spikes→153 at sink **61.0%** (sinks find non-magnitude strategies).
   - **Head dimension is the sink driver**: d_head 8→128 gives sink ratio 4.1%→46.0% (monotone); at fixed total capacity, fewer/larger heads = stronger sinks + better perplexity; more heads at fixed d_head saturates.
   - **Conditional gating eliminates sinks**: gate conditioned on the current representation, per-channel (sink 4.5%) or per-head (6.4%), at parity perplexity; unconditional/static/positional/token gates do nothing. Sinks = **implicit input-conditioned gating** the model abandons when given a real gate.
   - **Context length drives sinks**: short-context training creates them; long-only distributions collapse sink ratio to 1.2–13%. Sinks are the mechanism for short-range prediction under global attention (corroborates Xiao et al. streaming-heads account).
6. **Each phenomenon independently suppressible at parity perplexity** — their overlap is an artifact of the pre-norm + recipe default, not functional necessity.

**Published landscape (checked §4):** the paper is ICML 2026. Prior art it builds on: Sun et al. 2024 (massive activations, COLM), Yu et al. 2024 (super weights, data-free), Owen et al. 2025 (refined analysis, arXiv:2503.22329), Kaul et al. 2024 (arXiv:2410.17174), Queipo-de-Llano et al. 2025 (arXiv:2510.06477, sinks = compression valleys), Gu et al. 2025 (when sinks emerge). Follow-up found: "Massive Activations as Gradient Regulators in Transformers" (May 2026, arXiv ID unpinned — search timed out twice; cited as landscape, not pinned). Mitigation families (sandwich norm, DynamicTanh, gated attention, KVSink/IntactKV, Hadamard rotations) are all pre-existing — **the paper's novelty is the mechanistic causal account + the decoupling demonstrations**, not the mitigations. Our extractions below are applications of published findings to serving/quant surfaces, deliberately not framed as novel primitives.

---

## 2. Distillation

### 2.1 Extraction E1 — Weight-derived spike census (modelless diagnostic, data-free)

From finding 2, a spike channel k in an FFN block admits a **weight-only** screening statistic. For the dominant rank-1 term (`i` = the high-gain intermediate dim), the quadratic form's gain is approximately `|W_down(k,i)| · ‖W_gate(i)‖ · ‖W_up(i)‖` (the sole singular value of a rank-1 outer product; if one i dominates the sum, `‖U_k‖_F` is dominated by it). **RMSNorm-scale correction:** the paper omits γ because it "can be absorbed into the subsequent weight matrix" — but GGUF stores γ separately (and Gemma uses the `(1+γ)` convention), so the census must fold the stored per-block γ into the gate/up rows or it scores the wrong matrices. Census recipe per FFN block:

1. Scan `|W_down|` for entries above threshold (the anomalous-entry correlate) — O(d_model × d_ffn) per block, seconds offline. **Ternary models (Bonsai-class) need the scale-aware variant**: ternary `W_down` entries are `{−1, 0, +1}` × group scale, so an entry-magnitude scan collapses — score **dequantized weights** or the **group scales** directly.
2. For each candidate (k, i): score `s(k,i) = |W_down(k,i)| · ‖γ⊙W_gate(i)‖ · ‖γ⊙W_up(i)‖`; check collinearity `cos(γ⊙W_gate(i), γ⊙W_up(i))` (the paper reports high collinearity).
3. Spike channels = argmax-k rows; **trigger direction** `s⋆ ≈ γ⊙W_gate(i)/‖γ⊙W_gate(i)‖`; expected block-locality = 1–2 early (step-up) + 1–2 late (step-down) blocks.
4. Validate against one calibration forward (per-channel max at predicted blocks — precision/recall of the census).

Output: a tiny sidecar (blocks × channels × trigger direction), BLAKE3-committable, deterministic, computed from the GGUF weights alone — **no activation data** (the delta vs AWQ/SmoothQuant-class calibration; vs Yu et al. super-weight detection it adds the structural account: block locality, shared trigger, fixed ratios). The screening stat is an **approximation** (dominant-rank-1-term argument; the paper measures exact `‖U_k‖_F`), so T2's validation leg gates any use. Consumers: quant policy (per-channel scale exemption / mixed precision **exactly** on census channels of census blocks — everything else uniform; closes 487's Q8KV per-block-absmax gap), and a pre/post-merge drift check (see E4, low-prior hypothesis).

### 2.2 Extraction E2 — Sink near-constancy: what's exact, what's open (KV substrate)

Two distinct claims must not be pooled:

**(a) Position-0 sink rows are exact constants — but trivially so, independent of this paper.** Under the causal mask, position 0's hidden state depends only on token 0, and RoPE at position 0 is the identity ⇒ the BOS sink's K and V are bit-identical across prompts *by construction* (up to kernel nondeterminism). Prefix caching and ShardKV's lossless sink storage (Plan 147) already exploit this. The memory delta is **one row per (layer, head)** (≈ 1/T of a T-token cache) — "long-context memory shrinks" and "strictly dominates ShardKV" do NOT hold; at best it ties ShardKV and pins that row at full precision, which is the KVSink/IntactKV posture we already have. No new work justified on this half alone.

**(b) Delimiter-class sinks (self-sinking delimiters) — the open question.** These sit at varying positions ⇒ **post-RoPE K cannot fold as a position-independent constant** (RoPE makes K position-dependent). Candidate constant forms: **pre-RoPE K** and **V** (both inherit `h̃(s)`'s near-constancy — the paper measures hidden-state cos ≈ 1.0 post-step-up over 1024 C4 sentences but never quantifies K/V constancy, and never for delimiters), plus a per-sequence sink-position list (delimiters are detectable but positions vary). Exploitation shape if the probe passes: store pre-RoPE sink K/V as per-(layer,head) constants, apply RoPE at the sink's runtime position (a rotation, not a recompute), quantize the constant rows offline at full precision — zero runtime quant error on the rows carrying the logit gaps, sharper than 487's position-prior and KVSink's preserve-at-runtime posture. **Go/no-go probe (T3, rescoped): delimiter sinks only, pre-RoPE K and V measured separately per position class, cos > 0.99 bar.** Constant folding (T4) is demoted from main deliverable — the primary value of this note sits in the census → quant policy (E1/T5).

**Fusion (paper × 487 × 258):** 487 established the sink-position quant prior + the Q8KV per-block-absmax gap; 258 split sinks into NOP (gate it) vs Broadcast (register it); this paper supplies the generator account (step-up FFN quadratic gain) and the near-constant structure the folding hypothesis rests on. Combination: **census sidecar + delimiter-class pre-RoPE sink folding** — a concrete KV/quant posture none of the three states alone, and a cheaper route to what Plan 287's cached variant wanted (folding constants removes per-call classification work entirely).

### 2.3 Extraction E3 — Recipe levers + sink-ratio health metric (riir-train)

Training-side actionable set (filed riir-train Issue 614): (a) **sink ratio (Gu et al.: ε=0.3, T=64) as a cheap optimization-health monitor** — rises with training budget, collapses under mis-specified β2/extreme LR; (b) architecture levers for models we train: sandwich norm (kill spikes at parity; the QKNorm cell was measured **on top of sandwich**, not alone), conditional per-channel/per-head gating (kill sinks AND spikes — Qiu et al. 2025; cross-links MGR Research 095), head-dim choice governs sink strength (small-model d_head ⇒ KV-policy posture), context-length distribution decides whether sinks exist at all (train long-context ⇒ expect sink-free; set KV policy accordingly); (c) the **wd=0 spike-inflation hypothesis (low prior, stated as such)**: the paper's weight decay 0 grew spikes to 12,275 (3×+ baseline) at unchanged perplexity — and wd=0 also *lowered* the sink ratio (33.8% vs 46.0%), so it is not a neutral knob. That result is 100B-token pretraining; whether a low-rank LoRA merge inflates spike channels is a hypothesis with a low prior, not a paper finding. The census (Issue 919's T5 artifact) runs before/after merge at riir-train Issue 614 T4, where merges happen; drop the gate if the first merge shows no inflation.

### 2.4 Extraction E4 — Model-fit expectations for the fleet (probe design input)

- **gemma-2-2b-it**: gemma-2 applies post-attention AND post-FFN norms (a sandwich-norm-family architecture) ⇒ predict **weaker spikes** than a Llama-class 2B, sinks still present. Caveats: gemma-2's FFN is **GeGLU, not SwiGLU** — the quadratic approximation still holds because `GELU(x) ≈ x` for large positive x, but the amplifier constant differs; gemma-2 is NOT among the paper's models (Llama 2/3, Qwen 2.5/3), so everything here is a falsifiable prediction, not a measurement. First census target precisely because it deviates from the paper's baseline; fold the stored γ with the Gemma `(1+γ)` convention (E1 step 2).
- **qwen3.8-27B (GDN+FA hybrid)**: pre-norm FA layers carry full spike anatomy; GDN layers are linear-attention (no softmax ⇒ no sinks) — consistent with 487's PAS/ISP account and 538's GDN quant-survival finding.
- **Bonsai ternary / TernaryDraftModel**: in scope **with the scale-aware census variant** — ternary `W_down` entries are `{−1, 0, +1}` × group scale, so the plain entry-magnitude scan collapses; score dequantized weights or group scales directly (E1 step 1). Bonsai is the priority fleet model, so the scale-aware arm is a named T1/T2 target in Issue 919, not an afterthought.
- Serving note: our prompts start with system/BOS — position 0 = spike token by finding 3, and its K/V constancy is exact by construction (E2a); standard per-prefix caching already includes it. The census makes the *why* explicit and measurable.

### 2.5 Signal-diff per closest cousin (§3.6 discipline)

- **487 (arXiv:2608.12149)**: same phenomenon family, different layer class (hybrid linear-attention morphology PAS/ISP; no weight-derived prediction; quant prior is position-based, not constant-based). This paper adds the causal generator (U_k forms, step-up/down) + the decoupling levers. Complementary — the note cross-links rather than supersedes.
- **258 (arXiv:2606.08105)**: sink *taxonomy* (NOP vs Broadcast) with Lemma-2's "spectral gating OR massive activation" adaptivity regimes. This paper explains *where* the massive activation comes from (step-up FFN quadratic gain) and *what makes the sink keys special* (1–2 dims of W_K row space, near-constant). No overlap in outputs; the delimiter-folding hypothesis is absent there.
- **Plan 147 (ShardKV)**: lossless sink+window storage — already captures the position-0 exactness (E2a); the open value is delimiter-class pre-RoPE folding (E2b), which ShardKV does not attempt.
- **286 / Plan 306 (arXiv:2605.09992 Attention Drift)**: drafter-side magnitude accumulation with `magnitude_slope` diagnostic + post-norm fix. Same *direction* of insight (norm placement governs magnitude pathologies) at a different layer (recursive drafter residual vs pretrained FFN step-up). Cross-cite, no coverage.
- **566 (arXiv:2601.15380)**: prior-logit lane + sink-margin forecast = serving-side sink *compensation*. The paper's head-dim/context-length/gating findings are upstream causes; complementary.
- **588 AWQ / 200 outlier collapse**: AWQ needs calibration activations; the census is weight-only. Outlier-collapse (200) is a security posture at runtime; the census is an offline artifact feeding it. Different signal, different stage.
- **095 MGR**: claimed "massive activations eliminated" via multi-gate residuals — the paper's conditional-gating ablation is the controlled 7B confirmation of that architectural family; MGR's claim gains mechanistic backing.

### 2.6 Game-context reframe (step 4)

The engine serves GGUF LLMs (gemma-2-2b-it, qwen3.8 GDN hybrid, Bonsai ternary) for NPC cognition at 20 Hz. Massive activations dictate where quantization error concentrates (decode quality of NPC dialogue at PQ2_0/Q8), and attention sinks dictate which KV rows are load-bearing under eviction during long multi-NPC context windows. Both paper findings convert directly into cheaper memory/precision allocation on the serving path: census-guided per-channel scales (quant quality at equal bit budget) and constant-folded sink rows (KV memory + eviction safety). Latent-to-raw boundary untouched — these are inference-substrate concerns, no game-state semantics.

### 2.7 Consumer-context reframe (healer, priority #2)

No healer surface consumes transformer-internal activation statistics: rust_perf/kernel_opt corpora consume Rust/GPU code shapes, and the outlier-mitigation kernel family (Hadamard, per-channel scales, exact-widening riders from Batches 176/193/199) is already mined. The paper proposes no kernel and no Rust-level pattern → no miner lead beyond what's on record. One-line answer, checked.

---

## 3. Verdict

**Tiers:**

| Tier | Criteria | Routing |
|---|---|---|
| Super-GOAT | ✗ — the paper is published prior art for its own findings; our extractions are applications, no new capability class | — |
| GOAT | ✗ (CLOSED NEGATIVE 2026-10-07) — the Issue-919 PoC measured both hypothesized gains out: census channel key refuted (P@4 = 0), KV exemption no-go on QK-normed archs, delimiter folding no-go | HISTORY.md (Issue 919) |
| **Gain** | ✓ — actionable, cheap, modelless extractions on surfaces we own (KV stack, quant lanes, training recipes); densest value is sharpening already-shipped sinks/KV/quant work (487's Q8KV gap, Plan 287's deferred cached variant, ShardKV sink storage) | Research note + 2 issues |
| Pass | ✗ — clearly actionable | — |

**One-line reasoning:** published causal anatomy whose engineering extractions (weight-only spike census; delimiter-class sink constancy probe; recipe levers) are cheap, falsifiable, and consume substrate we already ship — file, probe, and only then promote anything toward GOAT.

**MOAT gate (§1.6):** `katgpt-rs` MOAT row covers "Transformer stack (layers/attn/KV/sampling/quant-aware inference)" — the census + sink-folding extractions sit squarely there; precedent 487 filed the same family here. riir-infer consumes via its quant/transformer lanes (cross-ref); riir-train gets the recipe lever set (Issue 614). Routing consistent with the fusion ladder (inference-perf league, priority #3).

---

## 4. Follow-ups

- [x] Research note (this file)
- [x] katgpt-rs Issue 919 — census (γ-folded + Bonsai scale-aware arm) → validation → quant-policy PoC; delimiter-sink pre-RoPE constancy probe — **CLOSED 2026-10-07 NEGATIVE on all three hypotheses** (verdicts: HISTORY.md)
- [x] riir-train Issue 614 — recipe levers + sink-ratio health monitor + wd=0 merge-drift gate (owns the merge gate)
- [-] (Issue 919 verdict: neither condition passed) quant GOAT plan + folding follow-on NOT filed — census channel key refuted, KV exemption no-go (QK-norm), delimiter constancy missed its bar; Plan 287's cached variant stays unaffected (position-0 folding was already exact-by-construction and cost-free)

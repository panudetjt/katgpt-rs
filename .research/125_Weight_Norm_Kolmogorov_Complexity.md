# Research 125: Neural Weight Norm = Kolmogorov Complexity

**Paper:** arXiv:2605.10878 (Musat, ETH Zürich, May 2026)
**Verdict:** ⚠️ THEORETICAL VALIDATION ONLY — no algorithmic gain, no perf hurt, no plan needed

---

## TL;DR

In any fixed-precision regime, the smallest weight norm of a looped neural network outputting a binary string equals the Kolmogorov complexity of that string, up to a logarithmic factor. Weight decay therefore induces a prior matching Solomonoff's universal prior. This is a conceptual proof, not an actionable algorithm.

---

## Key Theorem (The Sandwich Bound)

For computable string s, neural complexity N(s) = min non-zero parameters of fixed-precision looped network outputting s:

```
N(s) ≤ K(s) + c_U          (upper: program → network at unit cost)
K(s) ≤ c_d · N(s) log N(s) + c_d  (lower: network → program at O(log W) per param)
```

Both bounds are **tight** — the permutation witness saturates the log factor.

---

## Lp Collapse in Fixed Precision

In fixed precision (fp16, bf16, int8, int4), every Lp norm raised to its power equals non-zero parameter count up to constants:

```
δ^p · ‖θ‖₀ ≤ ‖θ‖_p^p ≤ M^p · ‖θ‖₀
```

This means L1, squared L2, and any Lp regularizer all control the **same** quantity: sparsity.

---

## What This Validates in Our Stack

| Our Feature | Paper's Relevance | Impact |
|-------------|-------------------|--------|
| **AdamW weight_decay=0.01** | L2 weight decay ≈ Solomonoff prior on outputs | Theoretical validation of existing default |
| **OCTOPUS/SpectralQuant KV** | Quantization strengthens the MDL bias | Confirms quantized KV is implicitly doing MDL |
| **LT2 Looped Inference (Plan 108)** | Looped depth helps low-K targets | Validates looped inference for structured outputs |
| **TF Loop (Plan 136)** | Same — looped depth = more description length efficiency | Confirms training-free loop design |
| **PlasmaPath ternary SIMD** | Ternary = cleanest fixed-precision regime (‖θ‖₁ = ‖θ‖₀ exactly) | Validates ternary as the cleanest MDL correspondence — not an optimality proof |
| **Sparse MLP** | Sparsity ≈ description length minimization | Confirms sparse forward is conceptually right |
| **Bandit + HL pruning** | Pruning = removing non-essential parameters = reducing K | Validates pruning philosophy |
| **LEO/Dual LEO** | Sparsity-inducing priors converge to same Solomonoff prior | All-goals Q-values are low-K structure |

---

## Why No Plan / No Feature Gate

1. **No new algorithm** — the paper proves a theorem about existing regularizers, doesn't propose new training methods
2. **No perf gain** — weight decay is already configured, quantization is already deployed
3. **Constants are large** — authors explicitly state "conceptual rather than predictive at small scales"
4. **No implementation gap** — everything the paper validates is already in our default stack

---

## Cross-Reference: riir-ai (Private)

The paper has a game-specific implication: game LoRA adapters (`lora.bin`, Secret A) in fixed precision encode domain knowledge proportional to their non-zero parameter count. This means:
- Smaller, sparser game LoRA = lower Kolmogorov complexity = more "compressible" game knowledge
- Our weight decay during LoRA training is implicitly selecting the simplest hypothesis that fits game data
- Selling point (softened 2026-09-30, per the arXiv:2609.07755 synthesis below): "Our game adapters carry compressed game knowledge whose fixed-precision norm tracks its Kolmogorov complexity up to a log factor — weight decay biases training toward the simplest sufficient hypothesis."
  - The 2026-05-27 form ("provably optimal … under Solomonoff's universal prior") over-claimed on three counts: the sandwich bound is a two-sided approximation whose constants the authors call "conceptual rather than predictive at small scales"; an induced prior biases the search, it does not guarantee the optimum is reached; and for LoRA-over-frozen-base lanes WD regularizes the adapter delta toward the frozen base, not the composed function's complexity.

This stays in riir-train domain (Research 016; moved from riir-ai in the training split) as it's about game LoRA training configuration.

---

## Conjectures from Paper (Future Watch)

| Conjecture | Our Relevance | Watch Level |
|------------|---------------|-------------|
| Weight norm tracks data complexity K(S) | Could inform LoRA training diagnostics | LOW |
| Flat minima are downstream of MDL | Validates our existing training stability | LOW |
| Effective complexity = ‖θ‖² log ‖θ‖² | Could improve generalization bounds for bandit | LOW |

---

**Date:** 2026-05-27
**Status:** Research only — no plan, no feature gate, no code change

> **PASS-Redirects (synthesis):** Wang, Kevrekidis & Belkin [arXiv:2609.07755 "A Theoretical Analysis of Generalization Dynamics in Neural Networks under Gradient Descent with Weight Decay"] — the dynamical-mechanism companion to this note's static WD=Solomonoff claim, scoped to FULL-WEIGHT GD+WD training: weight decay does not merely select low-K solutions as a prior, it actively contracts the learned function's within-cell prediction variation at per-step rate 2(Σ_{r≤ℓ} s_r + ℓ+2)·ηλ toward an approximate-homogeneity floor 2C(ℓ+2)ε_ah (the Euler residual of the network blocks; inner weights carry the error, outer linear weights are exact) — generalization IS that smoothing, and it is slow (grokking = its delay behind interpolation, t ~ log(V₀/ε)/(2ℓηλ)). Scope note: the contraction law does NOT transfer to LoRA-over-frozen-base lanes (WD there pulls the adapter delta toward the base function, not the network's PV) — lane arithmetic in the riir-train 016 redirect.

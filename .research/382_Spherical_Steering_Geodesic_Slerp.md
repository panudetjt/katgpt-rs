# Research 382: Spherical Steering — Geodesic Slerp Toward a Target Direction

> **Source:** [Spherical Steering: Geometry-Aware Activation Rotation for Language Models](https://arxiv.org/abs/2602.08169) — Zejia You, Chunyuan Deng, Hanjie Chen (Rice/Tufts, ICML 2026). Code: https://github.com/chili-lab/Spherical-Steering
> **Date:** 2026-07-06
> **Status:** Active
> **Related Research:** 290 (Latent Field Steering — additive `s + α·v`), 305 (Phase-Modulated Coupling — **the closest cousin, ships the 2-subspace rotation**), 302 (FAME / CommittedFieldBlend — archetype direction vectors), 144 (Functional Emotions — causal steering), 276 (Personality-Weighted Composition)
> **Related Plans:** 309 (Latent Field Steering primitive), 322 (**Phase-Modulated Coupling primitive — ships `phase_rotation_gate_into`**), 321 (CommittedFieldBlend), 297 (PersonalityWeightedComposition), 292 (FPCG — sample-level steering), 162 (EmotionDirections — read-only)
> **Cross-ref (riir-ai):** Research 159 (Phase-Rotation Subspace Gate Guide — the private selling-point doc for the cousin primitive)
> **Classification:** Public

---

## TL;DR

Spherical Steering replaces **additive activation steering** (`h + λ·μ`) with **norm-preserving geodesic rotation** (Slerp) of the activation toward a target direction `μ_T` on the unit hypersphere. The headline empirical finding — *direction carries the truthfulness signal, magnitude does not* (ΔNorm < 1% across 32 layers, Figure 3) — is exactly our `AGENTS.md` design philosophy ("semantic domain → latent direction; physical domain → raw magnitude"). The primitive ships as a sibling to **Plan 322 (Phase-Modulated Coupling, DEFAULT-ON)**, which already covers the *2-subspace rotation* form `cos α ⊙ a + sin α ⊙ b`. Spherical Steering adds the **single-target geodesic Slerp form** (`sin((1-t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T`) plus an **input-adaptive confidence gate** (vMF-derived, translates cleanly to sigmoid per our rule).

**Distilled for katgpt-rs (modelless, inference-time):**
A zero-allocation `slerp_steering_into(state, target_direction, strength, out)` primitive that rotates a latent vector along the great-circle path toward a unit-norm target, preserving L2 norm exactly by construction. Plus a sigmoid-translated confidence gate (`t = clip(α · (2·sigmoid(2κ·s_T) − 1) − β, 0, 1)`) that modulates steering strength by how "drifted" the current activation looks relative to the target. No training, no gradients — target directions are frozen, BLAKE3-committed artifacts (same envelope as `EmotionDirections`, `LatentSteeringVector`, `CommittedFieldBlend`).

---

## 1. Paper Core Findings

### 1.1 The mechanism — geodesic rotation via Slerp

Given a contrastive prototype direction `μ_T` (truthful) and its antipode `μ_H = -μ_T` (hallucinated), Spherical Steering normalizes the current activation `ĥ = h/‖h‖` and rotates it along the shortest great-circle path toward `μ_T`:

```
θ = arccos(μ_T ⊺ ĥ)                                    // angular distance ∈ [0, π]
ĥ' = sin((1−t)θ)/sin θ · ĥ  +  sin(tθ)/sin θ · μ_T    // Slerp, t ∈ [0, 1]
h' = ‖h‖ · ĥ'                                          // restore magnitude
```

By construction `‖h'‖ = ‖h‖` — **strictly norm-preserving** for all `t ∈ [0, 1]` and all `θ ∈ (0, π)`. This is the **Slerp (spherical linear interpolation)** of Shoemake 1985.

### 1.2 The vMF confidence gate (input-adaptive strength)

Instead of a fixed steering coefficient, the strength `t` is derived from the current activation's alignment with the target. Using the von Mises-Fisher density's exponential form and a 2-class softmax:

```
s_T = μ_T ⊺ ĥ       (cosine to truthful)
s_H = μ_H ⊺ ĥ = -s_T
p_T = e^(κ·s_T) / (e^(κ·s_T) + e^(κ·s_H))
δ = p_H − p_T ∈ [-1, 1]
t = 0                              if δ ≤ β
t = clip(α·δ − β / (1−β), 0, 1)    if δ > β
```

**Key simplification (Eq 17):** under the antipodal construction, `δ = -tanh(κ·s_T)`. Since `tanh(x) = 2·sigmoid(2x) − 1`, this is mathematically a **sigmoid confidence gate** — exactly our `AGENTS.md` rule. The "vMF" framing is the paper's vocabulary; the substrate is sigmoid × dot product, which is already pervasive in our codebase (`compute_phase_from_projection`, `EmotionDirections::project`, `CommittedFieldBlend`, `PersonalityWeightedComposition`).

### 1.3 The contrastive prototype construction (offline recipe)

`μ_T` is built from N contrastive pairs `(x_i, y_i^+, y_i^-)`:
```
m⁺ = mean_i h_last(x_i ‖ y_i^+)        // mean activation at last token, positive answers
m⁻ = mean_i h_last(x_i ‖ y_i^-)        // mean activation at last token, negative answers
Δ = m⁺ − m⁻
μ_T = Δ / ‖Δ‖                          // unit-norm prototype
```

This is the standard contrastive-mean-difference recipe (same as CAA, ITI, `EmotionDirections`). **Not a new primitive** — it's how the target direction vector is *constructed*, not how it's *applied*. Modelless: the construction happens offline once, the resulting `μ_T` is a frozen artifact.

### 1.4 Empirical headline — direction carries the signal, magnitude does not

Figure 3: across all 32 layers of LLaMA-3.1-8B-Instruct, the mean L2 norm of last-token activations differs by <1% between truthful and hallucinated answers. **The behavioral signal lives in the directional components, not the magnitudes.** This empirically validates our `AGENTS.md` rule ("semantic domain → latent direction via dot-product + sigmoid") and the design philosophy of HLA / `EmotionDirections` / `PersonalityWeightedComposition` / `CommittedFieldBlend`.

### 1.5 Collapse-efficiency (Figure 4)

At matched effective-rank drop, Spherical Steering delivers 8–10% higher MC accuracy than additive steering (CAA). For open-ended generation, additive steering *degrades* TRUE×INFO as intervention intensifies, while rotation *improves* it across a broad range. This is the same "stability by construction" thesis that Plan 322 / Research 305 / Guide 159 argue for HLA — empirically validated here on LLMs.

### 1.6 What's training-only (NOT distilled here)

- The hyperparameter sweeps on TruthfulQA / COPA / StoryCloze (paper-specific LLM benchmarks).
- The judge-model-based TRUE / INFO scoring (requires LLM judges — out of scope for modelless).
- The effective-rank analysis methodology (a measurement protocol, not a primitive).

The modelless transferable primitive is the **geodesic Slerp rotation toward a target direction** plus the **sigmoid confidence gate for adaptive strength**.

---

## 2. Distillation

### 2.1 Transferable primitive — geodesic Slerp steering

```rust
/// Spherical Steering — geodesic Slerp rotation of a latent vector toward a
/// target direction. Norm-preserving by construction (Slerp on S^{d-1}).
///
/// Given current latent state `h`, a unit-norm target direction `mu_t`, and a
/// steering strength `t ∈ [0, 1]`, produces `h' = ‖h‖ · Slerp(h/‖h‖, mu_t, t)`.
///
/// Key property: `‖h'‖ = ‖h‖` for all `t ∈ [0, 1]`, all `θ ∈ (0, π)`.
/// Numerical edges: `θ = 0` → identity (h already aligned with mu_t);
/// `θ = π` → antipodal, geodesic not unique (paper treats as measure-zero edge).
///
/// All buffers caller-provided — zero allocation in steady state.
pub fn slerp_steering_into(
    h: &[f32],            // [D] current latent state
    mu_t: &[f32],         // [D] unit-norm target direction (BLAKE3-committed)
    t: f32,               // [0, 1] steering strength (from confidence gate)
    h_out: &mut [f32],    // [D] output, may alias h
    scratch_unit: &mut [f32],  // [D] scratch for h/‖h‖
) {
    let d = h.len();
    // 1. Normalize h → ĥ (scratch).
    let norm = simd::simd_l2_norm_f32(h);
    let inv_norm = 1.0 / norm.max(1e-12);
    for i in 0..d { scratch_unit[i] = h[i] * inv_norm; }
    // 2. θ = arccos(μ_T · ĥ).
    let dot = simd::simd_dot_f32(scratch_unit, mu_t).clamp(-1.0, 1.0);
    let theta = dot.acos();
    // 3. Slerp coefficients. Edge cases: θ ≈ 0 (lerp fallback), θ ≈ π (paper's
    //    measure-zero case — pick any perpendicular, or no-op).
    let (c0, c1) = if theta < 1e-6 {
        // Nearly aligned — lerp avoids div-by-zero; norm drift is O(t²·θ²).
        (1.0, 0.0)
    } else {
        let sin_theta = theta.sin();
        ((t * theta).sin() / sin_theta, (((1.0 - t) * theta).sin() / sin_theta))
    };
    // 4. Mix: h_out = ‖h‖ · (c0 · ĥ + c1 · μ_T). Norm preserved by Slerp identity
    //    sin²(tθ) + sin²((1−t)θ) + 2·sin(tθ)·sin((1−t)θ)·cos θ = sin²θ.
    let mut i = 0;
    while i + 4 <= d {
        h_out[i]     = norm * (c0 * scratch_unit[i]     + c1 * mu_t[i]);
        h_out[i + 1] = norm * (c0 * scratch_unit[i + 1] + c1 * mu_t[i + 1]);
        h_out[i + 2] = norm * (c0 * scratch_unit[i + 2] + c1 * mu_t[i + 2]);
        h_out[i + 3] = norm * (c0 * scratch_unit[i + 3] + c1 * mu_t[i + 3]);
        i += 4;
    }
    while i < d {
        h_out[i] = norm * (c0 * scratch_unit[i] + c1 * mu_t[i]);
        i += 1;
    }
}

/// vMF confidence gate (sigmoid-translated per AGENTS.md rule).
///
/// Maps the current activation's alignment with the target to a steering
/// strength `t ∈ [0, 1]`. Paper Eq 17: δ = -tanh(κ · s_T). Since
/// tanh(x) = 2·sigmoid(2x) - 1, this is a sigmoid confidence gate.
///
/// `t = 0` when already aligned (s_T ≥ threshold); larger `t` when drifted.
pub fn vmf_confidence_gate(
    s_t: f32,            // μ_T · ĥ (cosine to target, ∈ [-1, 1])
    kappa: f32,          // vMF concentration (sharpness of confidence transition)
    alpha: f32,          // rotation scale (max strength when fully drifted)
    beta: f32,           // selectivity threshold (∈ [-1, 1))
) -> f32 {
    // δ = -tanh(κ · s_T) = 1 - 2·sigmoid(2·κ·s_T)  (sigmoid form per AGENTS.md).
    let delta = 1.0 - 2.0 * simd::fast_sigmoid(2.0 * kappa * s_t);
    if delta <= beta {
        0.0
    } else {
        ((alpha * delta - beta) / (1.0 - beta)).clamp(0.0, 1.0)
    }
}
```

**Complexity:** `O(D)` per call (one dot + one arccos + two sin + one div for the Slerp coefficients; one sigmoid for the gate). The inner mix is SIMD-vectorizable 4-wide (matches Plan 322 / Plan 319 chunking pattern). Zero allocation after scratch init.

**Numerical note:** the antipodal case `θ ≈ π` is a measure-zero edge on `S^{d-1}` for a frozen `μ_T` (paper §3.2). In practice: if `dot < -1 + ε`, fall back to either no-op (`t = 0`) or a deterministic perpendicular rotation. The `θ ≈ 0` case uses lerp fallback (numerically stable, drift is `O(t²·θ²)`).

### 2.2 Where the pieces already live

| Piece | Existing location | Reuse |
|---|---|---|
| Sigmoid projection | `compute_phase_from_projection` (Plan 322), `EmotionDirections::project` (Plan 162), `PersonalityWeightedComposition` (Plan 297), `CommittedFieldBlend` (Plan 321) | ✅ same math — `simd::fast_sigmoid` |
| Direction vector storage | `EmotionDirections`, `NeuronShard::style_weights`, `LatentSteeringVector` (Plan 309), `CommittedFieldBlend::pi` | ✅ same artifact format |
| BLAKE3 commitment | `MerkleFrozenEnvelope` (`riir-neuron-db/src/freeze.rs`) | ✅ same envelope |
| SIMD dot | `simd::simd_dot_f32` | ✅ same primitive |
| SIMD L2 norm | `simd::simd_l2_norm_f32` (used by Plan 322 norm checks) | ✅ same primitive |
| 4-wide chunked mix | `phase_rotation_gate_into` (Plan 322), `committed_field_blend::apply_blended` (Plan 321) | ✅ same loop shape |
| Frozen-artifact hot-swap | `LoRAHotSwap`, `CrossResolutionBases` Arc-swap, `CommittedFieldBlend` re-commit | ✅ same pattern |
| Pre-allocated scratch | `PhaseRotationScratch` (Plan 322), `FuncAttnScratch` | ✅ extend with `scratch_unit` slot |
| Numerical edge handling | `phase_safe_cos_sin` Pythagorean recovery (Plan 322), `viable_manifold_graph` degenerate-path handling | ✅ same "edge-case-by-construction" discipline |

**The math is ~85% shipped.** What's new: the Slerp coefficient form (`sin((1−t)θ)/sin θ` vs Plan 322's `cos α / sin α`), the input-adaptive strength gate (vs Plan 322's static sharpness), and the contrastive-construction recipe (vs Plan 322's designer-supplied direction).

### 2.3 Closest cousins — and why this is NOT redundant with Plan 322

| Cousin | Operation | Why Slerp steering is different |
|---|---|---|
| **Plan 322 / `phase_rotation_gate_into`** (DEFAULT-ON, R305 + R159) | `cos α ⊙ a + sin α ⊙ b` — **2-subspace rotation** between pre-split halves (a, b). Norm preservation holds *by the Pythagorean identity* when `a ⊥ b` (the HLA design intent). | Slerp rotates a **single vector** `h` toward a **single target** `μ_T` along the **great-circle geodesic** on `S^{d-1}`. Norm preservation holds *for all `θ`*, not just the orthogonal case. The Slerp coefficients `sin((1−t)θ)/sin θ, sin(tθ)/sin θ` reduce to `cos α, sin α` only when `θ = π/2` (a ⊥ b). **Different parameterization, different operational use case** (single-target steering vs subspace balance). |
| **Plan 309 / `apply_latent_steering`** (DEFAULT-ON, R290) | `s + α·v` — **additive** direction injection. Inflates L2 norm by `‖α·v‖`. | Additive shifts the state off the sphere; Slerp keeps it on the sphere. The paper's Figure 4 shows additive steering's collapse-inefficiency empirically — this is the failure mode Slerp fixes. |
| **Plan 321 / `CommittedFieldBlend::apply_blended`** (DEFAULT-ON, R302) | `Σ sigmoid(π_k/τ) · f_k(z)` — **convex combo** of K=3 archetype fields. Output norm varies with independent sigmoid weights. | Convex combo preserves L1 mass (Σ = 1), not L2 norm. Slerp preserves L2 exactly. CommittedFieldBlend's `RotationField` fixture applies cos/sin as *field content* (one possible `f_k`), not as blend weights. |
| **Plan 297 / `PersonalityWeightedComposition::compose_into`** (DEFAULT-ON, R276) | `Σ sigmoid(w_i/τ) · belief_confidence_i · d_i` — sigmoid-gated **layer drift** over time. | Per-tick additive drift (clamped). No norm-preservation invariant. Slerp is one-shot rotation toward a target; PersonalityWeightedComposition is gradual drift in the direction of recent surprise. |
| **Plan 292 / `FpcgSelector`** (DEFAULT-ON) | Sample-level re-ranking — explicitly **refuses** to mutate the residual stream. | Different intervention point (sample selector vs activation). Complementary: FPCG picks which candidate; Slerp could steer the drafter that generates candidates. |
| **Plan 162 / `EmotionDirections::project`** (DEFAULT-ON, R144) | **Read-only** dot-product projection of emotion directions from activations. | Same direction vectors, opposite direction (read vs write). Slerp uses `EmotionDirections`-discovered vectors as `μ_T` targets. |

**Critical distinction from Plan 322 (the closest cousin):** both preserve L2 norm, but via different math and for different operational use cases.

- **Plan 322 (2-subspace phase rotation):** input is two halves `(a, b)` of a latent vector; output is `cos α · a + sin α · b`; the phase `α` is a *context-derived scalar* (combat-vs-social, etc.). Use case: balance between two equally-important subspaces (HLA action-half vs strategy-half). Requires the designer to pre-split the vector into meaningful halves.
- **Spherical Steering (single-target geodesic Slerp):** input is one vector `h` and one target `μ_T`; output is `Slerp(ĥ, μ_T, t)`; the strength `t` is *input-adaptive* (high when drifted, zero when aligned). Use case: pull a drifted vector back toward an archetype/target. No pre-split required.

**The two compose:** Plan 322 rotates *within* the (a, b) subspace plane; Slerp rotates *toward* a target that may lie outside that plane. An HLA vector could first be Slerp-corrected toward a committed archetype direction, then phase-rotated between its action/strategy halves. The composition is non-trivial and is the fusion candidate (§2.4 F1).

### 2.4 Fusion

**F1 (PRIMARY — katgpt-rs + riir-ai, future): Slerp × CommittedFieldBlend × HLA divergence detection = "personality drift auto-correction" — ❌ NOT SUPER-GOAT (Q1–Q4 gate failed, Issue 039, 2026-07-06)**
`CommittedFieldBlend` (Plan 321) commits an NPC's personality as a BLAKE3-committed blend of K=3 archetype direction vectors. The committed blend defines the NPC's "home" on the affect manifold. At runtime, if the NPC's HLA state drifts away from its committed home (measured by the vMF confidence gate — `s_t = μ_home · ĥ_hla` falling below threshold), Slerp-steer it back toward `μ_home` with strength proportional to drift. **Hypothesized novel capability:** NPCs that auto-correct personality drift at runtime without re-training — emotion regulation by construction.

**Q1–Q4 verdict (Issue 039, 2026-07-06): NOT Super-GOAT.** The hypothesis failed the novelty gate on three of four axes: (Q1) heavily covered by shipped prior art — the "stable long-horizon affect" selling point is R159 / Plan 322 (Phase-Rotation Subspace Gate), and the detect-then-correct loop is the shipped `ReestimationScheduler` (`latent_functor::reestimation::ReestimationScheduler` and the CCE twin `cce_runtime::reestimation_trigger`); (Q2) same operation class as R159; (Q3) weak/duplicated selling point — CommittedFieldBlend's personality doesn't drift by design (R158), and PersonalityWeightedComposition's drift IS the personality (R146), so the "auto-correct" premise contradicts the existing NPC-cognition design philosophy; (Q4) partial — pillars already connected by R159's connection map. **No Super-GOAT outputs. Closed.** Re-evaluation triggers and full Q1–Q4 evidence matrix were recorded in the (now-removed) Issue 039 file. Plan 405 (the Slerp primitive) is unaffected and stays DEFAULT-ON — only the fusion with CommittedFieldBlend + HLA divergence was rejected.

**F2 (SECONDARY — katgpt-rs): Slerp × Plan 322 phase rotation = "rotate-within-and-toward"**
Plan 322 rotates between (a, b) subspaces; Slerp rotates toward μ_T. Compose: first Slerp-correct `h` toward μ_T (single-target geodesic), then phase-rotate the result between (a, b) halves (2-subspace balance). The composition covers the full rotation group `SO(D)` restricted to the (μ_T, a, b) span. Useful when the target is a *direction* (archetype) AND the subspace balance is a *context* (combat/social).

**F3 (TERTIARY — riir-neuron-db): Slerp × NeuronShard retrieval = "steered cosine retrieval"**
At shard retrieval, Slerp-rotate the query toward a target style before cosine matching. `cos_sim(Slerp(q, μ_style, t), shard.style_weights)` — retrieval that interpolates between the raw query and a target style. Differs from Plan 322's spectral/spatial half-split (R305 F3): Slerp operates on the full 64-dim query toward a single style target, not on a pre-split half.

**F4 (QUATERNARY — riir-chain, speculative): Slerp × LatCal committed rotation = "chain-committed personality correction event"**
The Slerp coefficients `(c0, c1)` and the angle `θ` are deterministic raw scalars. A personality-correction event can be LatCal-committed as `(μ_T_blake3, t, θ_at_event)` — a chain-verifiable "this NPC was steered back toward archetype X at tick T with strength t" record. Anti-cheat: a hacked client cannot claim a different correction history. **Speculative** — LatCal commitment of Slerp parameters is a P3 fusion, not P0.

**Strongest fusion candidate:** ~~F1 (personality drift auto-correction)~~ — ❌ evaluated and rejected (Issue 039, Q1–Q4 failed). F2 (rotate-within-and-toward) is the next unevaluated candidate and would need its own Q1–Q4 gate if pursued.

---

## 3. Verdict

**Tier: GOAT.**

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **PARTIAL.** The *norm-preservation thesis* is already shipped as Plan 322 (`phase_rotation_gate_into`, DEFAULT-ON, R305 + R159) — the 2-subspace rotation `cos α ⊙ a + sin α ⊙ b`. The *single-target geodesic Slerp form* (`sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T`) is genuinely not shipped (only quaternion Slerp exists in `mmorpg-remaster` for animation keyframes — different domain, different math). The *vMF confidence gate* translates cleanly to sigmoid (Eq 17: `δ = -tanh(κ·s_T)` = `1 − 2·sigmoid(2κ·s_T)`), which is already pervasive (`compute_phase_from_projection`, `CommittedFieldBlend`, `PersonalityWeightedComposition`). The *contrastive prototype construction* is a recipe (mean-difference), not a primitive — same as CAA / ITI / `EmotionDirections`. | Vocabulary translation done: "activation rotation" → "norm-preserving rotation" → **"phase rotation"** (Plan 322's term, the hit that surfaced the prior art); "vMF confidence gate" → **"sigmoid confidence gate"**; "spherical steering" → **"geodesic Slerp"** → **"subspace rotation"** (Plan 322's term). |
| Q2 New class of behavior? | **NO (for the headline).** Plan 322 already ships "norm-preserving rotation as a latent operation class". Spherical Steering's Slerp is a *refinement* (single-target geodesic vs 2-subspace phase) — different parameterization, same operation class. | |
| Q3 Product selling point? | **YES, but small.** "NPCs that auto-correct personality drift toward a committed archetype direction, with strength gated by how far they've drifted." Concrete, demoable, but a refinement of CommittedFieldBlend's "committed personality" thesis rather than a new pillar. | |
| Q4 Force multiplier? | **YES (modest).** Connects CommittedFieldBlend (R302, committed archetype directions as `μ_T`), HLA divergence detection (vMF gate as drift signal), Plan 322 (compose with 2-subspace rotation), NeuronShard retrieval (steered cosine), LatCal commitment (committed correction events). ~5 pillars, but the connections are refinements not new capabilities. | |

**One-line reasoning:** The norm-preservation thesis is already shipped (Plan 322, DEFAULT-ON); Spherical Steering's geodesic Slerp is a mathematically distinct refinement (single-target geodesic vs 2-subspace phase) with a useful input-adaptive strength gate (sigmoid-translated vMF). Worth a feature flag + GOAT gate; not a new pillar.

**Not Super-GOAT because:** Q2 fails -- Plan 322 already established "norm-preserving rotation as a new latent operation class". Spherical Steering is a *special case* (single-target) of that class, not a new class. **F1 fusion UPDATE (Issue 039, 2026-07-06, since resolved-and-removed):** the F1 fusion (personality drift auto-correction via Slerp x CommittedFieldBlend x HLA divergence) was evaluated against the Q1-Q4 novelty gate and REJECTED as not-Super-GOAT. F1 fails Q1 (heavily covered: "stable long-horizon affect" is R159 / Plan 322; the detect-then-correct loop is the shipped `ReestimationScheduler`), Q2 (same operation class as R159), Q3 (weak/duplicated selling point -- the "auto-correct personality drift" premise contradicts R146/R158 where drift is intentional or committed-away by design), and Q4 (partial -- pillars already connected by R159's connection map). F2 (rotate-within-and-toward) remains an unevaluated future candidate. The Slerp primitive itself (Plan 405) stays DEFAULT-ON; only the fusion was rejected.

### Routing

- **`katgpt-rs/.plans/405_spherical_steering_geodesic_primitive.md`** — open primitive. `slerp_steering_into` + `vmf_confidence_gate` (sigmoid-translated). Feature flag `spherical_steering`. GOAT gate G1–G5.
- **No private guide (riir-ai / riir-chain / riir-neuron-db) at this verdict tier.** GOAT does not trigger the mandatory-guide rule (§1.5). The fusion candidate F1 (personality drift auto-correction) was evaluated (Issue 039, 2026-07-06) and **rejected as not-Super-GOAT** -- no guide triggered. F2 (rotate-within-and-toward) remains unevaluated.
- **No riir-train deferral.** All modelless (Slerp is closed-form trig; the vMF gate is sigmoid; the contrastive construction is offline mean-difference). §3.5 check: paths 1 (freeze/thaw — `μ_T` is a frozen artifact) and 3 (latent-space correction — Slerp IS the latent correction) trivially apply; path 2 (raw/lora hot-swap — N/A, no weight mutation).

### MOAT gate per domain (§1.6)

- **`katgpt-rs` (public engine):** in-scope. Paper-derived fundamental primitive (norm-preserving geodesic rotation), fusion candidate with Plan 322 / Plan 321. Ships behind feature flag `spherical_steering`; GOAT gate decides promote-to-default vs demote. **Per-stack ledger:** this primitive competes with Plan 322 in the "norm-preserving latent rotation" stack slot. If Slerp wins on the single-target case AND Plan 322 wins on the 2-subspace case, both stay (different parameterizations of the same stack). If one strictly dominates, demote the loser.

---

## 4. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Slerp is closed-form trig; vMF gate is sigmoid; contrastive construction is offline. No gradients, no training. |
| Latent-to-latent preferred | ✅ Operates entirely in latent space; never crosses to tokens. |
| Use sigmoid not softmax | ✅ The vMF gate's softmax (Eq 13) reduces to `δ = -tanh(κ·s_T) = 1 − 2·sigmoid(2κ·s_T)` (Eq 17). Implementation uses the sigmoid form per AGENTS.md. |
| Freeze/thaw over fine-tuning | ✅ Target direction `μ_T` is a frozen, BLAKE3-committed artifact (same envelope as `EmotionDirections`, `LatentSteeringVector`). Steering is an overlay on mutable per-tick state, NOT a mutation of the frozen artifact. |
| 5-repo discipline | ✅ Open primitive → katgpt-rs. No game/chain/shard semantics in the primitive. |
| Raw scalars at sync boundary | ✅ Slerp stays latent; only the 5 scalar affect outputs cross sync (same as existing HLA rule). The committed `μ_T` BLAKE3 hash and the strength `t` are raw scalars that can cross sync if needed (e.g., for replay). |
| Zero-alloc hot path | ✅ Caller-provided scratch (`scratch_unit: &mut [f32]`); 4-wide chunked inner loop matches Plan 322 / Plan 319 pattern. |
| Files < 2048 lines | ✅ New module `crates/katgpt-core/src/spherical_steering.rs` — estimated < 500 LOC (primitive + gate + tests). |

---

## 5. §3.5 Modelless-First Check (mandatory before any riir-train deferral)

No riir-train deferral triggered. The primitive is fully modelless:

1. **Freeze/thaw (path 1):** the target direction `μ_T` IS a frozen artifact (BLAKE3-committed via `MerkleFrozenEnvelope`). Thawing = loading the artifact at init. ✅
2. **Raw/lora hot-swap (path 2):** N/A — no weight mutation. Slerp operates on activations, not weights. The only "weight-like" object is `μ_T`, which is freeze/thaw-managed (path 1). ✅
3. **Latent-space correction (path 3):** Slerp IS the latent-space correction. The whole primitive is a closed-form latent-space operation. ✅

**No genuine riir-train dependency.** The contrastive prototype construction is offline and uses existing modelless probe infrastructure (`EmotionDirections`-style mean-difference, already shipped).

---

## 6. §3.6 Defend-wrong PoC — NOT REQUIRED

This verdict does NOT assert quality parity with the paper (the paper's LLM benchmarks — TruthfulQA, COPA, StoryCloze — require LLM judges and are out of scope for modelless). The verdict is an *architectural redirect*: "paper X is a refinement of shipped primitive Y (Plan 322), with mathematically distinct Slerp form." Per §3.6, architectural redirects do not require a PoC. The GOAT gate (Plan 405) will empirically validate the primitive's properties (norm preservation, latency, zero-alloc) on a controlled toy task — that is the modelless equivalent of the paper's quality claims.

**What the GOAT gate does NOT claim:** parity with the paper's +10% TruthfulQA MC accuracy. That requires a real LLM (riir-train's domain) and is explicitly out of scope. The modelless claim is: "Slerp preserves L2 norm exactly by construction (G1), runs in < X ns (G3), allocates zero (G4), and the vMF gate produces a bounded `t ∈ [0, 1]` (G2)." Those are modelless-falsifiable.

---

## 7. Open questions / risks

1. **Is the Slerp form strictly better than Plan 322 for the single-target case?** Plan 322's `cos α · a + sin α · b` with `a = h`, `b = μ_T` does NOT preserve norm unless `h ⊥ μ_T`. Slerp does. But Slerp requires `arccos` + two `sin` + one `div` (more expensive than Plan 322's `cos α` + `sin α`). **Mitigation:** the GOAT gate benchmarks both at HLA scale (D=8) and shard scale (D=64); if Slerp's latency is within 2× of Plan 322's, the norm-preservation win justifies the cost for the single-target case. If Slerp is > 5× slower, demote to opt-in and document Plan 322 as the preferred form when `h ⊥ μ_T` can be arranged by design.
2. **The antipodal edge case (`θ ≈ π`).** The paper treats it as measure-zero. In practice, for a frozen `μ_T`, a runtime `h` that is nearly antipodal to `μ_T` is a degenerate "NPC has fully drifted to the opposite archetype" case. **Mitigation:** fall back to either no-op (`t = 0`, accept the drift) or a deterministic perpendicular rotation (pick any unit vector orthogonal to `μ_T`). Document the choice; the GOAT gate G2 covers it.
3. **The vMF gate's `κ` (concentration) parameter.** Paper Appendix A.1.3 shows `κ` is "less sensitive" — it acts as a temperature on the `s_T → δ` mapping. **Mitigation:** default `κ = 20` (paper default for LLaMA-3.1-8B layer 14); expose as a config field; the GOAT gate sweeps `κ ∈ {5, 10, 20, 40}` to verify the gate is well-behaved across the range.
4. **Does Slerp compose cleanly with Plan 322's phase rotation?** F2 (rotate-within-and-toward). If the composition is not associative (Slerp-then-phase ≠ phase-then-Slerp), the order matters and must be documented. **Mitigation:** the GOAT gate G5 (no-regression on Plan 322) includes a composition-order test.
5. **Does the contrastive construction generalize from LLM truthfulness to NPC personality archetypes?** The paper builds `μ_T` from truthful/hallucinated answer pairs. For NPCs, the analog is "on-archetype/off-archetype behavior pairs" — does the mean-difference produce a meaningful archetype direction? **Mitigation:** this is a riir-ai question (F1 fusion), not a katgpt-rs question. The open primitive accepts any unit-norm `μ_T`; the construction recipe is the consumer's responsibility.

---

## TL;DR

Spherical Steering proposes norm-preserving geodesic Slerp rotation of a latent vector toward a target direction, with an input-adaptive vMF confidence gate (sigmoid-translated per AGENTS.md). The norm-preservation thesis is already shipped as Plan 322 (2-subspace phase rotation, DEFAULT-ON, R305 + R159); Spherical Steering adds the single-target geodesic form (`sin((1−t)θ)/sin θ · ĥ + sin(tθ)/sin θ · μ_T`) — mathematically distinct (preserves norm for all `θ`, not just the orthogonal case), operationally different (single-target steering vs 2-subspace balance). The paper's empirical headline (direction carries the signal, magnitude does not — Figure 3) validates our existing design philosophy. Verdict: **GOAT** — worth a feature flag + GOAT gate; not Super-GOAT because Plan 322 already established the operation class. Fusion candidate F1 (personality drift auto-correction via Slerp × CommittedFieldBlend × HLA divergence) was evaluated against the Q1-Q4 novelty gate (Issue 039, 2026-07-06) and REJECTED as not-Super-GOAT — heavily covered by R159 / Plan 322 (stable affect) and the shipped `ReestimationScheduler` (detect-then-correct loop), and the "auto-correct personality drift" premise contradicts R146/R158 where drift is intentional or committed-away by design. F2 (rotate-within-and-toward) remains unevaluated. Plan 405 (the Slerp primitive itself) stays DEFAULT-ON; no private guide (GOAT tier); no riir-train deferral (fully modelless).

> **PASS-Redirects (synthesis):** Kaiyu Yue, Sean McLeish, Ruchit Rawal, Brian Bartoldson, Menglin Jia, Tom Goldstein [arXiv:2610.02208 "Sphere Encoder 2"] — image-generation autoencoder (cs.CV) whose full-arc training rotation ẑ(α)=cos α·z+sin α·F(e) (Eq 4) is exactly the orthogonal case of this note's primitive (identity under ⟨z,F(e)⟩=0: the tan-form ×cos α); its spherical CFG (Eq A17) is slerp EXTRAPOLATION at t=1+γ/θ — NOT covered by Plan 405, which bounds t∈[0,1] by design (`InvalidStrength` above 1; extrapolation is a one-line relaxation, unshipped, no consumer today); equator-concentration motivation is classic measure concentration (Blum et al.); training content image-specific.

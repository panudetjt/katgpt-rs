# Research 307: Fourier Neural Operators — Practical Perspective (Survey)

> **Source:** [Fourier Neural Operators Explained: A Practical Perspective](https://arxiv.org/abs/2512.01421) — Duruisseaux, Kossaifi, Anandkumar (Caltech + NVIDIA), arxiv 2512.01421v2, 22 Jan 2026, 96pp
> **Date:** 2026-06-25
> **Status:** Done
> **Related Research:** 219 (DEC operators — spectral diff in DEC vocabulary), 257/290 (FUNCATTN), 291 (Cross-Resolution Spectral Transport — **the headline FNO primitive, already shipped**), 039 (SpectralQuant eigenbasis KV), 269 (ChiARoscuro spectral salience), 100 (EGA spectral attention)
> **Related Plans:** 251 (DEC), 308 (KARC Fourier basis), 242 (FFT-smoothed potential fields), 265 (LatCal spectral fixed-point), 310 (Cross-Resolution Spectral Transport — DEFAULT-ON)
> **Classification:** Public
>
> **PASS-Redirects (synthesis):** Csillag, Schuller, Dall'Antonia, Guibas, Velho, Novello [arXiv:2606.16926 "Functional Gradient Descent with Adaptive Representations"] — FGD with adaptively refined gradient representations (trees / frequency grids / 3D grids), converging through the approximation via a relative-error certificate; the optimization half of this note's function-space family — verdict **Pass**: its PDE-via-energy-minimization arm is the training-side sibling of iFNO's adaptive-mode expansion (same trigger shape), its modelless extractions are consumerless in-stack (no field-fitting lane), and its width/rank-staging residue is covered by the measured k_V rung law, which ships as a gate-passed opt-in key (riir-train Issue 459 T2, `sample_measured`; the hand-set ladder remains `train()`'s default) — wiring it is Issue 459's residue, independent of this paper.

---

## TL;DR

FNO paper is a **96-page practitioner guide** for the existing Fourier Neural Operator framework, not a new architecture. Its headline inference primitive — **resolution-invariant spectral transport** between function spaces via frozen Fourier bases — **already ships** as our Super-GOAT `cross_resolution_transport` (Research 291 → Plan 310, elevated DEFAULT-ON 2026-06-23). Most of the paper's *interesting* parts (PINO physics-informed training, UQNO conformal uncertainty, RNO recurrence, iFNO incremental mode expansion) are training procedures → **riir-train**. The narrow modelless gaps left for us: **Fourier continuation** (closed-form least-squares periodic extension for non-periodic domains), **standalone spectral differentiation** (multiply by `(ik)^m` in Fourier basis — currently embedded in DEC `exterior_derivative` only), and **Tucker/HOSVD tensor factorization** (TFNO weight compression — currently only 2D `thin_svd` ships). All three are incremental **Gain**-tier primitives.

**Distilled for katgpt-rs (modelless, inference-time):**
- **Resolution-invariant spectral transport**: project `s ∈ R^{d_src}` to k-dim spectral via frozen `Φ_src^T`, reconstruct at `R^{d_dst}` via frozen `Ψ_dst`. **SHIPPED** as `cross_resolution.rs::transport_cross_resolution_into`.
- **Spectral differentiation**: `F{∂^m_x f}(k) = (ik)^m f̂_k` — multiply Fourier coefficients by `(ik)^m`, IFFT back. **EMBEDDED** in DEC `exterior_derivative` (where the basis is the cell incidence matrix) but not exposed as a standalone FFT-based primitive.
- **Spectral interpolation / super-resolution**: zero-pad in frequency domain, IFFT to denser grid. **SHIPPED** as part of `cross_resolution.rs` (different mechanism — basis projection rather than zero-padding — but same capability class).
- **Spectral downsampling / low-pass truncation**: keep first K modes, zero the rest, IFFT. **SHIPPED** as `crates/katgpt-core/src/flow/fft.rs::fft_smooth` (Nyquist cutoff for LEO potential fields) and `freq_bandit.rs::token_stream_spectrum` (DFT up to Nyquist).
- **Spectral loss**: `Σ_k |F(ĝ)(k) − F(g)(k)|²` — modelless diagnostic. **NOT shipped** as a standalone metric.
- **Fourier continuation** (FC-Legendre, FC-Gram, spectrum-optimization): closed-form least-squares polynomial extension making non-periodic signals periodic. **NOT shipped**.
- **Tucker / HOSVD tensor factorization** for frozen weight compression (TFNO): generalization of SVD to higher-order tensors. **PARTIAL** — `subspace_phase_gate::thin_svd_into` ships 2D SVD; full N-mode Tucker does not.

---

## 1. Paper Core Findings

### 1.1 What the paper IS (and isn't)

This is a **reference / survey paper**, not a new architecture. It consolidates the FNO framework as implemented in `NeuralOperator 2.0.0` (the official NVIDIA/Caltech PyTorch library), with emphasis on practitioner pitfalls and clarifying common misconceptions. There is **no novel mechanism introduced here** — every architectural piece (SpectralConv, ChannelMLP, FNOBlock, TFNO, SFNO, Geo-FNO, GINO, OTNO, PINO, UQNO, RNO, iFNO) cites a prior paper.

### 1.2 Transferable modelless primitives (the parts not requiring backprop)

The paper is training-heavy in its main contributions, but several underlying primitives are pure modelless inference operations:

| Primitive | Math | Modelless? |
|-----------|------|-----------|
| **Spectral differentiation** | `F{∂^m_x f}(k) = (ik)^m · f̂_k` | ✅ Free (1 FFT + scalar mult + 1 IFFT) |
| **Spectral interpolation / resampling** | zero-pad coefficients in freq domain, IFFT to denser grid | ✅ Free |
| **Spectral downsampling** | truncate to first K modes, IFFT to coarser grid | ✅ Free |
| **Power spectrum diagnostic** | `P_k = |X_k|²` | ✅ Free |
| **Spectral loss** | `Σ_k |F(ĝ) − F(g)|²` | ✅ Free |
| **Fourier continuation (FC-Legendre / FC-Gram)** | fit polynomial of degree `2d−1` across left+right boundary vectors, append as periodic extension | ✅ Closed-form, no gradient |
| **Spectrum-optimization extension** | solve `min ||f̃||²_{H^s}` over extension samples — closed-form linear least squares | ✅ Free |
| **Tucker / HOSVD weight factorization (TFNO)** | `W ∈ C^{K×I×O} ≈ Σ_core ×_1 U^(K) ×_2 U^(I) ×_3 U^(O)` — frozen factorization applied at inference | ✅ Free (matmul-only at inference) |
| **Spectral projection for divergence-free** | represent velocity field as curl of potential (incompressibility by construction) | ✅ Free — equivalent to DEC `coexact_flow` |

### 1.3 Training-side contributions → riir-train

These all require gradient descent; they route out of this workflow:

- **PINO** (Physics-Informed Neural Operator) — pretraining + instance-wise fine-tuning with PDE residual loss
- **UQNO** — residual operator trained with quantile loss + split conformal calibration (the conformal *step* is modelless, but the residual operator itself is trained)
- **RNO** (Recurrent Neural Operator) — GRU-style gated recurrence over function spaces
- **iFNO** (Incremental FNO) — adaptive mode expansion triggered by training-loss stagnation or spectral-energy ratio
- **Multi-resolution curriculum training**, **autoregressive rollout training** (pushforward loss), **multi-objective loss balancing** (SoftAdapt, ReLoBRaLo)
- **Hyperparameter tuning** (`n_modes`, `n_layers`, `hidden_channels`)

Per skill §3.5 modelless-unblock check: NONE of these can be unblocked via freeze/thaw or raw/lora hot-swap or latent correction — they are genuinely training procedures. → riir-train.

### 1.4 Practitioner-pitfall insights worth knowing

Even where the paper adds no new mechanism, several practical lessons are worth recording for our `cross_resolution_transport` and `KarcForecaster<FourierBasis>` consumers:

1. **`n_modes` must stay below Nyquist** — including the bandwidth-broadening effect of nonlinearities (σ(x)=x² doubles bandwidth). Aliasing in training folds back as spurious low-frequency energy. → applies to our `freq_bandit` Nyquist cutoff.
2. **Spectral downsampling beats stride downsampling** for any signal where spatial correlations matter — stride introduces discontinuities, phase shifts, spurious patterns. Our `crates/katgpt-core/src/flow/fft.rs` low-pass before grid ops already does this right.
3. **Non-periodic domains need Fourier continuation**, not naive FFT — Gibbs phenomenon at boundaries corrupts derivatives. **This is a real gap in our stack** (we have FFT for periodic LEO grids; non-periodic latent fields would need FC).
4. **Sinusoidal embeddings for scalar parameters** (amplitude or frequency modulation) beat constant inputs because they activate all Fourier modes from layer 1. Relevant to `KarcForecaster` constant-parameter handling.
5. **ChannelMLP restores high-frequency content** that SpectralConv truncates — the FNO pattern is *spectral global + channel-wise local*, not spectral-only. (Our `funcattn` ships the spectral side; the channel-MLP complement is implicit in downstream consumers.)
6. **Tensor decomposition (Tucker) reduces SpectralConv params 5–20×** at fixed expressivity — relevant to `NeuronShard` cold-tier compaction (`ShardCompactor`) where the current `n_am_queries=1` mode collapses to rank-1 (see Plan 319 G5 split verdict).

---

## 2. Distillation

### 2.1 Transferable primitive (modelless subset)

The paper's modelless content reduces to **fourier-basis algebra on discrete samples**:

```
given x ∈ R^N (real samples on uniform grid of N points):

X = FFT(x)                          // N complex coefficients
∂^m_x x  = IFFT( (ik)^m ⊙ X )      // spectral differentiation, k = freq index
x_up     = IFFT( zero-pad(X, M) )   // super-resolution to M > N grid
x_down   = IFFT( X[0..K] ++ zeros ) // low-pass + downsample
```

This is the substrate. Everything else in the paper layers trained weights on top of it.

### 2.2 Where the pieces already live (BOTH layers, ALL repos)

| Layer | Artifact | Match to FNO primitive |
|---|---|---|
| Code | `katgpt-rs/crates/katgpt-core/src/cross_resolution.rs` | **`transport_cross_resolution_into` = FNO super-resolution / cross-resolution spectral transport.** Frozen BLAKE3-committed basis pair `(Φ_src, Ψ_dst)`, k-dim spectral projection, reconstruct at any dim. DEFAULT-ON. **The headline FNO inference primitive.** |
| Code | `katgpt-rs/crates/katgpt-core/src/funcattn/mod.rs` | FUNCATTN = spectral attention with frozen bases (Research 257/290). SpectralConv-equivalent. |
| Code | `katgpt-rs/crates/katgpt-core/src/flow/fft.rs` | `fft_smooth` with Nyquist-cutoff low-pass on potential fields. FNO spectral downsampling for periodic grids. |
| Code | `katgpt-rs/crates/katgpt-pruners/src/freq_bandit.rs::token_stream_spectrum` | DFT up to Nyquist. FNO power-spectrum diagnostic. |
| Code | `katgpt-rs/crates/katgpt-spectral/src/spectral_kv_cache.rs` | Eigenbasis KV compression (Research 039). SpectralConv applied to KV cache. |
| Code | `crates/katgpt-dec/src/operators.rs` | `exterior_derivative` (d), `codifferential` (δ), `hodge_laplacian` (Δ). FNO spectral differentiation **in DEC vocabulary** — d on a periodic cell complex IS spectral differentiation. |
| Code | `katgpt-rs/crates/katgpt-core/src/subspace_phase_gate.rs::thin_svd_into` | 2D SVD (one-sided Jacobi). TFNO's Tucker is the N-mode generalization — partial coverage. |
| Code | `riir-ai/crates/riir-games-civ/src/civ/map_tick/karc.rs` (Plan 308) | `KarcForecaster<FourierBasis>` — Fourier delay-basis ridge regression. Fourier-basis forecasting already shipped. |
| Code | `riir-chain/src/encoding/latcal_fixed.rs::LatCalSpectralFixed` | Fixed-point Fourier coefficients `(freq × 10^6, amp × 10^6, phase × 10^6)` for chain commitment. FNO coefficients crossing the sync boundary. |
| Code | `riir-chain/src/catchup/shard_quorum.rs::spectral_diversification` | Cosine-based shard ensemble diversity. FNO spectral mode coverage analog. |
| Code | `riir-chain/src/consensus/curator_bridge.rs::verify_spectral_shard` | Spectral shard condition-number bound (BLAKE3-committed integrity). |
| Notes | `katgpt-rs/.research/291_*` + `katgpt-rs/.plans/310_*` | **The FNO paper's headline inference primitive already framed + shipped as our Super-GOAT.** Promoted to DEFAULT-ON 2026-06-23. |
| Notes | `katgpt-rs/.research/219_*` (DEC) + `.plans/251_*` | DEC operators — the Stokes-calculus substrate. Maps FNO spectral differentiation to `exterior_derivative`. |
| Notes | `katgpt-rs/.research/296_*` (Stokes vocabulary crosswalk) | Documents that paper-vocabulary grep for "stokes/divergence/fokker-planck" returns ZERO hits because the math ships as DEC operators. **Same failure mode would occur for "FNO/spectral convolution"** — ships as `cross_resolution`/`funcattn`. |
| Notes | `katgpt-rs/.research/039_*` (SpectralQuant) | Eigenbasis KV compression — FNO SpectralConv applied to KV cache. |

### 2.3 Latent-space reframing (mandatory §3)

Re-cast FNO's mechanism against each Super-GOAT factory module:

- **HLA per-NPC latent state** (riir-ai/`hla/`): FNO's "channels" = HLA's 8 affect channels (valence/arousal/desperation/calm/fear + 3). SpectralConv's mode-wise channel mixing `R(k) ∈ C^{d_v × d_v}` = funcattn-style projection in latent space. Crowd-scale HLA field over a zone grid → natural FNO input (2D field of 8-ch latents). Spectral transport moves affect across resolution tiers (zone-aggregate ↔ per-NPC). **Already wired** via `apply_field_to_crowd` (Plan 309 latent steering).
- **`latent_functor/`**: spectral convolution = a Fourier-domain functor application. `transport_cross_resolution_into` IS the asymmetric-basis functor. **SHIPPED.**
- **`cgsp_runtime/`**: curiosity signal projected spectrally → band-curiosity (which Fourier band is the NPC exploring?). Latent reframing valid but not yet wired; niche.
- **LatCal fixed-point commitment** (riir-chain): `LatCalSpectralFixed` already commits Fourier `(freq, amp, phase)` as i64 × 10^6 for cross-platform determinism. FNO coefficients cross the sync boundary as raw fixed-point scalars — the bridge pattern holds. **SHIPPED (Plan 265).**
- **`NeuronShard` `style_weights[64]`** (riir-neuron-db): TFNO Tucker factorization would compress the 8×8 reshaped weight matrix as `Σ_core ×_1 U^(K) ×_2 U^(I) ×_3 U^(O)`. **Currently `subspace_phase_gate::thin_svd_into` ships the 2D SVD view;** the N-mode Tucker generalization is the missing piece. Fusion with `ShardCompactor` (cold-tier) and `semantic_axes` (runtime personality extraction) is the Super-GOAT angle — but it's narrow.
- **DEC Stokes-calculus** (`dec/`): FNO spectral differentiation = DEC `exterior_derivative` on a periodic grid. The discrete d operator on a periodic 2D grid multiplies by `(i k_x, i k_y)` in the Fourier basis. **The machinery ships in DEC form.** A standalone FFT-based `spectral_differentiate` would be a thin specialized wrapper, useful only when the cell complex is regular and periodic (where DEC's full machinery is overkill).

### 2.4 Closest cousins (fusion candidates)

The three closest existing primitives, ranked:

1. **`cross_resolution_transport` (Plan 310 / Research 291)** — **strictly stronger than the FNO paper's standalone super-resolution** because it composes with the FUNCATTN cross-domain operator `C ∈ R^{k×k}` (transport between semantic domains *and* resolutions in one 4-matrix product). FNO only handles resolution.
2. **`funcattn` (Plan 286)** — the SpectralConv analog with frozen bases.
3. **DEC `exterior_derivative`** — the spectral-differentiation substrate in DEC vocabulary.

**Fusion idea (novelty TBD — Gain at best):** Tucker-HOSVD factorization of `NeuronShard::style_weights[64]` reshaped as `(K=8, I=8, O=8)` tensor, integrated with the existing `semantic_axes` SVD and `ShardCompactor` cold-tier compaction. Could yield 5–20× weight compression (TFNO §6.1 numbers) without losing the personality axes. But this is a single narrow optimization on the shard path — Gain-tier, not Super-GOAT.

---

## 3. Verdict

### **Gain** — narrow modelless primitives; plan-only, behind feature flags.

**One-line reasoning:** The FNO paper's headline inference primitive (resolution-invariant spectral transport between function spaces via frozen Fourier bases) **already ships as our Super-GOAT `cross_resolution_transport`** (Research 291, Plan 310, DEFAULT-ON since 2026-06-23). The paper's training-heavy contributions (PINO, UQNO, RNO, iFNO) are training procedures → riir-train. Three narrow modelless gaps remain: Fourier continuation (for non-periodic latent fields), standalone FFT-based spectral differentiation, and Tucker/HOSVD tensor factorization for `NeuronShard` compression. None is a new capability class; none is a Super-GOAT.

### Routing

- **No Super-GOAT guide** (novelty gate fails Q1: prior art is overwhelming — `cross_resolution_transport` already covers the headline primitive).
- **No katgpt-rs plan opened in this session** — the three Gain-tier gaps are listed below as candidate plans; user decides whether to open them.
- **Training parts → riir-train** (one-line note): PINO physics-informed training, UQNO conformal residual-operator training, RNO gated recurrence training, iFNO incremental mode-expansion training, multi-resolution curriculum, autoregressive pushforward loss, multi-objective loss balancing (SoftAdapt / ReLoBRaLo).

### Candidate plans (Gain-tier, deferred to user decision)

| # | Plan | Primitive | Where |
|---|------|-----------|-------|
| 1 | Fourier continuation for non-periodic latent fields | FC-Legendre / FC-Gram / spectrum-optimization — closed-form least-squares periodic extension | `katgpt-rs/crates/katgpt-core/src/spectral/continuation.rs` (new), feature `fourier_continuation` |
| 2 | Standalone FFT-based spectral differentiation | `spectral_differentiate(x, order)` — multiply by `(ik)^m`, IFFT back. Specialized wrapper around FFT for periodic uniform grids (DEC `exterior_derivative` is the general-case operator) | `katgpt-rs/crates/katgpt-core/src/spectral/differentiation.rs` (new), feature `spectral_differentiation` |
| 3 | Tucker / HOSVD tensor factorization for shard compaction | N-mode generalization of `thin_svd_into` — applies TFNO weight compression to `NeuronShard::style_weights[64]` reshaped `(8,8,8)`. Fuse with `ShardCompactor` cold-tier and `semantic_axes` runtime SVD | `katgpt-rs/crates/katgpt-core/src/linalg/tucker.rs` (open primitive) + `riir-neuron-db/src/shard_compactor.rs` (integration), feature `tucker_factorization` |

Each is small, single-feature, GOAT-gated. None is urgent — current consumers (`cross_resolution_transport`, `funcattn`, DEC) cover the headline use cases.

### Novelty gate (Q1–Q4)

| Q | Answer | Notes |
|---|--------|-------|
| **Q1 No prior art?** | ❌ NO | `cross_resolution_transport` (DEFAULT-ON) covers the headline FNO inference primitive; DEC covers spectral differentiation; `fft_smooth` covers low-pass; `LatCalSpectralFixed` covers sync-boundary commitment. Only Fourier continuation + Tucker/HOSVD are genuinely missing, and both are narrow. |
| **Q2 New capability class?** | ❌ NO | All three candidate plans are incremental refinements of shipped capability classes. |
| **Q3 Product selling point?** | ❌ NO | Cannot finish "our NPCs do X that no competitor can" with anything from this paper — the relevant X (cross-resolution latent transport) already shipped and is already a selling point. |
| **Q4 Force multiplier (≥2 pillars)?** | ❌ NO | Fourier continuation touches only DEC + spectral-transport; Tucker touches only neuron-shard compaction. |

**Verdict: 0/4 YES → Gain, not Super-GOAT.** No guide required, no Super-GOAT plan required.

---

## 4. Cross-references

- `katgpt-rs/.research/291_cross_resolution_spectral_transport_open_primitive.md` — **the FNO headline primitive, already shipped**
- `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` — DEC substrate (spectral diff in DEC vocabulary)
- `katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` — vocabulary-translation lesson that applies here too (FNO ↔ `cross_resolution`/`funcattn`/DEC)
- `katgpt-rs/.research/039_SpectralQuant_Calibrated_Eigenbasis_KV_Compression.md` — SpectralConv applied to KV cache
- `katgpt-rs/.plans/310_cross_resolution_spectral_transport_primitive.md` — execution record for the headline primitive
- `katgpt-rs/.plans/308_karc_kolmogorov_arnold_reservoir.md` — `FourierBasis` delay-basis forecaster
- `katgpt-rs/.plans/242_Fourier_Smoothed_Potential_Fields_LEO.md` — FFT low-pass on LEO potential grids
- `riir-chain/.research/004_LatCal_Committed_Karc_Readout.md` + `riir-chain/src/encoding/latcal_fixed.rs::LatCalSpectralFixed` — fixed-point Fourier coefficients crossing the sync boundary
- → **riir-train** for PINO, UQNO residual-operator training, RNO, iFNO incremental training, multi-resolution curriculum, pushforward loss, SoftAdapt/ReLoBRaLo loss balancing

## TL;DR

FNO practical-perspective paper is a 96-page practitioner guide for the existing NVIDIA/Caltech FNO framework — **no new architecture**, mostly training-side content (PINO/UQNO/RNO/iFNO → riir-train). The headline inference primitive — resolution-invariant spectral transport via frozen Fourier bases — **already ships as our Super-GOAT `cross_resolution_transport`** (Research 291 / Plan 310, DEFAULT-ON since 2026-06-23), and in fact our version is strictly stronger because it composes cross-resolution with cross-domain transport in one 4-matrix product. DEC `exterior_derivative` covers FNO spectral differentiation in DEC vocabulary; `fft_smooth` covers low-pass; `LatCalSpectralFixed` covers sync-boundary Fourier commitment. **Three narrow Gain-tier gaps remain** (Fourier continuation for non-periodic latent fields, standalone FFT-based spectral differentiation, Tucker/HOSVD for `NeuronShard` compaction) — none is a new capability class, none clears the Super-GOAT novelty gate (0/4 YES), none warrants a guide. Verdict: **Gain**, plan-only, user decides whether to open the three candidate plans. The vocabulary-translation lesson from Research 296 (Stokes↔DEC) applies again here: paper-vocabulary grep for "FNO/spectral convolution" returns partial hits only because we shipped the same math under `cross_resolution`/`funcattn`/DEC names.

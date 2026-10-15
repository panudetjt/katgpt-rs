# Plan 618: `gmm_support` — Density-Ratio Support Gate Primitive

**Status:** LANDED (substrate + first consumer) 2026-10-05 — GOAT ALL PASS, Bench 908; opt-in per the no-default-consumer rule. From Research 604 (arXiv:2610.02126 Local Support Learning), GOAT verdict (downgraded from SUPER-GOAT at round 1) 2026-10-03. Modelless; feature `gmm_support` (implies `factorized_action`). First consumer LANDED: riir-reflex Issue 066 (closed — reflex `9ecface`, Bench 124: wired + measured, verdict NOT CERTIFIED at the ρ=30 per-axis posture, lever stays opt-in/report-only; the many-label directional signal + the consumer var-floor re-pin recorded there). Remaining T12 guide rows (riir-refine ScopeModel sibling, riir-engine `LoRAHotSwap` dispatch) are their own repos' calls.

## Goal

Ship the continuous-latent two-density support gate as a katgpt-core primitive by CONSUMING and EXTENDING shipped substrate (verdict round 1 — no parallel GMM type): extend the `RegionSubspaceField`-class Gaussian machinery with per-component variances + a logsumexp mixture density + EM refinement; seed EM from the shipped deterministic `fit_codebook_kmeans_into`; revive `JlProjectionMatrix` at E=256; add the pos/neg ratio gate, EMA smoother, and the offline excess-mass certification validator. Consumers: reflex fused-abstain density half (riir-reflex Issue 066), healer continuous ScopeModel sibling, riir-engine `LoRAHotSwap` gated dispatch (guide `riir-ai/.research/394`), vessel-packed per-phase artifacts.

- [x] T0 — substrate consume-or-extend map (REWRITTEN per verdict round 1): `RegionSubspaceField` (Plan 416 — centroids, log_pi, SHARED psi_inv, sigmoid membership gates, BLAKE3 commitment) · `fit_codebook_kmeans_into` (factorized_action — deterministic Lloyd + k-means++, no GD) · `JlProjectionMatrix` (shard_embedding — deprecated at m=8 for JL-bound breakage, Issue 139; SOUND at E≥256) · `data_probe/gaussianity.rs` (`sketched_gaussianity`, feature `gaussianity_probe`) · `GaussianMixtureSpace` (fixture-grade, NOT a gate). Re-grep at implementation time for anything newer. **DONE 2026-10-05 — re-grepped at implementation; no newer substrate (the `distributional_steering::lse` twin is pinned-by-spec in `mixture::lse` — its module feature is not implied for one helper; the `stale_residual::logsumexp_shifted` twin likewise private + gated).**
- [x] T1 — mixture-density extension of the RegionSubspaceField class: **`DiagGmm<E, K>`** (`src/gmm_support/mixture.rs`) — per-region diagonal VARIANCES, `loglik_mixture` in logsumexp form, zero-alloc eval (stack scratch `[f32; K]`), precomputed `log_norm`; commitment REUSING `compute_field_commitment`'s convention (LE f32, pinned order; the per-component variance block where the shared psi_inv sat; no loadings block — a diagonal GMM has no subspace). **Fit-space law honored in-module (doc + PRE-CHECK citation).**
- [x] T2 — EM refinement **`fit_diag_gmm` + `EmConfig`** (`src/gmm_support/em.rs`): E-step log-space responsibilities; M-step closed-form (means, per-region variances, weights) with f64 accumulators and pinned reduction order; seeded FROM `fit_codebook_kmeans_into` (the feature implies `factorized_action`); Jeffreys-smoothed weights keep components alive (λ=0.5); iteration-capped + tol early-stop; determinism pinned by the artifact hash (test + bench G1).
- [x] T3 — **`JlProjector<D, E>`** (`src/gmm_support/projector.rs`): revived at E ≥ 32 as a const-generic type with a COMPILE-TIME shape guard (associated-const `assert!` — instantiating E < 32 is E0080 at monomorphization, verified); seeded Rademacher; `project(&[f32; D], &mut [f32; E])`; **packed sign-bit storage** (one `Vec<u64>` at construction — `E·ceil(D/64)` words, apply slice-only zero-alloc; add/sub only, `1/√E` once per row); hash pinned over `D || E || words`; distortion + norm tests; the Issue-139 history in the module docs. (Packed storage is a `Vec`, not a nested array — stable Rust forbids generic const-expr array lengths; the cold-path allocation is documented.)
- [x] T4 — **`SupportGate<E, K>`** (`src/gmm_support/gate.rs`): holds pos+neg densities (`neg` shared per shape class at the consumer's discretion); `log_ratio`, `open` (`ℓ > 0`), `confidence = sigmoid(ℓ/τ)` via `exact_sigmoid`; unfitted/empty = closed-always (`log_ratio = −∞`, confidence 0, the `CorpusDistanceGate` precedent); τ consumer-pinned with `DEFAULT_TAU = 1.0` named POC-scale.
- [x] T5 — **`GateSmoother`** (`gate.rs`): per-stream EMA `s ← α·g + (1−α)·s`, neutral 0.5 start, `decides(threshold)` + `is_open()` at 0.5; plain struct, never synced, never a weight.
- [x] T6 — Certification validator (`src/gmm_support/certify.rs`): `certify` (deficit = wrongly-closed rate on held-out A; excess = Φ_neg-weighted mass by MC sampling); **two-sided canary, leak BY CONSTRUCTION** — `LinearDiscriminatorGate::fit` + extrapolation probes planted far out along its positive normal; `canary_fire_rate` returns `CanaryDidNotFire` (the exit-2 class) unless the discriminator fires on ALL probes. Test + bench G5 pin: discriminator fires, density gate closed on the same probes, density excess < leaky excess.
- [x] T7 — Bound-acceptance check (`bound_acceptance`): `‖P̂−P‖₁ = E_P[|p̂/p − 1|]` MC-estimated in closed form on known-P fixtures; arms at low-D (E=8) AND the E=64 band; monotonicity arm via DIRECT error injection (fit means drifted toward the reference center, δ ∈ {0.5, 1.0} — L1 grows strictly, excess within slack; sample starvation measured NOT monotone and was replaced, the datum recorded in Bench 908). **Measured: the Φ_neg-weighted excess/L1 constant is 0.67–1.4× across the fixture family → the check ships as `excess ≤ bound_multiplier·L1 + slack`, multiplier 2.0 (measured ceiling + headroom), consumer-pinned.** No real-corpus volume claims.
- [x] T8 — Deterministic corpus fixture (`src/gmm_support/fixture.rs`): fixed-seed `build_fixture` (in-support GMM blob with KNOWN true params on the radius-`separation` sphere — bounded-shell geometry so off-domain probes sit genuinely outside; generic reference; off-domain B; discriminator extrapolation probes), CLT-12 libm-free sampler (bit-identical cross-platform), `sample_mixture_into` the ONE sampler home (certify consumes it).
- [x] T9 — Bench `bench_gmm_support_gate`: G2 with the projection COUNTED — per-request posture **8529 ns = 0.67×** the decision-pass proxy (first-measurement bar ≤ 1.0×); per-matrix at LAYER scope, added ≤ 12% bar (measured 0.1% on the scalar-MAC-loop proxy — a LOWER bound; FLOP arithmetic ≈ 6.0% of a Qwen2.5-1.5B-shaped layer printed beside it); K∈{2,16,32}×E=256 table (485/3717/7318 ns); G4 alloc canary (0 allocs); `black_box`-defended timed regions.
- [x] T10 — GOAT gate run: **Bench 908 ALL PASS** (G1/G2a/G2b/G4/G5). Promotion REFUSED per the no-default-consumer rule (the repo's own law — `gaussianity_probe` precedent): `gmm_support` stays opt-in; default promotion rides the first consumer's GOAT. UQ Report-the-Floor scoped to the consumer A/B (the primitive has no outcome space) — recorded in the bench, carried by riir-reflex Issue 066's own task row.
- [x] T11 — Doc: `.docs/09_feature_catalog/opt_in_features.md` entry 142 + README feature section + the plan status (this edit) + feature-count claims bumped 670→671 at all five sites (`count_features.py` green). PASS-Redirects: `grep 2610.02126` hits Research 604 + this plan + the catalog entry.
- [x] T12 — Consumer wiring handoffs (separate repos, NOT in this plan's compile surface): reflex Issue 066 (abstain density half) — **DONE 2026-10-05** (reflex `9ecface`, Bench 124: the density half wired behind reflex's `density_gate` feature + the paired report-only A/B; verdict NOT CERTIFIED — the consumer A/B's own gate task decided opt-in/report-only, exactly the no-default-consumer rule's promotion-rides-a-GOAT shape); riir-refine ScopeModel sibling (guide row) + riir-engine gated `LoRAHotSwap` dispatch (guide row, `riir-ai/.research/394`) — recorded, their own repos' calls (open guide rows, not deliverables of this plan).

## Non-goals

- No in-model conditional-LoRA forward (that is riir-infer-laya feature material, gated on riir-train Plan 439's pilot).
- No fp16/int8 GMM quantization this pass (recorded as a follow-up row in Research 604; artifact bytes already KB-class).
- No streaming moment drift-sentinel (B3) — record as future issue if a serving consumer asks.

## Risks

- EM determinism across platforms (float summation order): pin the reduction order; artifact hash is the regression tripwire.
- τ/κ defaults are POC-scale (the `contrastive_scope` precedent): consumers re-pin; the bench asserts sensitivity only, not universal optima.

## Measured constraint (consumer-side evidence, 2026-10-04)

**The GMM must be fit in the PROJECTED space — raw hashed-bag embedding space REJECTS
Gaussianity universally.** Measured at the first consumer (riir-reflex Issue 066
PRE-CHECK, reflex `bd18b86`, `examples/gaussianity_probe.rs` consuming
`data_probe/gaussianity.rs::sketched_gaussianity` on the real seat-corpus embedding
populations — whole-corpus + per-label, deterministic, two runs byte-identical):

- raw 256-d hashed-bag space: **0/154 per-label pools accept** (median score 0.0000 on
  every suite) — a diagonal-GMM fit there models a distribution the data provably is
  not;
- JL projection (Rademacher ±1/√k) at **k=64 rescues it: 253/254 pools accept,
  median ≥ 0.95, k-stable across 64/32/16**;
- whole-corpus pools stay REJECTED on many-label suites (massive 60 labels,
  banking77 77) even projected — the label-mixture multimodality the per-label
  components exist to model (this is T1/T2's component structure working as designed,
  not a blocker).

Consequences for this plan: T1/T2 fit ONLY post-T3-projection inputs (the `JlProjector`
is a hard prerequisite of any real-data fit, not an optional accelerator); T6/T7
validators and T9 benches that use real-corpus-like fixtures must project first or
their Φ references are meaningless; T12 consumer handoffs carry the same law (a
consumer fitting in raw space will measure a gate that is wrong by construction).

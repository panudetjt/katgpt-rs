# Bench 908 — `gmm_support` GOAT gate (Plan 618 / Research 604)

**Status:** COMPLETE — ALL GOAT GATES PASS (2026-10-05).

- **Primitive:** `katgpt-core` feature `gmm_support` (opt-in, implies
  `factorized_action`) — `DiagGmm` (mixture-density extension of the
  `RegionSubspaceField` class), `fit_diag_gmm` (EM seeded from
  `fit_codebook_kmeans_into`), `JlProjector` (JL revived at E ≥ 32, packed
  sign bits), `SupportGate` + `GateSmoother`, the T6/T7 certification
  instruments, deterministic fixtures.
- **Source:** Research 604 (arXiv:2610.02126 "Local Support Learning",
  Ben-Kish et al.) — verdict round 1 GOAT, plan `.plans/618_gmm_support_gate.md`.
- **Run:** `cargo bench -p katgpt-core --features gmm_support
  --no-default-features --bench bench_gmm_support_gate -- --nocapture`
  (release, the bench-profile default).

## Box state (the G2 law)

M3 Max, AC (battery 80% charging), loadavg **3.27 / 3.07 / 2.83** at run
start (`uptime`: up 19 days). A LOADED box (sibling sessions) — every
latency below is a REGRESSION CEILING, not a quiet-box figure.

## Gates

| Gate | Verdict | Reading |
|---|---|---|
| G1 correctness/determinism | **PASS** | EM two-run BLAKE3 identity; frozen-decision identity across twin artifacts; unfitted = closed-always, confidence exactly 0. Lib tests 19/19 (closed-form K=1 anchor, two-mode responsibilities, tamper-sensitive commitment, JL distortion/norms, smoother math, bad-input refusals). |
| G2a request posture | **PASS** | one projection (D=256→E=64) + one pair-eval (K=16) = **8529 ns = 0.67×** the decision-pass proxy (128-option × 256-dim matvec argmax). Bar: first-measurement ≤ 1.0× (the plan's "set the bar from the first measurement"). The proxy is CONSERVATIVE — the real reflex decision pass also pays chunking + hashing, so 0.67× OVERSTATES the gate's share. |
| G2b layer posture | **PASS** | added **0.1%** measured vs the ungated adapter path at D=1536/E=256/K=32/r=128 with per-input-group gate sharing (q/k/v/gate/up share one projection; o and down carry their own). The scalar MAC-loop proxy is 10–100× slower per MAC than a BLAS GEMV, so the measured % is a LOWER bound; the deterministic **FLOP arithmetic is ≈ 6.0%** of a Qwen2.5-1.5B-shaped layer (3.15M gate ops vs 52.3M layer MACs — the Research 604 arithmetic). Bar ≤ 12% (plan T9). |
| G2 throughput | (table) | pair-eval ns at E=256: **K=2: 485 ns · K=16: 3717 ns · K=32: 7318 ns** — linear in K as designed (no early-exit tricks). |
| G4 alloc | **PASS** | 0 allocations / 1024 (projection + pair-eval) iterations, counting-allocator canary live. |
| G5 certification | **PASS** | the two-sided canary: the linear-discriminator control fires on **ALL** extrapolation probes (leak demonstrated, instrument alive); the density gate stays **closed** on the same probes; the density gate's Φ_neg-weighted excess strictly beats the leaky control's; deficit (wrongly-closed rate on held-out support) < 0.2; the fixture-scoped bound holds. |

## Honest notes

1. **The T7 bound's constant is not 1 under Φ_neg weighting.** The paper's
   App-E relation bounds Lebesgue volume (unmeasurable under unbounded
   Gaussians in 256-D). The implementable form measures reference-weighted
   excess mass, whose ratio to the L1 estimate measured **0.67–1.4×**
   across the fixture family — the check ships as
   `excess ≤ bound_multiplier·L1 + slack` with the multiplier defaulted to
   the measured ceiling (2.0, headroom), consumer-pinned and documented in
   `CertifyConfig`. The instrument's teeth are the monotonicity arm +
   the leak canary, not the absolute bar.
2. **Monotonicity flatness recorded.** Across the injected-error ladder
   (fit means drifted toward the reference center at δ ∈ {0.5, 1.0}) the
   L1 estimate grows strictly but the excess moves only within slack —
   the Φ_neg-weighted excess is dominated by the fixture's baseline
   overlap, not the injected mean error. The plan's "bigger ε ⇒ bigger
   excess WITHIN SLACK" is satisfied as written; strict growth is not
   claimed. (Sample starvation was measured NOT monotone — a starved
   k-means seed still lands reasonable centroids while its inflated
   variances make the gate more conservative — and replaced by the
   direct injection arm.)
3. **UQ Report-the-Floor** rides the first consumer's A/B (riir-reflex
   Issue 066 carries it as its own task item): the primitive ships a gate
   signal with no outcome space of its own — a conformal-naive comparison
   needs the consumer's prediction task. Recorded here, not dodged.
4. **`deficit_rate` semantics** (fixed during the landing): the report's
   deficit is the WRONGLY-CLOSED fraction on held-out support
   (`1 − open rate`) — an early draft reported the open rate under the
   deficit name; caught by the module tests before any external consumer.
5. **Determinism scope**: same inputs → identical artifacts on one
   platform (BLAKE3 pin); cross-platform bit-identity of the FIT path is
   limited by the platform `exp`/`ln` in the logsumexp (the plan's risk
   row). The EVAL path and the packed-sign projector are pure IEEE
   add/sub/mul (+ one exactly-rounded `sqrt` at construction). The
   fixture sampler is deliberately libm-free (CLT-12 sum of uniforms —
   the `floor_harness` precedent) so fixture bytes are cross-platform
   bit-identical.

## Fit-space law (binding, measured at the first consumer)

The GMM pair is fit/eval'd in the **JL-projected** space (E ≈ 32–64
suffices; k-stable 64/32/16), never raw hashed-bag D-space — raw space
REJECTS Gaussianity universally (0/154 per-label pools; riir-reflex Issue
066 PRE-CHECK, reflex `bd18b86`). Rides the module docs, the plan's
"Measured constraint", and the consumer issue.

## Verdict

ALL GATES PASS. `gmm_support` stays **opt-in** per the no-default-consumer
rule — promotion rides the first consumer's GOAT (riir-reflex Issue 066:
the fused-abstain density half; its A/B carries the Report-the-Floor law).

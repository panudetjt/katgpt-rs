# Bench 906 — LoopCD GOAT: Recurrent-Depth Contrast Guidance on a Non-Saturated Looped Fixture

**Plan:** [617_loop_depth_contrast_guidance](../.plans/617_loop_depth_contrast_guidance.md) (T3.1–T3.5)
**Date:** 2026-10-03
**Source paper:** arXiv:2610.02185 (Apple, decoding looped transformers) + twin prior art arXiv:2609.24196
**Verdict: NEGATIVE — the class is 0-for-3 on this stack (Bench 847, Bench 850, this bench). Guidance is never strictly better than unguided at any swept posture; at the pre-registered posture it is destructive. The feature stays opt-in, dead-or-narrow.**

---

## Box state (the G2 law: load class beside every number)

Windows 11, i7-13700K (16 threads), 31.8 GiB RAM, AC power. **LOADED**: the
sibling agent's riir-train 607 laya training was running on the GPU
throughout (CPU data pipeline active on the same box); all wall-clock A/B
numbers below are noise-band (per-round ratios spread 0.50–1.59) and the
G2 gate deliberately asserts only the STRUCTURAL bounds (head-pass counts,
allocation counts, the +5%/+60% ceilings), which are load-invariant. The
GPU was NOT used by this bench (CPU-only model). No latency number in this
record is a quiet-box measurement.

## Provenance + ordering (T3.1's frozen-fixture law)

The fixture and its non-degeneracy assertions were committed BEFORE any
guided arm ran: `tests/loopcd_fixture_gate.rs` at `6c4b5b5f7`, then Phase 2
at `c6eb24d34`, then this bench. The fixture is hand-constructed modelless
weights (no RNG, no training): prefix vote-counting over 142 enumerated
(len, imbalance) cases; the decision axis grows via the ReLU MLP
(`out[e2] = δ·(x̂0−x̂1)`, δ=0.2 the measured clock); the head bias gap Δ
encodes decision closeness. Measured non-degeneracy (the Bench-847 lesson
mechanized):

- depth→accuracy slope: acc(12) − acc(1) = **0.50** (acc 0.5 → 1.0; crossings spread over depths 4–9) — floor 0.25;
- close-decision stratum at depth 12: **19 cases** (|margin| < 0.5) — floor 5;
- disagreement floor: **71/142 (50%)** of cases have `argmax(z₁) ≠ argmax(z₁₂)` — floor 10;
- finiteness, byte-determinism, query-row-never-wins: pinned.

A tuning note committed in the fixture: δ=1.0 collapsed every crossing into
depths 1–6 (close stratum 0 — the exact Bench-850 failure shape) because
the head-bias axis decays geometrically under the in-layer double RMSNorm;
δ=0.2 spreads the crossings across the full depth range. That decay is also
the mechanism finding below — see §Why.

## The G1 ladder (greedy argmax, pre-registered posture = `default_adaptive_config`: adaptive Eq-4, ω_max 0.5, logit mode, ref_loop 1)

| Arm | full (n=142) | close (n=19) |
|---|---|---|
| **unguided-full (R=12)** | **1.0000** | **1.0000** |
| guided-full (pre-registered) | 0.8028 | 0.5789 |
| unguided-half (R=6) | 0.9225 | — |
| guided-half (R=6) | 0.9366 | — |
| early-tap incumbent (masked-hidden weak side, same Eq-4 gate) | 1.0000 | 1.0000 |
| product-policy incumbent (w=0.5, weak side = depth-1 readout) | 0.5000 | 0.4211 |
| cheap-weak control (masked-LOGIT weak side, same Eq-4 gate) | 1.0000 | 1.0000 |
| exit-only incumbent (AdvantageMarginGate @ default 0.01) | 0.5000 | 0.4211 |

Every G1 leg the plan hoped to pass, **failed**:

- (a) guided-full ≥ unguided-full: **0.8028 vs 1.0000 — FAIL (−19.7 pts)**;
- (b) guided-half ≥ unguided-full: 0.9366 vs 1.0000 — FAIL;
- (c) guided > all three incumbents on the close stratum: loses to early-tap (1.0) and cheap-weak (1.0); beats only the also-destructive product-policy (0.4211) — FAIL as stated;
- (d) the cheap-weak control MATCHES unguided while the depth arm destroys — the alignment-source story is **refuted in the harmful direction**: the depth-sourced contrast is precisely the destructive one on this fixture.

The verdict is pinned as the inverted-bar gate `tests/loop_guidance_goat.rs
::loopcd_goat_g1_ladder` (the Bench-847/850 pattern): the measured NEGATIVE
directions are asserted with their numbers, and ANY flip in either
direction reds the gate and forces a re-adjudication (T4.1's trigger).

## The posture sweep (disclosure — hyperparameter honesty, not a rescue)

24 postures (mode × ref_loop ∈ {1,3,6,9} × ω_max ∈ {0.1,0.25,0.5}):

- ref_loop=1: full 0.80–0.96, close 0.58–0.68 — destructive at every strength;
- ref_loop=3: partial recovery (full 0.94–0.99, close 0.58–0.89);
- **ref_loop ≥ 6: EXACTLY 1.0000/1.0000 at every ω_max, both modes** — perfect neutrality, never superiority.

A sweep-selected "win" does not exist: the best posture TIES unguided. And
the neutrality at late references is NEUTRALITY BY VANISHING CONTRAST — at
ref_loop ≥ 6 on a monotone-refinement fixture, z_k already agrees with z_R
at the argmax, so there is nothing for the combine to correct. The half-depth headline leg fails at every posture (guided-half ≈ unguided-half
0.92 < unguided-full 1.0).

## Why it fails (mechanism, measured — the record's payload)

The depth-1 reference on this fixture is dominated by the head-BIAS axis:
the bias gap decays geometrically under the in-layer RMSNorm (x3 ← x3/rms
per sublayer, never replenished), so at ref_loop=1 the bias contributes
most of the logit gap. The contrast direction `z_R − z_k` is therefore
mostly **bias removal**, not decision signal — and the combine
faithfully amplifies it, flipping close B-correct cases (guided-lane depth
sensitivity: 1/142 answers move between exit depth and full depth at the
guaranteed window; 60/142 flip at the pre-registered posture's full-depth
readout — 0.8028 vs 1.0).

The paper's disagreement-tracking story presumes the k→R difference is
DECISION-dominated. That presumption holds for trained looped checkpoints
with (near-)unbiased heads. This fixture — like any real checkpoint with
head bias or any non-decision state drift — violates it, and the guidance
has no mechanism to distinguish the two directions: **the contrast
amplifies whatever differs between k and R, selectively or not.** When the
difference IS decision-dominated (post burn-in), a monotone fixture needs
no correction — guidance can only tie. The mechanism is squeezed between
"amplifies noise when it has signal" and "has no signal when it could tie":
on this class of fixture there is no posture where it helps.

Honest scope: this is a toy, monotone-refinement fixture. A checkpoint
with genuinely non-monotone refinement (oscillating close decisions) could
still carry corrective post-burn-in signal — that is the recorded reopen
shape, and it needs a real looped checkpoint, which this stack does not
have (the plan's standing scope note).

## The incumbents (T3.3c demote-the-loser data)

- **early-tap** (865 lane operationalized: weak side = masked readout of
  the SAME final state, same Eq-4 gate): 1.0/1.0 — the state-space
  perturbation is near-neutral here. It BEATS the depth arm.
- **product-policy** (w=0.5, weak side = the depth-1 readout in policy
  space): 0.50/0.42 — ALSO destructive (it blends the biased weak side in,
  in log-prob space). The depth-contrast damage is not specific to the
  affine family.
- **exit-only** (AdvantageMarginGate @ default 0.01): 0.50/0.42 — the
  candidate-improvement margin fires inside the fixture's stable-wrong
  windows (tiny pre-crossing improvements) and exits before the crossing.
  Its 100%-parity claim (Plan 283, proven on ITS fixtures at vocab ≤ 128)
  does NOT transfer to this fixture — recorded, its own benches still pin
  it there.
- **cheap-weak control**: 1.0/1.0 — matches unguided. See (d) above.

## G2 overhead (structural bounds, load-invariant)

- logit-mode guided/unguided: interleaved median-of-ratios 0.973 (noise
  band 0.65–1.59; ≤ the +60% two-head-pass ceiling — the head pass is
  15.6% of a loop block at V=5, and +1.1% of total step FLOPs);
- hidden-mode: median 0.971 (noise band 0.50–1.01) — the +5% bar holds by
  construction (one head pass, one axpy, one margin pass over V);
- net-FLOP disclosure: at V=128k the scratch head pass would be ~5× a
  loop block — the plan's demand that the bounded candidate-set readout be
  CHOSEN before any half-depth claim stands, and the settle-exit's Full
  readout accounting is printed with it;
- settle-exit fusion (d_min=3, p=2, Full readout): fires 142/142, loop
  reduction **65.1%** — but that config's accuracy cost is the measured
  stable-wrong window (16/142 wrong fires unguided, the settle gates' pin);
  at the parity-guaranteed window (9,2) the reduction is 0.2% (the settle
  gates' print). The savings-vs-parity trade-off curve is real and thin.

## G3 / G4

- G3: flag-off (ω=0) byte-identity re-pinned in the GOAT (8 cases);
  `probe_guidance_goat` 5/5 green after the T1.2 refactor; the exit-family
  tests green (`issue_731_t1_residual_exit` 3/3, `issue_698_t4_halter_floors`
  1/1); docs gate 35/35.
- G4: `loop_guidance_alloc_gate` — guided, settle-Full, and
  settle-CandidateSet each add ZERO allocations over unguided (single-test
  sequential arms; the process-wide counter cannot hold parallel siblings).

## Verdict (T4.1 input)

**NEGATIVE, pinned.** The adjudication question — does the alignment
source (same block/head/prefix, only compute differs) plus a non-saturated
structured-depth lane change the Bench-847/850 verdict — is answered NO on
this stack, with the mechanism isolated: the k→R contrast amplifies its
DOMINANT difference component, which is bias/drift in every biased-head
regime; when post-burn-in makes it decision-dominated, a monotone fixture
needs no correction. The class stays closed here:

- `loop_guidance` stays OPT-IN, not default, dead-or-narrow per T4.1 —
  never promoted on a tie;
- the reopen trigger (recorded in the research note §2.35 lineage): a real
  looped checkpoint with measured non-monotone close-decision churn — the
  only regime where the contrast can carry post-burn-in corrective signal;
- the settle-exit (Phase 2) is independently valuable and verdict-clean
  (unguided parity pinned, budget accounting, zero-alloc) — it composes
  with any future exit-family work without the contrast guidance.

## Files

- Fixture: `tests/common/loopcd_fixture.rs` + `tests/loopcd_fixture_gate.rs` (`6c4b5b5f7`)
- Phase 2: `src/transformer/loop_guidance.rs` (SettleExit) + `variants.rs` wiring (`c6eb24d34`)
- GOAT: `tests/loop_guidance_goat.rs` (this bench)

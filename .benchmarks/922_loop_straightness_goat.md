# Bench 922 — loop_straightness GOAT (Issue 922, LiFT appendix C modelless extraction)

**Feature:** `loop_straightness` (katgpt-dec, OPT-IN; katgpt-core forward
`loop_straightness = ["dec_operators", "katgpt-dec/loop_straightness"]`; root
forward). Promotion waits on a production consumer (the `motor_gated_field`
posture) — riir-ai Issue 1037's loop-control consumers are the named riders.

**What:** `path_action(points, stride, weights) -> (Σ‖Δ_k‖²/δ_k, chord², Σδ)`
and `action_efficiency -> η = chord²/(Σ·Σδ) ∈ (0,1]` — the Cauchy–Schwarz
action floor + efficiency over an iterative refinement trajectory. η = 1 iff
straight at constant grid-speed; 1−η = exact wasted-motion fraction
(backtracking / orbiting / A/B/A oscillation). Grid-invariant over any
positive nonuniform weights (η normalizes by Σδ internally). Zero-alloc,
summation axis K (no curse of dimensionality). Pure functions, no wiring, no
kill-switch needed (nothing runs unless called — the signal never gates).

**Signal-diff (pre-implementation, recorded in the issue):** every shipped
cousin is a counter or an entropy read, none is geometric —
`swarm/deliberation.rs` gapped-oscillator flee-tick counters;
`cgsp_runtime` `EntropyCollapse`/`CollapseSignal`; `latent_functor`
`NoiseSustainedOscillation` crowd-regime classification; refine `self_evolve`
trajectory outcomes. Vocabulary-translation grep (path_action /
action_efficiency / straightness / chord projection / wasted motion /
oscillation energy, katgpt-rs + riir-ai) returned only incidental word
collisions (`beam_width.min(actions.len())`, "warm-path action decision").

## Gates

- **G1 (correctness)** — 8 in-module tests, all green
  (`cargo test -p katgpt-dec --features loop_straightness --lib`):
  - η bit-exact **1.0** on dyadic straight paths (1-D and 2-D, nonuniform
    weights) — exact f32 arithmetic, no tolerance.
  - Closed A/B/A (returns to start) scores **0.0** exactly; the OPEN
    three-flight oscillator scores its hand-derived **1/9**.
  - Partial backtrack 0→1→0.5 scores hand-derived **0.1** (Σ = 2.5 at δ = ½;
    the naive 0.2 forgets the /δ doubling of both terms — the review catch).
  - **Cauchy–Schwarz floor sweep**: 500 seeded random walks, dims 1–4, K 2–8,
    uniform AND nonuniform grids: η ≤ 1 + 1e-5 and ≥ 0 everywhere. This arm
    caught the landing's own formula bug — the first cut multiplied by Σδ
    instead of dividing (η = 1.38 on nonuniform grids; uniform Σδ = 1 masked
    it in three fixtures).
  - Dyadic midpoint refinement of a straight path keeps η = 1.0 bit-exact
    (the knot-wise minimizer / grid-invariance half).
  - Degenerate pins: single knot → 1.0; frozen path → 1.0; NaN propagates.
- **G2 (latency)** — `bench_922_loop_straightness_goat` (`--release`, best of
  7 × 1000 calls, `black_box` both ends): **K=8 dim=4 (the deliberation-cadence
  shape): 38.7 ns/call** against the 1,000 ns budget — 26× margin. K=64 dim=32
  (the cgsp-bridge shape) disclosed at 1,137.0 ns/call with the scaling line:
  K·dim grew 64×, latency grew 29.4× (sublinear — one pass, no per-knot
  allocation). Box: M3 Max on AC, shared with sibling agent/cargo sessions —
  read the 26× margin, not the absolute ns (a range, not a figure).
- **G3 (no-regression)** — clippy `-p katgpt-dec --all-targets` green at BOTH
  postures (default; `--features loop_straightness`), `-D warnings` including
  the repo's denied-lint list (`needless_range_loop`, `redundant_slicing` —
  both caught and fixed pre-landing). Default lib surface unchanged: 249
  default tests, module compiles to nothing without the feature.
- **G4 (alloc)** — **0 allocations / 1000 `path_action` calls** (counting
  allocator, the bench_407/775 shared-macro pattern).

## Registration

- test_gate row `katgpt-dec:257:loop_straightness` (249 default + 8 gated).
- README + examples/README feature counts 674→675 (opt-in; default-on stays
  205 — no promotion without a consumer).

## The landing's own lesson

The G1 random-walk arm earned its keep immediately: the first implementation
normalized η as `chord²·Σδ/Σ` (η = 1.38 > 1 on nonuniform grids — a violated
floor) where the correct Cauchy–Schwarz quotient is `chord²/(Σ·Σδ)`. Uniform
fixtures (Σδ = 1) are blind to that class — the nonuniform arm is what makes
the floor test a floor test. Recorded here so the next trajectory-shaped
primitive starts with the adversarial grid, not the uniform one.

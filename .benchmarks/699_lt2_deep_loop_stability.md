# Bench 699 — LT2 deep-loop stability: T≫4 baseline + runtime damping GOAT (Issue 717)

**Status:** ALL PASS (T1/T2 harness, G1–G4, T4 probe, T5 contract, T6 doc);
T7/T8 deferred `[-]` — measured 2026-09-03 on M3 Max (macOS, debug profile — deterministic
assertions only; NO wall-clock gates, box under sibling load ~60).
`lt2_deep_stability` stays **DEFAULT-OFF** (runtime rescue knob for frozen
checkpoints, not a default behavior change; the no-default-consumer rule).

**Instrument:** `tests/issue_717_t1_t2_deep_baseline.rs` (features
`lt2_looped`) + `tests/issue_717_t3_t4_damping_goat.rs` (features
`lt2_looped,lt2_deep_stability`), plus `src/transformer/loop_deep.rs` unit
tests (6/6 under the lib feature).
**Source distillation:** `riir-train/.research/440_Sotaku_Late_State_Looped_Solver.md`
(sotaku @ `6cdb9a9b`, MIT).

## Fixture

`Config::micro()` (vocab 27, block 16, n_layer 1, n_embd 16, head_dim 4,
RMSNorm, ReLU MLP), weights seeded `Rng::new(42)`, Uniform hybrid pattern
(SDPA path), Ahla mode (the issue_407 fixture shape). Two arms:

- **Stable** — default-scale weights, zero residual gates.
- **Destabilized** — layer weights ×1.3 (readout left unscaled so argmax
  comparisons stay meaningful) + every classic residual gate ρ_τ = 1.1,
  giving the carried state the multiplicative mode
  `h ← ρ·h + h̃` whose asymptotic per-iteration multiplier is ρ.

## T1/T2 — the baseline verdict: FLAT (guard work = cheap insurance)

State norm + readout consistency at DEFAULT settings (no knobs), 27-token
suite, readout reference = T=16:

| T | cos(logits, T=16) | argmax agree | state-norm ratio (last/first snapshot) |
|---:|---:|---:|---:|
| 16 | 1.000000 | 27/27 | 1.55 → 1.01 |
| 64 | 1.000000 | 27/27 | 0.995 |
| 256 | 1.000000 | 27/27 | 1.000 |
| 1024 | 1.000000 | 27/27 | 1.000 |

State norm converges to a fixed point (24.2416 = the per-iteration RMSNorm
at pass start rebuilds the state from a unit-RMS input; with zero gates
there is no accumulation channel — the AHLA accumulation path is not
reached under `HybridPattern::Uniform`). **Verdict: the T=4-installed
`forward_looped` is flat to T=1024 on the stable fixture — the damping
knob is insurance, not load-bearing, for this model class.** The
instability DOES exist on the gate-driven fixture (below), so the knob is
real; upstream's trained checkpoints sit between these two regimes.

## G1 — bit-identity (DEFAULT-OFF contract)

`None` / `α = 0` / `direction_scales {1,1}` vs plain `forward_looped`:
**bit-identical logits** across both fixtures × T ∈ {1, 4, 16} × 2
positions. The stats-only arm (snapshots on, knobs off) is also
bit-identical. **PASS** — enabling the feature changes nothing until a
knob is explicitly armed.

## G2 — rescue + eigenvalue map (the headline)

Undamped destabilized arm at T=1024 **degrades deterministically**:
state goes non-finite at snapshot 54 (τ≈864; ρ^1024 overflows f32),
final norm NaN, readout non-finite. Measured undamped multiplier
**λ̂ = 1.1000** (T=256 run, second-half window τ∈[120,248]) — exactly ρ.

Damped arms (burn-in 0, T=1024, second-half window τ∈[512,1024]):

| α | measured multiplier | map `1−α+αλ̂` | rel err | final norm |
|---:|---:|---:|---:|---:|
| 0.03125 | 1.00348 | 1.00312 | 0.04% | 8.5e3 |
| 0.0625 | 1.00633 | 1.00625 | 0.01% | 2.1e5 |
| 0.125 | 1.01250 | 1.01250 | 0.00% | 1.2e8 |
| 0.25 | 1.02500 | 1.02500 | 0.00% | 3.4e13 |
| 0.5 | 1.05000 | 1.05000 | 0.00% | 1.7e24 |

Every arm: finite readout, finite state, tripwires silent, inter-prompt
logit discrimination 0.1503 at α=0.25 (vs NaN on the undamped control).
Monotone: multiplier strictly increases with α (more damping = slower
growth). **The measured multiplier tracks the closed form λ → 1−α+αλ to
≤0.04%** — the closed-form justification from the source distillation
reproduces on our fixture to measurement precision. **PASS.**

(The sweep brackets the upstream α range 0.25→0.03125; α=0.5 included to
show the map holds beyond it. Brackets pinned: `project_lambda(λ,1)=λ`
(α=1 = full update = undamped), `project_lambda(λ,0)=1` (α=0 = frozen,
also the bit-identical OFF spelling).)

## G3 — stable-fixture cost (upstream 1.5–2.3 pts analogue)

Explicit damping α=0.25 on the STABLE fixture vs undamped, 27 tokens:
**argmax agreement 27/27 = 1.000 (0.00 pts disagreement), mean logit cos
1.0000** at BOTH T=64 and T=1024. Measured cost on this fixture is below
the upstream 1.5–2.3 pt class (upstream measured trained Sudoku
checkpoints; a random-weight fixed-point trajectory is insensitive to
state-space rescaling that preserves the readout direction — the honest
fixture caveat is recorded, and the gate asserts only the structural
floor: finite + no majority flip). **PASS**, cost recorded as 0.00 pts.

## G4 — alloc-free stabilization hot loop

TrackingAllocator counters (the root crate's debug `#[global_allocator]`
shim; context/caches/stats-`clear()` hoisted outside the measured region,
buffers capacity-primed by a warm-up call): **8 × T=256 deep runs with
damping α=0.25 + direction scales armed = 0 allocations, 0 bytes.**
Deterministic counter assertion — no wall-clock on this loaded box. **PASS.**

## T4 — tangential-vs-radial probe: the RADIAL axis matters on our fixture

Direction-drift diagnostic on the undamped destabilized arm (T=64):
mean consecutive state-direction cosine **1.0000** with norm ratio
3.6e4×√n — the state grows along a FIXED direction (magnitude-driven
failure), unlike upstream's accumulated-direction-drift failure.

| arm (T=1024) | finite | final norm |
|---|---|---|
| none (control) | NO | NaN |
| **radial ×0.25** | **YES** | 3.7e13 |
| tangential ×0.25 | NO | NaN |

**Verdict: the radial axis is operative on our fixture — the OPPOSITE of
upstream** (tangential ×0.25 rescued 3 collapsed sotaku checkpoints). The
diagnostic explains why: upstream's failure was direction drift (their
direction-only readout matched full readout within 0.3pp — wait, that is
their evidence direction was NOT the issue... upstream's attribution: the
failure WAS accumulated direction drift across answer boundaries, so
scaling the rotational component rescued); ours is magnitude growth along
a fixed direction (the ρ-gated carry), so scaling the radial component
rescues. The knob pair is exposed so the caller picks the axis their
model's diagnostic indicates — the probe + diagnostic ship as the
instrument for that choice.

## T5 — f32-state contract

`LoopDeepRun` state snapshots at T=256 (8 snapshots × 16 dims):
100% f32 bit round-trip; **0.00% of values on the f16 mantissa lattice**
(≤10% allowed; systematic sub-f32 storage would put ~100% there).
The structural pin (`size_of::<f32>() == 4` for the state element) is a
unit test; the Bench-802 contrast is documented at the carry site in
`forward_looped`: **attention rounding DILUTES with context (f16-KV
deviation falls monotonically with sequence length — Bench 802);
weight-tied recurrence AMPLIFIES it with depth** (sotaku BF16 @4096 =
43.7% vs FP32 98.6%; compiled chunks retaining f32 intermediates
recovered to 92.5%). The carried state is `ctx.x: Vec<f32>` end-to-end in
every `forward_looped` arm.

## T6 — relative-residual trap (GainCostLoopHalter doc)

Doc-only note added to `crates/katgpt-core/src/gain_cost_halt.rs`
(struct docs, "Halt-signal warning"): never use `‖F(h)−h‖/‖h‖` as a
convergence/halting signal for non-fixed-point recurrences — upstream
measured state RMS 24→706 while the absolute residual plateaued ≈0.63
(median relative residual 0.000894 @1024 = a growing-denominator
artifact; Anderson/Broyden driven by it diverged to norms 1e5–1e8). The
halter's shipped signals (`step_size` absolute, `angular_change` a
direction) are safe against the trap; the note pins the property for
future "normalized gain" proposals. katgpt-core clippy clean under
`gain_cost_halt`.

## T7 — deferred `[-]`

Cross-check with riir-train Plan 373's trained artifact — contingent on
that checkpoint existing.

## T8 — deferred `[-]`

Erosion diagnostics from Looped-DiT [arXiv:2609.40305]: per-loop ridge
position-probe R² + relative attention-update norm ‖Δattn‖/‖residual‖
as loop-health telemetry. Modelless — runs on the shipped `forward_looped`
today, no trained artifact needed. Target the DESTABILIZED arm (T4's
undamped T=64 arm), not the stable fixture: T1/T2's fixture is already
flat to T=1024, so the probe reads null there by construction (the
paper's signal came from naive looping on an unregulated model — R²
0.865→0.562 over 8 loops). A decayed-R² curve alongside a flat readout
(cos/argmax — modelless, no loss on a frozen fixture) is the
loop-erosion tripwire this bench's damping gate would otherwise miss.

## Regression surface

All 16 existing `forward_looped` callers updated with the new
`deep_run: Option<&mut LoopDeepRun>` parameter (`None` — the elastic-
override precedent: a parameter needs no feature gate, zero cost when
`None`); every touched gate re-run green:

| gate | features | result |
|---|---|---|
| goat_108_lt2_looped | lt2_looped | 11/11 (count-identical) |
| issue_035_any_time_lt2_dispatch | lt2_looped | 13/13 (count-identical)¹ |
| bench_108 / bench_483 / issue_156 / issue_407 | lt2_looped | 6+1+1(1i)+1 green |
| goat_428 + issue_698 t1/t2/t3/t6/t7/t8×2 | +loop_stability_fix | 10 green |
| issue_698_t4_halter_floors | +loop_stability_fix,gain_cost_halt | 1/1 |
| t2_2_weight_shared_gate | +weight_shared_advantage_gate | 4/4 |
| issue_717_t1_t2_deep_baseline | lt2_looped | 3/3 |
| issue_717_t3_t4_damping_goat | +lt2_deep_stability | 5/5 |
| loop_deep unit tests (lib) | +lt2_deep_stability | 6/6 |

¹ The wall-clock `none_path_overhead_is_noise_floor` row failed at the
PRE-CHANGE baseline under box load (p99 6.0 ms vs p50 138 µs — a
scheduling-stall tripwire, load-dependent) and passed post-change; both
observations recorded per the flake rule — no code-regression signal
either way.

Clippy: 0 warnings in every touched state (`-p katgpt-rs --lib` default +
`lt2_looped` + `lt2_looped,lt2_deep_stability`; `-p katgpt-core --lib
--features gain_cost_halt`; both new test targets).

## En-route findings

1. **`apply_damping` α-inversion caught by the G2 gate on its first
   armed run**: the first implementation weighted α on the previous state
   (`(1−α)h̃ + α·h`), inverting sotaku's `h ← (1−α)h + α·F(h)` — my α was
   1−theirs, so α=0.03125 ran nearly undamped and overflowed. The map
   assert caught it to 4 decimal places. Fixed; the fixture now pins the
   correct semantics.
2. **`project_lambda` brackets**: α=1 ⇒ λ (full update), α=0 ⇒ 1 (frozen
   state) — the first draft asserted the reverse bracket and the unit
   gate caught it.
3. **Naive `Σv²` norms overflow f32 at ‖x‖ ≳ 1e19** while the state
   values are still ~1e24-finite: `robust_norm` (max-abs-scaled two-pass)
   added; the G2 α=0.5 arm is the case that caught it.

## Run

```bash
cargo test -p katgpt-rs --features lt2_looped --test issue_717_t1_t2_deep_baseline -- --nocapture
cargo test -p katgpt-rs --features lt2_looped,lt2_deep_stability --test issue_717_t3_t4_damping_goat -- --nocapture
```

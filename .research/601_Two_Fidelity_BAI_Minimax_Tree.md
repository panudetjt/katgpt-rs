# Research 601: Two-Fidelity Best-Action Identification for Stochastic Minimax Trees

**Status:** DONE — Plan 615 shipped: Phase 1 core + Phase 2 GOAT **PASS** (2026-10-03, `.benchmarks/615_two_fidelity_bai_goat.md`: G1 0/300 errors CP95 0.00994, 300/300 certified, 0 invalid intervals; G2 wins vs the δ-correct LUCB-MCTS baseline and negamax-UCT at matched accuracy; caveats: conservative-baseline-instance + the 916(b) cost-debt vs the paper's instance — CLOSED 2026-10-03 by the opt-in `SearchConfig::lazy_discharge` resolver ported from the authors' code, `ca9fb1617`: G1 0/300 in two regimes at 196–1117× below v1, Bench 615 addendum). The 906-era G1 negative was a bench-fixture contract violation, not a primitive defect (Issue 915, closed). Phase 3 seam: the riir-ai issue cites this research + the bench.

> **Source:** *Two-Fidelity Best-Action Identification for Stochastic Minimax Tree* — Peter Chen (UC Berkeley), Xi Chen (NYU), [arXiv:2606.01708](https://arxiv.org/abs/2606.01708) v2, NeurIPS 2026. Numerical implementation: `github.com/PeterLauLukChen/2FFS`.
> **Date:** 2026-10-01
> **Related Research:** 510 (certified frontier — flat SafeOpt stopping), 218 (breakeven complexity — cost-tier routing), 027 (STRATEGA — forward-model arenas), 020/310-riir-ai (multi-fidelity NPC simulation, unshipped)
> **Related Plans:** 615 (this note's implementation plan)
> **Classification:** Public

---

## TL;DR

2FFS brings multi-fidelity best-arm identification into minimax trees: every node can be evaluated by a **fast oracle** (cheap, deterministic, biased within a known envelope `B(h)`) or a **slow oracle** (expensive, stochastic, unbiased), and the algorithm adaptively races *minimax-style fast expansion* against *MCTS-style slow certification* until the root action is certified ε-optimal with confidence 1−δ. It proves PAC correctness, finite stopping, and a cost bound `O(D²)·H*` (H* = ideal recursive oracle complexity), and measured **163–1458× fewer samples** and **1.9–4.6× fewer operations** than BAI-MCTS on synthetic trees. Nothing in our stack ships interval-propagation-with-stopping in trees, and nothing anywhere ships a two-oracle cost race inside minimax search — this is the missing primitive connecting our shipped flat LCB machinery (`best_belief`), our minimax-over-hypotheses planners (`BoMMinimaxPlanner`), and our budgeted single-fidelity MCTS (`mcts.rs`/`chance_puct.rs`).

**Distilled for katgpt-rs (modelless, inference-time):** the algorithm is pure query-allocation + interval bookkeeping — zero gradient descent, zero weights. The transferable mechanism is the *certified two-fidelity race*: local interval = `[V_F ± B(h)] ∩ (running intersection of time-uniform slow CIs)`, propagated through minimax backups (`L = max/min of child L`, `U = max/min of child U` — width never increases), a leader/challenger ε-stopping rule at the root, and a per-node-side-scale **race budget** `α_h·Γ_v^(k)` that decides "expand deeper with fast evals" vs "sample here with slow evals", with monotone latched side-certificates (no rework) and **effective-gap** precision allocation (a node only needs precision at the scale that can still change the root decision).

---

## 1. Paper Core Findings

### 1.1 Setting

Stochastic minimax tree (alternating Max/Min, root = Max), fixed-confidence BAI at the root: identify `a* = argmax V*(r,·)` with `P(error > ε) ≤ δ` while minimizing total cost `C = N_fast + c·N_slow` (slow costs `c ≥ 1` per query). Two oracles per exposed node:

- **Fast oracle F**: deterministic value `V_F(v)`, biased with a *known, nondecreasing-in-remaining-depth* envelope `|V_F(v) − V*(v)| ≤ B(h(v))`, `B(0) = 0`. Motivating case (Remark 2.2): a minimax Bellman-residual critic in finite-horizon MARL — `B(h) = η̄·Σ γ^i` falls out of Bellman nonexpansiveness.
- **Slow oracle S**: independent σ-sub-Gaussian samples, mean `V*(v)` — unbiased. Unlike BAI-MCTS (samples attached to leaf rollouts), queryable at *any exposed node*.

### 1.2 The interval machinery

- Local: `I_v^loc = I_v^F ∩ I_v^S` where `I_v^S` is the **running intersection** of time-uniform CIs over the sample history (Eq. 4) — monotone shrink.
- Child backup (Eq. 6): Max node → `[max L_u, max U_u]`; Min node → `[min L_u, min U_u]`. Lemma 2.3: validity AND width preserved upward (a max of intervals whose widths ≤ w has width ≤ w).
- Effective: `I_v = I_v^loc ∩ I_v^ch`. Lemma 2.4: simultaneous validity on the event `E_δ` (prob ≥ 1−δ via union bound over a feasible node-wise allocation `Σ δ_v ≤ δ`).

### 1.3 The algorithm (2FFS)

1. **Root loop**: leader `a_t = argmax L_a`, challenger `b_t = argmax_{≠a} U_a`. Stop when `L_â ≥ max_{a≠â} U_a − ε`. Otherwise refine the *coarser unresolved* of (leader's L-side, challenger's contender side), via `Resolve(v, s, k)` at dyadic scale `ρ_k` (`ρ_{k+1} = ρ_k/2`, `ρ_0` = widest root-child interval).
2. **Resolve(v, side, scale)** races two routes:
   - *Local*: one slow sample at v (shrinks `I_v^loc`).
   - *Recursive*: expand v (fast-query all children), then recurse into the **live/blocking child** only — in selector cases (Min,L)/(Max,U) the endpoint witness; in comparison cases (Max,L)/(Min,U) the child still inside the scale-k comparison margin `ρ_k/2` (lazy discharge: a child outside the margin is dropped without being fully certified).
   - Recursive spending capped by race budget `B^rec_{v,k} = α_{h(v)}·Γ_v^(k)(δ_v)` (α_h = (h+1)² works); budget exhausted → fall back to local sample. **Local reversibility**: slow samples remain available at v even after expansion — the cheaper route is unknown in advance.
3. **Certificates**: `Comp_s(v,k)` = latched `Done_s(v,k)` flag OR observable `Cert_s(v,k)` (width ≤ ρ_k/2, or local width ≤ ρ_k/2, or recursive discharge of all children). Latching = zero rework per (node, side, scale).
4. **Effective gap** `Δ_v^eff = max{Δ^eff_parent, |V*(v) − V*(parent)|}` — the largest bottleneck on the root-to-v path. Prop. B.10: every active call satisfies `Δ_v^eff ≤ 2ρ_k` — the algorithm only ever certifies to the precision that can affect the root decision.

### 1.4 Results

- **Thm 3.1 (PAC)**: every finite-time recommendation is ε-optimal on `E_δ`.
- **Thm 3.6**: finite stopping + `C_τ ≤ P_D·H(δ)`, `P_D = O(D²)` with α_h=(h+1)²; H(δ) = ideal recursive oracle complexity (Eq. 8: each node takes `min{certify locally, expand + Σ child costs}` under perfect gap knowledge), `H* = inf over allocations`.
- **Robustness**: biased slow oracle (known ξ) → inflate radius by ξ, **no depth accumulation** (Thm 3.7); loose fast envelope `B̂ ≥ B` preserves PAC (Thm 3.8).
- **Measured** (100 trees/setting, balanced b-ary): (D=5,b=8) samples 5.39e3 vs BAI-MCTS 8.80e5 (**163×**), ops 2.84×; (D=7,b=6) 1.77e4 vs 1.75e7 (**988×**), ops 4.58×; (D=10,b=3) 1.31e4 vs 1.91e7 (**1458×**), ops 1.86×. Accuracy 1.00 everywhere; minimax-fast (fast-only) accuracy 0.88–0.91; slow-only often fails to stop.

### 1.5 Prior art (checked, §4)

- **BAI-MCTS** (Kaufmann & Koolen 2017, arXiv:1706.02986): single-fidelity, unbiased leaf samples, leader/challenger interval stopping — the paper's baseline. Not the two-oracle race.
- **MF-BAI flat line**: Kandasamy 2016, Poiani 2022 (NeurIPS), Wang 2023, Poiani 2024 (optimal, arXiv:2406.03033) — flat arm spaces only. Poiani 2022's motivation literally names depth-truncated planning as the cheap fidelity but solves the flat problem.
- **MFHOO/MFPOO** (Sen et al. 2019): multi-fidelity tree search for noisy black-box optimization over hierarchical *partitions of a continuous domain* — simple-regret objective, no minimax alternation, no root-action certification (paper's Appendix A distinction).
- No published tree multi-fidelity BAI other than this paper. In-stack: nothing (see §2).

---

## 2. Distillation

### 2.1 Closest shipped substrate (vocabulary translation)

| Paper term | Codebase equivalent | Ships? | Signal-diff |
|---|---|---|---|
| Slow oracle (unbiased stochastic samples) | MCTS rollouts (`mcts.rs` `RolloutPolicy` — note: `mcts_search` is **single-player UCT**, fixed `player_id`, no Min nodes, so it is *not* a minimax baseline at all); BoM K-hypothesis sampling (`katgpt-micro-belief/src/bom_arena.rs`, re-exported via katgpt-core's `micro_belief`; riir-ai `multi_hypothesis_planner.rs`) | ✅ | Rollouts are budget-counted, never confidence-certified; BoM always samples fixed K — no stopping rule |
| Fast oracle (cheap, biased, known envelope) | `StateHeuristic` at leaves (`mcts_search_informed`); 1-ply prior scores (`chance_puct.rs`); trained critic (riir-instinct Tetris) | ✅ | Heuristic is used *instead of* sampling, never *raced against* it with a cost-aware certificate; no bias-envelope contract |
| Interval minimax backup (L/U through Max/Min) | — | ❌ | Nothing propagates confidence intervals through minimax; `BoMMinimaxPlanner` backs up point values |
| Leader/challenger ε-stopping at root | Flat analogs: `best_belief_score` Beta-LCB (katgpt-core), `grouped_evidence` BAI GOAT (Bench 905), reflex cascade-worthiness LCB legs (Bench 063/066/070) | ✅ flat only | All flat; none lift intervals through a tree or propagate them through adversarial min-backups |
| Certified stopping ("when to stop looking") | `certified_frontier.rs` `should_advance` (Research 510, SafeOpt lineage) | ✅ flat | SafeOpt safe-set growth over latent cells with a single binary verifier — no tree, no two-oracle cost race, no minimax |
| Effective-gap precision allocation | — | ❌ | No shipped mechanism allocates estimation precision by decision-relevance along root-to-node paths |
| Cost-tier selection (cheap vs expensive oracle) | riir-train `multifidelity.rs`/`breakeven.rs` (Plan 276); Research 218 inference router; riir-ai fog-of-war Plasma/Hot tiering (Research 020/310, **unshipped**) | ✅ offline/off | Static cost-model selection at plan time — not online certified racing; 310's own verdict: multi-fidelity NPCs are a selling point "only if we actually ship" them |
| Race budget + latched certificates + dyadic scales | — | ❌ | Nothing; `proof_cache`/`twist_cache` cache results, they do not certificate scales |

### 2.2 The fusion (what paper × our substrate produces)

**Fusion A (primary, priority #1 game runtime): certified stopping for the BoM minimax planner.**
⚠ *Novelty accounting: the near-term per-step shape of this fusion is FLAT two-fidelity BAI — published prior art (Poiani 2022/2024) — and does NOT count toward the novelty claim. Only the multi-step (tree) version counts, and multi-step BoM planning does not exist yet in that lane; the fusion is recorded as the consumer seam, not as novelty.*
`MultiHypothesisBoMMinimaxPlanner` (riir-ai engine, Bench 281 G2; the katgpt-micro-belief `BoMMinimaxPlanner` in `bom_arena.rs` is the wiring reference) samples K belief hypotheses and minimaxes over them — every plan_action pays full K·|A| kernel evals. The 2FFS reframing on the *flat* per-step version: fast oracle = worst-case score under the **mean belief** (deterministic, one kernel pass; bias envelope = BoM dispersion, which **shrinks as the belief concentrates** — a self-tightening `B(h)` analog); slow oracle = minimax over a freshly sampled hypothesis (unbiased-for-robust-value, costs a noise query + kernel evals). The tree version applies the moment planning goes multi-step (the arena steps repeatedly; strategic re-planning is the cadence). Output: same action quality at a fraction of the queries **plus a certificate** — "this action is ε-robust against the worst-case hypothesis with confidence 1−δ". That is a *deliberation-budget* mechanism: it composes with the limelight cognition-budget lane (Bench 960) — limelight allocates *observation* budget across NPCs; 2FFS allocates *deliberation* budget within one decision.

**Fusion B (consumer #2, healer — honest one-liner after reading the surface):** the healer's cheap/expensive oracle pair is propose (latent retrieval, µs) vs `--verify` compile gate (seconds) — but it is a *flat, per-fix* decision with no tree structure and no repeated sampling per node; the transferable half is only the leader/challenger early-stop shape, which the cascade-worthiness lanes already implement empirically in reflex. Weak — recorded, not filed.

**Fusion C (watch, inference-perf league):** riir-infer's load-time kernel-variant trials (B193 `dual-layout-transpose-at-stage`: "picked per shape by a double-bitwise-gated streaming trial, adopt 0.02/drop 0.05") are flat two-armed empirical races; the effective-gap idea (stop measuring once variants are separated beyond the decision margin) is the certified version of early stopping. Marginal — the trials are already cheap.

**Fusion D (watch, riir-instinct):** Tetris T7 round-4 priced "critic-guided search" and found the ~240-eval next-piece expectation "TIGHT against G2 at 0.84 ms". 2FFS is exactly critic-guided search with certified early stopping — cheap critic = fast oracle, full eval = slow oracle — and its sample counts drop 100–1000× when gaps are clear. The serve-bar (1 ms) remains the open question; the *arena* path (offline, seconds) is immediately applicable. Watch item, gated on the katgpt-core primitive landing.

### 2.3 Game-context reframe (§1 step 4)

Per-NPC, 20 Hz tick: **no** — the certificate machinery's operation counts (10⁷–10⁹ in the paper's own tables) are strategic-tier, not per-tick. Where it lands instead: (a) strategic re-planning (the goal_salience cadence tier — 11–12 ns/NPC/tick for the *gate*, with re-planning at ms+), (b) boss/raid AI where a wrong robust-decision is expensive, (c) quest/encounter planning over adversarial uncertainty (fog-of-war worst case). The behavior signal: an NPC that *escalates deliberation only when the decision is genuinely contested* — measurable as kernel-evals-per-decision dropping on clear-cut situations while robustness guarantees hold.

Consumer-context reframe (§1 step 4, healer): stated in Fusion B — weak, recorded.

### 2.4 What is genuinely new here (pinned claim)

> **Cost-certified two-fidelity best-action identification for minimax tree search** — interval minimax backup (fast-envelope ∩ time-uniform slow CI, running-intersection) with leader/challenger ε-stopping and a budgeted local-vs-recursive race per node-side-dyadic-scale — for game-tree planning and per-NPC robust decision-making under adversarial uncertainty, consuming (cheap deterministic biased evaluation, expensive stochastic unbiased samples), **distinguished from** `certified_frontier` (flat SafeOpt safe-set growth, single verifier, no tree) by the tree/minimax structure and two-oracle cost race; from `mcts.rs`/`chance_puct.rs` (single-fidelity, budget-counted) by fixed-confidence stopping with cost certificates; from flat MF-BAI (`best_belief`, Bench 905) by interval propagation through adversarial backups.

---

## 3. Verdict

**Per-track:**

| Track | Verdict | One-line reason |
|---|---|---|
| (a) Modelless inference | **GOAT** — plan it (Plan 615) | Pure query-allocation algorithm; provable sample/ops gain at matched accuracy over proper adversarial baselines (BAI-MCTS per the paper; a negamax-UCT adapter — `mcts_search` itself is single-player UCT and cannot serve as a minimax baseline); fills an empty stack slot (interval minimax + certified stopping) |
| (b) Self-adaptive runtime | Folded into (a)'s consumer map | The certificate-latch + budget race is the runtime-compute-allocation pattern; its embodiment is Fusion A, filed under (a)'s plan |
| (c) Model-based | **Audited discard** — watch only | No training-loop content (no optimizer/loss/schedule); the trained-critic-as-fast-oracle angle (Fusion D) is an inference-time gating decision that consumes the (a) primitive; no recipe worth a riir-train plan |

Adversarial panel: **skipped — clearly inference-side** (a decision-time search/query-allocation algorithm; the only RL content is future-work MARL application in the paper's appendix). §3.5 Path 0 inventory (component-level, even though no deferral occurred):

| Component | Coverage | Extraction |
|---|---|---|
| Interval minimax backup | no analog | ✅ closed-form (Lemma 2.3 arithmetic) — open primitive |
| Leader/challenger ε-stop | flat analogs ship (`best_belief`, cascade LCB) | ✅ lift into tree = the new half |
| Time-uniform CI radius | `stats.rs`/`best_belief` are fixed-n; `welford.rs` streams | ✅ stitching-style radius, closed-form |
| Fast/slow race budget | no analog (breakeven = offline cost model) | ✅ closed-form rule (α_h·Γ) |
| Effective-gap allocation | no analog | ✅ derived quantity (path max-bottleneck) |
| Dyadic certificates + latching | `proof_cache` caches, does not certificate | ✅ bookkeeping structure |

**Tier scoring (§1.5):**
1. **Q1 no prior art?** YES — signal-diffs in §2.1 (every "ships?" ✅ row differs at mechanism level: flat vs tree, budget vs certificate, offline vs online race); published art is the paper itself + flat MF-BAI.
2. **Q2 new behavior class?** Qualified YES — "certified deliberation stopping under adversarial uncertainty" is a new class *for the strategic tier*; per-tick is out of scope by operation count.
3. **Q3 product selling point?** NO (weak) — "NPCs that know when they've thought enough" is plausible but speculative; Research 310's own caution about unshipped multi-fidelity NPCs applies symmetrically.
4. **Q4 force multiplier?** YES — connects mcts family + best_belief/LCB + BoM planner (+ limelight adjacency).

**3/4 → GOAT, not Super-GOAT.** Mandatory outputs: this note + Plan 615 (feature-gagged `two_fidelity_bai` module in katgpt-core + GOAT bench on the paper's own synthetic-tree class). No Super-GOAT guide, no "candidate" wording.

**MOAT gate (katgpt-rs):** fits — "fundamental/principle base primitive via fusion" in the Transformer-stack/search slot (MCTS/bandits/pruners family); promote-to-default decision deferred to the GOAT gate (expect: stays opt-in — it is a *certified-search* tool, not a default path; the default MCTS slot keeps `mcts_search`).

**Routing:** primitive → `katgpt-rs/crates/katgpt-core/src/two_fidelity_bai.rs` (public, MIT); consumer wiring (BoM certified stopping) → riir-ai, gated on the primitive landing (Plan 615 Phase 3 records the seam; the riir-ai issue is filed when the primitive exists). No chain/shard/training content.

---

## 4. Implementation notes distilled for Plan 615

- **Trait shape**: a `MinimaxSpace` with `children_into(node) -> bool` (expansion reveals children), `fast_value(node) -> f32`, `bias_envelope(remaining_depth) -> f32`, `slow_sample(node, &mut Rng) -> f32`, `node_kind(node) -> Max|Min`. Zero-alloc arena like `chance_puct.rs` (Vec nodes, contiguous children, scratch reuse).
- **Confidence radius**: time-uniform sub-Gaussian via stitching (Howard et al. 2021 confidence sequences; the paper's refs [14,15]) — any non-increasing β(n, δ_v) with the Eq. 3 property works; v1 uses a documented stitching law, not plain Hoeffding (which is not time-uniform).
- **δ allocation**: feasible node-wise allocation Σ δ_v ≤ δ; v1 = uniform over an a-priori node cap (`MAX_TREE_SIZE`-style, the `mcts.rs` precedent) with the tradeoff documented (dynamic trees would want a dyadic rank schedule).
- **Certificates**: `Done` bitflags per (side, scale) in the node struct; scales as `u8` (ρ_k = ρ_0·2^-k).
- **Bench fixture** (paper's own, cheap): balanced b-ary trees, leaf means ~ U[0,1], slow = mean + σ·noise, fast = true value + bias scaled to B(h) — INCLUDING an adversarial-bias arm whose sign is chosen to flip the fast-only minimax answer at the root (a fixed-sign bias lets fast-only win for free and never forces the two-oracle race); ground truth by exact minimax. Settings (D,b) ∈ {(5,8),(7,6),(10,3)}; 100 seeds each.
- **Baselines**: (a) **BAI-MCTS** (UGapE-MCTS / LUCB-MCTS per Kaufmann & Koolen 2017 — the paper's own baseline; gives the parity row for the 163–1458× figure); (b) a **negamax-UCT adapter** over the same `MinimaxSpace` (two-player UCT with sign-flipped backups); (c) fast-only; (d) slow-only; `mcts_search` appears only as a labelled "not adversarial — context only" row.
- **Cost model (one per setting, every arm scored on it)**: `cost = n_fast + c·n_slow` with c fixed per setting, plus a separately reported ops count. Never mix `advance()`-budget units with oracle-query units.
- **GOAT gates**: G1 empirical PAC error with δ named (δ = 0.05) and judged by a one-sided Clopper–Pearson 95% upper bound on the error count (a zero-error run reads as "consistent with δ", not as a pass-by-fiat) + Lemma-2.3 width-preservation property test; G2 strict paired win (LB95 > 0, the Bench 905 protocol) on the unified cost model vs **BAI-MCTS and negamax-UCT** at matched accuracy (the paper's 163–1458× is the target class vs BAI-MCTS; the bar is a strict win, not parity); G3 untouched `mcts.rs`/`chance_puct.rs` (new module); G4 zero-alloc steady-state search loop (counting allocator, the chance_puct G4 arm precedent).
- **Honest caveats**: the O(D²) overhead factor and the regularity assumption (Assumption 3.3) are analysis devices, not free; the knife-edge at the fast-bias cutoff (B.8) means v1 should prefer the loose-envelope posture (Thm 3.8) where the caller over-approximates B; operation counts are large — this is a strategic-tier primitive, and the plan must NOT wire it into any per-tick path.

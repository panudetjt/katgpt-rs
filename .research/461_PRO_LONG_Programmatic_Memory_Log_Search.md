# Research 461: PRO-LONG — Programmatic Memory (Lossless Log + Programmatic Search) for Long-Horizon Reasoning

> **Source:** [PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning](https://arxiv.org/abs/2607.20064) — Fox, Wang, Rosu, Dhingra (Duke), 2026-07-23
> **Date:** 2026-07-29
> **Status:** Active
> **Related Research:** 024 (δ-Mem — the online associative memory substrate), 278 (Engram — conditional memory + latent lookup), 289 (RecursiveMAS — already-shipped), 300 (riir-neuron-db Trellis/Experience Graph — the R300 database lesson + closest shipped query layer), 368 (AutoMem — the R368 LLM-orchestration-vs-modelless lesson + closest cousin verdict); riir-ai 169 (AgentMemBench — F6/F7 validation signals for PRO-LONG's thesis), 147 (Engram NPC Guide), 323 (SimRing — game state sync ring buffer)
> **Related Plans:** riir-neuron-db 319 (Experience Graph Query Layer — the closest shipped query layer), katgpt-rs 124 (EventLog — the closest shipped lossless log)
> **Classification:** Public

---

## TL;DR

**Verdict: Gain.** PRO-LONG's access pattern (lossless append-all write + programmatic read via regex/Python/grep) is a substrate-independent memory architecture whose value decomposes modellessly per the R368 lesson — LLM-as-coding-agent-searcher is the *instantiation*, the access pattern (search-at-read-time, decide-nothing-at-write-time) is the *mechanism*. We ship the lossless append substrate in multiple forms (`EventLog`, `TrialLog`, `ExperienceLog`) and we ship a query layer (`experience_graph` — latent-seeded NS traversal, OFFLINE ONLY), but we do NOT ship the **programmatic/pattern-based search axis** as a coherent primitive — that is the genuine gap, validated by R169 findings F6 (raw > compressed) + F7 (late filtering > early filtering) and by PRO-LONG's own Table 1 (Read 23.1% < +Grep 27.2% < +Python 38.3% < +Write/Edit 41.2%).

**Not PASS** because the programmatic-search axis is a concrete missing feature (not "out of scope" — the substrate ships, the query axis does not). **Not Super-GOAT** because the cross-session memory capability already ships via `experience_graph` (Q2 fails — not a new capability class); the "NPCs grep their own history" framing is a developer-facing feature, not a product differentiator (Q3 fails). **Gain** because it is actionable: a deterministic query DSL over the lossless log, composable with `experience_graph` + Engram + Raven/δ-Mem consolidation, with a clear PoC target.

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is the **fidelity–tractability tradeoff resolved at read time**: write everything losslessly (no learned/heuristic compression at write time); search programmatically at read time (regex/pattern/filter over structured log entries, not vector ANN). The PRO-LONG paper proves this beats lossy-summary + embedding-retrieval on ARC-AGI-3 by 18.0 percentage points average across frontier models, at 4.2–5.8× fewer tokens. The distillation: a **deterministic query combinator** over `EventLog`/`TrialLog` that lets a modelless consumer (per-NPC cognition, MCTS planner, consolidation pipeline) express "find all events matching pattern P in the last N ticks" without an LLM in the loop.

---

## 1. Paper Core Findings

### 1.1 The thesis — fidelity–tractability tradeoff, resolved at read time

Agent memory systems face a tradeoff: preserving more information makes retrieving relevant details less tractable. Existing systems resolve it at **write time** (decide what to compress/summarize/embed) — PRO-LONG resolves it at **read time** (write everything; search programmatically when needed). The paper frames this as a `write`/`read` operation pair:

- **`write`** = append every observation, action, and outcome into a structured log (`logs.txt`). No learned or heuristic decision about what to store.
- **`read`** = programmatic search over the log via the coding agent's native tools (grep, regex, Python, bash). No specialized retrieval (no vector DB, no embedding index).

Three design principles: **simplicity** (write needs no decisions), **losslessness** (nothing compressed or summarized), **compatibility with coding agents** (programmatic search is native to LLMs with code execution).

### 1.2 The empirical results (ARC-AGI-3, 25 public games)

| Model | PRO-LONG pass@1 | Best prior | Token savings |
|---|---|---|---|
| GPT-5.5 (Codex) | 41.2% | 45.1% (WorldModeler) | 5.8× fewer |
| Opus 4.6 (Claude Code) | 42.4% | 39.0% (Arcgentica) | — |
| Opus 4.6 best@2 | 52.0% | — | — |
| Fable 5 best@2 (500 actions) | 76.1% | 84.4% (Schema) | 4.2× fewer |
| Fable 5 best@2 (2000 actions) | 97.4% | 99.0% (Schema) | 3× cheaper ($1,750 vs $6,447) |

PRO-LONG improves over the no-log base coding agent by **15.7–21.0 percentage points** across all frontier models.

### 1.3 The tool ladder ablation (Table 1 — the load-bearing finding)

| Tools enabled | Score (%) |
|---|---|
| Read only | 23.1 |
| + Grep / Regex Search | 27.2 (+4.1) |
| + Python | 38.3 (+11.1) |
| + Write, Edit | 41.2 (+2.9) |

**Programmatic tools (grep + Python) account for +15.2 of the +18.1 gain.** Write/Edit (the agent's means of maintaining OTHER forms of memory like notes) adds only +2.9. This is the empirical evidence that the **programmatic search axis** is the value, not the workspace persistence.

### 1.4 The workspace persistence ablation (Table 2)

| Workspace | PRO-LONG | No-Log |
|---|---|---|
| Persistent | 41.2 ±3.5 | 24.0 ±2.0 |
| Cleared every call | 40.7 ±3.6 | 19.9 ±2.1 |

Clearing the workspace every call costs PRO-LONG only 0.5 points; it costs no-log 4.1 points. **PRO-LONG's value is in the log, not in self-authored notes.** This directly contradicts the AutoMem (R368) thesis that "memory management is a separable skill" optimized by writing better notes — PRO-LONG shows that *not writing notes at all* (just searching the log) is competitive.

### 1.5 The best@k widening gap (Figure 3)

The gap between full-log and last-25-actions widens with k (independent runs): at k=1 the gap is ~7 points; at k=5 it is ~18 points. **Fuller memory covers more games** (expands the set of games solvable at all), and games requiring long-horizon reasoning have higher variance. This is the evidence that lossless log beats truncated log specifically on the hard long-horizon cases.

### 1.6 The qualitative pattern — agents code their own world models

On `m0r0` (two-block maze), PRO-LONG agents spontaneously code a transition-function world model + breadth-first search over joint positions, finding 50+ action routes. On `g50t` (ghost-clone rewind), logs exceed 320,000 lines and agents score up to 56.3%. The paper's framing: "programmatic perception and reasoning (code to parse boards, model transitions, plan with BFS) are broadly effective for continual learning tasks." The harness elicits this; it does not hardcode it.

---

## 2. Distillation

### 2.1 The R368 lesson applied (decisive check)

Per R368: *"when you see 'N LLM calls/step', the FIRST question is 'what decision is each LLM call computing?' — not 'this violates the 20Hz budget, NO-GAIN.'"*

PRO-LONG's "coding agent" is an LLM with code execution. The paper's value decomposes:

| PRO-LONG operation | LLM instantiation | Our modelless substrate? |
|---|---|---|
| `write` = append observation/action/outcome to log | LLM calls a tool to append | **YES** — `EventLog::push`, `TrialLog::push`, `ExperienceLog` append (all ship) |
| `read` = search log programmatically | LLM writes Python/grep to query | **PARTIAL** — `experience_graph` does latent-seeded NS traversal (vector ANN + KG edge walk); **NO programmatic/pattern search** (grep for `query_dsl\|log_search\|pattern_match.*event\|search_log\|grep_log\|filter_events` returns ZERO hits) |
| Lossless (no compression at write) | Implicit (append-all) | **YES** — raw Pod, BLAKE3 hash chain, no lossy summary |
| Coding-agent compatibility | LLM writes code to reason over log | **NO direct analog** — but the *access pattern* (search at read time) is modellessly instantiable as a deterministic query combinator |
| Spontaneous world-model coding | LLM generates transition functions | **NO modelless analog** — genuine NO-GAIN if the value IS the code generation |

**The R368 correction:** the initial temptation is to PASS this paper as "LLM-orchestration layer, ≥1 LLM call/step, violates 20Hz budget" (the R169 failure mode). This is wrong for the same reason R368's initial PASS of AutoMem was wrong. The *access pattern* (lossless write + programmatic read) is implementation-agnostic; the LLM is one instantiation. The modelless instantiation is a **deterministic query combinator** over the lossless log.

**Where the R368 decomposition stops:** the paper's qualitative finding that "agents spontaneously code world models + BFS planners" (§1.6) IS genuinely LLM-dependent (semantic code generation has no modelless analog — same as AIDE² R440). That slice → no modelless analog. But that is a *consequence* of the access pattern, not the access pattern itself. The access pattern (search the log) is the distillable primitive; the emergent world-model coding is a benefit that requires the LLM.

### 2.2 Gap map — what ships vs. what's missing (verified by source read)

| PRO-LONG concept | Ships? | Where |
|---|---|---|
| Lossless append-only event log | ✅ | `katgpt-pruners/src/event_log.rs` `EventLog<A>` (public, Plan 124) — `push`, `replay`, `fork`, `diff`, `EvalCache` |
| Per-action reward-bearing log with integrity | ✅ | `riir-games/src/trial_log.rs` `TrialRecord` + `TrialLog` — BLAKE3 hash chain, `verify()` |
| AS-OF temporal reconstruction | ✅ | `riir-neuron-db/src/experience_graph/as_of.rs` `ExperienceLog::reconstruct_as_of` (Plan 319 Phase 2) |
| Latent-seeded graph traversal (vector ANN + NS edges) | ✅ (OFFLINE ONLY) | `riir-neuron-db/src/experience_graph/graph.rs` `ExperienceGraph::latent_seeded_ns_traversal` — 5–10ms BFS, NOT per-tick |
| Merkle-committed freeze/thaw over log snapshots | ✅ | `riir-neuron-db/src/freeze.rs` `MerkleFrozenEnvelope` + `ExperienceLog::freeze_envelope`/`thaw_envelope` |
| Conditional hash-addressed lookup | ✅ | `riir-engine/src/engram_runtime/` (access_matrix, local_cache, shard_first) |
| Content-addressed eval cache | ✅ | `EventLog::EvalCache` (state hash → cached score) |
| **Programmatic/pattern search over log entries** | ❌ **GAP** | Zero hits on `query_dsl\|log_search\|pattern_match.*event\|search_log\|grep_log\|filter_events`. `EventLog::iter()` is linear; no filter/query/predicate API. |
| **Deterministic query combinator (the modelless analog of "coding agent searches log")** | ❌ **GAP** | No `EventLog::filter(predicate) -> &[&Event]`, no `TrialLog::query_range(tick_range, action_tag_pattern)`, no pattern DSL. |
| LLM-coded world model + BFS (§1.6) | ❌ (genuinely LLM-dependent) | → no modelless analog; out of scope for this workflow |

### 2.3 §3.5 Modelless-unblock check (mandatory before any riir-train deferral)

PRO-LONG is not a training paper — it is a context-management paper. No §3.5 deferral to riir-train is considered. The only LLM-dependent slice (spontaneous world-model coding, §1.6) is a *consequence* of the access pattern, not a training method. The access pattern itself is modelless.

### 2.4 Latent vs raw boundary (mandatory check)

PRO-LONG's log is **raw** (board states as text grids, actions as JSON, scores as numbers). The modelless instantiation preserves this:
- **Log entries:** raw (tick, action_tag, action_params, score, state_hash) — `TrialRecord` shape, syncable, replay-deterministic.
- **Query predicates:** may be latent (dot-product onto a "relevance" direction vector + sigmoid) OR raw (regex on action_tag, tick range, score threshold). The raw predicates are the direct PRO-LONG analog; the latent predicates compose with `experience_graph`'s vector-seeded traversal.
- **Sync boundary:** log entries cross as raw (BLAKE3-committed via existing `chain_engram_commit` / TrialLog hash chain); query results stay local (the predicate evaluation is local; only scalar outcomes cross sync).

No boundary violation. The existing raw/latent discipline holds.

### 2.5 Latent-space reframing (mandatory per skill)

How PRO-LONG's access pattern looks on each Super-GOAT factory module:

| Module | Reframing |
|---|---|
| **`EventLog` (katgpt-pruners)** | Add a `filter(predicate) -> impl Iterator<Item = &Event<A>>` + a `query_window(event_id_range, event_type_filter) -> &[Event<A>]` API. The `EvalCache` is already a content-addressed query; generalize to a predicate-addressed query. This is the open primitive. |
| **`TrialLog` (riir-games)** | Add `query_by_action_tag(pattern: &str) -> impl Iterator<Item = &TrialRecord>` + `query_by_tick_range(start, end) -> &[TrialRecord]` + `query_by_score_range(min, max) -> ...`. These are the deterministic analogs of "grep the log for score changes" (the paper's example). |
| **`experience_graph` (riir-neuron-db)** | Already ships latent-seeded NS traversal. PRO-LONG's programmatic search is a **complementary axis**: vector-seeded finds "events semantically similar to X"; programmatic search finds "events matching pattern P". A unified query layer composes both: `query(where=Predicate::And(Pattern(...), Semantic(similarity > threshold)))`. |
| **`engram_runtime` (riir-engine)** | Engram's hash-addressed lookup is a third axis (content-addressed, not pattern or semantic). The three axes (pattern / semantic / content-addressed) are orthogonal retrieval paradigms; PRO-LONG adds the pattern axis that is currently missing. |
| **DEC operators (katgpt-core)** | A lossless log IS a 1D cell complex (linear chain of events). Programmatic search = selecting a subcomplex by a predicate. The `d∘d=0` identity enforces "can't select the same event twice." This is a thin wrapper, not a new substrate. |

### 2.6 Fusion — what novel combination does this enable?

**Closest cousins across all seven repos:**

1. **R368 (AutoMem, GOAT)** — the closest verdict cousin. AutoMem's LOG/PLAN two-phase memory management decomposes the per-tick memory decision into "what to record" + "what to recall". PRO-LONG's thesis is more radical: **don't decide what to record at all — record everything; decide what to recall via programmatic search.** AutoMem optimizes the write decision; PRO-LONG eliminates it. The two are complementary, not competing: AutoMem's consult-before-write gate (the component that produced the 50.7% write/search ratio drop) can be composed ON TOP of PRO-LONG's append-all log (consult-before-write decides whether to ALSO write a structured note, not whether to append to the log).
2. **R300 (Trellis/Experience Graph, Gain → Super-GOAT after PoC)** — the closest shipped query layer. `experience_graph` does latent-seeded NS traversal (vector ANN + KG edge walk). PRO-LONG adds the **pattern-based search axis**. The fusion: a unified query layer that composes all three retrieval paradigms (pattern / semantic / content-addressed) over the same lossless log substrate.
3. **R169 (AgentMemBench, PASS)** — the validation-signal cousin. F6 (raw > compressed for retrieval faithfulness) and F7 (late filtering > early filtering) directly validate PRO-LONG's thesis. F9 (conservative consolidation > aggressive) warns against over-eager compaction of the lossless log.
4. **R024 (δ-Mem)** — the online associative memory substrate. δ-Mem's `DeltaMemoryState` is a compressed/feature-hashed view; PRO-LONG's lossless log is the uncompressed source it consolidates FROM. The Raven/δ-Mem consolidation pipeline is the "sleep cycle" that PRO-LONG doesn't need (it searches the raw log directly).
5. **R278 (Engram)** — the conditional memory Super-GOAT. Engram's `EngramTable::lookup_into` is content-addressed (hash → slot); PRO-LONG's programmatic search is pattern-addressed (predicate → matching events). Orthogonal axes.

**Fusion idea — Gain-tier, actionable:**

> **Deterministic Query Combinator over Lossless Log** — extend the shipped `EventLog<A>` + `TrialLog` with a query API that lets a modelless consumer express pattern-based search without an LLM in the loop. Three predicate types compose:
> 1. **Raw pattern predicates** (direct PRO-LONG analog): `action_tag matches regex`, `tick in [start, end]`, `score > threshold`, `event_type == Action`.
> 2. **Semantic predicates** (composes with `experience_graph`): `cosine(event.payload_embedding, query_embedding) > threshold`.
> 3. **Content-addressed predicates** (composes with Engram): `hash(event) == target_hash`.
>
> The combinator `EventLog::query(where = Predicate::And(...)) -> impl Iterator<Item = &Event<A>>` is the modelless analog of "coding agent greps the log." It lands in `katgpt-pruners` (where `EventLog` already lives) as a public primitive, with game-domain query helpers in `riir-games` (where `TrialLog` lives) and latent-predicate bridges in `riir-engine` (where `experience_graph` lives).
>
> This fuses `EventLog` (katgpt-pruners) × `TrialLog` (riir-games) × `experience_graph` (riir-neuron-db) × Engram (riir-engine) × Raven/δ-Mem consolidation. It does NOT unlock a new capability class (cross-session memory already ships via `experience_graph`) — it adds a **retrieval axis** (pattern-based) that complements the existing axes (semantic, content-addressed). Gain, not Super-GOAT.

---

## 3. Verdict

**Tier: Gain.**

The lossless append substrate ships (multiple instances). The latent-seeded query layer ships (`experience_graph`, OFFLINE ONLY). The **programmatic/pattern-based search axis** — PRO-LONG's distinctive contribution, validated by R169 F6/F7 + PRO-LONG Table 1 — does NOT ship as a coherent primitive. A deterministic query combinator over the lossless log is the modelless analog (per R368), composable with the existing query axes.

| Criterion | Honest answer |
|---|---|
| **Mechanism ships?** | **Partially** — lossless append ships (`EventLog`, `TrialLog`, `ExperienceLog`); latent-seeded query ships (`experience_graph`); programmatic/pattern search does NOT ship (grep confirms zero hits on query DSL patterns). |
| **Actionable improvements?** | **YES** — a concrete missing feature (deterministic query combinator over lossless log) with a clear PoC target on our existing substrate. |
| **§1.55 actionable bar?** | **YES** — the gap is "missing feature motivated by the paper in shipped code" (the `EventLog::iter()` linear scan is the explicit bottleneck PRO-LONG's grep/Python addresses). |
| **Super-GOAT?** | **NO** — Q2 fails (cross-session memory capability already ships via `experience_graph`; this adds a retrieval axis, not a new class); Q3 fails ("NPCs grep their history" is developer-facing, not a product differentiator); Q4 partial (connects EventLog + TrialLog + experience_graph + Engram, but these are already composed via Raven/δ-Mem consolidation). |

### One-line reasoning

The lossless append substrate ships, the latent-seeded query layer ships, but PRO-LONG's distinctive programmatic/pattern search axis over the lossless log does NOT ship — Gain, with a clear modelless analog (deterministic query combinator) composable with the existing retrieval axes.

### Why not PASS

The initial temptation (per R169's failure mode) is to PASS as "LLM-orchestration layer, ≥1 LLM call/step, violates 20Hz budget." Wrong per R368: the access pattern (lossless write + programmatic read) is implementation-agnostic; the LLM is one instantiation. The substrate ships (`EventLog`); the query axis does not. That is a concrete missing feature, not "out of scope." Per §1.55, actionable = Gain.

### Why not Super-GOAT (yet)

Q1–Q4 novelty gate:
- **Q1 (No prior art?):** Borderline YES for the coherent pattern (grep confirms zero hits on query DSL patterns over logs). But the components ship (EventLog append, experience_graph query).
- **Q2 (New class of behavior?):** **NO** — `experience_graph` already does cross-session NPC experience reuse via vector-seeded traversal (R300, Plan 319, landed). Adding pattern-based search is a different retrieval axis, not a new capability class.
- **Q3 (Product selling point?):** **NO** — "NPCs can grep their own history" is a developer-facing feature, not a product differentiator. The product differentiator (cross-session memory) already ships.
- **Q4 (Force multiplier?):** Moderate — connects EventLog + TrialLog + experience_graph + Engram. But these are already connected via Raven/δ-Mem consolidation.

Q2 + Q3 fail → not Super-GOAT. The fusion idea (§2.6) is recorded as Gain-tier actionable, not a Super-GOAT candidate.

### §3.6 PoC requirement — NOT triggered

Per §3.6, a PoC is mandatory only for verdicts that assert quality parity ("matches", "competitive with") or Super-GOAT claims. This verdict asserts neither:
- **Not asserting parity** — we explicitly do NOT ship the programmatic search axis; no parity claim is possible.
- **Not Super-GOAT** — Q2/Q3 fail.

A PoC becomes mandatory IF this Gain is ever promoted to GOAT/Super-GOAT (e.g., if a future benchmark shows pattern-based search unlocks a capability that vector-seeded traversal cannot reach). Track in an `.issues/` entry if promoted.

---

## 4. Routing

- **Open primitive** → `katgpt-rs/crates/katgpt-pruners/src/event_log.rs` (where `EventLog<A>` already lives). Add `filter`/`query`/`query_window` API behind a new feature flag (`event_log_query`). The primitive is generic (works over any `A: Clone + Debug`), no game semantics.
- **Game-domain query helpers** → `riir-games/src/trial_log.rs` (where `TrialRecord` lives). Add `query_by_action_tag` / `query_by_tick_range` / `query_by_score_range` thin wrappers over the open primitive.
- **Latent-predicate bridge** → `riir-engine` (where `experience_graph` consumer lives). Add a bridge that composes pattern predicates with semantic predicates (`cosine(event.payload_embedding, query_embedding) > threshold`).
- **No private guide** (Gain, not Super-GOAT per §1.5).
- **No plan this session** (Gain identifies the gap + the modelless analog; a plan opens when a consumer materializes — e.g., a per-NPC cognition runtime that needs pattern-based trajectory search, or a consolidation pipeline that needs "find all events matching pattern P in the last N ticks").

---

## 5. What This Is NOT

1. **Not a training paper.** PRO-LONG ships no optimizer, no loss, no curriculum. It is a context-management framework. No riir-train deferral.
2. **Not a claim that we ship parity.** Per §3.6, we explicitly do NOT ship the programmatic search axis. The lossless append substrate ships; the query axis does not. No parity claim is made.
3. **Not a Super-GOAT.** The cross-session memory capability already ships via `experience_graph` (R300). PRO-LONG adds a retrieval axis (pattern-based), not a new capability class.
4. **Not a claim that the LLM-coded world-model finding (§1.6) is modellessly distillable.** The spontaneous transition-function + BFS coding is genuinely LLM-dependent (semantic code generation, same as AIDE² R440). That slice has no modelless analog. The *access pattern* (search the log) is the distillable primitive; the *emergent world-model coding* is a benefit that requires the LLM and is out of scope.

---

## Cross-references

- **Closest verdict cousin:** `katgpt-rs/.research/368_AutoMem_Metamemory_LLM_Orchestration_PASS.md` — same R368 LLM-orchestration-vs-modelless lesson; AutoMem optimizes the write decision (LOG/PLAN), PRO-LONG eliminates it (append-all). Composable, not competing.
- **Closest shipped query layer:** `riir-neuron-db/.research/300_Experience_Graph_Database_Foundation_Gain.md` + `301_Experience_Graph_Super_Goat_Guide.md` — the R300 database lesson + the latent-seeded NS traversal that already ships. PRO-LONG adds the pattern-based axis.
- **Validation signals:** `riir-ai/.research/169_Agent_Native_Memory_Benchmark_PASS.md` — F6 (raw > compressed) + F7 (late filtering > early filtering) directly validate PRO-LONG's thesis; F9 (conservative consolidation) warns against over-eager log compaction.
- **Substrate:** `katgpt-rs/.research/024_Delta_Mem_Online_Associative_Memory.md` (δ-Mem, the online associative memory substrate), `katgpt-rs/.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md` (Engram, content-addressed conditional memory).
- **Shipped substrate (code):** `katgpt-rs/crates/katgpt-pruners/src/event_log.rs` (`EventLog<A>`), `riir-ai/crates/riir-games/src/trial_log.rs` (`TrialRecord`/`TrialLog`), `riir-neuron-db/src/experience_graph/` (`ExperienceLog` + `ExperienceGraph` + `ExperienceNode`).

## Re-evaluation guard

This note exists to prevent a future agent from re-running the full mandatory pre-flight + 7-repo fusion search on the same paper. If you arrived here from a grep, the verdict is **Gain**; do not re-distill unless the paper has a new version with a novel mechanism. The gap (programmatic/pattern search over lossless log) is recorded; the modelless analog (deterministic query combinator) is identified; the routing (open primitive in `katgpt-pruners`, game helpers in `riir-games`, latent bridge in `riir-engine`) is specified. A plan opens when a consumer materializes.

## Addendum (2026-10-04) — the law is now twice-ablated; the first consumer is filed

Han, Hu, Qiu, Wu & He [arXiv:2610.02200 "VISTA: A Visual Harness for Reasoning in an Interactive World"] provide the **second independent ablation of the same law in the visual domain**: lossless raw-frame memory + model-directed inspection is worth **+24.05 RHAE** of VISTA's ARC-AGI-3 harness ladder (70.05 → 94.10, the single largest component; +4.9 more from exact pixel readout) — against PRO-LONG's +15.2 via programmatic search. Convergence across modality AND retrieval actor (grep vs VLM-directed crop) isolates the **access pattern** as the value: lossless append + indexed retrieval + reasoner-decided queries. The first consumer materialized: **riir-ai Plan 621** (episodic sighting ring + surprise-gated re-derivation for the think brain — also this note's Phase-3-T3.1 cognition consumer); the neuron-db symmetric case is filed as riir-neuron-db Issue 626. Full distillation: `riir-ai/.research/395`.

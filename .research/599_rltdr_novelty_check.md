# Research 599: RLTL;DR (arXiv:2609.37633) — Novelty Check Against Published Prior Art

**Status:** RECORD (web-search novelty sweep complete, 2026-10-01)

**Candidate paper:** RLTL;DR: Self-improvement by Internalizing Self-generated Feedback (Apple, Sep 2026, arXiv:2609.37633). LLM agent RL where failed attempts + verifier outputs → self-generated one-sentence "insight" → conditions next rollout → SFT loss on insight tokens internalizes the task→insight mapping; reduced form trains ONLY on (task, insight) tuples (no rollouts needed).

**Method note:** Web-search-only sweep (web_search_prime); several queries timed out or returned noise; timeouts retried with re-phrasings. Absence below = "not found via search", never "does not exist".

**Addendum (2026-10-01, verdict-review round 1):** targeted second pass on claim 7 found the near-miss the sweep missed — **Generative Agents (Park et al., arXiv:2304.03442)**: the reflection mechanism (episodic stream → synthesized higher-level ideas → recency+relevance+importance retrieval, in a simulated agent town) IS episode→generic consolidation + similarity retrieval for simulated agents. Claim 7's mechanism novelty is therefore RETRACTED; the surviving delta is the engineering envelope only (crowd scale at 20Hz, two-brain sync discipline, L1 literal-density admission filter, L4 competence-gated consultation) — recorded in riir-ai `.issues/1022`.

---

## Claim 1 — "Generic short lessons transfer better than detailed feedback traces when stored in agent memory"

**VERDICT: NOT FOUND as a published ablation finding.** Adjacent work exists on memory granularity and rule-based guidance, but no paper found that isolates "generic vs. detailed" as a transfer-quality ablation.

Closest:
- **Coarse-to-Fine Grounded Memory (CFGM)** — arXiv:2508.15305 (Aug 2025). LLM agent planning with coarse-to-fine memory granularity levels. Closest on the "granularity" axis, but it is a memory-structure architecture, not an abstraction-level transfer ablation. https://arxiv.org/abs/2508.15305
- **AutoGuide** — Fu et al., NeurIPS 2024 (arXiv:2403.04988). Generates concise natural-language guidelines from offline experiences, selected per state — shows concise rules help, does not ablate against detailed traces. https://proceedings.neurips.cc/paper_files/paper/2024/file/d8efbb5dd415974eb095c3f06bff1f48-Paper-Conference.pdf
- **Meta-Policy Reflexion (MPM)** — arXiv:2509.03990. Structured reusable rules derived from past experience (lightweight, prompt-only). Same shape as above — reusable rules, no generic-vs-detailed ablation. https://arxiv.org/html/2509.03990v1
- **MemHarness** — arXiv:2607.28272 (Jul 2026). Shifts memory-augmented agents from verbatim replay to state-conditioned reconstruction — argues verbatim (detailed) replay is the wrong unit, adjacent evidence that detail level matters. https://arxiv.org/html/2607.28272v1

**Closeness:** MEDIUM — the ecosystem clearly studies concise rules and memory granularity, but the specific directional finding "generic transfers better than detailed" was not found published.

## Claim 2 — "Runtime insight database with similarity retrieval for untrainable models"

**VERDICT: PRIOR ART FOUND (rich).** This is a well-populated area; the three papers the candidate cites all exist, and there is substantial additional prior art.

Cited-by-paper (all verified):
- **MemRL** — arXiv:2601.03192. "Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory" — frozen LLM + non-parametric runtime RL over episodic memory; continuous refinement of Q-values from environmental feedback; Two-Phase Retrieval filtering noise to find high-utility strategies. https://arxiv.org/abs/2601.03192
- **WikiSkill** — arXiv:2608.27454. "Compiling Agent Experience into Persistent Knowledge for Skill Evolution" (Google) — co-evolves agent skills with a persistent wiki KB; three layers (immutable traces / active skill file / wiki between them). https://arxiv.org/abs/2608.27454
- **CORE** — arXiv:2605.28742. "Contrastive Reflection Enables Rapid Improvements in Reasoning" — non-parametric learning algorithm using contrastive (success-vs-failure) reflection. https://tldr.takara.ai / https://www.alphaxiv.org (CORE; code: github.com/LinasNas/core-reasoning)

Additional prior art (not cited, found by search):
- **ExpeL** — arXiv:2308.10144 (AAAI 2024). The canonical 2023 prior art: autonomously gathers experience, extracts natural-language insights, retrieves task-relevant insights at inference (similarity retrieval). VERY close to the "runtime insight database with similarity retrieval" claim. https://arxiv.org/abs/2308.10144
- **Agent Workflow Memory** — arXiv:2409.07429 (Wang et al., 2024). Induces and reuses evolvable sub-workflows from experience. (Verified via Agent KB citation.)
- **Voyager** — arXiv:2305.16291 (2023). Skill library of verified programs, lifelong accumulation, retrieval by embedding similarity. https://arxiv.org/abs/2305.16291
- **Agent KB** — arXiv:2503.02444 (approx; OpenReview + arXiv confirmed, "Leveraging Cross-Domain Experience for Agent Learning"), hierarchical Reason-Retrieve-Refine memory shared across agent frameworks. https://openreview.net/forum?id=... (hierarchical memory framework)
- **Online Experiential Learning (OEL)** — arXiv:2603.16856. Continuously improve from own deployment experience. https://arxiv.org/pdf/2603.16856
- **Experiential Reflective Learning** — arXiv:2903.24639 [typo guard: 2603.24639]. Insights extraction from successful trajectories, retrieved few-shot at inference. https://arxiv.org/abs/2603.24639
- **From Memory to Skills** — arXiv:2607.16621. Converts prior traces into executable skills (evidence-grounded co-evolution). https://arxiv-memory-to-skills / https://arxiv.org/html/2607.16621v1
- **SAMem** — ACL 2026 Findings. State-aware structured memory for situationally focused guidance. https://aclanthology.org/2026.findings-acl.722.pdf

**Closeness:** HIGH — the mechanism (insight/experience DB + similarity retrieval at runtime, weights frozen) is squarely prior art (ExpeL 2023, Voyager 2023, MemRL 2026, WikiSkill 2026, CORE 2026). RLTL;DR's cited set is accurate but incomplete; ExpeL/Voyager/AWM are prominent omissions.

## Claim 3 — "Conditioning next fix attempt on distilled lessons from failed attempts, for code repair / compile-error fixing"

**VERDICT: PRIOR ART FOUND (mechanism in-domain, lesson-persistence variants differ).**

Closest:
- **Self-Debugging** — arXiv:2304.05128 (Google/Berkeley, ICLR 2024). Teaches LLMs to debug predicted programs via execution-feedback-driven refinement — conditions next fix attempt on feedback (within-episode). No persistent cross-task lesson memory. https://arxiv.org/abs/2304.05128
- **RustAssistant** — arXiv:2308.05177 (Microsoft; ICSE 2025). Iterative repair loop for Rust compilation errors: repeatedly queries LLM and feeds compiler diagnostics back until it compiles — the exact domain (rustc errors, iterative fix loop), no learned lesson memory. https://arxiv.org/abs/2308.05177
- **InferFix** — (Microsoft; "End-to-End Program Repair with LLMs integrated in CI" is the follow-up; original retrieval-augmented repair via dual-task prompts, ~arXiv:2303.07263 — ID not re-verified this session). Retrieval of similar past fixes conditions repair — closest to "experience-conditioned repair". (https://siesta.si.usi.ch listing)
- **Reflexion** — arXiv:2303.11366 (NeurIPS 2023). Verbal self-reflection on failure maintained in episodic memory, conditioning next attempt — the general mechanism (incl. code tasks like HumanEval), pre-dating the insight-DB framing. https://arxiv.org/abs/2303.11366
- Also: "Revisit Self-Debugging with Self-Generated Tests" (ACL 2025) — self-debugging + self-generated tests; "Training LLMs to Better Self-Debug and Explain Code" (NeurIPS 2024) — fine-tunes models to use execution feedback.

**Closeness:** HIGH for the loop (feedback-conditioned retry is mature prior art); MEDIUM for the memory half (persistent distilled lessons across tasks for compile-error repair specifically — Self-Debugging/RustAssistant are within-episode; Reflexion/ExpeL are cross-task but not compile-error-specific).

## Claim 4 — "Sequential insight-conditioned sampling beats i.i.d. rollouts for exploration"

**VERDICT: PRIOR ART FOUND (closely).** The candidate's own protocol line ("next rollout conditioned on all previous insights, sequentially sample until solution") has a very close relative in hint-conditioned RL.

Closest:
- **HiLL — Learning to Hint for Reinforcement Learning** — arXiv:2604.xxxxx (Apr 1, 2026; exact ID not captured — "Learning to Hint for Reinforcement Learning", proposes HiLL: jointly trains hinter policy + reasoner policy during RL; hints generated online conditioned on the current reasoner's INCORRECT rollout). This is nearly the same mechanism: next-attempt conditioning on failure-derived hints. https://arxiv.org (Apr 1, 2026) / https://huggingface.co/papers (Learning to Hint for Reinforcement Learning)
- **RLTF / RLTF-SD** — arXiv:2602.02482. Trains single-turn policy to match its own feedback-conditioned second-turn generations — sequential feedback-conditioning internalized. https://github.com/lili-chen/rltf
- **Awesome LLM Hint-based RLVR** (github.com; curated list) — documents the hint-conditioned-RLVR family (hints = signals beyond what ordinary iid rollouts under the current policy can produce).
- **GHPO** — adaptive guidance balancing imitation (for problems beyond reach) with exploration RL — difficulty-adaptive, relevant to the exploration story. https://arxiv.org (GHPO: Adaptive Guidance for Stable and Efficient LLM RL)

**Closeness:** HIGH — hint-generation conditioned on the current failed rollout + joint training is published (HiLL, 2026); the specific "sequential vs iid rollouts" ablation framing was not found isolated, but the mechanism is not novel.

## Claim 5 — "SFT on (task, hint) tuples without rollouts" / context distillation

**VERDICT: PRIOR ART FOUND (rich, and the candidate cites it accurately).**

Cited-by-paper (all verified):
- **Context distillation origin** — Askell et al. 2021, "A General Language Assistant as a Laboratory for Alignment" (arXiv:2112.00861) — first context-distillation formulation. https://arxiv.org/abs/2112.00861
- **Learning by Distilling Context** — Snell et al. 2022, arXiv:2209.15189 — "context distillation is a general method... internalize 3 types of training signals", incl. internalizing performance gains. https://arxiv.org/abs/2209.15189
- **SDPO** — arXiv:2601.20802. Self-distillation policy optimization: same model as teacher and student under different conditioning contexts. (Verified via RL-for-LLMs wiki + "Rebellious Student" description.) https://huggingface.co (RL-for-LLMs wiki entry)
- **RLTF** — arXiv:2602.02482. Reinforcement Learning from Text Feedback; RLTF-SD variant trains single-turn policy to match its own feedback-conditioned second-turn outputs — the "internalize the conditioned behavior into the unconditioned policy" move. https://github.com(ins 2">lili-chen/rltf)
- **ECHO** — arXiv:2605.24517 (Microsoft). Environment Cross-entropy Hybrid Objective: policy-gradient on action tokens + on-policy cross-entropy on environment-observation tokens — SFT-style loss on feedback/observation tokens alongside RL. https://arxiv.org/abs/2605.24517
- **Programming by Backprop (PBB)** — Cook, ICLR 2026. "An Instruction is Worth 100 Examples When Finetuning LLMs" — acquire procedural knowledge from declarative instructions via backprop (no rollouts). https://proceedings.iclr.cc (ICLR 2026) / github.com/jonathan-cook235/Programming-by-Backprop

Additional/newer found by search:
- **Prompt Injection: Parameterization of Fixed Inputs** — (Choi et al.) injecting fixed prompts into parameters as an ICL alternative. https://openreview.net
- **Efficient LLM Context Distillation** — arXiv (2026): context distillation vs ICL vs FT comparison. https://arxiv.org
- **Flux-OPD** — on-policy distillation with evolving contexts (Aug 2026). https://www.researchgate.net
- **HiLL** (above) — jointly trained hinter + reasoner.
- **tinker-cookbook Prompt Distillation** (Thinking Machines; repo) — practical context/prompt distillation, cites [1,2] = Askell/Snell lineage.

**Closeness:** HIGH — "SFT on (task, insight) tuples, no rollouts" is a context-distillation/PBB-shaped contribution; the candidate's citation set is accurate. Novelty would have to live in the task→insight mapping being self-generated during RL, not in the reduced-form training.

## Claim 6 — "Success-rate-gated memory consultation (goldilocks zone)"

**VERDICT: NOT FOUND as a published finding.** Adjacent work on memory-admission gating and difficulty-adaptive guidance exists, but the specific success-rate-gated "goldilocks zone" consultation policy was not found.

Closest:
- **A-MAC** — arXiv:2603.04549. "Adaptive Memory Admission Control for LLM Agents" — scores candidate memories across five interpretable dimensions (which memories to ADMIT, not when to consult). https://arxiv.org/html/2603.04549v1
- **GHPO** — adaptive guidance: balances imitation vs exploration RL by whether problems are currently beyond the model's reach — a difficulty/capability gate on guidance, adjacent to the goldilocks idea. https://arxiv.org (GHPO)
- **Evaluating Memory Structure in LLM Agents / StructMemEval** — arXiv, Sep 10 2026 (ICLR submission). Uses hint-insertion as a diagnostic: "if an agent fails a problem as-is but solves it reliably with the hint..." — diagnostic use of hint insertion, not a consultation-gating policy. https://arxiv.org / https://iclr.cc (Apr 27, 2026)
- (Product blog, non-paper: mem0.ai "Proactive Memory" — gating layer with demand detector deciding when retrieval is worth paying for.)

**Closeness:** MEDIUM — gating WHAT enters memory (A-MAC) and gating guidance by difficulty (GHPO) are published; gating WHEN to consult memory by task success-rate, with a goldilocks-zone finding, was not found.

## Claim 7 — Insight-memory / lesson-internalization applied to GAME NPCs or crowd AI

**VERDICT: NOT FOUND.** LLM-NPC work exists (personality, dialogue, RL-trained behaviour) but nothing found on insight-memory or lesson-internalization for NPCs/crowds.

Closest:
- **LLM-Guided RL for Adaptive NPC** — arXiv:2609.02931. RL for adaptive NPC behaviour with LLM guidance; no lesson memory / internalization. https://arxiv.org/html/2609.02931v1
- **LLM-Based Behavior Agent with Natural Language Personality** (Tarigan 2025, ETASR) — personality-driven NPC behaviour. https://etasr.com/index.php/ETASR/article/view/12631
- **Memory-Based Advantage Shaping for LLM-Guided RL agents** — arXiv:2602.17931 (AAAI) — memory graph of subgoals+trajectories shaping advantage for LLM-guided RL agents (game/robotics domains; not insight-internalization into weights, not crowd). https://arxiv.org/abs/2602.17831 [typo guard: 2602.17931]
- Game-agent reflection memory: searches ("LLM game agent learning from defeat reflection") returned nothing on-point; GITM/Voyager-style skill libraries are the nearest family but are not NPC/crowd products.

**Closeness:** LOW — no direct prior art found for lesson-internalized game NPCs / crowd AI. This appears genuinely open (matches our riir-games / NPC cognition roadmap).

---

## Summary table

| # | Claim | Verdict | Closest prior art |
|---|---|---|---|
| 1 | Generic lessons > detailed traces (ablation) | NOT FOUND | CFGM 2508.15305; AutoGuide 2403.04988; MPM 2509.03990; MemHarness 2607.28272 |
| 2 | Runtime insight DB + similarity retrieval | FOUND (rich) | ExpeL 2308.10144; MemRL 2601.03192; WikiSkill 2608.27454; CORE 2605.28742; Voyager 2305.16291; AWM 2409.07429 |
| 3 | Lesson-conditioned retry for code/compile repair | FOUND (loop in-domain) | Self-Debugging 2304.05128; RustAssistant 2308.05177; Reflexion 2303.11366; InferFix |
| 4 | Sequential insight-conditioned sampling > iid | FOUND (close) | HiLL (Apr 2026); RLTF 2602.02482; GHPO; hint-based RLVR family |
| 5 | SFT on (task,hint) tuples / context distillation | FOUND (rich) | Askell 2112.00861; Snell 2209.15189; SDPO 2601.20802; RLTF 2602.02482; ECHO 2605.24517; PBB (ICLR 2026) |
| 6 | Success-rate-gated memory consultation | NOT FOUND | A-MAC 2603.04549; GHPO; StructMemEval (Sep 2026) |
| 7 | Insight-memory for game NPCs / crowd AI | NOT FOUND | LLM-NPC personality (Tarigan 2025); LLM-guided RL NPC 2609.02931; Memory-based Advantage Shaping 2602.17931 |

## Honest caveats

- Several searches timed out or returned noise; "NOT FOUND" means "not found via this search session", not proven absence. Claims 1, 6, 7 deserve a second-pass with Google Scholar / Semantic Scholar / arXiv listing pulls before declaring open territory.
- The HiLL arXiv ID and the original InferFix ID were not pinned this session — verify before citing.
- CORE's arXiv listing page returned empty on direct ID search; verified via secondary sources (tldr.takara.ai, GitHub code release, AI Insight Lab briefing) — ID 2605.28742 is consistent across them.
- One search snippet listed "arXiv:2903.24639" for Experiential Reflective Learning — that is a typo in this note's source; the correct ID from the primary snippet is 2603.24639.

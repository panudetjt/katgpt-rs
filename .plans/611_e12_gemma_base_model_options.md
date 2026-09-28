# E12 one-pager — the base-model recirculation check: accept the Gemma license, or retire the check

**Status:** DRAFT for owner decision (master `riir-ai/.issues/1016` row E12; this repo's `.issues/906_owner_gate_pickup.md`; evidence `.benchmarks/668_recirculation_goat_and_poc.md:118-131`). No option executed; the pick is the owner's.

## The decision in one paragraph

The recirculation PoC's negative verdict (12/12 recirc cells harmful at session scale) is scoped to
**IT-tuned** residual streams — the paper validated **base** models, and no public base
(non-IT) Gemma-2-family GGUF exists ungated (verified 2026-08-20 via the HF API; re-verified
2026-09-28 on BOTH boxes: the only Gemma GGUFs on the M3 and the 4090 are `gemma-2-2b-it-f16` and
`gemma-4-12B-it` — both IT-tuned). The only gated source is `google/gemma-2-2b-GGUF`
(`gated=manual`: Gemma license acceptance + an authenticated HF token). The owner picks:

- **Option (a) — accept + provision** (~10 GB disk on the 4090's E:, ~10 min of owner time, then a
  ~5-13 min session-scale PoC run): accept the Gemma license on `google/gemma-2-2b-GGUF`, create an
  HF **read** token, and put it in the 4090's env (one line in the box's PowerShell profile or a
  scheduled-task context var — the token must never be committed; the download is
  `Invoke-WebRequest` with the `Authorization: Bearer` header, target `E:\git\riir-train\data\`).
  Then the check runs and the negative verdict either stands AT BASE (final at scale per the paper's
  register) or flips (a real finding — the reopen condition did its job).
- **Option (b) — retire the check as permanently blocked** (zero cost): record in Bench 668 that the
  base-model check is retired-unrun, the negative verdict remains scoped to IT-tuned streams, and
  the recirculation primitive stays closed unless a base GGUF arrives by another route (e.g. a
  self-converted safetensors run — same token prerequisite, so no cheaper).

## Freshness fact found this pass (2026-09-28) — the OTHER reopen condition never collected either

The paper-scale Run 3 ("RUNNING detached on the 4090", launched 2026-08-20) **left no output**: the
scheduled task `recirc_paperscale` exists in state `Ready` (not `Running`, never re-fired), and
`E:\git\riir-ai\target\recirc\paperscale_out.txt` does not exist — the run either died before
writing or its output was cleaned with the target dir. Only its first two arm-lines (the 06:56
progress note) ever made it into Bench 668. So **both** reopen conditions in Bench 668 §caveats
(paper-scale re-run + base-model check) are unmet, and the negative verdict's scope caveats stand
exactly as written. This does not change option (a) vs (b) — but the owner should know the
paper-scale leg is not "in flight"; it is dead unless re-armed separately (out of E12's scope).

## What each option costs / buys

| | (a) accept + provision | (b) retire the check |
|---|---|---|
| Owner effort | license click + one token + one env line | one "retired" line (this file's verdict row) |
| Box cost | ~10 GB on E: (55 GB free at last record) | none |
| Knowledge | the verdict becomes final at the paper's own register, or a real sign-flip finding | the verdict stays honestly scoped to IT-tuned; the caveat paragraph in Bench 668 stays forever |
| Risk | a Google license acceptance binds the account (legal, not technical); the token is read-scoped and revocable | none — except the small chance the sign flips at base and nobody ever learns |

**Recommendation on file (agent, 2026-09-28): (b) retire.** The recirculation PoC measured 12/12
harmful cells at session scale with a stable per-arm rate; the IT-vs-base register caveat is a
refinement of a negative, not a plausible sign-flip; and the stack's active lanes (Bonsai ternary,
GDN) do not route through recirculation. If the owner disagrees, (a)'s runbook is the four steps in
Bench 668 §"Base-model check" verbatim — the only new fact this pass adds is "download to
`E:\git\riir-train\data\`" and "verify `from_gguf` accepts the f32 type id first" (already recorded
there).

## Verdict row (owner fills one line)

- [ ] **(a)** accept + provision — date: ______ , token location: box env only
- [ ] **(b)** retire the check as permanently blocked — date: ______

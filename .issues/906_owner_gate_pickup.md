# Owner-gate pickup — katgpt-rs

**Status:** OPEN — this repo's remaining owner-gated items. Public-repo hygiene note: the cross-workspace summary that briefly lived here as `905_workspace_owner_gated_decisions_summary.md` (2026-09-28) was relocated to the private workspace hub the same day and is deliberately NOT referenced from this public repo — no private paths, hashes, or decision content belong here. Number 905 is SPENT — never reuse it.

## Items

- [x] E12 — **ONE-PAGER PREPPED 2026-09-28** (`.plans/611_e12_gemma_base_model_options.md`): both options costed + a verdict row for the owner; agent recommendation on file is **(b) retire** (the negative is 12/12 harmful cells with a stable rate; the IT-vs-base caveat refines a negative, it does not flip it). Freshness facts found while prepping: the paper-scale Run 3 NEVER COLLECTED (task `Ready`, no output file — addendum landed in Bench 668), and both boxes re-verified to carry only IT-tuned Gemma GGUFs. **OWNER-GATED: the (a)/(b) pick lands on the owner's mark.**
- [ ] E13 — workspace `rust-version` pin (evidence `HISTORY.md:3148`): prep the `[workspace.package] rust-version` patch + the list of ~30 affected manifests; land only on owner go.

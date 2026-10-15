# Plan 616 — promote `minmax_partition` to katgpt-core as an open primitive

**Status:** COMPLETE 2026-10-02 (T0–T6 landed: katgpt-rs `4ac1448c6`, riir-infer landing commit — the primitive + pair-distance substrate promoted verbatim, riir-infer's own copy died same commit; deviations T3-fixture-source + G4-ceiling recorded in-task; docs gate 35/35, twt gates 15/15 through the shim, bench bench_616 PASS with determinism digest `92bae2a0…`)

## Why now

The TWT PoC closed 2026-10-01 with the partition machinery battle-proven on
three fronts: it priced a real 64-layer ternary checkpoint end to end
(11 ε points, m(ε) machined-monotone, 8 measured known answers), it drove a
typed variant (`minmax_partition_typed`) whose tier-(i) rule became the
audition's structural gate, and it fed a validated collapsed checkpoint
(agreement 0.9486 at the 4.7% cut — the PoC's only passing point). The
primitive is generic — min-max partition of an ordered sequence under a
pairwise-distance oracle with ε-free (k-free) granularity — and Research
594's own table (row 2) pre-committed this promotion: *"open-primitive
promotion to katgpt-core after the PoC (single implementation home)"*.

## The DRY question the plan must settle FIRST (substrate-first, T0)

katgpt-core already ships a contiguous K-block DP:
`crates/katgpt-core/src/ugc_schedule.rs:692 dp_partition(profile, k)` —
sum-of-costs oracle, **fixed k**, min-SUM objective. The TWT DP is
**min-MAX worst-case, ε-constrained, k free**. Research 594's recorded
rule: *"generalize dp_partition or land beside it with a cross-link; two
partition DPs that don't know about each other is the DRY violation."*

- [x] **T0.1** Read both DPs side by side; decide GENERALIZE (one
  implementation with an objective parameter) vs NEIGHBOR (two
  functions, one module, cross-linked docs + a shared type). The
  decision criterion: can `dp_partition`'s existing callers (UGC
  schedule) consume the generalized form byte-identically at their
  current call sites? If any behavior changes, NEIGHBOR wins — a
  promotion must not move a shipped consumer.
  **DECIDED 2026-10-02: NEIGHBOR.** The two DPs differ on THREE axes
  (objective min-SUM vs min-MAX, constraint fixed-k vs ε-feasibility,
  input UgcProfile-closure vs generic distance matrix) — a generalized
  form changes every `dp_partition` call-site signature, failing the
  criterion; and the UGC recurrence's value IS the paper's shape (Eq
  39). Cross-links landed in both module docs (`partition.rs` header +
  `dp_partition` doc).
- [x] **T0.2** Whatever the decision, ONE module home
  (`katgpt-core/src/partition.rs` proposed), one vocabulary, and the
  min-max DP keeps its riir-infer name and semantics verbatim —
  promote by MOVE, not copy (riir-infer `twt::partition` re-exports the
  katgpt-core items; its own copy dies in the same commit).
  **DONE: `crates/katgpt-core/src/partition.rs` hosts the DP verbatim;
  riir-infer `twt/partition.rs` is now a re-export shim + the lane-local
  kill rule (its copy died in the riir-infer landing commit).**

## Scope

- [x] **T1** Move `minmax_partition` + `minmax_partition_typed` +
  `brute_force_optimal_typed` + the `SMatrix` pair-distance accumulator
  + `partition_worst` into the chosen home, `#[repr]`/allocation shape
  unchanged (the 64×64 artifact path must stay byte-deterministic —
  the emit known-answer pins depend on it).
  **DONE: moved minmax_partition(_typed), partition_dp core,
  partition_worst, brute_force_optimal AND its typed twin (the untyped
  brute force is the G1 comparator — splitting the pair would strand the
  gate), forced_min_blocks, Block, SMatrix, PairCosineAccum,
  PartitionError.** Verbatim algorithms, identical float ops and
  index-ordered argmin. `SMatrix` gained `from_parts` (the builder
  path's eager-finiteness constructor — the private-field construction
  it replaces). PartitionError is hand-rolled Display/Error (the house
  convention — no thiserror in katgpt-core); the typed length-mismatch
  error leaves the lane's `TwtError::GgufWrite` mis-use for its own
  `TypesLengthMismatch` variant.
- [x] **T2** Feature flag `minmax_partition` (opt-in), katgpt-core
  default build unchanged; riir-infer forwards the feature
  (`twt_profile = ["katgpt-core/minmax_partition", …]` shape).
  **DONE: feature row with the full manifest-comment contract;
  riir-infer `twt_profile = ["katgpt-core/svd_cca",
  "katgpt-core/minmax_partition"]`. Default lib 2074 (off) / 2089 (on)
  — the off count is byte-unchanged.**
- [x] **T3** Cross-repo determinism pin: the 11-point m table
  (61,57,49,35,25,14,8,5,3,2,1 over the bonsai profile artifact) becomes
  a katgpt-core-side known-answer test keyed on a committed S-matrix
  FIXTURE (the artifact itself stays gitignored in riir-infer — commit
  the 2016-entry upper triangle as a small fixture file, BLAKE3-pinned).
  **DONE WITH A RECORDED DEVIATION: the bonsai 64×64 artifact lives on
  the M3 box (gitignored, absent here); the fixture committed instead is
  the gemma2-2b S-matrix — 26×26, 325 upper entries, measured
  2026-09-29 on THIS box (`twt_gemma2_profile`, corpus BLAKE3
  `44aabe8e…`), 1308 bytes, BLAKE3 `ecdc0d12…`. The pinned known answer
  is its derived m table [26,17,10,6,4,3,1] at the artifact's own 7-point
  grid + monotonicity + constraint post-conditions + the all-true-typed
  cross-check (typed == unconstrained at one type). The determinism
  CLAIM is unchanged: committed BYTES pin the DP, not capture
  provenance — and the bonsai 11-point table stays pinned where the
  artifact lives (riir-infer `twt_pareto_check` known answers).**
- [x] **T4** Consumers inventory (the four-oracle reuse list from
  Research 594 §6.4): riir-infer twt (existing), ternary group-scale
  boundaries under an activation-range ε, KV-cache segment tiers,
  tick-series regime segmentation. Each future consumer files its own
  issue; this plan only guarantees the primitive + one home.
  **RECORDED in the feature-row manifest comment; no consumer code
  beyond riir-infer's re-export (correct — each future consumer files
  its own issue).**
- [x] **T5** Bench + GOAT gate: DP cost is O(L²·m) worst case —
  microbench at L ∈ {16, 64, 256, 1024} over a synthetic oracle;
  GOAT gates G1 (partition optimality vs brute force at n ≤ 16 —
  the property the brute_force twin already asserts), G2 (DP runtime
  floor/ceiling vs the L² model), G3 (no regression: the two riir-infer
  call sites re-run their existing gates unchanged), G4 (alloc shape:
  zero alloc per call at fixed L — the artifact path is
  allocation-counted today). Promote to default ONLY on a GOAT pass;
  the flag-off posture stays byte-identical.
  **DONE with a RECORDED G4 DEVIATION: the DP allocates its scratch
  per call (the plan's "zero alloc" wording was aspirational — the DP
  was never alloc-free); the gate pins the MEASURED ceiling instead
  (4 untyped / 5 typed, printed on every pass — a silent ceiling over
  an unmeasured count is the green-zero shape). Bench `bench_616`
  PASS: planted recovery exact at L ∈ {16,64,256,1024}; runtime ladder
  0.4/8.0/32.3/1904 µs (n=1024 reads 3.68× the n² model — the 4 MiB
  worst-table crossing the cache boundary, not algorithmic; the 5 ms
  O(n³)-regression bound is far away); determinism digest
  `92bae2a0…` is the cross-box anchor. G1 brute-force agreement lives
  in the unit battery (exponential instrument, not bench material).
  G3: twt_gates 15/15 + twt_g4_alloc 1/1 through the shim; the ONE
  call-site edit is `t2_2_invalid_eps_refused` matching PartitionError
  (the error type moved with the primitive — the gate's claim, invalid
  ε refused, unchanged). NOT promoted to default (opt-in per the
  no-default-consumer rule; the flag-off posture is byte-identical by
  construction — the module compiles away).**
- [x] **T6** Docs: katgpt-core module doc + README feature-gate row +
  Research 594 row 2's promotion note flipped to SHIPPED (with commit).
  **DONE: module doc + lib.rs row + manifest comment + README counts
  666→668 (count_features aligned) + docs gate 35/35; Research 594
  flipped in the same push.**

## Non-goals

- No behavior change in the UGC schedule path (T0 decides, never edits
  its semantics).
- No new oracle implementations (consumers file their own).
- No promotion of the collapse WRITER/loader — that stays riir-infer
  (weights-domain substrate); this plan moves the pure partition
  primitive only.

## En-route incident (recorded, the Plan 596 class)

`rustfmt --edition 2024` on `src/twt/mod.rs` followed the `mod`
declarations recursively and reformatted four UNRELATED twt siblings
(audition/collapse_writer/delta/ternarize) plus fmt noise on the two
files this landing legitimately edits (smatrix/mod) and the gates file.
All UNRELATED reverts applied (git checkout); the two edited files were
re-edited after their revert so the final diff carries ONLY the intended
changes (+770/−112 over 6 files, every hunk this plan's). Repo-wide fmt
is not this plan's lane; the drifted-at-HEAD files stay exactly as the
sibling sessions left them.

## Provenance

- PoC record: `../riir-infer/.issues/022` (T5/T6), benches
  `022_t5_collapsed_goat_agreement.md` + `022_t55_pareto.md`.
- Research: `.research/594` §"MOAT gate" + §7 (the close-out), row 2 of
  the component table.
- Existing consumer: `../riir-infer/src/twt/partition.rs` (+ `s_matrix.rs`).

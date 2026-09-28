# Benchmark 668 — Recirculation operator GOAT + defend-wrong PoC (Issue 673)

**Date:** 2026-08-19
**Feature:** `recirculation` (opt-in — **the PoC quality axis FAILED on our substrate; default stays OFF**)
**Source:** Research 492 / arXiv:2608.17981 (Recirculation, Mozer et al., Google DeepMind).
**Modules:** `crates/katgpt-core/src/recirculation.rs` + `tests/recirculation_alloc_check.rs` (katgpt-rs); `crates/riir-poc/tests/recirculation_poc.rs` (riir-ai, the Phase 2 PoC); riir-engine enablers (`PostLayerHook` pub + `forward_one_token_hooked`).

## Phase 1 — operator GOAT gates

| Gate | Evidence | Verdict |
|---|---|---|
| **G1 determinism** | Fixed α + fixed buffers ⇒ **bit-identical repeat** (`deterministic_bit_identical_repeat`, 50 composed steps × D=128); the `l2_norm` 8-way unroll is a compile-time-constant association order (deterministic, latency-chained broken). | **PASS** |
| **G2 overhead** | `g2_step_mixture_under_1us_at_d2048` (release): **553.7 ns/step at D=2048** — 1.8× under the ≤1µs gate. (First implementation measured 3068.6 ns — a single-accumulator serial dependency chain; the fixed 8-way unroll cut it 5.5× with determinism preserved.) | **PASS** |
| **G3 no-regression** | Default lib suite 1897/0/6 exact baseline (module cfg'd away); `--all-features` clean; clippy 0 in default + feature + tests states. | **PASS** |
| **G4 alloc-free** | `recirculation_alloc_check` (CountingAllocator, release): capture+mix cross-step loop, 50,000 steps at D=2048 = **0 allocations** (the `RecircBuffer` is a one-time fixed-D allocation). | **PASS** |

Operator unit tests: convex+norm-match boundedness (triangle-inequality class), norm-match identity (`x/x == 1.0` exactly), ramp schedule + step-0 bit-identical no-op, β=1 non-convex exact-fp recurrence, cross-step composition boundedness over 200 synthetic steps, paper anchor pairs {11,4}/{18,9}/{35,16} round-trip at 26/34/48 layers.

## Phase 2 — defend-wrong PoC on gemma-2-2b-it (riir-poc)

**Setup:** `Gemma2PatchedForward::from_gguf(gemma-2-2b-it-f16.gguf)` + SentencePiece; serial token-by-token windows (the honest recirculation prefill mode); paper layer pair for 26 layers src=11/dst=4 (`RecircPair::for_depth`); ramp 10; NLL over each position's next token. Two local text registers as datasets (honestly labeled): **A** = katgpt-rs `.research/*.md` (arXiv-style prose), **B** = riir-ai `.docs/**/*.md` (long-form technical prose — PG19-substitute; PG19 not on disk).

**Run 1** (2×96 tokens/arm/dataset; dst_hook=4):

| dataset | baseline | α=0.07 | α=0.10 | α=0.15 | overwrite | temp-1.2 |
|---|---|---|---|---|---|---|
| A | 91.32 | 99.66 (−9.1%) | 104.80 (−14.8%) | 114.28 (−25.1%) | 2.2e8 (**clobbered**) | 68.38 (+25.1%) |
| B | 311.20 | 329.88 (−6.0%) | 342.66 (−10.1%) | 374.97 (−20.5%) | 5.4e8 (**clobbered**) | 197.16 (+36.7%) |

**Run 2 — injection-point robustness** (3×160 tokens/arm/dataset; `RECIRC_POC_DST_SHIFT=-1` → dst_hook=3, the pre-dst input-injection equivalent):

| dataset | baseline | α=0.07 | α=0.10 | α=0.15 | overwrite | temp-1.2 |
|---|---|---|---|---|---|---|
| A | 339.03 | 361.97 (−6.8%) | 380.10 (−12.1%) | 432.03 (−27.4%) | 4.4e9 (**clobbered**) | 233.87 (+31.0%) |
| B | 342.44 | 370.27 (−8.1%) | 389.42 (−13.7%) | 439.43 (−28.3%) | 8.5e9 (**clobbered**) | 233.11 (+31.9%) |

### Verdict

- **Gate (b) FAILS: ppl reduction > 0 on ≥2 datasets — recirculation INCREASES ppl on our substrate**, dose-dependently in α, at BOTH injection points, on BOTH registers (12/12 recirc cells harmful). The paper's Gemma3-family training-free gains do **not** transfer to gemma-2-2b-**it** under this harness.
- **Safety check PASSES: recirc is strictly safer than the R417 overwrite** at equal pairs (mixture −7…−28% vs overwrite −10⁸% — the clobbering catastrophe, reproducing Plan 431's failure mode on a real model, the strongest possible contrast arm). The operator's *relative* semantics are exactly as designed.
- **temp-1.2 control: ppl IMPROVES** (+25…+37%) — same direction as the paper's temp control (−8.5% alone there), so the harness measures real distribution effects; recirculation's harm is not a temperature artifact.

### Honest caveats (recorded, binding for any reopen)

1. **Session-scaled counts** (192–480 tokens/arm; paper-scale is ~500×1024): the direction is consistent and monotone in α across 12 cells, but the magnitudes are small-sample.
2. **Model register:** gemma-2-2b-**it** (instruction-tuned), not a base Gemma2/Gemma3; the paper validated base models. IT-tuned residual-stream statistics may differ.
3. **Data register:** local markdown corpora (code fences, tables), not wikitext/PG19 prose.
4. **Reopen conditions:** paper-scale re-run (`RECIRC_POC_WINDOWS=50 RECIRC_POC_WINDOW_TOKENS=512`) + a base-model check (the 4090 + a base Gemma-family GGUF) before citing this negative as final at scale.

## Phase decisions (per the issue's gates)

- **Phase 3 (decode-stack integration): DOES NOT PROCEED** — gated on Phase 2 PASS; it failed.
- **Phase 4 promotion: default stays OFF** — G1–G4 pass at the operator level, but the quality axis failed on our substrate. The operator ships opt-in as substrate (the mixture semantics are measurably the safe member of the relocation family).
- **Phase 5 (belief-recirculation guide): DOES NOT TRIGGER** (quality axis unproven).
- **Cost accounting (recorded per the issue):** decode = 2 stack instances/step (~2× FLOPs serial, 2× KV footprint); prefill serial. The PoC measured ~8.7 tok/s serial prefill on M3 CPU for the 2B f16 — the KV doubling was never reached (integration not warranted).

## Run

```bash
# Operator GOAT
cargo test -p katgpt-core --features recirculation --lib recirculation::
cargo test --release -p katgpt-core --features recirculation --lib g2_step_mixture -- --nocapture
cargo test --release -p katgpt-core --features recirculation --test recirculation_alloc_check --release
# PoC (real model; ~5-13 min per config on M3)
GEMMA2_2B_GGUF=... TOKENIZER_MODEL=... cargo test -p riir-poc --test recirculation_poc --release -- --nocapture --ignored
```

## Reopen attempt (2026-08-20, in flight / blocked — collection instructions)

**Status: the paper-scale re-run is RUNNING detached on the 4090 box; the
base-model check is BLOCKED on owner action.**

> **⚠ Freshness addendum 2026-09-28 (E12 prep, `.plans/611_e12_gemma_base_model_options.md`):**
> the paper-scale run NEVER COLLECTED — the `recirc_paperscale` scheduled task sits in state
> `Ready` and `paperscale_out.txt` does not exist; only the two arm-lines quoted in the progress
> note below were ever captured. "RUNNING" in this section is the 08-20 launch-time state, not
> the current one. The base-model check remains blocked as described (re-verified 2026-09-28:
> both boxes carry only IT-tuned Gemma GGUFs). The owner one-pager for the (a)-accept/(b)-retire
> pick lives at `.plans/611_e12_gemma_base_model_options.md`.

### Paper-scale re-run (in flight)

Launched via `schtasks /TN recirc_paperscale` (the Windows-native detached
run — Start-Process children DIE with the SSH session's job-object teardown,
the same class as the M3 nohup note; scheduled-task processes survive).

- **Output:** `E:\git\riir-ai\target\recirc\paperscale_out.txt` (12
  per-arm lines + the verdict table; poll for `[poc] ... ppl=` lines).
- **Config:** `RECIRC_POC_WINDOWS=50 RECIRC_POC_WINDOW_TOKENS=512`, both
  dataset dirs env-pinned (`RECIRC_POC_DS_A_DIR=E:\git\katgpt-rs\.research`,
  `RECIRC_POC_DS_B_DIR=E:\git\riir-ai\.docs` — the harness gained the
  env knobs after the hardcoded `/Users/katopz` paths silently produced
  1-token datasets on the box; riir-ai `7b931d0db`).
- **Model:** gemma-2-2b-it-f16 (the same register as Runs 1-2 — this run
  discharges the SCALE caveat, not the register caveat).
- **ETA (REVISED 2026-08-20 05:35, measured):** the original 4-11h
  estimate assumed 8-24 tok/s; the MEASURED baseline-arm rate is **4.9 tok/s**
  (arm 1: `A baseline ppl=112.2940, 5208.7s for 25600 tokens` — 512-token
  windows pay ~5× the per-token context cost of Runs 1-2's 96-160-token
  windows). 12 arms at this rate ≈ 17-24h+ (the 4 recirc/overwrite arms run
  2 stack instances/token and will be slower still). **Realistic completion:
  2026-08-21 early morning box time.** Health signature while running:
  ~4.2 cores sustained, output grows ~one line per completed arm — a static
  file for hours mid-arm is NORMAL (block-buffered stdout), do not diagnose
  a hang from a static file alone; check CPU advancement instead.
- **Progress (2026-08-20 06:56, second data point):** arm 2 landed —
  `A_research_arxiv recirc_a007: ppl=117.1081 (mean NLL 4.7631) [5208.0s,
  25600/arm]`. Two facts: (1) the per-arm rate is CONFIRMED STABLE
  (5208.0s vs 5208.7s — single-stack arms are ~87 min each, so arms 1-8
  finish ~20:30 box time TONIGHT; only the 4 dual-stack recirc/overwrite
  arms add uncertainty beyond that); (2) first paper-scale cell agrees
  with session scale — α=0.07 recirc is HARMFUL here too (+4.81 ppl vs
  the run's own baseline 112.2940; compare Run 3 cells against Run 3's
  baseline, NOT Run 1's 91.32 — the register shift moves absolute ppl).
- **Box state:** both repos synced to develop (riir-ai `7b931d0db`,
  katgpt-rs `e580ad35e`); toolchain `1.95.0-x86_64-pc-windows-msvc`;
  GPU idle/unused (CPU harness).
- **Collection:** when the verdict table lands, append it here as **Run 3
  (paper-scale)** + update the verdict row if the direction flips on any
  dataset (12/12 recirc cells were harmful at session scale; the reopen
  question is whether scale changes the sign or only the magnitudes).
  First data point already on record: `A baseline ppl=112.2940` (Run 1's
  2×96-token baseline was 91.32 — the 512-token register shifts absolute
  ppl; compare Run 3 cells against Run 3's OWN baseline, not Run 1's).

### Base-model check (BLOCKED — owner action required)

No public **base** (non-IT) Gemma-2-family GGUF exists on HF (verified
2026-08-20 via the HF API: bartowski/lmstudio-community have only IT
quantizations; `google/gemma-2-2b-GGUF` EXISTS but is `gated=manual` —
requires Gemma license acceptance + an authenticated HF token). The two
runnable paths, both owner-dependent:

1. Accept the license on `google/gemma-2-2b-GGUF` + provide the box an
   HF token → `Invoke-WebRequest` the f32 GGUF (~10 GB; E: has 55 GB free)
   → re-run the PoC session-scale against it (the f16 loader handles f32
   GGML types via the same dequant path — verify `from_gguf` accepts the
   type id first).
2. Convert `google/gemma-2-2b` safetensors via llama.cpp's
   `convert_hf_to_gguf.py` (also gated; same token prerequisite).

Until one lands, the model-register caveat stands and the negative verdict
remains scoped to **IT-tuned** residual streams.

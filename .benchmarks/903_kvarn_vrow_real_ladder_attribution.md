# Bench 903 — Issue 907: the KVarN V-row bit-ladder attribution on REAL rows

**Status:** COMPLETE — the inversion reproduces at the cache level; the crossed-config sweep attributes it; the consumer rule lands in `kvarn/mod.rs`.

## Setup

- Fixture: `.raw/vrow/gemma2_vrows.bin` (218,103,891 B, BLAKE3 `5ae9e74c…`, gitignored) — 26 layers × 2048 rows × kv_dim 1024 post-W_V f32 rows from gemma-2-2b-it f16 on the chat_probe stream (riir-infer `vrow_capture`, 2 chunks × 1024, corpus BLAKE3 `44aabe8e…`). The same model/corpus/stream the Bench 011 anomaly was measured on.
- Bench: `crates/katgpt-kv/tests/bench_903_kvarn_vrow_real_ladder.rs` (`harness = false`, `required-features = ["kvarn", "quant_mode_override"]`), loud-skip without the fixture, BLAKE3-pinned.
- New substrate surface: `KVarNKVCache::set_quant_mode` (feature `quant_mode_override`) — the measurement-only machinery override; resizes the RTN scratch to the crossed mode's worst case (`with_config` sized it for the DERIVED group count — the naive override panicked 32768-vs-1024 on the b3-crossed arm before the resize).
- Box: 4090 workstation, CPU lane, AC; T2 (`kv_plus_ladder`) running concurrently (~2 cores, accuracy-class work both sides). No latency claims.

## Measured (26 layers × 2048 real rows, rel-MSE = MSE / mean row energy; cosine)

| arm | bits | machinery | rel-MSE | cosine |
|---|---|---|---|---|
| plain-b2 (with_config) | 2 | skip-varn + grouped-4 RTN | **24.09** | 0.9886 |
| plain-b3 (with_config) | 3 | var-norm | **132.53** | 0.9416 |
| plain-b4 (with_config) | 4 | var-norm | 28.83 | 0.9863 |
| cross b3 → b2 machinery | 3 | skip-varn + g4 | 1350.87 | 0.6058 |
| cross b4 → b2 machinery | 4 | skip-varn + g4 | 1344.45 | 0.6069 |
| cross b2 → var-norm | 2 | var-norm | 827.98 | 0.7923 |

## Findings

1. **The inversion reproduces at the primitive level** — no model in the loop: b3 is 5.5× worse than b2 and 4.6× worse than b4 (Bench 011 read b3 flips 5.5× b2's at the model level; same shape, same ordering).
2. **The machinery classes don't compose.** Every crossed arm is worse than every plain arm (~828–1351 vs 24–133): skip-varn+grouped-4 tuned for 2-bit collapses at 3/4-bit; var-norm at 2-bit can't resolve. The crossed arms are OUT-OF-DISTRIBUTION quantizers — crossing attributes by ELIMINATION, not by rescue.
3. **Attribution: the b3 var-norm arm itself is the defect site.** Within the SAME var-norm machinery, 3 bits cost 4.6× the error of 4 bits (132.5 vs 28.8), while b2's separate machinery achieves 24.1 — there is NO configuration in the family where 3-bit is competitive. The 3-bit var-norm scale-field handling (the log-scale rounding granularity at 8 code values ±0/±1/±2/±3 interacting with the dual-scale rescale) is where the fix would land, if anyone ever wants b3.
4. **b2's cosine (0.9886) beats b4's (0.9863)** — consistent with the model-level record that b2 is a legitimately strong arm, not a cheap fallback.

## The consumer rule (lands in `kvarn/mod.rs`)

**Never interpolate V-row quality across KVarN bit arms** — each arm is a distinct quantizer; the ladder is not monotone in bits on real rows (b2 < b4 < b3), and the machinery classes don't compose across widths.

## En-route fixes recorded

- `KVarNKVCache::set_quant_mode` (new, `quant_mode_override`): the scratch-resize is LOAD-BEARING — the naive field-only override panics (`range end index 32768 out of range for slice of length 1024`) because `with_config` sized `scratch_rtn_scales/zp` for the derived group count (1 at b ≥ 3).
- The bench's fixture resolution walks up from CWD joining per-component — pushing a slash-separated relative string then popping MISMATCHES Windows' component model (measured: the path oscillated forever, 100% CPU, zero output — the wedge this bench shipped with and the walker replaced).

## Reproduce

```bash
# fixture (riir-infer, gitignored):
cargo run --release -p riir-infer-core --features vk_calibration --bin vrow_capture -- \
  --tokens 2048   # writes .raw/vrow/gemma2_vrows.bin; copy beside katgpt-rs
# bench (katgpt-rs):
cargo test -p katgpt-kv --release --features kvarn,quant_mode_override \
  --test bench_903_kvarn_vrow_real_ladder -- --nocapture
```

# pyramid_612 — FA-layer Q/K real-tensor fixtures (Plan 612 T2.1)

Real pretrained attention tensors for the PISA pyramid selection GOAT gate
(T2.2+). Captured 2026-09-29 on the 4090 (RTX 4090, exclusive) by
`riir-infer`'s `qwen38_pyramid_capture` bin (riir-infer `836d28e`) off the
qwen38 dense cudarc lane — the whole-model Q4_K_M decode composition — over
a deterministic nested-prefix prompt.

- `manifest.json` is the provenance record (schema `pyramid_612_capture_v1`):
  model BLAKE3 (`qwen38-27b-dbirks-Q4_K_M.gguf`, 16.8 GB), prompt + token
  digests, config echo (24 q-heads / 4 kv-heads / head_dim 256 / 16 FA
  layers, interval 4), the f16-KV capture posture, per-file BLAKE3s, and the
  box-state + GPU-exclusivity probe results.
- `k_layer{L}_head{h}_L{N}.f32` — post-RoPE K rows `[rows, 256]` f32 LE for
  the two target FA layers (model layers **3** and **63**) × two committed
  kv heads (**0**, **3**). Rows are the COMMIT subset: uniform stride to the
  4096-row cap + the forced positions (`0, N/4, N/2, 3N/4, N-1`) + that
  length's Q-sampled positions — the exact row set is `positions_L{N}.u32`
  (u32 LE, sorted, one per row).
- `q_layer{L}_pos{p}.f32` — post-RoPE Q `[24, 256]` f32 LE (all query heads)
  for the two target layers at the 15 sampled positions
  ({0, N/4, N/2, 3N/4, N−1} of each length — the union across lengths).
- Q/K are UNSCALED (the kernels apply `1/sqrt(head_dim)` internally — a
  positive monotone constant, irrelevant to selection ranking). K was stored
  under the shipped f16-KV arm at capture time; Q is the f32 tap buffer.

**Full captures** (all 4 kv heads × both layers at every length, all 16 FA
layers' Q, prompt.txt, tokens.u32 — ~1.2 GB) live in gitignored storage, not
in this repo: `/Volumes/SDXC1TB/pyramid_612/` (M3) and
`E:/git/_sync/pycap/` (4090), digest-pinned by the same manifest. The T2.2
gate loads committed bins relative to `CARGO_MANIFEST_DIR` and the full set
via `PYRAMID_612_FULL_DIR` (point it at the `full/` directory; every file is
BLAKE3-verified against the manifest before use) — skip-loud when
absent, never a green zero.

**The PRIMARY model (Ternary-Bonsai-2-27B PQ2_0) is a documented blocker**:
its GGUF is qwen35-family (config parses: 64 layers, 16 FA interval-4, same
head geometry) but its PQ2_0 ternary weights refuse the dense lane's Q4_K
loader — `tensor 'token_embd.weight': expected Q4_K/Q6_K, got Q2_0`. A
Bonsai capture needs the ternary whole-model lane, which does not exist in
riir-infer-gpu (katgpt-rs `.issues/908` record, closed with this blocker).

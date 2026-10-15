# LatticeMemory: E2LSH-Addressed Delta-Rule Cell Lattice — the Candidate Fifth Retrieval Class

> **Status: OPT-IN, NOT PROMOTED at 10⁶** (2026-10-07, Plan 619 T1.7, Bench 919:
> the GOAT floor's fixed G5 bar — graded near-miss recovery at ε = 0.05σ — is
> unreachable at 10⁶ items with d_k = 32; the NEGATIVE and the governing law are
> the deliverable). Feature `lattice_memory`. Zero runtime cost unless a caller
> constructs `LatticeMemory`.

## What It Is

A 2D lattice of delta-rule cells addressed by **continuous E2LSH per-axis
coordinates** (Plan 619, the deterministic extraction of the Spotlight-class
addressing math — SDM arXiv:2607.07386 / MARCH 2608.12435 / Memory Layers
2412.09764 lineage; no gradient descent). A query addresses a primary cell per
axis `(a·x + b)/w` (integer part = cell, **fractional part = bump input**), the
read blends the 3×3 neighborhood weighted by a radius-1.5 tent partition of
unity, and each cell is a `d_k × d_v` **delta-rule associative memory**
`S ← (I − βkkᵀ)S + βkvᵀ` — closed-form rank-1 removal along the key, two MAC
passes, no GD. Cells allocate on first write into one preallocated slab; the
hot path is allocation-free (counting-allocator gated).

## The Fifth Complexity Class — Candidate, Envelope-Bounded

| Retriever | Cost | Graded near-miss | Envelope law | Feature |
|---|---|---|---|---|
| Raven RSM | O(1) routing | — | ~10³ experts | always compiled |
| Engram | O(1) hash | **no — hash cliff** | ~10⁵ slots before collisions dominate | `engram` |
| δ-Mem | O(r) associative | rank-r | rank-bounded | `delta_mem` |
| PKM | O(√N) factored | coarse (quantization plateau) | ~10⁶ slots | `product_key_memory` (default) |
| **Lattice (this)** | **O(1) — 9 cells × d_k·d_v MACs** | **yes — graded inside support** | **N_max ≈ 0.35·d_k·(3.76/R)²** | `lattice_memory` (opt-in) |

The lattice is the only class whose recall DECAYS GRADEDLY with query
perturbation (Bench 919: 0.983 → 0.912 → 0.399 → 0.157 across 0.1R → 2R of the
support) where Engram cliffs instantly (0.379 → 0.004) and PKM plateaus flat.
That property is the Claim A differentiator — and it is sold at a price:

## The Governing Law — Support × Capacity

```
support radius  R ≈ 3.76σ / L          (L = grid side; the bump's ±1.5-cell
                                        support in fitted-σ units)
capacity/cell   0.35·d_k               (first-order erosion bound: same-cell
                                        keys are ~N(0, 1/d_k) correlated)
⇒ N_max ≈ 0.35·d_k·(3.76/R)²          — 63,336 items at d_k=32, R=0.05σ
```

Near-miss radius and item count trade against each other THROUGH d_k: widen the
support (coarser grid) and the cells crowd; crowd past capacity and the
delta-rule cross-talk drowns the signal. The sizer
(`LatticeMemory::sized(...)`) enforces the law in both directions — the support
drives the grid, and `OverCapacity { required_d_k }` refuses operating points
past the capacity, naming the remedy.

## Measured (Bench 919 — the GOAT record)

| Gate | Result |
|---|---|
| G1 recall@1 vs brute | flat 0.979–1.0 across 10³→10⁶ (growing state ✓) |
| G2 read latency | 1.53× WIN vs PKM O(√N) at 10⁶ (3755 vs 5730 ns); Engram O(1) 280 ns stays the latency floor; the constant is cache-shaped (slab 11 MB→767 MB leaves L2/L3) |
| G3 overwrite/forgetting | PASS — newest ≥ Engram (last-write-wins parity), old eroded to 0.034 (the sin²θ removal law), **far-key retained at kernel scale** — the structural win Engram cannot have |
| G5 near-miss | graded inside support vs Engram's cliff ✓; the FIXED 0.05σ bar ✗ at 10⁶ (support 0.006σ) → **NOT PROMOTED** |
| G4 allocs | 0 (counting allocator, release posture) |
| G6 memory | bytes/item = 4·d_k·d_v/λ (805–11829 B/item measured); linear memory disclosed, not hidden |

Kernel A/B (T1.8): **tent wins** — cos² bought +6% near-miss AUC with −0.4pt
exact recall; `BumpKernel::Tent` is the default, `Cos2` stays selectable with
its truncation-leak shape pinned (far-tap mass ~0.078 at the primary switch
where the tent is exactly 0 — the hard ±1-cell support is a tent property).

## Freeze/Thaw (T1.9 — the Plan 199 T1.B consumer surface)

`snapshot()` / `restore()` / `commitment()`: the freeze half produces a
serde-serializable `LatticeSnapshot` (cells + slab + bump cursor) plus a
BLAKE3 commitment over the full state (geometry header + cells + slab,
little-endian — comparable across machines); the thaw is a whole-artifact
swap — never a blend — validated BEFORE any write (geometry mismatch,
corrupt cursor, or a cell claiming past the slab are all refused by name, so
a thawed lie can never reach the hot path's slab indexing). The bump cursor
is part of the state: post-thaw writes continue claiming from exactly where
the freeze stopped. Batch-path by design (it clones the slab); the G4
zero-alloc law covers the hot write/read path only.

## When To Reach For It

- The consumer's (N, R) fits `N_max ≈ 0.35·d_k·(3.76/R)²` — e.g. ≤ 63k items at
  0.05σ with d_k = 32 (the riir-refine traj/insight evidence-memory slot,
  Plan 199, fits comfortably) — and graded near-miss recovery is worth more
  than Engram's raw latency.
- Anything larger needs d_k > 32, i.e. a key-PROJECTION widening (the key stops
  being the embedding) — that is its own gate, not a knob here.

Design record: [`.plans/619_lattice_memory_e2lsh_delta_cells.md`](../../.plans/619_lattice_memory_e2lsh_delta_cells.md) ·
measurement: [`.benchmarks/919_lattice_family_goat.md`](../../.benchmarks/919_lattice_family_goat.md).
Name disambiguation: NOT `analytic_lattice` (Plan 330 transport operators).

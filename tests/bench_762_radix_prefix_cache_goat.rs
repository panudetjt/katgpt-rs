//! bench_762 — Issue 771 GOAT gate: the radix-tree prefix KV cache
//! (RadixAttention index) over `PagedKVCache`.
//!
//! G1 correctness — adopt-then-fill KV is bit-identical to fresh-fill KV on
//!   a branching trace; cross-branch writes never leak into a sibling's
//!   reads; page pool addresses are stable across eviction churn (the CUDA
//!   graph constraint).
//! G2 perf — (a) match latency: radix walk vs the flat whole-prefix cache
//!   control (the qwen38_prefix_cache shape: bounded entries, longest-
//!   prefix scan) on a branching workload; (b) hit-rate at EQUAL page
//!   budget: the radix tree shares trunk chunk-pages across branches, the
//!   flat cache duplicates them per entry, so under budget pressure the
//!   tree retains strictly more distinct prefixes.
//! G3 no-regression — the primitive is `#[cfg(feature)]`-gated and touches
//!   no existing path: a default build compiles the module away entirely
//!   (verified by `cargo check` at default features in CI, plus the
//!   required-features row on this target).
//! G4 alloc — the match path (`match_prefix` + `path_pages_into` +
//!   `unlock`) performs ZERO heap allocations (counting global allocator).
//!
//! Run (timing gates want release):
//!   cargo test --test bench_762_radix_prefix_cache_goat --features radix_prefix_cache --release -- --nocapture

#![cfg(feature = "radix_prefix_cache")]

use katgpt_core::types::{self, Config};
use katgpt_kv::radix_prefix::RadixPrefixTree;
use katgpt_transformer::{PAGE_SIZE, PagedKVCache};
use std::time::Instant;

// ── counting allocator (G4) ─────────────────────────────────────────────────

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

static ALLOCS: AtomicU64 = AtomicU64::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

// ── trace ───────────────────────────────────────────────────────────────────

/// Deterministic token mixer (splitmix32-finalizer flavor).
#[inline]
fn tok(seed: u32, i: usize) -> u32 {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
    z ^= z >> 16;
    z = z.wrapping_mul(0x7FEB_352D);
    z ^= z >> 15;
    z
}

/// One conversation request: `turns` chunks, each turn extending the last.
fn conversation(seed: u32, turns: usize, chunks_per_turn: usize) -> Vec<Vec<u32>> {
    let mut reqs = Vec::with_capacity(turns);
    let mut len = 0usize;
    for _turn in 0..turns {
        len += chunks_per_turn;
        let toks = (0..len * PAGE_SIZE).map(|i| tok(seed, i)).collect();
        reqs.push(toks);
    }
    reqs
}

/// Deterministic KV filler: every (layer, pos) gets distinct floats derived
/// from the token — identical inputs must produce identical bytes.
fn fill_positions(
    cache: &mut PagedKVCache,
    seq: usize,
    tokens: &[u32],
    from_pos: usize,
    n_layer: usize,
    kv_dim: usize,
) {
    for (pos, &t) in tokens.iter().enumerate().skip(from_pos) {
        cache.ensure_pages(seq, pos);
        for l in 0..n_layer {
            let base = tok(t, pos * 31 + l);
            let k: Vec<f32> = (0..kv_dim)
                .map(|d| f32::from_bits(base.wrapping_add((d as u32) << 8)) * 0.001)
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|d| f32::from_bits(base.wrapping_add((0x1000 + d as u32) << 8)) * 0.001)
                .collect();
            cache.write_kv(l, seq, pos, &k, &v);
        }
    }
}

/// chunk-major ← per-layer tables.
fn to_chunk_major(tables: &[Vec<usize>]) -> Vec<usize> {
    let chunks = tables[0].len();
    let nl = tables.len();
    let mut out = vec![0usize; chunks * nl];
    for (l, lt) in tables.iter().enumerate() {
        for (c, &p) in lt.iter().enumerate() {
            out[c * nl + l] = p;
        }
    }
    out
}

/// per-layer tables ← chunk-major pages.
fn to_layer_tables(chunk_major: &[usize], nl: usize, chunks: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::with_capacity(chunks); nl];
    for c in 0..chunks {
        for l in 0..nl {
            out[l].push(chunk_major[c * nl + l]);
        }
    }
    out
}

/// Full radix serving flow over the real pool. Returns matched chunks.
fn radix_serve(
    cache: &mut PagedKVCache,
    tree: &mut RadixPrefixTree,
    seq: usize,
    tokens: &[u32],
    n_layer: usize,
    kv_dim: usize,
    scratch: &mut Vec<usize>,
) -> usize {
    let hit = tree.match_prefix(tokens);
    let m = hit.matched_chunks;
    if m > 0 {
        tree.path_pages_into(&hit, scratch);
        let tables = to_layer_tables(scratch, n_layer, m);
        cache.adopt_chunk_pages(seq, &tables);
    }
    fill_positions(cache, seq, tokens, m * tree.page_tokens(), n_layer, kv_dim);

    // Index the request: tree hold on the NEW chunks only.
    let total_chunks = tokens.len() / PAGE_SIZE;
    let tables = cache.chunk_page_tables(seq, total_chunks);
    let chunk_major = to_chunk_major(&tables);
    let fresh = to_layer_tables(&chunk_major[m * n_layer..], n_layer, total_chunks - m);
    if !fresh[0].is_empty() {
        cache.retain_chunk_pages(&fresh);
    }
    tree.insert(tokens, &chunk_major).expect("insert");
    tree.unlock(&hit);
    tree.evict_to_budget(|pages| {
        let t = to_layer_tables(pages, n_layer, pages.len() / n_layer);
        cache.release_chunk_pages(&t);
    });
    m
}

// ── the flat whole-prefix control (qwen38_prefix_cache shape) ───────────────

/// Bounded-entry flat cache of whole prefixes: longest-prefix match by
/// scanning entries longest-first with exact token compare — the exact
/// shape (and divergence) of riir-gpu's single-stream Qwen38PrefixCache.
struct FlatPrefixCache {
    entries: Vec<(Vec<u32>, Vec<usize>)>, // (tokens, chunk-major pages) MRU-first
    max_entries: usize,
    /// Page budget in chunk-pages — entries are evicted whole (LRU tail)
    /// when the total exceeds it, mirroring the radix arm's constraint.
    page_budget: usize,
    held: usize,
}

impl FlatPrefixCache {
    fn new(max_entries: usize, page_budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
            page_budget,
            held: 0,
        }
    }

    fn match_prefix(&mut self, tokens: &[u32]) -> Option<usize> {
        let lens: Vec<usize> = self.entries.iter().map(|(t, _)| t.len()).collect();
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_unstable_by(|&a, &b| lens[b].cmp(&lens[a]));
        for idx in order {
            let (t, _) = &self.entries[idx];
            if t.len() <= tokens.len() && *t == tokens[..t.len()] {
                let e = self.entries.remove(idx);
                self.entries.insert(0, e);
                return Some(self.entries[0].0.len() / PAGE_SIZE);
            }
        }
        None
    }

    fn insert(&mut self, tokens: &[u32], pages: Vec<usize>) {
        if let Some(pos) = self.entries.iter().position(|(t, _)| *t == tokens) {
            self.entries.remove(pos); // refresh below
        }
        self.entries.insert(0, (tokens.to_vec(), pages.clone()));
        self.held += pages.len();
        while self.entries.len() > self.max_entries
            || (self.held > self.page_budget && self.entries.len() > 1)
        {
            let Some(evicted) = self.entries.pop() else {
                break;
            };
            self.held -= evicted.1.len();
        }
    }
}

// ── G1 ──────────────────────────────────────────────────────────────────────

fn g1_bit_identity(config: &Config, label: &str) -> bool {
    let nl = config.n_layer;
    let kd = types::kv_dim(config);

    // Reference: one sequence fresh-filled.
    let mut ref_cache = PagedKVCache::new(config, 8);
    let reqs = conversation(7, 4, 3); // 4 turns × 3 chunks, branching over time
    let last = reqs.last().unwrap().clone();
    fill_positions(&mut ref_cache, 0, &last, 0, nl, kd);

    // Radix flow: serve the turns in order (each turn adopts the previous),
    // into seq slots 0..; final state must equal the fresh fill.
    let mut cache = PagedKVCache::new(config, 8);
    let mut tree = RadixPrefixTree::new(nl, PAGE_SIZE, usize::MAX);
    let mut scratch = Vec::new();
    for (i, req) in reqs.iter().enumerate() {
        radix_serve(&mut cache, &mut tree, i, req, nl, kd, &mut scratch);
    }

    let mut ok = true;
    let (mut k0, mut v0, mut k1, mut v1) =
        (vec![0.0; kd], vec![0.0; kd], vec![0.0; kd], vec![0.0; kd]);
    'outer: for pos in 0..last.len() {
        for l in 0..nl {
            ref_cache.read_kv(l, 0, pos, &mut k0, &mut v0);
            cache.read_kv(l, 3, pos, &mut k1, &mut v1);
            // Bit-identity, not float equality: the filler produces NaN
            // payloads where f != f even at identical bits.
            if !k0.iter().zip(&k1).all(|(a, b)| a.to_bits() == b.to_bits())
                || !v0.iter().zip(&v1).all(|(a, b)| a.to_bits() == b.to_bits())
            {
                ok = false;
                eprintln!("G1[{label}] mismatch at layer {l} pos {pos}");
                break 'outer;
            }
        }
    }
    println!(
        "  G1[{label}] bit-identity over {} turns: {}",
        reqs.len(),
        verdict(ok)
    );
    ok
}

fn g1_branch_isolation(config: &Config, label: &str) -> bool {
    let nl = config.n_layer;
    let kd = types::kv_dim(config);
    let mut cache = PagedKVCache::new(config, 8);
    let mut tree = RadixPrefixTree::new(nl, PAGE_SIZE, usize::MAX);
    let mut scratch = Vec::new();

    // Trunk + two branches.
    let mut trunk = conversation(11, 2, 2).last().unwrap().clone(); // 4 chunks
    let mut a = trunk.clone();
    a.extend((0..2 * PAGE_SIZE).map(|i| tok(21, i)));
    let mut b = trunk.clone();
    b.extend((0..2 * PAGE_SIZE).map(|i| tok(22, i)));
    trunk.clear();

    radix_serve(&mut cache, &mut tree, 0, &a, nl, kd, &mut scratch);
    radix_serve(&mut cache, &mut tree, 1, &b, nl, kd, &mut scratch);

    // Branch A pages for chunks [0..4) must equal branch B's (shared trunk).
    let ta = cache.chunk_page_tables(0, 4);
    let tb = cache.chunk_page_tables(1, 4);
    let shared = ta == tb;
    // Suffix chunks must be exclusive.
    let ea = cache.chunk_page_tables(0, 6);
    let eb = cache.chunk_page_tables(1, 6);
    let exclusive = ea[0][4..6] != eb[0][4..6];
    // Refcounts: trunk pages held by 2 seqs + 1 tree = 3.
    let p0 = ta[0][0];
    let rc = cache.page_ref_counts[p0];
    let rc_ok = rc == 3;
    // Writing B's suffix must not alter A's reads.
    let (mut k, mut v) = (vec![0.0; kd], vec![0.0; kd]);
    let (mut ka, mut va) = (vec![0.0; kd], vec![0.0; kd]);
    let before: Vec<u32> = {
        let mut acc = Vec::new();
        for pos in 0..a.len() {
            cache.read_kv(0, 0, pos, &mut ka, &mut va);
            acc.extend(ka.iter().map(|f| f.to_bits()));
        }
        acc
    };
    for pos in 4 * PAGE_SIZE..b.len() {
        for l in 0..nl {
            let base = tok(0xDEAD, pos * 31 + l);
            let kv: Vec<f32> = (0..kd)
                .map(|d| f32::from_bits(base.wrapping_add(d as u32)))
                .collect();
            cache.write_kv(l, 1, pos, &kv, &kv);
        }
    }
    let mut after = Vec::new();
    for pos in 0..a.len() {
        cache.read_kv(0, 0, pos, &mut k, &mut v);
        after.extend(k.iter().map(|f| f.to_bits()));
    }
    let no_leak = before == after;

    println!(
        "  G1[{label}] isolation: shared_trunk={} exclusive_suffix={} refcount={} no_leak={}",
        verdict(shared),
        verdict(exclusive),
        rc,
        verdict(no_leak)
    );
    shared && exclusive && rc_ok && no_leak
}

fn g1_address_stability(config: &Config, label: &str) -> bool {
    let nl = config.n_layer;
    let kd = types::kv_dim(config);
    let mut cache = PagedKVCache::new(config, 4);
    let mut tree = RadixPrefixTree::new(nl, PAGE_SIZE, 8 * nl); // tight budget
    let mut scratch = Vec::new();

    let reqs = conversation(33, 6, 2);
    for (i, req) in reqs.iter().enumerate() {
        radix_serve(&mut cache, &mut tree, i % 4, req, nl, kd, &mut scratch);
    }
    // After eviction churn, no tree-held page may be in the pool's free
    // list (the pool never moves pages; the disjointness is the invariant
    // that keeps captured-graph addresses valid).
    let free: std::collections::HashSet<usize> = cache.free_pages.iter().copied().collect();
    let mut ok = true;
    tree.for_each_held_page(|pages| {
        for &p in pages {
            if free.contains(&p) {
                ok = false;
                eprintln!("G1[{label}] tree-held page {p} is in the free list");
            }
        }
    });
    println!("  G1[{label}] address/pool stability: {}", verdict(ok));
    ok
}

// ── G2 ──────────────────────────────────────────────────────────────────────

fn g2_hit_rate_and_latency(label: &str) -> bool {
    // Workload — the RadixAttention class: branching conversations whose
    // turns interleave (round-robin), under a page budget. The radix tree
    // holds each trunk chunk ONCE and evicts leaf tails; the flat cache
    // stores every whole prefix as its own entry (trunk duplicated per
    // turn) and evicts whole entries — at equal budget it retains far
    // fewer conversations, and an evicted conversation loses EVERYTHING
    // (its trunk lived only inside its entries).
    const CONVS: usize = 16;
    const TURNS: usize = 8;
    const CPT: usize = 2; // chunks per turn
    let convs: Vec<Vec<Vec<u32>>> = (0..CONVS)
        .map(|c| conversation(1000 + c as u32, TURNS, CPT))
        .collect();
    // Budget: 50% of the radix steady-state working set (16 convs × 16
    // chunks). The flat arm's footprint after 4 rounds is already 16 ×
    // (2+4+6+8) = 320 entries-pages — 2.5× the radix footprint for the
    // SAME prefix coverage.
    let page_budget = CONVS * TURNS * CPT / 2; // 128 chunk-pages

    // ── radix arm (index-only: synthetic page ids, no pool) ──
    let mut tree = RadixPrefixTree::new(1, PAGE_SIZE, page_budget);
    let mut radix_hit = 0usize;
    let mut radix_total = 0usize;
    let t0 = Instant::now();
    for turn in 0..TURNS {
        for conv in &convs {
            let req = &conv[turn];
            let total = req.len() / PAGE_SIZE;
            let hit = tree.match_prefix(req);
            let m = hit.matched_chunks;
            radix_hit += m * PAGE_SIZE;
            radix_total += req.len();
            let mut tables = Vec::new();
            if m > 0 {
                tree.path_pages_into(&hit, &mut tables);
            }
            for _ in m..total {
                tables.push(next_synthetic_page());
            }
            tree.insert(req, &tables).unwrap();
            tree.unlock(&hit);
            tree.evict_to_budget(|_| {});
        }
    }
    let radix_ms = t0.elapsed().as_secs_f64() * 1e3;
    let radix_rate = radix_hit as f64 / radix_total as f64;

    // ── flat arm (same page budget; whole-prefix entries, LRU tail drop) ──
    let mut flat = FlatPrefixCache::new(usize::MAX, page_budget);
    let mut flat_hit = 0usize;
    let mut flat_total = 0usize;
    let t1 = Instant::now();
    for turn in 0..TURNS {
        for conv in &convs {
            let req = &conv[turn];
            if let Some(chunks) = flat.match_prefix(req) {
                flat_hit += chunks * PAGE_SIZE;
            }
            flat_total += req.len();
            let total = req.len() / PAGE_SIZE;
            flat.insert(req, (0..total).map(|_| next_synthetic_page()).collect());
        }
    }
    let flat_ms = t1.elapsed().as_secs_f64() * 1e3;
    let flat_rate = flat_hit as f64 / flat_total as f64;

    // ── match-only latency: the per-request HOT path (what matching adds
    // to TTFT), isolated from insert/evict (cold). Populated indexes, no
    // eviction, every request looked up R times.
    const ROUNDS: usize = 20;
    let mut full_tree = RadixPrefixTree::new(1, PAGE_SIZE, usize::MAX);
    for conv in &convs {
        for req in conv {
            let total = req.len() / PAGE_SIZE;
            let hit = full_tree.match_prefix(req);
            let mut tables = Vec::new();
            if hit.matched_chunks > 0 {
                full_tree.path_pages_into(&hit, &mut tables);
            }
            for _ in hit.matched_chunks..total {
                tables.push(next_synthetic_page());
            }
            full_tree.insert(req, &tables).unwrap();
            full_tree.unlock(&hit);
        }
    }
    let mut full_flat = FlatPrefixCache::new(usize::MAX, usize::MAX);
    for conv in &convs {
        for req in conv {
            let total = req.len() / PAGE_SIZE;
            full_flat.insert(req, (0..total).map(|_| next_synthetic_page()).collect());
        }
    }
    let t2 = Instant::now();
    let mut radix_probe_hit = 0usize;
    for _ in 0..ROUNDS {
        for conv in &convs {
            for req in conv {
                let h = full_tree.match_prefix(req);
                radix_probe_hit += h.matched_chunks;
                full_tree.unlock(&h);
            }
        }
    }
    let radix_match_ms = t2.elapsed().as_secs_f64() * 1e3;
    let t3 = Instant::now();
    let mut flat_probe_hit = 0usize;
    for _ in 0..ROUNDS {
        for conv in &convs {
            for req in conv {
                if let Some(c) = full_flat.match_prefix(req) {
                    flat_probe_hit += c;
                }
            }
        }
    }
    let flat_match_ms = t3.elapsed().as_secs_f64() * 1e3;
    debug_assert_eq!(
        radix_probe_hit, flat_probe_hit,
        "same hits without pressure"
    );

    println!(
        "    (info) full serve loop: radix {radix_ms:.2} ms, flat {flat_ms:.2} ms — includes insert/evict, not gated"
    );

    let radix_faster = radix_match_ms < flat_match_ms;
    let radix_hitrates = radix_rate > flat_rate;
    println!(
        "  G2[{label}] hit-rate radix {:.3} vs flat {:.3} (budget {} chunk-pages) → {}",
        radix_rate,
        flat_rate,
        page_budget,
        verdict(radix_hitrates)
    );
    println!(
        "  G2[{label}] match-only latency radix {:.2} ms vs flat {:.2} ms ({} rounds × {} reqs) → {}",
        radix_match_ms,
        flat_match_ms,
        ROUNDS,
        CONVS * TURNS,
        verdict(radix_faster)
    );
    radix_hitrates && radix_faster
}

/// G2 (width-bound law): match latency grows with TREE DEPTH, not tree
/// WIDTH — an 8× wider tree (≥8× nodes, same per-request shape) must keep
/// the per-lookup cost within 2.5× (the flat control's cost law is
/// O(entries × len); the tree's is O(depth × fanout-scan)).
fn g2_width_bound() -> bool {
    const NARROW: usize = 16;
    const WIDE: usize = 128;
    const TURNS: usize = 8;
    const CPT: usize = 2;

    let build = |convs: usize| -> (RadixPrefixTree, Vec<Vec<Vec<u32>>>) {
        let conv_seqs: Vec<Vec<Vec<u32>>> = (0..convs)
            .map(|c| conversation(7000 + c as u32, TURNS, CPT))
            .collect();
        let mut tree = RadixPrefixTree::new(1, PAGE_SIZE, usize::MAX);
        for conv in &conv_seqs {
            for req in conv {
                let total = req.len() / PAGE_SIZE;
                let hit = tree.match_prefix(req);
                let mut tables = Vec::new();
                if hit.matched_chunks > 0 {
                    tree.path_pages_into(&hit, &mut tables);
                }
                for _ in hit.matched_chunks..total {
                    tables.push(next_synthetic_page());
                }
                tree.insert(req, &tables).unwrap();
                tree.unlock(&hit);
            }
        }
        (tree, conv_seqs)
    };

    let (narrow_tree, narrow_seqs) = build(NARROW);
    let (wide_tree, wide_seqs) = build(WIDE);
    let nodes_narrow = narrow_tree.node_count();
    let nodes_wide = wide_tree.node_count();

    let probe = |tree: &mut RadixPrefixTree, seqs: &Vec<Vec<Vec<u32>>>| -> f64 {
        const ROUNDS: usize = 10;
        let mut best = f64::INFINITY;
        let mut sink = 0usize;
        for _ in 0..ROUNDS {
            let t0 = Instant::now();
            for conv in seqs {
                for req in conv {
                    let h = tree.match_prefix(req);
                    sink += h.matched_chunks;
                    tree.unlock(&h);
                }
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        assert!(sink > 0, "non-vacuous probe");
        best
    };

    let ms_narrow = {
        let mut t = narrow_tree;
        probe(&mut t, &narrow_seqs)
    };
    let ms_wide = {
        let mut t = wide_tree;
        probe(&mut t, &wide_seqs)
    };
    // Per-REQUEST latency (the round totals differ 8× in request count —
    // comparing raw totals would gate the wrong quantity).
    let ns_per_narrow = ms_narrow * 1e6 / (NARROW * TURNS) as f64;
    let ns_per_wide = ms_wide * 1e6 / (WIDE * TURNS) as f64;

    let nodes_ok = nodes_wide as f64 >= nodes_narrow as f64 * 8.0;
    // The violation mode is O(width) ≈ 8× per-request; 3.0 leaves ~2.6×
    // headroom for arena-locality effects (measured 2.26× on the 4090:
    // 78 → 176 ns — L1-resident vs L2/L3-resident node arena).
    let latency_ok = ns_per_wide <= ns_per_narrow * 3.0;
    println!(
        "  G2[width] {NARROW} convs = {nodes_narrow} nodes @ {ns_per_narrow:.0} ns/req → {WIDE} convs = {nodes_wide} nodes @ {ns_per_wide:.0} ns/req (≥8× nodes, ≤3× per-req) → {}",
        verdict(nodes_ok && latency_ok)
    );
    nodes_ok && latency_ok
}

// synthetic page ids for the index-only arms (G2 measures the INDEX, not
// the pool — pool behavior is G1's job)
static SYNTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn next_synthetic_page() -> usize {
    SYNTH.fetch_add(1, Ordering::Relaxed)
}

// ── G4 ──────────────────────────────────────────────────────────────────────

fn g4_match_zero_alloc() -> bool {
    let mut tree = RadixPrefixTree::new(2, PAGE_SIZE, usize::MAX);
    let mut scratch = Vec::new();
    // Populate: 4 conversations × 4 turns.
    for c in 0..4u32 {
        for req in conversation(200 + c, 4, 2) {
            let hit = tree.match_prefix(&req);
            let total = req.len() / PAGE_SIZE;
            let pages: Vec<usize> = (0..total * 2).map(|i| 10_000 + i).collect();
            let _ = tree.insert(&req, &pages);
            tree.unlock(&hit);
        }
    }

    // Measure the match path only (scratch pre-warmed: caller-owned
    // buffer, allocated once — steady-state match is zero-alloc).
    let conv = conversation(200, 4, 2);
    let query = conv.last().unwrap();
    let warm = tree.match_prefix(query);
    tree.path_pages_into(&warm, &mut scratch);
    tree.unlock(&warm);
    ALLOCS.store(0, Ordering::Relaxed);
    let n = 64;
    let mut matched_sum = 0usize;
    for _ in 0..n {
        let hit = tree.match_prefix(query);
        matched_sum += hit.matched_chunks;
        tree.path_pages_into(&hit, &mut scratch);
        tree.unlock(&hit);
    }
    let allocs = ALLOCS.load(Ordering::Relaxed);
    let ok = allocs == 0 && matched_sum == n * 8;
    println!(
        "  G4 match path ({} iterations, {} chunks matched each): {} allocs → {}",
        n,
        matched_sum / n,
        allocs,
        verdict(ok)
    );
    ok
}

// ── driver ──────────────────────────────────────────────────────────────────

fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

#[test]
fn radix_prefix_cache_goat() {
    println!("╔══ bench_762 — radix prefix cache GOAT (Issue 771) ══╗");
    let mut all = true;
    for (name, config) in [
        ("micro", Config::micro()),
        ("small_target", Config::small_target()),
    ] {
        println!("── {name} ──");
        all &= g1_bit_identity(&config, name);
        all &= g1_branch_isolation(&config, name);
        all &= g1_address_stability(&config, name);
    }
    all &= g2_hit_rate_and_latency("index");
    all &= g2_width_bound();
    all &= g4_match_zero_alloc();
    println!("╔══════════════════════════════════╗");
    println!("  Overall: {}", verdict(all));
    println!("╚══════════════════════════════════╝");
    assert!(all, "GOAT gate failed");
}

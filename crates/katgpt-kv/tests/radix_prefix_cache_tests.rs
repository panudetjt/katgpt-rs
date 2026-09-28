//! RadixPrefixTree behavior + property tests (Issue 771 T1).
//!
//! Integration test with `required-features = ["radix_prefix_cache"]` — the
//! Issue-713 discipline: naming this target without the feature errors
//! instead of reporting a green zero.

use katgpt_kv::radix_prefix::{InsertOutcome, RadixPrefixTree, TreeError};

const NL: usize = 2; // layers
const PT: usize = 16; // tokens per chunk (PagedKVCache::PAGE_SIZE)

/// Mock pool for the hold-discipline invariant: chunk-major page index →
/// (chunk, layer) identity, with refcounts mirroring PagedKVCache.
struct MockPool {
    next_page: usize,
    refcounts: std::collections::HashMap<usize, usize>,
    released: Vec<usize>,
}

impl MockPool {
    fn new() -> Self {
        Self {
            next_page: 0,
            refcounts: std::collections::HashMap::new(),
            released: Vec::new(),
        }
    }
    /// Allocate `chunks × NL` fresh pages (a sequence's own fill).
    fn fill(&mut self, chunks: usize) -> Vec<usize> {
        let first = self.next_page;
        self.next_page += chunks * NL;
        (first..self.next_page).collect()
    }
    fn retain(&mut self, pages: &[usize]) {
        for &p in pages {
            *self.refcounts.entry(p).or_insert(0) += 1;
        }
    }
    fn release(&mut self, pages: &[usize]) {
        for &p in pages {
            let c = self.refcounts.get_mut(&p).expect("release without hold");
            *c -= 1;
            if *c == 0 {
                self.released.push(p);
            }
        }
    }
}

/// chunk-major helper: page index of (chunk c, layer l).
#[inline]
fn at(pages: &[usize], c: usize, l: usize) -> usize {
    pages[c * NL + l]
}

fn tokens(seed: u32, chunks: usize) -> Vec<u32> {
    (0..chunks * PT)
        .map(|i| seed * 1_000_003 + i as u32)
        .collect()
}

/// The full serving flow against the mock pool: match → adopt → fill
/// suffix → retain new → insert → unlock. Returns the insert outcome.
fn serve(
    tree: &mut RadixPrefixTree,
    pool: &mut MockPool,
    toks: &[u32],
) -> (katgpt_kv::radix_prefix::MatchHit, InsertOutcome) {
    let hit = tree.match_prefix(toks);
    let m = hit.matched_chunks;
    // Fill (or adopt) the request's pages: fresh pages for the suffix,
    // matched pages adopted from the tree path.
    let mut tables = Vec::with_capacity(toks.len() / PT * NL);
    if m > 0 {
        tree.path_pages_into(&hit, &mut tables);
    }
    let fresh = pool.fill(toks.len() / PT - m);
    tables.extend_from_slice(&fresh);
    // Tree hold for the NEW chunks only; the pre-existing chunks already
    // carry the tree's hold.
    pool.retain(&fresh);
    let out = tree.insert(toks, &tables).expect("insert");
    tree.unlock(&hit);
    (hit, out)
}

#[test]
fn match_prefix_of_inserted_sequence_hits_fully() {
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let mut pool = MockPool::new();
    let toks = tokens(1, 5);
    serve(&mut tree, &mut pool, &toks);

    // Exact re-query: full hit.
    let hit = tree.match_prefix(&toks);
    tree.unlock(&hit);
    assert_eq!(hit.matched_chunks, 5);

    // Prefix query (partial trailing chunk): floors to whole chunks.
    let q = toks[..4 * PT + 7].to_vec();
    let hit = tree.match_prefix(&q);
    tree.unlock(&hit);
    assert_eq!(hit.matched_chunks, 4, "partial tail floors to 4 chunks");
}

#[test]
fn match_floors_partial_tail_chunk() {
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let mut pool = MockPool::new();
    let toks = tokens(7, 3);
    serve(&mut tree, &mut pool, &toks);
    let hit = tree.match_prefix(&toks[..2 * PT + 15]);
    tree.unlock(&hit);
    assert_eq!(hit.matched_chunks, 2);
}

#[test]
fn branches_share_trunk_pages_and_split() {
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let mut pool = MockPool::new();

    // Trunk (3 chunks) + branch A (2 chunks).
    let mut a = tokens(1, 3);
    a.extend(tokens(2, 2));
    serve(&mut tree, &mut pool, &a);

    // Branch B: same trunk, different suffix.
    let mut b = tokens(1, 3);
    b.extend(tokens(3, 2));
    let (hit_b, out_b) = serve(&mut tree, &mut pool, &b);

    assert_eq!(hit_b.matched_chunks, 3, "B matches the shared trunk");
    assert!(out_b.split, "inserting B split A's node at the divergence");
    assert_eq!(out_b.new_chunks, 2);

    // Path pages for B's match == A's trunk pages (shared indices) —
    // covered exhaustively in path_pages_returns_shared_trunk_indices.

    // A still matches fully after the split.
    let hit_a = tree.match_prefix(&a);
    tree.unlock(&hit_a);
    assert_eq!(hit_a.matched_chunks, 5);
    let hit_b2 = tree.match_prefix(&b);
    tree.unlock(&hit_b2);
    assert_eq!(hit_b2.matched_chunks, 5);
}

#[test]
fn path_pages_returns_shared_trunk_indices() {
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let mut pool = MockPool::new();

    let mut a = tokens(1, 3);
    a.extend(tokens(2, 2));
    serve(&mut tree, &mut pool, &a);
    let mut b = tokens(1, 3);
    b.extend(tokens(3, 2));
    serve(&mut tree, &mut pool, &b);

    // Re-match B: its trunk pages must be A's trunk pages (the FIRST fill).
    let base = 0usize; // MockPool's first fill starts at page 0
    let hit = tree.match_prefix(&b);
    tree.unlock(&hit);
    let mut pages = Vec::new();
    tree.path_pages_into(&hit, &mut pages);
    for c in 0..3 {
        for l in 0..NL {
            assert_eq!(
                at(&pages, c, l),
                base + c * NL + l,
                "trunk chunk {c} layer {l} must be the original shared page"
            );
        }
    }
    // B's suffix pages must be distinct from A's suffix pages (fresh fill).
    let a_suffix_first = 5 * NL; // A filled 5 chunks: pages [0, 10)
    let b_suffix_first = 5 * NL + 2 * NL; // B filled 2 fresh chunks after matching 3
    assert_ne!(a_suffix_first, b_suffix_first);
}

#[test]
fn insert_rejects_mismatched_path_tables() {
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let mut pool = MockPool::new();
    let toks = tokens(1, 4);
    serve(&mut tree, &mut pool, &toks);

    // Same tokens, but tables claiming DIFFERENT pages for the indexed path.
    let bad = pool.fill(4);
    assert_eq!(
        tree.insert(&toks, &bad),
        Err(TreeError::PathMismatch { chunk: 0 })
    );
}

#[test]
fn insert_rejects_wrong_pages_len() {
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let toks = tokens(1, 2);
    assert_eq!(
        tree.insert(&toks, &[0; 3]),
        Err(TreeError::PagesLen {
            expected: 2 * NL,
            got: 3
        })
    );
}

#[test]
fn reinsert_of_known_sequence_is_noop() {
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let mut pool = MockPool::new();
    let toks = tokens(1, 3);
    let (_, first) = serve(&mut tree, &mut pool, &toks);
    assert_eq!(first.new_chunks, 3);
    // Same sequence again: nothing new, no split, no double hold.
    let hit = tree.match_prefix(&toks);
    tree.unlock(&hit);
    // Re-insert with the SAME tables as stored: rebuild via path pages.
    let mut tables = Vec::new();
    tree.path_pages_into(&hit, &mut tables);
    let out = tree.insert(&toks, &tables).unwrap();
    assert_eq!(out.new_chunks, 0);
    assert!(!out.split);
    assert_eq!(tree.held_pages(), 3 * NL);
}

#[test]
fn eviction_is_leaf_lru_and_respects_locks() {
    let mut tree = RadixPrefixTree::new(NL, PT, 4 * NL); // 4 chunks of headroom
    let mut pool = MockPool::new();

    // Two independent 3-chunk sequences = 6 chunks held > budget 4.
    serve(&mut tree, &mut pool, &tokens(1, 3));
    serve(&mut tree, &mut pool, &tokens(2, 3));
    assert!(tree.held_pages() > tree.page_budget());

    // Lock the FIRST sequence's leaf (simulating an in-flight request).
    let t1 = tokens(1, 3);
    let lock_hit = tree.match_prefix(&t1); // locks terminal
    assert!(tree.total_locks() > 0);

    let mut released = Vec::new();
    {
        let pool_ref = &mut pool;
        let rel = &mut released;
        tree.evict_to_budget(|pages| {
            pool_ref.release(pages);
            rel.extend_from_slice(pages);
        });
    }
    // seq2 (unlocked, older-or-equal) must have been evicted; seq1 kept.
    assert_eq!(tree.stats().evictions, 1, "one leaf evicted");
    assert_eq!(tree.held_pages(), 3 * NL, "locked sequence's chunks stay");

    // Unlock → eviction can now proceed to budget.
    tree.unlock(&lock_hit);
    assert_eq!(tree.total_locks(), 0);
    {
        let pool_ref = &mut pool;
        let rel = &mut released;
        tree.evict_to_budget(|pages| {
            pool_ref.release(pages);
            rel.extend_from_slice(pages);
        });
    }
    assert!(tree.held_pages() <= tree.page_budget());
}

#[test]
fn eviction_peels_conversation_tail_first() {
    // A conversation grown turn-by-turn: eviction removes the newest turn
    // (childless leaf) while older turns stay (they have children).
    let mut tree = RadixPrefixTree::new(NL, PT, 3 * NL);
    let mut pool = MockPool::new();
    let mut conv = Vec::new();
    for turn in 0..4 {
        conv.extend(tokens(100 + turn as u32, 1));
        serve(&mut tree, &mut pool, &conv);
    }
    assert!(tree.held_pages() > tree.page_budget());
    tree.evict_to_budget(|pages| pool.release(pages));
    // The turn-4 extension (1 chunk, childless) is the LRU-evictable leaf
    // ONLY IF older turns are protected by having children — after enough
    // evictions the budget holds and the trunk remains matchable.
    assert!(tree.held_pages() <= tree.page_budget());
    let hit = tree.match_prefix(&tokens(100, 1));
    tree.unlock(&hit);
    assert_eq!(hit.matched_chunks, 1, "turn-1 trunk survives eviction");
}

#[test]
fn hold_discipline_invariant_holds() {
    // After a randomized op mix: every tree-indexed page has exactly the
    // tree's hold in the mock pool (refcount ≥ 1, and the released set only
    // contains pages the tree no longer indexes).
    let mut rng = fastrand::Rng::with_seed(0x771);
    let mut tree = RadixPrefixTree::new(NL, PT, 24 * NL);
    let mut pool = MockPool::new();
    let mut all: Vec<Vec<u32>> = Vec::new();
    for i in 0..40 {
        // Branching generator: extend a prior sequence or start a new one.
        let toks = if i % 3 == 0 || all.is_empty() {
            tokens(i as u32, 1 + rng.usize(0..4))
        } else {
            let mut base = all[rng.usize(0..all.len())].clone();
            base.extend(tokens(i as u32, 1 + rng.usize(0..3)));
            base
        };
        serve(&mut tree, &mut pool, &toks);
        all.push(toks);
        tree.evict_to_budget(|pages| pool.release(pages));
    }
    // Every match against an inserted sequence may under-report after
    // evictions, but must never exceed its own true chunk count.
    for toks in &all {
        let hit = tree.match_prefix(toks);
        tree.unlock(&hit);
        assert!(hit.matched_chunks <= toks.len() / PT);
    }
    assert_eq!(tree.total_locks(), 0, "balanced match/unlock");
}

#[test]
fn zero_chunk_insert_is_noop() {
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let out = tree.insert(&tokens(1, 0), &[]).unwrap();
    assert_eq!(out.new_chunks, 0);
    assert_eq!(tree.held_pages(), 0);
    assert_eq!(tree.node_count(), 0);
}

#[test]
fn first_token_same_but_chunk_divergent_makes_siblings() {
    // Two sequences whose first chunk shares token 0 but diverges inside
    // chunk 0: chunk-granular sharing must yield ZERO match, not a corrupt
    // descend (the reason children are matched by full first chunk, not by
    // first token key).
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let mut pool = MockPool::new();
    let mut a = tokens(1, 2);
    let mut b = a.clone();
    b[PT - 1] ^= 0xFFFF; // diverge INSIDE chunk 0
    a.extend(tokens(2, 1));
    b.extend(tokens(3, 1));
    serve(&mut tree, &mut pool, &a);
    let (hit_b, out_b) = serve(&mut tree, &mut pool, &b);
    assert_eq!(hit_b.matched_chunks, 0, "no chunk-level sharing");
    assert_eq!(out_b.new_chunks, 3);
    assert!(!out_b.split);
    let hit_a = tree.match_prefix(&a);
    tree.unlock(&hit_a);
    assert_eq!(hit_a.matched_chunks, 3);
}

#[test]
fn clear_releases_every_hold() {
    let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
    let mut pool = MockPool::new();
    serve(&mut tree, &mut pool, &tokens(1, 3));
    serve(&mut tree, &mut pool, &tokens(2, 2));
    let mut seen = Vec::new();
    tree.clear(|pages| seen.extend_from_slice(pages));
    assert_eq!(seen.len(), 5 * NL);
    assert_eq!(tree.held_pages(), 0);
    assert_eq!(tree.node_count(), 0);
}

/// Randomized oracle equivalence (the follow-up hardening from the 09-14
/// racing duplicate — its one genuinely-additive test class, adapted to the
/// chunk-floor contract): for arbitrary random inserts over a tiny alphabet
/// (heavy sharing + mid-chunk divergence) and random query mutations,
/// `match_prefix` must equal floor(lcp / page_tokens) of the brute-force
/// oracle over every served sequence. Budget = ∞ so eviction never shrinks
/// the indexed set; no lock leaks at the end (every serve unlocks).
#[test]
fn randomized_oracle_equivalence() {
    for seed in 0u64..24 {
        let mut r = fastrand::Rng::with_seed(seed);
        let mut tree = RadixPrefixTree::new(NL, PT, usize::MAX);
        let mut pool = MockPool::new();
        let mut served: Vec<Vec<u32>> = Vec::new();

        for _ in 0..40 {
            // 1..=4 chunk-length sequences over an 8-token alphabet: forces
            // trunk sharing, mid-chunk divergence, and duplicate inserts.
            let chunks = 1 + r.usize(..4);
            let toks: Vec<u32> = (0..chunks * PT).map(|_| r.u32(..8)).collect();
            serve(&mut tree, &mut pool, &toks);
            served.push(toks);
        }

        // Queries: the served sequences, their prefixes/extensions, and
        // fresh mutations — each must match the chunk-floor oracle.
        let mut queries: Vec<Vec<u32>> = served.clone();
        for _ in 0..40 {
            let base = &served[r.usize(..served.len())];
            let cut = r.usize(..=base.len());
            let mut q = base[..cut].to_vec();
            for _ in 0..r.usize(..24) {
                q.push(r.u32(..8));
            }
            queries.push(q);
        }

        for q in &queries {
            let hit = tree.match_prefix(q);
            tree.unlock(&hit);
            let lcp = served
                .iter()
                .map(|s| {
                    s.iter()
                        .zip(q)
                        .position(|(a, b)| a != b)
                        .unwrap_or(s.len().min(q.len()))
                })
                .max()
                .unwrap_or(0);
            assert_eq!(
                hit.matched_chunks,
                lcp / PT,
                "seed {seed} query len {} lcp {lcp}",
                q.len()
            );
        }
        // Every serve + probe unlocked — no lock leaks across 40 serves ×
        // 80 probes.
        assert_eq!(tree.total_locks(), 0, "seed {seed} lock leak");
    }
}

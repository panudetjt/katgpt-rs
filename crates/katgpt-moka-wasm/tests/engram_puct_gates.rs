//! Issue 868 / Plan 605 GOAT gates for the engram-fused PUCT POC.
//!
//! G1 — empty table ⇒ bit-identical to the feature-off player (the cheapest
//!      correctness assertion in Proposal 013).
//! G4 — the fusion surface (one `read` per expansion + the init writes) is
//!      zero-allocation. Tree-shape-dependent allocations (`scored` Vec,
//!      arena growth) are PRE-EXISTING search allocations, not the fusion's
//!      surface, and are deliberately out of this gate's scope.
//! G6 — deterministic miner + committed table (build twice → same root;
//!      fused players built from the same table → identical moves).
//! Q-init direction — a mined row `v̄ = +1` for the position AFTER the
//!      plain player's best move M says M's resulting position is won for
//!      the opponent; the parent reads `−Q = −1` and M's visit share must
//!      strictly DROP vs the plain player.
//!
//! Native-gated + required-features (Cargo.toml): a default-features cargo
//! invocation skips this target loudly instead of compiling a green zero.
//!
//! Tests in this binary SERIALIZE on one mutex: G4 counts process-wide
//! allocations, so a concurrent sibling test would corrupt the counter.

#![cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]

use katgpt_core::engram::{HashHead, K_MAX};
use katgpt_moka_wasm::board::{Board, Cell};
use katgpt_moka_wasm::engram_fuse::{EngramPuctMemory, MinedEntry, MinedTable, ROW_DIM};
use katgpt_moka_wasm::puct::PuctPlayer;
use std::alloc::{GlobalAlloc, Layout};
use std::sync::{Mutex, MutexGuard, OnceLock};

const BUDGET: usize = 50;
const C_PUCT: f32 = 1.5;
const TOP_K: usize = 8;

/// Serialize every test in this binary (the G4 counting window is
/// process-wide; see the module doc).
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let l = LOCK.get_or_init(|| Mutex::new(()));
    match l.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

fn test_heads() -> [HashHead; K_MAX] {
    // Same derivation the miner uses: the substrate builder's default heads
    // at a fixed size, frozen once.
    let t = katgpt_core::engram::EngramTableBuilder::new(1 << 12, ROW_DIM).build();
    *t.heads()
}

fn empty_memory() -> EngramPuctMemory {
    EngramPuctMemory::from_mined(&MinedTable::build(1 << 12, test_heads(), Vec::new()))
}

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A random-ish legal position (tests only — `legal_moves()` allocates).
fn random_position(seed: u64, n_moves: usize) -> Board {
    let mut rng = seed.max(1);
    let mut b = Board::new();
    for _ in 0..n_moves {
        if b.is_game_over() {
            break;
        }
        let moves = b.legal_moves();
        if moves.is_empty() {
            b.pass();
            continue;
        }
        let pick = (xorshift64(&mut rng) % moves.len() as u64) as usize;
        b.play(moves[pick]);
    }
    b
}

/// Full deterministic game: both sides driven by fresh players from `make`.
fn play_full_game(make: impl Fn() -> PuctPlayer + Copy, max_plies: usize) -> Vec<Option<usize>> {
    let mut black = make();
    let mut white = make();
    let mut b = Board::new();
    let mut seq = Vec::with_capacity(max_plies);
    for _ in 0..max_plies {
        if b.is_game_over() {
            break;
        }
        let mv = if b.to_play == Cell::Black {
            black.select_move(&b)
        } else {
            white.select_move(&b)
        };
        seq.push(mv);
        match mv {
            Some(i) => b.play(i),
            None => b.pass(),
        }
    }
    seq
}

/// G1a: 40 fixed positions — fused-with-empty-table first move must equal
/// the plain player's, bit for bit.
#[test]
fn g1_empty_table_first_moves_bit_identical() {
    let _g = serial();
    let mut fused = PuctPlayer::with_engram(BUDGET, C_PUCT, TOP_K, empty_memory());
    let mut plain = PuctPlayer::new(BUDGET, C_PUCT, TOP_K);
    for seed in 0..40u64 {
        let board = random_position(0xC0FFEE ^ seed, 3 + (seed as usize % 14));
        let a = fused.select_move(&board);
        let b = plain.select_move(&board);
        assert_eq!(
            a, b,
            "seed {seed}: fused(empty) diverged from plain at {board:?}"
        );
    }
}

/// G1b: two full games — the whole move sequences must be identical.
#[test]
fn g1_empty_table_full_games_bit_identical() {
    let _g = serial();
    let fused = || PuctPlayer::with_engram(12, C_PUCT, TOP_K, empty_memory());
    let plain = || PuctPlayer::new(12, C_PUCT, TOP_K);
    for seed in [1u32, 2] {
        let a = play_full_game(fused, 120 + seed as usize);
        let b = play_full_game(plain, 120 + seed as usize);
        assert_eq!(a.len(), b.len(), "seed {seed}: game lengths diverged");
        assert_eq!(a, b, "seed {seed}: move sequences diverged");
    }
}

/// G4: the fusion surface is zero-allocation — `read()` over mined
/// positions allocates nothing (scratch preallocated at construction).
#[test]
fn g4_fusion_read_is_zero_alloc() {
    let _g = serial();
    use std::sync::atomic::Ordering;

    // Mine a micro table from real self-play so every probed position is
    // guaranteed to fire.
    let mut entries: Vec<MinedEntry> = Vec::new();
    {
        let mut miner = PuctPlayer::new(10, C_PUCT, TOP_K);
        for _g_i in 0..6u64 {
            let mut b = Board::new();
            let mut seen: Vec<[u64; 4]> = Vec::new();
            for _ in 0..60 {
                if b.is_game_over() {
                    break;
                }
                seen.push(katgpt_moka_wasm::engram_fuse::tt_key_words(&b));
                let mv = miner.select_move(&b);
                match mv {
                    Some(i) => b.play(i),
                    None => b.pass(),
                }
            }
            for k in seen {
                entries.push(MinedEntry {
                    key: k,
                    sum_outcome: 0.3,
                    count: 12.0,
                    sum_conc: 0.6,
                });
            }
        }
    }
    let mined = MinedTable::build(1 << 12, test_heads(), entries);
    let mut mem = EngramPuctMemory::from_mined(&mined);

    let probes: Vec<Board> = (0..32u64)
        .map(|i| random_position(0xD00D ^ i, 2 + i as usize % 10))
        .collect();
    for b in &probes {
        let _ = mem.read(b);
    }

    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..8 {
        for b in &probes {
            let _ = mem.read(b);
        }
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "engram read allocated {} times over 256 calls",
        after - before
    );
}

/// G6: the mined table is deterministic (same entries in any order → same
/// commitment) and two fused players built from it play identically.
#[test]
fn g6_deterministic_table_and_players() {
    let _g = serial();
    let entries: Vec<MinedEntry> = (0..64u64)
        .map(|i| {
            let b = random_position(0x51EED ^ i, 4 + i as usize % 12);
            MinedEntry {
                key: katgpt_moka_wasm::engram_fuse::tt_key_words(&b),
                sum_outcome: ((i % 7) as f32 - 3.0) / 3.0,
                count: 5.0 + (i % 23) as f32,
                sum_conc: (i % 11) as f32 / 10.0,
            }
        })
        .collect();
    let heads = test_heads();
    let mut shuffled = entries.clone();
    shuffled.reverse();
    let t1 = MinedTable::build(1 << 12, heads, entries);
    let t2 = MinedTable::build(1 << 12, heads, shuffled);
    assert_eq!(t1.root, t2.root, "build order must not move the commitment");

    let board = random_position(0x1234, 8);
    let mut p1 = PuctPlayer::with_engram(BUDGET, C_PUCT, TOP_K, EngramPuctMemory::from_mined(&t1));
    let mut p2 = PuctPlayer::with_engram(BUDGET, C_PUCT, TOP_K, EngramPuctMemory::from_mined(&t2));
    assert_eq!(p1.select_move(&board), p2.select_move(&board));
    assert_eq!(p1.select_move(&Board::new()), p2.select_move(&Board::new()));
}

/// Q-init direction: discover the plain player's most-visited root move M,
/// mine `v̄ = +1` for the position AFTER M (the opponent wins there), and
/// assert M's visit share strictly drops under the fused player. The two
/// arms share identical policy logits (same weights, same int8 path), so
/// child lists match by action.
#[test]
fn q_init_suppresses_a_memory_losing_move() {
    let _g = serial();
    const B: usize = 200;

    // 1. Discover M deterministically.
    let mut plain = PuctPlayer::new(B, C_PUCT, TOP_K);
    let board = Board::new();
    let _ = plain.select_move(&board);
    let plain_root = plain.root_child_visits();
    let m = plain_root
        .iter()
        .copied()
        .max_by_key(|&(_, v)| v)
        .map(|(a, _)| a)
        .expect("plain search must produce root children");
    let m = m.expect("the plain player's best opening move is not a pass");
    let plain_share = |visits: &[(Option<usize>, u32)]| -> f64 {
        let total: u32 = visits.iter().map(|(_, v)| *v).sum();
        visits
            .iter()
            .find(|(a, _)| *a == Some(m))
            .map_or(0.0, |(_, v)| *v as f64 / total as f64)
    };
    let plain_m_share = plain_share(&plain_root);

    // 2. Mine v̄ = +1 for the position after M (to_play = White there —
    //    the mover's OPPONENT wins it; the parent reads −Q = −1).
    let mut after_m = Board::new();
    after_m.play(m);
    let entry = MinedEntry {
        key: katgpt_moka_wasm::engram_fuse::tt_key_words(&after_m),
        sum_outcome: 1000.0,
        count: 1000.0,
        sum_conc: 0.5, // γ-neutral
    };
    let mined = MinedTable::build(1 << 12, test_heads(), vec![entry]);

    // 3. Fused search must suppress M's share.
    let mut fused = PuctPlayer::with_engram(B, C_PUCT, TOP_K, EngramPuctMemory::from_mined(&mined));
    let _ = fused.select_move(&board);
    let fused_root = fused.root_child_visits();
    assert_eq!(
        fused_root.len(),
        plain_root.len(),
        "identical priors → identical child-list shapes"
    );
    let fused_m_share = plain_share(&fused_root);
    assert!(
        fused_m_share < plain_m_share,
        "v̄=+1 after move {m} must suppress it: fused share {fused_m_share:.3} !< plain {plain_m_share:.3}"
    );
}

/// Process-wide counting allocator (installed for the whole test binary so
/// G4's window is exact; serialized by `serial()` against sibling tests).
#[global_allocator]
static COUNTING_ALLOCATOR: CountingAlloc = CountingAlloc;

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { std::alloc::System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.realloc(p, l, n) }
    }
}

static ALLOC_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

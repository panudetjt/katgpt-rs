//! Issue 868 / Plan 605 T1 — the offline engram miner (Proposal 013 Phase 1).
//!
//! Plays N full-PUCT self-play games (both sides `PuctPlayer`, b50/c1.5/k8),
//! accumulates `(Σ outcome, count, Σ concentration)` per exact TT key
//! `(board, ko_point, to_play)`, and freezes the result into a BLAKE3-
//! committed `MinedTable` file the arena consumes. Modelless statistics
//! only — no training, no backprop (the modelless-first mandate).
//!
//! Also prints the T1.3 table-sanity gates WITHOUT which a G5 FAIL is
//! uninterpretable (Issue 868: "memory doesn't help" must be
//! distinguishable from "memory never fired"):
//! - count distribution (deciles of the per-position visit count),
//! - hit-rate curve vs table size {2¹⁶, 2¹⁸, 2²⁰, canonical} over the mined
//!   positions,
//! - collision audit at the canonical size (per-head multi-owner slots +
//!   the fraction of entries sharing ≥ 1 slot with another entry).
//!
//! Native-gated (`engram_puct` + not wasm32, via required-features + the
//! module gates). Determinism (G6): the game loop is seeded, the entry
//! build is key-sorted, the table build order is therefore deterministic.
//!
//! ```sh
//! cargo run --release -p katgpt-moka-wasm --features engram_puct \
//!     --example engram_miner -- --games 240
//! ```

use katgpt_core::engram::{HashHead, K_MAX};
use katgpt_moka_wasm::board::{Board, Cell};
use katgpt_moka_wasm::engram_fuse::{
    EngramPuctMemory, MinedEntry, MinedTable, ROW_DIM, keys_for, tt_key_words, visit_concentration,
};
use katgpt_moka_wasm::puct::PuctPlayer;

const BUDGET: usize = 50;
const C_PUCT: f32 = 1.5;
const TOP_K: usize = 8;
const MAX_MOVES: usize = 200;
const OPENING_MOVES: usize = 4;
/// Mining seed base — deliberately disjoint from the arena's eval seeds
/// (T3.2a leakage control: mine on set A, evaluate on set B).
const MINING_SEED_BASE: u64 = 0x4D49_4E45_5345_5441; // "MINESETA"

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_opening(board: &mut Board, n: usize, seed: u64) {
    let mut rng = seed.max(1);
    for _ in 0..n {
        if board.is_game_over() {
            break;
        }
        let moves = board.legal_moves();
        if moves.is_empty() {
            continue;
        }
        let pick = (xorshift64(&mut rng) % moves.len() as u64) as usize;
        board.play(moves[pick]);
    }
}

fn parse_arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map_or(default, |v| v.parse().expect("numeric arg"))
}

fn parse_path_arg(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let games = parse_arg("--games", 240).max(1);
    let slots = parse_arg("--slots", 1 << 20).max(1 << 10);
    let out = parse_path_arg("--out", "/tmp/katgpt_engram_puct_table.kept");

    println!(
        "=== engram_miner: {games} self-play games (b{BUDGET}/c{C_PUCT}/k{TOP_K}), slots={slots} ==="
    );
    let t0 = std::time::Instant::now();

    // One player instance serves both colours (stateless between
    // select_move calls); weights load once for the whole run.
    let mut player = PuctPlayer::new(BUDGET, C_PUCT, TOP_K);

    // Per-exact-key accumulation (f64 sums — offline, precision over speed).
    let mut acc: std::collections::HashMap<[u64; 4], (f64, u64, f64)> =
        std::collections::HashMap::new();
    // One probe Board per distinct key (the T1.3 curve reads through a
    // Board; kept here so mining order is the only source of truth).
    let mut probe_boards: std::collections::HashMap<[u64; 4], Board> =
        std::collections::HashMap::new();
    let mut total_plies = 0usize;

    for g in 0..games {
        let seed = MINING_SEED_BASE.wrapping_mul((g as u64).wrapping_add(1));
        let mut board = Board::new();
        random_opening(&mut board, OPENING_MOVES, seed);
        // Positions encountered this game, with their to_play at the time.
        let mut seen: Vec<([u64; 4], Cell)> = Vec::new();
        for _ in 0..MAX_MOVES {
            if board.is_game_over() {
                break;
            }
            let key = tt_key_words(&board);
            seen.push((key, board.to_play));
            probe_boards.entry(key).or_insert(board);
            let mv = player.select_move(&board);
            let conc = {
                let counts: Vec<u32> = player.root_child_visits().iter().map(|(_, v)| *v).collect();
                visit_concentration(&counts)
            };
            total_plies += 1;
            match mv {
                Some(i) => board.play(i),
                None => board.pass(),
            }
            // The concentration of THIS visit accumulates on the position
            // just searched (same key — same board, same to_play).
            let e = acc.entry(key).or_insert((0.0, 0, 0.0));
            e.2 += conc as f64;
        }
        // Game over: credit every recorded position with the terminal
        // outcome from ITS to_play perspective, tanh-mapped like the value
        // head (`2*reward − 1`).
        for (key, tp) in seen {
            let e = acc.entry(key).or_insert((0.0, 0, 0.0));
            e.0 += (2.0 * board.reward(tp) - 1.0) as f64;
            e.1 += 1;
        }
        if (g + 1) % 20 == 0 || g + 1 == games {
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "  game {}/{} — distinct keys so far: {} — {:.1} s ({:.2} s/game)",
                g + 1,
                games,
                acc.len(),
                dt,
                dt / (g + 1) as f64
            );
        }
    }

    // Entries (f32 narrowing at the file boundary — f64 only in the
    // accumulation, per the miner's precision-over-speed choice).
    let entries: Vec<MinedEntry> = acc
        .iter()
        .map(|(k, (s, c, b))| MinedEntry {
            key: *k,
            sum_outcome: *s as f32,
            count: *c as f32,
            sum_conc: if *c > 0 { (*b / *c as f64) as f32 } else { 0.0 },
        })
        .collect();
    let n_entries = entries.len();

    println!("\nmined: {games} games, {total_plies} plies, {n_entries} distinct positions");

    // Heads: the substrate builder's default heads at the canonical size,
    // frozen into the file (every later build reuses them).
    let heads: [HashHead; K_MAX] = {
        let t = katgpt_core::engram::EngramTableBuilder::new(slots, ROW_DIM).build();
        *t.heads()
    };

    // T1.3a — count distribution deciles.
    let mut counts: Vec<f32> = entries.iter().map(|e| e.count).collect();
    counts.sort_by(|a, b| a.total_cmp(b));
    let dec = |q: f64| -> f32 {
        let i = ((counts.len() as f64 - 1.0) * q).round() as usize;
        counts.get(i).copied().unwrap_or(0.0)
    };
    println!(
        "T1.3 count deciles (n): p10={} p25={} p50={} p75={} p90={} p99={} max={}",
        dec(0.10),
        dec(0.25),
        dec(0.50),
        dec(0.75),
        dec(0.90),
        dec(0.99),
        counts.last().copied().unwrap_or(0.0)
    );
    let rumors = counts.iter().filter(|&&c| c < 4.0).count();
    println!(
        "T1.3 rumor fraction (n<4): {:.1}%",
        100.0 * rumors as f64 / n_entries.max(1) as f64
    );

    // T1.3b — hit-rate curve vs table size over the mined positions.
    println!("T1.3 hit-rate curve (probe = mined positions):");
    for size in [1usize << 16, 1 << 18, 1 << 20, slots] {
        let mined = MinedTable::build(size, heads, entries.clone());
        let mut mem = EngramPuctMemory::from_mined(&mined);
        let mut fires = 0usize;
        let mut gate_half = 0usize;
        for e in &entries {
            let board = probe_boards
                .get(&e.key)
                .copied()
                .expect("probe board for every distinct key");
            if let Some(row) = mem.read(&board) {
                fires += 1;
                if row.gate >= 0.5 {
                    gate_half += 1;
                }
            }
        }
        println!(
            "  slots 2^{}: fires {fires}/{n_entries} ({:.1}%), gate≥0.5 {gate_half}/{n_entries} ({:.1}%)",
            size.trailing_zeros(),
            100.0 * fires as f64 / n_entries.max(1) as f64,
            100.0 * gate_half as f64 / n_entries.max(1) as f64
        );
    }

    // T1.3c — collision audit at the canonical size (per-head multi-owner
    // slots over distinct entries).
    {
        let mut total_multi = 0usize;
        let mut max_multi = 0usize;
        let mut entries_sharing = std::collections::HashSet::new();
        for head_k in 0..K_MAX {
            let mut slot_map: std::collections::HashMap<u64, Vec<usize>> =
                std::collections::HashMap::new();
            for (i, e) in entries.iter().enumerate() {
                let keys = keys_for(&e.key, &heads);
                let slot = keys[head_k].0 % slots as u64;
                slot_map.entry(slot).or_default().push(i);
            }
            let mut multi = 0usize;
            for owners in slot_map.values() {
                if owners.len() >= 2 {
                    multi += 1;
                    entries_sharing.extend(owners.iter().map(|&i| i as u64));
                }
            }
            total_multi += multi;
            max_multi = max_multi.max(multi);
        }
        println!(
            "T1.3 collision audit (slots={slots}): multi-owner slots per head max={max_multi}, total={total_multi}; entries sharing ≥1 slot: {}/{} ({:.2}%)",
            entries_sharing.len(),
            n_entries,
            100.0 * entries_sharing.len() as f64 / n_entries.max(1) as f64
        );
    }

    // Freeze + save (G6: key-sorted build, root committed in the file).
    let mined = MinedTable::build(slots, heads, entries);
    mined
        .save(std::path::Path::new(&out))
        .unwrap_or_else(|e| panic!("save {out}: {e}"));
    let root_hex: String = mined
        .root
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect();
    println!(
        "\nsaved {} entries → {out} (n_slots={}, root={root_hex}…, {:.1} s total)",
        mined.entries.len(),
        mined.n_slots,
        t0.elapsed().as_secs_f64()
    );
}

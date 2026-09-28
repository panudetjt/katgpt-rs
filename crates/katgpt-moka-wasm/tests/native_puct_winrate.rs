//! Native win-rate parity test: int8 PUCT vs f32 PUCT vs greedy Moka, all in
//! native Rust (no wasm boundary, no wasmi). Fast enough to run in CI (~24s
//! for n=20 at native speed, vs ~11min under wasmi).
//!
//! This fills the "native int8 winrate" gap noted in `moka_head_to_head.md`'s
//! int8 table — the wasmi test (`wasmi_puct_int8_winrate.rs`) measured 85%
//! through the wasm binary, but the native int8 path uses a different dot
//! kernel (SDOT inline asm vs wasm extmul), so the numbers aren't guaranteed
//! identical. This test measures directly.

use katgpt_moka_wasm::board::{AREA as BOARD_AREA, Board, Cell, SIZE as BOARD_SIZE};
use katgpt_moka_wasm::moka;
use katgpt_moka_wasm::puct::PuctPlayer;

const MAX_MOVES: usize = 200;
const OPENING_MOVES: usize = 4;
const NUM_GAMES: usize = 20;

/// xorshift64 — same RNG as the wasmi winrate tests (identical seeds →
/// identical opening sequences → the games are directly comparable across
/// native/wasmi).
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Play `n` random legal opening moves (mirrors the wasmi arena's
/// `random_opening`). Uses the same seed convention so opening sequences
/// match the wasmi tests bit-for-bit.
fn random_opening(
    board: &mut Board,
    history: &mut Vec<Option<(usize, usize)>>,
    n: usize,
    seed: u64,
) {
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
        let idx = moves[pick];
        board.play(idx);
        history.push(Some((idx / BOARD_SIZE, idx % BOARD_SIZE)));
    }
}

/// One greedy Moka move: forward pass + argmax over legal moves vs pass logit.
/// Mirrors `wasmi_arena_search_greedy` exactly. Returns true if the game is
/// over after this move.
fn greedy_move(
    board: &mut Board,
    history: &mut Vec<Option<(usize, usize)>>,
    weights: &moka::MokaWeights,
    scratch: &mut moka::MokaScratch,
    features_buf: &mut [f32],
) -> bool {
    features_buf.fill(0.0);
    moka::encode_features_into(board, history, features_buf);
    let (policy, _value) = moka::forward_with_scratch(weights, features_buf, scratch);

    let mut best_logit = policy[BOARD_AREA]; // pass logit
    let mut best_move: Option<usize> = None;
    for i in board.legal_moves() {
        if policy[i] > best_logit {
            best_logit = policy[i];
            best_move = Some(i);
        }
    }
    if let Some(idx) = best_move {
        board.play(idx);
        history.push(Some((idx / BOARD_SIZE, idx % BOARD_SIZE)));
    } else {
        board.pass();
        history.push(None);
    }
    board.is_game_over()
}

/// Run one full game. `puct_color` is Black (0) or White (1); greedy plays the
/// other color. The PUCT player manages its own internal state via
/// `select_move`. Returns true if `puct_color` won.
fn play_game_puct_vs_greedy<F>(puct_color: Cell, seed: u64, make_puct: F) -> bool
where
    F: FnOnce() -> PuctPlayer,
{
    let mut board = Board::new();
    let mut history: Vec<Option<(usize, usize)>> = Vec::new();
    random_opening(&mut board, &mut history, OPENING_MOVES, seed);

    let mut puct = make_puct();
    let weights = moka::MokaWeights::load();
    let mut scratch = moka::MokaScratch::new();
    let mut features_buf = vec![0.0f32; moka::INPUT_ELEMENT_COUNT];

    for _ in 0..MAX_MOVES {
        if board.is_game_over() {
            break;
        }
        if board.to_play == puct_color {
            // PUCT player both picks and plays (mirrors GoPuctMokaPlayer contract).
            let mv = puct.select_move(&board);
            // Replay the move on our local board + history so the greedy player
            // sees the same position the PUCT player left behind. The PUCT
            // player's `select_move` advances its OWN internal board copy; we
            // keep a separate board for the greedy forward pass.
            if let Some(idx) = mv {
                board.play(idx);
                history.push(Some((idx / BOARD_SIZE, idx % BOARD_SIZE)));
            } else {
                board.pass();
                history.push(None);
            }
        } else {
            let _ = greedy_move(
                &mut board,
                &mut history,
                &weights,
                &mut scratch,
                &mut features_buf,
            );
        }
    }

    board.reward(puct_color) > 0.5
}

#[test]
fn native_puct_winrate_f32_vs_greedy() {
    let start = std::time::Instant::now();
    let mut wins = 0usize;
    for game_i in 0..NUM_GAMES {
        let puct_color = if game_i % 2 == 0 {
            Cell::Black
        } else {
            Cell::White
        };
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((game_i as u64).wrapping_add(1));
        if play_game_puct_vs_greedy(puct_color, seed, || PuctPlayer::with_f32(50, 1.5, 8)) {
            wins += 1;
        }
    }
    let elapsed = start.elapsed();
    let rate = wins as f64 / NUM_GAMES as f64 * 100.0;
    println!(
        "\n=== Native f32 PUCT b50 vs greedy Moka (n={NUM_GAMES}) ===\nWin rate: {rate:.1}% ({wins}/{NUM_GAMES})\nWall clock: {:.1}s",
        elapsed.as_secs_f64()
    );
    // Native Bench 205 reference: 94% at n=100. At n=20 the floor is 75%.
    assert!(rate >= 75.0, "f32 native win rate {rate:.1}% < 75% floor");
}

#[test]
fn native_puct_winrate_int8_vs_greedy() {
    let start = std::time::Instant::now();
    let mut wins = 0usize;
    for game_i in 0..NUM_GAMES {
        let puct_color = if game_i % 2 == 0 {
            Cell::Black
        } else {
            Cell::White
        };
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((game_i as u64).wrapping_add(1));
        if play_game_puct_vs_greedy(puct_color, seed, || PuctPlayer::with_int8(50, 1.5, 8)) {
            wins += 1;
        }
    }
    let elapsed = start.elapsed();
    let rate = wins as f64 / NUM_GAMES as f64 * 100.0;
    println!(
        "\n=== Native int8 PUCT b50 vs greedy Moka (n={NUM_GAMES}) ===\nWin rate: {rate:.1}% ({wins}/{NUM_GAMES})\nWall clock: {:.1}s",
        elapsed.as_secs_f64()
    );
    // Same 75% parity floor as the f32 + wasmi tests.
    assert!(rate >= 75.0, "int8 native win rate {rate:.1}% < 75% floor");
}

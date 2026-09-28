//! Plan 563: Go 11 — Moka v1 (real weights) Head-to-Head.
//!
//! Runs our modelless Go players (Greedy/Validator/HL/GZero/MCTS) — plus
//! `GoMctsMokaPlayer`, MCTS using Moka's own value head as its leaf
//! evaluator (a neuro-symbolic composition, not a new trained model) —
//! against **Moka v1**: a real, MIT-licensed, 105,353-parameter 9×9 Go
//! policy/value network (github.com/millionco/moka), ported natively to
//! Rust in `katgpt_pruners::go::moka_net` (no Python, no Node, no
//! WASM/browser — see `.plans/563_go_moka_baseline_poc.md`). Same
//! match-loop shape as `go_02_tournament`, plus per-side move latency and a
//! final table comparing model size against Moka's own public numbers.
//!
//! ```sh
//! cargo run --features go --example go_11_moka_head_to_head
//!
//! # Custom game count / board / MCTS budgets
//! GO_GAMES=20 GO_MCTS_BUDGET=500 GO_MCTS_MOKA_BUDGET=50 cargo run --release --features go --example go_11_moka_head_to_head
//! ```

use std::env;
use std::time::{Duration, Instant};

use fastrand::Rng;
use katgpt_rs::pruners::go::{
    DEFAULT_KOMI, GoAction, GoCell, GoGZeroPlayer, GoGreedyPlayer, GoHLPlayer, GoMctsMokaPlayer,
    GoMctsPlayer, GoMokaSearchPlayer, GoOpeningBookSearchPlayer, GoPlayer, GoPuctMokaPlayer,
    GoReplay, GoState, GoValidatorPlayer, MokaPlayer,
};

const DEFAULT_NUM_GAMES: usize = 20;
const DEFAULT_BOARD_SIZE: usize = 9;
const MAX_MOVES: usize = 300;

// ── Player factory ─────────────────────────────────────────────

/// Search parameters for the players that take them. Bundled so adding a
/// knob doesn't ripple through every call site.
#[derive(Clone, Copy)]
struct SearchConfig {
    /// Plain `GoMctsPlayer` UCB1 budget.
    mcts_budget: usize,
    /// `GoMctsMokaPlayer` budget + rollout plies before the value head judges.
    mcts_moka_budget: usize,
    mcts_moka_depth: usize,
    /// `GoMokaSearchPlayer` negamax depth (plies) + per-node branching.
    search_depth: usize,
    search_top_k: usize,
    /// `GoOpeningBookSearchPlayer`: star-point plies before delegating to search.
    opening_book_moves: usize,
    /// `GoPuctMokaPlayer`: PUCT simulation budget + c_puct exploration constant.
    puct_budget: usize,
    puct_c_puct: f32,
}

fn make_player(name: &str, cfg: SearchConfig) -> Box<dyn GoPlayer> {
    match name {
        "greedy" => Box::new(GoGreedyPlayer),
        "validator" => Box::new(GoValidatorPlayer),
        "hl" => Box::new(GoHLPlayer::new()),
        "mcts" => Box::new(GoMctsPlayer::new(cfg.mcts_budget, 50)),
        "mcts-moka" => Box::new(GoMctsMokaPlayer::new(
            cfg.mcts_moka_budget,
            cfg.mcts_moka_depth,
        )),
        "moka-search" => Box::new(GoMokaSearchPlayer::new(cfg.search_depth, cfg.search_top_k)),
        "moka-openingbook" => Box::new(GoOpeningBookSearchPlayer::new(
            cfg.search_depth,
            cfg.search_top_k,
            cfg.opening_book_moves,
        )),
        "moka-puct" => Box::new(GoPuctMokaPlayer::new(
            cfg.puct_budget,
            cfg.puct_c_puct,
            cfg.search_top_k,
        )),
        "gzero" => Box::new(GoGZeroPlayer::new()),
        "moka" => Box::new(MokaPlayer::new()),
        _ => panic!("Unknown player: {name}"),
    }
}

/// Bandit-based players (HL, GZero) learn across games — feed the outcome back.
fn update_player_outcome(player: &mut dyn GoPlayer, won: bool) {
    if let Some(hl) = player.as_any_mut().downcast_mut::<GoHLPlayer>() {
        hl.update_outcome(won);
    }
    if let Some(gz) = player.as_any_mut().downcast_mut::<GoGZeroPlayer>() {
        gz.update_outcome(won);
    }
}

/// The Moka-family players track their own last-two-plies history internally
/// (`GoState` carries none) — when a ply is made by anyone else, they must be
/// told, or their last-move feature planes (7-10) go stale. Mirrors
/// `update_player_outcome`'s downcast idiom.
fn notify_history(player: &mut dyn GoPlayer, action: &GoAction) {
    let any = player.as_any_mut();
    if let Some(moka) = any.downcast_mut::<MokaPlayer>() {
        moka.observe_external_move(action);
        return;
    }
    if let Some(search) = any.downcast_mut::<GoMokaSearchPlayer>() {
        search.observe_external_move(action);
        return;
    }
    if let Some(ob) = any.downcast_mut::<GoOpeningBookSearchPlayer>() {
        ob.observe_external_move(action);
        return;
    }
    if let Some(puct) = any.downcast_mut::<GoPuctMokaPlayer>() {
        puct.observe_external_move(action);
    }
}

// ── Results ────────────────────────────────────────────────────

struct GameResult {
    first_player_won: bool,
    score: f32,
    moves: usize,
    duration: Duration,
    first_player_move_time: Duration,
    first_player_moves: usize,
    second_player_move_time: Duration,
    second_player_moves: usize,
}

struct MatchupResult {
    first_player: String,
    games: Vec<GameResult>,
}

impl MatchupResult {
    fn wins(&self) -> usize {
        self.games.iter().filter(|g| g.first_player_won).count()
    }
    fn losses(&self) -> usize {
        self.games.len() - self.wins()
    }
    fn win_rate(&self) -> f64 {
        self.wins() as f64 / self.games.len() as f64 * 100.0
    }
    fn avg_moves(&self) -> f64 {
        let total: usize = self.games.iter().map(|g| g.moves).sum();
        total as f64 / self.games.len() as f64
    }
    /// Average microseconds per move for the first (tested) player.
    fn avg_first_player_latency_us(&self) -> f64 {
        let total_time: Duration = self.games.iter().map(|g| g.first_player_move_time).sum();
        let total_moves: usize = self.games.iter().map(|g| g.first_player_moves).sum();
        if total_moves == 0 {
            return 0.0;
        }
        total_time.as_secs_f64() * 1_000_000.0 / total_moves as f64
    }
    /// Average microseconds per move for Moka (the second player in every
    /// matchup this example runs).
    fn avg_second_player_latency_us(&self) -> f64 {
        let total_time: Duration = self.games.iter().map(|g| g.second_player_move_time).sum();
        let total_moves: usize = self.games.iter().map(|g| g.second_player_moves).sum();
        if total_moves == 0 {
            return 0.0;
        }
        total_time.as_secs_f64() * 1_000_000.0 / total_moves as f64
    }
}

// ── Game loop ──────────────────────────────────────────────────

fn play_game(
    player_a: &mut dyn GoPlayer,
    player_b: &mut dyn GoPlayer,
    player_a_color: GoCell,
    board_size: usize,
    opening_moves: usize,
    rng: &mut Rng,
) -> GameResult {
    let start = Instant::now();
    let mut state = GoState::new(board_size);
    let mut replay = GoReplay::new(board_size, DEFAULT_KOMI);
    let mut moves = 0usize;
    let (mut a_time, mut a_moves) = (Duration::ZERO, 0usize);
    let (mut b_time, mut b_moves) = (Duration::ZERO, 0usize);

    // Per-game state reset. Both Moka-family players carry a move history that
    // MUST NOT leak across games (stale entries corrupt the last-move feature
    // planes on the next game's opening). `GoHLPlayer`/`GoGZeroPlayer::reset`
    // only clears per-episode trace state — bandit Q-values survive, so
    // cross-game learning is preserved.
    player_a.reset();
    player_b.reset();

    // Randomized opening (mirrors Moka's own arena, which randomizes openings
    // for the same reason): both players here are DETERMINISTIC, so without
    // this every game with the same color assignment replays identically and
    // "N games" collapses to 2 distinct samples. Random opening plies make the
    // games genuinely independent.
    for _ in 0..opening_moves {
        if state.is_terminal() {
            break;
        }
        let legal = state.legal_moves();
        if legal.is_empty() {
            break;
        }
        let (r, c) = legal[rng.usize(..legal.len())];
        let action = GoAction::Place(r, c);
        // Neither player chose this ply, so both must be told about it.
        notify_history(player_a, &action);
        notify_history(player_b, &action);
        state.play_move(r, c);
        replay.record(&action, state.to_play.opponent(), legal.len());
        moves += 1;
    }

    for _ in 0..MAX_MOVES {
        if state.is_terminal() {
            break;
        }

        let legal = state.legal_moves();
        let legal_count = state.legal_move_count();
        let a_turn = state.to_play == player_a_color;

        let move_start = Instant::now();
        let action = if a_turn {
            player_a.select_move(&state, &legal, rng)
        } else {
            player_b.select_move(&state, &legal, rng)
        };
        let elapsed = move_start.elapsed();
        if a_turn {
            a_time += elapsed;
            a_moves += 1;
            notify_history(player_b, &action);
        } else {
            b_time += elapsed;
            b_moves += 1;
            notify_history(player_a, &action);
        }

        match &action {
            GoAction::Place(row, col) => {
                let ok = state.play_move(*row, *col);
                debug_assert!(ok, "Player selected illegal move ({row},{col})");
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }

        replay.record(&action, state.to_play.opponent(), legal_count);
        moves += 1;
    }

    if !state.is_terminal() {
        state.play_pass();
        state.play_pass();
        moves += 2;
    }

    let score = state.score();
    let winner = state.get_winner();
    let first_player_won = winner == Some(player_a_color);
    replay.finalize(winner, score);

    GameResult {
        first_player_won,
        score,
        moves,
        duration: start.elapsed(),
        first_player_move_time: a_time,
        first_player_moves: a_moves,
        second_player_move_time: b_time,
        second_player_moves: b_moves,
    }
}

fn run_matchup(
    first_player_name: &str,
    second_player_name: &str,
    num_games: usize,
    board_size: usize,
    opening_moves: usize,
    cfg: SearchConfig,
    rng: &mut Rng,
) -> MatchupResult {
    let mut player_a = make_player(first_player_name, cfg);
    let mut player_b = make_player(second_player_name, cfg);
    let mut games = Vec::with_capacity(num_games);

    for i in 0..num_games {
        let player_a_color = if i % 2 == 0 {
            GoCell::Black
        } else {
            GoCell::White
        };
        let color_label = if player_a_color == GoCell::Black {
            "B"
        } else {
            "W"
        };

        print!(
            "  [{:>2}/{}] {}({}) vs Moka ",
            i + 1,
            num_games,
            player_a.name(),
            color_label
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let result = play_game(
            player_a.as_mut(),
            player_b.as_mut(),
            player_a_color,
            board_size,
            opening_moves,
            rng,
        );

        let outcome = if result.first_player_won { "W" } else { "L" };
        let score_display = if result.score > 0.0 {
            format!("B+{:.1}", result.score)
        } else {
            format!("W+{:.1}", result.score.abs())
        };
        println!(
            "{outcome} {:>8} {:>3} moves ({:.1}s)",
            score_display,
            result.moves,
            result.duration.as_secs_f64()
        );

        update_player_outcome(player_a.as_mut(), result.first_player_won);
        update_player_outcome(player_b.as_mut(), !result.first_player_won);

        games.push(result);
    }

    // No end-of-matchup reset needed — `play_game` resets per game.

    MatchupResult {
        first_player: first_player_name.to_string(),
        games,
    }
}

// ── Output ─────────────────────────────────────────────────────

fn print_header(num_games: usize, board_size: usize) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   Go 11 — Modelless Players vs Moka v1 (real weights)              ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Games per matchup: {num_games:<4}  Board: {board_size}×{board_size}                          ║"
    );
    println!(
        "║  Moka komi convention: 7.0 (not this repo's default {DEFAULT_KOMI})              ║"
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_final_table(results: &[MatchupResult]) {
    println!("══════════════════════════════════════════════════════════════════════════");
    println!("  FINAL RESULTS — vs Moka v1 (105,353 params, 113,648 B int8 weights)");
    println!("══════════════════════════════════════════════════════════════════════════");
    println!(
        "  {:<12} {:>10} {:>8} {:>9}   {:>14} {:>14}",
        "Player", "W/L", "Win%", "AvgMoves", "µs/move (ours)", "µs/move (moka)"
    );
    println!("  ──────────── ────────── ──────── ─────────   ────────────── ──────────────");
    for result in results {
        let wl = format!("{}W/{}L", result.wins(), result.losses());
        println!(
            "  {:<12} {:>10} {:>7.1}% {:>9.1}   {:>14.1} {:>14.1}",
            result.first_player,
            wl,
            result.win_rate(),
            result.avg_moves(),
            result.avg_first_player_latency_us(),
            result.avg_second_player_latency_us(),
        );
    }
    println!("══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Reference (external, not measured by this harness):");
    println!(
        "  Gemma 2 2B (riir-ai Plan 393/408/410): 0% gain over random baseline, ~50 s/move CPU"
    );
}

/// Time the raw forward pass, both the allocating convenience wrapper and the
/// scratch-reusing hot path. ~5.8M MACs per pass, so the MAC/s figure says
/// directly how much headroom the kernel still has.
fn bench_forward() {
    use katgpt_rs::pruners::go::{MokaScratch, forward, forward_with_scratch};

    const MACS_PER_FORWARD: f64 = 5_800_000.0;
    let weights = katgpt_rs::pruners::go::MokaWeights::load();

    // A mid-game position — realistic feature-plane density (stones, atari
    // groups, history planes all populated) rather than an empty board.
    let mut state = GoState::new(9);
    let mut history: Vec<Option<(usize, usize)>> = Vec::new();
    let mut rng = fastrand::Rng::with_seed(1);
    for _ in 0..30 {
        let legal = state.legal_moves();
        if legal.is_empty() {
            break;
        }
        let (r, c) = legal[rng.usize(..legal.len())];
        state.play_move(r, c);
        history.push(Some((r, c)));
    }
    let features = katgpt_rs::pruners::go::encode_features(&state, &history);

    let iters = 300;
    // Warm up so we measure steady state, not first-touch page faults.
    let mut scratch = MokaScratch::new();
    for _ in 0..20 {
        std::hint::black_box(forward_with_scratch(&weights, &features, &mut scratch));
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(forward_with_scratch(&weights, &features, &mut scratch));
    }
    let reuse = t0.elapsed().as_secs_f64() / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(forward(&weights, &features));
    }
    let alloc = t1.elapsed().as_secs_f64() / iters as f64;

    println!("Moka forward-pass latency ({iters} iters, mid-game position)");
    println!("  ─────────────────────────────────────────────────────────");
    println!(
        "  forward_with_scratch : {:>8.3} ms/pass  ({:.2} GMAC/s)",
        reuse * 1000.0,
        MACS_PER_FORWARD / reuse / 1e9
    );
    println!(
        "  forward (allocating) : {:>8.3} ms/pass  ({:.2} GMAC/s)",
        alloc * 1000.0,
        MACS_PER_FORWARD / alloc / 1e9
    );
    println!(
        "  scratch reuse saves  : {:>8.1}%",
        (1.0 - reuse / alloc) * 100.0
    );
    println!();
    println!("  Projected per-move cost (1 forward per visited node):");
    for (label, nodes) in [
        ("moka (greedy, 1 node)", 1.0),
        ("depth1 top8 (~9)", 9.0),
        ("depth2 top8 (~40 after pruning)", 40.0),
    ] {
        println!("    {label:<34} {:>8.1} ms", reuse * nodes * 1000.0);
    }
}

fn main() {
    let num_games: usize = env::var("GO_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_GAMES);
    let board_size: usize = env::var("GO_BOARD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BOARD_SIZE);
    let mcts_budget: usize = env::var("GO_MCTS_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    // GoMctsMokaPlayer calls a full Moka forward pass (~3ms) per rollout leaf
    // eval — keep its budget far smaller than plain MCTS or per-move cost
    // grows fast. Default chosen for a first read, not tuned.
    let mcts_moka_budget: usize = env::var("GO_MCTS_MOKA_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    // Rollout plies BEFORE Moka's value head judges the position. 0 = evaluate
    // immediately after each candidate move (in-distribution for a value net
    // trained on real self-play) instead of after N random plies (likely
    // out-of-distribution). See go_arena.md "GoMctsMokaPlayer" section.
    let mcts_moka_depth: usize = env::var("GO_MCTS_MOKA_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // GoMokaSearchPlayer: negamax plies + per-node branching. Cost is ~1
    // forward pass (~3ms) per visited node, bounded by top_k^depth before
    // alpha-beta pruning — 2/8 is ~75 passes ≈ 220ms/move worst case.
    let search_depth: usize = env::var("GO_SEARCH_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let search_top_k: usize = env::var("GO_SEARCH_TOPK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    // Opening-book wrapper: star-point plies before delegating to search.
    // Only used by the `moka-openingbook` player.
    let opening_book_moves: usize = env::var("GO_OPENING_BOOK_MOVES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    // PUCT player: simulation budget per move + exploration constant.
    let puct_budget: usize = env::var("GO_PUCT_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let puct_c_puct: f32 = env::var("GO_PUCT_C")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.5);
    // Random opening plies per game. Both Moka-family players are fully
    // deterministic, so with 0 every same-color game replays identically and
    // "N games" is really 2 samples. Non-zero makes games independent.
    let opening_moves: usize = env::var("GO_OPENING_MOVES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let cfg = SearchConfig {
        mcts_budget,
        mcts_moka_budget,
        mcts_moka_depth,
        search_depth,
        search_top_k,
        opening_book_moves,
        puct_budget,
        puct_c_puct,
    };

    if board_size != 9 {
        eprintln!("Moka v1 is a 9x9-only network — GO_BOARD must be 9 (got {board_size}).");
        std::process::exit(1);
    }

    // `GO_BENCH_FORWARD=1` times the raw kernel instead of playing games —
    // the forward pass is the whole latency budget (one per visited search
    // node), so it needs a direct number, not one inferred from win rates.
    if env::var("GO_BENCH_FORWARD").as_deref() == Ok("1") {
        bench_forward();
        return;
    }

    print_header(num_games, board_size);

    let mut rng = fastrand::Rng::with_seed(42);
    // `GO_MATCHUPS=moka-search` runs a single matchup — the search players
    // cost ~100ms+/move, so iterating on one of them shouldn't re-run the
    // whole cheap-player sweep every time.
    let default_matchups = "greedy,validator,hl,gzero,mcts,mcts-moka,moka-search";
    let matchup_spec = env::var("GO_MATCHUPS").unwrap_or_else(|_| default_matchups.to_string());
    let matchups: Vec<&str> = matchup_spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let mut all_results = Vec::with_capacity(matchups.len());

    for (idx, name) in matchups.iter().enumerate() {
        println!("Matchup {}/{}: {} vs moka", idx + 1, matchups.len(), name);
        let result = run_matchup(
            name,
            "moka",
            num_games,
            board_size,
            opening_moves,
            cfg,
            &mut rng,
        );
        println!(
            "  Result: {} {}W/{}L ({:.1}%)\n",
            result.first_player,
            result.wins(),
            result.losses(),
            result.win_rate()
        );
        all_results.push(result);
    }

    print_final_table(&all_results);
}
